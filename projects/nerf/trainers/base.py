'''
-----------------------------------------------------------------------------
Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
-----------------------------------------------------------------------------
'''
import os.path
import sys

import torch
import wandb
from imaginaire.trainers.base import BaseTrainer
from imaginaire.utils.distributed import is_master, master_only
from tqdm import tqdm

from projects.nerf.utils.misc import collate_test_data_batches, get_unique_test_data, trim_test_samples

from imaginaire.utils.visualization import preprocess_image
import torchvision
from torchvision.transforms import functional as torchvision_F
from projects.NeuralLumen.utils.utils import interpolate_pose, img_to_np, create_collage
import cv2 as cv
import numpy as np


class BaseTrainer(BaseTrainer):
    """
    A customized BaseTrainer.
    """

    def __init__(self, cfg, is_inference=True, seed=0):
        super().__init__(cfg, is_inference=is_inference, seed=seed)
        self.metrics = dict()
        # The below configs should be properly overridden.
        cfg.setdefault("wandb_scalar_iter", 9999999999999)
        cfg.setdefault("wandb_image_iter", 9999999999999)
        cfg.setdefault("validation_epoch", 9999999999999)
        cfg.setdefault("validation_iter", 9999999999999)

    def init_losses(self, cfg):
        super().init_losses(cfg)
        self.weights = {key: value for key, value in cfg.trainer.loss_weight.items() if value}

    def _end_of_iteration(self, data, current_epoch, current_iteration):
        # Log to wandb.
        if current_iteration % self.cfg.wandb_scalar_iter == 0:
            # Compute the elapsed time (as in the original base trainer).
            self.timer.time_iteration = self.elapsed_iteration_time / self.cfg.wandb_scalar_iter
            self.elapsed_iteration_time = 0
            # Log scalars.
            self.log_wandb_scalars(data, mode="train")
            # Exit if the training loss has gone to NaN/inf.
            if is_master() and self.losses["total"].isnan():
                self.finalize(self.cfg)
                raise ValueError("Training loss has gone to NaN!!!")
            if is_master() and self.losses["total"].isinf():
                self.finalize(self.cfg)
                raise ValueError("Training loss has gone to infinity!!!")
        if current_iteration % self.cfg.wandb_image_iter == 0:
            self.log_wandb_images(data, mode="train")
        # Run validation on val set.
        if current_iteration % self.cfg.validation_iter == 0:
            data_all = self.test(self.eval_data_loader, mode="val")
            # Log the results to W&B.
            if is_master():
                self.log_wandb_scalars(data_all, mode="val")
                self.log_wandb_images(data_all, mode="val", max_samples=self.cfg.data.val.max_viz_samples)

    def _end_of_epoch(self, data, current_epoch, current_iteration):
        # Run validation on val set.
        if current_epoch % self.cfg.validation_epoch == 0:
            data_all = self.test(self.eval_data_loader, mode="val")
            # Log the results to W&B.
            if is_master():
                self.log_wandb_scalars(data_all, mode="val")
                self.log_wandb_images(data_all, mode="val", max_samples=self.cfg.data.val.max_viz_samples)

    @master_only
    def log_wandb_scalars(self, data, mode=None):
        scalars = dict()
        # Log scalars (basic info & losses).
        if mode == "train":
            scalars.update({"optim/lr": self.sched.get_last_lr()[0]})
            scalars.update({"time/iteration": self.timer.time_iteration})
            scalars.update({"time/epoch": self.timer.time_epoch})
        scalars.update({f"{mode}/loss/{key}": value for key, value in self.losses.items()})
        scalars.update(iteration=self.current_iteration, epoch=self.current_epoch)
        wandb.log(scalars, step=self.current_iteration)

    @master_only
    def log_wandb_images(self, data, mode=None, max_samples=None):
        trim_test_samples(data, max_samples=max_samples)

    def model_forward(self, data):
        # Model forward.
        output = self.model(data)  # data = self.model(data) will not return the same data in the case of DDP.
        data.update(output)
        # Compute loss.
        self.timer._time_before_loss()
        self._compute_loss(data, mode="train")
        total_loss = self._get_total_loss()
        return total_loss

    def _compute_loss(self, data, mode=None):
        raise NotImplementedError

    def train(self, cfg, data_loader, single_gpu=False, profile=False, show_pbar=False):
        self.current_epoch = self.checkpointer.resume_epoch or self.current_epoch
        self.current_iteration = self.checkpointer.resume_iteration or self.current_iteration
        if ((self.current_epoch % self.cfg.validation_epoch == 0 or
             self.current_iteration % self.cfg.validation_iter == 0)):
            # Do an initial validation.
            data_all = self.test(self.eval_data_loader, mode="val", show_pbar=show_pbar)
            # Log the results to W&B.
            if is_master():
                self.log_wandb_scalars(data_all, mode="val")
                self.log_wandb_images(data_all, mode="val", max_samples=self.cfg.data.val.max_viz_samples)
            # very important, the following will do the training. This data_all will not release automatically!!!!!
            del data_all
            torch.cuda.empty_cache()
        # Train.
        super().train(cfg, data_loader, single_gpu, profile, show_pbar)

    @torch.no_grad()
    def test(self, data_loader, output_dir=None, inference_args=None, mode="test", show_pbar=False):
        """The evaluation/inference engine.
        Args:
            data_loader: The data loader.
            output_dir: Output directory to dump the test results.
            inference_args: (unused)
            mode: Evaluation mode {"val", "test"}. Can be other modes, but will only gather the data.
        Returns:
            data_all: A dictionary of all the data.
        """
        if self.cfg.trainer.ema_config.enabled:
            model = self.model.module.averaged_model
        else:
            model = self.model.module
        model.eval()
        if show_pbar:
            data_loader = tqdm(data_loader, desc="Evaluating", leave=False)
        data_batches = []
        if mode == "test":
            c_iteration = sys.maxsize
        else:
            c_iteration = self.current_iteration
        for it, data in enumerate(data_loader):
            data = self.start_of_iteration(data, current_iteration=c_iteration)
            output = model.inference(data)
            data.update(output)
            # data = {k: v.cpu() for k, v in data.items()}
            data_batches.append(data)
        # Aggregate the data from all devices and process the results.
        data_gather = collate_test_data_batches(data_batches)
        # Only the master process should process the results; slaves will just return.
        if is_master():
            data_all = get_unique_test_data(data_gather, data_gather["idx"])
            tqdm.write(f"Evaluating with {len(data_all['idx'])} samples.")
            # Validate/test.
            if mode == "val":
                self._compute_loss(data_all, mode=mode)
                _ = self._get_total_loss()
            if mode == "test":
                # Dump the test results for postprocessing.
                self.dump_test_results(data_all, output_dir)
            return data_all
        else:
            return

    @torch.no_grad()
    def test_save(self, data_loader, output_dir=None, inference_args=None, mode="test", show_pbar=False):
        """The evaluation/inference engine.
        Args:
            data_loader: The data loader.
            output_dir: Output directory to dump the test results.
            inference_args: (unused)
            mode: Evaluation mode {"val", "test"}. Can be other modes, but will only gather the data.
        Returns:
            data_all: A dictionary of all the data.
        """
        if self.cfg.trainer.ema_config.enabled:
            model = self.model.module.averaged_model
        else:
            model = self.model.module
        model.eval()
        if show_pbar:
            data_loader = tqdm(data_loader, desc="Evaluating", leave=False)
        data_batches = []
        if mode == "test":
            c_iteration = sys.maxsize
        else:
            c_iteration = self.current_iteration

        def save_image(image, name, from_range=(0, 1)):
            image = image.squeeze(0)
            image = preprocess_image(image, from_range=from_range)
            pil_image = torchvision_F.to_pil_image(image)
            pil_image.save(os.path.join(output_dir, name + '.png'))
        for it, data in enumerate(data_loader):
            data = self.start_of_iteration(data, current_iteration=c_iteration)
            output = model.inference(data)

            prefix = str(it) + '_'
            for key in output.keys():
                if 'map' in key:
                    save_image(output[key], prefix + key)
            if 'image' in data.keys():
                save_image(data['image'], prefix + 'rgb_target')
        return


    @torch.no_grad()
    def test_images(self, data_loader, output_dir=None, setting_list=None, mode="test", show_pbar=False):
        """The evaluation/inference engine.
        Args:
            data_loader: The data loader.
            output_dir: Output directory to dump the test results.
            inference_args: (unused)
            mode: Evaluation mode {"val", "test"}. Can be other modes, but will only gather the data.
        Returns:
            data_all: A dictionary of all the data.
        """
        if self.cfg.trainer.ema_config.enabled:
            model = self.model.module.averaged_model
        else:
            model = self.model.module
        model.eval()

        if mode == "test":
            c_iteration = sys.maxsize
        else:
            c_iteration = self.current_iteration

        dataset = data_loader.dataset
        dataset.sample_train_rays = False  # let the dataset return the full image

        os.makedirs(output_dir, exist_ok=True)
        def save_image(image, name, from_range=(0, 1)):
            image = image.squeeze(0)
            image = preprocess_image(image, from_range=from_range)
            pil_image = torchvision_F.to_pil_image(image)
            pil_image.save(os.path.join(output_dir, name + '.png'))
        for setting in setting_list:
            data = dataset[dataset.find_idx_cam_light(setting)]
            for key in data.keys():
                if torch.is_tensor(data[key]):
                    data[key] = data[key].unsqueeze(0)
            data = self.start_of_iteration(data, current_iteration=c_iteration)
            output = model.inference(data)

            prefix = setting + '_'
            save_image(data['image'], prefix + 'rgb_target')
            for key in output.keys():
                if 'map' in key:
                    save_image(output[key], prefix + key)

        return

    @torch.no_grad()
    def test_video(self, data_loader, setting1, setting2, output_dir=None, inference_args=None, mode="test", show_pbar=False,
                   video_content=('rgb', 'gt', 'o_r', 'o_s')):
        """The evaluation/inference engine.
        Args:
            data_loader: The data loader.
            output_dir: Output directory to dump the test results.
            inference_args: (unused)
            mode: Evaluation mode {"val", "test"}. Can be other modes, but will only gather the data.
        Returns:
            data_all: A dictionary of all the data.
        """
        if self.cfg.trainer.ema_config.enabled:
            model = self.model.module.averaged_model
        else:
            model = self.model.module
        model.eval()
        dataset = data_loader.dataset
        dataset.sample_train_rays = False   # let the dataset return the full image
        dataset.has_pseudo_label = False

        print("setting1: ", setting1)
        print("setting2: ", setting2)
        # sample1 = dataset[dataset.find_idx_cam_light(setting1)]
        # sample2 = dataset[dataset.find_idx_cam_light(setting2)]
        sample1 = dataset[int(setting1)]
        sample2 = dataset[int(setting2)]

        n_frames = 60
        images = []
        for i in range(n_frames):
            if show_pbar:
                print(i)
            ratio = torch.sin(torch.Tensor([((i / n_frames) - 0.5) * torch.pi])) * 0.5 + 0.5
            data = dict(idx=None)
            data['intr'] = sample1['intr'].unsqueeze(0)   # remember unsqueeze for batch size
            # the quaternion in camera.Pose().interpolate do not work well
            # data['pose'] = camera.Pose().interpolate(sample1['pose'], sample2['pose'], ratio)
            # data['pose_light'] = camera.Pose().interpolate(sample1['pose_light'], sample2['pose_light'], ratio)
            data['pose'] = interpolate_pose(sample1['pose'], sample2['pose'], ratio).unsqueeze(0)
            data['pose_light'] = interpolate_pose(sample1['pose_light'], sample2['pose_light'], ratio).unsqueeze(0)

            if mode == "test":
                c_iteration = sys.maxsize
            else:
                c_iteration = self.current_iteration
            data = self.start_of_iteration(data, current_iteration=c_iteration)
            output = model.inference(data)
            data.update(output)
            frame_imgs = []
            if 'rgb' in video_content:
                img_numpy = img_to_np(output['rgb_map'][0], add_text=True, text='Image (render)')
                frame_imgs.append(img_numpy)
            if 'gt' in video_content:
                closest_idx = dataset.find_closest_idx(data['pose'].cpu(), data['pose_light'].cpu())
                img_numpy = img_to_np(dataset[closest_idx]['image'], add_text=True, text='Image (the closest GT)')
                frame_imgs.append(img_numpy)
            if 'o_r' in video_content:
                img_numpy = img_to_np(output['o_r_map'][0], add_text=True, text='Reflectance')
                frame_imgs.append(img_numpy)
            if 'o_s' in video_content:
                img_numpy = img_to_np(output['o_s_map'][0], add_text=True, text='Shading')
                frame_imgs.append(img_numpy)
            if 'o_re' in video_content:
                img_numpy = img_to_np(output['o_re_map'][0], add_text=True, text='Residual')
                frame_imgs.append(img_numpy)
            frame = create_collage(frame_imgs)
            images.append(frame)
        for i in range(n_frames):
            images.append(images[n_frames - i - 1])

        fourcc = cv.VideoWriter_fourcc(*'mp4v')
        video_dir = os.path.join(output_dir, 'render')
        os.makedirs(video_dir, exist_ok=True)
        h, w, _ = images[0].shape
        writer = cv.VideoWriter(os.path.join(video_dir,
                                             '{}_{}.mp4'.format(setting1, setting2)),
                                fourcc, 30, (w, h))

        for image in images:
            writer.write(image)

        writer.release()

    def dump_test_results(self, data_all, output_dir):
        def dump_image(output_name, images, from_range=(0, 1)):
            image_grid = torchvision.utils.make_grid(images, nrow=1, pad_value=1)
            img = preprocess_image(image_grid, from_range=from_range)
            pil_image = torchvision_F.to_pil_image(img)
            pil_image.save(os.path.join(output_dir, output_name + '.jpg'))
        dump_image("rgb_target", data_all["image"])
        dump_image("rgb_render", data_all["rgb_map"])
        dump_image("rgb_error", (data_all["rgb_map"] - data_all["image"]).abs())
        dump_image("normal", data_all["normal_map"], from_range=(-1, 1))
        depth = data_all["depth_map"] - data_all["depth_map"].min()
        depth = depth / depth.max()
        dump_image("depth", depth)
        dump_image("inv_depth", 1 / (data_all["depth_map"] + 1e-8) * self.cfg.trainer.depth_vis_scale)
        dump_image("opacity", data_all["opacity_map"])
        if "normal_x_light_map" in data_all.keys():
            dump_image("normal_x_light_map", data_all["normal_x_light_map"].clamp(0.0, 1.0))
            dump_image("visibility_map", data_all["visibility_map"].float())
            normal_light_visibility = data_all["normal_x_light_map"].clamp(0.0, 1.0) * data_all["visibility_map"].float()
            dump_image("normal_light_visibility_map", normal_light_visibility)
            ref = data_all["rgb_map"] / normal_light_visibility.clip(min=1e-8, max=1.0)
            dump_image("ref_map", ref)
            ref_wo_vis = data_all["rgb_map"] / data_all["normal_x_light_map"].clip(min=1e-8, max=1.0)
            dump_image("ref_wo_vis_map", ref_wo_vis)
        if "light_opacity_map" in data_all.keys():
            dump_image("light_opacity", data_all["light_opacity_map"])
        if "indirect_rgb_map" in data_all.keys():
            dump_image("indirect_rgb", data_all["indirect_rgb_map"])
        for key in ['o_r_map', 'o_s_map', 'o_re_map', 'visibility_map', 'normal_x_light_map']:
            if key in data_all.keys():
                dump_image(key, data_all[key])

    @torch.no_grad()
    def test_light(self, data_loader, light_pose, output_dir=None, inference_args=None, mode="test", show_pbar=False):
        """The evaluation/inference engine.
        Args:
            data_loader: The data loader.
            output_dir: Output directory to dump the test results.
            inference_args: (unused)
            mode: Evaluation mode {"val", "test"}. Can be other modes, but will only gather the data.
        Returns:
            data_all: A dictionary of all the data.
        """
        if self.cfg.trainer.ema_config.enabled:
            model = self.model.module.averaged_model
        else:
            model = self.model.module
        model.eval()
        if show_pbar:
            data_loader = tqdm(data_loader, desc="Evaluating", leave=False)
        data_batches = []
        if mode == "test":
            c_iteration = sys.maxsize
        else:
            c_iteration = self.current_iteration
        inter_pts_range_list = []
        for it, data in enumerate(data_loader):
            data = self.start_of_iteration(data, current_iteration=c_iteration)
            output, inter_pts_range = model.inference_light(data, light_pose)
            inter_pts_range_list.append(inter_pts_range)
            data.update(output)
            data_batches.append(data)
        # # gather the range of intersection points.
        # xyz_min = torch.stack([x[0] for x in inter_pts_range_list], dim=0).min(dim=0)
        # xyz_max = torch.stack([x[1] for x in inter_pts_range_list], dim=0).max(dim=0)
        # norm_min = torch.stack([x[2] for x in inter_pts_range_list], dim=0).min(dim=0)
        # norm_max = torch.stack([x[3] for x in inter_pts_range_list], dim=0).max(dim=0)
        # Aggregate the data from all devices and process the results.
        data_gather = collate_test_data_batches(data_batches)
        # Only the master process should process the results; slaves will just return.
        if is_master():
            data_all = get_unique_test_data(data_gather, data_gather["idx"])
            tqdm.write(f"Evaluating with {len(data_all['idx'])} samples.")
            # Validate/test.
            if mode == "val":
                self._compute_loss(data_all, mode=mode)
                _ = self._get_total_loss()
            if mode == "test":
                # Dump the test results for postprocessing.
                self.dump_test_results(data_all, output_dir)
            return data_all
        else:
            return
