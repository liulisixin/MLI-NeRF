import copy
import os
import sys
from imaginaire.utils.distributed import master_only
from projects.neuralangelo.trainer import Trainer as AngeloTrainer
import wandb
from imaginaire.utils.visualization import wandb_image
from projects.neuralangelo.utils.misc import eikonal_loss, curvature_loss
import torch.nn.functional as torch_F
import torch
from projects.NeuralLumen.utils.utils import weighted_shading_loss, intrinsic_loss, regularize_re_loss, get_random_other_index
from imaginaire.utils.misc import requires_grad
from torch.cuda.amp import autocast
from torchvision.transforms import functional as torchvision_F
from imaginaire.utils.visualization import preprocess_image
from functools import partial
from tqdm import tqdm


class Trainer(AngeloTrainer):

    def __init__(self, cfg, is_inference=True, seed=0):
        super().__init__(cfg, is_inference=is_inference, seed=seed)
        if hasattr(self.model.module, "visibility_bounding_box_aabb"):
            self.model.module.visibility_bounding_box_aabb = self.model.module.visibility_bounding_box_aabb.to(
                self.model.module.device())
        if hasattr(cfg.model, "use_pre_trained"):
            pt_filename = cfg.model.use_pre_trained.pt_filename
            if pt_filename.endswith('.txt'):
                with open(pt_filename, 'r') as f:
                    pt_name = f.readline().strip()
                if pt_name:
                    pt_filename = os.path.join(os.path.dirname(pt_filename), pt_name)
                else:
                    raise FileNotFoundError
            state_dict = torch.load(pt_filename, map_location=lambda storage, loc: storage)
            print(f"Loading pt_filename (local): {pt_filename}")
            # Load the state dicts.
            print('- Loading the model...')
            result = self.model.load_state_dict(state_dict['model'], strict=False)
            print("Missing keys:", result.missing_keys)
            print("Unexpected keys:", result.unexpected_keys)

        if hasattr(cfg.trainer, "partial_grad"):
            self.partial_grad_keywords = cfg.trainer.partial_grad
            # First, disable gradient computations for all parameters
            requires_grad(self.model_module, False)
            # Then, only enable gradients for parameters that contain specific keywords
            for name, param in self.model_module.named_parameters():
                if any(keyword in name for keyword in self.partial_grad_keywords):
                    param.requires_grad = True
                    print(f"Enabling grad for: {name}")
                else:
                    print(f"Disabling grad for: {name}")

        if hasattr(cfg.trainer.loss_weight, "intrinsic"):
            para_intrinsic_loss = cfg.trainer.para_intrinsic_loss
            self.criteria_intrinsic = partial(intrinsic_loss,
                                              weight_map_range_shading=tuple(
                                                  para_intrinsic_loss['weight_map_range_shading']),
                                              weight_map_range_visibility=tuple(
                                                  para_intrinsic_loss['weight_map_range_visibility']),
                                              factor_ref=para_intrinsic_loss['factor_ref'],
                                              factor_sha=para_intrinsic_loss['factor_sha'])
        if hasattr(cfg.trainer.loss_weight, "regularize_re"):
            para_regularize_re_loss = cfg.trainer.para_regularize_re_loss
            self.criteria_regularize_re = partial(regularize_re_loss,
                                                  factor_negative=para_regularize_re_loss['factor_negative'],
                                                  factor_positive=para_regularize_re_loss['factor_positive'],
                                                  exponent_positive=para_regularize_re_loss['exponent_positive'])
            self.o_re_range = (-1.0, 1.0)


    @master_only
    def log_wandb_scalars(self, data, mode=None):
        scalars = dict()
        # Log scalars (basic info & losses).
        scalars.update(iteration=self.current_iteration, epoch=self.current_epoch)
        scalars = {
            f"{mode}/PSNR": self.metrics["psnr"].detach(),
            f"{mode}/s-var": self.model_module.s_var.item(),
        }
        if mode == "train":
            scalars.update({"optim/lr": self.sched.get_last_lr()[0]})
            scalars.update({"time/iteration": self.timer.time_iteration})
            scalars.update({"time/epoch": self.timer.time_epoch})
            # log loss
            scalars.update({f"{mode}/loss/{key}": value for key, value in self.losses.items()})
            # log weight
            for key in self.weights.keys():
                scalars[f"{mode}/weight/{key}"] = self.weights[key]
        else:
            # log loss but filter the losses that are not in val.
            for key, value in self.losses.items():
                if key not in ["intrinsic", "curvature"]:
                    scalars[f"{mode}/loss/{key}"] = value

        if mode == "train" and self.cfg_gradient.mode == "numerical":
            scalars[f"{mode}/epsilon"] = self.model.module.neural_sdf.normal_eps
        if self.cfg.model.object.sdf.encoding.coarse2fine.enabled:
            scalars[f"{mode}/active_levels"] = self.model.module.neural_sdf.active_levels

        wandb.log(scalars, step=self.current_iteration)

    @master_only
    def log_wandb_images(self, data, mode=None, max_samples=None):
        images = {"iteration": self.current_iteration, "epoch": self.current_epoch}
        if mode == "val":
            images_error = (data["rgb_map"] - data["image"]).abs()
            images.update({
                f"{mode}/vis/rgb_target": wandb_image(data["image"]),
                f"{mode}/vis/rgb_render": wandb_image(data["rgb_map"]),
                f"{mode}/vis/rgb_error": wandb_image(images_error),
                f"{mode}/vis/normal": wandb_image(data["normal_map"], from_range=(-1, 1)),
                f"{mode}/vis/inv_depth": wandb_image(1 / (data["depth_map"] + 1e-8) * self.cfg.trainer.depth_vis_scale),
                f"{mode}/vis/opacity": wandb_image(data["opacity_map"]),
            })
            for key in ['o_r_map', 'o_s_map', 'o_re_map', 'visibility_map', 'normal_x_light_map', 'pseudo_shading_map']:
                if key in data.keys():
                    images.update({
                        f"{mode}/vis/{key}": wandb_image(data[key])
                    })
            if 'o_re_map' in data.keys():
                if hasattr(self, 'o_re_range'):
                    o_re_range = self.o_re_range
                else:
                    o_re_range = (0, 1.0)
                images.update({
                    f"{mode}/vis/o_re_map": wandb_image(data['o_re_map'], from_range=o_re_range)
                })
        wandb.log(images, step=self.current_iteration)

    def _compute_loss(self, data, mode=None):
        if mode == "train":
            # Compute loss only on randomly sampled rays.
            self.losses["render"] = self.criteria["render"](data["rgb"], data["image_sampled"]) * 3  # FIXME:sumRGB?!
            self.metrics["psnr"] = -10 * torch_F.mse_loss(data["rgb"], data["image_sampled"]).log10()
            if "eikonal" in self.weights.keys():
                self.losses["eikonal"] = eikonal_loss(data["gradients"], outside=data["outside"])
            if "curvature" in self.weights:
                self.losses["curvature"] = curvature_loss(data["hessians"], outside=data["outside"])
            if "weighted_shading" in self.weights.keys():
                self.losses["weighted_shading"] = self.criteria_weighted_shading(data['o_s'], data['pseudo_shading'])
            if "intrinsic" in self.weights.keys():
                self.losses["intrinsic"] = self.criteria_intrinsic(data['o_r'], data['o_s'], data['pseudo_ref_sampled'],
                                                                   data['pseudo_sha_sampled'],
                                                                   data['pseudo_visibility_certainty_sampled'])
            if "regularize_re" in self.weights.keys():
                self.losses['regularize_re'] = self.criteria_regularize_re(data['o_re'])
        else:
            # Compute loss on the entire image.
            # yyx: it is better to calculate all the losses. Otherwise, there will be a small bug for wandb.
            # losses in train will be shown in the scaler of val.
            self.losses["render"] = self.criteria["render"](data["rgb_map"], data["image"])
            self.metrics["psnr"] = -10 * torch_F.mse_loss(data["rgb_map"], data["image"]).log10()
            if "eikonal" in self.weights.keys():
                self.losses["eikonal"] = eikonal_loss(data["gradients"], outside=data["outside"])
            # if "curvature" in self.weights:
            #     self.losses["curvature"] = 0.0
            if "weighted_shading" in self.weights.keys():
                self.losses["weighted_shading"] = self.criteria_weighted_shading(data['o_s_map'], data['pseudo_shading_map'])
            if "regularize_re" in self.weights.keys():
                self.losses['regularize_re'] = self.criteria_regularize_re(data['o_re_map'])

    def train_step(self, data, last_iter_in_epoch=False):
        r"""One training step.

        Args:
            data (dict): Data used for the current iteration.
        """
        # Set requires_grad flags.
        if hasattr(self, 'partial_grad_keywords'):
            # First, disable gradient computations for all parameters
            requires_grad(self.model_module, False)
            # Then, only enable gradients for parameters that contain specific keywords
            for name, param in self.model_module.named_parameters():
                if any(keyword in name for keyword in self.partial_grad_keywords):
                    param.requires_grad = True

        # Compute the loss.
        self.timer._time_before_forward()

        autocast_dtype = getattr(self.cfg.trainer.amp_config, 'dtype', 'float16')
        autocast_dtype = torch.bfloat16 if autocast_dtype == 'bfloat16' else torch.float16
        amp_kwargs = {
            'enabled': self.cfg.trainer.amp_config.enabled,
            'dtype': autocast_dtype
        }
        with autocast(**amp_kwargs):
            total_loss = self.model_forward(data)
            # Scale down the loss w.r.t. gradient accumulation iterations.
            total_loss = total_loss / float(self.cfg.trainer.grad_accum_iter)

        # Backpropagate the loss.
        self.timer._time_before_backward()
        self.scaler.scale(total_loss).backward()

        self._extra_step(data)

        # Perform an optimizer step. This enables gradient accumulation when grad_accum_iter is not 1.
        if (self.current_iteration + 1) % self.cfg.trainer.grad_accum_iter == 0 or last_iter_in_epoch:
            self.timer._time_before_step()
            self.scaler.step(self.optim)
            self.scaler.update()
            # Zero out the gradients.
            self.optim.zero_grad(**self.optim_zero_grad_kwargs)

        # Update model average.
        self.timer._time_before_model_avg()
        if self.cfg.trainer.ema_config.enabled:
            self.model.module.update_average()

        self._detach_losses()
        self.timer._time_before_leave_gen()

    @torch.no_grad()
    def test_all_light(self, data_loader, output_dir=None, mode="test", dataset_type='pair', sample_num=4, seed=999):
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
        dataset.sample_train_rays = False   # let the dataset return the full image

        if dataset_type == 'pair':
            frame_info = dataset.list
            index_info = {}
            for frame_index, frame in enumerate(frame_info):
                if frame['camera_index'] not in index_info.keys():
                    index_info[frame['camera_index']] = {}
                index_info[frame['camera_index']][frame['light_index']] = frame_index
        elif dataset_type == 'unpair':
            input_info = get_random_other_index(dataset.__len__(), sample_num, seed)
            index_info = {}
            for camera_index in range(len(input_info)):
                index_info[camera_index] = {}
                for light_index, frame_index in enumerate(input_info[camera_index]):
                    index_info[camera_index][light_index] = frame_index
        elif dataset_type == 'limitedlights':
            frame_info = dataset.list
            # find the corresponding frame_index for each pl_index
            pl_index_dict = {}
            for frame_index in range(sample_num):
                pl_index = frame_info[frame_index]['pl_index']
                pl_index_dict[pl_index] = frame_index
            index_info = {}
            for camera_index in range(len(frame_info)):
                index_info[camera_index] = {}
                index_info[camera_index][0] = camera_index
                pl_index_0 = frame_info[camera_index]['pl_index']
                pl_index_list = copy.deepcopy(list(pl_index_dict.keys()))
                pl_index_list.remove(pl_index_0)
                for index, pl_index in enumerate(pl_index_list):
                    light_index = index + 1
                    frame_index = pl_index_dict[pl_index]
                    index_info[camera_index][light_index] = frame_index
        else:
            raise NotImplementedError

        results_cam = {}
        for camera_index in index_info.keys():
            save_path = os.path.join(output_dir, str(camera_index))
            os.makedirs(save_path, exist_ok=True)
            def save_image(image, name, from_range=(0, 1)):
                image = image.squeeze(0)
                image = preprocess_image(image, from_range=from_range)
                pil_image = torchvision_F.to_pil_image(image)
                pil_image.save(os.path.join(save_path, name + '.png'))
            results_light = {}
            for light_index in tqdm(index_info[camera_index]):
                if dataset_type == 'pair':
                    sample_index = index_info[camera_index][light_index]
                    data = dataset[sample_index]
                else:
                    if light_index == 0:
                        data_input = dataset[camera_index]
                    else:
                        data_input['pose_light'] = dataset.get_light(light_index)
                    data = copy.deepcopy(data_input)
                for key in data.keys():
                    if torch.is_tensor(data[key]):
                        data[key] = data[key][None, ...]

                data = self.start_of_iteration(data, current_iteration=c_iteration)
                output = model.inference(data)

                prefix = str(light_index) + '_'
                if dataset_type == 'pair' or light_index == 0:
                    save_image(data['image'], prefix + 'rgb_target')
                save_image(output['rgb_map'], prefix + 'rgb_render')
                save_image(output['normal_map'], prefix + 'normal', from_range=(-1, 1))
                save_image(output['visibility_map'], prefix + 'visibility')
                save_image(output['inter_dist_map'], prefix + 'inter_dist', from_range=(output['inter_dist_map'].min(),
                                                                                        output['inter_dist_map'].max()))
                save_image(output['inter_mask_map'], prefix + 'inter_mask')
                save_image(output['normal_x_light_map'], prefix + 'normal_x_light')

                pseudo_shading = output['visibility_map'].float() * output['normal_x_light_map']
                save_image(pseudo_shading, prefix + 'pseudo_shading')

                results = {'normal': output['normal_map'].detach().cpu(),
                           'normal_x_light': output['normal_x_light_map'].detach().cpu(),
                           'rgb_render': output['rgb_map'].detach().cpu(),
                           'visibility': output['visibility_map'].detach().cpu(),
                           'inter_mask': output['inter_mask_map'].detach().cpu()}
                if dataset_type == 'pair':
                    results['rgb_target'] = data['image'].detach().cpu()
                results_light[str(light_index)] = results
            results_cam[str(camera_index)] = results_light
        torch.save(results_cam, os.path.join(output_dir, 'results_all.pt'))
