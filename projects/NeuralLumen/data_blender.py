import os.path

import torch
import numpy as np
from projects.NeuralLumen.data import Dataset as LumenDataset
from projects.nerf.utils import camera
from PIL import Image, ImageFile
import torchvision.transforms.functional as torchvision_F
ImageFile.LOAD_TRUNCATED_IMAGES = True


class Dataset(LumenDataset):

    def __init__(self, cfg, is_inference=False, is_test=False):
        cfg_data = cfg.data
        self.data_source = cfg_data.data_source if hasattr(cfg_data, 'data_source') else None

        super().__init__(cfg, is_inference=is_inference, is_test=is_test)

        self.white_background = cfg_data.white_background
        # intrinsic image decomposition (iid)
        self.load_iid = cfg_data[self.split].load_iid
        if self.load_iid:
            if cfg_data.preload:
                self.iids = self.preload_threading(self.get_iid, cfg_data.num_workers, data_str="iids")


    def get_light(self, idx):
        """
        If only pl_pos is provided, use unit matrix to replace R.
        """
        # light pose.
        c2w_gl = torch.eye(4, dtype=torch.float32)
        c2w_gl[:3, 3] = torch.tensor(self.list[idx]['pl_pos'], dtype=torch.float32)
        c2w = self._gl_to_cv(c2w_gl)

        # center scene
        center = np.array(self.meta["sphere_center"])
        center += np.array(getattr(self.readjust, "center", [0])) if self.readjust else 0.
        c2w[:3, -1] -= center
        # scale scene
        scale = np.array(self.meta["sphere_radius"])
        scale *= getattr(self.readjust, "scale", 1.) if self.readjust else 1.
        c2w[:3, -1] /= scale

        w2c = camera.Pose().invert(c2w[:3])
        return w2c

    def get_camera(self, idx):
        meta = self.meta
        # Camera intrinsics.
        if 'camera_intrinsics' in meta:
            intrinsics = meta['camera_intrinsics']
            cx = intrinsics[0]
            cy = intrinsics[1]
            fx = intrinsics[2]
            fy = intrinsics[3]
        else:  # fall back to camera_angle_x
            W = self.raw_W
            H = self.raw_H
            camera_angle_x = float(meta['camera_angle_x'])
            focal = float(.5 * W / np.tan(.5 * camera_angle_x))
            cx = W / 2.
            cy = H / 2.
            fx = focal
            fy = focal
        intr = torch.tensor([[fx, 0.0, cx],
                             [0.0, fy, cy],
                             [0, 0, 1]]).float()
        # Camera pose.
        c2w_gl = torch.tensor(self.list[idx]["transform_matrix"], dtype=torch.float32)
        c2w = self._gl_to_cv(c2w_gl)

        # center scene
        center = np.array(self.meta["sphere_center"])
        center += np.array(getattr(self.readjust, "center", [0])) if self.readjust else 0.
        c2w[:3, -1] -= center
        # scale scene
        scale = np.array(self.meta["sphere_radius"])
        scale *= getattr(self.readjust, "scale", 1.) if self.readjust else 1.
        c2w[:3, -1] /= scale

        w2c = camera.Pose().invert(c2w[:3])
        return intr, w2c

    def get_image(self, idx):
        if self.data_source == 'NRHints':
            fpath = self.list[idx]["file_path"] + self.list[idx]["file_ext"]
        else:
            fpath = self.list[idx]["file_path"] + 'Img.png'
        image_fname = os.path.normpath(os.path.join(self.root, fpath))
        image = Image.open(image_fname)
        image.load()
        image_size_raw = image.size
        return image, image_size_raw

    def get_iid(self, idx):
        iid = {}
        for key in ['Ref', 'Sha', 'Res']:
            fpath = self.list[idx]["file_path"] + key + '.png'
            image_fname = os.path.normpath(os.path.join(self.root, fpath))
            image = Image.open(image_fname)
            image.load()
            iid[key] = image
        return iid

    def preprocess_image(self, image):
        # Resize the image.
        image = image.resize((self.W, self.H))
        image = torchvision_F.to_tensor(image)
        if self.white_background:
            image = image[:3] * image[3:] + (1. - image[3:])
        else:
            image = image[:3]
        return image

    def preprocess_image_iid(self, image, iid):
        """
        Args:
            image: rgb
            iid: dict of Ref, Sha and Res
        All of them from blender are in RGBA format.
        However, only the A channel in rgb is useful while the A channel of others are all 1.0.

        Returns:
        """
        # Resize the image.
        image = image.resize((self.W, self.H))
        image = torchvision_F.to_tensor(image)
        for key in iid.keys():
            iid[key] = torchvision_F.to_tensor(iid[key].resize((self.W, self.H)))
            iid[key] = iid[key][:3]
        if self.white_background:
            transparency = image[3:]
            for key in iid.keys():
                iid[key] = iid[key] * transparency + (1. - transparency)
            image = image[:3] * image[3:] + (1. - image[3:])
        else:
            image = image[:3]
            for key in iid.keys():
                assert iid[key].size(0) <= 3
        return image, iid

    def __getitem__(self, idx):
        """Process raw data and return processed data in a dictionary.

        Args:
            idx: The index of the sample of the dataset.
        Returns: A dictionary containing the data.
                 idx (scalar): The index of the sample of the dataset.
                 image (R tensor): Image idx for per-image embedding.
                 image (Rx3 tensor): Image with pixel values in [0,1] for supervision.
                 intr (3x3 tensor): The camera intrinsics of `image`.
                 pose (3x4 tensor): The camera extrinsics [R,t] of `image`.
        """
        # Keep track of sample index for convenience.
        sample = dict(idx=idx)
        # Get the images.
        image, image_size_raw = self.images[idx] if self.preload else self.get_image(idx)
        if not self.load_iid:
            image = self.preprocess_image(image)
        else:
            iid = self.iids[idx] if self.preload else self.get_iid(idx)
            image, iid = self.preprocess_image_iid(image, iid)
        if self.has_pseudo_label:
            # frame = self.list[idx]
            pseudo_elements = {}
            pseudo_elements['pseudo_ref'] = self.pseudo_label[str(idx)]['pseudo_reflectance']
            pseudo_elements['pseudo_sha'] = self.pseudo_label[str(idx)][str(0)]['pseudo_shading_gamma']
            pseudo_elements['pseudo_visibility_certainty'] = self.pseudo_label[str(idx)][str(0)]['visibility_certainty']
            pass
        # Get the cameras (intrinsics and pose).
        # Here this pose means w2c.
        intr, pose = self.cameras[idx] if self.preload else self.get_camera(idx)
        intr, pose = self.preprocess_camera(intr, pose, image_size_raw)
        # get light
        pose_light = self.lights[idx] if self.preload else self.get_light(idx)
        # Pre-sample ray indices.
        if self.sample_train_rays:
            ray_idx = torch.randperm(self.H * self.W)[:self.num_rays]  # [R]
            image_sampled = image.flatten(1, 2)[:, ray_idx].t()  # [R,3]
            sample.update(
                ray_idx=ray_idx,
                image_sampled=image_sampled,
                intr=intr,
                pose=pose,
                pose_light=pose_light
            )
            if self.load_iid:
                for key in iid.keys():
                    assert len(iid[key].size()) == len(image.size())
                    sample[key + '_sampled'] = iid[key].flatten(1, 2)[:, ray_idx].t()  # [R,3]
            if self.has_pseudo_label:
                for key in pseudo_elements.keys():
                    sample[key + '_sampled'] = pseudo_elements[key].flatten(1, 2)[:, ray_idx].t()
        else:  # keep image during inference
            sample.update(
                image=image,
                intr=intr,
                pose=pose,
                pose_light=pose_light
            )
            if self.load_iid:
                for key in iid.keys():
                    sample[key] = iid[key]
        return sample



