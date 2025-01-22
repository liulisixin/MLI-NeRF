import numpy as np
import torch
from torch.nn.functional import cosine_similarity
from PIL import ImageFile
import re
from projects.nerf.utils import camera
from projects.neuralangelo.data import Dataset as AngeloDataset
from projects.nerf.utils.camera import cam2world

ImageFile.LOAD_TRUNCATED_IMAGES = True

class Dataset(AngeloDataset):

    def __init__(self, cfg, is_inference=False, is_test=False):
        super().__init__(cfg, is_inference=is_inference, is_test=is_test)
        # Preload dataset if possible.
        cfg_data = cfg.data
        if cfg_data.preload:
            self.lights = self.preload_threading(self.get_light, cfg_data.num_workers, data_str="lights")
        if self.split == 'train' and hasattr(cfg_data[self.split], 'pseudo_label') and cfg_data[self.split].pseudo_label.enabled:
            self.pseudo_label = torch.load(cfg_data[self.split].pseudo_label.pt_file)
            self.has_pseudo_label = True
        else:
            self.has_pseudo_label = False
        if self.split == "train":
            self.sample_train_rays = True
        else:
            self.sample_train_rays = False

    def get_light(self, idx):
        # light pose.
        c2w_gl = torch.tensor(self.list[idx]["transform_matrix_light"], dtype=torch.float32)
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

    def find_closest_idx(self, pose_cam, pose_light):
        direction_3D = torch.tensor([[0.0, 0.0, 1.0]])  # [HW,3]/[B,HW,3]
        center_3D = torch.tensor([[0.0, 0.0, 0.0]])  # [HW,3]/[B,HW,3]
        # first, calculate the info in the dataset
        if not (hasattr(self, 'dataset_center_ray_info') and self.dataset_center_ray_info):
            lights_pose = torch.stack(self.lights)
            cams_pose = torch.stack([x[1] for x in self.cameras])

            # Transform from camera to world coordinates.
            direction_cams = cam2world(direction_3D, cams_pose)  # [HW,3]/[B,HW,3]
            self.center_cams = cam2world(center_3D, cams_pose)  # [HW,3]/[B,HW,3]
            self.ray_cams = direction_cams - self.center_cams

            self.center_lights = cam2world(center_3D, lights_pose)
            self.dataset_center_ray_info = True
        target_direction_cam = cam2world(direction_3D, pose_cam)  # [HW,3]/[B,HW,3]
        target_center_cam = cam2world(center_3D, pose_cam)  # [HW,3]/[B,HW,3]
        target_ray_cam = target_direction_cam - target_center_cam

        target_center_light = cam2world(center_3D, pose_light)

        dist_center_cam = torch.abs(target_center_cam - self.center_cams).norm(dim=-1)
        dist_ray_cam = 1.0 - cosine_similarity(target_ray_cam, self.ray_cams, dim=-1)
        dist_center_light = torch.abs(target_center_light - self.center_lights).norm(dim=-1)

        dist_total = dist_center_cam + dist_ray_cam + dist_center_light

        closest_idx = torch.argmin(dist_total)

        return closest_idx

    def find_idx_cam_light(self, str_input='c00l00'):
        # str_input is like 'c00l00'
        digits = re.findall(r'\d+', str_input)
        cam_idx = int(digits[0]) if digits else None
        light_idx = int(digits[-1]) if digits else None

        for index, frame in enumerate(self.list):
            if frame['camera_index'] == cam_idx and frame['light_index'] == light_idx:
                return index
        return None

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
        image = self.preprocess_image(image)
        if self.has_pseudo_label:
            frame = self.list[idx]
            camera_index = frame['camera_index']
            light_index = frame['light_index']
            pseudo_elements = {}
            pseudo_elements['pseudo_ref'] = self.pseudo_label[str(camera_index)]['pseudo_reflectance']
            pseudo_elements['pseudo_sha'] = self.pseudo_label[str(camera_index)][str(light_index)]['pseudo_shading_gamma']
            pseudo_elements['pseudo_visibility_certainty'] = self.pseudo_label[str(camera_index)][str(light_index)]['visibility_certainty']
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
        return sample
