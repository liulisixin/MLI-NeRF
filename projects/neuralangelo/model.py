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

from functools import partial
import torch
import torch.nn.functional as torch_F
from collections import defaultdict

from imaginaire.models.base import Model as BaseModel
from projects.nerf.utils import nerf_util, camera, render
from projects.neuralangelo.utils import misc
from projects.neuralangelo.utils.modules import NeuralSDF, NeuralRGB, BackgroundNeRF
from projects.neuralangelo.utils.semi_sphere import semi_sphere_rays

from projects.NeuralLumen.utils.utils import intersect_aabb

from tqdm import tqdm


class Model(BaseModel):

    def __init__(self, cfg_model, cfg_data):
        super().__init__(cfg_model, cfg_data)
        self.cfg_render = cfg_model.render
        self.white_background = cfg_model.background.white
        self.with_background = cfg_model.background.enabled
        self.with_appear_embed = cfg_model.appear_embed.enabled
        self.anneal_end = cfg_model.object.s_var.anneal_end
        self.outside_val = 1000. * (-1 if cfg_model.object.sdf.mlp.inside_out else 1)
        self.image_size_train = cfg_data.train.image_size
        self.image_size_val = cfg_data.val.image_size
        # Define models.
        self.build_model(cfg_model, cfg_data)
        # Define functions.
        self.ray_generator = partial(nerf_util.ray_generator,
                                     camera_ndc=False,
                                     num_rays=cfg_model.render.rand_rays)
        self.sample_dists_from_pdf = partial(nerf_util.sample_dists_from_pdf,
                                             intvs_fine=cfg_model.render.num_samples.fine)
        self.to_full_val_image = partial(misc.to_full_image, image_size=cfg_data.val.image_size)
        if hasattr(cfg_data, "bounding_type") and cfg_data.bounding_type == "box":
            self.bounding_type = "box"
            self.bounding_box_aabb = torch.tensor(cfg_data.bounding_box_aabb)
        else:
            self.bounding_type = "unit_sphere"
        if hasattr(cfg_model, "create_semi_sphere") and cfg_model.create_semi_sphere:
            self.indirect_resolution = 300
            self.semi_sphere = semi_sphere_rays()
            self.semi_sphere.create_semi_sphere_spiral(self.indirect_resolution ** 2)
            self.num_rays = cfg_model.render.rand_rays

    def build_model(self, cfg_model, cfg_data):
        # appearance encoding
        if cfg_model.appear_embed.enabled:
            assert cfg_data.num_images is not None
            self.appear_embed = torch.nn.Embedding(cfg_data.num_images, cfg_model.appear_embed.dim)
            if cfg_model.background.enabled:
                self.appear_embed_outside = torch.nn.Embedding(cfg_data.num_images, cfg_model.appear_embed.dim)
            else:
                self.appear_embed_outside = None
        else:
            self.appear_embed = self.appear_embed_outside = None
        self.neural_sdf = NeuralSDF(cfg_model.object.sdf)
        self.neural_rgb = NeuralRGB(cfg_model.object.rgb, feat_dim=cfg_model.object.sdf.mlp.hidden_dim,
                                    appear_embed=cfg_model.appear_embed)
        if cfg_model.background.enabled:
            self.background_nerf = BackgroundNeRF(cfg_model.background, appear_embed=cfg_model.appear_embed)
        else:
            self.background_nerf = None
        self.s_var = torch.nn.Parameter(torch.tensor(cfg_model.object.s_var.init_val, dtype=torch.float32))

    def forward(self, data):
        # Randomly sample and render the pixels.
        output = self.render_pixels(data["pose"], data["intr"], image_size=self.image_size_train,
                                    stratified=self.cfg_render.stratified, sample_idx=data["idx"],
                                    ray_idx=data["ray_idx"])
        return output

    @torch.no_grad()
    def inference(self, data):
        self.eval()
        # Render the full images.
        output = self.render_image(data["pose"], data["intr"], image_size=self.image_size_val,
                                   stratified=False, sample_idx=data["idx"])  # [B,N,C]
        # Get full rendered RGB and depth images.
        rot = data["pose"][..., :3, :3]  # [B,3,3]
        normal_cam = -output["gradient"] @ rot.transpose(-1, -2)  # [B,HW,3]
        output.update(
            rgb_map=self.to_full_val_image(output["rgb"]),  # [B,3,H,W]
            opacity_map=self.to_full_val_image(output["opacity"]),  # [B,1,H,W]
            depth_map=self.to_full_val_image(output["depth"]),  # [B,1,H,W]
            normal_map=self.to_full_val_image(normal_cam),  # [B,3,H,W]
        )
        return output

    def render_image(self, pose, intr, image_size, stratified=False, sample_idx=None):
        """ Render the rays given the camera intrinsics and poses.
        Args:
            pose (tensor [batch,3,4]): Camera poses ([R,t]).
            intr (tensor [batch,3,3]): Camera intrinsics.
            stratified (bool): Whether to stratify the depth sampling.
            sample_idx (tensor [batch]): Data sample index.
        Returns:
            output: A dictionary containing the outputs.
        """
        output = defaultdict(list)
        for center, ray, _ in self.ray_generator(pose, intr, image_size, full_image=True):
            ray_unit = torch_F.normalize(ray, dim=-1)  # [B,R,3]
            output_batch = self.render_rays(center, ray_unit, sample_idx=sample_idx, stratified=stratified)
            if not self.training:
                dist = render.composite(output_batch["dists"], output_batch["weights"])  # [B,R,1]
                depth = dist / ray.norm(dim=-1, keepdim=True)
                output_batch.update(depth=depth)
            for key, value in output_batch.items():
                if value is not None:
                    output[key].append(value.detach())
        # Concat each item (list) in output into one tensor. Concatenate along the ray dimension (1)
        for key, value in output.items():
            output[key] = torch.cat(value, dim=1)
        return output

    @torch.no_grad()
    def inference_light(self, data, light_pose):
        self.eval()
        # Render the full images.
        output = self.render_image_light(data["pose"], data["intr"], light_pose, image_size=self.image_size_val,
                                   stratified=False, sample_idx=data["idx"])  # [B,N,C]
        inter_pts_range = [output["inter_pts"].min(dim=1)[0], output["inter_pts"].max(dim=1)[0],
                           output["inter_pts"].norm(dim=-1).min(dim=1)[0], output["inter_pts"].norm(dim=-1).max(dim=1)[0]]
        # Get full rendered RGB and depth images.
        rot = data["pose"][..., :3, :3]  # [B,3,3]
        normal_cam = -output["gradient"] @ rot.transpose(-1, -2)  # [B,HW,3]
        output.update(
            rgb_map=self.to_full_val_image(output["rgb"]),  # [B,3,H,W]
            opacity_map=self.to_full_val_image(output["opacity"]),  # [B,1,H,W]
            depth_map=self.to_full_val_image(output["depth"]),  # [B,1,H,W]
            normal_map=self.to_full_val_image(normal_cam),  # [B,3,H,W]
            normal_x_light_map=self.to_full_val_image(output["normal_x_light"]),
            visibility_map=self.to_full_val_image(output['visibility']),
            light_opacity_map=self.to_full_val_image(output['light_opacity']),
            indirect_rgb_map=self.to_full_val_image(output["indirect_rgb"], from_vec=False)
        )
        return output, inter_pts_range

    def render_image_light(self, pose, intr, light_pose, image_size, stratified=False, sample_idx=None):
        """ Render the rays given the camera intrinsics and poses.
        Args:
            pose (tensor [batch,3,4]): Camera poses ([R,t]).
            intr (tensor [batch,3,3]): Camera intrinsics.
            stratified (bool): Whether to stratify the depth sampling.
            sample_idx (tensor [batch]): Data sample index.
        Returns:
            output: A dictionary containing the outputs.
        """
        output = defaultdict(list)
        for center, ray, _ in tqdm(self.ray_generator(pose, intr, image_size, full_image=True)):
            ray_unit = torch_F.normalize(ray, dim=-1)  # [B,R,3]
            output_batch = self.render_rays(center, ray_unit, sample_idx=sample_idx, stratified=stratified)
            if not self.training:
                dist = render.composite(output_batch["dists"], output_batch["weights"])  # [B,R,1]
                depth = dist / ray.norm(dim=-1, keepdim=True)
                output_batch.update(depth=depth)
                method_intersection = "sphere_tracing"
                if method_intersection != "sphere_tracing":
                    inter_pts = camera.get_3D_points_from_dist(center, ray_unit, dist, multi=False)
                    output_batch.update(inter_pts=inter_pts)
                else:
                    with torch.no_grad():
                        near, far, outside = self.get_dist_bounds(center, ray_unit)
                    _, inter_pts, _ = self.sphere_tracing_intersection(center, ray_unit, near, far)
                    output_batch.update(inter_pts=inter_pts)
                # light ray
                light_loc = torch.from_numpy(light_pose[:3, 3]).to(self.device())
                light_ray = inter_pts - light_loc
                light_ray_unit = torch_F.normalize(light_ray, dim=-1)
                setting_dist = 0.20
                center_pts = inter_pts - light_ray_unit * setting_dist
                # visibility
                if method_intersection != "sphere_tracing":
                    output_batch_light = self.render_rays(center_pts, light_ray_unit, sample_idx=sample_idx,
                                                          stratified=stratified)
                    dist_light = render.composite(output_batch_light["dists"], output_batch_light["weights"])  # [B,R,1]
                    difference = torch.abs(setting_dist - dist_light)
                    visibility = difference < 0.05
                else:
                    far = light_ray.norm(dim=-1, keepdim=True) - 1e-3  # 1e-3 is a tolerance for reaching the limit in sphere_tracing_intersection
                    near = far - 0.20
                    radius_considering_visibility = 0.20
                    dist_near, dist_far = nerf_util.intersect_with_sphere(light_loc[None, None, :], light_ray_unit, radius=radius_considering_visibility)
                    dist_near.relu_()  # Distance (and thus depth) should be non-negative.
                    outside = dist_near.isnan()
                    near = torch.maximum(near, dist_near)
                    # near[outside] = far[outside]
                    dist_light, inter_pts_light, mask_light = self.sphere_tracing_intersection(light_loc[None, None, :], light_ray_unit, near, far)
                    # distance = (inter_pts - inter_pts_light).norm(dim=-1, keepdim=True)
                    # visibility = distance < 0.01
                    visibility = torch.logical_or(~mask_light, outside)

                # opacity
                # light_opacity = output_batch_light["opacity"]
                # light_opacity = output_batch_light['rgb']
                light_opacity = 1 / (dist_light + 1e-8) * 0.5
                # light_opacity = (0.05 / difference).clip(0.0, 1.0)
                # shading
                normal_ray = -output_batch["gradient"]
                dot_product = (normal_ray * light_ray).sum(dim=-1, keepdim=True)
                incident_norm = light_ray.norm(dim=-1, keepdim=True)
                gradients_norm = normal_ray.norm(dim=-1, keepdim=True)
                normal_x_light = dot_product / (incident_norm * gradients_norm)

                # update
                output_batch.update(normal_x_light=normal_x_light)
                output_batch.update(visibility=visibility)
                output_batch.update(light_opacity=light_opacity)

                output_batch.update(normal_ray=normal_ray)

            for key, value in output_batch.items():
                if value is not None:
                    output[key].append(value.detach())

        # Concat each item (list) in output into one tensor. Concatenate along the ray dimension (1)
        for key, value in output.items():
            output[key] = torch.cat(value, dim=1)

        # get indirect light for the center point
        inter_pts_pic = output['inter_pts'].unflatten(dim=1, sizes=image_size)
        normal_ray_pic = output['normal_ray'].unflatten(dim=1, sizes=image_size)
        skip_distance = int(self.indirect_resolution / 10 * 3)
        indice_1 = list(range(int(skip_distance / 2), image_size[0], skip_distance))
        indice_2 = list(range(int(skip_distance / 2), image_size[1], skip_distance))

        grid_i, grid_j = torch.meshgrid(torch.tensor(indice_1), torch.tensor(indice_2), indexing='ij')

        selected_inter_pts = inter_pts_pic[:, grid_i, grid_j, :]
        # normal points to the inside, use -1
        selected_normal_ray = -normal_ray_pic[:, grid_i, grid_j, :]

        ori_vector = self.semi_sphere.original_semi_sphere.to(self.device())
        rotation_matrix = self.semi_sphere.rotation_matrix_from_z_to_vector(selected_normal_ray)
        rotated_vectors = torch.einsum('...ij,kj->...ki', rotation_matrix, ori_vector)
        center_inter_pts = selected_inter_pts.unsqueeze(-2).expand(rotated_vectors.shape)

        original_shape = rotated_vectors.shape
        rotated_vectors = rotated_vectors.flatten(start_dim=1, end_dim=-2)
        center_inter_pts = center_inter_pts.flatten(start_dim=1, end_dim=-2)

        rotated_vectors = torch_F.normalize(rotated_vectors, dim=-1)  # [B,R,3]
        indirect_rgb = []
        in_range_mask = []
        for c in tqdm(range(0, rotated_vectors.shape[1], self.num_rays)):
            center_indirect = center_inter_pts[:, c:c + self.num_rays]
            ray_unit_indirect = rotated_vectors[:, c:c + self.num_rays]

            output_batch = self.render_rays(center_indirect, ray_unit_indirect, sample_idx=sample_idx, stratified=stratified)
            indirect_rgb.append(output_batch['rgb'].detach())

            # here we need to determin whether the rays go beyond the view field of cameras
            method = "method1"
            if method == "method1":
                # method 1, 51sec.
                dist = render.composite(output_batch["dists"], output_batch["weights"])  # [B,R,1]
                inter_pts = camera.get_3D_points_from_dist(center_indirect, ray_unit_indirect, dist, multi=False)
            else:
                # method 2, 2min44sec. It fails because, in L-tracing, starting from a surface point is very slow.
                with torch.no_grad():
                    near, far, outside = self.get_dist_bounds(center_indirect, ray_unit_indirect)
                near[near < 0.01] = 0.01
                _, inter_pts, _ = self.sphere_tracing_intersection(center_indirect, ray_unit_indirect, near, far, Num_iters=50)
            in_range_mask.append(camera.inside_camera_view(intr, pose, image_size, inter_pts))

        indirect_rgb = torch.cat(indirect_rgb, dim=1)
        in_range_mask = torch.cat(in_range_mask, dim=1)
        indirect_rgb[~in_range_mask, :] = 0.0
        indirect_rgb = indirect_rgb.reshape(original_shape)
        # transform from the indices of spiral indices to the square interpolation indices
        indirect_rgb = indirect_rgb[:, :, :, self.semi_sphere.square_interpolation_index, :]
        indirect_rgb[:, :, :, ~self.semi_sphere.square_mask, :] = 1.0
        square_size = int(indirect_rgb.shape[-2] ** 0.5)
        indirect_rgb = indirect_rgb.unflatten(dim=-2, sizes=(square_size, square_size))
        indirect_rgb = indirect_rgb.permute(0, 1, 4, 2, 3, 5)
        indirect_rgb = indirect_rgb.contiguous()
        indirect_rgb = indirect_rgb.flatten(start_dim=3, end_dim=4)
        indirect_rgb = indirect_rgb.flatten(start_dim=1, end_dim=2)
        output['indirect_rgb'] = indirect_rgb

        return output

    def sphere_tracing_intersection(self, center, ray_unit, near, far, Num_iters=20, dist_start=None):
        """
        Borrowed from L-Tracing: Fast Light Visibility Estimation on Neural Surfaces by Sphere Tracing [ECCV'22]
        But with important modification. (yyx)
        """
        if dist_start is not None:
            dist = dist_start
        else:
            dist = near.clone()
        # mask = near < far
        mask = torch.ones_like(dist, dtype=torch.bool).to(self.device())
        for _ in range(Num_iters):
            pts = center + ray_unit * dist
            sdfs, feats = self.neural_sdf.forward(pts)
            # if (sdfs < 0).any():
            #     print('Warning: negative sdf')   # sdf can be negative inside the surface.
            dist[mask] += sdfs[mask]
            mask[dist > far] = False
            # mask[dist < 0] = False
            """ yyx
            Here we should use near. The previous one do not consider that sdf may be negative. 
            Actually, it is enough to check only in the range near < dist < far.
            """
            mask[dist < near] = False
        dist = torch.clamp(dist, near, far)   # yyx: clamp the range
        pts = center + ray_unit * dist

        return dist, pts, mask



    def render_pixels(self, pose, intr, image_size, stratified=False, sample_idx=None, ray_idx=None):
        center, ray = camera.get_center_and_ray(pose, intr, image_size)  # [B,HW,3]
        center = nerf_util.slice_by_ray_idx(center, ray_idx)  # [B,R,3]
        ray = nerf_util.slice_by_ray_idx(ray, ray_idx)  # [B,R,3]
        ray_unit = torch_F.normalize(ray, dim=-1)  # [B,R,3]
        output = self.render_rays(center, ray_unit, sample_idx=sample_idx, stratified=stratified)
        return output

    def render_rays(self, center, ray_unit, sample_idx=None, stratified=False):
        with torch.no_grad():
            near, far, outside = self.get_dist_bounds(center, ray_unit)
        app, app_outside = self.get_appearance_embedding(sample_idx, ray_unit.shape[1])
        output_object = self.render_rays_object(center, ray_unit, near, far, outside, app, stratified=stratified)
        if self.with_background:
            output_background = self.render_rays_background(center, ray_unit, far, app_outside, stratified=stratified)
            # Concatenate object and background samples.
            rgbs = torch.cat([output_object["rgbs"], output_background["rgbs"]], dim=2)  # [B,R,No+Nb,3]
            dists = torch.cat([output_object["dists"], output_background["dists"]], dim=2)  # [B,R,No+Nb,1]
            alphas = torch.cat([output_object["alphas"], output_background["alphas"]], dim=2)  # [B,R,No+Nb]
        else:
            rgbs = output_object["rgbs"]  # [B,R,No,3]
            dists = output_object["dists"]  # [B,R,No,1]
            alphas = output_object["alphas"]  # [B,R,No]
        weights = render.alpha_compositing_weights(alphas)  # [B,R,No+Nb,1]
        # Compute weights and composite samples.
        rgb = render.composite(rgbs, weights)  # [B,R,3]
        if self.white_background:
            opacity_all = render.composite(1., weights)  # [B,R,1]
            rgb = rgb + (1 - opacity_all)
        # Collect output.
        output = dict(
            rgb=rgb,  # [B,R,3]
            opacity=output_object["opacity"],  # [B,R,1]/None
            outside=outside,  # [B,R,1]
            dists=dists,  # [B,R,No+Nb,1]
            weights=weights,  # [B,R,No+Nb,1]
            gradient=output_object["gradient"],  # [B,R,3]/None
            gradients=output_object["gradients"],  # [B,R,No,3]
            hessians=output_object["hessians"],  # [B,R,No,3]/None
        )
        return output

    def render_rays_object(self, center, ray_unit, near, far, outside, app, stratified=False):
        with torch.no_grad():
            dists = self.sample_dists_all(center, ray_unit, near, far, stratified=stratified)  # [B,R,N,3]
        points = camera.get_3D_points_from_dist(center, ray_unit, dists)  # [B,R,N,3]
        sdfs, feats = self.neural_sdf.forward(points)  # [B,R,N,1],[B,R,N,K]
        sdfs[outside[..., None].expand_as(sdfs)] = self.outside_val
        # Compute 1st- and 2nd-order gradients.
        rays_unit = ray_unit[..., None, :].expand_as(points).contiguous()  # [B,R,N,3]
        gradients, hessians = self.neural_sdf.compute_gradients(points, training=self.training, sdf=sdfs)
        normals = torch_F.normalize(gradients, dim=-1)  # [B,R,N,3]
        rgbs = self.neural_rgb.forward(points, normals, rays_unit, feats, app=app)  # [B,R,N,3]
        # SDF volume rendering.
        alphas = self.compute_neus_alphas(ray_unit, sdfs, gradients, dists, dist_far=far[..., None],
                                          progress=self.progress)  # [B,R,N]
        if not self.training:
            weights = render.alpha_compositing_weights(alphas)  # [B,R,N,1]
            opacity = render.composite(1., weights)  # [B,R,1]
            gradient = render.composite(gradients, weights)  # [B,R,3]
        else:
            opacity = None
            gradient = None
        # Collect output.
        output = dict(
            rgbs=rgbs,  # [B,R,N,3]
            sdfs=sdfs[..., 0],  # [B,R,N]
            dists=dists,  # [B,R,N,1]
            alphas=alphas,  # [B,R,N]
            opacity=opacity,  # [B,R,3]/None
            gradient=gradient,  # [B,R,3]/None
            gradients=gradients,  # [B,R,N,3]
            hessians=hessians,  # [B,R,N,3]/None
        )
        return output

    def render_rays_background(self, center, ray_unit, far, app_outside, stratified=False):
        with torch.no_grad():
            dists = self.sample_dists_background(ray_unit, far, stratified=stratified)
        points = camera.get_3D_points_from_dist(center, ray_unit, dists)  # [B,R,N,3]
        rays_unit = ray_unit[..., None, :].expand_as(points)  # [B,R,N,3]
        rgbs, densities = self.background_nerf.forward(points, rays_unit, app_outside)  # [B,R,N,3]
        alphas = render.volume_rendering_alphas_dist(densities, dists)  # [B,R,N]
        # Collect output.
        output = dict(
            rgbs=rgbs,  # [B,R,3]
            dists=dists,  # [B,R,N,1]
            alphas=alphas,  # [B,R,N]
        )
        return output

    @torch.no_grad()
    def get_dist_bounds(self, center, ray_unit):
        if self.bounding_type == "box":
            dist_near, dist_far, outside = intersect_aabb(center, ray_unit, aabb=self.bounding_box_aabb)
            dist_near[outside], dist_far[outside] = 1, 1.2
        else:
            dist_near, dist_far = nerf_util.intersect_with_sphere(center, ray_unit, radius=1.)
            dist_near.relu_()  # Distance (and thus depth) should be non-negative.
            outside = dist_near.isnan()
            dist_near[outside], dist_far[outside] = 1, 1.2  # Dummy distances. Density will be set to 0.
        return dist_near, dist_far, outside

    def get_appearance_embedding(self, sample_idx, num_rays):
        if self.with_appear_embed:
            # Object appearance embedding.
            num_samples_all = self.cfg_render.num_samples.coarse + \
                self.cfg_render.num_samples.fine * self.cfg_render.num_sample_hierarchy
            app = self.appear_embed(sample_idx)[:, None, None]  # [B,1,1,C]
            app = app.expand(-1, num_rays, num_samples_all, -1)  # [B,R,N,C]
            # Background appearance embedding.
            if self.with_background:
                app_outside = self.appear_embed_outside(sample_idx)[:, None, None]  # [B,1,1,C]
                app_outside = app_outside.expand(-1, num_rays, self.cfg_render.num_samples.background, -1)  # [B,R,N,C]
            else:
                app_outside = None
        else:
            app = app_outside = None
        return app, app_outside

    @torch.no_grad()
    def sample_dists_all(self, center, ray_unit, near, far, stratified=False):
        dists = nerf_util.sample_dists(ray_unit.shape[:2], dist_range=(near[..., None], far[..., None]),
                                       intvs=self.cfg_render.num_samples.coarse, stratified=stratified)
        if self.cfg_render.num_sample_hierarchy > 0:
            points = camera.get_3D_points_from_dist(center, ray_unit, dists)  # [B,R,N,3]
            sdfs = self.neural_sdf.sdf(points)  # [B,R,N]
        for h in range(self.cfg_render.num_sample_hierarchy):
            dists_fine = self.sample_dists_hierarchical(dists, sdfs, inv_s=(64 * 2 ** h))  # [B,R,Nf,1]
            dists = torch.cat([dists, dists_fine], dim=2)  # [B,R,N+Nf,1]
            dists, sort_idx = dists.sort(dim=2)
            if h != self.cfg_render.num_sample_hierarchy - 1:
                points_fine = camera.get_3D_points_from_dist(center, ray_unit, dists_fine)  # [B,R,Nf,3]
                sdfs_fine = self.neural_sdf.sdf(points_fine)  # [B,R,Nf]
                sdfs = torch.cat([sdfs, sdfs_fine], dim=2)  # [B,R,N+Nf]
                sdfs = sdfs.gather(dim=2, index=sort_idx.expand_as(sdfs))  # [B,R,N+Nf,1]
        return dists

    def sample_dists_hierarchical(self, dists, sdfs, inv_s, robust=True, eps=1e-5):
        sdfs = sdfs[..., 0]  # [B,R,N]
        prev_sdfs, next_sdfs = sdfs[..., :-1], sdfs[..., 1:]  # [B,R,N-1]
        prev_dists, next_dists = dists[..., :-1, 0], dists[..., 1:, 0]  # [B,R,N-1]
        mid_sdfs = (prev_sdfs + next_sdfs) * 0.5  # [B,R,N-1]
        cos_val = (next_sdfs - prev_sdfs) / (next_dists - prev_dists + 1e-5)  # [B,R,N-1]
        if robust:
            prev_cos_val = torch.cat([torch.zeros_like(cos_val)[..., :1], cos_val[..., :-1]], dim=-1)  # [B,R,N-1]
            cos_val = torch.stack([prev_cos_val, cos_val], dim=-1).min(dim=-1).values  # [B,R,N-1]
        dist_intvs = dists[..., 1:, 0] - dists[..., :-1, 0]  # [B,R,N-1]
        est_prev_sdf = mid_sdfs - cos_val * dist_intvs * 0.5  # [B,R,N-1]
        est_next_sdf = mid_sdfs + cos_val * dist_intvs * 0.5  # [B,R,N-1]
        prev_cdf = (est_prev_sdf * inv_s).sigmoid()  # [B,R,N-1]
        next_cdf = (est_next_sdf * inv_s).sigmoid()  # [B,R,N-1]
        alphas = ((prev_cdf - next_cdf) / (prev_cdf + eps)).clip_(0.0, 1.0)  # [B,R,N-1]
        weights = render.alpha_compositing_weights(alphas)  # [B,R,N-1,1]
        dists_fine = self.sample_dists_from_pdf(dists, weights=weights[..., 0])  # [B,R,Nf,1]
        return dists_fine

    def sample_dists_background(self, ray_unit, far, stratified=False, eps=1e-5):
        inv_dists = nerf_util.sample_dists(ray_unit.shape[:2], dist_range=(1, 0),
                                           intvs=self.cfg_render.num_samples.background, stratified=stratified)
        dists = far[..., None] / (inv_dists + eps)  # [B,R,N,1]
        return dists

    def compute_neus_alphas(self, ray_unit, sdfs, gradients, dists, dist_far=None, progress=1., eps=1e-5):
        sdfs = sdfs[..., 0]  # [B,R,N]
        # SDF volume rendering in NeuS.
        inv_s = self.s_var.exp()
        true_cos = (ray_unit[..., None, :] * gradients).sum(dim=-1, keepdim=False)  # [B,R,N]
        iter_cos = self._get_iter_cos(true_cos, progress=progress)  # [B,R,N]
        # Estimate signed distances at section points
        if dist_far is None:
            dist_far = torch.empty_like(dists[..., :1, :]).fill_(1e10)  # [B,R,1,1]
        dists = torch.cat([dists, dist_far], dim=2)  # [B,R,N+1,1]
        dist_intvs = dists[..., 1:, 0] - dists[..., :-1, 0]  # [B,R,N]
        est_prev_sdf = sdfs - iter_cos * dist_intvs * 0.5  # [B,R,N]
        est_next_sdf = sdfs + iter_cos * dist_intvs * 0.5  # [B,R,N]
        prev_cdf = (est_prev_sdf * inv_s).sigmoid()  # [B,R,N]
        next_cdf = (est_next_sdf * inv_s).sigmoid()  # [B,R,N]
        alphas = ((prev_cdf - next_cdf) / (prev_cdf + eps)).clip_(0.0, 1.0)  # [B,R,N]
        # weights = render.alpha_compositing_weights(alphas)  # [B,R,N,1]
        return alphas

    def _get_iter_cos(self, true_cos, progress=1.):
        anneal_ratio = min(progress / self.anneal_end, 1.)
        # The anneal strategy below keeps the cos value alive at the beginning of training iterations.
        return -((-true_cos * 0.5 + 0.5).relu() * (1.0 - anneal_ratio) +
                 (-true_cos).relu() * anneal_ratio)  # always non-positive
