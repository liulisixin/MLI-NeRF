
from functools import partial
import torch
import torch.nn.functional as torch_F
from collections import defaultdict

from projects.nerf.utils import nerf_util, camera, render
from projects.neuralangelo.utils.modules import NeuralSDF

from projects.NeuralLumen.utils.modules import LumenRGB, LumenBackgroundNeRF
from projects.NeuralLumen.utils.ray_generator import ray_generator
from projects.NeuralLumen.utils.utils import get_center, intersect_aabb

from projects.neuralangelo.model import Model as AngeloModel


class Model(AngeloModel):

    def __init__(self, cfg_model, cfg_data):
        super().__init__(cfg_model, cfg_data)
        rand_rays_val = cfg_model.render.rand_rays_val if hasattr(cfg_model.render, 'rand_rays_val') else cfg_model.render.rand_rays
        self.ray_generator = partial(ray_generator,
                                     camera_ndc=False,
                                     num_rays=rand_rays_val)
        self.flag_light_visibility = hasattr(cfg_model, 'light_visibility') and cfg_model.light_visibility.enabled
        if self.flag_light_visibility:
            self.para_light_visibility = cfg_model.light_visibility
            if self.para_light_visibility.visibility_bounding_type == "box":
                self.visibility_bounding_box_aabb = torch.tensor(
                    self.para_light_visibility.visibility_bounding_box_aabb)
            if hasattr(self.para_light_visibility, 'gamma_correlation'):
                self.flag_gamma_correlation = True
                self.gamma_for_shading = self.para_light_visibility.gamma_correlation
            else:
                self.flag_gamma_correlation = False

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
        # for light
        self.rgb_network_mode = cfg_model.object.rgb.network_mode if hasattr(cfg_model.object.rgb,
                                                                             'network_mode') else None
        self.neural_rgb = LumenRGB(cfg_model.object.rgb, feat_dim=cfg_model.object.sdf.mlp.hidden_dim,
                                    appear_embed=cfg_model.appear_embed)
        if cfg_model.background.enabled:
            self.background_nerf = LumenBackgroundNeRF(cfg_model.background, appear_embed=cfg_model.appear_embed)
        else:
            self.background_nerf = None
        self.s_var = torch.nn.Parameter(torch.tensor(cfg_model.object.s_var.init_val, dtype=torch.float32))

    @torch.no_grad()
    def inference(self, data):
        self.eval()
        # Render the full images.
        output = self.render_image_lumen(data["pose"], data["intr"], data["pose_light"], image_size=self.image_size_val,
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
        for key in ['o_r', 'o_s', 'o_re']:
            if key in output.keys():
                output[key + '_map'] = self.to_full_val_image(output[key])
        if self.flag_light_visibility:
            output['visibility_map'] = self.to_full_val_image(output['visibility']).float()
            output['normal_x_light_map'] = self.to_full_val_image(output['normal_x_light'])
            output['pseudo_shading_map'] = self.to_full_val_image(output['pseudo_shading'])
            output['inter_dist_map'] = self.to_full_val_image(output['inter_dist']).float()
            output['inter_mask_map'] = self.to_full_val_image(output['inter_mask']).float()
        return output

    def render_image_lumen(self, pose, intr, pose_light, image_size, stratified=False, sample_idx=None):
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
        for center, ray, pts_light, _ in self.ray_generator(pose, intr, pose_light, image_size, full_image=True):
            ray_unit = torch_F.normalize(ray, dim=-1)  # [B,R,3]
            output_batch = self.render_rays_lumen(center, ray_unit, pts_light, sample_idx=sample_idx,
                                                  stratified=stratified)
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

    def forward(self, data):
        # Randomly sample and render the pixels.
        output = self.render_pixels_lumen(data["pose"], data["intr"], data["pose_light"], image_size=self.image_size_train,
                                    stratified=self.cfg_render.stratified, sample_idx=data["idx"],
                                    ray_idx=data["ray_idx"])
        return output

    def render_pixels_lumen(self, pose, intr, pose_light, image_size, stratified=False, sample_idx=None, ray_idx=None):
        # camera ray
        center, ray = camera.get_center_and_ray(pose, intr, image_size)  # [B,HW,3]
        center = nerf_util.slice_by_ray_idx(center, ray_idx)  # [B,R,3]
        ray = nerf_util.slice_by_ray_idx(ray, ray_idx)  # [B,R,3]
        ray_unit = torch_F.normalize(ray, dim=-1)  # [B,R,3]
        # light ray
        pts_light = get_center(pose_light, image_size)  # [B,HW,3]
        pts_light = nerf_util.slice_by_ray_idx(pts_light, ray_idx)

        output = self.render_rays_lumen(center, ray_unit, pts_light, sample_idx=sample_idx, stratified=stratified)
        return output

    def get_light_visibility(self, center, ray_unit, pts_light, near, far, outside, render_output, stratified=False):
        with torch.no_grad():
            camera_ray_type = self.para_light_visibility.camera_ray_type
            if camera_ray_type == 'blend_z_sphere_tracing':
                blend_dist = render.composite(render_output["dists"], render_output["weights"])  # [B,R,1]
                inter_dist, inter_pts, inter_mask = self.sphere_tracing_intersection(center, ray_unit, near, far,
                                                                                     dist_start=blend_dist)
            elif camera_ray_type == 'blend_z':
                inter_dist = render.composite(render_output["dists"], render_output["weights"])  # [B,R,1]
                inter_pts = center + ray_unit * inter_dist
                inter_mask = inter_dist > 0.0
            elif camera_ray_type == 'sphere_tracing':
                inter_dist, inter_pts, inter_mask = self.sphere_tracing_intersection(center, ray_unit, near, far)
            else:
                raise NotImplementedError
        # light ray
        light_loc = pts_light
        light_ray = inter_pts - light_loc
        dists_light_inter = light_ray.norm(dim=-1, keepdim=True)
        light_ray_unit = torch_F.normalize(light_ray, dim=-1)

        method = self.para_light_visibility.type
        if method == "render_light_visibility":
            with torch.no_grad():
                near_light, far_light, outside_light = self.get_dist_bounds(light_loc, light_ray_unit)
            inside_space = torch.logical_and(~outside, ~outside_light)
            visibility = self.render_light_visibility(light_loc, light_ray_unit, near_light, far_light, outside_light,
                                                            dists_light_inter,
                                                            stratified=stratified)
            visibility = torch.logical_and(visibility, inside_space)
        elif method == "sphere_tracing":
            with torch.no_grad():
                near_light, far_light, outside_light = self.get_dist_bounds_visibility(light_loc, light_ray_unit)
                far_tracing = light_ray.norm(dim=-1, keepdim=True) - 1e-3
                # here we only consider the points inside the bounding.
                inside_bounding = torch.logical_and(near_light < far_tracing, far_tracing < far_light)
                inside_bounding = torch.logical_and(inside_bounding, ~outside_light)
                # 1e-3 is a tolerance for reaching the limit in sphere_tracing_intersection
                dist_light, inter_pts_light, mask_light = self.sphere_tracing_intersection(light_loc, light_ray_unit,
                                                                                           near_light, far_tracing)
            visibility = torch.logical_or(~mask_light, ~inside_bounding)
        else:
            raise NotImplementedError

        # normal * light
        normal_ray = -render_output["gradient"]
        normal_ray_unit = torch_F.normalize(normal_ray, dim=-1)
        normal_x_light = (normal_ray_unit * light_ray_unit).sum(dim=-1, keepdim=True)
        normal_x_light = normal_x_light.relu_()
        # incident_norm = light_ray.norm(dim=-1, keepdim=True)
        # gradients_norm = normal_ray.norm(dim=-1, keepdim=True)
        return visibility, normal_x_light, inter_dist, inter_mask

    @torch.no_grad()
    def get_dist_bounds_visibility(self, center, ray_unit):
        bounding_type = self.para_light_visibility.visibility_bounding_type
        if bounding_type == "box":
            dist_near, dist_far, outside = intersect_aabb(center, ray_unit, aabb=self.bounding_box_aabb)
            dist_near[outside], dist_far[outside] = 1, 1.2
        elif bounding_type == "sphere":
            bounding_radius = self.para_light_visibility.visibility_sphere_radius
            dist_near, dist_far = nerf_util.intersect_with_sphere(center, ray_unit, radius=bounding_radius)
            dist_near.relu_()  # Distance (and thus depth) should be non-negative.
            outside = dist_near.isnan()
            dist_near[outside], dist_far[outside] = 1, 1.2  # Dummy distances. Density will be set to 0.
        else:
            raise NotImplementedError
        return dist_near, dist_far, outside

    def render_light_visibility(self, center, ray_unit, near, far, outside, dists_inter, stratified=False):
        with torch.no_grad():
            dists = self.sample_dists_all(center, ray_unit, near, far, stratified=stratified)  # [B,R,N,3]
        points = camera.get_3D_points_from_dist(center, ray_unit, dists)  # [B,R,N,3]
        sdfs, feats = self.neural_sdf.forward(points)  # [B,R,N,1],[B,R,N,K]
        sdfs[outside[..., None].expand_as(sdfs)] = self.outside_val
        # Compute 1st- and 2nd-order gradients.
        gradients, hessians = self.neural_sdf.compute_gradients(points, training=self.training, sdf=sdfs)
        # SDF volume rendering.
        alphas = self.compute_neus_alphas(ray_unit, sdfs, gradients, dists, dist_far=far[..., None],
                                          progress=self.progress)  # [B,R,N]
        weights = render.alpha_compositing_weights(alphas)  # [B,R,N,1]
        # opacity = render.composite(1., weights)  # [B,R,1]

        dist_index = torch.searchsorted(dists.squeeze(-1), dists_inter)
        range_left = 1 - int(self.light_visibility_sample_tolerance / 2)
        range_right = 1 + int(self.light_visibility_sample_tolerance / 2)
        indices = dist_index + torch.arange(range_left, range_right).unsqueeze(0).unsqueeze(0).to(dist_index.device)
        # assert indices.max() < 128
        # remember to clamp the range.
        indices = indices.clamp(min=0, max=weights.size()[2] - 1)
        expanded_indices = indices.unsqueeze(-1)
        selected_weights = torch.gather(weights, 2, expanded_indices)

        max_weight_around_intersection, _ = selected_weights.max(dim=2)
        max_weight_total, _ = weights.max(dim=2)
        visibility = max_weight_around_intersection / max_weight_total

        return visibility

    def render_rays_lumen(self, center, ray_unit, pts_light, sample_idx=None, stratified=False):
        with torch.no_grad():
            near, far, outside = self.get_dist_bounds(center, ray_unit)
        app, app_outside = self.get_appearance_embedding(sample_idx, ray_unit.shape[1])
        output_object = self.render_rays_object_lumen(center, ray_unit, pts_light, near, far, outside, app,
                                                      stratified=stratified)
        if self.rgb_network_mode in ['r_s', 'rgb_r_s']:
            intrinsic_dict = {'o_r': None,
                              'o_s': None}
        elif self.rgb_network_mode == 'r_s_re':
            intrinsic_dict = {'o_r': None,
                              'o_s': None,
                              'o_re': None}
        elif self.rgb_network_mode == 'rgb_r':
            intrinsic_dict = {'o_r': None}
        if self.with_background:
            if hasattr(self, 'rgb_network_mode'):
                raise NotImplementedError
            output_background = self.render_rays_background_lumen(center, ray_unit, pts_light, far, app_outside,
                                                                  stratified=stratified)
            # Concatenate object and background samples.
            rgbs = torch.cat([output_object["rgbs"], output_background["rgbs"]], dim=2)  # [B,R,No+Nb,3]
            dists = torch.cat([output_object["dists"], output_background["dists"]], dim=2)  # [B,R,No+Nb,1]
            alphas = torch.cat([output_object["alphas"], output_background["alphas"]], dim=2)  # [B,R,No+Nb]
            if hasattr(self, 'rgb_network_mode'):
                for key in intrinsic_dict.keys():
                    intrinsic_dict[key] = torch.cat([output_object[key], output_background[key]], dim=2)  # [B,R,No+Nb,3]
        else:
            rgbs = output_object["rgbs"]  # [B,R,No,3]
            dists = output_object["dists"]  # [B,R,No,1]
            alphas = output_object["alphas"]  # [B,R,No]
            if self.rgb_network_mode in ['r_s', 'r_s_re', 'rgb_r', 'rgb_r_s']:
                for key in intrinsic_dict.keys():
                    intrinsic_dict[key] = output_object[key]
        weights = render.alpha_compositing_weights(alphas)  # [B,R,No+Nb,1]
        # Compute weights and composite samples.
        # yyx: we should merge reflectance and shading at here.
        if self.rgb_network_mode in ['r_s', 'r_s_re']:
            intrinsic_dict_accu = {}
            for key in intrinsic_dict.keys():
                intrinsic_dict_accu[key] = render.composite(intrinsic_dict[key], weights)
            if self.white_background:
                opacity_all = render.composite(1., weights)  # [B,R,1]
                for key in intrinsic_dict_accu.keys():
                    intrinsic_dict_accu[key] = intrinsic_dict_accu[key] + (1 - opacity_all)
            # multiple to get the rgb
            if self.rgb_network_mode == 'r_s':
                rgb = intrinsic_dict_accu['o_r'] * intrinsic_dict_accu['o_s']
            elif self.rgb_network_mode == 'r_s_re':
                rgb = intrinsic_dict_accu['o_r'] * intrinsic_dict_accu['o_s'] + intrinsic_dict_accu['o_re']
            else:
                raise NotImplementedError
        elif self.rgb_network_mode == 'rgb_r':
            rgb = render.composite(rgbs, weights)  # [B,R,3]
            intrinsic_dict_accu = {}
            intrinsic_dict_accu['o_r'] = render.composite(intrinsic_dict['o_r'], weights)
            if self.white_background:
                opacity_all = render.composite(1., weights)  # [B,R,1]
                rgb = rgb + (1 - opacity_all)
                intrinsic_dict_accu['o_r'] = intrinsic_dict_accu['o_r'] + (1 - opacity_all)
            # divide to get the shading
            intrinsic_dict_accu['o_s'] = rgb / intrinsic_dict_accu['o_r']
        elif self.rgb_network_mode == 'rgb_r_s':
            rgb = render.composite(rgbs, weights)  # [B,R,3]
            intrinsic_dict_accu = {}
            intrinsic_dict_accu['o_r'] = render.composite(intrinsic_dict['o_r'], weights)
            intrinsic_dict_accu['o_s'] = render.composite(intrinsic_dict['o_s'], weights)
            if self.white_background:
                opacity_all = render.composite(1., weights)  # [B,R,1]
                rgb = rgb + (1 - opacity_all)
                intrinsic_dict_accu['o_r'] = intrinsic_dict_accu['o_r'] + (1 - opacity_all)
                intrinsic_dict_accu['o_s'] = intrinsic_dict_accu['o_s'] + (1 - opacity_all)
            # get the residual
            intrinsic_dict_accu['o_re'] = rgb - intrinsic_dict_accu['o_r'] * intrinsic_dict_accu['o_s']   # range=[-1,1]
        else:
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
        if self.rgb_network_mode in ['r_s', 'r_s_re', 'rgb_r', 'rgb_r_s']:
            output.update(intrinsic_dict_accu)

        if self.flag_light_visibility:
            visibility, normal_x_light, inter_dist, inter_mask = self.get_light_visibility(center, ray_unit, pts_light, near, far, outside, output,
                                                                   stratified=stratified)
            output['visibility'] = visibility
            output['normal_x_light'] = normal_x_light
            output['pseudo_shading'] = normal_x_light * visibility.float()
            output['inter_dist'] = inter_dist
            output['inter_mask'] = inter_mask
            if self.flag_gamma_correlation:
                output['pseudo_shading'] = torch.pow(output['pseudo_shading'], 1.0 / self.gamma_for_shading)

        return output

    def render_rays_object_lumen(self, center, ray_unit, pts_light, near, far, outside, app, stratified=False):
        with torch.no_grad():
            dists = self.sample_dists_all(center, ray_unit, near, far, stratified=stratified)  # [B,R,N,3]
        points = camera.get_3D_points_from_dist(center, ray_unit, dists)  # [B,R,N,3]
        sdfs, feats = self.neural_sdf.forward(points)  # [B,R,N,1],[B,R,N,K]
        sdfs[outside[..., None].expand_as(sdfs)] = self.outside_val
        # Compute 1st- and 2nd-order gradients.
        rays_unit = ray_unit[..., None, :].expand_as(points).contiguous()  # [B,R,N,3]
        gradients, hessians = self.neural_sdf.compute_gradients(points, training=self.training, sdf=sdfs)
        normals = torch_F.normalize(gradients, dim=-1)  # [B,R,N,3]
        # Repeat pts_light for each sample points on the ray
        pts_light_expand = pts_light[..., None, :].expand_as(points).contiguous()
        if self.rgb_network_mode == 'r_s':
            o_r, o_s = self.neural_rgb.forward(points, normals, rays_unit, feats, pts_light_expand, app=app)
            rgbs = None
        elif self.rgb_network_mode == 'r_s_re':
            o_r, o_s, o_re = self.neural_rgb.forward(points, normals, rays_unit, feats, pts_light_expand, app=app)
            rgbs = None
        elif self.rgb_network_mode == 'rgb_r':
            rgbs, o_r = self.neural_rgb.forward(points, normals, rays_unit, feats, pts_light_expand, app=app)
        elif self.rgb_network_mode == 'rgb_r_s':
            rgbs, o_r, o_s = self.neural_rgb.forward(points, normals, rays_unit, feats, pts_light_expand, app=app)
        else:
            rgbs = self.neural_rgb.forward(points, normals, rays_unit, feats, pts_light_expand, app=app)  # [B,R,N,3]
        # SDF volume rendering.
        alphas = self.compute_neus_alphas(ray_unit, sdfs, gradients, dists, dist_far=far[..., None],
                                          progress=self.progress)  # [B,R,N]
        if not self.training:
            weights = render.alpha_compositing_weights(alphas)  # [B,R,N,1]
            opacity = render.composite(1., weights)  # [B,R,1]
            gradient = render.composite(gradients, weights)  # [B,R,3]
        else:
            if self.flag_light_visibility:
                weights = render.alpha_compositing_weights(alphas)  # [B,R,N,1]
                gradient = render.composite(gradients, weights)  # [B,R,3]
                opacity = None
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
        if self.rgb_network_mode in ['r_s', 'rgb_r_s']:
            output.update(
                o_r=o_r,
                o_s=o_s,
            )
        elif self.rgb_network_mode == 'r_s_re':
            output.update(
                o_r=o_r,
                o_s=o_s,
                o_re=o_re
            )
        elif self.rgb_network_mode == 'rgb_r':
            output.update(
                o_r=o_r,
            )
        return output

    def render_rays_background_lumen(self, center, ray_unit, pts_light, far, app_outside, stratified=False):
        with torch.no_grad():
            dists = self.sample_dists_background(ray_unit, far, stratified=stratified)
        points = camera.get_3D_points_from_dist(center, ray_unit, dists)  # [B,R,N,3]
        rays_unit = ray_unit[..., None, :].expand_as(points)  # [B,R,N,3]
        # for light
        pts_light_expand = pts_light[..., None, :].expand_as(points)
        rgbs, densities = self.background_nerf.forward(points, rays_unit, pts_light_expand, app_outside)  # [B,R,N,3]
        alphas = render.volume_rendering_alphas_dist(densities, dists)  # [B,R,N]
        # Collect output.
        output = dict(
            rgbs=rgbs,  # [B,R,3]
            dists=dists,  # [B,R,N,1]
            alphas=alphas,  # [B,R,N]
        )
        return output

    def get_param_groups(self, cfg_optim):
        """Allow the network to use different hyperparameters (e.g., learning rate) for different parameters.
        Returns:
            PyTorch parameter group (list or generator). See the PyTorch documentation for details.
        """
        if hasattr(cfg_optim, "partial_training"):
            keyword_list = cfg_optim.partial_training
            params_to_optimize = []
            for name, param in self.named_parameters():
                if any(keyword in name for keyword in keyword_list):
                    params_to_optimize.append(param)
                    print(f"Including parameter: {name}")
                else:
                    print(f"Not including parameter: {name}")
            return params_to_optimize
        else:
            return self.parameters()
