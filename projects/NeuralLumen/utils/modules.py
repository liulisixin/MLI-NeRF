import torch
from functools import partial

from projects.neuralangelo.utils.spherical_harmonics import get_spherical_harmonics
from projects.neuralangelo.utils.misc import get_activation
from projects.nerf.utils import nerf_util


class LumenRGB(torch.nn.Module):

    def __init__(self, cfg_rgb, feat_dim, appear_embed):
        super().__init__()
        self.cfg_rgb = cfg_rgb
        self.cfg_appear_embed = appear_embed
        encoding_view_dim = self.build_encoding(cfg_rgb.encoding_view)
        if hasattr(self.cfg_rgb, 'network_mode') and self.cfg_rgb.network_mode == 'r_s':
            self.network_mode = 'r_s'
            app_dim = appear_embed.dim if appear_embed.enabled else 0
            points_3D_dim, view_enc_dim, normals_dim, feats_dim, pts_light_enc_dim = (3, encoding_view_dim, 3, feat_dim,
                                                                                      encoding_view_dim)
            input_dim = [points_3D_dim + normals_dim + feats_dim + app_dim,
                         points_3D_dim + view_enc_dim + normals_dim + feats_dim + pts_light_enc_dim + app_dim]
            self.build_mlp_r_s(cfg_rgb.mlp, input_dim=input_dim)
        elif hasattr(self.cfg_rgb, 'network_mode') and self.cfg_rgb.network_mode == 'r_s_re':
            self.network_mode = 'r_s_re'
            app_dim = appear_embed.dim if appear_embed.enabled else 0
            points_3D_dim, view_enc_dim, normals_dim, feats_dim, pts_light_enc_dim = (3, encoding_view_dim, 3, feat_dim,
                                                                                      encoding_view_dim)
            input_dim = [points_3D_dim + normals_dim + feats_dim + app_dim,
                         points_3D_dim + normals_dim + feats_dim + pts_light_enc_dim + app_dim,
                         points_3D_dim + view_enc_dim + normals_dim + feats_dim + pts_light_enc_dim + app_dim]
            self.build_mlp_r_s_re(cfg_rgb.mlp, input_dim=input_dim)
        elif hasattr(self.cfg_rgb, 'network_mode') and self.cfg_rgb.network_mode == 'rgb_r':
            self.network_mode = 'rgb_r'
            app_dim = appear_embed.dim if appear_embed.enabled else 0
            points_3D_dim, view_enc_dim, normals_dim, feats_dim, pts_light_enc_dim = (3, encoding_view_dim, 3, feat_dim,
                                                                                      encoding_view_dim)
            input_dim = [points_3D_dim + view_enc_dim + normals_dim + feats_dim + pts_light_enc_dim + app_dim,
                         points_3D_dim + normals_dim + feats_dim + app_dim]
            self.build_mlp_rgb_r(cfg_rgb.mlp, input_dim=input_dim)
        elif hasattr(self.cfg_rgb, 'network_mode') and self.cfg_rgb.network_mode == 'rgb_r_s':
            self.network_mode = 'rgb_r_s'
            app_dim = appear_embed.dim if appear_embed.enabled else 0
            points_3D_dim, view_enc_dim, normals_dim, feats_dim, pts_light_enc_dim = (3, encoding_view_dim, 3, feat_dim,
                                                                                      encoding_view_dim)
            input_dim = [points_3D_dim + view_enc_dim + normals_dim + feats_dim + pts_light_enc_dim + app_dim,
                         points_3D_dim + normals_dim + feats_dim + app_dim,
                         points_3D_dim + normals_dim + feats_dim + pts_light_enc_dim + app_dim]
            self.build_mlp_rgb_r_s(cfg_rgb.mlp, input_dim=input_dim, shading_dim=self.cfg_rgb.shading_dim)
        else:
            self.network_mode = 'rgb'
            input_base_dim = 6 if cfg_rgb.mode == "idr" else 3   # points_3D and normals.
            input_dim = (input_base_dim + encoding_view_dim * 2
                         + feat_dim + (appear_embed.dim if appear_embed.enabled else 0))
            self.build_mlp(cfg_rgb.mlp, input_dim=input_dim)

    def build_encoding(self, cfg_encoding_view):
        if cfg_encoding_view.type == "fourier":
            encoding_view_dim = 6 * cfg_encoding_view.levels
        elif cfg_encoding_view.type == "spherical":
            self.spherical_harmonic_encoding = partial(get_spherical_harmonics, levels=cfg_encoding_view.levels)
            encoding_view_dim = (cfg_encoding_view.levels + 1) ** 2
        else:
            raise NotImplementedError("Unknown encoding type")
        return encoding_view_dim

    def build_mlp(self, cfg_mlp, input_dim=3):
        # RGB prediction
        layer_dims = [input_dim] + [cfg_mlp.hidden_dim] * cfg_mlp.num_layers + [3]
        activ = get_activation(cfg_mlp.activ, **cfg_mlp.activ_params)
        self.mlp = nerf_util.MLPwithSkipConnection(layer_dims, skip_connection=cfg_mlp.skip, activ=activ,
                                                   use_weightnorm=cfg_mlp.weight_norm)

    def build_mlp_rgb_r(self, cfg_mlp, input_dim=None):
        activ = get_activation(cfg_mlp.activ, **cfg_mlp.activ_params)
        mlp_func = partial(nerf_util.MLPwithSkipConnection, skip_connection=cfg_mlp.skip, activ=activ,
                           use_weightnorm=cfg_mlp.weight_norm)
        self.mlp = mlp_func([input_dim[0]] + [cfg_mlp.hidden_dim] * cfg_mlp.num_layers + [3])
        self.mlp_r = mlp_func([input_dim[1]] + [cfg_mlp.hidden_dim] * cfg_mlp.num_layers + [3])

    def build_mlp_rgb_r_s(self, cfg_mlp, input_dim=None, shading_dim=None):
        activ = get_activation(cfg_mlp.activ, **cfg_mlp.activ_params)
        mlp_func = partial(nerf_util.MLPwithSkipConnection, skip_connection=cfg_mlp.skip, activ=activ,
                           use_weightnorm=cfg_mlp.weight_norm)
        self.mlp = mlp_func([input_dim[0]] + [cfg_mlp.hidden_dim] * cfg_mlp.num_layers + [3])
        self.mlp_r = mlp_func([input_dim[1]] + [cfg_mlp.hidden_dim] * cfg_mlp.num_layers + [3])
        self.mlp_s = mlp_func([input_dim[2]] + [cfg_mlp.hidden_dim] * cfg_mlp.num_layers + [shading_dim])

    def build_mlp_r_s(self, cfg_mlp, input_dim=None):
        activ = get_activation(cfg_mlp.activ, **cfg_mlp.activ_params)
        mlp_func = partial(nerf_util.MLPwithSkipConnection, skip_connection=cfg_mlp.skip, activ=activ,
                           use_weightnorm=cfg_mlp.weight_norm)
        self.mlp_r = mlp_func([input_dim[0]] + [cfg_mlp.hidden_dim] * cfg_mlp.num_layers + [3])
        self.mlp_s = mlp_func([input_dim[1]] + [cfg_mlp.hidden_dim] * cfg_mlp.num_layers + [3])
        # self.custom_clamp = CustomClamp(0.0, 1.0)

    def build_mlp_r_s_re(self, cfg_mlp, input_dim=None):
        activ = get_activation(cfg_mlp.activ, **cfg_mlp.activ_params)
        mlp_func = partial(nerf_util.MLPwithSkipConnection, skip_connection=cfg_mlp.skip, activ=activ,
                           use_weightnorm=cfg_mlp.weight_norm)
        self.mlp_r = mlp_func([input_dim[0]] + [cfg_mlp.hidden_dim] * cfg_mlp.num_layers + [3])
        self.mlp_s = mlp_func([input_dim[1]] + [cfg_mlp.hidden_dim] * cfg_mlp.num_layers + [3])
        self.mlp_re = mlp_func([input_dim[2]] + [cfg_mlp.hidden_dim] * cfg_mlp.num_layers + [3])
        # self.custom_clamp = CustomClamp(0.0, 1.0)

    def forward(self, points_3D, normals, rays_unit, feats, pts_light, app):
        view_enc = self.encode_view(rays_unit)  # [...,LD]
        # Here we encode the position of lights as view.
        pts_light_enc = self.encode_view(pts_light)
        if self.network_mode == 'r_s':
            input_r = [points_3D, normals, feats]
            input_s = [points_3D, view_enc, normals, feats, pts_light_enc]
            if app is not None:
                input_r.append(app)
                input_s.append(app)
            input_r_vec = torch.cat(input_r, dim=-1)
            input_s_vec = torch.cat(input_s, dim=-1)
            output_r = self.mlp_r(input_r_vec).sigmoid_()
            output_s = self.mlp_s(input_s_vec)
            return output_r, output_s
        elif self.network_mode == 'r_s_re':
            input_r = [points_3D, normals, feats]
            input_s = [points_3D, normals, feats, pts_light_enc]
            input_re = [points_3D, view_enc, normals, feats, pts_light_enc]
            if app is not None:
                input_r.append(app)
                input_s.append(app)
                input_re.append(app)
            input_r_vec = torch.cat(input_r, dim=-1)
            input_s_vec = torch.cat(input_s, dim=-1)
            input_re_vec = torch.cat(input_re, dim=-1)
            output_r = self.mlp_r(input_r_vec).sigmoid_()
            output_s = self.mlp_s(input_s_vec).sigmoid_()
            output_re = self.mlp_re(input_re_vec).sigmoid_()
            return output_r, output_s, output_re
        elif self.network_mode == 'rgb_r':
            input_rgb = [points_3D, view_enc, normals, feats, pts_light_enc]
            input_r = [points_3D, normals, feats]
            if app is not None:
                input_rgb.append(app)
                input_r.append(app)
            input_rgb_vec = torch.cat(input_rgb, dim=-1)
            input_r_vec = torch.cat(input_r, dim=-1)

            rgb = self.mlp(input_rgb_vec).sigmoid_()
            output_r = self.mlp_r(input_r_vec).sigmoid_()
            return rgb, output_r
        elif self.network_mode == 'rgb_r_s':
            input_rgb = [points_3D, view_enc, normals, feats, pts_light_enc]
            input_r = [points_3D, normals, feats]
            input_s = [points_3D, normals, feats, pts_light_enc]
            if app is not None:
                input_rgb.append(app)
                input_r.append(app)
                input_s.append(app)
            input_rgb_vec = torch.cat(input_rgb, dim=-1)
            input_r_vec = torch.cat(input_r, dim=-1)
            input_s_vec = torch.cat(input_s, dim=-1)

            rgb = self.mlp(input_rgb_vec).sigmoid_()
            output_r = self.mlp_r(input_r_vec).sigmoid_()
            output_s = self.mlp_s(input_s_vec).sigmoid_()
            return rgb, output_r, output_s
        else:
            input_list = [points_3D, view_enc, normals, feats, pts_light_enc]
            if app is not None:
                input_list.append(app)
            if self.cfg_rgb.mode == "no_view_dir":
                input_list.remove(view_enc)
            if self.cfg_rgb.mode == "no_normal":
                input_list.remove(normals)
            input_vec = torch.cat(input_list, dim=-1)
            rgb = self.mlp(input_vec).sigmoid_()
            return rgb  # [...,3]

    def encode_view(self, rays_unit):
        if self.cfg_rgb.encoding_view.type == "fourier":
            view_enc = nerf_util.positional_encoding(rays_unit, num_freq_bases=self.cfg_rgb.encoding_view.levels)
        elif self.cfg_rgb.encoding_view.type == "spherical":
            view_enc = self.spherical_harmonic_encoding(rays_unit)
        else:
            raise NotImplementedError("Unknown encoding type")
        return view_enc

class LumenBackgroundNeRF(torch.nn.Module):

    def __init__(self, cfg_background, appear_embed):
        super().__init__()
        self.cfg_background = cfg_background
        self.cfg_appear_embed = appear_embed
        encoding_dim, encoding_view_dim = self.build_encoding(cfg_background.encoding, cfg_background.encoding_view)
        input_dim = 4 + encoding_dim
        input_view_dim = cfg_background.mlp.hidden_dim + encoding_view_dim * 2 + \
            (appear_embed.dim if appear_embed.enabled else 0)
        self.build_mlp(cfg_background.mlp, input_dim=input_dim, input_view_dim=input_view_dim)

    def build_encoding(self, cfg_encoding, cfg_encoding_view):
        # Positional encoding.
        if cfg_encoding.type == "fourier":
            encoding_dim = 8 * cfg_encoding.levels
        else:
            raise NotImplementedError("Unknown encoding type")
        # View encoding.
        if cfg_encoding_view.type == "fourier":
            encoding_view_dim = 6 * cfg_encoding_view.levels
        elif cfg_encoding_view.type == "spherical":
            self.spherical_harmonic_encoding = partial(get_spherical_harmonics, levels=cfg_encoding_view.levels)
            encoding_view_dim = (cfg_encoding_view.levels + 1) ** 2
        else:
            raise NotImplementedError("Unknown encoding type")
        return encoding_dim, encoding_view_dim

    def build_mlp(self, cfg_mlp, input_dim=3, input_view_dim=3):
        activ = get_activation(cfg_mlp.activ, **cfg_mlp.activ_params)
        # Point-wise feature.
        layer_dims = [input_dim] + [cfg_mlp.hidden_dim] * (cfg_mlp.num_layers - 1) + [cfg_mlp.hidden_dim + 1]
        self.mlp_feat = nerf_util.MLPwithSkipConnection(layer_dims, skip_connection=cfg_mlp.skip, activ=activ)
        self.activ_density = get_activation(cfg_mlp.activ_density, **cfg_mlp.activ_density_params)
        # RGB prediction.
        layer_dims_rgb = [input_view_dim] + [cfg_mlp.hidden_dim_rgb] * (cfg_mlp.num_layers_rgb - 1) + [3]
        self.mlp_rgb = nerf_util.MLPwithSkipConnection(layer_dims_rgb, skip_connection=cfg_mlp.skip_rgb, activ=activ)

    def forward(self, points_3D, rays_unit, pts_light, app_outside):
        points_enc = self.encode(points_3D)  # [...,4+LD]
        # Volume density prediction.
        out = self.mlp_feat(points_enc)
        density, feat = self.activ_density(out[..., 0]), self.mlp_feat.activ(out[..., 1:])  # [...],[...,K]
        # RGB color prediction.
        if self.cfg_background.view_dep:
            view_enc = self.encode_view(rays_unit)  # [...,LD]
            pts_light_enc = self.encode_view(pts_light)
            input_list = [feat, view_enc, pts_light_enc]
            if app_outside is not None:
                input_list.append(app_outside)
            input_vec = torch.cat(input_list, dim=-1)
            rgb = self.mlp_rgb(input_vec).sigmoid_()  # [...,3]
        else:
            raise NotImplementedError
        return rgb, density

    def encode(self, points_3D):
        # Reparametrize the 3D points.
        # TODO: revive this.
        if True:
            points_3D_norm = points_3D.norm(dim=-1, keepdim=True)  # [B,R,N,1]
            points = torch.cat([points_3D / points_3D_norm, 1.0 / points_3D_norm], dim=-1)  # [B,R,N,4]
        else:
            points = points_3D
        # Positional encoding.
        if self.cfg_background.encoding.type == "fourier":
            points_enc = nerf_util.positional_encoding(points, num_freq_bases=self.cfg_background.encoding.levels)
        else:
            raise NotImplementedError("Unknown encoding type")
        # TODO: 1/x?
        points_enc = torch.cat([points, points_enc], dim=-1)  # [B,R,N,4+LD]
        return points_enc

    def encode_view(self, rays_unit):
        if self.cfg_background.encoding_view.type == "fourier":
            view_enc = nerf_util.positional_encoding(rays_unit, num_freq_bases=self.cfg_background.encoding_view.levels)
        elif self.cfg_background.encoding_view.type == "spherical":
            view_enc = self.spherical_harmonic_encoding(rays_unit)
        else:
            raise NotImplementedError("Unknown encoding type")
        return view_enc


class ClampWithGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, min_value, max_value):
        ctx.save_for_backward(x)
        ctx.min_value = min_value
        ctx.max_value = max_value
        return x.clamp(min_value, max_value)

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[(x < ctx.min_value) | (x > ctx.max_value)] = 1
        return grad_input, None, None


class CustomClamp(torch.nn.Module):
    """
    CustomClamp is a custom module that applies element-wise clamping to the input tensor.
    It ensures that the output values are within a specified range [min_value, max_value],
    while maintaining gradients outside the clamping range for learning purposes.
    Attributes:
        min_value (float): The lower bound of the clamping range.
        max_value (float): The upper bound of the clamping range.
    """
    def __init__(self, min_value, max_value):
        super(CustomClamp, self).__init__()
        self.min_value = min_value
        self.max_value = max_value

    def forward(self, x):
        return ClampWithGradient.apply(x, self.min_value, self.max_value)

