import torch
from projects.nerf.utils.camera import cam2world
from typing import Tuple
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial.transform import Slerp
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import random


def interpolate_pose(pose1, pose2, ratio):
    # transform to the world first, then do interpolation.
    # Note: this pose is already in w2c
    flag_tensor = False
    if torch.is_tensor(pose1):
        flag_tensor = True
        pose1 = pose1.numpy()
        pose2 = pose2.numpy()
    rot_0 = pose1[:3, :3]
    rot_1 = pose2[:3, :3]
    rots = Rot.from_matrix(np.stack([rot_0, rot_1]))
    key_times = [0, 1]
    slerp = Slerp(key_times, rots)
    rot = slerp(ratio)
    pose = np.diag([1.0, 1.0, 1.0, 1.0])
    pose = pose.astype(np.float32)
    pose[:3, :3] = rot.as_matrix()
    pose[:3, 3] = ((1.0 - ratio) * pose1 + ratio * pose2)[:3, 3]
    pose = pose[:3]
    if flag_tensor:
        pose = torch.from_numpy(pose)
    return pose


def img_to_np(tensor, add_text=True, text=' '):
    img_np = (tensor.detach().cpu().numpy().transpose([1, 2, 0]) * 256).clip(0, 255).astype(np.uint8)
    if img_np.shape[2] == 1:
        img_np = np.repeat(img_np, 3, axis=2)
    if add_text:
        text_bg_height = int(img_np.shape[0] / 10)  # pixel height
        # Create a white background area
        text_bg = np.ones((text_bg_height, img_np.shape[1], 3), dtype=np.uint8) * 255
        # Vertically stack the original image and the white background area
        img_with_text_bg = np.vstack((img_np, text_bg))
        # Set the position of the text (bottom left), ensuring the text is within the white background area
        text_x = 10  # Distance from the image's left edge
        text_y = img_with_text_bg.shape[
                     0] - 10  # Distance from the image's bottom edge, ensuring it is within the white area
        # Set the font, scale, color, and line type
        font = cv.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        color = (0, 0, 0)  # Black text
        line_type = cv.LINE_AA
        # Add the text to the white background area
        cv.putText(img_with_text_bg, text, (text_x, text_y), font, font_scale, color, 1, line_type)
        img_np = img_with_text_bg
    return cv.cvtColor(img_np, cv.COLOR_RGB2BGR)


def get_center(pose, image_size):
    """
    Args:
        pose (tensor [3,4]/[B,3,4]): Camera pose.
        image_size (list of int): Image size.
    Returns:
        center_3D (tensor [HW,3]/[B,HW,3]): Center of the pose.
    """
    H, W = image_size
    # Given the intrinsic/extrinsic matrices, get the camera center and ray directions.
    with torch.no_grad():
        center_3D = torch.zeros(H * W, 3, device=pose.device)
    # Compute center and ray.
    if len(pose.shape) == 3:
        batch_size = len(pose)
        center_3D = center_3D.repeat(batch_size, 1, 1)  # [B,HW,2]
    # Transform from camera to world coordinates.
    center_3D = cam2world(center_3D, pose)  # [HW,3]/[B,HW,3]
    return center_3D


"""
Borrowed from
https://docs.nerf.studio/_modules/nerfstudio/utils/math.html#intersect_aabb
"""
def intersect_aabb(
    origins: torch.Tensor,
    directions: torch.Tensor,
    aabb: torch.Tensor,
    max_bound: float = 1e10,
    invalid_value: float = 1e10,
) -> Tuple[torch.Tensor, torch.Tensor, torch.BoolTensor]:
    """
    Implementation of ray intersection with AABB box

    Args:
        origins: [B,N,3] or [N,3] tensor of 3d positions
        directions: [B,N,3] or [N,3] tensor of normalized directions
        aabb: [6] array of aabb box in the form of [x_min, y_min, z_min, x_max, y_max, z_max]
        max_bound: Maximum value of t_max
        invalid_value: Value to return in case of no intersection

    Returns:
        t_min, t_max - two tensors of shapes [(B), N, 1] representing distance of intersection from the origin.
    """

    tx_min = (aabb[:3] - origins) / directions
    tx_max = (aabb[3:] - origins) / directions

    t_min = torch.stack((tx_min, tx_max)).amin(dim=0)
    t_max = torch.stack((tx_min, tx_max)).amax(dim=0)

    t_min = t_min.amax(dim=-1, keepdim=True)
    t_max = t_max.amin(dim=-1, keepdim=True)

    t_min = torch.clamp(t_min, min=0, max=max_bound)
    t_max = torch.clamp(t_max, min=0, max=max_bound)

    cond = t_max <= t_min
    # t_min = torch.where(cond, invalid_value, t_min)
    # t_max = torch.where(cond, invalid_value, t_max)

    return t_min, t_max, cond


def weighted_shading_loss(predicted_shading, pseudo_shading, weight_range=(0.0, 1.0)):
    threshold_high_probability = 0.5   # the angle between light and normal < 60 deg
    weight = pseudo_shading / threshold_high_probability
    weight = torch.clamp(weight, 0.0, 1.0)
    weight = torch.pow(weight, 2) * (weight_range[1] - weight_range[0]) + weight_range[0]    # give the dark area a small weight
    # Note: detach the weight
    weight = weight.detach()

    # here may contain the operation of broadcasting
    abs_diff = torch.abs(predicted_shading - pseudo_shading)
    weighted_abs_diff = abs_diff * weight
    loss = weighted_abs_diff.mean() / (weight.mean() + 1e-6)

    return loss


def intrinsic_loss(output_ref, output_sha, pseudo_ref, pseudo_sha, pseudo_visibility_certainty,
                   weight_map_range_shading=(0.25, 1.0), weight_map_range_visibility=(0.25, 1.0),
                   factor_ref=1.0, factor_sha=1.0):
    def normalize(x, range_a, range_b):
        x_min = x.min()
        x_max = x.max()
        normalized = range_a + (x - x_min) / torch.clamp((x_max - x_min), min=1e-6) * (range_b - range_a)
        return normalized
    # weight map for shading
    weight_map_sha = pseudo_sha.detach()
    weight_map_sha = normalize(weight_map_sha, weight_map_range_shading[0], weight_map_range_shading[1])
    # weight map for reflectance
    weight_map_vis = pseudo_visibility_certainty.detach()
    weight_map_vis = normalize(weight_map_vis, weight_map_range_visibility[0], weight_map_range_visibility[1])
    weight_map_ref = torch.minimum(weight_map_vis, weight_map_sha)

    distance_l1_ref = torch.mean(torch.abs(output_ref - pseudo_ref) * weight_map_ref)
    distance_l1_sha = torch.mean(torch.abs(output_sha - pseudo_sha) * weight_map_sha)

    loss = distance_l1_ref * factor_ref + distance_l1_sha * factor_sha
    return loss


def regularize_re_loss(output_re, factor_negative=10.0, factor_positive=1.0, exponent_positive=1.0):
    # first, constrain the residual > 0.0
    part_negative = torch.where(output_re < 0.0, output_re, torch.tensor(0.0, device=output_re.device))
    regularization_term_negative = torch.mean(torch.abs(part_negative))
    # second, constrain the area of residual > 0.0 to be small
    part_positive = torch.where(output_re >= 0.0, output_re, torch.tensor(0.0, device=output_re.device))
    regularization_term_high_value = torch.mean(torch.pow(part_positive, exponent_positive))

    loss = regularization_term_negative * factor_negative + regularization_term_high_value * factor_positive
    return loss


def create_collage(frame_imgs, padding=5):
    """
    Create a collage from a list of images.
    """
    # Assuming all images have the same size
    img_height, img_width, _ = frame_imgs[0].shape
    rows = int(np.sqrt(len(frame_imgs)))
    cols = int(np.ceil(len(frame_imgs) / rows))

    # Create a white background
    collage_height = img_height * rows
    collage_width = img_width * cols + padding * (cols - 1)
    collage = np.ones((collage_height, collage_width, 3), dtype=np.uint8) * 255

    # Place images onto the collage
    for idx, img in enumerate(frame_imgs):
        row_idx = idx // cols
        col_idx = idx % cols
        y_start = row_idx * img_height
        x_start = col_idx * (img_width + padding)
        collage[y_start:y_start + img_height, x_start:x_start + img_width, :] = img

    return collage


def plot_line(x, y):
    """
    Plot a line graph of y versus x using Tensor vectors.

    Parameters:
    - x: A 1D Tensor representing the values on the x-axis.
    - y: A 1D Tensor representing the values on the y-axis.

    Returns:
    None
    """
    # Ensure x and y are 1D and have the same length
    assert x.ndim == 1 and y.ndim == 1, "x and y must be 1D Tensors."
    assert len(x) == len(y), "x and y must have the same length."

    # Convert Tensors to NumPy arrays
    x_np = x.cpu().numpy()
    y_np = y.cpu().numpy()

    # Plot the line graph
    plt.plot(x_np, y_np, marker='o')  # 'marker='o'' is optional, marks each data point with a circle
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Line plot of x vs. y')
    plt.grid(True)  # Display grid
    plt.show()


def get_random_other_index(num_indexes, length_selected, seed=0):
    # Set the seed for reproducibility
    random.seed(seed)

    # Initialize the list to store all lists
    index_lists = []

    # Generate the lists
    for i in range(num_indexes):
        # Start with the current index
        current_list = [i]

        # Generate a list of other indexes excluding the current one
        other_indexes = list(range(num_indexes))
        other_indexes.remove(i)

        # Randomly select 9 other indexes and add to the current list
        current_list.extend(random.sample(other_indexes, length_selected-1))

        # Add the current list to the main list
        index_lists.append(current_list)

    return index_lists


if __name__ == "__main__":
    center_test = torch.tensor([0, 2, 0])[None, None, ...]
    ray_unit_test = torch.tensor([1, 0, 0])[None, None, ...]
    bounding_box_aabb = torch.tensor([-0.55, -0.43, -0.15, 0.55, 0.35, 0.25])
    intersect_aabb(center_test, ray_unit_test, aabb=bounding_box_aabb)
