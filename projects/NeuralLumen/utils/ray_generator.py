import torch
from projects.nerf.utils import camera
from projects.NeuralLumen.utils.utils import get_center


def ray_generator(pose, intr, pose_light, image_size, num_rays, full_image=False, camera_ndc=False,
                  ray_indices=None):
    """Yield sampled rays for coordinate-based model to predict NeRF.
    Args:
        pose (tensor [bs,3,4]): Camera poses ([R,t]).
        intr (tensor [bs,3,3]): Camera intrinsics.
        image_size: (tensor [bs,2]): Image size [height, width].
        num_rays (int): Number of rays to sample (random rays unless full_image=True).
        full_image (bool): Sample rays from the full image.
        camera_ndc (bool): Use normalized device coordinate for camera.
    Returns:
        center_slice (tensor [bs, ray, 3]): Sampled 3-D center in the world coordinate.
        ray_slice (tensor [bs, ray, 3]): Sampled 3-D ray in the world coordinate.
        ray_idx (tensor [bs, ray]): Sampled indices to index sampled pixels on images.
    """
    # Create a grid of centers and rays on an image.
    batch_size = pose.shape[0]
    # We used to randomly sample ray indices here. Now, we assume they are pre-generated and passed in.
    if ray_indices is None:
        num_pixels = image_size[0] * image_size[1]
        if full_image:
            # Sample rays from the full image.
            ray_indices = torch.arange(0, num_pixels, device=pose.device).repeat(batch_size, 1)  # [B,HW]
        else:
            # Sample rays randomly. The below is equivalent to batched torch.randperm().
            ray_indices = torch.rand(batch_size, num_pixels, device=pose.device).argsort(dim=1)[:, :num_rays]  # [B,R]
    center, ray = camera.get_center_and_ray(pose, intr, image_size)  # [B,HW,3]
    # for light
    center_light = get_center(pose_light, image_size)
    # Convert center/ray representations to NDC if necessary.
    if camera_ndc == "new":
        center, ray = camera.convert_NDC2(center, ray, intr=intr)
    elif camera_ndc:
        center, ray = camera.convert_NDC(center, ray, intr=intr)
    # Yield num_rays of sampled rays in each iteration (when random, the loop will only iterate once).
    for c in range(0, ray_indices.shape[1], num_rays):
        ray_idx = ray_indices[:, c:c + num_rays]  # [B,R]
        batch_idx = torch.arange(batch_size, device=pose.device).repeat(ray_idx.shape[1], 1).t()  # [B,R]
        center_slice = center[batch_idx, ray_idx]  # [B,R,3]
        ray_slice = ray[batch_idx, ray_idx]  # [B,R,3]
        center_light_slice = center_light[batch_idx, ray_idx]
        yield center_slice, ray_slice, center_light_slice, ray_idx

