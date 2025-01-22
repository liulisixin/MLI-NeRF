import sys
import os
sys.path.append(os.getcwd())
from functools import partial
from tqdm import tqdm
import torch
import torchvision.transforms.functional as torchvision_F
import torch.nn.functional as F
from imaginaire.utils.visualization import preprocess_image
from torch_kmeans import KMeans
import math
import numpy as np
from scipy.spatial import KDTree, distance
import argparse
import time
import concurrent.futures


def save_image_path(image, name, save_path, from_range=(0, 1)):
    if len(image.size()) > 3:
        image = image.squeeze(0)
    image = preprocess_image(image, from_range=from_range)
    pil_image = torchvision_F.to_pil_image(image)
    pil_image.save(os.path.join(save_path, name + '.png'))


def erosion(input, kernel_size):
    padding = kernel_size // 2
    kernel = torch.ones((1, 1, kernel_size, kernel_size), dtype=input.dtype, device=input.device)
    input = F.pad(input, (padding, padding, padding, padding), mode='replicate')
    output = F.conv2d(input, kernel, padding=0, stride=1)
    output = torch.where(output == kernel.sum(), 1.0, 0.0)

    return output


def dilation(input, kernel_size):
    output = 1 - erosion(1 - input, kernel_size)
    return output


def edge(input, kernel_size):
    output = dilation(input, kernel_size) - erosion(input, kernel_size)
    return output


def edge_weight(input, step):
    weight = torch.zeros_like(input, device=input.device)
    for i in range(1, step + 1):
        weight = weight + edge(input, 2 * i + 1)
    if weight.max() > 0.0:
        weight = weight / weight.max()
    weight = 1 - weight
    return weight


def find_best_ref(mask_shading, kmeans_label, kmeans_num_clusters, pseudo_shadings, shading_threshold_wrt_max, ref):
    # mask the shading = 0
    mask_shading = mask_shading.squeeze(1)
    kmeans_label[~mask_shading] = kmeans_num_clusters

    # find the label with the highest probability.
    num_label = torch.zeros(kmeans_num_clusters, kmeans_label.size(1), kmeans_label.size(2), dtype=torch.long)
    for i in range(kmeans_num_clusters):
        num_label[i] = (kmeans_label == i).sum(dim=0)
    max_indices = num_label == num_label.max(dim=0)[0]
    false_row = torch.zeros(1, max_indices.size(1), max_indices.size(2), dtype=torch.bool)
    max_indices = torch.cat((max_indices, false_row), dim=0)
    gathered = torch.gather(max_indices, 0, kmeans_label)

    pseudo_shadings_copy = pseudo_shadings.clone().squeeze(1)
    pseudo_shadings_max = (pseudo_shadings_copy * gathered.float()).max(dim=0)[0]
    pseudo_shadings_max = pseudo_shadings_max.unsqueeze(0)
    mask_high_shading = pseudo_shadings_copy > shading_threshold_wrt_max * pseudo_shadings_max

    final_mask_ref = torch.logical_and(gathered, mask_high_shading)
    expanded_mask = final_mask_ref.unsqueeze(1).expand_as(ref)
    masked_ref = torch.where(expanded_mask, ref, torch.tensor(0.0, device=ref.device))
    summed_ref = masked_ref.sum(dim=0)
    true_counts = expanded_mask.sum(dim=0)
    true_counts = true_counts.clamp(min=1)
    average_ref = summed_ref / true_counts
    return average_ref


def rgb2opp(imgs):
    r = imgs[..., 0]
    g = imgs[..., 1]
    b = imgs[..., 2]
    o1 = (r - g) / math.sqrt(2)
    o2 = (r + g - 2.0 * b) / math.sqrt(6)
    opp = torch.stack([o1, o2], dim=-1)
    return opp


def kmeans_cluster(imgs, kmeans_num_clusters):
    img_size = imgs.size()[-2:]
    imgs = imgs.reshape(imgs.size(0), imgs.size(1), -1)
    imgs = imgs.permute(2, 0, 1)
    # only consider the chromaticity, so transform to opponent color space.
    imgs = rgb2opp(imgs)
    # run kmeans clustering
    imgs = imgs.cuda()
    if kmeans_num_clusters > 1:
        model = KMeans(n_clusters=kmeans_num_clusters)
        kmeans_result = model(imgs)

        kmeans_label = kmeans_result.labels
        kmeans_center = kmeans_result.centers
    else:
        kmeans_label = torch.zeros(imgs.size(0), imgs.size(1), dtype=torch.int64)
        kmeans_center = imgs.mean(dim=1, keepdim=True)

    kmeans_label = kmeans_label.permute(1, 0)
    kmeans_label = kmeans_label.reshape(kmeans_label.size(0), img_size[0], img_size[1])
    kmeans_center = kmeans_center.permute(1, 2, 0)
    kmeans_center = kmeans_center.reshape(kmeans_center.size(0), kmeans_center.size(1), img_size[0], img_size[1])

    kmeans_label = kmeans_label.detach().cpu()
    kmeans_center = kmeans_center.detach().cpu()

    return kmeans_label, kmeans_center


def d_angle(v1, v2):
    v1_normalized = v1 / np.linalg.norm(v1)
    v2_normalized = v2 / np.linalg.norm(v2)
    dot_product = np.dot(v1_normalized, v2_normalized)
    dot_product = np.clip(dot_product, -1.0, 1.0)
    angle_radians = np.arccos(dot_product)
    return angle_radians

def d_min_mean(imgs_hole, imgs_nonhole):
    dist_matrix = distance.cdist(imgs_hole, imgs_nonhole, 'euclidean')
    # Find the minimum distance for each vector in imgs_hole
    min_distances = np.min(dist_matrix, axis=1)
    # Calculate the average of these minimum distances
    average_distance = np.mean(min_distances)
    return average_distance

# Custom distance function
def custom_distance(normal_hole, imgs_hole, normal_nonhole, imgs_nonhole, dist_normalized):
    # first, normal distance
    d_normal = d_angle(normal_hole, normal_nonhole)
    # second, group color distance
    d_color = d_min_mean(imgs_hole, imgs_nonhole)
    # normalized coordinate distance
    d_coordinate = dist_normalized
    return d_normal + d_color + d_coordinate


def compute_best_color(hole_pos, ref, normal, imgs_feature, non_hole_positions, kdtree, img_size_mean, search_points):
    # hole information
    normal_hole = normal[:, hole_pos[0], hole_pos[1]]
    feature_hole = imgs_feature[:, :, hole_pos[0], hole_pos[1]]
    # Use KDTree to find the nearest N non-hole pixels based on position
    dists, indices = kdtree.query([hole_pos], k=search_points)  # Example searches for the nearest 10 points
    best_distance = float('inf')
    best_color = None
    for dist, index in zip(dists[0], indices[0]):
        dist_normalized = dist / img_size_mean
        non_hole_pos = non_hole_positions[index]
        non_hole_color = ref[:, non_hole_pos[0], non_hole_pos[1]]
        # Apply custom distance function
        normal_nonhole = normal[:, non_hole_pos[0], non_hole_pos[1]]
        feature_nonhole = imgs_feature[:, :, non_hole_pos[0], non_hole_pos[1]]
        custom_dist = custom_distance(normal_hole, feature_hole, normal_nonhole, feature_nonhole, dist_normalized)
        if custom_dist < best_distance:
            best_distance = custom_dist
            best_color = non_hole_color
    # Fill the hole
    return hole_pos, best_color

def fill_holes(ref, normal, imgs_feature, mask, search_points):
    # work in numpy
    ref = ref.numpy()
    normal = normal.numpy()
    imgs_feature = imgs_feature.numpy()
    mask = mask.numpy()

    hole_positions = np.column_stack(np.where(~mask))
    non_hole_positions = np.column_stack(np.where(mask))
    # Build KDTree from non-hole pixels
    kdtree = KDTree(non_hole_positions)
    filled_image = ref.copy()
    img_size = filled_image.shape
    img_size_mean = 0.5 * (img_size[-2] + img_size[-1])

    num_workers = 1
    if num_workers > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Wrap hole_positions with tqdm to display the progress bar
            results = list(tqdm(executor.map(
                lambda hole_pos: compute_best_color(hole_pos, ref, normal, imgs_feature, non_hole_positions, kdtree,
                                                    img_size_mean, search_points),
                hole_positions), total=len(hole_positions), desc="Filling holes"))
    else:
        results = []
        for hole_pos in tqdm(hole_positions, total=len(hole_positions), desc="Filling holes"):
            best_color = compute_best_color(hole_pos, ref, normal, imgs_feature, non_hole_positions, kdtree, img_size_mean,
                                            search_points)
            results.append((hole_pos, best_color))

    for hole_pos, best_color in results:
        filled_image[:, hole_pos[0], hole_pos[1]] = best_color

    return torch.from_numpy(filled_image)


def fill_holes_kd(ref_tensor, normal_tensor, color_feature_tensor, mask_tensor, search_points):
    # work in numpy
    ref = ref_tensor.numpy()
    normal = normal_tensor.numpy()
    # normalize normal
    normal = normal / (np.linalg.norm(normal, axis=0) + 1e-10)[None, :, :]
    color_feature = color_feature_tensor.numpy()
    mask = mask_tensor.numpy()

    # here we need to merge all features including
    weight_position = 4.0
    weight_normal = 1.0
    weight_color = 1.0

    height = ref.shape[1]
    width = ref.shape[2]
    x = np.arange(width)
    y = np.arange(height)
    X, Y = np.meshgrid(x, y)
    positions = np.stack([Y, X], axis=0)
    positions_norm = positions / positions.max() * weight_position

    normal = normal * weight_normal

    color_feature = color_feature * weight_color

    """
    Here we need to process different color features. The distance in color features is the min of the distance of all 
    possible colors between two pixels.
    closest point = argmin(pos, normal, min(c_i, c_j))
    positions_norm.shape: (feature_length, H, W) (2, 270, 360)
    normal.shape: (feature_length, H, W) (3, 270, 360)
    color_feature.shape: (optional length, feature_length, H, W) (3, 2, 270, 360)
    """
    def process_repeat(x_np):
        return x_np[None, ...].repeat(color_feature.shape[0], axis=0)
    positions = process_repeat(positions)
    positions_norm = process_repeat(positions_norm)
    normal = process_repeat(normal)

    all_feature = np.concatenate([positions_norm, normal, color_feature], axis=1)

    # reshape
    positions = np.transpose(positions, axes=(0, 2, 3, 1))   # (optional length, H, W, 2)
    all_feature = np.transpose(all_feature, axes=(0, 2, 3, 1))   # (optional length, H, W, feature_length)
    # positions = positions.reshape(-1, 2)
    # all_feature = all_feature.reshape(-1, 7)
    # mask = mask.reshape(-1)

    hole_feature = all_feature[:, ~mask, :]   # (optional length, H*W, feature_length)
    non_hole_feature = all_feature[:, mask, :]
    hole_positions = positions[:, ~mask, :]
    non_hole_positions = positions[:, mask, :]

    non_hole_feature = non_hole_feature.reshape(-1, non_hole_feature.shape[-1])
    non_hole_positions = non_hole_positions.reshape(-1, non_hole_positions.shape[-1])

    # Build KDTree from non-hole pixels
    kdtree = KDTree(non_hole_feature)
    distance, kd_index = kdtree.query(hole_feature)

    # continue to find the minimum distance
    min_indices = np.argmin(distance, axis=0)
    column_indices = np.arange(kd_index.shape[1])
    # Use min_indices for row indices and column_indices for column indices to extract values from kd_index
    selected_values = kd_index[min_indices, column_indices]

    closest_indices = non_hole_positions[selected_values]

    hole_positions = hole_positions[0]
    ref[:, hole_positions[:, 0], hole_positions[:, 1]] = ref[:, closest_indices[:, 0], closest_indices[:, 1]]

    return torch.from_numpy(ref)



if __name__ == "__main__":
    start_time = time.time()
    parser = argparse.ArgumentParser(description='pseudo label')
    parser.add_argument('--workdir', help='Dir for the working path.', default=None)
    parser.add_argument('--setting', choices=['pair', 'unpair', 'single_light'], help='Choose between "pair" or "unpair" or "single_light".')
    args = parser.parse_args()
    work_path = args.workdir
    print("Begin processing in: ", work_path)
    results_all = torch.load(os.path.join(work_path, 'results_all.pt'))
    output_path = work_path + '_pseudo_label'
    os.makedirs(output_path, exist_ok=True)

    # parameters
    if args.setting == 'unpair':
        para = {
            "kernel_erosion_visibility": 7,
            "edge_step_visibility_certainty": 7,
            "kmeans_num_clusters": 2,
            "shading_threshold": 0.0,
            "shading_threshold_wrt_max": 0.6,
            "gamma_correlation_factor": 2.2,
            "fill_search_points": 10,
        }
    elif args.setting == 'pair':
        para = {
            "kernel_erosion_visibility": 7,
            "edge_step_visibility_certainty": 7,
            "kmeans_num_clusters": 3,
            "shading_threshold": 0.0,
            "shading_threshold_wrt_max": 0.6,
            "gamma_correlation_factor": 2.2,
            "fill_search_points": 1000,
        }
    elif args.setting == 'single_light':
        para = {
            "kernel_erosion_visibility": 3,
            "edge_step_visibility_certainty": 7,
            "kmeans_num_clusters": 1,
            "shading_threshold": 0.0,
            "shading_threshold_wrt_max": 0.6,
            "gamma_correlation_factor": 2.2,
            "fill_search_points": 1000,
        }
    else:
        raise NotImplementedError
    # save the parameter
    with open(os.path.join(output_path, 'parameters.txt'), "w") as file:
        for name in para.keys():
            file.write(f"{name}: {para[name]}\n")

    # # cam
    # n_cams = 50
    # test_cams = [4, 8, 15]
    # val_cams = [25, 42, 47]
    # train_cams = [i for i in range(n_cams) if i not in test_cams + val_cams]
    # # light
    # n_lights = 40
    # test_lights = [2, 21, 34]
    # train_lights = [i for i in range(n_lights) if i not in test_lights]

    pseudo_label_all = {}
    for camera_index in tqdm(results_all.keys()):
        pseudo_label_all[str(camera_index)] = {}
        print("camera_index", camera_index)
        img_save_path = os.path.join(output_path, str(camera_index))
        os.makedirs(img_save_path, exist_ok=True)
        save_image = partial(save_image_path, save_path=img_save_path)
        data_list = {}
        for light_index in results_all[camera_index].keys():
            pseudo_label_all[str(camera_index)][str(light_index)] = {}
            data = results_all[str(camera_index)][str(light_index)]
            for key in data.keys():
                data[key] = data[key].squeeze(0)
            visibility_erosion = erosion(data['visibility'], kernel_size=para['kernel_erosion_visibility'])
            data['pseudo_shading'] = data['normal_x_light'] * visibility_erosion
            if args.setting == 'unpair':
                data['pseudo_shading'] = data['pseudo_shading'] * data['inter_mask']
            data_list[light_index] = data
            # get and save the visibility_certainty
            visibility_certainty = edge_weight(data['visibility'], para['edge_step_visibility_certainty'])
            pseudo_label_all[str(camera_index)][str(light_index)]['visibility_certainty'] = visibility_certainty
            save_image(visibility_certainty, f"{str(camera_index)}_{str(light_index)}_visibility_certainty")
            # get and save the shading
            s_gamma = torch.pow(data['pseudo_shading'], 1 / para['gamma_correlation_factor'])
            pseudo_label_all[str(camera_index)][str(light_index)]['pseudo_shading_gamma'] = s_gamma
            save_image(s_gamma, f"{str(camera_index)}_{str(light_index)}_pseudo_shading_gamma")
        if all(['rgb_target' in data_list[key].keys() for key in data_list]):
            use_key = 'rgb_target'
        else:
            use_key = 'rgb_render'
        image_list = [data_list[key][use_key] for key in data_list]
        imgs = torch.stack(image_list)

        # Step 1: kmeans clustering
        print("Step 1: kmeans clustering along pixels")
        kmeans_label, kmeans_center = kmeans_cluster(imgs.clone(), para['kmeans_num_clusters'])
        torch.cuda.empty_cache()

        # Step 2: generate the pseudo ref where shading > 0
        print("Step 2: generate the pseudo ref where shading > ", para['shading_threshold'])
        pseudo_shading_list = [data_list[key]['pseudo_shading'] for key in data_list]
        pseudo_shadings = torch.stack(pseudo_shading_list)
        # This is a mask that divides the image into 2 parts: part 1 obtains the reflectance from the shading,
        # and part 2 get the reflectance from the surroundings. The mask is obtained from linear shading.
        mask_shading = pseudo_shadings > para['shading_threshold']
        # gamma correlation
        pseudo_shadings_gamma = torch.pow(pseudo_shadings, 1 / para['gamma_correlation_factor'])
        ref = imgs / pseudo_shadings_gamma
        for i in range(ref.size(0)):
            save_image(ref[i], f'{str(camera_index)}_{str(i)}_ref')

        average_ref = find_best_ref(mask_shading, kmeans_label, para['kmeans_num_clusters'], pseudo_shadings,
                                    para['shading_threshold_wrt_max'], ref)
        save_image(average_ref, f'{str(camera_index)}_average_ref')

        # Step 3: fill the hole with reflectance from the surroundings
        # Fill holes
        print("Step 3: fill the hole with reflectance from the surroundings. k=" + str(para['fill_search_points']))
        first_key = next(iter(results_all[str(camera_index)]))
        normal = results_all[str(camera_index)][first_key]['normal']
        mask_empty = mask_shading.any(dim=0).squeeze(0)
        # here we need to consider some scenes with empty background.
        # (check this in other scenes)
        if args.setting != 'pair':
            mask_considered = results_all[str(camera_index)][first_key]['inter_mask'].squeeze(0) > 0
            mask_empty = torch.logical_or(mask_empty, ~mask_considered)

        filled_ref = fill_holes_kd(average_ref, normal, kmeans_center, mask_empty, para['fill_search_points'])
        # save results
        pseudo_label_all[str(camera_index)]['pseudo_reflectance'] = filled_ref
        save_image(filled_ref, f"{str(camera_index)}_pseudo_reflectance")

    torch.save(pseudo_label_all, os.path.join(output_path, 'pseudo_label_all.pt'))

    end_time = time.time()
    execution_time = (end_time - start_time) / 3600
    print(f"Finish the processing in {execution_time}h")
