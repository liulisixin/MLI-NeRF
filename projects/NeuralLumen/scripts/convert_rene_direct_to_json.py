import copy

import numpy as np
import json
from argparse import ArgumentParser
import os
import cv2
from PIL import Image, ImageFile
from glob import glob
import math
import sys
from pathlib import Path
from rene.utils.loaders import ReneDataset
from tqdm import tqdm

from projects.neuralangelo.scripts.convert_data_to_json import _cv_to_gl  # noqa: E402

from projects.neuralangelo.scripts.convert_dtu_to_json import load_K_Rt_from_P


def rene_to_json(rene, scene_name, output_path):
    center_point = np.array([0.0, 0.0, 0.0])

    # cam
    n_cams = 50
    test_cams = [4, 8, 15]
    val_cams = [25, 42, 47]
    train_cams = [i for i in range(n_cams) if i not in test_cams + val_cams]
    cams_dict = {
        'train': train_cams,
        'val': val_cams,
        'test': test_cams
    }

    # light
    n_lights = 40
    test_lights = [2, 21, 34]
    train_lights = [i for i in range(n_lights) if i not in test_lights]
    lights_dict = {
        'train': train_lights,
        'test': test_lights
    }

    # Here we need to calculate the range of the scene.
    pts = []
    for cam_id in range(n_cams):
        sample = rene[scene_name][0][cam_id]
        pose_cam = sample["pose"]()
        pts.append((pose_cam @ np.array([0, 0, 0, 1])[:, None]).squeeze()[:3])
    for light_id in range(n_lights):
        sample = rene[scene_name][light_id][0]
        pose_light = sample["light"]()
        pts.append((pose_light @ np.array([0, 0, 0, 1])[:, None]).squeeze()[:3])
    vertices = np.stack(pts)
    center = center_point
    scale_factor = 1.0
    factor = scale_factor
    radius = np.linalg.norm(vertices - center, ord=2, axis=-1).max() * factor
    scale_mat = np.diag([radius, radius, radius, 1.0]).astype(np.float32)
    scale_mat[:3, 3] = center

    bounding_box_aabb = np.array([-0.55, -0.43, -0.15, 0.55, 0.35, 0.25]) * 1.2 / scale_factor

    """
    Follow the setting in Rene paper.
    """
    pairs_dict = {}

    # pairs_dict['all'] = [[i, j] for i in range(n_cams) for j in range(n_lights)]
    #
    # pairs_dict['train'] = [[i, j] for i in cams_dict['train'] for j in lights_dict['train']]
    # pairs_dict['val'] = [[25, 10], [25, 12], [25, 27],
    #                      [42, 3], [42, 20], [42, 23],
    #                      [47, 26], [47, 30], [47, 39]]
    # # add some samples from train set, only for viewing.
    # pairs_dict['val_add'] = copy.deepcopy(pairs_dict['val'])
    # pairs_dict['val_add'].extend([[10, 0], [10, 17], [27, 0], [27, 17]])
    # pairs_dict['test_all'] = [[i, j] for i in cams_dict['test'] for j in range(n_lights)]
    # pairs_dict['test_easy'] = [[i, j] for i in cams_dict['test'] for j in range(n_lights) if j in lights_dict['train']]
    # pairs_dict['test_hard'] = [[i, j] for i in cams_dict['test'] for j in range(n_lights) if
    #                            j not in lights_dict['train']]

    # Since the original testing dataset is not shared.
    # Here we use the images in validation (3 cameras) as test.
    # Notice that the relighting cannot be tested.
    pairs_dict['test_custom'] = [[i, j] for i in cams_dict['val'] for j in lights_dict['train']]

    for split in pairs_dict.keys():
        out = {
            "k1": 0.0,  # take undistorted images only
            "k2": 0.0,
            "k3": 0.0,
            "k4": 0.0,
            "p1": 0.0,
            "p2": 0.0,
            "is_fisheye": False,
            "frames": []
        }
        # folder_name = 'image_' + split
        # os.makedirs(os.path.join(output_path, case_name, folder_name), exist_ok=True)
        for index, pair in tqdm(enumerate(pairs_dict[split])):
            cam_id = pair[0]
            light_id = pair[1]
            sample = rene[scene_name][light_id][pair[0]]

            # for camera
            pose_raw = sample["pose"]()
            pose = pose_raw.astype(np.float32)
            intrinsic = np.diag([1.0, 1.0, 1.0, 1.0]).astype(np.float32)
            intrinsic[:3, :3] = sample["camera"]()['intrinsics']['camera_matrix']
            w2c = np.linalg.inv(pose)
            world_mat = intrinsic @ w2c
            world_mat = world_mat.astype(np.float32)
            # scale and decompose
            P = world_mat @ scale_mat
            P = P[:3, :4]
            intrinsic_param, c2w = load_K_Rt_from_P(None, P)
            c2w_gl = _cv_to_gl(c2w)

            # for light
            pose_light_raw = sample["light"]()
            pose_light = pose_light_raw.astype(np.float32)
            # during the scale, only the t is changed and the R keeps the same.
            pose_light[:, 3] = np.linalg.inv(scale_mat) @ pose_light[:, 3]    # c2w
            c2w_light_gl = _cv_to_gl(pose_light)

            image_name = 'lset{:0>3d}/data/{:0>2d}_image.png'.format(light_id, cam_id)
            frame = {"index": index, "file_path": image_name,
                     "light_index": light_id, "camera_index": cam_id,
                     "transform_matrix": c2w_gl.tolist(),
                     "transform_matrix_light": c2w_light_gl.tolist()}
            out["frames"].append(frame)

        fl_x = intrinsic_param[0][0]
        fl_y = intrinsic_param[1][1]
        cx = intrinsic_param[0][2]
        cy = intrinsic_param[1][2]
        sk_x = intrinsic_param[0][1]
        sk_y = intrinsic_param[1][0]
        img = sample["image"]()
        w, h = img.shape[1], img.shape[0]

        angle_x = math.atan(w / (fl_x * 2)) * 2
        angle_y = math.atan(h / (fl_y * 2)) * 2

        scale_mat = scale_mat.astype(float)

        out.update({
            "camera_angle_x": angle_x,
            "camera_angle_y": angle_y,
            "fl_x": fl_x,
            "fl_y": fl_y,
            "cx": cx,
            "cy": cy,
            "sk_x": sk_x,
            "sk_y": sk_y,
            "w": int(w),
            "h": int(h),
            "aabb_scale": np.exp2(np.rint(np.log2(scale_mat[0, 0]))),  # power of two, for INGP resolution computation
            "sphere_center": [0., 0., 0.],
            "sphere_radius": 1.,
            "original_dataset_center": center_point.tolist(),
            "original_dataset_scale": scale_factor,
            "bounding_box_aabb": bounding_box_aabb.tolist(),
            "radius_scale_mat": radius
        })

        file_path = os.path.join(output_path, split + '_transforms.json')
        with open(file_path, "w") as outputfile:
            json.dump(out, outputfile, indent=2)
        print('Writing data to json file: ', file_path)


if __name__ == '__main__':
    dataset_path = "/ghome/yyang/dataset/rene_dataset"
    rene = ReneDataset(input_folder=dataset_path)
    # scene_list = ['savannah',
    #               'cheetah',
    #               'cube',
    #               'fruits',
    #               'garden',
    #               'lego',
    #               'apple',
    #               'dinosaurs']
    scene_list = ['savannah',
                  'apple',
                  'garden',
                  'cube']
    for scene_name in scene_list:
        json_output_path = "/ghome/yyang/PycharmProjects/neuralangelo/dataset_rene/" + scene_name
        os.makedirs(json_output_path, exist_ok=True)

        rene_to_json(rene, scene_name, json_output_path)
