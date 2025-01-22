import json
from functools import partial
import torch
import trimesh
import numpy as np
from tqdm import tqdm

def gl_to_cv(gl):
    # convert to CV convention used in Imaginaire
    cv = gl * np.array([1, -1, -1, 1])
    return cv

def get_points(transform_matrix, sphere_center, sphere_radius):
    c2w_gl = np.array(transform_matrix, dtype=np.float32)
    c2w = gl_to_cv(c2w_gl)
    # center scene
    center = np.array(sphere_center)
    c2w[:3, -1] -= center
    # scale scene
    scale = np.array(sphere_radius)
    c2w[:3, -1] /= scale
    pts_frame = []
    for i in range(5):
        pts_frame.append((c2w @ np.array([0, 0, float(i * i) * 0.01, 1])[:, None]).squeeze()[:3])
    return pts_frame

def extract_point(filename):
    with open(filename) as file:
        meta = json.load(file)
    frames = meta["frames"]
    pts = []
    get_pt = partial(get_points, sphere_center=np.array(meta["sphere_center"]),
                     sphere_radius=np.array(meta["sphere_radius"]))
    for frame in tqdm(frames):
        pts.extend(get_pt(frame["transform_matrix"]))
        pts.extend(get_pt(frame["transform_matrix_light"]))

    pts = np.stack(pts, axis=0)
    pcd = trimesh.PointCloud(pts)
    pcd.export(filename.replace('.json', '.ply'))
    print('Process done!')



if __name__ == "__main__":
    extract_point("/ghome/yyang/PycharmProjects/neuralangelo/dataset_rene/dinosaurs/train_transforms.json")
