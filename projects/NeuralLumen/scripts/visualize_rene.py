from rene.utils.loaders import ReneDataset
import matplotlib.pyplot as plt
import numpy as np


def plot_point(points, name):
    plt.figure(figsize=(10, 6))

    for i, point in enumerate(points):
        plt.plot(point[0], point[1], 'bo')
        plt.text(point[0], point[1], f'{i}', fontsize=15)

    plt.title('2D Points ' + name)
    plt.xlabel('X axis')
    plt.ylabel('Y axis')
    plt.grid(True)
    plt.show()


def visualize_rene(rene, scene_name):
    # cam
    n_cams = 50
    # light
    n_lights = 40

    cam_pose_0 = rene[scene_name][0][0]["pose"]()
    cam_pose_0_inv = np.linalg.inv(cam_pose_0)
    light_pose_0 = rene[scene_name][0][0]["light"]()
    light_pose_0_inv = np.linalg.inv(light_pose_0)
    pts_cam = []
    for cam_id in range(n_cams):
        sample = rene[scene_name][0][cam_id]
        pose_cam = sample["pose"]()
        position_world = (pose_cam @ np.array([0, 0, 0, 1])[:, None]).squeeze()
        position_0 = (cam_pose_0_inv @ position_world[:, None].squeeze())[:2]
        pts_cam.append(position_0)
    plot_point(pts_cam, 'camera')
    pts_light = []
    for light_id in range(n_lights):
        sample = rene[scene_name][light_id][0]
        pose_light = sample["light"]()
        position_world = (pose_light @ np.array([0, 0, 0, 1])[:, None]).squeeze()
        position_0 = (light_pose_0_inv @ position_world[:, None].squeeze())[:2]
        pts_light.append(position_0)
    plot_point(pts_light, 'light')


if __name__ == '__main__':
    dataset_path = "/ghome/yyang/dataset/rene_dataset"
    rene = ReneDataset(input_folder=dataset_path)
    scene_name = "apple"

    visualize_rene(rene, scene_name)
