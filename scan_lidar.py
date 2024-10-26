import glob
import os.path
from typing import Tuple

import numpy as np
import open3d as o3d
from tqdm import tqdm

import trimesh
from virtrual_scanner import Lidar, LidarScanner


def calculate_distance(extent, direction):
    if direction == 'front' or direction == 'back':
        max_extent = max(extent[0], extent[2])
    elif direction == 'left' or direction == 'right':
        max_extent = max(extent[1], extent[2])
    elif direction == 'top' or direction == 'bottom':
        max_extent = max(extent[0], extent[1])
    else:
        raise ValueError(f'Direction {direction} missmatch')
    return max_extent


def get_lidars(extent, resolution, fov):
    views = {
        "front":  Lidar(resolution, fov, np.array([0, 0, 0]), np.array([0, calculate_distance(extent, 'front'), 0]), np.array([0, 0, 1])),
        "back":   Lidar(resolution, fov, np.array([0, 0, 0]), np.array([0, -calculate_distance(extent, 'back'), 0]), np.array([0, 0, 1])),
        "left":   Lidar(resolution, fov, np.array([0, 0, 0]), np.array([calculate_distance(extent, 'left'), 0, 0]), np.array([0, 1, 0])),
        "right":  Lidar(resolution, fov, np.array([0, 0, 0]), np.array([-calculate_distance(extent, 'right'), 0, 0]), np.array([0, 1, 0])),
        "top":    Lidar(resolution, fov, np.array([0, 0, 0]), np.array([0, 0, calculate_distance(extent, 'top')]), np.array([0, 1, 0])),
        "bottom": Lidar(resolution, fov, np.array([0, 0, 0]), np.array([0, 0, -calculate_distance(extent, 'bottom')]), np.array([0, 1, 0])),
    }

    return views


def get_center_and_extent(mesh: o3d.geometry.TriangleMesh) -> Tuple[np.ndarray, np.ndarray]:
    bbox = mesh.get_axis_aligned_bounding_box()
    extent = bbox.get_extent()
    center = bbox.get_center()

    return center, extent


def parse_direction(direction: str) -> int:
    dictionary = {
        "front":  0,
        "back":   1,
        "left":   2,
        "right":  3,
        "top":    4,
        "bottom": 5,
    }
    return dictionary[direction]


def main():
    resolution = (0.05, 0.05)
    fov = (10., 10., 10., 10.)
    distance_noise_std = 0.015
    angle_noise_std = 100
    mesh_dir = 'D:/data/normal/meshes_data'
    mesh_list = glob.glob(mesh_dir + '/**/*.obj', recursive=True)
    save_dir = 'D:/data/normal/pcd_data/distance_0.015_angle_100_fov_10'
    os.makedirs(save_dir, exist_ok=True)
    cameras = get_lidars([10, 10, 10], resolution, fov)
    for item in tqdm(sorted(mesh_list)):
        mesh = o3d.io.read_triangle_mesh(item)
        center, extent = get_center_and_extent(mesh)
        mesh.translate(-center)
        mesh.scale(min(1.5, 1.5 / np.max(extent)), center=[0, 0, 0])
        center, extent = get_center_and_extent(mesh)
        # print(center, extent)
        pcd_all = o3d.t.geometry.PointCloud()
        pcd_all.point['positions'] = o3d.core.Tensor(np.zeros([0, 3], dtype=np.float32))
        pcd_all.point['colors'] = o3d.core.Tensor(np.zeros([0, 3], dtype=np.float32))
        pcd_all.point['normals'] = o3d.core.Tensor(np.zeros([0, 3], dtype=np.float32))
        pcd_all.point['theta'] = o3d.core.Tensor(np.zeros([0, 1], dtype=np.float32))
        pcd_all.point['phi'] = o3d.core.Tensor(np.zeros([0, 1], dtype=np.float32))
        pcd_all.point['source'] = o3d.core.Tensor(np.zeros([0, 1], dtype=np.int16))
        for direction, camera in cameras.items():
            scanner = LidarScanner(camera, distance_noise_std, np.deg2rad(angle_noise_std / 3600))
            tri_mesh = trimesh.Trimesh(vertices=np.asarray(mesh.vertices), faces=np.asarray(mesh.triangles))
            _points, _normals, _rays = scanner.virtual_scan(tri_mesh, use_noise=True)
            _dots = np.sum(_rays * _normals, axis=1)
            print(np.sum(_dots > 0.1))
            _theta, _phi = LidarScanner.direction_to_theta_phi(_rays)
            tpcd = o3d.t.geometry.PointCloud()
            tpcd.point['positions'] = o3d.core.Tensor(_points, dtype=o3d.core.Dtype.Float32)
            tpcd.point['colors'] = o3d.core.Tensor(_rays / 2 + 0.5, dtype=o3d.core.Dtype.Float32)
            tpcd.point['normals'] = o3d.core.Tensor(_normals, dtype=o3d.core.Dtype.Float32)
            tpcd.point['theta'] = o3d.core.Tensor(_theta.reshape(-1, 1), dtype=o3d.core.Dtype.Float32)
            tpcd.point['phi'] = o3d.core.Tensor(_phi.reshape(-1, 1), dtype=o3d.core.Dtype.Float32)
            tpcd.point['source'] = o3d.core.Tensor(np.full((_points.shape[0], 1), parse_direction(direction),
                                                           dtype=np.int16))
            pcd_all += tpcd
        o3d.t.io.write_point_cloud(f'{save_dir}/{os.path.basename(item)[:-4]}.pcd', pcd_all)



if __name__ == "__main__":
    main()
