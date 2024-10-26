import glob
import os.path
from typing import Tuple

import numpy as np
import open3d as o3d
from tqdm import tqdm

import trimesh
from virtual_scanner import Lidar, LidarScanner, PointCloudManager


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
    distance_noise_std = 0.01
    angle_noise_std = 50
    mesh_dir = 'D:/data/normal/meshes_data'
    mesh_list = glob.glob(mesh_dir + '/**/*.obj', recursive=True)
    save_dir = 'D:/data/normal/pcd_data/distance_0.01_angle_50_fov_10'
    os.makedirs(save_dir, exist_ok=True)
    cameras = get_lidars([10, 10, 10], resolution, fov)
    for item in tqdm(sorted(mesh_list)):
        mesh = o3d.io.read_triangle_mesh(item)
        center, extent = get_center_and_extent(mesh)
        mesh.translate(-center)
        mesh.scale(min(1.5, 1.5 / np.max(extent)), center=[0, 0, 0])
        center, extent = get_center_and_extent(mesh)
        # print(center, extent)
        pcd = PointCloudManager()
        for direction, camera in cameras.items():
            scanner = LidarScanner(camera, distance_noise_std, np.deg2rad(angle_noise_std / 3600))
            tri_mesh = trimesh.Trimesh(vertices=np.asarray(mesh.vertices), faces=np.asarray(mesh.triangles))
            _points, _normals, _rays = scanner.virtual_scan(tri_mesh, use_noise=True)
            _dots = np.sum(_rays * _normals, axis=1)
            print(np.sum(_dots > 0.1), _dots[_dots > 0.1])
            _theta, _phi = LidarScanner.direction_to_theta_phi(_rays)
            points = _points.astype(np.float32)
            colors = (_rays / 2 + 0.5).astype(np.float32)
            normals = _normals.astype(np.float32)
            theta = _theta.astype(np.float32).reshape(-1, 1)
            phi = _phi.astype(np.float32).reshape(-1, 1)
            source = np.full((_points.shape[0], 1), parse_direction(direction), dtype=np.int32)
            pcd.add(positions=points, colors=colors, normals=normals, theta=theta, phi=phi, source=source)
        pcd.save(f'{save_dir}/{os.path.basename(item)[:-4]}.pcd')


if __name__ == "__main__":
    main()
