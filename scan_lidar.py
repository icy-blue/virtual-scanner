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
    resolution = (200, 200)
    distance = 5
    fov_y = np.rad2deg(np.arctan2(0.5, distance))
    fov_x = fov_y / resolution[1] * resolution[0]
    print(fov_x, fov_y)
    distance_noise_rate = 0.0025
    # angle_noise_std = 100
    mesh_dir = './meshes'
    mesh_list = glob.glob(mesh_dir + '/**/*.obj', recursive=True)
    save_dir = f'./pcds/noise_{distance_noise_rate * 100:.2f}-{distance}m'
    print(save_dir)
    os.makedirs(save_dir, exist_ok=True)
    cameras = get_lidars([distance] * 3, resolution, (fov_x, fov_x, fov_y, fov_y))
    for item in tqdm(sorted(mesh_list)):
        print(item)
        if os.path.exists(f'{save_dir}/{os.path.basename(item)[:-4]}.pcd'):
            continue
        mesh = o3d.io.read_triangle_mesh(item)
        center, extent = get_center_and_extent(mesh)
        mesh.translate(-center)
        mesh.scale(1 / np.max(extent), center=[0, 0, 0])
        center, extent = get_center_and_extent(mesh)
        # print(np.linalg.norm(np.array(extent)) * distance_noise_rate)
        distance_noise_std = np.linalg.norm(np.array(extent)) * distance_noise_rate
        tri_mesh = trimesh.Trimesh(vertices=np.asarray(mesh.vertices), faces=np.asarray(mesh.triangles))
        # print(center, extent)
        pcd = PointCloudManager()
        for direction, camera in cameras.items():
            # scanner = LidarScanner(camera, distance_noise_std, np.deg2rad(angle_noise_std))
            scanner = LidarScanner(camera, distance_noise_std, np.arctan2(distance_noise_std, distance))
            _points, _normals, _rays, _triangle, o_dots = scanner.virtual_scan(tri_mesh,
                                                                               use_noise=scanner.distance_noise_std != 0,
                                                                               edit_normal=True)
            new_dots = np.sum(_normals * _rays, axis=1)
            index = o_dots > 0
            print(np.sum(index), np.sum(~index), new_dots[index])
            print(np.unique(np.sign(o_dots[~index]) * np.sign(new_dots[~index]), return_counts=True))
            _rays, _points, _normals = _rays[~index], _points[~index], _normals[~index]
            _theta, _phi = LidarScanner.direction_to_theta_phi(_rays)
            points = _points.astype(np.float32)
            colors = (_rays / 2 + 0.5).astype(np.float32)
            normals = _normals.astype(np.float32)
            theta = _theta.astype(np.float32).reshape(-1, 1)
            phi = _phi.astype(np.float32).reshape(-1, 1)
            source = np.full((_points.shape[0], 1), parse_direction(direction), dtype=np.int32)
            traingle = _triangle.astype(np.int32).reshape(-1, 1)
            pcd.add(positions=points, colors=colors, normals=normals, theta=theta, phi=phi, source=source, traingle=traingle)
        pcd.save(f'{save_dir}/{os.path.basename(item)[:-4]}.pcd')


if __name__ == "__main__":
    main()
