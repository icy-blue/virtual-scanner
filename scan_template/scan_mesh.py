import glob
import math
import os
from datetime import datetime
from pytz import timezone
import numpy as np
import trimesh
from virtual_scanner import Lidar, LidarScanner, PointCloudManager

from scan_base import add_to_pcd, check_back_face, fast_visual_pc


def get_lidars(distance, resolution, fov):
    center = np.array([0, 0, 0])
    return [
        Lidar(resolution, fov, center, np.array([0, distance, -0]), np.array([0, 0, 1])),  # front
        Lidar(resolution, fov, center, np.array([0, -distance, 0]), np.array([0, 0, 1])),  # back
        Lidar(resolution, fov, center, np.array([distance, -0, 0]), np.array([0, 1, 0])),  # left
        Lidar(resolution, fov, center, np.array([-distance, 0, 0]), np.array([0, 1, 0])),  # right
        Lidar(resolution, fov, center, np.array([0, -0, distance]), np.array([0, 1, 0])),  # top
        Lidar(resolution, fov, center, np.array([0, 0, -distance]), np.array([0, 1, 0])),  # bottom
    ]


def get_center_extent(mesh: trimesh.Trimesh) -> [np.ndarray, np.ndarray]:
    minimum, maximum = mesh.bounds
    center = (minimum + maximum) / 2
    extent = maximum - minimum
    return center, extent


def scale_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    center, extent = get_center_extent(mesh)
    mesh.apply_translation(-center)
    mesh.apply_scale(1 / np.max(extent))
    return mesh


def diag_distance(mesh: trimesh.Trimesh) -> float:
    _, extent = get_center_extent(mesh)
    return np.linalg.norm(extent)


def get_scanners(distance, resolution, remap_sample_count):
    fov_y = np.rad2deg(np.arctan2(0.5, distance))
    fov_x = fov_y / resolution[1] * resolution[0]
    lidars = get_lidars(distance, resolution, (fov_x, fov_x, fov_y, fov_y))
    return [LidarScanner(lidar, scanner_id=i, remap_sample_count=remap_sample_count)
            for i, lidar in enumerate(lidars)]


def scan(mesh_file, noise_list, distance, scanners, back_face_points, save_dirs):
    mesh = trimesh.load_mesh(mesh_file)
    mesh = scale_mesh(mesh)
    distance_noise_std = [diag_distance(mesh) * noise for noise in noise_list]
    pcd_list = []
    for _ in noise_list:
        pcd_list.append(PointCloudManager())
    for scanner in scanners:
        __points, __normals, __rays, o_dots = scanner.virtual_scan_mitsuba(mesh)
        if len(__points) == 0:
            continue
        _points, _normals, _rays = check_back_face(__points, __normals, __rays, o_dots, back_face_points)
        for noise, pcd in zip(distance_noise_std, pcd_list):
            _points, _normals, _rays = scanner.apply_noise(_points, _rays, mesh, np.arctan2(noise, distance), noise)
            add_to_pcd(_points, _normals, _rays, scanner, pcd)

    for pcd, _dir in zip(pcd_list, save_dirs):
        # fast_visual_pc(pcd['positions'])
        pcd.save(os.path.join(_dir, os.path.basename(mesh_file)[:-4]))


def scan_mesh(noise_list, distance, resolution, remap_sample_count, mesh_dir, back_face_points):
    beijing_tz = timezone('Asia/Shanghai')
    today = datetime.now(beijing_tz).strftime('%m%d')
    save_dirs = [f'./{os.path.basename(mesh_dir)}_result-{today}/noise_{noise * 100:.2f}-{distance}m'
                 for noise in noise_list]
    for _dir in save_dirs:
        os.makedirs(_dir, exist_ok=True)
    scanners = get_scanners(distance, resolution, remap_sample_count)
    mesh_list = glob.glob(mesh_dir + '/**/*.obj', recursive=True)
    for i, mesh_file in enumerate(mesh_list):
        print(f'[{i}/{len(mesh_list)}] Scanning mesh {mesh_file}...')
        scan(mesh_file, noise_list, distance, scanners, back_face_points, save_dirs)