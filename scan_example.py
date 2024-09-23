import glob
import os.path
from typing import Tuple

import numpy as np
import open3d as o3d
from tqdm import tqdm
from knn import KNN

import knn
from virtual_scanner.camera import Camera
from virtual_scanner.scanner import CameraScanner


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


def get_cameras(extent, resolution, fov):
    views = {
        "front":  Camera(resolution, fov, np.array([0, 0, 0]), np.array([0, calculate_distance(extent, 'front'), 0]), np.array([0, 0, 1])),
        "back":   Camera(resolution, fov, np.array([0, 0, 0]), np.array([0, -calculate_distance(extent, 'back'), 0]), np.array([0, 0, 1])),
        "left":   Camera(resolution, fov, np.array([0, 0, 0]), np.array([calculate_distance(extent, 'left'), 0, 0]), np.array([0, 1, 0])),
        "right":  Camera(resolution, fov, np.array([0, 0, 0]), np.array([-calculate_distance(extent, 'right'), 0, 0]), np.array([0, 1, 0])),
        "top":    Camera(resolution, fov, np.array([0, 0, 0]), np.array([0, 0, calculate_distance(extent, 'top')]), np.array([0, 1, 0])),
        "bottom": Camera(resolution, fov, np.array([0, 0, 0]), np.array([0, 0, -calculate_distance(extent, 'bottom')]), np.array([0, 1, 0])),
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
    resolution = (320, 240)
    fov = 60.0
    mesh_dir = 'D:\\data\\abc_gt_test'
    save_dir = 'D:\\data\\abc_scan_result'
    mesh_list = glob.glob(mesh_dir + '/*.obj')
    for item in tqdm(sorted(mesh_list)):
        mesh = o3d.io.read_triangle_mesh(item)
        center, extent = get_center_and_extent(mesh)
        # print(center, extent)
        mesh.translate(-center)
        mesh.scale(min(0.5, 0.5 / np.max(extent)), center=[0, 0, 0])
        center, extent = get_center_and_extent(mesh)
        print(center, extent)
        cameras = get_cameras(extent * 2, resolution, fov)
        pcd_all = o3d.t.geometry.PointCloud()
        pcd_all.point['positions'] = o3d.core.Tensor(np.zeros([0, 3], dtype=np.float64))
        pcd_all.point['source'] = o3d.core.Tensor(np.zeros([0, 1], dtype=np.int32))
        coord = o3d.geometry.TriangleMesh().create_coordinate_frame(size=np.max(extent) / 2)
        for direction, camera in cameras.items():
            scanner = CameraScanner(camera)
            depth_image = scanner.virtual_scan([mesh])
            polar_coordinates, valid_mask = scanner.depth_to_polar_coord(depth_image)
            point_cloud = scanner.polar_coord_to_point_cloud(polar_coordinates, valid_mask)
            _points = np.asarray(point_cloud.points)
            tpcd = o3d.t.geometry.PointCloud()
            tpcd.point['positions'] = o3d.core.Tensor(_points)
            tpcd.point['source'] = o3d.core.Tensor(np.full((_points.shape[0], 1), parse_direction(direction),
                                                           dtype=np.int32))
            # point_cloud = scanner.depth_to_point_cloud(depth_image)
            # o3d.visualization.draw_geometries([point_cloud, mesh, coord])
            pcd_all += tpcd

        sampled = mesh.sample_points_uniformly(4000000, use_triangle_normal=True)
        sampled = np.hstack([np.asarray(sampled.points), np.asarray(sampled.normals)])

        dists, idx = KNN.huge_point_cloud_nn(pcd_all.point['positions'].numpy(), sampled[:, :3], grid_length=0.1,
                                             expand_length=0.01, patch_size=100000, verbose=False)
        pcd_all.point['normals'] = o3d.core.Tensor(sampled[idx, 3:6])
        pcd_all.point['colors'] = o3d.core.Tensor(sampled[idx, 3:6] / 2 + 0.5)

        # o3d.visualization.draw_geometries([pcd_all, mesh, coord])
        basename = os.path.basename(item).split('.')[0]
        o3d.t.io.write_point_cloud(os.path.join(save_dir, basename + '.pcd'), pcd_all)


if __name__ == "__main__":
    main()
