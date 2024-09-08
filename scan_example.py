from typing import Tuple

import numpy as np
import open3d as o3d

from camera import Camera
from scanner import Scanner


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


def main():
    resolution = (640, 480)
    fov = 60.0
    mesh = o3d.geometry.TriangleMesh().create_cylinder(0.3, 1.0)
    center, extent = get_center_and_extent(mesh)
    # mesh.translate([1, 0, 0])
    print(center, extent)
    cameras = get_cameras(extent * 2, resolution, fov)
    pcd_all = o3d.geometry.PointCloud()
    coord = o3d.geometry.TriangleMesh().create_coordinate_frame(size=np.max(extent) / 2)
    for direction, camera in cameras.items():
        scanner = Scanner(camera)
        depth_image = scanner.virtual_scan([mesh])
        polar_coordinates, valid_mask = scanner.depth_to_polar_coord(depth_image)
        point_cloud = scanner.polar_coord_to_point_cloud(polar_coordinates, valid_mask)
        # point_cloud = scanner.depth_to_point_cloud(depth_image)
        o3d.visualization.draw_geometries([point_cloud, mesh, coord])
        pcd_all += point_cloud

    o3d.visualization.draw_geometries([pcd_all, mesh, coord])


if __name__ == "__main__":
    main()
