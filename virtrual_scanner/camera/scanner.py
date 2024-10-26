import warnings
from typing import Tuple, List

import numpy as np
import open3d as o3d
from open3d.visualization.rendering import MaterialRecord

from .camera import Camera


class CameraScanner:
    def __init__(self, camera: Camera):
        self.camera = camera

    def virtual_scan(self, mesh_list: List[o3d.geometry.TriangleMesh],
                     material: MaterialRecord = MaterialRecord()) -> np.ndarray:
        width, height = self.camera.resolution

        renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
        for i, mesh in enumerate(mesh_list):
            renderer.scene.add_geometry(f"mesh {i}", mesh, material)

        # 下面三行是一样的
        # renderer.setup_camera(self.camera.get_vertical_fov(), self.camera.center.reshape([-1, 1]),
        #                       self.camera.eye.reshape([-1, 1]), self.camera.up.reshape([-1, 1]))
        renderer.setup_camera(self.camera.get_o3d_intrinsics(), self.camera.get_extrinsics())
        # renderer.setup_camera(self.camera.get_intrinsics(), self.camera.get_extrinsics(), width, height)

        depth_image = renderer.render_to_depth_image(z_in_view_space=True)
        depth_image = np.asarray(depth_image)

        return depth_image

    def depth_to_point_cloud(self, depth_image: np.ndarray) -> o3d.geometry.PointCloud:
        depth_image[depth_image == np.inf] = 0
        depth_image = o3d.geometry.Image(depth_image)
        point_cloud = o3d.geometry.PointCloud().create_from_depth_image(depth_image, self.camera.get_o3d_intrinsics(),
                                                                        self.camera.get_extrinsics())
        return point_cloud

    def depth_to_polar_coord(self, depth_image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        height, width = depth_image.shape[:2]
        focal_length = self.camera.get_focal_length()

        u, v = np.meshgrid(np.arange(width), np.arange(height))
        valid_mask = np.isfinite(depth_image) & (depth_image > 0)
        polar_coordinates = np.full((height, width, 3), fill_value=np.nan)

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            Z_c = depth_image
            X_c = (u - width // 2) * Z_c / focal_length
            Y_c = (v - height // 2) * Z_c / focal_length

        r = np.sqrt(X_c ** 2 + Y_c ** 2 + Z_c ** 2)
        theta = np.arctan2(np.sqrt(X_c ** 2 + Y_c ** 2), Z_c)  # 仰角 theta
        phi = np.arctan2(Y_c, X_c)  # 方位角 phi

        polar_coordinates[valid_mask, 0] = r[valid_mask]
        polar_coordinates[valid_mask, 1] = theta[valid_mask]
        polar_coordinates[valid_mask, 2] = phi[valid_mask]

        return polar_coordinates, valid_mask

    def polar_coord_to_point_cloud(self, polar_coordinates: np.ndarray,
                                   valid_mask: np.ndarray) -> o3d.geometry.PointCloud:
        valid_polar = polar_coordinates[valid_mask]

        r = valid_polar[:, 0]
        theta = valid_polar[:, 1]  # 仰角 theta
        phi = valid_polar[:, 2]  # 方位角 phi

        X_c = r * np.sin(theta) * np.cos(phi)
        Y_c = r * np.sin(theta) * np.sin(phi)
        Z_c = r * np.cos(theta)

        ones = np.ones_like(X_c)
        points_camera = np.stack((X_c, Y_c, Z_c, ones), axis=-1)

        points_world = np.linalg.inv(self.camera.get_extrinsics()) @ points_camera.T
        point_cloud_world = points_world.T[:, :3]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_cloud_world)
        return pcd
