from typing import Tuple

import numpy as np
import open3d as o3d


class Camera:
    def __init__(self, resolution: 'Tuple[int, int]', fov: 'float', center: 'np.ndarray', eye: 'np.ndarray',
                 up: 'np.ndarray'):
        self.resolution = resolution
        self.fov = fov
        self.center = center
        self.eye = eye
        self.up = up

    def get_focal_length(self):
        width, _ = self.resolution
        fov_rad = np.deg2rad(self.fov)
        focal_length = width / (2 * np.tan(fov_rad / 2))
        return focal_length

    def get_extrinsics(self) -> np.ndarray:
        z_axis = self.center - self.eye
        z_axis = z_axis / np.linalg.norm(z_axis)

        x_axis = np.cross(self.up, z_axis)
        x_axis = x_axis / np.linalg.norm(x_axis)

        y_axis = np.cross(z_axis, x_axis)

        rotation_matrix = np.vstack([x_axis, y_axis, z_axis]).T
        translation_vector = -rotation_matrix @ self.eye.reshape([-1, 1])
        extrinsic_matrix_3x4 = np.hstack([rotation_matrix, translation_vector.reshape(-1, 1)])
        extrinsic_matrix_4x4 = np.vstack([extrinsic_matrix_3x4, [0, 0, 0, 1]])

        return extrinsic_matrix_4x4

    def get_intrinsics(self) -> np.ndarray:
        width, height = self.resolution

        focal_length = self.get_focal_length()

        cx = width / 2
        cy = height / 2

        K = np.array([
            [focal_length, 0, cx],
            [0, focal_length, cy],
            [0, 0, 1]
        ])

        return K

    def get_o3d_intrinsics(self) -> o3d.camera.PinholeCameraIntrinsic:
        width, height = self.resolution

        focal_length = self.get_focal_length()

        return o3d.camera.PinholeCameraIntrinsic(width, height, focal_length, focal_length, width / 2, height / 2)

    def get_vertical_fov(self) -> float:
        _, height = self.resolution
        return np.rad2deg(2 * np.arctan(height / (2 * self.get_focal_length())))

    def __str__(self) -> str:
        return f"Camera(resolution={self.resolution}, fov={self.fov}, eye={self.eye}, center={self.center}, up={self.up})"
