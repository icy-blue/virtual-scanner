from typing import Tuple

import numpy as np
import open3d as o3d


class Ray:
    def __init__(self, start: 'Tuple[float, float, float]', direction: 'Tuple[float, float, float]'):
        self.start = start
        self.direction = direction

    @classmethod
    def from_numpy(cls, start: np.ndarray, direction: np.ndarray):
        rays = []
        for i in range(start.shape[1]):
            rays.append(cls(start[:, i], direction[:, i]))
        return rays

    def __str__(self):
        return f"Ray(start={self.start}, direction={self.direction})"
        

class Lidar:
    def __init__(self, resolution: 'Tuple[float, float]', fov: 'Tuple[float, float, float, float]', center: 'np.ndarray',
                 eye: 'np.ndarray', up: 'np.ndarray'):
        self.resolution = resolution
        self.fov = fov # Left, Right, Up, Down
        self.center = center
        self.eye = eye
        self.up = up

    def get_extrinsics(self) -> np.ndarray:
        z_axis = self.center - self.eye
        z_axis = z_axis / np.linalg.norm(z_axis)

        x_axis = np.cross(self.up, z_axis)
        x_axis = x_axis / np.linalg.norm(x_axis)

        y_axis = np.cross(z_axis, x_axis)

        rotation_matrix = np.vstack([x_axis, y_axis, z_axis]).T
        translation_vector = -rotation_matrix @ self.eye.reshape([-1, 1])
        extrinsic_matrix_3x4 = np.hstack([rotation_matrix, translation_vector.reshape(-1, 1)])
        # print(extrinsic_matrix_3x4)
        extrinsic_matrix_4x4 = np.vstack([extrinsic_matrix_3x4, [0, 0, 0, 1]])

        return extrinsic_matrix_4x4

    def get_rays(self) -> Tuple[np.ndarray, np.ndarray]:
        fov_l, fov_r, fov_u, fov_d = self.fov  # 视场角度
        res_x, res_y = self.resolution
        
        forward = (self.center - self.eye) / np.linalg.norm(self.center - self.eye)
        right = np.cross(forward, self.up)
        right /= np.linalg.norm(right)
        new_up = np.cross(right, forward)
        
        angle_x = np.linspace(-fov_l, fov_r, res_x)
        angle_y = np.linspace(-fov_d, fov_u, res_y)

        tan_angle_x, tan_angle_y = np.tan(np.radians(angle_x)), np.tan(np.radians(angle_y))
        tan_grid_x, tan_grid_y = np.meshgrid(tan_angle_x, tan_angle_y, indexing='ij')

        rays_local = np.stack([tan_grid_x, tan_grid_y, np.ones_like(tan_grid_x)], axis=-1)

        rays_world = (
            forward + 
            tan_grid_x[..., np.newaxis] * right + 
            tan_grid_y[..., np.newaxis] * new_up
        )

        rays_local /= np.linalg.norm(rays_local, axis=-1, keepdims=True)
        rays_world /= np.linalg.norm(rays_world, axis=-1, keepdims=True)
        
        rays_local = rays_local.reshape(-1, 3)
        rays_world = rays_world.reshape(-1, 3)
        
        return rays_local, rays_world
    
    def __str__(self) -> str:
        return f"Lidar(resolution={self.resolution}, fov={self.fov}, eye={self.eye}, center={self.center}, up={self.up})"
