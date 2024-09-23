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
    def __init__(self, resolution: 'Tuple[float, float]', fov: 'Tuple[float, float]', center: 'np.ndarray',
                 eye: 'np.ndarray', up: 'np.ndarray'):
        self.resolution = resolution
        self.fov = fov
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
        extrinsics = self.get_extrinsics()
        
        fov_x, fov_y = self.fov  # 视场角
        res_x, res_y = self.resolution  # 角度采样间隔
        
        # 计算射线数量
        num_rays_x = int(np.ceil(fov_x / res_x)) + 1
        num_rays_y = int(np.ceil(fov_y / res_y)) + 1
        
        # 生成角度网格
        angles_x = (np.arange(num_rays_x) - num_rays_x // 2) * res_x
        angles_y = (np.arange(num_rays_y) - num_rays_y // 2) * res_y
        
        # 角度转换为弧度
        angles_x_rad = np.deg2rad(angles_x)
        angles_y_rad = np.deg2rad(angles_y)
        
        # 创建角度网格
        angles_x_grid, angles_y_grid = np.meshgrid(angles_x_rad, angles_y_rad)
        
        # 计算Lidar坐标系下的射线方向
        ray_dir_lidar_x = np.cos(angles_y_grid) * np.sin(angles_x_grid)
        ray_dir_lidar_y = np.sin(angles_y_grid)
        ray_dir_lidar_z = np.cos(angles_y_grid) * np.cos(angles_x_grid)
        
        # 合并方向向量并归一化
        rays_lidar = np.stack([ray_dir_lidar_x, ray_dir_lidar_y, ray_dir_lidar_z], axis=-1)
        rays_lidar = rays_lidar.reshape(-1, 3)
        rays_lidar /= np.linalg.norm(rays_lidar, axis=1, keepdims=True)
        
        # 将射线转换到世界坐标系
        rays_world = (extrinsics[:3, :3] @ rays_lidar.T).T
        rays_world /= np.linalg.norm(rays_world, axis=1, keepdims=True)
        
        return rays_lidar, rays_world

    def __str__(self) -> str:
        return f"Lidar(resolution={self.resolution}, fov={self.fov}, eye={self.eye}, center={self.center}, up={self.up})"
