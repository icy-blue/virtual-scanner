import warnings
from typing import Tuple, List

import numpy as np
import trimesh
import open3d as o3d
from .lidar import Lidar


class LidarScanner:
    def __init__(self, lidar: Lidar, distance_noise_std: float = 0.001, angle_noise_std: float = np.deg2rad(10 / 3600)):
        """
        :param lidar: Lidar instance
        :param distance_noise_std: 距离噪声的标准差 (单位：米)，默认为1毫米
        :param angle_noise_std: 角度噪声的标准差 (单位：弧度)，默认为10角秒
        """
        self.lidar = lidar
        self.distance_noise_std = distance_noise_std
        self.angle_noise_std = angle_noise_std
    
    def apply_angular_noise(self, ray_lidar: np.ndarray) -> np.ndarray:
        """
        在Lidar的极坐标系下对角度进行噪声扰动
        :param ray_lidar: Lidar坐标系下的射线方向 (Nx3)
        :return: 加入角度噪声后的Lidar坐标系下射线方向 (Nx3)
        """
        # 将Lidar射线方向转换为极坐标
        horizontal_angle = np.arctan2(ray_lidar[:, 0], ray_lidar[:, 2])  # 水平方向的角度
        vertical_angle = np.arcsin(ray_lidar[:, 1])  # 垂直方向的角度

        # 添加高斯噪声到角度
        horizontal_angle += np.random.normal(0, self.angle_noise_std, size=horizontal_angle.shape)
        vertical_angle += np.random.normal(0, self.angle_noise_std, size=vertical_angle.shape)

        # 将扰动后的极坐标转换回笛卡尔坐标
        ray_lidar_noisy = np.zeros_like(ray_lidar)
        ray_lidar_noisy[:, 0] = np.cos(vertical_angle) * np.sin(horizontal_angle)  # X方向
        ray_lidar_noisy[:, 1] = np.sin(vertical_angle)  # Y方向
        ray_lidar_noisy[:, 2] = np.cos(vertical_angle) * np.cos(horizontal_angle)  # Z方向

        # 归一化
        ray_lidar_noisy /= np.linalg.norm(ray_lidar_noisy, axis=1, keepdims=True)

        return ray_lidar_noisy

    def apply_noise(self, point_cloud: np.ndarray, rays_lidar: np.ndarray, rays_world: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        应用角度和距离噪声，返回带噪声的点云
        :param point_cloud: 原始点云 (Nx3)
        :param rays_lidar: Lidar坐标系下的射线方向 (Nx3)
        :param rays_world: 世界坐标系下的射线方向 (Nx3)
        :return: 加入噪声后的点云和射线方向
        """
        # 1. 在Lidar坐标系下添加角度噪声
        noisy_rays_lidar = self.apply_angular_noise(rays_lidar)

        # 2. 将加入噪声的射线方向转换到世界坐标系
        extrinsics = self.lidar.get_extrinsics()
        noisy_rays_world = (extrinsics[:3, :3] @ noisy_rays_lidar.T).T

        # 3. 计算距离噪声并更新点云位置
        distances = np.linalg.norm(point_cloud - self.lidar.eye, axis=1)
        distance_noise = np.random.normal(0, self.distance_noise_std, size=distances.shape)
        noisy_distances = distances + distance_noise

        # 重新计算带噪声的点云位置
        noisy_point_cloud = self.lidar.eye + noisy_rays_world * noisy_distances[:, np.newaxis]

        return noisy_point_cloud, noisy_rays_world
    
    def virtual_scan(self, mesh: trimesh.Trimesh, use_noise: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        rays_lidar, rays_world = self.lidar.get_rays()
        origin = np.repeat([self.lidar.eye], rays_world.shape[0], axis=0)

        # 进行射线与mesh的相交计算
        index_triangle, index_ray, locations = mesh.ray.intersects_id(
            origin, rays_world, multiple_hits=False, return_locations=True
        )
        point_cloud = locations
        normal = mesh.face_normals[index_triangle]
        rays_direction_lidar = rays_lidar[index_ray]
        rays_direction_world = rays_world[index_ray]

        if not use_noise:
            return point_cloud, normal, rays_direction_world

        # 先添加角度噪声，再计算距离噪声并更新点云
        noisy_point_cloud, noisy_rays_direction_world = self.apply_noise(point_cloud, rays_direction_lidar, rays_direction_world)

        return noisy_point_cloud, normal, noisy_rays_direction_world

        