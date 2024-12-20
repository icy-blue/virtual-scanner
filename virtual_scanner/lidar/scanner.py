from typing import Tuple

import numpy as np
import trimesh
from .lidar import Lidar
from ..tool import KNN as knn, mitsuba_intersect


class LidarScanner:
    def __init__(self,
                 lidar: Lidar,
                 distance_noise_std: float = 0.001,
                 angle_noise_std: float = np.deg2rad(10 / 3600),
                 remap_sample_count: int = 100000):
        """
        :param lidar: Lidar instance
        :param distance_noise_std: 距离噪声的标准差 (单位：米)，默认为1毫米
        :param angle_noise_std: 角度噪声的标准差 (单位：弧度)，默认为10角秒
        :param remap_sample_count: 修正法向时采样点数量
        """
        self.lidar = lidar
        self.distance_noise_std = distance_noise_std
        self.angle_noise_std = angle_noise_std
        self.remap_sample_count = remap_sample_count

    
    def apply_angular_noise(self, ray_lidar: np.ndarray) -> np.ndarray:
        """
        在Lidar的极坐标系下对角度进行噪声扰动
        :param ray_lidar: Lidar坐标系下的射线方向 (Nx3)
        :return: 加入角度噪声后的Lidar坐标系下射线方向 (Nx3)
        """
        theta, phi = self.direction_to_theta_phi(ray_lidar)

        # 添加高斯噪声到角度
        theta += np.random.normal(0, self.angle_noise_std, size=theta.shape)
        phi += np.random.normal(0, self.angle_noise_std, size=phi.shape)

        # 将扰动后的极坐标转换回笛卡尔坐标
        ray_lidar_noisy = self.theta_phi_to_direction(theta, phi)

        return ray_lidar_noisy

    def apply_noise(self, point_cloud: np.ndarray, rays_world: np.ndarray, mesh: trimesh.Trimesh) \
            -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        应用角度和距离噪声，返回带噪声的点云
        :param point_cloud: 原始点云 (Nx3)
        :param rays_world: 世界坐标系下的射线方向 (Nx3)
        :param mesh: 用于扫描的mesh
        :return: 加入噪声后的点云和射线方向
        """
        noisy_rays_world = self.apply_angular_noise(rays_world)

        distances = np.linalg.norm(point_cloud - self.lidar.eye, axis=1)
        distance_noise = np.random.normal(0, self.distance_noise_std, size=distances.shape)
        noisy_distances = distances + distance_noise

        noisy_point_cloud = self.lidar.eye + noisy_rays_world * noisy_distances[:, np.newaxis]

        sampled_points, face_indices = mesh.sample(self.remap_sample_count, return_index=True)
        if self.remap_sample_count >= 1e6:
            _, _idx = knn.huge_point_cloud_nn(noisy_point_cloud, sampled_points, grid_length=0.5, expand_length=0.1)
        else:
            _, _idx = knn.minibatch_nn(noisy_point_cloud, sampled_points)
        normal = mesh.face_normals[face_indices[_idx]]

        return noisy_point_cloud, normal, noisy_rays_world
    
    def virtual_scan(self, 
                     mesh: trimesh.Trimesh) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        进行虚拟激光雷达扫描
        :param mesh: 三角网格模型
        :param use_noise: 是否使用噪声
        :param edit_normal: 是否更新法向量
        :return: 点云 (Nx3), 法向量 (Nx3), 射线方向 (Nx3)
        """
        _, rays_world = self.lidar.get_rays()
        origin = np.repeat([self.lidar.eye], rays_world.shape[0], axis=0)

        # 进行射线与mesh的相交计算
        index_triangle, index_ray, locations = mesh.ray.intersects_id(
            origin, rays_world, multiple_hits=False, return_locations=True
        )

        point_cloud = locations
        normal = mesh.face_normals[index_triangle]
        rays_direction_world = rays_world[index_ray]
        origin_dot = np.sum(normal * rays_direction_world, axis=1)

        return point_cloud, normal, rays_direction_world, index_triangle, origin_dot

    def virtual_scan_mitsuba(self, mesh: trimesh.Trimesh) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        进行虚拟激光雷达扫描
        :param mesh: 三角网格模型
        :param use_noise: 是否使用噪声
        :param edit_normal: 是否更新法向量
        :return: 点云 (Nx3), 法向量 (Nx3), 射线方向 (Nx3), 点乘结果 (Nx3)
        """

        _, direct_world = self.lidar.get_rays()
        origin = np.repeat([self.lidar.eye], direct_world.shape[0], axis=0)

        point_cloud, directions, normals = mitsuba_intersect(mesh, origin, direct_world)

        origin_dot = np.sum(normals * directions, axis=1)

        return point_cloud, normals, directions, origin_dot

    @staticmethod
    def direction_to_theta_phi(direction: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        将方向向量转换为极坐标系下的角度
        :param direction: 方向向量 (Nx3)
        :return: theta (水平方向角度), phi (垂直方向角度)
        """
        x, y, z = direction[:, 0], direction[:, 1], direction[:, 2]
        theta = np.arctan2(y, x)
        r = np.linalg.norm(direction[:, :2], axis=1)
        phi = np.arctan2(z, r)
        return theta, phi

    @staticmethod
    def theta_phi_to_direction(theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
        """
        将极坐标系下的角度转换为方向向量
        :param theta: 水平方向角度
        :param phi: 垂直方向角度
        :return: 方向向量 (Nx3)
        """
        x = np.cos(phi) * np.cos(theta)
        y = np.cos(phi) * np.sin(theta)
        z = np.sin(phi)
        return np.stack([x, y, z], axis=-1)
