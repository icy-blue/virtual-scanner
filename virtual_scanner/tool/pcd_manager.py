import open3d as o3d
import numpy as np
import math
from typing import List, Dict

class PointCloudManager:
    def __init__(self, split_length=5000_0000):
        self.point_cloud = {}
        self.split_length = split_length

    @classmethod
    def read_o3d_pcd(cls, path: str):
        pcd = o3d.t.io.read_point_cloud(path)
        my_pcd = cls()
        pcd_dict = {}
        for key in pcd.point:
            pcd_dict[key] = pcd.point[key].numpy()
        my_pcd.add(**pcd_dict)
        return my_pcd

    def merge(self, point_cloud: 'PointCloudManager'):
        self.add(**point_cloud.point_cloud)

    def add(self, **kwargs):
        self._check_shape(kwargs)
        if len(self.point_cloud.keys()) == 0:
            for key, value in kwargs.items():
                self.point_cloud[key] = value
            return
        assert set(self.point_cloud.keys()) == set(kwargs.keys()), "Keys of point clouds are not the same"
        for key, value in kwargs.items():
            self.point_cloud[key] = np.vstack([self.point_cloud[key], value])
    
    def _check_shape(self, point_cloud: Dict[str, np.ndarray]):
        point_length = None
        for key, value in point_cloud.items():
            if len(value.shape) == 1:
                point_cloud[key] = value.reshape(-1, 1)
            if point_length is None:
                point_length = value.shape[0]
            else:
                assert point_length == value.shape[0], f"The number of points is not the same, {point_length} != {value.shape[0]}"
        return point_cloud
    
    def add_batch(self, point_clouds: 'List[Dict[str, np.ndarray]]'):
        if len(point_clouds) == 0:
            return
        point_clouds = [self._check_shape(point_cloud) for point_cloud in point_clouds]
        if len(self.point_cloud.keys()) == 0:
            for key in point_clouds[0].keys():
                self.point_cloud[key] = np.vstack([point_cloud[key] for point_cloud in point_clouds])
            return
        assert set(self.point_cloud.keys()) == set(point_clouds[0].keys()), f"Keys of point clouds are not the same, {self.point_cloud.keys()} != {point_clouds[0].keys()}"
        for key in self.point_cloud.keys():
            self.point_cloud[key] = np.vstack([self.point_cloud[key], *[point_cloud[key] for point_cloud in point_clouds]])
        length = None
        for key in self.point_cloud.keys():
            if length is None:
                length = self.point_cloud[key].shape[0]
            else:
                assert length == self.point_cloud[key].shape[0], f"The number of points is not the same, {length} != key {key} {self.point_cloud[key].shape[0]}"
    
    def save(self, path: str, split: bool = True):
        path = path[:-4] if path.endswith('.pcd') else path
        pcd = o3d.t.geometry.PointCloud()
        block_num = math.ceil(len(self.point_cloud['positions']) / self.split_length)
        if not split or block_num == 1:
            for key, value in self.point_cloud.items():
                pcd.point[key] = o3d.core.Tensor(value)
            o3d.t.io.write_point_cloud(f"{path}.pcd", pcd)
            return
        for i in range(block_num):
            start = i * self.split_length
            end = min((i + 1) * self.split_length, len(self.point_cloud['positions']))
            for key, value in self.point_cloud.items():
                pcd.point[key] = o3d.core.Tensor(value[start:end])
            o3d.t.io.write_point_cloud(f"{path}_block{i}.pcd", pcd)
