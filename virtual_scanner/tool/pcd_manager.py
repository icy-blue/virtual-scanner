import open3d as o3d
import numpy as np
import math
from typing import List, Dict
import os


class PointCloudManager:
    def __init__(self, split_length=5000_0000, deduplication_precision=1e-6):
        self.point_cloud = {}
        self.split_length = split_length
        self.lazy_list = []
        self.deduplication_precision = deduplication_precision

    @classmethod
    def read_o3d_pcd(cls, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError
        if os.path.getsize(path) == 0:
            raise RuntimeError(f"File {path} is empty.")
        pcd = o3d.t.io.read_point_cloud(path)
        my_pcd = cls()
        pcd_dict = {}
        for key in pcd.point:
            pcd_dict[key] = pcd.point[key].numpy()
        my_pcd.add(**pcd_dict)
        print(f"Log: read keys {pcd_dict.keys()}")
        return my_pcd

    def merge(self, point_cloud: 'PointCloudManager'):
        self.add(**point_cloud.point_cloud)

    def lazy_process(self):
        if len(self.lazy_list) == 0:
            return
        for item in self.lazy_list:
            self.add(lazy=False, **item)
        self.lazy_list = []

    def add(self, lazy=True, **kwargs):
        self._check_shape(kwargs)
        if lazy and len(self.point_cloud.keys()) != 0:
            self.lazy_list.append(kwargs)
            return
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
                assert point_length == value.shape[0], \
                    f"The number of points is not the same, {point_length} != {value.shape[0]}"
        return point_cloud

    def add_batch(self, point_clouds: 'List[Dict[str, np.ndarray]]'):
        if len(point_clouds) == 0:
            return
        point_clouds = [self._check_shape(point_cloud) for point_cloud in point_clouds]
        if len(self.point_cloud.keys()) == 0:
            for key in point_clouds[0].keys():
                self.point_cloud[key] = np.vstack([point_cloud[key] for point_cloud in point_clouds])
            return
        assert set(self.point_cloud.keys()) == set(point_clouds[0].keys()), \
            f"Keys of point clouds are not the same, {self.point_cloud.keys()} != {point_clouds[0].keys()}"
        for key in self.point_cloud.keys():
            self.point_cloud[key] = np.vstack([self.point_cloud[key],
                                               *[point_cloud[key] for point_cloud in point_clouds]])
        length = None
        for key in self.point_cloud.keys():
            if length is None:
                length = self.point_cloud[key].shape[0]
            else:
                assert length == self.point_cloud[key].shape[0], \
                    f"The number of points is not the same, {length} != key {key} {self.point_cloud[key].shape[0]}"

    def save(self, path: str, split: bool = True):
        if len(self.lazy_list) != 0:
            self.lazy_process()
        self.deduplicate()
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

    def slice(self, indices: np.ndarray, update: bool = False):
        if update:
            for key, value in self.point_cloud.items():
                self.point_cloud[key] = value[indices]
            return
        sliced = PointCloudManager()
        for key, value in self.point_cloud.items():
            sliced.point_cloud[key] = value[indices]
        return sliced

    def to_simple_o3d_pcd(self) -> o3d.geometry.PointCloud:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.point_cloud['positions'])
        if 'colors' in self.point_cloud:
            pcd.colors = o3d.utility.Vector3dVector(self.point_cloud['colors'])
        if 'normals' in self.point_cloud:
            pcd.normals = o3d.utility.Vector3dVector(self.point_cloud['normals'])
        return pcd

    def __len__(self):
        return len(self['positions'])

    def __getitem__(self, indices):
        if len(self.lazy_list) != 0:
            self.lazy_process()
        if isinstance(indices, str):
            return self.point_cloud[indices]
        return self.slice(indices)

    @classmethod
    def from_simple_o3d_pcd(cls, pcd: o3d.geometry.PointCloud) -> 'PointCloudManager':
        manager = cls()
        manager.point_cloud['positions'] = np.asarray(pcd.points)
        if len(pcd.colors) > 0:
            manager.point_cloud['colors'] = np.asarray(pcd.colors)
        if len(pcd.normals) > 0:
            manager.point_cloud['normals'] = np.asarray(pcd.normals)
        return manager

    def deduplicate(self, precision=None):
        if precision is None:
            precision = self.deduplication_precision
        xyz = self['positions']
        scaled_points = np.round(xyz / precision).astype(np.int64)
        _, indices = np.unique(scaled_points, axis=0, return_index=True)
        self.slice(indices, update=True)

    def remove_invalid_points(self):
        xyz = self['positions']
        valid_mask = np.isfinite(xyz).all(axis=1)
        self.slice(valid_mask, update=True)
