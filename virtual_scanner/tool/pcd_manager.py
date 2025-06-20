import re
import sys
from collections import defaultdict
from os import PathLike
import warnings

import open3d as o3d
import numpy as np

import torch

if sys.version_info[1] >= 11:
    from typing import List, Dict, Optional, Self, Union
import os


class PointCloudManager:
    PCD_MAX_LENGTH = 2000_0000
    DEFAULT_KEYS = ['positions', 'normals', 'colors']

    def __init__(
        self, deduplication_precision: float = 1e-6, auto_deduplicate: bool = True, default_extension: str = ".pcd"
    ):
        self.point_cloud = {}
        self.lazy_list = []
        self.deduplication_precision = deduplication_precision
        self.auto_deduplicate = auto_deduplicate
        self.default_extension = default_extension

    @classmethod
    def _merge_parts(cls, data: dict):
        pattern = re.compile(r"^(?P<name>.+)_part(?P<index>\d+)$")
        merged_data = defaultdict(list)
        result = {}
        for key, value in data.items():
            match = pattern.match(key)
            if match:
                name = match.group("name")
                index = int(match.group("index"))
                merged_data[name].append((index, data[key]))
            else:
                result[key] = value

        for name, parts in merged_data.items():
            parts.sort()
            for k, v in parts:
                data.pop(f'{name}_part{k}')
            result[name] = np.hstack([value for _, value in parts])

        return result

    @classmethod
    def read_o3d_pcd(cls, path: 'Union[str, PathLike]', verbose: bool = True) -> 'Self':
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        if os.path.getsize(path) == 0:
            raise RuntimeError(f"File {path} is empty.")
        pcd = o3d.t.io.read_point_cloud(path)
        my_pcd = cls()
        pcd_dict = {}
        for key in pcd.point:
            pcd_dict[key] = pcd.point[key].numpy()
        pcd_dict = cls._merge_parts(pcd_dict)
        my_pcd.add(**pcd_dict)
        if verbose:
            print(f"Log: read keys {pcd_dict.keys()}")
        return my_pcd

    def merge(self, other: 'PointCloudManager') -> None:
        self.add(**other.point_cloud)

    def process_lazy(self) -> None:
        if len(self.lazy_list) == 0:
            return
        for item in self.lazy_list:
            self.add(lazy=False, **item)
        self.lazy_list.clear()

    @staticmethod
    def _parse_to_numpy(**kwargs) -> dict:
        for key, value in kwargs.items():
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().numpy()
            if value.dtype == np.float64 or key in PointCloudManager.DEFAULT_KEYS:
                value = value.astype(np.float32)
            kwargs[key] = value
        return kwargs

    def add(self, lazy: bool = True, **kwargs) -> None:
        kwargs = self._parse_to_numpy(**kwargs)
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

    def _check_shape(self, point_cloud: 'Dict[str, np.ndarray]') -> 'Dict[str, np.ndarray]':
        point_length = None
        for key, value in point_cloud.items():
            if len(value.shape) == 1:
                point_cloud[key] = value.reshape(-1, 1)
            if point_length is None:
                point_length = value.shape[0]
            else:
                assert (
                    point_length == value.shape[0]
                ), f"The number of points is not the same, {point_length} != {value.shape[0]}"
        return point_cloud

    def add_batch(self, point_clouds: 'List[Dict[str, np.ndarray]]') -> None:
        print('[Warning]: `add_batch()` is deprecated and will be removed.', file=sys.stderr)
        if len(point_clouds) == 0:
            return
        point_clouds = [self._check_shape(point_cloud) for point_cloud in point_clouds]
        if len(self.point_cloud.keys()) == 0:
            for key in point_clouds[0]:
                self.point_cloud[key] = np.vstack([point_cloud[key] for point_cloud in point_clouds])
            return
        assert set(self.point_cloud.keys()) == set(
            point_clouds[0].keys()
        ), f"Keys of point clouds are not the same, {self.point_cloud.keys()} != {point_clouds[0].keys()}"
        for key in self.point_cloud:
            self.point_cloud[key] = np.vstack(
                [self.point_cloud[key], *[point_cloud[key] for point_cloud in point_clouds]]
            )
        length = None
        for k, v in self.point_cloud:
            if length is None:
                length = v.shape[0]
            else:
                assert length == v.shape[0], f"The number of points is not the same, {length} != key {k} {v.shape[0]}"

    def to_o3d_tpcd(self, split: bool) -> 'o3d.t.geometry.PointCloud':
        pcd = o3d.t.geometry.PointCloud()
        for key, value in self.point_cloud.items():
            if key in self.DEFAULT_KEYS or value.shape[1] == 1:
                pcd.point[key] = o3d.core.Tensor(value)
                continue
            if not split:
                raise ValueError(
                    f'Open3D does not support multi-dimensional tensor (key {key} has shape {value.shape}) '
                    'except `positions`, `normals` and `colors`.'
                )
            for i in range(value.shape[1]):
                pcd.point[f'{key}_part{i}'] = o3d.core.Tensor(value[:, i : i + 1])
        return pcd

    def save(self, path: 'Union[str, PathLike]', split: bool = True) -> None:
        self.process_lazy()
        old_points = len(self)
        if self.auto_deduplicate:
            self.deduplicate()
            if len(self) != old_points:
                warnings.warn(
                    (f'Warning: {old_points - len(self)} of {old_points} points are duplicated '
                     f'(using precision {self.deduplication_precision}). Auto deduplicating...\n'
                     f'Filepath: {path}\n'
                     f'Set `auto_deduplicate=False` while initializing {self.__class__.__name__} to turn off.'),
                    UserWarning,
                )
        invalid_mask = self.find_invalid_mask()
        if np.sum(invalid_mask) > 0:
            warnings.warn(f'Warning: {np.sum(invalid_mask)} of {old_points} points are invalid.', UserWarning)
        ext = os.path.splitext(path)[1]
        if ext == '':
            path = path + self.default_extension
        if len(self) > self.PCD_MAX_LENGTH and ext == 'pcd':
            warnings.warn(
                (f'Warning: PointCloud with points {len(self)} is too large for Open3D.'
                  'Try `.ply` extension if error occurs.\nSee https://github.com/isl-org/Open3D/issues/6607'),
                UserWarning,
            )
        pcd = self.to_o3d_tpcd(split)
        o3d.t.io.write_point_cloud(path, pcd)

    def slice(self, indices: np.ndarray, update: bool = False) -> 'Self':
        self.process_lazy()
        if update:
            for key, value in self.point_cloud.items():
                self.point_cloud[key] = value[indices]
            return self
        sliced = PointCloudManager()
        for key, value in self.point_cloud.items():
            sliced.point_cloud[key] = value[indices]
        return sliced

    def to_simple_o3d_pcd(self) -> o3d.geometry.PointCloud:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.point_cloud['positions'])
        if 'colors' in self:
            pcd.colors = o3d.utility.Vector3dVector(self.point_cloud['colors'])
        if 'normals' in self:
            pcd.normals = o3d.utility.Vector3dVector(self.point_cloud['normals'])
        return pcd

    def __len__(self) -> int:
        self.process_lazy()
        if 'positions' not in self.point_cloud:
            return 0
        return len(self['positions'])

    def __getitem__(self, indices) -> 'Union[Self, np.ndarray]':
        self.process_lazy()
        if isinstance(indices, str):
            return self.point_cloud[indices]
        return self.slice(indices)

    def __setitem__(self, indices, value) -> None:
        self.process_lazy()
        if not isinstance(value, np.ndarray):
            value = np.array(value)
        if len(value.shape) == 1:
            value = value.reshape([-1, 1])
        elif len(value.shape) > 2:
            raise ValueError(f'Found invalid value (indices {indices}: shape {value.shape}).')
        if not isinstance(indices, str):
            raise ValueError(f'Indices must be a string but {type(indices)} is given')
        length = len(self)
        if length != value.shape[0]:
            raise ValueError(
                f'Value should be same length as point cloud, found {value.shape[0]} != pcd\' length {length}'
            )
        self.point_cloud[indices] = value

    def __contains__(self, item: any) -> bool:
        self.process_lazy()
        if isinstance(item, str):
            return item in self.point_cloud
        raise NotImplementedError(item)

    def __str__(self) -> str:
        self.process_lazy()
        return f'PointCloud({len(self)} points, keys: {", ".join(self.point_cloud.keys())})'

    @classmethod
    def from_simple_o3d_pcd(cls, pcd: 'o3d.geometry.PointCloud') -> 'PointCloudManager':
        manager = cls()
        manager.point_cloud['positions'] = np.asarray(pcd.points)
        if len(pcd.colors) > 0:
            manager.point_cloud['colors'] = np.asarray(pcd.colors)
        if len(pcd.normals) > 0:
            manager.point_cloud['normals'] = np.asarray(pcd.normals)
        return manager

    def deduplicate(self, precision: 'Optional[float]' = None) -> 'np.ndarray':
        if precision is None:
            precision = self.deduplication_precision
        xyz = self['positions']
        scaled_points = np.round(xyz / precision).astype(np.int64)
        _, indices = np.unique(scaled_points, axis=0, return_index=True)
        if len(indices) != len(self):
            self.slice(indices, update=True)
        return indices

    def remove_invalid_points(self) -> None:
        invalid_mask = self.find_invalid_mask()
        valid_mask = ~invalid_mask
        self.slice(valid_mask, update=True)

    def find_invalid_mask(self) -> 'np.ndarray':
        xyz = self['positions']
        invalid_mask = ~np.isfinite(xyz).all(axis=1)
        return invalid_mask
