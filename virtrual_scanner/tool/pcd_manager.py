import open3d as o3d
import numpy as np
from typing import List, Dict

class PointCloudManager:
    def __init__(self):
        self.point_cloud = {}

    def add(self, **kwargs):
        if len(self.point_cloud.keys()) == 0:
            for key, value in kwargs.items():
                self.point_cloud[key] = value
            return
        assert set(self.point_cloud.keys()) == set(kwargs.keys()), "Keys of point clouds are not the same"
        for key, value in kwargs.items():
            self.point_cloud[key] = np.vstack([self.point_cloud[key], value])
    
    def add_batch(self, point_clouds: 'List[Dict[str, np.ndarray]]'):
        if len(point_clouds) == 0:
            return
        if len(self.point_cloud.keys()) == 0:
            for key in point_clouds[0].keys():
                self.point_cloud[key] = np.vstack([point_cloud[key] for point_cloud in point_clouds])
            return
        assert set(self.point_cloud.keys()) == set(point_clouds[0].keys()), "Keys of point clouds are not the same"
        for key in self.point_cloud.keys():
            self.point_cloud[key] = np.vstack([self.point_cloud[key], *[point_cloud[key] for point_cloud in point_clouds]])
    
    def save(self, path: str):
        pcd = o3d.t.geometry.PointCloud()
        for key, value in self.point_cloud.items():
            dtype = o3d.core.Dtype.Float32 if value.dtype == np.float32 else o3d.core.Dtype.Int32
            pcd.point[key] = o3d.core.Tensor(value, dtype=dtype)
        o3d.t.io.write_point_cloud(path, pcd)
