import numpy as np
import torch
from .pcd_manager import PointCloudManager


class PointDict:
    def __init__(self, primary_key: str, _type: str = 'numpy'):
        """
        :param primary_key: 主键，用于对接 PointCloudManager 及计算点数
        :param _type: 可选 `numpy` `torch` `torch-cuda`
        """
        self.data = {}
        self.meta_data = {}
        self.primary_key = primary_key
        self.type = _type

    def __getitem__(self, item):
        if isinstance(item, str):
            if item in self.data:
                return self.data[item]
            return self.meta_data[item]
        return self.slice(item)

    def __setitem__(self, key, value):
        if isinstance(value, str) or np.asarray(value).size <= 1:
            self.meta_data[key] = value
            return
        if self.type == 'numpy':
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().numpy()
            elif not isinstance(value, np.ndarray):
                value = np.array(value)
        elif self.type[:5] == 'torch':
            if isinstance(value, np.ndarray):
                value = torch.from_numpy(value)
            elif not isinstance(value, torch.Tensor):
                value = torch.Tensor(value)
            if self.type == 'torch-cuda':
                value = value.cuda()
        self.data[key] = value

    def __len__(self):
        if self.primary_key not in self.data:
            return 0
        return len(self.data[self.primary_key])

    def __contains__(self, item):
        if isinstance(item, str):
            return item in self.data or item in self.meta_data
        return False

    def length_check(self):
        assert self.primary_key in self.data
        point_length = len(self.data[self.primary_key])
        for k, v in self.data.items():
            if v.shape[0] != point_length:
                print(f'Length check failed for {k}: {v.shape[0]}, expected {point_length}.', file=sys.stderr)
                return False
        return True

    def slice(self, item):
        new = PointDict(self.primary_key)
        for k, v in self.data.items():
            new.data[k] = v[item]
        return new

    def change_type(self, new_type):
        assert new_type in ['numpy', 'torch', 'torch-cuda']
        if new_type == self.type:
            return
        self.type = new_type
        for k, v in self.data.items():
            self[k] = v

    def to_pcd_manager(self):
        manager = PointCloudManager()
        pos = self[self.primary_key]
        self.data.pop(self.primary_key)
        self['positions'] = pos
        manager.add(**self.data, lazy=False)
        return manager

    @classmethod
    def from_pcd_manager(cls, manager: 'PointCloudManager'):
        point_dict = cls(primary_key='positions')
        manager.process_lazy()
        for k, v in manager.point_cloud.items():
            point_dict[k] = v
        return point_dict
