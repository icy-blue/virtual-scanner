import sys
import numpy as np
import torch
from .pcd_manager import PointCloudManager
if sys.version_info[1] >= 9:
    from typing import List, Self


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
        if self._get_size(value) <= 1:
            self.meta_data[key] = value
            return
        value = self._parse_value(value, self.type)
        self.data[key] = value

    def __len__(self):
        if self.primary_key not in self.data:
            return 0
        return len(self.data[self.primary_key])

    def __contains__(self, item):
        if isinstance(item, str):
            return item in self.data or item in self.meta_data
        return False

    def __str__(self) -> str:
        return(f'PointDict with primary key: {self.primary_key}, length: {len(self)}, type: {self.type}\n'
               f'Data keys: {self.data.keys()}, meta keys: {self.meta_data.keys()}\n')

    @staticmethod
    def _parse_value(value: 'Any', type_: 'str'):
        if type_ == 'numpy':
            if isinstance(value, torch.Tensor):
                return value.detach().cpu().numpy()
            if isinstance(value, np.ndarray):
                return value
            return np.array(value)
        if type_[:5] == 'torch':
            if isinstance(value, np.ndarray):
                value = torch.from_numpy(value)
            elif not isinstance(value, torch.Tensor):
                value = torch.Tensor(value)
            if type_ == 'torch-cuda':
                value = value.cuda()
            return value
        raise TypeError(f'Unsupported type: {type_}')


    def _get_size(self, item):
        if isinstance(item, str):
            return 1
        if isinstance(item, torch.Tensor):
            return item.numel()
        return np.asarray(item).size

    def gets(self, *args, type_: 'str' = None):
        result = [self.__getitem__(x) for x in args]
        if type_ is not None and type_ != self.type:
            result = [self._parse_value(v, type_) for v in result]
        return result

    def pop(self, *args):
        for x in args:
            if x in self.data:
                self.data.pop(x)
            elif x in self.meta_data:
                self.meta_data.pop(x)
            else:
                raise KeyError(x)

    def pop_except(self, *args):
        for k in self.data:
            if k not in args:
                self.data.pop(k)
        for k in self.meta_data:
            if k not in args:
                self.meta_data.pop(k)

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
        for k, v in self.meta_data.items():
            new.meta_data[k] = v[item] if isinstance(v, (np.ndarray, torch.Tensor, list)) else v
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

    @classmethod
    def collate(cls, batch: 'List[PointDict]') -> 'Self':
        assert len(batch) > 0
        first = batch[0]
        result = cls(primary_key=first.primary_key, _type=first.type)
        for k in first.data.keys():
            values = [info[k] for info in batch]
            if first.type == 'numpy':
                value = np.array(values)
            else:
                value = torch.stack(values, dim=0)
            if first.type == 'torch-cuda':
                value = value.cuda()
            result[k] = value
        for k in first.meta_data:
            result.meta_data[k] = [point_dict.meta_data[k] for point_dict in batch]
        return result