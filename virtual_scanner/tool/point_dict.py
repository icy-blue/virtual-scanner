import numpy as np
import torch


class PointDict:
    def __init__(self, primary_key: str, _type: str = 'numpy'):
        self.data = {}
        self.meta_data = {}
        self.primary_key = primary_key
        self.type = _type

    def __getitem__(self, item):
        if isinstance(item, str):
            return self.data[item]
        return self.slice(item)

    def __setitem__(self, key, value):
        if self.type == 'numpy':
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().numpy()
            else:
                value = np.array(value)
        elif self.type == 'torch':
            if isinstance(value, np.ndarray):
                value = torch.from_numpy(value)
            else:
                value = torch.Tensor(value)
        self.data[key] = value

    def __len__(self):
        return len(self.data[self.primary_key])

    def __contains__(self, item):
        if isinstance(item, str):
            return self.primary_key in item
        try:
            _ = self.data[item]
            return True
        except:
            return False

    def slice(self, item):
        new = PointDict(self.primary_key)
        for k, v in self.data.items():
            new.data[k] = v[item]
        return new
