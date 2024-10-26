import sys
import time
from typing import Tuple, Union, Callable, Sequence

import numpy as np

import torch
from pytorch3d.ops import knn_points
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm


class KNN:
    @classmethod
    def grid_partition(cls, points: torch.Tensor, grid_length: float) -> ('Tuple[torch.Tensor, torch.Tensor, '
                                                                          'torch.Tensor]'):
        device = points.device
        min_bounds = torch.floor(points.min(dim=0).values).to(device)
        max_bounds = torch.ceil(points.max(dim=0).values).to(device)

        coords = [torch.arange(torch.floor(min_bounds[i]).item(), torch.ceil(max_bounds[i]).item(),
                               grid_length) for i in range(3)]
        grid_x, grid_y, grid_z = torch.meshgrid(*coords, indexing='ij')
        grid_points = torch.stack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()], dim=1).to(device)

        return grid_points, min_bounds, max_bounds

    @classmethod
    def pc_to_tensor(cls, pc: any, device: torch.device) -> torch.Tensor:
        if isinstance(pc, np.ndarray):
            return torch.from_numpy(pc)[None, :, :].double().to(device)
        elif isinstance(pc, torch.Tensor):
            return pc[None, :, :].double().to(device)
        else:
            raise TypeError(f'Unknown type {type(pc)}')

    @classmethod
    def minibatch_nn(cls, pc0: any, pc1: any, verbose: bool = False, return_tensor=False) -> \
            'Union[Tuple[np.ndarray, np.ndarray], Tuple[torch.Tensor, torch.Tensor]]':
        """
        pc0 到 pc1 的最近邻 1nn
        """
        tick = time.time()
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        pc0_tensor, pc1_tensor = cls.pc_to_tensor(pc0, device), cls.pc_to_tensor(pc1, device)

        _knn = knn_points(pc0_tensor[:, :, :3], pc1_tensor[:, :, :3], K=1)
        dists, idx = _knn.dists, _knn.idx
        dists = torch.sqrt(dists).reshape(-1)
        idx = idx.reshape(-1)

        if not return_tensor:
            dists = dists.cpu().numpy()
            idx = idx.cpu().numpy()

        if verbose:
            print(f'1nn with shape {pc0.shape[0]}-{pc1.shape[0]} use time', time.time() - tick, file=sys.stderr)
        return dists, idx

    @classmethod
    def compute_mask(cls, positions: torch.Tensor, bound_min: torch.Tensor, bound_max: torch.Tensor) -> torch.Tensor:
        return (torch.all(positions[:, :3] >= bound_min, dim=1)) & (torch.all(positions[:, :3] < bound_max, dim=1))

    @classmethod
    def huge_point_cloud_nn(cls, pc0: any, pc1: any, grid_length: float = 10, expand_length: float = 0.5,
                            patch_size: int = 100000, verbose: bool = False) -> 'Tuple[np.ndarray, np.ndarray]':
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        pc0, pc1 = cls.pc_to_tensor(pc0, device)[0, :, :3], cls.pc_to_tensor(pc1, device)[0, :, :3]
        result = torch.zeros((pc0.shape[0], 2), dtype=torch.float32).to(device)
        result[:] = torch.tensor([1e5, -1]).to(device)

        grid_points, bound_min, bound_max = cls.grid_partition(pc0, grid_length)

        if verbose:
            print('bounding:', bound_min[:3].tolist(), bound_max[:3].tolist())
            print('grids length:', len(grid_points))

        with tqdm(total=pc0.shape[0]) as _t:
            for index, _ijk in enumerate(grid_points):
                _min = _ijk - expand_length
                _max = _ijk + grid_length + expand_length
                pc0_mask = torch.where(cls.compute_mask(pc0, _ijk, _ijk + grid_length))[0]
                sub_pc0 = pc0[pc0_mask]
                if sub_pc0.shape[0] == 0:
                    # print(f'pc0 in {list(_ijk)} has 0 points')
                    continue
                pc1_mask = torch.where(cls.compute_mask(pc1, _min, _max))[0]
                sub_pc1 = pc1[pc1_mask]
                if sub_pc1.shape[0] == 0:
                    _t.update(sub_pc0.shape[0])
                    # print(f'Warning; pc1 has no points in area {_min.tolist()} to {_max.tolist()}\n' +
                    #       f'while pc0 has {sub_pc0.shape[0]} points there.', file=sys.stderr)
                    continue
                if verbose:
                    print('processing:', _ijk.tolist())
                    tick = time.time()
                if sub_pc0.shape[0] > 5 * patch_size:
                    batch_result = torch.zeros([sub_pc0.shape[0], 2]).to(device)
                    batch_result[:] = torch.tensor([1e5, -1]).to(device)
                    cluster = MiniBatchKMeans(n_clusters=sub_pc0.shape[0] // patch_size, n_init='auto')
                    cluster.fit(sub_pc0[:, :3].cpu().numpy())
                    _label = torch.from_numpy(cluster.labels_).reshape(-1).to(device)
                    uniques = torch.unique(_label)
                    for _i in uniques:
                        _mask = torch.where(_label == _i)[0]
                        batch_pc0 = sub_pc0[_mask]
                        dists, idx = cls.minibatch_nn(batch_pc0, sub_pc1, return_tensor=True)
                        cover_mask = batch_result[_mask, 0] > dists.float()
                        batch_result[_mask[cover_mask]] = torch.cat((dists[cover_mask, None].float(),
                                                                     pc1_mask[idx[cover_mask, None]].float()), dim=1)
                        _t.update(batch_pc0.shape[0])
                    cover_mask = result[pc0_mask, 0] > batch_result[:, 0]
                    result[pc0_mask[cover_mask]] = batch_result[cover_mask]
                else:
                    dists, idx = cls.minibatch_nn(sub_pc0, sub_pc1, return_tensor=True)
                    result[pc0_mask] = torch.cat((dists[:, None].float(), pc1_mask[idx[:, None]].float()), dim=1)
                    _t.update(sub_pc0.shape[0])
                if verbose:
                    print(f'finish knn for shape {sub_pc0.shape[0]}-{sub_pc1.shape[0]} with time',
                          time.time() - tick)

        return result[:, 0].cpu().numpy(), result[:, 1].int().cpu().numpy()

    @classmethod
    def _minibatch_median(cls, pc: torch.Tensor, block_indices: torch.Tensor, neighbor_indices: torch.Tensor, k: int) \
            -> torch.Tensor:
        block_points = pc[block_indices, :3].unsqueeze(0)  # (1, B, 3)，其中 B 是 block 的点数
        neighbor_points = pc[neighbor_indices, :3].unsqueeze(0)  # (1, N, 3)，其中 N 是 neighbor 的点数
        neighbor_feats = pc[neighbor_indices, 3:]  # (N, m)，其中 m 是特征的维度

        knn_idx = knn_points(block_points, neighbor_points, K=k).idx.squeeze(0)  # (B, k)

        gathered_feats = neighbor_feats[knn_idx]  # (B, k, m)
        new_feats = torch.median(gathered_feats, dim=1)[0]  # (B, m)

        return new_feats

    @classmethod
    def huge_point_cloud_knn_mean(cls, pc: any, k: int = 20, grid_length: float = 10, expand_length: float = 2,
                                  verbose: bool = False) -> np.ndarray:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        pc = cls.pc_to_tensor(pc, device)[0]
        points, feats = pc[:, :3], pc[:, 3:]

        grid_points, bound_min, bound_max = cls.grid_partition(points, grid_length)
        new_feats = torch.zeros_like(feats, device=device)

        if verbose:
            print('bounding:', bound_min[:3].tolist(), bound_max[:3].tolist())
            print('grids length:', len(grid_points))

        with tqdm(total=pc.shape[0]) as _t:
            for grid_point in grid_points:
                block_mask = cls.compute_mask(points, grid_point, grid_point + grid_length)
                block_indices = torch.where(block_mask)[0]

                if block_indices.shape[0] > 0:
                    neighbors_mask = cls.compute_mask(points, grid_point - expand_length,
                                                      grid_point + grid_length + expand_length)
                    neighbor_indices = torch.where(neighbors_mask)[0]

                    new_feats[block_indices] = cls._minibatch_median(pc, block_indices, neighbor_indices, k)
                    _t.update(block_indices.shape[0])

        return new_feats.cpu().numpy()

    @classmethod
    def huge_point_cloud_kmeans_mean_mask(cls, pc: any, patch_size: int = 100000,
                                          calculate_split: 'Callable[[Sequence[float]], '
                                                           'Sequence[float]]' = None) -> np.ndarray:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        pc = cls.pc_to_tensor(pc, device)[0]
        points, feats = pc[:, :3], pc[:, 3:]

        kmeans = MiniBatchKMeans(n_clusters=int(pc.shape[0] / patch_size))
        kmeans.fit(points.cpu().numpy())
        labels = torch.from_numpy(kmeans.labels_).to(device)
        mask = torch.zeros_like(feats, device=device, dtype=torch.bool)

        for i in torch.unique(labels):
            block_feat = feats[labels == i]
            split = torch.mean(block_feat, dim=0)
            if calculate_split is not None:
                split = calculate_split(split)
            mask[labels == i] = block_feat >= split

        return mask.cpu().numpy()