import sys
import numpy as np
from virtual_scanner import LidarScanner
import open3d as o3d


def check_back_face(_points, _normals, _rays, o_dots, back_face_points):
    mask = o_dots > 0
    if back_face_points == 'filter':
        _points, _normals, _rays = _points[~mask], _normals[~mask], _rays[~mask]
    elif back_face_points == 'revert':
        if np.any(mask):
            print(f'Info: {np.sum(mask)} of {mask.size} normals have reverted.', file=sys.stderr)
            _normals[mask] *= -1
    elif back_face_points == 'ignore':
        if np.any(mask):
            print(f'Warning: {np.sum(mask)} of {mask.size} points have invalid dot product.', file=sys.stderr)
    else:
        raise ValueError(f'Invalid param of back_face_points: {back_face_points}')
    return _points, _normals, _rays


def add_to_pcd(_points, _normals, _rays, scanner, pcd_manager):
    _theta, _phi = LidarScanner.direction_to_theta_phi(_rays)
    points = _points.astype(np.float32)
    colors = (_rays / 2 + 0.5).astype(np.float32)
    normals = _normals.astype(np.float32)
    theta = _theta.astype(np.float32).reshape(-1, 1)
    phi = _phi.astype(np.float32).reshape(-1, 1)
    source = np.full((_points.shape[0], 1), scanner.scanner_id, dtype=np.int32)
    pcd_manager.add(
        positions=points,
        colors=colors,
        normals=normals,
        theta=theta,
        phi=phi,
        source=source
    )

def fast_visual_pc(xyz):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    o3d.visualization.draw_geometries([pcd])