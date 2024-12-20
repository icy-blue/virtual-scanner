import os
import tempfile
from typing import Tuple
import numpy as np
import trimesh
import mitsuba as mi

mi.set_variant("cuda_ad_rgb")

def mitsuba_intersect(mesh: trimesh.Trimesh, origins: np.ndarray, directions: np.ndarray) \
        -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    mitsuba 求交
    :param mesh: 三角网格模型
    :param origins: (N, 3) 射线起点
    :param directions: (N, 3) 射线方向
    :return: (position, direction, normal)
    """

    with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as temp_file:
        temp_mesh_path = temp_file.name  # 获取临时文件路径
    mesh.export(temp_mesh_path)

    # Mitsuba init
    scene = mi.load_dict({
        'type': 'scene',
        'shape': {
            'type': 'ply',
            'filename': temp_mesh_path
        }
    })

    try:
        os.remove(temp_mesh_path)
    except Exception as e:
        print(f"Failed to delete temporary file: {e}")

    ray = mi.Ray3f(o=mi.Point3f(origins.T.astype(np.float32)), d=mi.Vector3f(directions.T.astype(np.float32)))
    sis = scene.ray_intersect(ray)

    valid_mask = np.asarray(sis.is_valid())

    positions = np.asarray(sis.p).T[valid_mask]
    normals = np.asarray(sis.n).T[valid_mask]
    directions = directions[valid_mask]
    return positions, directions, normals