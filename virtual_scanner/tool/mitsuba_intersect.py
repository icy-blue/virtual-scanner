import tempfile
from typing import Tuple, Any
import numpy as np
import trimesh
import mitsuba as mi
import open3d as o3d

mi.set_variant("cuda_ad_rgb")

def mitsuba_intersect(mesh: Any, origins: np.ndarray,
                      directions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    mitsuba 求交
    :param mesh: 三角网格模型
    :param origins: (N, 3) 射线起点
    :param directions: (N, 3) 射线方向
    :return: (position, direction, normal)
    """

    with tempfile.NamedTemporaryFile(suffix=".ply", delete=True) as temp_file:
        if isinstance(mesh, o3d.geometry.TriangleMesh):
            faces = np.asarray(mesh.triangles)
        elif isinstance(mesh, trimesh.Trimesh):
            faces = np.asarray(mesh.faces)
        else:
            raise NotImplementedError('Detected type', type(mesh))
        vertices = np.asarray(mesh.vertices)
        new_mesh = trimesh.Trimesh(vertices, faces)
        new_mesh.export(temp_file.name)

        # Mitsuba init
        scene = mi.load_dict({
            'type': 'scene',
            'shape': {
                'type': 'ply',
                'filename': temp_mesh_path
            }
        })

    ray = mi.Ray3f(o=mi.Point3f(origins.T.astype(np.float32)), d=mi.Vector3f(directions.T.astype(np.float32)))
    sis = scene.ray_intersect(ray)

    valid_mask = np.asarray(sis.is_valid())

    positions = np.asarray(sis.p).T[valid_mask]
    normals = np.asarray(sis.n).T[valid_mask]
    directions = directions[valid_mask]
    return positions, directions, normals