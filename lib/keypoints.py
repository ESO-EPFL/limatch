import open3d as o3d
import numpy as np
from scipy.spatial import KDTree

from lib.data_handling import Tile

def detect_keypoints_iss(xyz,cfg):
    """
    Compute keypoints from a point cloud
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)

    if cfg['iss_vox_s'] > 0:
        pcd = pcd.voxel_down_sample(cfg['iss_vox_s'])

    return o3d.geometry.keypoint.compute_iss_keypoints(pcd,
                                                    salient_radius=cfg['iss_sln_r'],
                                                    non_max_radius=cfg['iss_nm_r'],
                                                    gamma_21=cfg['iss_g21'],
                                                    gamma_32=cfg['iss_g32'],
                                                    min_neighbors=cfg['iss_min_n'])

def downsample_keypoints(tile: Tile, cfg):
    ''' 
    Downsample keypoints if number of keypoints per tile is above max value from config
    '''
    filt_kpts_id = []
    
    print(f"Initial kpts number: {len(tile.kpts_id)}")
    print(f"Filtering keypoints to {cfg['max_kpts']} per tile...")
    for i in np.unique(tile.rsc_id):
        
        kpts_id_tile_i = tile.kpts_id[tile.rsc_id[tile.kpts_id] == i]
        if len(kpts_id_tile_i) > cfg['max_kpts']:
            kpts_id_tile_i = np.random.choice(kpts_id_tile_i, cfg['max_kpts'], replace=False)
        filt_kpts_id.extend(kpts_id_tile_i)

    tile.kpts_id = np.array(filt_kpts_id)
    tile.n_kpts = len(tile.kpts_id)

def delete_useless_keypoints(tile_key: Tile, tile_target: Tile, cfg):
    """
    Check if keypoints have at least one target keypoint in vicinity
    If not, delete it
    """
 
    kdt = KDTree(tile_target.xyz[tile_target.kpts_id])
    dist, _ = kdt.query(tile_key.xyz[tile_key.kpts_id], workers=-1)

    tile_key.kpts_id = tile_key.kpts_id[dist < 2*cfg['uncertainty_r']]
    tile_key.n_kpts = len(tile_key.kpts_id)
    print(f"Final kpts number: {tile_key.n_kpts}")

def get_candidates(tile_key: Tile, tile_target: Tile, cfg):
    """
    Generate list of candidates for each keypoint to match,
    using batching and compact NumPy arrays to save memory.
    """

    kdt = KDTree(tile_target.xyz[tile_target.kpts_id])
    query_pts = tile_key.xyz[tile_key.kpts_id]

    batch_size = cfg["main_batch"]  # configurable batch size
    radius = 2 * cfg["uncertainty_r"]

    candidates = []
    for start in range(0, query_pts.shape[0], batch_size):
        end = min(start + batch_size, query_pts.shape[0])
        # Run query_ball_point on this batch
        batch_res = kdt.query_ball_point(query_pts[start:end], radius, workers=-1)
        # Convert each sublist into a compact NumPy array
        batch_res = [np.array(r, dtype=np.uint32) for r in batch_res]
        candidates.extend(batch_res)

    tile_key.candidates = candidates