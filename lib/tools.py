import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation as R, Slerp
from pyproj import Transformer

from lib.georef import correct_laser_vector
from lib.data_handling import Corres, Tile
from lib.tools import *

import logging
logger = logging.getLogger("LiMatch")
from lib.logger import log_sub, log_sub_sub

def prepare_overlap(tile_a: Tile, tile_b: Tile, cfg):
    '''
    Prepare the data for tiling by filtering out non-overlapping sections and assigning a tile id to each point
    '''
    if cfg['tile']:
        log_sub(logger, f"Tiling with size {cfg['step_x']}x{cfg['step_y']}...")
        xyz_a = tile_a.xyz
        xyz_b = tile_b.xyz

        xmin = max(np.min(xyz_a[:, 0]), np.min(xyz_b[:, 0]))
        ymin = max(np.min(xyz_a[:, 1]), np.min(xyz_b[:, 1]))

        tile_id_a = np.concatenate((np.floor((xyz_a[:, 0]-xmin)/cfg['step_x']).reshape(-1, 1),
                                    np.floor((xyz_a[:, 1]-ymin)/cfg['step_y']).reshape(-1, 1)),
                                    axis=1)
        tile_id_b = np.concatenate((np.floor((xyz_b[:, 0]-xmin)/cfg['step_x']).reshape(-1, 1),
                                    np.floor((xyz_b[:, 1]-ymin)/cfg['step_y']).reshape(-1, 1)),
                                    axis=1)
        kept_id = 0
        kept_id_a = np.zeros((xyz_a.shape[0])).astype(np.uint8)
        kept_id_b = np.zeros((xyz_b.shape[0])).astype(np.uint8)

        for i in range(int(np.max(tile_id_a[:, 0])+1)):
            for j in range(int(np.max(tile_id_a[:, 1])+1)):
                mask_a = np.all([tile_id_a[:, 0] == i, tile_id_a[:, 1] == j], axis=0)
                mask_b = np.all([tile_id_b[:, 0] == i, tile_id_b[:, 1] == j], axis=0)

                den_a = np.sum(mask_a)/cfg['step_x']/cfg['step_y']
                den_b = np.sum(mask_b)/cfg['step_x']/cfg['step_y']

                if den_a > cfg['min_den'] and den_b > cfg['min_den']:
                    kept_id += 1
                    kept_id_a[mask_a] = kept_id
                    kept_id_b[mask_b] = kept_id
        tile_a.rsc_id = kept_id_a.astype(np.uint16)
        tile_b.rsc_id = kept_id_b.astype(np.uint16)

        tile_a.apply_mask(kept_id_a > 0)
        tile_b.apply_mask(kept_id_b > 0)

    else:
        kept_id_a = np.zeros((tile_a.xyz.shape[0],), dtype=np.uint8)
        kept_id_b = np.zeros((tile_b.xyz.shape[0],), dtype=np.uint8)
        log_sub(logger, "No tiling... (all points kept)")

    shift = tile_a.xyz.mean(axis=0)

    tile_a.shift = shift
    tile_b.shift = shift

    log_sub(logger, f"Shifting point clouds toward origin...")
    with np.printoptions(precision=2, suppress=True):
        log_sub_sub(logger, f"{shift.flatten()} m...")
    log_sub_sub(logger, f"(Coordinates shifted back at export)")
    tile_a.xyz = (tile_a.xyz - shift).astype(np.float32)
    tile_b.xyz = (tile_b.xyz - shift).astype(np.float32)

    tile_a.rebuild_kdt()
    tile_b.rebuild_kdt()

    try:
        max_tile = int(np.max(kept_id_a))
    except Exception:
        max_tile = 0

    log_sub(logger, f"Generated {max_tile} valid tiles.")

def build_corres_file(c: Corres, tile_a: Tile, tile_b: Tile, cfg):
    """
    Build and save correspondences file for RANSAC and ICP stages.
    """
    if c is None or c.idx_a.shape[0] == 0:
        log_sub(logger, "No correspondences to save.")
        return
    idx_a = c.idx_a.astype(int)
    idx_b = c.idx_b.astype(int)
    time_a = tile_a.time[idx_a].reshape(-1, 1)
    time_b = tile_b.time[idx_b].reshape(-1, 1)

    icp_vec = c.icp_vec

    if cfg.get("simulateLasVec", False):
        log_sub(logger, "Simulating laser vectors from trajectory...")
        las_vec_a, las_vec_b = simulate_las_vec(
            time_a.reshape(-1),
            time_b.reshape(-1),
            tile_a.xyz[idx_a]+ tile_a.shift,
            tile_b.xyz[idx_b]+ tile_b.shift,
            cfg['trj_path'],
            cfg['R_sensor2body'],
            np.array(cfg.get('lever_arm', [0.0, 0.0, 0.0])),
            cfg['point_epsg'])
    else:
        log_sub(logger, "Fetching laser vectors from input...")
        las_vec_a = tile_a.las_vec[idx_a]
        las_vec_b = tile_b.las_vec[idx_b]


    if cfg['adjustLasVec']:
        out_data = np.concatenate((time_b, time_a, las_vec_b, las_vec_a), axis=1)
        out_data = out_data[out_data[:, 0].argsort()]
        np.savetxt(cfg['prj_folder'] + f"cor_outputs/LiDAR_p2p_noRefinement_{cfg['tile_id']}.txt",
                    out_data,
                    delimiter=',',
                    fmt='%.9f, %.9f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f')
        
        trj = np.loadtxt(cfg['trj_path'], delimiter=',', skiprows=1)
        trj_t = trj[:, 0]
        trj_q = trj[:, 4:]
        trj_q = trj_q[:, [1, 2, 3, 0]]  # w,x,y,z -> x,y,z,w according to Scipy notation

        las_vec_a = correct_laser_vector(time_a.reshape(-1), las_vec_a, cfg['R_sensor2body'], trj_t, trj_q, icp_vec, None)
        out_data = np.concatenate((time_b, time_a, las_vec_b, las_vec_a), axis=1)
        out_data = out_data[out_data[:, 0].argsort()]
        np.savetxt(cfg['prj_folder'] + f"cor_outputs/LiDAR_p2p_{cfg['tile_id']}.txt",
                    out_data,
                    fmt='%.9f, %.9f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f')

    else:
        xyz_a = tile_a.xyz[idx_a] + tile_a.shift
        xyz_b = tile_b.xyz[idx_b] + tile_b.shift
     
        out_data = np.concatenate((time_b, time_a, xyz_b, xyz_a, icp_vec), axis=1)
        out_data = out_data[out_data[:, 0].argsort()]
        np.savetxt(cfg['prj_folder'] + f"cor_outputs/LiDAR_p2p.txt",
                    out_data,
                    fmt='%.9f, %.9f,  %.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f',
                    header='time_b, time_a, x_b, y_b, z_b, x_a, y_a, z_a, icp_x, icp_y, icp_z (xyz_a - icp_vec_a = refined xyz_a)')

def simulate_las_vec(time_a, time_b, xyz_a, xyz_b, trj_path, R_s2b, a_s, point_epsg):
    """
    Simulate laser vectors from trajectory (in ECEF) and tile points (any EPSG).

    """
    R_s2b = np.array(R_s2b)
    trj = np.loadtxt(trj_path, delimiter=',', skiprows=1)
    trj_t = trj[:, 0]
    trj_xyz = trj[:, 1:4]             # ECEF
    trj_q = trj[:, 4:8]               # qw, qx, qy, qz
    trj_q = trj_q[:, [1, 2, 3, 0]]  # w,x,y,z -> x,y,z,w according to Scipy notation
    rots = R.from_quat(trj_q)         # scipy expects (x,y,z,w)

    pos_interp = interp1d(trj_t, trj_xyz, axis=0, kind="linear", fill_value="extrapolate")
    slerp = Slerp(trj_t, rots)

    transformer = Transformer.from_crs(point_epsg, "epsg:4978", always_xy=True)

    ecef_a = np.vstack(transformer.transform(xyz_a[:, 0], xyz_a[:, 1], xyz_a[:, 2])).T
    ecef_b = np.vstack(transformer.transform(xyz_b[:, 0], xyz_b[:, 1], xyz_b[:, 2])).T

    las_vec_a = np.zeros_like(ecef_a)
    las_vec_b = np.zeros_like(ecef_b)

    for i, (ta, tb, pa, pb) in enumerate(zip(time_a, time_b, ecef_a, ecef_b)):
        Ra = slerp([ta])[0].as_matrix()
        Rb = slerp([tb])[0].as_matrix()
        pos_a = pos_interp(ta)
        pos_b = pos_interp(tb)

        inner_a = Ra.T @ (pa - pos_a) - a_s
        inner_b = Rb.T @ (pb - pos_b) - a_s
        las_vec_a[i, :] = R_s2b.T @ inner_a
        las_vec_b[i, :] = R_s2b.T @ inner_b

    return las_vec_a, las_vec_b