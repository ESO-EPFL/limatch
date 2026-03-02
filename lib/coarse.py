import numpy as np
import copy

from lib.rotations import R1, R2, R3

from lib.pipeline_blocks import (
    detect_keypoints,
    compute_descriptors,
    match_features,
    filter_matches
)
import logging
logger = logging.getLogger("LiMatch")
from lib.logger import log_sub, log_progress

def run_coarse_bootstrap(tile_a, tile_b, model, device, cfg):

    log_progress(logger, 0, "Running coarse bootstrap...")

    coarse_cfg = make_coarse_cfg(cfg)

    tile_a_c = tile_a.copy_minimal()
    tile_b_c = tile_b.copy_minimal()

    R_true, t_true = emulate_rotations(tile_a_c)

    if coarse_cfg.get("vox_size", 0) > 0:
        tile_a_c.apply_voxelization(coarse_cfg)
        tile_b_c.apply_voxelization(coarse_cfg)

    detect_keypoints(tile_a_c, tile_b_c, coarse_cfg)

    compute_descriptors(tile_a_c, tile_b_c, model, device, coarse_cfg)

    corres = match_features(tile_a_c, tile_b_c, coarse_cfg)

    tile_ids = np.unique(tile_a_c.rsc_id)
    corres_rsc = filter_matches(corres, tile_ids, coarse_cfg)

    if corres_rsc.idx_a.size < 5:
        log_sub(logger, "Not enough coarse matches.")
        return None

    A = corres_rsc.xyz_a
    B = corres_rsc.xyz_b

    R, t = estimate_rigid_transform(A, B)

    log_sub(logger, f"Coarse matches: {corres_rsc.idx_a.size}")
    log_sub(logger, "Coarse transform applied.")

    init_res = np.linalg.norm(A - B, axis=1)
    
    A_aligned = (R @ A.T).T + t
    adj_res = np.linalg.norm(A_aligned - B, axis=1)
    
    log_sub(logger, f"Initial coarse residual: {init_res.mean():.2f} m")
    log_sub(logger, f"Adjusted coarse residual: {adj_res.mean():.2f} m")

    R_err = R @ R_true.T
    angle = np.degrees(np.arccos((np.trace(R_err)-1)/2))
    log_sub(logger, f"Coarse rotation error: {angle:.2f} degrees")
    
    return R, t

def make_coarse_cfg(cfg):

    c = cfg.copy()
    coarse = cfg.get("coarse", {})

    if not coarse.get("enabled", False):
        return c

    # voxel
    c["vox_size"] = coarse.get("vox_size", cfg.get("vox_size"))

    # ISS radii
    c["iss_sln_r"] = coarse.get("iss_sln_r", cfg.get("iss_sln_r"))
    c["iss_nm_r"]  = coarse.get("iss_nm_r",  cfg.get("iss_nm_r"))

    # ISS ratios
    c["iss_g21"] = coarse.get("iss_g21", cfg.get("iss_g21"))
    c["iss_g32"] = coarse.get("iss_g32", cfg.get("iss_g32"))

    # descriptor
    c["lcd_patch_r"] = coarse.get("lcd_patch_r", cfg.get("lcd_patch_r"))
    c["uncertainty_r"] = coarse.get("uncertainty_r", cfg.get("uncertainty_r"))

    # RANSAC
    c["rsc_thr"] = coarse.get("rsc_thr", cfg.get("rsc_thr"))
    c["rsc_min_inliers"] = coarse.get("ransac_min_inliers", 30)

    assert c["rsc_thr"] is not None
    assert c["iss_g21"] is not None
    assert c["vox_size"] is not None

    return c

def estimate_rigid_transform(A, B):
    centroid_A = A.mean(axis=0)
    centroid_B = B.mean(axis=0)

    AA = A - centroid_A
    BB = B - centroid_B

    H = AA.T @ BB
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    if np.linalg.det(R) < 0:
        Vt[2,:] *= -1
        R = Vt.T @ U.T

    t = centroid_B - R @ centroid_A
    log_sub(logger, f"SVD singular values: {S}")
    log_sub(logger, f"Estimated coarse rotation:\n{R}")
    log_sub(logger, f"Estimated coarse translation: {t}")

    angle = np.arccos((np.trace(R) - 1) / 2)
    log_sub(logger, f"Rotation magnitude: {np.degrees(angle):.3f} deg")
    log_sub(logger, f"Translation magnitude: {np.linalg.norm(t):.3f} m")

    return R, t

def emulate_rotations(tile, rpy = [3,3,3], t=np.array([0,0,0])):
    log_sub(logger, f"Emulating rotations on tile with roll={rpy[0]}, pitch={rpy[1]}, yaw={rpy[2]} (degrees)")
    log_sub(logger, f"and translation {t} (meters)")

    r = np.radians(rpy[0])
    p = np.radians(rpy[1])
    y = np.radians(rpy[2])

    R = R1(r) @ R2(p) @ R3(y)

    log_sub(logger, f"Emulated rotation matrix:\n{R}")

    tile.apply_transform(R, t)

    return R, t