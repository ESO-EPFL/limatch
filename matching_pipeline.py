import torch
import time
import numpy as np
import os
import psutil
from pathlib import Path
import copy

from lib import stats
from lib.io import create_project_folder
from lib.data_handling import Tile, Corres, concatenate_corres, build_output, prepare_overlap
from lib.utils_LCD import get_features, load_model, feat_search
from lib.keypoints import *
from lib.vis import vis_kpts
from lib.filter import ransac_filter, reciprocity_test
from lib.icp import refine_cor_icp
from lib.unit_test import run_tests

process = psutil.Process(os.getpid())

import logging
logger = logging.getLogger("LiMatch")
from lib.logger import log_sub, log_progress, get_logger

def run_pipeline(cloud1_path, cloud2_path, cfg):
    """
    Runs the full matching pipeline.
    """
    
    stats = {}
    t_0 = time.time()
    stats["metrics"] = {}
    
    logger = get_logger(cfg)
    logger.info("Starting LiMatch pipeline...")

    cfg['tile_id'] = f'{Path(cloud1_path).stem}_{Path(cloud2_path).stem}'
    create_project_folder(cfg['prj_folder'])

    logger.info(f"Visualization set to {cfg.get('visualize', False)}")

    # === LOAD MODEL ==================================================
    model, device = load_model(cfg)
    stats['time']['model_setup'] = time.time()


    # === PREPROCESSING ===============================================
    tile_a, tile_b = load_tiles(cloud1_path, cloud2_path, cfg)
    if cfg.get("coarse_to_fine", True):
        logger.info("Preprocessing coarse tiles...")
        cfg_coarse = coarse_config(cfg, factor=10)
        tile_a_c = copy.deepcopy(tile_a)
        tile_b_c = copy.deepcopy(tile_b)
        preprocess_tiles(tile_a_c, tile_b_c, cfg_coarse)
        logger.info("Preprocessing fine tiles...")

    preprocess_tiles(tile_a, tile_b, cfg)

    stats["metrics"]["Points A"] = tile_a.xyz.shape[0]
    stats["metrics"]["Points B"] = tile_b.xyz.shape[0]
    stats["metrics"]["Tiles num."] = int(np.max(tile_a.rsc_id)) if hasattr(tile_a, "rsc_id") else 1

    if cfg.get("coarse_to_fine", True):
        logger.info("Running coarse stage...")
        R, t = run_matching_stage(tile_a_c, tile_b_c, model, device, cfg_coarse, stats, coarse=True)

        logger.info("Applying coarse transform...")
        apply_transform_tile(tile_b, R, t)

    logger.info("Running fine stage...")

    corres_final, stats = run_matching_stage(tile_a, tile_b, model, device, cfg, stats, coarse=False)

    stats['time'] = time.time() - t_0
    logger.info(f"Total pipeline time: {stats['time']:.2f} seconds")
    return corres_final, stats

def run_matching_stage(tile_a, tile_b, model, device, cfg, stats, coarse=False):


    # === KEYPOINT DETECTION ==========================================
    detect_keypoints(tile_a, tile_b, cfg) 
    if not coarse:
        stats["metrics"]["Keypoints A"] = tile_a.kpts_id.shape[0]
        stats["metrics"]["Keypoints B"] = tile_b.kpts_id.shape[0]

    # === DESCRIPTION =================================================
    compute_descriptors(tile_a, tile_b, model, device, cfg)
    
    # === MATCHING ====================================================
    corres = match_features(tile_a, tile_b, cfg)
    if not coarse:
        stats["metrics"]["Raw matches"] = corres.idx_a.shape[0]

    # === RANSAC ======================================================
    rsc_tile_ids = np.unique(tile_a.rsc_id)
    corres_rsc = filter_matches(corres, rsc_tile_ids, cfg)
    if not coarse:
        stats["metrics"]["RANSAC matches"] = (
            corres_rsc.idx_a.shape[0] if corres_rsc is not None else 0
        )

    # === ICP =========================================================
    corres_icp = refine_icp(corres_rsc, tile_a, tile_b, cfg)

    if coarse:
        return estimate_helmert(corres_icp)
    
    else:
        stats["metrics"]["ICP matches"] = (
        corres_icp.idx_a.shape[0] if corres_icp is not None else 0
        )
        out_and_plot(corres, corres_rsc, corres_icp, tile_a, tile_b, cfg)

        logger.info("[Pipeline summary]")

        logger.info("[Metrics]")
        for k, v in stats["metrics"].items():
            logger.info(f"  {k:>15}: {v}")
        logger.info(f"Peak RAM: {process.memory_info().rss / (1024 ** 3):.2f} GB")
        logger.info("LiMatch pipeline completed.")
        return corres_icp, stats

def load_tiles(cloud1_path, cloud2_path, cfg):
    """
    Load Tile objects from file paths (txt or las/laz).
    Returns (tile_a, tile_b)
    """
    log_progress(logger, 1, "Preprocessing")
    log_sub(logger, "Loading point clouds...")
    if cfg['cloud_fmt'] == 'txt':
        tile_a = Tile.fromASCII(cloud1_path, cfg)
        tile_b = Tile.fromASCII(cloud2_path, cfg)
    elif cfg['cloud_fmt'] in ('las', 'laz'):
        tile_a = Tile.fromLAS(cloud1_path, cfg)
        tile_b = Tile.fromLAS(cloud2_path, cfg)
    else:
        raise ValueError(f"Unsupported cloud_fmt: {cfg['cloud_fmt']}")
    return tile_a, tile_b

def preprocess_tiles(tile_a, tile_b, cfg):
    """
    Preprocessing steps: overlap tiling, optional voxelization,
    and optional tile saving to csv. 
    """
    prepare_overlap(tile_a, tile_b, cfg)

    if cfg.get('vox_size', 0) > 0:
        log_sub(logger, f"Applying voxelization with size {cfg['vox_size']} m...")
        tile_a.apply_voxelization(cfg)
        tile_b.apply_voxelization(cfg)

    if cfg.get('save_tiles', False):
        log_sub(logger, "Saving tiles to csv...")
        np.savetxt(cfg['prj_folder'] + f"tiles/{cfg['tile_id']}_a.csv",
                   np.concatenate([tile_a.rsc_id.reshape(-1, 1), tile_a.xyz], axis=1),
                   delimiter=',')
        np.savetxt(cfg['prj_folder'] + f"tiles/{cfg['tile_id']}_b.csv",
                   np.concatenate([tile_b.rsc_id.reshape(-1, 1), tile_b.xyz], axis=1),
                   delimiter=',')

def detect_keypoints(tile_a, tile_b, cfg):
    """
    Detect ISS keypoints and set tile_a.kpts_id / tile_b.kpts_id.
    Applies optional downsampling and cleaning as in original script.
    """
    log_progress(logger, 2, "Keypoint detection")
    kpts_a = detect_keypoints_iss(tile_a.xyz, cfg)
    kpts_b = detect_keypoints_iss(tile_b.xyz, cfg)
    _, tile_a.kpts_id = tile_a.kdt.query(np.asarray(kpts_a.points), workers=-1)
    _, tile_b.kpts_id = tile_b.kdt.query(np.asarray(kpts_b.points), workers=-1)

    if 'max_kpts' in cfg and cfg['max_kpts'] is not None:
        log_sub(logger, f"Downsampling keypoints to max {cfg['max_kpts']} points...")
        downsample_keypoints(tile_a, cfg)
        downsample_keypoints(tile_b, cfg)

    log_sub(logger, f"Deleting kpts without candidate in radius 2*{cfg['uncertainty_r']}m...")
    delete_useless_keypoints(tile_a, tile_b, cfg)
    delete_useless_keypoints(tile_b, tile_a, cfg)

    if cfg.get('visualize', False):
        vis_kpts(tile_a, tile_b, kpts_a, kpts_b)

def compute_descriptors(tile_a, tile_b, model, device, cfg):
    """
    Compute LCD descriptors for both tiles.
    """
    log_progress(logger, 3, "Description")
    with torch.no_grad():
        log_sub(logger, "Computing descriptors for Tile A...")
        tile_a.feat = get_features(tile_a, model, device, cfg)
        log_sub(logger, "Computing descriptors for Tile B...")
        tile_b.feat = get_features(tile_b, model, device, cfg)
        assert getattr(tile_a, "kpts_id", None) is None or tile_a.feat.shape[0] == tile_a.kpts_id.shape[0]
        assert getattr(tile_b, "kpts_id", None) is None or tile_b.feat.shape[0] == tile_b.kpts_id.shape[0]

def match_features(tile_a, tile_b, cfg):
    """
    Candidate generation and feature-based nearest neighbor search.
    """
    log_progress(logger, 4, "Matching")
    get_candidates(tile_a, tile_b, cfg)
    get_candidates(tile_b, tile_a, cfg)

    tile_a.cor_id, feat_dist_a, _ = feat_search(tile_a, tile_b)
    tile_b.cor_id, _, _ = feat_search(tile_b, tile_a)
    tile_a.clear_features()
    tile_b.clear_features()

    corres = Corres.from_tiles(tile_a, tile_b, feat_dist_a)

    if cfg.get('reciprocity_test', False):
        log_sub(logger, "Applying reciprocity test...")
        mask = reciprocity_test(tile_a, tile_b)
        log_sub(logger, f"Keeping {100*np.sum(mask)/mask.shape[0]:.2f}% reciprocal matches")
        corres = corres.apply_mask(mask)

    tile_a.clear_candidates()
    tile_b.clear_candidates()

    return corres

def filter_matches(corres, tile_ids, cfg):
    """
    Apply per-tile RANSAC filtering exactly like the original script.
    Returns concatenated corres_rsc array.
    """
    log_progress(logger, 5, "RANSAC filtering")
    corres_rsc = None
    min_inliers = cfg.get("rsc_min_inliers", 50)

    for tid in tile_ids:
        

        tile_mask = (corres.rsc_id == tid)
        corres_tile = corres.apply_mask(tile_mask)

        idx = ransac_filter(corres_tile, cfg)

        if idx.shape[0] < min_inliers:
            log_sub(logger, f"WARNING Tile {tid}: only {idx.shape[0]} inliers < {min_inliers}, deleting...")
            continue
        elif idx.shape[0] > cfg.get('max_cor_per_tile', np.inf):
            log_sub(logger, f"Keeping max {cfg['max_cor_per_tile']} correspondences for tile {tid}...")
            rand = np.random.default_rng()
            idx = rand.choice(idx, size=cfg['max_cor_per_tile'], replace=False).reshape(-1,1)
        log_sub(logger, f"Tile {tid}: keeping {100*idx.shape[0]/corres_tile.idx_a.shape[0]:.2f}%") 
        filtered_tile = corres_tile.apply_mask(idx[:, 0])

        corres_rsc = filtered_tile if corres_rsc is None else concatenate_corres(corres_rsc, filtered_tile)
    if corres_rsc is None:
        log_sub(logger, "No matches survived RANSAC filtering.")
        return Corres.empty()
    else:
        return corres_rsc

def refine_icp(corres_rsc, tile_a, tile_b, cfg):
    """
    Run per-tile ICP refinement exactly like original corrICP usage.
    Returns corres_icp (concatenated per-tile arrays).
    """
    log_progress(logger, 6, "ICP refinement")
    if corres_rsc is None or corres_rsc.idx_a.size == 0:
        log_sub(logger, "No matches to refine with ICP.")
        return Corres.empty()

    corres_icp = None
    tile_ids = np.unique(tile_a.rsc_id)

    for tid in tile_ids:
        log_sub(logger, f"ICP refining tile {tid} with {np.sum(corres_rsc.rsc_id == tid)} correspondences...")

        tile_mask = (corres_rsc.rsc_id == tid)
        corres_tile = corres_rsc.apply_mask(tile_mask)

        corres_tile = refine_cor_icp(tile_a, tile_b, corres_tile, cfg)

        corres_icp = corres_tile if corres_icp is None else concatenate_corres(corres_icp, corres_tile)

    return corres_icp

def coarse_config(cfg, factor=10):
    """
    Estimate coarse matching configuration from curren config
    """

    changed_keys = [
        "vox_size",
        "iss_sln_r",
        "iss_nm_r",
        "iss_vox_s",
        "lcd_patch_r",
        "rsc_thr",
        "icp_patch_r",
        "icp_thresh",
        "icp_vox_s",
    ]

    coarse_config = cfg.copy()

    for k in changed_keys:
        if k in cfg and cfg[k] is not None and isinstance(cfg[k], (int, float)):
            coarse_config[k] = coarse_config[k] * factor
            logger.info(f"Coarse config: setting {k} from {cfg[k]} to {coarse_config[k]}")

    return coarse_config

def estimate_helmert(corres):
    """
    Estimate rigid transform (R, t) such that:
        x_A ≈ R x_B + t

    Parameters
    ----------
    corres : Corres
        Must contain xyz_a and xyz_b arrays of shape (N, 3)

    Returns
    -------
    R : (3,3) ndarray
    t : (3,) ndarray
    """

    if corres is None or corres.idx_a.size < 3:
        raise ValueError("Not enough correspondences to estimate transform.")

    X = corres.xyz_a   # target
    Y = corres.xyz_b   # source

    if X.shape[0] < 3:
        raise ValueError("At least 3 correspondences required.")

    mu_X = np.mean(X, axis=0)
    mu_Y = np.mean(Y, axis=0)

    Xc = X - mu_X
    Yc = Y - mu_Y

    H = Yc.T @ Xc

    U, S, Vt = np.linalg.svd(H)

    R = Vt.T @ U.T

    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    t = mu_X - R @ mu_Y

    return R, t

def apply_transform_tile(tile, R, t):
    """
    Applies rigid transform to tile in-place.
    """
    tile.xyz = (R @ tile.xyz.T).T + t

    if hasattr(tile, "kdt"):
        tile.build_kdtree()


def out_and_plot(corres, corres_rsc, corres_icp, tile_a, tile_b, cfg):
    """
    Wrap final output generation and stats building.
    """
    log_progress(logger, 7, "Building output")
    build_output(corres_icp, tile_a, tile_b, cfg)

    if cfg.get('run_tests', False):
        run_tests(corres_rsc, tile_a, tile_b, cfg)


    if cfg.get('save_stats', False):
        stats_raw = stats.compute_stats(corres, cfg['tile_id'])
        stats_rsc = stats.compute_stats(corres_rsc, cfg['tile_id']) 
        stats_icp = stats.compute_stats(corres_icp, cfg['tile_id'])

        if stats_raw is not None:
            stats.plot_stats(stats_raw, cfg['prj_folder'] + "plots/", f'raw_{cfg["tile_id"]}')
        if stats_rsc is not None:
            stats.plot_stats(stats_rsc, cfg['prj_folder'] + "plots/", f'rsc_{cfg["tile_id"]}')
        if stats_icp is not None:
            stats.plot_stats(stats_icp, cfg['prj_folder'] + "plots/", f'icp_{cfg["tile_id"]}')

        if stats_raw is not None and stats_rsc is not None and stats_icp is not None:
            stats.plot_final(stats_raw, stats_rsc, stats_icp, cfg)

    return True

if __name__ == "__main__":
    print("This is a module file. Please run the pipeline through:")
    print("1. from separate script. Example:")
    print("from matching_pipeline import run_pipeline")
    print("corres_icp, stats = run_pipeline('path/to/cloud1.txt', 'path/to/cloud2.txt', cfg)")
    print("2. from command line with main.py. Example")
    print("python main.py --c1 path/to/cloud1.txt --c2 path/to/cloud2.txt --y path/to/config.yaml")
    pass