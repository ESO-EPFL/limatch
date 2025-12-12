import torch
import time
import numpy as np
import os
import psutil
from pathlib import Path

from lib import stats
from lib.io import create_project_folder
from lib.data_handling import Tile, Corres, concatenate_corres
from lib.utils_LCD import get_features, load_model, feat_search
from lib.tools import *
from lib.keypoints import *
from lib.vis import vis_kpts
from lib.filter import ransac_filter, reciprocity_test
from lib.icp import refine_cor_icp

process = psutil.Process(os.getpid())

def run_pipeline(cloud1_path, cloud2_path, cfg):
    """
    Runs the full matching pipeline.
    """
    time0 = time.time()

    cfg['tile_id'] = f'{Path(cloud1_path).stem}_{Path(cloud2_path).stem}'
    create_project_folder(cfg['prj_folder'])

    print(f"Processing  {cfg['tile_id']} ...")
    print('Visualization set to '+str(cfg.get('visualize', False)))

    # === LOAD MODEL ==================================================
    model, device = load_model(cfg)
    time1 = time.time()

    # === PREPROCESSING ===============================================
    tile_a, tile_b = load_tiles(cloud1_path, cloud2_path, cfg)
    preprocess_tiles(tile_a, tile_b, cfg)
    time2 = time.time()

    # === KEYPOINT DETECTION ==========================================
    detect_keypoints(tile_a, tile_b, cfg)
    time3 = time.time()

    # === DESCRIPTION =================================================
    compute_descriptors(tile_a, tile_b, model, device, cfg)
    time4 = time.time()

    # === MATCHING ====================================================
    corres = match_features(tile_a, tile_b, cfg)
    time5 = time.time()

    # === RANSAC ======================================================
    rsc_tile_ids = np.unique(tile_a.rsc_id)
    corres_rsc = filter_matches(corres, rsc_tile_ids, cfg)
    time6 = time.time()

    # === ICP =========================================================
    corres_icp = refine_icp(corres_rsc, tile_a, tile_b, cfg)
    time7 = time.time()

    # === FINAL OUTPUT ===============================================
    build_output(corres, corres_rsc, corres_icp, tile_a, tile_b, cfg)

    stats_dict = {
        "setup": time1 - time0,
        "preprocess": time2 - time1,
        "detect": time3 - time2,
        "describe": time4 - time3,
        "match": time5 - time4,
        "ransac": time6 - time5,
        "icp": time7 - time6,
        "total": time7 - time0,
    }

    return corres_icp, stats_dict

def load_tiles(cloud1_path, cloud2_path, cfg):
    """
    Load Tile objects from file paths (txt or las/laz).
    Returns (tile_a, tile_b)
    """
    print("Loading data...")
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
    print("Preprocessing data...")
    prepare_overlap(tile_a, tile_b, cfg)

    if cfg.get('vox_size', 0) > 0:
        print("Initial voxelization...")
        tile_a.apply_voxelization(cfg)
        tile_b.apply_voxelization(cfg)

    if cfg.get('save_tiles', False):
        print("Saving tile to csv...")
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
    print("Keypoints estimation and tracking...")
    kpts_a = detect_keypoints_iss(tile_a.xyz, cfg)
    kpts_b = detect_keypoints_iss(tile_b.xyz, cfg)
    _, tile_a.kpts_id = tile_a.kdt.query(np.asarray(kpts_a.points), workers=-1)
    _, tile_b.kpts_id = tile_b.kdt.query(np.asarray(kpts_b.points), workers=-1)

    if 'max_kpts' in cfg and cfg['max_kpts'] is not None:
        downsample_keypoints(tile_a, cfg)
        downsample_keypoints(tile_b, cfg)

    delete_useless_keypoints(tile_a, tile_b, cfg)
    delete_useless_keypoints(tile_b, tile_a, cfg)

    if cfg.get('visualize', False):
        vis_kpts(tile_a, tile_b, kpts_a, kpts_b)

def compute_descriptors(tile_a, tile_b, model, device, cfg):
    """
    Compute LCD descriptors for both tiles.
    """
    print("Description...")
    with torch.no_grad():
        tile_a.feat = get_features(tile_a, model, device, cfg)
        tile_b.feat = get_features(tile_b, model, device, cfg)
        assert getattr(tile_a, "kpts_id", None) is None or tile_a.feat.shape[0] == tile_a.kpts_id.shape[0]
        assert getattr(tile_b, "kpts_id", None) is None or tile_b.feat.shape[0] == tile_b.kpts_id.shape[0]
    print(f"\033[FDescription... Done")

def match_features(tile_a, tile_b, cfg):
    """
    Candidate generation and feature-based nearest neighbor search.
    """
    print("Point cloud matching...")
    get_candidates(tile_a, tile_b, cfg)
    get_candidates(tile_b, tile_a, cfg)

    tile_a.cor_id, feat_dist_a, _ = feat_search(tile_a, tile_b)
    tile_b.cor_id, _, _ = feat_search(tile_b, tile_a)
    tile_a.clear_features()
    tile_b.clear_features()

    corres = Corres.from_tiles(tile_a, tile_b, feat_dist_a)

    if cfg.get('reciprocity_test', False):
        print("Reciprocity test...", end=' ')
        mask = reciprocity_test(tile_a, tile_b)
        print(f"{100 * np.sum(mask) / float(tile_a.kpts_id.shape[0]) if tile_a.kpts_id.shape[0] > 0 else 0.0:.2f}% kept")
        corres = corres.apply_mask(mask)

    tile_a.clear_candidates()
    tile_b.clear_candidates()

    return corres

def filter_matches(corres, tile_ids, cfg):
    """
    Apply per-tile RANSAC filtering exactly like the original script.
    Returns concatenated corres_rsc array.
    """
    print("RANSAC filtering...")
    corres_rsc = None
    min_inliers = cfg.get("rsc_min_inliers", 50)

    for tid in tile_ids:
        print(f"\033[FRANSAC filtering, tile {tid}...", end=" ")

        tile_mask = (corres.rsc_id == tid)
        corres_tile = corres.apply_mask(tile_mask)

        idx = ransac_filter(corres_tile, cfg)

        if idx.shape[0] < min_inliers:
            print(f"skipped (<{min_inliers} inliers)")
            continue

        filtered_tile = corres_tile.apply_mask(idx[:, 0])

        corres_rsc = filtered_tile if corres_rsc is None else concatenate_corres(corres_rsc, filtered_tile)
    if corres_rsc is None:
        print("RANSAC resulted in zero matches.")
        return Corres.empty()
    else:
        print("RANSAC Done")
        return corres_rsc

def refine_icp(corres_rsc, tile_a, tile_b, cfg):
    """
    Run per-tile ICP refinement exactly like original corrICP usage.
    Returns corres_icp (concatenated per-tile arrays).
    """
    print("ICP refinement...")
    if corres_rsc is None or corres_rsc.idx_a.size == 0:
        print("No matches to refine with ICP.")
        return Corres.empty()

    corres_icp = None
    tile_ids = np.unique(tile_a.rsc_id)

    for tid in tile_ids:
        print(f"\033[FICP refinement, tile {tid}...")

        tile_mask = (corres_rsc.rsc_id == tid)
        corres_tile = corres_rsc.apply_mask(tile_mask)

        corres_tile = refine_cor_icp(tile_a, tile_b, corres_tile, cfg)

        corres_icp = corres_tile if corres_icp is None else concatenate_corres(corres_icp, corres_tile)

    return corres_icp

def build_output(corres, corres_rsc, corres_icp, tile_a, tile_b, cfg):
    """
    Wrap final output generation and stats building.
    """

    build_corres_file(corres_icp, tile_a, tile_b, cfg)

    if cfg.get('save_stats', False):
        print("Building stats & plots...")
        stats_raw = stats.compute_stats(corres.to_array(), cfg['tile_id'])
        stats_rsc = stats.compute_stats(corres_rsc.to_array(), cfg['tile_id']) 
        stats_icp = stats.compute_stats(corres_icp.to_array(), cfg['tile_id'])

        if stats_raw is not None:
            stats.plot_stats(stats_raw, cfg['prj_folder'] + "plots/", f'raw_{cfg["tile_id"]}')
        if stats_rsc is not None:
            stats.plot_stats(stats_rsc, cfg['prj_folder'] + "plots/", f'rsc_{cfg["tile_id"]}')
        if stats_icp is not None:
            stats.plot_stats(stats_icp, cfg['prj_folder'] + "plots/", f'icp_{cfg["tile_id"]}')

        if stats_raw is not None and stats_rsc is not None and stats_icp is not None:
            stats.plot_final(stats_raw, stats_rsc, stats_icp, cfg)

    return True

