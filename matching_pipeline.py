import torch
import time
import numpy as np
import os
import psutil

from lib import stats, icp
from lib.utils_LCD import getFeatures, load_model
from lib.tools import *
from lib.vis import visKpts, visMatchPts
from lib.filter import ransacFilter, reciprocityTest
from lib.georef import R_enu2ecef

process = psutil.Process(os.getpid())


def get_ram_usage_mb():
    return process.memory_info().rss / 1024**2

def run_pipeline(cloud1_path, cloud2_path, cfg):
    """
    Runs the full matching pipeline.
    """
    time0 = time.time()

    cfg['tile_id'] = f'{Path(cloud1_path).stem}_{Path(cloud2_path).stem}'
    createProjectFolder(cfg['prj_folder'])

    print(f"Processing  {cfg['tile_id']} ...")
    print('Visualization set to '+str(cfg['visualize']))

    # --------------------- LOAD MODEL -----------------------
    model, device = load_model(cfg)
    time1 = time.time()

    # ------------------- PREPROCESSING ----------------------
    tile_a, tile_b = load_tiles(cloud1_path, cloud2_path, cfg)
    preprocess_tiles(tile_a, tile_b, cfg)
    time2 = time.time()

    # ------------------ KEYPOINT DETECTION ------------------
    detect_keypoints(tile_a, tile_b, cfg)
    time3 = time.time()

    # ------------------- DESCRIPTION ------------------------
    compute_descriptors(tile_a, tile_b, model, device, cfg)
    time4 = time.time()

    # ------------------- MATCHING ---------------------------
    corres = match_features(tile_a, tile_b, cfg)
    time5 = time.time()

    # ------------------- RANSAC -----------------------------
    corres_rsc = filter_matches(corres, tile_a, tile_b, cfg)
    time6 = time.time()

    # ------------------- ICP -------------------------------
    corres_icp = refine_icp(corres_rsc, tile_a, tile_b, cfg)
    time7 = time.time()

    # ------------------- FINAL OUTPUT -----------------------
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
    Preprocessing steps: overlap tiling, optional voxel tracing,
    and optionally save tiles to csv. 
    """
    print("Preprocessing data...")
    prepOverlap(tile_a, tile_b, cfg)

    if cfg.get('vox_size', 0) > 0:
        print("Initial voxelization...")
        tile_a.voxTracing(cfg)
        tile_b.voxTracing(cfg)

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
    kpts_a = issKpts(tile_a.xyz, cfg)
    kpts_b = issKpts(tile_b.xyz, cfg)
    _, tile_a.kpts_id = tile_a.kdt.query(np.asarray(kpts_a.points), workers=-1)
    _, tile_b.kpts_id = tile_b.kdt.query(np.asarray(kpts_b.points), workers=-1)

    if 'max_kpts' in cfg and cfg['max_kpts'] is not None:
        downSampleKpts(tile_a, cfg)
        downSampleKpts(tile_b, cfg)

    cleanKpts(tile_a, tile_b, cfg)
    cleanKpts(tile_b, tile_a, cfg)

    if cfg.get('visualize', False):
        visKpts(tile_a, tile_b, kpts_a, kpts_b)

    print(f"RAM usage: {get_ram_usage_mb():.2f} MB")

def compute_descriptors(tile_a, tile_b, model, device, cfg):
    """
    Compute LCD descriptors for both tiles.
    """
    print("Description...")
    with torch.no_grad():
        tile_a.feat = getFeatures(tile_a, model, device, cfg)
        tile_b.feat = getFeatures(tile_b, model, device, cfg)
        print(f"RAM usage: {get_ram_usage_mb():.2f} MB")
    print(f"\033[FDescription... Done")

def match_features(tile_a, tile_b, cfg):
    """
    Candidate generation and feature-based nearest neighbor search.
    """
    print("Point cloud matching...")
    getCandidates(tile_a, tile_b, cfg)
    getCandidates(tile_b, tile_a, cfg)

    tile_a.cor_id, feat_dist_a, _ = featSearch(tile_a, tile_b)
    tile_b.cor_id, feat_dist_b, _ = featSearch(tile_b, tile_a)
    print(f"RAM usage: {get_ram_usage_mb():.2f} MB")

    try:
        del tile_a.feat, tile_b.feat
    except Exception:
        pass

    corres = buildCorres(tile_a, tile_b, feat_dist_a)

    if cfg.get('reciprocity_test', False):
        print("Reciprocity test...", end=' ')
        reciprocal_mask = reciprocityTest(tile_a, tile_b)
        kept_pct = 100 * np.sum(reciprocal_mask) / float(tile_a.kpts_id.shape[0]) if tile_a.kpts_id.shape[0] > 0 else 0.0
        print(f"{kept_pct:.2f}% kept")
        corres = corres[reciprocal_mask]
        print(f"RAM usage: {get_ram_usage_mb():.2f} MB")

    try:
        del tile_a.candidates, tile_b.candidates
    except Exception:
        pass

    return corres

def filter_matches(corres, tile_a, tile_b, cfg):
    """
    Apply per-tile RANSAC filtering exactly like the original script.
    Returns concatenated corres_rsc array.
    """
    print("RANSAC filtering...")
    corres_rsc_list = []

    unique_tiles = np.unique(tile_a.rsc_id)
    for i in unique_tiles:
        print(f"\033[FRansac filtering, tile {int(i)},", end=' ')
        corres_tile = corres[corres[:, 10] == i]
        idx_a_rsc = ransacFilter(corres_tile, cfg)

        if idx_a_rsc.shape[0] < 50:
            print(f"Warning tile {int(i)} has <50 matches -> skipping")
            continue
        corres_rsc_list.append(corres_tile[idx_a_rsc[:, 0], :])

    if len(corres_rsc_list) == 0:
        return np.empty((0, corres.shape[1]))

    corres_rsc = np.concatenate(corres_rsc_list, axis=0)
    print("RANSAC Done")
    return corres_rsc

def refine_icp(corres_rsc, tile_a, tile_b, cfg):
    """
    Run per-tile ICP refinement exactly like original corrICP usage.
    Returns corres_icp (concatenated per-tile arrays).
    """
    print("ICP refinement...")
    corres_icp = None
    unique_tiles = np.unique(tile_a.rsc_id)

    for i in unique_tiles:
        print(f"\033[FICP refinement, tile {int(i)}...")
        corres_tile = corres_rsc[corres_rsc[:, 10] == i]
        icp.corrICP(tile_a, tile_b, corres_tile, cfg)
        if corres_icp is None:
            corres_icp = corres_tile.copy()
        else:
            corres_icp = np.concatenate((corres_icp, corres_tile), axis=0)

    if corres_icp is None:
        return np.empty((0, 14))

    print(f"RAM usage: {get_ram_usage_mb():.2f} MB")
    return corres_icp

def build_output(corres, corres_rsc, corres_icp, tile_a, tile_b, cfg):
    """
    Wrap final output generation and stats building.
    """
    if corres_icp.size == 0:
        icp_vec = np.empty((0, 3))
    else:
        icp_vec = -corres_icp[:, -3:]

    R_enu2ecef_mat = None
    buildCorresFile(corres_rsc, tile_a, tile_b, cfg, icp_vec, R_enu2ecef_mat)

    stats_raw = stats_rsc = stats_icp = None
    if cfg.get('save_stats', False):
        print("Building stats & plots...")
        stats_raw = stats.compute_stats(corres, cfg['tile_id'])
        stats_rsc = stats.compute_stats(corres_rsc, cfg['tile_id']) if corres_rsc.size != 0 else None
        stats_icp = stats.compute_stats(corres_icp, cfg['tile_id']) if corres_icp.size != 0 else None

        if stats_raw is not None:
            stats.plot_stats(stats_raw, cfg['prj_folder'] + "plots/", f'raw_{cfg["tile_id"]}')
        if stats_rsc is not None:
            stats.plot_stats(stats_rsc, cfg['prj_folder'] + "plots/", f'rsc_{cfg["tile_id"]}')
        if stats_icp is not None:
            stats.plot_stats(stats_icp, cfg['prj_folder'] + "plots/", f'icp_{cfg["tile_id"]}')

        if stats_raw is not None and stats_rsc is not None and stats_icp is not None:
            stats.plot_final(stats_raw, stats_rsc, stats_icp, cfg)

    return True

