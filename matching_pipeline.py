import torch
import time
import numpy as np
import os
import psutil
from pathlib import Path

from lib.io import create_project_folder
from lib.utils_LCD import load_model
from lib.keypoints import *
from lib.coarse import run_coarse_bootstrap

process = psutil.Process(os.getpid())

import logging
logger = logging.getLogger("LiMatch")
from lib.logger import get_logger

from lib.pipeline_blocks import (
    load_tiles, preprocess_tiles, detect_keypoints, compute_descriptors,
    match_features, filter_matches, refine_icp, out_and_plot
)

def run_pipeline(cloud1_path, cloud2_path, cfg):
    """
    Runs the full matching pipeline.
    """
    time0 = time.time()
    stats = {}
    stats["time"] = {}
    stats["metrics"] = {}
    
    logger = get_logger(cfg)
    logger.info("Starting LiMatch pipeline...")

    cfg['tile_id'] = f'{Path(cloud1_path).stem}_{Path(cloud2_path).stem}'
    create_project_folder(cfg['prj_folder'])

    logger.info(f"Visualization set to {cfg.get('visualize', False)}")

    # === LOAD MODEL ==================================================
    model, device = load_model(cfg)
    time_model = time.time()
    stats['time']['Model setup'] = time_model - time0


    # === PREPROCESSING ===============================================
    tile_a, tile_b = load_tiles(cloud1_path, cloud2_path, cfg)
    preprocess_tiles(tile_a, tile_b, cfg)

    time_prep = time.time()
    stats['time']['Preprocessing'] = time_prep - time_model
    stats["metrics"]["Points A"] = tile_a.xyz.shape[0]
    stats["metrics"]["Points B"] = tile_b.xyz.shape[0]
    stats["metrics"]["Tiles num."] = int(np.max(tile_a.rsc_id)) if hasattr(tile_a, "rsc_id") else 1

    # ==== COARSE BOOTSTRAP (OPTIONAL) ==========================================
    if cfg.get("coarse", {}).get("enabled", False):
        R,t = run_coarse_bootstrap(tile_a, tile_b, model, device, cfg)
        if R is not None and t is not None:
            tile_a.apply_transform(R, t)
        

    # === KEYPOINT DETECTION ==========================================
    detect_keypoints(tile_a, tile_b, cfg)
    time_detect = time.time()
    stats['time']['Detection'] = time_detect - time_prep
    stats["metrics"]["Keypoints A"] = tile_a.kpts_id.shape[0]
    stats["metrics"]["Keypoints B"] = tile_b.kpts_id.shape[0]

    # === DESCRIPTION =================================================
    compute_descriptors(tile_a, tile_b, model, device, cfg)
    time_describe = time.time()
    stats['time']['Description'] = time_describe - time_detect
    
    # === MATCHING ====================================================
    corres = match_features(tile_a, tile_b, cfg)
    time_match = time.time()
    stats['time']['Matching'] = time_match - time_describe
    stats["metrics"]["Raw matches"] = corres.idx_a.shape[0]

    # === RANSAC ======================================================
    rsc_tile_ids = np.unique(tile_a.rsc_id)
    corres_rsc = filter_matches(corres, rsc_tile_ids, cfg)
    time_ransac = time.time()
    stats['time']['RANSAC'] = time_ransac - time_match
    stats["metrics"]["RANSAC matches"] = (
        corres_rsc.idx_a.shape[0] if corres_rsc is not None else 0
    )

    # === ICP =========================================================
    corres_icp = refine_icp(corres_rsc, tile_a, tile_b, cfg)
    time_icp = time.time()
    stats['time']['ICP'] = time_icp - time_ransac

    # === FINAL OUTPUT ===============================================
    out_and_plot(corres, corres_rsc, corres_icp, tile_a, tile_b, cfg)
    time_end = time.time()
    stats['time']['Total'] = time_end - time0

    logger.info("[Pipeline summary]")
    for k in ["Model setup", "Preprocessing", "Detection", "Description", "Matching", "RANSAC", "ICP", "Total"]:
        logger.info(f"  {k:>15}: {stats['time'][k]:.2f}s")

    logger.info("[Metrics]")
    for k, v in stats["metrics"].items():
        logger.info(f"  {k:>15}: {v}")
    logger.info(f"Peak RAM: {process.memory_info().rss / (1024 ** 3):.2f} GB")
    logger.info("LiMatch pipeline completed.")
    return corres_icp, stats
    
if __name__ == "__main__":
    print("This is a module file. Please run the pipeline through:")
    print("1. from separate script. Example:")
    print("from matching_pipeline import run_pipeline")
    print("corres_icp, stats = run_pipeline('path/to/cloud1.txt', 'path/to/cloud2.txt', cfg)")
    print("2. from command line with main.py. Example")
    print("python main.py --c1 path/to/cloud1.txt --c2 path/to/cloud2.txt --y path/to/config.yaml")
    pass