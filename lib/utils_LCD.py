import math
import torch
import numpy as np
from pathlib import Path

from submodules.lcd.lcd import models

from lib.data_handling import Tile

import logging
logger = logging.getLogger("LiMatch")
from lib.logger import log_progress, log_sub, log_sub_sub

def load_model(cfg):
    """
    Load the descriptor model safely and robustly.
    Handles both:
      - pure state_dict
      - {"model": state_dict}
    Ensures model.eval() and torch.no_grad() usage.
    """
    log_progress(logger, 0, "Model setup")
    model_path = Path(cfg["nn_path"])
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    if torch.cuda.is_available():
        device = torch.device("cuda")
        log_sub(logger, "Using CUDA for descriptor inference.")
    else:
        device = torch.device("cpu")
        log_sub(logger, "CUDA not available. Running on CPU. Expect slower descriptor inference.")

    model = models.PointNetAutoencoder(256, 6, 6, True)

    log_sub(logger, f"Loading from: {model_path}")
    try:
        checkpoint = torch.load(
            model_path,
            map_location=device,
            weights_only=False 
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint {model_path}: {e}")

    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state = checkpoint["model"]
        log_sub(logger, "Loaded nested checkpoint with key 'model'.")
    elif isinstance(checkpoint, dict) and all(
        isinstance(v, torch.Tensor) for v in checkpoint.values()
    ):
        state = checkpoint
        log_sub(logger, "Loaded direct state_dict checkpoint.")
    else:
        raise ValueError(
            f"Unrecognized checkpoint structure. Expected state_dict or dict with 'model'. "
            f"Got keys: {list(checkpoint.keys())}"
        )

    try:
        model.load_state_dict(state)
    except Exception as e:
        raise RuntimeError(
            f"Checkpoint structure does not match model architecture: {e}"
        )
    model.to(device)
    model.eval()

    log_sub(logger, "Descriptor model successfully loaded and set to eval() mode.")
    return model, device

def extract_patches(tile, cfg, idx):
    '''
    Extracts a patch around each query points from point cloud 
    If dual_tile is provided, the function will extract patches from both tiles and merge them to generate a fused patch
    Parameters
    ----------
    tile : tools.tile
        Tile object containing the point cloud and the kdtree.
    cfg : dict
        configuration dictionary.
    idx : int
        index of the main batch loop.
    
    Returns
    -------
    patches : np.array(Nx1024x6)
        pointwise neighborhood.
    '''
    num_pts = cfg['lcd_patch_n']
    query_pts = tile.xyz[tile.kpts_id[idx:idx+cfg['main_batch']]]
    indexes = tile.kdt.query_ball_point(query_pts, cfg['lcd_patch_r'], workers=-1)
    
    patches = np.zeros((len(query_pts), num_pts,6))

    for i,index in enumerate(indexes):
        if(len(index) < num_pts):
            rep = math.floor(num_pts/len(index))
            remainder = num_pts % len(index)

            index_full = np.tile(index, rep)
            index_full = np.concatenate(
                [index_full, np.random.choice(index_full, remainder)], axis=0)
        else:
            index_full = np.random.choice(index, num_pts, replace=False)

        patches[i,:,:3] = ((tile.xyz[index_full]-query_pts[i])/cfg['lcd_patch_r']).astype('float16')       
      
    return patches

def compute_lcd(patches, model, batch_size, device):
    """
    Compute LCD descriptor given an input for the function extract_uniform_patches

    Parameters
    ----------
    patches : np.array(Nx1024x6)
        output of extract_uniform_patches, pointwise neighborhood.
    model : lcd.models.pointnet.PointNetAutoencoder
        retrained LCD model (3D part only).
    batch_size : int
        number of point to process at once, dependent on GPU memory available
        and size of input.
    device : torch.device
        GPU or CPU to perform inference.

    Returns
    -------
    descriptor matrix  : np.array (Nx256)
        output of the network, pointwise descriptor of 256 dim.

    """
    batches = torch.tensor(patches, dtype=torch.float32)
    batches = torch.split(batches, batch_size)
    descriptors = []
    with torch.no_grad():
        for x in batches:
            x = x.to(device)
            z = model.encode(x)
            z = z.cpu().numpy()
            descriptors.append(z)
    return np.concatenate(descriptors, axis=0).astype('float32')

def get_features(tile: Tile, model, device, cfg):
    feat = np.zeros((tile.kpts_id.shape[0], 256),dtype='float32')
    for i in range(0, tile.kpts_id.shape[0], cfg['main_batch']):
        batch_id = int(np.ceil(i / cfg['main_batch']) + 1)
        log_sub_sub(logger, f"batch {batch_id}/{int(np.ceil(tile.kpts_id.shape[0]/cfg['main_batch']))}...")
        patches = extract_patches(tile, cfg, i)
        feat[i:i+cfg['main_batch']] = compute_lcd(patches, model, cfg['lcd_batch'], device)
    del patches

    return feat

def feat_search(tile_key: Tile, tile_target: Tile):


    candidate = tile_key.candidates
    feats_k = tile_key.feat
    feats_t = tile_target.feat

    n_k = feats_k.shape[0]

    f_dist = np.empty(n_k)
    idx_t = np.empty(n_k, dtype=np.uint32)

    for i in range(n_k):
        cand_idx = candidate[i]                  # indices in target cloud
        cand_feats = feats_t[cand_idx]           # (Nc, D)

        # Compute L2 distances (vectorized)
        diff = cand_feats - feats_k[i]           # (Nc, D)
        dists = np.sum(diff * diff, axis=1)      # squared L2

        local_min = np.argmin(dists)

        f_dist[i] = dists[local_min]
        idx_t[i] = tile_target.kpts_id[cand_idx[local_min]]

    return idx_t, f_dist, tile_target.xyz[idx_t]
