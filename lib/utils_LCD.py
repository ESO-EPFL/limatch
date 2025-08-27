import math
import torch
import numpy as np

def extractPatches(tile, cfg, idx):
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
    dual_tile : tools.tile, optional if dual lidar payload is used
    
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


def computeLCD(patches, model, batch_size, device):
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

def getFeatures(tile, model, device, cfg):
    feat = np.zeros((tile.kpts_id.shape[0], 256),dtype='float32')
    for i in range(0, tile.kpts_id.shape[0], cfg['main_batch']):
        print(f"\033[FDescription {int(100*i/tile.kpts_id.shape[0])}%...")
        patches = extractPatches(tile, cfg, i)
        feat[i:i+cfg['main_batch']] = computeLCD(patches, model, cfg['lcd_batch'], device)
    del patches

    return feat