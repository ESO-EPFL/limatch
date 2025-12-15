
import numpy as np
from pathlib import Path
import laspy as lp

import logging
logger = logging.getLogger("LiMatch")
from lib.logger import log_sub

def create_project_folder(path):
    '''
    Create the folder structure for the project
    '''
    log_sub(logger, f"Will save data at: {path}")
    Path(path).mkdir(parents=True, exist_ok=True)
    Path(path+"tiles").mkdir(parents=True, exist_ok=True) 
    Path(path+"plots").mkdir(parents=True, exist_ok=True) 
    Path(path+"cor_outputs").mkdir(parents=True, exist_ok=True) 

def load_ascii_cloud(file, cfg):
    """
    Load a point cloud from an ASCII file
    """

    if 'delimiter' in cfg and cfg['delimiter'] is not None:
        raw = np.loadtxt(file, delimiter=cfg['delimiter'], skiprows=cfg['header'])
    else:
        raw = np.loadtxt(file, skiprows=cfg['header'])
    
    xyz =  raw[:, cfg['xyz_col']]

    if 't_col' in cfg and cfg['t_col'] is not None:
        time = raw[:, cfg['t_col']] 

    else:
        time = np.zeros((xyz.shape[0], 1))

    if 'lasvec_col' in cfg and cfg['lasvec_col'] is not None:
        las_vec = raw[:, cfg['lasvec_col']]

    else:
        las_vec = np.zeros((xyz.shape[0], 3))

    return xyz, time, las_vec

def load_las_cloud(file):
    """
    Load a point cloud from a LAS file
    """
    with lp.open(file) as fh:
        las = fh.read()

        has_gps_time = "gps_time" in [d.name for d in las.point_format.dimensions]

        if has_gps_time:
            time = las.gps_time
        else:
            time = np.zeros((las.xyz.shape[0], 1))

        las_vec = np.zeros((las.xyz.shape[0], 3))

    return las.xyz, time, las_vec

