import numpy as np
import laspy as lp
from pathlib import Path

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

def load_sbet(file):
    """
    Decodes an APPLANIX SNV/SBET file.

    Parameters:
    - settings: path to SBET

    Returns:
    - 

  Input record: 17xdouble=(136 bytes)
       0  time  			sec_of_week 
       1  latitude   		rad
       2  longitude  		rad
       3  altitude       meters
       4  x_wander_vel   m/s
       5  y_wander_vel   m/s
       6  z_wander_vel  	m/s
       7  roll          	radians
       8  pitch         	radians
       9  wander_heading radians
       10 wander angle   radians
       11 x body accel   m/s^2
       12 y body accel   m/s^2
       13 z body accel   m/s^2
       14 x angular rate rad/s
       15 y angular rate rad/s
	   16 z angular rate rad/s					
 This is what is written in the ouput record:
       0   time            sec_of_week
       1   latitude        rad
       2   longitude       rad
       3   altitude        m
       4   roll            rad
       5   pitch           rad
       6   heading         rad 
    """

    try:
        with open(file, "rb") as f:
            data = np.fromfile(f, dtype=np.float64).reshape(-1,17)
    except Exception as e:
        errmsg = f"Cannot open file! {str(e)}"
        raise ValueError(errmsg)
    
    # True heading = wander_heading - wander_angle, hence data[:, 9]-data[:, 10]
    return  np.column_stack((data[:, 0], data[:, 1:4], data[:, 7:9], data[:, 9]-data[:, 10]))

