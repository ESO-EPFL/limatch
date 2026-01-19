import os
import numpy as np
from pyproj import Transformer

from scipy.spatial.transform import Rotation as R, Slerp


from lib.io import load_sbet
from lib.rotations import *


import logging
logger = logging.getLogger("LiMatch")
from lib.logger import log_sub, log_sub_sub

lla2ecefTransformer = Transformer.from_crs("EPSG:4326", "EPSG:4978")
ecef2llaTransformer = Transformer.from_crs("EPSG:4978", "EPSG:4326")

class Trajectory:
    def __init__(self, time, lla, rpy):

        self.t = time
        self.lla = lla
        self.ecef = np.zeros(lla.shape)
        for i in range(lla.shape[0]):
            self.ecef[i, :] = lla2ecefTransformer.transform(lla[i, 0], lla[i, 1], lla[i, 2], radians=True)
        self.rpy = rpy

        R_ned2b_list = []
        R_ned2e_list = []

        for i in range(len(time)):
            R_ned2b_list.append(R_ned2b(rpy[i,0], rpy[i,1], rpy[i,2]))
            R_ned2e_list.append(R_ned2e(lla[i,0], lla[i,1]))

        self.R_ned2b = R.from_matrix(np.array(R_ned2b_list))
        self.R_ned2e = R.from_matrix(np.array(R_ned2e_list))

        self.slerp_ned2b = Slerp(time, self.R_ned2b)
        self.slerp_ned2e = Slerp(time, self.R_ned2e)

    @classmethod
    def fromSBET(cls, path):
        log_sub(logger, f"Loading trajectory from SBET: {path}...")
        trj = load_sbet(path)
        log_sub_sub(logger, f"Loaded {trj.shape[0]} trajectory points.")
        log_sub_sub(logger, f"Time span: {trj[0,0]} to {trj[-1,0]} s.")
        return cls(
            time= trj[:,0],
            lla=trj[:,1:4],
            rpy=trj[:,4:7],
        )
    
    def get_poses(self, t_query):
        t_query = t_query.reshape(-1)

        ecef_query = np.empty((len(t_query), 3))
        lla_query = np.empty((len(t_query), 3))
        for i in range(3):
            ecef_query[:,i] = np.interp(t_query, self.t, self.ecef[:,i]).reshape(-1)
            lla_query[:,i] = np.interp(t_query, self.t, self.lla[:,i]).reshape(-1)

        R_ned2b_interp = self.slerp_ned2b(t_query).as_matrix()
        R_ned2e_interp = self.slerp_ned2e(t_query).as_matrix()

        R_b2e_query = R_ned2e_interp @ R_ned2b_interp.transpose(0, 2, 1)

        return ecef_query, lla_query, R_b2e_query

class TangentPlane:
    def __init__(self, lat, lon):
        print(f"Setting up tangent plane at lat: {lat}°, lon: {lon}°")
        self.lat = np.radians(lat)
        self.lon = np.radians(lon)

        self.xyz0 = np.array(lla2ecefTransformer.transform(self.lat, self.lon, 0, radians=True)).reshape(1,3)

        self.R_ecef2enu = T_enu_ned() @ R_ned2e(self.lat, self.lon).T

    def enu_to_ecef(self, xyz_enu):
        """
        Convert local ENU coordinates to ECEF.

        Parameters
        ----------
        xyz_enu : (N,3) 
            Coordinates in local tangent plane [E, N, U]

        Returns
        -------
        xyz_ecef : (N,3)
        """

        xyz_ecef = (
            self.xyz0.reshape(1,3)
            + (self.R_enu2ecef @ xyz_enu.T).T
        )

        return xyz_ecef

def simulate_las_vec(traj: Trajectory, time, xyz, R_s2b, a_s, point_epsg, ltp_origin=None):
    """
    Simulate laser vectors from trajectory info (ecef) and points from any epsg

    """
    R_s2b = np.array(R_s2b)

    ecef_traj,_ , R_b2e = traj.get_poses(time)

    lla = np.zeros_like(xyz)
    ecef_pts = np.zeros_like(xyz)

    if point_epsg == 'local':
        ltp = TangentPlane(ltp_origin[0], ltp_origin[1])
        ecef_pts = ltp.enu_to_ecef(xyz)
    elif point_epsg == 2056:
        log_sub_sub(logger, f"Converting points from LV95 to ECEF...")
        log_sub_sub(logger, f"WARNING: using approximate transform with meter level accuracy ...")
        log_sub_sub(logger, f"Constant bias in laser vector expected, see Swisstop documentation for details.")
        for i in range(xyz.shape[0]):
            lla[i, :] = lv95_to_wgs84_geodetic(xyz[i,0], xyz[i,1], xyz[i,2])
            ecef_pts[i, :] = np.vstack(lla2ecefTransformer.transform(lla[i, 0], lla[i, 1], lla[i, 2], radians=True)).T
    else:
        log_sub_sub(logger, f"Converting points from EPSG:{point_epsg} to ECEF...")
        transformer = Transformer.from_crs(f"EPSG:{point_epsg}", "EPSG:4326", always_xy=True)
        for i in range(xyz.shape[0]):
            lon, lat, h = transformer.transform(xyz[i,0], xyz[i,1], xyz[i,2], radians=True)
            lla[i, :] = np.array([lat, lon, h])
            ecef_pts[i, :] = np.vstack(lla2ecefTransformer.transform(lla[i, 0], lla[i, 1], lla[i, 2], radians=True)).T
        
    las_vec = np.zeros_like(ecef_pts)

    for i, pa in enumerate(ecef_pts):

        vec_body_a = R_b2e[i].T @ (pa - ecef_traj[i]) - a_s

        las_vec[i, :] = R_s2b.T @ vec_body_a.T

    return las_vec

def correct_laser_vector(traj: Trajectory, lasvec, R_sensor2body, time, icp_vec):
    """
    Correct laser vector with ICP vector, mapping frame is approximated as a local ENU frame for simplicity, assumption acceptable for short vectors (submetric)
    """
    R_s2b = np.array(R_sensor2body)

    _, lla_traj, R_b2e = traj.get_poses(time)

    lasvec_icp = np.zeros_like(lasvec)

    for i in range(len(time)):
        # ENU -> ECEF
        R_enu2ecef_i = R_enu2e(lla_traj[i,0], lla_traj[i,1])

        # ICP: ENU -> ECEF -> body -> sensor
        icp_sensor =  R_s2b.T @ R_b2e[i].T @ (R_enu2ecef_i @ icp_vec[i])

        lasvec_icp[i] = lasvec[i] + icp_sensor

    return lasvec_icp

def lv95_to_wgs84_geodetic(E, N, H):
    E_ = (E - 2600000.0) / 1e6
    N_ = (N - 1200000.0) / 1e6

    lat = (
        16.9023892
        + 3.238272 * N_
        - 0.270978 * E_**2
        - 0.002528 * N_**2
        - 0.0447   * E_**2 * N_
        - 0.0140   * N_**3
    )

    lon = (
        2.6779094
        + 4.728982 * E_
        + 0.791484 * E_ * N_
        + 0.1306   * E_ * N_**2
        - 0.0436   * E_**3
    )

    h = H + 49.55 - 12.60 * E_ - 22.64 * N_

    lat = lat * 100 / 36
    lon = lon * 100 / 36

    return np.array([np.deg2rad(lat), np.deg2rad(lon), h])