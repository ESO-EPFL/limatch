from scipy.spatial.transform import Rotation as R
from scipy.interpolate import interp1d
import numpy as np
from pyproj import CRS, Transformer

def georefPts(t_lasvec, lasvec, lever_arm_body, q_sensor2body, trj_t, trj_xyz_map, trj_q_body2map):
    """
    Georeference points from sensor to mapping frame
    """
    trj_xyz = interp1d(trj_t, trj_xyz_map, axis=0, fill_value='extrapolate',kind='cubic')(t_lasvec)
    trj_q = interp1d(trj_t, trj_q_body2map, axis=0, fill_value='extrapolate',kind='cubic')(t_lasvec)

    R_s2b = R.from_quat(q_sensor2body)
    lasvec_body = R_s2b.apply(lasvec) + lever_arm_body

    R_b2m = R.from_quat(trj_q)
    xyz_map = R_b2m.apply(lasvec_body) + trj_xyz

    return xyz_map

def correctLasVecICP(t_lasvec, lasvec, R_sensor2body, trj_t, trj_q_body2map, icp_vec, R_enu2ecef=None):
    """
    Correct lasvec with ICP vector
    """

    trj_q = interp1d(trj_t, trj_q_body2map, axis=0, fill_value='extrapolate',kind='cubic')(t_lasvec)

    if R_enu2ecef is not None:
        R_enu2ecef = R.from_matrix(R_enu2ecef)
        icp_vec = R_enu2ecef.apply(icp_vec)

    R_b2m = R.from_quat(trj_q)
    icp_body = R_b2m.inv().apply(icp_vec)

    R_s2b = R.from_matrix(R_sensor2body)
    icp_sensor = R_s2b.inv().apply(icp_body)

    lasvec_icp = lasvec + icp_sensor

    return lasvec_icp

def R_enu2ecef(xyz, epsg):
    """
    Build ENU -> ECEF rotation matrix from correspondences in local EPSG coordinates.

    """
    xyz_mean = np.mean(xyz, axis=0)
    crs_local = CRS.from_epsg(epsg)
    crs_ecef  = CRS.from_epsg(4978)  # WGS84 ECEF
    crs_geo   = CRS.from_epsg(4979)  # WGS84 lat/lon/height

    transformer_local2ecef = Transformer.from_crs(crs_local, crs_ecef, always_xy=True)
    x_ecef, y_ecef, z_ecef = transformer_local2ecef.transform(*xyz_mean)

    transformer_ecef2geo = Transformer.from_crs(crs_ecef, crs_geo, always_xy=True)
    lon_deg, lat_deg, _ = transformer_ecef2geo.transform(x_ecef, y_ecef, z_ecef)

    print("Reference point (lat, lon):", lat_deg, lon_deg)

    lat = np.deg2rad(lat_deg)
    lon = np.deg2rad(lon_deg)

    e = np.array([-np.sin(lon), np.cos(lon), 0])
    n = np.array([-np.sin(lat)*np.cos(lon), -np.sin(lat)*np.sin(lon), np.cos(lat)])
    u = np.array([np.cos(lat)*np.cos(lon), np.cos(lat)*np.sin(lon), np.sin(lat)])

    R_enu2ecef = np.vstack([e, n, u]).T

    return R_enu2ecef

