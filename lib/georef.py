from scipy.spatial.transform import Rotation as R
from scipy.interpolate import interp1d

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

def correctLasVecICP(t_lasvec, lasvec, R_sensor2body, trj_t, trj_q_body2map, icp_vec):
    """
    Correct lasvec with ICP vector
    """

    trj_q = interp1d(trj_t, trj_q_body2map, axis=0, fill_value='extrapolate',kind='cubic')(t_lasvec)

    R_b2m = R.from_quat(trj_q)
    icp_body = R_b2m.inv().apply(icp_vec)

    R_s2b = R.from_matrix(R_sensor2body)
    icp_sensor = R_s2b.inv().apply(icp_body)

    lasvec_icp = lasvec + icp_sensor

    return lasvec_icp
