import numpy as np

c = np.cos
s = np.sin

def R1(r):
    """
    Rotation matrix around the x-axis, r in radians
    """
    return np.array([[1,    0,    0],
                     [0, c(r), s(r)],
                     [0,-s(r), c(r)]])

def R2(p):
    """
    Rotation matrix around the y-axis, p in radians
    """
    return np.array([[c(p), 0,-s(p)],
                     [   0, 1,    0],
                     [s(p), 0, c(p)]])

def R3(y):
    """
    Rotation matrix around the z-axis, y in radians
    """
    return np.array([[ c(y), s(y), 0],
                     [-s(y), c(y), 0],
                     [    0,    0, 1]])

def R_b2ned(r, p, y):
    """
    Rotation matrix from body to local NED frame given roll (r), pitch (p), yaw (y) in radians
    """
    return (R1(r) @ R2(p) @ R3(y)).T

def R_ned2b(r, p, y):
    """
    Rotation matrix from ned to body frame given roll (r), pitch (p), yaw (y) in radians
    """
    return R1(r) @ R2(p) @ R3(y)

def R_ned2e(lat,lon):
    """
    Rotation matrix from local level NED to ECEF frame.
    :param lat, lon: latitude and longitude in radians
    :return: rotation matrix
    """
    return np.array([[ -s(lat)*c(lon),-s(lon), -c(lat)*c(lon)],
                     [ -s(lat)*s(lon), c(lon), -c(lat)*s(lon)],
                     [         c(lat),      0,        -s(lat)]])

def T_enu_ned():
    """
    Rotation matrix from local level ENU to NED frame and vice versa
    """
    return np.array([[0, 1, 0],
                     [1, 0, 0],
                     [0, 0,-1]])

#mapping frame m refers to local enu tangent plane with specified, fixed, origin.
def R_b2m(lat, lon, r, p, y, R_e2m):
    """
    Rotation matrix from body to mapping enu frame 
    """
    R_b2ned = R_b2ned(r, p, y)

    R_ned2e = R_ned2e(lat, lon)

    return R_e2m @ R_ned2e @ R_b2ned

def R_enu2e(lat, lon):
    """
    Build ENU -> ECEF rotation matrix from correspondences in local EPSG coordinates.

    """

    return np.array([
        [ -s(lon), -s(lat)*c(lon),  c(lat)*c(lon)],
        [  c(lon), -s(lat)*s(lon),  c(lat)*s(lon)],
        [       0,          c(lat),         s(lat)]
    ])
