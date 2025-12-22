import numpy as np
import matplotlib.pyplot as plt

from lib.map import Trajectory
from lib.data_handling import Corres, Tile
from lib.map import simulate_las_vec, correct_laser_vector

import logging
logger = logging.getLogger("LiMatch")
from lib.logger import log_sub, log_sub_sub

def test_corres_apply_mask_keeps_alignment():
    log_sub_sub(logger, "Testing Corres apply_mask method for alignment...")
    c = Corres(
        idx_a=np.array([0,1,2]),
        idx_b=np.array([3,4,5]),
        time_a=np.array([10., 20., 30.]),
        time_b=np.array([11., 21., 31.]),
        d_xyz=np.array([1.,2.,3.]),
        d_feat=np.array([0.1,0.2,0.3]),
        xyz_a=np.random.rand(3,3),
        xyz_b=np.random.rand(3,3),
        rsc_id=np.array([0,0,1]),
        icp_vec=np.random.rand(3,3),
    )

    c2 = c.apply_mask(np.array([True, False, True]))

    assert len(c2.time_a) == 2
    assert c2.time_a[1] == 30.
    assert np.allclose(c2.icp_vec[0], c.icp_vec[0])


def test_correct_laser_vector_zero_icp(traj: Trajectory):
    log_sub_sub(logger, "Testing correct_laser_vector with zero ICP correction...")
    N = 5
    lasvec = np.random.randn(N, 3)
    icp_vec = np.zeros((N, 3))
    time = np.linspace(traj.t[0], traj.t[-1], N).reshape(-1,1)

    lasvec_corr = correct_laser_vector(
        traj,
        lasvec,
        R_sensor2body=np.eye(3),
        time=time,
        icp_vec=icp_vec,
    )

    assert np.allclose(lasvec, lasvec_corr)
    assert np.max(np.linalg.norm(lasvec - lasvec_corr, axis=1)) < 1e-9
    mean_diff = np.mean(np.linalg.norm(lasvec - lasvec_corr, axis=1))
    log_sub_sub(logger, f"Mean difference with zero ICP correction: {mean_diff:.6f} m")
    

def test_las_vec_sim(corr: Corres, tile_a: Tile, tile_b: Tile, traj: Trajectory, cfg):
    time_a = corr.time_a.reshape(-1,1)
    time_b = corr.time_b.reshape(-1,1)

    xyz_a = corr.xyz_a + tile_a.shift
    xyz_b = corr.xyz_b + tile_b.shift

    las_vec_a = simulate_las_vec(
        traj,
        time=time_a,
        xyz=xyz_a,
        R_s2b=cfg['R_sensor2body'],
        a_s=cfg['lever_arm'],
        point_epsg=cfg['point_epsg'],
    )

    las_vec_b = simulate_las_vec(
        traj,
        time=time_b,
        xyz=xyz_b,
        R_s2b=cfg['R_sensor2body'],
        a_s=cfg['lever_arm'],
        point_epsg=cfg['point_epsg'],
    )

    true_las_vec_a = tile_a.las_vec[corr.idx_a]
    true_las_vec_b = tile_b.las_vec[corr.idx_b]

    err_a = las_vec_a - true_las_vec_a
    err_b = las_vec_b - true_las_vec_b
    diff_a = np.linalg.norm(err_a, axis=1)
    diff_b = np.linalg.norm(err_b, axis=1)

    mean_diff_a = np.mean(diff_a)
    mean_diff_b = np.mean(diff_b)
    med_diff_a = np.median(diff_a)
    med_diff_b = np.median(diff_b)

    # assert mean_diff_a < 1e-2
    # assert mean_diff_b < 1e-2

    q_05_a = np.quantile(diff_a, 0.05)
    q_05_b = np.quantile(diff_b, 0.05)
    q_95_a = np.quantile(diff_a, 0.95)
    q_95_b = np.quantile(diff_b, 0.95)
    
    log_sub_sub(logger, f"Laser Vector A - Mean diff: {mean_diff_a:.6f} m, Median diff: {med_diff_a:.6f} m, 5th perc.: {q_05_a:.6f} m, 95th perc.: {q_95_a:.6f} m")
    log_sub_sub(logger, f"Laser Vector B - Mean diff: {mean_diff_b:.6f} m, Median diff: {med_diff_b:.6f} m, 5th perc.: {q_05_b:.6f} m, 95th perc.: {q_95_b:.6f} m")

    log_sub_sub(logger, "Plotting histogram of differences...")
    plt.figure()
    plt.hist([diff_a, diff_b], bins=50, alpha=0.7, color=['blue', 'orange'], edgecolor='black')

    plt.title('Histogram of Differences between Simulated and True Laser Vectors A')
    plt.xlabel('Difference (m)')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

    log_sub_sub(logger, "Plotting 3D error for Laser Vector A, B...")
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(err_a[:,0], err_a[:,1], err_a[:,2], c='blue', label='Laser Vector A', alpha=0.5)
    ax.scatter(err_b[:,0], err_b[:,1], err_b[:,2], c='orange', label='Laser Vector B', alpha=0.5)
    ax.legend()
    ax.set_title('Direction of Error between Simulated and True Laser Vectors A')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

def run_tests(corres: Corres, tile_a: Tile, tile_b: Tile, traj: Trajectory, cfg):
    log_sub(logger, "Running unit tests...")
    log_sub(logger, "Testing Corres apply_mask method...")
    test_corres_apply_mask_keeps_alignment()
    log_sub(logger, "Testing correct_laser_vector with zero ICP correction...")
    test_correct_laser_vector_zero_icp(traj)

    log_sub(logger, "Testing laser vector simulation...")
    test_las_vec_sim(corres, tile_a, tile_b, traj, cfg)