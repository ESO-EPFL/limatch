import numpy as np
import open3d as o3d
import multiprocessing as mp

def ransacFilter(cor, cfg):
    """
    RANSAC filter based on correspondences
    """
    n_corr = cor.shape[0]
    kpts_xyz = cor[:, 4:7]
    target_xyz = cor[:, 7:10]
    thresh = cfg['rsc_thr']
    
    correspondences_index = np.ones(
        (kpts_xyz.shape[0], 2))*np.arange(kpts_xyz.shape[0]).reshape(-1, 1)
    
    cor = o3d.utility.Vector2iVector(correspondences_index)

    kpts_pcd = o3d.geometry.PointCloud()
    target_pcd = o3d.geometry.PointCloud()

    kpts_pcd.points = o3d.utility.Vector3dVector(kpts_xyz)
    target_pcd.points = o3d.utility.Vector3dVector(target_xyz)
    results = o3d.pipelines.registration.registration_ransac_based_on_correspondence(
        source=kpts_pcd,
        target=target_pcd,
        corres=cor,
        max_correspondence_distance=thresh,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=6,
        checkers=[o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(cfg['rsc_lenCheck']),
                  o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(2*cfg['uncertainty_r'])],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(cfg['rsc_maxN'], cfg['rsc_p']))
    
    filt_cor = np.asarray(results.correspondence_set)
    permutations = np.where(filt_cor[:,0]==filt_cor[:,1])
    filt_cor = filt_cor[permutations]
    if n_corr == 0:
        print("                           !Warning at least 1 ransac filter failed!")
    else: 
        print(f"inliers: {int(100*filt_cor.shape[0]/n_corr)}% ")
    return filt_cor
