import numpy as np
import open3d as o3d

from lib.data_handling import Corres, Tile

def run_icp(target, ref, cfg): 
    '''
    Run ICP on two point clouds and return the translation vector that align target to ref (target + icp_vec ≃ ref)
    '''
    icp = o3d.pipelines.registration.registration_icp(target, ref, cfg['icp_thresh'],
                                                      estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                                                      criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=cfg['icp_max_n']
                                                                                                                ,relative_fitness=cfg['icp_conv']
                                                                                                                ,relative_rmse=cfg['icp_conv']))
        
    return icp.transformation[:3,-1]

def refine_cor_icp(tile_k: Tile, tile_t: Tile, corr: Corres, cfg):
    '''
    Run ICP on each pair of corresponding points and update the correspondence matrix
    Transormation aligns target patch to key patch
    '''
    corr_icp = corr.deep_copy()
    for i in range(corr.xyz_a.shape[0]):
        xyz_k = corr_icp.xyz_a[i,:]
        xyz_t = corr_icp.xyz_b[i,:]

        patch_k = tile_k.xyz[tile_k.kdt.query_ball_point(xyz_k, cfg['icp_patch_r'])] - xyz_k 
        patch_t = tile_t.xyz[tile_t.kdt.query_ball_point(xyz_t, cfg['icp_patch_r'])] - xyz_t

        pcd_k = o3d.geometry.PointCloud()
        pcd_t = o3d.geometry.PointCloud()

        pcd_k.points = o3d.utility.Vector3dVector(patch_k)
        pcd_t.points = o3d.utility.Vector3dVector(patch_t)

        if cfg['icp_vox_s'] > 0:
            pcd_k = pcd_k.voxel_down_sample(cfg['icp_vox_s'])
            pcd_t = pcd_t.voxel_down_sample(cfg['icp_vox_s'])

        icp_vec = run_icp(pcd_k, pcd_t, cfg)
        xyz_t = xyz_t + icp_vec

        corr_icp.icp_vec[i] = icp_vec
        corr_icp.d_xyz[i] = np.linalg.norm(xyz_k - xyz_t)

    return corr_icp
        
