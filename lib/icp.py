# %%
import numpy as np
import open3d as o3d

def runICP(target, ref, cfg): 
    '''
    Run ICP on two point clouds and return the transformation matrix aligning target to ref
    '''
    icp = o3d.pipelines.registration.registration_icp(target, ref, cfg['icp_thresh'],
                                                      estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                                                      criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=cfg['icp_max_n']
                                                                                                                ,relative_fitness=cfg['icp_conv']
                                                                                                                ,relative_rmse=cfg['icp_conv']))
        
    return icp

def corrICP(tile_k, tile_t, corr, cfg):
    '''
    Run ICP on each pair of corresponding points and update the correspondence matrix
    Transormation aligns target patch to key patch
    '''
    for i in range(corr.shape[0]):
        xyz_k = tile_k.xyz[corr[i,0].astype(int)]
        xyz_t = tile_t.xyz[corr[i,1].astype(int)]

        patch_k = tile_k.xyz[tile_k.kdt.query_ball_point(xyz_k, cfg['icp_patch_r'])] - xyz_k 
        patch_t = tile_t.xyz[tile_t.kdt.query_ball_point(xyz_t, cfg['icp_patch_r'])] - xyz_t

        pcd_k = o3d.geometry.PointCloud()
        pcd_t = o3d.geometry.PointCloud()

        pcd_k.points = o3d.utility.Vector3dVector(patch_k)
        pcd_t.points = o3d.utility.Vector3dVector(patch_t)

        if cfg['icp_vox_s'] > 0:
            pcd_k = pcd_k.voxel_down_sample(cfg['icp_vox_s'])
            pcd_t = pcd_t.voxel_down_sample(cfg['icp_vox_s'])

        icp = runICP(pcd_k, pcd_t, cfg)

        xyz_t = xyz_t + icp.transformation[:3,-1]

        corr[i,7:10] = xyz_t 
        corr[i,11:] = icp.transformation[:3,-1]
        corr[i,2] = np.linalg.norm(xyz_k - xyz_t)


        
