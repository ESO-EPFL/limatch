import open3d as o3d

def create_sphere(radius, position):
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    sphere.paint_uniform_color([1, 0.2, 0.2])
    sphere.translate(position)  
    return sphere

def visKpts(tile_a, tile_b, kpts_a, kpts_b):
    pcd_a = o3d.geometry.PointCloud()
    pcd_b = o3d.geometry.PointCloud()

    pcd_a.points = o3d.utility.Vector3dVector(tile_a.xyz)
    pcd_b.points = o3d.utility.Vector3dVector(tile_b.xyz)
    pcd_a.paint_uniform_color([0, 116/256, 128/256])
    pcd_b.paint_uniform_color([181/256, 31/256, 31/256])
    kpts_a.paint_uniform_color([0, 0, 1])
    kpts_b.paint_uniform_color([1, 0, 0])

    o3d.visualization.draw([pcd_a, pcd_b, kpts_a, kpts_b])

def visMatchPts(xyz_key, correspondences):
    pcd_a = o3d.geometry.PointCloud()
    pcd_a.points = o3d.utility.Vector3dVector(xyz_key)
    pcd_a.paint_uniform_color([0, 116/256, 128/256])

    cor_xyz = correspondences[:, 4:7]

    spheres = [pcd_a]

    for i in range(cor_xyz.shape[0]):
        spheres.append(create_sphere(0.25, cor_xyz[i])) 


    pcd_a.estimate_normals(
       search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=2., max_nn=60))


    o3d.visualization.draw(spheres)

def vis_candidates(tile_key, tile_target):
    pcd = o3d.geometry.PointCloud()
    pcd_to_descr = o3d.geometry.PointCloud()
    kpts = o3d.geometry.PointCloud()

    pcd.points = o3d.utility.Vector3dVector(tile_target.xyz)
    pcd.paint_uniform_color([0, 116/256, 128/256])

    pcd_to_descr.points = o3d.utility.Vector3dVector(tile_target.xyz[tile_target.candidate_1d])
    pcd_to_descr.paint_uniform_color([181/256, 31/256, 31/256])

    kpts.points = o3d.utility.Vector3dVector(tile_key.xyz[tile_key.kpts_id])
    kpts.paint_uniform_color([0, 1, 0])

    o3d.visualization.draw([pcd, pcd_to_descr, kpts])

    