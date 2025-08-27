import numpy as np
import open3d as o3d
from scipy.spatial import KDTree
import faiss
from pathlib import Path
import laspy as lp
from lib.georef import correctLasVecICP

def createProjectFolder(path):
    '''
    Create the folder structure for the project
    '''
    Path(path).mkdir(parents=True, exist_ok=True)
    Path(path+"tiles").mkdir(parents=True, exist_ok=True) 
    Path(path+"plots").mkdir(parents=True, exist_ok=True) 
    Path(path+"cor_outputs").mkdir(parents=True, exist_ok=True) 

def loadAsciiCloud(file, cfg):
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

def loadLasCloud(file, cfg):
    """
    Load a point cloud from a LAS file
    """
    has_gps_time = "gps_time" in las.point_format.dimensions

    if 'extraDim' in cfg and cfg['extraDim'] is not None:
        has_extraDim = True

    with lp.open(file) as fh:
        las = fh.read()
        if has_gps_time:
            time = las.gps_time
        else:
            time = np.zeros((las.xyz.shape[0], 1))

        if has_extraDim:
            extra = np.array([las[d] for d in cfg['extraDim']]).T
        else:
            extra = np.zeros((las.xyz.shape[0], 0))

        las_vec = np.zeros((las.xyz.shape[0], 3))

    return las.xyz, time, las_vec, extra

def prepOverlap(tile_a, tile_b, cfg):
    '''
    Prepare the data for tiling by filtering out non-overlapping sections and assigning a tile id to each point
    '''
    if cfg['tile']:
        print(f"Tiling with size {cfg['step_x']}x{cfg['step_y']}...")
        xyz_a = tile_a.xyz
        xyz_b = tile_b.xyz

        xmin = max(np.min(xyz_a[:, 0]), np.min(xyz_b[:, 0]))
        ymin = max(np.min(xyz_a[:, 1]), np.min(xyz_b[:, 1]))

        tile_id_a = np.concatenate((np.floor((xyz_a[:, 0]-xmin)/cfg['step_x']).reshape(-1, 1),
                                    np.floor((xyz_a[:, 1]-ymin)/cfg['step_y']).reshape(-1, 1)),
                                    axis=1)
        tile_id_b = np.concatenate((np.floor((xyz_b[:, 0]-xmin)/cfg['step_x']).reshape(-1, 1),
                                    np.floor((xyz_b[:, 1]-ymin)/cfg['step_y']).reshape(-1, 1)),
                                    axis=1)
        kept_id = 0
        kept_id_a = np.zeros((xyz_a.shape[0])).astype(np.uint8)
        kept_id_b = np.zeros((xyz_b.shape[0])).astype(np.uint8)

        for i in range(int(np.max(tile_id_a[:, 0])+1)):
            for j in range(int(np.max(tile_id_a[:, 1])+1)):
                mask_a = np.all([tile_id_a[:, 0] == i, tile_id_a[:, 1] == j], axis=0)
                mask_b = np.all([tile_id_b[:, 0] == i, tile_id_b[:, 1] == j], axis=0)

                den_a = np.sum(mask_a)/cfg['step_x']/cfg['step_y']
                den_b = np.sum(mask_b)/cfg['step_x']/cfg['step_y']

                if den_a > cfg['min_den'] and den_b > cfg['min_den']:
                    kept_id += 1
                    kept_id_a[mask_a] = kept_id
                    kept_id_b[mask_b] = kept_id
        tile_a.rsc_id = kept_id_a.astype(np.uint16)
        tile_b.rsc_id = kept_id_b.astype(np.uint16)

        tile_a.filterByMask(kept_id_a > 0)
        tile_b.filterByMask(kept_id_b > 0)

    else:
        print(f"No tiling...")

    shift = tile_a.xyz.mean(axis=0)

    tile_a.shift = shift
    tile_b.shift = shift

    print(f"Shifting point clouds toward origin, {shift} m...")
    tile_a.xyz = (tile_a.xyz - shift).astype(np.float32)
    tile_b.xyz = (tile_b.xyz - shift).astype(np.float32)

    tile_a.kdt = KDTree(tile_a.xyz)
    tile_b.kdt = KDTree(tile_b.xyz)
    print(f"{np.max(kept_id_a)} valid tiles generated...")

class tile:
    def __init__(self, time, xyz, lasvec, extraData = None):
        """
        Class to store the tiled data and keypoints

        """
        self.time = time
        self.xyz = xyz 
        self.las_vec = lasvec.astype(np.float32)
        self.rsc_id = np.ones((xyz.shape[0],), dtype=np.uint16)

        self.kdt = KDTree(self.xyz)

        if extraData is not None:
            self.extraData = extraData
        else:
            self.extraData = np.empty((xyz.shape[0], 0))

    @classmethod
    def fromASCII(cls, file, cfg):
        """
        Load point cloud data from an ASCII file
        """
        xyz, time, las_vec = loadAsciiCloud(file, cfg)
        return cls(time, xyz, las_vec)
    
    @classmethod
    def fromLAS(cls, file, cfg):
        """
        Load point cloud data from a LAS file
        """
        xyz, time, las_vec, extra = loadLasCloud(file, cfg)
        return cls(time, xyz, las_vec, extra)
    
    def filterByMask(self, mask):
        """
        Filter points by masks
        """
        assert mask.shape[0] == self.xyz.shape[0], "xyz and id array must have the same length"

        self.time = self.time[mask]
        self.xyz = self.xyz[mask]
        self.las_vec = self.las_vec[mask]
        self.rsc_id = self.rsc_id[mask]
        self.extraData = self.extraData[mask]
        
        self.kdt = KDTree(self.xyz)
          
    def voxTracing(self, cfg):

        pcd_raw = o3d.geometry.PointCloud()
        pcd_raw.points = o3d.utility.Vector3dVector(self.xyz)

        pcd_raw = pcd_raw.voxel_down_sample(cfg['vox_size'])
        print(f"Raw: {self.xyz.shape[0]} pts -> Voxelized: {pcd_raw.points.__len__()} pts")

        self.kdt = KDTree(self.xyz)
        _, id_vox = self.kdt.query(np.array(pcd_raw.points), k=1, workers=-1)
        mask = np.zeros((self.xyz.shape[0],), dtype=bool)
        mask[id_vox] = True
        self.filterByMask(mask)

def issKpts(xyz,cfg):
    """
    Compute keypoints from a point cloud
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)

    if cfg['iss_vox_s'] > 0:
        pcd = pcd.voxel_down_sample(cfg['iss_vox_s'])

    return o3d.geometry.keypoint.compute_iss_keypoints(pcd,
                                                    salient_radius=cfg['iss_sln_r'],
                                                    non_max_radius=cfg['iss_nm_r'],
                                                    gamma_21=cfg['iss_g21'],
                                                    gamma_32=cfg['iss_g32'],
                                                    min_neighbors=cfg['iss_min_n'])

def downSampleKpts(tile, cfg):
    ''' 
    Downsample keypoints if number of keypoints per tile is above max value from config
    '''
    filt_kpts_id = []
    
    print(f"Initial kpts number: {len(tile.kpts_id)}")
    print(f"Filtering keypoints to {cfg['max_kpts']} per tile...")
    for i in np.unique(tile.rsc_id):
        
        kpts_id_tile_i = tile.kpts_id[tile.rsc_id[tile.kpts_id] == i]
        if len(kpts_id_tile_i) > cfg['max_kpts']:
            kpts_id_tile_i = np.random.choice(kpts_id_tile_i, cfg['max_kpts'], replace=False)
        filt_kpts_id.extend(kpts_id_tile_i)

    tile.kpts_id = np.array(filt_kpts_id)
    tile.n_kpts = len(tile.kpts_id)

def cleanKpts(tile_key, tile_target, cfg):
    """
    Check if keypoints have at least one target keypoint in vicinity
    If not, delete it
    """
 
    kdt = KDTree(tile_target.xyz[tile_target.kpts_id])
    dist, _ = kdt.query(tile_key.xyz[tile_key.kpts_id], workers=-1)

    tile_key.kpts_id = tile_key.kpts_id[dist < 2*cfg['uncertainty_r']]
    tile_key.n_kpts = len(tile_key.kpts_id)
    print(f"Final kpts number: {tile_key.n_kpts}")

def getCandidates(tile_key, tile_target, cfg):
    """
    Generate list of candidates for each keypoint to match,
    using batching and compact NumPy arrays to save memory.
    """

    kdt = KDTree(tile_target.xyz[tile_target.kpts_id])
    query_pts = tile_key.xyz[tile_key.kpts_id]

    batch_size = cfg["main_batch"]  # configurable batch size
    radius = 2 * cfg["uncertainty_r"]

    candidates = []
    for start in range(0, query_pts.shape[0], batch_size):
        end = min(start + batch_size, query_pts.shape[0])
        # Run query_ball_point on this batch
        batch_res = kdt.query_ball_point(query_pts[start:end], radius, workers=-1)
        # Convert each sublist into a compact NumPy array
        batch_res = [np.array(r, dtype=np.uint32) for r in batch_res]
        candidates.extend(batch_res)

    tile_key.candidates = candidates
def featSearch(tile_key, tile_target):
    """
    Find nearest neighbors in feature space for a set of keypoints
    """
    candidate = tile_key.candidates

    feats_k = tile_key.feat
    feats = tile_target.feat

    f_dist = np.empty((feats_k.shape[0]))
    idx_t = np.empty((feats_k.shape[0]), dtype=np.uint32)  

    for i in range(len(candidate)):
        #Build idx with only candidate points for kpt i
        flat_idx = faiss.IndexFlatL2(feats.shape[1])    
        flat_idx.add(feats[candidate[i]])   
        #Find nearest neigh in feat space for kpt i
        f_dist[i], idx_local = flat_idx.search(feats_k[i].reshape(1,-1), 1)
        #Find id of the candidate identified as match in the original cloud
        idx_t[i] = tile_target.kpts_id[candidate[i][idx_local]]

    return idx_t, f_dist, tile_target.xyz[idx_t]

def buildCorres(tile_key, tile_tgt, feat_dist_a):
    """
    Just formatting the correspondences so that we store the info somewhere.
    """
    kpts_xyz_a = tile_key.xyz[tile_key.kpts_id]
    corres_xyz_a = tile_tgt.xyz[tile_key.cor_id]

    correspondences = np.empty([kpts_xyz_a.shape[0], 14])
    correspondences[:, 0] = tile_key.kpts_id.reshape(-1) # Kpts id in native tile
    correspondences[:, 1] = tile_key.cor_id.reshape(-1) # Corres id in the target tile
    correspondences[:, 2] = np.linalg.norm((kpts_xyz_a-corres_xyz_a), axis=1) # Distance between the two points
    correspondences[:, 3] = feat_dist_a # Distance in feature space
    correspondences[:, 4:7] = kpts_xyz_a # Kpts coordinates
    correspondences[:, 7:10] = corres_xyz_a # Corres coordinates
    correspondences[:, 10] = tile_key.rsc_id[tile_key.kpts_id] #Tile id for RANSAC filtring
   
    return correspondences

def buildCorresFile(corres, tile_a, tile_b, cfg, icp_vec):
    """
    Build and save correspondences file for RANSAC and ICP stages.
    """
    ind_a = corres[:, 0].astype(int)
    ind_b = corres[:, 1].astype(int)
    time_a = tile_a.time[ind_a].reshape(-1, 1)
    time_b = tile_b.time[ind_b].reshape(-1, 1)

    if cfg['adjustLasVec']:
        las_vec_a = tile_a.las_vec[ind_a]
        las_vec_b = tile_b.las_vec[ind_b]
        out_data = np.concatenate((time_b, time_a, las_vec_b, las_vec_a), axis=1)
        out_data = out_data[out_data[:, 0].argsort()]
        np.savetxt(cfg['prj_folder'] + f"cor_outputs/LiDAR_p2p_noRefinement.txt",
                    out_data,
                    delimiter=',',
                    fmt='%.9f, %.9f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f')
        
        trj = np.loadtxt(cfg['trj_path'], delimiter=',', skiprows=1)
        trj_t = trj[:, 0]
        trj_q = trj[:, 4:]
        trj_q = trj_q[:, [1, 2, 3, 0]]  # w,x,y,z -> x,y,z,w according to Scipy notation
        las_vec_a = correctLasVecICP(time_a.reshape(-1), las_vec_a, cfg['R_sensor2body'], trj_t, trj_q, icp_vec)
        out_data = np.concatenate((time_b, time_a, las_vec_b, las_vec_a), axis=1)
        out_data = out_data[out_data[:, 0].argsort()]
        np.savetxt(cfg['prj_folder'] + f"cor_outputs/LiDAR_p2p.txt",
                    out_data,
                    fmt='%.9f, %.9f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f')

    else:
        xyz_a = tile_a.xyz[ind_a] + tile_a.xyz_shift
        xyz_b = tile_b.xyz[ind_b] + tile_b.xyz_shift

        if cfg['extendedLasData']:
            extra_data_a = tile_a.extraData[ind_a]
            extra_data_b = tile_b.extraData[ind_b]
            out_data = np.concatenate((time_b, time_a, xyz_b, xyz_a, icp_vec, extra_data_b, extra_data_a), axis=1)
            out_data = out_data[out_data[:, 0].argsort()]
            np.savetxt(cfg['prj_folder'] + f"cor_outputs/LiDAR_p2p.txt",
                        out_data,
                        header='time_b, time_a, x_b, y_b, z_b, x_a, y_a, z_a, icp_x, icp_y, icp_z, extra_b (multi col), extra_a (multi col), (xyz_a + icp_vec_a = refined xyz_a)',)
        else:        
            out_data = np.concatenate((time_b, time_a, xyz_b, xyz_a, icp_vec), axis=1)
            out_data = out_data[out_data[:, 0].argsort()]
            np.savetxt(cfg['prj_folder'] + f"cor_outputs/LiDAR_p2p.txt",
                        out_data,
                        fmt='%.9f, %.9f,  %.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f',
                        header='time_b, time_a, x_b, y_b, z_b, x_a, y_a, z_a, icp_x, icp_y, icp_z (xyz_a + icp_vec_a = refined xyz_a)')

def removeBoundaryPts(pcd, keypts, buffer):
    """
    Remove keypoints that are to close to the tile boundary to avoid border effect

    Parameters
    ----------
    pcd : open3d.Geometry.PointCloud()
    keypts : open3d.Geometry.PointCloud()
        point cloud that contain only the keypoints related to pcd
    buffer : float
        width of the buffer applied on pcd to filter keypts close to pcd boundary

    Returns
    -------
    cropped_kpts : open3d.Geometry.PointCloud()
        Filtered keypoints.

    """
    bbox = pcd.get_axis_aligned_bounding_box()
    boundary_pts = np.asarray(bbox.get_box_points())

    for i in range(boundary_pts.shape[0]):

        if boundary_pts[i, 0] == bbox.get_min_bound()[0]:
            boundary_pts[i, 0] = boundary_pts[i, 0] + buffer
        else:
            boundary_pts[i, 0] = boundary_pts[i, 0] - buffer

        if boundary_pts[i, 1] == bbox.get_min_bound()[1]:
            boundary_pts[i, 1] = boundary_pts[i, 1] + buffer
        else:
            boundary_pts[i, 1] = boundary_pts[i, 1] - buffer

    small_box = o3d.geometry.PointCloud()
    small_box.points = o3d.utility.Vector3dVector(boundary_pts)
    small_box = small_box.get_axis_aligned_bounding_box()

    cropped_kpts = keypts.crop(small_box)
    return cropped_kpts
