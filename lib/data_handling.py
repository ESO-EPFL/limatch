from dataclasses import dataclass
import numpy as np
import sys
import copy
import open3d as o3d
from scipy.spatial import KDTree
from lib.io import load_ascii_cloud, load_las_cloud
from lib.map import Trajectory, simulate_las_vec, correct_laser_vector

import logging
logger = logging.getLogger("LiMatch")
from lib.logger import log_sub, log_sub_sub

class Tile:
    def __init__(self, time, xyz, lasvec):
        """
        Tile object storing pointwise data and small helpers.
        Backwards-compatible with previous Tile: same attribute names.

        Persistent (kept unless explicitly cleared):
            - time
            - xyz
            - las_vec
            - rsc_id
            - extraData
            - shift (may be set later)
            - kdt

        Ephemeral (created/cleared by pipeline steps):
            - kpts_id
            - n_kpts
            - feat
            - candidates
            - cor_id
        """
        self.time = time
        self.xyz = np.asarray(xyz)
        self.las_vec = lasvec.astype(np.float32)
        self.rsc_id = np.ones((self.xyz.shape[0],), dtype=np.uint16)

        self.kdt = KDTree(self.xyz)

    @classmethod
    def fromASCII(cls, file, cfg):
        """
        Load point cloud data from an ASCII file and return Tile.
        """
        xyz, time, las_vec = load_ascii_cloud(file, cfg)
        return cls(time, xyz, las_vec)

    @classmethod
    def fromLAS(cls, file, cfg):
        """
        Load point cloud data from a LAS file and return Tile.
        """
        xyz, time, las_vec = load_las_cloud(file, cfg)
        return cls(time, xyz, las_vec)

    def apply_mask(self, mask):
        """
        Apply a boolean mask or integer-index array to the tile to keep only
        selected points. Guarantees pointwise consistency and rebuilds KDTree.
        Also invalidates ephemeral attributes that refer to point indices.

        Parameters
        ----------
        mask : array-like
            - boolean array of length N (where N == len(self.xyz))
            - or an integer index array of selected indices
        """
        if mask is None:
            return

        mask = np.asarray(mask)

        # If mask contains integers (indices), convert to boolean mask
        if np.issubdtype(mask.dtype, np.integer):
            idx = mask
            bool_mask = np.zeros((self.xyz.shape[0],), dtype=bool)
            bool_mask[idx] = True
            mask = bool_mask
        else:
            if mask.dtype != bool:
                mask = mask.astype(bool)
            if mask.shape[0] != self.xyz.shape[0]:
                raise AssertionError("Mask length must match number of points in xyz")

        self.xyz = self.xyz[mask]
        if hasattr(self, "time") and self.time is not None and self.time.shape[0] == mask.shape[0]:
            self.time = self.time[mask]
        if hasattr(self, "las_vec") and self.las_vec is not None and self.las_vec.shape[0] == mask.shape[0]:
            self.las_vec = self.las_vec[mask]
        if hasattr(self, "rsc_id") and self.rsc_id is not None and self.rsc_id.shape[0] == mask.shape[0]:
            self.rsc_id = self.rsc_id[mask]


        self.rebuild_kdt()

        self.clear_keypoint_data()
        self.clear_features()
        self.clear_candidates()
        self.clear_correspondences()

    def rebuild_kdt(self):
        """Rebuild KDTree from current xyz."""
        if self.xyz.shape[0] == 0:
            self.kdt = KDTree(np.empty((0, 3)))
        else:
            self.kdt = KDTree(self.xyz)

    def apply_voxelization(self, cfg):
        """
        Voxel-downsamples the point cloud while preserving consistency via apply_mask. 
        """
        pcd_raw = o3d.geometry.PointCloud()
        pcd_raw.points = o3d.utility.Vector3dVector(self.xyz)

        pcd_down = pcd_raw.voxel_down_sample(cfg['vox_size'])
        log_sub_sub(logger, f"voxelized from {self.xyz.shape[0]} to {pcd_down.points.__len__()} pts ({100*pcd_down.points.__len__()/self.xyz.shape[0]:.2f}%)")

        mask = np.zeros((self.xyz.shape[0],), dtype=bool)

        if pcd_down.points.__len__() > 0:
            _, id_vox = self.kdt.query(np.asarray(pcd_down.points), k=1, workers=-1)
            mask[id_vox] = True

        self.apply_mask(mask)

    def copy_minimal(self):
        new = Tile.__new__(Tile)

        new.time = self.time.copy() if hasattr(self, "time") else None
        new.xyz = self.xyz.copy()
        new.las_vec = self.las_vec.copy() if hasattr(self, "las_vec") else None
        new.rsc_id = np.ones_like(self.rsc_id) 

        new.shift = getattr(self, "shift", None)

        new.rebuild_kdt()

        return new

    def apply_transform(self, R, t):
        self.xyz = (R @ self.xyz.T).T + t
        self.rebuild_kdt()  

    def clear_features(self):
        """Delete descriptor matrix if present."""
        if hasattr(self, "feat"):
            try:
                del self.feat
            except Exception:
                self.feat = None

    def clear_candidates(self):
        """Delete candidate lists if present."""
        if hasattr(self, "candidates"):
            try:
                del self.candidates
            except Exception:
                self.candidates = None

    def clear_correspondences(self):
        """Delete correspondence id mapping (cor_id) if present."""
        if hasattr(self, "cor_id"):
            try:
                del self.cor_id
            except Exception:
                self.cor_id = None

    def clear_keypoint_data(self):
        """
        Remove keypoint-related data that index into xyz (kpts_id and related fields).
        Called whenever xyz is changed.
        """
        if hasattr(self, "kpts_id"):
            try:
                del self.kpts_id
            except Exception:
                self.kpts_id = np.array([], dtype=int)
        if hasattr(self, "n_kpts"):
            try:
                del self.n_kpts
            except Exception:
                self.n_kpts = 0

    def assert_pointwise_consistency(self):
        """Run quick consistency checks (debugging)."""
        n = self.xyz.shape[0]
        if hasattr(self, "time"):
            assert self.time.shape[0] == n, "time length mismatch with xyz"
        if hasattr(self, "las_vec"):
            assert self.las_vec.shape[0] == n, "las_vec length mismatch with xyz"
        if hasattr(self, "rsc_id"):
            assert self.rsc_id.shape[0] == n, "rsc_id length mismatch with xyz"
        if hasattr(self, "extraData"):
            assert self.extraData.shape[0] == n, "extraData length mismatch with xyz"

@dataclass
class Corres:
    idx_a: np.ndarray        # (N,) int — keypoint id in tile A
    idx_b: np.ndarray        # (N,) int — matched keypoint id in tile B

    time_a: np.ndarray       # (N,) float — time of point in tile A
    time_b: np.ndarray       # (N,) float — time of point in tile B

    d_xyz: np.ndarray        # (N,) float — spatial distance (updated post-ICP)
    d_feat: np.ndarray       # (N,) float — descriptor distance

    xyz_a: np.ndarray        # (N, 3) float
    xyz_b: np.ndarray        # (N, 3) float (updated after ICP)

    rsc_id: np.ndarray       # (N,) int — region id for RANSAC grouping

    icp_vec: np.ndarray      # (N, 3) float — ICP translation from A to B
    @classmethod
    def empty(cls):
        """
        Create an empty Corres object with correctly-shaped zero-length fields.
        """
        return cls(
            idx_a=np.empty((0,), dtype=np.int32),
            idx_b=np.empty((0,), dtype=np.int32),
            time_a=np.empty((0,), dtype=float),
            time_b=np.empty((0,), dtype=float),
            d_xyz=np.empty((0,), dtype=float),
            d_feat=np.empty((0,), dtype=float),
            xyz_a=np.empty((0, 3), dtype=float),
            xyz_b=np.empty((0, 3), dtype=float),
            rsc_id=np.empty((0,), dtype=np.int32),
            icp_vec=np.empty((0, 3), dtype=float),
        )
    @classmethod
    def from_tiles(cls, tile_key, tile_tgt, feat_dist):
        """
        Create a Corres object from two tiles and a feature distance array.
        """
        xyz_a = tile_key.xyz[tile_key.kpts_id]
        xyz_b = tile_tgt.xyz[tile_key.cor_id]

        return cls(
            idx_a = tile_key.kpts_id.copy(),
            idx_b = tile_key.cor_id.copy(),
            time_a = tile_key.time[tile_key.kpts_id].copy(),
            time_b = tile_tgt.time[tile_key.cor_id].copy(),
            d_xyz = np.linalg.norm(xyz_a - xyz_b, axis=1),
            d_feat = feat_dist.copy(),
            xyz_a = xyz_a,
            xyz_b = xyz_b,
            rsc_id = tile_key.rsc_id[tile_key.kpts_id].copy(),
            icp_vec = np.zeros((xyz_a.shape[0], 3), dtype=float),
        )
      
    def deep_copy(self):
        """
        Create a deep copy of another Corres object.
        """
        return copy.deepcopy(self)
    
    def apply_mask(self, mask: np.ndarray):
        """
        Return a new Corres object containing only entries where mask==True.
        Mask is (N,) boolean.
        """
        return Corres(
            idx_a   = self.idx_a[mask],
            idx_b   = self.idx_b[mask],
            time_a  = self.time_a[mask],
            time_b  = self.time_b[mask],
            d_xyz   = self.d_xyz[mask],
            d_feat  = self.d_feat[mask],
            xyz_a   = self.xyz_a[mask],
            xyz_b   = self.xyz_b[mask],
            rsc_id  = self.rsc_id[mask],
            icp_vec = self.icp_vec[mask],
        )
    
    def save_ascii(self, path, shift):
        """
        Save correspondence data to an ASCII file.
        """
        time_a = self.time_a.reshape(-1, 1)
        time_b = self.time_b.reshape(-1, 1)

        xyz_a = self.xyz_a
        xyz_b = self.xyz_b

        icp_vec = self.icp_vec

        xyz_dist = self.d_xyz.reshape(-1, 1)

        time_sort = np.argsort(time_a.flatten())

        data = np.hstack((
            time_a,
            xyz_a + shift,
            time_b,
            xyz_b + shift,
            xyz_dist,
            icp_vec,
        ))
        data = data[time_sort, :]
        header = "time_a x_a y_a z_a time_b x_b y_b z_b d_xyz icp_x icp_y icp_z (xyz_a ≃ xyz_b + icp_vec)"
        fmt = "%.6f, %.3f, %.3f, %.3f, %.6f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f"

        np.savetxt(path, data, header=header, comments='', fmt=fmt)
           
def concatenate_corres(a: Corres, b: Corres) -> Corres:
    return Corres(
        idx_a = np.concatenate([a.idx_a, b.idx_a]),
        idx_b = np.concatenate([a.idx_b, b.idx_b]),
        time_a = np.concatenate([a.time_a, b.time_a]),
        time_b = np.concatenate([a.time_b, b.time_b]),
        d_xyz = np.concatenate([a.d_xyz, b.d_xyz]),
        d_feat = np.concatenate([a.d_feat, b.d_feat]),
        xyz_a = np.concatenate([a.xyz_a, b.xyz_a]),
        xyz_b = np.concatenate([a.xyz_b, b.xyz_b]),
        rsc_id = np.concatenate([a.rsc_id, b.rsc_id]),
        icp_vec = np.concatenate([a.icp_vec, b.icp_vec]),
    )

def prepare_overlap(tile_a: Tile, tile_b: Tile, cfg):
    '''
    Prepare the data for tiling by filtering out non-overlapping sections and assigning a tile id to each point
    '''
    if cfg['tile']:
        log_sub(logger, f"Tiling with size {cfg['step_x']}x{cfg['step_y']}...")
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

        if np.max(kept_id_a) == 0 or np.max(kept_id_b) == 0:
            log_sub(logger, "WARNING No valid overlapping tiles detected after tiling.")
            log_sub_sub(logger, "Check tile size and min density param. in configuration.")
            sys.exit()
        tile_a.apply_mask(kept_id_a > 0)
        tile_b.apply_mask(kept_id_b > 0)

    else:
        kept_id_a = np.ones((tile_a.xyz.shape[0],), dtype=np.uint8)
        kept_id_b = np.ones((tile_b.xyz.shape[0],), dtype=np.uint8)
        log_sub(logger, "No tiling... (all points kept)")

    shift = tile_a.xyz.mean(axis=0)

    tile_a.shift = shift
    tile_b.shift = shift

    log_sub(logger, f"Shifting point clouds toward origin...")
    with np.printoptions(precision=2, suppress=True):
        log_sub_sub(logger, f"{shift.flatten()} m...")
    log_sub_sub(logger, f"(Coordinates shifted back at export)")
    tile_a.xyz = (tile_a.xyz - shift).astype(np.float32)
    tile_b.xyz = (tile_b.xyz - shift).astype(np.float32)

    tile_a.rebuild_kdt()
    tile_b.rebuild_kdt()

    try:
        max_tile = int(np.max(kept_id_a))
    except Exception:
        max_tile = 0

    log_sub(logger, f"Generated {max_tile} valid tiles.")
    return

def build_output(c: Corres, tile_a: Tile, tile_b: Tile, cfg):
    """
    Build and save correspondences file for RANSAC and ICP stages.
    """

    if c is None or c.idx_a.shape[0] == 0:
        log_sub(logger, "No correspondences to save.")
        return

    if 'lasvec_source' in cfg and (cfg.get("lasvec_source", "simulate") or cfg.get("lasvec_source", "input")):
        traj = Trajectory.fromSBET(cfg["trajectory"])

        if cfg["lasvec_source"] == "simulate":
            log_sub(logger, "Simulating laser vectors for correspondences...")

            las_vec_a = simulate_las_vec(
                traj,
                time=c.time_a.reshape(-1,1),
                xyz= c.xyz_a + tile_a.shift,
                R_s2b=cfg['R_sensor2body'],
                a_s=cfg['lever_arm'],
                point_epsg=cfg['point_epsg'],
            )
            las_vec_b = simulate_las_vec(
                traj,
                time=c.time_b.reshape(-1,1),
                xyz= c.xyz_b + tile_b.shift,
                R_s2b=cfg['R_sensor2body'],
                a_s=cfg['lever_arm'],
                point_epsg=cfg['point_epsg'],
            )
        elif cfg["lasvec_source"] == "input":
            log_sub(logger, "Using input laser vectors for correspondences...")
            las_vec_a = tile_a.las_vec[c.idx_a]
            las_vec_b = tile_b.las_vec[c.idx_b]
        assert las_vec_a.shape[1] == 3
        assert las_vec_a.shape == las_vec_b.shape

        p2p = np.concatenate((c.time_b.reshape(-1,1), c.time_a.reshape(-1,1), las_vec_b, las_vec_a), axis=1)
        p2p = p2p[p2p[:, 0].argsort()]

        log_sub(logger, "Correcting laser vectors B with ICP results...")
        las_vec_b_corrected = correct_laser_vector(traj,
                                                las_vec_b,
                                                cfg['R_sensor2body'],
                                                c.time_b,
                                                c.icp_vec)
        
        p2p_icp = np.concatenate((c.time_b.reshape(-1,1), c.time_a.reshape(-1,1), las_vec_b_corrected, las_vec_a), axis=1)
        p2p_icp = p2p_icp[p2p_icp[:, 0].argsort()]

        log_sub(logger, f"Saving to {cfg['prj_folder']}cor_outputs/")
        
        np.savetxt(cfg['prj_folder'] + f"cor_outputs/LiDAR_p2p_rsc_{cfg['tile_id']}.txt",
                    p2p,
                    delimiter=',',
                    fmt='%.9f, %.9f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f')
        np.savetxt(cfg['prj_folder'] + f"cor_outputs/LiDAR_p2p_{cfg['tile_id']}.txt",
                    p2p_icp,
                    fmt='%.9f, %.9f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f')
    else:
        log_sub(logger, "No laser vector processing requested, or option invalid, should be 'simulate' or 'input'.")

    assert c.icp_vec.shape == c.xyz_a.shape
    assert c.time_a.shape == c.time_b.shape
    assert c.xyz_a.shape == c.xyz_b.shape
    assert c.icp_vec.shape == c.xyz_a.shape

    c.save_ascii(cfg['prj_folder'] + f"cor_outputs/corres_{cfg['tile_id']}.txt", tile_a.shift)






    