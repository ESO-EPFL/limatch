from dataclasses import dataclass
import numpy as np
import copy
import open3d as o3d
from scipy.spatial import KDTree
from lib.io import load_ascii_cloud, load_las_cloud

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
    def fromLAS(cls, file):
        """
        Load point cloud data from a LAS file and return Tile.
        """
        xyz, time, las_vec = load_las_cloud(file)
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
        print(f"Raw: {self.xyz.shape[0]} pts -> Voxelized: {pcd_down.points.__len__()} pts")

        mask = np.zeros((self.xyz.shape[0],), dtype=bool)

        if pcd_down.points.__len__() > 0:
            _, id_vox = self.kdt.query(np.asarray(pcd_down.points), k=1, workers=-1)
            mask[id_vox] = True

        self.apply_mask(mask)


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
    
    def to_array(self):
        N = self.idx_a.shape[0]
        out = np.zeros((N, 14), dtype=float)

        out[:, 0] = self.idx_a
        out[:, 1] = self.idx_b
        out[:, 2] = self.d_xyz
        out[:, 3] = self.d_feat
        out[:, 4:7] = self.xyz_a
        out[:, 7:10] = self.xyz_b
        out[:, 10] = self.rsc_id
        out[:, 11:14] = self.icp_vec

        return out

    def apply_mask(self, mask: np.ndarray):
        """
        Return a new Corres object containing only entries where mask==True.
        Mask is (N,) boolean.
        """
        return Corres(
            idx_a   = self.idx_a[mask],
            idx_b   = self.idx_b[mask],
            d_xyz   = self.d_xyz[mask],
            d_feat  = self.d_feat[mask],
            xyz_a   = self.xyz_a[mask],
            xyz_b   = self.xyz_b[mask],
            rsc_id  = self.rsc_id[mask],
            icp_vec = self.icp_vec[mask],
        )
    
def concatenate_corres(a: Corres, b: Corres) -> Corres:
    return Corres(
        idx_a = np.concatenate([a.idx_a, b.idx_a]),
        idx_b = np.concatenate([a.idx_b, b.idx_b]),
        d_xyz = np.concatenate([a.d_xyz, b.d_xyz]),
        d_feat = np.concatenate([a.d_feat, b.d_feat]),
        xyz_a = np.concatenate([a.xyz_a, b.xyz_a]),
        xyz_b = np.concatenate([a.xyz_b, b.xyz_b]),
        rsc_id = np.concatenate([a.rsc_id, b.rsc_id]),
        icp_vec = np.concatenate([a.icp_vec, b.icp_vec]),
    )