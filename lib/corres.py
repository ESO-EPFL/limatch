from dataclasses import dataclass
import numpy as np

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
    def from_array(cls, arr: np.ndarray):
        """
        Accept the existing 14-col array and build a Corres object.
        """
        return cls(
            idx_a = arr[:, 0].astype(np.int32),
            idx_b = arr[:, 1].astype(np.int32),
            d_xyz = arr[:, 2],
            d_feat = arr[:, 3],
            xyz_a = arr[:, 4:7],
            xyz_b = arr[:, 7:10],
            rsc_id = arr[:, 10].astype(np.int32),
            icp_vec = arr[:, 11:14],
        )

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