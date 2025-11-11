from dataclasses import dataclass
import torch
import numpy as np

from diffcollision.cpp._coal_openmp import batched_coal_distance
from diffcollision.core.base import _BaseCollision, _BaseConfig


@dataclass
class FDConfig(_BaseConfig):
    """
    Configuration for `method="FD"` in `DiffCollision`.

    Parameters
    ----------
    n_thread: int, optional
        CPU thread number for coal library. Default: 16.
    egt : bool, optional
        Whether to enable equivalent gradient transport (EGT). Default: True.
    egt_step_r : float, optional
        The step size for the rotation in EGT. Default: 1.0.
    egt_step_t : float, optional
        The step size for the translation in EGT. The relative step between r and t matters. Default: 0.001.
    tp1_o : torch.Tensor, optional
        Target points in the **object local frame** on the first mesh of each collision pair.
        Used only for debugging and visualization. Default: None.
    tp2_o : torch.Tensor, optional
        Target points in the **object local frame** on the first mesh of each collision pair.
        Used only for debugging and visualization. Default: None.
    eps_r : float, optional
        Magnitude of rotation perturbation for finite difference estimation. Default: 0.1.
    eps_t : float, optional
        Magnitude of translation perturbation for finite difference estimation. Default: 0.01.
    """

    eps_r: float = 0.1
    eps_t: float = 0.01


class FDCollision(_BaseCollision):
    @staticmethod
    def backward(ctx, grad_wp1, grad_wp2, grad_d_sign):
        T1, T2, _, _, _, _ = ctx.saved_tensors  # batched
        cfg: FDConfig = ctx.cfg
        eps_r, eps_t, ts = cfg.eps_r, cfg.eps_t, cfg._ts
        cvx_lst, sph_lst = cfg._cvx_lst, cfg._sph_lst

        n_batch, n_mesh_pair = T1.shape[:2]
        n_jitter = 48  # n_jitter is fixed for FD
        delta = torch.zeros((24, 4, 4))
        rows, cols = torch.meshgrid(torch.arange(3), torch.arange(4), indexing="ij")
        rows = rows.flatten()
        cols = cols.flatten()
        pos_idx = torch.arange(3 * 4) * 2
        neg_idx = pos_idx + 1
        delta[pos_idx, rows, cols] = 1.0
        delta[neg_idx, rows, cols] = -1.0
        delta1 = torch.cat([delta, torch.zeros((24, 4, 4))], dim=0)
        delta2 = torch.cat([torch.zeros((24, 4, 4)), delta], dim=0)
        mask = ts.to(torch.ones(1, 1, 1, 4, 4)) * eps_r
        mask[..., :3, 3] *= eps_t / eps_r
        delta1 = ts.to(delta1).view(1, 48, 1, 4, 4) * mask
        delta2 = ts.to(delta2).view(1, 48, 1, 4, 4) * mask
        T1_new = T1.unsqueeze(1) + delta1  # b, j, p, 4, 4
        T2_new = T2.unsqueeze(1) + delta2

        # Broad-phase filter based on bounding spheres
        # NOTE: The current bounding sphere test may be wrong for concave objects in severe penetration
        sph1_o, sph2_o = sph_lst[cfg._cl2cp_idx1], sph_lst[cfg._cl2cp_idx2]  # k, 4
        s2s_max, s2s_min = cfg._warp_sphere_dist.forward(
            T1_new, T2_new, sph1_o, sph2_o, cfg._mp2cp_idx1, cfg._mp2cp_idx2
        )
        batched_pair_idx = cfg._cp2mp_idx[None, None].expand_as(s2s_max)  # b, j, k
        s2s_max_sct = ts.to(torch.zeros((n_batch, n_jitter, cfg._cp2mp_idx.max() + 1)))
        s2s_max_sct.scatter_reduce_(
            2, batched_pair_idx, s2s_max, "amin", include_self=False
        )  # b, j, p
        valid = s2s_min - s2s_max_sct.gather(2, batched_pair_idx)  # b, j, k
        valid_idx = torch.where(valid.view(-1) < 0.03)[0].cpu().numpy()

        # Narrow phase GJK
        n_cvx_pair = valid.shape[-1]
        n_valid = len(valid_idx)
        dist_out = np.ones((n_batch, n_jitter, n_mesh_pair)) * 100
        normal_out = np.zeros((n_batch, n_jitter, n_mesh_pair, 3))
        wp1_out = np.zeros((n_batch, n_jitter, n_mesh_pair, 3))
        wp2_out = np.zeros((n_batch, n_jitter, n_mesh_pair, 3))
        min_idx_out = np.zeros((n_batch, n_jitter, n_mesh_pair), dtype=np.uintp)
        batched_coal_distance(
            cvx_lst,
            cfg._cl2cp_idx1.cpu().numpy().reshape(-1),
            T1_new.cpu().numpy().reshape(-1),
            cfg._cl2cp_idx2.cpu().numpy().reshape(-1),
            T2_new.cpu().numpy().reshape(-1),
            cfg._cp2mp_idx.cpu().numpy().reshape(-1),
            valid_idx,
            n_batch * n_jitter,
            n_cvx_pair,
            n_mesh_pair,
            n_valid,
            cfg.n_thread,
            dist_out.reshape(-1),
            normal_out.reshape(-1),
            wp1_out.reshape(-1),
            wp2_out.reshape(-1),
            min_idx_out.reshape(-1),
        )
        wp1_jt, wp2_jt = ts.to(wp1_out), ts.to(wp2_out)

        J_wp1_T1 = torch.einsum(
            "bjpx, bjpyz->bpxyz", wp1_jt[:, 0::2] - wp1_jt[:, 1::2], delta1[:, 0::2]
        ) / (2 * mask**2)
        J_wp1_T2 = torch.einsum(
            "bjpx, bjpyz->bpxyz", wp1_jt[:, 0::2] - wp1_jt[:, 1::2], delta2[:, 0::2]
        ) / (2 * mask**2)
        J_wp2_T1 = torch.einsum(
            "bjpx, bjpyz->bpxyz", wp2_jt[:, 0::2] - wp2_jt[:, 1::2], delta1[:, 0::2]
        ) / (2 * mask**2)
        J_wp2_T2 = torch.einsum(
            "bjpx, bjpyz->bpxyz", wp2_jt[:, 0::2] - wp2_jt[:, 1::2], delta2[:, 0::2]
        ) / (2 * mask**2)

        grad1 = torch.einsum("bpxyz, bpx -> bpyz", J_wp1_T1, grad_wp1) + torch.einsum(
            "bpxyz, bpx -> bpyz", J_wp2_T1, grad_wp2
        )
        grad2 = torch.einsum("bpxyz, bpx -> bpyz", J_wp1_T2, grad_wp1) + torch.einsum(
            "bpxyz, bpx -> bpyz", J_wp2_T2, grad_wp2
        )

        return grad1, grad2, None, None
