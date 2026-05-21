from dataclasses import dataclass
from typing import Literal
import torch
import numpy as np

from diffcollision.cpp._coal_openmp import batched_coal_distance
from diffcollision.core.base import _BaseCollision, BaseCollisionConfig, DCContext


@dataclass
class RS0Config(BaseCollisionConfig):
    """
    Configuration for `method="RS0"` in `DiffCollision`.

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
    n_jitter : int, optional
        Number of jittered transformations for finite difference estimation. Default: 48.
    eps_r : float, optional
        Magnitude of rotation perturbation for finite difference estimation. Default: 0.1.
    eps_t : float, optional
        Magnitude of translation perturbation for finite difference estimation. Default: 0.01.
    """

    type: Literal["RS0"] = "RS0"
    n_jitter: int = 48
    eps_r: float = 0.1
    eps_t: float = 0.01

    @property
    def method(self):
        return RS0Collision


class RS0Collision(_BaseCollision):
    @staticmethod
    def backward(
        ctx, grad_wp1, grad_wp2, grad_n, grad_d_sign, grad_coll, grad_contact_counts
    ):
        (
            T1,
            T2,
            _T1_contact,
            _T2_contact,
            _normal,
            _wp1,
            _wp2,
            _d_sign,
            _cvx_min_idx,
            coll,
            grad_wp1,
            grad_wp2,
            _grad_n,
            b,
            p,
            n_contact,
        ) = _BaseCollision.unpack_saved_tensors(ctx, grad_wp1, grad_wp2, grad_n)
        if n_contact == 0:
            return _BaseCollision.zero_grads(T1, T2)
        grad_wp1_dense = torch.zeros(b * p, 3, device=T1.device, dtype=T1.dtype)
        grad_wp2_dense = torch.zeros_like(grad_wp1_dense)
        grad_wp1_dense.index_add_(0, coll, grad_wp1)
        grad_wp2_dense.index_add_(0, coll, grad_wp2)
        grad_wp1 = grad_wp1_dense.view(b, p, 3)
        grad_wp2 = grad_wp2_dense.view(b, p, 3)
        cfg: RS0Config = ctx.cfg
        dc_ctx: DCContext = ctx.dc_ctx
        eps_r, eps_t = cfg.eps_r, cfg.eps_t
        n_jitter, ts = cfg.n_jitter, dc_ctx.ts
        cvx_lst, sph_lst = dc_ctx.cvx_lst, dc_ctx.sph_lst

        n_batch, n_mesh_pair = T1.shape[:2]
        mask = ts.to(torch.ones(1, 1, 1, 4, 4)) * eps_r
        mask[..., :3, 3] *= eps_t / eps_r
        T1_jitter = ts.to(torch.randn(n_batch, n_jitter, n_mesh_pair, 4, 4)) * mask
        T2_jitter = ts.to(torch.randn(n_batch, n_jitter, n_mesh_pair, 4, 4)) * mask
        T1_new = T1.unsqueeze(1) + T1_jitter  # b, j, p, 4, 4
        T2_new = T2.unsqueeze(1) + T2_jitter

        # Broad-phase filter based on bounding spheres
        # NOTE: The current bounding sphere test may be wrong for concave objects in severe penetration
        sph1_o, sph2_o = sph_lst[dc_ctx.cl2cp_idx1], sph_lst[dc_ctx.cl2cp_idx2]  # k, 4
        s2s_max, s2s_min = dc_ctx.warp_sphere_dist.forward(
            T1_new, T2_new, sph1_o, sph2_o, dc_ctx.mp2cp_idx1, dc_ctx.mp2cp_idx2
        )
        batched_pair_idx = dc_ctx.cp2mp_idx[None, None].expand_as(s2s_max)  # b, j, k
        s2s_max_sct = ts.to(
            torch.zeros((n_batch, n_jitter, dc_ctx.cp2mp_idx.max() + 1))
        )
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
            dc_ctx.cl2cp_idx1.cpu().numpy().reshape(-1),
            T1_new.cpu().numpy().reshape(-1),
            dc_ctx.cl2cp_idx2.cpu().numpy().reshape(-1),
            T2_new.cpu().numpy().reshape(-1),
            dc_ctx.cp2mp_idx.cpu().numpy().reshape(-1),
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

        J_wp1_T1 = torch.einsum("bjpx, bjpyz->bpxyz", wp1_jt, T1_jitter) / (
            n_jitter * mask**2
        )
        J_wp1_T2 = torch.einsum("bjpx, bjpyz->bpxyz", wp1_jt, T2_jitter) / (
            n_jitter * mask**2
        )
        J_wp2_T1 = torch.einsum("bjpx, bjpyz->bpxyz", wp2_jt, T1_jitter) / (
            n_jitter * mask**2
        )
        J_wp2_T2 = torch.einsum("bjpx, bjpyz->bpxyz", wp2_jt, T2_jitter) / (
            n_jitter * mask**2
        )

        grad1 = torch.einsum("bpxyz, bpx -> bpyz", J_wp1_T1, grad_wp1) + torch.einsum(
            "bpxyz, bpx -> bpyz", J_wp2_T1, grad_wp2
        )
        grad2 = torch.einsum("bpxyz, bpx -> bpyz", J_wp1_T2, grad_wp1) + torch.einsum(
            "bpxyz, bpx -> bpyz", J_wp2_T2, grad_wp2
        )

        return grad1, grad2, None, None, None
