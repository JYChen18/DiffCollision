from dataclasses import dataclass
from typing import Literal
import torch

from diffcollision.core.base import _BaseCollision, DCContext
from diffcollision.core.rs1dist import _local_sample, RS1DistConfig


@dataclass
class RS1DirConfig(RS1DistConfig):
    """
    Configuration for `method="RS1Dir"` in `DiffCollision`.

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
    sample : str, optional
        Sampling strategy for local samples. Options are:
        - "adp": Adaptive sampling around witness points.
        - "fix": Fixed sampling around witness points.
        - "nbr": Neighbor-based sampling on the mesh surface.
        Default: "fix".
    n_global : int, optional
        Number of global samples. Required if `sample` is "adp" or "fix". Default: 1024.
    n_local : int, optional
        Number of local samples. Default: 16.
    n_level : int, optional
        Number of neighbor levels. Required if `sample` is "nbr". Default: 5.
    dthre : float, optional
        Distance threshold for local sampling, relative to the object scale.
        Required if `sample` is "fix". Default: 1.0.
    min_dthre : float, optional
        Minimum distance threshold for local sampling, absolute value.
        Required if `sample` is "adp" or "fix". Default: 0.01.
    eps : float, optional
        Softmax temperature for differentiable witness point computation. Default: 1e-3.
    """

    type: Literal["RS1Dir"] = "RS1Dir"
    eps: float = 1e-3

    @property
    def method(self):
        return RS1DirCollision


class RS1DirCollision(_BaseCollision):
    @staticmethod
    def backward(
        ctx, grad_wp1, grad_wp2, grad_n, grad_d_sign, grad_coll, grad_contact_counts
    ):
        (
            T1_raw,
            T2_raw,
            T1,
            T2,
            normal,
            wp1,
            wp2,
            d_sign,
            cvx_min_idx,
            coll,
            grad_wp1,
            grad_wp2,
            _grad_n,
            b,
            p,
            n_contact,
        ) = _BaseCollision.unpack_saved_tensors(ctx, grad_wp1, grad_wp2, grad_n)
        if n_contact == 0:
            return _BaseCollision.zero_grads(T1_raw, T2_raw)

        y = d_sign.unsqueeze(1) * (wp1 - wp2)
        cfg: RS1DirConfig = ctx.cfg
        dc_ctx: DCContext = ctx.dc_ctx

        with torch.no_grad():
            ls1_o, ls2_o = _local_sample(
                cfg, dc_ctx, T1, T2, wp1, wp2, normal, b, cvx_min_idx, coll, p
            )
            ls1_o, ls2_o = ls1_o[..., :3], ls2_o[..., :3]
            if ctx.vis is not None:
                ls1 = ls1_o @ T1[:, :3, :3].transpose(-1, -2) + T1[:, None, :3, 3]
                ls2 = ls2_o @ T2[:, :3, :3].transpose(-1, -2) + T2[:, None, :3, 3]
                batch_ids = coll // p
                counts = torch.bincount(batch_ids, minlength=b)
                offsets = counts.cumsum(dim=0) - counts
                slot_ids = torch.arange(
                    n_contact, device=coll.device
                ) - torch.repeat_interleave(offsets, counts)
                take_mask = slot_ids < cfg.nconmax
                ls1_vis = torch.zeros(
                    b, cfg.nconmax, cfg.n_local, 3, device=T1.device, dtype=T1.dtype
                )
                ls2_vis = torch.zeros_like(ls1_vis)
                ls1_vis[batch_ids[take_mask], slot_ids[take_mask]] = ls1[take_mask]
                ls2_vis[batch_ids[take_mask], slot_ids[take_mask]] = ls2[take_mask]
                ctx.vis.ls1.append(ls1_vis.detach().cpu())
                ctx.vis.ls2.append(ls2_vis.detach().cpu())

        def partial_sigma_x(points, x):
            z = points @ x
            a = torch.softmax(z / cfg.eps, dim=-1)  # Eq.(27)
            return points.T @ a

        def wp1_func(yi, T1i, p1i):
            p1_w = p1i @ T1i[:3, :3].T + T1i[:3, 3]
            wp1i = partial_sigma_x(p1_w, -yi)
            return wp1i

        def wp2_func(yi, T2i, p2i):
            p2_w = p2i @ T2i[:3, :3].T + T2i[:3, 3]
            wp2i = partial_sigma_x(p2_w, yi)
            return wp2i

        jacb_fun1 = torch.vmap(torch.func.jacrev(wp1_func, argnums=(0, 1)))
        Jp_wp1_y, Jp_wp1_T1 = jacb_fun1(y, T1, ls1_o)

        jacb_fun2 = torch.vmap(torch.func.jacrev(wp2_func, argnums=(0, 1)))
        Jp_wp2_y, Jp_wp2_T2 = jacb_fun2(y, T2, ls2_o)

        Idi = torch.eye(3, device=T1.device, dtype=T1.dtype)
        Id = Idi.unsqueeze(0).expand(n_contact, -1, -1)
        J_f_y = Id + d_sign[:, None, None] * (Jp_wp2_y - Jp_wp1_y)
        J_f_T1 = -d_sign[:, None, None] * Jp_wp1_T1.view(n_contact, 3, -1)
        J_f_T2 = d_sign[:, None, None] * Jp_wp2_T2.view(n_contact, 3, -1)

        J_y_T1 = torch.linalg.solve(J_f_y, -J_f_T1).view(n_contact, 3, 4, 4)
        J_y_T2 = torch.linalg.solve(J_f_y, -J_f_T2).view(n_contact, 3, 4, 4)

        J_wp1_T1 = Jp_wp1_T1 + torch.einsum("bij, bjkl -> bikl", Jp_wp1_y, J_y_T1)
        J_wp1_T2 = torch.einsum("bij, bjkl -> bikl", Jp_wp1_y, J_y_T2)
        J_wp2_T1 = torch.einsum("bij, bjkl -> bikl", Jp_wp2_y, J_y_T1)
        J_wp2_T2 = Jp_wp2_T2 + torch.einsum("bij, bjkl -> bikl", Jp_wp2_y, J_y_T2)

        grad1 = torch.einsum("cijk, ci -> cjk", J_wp1_T1, grad_wp1) + torch.einsum(
            "cijk, ci -> cjk", J_wp2_T1, grad_wp2
        )
        grad2 = torch.einsum("cijk, ci -> cjk", J_wp2_T2, grad_wp2) + torch.einsum(
            "cijk, ci -> cjk", J_wp1_T2, grad_wp1
        )
        return _BaseCollision.scatter_grads(T1_raw, T2_raw, coll, grad1, grad2)
