from dataclasses import dataclass
import torch

from diffcollision.core.base import _BaseCollision
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
    tp1_o : torch.Tensor, optional
        Target points in the **object local frame** on the first mesh of each collision pair.
        Required if `sample` is "adp", otherwise used only for debugging. Default: None.
    tp2_o : torch.Tensor, optional
        Target points in the **object local frame** on the first mesh of each collision pair.
        Required if `sample` is "adp", otherwise used only for debugging. Default: None.
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

    eps: float = 1e-3


class RS1DirCollision(_BaseCollision):
    @staticmethod
    def backward(ctx, grad_wp1, grad_wp2, grad_d_sign):
        T1_raw, T2_raw, dist_raw, normal_raw, wp1_raw, wp2_raw = ctx.saved_tensors
        b, p = T1_raw.shape[:2]
        T1, T2 = T1_raw.view(b * p, 4, 4), T2_raw.view(b * p, 4, 4)
        wp1, wp2 = wp1_raw.view(b * p, 3), wp2_raw.view(b * p, 3)
        dist, normal = dist_raw.view(b * p), normal_raw.view(b * p, 3)
        d_sign = (dist > 0).int() * 2 - 1
        y = d_sign.unsqueeze(1) * (wp1 - wp2)
        cfg: RS1DirConfig = ctx.cfg

        with torch.no_grad():
            ls1_o, ls2_o = _local_sample(cfg, T1, T2, wp1, wp2, normal, b)
            if ctx.vis is not None:
                ls1 = ls1_o @ T1[:, :3, :3].transpose(-1, -2) + T1[:, None, :3, 3]
                ls2 = ls2_o @ T2[:, :3, :3].transpose(-1, -2) + T2[:, None, :3, 3]
                ctx.vis.ls1.append(ls1.detach().cpu())
                ctx.vis.ls2.append(ls2.detach().cpu())

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

        Idi = torch.eye(3, device=T1.device, dtype=T1.dtype)  # (3,3)
        Id = Idi.unsqueeze(0).expand(b * p, -1, -1)
        J_f_y = Id + d_sign[:, None, None] * (Jp_wp2_y - Jp_wp1_y)  # Eq.(19)
        J_f_T1 = -d_sign[:, None, None] * Jp_wp1_T1.view(b * p, 3, -1)
        J_f_T2 = d_sign[:, None, None] * Jp_wp2_T2.view(b * p, 3, -1)

        # Implicit function differentiation. X = torch.linalg.solve(A, B) satisfies AX=B
        J_y_T1 = torch.linalg.solve(J_f_y, -J_f_T1).view(b * p, 3, 4, 4)
        J_y_T2 = torch.linalg.solve(J_f_y, -J_f_T2).view(b * p, 3, 4, 4)

        J_wp1_T1 = Jp_wp1_T1 + torch.einsum("bij, bjkl -> bikl", Jp_wp1_y, J_y_T1)
        J_wp1_T2 = torch.einsum("bij, bjkl -> bikl", Jp_wp1_y, J_y_T2)
        J_wp2_T1 = torch.einsum("bij, bjkl -> bikl", Jp_wp2_y, J_y_T1)
        J_wp2_T2 = Jp_wp2_T2 + torch.einsum("bij, bjkl -> bikl", Jp_wp2_y, J_y_T2)

        grad1 = torch.einsum(
            "bpijk, bpi -> bpjk", J_wp1_T1.view(b, p, 3, 4, 4), grad_wp1
        ) + torch.einsum("bpijk, bpi -> bpjk", J_wp2_T1.view(b, p, 3, 4, 4), grad_wp2)
        grad2 = torch.einsum(
            "bpijk, bpi -> bpjk", J_wp2_T2.view(b, p, 3, 4, 4), grad_wp2
        ) + torch.einsum("bpijk, bpi -> bpjk", J_wp1_T2.view(b, p, 3, 4, 4), grad_wp1)
        return grad1, grad2, None, None
