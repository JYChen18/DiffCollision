from dataclasses import dataclass
from typing import Literal
import torch
import numpy as np

from diffcollision.cpp._coal_openmp import batched_get_neighbor
from diffcollision.core.base import _BaseCollision, BaseCollisionConfig, DCContext
from diffcollision.utils import (
    local_sample_w_dthre,
    global_sample_v_and_f,
    torch_normalize_vector,
)


@dataclass
class RS1DistConfig(BaseCollisionConfig):
    """
    Configuration for `method="RS1Dist"` in `DiffCollision`.

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
    nthre : float, optional
        Normal threshold for local sampling.
        Required if `sample` is "adp" or "fix". Default: 2 * np.pi / 3 (unit: radian).
    """

    type: Literal["RS1Dist"] = "RS1Dist"
    sample: str = "fix"
    n_global: int = 1024
    n_local: int = 16
    n_level: int = 5
    dthre: float = 1.0
    min_dthre: float = 0.01
    nthre: float = 2 * np.pi / 3

    @property
    def method(self):
        return RS1DistCollision


def _prepare_for_backward(cfg: RS1DistConfig, dc_ctx: DCContext, batch):
    if cfg.sample == "adp":
        if dc_ctx.tp_o is None:
            raise ValueError(
                "Please specify target points `tp1_o` and `tp2_o` when using adaptive sampling"
            )
        if dc_ctx.tp1_o.shape[0] != batch:
            raise ValueError(
                "tp1_o and tp2_o batch dimension should match the transforms batch size"
            )

    # Prepare global samples and distance thresholds for each mesh pair. No update.
    n_mesh = len(dc_ctx.meshes)
    if dc_ctx.gs_o_mesh is None:
        dc_ctx.dthre_mesh = dc_ctx.ts.to(torch.zeros(n_mesh))
        dc_ctx.min_dthre_mesh = dc_ctx.ts.to(torch.zeros(n_mesh))
        dc_ctx.gs_o_mesh = dc_ctx.ts.to(torch.zeros(n_mesh, cfg.n_global, 6))
        for i, m in enumerate(dc_ctx.meshes):
            cm, fm = m.coarse_mesh, m.fine_mesh
            obj_scale = np.linalg.norm(cm.bounds[0] - cm.bounds[1])
            safe_dthre = 2 * np.sqrt(cfg.n_local * cm.area / np.pi / cfg.n_global)
            dc_ctx.dthre_mesh[i] = cfg.dthre * obj_scale
            dc_ctx.min_dthre_mesh[i] = max(safe_dthre, cfg.min_dthre)
            dc_ctx.gs_o_mesh[i, :, :3], dc_ctx.gs_o_mesh[i, :, 3:] = (
                global_sample_v_and_f(cm, fm, cfg.n_global)
            )

    # Prepare per mesh-pair parameters. Update when mesh pairs change.
    m2g_idx = torch.stack([dc_ctx.ml2mp_idx1, dc_ctx.ml2mp_idx2], dim=-1).reshape(-1)
    dc_ctx.dthre_pair = dc_ctx.dthre_mesh[m2g_idx].repeat(batch)
    dc_ctx.min_dthre_pair = dc_ctx.min_dthre_mesh[m2g_idx].repeat(batch)
    dc_ctx.gs_o_pair = dc_ctx.gs_o_mesh[m2g_idx].repeat(batch, 1, 1)
    return


def _local_sample(
    cfg: RS1DistConfig, dc_ctx: DCContext, T1, T2, wp1, wp2, normal, batch, cvx_min_idx
):
    if cfg.sample == "adp" or cfg.sample == "fix":
        if dc_ctx.gs_o_pair is None:
            _prepare_for_backward(cfg, dc_ctx, batch)
        tp_o, gs_o, n_local = dc_ctx.tp_o, dc_ctx.gs_o_pair, cfg.n_local
        dthre, min_dthre = dc_ctx.dthre_pair, dc_ctx.min_dthre_pair

        # Transform witness points from world frame to object local frame
        T = torch.stack([T1, T2], dim=-3).reshape(-1, 4, 4)
        wp = torch.stack([wp1, wp2], dim=-2).reshape(-1, 3)
        nn = torch.stack([normal, -normal], dim=-2).reshape(-1, 3)
        witness_o = torch.empty_like(gs_o[:, -1:])
        witness_o[:, -1, :3] = torch.einsum(
            "bji,bj->bi", T[:, :3, :3], wp - T[:, :3, 3]
        )
        witness_o[:, -1, 3:] = torch.einsum("bji,bj->bi", T[:, :3, :3], nn)

        # Local sampling around current witness points
        ls_o = local_sample_w_dthre(
            gs_o, tp_o, witness_o, dthre, min_dthre, cfg.nthre, n_local, cfg.sample
        )
        ls_o = ls_o.reshape(-1, 2, n_local, 6)
    elif cfg.sample == "nbr":
        ts, n_level, n_local = dc_ctx.ts, cfg.n_level, cfg.n_local
        ls_o = np.zeros((batch, 2, n_local, 6))
        normal1_o = torch.einsum("bji,bj->bi", T1[:, :3, :3], normal)
        normal2_o = torch.einsum("bji,bj->bi", T2[:, :3, :3], -normal)
        normal_o = torch.stack([normal1_o, normal2_o], dim=-2)
        cvx_idx = torch.cat(
            [dc_ctx.cl2cp_idx1[cvx_min_idx], dc_ctx.cl2cp_idx2[cvx_min_idx]],
            dim=-1,
        )
        batched_get_neighbor(
            dc_ctx.cvx_lst,
            cvx_idx.cpu().numpy().reshape(-1),
            normal_o.cpu().numpy().reshape(-1),
            2 * batch,
            n_level,
            n_local,
            cfg.n_thread,
            ls_o.reshape(-1),
        )
        ls_o = ts.to(ls_o)
    else:
        raise ValueError(
            f"Unknown sampling strategy: {cfg.sample}. Choices are 'adp', 'fix', 'nbr'."
        )
    return ls_o[:, 0], ls_o[:, 1]


class RS1DistCollision(_BaseCollision):
    @staticmethod
    def backward(ctx, grad_wp1, grad_wp2, grad_n, grad_d_sign, grad_mask):
        grad_wp1, grad_wp2, grad_n = _BaseCollision.pre_backward_logic(
            ctx, grad_wp1, grad_wp2, grad_n
        )
        T1_raw, T2_raw, dist_raw, normal_raw, wp1_raw, wp2_raw = ctx.saved_tensors
        b, p = T1_raw.shape[:2]
        T1, T2 = T1_raw.view(b * p, 4, 4), T2_raw.view(b * p, 4, 4)
        wp1, wp2 = wp1_raw.view(b * p, 3), wp2_raw.view(b * p, 3)
        dist, normal = dist_raw.view(b * p, 1), normal_raw.view(b * p, 3)
        cfg: RS1DistConfig = ctx.cfg
        dc_ctx: DCContext = ctx.dc_ctx

        with torch.no_grad():
            ls1_o, ls2_o = _local_sample(
                cfg, dc_ctx, T1, T2, wp1, wp2, normal, b, ctx.cvx_min_idx
            )
            if ctx.vis is not None:
                ls1 = (
                    ls1_o[..., :3] @ T1[:, :3, :3].transpose(-1, -2)
                    + T1[:, None, :3, 3]
                )
                ls2 = (
                    ls2_o[..., :3] @ T2[:, :3, :3].transpose(-1, -2)
                    + T2[:, None, :3, 3]
                )
                ctx.vis.ls1.append(ls1.detach().cpu().view(b, p, -1, 3))
                ctx.vis.ls2.append(ls2.detach().cpu().view(b, p, -1, 3))

        def wp_func(Ti, wpj, lsi_o):
            lsi = lsi_o[:, :3] @ Ti[:3, :3].T + Ti[:3, 3]
            pdist = (lsi - wpj).norm(dim=-1)
            weight = torch.softmax(-pdist / pdist.std().sqrt(), dim=-1)
            wpi = (weight.unsqueeze(-1) * lsi).sum(dim=-2)
            lti = lsi_o[:, 3:] @ Ti[:3, :3].T
            ni = (weight.unsqueeze(-1) * lti).sum(dim=-2)
            return (wpi, ni), ni

        jacb_fun = torch.vmap(torch.func.jacrev(wp_func, argnums=(0, 1), has_aux=True))

        ((J_f_T1, J_f_wp2), (J_fn_T1, J_fn_wp2)), n1 = jacb_fun(T1, wp2, ls1_o)
        ((J_g_T2, J_g_wp1), (J_gn_T2, J_gn_wp1)), n2 = jacb_fun(T2, wp1, ls2_o)

        # solve the system of equations:
        #   (1) J_wp1_T1 = J_f_wp2 @ J_wp2_T1 + J_f_T1
        #   (2) J_wp2_T1 = J_g_wp1 @ J_wp1_T1
        help_eye = dc_ctx.ts.to(torch.eye(3)[None].expand(b * p, -1, -1))
        J_wp1_T1 = torch.linalg.solve(
            help_eye - J_f_wp2 @ J_g_wp1, J_f_T1.view(b * p, 3, 16)
        ).view(b * p, 3, 4, 4)
        J_wp2_T1 = torch.einsum("bij, bjkl -> bikl", J_g_wp1, J_wp1_T1)
        J_wp2_T2 = torch.linalg.solve(
            help_eye - J_g_wp1 @ J_f_wp2, J_g_T2.view(b * p, 3, 16)
        ).view(b * p, 3, 4, 4)
        J_wp1_T2 = torch.einsum("bij, bjkl -> bikl", J_f_wp2, J_wp2_T2)

        # Chain rule
        grad1 = torch.einsum(
            "bpijk, bpi -> bpjk", J_wp1_T1.view(b, p, 3, 4, 4), grad_wp1
        ) + torch.einsum("bpijk, bpi -> bpjk", J_wp2_T1.view(b, p, 3, 4, 4), grad_wp2)
        grad2 = torch.einsum(
            "bpijk, bpi -> bpjk", J_wp2_T2.view(b, p, 3, 4, 4), grad_wp2
        ) + torch.einsum("bpijk, bpi -> bpjk", J_wp1_T2.view(b, p, 3, 4, 4), grad_wp1)

        if grad_n is not None and torch.any(grad_n != 0):
            J_n1_T1 = J_fn_T1 + torch.einsum("bij, bjkl -> bikl", J_fn_wp2, J_wp2_T1)
            J_n2_T1 = torch.einsum("bij, bjkl -> bikl", J_gn_wp1, J_wp1_T1)
            J_n1_T2 = torch.einsum("bij, bjkl -> bikl", J_fn_wp2, J_wp2_T2)
            J_n2_T2 = J_gn_T2 + torch.einsum("bij, bjkl -> bikl", J_gn_wp1, J_wp1_T2)

            def normal_func(n1i, n2i):
                return torch_normalize_vector(n1i - n2i)

            normal_jacb_fun = torch.vmap(torch.func.jacrev(normal_func, argnums=(0, 1)))
            J_n_n1, J_n_n2 = normal_jacb_fun(n1, n2)
            J_n_T1 = torch.einsum("bij, bjkl -> bikl", J_n_n1, J_n1_T1) + torch.einsum(
                "bij, bjkl -> bikl", J_n_n2, J_n2_T1
            )
            J_n_T2 = torch.einsum("bij, bjkl -> bikl", J_n_n1, J_n1_T2) + torch.einsum(
                "bij, bjkl -> bikl", J_n_n2, J_n2_T2
            )

            grad1 += torch.einsum(
                "bpijk, bpi -> bpjk", J_n_T1.view(b, p, 3, 4, 4), grad_n
            )
            grad2 += torch.einsum(
                "bpijk, bpi -> bpjk", J_n_T2.view(b, p, 3, 4, 4), grad_n
            )

        return grad1, grad2, None, None, None
