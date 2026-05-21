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
    cfg: RS1DistConfig,
    dc_ctx: DCContext,
    T1,
    T2,
    wp1,
    wp2,
    normal,
    batch,
    cvx_min_idx,
    coll=None,
    n_mesh_pair=None,
):
    n_contact = T1.shape[0]
    if cfg.sample == "adp" or cfg.sample == "fix":
        if n_mesh_pair is None:
            n_mesh_pair = dc_ctx.mesh_pair_ids.shape[0]
        expected_pair_samples = 2 * batch * n_mesh_pair
        if (
            dc_ctx.gs_o_pair is None
            or dc_ctx.gs_o_pair.shape[0] != expected_pair_samples
        ):
            _prepare_for_backward(cfg, dc_ctx, batch)
        tp_o, gs_o, n_local = dc_ctx.tp_o, dc_ctx.gs_o_pair, cfg.n_local
        dthre, min_dthre = dc_ctx.dthre_pair, dc_ctx.min_dthre_pair
        if coll is not None:
            pair_sample_idx = (
                coll[:, None] * 2
                + torch.arange(2, device=coll.device, dtype=coll.dtype)
            ).reshape(-1)
            gs_o = gs_o[pair_sample_idx]
            dthre = dthre[pair_sample_idx]
            min_dthre = min_dthre[pair_sample_idx]
            if tp_o is not None:
                tp_o = tp_o[pair_sample_idx]

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
        ls_o = np.zeros((n_contact, 2, n_local, 6))
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
            2 * n_contact,
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
            _d_sign,
            cvx_min_idx,
            coll,
            grad_wp1,
            grad_wp2,
            grad_n,
            b,
            p,
            n_contact,
        ) = _BaseCollision.unpack_saved_tensors(ctx, grad_wp1, grad_wp2, grad_n)
        cfg: RS1DistConfig = ctx.cfg
        dc_ctx: DCContext = ctx.dc_ctx
        if n_contact == 0:
            return _BaseCollision.zero_grads(T1_raw, T2_raw)

        with torch.no_grad():
            ls1_o, ls2_o = _local_sample(
                cfg, dc_ctx, T1, T2, wp1, wp2, normal, b, cvx_min_idx, coll, p
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

        help_eye = dc_ctx.ts.to(torch.eye(3)[None].expand(n_contact, -1, -1))
        J_wp1_T1 = torch.linalg.solve(
            help_eye - J_f_wp2 @ J_g_wp1, J_f_T1.view(n_contact, 3, 16)
        ).view(n_contact, 3, 4, 4)
        J_wp2_T1 = torch.einsum("bij,bjkl->bikl", J_g_wp1, J_wp1_T1)
        J_wp2_T2 = torch.linalg.solve(
            help_eye - J_g_wp1 @ J_f_wp2, J_g_T2.view(n_contact, 3, 16)
        ).view(n_contact, 3, 4, 4)
        J_wp1_T2 = torch.einsum("bij,bjkl->bikl", J_f_wp2, J_wp2_T2)

        grad1 = torch.einsum("cijk,ci->cjk", J_wp1_T1, grad_wp1) + torch.einsum(
            "cijk, ci -> cjk", J_wp2_T1, grad_wp2
        )
        grad2 = torch.einsum("cijk,ci->cjk", J_wp2_T2, grad_wp2) + torch.einsum(
            "cijk, ci -> cjk", J_wp1_T2, grad_wp1
        )

        if torch.any(grad_n != 0):
            J_n1_T1 = J_fn_T1 + torch.einsum("bij,bjkl->bikl", J_fn_wp2, J_wp2_T1)
            J_n2_T1 = torch.einsum("bij,bjkl->bikl", J_gn_wp1, J_wp1_T1)
            J_n1_T2 = torch.einsum("bij,bjkl->bikl", J_fn_wp2, J_wp2_T2)
            J_n2_T2 = J_gn_T2 + torch.einsum("bij,bjkl->bikl", J_gn_wp1, J_wp1_T2)

            def normal_func(n1i, n2i):
                return torch_normalize_vector(n1i - n2i)

            normal_jacb_fun = torch.vmap(torch.func.jacrev(normal_func, argnums=(0, 1)))
            J_n_n1, J_n_n2 = normal_jacb_fun(n1, n2)
            J_n_T1 = torch.einsum("bij,bjkl->bikl", J_n_n1, J_n1_T1) + torch.einsum(
                "bij, bjkl -> bikl", J_n_n2, J_n2_T1
            )
            J_n_T2 = torch.einsum("bij,bjkl->bikl", J_n_n1, J_n1_T2) + torch.einsum(
                "bij, bjkl -> bikl", J_n_n2, J_n2_T2
            )

            grad1 += torch.einsum("cijk,ci->cjk", J_n_T1, grad_n)
            grad2 += torch.einsum("cijk,ci->cjk", J_n_T2, grad_n)

        return _BaseCollision.scatter_grads(T1_raw, T2_raw, coll, grad1, grad2)
