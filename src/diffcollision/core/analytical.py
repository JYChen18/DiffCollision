from dataclasses import dataclass
from typing import Literal
import torch

from diffcollision.core.base import BaseCollisionConfig, _BaseCollision, DCContext
from diffcollision.utils import point_to_triangle_distance_and_closest


@dataclass
class AnalyticalConfig(BaseCollisionConfig):
    """
    Configuration for `method="Analytical"` in `DiffCollision`.

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
    """

    type: Literal["Analytical"] = "Analytical"

    @property
    def method(self):
        return AnalyticalCollision


class AnalyticalCollision(_BaseCollision):
    @staticmethod
    def backward(
        ctx, grad_wp1, grad_wp2, grad_n, grad_d_sign, grad_coll, grad_contact_counts
    ):
        (
            T1_raw,
            T2_raw,
            T1,
            T2,
            _normal,
            wp1,
            wp2,
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
            return _BaseCollision.zero_grads(T1_raw, T2_raw)

        dc_ctx: DCContext = ctx.dc_ctx
        meshes, ts = dc_ctx.meshes, dc_ctx.ts

        wp1_o = torch.einsum("cji,cj->ci", T1[:, :3, :3], wp1 - T1[:, :3, 3])
        wp2_o = torch.einsum("cji,cj->ci", T2[:, :3, :3], wp2 - T2[:, :3, 3])
        pair_ids = coll % p
        p1_o, p2_o = [], []
        for i, pair_id in enumerate(pair_ids.tolist()):
            cm1 = meshes[int(dc_ctx.ml2mp_idx1[pair_id])].coarse_mesh
            cm2 = meshes[int(dc_ctx.ml2mp_idx2[pair_id])].coarse_mesh
            _, _, f1 = cm1.nearest.on_surface(wp1_o[i : i + 1].cpu().numpy())
            _, _, f2 = cm2.nearest.on_surface(wp2_o[i : i + 1].cpu().numpy())
            p1_o.append(
                torch.cat([ts.to(cm1.triangles[f1]), wp1_o[i : i + 1, None]], dim=-2)
            )
            p2_o.append(
                torch.cat([ts.to(cm2.triangles[f2]), wp2_o[i : i + 1, None]], dim=-2)
            )
        p1_o, p2_o = torch.cat(p1_o, dim=0), torch.cat(p2_o, dim=0)

        def x_func(T1i, T2i, p1i_o, p2i_o):
            p1_w = p1i_o @ T1i[:3, :3].T + T1i[:3, 3]
            p2_w = p2i_o @ T2i[:3, :3].T + T2i[:3, 3]
            _, wp2 = point_to_triangle_distance_and_closest(
                p1_w[-1][None], p2_w[:3][None]
            )
            _, wp1 = point_to_triangle_distance_and_closest(
                p2_w[-1][None], p1_w[:3][None]
            )
            return wp1.squeeze(0), wp2.squeeze(0)

        jacb_fun = torch.vmap(torch.func.jacrev(x_func, argnums=(0, 1)))
        (J_wp1_T1, J_wp1_T2), (J_wp2_T1, J_wp2_T2) = jacb_fun(
            T1,
            T2,
            p1_o,
            p2_o,
        )
        grad1 = torch.einsum("cijk, ci -> cjk", J_wp1_T1, grad_wp1) + torch.einsum(
            "cijk, ci -> cjk", J_wp2_T1, grad_wp2
        )
        grad2 = torch.einsum("cijk, ci -> cjk", J_wp2_T2, grad_wp2) + torch.einsum(
            "cijk, ci -> cjk", J_wp1_T2, grad_wp1
        )
        return _BaseCollision.scatter_grads(T1_raw, T2_raw, coll, grad1, grad2)
