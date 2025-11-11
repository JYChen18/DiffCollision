from dataclasses import dataclass
import torch

from diffcollision.core.base import _BaseConfig, _BaseCollision
from diffcollision.utils import point_to_triangle_distance_and_closest


@dataclass
class AnalyticalConfig(_BaseConfig):
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
    tp1_o : torch.Tensor, optional
        Target points in the **object local frame** on the first mesh of each collision pair.
        Used only for debugging and visualization. Default: None.
    tp2_o : torch.Tensor, optional
        Target points in the **object local frame** on the first mesh of each collision pair.
        Used only for debugging and visualization. Default: None.
    """

    pass


class AnalyticalCollision(_BaseCollision):
    @staticmethod
    def backward(ctx, grad_wp1, grad_wp2, grad_d_sign):
        T1, T2, _, _, wp1, wp2 = ctx.saved_tensors
        b, p = T1.shape[:2]
        cfg: AnalyticalConfig = ctx.cfg
        meshes, ts = cfg._meshes, cfg._ts

        wp1_o = torch.einsum("bpji,bpj->bpi", T1[:, :, :3, :3], wp1 - T1[:, :, :3, 3])
        wp2_o = torch.einsum("bpji,bpj->bpi", T2[:, :, :3, :3], wp2 - T2[:, :, :3, 3])
        p1_o, p2_o = [], []
        for i in range(p):
            cm1 = meshes[cfg._ml2mp_idx1[i]].coarse_mesh
            cm2 = meshes[cfg._ml2mp_idx2[i]].coarse_mesh
            _, _, f1 = cm1.nearest.on_surface(wp1_o[:, i].cpu().numpy())
            _, _, f2 = cm2.nearest.on_surface(wp2_o[:, i].cpu().numpy())
            p1_o.append(
                torch.cat([ts.to(cm1.triangles[f1]), wp1_o[:, i, None]], dim=-2)
            )
            p2_o.append(
                torch.cat([ts.to(cm2.triangles[f2]), wp2_o[:, i, None]], dim=-2)
            )
        p1_o, p2_o = torch.stack(p1_o, dim=1), torch.stack(p2_o, dim=1)

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
            T1.view(b * p, 4, 4),
            T2.view(b * p, 4, 4),
            p1_o.view(b * p, -1, 3),
            p2_o.view(b * p, -1, 3),
        )
        grad1 = torch.einsum(
            "bpijk, bpi -> bpjk", J_wp1_T1.view(b, p, 3, 4, 4), grad_wp1
        ) + torch.einsum("bpijk, bpi -> bpjk", J_wp2_T1.view(b, p, 3, 4, 4), grad_wp2)
        grad2 = torch.einsum(
            "bpijk, bpi -> bpjk", J_wp2_T2.view(b, p, 3, 4, 4), grad_wp2
        ) + torch.einsum("bpijk, bpi -> bpjk", J_wp1_T2.view(b, p, 3, 4, 4), grad_wp1)
        return grad1, grad2, None, None
