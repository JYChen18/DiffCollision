from dataclasses import dataclass
import torch
import numpy as np

from diffcollision.cpp._coal_openmp import batched_get_neighbor
from diffcollision.core.base import _BaseCollision, _BaseConfig
from diffcollision.utils import local_sample_w_dthre, global_sample_v_and_f


@dataclass
class RS1DistConfig(_BaseConfig):
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
    """

    sample: str = "fix"
    n_global: int = 1024
    n_local: int = 16
    n_level: int = 5
    dthre: float = 1.0
    min_dthre: float = 0.01

    # --- For each mesh ---
    _gs_o_mesh: torch.Tensor = None  # global samples in the object local frame
    _dthre_mesh: list[float] = None  # distance thresholds
    _min_dthre_mesh: list[float] = None  # min distance thresholds

    # --- For each collision pair ---
    _tp_o: torch.Tensor = None  # target points in the object local frame
    _gs_o_pair: torch.Tensor = None
    _dthre_pair: list[float] = None
    _min_dthre_pair: list[float] = None

    def prepare_for_backward(self, batch):
        if self.sample == "adp":
            if self.tp1_o is None or self.tp2_o is None:
                raise ValueError(
                    "Please specify target points `tp1_o` and `tp2_o` when using adaptive sampling"
                )
            else:
                assert (
                    len(self.tp1_o.shape) == 3 and len(self.tp2_o.shape) == 3
                ), "tp1_o and tp2_o should have shape (batch, n_pair, 3)"
                self._tp_o = torch.stack([self.tp1_o, self.tp2_o], dim=-2).reshape(
                    -1, 3
                )

        # Prepare global samples and distance thresholds for each mesh pair. No update.
        n_mesh = len(self._meshes)
        if self._gs_o_mesh is None:
            self._dthre_mesh = self._ts.to(torch.zeros(n_mesh))
            self._min_dthre_mesh = self._ts.to(torch.zeros(n_mesh))
            self._gs_o_mesh = self._ts.to(torch.zeros(n_mesh, self.n_global, 3))
            for i, m in enumerate(self._meshes):
                cm, fm = m.coarse_mesh, m.fine_mesh
                obj_scale = np.linalg.norm(cm.bounds[0] - cm.bounds[1])
                safe_dthre = 2 * np.sqrt(self.n_local * cm.area / np.pi / self.n_global)
                self._dthre_mesh[i] = self.dthre * obj_scale
                self._min_dthre_mesh[i] = max(safe_dthre, self.min_dthre)
                self._gs_o_mesh[i], _ = global_sample_v_and_f(cm, fm, self.n_global)

        # Prepare per mesh-pair parameters. Update when collision pairs change.
        m2g_idx = torch.stack([self._ml2mp_idx1, self._ml2mp_idx2], dim=-1).reshape(-1)
        self._dthre_pair = self._dthre_mesh[m2g_idx].repeat(batch)
        self._min_dthre_pair = self._min_dthre_mesh[m2g_idx].repeat(batch)
        self._gs_o_pair = self._gs_o_mesh[m2g_idx].repeat(batch, 1, 1)
        return

    def update_collision_pairs(self, collision_pairs, tp1_o, tp2_o):
        super().update_collision_pairs(collision_pairs, tp1_o, tp2_o)
        self._gs_o_pair = self._dthre_pair = self._min_dthre_pair = None
        return


def _local_sample(cfg: RS1DistConfig, T1, T2, wp1, wp2, normal, batch):
    if cfg.sample == "adp" or cfg.sample == "fix":
        if cfg._gs_o_pair is None:
            cfg.prepare_for_backward(batch)
        tp_o, gs_o, n_local = cfg._tp_o, cfg._gs_o_pair, cfg.n_local
        dthre, min_dthre = cfg._dthre_pair, cfg._min_dthre_pair

        # Transform witness points from world frame to object local frame
        T = torch.stack([T1, T2], dim=-3).reshape(-1, 4, 4)
        wp = torch.stack([wp1, wp2], dim=-2).reshape(-1, 3)
        gs_o[:, -1] = torch.einsum("bji,bj->bi", T[:, :3, :3], wp - T[:, :3, 3])

        # Local sampling around current witness points
        ls_o = local_sample_w_dthre(gs_o, tp_o, dthre, min_dthre, n_local, cfg.sample)
        ls_o = ls_o.reshape(-1, 2, n_local, 3)
    elif cfg.sample == "nbr":
        ts, n_level, n_local = cfg._ts, cfg.n_level, cfg.n_local
        ls_o = np.zeros((batch, 2, n_local, 3))
        normal1_o = torch.einsum("bji,bj->bi", T1[:, :3, :3], normal)
        normal2_o = torch.einsum("bji,bj->bi", T2[:, :3, :3], -normal)
        normal_o = torch.stack([normal1_o, normal2_o], dim=-2)
        cvx_idx = torch.cat(
            [cfg._cl2cp_idx1[cfg._cvx_min_idx], cfg._cl2cp_idx2[cfg._cvx_min_idx]],
            dim=-1,
        )
        batched_get_neighbor(
            cfg._cvx_lst,
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
    def backward(ctx, grad_wp1, grad_wp2, grad_d_sign):
        T1_raw, T2_raw, _, normal_raw, wp1_raw, wp2_raw = ctx.saved_tensors
        b, p = T1_raw.shape[:2]
        T1, T2 = T1_raw.view(b * p, 4, 4), T2_raw.view(b * p, 4, 4)
        wp1, wp2 = wp1_raw.view(b * p, 3), wp2_raw.view(b * p, 3)
        normal = normal_raw.view(b * p, 3)
        cfg: RS1DistConfig = ctx.cfg

        with torch.no_grad():
            ls1_o, ls2_o = _local_sample(cfg, T1, T2, wp1, wp2, normal, b)
            if ctx.vis is not None:
                ls1 = ls1_o @ T1[:, :3, :3].transpose(-1, -2) + T1[:, None, :3, 3]
                ls2 = ls2_o @ T2[:, :3, :3].transpose(-1, -2) + T2[:, None, :3, 3]
                ctx.vis.ls1.append(ls1.detach().cpu().view(b, p, -1, 3))
                ctx.vis.ls2.append(ls2.detach().cpu().view(b, p, -1, 3))

        def wp_func(Ti, wpj, lsi_o):
            lsi = lsi_o @ Ti[:3, :3].T + Ti[:3, 3]
            pdist = (lsi - wpj).norm(dim=-1)
            weight = torch.softmax(-pdist / pdist.std().sqrt(), dim=-1)
            wpi = (weight.unsqueeze(-1) * lsi).sum(dim=-2)
            return wpi

        jacb_fun = torch.vmap(torch.func.jacrev(wp_func, argnums=(0, 1)))

        # Assume wp2 is fixed when computing J_wp1_T1
        J_wp1_T1, J_wp1_wp2 = jacb_fun(T1, wp2, ls1_o)
        J_wp2_T2, J_wp2_wp1 = jacb_fun(T2, wp1, ls2_o)

        # Use precomputed J_wp1_T1 and J_wp2_T2 to avoid reusing the assumption
        J_wp2_T1 = torch.einsum("bij, bjkl -> bikl", J_wp2_wp1, J_wp1_T1)
        J_wp1_T2 = torch.einsum("bij, bjkl -> bikl", J_wp1_wp2, J_wp2_T2)

        # Chain rule
        grad1 = torch.einsum(
            "bpijk, bpi -> bpjk", J_wp1_T1.view(b, p, 3, 4, 4), grad_wp1
        ) + torch.einsum("bpijk, bpi -> bpjk", J_wp2_T1.view(b, p, 3, 4, 4), grad_wp2)
        grad2 = torch.einsum(
            "bpijk, bpi -> bpjk", J_wp2_T2.view(b, p, 3, 4, 4), grad_wp2
        ) + torch.einsum("bpijk, bpi -> bpjk", J_wp1_T2.view(b, p, 3, 4, 4), grad_wp1)
        return grad1, grad2, None, None
