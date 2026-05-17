from dataclasses import dataclass, fields
from typing import get_type_hints
import torch
import numpy as np
import logging

from diffcollision.cpp._coal_openmp import batched_coal_distance
from diffcollision.wp_utils import _WarpSphereDist
from diffcollision.utils import DCTensorSpec
from diffcollision.io import DCMesh


@dataclass
class _BaseConfig:
    # --- Public API ---
    n_thread: int = 16  # cpu thread number for coal library
    tp1_o: torch.Tensor | None = None  # for adaptive sampling and visualization
    tp2_o: torch.Tensor | None = None
    egt: bool = True  # whether to enable equivalent gradient transport
    egt_step_r: float = 1.0  # the relative step between r and t matters
    egt_step_t: float = 0.001  # the relative step between r and t matters
    margin: float | list | torch.Tensor = (
        10.0  # convex-piece pairs with distance greater than margin are pruned in broad phase
    )
    per_env_max_contact_num: int = 20

    # --- Internal Fields ---
    _meshes: list[DCMesh] = None
    _mesh_ids: list[int] | torch.Tensor = None
    # External/public mesh id pairs, e.g. MuJoCo body ids. Returned in cpidx.
    _mesh_pair_ids: list[tuple[int, int]] | torch.Tensor = None
    # Internal compact mesh-list index pairs. Used only for tensor indexing.
    _mesh_pair_indices: torch.Tensor = None
    _margin: torch.Tensor = None
    _cvx_lst: list = None
    _sph_lst: torch.Tensor = None
    _ts: DCTensorSpec = None
    _warp_sphere_dist: _WarpSphereDist = None  # Save GPU memory (<1/10 of pytorch ops)

    _cvx_n_sum: torch.Tensor = None
    _cvx_min_idx: torch.Tensor = (
        None  # convex piece id that the witness point lies on, only used for neighbor sampling
    )
    _near_mask: torch.Tensor = None
    _ml2mp_idx1: torch.Tensor = None  # mesh list index -> mesh pair
    _ml2mp_idx2: torch.Tensor = None
    _cl2cp_idx1: torch.Tensor = None  # convex piece list -> convex piece pair
    _cl2cp_idx2: torch.Tensor = None
    _mp2cp_idx1: torch.Tensor = None  # mesh pair -> convex piece pair
    _mp2cp_idx2: torch.Tensor = None
    _cp2mp_idx: torch.Tensor = None  # convex piece pair -> mesh pair

    def _check_public_param(self):
        hints = get_type_hints(self.__class__)
        for f in fields(self):
            if f.name.startswith("_"):  # skip internal params
                continue
            expected_type = hints[f.name]
            value = getattr(self, f.name)
            if not isinstance(value, expected_type):
                raise TypeError(
                    f"Field '{f.name}' expects {expected_type}, got {type(value)} (value={value!r})"
                )
        return

    def __post_init__(self):
        self._check_public_param()
        self._ts = DCTensorSpec(
            device=self._meshes[0].bounding_spheres.device,
            dtype=self._meshes[0].bounding_spheres.dtype,
        )
        self._warp_sphere_dist = _WarpSphereDist(self._ts)

        n_mesh = len(self._meshes)
        if self._mesh_ids is None:
            self._mesh_ids = range(n_mesh)
        self._mesh_ids = self._ts.to_idx(self._mesh_ids)
        assert (
            len(self._mesh_ids.shape) == 1 and self._mesh_ids.shape[0] == n_mesh
        ), "mesh_ids should be a list or tensor with one id for each mesh"
        assert (
            len(set(self._mesh_ids.tolist())) == n_mesh
        ), "mesh_ids should be unique"

        if self._mesh_pair_ids is None:
            self._mesh_pair_indices = self._ts.to_idx(
                torch.triu_indices(n_mesh, n_mesh, offset=1).T
            )
            self._mesh_pair_ids = self._mesh_ids[self._mesh_pair_indices]

        self._sph_lst = []
        self._cvx_lst = []
        self._cvx_n_sum = []
        n_cvx_sum = 0
        for m in self._meshes:
            self._sph_lst.append(m.bounding_spheres)
            self._cvx_lst.extend(m.convex_pieces)
            self._cvx_n_sum.append(n_cvx_sum)
            n_cvx_sum += m.n_cvx
        self._sph_lst = torch.cat(self._sph_lst, dim=0)
        self.update_collision_pairs(
            self._mesh_pair_ids, self.tp1_o, self.tp2_o, self.margin
        )
        return

    def _set_mesh_pair_ids(self, mesh_pair_ids):
        self._mesh_pair_ids = self._ts.to_idx(mesh_pair_ids)
        assert (
            len(self._mesh_pair_ids.shape) == 2
            and self._mesh_pair_ids.shape[-1] == 2
        ), "collision_pairs should be a list of tuple of two mesh ids or a tensor of shape (n_pair, 2)"

        mesh_id_to_idx = {
            int(mesh_id): idx for idx, mesh_id in enumerate(self._mesh_ids.tolist())
        }
        self._mesh_pair_indices = self._ts.to_idx(
            [
                [mesh_id_to_idx[int(idx1)], mesh_id_to_idx[int(idx2)]]
                for idx1, idx2 in self._mesh_pair_ids.tolist()
            ]
        )

    def _set_pair_margins(self, margin):
        margin = self._ts.to(margin).reshape(-1)
        if margin.shape[0] == 1:
            margin = margin.expand(self._mesh_pair_ids.shape[0]).clone()
        assert margin.shape == (
            self._mesh_pair_ids.shape[0],
        ), "margin should be a scalar or a tensor with one value for each collision pair"
        self.margin = margin
        self._margin = margin

    def update_collision_pairs(self, collision_pairs, tp1_o, tp2_o, margin=None):
        self._set_mesh_pair_ids(collision_pairs)
        self._set_pair_margins(self.margin if margin is None else margin)
        self.tp1_o, self.tp2_o = tp1_o, tp2_o
        self._cp2mp_idx = []
        self._cl2cp_idx1 = []
        self._cl2cp_idx2 = []
        self._mp2cp_idx1 = []
        self._mp2cp_idx2 = []
        for i, (idx1, idx2) in enumerate(self._mesh_pair_indices.tolist()):
            n_cvx1, n_cvx2 = self._meshes[idx1].n_cvx, self._meshes[idx2].n_cvx
            self._cp2mp_idx.extend([i] * n_cvx1 * n_cvx2)
            self._cl2cp_idx1.append(
                (self._cvx_n_sum[idx1] + self._ts.to_idx(range(n_cvx1))).repeat(
                    n_cvx2
                )
            )
            self._cl2cp_idx2.append(
                (
                    self._cvx_n_sum[idx2] + self._ts.to_idx(range(n_cvx2))
                ).repeat_interleave(n_cvx1)
            )
            self._mp2cp_idx1.append(self._ts.to_idx([i] * n_cvx1).repeat(n_cvx2))
            self._mp2cp_idx2.append(
                self._ts.to_idx([i] * n_cvx2).repeat_interleave(n_cvx1)
            )
        self._ml2mp_idx1 = self._ts.to_idx(self._mesh_pair_indices[..., 0])
        self._ml2mp_idx2 = self._ts.to_idx(self._mesh_pair_indices[..., 1])
        self._cl2cp_idx1 = self._ts.to_idx(torch.cat(self._cl2cp_idx1))
        self._cl2cp_idx2 = self._ts.to_idx(torch.cat(self._cl2cp_idx2))
        self._mp2cp_idx1 = self._ts.to_idx(torch.cat(self._mp2cp_idx1))
        self._mp2cp_idx2 = self._ts.to_idx(torch.cat(self._mp2cp_idx2))
        self._cp2mp_idx = self._ts.to_idx(self._cp2mp_idx)
        return


class _BaseCollision(torch.autograd.Function):
    @staticmethod
    def forward(ctx, T1: torch.Tensor, T2: torch.Tensor, cfg: _BaseConfig, vis):
        cvx_lst, sph_lst, ts = cfg._cvx_lst, cfg._sph_lst, cfg._ts
        n_batch, n_mesh_pair = T2.shape[:2]  # b, p
        margin = cfg._margin.view(1, n_mesh_pair)
        batched_pair_idx = cfg._cp2mp_idx.expand(n_batch, -1)  # (b, k)

        # Broad-phase filter

        # min/max distance of bounding spheres
        sph1_o, sph2_o = sph_lst[cfg._cl2cp_idx1], sph_lst[cfg._cl2cp_idx2]  # (k, 4)
        s2s_max, s2s_min = cfg._warp_sphere_dist.forward(
            T1, T2, sph1_o, sph2_o, cfg._mp2cp_idx1, cfg._mp2cp_idx2
        )  # both (b, k)

        # min upper bound per mesh-pair
        s2s_max_sct = ts.to(torch.zeros((n_batch, n_mesh_pair)))
        s2s_max_sct.scatter_reduce_(
            1, batched_pair_idx, s2s_max, "amin", include_self=False
        )  # (b, p)

        # min lower bound per mesh-pair
        s2s_min_sct = ts.to(torch.zeros((n_batch, n_mesh_pair)))
        s2s_min_sct.scatter_reduce_(
            1, batched_pair_idx, s2s_min, "amin", include_self=False
        )  # (b, p)

        # prune convex-piece-pairs if they belong to faraway mesh-pairs
        broadphase_mask = s2s_min_sct < margin  # (b, p)
        near_cp_mask = broadphase_mask.gather(1, batched_pair_idx)  # (b, k)

        # prune convex-piece-pair if min >= max_sct
        valid = s2s_min - s2s_max_sct.gather(1, batched_pair_idx)  # (b, k)
        valid_idx = torch.where(((valid < 0) & near_cp_mask).view(-1))[0].cpu().numpy()

        n_cvx_pair = valid.shape[-1]
        n_valid = len(valid_idx)
        dist_out = np.zeros((n_batch, n_mesh_pair))
        normal_out = np.zeros((n_batch, n_mesh_pair, 3))
        wp1_out = np.zeros((n_batch, n_mesh_pair, 3))
        wp2_out = np.zeros((n_batch, n_mesh_pair, 3))
        min_idx_out = np.zeros((n_batch, n_mesh_pair), dtype=np.uintp)

        # Narrow-phase GJK
        if n_valid > 0:
            batched_coal_distance(
                cvx_lst,
                cfg._cl2cp_idx1.cpu().numpy().reshape(-1),
                T1.cpu().numpy().reshape(-1),
                cfg._cl2cp_idx2.cpu().numpy().reshape(-1),
                T2.cpu().numpy().reshape(-1),
                cfg._cp2mp_idx.cpu().numpy().reshape(-1),
                valid_idx,
                n_batch,
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

        dist, normal = ts.to(dist_out), ts.to(normal_out)
        wp1, wp2 = ts.to(wp1_out), ts.to(wp2_out)
        cvx_min_idx = ts.to_idx(min_idx_out)
        near_mask = broadphase_mask & (dist < margin)
        cvx_min_idx[~near_mask] = 0
        if torch.any(near_mask.sum(dim=-1) > cfg.per_env_max_contact_num):
            logging.warning("Valid contact number exceeds per_env_max_contact_num")
        d_sign = 2 * (dist > 0) - 1
        if dist[near_mask].shape[0] and dist[near_mask].max() > 1:
            logging.warning(f"Distance {dist[near_mask].max()}")

        cfg._cvx_min_idx = cvx_min_idx
        cfg._near_mask = ts.to_idx(near_mask)
        ctx.cfg = cfg
        ctx.vis = vis
        ctx.save_for_backward(T1, T2, dist, normal, wp1, wp2)
        return wp1, wp2, normal, d_sign, near_mask

    @staticmethod
    def pre_backward_logic(ctx, grad_wp1, grad_wp2, grad_n):
        cfg: _BaseConfig = ctx.cfg
        grad_wp1 = grad_wp1 * cfg._near_mask.unsqueeze(-1)
        grad_wp2 = grad_wp2 * cfg._near_mask.unsqueeze(-1)
        grad_n = grad_n * cfg._near_mask.unsqueeze(-1)
        return grad_wp1, grad_wp2, grad_n
