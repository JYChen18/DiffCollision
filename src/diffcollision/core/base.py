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

    # --- Internal Fields ---
    _meshes: list[DCMesh] = None
    _collision_pairs: list[tuple[int, int]] | torch.Tensor = None
    _cvx_lst: list = None
    _sph_lst: torch.Tensor = None
    _ts: DCTensorSpec = None
    _warp_sphere_dist: _WarpSphereDist = None  # Save GPU memory (<1/10 of pytorch ops)

    _cvx_n_sum: torch.Tensor = None
    _cvx_min_idx: torch.Tensor = None  # convex piece id that the witness point lies on
    _ml2mp_idx1: torch.Tensor = None  # mesh list -> mesh pair
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
        if self._collision_pairs is None:
            self._collision_pairs = torch.triu_indices(n_mesh, n_mesh, offset=1).T
        elif not isinstance(self._collision_pairs, torch.Tensor):
            self._collision_pairs = torch.tensor(self._collision_pairs)
        assert (
            len(self._collision_pairs.shape) == 2
            and self._collision_pairs.shape[-1] == 2
        ), "collision_pairs should be a list of tuple of two mesh indices or a tensor of shape (n_pair, 2)"

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
        self.update_collision_pairs(self._collision_pairs, self.tp1_o, self.tp2_o)
        return

    def update_collision_pairs(self, collision_pairs, tp1_o, tp2_o):
        self._collision_pairs = collision_pairs
        self.tp1_o, self.tp2_o = tp1_o, tp2_o
        self._cp2mp_idx = []
        self._cl2cp_idx1 = []
        self._cl2cp_idx2 = []
        self._mp2cp_idx1 = []
        self._mp2cp_idx2 = []
        for i, (idx1, idx2) in enumerate(self._collision_pairs):
            n_cvx1, n_cvx2 = self._meshes[idx1].n_cvx, self._meshes[idx2].n_cvx
            self._cp2mp_idx.extend([i] * n_cvx1 * n_cvx2)
            self._cl2cp_idx1.append(
                (self._cvx_n_sum[idx1] + torch.arange(n_cvx1)).repeat(n_cvx2)
            )
            self._cl2cp_idx2.append(
                (self._cvx_n_sum[idx2] + torch.arange(n_cvx2)).repeat_interleave(n_cvx1)
            )
            self._mp2cp_idx1.append(torch.tensor([i] * n_cvx1).repeat(n_cvx2))
            self._mp2cp_idx2.append(
                torch.tensor([i] * n_cvx2).repeat_interleave(n_cvx1)
            )
        self._ml2mp_idx1 = self._ts.to_idx(self._collision_pairs[..., 0])
        self._ml2mp_idx2 = self._ts.to_idx(self._collision_pairs[..., 1])
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
        n_batch, n_mesh_pair = T2.shape[:2]

        # Broad-phase filter based on bounding spheres
        sph1_o, sph2_o = sph_lst[cfg._cl2cp_idx1], sph_lst[cfg._cl2cp_idx2]  # k, 4
        s2s_max, s2s_min = cfg._warp_sphere_dist.forward(
            T1, T2, sph1_o, sph2_o, cfg._mp2cp_idx1, cfg._mp2cp_idx2
        )
        batched_pair_idx = cfg._cp2mp_idx[None].expand_as(s2s_max)  # b, k
        s2s_max_sct = ts.to(torch.zeros((n_batch, cfg._cp2mp_idx.max() + 1)))
        s2s_max_sct.scatter_reduce_(
            1, batched_pair_idx, s2s_max, "amin", include_self=False
        )  # b, p
        valid = s2s_min - s2s_max_sct.gather(1, batched_pair_idx)  # b, k
        valid_idx = torch.where(valid.view(-1) < 0)[0].cpu().numpy()

        # Narrow-phase GJK
        n_cvx_pair = valid.shape[-1]
        n_valid = len(valid_idx)
        dist_out = np.ones((n_batch, n_mesh_pair)) * 100
        normal_out = np.zeros((n_batch, n_mesh_pair, 3))
        wp1_out = np.zeros((n_batch, n_mesh_pair, 3))
        wp2_out = np.zeros((n_batch, n_mesh_pair, 3))
        min_idx_out = np.zeros((n_batch, n_mesh_pair), dtype=np.uintp)
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
        d_sign = 2 * (dist > 0) - 1
        if dist.max() > 1:
            logging.warning(f"Distance {dist.max()}")

        cfg._cvx_min_idx = ts.to_idx(min_idx_out)
        ctx.cfg = cfg
        ctx.vis = vis
        ctx.save_for_backward(T1, T2, dist, normal, wp1, wp2)
        return wp1, wp2, d_sign
