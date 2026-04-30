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
    skip_bound: float = 0.01

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
        n_batch, n_mesh_pair = T2.shape[:2]  # b, p
        skip_bound = cfg.skip_bound
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

        # for each mesh-pair, find the convex-piece-pair with smallest s2s_min
        best_cp_idx = ts.to_idx(torch.zeros((n_batch, n_mesh_pair)))
        for mp_idx in range(n_mesh_pair):
            cp_mask = cfg._cp2mp_idx == mp_idx
            cp_idx = torch.where(cp_mask)[0]
            local_s2s_min = s2s_min[:, cp_mask]  # (b, n_cp_for_this_mesh_pair)
            local_argmin = local_s2s_min.argmin(dim=1)  # (b,)
            best_cp_idx[:, mp_idx] = cp_idx[local_argmin]

        # closest bounding spheres per mesh-pair
        best_sph1 = sph1_o[best_cp_idx]  # (b, p, 4)
        best_sph2 = sph2_o[best_cp_idx]  # (b, p, 4)

        # apply transforms
        c1 = (
            torch.einsum("bpij,bpj->bpi", T1[..., :3, :3], best_sph1[..., :3])
            + T1[..., :3, 3]
        )  # (b, p, 3)
        c2 = (
            torch.einsum("bpij,bpj->bpi", T2[..., :3, :3], best_sph2[..., :3])
            + T2[..., :3, 3]
        )  # (b, p, 3)

        r1 = best_sph1[..., 3:4]  # (b, p, 1)
        r2 = best_sph2[..., 3:4]  # (b, p, 1)
        delta = c2 - c1
        center_dist = delta.norm(dim=-1, keepdim=True)  # (b, p, 1)

        # normals for sphere-defined collisions
        normal_sph = delta / center_dist.clamp_min(1e-8)
        fallback_axis = ts.to(torch.tensor([1.0, 0.0, 0.0])).view(1, 1, 3)
        normal_sph = torch.where(
            center_dist > 1e-8, normal_sph, fallback_axis.expand_as(normal_sph)
        )  # (b, p, 3)

        # nearest points for sphere-defined collisions
        wp1_sph = c1 + normal_sph * r1  # (b, p, 3)
        wp2_sph = c2 - normal_sph * r2  # (b, p, 3)

        # distance for sphere-defined collisions
        dist_sph = center_dist[..., 0] - r1[..., 0] - r2[..., 0]  # (b, p)

        # prune convex-piece-pairs if they belong to faraway mesh-pairs
        far_mask = s2s_min_sct > skip_bound  # (b, p)
        far_cp_mask = far_mask.gather(1, batched_pair_idx)  # (b, k)
        # prune convex-piece-pair if min >= max_sct
        valid = s2s_min - s2s_max_sct.gather(1, batched_pair_idx)  # (b, k)
        valid_idx = (
            torch.where(((valid < 0) & (~far_cp_mask)).view(-1))[0].cpu().numpy()
        )

        n_cvx_pair = valid.shape[-1]
        n_valid = len(valid_idx)
        dist_out = np.ones((n_batch, n_mesh_pair)) * 100
        normal_out = np.zeros((n_batch, n_mesh_pair, 3))
        wp1_out = np.zeros((n_batch, n_mesh_pair, 3))
        wp2_out = np.zeros((n_batch, n_mesh_pair, 3))
        min_idx_out = np.zeros((n_batch, n_mesh_pair), dtype=np.uintp)

        # Fill faraway pairs directly from bounding spheres
        far_mask_np = far_mask.cpu().numpy()
        dist_out[far_mask_np] = dist_sph.detach().cpu().numpy()[far_mask_np]
        normal_out[far_mask_np] = normal_sph.detach().cpu().numpy()[far_mask_np]
        wp1_out[far_mask_np] = wp1_sph.detach().cpu().numpy()[far_mask_np]
        wp2_out[far_mask_np] = wp2_sph.detach().cpu().numpy()[far_mask_np]
        min_idx_out[far_mask_np] = (
            best_cp_idx.detach().cpu().numpy()[far_mask_np].astype(np.uintp)
        )

        # Narrow-phase GJK
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
        return wp1, wp2, normal, d_sign
