from __future__ import annotations

from dataclasses import dataclass
import torch
import numpy as np
import logging

from diffcollision.cpp._coal_openmp import batched_coal_distance
from diffcollision.wp_utils import _WarpSphereDist
from diffcollision.utils import DCTensorSpec
from diffcollision.io import DCMesh


@dataclass
class BaseCollisionConfig:
    type: str = ""
    enable_debug: bool = False
    n_thread: int = 16  # cpu thread number for coal library
    egt: bool = True  # whether to enable equivalent gradient transport
    egt_step_r: float = 1.0  # the relative step between r and t matters
    egt_step_t: float = 0.001  # the relative step between r and t matters
    per_env_max_contact_num: int = 20


@dataclass
class DCContext:
    meshes: list[DCMesh]
    mesh_ids: torch.Tensor
    mesh_pair_ids: torch.Tensor
    mesh_pair_indices: torch.Tensor
    margin: torch.Tensor
    cvx_lst: list
    sph_lst: torch.Tensor
    ts: DCTensorSpec
    warp_sphere_dist: _WarpSphereDist
    cvx_n_sum: torch.Tensor
    ml2mp_idx1: torch.Tensor
    ml2mp_idx2: torch.Tensor
    cl2cp_idx1: torch.Tensor
    cl2cp_idx2: torch.Tensor
    mp2cp_idx1: torch.Tensor
    mp2cp_idx2: torch.Tensor
    cp2mp_idx: torch.Tensor
    tp1_o: torch.Tensor | None = None
    tp2_o: torch.Tensor | None = None
    tp_o: torch.Tensor | None = None

    # Method-specific runtime caches.
    gs_o_mesh: torch.Tensor = None
    dthre_mesh: torch.Tensor = None
    min_dthre_mesh: torch.Tensor = None
    gs_o_pair: torch.Tensor = None
    dthre_pair: torch.Tensor = None
    min_dthre_pair: torch.Tensor = None

    @classmethod
    def from_meshes(
        cls,
        meshes: list[DCMesh],
        collision_pairs: list[tuple[int, int]] | torch.Tensor,
        mesh_ids: list[int] | torch.Tensor,
        margin: float | list | torch.Tensor = 10.0,
        tp1_o: torch.Tensor | None = None,
        tp2_o: torch.Tensor | None = None,
    ):
        ts = DCTensorSpec(
            device=meshes[0].bounding_spheres.device,
            dtype=meshes[0].bounding_spheres.dtype,
        )
        warp_sphere_dist = _WarpSphereDist(ts)

        n_mesh = len(meshes)
        if mesh_ids is None:
            mesh_ids = range(n_mesh)
        mesh_ids = ts.to_idx(mesh_ids)
        assert (
            len(mesh_ids.shape) == 1 and mesh_ids.shape[0] == n_mesh
        ), "mesh_ids should be a list or tensor with one id for each mesh"
        assert len(set(mesh_ids.tolist())) == n_mesh, "mesh_ids should be unique"

        sph_lst = []
        cvx_lst = []
        cvx_n_sum = []
        n_cvx_sum = 0
        for m in meshes:
            sph_lst.append(m.bounding_spheres)
            cvx_lst.extend(m.convex_pieces)
            cvx_n_sum.append(n_cvx_sum)
            n_cvx_sum += m.n_cvx
        sph_lst = torch.cat(sph_lst, dim=0)

        dc_ctx = cls(
            meshes=meshes,
            mesh_ids=mesh_ids,
            mesh_pair_ids=ts.to_idx([]),
            mesh_pair_indices=ts.to_idx([]),
            margin=ts.to(0.0),
            cvx_lst=cvx_lst,
            sph_lst=sph_lst,
            ts=ts,
            warp_sphere_dist=warp_sphere_dist,
            cvx_n_sum=ts.to_idx(cvx_n_sum),
            ml2mp_idx1=ts.to_idx([]),
            ml2mp_idx2=ts.to_idx([]),
            cl2cp_idx1=ts.to_idx([]),
            cl2cp_idx2=ts.to_idx([]),
            mp2cp_idx1=ts.to_idx([]),
            mp2cp_idx2=ts.to_idx([]),
            cp2mp_idx=ts.to_idx([]),
        )
        dc_ctx.configure_collision_pairs(collision_pairs, margin, tp1_o, tp2_o)
        return dc_ctx

    def configure_collision_pairs(
        self,
        collision_pairs,
        margin: float | list | torch.Tensor = 10.0,
        tp1_o: torch.Tensor | None = None,
        tp2_o: torch.Tensor | None = None,
    ):
        n_mesh = len(self.meshes)
        if collision_pairs is None:
            self.mesh_pair_indices = self.ts.to_idx(
                torch.triu_indices(n_mesh, n_mesh, offset=1).T
            )
            self.mesh_pair_ids = self.mesh_ids[self.mesh_pair_indices]
        else:
            self.mesh_pair_ids = self.ts.to_idx(collision_pairs)
            assert (
                len(self.mesh_pair_ids.shape) == 2
                and self.mesh_pair_ids.shape[-1] == 2
            ), "collision_pairs should be a list of tuple of two mesh ids or a tensor of shape (n_pair, 2)"
            mesh_id_to_idx = {
                int(mesh_id): idx for idx, mesh_id in enumerate(self.mesh_ids.tolist())
            }
            self.mesh_pair_indices = self.ts.to_idx(
                [
                    [mesh_id_to_idx[int(idx1)], mesh_id_to_idx[int(idx2)]]
                    for idx1, idx2 in self.mesh_pair_ids.tolist()
                ]
            )

        pair_margin = self.ts.to(margin).reshape(-1)
        if pair_margin.shape[0] == 1:
            pair_margin = pair_margin.expand(self.mesh_pair_ids.shape[0]).clone()
        assert pair_margin.shape == (
            self.mesh_pair_ids.shape[0],
        ), "margin should be a scalar or a tensor with one value for each collision pair"
        self.margin = pair_margin

        if (tp1_o is None) != (tp2_o is None):
            raise ValueError("tp1_o and tp2_o should both be provided or both be None")
        if tp1_o is None:
            self.tp1_o = None
            self.tp2_o = None
            self.tp_o = None
        else:
            tp1_o = self.ts.to(tp1_o)
            tp2_o = self.ts.to(tp2_o)
            expected_tail = (self.mesh_pair_ids.shape[0], 3)
            if tp1_o.ndim != 3 or tp2_o.ndim != 3:
                raise ValueError("tp1_o and tp2_o should have shape (batch, n_pair, 3)")
            if tp1_o.shape != tp2_o.shape or tuple(tp1_o.shape[1:]) != expected_tail:
                raise ValueError("tp1_o and tp2_o should have shape (batch, n_pair, 3)")
            self.tp1_o = tp1_o
            self.tp2_o = tp2_o
            self.tp_o = torch.stack([tp1_o, tp2_o], dim=-2).reshape(-1, 3)

        cp2mp_idx = []
        cl2cp_idx1 = []
        cl2cp_idx2 = []
        mp2cp_idx1 = []
        mp2cp_idx2 = []
        for i, (idx1, idx2) in enumerate(self.mesh_pair_indices.tolist()):
            n_cvx1, n_cvx2 = self.meshes[idx1].n_cvx, self.meshes[idx2].n_cvx
            cp2mp_idx.extend([i] * n_cvx1 * n_cvx2)
            cl2cp_idx1.append(
                (self.cvx_n_sum[idx1] + self.ts.to_idx(range(n_cvx1))).repeat(n_cvx2)
            )
            cl2cp_idx2.append(
                (
                    self.cvx_n_sum[idx2] + self.ts.to_idx(range(n_cvx2))
                ).repeat_interleave(n_cvx1)
            )
            mp2cp_idx1.append(self.ts.to_idx([i] * n_cvx1).repeat(n_cvx2))
            mp2cp_idx2.append(
                self.ts.to_idx([i] * n_cvx2).repeat_interleave(n_cvx1)
            )
        self.ml2mp_idx1 = self.ts.to_idx(self.mesh_pair_indices[..., 0])
        self.ml2mp_idx2 = self.ts.to_idx(self.mesh_pair_indices[..., 1])
        self.cl2cp_idx1 = self.ts.to_idx(torch.cat(cl2cp_idx1))
        self.cl2cp_idx2 = self.ts.to_idx(torch.cat(cl2cp_idx2))
        self.mp2cp_idx1 = self.ts.to_idx(torch.cat(mp2cp_idx1))
        self.mp2cp_idx2 = self.ts.to_idx(torch.cat(mp2cp_idx2))
        self.cp2mp_idx = self.ts.to_idx(cp2mp_idx)

        self.gs_o_pair = None
        self.dthre_pair = None
        self.min_dthre_pair = None
        return


class _BaseCollision(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        T1: torch.Tensor,
        T2: torch.Tensor,
        cfg: BaseCollisionConfig,
        dc_ctx: DCContext,
        vis,
    ):
        cvx_lst, sph_lst, ts = dc_ctx.cvx_lst, dc_ctx.sph_lst, dc_ctx.ts
        n_batch, n_mesh_pair = T2.shape[:2]  # b, p
        margin = dc_ctx.margin.view(1, n_mesh_pair)
        batched_pair_idx = dc_ctx.cp2mp_idx.expand(n_batch, -1)  # (b, k)

        # Broad-phase filter

        # min/max distance of bounding spheres
        sph1_o, sph2_o = (
            sph_lst[dc_ctx.cl2cp_idx1],
            sph_lst[dc_ctx.cl2cp_idx2],
        )  # (k, 4)
        s2s_max, s2s_min = dc_ctx.warp_sphere_dist.forward(
            T1, T2, sph1_o, sph2_o, dc_ctx.mp2cp_idx1, dc_ctx.mp2cp_idx2
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
                dc_ctx.cl2cp_idx1.cpu().numpy().reshape(-1),
                T1.cpu().numpy().reshape(-1),
                dc_ctx.cl2cp_idx2.cpu().numpy().reshape(-1),
                T2.cpu().numpy().reshape(-1),
                dc_ctx.cp2mp_idx.cpu().numpy().reshape(-1),
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

        ctx.cfg = cfg
        ctx.dc_ctx = dc_ctx
        ctx.vis = vis
        ctx.cvx_min_idx = cvx_min_idx
        ctx.near_mask = ts.to_idx(near_mask)
        ctx.save_for_backward(T1, T2, dist, normal, wp1, wp2)
        return wp1, wp2, normal, d_sign, near_mask

    @staticmethod
    def pre_backward_logic(ctx, grad_wp1, grad_wp2, grad_n):
        near_mask = ctx.near_mask.unsqueeze(-1)
        grad_wp1 = grad_wp1 * near_mask
        grad_wp2 = grad_wp2 * near_mask
        grad_n = grad_n * near_mask
        return grad_wp1, grad_wp2, grad_n
