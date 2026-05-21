from __future__ import annotations

from dataclasses import dataclass
import torch
import numpy as np

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
    nconmax: int = 20


@dataclass
class DCContext:
    meshes: list[DCMesh]
    mesh_ids: torch.Tensor
    mesh_pair_ids: torch.Tensor
    mesh_pair_indices: torch.Tensor
    pair_margin: torch.Tensor
    pair_gap: torch.Tensor
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
        mesh_pairs: list[tuple[int, int]] | torch.Tensor,
        mesh_ids: list[int] | torch.Tensor,
        pair_margin: float | list | torch.Tensor = 10.0,
        pair_gap: float | list | torch.Tensor = 0.0,
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
            pair_margin=ts.to(0.0),
            pair_gap=ts.to(0.0),
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
        dc_ctx.configure_mesh_pairs(
            mesh_pairs,
            pair_margin,
            tp1_o,
            tp2_o,
            pair_gap=pair_gap,
        )
        return dc_ctx

    def configure_mesh_pairs(
        self,
        mesh_pairs,
        pair_margin: float | list | torch.Tensor = 10.0,
        tp1_o: torch.Tensor | None = None,
        tp2_o: torch.Tensor | None = None,
        pair_gap: float | list | torch.Tensor = 0.0,
    ):
        n_mesh = len(self.meshes)
        if mesh_pairs is None:
            self.mesh_pair_indices = self.ts.to_idx(
                torch.triu_indices(n_mesh, n_mesh, offset=1).T
            )
            self.mesh_pair_ids = self.mesh_ids[self.mesh_pair_indices]
        else:
            self.mesh_pair_ids = self.ts.to_idx(mesh_pairs)
            assert (
                len(self.mesh_pair_ids.shape) == 2 and self.mesh_pair_ids.shape[-1] == 2
            ), "mesh_pairs should be a list of tuple of two mesh ids or a tensor of shape (n_pair, 2)"
            mesh_id_to_idx = {
                int(mesh_id): idx for idx, mesh_id in enumerate(self.mesh_ids.tolist())
            }
            self.mesh_pair_indices = self.ts.to_idx(
                [
                    [mesh_id_to_idx[int(idx1)], mesh_id_to_idx[int(idx2)]]
                    for idx1, idx2 in self.mesh_pair_ids.tolist()
                ]
            )

        pair_margin = self.ts.to(pair_margin).reshape(-1)
        if pair_margin.shape[0] == 1:
            pair_margin = pair_margin.expand(self.mesh_pair_ids.shape[0]).clone()
        assert pair_margin.shape == (
            self.mesh_pair_ids.shape[0],
        ), "pair_margin should be a scalar or a tensor with one value for each mesh pair"
        self.pair_margin = pair_margin

        pair_gap = self.ts.to(pair_gap).reshape(-1)
        if pair_gap.shape[0] == 1:
            pair_gap = pair_gap.expand(self.mesh_pair_ids.shape[0]).clone()
        assert pair_gap.shape == (
            self.mesh_pair_ids.shape[0],
        ), "pair_gap should be a scalar or a tensor with one value for each mesh pair"
        self.pair_gap = pair_gap

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
            mp2cp_idx2.append(self.ts.to_idx([i] * n_cvx2).repeat_interleave(n_cvx1))
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
        pair_margin = dc_ctx.pair_margin.view(1, n_mesh_pair)
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
        broadphase_mask = s2s_min_sct < pair_margin  # (b, p)
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
        near_mask = broadphase_mask & (dist < pair_margin)
        cvx_min_idx[~near_mask] = 0
        contact_counts = near_mask.sum(dim=-1)
        d_sign = 2 * (dist > 0) - 1

        ctx.cfg = cfg
        ctx.dc_ctx = dc_ctx
        ctx.vis = vis

        coll = torch.where(near_mask.reshape(-1))[0]
        wp1_flat, wp2_flat = wp1.reshape(-1, 3)[coll], wp2.reshape(-1, 3)[coll]
        normal_flat = normal.reshape(-1, 3)[coll]
        d_sign_flat = d_sign.reshape(-1)[coll]
        cvx_min_idx_flat = cvx_min_idx.reshape(-1)[coll]

        ctx.save_for_backward(
            T1, T2, normal_flat, wp1_flat, wp2_flat, d_sign_flat, cvx_min_idx_flat, coll
        )
        return wp1_flat, wp2_flat, normal_flat, d_sign_flat, coll, contact_counts

    @staticmethod
    def unpack_saved_tensors(ctx, grad_wp1=None, grad_wp2=None, grad_n=None):
        T1_raw, T2_raw, normal, wp1, wp2, d_sign, cvx_min_idx, coll = ctx.saved_tensors
        b, p = T1_raw.shape[:2]
        n_contact = wp1.shape[0]
        if grad_wp1 is None:
            grad_wp1 = torch.zeros_like(wp1)
        if grad_wp2 is None:
            grad_wp2 = torch.zeros_like(wp2)
        if grad_n is None:
            grad_n = torch.zeros_like(normal)
        T1 = T1_raw.reshape(b * p, 4, 4)[coll]
        T2 = T2_raw.reshape(b * p, 4, 4)[coll]
        return (
            T1_raw,
            T2_raw,
            T1,
            T2,
            normal,
            wp1,
            wp2,
            d_sign,
            cvx_min_idx,
            coll,
            grad_wp1,
            grad_wp2,
            grad_n,
            b,
            p,
            n_contact,
        )

    @staticmethod
    def zero_grads(T1_raw, T2_raw):
        return torch.zeros_like(T1_raw), torch.zeros_like(T2_raw), None, None, None

    @staticmethod
    def scatter_grads(T1_raw, T2_raw, coll, grad1, grad2):
        b, p = T1_raw.shape[:2]
        grad1_flat = torch.zeros(b * p, 4, 4, device=T1_raw.device, dtype=T1_raw.dtype)
        grad2_flat = torch.zeros_like(grad1_flat)
        grad1_flat.index_add_(0, coll, grad1)
        grad2_flat.index_add_(0, coll, grad2)
        return grad1_flat.view_as(T1_raw), grad2_flat.view_as(T2_raw), None, None, None
