from dataclasses import dataclass, field
from typing import Annotated, Union
import torch
import mujoco
import tyro
import dacite

from diffcollision.utils import eqv_grad, torch_normalize_vector, DCTensorSpec
from diffcollision.core.rs1dist import RS1DistCollision, RS1DistConfig
from diffcollision.core.rs1dir import RS1DirCollision, RS1DirConfig
from diffcollision.core.rs0 import RS0Collision, RS0Config
from diffcollision.core.fd import FDCollision, FDConfig
from diffcollision.core.analytical import AnalyticalCollision, AnalyticalConfig
from diffcollision.io import DCMesh
from diffcollision.mjmesh import (
    get_mesh_from_mjmodel,
    get_mesh_pair_margins_from_mjmodel,
)
from diffcollision.core.base import DCContext

DiffCollisionConfig = Union[
    Annotated[RS1DistConfig, tyro.conf.subcommand(name="RS1Dist")],
    Annotated[RS1DirConfig, tyro.conf.subcommand(name="RS1Dir")],
    Annotated[RS0Config, tyro.conf.subcommand(name="RS0")],
    Annotated[FDConfig, tyro.conf.subcommand(name="FD")],
    Annotated[AnalyticalConfig, tyro.conf.subcommand(name="Analytical")],
]


@dataclass
class _DCConfigWrapper:
    config: DiffCollisionConfig


def build_diffcoll_config(config_dict: dict) -> DiffCollisionConfig:
    return dacite.from_dict(
        _DCConfigWrapper,
        {"config": config_dict},
        config=dacite.Config(strict=True, strict_unions_match=True),
    ).config


@dataclass
class DCDebugDict:
    """
    Container for visualization and debugging of intermediate results.

    Attributes
    ----------
    meshes : list of DCMesh
        The list of meshes in the scene.
    transforms : list of torch.Tensor
        Transformation matrices `(b, n, 4, 4)` for all meshes, recorded per call.
    tp1, tp2 : list of torch.Tensor
        Target points on each mesh pair (if available), recorded per call.
    wp1, wp2 : list of torch.Tensor
        Witness points on each mesh pair, recorded per call.
    ls1, ls2 : list of torch.Tensor
        Local samples around witness points, recorded per call.
    """

    meshes: list[DCMesh]
    transforms: list = field(default_factory=list)
    tp1: list = field(default_factory=list)
    tp2: list = field(default_factory=list)
    wp1: list = field(default_factory=list)
    wp2: list = field(default_factory=list)
    ls1: list = field(default_factory=list)
    ls2: list = field(default_factory=list)


@dataclass
class DCResult:
    """
    Container for differentiable collision detection results.

    Attributes
    ----------
    wp1, wp2 : torch.Tensor, shape (b, p, 3)
        Witness points in the **world frame** for each mesh pair.
    normal : torch.Tensor, shape (b, p, 3)
        Contact normal in the **world frame**, pointing outward from object 1 (even when penetrating).
    sdf : torch.Tensor, shape (b, p)
        Signed distance between witness points. Positive if separated, negative if penetrating.

    wp1_o, wp2_o : torch.Tensor, optional
        Witness points in the **object local frame** of each mesh pair.
    n1_o, n2_o : torch.Tensor, optional
        Contact normals in the **object local frame**.
        `n1_o` points outward from object 1, and `n2_o` points outward from object 2.
    """

    wp1: torch.Tensor
    wp2: torch.Tensor
    normal: torch.Tensor
    sdf: torch.Tensor
    cpidx: torch.Tensor

    wp1_o: torch.Tensor = None
    wp2_o: torch.Tensor = None
    n1_o: torch.Tensor = None
    n2_o: torch.Tensor = None


class _GradTransportLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, T1: torch.Tensor, T2: torch.Tensor, step_r: float, step_t: float):
        ctx.save_for_backward(T1, T2)
        ctx.step_r = step_r
        ctx.step_t = step_t
        return T1, T2

    @staticmethod
    def backward(ctx, grad1, grad2):
        T1, T2 = ctx.saved_tensors
        eg2 = eqv_grad(T1, T2, grad1, ctx.step_r, ctx.step_t)
        eg1 = eqv_grad(T2, T1, grad2, ctx.step_r, ctx.step_t)
        return grad1 + eg1, grad2 + eg2, None, None


class DiffCollision:
    """
    Differentiable collision detection module.

    This class provides a unified PyTorch-compatible interface for computing
    differentiable witness points across multiple mesh pairs and batches. It
    leverages the OpenMP-accelerated Coal backend for forward collision queries
    and supports multiple algorithmic backends for gradient computation.

    Supported Methods
    -----------------
    - "RS1Dist" : Distance-based first-order random smoothing (**recommended**).
    - "RS1Dir" : Direction-based first-order random smoothing.
    - "RS0" : Zero-order random smoothing.
    - "FD" : Finite difference.
    - "Analytical" : Analytical gradient of brute-force vertice-face check.

    Example
    -------
    >>> diffcoll = DiffCollision(
    ...     meshes, mesh_pairs=[[0, 1]], config=RS1DistConfig()
    ... )
    >>> result = diffcoll.forward(transforms)
    >>> wp1, wp2 = result.wp1, result.wp2   # witness points on each mesh pair (in world frame)
    >>> n, sdf = result.normal, result.sdf  # contact normal & signed distance (in world frame)
    >>> w1_o, w2_o = result.w1_o, result.w2_o   # in object local frame
    >>> n1_o, n2_o = result.n1_o, result.n2_o   # in object local frame
    """

    def __init__(
        self,
        meshes: list[DCMesh],
        mesh_pairs: list[tuple[int, int]] | torch.Tensor = None,
        mesh_ids: list[int] | torch.Tensor = None,
        pair_margin: float | list | torch.Tensor = 10.0,
        tp1_o: torch.Tensor | None = None,
        tp2_o: torch.Tensor | None = None,
        config: DiffCollisionConfig = RS1DirConfig(),
    ):
        """
        Initialize the differentiable collision module.

        Parameters
        ----------
        meshes : list[DCMesh], optional
            Meshes to evaluate. May be omitted when `mj_model` is supplied.
        mesh_pairs : list of tuple(int, int) or torch.Tensor, optional
            Mesh id pairs to evaluate. Defaults to all unique pairs, or to
            MuJoCo-compatible pairs when `mj_model` is supplied.
        mesh_ids : list[int] or torch.Tensor, optional
            Public ids for `meshes`. Defaults to compact indices, or to MuJoCo
            body ids when `mj_model` is supplied.
        config : DiffCollisionConfig, optional
            Collision method config. Defaults to `RS1DirConfig()`.
        pair_margin : float, list, or torch.Tensor, optional
            Contact detection distance threshold. May be a scalar or one value per mesh pair.
        tp1_o, tp2_o : torch.Tensor, optional
            Target points in object-local frame, with shape `(batch, n_pair, 3)`.
            Required when using adaptive sampling.
        """
        if not hasattr(config, "method"):
            raise TypeError("config should be a DiffCollisionConfig instance")
        self.cfg = config

        self.ctx = DCContext.from_meshes(
            meshes,
            mesh_pairs,
            mesh_ids,
            pair_margin,
            tp1_o,
            tp2_o,
        )

        self.func_cls = config.method
        self.debug_dict = DCDebugDict(meshes=meshes) if self.cfg.enable_debug else None

    @classmethod
    def from_mjmodel(
        cls,
        mj_model: mujoco.MjModel,
        config: DiffCollisionConfig = RS1DirConfig(),
        device: str = "cpu",
        dtype: str = "float",
    ) -> "DiffCollision":

        if mj_model is not None:
            ts = DCTensorSpec(device, dtype)
            mesh_ids, meshes = get_mesh_from_mjmodel(mj_model, ts)
            mesh_pairs, pair_margins = get_mesh_pair_margins_from_mjmodel(mj_model)
            pair_margins = ts.to(pair_margins)

        return cls(
            meshes=meshes,
            mesh_pairs=mesh_pairs,
            mesh_ids=mesh_ids,
            pair_margin=pair_margins,
            config=config,
        )

    def forward(
        self,
        transforms: torch.Tensor,
        return_local: bool = True,
        skip_debug: bool = False,
    ):
        """
        Perform differentiable collision detection across all specified mesh pairs.

        Parameters
        ----------
        transforms : torch.Tensor, shape (b, n, 4, 4)
            Transformation matrices between **object local frame** and **world frame**.
            `b` is the batch size and `n` is the number of meshes.
        return_local : bool, optional
            If True, also returns results in each mesh's **object local frame**.
            Default: True.
        skip_debug : bool, optional
            If True, skips saving intermediate results even when debugging
            is globally enabled. Default: False.

        Returns
        -------
        DCResult
            A structured container of world-frame and (optionally) object local-frame results.
        """
        self.assert_valid_transforms(transforms)
        if (
            self.ctx.tp1_o is not None
            and self.ctx.tp1_o.shape[0] != transforms.shape[0]
        ):
            raise ValueError(
                "tp1_o and tp2_o batch dimension should match the transforms batch size"
            )
        T1 = transforms[:, self.ctx.ml2mp_idx1]
        T2 = transforms[:, self.ctx.ml2mp_idx2]

        if not self.cfg.egt:
            T1_egt, T2_egt = T1, T2
        else:
            T1_egt, T2_egt = _GradTransportLayer.apply(
                T1, T2, self.cfg.egt_step_r, self.cfg.egt_step_t
            )

        wp1_all, wp2_all, normal_all, d_sign, near_mask = self.func_cls.apply(
            T1_egt,
            T2_egt,
            self.cfg,
            self.ctx,
            self.debug_dict if not skip_debug else None,
        )

        if self.debug_dict is not None and not skip_debug:
            with torch.no_grad():
                self.debug_dict.transforms.append(transforms.detach().cpu())
                self.debug_dict.wp1.append(wp1_all.detach().cpu())
                self.debug_dict.wp2.append(wp2_all.detach().cpu())
                if self.ctx.tp1_o is not None and self.ctx.tp2_o is not None:
                    tp1 = (
                        torch.einsum("bkij,bkj->bki", T1[..., :3, :3], self.ctx.tp1_o)
                        + T1[..., :3, 3]
                    )
                    tp2 = (
                        torch.einsum("bkij,bkj->bki", T2[..., :3, :3], self.ctx.tp2_o)
                        + T2[..., :3, 3]
                    )
                    self.debug_dict.tp1.append(tp1.detach().cpu())
                    self.debug_dict.tp2.append(tp2.detach().cpu())

        # NOTE: The following normal's gradient will have numerical issues when wp1 is close to wp2.
        # We have only implemented a smooth normal derivative for `method=RS1Dist`.
        if self.func_cls != RS1DistCollision:
            normal_all = d_sign.unsqueeze(-1) * torch_normalize_vector(
                wp2_all - wp1_all
            )

        sdf_all = d_sign * (wp2_all - wp1_all).norm(dim=-1)
        if return_local:  # NOTE: use T_egt to ensure correct gradient flow
            all_wp1_o = torch.einsum(
                "bpji,bpj->bpi", T1_egt[..., :3, :3], wp1_all - T1_egt[..., :3, 3]
            )
            all_wp2_o = torch.einsum(
                "bpji,bpj->bpi", T2_egt[..., :3, :3], wp2_all - T2_egt[..., :3, 3]
            )
            all_n1_o = torch.einsum("bpji,bpj->bpi", T1_egt[..., :3, :3], normal_all)
            all_n2_o = torch.einsum("bpji,bpj->bpi", T2_egt[..., :3, :3], -normal_all)

        # gather valid contacts according to near_mask
        ts = DCTensorSpec(T1.device, T1.dtype)
        n_batch = T1.shape[0]
        slot_idx = ts.to_idx(near_mask.cumsum(dim=1)) - 1
        take_mask = near_mask & (slot_idx < self.cfg.per_env_max_contact_num)
        wp1 = ts.to(torch.zeros(n_batch, self.cfg.per_env_max_contact_num, 3))
        wp2 = ts.to(torch.zeros(n_batch, self.cfg.per_env_max_contact_num, 3))
        normal = ts.to(torch.zeros(n_batch, self.cfg.per_env_max_contact_num, 3))
        sdf = ts.to(
            torch.full((n_batch, self.cfg.per_env_max_contact_num), float("inf"))
        )
        cpidx = ts.to_idx(
            torch.full((n_batch, self.cfg.per_env_max_contact_num, 2), -1)
        )

        batch_ids, pair_ids = torch.where(take_mask)
        slot_ids = slot_idx[batch_ids, pair_ids]
        wp1[batch_ids, slot_ids] = wp1_all[batch_ids, pair_ids]
        wp2[batch_ids, slot_ids] = wp2_all[batch_ids, pair_ids]
        normal[batch_ids, slot_ids] = normal_all[batch_ids, pair_ids]
        sdf[batch_ids, slot_ids] = sdf_all[batch_ids, pair_ids]
        cpidx[batch_ids, slot_ids] = self.ctx.mesh_pair_ids[pair_ids]

        if return_local:
            wp1_o = ts.to(torch.zeros(n_batch, self.cfg.per_env_max_contact_num, 3))
            wp2_o = ts.to(torch.zeros(n_batch, self.cfg.per_env_max_contact_num, 3))
            n1_o = ts.to(torch.zeros(n_batch, self.cfg.per_env_max_contact_num, 3))
            n2_o = ts.to(torch.zeros(n_batch, self.cfg.per_env_max_contact_num, 3))
            wp1_o[batch_ids, slot_ids] = all_wp1_o[batch_ids, pair_ids]
            wp2_o[batch_ids, slot_ids] = all_wp2_o[batch_ids, pair_ids]
            n1_o[batch_ids, slot_ids] = all_n1_o[batch_ids, pair_ids]
            n2_o[batch_ids, slot_ids] = all_n2_o[batch_ids, pair_ids]
        else:
            wp1_o = wp2_o = n1_o = n2_o = None

        return DCResult(wp1, wp2, normal, sdf, cpidx, wp1_o, wp2_o, n1_o, n2_o)

    def get_debug_dict(self) -> DCDebugDict:
        """
        Retrieve the stored debug dictionary.

        Returns
        -------
        DCDebugDict
            Object containing all recorded intermediate results.
        """
        if self.debug_dict is None:
            raise RuntimeError(
                "Debugging is not enabled. Please set enable_debug=True when initializing DiffCollision."
            )
        return self.debug_dict

    def get_cfg(self) -> DiffCollisionConfig:
        """
        Retrieve the configuration object of the selected method.

        Returns
        -------
        DiffCollisionConfig
            Configuration instance (e.g., `RS1DistConfig`, `FDConfig`, etc.).
        """
        return self.cfg

    def get_context(self) -> DCContext:
        """
        Retrieve the internal runtime context.

        Returns
        -------
        DCContext
            Meshes, tensor specs, pair mappings, broad-phase indices, and caches.
        """
        return self.ctx

    def assert_valid_transforms(self, transforms):
        ts: DCTensorSpec = self.ctx.ts
        # type check
        assert (
            isinstance(transforms, torch.Tensor)
            and transforms.dtype == ts.dtype
            and transforms.device == ts.device
        ), (
            "Invalid transforms tensor: expected a torch.Tensor with "
            f"dtype={ts.dtype} and device={ts.device}, but got "
            f"type={type(transforms)}, dtype={getattr(transforms, 'dtype', None)}, "
            f"device={getattr(transforms, 'device', None)}."
        )

        # shape check
        assert (
            transforms.ndim == 4
            and transforms.shape[-2:] == (4, 4)
            and transforms.shape[1] == len(self.ctx.meshes)
        ), f"transforms have the wrong shape. Got {transforms.shape}, expected (b, {len(self.ctx.meshes)}, 4, 4)."

        # bottom row check
        bottom = transforms[..., 3, :]
        assert torch.allclose(
            bottom, ts.to([0.0, 0.0, 0.0, 1.0])
        ), f"Last row must be [0,0,0,1]"

        # rotation orthonormality check
        R = transforms[..., :3, :3]
        should_be_identity = R @ R.transpose(-1, -2)
        assert torch.allclose(
            should_be_identity, ts.to(torch.eye(3)), atol=1e-5
        ), "Rotation part is not orthonormal"

        # determinant +1 check
        det = torch.det(R)
        assert torch.allclose(
            det, ts.to(torch.ones_like(det)), atol=1e-5
        ), f"Rotation matrices must have det=1"

        return True
