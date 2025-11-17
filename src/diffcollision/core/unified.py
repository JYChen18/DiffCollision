from dataclasses import dataclass, field, fields
import torch

from diffcollision.utils import eqv_grad, torch_normalize_vector, DCTensorSpec
from diffcollision.core.rs1dist import RS1DistCollision, RS1DistConfig
from diffcollision.core.rs1dir import RS1DirCollision, RS1DirConfig
from diffcollision.core.rs0 import RS0Collision, RS0Config
from diffcollision.core.fd import FDCollision, FDConfig
from diffcollision.core.analytical import AnalyticalCollision, AnalyticalConfig
from diffcollision.io import DCMesh
from diffcollision.core.base import _BaseConfig as DCBaseConfig


DIFFCOLL_CONFIG_REGISTRY = {
    "RS1Dist": RS1DistConfig,
    "RS1Dir": RS1DirConfig,
    "RS0": RS0Config,
    "FD": FDConfig,
    "Analytical": AnalyticalConfig,
}

DIFFCOLL_FUNC_REGISTRY = {
    "RS1Dist": RS1DistCollision,
    "RS1Dir": RS1DirCollision,
    "RS0": RS0Collision,
    "FD": FDCollision,
    "Analytical": AnalyticalCollision,
}


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
        Witness points in the **world frame** for each collision pair.
    normal : torch.Tensor, shape (b, p, 3)
        Contact normal in the **world frame**, pointing outward from object 1 (even when penetrating).
    sdf : torch.Tensor, shape (b, p)
        Signed distance between witness points. Positive if separated, negative if penetrating.

    wp1_o, wp2_o : torch.Tensor, optional
        Witness points in the **object local frame** of each collision pair.
    n1_o, n2_o : torch.Tensor, optional
        Contact normals in the **object local frame**.
        `n1_o` points outward from object 1, and `n2_o` points outward from object 2.
    """

    wp1: torch.Tensor
    wp2: torch.Tensor
    normal: torch.Tensor
    sdf: torch.Tensor

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
    >>> diffcoll = DiffCollision(meshes, collision_pairs=[[0, 1]])
    >>> result = diffcoll.forward(transforms)
    >>> wp1, wp2 = result.wp1, result.wp2   # witness points on each collision pair (in world frame)
    >>> n, sdf = result.normal, result.sdf  # contact normal & signed distance (in world frame)
    >>> w1_o, w2_o = result.w1_o, result.w2_o   # in object local frame
    >>> n1_o, n2_o = result.n1_o, result.n2_o   # in object local frame
    """

    def __init__(
        self,
        meshes: list[DCMesh],
        collision_pairs: list[tuple[int, int]] | torch.Tensor = None,
        method: str = "RS1Dist",
        enable_debug: bool = False,
        **kwargs,
    ):
        """
        Initialize the differentiable collision module.

        Parameters
        ----------
        meshes : list of DCMesh
            All meshes in the scene.
        collision_pairs : list of tuple(int, int) or torch.Tensor, optional
            Pairs of mesh indices to check for collisions. If None, all unique
            unordered pairs will be generated automatically. Default: None.
        method : str, optional
            Algorithm used for differentiable collision computation.
            Default: `"RS1Dist"`.
        enable_debug : bool, optional
            If True, intermediate results are stored for visualization and
            can be accessed via :meth:`get_debug_dict()`. Default: False.
        **kwargs :
            Additional configuration parameters for the selected method.
            To view available parameters, see the corresponding config class,
            e.g. `help(RS1DistConfig)`.
        """
        config_cls = DIFFCOLL_CONFIG_REGISTRY[method]
        func_cls = DIFFCOLL_FUNC_REGISTRY[method]

        # Filter only valid arguments for the selected configuration
        valid_fields = {
            f.name for f in fields(config_cls) if not f.name.startswith("_")
        }
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_fields}

        self.cfg = config_cls(
            _meshes=meshes, _collision_pairs=collision_pairs, **filtered_kwargs
        )
        self.func_cls = func_cls
        self.debug_dict = DCDebugDict(meshes=meshes) if enable_debug else None

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
        T1 = transforms[:, self.cfg._ml2mp_idx1]
        T2 = transforms[:, self.cfg._ml2mp_idx2]

        if not self.cfg.egt:
            T1_egt, T2_egt = T1, T2
        else:
            T1_egt, T2_egt = _GradTransportLayer.apply(
                T1, T2, self.cfg.egt_step_r, self.cfg.egt_step_t
            )

        wp1, wp2, d_sign = self.func_cls.apply(
            T1_egt, T2_egt, self.cfg, self.debug_dict if not skip_debug else None
        )

        if self.debug_dict is not None and not skip_debug:
            with torch.no_grad():
                self.debug_dict.transforms.append(transforms.detach().cpu())
                self.debug_dict.wp1.append(wp1.detach().cpu())
                self.debug_dict.wp2.append(wp2.detach().cpu())
                if self.cfg.tp1_o is not None and self.cfg.tp2_o is not None:
                    tp1 = (
                        torch.einsum("bkij,bkj->bki", T1[..., :3, :3], self.cfg.tp1_o)
                        + T1[..., :3, 3]
                    )
                    tp2 = (
                        torch.einsum("bkij,bkj->bki", T2[..., :3, :3], self.cfg.tp2_o)
                        + T2[..., :3, 3]
                    )
                    self.debug_dict.tp1.append(tp1.detach().cpu())
                    self.debug_dict.tp2.append(tp2.detach().cpu())

        normal = d_sign.unsqueeze(-1) * torch_normalize_vector(wp2 - wp1)
        signed_dist = d_sign * (wp2 - wp1).norm(dim=-1)
        if return_local:  # NOTE: use T_egt to ensure correct gradient flow
            wp1_o = torch.einsum(
                "bpji,bpj->bpi", T1_egt[..., :3, :3], wp1 - T1_egt[..., :3, 3]
            )
            wp2_o = torch.einsum(
                "bpji,bpj->bpi", T2_egt[..., :3, :3], wp2 - T2_egt[..., :3, 3]
            )
            n1_o = torch.einsum("bpji,bpj->bpi", T1_egt[..., :3, :3], normal)
            n2_o = torch.einsum("bpji,bpj->bpi", T2_egt[..., :3, :3], -normal)
        else:
            wp1_o = wp2_o = n1_o = n2_o = None
        return DCResult(wp1, wp2, normal, signed_dist, wp1_o, wp2_o, n1_o, n2_o)

    def update_collision_pairs(
        self,
        collision_pairs: list[tuple[int, int]] | torch.Tensor,
        tp1_o: torch.Tensor = None,
        tp2_o: torch.Tensor = None,
    ):
        """
        Dynamically update the set of mesh pairs to be checked for collisions.

        Parameters
        ----------
        collision_pairs : list of tuple(int, int) or torch.Tensor
            New set of mesh index pairs to evaluate.
        tp1_o, tp2_o : torch.Tensor, optional
            Target points in the object local frame of each collision pair.
            Required when `method` is `"RS1Dist"` or `"RS1Dir"` and `sample="adp"`. Default: None.
        """
        self.cfg.update_collision_pairs(collision_pairs, tp1_o, tp2_o)
        return

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

    def get_cfg(self) -> DCBaseConfig:
        """
        Retrieve the configuration object of the selected method.

        Returns
        -------
        DCBaseConfig
            Configuration instance (e.g., `RS1DistConfig`, `FDConfig`, etc.).
        """
        return self.cfg

    def assert_valid_transforms(self, transforms):
        ts: DCTensorSpec = self.cfg._ts
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
            and transforms.shape[1] == len(self.cfg._meshes)
        ), f"transforms have the wrong shape. Got {transforms.shape}, expected (b, {len(self.cfg._meshes)}, 4, 4)."

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
