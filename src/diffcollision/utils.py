from dataclasses import dataclass
import torch
import numpy as np
import trimesh
import logging


@dataclass
class DCTensorSpec:
    """Utility class to standardize tensor device and dtype conversion."""

    device: torch.device | str = "cpu"
    dtype: torch.dtype | str = "float"

    def __post_init__(self):
        if isinstance(self.device, str):
            self.device = torch.device(self.device)
        if isinstance(self.dtype, str):
            self.dtype = getattr(torch, self.dtype)

    def to(self, x: list | np.ndarray | torch.Tensor) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            return x.to(device=self.device, dtype=self.dtype)
        else:
            return torch.tensor(x, device=self.device, dtype=self.dtype)

    def to_idx(self, x: list | np.ndarray | torch.Tensor) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            return x.to(device=self.device, dtype=getattr(torch, "long"))
        else:
            return torch.tensor(x, device=self.device, dtype=getattr(torch, "long"))


def torch_normalize_vector(v: torch.Tensor):
    return v / torch.clamp(v.norm(dim=-1, p=2, keepdim=True), min=1e-12)


def skew_symmetric(v: torch.Tensor) -> torch.Tensor:
    """
    Convert 3D vectors into skew-symmetric matrices.
    Supports batched input.

    Args:
        v: (..., 3) tensor

    Returns:
        (..., 3, 3) tensor of skew-symmetric matrices
    """
    v1, v2, v3 = v.unbind(-1)
    O = torch.zeros_like(v1)
    return torch.stack(
        [
            torch.stack([O, -v3, v2], dim=-1),
            torch.stack([v3, O, -v1], dim=-1),
            torch.stack([-v2, v1, O], dim=-1),
        ],
        dim=-2,
    )


def torch_se3_to_matrix_grad(A, xi):
    """
    Batched inverse se3 gradient projection.

    Args:
        A: (..., 4, 4) SE(3) matrices
        xi: (..., 6) tangent vectors

    Returns:
        A_grad: (..., 4, 4) Euclidean gradient matrices
    """
    rho, phi = xi[..., :3], xi[..., 3:]
    skew_rho = skew_symmetric(rho)

    G_body = torch.zeros_like(A)
    G_body[..., :3, :3] = skew_rho
    G_body[..., :3, 3] = phi

    return A @ G_body


def torch_matrix_grad_to_se3(A, A_grad):
    """
    Batched se3 gradient projection.

    Args:
        A: (..., 4, 4) SE(3) matrices
        A_grad: (..., 4, 4) Euclidean gradients

    Returns:
        xi: (..., 6) tangent vectors
    """
    G_body = torch.linalg.solve(A.detach(), A_grad)

    phi = G_body[..., :3, 3]
    R_block = G_body[..., :3, :3]
    S = R_block - R_block.transpose(-1, -2)

    rho = 0.5 * torch.stack([S[..., 2, 1], S[..., 0, 2], S[..., 1, 0]], dim=-1)
    return torch.cat([rho, phi], dim=-1)


def torch_se3_exp_map(xi, step_r=1.0, step_t=1.0):
    """
    Batched SE(3) exponential map.
    """
    rho, phi = xi[..., :3] * step_r, xi[..., 3:] * step_t
    theta = torch.norm(rho, dim=-1)

    eye3 = torch.eye(3, device=xi.device, dtype=xi.dtype)
    eye4 = torch.eye(4, device=xi.device, dtype=xi.dtype)

    small = theta < 1e-18
    skew_rho = skew_symmetric(rho)

    # Small angle
    R_small = eye3 + skew_rho + 0.5 * (skew_rho @ skew_rho)
    V_small = eye3 + 0.5 * skew_rho
    t_small = V_small @ phi.unsqueeze(-1)

    # General case
    axis = rho / theta.clamp_min(1e-18).unsqueeze(-1)
    skew_axis = skew_symmetric(axis)

    sin_theta, cos_theta = torch.sin(theta), torch.cos(theta)

    R_general = (
        cos_theta[..., None, None] * eye3
        + (1 - cos_theta)[..., None, None] * (axis[..., :, None] @ axis[..., None, :])
        + sin_theta[..., None, None] * skew_axis
    )

    V = (
        eye3
        + ((1 - cos_theta) / (theta**2))[..., None, None] * skew_rho
        + ((theta - sin_theta) / (theta**3))[..., None, None] * (skew_rho @ skew_rho)
    )
    t_general = V @ phi.unsqueeze(-1)

    # Choose by mask
    R = torch.where(small[..., None, None], R_small, R_general)
    t = torch.where(small[..., None, None], t_small, t_general)[..., 0]

    T = eye4.expand(xi.shape[:-1] + (4, 4)).clone()
    T[..., :3, :3] = R
    T[..., :3, 3] = t
    return T


def torch_so3_log_map(R):
    """
    Batched SO(3) log map.
    """
    trace_R = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    theta = torch.acos(torch.clamp((trace_R - 1) / 2, -1.0, 1.0))

    skew_sym = 0.5 * (R - R.transpose(-1, -2))
    small = theta < 1e-18

    rho_small = torch.stack(
        [skew_sym[..., 2, 1], skew_sym[..., 0, 2], skew_sym[..., 1, 0]], dim=-1
    )

    skew_sym_normed = skew_sym / torch.sin(theta)[..., None, None]
    w = torch.stack(
        [
            skew_sym_normed[..., 2, 1],
            skew_sym_normed[..., 0, 2],
            skew_sym_normed[..., 1, 0],
        ],
        dim=-1,
    )
    rho_general = theta[..., None] * w

    return torch.where(small[..., None], rho_small, rho_general)


def torch_se3_log_map(T, step_r=1.0, step_t=1.0):
    """
    Batched SE(3) log map.
    """
    R, t = T[..., :3, :3], T[..., :3, 3]
    rho = torch_so3_log_map(R)
    theta = torch.norm(rho, dim=-1)

    skew_rho = skew_symmetric(rho)
    small = theta < 1e-18

    eye3 = torch.eye(3, device=T.device, dtype=T.dtype)

    # Small angle case
    V_inv_small = eye3 - 0.5 * skew_rho
    phi_small = (V_inv_small @ t.unsqueeze(-1))[..., 0]

    # General case
    half_theta = theta / 2
    cot_half_theta = torch.cos(half_theta) / torch.sin(half_theta)
    coef = (1 / (theta**2)) * (1 - (theta / 2) * cot_half_theta)

    V_inv = eye3 - 0.5 * skew_rho + coef[..., None, None] * (skew_rho @ skew_rho)
    phi_general = (V_inv @ t.unsqueeze(-1))[..., 0]
    phi = torch.where(small.unsqueeze(-1), phi_small, phi_general)
    xi = torch.cat([rho / step_r, phi / step_t], dim=-1)
    return xi


def adjoint_from_transform(T: torch.Tensor) -> torch.Tensor:
    """
    Compute the 6x6 Adjoint matrix for a batch of SE(3) transforms.

    Args:
        T: (..., 4, 4) SE(3) matrices

    Returns:
        Ad: (..., 6, 6) Adjoint matrices
    """
    assert T.shape[-2:] == (4, 4), "Input must be (...,4,4) transform matrices"

    R = T[..., :3, :3]  # (..., 3, 3)
    p = T[..., :3, 3]  # (..., 3)

    px = skew_symmetric(p)

    # build Adjoint matrix
    Ad = torch.zeros(*T.shape[:-2], 6, 6, device=T.device, dtype=T.dtype)
    Ad[..., :3, :3] = R
    Ad[..., 3:, :3] = px @ R
    Ad[..., 3:, 3:] = R

    return Ad


def eqv_grad(T1, T2, grad_T1, step_r=1, step_t=1):
    se3_T1 = torch_matrix_grad_to_se3(T1, grad_T1)
    se3_T1[..., :3] *= step_r
    se3_T1[..., 3:] *= step_t
    T_rel = torch.linalg.solve(T2, T1)
    se3_T2 = torch.einsum("...ij, ...j-> ...i", adjoint_from_transform(T_rel), se3_T1)
    se3_T2[..., :3] /= step_r
    se3_T2[..., 3:] /= step_t
    grad_T2 = torch_se3_to_matrix_grad(T2, -se3_T2)
    return grad_T2


_local_sample_warn_once = False


def local_sample_w_dthre(
    global_sample: torch.Tensor,
    target_point: torch.Tensor,
    dist_thre: float,
    min_thre: float,
    n_local: int,
    sample_strategy: str,
):
    dist = (global_sample - global_sample[:, -1:]).norm(dim=-1)  # b, s
    if sample_strategy == "adp":
        dist2 = (global_sample[:, -1] - target_point).norm(dim=-1)  # b
        valid = dist < torch.maximum(2 * dist2, min_thre).unsqueeze(-1)  # b, s
    elif sample_strategy == "fix":
        valid = dist < torch.maximum(dist_thre, min_thre).unsqueeze(-1)  # b, s
    else:
        raise ValueError(
            f"Unknown sample strategy {sample_strategy}. Available: 'adp', 'fix'."
        )

    # If buggy batches found, re-sample with adjusted threshold
    buggy_idx = torch.where(valid.sum(dim=-1) < max(int(n_local // 4), 2))[0]
    if len(buggy_idx) > 0:
        logging.warning(
            f"Found {len(buggy_idx)} batches (out of {valid.shape[0]}) with insufficient local samples. "
        )
        global _local_sample_warn_once
        if not _local_sample_warn_once:
            logging.warning(
                "If this appears frequently across meshes, consider increasing `dthre` or `n_global` in the config. "
                "If only for specific meshes with very few batches, those meshes may be problematic (e.g., containing disconnected pieces)."
            )
            _local_sample_warn_once = True
        new_thre = torch.topk(dist[buggy_idx], k=n_local, dim=-1, largest=False)[0][
            :, -1
        ]
        valid[buggy_idx] = dist[buggy_idx] < new_thre.unsqueeze(-1)

    probs = valid / valid.sum(dim=-1, keepdim=True)
    # NOTE: replacement=False may get samples with prob=0
    sampled_idx = torch.multinomial(probs, n_local, replacement=True)
    sampled_padded = sampled_idx.unsqueeze(-1).expand(-1, -1, 3)
    return global_sample.gather(1, sampled_padded)


def global_sample_v_and_f(
    coarse_mesh: trimesh.Trimesh, fine_mesh: trimesh.Trimesh, n_sample: int
):
    vp, vn = global_sample_v_or_f(coarse_mesh, fine_mesh, n_sample // 2, "v")
    sp, sn = global_sample_v_or_f(coarse_mesh, fine_mesh, n_sample // 2, "f")
    return torch.cat([vp, sp], dim=0), torch.cat([vn, sn], dim=0)


def global_sample_v_or_f(
    coarse_mesh: trimesh.Trimesh,
    fine_mesh: trimesh.Trimesh,
    n_sample: int,
    type_sample: str,
):
    if type_sample == "v":
        v_ind = np.random.choice(range(len(fine_mesh.vertices)), n_sample)
        p, n = fine_mesh.vertices[v_ind], fine_mesh.vertex_normals[v_ind]
    elif type_sample == "f":
        p1, _ = trimesh.sample.sample_surface_even(coarse_mesh, n_sample)
        n_remain = n_sample - p1.shape[0]
        if n_remain > 0:
            new_target_point, new_f = trimesh.sample.sample_surface(
                coarse_mesh, n_remain
            )
            p1 = np.concatenate([p1, new_target_point], axis=0)
            # f = np.concatenate([f, new_f], axis=0)
        p, _, f = fine_mesh.nearest.on_surface(p1)
        n = fine_mesh.face_normals[f]
    else:
        raise ValueError(
            f"Unsupported sample type: {type_sample}. Available choices: 'v' or 'f', indicating vertices and faces. "
        )

    return torch.tensor(p), torch.tensor(n)


def point_to_triangle_distance_and_closest(points, triangles, eps=1e-20):
    """
    Batched shortest distance from points to triangles in 3D.
    Also returns the closest points on the triangles.
    Handles degenerate triangles robustly.

    Args:
        points: (N,3) tensor of points
        tris:   (N,3,3) tensor of triangles
        eps:    small value to avoid division by zero

    Returns:
        distances: (N,) tensor of distances
        closest_points: (N,3) tensor of closest points on triangles
    """
    # Extract vertices
    A, B, C = triangles.unbind(dim=1)  # each (N, 3)

    # Compute edges
    AB = B - A
    AC = C - A
    BC = C - B

    # Vector from A to point
    AP = points - A

    # Compute barycentric coordinates
    dot00 = torch.sum(AB * AB, dim=1)
    dot01 = torch.sum(AB * AC, dim=1)
    dot02 = torch.sum(AB * AP, dim=1)
    dot11 = torch.sum(AC * AC, dim=1)
    dot12 = torch.sum(AC * AP, dim=1)

    inv_denom = 1.0 / (dot00 * dot11 - dot01 * dot01 + eps)
    u = (dot11 * dot02 - dot01 * dot12) * inv_denom
    v = (dot00 * dot12 - dot01 * dot02) * inv_denom

    # Check if inside triangle
    inside = (u >= 0) & (v >= 0) & (u + v <= 1)

    # Projection onto plane
    proj = A + u.unsqueeze(1) * AB + v.unsqueeze(1) * AC

    # Closest points on edges
    # Edge AB
    t_ab = torch.clamp(torch.sum(AP * AB, dim=1) / (dot00 + eps), 0, 1)
    p_ab = A + t_ab.unsqueeze(1) * AB

    # Edge AC
    t_ac = torch.clamp(torch.sum(AP * AC, dim=1) / (dot11 + eps), 0, 1)
    p_ac = A + t_ac.unsqueeze(1) * AC

    # Edge BC
    BP = points - B
    dot_bc = torch.sum(BC * BC, dim=1)
    t_bc = torch.clamp(torch.sum(BP * BC, dim=1) / (dot_bc + eps), 0, 1)
    p_bc = B + t_bc.unsqueeze(1) * BC

    # Compute distances
    dist_proj = torch.norm(points - proj, dim=1)
    dist_ab = torch.norm(points - p_ab, dim=1)
    dist_ac = torch.norm(points - p_ac, dim=1)
    dist_bc = torch.norm(points - p_bc, dim=1)

    # Find minimum distance for outside points
    dist_out, idx_out = torch.min(
        torch.stack([dist_ab, dist_ac, dist_bc], dim=1), dim=1
    )
    closest_out = torch.stack([p_ab, p_ac, p_bc], dim=1)[
        torch.arange(points.shape[0]), idx_out
    ]

    # Final results
    distances = torch.where(inside, dist_proj, dist_out)
    closest_points = torch.where(inside.unsqueeze(1), proj, closest_out)

    return distances, closest_points
