import numpy as np
import torch
import random
import logging
import sys

logging.getLogger("trimesh").setLevel(logging.ERROR)
root_logger = logging.getLogger()
if not root_logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(name)s][%(levelname)s] - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

from diffcollision import DCMesh, DiffCollision, DCTensorSpec
from diffcollision.utils import global_sample_v_or_f


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    return


def torch_normalize_vector(v: torch.Tensor):
    return v / torch.clamp(v.norm(dim=-1, p=2, keepdim=True), min=1e-12)


def torch_quaternion_to_matrix(quaternions):
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """

    quaternions = torch.as_tensor(quaternions)
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    if torch.is_grad_enabled():
        ret[positive_mask] = torch.sqrt(x[positive_mask])
    else:
        ret = torch.where(positive_mask, torch.sqrt(x), ret)
    return ret


def standardize_quaternion(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert a unit quaternion to a standard form: one in which the real
    part is non negative.

    Args:
        quaternions: Quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Standardized quaternions as tensor of shape (..., 4).
    """
    return torch.where(quaternions[..., 0:1] < 0, -quaternions, quaternions)


def torch_matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)
    out = quat_candidates[
        torch.nn.functional.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :
    ].reshape(batch_dim + (4,))
    return standardize_quaternion(out)


def torch_normal_to_rot(
    axis_0, rot_base1=torch.tensor([0, 0, 1.0]), rot_base2=torch.tensor([0, 1.0, 0])
):
    tmp_rot_base1 = rot_base1.view([1] * (len(axis_0.shape) - 1) + [3]).to(
        axis_0.device
    )
    tmp_rot_base2 = rot_base2.view([1] * (len(axis_0.shape) - 1) + [3]).to(
        axis_0.device
    )

    proj_xy = (axis_0 * tmp_rot_base1).sum(dim=-1, keepdim=True).abs()
    axis_1 = torch.where(
        proj_xy > 0.99, tmp_rot_base2, tmp_rot_base1
    )  # avoid normal prependicular to axis_y1
    axis_1 = torch_normalize_vector(
        axis_1 - (axis_1 * axis_0).sum(dim=-1, keepdim=True) * axis_0
    )
    axis_2 = torch.cross(axis_0, axis_1, dim=-1)
    return torch.stack([axis_0, axis_1, axis_2], dim=-1)


def torch_quaternion_rotate_points(q, points):
    """
    Rotate 3D points using quaternions with batched support.

    Args:
        q: torch.Tensor of shape (..., 4) - batched quaternions [w, x, y, z]
        points: torch.Tensor of shape (..., 3) - batched 3D points to rotate

    Returns:
        torch.Tensor of shape (..., 3) - rotated points
    """
    # Normalize quaternions
    q = q / torch.norm(q, dim=-1, keepdim=True)

    # Split quaternion into scalar and vector parts
    q_w = q[..., 0:1]  # shape (..., 1)
    q_xyz = q[..., 1:]  # shape (..., 3)

    # Compute cross product: cross(q_xyz, points)
    cross_p = torch.cross(q_xyz, points, dim=-1)

    # Compute the rotation: v' = v + 2 * q_w * cross(q_xyz, v) + 2 * cross(q_xyz, cross(q_xyz, v))
    rotated_points = (
        points + 2 * q_w * cross_p + 2 * torch.cross(q_xyz, cross_p, dim=-1)
    )

    return rotated_points


def sample_target_point(
    mesh1: DCMesh,
    mesh2: DCMesh,
    n_tp: int,
    tp_type: tuple[str, str],
    check: bool,
    ts: DCTensorSpec = DCTensorSpec(),
):
    diffcoll = DiffCollision([mesh1, mesh2])
    cm1, fm1 = mesh1.coarse_mesh, mesh1.fine_mesh
    cm2, fm2 = mesh2.coarse_mesh, mesh2.fine_mesh
    final_tp1, final_tp2 = ts.to(torch.zeros((0, 3))), ts.to(torch.zeros((0, 3)))
    while len(final_tp1) < n_tp:
        # Get samples
        tp1, n1 = global_sample_v_or_f(cm1, fm1, n_tp, tp_type[0])
        tp2, n2 = global_sample_v_or_f(cm2, fm2, n_tp, tp_type[1])
        tp1, tp2 = ts.to(tp1), ts.to(tp2)
        n1, n2 = ts.to(n1), ts.to(n2)

        if not check:
            return tp1, tp2

        # Calculate GT transformation and validate samples
        r1 = torch_normal_to_rot(n1)
        r2 = torch_normal_to_rot(-n2)
        r_rel = r1 @ r2.transpose(-1, -2)
        t_rel = tp1 - (r_rel @ tp2[..., None]).squeeze(-1)
        T2 = ts.to(torch.zeros((r1.shape[0], 4, 4)))
        T2[:, :3, :3] = r_rel
        T2[:, :3, 3] = t_rel
        T2[:, 3, 3] = 1
        T1 = ts.to(torch.eye(4).expand_as(T2))

        res = diffcoll.forward(torch.stack([T1, T2], dim=-3), return_local=False)
        tp2_w = torch.einsum("bij, bj->bi", r_rel, tp2) + t_rel
        error = (res.wp1.squeeze(1) - tp1).norm(dim=-1) + (
            res.wp2.squeeze(1) - tp2_w
        ).norm(dim=-1)
        vind = torch.where(error < 1e-2)[0]

        final_tp1 = torch.cat([final_tp1, tp1[vind]], dim=0)[:n_tp]
        final_tp2 = torch.cat([final_tp2, tp2[vind]], dim=0)[:n_tp]

    return final_tp1[:, None], final_tp2[:, None]
