import sys
from pathlib import Path

import pytest
import torch
import trimesh
from loguru import logger

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(ROOT_DIR))

from diffcollision import (
    AnalyticalConfig,
    DCMesh,
    DCTensorSpec,
    DiffCollision,
    FDConfig,
    RS0Config,
    RS1DirConfig,
    RS1DistConfig,
)
from diffcollision.core import rs1dist


@pytest.fixture(scope="module")
def ts():
    return DCTensorSpec(dtype="double")


@pytest.fixture(scope="module")
def sphere_meshes(ts):
    sphere = trimesh.primitives.Sphere(radius=0.1, subdivisions=2)
    mesh = DCMesh.from_data(sphere.vertices, sphere.faces, ts)
    return [mesh, mesh, mesh, mesh]


def make_transforms(ts, x_positions):
    n_mesh = len(x_positions[0])
    transforms = ts.to(torch.eye(4)[None, None].repeat(len(x_positions), n_mesh, 1, 1))
    transforms[:, :, 0, 3] = ts.to(x_positions)
    return transforms


def test_rs1dist_regroups_compact_contacts_by_batch(sphere_meshes, ts):
    diffcoll = DiffCollision(
        sphere_meshes,
        mesh_pairs=[[0, 1], [0, 2], [0, 3]],
        pair_margin=0.12,
        config=RS1DistConfig(nconmax=3, n_thread=1),
    )
    transforms = make_transforms(
        ts,
        [
            [0.0, 0.25, 1.0, 0.28],
            [0.0, 1.0, 0.26, 1.0],
        ],
    )

    result = diffcoll.forward(transforms)

    assert torch.isfinite(result.dist[0, :2]).all()
    assert torch.isinf(result.dist[0, 2])
    assert torch.isfinite(result.dist[1, 0])
    assert torch.isinf(result.dist[1, 1:]).all()
    assert result.bodyid[0, :2].tolist() == [[0, 1], [0, 3]]
    assert result.bodyid[1, 0].tolist() == [0, 2]
    assert result.pos_o.shape == (2, 3, 2, 3)


@pytest.mark.parametrize(
    "config",
    [
        RS1DistConfig(nconmax=3, n_thread=1, n_global=64, n_local=8),
        RS1DirConfig(nconmax=3, n_thread=1, n_global=64, n_local=8),
        RS0Config(nconmax=3, n_thread=1, n_jitter=4),
        FDConfig(nconmax=3, n_thread=1),
        AnalyticalConfig(nconmax=3, n_thread=1),
    ],
)
def test_all_methods_regroup_compact_contacts_by_batch(sphere_meshes, ts, config):
    diffcoll = DiffCollision(
        sphere_meshes,
        mesh_pairs=[[0, 1], [0, 2], [0, 3]],
        pair_margin=0.12,
        config=config,
    )
    transforms = make_transforms(
        ts,
        [
            [0.0, 0.25, 1.0, 0.28],
            [0.0, 1.0, 0.26, 1.0],
        ],
    )

    result = diffcoll.forward(transforms)

    assert torch.isfinite(result.dist[0, :2]).all()
    assert torch.isinf(result.dist[0, 2])
    assert torch.isfinite(result.dist[1, 0])
    assert torch.isinf(result.dist[1, 1:]).all()
    assert result.bodyid[0, :2].tolist() == [[0, 1], [0, 3]]
    assert result.bodyid[1, 0].tolist() == [0, 2]


def test_rs1dist_backward_samples_only_compact_contacts(sphere_meshes, ts, monkeypatch):
    calls = []
    original_local_sample = rs1dist._local_sample

    def tracked_local_sample(*args, **kwargs):
        coll = kwargs.get("coll", args[9] if len(args) > 9 else None)
        calls.append(
            {
                "n_contact": int(args[2].shape[0]),
                "coll": coll.detach().cpu().tolist(),
            }
        )
        return original_local_sample(*args, **kwargs)

    monkeypatch.setattr(rs1dist, "_local_sample", tracked_local_sample)
    diffcoll = DiffCollision(
        sphere_meshes,
        mesh_pairs=[[0, 1], [0, 2], [0, 3]],
        pair_margin=0.12,
        config=RS1DistConfig(nconmax=3, n_thread=1, n_global=64, n_local=8),
    )
    transforms = make_transforms(
        ts,
        [
            [0.0, 0.25, 1.0, 0.28],
            [0.0, 1.0, 0.26, 1.0],
        ],
    )
    transforms.requires_grad_()

    result = diffcoll.forward(transforms)
    finite_dist = result.dist[torch.isfinite(result.dist)]
    loss = finite_dist.sum() + result.pos[torch.isfinite(result.dist)].sum()
    loss.backward()

    assert calls == [{"n_contact": 3, "coll": [0, 2, 4]}]
    assert transforms.grad is not None
    assert torch.isfinite(transforms.grad).all()


@pytest.mark.parametrize(
    "config",
    [
        RS1DistConfig(nconmax=3, n_thread=1, n_global=64, n_local=8),
        RS1DirConfig(nconmax=3, n_thread=1, n_global=64, n_local=8),
        RS0Config(nconmax=3, n_thread=1, n_jitter=4),
        FDConfig(nconmax=3, n_thread=1),
        AnalyticalConfig(nconmax=3, n_thread=1),
    ],
)
def test_all_methods_backward_with_filtered_pairs(sphere_meshes, ts, config):
    diffcoll = DiffCollision(
        sphere_meshes,
        mesh_pairs=[[0, 1], [0, 2], [0, 3]],
        pair_margin=0.12,
        config=config,
    )
    transforms = make_transforms(
        ts,
        [
            [0.0, 0.25, 1.0, 0.28],
            [0.0, 1.0, 0.26, 1.0],
        ],
    )
    transforms.requires_grad_()

    result = diffcoll.forward(transforms)
    finite_dist = result.dist[torch.isfinite(result.dist)]
    loss = finite_dist.sum() + result.pos[torch.isfinite(result.dist)].sum()
    loss.backward()

    assert transforms.grad is not None
    assert torch.isfinite(transforms.grad).all()
    assert transforms.grad.abs().sum() > 0


def test_rs1dist_logs_error_and_truncates_when_nconmax_exceeded(sphere_meshes, ts):
    messages = []
    sink_id = logger.add(lambda message: messages.append(str(message)), level="ERROR")
    try:
        diffcoll = DiffCollision(
            sphere_meshes,
            mesh_pairs=[[0, 1], [0, 2], [0, 3]],
            pair_margin=0.12,
            config=RS1DistConfig(nconmax=1, n_thread=1),
        )
        transforms = make_transforms(ts, [[0.0, 0.25, 0.26, 0.27]])
        result = diffcoll.forward(transforms)
    finally:
        logger.remove(sink_id)

    assert any("exceeds nconmax 1" in message for message in messages)
    assert torch.isfinite(result.dist[0, 0])
    assert result.dist.shape == (1, 1)
    assert result.bodyid[0, 0].tolist() == [0, 1]


@pytest.mark.parametrize(
    "config",
    [
        RS1DistConfig(nconmax=2, n_thread=1),
        RS1DirConfig(nconmax=2, n_thread=1),
        RS0Config(nconmax=2, n_thread=1, n_jitter=4),
        FDConfig(nconmax=2, n_thread=1),
        AnalyticalConfig(nconmax=2, n_thread=1),
    ],
)
def test_all_methods_zero_contact_backward_returns_zero_gradients(
    sphere_meshes, ts, config
):
    diffcoll = DiffCollision(
        sphere_meshes[:2],
        pair_margin=0.01,
        config=config,
    )
    transforms = make_transforms(ts, [[0.0, 1.0]])
    transforms.requires_grad_()
    T1 = transforms[:, diffcoll.ctx.ml2mp_idx1]
    T2 = transforms[:, diffcoll.ctx.ml2mp_idx2]

    wp1, wp2, normal, _d_sign, coll, contact_counts = diffcoll.func_cls.apply(
        T1, T2, diffcoll.cfg, diffcoll.ctx, None
    )
    assert coll.numel() == 0
    assert contact_counts.tolist() == [0]

    (wp1.sum() + wp2.sum() + normal.sum()).backward()

    assert transforms.grad is not None
    assert torch.equal(transforms.grad, torch.zeros_like(transforms.grad))
