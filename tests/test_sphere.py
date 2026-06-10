import logging
import sys
import os
from pathlib import Path

import pytest
import torch
import trimesh

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(ROOT_DIR))

from diffcollision import (
    DCMesh,
    DiffCollision,
    DCTensorSpec,
    AnalyticalConfig,
    FDConfig,
    RS0Config,
    RS1DirConfig,
    RS1DistConfig,
)
from diffcollision.utils import torch_matrix_grad_to_se3, torch_se3_exp_map
from examples.util.rotation import set_seed


@pytest.fixture(scope="module")
def ts():
    return DCTensorSpec(dtype="double")


@pytest.fixture(scope="module")
def mesh_lst(ts):
    sphere = trimesh.primitives.Sphere(radius=0.1, subdivisions=5)
    mesh1 = DCMesh.from_data(sphere.vertices, sphere.faces, ts)
    mesh2 = DCMesh.from_data(sphere.vertices, sphere.faces, ts)
    return [mesh1, mesh2]


def test_forward(mesh_lst, ts):
    T = ts.to(torch.eye(4)[None, None].repeat(1, 2, 1, 1))
    T[:, 1, 0, 3] = 0.5

    diffcoll = DiffCollision(mesh_lst)
    res = diffcoll.forward(T, return_local=False)
    assert torch.isclose(res.dist[0, 0], ts.to(0.3))
    logging.info("Pass forward test")


def test_init_scalar_pair_margin_expands_into_context(mesh_lst, ts):
    diffcoll = DiffCollision(
        mesh_lst,
        config=RS1DistConfig(nconmax=3),
        pair_margin=0.5,
    )

    assert diffcoll.cfg.nconmax == 3
    assert diffcoll.ctx.pair_margin.shape == (1,)
    assert torch.allclose(diffcoll.ctx.pair_margin, ts.to([0.5]))


def test_init_pair_margin_stays_pair_aligned(mesh_lst, ts):
    diffcoll = DiffCollision(
        [mesh_lst[0], mesh_lst[1], mesh_lst[0]],
        mesh_pairs=[[0, 1], [1, 2]],
        pair_margin=ts.to([0.2, 0.3]),
    )

    assert diffcoll.ctx.pair_margin.shape == (2,)
    assert torch.allclose(diffcoll.ctx.pair_margin, ts.to([0.2, 0.3]))


def test_pair_gap_extends_detection_buffer(mesh_lst, ts):
    T = ts.to(torch.eye(4)[None, None].repeat(1, 2, 1, 1))
    T[:, 1, 0, 3] = 0.25

    diffcoll = DiffCollision(
        mesh_lst,
        pair_margin=0.0,
        pair_gap=0.1,
    )

    res = diffcoll.forward(T, return_local=False)

    assert torch.allclose(diffcoll.ctx.pair_margin, ts.to([0.0]))
    assert torch.allclose(diffcoll.ctx.pair_gap, ts.to([0.1]))
    assert torch.isclose(res.dist[0, 0], ts.to(0.05), atol=1e-6)
    assert torch.isinf(res.dist[0, 1:]).all()


def test_target_points_validate_pair_shape(mesh_lst, ts):
    tp1_o = ts.to([[[0, 0.1, 0], [0.1, 0, 0]]])
    tp2_o = ts.to([[[0.1, 0, 0], [0, 0.1, 0]]])

    with pytest.raises(ValueError, match="shape"):
        DiffCollision(mesh_lst, tp1_o=tp1_o, tp2_o=tp2_o)


def test_adaptive_sampling_requires_target_points(mesh_lst, ts):
    T = ts.to(torch.eye(4)[None, None].repeat(1, 2, 1, 1))
    T[:, 1, 0, 3] = 0.5
    T.requires_grad_()
    diffcoll = DiffCollision(mesh_lst, config=RS1DistConfig(sample="adp"))

    res = diffcoll.forward(T, return_local=False)
    with pytest.raises(ValueError, match="target points"):
        res.pos.sum().backward()


def test_nested_collision_config_dispatches_all_methods(mesh_lst):
    configs = [
        RS1DistConfig(),
        RS1DirConfig(),
        RS0Config(n_jitter=4),
        FDConfig(),
        AnalyticalConfig(),
    ]

    for collision_cfg in configs:
        diffcoll = DiffCollision(mesh_lst, config=collision_cfg)
        assert diffcoll.get_cfg().type == collision_cfg.type


def test_backward_easy(mesh_lst, ts):
    set_seed(1)
    T1 = torch.eye(4)[None]
    T2 = torch.eye(4)[None]
    T2[:, 0, 3] = 0.5
    T1, T2 = ts.to(T1), ts.to(T2)
    T2.requires_grad_()

    step_r = 1.0
    step_t = 0.1

    diffcoll = DiffCollision(
        mesh_lst,
        config=RS1DistConfig(egt_step_r=step_r, egt_step_t=step_t),
    )

    for i in range(51):
        with torch.no_grad():
            T2.grad = None
        T = torch.stack([T1, T2], dim=-3)
        res = diffcoll.forward(T, return_local=False)
        loss = ((res.pos[..., 0, :] - res.pos[..., 1, :]) ** 2).sum()
        loss.backward()
        with torch.no_grad():
            proj2 = torch_matrix_grad_to_se3(T2, T2.grad)
            T2[:] = T2 @ torch_se3_exp_map(-proj2, step_r, step_t)
        # if i % 10 == 0:
        #     logging.info(f"Iter:{i}, Loss: {loss}")
    assert loss < 1e-10
    logging.info("Pass backward-easy test")


def test_backward_hard(mesh_lst, ts):
    set_seed(1)
    T1 = torch.eye(4)[None]
    T2 = torch.eye(4)[None]
    T2[:, 0, 3] = 0.5
    T1, T2 = ts.to(T1), ts.to(T2)
    T2.requires_grad_()

    tp1_o = ts.to([[[0, 0.1, 0]]])
    tp2_o = ts.to([[[0.1, 0, 0]]])
    step_r = 10.0
    step_t = 0.1
    diffcoll = DiffCollision(
        mesh_lst,
        config=RS1DistConfig(enable_debug=True, egt_step_r=step_r, egt_step_t=step_t),
        tp1_o=tp1_o,
        tp2_o=tp2_o,
    )

    for i in range(101):
        with torch.no_grad():
            T2.grad = None
        T = torch.stack([T1, T2], dim=-3)
        res = diffcoll.forward(T)
        loss = (
            ((res.pos[:, 0, 0] - res.pos[:, 0, 1]) ** 2).sum()
            + ((tp1_o[:, 0] - res.pos_o[:, 0, 0]) ** 2).sum()
            + ((tp2_o[:, 0] - res.pos_o[:, 0, 1]) ** 2).sum()
        )
        loss.backward()
        with torch.no_grad():
            proj2 = torch_matrix_grad_to_se3(T2, T2.grad)
            T2[:] = T2 @ torch_se3_exp_map(-proj2, step_r, step_t)
        # if i % 10 == 0:
        #     logging.info(f"Iter:{i}, Loss: {loss}")
    assert loss < 1e-10
    logging.info("Pass backward-hard test")
    try:
        from examples.util.vis import vis_usd

        vis_usd(diffcoll.get_debug_dict(), [0], "output/sphere")
    except ImportError:
        logging.info(
            "Core library verified. For USD visualization: `pip install -e '.[examples]'`"
        )


if __name__ == "__main__":
    ts = DCTensorSpec(dtype="double")
    sphere = trimesh.primitives.Sphere(radius=0.1, subdivisions=5)
    mesh1 = DCMesh.from_data(sphere.vertices, sphere.faces, ts)
    mesh2 = DCMesh.from_data(sphere.vertices, sphere.faces, ts)
    mesh_lst = [mesh1, mesh2]

    set_seed(1)
    test_forward(mesh_lst, ts)
    test_backward_easy(mesh_lst, ts)
    test_backward_hard(mesh_lst, ts)
