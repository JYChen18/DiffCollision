import os
import sys
import logging
from pathlib import Path

import pytest
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(ROOT_DIR))

from diffcollision import DiffCollision, DCMesh, RS1DistConfig
from diffcollision.utils import torch_matrix_grad_to_se3, torch_se3_exp_map
from examples.util.rotation import set_seed, sample_target_point


@pytest.fixture(scope="module")
def mesh_lst():
    set_seed(1)
    asset_dir = ROOT_DIR / "examples/assets/object/DGN_5k/processed_data"
    assert asset_dir.exists(), (
        f"{asset_dir} does not exist! Please download `DGN_5k` dataset following "
        "the instruction in README.md"
    )
    obj_lst = sorted(os.listdir(asset_dir))
    return [
        DCMesh.from_file(asset_dir / obj_lst[i], scale=0.1, convex_hull=False)
        for i in range(5)
    ]


@pytest.fixture(scope="module")
def collision_pairs():
    return torch.tensor([[0, 1], [3, 2], [1, 4]])


@pytest.fixture(scope="module")
def shuffle_lst():
    return [2, 0, 1]


def sample_target_points_for_pairs(mesh_lst, collision_pairs):
    set_seed(1)
    tp1_o, tp2_o = [], []
    for idx1, idx2 in collision_pairs:
        tp1_single, tp2_single = sample_target_point(
            mesh_lst[idx1], mesh_lst[idx2], 7, ["v", "v"], True
        )
        tp1_o.append(tp1_single)
        tp2_o.append(tp2_single)
    return torch.cat(tp1_o, dim=1), torch.cat(tp2_o, dim=1)


@pytest.fixture(scope="module")
def target_points(mesh_lst, collision_pairs):
    return sample_target_points_for_pairs(mesh_lst, collision_pairs)


def make_transforms():
    T = torch.eye(4).view(1, 1, 4, 4).repeat(7, 5, 1, 1)
    for i in range(5):
        T[:, i, 0, 3] = 0.1 * i
    return T


def assert_forward_matches_pair_order(mesh_lst, collision_pairs, pair_order):
    diffcoll = DiffCollision(mesh_lst, collision_pairs=collision_pairs)
    ordered_diffcoll = DiffCollision(
        mesh_lst, collision_pairs=collision_pairs[pair_order]
    )
    T = make_transforms()
    res = diffcoll.forward(T, return_local=False)
    ordered_res = ordered_diffcoll.forward(T, return_local=False)
    assert (res.sdf[:, pair_order] - ordered_res.sdf[:, :3]).abs().max() < 1e-10


def test_collision_pairs_forward(mesh_lst, collision_pairs):
    assert_forward_matches_pair_order(mesh_lst, collision_pairs, [0, 1, 2])
    logging.info("Pass forward test")


def test_shuffled_collision_pairs_forward(mesh_lst, collision_pairs, shuffle_lst):
    assert_forward_matches_pair_order(mesh_lst, collision_pairs, shuffle_lst)
    logging.info("Pass shuffled forward test")


def assert_backward_converges(mesh_lst, collision_pairs, tp1_o, tp2_o):
    step_r = 10.0
    step_t = 0.1
    diffcoll = DiffCollision(
        mesh_lst,
        collision_pairs=collision_pairs,
        config=RS1DistConfig(egt_step_r=step_r, egt_step_t=step_t),
        tp1_o=tp1_o,
        tp2_o=tp2_o,
    )

    T = make_transforms()
    T.requires_grad_()
    for i in range(501):
        with torch.no_grad():
            T.grad = None
        res = diffcoll.forward(T)
        loss = (
            ((res.wp1[:, :3] - res.wp2[:, :3]) ** 2).sum()
            + ((tp1_o[:, :3] - res.wp1_o[:, :3]) ** 2).sum()
            + ((tp2_o[:, :3] - res.wp2_o[:, :3]) ** 2).sum()
        )
        loss.backward()
        with torch.no_grad():
            proj2 = torch_matrix_grad_to_se3(T, T.grad)
            T[:] = T @ torch_se3_exp_map(-proj2, step_r, step_t)
        if i % 100 == 0:
            logging.info(f"Iter:{i}, Loss: {loss}")
    assert loss < 5e-5


def test_collision_pairs_backward(mesh_lst, collision_pairs, target_points):
    tp1_o, tp2_o = target_points
    assert_backward_converges(mesh_lst, collision_pairs, tp1_o, tp2_o)
    logging.info("Pass backward test")


def test_shuffled_collision_pairs_backward(
    mesh_lst, collision_pairs, shuffle_lst, target_points
):
    tp1_o, tp2_o = target_points
    assert_backward_converges(
        mesh_lst,
        collision_pairs[shuffle_lst],
        tp1_o[:, shuffle_lst],
        tp2_o[:, shuffle_lst],
    )
    logging.info("Pass shuffled backward test")


if __name__ == "__main__":
    set_seed(1)
    asset_dir = ROOT_DIR / "examples/assets/object/DGN_5k/processed_data"
    assert os.path.exists(
        asset_dir
    ), f"{asset_dir} does not exist! Please download `DGN_5k` dataset following the instruction in README.md"
    obj_lst = sorted(os.listdir(asset_dir))
    mesh_lst = []
    for i in range(5):
        mesh = DCMesh.from_file(
            os.path.join(asset_dir, obj_lst[i]), scale=0.1, convex_hull=False
        )
        mesh_lst.append(mesh)

    collision_pairs = torch.tensor([[0, 1], [3, 2], [1, 4]])
    shuffle_lst = [2, 0, 1]

    assert_forward_matches_pair_order(mesh_lst, collision_pairs, [0, 1, 2])
    assert_forward_matches_pair_order(mesh_lst, collision_pairs, shuffle_lst)
    tp1_o, tp2_o = sample_target_points_for_pairs(mesh_lst, collision_pairs)
    assert_backward_converges(mesh_lst, collision_pairs, tp1_o, tp2_o)
    assert_backward_converges(
        mesh_lst,
        collision_pairs[shuffle_lst],
        tp1_o[:, shuffle_lst],
        tp2_o[:, shuffle_lst],
    )
