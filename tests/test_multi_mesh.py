import os
import sys
import logging
import torch
from diffcollision import DiffCollision, DCMesh

from diffcollision.utils import torch_matrix_grad_to_se3, torch_se3_exp_map

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from examples.util.rotation import set_seed, sample_target_point


def test_update_collision_pairs_forward(mesh_lst, collision_pairs, shuffle_lst):
    diffcoll = DiffCollision(mesh_lst, collision_pairs)

    T = torch.eye(4).view(1, 1, 4, 4).repeat(7, 5, 1, 1)
    for i in range(5):
        T[:, i, 0, 3] = 0.1 * i
    res1 = diffcoll.forward(T, return_local=False)

    new_coll_pair = collision_pairs[shuffle_lst]
    diffcoll.update_collision_pairs(new_coll_pair)

    res2 = diffcoll.forward(T, return_local=False)
    assert (res1.sdf[:, shuffle_lst] - res2.sdf).abs().max() < 1e-10
    logging.info("Pass forward test")


def test_update_collision_pairs_backward(mesh_lst, collision_pairs, shuffle_lst):
    step_r = 10.0
    step_t = 0.1
    tp1_o, tp2_o = [], []
    for idx1, idx2 in collision_pairs:
        tp1_single, tp2_single = sample_target_point(
            mesh_lst[idx1], mesh_lst[idx2], 7, ["v", "v"], True
        )
        tp1_o.append(tp1_single)
        tp2_o.append(tp2_single)
    tp1_o = torch.cat(tp1_o, dim=1)
    tp2_o = torch.cat(tp2_o, dim=1)
    diffcoll = DiffCollision(
        mesh_lst, collision_pairs, egt_step_r=step_r, egt_step_t=step_t
    )

    T = torch.eye(4).view(1, 1, 4, 4).repeat(7, 5, 1, 1)
    for i in range(5):
        T[:, i, 0, 3] = 0.1 * i
    T.requires_grad_()
    for i in range(501):
        with torch.no_grad():
            T.grad = None
        res = diffcoll.forward(T)
        loss = (
            ((res.wp1 - res.wp2) ** 2).sum()
            + ((tp1_o - res.wp1_o) ** 2).sum()
            + ((tp2_o - res.wp2_o) ** 2).sum()
        )
        loss.backward()
        with torch.no_grad():
            proj2 = torch_matrix_grad_to_se3(T, T.grad)
            T[:] = T @ torch_se3_exp_map(-proj2, step_r, step_t)
        # if i % 100 == 0:
        #     logging.info(f"Iter:{i}, Loss: {loss}")
    assert loss < 1e-5

    new_coll_pair = collision_pairs[shuffle_lst]
    tp1_o = tp1_o[:, shuffle_lst]
    tp2_o = tp2_o[:, shuffle_lst]
    diffcoll.update_collision_pairs(new_coll_pair)

    T = torch.eye(4).view(1, 1, 4, 4).repeat(7, 5, 1, 1)
    for i in range(5):
        T[:, i, 0, 3] = 0.1 * i
    T.requires_grad_()
    for i in range(501):
        with torch.no_grad():
            T.grad = None
        res = diffcoll.forward(T)
        loss = (
            ((res.wp1 - res.wp2) ** 2).sum()
            + ((tp1_o - res.wp1_o) ** 2).sum()
            + ((tp2_o - res.wp2_o) ** 2).sum()
        )
        loss.backward()
        with torch.no_grad():
            proj2 = torch_matrix_grad_to_se3(T, T.grad)
            T[:] = T @ torch_se3_exp_map(-proj2, step_r, step_t)
        # if i % 100 == 0:
        #     logging.info(f"Iter:{i}, Loss: {loss}")
    assert loss < 1e-5
    logging.info("Pass backward test")


if __name__ == "__main__":
    set_seed(1)
    asset_dir = "examples/assets/object/DGN_5k/processed_data"
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

    test_update_collision_pairs_forward(mesh_lst, collision_pairs, shuffle_lst)
    test_update_collision_pairs_backward(mesh_lst, collision_pairs, shuffle_lst)
