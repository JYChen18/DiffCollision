import torch
import trimesh
import logging
import sys
import os

from diffcollision import DCMesh, DiffCollision, DCTensorSpec
from diffcollision.utils import torch_matrix_grad_to_se3, torch_se3_exp_map

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from examples.util.rotation import set_seed


def test_forward(mesh_lst, ts):
    T = torch.eye(4)[None, None].repeat(1, 2, 1, 1)
    T[:, 1, 0, 3] = 0.5

    diffcoll = DiffCollision(mesh_lst)
    res = diffcoll.forward(T, return_local=False)
    assert res.sdf == 0.3
    logging.info("Pass forward test")


def test_backward_easy(mesh_lst, ts):
    T1 = torch.eye(4)[None]
    T2 = torch.eye(4)[None]
    T2[:, 0, 3] = 0.5
    T1, T2 = ts.to(T1), ts.to(T2)
    T2.requires_grad_()

    step_r = 1.0
    step_t = 0.1

    diffcoll = DiffCollision(mesh_lst, egt_step_r=step_r, egt_step_t=step_t)

    for i in range(51):
        with torch.no_grad():
            T2.grad = None
        T = torch.stack([T1, T2], dim=-3)
        res = diffcoll.forward(T, return_local=False)
        loss = ((res.wp1 - res.wp2) ** 2).sum()
        loss.backward()
        with torch.no_grad():
            proj2 = torch_matrix_grad_to_se3(T2, T2.grad)
            T2[:] = T2 @ torch_se3_exp_map(-proj2, step_r, step_t)
        # if i % 10 == 0:
        #     logging.info(f"Iter:{i}, Loss: {loss}")
    assert loss < 1e-10
    logging.info("Pass backward-easy test")


def test_backward_hard(mesh_lst, ts):
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
        mesh_lst, egt_step_r=step_r, egt_step_t=step_t, enable_debug=True
    )

    for i in range(101):
        with torch.no_grad():
            T2.grad = None
        T = torch.stack([T1, T2], dim=-3)
        res = diffcoll.forward(T)
        loss = (
            ((res.wp1 - res.wp2) ** 2).sum()
            + ((tp1_o - res.wp1_o) ** 2).sum()
            + ((tp2_o - res.wp2_o) ** 2).sum()
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
        raise ImportError(
            "Core library verified. For USD visualization: `pip install -e '.[examples]'`"
        )


if __name__ == "__main__":
    ts = DCTensorSpec()
    sphere = trimesh.primitives.Sphere(radius=0.1, subdivisions=5)
    mesh1 = DCMesh.from_data(sphere.vertices, sphere.faces, ts)
    mesh2 = DCMesh.from_data(sphere.vertices, sphere.faces, ts)
    mesh_lst = [mesh1, mesh2]

    set_seed(1)
    test_forward(mesh_lst, ts)
    test_backward_easy(mesh_lst, ts)
    test_backward_hard(mesh_lst, ts)
