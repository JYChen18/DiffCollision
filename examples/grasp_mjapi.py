import numpy as np
import os
import mujoco
import logging
from dataclasses import replace

import torch
import pytorch_kinematics as pk
from diffcollision.utils import DCTensorSpec
from diffcollision import DCMesh, DiffCollision
from example_config import MainConfig

from util.vis import vis_usd
from util.rotation import set_seed, torch_normalize_vector, torch_quaternion_to_matrix


def main(cfg: MainConfig):
    set_seed(cfg.seed)
    ts = DCTensorSpec(cfg.device, cfg.dtype)
    xml_path = "examples/assets/grasp_env.xml"
    model = mujoco.MjModel.from_xml_path(xml_path)
    collision_num = 5

    npy_path = "examples/assets/grasp.npy"
    data = np.load(npy_path, allow_pickle=True).item()

    tp1_o = ts.to(
        [[0.000587, -0.006644, 0.018874]] * 4 + [[-0.004527, -0.006819, 0.023165]]
    ).unsqueeze(0)
    tp2_o = ts.to(
        [
            [0.028, -0.047027, -0.012547],
            [0.0, -0.047027, -0.012547],
            [-0.022, -0.047027, -0.012547],
            [-0.043, -0.047027, -0.012547],
            [0.014448, -0.047, 0.012569],
        ]
    ).unsqueeze(0)

    diffcoll = DiffCollision.from_mjmodel(
        model,
        config=replace(cfg.dcd, enable_debug=cfg.vis),
        tp1_o=tp1_o,
        tp2_o=tp2_o,
    )
    mesh_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(mesh_id))
        for mesh_id in diffcoll.ctx.mesh_ids
    ]
    hand_body_name = mesh_names[:-1]

    T_obj = ts.to(torch.eye(4)[None])
    joint_angle = ts.to(data["grasp_qpos"])
    joint_angle[..., 1] -= 0.1
    joint_angle.requires_grad_()

    # Forward kinematics
    xml_path = "examples/assets/hand/shadow/right.xml"
    mjcf_string = open(xml_path).read()
    rel_mesh_path = mjcf_string.split('meshdir="')[-1].split('"')[0]
    abs_mesh_path = os.path.join(os.path.dirname(xml_path), rel_mesh_path)
    mjcf_string = mjcf_string.replace(
        'meshdir="' + rel_mesh_path, 'meshdir="' + abs_mesh_path
    )
    chain = pk.build_chain_from_mjcf(mjcf_string).to(dtype=ts.dtype, device=ts.device)

    for i in range(cfg.iter + 1):
        with torch.no_grad():
            joint_angle.grad = None

        T2_dict = chain.forward_kinematics(joint_angle[..., 7:])
        global_rot = torch_quaternion_to_matrix(joint_angle[..., 3:7])
        global_T = torch.cat([global_rot, joint_angle[..., :3].unsqueeze(-1)], dim=-1)
        global_T = torch.cat([global_T, ts.to([[[0, 0, 0, 1]]])], dim=-2)
        for k, v in T2_dict.items():
            T2_dict[k] = global_T @ v.get_matrix()

        transforms = []
        for bn in hand_body_name:
            transforms.append(T2_dict[bn])
        transforms.append(T_obj)
        res = diffcoll.forward(torch.stack(transforms, dim=1))
        if i == 0:
            print(f"wp1: {res.wp1}")
            print(f"sdf: {res.sdf}")
            print(f"cpidx: {res.cpidx}")
            print(f"normal: {res.normal}")
            cpidx = res.cpidx.tolist()[0]
            id_to_name = dict(zip(diffcoll.ctx.mesh_ids.tolist(), mesh_names))
            print(
                [(id_to_name[cpidx[i][0]], id_to_name[cpidx[i][1]]) for i in range(5)]
            )
        loss = (
            (
                (
                    res.wp1[:, :collision_num]
                    - res.wp2[:, :collision_num]
                    + cfg.target_margin * res.normal[:, :collision_num]
                )
                ** 2
            ).sum()
            + ((tp1_o - res.wp1_o[:, :collision_num]) ** 2).sum()
            + ((tp2_o - res.wp2_o[:, :collision_num]) ** 2).sum()
        ) / collision_num

        loss.backward()
        if i % 100 == 0:
            logging.info(f"Iteration: {i}, Avg loss: {float(loss)}")
        if i > 0.8 * cfg.iter:
            step = 0.01
        elif i > 0.5 * cfg.iter:
            step = 0.1
        else:
            step = 1
        with torch.no_grad():
            joint_angle[:, :3] = (
                joint_angle[:, :3]
                - 0.001
                * step
                * joint_angle.grad[:, :3]
                / joint_angle.grad[:, :3].norm()
            )
            joint_angle[:, 3:7] = torch_normalize_vector(
                joint_angle[:, 3:7]
                - 0.001
                * step
                * joint_angle.grad[:, 3:7]
                / joint_angle.grad[:, 3:7].norm()
            )
            joint_angle[:, 7:] -= (
                0.01 * step * joint_angle.grad[:, 7:] / joint_angle.grad[:, 7:].norm()
            )

    if cfg.vis:
        vis_dict = diffcoll.get_debug_dict()
        name2material = {"tp1": "green", "tp2": "green", "wp1": "red", "wp2": "red"}
        for i in range(len(hand_body_name)):
            mesh_name = "mesh" + str(i + 1)
            name2material[mesh_name] = "orange"
        name2material["mesh" + str(len(hand_body_name) + 1)] = "blue"
        if cfg.vis_sample:
            name2material["ls1"], name2material["ls2"] = "purple", "purple"
        save_folder = os.path.join(cfg.log_dir, "vusd")
        vis_usd(vis_dict, [0], save_folder, name2material)


if __name__ == "__main__":
    main(MainConfig.from_yaml("examples/config/base.yaml").cli())
