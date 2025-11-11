import numpy as np
import trimesh
import os
import mujoco
import logging

import hydra
from omegaconf import OmegaConf
import torch
import pytorch_kinematics as pk
from diffcollision.utils import DCTensorSpec
from diffcollision import DCMesh, DiffCollision

from util.vis import vis_usd
from util.rotation import set_seed, torch_normalize_vector, torch_quaternion_to_matrix


def get_meshes_from_mjcf(xml_path, vis_mode="visual"):
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    assert vis_mode == "visual" or vis_mode == "collision"

    body_mesh_dict = {}
    body_id_dict = {}
    for i in range(model.ngeom):
        geom = model.geom(i)
        mesh_id = geom.dataid
        body_id = geom.bodyid[0]
        body_name = model.body(body_id).name
        body_id_dict[body_name] = body_id

        if geom.contype == 0 and vis_mode != "visual":
            continue
        if geom.contype != 0 and vis_mode != "collision":
            continue

        if mesh_id == -1:  # Primitives
            if geom.type == 6:
                tm = trimesh.creation.box(extents=2 * geom.size)
            elif geom.type == 5:
                tm = trimesh.creation.cylinder(
                    radius=geom.size[0], height=2 * geom.size[1]
                )
            elif geom.type == 2:
                tm = trimesh.creation.icosphere(radius=geom.size[0])
            elif geom.type == 3:
                tm = trimesh.creation.capsule(
                    radius=geom.size[0], height=2 * geom.size[1]
                )
            elif geom.type == 0:
                tm = trimesh.creation.box(
                    extents=[2 * geom.size[-1], 2 * geom.size[-1], 0.001]
                )
            else:
                raise NotImplementedError(
                    f"Unsupported mujoco primitive type: {geom.type}. Available choices: 2(icosphere), 3(capsule), 5(cylinder), 6(box)."
                )
        else:  # Meshes
            mjm = model.mesh(mesh_id)
            vert = model.mesh_vert[mjm.vertadr[0] : mjm.vertadr[0] + mjm.vertnum[0]]
            face = model.mesh_face[mjm.faceadr[0] : mjm.faceadr[0] + mjm.facenum[0]]
            tm = trimesh.Trimesh(vertices=vert, faces=face)

        if vis_mode == "collision":
            tm = tm.convex_hull

        geom_rot = data.geom_xmat[i].reshape(3, 3)
        geom_trans = data.geom_xpos[i]
        body_rot = data.xmat[body_id].reshape(3, 3)
        body_trans = data.xpos[body_id]
        tm.vertices = (tm.vertices @ geom_rot.T + geom_trans - body_trans) @ body_rot

        if body_name not in body_mesh_dict:
            body_mesh_dict[body_name] = [tm]
        else:
            body_mesh_dict[body_name].append(tm)

    for body_name, body_mesh in body_mesh_dict.items():
        body_mesh_dict[body_name] = trimesh.util.concatenate(body_mesh)
    return list(body_mesh_dict.keys()), list(body_mesh_dict.values())


def load_scene_cfg(scene_path: str) -> dict:
    scene_cfg = np.load(scene_path, allow_pickle=True).item()

    def update_relative_path(d: dict):
        for k, v in d.items():
            if isinstance(v, dict):
                update_relative_path(v)
            elif k.endswith("_path") and isinstance(v, str):
                d[k] = os.path.join(os.path.dirname(scene_path), v)
        return

    update_relative_path(scene_cfg["scene"])

    return scene_cfg


@hydra.main(config_path="config", config_name="base", version_base=None)
def main(cfg):
    set_seed(cfg.seed)
    ts = DCTensorSpec(cfg.device, cfg.dtype)

    npy_path = "examples/assets/grasp.npy"
    xml_path = "examples/assets/hand/shadow/right.xml"
    hand_body_name, hand_body_mesh = get_meshes_from_mjcf(xml_path)

    data = np.load(npy_path, allow_pickle=True).item()
    scene_cfg = load_scene_cfg(os.path.join("examples", data["scene_path"]))
    obj_name = scene_cfg["task"]["obj_name"]
    obj_path = os.path.dirname(
        os.path.dirname((scene_cfg["scene"][obj_name]["file_path"]))
    )
    obj_scale = scene_cfg["scene"][obj_name]["scale"]

    fingertip_name = [
        "rh_ffdistal",
        "rh_mfdistal",
        "rh_rfdistal",
        "rh_lfdistal",
        "rh_thdistal",
    ]
    tp1_o = ts.to(
        [
            [0.028, -0.047027, -0.012547],
            [0.0, -0.047027, -0.012547],
            [-0.022, -0.047027, -0.012547],
            [-0.043, -0.047027, -0.012547],
            [0.014448, -0.047, 0.012569],
        ]
    ).unsqueeze(0)
    tp2_o = ts.to(
        [[0.000587, -0.006644, 0.018874]] * 4 + [[-0.004527, -0.006819, 0.023165]]
    ).unsqueeze(0)

    mesh_lst = []
    for name, mesh in zip(hand_body_name, hand_body_mesh):
        mesh_lst.append(DCMesh.from_data(mesh.vertices, mesh.faces, ts))
    mesh_lst.append(DCMesh.from_file(obj_path, obj_scale[0], False, ts))

    obj_id = -1
    collision_pairs = []
    for name in fingertip_name:
        collision_pairs.append((obj_id, hand_body_name.index(name)))

    dcd_cfg = OmegaConf.to_container(cfg.dcd, resolve=True)
    diffcoll = DiffCollision(
        mesh_lst, collision_pairs, tp1_o=tp1_o, tp2_o=tp2_o, **dcd_cfg
    )

    T_obj = ts.to(torch.eye(4)[None])
    joint_angle = ts.to(data["grasp_qpos"])
    joint_angle.requires_grad_()

    # Forward kinematics
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
        loss = (
            ((res.wp1 - res.wp2 + cfg.margin * res.normal) ** 2).sum()
            + ((tp1_o - res.wp1_o) ** 2).sum()
            + ((tp2_o - res.wp2_o) ** 2).sum()
        ) / len(collision_pairs)

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
            vis_dict.meshes[i].coarse_mesh = hand_body_mesh[i]
            mesh_name = "mesh" + str(i + 1)
            name2material[mesh_name] = "orange"
        name2material["mesh" + str(len(hand_body_name) + 1)] = "blue"
        if cfg.vis_sample:
            name2material["ls1"], name2material["ls2"] = "purple", "purple"
        save_folder = os.path.join(cfg.log_dir, "vusd")
        vis_usd(vis_dict, [0], save_folder, name2material)


if __name__ == "__main__":
    main()
