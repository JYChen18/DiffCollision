import torch
import numpy as np
import hydra
from omegaconf import OmegaConf
import logging
import os
import traceback

from util.rotation import (
    set_seed,
    torch_normalize_vector,
    torch_quaternion_to_matrix,
    sample_target_point,
)
from util.vis import vis_usd

from diffcollision.utils import (
    DCTensorSpec,
    torch_se3_exp_map,
    torch_matrix_grad_to_se3,
)
from diffcollision import DiffCollision, DCMesh


def rand_problem(cfg):
    if cfg.prob_rand is None:
        cfg.prob_rand = []
    if "obj" in cfg.prob_rand:
        all_obj = sorted(os.listdir(cfg.asset_dir))
        obj_id = np.random.choice(all_obj, 2)
    else:
        obj_id = cfg.obj

    if "scale" in cfg.prob_rand:
        scale = np.clip(np.random.rand(2) * 0.2, a_min=0.01, a_max=0.2) / 2
    else:
        scale = cfg.scale

    if "convex" in cfg.prob_rand:
        convex = np.random.rand(2) < 0.5
    else:
        convex = cfg.convex

    if "tp" in cfg.prob_rand:
        rand_num = np.random.rand(1)
        if rand_num < 1 / 3:
            tp_type = ["f", "v"]
        elif rand_num < 2 / 3:
            tp_type = ["v", "f"]
        else:
            tp_type = ["v", "v"]
    else:
        tp_type = cfg.tp
    return obj_id, scale, convex, tp_type


# Global random state for loading objects to ensure reproducibility
global PROB_RAND_STATE
PROB_RAND_STATE = None


def single_problem(prob_id, cfg):
    torch.cuda.empty_cache()  # avoid GPU memory leakage
    ts = DCTensorSpec(cfg.device, cfg.dtype)

    # Load objects
    read_flag = True
    global PROB_RAND_STATE
    while read_flag:
        if PROB_RAND_STATE is not None:
            np.random.set_state(PROB_RAND_STATE)
        obj_id, scale, convex, tp_type = rand_problem(cfg)
        try:
            mesh1 = DCMesh.from_file(
                os.path.join(cfg.asset_dir, obj_id[0]), scale[0], convex[0], ts
            )
            mesh2 = DCMesh.from_file(
                os.path.join(cfg.asset_dir, obj_id[1]), scale[1], convex[1], ts
            )
            read_flag = False
        except:
            logging.info(
                f"Loading error: \n Asset folder: {cfg.asset_dir} \n Object IDs: {obj_id[0]} {obj_id[1]} \n {traceback.format_exc()}"
            )
        PROB_RAND_STATE = np.random.get_state()

    # Sample target points (in the object local frame)
    b = cfg.n_tp
    tp1_o, tp2_o = sample_target_point(
        mesh1, mesh2, cfg.n_tp, tp_type, cfg.tp_check, ts
    )  # b, 1, 3

    # Setup configurations
    dcd_cfg = OmegaConf.to_container(cfg.dcd, resolve=True)
    diffcoll = DiffCollision([mesh1, mesh2], tp1_o=tp1_o, tp2_o=tp2_o, **dcd_cfg)

    # Initialize object poses
    t1 = ts.to([[0, 0, 0]]).expand(b, 3)
    qraw1 = ts.to([[1, 0, 0, 0]]).expand(b, 4)
    r1 = torch_quaternion_to_matrix(qraw1)
    T1 = ts.to(torch.zeros((b, 4, 4)))
    T1[:, :3, :3] = r1
    T1[:, :3, 3] = t1
    T1[:, 3, 3] = 1
    T1.requires_grad_()

    t2 = ts.to(torch.randn((b, 3)) * 0.1)
    qraw2 = ts.to(torch.randn((b, 4)))
    r2 = torch_quaternion_to_matrix(qraw2)
    T2 = ts.to(torch.zeros((b, 4, 4)))
    T2[:, :3, :3] = r2
    T2[:, :3, 3] = t2
    T2[:, 3, 3] = 1
    T2.requires_grad_()

    # Main optimization loop
    for i in range(cfg.iter):
        with torch.no_grad():
            T1.grad = None
            T2.grad = None
        res = diffcoll.forward(torch.stack([T1, T2], dim=-3))
        loss = (
            ((res.wp1 - res.wp2 + cfg.margin * res.normal) ** 2).sum()
            + ((tp1_o - res.wp1_o) ** 2).sum()
            + ((tp2_o - res.wp2_o) ** 2).sum()
        )
        loss.backward()
        with torch.no_grad():
            if i < 0.1 * cfg.iter:
                step = 1.0
            elif i > 0.9 * cfg.iter:
                step = 0.01
            else:
                step = 0.1

            if cfg.upd1:  # Update T1
                step *= 0.5
                proj1 = torch_matrix_grad_to_se3(T1, T1.grad)
                proj1 = torch_normalize_vector(proj1) * step
                T1[:] = T1 @ torch_se3_exp_map(-proj1, cfg.step_r, cfg.step_t)

            # Update T2
            proj2 = torch_matrix_grad_to_se3(T2, T2.grad)
            proj2 = torch_normalize_vector(proj2) * step
            T2[:] = T2 @ torch_se3_exp_map(-proj2, cfg.step_r, cfg.step_t)

    # Final evaluation
    with torch.no_grad():
        res = diffcoll.forward(torch.stack([T1, T2], dim=-3), skip_debug=True)
        final_loss = (
            ((res.wp1 - res.wp2 + cfg.margin * res.normal) ** 2).sum(dim=-1)
            + ((tp1_o - res.wp1_o) ** 2).sum(dim=-1)
            + ((tp2_o - res.wp2_o) ** 2).sum(dim=-1)
        ).squeeze(1)

        Err_Med = torch.quantile(final_loss, 0.5)
        Err_D9 = torch.quantile(final_loss, 0.9, interpolation="nearest")
        acc6_sum = (final_loss < 1e-6).sum() / b
        logging.info(f"Problem: {prob_id}")
        logging.info(f"Object: {obj_id[0]},{obj_id[1]}")
        logging.info(f"Scale: {scale[0]},{scale[1]}")
        logging.info(f"Convex: {convex[0]},{convex[1]}")
        logging.info(f"Tp: {tp_type[0]},{tp_type[1]}")
        logging.info(
            f"ErrD5: {Err_Med:.1e}, ErrD9: {Err_D9:.1e}, Acc: {acc6_sum*100:.1f}%"
        )

        # Visualization
        if cfg.vis:
            vis_ids = torch.topk(final_loss, min(10, cfg.n_tp))[-1]
            save_path = os.path.join(cfg.log_dir, "vusd", f"prob{prob_id}")
            name2material = {
                "mesh1": "blue",
                "mesh2": "orange",
                "tp1": "green",
                "tp2": "green",
                "wp1": "red",
                "wp2": "red",
            }
            if cfg.vis_sample:
                name2material["ls1"], name2material["ls2"] = "purple", "purple"
            vis_usd(diffcoll.get_debug_dict(), vis_ids.cpu(), save_path, name2material)
    return final_loss


@hydra.main(config_path="config", config_name="base", version_base=None)
def main(cfg):
    try:
        set_seed(cfg.seed)
        total_loss = []
        for prob_id in range(cfg.n_prob):
            total_loss.append(single_problem(prob_id, cfg))
        total_loss = torch.cat(total_loss).view(-1)
        Err_Med = torch.quantile(total_loss, 0.5)
        Err_D9 = torch.quantile(total_loss, 0.9, interpolation="nearest")
        Acc6 = (total_loss < 1e-6).sum() / len(total_loss)
        logging.info(
            f"(ALL) ErrD5: {Err_Med:.1e}, ErrD9: {Err_D9:.1e}, Acc: {Acc6*100:.1f}%"
        )
    except Exception as e:
        logging.error(f"{traceback.format_exc()}")
    return


if __name__ == "__main__":
    main()
