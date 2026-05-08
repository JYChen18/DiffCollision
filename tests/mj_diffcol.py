import mujoco
import numpy as np

MARGIN = 10.0


def _geom_name(model: mujoco.MjModel, gid: int) -> str:
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid)
    return name if name is not None else f"geom_{gid}"


def _configure_model_margin(model: mujoco.MjModel, margin: float) -> None:
    model.geom_margin[:] = margin
    model.geom_gap[:] = 0.0
    model.body_margin[:] = 2 * margin


def _print_contacts(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    ncon = data.ncon
    geom = data.contact.geom[:ncon]

    print(f"detected_contacts={ncon}")
    for i in range(ncon):
        g1, g2 = int(geom[i, 0]), int(geom[i, 1])
        d = mujoco.mj_geomDistance(
            model, data, g1, g2, float("inf"), np.zeros(6, dtype=np.float64)
        )
        print(f"{i:03d} {g1} {g2} {_geom_name(model, g1)} {_geom_name(model, g2)} {d}")


def main() -> None:
    xml_path = "examples/assets/grasp_env.xml"
    npy_path = "examples/assets/grasp.npy"
    data = np.load(npy_path, allow_pickle=True).item()

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_MULTICCD
    _configure_model_margin(model, MARGIN)

    mj_data = mujoco.MjData(model)
    mj_data.qpos[:29] = data["grasp_qpos"].squeeze(0)
    mj_data.qpos[1] -= 0.1
    print(mj_data.qpos)

    mujoco.mj_forward(model, mj_data)

    print(f"xml_path={xml_path}")
    print(f"geom_margin={MARGIN}")
    print(f"nbody={model.nbody}, ngeom={model.ngeom}, npair={model.npair}")
    _print_contacts(model, mj_data)


if __name__ == "__main__":
    main()
