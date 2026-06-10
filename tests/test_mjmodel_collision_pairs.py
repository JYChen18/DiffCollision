import mujoco
import torch
from dataclasses import fields
import pytest

from diffcollision import (
    AnalyticalConfig,
    DiffCollision,
    build_diffcoll_config,
    FDConfig,
    RS0Config,
    RS1DirConfig,
    RS1DistConfig,
)
from diffcollision.mjmesh import get_mesh_pair_params_from_mjmodel


def _model(xml: str) -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_string(xml)


def _pair_names(model: mujoco.MjModel, diffcoll: DiffCollision) -> set[tuple[str, str]]:
    return {
        (
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(idx1)),
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(idx2)),
        )
        for idx1, idx2 in diffcoll.ctx.mesh_pair_ids
    }


def _body_pair_ids(model: mujoco.MjModel, name1: str, name2: str) -> tuple[int, int]:
    return tuple(
        sorted(
            (
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name1),
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name2),
            )
        )
    )


def _mesh_pair_margins_from_mjmodel(
    model: mujoco.MjModel,
) -> dict[tuple[int, int], float]:
    mesh_pairs, pair_margins, _ = get_mesh_pair_params_from_mjmodel(model)
    return dict(zip(mesh_pairs, pair_margins))


def _mesh_pair_params_from_mjmodel(
    model: mujoco.MjModel,
) -> dict[tuple[int, int], tuple[float, float]]:
    mesh_pairs, pair_margins, pair_gaps = get_mesh_pair_params_from_mjmodel(model)
    return dict(zip(mesh_pairs, zip(pair_margins, pair_gaps)))


def _mesh_pairs_from_mjmodel(model: mujoco.MjModel) -> set[tuple[int, int]]:
    mesh_pairs, _, _ = get_mesh_pair_params_from_mjmodel(model)
    return set(mesh_pairs)


def _diffcollision_from_mjmodel(model: mujoco.MjModel) -> DiffCollision:
    return DiffCollision.from_mjmodel(model)


def test_config_from_mjmodel_keeps_compatible_non_adjacent_body_pairs():
    model = _model("""
        <mujoco>
          <worldbody>
            <body name="a" pos="-1 0 0">
              <joint type="free"/>
              <geom name="ga" type="box" size=".1 .1 .1"/>
            </body>
            <body name="b" pos="1 0 0">
              <joint type="free"/>
              <geom name="gb" type="box" size=".1 .1 .1"/>
            </body>
          </worldbody>
        </mujoco>
        """)

    diffcoll = _diffcollision_from_mjmodel(model)

    assert _pair_names(model, diffcoll) == {("a", "b")}


def test_config_from_mjmodel_exposes_body_ids_not_compact_mesh_indices():
    model = _model("""
        <mujoco>
          <worldbody>
            <body name="empty"/>
            <body name="a" pos="-1 0 0">
              <joint type="free"/>
              <geom name="ga" type="box" size=".1 .1 .1"/>
            </body>
            <body name="b" pos="1 0 0">
              <joint type="free"/>
              <geom name="gb" type="box" size=".1 .1 .1"/>
            </body>
          </worldbody>
        </mujoco>
        """)

    diffcoll = _diffcollision_from_mjmodel(model)
    body_id_a = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "a")
    body_id_b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "b")

    assert diffcoll.ctx.mesh_ids.tolist() == [body_id_a, body_id_b]
    assert diffcoll.ctx.mesh_pair_ids.tolist() == [[body_id_a, body_id_b]]
    assert diffcoll.ctx.mesh_pair_indices.tolist() == [[0, 1]]


def test_forward_bodyid_uses_body_ids():
    model = _model("""
        <mujoco>
          <worldbody>
            <body name="empty"/>
            <body name="a">
              <joint type="free"/>
              <geom name="ga" type="box" size=".1 .1 .1"/>
            </body>
            <body name="b">
              <joint type="free"/>
              <geom name="gb" type="box" size=".1 .1 .1"/>
            </body>
          </worldbody>
        </mujoco>
        """)

    diffcoll = _diffcollision_from_mjmodel(model)
    transforms = (
        torch.eye(4).view(1, 1, 4, 4).repeat(1, len(diffcoll.ctx.mesh_ids), 1, 1)
    )
    result = diffcoll.forward(transforms)

    body_id_a = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "a")
    body_id_b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "b")
    assert result.bodyid[0, 0].tolist() == [body_id_a, body_id_b]
    assert result.pos.shape[-2:] == (2, 3)
    assert result.pos_o.shape[-2:] == (2, 3)
    assert result.frame.shape[-2:] == (3, 3)
    assert not hasattr(result, "wp1")
    assert not hasattr(result, "normal")
    assert not hasattr(result, "sdf")
    assert not hasattr(result, "n1_o")


def test_config_from_mjmodel_reads_pair_margin_tensor_from_mujoco_geoms():
    model = _model("""
        <mujoco>
          <worldbody>
            <body name="a" pos="-1 0 0">
              <joint type="free"/>
              <geom name="ga" type="box" size=".1 .1 .1" margin=".03"/>
            </body>
            <body name="b" pos="1 0 0">
              <joint type="free"/>
              <geom name="gb" type="box" size=".1 .1 .1" margin=".07"/>
            </body>
          </worldbody>
        </mujoco>
        """)

    diffcoll = _diffcollision_from_mjmodel(model)

    assert diffcoll.ctx.pair_margin.shape == (1,)
    assert torch.allclose(diffcoll.ctx.pair_margin, torch.tensor([0.10]))
    assert diffcoll.ctx.pair_gap.shape == (1,)
    assert torch.allclose(diffcoll.ctx.pair_gap, torch.tensor([0.0]))


def test_config_from_mjmodel_stores_force_margin_and_gap_from_mujoco_geoms():
    model = _model("""
        <mujoco>
          <worldbody>
            <body name="a" pos="-1 0 0">
              <joint type="free"/>
              <geom name="ga1" type="box" size=".1 .1 .1" margin=".03" gap=".01"/>
              <geom name="ga2" type="box" size=".1 .1 .1" margin=".10" gap=".06"/>
            </body>
            <body name="b" pos="1 0 0">
              <joint type="free"/>
              <geom name="gb" type="box" size=".1 .1 .1" margin=".07" gap=".02"/>
            </body>
          </worldbody>
        </mujoco>
        """)

    diffcoll = _diffcollision_from_mjmodel(model)

    assert torch.allclose(diffcoll.ctx.pair_margin, torch.tensor([0.17]))
    assert torch.allclose(diffcoll.ctx.pair_gap, torch.tensor([0.08]))
    assert torch.allclose(
        diffcoll.ctx.pair_margin + diffcoll.ctx.pair_gap,
        torch.tensor([0.25]),
    )


def test_diffcollision_config_is_public_and_context_holds_runtime_state():
    model = _model("""
        <mujoco>
          <worldbody>
            <body name="a">
              <joint type="free"/>
              <geom name="ga" type="box" size=".1 .1 .1"/>
            </body>
            <body name="b">
              <joint type="free"/>
              <geom name="gb" type="box" size=".1 .1 .1"/>
            </body>
          </worldbody>
        </mujoco>
        """)

    diffcoll = _diffcollision_from_mjmodel(model)

    assert all(not f.name.startswith("_") for f in fields(diffcoll.cfg))
    assert diffcoll.get_context() is diffcoll.ctx
    assert isinstance(
        DiffCollision(
            diffcoll.ctx.meshes,
            mesh_pairs=diffcoll.ctx.mesh_pair_ids,
            mesh_ids=diffcoll.ctx.mesh_ids,
        ),
        DiffCollision,
    )
    with pytest.raises(TypeError, match="config"):
        DiffCollision(diffcoll.ctx.meshes, config=object())


def test_diffcollision_config_loads_typed_collision_union_from_dict():
    expected_types = {
        "RS1Dist": RS1DistConfig,
        "RS1Dir": RS1DirConfig,
        "RS0": RS0Config,
        "FD": FDConfig,
        "Analytical": AnalyticalConfig,
    }

    for method, config_type in expected_types.items():
        cfg = build_diffcoll_config({"type": method})
        assert isinstance(cfg, config_type)

    cfg = build_diffcoll_config({"type": "FD", "eps_r": 0.2})
    assert cfg.eps_r == 0.2


def test_explicit_geom_pair_uses_pair_margin():
    model = _model("""
        <mujoco>
          <worldbody>
            <body name="a" pos="-1 0 0">
              <joint type="free"/>
              <geom name="ga" type="box" size=".1 .1 .1" margin=".03"/>
            </body>
            <body name="b" pos="1 0 0">
              <joint type="free"/>
              <geom name="gb" type="box" size=".1 .1 .1" margin=".07"/>
            </body>
          </worldbody>
          <contact>
            <pair geom1="ga" geom2="gb" margin=".04"/>
          </contact>
        </mujoco>
        """)

    pair_margins = _mesh_pair_margins_from_mjmodel(model)

    assert pair_margins[_body_pair_ids(model, "a", "b")] == 0.04


def test_explicit_geom_pair_uses_pair_gap():
    model = _model("""
        <mujoco>
          <worldbody>
            <body name="a" pos="-1 0 0">
              <joint type="free"/>
              <geom name="ga" type="box" size=".1 .1 .1" margin=".03" gap=".01"/>
            </body>
            <body name="b" pos="1 0 0">
              <joint type="free"/>
              <geom name="gb" type="box" size=".1 .1 .1" margin=".07" gap=".02"/>
            </body>
          </worldbody>
          <contact>
            <pair geom1="ga" geom2="gb" margin=".04" gap=".03"/>
          </contact>
        </mujoco>
        """)

    pair_params = _mesh_pair_params_from_mjmodel(model)

    assert pair_params[_body_pair_ids(model, "a", "b")] == pytest.approx((0.04, 0.03))


def test_parent_child_body_pairs_are_filtered_by_default():
    model = _model("""
        <mujoco>
          <worldbody>
            <body name="parent">
              <joint type="free"/>
              <geom name="gp" type="box" size=".1 .1 .1"/>
              <body name="child" pos=".3 0 0">
                <joint type="hinge" axis="0 0 1"/>
                <geom name="gc" type="box" size=".1 .1 .1"/>
              </body>
            </body>
          </worldbody>
        </mujoco>
        """)

    assert _body_pair_ids(model, "parent", "child") not in _mesh_pairs_from_mjmodel(
        model
    )


def test_parent_child_body_pairs_are_kept_when_filterparent_is_disabled():
    model = _model("""
        <mujoco>
          <option>
            <flag filterparent="disable"/>
          </option>
          <worldbody>
            <body name="parent">
              <joint type="free"/>
              <geom name="gp" type="box" size=".1 .1 .1"/>
              <body name="child" pos=".3 0 0">
                <joint type="hinge" axis="0 0 1"/>
                <geom name="gc" type="box" size=".1 .1 .1"/>
              </body>
            </body>
          </worldbody>
        </mujoco>
        """)

    assert _body_pair_ids(model, "parent", "child") in _mesh_pairs_from_mjmodel(model)


def test_same_weld_body_pairs_are_filtered():
    model = _model("""
        <mujoco>
          <worldbody>
            <body name="root">
              <body name="a" pos="-.3 0 0">
                <geom name="ga" type="box" size=".1 .1 .1"/>
              </body>
              <body name="b" pos=".3 0 0">
                <geom name="gb" type="box" size=".1 .1 .1"/>
              </body>
            </body>
          </worldbody>
        </mujoco>
        """)

    assert _body_pair_ids(model, "a", "b") not in _mesh_pairs_from_mjmodel(model)


def test_incompatible_contact_masks_are_filtered():
    model = _model("""
        <mujoco>
          <worldbody>
            <body name="a" pos="-1 0 0">
              <joint type="free"/>
              <geom name="ga" type="box" size=".1 .1 .1" contype="1" conaffinity="0"/>
            </body>
            <body name="b" pos="1 0 0">
              <joint type="free"/>
              <geom name="gb" type="box" size=".1 .1 .1" contype="2" conaffinity="0"/>
            </body>
          </worldbody>
        </mujoco>
        """)

    assert _body_pair_ids(model, "a", "b") not in _mesh_pairs_from_mjmodel(model)


def test_explicit_geom_pair_bypasses_parent_and_mask_filters():
    model = _model("""
        <mujoco>
          <worldbody>
            <body name="parent">
              <joint type="free"/>
              <geom name="gp" type="box" size=".1 .1 .1" contype="1" conaffinity="0"/>
              <body name="child" pos=".3 0 0">
                <joint type="hinge" axis="0 0 1"/>
                <geom name="gc" type="box" size=".1 .1 .1" contype="2" conaffinity="0"/>
              </body>
            </body>
          </worldbody>
          <contact>
            <pair geom1="gp" geom2="gc"/>
          </contact>
        </mujoco>
        """)

    assert _body_pair_ids(model, "parent", "child") in _mesh_pairs_from_mjmodel(model)
