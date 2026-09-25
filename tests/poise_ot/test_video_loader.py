"""Regression test for video_loader.load_clip_raw, built from real HOI4D
sample data (category Safe, clip ZY20210800001/H1/C6/N03/S247/s01/T1,
frames 0-1 and instance 003's mobility_v2.json -- captured verbatim from a
real HOI4D_annotations.zip / HOI4D_CAD_Model_for_release.zip)."""

import json

import numpy as np
import pytest

from poise_ot.video.video_loader import load_clip_raw

CLIP_ID = "ZY20210800001/H1/C6/N03/S247/s01/T1"

FRAME_0 = {
    "frameId": 1, "filePath": "16-x/x/x/x/0.pcd", "isEffective": 1,
    "dataList": [
        {"id": 1, "label": "Safebox", "center": {"x": -0.13970132127910612, "y": 0.07367290844664431, "z": 0.9411795698661569},
         "dimensions": {"height": 0.18499179908054858, "length": 0.10972899125415074, "width": 0.12432640468980759},
         "rotation": {"x": 2.307217836380005, "y": -0.016146354377269745, "z": -1.4604086875915527}, "properties": {}},
        {"id": 2, "label": "Safedoor", "center": {"x": -0.13479353458595783, "y": 0.11080812000154888, "z": 0.9004963767618762},
         "dimensions": {"height": 0.14870844536137956, "length": 0.020282726448277492, "width": 0.08910031690275472},
         "rotation": {"x": 2.2967536449432373, "y": -0.015749791637063026, "z": -1.4600938558578491}, "properties": {}},
    ],
}

FRAME_1 = {
    "frameId": 1, "filePath": "", "isEffective": 1,
    "dataList": [
        {"id": 1, "label": "Safebox", "center": {"x": -0.13970504027228753, "y": 0.07341370286785914, "z": 0.9409244605616297},
         "dimensions": {"height": 0.18499179908054858, "length": 0.10972899125415074, "width": 0.12432640468980759},
         "rotation": {"x": 2.3084141592758023, "y": -0.014208050490420288, "z": -1.460569611969422}, "properties": {}},
        {"id": 2, "label": "Safedoor", "center": {"x": -0.134617687854656, "y": 0.11080646881737113, "z": 0.9004096765489661},
         "dimensions": {"height": 0.14870844536137956, "length": 0.020282726448277492, "width": 0.08910031690275472},
         "rotation": {"x": 2.298309454008936, "y": -0.013769868220449233, "z": -1.4601980652213966}, "properties": {}},
    ],
}

MOBILITY_V2 = [
    {"id": 0, "parent": -1, "joint": "自由", "name": "frame", "parts": [{"id": 2, "name": "frame", "children": []}], "jointData": {}},
    {"id": 1, "parent": 0, "joint": "铰链（旋转）", "name": "door_frame", "parts": [{"id": 4, "name": "door_frame", "children": []}],
     "jointData": {"axis": {"origin": [0.008346977725625038, 0.03893748791102886, 0], "direction": [0, 0, 1]},
                   "limit": {"a": 0, "b": -104.76000000000002, "noLimit": False}}},
]


@pytest.fixture
def hoi4d_root(tmp_path):
    objpose_dir = tmp_path / "HOI4D_annotations" / CLIP_ID / "objpose"
    objpose_dir.mkdir(parents=True)
    (objpose_dir / "0.json").write_text(json.dumps(FRAME_0))
    (objpose_dir / "1.json").write_text(json.dumps(FRAME_1))

    cad_dir = tmp_path / "HOI4D_CAD_Model_for_release" / "articulated" / "Safe" / "003"
    cad_dir.mkdir(parents=True)
    (cad_dir / "mobility_v2.json").write_text(json.dumps(MOBILITY_V2, ensure_ascii=False), encoding="utf-8")
    return tmp_path


def test_load_clip_raw_recovers_near_zero_angle_at_clip_start(hoi4d_root):
    # The door hasn't started opening yet at frames 0-1 of this real clip
    # (joint limit is 0 deg to -104.76 deg) -- the relative-rotation-onto-
    # axis computation should recover a near-zero angle here, not the
    # ~131.6 deg a naive read of Safedoor's raw (camera-frame-contaminated)
    # rotation.x would give.
    clip = load_clip_raw(CLIP_ID, root=hoi4d_root)

    assert clip.joint_type == "revolute"
    assert clip.category == "Safe"
    assert np.isfinite(clip.theta).all()
    assert np.all(np.abs(clip.theta) < np.radians(1.0))
    np.testing.assert_allclose(clip.t, [0.0, 1.0 / 15.0])


def test_load_clip_raw_missing_clip_raises_clear_error(hoi4d_root):
    with pytest.raises(FileNotFoundError):
        load_clip_raw("ZY20210800001/H1/C6/N99/S247/s01/T1", root=hoi4d_root)
