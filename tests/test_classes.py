import pytest

from bridge_seg.classes import bridge_family_from_scene, source_type_from_scene


def test_scene_metadata_mapping() -> None:
    assert source_type_from_scene("c-bridge4") == "real"
    assert source_type_from_scene("c-f-bridge4_s2") == "virtual"
    assert bridge_family_from_scene("c-f-bridge4_s2") == "c-bridge4"
    assert bridge_family_from_scene("s-f-bridge1_le3") == "s-bridge1"
    assert bridge_family_from_scene("bridge_13_fr_rtc") == "semanticbridge-fr"


def test_invalid_scene_name_rejected() -> None:
    with pytest.raises(ValueError):
        bridge_family_from_scene("unknown-scene")
