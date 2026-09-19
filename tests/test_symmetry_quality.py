import numpy as np

from bridge_seg.symmetry import symmetry_axis_quality


def test_symmetry_quality_accepts_left_right_symmetric_bridge() -> None:
    rng = np.random.default_rng(5)
    half = 250
    x = rng.uniform(-20.0, 20.0, size=half)
    y = rng.uniform(0.2, 5.0, size=half)
    z = rng.normal(scale=0.2, size=half)
    xyz = np.concatenate(
        [
            np.stack([x, y, z], axis=1),
            np.stack([x, -y, z], axis=1),
        ],
        axis=0,
    )
    labels = np.zeros(len(xyz), dtype=np.int64)

    quality = symmetry_axis_quality(xyz, labels)

    assert quality["eligible"] is True
    assert quality["wrong_to_correct_ratio"] > 1.0


def test_symmetry_quality_rejects_fore_aft_periodic_geometry() -> None:
    rng = np.random.default_rng(6)
    x = np.tile(np.linspace(-30.0, 30.0, 20), 50)
    y = rng.uniform(0.2, 5.0, size=len(x))
    z = rng.normal(scale=0.2, size=len(x))
    xyz = np.stack([x, y, z], axis=1)
    labels = np.zeros(len(x), dtype=np.int64)

    quality = symmetry_axis_quality(xyz, labels)

    assert quality["eligible"] is False
