import numpy as np

from bridge_seg.occlusion import occlude_points


def make_points(count: int = 100) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)
    xyz = rng.normal(size=(count, 3)).astype(np.float32)
    rgb = rng.random(size=(count, 3)).astype(np.float32)
    labels = np.arange(count, dtype=np.int64) % 3
    return xyz, rgb, labels


def test_random_occlusion_is_exact_and_deterministic() -> None:
    xyz, rgb, labels = make_points()

    first = occlude_points(xyz, rgb, labels, 0.25, "random", seed=7)
    second = occlude_points(xyz, rgb, labels, 0.25, "random", seed=7)

    assert first.metadata["removed_count"] == 25
    assert first.metadata["kept_count"] == 75
    assert np.array_equal(first.kept_indices, second.kept_indices)


def test_viewpoint_occlusion_is_exact_for_all_rates() -> None:
    xyz, rgb, labels = make_points()

    for rate, expected_kept in ((0.25, 75), (0.50, 50), (0.75, 25)):
        result = occlude_points(xyz, rgb, labels, rate, "viewpoint", seed=11)
        assert result.metadata["kept_count"] == expected_kept
        assert result.xyz.shape == (expected_kept, 3)
