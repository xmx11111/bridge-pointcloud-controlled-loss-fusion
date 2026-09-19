import numpy as np

from bridge_seg.symmetry import estimate_bridge_frame, mirror_points


def test_bridge_frame_is_orthonormal_and_mirror_is_involutive() -> None:
    rng = np.random.default_rng(42)
    xyz = rng.normal(size=(500, 3))
    xyz[:, 0] *= 20.0
    xyz[:, 1] *= 4.0
    xyz[:, 2] *= 1.0

    frame = estimate_bridge_frame(xyz)
    basis = np.stack(
        [frame.longitudinal, frame.transverse, frame.vertical],
        axis=1,
    )

    np.testing.assert_allclose(basis.T @ basis, np.eye(3), atol=1e-10)
    mirrored = mirror_points(xyz, frame, axis="transverse")
    restored = mirror_points(mirrored, frame, axis="transverse")
    np.testing.assert_allclose(restored, xyz, atol=1e-10)
    np.testing.assert_allclose(
        np.linalg.norm(mirrored - frame.origin, axis=1),
        np.linalg.norm(xyz - frame.origin, axis=1),
        atol=1e-10,
    )


def test_girder_points_dominate_axis_estimation() -> None:
    rng = np.random.default_rng(7)
    girder = rng.normal(size=(500, 3))
    girder[:, 0] *= 20.0
    girder[:, 1] *= 3.0
    girder[:, 2] *= 0.5
    piers = rng.normal(size=(5000, 3))
    piers[:, 0] *= 1.0
    piers[:, 1] *= 8.0
    piers[:, 2] *= 4.0
    xyz = np.concatenate([girder, piers], axis=0)
    labels = np.concatenate(
        [np.zeros(len(girder), dtype=np.int64), np.ones(len(piers), dtype=np.int64)]
    )

    frame = estimate_bridge_frame(xyz, labels)
    all_point_frame = estimate_bridge_frame(xyz)

    np.testing.assert_allclose(
        np.abs(frame.longitudinal),
        np.array([1.0, 0.0, 0.0]),
        atol=0.2,
    )
    assert np.linalg.norm(frame.longitudinal - all_point_frame.longitudinal) > 0.1
