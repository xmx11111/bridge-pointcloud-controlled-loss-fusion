import numpy as np

from bridge_seg.periodicity import estimate_periodicity
from bridge_seg.symmetry import BridgeFrame


def make_frame() -> BridgeFrame:
    return BridgeFrame(
        origin=np.zeros(3),
        longitudinal=np.asarray([1.0, 0.0, 0.0]),
        transverse=np.asarray([0.0, 1.0, 0.0]),
        vertical=np.asarray([0.0, 0.0, 1.0]),
    )


def test_periodicity_detects_repeated_cross_sections() -> None:
    rng = np.random.default_rng(42)
    period = 12.0
    chunks = []
    for center in (0.0, period, 2.0 * period, 3.0 * period):
        x = center + rng.normal(scale=0.25, size=1500)
        y = rng.normal(scale=2.0, size=1500)
        z = rng.normal(scale=1.5, size=1500)
        chunks.append(np.stack([x, y, z], axis=1))
    xyz = np.concatenate(chunks, axis=0)

    result = estimate_periodicity(xyz, make_frame())

    assert abs(result.period_length - period) < 2.0
    assert result.confidence > 0.0
    assert result.repeats > 2.0


def test_periodicity_is_stronger_for_repeated_geometry_than_noise() -> None:
    rng = np.random.default_rng(7)
    periodic = []
    for center in np.linspace(0.0, 36.0, 4):
        periodic.append(
            np.stack(
                [
                    center + rng.normal(scale=0.2, size=1000),
                    rng.normal(scale=2.0, size=1000),
                    rng.normal(scale=1.0, size=1000),
                ],
                axis=1,
            )
        )
    noisy = rng.normal(size=(4000, 3))
    noisy[:, 0] *= 12.0
    noisy[:, 1] *= 2.0
    noisy[:, 2] *= 1.0

    repeated = estimate_periodicity(np.concatenate(periodic), make_frame())
    random = estimate_periodicity(noisy, make_frame())

    assert repeated.confidence > random.confidence
