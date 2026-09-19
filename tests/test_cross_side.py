import torch

from bridge_seg.models import PointNetPlusPlusSegmentation, PointNetSegmentation


def test_cross_side_forward_shapes_and_original_output() -> None:
    model = PointNetSegmentation(
        input_channels=9,
        num_classes=3,
        use_cross_side=True,
    )
    features = torch.randn(2, 9, 16)
    mirrored = torch.randn(2, 9, 16)

    fused, original = model.forward_cross_side(
        features,
        mirrored,
        return_original=True,
    )

    assert fused.shape == (2, 3, 16)
    assert original.shape == (2, 3, 16)
    assert torch.isfinite(fused).all()
    assert torch.isfinite(original).all()


def test_pointnet_plus_plus_forward_shape() -> None:
    model = PointNetPlusPlusSegmentation(
        input_channels=9,
        num_classes=3,
    )
    features = torch.randn(2, 9, 64)

    logits = model(features)

    assert logits.shape == (2, 3, 64)
    assert torch.isfinite(logits).all()
