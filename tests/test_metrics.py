import numpy as np

from bridge_seg.metrics import class_iou, confusion_matrix, segmentation_metrics


def test_confusion_matrix_and_iou() -> None:
    truth = np.asarray([0, 0, 1, 1, 2, 2])
    prediction = np.asarray([0, 1, 1, 1, 2, 0])

    matrix = confusion_matrix(truth, prediction, num_classes=3)
    ious = class_iou(truth, prediction, num_classes=3)

    assert matrix.tolist() == [[1, 1, 0], [0, 2, 0], [1, 0, 1]]
    assert np.allclose(ious, [1.0 / 3.0, 2.0 / 3.0, 0.5])


def test_segmentation_metrics_shape() -> None:
    metrics = segmentation_metrics([0, 1, 2], [0, 2, 1], num_classes=3)

    assert metrics["accuracy"] == 1.0 / 3.0
    assert len(metrics["per_class_iou"]) == 3
