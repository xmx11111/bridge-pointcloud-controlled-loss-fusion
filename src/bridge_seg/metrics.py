from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def confusion_matrix(
    y_true: Sequence[int] | np.ndarray,
    y_pred: Sequence[int] | np.ndarray,
    num_classes: int,
) -> np.ndarray:
    true = np.asarray(y_true, dtype=np.int64).reshape(-1)
    pred = np.asarray(y_pred, dtype=np.int64).reshape(-1)
    if true.shape != pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    if num_classes <= 0:
        raise ValueError("num_classes must be positive")
    if np.any((true < 0) | (true >= num_classes)):
        raise ValueError("y_true contains an out-of-range label")
    if np.any((pred < 0) | (pred >= num_classes)):
        raise ValueError("y_pred contains an out-of-range label")
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(matrix, (true, pred), 1)
    return matrix


def class_iou(
    y_true: Sequence[int] | np.ndarray,
    y_pred: Sequence[int] | np.ndarray,
    num_classes: int,
) -> np.ndarray:
    matrix = confusion_matrix(y_true, y_pred, num_classes)
    true_positive = np.diag(matrix).astype(np.float64)
    false_positive = matrix.sum(axis=0) - true_positive
    false_negative = matrix.sum(axis=1) - true_positive
    return true_positive / np.maximum(true_positive + false_positive + false_negative, 1.0)


def macro_f1(
    y_true: Sequence[int] | np.ndarray,
    y_pred: Sequence[int] | np.ndarray,
    num_classes: int,
) -> float:
    matrix = confusion_matrix(y_true, y_pred, num_classes)
    true_positive = np.diag(matrix).astype(np.float64)
    false_positive = matrix.sum(axis=0) - true_positive
    false_negative = matrix.sum(axis=1) - true_positive
    f1 = 2.0 * true_positive / np.maximum(
        2.0 * true_positive + false_positive + false_negative,
        1.0,
    )
    return float(f1.mean())


def segmentation_metrics(
    y_true: Sequence[int] | np.ndarray,
    y_pred: Sequence[int] | np.ndarray,
    num_classes: int,
) -> dict[str, object]:
    matrix = confusion_matrix(y_true, y_pred, num_classes)
    ious = class_iou(y_true, y_pred, num_classes)
    accuracy = float(np.diag(matrix).sum() / max(int(matrix.sum()), 1))
    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1(y_true, y_pred, num_classes),
        "mean_iou": float(ious.mean()),
        "per_class_iou": ious.tolist(),
        "confusion_matrix": matrix.tolist(),
    }
