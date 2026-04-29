"""Anomaly evaluation metrics for DFRNet."""
import numpy as np
from sklearn import metrics


def getImageThreshold(predict, label):
    """
    Computes the optimal threshold based on the highest F1-score.

    Args:
        predict: [1 x N] array of predicted anomaly scores (0 to 1).
        label: [1 x N] array of ground truth labels (0 or 1).

    Returns:
        best_f1_score: The maximum F1-score achieved.
        optimal_threshold: The threshold yielding the best F1-score.
    """
    precisions, recalls, thresholds = metrics.precision_recall_curve(label, predict)

    # Calculate F1-scores and find the optimal index (added 1e-8 for numerical stability)
    f1_scores = (2 * precisions * recalls) / (precisions + recalls + 1e-8)
    best_f1_score = np.max(f1_scores[np.isfinite(f1_scores)])
    best_f1_score_index = np.argmax(f1_scores[np.isfinite(f1_scores)])

    return best_f1_score, thresholds[best_f1_score_index]


def compute_imagewise_retrieval_metrics(
    anomaly_prediction_weights, anomaly_ground_truth_labels
):
    """
    Computes object-level/image-level retrieval statistics (AUROC, FPR, TPR).

    Args:
        anomaly_prediction_weights: [np.ndarray or list] [N] Assignment weights
                                    per point cloud/image. Higher indicates higher
                                    probability of being an anomaly.
        anomaly_ground_truth_labels: [np.ndarray or list] [N] Binary labels - 1
                                    if the object is an anomaly, 0 if normal.

    Returns:
        Dictionary containing AUROC, FPR, TPR, and thresholds.
    """
    fpr, tpr, thresholds = metrics.roc_curve(
        anomaly_ground_truth_labels, anomaly_prediction_weights
    )
    auroc = metrics.roc_auc_score(
        anomaly_ground_truth_labels, anomaly_prediction_weights
    )

    return {"auroc": auroc, "fpr": fpr, "tpr": tpr, "threshold": thresholds}


def compute_pixelwise_retrieval_metrics(
    anomaly_segmentations, ground_truth_masks
):
    """
    Computes point-level/pixel-wise statistics (AUROC, FPR, TPR) for anomaly localization.

    Args:
        anomaly_segmentations: [list of np.arrays or np.array] Contains
                                generated spatial/point anomaly scores.
        ground_truth_masks: [list of np.arrays or np.array] Contains
                            predefined ground truth anomaly masks.

    Returns:
        Dictionary containing point-level AUROC, optimal thresholds, and error rates.
    """
    if isinstance(anomaly_segmentations, list):
        anomaly_segmentations = np.stack(anomaly_segmentations)
    if isinstance(ground_truth_masks, list):
        ground_truth_masks = np.stack(ground_truth_masks)

    flat_anomaly_segmentations = anomaly_segmentations.ravel()
    flat_ground_truth_masks = ground_truth_masks.ravel()

    fpr, tpr, thresholds = metrics.roc_curve(
        flat_ground_truth_masks.astype(int), flat_anomaly_segmentations
    )
    auroc = metrics.roc_auc_score(
        flat_ground_truth_masks.astype(int), flat_anomaly_segmentations
    )

    precision, recall, thresholds = metrics.precision_recall_curve(
        flat_ground_truth_masks.astype(int), flat_anomaly_segmentations
    )
    F1_scores = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) != 0,
    )

    optimal_threshold = thresholds[np.argmax(F1_scores)]
    predictions = (flat_anomaly_segmentations >= optimal_threshold).astype(int)
    fpr_optim = np.mean(predictions > flat_ground_truth_masks)
    fnr_optim = np.mean(predictions < flat_ground_truth_masks)

    return {
        "auroc": auroc,
        "fpr": fpr,
        "tpr": tpr,
        "optimal_threshold": optimal_threshold,
        "optimal_fpr": fpr_optim,
        "optimal_fnr": fnr_optim,
    }