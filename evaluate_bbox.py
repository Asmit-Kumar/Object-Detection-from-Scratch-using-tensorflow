import numpy as np
from bbox_detector import BboxDetector


def calculate_iou(boxA, boxB):
    """
    Calculate Intersection over Union (IoU)
    box format: [x_min, y_min, x_max, y_max]
    """

    if boxA is None or boxB is None:
        return None

    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_width = max(0, xB - xA)
    inter_height = max(0, yB - yA)
    inter_area = inter_width * inter_height

    boxA_area = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
    boxB_area = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])

    union_area = boxA_area + boxB_area - inter_area

    if union_area == 0:
        return 0.0

    return inter_area / union_area


def evaluate_bbox_detection():

    images_path = "D:\Object-Detection-from-Scratch\Images"
    labels_path = "D:\Object-Detection-from-Scratch\label"

    detector = BboxDetector(images_path, labels_path)

    # Load ground truth and predictions
    detector.get_labels()
    detector.get_pred_bboxes()

    actual = np.array(detector.actual_bboxes)
    predicted = np.array(detector.pred_bboxes)

    if len(actual) != len(predicted):
        raise ValueError("Mismatch between actual and predicted bbox count!")

    print("Actual:", actual[0])
    print("Pred:", predicted[0])

    print("=" * 60)
    print("Bounding Box Detection Evaluation")
    print("=" * 60)
    print(f"Total samples: {len(actual)}")
    print()

    # ----------- MAE / MSE (Vectorized) -----------

    mae_per_sample = np.mean(np.abs(actual - predicted), axis=1)
    mse_per_sample = np.mean((actual - predicted) ** 2, axis=1)

    overall_mae = np.mean(mae_per_sample)
    overall_mse = np.mean(mse_per_sample)
    rmse = np.sqrt(overall_mse)

    # ----------- IoU Calculation -----------

    iou_values = np.array([
        calculate_iou(a, p) for a, p in zip(actual, predicted)
    ])

    mean_iou = np.mean(iou_values)
    iou_50 = np.mean(iou_values >= 0.5)
    iou_75 = np.mean(iou_values >= 0.70)

    # ----------- Print Results -----------

    print("=" * 60)
    print("Overall Metrics")
    print("=" * 60)
    print(f"MAE:   {overall_mae:.4f}")
    print(f"MSE:   {overall_mse:.4f}")
    print(f"RMSE:  {rmse:.4f}")
    print(f"Mean IoU: {mean_iou:.4f}")
    print(f"IoU ≥ 0.50: {iou_50:.4f}")
    print(f"IoU ≥ 0.75: {iou_75:.4f}")
    print("=" * 60)

    return {
        "mae": overall_mae,
        "mse": overall_mse,
        "rmse": rmse,
        "mean_iou": mean_iou,
        "iou_50": iou_50,
        "iou_75": iou_75,
        "mae_per_sample": mae_per_sample,
        "mse_per_sample": mse_per_sample,
        "iou_per_sample": iou_values
    }


if __name__ == "__main__":
    results = evaluate_bbox_detection()
