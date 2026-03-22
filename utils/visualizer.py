import matplotlib.pyplot as plt
import tensorflow as tf


class Visualizer:
    """
    Utility class for visualizing object detection results.
    """

    @staticmethod
    def visualize_detection(image, pred_bbox, actual_bbox=None, digit_class=None):
        """
        Visualize detection results with bounding boxes.

        Args:
            image (tf.Tensor): The image tensor to display.
            pred_bbox (list): Predicted bounding box [x_min, y_min, x_max, y_max] (normalized or relative).
                              The code assumes these are normalized to 0-1 or relative to 128x128.
                              Wait, based on the code: `pred_bbox[0] * 128`, it seems `pred_bbox` is expected to be normalized [0, 1].
            actual_bbox (list, optional): Ground truth bounding box [x_min, y_min, x_max, y_max] (normalized). Defaults to None.
            digit_class (int/str, optional): Detected digit class to display in title. Defaults to None.
        """
        plt.figure(figsize=(10, 10))
        plt.imshow(tf.squeeze(image), cmap="gray")

        # Draw predicted bbox
        rect = plt.Rectangle(
            (pred_bbox[0] * 128, pred_bbox[1] * 128),
            (pred_bbox[2] - pred_bbox[0]) * 128,
            (pred_bbox[3] - pred_bbox[1]) * 128,
            edgecolor='red',
            facecolor='none',
            linewidth=2,
            label="Predicted"
        )
        plt.gca().add_patch(rect)

        # Draw actual bbox if provided
        if actual_bbox is not None:
            rect = plt.Rectangle(
                (actual_bbox[0] * 128, actual_bbox[1] * 128),
                (actual_bbox[2] - actual_bbox[0]) * 128,
                (actual_bbox[3] - actual_bbox[1]) * 128,
                edgecolor='green',
                facecolor='none',
                linewidth=2,
                label="Actual"
            )
            plt.gca().add_patch(rect)

        plt.legend(loc="upper left")
        if digit_class is not None:
            plt.title(f"Detected Digit: {digit_class}")
        plt.axis('off')
        plt.show()
