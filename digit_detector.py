import numpy as np
import tensorflow as tf
from scipy.ndimage import label
import json
from datetime import datetime


class DigitDetectionPipeline:
    """
    Pipeline for end-to-end digit detection and recognition.

    Integrates a bounding box detection model (or classical fallback)
    with a digit classification model.
    """

    def __init__(
        self,
        bbox_model=None,
        classifier_model=None,
        image_shape=(128, 128),
        use_classical_fallback=True,
        normalize_bbox=False
    ):
        """
        Initialize the pipeline.

        Args:
            bbox_model (tf.keras.Model, optional): Trained model for bbox regression. Defaults to None.
            classifier_model (tf.keras.Model, optional): Trained model for digit classification. Defaults to None.
            image_shape (tuple, optional): Input image shape. Defaults to (128, 128).
            use_classical_fallback (bool, optional): Whether to use Connected Components if CNN fails/is absent. Defaults to True.
            normalize_bbox (bool, optional): If True, denormalize [0,1] bbox coords from CNN to pixel space. Defaults to False.
        """
        self.bbox_model = bbox_model
        self.classifier_model = classifier_model
        self.image_shape = image_shape
        self.use_classical_fallback = use_classical_fallback
        self.normalize_bbox = normalize_bbox

    @staticmethod
    def find_digit_bbox(img, threshold=0.15, margin=2):
        """
        Find bounding box using connected components (Classical Fallback).

        Args:
            img (tf.Tensor | np.ndarray): Input image.
            threshold (float): Binarization threshold.
            margin (int): Margin around bbox.

        Returns:
            list: [x_min, y_min, x_max, y_max] or None.
        """
        img = img.numpy().squeeze()

        binary = img > threshold
        labels, num = label(binary)

        if num == 0:
            return None

        largest_label = max(
            range(1, num + 1),
            key=lambda i: np.sum(labels == i)
        )

        ys, xs = np.where(labels == largest_label)

        y_min, y_max = ys.min(), ys.max()
        x_min, x_max = xs.min(), xs.max()

        y_min = max(0, y_min - margin)
        x_min = max(0, x_min - margin)
        y_max = min(img.shape[0] - 1, y_max + margin)
        x_max = min(img.shape[1] - 1, x_max + margin)

        return [x_min, y_min, x_max, y_max]

    def is_valid_bbox(self, bbox):
        """
        Validate if a bounding box is reasonable.
        
        Checks for:
        - Non-null
        - Valid coordinates (not NaN/Inf)
        - Positive area
        - Within image boundaries
        - Not excessively large (covering > 90% of image)

        Args:
            bbox (list): [x_min, y_min, x_max, y_max]

        Returns:
            bool: True if valid.
        """
        if bbox is None:
            return False

        bbox = np.asarray(bbox, dtype=float)

        if np.any(np.isnan(bbox)) or np.any(np.isinf(bbox)):
            return False

        x_min, y_min, x_max, y_max = bbox

        if x_max <= x_min or y_max <= y_min:
            return False

        if x_min < 0 or y_min < 0:
            return False

        if x_max > self.image_shape[1] or y_max > self.image_shape[0]:
            return False

        area = (x_max - x_min) * (y_max - y_min)
        if area > 0.9 * self.image_shape[0] * self.image_shape[1]:
            return False

        return True

    def get_bbox(self, image):
        """
        Get bounding box for an image using available methods.
        
        Tries Bbox CNN first. If prediction is invalid and fallback is enabled,
        uses classical method.

        Args:
            image (tf.Tensor): Input image.

        Returns:
            list | None: Bounding box coordinates.
        """
        self.last_bbox_source = None
        cnn_bbox = None

        if self.bbox_model is not None:
            cnn_bbox = self.bbox_model.predict(
                tf.expand_dims(image, axis=0),
                verbose=0
            )[0]

            if self.normalize_bbox and cnn_bbox is not None:
                h, w = self.image_shape
                cnn_bbox = cnn_bbox * np.array([w, h, w, h], dtype=float)

        if self.is_valid_bbox(cnn_bbox):
            self.last_bbox_source = "cnn"
            return cnn_bbox

        if self.use_classical_fallback:
            self.last_bbox_source = "classical"
            return self.find_digit_bbox(image)

        return None

    def _ensure_2d_image(self, image):
        """
        Converts image to shape (H, W) regardless of input format.
        """
        if isinstance(image, tf.Tensor):
            image = image.numpy()

        image = np.asarray(image)

        if image.ndim == 4:
            image = image[0]

        if image.ndim == 3:
            image = image.squeeze(-1)

        return image

    def _crop_and_resize(self, image, bbox, size=(28, 28)):
        """
        Crop image to bbox and resize to target size for classification.

        Args:
            image (tf.Tensor): Input image.
            bbox (list): Bounding box.
            size (tuple): Target size for classifier input.

        Returns:
            tf.Tensor: Cropped and resized image, or None if invalid crop.
        """
        image = self._ensure_2d_image(image)

        x_min, y_min, x_max, y_max = [int(round(v)) for v in bbox]

        x_min = max(0, min(x_min, image.shape[1] - 1))
        x_max = max(0, min(x_max, image.shape[1]))
        y_min = max(0, min(y_min, image.shape[0] - 1))
        y_max = max(0, min(y_max, image.shape[0]))

        if x_max <= x_min or y_max <= y_min:
            return None

        crop = image[y_min:y_max, x_min:x_max]

        if crop.size == 0:
            return None

        crop = tf.expand_dims(crop, axis=0)
        crop = tf.expand_dims(crop, axis=-1)

        crop = tf.image.resize(crop, size)

        return crop[0]

    def init_logger(self, log_path="logs/experiments.json"):
        """Initialize result logger"""
        self.log_path = log_path
        self.logs = []

    def _log_result(self, image_id, result, bbox_source):
        """Log a single prediction result"""
        log_entry = {
            "image_id": image_id,
            "bbox_source": bbox_source,
            "bbox": [float(v) for v in result["bbox"]],
            "digit": result["digit"],
            "confidence": result["confidence"],
        }

        self.logs.append(log_entry)

    def save_logs(self):
        """Save populated logs to JSON file"""
        import os
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

        with open(self.log_path, "w") as f:

            json.dump(
                {
                    "created_at": datetime.now().isoformat(),
                    "num_samples": len(self.logs),
                    "results": self.logs
                },
                f,
                indent=2
            )


    def predict(self, image, image_id=None):
        """
        Run full pipeline: Detect Box -> Crop -> Classify.

        Args:
            image (tf.Tensor): Input image.
            image_id (str, optional): Identifier for logging.

        Returns:
            dict: { "bbox": ..., "digit": ..., "confidence": ... }
        """
        bbox = self.get_bbox(image)
        if bbox is None:
            return None

        crop = self._crop_and_resize(image, bbox)
        if crop is None:
            return None

        pred = self.classifier_model.predict(
            tf.expand_dims(crop, axis=0),
            verbose=0
        )[0]

        result = {
            "bbox": bbox,
            "digit": int(tf.argmax(pred)),
            "confidence": float(tf.reduce_max(pred))
        }

        if image_id is not None and hasattr(self, "logs"):
            self._log_result(image_id, result, self.last_bbox_source)

        return result

    def predict_batch(self, images, batch_size=32):
        """
        Run the full pipeline on a batch of images efficiently.

        Instead of calling predict() per image (N model calls each),
        this batches the bbox and classifier predictions into single calls.

        Args:
            images (tf.Tensor): Batch of images, shape (N, H, W, 1).
            batch_size (int): Batch size for model.predict(). Defaults to 32.

        Returns:
            list[dict | None]: List of prediction dicts (or None for failed detections).
                Each dict: { "bbox": [...], "digit": int, "confidence": float, "bbox_source": str }
        """
        n = len(images)
        results = [None] * n

        # --- Step 1: Batch bbox prediction ---
        bboxes = [None] * n
        bbox_sources = [""] * n

        if self.bbox_model is not None:
            raw_bboxes = self.bbox_model.predict(images, batch_size=batch_size, verbose=0)

            for i in range(n):
                bbox = raw_bboxes[i]

                if self.normalize_bbox:
                    h, w = self.image_shape
                    bbox = bbox * np.array([w, h, w, h], dtype=float)

                if self.is_valid_bbox(bbox):
                    bboxes[i] = bbox
                    bbox_sources[i] = "cnn"

        # Fallback for images where CNN bbox was invalid or model is absent
        if self.use_classical_fallback:
            for i in range(n):
                if bboxes[i] is None:
                    fb_bbox = self.find_digit_bbox(images[i])
                    if fb_bbox is not None:
                        bboxes[i] = fb_bbox
                        bbox_sources[i] = "classical"

        # --- Step 2: Crop and resize all valid detections ---
        crops = []
        crop_indices = []  # track which original index each crop maps to

        for i in range(n):
            if bboxes[i] is None:
                continue

            crop = self._crop_and_resize(images[i], bboxes[i])
            if crop is not None:
                crops.append(crop)
                crop_indices.append(i)

        if len(crops) == 0:
            return results

        crops_tensor = tf.stack(crops)  # (M, 28, 28, 1)

        # --- Step 3: Batch classification ---
        preds = self.classifier_model.predict(
            crops_tensor, batch_size=batch_size, verbose=0
        )

        for j, i in enumerate(crop_indices):
            pred = preds[j]
            results[i] = {
                "bbox": bboxes[i] if isinstance(bboxes[i], list) else bboxes[i].tolist(),
                "digit": int(tf.argmax(pred)),
                "confidence": float(tf.reduce_max(pred)),
                "bbox_source": bbox_sources[i]
            }

            if hasattr(self, "logs"):
                self._log_result(str(i), results[i], bbox_sources[i])

        return results


# Currently unused
def log_prediction(method):
    """Decorator to log prediction details if logging is initialized"""
    def wrapper(self, image, image_id=None, *args, **kwargs):
        result = method(self, image, image_id=image_id, *args, **kwargs)

        if (
            result is not None and
            image_id is not None and
            hasattr(self, "logs")
        ):
            self._log_result(
                image_id=image_id,
                result=result,
                bbox_source=self.last_bbox_source
            )

        return result
    return wrapper


if __name__=="__main__":

    from utils.reader import FileReader as FR
    from os import path

    bbox_model = tf.keras.models.load_model("bbox_model.keras")
    classifier_model = tf.keras.models.load_model("Models/DigitRecog.h5")
    pipeline = DigitDetectionPipeline(
        bbox_model=bbox_model,
        classifier_model=classifier_model,
        normalize_bbox=True
    )
    pipeline.init_logger("logs/run_03.json")
    predictions = {}
    for img in FR.read_files("TestImages", "png"):
        predictions[img.split(".")[0]] = pipeline.predict(FR.read_image(path.join("TestImages", img)), img.split(".")[0])
    # print(predictions)
    pipeline.save_logs()
