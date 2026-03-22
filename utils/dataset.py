import os
import tensorflow as tf
from utils.reader import FileReader as FR


class DatasetBuilder:
    """
    Utility class for building TensorFlow tensor datasets
    from image and label directories.
    """

    @staticmethod
    def load_bbox_dataset(
        images_dir,
        labels_dir,
        image_ext="png",
        image_shape=(128, 128),
        normalize_bbox=False,
        limit=None
    ):
        """
        Load images and bounding box labels as TensorFlow tensors.

        Reads all matching image/label pairs from the given directories
        and returns stacked tensors ready for model training or evaluation.

        Args:
            images_dir (str): Path to the directory containing images.
            labels_dir (str): Path to the directory containing bbox label files (.txt).
            image_ext (str): Image file extension. Defaults to "png".
            image_shape (tuple): Target image size (H, W). Defaults to (128, 128).
            normalize_bbox (bool): If True, normalize bbox coords to [0, 1]
                                   relative to image_shape. Defaults to False.
            limit (int, optional): Max number of samples to load. Defaults to None.

        Returns:
            tuple: (images_tensor, bboxes_tensor)
                - images_tensor: tf.Tensor of shape (N, H, W, 1), float32
                - bboxes_tensor: tf.Tensor of shape (N, 4), float32
        """
        image_files = FR.read_files(images_dir, image_ext, limit=limit)
        if image_files is None:
            raise FileNotFoundError(f"Images directory not found: {images_dir}")

        images = []
        bboxes = []

        for img_file in image_files:
            name = os.path.splitext(img_file)[0]
            label_path = os.path.join(labels_dir, f"{name}.txt")

            if not os.path.exists(label_path):
                continue

            image = FR.read_image(os.path.join(images_dir, img_file))
            bbox = FR.read_label(label_path)

            images.append(image)
            bboxes.append(bbox)

        if len(images) == 0:
            raise ValueError("No matching image-label pairs found.")

        images_tensor = tf.stack(images)            # (N, H, W, 1)
        bboxes_tensor = tf.constant(bboxes, dtype=tf.float32)  # (N, 4)

        if normalize_bbox:
            h, w = image_shape
            scale = tf.constant([w, h, w, h], dtype=tf.float32)
            bboxes_tensor = bboxes_tensor / scale

        return images_tensor, bboxes_tensor

    @staticmethod
    def load_classification_dataset(
        images_dir,
        class_dir,
        image_ext="png",
        num_classes=10,
        one_hot=True,
        limit=None
    ):
        """
        Load images and class labels as TensorFlow tensors.

        Args:
            images_dir (str): Path to the directory containing images.
            class_dir (str): Path to the directory containing class label files (.txt).
            image_ext (str): Image file extension. Defaults to "png".
            num_classes (int): Number of classes for one-hot encoding. Defaults to 10.
            one_hot (bool): If True, return one-hot encoded labels. Defaults to True.
            limit (int, optional): Max number of samples to load. Defaults to None.

        Returns:
            tuple: (images_tensor, labels_tensor)
                - images_tensor: tf.Tensor of shape (N, H, W, 1), float32
                - labels_tensor: tf.Tensor of shape (N, num_classes) if one_hot,
                                 else (N,) int32
        """
        image_files = FR.read_files(images_dir, image_ext, limit=limit)
        if image_files is None:
            raise FileNotFoundError(f"Images directory not found: {images_dir}")

        images = []
        labels = []

        for img_file in image_files:
            name = os.path.splitext(img_file)[0]
            class_path = os.path.join(class_dir, f"{name}.txt")

            if not os.path.exists(class_path):
                continue

            image = FR.read_image(os.path.join(images_dir, img_file))
            with open(class_path, "r") as f:
                label = int(f.read().strip())

            images.append(image)
            labels.append(label)

        if len(images) == 0:
            raise ValueError("No matching image-label pairs found.")

        images_tensor = tf.stack(images)
        labels_tensor = tf.constant(labels, dtype=tf.int32)

        if one_hot:
            labels_tensor = tf.one_hot(labels_tensor, depth=num_classes)

        return images_tensor, labels_tensor

    @staticmethod
    def as_tf_dataset(images, labels, batch_size=32, shuffle=True, buffer_size=1000):
        """
        Wrap tensor arrays into a batched tf.data.Dataset.

        Args:
            images (tf.Tensor): Images tensor.
            labels (tf.Tensor): Labels tensor.
            batch_size (int): Batch size. Defaults to 32.
            shuffle (bool): Whether to shuffle the dataset. Defaults to True.
            buffer_size (int): Shuffle buffer size. Defaults to 1000.

        Returns:
            tf.data.Dataset: Batched (and optionally shuffled) dataset.
        """
        dataset = tf.data.Dataset.from_tensor_slices((images, labels))

        if shuffle:
            dataset = dataset.shuffle(buffer_size=buffer_size)

        dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

        return dataset
