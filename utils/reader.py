import os
import tensorflow as tf

class FileReader:
    """
    Utility class for reading files and processing images/labels.
    """

    @staticmethod
    def read_files(path, extension, limit=None):
        """
        Read files with specific extension from directory.

        Args:
            path (str): Directory path to search in.
            extension (str): File extension to filter by (e.g., 'png', 'txt').
            limit (int, optional): Maximum number of files to return. Defaults to None.

        Returns:
            list: List of filenames matching the extension, or None if path doesn't exist.
        """
        if os.path.exists(path):
            files = [f for f in os.listdir(path) if f.endswith(extension)]
            if limit:
                files = files[:limit]
            return files
        return None

    @staticmethod
    def read_label(label_path):
        """
        Read and process label file containing bounding box coordinates.

        Args:
            label_path (str): Path to the label text file.

        Returns:
            list: [x_min, y_min, x_max, y_max] coordinates.
        """
        with open(label_path, 'r') as file:
            lines = file.readlines()
            x_min, y_min, x_max, y_max = list(map(int, lines[0].strip().split()))
            return [x_min, y_min, x_max, y_max]

    @staticmethod
    def read_image(image_path):
        """
        Read and preprocess image for the model.
        
        Reads the image, decodes it as grayscale, resizes to 128x128, 
        and normalizes pixel values to [0, 1].

        Args:
            image_path (str): Path to the image file.

        Returns:
            tf.Tensor: Preprocessed image tensor of shape (128, 128, 1).
        """
        image = tf.io.read_file(image_path)
        image = tf.image.decode_png(image, channels=1)
        image = tf.image.resize(image, [128, 128])
        image = tf.cast(image, tf.float32) / 255.0
        return image

