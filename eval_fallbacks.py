import json
import os
from evaluate_bbox import calculate_iou


class Evaluator:

    def __init__(self, log_file, images_dir, gt_bbox_dir):
        self.log_file = log_file
        self.images_dir = images_dir
        self.gt_bbox_dir = gt_bbox_dir

        self.fallback_entries = []
        self.cnn_entries = []

        self.cnn_iou_results = {}
        self.fallback_iou_results = {}

    def load_fallback_cases(self):
        with open(self.log_file, "r") as f:
            logs = json.load(f)["results"]

        self.fallback_entries = [
            e for e in logs if e["bbox_source"].lower() == "classical"
        ]

        print(f"Found {len(self.fallback_entries)} fallback cases")

    def load_cnn_entries(self, limit=None):
        with open(self.log_file, "r") as f:
            logs = json.load(f)["results"]

        cnn = [e for e in logs if e["bbox_source"].lower() == "cnn"]
        self.cnn_entries = cnn[:limit] if limit else cnn

        print(f"Found {len(self.cnn_entries)} cnn cases")

    def _load_gt_bbox(self, image_id):
        path = os.path.join(self.gt_bbox_dir, f"{image_id}.txt")
        with open(path, "r") as f:
            return list(map(float, f.read().strip().split()))

    def analyze_cnn_entries(self):
        for entry in self.cnn_entries:
            image_id = entry["image_id"]

            cnn_bbox = entry["bbox"]
            gt_bbox = self._load_gt_bbox(image_id)

            self.cnn_iou_results[image_id] = {
                "iou_cnn_gt": calculate_iou(cnn_bbox, gt_bbox)
            }

    def summarize_cnn(self):
        ious = [v["iou_cnn_gt"] for v in self.cnn_iou_results.values()]

        return {
            "mean_iou_cnn_gt": sum(ious) / len(ious) if ious else 0.0,
            "count": len(ious)
        }

    def analyze_fallback_entries(self):
        for entry in self.fallback_entries:
            image_id = entry["image_id"]

            classical_bbox = entry["bbox"]
            gt_bbox = self._load_gt_bbox(image_id)

            self.fallback_iou_results[image_id] = {
                "iou_classical_gt": calculate_iou(classical_bbox, gt_bbox)
            }

    def summarize_fallback(self):
        ious = [v["iou_classical_gt"] for v in self.fallback_iou_results.values()]

        return {
            "mean_iou_classical_gt": sum(ious) / len(ious) if ious else 0.0,
            "count": len(ious)
        }



if __name__ == "__main__":

    LOG_FILE = "./logs/run_01.json"
    IMAGES_DIR = "./TestImages"
    BBOXES_DIR = "./Testlabel"

    evaluator = Evaluator(
        log_file=LOG_FILE,
        images_dir=IMAGES_DIR,
        gt_bbox_dir=BBOXES_DIR
    )

    evaluator.load_cnn_entries(limit=280)
    evaluator.analyze_cnn_entries()
    cnn_summary = evaluator.summarize_cnn()

    print("\nCNN SUCCESS ANALYSIS")
    print("===================")
    for k, v in cnn_summary.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")

    evaluator.load_fallback_cases()
    evaluator.analyze_fallback_entries()
    fallback_summary = evaluator.summarize_fallback()

    print("\nFALLBACK ANALYSIS")
    print("================")
    for k, v in fallback_summary.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")
