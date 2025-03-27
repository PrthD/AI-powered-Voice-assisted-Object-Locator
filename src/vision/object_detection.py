import cv2
import numpy as np
import torch
import logging
from typing import List, Dict, Any

logger = logging.getLogger("AIVOL.vision.object_detection")


class YOLODetector:
    def __init__(self,
                 weights_path: str = "models/yolo/yolov5m_Objects365.pt",
                 conf_threshold: float = 0.25,
                 iou_threshold: float = 0.45):
        """
        Initialize the YOLOv5 detector by loading the pretrained model from Objects365.

        Args:
            weights_path (str): Path to the YOLOv5m Objects365 pretrained weights.
            conf_threshold (float): Confidence threshold for detections.
            iou_threshold (float): IOU threshold for non-max suppression.
        """
        logger.info("Loading YOLOv5 model from: %s", weights_path)
        # Load the custom YOLOv5 model via torch.hub
        self.model = torch.hub.load(
            'ultralytics/yolov5', 'custom', path=weights_path, force_reload=False)
        # Set model hyperparameters
        self.model.conf = conf_threshold
        self.model.iou = iou_threshold
        # The model.names attribute is a dictionary mapping class indices to names.
        self.class_names = self.model.names
        logger.info("Loaded model with %d classes", len(self.class_names))

    def get_class_names(self) -> List[str]:
        """
        Get the list of class names from the model.

        Returns:
            List[str]: List of object class names.
        """
        return list(self.class_names.values())

    def detect(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect objects in the provided image.

        Args:
            image (np.ndarray): The input image (BGR format).

        Returns:
            List[Dict[str, Any]]: List of detections, where each detection is a dictionary with:
                - "label": Class label,
                - "confidence": Confidence score,
                - "bbox": Bounding box coordinates [x1, y1, x2, y2].
        """
        results = self.model(image)
        detections = []
        # results.xyxy[0] is a tensor with columns: [x1, y1, x2, y2, confidence, class]
        for *box, conf, cls in results.xyxy[0].tolist():
            x1, y1, x2, y2 = box
            class_idx = int(cls)
            label = self.class_names[class_idx]
            detection = {
                "label": label,
                "confidence": conf,
                "bbox": [x1, y1, x2, y2]
            }
            detections.append(detection)
        return detections


if __name__ == "__main__":
    # For incremental testing: load the model, print class names,
    # and run detection on a sample static image.
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    detector = YOLODetector()
    logger.info("Class names: %s", detector.get_class_names())

    sample_image_path = "tests/sample1.png"
    image = cv2.imread(sample_image_path)
    if image is None:
        logger.error("Sample image not found at %s", sample_image_path)
    else:
        detections = detector.detect(image)
        logger.info("Detections: %s", detections)
        # Draw detections on the image
        for detection in detections:
            x1, y1, x2, y2 = map(int, detection["bbox"])
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, f"{detection['label']} {detection['confidence']:.2f}",
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.imshow("Detections", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
