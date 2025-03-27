#!/usr/bin/env python3
"""
AIVOL - AI-powered Voice-assisted Object Locator
Main Controller Module

This module integrates voice recognition, text processing, object detection,
and detection filtering.
"""

from text.text_processor import ObjectInfo
from vision.object_detection import YOLODetector
from integration.detection_filter import filter_detections
from integration.voice_to_object import extract_object_from_voice
import sys
import os
import logging
import yaml
import cv2
from typing import Dict, Any

# Add parent directory to the path for proper imports
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Import integration modules

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('aivol.log')
    ]
)
logger = logging.getLogger("AIVOL.main")


def load_config() -> Dict[str, Any]:
    """
    Load configuration from config.yaml file.
    """
    try:
        config_path = os.path.join(os.path.dirname(
            os.path.dirname(__file__)), 'config.yaml')
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            logger.info("Configuration loaded successfully.")
            return config
    except Exception as e:
        logger.error("Error loading configuration: %s", e)
        return {}


def initialize_system():
    logger.info(
        "Initializing AI-Powered Voice-Assisted Object Locator (AIVOL)...")
    # Check required directories
    required_dirs = ["models/yolo", "logs"]
    for directory in required_dirs:
        if not os.path.exists(directory):
            os.makedirs(directory)
            logger.info("Created missing directory: %s", directory)
    logger.info("System initialization complete.")


def main():
    logger.info("Starting AIVOL Main Controller...")

    # Load configuration (if needed)
    config = load_config()

    # Initialize system components
    initialize_system()

    # ---- Integration Step 1: Voice-to-Object Extraction ----
    object_info, voice_command = extract_object_from_voice()
    if not voice_command:
        logger.warning(
            "No voice command recognized; using default test query.")
        # Create a default query for testing
        voice_command = "hey can you help me find my wallet it's black in color and it's usually on the dining table"
        object_info = ObjectInfo("wallet")
        object_info.color = "black"
        object_info.location = "on the dining table"

    logger.info("Voice command: %s", voice_command)
    logger.info("Extracted object info: %s", object_info)

    # ---- Integration Step 2: Object Detection on a Sample Image ----
    detector = YOLODetector()  # Loads the pretrained YOLOv5m_Objects365.pt model
    sample_image_path = "tests/sample.png"
    image = cv2.imread(sample_image_path)
    if image is None:
        logger.error("Sample image not found at %s", sample_image_path)
        sys.exit(1)

    detections = detector.detect(image)
    logger.info("Raw detections: %s", detections)

    # ---- Integration Step 3: Filter Detections Based on Query ----
    # Convert object_info to dictionary for filtering purposes.
    query = object_info.to_dict()
    filtered = filter_detections(query, detections)
    logger.info("Filtered detections: %s", filtered)

    # Print the results
    print("Voice Command:", voice_command)
    print("Extracted Object Info:", query)
    print("Filtered Detection Results:", filtered)

    # Optional: Display image with all detections for visual debugging
    for det in detections:
        x1, y1, x2, y2 = map(int, det["bbox"])
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, f"{det['label']} {det['confidence']:.2f}",
                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.imshow("Detections", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    logger.info(
        "Integration test complete. AIVOL is ready for further development!")


if __name__ == "__main__":
    main()
