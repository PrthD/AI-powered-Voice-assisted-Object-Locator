#!/usr/bin/env python3
"""
AIVOL - AI-powered Voice-assisted Object Locator
Merged Wake-Word + Person-Relative Detection

1) Wait for user to say "Hey Assistant."
2) Listen for the object's name (e.g., "Find my wallet.")
3) Detect both "person" and the target object in the same camera frame.
4) Provide directions relative to the person's bounding box (left/right/above/below).
5) Return to idle state after finishing, waiting again for "Hey Assistant."

Use:
  voice_recognition.py (with device_index=2) for capturing audio.
  text_processor.py for parsing the object from the user’s command.
  object_detection.py + detection_filter.py for YOLO + filtering.
"""

import sys
import os
import logging
import time
import cv2
import yaml
from typing import Dict, Any, List, Optional

# ----------------------------------------------------------------------
# Imports for voice & detection
# ----------------------------------------------------------------------
# Make sure these align with your project structure
from audio.voice_recognition import get_voice_command
from text.text_processor import process_user_prompt, ObjectInfo
from vision.object_detection import YOLODetector
from integration.detection_filter import filter_detections

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('aivol_person_relative.log')
    ]
)
logger = logging.getLogger("AIVOL.main")

# ----------------------------------------------------------------------
# Configuration / Constants
# ----------------------------------------------------------------------
WAKE_WORD = "hey assistant"
WAKE_WORD_ALTS = {WAKE_WORD, "hey assistant."}  # handle punctuation
WAKE_TIMEOUT = 5
WAKE_PHRASE_LIMIT = 2

CMD_TIMEOUT = 5
CMD_PHRASE_LIMIT = 30


def load_config() -> Dict[str, Any]:
    """
    Load configuration from config.yaml if available. Otherwise defaults.
    """
    base_dir = os.path.dirname(__file__)
    config_path = os.path.join(base_dir, "config.yaml")
    config = {}
    if os.path.isfile(config_path):
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                logger.info("Configuration loaded from %s", config_path)
        except Exception as e:
            logger.error("Error loading configuration: %s", e)
    else:
        logger.warning("No config.yaml file found. Using defaults.")
    return config


def initialize_system():
    """
    Create needed directories & basic setup.
    """
    logger.info("Initializing AIVOL system (Wake Word + Person-Relative).")
    for directory in ["models/yolo", "logs"]:
        if not os.path.exists(directory):
            os.makedirs(directory)
            logger.info("Created missing directory: %s", directory)
    logger.info("System initialization complete.")


# ----------------------------------------------------------------------
# Wake Word Listening
# ----------------------------------------------------------------------
def listen_for_wake_word() -> bool:
    """
    Continuously do short captures to detect the wake word. 
    Return True once recognized, or False if user interrupts.
    """
    logger.info("[Idle] Listening for wake word: '%s'", WAKE_WORD)
    while True:
        try:
            phrase = get_voice_command(
                timeout=WAKE_TIMEOUT,
                phrase_time_limit=WAKE_PHRASE_LIMIT,
                ambient_noise_duration=0.5
            )
            if phrase is None:
                logger.info("No wake word detected this round. Retrying...")
                continue

            phrase = phrase.lower().strip()
            logger.info("Heard: '%s'", phrase)
            if phrase in WAKE_WORD_ALTS:
                logger.info("Wake word detected! Proceeding...")
                return True
            else:
                logger.info("Not the wake word. Listening again.")
        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt. Exiting wake word loop.")
            return False


# ----------------------------------------------------------------------
# Command Listening
# ----------------------------------------------------------------------
def listen_for_command() -> Optional[str]:
    """
    After hearing wake word, capture a longer command from user.
    E.g., "Find my black wallet."
    """
    logger.info("[Active] Listening for user command (long capture).")
    command = get_voice_command(
        timeout=CMD_TIMEOUT,
        phrase_time_limit=CMD_PHRASE_LIMIT,
        ambient_noise_duration=0.5
    )
    if command:
        logger.info("User command recognized: %s", command)
        return command.lower().strip()
    else:
        logger.warning("No user command recognized (timed out or error).")
        return None


# ----------------------------------------------------------------------
# Person-Relative Logic
# ----------------------------------------------------------------------
def find_best_person(detections: List[dict]) -> dict:
    """
    Among YOLO detections, pick bounding box for label='person'
    with largest bounding box area or highest confidence.
    Returns {} if no person found.
    """
    person_detections = [d for d in detections if d["label"].lower() == "person"]
    if not person_detections:
        return {}

    # Option B: largest bounding box area
    def bbox_area(d):
        x1, y1, x2, y2 = d["bbox"]
        return (x2 - x1) * (y2 - y1)

    best_person = max(person_detections, key=bbox_area)
    return best_person


def get_direction_string(
    obj_center_x: float,
    obj_center_y: float,
    ref_center_x: float,
    ref_center_y: float,
    object_area: float = 0.0
) -> str:
    """
    Return approximate direction + distance from person center to object center.
    Negative offset_x => left, positive => right;
    Negative offset_y => above, positive => below.
    Also uses bounding box area to guess distance.
    """
    offset_x = obj_center_x - ref_center_x
    offset_y = obj_center_y - ref_center_y

    # Horizontal direction
    horizontal_dir = ""
    abs_x = abs(offset_x)
    if abs_x < 30:
        horizontal_dir = "center (same horizontal)"
    elif abs_x < 100:
        horizontal_dir = "slightly left" if offset_x < 0 else "slightly right"
    else:
        horizontal_dir = "far left" if offset_x < 0 else "far right"

    # Vertical direction
    vertical_dir = ""
    abs_y = abs(offset_y)
    if abs_y < 30:
        vertical_dir = "same height"
    elif abs_y < 100:
        vertical_dir = "slightly above" if offset_y < 0 else "slightly below"
    else:
        vertical_dir = "far above" if offset_y < 0 else "far below"

    # Combine
    if "center (same horizontal)" in horizontal_dir and "same height" in vertical_dir:
        direction_str = "in front of you"
    else:
        direction_str = f"{horizontal_dir} and {vertical_dir}"

    # Distance from bounding box area
    # Tweak thresholds to your environment
    if object_area > 120000:
        distance_str = "very close"
    elif object_area > 40000:
        distance_str = "close"
    elif object_area > 10000:
        distance_str = "medium distance"
    else:
        distance_str = "far away"

    if direction_str == "in front of you":
        return f"{direction_str}, appears {distance_str}"
    else:
        return f"{direction_str}, appears {distance_str}"


# ----------------------------------------------------------------------
# Detection Loop
# ----------------------------------------------------------------------
def run_person_relative_detection(object_name: str, config: Dict[str, Any]):
    """
    Opens camera, detects both 'person' and target object. 
    Provides directions relative to person's bounding box center.
    """
    logger.info("User wants to find: '%s' with person-relative logic.", object_name)

    model_path = config.get("vision", {}).get("model_path", "models/yolo/yolov5m_Objects365.pt")
    conf_thr = config.get("vision", {}).get("confidence_threshold", 0.25)
    iou_thr = config.get("vision", {}).get("nms_threshold", 0.45)

    detector = YOLODetector(weights_path=model_path, conf_threshold=conf_thr, iou_threshold=iou_thr)
    logger.info("YOLOv5 model loaded. Attempting to detect person + '%s'...", object_name)

    camera_index = config.get("device", {}).get("camera_index", 0)
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        logger.error("Could not open camera (index=%s). Exiting detection loop.", camera_index)
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    frame_count = 0
    logger.info("Press 'Q' to quit this detection session.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.warning("Failed to read frame from camera. Breaking loop.")
                break

            frame_count += 1
            # Skip frames for performance
            if frame_count % 2 != 0:
                cv2.imshow("AIVOL - Person Relative", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

            detections = detector.detect(frame)
            if not detections:
                cv2.putText(frame, "No detections", (30,30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)
                cv2.imshow("AIVOL - Person Relative", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

            # Identify the best person
            best_person = find_best_person(detections)

            # Filter for the target object
            filtered = filter_detections({"name": object_name}, detections)

            direction_text = ""
            if best_person:
                # Draw person's bounding box
                x1p, y1p, x2p, y2p = map(int, best_person["bbox"])
                cv2.rectangle(frame, (x1p, y1p), (x2p, y2p), (255, 0, 0), 2)
                cv2.putText(frame, "Person",
                            (x1p, y1p - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            else:
                # No person => fallback or warn
                cv2.putText(frame, "No person found in frame", (30,30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

            if filtered:
                obj_det = filtered[0]
                x1o, y1o, x2o, y2o = map(int, obj_det["bbox"])
                cv2.rectangle(frame, (x1o, y1o), (x2o, y2o), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"{obj_det['label']} {obj_det['confidence']:.2f}",
                    (x1o, y1o - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

                # Person center or fallback to frame center
                if best_person:
                    px_center = (x1p + x2p) / 2.0
                    py_center = (y1p + y2p) / 2.0
                else:
                    frame_h, frame_w = frame.shape[:2]
                    px_center = frame_w / 2.0
                    py_center = frame_h / 2.0

                ox_center = (x1o + x2o) / 2.0
                oy_center = (y1o + y2o) / 2.0
                obj_area = (x2o - x1o) * (y2o - y1o)

                direction_text = get_direction_string(ox_center, oy_center, px_center, py_center, obj_area)
                logger.info("Object direction: %s", direction_text)

                cv2.putText(
                    frame,
                    direction_text,
                    (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2
                )
            else:
                cv2.putText(
                    frame,
                    f"Target '{object_name}' not found",
                    (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2
                )

            cv2.imshow("AIVOL - Person Relative", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                logger.info("User pressed 'q'. Exiting detection.")
                break

    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt in detection loop.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        logger.info("Person-relative detection session ended.")


# ----------------------------------------------------------------------
# Main Loop (Wake Word => Command => Person-Relative Detection)
# ----------------------------------------------------------------------
def main():
    logger.info("Starting AIVOL Main (Wake Word + Person-Relative).")

    config = load_config()
    initialize_system()

    while True:
        # 1) Wait for "Hey Assistant"
        woke = listen_for_wake_word()
        if not woke:
            logger.info("Exiting main loop. (KeyboardInterrupt or error in wake word).")
            break

        # 2) Listen for user command (e.g., "Find my cup")
        user_cmd = listen_for_command()
        if not user_cmd:
            logger.info("No user command recognized; returning to idle.")
            continue

        # 3) Parse object from user command
        obj_info = process_user_prompt(user_cmd)
        if not obj_info.name:
            logger.warning("Could not parse an object from command '%s'.", user_cmd)
            continue

        # 4) Person-relative detection
        run_person_relative_detection(obj_info.name, config)

        # 5) Return to idle (wake word listening)
        logger.info("Finished detection. Returning to wake word state...")

    logger.info("AIVOL system shutting down.")


if __name__ == "__main__":
    main()
