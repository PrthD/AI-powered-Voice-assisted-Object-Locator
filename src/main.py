#!/usr/bin/env python3
"""
Threaded approach:
1) One background thread constantly listens for "Hey Assistant."
2) Main thread does detection. If user says "Hey Assistant," we set a shared interrupt flag.
3) 30s timeout if object not found, speak "not found," return to idle.
4) Speak "I'm listening" before capturing user object command.

Assumes you have:
- audio.voice_recognition -> get_voice_command
- audio.text_to_speech_gTTS -> GTTSSpeaker
- text.text_processor -> process_user_prompt
- vision.object_detection -> YOLODetector
- integration.detection_filter -> filter_detections
- vision.mediapipe_tracker -> MediaPipeTracker
"""

import sys
import os
import logging
import time
import math
import cv2
import yaml
from typing import Dict, Any, Optional, List
import threading

# Audio input
from audio.voice_recognition import get_voice_command
# gTTS TTS
from audio.text_to_speech_gTTS import GTTSSpeaker
# text processing
from text.text_processor import process_user_prompt
# YOLO detection
from vision.object_detection import YOLODetector
# detection filter
from integration.detection_filter import filter_detections
# mediapipe
from vision.mediapipe_tracker import MediaPipeTracker

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('aivol_threaded.log')
    ]
)
logger = logging.getLogger("AIVOL.main")

# Wake-word constants
WAKE_WORD = "hey assistant"
WAKE_WORD_ALTS = {WAKE_WORD, "hey assistant."}
WAKE_TIMEOUT = 5
WAKE_PHRASE_LIMIT = 2

# Command capture
CMD_TIMEOUT = 5
CMD_PHRASE_LIMIT = 30

OBJECT_FIND_TIMEOUT_SECS = 20.0  # If object not found, speak and exit

class SharedState:
    """
    Holds shared flags/data that both the main thread (detection) and
    the wake-word listener thread can access.
    """
    def __init__(self):
        # If True, detection should stop
        self.interrupt_detection = False
        # If True, we want to exit the entire program
        self.exit_program = False

def load_config() -> Dict[str, Any]:
    base_dir = os.path.dirname(__file__)
    config_path = os.path.join(base_dir, "config.yaml")
    config = {}
    if os.path.isfile(config_path):
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                logger.info("Configuration loaded from %s", config_path)
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
    else:
        logger.warning("No config.yaml file found. Using defaults.")
    return config

def initialize_system():
    logger.info("Initializing AIVOL system (Threaded, YOLO, MediaPipe, gTTS).")
    for d in ["models/yolo", "logs"]:
        if not os.path.exists(d):
            os.makedirs(d)
            logger.info("Created missing directory: %s", d)
    logger.info("System initialization complete.")

def wake_word_listener(state: SharedState, speaker: GTTSSpeaker):
    """
    Background thread that constantly listens for short segments to detect "Hey Assistant."
    If recognized, set state.interrupt_detection = True
    If the main loop sets state.exit_program = True, we stop.
    """
    logger.info("WakeWordListener thread started.")
    while not state.exit_program:
        try:
            phrase = get_voice_command(
                timeout=1,             # short wait
                phrase_time_limit=10,  # short phrase
                ambient_noise_duration=0.3
            )
            if phrase:
                phrase_low = phrase.lower().strip()
                if phrase_low in WAKE_WORD_ALTS:
                    logger.info("Wake word re-detected in background thread. Setting interrupt.")
                    state.interrupt_detection = True
        except:
            # If there's any error or waitTimeout
            pass

        # small sleep so we don't spin CPU if user isn't speaking
        time.sleep(0.1)
    logger.info("WakeWordListener thread stopping.")

def get_user_command(speaker: GTTSSpeaker) -> Optional[str]:
    """
    Speak "I'm listening," then do a longer capture for the user command.
    """
    if speaker:
        speaker.speak("I'm listening now. Please tell me the object you want me to find.")
    command = get_voice_command(
        timeout=CMD_TIMEOUT,
        phrase_time_limit=CMD_PHRASE_LIMIT,
        ambient_noise_duration=0.5
    )
    if command:
        return command.lower().strip()
    return None

def choose_nearest_object(detections: List[dict], user_cx: float, user_cy: float) -> dict:
    best_det = {}
    best_dist = float('inf')
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        objcx = (x1 + x2)/2
        objcy = (y1 + y2)/2
        dist = math.hypot(objcx - user_cx, objcy - user_cy)
        if dist < best_dist:
            best_dist = dist
            best_det = det
    return best_det

def generate_feedback(obj_cx, obj_cy, user_cx, user_cy, left_wrist, right_wrist, object_area):
    offset_x = obj_cx - user_cx
    offset_y = obj_cy - user_cy

    # horizontal
    abs_x = abs(offset_x)
    if abs_x < 30:
        horiz = "center"
    elif abs_x < 100:
        horiz = "a bit to your right" if offset_x < 0 else "a bit to your left"
    else:
        horiz = "far right" if offset_x < 0 else "far left"

    # vertical
    abs_y = abs(offset_y)
    if abs_y < 30:
        vert = "same level"
    elif abs_y < 100:
        vert = "slightly above" if offset_y < 0 else "slightly below"
    else:
        vert = "far above" if offset_y < 0 else "far below"

    if horiz == "center" and vert == "same level":
        dir_str = "directly in front of you"
    else:
        dir_str = f"{horiz}, {vert}"

    # distance by object_area
    if object_area > 120000:
        dist_str = "very close"
    elif object_area > 40000:
        dist_str = "close"
    elif object_area > 10000:
        dist_str = "medium distance"
    else:
        dist_str = "far away"

    # which hand
    dist_left = 9999
    dist_right = 9999
    if left_wrist:
        lx, ly = left_wrist
        dist_left = math.hypot(obj_cx - lx, obj_cy - ly)
    if right_wrist:
        rx, ry = right_wrist
        dist_right = math.hypot(obj_cx - rx, obj_cy - ry)

    if min(dist_left, dist_right) < 120:
        if dist_left < dist_right:
            hand_str = "near your left hand"
        else:
            hand_str = "near your right hand"
    else:
        hand_str = "near your body"

    return f"The object is {dir_str}, {dist_str}, {hand_str}."

def run_detection(obj_name: str, config: Dict[str,Any], speaker: GTTSSpeaker, state: SharedState):
    """
    Runs YOLO + MediaPipe. 
    - Times out after 30s if object not found at all.
    - If user says hey assistant in the background thread => state.interrupt_detection = True => we stop
    """
    if speaker:
        speaker.speak(f"Looking for {obj_name} now. I'll let you know if I find it.")

    # load YOLO
    from vision.object_detection import YOLODetector
    from vision.mediapipe_tracker import MediaPipeTracker
    from integration.detection_filter import filter_detections

    model_path = config.get("vision", {}).get("model_path", "models/yolo/yolov5m_Objects365.pt")
    conf_thr = config.get("vision", {}).get("confidence_threshold", 0.10)
    iou_thr = config.get("vision", {}).get("nms_threshold", 0.45)
    detector = YOLODetector(weights_path=model_path, conf_threshold=conf_thr, iou_threshold=iou_thr)

    mp_tracker = MediaPipeTracker()

    cap = cv2.VideoCapture(config.get("device", {}).get("camera_index", 0))
    if not cap.isOpened():
        if speaker:
            speaker.speak("Camera not available. Cancelling detection.")
        return

    # set resolution to speed up
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

    start_time = time.time()
    found_any_object = False
    frames_since_speak = 0
    grab_frames_count = 0
    last_spoken_text = None

    while True:
        if state.exit_program:
            logger.info("System exit flagged. Stopping detection.")
            break
        if state.interrupt_detection:
            logger.info("Detection interrupted by new wake word.")
            break

        elapsed = time.time() - start_time
        if elapsed > OBJECT_FIND_TIMEOUT_SECS and not found_any_object:
            if speaker:
                speaker.speak("I could not find that object. Please move the camera or try again.")
            break

        ret, frame = cap.read()
        if not ret:
            logger.warning("Failed to read camera.")
            break

        # YOLO detect
        detections = detector.detect(frame)
        filtered = filter_detections({"name": obj_name}, detections)

        user_landmarks = mp_tracker.process_frame(frame)
        if not user_landmarks:
            cv2.putText(frame, "No user landmarks", (30,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),2)
            cv2.imshow("AIVOL - Threaded", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        if "mid_shoulder" in user_landmarks:
            user_cx, user_cy = user_landmarks["mid_shoulder"]
        else:
            user_cx, user_cy = (frame.shape[1]//2, frame.shape[0]//2)

        left_wrist = user_landmarks.get("left_wrist")
        right_wrist = user_landmarks.get("right_wrist")

        if not filtered:
            cv2.putText(frame, f"Target '{obj_name}' not found", (30,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),2)
            cv2.imshow("AIVOL - Threaded", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        found_any_object = True

        # pick nearest
        best_obj = choose_nearest_object(filtered, user_cx, user_cy)
        if not best_obj:
            cv2.putText(frame, "No nearest obj", (30,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),2)
            cv2.imshow("AIVOL - Threaded", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        x1, y1, x2, y2 = map(int, best_obj["bbox"])
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0),2)
        label = best_obj["label"]
        conf = best_obj["confidence"]
        cv2.putText(frame, f"{label} {conf:.2f}", (x1,y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0),2)

        obj_cx = (x1 + x2)//2
        obj_cy = (y1 + y2)//2
        obj_area = (x2 - x1)*(y2 - y1)

        # direction
        direction_text = generate_feedback(obj_cx, obj_cy, user_cx, user_cy,
                                           left_wrist, right_wrist, obj_area)
        cv2.putText(frame, direction_text, (30,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255),2)

        # grabbing
        grabbed_this_frame = False
        if right_wrist:
            rx, ry = right_wrist
            if math.hypot(obj_cx - rx, obj_cy - ry) < 40:
                grabbed_this_frame = True
        if left_wrist and not grabbed_this_frame:
            lx, ly = left_wrist
            if math.hypot(obj_cx - lx, obj_cy - ly) < 40:
                grabbed_this_frame = True

        if grabbed_this_frame:
            grab_frames_count += 1
        else:
            grab_frames_count = 0

        if grab_frames_count >= 6:
            if speaker:
                speaker.speak(f"You've firmly grabbed the {obj_name}. Good job!")
            break

        # speak directions ~ every 5 frames
        frames_since_speak += 1
        if direction_text and frames_since_speak > 5:
            if speaker:
                speaker.speak(direction_text)
            frames_since_speak = 0

        cv2.imshow("AIVOL - Threaded", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    logger.info("Detection for %s done." % obj_name)

def main():
    logger.info("Starting AIVOL with threaded wake-word approach.")
    config = load_config()
    initialize_system()

    from audio.text_to_speech_gTTS import GTTSSpeaker
    speaker = GTTSSpeaker(language="en", slow=False, output_folder="audio_outputs")

    # shared state
    state = SharedState()

    # background thread for wake word
    t = threading.Thread(target=wake_word_listener, args=(state, speaker), daemon=True)
    t.start()

    while True:
        if state.exit_program:
            break

        # Wait for initial "Hey Assistant"
        logger.info("[Main] Looking for initial wake word.")
        woke = wait_for_initial_wake_word(state, speaker)
        if not woke or state.exit_program:
            logger.info("Exiting main loop.")
            break

        # get user command
        user_cmd = get_user_command(speaker)
        if not user_cmd:
            logger.info("No user command recognized. Return to idle.")
            continue

        from text.text_processor import process_user_prompt
        obj_info = process_user_prompt(user_cmd)
        if not obj_info.name:
            if speaker:
                speaker.speak("I couldn't understand which object to find.")
            continue

        # reset detection interrupt
        state.interrupt_detection = False

        # run detection
        run_detection(obj_info.name, config, speaker, state)

        # if we didn't forcibly interrupt, we speak done
        if not state.interrupt_detection:
            if speaker:
                speaker.speak("Detection finished. Returning to idle mode.")
        else:
            logger.info("Detection interrupted. Returning to idle.")
            state.interrupt_detection = False

    state.exit_program = True
    logger.info("Stopping. The wake-word thread ends with main thread.")
    logger.info("Program ended.")

def wait_for_initial_wake_word(state: SharedState, speaker: GTTSSpeaker) -> bool:
    """
    Synchronously wait for "hey assistant" with a short approach.
    If user says "exit," we set exit_program and return False.
    If user says "hey assistant," return True.
    """
    while not state.exit_program:
        phrase = get_voice_command(
            timeout=WAKE_TIMEOUT,
            phrase_time_limit=WAKE_PHRASE_LIMIT,
            ambient_noise_duration=0.5
        )
        if phrase is None:
            logger.info("No wake word detected. Retrying...")
            continue
        phrase = phrase.lower().strip()
        logger.info("Heard: '%s'", phrase)
        if phrase in WAKE_WORD_ALTS:
            logger.info("Wake word detected!")
            return True
        elif phrase == "exit":
            state.exit_program = True
            return False
        else:
            logger.info("Not the wake word.")
    return False

if __name__ == "__main__":
    main()
