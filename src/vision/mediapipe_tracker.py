"""
Module: mediapipe_tracker.py
Purpose: Use MediaPipe Pose to track upper-body landmarks (face, shoulders, wrists, etc.).
"""

import cv2
import mediapipe as mp

class MediaPipeTracker:
    def __init__(self, 
                 min_detection_confidence=0.5, 
                 min_tracking_confidence=0.5):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

        # For annotation if you want to draw
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose_landmarks_style = self.mp_drawing.DrawingSpec(color=(255,0,0), thickness=2)
        self.pose_connections_style = self.mp_drawing.DrawingSpec(color=(0,255,0), thickness=2)

    def process_frame(self, frame):
        """
        Process a single BGR frame with MediaPipe Pose.
        Returns a dictionary of pixel coordinates for key landmarks we care about:
            - nose, left_shoulder, right_shoulder, left_wrist, right_wrist, mid_shoulder, mid_hip (if visible)
        If a landmark is not found or results are None, keys may be missing or set to None.
        """
        frame_height, frame_width = frame.shape[:2]
        # Convert to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(frame_rgb)

        landmarks_dict = {}

        if results.pose_landmarks:
            # Extract raw landmarks
            lm = results.pose_landmarks.landmark

            # Nose
            nose = lm[self.mp_pose.PoseLandmark.NOSE]
            nose_x = int(nose.x * frame_width)
            nose_y = int(nose.y * frame_height)
            landmarks_dict["nose"] = (nose_x, nose_y)

            # Shoulders
            left_shoulder = lm[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = lm[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
            ls_x = int(left_shoulder.x * frame_width)
            ls_y = int(left_shoulder.y * frame_height)
            rs_x = int(right_shoulder.x * frame_width)
            rs_y = int(right_shoulder.y * frame_height)
            landmarks_dict["left_shoulder"] = (ls_x, ls_y)
            landmarks_dict["right_shoulder"] = (rs_x, rs_y)

            # Mid-shoulder
            mid_sh_x = int((ls_x + rs_x) / 2)
            mid_sh_y = int((ls_y + rs_y) / 2)
            landmarks_dict["mid_shoulder"] = (mid_sh_x, mid_sh_y)

            # Wrists
            left_wrist = lm[self.mp_pose.PoseLandmark.LEFT_WRIST]
            right_wrist = lm[self.mp_pose.PoseLandmark.RIGHT_WRIST]
            lw_x = int(left_wrist.x * frame_width)
            lw_y = int(left_wrist.y * frame_height)
            rw_x = int(right_wrist.x * frame_width)
            rw_y = int(right_wrist.y * frame_height)
            landmarks_dict["left_wrist"] = (lw_x, lw_y)
            landmarks_dict["right_wrist"] = (rw_x, rw_y)

            # Hips (if upper body only, might be out of frame, but let's do it):
            left_hip = lm[self.mp_pose.PoseLandmark.LEFT_HIP]
            right_hip = lm[self.mp_pose.PoseLandmark.RIGHT_HIP]
            lh_x = int(left_hip.x * frame_width)
            lh_y = int(left_hip.y * frame_height)
            rh_x = int(right_hip.x * frame_width)
            rh_y = int(right_hip.y * frame_height)
            landmarks_dict["left_hip"] = (lh_x, lh_y)
            landmarks_dict["right_hip"] = (rh_x, rh_y)

            # mid-hip
            mid_hip_x = int((lh_x + rh_x) / 2)
            mid_hip_y = int((lh_y + rh_y) / 2)
            landmarks_dict["mid_hip"] = (mid_hip_x, mid_hip_y)

        return landmarks_dict

    def draw_landmarks(self, frame, results):
        """
        (Optional) Use MediaPipe's drawing utilities to overlay pose landmarks on the frame.
        If you want a visual debug of the user landmarks, call this after process_frame.
        """
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.mp_drawing.draw_landmarks(
            frame_rgb,
            results.pose_landmarks,
            self.mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=self.pose_landmarks_style,
            connection_drawing_spec=self.pose_connections_style
        )
        # Convert back to BGR for display
        return cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
