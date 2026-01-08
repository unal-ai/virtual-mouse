"""Gesture recognition module for Virtual Mouse.

Uses MediaPipe Hand Landmarker (Tasks API) for hand landmark detection and converts
landmarks into recognizable gestures for mouse control.

This module uses LIVE_STREAM mode for asynchronous, non-blocking detection.
"""

import math
import os
import threading
import time
from enum import IntEnum
from typing import Optional, Tuple, List, NamedTuple

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from .config import GestureConfig


# Get the model path (in project root by default)
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "hand_landmarker.task")


class Gesture(IntEnum):
    """Gesture encodings using binary representation for finger states."""
    # Binary encoded (each bit represents a finger: pinky, ring, mid, index)
    # Bit order: Index(8) Middle(4) Ring(2) Pinky(1)
    FIST = 0          # 0000 - All fingers down
    PINKY = 1         # 0001 - Only pinky up
    RING = 2          # 0010 - Only ring up
    MID = 4           # 0100 - Only middle up
    LAST3 = 7         # 0111 - Pinky + Ring + Middle up
    INDEX = 8         # 1000 - Only index up
    FIRST2 = 12       # 1100 - Index + Middle up (2 fingers)
    FIRST3 = 14       # 1110 - Index + Middle + Ring up (3 fingers)
    LAST4 = 15        # 1111 - All 4 fingers except thumb
    FOUR_FINGERS = 15 # Alias for LAST4
    PALM = 31         # All fingers up (including thumb bit)
    
    # Special gestures (detected through additional logic)
    THUMB_UP = 32         # Thumb up with fist (detected via _is_thumb_up)
    V_GEST = 33           # V gesture (index + middle spread)
    TWO_FINGER_CLOSED = 34  # Index + middle together (for click)
    PINCH_MAJOR = 35      # Pinch with major (dominant) hand
    PINCH_MINOR = 36      # Pinch with minor (non-dominant) hand
    THREE_FINGERS = 37    # Three fingers extended (for right click in trackpad mode)
    
    UNKNOWN = 99          # Unknown gesture


class HandLabel(IntEnum):
    """Labels for distinguishing between hands."""
    MINOR = 0  # Non-dominant hand (left for right-handed)
    MAJOR = 1  # Dominant hand (right for right-handed)


class NormalizedLandmark(NamedTuple):
    """Simple landmark container to mimic old API structure."""
    x: float
    y: float
    z: float


class HandLandmarksWrapper:
    """Wrapper to provide the old .landmark interface for new API results."""
    
    def __init__(self, landmarks_list, score: float = 0.0, world_landmarks_list=None):
        """Wrap a list of NormalizedLandmarks from new API.
        
        Args:
            landmarks_list: List of normalized landmarks (x, y, z in 0-1 range).
            score: Detection confidence score.
            world_landmarks_list: Optional list of world landmarks (x, y, z in meters).
        """
        self.landmark = [
            NormalizedLandmark(lm.x, lm.y, lm.z) 
            for lm in landmarks_list
        ]
        self.score = score
        
        # World landmarks in real-world meters (more accurate for 3D calculations)
        if world_landmarks_list:
            self.world_landmark = [
                NormalizedLandmark(lm.x, lm.y, lm.z)
                for lm in world_landmarks_list
            ]
        else:
            self.world_landmark = None


class HandRecognizer:
    """Recognizes gestures from hand landmarks."""
    
    def __init__(self, hand_label: HandLabel, stability_frames: int = 4, thumb_sensitivity: float = 1.0):
        """Initialize hand recognizer.
        
        Args:
            hand_label: Whether this is the major or minor hand.
            stability_frames: Number of frames to wait before confirming gesture change.
            thumb_sensitivity: Multiplier for thumb detection threshold (higher = stricter).
        """
        self.hand_label = hand_label
        self.stability_frames = stability_frames
        self.thumb_sensitivity = thumb_sensitivity
        
        self.finger: int = 0
        self.current_gesture: Gesture = Gesture.PALM
        self.prev_gesture: Gesture = Gesture.PALM
        self.stable_gesture: Gesture = Gesture.PALM
        self.frame_count: int = 0
        self.thumb_up_state: bool = False
        
        self.hand_result = None
    
    def update(self, hand_landmarks) -> None:
        """Update with new hand landmarks.
        
        Args:
            hand_landmarks: HandLandmarksWrapper with landmarks.
        """
        self.hand_result = hand_landmarks

    @property
    def score(self) -> float:
        """Get detection confidence score."""
        return self.hand_result.score if self.hand_result else 0.0

    @property
    def raw_gesture(self) -> Gesture:
        """Get instantaneous gesture (before stability filter)."""
        return self.current_gesture

    @property
    def finger_state(self) -> int:
        """Get bitmask of detected fingers."""
        state = self.finger
        if self.thumb_up_state:
            state |= 16  # Add 5th bit for thumb
        return state
    
    def _get_signed_distance(self, points: List[int]) -> float:
        """Calculate signed distance between two landmark points.
        
        Args:
            points: List of two landmark indices.
        
        Returns:
            Signed distance (positive if first point is above second).
        """
        if self.hand_result is None:
            return 0.0
        
        p1 = self.hand_result.landmark[points[0]]
        p2 = self.hand_result.landmark[points[1]]
        
        sign = 1 if p1.y < p2.y else -1
        dist = math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)
        
        return dist * sign
    
    def _get_distance(self, points: List[int]) -> float:
        """Calculate absolute distance between two landmark points.
        
        Args:
            points: List of two landmark indices.
        
        Returns:
            Absolute distance.
        """
        if self.hand_result is None:
            return 0.0
        
        p1 = self.hand_result.landmark[points[0]]
        p2 = self.hand_result.landmark[points[1]]
        
        return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)
    
    def _get_z_distance(self, points: List[int]) -> float:
        """Calculate z-axis distance between two landmark points.
        
        Args:
            points: List of two landmark indices.
        
        Returns:
            Absolute z-distance.
        """
        if self.hand_result is None:
            return 0.0
        
        p1 = self.hand_result.landmark[points[0]]
        p2 = self.hand_result.landmark[points[1]]
        
        return abs(p1.z - p2.z)
    
    def _detect_finger_state(self) -> None:
        """Detect which fingers are up based on landmarks.
        
        Sets self.finger as a binary encoding:
        - Bit 0: Pinky
        - Bit 1: Ring
        - Bit 2: Middle
        - Bit 3: Index
        """
        if self.hand_result is None:
            return
        
        # Finger tip and base points: [tip, pip, mcp]
        # Index 8,5,0 | Middle 12,9,0 | Ring 16,13,0 | Pinky 20,17,0
        finger_points = [
            [8, 5, 0],   # Index
            [12, 9, 0],  # Middle
            [16, 13, 0], # Ring
            [20, 17, 0]  # Pinky
        ]
        
        self.finger = 0
        
        for i, points in enumerate(finger_points):
            # Distance from tip to pip
            dist1 = self._get_signed_distance(points[:2])
            # Distance from pip to mcp
            dist2 = self._get_signed_distance(points[1:])
            
            try:
                ratio = round(dist1 / dist2, 1)
            except ZeroDivisionError:
                ratio = round(dist1 / 0.01, 1)
            
            # Shift left and add 1 if finger is up
            self.finger = self.finger << 1
            if ratio > 0.5:
                self.finger = self.finger | 1
                
    def _is_thumb_up(self) -> bool:
        """Check if thumb is up/extended."""
        if self.hand_result is None:
            return False
            
        # Check angle or distance of thumb
        # 1. Check if Tip(4) is far from Pinky MCP(17) - implies open hand or thumb out
        dist_thumb_pinky = self._get_distance([4, 17])
        
        # 2. Check if Thumb Tip(4) is further from MCP(2) than IP(3) is from MCP(2)
        # Using simple distance check vs Index finger MCP(5) as reference for scale
        scale_ref = self._get_distance([5, 17]) # Width of palm
        
        if scale_ref == 0:
            return False
            
        return dist_thumb_pinky > (scale_ref * self.thumb_sensitivity)
    
    def get_gesture(self) -> Gesture:
        """Get the current stable gesture.
        
        Returns:
            The detected gesture after stability filtering.
        """
        if self.hand_result is None:
            return Gesture.PALM
        
        self._detect_finger_state()
        thumb_up = self._is_thumb_up()
        self.thumb_up_state = thumb_up
        
        # Determine current gesture
        current = Gesture.PALM
        
        # Check for pinch gesture (thumb and index close together)
        if self.finger in [Gesture.LAST3, Gesture.LAST4] and self._get_distance([8, 4]) < 0.05:
            current = Gesture.PINCH_MINOR if self.hand_label == HandLabel.MINOR else Gesture.PINCH_MAJOR
        
        # Check for V gesture or two fingers closed
        elif self.finger == Gesture.FIRST2:
            # Measure spread between index and middle
            dist_tips = self._get_distance([8, 12])
            dist_base = self._get_distance([5, 9])
            ratio = dist_tips / dist_base if dist_base > 0 else 0
            
            if ratio > 1.7:
                current = Gesture.V_GEST
            else:
                # Check z-distance to determine if fingers are touching
                if self._get_z_distance([8, 12]) < 0.1:
                    current = Gesture.TWO_FINGER_CLOSED
                else:
                    current = Gesture.MID
                    
        elif self.finger == Gesture.FIST:
            # Check for Thumb Up
            if thumb_up:
                current = Gesture.THUMB_UP
            else:
                current = Gesture.FIST
        
        elif self.finger == Gesture.FIRST3:  # 1110 - Index + Middle + Ring
            current = Gesture.THREE_FINGERS
        
        elif self.finger == Gesture.LAST4:  # 1111 - All 4 fingers
            # If thumb is also up, this is PALM (5 fingers)
            if thumb_up:
                current = Gesture.PALM
            else:
                current = Gesture.FOUR_FINGERS

        else:
            # Try to match to known gesture, fallback to PALM for unrecognized states
            try:
                current = Gesture(self.finger)
            except ValueError:
                # Finger state doesn't match any defined gesture
                current = Gesture.PALM
        
        self.current_gesture = current
        
        # Stability filtering
        if current == self.prev_gesture:
            self.frame_count += 1
        else:
            self.frame_count = 0
        
        self.prev_gesture = current
        
        if self.frame_count > self.stability_frames:
            self.stable_gesture = current
        
        return self.stable_gesture


    # ... (Keep existing methods: get_landmark_position, get_index_tip) ...


    
    def get_landmark_position(self, landmark_idx: int = 9) -> Optional[Tuple[float, float]]:
        """Get normalized position of a specific landmark.
        
        Args:
            landmark_idx: Index of the landmark (default 9 = middle finger MCP).
        
        Returns:
            Tuple of (x, y) normalized coordinates, or None if no hand detected.
        """
        if self.hand_result is None:
            return None
        
        landmark = self.hand_result.landmark[landmark_idx]
        return (landmark.x, landmark.y)
    
    def get_index_tip(self) -> Optional[Tuple[float, float]]:
        """Get position of index finger tip (landmark 8).
        
        Returns:
            Tuple of (x, y) normalized coordinates, or None if no hand.
        """
        return self.get_landmark_position(8)
    
    def get_two_finger_position(self) -> Optional[Tuple[float, float]]:
        """Get average position of index and middle fingertips.
        
        Uses landmarks 8 (index tip) and 12 (middle tip) to calculate
        a stable position for 2-finger tracking that doesn't shift
        when other fingers move.
        
        Returns:
            Tuple of (x, y) normalized coordinates, or None if no hand.
        """
        if self.hand_result is None or len(self.hand_result.landmark) < 13:
            return None
        
        index_tip = self.hand_result.landmark[8]
        middle_tip = self.hand_result.landmark[12]
        
        avg_x = (index_tip.x + middle_tip.x) / 2
        avg_y = (index_tip.y + middle_tip.y) / 2
        
        return (avg_x, avg_y)


class GestureDetector:
    """Main gesture detection class using MediaPipe Hand Landmarker.
    
    Uses LIVE_STREAM mode for async, non-blocking detection.
    """
    
    def __init__(self, config: GestureConfig):
        """Initialize gesture detector.
        
        Args:
            config: Gesture configuration object.
        """
        self.config = config
        
        # Thread-safe result storage
        self._result_lock = threading.Lock()
        self._latest_result = None
        self._frame_timestamp_ms = 0
        
        # Initialize MediaPipe Hand Landmarker with LIVE_STREAM mode
        base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=2,
            min_hand_detection_confidence=config.detection_confidence,
            min_tracking_confidence=config.tracking_confidence,
            running_mode=vision.RunningMode.LIVE_STREAM,
            result_callback=self._on_detection_result
        )
        self.hand_landmarker = vision.HandLandmarker.create_from_options(options)
        
        # Hand recognizers
        self.major_hand = HandRecognizer(HandLabel.MAJOR, config.stability_frames, config.thumb_sensitivity)
        self.minor_hand = HandRecognizer(HandLabel.MINOR, config.stability_frames, config.thumb_sensitivity)
        
        # Dominant hand setting (True = right-handed)
        self.dom_hand_right = True
        
        # For drawing landmarks (store raw results)
        self._draw_landmarks_cache = None
    
    def _on_detection_result(self, result, output_image, timestamp_ms: int) -> None:
        """Callback for async detection results.
        
        Args:
            result: HandLandmarkerResult from MediaPipe.
            output_image: The input image (unused).
            timestamp_ms: Timestamp of the processed frame.
        """
        with self._result_lock:
            self._latest_result = result
            self._draw_landmarks_cache = result.hand_landmarks if result else None
    
    def process_async(self, frame) -> None:
        """Submit frame for asynchronous processing (non-blocking).
        
        Args:
            frame: BGR image from OpenCV.
        """
        import cv2
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Create MediaPipe Image from numpy array
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Increment timestamp (must be monotonically increasing)
        self._frame_timestamp_ms += 33  # ~30fps
        
        # Submit for async processing (non-blocking)
        self.hand_landmarker.detect_async(mp_image, self._frame_timestamp_ms)
    
    def get_latest_result(self) -> Tuple[bool, Gesture, Optional[HandRecognizer]]:
        """Get the latest detection result (thread-safe).
        
        Returns:
            Tuple of (hands_detected, gesture, hand_recognizer).
        """
        with self._result_lock:
            result = self._latest_result
        
        if result is None or len(result.hand_landmarks) == 0:
            self.major_hand.update(None)
            self.minor_hand.update(None)
            return False, Gesture.PALM, None
        
        # Classify hands
        major_landmarks, minor_landmarks = self._classify_hands_from_result(result)
        
        # Update recognizers
        self.major_hand.update(major_landmarks)
        self.minor_hand.update(minor_landmarks)
        
        # Get gestures
        minor_gesture = self.minor_hand.get_gesture()
        major_gesture = self.major_hand.get_gesture()
        
        # Priority Logic:
        # 1. If Major hand has active gesture (not PALM), use Major.
        # 2. If Minor hand has active gesture (not PALM), use Minor.
        # 3. If both PALM, return the one that is actually detected (score > 0).
        # 4. Fallback to Major.
        
        if major_gesture != Gesture.PALM:
             return True, major_gesture, self.major_hand
        elif minor_gesture != Gesture.PALM:
             return True, minor_gesture, self.minor_hand
        
        # Both are PALM/Idle, check presence
        if self.major_hand.score > 0:
            return True, major_gesture, self.major_hand
        elif self.minor_hand.score > 0:
            return True, minor_gesture, self.minor_hand
            
        # Default to major hand if both missing
        return True, major_gesture, self.major_hand
    
    def _classify_hands_from_result(self, result) -> Tuple[Optional[HandLandmarksWrapper], Optional[HandLandmarksWrapper]]:
        """Classify detected hands as major or minor from result.
        
        Args:
            result: HandLandmarkerResult from MediaPipe.
        
        Returns:
            Tuple of (major_landmarks, minor_landmarks).
        """
        left = None
        right = None
        
        for i, hand_landmarks in enumerate(result.hand_landmarks):
            try:
                handedness = result.handedness[i][0]
                label = handedness.category_name
                score = handedness.score
                
                # Get world landmarks if available
                world_landmarks = None
                if hasattr(result, 'hand_world_landmarks') and i < len(result.hand_world_landmarks):
                    world_landmarks = result.hand_world_landmarks[i]
                
                wrapped = HandLandmarksWrapper(hand_landmarks, score, world_landmarks)
                
                if label == 'Right':
                    right = wrapped
                else:
                    left = wrapped
            except (IndexError, AttributeError):
                pass
        
        # Assign based on dominance
        if self.dom_hand_right:
            return right, left
        else:
            return left, right
    
    # Legacy sync method for compatibility
    def process(self, frame) -> bool:
        """Process a video frame (legacy sync wrapper).
        
        For backwards compatibility. Calls async then immediately gets result.
        
        Args:
            frame: BGR image from OpenCV.
        
        Returns:
            True if at least one hand was detected.
        """
        self.process_async(frame)
        # Small delay to allow callback to fire
        time.sleep(0.001)
        has_hands, _, _ = self.get_latest_result()
        return has_hands
    
    def get_gesture(self) -> Tuple[Gesture, Optional[HandRecognizer]]:
        """Get the current gesture (legacy compatibility).
        
        Returns:
            Tuple of (gesture, hand_recognizer).
        """
        _, gesture, hand = self.get_latest_result()
        return gesture, hand
    
    def draw_landmarks(self, frame) -> None:
        """Draw hand landmarks on the frame.
        
        Args:
            frame: BGR image to draw on.
        """
        import cv2
        
        with self._result_lock:
            landmarks_cache = self._draw_landmarks_cache
        
        if landmarks_cache is None or len(landmarks_cache) == 0:
            return
        
        height, width = frame.shape[:2]
        
        # Draw connections and landmarks manually
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
            (0, 5), (5, 6), (6, 7), (7, 8),  # Index
            (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
            (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
            (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky
            (5, 9), (9, 13), (13, 17)  # Palm
        ]
        
        for hand_landmarks in landmarks_cache:
            # Draw connections
            for start_idx, end_idx in connections:
                start = hand_landmarks[start_idx]
                end = hand_landmarks[end_idx]
                start_pt = (int(start.x * width), int(start.y * height))
                end_pt = (int(end.x * width), int(end.y * height))
                cv2.line(frame, start_pt, end_pt, (0, 255, 0), 2)
            
            # Draw landmarks
            for lm in hand_landmarks:
                pt = (int(lm.x * width), int(lm.y * height))
                cv2.circle(frame, pt, 5, (255, 0, 0), -1)
    
    def release(self) -> None:
        """Release MediaPipe resources."""
        self.hand_landmarker.close()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.release()
        return False


# Gesture name mapping for display
GESTURE_NAMES = {
    Gesture.FIST: "Fist",
    Gesture.PALM: "Palm",
    Gesture.V_GEST: "V-Gesture (Move)",
    Gesture.INDEX: "Index (Right Click)",
    Gesture.MID: "Middle",
    Gesture.TWO_FINGER_CLOSED: "2 Fingers Closed (Click)",
    Gesture.PINCH_MAJOR: "Pinch (Double Click)",
    Gesture.PINCH_MINOR: "Pinch Left (Scroll)",
    Gesture.FIRST2: "Two Fingers (Move)",
    Gesture.THREE_FINGERS: "Three Fingers (Click)",
    Gesture.THUMB_UP: "Thumb Up (Click)",
    Gesture.FOUR_FINGERS: "Four Fingers (Back)",
    Gesture.UNKNOWN: "Unknown"
}


def get_gesture_name(gesture: Gesture) -> str:
    """Get human-readable name for a gesture.
    
    Args:
        gesture: The gesture enum value.
    
    Returns:
        Human-readable string.
    """
    return GESTURE_NAMES.get(gesture, f"Gesture {gesture}")
