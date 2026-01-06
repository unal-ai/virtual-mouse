"""Gesture recognition module for Virtual Mouse.

Uses MediaPipe Hands for hand landmark detection and converts
landmarks into recognizable gestures for mouse control.
"""

import math
from enum import IntEnum
from typing import Optional, Tuple, List

import mediapipe as mp
from google.protobuf.json_format import MessageToDict

from .config import GestureConfig


# MediaPipe solutions
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


class Gesture(IntEnum):
    """Gesture encodings using binary representation for finger states."""
    # Binary encoded (each bit represents a finger: pinky, ring, mid, index)
    FIST = 0          # 0000 - All fingers down
    PINKY = 1         # 0001 - Only pinky up
    RING = 2          # 0010 - Only ring up
    MID = 4           # 0100 - Only middle up
    LAST3 = 7         # 0111 - Pinky + Ring + Middle up
    INDEX = 8         # 1000 - Only index up
    FIRST2 = 12       # 1100 - Index + Middle up
    LAST4 = 15        # 1111 - All except thumb
    THUMB = 16        # Thumb up (special case)
    PALM = 31         # All fingers up
    
    # Special gestures (detected through additional logic)
    V_GEST = 33           # V gesture (index + middle spread)
    TWO_FINGER_CLOSED = 34  # Index + middle together
    PINCH_MAJOR = 35      # Pinch with major (dominant) hand
    PINCH_MINOR = 36      # Pinch with minor (non-dominant) hand
    
    UNKNOWN = 99          # Unknown gesture


class HandLabel(IntEnum):
    """Labels for distinguishing between hands."""
    MINOR = 0  # Non-dominant hand (left for right-handed)
    MAJOR = 1  # Dominant hand (right for right-handed)


class HandRecognizer:
    """Recognizes gestures from hand landmarks."""
    
    def __init__(self, hand_label: HandLabel, stability_frames: int = 4):
        """Initialize hand recognizer.
        
        Args:
            hand_label: Whether this is the major or minor hand.
            stability_frames: Number of frames to wait before confirming gesture change.
        """
        self.hand_label = hand_label
        self.stability_frames = stability_frames
        
        self.finger: int = 0
        self.current_gesture: Gesture = Gesture.PALM
        self.prev_gesture: Gesture = Gesture.PALM
        self.stable_gesture: Gesture = Gesture.PALM
        self.frame_count: int = 0
        
        self.hand_result = None
    
    def update(self, hand_landmarks) -> None:
        """Update with new hand landmarks.
        
        Args:
            hand_landmarks: MediaPipe hand landmarks.
        """
        self.hand_result = hand_landmarks
    
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
    
    def get_gesture(self) -> Gesture:
        """Get the current stable gesture.
        
        Returns:
            The detected gesture after stability filtering.
        """
        if self.hand_result is None:
            return Gesture.PALM
        
        self._detect_finger_state()
        
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
        else:
            # Try to match to known gesture, fallback to PALM for unrecognized states
            try:
                current = Gesture(self.finger)
            except ValueError:
                # Finger state doesn't match any defined gesture
                current = Gesture.PALM
        
        # Stability filtering
        if current == self.prev_gesture:
            self.frame_count += 1
        else:
            self.frame_count = 0
        
        self.prev_gesture = current
        
        if self.frame_count > self.stability_frames:
            self.stable_gesture = current
        
        return self.stable_gesture
    
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


class GestureDetector:
    """Main gesture detection class using MediaPipe Hands."""
    
    def __init__(self, config: GestureConfig):
        """Initialize gesture detector.
        
        Args:
            config: Gesture configuration object.
        """
        self.config = config
        
        # Initialize MediaPipe Hands
        self.hands = mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=config.detection_confidence,
            min_tracking_confidence=config.tracking_confidence
        )
        
        # Hand recognizers
        self.major_hand = HandRecognizer(HandLabel.MAJOR, config.stability_frames)
        self.minor_hand = HandRecognizer(HandLabel.MINOR, config.stability_frames)
        
        # Results storage
        self.results = None
        self.major_landmarks = None
        self.minor_landmarks = None
        
        # Dominant hand setting (True = right-handed)
        self.dom_hand_right = True
    
    def process(self, frame) -> bool:
        """Process a video frame to detect hands.
        
        Args:
            frame: BGR image from OpenCV.
        
        Returns:
            True if at least one hand was detected.
        """
        import cv2
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        
        # Process with MediaPipe
        self.results = self.hands.process(rgb_frame)
        
        # Classify hands
        self._classify_hands()
        
        # Update recognizers
        self.major_hand.update(self.major_landmarks)
        self.minor_hand.update(self.minor_landmarks)
        
        return self.results.multi_hand_landmarks is not None
    
    def _classify_hands(self) -> None:
        """Classify detected hands as major (dominant) or minor."""
        self.major_landmarks = None
        self.minor_landmarks = None
        
        if self.results is None or self.results.multi_hand_landmarks is None:
            return
        
        left = None
        right = None
        
        for i, hand_landmarks in enumerate(self.results.multi_hand_landmarks):
            try:
                handedness = MessageToDict(self.results.multi_handedness[i])
                label = handedness['classification'][0]['label']
                
                if label == 'Right':
                    right = hand_landmarks
                else:
                    left = hand_landmarks
            except (IndexError, KeyError):
                pass
        
        # Assign based on dominance
        if self.dom_hand_right:
            self.major_landmarks = right
            self.minor_landmarks = left
        else:
            self.major_landmarks = left
            self.minor_landmarks = right
    
    def get_gesture(self) -> Tuple[Gesture, Optional[HandRecognizer]]:
        """Get the current gesture and the hand making it.
        
        Prioritizes pinch gesture from minor hand, otherwise uses major hand.
        
        Returns:
            Tuple of (gesture, hand_recognizer).
        """
        minor_gesture = self.minor_hand.get_gesture()
        
        # Pinch from minor hand has priority (for scrolling)
        if minor_gesture == Gesture.PINCH_MINOR:
            return minor_gesture, self.minor_hand
        
        major_gesture = self.major_hand.get_gesture()
        return major_gesture, self.major_hand
    
    def draw_landmarks(self, frame) -> None:
        """Draw hand landmarks on the frame.
        
        Args:
            frame: BGR image to draw on.
        """
        if self.results is None or self.results.multi_hand_landmarks is None:
            return
        
        for hand_landmarks in self.results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )
    
    def release(self) -> None:
        """Release MediaPipe resources."""
        self.hands.close()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.release()
        return False


# Gesture name mapping for display
GESTURE_NAMES = {
    Gesture.FIST: "✊ Fist (Drag)",
    Gesture.PALM: "🖐 Palm (Stop)",
    Gesture.V_GEST: "✌️ V-Gesture (Move)",
    Gesture.INDEX: "☝️ Index (Right Click)",
    Gesture.MID: "🖕 Middle (Move)",
    Gesture.TWO_FINGER_CLOSED: "✌️ Closed (Left Click)",
    Gesture.PINCH_MAJOR: "🤏 Pinch (Double Click)",
    Gesture.PINCH_MINOR: "🤏 Pinch Left (Scroll)",
    Gesture.FIRST2: "✌️ Two Fingers",
    Gesture.UNKNOWN: "❓ Unknown"
}


def get_gesture_name(gesture: Gesture) -> str:
    """Get human-readable name for a gesture.
    
    Args:
        gesture: The gesture enum value.
    
    Returns:
        Human-readable string.
    """
    return GESTURE_NAMES.get(gesture, f"Gesture {gesture}")
