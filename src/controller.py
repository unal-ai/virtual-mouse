"""Cross-platform mouse controller for Virtual Mouse.

Uses PyAutoGUI for mouse control with cursor smoothing and debouncing.
"""

import time
from typing import Optional, Tuple, Callable

import pyautogui

from .config import MouseConfig
from .gestures import Gesture


# Disable PyAutoGUI fail-safe (moving mouse to corner won't stop program)
pyautogui.FAILSAFE = False

# Set movement duration to 0 for instant response
pyautogui.MOVETO_DURATION = 0


class MouseController:
    """Cross-platform mouse controller with smoothing and gesture handling."""
    
    def __init__(self, config: MouseConfig):
        """Initialize mouse controller.
        
        Args:
            config: Mouse configuration object.
        """
        self.config = config
        
        # Screen dimensions
        self.screen_width, self.screen_height = pyautogui.size()
        
        # Previous position for smoothing
        self.prev_x: Optional[float] = None
        self.prev_y: Optional[float] = None
        
        # State flags
        self.is_dragging: bool = False
        self.click_ready: bool = False  # Flag for click confirmation
        self.last_click_time: float = 0
        
        # Pinch control state
        self.pinch_active: bool = False
        self.pinch_start_x: float = 0
        self.pinch_start_y: float = 0
        self.pinch_direction: Optional[str] = None  # 'horizontal' or 'vertical'
        self.prev_pinch_level: float = 0
        self.pinch_frame_count: int = 0
    
    def _smooth_position(self, x: float, y: float) -> Tuple[float, float]:
        """Apply smoothing to cursor position to reduce jitter.
        
        Args:
            x: Target x position.
            y: Target y position.
        
        Returns:
            Smoothed (x, y) position.
        """
        if self.prev_x is None or self.prev_y is None:
            self.prev_x = x
            self.prev_y = y
            return x, y
        
        # Calculate delta
        delta_x = x - self.prev_x
        delta_y = y - self.prev_y
        dist_sq = delta_x ** 2 + delta_y ** 2
        
        # Adaptive smoothing based on movement speed
        if dist_sq <= 25:
            # Very small movement - ignore (reduces jitter)
            ratio = 0
        elif dist_sq <= 900:
            # Medium movement - smooth
            ratio = 0.07 * (dist_sq ** 0.5)
        else:
            # Large movement - respond quickly
            ratio = 2.1
        
        # Get current position
        curr_x, curr_y = pyautogui.position()
        
        # Calculate new position
        new_x = curr_x + delta_x * ratio * self.config.sensitivity
        new_y = curr_y + delta_y * ratio * self.config.sensitivity
        
        # Update previous position
        self.prev_x = x
        self.prev_y = y
        
        return new_x, new_y
    
    def _can_click(self) -> bool:
        """Check if enough time has passed since last click for debouncing.
        
        Returns:
            True if click is allowed.
        """
        current_time = time.time()
        if current_time - self.last_click_time >= self.config.click_debounce:
            self.last_click_time = current_time
            return True
        return False
    
    def move_cursor(self, norm_x: float, norm_y: float) -> None:
        """Move cursor to normalized position.
        
        Args:
            norm_x: Normalized x position (0-1).
            norm_y: Normalized y position (0-1).
        """
        # Convert normalized to screen coordinates
        if self.config.screen_region.enabled:
            region = self.config.screen_region
            x = region.x + norm_x * region.width
            y = region.y + norm_y * region.height
        else:
            x = norm_x * self.screen_width
            y = norm_y * self.screen_height
        
        # Apply smoothing
        smooth_x, smooth_y = self._smooth_position(x, y)
        
        # Clamp to screen bounds
        smooth_x = max(0, min(self.screen_width - 1, smooth_x))
        smooth_y = max(0, min(self.screen_height - 1, smooth_y))
        
        # Move cursor
        pyautogui.moveTo(int(smooth_x), int(smooth_y))
    
    def left_click(self) -> None:
        """Perform left mouse click."""
        if self._can_click():
            pyautogui.click()
    
    def right_click(self) -> None:
        """Perform right mouse click."""
        if self._can_click():
            pyautogui.click(button='right')
    
    def double_click(self) -> None:
        """Perform double click."""
        if self._can_click():
            pyautogui.doubleClick()
    
    def start_drag(self) -> None:
        """Start drag operation."""
        if not self.is_dragging:
            self.is_dragging = True
            pyautogui.mouseDown(button='left')
    
    def stop_drag(self) -> None:
        """Stop drag operation."""
        if self.is_dragging:
            self.is_dragging = False
            pyautogui.mouseUp(button='left')
    
    def scroll_vertical(self, amount: int = 120) -> None:
        """Scroll vertically.
        
        Args:
            amount: Scroll amount (positive = up, negative = down).
        """
        pyautogui.scroll(amount)
    
    def scroll_horizontal(self, amount: int = 120) -> None:
        """Scroll horizontally.
        
        Args:
            amount: Scroll amount (positive = right, negative = left).
        """
        # Horizontal scroll using shift+ctrl+scroll
        pyautogui.keyDown('shift')
        pyautogui.keyDown('ctrl')
        pyautogui.scroll(-amount)
        pyautogui.keyUp('ctrl')
        pyautogui.keyUp('shift')
    
    def init_pinch_control(self, start_x: float, start_y: float) -> None:
        """Initialize pinch control state.
        
        Args:
            start_x: Starting x position of pinch.
            start_y: Starting y position of pinch.
        """
        self.pinch_start_x = start_x
        self.pinch_start_y = start_y
        self.prev_pinch_level = 0
        self.pinch_frame_count = 0
        self.pinch_direction = None
    
    def update_pinch_control(self, current_x: float, current_y: float,
                            horizontal_action: Callable, 
                            vertical_action: Callable) -> None:
        """Update pinch control based on hand movement.
        
        Args:
            current_x: Current x position.
            current_y: Current y position.
            horizontal_action: Function to call for horizontal movement.
            vertical_action: Function to call for vertical movement.
        """
        # Calculate movement from start
        dx = (current_x - self.pinch_start_x) * 10
        dy = (self.pinch_start_y - current_y) * 10
        
        threshold = 0.3
        
        # Determine direction based on larger movement
        if abs(dy) > abs(dx) and abs(dy) > threshold:
            if self.pinch_direction is None:
                self.pinch_direction = 'vertical'
            
            if self.pinch_direction == 'vertical':
                if abs(self.prev_pinch_level - dy) < threshold:
                    self.pinch_frame_count += 1
                else:
                    self.prev_pinch_level = dy
                    self.pinch_frame_count = 0
                
                if self.pinch_frame_count >= 5:
                    self.pinch_frame_count = 0
                    vertical_action(120 if dy > 0 else -120)
        
        elif abs(dx) > threshold:
            if self.pinch_direction is None:
                self.pinch_direction = 'horizontal'
            
            if self.pinch_direction == 'horizontal':
                if abs(self.prev_pinch_level - dx) < threshold:
                    self.pinch_frame_count += 1
                else:
                    self.prev_pinch_level = dx
                    self.pinch_frame_count = 0
                
                if self.pinch_frame_count >= 5:
                    self.pinch_frame_count = 0
                    horizontal_action(120 if dx > 0 else -120)
    
    def handle_gesture(self, gesture: Gesture, hand_position: Optional[Tuple[float, float]]) -> str:
        """Handle a detected gesture and perform corresponding action.
        
        Args:
            gesture: The detected gesture.
            hand_position: Normalized (x, y) position of the hand.
        
        Returns:
            Description of the action taken.
        """
        action = "Idle"
        
        # Reset states when gesture changes
        if gesture != Gesture.FIST and self.is_dragging:
            self.stop_drag()
        
        if gesture not in [Gesture.PINCH_MAJOR, Gesture.PINCH_MINOR] and self.pinch_active:
            self.pinch_active = False
        
        # Handle gestures
        if gesture == Gesture.PALM:
            # All fingers up - do nothing (stop)
            self.prev_x = None
            self.prev_y = None
            action = "Stop"
        
        elif gesture == Gesture.V_GEST:
            # V gesture - move cursor
            self.click_ready = True
            if hand_position:
                self.move_cursor(hand_position[0], hand_position[1])
                action = "Moving Cursor"
        
        elif gesture == Gesture.FIST:
            # Fist - drag
            if not self.is_dragging:
                self.start_drag()
            if hand_position:
                self.move_cursor(hand_position[0], hand_position[1])
            action = "Dragging"
        
        elif gesture == Gesture.TWO_FINGER_CLOSED and self.click_ready:
            # Two fingers closed (from V-gesture) - LEFT CLICK
            # This is the most natural: just close fingers from V-gesture
            self.left_click()
            self.click_ready = False
            action = "Left Click"
        
        elif gesture == Gesture.INDEX and self.click_ready:
            # Index finger only (point) - RIGHT CLICK  
            # Lower middle finger from V-gesture
            self.right_click()
            self.click_ready = False
            action = "Right Click"
        
        elif gesture == Gesture.PINCH_MAJOR and self.click_ready:
            # Pinch with dominant hand - DOUBLE CLICK
            self.double_click()
            self.click_ready = False
            action = "Double Click"
        
        elif gesture == Gesture.PINCH_MINOR:
            # Minor hand pinch - scroll
            if hand_position:
                if not self.pinch_active:
                    self.pinch_active = True
                    self.init_pinch_control(hand_position[0], hand_position[1])
                else:
                    self.update_pinch_control(
                        hand_position[0], hand_position[1],
                        self.scroll_horizontal, self.scroll_vertical
                    )
            action = "Scrolling"
        
        elif gesture == Gesture.MID:
            # Middle finger only - move cursor (alternative to V-gesture)
            self.click_ready = True
            if hand_position:
                self.move_cursor(hand_position[0], hand_position[1])
                action = "Moving Cursor"
        
        return action
    
    def reset(self) -> None:
        """Reset controller state."""
        self.prev_x = None
        self.prev_y = None
        self.stop_drag()
        self.pinch_active = False
        self.click_ready = False
