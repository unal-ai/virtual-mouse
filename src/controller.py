"""Cross-platform mouse controller for Virtual Mouse.

Uses PyAutoGUI with threading to prevent blocking the main loop.
Mouse operations run in a dedicated thread for consistent frame rates.
"""

import queue
import threading
import time
from typing import Optional, Tuple, Callable

import pyautogui

from .config import MouseConfig
from .gestures import Gesture


# Disable PyAutoGUI fail-safe
pyautogui.FAILSAFE = False
pyautogui.MOVETO_DURATION = 0


class OneEuroFilter:
    """One Euro Filter for smooth, low-latency cursor tracking.
    
    Adaptive low-pass filter that:
    - Filters aggressively during slow movement (removes jitter)
    - Responds instantly during fast movement (low latency)
    
    Reference: https://cristal.univ-lille.fr/~casiez/1euro/
    """
    
    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.007, d_cutoff: float = 1.0):
        """Initialize filter.
        
        Args:
            min_cutoff: Minimum cutoff frequency (lower = more smoothing when slow)
            beta: Speed coefficient (higher = less smoothing when fast)
            d_cutoff: Derivative cutoff frequency
        """
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        
        self.x_prev: Optional[float] = None
        self.dx_prev: float = 0.0
        self.t_prev: Optional[float] = None
    
    def _smoothing_factor(self, t_e: float, cutoff: float) -> float:
        """Compute exponential smoothing factor."""
        r = 2 * 3.14159 * cutoff * t_e
        return r / (r + 1)
    
    def _exp_smooth(self, a: float, x: float, x_prev: float) -> float:
        """Apply exponential smoothing."""
        return a * x + (1 - a) * x_prev
    
    def filter(self, x: float, t: Optional[float] = None) -> float:
        """Filter a value.
        
        Args:
            x: Input value
            t: Timestamp (uses time.time() if not provided)
        
        Returns:
            Filtered value
        """
        if t is None:
            t = time.time()
        
        if self.x_prev is None:
            self.x_prev = x
            self.t_prev = t
            return x
        
        t_e = t - self.t_prev
        if t_e <= 0:
            t_e = 1e-6
        
        # Estimate derivative
        dx = (x - self.x_prev) / t_e
        a_d = self._smoothing_factor(t_e, self.d_cutoff)
        dx_hat = self._exp_smooth(a_d, dx, self.dx_prev)
        
        # Adaptive cutoff based on speed
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        
        # Filter the value
        a = self._smoothing_factor(t_e, cutoff)
        x_hat = self._exp_smooth(a, x, self.x_prev)
        
        # Update state
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t
        
        return x_hat
    
    def reset(self) -> None:
        """Reset filter state."""
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None


class MouseController:
    """Cross-platform mouse controller with threading and smoothing."""
    
    def __init__(self, config: MouseConfig):
        """Initialize mouse controller.
        
        Args:
            config: Mouse configuration object.
        """
        self.config = config
        
        # Screen dimensions
        self.screen_width, self.screen_height = pyautogui.size()
        
        # One Euro Filters for smooth, low-latency tracking
        # min_cutoff=1.0: smooth when slow, beta=0.007: responsive when fast
        self._filter_x = OneEuroFilter(min_cutoff=1.0, beta=0.007)
        self._filter_y = OneEuroFilter(min_cutoff=1.0, beta=0.007)
        
        # Track current cursor position
        self._curr_x: float = self.screen_width / 2
        self._curr_y: float = self.screen_height / 2
        
        # State flags
        self.is_dragging: bool = False
        self.click_ready: bool = False
        self.last_click_time: float = 0
        
        # Trackpad mode state
        self.trackpad_mode: bool = config.trackpad_mode
        self.trackpad_sensitivity: float = config.trackpad_sensitivity
        self.prev_hand_x: Optional[float] = None
        self.prev_hand_y: Optional[float] = None
        
        # Pinch control state
        self.pinch_active: bool = False
        self.pinch_start_x: float = 0
        self.pinch_start_y: float = 0
        self.pinch_direction: Optional[str] = None
        self.prev_pinch_level: float = 0
        self.pinch_frame_count: int = 0
        
        # Threading: command queue and worker thread
        self._command_queue: queue.Queue = queue.Queue()
        self._running = True
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker_thread.start()
    
    def _worker_loop(self) -> None:
        """Worker thread that processes mouse commands."""
        while self._running:
            try:
                # Get command with timeout to allow clean shutdown
                cmd = self._command_queue.get(timeout=0.1)
                if cmd is None:
                    break
                
                action, args = cmd
                
                if action == 'move':
                    pyautogui.moveTo(args[0], args[1])
                elif action == 'click':
                    pyautogui.click()
                elif action == 'right_click':
                    pyautogui.click(button='right')
                elif action == 'double_click':
                    pyautogui.doubleClick()
                elif action == 'mouse_down':
                    pyautogui.mouseDown()
                elif action == 'mouse_up':
                    pyautogui.mouseUp()
                elif action == 'scroll':
                    pyautogui.scroll(args[0])
                elif action == 'hscroll':
                    pyautogui.keyDown('shift')
                    pyautogui.scroll(args[0])
                    pyautogui.keyUp('shift')
                elif action == 'back':
                    pyautogui.hotkey('command', '[')
                elif action == 'tab':
                    pyautogui.hotkey('command', 'tab')
                
                self._command_queue.task_done()
            except queue.Empty:
                continue
            except Exception:
                pass  # Ignore errors in worker thread
    
    def _send_command(self, action: str, *args) -> None:
        """Send a command to the worker thread (non-blocking)."""
        # Drop old move commands to avoid queue buildup
        if action == 'move':
            # Clear any pending move commands
            try:
                while True:
                    cmd = self._command_queue.get_nowait()
                    if cmd and cmd[0] != 'move':
                        # Put non-move commands back
                        self._command_queue.put(cmd)
                        break
            except queue.Empty:
                pass
        
        self._command_queue.put((action, args))
    
    def _can_click(self) -> bool:
        """Check if enough time has passed since last click."""
        current_time = time.time()
        if current_time - self.last_click_time >= self.config.click_debounce:
            self.last_click_time = current_time
            return True
        return False
    
    def move_cursor(self, norm_x: float, norm_y: float) -> None:
        """Move cursor to normalized position (non-blocking).
        
        Uses One Euro Filter for adaptive smoothing:
        - Slow movement -> strong smoothing (removes jitter)
        - Fast movement -> weak smoothing (responsive)
        """
        if not self.trackpad_mode:
            # Absolute positioning (Touchscreen mode)
            if self.config.screen_region.enabled:
                region = self.config.screen_region
                x = region.x + norm_x * region.width
                y = region.y + norm_y * region.height
            else:
                x = norm_x * self.screen_width
                y = norm_y * self.screen_height
        
            # Apply One Euro Filter for smooth, low-latency tracking
            t = time.time()
            smooth_x = self._filter_x.filter(x, t)
            smooth_y = self._filter_y.filter(y, t)
            
            smooth_x = max(0, min(self.screen_width - 1, smooth_x))
            smooth_y = max(0, min(self.screen_height - 1, smooth_y))
            
            self._curr_x = smooth_x
            self._curr_y = smooth_y
            
            self._send_command('move', int(smooth_x), int(smooth_y))

        else:
            # Relative positioning (Trackpad mode)
            if self.prev_hand_x is None:
                self.prev_hand_x = norm_x
                self.prev_hand_y = norm_y
                return
            
            # Calculate delta with sensitivity
            delta_x = (norm_x - self.prev_hand_x) * self.screen_width * self.trackpad_sensitivity
            delta_y = (norm_y - self.prev_hand_y) * self.screen_height * self.trackpad_sensitivity
            
            # Update current cursor position
            self._curr_x += delta_x
            self._curr_y += delta_y
            
            # Clamp to screen
            self._curr_x = max(0, min(self.screen_width - 1, self._curr_x))
            self._curr_y = max(0, min(self.screen_height - 1, self._curr_y))
            
            # Update previous hand position
            self.prev_hand_x = norm_x
            self.prev_hand_y = norm_y
            
            self._send_command('move', int(self._curr_x), int(self._curr_y))
    
    def left_click(self) -> None:
        """Perform left click (non-blocking)."""
        if self._can_click():
            self._send_command('click')
    
    def right_click(self) -> None:
        """Perform right click (non-blocking)."""
        if self._can_click():
            self._send_command('right_click')
    
    def double_click(self) -> None:
        """Perform double click (non-blocking)."""
        if self._can_click():
            self._send_command('double_click')

    def back(self) -> None:
        """Perform back action (non-blocking)."""
        if self._can_click():
            self._send_command('back')

    def tab(self) -> None:
        """Perform tab switch action (non-blocking)."""
        if self._can_click():
            self._send_command('tab')
    
    def start_drag(self) -> None:
        """Start drag operation."""
        if not self.is_dragging:
            self.is_dragging = True
            self._send_command('mouse_down')
    
    def stop_drag(self) -> None:
        """Stop drag operation."""
        if self.is_dragging:
            self.is_dragging = False
            self._send_command('mouse_up')
    
    def scroll_vertical(self, amount: int = 120) -> None:
        """Scroll vertically (non-blocking)."""
        self._send_command('scroll', amount)
    
    def scroll_horizontal(self, amount: int = 120) -> None:
        """Scroll horizontally (non-blocking)."""
        self._send_command('hscroll', -amount)
    
    def init_pinch_control(self, start_x: float, start_y: float) -> None:
        """Initialize pinch control state."""
        self.pinch_start_x = start_x
        self.pinch_start_y = start_y
        self.prev_pinch_level = 0
        self.pinch_frame_count = 0
        self.pinch_direction = None
    
    def update_pinch_control(self, current_x: float, current_y: float,
                            horizontal_action: Callable, 
                            vertical_action: Callable) -> None:
        """Update pinch control based on hand movement."""
        dx = (current_x - self.pinch_start_x) * 10
        dy = (self.pinch_start_y - current_y) * 10
        
        threshold = 0.3
        
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
        """Handle a detected gesture and perform corresponding action."""
        action = "Idle"
        
        # Reset trackpad state if no hand position (hand lost)
        if hand_position is None and self.prev_hand_x is not None:
            self.prev_hand_x = None
            self.prev_hand_y = None

        if gesture != Gesture.FIST and self.is_dragging:
            self.stop_drag()
        
        if gesture not in [Gesture.PINCH_MAJOR, Gesture.PINCH_MINOR] and self.pinch_active:
            self.pinch_active = False
        
        if gesture == Gesture.PALM:
            self.prev_x = None
            self.prev_y = None
            # In trackpad mode, PALM is Tab
            if self.trackpad_mode:
                if self.click_ready:
                    self.tab()
                    self.click_ready = False
                action = "Tab Switch"
            else:
                action = "Stop"
        
        elif gesture == Gesture.V_GEST:
            self.click_ready = True
            if hand_position:
                self.move_cursor(hand_position[0], hand_position[1])
                action = "Moving Cursor"
        
        elif gesture == Gesture.FIST:
            if self.trackpad_mode:
                 # In trackpad mode, skip drag for now as per simplified spec, or map it to something else?
                 # Spec didn't ask for drag. But FIST is "All fingers closed".
                 # User asked for:
                 # Move (V)
                 # Left Click (Thumb Up) -> Need new gesture enum
                 # Back (4 fingers) -> Need new gesture enum
                 # Tab (Palm) -> Handled above
                 pass
            else:
                if not self.is_dragging:
                    self.start_drag()
                if hand_position:
                    self.move_cursor(hand_position[0], hand_position[1])
                action = "Dragging"
        
        elif gesture == Gesture.TWO_FINGER_CLOSED:
             # Standard mode: click.
             if not self.trackpad_mode and self.click_ready:
                 self.left_click()
                 self.click_ready = False
                 action = "Left Click"
        
        elif gesture == Gesture.THUMB_UP: # New Gesture
            if self.trackpad_mode and self.click_ready:
                self.left_click()
                self.click_ready = False
                action = "Left Click"

        elif gesture == Gesture.FOUR_FINGERS: # New Gesture
            if self.trackpad_mode and self.click_ready:
                self.back()
                self.click_ready = False
                action = "Back"

        elif gesture == Gesture.INDEX and self.click_ready:
            if not self.trackpad_mode:
                self.right_click()
                self.click_ready = False
                action = "Right Click"
        
        elif gesture == Gesture.PINCH_MAJOR and self.click_ready:
             if not self.trackpad_mode:
                self.double_click()
                self.click_ready = False
                action = "Double Click"
        
        elif gesture == Gesture.PINCH_MINOR:
            if not self.trackpad_mode:
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
            if not self.trackpad_mode:
                self.click_ready = True
                if hand_position:
                    self.move_cursor(hand_position[0], hand_position[1])
                    action = "Moving Cursor"
        
        return action
    
    def reset(self) -> None:
        """Reset controller state."""
        self._filter_x.reset()
        self._filter_y.reset()
        self.stop_drag()
        self.pinch_active = False
        self.click_ready = False
        self.prev_hand_x = None
        self.prev_hand_y = None
    
    def shutdown(self) -> None:
        """Shutdown the worker thread."""
        self._running = False
        self._command_queue.put(None)
        self._worker_thread.join(timeout=1.0)
