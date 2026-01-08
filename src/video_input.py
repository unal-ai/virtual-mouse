"""Video input handler for Virtual Mouse.

Supports multiple video sources:
- Built-in webcam (index 0)
- External USB camera (configurable index)
- RTSP video stream
"""

import cv2
import time
from typing import Optional, Tuple, Union, Generator, Callable
from .config import CameraConfig


class VideoInput:
    """Unified video input handler with auto-reconnect for RTSP streams."""
    
    def __init__(self, config: CameraConfig, on_reconnect_fail: Optional[Callable[[], Optional[str]]] = None):
        """Initialize video input.
        
        Args:
            config: Camera configuration object.
            on_reconnect_fail: Callback function to run when reconnection fails.
                              Should return new RTSP URL if found, or None.
        """
        self.config = config
        self.cap: Optional[cv2.VideoCapture] = None
        self._reconnect_delay = 2.0  # seconds
        self._max_reconnect_attempts = 5
        self._is_rtsp = isinstance(config.source, str) and config.source.startswith("rtsp://")
        self._on_reconnect_fail = on_reconnect_fail
        
    def open(self) -> bool:
        """Open the video capture device.
        
        Returns:
            True if successfully opened, False otherwise.
        """
        try:
            self.cap = cv2.VideoCapture(self.config.source)
            
            if not self.cap.isOpened():
                print(f"Error: Could not open video source: {self.config.source}")
                return False
            
            # Set resolution for webcam (may not work for RTSP)
            if not self._is_rtsp:
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
            
            # Get actual dimensions
            self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            print(f"Video source opened: {self.config.source}")
            print(f"Resolution: {self.width}x{self.height}")
            
            return True
            
        except Exception as e:
            print(f"Error opening video source: {e}")
            return False
    
    def read(self) -> Tuple[bool, Optional["cv2.Mat"]]:
        """Read a frame from the video source.
        
        Returns:
            Tuple of (success, frame). Frame is None if read failed.
        """
        if self.cap is None or not self.cap.isOpened():
            return False, None
        
        success, frame = self.cap.read()
        
        if not success:
            if self._is_rtsp:
                # Try to reconnect for RTSP streams
                print("RTSP stream disconnected, attempting to reconnect...")
                if self._reconnect():
                    return self.read()
            return False, None
        
        # Apply horizontal flip if configured
        if self.config.flip_horizontal:
            frame = cv2.flip(frame, 1)
        
        return True, frame
    
    def _reconnect(self) -> bool:
        """Attempt to reconnect to RTSP stream with exponential backoff.
        
        Uses infinite retry for RTSP streams - keeps trying until success.
        
        Returns:
            True if reconnected successfully, False to signal temporary failure.
        """
        delay = self._reconnect_delay
        max_delay = 30.0  # Cap at 30 seconds
        attempt = 0
        
        while True:
            attempt += 1
            print(f"Reconnect attempt {attempt} (waiting {delay:.1f}s)...")
            
            # Smart Reconnection: Check for new stream URL every few attempts
            if attempt % 3 == 0 and self._on_reconnect_fail:
                print("Checking for new stream location...")
                new_url = self._on_reconnect_fail()
                if new_url and new_url != self.config.source:
                    print(f"Found new stream URL: {new_url}")
                    self.config.source = new_url
                    attempt = 0 # Reset attempts for new URL
                    delay = self._reconnect_delay # Reset delay
            
            time.sleep(delay)
            
            if self.cap is not None:
                self.cap.release()
            
            try:
                self.cap = cv2.VideoCapture(self.config.source)
                if self.cap.isOpened():
                    # Test read to confirm connection
                    ret, _ = self.cap.read()
                    if ret:
                        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        print(f"Reconnected successfully! Resolution: {self.width}x{self.height}")
                        return True
            except Exception as e:
                print(f"Reconnect error: {e}")
            
            # Exponential backoff with cap
            delay = min(delay * 1.5, max_delay)
            print(f"Connection failed, retrying...")
    
    def frames(self) -> Generator[Tuple[bool, Optional["cv2.Mat"]], None, None]:
        """Generator that yields frames continuously.
        
        Yields:
            Tuple of (success, frame) for each frame.
        """
        while True:
            success, frame = self.read()
            if not success and not self._is_rtsp:
                break
            yield success, frame
    
    def release(self):
        """Release the video capture device."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            print("Video source released.")
    
    def is_opened(self) -> bool:
        """Check if video capture is open.
        
        Returns:
            True if open, False otherwise.
        """
        return self.cap is not None and self.cap.isOpened()
    
    def __enter__(self):
        """Context manager entry."""
        self.open()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.release()
        return False


def create_video_input(source: Union[int, str], 
                       width: int = 640, 
                       height: int = 480,
                       flip: bool = True,
                       on_reconnect_fail: Optional[Callable[[], Optional[str]]] = None) -> VideoInput:
    """Create a VideoInput with the given parameters.
    
    Args:
        source: Video source (camera index or RTSP URL).
        width: Desired frame width.
        height: Desired frame height.
        flip: Whether to flip horizontally.
    
    Returns:
        Configured VideoInput instance.
    """
    config = CameraConfig(
        source=source,
        width=width,
        height=height,
        flip_horizontal=flip
    )
    return VideoInput(config, on_reconnect_fail)
