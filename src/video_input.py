"""Video input handler for Virtual Mouse.

Supports multiple video sources:
- Built-in webcam (index 0)
- External USB camera (configurable index)
- RTSP video stream
"""

import cv2
import time
from typing import Optional, Tuple, Union, Generator
from .config import CameraConfig


class VideoInput:
    """Unified video input handler with auto-reconnect for RTSP streams."""
    
    def __init__(self, config: CameraConfig):
        """Initialize video input.
        
        Args:
            config: Camera configuration object.
        """
        self.config = config
        self.cap: Optional[cv2.VideoCapture] = None
        self._reconnect_delay = 2.0  # seconds
        self._max_reconnect_attempts = 5
        self._is_rtsp = isinstance(config.source, str) and config.source.startswith("rtsp://")
        
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
        """Attempt to reconnect to RTSP stream.
        
        Returns:
            True if reconnected successfully, False otherwise.
        """
        for attempt in range(self._max_reconnect_attempts):
            print(f"Reconnect attempt {attempt + 1}/{self._max_reconnect_attempts}...")
            time.sleep(self._reconnect_delay)
            
            if self.cap is not None:
                self.cap.release()
            
            if self.open():
                print("Reconnected successfully!")
                return True
        
        print("Failed to reconnect after maximum attempts.")
        return False
    
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
                       flip: bool = True) -> VideoInput:
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
    return VideoInput(config)
