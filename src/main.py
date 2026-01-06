"""Main entry point for Virtual Mouse application.

Provides CLI interface and main processing loop.
"""

import argparse
import sys
import cv2
from typing import Optional

from .config import load_config, CameraConfig
from .video_input import VideoInput
from .gestures import GestureDetector, get_gesture_name, Gesture
from .controller import MouseController


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.
    
    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="AI Virtual Mouse - Control your computer with hand gestures",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Gestures:
  ✌️ V-Gesture (index + middle spread)  → Move cursor
  ✌️ Close two fingers (from V)         → Left click (easiest!)
  ☝️ Index finger only                   → Right click
  🤏 Pinch (thumb + index)               → Double click
  ✊ Fist (all fingers closed)           → Drag and drop
  🤏 Pinch with left hand                → Scroll
  🖐 Palm open (all fingers up)          → Stop / Idle

Examples:
  python -m src.main                                    # Use default webcam
  python -m src.main --source 1                         # Use external camera
  python -m src.main --source "rtsp://192.168.1.100:554/stream"  # Use RTSP
  python -m src.main --headless                         # Run without preview
        """
    )
    
    parser.add_argument(
        '--source', '-s',
        type=str,
        default=None,
        help='Video source: camera index (0, 1, ...) or RTSP URL'
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        default=None,
        help='Path to configuration file (default: config.yaml)'
    )
    
    parser.add_argument(
        '--headless',
        action='store_true',
        help='Run without preview window'
    )
    
    parser.add_argument(
        '--no-landmarks',
        action='store_true',
        help='Do not draw hand landmarks on preview'
    )
    
    parser.add_argument(
        '--flip',
        action='store_true',
        default=None,
        help='Enable horizontal flip (mirror mode)'
    )
    
    parser.add_argument(
        '--no-flip',
        action='store_true',
        help='Disable horizontal flip'
    )
    
    return parser.parse_args()


def draw_info_overlay(frame, gesture: Gesture, action: str, fps: float) -> None:
    """Draw information overlay on the frame.
    
    Args:
        frame: Image to draw on.
        gesture: Current detected gesture.
        action: Current action being performed.
        fps: Current frames per second.
    """
    height, width = frame.shape[:2]
    
    # Semi-transparent background for text
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (350, 100), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
    
    # Gesture name
    gesture_text = get_gesture_name(gesture)
    cv2.putText(frame, f"Gesture: {gesture_text}", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Action
    cv2.putText(frame, f"Action: {action}", (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # FPS
    cv2.putText(frame, f"FPS: {fps:.1f}", (20, 85),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    # Instructions at bottom
    cv2.putText(frame, "Press ESC or Enter to exit", (10, height - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)


def run_virtual_mouse(config_path: Optional[str] = None,
                     source: Optional[str] = None,
                     headless: bool = False,
                     show_landmarks: bool = True,
                     flip: Optional[bool] = None) -> int:
    """Run the virtual mouse application.
    
    Args:
        config_path: Path to configuration file.
        source: Video source override.
        headless: Run without preview window.
        show_landmarks: Draw hand landmarks.
        flip: Horizontal flip override.
    
    Returns:
        Exit code (0 for success).
    """
    # Load configuration
    config = load_config(config_path)
    
    # Apply command line overrides
    if source is not None:
        # Try to parse as integer
        try:
            config.camera.source = int(source)
        except ValueError:
            config.camera.source = source
    
    if flip is not None:
        config.camera.flip_horizontal = flip
    
    if headless:
        config.display.show_preview = False
    
    if not show_landmarks:
        config.display.show_landmarks = False
    
    print("=" * 50)
    print("  AI Virtual Mouse - Gesture Control System")
    print("=" * 50)
    print(f"  Video source: {config.camera.source}")
    print(f"  Preview: {'Enabled' if config.display.show_preview else 'Disabled'}")
    print("=" * 50)
    print("  Controls:")
    print("    ESC / Enter  : Exit")
    print("    ✌️ V-Gesture    : Move cursor")
    print("    ✌️ Close fingers: Left click")
    print("    ☝️ Index only   : Right click")
    print("    🤏 Pinch        : Double click")
    print("    ✊ Fist         : Drag")
    print("    🤏 Left pinch   : Scroll")
    print("    🖐 Palm         : Stop")
    print("=" * 50)
    
    # Initialize components
    video = VideoInput(config.camera)
    detector = GestureDetector(config.gesture)
    controller = MouseController(config.mouse)
    
    if not video.open():
        print("Error: Failed to open video source")
        return 1
    
    print("Virtual mouse started. Move your hand in front of the camera.")
    
    # FPS calculation
    fps = 0
    frame_count = 0
    start_time = cv2.getTickCount()
    
    try:
        while True:
            success, frame = video.read()
            
            if not success:
                if frame is None:
                    print("Warning: Empty frame received")
                    continue
                break
            
            # Process frame for hand detection
            hands_detected = detector.process(frame)
            
            # Get current gesture and handle it
            gesture = Gesture.PALM
            action = "No Hand"
            
            if hands_detected:
                gesture, hand = detector.get_gesture()
                position = hand.get_landmark_position(9) if hand else None
                action = controller.handle_gesture(gesture, position)
            else:
                controller.reset()
            
            # Calculate FPS
            frame_count += 1
            if frame_count >= 10:
                end_time = cv2.getTickCount()
                fps = 10 / ((end_time - start_time) / cv2.getTickFrequency())
                frame_count = 0
                start_time = cv2.getTickCount()
            
            # Display preview if enabled
            if config.display.show_preview:
                # Draw landmarks
                if config.display.show_landmarks:
                    detector.draw_landmarks(frame)
                
                # Draw info overlay
                if config.display.show_gesture:
                    draw_info_overlay(frame, gesture, action, fps)
                
                # Apply scale
                if config.display.preview_scale != 1.0:
                    new_width = int(frame.shape[1] * config.display.preview_scale)
                    new_height = int(frame.shape[0] * config.display.preview_scale)
                    frame = cv2.resize(frame, (new_width, new_height))
                
                cv2.imshow('Virtual Mouse', frame)
            
            # Check for exit keys
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == 13:  # ESC or Enter
                break
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        # Cleanup
        video.release()
        detector.release()
        controller.reset()
        cv2.destroyAllWindows()
        print("Virtual mouse stopped.")
    
    return 0


def main() -> int:
    """Main entry point.
    
    Returns:
        Exit code.
    """
    args = parse_args()
    
    # Determine flip setting
    flip = None
    if args.flip:
        flip = True
    elif args.no_flip:
        flip = False
    
    return run_virtual_mouse(
        config_path=args.config,
        source=args.source,
        headless=args.headless,
        show_landmarks=not args.no_landmarks,
        flip=flip
    )


if __name__ == '__main__':
    sys.exit(main())
