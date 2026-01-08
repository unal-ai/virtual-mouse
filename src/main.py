"""Main entry point for Virtual Mouse application.

Provides CLI interface and main processing loop.
"""

import argparse
import sys
import cv2
import time
from typing import Optional

from .config import load_config, CameraConfig
from .video_input import VideoInput
from .gestures import GestureDetector, get_gesture_name, Gesture, HandRecognizer
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
    
    parser.add_argument(
        '--demo', '--trackpad',
        action='store_true',
        help='Enable trackpad mode (simplifed gestures, relative movement)'
    )
    
    parser.add_argument(
        '--discover',
        action='store_true',
        help='Scan network for RTSP streams (requires zeroconf)'
    )
    
    parser.add_argument(
        '--auto-connect',
        action='store_true',
        help='Automatically connect to "Virtual Mouse" RTSP stream if found'
    )

    parser.add_argument(
        '--rtsp-wait',
        action='store_true',
        help='Wait indefinitely for "Virtual Mouse" RTSP stream (implies --auto-connect)'
    )
    
    return parser.parse_args()


def draw_info_overlay(frame, gesture: Gesture, action: str, fps: float, hand: Optional[HandRecognizer] = None) -> None:
    """Draw information overlay on the frame.
    
    Args:
        frame: Image to draw on.
        gesture: Current detected gesture.
        action: Current action being performed.
        fps: Current frames per second.
        hand: HandRecognizer instance for debug info.
    """
    height, width = frame.shape[:2]
    
    # Semi-transparent background for text (larger for debug info)
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (450, 160), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
    
    # Gesture name (Stable)
    gesture_text = get_gesture_name(gesture)
    cv2.putText(frame, f"Gesture: {gesture_text}", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Action
    cv2.putText(frame, f"Action: {action}", (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # FPS
    cv2.putText(frame, f"FPS: {fps:.1f}", (20, 85),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                
    if hand:
        # Debug Info
        score = hand.score
        raw_g = get_gesture_name(hand.raw_gesture).split('(')[0].strip() # Short name
        fingers = f"{hand.finger_state:05b}" # Binary representation
        
        cv2.putText(frame, f"Score: {score:.2f} | P: {fingers}", (20, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        cv2.putText(frame, f"Raw: {raw_g}", (20, 135),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

    # Instructions at bottom...
    cv2.putText(frame, "Press ESC or Enter to exit", (10, height - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)


def run_virtual_mouse(config_path: Optional[str] = None,
                     source: Optional[str] = None,
                     headless: bool = False,
                     show_landmarks: bool = True,
                     flip: Optional[bool] = None,
                     demo_mode: bool = False) -> int:
    """Run the virtual mouse application.
    
    Args:
        config_path: Path to configuration file.
        source: Video source override.
        headless: Run without preview window.
        show_landmarks: Draw hand landmarks.
        flip: Horizontal flip override.
        demo_mode: Enable trackpad/demo mode.
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
        
    if demo_mode:
        config.mouse.trackpad_mode = True
        config.mouse.trackpad_sensitivity = 2.0 # Default High sensitivity for trackpad
    
    if headless:
        config.display.show_preview = False
    
    if not show_landmarks:
        config.display.show_landmarks = False
    
    print("=" * 50)
    if config.mouse.trackpad_mode:
        print("  AI Virtual Mouse - Trackpad Mode (Demo)")
    else:
        print("  AI Virtual Mouse - Gesture Control System")
    print("=" * 50)
    print(f"  Video source: {config.camera.source}")
    print(f"  Preview: {'Enabled' if config.display.show_preview else 'Disabled'}")
    print("=" * 50)
    print("  Controls:")
    print("    ESC / Enter  : Exit")
    if config.mouse.trackpad_mode:
        print("    2 Fingers    : Move cursor")
        print("    2 Fingers ✌️  : Close to Click")
        print("    3 Fingers    : Hold to Drag")
        print("    5 Fingers    : Back")
    else:
        print("    V-Gesture    : Move cursor")
        print("    Close fingers: Left click")
        print("    Index only   : Right click")
        print("    Pinch        : Double click")
        print("    Fist         : Drag")
        print("    Left pinch   : Scroll")
        print("    Palm         : Stop")
    print("=" * 50)
    
    # Create smart discovery callback for RTSP self-healing
    smart_discovery = None
    if config.mouse.trackpad_mode: 
        def smart_discovery_cb() -> Optional[str]:
            # This callback is triggered by VideoInput when reconnection fails repeatedly
            try:
                from .rtsp_discovery import discover_rtsp_streams
                # Check for new stream location
                streams = discover_rtsp_streams(timeout=2.0, auto_connect=True)
                if streams:
                    return streams[0].url
            except ImportError:
                pass
            return None
        smart_discovery = smart_discovery_cb

    # Initialize components
    video = VideoInput(config.camera, on_reconnect_fail=smart_discovery)
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
                # For RTSP streams, the reconnection logic in video.read() handles it
                # Just continue the loop and wait for frames
                if video._is_rtsp:
                    # Show reconnecting status
                    if config.display.show_preview:
                        status_frame = frame if frame is not None else None
                        if status_frame is None:
                            # Create a blank frame for status display
                            import numpy as np
                            status_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                            cv2.putText(status_frame, "Reconnecting to RTSP...", (50, 240),
                                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                            cv2.imshow('Virtual Mouse', status_frame)
                            cv2.waitKey(100)
                    continue
                else:
                    print("Warning: Frame read failed")
                    break
            
            # Submit frame for async processing (non-blocking)
            detector.process_async(frame)
            
            # Get latest detection result (may be from previous frame)
            hands_detected, gesture, hand = detector.get_latest_result()
            
            # Handle gesture
            action = "No Hand"
            
            if hands_detected:
                # Use 2-finger position in trackpad mode for stable tracking
                if config.mouse.trackpad_mode:
                    position = hand.get_two_finger_position() if hand else None
                else:
                    position = hand.get_landmark_position(9) if hand else None
                action = controller.handle_gesture(gesture, position)
            else:
                controller.reset()  # Reset trackpad state when hand lost
            
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
                    draw_info_overlay(frame, gesture, action, fps, hand)
                
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
    
    # Handle --discover separately
    if args.discover:
        from .rtsp_discovery import main as discover_main
        return discover_main()

    # Handle --auto-connect or --rtsp-wait
    if args.auto_connect or args.rtsp_wait:
        print("Auto-connecting to Virtual Mouse stream...")
        try:
            from .rtsp_discovery import discover_rtsp_streams
            
            while True:
                streams = discover_rtsp_streams(timeout=2.0, auto_connect=True)
                if streams:
                    print(f"Found trusted stream: {streams[0].name}")
                    args.source = streams[0].url
                    break
                
                if not args.rtsp_wait:
                    print("No Virtual Mouse stream found. Falling back to default.")
                    break
                
                print("Waiting for Virtual Mouse stream... (Scanning)")
                time.sleep(1.0)
                
        except ImportError:
            print("Warning: zeroconf not installed, skipping auto-connect.")

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
        flip=flip,
        demo_mode=args.demo
    )


if __name__ == '__main__':
    sys.exit(main())
