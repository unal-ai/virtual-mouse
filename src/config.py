"""Configuration loader for Virtual Mouse."""

import os
import yaml
from dataclasses import dataclass, field
from typing import Union, Optional


@dataclass
class CameraConfig:
    source: Union[int, str] = 0
    width: int = 640
    height: int = 480
    flip_horizontal: bool = True


@dataclass
class GestureConfig:
    detection_confidence: float = 0.7
    tracking_confidence: float = 0.5
    smoothing_factor: float = 0.3
    stability_frames: int = 4


@dataclass
class ScreenRegion:
    enabled: bool = False
    x: int = 0
    y: int = 0
    width: int = 1920
    height: int = 1080


@dataclass
class MouseConfig:
    screen_region: ScreenRegion = field(default_factory=ScreenRegion)
    sensitivity: float = 1.0
    click_debounce: float = 0.3


@dataclass
class DisplayConfig:
    show_preview: bool = True
    show_landmarks: bool = True
    show_gesture: bool = True
    preview_scale: float = 1.0


@dataclass
class Config:
    camera: CameraConfig = field(default_factory=CameraConfig)
    gesture: GestureConfig = field(default_factory=GestureConfig)
    display: DisplayConfig = field(default_factory=DisplayConfig)
    mouse: MouseConfig = field(default_factory=MouseConfig)


def load_config(config_path: Optional[str] = None) -> Config:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to config file. If None, looks for config.yaml
                    in the project root directory.
    
    Returns:
        Config object with loaded settings.
    """
    if config_path is None:
        # Look for config.yaml in project root
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.path.join(project_root, "config.yaml")
    
    config = Config()
    
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                data = yaml.safe_load(f) or {}
            
            # Load camera settings
            if 'camera' in data:
                cam = data['camera']
                config.camera = CameraConfig(
                    source=cam.get('source', 0),
                    width=cam.get('width', 640),
                    height=cam.get('height', 480),
                    flip_horizontal=cam.get('flip_horizontal', True)
                )
            
            # Load gesture settings
            if 'gesture' in data:
                gest = data['gesture']
                config.gesture = GestureConfig(
                    detection_confidence=gest.get('detection_confidence', 0.7),
                    tracking_confidence=gest.get('tracking_confidence', 0.5),
                    smoothing_factor=gest.get('smoothing_factor', 0.3),
                    stability_frames=gest.get('stability_frames', 4)
                )
            
            # Load display settings
            if 'display' in data:
                disp = data['display']
                config.display = DisplayConfig(
                    show_preview=disp.get('show_preview', True),
                    show_landmarks=disp.get('show_landmarks', True),
                    show_gesture=disp.get('show_gesture', True),
                    preview_scale=disp.get('preview_scale', 1.0)
                )
            
            # Load mouse settings
            if 'mouse' in data:
                mouse = data['mouse']
                region_data = mouse.get('screen_region', {})
                screen_region = ScreenRegion(
                    enabled=region_data.get('enabled', False),
                    x=region_data.get('x', 0),
                    y=region_data.get('y', 0),
                    width=region_data.get('width', 1920),
                    height=region_data.get('height', 1080)
                )
                config.mouse = MouseConfig(
                    screen_region=screen_region,
                    sensitivity=mouse.get('sensitivity', 1.0),
                    click_debounce=mouse.get('click_debounce', 0.3)
                )
                
        except Exception as e:
            print(f"Warning: Error loading config file: {e}")
            print("Using default configuration.")
    
    return config
