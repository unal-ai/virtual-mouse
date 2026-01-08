"""RTSP Service Discovery using mDNS/Zeroconf.

Discovers RTSP streams on the local network using mDNS service discovery.
Supports common service types used by IP cameras and streaming apps.
"""

import time
from typing import List, Dict, Optional, Callable
from dataclasses import dataclass

try:
    from zeroconf import ServiceBrowser, ServiceListener, Zeroconf, ServiceInfo
    ZEROCONF_AVAILABLE = True
except ImportError:
    ZEROCONF_AVAILABLE = False


@dataclass
class RTSPStream:
    """Represents a discovered RTSP stream."""
    name: str
    host: str
    port: int
    path: str = ""
    properties: Dict[str, str] = None
    
    @property
    def url(self) -> str:
        """Get the full RTSP URL."""
        if self.path:
            return f"rtsp://{self.host}:{self.port}/{self.path}"
        return f"rtsp://{self.host}:{self.port}"
    
    def __post_init__(self):
        if self.properties is None:
            self.properties = {}


class RTSPServiceListener:
    """Listener for RTSP service discovery events."""
    
    def __init__(self, on_found: Optional[Callable[[RTSPStream], None]] = None):
        self.streams: Dict[str, RTSPStream] = {}
        self.on_found = on_found
    
    def add_service(self, zc: "Zeroconf", service_type: str, name: str) -> None:
        """Called when a service is discovered."""
        info = zc.get_service_info(service_type, name)
        if info:
            self._process_service(info, name)
    
    def update_service(self, zc: "Zeroconf", service_type: str, name: str) -> None:
        """Called when a service is updated."""
        info = zc.get_service_info(service_type, name)
        if info:
            self._process_service(info, name)
    
    def remove_service(self, zc: "Zeroconf", service_type: str, name: str) -> None:
        """Called when a service is removed."""
        if name in self.streams:
            del self.streams[name]
            print(f"RTSP stream removed: {name}")
    
    def _process_service(self, info: "ServiceInfo", name: str) -> None:
        """Process discovered service info."""
        # Get host address
        addresses = info.parsed_addresses()
        if not addresses:
            return
        
        host = addresses[0]
        port = info.port
        
        # Parse properties
        properties = {}
        if info.properties:
            for key, value in info.properties.items():
                if isinstance(key, bytes):
                    key = key.decode('utf-8', errors='ignore')
                if isinstance(value, bytes):
                    value = value.decode('utf-8', errors='ignore')
                properties[key] = value
        
        # Try to get path from properties
        path = properties.get('path', properties.get('url', ''))
        
        stream = RTSPStream(
            name=name.replace('._rtsp._tcp.local.', ''),
            host=host,
            port=port,
            path=path,
            properties=properties
        )
        
        self.streams[name] = stream
        print(f"Found RTSP stream: {stream.name} at {stream.url}")
        
        if self.on_found:
            self.on_found(stream)


# Common mDNS service types for RTSP/video streams
RTSP_SERVICE_TYPES = [
    "_virtual-mouse._tcp.local.", # Custom type for this app
    "_rtsp._tcp.local.",          # Standard RTSP
    "_http._tcp.local.",          # HTTP (may have RTSP endpoints)
]


def discover_rtsp_streams(timeout: float = 3.0, 
                          on_found: Optional[Callable[[RTSPStream], None]] = None,
                          auto_connect: bool = False) -> List[RTSPStream]:
    """Discover RTSP streams on the local network.
    
    Args:
        timeout: How long to wait for discovery (seconds).
        on_found: Optional callback called for each discovered stream.
        auto_connect: If True, only returns streams matching the custom service type
                      or high-confidence matches promptly.
    
    Returns:
        List of discovered RTSPStream objects.
    """
    if not ZEROCONF_AVAILABLE:
        print("Error: zeroconf library not installed.")
        print("Install with: pip install zeroconf")
        return []
    
    print("Scanning for RTSP streams...")
    
    zc = Zeroconf()
    # If auto-connect, use a specific listener that can signal early exit
    # For now, we use the standard listener but prioritize results
    listener = RTSPServiceListener(on_found)
    
    browsers = []
    # If auto-connect, check the custom type FIRST and maybe exclusively?
    # For robust discovery, we check all, but prioritization happens in filtering
    for service_type in RTSP_SERVICE_TYPES:
        browser = ServiceBrowser(zc, service_type, listener)
        browsers.append(browser)
    
    # Wait for discovery
    start_time = time.time()
    while time.time() - start_time < timeout:
        if auto_connect:
            # Check if we found our specific target
            for name, stream in listener.streams.items():
                if "_virtual-mouse._tcp.local" in name:
                    print(f"Auto-connect target found: {name}")
                    zc.close()
                    return [stream]
        time.sleep(0.5)
    
    # Clean up
    zc.close()
    
    streams = list(listener.streams.values())
    if auto_connect:
        # Filter for only trusted streams
        trusted_streams = [s for s in streams if "_virtual-mouse._tcp.local" in s.name or "VirtualMouse" in s.name]
        if trusted_streams:
            return trusted_streams
        return [] # Don't auto-connect to random devices
        
    print(f"Discovery complete. Found {len(streams)} streams.")
    
    return streams


def main():
    """CLI entry point for RTSP discovery."""
    print("=" * 50)
    print("  RTSP Stream Discovery")
    print("=" * 50)
    
    if not ZEROCONF_AVAILABLE:
        print("\nError: zeroconf library not installed.")
        print("Install with: pip install zeroconf")
        return 1
    
    streams = discover_rtsp_streams(timeout=10.0)
    
    if streams:
        print("\nDiscovered Streams:")
        print("-" * 50)
        for i, stream in enumerate(streams, 1):
            print(f"{i}. {stream.name}")
            print(f"   URL: {stream.url}")
            if stream.properties:
                print(f"   Properties: {stream.properties}")
            print()
    else:
        print("\nNo RTSP streams found on the network.")
        print("Make sure your streaming app is running and broadcasting.")
    
    return 0


if __name__ == "__main__":
    exit(main())
