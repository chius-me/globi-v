#!/usr/bin/env python3
"""Camera capture server for Globi-V DuoS board.
Runs on WSL host. Grabs RTSP frames from the board and returns JPEG.
Board accesses via: http://172.21.150.38:8899/capture
"""

import http.server
import subprocess
import tempfile
import os
import sys
from pathlib import Path

RTSP_URL = "rtsp://192.168.42.1/h264"
HOST = "0.0.0.0"
PORT = 8899
CACHE_DIR = Path("/tmp/globi_camera")

class CaptureHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/capture" or self.path == "/capture.jpg":
            self.capture()
        elif self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"ok")
        else:
            self.send_response(404)
            self.end_headers()
    
    def capture(self):
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        jpg_path = CACHE_DIR / "latest.jpg"
        
        try:
            subprocess.run([
                "ffmpeg", "-y", "-loglevel", "error",
                "-rtsp_transport", "tcp",
                "-i", RTSP_URL,
                "-frames:v", "1",
                "-q:v", "3",
                str(jpg_path)
            ], timeout=15, check=True)
            
            if jpg_path.stat().st_size > 100:
                data = jpg_path.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("X-Camera-Time", str(jpg_path.stat().st_mtime))
                self.end_headers()
                self.wfile.write(data)
            else:
                self.send_error(502, "Empty frame from camera")
        except subprocess.TimeoutExpired:
            self.send_error(504, "RTSP grab timed out")
        except subprocess.CalledProcessError as e:
            self.send_error(502, f"ffmpeg error: {e}")
        except Exception as e:
            self.send_error(500, str(e))
    
    def log_message(self, format, *args):
        print(f"[{self.log_date_time_string()}] {args[0]}")

if __name__ == "__main__":
    print(f"📷 Globi Camera Server")
    print(f"   RTSP: {RTSP_URL}")
    print(f"   Listen: http://{HOST}:{PORT}/capture")
    server = http.server.HTTPServer((HOST, PORT), CaptureHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.shutdown()
