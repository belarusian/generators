#!/usr/bin/env python3
"""OCR server -- receives screenshots, returns text with bounding boxes.

Local Mac variant (MPS/Metal). Can also be run on a CUDA machine.

POST /ocr
    Body: PNG image bytes
    Returns: JSON list of detected text regions:
    [
        {"text": "Save", "box": [[x1,y1],[x2,y2],[x3,y3],[x4,y4]], "center": [cx, cy], "confidence": 0.97},
        ...
    ]

GET /health
    Returns: {"status": "ok"}
"""
import io
import json
import sys
import time
from http.server import HTTPServer, BaseHTTPRequestHandler

import numpy as np
from PIL import Image
import easyocr


reader = None


def get_reader():
    global reader
    if reader is None:
        print("[ocr] Loading EasyOCR (GPU)...")
        t0 = time.time()
        reader = easyocr.Reader(["en"], gpu=True)
        print(f"[ocr] Ready in {time.time() - t0:.1f}s")
    return reader


def run_ocr(image_bytes: bytes) -> list[dict]:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_np = np.array(img)

    results = get_reader().readtext(img_np)

    detections = []
    for (box, text, confidence) in results:
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        cx = int(sum(xs) / 4)
        cy = int(sum(ys) / 4)
        detections.append({
            "text": text,
            "box": [[int(p[0]), int(p[1])] for p in box],
            "center": [cx, cy],
            "confidence": round(float(confidence), 3),
        })

    return detections


class OCRHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path != "/ocr":
            self.send_error(404)
            return

        content_length = int(self.headers.get("Content-Length", 0))
        if content_length == 0:
            self.send_error(400, "No image data")
            return

        image_bytes = self.rfile.read(content_length)

        t0 = time.time()
        detections = run_ocr(image_bytes)
        elapsed = time.time() - t0

        print(f"[ocr] {len(detections)} regions in {elapsed:.2f}s")

        body = json.dumps(detections).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            body = b'{"status": "ok"}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_error(404)

    def log_message(self, format, *args):
        pass


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 9003
    get_reader()
    server = HTTPServer(("0.0.0.0", port), OCRHandler)
    print(f"[ocr] Listening on :{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[ocr] Shutting down.")
        server.server_close()


if __name__ == "__main__":
    main()
