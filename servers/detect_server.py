#!/usr/bin/env python3
"""Object detection server -- Grounding DINO.

Local Mac variant (MPS/Metal). Can also be run on a CUDA machine.

POST /detect
    Body: JSON {"image": "<base64 PNG>", "query": "red close button"}
    Returns: JSON list of detections:
    [
        {"label": "close button", "box": [x1, y1, x2, y2], "center": [cx, cy], "confidence": 0.92},
        ...
    ]

POST /detect_raw
    Body: PNG image bytes
    Headers: X-Query: "red close button"
    Returns: same as /detect

GET /health
    Returns: {"status": "ok", "model": "grounding-dino-base"}
"""
import base64
import io
import json
import os
import sys
import time
from http.server import HTTPServer, BaseHTTPRequestHandler

os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TRANSFORMERS_NO_FLAX"] = "1"

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection


MODEL_ID = "IDEA-Research/grounding-dino-base"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
BOX_THRESHOLD = 0.3
TEXT_THRESHOLD = 0.25
model = None
processor = None


def get_model():
    global model, processor
    if model is None:
        print(f"[detect] Loading {MODEL_ID} on {DEVICE}...")
        t0 = time.time()
        processor = AutoProcessor.from_pretrained(MODEL_ID)
        model = AutoModelForZeroShotObjectDetection.from_pretrained(
            MODEL_ID
        ).to(DEVICE)
        print(f"[detect] Ready in {time.time() - t0:.1f}s")
    return model, processor


def run_detection(image_bytes: bytes, query: str) -> list[dict]:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    w, h = img.size
    mdl, proc = get_model()

    text = query.lower().strip()
    if not text.endswith("."):
        text += "."

    inputs = proc(images=img, text=text, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        outputs = mdl(**inputs)

    results = proc.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD,
        target_sizes=[(h, w)],
    )

    detections = []
    result = results[0]
    boxes = result["boxes"]
    scores = result["scores"]
    labels = result.get("text_labels", result.get("labels", []))

    for i in range(len(boxes)):
        x1, y1, x2, y2 = [int(c) for c in boxes[i].tolist()]
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        label = labels[i] if i < len(labels) else query
        if isinstance(label, str):
            label = label.strip().rstrip(".")
        else:
            label = query
        detections.append({
            "label": label,
            "box": [x1, y1, x2, y2],
            "center": [cx, cy],
            "confidence": round(scores[i].item(), 3),
        })

    return detections


def run_open_detection(image_bytes: bytes) -> list[dict]:
    generic_query = "button. icon. menu. link. image. text field. checkbox."
    return run_detection(image_bytes, generic_query)


class DetectHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", 0))
        if content_length == 0:
            self.send_error(400, "No data")
            return

        if self.path == "/detect":
            raw = self.rfile.read(content_length)
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                self.send_error(400, "Invalid JSON")
                return

            image_b64 = payload.get("image", "")
            query = payload.get("query", "")

            if not image_b64:
                self.send_error(400, "Missing 'image' field (base64 PNG)")
                return

            image_bytes = base64.b64decode(image_b64)

            t0 = time.time()
            if query:
                detections = run_detection(image_bytes, query)
            else:
                detections = run_open_detection(image_bytes)
            elapsed = time.time() - t0

            mode = f"query='{query}'" if query else "open detection"
            print(f"[detect] {len(detections)} boxes in {elapsed:.2f}s ({mode})")

        elif self.path == "/detect_raw":
            image_bytes = self.rfile.read(content_length)
            query = self.headers.get("X-Query", "")

            t0 = time.time()
            if query:
                detections = run_detection(image_bytes, query)
            else:
                detections = run_open_detection(image_bytes)
            elapsed = time.time() - t0

            mode = f"query='{query}'" if query else "open detection"
            print(f"[detect] {len(detections)} boxes in {elapsed:.2f}s ({mode})")

        else:
            self.send_error(404)
            return

        body = json.dumps(detections).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            body = json.dumps({"status": "ok", "model": MODEL_ID}).encode()
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
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 9004
    get_model()
    server = HTTPServer(("0.0.0.0", port), DetectHandler)
    print(f"[detect] Listening on :{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[detect] Shutting down.")
        server.server_close()


if __name__ == "__main__":
    main()
