import sys
import os

os.environ["PYTHONIOENCODING"] = "utf-8"
if sys.stdout.encoding != "utf-8":
    sys.stdout = open(sys.stdout.fileno(), mode="w", encoding="utf-8", errors="replace", buffering=1)
    sys.stderr = open(sys.stderr.fileno(), mode="w", encoding="utf-8", errors="replace", buffering=1)

import json
from flask import Flask, request, Response
from flask_cors import CORS
from deepface import DeepFace
import cv2
import numpy as np
import base64
import time


class NumpySafeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def json_response(data, status=200):
    body = json.dumps(data, cls=NumpySafeEncoder, separators=(",", ":"))
    return Response(body, status=status, mimetype="application/json")


app = Flask(__name__)
CORS(app)

MAX_INPUT_DIM = 320

EMOTION_MAP = {
    "happy": "Joy",
    "sad": "Sadness",
    "angry": "Anger",
    "neutral": "Neutral",
    "disgust": "Neutral",
    "fear": "Neutral",
    "surprise": "Neutral",
}

EMOJI_MAP = {
    "Joy": "\U0001F60A",
    "Sadness": "\U0001F622",
    "Anger": "\U0001F621",
    "Neutral": "\U0001F610",
}


def decode_and_downscale(b64_string):
    if "," in b64_string:
        b64_string = b64_string.split(",")[-1]
    image_data = base64.b64decode(b64_string)
    np_arr = np.frombuffer(image_data, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if frame is None:
        return None

    h, w = frame.shape[:2]
    if max(h, w) > MAX_INPUT_DIM:
        scale = MAX_INPUT_DIM / max(h, w)
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    return frame


@app.route("/health", methods=["GET"])
def health():
    return json_response({"status": "ok", "timestamp": float(time.time())})


@app.route("/analyze", methods=["POST"])
def analyze():
    t_start = time.perf_counter()

    data = request.get_json()
    if not data or "image" not in data:
        return json_response({"error": "No base64 image found in request"}, 400)

    try:
        t0 = time.perf_counter()
        frame = decode_and_downscale(data["image"])
        t_decode = time.perf_counter() - t0

        if frame is None:
            return json_response({"error": "Failed to decode image"}, 400)

        t0 = time.perf_counter()
        results = DeepFace.analyze(
            frame,
            actions=["emotion"],
            enforce_detection=False,
            detector_backend="opencv",
            silent=True,
        )
        t_model = time.perf_counter() - t0

        if not results:
            return json_response({"emotion": "none", "confidence": 0.0})

        result = results[0] if isinstance(results, list) else results
        dominant = str(result.get("dominant_emotion", "neutral"))
        raw_scores = {k: float(v) for k, v in result.get("emotion", {}).items()}
        confidence = round(raw_scores.get(dominant, 0.0) / 100.0, 3)

        mapped = EMOTION_MAP.get(dominant, "Neutral")
        emoji = EMOJI_MAP.get(mapped, "")

        t_total = time.perf_counter() - t_start

        print(
            f"[{time.strftime('%H:%M:%S')}] "
            f"{emoji} {mapped} ({confidence:.0%}) "
            f"[raw:{dominant}] | "
            f"decode:{t_decode*1000:.0f}ms model:{t_model*1000:.0f}ms total:{t_total*1000:.0f}ms"
        )

        return json_response({
            "emotion": mapped,
            "confidence": confidence,
            "label": f"{emoji} {mapped}",
            "latency_ms": round(t_total * 1000),
        })

    except Exception as e:
        print(f"Error during analysis: {e}")
        return json_response({"error": str(e)}, 500)


if __name__ == "__main__":
    print("DeepFace Emotion Server starting on 0.0.0.0:5001")
    print(f"  Max input: {MAX_INPUT_DIM}px | Detector: opencv")
    print("Warming up DeepFace model...")

    try:
        dummy = np.zeros((48, 48, 3), dtype=np.uint8)
        DeepFace.analyze(
            dummy, actions=["emotion"], enforce_detection=False,
            detector_backend="opencv", silent=True,
        )
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Warmup note: {e} (will resolve on first real image)")

    print("Server ready. Endpoints:")
    print("  GET  /health   - connectivity check")
    print("  POST /analyze  - emotion detection")
    app.run(host="0.0.0.0", port=5001, debug=False, threaded=True)
