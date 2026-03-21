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
from collections import deque


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

# --- Configuration ---
MAX_INPUT_DIM = 320  # downscale camera frame to this before analysis
SMOOTHING_ALPHA = 0.45  # EMA blending factor (higher = more responsive, lower = more stable)
HISTORY_SIZE = 5  # number of frames for majority-vote smoothing
DETECTOR_BACKEND = "opencv"  # fastest face detector; skip = no detection at all

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

TARGET_EMOTIONS = ["Joy", "Sadness", "Anger", "Neutral"]

# --- Temporal smoothing state ---
ema_scores = {e: 0.0 for e in TARGET_EMOTIONS}
recent_votes = deque(maxlen=HISTORY_SIZE)


def reset_smoothing():
    global ema_scores, recent_votes
    ema_scores = {e: 0.0 for e in TARGET_EMOTIONS}
    recent_votes.clear()


def smooth_prediction(raw_mapped_scores):
    """
    Combines exponential moving average with majority vote.
    Returns (emotion, confidence) after smoothing.
    """
    global ema_scores

    for e in TARGET_EMOTIONS:
        raw = raw_mapped_scores.get(e, 0.0)
        ema_scores[e] = SMOOTHING_ALPHA * raw + (1.0 - SMOOTHING_ALPHA) * ema_scores[e]

    ema_best = max(TARGET_EMOTIONS, key=lambda e: ema_scores[e])
    recent_votes.append(ema_best)

    # Majority vote over recent window for final stability
    vote_counts = {}
    for v in recent_votes:
        vote_counts[v] = vote_counts.get(v, 0) + 1
    final_emotion = max(vote_counts, key=vote_counts.get)

    return final_emotion, round(ema_scores[final_emotion], 3)


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


@app.route("/reset", methods=["POST"])
def reset():
    """Call when face is lost to reset smoothing state."""
    reset_smoothing()
    return json_response({"status": "reset"})


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
            detector_backend=DETECTOR_BACKEND,
            silent=True,
        )
        t_model = time.perf_counter() - t0

        if not results:
            return json_response({"emotion": "none", "confidence": 0.0})

        result = results[0] if isinstance(results, list) else results
        raw_scores = {k: float(v) for k, v in result.get("emotion", {}).items()}

        # Map DeepFace's 7 emotions to our 4 target emotions
        mapped_scores = {e: 0.0 for e in TARGET_EMOTIONS}
        for deepface_emotion, score in raw_scores.items():
            target = EMOTION_MAP.get(deepface_emotion, "Neutral")
            mapped_scores[target] += score / 100.0

        # Apply temporal smoothing
        emotion, confidence = smooth_prediction(mapped_scores)
        emoji = EMOJI_MAP.get(emotion, "")

        t_total = time.perf_counter() - t_start

        print(
            f"[{time.strftime('%H:%M:%S')}] "
            f"{emoji} {emotion} ({confidence:.0%}) | "
            f"decode:{t_decode*1000:.0f}ms model:{t_model*1000:.0f}ms total:{t_total*1000:.0f}ms | "
            f"frame:{frame.shape[1]}x{frame.shape[0]}"
        )

        return json_response({
            "emotion": emotion,
            "confidence": confidence,
            "label": f"{emoji} {emotion}",
            "latency_ms": round(t_total * 1000),
        })

    except Exception as e:
        print(f"Error during analysis: {e}")
        return json_response({"error": str(e)}, 500)


if __name__ == "__main__":
    print("DeepFace Emotion Server starting on 0.0.0.0:5001")
    print(f"  Detector: {DETECTOR_BACKEND} | Max input: {MAX_INPUT_DIM}px")
    print(f"  Smoothing: alpha={SMOOTHING_ALPHA}, window={HISTORY_SIZE}")
    print("Warming up DeepFace model...")

    try:
        dummy = np.zeros((48, 48, 3), dtype=np.uint8)
        DeepFace.analyze(
            dummy, actions=["emotion"], enforce_detection=False,
            detector_backend=DETECTOR_BACKEND, silent=True,
        )
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Warmup note: {e} (will resolve on first real image)")

    print("Server ready. Endpoints:")
    print("  GET  /health   - connectivity check")
    print("  POST /analyze  - emotion detection")
    print("  POST /reset    - reset smoothing state (call on face lost)")
    app.run(host="0.0.0.0", port=5001, debug=False, threaded=True)
