import sys
import os
import threading
import io
import wave

os.environ["PYTHONIOENCODING"] = "utf-8"
if sys.stdout.encoding != "utf-8":
    sys.stdout = open(sys.stdout.fileno(), mode="w", encoding="utf-8", errors="replace", buffering=1)
    sys.stderr = open(sys.stderr.fileno(), mode="w", encoding="utf-8", errors="replace", buffering=1)

import json
import re
from flask import Flask, request, Response
from flask_cors import CORS
from deepface import DeepFace
import cv2
import numpy as np
import base64
import time
from dotenv import load_dotenv

load_dotenv()

import whisper as _whisper
import sounddevice as sd
from openai import OpenAI


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------

app = Flask(__name__)
CORS(app)

MAX_INPUT_DIM = 480

EMOTION_MAP = {
    "happy": "Joy",
    "sad": "Sadness",
    "angry": "Neutral",
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


# ---------------------------------------------------------------------------
# Shared state (thread-safe via lock)
# ---------------------------------------------------------------------------

_state_lock = threading.Lock()
_latest_hint = ""          # most recent follow-up hint from GPT
_latest_emotion = "Neutral"  # most recent emotion from /analyze (fed to GPT)
_last_transcript = ""      # last sentence Person 2 said
_mic_status = "not started"
_name_tag = ""             # detected name from "my name is X"

_NAME_PATTERN = re.compile(r"\bmy name is (\w+)", re.IGNORECASE)


def _set_hint(hint, transcript):
    global _latest_hint, _last_transcript
    with _state_lock:
        _latest_hint = hint
        _last_transcript = transcript


def _consume_hint():
    """Return the current hint and clear it (one-shot delivery)."""
    global _latest_hint
    with _state_lock:
        hint = _latest_hint
        _latest_hint = ""
        return hint


def _set_emotion(emotion):
    global _latest_emotion
    with _state_lock:
        _latest_emotion = emotion


def _get_emotion():
    with _state_lock:
        return _latest_emotion


def _set_mic_status(status):
    global _mic_status
    with _state_lock:
        _mic_status = status


def _set_name_tag(name):
    global _name_tag
    with _state_lock:
        _name_tag = name


def _get_name_tag():
    with _state_lock:
        return _name_tag


def _extract_name(transcript):
    """Check transcript for 'my name is X' and return X, or empty string."""
    m = _NAME_PATTERN.search(transcript)
    if m:
        return m.group(1).capitalize()
    return ""


# ---------------------------------------------------------------------------
# OpenAI GPT integration
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a gentle conversation coach for a person with autism who is wearing AR glasses.
You observe what the other person in the conversation just said, and their visible emotion.
Your job is to provide ONE short, simple follow-up hint that the wearer can say or ask next.

Rules:
- Keep it to one short sentence (under 15 words).
- Use warm, encouraging, simple language.
- Frame as a suggestion, not a command (e.g. "Maybe ask about..." or "You could say...").
- Match the emotional tone — if the person seems sad, be gentle; if happy, be upbeat.
- Never be judgmental or complex.
- Return ONLY valid JSON: {"follow_up_hint": "your suggestion here"}"""

_openai_client = None


def _get_openai_client():
    global _openai_client
    if _openai_client is None:
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            print("[GPT] WARNING: OPENAI_API_KEY not set. Hints will be empty.")
            return None
        _openai_client = OpenAI(api_key=api_key)
    return _openai_client


def _generate_hint(transcript, emotion):
    client = _get_openai_client()
    if client is None:
        return ""
    try:
        t0 = time.perf_counter()
        user_msg = f"Person 2 just said: \"{transcript}\"\nTheir detected emotion: {emotion}"
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.7,
            max_tokens=80,
            response_format={"type": "json_object"},
        )
        raw = resp.choices[0].message.content.strip()
        parsed = json.loads(raw)
        hint = parsed.get("follow_up_hint", "")
        dt = time.perf_counter() - t0
        print(f"[GPT] \"{hint}\" ({dt*1000:.0f}ms)")
        return hint
    except Exception as e:
        print(f"[GPT] Error: {e}")
        return ""


# ---------------------------------------------------------------------------
# MicTranscriber — background thread for lapel mic
# ---------------------------------------------------------------------------

_whisper_model = None

MIC_SAMPLE_RATE = 16000
MIC_CHUNK_SEC = 0.5
CALIBRATION_SEC = 4              # seconds of ambient noise to measure at startup
NOISE_MULTIPLIER = 2.5           # speech threshold = ambient_rms * this
MIN_SPEECH_THRESHOLD = 300       # floor so a silent room still works
SILENCE_DURATION_SEC = 1.5
MAX_BUFFER_SEC = 15              # safety valve: send even if still talking
_processing_lock = threading.Lock()
_speech_threshold = MIN_SPEECH_THRESHOLD  # set after calibration


def _rms(audio_chunk):
    """Root-mean-square energy of int16 audio."""
    if len(audio_chunk) == 0:
        return 0
    samples = audio_chunk.astype(np.float64)
    return np.sqrt(np.mean(samples ** 2))


def _transcribe_audio(pcm_int16, sample_rate):
    """Run Whisper on raw PCM16 mono audio."""
    global _whisper_model
    if _whisper_model is None:
        return ""
    audio_f32 = pcm_int16.astype(np.float32) / 32768.0
    import whisper.audio as _wa
    if sample_rate != _wa.SAMPLE_RATE:
        import torch
        audio_tensor = torch.from_numpy(audio_f32)
        audio_f32 = torch.nn.functional.interpolate(
            audio_tensor.unsqueeze(0).unsqueeze(0),
            scale_factor=_wa.SAMPLE_RATE / sample_rate,
            mode="linear",
        ).squeeze().numpy()
    result = _whisper_model.transcribe(
        audio_f32, language="en", fp16=False,
        condition_on_previous_text=False,
    )
    return result.get("text", "").strip()


def _calibrate_noise_floor(stream, chunk_samples):
    """Record a few seconds of ambient audio and return the average RMS."""
    global _speech_threshold
    cal_chunks = int(CALIBRATION_SEC / MIC_CHUNK_SEC)
    rms_values = []

    print(f"[Mic] Calibrating ambient noise ({CALIBRATION_SEC}s — don't talk into the mic)...")
    for _ in range(cal_chunks):
        data, overflowed = stream.read(chunk_samples)
        chunk_int16 = (data[:, 0] * 32767).astype(np.int16)
        rms_values.append(_rms(chunk_int16))

    ambient_rms = np.mean(rms_values) if rms_values else 0
    _speech_threshold = max(ambient_rms * NOISE_MULTIPLIER, MIN_SPEECH_THRESHOLD)
    print(f"[Mic] Ambient RMS: {ambient_rms:.0f} -> speech threshold: {_speech_threshold:.0f}")
    return _speech_threshold


def _mic_thread_fn(device_index=None):
    """Continuously capture from the lapel mic, detect silence boundaries,
    transcribe completed sentences, and generate GPT hints.
    Uses adaptive noise calibration so it works in loud rooms."""
    global _speech_threshold
    _set_mic_status("starting")
    chunk_samples = int(MIC_SAMPLE_RATE * MIC_CHUNK_SEC)
    silence_chunks_needed = int(SILENCE_DURATION_SEC / MIC_CHUNK_SEC)
    max_buffer_chunks = int(MAX_BUFFER_SEC / MIC_CHUNK_SEC)

    print(f"[Mic] Opening device={device_index}, rate={MIC_SAMPLE_RATE}, chunk={chunk_samples}")

    try:
        with sd.InputStream(
            device=device_index,
            samplerate=MIC_SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=chunk_samples,
        ) as stream:
            _calibrate_noise_floor(stream, chunk_samples)

            _set_mic_status("listening")
            print(f"[Mic] Listening for Person 2 (threshold={_speech_threshold:.0f})...")

            audio_buffer = []
            silence_count = 0
            has_speech = False
            total_chunks = 0

            while True:
                data, overflowed = stream.read(chunk_samples)
                if overflowed:
                    print("[Mic] WARNING: audio overflow")

                chunk_int16 = (data[:, 0] * 32767).astype(np.int16)
                energy = _rms(chunk_int16)

                if energy > _speech_threshold:
                    audio_buffer.append(chunk_int16)
                    silence_count = 0
                    has_speech = True
                    total_chunks += 1
                elif has_speech:
                    silence_count += 1
                    audio_buffer.append(chunk_int16)
                    total_chunks += 1

                    hit_silence = silence_count >= silence_chunks_needed
                    hit_max = total_chunks >= max_buffer_chunks

                    if hit_silence or hit_max:
                        if hit_max and not hit_silence:
                            print(f"[Mic] Max buffer reached ({MAX_BUFFER_SEC}s), sending anyway")
                        full_audio = np.concatenate(audio_buffer)
                        audio_buffer = []
                        silence_count = 0
                        has_speech = False
                        total_chunks = 0
                        threading.Thread(
                            target=_process_sentence,
                            args=(full_audio,),
                            daemon=True,
                        ).start()

    except Exception as e:
        _set_mic_status(f"error: {e}")
        print(f"[Mic] Error: {e}")


def _process_sentence(pcm_int16):
    """Transcribe a completed sentence and generate a GPT hint."""
    if not _processing_lock.acquire(blocking=False):
        return
    try:
        duration = len(pcm_int16) / MIC_SAMPLE_RATE
        if duration < 0.5:
            return

        t0 = time.perf_counter()
        text = _transcribe_audio(pcm_int16, MIC_SAMPLE_RATE)
        dt = time.perf_counter() - t0

        if not text or len(text.strip()) < 2:
            return

        print(f"[Mic] Person 2: \"{text}\" ({dt*1000:.0f}ms, {duration:.1f}s audio)")

        name = _extract_name(text)
        if name:
            _set_name_tag(name)
            print(f"[Mic] Name detected: \"{name}\"")

        emotion = _get_emotion()
        hint = _generate_hint(text, emotion)
        if hint:
            _set_hint(hint, text)
    finally:
        _processing_lock.release()


# ---------------------------------------------------------------------------
# Flask routes
# ---------------------------------------------------------------------------

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
            detector_backend="mtcnn",
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

        _set_emotion(mapped)

        t_total = time.perf_counter() - t_start

        print(
            f"[{time.strftime('%H:%M:%S')}] "
            f"{emoji} {mapped} ({confidence:.0%}) "
            f"[raw:{dominant}] | "
            f"decode:{t_decode*1000:.0f}ms model:{t_model*1000:.0f}ms total:{t_total*1000:.0f}ms"
        )

        resp_data = {
            "emotion": mapped,
            "confidence": confidence,
            "label": f"{emoji} {mapped}",
            "latency_ms": round(t_total * 1000),
        }

        hint = _consume_hint()
        if hint:
            resp_data["follow_up_hint"] = hint

        name_tag = _get_name_tag()
        if name_tag:
            resp_data["name_tag"] = name_tag

        return json_response(resp_data)

    except Exception as e:
        print(f"Error during analysis: {e}")
        return json_response({"error": str(e)}, 500)


@app.route("/mic/status", methods=["GET"])
def mic_status():
    with _state_lock:
        return json_response({
            "mic_status": _mic_status,
            "last_transcript": _last_transcript,
            "latest_hint": _latest_hint,
            "latest_emotion": _latest_emotion,
        })


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("CareXR Backend Server")
    print("=" * 60)

    print("\n[1/3] Loading Whisper model (base)...")
    _whisper_model = _whisper.load_model("base")
    print("  Whisper ready.")

    print("\n[2/3] Warming up DeepFace (mtcnn)...")
    try:
        dummy = np.zeros((48, 48, 3), dtype=np.uint8)
        DeepFace.analyze(
            dummy, actions=["emotion"], enforce_detection=False,
            detector_backend="mtcnn", silent=True,
        )
        print("  DeepFace ready.")
    except Exception as e:
        print(f"  Warmup note: {e}")

    print("\n[3/3] Starting lapel mic listener...")
    print("  Available audio devices:")
    print(sd.query_devices())
    mic_device = os.environ.get("MIC_DEVICE_INDEX")
    mic_idx = int(mic_device) if mic_device else None
    if mic_idx is not None:
        print(f"  Using device index: {mic_idx}")
    else:
        print("  Using default input device (set MIC_DEVICE_INDEX in .env to override)")

    mic_thread = threading.Thread(target=_mic_thread_fn, args=(mic_idx,), daemon=True)
    mic_thread.start()

    api_key = os.environ.get("OPENAI_API_KEY", "")
    print(f"\n  OpenAI API key: {'configured' if api_key else 'NOT SET (hints disabled)'}")

    print("\nServer ready on 0.0.0.0:5001")
    print("  GET  /health      - connectivity check")
    print("  POST /analyze     - emotion detection (+ follow_up_hint)")
    print("  GET  /mic/status  - mic & hint debug info")
    print("=" * 60)

    try:
        from waitress import serve
        print("Using waitress (production server)")
        serve(app, host="0.0.0.0", port=5001, threads=4)
    except ImportError:
        print("Using Flask dev server (install waitress for production)")
        app.run(host="0.0.0.0", port=5001, debug=False, threaded=True)
