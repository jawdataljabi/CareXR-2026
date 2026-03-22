# CareXR-2026 Project Summary

## Overview

CareXR-2026 is a **Snap Spectacles AR application** built in **Lens Studio** that provides real-time emotion detection for the person in front of the glasses wearer. The project uses an external Python backend for facial emotion recognition via DeepFace, with communication tunneled through ngrok.

A reference project (**CueTips**) in the `Other Projects/` directory served as architectural inspiration, particularly for the ngrok tunneling pattern and VoiceML transcription approach.

---

## Architecture

```
Spectacles (Lens)                    Local Machine
┌─────────────────────┐              ┌──────────────────────────────┐
│ BackendEmotion      │  HTTPS/ngrok │ emotion_server.py (Flask)    │
│ Detector.js         │─────────────>│  - DeepFace emotion analysis │
│  - CameraModule     │              │  - MTCNN face detector       │
│  - Base64 encode    │<─────────────│  - waitress WSGI server      │
│  - InternetModule   │  JSON resp   │  - Port 5001                 │
│    .fetch()         │              └──────────────────────────────┘
│                     │                        ▲
│ EmotionLabel        │              ┌─────────┴──────────┐
│  (Text3D above head)│              │ ngrok http 5001    │
└─────────────────────┘              │ Public HTTPS URL   │
                                     └────────────────────┘
```

---

## Key Files

### Lens Studio (Assets/)

| File | Purpose | Status |
|------|---------|--------|
| `BackendEmotionDetector.js` | Captures camera frames on Spectacles, sends Base64 images to the Python backend, displays emotion as Text3D above detected face | **Active** (Spectacles only) |
| `EmotionDetector.js` | Original blend-shape-based emotion detector using face mesh weights | **Disabled** (blend shapes return near-zero on Spectacles) |

### Backend (Backend/)

| File | Purpose |
|------|---------|
| `emotion_server.py` | Flask server with DeepFace for emotion recognition. Uses MTCNN detector, waitress production server, custom NumpySafeEncoder for JSON serialization |
| `requirements.txt` | Python dependencies: flask, flask-cors, deepface, tf-keras, opencv-python-headless, numpy, mtcnn, waitress, python-dotenv |

### Configuration

| File | Purpose |
|------|---------|
| `.gitignore` | Excludes `.env` files, Python cache, node_modules, debug images, temp audio, and Lens Studio cache |

---

## Lens Studio Scene Graph

```
Scene
├── Camera Object (perspective camera, layer 1)
│   └── Effects
│       └── Head Binding (face tracking, faceIndex 0)
│           ├── EmotionFaceMesh (RenderMeshVisual, blend shapes enabled, layer 0/hidden)
│           └── EmotionLabel (Text3D, size 72, positioned x:-8 y:14, scale 1.2)
├── Lighting
│   ├── Envmap (environment light)
│   └── Light (directional light)
├── Orthographic Camera (layer 1048576, for 2D UI)
│   └── Full Frame Region (ScreenTransform + ScreenRegion)
├── EmotionDetectorController [DISABLED] (EmotionDetector.js - blend shape approach)
└── BackendEmotionController [ENABLED] (BackendEmotionDetector.js - DeepFace backend approach)
```

---

## Emotion Detection Pipeline

### On Spectacles (BackendEmotionDetector.js)

1. **Camera Init**: Uses `CameraModule.createCameraRequest()` to access the Spectacles camera
2. **Face Tracking**: Listens for `FaceFoundEvent` / `FaceLostEvent` (faceIndex 0)
3. **Frame Capture**: On each new camera frame (throttled to 0.1s interval), encodes the frame as Base64 JPEG using `Base64.encodeTextureAsync()` with `CompressionQuality.LowQuality`
4. **Backend Request**: Sends POST to ngrok URL with `{ image: <base64> }` and `ngrok-skip-browser-warning: true` header
5. **Display**: Shows emotion label (emoji + name) on the EmotionLabel Text3D above the tracked head

### Backend (emotion_server.py)

1. **Decode**: Base64 decode + downscale to max 480px
2. **Analyze**: `DeepFace.analyze()` with MTCNN detector backend
3. **Map**: Raw emotions mapped to 4 categories:
   - `happy` → Joy, `sad` → Sadness, `angry` → Anger
   - `neutral`, `disgust`, `fear`, `surprise` → Neutral
4. **Response**: Returns `{ emotion, confidence, label, latency_ms }`
5. **Server**: Runs on waitress (production WSGI) with 4 threads on port 5001

---

## How to Run

### 1. Start the Backend

```powershell
cd Backend
pip install -r requirements.txt
python emotion_server.py
```

Expected output:
```
DeepFace Emotion Server starting on 0.0.0.0:5001
Model loaded successfully.
Server ready.
Using waitress (production server)
```

### 2. Start ngrok Tunnel

```powershell
ngrok http 5001
```

Note the `Forwarding` URL (e.g., `https://xxx.ngrok-free.dev`).

### 3. Configure Lens Studio

In Lens Studio, select `BackendEmotionController` in the scene hierarchy. In the Inspector panel, set the `backendUrl` property to your ngrok URL + `/analyze`:
```
https://YOUR-NGROK-URL.ngrok-free.dev/analyze
```

### 4. Push to Spectacles

Use Lens Studio's "Push to Device" to deploy the lens to connected Spectacles.

### 5. Kill/Restart the Backend

```powershell
Get-Process python -ErrorAction SilentlyContinue | Stop-Process -Force
python emotion_server.py
```

---

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `Backend error (0):` on Spectacles | Backend server unresponsive or ngrok disconnected | Kill all Python processes and restart `emotion_server.py`. Verify ngrok is running. Re-push the lens. |
| `Backend error (500): float32 not JSON serializable` | Numpy types in DeepFace output | Fixed with `NumpySafeEncoder` class in emotion_server.py |
| `Exception: fetch method moved to InternetModule` | Wrong Lens Studio module for HTTP | Use `require('LensStudio:InternetModule')` for `fetch()` |
| Emotions always "Neutral" on Spectacles | Blend shapes return near-zero on device | Use BackendEmotionDetector (DeepFace backend), not EmotionDetector (blend shapes) |
| Flask server becomes unresponsive | Flask dev server deadlocks under load | Switched to `waitress` production WSGI server |
| `Camera init failed` in preview | CameraModule is Spectacles-only API | Expected in preview; BackendEmotionDetector only works on device |
| `Base64.encodeTextureAsync` silently fails | MediumQuality compression causes failure on Spectacles | Use `CompressionQuality.LowQuality` |

---

## Design Decisions & Lessons Learned

1. **Blend shapes don't work on Spectacles**: The initial approach using `EmotionDetector.js` with face mesh blend shape weights (MouthSmileLeft, BrowsDownLeft, etc.) worked well in Lens Studio preview but returned near-zero values on actual Spectacles hardware. This forced a pivot to the backend approach.

2. **DeepFace over LLM**: Unlike CueTips which used Google Gemini for emotion analysis, CareXR uses DeepFace -- a local Python library with pre-trained models. This avoids API costs and latency from LLM calls.

3. **waitress over Flask dev server**: The Flask development server with `threaded=True` repeatedly became unresponsive (deadlocked) under sustained load from Spectacles. Switching to waitress resolved this.

4. **MTCNN over OpenCV detector**: DeepFace supports multiple face detector backends. MTCNN provides better accuracy than OpenCV's Haar cascades, with acceptable speed (~300ms per analysis).

5. **LowQuality compression**: `Base64.encodeTextureAsync` with `CompressionQuality.MediumQuality` silently fails on Spectacles. `LowQuality` works reliably and keeps payloads at 47-84 KB.

6. **ngrok-skip-browser-warning header**: Required on all requests to ngrok free URLs to bypass the interstitial warning page that would otherwise return HTML instead of JSON.

7. **VoiceML conflicts with face tracking**: Enabling VoiceML (`VoiceMLModule.startListening()`) on Spectacles appeared to prevent `FaceFoundEvent` from firing, effectively disabling face tracking. This remains an open investigation.

---

## Future Work / In Progress

- **Speech Transcription**: Using Snap's VoiceML (on-device ASR) to transcribe speech from both the wearer and people nearby. Implementation started (`AudioTranscriber.js`) but was paused due to VoiceML conflicting with face tracking on Spectacles. Needs further investigation to run both simultaneously.

- **Emotion Map Customization**: The `EMOTION_MAP` in `emotion_server.py` can be adjusted to remap DeepFace's 7 raw emotions (happy, sad, angry, neutral, disgust, fear, surprise) to any desired labels.

---

## Reference: CueTips Project

The `Other Projects/CueTips/` directory contains a separate Snap Spectacles project that served as reference for:

- **ngrok tunneling pattern**: Exposing a local backend to Spectacles via ngrok
- **VoiceML transcription**: Using `VoiceMLModule` for on-device ASR (`Practice.js`)
- **Backend architecture**: Flask server with image/audio processing endpoints (`image_text_api.py`)
- **Remote fetch pattern**: Using `RemoteServiceModule` / `InternetModule` for HTTP from Lens Studio

Key difference: CueTips uses Google Gemini (LLM) for emotion analysis and response generation, while CareXR uses DeepFace (local ML model) for direct emotion classification.
