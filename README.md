# SoftEyes

🏆 **1st Place, Snap Inc. Sponsored Category - McGillXR CareXR26 Hackathon**

[Demo Video](https://www.youtube.com/watch?v=StcCCB5jzkQ) &nbsp;|&nbsp; [Devpost](https://devpost.com/software/project-wy8hfleo50qp)

---

What if your glasses could read the room for you, picking up the cues most people miss and quietly guiding you through the conversation in real time.

SoftEyes is a real-time social companion built for [Snap Spectacles](https://www.spectacles.com/). It runs quietly in the wearer's field of vision during face-to-face conversations and provides unobtrusive, research-grounded support for people who find social interaction overwhelming, including autistic individuals and those with social anxiety.

---

## The Problem

Over 70 million people worldwide are autistic, and hundreds of millions more live with social anxiety. For many of them, face-to-face conversation is genuinely overwhelming. Tracking what to say next, reading the other person's emotional state, managing eye contact, and keeping up with the flow of a conversation can make even a simple exchange feel impossible.

Almost every tool designed to help with social communication is a training tool; something used at home or in a clinic, not in the moment. SoftEyes is different. It meets people where they actually struggle: in real life, in real time.

---

## Features

### Emotion Detection and Contextual Nudges

SoftEyes reads the emotional state of the person across from the wearer in real time using computer vision and [DeepFace](https://github.com/serengil/deepface), then surfaces a gentle, contextual suggestion in AR; for example, "Follow up on their university studies they mentioned" or "Ask what breed their dog is." The goal is not to script the conversation but to provide just enough signal for the user to engage naturally and build real social skills over time, gradually reducing reliance on the glasses.

### Eye Contact Guidance

A soft, calm glow is placed around the other person's eyes in AR, giving the wearer a natural and comfortable place to look without forcing rigid eye contact. This directly addresses one of the most commonly reported sources of anxiety in face-to-face interaction.

### Accessibility-First UI Design

Every design decision maps to published research on sensory and cognitive accessibility for autistic individuals:

- Pastel colours only, no saturated or high-contrast elements
- All overlays fade gently in and out, no sudden appearances or jarring transitions
- Minimal information density throughout
- Overlays are spatially anchored near the conversation partner to keep the wearer's attention in the conversation, not in a UI corner
- A noise-cancellation microphone pipeline suppresses ambient audio so the wearer can focus on the voice in front of them

---

## Architecture

```
+---------------------------------------------+
|             Snap Spectacles                  |
|                                              |
|  CameraModule -> Base64 JPEG (0.1s throttle) |
|  InternetModule -> POST /analyze             |
|  Text3D overlay anchored via Head Binding    |
|  GLSL shaders -> soft glow + fade transitions|
+--------------------+------------------------+
                     | HTTPS (ngrok tunnel)
+--------------------v------------------------+
|         Python / Flask Backend               |
|                                              |
|  emotion_server.py                           |
|  Waitress WSGI (4 threads, port 5001)        |
|  DeepFace.analyze() with MTCNN detector      |
|  7 raw emotions -> 4 simplified categories   |
|  Returns: { emotion, confidence, latency }   |
+---------------------------------------------+
```

### Lens Side (JavaScript / GLSL)

`BackendEmotionDetector.js` uses Lens Studio's `CameraModule` to access the Spectacles outward-facing camera. It listens for `FaceFoundEvent` and `FaceLostEvent` to track the presence of the other person. On each camera frame, throttled to 0.1-second intervals, it encodes the frame as a Base64 JPEG using `Base64.encodeTextureAsync()` with `LowQuality` compression, then POSTs it to the backend via Lens Studio's `InternetModule`. The emotion label is rendered as a `Text3D` component anchored above the tracked face via a `Head Binding` in the scene hierarchy.

### Backend (Python / Flask / DeepFace)

`emotion_server.py` is a Flask server running on Waitress (4 threads) on port 5001. On receiving a request, it decodes the Base64 image, downscales it, and runs `DeepFace.analyze()` using the MTCNN face detector. DeepFace's 7 raw emotion outputs are mapped to 4 simplified categories: Joy, Sadness, Anger, and Neutral. Results are returned as JSON with emotion, confidence, and latency.

### Tunneling

ngrok exposes the local Flask server to the Spectacles device over HTTPS.

---

## Repo Structure

```
CareXR-2026/
├── Assets/                  # Lens Studio assets (scripts, materials, textures)
├── Backend/                 # Python Flask emotion server
│   └── emotion_server.py
├── Cache/                   # Lens Studio build cache
├── Other Projects/          # Experimental lens work
├── PluginsUserPreferences/
├── Support/
├── Workspaces/              # Lens Studio workspace config
├── CareXR.esproj            # Lens Studio project file
└── jsconfig.json
```

---

## Getting Started

### Prerequisites

- [Snap Spectacles](https://www.spectacles.com/) (developer unit)
- [Lens Studio 5.15.4+](https://ar.snap.com/lens-studio)
- Python 3.9+
- [ngrok](https://ngrok.com/)

### 1. Start the Backend

```bash
cd Backend
pip install flask waitress deepface tf-keras
python emotion_server.py
```

The server will start on `http://localhost:5001`.

### 2. Expose the Backend via ngrok

```bash
ngrok http 5001
```

Copy the `https://` forwarding URL that ngrok provides.

### 3. Configure the Lens

In `Assets/BackendEmotionDetector.js`, update the backend URL with your ngrok HTTPS URL:

```js
const BACKEND_URL = "https://your-ngrok-url.ngrok-free.app/analyze";
```

### 4. Open in Lens Studio and Deploy

1. Open `CareXR.esproj` in Lens Studio 5.15.4+
2. Connect your Snap Spectacles via the Lens Studio device panel
3. Click **Push to Device**

---

## Built With

- [Lens Studio 5.15.4](https://ar.snap.com/lens-studio)
- JavaScript (Lens-side scripting)
- GLSL (custom shaders for glow and fade transitions)
- Python, Flask, Waitress (backend inference server)
- [DeepFace](https://github.com/serengil/deepface) with MTCNN (facial emotion analysis)
- ngrok (HTTPS tunneling)
