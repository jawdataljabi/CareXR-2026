// BackendEmotionDetector.js
// Captures camera frames on Spectacles, sends to a Python/DeepFace backend
// for emotion recognition, and displays the result as 3D text above the head.

var internetModule;
var cameraModule;
try {
    internetModule = require('LensStudio:InternetModule');
    cameraModule = require('LensStudio:CameraModule');
} catch (e) {
    print("[BackendEmotion] Spectacles modules not available (preview mode). Script disabled.");
}

// @input Component.Text emotionText {"hint": "Text component to display detected emotion (with bubble background)"}
// @input Component.Text hintText {"hint": "Text component to display follow-up hint from LLM (with bubble background)"}
// @input string backendUrl = "https://YOUR-NGROK-URL.ngrok-free.app/analyze" {"hint": "Backend URL for emotion analysis"}
// @input float interval = 0.1 {"hint": "Seconds between backend API calls"}
// @input bool debugMode = true {"hint": "Log debug info to console"}

var cameraRequest;
var cameraTexture;
var cameraTextureProvider;
var lastSentTime = 0;
var isSending = false;
var faceIsTracked = false;
var lastLabel = "";
var isSpectacles = false;

var EMOJI_MAP = {
    "Joy": "\u{1F60A}",
    "Sadness": "\u{1F622}",
    "Anger": "\u{1F621}",
    "Neutral": "\u{1F610}"
};

function showEmotion(label) {
    if (!script.emotionText) return;
    script.emotionText.text = label;
    script.emotionText.getSceneObject().enabled = true;
}

function hideEmotion() {
    if (!script.emotionText) return;
    script.emotionText.text = "";
    script.emotionText.getSceneObject().enabled = false;
}

function showHint(text) {
    if (!script.hintText) return;
    script.hintText.text = text;
    script.hintText.getSceneObject().enabled = true;
}

function hideHint() {
    if (!script.hintText) return;
    script.hintText.text = "";
    script.hintText.getSceneObject().enabled = false;
}

function debugLog(msg) {
    if (script.debugMode) {
        print("[BackendEmotion] " + msg);
    }
}

script.createEvent('OnStartEvent').bind(function() {
    if (!internetModule || !cameraModule) {
        print("[BackendEmotion] Not running on Spectacles. This script only works on device.");
        print("[BackendEmotion] For preview, enable the EmotionDetectorController (blend shapes) instead.");
        return;
    }

    print("[BackendEmotion] Initializing on Spectacles...");
    print("[BackendEmotion] Backend URL: " + script.backendUrl);

    try {
        cameraRequest = CameraModule.createCameraRequest();
        cameraRequest.cameraId = CameraModule.CameraId.Default_Color;

        cameraTexture = cameraModule.requestCamera(cameraRequest);
        cameraTextureProvider = cameraTexture.control;
        isSpectacles = true;
    } catch (e) {
        print("[BackendEmotion] Camera init failed (not on Spectacles): " + e);
        return;
    }

    hideEmotion();
    hideHint();

    var faceFound = script.createEvent("FaceFoundEvent");
    faceFound.faceIndex = 0;
    faceFound.bind(function() {
        faceIsTracked = true;
        debugLog("Face found.");
    });

    var faceLost = script.createEvent("FaceLostEvent");
    faceLost.faceIndex = 0;
    faceLost.bind(function() {
        faceIsTracked = false;
        lastLabel = "";
        hideEmotion();
        hideHint();
        debugLog("Face lost.");
    });

    cameraTextureProvider.onNewFrame.add(function(cameraFrame) {
        if (!faceIsTracked) return;
        if (isSending) return;

        var now = getTime();
        if (now - lastSentTime < script.interval) return;
        lastSentTime = now;

        captureAndSend();
    });

    print("[BackendEmotion] Ready. Waiting for face...");
});

function captureAndSend() {
    if (!isSpectacles) return;
    isSending = true;

    Base64.encodeTextureAsync(
        cameraTexture,
        function(base64Image) {
            sendToBackend(base64Image);
        },
        function() {
            print("[BackendEmotion] Failed to encode camera frame.");
            isSending = false;
        },
        CompressionQuality.LowQuality,
        EncodingType.Jpg
    );
}

async function sendToBackend(base64Image) {
    try {
        var requestBody = JSON.stringify({ image: base64Image });

        var request = new Request(script.backendUrl, {
            method: 'POST',
            body: requestBody,
            headers: {
                'Content-Type': 'application/json',
                'ngrok-skip-browser-warning': 'true'
            }
        });

        debugLog("Sending " + Math.round(base64Image.length / 1024) + " KB...");

        var response = await internetModule.fetch(request);

        if (response.status === 200) {
            var responseJson = await response.json();
            var emotion = responseJson.emotion || "none";
            var confidence = responseJson.confidence || 0;
            var label = responseJson.label || "";
            var latency = responseJson.latency_ms || 0;

            var followUpHint = responseJson.follow_up_hint || "";

            debugLog(emotion + " (" + (confidence * 100).toFixed(0) + "%) " + latency + "ms");
            if (followUpHint) {
                debugLog("Hint: " + followUpHint);
            }

            if (emotion !== "none" && faceIsTracked) {
                lastLabel = label || (EMOJI_MAP[emotion] || "") + " " + emotion;
                showEmotion(lastLabel);
            } else if (!faceIsTracked) {
                hideEmotion();
            }

            if (followUpHint && faceIsTracked) {
                showHint(followUpHint);
            }
        } else {
            var errorText = await response.text();
            print("[BackendEmotion] Backend error (" + response.status + "): " + errorText);
            if (lastLabel && faceIsTracked) showEmotion(lastLabel);
        }
    } catch (error) {
        print("[BackendEmotion] Network error: " + error);
        if (lastLabel && faceIsTracked) showEmotion(lastLabel);
    } finally {
        isSending = false;
    }
}