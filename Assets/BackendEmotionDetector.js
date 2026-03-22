// BackendEmotionDetector.js
// Captures camera frames on Spectacles, sends to a Python/DeepFace backend
// for emotion recognition. Displays emotion as color-coded pastel text bubbles
// with crossfade transitions, and LLM hints in a soft grey iMessage-style bubble.

var internetModule;
var cameraModule;
try {
    internetModule = require('LensStudio:InternetModule');
    cameraModule = require('LensStudio:CameraModule');
} catch (e) {
    print("[BackendEmotion] Spectacles modules not available (preview mode). Script disabled.");
}

// @input Component.Text joyText {"hint": "Joy emotion bubble (pastel green background)"}
// @input Component.Text sadnessText {"hint": "Sadness emotion bubble (pastel blue background)"}
// @input Component.Text neutralText {"hint": "Neutral emotion bubble (pastel beige background)"}
// @input Component.Text hintText {"hint": "LLM hint bubble (soft grey background, white text)"}
// @input string backendUrl = "https://YOUR-NGROK-URL.ngrok-free.app/analyze" {"hint": "Backend URL for emotion analysis"}
// @input float interval = 0.1 {"hint": "Seconds between backend API calls"}
// @input bool debugMode = true {"hint": "Log debug info to console"}

var cameraRequest;
var cameraTexture;
var cameraTextureProvider;
var lastSentTime = 0;
var isSending = false;
var faceIsTracked = false;
var isSpectacles = false;

var FADE_SPEED = 4.0;

var emotionAlpha = { Joy: 0, Sadness: 0, Neutral: 0 };
var emotionTarget = { Joy: 0, Sadness: 0, Neutral: 0 };
var emotionComps = {};

var EMOTION_BG = {
    Joy:     new vec3(0.686, 0.918, 0.710),
    Sadness: new vec3(0.686, 0.812, 0.941),
    Neutral: new vec3(0.941, 0.902, 0.827)
};
var EMOTION_TEXT_RGB = new vec3(0.200, 0.200, 0.220);
var EMOTION_BG_ALPHA = 0.90;

var HINT_BG_RGB = new vec3(0.400, 0.400, 0.420);
var HINT_TEXT_RGB = new vec3(1.0, 1.0, 1.0);
var HINT_BG_ALPHA = 0.85;

var hintAlpha = 0;
var hintTarget = 0;
var hintFadingOut = false;
var pendingHint = "";
var currentHintContent = "";

var EMOJI_MAP = {
    "Joy": "\u{1F60A}",
    "Sadness": "\u{1F622}",
    "Neutral": "\u{1F610}"
};

function applyAlpha(textComp, textRgb, bgRgb, alpha, bgMaxAlpha) {
    if (!textComp) return;
    textComp.textFill.color = new vec4(textRgb.x, textRgb.y, textRgb.z, alpha);
    textComp.backgroundSettings.fill.color = new vec4(bgRgb.x, bgRgb.y, bgRgb.z, alpha * bgMaxAlpha);
}

function debugLog(msg) {
    if (script.debugMode) print("[BackendEmotion] " + msg);
}

script.createEvent('OnStartEvent').bind(function () {
    if (!internetModule || !cameraModule) {
        print("[BackendEmotion] Not running on Spectacles. This script only works on device.");
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

    emotionComps["Joy"] = script.joyText;
    emotionComps["Sadness"] = script.sadnessText;
    emotionComps["Neutral"] = script.neutralText;

    for (var em in EMOJI_MAP) {
        var c = emotionComps[em];
        if (c) {
            c.text = EMOJI_MAP[em] + " " + em;
            applyAlpha(c, EMOTION_TEXT_RGB, EMOTION_BG[em], 0, EMOTION_BG_ALPHA);
        }
    }

    if (script.hintText) {
        script.hintText.text = "";
        applyAlpha(script.hintText, HINT_TEXT_RGB, HINT_BG_RGB, 0, HINT_BG_ALPHA);
    }

    var faceFound = script.createEvent("FaceFoundEvent");
    faceFound.faceIndex = 0;
    faceFound.bind(function () {
        faceIsTracked = true;
        debugLog("Face found.");
    });

    var faceLost = script.createEvent("FaceLostEvent");
    faceLost.faceIndex = 0;
    faceLost.bind(function () {
        faceIsTracked = false;
        for (var e in emotionTarget) emotionTarget[e] = 0;
        hintTarget = 0;
        hintFadingOut = false;
        pendingHint = "";
        debugLog("Face lost.");
    });

    cameraTextureProvider.onNewFrame.add(function () {
        if (!faceIsTracked || isSending) return;
        var now = getTime();
        if (now - lastSentTime < script.interval) return;
        lastSentTime = now;
        captureAndSend();
    });

    script.createEvent("UpdateEvent").bind(function () {
        var dt = getDeltaTime();
        animateEmotions(dt);
        animateHint(dt);
    });

    print("[BackendEmotion] Ready. Waiting for face...");
});

function lerpAlpha(current, target, step) {
    if (current < target) return Math.min(current + step, target);
    if (current > target) return Math.max(current - step, target);
    return current;
}

function animateEmotions(dt) {
    var step = FADE_SPEED * dt;
    for (var em in emotionAlpha) {
        var prev = emotionAlpha[em];
        emotionAlpha[em] = lerpAlpha(prev, emotionTarget[em], step);
        if (emotionAlpha[em] !== prev) {
            applyAlpha(emotionComps[em], EMOTION_TEXT_RGB, EMOTION_BG[em],
                       emotionAlpha[em], EMOTION_BG_ALPHA);
        }
    }
}

function animateHint(dt) {
    var step = FADE_SPEED * dt;
    var prev = hintAlpha;
    hintAlpha = lerpAlpha(prev, hintTarget, step);
    if (hintAlpha !== prev) {
        applyAlpha(script.hintText, HINT_TEXT_RGB, HINT_BG_RGB,
                   hintAlpha, HINT_BG_ALPHA);
    }
    if (hintFadingOut && hintAlpha <= 0.001) {
        hintFadingOut = false;
        hintAlpha = 0;
        if (pendingHint) {
            currentHintContent = pendingHint;
            if (script.hintText) script.hintText.text = pendingHint;
            pendingHint = "";
            hintTarget = 1;
        }
    }
}

function showEmotion(emotion) {
    if (!emotionTarget.hasOwnProperty(emotion)) emotion = "Neutral";
    for (var e in emotionTarget) {
        emotionTarget[e] = (e === emotion) ? 1 : 0;
    }
}

function showHint(text) {
    if (!script.hintText || text === currentHintContent) return;
    if (hintAlpha > 0.01) {
        pendingHint = text;
        hintFadingOut = true;
        hintTarget = 0;
    } else {
        currentHintContent = text;
        script.hintText.text = text;
        hintTarget = 1;
    }
}

function captureAndSend() {
    if (!isSpectacles) return;
    isSending = true;
    Base64.encodeTextureAsync(
        cameraTexture,
        function (base64Image) { sendToBackend(base64Image); },
        function () {
            print("[BackendEmotion] Failed to encode camera frame.");
            isSending = false;
        },
        CompressionQuality.LowQuality,
        EncodingType.Jpg
    );
}

async function sendToBackend(base64Image) {
    try {
        var request = new Request(script.backendUrl, {
            method: 'POST',
            body: JSON.stringify({ image: base64Image }),
            headers: {
                'Content-Type': 'application/json',
                'ngrok-skip-browser-warning': 'true'
            }
        });

        debugLog("Sending " + Math.round(base64Image.length / 1024) + " KB...");

        var response = await internetModule.fetch(request);

        if (response.status === 200) {
            var json = await response.json();
            var emotion = json.emotion || "none";
            var confidence = json.confidence || 0;
            var latency = json.latency_ms || 0;
            var followUpHint = json.follow_up_hint || "";

            debugLog(emotion + " (" + (confidence * 100).toFixed(0) + "%) " + latency + "ms");
            if (followUpHint) debugLog("Hint: " + followUpHint);

            if (emotion !== "none" && faceIsTracked) {
                showEmotion(emotion);
            }

            if (followUpHint && faceIsTracked) {
                showHint(followUpHint);
            }
        } else {
            var errorText = await response.text();
            print("[BackendEmotion] Backend error (" + response.status + "): " + errorText);
        }
    } catch (error) {
        print("[BackendEmotion] Network error: " + error);
    } finally {
        isSending = false;
    }
}
