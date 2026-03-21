// @input Component.RenderMeshVisual faceMeshVisual {"hint": "Reference to the Face Mesh RenderMeshVisual"}
// @input Component.Text3D emotionText {"hint": "Text3D component to display detected emotion above the head"}
// @input float updateInterval = 0.1 {"hint": "Seconds between emotion classification updates"}
// @input float emotionThreshold = 0.2 {"hint": "Minimum score to register an emotion (vs Neutral)"}
// @input float smoothingFactor = 0.4 {"hint": "Smoothing for score transitions (0=no smoothing, 1=instant)"}
// @input float hysteresis = 0.05 {"hint": "Extra threshold the current emotion gets to prevent flickering"}
// @input float expressionMultiplier = 1.0 {"hint": "Amplifies raw expression weights (1.0 for preview)"}
// @input bool debugMode = false {"hint": "Log raw expression weights periodically for tuning"}
// @input float debugLogInterval = 2.0 {"hint": "Seconds between debug weight logs"}

var lastUpdateTime = 0;
var lastDebugTime = 0;
var currentEmotion = "";
var latestWeights = null;
var faceIsTracked = false;

var smoothedScores = {
    Joy: 0,
    Sadness: 0,
    Anger: 0
};

var EMOTION_DEFS = {
    Joy: {
        positive: [
            { name: "MouthSmileLeft", w: 1.2 },
            { name: "MouthSmileRight", w: 1.2 },
            { name: "CheekSquintLeft", w: 0.9 },
            { name: "CheekSquintRight", w: 0.9 },
            { name: "MouthDimpleLeft", w: 0.3 },
            { name: "MouthDimpleRight", w: 0.3 }
        ],
        negative: [
            { name: "MouthFrownLeft", w: 0.6 },
            { name: "MouthFrownRight", w: 0.6 },
            { name: "BrowsDownLeft", w: 0.4 },
            { name: "BrowsDownRight", w: 0.4 }
        ]
    },
    Sadness: {
        positive: [
            { name: "MouthFrownLeft", w: 1.2 },
            { name: "MouthFrownRight", w: 1.2 },
            { name: "BrowsUpCenter", w: 1.0 },
            { name: "BrowsUpLeft", w: 0.5 },
            { name: "BrowsUpRight", w: 0.5 },
            { name: "LipsPucker", w: 0.4 },
            { name: "LowerLipDownLeft", w: 0.3 },
            { name: "LowerLipDownRight", w: 0.3 },
            { name: "MouthClose", w: 0.2 }
        ],
        negative: [
            { name: "MouthSmileLeft", w: 0.8 },
            { name: "MouthSmileRight", w: 0.8 },
            { name: "CheekSquintLeft", w: 0.4 },
            { name: "CheekSquintRight", w: 0.4 }
        ]
    },
    Anger: {
        positive: [
            { name: "BrowsDownLeft", w: 1.3 },
            { name: "BrowsDownRight", w: 1.3 },
            { name: "SneerLeft", w: 0.9 },
            { name: "SneerRight", w: 0.9 },
            { name: "JawForward", w: 0.5 },
            { name: "LowerLipRaise", w: 0.5 },
            { name: "MouthStretchLeft", w: 0.3 },
            { name: "MouthStretchRight", w: 0.3 },
            { name: "EyeSquintLeft", w: 0.4 },
            { name: "EyeSquintRight", w: 0.4 },
            { name: "LipsFunnel", w: 0.2 }
        ],
        negative: [
            { name: "MouthSmileLeft", w: 0.5 },
            { name: "MouthSmileRight", w: 0.5 },
            { name: "BrowsUpCenter", w: 0.6 },
            { name: "BrowsUpLeft", w: 0.3 },
            { name: "BrowsUpRight", w: 0.3 }
        ]
    }
};

var KEY_EXPRESSIONS = [
    "MouthSmileLeft", "MouthSmileRight",
    "CheekSquintLeft", "CheekSquintRight",
    "MouthFrownLeft", "MouthFrownRight",
    "BrowsDownLeft", "BrowsDownRight",
    "BrowsUpCenter", "BrowsUpLeft", "BrowsUpRight",
    "SneerLeft", "SneerRight",
    "JawOpen", "JawForward",
    "EyeSquintLeft", "EyeSquintRight",
    "LipsPucker", "LipsFunnel"
];

function getRawWeight(name) {
    if (!latestWeights) return 0;
    try {
        var v = latestWeights[name];
        return (v !== undefined && v !== null) ? v : 0;
    } catch (e) {
        return 0;
    }
}

function getWeight(name) {
    var raw = getRawWeight(name);
    return Math.min(1.0, raw * script.expressionMultiplier);
}

function computeScore(def) {
    var posSum = 0, posW = 0;
    for (var i = 0; i < def.positive.length; i++) {
        var e = def.positive[i];
        posSum += getWeight(e.name) * e.w;
        posW += e.w;
    }
    var posScore = posW > 0 ? posSum / posW : 0;

    var negSum = 0, negW = 0;
    for (var j = 0; j < def.negative.length; j++) {
        var n = def.negative[j];
        negSum += getWeight(n.name) * n.w;
        negW += n.w;
    }
    var negScore = negW > 0 ? negSum / negW : 0;

    var raw = posScore - negScore * 0.5;
    return Math.max(0, raw);
}

function classifyEmotion() {
    var emotionNames = ["Joy", "Sadness", "Anger"];
    var rawScores = {};

    for (var i = 0; i < emotionNames.length; i++) {
        rawScores[emotionNames[i]] = computeScore(EMOTION_DEFS[emotionNames[i]]);
    }

    var sf = script.smoothingFactor;
    for (var j = 0; j < emotionNames.length; j++) {
        var em = emotionNames[j];
        smoothedScores[em] = smoothedScores[em] * (1 - sf) + rawScores[em] * sf;
    }

    var bestEmotion = "Neutral";
    var bestScore = script.emotionThreshold;

    for (var k = 0; k < emotionNames.length; k++) {
        var name = emotionNames[k];
        var threshold = script.emotionThreshold;
        if (name === currentEmotion) {
            threshold -= script.hysteresis;
        }
        if (smoothedScores[name] > threshold && smoothedScores[name] > bestScore) {
            bestScore = smoothedScores[name];
            bestEmotion = name;
        }
    }

    return bestEmotion;
}

function getEmoji(emotion) {
    switch (emotion) {
        case "Joy": return "\u{1F60A}";
        case "Sadness": return "\u{1F622}";
        case "Anger": return "\u{1F621}";
        case "Neutral": return "\u{1F610}";
        default: return "";
    }
}

function showEmotion(emotion) {
    if (!script.emotionText) return;
    script.emotionText.text = getEmoji(emotion) + " " + emotion;
    script.emotionText.getSceneObject().enabled = true;
}

function hideEmotion() {
    if (!script.emotionText) return;
    script.emotionText.text = "";
    script.emotionText.getSceneObject().enabled = false;
}

function logDebugWeights() {
    if (!script.debugMode || !latestWeights) return;

    var now = getTime();
    if (now - lastDebugTime < script.debugLogInterval) return;
    lastDebugTime = now;

    var parts = [];
    for (var i = 0; i < KEY_EXPRESSIONS.length; i++) {
        var name = KEY_EXPRESSIONS[i];
        var raw = getRawWeight(name);
        if (raw > 0.005) {
            parts.push(name + ":" + raw.toFixed(3));
        }
    }

    var weightsStr = parts.length > 0 ? parts.join(" ") : "(all near zero)";
    print("[DEBUG] raw=" + weightsStr
        + " | scores J:" + smoothedScores.Joy.toFixed(3)
        + " S:" + smoothedScores.Sadness.toFixed(3)
        + " A:" + smoothedScores.Anger.toFixed(3)
        + " | mult:" + script.expressionMultiplier
        + " | thresh:" + script.emotionThreshold
        + " | current:" + currentEmotion);
}

script.createEvent("OnStartEvent").bind(function () {
    print("EmotionDetector: Initializing (mult=" + script.expressionMultiplier + " thresh=" + script.emotionThreshold + ")...");

    if (!script.faceMeshVisual) {
        print("EmotionDetector ERROR: faceMeshVisual input is not set.");
        return;
    }

    var faceProvider = script.faceMeshVisual.mesh.control;
    if (faceProvider && faceProvider.onExpressionWeightsUpdate) {
        faceProvider.onExpressionWeightsUpdate.add(function (expressionWeights) {
            latestWeights = expressionWeights;
        });
        print("EmotionDetector: Subscribed to onExpressionWeightsUpdate.");
    }

    var faceFound = script.createEvent("FaceFoundEvent");
    faceFound.faceIndex = 0;
    faceFound.bind(function () {
        faceIsTracked = true;
        print("EmotionDetector: Face found.");
    });

    var faceLost = script.createEvent("FaceLostEvent");
    faceLost.faceIndex = 0;
    faceLost.bind(function () {
        faceIsTracked = false;
        latestWeights = null;
        currentEmotion = "";
        smoothedScores.Joy = 0;
        smoothedScores.Sadness = 0;
        smoothedScores.Anger = 0;
        hideEmotion();
        print("EmotionDetector: Face lost, hiding emotion.");
    });

    hideEmotion();
    print("EmotionDetector: Ready. Waiting for face...");
});

script.createEvent("UpdateEvent").bind(function () {
    if (!script.faceMeshVisual) return;
    if (!faceIsTracked) return;
    if (!latestWeights) return;

    var now = getTime();
    if (now - lastUpdateTime < script.updateInterval) return;
    lastUpdateTime = now;

    var detectedEmotion = classifyEmotion();

    logDebugWeights();

    if (detectedEmotion !== currentEmotion) {
        currentEmotion = detectedEmotion;
        print("EmotionDetector: " + getEmoji(detectedEmotion) + " " + detectedEmotion
            + " (J:" + smoothedScores.Joy.toFixed(3)
            + " S:" + smoothedScores.Sadness.toFixed(3)
            + " A:" + smoothedScores.Anger.toFixed(3) + ")");
    }

    showEmotion(currentEmotion);
});
