// @input Component.RenderMeshVisual faceMeshVisual {"hint": "EyeGlowFaceMesh RenderMeshVisual"}
// @input float maxAlpha = 0.22 {"hint": "Peak glow alpha (keep 0.15-0.28 for subtlety)"}
// @input float fadeInTime = 1.0 {"hint": "Seconds to fade in"}
// @input float fadeOutTime = 1.0 {"hint": "Seconds to fade out"}
// @input float offTime = 3.0 {"hint": "Seconds off between pulses"}

var timer = 0;
var FADE_IN  = 0;
var FADE_OUT = 1;
var OFF      = 1.5;
var phase    = OFF;  // start in OFF so first glow triggers after offTime

function setAlpha(a) {
    if (!script.faceMeshVisual) return;
    var mat = script.faceMeshVisual.getMaterial(0);
    if (!mat) return;
    var c = mat.mainPass.baseColor;
    mat.mainPass.baseColor = new vec4(c.r, c.g, c.b, a);
}

var startEvent = script.createEvent("OnStartEvent");
startEvent.bind(function() {
    setAlpha(0);
});

var updateEvent = script.createEvent("UpdateEvent");
updateEvent.bind(function(eventData) {
    var dt = getDeltaTime();
    timer += dt;

    if (phase === OFF) {
        if (timer >= script.offTime) {
            timer = 0;
            phase = FADE_IN;
        }
    } else if (phase === FADE_IN) {
        var t = Math.min(timer / script.fadeInTime, 1.0);
        setAlpha(t * script.maxAlpha);
        if (timer >= script.fadeInTime) {
            timer = 0;
            phase = FADE_OUT;
        }
    } else if (phase === FADE_OUT) {
        var t = 1.0 - Math.min(timer / script.fadeOutTime, 1.0);
        setAlpha(t * script.maxAlpha);
        if (timer >= script.fadeOutTime) {
            timer = 0;
            phase = OFF;
            setAlpha(0);
        }
    }
});
