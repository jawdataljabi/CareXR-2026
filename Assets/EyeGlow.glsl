#include <std2.glsl>

// Pulse uniform driven from script (0.0 to 1.0)
uniform float pulseAmount;

void main() {
    vec2 uv = varTex01.xy;

    // --- Eye region UV masks on Snap's Face Mesh UV layout ---
    // Left eye (from wearer's perspective): roughly x 0.56-0.73, y 0.44-0.62
    // Right eye (from wearer's perspective): roughly x 0.27-0.44, y 0.44-0.62
    float leftEye  = smoothstep(0.56, 0.59, uv.x) * (1.0 - smoothstep(0.70, 0.73, uv.x))
                   * smoothstep(0.44, 0.47, uv.y) * (1.0 - smoothstep(0.59, 0.62, uv.y));
    float rightEye = smoothstep(0.27, 0.30, uv.x) * (1.0 - smoothstep(0.41, 0.44, uv.x))
                   * smoothstep(0.44, 0.47, uv.y) * (1.0 - smoothstep(0.59, 0.62, uv.y));

    float mask = clamp(leftEye + rightEye, 0.0, 1.0);

    // Soft radial falloff within each eye box so edges aren't hard
    // (smoothstep edges above already provide soft falloff)

    // Near-white glow colour — works well on waveguide additive display
    vec3 glowColor = vec3(0.95, 0.97, 1.0);

    // Max alpha kept subtle: 0.28 peak
    float glowAlpha = mask * pulseAmount * 0.28;

    gl_FragColor = vec4(glowColor * glowAlpha, glowAlpha);
}
