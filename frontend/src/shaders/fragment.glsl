#version 300 es
precision highp float;

uniform sampler2D u_baseTexture;

// Basic adjustments
uniform float u_exposure;
uniform float u_contrast;
uniform float u_highlights;
uniform float u_shadows;
uniform float u_whites;
uniform float u_blacks;
uniform float u_vibrance;
uniform float u_saturation;
uniform float u_dehaze;

// Effects
uniform float u_grainAmount;
uniform float u_vignetteAmount;
uniform vec2 u_resolution;
uniform float u_time;

// Film sim
uniform bool u_filmSimEnabled;
uniform float u_filmSimStrength;

in vec2 v_texCoord;
out vec4 fragColor;

float luminance(vec3 c) {
    return dot(c, vec3(0.2126, 0.7152, 0.0722));
}

float hash(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

void main() {
    vec4 texColor = texture(u_baseTexture, v_texCoord);
    vec3 color = texColor.rgb;

    // Exposure (EV stops)
    color *= pow(2.0, u_exposure);

    // Contrast
    float contrastFactor = (100.0 + u_contrast) / 100.0;
    color = (color - 0.5) * contrastFactor + 0.5;

    // Zone-based adjustments
    float lum = luminance(color);
    float highlightMask = smoothstep(0.5, 1.0, lum);
    color += color * highlightMask * (u_highlights / 200.0);

    float shadowMask = 1.0 - smoothstep(0.0, 0.5, lum);
    color += color * shadowMask * (u_shadows / 200.0);

    float whiteMask = smoothstep(0.75, 1.0, lum);
    color += color * whiteMask * (u_whites / 200.0);

    float blackMask = 1.0 - smoothstep(0.0, 0.25, lum);
    color += color * blackMask * (u_blacks / 200.0);

    // Vibrance
    float currentSat = max(max(color.r, color.g), color.b) - min(min(color.r, color.g), color.b);
    float vibranceFactor = 1.0 + (u_vibrance / 100.0) * (1.0 - currentSat);
    float lumV = luminance(color);
    color = lumV + (color - lumV) * vibranceFactor;

    // Saturation
    float satFactor = 1.0 + u_saturation / 100.0;
    float lumS = luminance(color);
    color = lumS + (color - lumS) * satFactor;

    // Dehaze
    if (u_dehaze != 0.0) {
        float hazeFactor = u_dehaze / 100.0;
        color = color * (1.0 + hazeFactor * 0.5) - hazeFactor * 0.1;
    }

    // Grain
    if (u_grainAmount > 0.0) {
        float grain = hash(v_texCoord * u_resolution + u_time) - 0.5;
        color += grain * (u_grainAmount / 100.0) * 0.3;
    }

    // Vignette
    if (u_vignetteAmount != 0.0) {
        vec2 center = v_texCoord - 0.5;
        float dist = length(center) * 1.414;
        float vignette = 1.0 - dist * dist * (u_vignetteAmount / 100.0);
        color *= vignette;
    }

    color = clamp(color, 0.0, 1.0);
    fragColor = vec4(color, texColor.a);
}
