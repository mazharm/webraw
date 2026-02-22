// WebGL2 fragment shader for real-time image adjustments
// Applied to the base render texture (stages 10-17 in the pipeline)

precision highp float;

uniform sampler2D u_baseTexture;
uniform vec2 u_resolution;

// Basic adjustments
uniform float u_exposure;
uniform float u_contrast;
uniform float u_highlights;
uniform float u_shadows;
uniform float u_whites;
uniform float u_blacks;
uniform float u_vibrance;
uniform float u_saturation;
uniform float u_texture;
uniform float u_clarity;
uniform float u_dehaze;

// Temperature/Tint
uniform float u_temperature;
uniform float u_tint;

// Tone curve (as 1D texture or via control points)
uniform sampler2D u_toneCurve;

// Film sim LUT (3D texture)
uniform sampler3D u_filmSimLUT;
uniform float u_filmSimStrength;
uniform bool u_filmSimEnabled;

// Effects
uniform float u_grainAmount;
uniform float u_vignetteAmount;
uniform float u_time; // for grain noise

in vec2 v_texCoord;
out vec4 fragColor;

// sRGB <-> Linear conversions
float srgbToLinear(float c) {
    return c <= 0.04045 ? c / 12.92 : pow((c + 0.055) / 1.055, 2.4);
}

float linearToSrgb(float c) {
    return c <= 0.0031308 ? c * 12.92 : 1.055 * pow(c, 1.0 / 2.4) - 0.055;
}

vec3 srgbToLinear(vec3 c) {
    return vec3(srgbToLinear(c.r), srgbToLinear(c.g), srgbToLinear(c.b));
}

vec3 linearToSrgb(vec3 c) {
    return vec3(linearToSrgb(c.r), linearToSrgb(c.g), linearToSrgb(c.b));
}

// Luminance
float luminance(vec3 c) {
    return dot(c, vec3(0.2126, 0.7152, 0.0722));
}

// Hash-based noise
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

    // Highlights/Shadows/Whites/Blacks (simplified zone-based)
    float lum = luminance(color);

    // Highlights affect bright areas
    float highlightMask = smoothstep(0.5, 1.0, lum);
    color += color * highlightMask * (u_highlights / 200.0);

    // Shadows affect dark areas
    float shadowMask = 1.0 - smoothstep(0.0, 0.5, lum);
    color += color * shadowMask * (u_shadows / 200.0);

    // Whites (extreme highlights)
    float whiteMask = smoothstep(0.75, 1.0, lum);
    color += color * whiteMask * (u_whites / 200.0);

    // Blacks (extreme shadows)
    float blackMask = 1.0 - smoothstep(0.0, 0.25, lum);
    color += color * blackMask * (u_blacks / 200.0);

    // Vibrance (smart saturation - less effect on already-saturated colors)
    float currentSat = max(max(color.r, color.g), color.b) - min(min(color.r, color.g), color.b);
    float vibranceFactor = 1.0 + (u_vibrance / 100.0) * (1.0 - currentSat);
    float lumV = luminance(color);
    color = lumV + (color - lumV) * vibranceFactor;

    // Saturation
    float satFactor = 1.0 + u_saturation / 100.0;
    float lumS = luminance(color);
    color = lumS + (color - lumS) * satFactor;

    // Dehaze (simplified)
    if (u_dehaze != 0.0) {
        float hazeFactor = u_dehaze / 100.0;
        color = color * (1.0 + hazeFactor * 0.5) - hazeFactor * 0.1;
    }

    // Film sim LUT (if enabled)
    if (u_filmSimEnabled) {
        vec3 lutCoord = clamp(color, 0.0, 1.0);
        vec3 lutColor = texture(u_filmSimLUT, lutCoord).rgb;
        color = mix(color, lutColor, u_filmSimStrength);
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

    // Clamp output
    color = clamp(color, 0.0, 1.0);

    fragColor = vec4(color, texColor.a);
}
