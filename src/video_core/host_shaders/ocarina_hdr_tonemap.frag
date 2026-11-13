// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// HDR stand-in for Ocarina of Time's tonemap pass (fragment shader bded5ce917183214 & co.,
// vk_ocarina_hdr.cpp). Original code, written from a description of the pass's behaviour: it
// reads the linear scene and a bloom texture, applies a colour matrix with offsets from its
// constant buffer, adds bloom raised to a power, scales by an exposure value and maps the result
// per channel with Krzysztof Narkowicz's published ACES filmic fit (2016), clamped to 0..1.
// This shader keeps that look up to a knee and then continues with a smooth C1 shoulder up to
// PEAK (paper white = 1) instead of clipping at 1. It reads the game's own constants at run time.
// Bindings are placeholders (set 7, 10..12); the emulator moves them onto the game's.
#version 450

layout(constant_id = 200) const float PEAK = 7.39;   // in the tonemap's units, gamma-adjusted
layout(constant_id = 201) const float KNEE_X = 0.356; // x where the HDR curve leaves the game's

layout(location = 0) in vec4 in_attr0;
layout(location = 0) out vec4 frag_color0;

layout(set = 7, binding = 10, std140) uniform fs_cbuf_block_4f {
    vec4 data[4096];
} c3;
layout(set = 7, binding = 11) uniform sampler2D fs_tex2_20;
layout(set = 7, binding = 12) uniform sampler2D fs_tex2_28;

float Word(uint i) {
    return c3.data[i >> 2][i & 3u];
}

vec3 AcesFit(vec3 x) {
    return (x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14);
}

float AcesFitSlope(float x) {
    const float n = x * (2.51 * x + 0.03);
    const float d = x * (2.43 * x + 0.59) + 0.14;
    return ((5.02 * x + 0.03) * d - n * (4.86 * x + 0.59)) / (d * d);
}

// The game's curve below the knee; above it y_k + (P - y_k) * t / sqrt(1 + t^2), with
// t = slope_k * (x - x_k) / (P - y_k): same value and slope at the knee, approaches PEAK.
vec3 HdrCurve(vec3 x) {
    const float y_k = (KNEE_X * (2.51 * KNEE_X + 0.03)) / (KNEE_X * (2.43 * KNEE_X + 0.59) + 0.14);
    const float slope = AcesFitSlope(KNEE_X);
    const float range = max(PEAK - y_k, 1e-3);
    const vec3 t = slope * max(x - KNEE_X, vec3(0.0)) / range;
    const vec3 shoulder = y_k + range * t * inversesqrt(1.0 + t * t);
    return mix(clamp(AcesFit(x), 0.0, 1.0), shoulder, greaterThan(x, vec3(KNEE_X)));
}

void main() {
    const vec2 uv = in_attr0.xy;
    const vec3 bloom = exp2(log2(abs(texture(fs_tex2_28, uv).rgb)) * Word(13u));
    const vec4 scene = texture(fs_tex2_20, uv);
    vec3 graded;
    graded.r = scene.r * Word(0u) + scene.g * Word(1u) + scene.b * Word(2u) + Word(3u);
    graded.g = scene.r * Word(4u) + scene.g * Word(5u) + scene.b * Word(6u) + Word(7u);
    graded.b = scene.r * Word(8u) + scene.g * Word(9u) + scene.b * Word(10u) + Word(11u);
    graded = clamp(graded, 0.0, 10000.0);
    const vec3 x = (graded + bloom) * Word(14u);
    frag_color0 = vec4(HdrCurve(x), scene.a);
}
