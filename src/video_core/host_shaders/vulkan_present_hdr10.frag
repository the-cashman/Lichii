// SPDX-License-Identifier: GPL-3.0-or-later
#version 460 core

// Window adaptation onto an HDR10 swapchain (A2B10G10R10 + ST.2084): the source holds SDR-style
// encoded values (guest frames: 0..1; Ocarina's HDR composite: above 1 for highlights).
// Decode (gamma 2.2 or the sRGB curve), scale to the UI white in nits, BT.709 -> BT.2020, PQ.
layout(constant_id = 203) const float GAMMA = 2.2;
layout(constant_id = 204) const float WHITE_NITS = 203.0;
layout(constant_id = 205) const float PEAK_NITS = 1500.0;

layout (location = 0) in vec2 frag_tex_coord;
layout (location = 0) out vec4 color;
layout (binding = 0) uniform sampler2D color_texture;

vec3 decode_srgb(vec3 c) {
    return mix(pow((c + 0.055) / 1.055, vec3(2.4)), c / 12.92, lessThanEqual(c, vec3(0.04045)));
}

vec3 encode_pq(vec3 nits) {
    const float m1 = 0.1593017578125;
    const float m2 = 78.84375;
    const float c1 = 0.8359375;
    const float c2 = 18.8515625;
    const float c3 = 18.6875;
    vec3 y = pow(clamp(nits / 10000.0, 0.0, 1.0), vec3(m1));
    return pow((c1 + c2 * y) / (1.0 + c3 * y), vec3(m2));
}

void main() {
    vec3 c = max(texture(color_texture, frag_tex_coord).rgb, vec3(0.0));
    vec3 linear = GAMMA == 0.0 ? decode_srgb(c) : pow(c, vec3(GAMMA));
    // Column-major: BT.709 -> BT.2020 (ITU-R BT.2087).
    const mat3 bt709_to_bt2020 = mat3(0.6274040, 0.0690970, 0.0163916,
                                      0.3292820, 0.9195400, 0.0880132,
                                      0.0433136, 0.0113612, 0.8955950);
    vec3 nits = min(bt709_to_bt2020 * linear * WHITE_NITS, vec3(PEAK_NITS));
    color = vec4(encode_pq(nits), 1.0);
}
