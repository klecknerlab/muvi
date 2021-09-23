/*
Copyright 2021 Dustin Kleckner

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#version 330

uniform mat4 modelMatrix = mat4(1.0);
uniform mat4 viewMatrix = mat4(1.0);
uniform mat4 perspectiveMatrix = mat4(1.0);
uniform float pixelsPerEm = 32.0;
uniform vec2 viewportSize = vec2(100.0, 100.0);

// layout (location = 0) in vec3 position;
// layout (location = 1) in vec3 offset;
// layout (location = 2) in vec4 atlas;

layout (location = 0) in vec3 anchor;
layout (location = 1) in vec4 glyph; // x/y = container, z/w = offset relative to bottom-left corner of container
layout (location = 2) in vec4 atlas; // x/y = x0/y0, z/w = x1/y1
layout (location = 3) in vec2 padding;
layout (location = 4) in vec3 baseline;
layout (location = 5) in vec3 up;
layout (location = 6) in uint flags;

// Flags:
// Lowest byte is anchoring:
// 0-7: upright text, anchor is bottom-left, bottom, bottom-right, right...
// 8-15: center aligned
// 16-31: text orientation determined by baseline vector, anchor the same otherwise
// 32-47: text orientation determined by up vector, anchor the same otherwise
// 48-63: upright text, shifted in direction of up to keep container outside of baseline.
// Upper 3 bytes are used as bitwise flags to determine visibility.
// If upper 3 bytes are 0, then text is always visible.

out GlyphData {
    vec2 offset;
    vec2 right;
    vec2 up;
    vec4 atlas;
    uint visible;
} gOut;

vec2 screenNormal(vec4 X, vec4 N) {
    //  - N.w * X.xy
    return normalize(viewportSize * (N.xy - (N.w / X.w) * X.xy));
}

void main()
{
    vec2 emSize = (atlas.zy - atlas.xw) / pixelsPerEm;
    mat4 MVP = perspectiveMatrix * viewMatrix * modelMatrix;
    gl_Position = MVP * vec4(anchor, 1.0);
    vec2 cornerOffset = vec2(0.0);
    vec2 R = vec2(1.0, 0.0); // Right
    vec2 U = vec2(0.0, 1.0); // Up
    vec2 container = glyph.xy + 2.0 * padding;
    bool flip = false;

    switch (flags & 0x0Fu) {
        case 0u: // Bottom left anchor
            cornerOffset = vec2(0.0, 0.0) + padding;
            break;
        case 1u: // Bottom anchor
            cornerOffset = vec2(-0.5 * container.x, 0.0) + padding;
            break;
        case 2u: // Bottom right anchor
            cornerOffset = vec2(-container.x, 0.0) + padding;
            break;
        case 3u: // Right anchor
            cornerOffset = vec2(-container.x, -0.5 * container.y) + padding;
            break;
        case 4u: // Top right anchor
            cornerOffset = vec2(-container.x, -container.y) + padding;
            break;
        case 5u: // Top anchor
            cornerOffset = vec2(-0.5 * container.x, -container.y) + padding;
            break;
        case 6u: // Top left anchor
            cornerOffset = vec2(0.0, -container.y) + padding;
            break;
        case 7u: // Left anchor
            cornerOffset = vec2(0.0, -0.5 * container.y) + padding;
            break;
        default: // Centered
            cornerOffset = -0.5 * glyph.xy;
    }

    vec2 b = screenNormal(gl_Position, MVP * vec4(baseline, 0.0));
    vec2 u = screenNormal(gl_Position, MVP * vec4(up, 0.0));

    uint af = flags & 0xFFu;

    if (af >= 48u) {
        u = normalize(u - b * dot(u, b));
        // This is the required offset... requires drawing a bunch of diagrams
        //   to understand why this works.  Essentially, however, we're finding
        //   which corner requires the most displacement to keep it outside
        //   the specified baseline.  Using abs allows to to check upper-left
        //   and lower right at the same time, for example.  The max just checks
        //   which corner needs more offset -- that's the one we want!

        // float d = max(
        //     abs(dot(u, glyph.xy)),
        //     abs(dot(u, vec2(glyph.x, -glyph.y)))
        // );

        float d =  abs(dot(u, 0.5*glyph.xy + vec2(padding.x, 0.0)));
        d = max(d, abs(dot(u, 0.5*glyph.xy + vec2(0.0, padding.y))));
        d = max(d, abs(dot(u, vec2(0.5*glyph.x + padding.x, -0.5*glyph.y))));
        d = max(d, abs(dot(u, vec2(0.5*glyph.x, -(0.5*glyph.y + padding.y)))));

        cornerOffset = d * u - 0.5 * glyph.xy;
    }
    else if (af >= 32u) {
        U = u;
        R = normalize(b - u * dot(u, b)); // Make this vector perp to U
        // If needed, flip the direction of R to form an orthonormal basis
        R = R * cross(vec3(R, 0.0), vec3(u, 0.0)).z; // crossproduct = -1 or 1!

        // Do we need to flip the text?
        if ((U.y < 0) && (U.x > -0.9)) {
            flip = true;
            // cornerOffset += container - cornerOffset;
             // - cornerOffset + 2.0 * padding;
            // U *= -1.0;
            // R *= -1.0;
        }
    } else if (af >= 16u) {
        U = normalize(u - b * dot(u, b)); // Make this vector perp to R
        // If needed, flip the direction of R to form an orthonormal basis
        R = b * cross(vec3(b, 0.0), vec3(U, 0.0)).z; // crossproduct = -1 or 1!

        // Do we need to flip the text?
        if ((U.y < 0) && (U.x > -0.9)) {
            flip = true;
            // cornerOffset += container - cornerOffset;
            // cornerOffset = container - cornerOffset + 2.0 * padding;
            // U *= -1.0;
            // R *= -1.0;
        }
    }

    gOut.right = R * emSize[0];
    gOut.up = U * emSize[1];
    gOut.atlas = atlas;
    gOut.visible = flags >> 8;

    if (flip) {
        cornerOffset += glyph.xy - glyph.zw;
        gOut.up *= -1;
        gOut.right *= -1;
    } else {
        cornerOffset += glyph.zw;
    }
    gOut.offset = cornerOffset.x * R + cornerOffset.y * U;
}
