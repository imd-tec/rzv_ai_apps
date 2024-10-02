#version 300 es // Specify the version of GLSL for OpenGL ES 3.0
precision mediump float; // Define precision

layout(location = 0) in vec2 aPos;      // Vertex position
layout(location = 1) in vec2 aTexCoord; // Texture coordinate

out vec2 TexCoord; // Output texture coordinate to the fragment shader

uniform mat4 projection; // Projection matrix uniform

void main()
{
    gl_Position = projection * vec4(aPos, 0.0, 1.0); // Transform vertex position
    TexCoord = aTexCoord; // Pass texture coordinate to fragment shader
}
