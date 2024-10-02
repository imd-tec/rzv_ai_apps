#version 300 es // Specify the version of GLSL for OpenGL ES 3.0
precision mediump float; // Define precision

in vec2 TexCoord; // Input from vertex shader
out vec4 FragColor; // Output color of the fragment

uniform sampler2D tex; // Texture sampler uniform
uniform vec4 col; // Color modulation uniform

void main()
{
    vec4 texColor = texture(tex, TexCoord); // Sample the texture
    FragColor = texColor * col; // Apply color modulation
}
