#version 100
#ifdef GL_ES
    precision mediump float;
#endif
uniform sampler2D Texture;
varying vec2 Frag_UV;
varying vec4 Frag_Color;
void main()
{
    vec4 texColor = texture2D(Texture, Frag_UV.st);
    gl_FragColor = vec4(texColor.b, texColor.g, texColor.r, texColor.a);
    //gl_FragColor = Frag_Color * texture2D(Texture, Frag_UV.st);
}
