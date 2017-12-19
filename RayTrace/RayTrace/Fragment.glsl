#version 400
in vec2 uv;
out vec4 fragment_color;

uniform sampler2D tex;

void main () {
	fragment_color = vec4 (texture(tex, uv).rgb, 1.0);
}
