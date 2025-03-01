#version 330 core
out vec4 FragColor;

float fog_colour_red = 0.5f;
float fog_colour_green = 0.5f;
float fog_colour_blue = 0.5f;
vec4 fog_colour = vec4(fog_colour_red, fog_colour_green, fog_colour_blue, 1.0f);

in vec2 TexCoords;
in vec4 PosRelativeToCam;  // 片段相对摄像机位置


uniform sampler2D diffuse;

void main() {

    FragColor = texture(diffuse, TexCoords);
     // 雾的影响
    float fog_maxdist = 50.0f;
    float fog_mindist = 30.0f;
    float dist = length(PosRelativeToCam.xyz);
    float fog_factor = (fog_maxdist - dist) / (fog_maxdist - fog_mindist);
    fog_factor = clamp(fog_factor, 0.5, 1.0);  // 可调节范围

    FragColor = mix(fog_colour, FragColor, fog_factor);

    

    
}
