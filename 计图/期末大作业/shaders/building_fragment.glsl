#version 330 core
struct Material {
    sampler2D diffuse;   // 漫反射纹理
    sampler2D specular;  // 镜面反射纹理
    float shininess;     // 高光强度
};

struct Light {
    vec3 position;       // 光源位置
    vec3 ambient;        // 环境光
    vec3 diffuse;        // 漫反射光
    vec3 specular;       // 镜面反射光
};

float fog_colour_red = 0.5f;
float fog_colour_green = 0.5f;
float fog_colour_blue = 0.5f;
vec4 fog_colour = vec4(fog_colour_red, fog_colour_green, fog_colour_blue, 1.0f);

in vec3 FragPos;         // 片段位置
in vec3 Normal;          // 法线向量
in vec2 TexCoords;       // 纹理坐标
in vec4 PosRelativeToCam;  // 片段相对摄像机位置

uniform vec3 viewPos;    // 观察者位置
uniform Material material; 
uniform Light light;

out vec4 FragColor;

void main() {
    // 从漫反射纹理中获取颜色，包括 Alpha 通道
    vec4 texColor = texture(material.diffuse, TexCoords);

    // 检查 Alpha 值，如果太透明，则丢弃当前片段
    if (texColor.a < 0.1) {
        discard;  // 丢弃透明的片段，避免显示白色背景
    }

    // 环境光
    vec3 ambient = light.ambient * texColor.rgb;

    // 漫反射
    vec3 norm = normalize(Normal);
    vec3 lightDir = normalize(light.position - FragPos);
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = light.diffuse * diff * texColor.rgb;

    // 镜面反射
    vec3 viewDir = normalize(viewPos - FragPos);
    vec3 halfDir = normalize(lightDir + viewDir); // 半向量
    float spec = pow(max(dot(viewDir, halfDir), 0.0), material.shininess);
    vec3 specular = light.specular * spec * texture(material.specular, TexCoords).rgb;

    // 合成最终颜色
    FragColor = vec4(ambient + diffuse + specular, texColor.a); // 保留透明度

    // 雾的影响
    float fog_maxdist = 50.0f;
    float fog_mindist = 30.0f;
    float dist = length(PosRelativeToCam.xyz);
    float fog_factor = (fog_maxdist - dist) / (fog_maxdist - fog_mindist);
    fog_factor = clamp(fog_factor, 0.0, 1.0);  // 可调节范围

    FragColor = mix(fog_colour, FragColor, fog_factor);
}
