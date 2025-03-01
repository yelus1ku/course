#version 330 core
struct Material {
    sampler2D diffuse;   // ����������
    sampler2D specular;  // ���淴������
    float shininess;     // �߹�ǿ��
};

struct Light {
    vec3 position;       // ��Դλ��
    vec3 ambient;        // ������
    vec3 diffuse;        // �������
    vec3 specular;       // ���淴���
};

float fog_colour_red = 0.5f;
float fog_colour_green = 0.5f;
float fog_colour_blue = 0.5f;
vec4 fog_colour = vec4(fog_colour_red, fog_colour_green, fog_colour_blue, 1.0f);

in vec3 FragPos;         // Ƭ��λ��
in vec3 Normal;          // ��������
in vec2 TexCoords;       // ��������
in vec4 PosRelativeToCam;  // Ƭ����������λ��

uniform vec3 viewPos;    // �۲���λ��
uniform Material material; 
uniform Light light;

out vec4 FragColor;

void main() {
    // �������������л�ȡ��ɫ������ Alpha ͨ��
    vec4 texColor = texture(material.diffuse, TexCoords);

    // ��� Alpha ֵ�����̫͸����������ǰƬ��
    if (texColor.a < 0.1) {
        discard;  // ����͸����Ƭ�Σ�������ʾ��ɫ����
    }

    // ������
    vec3 ambient = light.ambient * texColor.rgb;

    // ������
    vec3 norm = normalize(Normal);
    vec3 lightDir = normalize(light.position - FragPos);
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = light.diffuse * diff * texColor.rgb;

    // ���淴��
    vec3 viewDir = normalize(viewPos - FragPos);
    vec3 halfDir = normalize(lightDir + viewDir); // ������
    float spec = pow(max(dot(viewDir, halfDir), 0.0), material.shininess);
    vec3 specular = light.specular * spec * texture(material.specular, TexCoords).rgb;

    // �ϳ�������ɫ
    FragColor = vec4(ambient + diffuse + specular, texColor.a); // ����͸����

    // ���Ӱ��
    float fog_maxdist = 50.0f;
    float fog_mindist = 30.0f;
    float dist = length(PosRelativeToCam.xyz);
    float fog_factor = (fog_maxdist - dist) / (fog_maxdist - fog_mindist);
    fog_factor = clamp(fog_factor, 0.0, 1.0);  // �ɵ��ڷ�Χ

    FragColor = mix(fog_colour, FragColor, fog_factor);
}
