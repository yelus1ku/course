#ifndef TERRAIN_GENERATOR_H
#define TERRAIN_GENERATOR_H

#include <glad/glad.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <string>
#include "Shader.h"
#include "Camera.h"
#include "stb_perlin.h"
class FlatTerrain {
public:
    // ���캯����������εĿ�ȡ����Ⱥ�����·��
    FlatTerrain(float width, float length, const std::string& texturePath);

    // ��Ⱦ����
    void render(Shader& shader, Camera& camera, float aspectRatio);

private:
    GLuint VAO, VBO, EBO;  // OpenGL ����������
    GLuint textureID;      // ����ID
    size_t indexCount;     // ��������

    // ˽�з��������ɵ��ζ�������
    void generateTerrain(float width, float length);

    // ˽�з�������������
    GLuint loadTexture(const std::string& path);
};

#endif
