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
    // 构造函数：传入地形的宽度、长度和纹理路径
    FlatTerrain(float width, float length, const std::string& texturePath);

    // 渲染地形
    void render(Shader& shader, Camera& camera, float aspectRatio);

private:
    GLuint VAO, VBO, EBO;  // OpenGL 缓冲区对象
    GLuint textureID;      // 纹理ID
    size_t indexCount;     // 索引数量

    // 私有方法：生成地形顶点数据
    void generateTerrain(float width, float length);

    // 私有方法：加载纹理
    GLuint loadTexture(const std::string& path);
};

#endif
