#ifndef SKYBOX_H
#define SKYBOX_H

#include <vector>
#include <string>
#include <glad/glad.h>
#include <glm/glm.hpp>
#include "Shader.h"
#include "Camera.h"

class Skybox {
public:
    Skybox(const std::vector<std::string>& faces);
    void render(Shader& shader, Camera& camera, float aspectRatio); // 添加宽高比参数

private:
    GLuint VAO, VBO;
    GLuint textureID;

    GLuint loadCubemap(const std::vector<std::string>& faces);
};

#endif // SKYBOX_H

