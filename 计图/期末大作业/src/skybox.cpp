#include "Skybox.h"
#include <stb_image.h>
#include <iostream>

Skybox::Skybox(const std::vector<std::string>& faces) {
    textureID = loadCubemap(faces);

    float scaleFactor = 10.0f; // 放大因子

    float skyboxVertices[] = {
        // positions
        -1.0f * scaleFactor,  1.0f * scaleFactor, -1.0f * scaleFactor,
        -1.0f * scaleFactor, -1.0f * scaleFactor, -1.0f * scaleFactor,
         1.0f * scaleFactor, -1.0f * scaleFactor, -1.0f * scaleFactor,
         1.0f * scaleFactor, -1.0f * scaleFactor, -1.0f * scaleFactor,
         1.0f * scaleFactor,  1.0f * scaleFactor, -1.0f * scaleFactor,
        -1.0f * scaleFactor,  1.0f * scaleFactor, -1.0f * scaleFactor,

        -1.0f * scaleFactor, -1.0f * scaleFactor,  1.0f * scaleFactor,
        -1.0f * scaleFactor, -1.0f * scaleFactor, -1.0f * scaleFactor,
        -1.0f * scaleFactor,  1.0f * scaleFactor, -1.0f * scaleFactor,
        -1.0f * scaleFactor,  1.0f * scaleFactor, -1.0f * scaleFactor,
        -1.0f * scaleFactor,  1.0f * scaleFactor,  1.0f * scaleFactor,
        -1.0f * scaleFactor, -1.0f * scaleFactor,  1.0f * scaleFactor,

         1.0f * scaleFactor, -1.0f * scaleFactor, -1.0f * scaleFactor,
         1.0f * scaleFactor, -1.0f * scaleFactor,  1.0f * scaleFactor,
         1.0f * scaleFactor,  1.0f * scaleFactor,  1.0f * scaleFactor,
         1.0f * scaleFactor,  1.0f * scaleFactor,  1.0f * scaleFactor,
         1.0f * scaleFactor,  1.0f * scaleFactor, -1.0f * scaleFactor,
         1.0f * scaleFactor, -1.0f * scaleFactor, -1.0f * scaleFactor,

        -1.0f * scaleFactor, -1.0f * scaleFactor,  1.0f * scaleFactor,
        -1.0f * scaleFactor,  1.0f * scaleFactor,  1.0f * scaleFactor,
         1.0f * scaleFactor,  1.0f * scaleFactor,  1.0f * scaleFactor,
         1.0f * scaleFactor,  1.0f * scaleFactor,  1.0f * scaleFactor,
         1.0f * scaleFactor, -1.0f * scaleFactor,  1.0f * scaleFactor,
        -1.0f * scaleFactor, -1.0f * scaleFactor,  1.0f * scaleFactor,

        -1.0f * scaleFactor,  1.0f * scaleFactor, -1.0f * scaleFactor,
         1.0f * scaleFactor,  1.0f * scaleFactor, -1.0f * scaleFactor,
         1.0f * scaleFactor,  1.0f * scaleFactor,  1.0f * scaleFactor,
         1.0f * scaleFactor,  1.0f * scaleFactor,  1.0f * scaleFactor,
        -1.0f * scaleFactor,  1.0f * scaleFactor,  1.0f * scaleFactor,
        -1.0f * scaleFactor,  1.0f * scaleFactor, -1.0f * scaleFactor,

        -1.0f * scaleFactor, -1.0f * scaleFactor, -1.0f * scaleFactor,
        -1.0f * scaleFactor, -1.0f * scaleFactor,  1.0f * scaleFactor,
         1.0f * scaleFactor, -1.0f * scaleFactor, -1.0f * scaleFactor,
         1.0f * scaleFactor, -1.0f * scaleFactor, -1.0f * scaleFactor,
        -1.0f * scaleFactor, -1.0f * scaleFactor,  1.0f * scaleFactor,
         1.0f * scaleFactor, -1.0f * scaleFactor,  1.0f * scaleFactor
    };




    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);

    glBindVertexArray(VAO);

    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(skyboxVertices), &skyboxVertices, GL_STATIC_DRAW);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);

    glBindVertexArray(0);
}

void Skybox::render(Shader& shader, Camera& camera, float aspectRatio) {
    // 使用 GL_LEQUAL 代替 GL_LESS，确保深度测试允许天空盒渲染在最远处
    glDepthFunc(GL_LEQUAL);

    shader.use();

    glm::mat4 view = glm::mat4(glm::mat3(camera.GetViewMatrix())); // 移除平移分量
    glm::mat4 projection = glm::perspective(glm::radians(camera.Zoom), aspectRatio, 0.1f, 100.0f);
    shader.setMat4("view", view);
    shader.setMat4("projection", projection);

    glBindVertexArray(VAO);
    glBindTexture(GL_TEXTURE_CUBE_MAP, textureID);
    glDrawArrays(GL_TRIANGLES, 0, 36);
    glBindVertexArray(0);

    // 恢复默认深度测试方式
    glDepthFunc(GL_LESS);
}



GLuint Skybox::loadCubemap(const std::vector<std::string>& faces) {
    GLuint textureID;
    glGenTextures(1, &textureID);
    glBindTexture(GL_TEXTURE_CUBE_MAP, textureID);

    int width, height, nrChannels;
    for (unsigned int i = 0; i < faces.size(); i++) {
        unsigned char* data = stbi_load(faces[i].c_str(), &width, &height, &nrChannels, 0);
        if (data) {
            glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i,
                0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, data);
            stbi_image_free(data);
        }
        else {
            std::cerr << "Cubemap texture failed to load at path: " << faces[i] << std::endl;
            stbi_image_free(data);
        }
    }
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);

    return textureID;
}
