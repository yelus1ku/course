#define STB_PERLIN_IMPLEMENTATION
#include "TerrainGenerator.h"
#include "stb_image.h"
#include <vector>
#include <iostream>
#include <cstdlib>  // 用于随机数


FlatTerrain::FlatTerrain(float width, float length, const std::string& texturePath) {
    generateTerrain(width, length);         // 生成地形网格
    textureID = loadTexture(texturePath);   // 加载纹理
}

void FlatTerrain::generateTerrain(float width, float length) {
    std::vector<float> vertices;
    std::vector<unsigned int> indices;

    int gridX = 100; // 网格划分数
    int gridZ = 100;
    float dx = width / gridX;
    float dz = length / gridZ;

    float noiseScale = 0.1f;
    float heightScale = 2.0f; // 高度幅度
    float thickness = -2.0f; // 地形厚度（底面高度）

    // ---------------------
    // 生成顶部地形
    // ---------------------
    for (int z = 0; z <= gridZ; ++z) {
        for (int x = 0; x <= gridX; ++x) {
            float px = -width / 2 + x * dx;
            float pz = -length / 2 + z * dz;

            // 使用 Perlin 噪声生成高度
            float py = stb_perlin_noise3(px * noiseScale, 0.0f, pz * noiseScale, 0, 0, 0) * heightScale;

            vertices.push_back(px);  // X
            vertices.push_back(py);  // Y
            vertices.push_back(pz);  // Z

            vertices.push_back(x / (float)gridX); // U 纹理坐标
            vertices.push_back(z / (float)gridZ); // V 纹理坐标
        }
    }

    // 生成顶部索引
    for (int z = 0; z < gridZ; ++z) {
        for (int x = 0; x < gridX; ++x) {
            int topLeft = z * (gridX + 1) + x;
            int topRight = topLeft + 1;
            int bottomLeft = (z + 1) * (gridX + 1) + x;
            int bottomRight = bottomLeft + 1;

            indices.push_back(topLeft);
            indices.push_back(bottomLeft);
            indices.push_back(topRight);

            indices.push_back(topRight);
            indices.push_back(bottomLeft);
            indices.push_back(bottomRight);
        }
    }

    // ---------------------
    // 生成底部地形
    // ---------------------
    int baseVertexOffset = vertices.size() / 5;
    for (int z = 0; z <= gridZ; ++z) {
        for (int x = 0; x <= gridX; ++x) {
            float px = -width / 2 + x * dx;
            float pz = -length / 2 + z * dz;

            vertices.push_back(px);       // X
            vertices.push_back(thickness); // Y（固定底面高度）
            vertices.push_back(pz);       // Z

            vertices.push_back(x / (float)gridX);
            vertices.push_back(z / (float)gridZ);
        }
    }

    // 生成底部索引（与顶部反向）
    for (int z = 0; z < gridZ; ++z) {
        for (int x = 0; x < gridX; ++x) {
            int topLeft = baseVertexOffset + z * (gridX + 1) + x;
            int topRight = topLeft + 1;
            int bottomLeft = baseVertexOffset + (z + 1) * (gridX + 1) + x;
            int bottomRight = bottomLeft + 1;

            indices.push_back(topRight);
            indices.push_back(bottomLeft);
            indices.push_back(topLeft);

            indices.push_back(bottomRight);
            indices.push_back(bottomLeft);
            indices.push_back(topRight);
        }
    }

    // ---------------------
    // 生成侧面
    // ---------------------
    for (int x = 0; x < gridX; ++x) {
        // 前后面
        int topFront = x;
        int bottomFront = baseVertexOffset + x;

        int topBack = (gridZ * (gridX + 1)) + x;
        int bottomBack = baseVertexOffset + (gridZ * (gridX + 1)) + x;

        // 前面
        indices.push_back(topFront);
        indices.push_back(bottomFront);
        indices.push_back(topFront + 1);
        indices.push_back(topFront + 1);
        indices.push_back(bottomFront);
        indices.push_back(bottomFront + 1);

        // 后面
        indices.push_back(topBack);
        indices.push_back(topBack + 1);
        indices.push_back(bottomBack);
        indices.push_back(bottomBack);
        indices.push_back(topBack + 1);
        indices.push_back(bottomBack + 1);
    }
    
    // ...

    // 上传到 OpenGL
    indexCount = indices.size();

    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glGenBuffers(1, &EBO);

    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(float), vertices.data(), GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int), indices.data(), GL_STATIC_DRAW);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);

    glBindVertexArray(0);
}

GLuint FlatTerrain::loadTexture(const std::string& path) {
    GLuint texture;
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);

    int width, height, nrChannels;
    unsigned char* data = stbi_load(path.c_str(), &width, &height, &nrChannels, 0);
    if (data) {
        GLenum format = (nrChannels == 4) ? GL_RGBA : GL_RGB;
        glTexImage2D(GL_TEXTURE_2D, 0, format, width, height, 0, format, GL_UNSIGNED_BYTE, data);
        glGenerateMipmap(GL_TEXTURE_2D);
        stbi_image_free(data);
    }
    else {
        std::cerr << "Failed to load texture: " << path << std::endl;
    }

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    return texture;
}

void FlatTerrain::render(Shader& shader, Camera& camera, float aspectRatio) {
    shader.use();

    glm::mat4 model = glm::mat4(1.0f);
    model = glm::translate(model, glm::vec3(0.0f, -1.0f, 0.0f));
    shader.setMat4("model", model);
    shader.setMat4("view", camera.GetViewMatrix());
    shader.setMat4("projection", glm::perspective(glm::radians(camera.Zoom), aspectRatio, 0.1f, 100.0f));

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, textureID);

    glBindVertexArray(VAO);
    glDrawElements(GL_TRIANGLES, indexCount, GL_UNSIGNED_INT, 0);
    glBindVertexArray(0);
}
