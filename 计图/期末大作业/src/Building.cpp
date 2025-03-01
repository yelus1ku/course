

#include "Building.h"
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include <iostream>
#include "stb_image.h"
Building::Building(const std::string& objPath, const std::string& textureDirectory) {
    loadModel(objPath, textureDirectory);
    setupMesh();
}

void Building::loadModel(const std::string& path, const std::string& textureDirectory) {
    Assimp::Importer importer;
    const aiScene* scene = importer.ReadFile(path, aiProcess_Triangulate | aiProcess_FlipUVs | aiProcess_CalcTangentSpace);

    if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode) {
        std::cerr << "ERROR::ASSIMP::" << importer.GetErrorString() << std::endl;
        return;
    }

    processNode(scene->mRootNode, scene, textureDirectory);
}

void Building::processNode(aiNode* node, const aiScene* scene, const std::string& textureDirectory) {
    for (unsigned int i = 0; i < node->mNumMeshes; ++i) {
        aiMesh* mesh = scene->mMeshes[node->mMeshes[i]];
        processMesh(mesh, scene, textureDirectory);
    }

    for (unsigned int i = 0; i < node->mNumChildren; ++i) {
        processNode(node->mChildren[i], scene, textureDirectory);
    }
}

void Building::processMesh(aiMesh* mesh, const aiScene* scene, const std::string& textureDirectory) {
    unsigned int baseIndex = vertices.size();
    MeshEntry meshEntry;

    for (unsigned int i = 0; i < mesh->mNumVertices; ++i) {
        Vertex vertex;
        vertex.position = glm::vec3(mesh->mVertices[i].x, mesh->mVertices[i].y, mesh->mVertices[i].z);
        vertex.normal = mesh->HasNormals() ? glm::vec3(mesh->mNormals[i].x, mesh->mNormals[i].y, mesh->mNormals[i].z) : glm::vec3(0.0f);
        vertex.texCoords = mesh->HasTextureCoords(0) ? glm::vec2(mesh->mTextureCoords[0][i].x, mesh->mTextureCoords[0][i].y) : glm::vec2(0.0f);
       /*
       const float scaleFactor = 5.0f; // 缩放倍数，根据需要调整
vertex.texCoords = mesh->HasTextureCoords(0) 
    ? glm::vec2(mesh->mTextureCoords[0][i].x * scaleFactor, mesh->mTextureCoords[0][i].y * scaleFactor) 
    : glm::vec2(0.0f);

       */
        vertices.push_back(vertex);
    }

    for (unsigned int i = 0; i < mesh->mNumFaces; ++i) {
        aiFace face = mesh->mFaces[i];
        for (unsigned int j = 0; j < face.mNumIndices; ++j) {
            indices.push_back(face.mIndices[j] + baseIndex);
        }
    }

    meshEntry.baseIndex = baseIndex;
    meshEntry.indexCount = mesh->mNumFaces * 3;

    // 加载材质
    if (mesh->mMaterialIndex >= 0) {
        aiMaterial* material = scene->mMaterials[mesh->mMaterialIndex];
        aiString name;
        material->Get(AI_MATKEY_NAME, name);

        GLuint textureID = loadMaterialTexture(material, aiTextureType_DIFFUSE, textureDirectory);
        meshEntry.materialID = textureID;

        // 根据材质名称分类
        std::string matName = name.C_Str();
        if (matName.find("Bark") != std::string::npos) {
            treeTrunkEntries.push_back(meshEntry);
        }
        else if (matName.find("Tree_Branch") != std::string::npos) {
            treeLeavesEntries.push_back(meshEntry);
        }
    }
    meshEntries.push_back(meshEntry);
}





GLuint Building::loadMaterialTexture(aiMaterial* mat, aiTextureType type, const std::string& textureDirectory) {
    if (mat->GetTextureCount(type) > 0) {
        aiString path;
        mat->GetTexture(type, 0, &path);

        std::string fullPath = textureDirectory + "/" + std::string(path.C_Str());
        std::cout << "Attempting to load texture: " << fullPath << std::endl;

        GLuint textureID = loadTexture(fullPath);
        if (textureID != 0) {
            std::cout << "Texture successfully loaded: " << fullPath << " with ID: " << textureID << std::endl;
            return textureID;
        }
    }
    std::cerr << "No texture found for material of type: " << type << std::endl;
    return 0;
}



GLuint Building::loadTexture(const std::string& path) {
    GLuint textureID;
    glGenTextures(1, &textureID);

    int width, height, nrChannels;
    // 加载纹理
    unsigned char* data = stbi_load(path.c_str(), &width, &height, &nrChannels, 0);

    if (data) {
        GLenum format = (nrChannels == 4) ? GL_RGBA : GL_RGB;

        std::cout << "Successfully loaded texture: " << path << std::endl;
        std::cout << "  Width: " << width << ", Height: " << height
            << ", Channels: " << nrChannels << std::endl;

        // 绑定纹理并上传数据
        glBindTexture(GL_TEXTURE_2D, textureID);
        glTexImage2D(GL_TEXTURE_2D, 0, format, width, height, 0, format, GL_UNSIGNED_BYTE, data);
        glGenerateMipmap(GL_TEXTURE_2D);

        // 设置纹理参数
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

        stbi_image_free(data);
    }
    else {
        std::cerr << "Failed to load texture: " << path << std::endl;
        std::cerr << "  STB Error: " << stbi_failure_reason() << std::endl;
        stbi_image_free(data);
    }

    return textureID;
}


void Building::setupMesh() {
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glGenBuffers(1, &EBO);

    glBindVertexArray(VAO);

    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(Vertex), &vertices[0], GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int), &indices[0], GL_STATIC_DRAW);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, position));
    glEnableVertexAttribArray(0);

    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, texCoords));
    glEnableVertexAttribArray(1);

    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, normal));
    glEnableVertexAttribArray(2);

    glBindVertexArray(0);
}

void Building::render(Shader& shader, const glm::mat4& modelMatrix, const glm::mat4& viewMatrix, const glm::mat4& projectionMatrix) {
    shader.use();
    shader.setMat4("model", modelMatrix);
    shader.setMat4("view", viewMatrix);
    shader.setMat4("projection", projectionMatrix);

    glBindVertexArray(VAO);

    // 渲染树干
    glDisable(GL_BLEND);
    glEnable(GL_CULL_FACE);
    for (const auto& entry : treeTrunkEntries) {
        if (entry.materialID > 0) {
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, entry.materialID);
            shader.setInt("material.diffuse", 0);
        }
        glDrawElements(GL_TRIANGLES, entry.indexCount, GL_UNSIGNED_INT, (void*)(entry.baseIndex * sizeof(unsigned int)));
    }

    // 渲染树叶
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glDisable(GL_CULL_FACE);
    for (const auto& entry : treeLeavesEntries) {
        if (entry.materialID > 0) {
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, entry.materialID);
            shader.setInt("material.diffuse", 0);
        }
        glDrawElements(GL_TRIANGLES, entry.indexCount, GL_UNSIGNED_INT, (void*)(entry.baseIndex * sizeof(unsigned int)));
    }

    glBindVertexArray(0);

    // 恢复默认设置
    glDisable(GL_BLEND);
    glEnable(GL_CULL_FACE);
}

