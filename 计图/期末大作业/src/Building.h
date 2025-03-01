/*
#ifndef BUILDING_H
#define BUILDING_H

#include <string>
#include <vector>
#include <glad/glad.h>
#include "Shader.h"
#include "Camera.h"
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>
class Building {
public:
    Building(const std::string& modelPath);
    void render(Shader& shader, const glm::mat4& modelMatrix, Camera& camera, float aspectRatio);

private:
    GLuint VAO, VBO, EBO; // OpenGL 缓冲区
    GLuint diffuseTexture; // 漫反射纹理
    GLuint specularTexture; // 镜面反射纹理
    unsigned int indexCount; // 索引数量

    void loadModel(const std::string& path);
    void processMesh(aiMesh* mesh, const aiScene* scene);
    GLuint loadTexture(const std::string& path);
};

#endif // BUILDING_H
*/
/*
#ifndef BUILDING_H
#define BUILDING_H

#include <string>
#include <vector>
#include <map>
#include <unordered_map>
#include <glm/glm.hpp>
#include <glad/glad.h>
#include "Shader.h"

// 顶点结构体
struct Vertex {
    glm::vec3 position;
    glm::vec2 texCoords;
    glm::vec3 normal;
};

// 材质结构体
struct Material {
    GLuint diffuseMap = 0;
    GLuint specularMap = 0;
    GLuint normalMap = 0;
};

// 索引范围结构体
struct ObjectMaterialRange {
    unsigned int indexStart;
    unsigned int indexCount;
};

class Building {
public:
    Building(const std::string& objPath, const std::string& textureDirectory);

    void render(Shader& shader, const glm::mat4& modelMatrix, const glm::mat4& viewMatrix, const glm::mat4& projectionMatrix);

private:
    std::vector<Vertex> vertices;
    std::vector<unsigned int> indices;
    std::map<std::string, Material> materials;
    std::unordered_map<std::string, std::vector<ObjectMaterialRange>> objectIndexRanges;

    GLuint VAO, VBO, EBO;

    void loadOBJ(const std::string& path, const std::string& textureDirectory);
    void loadMTL(const std::string& path, const std::string& textureDirectory);
    GLuint loadTexture(const std::string& path);
    void setupMesh();

    glm::vec3 calculateNormal(glm::vec3 p1, glm::vec3 p2, glm::vec3 p3);
};

#endif // BUILDING_H
*/
#ifndef BUILDING_H
#define BUILDING_H
#include <map>
#include <unordered_map>
#include <vector>
#include <string>
#include <map>
#include <glm/glm.hpp>
#include <glad/glad.h>
#include "shader.h"

// 顶点结构体
struct Vertex {
    glm::vec3 position;   // 顶点位置
    glm::vec3 normal;     // 法向量
    glm::vec2 texCoords;  // 纹理坐标
    glm::vec3 tangent;    // 切线向量
    glm::vec3 bitangent;  // 副切线向量
};
// 索引范围结构体
struct ObjectMaterialRange {
    unsigned int indexStart;
    unsigned int indexCount;
};

// 网格条目结构体，用于存储网格数据
struct MeshEntry {
    unsigned int baseIndex;   // 索引的起始位置
    unsigned int indexCount;  // 索引的数量
    GLuint materialID;        // 关联的材质纹理 ID

    MeshEntry() : baseIndex(0), indexCount(0), materialID(0) {}
};

// 材质结构体，用于存储材质信息
struct Material {
    GLuint diffuseMap;  // 漫反射纹理
    Material() : diffuseMap(0) {}
};

class Building {
public:
    Building(const std::string& objPath, const std::string& textureDirectory);

    void render(Shader& shader, const glm::mat4& modelMatrix, const glm::mat4& viewMatrix, const glm::mat4& projectionMatrix);

private:
    // 顶点和索引数据
    struct MeshEntry {
        unsigned int baseIndex;
        unsigned int indexCount;
        GLuint materialID;     // 漫反射贴图
        GLuint normalMapID;    // 法线贴图
        GLuint roughnessMapID; // 粗糙度贴图
    };
    std::unordered_map<std::string, std::vector<ObjectMaterialRange>> objectIndexRanges;
    std::vector<MeshEntry> treeTrunkEntries; // 树干网格列表
    std::vector<MeshEntry> treeLeavesEntries; // 树叶网格列表
    std::vector<Vertex> vertices;
    std::vector<unsigned int> indices;

    // 网格条目列表
    std::vector<MeshEntry> meshEntries;

    // 材质存储
    std::map<std::string, Material> materials;

    // OpenGL VAO 和缓冲区对象
    GLuint VAO, VBO, EBO;

    void loadModel(const std::string& path, const std::string& textureDirectory);
    void processNode(struct aiNode* node, const struct aiScene* scene, const std::string& textureDirectory);
    void processMesh(struct aiMesh* mesh, const struct aiScene* scene, const std::string& textureDirectory);

    GLuint loadMaterialTexture(struct aiMaterial* mat, enum aiTextureType type, const std::string& textureDirectory);
    GLuint loadTexture(const std::string& path);
    glm::vec3 getRandomOffset(float range);
    void setupMesh();
};

#endif // BUILDING_H
