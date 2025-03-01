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
    GLuint VAO, VBO, EBO; // OpenGL ������
    GLuint diffuseTexture; // ����������
    GLuint specularTexture; // ���淴������
    unsigned int indexCount; // ��������

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

// ����ṹ��
struct Vertex {
    glm::vec3 position;
    glm::vec2 texCoords;
    glm::vec3 normal;
};

// ���ʽṹ��
struct Material {
    GLuint diffuseMap = 0;
    GLuint specularMap = 0;
    GLuint normalMap = 0;
};

// ������Χ�ṹ��
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

// ����ṹ��
struct Vertex {
    glm::vec3 position;   // ����λ��
    glm::vec3 normal;     // ������
    glm::vec2 texCoords;  // ��������
    glm::vec3 tangent;    // ��������
    glm::vec3 bitangent;  // ����������
};
// ������Χ�ṹ��
struct ObjectMaterialRange {
    unsigned int indexStart;
    unsigned int indexCount;
};

// ������Ŀ�ṹ�壬���ڴ洢��������
struct MeshEntry {
    unsigned int baseIndex;   // ��������ʼλ��
    unsigned int indexCount;  // ����������
    GLuint materialID;        // �����Ĳ������� ID

    MeshEntry() : baseIndex(0), indexCount(0), materialID(0) {}
};

// ���ʽṹ�壬���ڴ洢������Ϣ
struct Material {
    GLuint diffuseMap;  // ����������
    Material() : diffuseMap(0) {}
};

class Building {
public:
    Building(const std::string& objPath, const std::string& textureDirectory);

    void render(Shader& shader, const glm::mat4& modelMatrix, const glm::mat4& viewMatrix, const glm::mat4& projectionMatrix);

private:
    // �������������
    struct MeshEntry {
        unsigned int baseIndex;
        unsigned int indexCount;
        GLuint materialID;     // ��������ͼ
        GLuint normalMapID;    // ������ͼ
        GLuint roughnessMapID; // �ֲڶ���ͼ
    };
    std::unordered_map<std::string, std::vector<ObjectMaterialRange>> objectIndexRanges;
    std::vector<MeshEntry> treeTrunkEntries; // ���������б�
    std::vector<MeshEntry> treeLeavesEntries; // ��Ҷ�����б�
    std::vector<Vertex> vertices;
    std::vector<unsigned int> indices;

    // ������Ŀ�б�
    std::vector<MeshEntry> meshEntries;

    // ���ʴ洢
    std::map<std::string, Material> materials;

    // OpenGL VAO �ͻ���������
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
