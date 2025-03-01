#ifndef MODEL_H
#define MODEL_H

#include <glad/glad.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include "stb_image.h"
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

#include "mesh.h"
#include "shader.h"

#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <unordered_map>
#include <vector>
#include <memory>

using std::vector;
using std::string;
using std::unordered_map;
using std::cout;
using std::endl;

constexpr unsigned int MAX_BONES = 100;

// ��������ĸ�����������
inline unsigned int LoadTextureFromFile(const char* path, const string& directory, bool gamma = false);

// Model �ࣺ���غ͹��� 3D ģ��
class Model {
public:
    vector<Texture> textures_loaded; // �Ѽ��ص������棬�����ظ�����
    vector<Mesh> meshes;             // �洢ģ�͵�������������
    string directory;                // ģ���ļ���·��
    bool gammaCorrection;            // �Ƿ���� gamma ����

    // ���캯�������� 3D ģ���ļ�
    explicit Model(const string& path, bool gamma = false)
        : gammaCorrection(gamma)
    {
        loadModel(path);
    }
    
    // ����ģ�͵���������
    void Draw(Shader& shader) const {
        for (const auto& mesh : meshes) {
            mesh.Draw(shader);
        }
    }
    
private:
    // ����ģ���ļ�
    void loadModel(const string& path) {
        Assimp::Importer importer;
        const aiScene* scene = importer.ReadFile(path, aiProcess_Triangulate | aiProcess_GenSmoothNormals |
            aiProcess_FlipUVs | aiProcess_CalcTangentSpace);

        // �����ش���
        if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode) {
            cout << "ERROR::ASSIMP::" << importer.GetErrorString() << endl;
            return;
        }

        directory = path.substr(0, path.find_last_of('/'));
        processNode(scene->mRootNode, scene); // �Ӹ��ڵ㿪ʼ�ݹ鴦��
    }

    // �ݹ鴦��ڵ�
    void processNode(aiNode* node, const aiScene* scene) {
        // ����ǰ�ڵ����������
        for (unsigned int i = 0; i < node->mNumMeshes; i++) {
            aiMesh* mesh = scene->mMeshes[node->mMeshes[i]];
            meshes.emplace_back(processMesh(mesh, scene));
        }

        // �ݹ鴦���ӽڵ�
        for (unsigned int i = 0; i < node->mNumChildren; i++) {
            processNode(node->mChildren[i], scene);
        }
    }

    // ����������
    Mesh processMesh(aiMesh* mesh, const aiScene* scene) {
        vector<Vertex> vertices;
        vector<unsigned int> indices;
        vector<Texture> textures;

        // ����������
        for (unsigned int i = 0; i < mesh->mNumVertices; i++) {
            Vertex vertex;
            vertex.Position = glm::vec3(mesh->mVertices[i].x, mesh->mVertices[i].y, mesh->mVertices[i].z);
            vertex.Normal = mesh->HasNormals() ? glm::vec3(mesh->mNormals[i].x, mesh->mNormals[i].y, mesh->mNormals[i].z) : glm::vec3(0.0f);
            vertex.TexCoords = mesh->mTextureCoords[0] ? glm::vec2(mesh->mTextureCoords[0][i].x, mesh->mTextureCoords[0][i].y) : glm::vec2(0.0f);
            vertices.emplace_back(vertex);
        }

        // ������������
        for (unsigned int i = 0; i < mesh->mNumFaces; i++) {
            aiFace face = mesh->mFaces[i];
            for (unsigned int j = 0; j < face.mNumIndices; j++) {
                indices.push_back(face.mIndices[j]);
            }
        }

        // ���ز�������
        aiMaterial* material = scene->mMaterials[mesh->mMaterialIndex];
        loadMaterialTextures(material, aiTextureType_DIFFUSE, "texture_diffuse", textures);
        loadMaterialTextures(material, aiTextureType_SPECULAR, "texture_specular", textures);

        // ������������
        std::string meshName = mesh->mName.C_Str();
        if (meshName.empty()) {
            meshName = "Unnamed";
        }

        return Mesh(vertices, indices, textures, meshName);
    }


    // �������������ظ�����
    void loadMaterialTextures(aiMaterial* mat, aiTextureType type, const string& typeName, vector<Texture>& textures) {
        for (unsigned int i = 0; i < mat->GetTextureCount(type); i++) {
            aiString str;
            mat->GetTexture(type, i, &str);

            auto it = std::find_if(textures_loaded.begin(), textures_loaded.end(),
                [&](const Texture& t) { return t.path == str.C_Str(); });

            if (it != textures_loaded.end()) {
                textures.push_back(*it);
            }
            else {
                Texture texture;
                texture.id = LoadTextureFromFile(str.C_Str(), directory);
                texture.type = typeName;
                texture.path = str.C_Str();
                textures.push_back(texture);
                textures_loaded.push_back(texture);
            }
        }
    }
};

// ��������������
unsigned int LoadTextureFromFile(const char* path, const string& directory, bool gamma) {
    string filename = directory + '/' + string(path);

    unsigned int textureID;
    glGenTextures(1, &textureID);

    int width, height, nrComponents;
    unsigned char* data = stbi_load(filename.c_str(), &width, &height, &nrComponents, 0);
    if (data) {
        GLenum format = nrComponents == 1 ? GL_RED : (nrComponents == 3 ? GL_RGB : GL_RGBA);

        glBindTexture(GL_TEXTURE_2D, textureID);
        glTexImage2D(GL_TEXTURE_2D, 0, format, width, height, 0, format, GL_UNSIGNED_BYTE, data);
        glGenerateMipmap(GL_TEXTURE_2D);

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

        stbi_image_free(data);
    }
    else {
        cout << "Texture failed to load at path: " << path << endl;
        stbi_image_free(data);
    }
    return textureID;
}

#endif

