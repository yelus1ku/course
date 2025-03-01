#ifndef MESH_H
#define MESH_H
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp> // ��Ӵ�ͷ�ļ�
#include <glad/glad.h> // holds all OpenGL type declarations

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "shader.h"

#include <string>
#include <vector>
using namespace std;

#define MAX_BONE_INFLUENCE 4

struct Vertex {
    // position
    glm::vec3 Position;
    // normal
    glm::vec3 Normal;
    // texCoords
    glm::vec2 TexCoords;
    // tangent
    glm::vec3 Tangent;
    // bitangent
    glm::vec3 Bitangent;
    //bone indexes which will influence this vertex
    int m_BoneIDs[MAX_BONE_INFLUENCE];
    //weights from each bone
    float m_Weights[MAX_BONE_INFLUENCE];
};

struct Texture {
    unsigned int id;
    string type;
    string path;
};

class Mesh {
public:
    // mesh Data
    vector<Vertex>       vertices;
    vector<unsigned int> indices;
    vector<Texture>      textures;
    unsigned int VAO;
    std::string name; // ��������������

    // �޸ĺ�Ĺ��캯��
    Mesh(vector<Vertex> vertices, vector<unsigned int> indices, vector<Texture> textures, const std::string& meshName = "")
    {
        this->vertices = vertices;
        this->indices = indices;
        this->textures = textures;
        this->name = meshName; // ������������

        // ���ö��㻺����������ָ��
        setupMesh();
    }

    // render the mesh
    void Draw(Shader& shader, const glm::mat4& extraTransform = glm::mat4(1.0f)) const
    {
        // ��ģ�;��󴫵ݵ���ɫ���У����ڶ����ģ�Ϳռ䵽����ռ��ת��
        GLint modelLoc = glGetUniformLocation(shader.ID, "model"); // ��ȡ��ɫ���� 'model' ������λ��
        glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(extraTransform)); // ��ģ�;��󴫵ݵ���ɫ��

        // ���������������Ӧ������Ԫ��GL_TEXTURE0��
        glActiveTexture(GL_TEXTURE0);
        // ������������󶨵���ɫ���е� 'material.diffuse' ����
        glUniform1i(glGetUniformLocation(shader.ID, "material.diffuse"), 0);
        // ��ȡ���������� ID�����󶨵���ǰ�������Ԫ
        glBindTexture(GL_TEXTURE_2D, getTextureByType("texture_diffuse"));

        // ����淴�������Ӧ������Ԫ��GL_TEXTURE1��
        glActiveTexture(GL_TEXTURE1);
        // �����淴������󶨵���ɫ���е� 'material.specular' ����
        glUniform1i(glGetUniformLocation(shader.ID, "material.specular"), 1);
        // ��ȡ���淴������ ID�����󶨵���ǰ�������Ԫ
        glBindTexture(GL_TEXTURE_2D, getTextureByType("texture_specular"));

        // �󶨵�ǰ����� VAO������������󣩣����ö������ݵĲ���
        glBindVertexArray(VAO);
        // ���������е�������
        glDrawElements(GL_TRIANGLES, static_cast<unsigned int>(indices.size()), GL_UNSIGNED_INT, 0);
        // ��� VAO������Ժ������Ʋ�������
        glBindVertexArray(0);

        // ���û����ԪΪĬ�ϵ� GL_TEXTURE0��������Ⱦ״̬��һ����
        glActiveTexture(GL_TEXTURE0);
    }


    // ���������������������Ͳ�������ID
    unsigned int getTextureByType(const string& type) const
    {
        for (const auto& texture : textures)
        {
            if (texture.type == type)
                return texture.id;
        }
        return 0; // ���δ�ҵ�������0��Ĭ������
    }


private:
    // render data 
    unsigned int VBO, EBO;

    // initializes all the buffer objects/arrays
    void setupMesh()
    {
        // create buffers/arrays
        glGenVertexArrays(1, &VAO);
        glGenBuffers(1, &VBO);
        glGenBuffers(1, &EBO);

        glBindVertexArray(VAO);
        // load data into vertex buffers
        glBindBuffer(GL_ARRAY_BUFFER, VBO);
        // A great thing about structs is that their memory layout is sequential for all its items.
        // The effect is that we can simply pass a pointer to the struct and it translates perfectly to a glm::vec3/2 array which
        // again translates to 3/2 floats which translates to a byte array.
        glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(Vertex), &vertices[0], GL_STATIC_DRAW);

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int), &indices[0], GL_STATIC_DRAW);

        // set the vertex attribute pointers
        // vertex Positions
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)0);
        // vertex normals
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, Normal));
        // vertex texture coords
        glEnableVertexAttribArray(2);
        glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, TexCoords));
        // vertex tangent
        glEnableVertexAttribArray(3);
        glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, Tangent));
        // vertex bitangent
        glEnableVertexAttribArray(4);
        glVertexAttribPointer(4, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, Bitangent));
        // ids
        glEnableVertexAttribArray(5);
        glVertexAttribIPointer(5, 4, GL_INT, sizeof(Vertex), (void*)offsetof(Vertex, m_BoneIDs));

        // weights
        glEnableVertexAttribArray(6);
        glVertexAttribPointer(6, 4, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, m_Weights));
        glBindVertexArray(0);
    }
};
#endif
