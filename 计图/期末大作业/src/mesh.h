#ifndef MESH_H
#define MESH_H
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp> // 添加此头文件
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
    std::string name; // 新增：网格名称

    // 修改后的构造函数
    Mesh(vector<Vertex> vertices, vector<unsigned int> indices, vector<Texture> textures, const std::string& meshName = "")
    {
        this->vertices = vertices;
        this->indices = indices;
        this->textures = textures;
        this->name = meshName; // 设置网格名称

        // 设置顶点缓冲区和属性指针
        setupMesh();
    }

    // render the mesh
    void Draw(Shader& shader, const glm::mat4& extraTransform = glm::mat4(1.0f)) const
    {
        // 将模型矩阵传递到着色器中，用于顶点从模型空间到世界空间的转换
        GLint modelLoc = glGetUniformLocation(shader.ID, "model"); // 获取着色器中 'model' 变量的位置
        glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(extraTransform)); // 将模型矩阵传递到着色器

        // 激活漫反射纹理对应的纹理单元（GL_TEXTURE0）
        glActiveTexture(GL_TEXTURE0);
        // 将漫反射纹理绑定到着色器中的 'material.diffuse' 变量
        glUniform1i(glGetUniformLocation(shader.ID, "material.diffuse"), 0);
        // 获取漫反射纹理 ID，并绑定到当前活动的纹理单元
        glBindTexture(GL_TEXTURE_2D, getTextureByType("texture_diffuse"));

        // 激活镜面反射纹理对应的纹理单元（GL_TEXTURE1）
        glActiveTexture(GL_TEXTURE1);
        // 将镜面反射纹理绑定到着色器中的 'material.specular' 变量
        glUniform1i(glGetUniformLocation(shader.ID, "material.specular"), 1);
        // 获取镜面反射纹理 ID，并绑定到当前活动的纹理单元
        glBindTexture(GL_TEXTURE_2D, getTextureByType("texture_specular"));

        // 绑定当前网格的 VAO（顶点数组对象），设置顶点数据的布局
        glBindVertexArray(VAO);
        // 绘制网格中的三角形
        glDrawElements(GL_TRIANGLES, static_cast<unsigned int>(indices.size()), GL_UNSIGNED_INT, 0);
        // 解绑 VAO，避免对后续绘制产生干扰
        glBindVertexArray(0);

        // 重置活动纹理单元为默认的 GL_TEXTURE0，保持渲染状态的一致性
        glActiveTexture(GL_TEXTURE0);
    }


    // 辅助函数：根据纹理类型查找纹理ID
    unsigned int getTextureByType(const string& type) const
    {
        for (const auto& texture : textures)
        {
            if (texture.type == type)
                return texture.id;
        }
        return 0; // 如果未找到，返回0（默认纹理）
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
