#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp> // ���� value_ptr ����
#include <glm/gtc/type_ptr.hpp> // ��Ӵ�ͷ�ļ�
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <iomanip>
#include "Shader.h"
#include "Camera.h"
#include "Skybox.h"
#include "TerrainGenerator.h"
#include "model.h"
#include"stb_perlin.h"
#include <cmath>
#include "stb_perlin.h"

// ��Ļ���
const unsigned int SCREEN_WIDTH = 1600;
const unsigned int SCREEN_HEIGHT = 1200;

// ���������
// ��������� - ������������
Camera camera(glm::vec3(0.0f, 2.0f, 25.0f)); // �����λ�ã�Զ�뽨����һ�����룩
// ��������ϸ�����Զ��λ��
// ������ʼλ���Կ���ģ��

//float lastX = SCREEN_WIDTH / 2.0f, lastY = SCREEN_HEIGHT / 2.0f;
bool firstMouse = true;

// ʱ�����
float deltaTime = 0.0f;
float lastFrame = 0.0f;

// �ص���������
void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void mouse_callback(GLFWwindow* window, double xpos, double ypos);
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);
void processInput(GLFWwindow* window);

int main() {
    // ��ʼ�� GLFW
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_CORE_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    // ��������
    GLFWwindow* window = glfwCreateWindow(SCREEN_WIDTH, SCREEN_HEIGHT, "Residential Buildings", NULL, NULL);
    if (window == NULL) {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);

    // ���ûص�����
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    glfwSetCursorPosCallback(window, mouse_callback);
    glfwSetScrollCallback(window, scroll_callback);
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

    // ��ʼ�� GLAD
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cerr << "Failed to initialize GLAD" << std::endl;
        return -1;
    }

    // OpenGL ����
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    glEnable(GL_DEPTH_TEST);        // ������Ȳ���
    glEnable(GL_CULL_FACE);
    glCullFace(GL_BACK);  // �޳�����
 
    glDisable(GL_CULL_FACE);  // �����޳����е���



    camera = Camera(glm::vec3(0.0f, 7.0f, 35.0f));// 
    camera.Yaw = -85.0f;  // ���ҷ���Ƕ�
    camera.Pitch = 2.0f; // ���·���Ƕ�
    camera.updateCameraVectors();

    // ������ɫ��
    Shader ourShader("shaders/shipshader_vertex.glsl", "shaders/shipshader_fragment.glsl");
    Shader buildingShader("shaders/building_vertex.glsl", "shaders/building_fragment.glsl");
    Shader skyboxShader("shaders/skybox_vertex.glsl", "shaders/skybox_fragment.glsl");
    Shader streetShader("shaders/terrain_vertex.glsl", "shaders/terrain_fragment.glsl");
    Model Solar("resourses/models/uploads_files_3484563_ship.obj");
    Model Tree("resourses/models/uploads_files_3461467_tree.obj");
    // ���� Shader ������󶨵�
    streetShader.use();
    streetShader.setVec3("diffuse", glm::vec3(0.5f, 0.5f, 0.5f));  // ������
    streetShader.setInt("diffuse", 0); // ������������󶨵�����Ԫ 0
    // ���ù��պͲ��ʲ���
    glm::vec3 lightPos(0.0f, 0.0f, 0.0f); // ��Դλ��
    
    // ʹ����ɫ�����󶨲��ʺ͹�������
    buildingShader.use();

    // ���ù�Դ����
    buildingShader.setVec3("light.position", lightPos);
    buildingShader.setVec3("light.ambient", glm::vec3(0.4f,0.4f,0.4f));  // ������
    buildingShader.setVec3("light.diffuse", glm::vec3(0.5f, 0.5f, 0.5f));  // ������
    buildingShader.setVec3("light.specular", glm::vec3(1.0f, 1.0f, 1.0f)); // �����
    // ���ù۲���λ�ã����λ�ã�
    buildingShader.setVec3("viewPos", camera.Position);
    buildingShader.setFloat("material.shininess", 32.0f);
    // �󶨲�����ͼ������Ԫ
    buildingShader.setInt("material.diffuse", 0);  // ��������ͼ�󶨵�����Ԫ 0
    buildingShader.setInt("material.specular", 1); // ���淴����ͼ�󶨵�����Ԫ 1
    buildingShader.setFloat("material.shininess", 32.0f); // �߹�ϵ��
    





   


    


    //----------------
    
    // ʹ����ɫ�����󶨲��ʺ͹�������
    ourShader.use();

    // ���ù�Դ����
    ourShader.setVec3("light.position", lightPos);
    ourShader.setVec3("light.ambient", glm::vec3(0.2f, 0.2f, 0.2f));  // ������
    ourShader.setVec3("light.diffuse", glm::vec3(0.5f, 0.5f, 0.5f));  // ������
    ourShader.setVec3("light.specular", glm::vec3(1.0f, 1.0f, 1.0f)); // �����


    // ���ù۲���λ�ã����λ�ã�
    ourShader.setVec3("viewPos", camera.Position);
    ourShader.setFloat("material.shininess", 32.0f);
    // �󶨲�����ͼ������Ԫ
    
    ourShader.setInt("material.diffuse", 0);  // ��������ͼ�󶨵�����Ԫ 0
    ourShader.setInt("material.specular", 1); // ���淴����ͼ�󶨵�����Ԫ 1
    ourShader.setFloat("material.shininess", 32.0f); // �߹�ϵ��
    
    //---------------
    







    // ������պ�
    std::vector<std::string> skyboxFaces = {
    "resourses/skybox/chenwu_L.jpg","resourses/skybox/chenwu_R.jpg",
    "resourses/skybox/chenwu_U.jpg","resourses/skybox/chenwu_D.jpg",
    "resourses/skybox/chenwu_F.jpg", "resourses/skybox/chenwu_B.jpg"
    };
    Skybox skybox(skyboxFaces);

    // ���ص���
    FlatTerrain street(60.0f, 60.0f, "resourses/models/textures/sandydrysoil-albedo2b.png");

  

    // ��Ⱦѭ��
    glm::mat4 modelMatrix = glm::mat4(1.0f);
    modelMatrix = glm::scale(modelMatrix, glm::vec3(0.01f));  // ��Сģ�͵����ʱ���
    modelMatrix = glm::translate(modelMatrix, glm::vec3(0.0f, 0.0f, 0.0f));  // ƽ�Ƶ�������������
    while (!glfwWindowShouldClose(window)) {
        // ����ʱ��
        float currentFrame = glfwGetTime();
        deltaTime = currentFrame - lastFrame;
        lastFrame = currentFrame;

        // ���봦��
        processInput(window);

        // ����
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
 
        float scaleFactor = 1.0f;    //��������


        for (const auto& mesh : Tree.meshes) {
            glm::mat4 modelMatrix = glm::mat4(1.0f);

            if (mesh.name == "Tree_Branch") {
                // ��ȡ��ǰʱ��
                float time = glfwGetTime();

                // ���һζ��ķ��Ⱥ�Ƶ��
                float shakeAmplitudeX = 0.07f; // ����������
                float shakeFrequencyX1 = 1.0f; // ��Ƶ��
                float shakeFrequencyX2 = 1.5f; // ��Ƶ��

                // ���»ζ��ķ��Ⱥ�Ƶ��
                float shakeAmplitudeY = 0.06f; // ����������
                float shakeFrequencyY1 = 0.6f;
                float shakeFrequencyY2 = 1.2f;

                // �������ң�X�ᣩƫ�ƣ����Ҳ�
                float timeOffset = time + 0.5f; // ����ʱ��ƫ��
                float shakeOffsetX = shakeAmplitudeX * (
                    sin(timeOffset * shakeFrequencyX1) +
                    0.5f * cos(timeOffset * shakeFrequencyX2)
                    );

                // ʹ�� Perlin ��������ƽ����
                float noiseX = stb_perlin_noise3(time * 0.08f, time * 0.05f, time * 0.1f, 0, 0, 0); // ʹ��ʱ����Ϊ��������
                shakeOffsetX += 0.01f * noiseX; // ���ƽ��������X�ᣩ

                // �������£�Y�ᣩƫ�ƣ����Ҳ�
                float rawShakeY = shakeAmplitudeY * (
                    sin(timeOffset * shakeFrequencyY1) +
                    0.3f * cos(timeOffset * shakeFrequencyY2)
                    );
                float shakeOffsetY = (rawShakeY > 0) ? rawShakeY * 0.5f : rawShakeY; // ����ƫ����С

                // ʹ�� Perlin ��������ƽ����
                float noiseY = stb_perlin_noise3(time * 0.08f, time * 0.05f, time * 0.1f, 0, 0, 0); // ʹ��ʱ����Ϊ��������
                shakeOffsetY += 0.01f * noiseY; // ���ƽ��������Y�ᣩ


                // Ӧ��ƽ�ƺ����ű任
                modelMatrix = glm::translate(modelMatrix, glm::vec3(-5.0f+shakeOffsetX, -0.5+shakeOffsetY, 10.0f));
                modelMatrix = glm::scale(modelMatrix, glm::vec3(3.5f));

            }
            else if(mesh.name == "Cube_Cube.001")
            {
                modelMatrix = glm::translate(modelMatrix, glm::vec3(-5.0f, -0.55f, 10.1f));
                modelMatrix = glm::scale(modelMatrix, glm::vec3(3.5f));
            }


            buildingShader.use();
            buildingShader.setMat4("view", camera.GetViewMatrix());
            buildingShader.setMat4("projection", glm::perspective(glm::radians(camera.Zoom), (float)SCREEN_WIDTH / SCREEN_HEIGHT, 0.1f, 200.0f));

            mesh.Draw(buildingShader, modelMatrix);
        }
        //��
        for (const auto& mesh : Solar.meshes) {
            glm::mat4 model = glm::mat4(1.0f);
            model = glm::translate(model, glm::vec3(3.0f, 3.0f, -17.1f));
            // ˮƽ��ת���� Z ����ת 45 ��
            //model = glm::rotate(model, glm::radians(45.0f), glm::vec3(0.0f, 0.0f, 1.0f));

            // ����Ļ����ת���� X ����ת 45 ��
           // model = glm::rotate(model, glm::radians(45.0f), glm::vec3(1.0f, 0.0f, 0.0f));
            // ˮƽ��ת���� Y ����ת 45 ��
            model = glm::rotate(model, glm::radians(60.0f), glm::vec3(0.0f, 1.0f, 0.0f));
            model = glm::scale(model, glm::vec3(2.5f));
            ourShader.use();
            ourShader.setMat4("view", camera.GetViewMatrix());
            ourShader.setMat4("projection", glm::perspective(glm::radians(camera.Zoom), (float)SCREEN_WIDTH / SCREEN_HEIGHT, 0.1f, 200.0f));

            mesh.Draw(ourShader, model);
        }




        glm::mat4 model = glm::mat4(1.0f);
        model = glm::translate(model, glm::vec3(0.0f, -1.0f, 0.0f));
        // ��Ⱦ����
        
        streetShader.use();
        street.render(streetShader, camera, (float)SCREEN_WIDTH / SCREEN_HEIGHT);
        
        // ��Ⱦ��պ�
        skyboxShader.use();
        skybox.render(skyboxShader, camera, (float)SCREEN_WIDTH / SCREEN_HEIGHT);



        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glfwTerminate();
    return 0;
}

// �ص�����ʵ��
void framebuffer_size_callback(GLFWwindow* window, int width, int height) {
    glViewport(0, 0, width, height);
}

void mouse_callback(GLFWwindow* window, double xpos, double ypos) {
    static bool firstMouse = true;
    static float lastX = SCREEN_WIDTH / 2.0f;
    static float lastY = SCREEN_HEIGHT / 2.0f;

    if (firstMouse) {
        lastX = xpos;
        lastY = ypos;
        firstMouse = false;
    }

    float xoffset = xpos - lastX;
    float yoffset = lastY - ypos; // Y �����Ƿ���
    lastX = xpos;
    lastY = ypos;

    camera.processMouseMovement(xoffset, yoffset);
}

void scroll_callback(GLFWwindow* window, double xoffset, double yoffset) {
    camera.processMouseScroll(yoffset);
}

void processInput(GLFWwindow* window) {
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
        glfwSetWindowShouldClose(window, true); // ��Ǵ���Ϊ�ر�
    }
    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) {
        std::cout << "W key pressed" << std::endl;
        camera.processKeyboard(FORWARD, deltaTime);
    }
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) {
        std::cout << "S key pressed" << std::endl;
        camera.processKeyboard(BACKWARD, deltaTime);
    }
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) {
        std::cout << "A key pressed" << std::endl;
        camera.processKeyboard(LEFT, deltaTime);
    }
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) {
        std::cout << "D key pressed" << std::endl;
        camera.processKeyboard(RIGHT, deltaTime);
    }
}
