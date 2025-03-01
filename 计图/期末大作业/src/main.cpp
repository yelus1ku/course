#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp> // 包含 value_ptr 函数
#include <glm/gtc/type_ptr.hpp> // 添加此头文件
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

// 屏幕宽高
const unsigned int SCREEN_WIDTH = 1600;
const unsigned int SCREEN_HEIGHT = 1200;

// 摄像机设置
// 摄像机设置 - 面向建筑物正面
Camera camera(glm::vec3(0.0f, 2.0f, 25.0f)); // 摄像机位置（远离建筑物一定距离）
// 设置相机较高且稍远的位置
// 调整初始位置以看到模型

//float lastX = SCREEN_WIDTH / 2.0f, lastY = SCREEN_HEIGHT / 2.0f;
bool firstMouse = true;

// 时间变量
float deltaTime = 0.0f;
float lastFrame = 0.0f;

// 回调函数声明
void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void mouse_callback(GLFWwindow* window, double xpos, double ypos);
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);
void processInput(GLFWwindow* window);

int main() {
    // 初始化 GLFW
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_CORE_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    // 创建窗口
    GLFWwindow* window = glfwCreateWindow(SCREEN_WIDTH, SCREEN_HEIGHT, "Residential Buildings", NULL, NULL);
    if (window == NULL) {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);

    // 设置回调函数
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    glfwSetCursorPosCallback(window, mouse_callback);
    glfwSetScrollCallback(window, scroll_callback);
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

    // 初始化 GLAD
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cerr << "Failed to initialize GLAD" << std::endl;
        return -1;
    }

    // OpenGL 设置
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    glEnable(GL_DEPTH_TEST);        // 开启深度测试
    glEnable(GL_CULL_FACE);
    glCullFace(GL_BACK);  // 剔除背面
 
    glDisable(GL_CULL_FACE);  // 禁用剔除进行调试



    camera = Camera(glm::vec3(0.0f, 7.0f, 35.0f));// 
    camera.Yaw = -85.0f;  // 左右方向角度
    camera.Pitch = 2.0f; // 上下方向角度
    camera.updateCameraVectors();

    // 加载着色器
    Shader ourShader("shaders/shipshader_vertex.glsl", "shaders/shipshader_fragment.glsl");
    Shader buildingShader("shaders/building_vertex.glsl", "shaders/building_fragment.glsl");
    Shader skyboxShader("shaders/skybox_vertex.glsl", "shaders/skybox_fragment.glsl");
    Shader streetShader("shaders/terrain_vertex.glsl", "shaders/terrain_fragment.glsl");
    Model Solar("resourses/models/uploads_files_3484563_ship.obj");
    Model Tree("resourses/models/uploads_files_3461467_tree.obj");
    // 设置 Shader 的纹理绑定点
    streetShader.use();
    streetShader.setVec3("diffuse", glm::vec3(0.5f, 0.5f, 0.5f));  // 漫反射
    streetShader.setInt("diffuse", 0); // 将漫反射纹理绑定到纹理单元 0
    // 设置光照和材质参数
    glm::vec3 lightPos(0.0f, 0.0f, 0.0f); // 光源位置
    
    // 使用着色器并绑定材质和光照属性
    buildingShader.use();

    // 设置光源属性
    buildingShader.setVec3("light.position", lightPos);
    buildingShader.setVec3("light.ambient", glm::vec3(0.4f,0.4f,0.4f));  // 环境光
    buildingShader.setVec3("light.diffuse", glm::vec3(0.5f, 0.5f, 0.5f));  // 漫反射
    buildingShader.setVec3("light.specular", glm::vec3(1.0f, 1.0f, 1.0f)); // 镜面光
    // 设置观察者位置（相机位置）
    buildingShader.setVec3("viewPos", camera.Position);
    buildingShader.setFloat("material.shininess", 32.0f);
    // 绑定材质贴图到纹理单元
    buildingShader.setInt("material.diffuse", 0);  // 漫反射贴图绑定到纹理单元 0
    buildingShader.setInt("material.specular", 1); // 镜面反射贴图绑定到纹理单元 1
    buildingShader.setFloat("material.shininess", 32.0f); // 高光系数
    





   


    


    //----------------
    
    // 使用着色器并绑定材质和光照属性
    ourShader.use();

    // 设置光源属性
    ourShader.setVec3("light.position", lightPos);
    ourShader.setVec3("light.ambient", glm::vec3(0.2f, 0.2f, 0.2f));  // 环境光
    ourShader.setVec3("light.diffuse", glm::vec3(0.5f, 0.5f, 0.5f));  // 漫反射
    ourShader.setVec3("light.specular", glm::vec3(1.0f, 1.0f, 1.0f)); // 镜面光


    // 设置观察者位置（相机位置）
    ourShader.setVec3("viewPos", camera.Position);
    ourShader.setFloat("material.shininess", 32.0f);
    // 绑定材质贴图到纹理单元
    
    ourShader.setInt("material.diffuse", 0);  // 漫反射贴图绑定到纹理单元 0
    ourShader.setInt("material.specular", 1); // 镜面反射贴图绑定到纹理单元 1
    ourShader.setFloat("material.shininess", 32.0f); // 高光系数
    
    //---------------
    







    // 加载天空盒
    std::vector<std::string> skyboxFaces = {
    "resourses/skybox/chenwu_L.jpg","resourses/skybox/chenwu_R.jpg",
    "resourses/skybox/chenwu_U.jpg","resourses/skybox/chenwu_D.jpg",
    "resourses/skybox/chenwu_F.jpg", "resourses/skybox/chenwu_B.jpg"
    };
    Skybox skybox(skyboxFaces);

    // 加载地形
    FlatTerrain street(60.0f, 60.0f, "resourses/models/textures/sandydrysoil-albedo2b.png");

  

    // 渲染循环
    glm::mat4 modelMatrix = glm::mat4(1.0f);
    modelMatrix = glm::scale(modelMatrix, glm::vec3(0.01f));  // 缩小模型到合适比例
    modelMatrix = glm::translate(modelMatrix, glm::vec3(0.0f, 0.0f, 0.0f));  // 平移到世界坐标中心
    while (!glfwWindowShouldClose(window)) {
        // 计算时间
        float currentFrame = glfwGetTime();
        deltaTime = currentFrame - lastFrame;
        lastFrame = currentFrame;

        // 输入处理
        processInput(window);

        // 清屏
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
 
        float scaleFactor = 1.0f;    //缩放因子


        for (const auto& mesh : Tree.meshes) {
            glm::mat4 modelMatrix = glm::mat4(1.0f);

            if (mesh.name == "Tree_Branch") {
                // 获取当前时间
                float time = glfwGetTime();

                // 左右晃动的幅度和频率
                float shakeAmplitudeX = 0.07f; // 左右最大幅度
                float shakeFrequencyX1 = 1.0f; // 主频率
                float shakeFrequencyX2 = 1.5f; // 次频率

                // 上下晃动的幅度和频率
                float shakeAmplitudeY = 0.06f; // 上下最大幅度
                float shakeFrequencyY1 = 0.6f;
                float shakeFrequencyY2 = 1.2f;

                // 计算左右（X轴）偏移：正弦波
                float timeOffset = time + 0.5f; // 引入时间偏移
                float shakeOffsetX = shakeAmplitudeX * (
                    sin(timeOffset * shakeFrequencyX1) +
                    0.5f * cos(timeOffset * shakeFrequencyX2)
                    );

                // 使用 Perlin 噪声进行平滑化
                float noiseX = stb_perlin_noise3(time * 0.08f, time * 0.05f, time * 0.1f, 0, 0, 0); // 使用时间作为噪声输入
                shakeOffsetX += 0.01f * noiseX; // 添加平滑噪声（X轴）

                // 计算上下（Y轴）偏移：正弦波
                float rawShakeY = shakeAmplitudeY * (
                    sin(timeOffset * shakeFrequencyY1) +
                    0.3f * cos(timeOffset * shakeFrequencyY2)
                    );
                float shakeOffsetY = (rawShakeY > 0) ? rawShakeY * 0.5f : rawShakeY; // 向上偏移缩小

                // 使用 Perlin 噪声进行平滑化
                float noiseY = stb_perlin_noise3(time * 0.08f, time * 0.05f, time * 0.1f, 0, 0, 0); // 使用时间作为噪声输入
                shakeOffsetY += 0.01f * noiseY; // 添加平滑噪声（Y轴）


                // 应用平移和缩放变换
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
        //船
        for (const auto& mesh : Solar.meshes) {
            glm::mat4 model = glm::mat4(1.0f);
            model = glm::translate(model, glm::vec3(3.0f, 3.0f, -17.1f));
            // 水平旋转：绕 Z 轴旋转 45 度
            //model = glm::rotate(model, glm::radians(45.0f), glm::vec3(0.0f, 0.0f, 1.0f));

            // 向屏幕外旋转：绕 X 轴旋转 45 度
           // model = glm::rotate(model, glm::radians(45.0f), glm::vec3(1.0f, 0.0f, 0.0f));
            // 水平旋转：绕 Y 轴旋转 45 度
            model = glm::rotate(model, glm::radians(60.0f), glm::vec3(0.0f, 1.0f, 0.0f));
            model = glm::scale(model, glm::vec3(2.5f));
            ourShader.use();
            ourShader.setMat4("view", camera.GetViewMatrix());
            ourShader.setMat4("projection", glm::perspective(glm::radians(camera.Zoom), (float)SCREEN_WIDTH / SCREEN_HEIGHT, 0.1f, 200.0f));

            mesh.Draw(ourShader, model);
        }




        glm::mat4 model = glm::mat4(1.0f);
        model = glm::translate(model, glm::vec3(0.0f, -1.0f, 0.0f));
        // 渲染地形
        
        streetShader.use();
        street.render(streetShader, camera, (float)SCREEN_WIDTH / SCREEN_HEIGHT);
        
        // 渲染天空盒
        skyboxShader.use();
        skybox.render(skyboxShader, camera, (float)SCREEN_WIDTH / SCREEN_HEIGHT);



        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glfwTerminate();
    return 0;
}

// 回调函数实现
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
    float yoffset = lastY - ypos; // Y 方向是反的
    lastX = xpos;
    lastY = ypos;

    camera.processMouseMovement(xoffset, yoffset);
}

void scroll_callback(GLFWwindow* window, double xoffset, double yoffset) {
    camera.processMouseScroll(yoffset);
}

void processInput(GLFWwindow* window) {
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
        glfwSetWindowShouldClose(window, true); // 标记窗口为关闭
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
