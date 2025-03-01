#ifndef CAMERA_H
#define CAMERA_H

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <string>

// 摄像机移动方向
enum Camera_Movement {
    FORWARD,
    BACKWARD,
    LEFT,
    RIGHT
};

class Camera {
public:
    glm::vec3 Position;
    glm::vec3 Front;
    glm::vec3 Up;
    glm::vec3 Right;
    glm::vec3 WorldUp;

    // 欧拉角
    float Yaw;
    float Pitch;

    // 摄像机选项
    float MovementSpeed;
    float MouseSensitivity;
    float Zoom;

    // 构造函数
    Camera(glm::vec3 position, glm::vec3 up = glm::vec3(0.0f, 1.0f, 0.0f), float yaw = -90.0f, float pitch = 0.0f);

    // 获取视图矩阵
    glm::mat4 GetViewMatrix() const;

    // 获取投影矩阵
    glm::mat4 getProjectionMatrix(float aspectRatio) const;

    // 摄像机输入处理
    void processKeyboard(Camera_Movement direction, float deltaTime);
    void processMouseMovement(float xoffset, float yoffset, bool constrainPitch = true);
    void processMouseScroll(float yoffset);
    void updateCameraVectors(); // 更新摄像机向量
private:
    
};

#endif // CAMERA_H
