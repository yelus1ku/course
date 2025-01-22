import numpy as np
import matplotlib.pyplot as plt
import time

# 定义目标函数
def objective_func(A, b, x, reg_param):
    return np.sum([np.linalg.norm(A[j].dot(x) - b[j]) for j in range(A.shape[0])]) / 2 + reg_param * np.linalg.norm(x, ord=1)

# 软阈值函数
def soft_threshold(z, tau):
    return np.sign(z) * np.maximum(np.abs(z) - tau, 0)

# 邻近梯度法
def proximal_gradient(A, b, reg_param, step_size=0.0001, max_iters=5000, tolerance=1e-5):
    start_time = time.time()
    num_samples, dim1, dim2 = A.shape
    x = np.zeros(dim2)
    history = []

    for _ in range(max_iters):
        prev_x = x.copy()

        # 梯度计算
        grad = np.sum([A[j].T.dot(A[j].dot(x) - b[j]) for j in range(num_samples)], axis=0)

        # 更新 x
        x = soft_threshold(x - step_size * grad, reg_param * step_size)
        history.append(x)

        # 检查收敛条件
        if np.linalg.norm(x - prev_x, ord=2) < tolerance:
            break

    elapsed_time = time.time() - start_time
    print(f'Proximal gradient finished in {elapsed_time:.2f}s')
    return x, history

# ADMM算法
def admm_solver(A, b, reg_param, penalty=1, max_iters=1000, tolerance=1e-5):
    start_time = time.time()
    num_samples, dim1, dim2 = A.shape
    x = np.zeros(dim2)
    z = np.zeros(dim2)
    u = np.zeros(dim2)
    history = []

    for _ in range(max_iters):
        prev_x = x.copy()

        # 更新 x
        x = np.linalg.inv(np.sum([A[j].T.dot(A[j]) for j in range(num_samples)], axis=0) + penalty * np.eye(dim2)).dot(
            np.sum([A[j].T.dot(b[j]) for j in range(num_samples)], axis=0) + penalty * z - u)

        # 更新 z
        z = soft_threshold(x + u / penalty, reg_param / penalty)

        # 更新 u
        u += penalty * (x - z)
        history.append(x)

        # 检查收敛条件
        if np.linalg.norm(x - prev_x, ord=2) < tolerance:
            break

    elapsed_time = time.time() - start_time
    print(f'ADMM completed in {elapsed_time:.2f}s')
    return x, history

# 次梯度法
def subgradient_descent(A, b, reg_param, step_size=0.0001, max_iters=5000, tolerance=1e-5):
    start_time = time.time()
    num_samples, dim1, dim2 = A.shape
    x = np.zeros(dim2)
    history = []

    for _ in range(max_iters):
        prev_x = x.copy()

        # 计算次梯度
        sub_grad = reg_param * np.sign(x) + np.sum([A[j].T.dot(A[j].dot(x) - b[j]) for j in range(num_samples)], axis=0)

        # 更新 x
        x = x - step_size * sub_grad
        history.append(x)

        # 检查收敛条件
        if np.linalg.norm(x - prev_x, ord=2) < tolerance:
            break

    elapsed_time = time.time() - start_time
    print(f'Subgradient descent completed in {elapsed_time:.2f}s')
    return x, history

# 主实验
np.random.seed(42)
samples = 10
rows = 5
cols = 200
lambda_reg = 0.01

A = np.array([np.random.normal(0, 1, (rows, cols)) for _ in range(samples)])
true_x = np.zeros(cols)
nonzero_indices = np.random.choice(cols, 5, replace=False)
true_x[nonzero_indices] = np.random.normal(0, 1, 5)
b = np.array([A[j].dot(true_x) + np.random.normal(0, 0.1, rows) for j in range(samples)])

# 运行优化算法
x_proximal, history_proximal = proximal_gradient(A, b, lambda_reg, max_iters=3000)
x_admm, history_admm = admm_solver(A, b, lambda_reg, max_iters=3000)
x_subgradient, history_subgradient = subgradient_descent(A, b, lambda_reg, max_iters=3000)

# 绘制所有算法的结果
plt.figure(figsize=(18, 10))

plt.subplot(3, 2, 1)
plt.plot([np.linalg.norm(iter_x - true_x, ord=2) for iter_x in history_proximal], label='Distance to true')
plt.title('Proximal Gradient: Distance to True')
plt.xlabel('Iterations')
plt.ylabel('Distance')
plt.grid()
plt.legend()

plt.subplot(3, 2, 2)
plt.plot([np.linalg.norm(iter_x - x_proximal, ord=2) for iter_x in history_proximal], label='Distance to Optimal')
plt.title('Proximal Gradient: Distance to Optimal')
plt.xlabel('Iterations')
plt.ylabel('Distance')
plt.grid()
plt.legend()

plt.subplot(3, 2, 3)
plt.plot([np.linalg.norm(iter_x - true_x, ord=2) for iter_x in history_admm], label='Distance to true')
plt.title('ADMM: Distance to True')
plt.xlabel('Iterations')
plt.ylabel('Distance')
plt.grid()
plt.legend()

plt.subplot(3, 2, 4)
plt.plot([np.linalg.norm(iter_x - x_admm, ord=2) for iter_x in history_admm], label='Distance to Optimal')
plt.title('ADMM: Distance to Optimal')
plt.xlabel('Iterations')
plt.ylabel('Distance')
plt.grid()
plt.legend()

plt.subplot(3, 2, 5)
plt.plot([np.linalg.norm(iter_x - true_x, ord=2) for iter_x in history_subgradient], label='Distance to true')
plt.title('Subgradient Descent: Distance to True')
plt.xlabel('Iterations')
plt.ylabel('Distance')
plt.grid()
plt.legend()

plt.subplot(3, 2, 6)
plt.plot([np.linalg.norm(iter_x - x_subgradient, ord=2) for iter_x in history_subgradient], label='Distance to Optimal')
plt.title('Subgradient Descent: Distance to Optimal')
plt.xlabel('Iterations')
plt.ylabel('Distance')
plt.grid()
plt.legend()

plt.tight_layout()
plt.show()
