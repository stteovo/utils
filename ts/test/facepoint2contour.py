import numpy as np
import matplotlib.pyplot as plt

# B样条基函数的迭代计算
def b_spline_basis(t, i, k, knots):
    """计算B样条基函数"""
    if k == 0:
        return 1.0 if knots[i] <= t < knots[i + 1] else 0.0
    coef1 = (t - knots[i]) / (knots[i + k] - knots[i]) if knots[i + k] > knots[i] else 0
    coef2 = (knots[i + k + 1] - t) / (knots[i + k + 1] - knots[i + 1]) if knots[i + k + 1] > knots[i + 1] else 0
    return coef1 * b_spline_basis(t, i, k - 1, knots) + coef2 * b_spline_basis(t, i + 1, k - 1, knots)

# 生成 B 样条曲线
def generate_b_spline(control_points, degree, num_points=100):
    n = len(control_points) - 1
    knots = np.concatenate((
        np.zeros(degree + 1),                 # 前 degree+1 个节点为 0
        np.linspace(0, 1, n - degree + 1),   # 中间均匀分布的节点
        np.ones(degree + 1)                  # 后 degree+1 个节点为 1
    ))
    t_values = np.linspace(0, 1, num_points)
    curve = []

    for t in t_values:
        basis_values = np.array([b_spline_basis(t, i, degree, knots) for i in range(len(control_points))])
        point = np.sum(basis_values[:, None] * control_points, axis=0)
        curve.append(point)

    return np.array(curve)

# 快速检查凹性
def check_concavity(curve):
    x, y = curve[:, 0], curve[:, 1]
    dx = np.gradient(x)
    dy = np.gradient(y)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)

    # 曲率公式
    curvature = dx * ddy - dy * ddx
    return np.all(curvature <= 0)

# 调整控制点
def adjust_control_points(control_points, degree, max_iterations=100, learning_rate=1.0):
    n_points = len(control_points)
    for iteration in range(max_iterations):
        curve = generate_b_spline(control_points, degree)
        if check_concavity(curve):
            print(f"满足凹性约束，迭代次数: {iteration}")
            break

        # 更新控制点
        for i in range(1, n_points - 1):  # 避免调整首尾控制点
            control_points[i][1] -= learning_rate * (i / n_points)

    return control_points

# 新的点数据
points = np.array([
    [887, 1719], [891, 1724], [897, 1730], [899, 1735], [902, 1741],
    [904, 1747], [904, 1752], [906, 1758], [907, 1763], [909, 1769],
    [910, 1775], [911, 1780], [913, 1786], [914, 1791], [962, 1797],
    [994, 1865], [1033, 1927], [1079, 1984], [1131, 2034], [1190, 2077],
    [1253, 2116], [1318, 2149], [1390, 2171], [1466, 2179], [1542, 2171],
    [1613, 2147], [1676, 2110], [1735, 2067], [1792, 2021], [1844, 1969],
    [1889, 1911], [1926, 1847], [1955, 1826], [1961, 1804], [1968, 1783],
    [1974, 1762], [1979, 1740], [1984, 1719], [1991, 1698], [1995, 1676],
    [1997, 1655], [1997, 1633], [1999, 1612], [1997, 1591], [1990, 1569],
    [1991, 1548]
])

# 初始控制点
control_points = points.astype(float)
degree = 3

# 调整控制点
adjusted_control_points = adjust_control_points(control_points, degree)

# 生成最终曲线
curve = generate_b_spline(adjusted_control_points, degree)

# 绘制结果
plt.figure(figsize=(12, 6))
plt.plot(points[:, 0], points[:, 1], 'ro', label='原始点')
plt.plot(curve[:, 0], curve[:, 1], 'b-', label='优化后B样条')
plt.title("优化后的B样条曲线（凹性约束）")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.gca().invert_yaxis()  # 翻转 Y 轴
plt.show()
