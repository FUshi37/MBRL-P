import numpy as np
import matplotlib.pyplot as plt

# 从文件读取数据
def read_velocity_data(filename):
    y_velocities = []
    with open(filename, "r") as f:
        for line in f:
            line = line.strip()  # 去除换行符和空格
            if line:  # 跳过空行
                try:
                    y_velocities.append(float(line))
                except ValueError:
                    print(f"Warning: 无法解析行内容: {line}")
    return np.array(y_velocities)

# 绘制曲线
def plot_velocity():
    data = read_velocity_data("clip_lin_vel_y.txt")
    plt.figure(figsize=(10, 5))
    plt.plot(data, 'b-', label='Y方向速度 (m/s)')
    plt.ylim(-0.1, 0.2)  # <--- 关键行
    plt.xlabel("步数")
    plt.ylabel("速度 (m/s)")
    plt.title("Y方向速度跟踪曲线")
    plt.grid(True)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    plot_velocity()