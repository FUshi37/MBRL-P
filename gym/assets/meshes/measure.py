from stl import mesh
import numpy as np

# 读取 STL 文件
stl_mesh = mesh.Mesh.from_file('./base_link.STL')

# 获取所有顶点坐标
min_x, min_y, min_z = np.min(stl_mesh.vectors, axis=(0, 1))
max_x, max_y, max_z = np.max(stl_mesh.vectors, axis=(0, 1))

# 计算长、宽、高
length = max_x - min_x  # X 轴方向
width = max_y - min_y   # Y 轴方向
height = max_z - min_z  # Z 轴方向

print(f"Length (X): {length}")
print(f"Width  (Y): {width}")
print(f"Height (Z): {height}")
