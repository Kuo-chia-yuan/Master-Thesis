import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
from scipy.interpolate import RegularGridInterpolator
import matplotlib.cm as cm
import gc  # 添加垃圾回收模組
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 配置參數
# ============================================================
class SimulationConfig:
    """模擬配置類"""
    # 網格參數
    m = 27  # 觀測點數量 (3×3×3)
    n = 25  # 發射器數量（5×5陣列）
    
    # 物理常數
    f = 40e3  # 探頭頻率 (40 kHz)
    w = 2 * np.pi * f  # 角頻率 (rad/s)
    lambda_ = 343 / f  # 波長 (m)
    k = 2 * np.pi / lambda_  # 波數
    rho = 1.225  # 空氣密度 (kg/m^3)
    c = 343  # 聲速 (m/s)
    Area = 0.0002  # 探頭面積 (m^2)
    u = 3.086  # 質點速度幅值 (m/s)
    R = 0.000865  # 懸浮球半徑 (m)
    weight = 2e-7  # 球體重量 (kg)
    g = 9.81  # 重力加速度 (m/s^2)
    
    # 模擬參數
    time_steps = 60000  # 總時間步數
    dt = 0.0001  # 時間步長 (s)
    frame_skip = 15  # 動畫跳幀數
    
    # 阻尼參數
    air_viscosity = 1.81e-5  # 空氣動力黏度(kg/m·s)
    damping_multiplier = 30  # 阻尼倍增係數（增加以提高穩定性）
    
    # 邊界參數
    boundary_margin = 0.001  # 邊界安全距離 (1mm)
    max_velocity = 0.5  # 最大速度限制 (m/s)

config = SimulationConfig()

# ============================================================
# 網格設置
# ============================================================
# 生成探頭位置（5×5網格）
sensor_positions = np.zeros((config.n, 3))
spacing = 0.018  # 間距 1.8cm
start_x = -(spacing * 4)/2
start_y = -(spacing * 4)/2
idx = 0
for i in range(5):  # y方向
    for j in range(5):  # x方向
        x = start_x + j*spacing
        y = start_y + i*spacing
        sensor_positions[idx] = [x, y, 0]
        idx += 1

# 生成觀測點位置 (3×3×3網格)
point_positions = np.zeros((config.m, 3))
x_coords = np.array([-0.005, 0.0, 0.005])  # -5, 0, 5 mm
y_coords = np.array([-0.005, 0.0, 0.005])  # -5, 0, 5 mm
z_coords = np.array([0.01, 0.015, 0.02])   # 10, 15, 20 mm

idx = 0
for z in z_coords:
    for y in y_coords:
        for x in x_coords:
            point_positions[idx] = [x, y, z]
            idx += 1

print("=" * 60)
print("聲懸浮模擬系統初始化")
print("=" * 60)
print(f"觀測點網格: 3×3×3 = {config.m} 個點")
print(f"X 坐標: {x_coords * 1000} mm")
print(f"Y 坐標: {y_coords * 1000} mm")
print(f"Z 坐標: {z_coords * 1000} mm")
print(f"發射器數量: {config.n} (5×5 陣列)")
print(f"模擬時長: {config.time_steps * config.dt:.2f} 秒")
print("=" * 60 + "\n")

# ============================================================
# 計算常數項
# ============================================================
const = 2 * np.pi * config.R**3
term1_coef = 1/(6*config.rho*config.c**2)
term2_coef = -config.rho/4
c1 = const * term1_coef
c2 = const * term2_coef

# ============================================================
# 矩陣計算函數（優化版）
# ============================================================
def calculate_h_matrices():
    """計算h矩陣（向量化優化）"""
    jpck = 1j * config.rho * config.c * config.k
    constant_term = config.Area * jpck * config.u / (2*np.pi)
    
    h_matrices = []
    print("正在計算 h 矩陣...")
    
    for i in range(config.m):
        # 向量化計算距離
        diff = point_positions[i] - sensor_positions
        r = np.linalg.norm(diff, axis=1)
        
        # 計算h矩陣
        h_matrix = constant_term * np.exp(-1j*config.k*r) / r
        h_matrices.append(h_matrix.reshape(1, -1))
        
        if (i + 1) % 9 == 0:
            print(f"  進度: {i+1}/{config.m}")
    
    print("h 矩陣計算完成！\n")
    return h_matrices

def calculate_q_matrices():
    """計算q矩陣（優化版）"""
    jpck = 1j * config.rho * config.c * config.k
    constant_term = config.Area * jpck * config.u / (2*np.pi)
    delta = 1e-6
    
    q_matrices = []
    print("正在計算 q 矩陣...")
    
    for i in range(config.m):
        q_matrix = np.zeros((3, config.n), dtype=complex)
        
        for j in range(config.n):
            r0 = point_positions[i] - sensor_positions[j]
            
            # x方向梯度
            dx = np.array([delta, 0, 0])
            r_plus = r0 + dx
            r_minus = r0 - dx
            h_plus_x = constant_term * np.exp(-1j*config.k*np.linalg.norm(r_plus)) / np.linalg.norm(r_plus)
            h_minus_x = constant_term * np.exp(-1j*config.k*np.linalg.norm(r_minus)) / np.linalg.norm(r_minus)
            q_matrix[0,j] = -1/(1j*config.w*config.rho) * ((h_plus_x - h_minus_x)/(2*delta))
            
            # y方向梯度
            dy = np.array([0, delta, 0])
            r_plus = r0 + dy
            r_minus = r0 - dy
            h_plus_y = constant_term * np.exp(-1j*config.k*np.linalg.norm(r_plus)) / np.linalg.norm(r_plus)
            h_minus_y = constant_term * np.exp(-1j*config.k*np.linalg.norm(r_minus)) / np.linalg.norm(r_minus)
            q_matrix[1,j] = -1/(1j*config.w*config.rho) * ((h_plus_y - h_minus_y)/(2*delta))
            
            # z方向梯度
            dz = np.array([0, 0, delta])
            r_plus = r0 + dz
            r_minus = r0 - dz
            h_plus_z = constant_term * np.exp(-1j*config.k*np.linalg.norm(r_plus)) / np.linalg.norm(r_plus)
            h_minus_z = constant_term * np.exp(-1j*config.k*np.linalg.norm(r_minus)) / np.linalg.norm(r_minus)
            q_matrix[2,j] = -1/(1j*config.w*config.rho) * ((h_plus_z - h_minus_z)/(2*delta))
        
        q_matrices.append(q_matrix)
        
        if (i + 1) % 9 == 0:
            print(f"  進度: {i+1}/{config.m}")
    
    print("q 矩陣計算完成！\n")
    return q_matrices

# 計算h和q矩陣
h_matrices = calculate_h_matrices()
q_matrices = calculate_q_matrices()

# ============================================================
# 振幅和相位設置
# ============================================================
custom_amplitudes = np.array([
    5, 10, 15, 10, 5, 
    5, 15, 15, 5, 15, 
    10, 5, 15, 15, 15, 
    15, 15, 15, 5, 0, 
    10, 10, 15, 15, 0
])

custom_phases = np.array([
    0, 0.2, 0.4, 0.2, 0.2, 
    0.4, 0, -0.2, 0, 0, 
    0, 0.2, 0, 0, 0.6, 
    0, -0.2, 0, -0.2, 0.4, 
    0, 0, 0, 0.2, -0.2
]) * np.pi

# ============================================================
# 能量和力場計算函數
# ============================================================
def calculate_energy_from_amplitudes_phases(amplitudes, phases):
    """計算能量場"""
    A = amplitudes * np.exp(1j * phases)
    E = np.zeros(config.m)
    
    for i in range(config.m):
        h_term = np.matmul(h_matrices[i].conj().T, h_matrices[i])
        q_term = np.matmul(q_matrices[i].conj().T, q_matrices[i])
        E[i] = np.real(np.matmul(np.matmul(A.conj().T, (c1*h_term + c2*q_term)), A))
    
    return E

def create_energy_interpolator(energy_values):
    """創建能量插值函數（修正版本相容性問題）"""
    energy_grid = energy_values.reshape(3, 3, 3)
    
    # 使用 bounds_error=False 和 fill_value=None 來處理邊界
    # 這在所有 scipy 版本中都有效
    return RegularGridInterpolator(
        (z_coords, y_coords, x_coords), 
        energy_grid, 
        bounds_error=False,
        fill_value=None  # 使用最近鄰外插
    )

def calculate_force(position, energy_interpolator):
    """計算力場（優化版，添加邊界處理）"""
    delta = 1e-5
    
    try:
        # 確保位置在有效範圍內（添加小的容差）
        eps = 1e-7
        position = np.clip(position, 
                          [min(z_coords)+eps, min(y_coords)+eps, min(x_coords)+eps],
                          [max(z_coords)-eps, max(y_coords)-eps, max(x_coords)-eps])
        
        energy = energy_interpolator(position)
        
        # 如果能量是數組，取第一個元素
        if hasattr(energy, '__len__'):
            energy = energy[0]
        
        # 計算梯度
        force = np.zeros(3)
        directions = [
            np.array([delta, 0, 0]),  # z方向
            np.array([0, delta, 0]),  # y方向
            np.array([0, 0, delta])   # x方向
        ]
        
        for i, delta_vec in enumerate(directions):
            pos_plus = position + delta_vec
            pos_minus = position - delta_vec
            
            # 確保擾動位置也在範圍內
            pos_plus = np.clip(pos_plus,
                              [min(z_coords)+eps, min(y_coords)+eps, min(x_coords)+eps],
                              [max(z_coords)-eps, max(y_coords)-eps, max(x_coords)-eps])
            pos_minus = np.clip(pos_minus,
                               [min(z_coords)+eps, min(y_coords)+eps, min(x_coords)+eps],
                               [max(z_coords)-eps, max(y_coords)-eps, max(x_coords)-eps])
            
            energy_plus = energy_interpolator(pos_plus)
            energy_minus = energy_interpolator(pos_minus)
            
            # 處理數組情況
            if hasattr(energy_plus, '__len__'):
                energy_plus = energy_plus[0]
            if hasattr(energy_minus, '__len__'):
                energy_minus = energy_minus[0]
            
            force[i] = -(energy_plus - energy_minus) / (2 * delta)
        
        return force
        
    except Exception as e:
        print(f"計算力時出錯: {e}, 位置: {position}")
        return np.zeros(3)

# ============================================================
# 模擬函數（優化版）
# ============================================================
def simulate_ball_motion(amplitudes, phases, start_position, time_steps, dt):
    """
    優化的球體運動模擬
    添加了：
    1. 邊界限制
    2. 速度限制
    3. 自適應阻尼
    4. 詳細的診斷輸出
    """
    # 計算能量場
    energy = calculate_energy_from_amplitudes_phases(amplitudes, phases)
    energy_interpolator = create_energy_interpolator(energy)
    
    # 初始化
    position = np.array(start_position, dtype=float)
    velocity = np.zeros(3, dtype=float)
    
    # 儲存軌跡
    positions = [position.copy()]
    velocities = [velocity.copy()]
    accelerations = []
    energies = []
    
    # 診斷計數器
    boundary_warnings = 0
    velocity_limits = 0
    
    print("開始模擬球體運動...")
    print(f"初始位置: ({position[2]*1000:.2f}, {position[1]*1000:.2f}, {position[0]*1000:.2f}) mm\n")
    
    for t in range(time_steps):
        # 邊界檢查和限制
        margin = config.boundary_margin
        at_boundary = False
        
        if position[0] < min(z_coords) + margin:
            position[0] = min(z_coords) + margin
            velocity[0] = max(0, velocity[0])  # 只允許向上運動
            at_boundary = True
        elif position[0] > max(z_coords) - margin:
            position[0] = max(z_coords) - margin
            velocity[0] = min(0, velocity[0])  # 只允許向下運動
            at_boundary = True
            
        if position[1] < min(y_coords) + margin:
            position[1] = min(y_coords) + margin
            velocity[1] = max(0, velocity[1])
            at_boundary = True
        elif position[1] > max(y_coords) - margin:
            position[1] = max(y_coords) - margin
            velocity[1] = min(0, velocity[1])
            at_boundary = True
            
        if position[2] < min(x_coords) + margin:
            position[2] = min(x_coords) + margin
            velocity[2] = max(0, velocity[2])
            at_boundary = True
        elif position[2] > max(x_coords) - margin:
            position[2] = max(x_coords) - margin
            velocity[2] = min(0, velocity[2])
            at_boundary = True
        
        if at_boundary:
            boundary_warnings += 1
        
        try:
            # 計算聲場力
            acoustic_force = calculate_force(position, energy_interpolator)
            
            # 重力
            gravity_force = np.array([-config.weight * config.g, 0, 0])
            
            # 自適應阻尼（邊界處增加阻尼）
            damping_multiplier = config.damping_multiplier
            if at_boundary:
                damping_multiplier *= 3  # 邊界處增加阻尼
            
            damping_coefficient = 6 * np.pi * config.air_viscosity * config.R * damping_multiplier
            damping_force = -damping_coefficient * velocity
            
            # 合力
            total_force = acoustic_force + gravity_force + damping_force
            
            # 計算加速度
            acceleration = total_force / config.weight
            accelerations.append(acceleration.copy())
            
            # 更新速度
            velocity = velocity + acceleration * dt
            
            # 速度限制
            velocity_magnitude = np.linalg.norm(velocity)
            if velocity_magnitude > config.max_velocity:
                velocity = velocity * (config.max_velocity / velocity_magnitude)
                velocity_limits += 1
            
            velocities.append(velocity.copy())
            
            # 更新位置
            position = position + velocity * dt
            positions.append(position.copy())
            
            # 記錄能量
            current_energy = energy_interpolator(position)
            if hasattr(current_energy, '__len__'):
                current_energy = current_energy[0]
            kinetic_energy = 0.5 * config.weight * np.sum(velocity**2)
            potential_energy = config.weight * config.g * position[0]
            energies.append([current_energy, kinetic_energy, potential_energy])
            
            # 定期輸出診斷信息
            if t % 10000 == 0 and t > 0:
                print(f"時間 {t*dt:.3f}s:")
                print(f"  位置: ({position[2]*1000:.2f}, {position[1]*1000:.2f}, {position[0]*1000:.2f}) mm")
                print(f"  速度: {velocity_magnitude*1000:.2f} mm/s")
                print(f"  聲場力: {np.linalg.norm(acoustic_force)*1e6:.2f} μN")
                print(f"  邊界警告: {boundary_warnings}, 速度限制: {velocity_limits}\n")
            
        except Exception as e:
            print(f"錯誤發生在時間步 {t} (時間 {t*dt:.3f}s):")
            print(f"位置: {position}")
            print(f"錯誤: {e}")
            break
    
    print(f"\n模擬完成！")
    print(f"總軌跡點數: {len(positions)}")
    print(f"邊界警告總數: {boundary_warnings}")
    print(f"速度限制總數: {velocity_limits}\n")
    
    return np.array(positions), np.array(velocities), np.array(accelerations), np.array(energies)

# ============================================================
# 主程式執行
# ============================================================
# 選擇目標點
target_point_idx = 13  # 中心點
target_position = point_positions[target_point_idx]
target_position_zyx = np.array([target_position[2]-0.0001, target_position[1]-0.0001, target_position[0]-0.0001])

# 設置起始位置（在目標位置）
start_position = target_position_zyx.copy()

print(f"目標懸浮點索引: {target_point_idx}")
print(f"目標懸浮點位置: {target_position_zyx * 1000} mm")
print(f"釋放起始位置: {start_position * 1000} mm\n")

# 執行模擬
positions, velocities, accelerations, energies = simulate_ball_motion(
    custom_amplitudes, custom_phases, start_position, config.time_steps, config.dt
)

# 計算能量場
energy_values = calculate_energy_from_amplitudes_phases(custom_amplitudes, custom_phases)

# 找出能量最小點
min_energy_idx = np.argmin(energy_values)
min_energy_pos = point_positions[min_energy_idx]
min_energy_pos_zyx = np.array([min_energy_pos[2], min_energy_pos[1], min_energy_pos[0]])

print("=" * 60)
print("能量場分析")
print("=" * 60)
print(f"能量最小點索引: {min_energy_idx}")
print(f"能量最小點位置: {min_energy_pos_zyx * 1000} mm")
print(f"能量最小值: {energy_values[min_energy_idx]:.6e} J")
print(f"能量最大值: {energy_values.max():.6e} J")
print(f"能量範圍: {(energy_values.max() - energy_values.min()):.6e} J\n")

# 顯示所有觀測點的能量
print("所有觀測點的能量分布:")
print("-" * 60)
for i in range(config.m):
    pos = point_positions[i]
    marker = " ← 最小" if i == min_energy_idx else " ← 目標" if i == target_point_idx else ""
    print(f"點 {i:2d}: ({pos[0]*1000:5.1f}, {pos[1]*1000:5.1f}, {pos[2]*1000:5.1f}) mm, "
          f"能量: {energy_values[i]:.6e} J{marker}")
print("=" * 60 + "\n")

# ============================================================
# 繪圖部分
# ============================================================
print("開始生成圖表...")

# 1. 3D軌跡動畫
print("1/6 正在創建3D軌跡動畫...")
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

x_mm = point_positions[:, 0] * 1000
y_mm = point_positions[:, 1] * 1000
z_mm = point_positions[:, 2] * 1000

ax.scatter(x_mm, y_mm, z_mm, color='gray', alpha=0.3, s=30, label='Grid Points')

sensor_x_mm = sensor_positions[:, 0] * 1000
sensor_y_mm = sensor_positions[:, 1] * 1000
sensor_z_mm = sensor_positions[:, 2] * 1000
ax.scatter(sensor_x_mm, sensor_y_mm, sensor_z_mm, color='blue', alpha=0.5, s=50, marker='s', label='Transducers')

target_x = target_position[0] * 1000
target_y = target_position[1] * 1000
target_z = target_position[2] * 1000
ax.scatter([target_x], [target_y], [target_z], color='green', s=150, marker='*', label='Target', zorder=10)

min_e_x = min_energy_pos[0] * 1000
min_e_y = min_energy_pos[1] * 1000
min_e_z = min_energy_pos[2] * 1000
ax.scatter([min_e_x], [min_e_y], [min_e_z], color='purple', s=150, marker='o', label='Min Energy', zorder=10)

ball_x = positions[0, 2] * 1000
ball_y = positions[0, 1] * 1000
ball_z = positions[0, 0] * 1000
ball = ax.scatter([ball_x], [ball_y], [ball_z], color='red', s=120, label='Ball', zorder=15)

line, = ax.plot([], [], [], 'r-', linewidth=1.5, alpha=0.7)

ax.set_xlabel('X [mm]', fontsize=11, fontweight='bold')
ax.set_ylabel('Y [mm]', fontsize=11, fontweight='bold')
ax.set_zlabel('Z [mm]', fontsize=11, fontweight='bold')
ax.set_xlim([min(x_mm)-2, max(x_mm)+2])
ax.set_ylim([min(y_mm)-2, max(y_mm)+2])
ax.set_zlim([min(z_mm)-2, max(z_mm)+2])

title = ax.set_title('Ball Trajectory in Acoustic Field', fontsize=13, fontweight='bold')
ax.legend(loc='upper left', fontsize=9)
ax.grid(True, alpha=0.3)

line_x = []
line_y = []
line_z = []

def update_trajectory(frame):
    global line_x, line_y, line_z
    
    actual_frame = frame * config.frame_skip
    if actual_frame >= len(positions):
        actual_frame = len(positions) - 1
    
    ball_x = positions[actual_frame, 2] * 1000
    ball_y = positions[actual_frame, 1] * 1000
    ball_z = positions[actual_frame, 0] * 1000
    ball._offsets3d = ([ball_x], [ball_y], [ball_z])
    
    line_x.append(ball_x)
    line_y.append(ball_y)
    line_z.append(ball_z)
    
    if len(line_x) > 150:
        line_x = line_x[-150:]
        line_y = line_y[-150:]
        line_z = line_z[-150:]
    
    line.set_data(line_x, line_y)
    line.set_3d_properties(line_z)
    
    current_time = actual_frame * config.dt
    title.set_text(f'Ball Trajectory - Time: {current_time:.3f}s')
    
    return [ball, line, title]

total_frames = len(positions) // config.frame_skip
ani = FuncAnimation(fig, update_trajectory, frames=total_frames, interval=10, blit=True)

ani.save('ball_trajectory_single_point.gif', writer='pillow', fps=50, dpi=100)
plt.close()
gc.collect()
print("   ✓ 動畫已保存")

# 2. 位置-時間曲線圖
print("2/6 正在生成位置-時間曲線圖...")
fig, axes = plt.subplots(3, 1, figsize=(12, 10))
time = np.arange(len(positions)) * config.dt

axes[0].plot(time, positions[:, 2] * 1000, 'b-', linewidth=1.5)
axes[0].axhline(y=target_position[0] * 1000, color='g', linestyle='--', linewidth=2, label='Target X')
axes[0].set_title('X Position vs Time', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Time [s]', fontsize=10)
axes[0].set_ylabel('X Position [mm]', fontsize=10)
axes[0].grid(True, alpha=0.3)
axes[0].legend()

axes[1].plot(time, positions[:, 1] * 1000, 'g-', linewidth=1.5)
axes[1].axhline(y=target_position[1] * 1000, color='g', linestyle='--', linewidth=2, label='Target Y')
axes[1].set_title('Y Position vs Time', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Time [s]', fontsize=10)
axes[1].set_ylabel('Y Position [mm]', fontsize=10)
axes[1].grid(True, alpha=0.3)
axes[1].legend()

axes[2].plot(time, positions[:, 0] * 1000, 'r-', linewidth=1.5)
axes[2].axhline(y=target_position[2] * 1000, color='g', linestyle='--', linewidth=2, label='Target Z')
axes[2].set_title('Z Position vs Time', fontsize=12, fontweight='bold')
axes[2].set_xlabel('Time [s]', fontsize=10)
axes[2].set_ylabel('Z Position [mm]', fontsize=10)
axes[2].grid(True, alpha=0.3)
axes[2].legend()

plt.tight_layout()
plt.savefig('position_vs_time_single_point.png', dpi=150, bbox_inches='tight')
plt.close()
print("   ✓ 位置-時間圖已保存")

# 3. 能量場分布
print("3/6 正在繪製能量場分布...")
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')

scatter = ax.scatter(x_mm, y_mm, z_mm, c=energy_values, cmap='jet', s=150, alpha=0.8, edgecolors='k', linewidth=0.5)
colorbar = plt.colorbar(scatter, pad=0.1, shrink=0.8)
colorbar.set_label('Energy [J]', fontsize=11, fontweight='bold')

ax.scatter([target_x], [target_y], [target_z], color='green', s=250, marker='*', 
           label='Target', edgecolors='k', linewidth=2, zorder=10)
ax.scatter([min_e_x], [min_e_y], [min_e_z], color='purple', s=250, marker='o', 
           label='Min Energy', edgecolors='k', linewidth=2, zorder=10)
ax.scatter(sensor_x_mm, sensor_y_mm, sensor_z_mm, color='blue', s=60, marker='s', 
           label='Transducers', alpha=0.6)

ax.set_xlabel('X [mm]', fontsize=11, fontweight='bold')
ax.set_ylabel('Y [mm]', fontsize=11, fontweight='bold')
ax.set_zlabel('Z [mm]', fontsize=11, fontweight='bold')
ax.set_title('Energy Field Distribution', fontsize=13, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.savefig('energy_field_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("   ✓ 能量場分布圖已保存")

# 4. 振幅和相位分布
print("4/6 正在繪製振幅和相位分布...")
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

amp_grid = custom_amplitudes.reshape(5, 5)
phase_grid = custom_phases.reshape(5, 5) / np.pi

im0 = axes[0].imshow(amp_grid, cmap='viridis', origin='lower', interpolation='bilinear')
axes[0].set_title('Amplitude Distribution', fontsize=13, fontweight='bold')
axes[0].set_xlabel('X Index', fontsize=11)
axes[0].set_ylabel('Y Index', fontsize=11)
for i in range(5):
    for j in range(5):
        text = axes[0].text(j, i, f'{amp_grid[i, j]:.1f}',
                           ha="center", va="center", color="white", fontsize=9, fontweight='bold')
fig.colorbar(im0, ax=axes[0], label='Amplitude')

im1 = axes[1].imshow(phase_grid, cmap='hsv', origin='lower', interpolation='bilinear')
axes[1].set_title('Phase Distribution (π units)', fontsize=13, fontweight='bold')
axes[1].set_xlabel('X Index', fontsize=11)
axes[1].set_ylabel('Y Index', fontsize=11)
for i in range(5):
    for j in range(5):
        text = axes[1].text(j, i, f'{phase_grid[i, j]:.2f}',
                           ha="center", va="center", color="white", fontsize=8, fontweight='bold')
fig.colorbar(im1, ax=axes[1], label='Phase (π units)')

plt.tight_layout()
plt.savefig('amplitude_phase_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("   ✓ 振幅和相位分布圖已保存")

# 5. 力的計算和繪圖
print("5/6 正在計算並繪製力場...")

def calculate_forces_at_trajectory(positions, velocities, amplitudes, phases):
    """計算軌跡上每個位置的合力分量"""
    energy = calculate_energy_from_amplitudes_phases(amplitudes, phases)
    energy_interpolator = create_energy_interpolator(energy)
    
    acoustic_forces = []
    gravity_forces = []
    damping_forces = []
    total_forces = []
    
    for i in range(len(positions)):
        position = positions[i]
        velocity = velocities[i] if i < len(velocities) else np.zeros(3)
        
        acoustic_force = calculate_force(position, energy_interpolator)
        acoustic_forces.append(acoustic_force.copy())
        
        gravity_force = np.array([-config.weight * config.g, 0, 0])
        gravity_forces.append(gravity_force.copy())
        
        damping_coefficient = 6 * np.pi * config.air_viscosity * config.R * config.damping_multiplier
        damping_force = -damping_coefficient * velocity
        damping_forces.append(damping_force.copy())
        
        total_force = acoustic_force + gravity_force + damping_force
        total_forces.append(total_force.copy())
    
    return (np.array(acoustic_forces), np.array(gravity_forces), 
            np.array(damping_forces), np.array(total_forces))

acoustic_forces, gravity_forces, damping_forces, total_forces = calculate_forces_at_trajectory(
    positions, velocities, custom_amplitudes, custom_phases
)

time = np.arange(len(positions)) * config.dt

fig, axes = plt.subplots(4, 1, figsize=(14, 16))

axes[0].plot(time, total_forces[:, 2] * 1e6, 'b-', linewidth=1.5, label='Total Fx')
axes[0].axhline(y=0, color='k', linestyle='--', alpha=0.3)
axes[0].set_title('Total Force in X Direction vs Time', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Time [s]', fontsize=11)
axes[0].set_ylabel('Force [μN]', fontsize=11)
axes[0].grid(True, alpha=0.3)
axes[0].legend(loc='upper right')

axes[1].plot(time, total_forces[:, 1] * 1e6, 'g-', linewidth=1.5, label='Total Fy')
axes[1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
axes[1].set_title('Total Force in Y Direction vs Time', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Time [s]', fontsize=11)
axes[1].set_ylabel('Force [μN]', fontsize=11)
axes[1].grid(True, alpha=0.3)
axes[1].legend(loc='upper right')

axes[2].plot(time, total_forces[:, 0] * 1e6, 'r-', linewidth=1.5, label='Total Fz')
axes[2].axhline(y=0, color='k', linestyle='--', alpha=0.3)
axes[2].set_title('Total Force in Z Direction vs Time', fontsize=13, fontweight='bold')
axes[2].set_xlabel('Time [s]', fontsize=11)
axes[2].set_ylabel('Force [μN]', fontsize=11)
axes[2].grid(True, alpha=0.3)
axes[2].legend(loc='upper right')

total_force_magnitude = np.linalg.norm(total_forces, axis=1)
acoustic_force_magnitude = np.linalg.norm(acoustic_forces, axis=1)
axes[3].plot(time, total_force_magnitude * 1e6, 'purple', linewidth=2, label='Total |F|')
axes[3].plot(time, acoustic_force_magnitude * 1e6, 'c--', linewidth=1.5, alpha=0.6, label='Acoustic |F|')
axes[3].set_title('Total Force Magnitude vs Time', fontsize=13, fontweight='bold')
axes[3].set_xlabel('Time [s]', fontsize=11)
axes[3].set_ylabel('Force Magnitude [μN]', fontsize=11)
axes[3].grid(True, alpha=0.3)
axes[3].legend(loc='upper right')

plt.tight_layout()
plt.savefig('force_vs_time_three_axes.png', dpi=150, bbox_inches='tight')
plt.close()
print("   ✓ 力-時間圖已保存")

# 6. 力分量比較圖
print("6/6 正在繪製力分量比較圖...")
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

axes[0, 0].plot(time, acoustic_forces[:, 2] * 1e6, 'b-', label='Fx', linewidth=1.5)
axes[0, 0].plot(time, acoustic_forces[:, 1] * 1e6, 'g-', label='Fy', linewidth=1.5)
axes[0, 0].plot(time, acoustic_forces[:, 0] * 1e6, 'r-', label='Fz', linewidth=1.5)
axes[0, 0].axhline(y=0, color='k', linestyle='--', alpha=0.3)
axes[0, 0].set_title('Acoustic Forces', fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel('Time [s]')
axes[0, 0].set_ylabel('Force [μN]')
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].legend()

axes[0, 1].plot(time, damping_forces[:, 2] * 1e6, 'b-', label='Fx', linewidth=1.5)
axes[0, 1].plot(time, damping_forces[:, 1] * 1e6, 'g-', label='Fy', linewidth=1.5)
axes[0, 1].plot(time, damping_forces[:, 0] * 1e6, 'r-', label='Fz', linewidth=1.5)
axes[0, 1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
axes[0, 1].set_title('Damping Forces', fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel('Time [s]')
axes[0, 1].set_ylabel('Force [μN]')
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].legend()

axes[1, 0].plot(time, gravity_forces[:, 0] * 1e6, 'r-', label='Fz (Gravity)', linewidth=2)
axes[1, 0].axhline(y=0, color='k', linestyle='--', alpha=0.3)
axes[1, 0].set_title('Gravity Force', fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel('Time [s]')
axes[1, 0].set_ylabel('Force [μN]')
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].legend()

axes[1, 1].plot(time, total_forces[:, 2] * 1e6, 'b-', label='Fx', linewidth=1.5)
axes[1, 1].plot(time, total_forces[:, 1] * 1e6, 'g-', label='Fy', linewidth=1.5)
axes[1, 1].plot(time, total_forces[:, 0] * 1e6, 'r-', label='Fz', linewidth=1.5)
axes[1, 1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
axes[1, 1].set_title('Total Forces', fontsize=12, fontweight='bold')
axes[1, 1].set_xlabel('Time [s]')
axes[1, 1].set_ylabel('Force [μN]')
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].legend()

plt.tight_layout()
plt.savefig('force_components_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("   ✓ 力分量比較圖已保存")

# ============================================================
# 最終統計和總結
# ============================================================
print("\n" + "=" * 60)
print("模擬統計摘要")
print("=" * 60)
print(f"聲場力最大值: {np.max(np.abs(acoustic_forces)) * 1e6:.3f} μN")
print(f"聲場力平均值: {np.mean(np.abs(acoustic_forces)) * 1e6:.3f} μN")
print(f"阻尼力最大值: {np.max(np.abs(damping_forces)) * 1e6:.3f} μN")
print(f"重力: {np.abs(config.weight * config.g) * 1e6:.3f} μN")
print(f"合力最大值: {np.max(total_force_magnitude) * 1e6:.3f} μN")
print(f"合力最小值: {np.min(total_force_magnitude) * 1e6:.3f} μN")
print(f"合力平均值: {np.mean(total_force_magnitude) * 1e6:.3f} μN")

# 位置統計
final_position = positions[-1]
position_error = np.linalg.norm(final_position - target_position_zyx) * 1000
print(f"\n最終位置偏差: {position_error:.3f} mm")
print(f"X方向標準差: {np.std(positions[:, 2]) * 1000:.3f} mm")
print(f"Y方向標準差: {np.std(positions[:, 1]) * 1000:.3f} mm")
print(f"Z方向標準差: {np.std(positions[:, 0]) * 1000:.3f} mm")

# 速度統計
velocity_magnitude = np.linalg.norm(velocities, axis=1)
print(f"\n最大速度: {np.max(velocity_magnitude) * 1000:.3f} mm/s")
print(f"平均速度: {np.mean(velocity_magnitude) * 1000:.3f} mm/s")
print(f"最終速度: {np.linalg.norm(velocities[-1]) * 1000:.3f} mm/s")

print("\n" + "=" * 60)
print("所有圖表已生成完成！")
print("=" * 60)
print("生成的文件:")
print("  1. ball_trajectory_single_point.gif - 球體運動軌跡動畫")
print("  2. position_vs_time_single_point.png - 位置-時間曲線圖")
print("  3. energy_field_distribution.png - 能量場分布")
print("  4. amplitude_phase_distribution.png - 振幅和相位分布")
print("  5. force_vs_time_three_axes.png - 力-時間三軸圖")
print("  6. force_components_comparison.png - 力分量比較圖")
print("=" * 60)
