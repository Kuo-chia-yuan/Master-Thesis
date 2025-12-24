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

# 定義常數
m = 64  # 觀測點數量
n = 64

# 生成探頭位置
sensor_positions = np.zeros((n, 3))
# 使用linspace確保生成8個點，包含起點和終點
x_coords = np.linspace(-0.035, 0.035, 8)
y_coords = np.linspace(-0.035, 0.035, 8)
index = 0
for y in y_coords:
    for x in x_coords:
        sensor_positions[index] = [x, y, 0]
        index += 1

# 生成觀測點位置 (4×4×4網格)
point_positions = np.zeros((m, 3))
x_coords = np.linspace(-0.0075, 0.0075, 4)
y_coords = np.linspace(-0.0075, 0.0075, 4)
z_coords = np.linspace(0.01, 0.025, 4)
index = 0
for z in z_coords:
    for y in y_coords:
        for x in x_coords:
            point_positions[index] = [x, y, z]
            index += 1

# 定義常數
f = 40e3  # 探頭頻率 (40 kHz)
w = 2 * np.pi * f  # 角頻率 (rad/s)
lambda_ = 343 / f  # 波長 (m)
k = 2 * np.pi / lambda_  # 波數 (wavenumber)
rho = 1.225  # 空氣密度 (kg/m^3)
c = 343  # 空氣中的聲速 (m/s)
Area = 0.0008  # 探頭面積 (m^2)
u = 3.086  # 質點速度幅值 (m/s)
R = 0.000865  # 懸浮球半徑 (m)
weight = 2.84e-6  # 球體重量 (kg)  # 保麗龍球重量
g = 9.81  # 重力加速度 (m/s^2)

# 計算常數項
const = 2 * np.pi * R**3
term1_coef = 1/(6*rho*c**2)
term2_coef = -rho/4
c1 = const * term1_coef  # c1 = 2πR³/(6ρc²)
c2 = const * term2_coef  # c2 = 2πR³ * (-ρ/4)

# 定義模型
class AcousticModel(nn.Module):
    def __init__(self):
        super(AcousticModel, self).__init__()
        # 共享特徵提取層
        self.shared_layers = nn.Sequential(
            nn.Linear(64, 128),  # 修改輸入維度為64個觀測點
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU()
        )
        
        # 振幅預測分支
        self.amplitude_layers = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),  # 修改輸出維度為64個探頭
            nn.ReLU()  # 振幅應該是正數
        )
        
        # 相位預測分支
        self.phase_layers = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),  # 修改輸出維度為64個探頭
            nn.Sigmoid()  # 將輸出限制在 [0, 1] 範圍內
        )
        
    def forward(self, x):
        shared_features = self.shared_layers(x)
        
        amplitude = self.amplitude_layers(shared_features)
        phase = self.phase_layers(shared_features) * 2 * np.pi  # 將相位映射到 [0, 2π]
        
        return amplitude, phase

# 載入已訓練的模型
model = AcousticModel()
model.load_state_dict(torch.load('acoustic_model.pth'))
model.eval()

# 載入 h 和 q 矩陣
h_matrices_np = np.load('h_matrices.npy')
q_matrices_np = np.load('q_matrices.npy')

# 重建 h 和 q 矩陣
h_matrices = []
q_matrices = []
for i in range(m):
    h_matrix = h_matrices_np[i].reshape(1, n)
    h_matrices.append(h_matrix)
    
    q_matrix = q_matrices_np[i].reshape(3, n)
    q_matrices.append(q_matrix)

# 設定能量範圍
E_min = -6e-5  # 最小能量值
E_max = -4e-5  # 最大能量值

# 預測函數
def predict_custom_energy(model, energy_input):
    with torch.no_grad():
        # 將輸入能量放大 10^5 倍，與訓練時一致
        energy_tensor = torch.tensor(energy_input * 1e5, dtype=torch.float32).unsqueeze(0)
        pred_amplitude, pred_phase = model(energy_tensor)
        return pred_amplitude.numpy()[0], pred_phase.numpy()[0]

# 計算預測的振幅和相位所產生的能量
def calculate_energy_from_predictions(amplitude, phase):
    # 轉換為複數振幅
    A = amplitude * np.exp(1j * phase)
    
    # 計算能量
    E = np.zeros(m)
    for i in range(m):
        h_term = np.matmul(h_matrices[i].conj().T, h_matrices[i])
        q_term = np.matmul(q_matrices[i].conj().T, q_matrices[i])
        E[i] = np.real(np.matmul(np.matmul(A.conj().T, (c1*h_term + c2*q_term)), A))
    
    return E

# 創建高斯分布能量場
def create_gaussian_energy_field(min_point_idx):
    min_point_pos = point_positions[min_point_idx]
    custom_energy = np.zeros(64)
    
    for idx in range(64):
        # 計算當前點的實際座標
        current_pos = point_positions[idx]
        
        # 計算到選定最小值點的距離
        dist = np.sqrt(np.sum((current_pos - min_point_pos)**2))
        
        # 使用高斯分布創建平滑的能量分布
        sigma = 0.004  # 調整分布的寬度
        energy = E_min + (E_max - E_min) * (1 - np.exp(-dist**2/(2*sigma**2)))
        
        custom_energy[idx] = energy
    
    return custom_energy

# 創建3D能量場插值函數 (用於任意點的能量計算)
def create_energy_interpolator(energy_values):
    # 重塑能量值為4x4x4網格
    energy_grid = np.zeros((4, 4, 4))
    idx = 0
    for iz in range(4):
        for iy in range(4):
            for ix in range(4):
                energy_grid[iz, iy, ix] = energy_values[idx]
                idx += 1
    
    # 創建插值函數 - 注意順序必須與position的順序匹配 (z, y, x)
    return RegularGridInterpolator((z_coords, y_coords, x_coords), energy_grid, 
                                  bounds_error=False, fill_value=None)

# 計算力場 (F = -∇E)
def calculate_force(position, energy_interpolator):
    """
    計算給定位置的力
    
    參數:
    position - 位置 (z, y, x)
    energy_interpolator - 能量插值函數
    
    返回:
    force - 力向量 (Fz, Fy, Fx)
    """
    # 小位移用於計算梯度
    delta = 1e-4
    
    try:
        # 獲取當前位置的能量
        energy = energy_interpolator(position)
        
        # 計算x方向梯度
        pos_dx = position.copy()
        pos_dx[2] += delta
        energy_dx = energy_interpolator(pos_dx)
        fx = -(energy_dx - energy) / delta
        
        # 計算y方向梯度
        pos_dy = position.copy()
        pos_dy[1] += delta
        energy_dy = energy_interpolator(pos_dy)
        fy = -(energy_dy - energy) / delta
        
        # 計算z方向梯度
        pos_dz = position.copy()
        pos_dz[0] += delta
        energy_dz = energy_interpolator(pos_dz)
        fz = -(energy_dz - energy) / delta
        
        # 確保返回一維向量
        force = np.array([fz, fy, fx])
        if force.ndim > 1:
            force = force.flatten()
            
        return force
        
    except Exception as e:
        print(f"計算力時出錯: {e}")
        print(f"位置: {position}")
        # 返回零向量作為後備
        return np.zeros(3)

# 模擬保麗龍球運動 - 修改為支援不同時間段的不同能量場
def simulate_ball_motion_with_transition(amplitude1, phase1, amplitude2, phase2, 
                                        start_position, time_steps, dt, transition_time):
    """
    模擬保麗龍球在聲場中的運動，並在指定時間切換聲場
    
    參數:
    amplitude1, phase1 - 第一個懸浮點的探頭振幅和相位
    amplitude2, phase2 - 第二個懸浮點的探頭振幅和相位
    start_position - 起始位置 (z, y, x)
    time_steps - 時間步數
    dt - 時間步長 (秒)
    transition_time - 切換聲場的時間點 (秒)
    
    返回:
    positions - 保麗龍球位置軌跡
    velocities - 保麗龍球速度軌跡
    accelerations - 保麗龍球加速度軌跡
    energies - 系統能量記錄
    """
    # 計算兩個能量場
    energy1 = calculate_energy_from_predictions(amplitude1, phase1)
    energy2 = calculate_energy_from_predictions(amplitude2, phase2)
    
    # 創建兩個能量插值函數
    energy_interpolator1 = create_energy_interpolator(energy1)
    energy_interpolator2 = create_energy_interpolator(energy2)
    
    # 初始化位置、速度和加速度
    position = np.array(start_position)  # (z, y, x)
    velocity = np.zeros(3)  # (vz, vy, vx)
    
    # 儲存軌跡
    positions = [position.copy()]
    velocities = [velocity.copy()]
    accelerations = []
    energies = []
    
    # 計算切換時間步
    transition_step = int(transition_time / dt)
    
    # 模擬運動
    for t in range(time_steps):
        # 確定當前使用的能量插值器
        current_time = t * dt
        if current_time < transition_time:
            current_interpolator = energy_interpolator1
        else:
            current_interpolator = energy_interpolator2
        
        # 確保位置在有效範圍內
        if (position[0] < min(z_coords) or position[0] > max(z_coords) or
            position[1] < min(y_coords) or position[1] > max(y_coords) or
            position[2] < min(x_coords) or position[2] > max(x_coords)):
            # 如果超出範圍，停止模擬
            print(f"球體在時間步 {t} 超出有效範圍，停止模擬")
            break
        
        try:
            # 計算聲場力
            acoustic_force = calculate_force(position, current_interpolator)
            
            # 加上重力 (只在z方向)
            gravity_force = np.array([-weight * g, 0, 0])  # (Fz, Fy, Fx)
            
            # 計算阻尼力 (根據斯托克斯定律調整)
            air_viscosity = 1.81e-5  # 空氣動力黏度(kg/m·s)
            damping_coefficient = 6 * np.pi * air_viscosity * R  # 斯托克斯阻尼
            
            # 考慮高頻效應，增加阻尼係數
            damping_coefficient *= 10  # 調整係數，考慮高頻效應
            
            damping_force = -damping_coefficient * velocity
            
            # 合力
            total_force = acoustic_force + gravity_force + damping_force
            
            # 計算加速度 (F = ma)
            acceleration = total_force / weight
            
            # 確保加速度是一維向量
            if acceleration.ndim > 1:
                acceleration = acceleration.flatten()
                
            accelerations.append(acceleration.copy())
            
            # 更新速度 (v = v + a*dt)
            velocity = velocity + acceleration * dt
            velocities.append(velocity.copy())
            
            # 更新位置 (p = p + v*dt)
            position = position + velocity * dt
            positions.append(position.copy())
            
            # 記錄系統能量
            current_energy = current_interpolator(position)[0]
            kinetic_energy = 0.5 * weight * np.sum(velocity**2)
            potential_energy = weight * g * position[0]  # 重力勢能
            energies.append([current_energy, kinetic_energy, potential_energy])
            
        except Exception as e:
            print(f"錯誤發生在時間步 {t}:")
            print(f"位置: {position}")
            print(f"錯誤: {e}")
            break
    
    return np.array(positions), np.array(velocities), np.array(accelerations), np.array(energies)

# 主程式部分
# 選擇兩個懸浮點
first_point_idx = 25  # 第一個懸浮點
second_point_idx = 41  # 第二個懸浮點 (選擇不同的點)

# 為兩個懸浮點創建能量場
energy_field1 = create_gaussian_energy_field(first_point_idx)
energy_field2 = create_gaussian_energy_field(second_point_idx)

# 預測兩個懸浮點的振幅和相位
pred_amplitude1, pred_phase1 = predict_custom_energy(model, energy_field1)
pred_amplitude2, pred_phase2 = predict_custom_energy(model, energy_field2)

# 獲取目標位置
target_position1 = point_positions[first_point_idx]
target_position2 = point_positions[second_point_idx]

# 轉換為(z,y,x)格式
target_position1_zyx = np.array([target_position1[2], target_position1[1], target_position1[0]])
target_position2_zyx = np.array([target_position2[2], target_position2[1], target_position2[0]])

# 設置釋放位置(在第一個目標位置上方)
release = 0.002  # 釋放高度(比目標高2mm)
start_position = np.array([
    target_position1_zyx[0],  
    target_position1_zyx[1]-release,
    target_position1_zyx[2]
])

print(f"第一個懸浮點: {target_position1_zyx}")
print(f"第二個懸浮點: {target_position2_zyx}")
print(f"釋放起始位置: {start_position}")

# 模擬參數
time_steps = 60000  # 總時間步數
dt = 0.0001  # 時間步長
transition_time = 0.205  # 在?秒時切換到第二個懸浮點

# 模擬運動
positions, velocities, accelerations, energies = simulate_ball_motion_with_transition(
    pred_amplitude1, pred_phase1, 
    pred_amplitude2, pred_phase2,
    start_position, time_steps, dt, transition_time
)

print(f"模擬完成，軌跡點數: {len(positions)}")

# 創建3D軌跡動畫
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# 提取x, y, z坐標並轉換為毫米
x_mm = point_positions[:, 0] * 1000  # 轉換為毫米
y_mm = point_positions[:, 1] * 1000  # 轉換為毫米
z_mm = point_positions[:, 2] * 1000  # 轉換為毫米

# 繪製觀測點網格
ax.scatter(x_mm, y_mm, z_mm, color='gray', alpha=0.3, s=30)

# 標記兩個目標位置
target1_x = target_position1[0] * 1000  # 轉換為毫米
target1_y = target_position1[1] * 1000  # 轉換為毫米
target1_z = target_position1[2] * 1000  # 轉換為毫米
ax.scatter([target1_x], [target1_y], [target1_z], color='blue', s=100, marker='*', label='Target 1')

target2_x = target_position2[0] * 1000  # 轉換為毫米
target2_y = target_position2[1] * 1000  # 轉換為毫米
target2_z = target_position2[2] * 1000  # 轉換為毫米
ax.scatter([target2_x], [target2_y], [target2_z], color='green', s=100, marker='*', label='Target 2')

# 初始化球體位置
ball_x = positions[0, 2] * 1000  # 轉換為毫米
ball_y = positions[0, 1] * 1000  # 轉換為毫米
ball_z = positions[0, 0] * 1000  # 轉換為毫米
ball = ax.scatter([ball_x], [ball_y], [ball_z], color='red', s=100, label='Ball')

# 初始化軌跡線
line, = ax.plot([], [], [], 'r-', linewidth=1, alpha=0.6)

# 設置軸標籤和範圍
ax.set_xlabel('X [mm]')
ax.set_ylabel('Y [mm]')
ax.set_zlabel('Z [mm]')
ax.set_xlim([min(x_mm)-1, max(x_mm)+1])
ax.set_ylim([min(y_mm)-1, max(y_mm)+1])
ax.set_zlim([min(z_mm)-1, max(z_mm)+1])

# 設置標題
title = ax.set_title('Ball Trajectory - Step 0')
ax.legend()

# 創建軌跡存儲
line_x = []
line_y = []
line_z = []

# 使用跳幀來減少記憶體使用
frame_skip = 15  # 每5幀渲染一幀

# 更新函數
def update_trajectory(frame):
    global line_x, line_y, line_z
    
    # 使用跳幀後的實際幀索引
    actual_frame = frame * frame_skip
    
    if actual_frame >= len(positions):
        actual_frame = len(positions) - 1
    
    # 更新球體位置
    ball_x = positions[actual_frame, 2] * 1000  # 轉換為毫米
    ball_y = positions[actual_frame, 1] * 1000  # 轉換為毫米
    ball_z = positions[actual_frame, 0] * 1000  # 轉換為毫米
    ball._offsets3d = ([ball_x], [ball_y], [ball_z])
    
    # 更新軌跡線
    line_x.append(ball_x)
    line_y.append(ball_y)
    line_z.append(ball_z)
    
    # 只保留最近的100個點，避免軌跡過長
    if len(line_x) > 100:
        line_x = line_x[-100:]
        line_y = line_y[-100:]
        line_z = line_z[-100:]
    
    line.set_data(line_x, line_y)
    line.set_3d_properties(line_z)
    
    # 更新標題
    current_time = actual_frame * dt
    title.set_text(f'Ball Trajectory - Time: {current_time:.2f}s')
    
    return [ball, line, title]

# 計算總幀數並應用跳幀
total_frames = len(positions) // frame_skip

# 創建動畫
ani = FuncAnimation(fig, update_trajectory, frames=total_frames, interval=10, blit=True)

# 保存動畫
print("開始保存動畫...")
ani.save('ball_trajectory_transition3.gif', writer='pillow', fps=50, dpi=120)
print("動畫保存完成！")
plt.close()

# 釋放記憶體
#del ani
gc.collect()

# 創建位置隨時間變化的曲線圖
print("生成位置-時間曲線圖...")
fig, axes = plt.subplots(3, 1, figsize=(10, 12))

# 時間軸
time = np.arange(len(positions)) * dt  # 時間軸

# X位置隨時間變化
axes[0].plot(time, positions[:, 2] * 1000, 'b-')  # 轉換為毫米
axes[0].axhline(y=target_position1[0] * 1000, color='g', linestyle='--', label='Target 1 X')
axes[0].axhline(y=target_position2[0] * 1000, color='b', linestyle='--', label='Target 2 X')
axes[0].axvline(x=transition_time, color='r', linestyle=':', label='Transition')
axes[0].set_title('X Position vs Time')
axes[0].set_xlabel('Time [s]')
axes[0].set_ylabel('X Position [mm]')
axes[0].grid(True)
axes[0].legend()

# Y位置隨時間變化
axes[1].plot(time, positions[:, 1] * 1000, 'g-')  # 轉換為毫米
axes[1].axhline(y=target_position1[1] * 1000, color='g', linestyle='--', label='Target 1 Y')
axes[1].axhline(y=target_position2[1] * 1000, color='b', linestyle='--', label='Target 2 Y')
axes[1].axvline(x=transition_time, color='r', linestyle=':', label='Transition')
axes[1].set_title('Y Position vs Time')
axes[1].set_xlabel('Time [s]')
axes[1].set_ylabel('Y Position [mm]')
axes[1].grid(True)
axes[1].legend()

# Z位置隨時間變化
axes[2].plot(time, positions[:, 0] * 1000, 'r-')  # 轉換為毫米
axes[2].axhline(y=target_position1[2] * 1000, color='g', linestyle='--', label='Target 1 Z')
axes[2].axhline(y=target_position2[2] * 1000, color='b', linestyle='--', label='Target 2 Z')
axes[2].axvline(x=transition_time, color='r', linestyle=':', label='Transition')
axes[2].set_title('Z Position vs Time')
axes[2].set_xlabel('Time [s]')
axes[2].set_ylabel('Z Position [mm]')
axes[2].grid(True)
axes[2].legend()

plt.tight_layout()
plt.savefig('position_vs_time_transition3.png', dpi=150)
plt.close()

print("模擬完成！")
print("生成的文件:")
print("1. ball_trajectory_transition.gif - 球體運動軌跡動畫")
print("2. position_vs_time_transition.png - 球體位置隨時間變化曲線圖")