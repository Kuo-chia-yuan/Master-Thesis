import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import scipy.io as sio
import os

# 設定隨機種子以便結果可重現
torch.manual_seed(42)
np.random.seed(42)

# 定義常數
n = 64  # 探頭數量
m = 64  # 觀測點數量
f = 40e3  # 探頭頻率 (40 kHz)
w = 2 * np.pi * f  # 角頻率 (rad/s)
lambda_ = 343 / f  # 波長 (m)
k = 2 * np.pi / lambda_  # 波數 (wavenumber)
rho = 1.225  # 空氣密度 (kg/m^3)
c = 343  # 空氣中的聲速 (m/s)
Area = 0.0008  # 探頭面積 (m^2)
u = 3.086  # 質點速度幅值 (m/s)
R = 0.000865  # 懸浮球半徑 (1 mm)
weight = 2.84e-6  # 球體重量 (kg)

# 計算常數項
const = 2 * np.pi * R**3
term1_coef = 1/(6*rho*c**2)
term2_coef = -rho/4
c1 = const * term1_coef  # c1 = 2πR³/(6ρc²)
c2 = const * term2_coef  # c2 = 2πR³ * (-ρ/4)

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

# 生成觀測點
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

# 計算 h 矩陣
def calculate_h_matrices():
    h_matrices = [np.zeros((1, n), dtype=np.complex128) for _ in range(m)]
    jpck = 1j * rho * c * k
    constant_term = Area * jpck * u / (2*np.pi)
    
    for i in range(m):
        for j in range(n):
            dx = point_positions[i, 0] - sensor_positions[j, 0]
            dy = point_positions[i, 1] - sensor_positions[j, 1]
            dz = point_positions[i, 2] - sensor_positions[j, 2]
            r = np.sqrt(dx**2 + dy**2 + dz**2)
            h_matrices[i][0, j] = constant_term * np.exp(-1j*k*r) / r
    
    return h_matrices

# 計算 q 矩陣
def calculate_q_matrices():
    q_matrices = [np.zeros((3, n), dtype=np.complex128) for _ in range(m)]
    jpck = 1j * rho * c * k
    delta = 1e-6
    
    for i in range(m):
        for j in range(n):
            r0 = point_positions[i] - sensor_positions[j]
            
            # x方向梯度
            dx = np.array([delta, 0, 0])
            r_plus = r0 + dx
            r_minus = r0 - dx
            h_plus_x = Area * jpck * u * np.exp(-1j*k*np.linalg.norm(r_plus)) / (2*np.pi*np.linalg.norm(r_plus))
            h_minus_x = Area * jpck * u * np.exp(-1j*k*np.linalg.norm(r_minus)) / (2*np.pi*np.linalg.norm(r_minus))
            q_matrices[i][0, j] = -1/(1j*w*rho) * ((h_plus_x - h_minus_x)/(2*delta))
            
            # y方向梯度
            dy = np.array([0, delta, 0])
            r_plus = r0 + dy
            r_minus = r0 - dy
            h_plus_y = Area * jpck * u * np.exp(-1j*k*np.linalg.norm(r_plus)) / (2*np.pi*np.linalg.norm(r_plus))
            h_minus_y = Area * jpck * u * np.exp(-1j*k*np.linalg.norm(r_minus)) / (2*np.pi*np.linalg.norm(r_minus))
            q_matrices[i][1, j] = -1/(1j*w*rho) * ((h_plus_y - h_minus_y)/(2*delta))
            
            # z方向梯度
            dz = np.array([0, 0, delta])
            r_plus = r0 + dz
            r_minus = r0 - dz
            h_plus_z = Area * jpck * u * np.exp(-1j*k*np.linalg.norm(r_plus)) / (2*np.pi*np.linalg.norm(r_plus))
            h_minus_z = Area * jpck * u * np.exp(-1j*k*np.linalg.norm(r_minus)) / (2*np.pi*np.linalg.norm(r_minus))
            q_matrices[i][2, j] = -1/(1j*w*rho) * ((h_plus_z - h_minus_z)/(2*delta))
    
    return q_matrices

# 計算能量函數
def calculate_energy(amplitudes, phases):
    # 轉換為複數振幅
    A = amplitudes * np.exp(1j * phases)
    
    # 計算能量
    E = np.zeros(m)
    for i in range(m):
        h_term = np.matmul(h_matrices[i].conj().T, h_matrices[i])
        q_term = np.matmul(q_matrices[i].conj().T, q_matrices[i])
        E[i] = np.real(np.matmul(np.matmul(A.conj().T, (c1*h_term + c2*q_term)), A))
    
    return E

# 計算 h 和 q 矩陣
h_matrices = calculate_h_matrices()
q_matrices = calculate_q_matrices()

# 保存 h 和 q 矩陣到檔案
h_matrices_np = np.array([h.flatten() for h in h_matrices])
q_matrices_np = np.array([q.flatten() for q in q_matrices])
np.save('h_matrices.npy', h_matrices_np)
np.save('q_matrices.npy', q_matrices_np)

# 自定義資料集
class AcousticDataset(Dataset):
    def __init__(self, energy, amplitude, phase):
        # 將能量值放大 10^6 倍
        self.energy = torch.tensor(energy * 1e5, dtype=torch.float32)
        self.amplitude = torch.tensor(amplitude, dtype=torch.float32)
        self.phase = torch.tensor(phase, dtype=torch.float32)
        
    def __len__(self):
        return len(self.energy)
    
    def __getitem__(self, idx):
        return self.energy[idx], self.amplitude[idx], self.phase[idx]

# 讀取資料
df = pd.read_csv('training_data_64x64.csv')

# 提取能量、振幅和相位
energy_data = df.filter(regex='energy_').values
amplitude_data = df.filter(regex='amplitude_').values
phase_data = df.filter(regex='phase_').values

# 資料分割 - 加入驗證集
# 先分出測試集
X_train_val, X_test, amp_train_val, amp_test, phase_train_val, phase_test = train_test_split(
    energy_data, amplitude_data, phase_data, test_size=0.2, random_state=42)

# 再從訓練集中分出驗證集
X_train, X_val, amp_train, amp_val, phase_train, phase_val = train_test_split(
    X_train_val, amp_train_val, phase_train_val, test_size=0.2, random_state=42)

# 建立資料集和資料載入器
train_dataset = AcousticDataset(X_train, amp_train, phase_train)
val_dataset = AcousticDataset(X_val, amp_val, phase_val)
test_dataset = AcousticDataset(X_test, amp_test, phase_test)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

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

# 自定義損失函數 - 只計算能量損失
class EnergyLoss(nn.Module):
    def __init__(self, h_matrices, q_matrices, c1, c2):
        super(EnergyLoss, self).__init__()
        self.h_matrices = h_matrices
        self.q_matrices = q_matrices
        self.c1 = c1
        self.c2 = c2
        
    def forward(self, pred_amplitude, pred_phase, true_energy):
        batch_size = pred_amplitude.shape[0]
        pred_energy = torch.zeros((batch_size, 64), device=pred_amplitude.device)
        
        # 轉換為複數振幅
        pred_complex = pred_amplitude * torch.exp(1j * pred_phase)
        
        # 計算每個樣本的能量
        for b in range(batch_size):
            for i in range(m):
                h = torch.tensor(h_matrices[i], dtype=torch.complex128).to(pred_amplitude.device)
                q = torch.tensor(q_matrices[i], dtype=torch.complex128).to(pred_amplitude.device)
                
                h_term = torch.matmul(h.conj().T, h)
                q_term = torch.matmul(q.conj().T, q)
                
                A = pred_complex[b].to(torch.complex128)
                E = torch.real(torch.matmul(torch.matmul(A.conj(), (self.c1*h_term + self.c2*q_term)), A))
                # 將計算出的能量也放大 10^6 倍以匹配輸入
                pred_energy[b, i] = E.real * 1e5
        
        # 計算能量損失
        energy_loss = nn.MSELoss()(pred_energy, true_energy)
        
        return energy_loss

# 初始化模型、損失函數和優化器
model = AcousticModel()
energy_loss_fn = EnergyLoss(h_matrices, q_matrices, c1, c2)
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

# 評估函數 - 可用於驗證集和測試集
def evaluate_model(model, data_loader, energy_loss_fn):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for energy, amplitude, phase in data_loader:
            # 前向傳播
            pred_amplitude, pred_phase = model(energy)
            
            # 計算損失 - 只計算能量損失
            loss = energy_loss_fn(pred_amplitude, pred_phase, energy)
            
            total_loss += loss.item()
    
    # 平均損失
    avg_loss = total_loss / len(data_loader)
    
    return avg_loss

# 訓練函數 - 加入驗證集評估
def train_model(model, train_loader, val_loader, energy_loss_fn, optimizer, scheduler, num_epochs=200):
    model.train()
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_model_state = None
    
    for epoch in range(num_epochs):
        # 訓練階段
        model.train()
        epoch_loss = 0
        
        for energy, amplitude, phase in train_loader:
            optimizer.zero_grad()
            
            # 前向傳播
            pred_amplitude, pred_phase = model(energy)
            
            # 計算損失 - 只計算能量損失
            loss = energy_loss_fn(pred_amplitude, pred_phase, energy)
            
            # 反向傳播
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        # 平均訓練損失
        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # 驗證階段
        model.eval()
        val_loss = evaluate_model(model, val_loader, energy_loss_fn)
        val_losses.append(val_loss)
        
        # 更新學習率
        scheduler.step(val_loss)
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
        
        # 打印進度
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.8f}, Val Loss: {val_loss:.8f}')
    
    # 載入最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f'Loaded best model with validation loss: {best_val_loss:.8f}')
    
    return train_losses, val_losses

# 訓練模型 - 使用驗證集
train_losses, val_losses = train_model(model, train_loader, val_loader, energy_loss_fn, optimizer, scheduler, num_epochs=200)

# 評估模型 - 在測試集上
test_loss = evaluate_model(model, test_loader, energy_loss_fn)
print(f'Test Loss: {test_loss:.8f}')

# 繪製訓練和驗證損失曲線
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('U Loss vs Epoch')
plt.xlabel('Epoch')
plt.ylabel('U Loss')
plt.legend()
plt.grid(True)
plt.savefig('training_validation_loss1.png')
plt.show()

# 保存模型
torch.save(model.state_dict(), 'acoustic_model_test.pth')

# 預測和可視化
def visualize_predictions(model, test_loader):
    model.eval()
    energy_samples = []
    true_amplitudes = []
    true_phases = []
    pred_amplitudes = []
    pred_phases = []
    
    with torch.no_grad():
        for energy, amplitude, phase in test_loader:
            # 前向傳播
            pred_amplitude, pred_phase = model(energy)
            
            # 儲存結果
            # 將能量縮小回原始尺度
            energy_samples.append(energy.numpy() / 1e5)
            true_amplitudes.append(amplitude.numpy())
            true_phases.append(phase.numpy())
            pred_amplitudes.append(pred_amplitude.numpy())
            pred_phases.append(pred_phase.numpy())
    
    # 轉換為 numpy 陣列
    energy_samples = np.vstack(energy_samples)
    true_amplitudes = np.vstack(true_amplitudes)
    true_phases = np.vstack(true_phases)
    pred_amplitudes = np.vstack(pred_amplitudes)
    pred_phases = np.vstack(pred_phases)
    
    # 繪製振幅比較
    # 繪製振幅比較 - 選擇前16個探頭作為示例
    plt.figure(figsize=(20, 15))
    for i in range(16):
        plt.subplot(4, 4, i+1)
        plt.scatter(true_amplitudes[:, i], pred_amplitudes[:, i], alpha=0.5)
        plt.plot([0, 10], [0, 10], 'r--')
        plt.title(f'Amplitude {i+1}')
        plt.xlabel('True')
        plt.ylabel('Predicted')
        plt.grid(True)
    plt.tight_layout()
    plt.savefig('amplitude_comparison.png')
    plt.show()
    
    # 繪製相位比較 - 選擇前16個探頭作為示例
    plt.figure(figsize=(20, 15))
    for i in range(16):
        plt.subplot(4, 4, i+1)
        plt.scatter(true_phases[:, i], pred_phases[:, i], alpha=0.5)
        plt.plot([0, 2*np.pi], [0, 2*np.pi], 'r--')
        plt.title(f'Phase {i+1}')
        plt.xlabel('True')
        plt.ylabel('Predicted')
        plt.grid(True)
    plt.tight_layout()
    plt.savefig('phase_comparison.png')
    plt.show()
    
    # 計算預測的能量
    pred_energies = np.zeros((len(pred_amplitudes), 64))
    for i in range(len(pred_amplitudes)):
        # 計算原始尺度的能量
        pred_energies[i] = calculate_energy(pred_amplitudes[i], pred_phases[i])
    
    # 繪製能量比較
    plt.figure(figsize=(24, 24))
    for i in range(64):
        plt.subplot(8, 8, i+1)
        plt.scatter(energy_samples[:, i], pred_energies[:, i], alpha=0.5, s=10)  # 減小點的大小
        min_val = min(energy_samples[:, i].min(), pred_energies[:, i].min())
        max_val = max(energy_samples[:, i].max(), pred_energies[:, i].max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')
        plt.title(f'Energy {i+1}', fontsize=10)  # 減小標題字體大小
        plt.xlabel('True', fontsize=8)  # 減小標籤字體大小
        plt.ylabel('Predicted', fontsize=8)  # 減小標籤字體大小
        plt.grid(True)
        plt.tick_params(axis='both', which='major', labelsize=7)  # 減小刻度標籤大小
    
    plt.tight_layout()
    plt.savefig('energy_comparison_all_points.png', dpi=300)  # 增加DPI以提高清晰度
    plt.show()

    # 額外提供一個能量相關性的總體視圖
    plt.figure(figsize=(10, 8))
    all_true_energy = energy_samples.flatten()
    all_pred_energy = pred_energies.flatten()
    plt.scatter(all_true_energy, all_pred_energy, alpha=0.3, s=5)
    min_val = min(all_true_energy.min(), all_pred_energy.min())
    max_val = max(all_true_energy.max(), all_pred_energy.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    plt.title('Overall Energy Correlation')
    plt.xlabel('True Energy')
    plt.ylabel('Predicted Energy')
    plt.grid(True)
    plt.savefig('overall_energy_correlation.png')
    plt.show()
    
    # 計算並顯示相關係數
    correlation = np.corrcoef(all_true_energy, all_pred_energy)[0, 1]
    print(f"Overall energy correlation coefficient: {correlation:.4f}")

# 可視化預測結果
visualize_predictions(model, test_loader)

# 使用模型進行預測
def predict_with_model(model, energy_input):
    model.eval()
    with torch.no_grad():
        # 將輸入能量放大 10^6 倍
        energy_tensor = torch.tensor(energy_input * 1e5, dtype=torch.float32).unsqueeze(0)
        pred_amplitude, pred_phase = model(energy_tensor)
        return pred_amplitude.numpy()[0], pred_phase.numpy()[0]

# 測試模型預測
sample_energy = energy_data[0]
pred_amplitude, pred_phase = predict_with_model(model, sample_energy)
print("Sample Energy:", sample_energy)
print("Predicted Amplitude:", pred_amplitude)
print("Predicted Phase:", pred_phase)

# 計算預測能量 (原始尺度)
pred_energy = calculate_energy(pred_amplitude, pred_phase)
print("Recalculated Energy (first 10 points):", pred_energy[:10])
print("Original Energy (first 10 points):", sample_energy[:10])

# =========================
# 1️⃣ 學習率曲線
# =========================
def plot_learning_rate(optimizer, scheduler, num_epochs):
    lr_list = []

    # 如果使用 ReduceLROnPlateau，需要手動紀錄每次 step 的 lr
    for epoch in range(num_epochs):
        lr_list.append(optimizer.param_groups[0]['lr'])
        # 這裡只示意，實際學習率變化已在 train_model 內用 scheduler.step(val_loss)
        # 可以在 train_model 內每次 epoch 記錄 lr，更精確
    plt.figure(figsize=(10,5))
    plt.plot(lr_list, label='Learning Rate')
    plt.title('Epoch vs Learning Rate')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.grid(True)
    plt.legend()
    plt.savefig('learning_rate_curve.png')
    plt.show()

# 調用
plot_learning_rate(optimizer, scheduler, num_epochs=200)


# =========================
# 2️⃣ 能量分布分析
# =========================
def plot_energy_distribution(model, test_loader):
    model.eval()
    all_true_energy = []
    all_pred_energy = []

    with torch.no_grad():
        for energy, amplitude, phase in test_loader:
            pred_amplitude, pred_phase = model(energy)
            # 將能量縮小回原始尺度
            true_energy = energy.numpy() / 1e5
            pred_energy = np.zeros_like(true_energy)
            for i in range(len(pred_amplitude)):
                pred_energy[i] = calculate_energy(pred_amplitude[i], pred_phase[i])
            all_true_energy.append(true_energy)
            all_pred_energy.append(pred_energy)

    all_true_energy = np.vstack(all_true_energy).flatten()
    all_pred_energy = np.vstack(all_pred_energy).flatten()

    plt.figure(figsize=(12,6))
    plt.hist(all_true_energy, bins=50, alpha=0.5, label='True U')
    plt.hist(all_pred_energy, bins=50, alpha=0.5, label='Predicted U')
    plt.title('U Distribution Comparison')
    plt.xlabel('U')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True)
    plt.savefig('energy_distribution_hist.png')
    plt.show()

    # 可以額外做 KDE 曲線
    try:
        import seaborn as sns
        plt.figure(figsize=(12,6))
        sns.kdeplot(all_true_energy, label='True U', fill=True)
        sns.kdeplot(all_pred_energy, label='Predicted U', fill=True)
        plt.title('U Distribution')
        plt.xlabel('U')
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True)
        plt.savefig('energy_distribution_kde.png')
        plt.show()
    except ImportError:
        print("Seaborn not installed, skipping KDE plot.")

# 調用
plot_energy_distribution(model, test_loader)
