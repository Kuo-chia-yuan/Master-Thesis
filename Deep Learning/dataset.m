% 設定觀測點和探頭數量
m = 64;  % 觀測點數量
n = 64;   % 探頭數量

% 計算探頭位置 (3×3網格)
sensor_positions = zeros(64, 3);

% 計算起始位置(左下角探頭的位置)
start_x = -0.035; 
start_y = -0.035;

idx = 1;
for i = 1:8  % y方向
    for j = 1:8  % x方向
        x = start_x + (j-1)*0.01;
        y = start_y + (i-1)*0.01;
        sensor_positions(idx,:) = [x, y, 0];
        idx = idx + 1;
    end
end

% 計算觀測點位置 (4×4×4網格)
point_positions = zeros(64, 3); 
idx = 1;

for z = 0.01:0.005:0.025
    for y = -0.0075:0.005:0.0075
        for x = -0.0075:0.005:0.0075
            point_positions(idx,:) = [x, y, z];
            idx = idx + 1;
        end
    end
end

% 物理參數設置
f = 40e3;            % 探頭頻率 (40 kHz)
w = 2 * pi * f;      % 角頻率 (rad/s)
lambda = 343 / f;    % 波長 (m)
k = 2 * pi / lambda; % 波數 (wavenumber)
rho = 1.225;         % 空氣密度 (kg/m^3)
c = 343;             % 空氣中的聲速 (m/s)
Area = 0.0008;       % 探頭面積 (m^2)
u = 3.086;           % 質點速度幅值 (m/s)
R = 0.000865;        % 懸浮球半徑 (1 mm)
weight = 2.84e-6;    % 球體重量 (kg)

% -----矩陣 h-----

% 初始化儲存所有h向量的cell陣列
h = cell(1,m);
for i = 1:m
    h{i} = zeros(1,n);  % 每個h是1×n的向量
end

% 計算常數項
jpck = 1j * rho * c * k;
constant_term = Area * jpck * u / (2*pi);

% 使用for迴圈計算每個觀測點的h向量
for i = 1:m  % 遍歷每個觀測點
    for j = 1:n  % 遍歷每個探頭
        dx = point_positions(i,1) - sensor_positions(j,1);
        dy = point_positions(i,2) - sensor_positions(j,2);
        dz = point_positions(i,3) - sensor_positions(j,3);
        r = sqrt(dx^2 + dy^2 + dz^2);
        h{i}(j) = constant_term * exp(-1j*k*r) / r;
    end
end

% ----- 矩陣 q-----

% 定義小增量 delta（需要根據實際情況調整）
delta = 1e-6;

% 初始化q矩陣
q = cell(1,m);
for i = 1:m
    q{i} = zeros(3,n);
end

% 計算每個觀測點的q矩陣
for i = 1:m
    for j = 1:n
        % 基本位置向量
        r0 = point_positions(i,:) - sensor_positions(j,:);
        
        % x方向梯度
        dx = [delta, 0, 0];  % x方向的微小變動
        r_plus = r0 + dx;    % 在x方向上增加delta
        r_minus = r0 - dx;   % 在x方向上減少delta
        h_plus_x = Area * jpck * u * exp(-1j*k*norm(r_plus)) / (2*pi*norm(r_plus));
        h_minus_x = Area * jpck * u * exp(-1j*k*norm(r_minus)) / (2*pi*norm(r_minus));
        q{i}(1,j) = -1/(1j*w*rho) * ((h_plus_x - h_minus_x)/(2*delta));
        
        % y方向梯度
        dy = [0, delta, 0];  % y方向的微小變動
        r_plus = r0 + dy;    % 在y方向上增加delta
        r_minus = r0 - dy;   % 在y方向上減少delta
        h_plus_y = Area * jpck * u * exp(-1j*k*norm(r_plus)) / (2*pi*norm(r_plus));
        h_minus_y = Area * jpck * u * exp(-1j*k*norm(r_minus)) / (2*pi*norm(r_minus));
        q{i}(2,j) = -1/(1j*w*rho) * ((h_plus_y - h_minus_y)/(2*delta));
        
        % z方向梯度
        dz = [0, 0, delta];  % z方向的微小變動
        r_plus = r0 + dz;    % 在z方向上增加delta
        r_minus = r0 - dz;   % 在z方向上減少delta
        h_plus_z = Area * jpck * u * exp(-1j*k*norm(r_plus)) / (2*pi*norm(r_plus));
        h_minus_z = Area * jpck * u * exp(-1j*k*norm(r_minus)) / (2*pi*norm(r_minus));
        q{i}(3,j) = -1/(1j*w*rho) * ((h_plus_z - h_minus_z)/(2*delta));
    end
end

% 計算常數項
const = 2 * pi * R^3;
term1_coef = 1/(6*rho*c^2);
term2_coef = -rho/4;

% 計算c1和c2
c1 = const * term1_coef;      % c1 = 2πR³ * 1/(6ρc²)
c2 = const * term2_coef;      % c2 = 2πR³ * (-ρ/4)

% 創建CSV文件
csv_filename = 'training_data_64x64_1.csv';

% 生成CSV表頭
header = cell(1, n + n + m);  % 振幅 + 相位 + 能量
for i = 1:n
    header{i} = sprintf('amplitude_%d', i);
end
for i = 1:n
    header{n+i} = sprintf('phase_%d', i);
end
for i = 1:m
    header{2*n+i} = sprintf('energy_%d', i);
end

% 寫入CSV表頭
fid = fopen(csv_filename, 'w');
fprintf(fid, '%s,', header{1:end-1});
fprintf(fid, '%s\n', header{end});
fclose(fid);

% 設定總迭代次數和有效數據計數
target_valid_count = 10000;  % 目標有效數據數量
valid_count = 0;             % 有效數據計數器
max_iterations = 11000;      % 最大嘗試次數，防止無限循環

% 設定能量範圍的起始和結束值
start_E_min = -9.9e-5;
end_E_min = -2.1e-5;
start_E_max = -7.9e-5;
end_E_max = -0.1e-5;

% 創建進度條
progress_bar = waitbar(0, '生成訓練數據中...');

% 執行迭代直到獲得足夠的有效數據
iteration = 0;
while valid_count < target_valid_count && iteration < max_iterations
    iteration = iteration + 1;
    
    % 更新進度條
    waitbar(valid_count/target_valid_count, progress_bar, ...
            sprintf('生成訓練數據中... %d/%d (嘗試次數: %d)', valid_count, target_valid_count, iteration));
    
    % 計算當前能量值 (線性插值)
    if valid_count < target_valid_count
        progress = valid_count / target_valid_count;
        E_min = start_E_min + progress * (end_E_min - start_E_min);
        E_max = start_E_max + progress * (end_E_max - start_E_max);
    end
    
    % 執行求解 - 修正：接收min_point_idx作為返回值
    [A_complex, E_final, min_point_idx] = solve_A_silent(h, q, c1, c2, m, n, point_positions, E_min, E_max);
    
    % 檢查隨機選擇的中心點是否為能量最低點
    if E_final(min_point_idx) == min(E_final)
        % 符合條件，增加有效計數
        valid_count = valid_count + 1;
        
        % 準備相位數據（轉換為0-1範圍，代表0-2π）
        phases = angle(A_complex) / (2*pi);
        phases(phases < 0) = phases(phases < 0) + 1;  % 將負相位轉換為0-1範圍
        
        % 將數據寫入CSV - 修正維度問題
        data_row = [abs(A_complex(:)); phases(:); E_final(:)]';  % 確保所有數據都是列向量，然後轉置為行向量
        
        % 追加寫入CSV
        dlmwrite(csv_filename, data_row, '-append');
        
        % 每100個有效數據顯示一次進度
        if mod(valid_count, 100) == 0
            fprintf('已找到 %d 個有效數據，當前嘗試次數: %d，目前能量: E_min=%.2e, E_max=%.2e\n', ...
                    valid_count, iteration, E_min, E_max);
        end
    end
end

% 關閉進度條
close(progress_bar);

% 顯示完成訊息
if valid_count >= target_valid_count
    fprintf('已完成 %d 組有效數據的生成，共嘗試 %d 次，並保存至 %s\n', valid_count, iteration, csv_filename);
else
    fprintf('已達到最大嘗試次數 %d，但僅找到 %d 組有效數據，已保存至 %s\n', max_iterations, valid_count, csv_filename);
end

% ===== 函數定義 =====

% 無輸出版的求解函數（用於批量處理）
function [A_complex, E_final, min_point_idx] = solve_A_silent(h, q, c1, c2, m, n, point_positions, E_min, E_max)
    % 初始化64個觀測點的target_E
    target_E = zeros(1, 64);
    
    % 找出中間的八個點（2×2×2的中心點）
    central_points = [];
    central_x = [-0.0025, 0.0025];
    central_y = [-0.0025, 0.0025];
    central_z = [0.015, 0.02];
    
    % 儲存中間八個點的索引
    central_indices = [];
    
    % 找出中間八個點的索引
    for i = 1:m
        x = point_positions(i,1);
        y = point_positions(i,2);
        z = point_positions(i,3);
        
        if (ismember(round(x,6), central_x) && ismember(round(y,6), central_y) && ismember(round(z,6), central_z))
            central_indices = [central_indices, i];
            central_points = [central_points; x, y, z];
        end
    end
    
    % 確認是否找到了八個中心點
    if length(central_indices) ~= 8
        error('未能找到中間的八個點，實際找到 %d 個點', length(central_indices));
    end
    
    % 隨機選擇一個中心點作為局部最小值
    min_point_idx = central_indices(randi(8));
    min_point_pos = point_positions(min_point_idx,:);
    
    % 為64個觀測點創建能量分布
    for idx = 1:64
        % 計算當前點的實際座標
        current_pos = point_positions(idx,:);
        
        % 計算到選定最小值點的距離
        dist = sqrt(sum((current_pos - min_point_pos).^2));
        
        % 使用高斯分布創建平滑的能量分布
        sigma = 0.004;  % 調整分布的寬度
        energy = E_min + (E_max - E_min) * (1 - exp(-dist^2/(2*sigma^2)));
        
        target_E(idx) = energy;
    end
    
    % 多起點優化
    best_A = [];
    best_error = Inf;

    for trial = 1:1  % 減少嘗試次數以提高效率
        % 設定初始 A 向量 - 使用非零隨機初始值
        A0 = zeros(2*n, 1);
        % 設定實部（振幅）為隨機值
        A0(1:n) = 10 + 5*randn(n,1);  
        % 設定虛部（相位）為隨機值
        A0(n+1:end) = 2*pi*rand(n,1) - pi;  
        
        % 設定約束條件
        lb = -200 * ones(2*n, 1);
        ub = 200 * ones(2*n, 1);
        
        % 優化設定 - 減少迭代次數以提高效率
        options = optimoptions('fmincon', ...
            'Display', 'off', ...
            'MaxFunctionEvaluations', 15000, ...
            'MaxIterations', 1000, ...
            'OptimalityTolerance', 1e-10, ...
            'StepTolerance', 1e-10);
        
        % 使用 fmincon 求解
        [A_optimal, fval] = fmincon(@(x) calculate_objective(x, h, q, c1, c2, target_E, m, n), ...
            A0, [], [], [], [], lb, ub, [], options);
        
        % 如果這次的結果更好，則保存
        if fval < best_error
            best_error = fval;
            best_A = A_optimal;
        end
    end

    % 使用最佳結果
    A_optimal = best_A;
    
    % 轉換為複數形式
    A_complex = A_optimal(1:n) + 1i*A_optimal(n+1:end);
    
    % 計算最終能量
    E_final = zeros(1, m);
    for i = 1:m
        h_term = h{i}' * h{i};
        q_term = q{i}' * q{i};
        E_final(i) = real(A_complex' * (c1*h_term + c2*q_term) * A_complex);
    end
end

% 目標函數計算
function f = calculate_objective(A_vec, h, q, c1, c2, target_E, m, n)
    % 將輸入向量轉換為複數振幅
    A = A_vec(1:n) + 1i*A_vec(n+1:end);
    
    % 計算振幅
    amplitudes = abs(A);
    
    % 添加振幅下限懲罰
    min_amplitude = 0.1;  % 最小振幅要求
    amp_penalty = 1e6 * sum(max(0, min_amplitude - amplitudes).^2);
    
    % 計算當前能量
    E_calc = zeros(1, m);
    for i = 1:m
        h_term = h{i}' * h{i};
        q_term = q{i}' * q{i};
        E_calc(i) = real(A' * (c1*h_term + c2*q_term) * A);
    end
    
    % 使用相對誤差
    rel_error = 0;
    for i = 1:m
        if target_E(i) ~= 0
            rel_error = rel_error + ((E_calc(i) - target_E(i))/abs(target_E(i)))^2;
        else
            rel_error = rel_error + (E_calc(i))^2;
        end
    end
    
    % 最終目標函數
    f = rel_error + amp_penalty;
end