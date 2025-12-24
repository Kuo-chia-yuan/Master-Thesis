% 設定觀測點和探頭數量
m = 64;  % 觀測點數量
n = 25;   % 探頭數量

% 計算探頭位置 (8×8網格)
sensor_positions = zeros(25 , 3);

% 計算起始位置(左下角探頭的位置)
% 新的間隔
spacing = 0.01;  % 1.8cm = 0.018m

% 計算新的起始位置，使陣列中心在原點
start_x = -(spacing * (5-1))/2;  % = -0.063
start_y = -(spacing * (5-1))/2;  % = -0.063

idx = 1;
for i = 1:5  % y方向
    for j = 1:5  % x方向
        x = start_x + (j-1)*spacing;
        y = start_y + (i-1)*spacing;
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

% -----矩陣 E-----
E_final = zeros(1,m);

% 執行求解 (獲得最小值點位置)
[A, E_final, min_point_pos] = solve_A(h, q, c1, c2, m, n, point_positions);

% 確保 E_final 是列向量
E_final = E_final(:);  % 將 E_final 轉換為列向量

% 標記局部最小值點
hold on;
plot3(min_point_pos(1), min_point_pos(2), min_point_pos(3), '*', 'MarkerSize', 5);
text(min_point_pos(1)+0.001, min_point_pos(2)+0.001, min_point_pos(3)+0.001, '局部最小值', 'FontSize', 12);
hold off;

% 找出通過局部最小值點的軸線
tolerance = 1e-10;

% X 軸線 (通過局部最小值點的 y, z 座標)
x_line_indices = find(abs(point_positions(:,2) - min_point_pos(2)) < tolerance & ...
                     abs(point_positions(:,3) - min_point_pos(3)) < tolerance);
x_values = point_positions(x_line_indices,1);
Ex_values = E_final(x_line_indices);
[x_sorted, sort_idx] = sort(x_values);
Ex_sorted = Ex_values(sort_idx);

% Y 軸線 (通過局部最小值點的 x, z 座標)
y_line_indices = find(abs(point_positions(:,1) - min_point_pos(1)) < tolerance & ...
                      abs(point_positions(:,3) - min_point_pos(3)) < tolerance);
y_values = point_positions(y_line_indices,2);
Ey_values = E_final(y_line_indices);
[y_sorted, sort_idx] = sort(y_values);
Ey_sorted = Ey_values(sort_idx);

% Z 軸線 (通過局部最小值點的 x, y 座標)
z_line_indices = find(abs(point_positions(:,1) - min_point_pos(1)) < tolerance & ...
                      abs(point_positions(:,2) - min_point_pos(2)) < tolerance);
z_values = point_positions(z_line_indices,3);
Ez_values = E_final(z_line_indices);
[z_sorted, sort_idx] = sort(z_values);
Ez_sorted = Ez_values(sort_idx);

% 繪製三個子圖
figure('Position', [100, 100, 900, 300]);

% X 軸方向的能量分布
subplot(1,3,1);
plot(x_sorted*1000, Ex_sorted, 'b-o', 'LineWidth', 1.5, 'MarkerSize', 6);
grid on;
xlabel('X Position [mm]');
ylabel('U [J]');
title(sprintf('X Position vs U (y=%.2f mm, z=%.2f mm)', min_point_pos(2)*1000, min_point_pos(3)*1000));
set(gca, 'FontSize', 12);

% Y 軸方向的能量分布
subplot(1,3,2);
plot(y_sorted*1000, Ey_sorted, 'r-o', 'LineWidth', 1.5, 'MarkerSize', 6);
grid on;
xlabel('Y Position [mm]');
ylabel('U [J]');
title(sprintf('Y Position vs U (x=%.2f mm, z=%.2f mm)', min_point_pos(1)*1000, min_point_pos(3)*1000));
set(gca, 'FontSize', 12);

% Z 軸方向的能量分布
subplot(1,3,3);
plot(z_sorted*1000, Ez_sorted, 'g-o', 'LineWidth', 1.5, 'MarkerSize', 6);
grid on;
xlabel('Z Position [mm]');
ylabel('U [J]');
title(sprintf('Z Position vs U (x=%.2f mm, y=%.2f mm)', min_point_pos(1)*1000, min_point_pos(2)*1000));
set(gca, 'FontSize', 12);

% 調整子圖之間的間距
set(gcf, 'Units', 'Normalized', 'OuterPosition', [0 0 1 0.4]);

% 計算力場
F_new = calculate_force_direct(point_positions, E_final, m);

% 計算力的大小
F_magnitude = sqrt(sum(F_new.^2, 2));

% 繪製力場圖
figure;

% 繪製能量分布的散點圖
scatter3(point_positions(:,1)*1e3, point_positions(:,2)*1e3, point_positions(:,3)*1e3, ...
         80, E_final, 'filled');
hold on;


% 設置箭頭縮放因子（根據網格大小自動調整）
scale_factor = 100 * min([range(unique(point_positions(:,1))), ...
                          range(unique(point_positions(:,2))), ...
                          range(unique(point_positions(:,3)))]);

% 繪製力向量
quiver3(point_positions(:,1)*1e3, point_positions(:,2)*1e3, point_positions(:,3)*1e3, ...
        F_new(:,1), F_new(:,2), F_new(:,3), scale_factor, ...
        'LineWidth', 1.2, 'Color', 'k', 'MaxHeadSize', 0.5);

% 添加顏色條和標籤
colorbar_handle = colorbar;      % 生成 colorbar 並取得句柄
colormap(jet);
ylabel(colorbar_handle, 'U [J]');  % 為 colorbar 加上標籤
xlabel('X [mm]');
ylabel('Y [mm]');
zlabel('Z [mm]');
title('U vs F');
grid on;
axis equal;
view(45, 30);

% 添加圖例
legend('U', 'F', 'Location', 'best');

% ===== 函數定義 =====

% 求解函數
function [A_complex, E_final, min_point_pos] = solve_A(h, q, c1, c2, m, n, point_positions)    
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
    
    % 定義能量範圍
    E_min = -6e-5;   % 局部最小值（負值表示吸引力）
    E_max = -4e-5;   % 外圍最大值
    
    % 選擇要設置局部最小值的點（可以通過輸入參數來選擇）
    % 這裡我們輪流設置八個中心點
    min_point_idx = mod(randi(100), 8) + 1;  % 隨機選擇一個中心點
    min_point = central_indices(min_point_idx);
    min_point_pos = point_positions(min_point,:);
    
    fprintf('選擇的局部最小值點：索引 = %d, 位置 = (%.4f, %.4f, %.4f)\n', ...
            min_point, min_point_pos(1), min_point_pos(2), min_point_pos(3));
    
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
    
    % 測試聲場計算
    test_A = ones(n, 1);  % 所有探頭振幅為1，相位為0
    test_E = zeros(1, m);

    for i = 1:m
        h_term = h{i}' * h{i};
        q_term = q{i}' * q{i};
        test_E(i) = real(test_A' * (c1*h_term + c2*q_term) * test_A);
    end

    % 檢查能量範圍
    fprintf('測試能量範圍: [%.2e, %.2e]\n', min(test_E), max(test_E));
    
    % 多起點優化
    best_A = [];
    best_error = Inf;

    for trial = 1:1  % 嘗試2次不同的起點
        A0 = zeros(2*n, 1);
        % 設定實部（振幅）為隨機值
        A0(1:n) = 10 + 5*randn(n,1);  
        % 設定虛部（相位）為隨機值
        A0(n+1:end) = 2*pi*rand(n,1) - pi;  
        
        % 設定約束條件
        lb = -200 * ones(2*n, 1);
        ub = 200 * ones(2*n, 1);
        
        % 優化設定
        options = optimoptions('fmincon', ...
            'Display', 'iter', ...
            'MaxFunctionEvaluations', 15000, ... % 最大函數評估次數
            'MaxIterations', 1500, ...           % 最大迭代次數
            'OptimalityTolerance', 1e-10, ...    % 最佳化容許誤差
            'StepTolerance', 1e-10);             % 步長容許誤差
        
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
    
    % 印出結果
    fprintf('\n最終 A 向量：\n');
    for i = 1:n
        % 計算振幅和相位
        amplitude = abs(A_complex(i));
        phase_rad = angle(A_complex(i));
        phase_2pi = phase_rad / (2*pi);  % 轉換為以 2π 為單位
        
        fprintf('A%d = %.4f + %.4fi\n', i, real(A_complex(i)), imag(A_complex(i)));
        fprintf('振幅: %.4f, 相位: %.4f×2π\n', amplitude, phase_2pi);
    end
    
    fprintf('\n能量比較：\n');
    fprintf('觀測點  目標能量    最終能量    相對誤差\n');
    fprintf('----------------------------------------\n');

    rel_errors = zeros(1,m);
    all = 0;
    for i = 1:m
        if target_E(i) ~= 0
            rel_errors = abs(E_final(i)-target_E(i))/abs(target_E(i))*100;
            all = all + rel_errors;
        else
            rel_errors = NaN;
        end
        fprintf('%d    %.4e      %.4e      %.2f%%\n', ...
            i, target_E(i), E_final(i), rel_errors);
    end
    % 計算平均準確度
    average_error = all/64;  % 以百分比表示
    fprintf('\naverage error = %.2f%%\n', average_error);
    
    % 顯示振幅和相位的摘要
    fprintf('\n振幅和相位摘要：\n');
    fprintf('探頭    振幅    相位(2π單位)\n');
    fprintf('---------------------------\n');
    for i = 1:n
        amplitude = abs(A_complex(i));
        phase_rad = angle(A_complex(i));
        phase_2pi = phase_rad / (2*pi);  % 轉換為以 2π 為單位
        fprintf('A%d    %.4f    %.4f\n', i, amplitude, phase_2pi);
    end

    % 在 solve_A 函數中，計算並輸出振幅和相位後，添加以下代碼
    % 準備數據以寫入CSV檔案
    amplitudes = abs(A_complex);
    phases = angle(A_complex) / (2 * pi); % 相位以2π為單位
    
    % 創建表格
    probe_data = table((1:n)', amplitudes, phases, ...
        'VariableNames', {'Probe_ID', 'Amplitude', 'Phase_2pi'});
    
    % 寫入CSV檔案
    writetable(probe_data, 'probe_amplitude_phase.csv');
    
    fprintf('\n數據已成功寫入到 probe_amplitude_phase.csv 檔案中。\n');

end

% 目標函數計算
function f = calculate_objective(A_vec, h, q, c1, c2, target_E, m, n)
    % 將輸入向量轉換為複數振幅
    A = A_vec(1:n) + 1i*A_vec(n+1:end);
    
    % 計算振幅
    amplitudes = abs(A);
    
    % 添加振幅限制懲罰
    min_amplitude = 0;  % 最小振幅要求
    max_amplitude = 14;  % 最大振幅限制
    amp_penalty = 1e6 * sum(max(0, min_amplitude - amplitudes).^2) + ...
                  1e6 * sum(max(0, amplitudes - max_amplitude).^2);
    
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

% 力場計算函數
function F = calculate_force_direct(point_positions, E_values, m)
    F = zeros(m, 3);
    
    % 對於4×4×4網格，使用固定間距
    dx = 0.005;  % 5mm間距
    dy = 0.005;
    dz = 0.005;
    
    % 獲取所有唯一的x、y、z值
    unique_x = unique(round(point_positions(:,1), 6));
    unique_y = unique(round(point_positions(:,2), 6));
    unique_z = unique(round(point_positions(:,3), 6));
    
    % 建立位置到索引的映射
    pos_to_idx = containers.Map('KeyType', 'char', 'ValueType', 'double');
    for i = 1:m
        key = sprintf('%.6f,%.6f,%.6f', round(point_positions(i,1), 6), ...
                                        round(point_positions(i,2), 6), ...
                                        round(point_positions(i,3), 6));
        pos_to_idx(key) = i;
    end
    
    % 計算每個點的力
    for i = 1:m
        x = round(point_positions(i,1), 6);
        y = round(point_positions(i,2), 6);
        z = round(point_positions(i,3), 6);
        
        % X方向力
        if ismember(x + dx, unique_x) && ismember(x - dx, unique_x)
            % 中心差分
            key_plus = sprintf('%.6f,%.6f,%.6f', x + dx, y, z);
            key_minus = sprintf('%.6f,%.6f,%.6f', x - dx, y, z);
            
            if pos_to_idx.isKey(key_plus) && pos_to_idx.isKey(key_minus)
                idx_plus = pos_to_idx(key_plus);
                idx_minus = pos_to_idx(key_minus);
                F(i,1) = -(E_values(idx_plus) - E_values(idx_minus))/(2*dx);
            end
        elseif ismember(x + dx, unique_x)
            % 前向差分
            key_plus = sprintf('%.6f,%.6f,%.6f', x + dx, y, z);
            if pos_to_idx.isKey(key_plus)
                idx_plus = pos_to_idx(key_plus);
                F(i,1) = -(E_values(idx_plus) - E_values(i))/dx;
            end
        elseif ismember(x - dx, unique_x)
            % 後向差分
            key_minus = sprintf('%.6f,%.6f,%.6f', x - dx, y, z);
            if pos_to_idx.isKey(key_minus)
                idx_minus = pos_to_idx(key_minus);
                F(i,1) = -(E_values(i) - E_values(idx_minus))/dx;
            end
        end
        
        % Y方向力 (類似X方向的計算)
        if ismember(y + dy, unique_y) && ismember(y - dy, unique_y)
            key_plus = sprintf('%.6f,%.6f,%.6f', x, y + dy, z);
            key_minus = sprintf('%.6f,%.6f,%.6f', x, y - dy, z);
            
            if pos_to_idx.isKey(key_plus) && pos_to_idx.isKey(key_minus)
                idx_plus = pos_to_idx(key_plus);
                idx_minus = pos_to_idx(key_minus);
                F(i,2) = -(E_values(idx_plus) - E_values(idx_minus))/(2*dy);
            end
        elseif ismember(y + dy, unique_y)
            key_plus = sprintf('%.6f,%.6f,%.6f', x, y + dy, z);
            if pos_to_idx.isKey(key_plus)
                idx_plus = pos_to_idx(key_plus);
                F(i,2) = -(E_values(idx_plus) - E_values(i))/dy;
            end
        elseif ismember(y - dy, unique_y)
            key_minus = sprintf('%.6f,%.6f,%.6f', x, y - dy, z);
            if pos_to_idx.isKey(key_minus)
                idx_minus = pos_to_idx(key_minus);
                F(i,2) = -(E_values(i) - E_values(idx_minus))/dy;
            end
        end
        
        % Z方向力 (類似X方向的計算)
        if ismember(z + dz, unique_z) && ismember(z - dz, unique_z)
            key_plus = sprintf('%.6f,%.6f,%.6f', x, y, z + dz);
            key_minus = sprintf('%.6f,%.6f,%.6f', x, y, z - dz);
            
            if pos_to_idx.isKey(key_plus) && pos_to_idx.isKey(key_minus)
                idx_plus = pos_to_idx(key_plus);
                idx_minus = pos_to_idx(key_minus);
                F(i,3) = -(E_values(idx_plus) - E_values(idx_minus))/(2*dz);
            end
        elseif ismember(z + dz, unique_z)
            key_plus = sprintf('%.6f,%.6f,%.6f', x, y, z + dz);
            if pos_to_idx.isKey(key_plus)
                idx_plus = pos_to_idx(key_plus);
                F(i,3) = -(E_values(idx_plus) - E_values(i))/dz;
            end
        elseif ismember(z - dz, unique_z)
            key_minus = sprintf('%.6f,%.6f,%.6f', x, y, z - dz);
            if pos_to_idx.isKey(key_minus)
                idx_minus = pos_to_idx(key_minus);
                F(i,3) = -(E_values(i) - E_values(idx_minus))/dz;
            end
        end
    end
end