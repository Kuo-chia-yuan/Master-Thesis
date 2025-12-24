% =======================
% 探頭陣列 & 觀測點設定
% =======================
n = 25;  % 探頭數量

% 探頭位置 (5×5網格)
sensor_positions = zeros(n, 3);
spacing = 0.018;  % 1.8cm = 0.018m
start_x = -(spacing * (5-1))/2;
start_y = -(spacing * (5-1))/2;
idx = 1;
for i = 1:5
    for j = 1:5
        x = start_x + (j-1)*spacing;
        y = start_y + (i-1)*spacing;
        sensor_positions(idx,:) = [x, y, 0];
        idx = idx + 1;
    end
end

% 觀測點位置 (31×31×31網格)
dx = 0.0005;  % 間距 0.5mm
x_vals = linspace(-0.005, 0.005, 31);
y_vals = linspace(-0.005, 0.005, 31);
z_vals = linspace(0.01, 0.02, 31);

m = numel(x_vals) * numel(y_vals) * numel(z_vals);
point_positions = zeros(m, 3);

idx = 1;
for z = z_vals
    for y = y_vals
        for x = x_vals
            point_positions(idx, :) = [x, y, z];
            idx = idx + 1;
        end
    end
end

% =======================
% 物理參數
% =======================
f = 40e3;            % 探頭頻率 (Hz)
w = 2 * pi * f;      
lambda = 343 / f;    
k = 2 * pi / lambda; 
rho = 1.225;         
c = 343;             
Area = 0.0002;       
u = 3.086;           
R = 0.000865;        
weight = 2e-7;       

% =======================
% h矩陣計算
% =======================
jpck = 1j * rho * c * k;
constant_term = Area * jpck * u / (2*pi);

h = cell(1,m);
for i = 1:m
    h{i} = zeros(1,n);
end
for i = 1:m
    for j = 1:n
        dx = point_positions(i,1) - sensor_positions(j,1);
        dy = point_positions(i,2) - sensor_positions(j,2);
        dz = point_positions(i,3) - sensor_positions(j,3);
        r = sqrt(dx^2 + dy^2 + dz^2);
        h{i}(j) = constant_term * exp(-1j*k*r) / r;
    end
end

% =======================
% q矩陣計算
% =======================
delta = 1e-6;
q = cell(1,m);
for i = 1:m
    q{i} = zeros(3,n);
end
for i = 1:m
    for j = 1:n
        r0 = point_positions(i,:) - sensor_positions(j,:);
        % x方向
        dx = [delta, 0, 0];
        r_plus = r0 + dx; r_minus = r0 - dx;
        h_plus_x = Area * jpck * u * exp(-1j*k*norm(r_plus)) / (2*pi*norm(r_plus));
        h_minus_x = Area * jpck * u * exp(-1j*k*norm(r_minus)) / (2*pi*norm(r_minus));
        q{i}(1,j) = -1/(1j*w*rho) * ((h_plus_x - h_minus_x)/(2*delta));
        % y方向
        dy = [0, delta, 0];
        r_plus = r0 + dy; r_minus = r0 - dy;
        h_plus_y = Area * jpck * u * exp(-1j*k*norm(r_plus)) / (2*pi*norm(r_plus));
        h_minus_y = Area * jpck * u * exp(-1j*k*norm(r_minus)) / (2*pi*norm(r_minus));
        q{i}(2,j) = -1/(1j*w*rho) * ((h_plus_y - h_minus_y)/(2*delta));
        % z方向
        dz = [0, 0, delta];
        r_plus = r0 + dz; r_minus = r0 - dz;
        h_plus_z = Area * jpck * u * exp(-1j*k*norm(r_plus)) / (2*pi*norm(r_plus));
        h_minus_z = Area * jpck * u * exp(-1j*k*norm(r_minus)) / (2*pi*norm(r_minus));
        q{i}(3,j) = -1/(1j*w*rho) * ((h_plus_z - h_minus_z)/(2*delta));
    end
end

% =======================
% 常數項
% =======================
const = 2 * pi * R^3;
c1 = const * (1/(6*rho*c^2));
c2 = const * (-rho/4);

% =======================
% 直接指定探頭振幅和相位
% =======================
A_amp =  [5; 5; 10; 15; 5; 
          5; 10; 5; 5; 10; 
          15; 10; 0; 0; 5; 
          15; 15; 5; 15; 15; 
          10; 5; 10; 10; 5];

A_phase = [0; 0.2; 0.2; 0; 0; 
           0; 0; 0; 0; 0; 
           0; 0.4; 0; 0; 0; 
           0; 0; 0.2; 0; 0; 
           0; -0.2; 0.2; 0.4; 0]; 

A_complex = A_amp .* exp(1j * A_phase);   % 25×1複數

% =======================
% 計算每個觀測點能量
% =======================
E_final = zeros(1, m);
for i = 1:m
    h_term = h{i}' * h{i};
    q_term = q{i}' * q{i};
    E_final(i) = real(A_complex' * (c1*h_term + c2*q_term) * A_complex);
end

% =======================
% 計算力場
% =======================
F_new = calculate_force_direct(point_positions, E_final, m);

% =======================
% 識別所有能量陷阱 (局部最小值)
% =======================
% 首先重新組織數據為 3D 網格
E_grid = reshape(E_final, [length(x_vals), length(y_vals), length(z_vals)]);

% 初始化陷阱計數和位置
trap_count = 0;
trap_positions = [];
trap_energies = [];

% 遍歷內部網格點 (排除邊界)
for i = 2:length(x_vals)-1
    for j = 2:length(y_vals)-1
        for k = 2:length(z_vals)-1
            % 獲取當前點能量
            current_energy = E_grid(i,j,k);
            
            % 檢查是否為局部最小值 (與所有相鄰點比較)
            is_minimum = true;
            
            % 檢查 26 個相鄰點
            for di = -1:1
                for dj = -1:1
                    for dk = -1:1
                        % 跳過自身
                        if di == 0 && dj == 0 && dk == 0
                            continue;
                        end
                        
                        % 如果任何相鄰點能量更低，則不是局部最小值
                        if E_grid(i+di,j+dj,k+dk) < current_energy
                            is_minimum = false;
                            break;
                        end
                    end
                    if ~is_minimum
                        break;
                    end
                end
                if ~is_minimum
                    break;
                end
            end
            
            % 如果是局部最小值，記錄位置
            if is_minimum
                trap_count = trap_count + 1;
                
                % 計算實際位置座標
                pos_idx = (k-1)*length(x_vals)*length(y_vals) + (j-1)*length(x_vals) + i;
                trap_positions(trap_count,:) = point_positions(pos_idx,:);
                trap_energies(trap_count) = current_energy;
            end
        end
    end
end

% =======================
% 顯示結果
% =======================
fprintf('找到 %d 個能量陷阱 (局部最小值)\n', trap_count);

% 顯示每個陷阱的位置和能量
for i = 1:trap_count
    fprintf('陷阱 #%d: 位置 [%.4f, %.4f, %.4f] mm, 能量: %.4e J\n', ...
            i, trap_positions(i,1)*1000, trap_positions(i,2)*1000, trap_positions(i,3)*1000, trap_energies(i));
end

% =======================
% 繪圖部分 - 無論是否找到陷阱都執行
% =======================
% 設定字體大小
fontSize = 16;
titleSize = 20;

% 將座標轉換為毫米
point_positions_mm = point_positions * 1000;

if trap_count > 0
    % 有找到陷阱的情況
    [min_energy, min_idx] = min(trap_energies);
    min_trap_pos = trap_positions(min_idx,:);
    trap_positions_mm = trap_positions * 1000;
    min_trap_pos_mm = min_trap_pos * 1000;
    
    fprintf('\n能量最小的陷阱: #%d, 位置 [%.4f, %.4f, %.4f] mm, 能量: %.4e J\n', ...
            min_idx, min_trap_pos_mm(1), min_trap_pos_mm(2), min_trap_pos_mm(3), min_energy);
    
    % 設定顯示半徑
    display_radius = 0.0025;  % 2.5mm
    
    % 找出最小能量陷阱周圍的點
    distances = sqrt(sum((point_positions - min_trap_pos).^2, 2));
    display_indices = find(distances <= display_radius);
    
    % 繪製能量場圖
    figure('Position', [100, 100, 1000, 800]);
    
    % 只繪製局部最小值周圍的點
    scatter3(point_positions_mm(display_indices,1), point_positions_mm(display_indices,2), point_positions_mm(display_indices,3), ...
             20, E_final(display_indices), 'filled');
    hold on;
    
    % 標記最小能量陷阱
    scatter3(min_trap_pos_mm(1), min_trap_pos_mm(2), min_trap_pos_mm(3), ...
             200, 'r', 'filled', 'MarkerEdgeColor', 'k', 'LineWidth', 2);
    
    colorbar;
    colormap(jet);
    xlabel('X [mm]', 'FontSize', fontSize, 'FontWeight', 'bold');
    ylabel('Y [mm]', 'FontSize', fontSize, 'FontWeight', 'bold');
    zlabel('Z [mm]', 'FontSize', fontSize, 'FontWeight', 'bold');
    title(sprintf('能量最小陷阱位置 (%.2f, %.2f, %.2f) mm', min_trap_pos_mm(1), min_trap_pos_mm(2), min_trap_pos_mm(3)), ...
          'FontSize', titleSize, 'FontWeight', 'bold');
    grid on;
    axis equal;
    view(45, 30);
    set(gca, 'FontSize', fontSize);
    
    % =======================
    % 繪製三軸能量曲線圖
    % =======================
    % 尋找通過最小能量陷阱的三個軸線
    tolerance = 1e-12;
    
    % X軸線 (y,z固定)
    x_line_idx = find(abs(point_positions(:,2) - min_trap_pos(2)) < tolerance & ...
                      abs(point_positions(:,3) - min_trap_pos(3)) < tolerance);
    [x_sorted, sort_idx] = sort(point_positions(x_line_idx,1));
    Ex_sorted = E_final(x_line_idx(sort_idx));
    
    % Y軸線 (x,z固定)
    y_line_idx = find(abs(point_positions(:,1) - min_trap_pos(1)) < tolerance & ...
                      abs(point_positions(:,3) - min_trap_pos(3)) < tolerance);
    [y_sorted, sort_idx] = sort(point_positions(y_line_idx,2));
    Ey_sorted = E_final(y_line_idx(sort_idx));
    
    % Z軸線 (x,y固定)
    z_line_idx = find(abs(point_positions(:,1) - min_trap_pos(1)) < tolerance & ...
                      abs(point_positions(:,2) - min_trap_pos(2)) < tolerance);
    [z_sorted, sort_idx] = sort(point_positions(z_line_idx,3));
    Ez_sorted = E_final(z_line_idx(sort_idx));
    
else
    % 沒有找到陷阱的情況 - 使用全域最小能量點
    fprintf('\n沒有找到局部最小值陷阱，使用全域最小能量點進行視覺化\n');
    
    [min_energy, min_idx] = min(E_final);
    min_trap_pos = point_positions(min_idx,:);
    min_trap_pos_mm = min_trap_pos * 1000;
    
    fprintf('全域最小能量點: 位置 [%.4f, %.4f, %.4f] mm, 能量: %.4e J\n', ...
            min_trap_pos_mm(1), min_trap_pos_mm(2), min_trap_pos_mm(3), min_energy);
    
    % 設定顯示半徑
    display_radius = 0.0025;  % 2.5mm
    
    % 找出最小能量點周圍的點
    distances = sqrt(sum((point_positions - min_trap_pos).^2, 2));
    display_indices = find(distances <= display_radius);
    
    % 繪製能量場圖
    figure('Position', [100, 100, 1000, 800]);
    
    % 繪製最小能量點周圍的點
    scatter3(point_positions_mm(display_indices,1), point_positions_mm(display_indices,2), point_positions_mm(display_indices,3), ...
             20, E_final(display_indices), 'filled');
    hold on;
    
    % 標記最小能量點
    scatter3(min_trap_pos_mm(1), min_trap_pos_mm(2), min_trap_pos_mm(3), ...
             200, 'b', 'filled', 'MarkerEdgeColor', 'k', 'LineWidth', 2);
    
    colorbar;
    colormap(jet);
    xlabel('X [mm]', 'FontSize', fontSize, 'FontWeight', 'bold');
    ylabel('Y [mm]', 'FontSize', fontSize, 'FontWeight', 'bold');
    zlabel('Z [mm]', 'FontSize', fontSize, 'FontWeight', 'bold');
    title(sprintf('全域最小能量點位置 (%.2f, %.2f, %.2f) mm', min_trap_pos_mm(1), min_trap_pos_mm(2), min_trap_pos_mm(3)), ...
          'FontSize', titleSize, 'FontWeight', 'bold');
    grid on;
    axis equal;
    view(45, 30);
    set(gca, 'FontSize', fontSize);
    
    % =======================
    % 繪製三軸能量曲線圖
    % =======================
    tolerance = 1e-12;
    
    % X軸線 (y,z固定)
    x_line_idx = find(abs(point_positions(:,2) - min_trap_pos(2)) < tolerance & ...
                      abs(point_positions(:,3) - min_trap_pos(3)) < tolerance);
    [x_sorted, sort_idx] = sort(point_positions(x_line_idx,1));
    Ex_sorted = E_final(x_line_idx(sort_idx));
    
    % Y軸線 (x,z固定)
    y_line_idx = find(abs(point_positions(:,1) - min_trap_pos(1)) < tolerance & ...
                      abs(point_positions(:,3) - min_trap_pos(3)) < tolerance);
    [y_sorted, sort_idx] = sort(point_positions(y_line_idx,2));
    Ey_sorted = E_final(y_line_idx(sort_idx));
    
    % Z軸線 (x,y固定)
    z_line_idx = find(abs(point_positions(:,1) - min_trap_pos(1)) < tolerance & ...
                      abs(point_positions(:,2) - min_trap_pos(2)) < tolerance);
    [z_sorted, sort_idx] = sort(point_positions(z_line_idx,3));
    Ez_sorted = E_final(z_line_idx(sort_idx));
end

% =======================
% 繪製三軸能量曲線圖 (共通部分)
% =======================
fontSize = 18;
lineWidth = 2;
markerSize = 8;

figure('Position', [100,100,1200,400]);

% X方向
subplot(1,3,1);
plot(x_sorted*1000, Ex_sorted, 'b-o','LineWidth',lineWidth,'MarkerSize',markerSize); hold on;
xlabel('X Position [mm]', 'FontSize', fontSize, 'FontWeight', 'bold'); 
ylabel('U [J]', 'FontSize', fontSize, 'FontWeight', 'bold');
title('X Position vs U', 'FontSize', fontSize+2, 'FontWeight', 'bold');
grid on;
set(gca, 'FontSize', fontSize);

% Y方向
subplot(1,3,2);
plot(y_sorted*1000, Ey_sorted, 'r-o','LineWidth',lineWidth,'MarkerSize',markerSize); hold on;
xlabel('Y Position [mm]', 'FontSize', fontSize, 'FontWeight', 'bold'); 
ylabel('U [J]', 'FontSize', fontSize, 'FontWeight', 'bold');
title('Y Position vs U', 'FontSize', fontSize+2, 'FontWeight', 'bold');
grid on;
set(gca, 'FontSize', fontSize);

% Z方向
subplot(1,3,3);
plot(z_sorted*1000, Ez_sorted, 'g-o','LineWidth',lineWidth,'MarkerSize',markerSize); hold on;
xlabel('Z Position [mm]', 'FontSize', fontSize, 'FontWeight', 'bold'); 
ylabel('U [J]', 'FontSize', fontSize, 'FontWeight', 'bold');
title('Z Position vs U', 'FontSize', fontSize+2, 'FontWeight', 'bold');
grid on;
set(gca, 'FontSize', fontSize);

% 調整整體圖形大小和間距
set(gcf, 'Units','Normalized','OuterPosition',[0 0 1 0.5]);
set(gcf, 'PaperPositionMode', 'auto');

% =======================
% 力場計算函數
% =======================
function F = calculate_force_direct(point_positions, E_values, m)
    F = zeros(m, 3);
    
    dx = 0.0005; dy = 0.0005; dz = 0.0005;
    unique_x = unique(round(point_positions(:,1), 6));
    unique_y = unique(round(point_positions(:,2), 6));
    unique_z = unique(round(point_positions(:,3), 6));
    pos_to_idx = containers.Map('KeyType', 'char', 'ValueType', 'double');
    for i = 1:m
        key = sprintf('%.6f,%.6f,%.6f', round(point_positions(i,1), 6), ...
                                        round(point_positions(i,2), 6), ...
                                        round(point_positions(i,3), 6));
        pos_to_idx(key) = i;
    end
    for i = 1:m
        x = round(point_positions(i,1), 6);
        y = round(point_positions(i,2), 6);
        z = round(point_positions(i,3), 6);
        % X方向
        if ismember(x + dx, unique_x) && ismember(x - dx, unique_x)
            key_plus = sprintf('%.6f,%.6f,%.6f', x + dx, y, z);
            key_minus = sprintf('%.6f,%.6f,%.6f', x - dx, y, z);
            if pos_to_idx.isKey(key_plus) && pos_to_idx.isKey(key_minus)
                idx_plus = pos_to_idx(key_plus);
                idx_minus = pos_to_idx(key_minus);
                F(i,1) = -(E_values(idx_plus) - E_values(idx_minus))/(2*dx);
            end
        elseif ismember(x + dx, unique_x)
            key_plus = sprintf('%.6f,%.6f,%.6f', x + dx, y, z);
            if pos_to_idx.isKey(key_plus)
                idx_plus = pos_to_idx(key_plus);
                F(i,1) = -(E_values(idx_plus) - E_values(i))/dx;
            end
        elseif ismember(x - dx, unique_x)
            key_minus = sprintf('%.6f,%.6f,%.6f', x - dx, y, z);
            if pos_to_idx.isKey(key_minus)
                idx_minus = pos_to_idx(key_minus);
                F(i,1) = -(E_values(i) - E_values(idx_minus))/dx;
            end
        end
        % Y方向
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
        % Z方向
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
