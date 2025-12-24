%% 基本參數設置
f = 40e3;            % 探頭頻率 (40 kHz)
omega = 2 * pi * f;  % 角頻率 (rad/s)
lambda = 343 / f;    % 波長 (m)
k = 2 * pi / lambda; % 波數 (wavenumber)
rho0 = 1.225;        % 空氣密度 (kg/m^3)
c0 = 343;            % 空氣中的聲速 (m/s)
Area = 0.0002;       % 探頭的面積 (m^2)
u_n = 3.086;         % 質點速度幅值 (m/s)
R = 0.000865;        % 懸浮球半徑 (1 mm)
weight = 2.84e-7;    % 球體重量 (kg)

%% 定義發射點和觀測點 (z 軸方向)
z = linspace(0.0212, 0.0432, 100); % z 軸上的觀測點，範圍 0 ~ 3.6 cm
z_source1 = 0.0212;                % 第一個探頭的位置
z_source2 = 0.0432;                % 第二個探頭的位置

%% 定義相位範圍
phase_range = deg2rad(360:360:360);   % 將度數轉換為弧度

% 定義懸浮保麗龍球的所需最小力
F_required = weight * 9.81; % N

%% 初始化圖形 (力 + 能量同圖)
figure('Position', [100, 100, 800, 600]);  % 增大圖形尺寸
set(0, 'DefaultAxesFontSize', 14);         % 設置默認字體大小
set(0, 'DefaultTextFontSize', 14);         % 設置默認文字字體大小
hold on;

for phase2 = phase_range
    %% 計算探頭對觀測點的距離
    r1 = abs(z - z_source1);  
    r2 = abs(z - z_source2);  

    phase1 = 0;  % 下方探頭相位固定為 0

    %% 計算兩個探頭的聲壓
    p1 = (Area * (1j * rho0 * c0 * k) ./ (2 * pi * r1)) .* (9.3*exp(-1j * (k * r1 + phase1))) * u_n;
    p2 = (Area * (1j * rho0 * c0 * k) ./ (2 * pi * r2)) .* (9.3*exp(-1j * (k * r2 + phase2))) * u_n;

    %% 總聲壓
    p_total = p1 + p2;

    %% 聲壓平方的時間平均值
    p_squared_avg = abs(p_total.^2)./ 2;

    %% 聲壓梯度
    dp_dz = gradient(p_total, z);

    %% 質點速度 u
    u_z = -1 ./ (1j * omega * rho0) .* dp_dz;

    %% 質點速度平方的時間平均值
    u_squared_avg = abs(u_z.^2) ./ 2;

    %% Gor'kov 位能
    U = 2 * pi * R^3 * (1 / (3 * rho0 * c0^2) * p_squared_avg - rho0 / 2 * u_squared_avg);

    %% 阻尼
    damping_coefficient = 1.75e-6;  
    F_gravity = -weight * 9.81;   
    F_damping = -damping_coefficient * u_z; 
    F_radiation = -gradient(U, z);

    %% 總力
    F_z = F_radiation + F_gravity + F_damping;

    %% 畫圖 (雙 y 軸)
    yyaxis right
    plot(z*1e3, F_z, 'LineWidth', 2.0, ...  % 增加線寬
         'DisplayName', sprintf('F (Phase = %d°)', round(rad2deg(phase2))));

    yyaxis left
    plot(z*1e3, U, 'LineWidth', 2.0, ...    % 增加線寬
         'DisplayName', sprintf('U (Phase = %d°)', round(rad2deg(phase2))));
end

%% 力的需求線 
yyaxis right
yline(F_required, '--k', 'LineWidth', 2.5, 'DisplayName', 'F = 0');  % 增加線寬
ylabel('F [N]', 'FontSize', 25, 'FontWeight', 'bold');  % 加大字體
ylim([-0.1 0.1])

%% 能量軸
yyaxis left
ylabel('U [J]', 'FontSize', 25, 'FontWeight', 'bold');  % 加大字體
ylim([-5e-5 5e-5])

xlabel('Z Position [mm]', 'FontSize', 25, 'FontWeight', 'bold');  % 加大字體
xlim([24, 34]);
xticks(24:1:34);
title('Z Position vs U & F', 'FontSize', 25, 'FontWeight', 'bold');  % 加大標題字體

% 增加網格線粗細並調整圖例
grid on;
set(gca, 'GridLineStyle', '-', 'GridAlpha', 0.3, 'LineWidth', 1.5);
legend('Location', 'best', 'FontSize', 18);  % 增加圖例字體大小
