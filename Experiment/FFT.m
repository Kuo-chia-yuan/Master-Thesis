%% 1. 參數設定 (Parameter Setup)
clear; clc; close all;

% 取樣間隔 (Sampling Interval)
Ts = 2.5e-6; % 2.5 microseconds

% 計算取樣頻率 (Sampling Frequency)
Fs = 1 / Ts; % 單位：Hz

% 訊號點數 (Number of Samples)
N = 10;

%% 2. 產生訊號 (Signal Generation)
% =========================================================================
% ▼▼▼ 請在此處修改您要發送的訊號序列 (10個點，1代表高電位，0代表低電位) ▼▼▼
signal_bit = [1, 1, 1, 0, 0, 1, 1, 0, 0, 0];
% ▲▲▲ 請在此處修改您要發送的訊號序列 ▲▲▲
% =========================================================================

% 建立時間軸 (Time Vector)
% 從 0 開始，共 N 個點
t = (0:N-1) * Ts; 

% 您的週期性訊號
y = signal_bit;

%% 3. 執行傅立葉轉換 (Perform FFT)

% 執行 FFT
Y = fft(y);

% FFT 的結果是複數，我們需要計算其振幅
% 為了得到真實的振幅，需要除以點數 N
P2 = abs(Y / N);

% 由於頻譜是對稱的，我們只需要看前半部分 (從直流到奈奎斯特頻率)
% 取前 N/2+1 個點
P1 = P2(1:N/2+1);

% 單邊頻譜的振幅需要乘以 2 (直流成分和奈奎斯特頻率除外)
P1(2:end-1) = 2 * P1(2:end-1);

% 計算總能量 (Parseval's theorem)
total_energy = sum(P1.^2);

% 計算各頻率成分的能量占比 (%)
energy_ratio = (P1.^2 / total_energy) * 100;

% 建立對應的頻率軸
% 頻率解析度為 Fs/N
f = Fs * (0:(N/2)) / N;

%% 4. 找出主頻率 (Find Dominant Frequency)

% 找出能量占比最大值的位置 (不包含直流成分，即第一個點)
[max_ratio, index] = max(energy_ratio(2:end));

% 透過索引找到對應的頻率
% 因為我們是從第二個元素開始找，所以索引要加 1
dominant_frequency = f(index + 1);

% 在命令視窗顯示結果
fprintf('========== 頻譜分析結果 ==========\n');
fprintf('訊號的主要頻率成分在: %.2f kHz\n', dominant_frequency / 1000);
fprintf('該頻率的能量占比為: %.2f %%\n', max_ratio);
fprintf('該頻率的振幅為: %.4f\n', P1(index + 1));
fprintf('\n各頻率成分的能量分布:\n');
for i = 1:length(f)
    if energy_ratio(i) > 0.01 % 只顯示占比大於 0.01% 的成分
        fprintf('  %.2f kHz: %.2f %% (振幅 = %.4f)\n', f(i)/1000, energy_ratio(i), P1(i));
    end
end
fprintf('===================================\n');

%% 5. 繪圖 (Plotting)

% 建立一個圖形視窗
figure('Position', [100, 100, 800, 600]);

% --- 繪製時間域訊號 ---
subplot(2, 1, 1); % 將圖形視窗分割成 2x1，並使用第 1 個
stem(t * 1e6, y, 'filled', 'LineWidth', 1.5); % 使用 stem 函數繪製離散訊號，時間軸單位轉為 us
title('Time Domain Signal', 'FontSize', 12, 'FontWeight', 'bold');
xlabel('Time [\mus]', 'FontSize', 11);
ylabel('Potential', 'FontSize', 11);
grid on;
ylim([-0.2, 1.2]); % 設定 Y 軸範圍，讓圖形更清楚
yticks([0, 1]);
yticklabels({'LOW', 'HIGH'});

% --- 繪製頻譜圖 (能量占比) ---
subplot(2, 1, 2); % 使用第 2 個

hold on;
plot(f / 1000, energy_ratio, '-o', 'Color', [0.8, 0.3, 0.3], 'LineWidth', 1.5, 'MarkerSize', 6); % 疊加折線圖
title('Frequency Domain - Energy Distribution', 'FontSize', 12, 'FontWeight', 'bold');
xlabel('Frequency [kHz]', 'FontSize', 11);
ylabel('Energy Ratio [%]', 'FontSize', 11);
grid on;

% 動態調整縱軸範圍
max_y = max(energy_ratio) * 1.1;
ylim([0, max_y]);

% 在 40 kHz 處標記紅色星號
freq_mark = 40; % kHz
% 找到最接近 40 kHz 的頻率索引
[~, idx_40k] = min(abs(f/1000 - freq_mark));
plot(f(idx_40k) / 1000, energy_ratio(idx_40k), 'r*', 'MarkerSize', 15, 'LineWidth', 2);

% 在 40 kHz 處標註數值
%text(f(idx_40k) / 1000, energy_ratio(idx_40k) + max_y*0.2, ...
 %   sprintf('%.2f%%', energy_ratio(idx_40k)), ...
  %  'HorizontalAlignment', 'center', 'FontSize', 10, 'FontWeight', 'bold', 'Color', 'r');

hold off;