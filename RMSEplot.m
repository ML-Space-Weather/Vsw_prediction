clear 
clc
close all

load 'Results/long_MF_5days_10_101000.mat';
% load '/media/faraday/andong/GONG_NN/long_MF_5days_10_101000.mat';

% load '../long_MF_5days_10_10100_sigmoid.mat';

num = 121;

cali = vr_5days(:, num) - vr_5days_pred(:, num);

% RMSE_cali is to calibrate the start point of vr_5days_pred to vr_5days 
for n = 1:num-1
    RMSE(n) = sqrt(mean((vr_5days(:, num - n) - vr_5days_pred(:, num - n)).^2));
    RMSE_cali(n) = sqrt(mean((vr_5days(:, num - n) - vr_5days_pred(:, num - n) - cali).^2));
    RMSE0(n) = sqrt(mean((vr(:, num - n) - vr_5days(:, num - n)).^2));
    % Calculate Pearson correlation
    r(n) = corr(vr_5days(:, num - n), vr_5days_pred(:, num - n), 'Type', 'Pearson');
    r_cali(n) = corr(vr_5days(:, num - n), vr_5days_pred(:, num - n)+cali, 'Type', 'Pearson');
end

% Plot RMSE values on the left y-axis
figure;
yyaxis left
plot((1:num-1) / 24, RMSE, '-.', 'DisplayName', 'RMSE\_pred');
hold on;
plot((1:num-1) / 24, RMSE_cali, '-x', 'DisplayName', 'RMSE\_cali');
hold on;
plot((1:num-1) / 24, RMSE0, '-*', 'DisplayName', 'RMSE\_persist');
ylabel('RMSE');
xlabel('days ahead');
title('Plot of RMSE and Correlation of vr\_5days and vr\_5days\_pred');

% Plot r values on the right y-axis
yyaxis right
plot((1:num-1) / 24, r, '-o', 'DisplayName', 'Correlation');
ylabel('Correlation Coefficient (r)');
legend('RMSE\_pred', 'RMSE\_cali', 'RMSE\_persist', 'Correlation', 'Correlation_cali');

hold off;

% Save the plot as a .fig and .png
savefig('Figs/RMSE_and_Correlation.fig');
saveas(gcf, 'Figs/RMSE_and_Correlation.png');
