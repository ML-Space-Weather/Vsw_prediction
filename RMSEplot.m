clear 
clc
close all

load 'long_MF_5days_10_101000.mat';

num = 121;

for n = 1:num-1
    RMSE(n)=sqrt(mean((vr_5days(:, num - n) - vr_5days_pred(:, num - n)).^2));
    RMSE0(n)=sqrt(mean((vr(:, num - n) - vr_5days(:, num - n)).^2));
end

% plot(Time_sorted + hours(n), vr_sorted(:, n), 'DisplayName', 'vr')     
plot((1:num-1) / 24, RMSE, '-', 'DisplayName', 'RMSE\_pred');
hold on
plot((1:num-1) / 24, RMSE0, '-', 'DisplayName', 'RMSE\_persist')   
xlabel('days ahead');
ylabel('RMSE');
title('Plot of RMSE of vr\_5days and vr\_5days\_pred');
legend;
hold off

savefig('RMSE.fig')

% Save the plot as PNG
saveas(gcf, 'RMSE.png')