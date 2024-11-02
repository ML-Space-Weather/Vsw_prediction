clear 
clc
close all

load 'Results/long_MF_5days_10_101000.mat';

n = 96;
% Convert raw Time matrix to datetime format
Time = datetime(Time(:,1), Time(:,2), Time(:,3), Time(:,4), Time(:,5), 0); % Assuming last column is minutes

% Plot the data
figure;

% plot(Time + hours(120-115), vr_5days(:, 115), '.-', 'DisplayName', 'vr\_5days\_5')   
% hold on;
plot(Time + hours(121 - n), vr_5days(:, n), '.-', 'DisplayName', 'vr\_5days')     
hold on;
plot(Time + hours(121 - n), vr_5days_pred(:, n), 'DisplayName', 'vr\_5days\_pred')
xlabel('Time');
ylabel('Values');
% title(sprintf('Plot of vr\\_5days with different hrs ahead', 121 - n));
title(sprintf('Plot of vr\\_5days and vr\\_5days\\_pred %d hrs ahead', 121 - n));

legend;
hold off

savefig('Figs/vsw_vs_pred.fig')

% Save the plot as PNG
saveas(gcf, 'Figs/test.png')
