load('5-avg1.mat');
load('5-avg4.mat');
load('5-avg5.mat');
figure
surf([0 5 10],[0 5 10],avg1);
xlabel('m_p'),ylabel('m_n'),zlabel('AVG');
title('SVM-5');
shading interp;
zlim([50 100])
colormap(autumn);

figure
surf([0 5 10],[0 5 10],avg4);
xlabel('m_p'),ylabel('m_n'),zlabel('AVG');
title('LDM-5');
shading interp;
zlim([50 100])
colormap(autumn);

figure
surf([0 5 10],[0 5 10],avg5);
xlabel('m_p'),ylabel('m_n'),zlabel('AVG');
title('UPLDM-5');
shading interp;
zlim([50 100])
colormap(autumn);