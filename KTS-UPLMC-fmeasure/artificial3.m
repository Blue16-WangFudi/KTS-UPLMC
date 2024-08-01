%人工数据集三维图
%%
close all;
clear all;
%% 
clc
rand('state', 2015)
randn('state', 2015)
i=0;
disp('imbalance=2');
%% 
for ii = 10:10:30
    i=i+1;
    j=0;
    for jj = 10:10:30
        fprintf(' \n[ii=%1.0f,jj=%1.0f] \n',ii,jj);
        j=j+1;
        % 第一组
        mul=[0.5,-3.5]; % 均值
        S1=[0.2 0;0 0.3]; % 协方差
        data1=mvnrnd(mul, S1, 100); % 产生高斯分布数据
        data1(:,3)=-1;
        % 第二组数据
        mu2=[-0.5,3.5];
        S2=[0.2 0;0 0.3];
        data2=mvnrnd(mu2,S2,50);
        data2(:,3)=1;
        % noises of n
        mm1=ii;
        mu3=[0,0];
        S3=[1 -0.8;-0.8 1];
        data3=mvnrnd(mu3,S3,mm1);
        data3(:,3)=1;
        % noises of p
        mm2=jj;
        mu4=[0,0];
        S4=[1 -0.8;-0.8 1];
        data4=mvnrnd(mu4,S4,mm2);
        data4(:,3)=-1;
        data_noise = [data3;data4];

        
        %% all
        Data_Original=[data1;data2;data_noise];
        TrainRate = 0.5;       % The scale of the train set  
        [Data_Train, Data_Predict] = Data_Rate(Data_Original, TrainRate);   % Chose 3       
        Samples_Train = Data_Train(:, 1:end-1);
        Labels_Train = Data_Train(:, end);
        Ctest=Data_Predict(:, 1:end-1);
        dtest=Data_Predict(:, end);

        
        %% Parameter setting
        clear C;
        kernel=2;
        p1= 2^-2;
        lamb1 = [2^-7,2^-6,2^-5,2^-4,2^-3,2^-2,2^-1,2^0,2^1];
        lamb2 = [2^-7,2^-6,2^-5,2^-4,2^-3,2^-2,2^-1,2^0,2^1];
        c1val=[2^-6,2^-5,2^-4,2^-3,2^-2,2^-1,2^0,2^1,2^2,2^3];
            
        %%
        [opt_tau1,opt_tau2,opt_tau4,...
        acc_svm,acc_upsvm,acc_psvm,acc_ldm,acc_upldm,...
        time0,time1,time2,time3,time4,...
        opt_C0,opt_C1,opt_C2,opt_C3,opt_C4,...
        AUC0,Sensitivity0,Specificity0,Fmeasure0,Gmeans0,...
        AUC1,Sensitivity1,Specificity1,Fmeasure1,Gmeans1,...
        AUC2,Sensitivity2,Specificity2,Fmeasure2,Gmeans2,...
        AUC3,Sensitivity3,Specificity3,Fmeasure3,Gmeans3,...
        AUC4,Sensitivity4,Specificity4,Fmeasure4,Gmeans4] = tune_tau(Samples_Train,Labels_Train,Ctest,dtest,c1val,kernel,p1,lamb1,lamb2);

        %% SVM
        fprintf(['SVM Accuracy=%3.2f,AUC=%3.2f,Sensitivity=%3.2f,Specificity=%3.2f,Fmeasure=%3.2f,Gmeans=%3.2f,' ...
            'time = %3.2f,C = %3.2f,p1=%3.2f'], acc_svm, AUC0,Sensitivity0,Specificity0,Fmeasure0,Gmeans0,time0,opt_C0,p1);
        
        %% UPSVM
        fprintf(['\n UPSVM Accuracy=%3.2f,AUC=%3.2f,Sensitivity=%3.2f,Specificity=%3.2f,Fmeasure=%3.2f,Gmeans=%3.2f,' ...
            'time = %3.2f,tau = %3.2f,C = %3.2f,p1=%3.2f'],acc_upsvm,AUC1,Sensitivity1,Specificity1,Fmeasure1,Gmeans1,time1,opt_tau1,opt_C1,p1);
        
        %% PSVM
        fprintf(['\n PSVM Accuracy=%3.2f,,AUC=%3.2f,Sensitivity=%3.2f,Specificity=%3.2f,Fmeasure=%3.2f,Gmeans=%3.2f,' ...
            'time = %3.2f,tau = %3.2f,C = %3.2f,p1=%3.2f'],acc_psvm,AUC2,Sensitivity2,Specificity2,Fmeasure2,Gmeans2,time2,opt_tau2,opt_C2,p1);
        
        %% LDM
        fprintf(['\n LDM Accuracy=%3.2f,AUC=%3.2f,Sensitivity=%3.2f,Specificity=%3.2f,Fmeasure=%3.2f,Gmeans=%3.2f,' ...
            'time = %3.2f,C = %3.2f,p1=%3.2f'],acc_ldm,AUC3,Sensitivity3,Specificity3,Fmeasure3,Gmeans3,time3,opt_C3,p1);
        
        %% UPLDM
        fprintf(['\n UPLDM Accuracy=%3.2f,AUC=%3.2f,Sensitivity=%3.2f,Specificity=%3.2f,Fmeasure=%3.2f,Gmeans=%3.2f,' ...
            'time = %3.2f,tau = %3.2f,C = %3.2f,p1=%3.2f\n'],acc_upldm,AUC4,Sensitivity4,Specificity4,Fmeasure4,Gmeans4,time4,opt_tau4,opt_C4,p1);
        %%
        avg1(i,j) = (AUC0+Fmeasure0+Gmeans0)/3;
        avg2(i,j) = (AUC1+Fmeasure1+Gmeans1)/3;
        avg3(i,j) = (AUC2+Fmeasure2+Gmeans2)/3;
        avg4(i,j) = (AUC3+Fmeasure3+Gmeans3)/3;
        avg5(i,j) = (AUC4+Fmeasure4+Gmeans4)/3;
    end
end

figure
surf([10 20 30],[10 20 30],avg1);
xlabel('m_p'),ylabel('m_n'),zlabel('AVG');
title('SVM-2');
shading interp;
zlim([50 100])
colormap(autumn);

figure
surf([10 20 30],[10 20 30],avg2);
xlabel('m_p'),ylabel('m_n'),zlabel('AVG');
title('UPSVM-2');
shading interp;
zlim([50 100])
colormap(autumn);

figure
surf([10 20 30],[10 20 30],avg3);
xlabel('m_p'),ylabel('m_n'),zlabel('AVG');
title('PSVM-2');
shading interp;
zlim([50 100])
colormap(autumn);


figure
surf([10 20 30],[10 20 30],avg4);
xlabel('m_p'),ylabel('m_n'),zlabel('AVG');
title('LDM-2');
shading interp;
zlim([50 100])
colormap(autumn);


figure
surf([10 20 30],[10 20 30],avg5);
xlabel('m_p'),ylabel('m_n'),zlabel('AVG');
title('UPLDM-2');
shading interp;
zlim([50 100])
colormap(autumn);





i=0;
disp('imbalance=5');
%% 
for ii = 10:10:30
    i=i+1;
    j=0;
    for jj = 10:10:30
        fprintf(' \n[ii=%1.0f,jj=%1.0f] \n',ii,jj);
        j=j+1;
        % 第一组
        mul=[0.5,-3.5]; % 均值
        S1=[0.2 0;0 0.3]; % 协方差
        data1=mvnrnd(mul, S1, 250); % 产生高斯分布数据
        data1(:,3)=-1;
        % 第二组数据
        mu2=[-0.5,3.5];
        S2=[0.2 0;0 0.3];
        data2=mvnrnd(mu2,S2,50);
        data2(:,3)=1;
        % noises of n
        mm1=ii;
        mu3=[0,0];
        S3=[1 -0.8;-0.8 1];
        data3=mvnrnd(mu3,S3,mm1);
        data3(:,3)=1;
        % noises of p
        mm2=jj;
        mu4=[0,0];
        S4=[1 -0.8;-0.8 1];
        data4=mvnrnd(mu4,S4,mm2);
        data4(:,3)=-1;
        data_noise = [data3;data4];

        
        %% all
        Data_Original=[data1;data2;data_noise];
        TrainRate = 0.5;       % The scale of the train set  
        [Data_Train, Data_Predict] = Data_Rate(Data_Original, TrainRate);   % Chose 3       
        Samples_Train = Data_Train(:, 1:end-1);
        Labels_Train = Data_Train(:, end);
        Ctest=Data_Predict(:, 1:end-1);
        dtest=Data_Predict(:, end);

        
        %% Parameter setting
        clear C;
        kernel=2;
        p1= 2^-2;
        lamb1 = [2^-7,2^-6,2^-5,2^-4,2^-3,2^-2,2^-1,2^0,2^1];
        lamb2 = [2^-7,2^-6,2^-5,2^-4,2^-3,2^-2,2^-1,2^0,2^1];
        c1val=[2^-6,2^-5,2^-4,2^-3,2^-2,2^-1,2^0,2^1,2^2,2^3];
            
        %%
        [opt_tau1,opt_tau2,opt_tau4,...
        acc_svm,acc_upsvm,acc_psvm,acc_ldm,acc_upldm,...
        time0,time1,time2,time3,time4,...
        opt_C0,opt_C1,opt_C2,opt_C3,opt_C4,...
        AUC0,Sensitivity0,Specificity0,Fmeasure0,Gmeans0,...
        AUC1,Sensitivity1,Specificity1,Fmeasure1,Gmeans1,...
        AUC2,Sensitivity2,Specificity2,Fmeasure2,Gmeans2,...
        AUC3,Sensitivity3,Specificity3,Fmeasure3,Gmeans3,...
        AUC4,Sensitivity4,Specificity4,Fmeasure4,Gmeans4] = tune_tau(Samples_Train,Labels_Train,Ctest,dtest,c1val,kernel,p1,lamb1,lamb2);

        %% SVM
        fprintf(['SVM Accuracy=%3.2f,AUC=%3.2f,Sensitivity=%3.2f,Specificity=%3.2f,Fmeasure=%3.2f,Gmeans=%3.2f,' ...
            'time = %3.2f,C = %3.2f,p1=%3.2f'], acc_svm, AUC0,Sensitivity0,Specificity0,Fmeasure0,Gmeans0,time0,opt_C0,p1);
        
        %% UPSVM
        fprintf(['\n UPSVM Accuracy=%3.2f,AUC=%3.2f,Sensitivity=%3.2f,Specificity=%3.2f,Fmeasure=%3.2f,Gmeans=%3.2f,' ...
            'time = %3.2f,tau = %3.2f,C = %3.2f,p1=%3.2f'],acc_upsvm,AUC1,Sensitivity1,Specificity1,Fmeasure1,Gmeans1,time1,opt_tau1,opt_C1,p1);
        
        %% PSVM
        fprintf(['\n PSVM Accuracy=%3.2f,,AUC=%3.2f,Sensitivity=%3.2f,Specificity=%3.2f,Fmeasure=%3.2f,Gmeans=%3.2f,' ...
            'time = %3.2f,tau = %3.2f,C = %3.2f,p1=%3.2f'],acc_psvm,AUC2,Sensitivity2,Specificity2,Fmeasure2,Gmeans2,time2,opt_tau2,opt_C2,p1);
        
        %% LDM
        fprintf(['\n LDM Accuracy=%3.2f,AUC=%3.2f,Sensitivity=%3.2f,Specificity=%3.2f,Fmeasure=%3.2f,Gmeans=%3.2f,' ...
            'time = %3.2f,C = %3.2f,p1=%3.2f'],acc_ldm,AUC3,Sensitivity3,Specificity3,Fmeasure3,Gmeans3,time3,opt_C3,p1);
        
        %% UPLDM
        fprintf(['\n UPLDM Accuracy=%3.2f,AUC=%3.2f,Sensitivity=%3.2f,Specificity=%3.2f,Fmeasure=%3.2f,Gmeans=%3.2f,' ...
            'time = %3.2f,tau = %3.2f,C = %3.2f,p1=%3.2f\n'],acc_upldm,AUC4,Sensitivity4,Specificity4,Fmeasure4,Gmeans4,time4,opt_tau4,opt_C4,p1);
        %%
        avg1(i,j) = (AUC0+Fmeasure0+Gmeans0)/3;
        avg2(i,j) = (AUC1+Fmeasure1+Gmeans1)/3;
        avg3(i,j) = (AUC2+Fmeasure2+Gmeans2)/3;
        avg4(i,j) = (AUC3+Fmeasure3+Gmeans3)/3;
        avg5(i,j) = (AUC4+Fmeasure4+Gmeans4)/3;
    end
end

figure
surf([10 20 30],[10 20 30],avg1);
xlabel('m_p'),ylabel('m_n'),zlabel('AVG');
title('SVM-5');
shading interp;
zlim([50 100])
colormap(autumn);

figure
surf([10 20 30],[10 20 30],avg2);
xlabel('m_p'),ylabel('m_n'),zlabel('AVG');
title('UPSVM-5');
shading interp;
zlim([50 100])
colormap(autumn);

figure
surf([10 20 30],[10 20 30],avg3);
xlabel('m_p'),ylabel('m_n'),zlabel('AVG');
title('PSVM-5');
shading interp;
zlim([50 100])
colormap(autumn);


figure
surf([10 20 30],[10 20 30],avg4);
xlabel('m_p'),ylabel('m_n'),zlabel('AVG');
title('LDM-5');
shading interp;
zlim([50 100])
colormap(autumn);


figure
surf([10 20 30],[10 20 30],avg5);
xlabel('m_p'),ylabel('m_n'),zlabel('AVG');
title('UPLDM-5');
shading interp;
zlim([50 100])
colormap(autumn);






i=0;
disp('imbalance=10');
%% 
for ii = 10:10:30
    i=i+1;
    j=0;
    for jj = 10:10:30
        fprintf(' \n[ii=%1.0f,jj=%1.0f] \n',ii,jj);
        j=j+1;
        % 第一组
        mul=[0.5,-3.5]; % 均值
        S1=[0.2 0;0 0.3]; % 协方差
        data1=mvnrnd(mul, S1, 500); % 产生高斯分布数据
        data1(:,3)=-1;
        % 第二组数据
        mu2=[-0.5,3.5];
        S2=[0.2 0;0 0.3];
        data2=mvnrnd(mu2,S2,50);
        data2(:,3)=1;
        % noises of n
        mm1=ii;
        mu3=[0,0];
        S3=[1 -0.8;-0.8 1];
        data3=mvnrnd(mu3,S3,mm1);
        data3(:,3)=1;
        % noises of p
        mm2=jj;
        mu4=[0,0];
        S4=[1 -0.8;-0.8 1];
        data4=mvnrnd(mu4,S4,mm2);
        data4(:,3)=-1;
        data_noise = [data3;data4];

        
        %% all
        Data_Original=[data1;data2;data_noise];
        TrainRate = 0.5;       % The scale of the train set  
        [Data_Train, Data_Predict] = Data_Rate(Data_Original, TrainRate);   % Chose 3       
        Samples_Train = Data_Train(:, 1:end-1);
        Labels_Train = Data_Train(:, end);
        Ctest=Data_Predict(:, 1:end-1);
        dtest=Data_Predict(:, end);

        
        %% Parameter setting
        clear C;
        kernel=2;
        p1= 2^-2;
        lamb1 = [2^-7,2^-6,2^-5,2^-4,2^-3,2^-2,2^-1,2^0,2^1];
        lamb2 = [2^-7,2^-6,2^-5,2^-4,2^-3,2^-2,2^-1,2^0,2^1];
        c1val=[2^-6,2^-5,2^-4,2^-3,2^-2,2^-1,2^0,2^1,2^2,2^3];
            
        %%
        [opt_tau1,opt_tau2,opt_tau4,...
        acc_svm,acc_upsvm,acc_psvm,acc_ldm,acc_upldm,...
        time0,time1,time2,time3,time4,...
        opt_C0,opt_C1,opt_C2,opt_C3,opt_C4,...
        AUC0,Sensitivity0,Specificity0,Fmeasure0,Gmeans0,...
        AUC1,Sensitivity1,Specificity1,Fmeasure1,Gmeans1,...
        AUC2,Sensitivity2,Specificity2,Fmeasure2,Gmeans2,...
        AUC3,Sensitivity3,Specificity3,Fmeasure3,Gmeans3,...
        AUC4,Sensitivity4,Specificity4,Fmeasure4,Gmeans4] = tune_tau(Samples_Train,Labels_Train,Ctest,dtest,c1val,kernel,p1,lamb1,lamb2);

        %% SVM
        fprintf(['SVM Accuracy=%3.2f,AUC=%3.2f,Sensitivity=%3.2f,Specificity=%3.2f,Fmeasure=%3.2f,Gmeans=%3.2f,' ...
            'time = %3.2f,C = %3.2f,p1=%3.2f'], acc_svm, AUC0,Sensitivity0,Specificity0,Fmeasure0,Gmeans0,time0,opt_C0,p1);
        
        %% UPSVM
        fprintf(['\n UPSVM Accuracy=%3.2f,AUC=%3.2f,Sensitivity=%3.2f,Specificity=%3.2f,Fmeasure=%3.2f,Gmeans=%3.2f,' ...
            'time = %3.2f,tau = %3.2f,C = %3.2f,p1=%3.2f'],acc_upsvm,AUC1,Sensitivity1,Specificity1,Fmeasure1,Gmeans1,time1,opt_tau1,opt_C1,p1);
        
        %% PSVM
        fprintf(['\n PSVM Accuracy=%3.2f,,AUC=%3.2f,Sensitivity=%3.2f,Specificity=%3.2f,Fmeasure=%3.2f,Gmeans=%3.2f,' ...
            'time = %3.2f,tau = %3.2f,C = %3.2f,p1=%3.2f'],acc_psvm,AUC2,Sensitivity2,Specificity2,Fmeasure2,Gmeans2,time2,opt_tau2,opt_C2,p1);
        
        %% LDM
        fprintf(['\n LDM Accuracy=%3.2f,AUC=%3.2f,Sensitivity=%3.2f,Specificity=%3.2f,Fmeasure=%3.2f,Gmeans=%3.2f,' ...
            'time = %3.2f,C = %3.2f,p1=%3.2f'],acc_ldm,AUC3,Sensitivity3,Specificity3,Fmeasure3,Gmeans3,time3,opt_C3,p1);
        
        %% UPLDM
        fprintf(['\n UPLDM Accuracy=%3.2f,AUC=%3.2f,Sensitivity=%3.2f,Specificity=%3.2f,Fmeasure=%3.2f,Gmeans=%3.2f,' ...
            'time = %3.2f,tau = %3.2f,C = %3.2f,p1=%3.2f\n'],acc_upldm,AUC4,Sensitivity4,Specificity4,Fmeasure4,Gmeans4,time4,opt_tau4,opt_C4,p1);
        %%
        avg1(i,j) = (AUC0+Fmeasure0+Gmeans0)/3;
        avg2(i,j) = (AUC1+Fmeasure1+Gmeans1)/3;
        avg3(i,j) = (AUC2+Fmeasure2+Gmeans2)/3;
        avg4(i,j) = (AUC3+Fmeasure3+Gmeans3)/3;
        avg5(i,j) = (AUC4+Fmeasure4+Gmeans4)/3;
    end
end

figure
surf([10 20 30],[10 20 30],avg1);
xlabel('m_p'),ylabel('m_n'),zlabel('AVG');
title('SVM-10');
shading interp;
zlim([50 100])
colormap(autumn);

figure
surf([10 20 30],[10 20 30],avg2);
xlabel('m_p'),ylabel('m_n'),zlabel('AVG');
title('UPSVM-10');
shading interp;
zlim([50 100])
colormap(autumn);

figure
surf([10 20 30],[10 20 30],avg3);
xlabel('m_p'),ylabel('m_n'),zlabel('AVG');
title('PSVM-10');
shading interp;
zlim([50 100])
colormap(autumn);


figure
surf([10 20 30],[10 20 30],avg4);
xlabel('m_p'),ylabel('m_n'),zlabel('AVG');
title('LDM-10');
shading interp;
zlim([50 100])
colormap(autumn);


figure
surf([10 20 30],[10 20 30],avg5);
xlabel('m_p'),ylabel('m_n'),zlabel('AVG');
title('UPLDM-10');
shading interp;
zlim([50 100])
colormap(autumn);







i=0;
disp('imbalance=20');
%% 
for ii = 10:10:30
    i=i+1;
    j=0;
    for jj = 10:10:30
        fprintf(' \n[ii=%1.0f,jj=%1.0f] \n',ii,jj);
        j=j+1;
        % 第一组
        mul=[0.5,-3.5]; % 均值
        S1=[0.2 0;0 0.3]; % 协方差
        data1=mvnrnd(mul, S1, 1000); % 产生高斯分布数据
        data1(:,3)=-1;
        % 第二组数据
        mu2=[-0.5,3.5];
        S2=[0.2 0;0 0.3];
        data2=mvnrnd(mu2,S2,50);
        data2(:,3)=1;
        % noises of n
        mm1=ii;
        mu3=[0,0];
        S3=[1 -0.8;-0.8 1];
        data3=mvnrnd(mu3,S3,mm1);
        data3(:,3)=1;
        % noises of p
        mm2=jj;
        mu4=[0,0];
        S4=[1 -0.8;-0.8 1];
        data4=mvnrnd(mu4,S4,mm2);
        data4(:,3)=-1;
        data_noise = [data3;data4];

        
        %% all
        Data_Original=[data1;data2;data_noise];
        TrainRate = 0.5;       % The scale of the train set  
        [Data_Train, Data_Predict] = Data_Rate(Data_Original, TrainRate);   % Chose 3       
        Samples_Train = Data_Train(:, 1:end-1);
        Labels_Train = Data_Train(:, end);
        Ctest=Data_Predict(:, 1:end-1);
        dtest=Data_Predict(:, end);

        
        %% Parameter setting
        clear C;
        kernel=2;
        p1= 2^-2;
        lamb1 = [2^-7,2^-6,2^-5,2^-4,2^-3,2^-2,2^-1,2^0,2^1];
        lamb2 = [2^-7,2^-6,2^-5,2^-4,2^-3,2^-2,2^-1,2^0,2^1];
        c1val=[2^-6,2^-5,2^-4,2^-3,2^-2,2^-1,2^0,2^1,2^2,2^3];
            
        %%
        [opt_tau1,opt_tau2,opt_tau4,...
        acc_svm,acc_upsvm,acc_psvm,acc_ldm,acc_upldm,...
        time0,time1,time2,time3,time4,...
        opt_C0,opt_C1,opt_C2,opt_C3,opt_C4,...
        AUC0,Sensitivity0,Specificity0,Fmeasure0,Gmeans0,...
        AUC1,Sensitivity1,Specificity1,Fmeasure1,Gmeans1,...
        AUC2,Sensitivity2,Specificity2,Fmeasure2,Gmeans2,...
        AUC3,Sensitivity3,Specificity3,Fmeasure3,Gmeans3,...
        AUC4,Sensitivity4,Specificity4,Fmeasure4,Gmeans4] = tune_tau(Samples_Train,Labels_Train,Ctest,dtest,c1val,kernel,p1,lamb1,lamb2);

        %% SVM
        fprintf(['SVM Accuracy=%3.2f,AUC=%3.2f,Sensitivity=%3.2f,Specificity=%3.2f,Fmeasure=%3.2f,Gmeans=%3.2f,' ...
            'time = %3.2f,C = %3.2f,p1=%3.2f'], acc_svm, AUC0,Sensitivity0,Specificity0,Fmeasure0,Gmeans0,time0,opt_C0,p1);
        
        %% UPSVM
        fprintf(['\n UPSVM Accuracy=%3.2f,AUC=%3.2f,Sensitivity=%3.2f,Specificity=%3.2f,Fmeasure=%3.2f,Gmeans=%3.2f,' ...
            'time = %3.2f,tau = %3.2f,C = %3.2f,p1=%3.2f'],acc_upsvm,AUC1,Sensitivity1,Specificity1,Fmeasure1,Gmeans1,time1,opt_tau1,opt_C1,p1);
        
        %% PSVM
        fprintf(['\n PSVM Accuracy=%3.2f,,AUC=%3.2f,Sensitivity=%3.2f,Specificity=%3.2f,Fmeasure=%3.2f,Gmeans=%3.2f,' ...
            'time = %3.2f,tau = %3.2f,C = %3.2f,p1=%3.2f'],acc_psvm,AUC2,Sensitivity2,Specificity2,Fmeasure2,Gmeans2,time2,opt_tau2,opt_C2,p1);
        
        %% LDM
        fprintf(['\n LDM Accuracy=%3.2f,AUC=%3.2f,Sensitivity=%3.2f,Specificity=%3.2f,Fmeasure=%3.2f,Gmeans=%3.2f,' ...
            'time = %3.2f,C = %3.2f,p1=%3.2f'],acc_ldm,AUC3,Sensitivity3,Specificity3,Fmeasure3,Gmeans3,time3,opt_C3,p1);
        
        %% UPLDM
        fprintf(['\n UPLDM Accuracy=%3.2f,AUC=%3.2f,Sensitivity=%3.2f,Specificity=%3.2f,Fmeasure=%3.2f,Gmeans=%3.2f,' ...
            'time = %3.2f,tau = %3.2f,C = %3.2f,p1=%3.2f\n'],acc_upldm,AUC4,Sensitivity4,Specificity4,Fmeasure4,Gmeans4,time4,opt_tau4,opt_C4,p1);
        %%
        avg1(i,j) = (AUC0+Fmeasure0+Gmeans0)/3;
        avg2(i,j) = (AUC1+Fmeasure1+Gmeans1)/3;
        avg3(i,j) = (AUC2+Fmeasure2+Gmeans2)/3;
        avg4(i,j) = (AUC3+Fmeasure3+Gmeans3)/3;
        avg5(i,j) = (AUC4+Fmeasure4+Gmeans4)/3;
    end
end

figure
surf([10 20 30],[10 20 30],avg1);
xlabel('m_p'),ylabel('m_n'),zlabel('AVG');
title('SVM-20');
shading interp;
zlim([50 100])
colormap(autumn);

figure
surf([10 20 30],[10 20 30],avg2);
xlabel('m_p'),ylabel('m_n'),zlabel('AVG');
title('UPSVM-20');
shading interp;
zlim([50 100])
colormap(autumn);

figure
surf([10 20 30],[10 20 30],avg3);
xlabel('m_p'),ylabel('m_n'),zlabel('AVG');
title('PSVM-20');
shading interp;
zlim([50 100])
colormap(autumn);


figure
surf([10 20 30],[10 20 30],avg4);
xlabel('m_p'),ylabel('m_n'),zlabel('AVG');
title('LDM-20');
shading interp;
zlim([50 100])
colormap(autumn);


figure
surf([10 20 30],[10 20 30],avg5);
xlabel('m_p'),ylabel('m_n'),zlabel('AVG');
title('UPLDM-20');
shading interp;
zlim([50 100])
colormap(autumn);
