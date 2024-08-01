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
for ii = 10:20:30
    i=i+1;
    j=0;
    for jj = 10:20:30
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
        mu3=[0,3];
        S3=[1 -0.8;-0.8 1];
        data3=mvnrnd(mu3,S3,mm1);
        data3(:,3)=1;
        % noises of p
        mm2=jj;
        mu4=[0,-3];
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
        [opt_tau4,...
        acc_ldm,acc_upldm,...
        time3,time4,...
        opt_C3,opt_C4,...
        AUC3,Sensitivity3,Specificity3,Fmeasure3,Gmeans3,...
        AUC4,Sensitivity4,Specificity4,Fmeasure4,Gmeans4]= tune_tau1(Samples_Train,Labels_Train,Ctest,dtest,c1val,kernel,p1,lamb1,lamb2);
              
        %% LDM
        fprintf(['\n LDM AVG=%3.2f,Accuracy=%3.2f,AUC=%3.2f,Sensitivity=%3.2f,Specificity=%3.2f,Fmeasure=%3.2f,Gmeans=%3.2f,' ...
            'time = %3.2f,C = %3.2f,p1=%3.2f'],(AUC3+Fmeasure3+Gmeans3)/3,acc_ldm,AUC3,Sensitivity3,Specificity3,Fmeasure3,Gmeans3,time3,opt_C3,p1);
        
        %% UPLDM
        fprintf(['\n UPLDM AVG=%3.2f,Accuracy=%3.2f,AUC=%3.2f,Sensitivity=%3.2f,Specificity=%3.2f,Fmeasure=%3.2f,Gmeans=%3.2f,' ...
            'time = %3.2f,tau = %3.2f,C = %3.2f,p1=%3.2f\n'],(AUC4+Fmeasure4+Gmeans4)/3,acc_upldm,AUC4,Sensitivity4,Specificity4,Fmeasure4,Gmeans4,time4,opt_tau4,opt_C4,p1);
        %%
        avg4(i,j) = (AUC3+Fmeasure3+Gmeans3)/3;
        avg5(i,j) = (AUC4+Fmeasure4+Gmeans4)/3;
    end
end

figure
surf([0 5 10],[0 5 10],avg4);
xlabel('m_p'),ylabel('m_n'),zlabel('AVG');
title('LDM');
shading interp;
zlim([50 100])
colormap(autumn);


figure
surf([0 5 10],[0 5 10],avg5);
xlabel('m_p'),ylabel('m_n'),zlabel('AVG');
title('UPLDM');
shading interp;
zlim([50 100])
colormap(autumn);





i=0;
disp('imbalance=5');
for ii = 0:5:10
    i=i+1;
    j=0;
    for jj = 0:5:10
        fprintf(' \n[ii=%1.0f,jj=%1.0f] \n',ii,jj);
        j=j+1;
        % 第一组
        mul=[0.5,-3.5]; % 均值
        S1=[1.5 0;0 1.5]; % 协方差
        data1=mvnrnd(mul, S1, 417); % 产生高斯分布数据
        data1(:,3)=-1;
        % 第二组数据
        mu2=[-0.5,3.5];
        S2=[1.5 0;0 1.5];
        data2=mvnrnd(mu2,S2,83);
        data2(:,3)=1;
        % noises of n
        mm1=ii;
        mu3=[0,3];
        S3=[1 -0.8;-0.8 1];
        data3=mvnrnd(mu3,S3,mm1);
        data3(:,3)=1;
        % noises of p
        mm2=jj;
        mu4=[0,-3];
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
        [opt_tau4,...
        acc_ldm,acc_upldm,...
        time3,time4,...
        opt_C3,opt_C4,...
        AUC3,Sensitivity3,Specificity3,Fmeasure3,Gmeans3,...
        AUC4,Sensitivity4,Specificity4,Fmeasure4,Gmeans4]= tune_tau1(Samples_Train,Labels_Train,Ctest,dtest,c1val,kernel,p1,lamb1,lamb2);
              
        %% LDM
        fprintf(['\n LDM AVG=%3.2f,Accuracy=%3.2f,AUC=%3.2f,Sensitivity=%3.2f,Specificity=%3.2f,Fmeasure=%3.2f,Gmeans=%3.2f,' ...
            'time = %3.2f,C = %3.2f,p1=%3.2f'],(AUC3+Fmeasure3+Gmeans3)/3,acc_ldm,AUC3,Sensitivity3,Specificity3,Fmeasure3,Gmeans3,time3,opt_C3,p1);
        
        %% UPLDM
        fprintf(['\n UPLDM AVG=%3.2f,Accuracy=%3.2f,AUC=%3.2f,Sensitivity=%3.2f,Specificity=%3.2f,Fmeasure=%3.2f,Gmeans=%3.2f,' ...
            'time = %3.2f,tau = %3.2f,C = %3.2f,p1=%3.2f\n'],(AUC4+Fmeasure4+Gmeans4)/3,acc_upldm,AUC4,Sensitivity4,Specificity4,Fmeasure4,Gmeans4,time4,opt_tau4,opt_C4,p1);
        %%
        avg4(i,j) = (AUC3+Fmeasure3+Gmeans3)/3;
        avg5(i,j) = (AUC4+Fmeasure4+Gmeans4)/3;
    end
end

figure
surf([0 5 10],[0 5 10],avg4);
xlabel('m_p'),ylabel('m_n'),zlabel('AVG');
title('LDM');
shading interp;
zlim([50 100])
colormap(autumn);


figure
surf([0 5 10],[0 5 10],avg5);
xlabel('m_p'),ylabel('m_n'),zlabel('AVG');
title('UPLDM');
shading interp;
zlim([50 100])
colormap(autumn);






i=0;
disp('imbalance=10');
for ii = 0:5:10
    i=i+1;
    j=0;
    for jj = 0:5:10
        fprintf(' \n[ii=%1.0f,jj=%1.0f] \n',ii,jj);
        j=j+1;
        % 第一组
        mul=[0.5,-3.5]; % 均值
        S1=[1.5 0;0 1.5]; % 协方差
        data1=mvnrnd(mul, S1, 455); % 产生高斯分布数据
        data1(:,3)=-1;
        % 第二组数据
        mu2=[-0.5,3.5];
        S2=[1.5 0;0 1.5];
        data2=mvnrnd(mu2,S2,45);
        data2(:,3)=1;
        % noises of n
        mm1=ii;
        mu3=[0,3];
        S3=[1 -0.8;-0.8 1];
        data3=mvnrnd(mu3,S3,mm1);
        data3(:,3)=1;
        % noises of p
        mm2=jj;
        mu4=[0,-3];
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
        [opt_tau4,...
        acc_ldm,acc_upldm,...
        time3,time4,...
        opt_C3,opt_C4,...
        AUC3,Sensitivity3,Specificity3,Fmeasure3,Gmeans3,...
        AUC4,Sensitivity4,Specificity4,Fmeasure4,Gmeans4]= tune_tau1(Samples_Train,Labels_Train,Ctest,dtest,c1val,kernel,p1,lamb1,lamb2);
              
        %% LDM
        fprintf(['\n LDM AVG=%3.2f,Accuracy=%3.2f,AUC=%3.2f,Sensitivity=%3.2f,Specificity=%3.2f,Fmeasure=%3.2f,Gmeans=%3.2f,' ...
            'time = %3.2f,C = %3.2f,p1=%3.2f'],(AUC3+Fmeasure3+Gmeans3)/3,acc_ldm,AUC3,Sensitivity3,Specificity3,Fmeasure3,Gmeans3,time3,opt_C3,p1);
        
        %% UPLDM
        fprintf(['\n UPLDM AVG=%3.2f,Accuracy=%3.2f,AUC=%3.2f,Sensitivity=%3.2f,Specificity=%3.2f,Fmeasure=%3.2f,Gmeans=%3.2f,' ...
            'time = %3.2f,tau = %3.2f,C = %3.2f,p1=%3.2f\n'],(AUC4+Fmeasure4+Gmeans4)/3,acc_upldm,AUC4,Sensitivity4,Specificity4,Fmeasure4,Gmeans4,time4,opt_tau4,opt_C4,p1);

        %%
        avg4(i,j) = (AUC3+Fmeasure3+Gmeans3)/3;
        avg5(i,j) = (AUC4+Fmeasure4+Gmeans4)/3;
    end
end



figure
surf([0 5 10],[0 5 10],avg4);
xlabel('m_p'),ylabel('m_n'),zlabel('AVG');
title('LDM');
shading interp;
zlim([50 100])
colormap(autumn);


figure
surf([0 5 10],[0 5 10],avg5);
xlabel('m_p'),ylabel('m_n'),zlabel('AVG');
title('UPLDM');
shading interp;
zlim([50 100])
colormap(autumn);





i=0;
disp('imbalance=20');
for ii = 0:5:10
    i=i+1;
    j=0;
    for jj = 0:5:10
        fprintf(' \n[ii=%1.0f,jj=%1.0f] \n',ii,jj);
        j=j+1;
        % 第一组
        mul=[0.5,-3.5]; % 均值
        S1=[1.5 0;0 1.5]; % 协方差
        data1=mvnrnd(mul, S1, 476); % 产生高斯分布数据
        data1(:,3)=-1;
        % 第二组数据
        mu2=[-0.5,3.5];
        S2=[1.5 0;0 1.5];
        data2=mvnrnd(mu2,S2,24);
        data2(:,3)=1;
        % noises of n
        mm1=ii;
        mu3=[0,3];
        S3=[1 -0.8;-0.8 1];
        data3=mvnrnd(mu3,S3,mm1);
        data3(:,3)=1;
        % noises of p
        mm2=jj;
        mu4=[0,-3];
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
         [opt_tau4,...
        acc_ldm,acc_upldm,...
        time3,time4,...
        opt_C3,opt_C4,...
        AUC3,Sensitivity3,Specificity3,Fmeasure3,Gmeans3,...
        AUC4,Sensitivity4,Specificity4,Fmeasure4,Gmeans4]= tune_tau1(Samples_Train,Labels_Train,Ctest,dtest,c1val,kernel,p1,lamb1,lamb2);
              
        %% LDM
        fprintf(['\n LDM AVG=%3.2f,Accuracy=%3.2f,AUC=%3.2f,Sensitivity=%3.2f,Specificity=%3.2f,Fmeasure=%3.2f,Gmeans=%3.2f,' ...
            'time = %3.2f,C = %3.2f,p1=%3.2f'],(AUC3+Fmeasure3+Gmeans3)/3,acc_ldm,AUC3,Sensitivity3,Specificity3,Fmeasure3,Gmeans3,time3,opt_C3,p1);
        
        %% UPLDM
        fprintf(['\n UPLDM AVG=%3.2f,Accuracy=%3.2f,AUC=%3.2f,Sensitivity=%3.2f,Specificity=%3.2f,Fmeasure=%3.2f,Gmeans=%3.2f,' ...
            'time = %3.2f,tau = %3.2f,C = %3.2f,p1=%3.2f\n'],(AUC4+Fmeasure4+Gmeans4)/3,acc_upldm,AUC4,Sensitivity4,Specificity4,Fmeasure4,Gmeans4,time4,opt_tau4,opt_C4,p1);
        %%
        avg4(i,j) = (AUC3+Fmeasure3+Gmeans3)/3;
        avg5(i,j) = (AUC4+Fmeasure4+Gmeans4)/3;
    end
end


figure
surf([0 5 10],[0 5 10],avg4);
xlabel('m_p'),ylabel('m_n'),zlabel('AVG');
title('LDM');
shading interp;
zlim([50 100])
colormap(autumn);


figure
surf([0 5 10],[0 5 10],avg5);
xlabel('m_p'),ylabel('m_n'),zlabel('AVG');
title('UPLDM');
shading interp;
zlim([50 100])
colormap(autumn);


