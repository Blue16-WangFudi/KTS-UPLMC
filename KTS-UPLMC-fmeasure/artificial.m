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
%         % kmeans-smote
%         maj=length(find(Labels_Train==-1));
%         min=length(find(Labels_Train==1));
%         k=10;
%         cluster = kmeans(Samples_Train,k); 
%         dataa=[];
%         sparsity=zeros(1,k);
%         total_sparsity = 0;
%         for i=1:k
%             class=Data_Train(find(cluster==i),:);
%             class1_num=length(find(cluster==i & Labels_Train==1));
%             class2_num=length(find(cluster==i & Labels_Train==-1));
%             ratio=class1_num/class2_num;
%             if(ratio>1)
%                 idx = 1;
%                 dist_mat = zeros(sum(class1_num));
%                 subclass_sample = Samples_Train(find(cluster==i & Labels_Train==1),:);
%                 subclass_label = Labels_Train(find(cluster==i & Labels_Train==1),:);
%                 for j = 1:size(subclass_sample, 1)
%                     dist_vec = pdist2(subclass_sample(j,:), subclass_sample);
%                     dist_mat(idx,:) = dist_vec;
%                     idx = idx + 1;
%                 end
%                 avg_dist = mean(dist_mat(:));
%                 % 计算聚类密度和稀疏因子
%                 m = size(Samples_Train, 2);  % 特征数
%                 density = class1_num ./ (avg_dist .^ m);%密度
%                 sparsity(i) = 1 ./ density; % 稀疏度
%                 total_sparsity = total_sparsity + sparsity(i);
%             end
%         end
%         for i=1:k
%             class=Data_Train(find(cluster==i),:);
%             class1_num=length(find(cluster==i & Labels_Train==1));
%             class2_num=length(find(cluster==i & Labels_Train==-1));   
%             ratio=class1_num/class2_num;
%             if(ratio>1)
%                 sampling_weights = sparsity(i)/total_sparsity;
%                 S_Train=Data_Train(find(cluster==i),1:end-1);
%                 L_Train=Data_Train(find(cluster==i),end);
%                 S_Train= S_Train';
%                 L_Train=L_Train';
%                 Cost=sampling_weights*(maj-min);
%                 ClassType=unique(L_Train);
%                 [Att,~]=size(S_Train);
%                 attribute=zeros(Att,1);
%                 %------------- Generate new training set using FUNCTION OverSampling -------------------
%                 [S_Train,L_Train]=KSmoteOverSampling(S_Train, L_Train,ClassType,Cost,attribute,5,'numeric');
%                 S_Train= S_Train';
%                 L_Train=L_Train';
%                 Dataa = [S_Train,L_Train];
%                 dataa=[Dataa;dataa];
%             else
%                 S_Train=Data_Train(find(cluster==i),1:end-1);
%                 L_Train=Data_Train(find(cluster==i),end);
%                 Dataa = [S_Train,L_Train];
%                 dataa=[Dataa;dataa]; 
%             end   
%         end
%         Samples_Train= dataa(:,1:end-1);
%         Labels_Train=dataa(:,end);

        
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
for ii = 0:5:10
    i=i+1;
    j=0;
    for jj = 0:5:10
        fprintf(' \n[ii=%1.0f,jj=%1.0f] \n',ii,jj);
        j=j+1;
        % 第一组
        mul=[0.5,-3.5]; % 均值
        S1=[0.2 0;0 3]; % 协方差
        data1=mvnrnd(mul, S1, 417); % 产生高斯分布数据
        data1(:,3)=-1;
        % 第二组数据
        mu2=[-0.5,3.5];
        S2=[0.2 0;0 3];
        data2=mvnrnd(mu2,S2,83);
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
        % kmeans-smote
        maj=length(find(Labels_Train==-1));
        min=length(find(Labels_Train==1));
        k=10;
        cluster = kmeans(Samples_Train,k); 
        dataa=[];
        sparsity=zeros(1,k);
        total_sparsity = 0;
        for i=1:k
            class=Data_Train(find(cluster==i),:);
            class1_num=length(find(cluster==i & Labels_Train==1));
            class2_num=length(find(cluster==i & Labels_Train==-1));
            ratio=class1_num/class2_num;
            if(ratio>1)
                idx = 1;
                dist_mat = zeros(sum(class1_num));
                subclass_sample = Samples_Train(find(cluster==i & Labels_Train==1),:);
                subclass_label = Labels_Train(find(cluster==i & Labels_Train==1),:);
                for j = 1:size(subclass_sample, 1)
                    dist_vec = pdist2(subclass_sample(j,:), subclass_sample);
                    dist_mat(idx,:) = dist_vec;
                    idx = idx + 1;
                end
                avg_dist = mean(dist_mat(:));
                % 计算聚类密度和稀疏因子
                m = size(Samples_Train, 2);  % 特征数
                density = class1_num ./ (avg_dist .^ m);%密度
                sparsity(i) = 1 ./ density; % 稀疏度
                total_sparsity = total_sparsity + sparsity(i);
            end
        end
        for i=1:k
            class=Data_Train(find(cluster==i),:);
            class1_num=length(find(cluster==i & Labels_Train==1));
            class2_num=length(find(cluster==i & Labels_Train==-1));   
            ratio=class1_num/class2_num;
            if(ratio>1)
                sampling_weights = sparsity(i)/total_sparsity;
                S_Train=Data_Train(find(cluster==i),1:end-1);
                L_Train=Data_Train(find(cluster==i),end);
                S_Train= S_Train';
                L_Train=L_Train';
                Cost=sampling_weights*(maj-min);
                ClassType=unique(L_Train);
                [Att,~]=size(S_Train);
                attribute=zeros(Att,1);
                %------------- Generate new training set using FUNCTION OverSampling -------------------
                [S_Train,L_Train]=KSmoteOverSampling(S_Train, L_Train,ClassType,Cost,attribute,5,'numeric');
                S_Train= S_Train';
                L_Train=L_Train';
                Dataa = [S_Train,L_Train];
                dataa=[Dataa;dataa];
            else
                S_Train=Data_Train(find(cluster==i),1:end-1);
                L_Train=Data_Train(find(cluster==i),end);
                Dataa = [S_Train,L_Train];
                dataa=[Dataa;dataa]; 
            end   
        end
        Samples_Train= dataa(:,1:end-1);
        Labels_Train=dataa(:,end);

        
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
        opt_C0,opt_C1,opt_C2,opt_C3,opt_C4,h,...
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
surf([0 5 10],[0 5 10],avg1);
xlabel('m_p'),ylabel('m_n'),zlabel('AVG');
title('SVM');
shading interp;
zlim([50 100])
colormap(autumn);

figure
surf([0 5 10],[0 5 10],avg2);
xlabel('m_p'),ylabel('m_n'),zlabel('AVG');
title('UPSVM');
shading interp;
zlim([50 100])
colormap(autumn);

figure
surf([0 5 10],[0 5 10],avg3);
xlabel('m_p'),ylabel('m_n'),zlabel('AVG');
title('PSVM');
shading interp;
zlim([50 100])
colormap(autumn);


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
        S1=[0.2 0;0 3]; % 协方差
        data1=mvnrnd(mul, S1, 455); % 产生高斯分布数据
        data1(:,3)=-1;
        % 第二组数据
        mu2=[-0.5,3.5];
        S2=[0.2 0;0 3];
        data2=mvnrnd(mu2,S2,45);
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
        % kmeans-smote
        maj=length(find(Labels_Train==-1));
        min=length(find(Labels_Train==1));
        k=10;
        cluster = kmeans(Samples_Train,k); 
        dataa=[];
        sparsity=zeros(1,k);
        total_sparsity = 0;
        for i=1:k
            class=Data_Train(find(cluster==i),:);
            class1_num=length(find(cluster==i & Labels_Train==1));
            class2_num=length(find(cluster==i & Labels_Train==-1));
            ratio=class1_num/class2_num;
            if(ratio>1)
                idx = 1;
                dist_mat = zeros(sum(class1_num));
                subclass_sample = Samples_Train(find(cluster==i & Labels_Train==1),:);
                subclass_label = Labels_Train(find(cluster==i & Labels_Train==1),:);
                for j = 1:size(subclass_sample, 1)
                    dist_vec = pdist2(subclass_sample(j,:), subclass_sample);
                    dist_mat(idx,:) = dist_vec;
                    idx = idx + 1;
                end
                avg_dist = mean(dist_mat(:));
                % 计算聚类密度和稀疏因子
                m = size(Samples_Train, 2);  % 特征数
                density = class1_num ./ (avg_dist .^ m);%密度
                sparsity(i) = 1 ./ density; % 稀疏度
                total_sparsity = total_sparsity + sparsity(i);
            end
        end
        for i=1:k
            class=Data_Train(find(cluster==i),:);
            class1_num=length(find(cluster==i & Labels_Train==1));
            class2_num=length(find(cluster==i & Labels_Train==-1));   
            ratio=class1_num/class2_num;
            if(ratio>1)
                sampling_weights = sparsity(i)/total_sparsity;
                S_Train=Data_Train(find(cluster==i),1:end-1);
                L_Train=Data_Train(find(cluster==i),end);
                S_Train= S_Train';
                L_Train=L_Train';
                Cost=sampling_weights*(maj-min);
                ClassType=unique(L_Train);
                [Att,~]=size(S_Train);
                attribute=zeros(Att,1);
                %------------- Generate new training set using FUNCTION OverSampling -------------------
                [S_Train,L_Train]=KSmoteOverSampling(S_Train, L_Train,ClassType,Cost,attribute,5,'numeric');
                S_Train= S_Train';
                L_Train=L_Train';
                Dataa = [S_Train,L_Train];
                dataa=[Dataa;dataa];
            else
                S_Train=Data_Train(find(cluster==i),1:end-1);
                L_Train=Data_Train(find(cluster==i),end);
                Dataa = [S_Train,L_Train];
                dataa=[Dataa;dataa]; 
            end   
        end
        Samples_Train= dataa(:,1:end-1);
        Labels_Train=dataa(:,end);

        
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
        opt_C0,opt_C1,opt_C2,opt_C3,opt_C4,h,...
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
surf([0 5 10],[0 5 10],avg1);
xlabel('m_p'),ylabel('m_n'),zlabel('AVG');
title('SVM');
shading interp;
zlim([50 100])
colormap(autumn);

figure
surf([0 5 10],[0 5 10],avg2);
xlabel('m_p'),ylabel('m_n'),zlabel('AVG');
title('UPSVM');
shading interp;
zlim([50 100])
colormap(autumn);

figure
surf([0 5 10],[0 5 10],avg3);
xlabel('m_p'),ylabel('m_n'),zlabel('AVG');
title('PSVM');
shading interp;
zlim([50 100])
colormap(autumn);


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
        S1=[0.2 0;0 3]; % 协方差
        data1=mvnrnd(mul, S1, 476); % 产生高斯分布数据
        data1(:,3)=-1;
        % 第二组数据
        mu2=[-0.5,3.5];
        S2=[0.2 0;0 3];
        data2=mvnrnd(mu2,S2,24);
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
        % kmeans-smote
        maj=length(find(Labels_Train==-1));
        min=length(find(Labels_Train==1));
        k=10;
        cluster = kmeans(Samples_Train,k); 
        dataa=[];
        sparsity=zeros(1,k);
        total_sparsity = 0;
        for i=1:k
            class=Data_Train(find(cluster==i),:);
            class1_num=length(find(cluster==i & Labels_Train==1));
            class2_num=length(find(cluster==i & Labels_Train==-1));
            ratio=class1_num/class2_num;
            if(ratio>1)
                idx = 1;
                dist_mat = zeros(sum(class1_num));
                subclass_sample = Samples_Train(find(cluster==i & Labels_Train==1),:);
                subclass_label = Labels_Train(find(cluster==i & Labels_Train==1),:);
                for j = 1:size(subclass_sample, 1)
                    dist_vec = pdist2(subclass_sample(j,:), subclass_sample);
                    dist_mat(idx,:) = dist_vec;
                    idx = idx + 1;
                end
                avg_dist = mean(dist_mat(:));
                % 计算聚类密度和稀疏因子
                m = size(Samples_Train, 2);  % 特征数
                density = class1_num ./ (avg_dist .^ m);%密度
                sparsity(i) = 1 ./ density; % 稀疏度
                total_sparsity = total_sparsity + sparsity(i);
            end
        end
        for i=1:k
            class=Data_Train(find(cluster==i),:);
            class1_num=length(find(cluster==i & Labels_Train==1));
            class2_num=length(find(cluster==i & Labels_Train==-1));   
            ratio=class1_num/class2_num;
            if(ratio>1)
                sampling_weights = sparsity(i)/total_sparsity;
                S_Train=Data_Train(find(cluster==i),1:end-1);
                L_Train=Data_Train(find(cluster==i),end);
                S_Train= S_Train';
                L_Train=L_Train';
                Cost=sampling_weights*(maj-min);
                ClassType=unique(L_Train);
                [Att,~]=size(S_Train);
                attribute=zeros(Att,1);
                %------------- Generate new training set using FUNCTION OverSampling -------------------
                [S_Train,L_Train]=KSmoteOverSampling(S_Train, L_Train,ClassType,Cost,attribute,5,'numeric');
                S_Train= S_Train';
                L_Train=L_Train';
                Dataa = [S_Train,L_Train];
                dataa=[Dataa;dataa];
            else
                S_Train=Data_Train(find(cluster==i),1:end-1);
                L_Train=Data_Train(find(cluster==i),end);
                Dataa = [S_Train,L_Train];
                dataa=[Dataa;dataa]; 
            end   
        end
        Samples_Train= dataa(:,1:end-1);
        Labels_Train=dataa(:,end);

        
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
        opt_C0,opt_C1,opt_C2,opt_C3,opt_C4,h,...
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
surf([0 5 10],[0 5 10],avg1);
xlabel('m_p'),ylabel('m_n'),zlabel('AVG');
title('SVM');
shading interp;
zlim([50 100])
colormap(autumn);

figure
surf([0 5 10],[0 5 10],avg2);
xlabel('m_p'),ylabel('m_n'),zlabel('AVG');
title('UPSVM');
shading interp;
zlim([50 100])
colormap(autumn);

figure
surf([0 5 10],[0 5 10],avg3);
xlabel('m_p'),ylabel('m_n'),zlabel('AVG');
title('PSVM');
shading interp;
zlim([50 100])
colormap(autumn);


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


