%人工数据集散点图
%%
close all;
clear all;
%% 
clc
rand('state', 2015)
randn('state', 2015)
%% 
% 第一组
mul=[0.5,-3.5]; % 均值
S1=[1.5 0;0 1.5]; % 协方差
data1=mvnrnd(mul, S1, 1000); % 产生高斯分布数据
data1(:,3)=-1;
% 第二组数据
mu2=[-0.5,3.5];
S2=[1.5 0;0 1.5];
noise=[0.4106,-3.9505,1];
data2=mvnrnd(mu2,S2,50);
data2(:,3)=1;
data2=[data2;noise];


for flag=5
    %% NON
    if flag==1
        %figure
        scatter(data1(:,1), data1(:,2), 'filled', 'b');
        hold on;
        scatter(data2(:,1), data2(:,2), 'filled', 'r');
        xlabel('X-axis');  
        ylabel('Y-axis');  
        title('Scatter plot');  
        legend('data1', 'data2');
        hold off
    end
    %% SMOTE
    if flag==2
        Data_Original=[data1;data2];
        Samples_Train = Data_Original(:, 1:end-1);
        Labels_Train = Data_Original(:, end);
        Samples_Train= Samples_Train';
        Labels_Train=Labels_Train';
        Cost=[1,1];
        ClassType=unique(Labels_Train);
        [Att,~]=size(Samples_Train);
        attribute=zeros(Att,1);
        %------------- Generate new training set using FUNCTION OverSampling -------------------
        [Samples_Train,Labels_Train]=SmoteOverSamplingO(Samples_Train, Labels_Train,ClassType,Cost,attribute,5,'numeric');
        Samples_Train= Samples_Train';
        Labels_Train=Labels_Train';  
        Merged = [Samples_Train Labels_Train];
        MergedData1 = Merged(Merged(:,3) == -1, :);
        MergedData2 = Merged(Merged(:,3) == 1, :);
           
        %figure
        scatter(MergedData1(:,1), MergedData1(:,2), 'filled', 'b');
        hold on;
        scatter(MergedData2(:,1), MergedData2(:,2), 'filled', 'r');
        xlabel('X-axis');  
        ylabel('Y-axis');  
        title('Scatter plot');  
        legend('data1', 'data2');
        hold off
    end
    %% TS
    if flag==3
        Data_Original=[data1;data2];
        Samples_Train = Data_Original(:, 1:end-1);
        Labels_Train = Data_Original(:, end);
        Samples_Train= Samples_Train';
        Labels_Train=Labels_Train';
        Cost=[1,1];
        ClassType=unique(Labels_Train);
        [Att,~]=size(Samples_Train);
        attribute=zeros(Att,1);
        %------------- Generate new training set using FUNCTION OverSampling -------------------
        [Samples_Train,Labels_Train]=SmoteOverSampling(Samples_Train, Labels_Train,ClassType,Cost,attribute,5,'numeric');
        Samples_Train= Samples_Train';
        Labels_Train=Labels_Train';  
        Merged = [Samples_Train Labels_Train];
        MergedData1 = Merged(Merged(:,3) == -1, :);
        MergedData2 = Merged(Merged(:,3) == 1, :);
           
        %figure
        scatter(MergedData1(:,1), MergedData1(:,2), 'filled', 'b');
        hold on;
        scatter(MergedData2(:,1), MergedData2(:,2), 'filled', 'r');
        xlabel('X-axis');  
        ylabel('Y-axis');  
        title('Scatter plot');  
        legend('data1', 'data2');
        hold off
    end
    %% KSMOTE
    if flag==4
        Data_Original=[data1;data2]; 
        Samples_Train = Data_Original(:, 1:end-1);
        Labels_Train = Data_Original(:, end);
        maj=length(find(Labels_Train==-1));
        min=length(find(Labels_Train==1));
        k=10;
        cluster = kmeans(Samples_Train,k); 
        dataa=[];
        sparsity=zeros(1,k);
        total_sparsity = 0;
        for i=1:k
            class=Data_Original(find(cluster==i),:);
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
            class=Data_Original(find(cluster==i),:);
            class1_num=length(find(cluster==i & Labels_Train==1));
            class2_num=length(find(cluster==i & Labels_Train==-1));   
            ratio=class1_num/class2_num;
            if(ratio>1)
                sampling_weights = sparsity(i)/total_sparsity;
                S_Train=Data_Original(find(cluster==i),1:end-1);
                L_Train=Data_Original(find(cluster==i),end);
                S_Train= S_Train';
                L_Train=L_Train';
                Cost=sampling_weights*(maj-min);
                ClassType=unique(L_Train);
                [Att,~]=size(S_Train);
                attribute=zeros(Att,1);
                %------------- Generate new training set using FUNCTION OverSampling -------------------
                [S_Train,L_Train]=KSmoteOverSamplingO(S_Train, L_Train,ClassType,Cost,attribute,5,'numeric');
                S_Train= S_Train';
                L_Train=L_Train';
                Dataa = [S_Train,L_Train];
                dataa=[Dataa;dataa];
            else
                S_Train=Data_Original(find(cluster==i),1:end-1);
                L_Train=Data_Original(find(cluster==i),end);
                Dataa = [S_Train,L_Train];
                dataa=[Dataa;dataa]; 
            end   
        end
        Samples_Train= dataa(:,1:end-1);
        Labels_Train=dataa(:,end);
        
        Merged = [Samples_Train Labels_Train];
        MergedData1 = Merged(Merged(:,3) == -1, :);
        MergedData2 = Merged(Merged(:,3) == 1, :);
        
        %figure
        scatter(MergedData1(:,1), MergedData1(:,2), 'filled', 'b');
        hold on;
        scatter(MergedData2(:,1), MergedData2(:,2), 'filled', 'r');
        xlabel('X-axis');  
        ylabel('Y-axis');  
        title('Scatter plot');  
        legend('data1', 'data2');  
        hold off
    end
    %% KTS
    if flag==5
        Data_Original=[data1;data2]; 
        Samples_Train = Data_Original(:, 1:end-1);
        Labels_Train = Data_Original(:, end);
        maj=length(find(Labels_Train==-1));
        min=length(find(Labels_Train==1));
        k=10;
        cluster = kmeans(Samples_Train,k); 
        dataa=[];
        sparsity=zeros(1,k);
        total_sparsity = 0;
        for i=1:k
            class=Data_Original(find(cluster==i),:);
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
            class=Data_Original(find(cluster==i),:);
            class1_num=length(find(cluster==i & Labels_Train==1));
            class2_num=length(find(cluster==i & Labels_Train==-1));   
            ratio=class1_num/class2_num;
            if(ratio>1)
                sampling_weights = sparsity(i)/total_sparsity;
                S_Train=Data_Original(find(cluster==i),1:end-1);
                L_Train=Data_Original(find(cluster==i),end);
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
                S_Train=Data_Original(find(cluster==i),1:end-1);
                L_Train=Data_Original(find(cluster==i),end);
                Dataa = [S_Train,L_Train];
                dataa=[Dataa;dataa]; 
            end   
        end
        Samples_Train= dataa(:,1:end-1);
        Labels_Train=dataa(:,end);
        
        Merged = [Samples_Train Labels_Train];
        MergedData1 = Merged(Merged(:,3) == -1, :);
        MergedData2 = Merged(Merged(:,3) == 1, :);
        
        %figure
        scatter(MergedData1(:,1), MergedData1(:,2), 'filled', 'b');
        hold on;
        scatter(MergedData2(:,1), MergedData2(:,2), 'filled', 'r');
        xlabel('X-axis');  
        ylabel('Y-axis');  
        title('Scatter plot');  
        legend('data1', 'data2');   
        hold off
    end
end

