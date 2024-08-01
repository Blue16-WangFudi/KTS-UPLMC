tic
close all;
clear all;
clc
rand('state', 2015)
randn('state', 2015)
%%
for index=2
    %%
    if (index==1)
        Output = load('Data_mat\seeds.mat');
        Data_Name = fieldnames(Output);   % A struct data
        Data_Original = getfield(Output, Data_Name{1}); % Abstract the data
        % Normalization
        Data_Original = [mapminmax(Data_Original(:, 1:end-1)', 0, 1)', Data_Original(:, end)]; % Map the original data to value between [0, 1] by colum
        M_Original = size(Data_Original, 1);
        Data_Original = Data_Original(randperm(M_Original), :);
        disp('seeds');
    end
    if (index==2)
        Output = load('Data_mat\Wine dataset.mat');
        Data_Name = fieldnames(Output);   % A struct data
        Data_Original = getfield(Output, Data_Name{1}); % Abstract the data
        % Normalization
        Data_Original = [mapminmax(Data_Original(:, 1:end-1)', 0, 1)', Data_Original(:, end)]; % Map the original data to value between [0, 1] by colum
        M_Original = size(Data_Original, 1);
        Data_Original = Data_Original(randperm(M_Original), :);
        disp('Wine dataset');
    end
    if (index==3)
        Output = load('Data_mat\heart_failure_clinical_records_dataset.mat');
        Data_Name = fieldnames(Output);   % A struct data
        Data_Original = getfield(Output, Data_Name{1}); % Abstract the data
        % Normalization
        Data_Original = [mapminmax(Data_Original(:, 1:end-1)', 0, 1)', Data_Original(:, end)]; % Map the original data to value between [0, 1] by colum
        M_Original = size(Data_Original, 1);
        Data_Original = Data_Original(randperm(M_Original), :);
        disp('heart_failure_clinical_records_dataset');
    end

    if (index==4)
        Output = load('Data_mat\ionosphere_data.mat');
        Data_Name = fieldnames(Output);   % A struct data
        Data_Original = getfield(Output, Data_Name{1}); % Abstract the data
        % Normalization
        Data_Original = [mapminmax(Data_Original(:, 1:end-1)', 0, 1)', Data_Original(:, end)]; % Map the original data to value between [0, 1] by colum
        M_Original = size(Data_Original, 1);
        Data_Original = Data_Original(randperm(M_Original), :);
        disp('ionosphere_data');
    end
    if (index==5)
        Output = load('Data_mat\Pima_indians.mat');
        Data_Name = fieldnames(Output);   % A struct data
        Data_Original = getfield(Output, Data_Name{1}); % Abstract the data
        %Normalization
        Data_Original = [mapminmax(Data_Original(:, 1:end-1)', 0, 1)', Data_Original(:, end)]; % Map the original data to value between [0, 1] by colum
        M_Original = size(Data_Original, 1);
        Data_Original = Data_Original(randperm(M_Original), :);
        disp('Pima_indians');
    end
    if (index==6)
        Output = load('Data_mat\haberman.mat');
        Data_Name = fieldnames(Output);   % A struct data
        Data_Original = getfield(Output, Data_Name{1}); % Abstract the data
        % Normalization
        Data_Original = [mapminmax(Data_Original(:, 1:end-1)', 0, 1)', Data_Original(:, end)]; % Map the original data to value between [0, 1] by colum
        M_Original = size(Data_Original, 1);
        Data_Original = Data_Original(randperm(M_Original), :);
        disp('haberman');
    end
    
    
    %%
    for flag=1
        if flag==1
            TrainRate = 0.5;       % The scale of the train set  
            [Data_Train, Data_Predict] = Data_Rate(Data_Original, TrainRate);   % Chose 3       
            Samples_Train = Data_Train(:, 1:end-1);
            Labels_Train = Data_Train(:, end);
            Ctest=Data_Predict(:, 1:end-1);
            dtest=Data_Predict(:, end);
        end
        if flag==2
            TrainRate = 0.5;       % The scale of the train set  
            [Data_Train, Data_Predict] = Data_Rate(Data_Original, TrainRate);   % Chose 3       
            Samples_Train = Data_Train(:, 1:end-1);
            Labels_Train = Data_Train(:, end);
            Ctest=Data_Predict(:, 1:end-1);
            dtest=Data_Predict(:, end);
            % smote 
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
        end
        if flag==3
            TrainRate = 0.5;       % The scale of the train set  
            [Data_Train, Data_Predict] = Data_Rate(Data_Original, TrainRate);   % Chose 3       
            Samples_Train = Data_Train(:, 1:end-1);
            Labels_Train = Data_Train(:, end);
            Ctest=Data_Predict(:, 1:end-1);
            dtest=Data_Predict(:, end);
           % kmeans-smote
            maj=length(find(Labels_Train==-1));
            min=length(find(Labels_Train==1));
            k=5;
            cluster = kmeans(Samples_Train,k); 
            dataa=[];
            sparsity=zeros(1,k);
            total_sparsity =0;
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
                    % 璁＄畻鑱氱被瀵嗗害鍜岀█鐤忓洜瀛�
                    m = size(Samples_Train, 2);  % 鐗瑰緛鏁�
                    density = class1_num ./ (avg_dist .^ m);%瀵嗗害
                    sparsity(i) = 1 ./ density; % 绋�鐤忓害
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
        end

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
        AUC4,Sensitivity4,Specificity4,Fmeasure4,Gmeans4] = tune_tau1(Samples_Train,Labels_Train,Ctest,dtest,c1val,kernel,p1,lamb1,lamb2);

    %%
    if flag==1
         disp('NON');
    else
         disp('SMOTE');
    end    
    
    %% LDM
    fprintf(['\n LDM AVG=%3.2f,AUC=%3.2f,Fmeasure=%3.2f,Gmeans=%3.2f,' ...
        'time = %3.2f,C = %3.2f'],(AUC3+Fmeasure3+Gmeans3)/3,AUC3,Fmeasure3,Gmeans3,time3,opt_C3);
    
    %% UPLDM
    fprintf(['\n UPLDM AVG=%3.2f,AUC=%3.2f,Fmeasure=%3.2f,Gmeans=%3.2f,' ...
        'time = %3.2f,tau = %3.2f,C = %3.2f\n'],(AUC4+Fmeasure4+Gmeans4)/3,AUC4,Fmeasure4,Gmeans4,time4,opt_tau4,opt_C4);

    end
end





