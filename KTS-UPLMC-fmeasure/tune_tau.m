 function [opt_tau1,opt_tau2,opt_tau4,...
        acc_svm,acc_upsvm,acc_psvm,acc_ldm,acc_upldm,...
        time0,time1,time2,time3,time4,...
        opt_C0,opt_C1,opt_C2,opt_C3,opt_C4,...
        AUC0,Sensitivity0,Specificity0,Fmeasure0,Gmeans0,...
        AUC1,Sensitivity1,Specificity1,Fmeasure1,Gmeans1,...
        AUC2,Sensitivity2,Specificity2,Fmeasure2,Gmeans2,...
        AUC3,Sensitivity3,Specificity3,Fmeasure3,Gmeans3,...
        AUC4,Sensitivity4,Specificity4,Fmeasure4,Gmeans4]= tune_tau(Ctrain,dtrain,Ctest,dtest,C,kernel,p1,lamb1,lamb2)
tauval= -1:0.1:1;
fmeasure_k3 = zeros(length(lamb1),length(lamb2));fmeasure_k4 = zeros(length(lamb1),length(lamb2));
C_k3 = zeros(length(lamb1),length(lamb2));C_k4 = zeros(length(lamb1),length(lamb2));
fmeasure0_ = zeros(1,length(tauval));fmeasure1_ = zeros(1,length(tauval));fmeasure2_ = zeros(1,length(tauval));
fmeasure3_ = zeros(1,length(tauval));fmeasure4_ = zeros(1,length(tauval));count=0;

%%
for jj=1:length(tauval)
        time0=0;time1=0;time2=0;
        fprintf('%3.0f steps remaining...\n',length(tauval)-count);
        tau= tauval(jj);
        tic
        [acc_svm,AUC0,Sensitivity0,Specificity0,fmeasure0,Gmeans0,C0] = Unified_pin_svm(Ctrain, dtrain, Ctest,dtest, kernel, 0,C,p1);  %SVM
        time0 = time0 + toc;
        tic
        [acc_upsvm,AUC1,Sensitivity1,Specificity1,fmeasure1,Gmeans1,C1] = Unified_pin_svm(Ctrain, dtrain, Ctest,dtest, kernel, tau,C,p1);  %UPSVM
        time1 = time1 + toc;
        tic
        [acc_psvm,AUC2,Sensitivity2,Specificity2,fmeasure2,Gmeans2,C2] = pin_svm(Ctrain, dtrain, Ctest,dtest, kernel, tau,C,p1); %PSVM
        time2 = time2 + toc;
        tic
        for k1 = 1:length(lamb1)
            for k2 = 1:length(lamb2)
                time3=0;time4=0;
                tic
                [acc_ldm,AUC3,Sensitivity3,Specificity3,fmeasure3,Gmeans3,C3] = Unified_pin_ldm(Ctrain, dtrain, Ctest,dtest, kernel, 0,C,p1,lamb1(k1),lamb2(k2));  %LDM
                time3 = time3 + toc;
                tic
                [acc_upldm,AUC4,Sensitivity4,Specificity4,fmeasure4,Gmeans4,C4] = Unified_pin_ldm(Ctrain, dtrain, Ctest,dtest, kernel, tau,C,p1,lamb1(k1),lamb2(k2));  %UPLDM
                time4 = time4 + toc;
                fmeasure_k3(k1,k2)= fmeasure3;
                fmeasure_k4(k1,k2)= fmeasure4;
                C_k3(k1,k2)= C3;
                C_k4(k1,k2)= C4;
            end
           
        end
    
    fmeasure0_(jj)= fmeasure0;
    fmeasure1_(jj)= fmeasure1;
    fmeasure2_(jj)= fmeasure2;
    fmeasure3_(jj)= max(max(fmeasure_k3));
    fmeasure4_(jj)= max(max(fmeasure_k4));
    % 检查数组中是否全是NaN  
    is_all_nan = isnan(fmeasure_k3);  
    if is_all_nan  
        fmeasure_k3(:) = 0;
    end
    is_all_nan = isnan(fmeasure_k4);  
    if is_all_nan   
        fmeasure_k4(:) = 0;
    end
    [x3,y3]=find(fmeasure_k3==max(max(fmeasure_k3)));xx3=x3(1);yy3=y3(1);
    [x4,y4]=find(fmeasure_k4==max(max(fmeasure_k4)));xx4=x4(1);yy4=y4(1);
    C0_(jj)= C0;
    C1_(jj)= C1;
    C2_(jj)= C2;
    C3_(jj)= C_k3(xx3,yy3);
    C4_(jj)= C_k4(xx4,yy4);
   count = count+1; 
end
% Ymat=[fmeasure0_',fmeasure1_',fmeasure2_',fmeasure3_',fmeasure4_'];
% h=createfigure(tauval,Ymat);
[Fmeasure0,i0]=max(fmeasure0_);
[Fmeasure1,i1]=max(fmeasure1_);
[Fmeasure2,i2]=max(fmeasure2_);
[Fmeasure3,i3]=max(fmeasure3_);
[Fmeasure4,i4] = max(fmeasure4_);
opt_tau1=tauval(i1);
opt_tau2=tauval(i2);
opt_tau4=tauval(i4);
opt_C0 = C0_(i0);
opt_C1 = C1_(i1);
opt_C2 = C2_(i2);
opt_C3 = C3_(i3);
opt_C4 = C4_(i4);
end
