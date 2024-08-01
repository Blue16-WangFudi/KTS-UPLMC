 function [opt_tau1,opt_tau2,opt_tau4,...
        acc_svm,acc_upsvm,acc_psvm,acc_ldm,acc_upldm,...
        time0,time1,time2,time3,time4,...
        opt_C0,opt_C1,opt_C2,opt_C3,opt_C4,h,...
        AUC0,Sensitivity0,Specificity0,Fmeasure0,Gmeans0,...
        AUC1,Sensitivity1,Specificity1,Fmeasure1,Gmeans1,...
        AUC2,Sensitivity2,Specificity2,Fmeasure2,Gmeans2,...
        AUC3,Sensitivity3,Specificity3,Fmeasure3,Gmeans3,...
        AUC4,Sensitivity4,Specificity4,Fmeasure4,Gmeans4]= tune_tau(Ctrain,dtrain,Ctest,dtest,C,kernel,p1,lamb1,lamb2)
tauval= -1:0.1:1;
gm_k3 = zeros(length(lamb1),length(lamb2));gm_k4 = zeros(length(lamb1),length(lamb2));
C_k3 = zeros(length(lamb1),length(lamb2));C_k4 = zeros(length(lamb1),length(lamb2));
gm0_ = zeros(1,length(tauval));gm1_ = zeros(1,length(tauval));gm2_ = zeros(1,length(tauval));
gm3_ = zeros(1,length(tauval));gm4_ = zeros(1,length(tauval));count=0;

%%
for jj=1:length(tauval)
        time0=0;time1=0;time2=0;
        fprintf('%3.0f steps remaining...\n',length(tauval)-count);
        tau= tauval(jj);
        tic
        [acc_svm,AUC0,Sensitivity0,Specificity0,Fmeasure0,gm0,C0] = Unified_pin_svm(Ctrain, dtrain, Ctest,dtest, kernel, 0,C,p1);  %SVM
        time0 = time0 + toc;
        tic
        [acc_upsvm,AUC1,Sensitivity1,Specificity1,Fmeasure1,gm1,C1] = Unified_pin_svm(Ctrain, dtrain, Ctest,dtest, kernel, tau,C,p1);  %UPSVM
        time1 = time1 + toc;
        tic
        [acc_psvm,AUC2,Sensitivity2,Specificity2,Fmeasure2,gm2,C2] = pin_svm(Ctrain, dtrain, Ctest,dtest, kernel, tau,C,p1); %PSVM
        time2 = time2 + toc;
        tic
        for k1 = 1:length(lamb1)
            for k2 = 1:length(lamb2)
                time3=0;time4=0;
                tic
                [acc_ldm,AUC3,Sensitivity3,Specificity3,Fmeasure3,gm3,C3] = Unified_pin_ldm(Ctrain, dtrain, Ctest,dtest, kernel, 0,C,p1,lamb1(k1),lamb2(k2));  %LDM
                time3 = time3 + toc;
                tic
                [acc_upldm,AUC4,Sensitivity4,Specificity4,Fmeasure4,gm4,C4] = Unified_pin_ldm(Ctrain, dtrain, Ctest,dtest, kernel, tau,C,p1,lamb1(k1),lamb2(k2));  %UPLDM
                time4 = time4 + toc;
                gm_k3(k1,k2)= gm3;
                gm_k4(k1,k2)= gm4;
                C_k3(k1,k2)= C3;
                C_k4(k1,k2)= C4;
            end
           
        end
    
    gm0_(jj)= gm0;
    gm1_(jj)= gm1;
    gm2_(jj)= gm2;
    gm3_(jj)= max(max(gm_k3));
    gm4_(jj)= max(max(gm_k4));
    % 检查数组中是否全是NaN  
    is_all_nan = isnan(gm_k3);  
    if is_all_nan  
        gm_k3(:) = 0;
    end
    is_all_nan = isnan(gm_k4);  
    if is_all_nan   
        gm_k4(:) = 0;
    end
    [x3,y3]=find(gm_k3==max(max(gm_k3)));xx3=x3(1);yy3=y3(1);
    [x4,y4]=find(gm_k4==max(max(gm_k4)));xx4=x4(1);yy4=y4(1);
    C0_(jj)= C0;
    C1_(jj)= C1;
    C2_(jj)= C2;
    C3_(jj)= C_k3(xx3,yy3);
    C4_(jj)= C_k4(xx4,yy4);
   count = count+1; 
end
Ymat=[gm0_',gm1_',gm2_',gm3_',gm4_'];
h=createfigure(tauval,Ymat);
[Gmeans0,i0]=max(gm0_);
[Gmeans1,i1]=max(gm1_);
[Gmeans2,i2]=max(gm2_);
[Gmeans3,i3]=max(gm3_);
[Gmeans4,i4] = max(gm4_);
opt_tau1=tauval(i1);
opt_tau2=tauval(i2);
opt_tau4=tauval(i4);
opt_C0 = C0_(i0);
opt_C1 = C1_(i1);
opt_C2 = C2_(i2);
opt_C3 = C3_(i3);
opt_C4 = C4_(i4);
end
