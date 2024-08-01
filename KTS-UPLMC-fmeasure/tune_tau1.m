 function [opt_tau4,...
        acc_ldm,acc_upldm,...
        time3,time4,...
        opt_C3,opt_C4,...
        AUC3,Sensitivity3,Specificity3,Fmeasure3,Gmeans3,...
        AUC4,Sensitivity4,Specificity4,Fmeasure4,Gmeans4]= tune_tau1(Ctrain,dtrain,Ctest,dtest,C,kernel,p1,lamb1,lamb2)
tauval= -1:0.1:1;
fmeasure_k3 = zeros(length(lamb1),length(lamb2));fmeasure_k4 = zeros(length(lamb1),length(lamb2));
C_k3 = zeros(length(lamb1),length(lamb2));C_k4 = zeros(length(lamb1),length(lamb2));
fmeasure3_ = zeros(1,length(tauval));fmeasure4_ = zeros(1,length(tauval));count=0;

%%
for jj=1:length(tauval)
        fprintf('%3.0f steps remaining...\n',length(tauval)-count);
        tau= tauval(jj);
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
    C3_(jj)= C_k3(xx3,yy3);
    C4_(jj)= C_k4(xx4,yy4);
   count = count+1; 
end
% Ymat=[fmeasure0_',fmeasure1_',fmeasure2_',fmeasure3_',fmeasure4_'];
% h=createfigure(tauval,Ymat);
[Fmeasure3,i3]=max(fmeasure3_);
[Fmeasure4,i4] = max(fmeasure4_);
opt_tau4=tauval(i4);
opt_C3 = C3_(i3);
opt_C4 = C4_(i4);
end
