 function [acc_svm,...
        time0,...
        opt_C0,...
        AUC0,Sensitivity0,Specificity0,Fmeasure0,Gmeans0]= tune_tau2(Ctrain,dtrain,Ctest,dtest,C,kernel,p1)

%%
time0=0;
tic
[acc_svm,AUC0,Sensitivity0,Specificity0,fmeasure0,Gmeans0,C0] = Unified_pin_svm(Ctrain, dtrain, Ctest,dtest, kernel, 0,C,p1);  %SVM
time0 = time0 + toc;
tic
% Ymat=[fmeasure0_',fmeasure1_',fmeasure2_',fmeasure3_',fmeasure4_'];
% h=createfigure(tauval,Ymat);
Fmeasure0=fmeasure0;
opt_C0 = C0;
end
