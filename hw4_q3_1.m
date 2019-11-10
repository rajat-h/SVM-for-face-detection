clc;
clear;
[trD, trLb, valD, valLb, trRegs, valRegs] = HW4_Utils.getPosAndRandomNeg();
C = 0.1;
x = trD;
y = trLb;
xm = mean(x); xs = std(x);                  % Mean and Std. Dev.
x = (x - xm(ones(size(x,1),1),:))./xs(ones(size(x,1),1),:);
x(:,178) = 0;
x(:,193) = 0;
x = x';
svm.D = size(x,2);
svm.L = size(x,1);  
H = zeros(svm.L,svm.L);
for i=1:svm.L
    for j=1:svm.L
        H(i,j) = x(i,:)*x(j,:)'*y(i)*y(j); %H(i,j) = y(i)*y(j)*(x(i,:)*x(j,:)'+1)^2; 
    end
end  
f = -ones(svm.L,1);
A = [];
b = [];
Aeq = y';
beq = 0;
LB = zeros(svm.L,1);
UB = C*ones(svm.L,1);
options = optimset('MaxIter',1000,'LargeScale','off');
[alpha, obj] = quadprog(H,f,A,b,Aeq,beq,LB,UB,[],options);
svm.sv = find(alpha>0.0001);
svm.ns = length(svm.sv);
svm.w = sum(((alpha.*y)*ones(1,svm.D)).*x)';
svm.b = y(svm.sv(1)) - sum((alpha.*y).*(x*x(svm.sv(1),:)'));

%validation data
vx = valD;
vxm = mean(vx); vxs = std(vx);                  % Mean and Std. Dev.
vx = (vx - vxm(ones(size(vx,1),1),:))./vxs(ones(size(vx,1),1),:);
vx = vx';
vy = valLb;
svm.L = size(vx,1);
y_est = sign(vx*svm.w+svm.b);
prc = (sum(y_est(:)'== vy(:)')/svm.L)*100;
Co = confusionmat(vy,y_est);
disp([fprintf('\n'), 'Percentage of samples correctly classified: ', ...
    num2str(round(prc)), '%']);

%result generation
outFile = 'D:\Matlab_prog\Matlab\bin\hw4\rsltFile';
HW4_Utils.genRsltFile(svm.w, svm.b, 'val', outFile);
[ap, prec, rec] = HW4_Utils.cmpAP(outFile, 'val');
plot(rec,prec,'-', 'LineWidth', 3);
grid;
xlabel 'recall'
ylabel 'precision'
title('Precision-recall curve');
axis([0 1 0 1])