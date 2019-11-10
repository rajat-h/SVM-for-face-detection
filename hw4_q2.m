clc;
clear;
load q2_1_data.mat;
x = trD;
y = trLb;
xm = mean(x); xs = std(x);                  % Mean and Std. Dev.
x = (x - xm(ones(size(x,1),1),:))./xs(ones(size(x,1),1),:);
x(:,178) = 0;
x(:,193) = 0;
[w, b, alpha, objective_function] = compute_svm(x, y);
sv = find(alpha>0.0001);
ns = length(sv);
vx = valD;
vxm = mean(vx); vxs = std(vx);                  % Mean and Std. Dev.
vx = (vx - vxm(ones(size(vx,1),1),:))./vxs(ones(size(vx,1),1),:);
vx = vx';
vy = valLb;
L = size(vx,1);
y_est = sign(vx*w+b);
prc = (sum(y_est(:)'== vy(:)')/L)*100;
Co = confusionmat(vy,y_est);
disp([fprintf('\n'), 'Percentage of samples correctly classified: ', ...
    num2str(round(prc)), '%']);