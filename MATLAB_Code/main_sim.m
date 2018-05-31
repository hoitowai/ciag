% SVM with synthetic data 
close all; clc;

d = 50; m = 500; % d - problem dimension, m - no of data samples

TIME_LIM = 1500;

% Logistic Regression
prm = 2*rand(d,1)-1; b = 2*rand-1;

xi = 2*rand(d,m)-1; % data 
yi = sign(prm'*xi + b); % labeled data
optval = inf;

% Specify the function
flog_eval = @(p,x,y) m*log(1 + exp(-y.* (p'*[x; ones(size(y))]))) + norm(p,2)^2/2;
grad_flog = @(p,x,y) -m*y*[x; ones(size(y))] / (1+exp(y.* (p'*[x; ones(size(y))]))) ...
    + p; 
hessian_flog = @(p,x,y) m*(exp(y.* (p'*[x; ones(size(y))])) / ((1+exp(y.* (p'*[x; ones(size(y))])))^2))...
    * ([x; 1]*[x; 1]') + eye(size(p,1));

% calculate the Lipschitz constant
X_exp = [xi; ones(1,m)];
Lips = max( eig(eye(d+1) + 1* (X_exp*X_exp')) );
prm_0 = zeros(d+1,1);

% Parameters
mul_no = 1;
M = mul_no*m; % no of inner loop iterations
max_no_iter_gd = 2.5e1; 

tic;
%%%%%%%%%%%%%%%% Full GD %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
alpha = 1/Lips; t_s = toc;
[obj_fgd, prm_fgd, time_fgd] = full_gd( flog_eval, grad_flog, prm_0, alpha, max_no_iter_gd, xi, yi );
t_f = toc; t_fgd = t_f - t_s; fprintf('Full GD time: %f \n',t_fgd);

%%%%%%%%%%%%%%%% Full Newton %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
t_s = toc;
[obj_fnewt, prm_fnewt, time_fnewt] = full_newt( flog_eval, grad_flog, hessian_flog, ...
 prm_0, max_no_iter_gd, xi, yi , 0, TIME_LIM );
t_f = toc; t_newt = t_f - t_s; fprintf('Full Newton time: %f \n',t_newt);

max_no_iter_gd_eff = max_no_iter_gd / 1;
max_no_iter_Inc = M*max_no_iter_gd_eff;

%%%%%%%%%%%%%%%%%%%%%% CIAG Grad %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
t_s = toc;
[obj_curv, prm_curv, time_curv] = curvature_gd( flog_eval, grad_flog, hessian_flog, ...
    prm_0, 1/Lips, max_no_iter_gd_eff, M, xi, yi, 0, TIME_LIM);
t_f = toc; t_cigd = t_f - t_s; fprintf('CIAG time: %f \n',max(time_curv));

t_s = toc;
[obj_curv2, prm_curv2, time_curv2] = curvature_gd_nes( flog_eval, grad_flog, hessian_flog, ...
    prm_0, 1/Lips, max_no_iter_gd_eff, M, xi, yi, 1, TIME_LIM);
t_f = toc; t_cigd2 = t_f - t_s; fprintf('ACIAG (Nesterov) time: %f \n',max(time_curv2));


%%%%%%%%%%%%%%%%%%% NIM %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
alpha = 1; % step size
[obj_incn, prm_incn,time_incn] = nim( flog_eval, grad_flog,...
    hessian_flog, prm_0, max_no_iter_gd_eff, M, xi, yi, alpha, TIME_LIM);
fprintf('NIM time: %f \n',max(time_incn));

%%%%%%%%%%%%%%%%%%%% IQN %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
t_s = toc;
[obj_bfgs, prm_bfgs,time_bfgs] = iqn( flog_eval, grad_flog, ...
    prm_0, max_no_iter_gd_eff, M, xi, yi, TIME_LIM);
t_f = toc; t_bfgs = t_f - t_s; fprintf('IQN time: %f \n',max(time_bfgs));

% %%%%%%%%%%%%%%%%%% SVRG2 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
max_no_iter_gd_eff_sag = 5*max_no_iter_gd_eff;

alpha = 1/Lips; % step size for SVRG2
t_s = toc;
[obj_svrg2, prm_svrg2, time_svrg2] = svrg2_gd( flog_eval, grad_flog, hessian_flog, prm_0, alpha,...
    10*max_no_iter_gd, 0.1*M, xi, yi, TIME_LIM);
t_f = toc; t_svrg2 = t_f - t_s; fprintf('SVRG2 time: %f \n',max(time_svrg2));

% %%%%%%%%%%%%%%%%%% SAG & Friends %%%%%%%%%%%%%%%%%%%%%%%%%%
alpha = (50/Lips)/m;
t_s = toc;
[obj_iag, prm_iag, time_iag] = sag_real_gd( flog_eval, grad_flog, prm_0, alpha,...
    max_no_iter_gd_eff_sag, M,xi, yi, 1, TIME_LIM);
t_f = toc; t_iag = t_f - t_s; fprintf('IAG time: %f \n',max(time_iag));

t_s = toc;
[obj_sag, prm_sag, time_sag] = sag_real_gd( flog_eval, grad_flog, prm_0, alpha,...
    max_no_iter_gd_eff_sag, M,xi, yi, 0, TIME_LIM);
t_f = toc; t_sag = t_f - t_s; fprintf('SAG time: %f \n',max(time_sag));

alpha = (50/Lips)/m;
t_s = toc; B = 5;
[obj_asvrg, prm_asvrg, time_asvrg] = accsvrg_gd( flog_eval, grad_flog, prm_0, alpha,...
    max_no_iter_gd_eff_sag / 5, M, B, xi, yi, TIME_LIM);
t_f = toc; t_asvrg = t_f - t_s; fprintf('ASVRG time: %f \n',max(time_asvrg));

% estimate the optimal objective value
optval = min([optval;obj_fnewt;obj_curv;obj_curv2;obj_bfgs]);

%% Plotting
figure; semilogy( 1:floor(length(obj_curv)/m), obj_curv(1:m:end)-optval,...
    1:floor(length(obj_curv2)/m), obj_curv2(1:m:end)-optval,...
    1:floor(length(obj_svrg2)/m), obj_svrg2(1:m:end)-optval,...
    1:floor(length(obj_sag)/m), obj_sag(1:m:end)-optval,...
    1:floor(length(obj_asvrg)/m), obj_asvrg(1:m:end)-optval,...
    1:floor(length(obj_iag)/m), obj_iag(1:m:end)-optval,...
    1:floor(length(obj_incn)/m), obj_incn(1:m:end)-optval,...
    1:floor(length(obj_bfgs)/m), obj_bfgs(1:m:end)-optval,...
    'linewidth',2 ); 
set(gca,'fontsize',18)
legend('CIAG', 'A-CIAG (Nes)', 'SVRG2', 'SAG', 'A-SVRG', 'IAG','NIM','IQN'); 
ylabel('Optimality gap'); xlabel('#Effective passes over the data');
grid on
title(['m = ' num2str(m) ', d = ' num2str(d)]);
xlim([1,50])
