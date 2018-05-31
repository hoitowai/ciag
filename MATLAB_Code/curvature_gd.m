function [obj,prm,time_n] = curvature_gd( fct_obj, fct_grd, fct_hess, prm_0, ...
    alpha, max_iter, M, xi, yi, FLAG, TIME_LIMIT)
% CIAG method
% FLAG = 1 -- perform a run over the data samples for initializing the
% gradient / Hessian before we begin 
% FLAG = 0 -- self initialization

obj = zeros(M*max_iter,1); time_n = zeros(M*max_iter,1);
if numel(size(xi)) >= 3
    m = size(yi,2);
else
    m = length(yi);
end

prm = prm_0;
d = length(prm_0);
grd_save = zeros(d,m);
prm_save = zeros(d,m);
u_save = repmat(prm,1,m);

% stupid initialization for the gradient and hessian and parameter
hess_use = zeros(d); 
if FLAG == 1 % if FLAG = 1, then we initialize the variables
for mm = 1 : M
    grd_cur = fct_grd( prm, xi(:, mm), yi(mm) );
    hess_cur = fct_hess( prm, xi(:,mm), yi(mm) );
    hess_use = hess_use + hess_cur/m;
    u_save(:,mm) = hess_cur*prm_save(:,mm);
    grd_save(:,mm) = grd_cur;
end
end

tic;

for iter = 1 : max_iter
    % outer loop
    for mm = 1 : M
        mmk = mod( mm-1, m ) + 1; 
%         mmk = randi( m, 1 );
        if numel(size(xi)) >= 3
            grd_cur = fct_grd( prm, xi(:,:, mmk), yi(:,mmk) );
            hess_cur = fct_hess( prm, xi(:,:,mmk), yi(:,mmk) );
            if iter == 1
                hess_prev = zeros(d);
            else
                hess_prev = fct_hess( prm_save(:,mmk), xi(:,:,mmk), yi(:,mmk) );
            end
        else
            grd_cur = fct_grd( prm, xi(:,mmk), yi(mmk) );
            hess_cur = fct_hess( prm, xi(:,mmk), yi(mmk) );
            if (iter == 1) && (FLAG==0)
                hess_prev = zeros(d);
            else
                hess_prev = fct_hess( prm_save(:,mmk), xi(:,mmk), yi(mmk) );
            end
        end
        u_cur = hess_cur * prm;

        grd_use = sum(grd_save,2)/m + (grd_cur - grd_save(:,mmk))/m;
        hess_use = hess_use + (hess_cur - hess_prev)/m;
        u_use = sum(u_save,2)/m + (u_cur - u_save(:,mmk))/m;
        
        grd_save(:,mmk) = grd_cur; prm_save(:,mmk) = prm; u_save(:,mmk) = u_cur;

        % Let's try Nesterov
        prm = prm + alpha* (u_use - grd_use - hess_use*prm) ;
        
        time_n((iter-1)*M + mm) = toc;
        obj((iter-1)*M + mm) = sum(fct_obj( prm, xi,yi ))/m;
        
        if time_n((iter-1)*M+mm) > TIME_LIMIT
            time_n( (iter-1)*M+mm+1 : end ) = time_n((iter-1)*M+mm);
            obj((iter-1)*M + mm+1 : end) = obj( ((iter-1)*M + mm) );
            break; 
        end
    end
    if time_n((iter-1)*M+mm) > TIME_LIMIT
        break;
    end
end