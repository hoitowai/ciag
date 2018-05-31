function [obj,prm,time_n] = svrg2_gd( fct_obj, fct_grd, fct_hess, prm_0, alpha, max_iter,...
    M, xi, yi, TIME_LIMIT )
% SVRG2 -- using the full hessians
% alpha = step size
% max_iter = max no of iterations
% M = length of epoch
% xi,yi = input data

obj = zeros(M*max_iter,1); time_n = zeros(M*max_iter,1);
if numel(size(xi)) >= 3
    m = size(yi,2);
else
    m = length(yi);
end
prm = prm_0;
d = length(prm_0);

tic; 
for iter = 1 : max_iter
    prm_prev = prm;
    
    grd_agg = zeros(d,1); hess_agg = zeros(d);
    for mm = 1 : m
        if numel(size(xi)) >= 3
            grd_agg = grd_agg + fct_grd( prm, xi(:,:, mm), yi(:,mm) );
            hess_agg = hess_agg + fct_hess( prm, xi(:,:, mm), yi(:,mm) );
        else
            grd_agg = grd_agg + fct_grd( prm, xi(:, mm), yi(mm) );
            hess_agg = hess_agg + fct_hess( prm, xi(:, mm), yi(mm) );
        end
    end
    
    % outer loop
    for mm = 1 : M
        mmk = randi( m ); % select the function index
        if numel(size(xi)) >= 3
            grd_cur = fct_grd( prm, xi(:,:,mmk), yi(:,mmk) );
        else
            grd_cur = fct_grd( prm, xi(:,mmk), yi(mmk) );
        end
        if numel(size(xi))>=3
            grd_prev = fct_grd( prm_prev, xi(:,:,mmk), yi(:,mmk) );
            hess_prev = fct_hess( prm_prev, xi(:,:,mmk), yi(:,mmk) );
        else
            grd_prev = fct_grd( prm_prev, xi(:,mmk), yi(mmk) );
            hess_prev = fct_hess( prm_prev, xi(:,mmk), yi(mmk) );
        end
        
        grd_use = grd_agg/m + (hess_agg/m)*(prm - prm_prev) + grd_cur...
            - grd_prev - hess_prev*(prm - prm_prev);
        
        prm = prm - alpha*grd_use;
        
        time_n((iter-1)*M + mm) = toc;
        obj((iter-1)*M + mm) = sum(fct_obj( prm, xi,yi ))/m;
        if time_n( (iter-1)*M + mm ) > TIME_LIMIT
            time_n( (iter-1)*M+mm+1 : end ) = time_n((iter-1)*M+mm);
            obj((iter-1)*M + mm+1 : end) = obj( ((iter-1)*M + mm) );
            break;
        end
    end
    if time_n( (iter-1)*M + mm ) > TIME_LIMIT
        break;
    end
end