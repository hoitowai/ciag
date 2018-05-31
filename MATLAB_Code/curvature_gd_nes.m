function [obj,prm,time_n] = curvature_gd_nes( fct_obj, fct_grd, fct_hess, prm_0, ...
    alpha, max_iter, M, xi, yi, heavy_flag, TIME_LIMIT)
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
mu_k = 0.6;

prm = prm_0;
d = length(prm_0);
grd_save = zeros(d,m);
prm_save = zeros(d,m);
% hess_save = repmat(eye(d),[1 1 m]);
% hess_save = zeros(d,d,m);
u_save = repmat(prm,1,m);
% prm_save = repmat(prm,1,m);

tic;

for iter = 1 : max_iter
    % outer loop
    for mm = 1 : M
        if (mm == 1) && (iter == 1)
            hess_use = zeros(d); prm_old = zeros(d,1);
        end
        mmk = mod( mm-1, m ) + 1; 
       
        prm_use = prm + mu_k*heavy_flag*( prm - prm_old );
        
        if numel(size(xi)) >= 3
            grd_cur = fct_grd( prm_use, xi(:,:, mmk), yi(:,mmk) );
            hess_cur = fct_hess( prm_use, xi(:,:,mmk), yi(:,mmk) );
            if iter == 1
                hess_prev = zeros(d);
            else
                hess_prev = fct_hess( prm_save(:,mmk), xi(:,:,mmk), yi(:,mmk) );
            end
        else
            grd_cur = fct_grd( prm_use, xi(:,mmk), yi(mmk) );
            hess_cur = fct_hess( prm_use, xi(:,mmk), yi(mmk) );
            if iter == 1
                hess_prev = zeros(d);
            else
                hess_prev = fct_hess( prm_save(:,mmk), xi(:,mmk), yi(mmk) );
            end
        end
        u_cur = hess_cur * prm_use;

        grd_use = sum(grd_save,2)/m + (grd_cur - grd_save(:,mmk))/m;
        hess_use = hess_use + (hess_cur - hess_prev)/m;
        u_use = sum(u_save,2)/m + (u_cur - u_save(:,mmk)) / m;
        
        grd_save(:,mmk) = grd_cur; prm_save(:,mmk) = prm_use; u_save(:,mmk) = u_cur;
        
        

%         mu_k = 0.999;
        % Let's try Nesterov
%         prm = prm + alpha* (u_use - grd_use - hess_use*prm)...
%             + mu_k * heavy_flag * (prm - prm_save(:,mmk_old)) ;
        prm_old = prm;
        prm = prm_use + alpha* (u_use - grd_use - hess_use*prm_use);
        
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