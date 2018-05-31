function [obj,prm,time_n] = nim( fct_obj, fct_grd, fct_hess, ...
    prm_0, max_iter, M, xi, yi, alpha, TIME_LIMIT)
% NIM

obj = zeros(M*max_iter,1); time_n = zeros(M*max_iter,1);
if numel(size(xi)) >= 3
    m = size(yi,2);
else
    m = length(yi);
end

prm = prm_0;
d = length(prm_0);

grd_save = zeros(d,m);
u_save = repmat(prm,1,m);
prm_save = zeros(d,m);

tic;
for iter = 1 : max_iter
    % outer loop
    for mm = 1 : M
        if ((iter==1) && (mm==1))
            hess_use = eye(d);
        end
        
        mmk = mod( mm-1, m ) + 1;
        if numel(size(xi)) >= 3
            grd_cur = fct_grd( prm, xi(:,:, mmk), yi(:,mmk) );
            hess_cur = fct_hess( prm, xi(:,:,mmk), yi(:,mmk) );
            if iter == 1
                hess_prev = eye(d);
            else
                hess_prev = fct_hess( prm_save(:,mmk), xi(:,:,mmk), yi(:,mmk) );
            end
        else
            grd_cur = fct_grd( prm, xi(:,mmk), yi(mmk) );
            hess_cur = fct_hess( prm, xi(:,mmk), yi(mmk) );
            if iter == 1 
                hess_prev = eye(d);
            else
                hess_prev = fct_hess( prm_save(:,mmk), xi(:,mmk), yi(mmk) );
            end
        end
        u_cur = hess_cur * prm;
        
        grd_use = sum(grd_save,2)/m + (grd_cur - grd_save(:,mmk))/m;
        hess_use = hess_use + (hess_cur - hess_prev)/m;
        u_use = sum(u_save,2)/m + (u_cur - u_save(:,mmk)) / m;
        
        grd_save(:,mmk) = grd_cur; prm_save(:,mmk) = prm; u_save(:,mmk) = u_cur;
        
        prm = prm + alpha* ( (hess_use)^-1 * ( u_use - grd_use ) - prm ); 
        % (1/(1+0.2*iter))*prm + (1-(1/(1+0.2*iter))) * 
        time_n((iter-1)*M + mm) = toc;
        obj((iter-1)*M + mm) = sum(fct_obj( prm, xi,yi ))/m;
        
        if time_n((iter-1)*M+mm) > TIME_LIMIT
            time_n( (iter-1)*M+mm+1 : end ) = time_n((iter-1)*M+mm);
            obj((iter-1)*M + mm+1 : end) = obj( ((iter-1)*M + mm) );
            break;
        end
%         obj_inst = sum(fct_obj( prm, xi,yi ))/m;
    end
    if time_n((iter-1)*M+mm) > TIME_LIMIT
        break;
    end
end