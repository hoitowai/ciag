function [obj,prm, time_n] = full_newt( fct_obj, fct_grd, fct_hess, prm_0,...
    max_iter, xi, yi , damp, TIME_LIMIT )

obj = zeros(max_iter,1); 
time_n = zeros(max_iter,1);
d = size(prm_0,1);

beta = 0.5; % line search parameter
alpha = 0.49;

if numel(size(xi))>=3
    % we are in the quadratic case
    m = size(yi,2);
else
    % we are in the linear classifier case
    m = length(yi);
end
prm = prm_0;
tic;
for iter = 1 : max_iter
    grd_tmp = zeros(d,1); hess_tmp = zeros(d);
    for mm = 1 : m
        if numel(size(xi)) >= 3
            grd_tmp = grd_tmp + fct_grd( prm, xi(:,:,mm), yi(:,mm) )/m;
            hess_tmp = hess_tmp + fct_hess( prm, xi(:,:,mm), yi(:,mm) )/m;
        else
            grd_tmp = grd_tmp + fct_grd( prm, xi(:,mm), yi(mm) )/m;
            hess_tmp = hess_tmp + fct_hess( prm, xi(:,mm), yi(mm) )/m;
        end
    end
    newt_dir = pinv(hess_tmp)*grd_tmp;
    if damp ==  1
        % enter the line search phase
        if iter == 1
            obj_old = inf;
        else
            obj_old = obj(iter-1);
        end
        lambda_x = newt_dir'*hess_tmp*newt_dir;
        t = 1;
        while ( sum(fct_obj( prm - t*newt_dir, xi,yi ))/m - obj_old > -alpha*t*lambda_x )
            t = beta*t;
        end
        prm = prm - t * newt_dir;
    else
        prm = prm - newt_dir;
    end
    time_n(iter) = toc;
    obj(iter) = sum(fct_obj( prm, xi,yi ))/m;
    
%     norm( grd_tmp )
    
    if time_n(iter) > TIME_LIMIT
        time_n(iter+1:end) = time_n(iter);
        obj(iter+1:end) = obj(iter);
        break;
    end
end