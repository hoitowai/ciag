function [obj,prm,time_n] = full_gd( fct_obj, fct_grd, prm_0, alpha, max_iter, xi, yi )

obj = zeros(max_iter,1);
time_n = zeros(max_iter,1);
d = size(prm_0,1);
prm = prm_0;

if numel(size(xi))>=3
    % we are in the quadratic case
    m = size(yi,2);
else
    % we are in the linear classifier case
    m = length(yi);
end
    
tic;
for iter = 1 : max_iter
    grd_tmp = zeros(d,1); 
    for mm = 1 : m
        if numel(size(xi))>=3
            grd_tmp = grd_tmp + fct_grd( prm, xi(:,:,mm), yi(:,mm) )/m;
        else
            grd_tmp = grd_tmp + fct_grd( prm, xi(:,mm), yi(mm) )/m;
        end
    end
    prm = prm - alpha*grd_tmp;
    time_n(iter) = toc;
    obj(iter) = sum(fct_obj( prm, xi,yi ))/m;
end