function [obj,prm,time_n] = inc_gd( fct_obj, fct_grd, prm_0, alpha, max_iter, M, xi, yi )
% A.k.a. Stochastic GD

obj = zeros(M*max_iter,1);
time_n = zeros(M*max_iter,1);
if numel(size(xi)) >= 3
    m = size(yi,2);
else
    m = length(yi);
end
prm = prm_0;
tic;
for iter = 1 : max_iter
    % outer loop
    for mm = 1 : M
        mmk = mod( mm-1, m ) + 1;
        if numel(size(xi)) >= 3
            prm = prm - (alpha/(iter))*fct_grd( prm, xi(:,:,mmk), yi(:,mmk) );
        else
            prm = prm - (alpha/(iter))*fct_grd( prm, xi(:,mmk), yi(mmk) );
        end
        obj((iter-1)*M + mm) = sum(fct_obj( prm, xi,yi ))/m;
        time_n((iter-1)*M + mm) = toc;
    end
end