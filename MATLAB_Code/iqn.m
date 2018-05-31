function [obj,prm,time_n] = iqn( fct_obj, fct_grd, prm_0, max_iter, M, xi, yi, TIME_LIMIT)
% IQN

obj = zeros(M*max_iter,1);
time_n = zeros(M*max_iter,1);

if numel(size(xi)) >= 3
    m = size(yi,2);
else
    m = length(yi);
end
prm = prm_0;
d = length(prm_0);

tic;
% Initialize the algorithm
B_all = zeros(d,d,m); B_inv = (1/m)*eye(d); for mm = 1:m, B_all(:,:,mm) = eye(d); end
g_all = zeros(d,m); u_all = zeros(d,m);
for mm = 1 : m
    if numel(size(xi))>=3
        u_all(:,mm) = B_all(:,:,mm)*prm; g_all(:,mm) = fct_grd( prm, xi(:,:,mm), yi(:,mm) );
    else
        u_all(:,mm) = B_all(:,:,mm)*prm; g_all(:,mm) = fct_grd( prm, xi(:,mm), yi(mm) );
    end
end
z_all = repmat(prm,1,m);

for iter = 1 : max_iter
    % outer loop
    for mm = 1 : M
        mmk = mod( mm-1, m ) + 1;
%         B_inv = ( sum(B_all,3) )^-1;
        prm = B_inv * ( sum(u_all,2) - sum( g_all,2 ) );
        % get the gradient
        if numel(size(xi))>=3
            g_cur = fct_grd( prm, xi(:,:,mmk), yi(:,mmk) );
        else
            g_cur = fct_grd( prm, xi(:,mmk), yi(mmk) );
        end
        s_tmp = prm - z_all(:,mmk); y_tmp = g_cur - g_all(:,mmk); 
        % BFGS update
        B_old_cur = B_all(:,:,mmk); bs_prod = B_old_cur*s_tmp; 
        B_all(:,:,mmk) = B_old_cur + (y_tmp*y_tmp')/(s_tmp'*y_tmp) ...
            - ( bs_prod )*( bs_prod )' / ( s_tmp'*bs_prod );
        % update my current u_all
        u_all(:,mmk) = B_all(:,:,mmk)*prm; g_all(:,mmk) = g_cur; z_all(:,mmk) = prm;
        % update the B_inv
        U_tmp = B_inv - ( B_inv*(y_tmp*y_tmp')*B_inv ) / (y_tmp'*s_tmp + y_tmp'*B_inv*y_tmp);
        B_inv = U_tmp + ( U_tmp* ( bs_prod ) * ( bs_prod )' * U_tmp ) ...
            / (s_tmp'*bs_prod - ( bs_prod )'*U_tmp*( bs_prod ) );
%         true_inv = (sum(B_all,3))^-1;
        % save the values
        obj((iter-1)*M + mm) = sum(fct_obj( prm, xi,yi ))/m;
        time_n( (iter-1)*M + mm ) = toc; 
        
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