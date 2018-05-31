function [obj,prm,time_n] = accsvrg_gd( fct_obj, fct_grd, prm_0, alpha, max_iter,...
    M, B, xi, yi, TIME_LIMIT )
% AccSVRG -- a Nesterov's accelerated version of SVRG

obj = zeros(M*B*max_iter,1); time_n = zeros(M*B*max_iter,1);
if numel(size(xi)) >= 3
    m = size(yi,2);
else
    m = length(yi);
end
prm = prm_0;
d = length(prm_0);
mu = 0.85;

tic; 
for iter = 1 : max_iter
    prm_prev = prm;
    
    grd_agg = zeros(d,1); 
    for mm = 1 : m
        if numel(size(xi)) >= 3
            grd_agg = grd_agg + fct_grd( prm, xi(:,:, mm), yi(:,mm) );
        else
            grd_agg = grd_agg + fct_grd( prm, xi(:, mm), yi(mm) );
        end
    end
    
    prm_ex = prm;
    % outer loop
    for mm = 1 : M
        prm_old = prm;
        sel_fun = randperm( m ); % select the function index
        sel_fun = sel_fun(1:B);
        grd_cur = zeros(d,1); grd_prev = zeros(d,1);
        
        if numel(size(xi)) >= 3
            for bb = 1 : B
                mmk = sel_fun(bb);
                grd_cur = grd_cur + fct_grd( prm_ex, xi(:,:,mmk), yi(:,mmk) );
                grd_prev = grd_prev + fct_grd( prm_prev, xi(:,:,mmk), yi(:,mmk) );
            end
        else
            for bb = 1 : B
                mmk = sel_fun(bb);
                grd_cur = grd_cur + fct_grd( prm_ex, xi(:,mmk), yi(mmk) );
                grd_prev = grd_prev + fct_grd( prm_prev, xi(:,mmk), yi(mmk) );
            end
        end
        grd_cur = grd_cur / B; grd_prev = grd_prev / B;
        
        grd_use = grd_agg/m + grd_cur - grd_prev;
        prm = prm_ex - alpha*grd_use;
        
        prm_ex = prm + mu* ( prm - prm_old );
        
        time_n( ((iter-1)*M + mm - 1)*B + 1 : ((iter-1)*M + mm)*B ) = toc;
        
        obj( ((iter-1)*M + mm - 1)*B + 1 : ((iter-1)*M + mm)*B ) = sum(fct_obj( prm, xi,yi ))/m;
        
        
        if time_n( ((iter-1)*M + mm - 1)*B + 1 ) > TIME_LIMIT
            time_n( ((iter-1)*M + mm - 1)*B + 1 : end ) = time_n( ((iter-1)*M + mm - 1)*B + 1 );
            obj( ((iter-1)*M + mm - 1)*B + 1 : end) = obj( ((iter-1)*M + mm - 1)*B + 1 );
            break;
        end
    end
    if time_n( ((iter-1)*M + mm - 1)*B + 1 ) > TIME_LIMIT
        break;
    end
end