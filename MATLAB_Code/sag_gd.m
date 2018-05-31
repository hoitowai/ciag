function [obj,prm,time_n] = sag_gd( fct_obj, fct_grd, prm_0, alpha, max_iter,...
    M, xi, yi , mode, FLAG, TIME_LIMIT )
% A.k.a. Stochastic GD
% "mode" controls the switch from SAG to SAGA

obj = zeros(M*max_iter,1); time_n = zeros(M*max_iter,1);
if numel(size(xi)) >= 3
    m = size(yi,2);
else
    m = length(yi);
end
prm = prm_0;
d = length(prm_0);
if (strcmp(mode,'SAGA')) || (strcmp(mode,'SAG'))
    grd_save = zeros(d,m); prm_save = zeros(d,m);
end

tic; cnt_k = 0;
for iter = 1 : max_iter
    prm_prev = prm;
    if strcmp(mode,'SVRG') || strcmp(mode,'SVRGS')
        grd_agg = zeros(d,1); 
        for mm = 1 : m
            mmk = mod(mm-1,m) + 1; 
            if numel(size(xi)) >= 3
                grd_agg = grd_agg + fct_grd( prm, xi(:,:, mmk), yi(:,mmk) );
            else
                grd_agg = grd_agg + fct_grd( prm, xi(:, mmk), yi(mmk) );
            end
        end
    end
    % outer loop
    for mm = 1 : M
        mmk = mod( mm-1, m ) + 1; mmk_old = mod( mm-2, m ) + 1;
        if numel(size(xi)) >= 3
            grd_cur = fct_grd( prm, xi(:,:,mmk), yi(:,mmk) );
        else
            grd_cur = fct_grd( prm, xi(:,mmk), yi(mmk) );
        end
        if strcmp(mode,'SAGA')
            grd_use = sum(grd_save,2)/m + grd_cur - grd_save(:,mmk);
            grd_save(:,mmk) = grd_cur; prm_save(:,mmk) = prm;
        elseif strcmp(mode,'SAG')
            grd_use = sum(grd_save,2)/m + (grd_cur - grd_save(:,mmk))/m;
            grd_save(:,mmk) = grd_cur; prm_save(:,mmk) = prm; 
            cnt_k = cnt_k + 1; beta_k = 1 - ( m / (m + cnt_k) );
        elseif strcmp(mode,'SVRG')
            if numel(size(xi))>=3
                grd_prev = fct_grd( prm_prev, xi(:,:,mmk), yi(:,mmk) );
            else
                grd_prev = fct_grd( prm_prev, xi(:,mmk), yi(mmk) );
            end
            grd_use = grd_agg/m - grd_prev + grd_cur;
        elseif strcmp(mode,'SVRGS') % i also update the aggregated gradient
            if numel(size(xi)) >= 3
                grd_prev = fct_grd( prm_prev, xi(:,:,mmk), yi(:,mmk) );
            else
                grd_prev = fct_grd( prm_prev, xi(:,mmk), yi(mmk) );
            end
            grd_use = grd_agg/m - grd_prev + grd_cur;
            grd_agg = grd_agg - grd_prev + grd_cur;
        end
        prm = prm - alpha*grd_use + FLAG*beta_k*(prm - prm_save(:,mmk_old));
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