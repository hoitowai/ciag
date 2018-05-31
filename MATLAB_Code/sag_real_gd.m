function [obj,prm,time_n] = sag_real_gd( fct_obj, fct_grd, prm_0, alpha, max_iter,...
    M, xi, yi , SEL, TIME_LIMIT )
% SAG or IAG
% SEL = 1 -- cyclical selection (IAG)
% SEL = 0 -- random selection (SAG)

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

tic; 
for iter = 1 : max_iter
    % outer loop
    for mm = 1 : M
        if (SEL)
            mmk = mod( mm-1, m ) + 1; % cyclic
        else
            mmk = randi(m); % random
        end
        if numel(size(xi)) >= 3
            grd_cur = fct_grd( prm, xi(:,:,mmk), yi(:,mmk) );
        else
            grd_cur = fct_grd( prm, xi(:,mmk), yi(mmk) );
        end
        grd_use = sum(grd_save,2)/m + (grd_cur - grd_save(:,mmk))/m;
        grd_save(:,mmk) = grd_cur; prm_save(:,mmk) = prm;
            
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