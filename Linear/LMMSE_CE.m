function [H_est, err_var] = LMMSE_CE(Yp, Xp, Rh, N0)
[K, Tp] = size(Xp);
M = size(Yp, 1);
H_est = zeros(M, K);
err_var = zeros(M, M, K);
% This will work with orthogonal pilots
for i=1:K
    H_est(:,i) = (Tp*eye(M) + N0*inv(Rh(:,:,i)))\Yp*Xp(i,:)';
    err_var(:,:,i) = inv(Tp/N0*eye(M) + inv(Rh(:,:,i)));
end

% For non-orthogonal pilots, 
