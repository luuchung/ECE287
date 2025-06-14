function x_est = LMMSE_DD(par, H, y, noise_var)
if length(noise_var) == 1
    x_ = (H'*H + noise_var*eye(par.K))\H'*y;
else
    x_ = (H'*H + noise_var)\H'*y;
end
x_est = zeros(par.K, 1);
for i=1:par.K
    [~, idx] = min(abs(x_(i) - par.S));
    x_est(i) = par.S(idx);
end