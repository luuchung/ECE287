function x_est = LS_detec(par, y)

 x_ = y;

x_est = zeros(par.K, 1);
for i=1:par.K
    [~, idx] = min(abs(x_(i) - par.S));
    x_est(i) = par.S(idx);
end