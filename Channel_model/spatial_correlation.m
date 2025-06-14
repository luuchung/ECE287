function R = spatial_correlation(M, beta)
% R = beta.^(abs(ones(M,1)*(1:M) - (1:M)'));
R = zeros(M, M);
for i=1:M
    for j=1:M
        if (j>=i)
            R(i,j) = beta^(j-i);
        else
            R(i,j) = conj(R(j,i));
        end
    end
end