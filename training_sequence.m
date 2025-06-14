function Sp = training_sequence(K, Tp, a, power)
S = [1, 1j; 1, -1j];
for i=1:log2(Tp)-1
    S = [S, S; S, -S];
end
if nargin==3
    Sp = S(1:K,:)*a;
else
    Sp = S(1:K,:)*a/abs(a)*sqrt(power);
end