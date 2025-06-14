% Implementation with concatenated pilot and data received signals
function [X_est, H_, q, tau_H] = GAMP_JED_lite_I(par, Y, Xp, C, N0, opt)
if nargin < 5
    disp('Error! Not enough arguments!');
elseif nargin == 5
    opt = 'full';
end
    
K = par.K;
M = par.M;
Tp = par.Tp;
Td = par.Td;

% uncomment for random initialization
% H_ = zeros(M,K); %(1/sqrt(M*2))*(randn(M, K) + 1i*randn(M, K));
% tau_H = rand(M, K);

% uncomment for using LS solution
PT = sum(Xp.*conj(Xp), 2);
H_ = Y(:,1:Tp)*Xp'/diag(PT);
tau_H = rand(M, K);

X_ = [zeros(K, Tp), zeros(K, Td)];
% norm(H - H_)
count = 0;
q = zeros(par.const_size, K, Td+Tp);
Cov = zeros(M, M, K);
S = zeros(M, Tp+Td);
S_temp = zeros(M, Tp+Td);
X_temp = [zeros(K, Tp), zeros(K, Td)];
H_temp = zeros(M, K);
tau_S = zeros(M, Tp+Td);
tau_X = [zeros(K, Tp), ones(K, Td)];

diff = 1;
beta = .8;
decay = 1;
tx_noise_sq = par.tx.noise_sq_mean * ones(K, 1);
while diff > 1e-6 && count < par.iters
    count = count + 1;
    X_old = X_;
%     H_old = H_;
    % Output linear step
    tau_P = (H_.*conj(H_)) * tau_X + tau_H * (X_.*conj(X_));
    if strcmp(opt, 'lite')
        tau_P = mean(tau_P(:)) * ones(size(tau_P));
    end
    P = H_*X_ - tau_P.*S;
    tau_P = tau_P + tau_H * tau_X;
    tau_P = min(max(tau_P, 1e-6), 1);

    % Output nonlinear step
    for t=1:Tp+Td
        [S_temp(:,t), tau_S(:,t)] = denoiser_Gaussian(P(:,t), tau_P(:,t), Y(:,t), N0); 
    end
    if strcmp(opt, 'lite')
        tau_S = mean(tau_S(:))*ones(size(tau_S));
    end
    S = (1-beta)*S + beta*(S_temp - P)./tau_P;  % onsager correction
    tau_S = (1 - min(tau_S./tau_P, 0.99))./tau_P;
    
    % Input linear step - data
    tau_R = 1 ./ ((H_.*conj(H_))' * tau_S); % + tau_H'*tau_S); %(tau_Sp*XX');
    tau_R = min(tau_R, 1e6); %(tau_R > 1e6) = 1e6;
    gain_R = 1 - tau_R.*(tau_H'*tau_S);
    gain_R = min(1, max(0, gain_R));
    if strcmp(opt, 'lite')
        tau_R = mean(tau_R(:)) * ones(size(tau_R));
    end
    R = X_.*gain_R + tau_R.*(H_'*S);
    
    %  Input nonlinear step - data estimation
    for t=1:Tp
        for i=1:K
            [X_temp(i,t), tau_X(i,t), q(:,i,t)] = denoiser_GM(R(i,t), tau_R(i,t), tx_noise_sq(i), Xp(i, t), par.ps);
                denoiser_discrete(R(i,t), tau_R(i,t), par.S, par.ps);
        end
    end

    for t=1:Td
        for i=1:K
            [X_temp(i,t+Tp), tau_X(i,t+Tp), q(:,i,t+Tp)] = ...
                denoiser_GM(R(i,t+Tp), tau_R(i,t+Tp), tx_noise_sq(i), par.S, par.ps);
        end
    end

    X_ = (1-beta)*X_ + beta*X_temp;
    
    % Input linear step - channel 

    tau_Q = 1 ./ (tau_S * (X_.*conj(X_))'); % + tau_S*tau_X');
    %

    tau_Q = min(tau_Q, 1e6); %(tau_Q > 1e6) = 1e6;
    gain_Q = 1 - tau_Q.*(tau_S*tau_X');
    gain_Q = min(1, max(0, gain_Q));
    if strcmp(opt, 'lite')
        tau_Q = mean(tau_Q(:)) * ones(size(tau_Q));
    end
    Q = H_.*gain_Q + tau_Q.*(S*X_');

    % Input nonlinear step - channel estimation
    for i=1:K
        Cov(:,:,i) = inv(inv(C(:,:,i)) + diag(1./tau_Q(:,i)));
        H_temp(:,i) = Cov(:,:,i)*(Q(:,i) ./ tau_Q(:,i));
        tau_H(:, i) = diag(Cov(:,:,i));
    end
    H_ = (1-beta)*H_ + beta*H_temp;
    diff = norm(X_ - X_old, 'fro')/sqrt(Td);
    beta = beta*decay;
end
X_est = zeros(K, Td);
for i=1:K
    for t=1:Td
        [~, idx] = max(q(:,i,t+Tp));
        X_est(i,t) = par.S(idx);
    end
end
end