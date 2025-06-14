% % simulation with input noise
% simulation with input noise
clearvars;
clear;
addpath('AMP');
addpath('VB');
addpath('Channel_model');
addpath('Denoiser');
par.M = 32; % number of observation
beta = 1; % set system ratio beta = K/M;
par.K = beta*par.M;
par.const_size = 4;  % size of constellation
par.Td = 100; % data transmission time
par.Tp = 2^(ceil(log2(par.K)))*2; % pilot transmission time
par.P_dB = 0; % transmit power - set to 1
par.P = 10^(par.P_dB/10);
par.iters = 50;
par.iters_inner = 10; %10
par.ps = 1/par.const_size*ones(par.const_size, 1);
par.trials = 50; %100
opt = 0;
use_estimated_channel = 0;

if par.const_size == 16
    mod = '16QAM';
    par.S = sqrt(par.P/10)*[-3-3j; -1-3j; 3-3j; 1-3j; -3-1j; -1-1j; 3-1j; 1-1j; ...
                            -3+3j; -1+3j; 3+3j; 1+3j; -3+1j; -1+1j; 3+1j; 1+1j];
elseif par.const_size == 4
    mod = 'QPSK';
    par.S = sqrt(par.P/2)*[-1-1j; -1+1j; 1-1j; 1+1j];
end

par.SNR_dB = 0:2:20; % dB
SNR_dB = par.SNR_dB;


results.MF_VB_JED_I.SER = zeros(length(par.SNR_dB),1);
results.LMMSE_VB_JED_I.SER = zeros(length(par.SNR_dB),1);
results.LMMSE.SER = zeros(length(par.SNR_dB),1);
results.GAMP_I.SER = zeros(length(par.SNR_dB),1);



results.MF_VB_JED_I.NMSE = zeros(length(par.SNR_dB),1);
results.LMMSE_VB_JED_I.NMSE = zeros(length(par.SNR_dB),1);
results.GAMP_I.NMSE = zeros(length(par.SNR_dB),1);

% channel setting
alpha = 0; %0.5+1i*0.5;
%alpha = 0.5+1i*0.5;
R = spatial_correlation(par.M, alpha) / par.M;
% R = eye(par.M)/par.M;
par.zeta = ones(par.K, 1);
% C = dlmread('Cov_mat_1064N.txt');
% R = C(:,1:par.M) + 1i*C(:,par.M+1:end);
R_half = sqrtm(R);
R_inv = inv(R);
Rh = zeros(par.M, par.M, par.K);
for i=1:par.K
    Rh(:,:,i) = R/par.zeta(i);
end

% Use QPSK sequence - Set Pp = 1
Pp = 1;
PT = Pp * par.Tp * ones(par.K, 1);
Xp = training_sequence(par.K, par.Tp, par.S(1), Pp);  
par.EVM = -20; % dB
% TX setting
par.tx.noise_sq_mean = 10^(par.EVM/10)*par.P;
par.tx.noise_sq = par.tx.noise_sq_mean * ones(par.K, 1);

for ii=1:length(par.SNR_dB)
    fprintf('SNR %d dB \n', par.SNR_dB(ii))
    
    % RX setting
    N0 = 10^(-par.SNR_dB(ii)/10)*par.P*sum(1./par.zeta)/par.M;
    par.rx.noise_sq = N0;
    
    count = 0;
    X_est = zeros(par.K, par.Td);
    while count < par.trials %((results.MF_VB.SER(ii) < 100) || (count < 20)) && (count < 30)
        count = count + 1
        
        H = R_half/sqrt(2)*(randn(par.M, par.K) + 1j*randn(par.M, par.K))*diag(par.zeta.^-0.5);
        
         % pilot transmission
        N_p = sqrt(N0/2)*(randn(par.M, par.Tp) + 1j*randn(par.M, par.Tp));
        tx_noise_p = diag(sqrt(par.tx.noise_sq/2))*(randn(par.K, par.Tp) + 1i*randn(par.K, par.Tp));
        Yp = H*(Xp+tx_noise_p) + N_p;
              
        % data transmission phase
        data = randi([1, par.const_size], par.K, par.Td);
        X = par.S(data);
        N = sqrt(N0/2)*(randn(par.M, par.Td) + 1j*randn(par.M, par.Td));
        tx_noise = diag(sqrt(par.tx.noise_sq/2))*(randn(par.K, par.Td) + 1i*randn(par.K, par.Td));
        Yd = sqrt(par.P)*H*(X+tx_noise) + N;

        
        [X_est, Q, H_] = MF_VB_JED_I(par, Yd, Yp, Xp, Rh, N0, 'full');
        results.MF_VB_JED_I.SER(ii) = results.MF_VB_JED_I.SER(ii) + sum(sum(X_est ~= X));
        results.MF_VB_JED_I.NMSE(ii) = results.MF_VB_JED_I.NMSE(ii) + norm((H - H_), 'fro')^2/(par.M*par.K);

        [X_est, Q, H_] = LMMSE_VB_JED_I(par, Yd, Yp, Xp, Rh, 'full');
        results.LMMSE_VB_JED_I.SER(ii) = results.LMMSE_VB_JED_I.SER(ii) + sum(sum(X_est ~= X));
        results.LMMSE_VB_JED_I.NMSE(ii) = results.LMMSE_VB_JED_I.NMSE(ii) + norm((H - H_), 'fro')^2/(par.M*par.K);

        % set to lite version for faster, but worse performance
        [X_est, H_, Q, tau_H] =  GAMP_JED_lite_I(par, [Yp,Yd], Xp, Rh, N0, 'lite');
        results.GAMP_I.SER(ii)  = results.GAMP_I.SER(ii) + sum(sum(X_est ~= X));
        results.GAMP_I.NMSE(ii) = results.GAMP_I.NMSE(ii) + norm((H - H_), 'fro')^2/(par.M*par.K);

        % x_est = LMMSE_detector(par, H, Y(:,t), N0);
        % results.LMMSE.SER(ii) = results.LMMSE.SER(ii) + sum(x_est~=X(:,t));
    end
end

results.MF_VB_JED_I.SER = results.MF_VB_JED_I.SER/par.K/par.Td/par.trials;
results.LMMSE_VB_JED_I.SER = results.LMMSE_VB_JED_I.SER/par.K/par.Td/par.trials;
results.GAMP_I.SER = results.GAMP_I.SER/par.K/par.Td/par.trials;


results.MF_VB_JED_I.NMSE = 10*log10(results.MF_VB_JED_I.NMSE/par.trials);
results.LMMSE_VB_JED_I.NMSE = 10*log10(results.LMMSE_VB_JED_I.NMSE/par.trials);
results.GAMP_I.NMSE = 10*log10(results.GAMP_I.NMSE/par.trials);


figure()
semilogy(par.SNR_dB, results.GAMP_I.NMSE, 'm', 'LineWidth', 2);
xlabel('SNR (dB)');
ylabel('Symbol Error Rate');
axis([0, 20, -50, 0]);
legend('GAMP_I','interpreter','none');
grid on;
 
% figure()
% semilogy(par.SNR_dB, results.MF_VB_JED_I.SER, 'b', 'LineWidth', 2);
% hold on
% semilogy(par.SNR_dB, results.LMMSE_VB_JED_I.SER, 'm', 'LineWidth', 2);
% hold on
% semilogy(par.SNR_dB, results.GAMP_I.SER, 'k', 'LineWidth', 2);
% 
% xlabel('SNR (dB)');
% ylabel('Symbol Error Rate');
% axis([0, 30, 1e-4, 1]);
% legend('MF_VB_JED_I','LMMSE_VB_JED_I','GAMP_I','interpreter','none');
% grid on;

filename = sprintf('result/%d_%d_QPSK_JED_iid_channel_20noise.mat', par.M, par.K);
save(filename, 'results', 'par', 'R');

% filename = sprintf('result/%d_%d_16QAM_JED_corr_channel_30noise.mat', par.M, par.K);
% save(filename, 'results', 'par', 'R');
