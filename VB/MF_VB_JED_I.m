% opt_lite: default as 'lite'
% opt_lite = None: conventional QVB
function [X_est, Q, H_] = MF_VB_JED_I(par, Yd, Yp, Xp, C, N0, opt_lite)
K = par.K;
M = par.M;
Td = par.Td;
Tp = par.Tp;

X_ = zeros(K, Td);
X_p = zeros(K, Tp);
Q = zeros(par.const_size, K, Td);
Q_p = zeros(K, Tp);

PT = sum(Xp.*conj(Xp), 2);
H_ = Yp*Xp'/diag(PT); %zeros(M, K);
tx_noise_sq = par.tx.noise_sq_mean * ones(K, 1);

simple = 1;
for i = 1:par.K
    if ~isdiag(C(:,:,i)) || all(abs(diag(C(:,:,i)) - C(1,1,i)) > 1e-8)
        simple = 0;
        break;
    end
end
if simple == 1
    sigma_h = squeeze(C(1,1,:));
    trace_H = M * sigma_h;
else
    Sigma_H = C;
    trace_H = arrayfun(@(i)trace(C(:,:,i)), 1:size(C,3))';
    C_inv = pageinv(C);
end

diff = 1;
count = 0;

% initialization

G = ones(K, Td);
G_p = ones(K, Tp);
Z = zeros(K, Td);
Z_p = zeros(K, Tp);
norm_H = sum(H_.*conj(H_),1)';

Ep = Yp - H_*X_p;
Ed = Yd - H_*X_;
Xd_2 = abs(X_).^2 + G;
Xp_2 = abs(X_p).^2 + G_p;

while diff > 1e-6 && count < par.iters
    count = count + 1;
    X_old = X_;
    
    if strcmp(opt_lite, 'full')
        % Need to check these equations
        %gamma_p = M*Tp/(norm(Ep, 'fro')^2 + trace_H'*PT);

        gamma_p = M*Tp/(norm(Ep, 'fro')^2 + sum(Xp_2'*trace_H) + sum(G_p'*norm_H));
        gamma_d = Td./(sum(Ed.*conj(Ed), 1)' + Xd_2'*trace_H + G'*norm_H);

        %gamma_p = M*Tp/(norm(Ep, 'fro')^2 + sum(Xp_2'*trace_H) + sum(G_p'*norm_H));
        %gamma_d = M*Td/(norm(Ed, 'fro')^2 + sum(Xd_2'*trace_H) + sum(G'*norm_H)) * ones(Td, 1);
        
        %gamma_d = M/(norm(Ed, 'fro')^2 + sum(Xd_2'*trace_H) + sum(G'*norm_H)) * ones(Td, 1);
    elseif strcmp(opt_lite, 'None')
        gamma_p = 1/N0;
        gamma_d = 1/N0 * ones(Td, 1);
    else
        gamma_p = M*Tp/(norm(Ep, 'fro')^2 + trace_H'*PT);
        gamma_d = M*Td/(norm(Ed, 'fro')^2 + sum(Xd_2'*trace_H) + sum(G'*norm_H)) * ones(Td, 1);
    end
       
     % Update X_d
    Sigma_tilde = 1./((norm_H + trace_H)*gamma_d');
    for ii=1:3
    for i=1:K
        Z(i,:) = X_(i,:) + H_(:,i)'*Ed/norm_H(i);
        for t=1:Td
            Z_tilde = Z(i,t)*norm_H(i)/(norm_H(i) + trace_H(i));

            % denoise for x
            Q(:,i,t) = par.ps .* exp(-abs(Z_tilde - par.S).^2/(Sigma_tilde(i,t) + tx_noise_sq(i)));
            Q(:,i,t) = Q(:,i,t)/sum(Q(:,i,t));

            var_x = Sigma_tilde(i,t) * tx_noise_sq(i) / (Sigma_tilde(i,t) + tx_noise_sq(i));
            mean_x = (par.S*Sigma_tilde(i,t) + Z_tilde*tx_noise_sq(i)) / (Sigma_tilde(i,t) + tx_noise_sq(i));
            
            x_hat = sum(Q(:,i,t) .* mean_x);
            G(i,t) = var_x + sum(Q(:,i,t) .* abs(mean_x - x_hat).^2);  % error variance
            Ed(:,t) = Ed(:,t) + H_(:,i)*(X_(i,t) - x_hat);
            X_(i,t) = x_hat;
        end
    end
    end
    Xd_2 = abs(X_).^2 + G;
    
    % Update X_p
    Sigma_tilde_p = 1./((norm_H + trace_H)*gamma_p');
    for ii=1:3
    for i=1:K
        Z_p(i,:) = X_p(i,:) + H_(:,i)'*Ep/norm_H(i);
        for t=1:Tp
            Z_tilde_p = Z_p(i,t)*norm_H(i)/(norm_H(i) + trace_H(i));
            % denoise for x
            % Q_p(:,i,t) = par.ps .* exp(-abs(Z_tilde_p - par.S).^2/(Sigma_tilde_p(i) + tx_noise_sq(i)));
            % Q_p(:,i,t) = Q_p(:,i,t)/sum(Q_p(:,i,t));

            Q_p(i,t) = exp(-abs(Z_tilde_p - Xp(i, t)).^2/(Sigma_tilde_p(i) + tx_noise_sq(i)));
            Q_p(i,t) = Q_p(i,t)/sum(Q_p(i,t));

            var_x_p = Sigma_tilde_p(i) * tx_noise_sq(i) / (Sigma_tilde_p(i) + tx_noise_sq(i));
            %mean_x_p = (par.S*Sigma_tilde_p(i) + Z_tilde_p*tx_noise_sq(i)) / (Sigma_tilde_p(i) + tx_noise_sq(i));
            mean_x_p = (Xp(i, t)*Sigma_tilde_p(i) + Z_tilde_p*tx_noise_sq(i)) / (Sigma_tilde_p(i) + tx_noise_sq(i));
            

            x_p_hat = sum(Q_p(i,t) .* mean_x_p);

            G_p(i,t) = var_x_p + sum(Q_p(i,t) .* abs(mean_x_p - x_p_hat).^2);  % error variance
            Ep(:,t) = Ep(:,t) + H_(:,i)*(X_p(i,t) - x_p_hat);

            X_p(i,t) = x_p_hat;
        end
    end
    end
    Xp_2 = abs(X_p).^2 + G_p;
    
     % Update H_
%     for ii=1:3
    for i=1:K
        gamma_i = sum(Xp_2(i,:))*gamma_p + Xd_2(i,:)*gamma_d;
        k_i = (1 - G(i,:)*gamma_d/gamma_i)*H_(:,i) + ...
            (gamma_p*Ep*X_p(i,:)' + Ed*(X_(i,:)'.*gamma_d))/gamma_i;
        if simple
            sigma_h(i) = 1/(gamma_i + 1/C(1,1,i));
            H_hat = gamma_i*sigma_h(i)*k_i;
            trace_H(i) = M*sigma_h(i);
        % else
        %     Sigma_H(:,:,i) = pageinv(gamma_i*eye(M) + C_inv);
        %     H_hat = gamma_i*Sigma_H(:,:,i)*k_i;
        %     trace_H(i) = real(trace(Sigma_H(:,:,i)));
        else  
              Sigma_H = pageinv(gamma_i*eye(M) + C_inv);
              H_hat = gamma_i*Sigma_H(:,:,i)*k_i;
              trace_H(i) = real(trace(Sigma_H(:,:,i)));
        end
        % Sigma_H = pageinv(gamma_i*eye(M) + C_inv);
        % H_hat = gamma_i*Sigma_H(:,:,i)*k_i;
        % trace_H(i) = real(trace(Sigma_H(:,:,i)));

        Ep = Ep + (H_(:,i) - H_hat)*X_p(i,:);
        Ed = Ed + (H_(:,i) - H_hat)*X_(i,:);
        H_(:,i) = H_hat;
    end
%     end    
    norm_H = sum(H_.*conj(H_), 1)';
    diff = norm(X_old - X_, 'fro')/sqrt(Td);
end
X_est = zeros(K, Td);
for i=1:K
    for t=1:Td
        [~, idx] = max(Q(:,i,t));
        X_est(i,t) = par.S(idx);
    end
end
% count