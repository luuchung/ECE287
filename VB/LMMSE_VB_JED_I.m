function [X_est, Q, H_] = LMMSE_VB_JED_I(par, Yd, Yp, Xp, C, opt_lite)
if strcmp(opt_lite, 'full')
    lite = 0;
else
    lite = 1;
end
K = par.K;
M = par.M;
Td = par.Td;
Tp = par.Tp;
X_ = zeros(K, Td);
X_p = zeros(K, Tp);
Q = zeros(par.const_size, K, Td);
Q_p = zeros(K, Tp);
% [H_, Sigma_H, ~, ~] = VB_CE_FB_real(par, Yp, Yp_l, Yp_u, Xp, R, zeta, N0, delta, opt_gamma);
PT = sum(Xp.*conj(Xp), 2);
H_ = Yp*Xp'/diag(PT); %zeros(M, K);


Sigma_H = C;
trace_H = arrayfun(@(i)trace(C(:,:,i)), 1:size(C,3))';
C_inv = pageinv(C);



diff = 1;
count = 0;
% initialization

G = ones(K, Td);
G_p = ones(K, Tp);
% Z = zeros(K, Td);
W = zeros(M, M, Td);
Ep = Yp - H_*X_p;
Ed = Yd - H_*X_;
Xd_2 = abs(X_).^2 + G;
Xp_2 = abs(X_p).^2 + G_p;
T1 = zeros(K, Td);
T2 = zeros(K, Td);

T1_p = zeros(K, Tp);
T2_p = zeros(K, Tp);
tx_noise_sq = par.tx.noise_sq_mean * ones(K, 1);

while diff > 5e-5 && count < par.iters
    count = count + 1;
    X_old = X_;
    gamma_p = M*Tp/(norm(Ep, 'fro')^2 + trace_H'*PT);
    
    if lite
        temp = 0;
        for i=1:K
            temp = temp + sum(Xd_2(i,:)) * Sigma_H(:,:,i);
        end
        W = Td * inv(norm(Ed,'fro')^2/M*eye(M)+ ...
            temp + H_*diag(sum(G, 2))*H_');
    else
        for t=1:Td
            temp = 0;
            for i=1:K
                temp = temp + Xd_2(i,t) * Sigma_H(:,:,i);
            end
            W(:,:,t) = inv(norm(Ed(:,t))^2/M*eye(M)+ ...
               temp + H_*diag(G(:,t))*H_');
        end
    end
    
    % Update X_
    for i=1:K
        if lite
            T1(i,:) =  H_(:,i)'*W*H_(:,i);
            T2(i,:) = real(T1(i,:) + trace(W*Sigma_H(:,:,i)));
        else
            for t=1:Td
                T1(i,t) = H_(:,i)'*W(:,:,t)*H_(:,i);
                T2(i,t) = real(T1(i,t)) + trace(W(:,:,t)*Sigma_H(:,:,i));
            end
        end
    end
    
    for ii=1:3
    for i=1:K
        for t=1:Td
            if lite
                Z_tilde = (T1(i,t)*X_(i,t) + H_(:,i)'*W*Ed(:,t))/T2(i,t);
            else
                Z_tilde = (T1(i,t)*X_(i,t) + H_(:,i)'*W(:,:,t)*Ed(:,t))/T2(i,t);
            end
            Q(:,i,t) = par.ps .* exp(-abs(Z_tilde - par.S).^2*T2(i,t));
            Q(:,i,t) = Q(:,i,t)/sum(Q(:,i,t));

            var_x = T1(i,t) * tx_noise_sq(i) / (T1(i,t) + tx_noise_sq(i));
            mean_x = (par.S*T1(i,t) + Z_tilde*tx_noise_sq(i)) / (T1(i,t)+ tx_noise_sq(i));
            
            x_hat = sum(Q(:,i,t) .* mean_x);
            G(i,t) = var_x + sum(Q(:,i,t) .* abs(mean_x - x_hat).^2);  % error variance

            Ed(:,t) = Ed(:,t) + H_(:,i)*(X_(i,t) - x_hat);
            X_(i,t) = x_hat;
        end
    end
    end
    Xd_2 = abs(X_).^2 + G;

    if lite
        temp_p = 0;
        for i=1:K
            temp_p = temp_p + sum(Xp_2(i,:)) * Sigma_H(:,:,i);
        end
        W_p = Tp * inv(norm(Ep,'fro')^2/M*eye(M) + ...
            temp_p + H_*diag(sum(G, 2))*H_');
    else
        for t=1:Tp
            temp_p = 0;
            for i=1:K
                temp_p = temp + Xp_2(i,t) * Sigma_H(:,:,i);
            end
            % W_p(:,:,t) = inv(norm(Ep(:,t))^2/M*eye(M)+ ...
            %    temp_p + H_*diag(G_p(:,t))*H_');
            W_p(:,:,t) = eye(M);
        end
    end
    
    % Update X_p
    for i=1:K
        if lite
            T1_p(i,:) =  H_(:,i)'*W_p*H_(:,i);
            T2_p(i,:) = real(T1(i,:) + trace(W_p*Sigma_H(:,:,i)));
        else
            for t=1:Tp
                T1_p(i,t) = H_(:,i)'*W_p(:,:,t)*H_(:,i);
                T2_p(i,t) = real(T1_p(i,t)) + trace(W_p(:,:,t)*Sigma_H(:,:,i));
            end
        end
    end
    
    for ii=1:3
    for i=1:K
        for t=1:Tp
            if lite
                Z_tilde_p = (T1_p(i,t)*X_p(i,t) + H_(:,i)'*W_p*Ep(:,t))/T2_p(i,t);
            else
                Z_tilde_p = (T1_p(i,t)*X_p(i,t) + H_(:,i)'*W_p(:,:,t)*Ep(:,t))/T2_p(i,t);
            end
            Q_p(i,t) = exp(-abs(Z_tilde - Xp(i, t)).^2*T2_p(i,t));
            Q_p(i,t) = Q_p(i,t)/sum(Q_p(i,t));

            var_x_p = T1_p(i,t) * tx_noise_sq(i) / (T1_p(i,t)+ tx_noise_sq(i));
            mean_x_p = (Xp(i, t)*T1_p(i,t) + Z_tilde_p*tx_noise_sq(i)) / (T1_p(i,t) + tx_noise_sq(i));
            
            x_hat_p = sum(Q_p(i,t) .* mean_x_p);
            G_p(i,t) = var_x_p + sum(Q_p(i,t) .* abs(mean_x_p - x_hat_p).^2);  % error variance

            Ep(:,t) = Ep(:,t) + H_(:,i)*(X_p(i,t) - x_hat_p);
            X_p(i,t) = x_hat_p;
        end
    end
    end
    Xp_2 = abs(X_p).^2 + G_p;
    
    
     % Update H
%     for ii=1:3
    for i=1:K
        if lite
            %Gamma_i = gamma_p*sum(Xp_2(i,:))*eye(M) + sum(Xd_2(i,:)) * W;
            Gamma_i = gamma_p*sum(Xp_2(i,:)) * W_p + sum(Xd_2(i,:)) * W;
            k_i = (eye(M) - Gamma_i\W*sum(G(i,:)))*H_(:,i) + ...
                Gamma_i\(gamma_p*Ep*X_p(i,:)' + W*Ed*X_(i,:)');
        else        
            %Gamma_i = gamma_p*PT(i)*eye(M);
            %Gamma_i = gamma_p*sum(Xp_2(i,:))*eye(M);

            Gamma_i = 0;
            for t=1:Tp
                Gamma_i = Gamma_i + Xp_2(i,t) * eye(M);
                %temp2_p = temp2_p + gamma_p * eye(M) * Ep(:,t) * conj(X_p(i,t));
                %temp2_p = temp2_p + gamma_p * eye(M) * Ep(:,t);
            end

            Gamma_i = gamma_p*Gamma_i;
            temp1 = 0;
            temp2 = 0;

            for t=1:Td
                Gamma_i = Gamma_i + Xd_2(i,t) * W(:,:,t);
                temp1 = temp1 + G(i,t) * W(:,:,t);
                temp2 = temp2 + W(:,:,t) * Ed(:,t) * conj(X_(i,t));
            end
            

            % k_i = (eye(M) - Gamma_i\temp1)*H_(:,i) + ...
            %     Gamma_i\(gamma_p*Ep*X_p(i,:)' + temp2);

            k_i = (eye(M) - Gamma_i\temp1)*H_(:,i) + ...
                Gamma_i\(gamma_p*Ep*X_p(i,:)' + temp2);
        end

        Sigma_H(:,:,i) = inv(Gamma_i + C_inv(:,:,i));
        trace_H(i) = real(trace(Sigma_H(:,:,i)));
        H_hat = Sigma_H(:,:,i)*Gamma_i*k_i;
        Ep = Ep + (H_(:,i) - H_hat)*X_p(i,:);
        Ed = Ed + (H_(:,i) - H_hat)*X_(i,:);
        H_(:,i) = H_hat;
    end

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