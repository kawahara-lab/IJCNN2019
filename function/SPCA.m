function [Z, U_d] = SPCA(X, label, d)
%% Function that computes PCA modes via supervised PCA
%inputs:
% X - data
% label - class label
% d - number of dimension after dimension reduction
%outputs:
% Z - data after dimension reduction
% U_d = supervised PCA modes matrix


% number of attributes, number of time-series * number of data
[p, Ntau] = size(X);
% number of data
N = length(label);
% number of time-series 
tau = Ntau/N;

% convert label to number 
cate = categorical(label);
% number of element of each category
count_cate = countcats(cate);

% Ai is the value of diagonal elements of H
% Bi is the value of non-diagonal elements of H
Ai = (Ntau - tau * count_cate) / Ntau;
Bi = - tau * count_cate / Ntau;


% X_s * H * L in Algorithm 2 of the paper
XsHL = zeros(p, Ntau);

% matrix of size [l_11 ... l_N1]^T in equation (18) in the paper
Onematrix = ones(Ntau, tau);

% H * X_s^T in Algorithm2
HXs = zeros(Ntau, p);
for i = 1:N
    % the value of Q changes when class switch
    Q = fix((i-1) / count_cate(1));
    % HLvec is H multiply [l_11 ... l_N1]^T 
    HLvec = Onematrix * Bi(1);
    HLvec(Q * tau * count_cate(1) +1: (Q+1) * tau * count_cate(1), :) = Ai(1);
    
    % compute X_s * H * L in Algorithm 2
    temp = X * HLvec;
    XsHL(:, 1 + (i-1) * tau: i * tau) = temp;
    
    % compute diagonal component of H in Algorithm 2
    Hvec = Onematrix.' * (-1 / Ntau);
    for j = 1:tau
        Hvec(j, j+ (i-1) * tau) = (Ntau - 1) / Ntau;
    end
    % compute H * X_s^T in Algorithm2
    HXs((i-1) * tau +1: i * tau, :) = Hvec * X.';
end

clear Hvec Lvec HLvec Evec

% compute Q in Algorithm 2
Q = XsHL * HXs;

% compute U_s in Algorithm 2
[U , ~] = eigs(Q, d);
U_d = U(:, 1:d);

% compute dimension reduced data
Z = U_d.' * X;
end