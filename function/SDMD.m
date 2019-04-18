function [Phi] = SDMD(X, m, r, U_s)
%% Function that computes dynamic modes via supervised PCA modes
% for more details, see Algorithm 2 in the paper.

% inputs:
% X - time-series data matrix
% m - length of time-series
% r - rank of truncated SVD approximation to X1 in DMD algorithm. 
% U_s - projection matrix computed via supervised PCA
% outputs:
% Phi - dynamic modes

%divide data matrix
X1 = X(:, 1:m-1);
X2 = X(:, 2:m);

% SVD and rank truncation
[U,Sigma,V] = svd(X1, 'econ');
r = min(r, size(U, 2));

U_r = U(:, 1:r); % truncate to rank-r
S_r = Sigma(1:r, 1:r);
V_r = V(:, 1:r);

% build Atilde and dynamic modes
Atilde = U_s' * X2 * V_r / S_r * U_r' * U_s; % low-rank dynamics
[W,Lambda] = eig(Atilde);
Phi = X2 * V_r / S_r * U_r' * U_s * W / Lambda; % dynamic modes
end