function [Phi, Lambda] = DMD(X, m, r)
%% Function that computes the Dynamic Mode Decomposition(DMD)
% compute DMD modes and eigenvalues.
% for more details, see Algorithm 1 in the paper.
%
% inputs:
% X - time-series data matrix
% m - length of time-series
% r - rank of truncated SVD approximation to X1. 
%     NOTICE - r is equals to or learger than d_super, which is the
%              parmeter of supervised PCA.
% outputs:
% Phi - DMD modes
% Lambda - DMD eigenvalues

% divide data matrix
X1 = X(:, 1:m-1);
X2 = X(:, 2:m);

% SVD and rank truncation
[U, S, V] = svd(X1, 'econ');
r = min(r, size(U, 2));

U_r = U(:, 1:r); % truncate to rank-r
S_r = S(1:r, 1:r);
V_r = V(:, 1:r);

% build Atilde and DMD modes
Atilde = U_r' * X2 * V_r / S_r; % low-rank dynamics
[W_r, Lambda] = eig(Atilde);
Phi = X2 * V_r / S_r * W_r / Lambda; % DMD modes
