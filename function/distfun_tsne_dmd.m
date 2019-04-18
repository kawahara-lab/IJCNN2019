function D2 = distfun_tsne_dmd(X1, X2)
%% Function that computes distance for tSNE
%inputs:
% X1 - single observation
% X2 - multiple observation
% for more details, see URL below and look for the word "@distfun"
% https://jp.mathworks.com/help/stats/pdist.html?lang=en

% parameters
r = 9; % number of DMD modes to reconstruct data
p = 45; % number of attributes
kernel = 2; % 1:Binet Cauthy metric, 2:Projection metric

% reconstruct imaginary number
tau = size(X1, 2) / 2;
X1X = X1(:, 1:tau) + X1(:, tau+1:end) * 1j;
X2X = X2(:, 1:tau) + X2(:, tau+1:end) * 1j;

% reconstruct single observation
Phi1 = reshape(X1X, [p, r]);
[~, ri] = qr(Phi1);

D2 = inf(size(X2X, 1), 1);
for i = 1:size(X2X, 1)
    % reconstruct single observation
    Phi2 = reshape(X2X(i, :), [p, r]);
    
    % compute principale angle
    % for more details, see around formula (11) in the paper
    [~, rj] = qr(Phi2);
    [~, Sigma, ~] = ...
        svd((inv(ri(1:r, 1:r))).' * Phi1.' * Phi2 * inv(rj(1:r, 1:r)), 'econ');
    
    S = diag(Sigma);
    S(S > 1) = 1;
    if kernel == 1
        % Binet-Cauchy Metric
        % D2(j,1) is the distance between a observation and a observation
        D2(i, 1) = (1 - prod(S.^2))^(1/2);
    elseif kernel == 2
        % Projuction Metric
        % D2(j,1) is the distance between a observation and a observation
        D2(i, 1) = (r - sum(S.^2))^(1/2);
    else
        error('Kernel parameter must be 1 or 2')
    end
end
end