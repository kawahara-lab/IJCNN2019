function D2 = distfun_tsne_pca(X1, X2)
%% Function that computes distance for tSNE
%inputs:
% X1 - single observation
% X2 - multiple observation
% for more details, see URL below and look for the word "@distfun"
% https://jp.mathworks.com/help/stats/pdist.html?lang=en

% parameters
r = 9; % number of PCA modes to reconstruct data
p = 45; % number of attributes
kernel = 2; % 1:Binet Cauthy kernel, 2:Projection kernel

% reconstruct single observation
Phi1 = reshape(X1, [p, r]);
[~, ri] = qr(Phi1);

D2 = inf(size(X2, 1), 1);
for i = 1:size(X2, 1)
    % reconstruct single observation
    Phi2 = reshape(X2(i, :), [p, r]);
    [~, rj] = qr(Phi2);
    
    % compute principale angle
    % for more details, see around formula (11) in the paper
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
        error('Kernel parameter must 1 or 2')
    end
end

end
