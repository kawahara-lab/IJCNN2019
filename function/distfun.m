function D2 = distfun(X1, X2, d_super, p, kernel)
%% Function that computes distance between each data via dynamic modes.
% inputs:
% X1 - dynamic modes of a data. X1 = [modes1^T mode2^T ...]
% X2 - dynamic modes of full data. Each row is dynamic modes of a data.
% d_super - remaining dimension after supervised PCA or DMD.
%           If supervised PCA is performed, d_super
%           depends on supervised PCA. If not, d_super is equals to the
%           rank of the SVD approximation to X1 in DMD algorithm.
% p - number of attributes of a data. number of rows of a data.
% kernel - 1:Binet Cauthy kernel, 2:Projection kernel

% outputs:
% D2 - distance matrix

% reshape modes of a data
Phi1 = reshape(X1, [p, d_super]);
[~, ri] = qr(Phi1);

D2 = inf(size(X2, 1), 1);
for j = 1:size(X2, 1)
    % reshape modes of a data
    Phi2 = reshape(X2(j, :), [p, d_super]);

    [~, rj] = qr(Phi2);
    if kernel == 1
        % Binet-Cauchy kernel
        [~, Sigma, ~] = svd((inv(ri(1:d_super, 1:d_super))).' * Phi1.' ...
                           * Phi2 * inv(rj(1:d_super, 1:d_super)), 'econ');
        S = diag(Sigma);
        S(S > 1) = 1;
        % D2(j,1) is the distance between a data and a data
        D2(j, 1) = prod(S.^2);
    elseif kernel == 2
        % Projection kernel
        % D2(j,1) is the distance between a data and a data
        D2(j, 1) = norm((inv(ri(1:d_super, 1:d_super))).' * Phi1.' ...
                         * Phi2 * inv(rj(1:d_super, 1:d_super)), 'fro');
    else
        error('Kernel parameter must be 1 or 2.')
    end
end
end