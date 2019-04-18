%% This file visualizes feature vectors computed by PCA via tSNE.  
%% Preliminary
% add path 
addpath('function')
addpath('data')

% load data
load('DSADS.mat') % data of a person

% parameters
load('parameters.mat')
% if you use the same parameters setting in several experiments, 
% setting parameters in make_parameters.m and loading parameters.mat
% is convenient.
% if you use individual setting in individual experiment,
% you can set parameters as below.
% r = 9; % remainig dimension after PCA.
% NOTICE: if you change r from default setting(e.g. r = 9), you must change
% r in dist_fun_tsne_pca.m to the same value.

%%
% number of data
N = size(motiondata, 2);
% number of attributes, length of time-series
[p, m] = size(motiondata{1, 1});  

% compute PCA modes
PCA_modes = zeros(N, p*r);
for k = 1:N
    X_dmd = motiondata{1, k};
    [U, ~, ~] = svd(X_dmd, 'econ');
    U_r = U(:, 1:r);
    PCA_modes(k, :) = reshape(U_r, [1, p*r]);
end
clear X_dmd U_r U

% visualization
Y = tsne(PCA_modes, 'Distance', @distfun_tsne_pca);
figure
gscatter(Y(:, 1), Y(:, 2), label, '', 'oxps^oxps^oxps^oxps^', '', 'off');
saveas(gcf, 'tsne_PCA', 'fig')