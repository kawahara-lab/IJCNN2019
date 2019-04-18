%% This file visualizes feature vectors computed by DMD via tSNE.  
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
% r = 9; % r - rank of truncated SVD approximation to X1 in DMD algorithm. 
% NOTICE: if you change r from default setting(e.g. r = 9), you must change
% r in dist_fun_tsne_dmd.m to the same value.
%% 
% number of data
N = size(motiondata, 2);
% number of attributes, length of time-series
[p, m] = size(motiondata{1, 1});  

% compute DMD modes
DMD_modes = zeros(N, p*r);
for k = 1:N
    X_dmd = motiondata{1, k};
    temp = DMD(X_dmd, m, r);
    DMD_modes(k, :) = reshape(temp, [1, p*r]);
end
clear X_dmd temp

% divide real part and imaginary part
DMR = real(DMD_modes);
DMI = imag(DMD_modes);
X_tsne = [DMR DMI];

% visualization
Y = tsne(X_tsne, 'Distance', @distfun_tsne_dmd);
figure
gscatter(Y(:, 1), Y(:, 2), label, '', 'oxps^oxps^oxps^oxps^', '', 'off');
saveas(gcf, 'tsne_DMD', 'fig')