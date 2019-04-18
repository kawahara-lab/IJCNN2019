%% This file visualizes feature vectors computed by Supervised PCA + DMD via tSNE.  
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
% d_super = 9; % d_super - remaining dimension after supervised PCA.
% NOTICE: if you change r from default setting(e.g. r = 9), you must change
% r in dist_fun_tsne_dmd.m to the same value.

%%
% number of data
N = size(motiondata, 2);
% number of attributes, length of time-series
[p, m] = size(motiondata{1, 1});  

% labels
[G, GN, GL] = grp2idx(label);
nc = num2cell(G);

% combine data and class
tmp = [motiondata; nc.'];
tmp = tmp.';
% sort by class
sorted = (sortrows(tmp, 2)).';
% sorted data
motiondata = sorted(1, :);
% sorted label
label = (GL(cell2mat(sorted(2, :)))).';

clear ind tmp sorted nc

% arrange data
X_s = inf(p, (m-1)*N);
for j = 1:N
    temp = motiondata{j}(1:p, 1:m-1);
    X_s(1:p, (j-1)*(m-1)+1:j*(m-1)) = motiondata{j}(1:p, 1:m-1);
end

% supervised PCA
[Z, U_s] = SPCA(X_s, label, d_super);

% compute DMD modes
DMD_modes = zeros(length(motiondata), p*d_super);
for k = 1:length(motiondata)
    X_dmd = motiondata{1, k};
    Phi = SDMD(X_dmd, m, r, U_s);
    DMD_modes(k, :) = reshape(Phi, [1, p*d_super]);
end
clear X_dmd X_s

% divide real part and imaginary part
DMR = real(DMD_modes);
DMI = imag(DMD_modes);
X_tsne = [DMR DMI];

% visualization
Y = tsne(X_tsne, 'Distance', @distfun_tsne_dmd);
figure
gscatter(Y(:, 1), Y(:, 2), label, '', 'oxps^oxps^oxps^oxps^', '', 'off');
saveas(gcf, 'tsne_SDMD', 'fig')
