%% This file visualizes raw data via tSNE.  
%% Preliminary
% add path 
addpath('function')
addpath('data')

% load data
load('DSADS.mat') % data of a person
%%
% number of data
N = size(motiondata, 2);
% number of attributes, length of time-series
[p, m] = size(motiondata{1, 1});  


X_raw = zeros(N, p*m);
% substitute raw data in X_raw
for k = 1:N
    temp = motiondata{1, k};
    X_raw(k, :) = reshape(temp, [1, p*m]);
end

% visualization
Y = tsne(X_raw, 'Algorithm', 'exact');        
figure
gscatter(Y(:, 1), Y(:, 2), label, '', 'oxps^oxps^oxps^oxps^', '', 'off');
saveas(gcf, 'tsne_Euclidean', 'fig')