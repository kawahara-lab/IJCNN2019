%% This file performs classification via SVM.
% input of SVM is distance matrix computed via DMD modes.
%% Preliminary
% add path
addpath('libsvm-3.23\matlab')
addpath('function')
addpath('data')

% load data
load('DSADS_all') % data of all persons

% parameters
load('parameters.mat')
% if you use the same parameters setting in several experiments, 
% setting parameters in make_parameters.m and loading parameters.mat
% is convenient.
% if you use individual setting in individual experiment,
% you can set parameters as below.
% r = 9; % rank of truncated SVD approximation to X1 in DMD algorithm.       
% kernel = 2; % 1:Binet Cauthy kernel, 2:Projection kernel

%%
% number of data
N = size(motiondata, 2);
% number of attributes, length of time-series
[p, m] = size(motiondata{1, 1});  

% compute DMD modes of each data
DMD_modes = zeros(N, p*r);
for k=1:N
    temp = DMD(motiondata{1, k}, m, r);
    DMD_modes(k, :) = reshape(temp, [1, p*r]);
end

clear temp

% label
[G, GN, GL] = grp2idx(label);

% Leave One Out cross validation
K = 8; %data contains 8 persons
% accuracy rate of classification
acc_s = zeros(1, K); 

for valid_iter = 1:K
    % make index to divide training data and validation data
    % index of training data is 1 and of validation data is 0
    idx = ones(N, 1);
    idx(1140 * (valid_iter-1) + 1:1140 * valid_iter, 1) = 0;
    idx = logical(idx);
    
    % DMD modes of training data
    train_modes = DMD_modes(idx, :);
    train_label = G(idx);
    % DMD modes of test data
    test_modes = DMD_modes(~idx, :);
    test_label = G(~idx);
    
    % number of training data
    N_train = size(train_modes, 1);
    
    % distance matrix between training data computed via DMD modes
    dist_matrix_train = zeros(N_train, N_train);      
    for i = 1:N_train
        dist_matrix_train(i, :) = ...
            distfun(train_modes(i, :), train_modes, r, p, kernel);
    end
    
    % train svm classifier
    svm_model = ...
        svmtrain(train_label, [(1:N_train).', dist_matrix_train], '-t 4');
    
    % number of test data
    N_test = size(test_modes,1);   
    
    % distance matrix between test data and train data computed via DMD modes
    dist_matrix_test = zeros(N_test, N_train); 
    for i = 1:N_test
        dist_matrix_test(i,:) = ...
            distfun(test_modes(i, :), train_modes, r, p, kernel);
    end
    
    %predict class
    [predict_label, acc, ~] = ...
        svmpredict(test_label, [(1:N_test).', dist_matrix_test], svm_model);
    acc_s(1, valid_iter) = acc(1);
end

%save result as txt file
TT = ['remaining dimension after reduction is: ',  num2str(r), ...
    ',\naccuracy rates are: ', num2str(acc_s), ...
    ',\nmean of accuracy rates is: ', num2str(mean(acc_s)), ...
    ',\nstandard deviation of accuracy rate is: ', num2str(std(acc_s)), '.'];
disp('Result is written in txt')
fileID = fopen('result_DMD.txt', 'wt');
fprintf(fileID, TT);
fclose(fileID);
