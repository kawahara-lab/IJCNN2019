%% This file performs classification via SVM.
% input of SVM is distance matrix computed via supervised PCA + DMD modes.
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
% d_super = 9; % ramaining dimension after supervised PCA
% kernel = 2; % 1:Binet Cauthy kernel, 2:Projection kernel

%%
% number of data
N = size(motiondata, 2);
% number of attributes, length of time-series
[p, m] = size(motiondata{1, 1});  

% label
[G, GN, GL] = grp2idx(label);

% Leave One Out cross validation
K = 8; %data contains 8 persons
% accuracy rate of classification
acc_s = zeros(1,K);

for valid_iter = 1:K
    %% Prepare data
    % make index to divide training data and validation data
    % index of training data is 1 and of validation data is 0
    idx = ones(N,1);
    idx(1140*(valid_iter - 1) + 1 : 1140*valid_iter, 1) = 0;
    idx = logical(idx);
    
    % divide training data and test data
    train_data = motiondata(idx);
    train_label = G(idx);
    test_data = motiondata(~idx);
    test_label = G(~idx);
    
    % number of traininn data
    N_train = length(train_data);
    % number of test data 
    N_test = length(test_data);    
    
    %% SupervisedPCA
    % combine data and class
    nc = num2cell(train_label);     
    tmp = [train_data;nc.'].';
    % sort by class
    sorted = (sortrows(tmp, 2)).';
    train_data = sorted(1, :);
    train_label = (cell2mat(sorted(2, :))).';
    clear tmp nc smot
    
    % compute data matrix which is input of supervised PCA
    X_s = inf(p, (m-1) * N_train);
    for j = 1:N_train
        temp = train_data{j}(1:p, 1:m-1);
        X_s(1:p, (j-1) * (m-1) + 1: j * (m-1)) = train_data{j}(1:p, 1:m-1);
    end
    [Z, U_s] = SPCA(X_s, train_label, d_super);
    
    %% 
    % Compute DMD modes of training data mapped to supervised PCA space
    % then compute distance matrix 
    % then perform classification via SVM
    
    % DMD modes of training data
    DMD_modes_train = zeros(length(train_data), p*d_super);  
    for k = 1:N_train
        X_dmd = train_data{1, k};
        Phi = SDMD(X_dmd, m, r, U_s);
        DMD_modes_train(k, :) = reshape(Phi, [1, p*d_super]);
    end
    
    % distance matrix between training data computed via DMD modes
    dist_matrix_train = zeros(N_train, N_train);
    for i = 1:N_train
        dist_matrix_train(i, :) = ...
            distfun(DMD_modes_train(i, :), DMD_modes_train, d_super, p, kernel);
    end
    
    % train svm classifier
    svm_model = ...
        svmtrain(train_label, [(1:N_train).', dist_matrix_train], '-t 4');

    % DMD modes of test data
    DMD_modes_test = zeros(length(test_data), p*d_super);
    for l = 1:N_test
        X_dmd = test_data{1, l};
        Phi = SDMD(X_dmd, m, r, U_s);
        DMD_modes_test(l, :) = reshape(Phi, [1, p*d_super]);
    end
    
    % distance matrix between test data and train data computed via DMD modes
    dist_matrix_test = inf(N_test, N_train); 
    for i = 1:N_test
        dist_matrix_test(i, :) = ...
            distfun(DMD_modes_test(i, :), DMD_modes_train, d_super, p, kernel);
    end
    
    %predict class
    [predict_label, acc, ~] = ...
        svmpredict(test_label, [(1:N_test).', dist_matrix_test], svm_model);
    acc_s(1, valid_iter) = acc(1);
end

%save result as txt file
TT = ['remaining dimension after reduction is: ',  num2str(d_super), ...
    ',\naccuracy rates are: ', num2str(acc_s), ...
    ',\nmean of accuracy rates is: ', num2str(mean(acc_s)), ...
    ',\nstandard deviation of accuracy rate is: ', num2str(std(acc_s)), '.'];
disp('Result is written in txt')
fileID = fopen('result_SDMD.txt','wt');
fprintf(fileID,TT);
fclose(fileID);