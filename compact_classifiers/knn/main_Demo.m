% =========================================================================
%	k-NN classification
% Written by Mete Ahishali, Tampere University, Finland.
% =========================================================================
clc, clear, close all

path = '../features';

foldSize = 5;
classSize = 2;
data_train = {foldSize};
data_test = {foldSize};
label_train = {foldSize};
label_test = {foldSize};

% Load the data.
for i = 1:foldSize
    load(fullfile(path, strcat('features_train_fold_', num2str(i), '.mat')))
    load(fullfile(path, strcat('feautures_test_fold_', num2str(i), '.mat')))
    load(fullfile(path, strcat('y_train_fold_', num2str(i), '.mat')))
    load(fullfile(path, strcat('y_test_fold_', num2str(i), '.mat')))
    data_train{i} = features_train_fold';
    data_test{i} = features_test_fold';
    label_train{i} = zeros(size(features_train_fold, 1), 1);
    label_test{i} = zeros(size(features_test_fold, 1), 1);
    for j = 1:classSize
        label_train{i}(y_train_fold(:, j) == 1) = j;
        label_test{i}(y_test_fold(:, j) == 1) = j;
    end
    clear y_train_fold y_test_fold features_train_fold features_test_fold
end

nuR = 5; % Number of runs (folds)
MR = 0.5;
N = size(data_train{1}, 1);
m = floor(MR * N); %number of measurements

rng(1)

accuracy = zeros(nuR, 1);
specificity = zeros(nuR, 1);
sensitivity = zeros(nuR, 1);
CMTotal = zeros(nuR, classSize, classSize);
mdls{5} = []; % To store the best models.

for k = 1:nuR
    disp("Processing Fold " + num2str(k) + " ...");
    trainData = data_train{k};
    testData = data_test{k};
    trainIds = label_train{k};
    testIds = label_test{k};
    
    %dimensional reduction
    [phi,disc_value,Mean_Image]  =  Eigen_f(trainData, m);
    phi = phi';
    
    % data normalization
    trainData = phi*trainData;
    trainData = (trainData - repmat(mean(trainData, 2), 1, size(trainData, 2))) ./ repmat(std(trainData, 0, 2), 1, size(trainData, 2));
    testData = phi*testData;
    testData = (testData - repmat(mean(testData, 2), 1, size(testData, 2))) ./ repmat(std(testData, 0, 2), 1, size(testData, 2));


    %%N-N%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % found k-nn configuration for each fold
    if k == 1
        mdl = fitcknn((trainData)',(trainIds)', 'NumNeighbors', 106, 'Distance', 'correlation');
        
    elseif k == 2
        mdl = fitcknn((trainData)',(trainIds)', 'NumNeighbors', 106, 'Distance', 'cosine');
        
    elseif k == 3
        mdl = fitcknn((trainData)',(trainIds)', 'NumNeighbors', 106, 'Distance', 'cosine');
    elseif k == 4
        mdl = fitcknn((trainData)',(trainIds)', 'NumNeighbors', 106, 'Distance', 'correlation');
    elseif k == 5
        mdl = fitcknn((trainData)',(trainIds)', 'NumNeighbors', 106, 'Distance', 'spearman');
    end
    
    % Hyper-parameter search.
%     params = hyperparameters('fitcknn',(trainData)',(trainIds)');
%     mdl = fitcknn((trainData)',(trainIds)', 'OptimizeHyperparameters', params, 'HyperparameterOptimizationOptions', struct('Optimizer', 'gridsearch'));
% 
%     mdls{k} = mdl;
    
    tic
    ID = predict(mdl,(testData)')';
    toc
        
    cornum      =   sum(ID' == testIds);
    Rec         =   cornum/length(testIds); % recognition rate
    
    % Performance metrics
    CM = confusionmat(testIds, ID');
    CMTotal(k, :, :) = CM;
    total = sum(CM(:));
    sensitivity(k) = CM(2, 2) / (CM(2, 2) + CM(2, 1));
    specificity(k) = CM(1, 1) / (CM(1, 1) + CM(1, 2));
    accuracy(k) = (CM(1, 1) + CM(2, 2)) / total;
 
end
disp("Average accuracy: " + num2str(mean(accuracy)))
disp("Average sensitivity: " + num2str(mean(sensitivity)))
disp("Average specificity: " + num2str(mean(specificity)))
