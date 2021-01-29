% =========================================================================
% 	SRC Implementation
% 
% Methods and references:
% l1benchmark/L1Solvers/SolveDALM.m
% l1benchmark/L1Solvers/SolveHomotopy.m
%
% Implementation for the early COVID-19 detection,
% Authors: Mete Ahishali and Mehmet Yamac, Tampere University, Finland.
% =========================================================================
clc, clear

path = '../features/';
addpath(genpath('l1benchmark'));

dicSizeLight = 1; % If all training samples are not included in dictionary.

foldSize = 5;
classSize = 2;
data_train = {foldSize};
data_test = {foldSize};
label_train = {foldSize};
label_test = {foldSize};
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


param.dictionary_size = 625; % 25 x 25
N = size(data_train{1}, 1); %image size
nuR = 5; % Number of runs
MR = 0.5; % Measurement rate
m = floor(MR * N); %number of measurements
param.MR = MR;

rng(1)
accuracy_test = zeros(nuR,1);
specificity = zeros(nuR,1);
sensitivity = zeros(nuR,1);
CMTotal = zeros(nuR, classSize, classSize);

% read the folds
for i = 1:5
    [ Dic_all(i), train_all(i), test_all(i) ] = split_data(...
    data_train{i}, label_train{i}, data_test{i}, label_test{i}, param);
end

[maskM, maskN] = size(Dic_all(1).label_matrix);

for k = 1:nuR
    disp(strcat("Processing fold ", num2str(k), ' ...'));
    param.k = k;
    Dic = Dic_all(k);
    train = train_all(k);
    test = test_all(k);
    
    % Include the training samples as well.
    if ~dicSizeLight
        Dic.dictionary =[Dic.dictionary train.data];
        Dic.label =[Dic.label; train.label];
    end

    D=Dic.dictionary; %This is the dictionary
    
    %dimensional reduction
    [phi,disc_value,Mean_Image]  =  Eigen_f(D,m);
    phi = phi';
    
    A  =  phi*D;
    A  =  A./( repmat(sqrt(sum(A.*A)), [m,1]) ); %normalizarion

    %measurements for test
    Y2= phi*test.data;
    energ_of_Y2=sum(Y2.*Y2);
    tmp=find(energ_of_Y2==0);
    Y2(:,tmp)=[];
    test.label(tmp)=[];
    Y2=  Y2./( repmat(sqrt(sum(Y2.*Y2)), [m,1]) ); %normalization

    %%%testing with SRC methods.
    l1method={'solve_dalm','solve_homotopy'};
    l1method_names={'Dalm','Homotopy'};

    test_length=length(test.label);
    
    for i=1:length(l1method)
        ID = [];
        for indTest = 1:test_length
            [id]    =  L1_Classifier(A,Y2(:,indTest),Dic.label,l1method{i});
            ID      =   [ID id];
        end
             
        per.CMTotal(i, k, :, :) = confusionmat(test.label, ID');
        CM = confusionmat(test.label, ID');
        
        total = sum(CM(:));
        per.sensitivity(i, k) = CM(2, 2) / (CM(2, 2) + CM(2, 1));
        per.specificity(i, k) = CM(1, 1) / (CM(1, 1) + CM(1, 2));
        per.accuracy_test(i, k) = (CM(1, 1) + CM(2, 2)) / total;    
    end
end