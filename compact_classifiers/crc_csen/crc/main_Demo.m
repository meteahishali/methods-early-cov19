% =========================================================================
% 	CRC Implementation of
% 
%   Lei Zhang, Meng Yang, and Xiangchu Feng,
%   "Sparse Representation or Collaborative Representation: Which Helps Face
%    Recognition?" in ICCV 2011.
% 
% 
% Written by Meng Yang @ COMP HK-PolyU
% July, 2011.
% Modified by Mete Ahishali and Mehmet Yamac, Tampere University, Finland.
% =========================================================================
clc, clear

path = '../../features/';

versionCRC = 'light' % 'heavy' or 'light'

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

if strcmp(versionCRC, 'light')
    kappa             =   0.6; % l2 regularized parameter value
else
    kappa             =   100; % l2 regularized parameter value
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
    
    if strcmp(versionCRC, 'heavy') % CRC light or heavy.
        Dic.dictionary =[Dic.dictionary train.data];
        Dic.label =[Dic.label; train.label];
    end

    D=Dic.dictionary; %This is the dictionary
    
    %dimensional reduction
    [phi,disc_value,Mean_Image]  =  Eigen_f(D,m);
    phi = phi';
    
    A  =  phi*D;
    A  =  A./( repmat(sqrt(sum(A.*A)), [m,1]) ); %normalizarion

    %measurements for dictionary
    Y0 = phi * Dic.dictionary;
    energ_of_Y0 = sum(Y0.*Y0);
    tmp = find(energ_of_Y0 == 0);
    Y0(:,tmp)=[];
    train.label(tmp) = [];
    Y0 =  Y0./( repmat(sqrt(sum(Y0.*Y0)), [m,1]) ); %normalization

    %measurements for training
    Y1= phi*train.data;
    energ_of_Y1=sum(Y1.*Y1);
    tmp=find(energ_of_Y1==0);
    Y1(:,tmp)=[];
    train.label(tmp)=[];
    Y1=  Y1./( repmat(sqrt(sum(Y1.*Y1)), [m,1]) ); %normalization

    %measurements for test
    Y2= phi*test.data;
    energ_of_Y2=sum(Y2.*Y2);
    tmp=find(energ_of_Y2==0);
    Y2(:,tmp)=[];
    test.label(tmp)=[];
    Y2=  Y2./( repmat(sqrt(sum(Y2.*Y2)), [m,1]) ); %normalization
    
    %projection matrix computing
    % 'l2_norm'
    Proj_M = (A'*A+kappa*eye(size(A,2)))\A';

    %%%testing with CRC
    %testing
    ID = [];
    for indTest = 1:size(Y2,2)
        [id]    = CRC_RLS(A,Proj_M,Y2(:,indTest),Dic.label);
        ID      =   [ID id];
    end
    
    CMTotal(k, :, :) = confusionmat(test.label, ID');
    CM = confusionmat(test.label, ID');
    total = sum(CM(:));
    sensitivity(k) = CM(2, 2) / (CM(2, 2) + CM(2, 1));
    specificity(k) = CM(1, 1) / (CM(1, 1) + CM(1, 2));
    accuracy_test(k) = (CM(1, 1) + CM(2, 2)) / total;
    
    if strcmp(versionCRC, 'light') % CRC light.
        %%%% Save variables for CSEN
        param.prox_Y0 = Proj_M * Y0;
        param.prox_Y1 = Proj_M * Y1;
        param.prox_Y2 = Proj_M * Y2;
        param.MR = MR;
        param.lenDic = length(Dic.label);
        param.lentrain = length(train.label);
        param.lentest = length(test.label);
        param.maskM = maskM;
        param.maskN = maskN;

        prepareCSEN(Dic, train.label, test.label, param);
    end
end