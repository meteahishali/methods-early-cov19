function [] = prepareCSEN(Dic, trainLabel, testLabel, param)
    maskM = param.maskM;
    maskN = param.maskN;
    prox_Y0 = param.prox_Y0;
    prox_Y1 = param.prox_Y1;
    prox_Y2 = param.prox_Y2;
    lenDic = param.lenDic;
    lentrain = param.lentrain;
    lentest = param.lentest;
    
    x_dic = zeros(lenDic, maskM, maskN);
    x_train = zeros(lentrain, maskM, maskN);
    x_test = zeros(lentest, maskM, maskN);
    y_dic = zeros(lenDic, maskM, maskN);
    y_train = zeros(lentrain, maskM, maskN);
    y_test = zeros(lentest, maskM, maskN);

    %groud truth olusturma
    for i=1:lenDic
        x_dic(i,:,:) = reshape(prox_Y0(:, i), maskM, maskN);
        y_dic(i,:,:)=(Dic.label_matrix == Dic.label(i));
    end
    for i=1:lentrain
        x_train(i,:,:) = reshape(prox_Y1(:, i), maskM, maskN);
        y_train(i,:,:) = (Dic.label_matrix == trainLabel(i));
    end
    for i=1:lentest
        x_test(i,:,:) = reshape(prox_Y2(:, i), maskM, maskN);
        y_test(i,:,:)=(Dic.label_matrix==testLabel(i));
    end
    
    if ~exist('../CSENdata', 'dir')
       mkdir('../CSENdata')
    end    
    save(strcat("../CSENdata/data_dic_", num2str(param.MR), '_', num2str(param.k), (".mat")), ... 
        'x_train', 'x_test', 'y_train', 'y_test', 'x_dic', 'y_dic', '-v6');
    
    Dic.label_matrix;
    save('../CSENdata/dic_label.mat', 'ans');
end

