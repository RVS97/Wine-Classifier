function [predQual, trainErrorClass, trainErrorMSE, testErrorClass, testErrorMSE] = ridgeRegression(featuresTrain, QualityTrain, featuresTest, QualityTest, lambda)
    
    %% TRAINING
    features = zscore(featuresTrain);
    M = mean(QualityTrain);
    Quality = zscore(QualityTrain);
    I = eye(size(featuresTrain,2));

    % Cross-Validation
    Lambda = logspace(-5, 5, 10000);
    kfold = 10;
    N = size(features, 1);
    Indices = cvpartition(N,'Kfold', kfold); %divide data for CV
    error = zeros(kfold,1);
    errorMSE = zeros(kfold,1);
    trainErrorCV = zeros(length(Lambda),1);
    trainErrorCVMSE = zeros(length(Lambda),1);
    % Cross validation for lambda
    for j=1:length(Lambda)
        for i=1:kfold
            predW = (features(Indices.training(i),:)'*features(Indices.training(i),:) + Lambda(j)*eye(size(featuresTrain,2)))\features(Indices.training(i),:)'*Quality(Indices.training(i),:);     
            predQual = round(features(Indices.test(i),:)*predW + mean(QualityTrain(Indices.test(i))));
            predQualdec = features(Indices.test(i),:)*predW + mean(QualityTrain(Indices.test(i)));
            errorVec = predQual - QualityTrain(Indices.test(i),:);
            error(i) = nnz(errorVec)/length(QualityTrain(Indices.test(i),:));
            errorVecdec = predQualdec - QualityTrain(Indices.test(i),:); % no rounding for MSE
            errorMSE(i) = (errorVec'*errorVec)/length(errorVec); % use errorVec if no rounding wanted
        end
        trainErrorCV(j) = mean(error);
        trainErrorCVMSE(j) = mean(errorMSE);
    end
    minidx = find(trainErrorCV == min(trainErrorCV),1); % min lambda same for both errors
    yyaxis left
    plot(Lambda, trainErrorCV, 'Linewidth', 1.25)
    set(gca,'xscale', 'log')
    ylabel('Classification Error, %')
    hold on
    yyaxis right
    plot(Lambda, trainErrorCVMSE, 'Linewidth', 1.25)
    xlabel('Lambda')
    ylabel('Mean Squared Error')
    set(gca, 'Fontsize', 22)
    title('Lambda Cross Validation Error', 'Fontsize', 35)
    
    predW = (features'*features +Lambda(minidx)*I)\features'*Quality; %weights
    predQual = round(features*predW + M); % predicted quality (rounded)
    predQualdec = features*predW + M;
    errorVec = predQual - QualityTrain;
    errorVecdec = predQualdec -QualityTrain;
    trainErrorClass = nnz(errorVec)/length(QualityTrain);
    trainErrorMSE = (errorVecdec'*errorVecdec)/length(errorVecdec);
    
    %% TEST
    features = zscore(featuresTest);
    predQual2 = round(features*predW + M);
    predQual2dec = features*predW + M;
    errorVec2 = predQual2 - QualityTest;
    errorVec2dec = predQual2dec - QualityTest;
    testErrorClass = nnz(errorVec2)/length(QualityTest);
    testErrorMSE = (errorVec2dec'*errorVec2dec)/length(errorVec2dec);
    
end