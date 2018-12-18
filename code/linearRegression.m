function [predW, trainErrorClass, trainErrorMSE, testErrorClass, testErrorMSE] = linearRegression(featuresTrain, QualityTrain, featuresTest, QualityTest)
    
    %% TRAINING
    predW = pinv(featuresTrain)*QualityTrain; %weights
    predQual = round(featuresTrain*predW); %predicted quality
    errorVec = predQual - QualityTrain; 
    trainErrorClass = nnz(errorVec)/length(QualityTrain);
    trainErrorMSE = (errorVec'*errorVec)/length(errorVec);
    
    %% TEST
    predQual2 = round(featuresTest*predW); %predicted quality
    errorVec2 = predQual2 - QualityTest;
    testErrorClass = nnz(errorVec2)/length(QualityTest);
    testErrorMSE = (errorVec2'*errorVec2)/length(errorVec2);
    
end