function [trainErrorClass, trainErrorMSE, testErrorClass, testErrorMSE] = multiclassPerceptron(featuresTrain, QualityTrain, featuresTest, QualityTest)
    
    %% TRAINING
    % define learner
    t = templateLinear('Regularization', 'Ridge', 'solver', 'lbfgs'); %Limited-memory BFGS
%     t = templateLinear('Regularization', 'Lasso', 'solver', 'sparsa');
    % define multiclass perceptron
    mcPerceptron = fitcecoc(featuresTrain, QualityTrain, 'Learners', t); %, 'Coding', 'onevsall'
    
    predQual = predict(mcPerceptron, featuresTrain); % predicted quality
    errorVec = predQual - QualityTrain;
    trainErrorClass = nnz(errorVec)/length(QualityTrain);
    trainErrorMSE = (errorVec'*errorVec)/length(errorVec);
    
    %% TEST
    predQual2 = predict(mcPerceptron, featuresTest); % predicted quality
    errorVec2 = predQual2 - QualityTest;
    testErrorClass = nnz(errorVec2)/length(QualityTest);
    testErrorMSE = (errorVec2'*errorVec2)/length(errorVec2);  
    
    % plot confusion matrix
    figure
    confMatData = confusionmat(QualityTest, predQual2);
    imagesc(confMatData)
    colormap(summer)
    ax = gca;
    ax.XTickLabel = min(QualityTest):1:max(QualityTest);
    ax.YTickLabel = min(QualityTest):1:max(QualityTest);
    colorbar;
    xlabel('Prediction')
    ylabel('Target')
    set(gca, 'Fontsize', 22)
    title('Confusion Matrix for Multi-Class Perceptron', 'Fontsize', 35)
    
end