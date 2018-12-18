function [trainErrorClass, trainErrorMSE, testErrorClass, testErrorMSE] = multiclassSVM(featuresTrain, QualityTrain, featuresTest, QualityTest)
    
    %% TRAINING
    %Cross validation for C and sigma
    C = linspace(1.25, 1.75, 10);
    sigma = linspace(0.75, 1.25, 10);
    Table = array2table(featuresTrain);
    for i = 1:length(C)
        for j = 1:length(sigma)
            t = templateSVM('KernelFunction', 'gaussian', 'Standardize', true, 'BoxConstraint', C(1, i), 'KernelScale', sigma(1, j));
            mcSVM = fitcecoc(Tablw, QualityTrain, 'Learners', t, 'CrossVal', 'on', 'kFold', 10);
            Z(i,j) = kfoldLoss(Mdl);
        end
    end
   
    imagesc(C, sigma, Z);
    colorbar;
    colormap(summer);
    xlabel('C');
    ylabel('Sigma');
    set(gca, 'Fontsize', 30);
    title('CV Error for values of sigma and C', 'Fontsize', 40)
    
    
    Table = array2table(featuresTrain);
    t = templateSVM('KernelFunction', 'gaussian', 'Standardize', true);
    mcSVM = fitcecoc(Table, QualityTrain, 'Learners', t); %, 'Coding', 'onevsall'
    
    predQual = predict(mcSVM, featuresTrain);
    errorVec = predQual - QualityTrain;
    trainErrorClass = nnz(errorVec)/length(QualityTrain);
    trainErrorMSE = (errorVec'*errorVec)/length(errorVec);
    
    %% TEST
    predQual2 = predict(mcSVM, featuresTest);
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
    set(gca, 'Fontsize', 30)
    title('Confusion Matrix for Multi-Class SVM', 'Fontsize', 40)
    
end