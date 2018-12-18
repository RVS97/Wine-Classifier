function [trainErrorClass, trainErrorMSE, testErrorClass, testErrorMSE] = neuralnet(featuresTrain, QualityTrain, featuresTest, QualityTest)    
    
    %% TRAINING 
    % define nerual network
    net = patternnet([14 7]);
    %define transfer function
    net.layers{1}.transferFcn = char('poslin');     %logsig = logarithmic sigmoid, poslin = positive linear
    net.layers{2}.transferFcn = char('poslin');
    % compet, elliotsig, hardlim, hardlims, logsig, netinv, poslin,
    % purelin, radbas, radbasn, satlin, satlins, softmax, tansig, tribas
    
    Q = zeros(9,length(QualityTrain));
    for i= 1:length(QualityTrain)
        Q(QualityTrain(i),i) = 1;
    end
    [net,tr] = train(net,featuresTrain',Q); % train Neural Network
    predQual = vec2ind(net(featuresTrain')); % predicted quality
    errorVec = predQual' - QualityTrain;
    trainErrorClass = nnz(errorVec)/length(QualityTrain);
    trainErrorMSE = (errorVec'*errorVec)/length(errorVec);
    
    %% TEST
    predQual2 = vec2ind(net(featuresTest')); % predicted quality
    errorVec2 = predQual2' - QualityTest;
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
    title('Confusion Matrix for Neural Network', 'Fontsize', 35)
    
end