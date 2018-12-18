%% Information
% import data of red and white wines done by same function
% For different algorithms, run sections separately as var names coincide
% Training Error = CV Error for all except Linear Regression
% comment accordingly for each algorithm
clear all;

%% Load Data
% winequality-red.csv  or   winequality-white.csv
filename1 = 'winequality-red.csv';
filename2 = 'winequality-white.csv';
[trainFeaturesRed, trainQualRed, testFeaturesRed, testQualRed, trainFeaturesWhite, trainQualWhite, testFeaturesWhite, testQualWhite, trainFeatures, trainQual, testFeatures, testQual] = importData(filename1, filename2);

%% Linear Regression
% [predWr, trainErrorClassr, trainErrorMSEr, testErrorClassr, testErrorMSEr] = linearRegression(trainFeaturesRed, trainQualRed, testFeaturesRed, testQualRed);
% [predWw, trainErrorClassw, trainErrorMSEw, testErrorClassw, testErrorMSEw] = linearRegression(trainFeaturesWhite, trainQualWhite, testFeaturesWhite, testQualWhite);
% [predW, trainErrorClass, trainErrorMSE, testErrorClass, testErrorMSE] = linearRegression(trainFeatures, trainQual, testFeatures, testQual);

%% Ridge Regression
% [predWr, trainErrorClassr, trainErrorMSEr, testErrorClassr, testErrorMSEr] = ridgeRegression(trainFeaturesRed, trainQualRed, testFeaturesRed, testQualRed, 10.05);
% [predWw, trainErrorClassw, trainErrorMSEw, testErrorClassw, testErrorMSEw] = ridgeRegression(trainFeaturesWhite, trainQualWhite, testFeaturesWhite, testQualWhite, 10.05);
% [predW, trainErrorClass, trainErrorMSE, testErrorClass, testErrorMSE] = ridgeRegression(trainFeatures, trainQual, testFeatures, testQual, 10.05);

%% Multiclass Perceptron
% [trainErrorClassr, trainErrorMSEr, testErrorClassr, testErrorMSEr] = multiclassPerceptron(trainFeaturesRed, trainQualRed, testFeaturesRed, testQualRed);
% [trainErrorClassw, trainErrorMSEw, testErrorClassw, testErrorMSEw] = multiclassPerceptron(trainFeaturesWhite, trainQualWhite, testFeaturesWhite, testQualWhite);
% [trainErrorClass, trainErrorMSE, testErrorClass, testErrorMSE] = multiclassPerceptron(trainFeatures, trainQual, testFeatures, testQual);

%% SVM
% [trainErrorClassr, trainErrorMSEr, testErrorClassr, testErrorMSEr] = multiclassSVM(trainFeaturesRed, trainQualRed, testFeaturesRed, testQualRed);
% [trainErrorClassw, trainErrorMSEw, testErrorClassw, testErrorMSEw] = multiclassSVM(trainFeaturesWhite, trainQualWhite, testFeaturesWhite, testQualWhite);
% [trainErrorClass, trainErrorMSE, testErrorClass, testErrorMSE] = multiclassSVM(trainFeatures, trainQual, testFeatures, testQual);

%% Neural Network
% [trainErrorClassr, trainErrorMSEr, testErrorClassr, testErrorMSEr] = neuralNet(trainFeaturesRed, trainQualRed, testFeaturesRed, testQualRed);
% [trainErrorClassw, trainErrorMSEw, testErrorClassw, testErrorMSEw] = neuralNet(trainFeaturesWhite, trainQualWhite, testFeaturesWhite, testQualWhite);
% [trainErrorClass, trainErrorMSE, testErrorClass, testErrorMSE] = neuralNet(trainFeatures, trainQual, testFeatures, testQual);

