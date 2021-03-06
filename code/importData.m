%% Import data from text file.
% Script for importing data from the following text file:
%
%    C:\Program Files\MATLAB\R2017b\bin\Machine Learning\winequality-white.csv
%
% To extend the code to different selected data or a different text file,
% generate a function instead of a script.

% Auto-generated by MATLAB on 2018/03/16 17:13:12

function [trainFeatures1, trainQual1, testFeatures1, testQual1, trainFeatures2, trainQual2, testFeatures2, testQual2, trainFeatures, trainQual, testFeatures, testQual] = importData(filename1, filename2)

%% Initialize variables.
filename1 = strcat('data\',filename1);
filename2 = strcat('data\',filename2);
delimiter = ';';
startRow = 2;

%% Format for each line of text:
%   column1: double (%f)
%	column2: double (%f)
%   column3: double (%f)
%	column4: double (%f)
%   column5: double (%f)
%	column6: double (%f)
%   column7: double (%f)
%	column8: double (%f)
%   column9: double (%f)
%	column10: double (%f)
%   column11: double (%f)
%	column12: double (%f)
% For more information, see the TEXTSCAN documentation.
formatSpec = '%f%f%f%f%f%f%f%f%f%f%f%f%[^\n\r]';

%% Open the text file.
fileID = fopen(filename1,'r');

%% Read columns of data according to the format.
% This call is based on the structure of the file used to generate this
% code. If an error occurs for a different file, try regenerating the code
% from the Import Tool.
dataArray1 = textscan(fileID, formatSpec, 'Delimiter', delimiter, 'TextType', 'string', 'EmptyValue', NaN, 'HeaderLines' ,startRow-1, 'ReturnOnError', false, 'EndOfLine', '\r\n');

%% Close the text file.
fclose(fileID);
%% Repeat for 2nd file
fileID = fopen(filename2,'r');
dataArray2 = textscan(fileID, formatSpec, 'Delimiter', delimiter, 'TextType', 'string', 'EmptyValue', NaN, 'HeaderLines' ,startRow-1, 'ReturnOnError', false, 'EndOfLine', '\r\n');
fclose(fileID);

%% Post processing for unimportable data.
% No unimportable data rules were applied during the import, so no post
% processing code is included. To generate code which works for
% unimportable data, select unimportable cells in a file and regenerate the
% script.

%% Create output variable
wineTable1 = table(dataArray1{1:end-1}, 'VariableNames', {'fixedacidity','volatileacidity','citricacid','residualsugar','chlorides','freesulfurdioxide','totalsulfurdioxide','density','pH','sulphates','alcohol','quality'});
wineTable2 = table(dataArray2{1:end-1}, 'VariableNames', {'fixedacidity','volatileacidity','citricacid','residualsugar','chlorides','freesulfurdioxide','totalsulfurdioxide','density','pH','sulphates','alcohol','quality'});

trainRatio = 0.7;
testRatio = 0.3;

wineArray1 = table2array(wineTable1);
[Train1, ~,Test1] = dividerand(size(wineArray1,1), trainRatio, 0, testRatio);
idx = randperm(size(wineArray1,1));
trainFeatures1 = wineArray1(Train1,1:end-1);
trainQual1 = wineArray1(Train1, end);
testFeatures1 = wineArray1(Test1, 1:end-1);
testQual1 = wineArray1(Test1, end);

wineArray2 = table2array(wineTable2);
[Train2, ~,Test2] = dividerand(size(wineArray2,1), trainRatio, 0, testRatio);
idx = randperm(size(wineArray2,1));
trainFeatures2 = wineArray2(Train2,1:end-1);
trainQual2 = wineArray2(Train2, end);
testFeatures2 = wineArray2(Test2, 1:end-1);
testQual2 = wineArray2(Test2, end);

wineArrayAll = [wineArray1 ;wineArray2];
[Train, ~,Test] = dividerand(size(wineArrayAll,1), trainRatio, 0, testRatio);
idx = randperm(size(wineArrayAll,1));
trainFeatures = wineArrayAll(Train,1:end-1);
trainQual = wineArrayAll(Train, end);
testFeatures = wineArrayAll(Test, 1:end-1);
testQual = wineArrayAll(Test, end);


%% Clear temporary variables
clearvars filename delimiter startRow formatSpec fileID dataArray ans;
end