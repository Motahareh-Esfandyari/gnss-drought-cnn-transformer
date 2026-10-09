%% ========================================================================
%  GNSS-Drought : Drought Identification from GNSS Time Series (ISPRS 2026)
%  ------------------------------------------------------------------------
%  Script  : DroughtClassifierNet.m
%  Purpose : Trains the fully connected drought classifier on the labeled SSA
%            features, reports the test accuracy and confusion matrix, and
%            saves the trained network.
%            Persian comments translated to English.
%  ------------------------------------------------------------------------
%  Author  : Motahareh Esfandyari-Kaloukan
%  Paper   : Esfandyari-Kaloukan et al., ISPRS Annals XI-3-2026, 625-630
%            doi.org/10.5194/isprs-annals-XI-3-2026-625-2026
%  ========================================================================

% Read the labeled dataset
data = readtable('final_dataset_with_labels.csv');

% Convert Label to categorical
data.Label = categorical(data.Label);

% Split features and labels
X = data{:, 2:end-1};  % first column is Year, last column is Label
Y = data.Label;

% Normalize the features
X = normalize(X);

% Network architecture
layers = [
    featureInputLayer(size(X,2))
    fullyConnectedLayer(32)
    reluLayer
    fullyConnectedLayer(16)
    reluLayer
    fullyConnectedLayer(3)           % three classes
    softmaxLayer
    classificationLayer
];

% Train-test split
cv = cvpartition(Y, 'HoldOut', 0.2);
idxTrain = training(cv);
idxTest = test(cv);

XTrain = X(idxTrain,:);
YTrain = Y(idxTrain);
XTest = X(idxTest,:);
YTest = Y(idxTest);

% Training options
options = trainingOptions('adam', ...
    'MaxEpochs', 100, ...
    'MiniBatchSize', 8, ...
    'Shuffle','every-epoch', ...
    'ValidationData',{XTest, YTest}, ...
    'ValidationFrequency',10, ...
    'Verbose',false, ...
    'Plots','training-progress');

% Training
net = trainNetwork(XTrain, YTrain, layers, options);

% Predict labels
YPred = classify(net, XTest);

% Accuracy
accuracy = sum(YPred == YTest) / numel(YTest)

% Confusion matrix
figure
confusionchart(YTest, YPred)
title('Confusion Matrix - Drought Classification')

% Save the trained network and the held-out test indices
save('drought_classifier_net.mat', 'net', 'idxTest')
