%% ========================================================================
%  GNSS-Drought : Drought Identification from GNSS Time Series (ISPRS 2026)
%  ------------------------------------------------------------------------
%  Script  : Oversampling.m
%  Purpose : Class balancing by oversampling (repeating minority-class rows to
%            the size of the largest class), then a fully connected baseline
%            network on the balanced features, per-class metrics, and
%            classification of the new unlabeled years.
%            Formerly oversapling.m; Persian comments translated to English.
%  ------------------------------------------------------------------------
%  Author  : Motahareh Esfandyari-Kaloukan
%  Paper   : Esfandyari-Kaloukan et al., ISPRS Annals XI-3-2026, 625-630
%            doi.org/10.5194/isprs-annals-XI-3-2026-625-2026
%  ========================================================================

clc; clear; close all; format long g;

% Read the labeled feature table
data = readtable('final_dataset_with_labels.csv');

% Convert the label column to categorical
data.Label = categorical(data.Label);

% Check the number of samples per class
disp("Class distribution:");
summary(data.Label)

% Split the table by class
droughtData = data(data.Label == 'drought', :);
normalData = data(data.Label == 'normal', :);
wetData     = data(data.Label == 'wet', :);

% Largest class size among the three
maxCount = max([height(droughtData), height(normalData), height(wetData)]);

% Oversample by repeating rows
drought_aug = repmat(droughtData, ceil(maxCount / height(droughtData)), 1);
normal_aug  = repmat(normalData, ceil(maxCount / height(normalData)), 1);
wet_aug     = repmat(wetData, ceil(maxCount / height(wetData)), 1);

% Keep exactly maxCount samples of each class
drought_aug = drought_aug(1:maxCount, :);
normal_aug  = normal_aug(1:maxCount, :);
wet_aug     = wet_aug(1:maxCount, :);

% Merge the classes
data_aug = [drought_aug; normal_aug; wet_aug];

% Shuffle the whole table
data_aug = data_aug(randperm(height(data_aug)), :);

% Save for inspection
writetable(data_aug, 'augmented_dataset.csv');

data = readtable('augmented_dataset.csv');
data.Label = categorical(data.Label);

X = data{:, 2:end-1}; % first column is Year
Y = data.Label;

X = normalize(X);

layers = [
    featureInputLayer(size(X,2))
    fullyConnectedLayer(32)
    reluLayer
    fullyConnectedLayer(16)
    reluLayer
    fullyConnectedLayer(3)
    softmaxLayer
    classificationLayer
];

cv = cvpartition(Y, 'HoldOut', 0.2);
XTrain = X(training(cv), :);
YTrain = Y(training(cv));
XTest = X(test(cv), :);
YTest = Y(test(cv));

options = trainingOptions('adam', ...
    'MaxEpochs', 100, ...
    'MiniBatchSize', 8, ...
    'Shuffle','every-epoch', ...
    'ValidationData',{XTest, YTest}, ...
    'Verbose',false, ...
    'Plots','training-progress');

net = trainNetwork(XTrain, YTrain, layers, options);

% Prediction and evaluation
YPred = classify(net, XTest);
accuracy = sum(YPred == YTest) / numel(YTest)

figure
confusionchart(YTest, YPred)
title('Confusion Matrix After Oversampling')

% Per-class report (precision, recall, F1-score)
confMat = confusionmat(YTest, YPred);
classes = categories(YTest);

for i = 1:numel(classes)
    TP = confMat(i, i);
    FP = sum(confMat(:, i)) - TP;
    FN = sum(confMat(i, :)) - TP;
    precision = TP / (TP + FP);
    recall = TP / (TP + FN);
    f1 = 2 * (precision * recall) / (precision + recall);
    fprintf('Class: %-8s | Precision: %.2f | Recall: %.2f | F1-Score: %.2f\n', ...
        classes{i}, precision, recall, f1);
end

% New data: features only, no labels
new_data = readtable('yearly_features_from_ssa_matlab.csv');
Xnew = new_data{:, 2:end};  % drop Year and any non-numeric column
Xnew = normalize(Xnew);

% Predict with the trained model
Ynew_pred = classify(net, Xnew);

% Show the results
disp(table(new_data.Year, Ynew_pred))
f = findall(groot, 'Type', 'Figure', 'Name', 'Training Progress');
saveas(f, 'training_progress.png');   % save the training progress figure
