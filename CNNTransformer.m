%% ========================================================================
%  GNSS-Drought : Drought Identification from GNSS Time Series (ISPRS 2026)
%  ------------------------------------------------------------------------
%  Script  : CNNTransformer.m
%  Purpose : Main classifier of the paper. Trains the hybrid CNN + Transformer
%            on the SSA-derived yearly features (trend/annual/semi-annual
%            amplitude, energy, mean) labeled by SPEI, evaluates it (accuracy,
%            confusion matrix, per-class precision/recall/F1) and classifies
%            the new unlabeled years.
%  NOTE    : transformerLayer is not a built-in MATLAB layer; include the
%            custom implementation used for the paper, or replace it with
%            selfAttentionLayer (R2023a+). The cvpartition HoldOut here is a
%            random split; the paper describes a chronological 70/30 split.
%  ------------------------------------------------------------------------
%  Author  : Motahareh Esfandyari-Kaloukan
%  Paper   : Esfandyari-Kaloukan et al., ISPRS Annals XI-3-2026, 625-630
%            doi.org/10.5194/isprs-annals-XI-3-2026-625-2026
%  ========================================================================

clc; clear; close all; format long g;
disp("🔹 Step 1: Reading and preparing data...");

%% 1. Load and Prepare Data
data = readtable('final_dataset_with_labels.csv');
data.Label = categorical(data.Label);

X = data{:, 2:end-1};  % Features (exclude Year and Label)
Y = data.Label;

X = normalize(X);  % Normalize input

%% 2. Train-Test Split
cv = cvpartition(Y, 'HoldOut', 0.2);
XTrain = X(training(cv), :);
YTrain = Y(training(cv));
XTest  = X(test(cv), :);
YTest  = Y(test(cv));

%% 3. Reshape for sequence input (Transformer expects 3D input: [features x 1 x samples])
XTrain_seq = reshape(XTrain', [size(XTrain, 2), 1, size(XTrain, 1)]);
XTest_seq  = reshape(XTest', [size(XTest, 2), 1, size(XTest, 1)]);

%% 4. Define Network Architecture: CNN + Transformer
numFeatures = size(X,2);
numClasses = numel(categories(Y));

layers = [
    sequenceInputLayer([numFeatures 1],'Name','input')

    convolution2dLayer([3 1], 32, 'Padding','same','Name','conv1')
    batchNormalizationLayer('Name','bn1')
    reluLayer('Name','relu1')

    convolution2dLayer([3 1], 16, 'Padding','same','Name','conv2')
    batchNormalizationLayer('Name','bn2')
    reluLayer('Name','relu2')

    flattenLayer('Name','flatten')

    transformerLayer(...
        'NumHeads', 4, ...
        'NumLayers', 2, ...
        'HiddenSize', 64, ...
        'NumAttentionHeads', 4, ...
        'FeedForwardDimension', 128, ...
        'Name','transformer')

    fullyConnectedLayer(numClasses,'Name','fc')
    softmaxLayer('Name','softmax')
    classificationLayer('Name','output')
];

%% 5. Set Training Options
options = trainingOptions('adam', ...
    'MaxEpochs', 100, ...
    'MiniBatchSize', 8, ...
    'Shuffle','every-epoch', ...
    'ValidationData',{XTest_seq, YTest}, ...
    'Verbose',false, ...
    'Plots','training-progress');

%% 6. Train Network
net = trainNetwork(XTrain_seq, YTrain, layers, options);

%% 7. Save Model
save('drought_classifier_net.mat','net');

%% 8. Evaluate Model on Test Set
YPred = classify(net, XTest_seq);
accuracy = sum(YPred == YTest) / numel(YTest);
disp("Overall Accuracy: " + string(accuracy));

% Confusion Matrix
figure;
confusionchart(YTest, YPred);
title('Confusion Matrix - CNN + Transformer');
saveas(gcf, 'confusion_matrix_transformer.png');

% Classification Metrics
confMat = confusionmat(YTest, YPred);
classes = categories(YTest);
metrics = table('Size', [numel(classes), 4], ...
                'VariableTypes', {'string','double','double','double'}, ...
                'VariableNames', {'Class','Precision','Recall','F1_Score'});

for i = 1:numel(classes)
    TP = confMat(i,i);
    FP = sum(confMat(:,i)) - TP;
    FN = sum(confMat(i,:)) - TP;

    precision = TP / (TP + FP);
    recall = TP / (TP + FN);
    f1 = 2 * (precision * recall) / (precision + recall);

    metrics.Class(i) = classes{i};
    metrics.Precision(i) = precision;
    metrics.Recall(i) = recall;
    metrics.F1_Score(i) = f1;
end

disp(metrics);
writetable(metrics, 'performance_metrics_transformer.csv');

%% 9. Predict on New Unlabeled Data
new_data = readtable('yearly_features_from_ssa_test.csv');  % No labels here
Xnew = new_data{:, 2:end};  % Exclude Year
Xnew = normalize(Xnew);
Xnew_seq = reshape(Xnew', [size(Xnew,2), 1, size(Xnew,1)]);

Ynew_pred = classify(net, Xnew_seq);
results = table(new_data.Year, Ynew_pred, ...
    'VariableNames', {'Year','Predicted_Label'});
disp(results);
writetable(results, 'predicted_labels_new_data_transformer.csv');
