%% ========================================================================
%  GNSS-Drought : Drought Identification from GNSS Time Series (ISPRS 2026)
%  ------------------------------------------------------------------------
%  Script  : BaselineMLP_BiLSTM.m
%  Purpose : Baseline classifier: class balancing followed by an MLP + BiLSTM
%            network with k-fold evaluation, exporting per-fold metrics.
%            Formerly testnew.m.
%  ------------------------------------------------------------------------
%  Author  : Motahareh Esfandyari-Kaloukan
%  Paper   : Esfandyari-Kaloukan et al., ISPRS Annals XI-3-2026, 625-630
%            doi.org/10.5194/isprs-annals-XI-3-2026-625-2026
%  ========================================================================

clc; clear; close all; format long g;

%% 1. Load and Balance the Data
data = readtable('final_dataset_with_labels.csv');
data.Label = categorical(data.Label);

% Extract classes
classes = categories(data.Label);
maxPerClass = 15; % هدف: هر کلاس ۱۵ نمونه

% Balance the dataset using oversampling
balanced_data = [];
for i = 1:numel(classes)
    classData = data(data.Label == classes{i}, :);
    if height(classData) >= maxPerClass
        sampled = classData(1:maxPerClass, :);
    else
        reps = ceil(maxPerClass / height(classData));
        sampled = repmat(classData, reps, 1);
        sampled = sampled(1:maxPerClass, :);
    end
    balanced_data = [balanced_data; sampled];
end

% Shuffle the balanced data
balanced_data = balanced_data(randperm(height(balanced_data)), :);

% Extract features and labels
X = balanced_data{:, 2:end-1};  % assuming 1st column is Year
Y = balanced_data.Label;

% Normalize and reshape
X = balanced_data{:, 2:end-1};
X = X';
X = normalize(X);               % Normalize features
X = reshape(X, [size(X,1), 1, size(X,2)]);        % Convert to [samples × 1 × features]

inputSize = size(X, 3);
numClasses = numel(categories(Y));

%% 2. K-Fold Cross Validation (k = 5)
K = 5;
cv = cvpartition(Y', 'KFold', K);

allAccuracies = zeros(K,1);
allMetrics = [];

for fold = 1:K
    fprintf('\n--- Fold %d/%d ---\n', fold, K);

    trainIdx = training(cv, fold);
    testIdx = test(cv, fold);

    XTrain = X(trainIdx, :, :);
    YTrain = Y(trainIdx);
    XTest  = X(testIdx, :, :);
    YTest  = Y(testIdx);

    %% 3. Define MLP + BiLSTM Architecture
    inputSize = size(XTrain, 1);  % به جای عدد ثابت
    layers = [
        sequenceInputLayer(inputSize, 'Name', 'input')

        fullyConnectedLayer(64, 'Name', 'fc1')
        reluLayer('Name', 'relu1')

        fullyConnectedLayer(32, 'Name', 'fc2')
        reluLayer('Name', 'relu2')

        bilstmLayer(32, 'OutputMode', 'last', 'Name', 'bilstm')

        fullyConnectedLayer(numClasses, 'Name', 'fc_final')
        softmaxLayer('Name', 'softmax')
        classificationLayer('Name', 'output')
    ];

    %% 4. Training Options
    options = trainingOptions('adam', ...
        'MaxEpochs', 100, ...
        'MiniBatchSize', 4, ...
        'Shuffle', 'every-epoch', ...
        'Verbose', false, ...
        'Plots','none', ...
        'ValidationData',{XTest, YTest});

    %% 5. Train the Network
    net = trainNetwork(XTrain, YTrain, layers, options);

    %% 6. Evaluate on Test Data
    YPred = classify(net, XTest);
    acc = sum(YPred == YTest) / numel(YTest);
    allAccuracies(fold) = acc;
    fprintf('Accuracy: %.2f%%\n', acc * 100);

    %% 7. Confusion Matrix + Save
    figure;
    confusionchart(YTest, YPred);
    title(['Confusion Matrix - Fold ', num2str(fold)]);
    saveas(gcf, ['confusion_matrix_fold_' num2str(fold) '.png']);

    %% 8. Precision, Recall, F1-Score
    confMat = confusionmat(YTest, YPred);
    metrics = table();
    for i = 1:numClasses
        TP = confMat(i,i);
        FP = sum(confMat(:,i)) - TP;
        FN = sum(confMat(i,:)) - TP;
        precision = TP / (TP + FP + eps);
        recall    = TP / (TP + FN + eps);
        f1 = 2 * (precision * recall) / (precision + recall + eps);

        row = table(string(classes{i}), precision, recall, f1, ...
            'VariableNames', {'Class','Precision','Recall','F1Score'});
        metrics = [metrics; row];
    end

    metrics.Fold = repmat(fold, height(metrics), 1);
    allMetrics = [allMetrics; metrics];
end

%% 9. Report
fprintf('\n===== Summary Report =====\n');
fprintf('Average Accuracy (5-Fold): %.2f%%\n', mean(allAccuracies) * 100);
disp('All Metrics:');
disp(allMetrics);

% Save all metrics to file
writetable(allMetrics, 'metrics_MLP_BiLSTM_KFold.csv');
