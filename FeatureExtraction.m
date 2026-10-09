%% ========================================================================
%  GNSS-Drought : Drought Identification from GNSS Time Series (ISPRS 2026)
%  ------------------------------------------------------------------------
%  Script  : FeatureExtraction.m
%  Purpose : Yearly feature engineering from the SSA decomposition: for each
%            year, the mean, energy and amplitude of the trend, annual and
%            semi-annual components (Table 2 of the paper).
%            Formerly ExtrAn.m; Persian comments translated. The original
%            writematrix call on a table was replaced with writetable, since
%            writematrix does not accept tables.
%  ------------------------------------------------------------------------
%  Author  : Motahareh Esfandyari-Kaloukan
%  Paper   : Esfandyari-Kaloukan et al., ISPRS Annals XI-3-2026, 625-630
%            doi.org/10.5194/isprs-annals-XI-3-2026-625-2026
%  ========================================================================

clc; clear; close all; format long g;

% Read the SSA decomposition result (daily trend/annual/semi columns)
data = readtable('ssaresult.csv');

% Convert the date column to datetime
data.Date = datetime(data.date, 'InputFormat', 'yyyy-MM-dd');

% Extract the year
data.Year = year(data.Date);

% List of available years
years = unique(data.Year);

% Initialize the feature table
features = table();

for i = 1:length(years)
    y = years(i);

    % Data of this year only
    yearData = data(data.Year == y, :);

    % Yearly features of each SSA component
    Trend_Mean = mean(yearData.trend);
    Trend_Energy = sum(yearData.trend.^2);
    Trend_Amplitude = max(yearData.trend) - min(yearData.trend);

    Annual_Mean = mean(yearData.annual);
    Annual_Energy = sum(yearData.annual.^2);
    Annual_Amplitude = max(yearData.annual) - min(yearData.annual);

    SemiAnnual_Mean = mean(yearData.semi);
    SemiAnnual_Energy = sum(yearData.semi.^2);
    SemiAnnual_Amplitude = max(yearData.semi) - min(yearData.semi);

    % Append to the feature table
    features = [features; table(y, Trend_Mean, Trend_Energy, Trend_Amplitude, ...
        Annual_Mean, Annual_Energy, Annual_Amplitude, ...
        SemiAnnual_Mean, SemiAnnual_Energy, SemiAnnual_Amplitude)];
end

% Rename the year column
features.Properties.VariableNames{1} = 'Year';

% Show the result
disp(features);

% Save as CSV
writetable(features, 'yearly_features_from_ssa_matlab.csv');
writetable(features, 'features242.csv');
