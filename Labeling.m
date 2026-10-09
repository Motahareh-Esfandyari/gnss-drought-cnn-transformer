%% ========================================================================
%  GNSS-Drought : Drought Identification from GNSS Time Series (ISPRS 2026)
%  ------------------------------------------------------------------------
%  Script  : Labeling.m
%  Purpose : Climatic labeling: merges the yearly SSA features with the yearly
%            mean SPEI (Climate Engine gridMET export) and labels each year
%            drought (SPEI < -1), wet (SPEI > 1) or normal, producing the
%            final_dataset_with_labels.csv used by all classifiers.
%            Formerly labeling.m; Persian comments translated.
%  ------------------------------------------------------------------------
%  Author  : Motahareh Esfandyari-Kaloukan
%  Paper   : Esfandyari-Kaloukan et al., ISPRS Annals XI-3-2026, 625-630
%            doi.org/10.5194/isprs-annals-XI-3-2026-625-2026
%  ========================================================================

clc; clear; close all; format long g;

% Read the SSA yearly feature file
ssa = readtable('yearly_features_from_ssa_matlab.csv');

% Read the SPEI file (Climate Engine export, gridMET drought index)
spei = readtable('ClimateEngine.csv');

% Convert the date column to datetime
spei.Date = datetime(spei.Var1, 'InputFormat', 'yyyy-MM-dd');

% Extract the year
spei.Year = year(spei.Date);

% Yearly mean SPEI
spei_yearly = varfun(@mean, spei, 'InputVariables', 'x1YearSPEI_gridMETDrought_AtPolygon1_2005_01_01To2023_12_31', ...
                     'GroupingVariables', 'Year');

% Rename the SPEI column for readability
spei_yearly.Properties.VariableNames{'mean_x1YearSPEI_gridMETDrought_AtPolygon1_2005_01_01To2023_12_3'} = 'SPEI_Mean';

% Join the SSA features with SPEI on the year
final = innerjoin(ssa, spei_yearly, 'Keys', 'Year');

% Labeling rule of the paper: SPEI < -1 drought, SPEI > 1 wet, otherwise normal
label = strings(height(final), 1);
for i = 1:height(final)
    if final.SPEI_Mean(i) < -1
        label(i) = "drought";
    elseif final.SPEI_Mean(i) > 1
        label(i) = "wet";
    else
        label(i) = "normal";
    end
end

% Add the label column
final.Label = label;

% Save the labeled dataset
writetable(final, 'final_dataset_with_labels.csv');

% Show the first rows for a quick check
disp(final(:, {'Year', 'SPEI_Mean', 'Label'}));
