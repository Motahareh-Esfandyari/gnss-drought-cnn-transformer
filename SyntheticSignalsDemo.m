%% ========================================================================
%  GNSS-Drought : Drought Identification from GNSS Time Series (ISPRS 2026)
%  ------------------------------------------------------------------------
%  Script  : SyntheticSignalsDemo.m
%  Purpose : Illustrative synthetic signals for the paper's concept figures:
%            an annual drought signature (2019-2020), a semi-annual rainfall
%            signature (2017), and a one-day earthquake event (2015).
%            Formerly plotdaneshlstm.m; Persian comments translated.
%  ------------------------------------------------------------------------
%  Author  : Motahareh Esfandyari-Kaloukan
%  Paper   : Esfandyari-Kaloukan et al., ISPRS Annals XI-3-2026, 625-630
%            doi.org/10.5194/isprs-annals-XI-3-2026-625-2026
%  ========================================================================

start_year = 2010;
end_year = 2020;
m = (end_year - start_year) * 365;
time = (1:m)' ./ 365.25 + start_year;

% ---------- Signal 1: drought only in 2019 and 2020 ----------
dryness = zeros(m, 1);
for year = [2019, 2020]
    idx = find(time >= year & time < year + 1);
    dryness(idx) = 10 * sin(2*pi*(time(idx)-year));
end

figure;
plot(time, dryness, 'b', 'LineWidth', 1.5);
xlabel('Time (Year)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Displacement (mm)', 'FontSize', 12, 'FontWeight', 'bold');
title('Signal 1: Drought in 2019 and 2020 (Annual Frequency)', 'FontSize', 14, 'FontWeight', 'bold');
grid on; xlim([start_year end_year]);
set(gca, 'FontName', 'Times New Roman', 'FontSize', 12);

% ---------- Signal 2: rainfall only in 2017 ----------
rainfall = zeros(m, 1);
idx = find(time >= 2017 & time < 2018);
rainfall(idx) = 15 * sin(2*pi*2*(time(idx)-2017));  % twice the annual frequency = semi-annual

figure;
plot(time, rainfall, 'c', 'LineWidth', 1.5);
xlabel('Time (Year)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Displacement (mm)', 'FontSize', 12, 'FontWeight', 'bold');
title('Signal 2: Rainfall in 2017 (Semi-Annual Frequency)', 'FontSize', 14, 'FontWeight', 'bold');
grid on; xlim([start_year end_year]);
set(gca, 'FontName', 'Times New Roman', 'FontSize', 12);

% ---------- Signal 3: one-day earthquake in 2015 ----------
earthquake = zeros(m, 1);
idx_eq = round((2015 - start_year) * 365 + 180); % around mid 2015
earthquake(idx_eq) = 10;  % small amplitude

figure;
plot(time, earthquake, 'r', 'LineWidth', 1.5);
xlabel('Time (Year)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Displacement (mm)', 'FontSize', 12, 'FontWeight', 'bold');
title('Signal 3: Earthquake in 2015 (One-Day Event)', 'FontSize', 14, 'FontWeight', 'bold');
grid on; xlim([start_year end_year]);
set(gca, 'FontName', 'Times New Roman', 'FontSize', 12);
