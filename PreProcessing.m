%% ========================================================================
%  GNSS-Drought : Drought Identification from GNSS Time Series (ISPRS 2026)
%  ------------------------------------------------------------------------
%  Script  : PreProcessing.m
%  Purpose : Preprocessing of the P242 vertical series for the drought study:
%            gap detection and filling, white + flicker LS-VCE noise model,
%            A-mode offset detection (chi-square, alpha = 0.02), outlier
%            screening on the residuals, and SSA decomposition (window 1095)
%            for the trend/annual/semi-annual components.
%  ------------------------------------------------------------------------
%  Author  : Motahareh Esfandyari-Kaloukan
%  Paper   : Esfandyari-Kaloukan et al., ISPRS Annals XI-3-2026, 625-630
%            doi.org/10.5194/isprs-annals-XI-3-2026-625-2026
%  ========================================================================

clc; clear; close all; format long g;

%% Reading data
[time, nor, eas, ver] = readvars('P242.cwu.igs14.csv');
%ver = readvars('SERI.csv');
% t = 1:length(ver);
ver = ver(15:6819,:); time = time(15:6819,:);
%%ver = ver(:,:); time = time(:,:);
t = 1:length(ver);
%t = juliandate(time, 'modifiedjuliandate');
figure(1)
title('GNSS time series - p242'); xlabel('Time (years)'); ylabel('Original time series & Trend');
plot(t,ver,'LineWidth',1,'color',[0.13 0.12 0.87]); grid on;
%plot(T,Imf,'LineWidth',2.5,'color','r')
set(gca,'fontname','times new roman','LineWidth',1.5,'fontweight','bold','fontsize',12);
XTickLabel = 2005:1:2023 ;
XTick = t(1):365.25:t(end);
%(XTickLabel-t(1))
set(gca, 'XTick', XTick)
set(gca, 'XTickLabel', XTickLabel)
%subplot(2,1,2); hold on;`
xlabel('Time (years)'); ylabel('Vertical (mm)');title('GNSS time series - p242');
%% Finding GAP

m1 = min(t(:));    %% the start of recording data
m2 = max(t(:));    %% the end of recording data

T = m1 : m2;
T = T';
t1 = m1 :m2;
t1 = t1';
%%%% from the next step we are going to fing cells which are gaps %%%%
gap = diff(t);
idx = find(gap > 1);   %%% this shpws where a gap is located

F = 0;
LL = zeros(numel(idx) + 1 ,1);

for i = 1 : numel(idx)

    idxx(i) = idx(i) + F;
    LL (i + 1) = t(idx(i) + 1) - t (idx(i)) - 1 ;
    F = sum(LL);

end

%L = cumsum(L);
for i = 1: numel(idx)
    L = t(idx(i) + 1) - t(idx(i));
    t1 (idxx(i) + 1 : idxx(i) + L - 1 , 1) =  nan;
%     T (idxx(i) + 1 : idxx(i) + L - 1 , 1) =   T (idxx(i) + 1 : idxx(i) + L - 1 , 1);
end

%%%% ver
v = zeros(numel(t1),1);
v(:) = nan;

for i = 1 : numel(t1)

    if isnan(t1(i)) == 1
        continue
    end

    k = find( t == t1 (i) );
    v (i) = ver(k);

end
figure(2)
plot(t1,v,'LineWidth',1,'color',[0.13 0.12 0.87]); grid on;
XTickLabel = 2005:3:2023 ;
XTick = t(1):1095:t(end);
%(XTickLabel-t(1))
set(gca, 'XTick', XTick)
set(gca, 'XTickLabel', XTickLabel)
set(gca,'fontname','times new roman','LineWidth',1.5,'fontweight','bold','fontsize',12)
title('GNSS Time Series - P242 (GAP filling)'); xlabel('Time(years)'); ylabel('Vertical(mm)');
v = fillgaps(v);
% %% Finding outlier
% TF = isoutlier(v,'movmean',2000);
% figure(2)
% subplot(2,1,1)
% plot(t1,v,t1(TF), v(TF),'x','LineWidth',1,'color',[0.13 0.12 0.87]);
% legend('Data','Outlier')
% set(gca,'fontname','times new roman','LineWidth',1.5,'fontweight','bold','fontsize',12)
% XTickLabel = 2005:3:2022 ;
% XTick = t(1):1095:t(end);
% set(gca, 'XTick', XTick)
% set(gca, 'XTickLabel', XTickLabel)
% title('GNSS Time Series - P043 Station'); xlabel('Time(years)'); ylabel('Vertical(mm)');
% fidx = find(TF==1);
% v(TF==1) = nan;
% subplot(2,1,2)
% plot(t1, v,t1(TF), v(TF),'x','LineWidth',1,'color',[0.13 0.12 0.87]);
% legend('Data','Outlier')
% set(gca,'fontname','times new roman','LineWidth',1.5,'fontweight','bold','fontsize',12)
% XTickLabel = 2005:3:2022 ;
% XTick = t(1):1095:t(end);
% set(gca, 'XTick', XTick)
% set(gca, 'XTickLabel', XTickLabel)
% title('GNSS Time Series - P043 Station'); xlabel('Time(years)'); ylabel('Vertical(mm)');
% v = fillgaps(v);
save p242 t1 v
%% Noise estimation:

m = length(v);
w1 = 2*pi/365.25;
w2 = 2*w1;

Epsilon = 1e-2;

A = [ones(m,1) T sin(T*w1) cos(T*w2) sin(2*T*w1) cos(2*T*w2)];

Qw = WhiteNoise(m);
Qf = FlickerNoise(T,m);

s0 = [1, 1]';
Q(:,:,1)=Qw;
Q(:,:,2)=Qf;

[scap, c, Error, N]=Linear_LS_VCE(s0, Q, Epsilon, v, A);
p=length(s0); s=s0;
Qy = zeros(m);
    for k = 1:p
        Qy=Qy+s(k)*Q(:,:,k);
    end
%save GNSS_Analysis_p242 scap c Error N Qy Qw Qf A time

 %% Offset detection:
alpha = 0.02;
threshold = chi2inv(1-alpha,1);
sim_num = 7; %Y=y;
it = 1:sim_num;

[OFF A] = offset_Uni(A,v,Qy,T,T,threshold,0,it);
% A = A(:,[1:2 7:end]);
A = A(:,7:end);
Qi=inv(Qy);
xcap = inv(A'*Qi*A)*(A'*Qi*v);

% targetValue = OFF(:,1);
% for i = 1:length(targetValue)
%     v(targetValue(i)) = xcap(i);
% end
ycap = A*xcap;
ecap = v - ycap ;

figure(3)
plot(T,v,'LineWidth',1,'color',[0.13 0.12 0.87]); grid on;
XTickLabel = 2005:3:2023 ;
XTick = t(1):1095:t(end);
%(XTickLabel-t(1))
set(gca, 'XTick', XTick)
set(gca, 'XTickLabel', XTickLabel)
set(gca,'fontname','times new roman','LineWidth',1.5,'fontweight','bold','fontsize',12)
title('GNSS Time Series - P242 (Noise estimation & Offset detection)'); xlabel('Time(years)'); ylabel('Vertical(mm)');
% % xline(OFF(1,1),'--',"Offset no.1",'color',[238,0,0]/255)
% % xline(OFF(2,1),'--',"Offset no.1",'color',[238,0,0]/255)
% % xline(OFF(3,1),'--',"Offset no.1",'color',[238,0,0]/255)
% % xline(OFF(4,1),'--',"Offset no.1",'color',[238,0,0]/255)
% % xline(OFF(5,1),'--',"Offset no.1",'color',[238,0,0]/255)
%% Finding outlier
TF = isoutlier(ecap,'movmean',500);
figure(4)
subplot(2,1,1)
plot(t1,ecap,t1(TF), ecap(TF),'x','LineWidth',1,'color',[0.13 0.12 0.87]); grid on;
legend('Data','Outlier')
set(gca,'fontname','times new roman','LineWidth',1.5,'fontweight','bold','fontsize',12)
XTickLabel = 2005:3:2023 ;
XTick = t(1):1095:t(end);
set(gca, 'XTick', XTick)
set(gca, 'XTickLabel', XTickLabel)
title('GNSS Time Series - P242'); xlabel('Time(years)'); ylabel('Vertical(mm)');
fidx = find(TF==1);
ecap(TF==1) = nan;
subplot(2,1,2)
plot(t1, ecap,t1(TF), ecap(TF),'x','LineWidth',1,'color',[0.13 0.12 0.87]); grid on;
legend('Data','Outlier')
set(gca,'fontname','times new roman','LineWidth',1.5,'fontweight','bold','fontsize',12)
XTickLabel = 2005:3:2023 ;
XTick = t(1):1095:t(end);
set(gca, 'XTick', XTick)
set(gca, 'XTickLabel', XTickLabel)
title('GNSS Time Series - P242'); xlabel('Time(years)'); ylabel('Vertical(mm)');
ecap = fillgaps(ecap);%% Finding outlier
TF = isoutlier(ecap,'movmean',2000);
figure(5)
subplot(2,1,1)
plot(t1,ecap,t1(TF), ecap(TF),'x','LineWidth',1,'color',[0.13 0.12 0.87]); grid on;
legend('Data','Outlier')
set(gca,'fontname','times new roman','LineWidth',1.5,'fontweight','bold','fontsize',12)
XTickLabel = 2005:3:2023 ;
XTick = t(1):1095:t(end);
set(gca, 'XTick', XTick)
set(gca, 'XTickLabel', XTickLabel)
title('GNSS Time Series - P242'); xlabel('Time(years)'); ylabel('Vertical(mm)');
fidx = find(TF==1);
ecap(TF==1) = nan;
subplot(2,1,2)
plot(t1, ecap,t1(TF), ecap(TF),'x','LineWidth',1,'color',[0.13 0.12 0.87]); grid on;
legend('Data','Outlier')
set(gca,'fontname','times new roman','LineWidth',1.5,'fontweight','bold','fontsize',12)
XTickLabel = 2005:3:2023 ;
XTick = t(1):1095:t(end);
set(gca, 'XTick', XTick)
set(gca, 'XTickLabel', XTickLabel)
title('GNSS Time Series - P242 (Outlier detection)'); xlabel('Time(years)'); ylabel('Vertical(mm)');
ecap = fillgaps(ecap);

%% Sigular Spectrum Analysis:

time = T;
L = 1095;      %1754;
%[r, A, E, L, Np, C_BK] = SSA(T, v, L);
r=10;
[y,r,vr]=ssadow(v,L);
%[y, Power_y, xi, Power, Z, T, D] = autoSSA(v, r, L)
%xt1 = v - r(:, 1); % Detrending
%---->save GNSS_Analysis__deterended_toiy r y OFF T xcap v
% filename = 'p242resxt.xlsx';
% writematrix(r, filename);
% filename1 = 'p242rest.xlsx';
% writematrix(T, filename1);
% filename2 = 'p242resecap.xlsx';
% writematrix(ecap, filename2);
figure(6);
plot(T,ecap,'LineWidth',1,'color',[0.13 0.12 0.87]); grid on; plot(T,y,'LineWidth',1,'color','r')
set(gca,'fontname','times new roman','LineWidth',1.5,'fontweight','bold','fontsize',12);
title('GNSS time series - p242'); xlabel('Time (year)'); ylabel('Original & reconstructed time series');
XTickLabel = 2005:3:2023 ;
XTick = t(1):1059:t(end);
%(XTickLabel-t(1))
set(gca, 'XTick', XTick)
set(gca, 'XTickLabel', XTickLabel)

subplot(2,1,2); hold on;
xlabel('Time (year)'); ylabel('Residuals time series');
plot(T,r,'LineWidth',1,'color',[0.13 0.12 0.87]); grid on;
%[0.207843137 0.796078431 0.517647059]
set(gca,'fontname','times new roman','LineWidth',1.5,'fontweight','bold','fontsize',12)
XTickLabel = 2005:3:2023 ;
XTick = t(1):1059:t(end);
%(XTickLabel-t(1))
set(gca, 'XTick', XTick)
set(gca, 'XTickLabel', XTickLabel)

% figure(5)
% plot(T,y,'LineWidth',1);
% XTickLabel = 2005:3:2022 ;
% XTick = t(1):1059:t(end);
% %(XTickLabel-t(1))
% set(gca, 'XTick', XTick)
% set(gca, 'XTickLabel', XTickLabel)
% set(gca,'fontname','times new roman','LineWidth',1.5,'fontweight','bold','fontsize',12)
% title('The first 10 components of the GNSS time series-P043'); xlabel('Time(years)'); ylabel('F(time)');
% figure(5)
% plot(T,xt,'LineWidth',1,'color',[0.13 0.12 0.87])
% XTickLabel = 2005:3:2022 ;
% XTick = t(1):1059:t(end);
% %(XTickLabel-t(1))
% set(gca, 'XTick', XTick)
% set(gca, 'XTickLabel', XTickLabel)
% set(gca,'fontname','times new roman','LineWidth',1.5,'fontweight','bold','fontsize',12)
% title('GNSS Time Series - P043'); xlabel('Time(years)'); ylabel('Vertical(mm)');
% figure(6)
% plot(T,xt,'LineWidth',1,'color',[0.13 0.12 0.87])
% XTickLabel = 2007:3:2022 ;
% XTick = t(1):1059:t(end);
% %(XTickLabel-t(1))
% set(gca, 'XTick', XTick)
% set(gca, 'XTickLabel', XTickLabel)
% set(gca,'fontname','times new roman','LineWidth',1.5,'fontweight','bold','fontsize',12)
% title('Dterended GNSS Time Series - P242'); xlabel('Time(years)'); ylabel('Vertical (mm)');
% figure(10)
% plot(T,r(:, 1),'LineWidth',1,'color',[0.13 0.12 0.87]);
% XTickLabel = 2007:3:2022 ;
% XTick = t(1):1059:t(end);
% %(XTickLabel-t(1))
% set(gca, 'XTick', XTick)
% set(gca, 'XTickLabel', XTickLabel)
% set(gca,'fontname','times new roman','LineWidth',1.5,'fontweight','bold','fontsize',12)
% title('The trend of the GNSS time series - P242'); xlabel('Time(years)'); ylabel('Vertical(mm)');
