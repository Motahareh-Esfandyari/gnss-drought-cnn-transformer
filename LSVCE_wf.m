%% ========================================================================
%  GNSS-Drought : Drought Identification from GNSS Time Series (ISPRS 2026)
%  ------------------------------------------------------------------------
%  Function: LSVCE_wf.m
%  Purpose : Wrapper that estimates white + flicker noise variance components
%            of a series and returns the covariance matrix Qx.
%  ------------------------------------------------------------------------
%  Author  : Motahareh Esfandyari-Kaloukan
%  ========================================================================

function [Qx,SS]=LSVCE_wf(x,VCE_method)
% LSVCE_wf - compute fl+w noise variance components and covariance matrix Qx.
% Input:  x - the time series or x_d the detrended time series
%    VCE_method -  'A_FULL'    design matrix consists of trend+ annual+ semi annual
%                  'A_NULL'    design matrix is NULL matrix
%                  'A_trend'   design matrix consists of trend
% Output: Qx - full covariance matrix of data (w+fl)
%         SS - noise parameters computed by LSVCE

n=length(x);
time=[1:n]/365.25;
if strcmp(VCE_method,'A_FULL')==1
    A=make_TS_A(time);
elseif strcmp(VCE_method,'A_NULL')==1
    A=[];
elseif strcmp(VCE_method,'A_trend')==1
    A=[ones(n,1) time'];
end
Qw=eye(n,n);
Qf=makePL(1,time-time(1)+0.0027);
Qf=Qf*((365.25)^(1/4))^-2;

sigma0_1=1;sigma0_2=0;
[SIGMA,Q_sigma,Qx,x_cap,P_A_orto,e_y_cap,Q_y_inv]=vce2(A,Qw,Qf,x,sigma0_1,sigma0_2,'white noise','flicker noise');
SS=SIGMA(:,end);
