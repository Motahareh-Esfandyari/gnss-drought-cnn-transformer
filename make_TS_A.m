%% ========================================================================
%  GNSS-Drought : Drought Identification from GNSS Time Series (ISPRS 2026)
%  ------------------------------------------------------------------------
%  Function: make_TS_A.m
%  Purpose : Builds the functional-model design matrix (intercept, trend,
%            annual and semi-annual harmonics) for a time vector in years.
%  ------------------------------------------------------------------------
%  Author  : Motahareh Esfandyari-Kaloukan
%  ========================================================================

function A=make_TS_A(t) % t is times in year
freq=[1,2]%,1*365.25/351.4,2*365.25/351.4,3*365.25/351.4,4*365.25/351.4,5*365.25/351.4,6*365.25/351.4,7*365.25/351.4,8*365.25/351.4,365.25/13.63,365.25/14.76];

A(:,1)=ones(length(t),1);
A(:,2)=t;

for i=1:size(freq,2)
A(:,2*i+1)=cos((2*pi*freq(i))*t);
A(:,2*i+2)=sin((2*pi*freq(i))*t);
end
