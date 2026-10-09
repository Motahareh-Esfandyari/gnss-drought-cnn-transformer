%% ========================================================================
%  GNSS-Drought : Drought Identification from GNSS Time Series (ISPRS 2026)
%  ------------------------------------------------------------------------
%  Function: WhiteNoise.m
%  Purpose : White noise cofactor matrix (identity).
%  ------------------------------------------------------------------------
%  Author  : Motahareh Esfandyari-Kaloukan
%  ========================================================================

function [Qw] = WhiteNoise(m)
Qw = eye(m) ;
end
