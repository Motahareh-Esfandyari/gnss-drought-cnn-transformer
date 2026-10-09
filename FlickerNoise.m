%% ========================================================================
%  GNSS-Drought : Drought Identification from GNSS Time Series (ISPRS 2026)
%  ------------------------------------------------------------------------
%  Function: FlickerNoise.m
%  Purpose : Flicker noise cofactor matrix approximation.
%  ------------------------------------------------------------------------
%  Author  : Motahareh Esfandyari-Kaloukan
%  ========================================================================

function [Qf] = FlickerNoise(t1, m)
Qf = (9/8)*eye(m) ;
h = waitbar(0, 'Flicker noise cofactor matrix construction...') ;
for k = 1:m
    tau = abs(t1(k+1:end) - t1(k)) ;
    tau = tau' ;
    Qf(k, k+1:end) = (9/8)*(1 - (log10(tau)/log10(2)+2)/24) ;
    tau = tau' ;
    Qf(k+1:end, k) = (9/8)*(1 - (log10(tau)/log10(2)+2)/24) ;
    waitbar(k/m)
end
close(h)
end
