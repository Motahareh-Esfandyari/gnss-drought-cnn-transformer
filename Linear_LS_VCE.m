%% ========================================================================
%  GNSS-Drought : Drought Identification from GNSS Time Series (ISPRS 2026)
%  ------------------------------------------------------------------------
%  Function: Linear_LS_VCE.m
%  Purpose : Iterative LS-VCE for a linear model with a multi-component
%            stochastic model (Teunissen and Amiri-Simkooei, 2008).
%  ------------------------------------------------------------------------
%  Author  : Motahareh Esfandyari-Kaloukan
%  ========================================================================

function [scap, c, Error0, N,x] = Linear_LS_VCE(s0, Q, Epsilon, y, A)
m = length(y) ; p = length(s0) ; s = s0 ; x = inv(A'*A)*A'*y;
Error = 1000 ; Error0 = [] ;
c = 0 ;
while Error>Epsilon
    Qy = zeros(m) ;
    for k = 1:p
        Qy = Qy + s(k)*Q(:, :, k) ;
    end
    invQy = inv(Qy) ;
    PAo = eye(m) - A*inv(A'*invQy*A)*A'*invQy ;
    ecap = PAo*y ;
    for k = 1:p
        L0 = ecap'*invQy*Q(:, :, k)*invQy*ecap ;
        L(k, :) = (1/2)*trace(L0) ;
        for l = 1:p
            N0 = Q(:, :, k)*invQy ;
            N0 = N0*PAo ;
            N0 = N0*Q(:, :, l) ;
            N0 = N0*invQy ;
            N0 = N0*PAo ;
            N(k, l) = (1/2)*trace(N0) ;
        end
    end
    scap = inv(N)*L ;
    Error = norm(scap-s0) ;
    s0 = scap ;
    s = scap ;
    c = c + 1 ;
    Error0(c, :) = Error ;
end
end
