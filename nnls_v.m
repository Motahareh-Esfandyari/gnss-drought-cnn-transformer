%% ========================================================================
%  GNSS-Drought : Drought Identification from GNSS Time Series (ISPRS 2026)
%  ------------------------------------------------------------------------
%  Function: nnls_v.m
%  Purpose : Non-negative least squares solution for variance components,
%            keeps the estimated sigmas physically meaningful (>= 0).
%  ------------------------------------------------------------------------
%  Author  : Motahareh Esfandyari-Kaloukan
%  ========================================================================

function [s Qs] = nnls_v(N,L)
p=length(L);
mu0=-L;
s0=zeros(p,1);
stest=s0+1;
s=s0;
while norm(s-stest)>1e-12
    for k=1:p
        s(k,1)=max(0,s0(k)-mu0(k)/N(k,k));
        mu0 = mu0+(s(k)-s0(k))*N(:,k);
    end
    stest=s0;
    s0=s;
end
idx=find(s==0);
Ct=zeros(length(idx),p);
for i=1:length(idx)
    Ct(i,idx(i))=1;
end
C=Ct';
Ni=inv(N);
Pco=eye(p)-C*inv(Ct*Ni*C)*Ct*Ni;
Qs=Ni*Pco;
return
