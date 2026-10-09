%% ========================================================================
%  GNSS-Drought : Drought Identification from GNSS Time Series (ISPRS 2026)
%  ------------------------------------------------------------------------
%  Function: vce2.m
%  Purpose : Two-component variance component estimation with named noise
%            structures, returns the estimated sigmas and their precision.
%  ------------------------------------------------------------------------
%  Author  : Motahareh Esfandyari-Kaloukan
%  ========================================================================

function [SIGMA,Q_sigma,Q_y,x_cap,P_A_orto,e_y_cap,Q_y_inv]=vce2(A,Q_1,Q_2,y,sigma0_1,sigma0_2,Q_1_str,Q_2_str) %%Q_1_str eg: 'white noise'
Q{1}=Q_1;Q{2}=Q_2;
delta1=1;delta2=1;
SIGMA=[];
m=length(Q_1);
while max([delta1,delta2])>10^(-10)
    Q_y=sigma0_1*Q{1}+sigma0_2*Q{2};
    Q_y_inv=inv(Q_y);
    if size(A,1)==0
    P_A_orto=eye(m);
    else
    P_A_orto=eye(m)-A*inv(A'*Q_y_inv*A)*A'*Q_y_inv;
    end
    e_cap=P_A_orto*y;
    B=Q_y_inv*P_A_orto;
    for i=1:2
        l(i,1)=0.5*e_cap'*Q_y_inv*Q{i}*Q_y_inv*e_cap;
        for j=1:2
            n(i,j)=0.5*trace(B*Q{i}*B*Q{j});
        end
    end
    [sigma Q_sigma] = nnls_v(n,l);  % non negative LS_VCE
    delta1=abs(sigma(1)-sigma0_1);delta2=abs(sigma(2)-sigma0_2);
    sigma0_1=sigma(1);sigma0_2=sigma(2);
    SIGMA=[SIGMA,sigma]
end
Q_y=sigma0_1*Q{1}+sigma0_2*Q{2};
Q_sigma=inv(n);
%adjustment
Q_y_inv=inv(Q_y);
if size(A,1)==0
P_A_orto=eye(m);
x_cap=[];
y_cap=[];
Q_x_cap=[];
Q_y_cap=[];
e_y_cap=[];
else
P_A_orto=eye(m)-A*inv(A'*Q_y_inv*A)*A'*Q_y_inv;
x_cap=inv(A'*Q_y_inv*A)*(A'*Q_y_inv*y);
y_cap=A*x_cap;
Q_x_cap=inv(A'*Q_y_inv*A);
Q_y_cap=A*Q_x_cap*A';
e_y_cap=P_A_orto*y;
end

Eigenvalue=eig(Q_y);
s=size(SIGMA);
x=[1:s(1,2)];
end
