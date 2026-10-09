%% ========================================================================
%  GNSS-Drought : Drought Identification from GNSS Time Series (ISPRS 2026)
%  ------------------------------------------------------------------------
%  Function: offset_Uni.m
%  Purpose : Offset detection by hypothesis testing: adds candidate step
%            functions to the design matrix and tests them against a
%            chi-square threshold.
%  ------------------------------------------------------------------------
%  Author  : Motahareh Esfandyari-Kaloukan
%  ========================================================================

function [OFF A]=offset_Uni(A,v,Qy,RY,YEAR,CV,option,it_sim)
m=length(v);
[m n]=size(A);
Qi=inv(Qy);

it=0;
Jex=1;
Ak=[];
OFF=[];maxS=1000;
while 1
    [it_sim it maxS]
    it=it+1;
    A=[A Ak];
    AtQi=A'*Qi;
    HH=inv(AtQi*A)*AtQi;
    Pa=A*HH;
    E=v-Pa*v;
    QiE=Qi*E;
    Sig=(E'*QiE)/(m-n);
    M1=QiE*inv(Sig)*QiE';
    M2=Qi-Qi*Pa;
    S=zeros(m,1);
    for j=1:m
        Aj=zeros(m,1); Aj(m-j+1:m,1)=1; %MN
        if sum(ismember(Jex,m-j+1))
            S(m-j+1,1)=0;
        else
            AjtM1=Aj'*M1; AjtM2=Aj'*M2;
            S(m-j+1,1) = (AjtM1*Aj)/(AjtM2*Aj);
        end
    end
    S=conv([0.25 0.5 0.25],[S(1);S;S(end)]);
    S(1:2)=[]; S(end-1:end)=[];
    [maxS maxJ]=max(S);
    if it<=1
        OFF=[OFF;[maxJ maxS]];
        Jex=[Jex maxJ];
        Ak=zeros(m,1); Ak(maxJ:m,1)=1;
    elseif maxS>CV & it<=1   %MN
        OFF=[OFF;[maxJ maxS]];
        Jex=[Jex maxJ];
        Ak=zeros(m,1); Ak(maxJ:m,1)=1;
    else
        break
    end
end
if option==1
    OFF(end,:)=[];
    [Sort1 idx]=sort(OFF(:,1));
    OFF=OFF(idx,:);
    OFF=[YEAR(OFF(:,1)) OFF];

    A(:,7:end)=A(:,[6+idx]);
    Xh=inv(A'*Qi*A)*A'*Qi*v;
    Yh=A*Xh; Eh=v-Yh;
    Qx=kron(S,inv(A'*Qi*A));

    save Results8 OFF A Qi S Y Xh Yh Eh Qx
end
