%% ========================================================================
%  GNSS-Drought : Drought Identification from GNSS Time Series (ISPRS 2026)
%  ------------------------------------------------------------------------
%  Function: makePL2.m
%  Purpose : Vectorized variant of makePL.m for the power-law cofactor
%            matrix.
%  ------------------------------------------------------------------------
%  Author  : Motahareh Esfandyari-Kaloukan
%  ========================================================================

function U = makePL2(alpha, Year)
    Year = round(Year * 365.25);
    if size(Year, 1) == 1
        Year = Year';
    end

    m = Year(end);

    Year1(Year, 1) = Year;

    H(1) = 1;

    % ایجاد ماتریس تریولتز به صورت تدریجی
    A = zeros(m, m);
    for i = 1:m
        for j = 1:i
            A(i, j) = (alpha/2 + i - 2) * H(i - j + 1) / (i - j + 1);
        end
    end

    IDX = ismember(1:m, Year1);

    % ایجاد ماتریس نهایی با تقریب مقادیر ویژه محدود
    k = 100; % تعداد مقادیر ویژه محدود
    [U_svd, S, V] = svds(A(:, IDX), k);

    % تنظیم ماتریس U برابر با U_svd
    U = U_svd;

    return
end
