# =========================================================================
#  GNSS-Drought : Drought Identification from GNSS Time Series (ISPRS 2026)
#  -------------------------------------------------------------------------
#  Script  : feature_extraction.py
#  Purpose : Python version of FeatureExtraction.m. For every year,
#            computes the mean, energy and amplitude of the SSA trend,
#            annual and semi-annual components (the features of Table 2
#            of the paper) and writes yearly_features.csv.
#  Usage   : python feature_extraction.py          (expects ssaresult.csv)
#  -------------------------------------------------------------------------
#  Author  : Motahareh Esfandyari-Kaloukan
#  Paper   : doi.org/10.5194/isprs-annals-XI-3-2026-625-2026
# =========================================================================

import pandas as pd

data = pd.read_csv("ssaresult.csv", parse_dates=["date"])
data["Year"] = data["date"].dt.year

rows = []
for year, g in data.groupby("Year"):
    row = {"Year": year}
    for name, col in [("Trend", "trend"), ("Annual", "annual"),
                      ("SemiAnnual", "semi")]:
        x = g[col]
        row[f"{name}_Mean"] = x.mean()
        row[f"{name}_Energy"] = (x**2).sum()
        row[f"{name}_Amplitude"] = x.max() - x.min()
    rows.append(row)

features = pd.DataFrame(rows)
features.to_csv("yearly_features.csv", index=False)
print(features.round(3).to_string(index=False))
print("wrote yearly_features.csv")
