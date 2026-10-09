# =========================================================================
#  GNSS-Drought : Drought Identification from GNSS Time Series (ISPRS 2026)
#  -------------------------------------------------------------------------
#  Script  : labeling.py
#  Purpose : Python version of Labeling.m. Merges the yearly SSA features
#            with the yearly mean SPEI and labels each year following the
#            rule of the paper: SPEI < -1 drought, SPEI > 1 wet,
#            otherwise normal. Writes final_dataset_with_labels.csv.
#  Usage   : python labeling.py spei.csv
#            spei.csv needs a date column and a SPEI column (a Climate
#            Engine gridMET export works as is).
#  -------------------------------------------------------------------------
#  Author  : Motahareh Esfandyari-Kaloukan
#  Paper   : doi.org/10.5194/isprs-annals-XI-3-2026-625-2026
# =========================================================================

import sys
import pandas as pd

spei_path = sys.argv[1] if len(sys.argv) > 1 else "ClimateEngine.csv"

ssa = pd.read_csv("yearly_features.csv")

spei = pd.read_csv(spei_path)
spei.columns = [c.strip() for c in spei.columns]
datecol = spei.columns[0]
speicol = [c for c in spei.columns if "spei" in c.lower()][0]
spei["Year"] = pd.to_datetime(spei[datecol]).dt.year
spei_yearly = (spei.groupby("Year")[speicol].mean()
               .rename("SPEI_Mean").reset_index())

final = ssa.merge(spei_yearly, on="Year", how="inner")


def label(v):
    if v < -1:
        return "drought"
    if v > 1:
        return "wet"
    return "normal"


final["Label"] = final["SPEI_Mean"].apply(label)
final.to_csv("final_dataset_with_labels.csv", index=False)
print(final[["Year", "SPEI_Mean", "Label"]].to_string(index=False))
print("wrote final_dataset_with_labels.csv")
