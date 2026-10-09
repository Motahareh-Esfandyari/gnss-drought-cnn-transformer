# Python version of the pipeline

This folder is a Python port of the MATLAB pipeline of the paper. It uses
the standard PyTorch Transformer encoder, so nothing custom is needed, and
the train/test split is chronological (earliest 70 percent for training),
matching the evaluation protocol described in the paper.

## Install

```
pip install -r requirements.txt
```

## Run, in order

```
python preprocessing.py P242.cwu.igs14.csv     # -> ssaresult.csv
python feature_extraction.py                   # -> yearly_features.csv
python labeling.py ClimateEngine.csv           # -> final_dataset_with_labels.csv
python cnn_transformer.py                      # train + evaluate
python cnn_transformer.py --predict new_features.csv   # classify new years
```

The P242 csv comes from the NOTA/UNAVCO archive and the SPEI csv from
Climate Engine (gridMET 1-year SPEI export), the same inputs as the MATLAB
version.

## Differences from the MATLAB version

The offset test here uses a white-noise covariance; the full white plus
flicker version of the paper can be reproduced by building the covariance
with the power-law routines in the MATLAB lib folder. SSA grouping is done
automatically by the dominant frequency of each eigentriple instead of
manual component selection. Class imbalance is handled with class weights
in the loss instead of oversampling. With only about 19 yearly samples the
exact accuracy depends on the split and the seed, so expect numbers close
to, but not identical with, the MATLAB run.
