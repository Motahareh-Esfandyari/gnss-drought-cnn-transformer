# =========================================================================
#  GNSS-Drought : Drought Identification from GNSS Time Series (ISPRS 2026)
#  -------------------------------------------------------------------------
#  Script  : cnn_transformer.py
#  Purpose : PyTorch version of CNNTransformer.m. A 1D CNN extracts local
#            patterns from the yearly SSA feature vector and a Transformer
#            encoder (multi-head self-attention) models the interactions,
#            as described in Sec. 3.5 of the paper. Unlike the MATLAB
#            version, this uses the standard nn.TransformerEncoder, so no
#            custom layer is needed, and the train/test split is
#            CHRONOLOGICAL (earliest 70% train, latest 30% test), matching
#            the evaluation protocol of the paper.
#  Usage   : python cnn_transformer.py            (expects final_dataset_with_labels.csv)
#            python cnn_transformer.py --predict new_features.csv
#  -------------------------------------------------------------------------
#  Author  : Motahareh Esfandyari-Kaloukan
#  Paper   : doi.org/10.5194/isprs-annals-XI-3-2026-625-2026
# =========================================================================

import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             classification_report, ConfusionMatrixDisplay)

SEED = 1
CLASSES = ["drought", "normal", "wet"]


class CNNTransformer(nn.Module):
    """1D CNN front end + Transformer encoder + classification head."""

    def __init__(self, n_features, n_classes=3, d_model=32, n_heads=4,
                 n_layers=1, dropout=0.3):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, d_model, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        enc = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=64,
            dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(enc, num_layers=n_layers)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, n_classes),
        )

    def forward(self, x):                 # x: (batch, n_features)
        z = self.cnn(x.unsqueeze(1))      # (batch, d_model, n_features)
        z = z.permute(0, 2, 1)            # (batch, seq=n_features, d_model)
        z = self.transformer(z)
        z = z.mean(dim=1)                 # average over the sequence
        return self.head(z)


def load_dataset(path="final_dataset_with_labels.csv"):
    df = pd.read_csv(path).sort_values("Year").reset_index(drop=True)
    feat_cols = [c for c in df.columns
                 if c not in ("Year", "SPEI_Mean", "Label")]
    X = df[feat_cols].values.astype(np.float32)
    y = np.array([CLASSES.index(l) for l in df["Label"]])
    return df, X, y, feat_cols


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="final_dataset_with_labels.csv")
    ap.add_argument("--predict", default=None,
                    help="csv of unlabeled yearly features to classify")
    ap.add_argument("--epochs", type=int, default=400)
    args = ap.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    df, X, y, feat_cols = load_dataset(args.data)

    # Chronological 70/30 split (no temporal leakage), as in the paper
    n_train = int(round(0.7 * len(X)))
    mu, sd = X[:n_train].mean(0), X[:n_train].std(0) + 1e-9
    Xn = (X - mu) / sd
    Xtr, ytr = torch.tensor(Xn[:n_train]), torch.tensor(y[:n_train])
    Xte, yte = torch.tensor(Xn[n_train:]), torch.tensor(y[n_train:])
    print(f"train: {df.Year.iloc[0]}–{df.Year.iloc[n_train-1]}  "
          f"test: {df.Year.iloc[n_train]}–{df.Year.iloc[-1]}")

    # Class weights compensate the imbalance instead of oversampling
    counts = np.bincount(ytr.numpy(), minlength=3).astype(np.float32)
    weights = torch.tensor(counts.sum() / (3 * np.maximum(counts, 1)))

    model = CNNTransformer(n_features=X.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)
    lossf = nn.CrossEntropyLoss(weight=weights)

    best, best_state, patience = np.inf, None, 0
    for epoch in range(args.epochs):
        model.train()
        opt.zero_grad()
        loss = lossf(model(Xtr), ytr)
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            vloss = lossf(model(Xte), yte).item()
        if vloss < best - 1e-4:
            best, best_state, patience = vloss, model.state_dict(), 0
        else:
            patience += 1
            if patience > 60:          # early stopping
                break
    model.load_state_dict(best_state)
    print(f"stopped at epoch {epoch}, best val loss {best:.3f}")

    model.eval()
    with torch.no_grad():
        pred = model(Xte).argmax(1).numpy()
    acc = accuracy_score(yte.numpy(), pred)
    print(f"\nTest accuracy: {100*acc:.1f}%\n")
    print(classification_report(yte.numpy(), pred,
                                labels=[0, 1, 2], target_names=CLASSES,
                                zero_division=0))

    cm = confusion_matrix(yte.numpy(), pred, labels=[0, 1, 2])
    ConfusionMatrixDisplay(cm, display_labels=CLASSES).plot(cmap="Blues")
    plt.title("Confusion matrix (chronological test set)")
    plt.savefig("confusion_matrix.png", dpi=150, bbox_inches="tight")
    print("wrote confusion_matrix.png")

    torch.save({"state": model.state_dict(), "mu": mu, "sd": sd,
                "features": feat_cols}, "drought_cnn_transformer.pt")
    print("wrote drought_cnn_transformer.pt")

    if args.predict:
        new = pd.read_csv(args.predict)
        Xn2 = ((new[feat_cols].values.astype(np.float32)) - mu) / sd
        with torch.no_grad():
            p = model(torch.tensor(Xn2)).softmax(1).numpy()
        new["Predicted"] = [CLASSES[i] for i in p.argmax(1)]
        print("\n", new[["Year", "Predicted"]].to_string(index=False))


if __name__ == "__main__":
    main()
