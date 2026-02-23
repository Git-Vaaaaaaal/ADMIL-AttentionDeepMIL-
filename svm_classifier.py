from __future__ import print_function

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sklearn import svm
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from definition import (
    get_patients_and_labels,
    collate_fn,
)
from dataloader import DLBCLDataset
from model import GatedAttentionFeatures


#Variables
test_size = 0.20
seed = 42
clinical_csv = "clinical_data.csv"
features_dir = "features"

#Import Data
patients, labels, df = get_patients_and_labels(clinical_csv, features_dir)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Split patients into train, validation, and test sets
train_patients, val_patients, train_labels, val_labels = train_test_split(
        patients, labels,
        test_size=test_size,
        random_state=seed,
        stratify=labels
    )

train_dataset = DLBCLDataset(train_patients, df, features_dir)
val_dataset = DLBCLDataset(val_patients, df, features_dir)

#Load data
train_loader = DataLoader(
        train_dataset, batch_size=1, shuffle=True,
        num_workers=0, collate_fn=collate_fn, pin_memory=True
    )
val_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=False,
        num_workers=0, collate_fn=collate_fn, pin_memory=True
    )

model = GatedAttentionFeatures(
    input_dim=768,
    hidden_dim=128
).to(device)

model.eval()  # très important


def extract_bag_embeddings(model, loader, device):
    embeddings = []
    labels = []

    with torch.no_grad():
        for batch in loader:
            H = batch["features"].to(device)   # (K, 768)
            y = batch["label"].item()

            Z, A = model(H)                   # Z: (1, 768)
            Z = Z.squeeze(0).cpu().numpy()    # (768,)

            embeddings.append(Z)
            labels.append(y)

    return np.vstack(embeddings), np.array(labels)


X_train, y_train = extract_bag_embeddings(model, train_loader, device)
X_val, y_val     = extract_bag_embeddings(model, val_loader, device)

clf = svm.SVC(
    kernel="rbf",
    probability=True,
    random_state=seed
)

clf.fit(X_train, y_train)

y_train_pred = clf.predict(X_train)
y_val_pred   = clf.predict(X_val)

y_val_proba = clf.predict_proba(X_val)

print("Train accuracy:", accuracy_score(y_train, y_train_pred))
print("Val accuracy  :", accuracy_score(y_val, y_val_pred))
print("Val log-loss  :", log_loss(y_val, y_val_proba))
