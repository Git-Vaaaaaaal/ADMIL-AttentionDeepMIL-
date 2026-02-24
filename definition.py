import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split



def get_patients_and_labels(clinical_csv, features_dir):
    """Get all patients with features and their labels."""
    df = pd.read_csv(clinical_csv)
    id_col = df.columns[0]
    
    all_patients = []
    all_labels = []
    
    for _, row in df.iterrows():
        if pd.isna(row[id_col]) or pd.isna(row['status']):
            continue
        pid = str(int(row[id_col]))
        feature_path = os.path.join(features_dir, f"{pid}.pt")
        if os.path.exists(feature_path):
            all_patients.append(pid)
            all_labels.append(int(row['status']))
    
    return np.array(all_patients), np.array(all_labels), df


def collate_fn(batch):
    """
    Custom collate function for variable bag sizes.
    Since bag sizes vary, we process one bag at a time (batch_size=1).
    For batch_size > 1, we would need padding.
    If different instance is inside a bag, it will arrange the size to fit the batch_size
    """
    if len(batch) == 1:
        # Single sample - no need for padding
        item = batch[0]
        return {
            'features': item['features'],      # (N, feature_dim)
            'coords': item['coords'],          # (N, 2)
            'label': item['label'].unsqueeze(0),  # (1,)
            'patient_id': [item['patient_id']],
            'tile_names': [item['tile_names']]
        }
    else:
        # Multiple samples - would need padding
        # For simplicity, we concatenate but keep track of bag boundaries
        # This is mainly for batch_size=1 use case
        features_list = [int(item['features']) for item in batch]
        coords_list = [item['coords'] for item in batch]
        labels = torch.stack([item['label'] for item in batch])
        patient_ids = [item['patient_id'] for item in batch]
        tile_names = [item['tile_names'] for item in batch]
        
        # For batch > 1, return lists (model should handle one at a time)
        return {
            'features': features_list,
            'coords': coords_list,
            'label': labels,
            'patient_id': patient_ids,
            'tile_names': tile_names
        }
    
def collate_fn_resnet(batch): #n utilise pas les coords d une image
    # batch = [(bag, label), (bag, label), ...]
    item=[]
    bags = [item[0] for item in batch]
    labels = [item[1] for item in batch]

    # batch_size = 1 → on enlève la liste
    bags = bags[0]
    labels = labels[0].unsqueeze(0)

    return {
        'features': bags,   # (N, C, H, W)
        'label': labels    # (1,)
    }


def train_epoch_new(
    model,
    loader,
    optimizer,
    device,
    max_grad_norm=1.0,
    bag_dropout=0.0,
    resnet=False
):
    model.train()
    train_loss = 0.0
    train_error = 0.0

    criterion = torch.nn.BCEWithLogitsLoss()

    for batch in loader:
        features = batch['features']
        label = batch['label']

        if isinstance(features, list):
            features = features[0]
            label = label[0:1]

        data = features.to(device)
        bag_label = label.float().to(device)

        if bag_dropout > 0 and data.size(0) > 10:
            n_keep = max(10, int(data.size(0) * (1 - bag_dropout)))
            keep_idx = torch.randperm(data.size(0))[:n_keep]
            data = data[keep_idx]

        optimizer.zero_grad()
        if resnet == True :
            Y_prob, Y_hat, A = model(data)
            logits = Y_prob

            loss = criterion(logits.view(-1), bag_label)
        
        else:
            logits, A = model(data)
            loss = criterion(logits.view(-1), bag_label)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
        optimizer.step()

        train_loss += loss.item()

        # --- erreur de classification ---
        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).float()
        train_error += (preds != bag_label).float().mean().item()

    train_loss /= len(loader)
    train_error /= len(loader)

    return train_loss, train_error


def validate_epoch_new(model, loader, device, resnet=False):
    model.eval()
    loss_sum, error_sum = 0.0, 0.0

    criterion = torch.nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for batch in loader:
            features, label = batch['features'], batch['label']

            # Cas où le loader renvoie une liste (bag unique)
            if isinstance(features, list):
                features, label = features[0], label[0:1]

            data, bag_label = features.to(device), label.float().to(device)

            if resnet == True :
                Y_prob, Y_hat, A = model(data)
                logits = Y_prob

                loss = criterion(logits.view(-1), bag_label)
        
            else:
                logits, A = model(data)
                loss = criterion(logits.view(-1), bag_label)

            loss_sum += loss.item()

            # --- classification error ---
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).float()
            error_sum += (preds != bag_label).float().mean().item()

    n = len(loader)
    return loss_sum / n, error_sum / n



""" def validate_epoch_new(model, loader, device):
    model.eval()
    loss_sum, error_sum = 0.0, 0.0

    with torch.no_grad():
        for batch in loader:
            features, label = batch['features'], batch['label']

            if isinstance(features, list):
                features, label = features[0], label[0:1]

            data, bag_label = features.to(device), label.to(device)

            loss, _ = model.calculate_objective(data, bag_label)
            error, _ = model.calculate_classification_error(data, bag_label)

            loss_sum += loss.item()
            error_sum += error

    n = len(loader)
    return loss_sum / n, error_sum / n """