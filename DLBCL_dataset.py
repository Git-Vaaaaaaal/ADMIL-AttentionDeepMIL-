import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split



class DLBCLDataset(Dataset):    
    def __init__(self, patient_ids, labels_df, features_dir, label_col='status'):
        """
        Args:
            patient_ids: List of patient IDs to include
            labels_df: DataFrame with patient_id and label columns
            features_dir: Directory containing .pt feature files
            label_col: Name of the label column in labels_df
        """
        self.patient_ids = patient_ids
        self.features_dir = features_dir
        self.label_col = label_col
        
        # Create patient_id to label mapping
        # Handle the column name issue (patient+AF8-id -> patient_id)
        id_col = labels_df.columns[0]  # First column is patient_id
        self.labels = {}
        for _, row in labels_df.iterrows():
            pid = str(int(row[id_col])) if pd.notna(row[id_col]) else None
            if pid and pd.notna(row[label_col]):
                self.labels[pid] = int(row[label_col])  #Create dictionnary of patient id and label
        
        # Filter patient_ids to only those with labels and features
        #Extract the embedded data of patient using id
        self.valid_patients = []
        for pid in patient_ids:
            pid_str = str(pid)
            feature_path = os.path.join(features_dir, f"{pid_str}.pt")
            if pid_str in self.labels and os.path.exists(feature_path):
                self.valid_patients.append(pid_str)
        
        print(f"Dataset initialized with {len(self.valid_patients)} patients")
    
    def __len__(self):
        return len(self.valid_patients)
    
    def __getitem__(self, idx):
        patient_id = self.valid_patients[idx]
        
        # Load features
        feature_path = os.path.join(self.features_dir, f"{patient_id}.pt")
        data = torch.load(feature_path, weights_only=False)
        
        # Handle both new format (dict) and legacy format (tensor only)
        if isinstance(data, dict):
            features = data['features']  # Shape: (N, feature_dim)
            coords = data['coords']      # Shape: (N, 2) in (x, y) pixel format
            tile_names = data.get('tile_names', [])
        else:
            # Legacy format: data is just the features tensor
            features = data
            # Create dummy coordinates (will break heatmap generation)
            coords = torch.zeros((features.shape[0], 2), dtype=torch.long)
            tile_names = []
            print(f"  Warning: {patient_id} uses legacy format without coordinates")
        
        # Get label
        label = self.labels[patient_id]
        
        return {
            'features': features,
            'coords': coords,
            'label': torch.tensor(label, dtype=torch.long),
            'patient_id': patient_id,
            'tile_names': tile_names
        }

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
        features_list = [item['features'] for item in batch]
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


def train_epoch_new(
    model,
    loader,
    optimizer,
    device,
    max_grad_norm=1.0,
    bag_dropout=0.0
):
    model.train()
    train_loss = 0.0
    train_error = 0.0

    for batch_idx, batch in enumerate(loader):

        # --- extraction des données (comme dans train_epoch 1) ---
        features = batch['features']
        label = batch['label']

        if isinstance(features, list):
            features = features[0]
            label = label[0:1]

        data = features.to(device)
        bag_label = label.to(device)

        # --- bag dropout ---
        if bag_dropout > 0 and data.size(0) > 10:
            n_keep = max(10, int(data.size(0) * (1 - bag_dropout)))
            keep_idx = torch.randperm(data.size(0))[:n_keep]
            data = data[keep_idx]

        # --- reset gradients ---
        optimizer.zero_grad()

        # --- loss & error via le modèle (logique du 2ᵉ training) ---
        loss, _ = model.calculate_objective(data, bag_label)
        error, _ = model.calculate_classification_error(data, bag_label)

        train_loss += loss.item()
        train_error += error

        # --- backward ---
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
        optimizer.step()

    train_loss /= len(loader)
    train_error /= len(loader)

    return train_loss, train_error



def validate_epoch_new(model, loader, device):
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
    return loss_sum / n, error_sum / n