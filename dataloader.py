"""Pytorch dataset object that loads MNIST dataset as bags."""

from tkinter import Image

import numpy as np
import torch
import torch.utils.data as data_utils
from torchvision import datasets, transforms
import pandas as pd
import os

#Entrainement avec tuile extraites
class TilesBags():
    def __init__(
        self,
        img_path,
        label_path,
        transform=None
    ):
        self.img_path = img_path
        self.transform = transform

        df = pd.read_csv(label_path)
        self.labels = {
            str(row["patient+AF8-id"]): row["status"]
            for _, row in df.iterrows()
        }

        self.patients = [
            p for p in os.listdir(img_path)
            if p in self.labels
        ]

    def __len__(self):
        return len(self.patients)

    def __getitem__(self, idx):
        patient = self.patients[idx]
        patient_dir = os.path.join(self.img_path, patient)

        images = os.listdir(patient_dir)

        bag = []
        for img_name in images:
            img_path = os.path.join(patient_dir, img_name)
            img = Image.open(img_path).convert("RGB")
            if self.transform:
                img = self.transform(img)
            bag.append(img)

        bag = torch.stack(bag)  # (N, C, H, W)

        label = torch.tensor(self.labels[patient], dtype=torch.long)

        return bag, label


#Entrainement pour features
class DLBCLDataset():    
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
