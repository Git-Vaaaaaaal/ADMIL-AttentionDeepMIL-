from __future__ import print_function

import argparse

import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data_utils
from torch.autograd import Variable
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from torchvision import transforms
from sklearn.utils.class_weight import compute_class_weight

from definition import get_patients_and_labels, collate_fn, collate_fn_resnet, train_epoch_new, validate_epoch_new
from sklearn.model_selection import train_test_split
from model import AttentionResNet, GatedAttention, GatedAttentionFeatures
from torch.utils.data import DataLoader
from dataloader import DLBCLDataset, TilesBags

#Variables
clinical_csv = 'clinical_data.csv'
features_dir = 'features'
seed = 42
test_size = 0.15
val_size = 0.15
warmup_epochs = 5
max_epochs = 100
min_lr = 1e-6
output_dir = "results"
bag_weight = 0.8
max_grad_norm = 1.0
focal_gamma = 2.0
use_focal_loss = False
bag_dropout = 0.5
label_smoothing = 0.1
num_workers = 4

lr = 0.0005
weight_decay = 10e-5

model_mode = 1

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST bags Example')
parser.add_argument('--epochs', type=int, default=20, metavar='N',
                    help='number of epochs to train (default: 20)')

args = parser.parse_args()
args.cuda = False

torch.manual_seed(seed)

print('Load Train and Test Set')
loader_kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

#New data import
#Import Data
patients, labels, df = get_patients_and_labels(clinical_csv, features_dir)

# Split patients into train, validation, and test sets
trainval_patients, test_patients, trainval_labels, test_labels = train_test_split(
        patients, labels,
        test_size=test_size,
        random_state=seed,
        stratify=labels
    )
train_patients, val_patients, train_labels, val_labels = train_test_split(
        trainval_patients, trainval_labels,
        test_size=val_size,
        random_state=seed,
        stratify=trainval_labels
    )

if model_mode == 0:
    train_dataset = DLBCLDataset(train_patients, df, features_dir)
    val_dataset = DLBCLDataset(val_patients, df, features_dir)
    test_dataset = DLBCLDataset(test_patients, df, features_dir)
else :
    transform = transforms.Compose([
        transforms.ToTensor(),   #numpy to tensor
        ])
    img_path = "dataset_tiles_resnet" #Refaire le split pour les images
    label_path = "clinical_data.csv"
    train_dataset = TilesBags(img_path, label_path, transform)
    num = train_dataset.len_labels()
    print(num)
    val_dataset = TilesBags(img_path, label_path, transform)
    test_dataset = TilesBags(img_path, label_path, transform)

#Load data
train_loader = DataLoader(
        train_dataset, batch_size=1, shuffle=True,
        num_workers=0, collate_fn=collate_fn_resnet, pin_memory=True #mettre resnet si image raw
    )
val_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=False,
        num_workers=0, collate_fn=collate_fn_resnet, pin_memory=True
    )
test_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=False,
        num_workers=0, collate_fn=collate_fn_resnet, pin_memory=True
    )


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Class weights for imbalanced loss
train_labels_array = np.array([train_dataset.labels[p] for p in train_dataset.patients])
class_weights = compute_class_weight('balanced', classes=np.unique(train_labels_array), y=train_labels_array)
class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

#Old loading data
""" train_loader = data_utils.DataLoader(MnistBags(target_number=args.target_number,
                                               mean_bag_length=args.mean_bag_length,
                                               var_bag_length=args.var_bag_length,
                                               num_bag=args.num_bags_train,
                                               seed=args.seed,
                                               train=True),
                                     batch_size=1,
                                     shuffle=True,
                                     **loader_kwargs)

test_loader = data_utils.DataLoader(MnistBags(target_number=args.target_number,
                                              mean_bag_length=args.mean_bag_length,
                                              var_bag_length=args.var_bag_length,
                                              num_bag=args.num_bags_test,
                                              seed=args.seed,
                                              train=False),
                                    batch_size=1,
                                    shuffle=False,
                                    **loader_kwargs) """

print('Init Model')
if model_mode == 0 :
    model = GatedAttentionFeatures() #fonctionne avec features deja extraites
elif model_mode == 1 :
    model = AttentionResNet() # Attention simple, fonctionne avec image
    resnet_model = True
elif model_mode == 2 :
    model = GatedAttention() # Gated Attenion, fonctionne avec image

if args.cuda:
    model.cuda()

optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=weight_decay)

# --- Schedulers ---
def warmup_lambda(epoch):
    return (epoch + 1) / warmup_epochs if epoch < warmup_epochs else 1.0

warmup_scheduler = LambdaLR(optimizer, lr_lambda=warmup_lambda)
main_scheduler = CosineAnnealingLR(
    optimizer,
    T_max=max_epochs - warmup_epochs,
    eta_min=min_lr
)

best_val_error = float("inf")
best_epoch = 0

print(f"Starting training for {max_epochs} epochs...")

for epoch in range(max_epochs):

    train_loss, train_error = train_epoch_new(
        model,
        train_loader,
        optimizer,
        device,
        max_grad_norm=max_grad_norm,
        resnet = resnet_model
    )

    val_loss, val_error = validate_epoch_new(model,val_loader,device, resnet=resnet_model)

    print(
        f"Epoch {epoch+1:03d} | "
        f"Train Loss: {train_loss:.4f} | Train Err: {train_error:.4f} | "
        f"Val Loss: {val_loss:.4f} | Val Err: {val_error:.4f}"
    )
    if val_error < best_val_error:
        best_val_error = val_error
        best_epoch = epoch
        model_path = os.path.join(output_dir, "best_model.pt")
        torch.save(model.state_dict(), model_path)
        print(f"  >>> New Best Model Saved (Val Err: {val_error:.4f})")

    # --- scheduler step ---
    if epoch < warmup_epochs:
        warmup_scheduler.step()
    else:
        main_scheduler.step()

print(
    f"Training finished. "
    f"Best Val Error {best_val_error:.4f} at epoch {best_epoch+1}."
)


#Old approaches

""" def train(epoch):
    model.train()
    train_loss = 0.
    train_error = 0.
    for batch_idx, (data, label) in enumerate(train_loader):
        bag_label = label[0]
        if args.cuda:
            data, bag_label = data.cuda(), bag_label.cuda()
        data, bag_label = Variable(data), Variable(bag_label)

        # reset gradients
        optimizer.zero_grad()
        # calculate loss and metrics
        loss, _ = model.calculate_objective(data, bag_label)
        train_loss += loss.data[0]
        error, _ = model.calculate_classification_error(data, bag_label)
        train_error += error
        # backward pass
        loss.backward()
        # step
        optimizer.step()

    # calculate loss and error for epoch
    train_loss /= len(train_loader)
    train_error /= len(train_loader)

    print('Epoch: {}, Loss: {:.4f}, Train error: {:.4f}'.format(epoch, train_loss.cpu().numpy()[0], train_error))


def test():
    model.eval()
    test_loss = 0.
    test_error = 0.
    
    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(test_loader):
            bag_label = label[0]
            instance_labels = label[1]
            if args.cuda:
                data, bag_label = data.cuda(), bag_label.cuda()
            data, bag_label = Variable(data), Variable(bag_label)
            loss, attention_weights = model.calculate_objective(data, bag_label)
            test_loss += loss.data[0]
            error, predicted_label = model.calculate_classification_error(data, bag_label)
            test_error += error

            if batch_idx < 5:  # plot bag labels and instance labels for first 5 bags
                bag_level = (bag_label.cpu().data.numpy()[0], int(predicted_label.cpu().data.numpy()[0][0]))
                instance_level = list(zip(instance_labels.numpy()[0].tolist(),
                                    np.round(attention_weights.cpu().data.numpy()[0], decimals=3).tolist()))

                print('\nTrue Bag Label, Predicted Bag Label: {}\n'
                    'True Instance Labels, Attention Weights: {}'.format(bag_level, instance_level))

    test_error /= len(test_loader)
    test_loss /= len(test_loader)

    print('\nTest Set, Loss: {:.4f}, Test error: {:.4f}'.format(test_loss.cpu().numpy()[0], test_error))


if __name__ == "__main__":
    print('Start Training')
    for epoch in range(1, args.epochs + 1):
        train(epoch)
    print('Start Testing')
    test()
 """