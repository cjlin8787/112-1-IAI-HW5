import torch
from torch.utils.data import DataLoader

from dataset import HW5Dataset
from model import HW5Model

dataset = HW5Dataset('train_dataset/meta.csv')
random_seed_generator = torch.Generator().manual_seed(42)

# TODO (5P): Create training and validation dataset based on `dataset`
# and `random_seed_generator`. You should split the dataset into 400
# samples for the training, and 100 samples for the validation. You
# should use `random_seed_generator` to ensure consistency of the
# result.
# CHECK: https://pytorch.org/docs/stable/data.html#torch.utils.data.random_split
train_dataset = None
val_dataset = None

# TODO (5P): Create training and validation data loader based on 
# `train_dataset` and `val_dataset`. Please set `shuffle=False` for the
# consistency of the result.
# CHECK: https://pytorch.org/tutorials/beginner/basics/data_tutorial.html#preparing-your-data-for-training-with-dataloaders
train_loader = None
val_loader = None

# Optional: change different hyperparameters, to get the best model.
hidden_size = 16
num_layers = 2
lr = 1e-3
patience = 5
max_n_epochs = 15

model = HW5Model(hidden_size=hidden_size, num_layers=num_layers, lr=lr)
model.train_epochs(train_loader, val_loader, patience=patience, max_n_epochs=max_n_epochs)

y_pred_prob = model.predict_prob(val_loader)
y_pred = model.predict(val_loader)
y_true = torch.concat([batch[2] for batch in val_loader]).numpy()
print(f'Accuracy on validation set: {(y_pred == y_true).sum() / len(y_true):.2f}')
print(f'Area under precision recall curve on validation set: {model.evaluate(y_pred_prob, y_true):.2f}')