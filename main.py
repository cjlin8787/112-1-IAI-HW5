from torch.utils.data import DataLoader

from dataset import HW5Dataset
from model import HW5Model

train_dataset = HW5Dataset('sample/meta.txt', 'train')
val_dataset = HW5Dataset('sample/meta.txt', 'val')

train_loader = DataLoader(train_dataset, batch_size=1, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=1, pin_memory=True)

model = HW5Model()
model.setup_model()
model.train(train_loader, val_loader)
model.predict(val_loader)
