import argparse

from dataset import NerveClassificationDataset
from transform import preprocessing
from efficientnet_pytorch import EfficientNet
from trainer import Trainer

import torch
from torch.utils.data import DataLoader


parser = argparse.ArgumentParser()
parser.add_argument(
    '--batch_size', type=int, default=15
)
parser.add_argument(
    '--epoch', type=int, default=25
)
parser.add_argument(
    '--lr', type=float, default=0.001
)
parser.add_argument(
    '--dataset', type=str, default='./data/'
)
parser.add_argument(
    '--workers', type=int, default=4
)
parser.add_argument(
    '--save_model', type=str, default='./saved_model/'
)

cfg = parser.parse_args()
print(cfg)

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
print(device)

if __name__ == '__main__':
    ds_train = NerveClassificationDataset(root=cfg.dataset, train=True, transform=preprocessing)
    ds_test = NerveClassificationDataset(root=cfg.dataset, train=False, transform=preprocessing)
    dl_train = DataLoader(ds_train, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.workers)
    dl_test = DataLoader(ds_test, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.workers)

    print("DATA LOADED")

    efficient_list = ['efficientnet-b0', 'efficientnet-b1', 'efficientnet-b2', 'efficientnet-b3',
                    'efficientnet-b4', 'efficientnet-b5', 'efficientnet-b6', 'efficientnet-b7']
    model = EfficientNet.from_name(efficient_list[4])
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    trainer = Trainer(model, criterion, optimizer, device, None)
    fit = trainer.fit(dl_train, dl_test, num_epochs=cfg.epoch, checkpoints=cfg.save_model+model.__class__.__name__)
    torch.save(model.state_dict(), 'final'+'.pth')
