from utils import AverageMeter
from lib.stanford40_dataset import Stanford40Action
from lib.pascal_voc_dataset import VOCAction
from model.ho_relation_net import horelation_resnet50_v1d_st40

import time
import tqdm

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch import optim
from torch.nn import functional as F

import torchmetrics as tm

import torchvision
import torchvision.transforms as transforms
import pickle

## Args
num_workers = 0
batch_size = 1
device = 'cuda' if torch.cuda.is_available() else 'cpu'
epochs = 20
lr = 3e-5
max_lr = 1e-6

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def num_trainable_params(model):
  nums = sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6
  return nums

def save_best_model(model, output_name, loss, epoch):
  torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'loss': loss
}, f'/content/{output_name}.pth')

def get_dataset(dataset):
    train_tranform = transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.ToTensor()
    ])

    if dataset == 'std40':
        root = 'Datasets/Stanford40'
        train_dataset = Stanford40Action(root=root, transform=train_tranform, split='train', augment_box=True, load_box=True)
        val_dataset = Stanford40Action(root=root, transform=test_transform, split='test', augment_box=True, load_box=True)
    elif dataset == 'voc2012':
        root = 'Datasets/Voc2012'
        train_dataset = VOCAction(root=root, split='train', augment_box=True,
                                         load_box=True)
        val_dataset = VOCAction(root=root, split='val', augment_box=True,
                                       load_box=True)

    return train_dataset, val_dataset

def get_dataloader(val_dataset, batch_size, num_workers):
    """Get dataloader."""
    data =DataLoader(val_dataset, batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_fn)
    return data

def collate_fn(batch):
    data, label, hbox, box, pose = zip(*batch)
    return data, label, hbox, box, pose

def train(net_all, train_loader, valid_loader, loss_fn, optimizer, scheduler, metric):
    net_all.to(device)
    best_loss = 1e5
    all_train_loss = []
    all_valid_loss = []

    all_train_acc = []
    all_valid_acc = []

    for epoch in range(epochs):
        train_loss, accuracy, valid_loss, accuracy_valid, best_loss = train_one_epoch(net_all, train_loader, valid_loader, loss_fn, optimizer, scheduler, metric, epoch, best_loss)
        all_train_loss.append(train_loss)
        all_valid_loss.append(valid_loss)
        all_train_acc.append(accuracy)
        all_valid_acc.append(accuracy_valid)

    return all_train_loss, all_valid_loss, all_train_acc, all_valid_acc

def train_one_epoch(model, train_loader, valid_loader, loss_fn, optimizer, scheduler, metric, epoch=None, best_loss=1e5):

  model.train()
  train_loss = AverageMeter()
  metric.reset()

  with tqdm.tqdm(train_loader, unit='batch') as tr_loader:
    for data, labels, hbox, box, pose in tr_loader:
      tr_loader.set_description(f'Epoch: {epoch}')

      data = [item.to(device) for item in data]
      data = torch.stack(data)

      labels = [item.to(device) for item in labels]

      lebl = [item[0][4:5] for item in labels]
      lebl = torch.tensor(lebl, dtype=torch.long).to(device)

      bbox_human = hbox[0].to(device)

      box = [item.to(device) for item in box]
      pose = [item.to(device) for item in pose]

      output = model(data, bbox_human, box, pose)
      loss = loss_fn(output.unsqueeze(0), lebl)

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      metric.update(output.argmax(dim=0).unsqueeze(0), lebl)
      train_loss.update(loss.item(), n=len(lebl))

      tr_loader.set_postfix(loss=train_loss.avg)

  scheduler.step()
  accuracy = metric.compute()
  print(f'Train Accuracy {accuracy}, LR={scheduler.get_last_lr()}\n')

  #Valid Section
  valid_loss = AverageMeter()
  model.eval()
  metric.reset()

  with torch.no_grad():
    with tqdm.tqdm(valid_loader, unit='batch') as vl_loader:
      for data, labels, hbox, box, pose in vl_loader:
          data = [item.to(device) for item in data]
          data = torch.stack(data)

          labels = [item.to(device) for item in labels]

          lebl = [item[0][4:5] for item in labels]
          lebl = torch.tensor(lebl, dtype=torch.long).to(device)

          bbox_human = hbox[0].to(device)

          box = [item.to(device) for item in box]
          pose = [item.to(device) for item in pose]

          output = model(data, bbox_human, box, pose)
          loss = loss_fn(output.reshape(1, -1), lebl)

          valid_loss.update(loss.item(), n=len(lebl))
          metric.update(output.argmax(dim=0).unsqueeze(0), lebl)

          vl_loader.set_postfix(loss=valid_loss.avg)

  accuracy_valid = metric.compute()
  print(f'Loss Valid: {valid_loss.avg:.2f}, Valid Accuracy: {accuracy_valid:.2f}\n')

  if valid_loss.avg < best_loss:
    output_name = f'model{epoch}'
    save_best_model(model, output_name, loss, epoch)
    best_loss = valid_loss.avg
    print('Model Saved!!\n')

  return train_loss.avg, accuracy, valid_loss.avg, accuracy_valid, best_loss

if __name__ == '__main__':

    # train_dataset, val_dataset = get_dataset('std40')
    train_dataset, val_dataset = get_dataset('voc2012')

    val_dataset, test_dataset = torch.utils.data.random_split(val_dataset, [0.3, 0.7])

    # # 10 percent of dataset
    # train_dataset, _ = torch.utils.data.random_split(train_dataset, [0.004, 0.996])

    train_loader = get_dataloader(train_dataset, batch_size= batch_size, num_workers=num_workers)
    valid_loader = get_dataloader(val_dataset, batch_size=batch_size, num_workers=num_workers)
    test_loader = get_dataloader(test_dataset, batch_size=batch_size, num_workers=num_workers)

    net = horelation_resnet50_v1d_st40()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)
    metric = tm.Accuracy(task="multiclass", num_classes=40).to(device)

    item = train(net, train_loader, valid_loader, loss_fn, optimizer, scheduler, metric)
    
    with open('filename.pkl', 'wb') as handle:
      pickle.dump(item, handle)

    print('Finish!!')