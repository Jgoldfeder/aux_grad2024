import os
import torch
import torch.nn as nn
import wandb

from dataset_factory import create_dataset, SubSampler
from timm.models import create_model
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import dual
import argparse
import numpy as np
import torch.optim as optim

def train(model, trainloader, optimizer, criterion, device,args):
    model.train()
    train_loss = 0
    batch_count = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        if args.dual:
            loss = criterion(outputs, targets,seperate=True)
            model.balance(loss)
            loss = sum(loss)
        else:
            loss = criterion(outputs, targets)
            loss.backward()
            
        
        optimizer.step()
        train_loss += loss.item()

        batch_count += 1
    return train_loss / batch_count



def valid_category(model, validloader, criterion, device):
    model.eval()
    correct = 0
    test_total = 0
    valid_loss = 0
    batch_count = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(validloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            test_total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            loss = criterion(outputs, targets)
            valid_loss += loss.item()
            batch_count += 1
    return valid_loss / batch_count, correct / test_total





data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((230,230)),
        transforms.RandomRotation(15,),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
    ]),
    'eval': transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
    ]),
    'test': transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
    ]),
}


if __name__ == "__main__":
    # Params setup
    parser = argparse.ArgumentParser(description="Aux Reg")
    
    parser.add_argument(
        "--model", type=str, help="Image encoder in [vgg19, resnet110, resnet32]"
    )
    
    parser.add_argument("--level", type=int, default=100, help="Data level.")
    parser.add_argument("--batch-size", type=int, default=128, help="batch size.")
    parser.add_argument('--pretrained', action='store_true', default=False)
    parser.add_argument('--dual', action='store_true', default=False,
                   help='dual mode.')
    parser.add_argument(
        "--data-dir",
        type=str,
        help="Directory where datasets are stored",
        default="./data",
    )

    parser.add_argument("--dataset", type=str, help="Dataset to train on")
    parser.add_argument('--num-classes', type=int, default=None, metavar='N',
                   help='number of label classes (Model default if None)')
    parser.add_argument('--experiment', default='', type=str, metavar='NAME',
                   help='name of train experiment, name of sub-folder for output')
    parser.add_argument('--name', default='', type=str, metavar='NAME',
                   help='name of run')    
    args = parser.parse_args()

    
    wandb.init(project=args.experiment, config=args,name=args.name)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num_workers = 4

    # Loads train, validation, and test data
    dataset_train = create_dataset(
        args.dataset,
        root=args.data_dir,
        split="train",
        download=True,
    )
    dataset_test = create_dataset(
        args.dataset,
        root=args.data_dir,
        split="test",
        download=True,
    )

    dataset_train.transform = data_transforms['train']
    dataset_test.transform = data_transforms['test']


    train_dataloader = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False)
    
    criterion = nn.CrossEntropyLoss()

    
    in_chans=3
    model = create_model(
        args.model,
        pretrained=args.pretrained,
        in_chans=in_chans,
        num_classes=args.num_classes,
    )
    if args.dual:
        model = dual.DualModelSimple(model,args)
        criterion = dual.DualLoss(criterion,[1,1])
    model.to(device)
    
    optimizer = torch.optim.SGD(
        model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4
    )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.001, weight_decay=1e-4
    )
    
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[30, 60], gamma=0.1
    )
    epoch_stop = 90


    epoch_start = 0
    min_valid_loss = float("inf")
    max_valid_acc = 0.0
    train_loss = []
    valid_loss = []
    valid_acc = []

    
    # Trains model from epoch_start to epoch_stop
    for epoch in range(epoch_start, epoch_stop):
        new_train_loss = train(model, train_dataloader, optimizer, criterion, device,args)
        new_valid_loss, new_valid_acc = valid_category(
            model, test_dataloader, criterion, device
        )
        
        scheduler.step(epoch)
        train_loss.append(new_train_loss)
        valid_loss.append(new_valid_loss)
        valid_acc.append(new_valid_acc)
        print(
            "Epoch {} train loss {} | valid loss {} | valid acc {}".format(
                epoch + 1, new_train_loss, new_valid_loss, new_valid_acc
            )
        )
        if new_valid_acc > max_valid_acc or (
            new_valid_acc == max_valid_acc and new_valid_loss < min_valid_loss
        ):
            min_valid_loss = new_valid_loss
            max_valid_acc = new_valid_acc
