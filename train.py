import os
import argparse
from tqdm import tqdm

import torch
from torchvision import models

from dataloader import get_data_loader
from utils import set_seed


def get_parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=20, help='the iterations of training')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate of traing, eg. 1e-3, 1e-2')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='weight decay of training')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--gpu', type=int, default=0, help='gpu id')
    parser.add_argument('--resume', type=str, default=None, help='the path of pre-trained model param dict')
    opt = parser.parse_args()
    return opt


if __name__ == '__main__':
    opt = get_parse_opt()

    set_seed(opt.seed)
    
    num_classes = 10
    device = torch.device('cuda:' + str(opt.gpu) if torch.cuda.is_available() else 'cpu')

    # get model
    model = models.resnet18()

    # modify last fc layer
    num_in_features = model.fc.in_features
    model.fc = torch.nn.Linear(num_in_features, num_classes)

    # load pretrained params
    if opt.resume is not None:
        checkpoint = torch.load(opt.resume, map_location=device)
        model.load_state_dict(checkpoint)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)

    model.train()
    model = model.to(device)

    epochs = opt.epochs
    train_loader = get_data_loader('train')
    
    for epoch in range(epochs):
        correct, total = 0, 0
        running_loss = 0.0
        train_bar = tqdm(train_loader)
        for i, (images, attrs) in enumerate(train_bar):
            labels, colors = attrs[:, 0], attrs[:, 1]
            labels = labels.to(device)
            images = images.to(device)

            output = model(images)
            loss = criterion(output, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            predicted = torch.argmax(output, dim=1)
            correct += (predicted == labels).sum().item()
            total += images.shape[0]
            running_loss += loss.item()

            train_bar.desc = "train epoch[{}/{}] loss:{:.4f} acc:{:.4f}".format(epoch + 1, epochs, running_loss / (i + 1), correct / total)

    save_path = 'resnet18.pth'
    torch.save(model.state_dict(), save_path)

