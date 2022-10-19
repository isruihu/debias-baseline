import os
import argparse

import torch
from torchvision import models

from dataloader import get_data_loader
from utils import set_seed
import csv


def get_parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--gpu', type=int, default=0, help='gpu id')
    parser.add_argument('--resume', type=str, default='resnet18.pth', help='the path of pre-trained model param dict')
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

    model.eval()
    model = model.to(device)

    test_loader = get_data_loader('test')
    
    with torch.no_grad():
        record = []
        for i, (images, img_names) in enumerate(test_loader):
            images = images.to(device)
            output = model(images)
            predicted = torch.argmax(output, dim=1)

            # record the batch result
            for j in range(len(img_names)):
                record.append((int(img_names[j].split('.')[0]), int(predicted[j])))
        
        record = sorted(record, key=lambda x:x[0])
        # save result
        with open("result.csv", "w") as csvfile: 
            writer = csv.writer(csvfile)
            #先写入columns_name
            writer.writerow(["id","class"])
            #写入多行用writerows
            writer.writerows(record)

