import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import glob
import os
import numpy as np
from PIL import Image
import torchvision.models as models
import copy
from torchvision.utils import save_image
import PIL
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from lib import data_loader as dl
from lib import models as md
from lib import model_device_io as io
from torch.autograd import Variable

train_set = dl.OuluNpu('Data_Processed_Train')
train_loader = DataLoader(train_set, batch_size=16, shuffle=True, num_workers=0)

test_set = dl.OuluNpu('Data_Processed_Test')
test_loader = DataLoader(test_set, batch_size=16, shuffle=False, num_workers=0)

# device = io.GetCudaDevice(cuda = 1, seed = 123, log=True)
model = md.ResNet().to('cuda')
optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.5, 0.999))


def test():
    model.eval()
    loss, correct = 0, 0
    with torch.no_grad():
        for idx, datas in enumerate(test_loader):
            img, access = datas
            img = img.to('cuda')
            access = access.to('cuda')
            output = model(img)
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(access.view_as(pred)).sum().item()
            loss += F.cross_entropy(output, access).item() * len(access)

    loss /= len(test_loader.dataset)
    percentage = 100. * correct / len(test_loader.dataset)
    print('***Validation Result:\tLoss: {:.4f}\tAccuraccy: {:.2f}% ({}/{})\n'.format(loss, percentage,
                                                                                     correct, len(test_loader.dataset)))
    return percentage


def train(train_epoch=100):
    best_acc = 0
    for epoch in range(train_epoch):
        model.train()
        for batch_idx, datas in enumerate(train_loader):
            optimizer.zero_grad()
            img, access = datas
            img = img.to('cuda')
            access = access.to('cuda')

            # train
            output = model(img)
            loss = F.cross_entropy(output, access)
            loss.backward()
            optimizer.step()

            # log
            if (batch_idx + 1) % max(1, int(len(train_loader) / 5)) == 0:
                print('Training Epoch: {}/{}\tBatch: {}/{}\tLoss: {:.4f}'.format(epoch + 1,
                                                                                 train_epoch, batch_idx + 1,
                                                                                 len(train_loader), loss.item()))
        acc = test()
        if acc > best_acc:
            best_acc = acc
            io.saveModel('models/best.pth', model, optimizer)


# %%
train()