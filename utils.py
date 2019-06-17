import numpy as np

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd import Variable
from torchvision import datasets, utils

#hyper params
num_epochs = 10
batch_size = 16
learning_rate = 0.001
beta = 1

#these are pretty close for pretty much any image library
#(in lieu of computing these statistics at runtime)
std = [0.229, 0.224, 0.225]
mean = [0.485, 0.456, 0.406]


#Paths. These can be changed in the future
#once we decide to split the data into train/valid/test
#for now this is good enough to test if the thing is working at all.

TRAIN_PATH = './data/'
VALID_PATH = './data/'
TEST_PATH = './kodim/'

def customized_loss(S_prime, C_prime, S, C, B):
    ''' Calculates loss specified on the paper.'''
    
    loss_cover = torch.nn.functional.mse_loss(C_prime, C)
    loss_secret = torch.nn.functional.mse_loss(S_prime, S)
    loss_all = loss_cover + B * loss_secret
    return loss_all, loss_cover, loss_secret

def denormalize(image, std, mean):
    ''' Denormalizes a tensor of images.'''

    for t in range(3):
        image[t, :, :] = (image[t, :, :] * std[t]) + mean[t]
    return image

def imshow(img, idx, learning_rate, beta):
    '''Prints out an image given in tensor format.'''
    
    img = denormalize(img, std, mean)
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.title('Example '+str(idx)+', lr='+str(learning_rate)+', B='+str(beta))
    plt.show()
    return

def gaussian(tensor, mean=0, stddev=0.1):
    '''Adds random noise to a tensor.'''
    
    noise = torch.nn.init.normal(torch.Tensor(tensor.size()), 0, 0.1)
    if tensor.is_cuda: noise = noise.cuda()
    return Variable(tensor + noise)


# Creates training set
train_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(
        TRAIN_PATH,
        transforms.Compose([
        # transforms.Scale(256),
        # transforms.RandomCrop(224),
        transforms.RandomResizedCrop(128, scale=(1,1), ratio=(1,1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean,
        std=std)
        ])), batch_size=batch_size, num_workers=1, 
        pin_memory=True, shuffle=True, drop_last=True)

# Creates test set
test_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(
        TEST_PATH, 
        transforms.Compose([
        transforms.RandomResizedCrop(128, scale=(1,1), ratio=(1,1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean,
        std=std)
        ])), batch_size=2, num_workers=1, 
        pin_memory=True, shuffle=True, drop_last=True)
