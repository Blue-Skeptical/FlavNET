import numpy as np
import os
import math
import time
import logging

import torch
import torch.linalg
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.utils import save_image
from torchvision import utils

import cv2
import copy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Resize input images at <imsize * imsize> pixels
# and transform into torch sensor
def GetLoader(imageSize):
    return transforms.Compose([
        transforms.Resize(imageSize),
        transforms.ToTensor()])


def ImageLoader(fileName,loader):
    image = Image.open(fileName)
    # Add one dimension
    #   image is:       [3,imsize,imsize]
    #   next will be:   [1,3,imsize,imsize]
    # This new dimension should be the images batch
    # We don't need it, but network expects this dimension
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)


def ShowImage(tensor, title=None, save=False, file_name=None):
    image = GetImageFromTensor(tensor)
    plt.imshow(image)
    if save and (file_name is not None) and isinstance(file_name,str):
        image.save(file_name)

    if title is not None:
        plt.title(title)
    plt.pause(1.001)

def ShowImages(tensors):
    for i,tensor in enumerate(tensors):
        img = GetImageFromTensor(tensor)
        plt.subplot(1,len(tensors),i+1)
        plt.imshow(img)
    plt.pause(1)

def GetImageFromTensor(tensor):
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = transforms.ToPILImage()(image)
    return image

def SaveImage(tensor, name):
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    #image = transforms.ToPILImage()(image)
    #image = cv2.cvtColor(image.detach().numpy(),cv2.COLOR_BGR2RGB)
    #image = np.asarray(image)
    #image = Image.fromarray(image.astype(np.uint8))
    #image.save(name)
    save_image(image,name)

# Used for style loss
def GramMatrix(matrix):
    a, b, c, d = matrix.size()
    #   a: batch size (1 in our case)
    #   b: channels (feature maps)
    # c,d: dimensions (image size)

    # To compute gram product, we use a resize version of input [a*b, c*d]
    features = matrix.view(a * b, c * d)
    G = torch.mm(features, features.t())  # GRAM product

    # Return the normalized G (divided by number of elements)
    return G.div(a * b * c * d)


def GetLayerName(layer, i):
    if isinstance(layer, nn.Conv2d):
        return 'conv_{}'.format(i)
    elif isinstance(layer, nn.ReLU):
        return 'relu_{}'.format(i)
    elif isinstance(layer, nn.MaxPool2d):
        return 'pool_{}'.format(i)
    elif isinstance(layer, nn.BatchNorm2d):
        return 'bn_{}'.format(i)
    else:
        raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))
