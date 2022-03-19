import numpy as np
import os
import math
import time
import logging
from enum import Enum
import numbers


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


class OptimizerSelector(Enum):
    ADAM = 1,
    SGD = 2

class FunctionalMode(Enum):
    InverseRepresentation = 1,
    FilterVisualization = 2

# Resize input images at <imsize * imsize> pixels
# and transform into torch sensor
def GetLoader(imageSize):
    return transforms.Compose([
        transforms.Resize(imageSize),
        transforms.ToTensor()])


def LoadTensorFromFile(fileName,loader):
    image = Image.open(fileName)
    # Add one dimension
    #   image is:       [3,imsize,imsize]
    #   next will be:   [1,3,imsize,imsize]
    # This new dimension should be the images batch
    # We don't need it, but network expects this dimension
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)

def GetImageReshapedFromFile(fileName, loader):
    tensor = LoadTensorFromFile(fileName,loader)
    image = GetImageFromTensor(tensor)
    return image

def GetTensorFromImage(img, require_grad = False):
    tensor = transforms.ToTensor()(img).to(device).unsqueeze(0)
    tensor.requires_grad = True
    return tensor


def NormalizeImage(img):
    # Normalization for ImageNet
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    return (img - mean)/std

def UnnormalizeImage(img):
    assert isinstance(img, np.ndarray), f'Expected numpy image got {type(img)}'

    if img.shape[0] == 3:  # if channel-first format move to channel-last (CHW -> HWC)
        img = np.moveaxis(img, 0, 2)

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    mean = mean.reshape(1, 1, -1)
    std = std.reshape(1, 1, -1)
    img = (img * std) + mean  # de-normalize
    img = np.clip(img, 0., 1.)  # make sure it's in the [0, 1] range

    return img

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


def GetCurrentPyramidalShape(original_shape, current_level, pyramid_depth = 4, scale_rate = 1.5):
    DIM_LIMITS = 10
    e = current_level - pyramid_depth + 1
    new_shape = np.round(np.float32(original_shape) * (scale_rate**e)).astype(np.int32)

    if new_shape[0] < DIM_LIMITS or new_shape[1] < DIM_LIMITS:
        logging.error("Pyramid generated a small image, reduce parameters")
        exit(0)
    return new_shape

def GetResizedTensorFromImage(img,shape):
    img = cv2.resize(img,(shape[1], shape[0]))

    return GetTensorFromImage(img,require_grad=True)

def BlurTensor(tensor, size = 3, sigma = 1):
    out = tensor.squeeze(0).detach()
    out = transforms.GaussianBlur(kernel_size=(size,size), sigma=sigma)(out)
    return out