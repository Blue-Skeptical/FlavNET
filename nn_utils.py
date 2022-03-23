import numpy as np
import os
import math
import time
import logging
from enum import Enum
import numbers
from GUI.GUI_utils import *

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
    SGD = 2,
    GD = 3

class Losses(Enum):
    MEAN = 1,
    MSE = 2

class FunctionalMode(Enum):
    InverseRepresentation = 1,
    FilterVisualization = 2

class PretrainedNet(Enum):
    vgg16 = 1,
    vgg19 = 2

class GDOptimizer():
    def __init__(self,params,lr,direction='positive'):
        self.params = params
        self.lr = lr
        self.direction = direction

    def zero_grad(self):
        for p in self.params:
            p.grad.data = torch.zeros_like(p.grad)

    @torch.no_grad()
    def step(self):
        if self.direction == 'positive':
            for p in self.params:
                p.data += p.grad.data * self.lr
        elif self.direction == 'negative':
            for p in self.params:
                p.data -= p.grad.data * self.lr





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

class CascadeGaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing separately for each channel (depthwise convolution).

    Arguments:
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.

    """
    def __init__(self, kernel_size, sigma):
        super().__init__()

        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size, kernel_size]

        cascade_coefficients = [0.5, 1.0, 2.0]  # std multipliers, hardcoded to use 3 different Gaussian kernels
        sigmas = [[coeff * sigma, coeff * sigma] for coeff in cascade_coefficients]  # isotropic Gaussian

        self.pad = int(kernel_size[0] / 2)  # assure we have the same spatial resolution

        # The gaussian kernel is the product of the gaussian function of each dimension.
        kernels = []
        meshgrids = torch.meshgrid([torch.arange(size, dtype=torch.float32) for size in kernel_size])
        for sigma in sigmas:
            kernel = torch.ones_like(meshgrids[0])
            for size_1d, std_1d, grid in zip(kernel_size, sigma, meshgrids):
                mean = (size_1d - 1) / 2
                kernel *= 1 / (std_1d * math.sqrt(2 * math.pi)) * torch.exp(-((grid - mean) / std_1d) ** 2 / 2)
            kernels.append(kernel)

        gaussian_kernels = []
        for kernel in kernels:
            # Normalize - make sure sum of values in gaussian kernel equals 1.
            kernel = kernel / torch.sum(kernel)
            # Reshape to depthwise convolutional weight
            kernel = kernel.view(1, 1, *kernel.shape)
            kernel = kernel.repeat(3, 1, 1, 1)
            kernel = kernel.to(device)

            gaussian_kernels.append(kernel)

        self.weight1 = gaussian_kernels[0]
        self.weight2 = gaussian_kernels[1]
        self.weight3 = gaussian_kernels[2]
        self.conv = F.conv2d

    def forward(self, input):
        input = F.pad(input, [self.pad, self.pad, self.pad, self.pad], mode='reflect')

        # Apply Gaussian kernels depthwise over the input (hence groups equals the number of input channels)
        # shape = (1, 3, H, W) -> (1, 3, H, W)
        num_in_channels = input.shape[1]
        grad1 = self.conv(input, weight=self.weight1, groups=num_in_channels)
        grad2 = self.conv(input, weight=self.weight2, groups=num_in_channels)
        grad3 = self.conv(input, weight=self.weight3, groups=num_in_channels)

        return (grad1 + grad2 + grad3) / 3