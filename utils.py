import numpy as np
import os
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.utils import save_image

import copy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
imsize = 1024 if torch.cuda.is_available() else 512

# Resize input images at <imsize * imsize> pixels
# and transform into torch sensor
loader = transforms.Compose([
    transforms.Resize(imsize),
    transforms.ToTensor()])

# Reconvert into PIL image
unloader = transforms.ToPILImage()


def ImageLoader(fileName):
    image = Image.open(fileName)
    # Add one dimension
    #   image is:       [3,imsize,imsize]
    #   next will be:   [1,3,imsize,imsize]
    # This new dimension should be the images batch
    # We don't need it, but network expects this dimension
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)


def ShowImage(tensor, title=None):
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)


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
