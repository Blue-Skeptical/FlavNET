import random

import torch

from utils import *

VGG = models.vgg19(pretrained=True).features.to(device).eval()

imsize = 256
loader = GetLoader(imsize)
unloader = transforms.ToPILImage()

input1 = ImageLoader("./images/natalie.jpg", loader)
input2 = ImageLoader("./images/style1.jpg", loader)

ou_img = torch.rand(3, imsize, imsize)

class NET(nn.Module):
    def __init__(self):
        super(NET, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 3, 3, padding=1),
            #nn.ReLU(),
            #nn.AvgPool2d(2),
            #nn.Linear(64,128)
        )

    def forward(self, img):
        return self.net(img)


net = NET()
output = net(input1)

ShowImage(output)
print(output.size())
plt.show()