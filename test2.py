import matplotlib.pyplot as plt
import torch

from utils import *

"""w = torch.ones(3, requires_grad= True)
x = torch.rand(3, requires_grad= True)

model_output = (w*2*x).sum()
model_output.backward()
print(2*x)
print(w.grad)   # d(model_output)/dw
print(x.grad)   # d(model_output)/dx"""
IMG_DIM = 256
loader = GetLoader(IMG_DIM)
target_image_1 = ImageLoader("./images/me.jpg", loader)
target_image_2 = ImageLoader("./images/style2.jpg",loader)

input_image = torch.rand([1,3,IMG_DIM,IMG_DIM], requires_grad= True)

plt.ion()
ShowImage(input_image)

class MyModel(nn.Module):
    def __init__(self, target1,target2):
        super(MyModel,self).__init__()
        self.target1 = target1.detach()
        self.target2 = target2.detach()

    def forward(self,image):
        self.loss1 = F.mse_loss(image,self.target1)
        self.loss2 = F.mse_loss(image,self.target2)
        self.loss = self.loss1 + self.loss2
        return image

model = MyModel(target_image_1, target_image_2)
optimizer = torch.optim.Adam([input_image], lr=0.1)

while(True):
    plt.close('all')
    model(input_image)
    print("L1: " + str(model.loss1.item()))
    print("L2: " + str(model.loss2.item()))

    plt.pause(0.3)
    optimizer.zero_grad()
    model.loss.backward()
    optimizer.step()
    with torch.no_grad():
        input_image.clamp_(0, 1)
    ShowImage(input_image)
