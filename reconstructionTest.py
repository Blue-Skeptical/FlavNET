import matplotlib.pyplot as plt
import torch

from utils import *

IMG_DIM = 256
loader = GetLoader(IMG_DIM)
target_image = ImageLoader("./images/noi.jpg", loader)

input_image = ImageLoader("./images/white.jpg", loader)#torch.rand([1,3,IMG_DIM,IMG_DIM], requires_grad= True)

plt.ion()
ShowImage(input_image)

cnn = models.vgg19(pretrained=True).features.to(device).eval()
# VGG are trained with images normalized with these parameters.
# So we save them to normalize our input images, as VGG expects
cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

class ContentLayer(nn.Module):
    def __init__(self, target):
        super(ContentLayer,self).__init__()
        self.target = target

    def forward(self,image):
        self.loss = F.mse_loss(image, self.target)
        return image

class Normalization(nn.Module):
    """ Used to normalize the input images as
    told before in "LOAD PRETRAINED VGG" section """
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        # Reshape mean and std to make them [C,1,1] where C is channel
        # This way, the can work with tensor of shape [B,C,H,W]
        # Is the same as making: .view(1,C,1,1)
        # So you can broadcast and make the forward expressions
        self.mean = mean.view(-1, 1, 1).detach()
        self.std = std.view(-1, 1, 1).detach()

    def forward(self, img):
        return (img - self.mean) / self.std

normalization = Normalization(cnn_normalization_mean, cnn_normalization_std)
model = nn.Sequential(normalization)
conv_num = 0
layer_num = 0
convs = []

model.requires_grad_(False)
input_image.requires_grad_(True)

for layer in cnn.children():
    layer_num +=1
    if(isinstance(layer, nn.ReLU)):
        model.add_module("layer_" + str(layer_num), nn.ReLU(inplace=False))
    else:
        model.add_module("layer_" + str(layer_num), layer)

    if(isinstance(layer,nn.Conv2d)):
        conv_num += 1
        if(conv_num == 1 or conv_num == 2 or conv_num == 3 or conv_num == 4):
            print("_ Content layer added at " + str(conv_num) + "# conv layer")
            _target = model(target_image).detach()
            content_layer = ContentLayer(_target)
            model.add_module("content_" + str(conv_num), content_layer)
            convs.append(content_layer)

ShowImage(model(target_image)[0,[5,6,7],:,:])
input()

optimizer = torch.optim.Adam([input_image], lr=0.1)
while(True):
    plt.close('all')
    model(input_image)
    print("Loss: " + str(content_layer.loss.item()))

#    content_layer.loss.backward()
#    optimizer.step()

    loss = 0
    for conv in convs:
        loss += conv.loss

    loss.backward()
    optimizer.step()

    plt.pause(0.3)
    with torch.no_grad():
        optimizer.zero_grad()
        input_image.clamp_(0, 1)
    ShowImage(input_image)
