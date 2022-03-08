from utils import *

# _______________________________________________.
# ___________________INPUT_______________________|
# _______________________________________________|
style_file_name = "style_1.jpg"
content_file_name = "dune.jpg"
style_weight_pow = 10  # 10^(style_weight_pow)
epoch = 50
imsize = 1024 if torch.cuda.is_available() else 512


# _______________________________________________.
# _____________GET AND PLOT INPUT________________|
# _______________________________________________|
loader = GetLoader(imsize)
unloader = transforms.ToPILImage()  # Reconvert into PIL image

style_img = ImageLoader("./images/{}".format(style_file_name), loader)
content_img = ImageLoader("./images/{}".format(content_file_name), loader)

# To initialize with white noise:
# torch.randn(content_img.data.size(), device=device)
input_img = content_img.clone()

plt.ion()

plt.figure()
ShowImage(style_img, "STYLE")
plt.figure()
ShowImage(content_img, "CONTENT")
plt.figure()
ShowImage(input_img, "INPUT")


# _______________________________________________.
# ____________LOAD PRETRAINED VGG________________|
# _______________________________________________|
# eval() is used because some layers have different behaviour in training,
# so we specify we want vgg in evaluation mode.
cnn = models.vgg19(pretrained=True).features.to(device).eval()
# VGG are trained with images normalized with these parameters.
# So we save them to normalize our input images, as VGG expects
cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)


# _______________________________________________.
# __________DEFINING NEW LAYERS TO ADD___________|
# _______________________________________________|
class Normalization(nn.Module):
    """ Used to normalize the input images as
    told before in "LOAD PRETRAINED VGG" section """
    def __init__(self):
        super(Normalization, self).__init__()
        # Reshape mean and std to make them [C,1,1] where C is channel
        # This way, the can work with tensor of shape [B,C,H,W]
        # Is the same as making: .view(1,C,1,1)
        # So you can broadcast and make the forward expressions
        self.mean = torch.tensor(cnn_normalization_mean).view(-1, 1, 1)
        self.std = torch.tesor(cnn_normalization_std).view(-1, 1, 1)

    def forward(self, img):
        return (img - self.mean) / self.std


# _______


class ContentLoss(nn.Module):
    """Transparent layer (outputs the input as it is)
    but save the content loss in its parameter."""
    def __init__(self, target):
        super(ContentLoss,self).__init__()

        self.target = target.detach()

    def forward(self,input):
        self.loss = F.mse_loss(input,self.target)
        return input


# _______


class StyleLoss(nn.Module):
    """Transparent layer (outputs the input as it is)
    but save the style loss in its parameter."""
    def __init__(self, target):
        super.__init__(StyleLoss, self).__init__()
        self.target = GramMatrix(target).detach()

    def forward(self, input):
        self.loss = F.mse_loss(GramMatrix(input), self.target)
        return input


# _______

# _______________________________________________.
# _________ADDING NEW LAYERS TO NETWORK__________|
# _______________________________________________|
model = nn.Sequential()

normalization = Normalization().to(device)
model.add_module(normalization)

conv_count = 0  # Number of conv layers
for layer in cnn.children():
    name = GetLayerName(layer, conv_count)

    if isinstance(layer, nn.Conv2d):
        conv_count += 1
#    elif isinstance(layer, nn.ReLU):
#        layer = nn.ReLU(inplace=False)  # For better ContentLoss and StyleLoss
    model.add_module(name, layer)
