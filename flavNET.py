from utils import *

# _______________________________________________.
# ___________________INPUT_______________________|
# _______________________________________________|
style_file_name = "style_1.jpg"
content_file_name = "dune.jpg"
style_weight_pow = 10  # 10^(style_weight_pow)
epoch = 50
# To change image size, find imsize in utils.py


# _______________________________________________.
# _____________GET AND PLOT INPUT________________|
# _______________________________________________|
style_img = ImageLoader("./images/{}".format(style_file_name))
content_img = ImageLoader("./images/{}".format(content_file_name))

plt.ion()

plt.figure()
ShowImage(style_img, "STYLE")
plt.figure()
ShowImage(content_img, "CONTENT")


# _______________________________________________.
# ____________LOAD VGG PRETRAINED________________|
# _______________________________________________|
# eval() is used because some layers have different behaviour in training,
# so we specify we want vgg in evaluation mode.
cnn = models.vgg19(pretrained=True).features.to(device).eval()
# VGG are trained with images normalized with these parameters.
# So we save them to normalize our input images, as VGG expects
cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)


# _______________________________________________.

