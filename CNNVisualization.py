import torch

from utils import *

# ____________________________________.
# __________ INPUT ___________________|
# <editor-fold desc="INPUT">
IMG_SIZE = 512
EPOCH = 300
LEARNING_RATE = 0.02
WEIGHT_DECAY = 0.000005
REGULARIZATION_WEIGHT = 0.000001

target_file_name = "./images/cane.jpg"
representation_level = 0   # 0 means last layer (net output)
add_normalization_on_first_layer = False
apply_final_blur = False
# ____
pretrained_net = models.vgg19(pretrained=True).features.to(device).eval()
loader = GetLoader(IMG_SIZE)

target_image = ImageLoader(target_file_name, loader).to(device)
input_image = torch.rand([1, 3, IMG_SIZE, IMG_SIZE], requires_grad=True, device=device)
#input_image = ImageLoader("./images/me.jpg",loader)
input_image = input_image.to(device)
input_image.requires_grad_(True)
# </editor-fold>
# ____________________________________.


# ____________________________________.
# __________ MY LAYERS _______________|
#<editor-fold desc="MY LAYERS">
class TargetRepresentationLevel(nn.Module):
    def __init__(self, target):
        super(TargetRepresentationLevel, self).__init__()
        self.targetRep = target.detach()
        self.currentRep = torch.Tensor()

    def forward(self, image):
        self.currentRep = image
        return image


class NormalizationLevel(nn.Module):
    def __init__(self):
        super(NormalizationLevel,self).__init__()
        self.mean = torch.tensor([0.485, 0.456, 0.406]).to(device).view(-1, 1, 1).detach()
        self.std = torch.tensor([0.229, 0.224, 0.225]).to(device).view(-1, 1, 1).detach()

    def forward(self, image):
        return (image - self.mean)/self.std
#</editor-fold>
# ____________________________________.


# ____________________________________.
# _________ INIT MODEL _______________|
# <editor-fold desc="INIT MODEL">
model = nn.Sequential().to(device)
target_representation_level = None

if representation_level > len(list(pretrained_net.children())):
    logging.error("Pretrained net has only " + str(len(list(pretrained_net.children()))) + " layers!\nSelect another representation level")
    exit(-1)

if add_normalization_on_first_layer:
    normalization_level = NormalizationLevel()
    model.add_module("norm", normalization_level)

for i, layer in enumerate(pretrained_net.children()):
    if isinstance(layer, nn.ReLU):
        model.add_module("layer_" + str(i), nn.ReLU(inplace=False))
    else:
        model.add_module("layer_" + str(i), layer)

    if representation_level - 1 == i:
        print("____ TARGET LAYER AFTER LAYER " + str(i) + ": " + str(list(model.modules())[i]))
        _target = model(target_image).detach()
        target_representation_level = TargetRepresentationLevel(_target)
        model.add_module("layer_" + str(i+1), target_representation_level)
        break

model.requires_grad_(False)
# </editor-fold>
# ____________________________________.


# ____________________________________.
# ____________ OPTIMIZER _____________.
# <editor-fold desc="OPTIMIZER">
optimizer = optim.Adam([input_image], lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
#optimizer = optim.SGD([input_image], lr=LEARNING_RATE, momentum=0.9)

# </editor-fold>
# ____________________________________.


# ____________________________________.
# ___________VISUALIZATION ___________.
# <editor-fold desc="VISUALIZATION">
plt.ion()
start_time = time.time()
current_time = start_time
if representation_level != 0:
    target_rep = target_representation_level.targetRep
else:
    print("____ TARGET LAYER AT THE OUTPUT ____")
    target_rep = model(target_image).detach()

for i in range(0, EPOCH):
    loss = 0
    plt.close('all')

    current_output = model(input_image)

    if representation_level == 0:
        current_rep = current_output
    else:
        current_rep = target_representation_level.currentRep

    """    regularise = torch.linalg.norm(
        input_image.view(-1, 1, 1) - input_image.detach().view(-1, 1, 1).mean(), dim=1, ord=6
    ).sum()"""
    regularise = ((input_image.view(-1))**6).sum() * REGULARIZATION_WEIGHT

    loss = F.mse_loss(current_rep, target_rep) + regularise

    if i % 10 == 0:
        print("Loss " + str(i) + ": " + str(loss) + "\n -----> " + str(time.time() - current_time))

    loss.backward()
    optimizer.step()

    current_time = time.time()
    with torch.no_grad():
        optimizer.zero_grad()
        input_image.clamp_(0, 1)
# </editor-fold>
# ____________________________________.
print("Execution time: %.2f", time.time() - start_time)

if apply_final_blur:
    input_image = input_image.squeeze(0)
    input_image = transforms.GaussianBlur(kernel_size=(5,5), sigma=3)(input_image)

ShowImage(input_image,save=True,file_name= "gatto3R.jpg")
input()
