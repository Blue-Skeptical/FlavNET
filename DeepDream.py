import torch

from utils import *

# ____________________________________.
# __________ INPUT ___________________|
# <editor-fold desc="INPUT">
IMG_SIZE = 512
EPOCH = 200
LEARNING_RATE = 3
PYRAMID_DEPTH = 1
''' ADAM optimizer parameters'''
WEIGHT_DECAY = 0.0000005
''' SGD optimizer parameters'''
MOMENTUM = 0.9

REGULARIZATION_WEIGHT = 0.0001

optimizer_selector = OptimizerSelector.SGD

output_file_name = "dd.jpg"
representation_level = 0   # 0 means last layer (net output)
filter_selection = 1  # specify a filter or "None" to use all the filters
# ____
pretrained_net = models.vgg16(pretrained=True).features.to(device).eval()
loader = GetLoader(IMG_SIZE)

#input_image = torch.rand([1, 3, IMG_SIZE, IMG_SIZE], device=device).detach().cpu().squeeze(0).numpy()
#input_image = np.moveaxis(input_image,0,-1)
input_image = np.asarray(GetImageReshapedFromFile("./images/style4.jpg",loader))
#input_image = NormalizeImage(input_image)
input_tensor = GetTensorFromImage(input_image, require_grad=True)

IMAGENET_MEAN_1 = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD_1 = np.array([0.229, 0.224, 0.225], dtype=np.float32)
LOWER_IMAGE_BOUND = torch.tensor((-IMAGENET_MEAN_1 / IMAGENET_STD_1).reshape(1, -1, 1, 1)).to(device)
UPPER_IMAGE_BOUND = torch.tensor(((1 - IMAGENET_MEAN_1) / IMAGENET_STD_1).reshape(1, -1, 1, 1)).to(device)
# </editor-fold>
# ____________________________________.


# ____________________________________.
# __________ MY LAYERS _______________|
#<editor-fold desc="MY LAYERS">
class TargetRepresentationLevel(nn.Module):
    def __init__(self, filter_selected = None):
        super(TargetRepresentationLevel, self).__init__()
        self.currentRep = torch.Tensor()
        self.filterSelected = filter_selected

    def forward(self, image):
        if self.filterSelected is not None:
            self.currentRep = image[0, self.filterSelected, :, :]
        else:
            self.currentRep = image

        return image
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

for i, layer in enumerate(pretrained_net.children()):
    if isinstance(layer, nn.ReLU):
        model.add_module("layer_" + str(i), nn.ReLU(inplace=False))
    else:
        model.add_module("layer_" + str(i), layer)
    if representation_level - 1 == i:
        print("____ TARGET LAYER AFTER LAYER " + str(i) + ": " + str(list(model.modules())[i]))
        target_representation_level = TargetRepresentationLevel(filter_selected=filter_selection)
        model.add_module("layer_" + str(i+1), target_representation_level)
        break

if representation_level == 0:
    target_representation_level = TargetRepresentationLevel(filter_selected=filter_selection)
    model.add_module("layer_final",target_representation_level)

model.requires_grad_(False)
# </editor-fold>
# ____________________________________.


# ____________________________________.
# ____________ OPTIMIZER _____________.
# <editor-fold desc="OPTIMIZER">
if optimizer_selector is OptimizerSelector.ADAM:
    optimizer = optim.Adam([input_tensor], lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
elif optimizer_selector is OptimizerSelector.SGD:
    optimizer = optim.SGD([input_tensor], lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
print("____ OPTIMIZER SELECTED {:s}".format(optimizer_selector))
# </editor-fold>
# ____________________________________.


# ____________________________________.
# ___________VISUALIZATION ___________.
# <editor-fold desc="VISUALIZATION">
plt.ion()
start_time = time.time()
current_time = start_time

for l in range(0,PYRAMID_DEPTH):
    new_shape = GetCurrentPyramidalShape([IMG_SIZE, IMG_SIZE, 3], l, PYRAMID_DEPTH,1.4)
    input_image = cv2.resize(input_image, [new_shape[0], new_shape[1]])
    input_tensor = GetTensorFromImage(input_image, require_grad= True)
    ShowImage(input_tensor)

    for i in range(0, EPOCH):
        plt.close('all')

        current_output = model(input_tensor)

        if representation_level == 0:
            current_rep = current_output
        else:
            current_rep = target_representation_level.currentRep

#        loss = -torch.mean(current_rep)
        zeros = torch.zeros_like(current_rep,requires_grad=False)
        loss = -nn.MSELoss(reduction='mean')(current_rep,zeros)

        if i % 50 == 0:
            print("Loss " + str(i) + ": " + "{:.4f}".format(loss.item()) + "\t-----> {:.2f} s".format(time.time() - current_time))
            current_time = time.time()

        loss.backward()
        optimizer = optim.SGD([input_tensor], lr=LEARNING_RATE,momentum=MOMENTUM,  weight_decay=WEIGHT_DECAY)
        optimizer.step()
        #input_tensor.data += LEARNING_RATE * input_tensor.grad.data

        with torch.no_grad():
            optimizer.zero_grad()
            input_tensor.data = torch.max(torch.min(input_tensor, UPPER_IMAGE_BOUND), LOWER_IMAGE_BOUND)

    SaveImage(input_tensor, "./generated/{:d}_{:s}".format(l, output_file_name))
    input_image = input_tensor.detach().cpu().squeeze(0).numpy()
    input_image = np.moveaxis(input_image, 0, -1)
# </editor-fold>
# ____________________________________.
print("Execution time: {:.2f} s".format(time.time() - start_time))

#ShowImage(input_image,save=True, file_name="limone.jpg")
input()
