from utils import *
'''
GUIDA:
    -   Per effettuare rappresentazione inversa, selezionare la modalità FunctionalMode.InverseRepresentation.
        Selezionare l'immagine target, i livelli di output e i filtri considerati (tendenzialmente TUTTI, "None")
        Impostare come immagine in ingresso del rumore e vedere se la rete riesce a ricostruire l'immagine di partenza)
    -   Per effettuare visualizzazione dei filtri e una semplice deepdream (senza pyramid) selezionare
        FunctionalMode.FilterVisualization.     
'''
class InverseRepresentation():
    def __init__(self, tg_img, in_img,
                 img_size, epoch, lr,
                 optimizer, weight_decay, momentum,
                 net,layer,filter,regularization,modality):
        self.target_file_name = tg_img
        self.input_image = in_img
        self.img_size = img_size
        self.epoch = epoch
        self.lr = lr
        self.optim = optimizer
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.net = net
        self.layer = layer
        self.filter = filter
        self.regularization = regularization
        self.modality = modality

    def Fire(self):
        pass

# ____________________________________.
# __________ INPUT ___________________|
# <editor-fold desc="INPUT">
"""IMG_SIZE = 512
EPOCH = 500
LEARNING_RATE = 5
''' ADAM optimizer parameters'''
WEIGHT_DECAY = 0.000005
''' SGD optimizer parameters'''
MOMENTUM = 0.9

REGULARIZATION_WEIGHT = 0.0001

modality = FunctionalMode.FilterVisualization
optimizer_selector = OptimizerSelector.SGD

target_file_name = "./images/wolf.jpg"
output_file_name = "step.jpg"
representation_level = 0   # 0 means last layer (net output)
filter_selection = 1  # specify a filter or "None" to use all the filters
add_normalization_on_first_layer = True
apply_final_blur = False
# ____
pretrained_net = models.vgg19(pretrained=True).features.to(device).eval()
loader = GetLoader(IMG_SIZE)

target_image = LoadTensorFromFile(target_file_name, loader).to(device)
input_image = torch.rand([1, 3, IMG_SIZE, IMG_SIZE], requires_grad=True, device=device)
#input_image = LoadTensorFromFile("./images/wolf.jpg",loader)
input_image = input_image.to(device)
input_image.requires_grad_(True)
# </editor-fold>
# ____________________________________.


# ____________________________________.
# __________ MY LAYERS _______________|
#<editor-fold desc="MY LAYERS">
class TargetRepresentationLevel(nn.Module):
    def __init__(self, target, filter_selected = None):
        super(TargetRepresentationLevel, self).__init__()
        if filter_selected is not None:
            if filter_selected > target.size()[1]:
                logging.error("Target representation has only {:d} filter, "
                              "but you selected filter number {:d}".format(target.size()[1],filter_selected))
                exit(-2)
            self.targetRep = target[0, filter_selected, :, :].detach()
        else:
            self.targetRep = target.detach()

        self.currentRep = torch.Tensor()
        self.filterSelected = filter_selected

    def forward(self, image):
        if self.filterSelected is not None:
            self.currentRep = image[0, self.filterSelected, :, :]
        else:
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
        target_representation_level = TargetRepresentationLevel(_target, filter_selected=filter_selection)
        model.add_module("layer_" + str(i+1), target_representation_level)
        break

model.requires_grad_(False)
# </editor-fold>
# ____________________________________.


# ____________________________________.
# ____________ OPTIMIZER _____________.
# <editor-fold desc="OPTIMIZER">
if optimizer_selector is OptimizerSelector.ADAM:
    optimizer = optim.Adam([input_image], lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
elif optimizer_selector is OptimizerSelector.SGD:
    optimizer = optim.SGD([input_image], lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
print("____ OPTIMIZER SELECTED {:s}".format(optimizer_selector))
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
    plt.close('all')

    current_output = model(input_image)

    if representation_level == 0:
        current_rep = current_output
    else:
        current_rep = target_representation_level.currentRep

    if modality is FunctionalMode.InverseRepresentation:
        regularise = ((input_image.view(-1))**6).sum() * REGULARIZATION_WEIGHT
        loss = F.mse_loss(current_rep, target_rep) + regularise

    elif modality is FunctionalMode.FilterVisualization:
        loss = -torch.mean(current_rep)

    if i % 50 == 0:
        print("Loss " + str(i) + ": " + "{:.4f}".format(loss.item()) + "\t-----> {:.2f} s".format(time.time() - current_time))
        current_time = time.time()
        _save_tensor = input_image.detach()
        if apply_final_blur:
            _save_tensor = BlurTensor(input_image,size = 1, sigma=3)
        SaveImage(_save_tensor, "./generated/{:d}_{:s}".format(i, output_file_name))

    loss.backward()
    optimizer.step()

    with torch.no_grad():
        optimizer.zero_grad()
        input_image.clamp_(0, 1)
# </editor-fold>
# ____________________________________.
print("Execution time: {:.2f} s".format(time.time() - start_time))

#ShowImage(input_image,save=True, file_name="limone.jpg")
input()"""
