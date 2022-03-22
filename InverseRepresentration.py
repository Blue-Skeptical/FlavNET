import base64
import io

import GUI.GUI_utils
from nn_utils import *
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
                 net,layer,filter,regularization,modality, window=None, ou_img=None, console=None, progress_bar=None):
        self.tg_img_name = tg_img       # Target image name
        self.in_img_name = in_img       # Input file name
        self.img_size = img_size        # Image size
        self.epoch = epoch              # n# epoch
        self.lr = lr                    # learning rate
        self.optim = optimizer          # utils.OptimizerSelector
        self.weight_decay = weight_decay    # Weight decay
        self.momentum = momentum            # Momentum
        self.net = net                      # utils.PretrainedNet
        self.layer = layer                  # Index of target layer
        self.filter = filter                # Index of self.layer used
        self.regularization = regularization    # regularization weight
        self.modality = modality                # Inverse representation or Filter visualization
        self.model = nn.Sequential().to(device) # Final model used
        self.pretrained_net = None              # Pretrained net loaded
        self.target_image = None                # target image tensor
        self.input_image = None                 # input image tensor
        self.optimizer = None                   # optimizer used
        self.target_representation_level = None # Target layer
        self.loss = 0
        # GUI fields
        self.window = window                    # GUI window
        self.ou_img = ou_img                    # Output Image to show
        self.console = console
        self.progress_bar = progress_bar

    def InitInput(self):
        add_normalization_on_first_layer = True
        # ____
        if self.net == PretrainedNet.vgg19:
            self.pretrained_net = models.vgg19(pretrained=True).features.to(device).eval()
        elif self.net == PretrainedNet.vgg16:
            self.pretrained_net = models.vgg16(pretrained=True).features.to(device).eval()

        loader = GetLoader(self.img_size)

        if self.tg_img_name == 'RANDOM':
            self.target_image = torch.rand([1, 3, self.img_size, self.img_size], requires_grad=False, device=device)
        else:
            self.target_image = LoadTensorFromFile(self.tg_img_name, loader).to(device)
        if self.in_img_name == 'RANDOM':
            self.input_image = torch.rand([1, 3, self.img_size, self.img_size], requires_grad=True, device=device)
        else:
            self.input_image = LoadTensorFromFile(self.in_img_name,loader)

        self.input_image = self.input_image.to(device)
        self.input_image.requires_grad_(True)

    def InitModel(self):
        self.target_representation_level = None

        if self.layer > len(list(self.pretrained_net.children())):
            logging.error("Pretrained net has only " + str(
                len(list(self.pretrained_net.children()))) + " layers!\nSelect another representation level")
            return False

        normalization_level = NormalizationLevel()
        self.model.add_module("norm", normalization_level)

        for i, layer in enumerate(self.pretrained_net.children()):
            if isinstance(layer, nn.ReLU):
                self.model.add_module("layer_" + str(i), nn.ReLU(inplace=False))
            else:
                self.model.add_module("layer_" + str(i), layer)

            if self.layer - 1 == i:
                print("____ TARGET LAYER AFTER LAYER " + str(i) + ": " + str(list(self.model.modules())[i]))
                _target = self.model(self.target_image).detach()
                if self.filter-1 > _target.size()[1]:
                    logging.error("Target representation has only {:d} filter, "
                                  "but you selected filter number {:d}".format(_target.size()[1], self.filter))
                    return False
                self.target_representation_level = TargetRepresentationLevel(_target, filter_selected=self.filter)
                self.model.add_module("layer_" + str(i + 1), self.target_representation_level)
                break
        self.model.requires_grad_(False)
        return True

    def InitOptimizer(self):
        if self.optim is OptimizerSelector.ADAM:
            self.optimizer = optim.Adam([self.input_image], lr=self.lr, weight_decay=self.weight_decay)
        elif self.optim is OptimizerSelector.SGD:
            self.optimizer = optim.SGD([self.input_image], lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)

    def PrintOnOutputFrame(self,tensor):
        if self.ou_img is not None:
            bio = io.BytesIO()
            img = GetImageFromTensor(tensor).save(bio, format="PNG")
            del img
            self.ou_img.update(data=GUI.GUI_utils.OpenImage(bio.getvalue(), (220, 220)), size =(220,220))  #GUI.GUI_utils.OpenImage(bio.getvalue(), (220, 220))

    def Visualization(self):
        start_time = time.time()
        current_time = start_time

        if self.layer != 0:
            target_rep = self.target_representation_level.targetRep
        else:
            print("____ TARGET LAYER AT THE OUTPUT ____")
            target_rep = self.model(self.target_image).detach()
            if self.filter != 0:
                if target_rep.size()[1] < self.filter-1:
                    print("We have {:d} filters available at this level, not {:d}".format(target_rep.size()[1],self.filter))
                    return
                else:
                    target_rep = target_rep[0,self.filter-1,:,:]

        for i in range(0, self.epoch):
            if self.progress_bar:
                self.progress_bar.update(i*100/self.epoch)

            current_output = self.model(self.input_image)

            if self.layer == 0 and self.filter == 0:
                current_rep = current_output
            elif self.layer == 0 and self.filter != 0:
                current_rep = current_output[0,self.filter,:,:]
            else:
                current_rep = self.target_representation_level.currentRep

            if self.modality is FunctionalMode.InverseRepresentation:
                regularise = ((self.input_image.view(-1)) ** 6).sum() * self.regularization
                self.loss = F.mse_loss(current_rep, target_rep) + regularise
            elif self.modality is FunctionalMode.FilterVisualization:
                self.loss = -torch.mean(current_rep)

            if i % 50 == 0:
                print("Loss " + str(i) + ": " + "{:.4f}".format(self.loss.item()) + "\t-----> {:.2f} s".format(
                    time.time() - current_time))
                current_time = time.time()
                _save_tensor = self.input_image.detach()
                SaveImage(_save_tensor, "./generated/{:d}_{:s}".format(i, "STEP.jpg"))
                self.PrintOnOutputFrame(_save_tensor)
                #ClearConsole(self.console)

            self.loss.backward()
            self.optimizer.step()

            with torch.no_grad():
                self.optimizer.zero_grad()
                self.input_image.clamp_(0, 1)
        # </editor-fold>
        # ____________________________________.
        print("Execution time: {:.2f} s".format(time.time() - start_time))

    def Fire(self):
        self.InitInput()
        if not self.InitModel():
            return -1
        self.InitOptimizer()
        self.Visualization()


class TargetRepresentationLevel(nn.Module):
    def __init__(self, target, filter_selected = None):
        super(TargetRepresentationLevel, self).__init__()
        if filter_selected != 0:
            self.targetRep = target[0, filter_selected-1, :, :].detach()
        else:
            self.targetRep = target.detach()

        self.currentRep = torch.Tensor()
        self.filterSelected = filter_selected

    def forward(self, image):
        if self.filterSelected != 0:
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



#i = InverseRepresentation("./images/dune.jpg","RANDOM",64,200,0.2,OptimizerSelector.ADAM,
#                          0.001,0,PretrainedNet.vgg16,10,0,0.001,FunctionalMode.InverseRepresentation)
#i.Fire()
# ____________________________________.
# __________ INPUT ___________________|
# <editor-fold desc="INPUT">
"""
IMG_SIZE = 64
EPOCH = 500
LEARNING_RATE = 0.02
''' ADAM optimizer parameters'''
WEIGHT_DECAY = 0.000005
''' SGD optimizer parameters'''
MOMENTUM = 0.9

REGULARIZATION_WEIGHT = 0.0001

modality = FunctionalMode.InverseRepresentation
optimizer_selector = OptimizerSelector.SGD

target_file_name = "./images/dancing.jpg"
output_file_name = "step.jpg"
representation_level = 10   # 0 means last layer (net output)
filter_selection = 0  # specify a filter or "None" to use all the filters
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
input()
"""