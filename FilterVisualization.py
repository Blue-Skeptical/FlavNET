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
class FilterVisualization(Thread):
    def __init__(self):
        super(FilterVisualization,self).__init__()
        self.stop = False
        self.in_img_name = None       # Input file name
        self.img_size = None        # Image size
        self.epoch = None              # n# epoch
        self.lr = None                    # learning rate
        self.optim = None          # utils.OptimizerSelector
        self.weight_decay = None    # Weight decay
        self.momentum = None            # Momentum
        self.net = None                      # utils.PretrainedNet
        self.layer = None                  # Index of target layer
        self.filter = None                # Index of self.layer used
        self.modality = None                # Inverse representation or Filter visualization
        self.model = None # Final model used
        self.pretrained_net = None              # Pretrained net loaded
        self.input_image = None                 # input image tensor
        self.optimizer = None                   # optimizer used
        self.target_representation_level = None # Target layer
        self.loss = None
        # GUI fields
        self.window = None                    # GUI window
        self.ou_img = None                    # Output Image to show
        self.console = None
        self.progress_bar = None

    def LoadParam(self, in_img,
                 img_size, epoch, lr,
                 optimizer, weight_decay, momentum,
                 net,layer,filter,modality, window=None, ou_img=None, console=None, progress_bar=None):
        self.stop = False
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
        self.modality = modality                # Inverse representation or Filter visualization
        self.model = nn.Sequential().to(device) # Final model used
        self.pretrained_net = None              # Pretrained net loaded
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
                _target = self.model(self.input_image).detach()
                if self.filter-1 > _target.size()[1]:
                    logging.error("Target representation has only {:d} filter, "
                                  "but you selected filter number {:d}".format(_target.size()[1], self.filter))
                    return False
                self.target_representation_level = TargetRepresentationLevel(filter_selected=self.filter-1)
                self.model.add_module("layer_" + str(i + 1), self.target_representation_level)
                break

        if self.layer == 0:
            _target = self.model(self.input_image).detach()
            if self.filter-1 > _target.size()[1]:
                logging.error("Target representation has only {:d} filter, "
                              "but you selected filter number {:d}".format(_target.size()[1], self.filter))
                return False

            self.target_representation_level = TargetRepresentationLevel(filter_selected=self.filter-1)
            self.model.add_module("layer_" + str(i + 1), self.target_representation_level)

        self.model.requires_grad_(False)
        return True

    def InitOptimizer(self):
        if self.optim is OptimizerSelector.ADAM:
            self.optimizer = optim.Adam([self.input_image], lr=self.lr, weight_decay=self.weight_decay)
        elif self.optim is OptimizerSelector.SGD:
            self.optimizer = optim.SGD([self.input_image], lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)

    def Visualization(self):
        start_time = time.time()
        current_time = start_time

        for i in range(0, self.epoch):
            if self.stop: break
            if self.progress_bar:
                self.progress_bar.update(i*100/self.epoch)
            current_output = self.model(self.input_image)

            if self.layer == 0:
                current_rep = current_output
            else:
                current_rep = self.target_representation_level.currentRep

            self.loss = -torch.mean(current_rep)

            if i % 50 == 0:
                ClearConsole(self.console)
                print("Loss " + str(i) + ": " + "{:.4f}".format(self.loss.item()) + "\t-----> {:.2f} s".format(
                    time.time() - current_time))
                current_time = time.time()
                _save_tensor = self.input_image.detach()
                SaveImage(_save_tensor, "./generated/{:d}_{:s}".format(i, "STEP.jpg"))
                PrintOnOutputFrame(_save_tensor,self.ou_img)

            self.loss.backward()
            self.optimizer.step()

            with torch.no_grad():
                self.optimizer.zero_grad()
                self.input_image.clamp_(0, 1)
        # </editor-fold>
        # ____________________________________.
        print("Execution time: {:.2f} s".format(time.time() - start_time))

    def Stop(self):
        self.stop = True
        if self.progress_bar is not None:
            self.progress_bar.update(0)

    def run(self):
        self.stop = False
        self.InitInput()
        if not self.InitModel():
            return -1
        self.InitOptimizer()
        self.Visualization()


class TargetRepresentationLevel(nn.Module):
    def __init__(self, filter_selected = None):
        super(TargetRepresentationLevel, self).__init__()

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
