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
class InverseRepresentation(Thread):
    def __init__(self):
        super(InverseRepresentation,self).__init__()
        self.stop = False
        self.tg_img_name = None       # Target image name
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
        self.regularization = None    # regularization weight
        self.modality = None                # Inverse representation or Filter visualization
        self.model = None # Final model used
        self.pretrained_net = None              # Pretrained net loaded
        self.target_image = None                # target image tensor
        self.input_image = None                 # input image tensor
        self.optimizer = None                   # optimizer used
        self.target_representation_level = None # Target layer
        self.loss = None
        # GUI fields
        self.window = None                    # GUI window
        self.ou_img = None                    # Output Image to show
        self.console = None
        self.progress_bar = None

    def LoadParam(self, tg_img, in_img,
                 img_size, epoch, lr,
                 optimizer, weight_decay, momentum,
                 net,layer,filter,regularization,modality, window=None, ou_img=None, console=None, progress_bar=None):
        self.stop = False
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
                if _target.size()[1] < self.filter.start:
                    print("Selected layer has only {:d} filters! Not {:d}".format(_target.size()[1], self.filter.start))
                    return False
                if _target.size()[1] < self.filter.stop -1:
                    print("Selected layer has only {:d} filters! Not {:d}".format(_target.size()[1], self.filter.stop))
                    return False
                self.target_representation_level = TargetRepresentationLevel(_target, filter_selected=slice(
                    self.filter.start - 1,
                    self.filter.stop - 1))
                self.model.add_module("final_layer", self.target_representation_level)
                break

        if self.layer == 0:
            _target = self.model(self.target_image).detach()
            if _target.size()[1] < self.filter.start:
                print("Selected layer has only {:d} filters! Not {:d}".format(_target.size()[1], self.filter.start))
                return False
            if _target.size()[1] < self.filter.stop -1:
                print("Selected layer has only {:d} filters! Not {:d}".format(_target.size()[1], self.filter.stop))
                return False
            self.target_representation_level = TargetRepresentationLevel(_target, filter_selected=slice(self.filter.start -1,
                                                                                                        self.filter.stop -1))
            self.model.add_module("final_layer", self.target_representation_level)

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

        target_rep = self.target_representation_level.targetRep

        for i in range(0, self.epoch):
            if self.stop: break
            if self.progress_bar:
                self.progress_bar.update(i*100/self.epoch)

            current_output = self.model(self.input_image)

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
                if self.ou_img is not None:
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
    def __init__(self, target, filter_selected = None):
        super(TargetRepresentationLevel, self).__init__()
        if filter_selected.start != -1:
            self.targetRep = target[0, filter_selected, :, :].detach()
        else:
            self.targetRep = target.detach()

        self.currentRep = torch.Tensor()
        self.filterSelected = filter_selected

    def forward(self, image):
        if self.filterSelected.start != -1:
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



#i = InverseRepresentation()
#i.LoadParam("./images/us.jpg","RANDOM",512,400,0.02,OptimizerSelector.ADAM,
#                          0,0,PretrainedNet.vgg16,15,2,0,FunctionalMode.InverseRepresentation,None,None,None,None)
#i.run()