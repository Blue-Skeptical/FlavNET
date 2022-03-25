import threading

import matplotlib.pyplot as plt
import torch

import nn_utils
from nn_utils import *

class DeepDream(Thread):
    def __init__(self):
        super(DeepDream,self).__init__()
        self.stop = False
        self.in_img_name = None
        self.img_size = None
        self.epoch = None
        self.p_dept = None
        self.p_mul = None
        self.lr = None
        self.optimizer = None
        self.weight_decay = None
        self.momentum = None
        self.loss = None
        self.net = None
        self.layer = None
        self.filter = None
        #GUI
        self.window = None
        self.ou_img = None
        self.console = None
        self.progress_bar = None
        #
        self.input_image = None
        self.input_tensor = None
        self.pretrained_net = None
        self.model = None
        self.optim = None
        self.target_representation_level = None

    def LoadParam(self, in_img, img_size, epoch, p_depth, p_mul, lr, optimizer, weight_decay, momentum, loss,
                 net, layer, filter,
                 window=None, ou_img=None, console=None, progress_bar=None):
        self.stop = False
        self.in_img_name = in_img
        self.img_size = img_size
        self.epoch = epoch
        self.p_dept = p_depth
        self.p_mul = p_mul
        self.lr = lr
        self.optimizer = optimizer
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.loss = loss
        self.net = net
        self.layer = layer
        self.filter = filter
        #GUI
        self.window = window
        self.ou_img = ou_img
        self.console = console
        self.progress_bar = progress_bar
        #
        self.input_image = None
        self.input_tensor = None
        self.pretrained_net = None
        self.model = None
        self.optim = None
        self.target_representation_level = None

        IMAGENET_MEAN_1 = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        IMAGENET_STD_1 = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        self.LOWER_IMAGE_BOUND = torch.tensor((-IMAGENET_MEAN_1 / IMAGENET_STD_1).reshape(1, -1, 1, 1)).to(device)
        self.UPPER_IMAGE_BOUND = torch.tensor(((1 - IMAGENET_MEAN_1) / IMAGENET_STD_1).reshape(1, -1, 1, 1)).to(device)

    def InitInput(self):
        if self.net == PretrainedNet.vgg19:
            self.pretrained_net = models.vgg19(pretrained=True).features.to(device).eval()
        elif self.net == PretrainedNet.vgg16:
            self.pretrained_net = models.vgg16(pretrained=True).features.to(device).eval()
        loader = GetLoader(self.img_size)

        if self.in_img_name == 'RANDOM':
#            self.input_image = torch.rand([1,3,self.img_size,self.img_size], requires_grad=True, device=device)
            self.input_image = torch.rand([self.img_size, self.img_size,3]).detach().numpy()
            self.input_tensor = GetTensorFromImage(self.input_image, require_grad=True)
        else:
            self.input_image = np.asarray(GetImageReshapedFromFile(self.in_img_name,loader))
            self.input_tensor = GetTensorFromImage(self.input_image, require_grad=True)
#            self.input_image = LoadTensorFromFile(self.in_img_name,loader)
#            self.input_image.requires_grad_(True)
#            self.input_image = self.input_image.to(device)

    def InitModel(self):
        self.model = nn.Sequential().to(device)
        target_representation_level = None

        if self.layer > len(list(self.pretrained_net.children())):
            print("Pretrained net has only {:d} layers! Not {:d}".format(self.pretrained_net.children(),self.layer))
            return False

        for i, layer in enumerate(self.pretrained_net.children()):
            if isinstance(layer, nn.ReLU):
                self.model.add_module('layer_' + str(i), nn.ReLU(inplace=False))
            else:
                self.model.add_module('layer_' + str(i), layer)

            if self.layer - 1 == i:
                print("___ TARGET LAYER {:d} AFTER {:s}".format(i,str(list(self.model.modules())[i])))
                _temp = self.model(self.input_tensor)
                if _temp.size()[1] < self.filter:
                    print("Selected layer has only {:d} filters! Not {:d}".format(_temp.size()[1], self.filter))
                    return False
                self.target_representation_level = TargetRepresentationLevel(filter_selected= self.filter -1)
                self.model.add_module('layer_final', self.target_representation_level)
                break

        if self.layer == 0:
            _temp = self.model(self.input_tensor)
            if _temp.size()[1] < self.filter:
                print("Selected layer has only {:d} filters! Not {:d}".format(_temp.size()[1], self.filter))
                return False
            self.target_representation_level = TargetRepresentationLevel(filter_selected= self.filter)
            self.model.add_module('layer_final', self.target_representation_level)

        return True

    def OptimizerStep(self, params):
        if self.optimizer is OptimizerSelector.ADAM:
            self.optim = optim.Adam(params, lr=self.lr, weight_decay=self.weight_decay)
        elif self.optimizer is OptimizerSelector.SGD:
            self.optim = optim.SGD(params, lr=self.lr, weight_decay=self.weight_decay, momentum=self.momentum)
        elif self.optimizer is OptimizerSelector.GD:
            self.optim = nn_utils.GDOptimizer(params,lr=self.lr,direction='negative')

        self.optim.step()


    def Visualization(self):
        start_time = time.time()
        current_time = start_time

        for l in range(0, self.p_dept):
            new_shape = GetCurrentPyramidalShape([self.img_size, self.img_size, 3], l, self.p_dept, self.p_mul)
            self.input_image = cv2.resize(self.input_image, [new_shape[0], new_shape[1]])
            self.input_tensor = GetTensorFromImage(self.input_image, require_grad=True)

            for i in range(0, self.epoch):
                if self.stop: break
                if self.progress_bar is not None:
                    self.progress_bar.update((i + l*self.epoch)*100/(self.epoch*self.p_dept))

                plt.close('all')
                current_output = self.model(self.input_tensor)

                if self.layer == 0:
                    current_rep = current_output
                else:
                    current_rep = self.target_representation_level.currentRep

                if self.loss is Losses.MEAN:
                    loss = -torch.mean(current_rep)
                elif self.loss is Losses.MSE:
                    zeros = torch.zeros_like(current_rep, requires_grad=False)
                    loss = -nn.MSELoss(reduction='mean')(current_rep, zeros)

                if i % 50 == 0:
                    print("Loss " + str(i) + ": " + "{:.4f}".format(loss.item()) + "\t-----> {:.2f} s".format(
                        time.time() - current_time))
                    PrintOnOutputFrame(self.input_tensor.detach(),self.ou_img)
                    current_time = time.time()

                loss.backward()
                self.OptimizerStep([self.input_tensor])

                """
                optimizer = optim.SGD([input_tensor], lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
                optimizer.step()
                # input_tensor.data += LEARNING_RATE * input_tensor.grad.data
                """

                with torch.no_grad():
                    self.optim.zero_grad()
                    self.input_tensor.data = torch.max(torch.min(self.input_tensor, self.UPPER_IMAGE_BOUND), self.LOWER_IMAGE_BOUND)

            SaveImage(self.input_tensor, "./generated/{:d}_{:s}".format(l, "_STEP.jpg"))
            self.input_image = self.input_tensor.detach().cpu().clamp_(0, 1).squeeze(0).numpy()
            self.input_image = np.moveaxis(self.input_image, 0, -1)
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



#a = DeepDream("./images/gatto.jpg",128,100,3,1.5,0.5,OptimizerSelector.GD,0,0,Losses.MEAN,PretrainedNet.vgg16,0,0,None,None,None,None)
#a.Fire()


# ____________________________________.
# __________ INPUT ___________________|
# <editor-fold desc="INPUT">
"""
IMG_SIZE = 128
EPOCH = 100
LEARNING_RATE = 0.005
PYRAMID_DEPTH = 3
''' ADAM optimizer parameters'''
WEIGHT_DECAY = 0.0000005
''' SGD optimizer parameters'''
MOMENTUM = 0.9

REGULARIZATION_WEIGHT = 0.0001

optimizer_selector = OptimizerSelector.SGD

output_file_name = "dd.jpg"
representation_level = 0   # 0 means last layer (net output)
filter_selection = 0  # specify a filter or 0 to use all the filters
# ____
pretrained_net = models.vgg16(pretrained=True).features.to(device).eval()
loader = GetLoader(IMG_SIZE)

#input_image = torch.rand([1, 3, IMG_SIZE, IMG_SIZE], device=device).detach().cpu().squeeze(0).numpy()
#input_image = np.moveaxis(input_image,0,-1)
input_image = np.asarray(GetImageReshapedFromFile("./images/picasso.jpg",loader))
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
        if self.filterSelected != 0:
            self.currentRep = image[0, self.filterSelected-1, :, :]
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
    input_image = input_tensor.detach().cpu().clamp_(0, 1).squeeze(0).numpy()
    input_image = np.moveaxis(input_image, 0, -1)
# </editor-fold>
# ____________________________________.
print("Execution time: {:.2f} s".format(time.time() - start_time))

#ShowImage(input_image,save=True, file_name="limone.jpg")
input()
"""