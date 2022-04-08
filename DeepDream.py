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
            self.input_image = torch.rand([self.img_size, self.img_size,3]).detach().numpy()
            self.input_tensor = GetTensorFromImage(self.input_image, require_grad=True)
        else:
            self.input_image = np.asarray(GetImageReshapedFromFile(self.in_img_name,loader))
            self.input_tensor = GetTensorFromImage(self.input_image, require_grad=True)

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
                if _temp.size()[1] < self.filter.start:
                    print("Selected layer has only {:d} filters! Not {:d}".format(_temp.size()[1], self.filter.start))
                    return False
                if _temp.size()[1] < self.filter.stop - 1:
                    print("Selected layer has only {:d} filters! Not {:d}".format(_temp.size()[1], self.filter.start))
                    return False
                self.target_representation_level = TargetRepresentationLevel(filter_selected= slice(self.filter.start -1,
                                                                                                    self.filter.stop - 1))
                self.model.add_module('layer_final', self.target_representation_level)
                break

        if self.layer == 0:
            _temp = self.model(self.input_tensor)
            if _temp.size()[1] < self.filter.start:
                print("Selected layer has only {:d} filters! Not {:d}".format(_temp.size()[1], self.filter.start))
                return False
            if _temp.size()[1] < self.filter.stop - 1:
                print("Selected layer has only {:d} filters! Not {:d}".format(_temp.size()[1], self.filter.start))
                return False
            self.target_representation_level = TargetRepresentationLevel(filter_selected=slice(self.filter.start - 1,
                                                                                               self.filter.stop - 1))
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
        if self.filterSelected.start != -1:
            self.currentRep = image[0, self.filterSelected, :, :]
        else:
            self.currentRep = image

        return image



#a = DeepDream()
#a.LoadParam("./images/cane.jpg",256,200,3,1.5,0.01,OptimizerSelector.ADAM,0,0,Losses.MEAN,PretrainedNet.vgg16,6,slice(5,10),None,None,None,None)
#a.run()