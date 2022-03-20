import PySimpleGUI as sg
from colours_scheme import *
import copy

def GetImageSizeLayer(prefix):
    key = '--{:s}_image_size--'.format(prefix)
    keys = []
    for i in range(0,6):
        keys.append('--{:s}_image_size_{:d}--'.format(prefix, 2**(i+5)))

    l_image_size = []
    l_image_size.append([sg.Text("Input Image Size")])
    l_image_size.append([sg.Radio('32px', 'img_size', size=(12, 1), key=keys[0]), sg.Radio('64px', 'img_size', size=(12, 1), key=keys[1],enable_events=True)])
    l_image_size.append([sg.Radio('128px', 'img_size', size=(12, 1), key=keys[2]), sg.Radio('256px', 'img_size', size=(12, 1), key=keys[3])])
    l_image_size.append([sg.Radio('512px', 'img_size', size=(12, 1), key=keys[4]), sg.Radio('1024px', 'img_size', size=(12, 1), key=keys[5])])
    return copy.deepcopy((l_image_size))

def GetEpochLayer(prefix):
    key = '--{:s}_epoch--'.format(prefix)
    l_epoch = []
    l_epoch.append([sg.HorizontalSeparator()])
    l_epoch.append([sg.Text("Epoch")])
    l_epoch.append([sg.Slider((0,800),resolution=50, tick_interval=200, expand_x=True, key=key, orientation='h')])
    return copy.deepcopy(l_epoch)

def GetLearningRateLayer(prefix):
    key = '--{:s}_learning_rate--'.format(prefix)
    l_lr = []
    l_lr.append([sg.HorizontalSeparator()])
    l_lr.append([sg.Text("Learning Rate")])
    l_lr.append([sg.Slider((0.00,15),resolution=0.005, tick_interval=5, expand_x=True, key=key, orientation='h')])
    return copy.deepcopy(l_lr)

def GetOptimizerLayer(prefix):
    key1 = '--{:s}_ADAM_optimizer--'.format(prefix)
    key2 = '--{:s}_SGD_optimizer--'.format(prefix)
    key3 = '--{:s}_weight_decay--'.format(prefix)
    key4 = '--{:s}_momentum--'.format(prefix)
    l_optim = []
    l_optim.append([sg.HorizontalSeparator()])
    l_optim.append([sg.Text("Optimizer")])
    l_optim.append([sg.Radio('ADAM', 'optimizer',size=(12, 1), key=key1), sg.Radio('SGD', 'optimizer',size=(12, 12), key=key2)])
    l_optim.append([sg.Text("Weight Decay"), sg.Text("Momentum (SGD ONLY)")])
    l_optim.append([sg.Slider((0.00,0.1),resolution=0.00005, expand_x=True, key=key3, orientation='v'),sg.Slider((0.00,5),resolution=0.05, expand_x=True, key=key4, orientation='v')])
    return copy.deepcopy(l_optim)

def GetNetLayer(prefix):
    key1 = '--{:s}_VGG16_optimizer--'.format(prefix)
    key2 = '--{:s}_VGG19_optimizer--'.format(prefix)
    key3 = '--{:s}_layer--'.format(prefix)
    key4 = '--{:s}_filter--'.format(prefix)

    l_net = []
    l_net.append([sg.HorizontalSeparator()])
    l_net.append([sg.Text("Pretrained model")])
    l_net.append([sg.Radio('VGG16', 'model',size=(12, 1), key=key1), sg.Radio('VGG19', 'model',size=(12, 12), key=key2)])
    l_net.append([sg.Text("Layer (0 = output)")])
    l_net.append([sg.Slider((0.00,2000),resolution=1,tick_interval= 500, expand_x=True, key=key3, orientation='h')])
    l_net.append([sg.Text("Filter (0 = all)")])
    l_net.append([sg.Slider((0,2000),resolution=1, tick_interval= 500, expand_x=True, key=key4, orientation='h')])
    return copy.deepcopy(l_net)

def AppendAll(what,where):
    for widget,w in enumerate(what):
        where.append(what[widget])

    return copy.deepcopy(where)