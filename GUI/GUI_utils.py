import PIL
import PySimpleGUI as sg
import nn_utils
import torch
from PIL import Image
import copy
import io
import base64


def GetImageSizeLayer(prefix, getFrameOnly = False):
    key = '--{:s}_image_size--'.format(prefix)
    keys = []
    for i in range(0,6):
        keys.append('--{:s}_image_size_{:d}--'.format(prefix, 2**(i+5)))

    frame = sg.Frame(title='Image Size', layout=[
        [sg.Radio('32px', 'img_size', size=(12, 1), key=keys[0]),
         sg.Radio('64px', 'img_size', size=(12, 1), key=keys[1])],
        [sg.Radio('128px', 'img_size', size=(12, 1), key=keys[2]),
         sg.Radio('256px', 'img_size', size=(12, 1), key=keys[3])],
        [sg.Radio('512px', 'img_size', size=(12, 1), key=keys[4]),
         sg.Radio('1024px', 'img_size', size=(12, 1), key=keys[5])],
        ]
      )
    l_image_size = []
    l_image_size.append([frame])

    if getFrameOnly:
        return copy.deepcopy(frame)
    return copy.deepcopy(l_image_size)

def GetEpochLayer(prefix, getFrameOnly = False):
    key = '--{:s}_epoch--'.format(prefix)
    frame = sg.Frame(title='Epoch',layout=[[
        sg.Slider((0,800),resolution=50, tick_interval=200, expand_x=True, key=key, orientation='h')
    ]], expand_x=True, expand_y=False)
    l_epoch = []
    l_epoch.append([frame])
    if getFrameOnly:
        return copy.deepcopy(frame)
    return copy.deepcopy(l_epoch)

def GetPyramidParameters(prefix, getFrameOnly = False):
    key1 = '--{:s}_pyramid_depth--'.format(prefix)
    key2 = '--{:s}_pyramid_multiplier--'.format(prefix)
    frame = sg.Frame(title="Pyramid", layout=[
        [sg.Text("Depth", justification='l'), sg.Text("Multiplier",justification='r',expand_x=True)],
        [sg.Slider((1,8),resolution=1,tick_interval=4, expand_x=True, key=key1, orientation='h'),
         sg.Slider((1,3),resolution=0.2,tick_interval=1, expand_x=True, key=key2, orientation='h')]
    ], expand_x=True)
    l_pyramid = []
    l_pyramid.append([frame])
    if getFrameOnly:
        return copy.deepcopy(frame)
    return copy.deepcopy(l_pyramid)

def GetLossFunction(prefix, getFrameOnly = False):
    key1 = '--{:s}_loss_function_MEAN--'.format(prefix)
    key2 = '--{:s}_loss_function_MSE--'.format(prefix)
    frame = sg.Frame(title="Loss function", layout=[
        [sg.Radio('MEAN', 'loss_fun', size=(12, 1), key=key1),
         sg.Radio('MSE', 'loss_fun', size=(12, 1), key=key2)],
    ],expand_x=True)
    l_loss = []
    l_loss.append([frame])
    if getFrameOnly:
        return copy.deepcopy(frame)
    return copy.deepcopy(l_loss)

def GetLogoAndDescription(getFrameOnly = False, description = ''):
    frame = sg.Frame(title='', border_width=0, layout=[
        [sg.Image('../logo.png'),  sg.VerticalSeparator(), sg.Text(description, justification='c', expand_x=True)]
    ], vertical_alignment='t')
    l_ld =[]
    l_ld.append([frame])
    if getFrameOnly:
        return copy.deepcopy(frame)
    return copy.deepcopy(l_ld)

def GetLearningRateLayer(prefix, getFrameOnly = False):
    key = '--{:s}_learning_rate--'.format(prefix)
    frame = sg.Frame(title="Learning Rate", layout=[[
        sg.Slider((0.00, 15), resolution=0.005, tick_interval=5, expand_x=True, key=key, orientation='h')
    ]], expand_x=True)
    l_lr = []
    l_lr.append([frame])
    if getFrameOnly:
        return frame
    return copy.deepcopy(l_lr)

def GetOptimizerLayer(prefix, getFrameOnly=False):
    key1 = '--{:s}_ADAM_optimizer--'.format(prefix)
    key2 = '--{:s}_SGD_optimizer--'.format(prefix)
    key3 = '--{:s}_weight_decay--'.format(prefix)
    key4 = '--{:s}_momentum--'.format(prefix)

    frame = sg.Frame(title='Optimizer', layout=[
        [sg.Radio('ADAM', 'optimizer', size=(12, 1), key=key1), sg.Radio('SGD', 'optimizer', size=(12, 12), key=key2)],
        [sg.Text("Weight Decay"), sg.Text("Momentum (SGD ONLY)", justification='r', expand_x=True)],
        [sg.Slider((0.00,0.1),resolution=0.00005, expand_x=True, key=key3, orientation='h'),sg.Slider((0.00,5),resolution=0.05, expand_x=True, key=key4, orientation='h')]
    ], expand_x=True)
    l_optim = []
    l_optim.append([frame])
    if getFrameOnly:
        return frame
    return copy.deepcopy(l_optim)

def GetNetLayer(prefix,getFrameOnly=False):
    key1 = '--{:s}_VGG16_optimizer--'.format(prefix)
    key2 = '--{:s}_VGG19_optimizer--'.format(prefix)
    key3 = '--{:s}_layer--'.format(prefix)
    key4 = '--{:s}_filter--'.format(prefix)

    frame = sg.Frame(title="Pretrained Net", layout=[
        [sg.Radio('VGG16', 'model',size=(12, 1), key=key1), sg.Radio('VGG19', 'model',size=(12, 12), key=key2)],
        [sg.Text("Layer (0 = output)", justification='l', expand_x=True),sg.Text("Filter (0 = all)\nRange: [int,int[", justification='r', expand_x=True)],
        [sg.Input(key=key3, justification='l',expand_x=True), sg.Input(key=key4,justification='l',expand_x=True)],
    ], expand_x=True)

    l_net = []
    l_net.append([frame])
    if getFrameOnly:
        return frame
    return copy.deepcopy(l_net)

def GetRegularizationLayer(prefix,getFrameOnly=False):
    key = '--{:s}_regularization--'.format(prefix)
    frame = sg.Frame(title="Regularization", layout=[
        [sg.Text("Regularization weight", justification='c',expand_x=True)],
        [sg.Input(key=key, justification='c',expand_x=True)]
    ], expand_x=True, expand_y=False)
    l_reg = []
    l_reg.append([frame])
    if getFrameOnly:
        return frame
    return copy.deepcopy(l_reg)

def GetImageMenu(prefix):
    menu_def = [['Unused'],['load_{:s}_image'.format(prefix), 'set_{:s}_random'.format(prefix)]]
    return menu_def

def AppendRight(what,where):
    final_where = copy.deepcopy(where)
    final_where[0].append(what)

    return copy.deepcopy(final_where)

def AppendUnder(what,where):
    for widget,w in enumerate(what):
        where.append(what[widget])

    return copy.deepcopy(where)

def OpenImage(file_or_bytes, resize=None, rand = False):
    '''
    Will convert into bytes and optionally resize an image that is a file or a base64 bytes object.
    Turns into  PNG format in the process so that can be displayed by tkinter
    :param file_or_bytes: either a string filename or a bytes base64 image object
    :type file_or_bytes:  (Union[str, bytes])
    :param resize:  optional new size
    :type resize: (Tuple[int, int] or None)
    :return: (bytes) a byte-string object
    :rtype: (bytes)
    '''
    if rand:
        img = nn_utils.GetImageFromTensor(torch.rand([3,resize[0],resize[1]]))
        bio = io.BytesIO()
        img.save(bio, format="PNG")
        del img
        return bio.getvalue()

    if isinstance(file_or_bytes, str):
        img = PIL.Image.open(file_or_bytes)
    else:
        try:
            img = PIL.Image.open(io.BytesIO(base64.b64decode(file_or_bytes)))
        except Exception as e:
            dataBytesIO = io.BytesIO(file_or_bytes)
            img = PIL.Image.open(dataBytesIO)

    cur_width, cur_height = img.size
    if resize:
        new_width, new_height = resize
        scale = min(new_height/cur_height, new_width/cur_width)
        img = img.resize((int(cur_width*scale), int(cur_height*scale)), PIL.Image.ANTIALIAS)
    bio = io.BytesIO()
    img.save(bio, format="PNG")
    del img
    return bio.getvalue()

def PrintOnOutputFrame(tensor, sgImage):
    if sgImage is not None:
        bio = io.BytesIO()
        img = nn_utils.GetImageFromTensor(tensor).save(bio, format="PNG")
        del img
        sgImage.update(data=OpenImage(bio.getvalue(), (220, 220)), size =(220,220))  #GUI.GUI_utils.OpenImage(bio.getvalue(), (220, 220))


def ClearConsole(console):
    console.update("")
    pass
