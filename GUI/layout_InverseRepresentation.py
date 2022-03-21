import copy

import utils
from colours_scheme import *
import GUI_utils as GU
import PySimpleGUI as sg

sg.theme(FN_MAIN_THEME)

PREFIX = "IR"


layout = []
frame = sg.Frame(title="Parameters", layout=[
    [GU.GetImageSizeLayer(PREFIX,getFrameOnly=True), GU.GetEpochLayer(PREFIX, getFrameOnly=True)],
    [GU.GetLearningRateLayer(PREFIX,getFrameOnly=True)],
    [GU.GetOptimizerLayer(PREFIX,getFrameOnly=True)],
    [GU.GetNetLayer(PREFIX,getFrameOnly=True)]
])

frame_preview = sg.Frame(title="I/O", layout=[
    [sg.Frame(title='Target image', right_click_menu=GU.GetImageMenu("target"), vertical_alignment='t', layout=[[sg.Image(key='--target_image--')]], size=(220, 220))],
    [sg.Frame(title='Input image',right_click_menu=GU.GetImageMenu("input"), vertical_alignment='t', layout=[[sg.Image(key='--input_image--')]], size=(220, 220))],
    [sg.HorizontalSeparator()],
    [sg.Frame(title='Output image', vertical_alignment='t', layout=[[sg.Image()]], size=(220, 220))]
])

layout.append([frame,frame_preview])

class InverseRepresentatorHandler:
    def __init__(self):
        self.target_filename = ""
        self.input_filename = ""
        self.image_size = 0
        self.epoch = 0
        self.learning_rate = 0
        self.optimizer = None
        self.momentum = 0
        self.weight_decay = 0
        self.net = ""
        self.layer = 0
        self.filter = 0

    def HandleEvent(self,event,values,window):
        #LOAD TARGET IMAGE
        if event == "load_target_image":
            self.target_filename = sg.popup_get_file('file to open', no_window=True)

            if self.target_filename.split('.')[1] != 'jpg' and self.target_filename.split('.')[1] != 'jpeg' and self.target_filename.split('.')[1] != 'png':
                print('Select .png .jpeg .jpg')
                return 0
            else:
                window['--target_image--'].update(data=GU.OpenImage(self.target_filename,(220,220)))
        if event == "set_target_random":
            window['--target_image--'].update(data = GU.OpenImage("",(220,220),rand=True))
            self.target_filename = "RANDOM"
        #LOAD INPUT IMAGE
        if event == "load_input_image":
            input_filename = sg.popup_get_file('file to open', no_window=True)

            if input_filename.split('.')[1] != 'jpg' and input_filename.split('.')[1] != 'jpeg' and input_filename.split('.')[1] != 'png':
                print('Select .png .jpeg .jpg')
                return 0
            else:
                window['--input_image--'].update(data=GU.OpenImage(input_filename,(220,220)))
        if event == "set_input_random":
            window['--input_image--'].update(data = GU.OpenImage("",(220,220),rand=True))
            self.input_filename = "RANDOM"
        #LOAD PARAMETERS
        # <editor-fold desc="Image Size">
        if values['--{:s}_image_size_32--'.format(PREFIX)]: self.image_size = 32
        elif values['--{:s}_image_size_64--'.format(PREFIX)]: self.image_size = 64
        elif values['--{:s}_image_size_128--'.format(PREFIX)]: self.image_size = 128
        elif values['--{:s}_image_size_256--'.format(PREFIX)]: self.image_size = 256
        elif values['--{:s}_image_size_512--'.format(PREFIX)]: self.image_size = 512
        elif values['--{:s}_image_size_1024--'.format(PREFIX)]: self.image_size = 1024
        else:
            print('Select an image size')
            return 0
        # </editor-fold>
        # <editor-fold desc="Optimizer">
        if values['--{:s}_ADAM_optimizer--'.format(PREFIX)]: self.optimizer = utils.OptimizerSelector.ADAM
        elif values['--{:s}_SGD_optimizer--'.format(PREFIX)]: self.optimizer = utils.OptimizerSelector.SGD
        else:
            print('Select an optimizer')
            return 0

        # </editor-fold>
        # <editor-fold desc="Net">
        if values['--{:s}_VGG16_optimizer--'.format(PREFIX)]: self.net = utils.PretrainedNet.vgg16
        elif values['--{:s}_VGG16_optimizer--'.format(PREFIX)]: self.net = utils.PretrainedNet.vgg19
        else:
            print("Select a model")
            return 0
        # </editor-fold>
        #FIRE
        if event == '--fire--':
            if self.target_filename == "":
                print('Select a target image')
                return 0
            elif self.input_filename == "":
                print('Select an input image')
                return 0

inverseRepresentatorHandler = InverseRepresentatorHandler()

#sg.Frame(title='Source image',vertical_alignment='t', layout=[[sg.Image(OpenImage('../images/gatto.jpg', resize=(200,200)))]])