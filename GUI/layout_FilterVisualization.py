import copy

from nn_utils import *
from colours_scheme import *
import GUI_utils as GU
import PySimpleGUI as sg
from FilterVisualization import FilterVisualization

sg.theme(FN_MAIN_THEME)

PREFIX = "FV"

# Parameter needed for GUI
ou_img = sg.Image()

layout = []
frame = sg.Frame(title="Parameters", layout=[
    [GU.GetImageSizeLayer(PREFIX,getFrameOnly=True), GU.GetEpochLayer(PREFIX, getFrameOnly=True)],
    [GU.GetLearningRateLayer(PREFIX,getFrameOnly=True)],
    [GU.GetOptimizerLayer(PREFIX,getFrameOnly=True)],
    [GU.GetNetLayer(PREFIX,getFrameOnly=True)],
],vertical_alignment='b')

frame_preview = sg.Frame(title="I/O", layout=[
    [sg.Frame(title='Input image',right_click_menu=GU.GetImageMenu("input"), vertical_alignment='t', layout=[[sg.Image(key='--{:s}_input_image--'.format(PREFIX))]], size=(220, 220))],
    [sg.HorizontalSeparator()],
    [sg.Frame(title='Output image', vertical_alignment='t', layout=[[ou_img]], size=(220, 220))]
], vertical_alignment='t')

frame_logo = GetLogoAndDescription(getFrameOnly=True, description='loss: -mean(current_rep)')

frame_logo_parameters = sg.Frame(title='', border_width=0, layout=[
    [frame_logo],
    [frame]
], vertical_alignment='b', expand_y=True)


layout.append([frame_logo_parameters,frame_preview])

class FilterVisualizationHandler:
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
        self.regularization = 0

        self.fv = None

    def HandleEvent(self,event,values,window, console, progress_bar):
        #STOP
        if event == '--stop--':
            self.fv.Stop()
            self.fv.join()
            return
        #LOAD INPUT IMAGE
        if event == "load_input_image":
            self.input_filename = sg.popup_get_file('file to open', no_window=True)
            if self.input_filename == "": return

            if self.input_filename.split('.')[1] != 'jpg' and self.input_filename.split('.')[1] != 'jpeg' and self.input_filename.split('.')[1] != 'png':
                print('Select .png .jpeg .jpg')
                return 0
            else:
                window['--{:s}_input_image--'.format(PREFIX)].update(data=GU.OpenImage(self.input_filename,(220,220)))
        if event == "set_input_random":
            window['--{:s}_input_image--'.format(PREFIX)].update(data = GU.OpenImage("",(220,220),rand=True))
            self.input_filename = "RANDOM"
        #if FIRE
        if event != '--fire--': return
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
        # <editor-fold desc="Epoch">
        self.epoch = int(values['--{:s}_epoch--'.format(PREFIX)])
        # </editor-fold>
        # <editor-fold desc="Optimizer">
        if values['--{:s}_ADAM_optimizer--'.format(PREFIX)]: self.optimizer = OptimizerSelector.ADAM
        elif values['--{:s}_SGD_optimizer--'.format(PREFIX)]: self.optimizer = OptimizerSelector.SGD
        else:
            print('Select an optimizer')
            return 0
        self.momentum = float(values['--{:s}_momentum--'.format(PREFIX)])
        self.weight_decay = float(values['--{:s}_weight_decay--'.format(PREFIX)])
        # </editor-fold>
        # <editor-fold desc="Net">
        if values['--{:s}_VGG16_optimizer--'.format(PREFIX)]: self.net = PretrainedNet.vgg16
        elif values['--{:s}_VGG19_optimizer--'.format(PREFIX)]: self.net = PretrainedNet.vgg19
        else:
            print("Select a model")
            return 0
        if (values['--{:s}_layer--'.format(PREFIX)]).isnumeric():
            self.layer = int((values['--{:s}_layer--'.format(PREFIX)]))
        else:
            print("Layer must be int")
            return 0
        if (values['--{:s}_filter--'.format(PREFIX)]).isnumeric():
            self.filter = slice(int((values['--{:s}_filter--'.format(PREFIX)])),
                                int((values['--{:s}_filter--'.format(PREFIX)])) + 1)
        else:
            try:
                _start,_stop = values['--{:s}_filter--'.format(PREFIX)].split(',')
                if _start.isnumeric() and _stop.isnumeric():
                    self.filter = slice(int(_start),int(_stop))
                else:
                    print("Filters must be int")
                    return 0
            except:
                print("Insert a number(int) or a range(int,int) as filter!")
                return 0
        # </editor-fold>
        # <editor-fold desc="Learning rate">
        self.learning_rate = float((values['--{:s}_learning_rate--'.format(PREFIX)]))
        # </editor-fold>
        #FIRE
        if self.input_filename == "":
            print('Select an input image')
            return 0

        self.fv = FilterVisualization()
        self.fv.LoadParam(self.input_filename,self.image_size,self.epoch,
                                   self.learning_rate, self.optimizer, self.weight_decay, self.momentum,
                                   self.net, self.layer, self.filter, FunctionalMode.FilterVisualization,
                                   window, ou_img, console, progress_bar)

        self.fv.start()

filterVisualizationHandler = FilterVisualizationHandler()

#sg.Frame(title='Source image',vertical_alignment='t', layout=[[sg.Image(OpenImage('../images/gatto.jpg', resize=(200,200)))]])