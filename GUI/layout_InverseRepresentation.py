import copy

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

    def HandleEvent(self,event,values,window):
        #LOAD TARGET IMAGE
        if event == "load_target_image":
            self.target_filename = sg.popup_get_file('file to open', no_window=True)

            if self.target_filename.split('.')[1] != 'jpg' and self.target_filename.split('.')[1] != 'jpeg' and self.target_filename.split('.')[1] != 'png':
                print("Select .png .jpeg .jpg")
            else:
                window['--target_image--'].update(data=GU.OpenImage(self.target_filename,(220,220)))
        if event == "set_target_random":
            window['--target_image--'].update(data = GU.OpenImage("",(220,220),rand=True))
        #LOAD INPUT IMAGE
        if event == "load_input_image":
            input_filename = sg.popup_get_file('file to open', no_window=True)

            if input_filename.split('.')[1] != 'jpg' and input_filename.split('.')[1] != 'jpeg' and input_filename.split('.')[1] != 'png':
                print("Select .png .jpeg .jpg")
            else:
                window['--input_image--'].update(data=GU.OpenImage(input_filename,(220,220)))
        if event == "set_input_random":
            window['--input_image--'].update(data = GU.OpenImage("",(220,220),rand=True))
        #FIRE
        if event == '--fire--':
            print(self.target_filename)

inverseRepresentatorHandler = InverseRepresentatorHandler()

#sg.Frame(title='Source image',vertical_alignment='t', layout=[[sg.Image(OpenImage('../images/gatto.jpg', resize=(200,200)))]])