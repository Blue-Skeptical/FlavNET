from colours_scheme import *
import GUI_utils as GU
import PySimpleGUI as sg

sg.theme(FN_MAIN_THEME)

PREFIX = "DD"

# Parameter needed for GUI
ou_img = sg.Image()

layout = []
frame = sg.Frame(title="Parameters", layout=[
    [GU.GetImageSizeLayer(PREFIX,getFrameOnly=True), GU.GetEpochLayer(PREFIX, getFrameOnly=True)],
    [GU.GetLearningRateLayer(PREFIX,getFrameOnly=True)],
    [GU.GetOptimizerLayer(PREFIX,getFrameOnly=True)],
    [GU.GetNetLayer(PREFIX,getFrameOnly=True)],
    [GU.GetPyramidParameters(PREFIX,getFrameOnly=True)],
    [GU.GetLossFunction(PREFIX,getFrameOnly=True)]
],vertical_alignment='b')

frame_preview = sg.Frame(title="I/O", layout=[
    [sg.Frame(title='Input image',right_click_menu=GU.GetImageMenu("input"), vertical_alignment='t', layout=[[sg.Image(key='--{:s}_input_image--'.format(PREFIX))]], size=(220, 220))],
    [sg.HorizontalSeparator()],
    [sg.Frame(title='Output image', vertical_alignment='t', layout=[[ou_img]], size=(220, 220))]
], vertical_alignment='b')

layout.append([frame,frame_preview])





"""layout = []
layout = GU.AppendUnder(GU.GetImageSizeLayer(PREFIX), layout)
layout = GU.AppendRight(GU.GetEpochLayer(PREFIX, getFrameOnly=True), layout)
layout = GU.AppendUnder(GU.GetLearningRateLayer(PREFIX), layout)
layout = GU.AppendUnder(GU.GetOptimizerLayer(PREFIX), layout)
layout = GU.AppendUnder(GU.GetNetLayer(PREFIX), layout)"""