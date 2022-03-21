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
    [sg.Frame(title='Target image', right_click_menu=GU.GetImageMenu("target"), vertical_alignment='t', layout=[[sg.Image()]], size=(220, 220))],
    [sg.Frame(title='Input image',right_click_menu=GU.GetImageMenu("input"), vertical_alignment='t', layout=[[sg.Image()]], size=(220, 220))],
    [sg.HorizontalSeparator()],
    [sg.Frame(title='Output image', vertical_alignment='t', layout=[[sg.Image()]], size=(220, 220))]
])

layout.append([frame,frame_preview])

def HandleEvent(event,values):
    pass

#sg.Frame(title='Source image',vertical_alignment='t', layout=[[sg.Image(OpenImage('../images/gatto.jpg', resize=(200,200)))]])