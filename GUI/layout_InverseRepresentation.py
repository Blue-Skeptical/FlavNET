import copy

from colours_scheme import *
import GUI_utils as GU
import PySimpleGUI as sg

sg.theme(FN_MAIN_THEME)

PREFIX = "IR"

"""layout = []
parameters_layout = []
parameters_frame = []

parameters_layout = GU.AppendUnder(GU.GetImageSizeLayer(PREFIX), parameters_layout)
parameters_layout = GU.AppendRight(GU.GetEpochLayer(PREFIX, getFrameOnly=True), parameters_layout)
parameters_layout = GU.AppendUnder(GU.GetLearningRateLayer(PREFIX), parameters_layout)
parameters_layout = GU.AppendUnder(GU.GetOptimizerLayer(PREFIX), parameters_layout)
parameters_layout = GU.AppendUnder(GU.GetNetLayer(PREFIX), parameters_layout)

parameters_frame = sg.Frame(title="Inverse Representation", layout=parameters_layout)
layout = copy.deepcopy([parameters_frame])"""

layout = []
frame = sg.Frame(title="Parameters", layout=[
    [GU.GetImageSizeLayer(PREFIX,getFrameOnly=True), GU.GetEpochLayer(PREFIX, getFrameOnly=True)],
    [GU.GetLearningRateLayer(PREFIX,getFrameOnly=True)],
    [GU.GetOptimizerLayer(PREFIX,getFrameOnly=True)],
    [GU.GetNetLayer(PREFIX,getFrameOnly=True)]
])
layout.append([frame, sg.Frame(title='Source image',vertical_alignment='t', layout=[[sg.Image(GU.OpenImage('../images/gatto.jpg', resize=(200,200)))]])])

#sg.Frame(title='Source image',vertical_alignment='t', layout=[[sg.Image(OpenImage('../images/gatto.jpg', resize=(200,200)))]])