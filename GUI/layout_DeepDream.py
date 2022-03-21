from colours_scheme import *
import GUI_utils as GU
import PySimpleGUI as sg

sg.theme(FN_MAIN_THEME)

PREFIX = "DD"

layout = []
layout = GU.AppendUnder(GU.GetImageSizeLayer(PREFIX), layout)
layout = GU.AppendRight(GU.GetEpochLayer(PREFIX, getFrameOnly=True), layout)
layout = GU.AppendUnder(GU.GetLearningRateLayer(PREFIX), layout)
layout = GU.AppendUnder(GU.GetOptimizerLayer(PREFIX), layout)
layout = GU.AppendUnder(GU.GetNetLayer(PREFIX), layout)