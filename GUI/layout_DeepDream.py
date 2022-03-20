from colours_scheme import *
import GUI_utils as GU
import PySimpleGUI as sg

PREFIX = "DD"

layout = []
layout.append([sg.Text("HYPER PARAMETERS")])
layout = GU.AppendAll(GU.GetImageSizeLayer(PREFIX),layout)
layout = GU.AppendAll(GU.GetEpochLayer(PREFIX),layout)
layout.append([sg.Button('Read')])