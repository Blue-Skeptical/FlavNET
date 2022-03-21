from colours_scheme import *
from GUI_utils import *
import layout_InverseRepresentation as IR
import layout_DeepDream as DD
import PySimpleGUI as sg

sg.theme(FN_MAIN_THEME)
t1 = IR.layout  #Inverse Representation

t2 = DD.layout  #Deep Dream

layout = [
    [sg.TabGroup(
            [[
                sg.Tab("Inverse Representation", t1, key= "--IR--"),
                sg.Tab('Deep Dream', t2, key= "--DD--")
            ]])
    ],
    [sg.Button(button_text="FIRE", expand_y=True, expand_x=True),sg.Output(size=(60,5), key = '--OUTPUT--', expand_x=True)]
]

window = sg.Window('FlavNET', layout, default_element_size=(12,1))

while True:
    event, values = window.read()
    print(event,values)
    if event == sg.WIN_CLOSED:           # always,  always give a way out!
        break