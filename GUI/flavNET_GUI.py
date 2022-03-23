from colours_scheme import *
from GUI_utils import *
import layout_InverseRepresentation as IR
import layout_DeepDream as DD
import layout_FilterVisualization as FV
import PySimpleGUI as sg

sg.theme(FN_MAIN_THEME)
t1 = IR.layout  #Inverse Representation

t2 = DD.layout  #Deep Dream

t3 = FV.layout  #Filter Visualization

console = sg.Output(size=(95,3), key = '--OUTPUT--', expand_x=True)
progress_bar = sg.ProgressBar(max_value=100,orientation='h',key='--PROGRESSBAR--',expand_x=True,size=(40,8),bar_color=("green",FN_MAIN_COLOUR))

layout = [
    [sg.TabGroup(
            [[
                sg.Tab("Inverse Representation", t1, key= "--IR--"),
                sg.Tab('Deep Dream', t2, key= "--DD--"),
                sg.Tab('Filter Visualization',t3, key="--FV--")
            ]],key="--TAB--")
    ],
    [sg.Frame(title="", border_width=0, layout=[
        [sg.Button(button_text="FIRE", expand_y=True, expand_x=True, key='--fire--')],
        [sg.Button(button_text="STOP", expand_y=True, expand_x=True, key='--stop--', button_color='red')]
    ]), sg.Frame(title="",layout=[
        [progress_bar],
        [console]
    ], expand_x=True)]
]

window = sg.Window('FlavNET', layout, default_element_size=(12,1))

while True:
    event, values = window.read()

    ClearConsole(console)

    if values["--TAB--"] == "--IR--":
        IR.inverseRepresentatorHandler.HandleEvent(event,values,window,console,progress_bar)
    if values["--TAB--"] == "--FV--":
        FV.filterVisualizationHandler.HandleEvent(event,values,window,console,progress_bar)
    if values["--TAB--"] == "--DD--":
        DD.deepDreamHandler.HandleEvent(event,values,window,console,progress_bar)


    if event == sg.WIN_CLOSED:  # always,  always give a way out!
        break