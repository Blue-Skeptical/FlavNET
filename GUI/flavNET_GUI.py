from colours_scheme import *
import layout_InverseRepresentation as IR
import layout_DeepDream as DD
import PySimpleGUI as sg

sg.theme('DarkBrown1')
t1 = IR.layout  #Inverse Representation

t2 = DD.layout  #Deep Dream

layout = [
        [sg.TabGroup(
            [[
                sg.Tab("Inverse Representation", t1, key= "--IR--"),
                sg.Tab('Deep Dream', t2, key= "DD")
            ]])
        ]
]

window = sg.Window('FlavNET', layout, default_element_size=(12,1), background_color= FN_MAIN_COLOUR)

while True:
    event, values = window.read()
    print(event,values)
    if event == sg.WIN_CLOSED:           # always,  always give a way out!
        break