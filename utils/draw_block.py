import PySimpleGUI as sg
import torch
import argparse
import sys
import os
sys.path.append(os.path.join(sys.path[0], os.pardir))
from explain_elvira_ae import load_model, plot

parser = argparse.ArgumentParser()
parser.add_argument("block_size", type=int, choices=[8, 16], default=16, help="Size of the blocks. Just an int. Assumes square images.")
args = parser.parse_args()

model = load_model()

def hscroll(event):
    delta = int(event.delta/120)
    bg = int(event.widget["background"][1:], 16)
    if delta > 0:
        bg += int("0x111111", 16)
    else:
        bg -= int("0x111111", 16)
    if bg > int("0xFFFFFF", 16):
        event.widget.configure(background="#FFFFFF")
    elif bg < int("0x000000", 16):
        event.widget.configure(background="#FFFFFF")
    else:
        bg = f"#{hex(bg)[2:]}"
        event.widget.configure(background=bg)
    window.refresh()

def vscroll(event):
    delta = int(event.delta/120)
    if delta > 0:
        event.widget.configure(background="#FFFFFF")
    else:
        event.widget.configure(background="#000000")
    window.refresh()

def create_data(headers, size):
    data = [[f'({row}, {col})' for col in range(headers)]
        for row in range(size)]
    data[0] = [f'Column {col}' for col in range(headers)]
    for row in range(size):
        data[row][0] = f'Row {row}'
    data[0][0] = 'Features'
    return data

def make_torch_tensor(window):
    out = torch.zeros(rows, cols)
    for i in range(rows):
        for j in range(cols):
            widget = window[str((i, j))]
            bg = widget.Widget["background"][1:]
            # r, g, b = bg[:2], bg[2:4], bg[4:]
            # grayscale = torch.tensor([int(r), int(g), int(b)]).mean()
            # Using `cols - j` because here the origin is in the UPPER left corner
            # And I need it in the LOWER left
            out[rows-1-i, j] = int(f"0x{bg[:2]}", 16)
    torch.save(out, "utils/tmp_files/tmp_out.th")

headers = 20
size = 50
data = create_data(headers, size)

gap = 2
cols, rows = args.block_size, args.block_size

sg.theme('Darkgrey')

table = []
for y in range(0, rows):
    line = []
    for x in range(0, cols):
        x_pad = 1
        y_pad = 1
        pad = ((x_pad, x_pad), (y_pad, y_pad))
        bg = '#000000'
        line.append(
            sg.Text("", size=(5, 3), pad=pad, background_color=bg, key=str((y, x)))
         )
    table.append(line)

option2 = {'resolution':1, 'pad':(0, 0), 'disable_number_display':True,
    'enable_events':True}

right_col_layout = [
    [sg.InputText("0", k="kernel_nr", size=(5,1), tooltip="kernel to analyze"),],
    [sg.Button("Store Image"),],
    [sg.Button("Plot")],
    [sg.Text("", k="kernels_expert_weight")]

]

layout = [
    [sg.Column(table, background_color='black', pad=(0, 0), key='Table'),],
    [sg.Column(right_col_layout, pad=(1,1), justification="center")]
]

window = sg.Window('Simulate Table', layout, use_default_focus=False,
    finalize=True)

for y in range(rows):
    for x in range(cols):
        element = window[str((y, x))]
        # element.Widget.configure(takefocus=0)
        element.Widget.bind('<MouseWheel>', vscroll)
        element.Widget.bind('<Shift-MouseWheel>', hscroll)
        # element.ParentRowFrame.bind('<MouseWheel>', vscroll)
        # element.ParentRowFrame.bind('<Shift-MouseWheel>', hscroll)

while True:

    event, values = window.read()
    print(event)
    if event == sg.WINDOW_CLOSED:
        break
    elif event == 'Store Image':
        make_torch_tensor(window)
    elif event == 'Plot':
        kernels_expert_weight = plot(model, int(window["kernel_nr"].get()))
        window["kernels_expert_weight"].update(str([round(a, 3) for a in kernels_expert_weight.tolist()]))
window.close()