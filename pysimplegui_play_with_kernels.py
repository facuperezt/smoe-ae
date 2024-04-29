import PySimpleGUI as sg
# import PySimpleGUIQt as sg
# import PySimpleGUIWeb as sg

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasAgg
import matplotlib.figure
import matplotlib.pyplot as plt
import io

from matplotlib import cm
from mpl_toolkits.mplot3d.axes3d import get_test_data
from matplotlib.ticker import NullFormatter
 


"""
    Demo - Matplotlib Embedded figure in a window TEMPLATE
    
    The reason this program is labelled as a "Template" is that it functions on 3 
    PySimpleGUI ports by only changing the import statement. tk, Qt, Web(Remi) all
    run this same code and produce identical results.
    
    Copyright 2020-2023 PySimpleSoft, Inc. and/or its licensors. All rights reserved.
    
    Redistribution, modification, or any other use of PySimpleGUI or any portion thereof is subject to the terms of the PySimpleGUI License Agreement available at https://eula.pysimplegui.com.
    
    You may not redistribute, modify or otherwise use PySimpleGUI or its contents except pursuant to the PySimpleGUI License Agreement.
"""


def create_axis_grid():
    from mpl_toolkits.axes_grid1.axes_rgb import RGBAxes

    plt.close('all')

    def get_demo_image():
        # prepare image
        delta = 0.5

        extent = (-3, 4, -4, 3)
        x = np.arange(-3.0, 4.001, delta)
        y = np.arange(-4.0, 3.001, delta)
        X, Y = np.meshgrid(x, y)
        Z1 = np.exp(-X ** 2 - Y ** 2)
        Z2 = np.exp(-(X - 1) ** 2 - (Y - 1) ** 2)
        Z = (Z1 - Z2) * 2

        return Z, extent

    def get_rgb():
        Z, extent = get_demo_image()

        Z[Z < 0] = 0.
        Z = Z / Z.max()

        R = Z[:13, :13]
        G = Z[2:, 2:]
        B = Z[:13, 2:]

        return R, G, B

    fig = plt.figure(1)
    ax = RGBAxes(fig, [0.1, 0.1, 0.8, 0.8])

    r, g, b = get_rgb()
    kwargs = dict(origin="lower", interpolation="nearest")
    ax.imshow_rgb(r, g, b, **kwargs)

    ax.RGB.set_xlim(0., 9.5)
    ax.RGB.set_ylim(0.9, 10.6)

    plt.draw()
    return plt.gcf()



def create_figure():
    # ------------------------------- START OF YOUR MATPLOTLIB CODE -------------------------------
    fig = matplotlib.figure.Figure(figsize=(5, 4), dpi=100)
    t = np.arange(0, 3, .01)
    fig.add_subplot(111).plot(t, 2 * np.sin(2 * np.pi * t))

    return fig


def create_subplot_3d():


    fig = plt.figure()

    ax = fig.add_subplot(1, 2, 1, projection='3d')
    X = np.arange(-5, 5, 0.25)
    Y = np.arange(-5, 5, 0.25)
    X, Y = np.meshgrid(X, Y)
    R = np.sqrt(X ** 2 + Y ** 2)
    Z = np.sin(R)
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.jet,
                           linewidth=0, antialiased=False)
    ax.set_zlim3d(-1.01, 1.01)

    fig.colorbar(surf, shrink=0.5, aspect=5)

    ax = fig.add_subplot(1, 2, 2, projection='3d')
    X, Y, Z = get_test_data(0.05)
    ax.plot_wireframe(X, Y, Z, rstride=10, cstride=10)
    return fig




def create_pyplot_scales():

    plt.close('all')
    # Fixing random state for reproducibility
    np.random.seed(19680801)

    # make up some data in the interval ]0, 1[
    y = np.random.normal(loc=0.5, scale=0.4, size=1000)
    y = y[(y > 0) & (y < 1)]
    y.sort()
    x = np.arange(len(y))

    # plot with various axes scales
    plt.figure(1)

    # linear
    plt.subplot(221)
    plt.plot(x, y)
    plt.yscale('linear')
    plt.title('linear')
    plt.grid(True)

    # log
    plt.subplot(222)
    plt.plot(x, y)
    plt.yscale('log')
    plt.title('log')
    plt.grid(True)

    # symmetric log
    plt.subplot(223)
    plt.plot(x, y - y.mean())
    plt.yscale('symlog', linthreshy=0.01)
    plt.title('symlog')
    plt.grid(True)

    # logit
    plt.subplot(224)
    plt.plot(x, y)
    plt.yscale('logit')
    plt.title('logit')
    plt.grid(True)
    # Format the minor tick labels of the y-axis into empty strings with
    # `NullFormatter`, to avoid cumbering the axis with too many labels.
    plt.gca().yaxis.set_minor_formatter(NullFormatter())
    # Adjust the subplot layout, because the logit one may take more space
    # than usual, due to y-tick labels like "1 - 10^{-3}"
    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25,
                        wspace=0.35)
    return plt.gcf()

def initialize_kernel_plotter(n_kernels, block_size):
    from models.components.decoders.smoe_decoder import VanillaSMoE
    import torch
    from utils.visualize_kernels import plot_kernel_centers, plot_kernels_chol 
    decoder = VanillaSMoE(n_kernels, block_size, device="cpu")
    def plot_kernels_through_gui_inner(k1_params, k2_params, k3_params, current_kernel=None):
        plt.figure()
        ax = plt.gca()
        pd01, pd02, l01, l02, pos0, nu0 = k1_params
        th1 = pd01
        pd01 = [l01 * np.cos(th1), l01 * np.sin(th1)]
        th2 = pd02
        pd02 = [l02 * np.cos(th2), l02 * np.sin(th2)]
        try:
            P0 = np.array([pd01, pd02])
            D0 = np.array([[l01, 0], [0, l02]])
            chol0 = np.linalg.cholesky(P0@D0@np.linalg.inv(P0)).flatten()

            pd11, pd12, l11, l12, pos1, nu1 = k2_params
            th1 = pd11
            pd11 = [l11 * np.cos(th1), l11 * np.sin(th1)]
            th2 = pd12
            pd12 = [l12 * np.cos(th2), l12 * np.sin(th2)]
            P1 = np.array([pd11, pd12])
            D1 = np.array([[l11, 0], [0, l12]])
            chol1 = np.linalg.cholesky(P1@D1@np.linalg.inv(P1)).flatten()

            pd21, pd22, l21, l22, pos2, nu2 = k3_params
            th1 = pd21
            pd21 = [l21 * np.cos(th1), l21 * np.sin(th1)]
            th2 = pd22
            pd22 = [l22 * np.cos(th2), l22 * np.sin(th2)]
            P2 = np.array([pd21, pd22])
            D2 = np.array([[l21, 0], [0, l22]])
            chol2 = np.linalg.cholesky(P2@D2@np.linalg.inv(P2)).flatten()
        except np.linalg.LinAlgError:
            print("Singular matrix")
            return plt.gcf()
        mu = torch.tensor([pos0[0], pos1[0], pos2[0], pos0[1], pos1[1], pos2[1], nu0, nu1, nu2, *chol0, *chol1, *chol2])
        
        mu_img = decoder(mu.view(1, -1))
        plt.imshow(mu_img.detach().cpu().numpy().squeeze().T, cmap='gray', vmin=0, vmax=1)
        if current_kernel is not None:
            current_kernel = int(current_kernel.split(" ")[1])
            center = [pos0, pos1, pos2][current_kernel]
            center = [(a * block_size)-0.5 for a in center]
            vec1, vec2 = [[pd01, pd02], [pd11, pd12], [pd21, pd22]][current_kernel]
            plt.arrow(*center, *vec1, color='green')
            plt.arrow(*center, *vec2, color='green')

        plot_kernel_centers(mu, ax, block_size, n_kernels=n_kernels)
        plot_kernels_chol(mu, ax, block_size, n_kernels=n_kernels)

        return plt.gcf()
    return plot_kernels_through_gui_inner

# ----------------------------- The draw figure helpful function -----------------------------

def draw_figure(element, figure):
    """
    Draws the previously created "figure" in the supplied Image Element

    :param element: an Image Element
    :param figure: a Matplotlib figure
    :return: The figure canvas
    """


    plt.close('all')        # erases previously drawn plots
    canv = FigureCanvasAgg(figure)
    buf = io.BytesIO()
    canv.print_figure(buf, format='png')
    if buf is None:
        return None
    buf.seek(0)
    element.update(data=buf.read())
    return canv


# ----------------------------- The GUI Section -----------------------------

def _create_kernel_controls(i):
    layout = [
        [sg.T(f"Kernel {i}")],
        [sg.T("X"), sg.Slider((0, 1), 0.5, 0.01, size=(15,15), orientation='h', key=f'pos{i}_x', enable_events=True), sg.T("Y"), sg.Slider((0, 1), 0.5, 0.01, size=(15,15), orientation='h', key=f'pos{i}_y', enable_events=True)],
        # [sg.T("X 1"), sg.Slider((0, 1), 0.5, 0.01, size=(15,15), orientation='h', key='pos11'), sg.T("Y 1"), sg.Slider((0, 1), 0.5, 0.01, size=(15,15), orientation='h', key='pos12')],
        # [sg.T("X 2"), sg.Slider((0, 1), 0.5, 0.01, size=(15,15), orientation='h', key='pos21'), sg.T("Y 2"), sg.Slider((0, 1), 0.5, 0.01, size=(15,15), orientation='h', key='pos22')],
        [sg.T("PHI 0"), sg.Slider((-np.pi, np.pi), 1, 0.01, size=(15,15), orientation='h', key=f'dir{i}_1', enable_events=True), sg.T("PHI 1"), sg.Slider((-np.pi, np.pi), 0, 0.01, size=(15,15), orientation='h', key=f'dir{i}_2', enable_events=True)],
        # [sg.T("DX 1"), sg.Slider((-1, 1), 0, 0.01, size=(15,15), orientation='h', key=f'dir{i}_2x', enable_events=True), sg.T("DY 1"), sg.Slider((-1, 1), 1, 0.01, size=(15,15), orientation='h', key=f'dir{i}_2y', enable_events=True)],
        # [sg.T("DX 2"), sg.Slider((0, 1), 0.5, 0.01, size=(15,15), orientation='h', key='dir21'), sg.T("DY 2"), sg.Slider((0, 1), 0.5, 0.01, size=(15,15), orientation='h', key='dir22')],
        [sg.T("RHO 0"), sg.Slider((1e-1, 20), 1, 0.01, orientation='h', key=f'len{i}_1', enable_events=True), sg.T("RHO 1"), sg.Slider((1e-1, 20), 1, 0.01, orientation='h', key=f'len{i}_2', enable_events=True)],
        # [sg.T("EV X 1"), sg.Slider((-10, 10), 0.5, 0.01, orientation='h', key='len11'), sg.T("EV Y 1"), sg.Slider((-10, 10), 0.5, 0.01, orientation='h', key='len12')],
        # [sg.T("EV X 2"), sg.Slider((-10, 10), 0.5, 0.01, orientation='h', key='len21'), sg.T("EV Y 2"), sg.Slider((-10, 10), 0.5, 0.01, orientation='h', key='len22')],
        [sg.T("Nu 0"), sg.Slider((-2, 2), 0, 0.01, orientation='h', key=f'nu{i}', enable_events=True)],
        # [sg.T("Nu 1"), sg.Slider((-1, 1), 0, 0.01, orientation='h', key='nu11')],
        # [sg.T("Nu 2"), sg.Slider((-1, 1), 0, 0.01, orientation='h', key='nu21')],
    ]
    return sg.Tab(f"Kernel {i}", layout)


def main():
    tabgroup = sg.TabGroup([[_create_kernel_controls(i) for i in range(3)]], key='-TABS-')

    left_col = [[tabgroup]]

    layout = [ [sg.T('Matplotlib Example', font='Any 20')],
               [sg.Column(left_col), sg.Image(key='-IMAGE-')],
               [sg.B('Draw'), sg.B('Exit')] ]

    window = sg.Window('Matplotlib Template', layout)

    image_element = window['-IMAGE-']       # type: sg.Image

    plotter = initialize_kernel_plotter(3, 8)

    while True:
        event, values = window.read()
        # print(event)
        if event == 'Exit' or event == sg.WIN_CLOSED:
            break
        if event == 'Draw' or (type(event) == str and (any(event.startswith(f"dir{i}") for i in range(3)) or any(event.startswith(f"len{i}") for i in range(3)) or any(event.startswith(f"pos{i}") for i in range(3)) or any(event.startswith(f"nu{i}") for i in range(3)))):
            # get parameters from the sliders
            i = 0
            params = [[values[f"dir{i}_1"], values[f"dir{i}_2"], values[f'len{i}_1'], values[f'len{i}_2'], [values[f'pos{i}_x'], values[f'pos{i}_y']], values[f'nu{i}']] for i in range(3)]
            draw_figure(image_element, plotter(*params, current_kernel=window['-TABS-'].get()))

    window.close()


if __name__ == "__main__":
    main()