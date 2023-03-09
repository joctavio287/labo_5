import os, matplotlib.pyplot as plt, numpy as np
from herramientas.config.config_builder import Parser, save_dict, load_dict
from labos.ajuste import Ajuste
from labos.propagacion import Propagacion_errores
from sympy import symbols, lambdify, latex, log
from matplotlib.widgets import Slider, Button

# Importo los paths
glob_path = os.path.normpath(os.getcwd())
variables = Parser(configuration = 'glow').config()
input_path = os.path.join(glob_path + os.path.normpath(variables['input']))
output_path = os.path.join(glob_path + os.path.normpath(variables['output']))

# Armo dos arrays con los datos de presión por distancia y otro de tensión de ruptura
rupturas = []
pds = []
for i in range(1, 26):
    fname = os.path.join(input_path + os.path.normpath(f'/medicion_paschen_{i}.pkl'))
    datos_i = load_dict(fname = fname)
    rupturas.append(datos_i['ruptura'])
    pds.append(datos_i['pd']/10) # Torr* cm*0.0750062

# Transformo a arrays
rupturas = np.array(rupturas).reshape(-1,1)-3
pds = np.array(pds).reshape(-1,1)

a_0, a_1, a_2, p_d = symbols("a_0 a_1 a_2 p_d",   real = True)
expr = a_0*(p_d)/(log(a_1*p_d) + a_2) 
# expr = (p_d/.49)*(a_0 + a_1*.49)/(log(a_2*p_d))

pds_auxiliar = np.linspace(pds[0], pds[-1],1000)

with plt.style.context('seaborn-whitegrid'):
    fig, ax = plt.subplots(figsize = (12, 6))
    plt.subplots_adjust(bottom = .25)
    
    # plt.figure()
    ax.scatter(pds, rupturas, label = 'Datos')
    
    # Valores iniciales
    # A_0, A_1, A_2 = 555, .0000482, 11.458
    A_0, A_1, A_2 = 365, 10, 11.458
    # A_0, A_1, A_2  = 699, 0.001, 12
    # A_0, A_1 = 555, .0000482
 
    # CAmbia de sympy a numpy
    lam_x = lambdify([p_d, a_0, a_1, a_2], expr, modules=['numpy'])
    l, = plt.plot(pds_auxiliar, lam_x(pds_auxiliar, A_0, A_1, A_2), linewidth = 2)
    
    ax.margins(x = 0)
    # ax.set_xlim([0, 1.75])
    # ax.set_ylim([300, 600])
    ax.set_xlabel('pd', fontsize = 16)
    ax.set_ylabel('Tensión de ruptura [V]', fontsize = 16) #, rotation = 0)
    
    axcolor = 'lightgoldenrodyellow'
    ax_0 = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor = axcolor) # left, bottom, width, height
    ax_1 = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor = axcolor)
    ax_2 = plt.axes([0.25, 0.25, 0.65, 0.03], facecolor = axcolor)

    # See documentation of Slider to get configurations
    s_0 = Slider(ax_0, label = 'a_0', valmin = 100, valmax = 800, valinit = A_0, valstep = 1)
    s_1 = Slider(ax_1, label = 'a_1', valmin = .0000001, valmax = .0001, valinit = A_1, valstep = .0000001)
    s_2 = Slider(ax_2, label = 'a_2', valmin = 2, valmax = 20, valinit = A_2, valstep = .001)

    def update(val):
        a_0 = s_0.val
        a_1 = s_1.val
        a_2 = s_2.val
        l.set_ydata(lam_x(pds_auxiliar, a_0, a_1, a_2))
        fig.canvas.draw_idle()
    s_0.on_changed(update)
    s_1.on_changed(update)
    s_2.on_changed(update)

    resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
    button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')

    def reset(event):
        s_0.reset()
        s_1.reset()
        s_2.reset()
    button.on_clicked(reset)
    # rax = plt.axes([0.025, 0.5, 0.15, 0.15], facecolor=axcolor)
    # radio = RadioButtons(rax, ('red', 'blue', 'green'), active=0)
    # def colorfunc(label):
    #     l.set_color(label)
    #     fig.canvas.draw_idle()
    # radio.on_clicked(colorfunc)
    # plt.tight_layout()
    plt.show(block = False)
