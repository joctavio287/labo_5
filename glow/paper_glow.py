import os, pickle, matplotlib.pyplot as plt, numpy as np
from scipy.signal import savgol_filter
from labos.ajuste import Ajuste
from herramientas.config.config_builder import Parser, save_dict, load_dict, guardar_csv

# Importo los paths
glob_path = os.path.normpath(os.getcwd())
variables = Parser(configuration = 'glow').config()
input_path = os.path.join(glob_path + os.path.normpath(variables['input']))
output_path = os.path.join(glob_path + os.path.normpath(variables['output']))

# ==================
# Figura 2 del paper
# ==================
lista = []
texto = ''
for j in [2,6,10]:
    lista.append(j)
    fig, ax = plt.subplots(nrows = 1, ncols = 1)
    for i in lista:
        datos_ida = np.loadtxt(fname = os.path.join(input_path + os.path.normpath(f'/figuras_paper/figura2-{i}k-forward.txt')), delimiter = ';')
        datos_vuelta = np.loadtxt(fname = os.path.join(input_path + os.path.normpath(f'/figuras_paper/figura2-{i}k-reverse.txt')), delimiter = ';')
        ax.scatter(datos_ida[:, 0], datos_ida[:, 1], s = 5, label = 'Ida')
        ax.scatter(datos_vuelta[:, 0], datos_vuelta[:, 1], s = 5, label = 'Vuelta')
    ax.grid(visible = True)
    ax.set_xlabel('Intensidad de corriente en la descarga [mA]')
    ax.set_ylabel('Tensión entre los electrodos [V]')
    texto += f' R = {j} '+ r'K$\Omega$'
    fig.suptitle(t = texto)
    fig.legend(loc = 'right')
    fig.show()


# ==================
# Figura 3 del paper
# ==================

lista = []
texto = ''
for j in [2,6,10]:
    lista.append(j)
    fig, ax = plt.subplots(nrows = 1, ncols = 1)
    for i in lista:
        datos_ida = np.loadtxt(fname = os.path.join(input_path + os.path.normpath(f'/figuras_paper/figura3-{i}k-forward.txt')), delimiter = ';')
        datos_vuelta = np.loadtxt(fname = os.path.join(input_path + os.path.normpath(f'/figuras_paper/figura3-{i}k-reverse.txt')), delimiter = ';')
        ax.scatter(datos_ida[:, 0], datos_ida[:, 1], s = 5)
        ax.plot(datos_ida[:, 0], datos_ida[:, 1], label = 'Ida')
        ax.scatter(datos_vuelta[:, 0], datos_vuelta[:, 1], s = 5)
        ax.plot(datos_vuelta[:, 0], datos_vuelta[:, 1], label = 'Vuelta')
    ax.grid(visible = True)
    ax.set_xlabel('Tensión entre los electrodos [V]')
    ax.set_ylabel(r'Resistencia del gas [K$\Omega$]')
    texto += f' R = {j} '+ r'K$\Omega$'
    fig.suptitle(t = texto)
    fig.legend(loc = 'right')
    fig.show()

    
# ==================
# Figura 4 del paper
# ==================
lista = []
texto = ''
for j in [0.2,0.6,1.0]:
    lista.append(j)
    fig, ax = plt.subplots(nrows = 1, ncols = 1)
    for i in lista:
        datos_ida = np.loadtxt(fname = os.path.join(input_path + os.path.normpath(f'/figuras_paper/figura4-{i}-forward.txt')), delimiter = ';')
        datos_vuelta = np.loadtxt(fname = os.path.join(input_path + os.path.normpath(f'/figuras_paper/figura4-{i}-reverse.txt')), delimiter = ';')
        ax.scatter(datos_ida[:, 0], datos_ida[:, 1], s = 5, label = 'Ida')
        # ax.plot(datos_ida[:, 0], datos_ida[:, 1], label = 'Ida')
        ax.scatter(datos_vuelta[:, 0], datos_vuelta[:, 1], s = 5, label = 'Vuelta')
        # ax.plot(datos_vuelta[:, 0], datos_vuelta[:, 1], label = 'Vuelta')
    ax.grid(visible = True)
    ax.set_xlabel('Tensión entre los electrodos [V]')
    ax.set_ylabel(r'Resistencia del gas [K$\Omega$]')
    texto += f' P = {j} mbar'
    fig.suptitle(t = texto)
    fig.legend(loc = 'right')
    fig.show()

# ==================
# Figura 5 del paper
# ==================
lista = []
texto = ''
for j in [0.2,0.6,1.0]:
    lista.append(j)
    fig, ax = plt.subplots(nrows = 1, ncols = 1)
    for i in lista:
        datos_ida = np.loadtxt(fname = os.path.join(input_path + os.path.normpath(f'/figuras_paper/figura5-{i}-forward.txt')), delimiter = ';')
        datos_vuelta = np.loadtxt(fname = os.path.join(input_path + os.path.normpath(f'/figuras_paper/figura5-{i}-reverse.txt')), delimiter = ';')
        ax.scatter(datos_ida[:, 0], datos_ida[:, 1], s = 5)#, label = 'Ida')
        ax.plot(datos_ida[:, 0], datos_ida[:, 1], label = 'Ida')
        ax.scatter(datos_vuelta[:, 0], datos_vuelta[:, 1], s = 5)#, label = 'Vuelta')
        ax.plot(datos_vuelta[:, 0], datos_vuelta[:, 1], label = 'Vuelta')
    ax.grid(visible = True)
    ax.set_xlabel('Tensión entre los electrodos [V]')
    ax.set_ylabel(r'Resistencia del gas [K$\Omega$]')
    texto += f' P = {j} mbar'
    fig.suptitle(t = texto)
    fig.legend(loc = 'right')
    fig.show()    