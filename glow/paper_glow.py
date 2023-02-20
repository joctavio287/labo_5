import os, pickle, matplotlib.pyplot as plt, numpy as np
from scipy.signal import find_peaks
from scipy.signal import savgol_filter
from labos.ajuste import Ajuste
from herramientas.config.config_builder import Parser, save_dict, load_dict, guardar_csv

# Importo los paths
glob_path = os.path.normpath(os.getcwd())
variables = Parser(configuration = 'glow').config()
input_path = os.path.join(glob_path + os.path.normpath(variables['input']))
output_path = os.path.join(glob_path + os.path.normpath(variables['output']))



datos = np.loadtxt(fname = os.path.join(input_path + os.path.normpath('/figuras_paper/2k-forward.txt')), delimiter = ';')

fig, ax = plt.subplots(nrows = 1, ncols = 1)
ax.scatter(datos[:, 0], datos[:, 1])
ax.grid(visible = True)
fig.show()