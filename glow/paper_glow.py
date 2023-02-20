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

datos_ida = np.loadtxt(fname = os.path.join(input_path + os.path.normpath('/figuras_paper/2k-forward.txt')), delimiter = ';')
datos_vuelta = np.loadtxt(fname = os.path.join(input_path + os.path.normpath('/figuras_paper/2k-reverse.txt')), delimiter = ';')
datos = np.concatenate((datos_ida, np.flip(m = datos_vuelta, axis = 0)), axis = 0)

fig, ax = plt.subplots(nrows = 1, ncols = 1)
ax.scatter(datos_ida[:, 0], datos_ida[:, 1], s = 2, color = 'red', label = 'Subiendo')
# ax.plot(datos_ida[:, 0], datos_ida[:, 1], color = 'red', label = 'Ida')
ax.scatter(datos_vuelta[:, 0], datos_vuelta[:, 1], s = 2, color = 'green', label = 'Bajando')
# ax.plot(datos_vuelta[:, 0], datos_vuelta[:, 1], color = 'green', label = 'Vuelta')
# ax.scatter(datos[:, 0], datos[:, 1], s = 2)
# ini_flechas_x, ini_flechas_y = datos[:, 0], datos[:, 1]
# fin_flechas_x, fin_flechas_y = np.roll(datos[:, 0], -1), np.roll(datos[:, 1], -1)
# for X_f, Y_f, X_i, Y_i in zip(fin_flechas_x, fin_flechas_y, ini_flechas_x, ini_flechas_y):
#     ax.annotate(text = "",
#     xy = (X_f,Y_f), 
#     xytext = (X_i,Y_i),
#     arrowprops = dict(arrowstyle = "->"),#, color = c),
#     size = 7
#     )
ax.grid(visible = True)
ax.set_xlabel('Intensidad de corriente [mA]')
ax.set_ylabel('Tension [V]')
fig.legend()
fig.show()

# Graficamos todas las mediciones juntas
num = 4
cmap = plt.get_cmap('plasma')
cmap_values = np.linspace(0., 1., num)
colors = cmap(cmap_values)
colors_rgb = ['#{0:02x}{1:02x}{2:02x}'.format(int(255*a), int(255*b), int(255*c)) for a, b, c, _ in colors]

c = colors_rgb[1]
i = 4
fig, ax = plt.subplots(nrows = 1, ncols = 1)
# for c, i in zip(colors_rgb, np.arange(1, num + 1 , 1)): # for i in range(1, num + 1):
fname = os.path.join(input_path + os.path.normpath(f'/medicion_{i}.pkl'))
datos_leidos = load_dict(fname = fname)
pr = datos_leidos['presion']
ax.scatter(datos_leidos['corriente_t'], datos_leidos['tension_glow'], label = f'{pr} mbar', s = 2, color = c)
ini_flechas_x, ini_flechas_y = datos_leidos['corriente_t'], datos_leidos['tension_glow']
fin_flechas_x, fin_flechas_y = np.roll(datos_leidos['corriente_t'], -1), np.roll(datos_leidos['tension_glow'], -1)

# ini_flechas_x, ini_flechas_y = np.roll(datos_leidos['corriente_t'], 1), np.roll(datos_leidos['tension_glow'], 1)
# fin_flechas_x = np.concatenate((datos_leidos['corriente_t'][:-1],np.array([datos_leidos['corriente_t'][-2]])))
# fin_flechas_y = np.concatenate((datos_leidos['tension_glow'][:-1],np.array([datos_leidos['tension_glow'][-2]]))) 
for X_f, Y_f, X_i, Y_i in zip(fin_flechas_x, fin_flechas_y, ini_flechas_x, ini_flechas_y):
    ax.annotate(text = "",
    xy = (X_f,Y_f), 
    xytext = (X_i,Y_i),
    arrowprops = dict(arrowstyle = "->", color = c),
    size = 7
    )
ax.grid(visible = True)
ax.set_xlabel('Corriente [A]')
ax.set_ylabel('Tension [V]')
fig.legend()
fig.show()