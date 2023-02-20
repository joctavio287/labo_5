import os, matplotlib.pyplot as plt, numpy as np
from herramientas.config.config_builder import Parser, save_dict, load_dict

# Importo los paths
glob_path = os.path.normpath(os.getcwd())
variables = Parser(configuration = 'glow').config()
input_path = os.path.join(glob_path + os.path.normpath(variables['input']))
output_path = os.path.join(glob_path + os.path.normpath(variables['output']))


# Ultima medicion efectuada
num = 4

# Graficamos todas las mediciones juntas
cmap = plt.get_cmap('copper')
cmap = plt.get_cmap('seismic')
cmap = plt.get_cmap('hsv')
cmap_values = np.linspace(0., 1., num)
colors = cmap(cmap_values)
colors_rgb = ['#{0:02x}{1:02x}{2:02x}'.format(int(255*a), int(255*b), int(255*c)) for a, b, c, _ in colors]

fig, ax = plt.subplots(nrows = 1, ncols = 1)
for c, i in zip(colors_rgb, np.arange(1, num + 1 , 1)): # for i in range(1, num + 1):
    fname = os.path.join(input_path + os.path.normpath(f'/medicion_{i}.pkl'))
    datos_leidos = load_dict(fname = fname)
    pr = datos_leidos['presion']
    ax.scatter(datos_leidos['corriente_t'], datos_leidos['tension_glow'], label = f'{pr} mbar', s = 2, color = c)
    ini_flechas_x, ini_flechas_y = datos_leidos['corriente_t'], datos_leidos['tension_glow']
    fin_flechas_x, fin_flechas_y = np.roll(datos_leidos['corriente_t'], -1), np.roll(datos_leidos['tension_glow'], -1)
    for X_f, Y_f, X_i, Y_i in zip(fin_flechas_x, fin_flechas_y, ini_flechas_x, ini_flechas_y):
        ax.annotate(text = "",
        xy = (X_f,Y_f), 
        xytext = (X_i,Y_i),
        arrowprops = dict(arrowstyle = "->", color = c),
        size = 10
        )
ax.grid(visible = True)
ax.set_xlabel('Corriente [A]')
ax.set_ylabel('Tension [V]')
fig.legend()
fig.show()
# fig.savefig(fname = os.path.join(output_path + os.path.normpath('/mediciones_VI_dia1.png')))

