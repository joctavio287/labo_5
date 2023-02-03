import os, pickle, matplotlib.pyplot as plt, numpy as np
from herramientas.config.config_builder import Parser

# Importo los paths
glob_path = os.path.normpath(os.getcwd())
variables = Parser(configuration = 'espectroscopia').config()
input_path = os.path.join(glob_path + os.path.normpath(variables['input']))
output_path = os.path.join(glob_path + os.path.normpath(variables['output']))

# Grafico tension del fotodiodo (V) vs corriente del controlador (mA). Sin paso por rubidio
with open(file = os.path.join(input_path + os.path.normpath('/medicion_sinrb.pkl')), mode = "rb") as archive:
    datos_sinrb = pickle.load(file = archive)

fig, ax = plt.subplots(nrows = 1, ncols = 1)
ax.scatter(datos_sinrb['corriente_setting'], datos_sinrb['tension'], s = 15)
ax.set_xlabel('Corriente controlador láser [mA]')
ax.set_ylabel('Tensión de fotodiodo [V]')
ax.grid(visible = True)
fig.show()
# fig.savefig(fname = os.path.join(output_path + os.path.normpath('/medicion_sinrb.png')))

# Grafico tension del fotodiodo (V) vs corriente del controlador (mA). Con paso por rubidio
with open(file = os.path.join(input_path + os.path.normpath('/medicion_conrb.pkl')), mode = "rb") as archive:
    datos_conrb = pickle.load(file = archive)

fig, ax = plt.subplots(nrows = 1, ncols = 1)
ax.scatter(datos_conrb['corriente_setting'], datos_conrb['tension'], s = 15)
ax.set_xlabel('Corriente controlador láser [mA]')
ax.set_ylabel('Tensión de fotodiodo [V]')
ax.grid(visible = True)
fig.show()
# fig.savefig(fname =os.path.join(output_path + os.path.normpath('/medicion_conrb.png')))

# Grafico tension del fotodiodo (V) vs temperatura del controlador (ºC). Con paso por rubidio
with open(file = os.path.join(input_path + os.path.normpath('/medicion_1200uA.pkl')), mode = "rb") as archive:
    datos_1200uA = pickle.load(file = archive)

fig, ax = plt.subplots(nrows = 1, ncols = 1)
ax.scatter(datos_1200uA['temperatura'], datos_1200uA['tension'], s = 15)
ax.set_xlabel(r'Corriente controlador láser [$^{o}C$]')
ax.set_ylabel('Tensión de fotodiodo [V]')
ax.grid(visible = True)
fig.show()
# fig.savefig(fname =os.path.join(output_path + os.path.normpath('/medicion_1200uA.png')))

# Captura del canal 1 del osciloscopio a 24.126 ºC y modulado de rampa de 1Hz. Con paso por rubidio
with open(file = os.path.join(input_path + os.path.normpath('/absorcion_24_126C.pkl')), mode = "rb") as archive:
    datos_24_126C = pickle.load(file = archive)

fig, ax = plt.subplots(nrows = 1, ncols = 1)
ax.scatter(datos_24_126C['tiempo'], datos_24_126C['tension'], s = 2)
ax.set_xlabel('Tiempo [s]')
ax.set_ylabel('Tensión de fotodiodo [V]')
ax.grid(visible = True)
fig.show()
# fig.savefig(fname =os.path.join(output_path + os.path.normpath('/absorcion_24_126C.png')))
