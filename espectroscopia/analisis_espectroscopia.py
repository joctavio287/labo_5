import os, pickle, matplotlib.pyplot as plt, numpy as np
from herramientas.config.config_builder import Parser

# Importo los paths
glob_path = os.path.normpath(os.getcwd())
variables = Parser(configuration = 'espectroscopia').config()
input_path = os.path.join(glob_path + os.path.normpath(variables['input']))
output_path = os.path.join(glob_path + os.path.normpath(variables['output']))

# ==============================================================================
# Grafico tension del fotodiodo (V) vs corriente del controlador (mA). Sin pasar
# por rubidio
# ==============================================================================
with open(file = os.path.join(input_path + os.path.normpath('/dia_1/medicion_sinrb.pkl')), mode = "rb") as archive:
    datos_sinrb = pickle.load(file = archive)

fig, ax = plt.subplots(nrows = 1, ncols = 1)
ax.scatter(datos_sinrb['corriente_setting'], datos_sinrb['tension'], s = 5, color = 'black', label = 'Datos')
ax.errorbar(
datos_sinrb['corriente_setting'], 
datos_sinrb['tension'],
yerr = 2*(5*5/256), #(escala*divisiones/resolución)*2 (*2 pq asignamos el 200 porciento debido a ruido)
marker = '.', 
fmt = 'None', 
capsize = 2, 
color = 'black', 
label = 'Error de los datos')
ax.set_xlabel('Corriente controlador láser [mA]')
ax.set_ylabel('Tensión de fotodiodo [V]')
ax.grid(visible = True)
ax.legend()
fig.show()
# fig.savefig(fname = os.path.join(output_path + os.path.normpath('/dia_1/medicion_sinrb.png')))

# ==============================================================================
# Grafico tension del fotodiodo (V) vs corriente del controlador (mA). Pasando p
# or rubidio
# ==============================================================================
with open(file = os.path.join(input_path + os.path.normpath('/dia_1/medicion_conrb.pkl')), mode = "rb") as archive:
    datos_conrb = pickle.load(file = archive)

fig, ax = plt.subplots(nrows = 1, ncols = 1)
ax.scatter(datos_conrb['corriente_setting'], datos_conrb['tension'], s = 5, color = 'black', label = 'Datos')
ax.errorbar(
datos_conrb['corriente_setting'], 
datos_conrb['tension'],
yerr = 2*(5*5/256), #(escala*divisiones/resolución)*2 (*2 pq asignamos el 200 porciento debido a ruido)
marker = '.', 
fmt = 'None', 
capsize = 2, 
color = 'black', 
label = 'Error de los datos')
ax.set_xlabel('Corriente controlador láser [mA]')
ax.set_ylabel('Tensión de fotodiodo [V]')
ax.grid(visible = True)
ax.legend()
fig.show()
# fig.savefig(fname =os.path.join(output_path + os.path.normpath('/dia_1/medicion_conrb.png')))

# ==============================================================================
# Grafico tension del fotodiodo (V) vs temperatura del controlador (ºC). Con pas
# o por rubidio
# ==============================================================================
with open(file = os.path.join(input_path + os.path.normpath('/dia_1/medicion_1200uA.pkl')), mode = "rb") as archive:
    datos_1200uA = pickle.load(file = archive)

fig, ax = plt.subplots(nrows = 1, ncols = 1)
ax.scatter(datos_1200uA['temperatura'], datos_1200uA['tension'], s = 5, color = 'black', label = 'Datos')
ax.errorbar(
datos_1200uA['temperatura'], 
datos_1200uA['tension'],
yerr = 2*(5*5/256), #(escala*divisiones/resolución)*2 (*2 pq asignamos el 200 porciento debido a ruido)
marker = '.', 
fmt = 'None', 
capsize = 2, 
color = 'black', 
label = 'Error de los datos')
ax.set_xlabel(r'Corriente controlador láser [$^{o}C$]')
ax.set_ylabel('Tensión de fotodiodo [V]')
ax.grid(visible = True)
ax.legend()
fig.show()
# fig.savefig(fname =os.path.join(output_path + os.path.normpath('/dia_1/medicion_1200uA.png')))

# ==============================================================================
# Captura del canal 1 del osciloscopio a 24.126 ºC y modulado de rampa de 1Hz. C
# on paso por rubidio
# ==============================================================================
with open(file = os.path.join(input_path + os.path.normpath('/dia_1/absorcion_24_126C.pkl')), mode = "rb") as archive:
    datos_24_126C = pickle.load(file = archive)

fig, ax = plt.subplots(nrows = 1, ncols = 1)
ax.scatter(datos_24_126C['tiempo'][1230:1930], datos_24_126C['tension'][1230:1930], s = 5, color = 'black', label = 'Datos')
# ax.errorbar(
# datos_24_126C['tiempo'][1230:1930], 
# datos_24_126C['tension'][1230:1930],
# yerr = 2*(5*5/256), #(escala*divisiones/resolución)*2 (*2 pq asignamos el 200 porciento debido a ruido)
# marker = '.', 
# fmt = 'None', 
# capsize = .2, 
# color = 'black', 
# label = 'Error de los datos')
ax.set_xlabel('Tiempo [s]')
ax.set_ylabel('Tensión de fotodiodo [V]')
ax.grid(visible = True)
ax.legend()
fig.show()
# fig.savefig(fname =os.path.join(output_path + os.path.normpath('/absorcion_24_126C.png')))

################################################DIA2#######################################################

# ==============================================================================
# Vemos las mediciones con iman. Estas deberían tener 8 picos por efecto Zeeman.
# ==============================================================================
for i in range(1,20):
    i=1
    i +=1
    with open(file = os.path.join(input_path + os.path.normpath(f'/dia_2/medicion_{i}_sin_iman.pkl')), mode = "rb") as archive:
        datos_1 = pickle.load(file = archive)
    with open(file = f'C:/Users/jocta/Downloads/medicion_{i}_sin_iman.pkl', mode = "rb") as archive:
        datos = pickle.load(file = archive)
    datos_1['unidades']
    datos['unidades']
    
    
    uno, dos, ho = 100,2,1
    t = 55.0
    datos['unidades']  = f'Medimos diferencia. -OSC: VERT: CH1: {uno}mV, CH2: {dos}mV; HOR: {ho}ms. -GEN: FREQ: 10hz; AMPL: 20mV pk2pk. -RB: {t}ºC. -LAS: SETPOINT: 0.9 A.'
    f'Medimos un haz. -OSC: VERT: CH1: {uno}mV, CH2: {dos}mV; HOR: {ho}ms. -GEN: FREQ: 10hz; AMPL: 20mV pk2pk. -RB: {t}ºC. -LAS: SETPOINT: 0.9 A.'
    f'Medimos un haz. -OSC: VERT: CH1: {uno}V, CH2: {dos}mV; HOR: {ho}ms. -GEN: FREQ: 10hz; AMPL: 20mV pk2pk. -RB: {t}ºC. -LAS: SETPOINT: 0.9 A.'
    
    with open(file = os.path.join(input_path + os.path.normpath(f'/dia_2/medicion_{i}_sin_iman.pkl')), mode = "wb") as archive:
        datos = pickle.dump(obj = datos, file = archive)
fig, ax = plt.subplots(nrows = 1, ncols = 1)
ax.scatter(datos_24_126C['tiempo'][1230:1930], datos_24_126C['tension'][1230:1930], s = 5, color = 'black', label = 'Datos')
# ax.errorbar(
# datos_24_126C['tiempo'][1230:1930], 
# datos_24_126C['tension'][1230:1930],
# yerr = 2*(5*5/256), #(escala*divisiones/resolución)*2 (*2 pq asignamos el 200 porciento debido a ruido)
# marker = '.', 
# fmt = 'None', 
# capsize = .2, 
# color = 'black', 
# label = 'Error de los datos')
ax.set_xlabel('Tiempo [s]')
ax.set_ylabel('Tensión de fotodiodo [V]')
ax.grid(visible = True)
ax.legend()
fig.show()
# fig.savefig(fname =os.path.join(output_path + os.path.normpath('/absorcion_24_126C.png')))
