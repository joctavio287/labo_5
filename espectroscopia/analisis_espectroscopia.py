import os, pickle, matplotlib.pyplot as plt, numpy as np
from scipy.signal import find_peaks
from scipy.signal import savgol_filter
from labos.ajuste import Ajuste
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
for i in range(1,8):
    with open(file = os.path.join(input_path + os.path.normpath(f'/dia_2/medicion_{i}_con_iman.pkl')), mode = "rb") as archive:
        # globals()[f'datos_{i}'] = pickle.load(file = archive) #
        datos = pickle.load(file = archive)

fig, ax = plt.subplots(nrows = 1, ncols = 1)
ax.plot(eval(f'datos_{i}')['tiempo'], eval(f'datos_{i}')['tension_1'], color = 'black', label = 'Datos')
ax.set_xlabel('Tiempo [s]')
ax.set_ylabel('Tensión de fotodiodo [V]')
ax.grid(visible = True)
ax.legend()
fig.show()
# fig.savefig(fname =os.path.join(output_path + os.path.normpath('/absorcion_24_126C.png')))

# ==============================================================================
# La fórmula para deducir la corriente con la cual alimentamos el láser es, acor
# de al manual:
#
#    I_las_diode = I_las_diode_set + I_las_diode_max * tension_mod * (1/10V)
#
# Como la tensión de la moduladora era una funcion rampa de 20 Hz y 20 mV pk2pk;
# la intensidad de corriente que fijamos en el láser era de 0.9 mA y el máximo v
# alor de input de intensidad de corriente era ±2 mA. Entonces
# 
#            I_las_diode (t) = .0009 + .0002*canal_2(.01V, 20Hz, t) # A
#
#   TEMPERATURA LASER # TODO
# ==============================================================================

# Ajuste de prueba

# Leo los datos

i = 19
fname = os.path.join(input_path + os.path.normpath(f'/dia_2/medicion_{i}_sin_iman.pkl'))
with open(file = fname, mode = "rb") as archive:
    globals()[f'datos_{i}'] = pickle.load(file = archive)

aj = Ajuste(x = eval(f'datos_{i}')['tiempo'], y = eval(f'datos_{i}')['tension_2'], cov_y = eval(f'datos_{i}')['error_canal_2'])
aj.fit('a_0+a_1*x')
tension_2 = aj.parametros[0] + eval(f'datos_{i}')['tiempo']*aj.parametros[1]

indice_picos = find_peaks(
x = -eval(f'datos_{i}')['tension_1'].reshape(-1),
# x = -savgol_filter(eval(f'datos_{i}')['tension_1'].reshape(-1), window_length = 7, polyorder = 0),
distance = 15,
# threshold = .5,
# wlen = 5,
# prominence = (0.01,50), 
width = 10
)[0].tolist()
fig, ax = plt.subplots(nrows =1, ncols = 1)
ax.scatter(eval(f'datos_{i}')['tiempo'],eval(f'datos_{i}')['tension_1'], s = 2)
# ax.plot(eval(f'datos_{i}')['tiempo'],savgol_filter(eval(f'datos_{i}')['tension_1'].reshape(-1), 
#     window_length = 11, 
#     polyorder = 0), color ='red')
ax.scatter(eval(f'datos_{i}')['tiempo'][indice_picos],eval(f'datos_{i}')['tension_1'][indice_picos], color = 'black')
ax.scatter(eval(f'datos_{i}')['tiempo'], tension_2, s = 2)
ax.scatter(eval(f'datos_{i}')['tiempo'][indice_picos],tension_2[indice_picos], color = 'black')
fig.show()

# corriente = .0009 + .0002*tension_2[indice_picos]
# eval(f'datos_{i}')['indices_absorcion'] = indice_picos
# eval(f'datos_{i}')['unidades'] += ' ABSOR: unidad A'
# eval(f'datos_{i}')['corrientes_absorcion'] = corriente
with open(file = fname, mode = "wb") as archive:
    pickle.dump(file = archive, obj = eval(f'datos_{i}'))