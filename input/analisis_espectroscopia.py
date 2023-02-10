import os, pickle, matplotlib.pyplot as plt, numpy as np
from scipy.signal import find_peaks
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit
#from labos.ajuste import Ajuste
#from herramientas.config.config_builder import Parser

# Importo los paths
glob_path = 'C:/Users/Publico/Desktop/GRUPO 8 COOLS/labo_5-master'

input_path = '/input/espectroscopia/dia_2'
output_path = '/output/espectroscopia'

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

cmap = plt.get_cmap('plasma')
cmap_values = np.linspace(0., 1., 3)
colors = cmap(cmap_values)
colors_rgb = ['#{0:02x}{1:02x}{2:02x}'.format(int(255*a), int(255*b), int(255*c)) for a, b, c, _ in colors]

fig, ax = plt.subplots(nrows = 1, ncols = 1)
for i, c in zip(range(5,8), colors_rgb):
    with open(file = os.path.join(input_path + os.path.normpath(f'/dia_2/medicion_{i}_con_iman.pkl')), mode = "rb") as archive:
        datos = pickle.load(file = archive)
    temp = datos['temperatura_rb']
    ax.plot(datos['tiempo'], datos['tension_1'], label = f'Con imán {temp}'+r'$^{o}C$')
ax.set_xlabel('Tiempo [s]')
ax.set_ylabel('Tensión del fotodiodo [V]')
ax.grid(visible = True)
fig.legend()
fig.tight_layout()
fig.show()
# fig.savefig(fname =os.path.join(output_path + os.path.normpath('/ultimas_tempereaturas_con_iman.png')))

# ==============================================================================
# Vemos las mediciones sin iman. Estas deberían tener 4 picos por los dos isótop
# os.
# ==============================================================================
cmap = plt.get_cmap('plasma')
cmap_values = np.linspace(0., 1., 8)
colors = cmap(cmap_values)
colors_rgb = ['#{0:02x}{1:02x}{2:02x}'.format(int(255*a), int(255*b), int(255*c)) for a, b, c, _ in colors]

fig, ax = plt.subplots(nrows = 1, ncols = 1)
for i, c in zip(range(12,20), colors_rgb):
    with open(file = os.path.join(input_path + os.path.normpath(f'/dia_2/medicion_{i}_sin_iman.pkl')), mode = "rb") as archive:
        datos = pickle.load(file = archive)
    datos['unidades']
    temp = datos['temperatura_rb']
    ax.plot(datos['tiempo'], datos['tension_1'], label = f'Con imán {temp}'+r'$^{o}C$')
ax.set_xlabel('Tiempo [s]')
ax.set_ylabel('Tensión del fotodiodo [V]')
ax.grid(visible = True)
fig.legend()
fig.tight_layout()
fig.show()
# fig.savefig(fname =os.path.join(output_path + os.path.normpath('/ultimas_tempereaturas_sin_iman.png')))

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
frecuencias_transiciones = [
377106.2714604837,
377105.90987848374,
377108.9456109228,
377109.3071929228,
377105.2058140209, 
377104.39131402085, 
377111.2259966318, 
377112.0404966318]

# Leo los datos

i = 19
fname = os.path.join(os.path.normpath(glob_path) +os.path.normpath(input_path) + os.path.normpath(f'/medicion_{i}_sin_iman.pkl'))
with open(file = fname, mode = "rb") as archive:
    datos = pickle.load(file = archive)

def func(x, a_0,a_1):
    return a_0 + a_1*x
params, cov_params = curve_fit(f = func ,xdata = datos['tiempo'].reshape(-1), ydata = datos['tension_2'].reshape(-1), sigma = datos['error_canal_2'].reshape(-1))
tension_2 = params[0] + datos['tiempo']*params[1]

indice_picos = find_peaks(
x = -datos['tension_1'].reshape(-1),
# x = -savgol_filter(eval(f'datos_{i}')['tension_1'].reshape(-1), window_length = 7, polyorder = 0),
distance = 15,
# threshold = .5,
# wlen = 5,
# prominence = (0.01,50), 
width = 10
)[0].tolist()

cmap = plt.get_cmap('plasma')
cmap_values = np.linspace(0., 1., 8)
colors = cmap(cmap_values)
colors_rgb = ['#{0:02x}{1:02x}{2:02x}'.format(int(255*a), int(255*b), int(255*c)) for a, b, c, _ in colors]


fig, ax = plt.subplots(nrows =1, ncols = 1)
ax.scatter(datos['tiempo'],datos['tension_1'], s = 2)
# ax.plot(eval(f'datos_{i}')['tiempo'],savgol_filter(eval(f'datos_{i}')['tension_1'].reshape(-1), 
#     window_length = 11, 
#     polyorder = 0), color ='red')
ax.scatter(datos['tiempo'][datos['indices_absorcion']],datos['tension_1'][datos['indices_absorcion']], color = 'black')
ax.scatter(datos['tiempo'], tension_2, s = 2)
ax.scatter(datos['tiempo'][datos['indices_absorcion']],tension_2[datos['indices_absorcion']], color = 'black')
fig.show()

fig, ax = plt.subplots(nrows =1, ncols = 1)
for c, f in zip(colors_rgb, frecuencias_transiciones):
    ax.vlines(x = f,ymin=0,ymax =1,color=c, label = f'{f}')
# ax.plot(eval(f'datos_{i}')['tiempo'],savgol_filter(eval(f'datos_{i}')['tension_1'].reshape(-1), 
#     window_length = 11, 
#     polyorder = 0), color ='red')
#ax.scatter(datos['tiempo'][datos['indices_absorcion']],datos['tension_1'][datos['indices_absorcion']], color = 'black')
#ax.scatter(datos['tiempo'], tension_2, s = 2)
#ax.scatter(datos['tiempo'][datos['indices_absorcion']],tension_2[datos['indices_absorcion']], color = 'black')
fig.legend()
fig.show()

frecuencias_medias = [
377111.6332466318,
377109.1264019228,
377106.09066948376,
377104.7985640209
]        

params_viejo, cov_params = curve_fit(f = func ,xdata = datos['tiempo'].reshape(-1), ydata = datos['tension_2'].reshape(-1), sigma = datos['error_canal_2'].reshape(-1))
tension_picos = func(datos['tiempo'][datos['indices_absorcion']],params[0],params[1])
plt.figure()
plt.scatter(frecuencias_medias, tension_picos.reshape(-1))
plt.plot(frecuencias_medias,func(np.array(frecuencias_medias), *params))
plt.show()
params, _ = curve_fit(f = func, xdata = frecuencias_medias, ydata = tension_picos.reshape(-1))

frecuencias = np.array((func(datos['tiempo'],*params_viejo) - params[0])/params[1])
fig, ax = plt.subplots(nrows =1, ncols = 1)
ax.scatter(frecuencias,datos['tension_1'], s = 2)

ax.scatter(frecuencias[datos['indices_absorcion']],datos['tension_1'][datos['indices_absorcion']], color = 'black')
ax.scatter(frecuencias, tension_2, s = 2)
ax.scatter(frecuencias[datos['indices_absorcion']],tension_2[datos['indices_absorcion']], color = 'black')
fig.show()


# corriente = .0009 + .0002*tension_2[indice_picos]
# eval(f'datos_{i}')['indices_absorcion'] = indice_picos
# eval(f'datos_{i}')['unidades'] += ' ABSOR: unidad A'
# eval(f'datos_{i}')['corrientes_absorcion'] = corriente
#with open(file = fname, mode = "wb") as archive:
#    pickle.dump(file = archive, obj = eval(f'datos_{i}'))



#######
i = 19
fname = os.path.join(os.path.normpath(glob_path) +os.path.normpath(input_path) + os.path.normpath(f'/medicion_{i}_sin_iman.pkl'))
with open(file = fname, mode = "rb") as archive:
    datos = pickle.load(file = archive)