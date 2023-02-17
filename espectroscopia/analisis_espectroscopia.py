import os, pickle, matplotlib.pyplot as plt, numpy as np
from scipy.signal import find_peaks
from scipy.signal import savgol_filter
from labos.ajuste import Ajuste
from herramientas.config.config_builder import Parser, save_dict, load_dict

# Importo los paths
glob_path = os.path.normpath(os.getcwd())
variables = Parser(configuration = 'espectroscopia').config()
input_path = os.path.join(glob_path + os.path.normpath(variables['input']))
output_path = os.path.join(glob_path + os.path.normpath(variables['output']))

# ==============================================================================
# Grafico tension del fotodetector (V) vs corriente del controlador (mA). Sin pasar
# por rubidio
# ==============================================================================
fname = os.path.join(input_path + os.path.normpath('/dia_1/medicion_sinrb.pkl'))
datos_sinrb = load_dict(path = fname)

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
ax.set_ylabel('Tensión de fotodetector [V]')
ax.grid(visible = True)
ax.legend()
fig.show()
# fig.savefig(fname = os.path.join(output_path + os.path.normpath('/dia_1/medicion_sinrb.png')))

# ==============================================================================
# Grafico tension del fotodetector (V) vs corriente del controlador (mA). Pasando p
# or rubidio
# ==============================================================================
fname = os.path.join(input_path + os.path.normpath('/dia_1/medicion_conrb.pkl'))
datos_conrb = load_dict(path = fname)

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
ax.set_ylabel('Tensión de fotodetector [V]')
ax.grid(visible = True)
ax.legend()
fig.show()
# fig.savefig(fname =os.path.join(output_path + os.path.normpath('/dia_1/medicion_conrb.png')))

# ==============================================================================
# Grafico tension del fotodetector (V) vs temperatura del controlador (ºC). Con pas
# o por rubidio
# ==============================================================================
fname = os.path.join(input_path + os.path.normpath('/dia_1/medicion_1200uA.pkl'))
datos_1200uA = load_dict(path = fname)

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
ax.set_ylabel('Tensión de fotodetector [V]')
ax.grid(visible = True)
ax.legend()
fig.show()
# fig.savefig(fname =os.path.join(output_path + os.path.normpath('/dia_1/medicion_1200uA.png')))

# ==============================================================================
# Captura del canal 1 del osciloscopio a 24.126 ºC y modulado de rampa de 1Hz. C
# on paso por rubidio
# ==============================================================================
fname = os.path.join(input_path + os.path.normpath('/dia_1/absorcion_24_126C.pkl'))
datos_24_126C = load_dict(path = fname)

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
ax.set_ylabel('Tensión de fotodetector [V]')
ax.grid(visible = True)
ax.legend()
fig.show()
# fig.savefig(fname =os.path.join(output_path + os.path.normpath('/absorcion_24_126C.png')))

################################################DIA2#######################################################
# ==============================================================================
# Vemos las mediciones con iman. Estas deberían tener 8 picos por efecto Zeeman.
# ==============================================================================
fig, ax = plt.subplots(nrows = 1, ncols = 1)
for i in range(5, 8):
    fname = os.path.join(input_path + os.path.normpath(f'/dia_2/medicion_{i}_con_iman.pkl'))
    datos = load_dict(path = fname)
    temp = datos['temperatura_rb']
    ax.plot(datos['tiempo'], datos['tension_1'], label = f'T = {temp}'+r'$^{o}C$')
ax.set_xlabel('Tiempo [s]')
ax.set_ylabel('Tensión del fotodetector [V]')
ax.grid(visible = True)
fig.suptitle('Con imán')
fig.legend()
fig.tight_layout()
fig.show()
# fig.savefig(fname =os.path.join(output_path + os.path.normpath('/ultimas_tempereaturas_con_iman.png')))

# ==============================================================================
# Vemos las mediciones sin iman. Estas deberían tener 4 picos por los dos isótop
# os.
# ==============================================================================
fig, ax = plt.subplots(nrows = 1, ncols = 1)
for i in range(12, 20):
    fname = os.path.join(input_path + os.path.normpath(f'/dia_2/medicion_{i}_sin_iman.pkl'))
    datos = load_dict(path = fname)
    temp = datos['temperatura_rb']
    ax.plot(datos['tiempo'], datos['tension_1'], label = f'T = {temp}'+r'$^{o}C$')
ax.set_xlabel('Tiempo [s]')
ax.set_ylabel('Tensión del fotodetector [V]')
ax.grid(visible = True)
fig.suptitle('Sin imán')
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
# TEMPERATURA LASER: 23.927 ºC
# ==============================================================================

fname = os.path.join(input_path + os.path.normpath(f'/dia_2/medicion_{i}_con_iman.pkl'))
datos = load_dict(path = fname)

# Suavizo la tensión del canal 2
aj = Ajuste(
x = datos['tiempo'],
y = datos['tension_2'],
cov_y = datos['error_canal_2'])
aj.fit('a_0+a_1*x')
tension_2 = aj.parametros[0] + datos['tiempo']*aj.parametros[1]

indice_picos = find_peaks(
# x = -datos['tension_1'].reshape(-1),
x = -savgol_filter(datos['tension_1'].reshape(-1), window_length = 11, polyorder = 0),
distance = 25,
# threshold = .5,
# wlen = 5,
# prominence = (0.01,50), 
width = 25
)[0].tolist()

indice_picos += find_peaks(
# x = datos['tension_1'].reshape(-1),
x = savgol_filter(datos['tension_1'].reshape(-1), window_length = 11, polyorder = 0),
distance = 25,
# threshold = .5,
# wlen = 5,
# prominence = (0.01,50), 
width = 25
)[0].tolist()


fig, ax = plt.subplots(nrows =1, ncols = 1)
ax.scatter(
datos['tiempo'],
datos['tension_1'],
# savgol_filter(datos['tension_1'].reshape(-1), window_length = 11, polyorder = 0),
s = 2)

# ax.plot(datos['tiempo'],savgol_filter(datos['tension_1'].reshape(-1), 
#     window_length = 11, 
#     polyorder = 0), color ='red')
ax.scatter(datos['tiempo'][indice_picos],datos['tension_1'][indice_picos], color = 'black')
# ax.scatter(datos['tiempo'], tension_2, s = 2)
# ax.scatter(datos['tiempo'][indice_picos],tension_2[indice_picos], color = 'black')
fig.show()

indice_picos.sort()
indice_picos

# corriente = .0009 + .0002*tension_2[indice_picos]
# datos['indices_absorcion'] = indice_picos
# datos['unidades'] += ' ABSOR: unidad A'
# # datos['corrientes_absorcion'] = corriente
# with open(file = fname, mode = "wb") as archive:
#     pickle.dump(file = archive, obj = datos)

################################################DIA3#######################################################

# Estas son las frecuencias de transición del Rb en Ghz
frecuencias_transiciones = np.array([
377104.39131402085, # 87 2-->1'
377105.2058140209, # 87 2-->2'
377105.90987848374, # 85 2-->1'
377106.2714604837, # 85 2-->2'
377108.9456109228, # 85 1-->1'
377109.3071929228, # 85 1-->2'
377111.2259966318, # 87 1-->1' 
377112.0404966318 # 87 1-->2'
])

# Estas son las frecuencias medias de c/transición del Rb en Ghz
frecuencias_medias = np.array([
377111.6332466318, # 87 -->1
377109.1264019228, # 85 -->1
377106.09066948376, # 85 -->2
377104.7985640209 # 87 -->2
])

# Gráfico para entender el orden y la asignación
cmap = plt.get_cmap('plasma')
cmap_values = np.linspace(0., 1., len(frecuencias_transiciones))
colors = cmap(cmap_values)
colors_rgb = ['#{0:02x}{1:02x}{2:02x}'.format(int(255*a), int(255*b), int(255*c)) for a, b, c, _ in colors]

fig, ax = plt.subplots(nrows =1, ncols = 1)
for c, f in zip(colors_rgb, frecuencias_transiciones):
    ax.vlines(x = f,ymin=0,ymax =1,color=c, label = f'{f}')
fig.legend()
fig.show()

for i in range(4,20):
    fname = os.path.join(os.path.normpath(input_path) + os.path.normpath(f'/dia_2/medicion_{i}_sin_iman.pkl'))
    datos = load_dict(path = fname)

    # Hago un ajuste para suavizar el canal 2 y tener la tension de los picos
    aj = Ajuste(
    x = datos['tiempo'],
    y = datos['tension_2'],
    cov_y = datos['error_canal_2'])
    aj.fit('a_0+a_1*x')

    tension_2 = aj.parametros[0] + datos['tiempo']*aj.parametros[1]
    corriente_2 = .0009 + .0002*tension_2
    tension_picos = aj.parametros[0] + aj.parametros[1]*datos['tiempo'][datos['indices_absorcion']]
    corriente_picos = .0009 + .0002*tension_picos # A

    # fig, ax = plt.subplots(nrows =1, ncols = 1)
    # ax.scatter(datos['tiempo'],datos['tension_1'], s = 2)
    # ax.scatter(datos['tiempo'][datos['indices_absorcion']],datos['tension_1'][datos['indices_absorcion']], color = 'black')
    # ax.scatter(datos['tiempo'], tension_2, s = 2)
    # ax.scatter(datos['tiempo'][datos['indices_absorcion']],tension_2[datos['indices_absorcion']], color = 'black')
    # fig.show()


    aj_2 = Ajuste(x = frecuencias_medias.reshape(-1,1), y = corriente_picos)
    aj_2.fit('a_0+a_1*x')

    frecuencias = (corriente_2-aj_2.parametros[0])/aj_2.parametros[1]
    
    texto = ' Ghz, '.join([str(np.round(f)) for f in frecuencias[datos['indices_absorcion']].reshape(-1)]) + ' Ghz'
    
    fig, ax = plt.subplots(nrows =1, ncols = 1)
    # ax.scatter(frecuencias, datos['tension_1'], s = 2)
    temp = datos['temperatura_rb']
    ax.plot(frecuencias, datos['tension_1'], label = f'T = {temp}'+r'$^{o}C$')
    ax.scatter(
    frecuencias[datos['indices_absorcion']],
    datos['tension_1'][datos['indices_absorcion']],
    color = 'black')
    # label = texto)
    ax.set_xlim(left = frecuencias[datos['indices_absorcion']][-1,:]-2 , right = frecuencias[datos['indices_absorcion']][0,:]+1)
    ax.set_xticks(frecuencias[datos['indices_absorcion']].reshape(-1).tolist())
    ax.set_ylabel('Tensión [V]')
    ax.set_xlabel('Frecuencia [Ghz]')
    ax.grid(visible = True)
    fig.legend()
    fig.show()

