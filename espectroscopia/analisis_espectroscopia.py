import os, matplotlib.pyplot as plt, numpy as np, pandas as pd
from matplotlib import rcParams as rc
rc['axes.formatter.useoffset'] = True
rc['axes.formatter.min_exponent'] = 0# 0
from scipy.signal import find_peaks
from scipy.signal import savgol_filter
from labos.ajuste import Ajuste
from labos.propagacion import Propagacion_errores
from herramientas.config.config_builder import Parser, save_dict, load_dict

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

# Importo los paths
glob_path = os.path.normpath(os.getcwd())
variables = Parser(configuration = 'espectroscopia').config()
input_path = os.path.join(glob_path + os.path.normpath(variables['input']))
output_path = os.path.join(glob_path + os.path.normpath(variables['output']))

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

#========================================================================================
# Señal absorción temperatura 35 ºC con y sin componente lineal
# =======================================================================================
fname = os.path.join(os.path.normpath(input_path) + os.path.normpath(f'/dia_2/medicion_{8}_sin_iman_e.pkl'))
datos = load_dict(fname = fname)
temp  = datos['temperatura_rb']

# Ajuste para quitar componente lineal a los graficos de absorcion
tiempo_aj = np.concatenate((datos['tiempo'][:300,0], datos['tiempo'][1800:,0]))
tension_aj = np.concatenate((datos['tension_1'][:300,0], datos['tension_1'][1800:,0]))
errores_aj = np.concatenate((datos['error_canal_1'][:300,0], datos['error_canal_1'][1800:,0]))
aj_3 = Ajuste(tiempo_aj.reshape(-1,1), tension_aj.reshape(-1,1), errores_aj.reshape(-1,1))
aj_3.fit(formula = 'a_0+a_1*x')
ordenada, pendiente = aj_3.parametros[0], aj_3.parametros[1]

# Con componente lineal
plt.figure()
plt.scatter(
x = datos['tiempo'][100:2000,0], 
y = datos['tension_1'][100:2000,0], 
s = 5, 
label = f'T = {temp}'+r'$^{o}C$')
plt.xlabel(xlabel = 'Tiempo [s]')
plt.ylabel(ylabel = 'Tensión [V]')
plt.grid(visible = True)
plt.legend(loc  = 'upper left')
plt.tight_layout()
plt.show(block = False)            
# plt.savefig(fname = os.path.join(output_path + os.path.normpath('/informe/absorcion_35_con_lineal.svg')))

# Sin componente lineal
plt.figure()
plt.scatter(
x = datos['tiempo'][100:2000,0], 
y = datos['tension_1'][100:2000,0] - ordenada - pendiente*datos['tiempo'][100:2000,0], 
s = 5, 
label = f'T = {temp}'+r'$^{o}C$')
plt.xlabel(xlabel = 'Tiempo [s]')
plt.ylabel(ylabel = 'Tensión [V]')
plt.grid(visible = True)
plt.legend(loc  = 'lower left')
plt.tight_layout()
plt.show(block = False)            
# plt.savefig(fname = os.path.join(output_path + os.path.normpath('/informe/absorcion_35.svg')))
#========================================================================================
#========================================================================================

#========================================================================================
# Transformación de corriente a frecuencia para temperatura 35 ºC
#========================================================================================
fname = os.path.join(os.path.normpath(input_path) + os.path.normpath(f'/dia_2/medicion_{4}_sin_iman_e.pkl'))
datos = load_dict(fname = fname)

# Hago un ajuste para suavizar el canal 2 y tener la tension del canal 2 en los picos. Esto se traduce en la corriente con la cual alimentabamos el controlador
aj = Ajuste(x = datos['tiempo'],y = datos['tension_2'],cov_y = datos['error_canal_2'])
aj.fit('a_0+a_1*x')
tension_2 = aj.parametros[0] + datos['tiempo']*aj.parametros[1]
corriente_2 = .0009 + .1*tension_2 # A
tension_picos_2 = aj.parametros[0] + aj.parametros[1]*datos['tiempo'][datos['indices_absorcion']]
corriente_picos_2 = .0009 + .1*tension_picos_2 # A
error_corriente_picos_2 = [Propagacion_errores(formula = '.0009+.1*x', variables = [('x', t)], errores = np.full(shape = (1,1), fill_value = datos['error_canal_2'][0,0])).fit()[1] for t in tension_picos_2[:,0]]
error_corriente_picos_2 = np.array(error_corriente_picos_2).reshape(-1,1)

# Paso a mA para el gráfico
corriente_picos_2 = corriente_picos_2*1000 # mA
error_corriente_picos_2 = error_corriente_picos_2*(1000) # mA

# Ploteo las frecuencias de los picos (frecuencias_medias) contra la corriente en cada pico y determino la relación funcional
aj_2 = Ajuste(x = frecuencias_medias.reshape(-1,1), y = corriente_picos_2, cov_y = error_corriente_picos_2)
aj_2.fit('a_0+a_1*x')
x_auxiliar = np.linspace(frecuencias_medias[0]-10, frecuencias_medias[-1]+10, 1000)
franja_error = Propagacion_errores(
        variables = [('a_0',aj_2.parametros[0]), ('a_1',aj_2.parametros[1])], 
        errores = aj_2.cov_parametros*1, 
        formula = 'a_0+a_1*x', 
        dominio = x_auxiliar
        ).fit()[1]
plt.figure()
plt.scatter(x = frecuencias_medias, y = corriente_picos_2.reshape(-1), s = 5, color = 'black', label = 'Datos')
plt.errorbar(frecuencias_medias, corriente_picos_2.reshape(-1), yerr = error_corriente_picos_2.reshape(-1), marker = '.', fmt = 'None', capsize = 1.5, color = 'black', label = 'Error de los datos')
plt.plot(x_auxiliar, aj_2.parametros[0]+aj_2.parametros[1]*x_auxiliar, 'r-', label = 'Ajuste', alpha = .5)
plt.plot(x_auxiliar, aj_2.parametros[0]+aj_2.parametros[1]*x_auxiliar + franja_error, '--', color = 'green', label = 'Error del ajuste')
plt.plot(x_auxiliar, aj_2.parametros[0]+aj_2.parametros[1]*x_auxiliar - franja_error, '--', color = 'green')
plt.fill_between(x_auxiliar, aj_2.parametros[0]+aj_2.parametros[1]*x_auxiliar - franja_error, aj_2.parametros[0]+aj_2.parametros[1]*x_auxiliar + franja_error, facecolor = "gray", alpha = 0.3)
plt.xlabel(xlabel = 'Frecuencia [GHz]')
plt.ylabel(ylabel = 'Intensidad de corriente [mA]')
plt.grid(visible = True)
plt.legend()
plt.tight_layout()
plt.show(block = False)            
# plt.savefig(fname = os.path.join(output_path + os.path.normpath('/informe/lineal_corriente_frec_35.svg')))

# Parámetros y bondad:
aj_2.parametros, np.sqrt(np.diag(aj_2.cov_parametros))
aj_2.bondad()
#========================================================================================
#========================================================================================

#========================================================================================
# Señal absorción en función de frecuencia temperatura 35 ºC con y sin componente lineal
#========================================================================================
# Hago el eje de frecuencias con la primera medicion
i=4
fname = os.path.join(input_path + os.path.normpath(f'/dia_2/medicion_{i}_sin_iman_e.pkl'))
datos = load_dict(fname = fname)

# Suavizo el canal 2
aj = Ajuste(
x = datos['tiempo'],
y = datos['tension_2'],
cov_y = datos['error_canal_2'])
aj.fit('a_0+a_1*x')
tension_2 = aj.parametros[0] + datos['tiempo']*aj.parametros[1]

# Lo transformo a intensiad de corriente
corriente_2 = .0009 + .1*tension_2 # A
corriente_picos_2 = corriente_2[datos['indices_absorcion']]
tension_picos_2 = aj.parametros[0] + aj.parametros[1]*datos['tiempo'][datos['indices_absorcion']]
error_corriente_picos_2 = [Propagacion_errores(formula = '.0009+.1*x', variables = [('x', t)], errores = np.full(shape = (1,1), fill_value = datos['error_canal_2'][0,0])).fit()[1] for t in tension_picos_2[:,0]]
error_corriente_picos_2 = np.array(error_corriente_picos_2).reshape(-1,1)

# Segundo ajuste: para determinar la relación entre frecuencia (dada por conocida) e intensidad corriente
aj_2 = Ajuste(x = frecuencias_medias.reshape(-1,1), y = corriente_picos_2, cov_y = error_corriente_picos_2)
aj_2.fit('a_0+a_1*x') #-> de aca se despeja la frecuencia como (corriente_2-aj_2.parametros[0])/aj_2.parametros[1]
# for i in [8,12,19]:
# for i in [4,8,9,12,13,19]:
i = 8
fname = os.path.join(input_path + os.path.normpath(f'/dia_2/medicion_{i}_sin_iman_e.pkl'))
datos = load_dict(fname = fname)

# Hago una conversión de pico a texto para la transicion
cmap = plt.get_cmap('plasma')
cmap_values = np.linspace(0., 1., len(datos['indices_absorcion']))
colors = cmap(cmap_values)
colors_rgb = ['#{0:02x}{1:02x}{2:02x}'.format(int(255*a), int(255*b), int(255*c)) for a, b, c, _ in colors]

conversion = {
datos['indices_absorcion'][3]: r"$^{87}Rb: \,  3 \rightarrow 2',3'$",
datos['indices_absorcion'][2]: r"$^{85}Rb: \,  3 \rightarrow 2',3'$",
datos['indices_absorcion'][1]: r"$^{85}Rb: \,  2 \rightarrow 2',3'$",
datos['indices_absorcion'][0]: r"$^{87}Rb: \,  2 \rightarrow 2',3'$"}

# Suavizo el canal 2 y transformo a corriente con el ajuste de temperatura 30 ºC
aj_i = Ajuste(
x = datos['tiempo'],
y = datos['tension_2'],
cov_y = datos['error_canal_2'])
aj_i.fit('a_0+a_1*x')
tension_2 = aj_i.parametros[0] + datos['tiempo']*aj_i.parametros[1]
corriente_2 = .0009 + .1*tension_2 # A
frecuencias = (corriente_2-aj_2.parametros[0])/aj_2.parametros[1]

# Tercer ajuste: para quitar componente lineal a los graficos de absorcion
frecuencia_aj = np.concatenate((frecuencias[:300,0], frecuencias[1800:,0]))
tension_aj = np.concatenate((datos['tension_1'][:300,0], datos['tension_1'][1800:,0]))
errores_aj = np.concatenate((datos['error_canal_1'][:300,0], datos['error_canal_1'][1800:,0]))
aj_3 = Ajuste(frecuencia_aj.reshape(-1,1), tension_aj.reshape(-1,1), errores_aj.reshape(-1,1))
aj_3.fit(formula = 'a_0+a_1*x')
ordenada, pendiente = aj_3.parametros[0], aj_3.parametros[1]
temp = datos['temperatura_rb']

# Grafico
plt.figure()
plt.scatter(
frecuencias[100:2000,0], 
(datos['tension_1']- pendiente*frecuencias - ordenada)[100:2000,0],
s = 5)

for ind, c in zip(datos['indices_absorcion'], colors_rgb):
    plt.vlines(frecuencias[ind], ymin = (datos['tension_1']- pendiente*frecuencias - ordenada).min(), ymax = (datos['tension_1']- pendiente*frecuencias - ordenada).max(), color = c, label = conversion[ind])
plt.xlabel('Frecuencia [GHz]')
plt.ylabel('Tensión del fotodetector [V]')
plt.legend(loc = (.7, .1)) #(.078, .7)
plt.tight_layout()
plt.grid(visible = True)
plt.show(block = False)            
# plt.savefig(fname = os.path.join(output_path + os.path.normpath('/informe/absorcion_35_frec.svg')))
#========================================================================================
#========================================================================================

#========================================================================================
# TRES TEMPERATURAS ESQUEMATICO DEL CORRIEMIENTO DE LOS PICOS
#========================================================================================
# 17,18,19 --55 (19)
# 13,14,15,16 --50 (13)
# 11, 12 --45 (12)
# 9,10 --40 (9)
# 6,7,8 --35 (8)
# 1,2,3,4,5 --30 (4)

# Hago el eje de frecuencias con la primera medicion
i=4
fname = os.path.join(input_path + os.path.normpath(f'/dia_2/medicion_{i}_sin_iman_e.pkl'))
datos = load_dict(fname = fname)

# Primer ajuste: suavizar el canal 2
aj = Ajuste(
x = datos['tiempo'],
y = datos['tension_2'],
cov_y = datos['error_canal_2'])
aj.fit('a_0+a_1*x')
tension_2 = aj.parametros[0] + datos['tiempo']*aj.parametros[1]

# Lo transformo a intensiad de corriente
corriente_2 = .0009 + .1*tension_2 # A
corriente_picos_2 = corriente_2[datos['indices_absorcion']]
tension_picos_2 = aj.parametros[0] + aj.parametros[1]*datos['tiempo'][datos['indices_absorcion']]
error_corriente_picos_2 = [Propagacion_errores(formula = '.0009+.1*x', variables = [('x', t)], errores = np.full(shape = (1,1), fill_value = datos['error_canal_2'][0,0])).fit()[1] for t in tension_picos_2[:,0]]
error_corriente_picos_2 = np.array(error_corriente_picos_2).reshape(-1,1)

# Segundo ajuste: para determinar la relación entre frecuencia (dada por conocida) e intensidad corriente
aj_2 = Ajuste(x = frecuencias_medias.reshape(-1,1), y = corriente_picos_2, cov_y = error_corriente_picos_2)
aj_2.fit('a_0+a_1*x')
frecuencias = (corriente_2-aj_2.parametros[0])/aj_2.parametros[1]

plt.figure()
for i in [4,12,19]:
    fname = os.path.join(input_path + os.path.normpath(f'/dia_2/medicion_{i}_sin_iman.pkl'))
    datos = load_dict(fname = fname)
    
    # Ajuste intermedio: suavizar el canal 2
    aj_i = Ajuste(
    x = datos['tiempo'],
    y = datos['tension_2'],
    cov_y = datos['error_canal_2'])
    aj_i.fit('a_0+a_1*x')
    tension_2 = aj_i.parametros[0] + datos['tiempo']*aj_i.parametros[1]
    corriente_2 = .0009 + .1*tension_2 # A
    frecuencias = (corriente_2-aj_2.parametros[0])/aj_2.parametros[1]

    # Tercer ajuste: para quitar componente lineal a los graficos de absorcion
    frecuencia_aj = np.concatenate((frecuencias[:300,0], frecuencias[1800:,0]))
    tension_aj = np.concatenate((datos['tension_1'][:300,0], datos['tension_1'][1800:,0]))
    errores_aj = np.concatenate((datos['error_canal_1'][:300,0], datos['error_canal_1'][1800:,0]))
    aj = Ajuste(frecuencia_aj.reshape(-1,1), tension_aj.reshape(-1,1), errores_aj.reshape(-1,1))
    aj.fit(formula = 'a_0+a_1*x')
    ordenada, pendiente = aj.parametros[0], aj.parametros[1]
    temp = datos['temperatura_rb']
    if i == 4:
        plt.scatter(
            np.round(frecuencias[700:1600,0],2), 
            (datos['tension_1'] - pendiente*frecuencias - ordenada)[700:1600,0], 
            s = 2, 
            label = f'T = ({temp}'+r'$\pm$ 0.1'+')'+r' $^{o}C$')
        
    else:
        plt.scatter(
            np.round(frecuencias,2), 
            datos['tension_1'] - pendiente*frecuencias - ordenada,
            s = 2, 
            label = f'T = ({temp}'+r'$\pm$ 0.1'+')'+r' $^{o}C$')
plt.xlabel('Frecuencia [GHz]')
plt.ylabel('Tensión del fotodetector [V]')
plt.ticklabel_format(axis='x', useOffset=377100)
plt.grid(visible = True)
plt.legend(loc = (.6, .1)) #(.078, .7)
plt.tight_layout()
plt.show(block = False)
# plt.savefig(fname = os.path.join(output_path + os.path.normpath('/informe/tres_temperaturas.svg')))
#========================================================================================
# Gráfico de ancho de los picos en función de la temperatura para 85 Rb
#========================================================================================

# Transformo a frecuencia la primera medición
i=4
fname = os.path.join(input_path + os.path.normpath(f'/dia_2/medicion_{i}_sin_iman_e.pkl'))
datos = load_dict(fname = fname)

# Suavizo el canal 2
aj = Ajuste(
x = datos['tiempo'],
y = datos['tension_2'],
cov_y = datos['error_canal_2'])
aj.fit('a_0+a_1*x')
tension_2 = aj.parametros[0] + datos['tiempo']*aj.parametros[1]

# Lo transformo a intensiad de corriente
corriente_2 = .0009 + .1*tension_2 # A
corriente_picos_2 = corriente_2[datos['indices_absorcion']]
tension_picos_2 = aj.parametros[0] + aj.parametros[1]*datos['tiempo'][datos['indices_absorcion']]
error_corriente_picos_2 = [Propagacion_errores(
    formula = '.0009+.1*x', 
    variables = [('x', t)], 
    errores = np.full(shape = (1,1), 
                      fill_value = datos['error_canal_2'][0,0])
    ).fit()[1] for t in tension_picos_2[:,0]]
error_corriente_picos_2 = np.array(error_corriente_picos_2).reshape(-1,1)

# Determino la relación entre frecuencia (dada por conocida) e intensidad corriente
aj_2 = Ajuste(x = frecuencias_medias.reshape(-1,1), y = corriente_picos_2, cov_y = error_corriente_picos_2)
aj_2.fit('a_0+a_1*x')

# Encuentro el máximo de los dos picos más prominentes (85 Rb) y el ancho a mitad del pico
anchos_2 = {}
anchos_3 = {}
for i in [4,8,9,12,13,19]:
    fname = os.path.join(input_path + os.path.normpath(f'/dia_2/medicion_{i}_sin_iman_e.pkl'))
    datos = load_dict(fname = fname)
    
    # Suavizo el canal 2 y transformo a corriente
    aj_i = Ajuste(
    x = datos['tiempo'],
    y = datos['tension_2'],
    cov_y = datos['error_canal_2'])
    aj_i.fit('a_0+a_1*x')
    tension_2 = aj_i.parametros[0] + datos['tiempo']*aj_i.parametros[1]
    corriente_2 = .0009 + .1*tension_2 # A
    error_corriente_picos_2 = np.array([Propagacion_errores(
    formula = '.0009+.1*x', 
    variables = [('x', t)], 
    errores = np.full(shape = (1,1),fill_value = datos['error_canal_2'][0,0])).fit()[1] for t in tension_picos_2[:,0]]
    )
    # Transformo usando los datos ajustados a 30 ºC
    frecuencias = (corriente_2-aj_2.parametros[0])/aj_2.parametros[1] 
    errores = np.array([[error_corriente_picos_2[0]**2,0,0],[0,aj_2.cov_parametros[0,0],aj_2.cov_parametros[0,1]],[0,aj_2.cov_parametros[1,0],aj_2.cov_parametros[1,1]]])
    error_frecuencias = [Propagacion_errores(formula = f'(c_0-c_1)/c_2', variables = [('c_0', c),('c_1',aj_2.parametros[0]), ('c_2',aj_2.parametros[1])], errores = errores).fit()[1] for c in corriente_2[:,0]]

    # Quito la componente lineal a los graficos de absorcion
    frecuencia_aj = np.concatenate((frecuencias[:300,0], frecuencias[1800:,0]))
    tension_aj = np.concatenate((datos['tension_1'][:300,0], datos['tension_1'][1800:,0]))
    errores_aj = np.concatenate((datos['error_canal_1'][:300,0], datos['error_canal_1'][1800:,0]))
    aj_3 = Ajuste(frecuencia_aj.reshape(-1,1), tension_aj.reshape(-1,1), errores_aj.reshape(-1,1))
    aj_3.fit(formula = 'a_0+a_1*x')
    ordenada, pendiente = aj_3.parametros[0], aj_3.parametros[1]
    temp = datos['temperatura_rb']

    # Grafico para encontrar el primer máximo y el ancho visualmente
    fig, ax = plt.subplots(nrows = 1, ncols = 1)
    if i==4:
        ax.scatter(
        frecuencias[750:1750,0], 
        datos['tension_1'][750:1750,0]-pendiente*frecuencias[750:1750,0]-ordenada, 
        s = 2, 
        label = f'T = {temp}'+r'$^{o}C$')
    else:
        ax.scatter(
        frecuencias, 
        datos['tension_1']- pendiente*frecuencias - ordenada,
        s = 5, 
        label = f'T = {temp}'+r' $^{o}C$')
    
    # Encuentro la tension a la mitad del pico
    tension_sin_off = datos['tension_1']- pendiente*frecuencias - ordenada
    tension_en_ancho = np.round(tension_sin_off[datos['indices_absorcion'][1]]/2,5)
    ax.hlines(y = tension_en_ancho, xmin = frecuencias[0], xmax = frecuencias[-1], color = 'red')
    ax.grid(visible = True)

    # Guardo los clicks manualmente
    clicks_2 = fig.ginput(n = -1, timeout = -1)
    clicks_x_2 = [click[0] for click in clicks_2]

    error_picos_2 = np.array([error_frej/2cuencias[find_nearest(frecuencias, clicks_x_2[1])], error_frecuencias[find_nearest(frecuencias, clicks_x_2[0])]]).reshape(-1,1)
    error_anchos_2 = Propagacion_errores(formula = 'a_1-a_0', variables = [('a_1', clicks_x_2[1]),('a_0', clicks_x_2[0])], errores = error_picos_2).fit()[1]

    anchos_2[datos['temperatura_rb']] = clicks_x_2[1]-clicks_x_2[0], error_anchos_2
    fig.show()

    # Grafico para encontrar el segundo máximo y el ancho visualmente
    fig, ax = plt.subplots(nrows = 1, ncols = 1)
    if i==4:
        ax.scatter(
        frecuencias[750:1750,0], 
        datos['tension_1'][750:1750,0]-pendiente*frecuencias[750:1750,0]-ordenada, 
        s = 2, 
        label = f'T = {temp}'+r'$^{o}C$')
    else:
        ax.scatter(
        frecuencias, 
        datos['tension_1']- pendiente*frecuencias - ordenada,
        s = 5, 
        label = f'T = {temp}'+r' $^{o}C$')
    
    # Encuentro la tension a la mitad del pico
    tension_en_ancho = np.round(tension_sin_off[datos['indices_absorcion'][2]]/2,5)
    ax.hlines(y = tension_en_ancho, xmin = frecuencias[0], xmax = frecuencias[-1], color = 'red')
    ax.grid(visible = True)

    # Guardo los clicks manualmente
    clicks_3 = fig.ginput(n = -1, timeout = -1)
    clicks_x_3 = [click[0] for click in clicks_3]

    error_picos_3 = np.array([error_frecuencias[find_nearest(frecuencias, clicks_x_3[1])], error_frecuencias[find_nearest(frecuencias, clicks_x_3[0])]]).reshape(-1,1)
    error_anchos_3 = Propagacion_errores(formula = 'a_1-a_0', variables = [('a_1', clicks_x_3[1]),('a_0', clicks_x_3[0])], errores = error_picos_3).fit()[1]

    anchos_3[datos['temperatura_rb']] = clicks_x_3[1]-clicks_x_3[0], error_anchos_3
    fig.show()

plt.figure()
plt.errorbar(x = list(anchos_2.keys()), y = list(v for v,j in anchos_2.values()), yerr = list(j for v,j in anchos_2.values()), marker = 'o', fmt = 'None', capsize = 1.5)
plt.scatter(x = list(anchos_2.keys()), y = list(v for v,j in anchos_2.values()), label =  r"$^{85}Rb: \,  2 \rightarrow 2',3'$", s = 10)
plt.errorbar(x = list(anchos_2.keys()), y = list(v for v,j in anchos_3.values()), yerr = list(j for v,j in anchos_3.values()), marker = 'o', fmt = 'None', capsize = 1.5, color = 'green')
plt.scatter(x = list(anchos_2.keys()), y = list(v for v,j in anchos_3.values()), color = 'green', label = r"$^{85}Rb: \,  3 \rightarrow 2',3'$", s = 10)
plt.xlabel('Temperatura [' + r'$^{o}\, C$]')
plt.ylabel(r'$\Delta f$ [Ghz]')
plt.grid(visible = True)
plt.legend()
# plt.show(block = False)
# plt.savefig(fname =os.path.join(output_path + os.path.normpath('/informe/anchos.svg')))

#========================================================================================
# ZEEMAN: [(1, 29.9), (2, 35.1), (3, 40.7), (4, 45.0), (5, 50.0), (6, 55.0)]
#========================================================================================

# Hago el eje de frecuencias con la primera medicion
i=4
fname = os.path.join(input_path + os.path.normpath(f'/dia_2/medicion_{i}_sin_iman_e.pkl'))
datos = load_dict(fname = fname)

# Primer ajuste: suavizar el canal 2
aj = Ajuste(
x = datos['tiempo'],
y = datos['tension_2'],
cov_y = datos['error_canal_2'])
aj.fit('a_0+a_1*x')
tension_2 = aj.parametros[0] + datos['tiempo']*aj.parametros[1]

# Lo transformo a intensiad de corriente
corriente_2 = .0009 + .1*tension_2 # A
corriente_picos_2 = corriente_2[datos['indices_absorcion']]
tension_picos_2 = aj.parametros[0] + aj.parametros[1]*datos['tiempo'][datos['indices_absorcion']]
error_corriente_picos_2 = [Propagacion_errores(formula = '.0009+.1*x', variables = [('x', t)], errores = np.full(shape = (1,1), fill_value = datos['error_canal_2'][0,0])).fit()[1] for t in tension_picos_2[:,0]]
error_corriente_picos_2 = np.array(error_corriente_picos_2).reshape(-1,1)

# Segundo ajuste: para determinar la relación entre frecuencia (dada por conocida) e intensidad corriente
aj_2 = Ajuste(x = frecuencias_medias.reshape(-1,1), y = corriente_picos_2, cov_y = error_corriente_picos_2)
aj_2.fit('a_0+a_1*x')
frecuencias = (corriente_2-aj_2.parametros[0])/aj_2.parametros[1]

# Ahora vamos a ver Zeeman
i=2
fname = os.path.join(input_path + os.path.normpath(f'/dia_4/medicion_f_{i}_con_iman.pkl'))
datos = load_dict(fname = fname)

# Hago una conversión de pico a transicion
cmap = plt.get_cmap('plasma')
cmap_values = np.linspace(0., 1., len(datos['indices_absorcion']))
colors = cmap(cmap_values)
colors_rgb = ['#{0:02x}{1:02x}{2:02x}'.format(int(255*a), int(255*b), int(255*c)) for a, b, c, _ in colors]

conversion = {# TODO: cambiar MON +- por sigma de polarizacion mas menos: 
datos['indices_absorcion'][7]: r"$\sigma_{+}:^{87}Rb: \,  3 \rightarrow 2',3'$",
datos['indices_absorcion'][6]: r"$\sigma_{-}:^{87}Rb: \,  3 \rightarrow 2',3'$",
datos['indices_absorcion'][5]: r"$\sigma_{+}:^{85}Rb: \,  3 \rightarrow 2',3'$",
datos['indices_absorcion'][4]: r"$\sigma_{-}:^{85}Rb: \,  3 \rightarrow 2',3'$",
datos['indices_absorcion'][3]: r"$\sigma_{+}:^{85}Rb: \,  2 \rightarrow 2',3'$",
datos['indices_absorcion'][2]: r"$\sigma_{-}:^{85}Rb: \,  2 \rightarrow 2',3'$",
datos['indices_absorcion'][1]: r"$\sigma_{+}:^{87}Rb: \,  2 \rightarrow 2',3'$",
datos['indices_absorcion'][0]: r"$\sigma_{-}:^{87}Rb: \,  2 \rightarrow 2',3'$"}

# Ajuste intermedio: suavizar el canal 2
aj_i = Ajuste(
x = datos['tiempo'],
y = datos['tension_2'],
cov_y = datos['error_canal_2'])
aj_i.fit('a_0+a_1*x')
tension_2 = aj_i.parametros[0] + datos['tiempo']*aj_i.parametros[1]
corriente_2 = .0009 + .1*tension_2 # A
frecuencias = (corriente_2-aj_2.parametros[0])/aj_2.parametros[1]

temp = datos['temperatura_rb']

plt.figure()
plt.scatter(frecuencias[600:2100,0], datos['tension_1'][600:2100,0], s = 3)#, label = f'T = {temp}'+r' $^{o}C$')
for ind, c in zip(datos['indices_absorcion'], colors_rgb):
    plt.vlines(frecuencias[ind], ymin = datos['tension_1'].min(), ymax = datos['tension_1'].max(), color = c, label = conversion[ind])

plt.xlabel('Frecuencias [GHz]')
plt.ylabel('Tensión del fotodetector [V]')
plt.grid(visible = True)
legend = plt.legend(loc ='best', facecolor=(1, 1, 1, 0.01))# edgecolor = 'transparent'
# for lh in legend.legendHandles: 
#     lh.set_alpha(1.75)
plt.tight_layout()
plt.show(block = False)
# plt.savefig(fname =os.path.join(output_path + os.path.normpath('/informe/zeeman.svg')))
#========================================================================================
#========================================================================================


#========================================================================================
# PID
#========================================================================================

# Se hace la coincidencia solo con la primera medicion
fname = os.path.join(input_path + os.path.normpath(f'/dia_4/offset 0/P=50, I=0, D=0.csv'))
df = pd.read_csv(fname, delimiter = ',', header = 1)
tiempo, tension_1, tension_2 = df.values[:,0]*1e-6, df.values[:,1], df.values[:,3] # s, V, V
plt.figure()
plt.plot(tiempo, tension_1,  '.-',  label = 'P, I, D = 50, 0, 0')
plt.xlabel('Tiempo [s]')
plt.ylabel('Tensión [V]')
plt.grid(visible = True)
plt.legend()
plt.show(block = False)
# plt.savefig(fname =os.path.join(output_path + os.path.normpath('/informe/P=50.svg')))

fname = os.path.join(input_path + os.path.normpath(f'/dia_4/offset 0/P=80, I=0, D=0.csv'))
df = pd.read_csv(fname, delimiter = ',', header = 1)
tiempo, tension_1, tension_2 = df.values[:,0]*1e-6, df.values[:,1], df.values[:,3] # s, V, V
plt.figure()
plt.plot(tiempo, tension_1,  '.-',  label = 'P, I, D = 80, 0, 0')
plt.xlabel('Tiempo [s]')
plt.ylabel('Tensión [V]')
plt.grid(visible = True)
plt.legend()
plt.show(block = False)
# plt.savefig(fname =os.path.join(output_path + os.path.normpath('/informe/P=80.svg')))

#========================================================================================
#========================================================================================
'P = 80, I = 50, D = 0.csv'
