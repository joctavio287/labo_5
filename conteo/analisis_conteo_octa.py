import os, matplotlib.pyplot as plt, numpy as np, pandas as pd
from matplotlib import rcParams as rc
from scipy.signal import find_peaks
from scipy.signal import savgol_filter
from labos.ajuste import Ajuste
from labos.propagacion import Propagacion_errores
from herramientas.config.config_builder import Parser, save_dict, load_dict

# Importo los paths
glob_path = os.path.normpath(os.getcwd())
variables = Parser(configuration = 'conteo').config()
input_path = os.path.join(glob_path + os.path.normpath(variables['input']))
output_path = os.path.join(glob_path + os.path.normpath(variables['output']))

def find_nearest_arg(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def definir_tension(freq:float):
 
    '''
    Para convertir frecuencia del disco esmerilado a tensión con la cual debo alimentar
    
    Parameters
    ----------
    freq : float
        frecuencia del disco.

    Returns
    -------
    float
        tension fuente.

    '''
    return .35639131796658835*freq + 1.6922156274324787

def definir_frecuencia(tension:float):
 
    '''
    Para convertir tension a frecuencia del disco esmerilado
    
    Parameters
    ----------
    freq : float
        frecuencia del disco.

    Returns
    -------
    float
        tension fuente.

    '''
    return (tension - 1.6922156274324787)/.35639131796658835

# =================
# Fuente no térmica
# =================

# Determinamos el umbral

# Definición del umbral
paso = 2*.005*10/256 # medicion['unidades'].split('CH1: ')[1].split(';')[0]
maximo_altura = .02
bins = np.arange(-maximo_altura, 0, paso)

# Laser
tensiones = []
# carpeta = '/poisson(10ms)/laser_2v_bis/'
carpeta = '/viejo/laser_prendido_4/'

for f in os.listdir(os.path.join(input_path + carpeta)):
    medicion = load_dict(fname = os.path.join(input_path + carpeta + f))
    # medicion['unidades'] = 'Reescribimos el umbral en 0.' +medicion['unidades']
    # medicion['indice_picos'] = find_peaks(-medicion['tension'], height = 0)[0]#, distance = 50)[0]
    # medicion['indice_picos'] = medicion['indice_picos'][medicion['tension'][medicion['indice_picos']]<0]
    # medicion['tiempo_picos'] = medicion['tiempo'][medicion['indice_picos']]
    # medicion['tension_picos'] = medicion['tension'][medicion['indice_picos']]
    
    # save_dict(os.path.join(input_path + carpeta + f), medicion, rewrite = True)
    tensiones.append(medicion['tension_picos'].reshape(-1,1))
    # tensiones.append(medicion['tension'].reshape(-1,1))

tensiones = np.concatenate(tensiones, axis = 0)

# Ruido
tensiones_ruido = []
carpeta = '/poisson(10ms)/ruido_2v/'
for f in os.listdir(os.path.join(input_path + carpeta)):
    medicion = load_dict(fname = os.path.join(input_path + carpeta + f))
    tensiones_ruido.append(medicion['tension_picos'].reshape(-1,1))
tensiones_ruido = np.concatenate(tensiones_ruido, axis = 0)

plt.figure()
# plt.hist(tensiones_ruido,
#           bins = bins,
#           label = "Ruido",
#           histtype = "step", 
#           color = "green")
# plt.vlines(umbral, 0, 80000, color = 'red', label = f'umbral: {umbral}')

plt.hist(tensiones,
          bins = bins,
          label = "Laser prendido",
          histtype = "step", 
          color = "blue")
plt.legend()
plt.xlabel('Tensión [V]')
plt.ylabel('Número de eventos')
plt.grid(visible = True, alpha=0.3)
plt.yscale('log')
plt.show(block = False)

# Hacemos las cuentas habiendo fijado el umbral
umbral = 2 #tensiones_ruido.mean() - 3*np.std(tensiones_ruido)
ocurrencias = []
for f in os.listdir(os.path.join(input_path + carpeta)):
    medicion = load_dict(fname = os.path.join(input_path + carpeta + f))

    # Sacamos los indices   
    indice_picos = find_peaks(-medicion['tension'], height = umbral, distance = 10)[0]
    tension_picos = medicion['tension'][indice_picos]
    ocurrencia = len(tension_picos[tension_picos<-umbral])

    # if ocurrencia>0:
    #     plt.figure()
    #     plt.plot(medicion['tiempo'], medicion['tension'],'o-', markersize =1)
    #     plt.scatter(medicion['tiempo'][indice_picos], medicion['tension'][indice_picos], color = 'red')
    #     plt.show(block = False)
    ocurrencias.append(ocurrencia)

# ocurrencias = np.array(ocurrencias)
# cuentas, bordes = np.histogram(ocurrencias)
cuentas, frecuencia = np.unique(ocurrencias, return_counts = True)
frecuencia = frecuencia

plt.figure()
# plt.stairs(cuentas, bordes, fill=True)
# plt.vlines(bordes, 0, cuentas.max(), colors='w')
plt.bar(cuentas, frecuencia)
plt.xlabel('Número de fotones')
plt.ylabel('Ocurrencia')
plt.grid(visible = True, alpha=0.3)
plt.show(block = False)

# ====================
# Distribución Poisson
# ====================

# Determinamos el umbral

# Definición del umbral
paso = 2*.5*10/256 # medicion['unidades'].split('CH1: ')[1].split(';')[0]
maximo_altura =6
bins = np.arange(-maximo_altura, 0, paso)

# Laser
tensiones = []
carpeta = '/poisson(10ms)/laser_2v_bis/'

for f in os.listdir(os.path.join(input_path + carpeta)):
    medicion = load_dict(fname = os.path.join(input_path + carpeta + f))
    # medicion['unidades'] = 'Reescribimos el umbral en 0.' +medicion['unidades']
    # medicion['indice_picos'] = find_peaks(-medicion['tension'], height = 0)[0]#, distance = 50)[0]
    # medicion['indice_picos'] = medicion['indice_picos'][medicion['tension'][medicion['indice_picos']]<0]
    # medicion['tiempo_picos'] = medicion['tiempo'][medicion['indice_picos']]
    # medicion['tension_picos'] = medicion['tension'][medicion['indice_picos']]
    
    # save_dict(os.path.join(input_path + carpeta + f), medicion, rewrite = True)
    tensiones.append(medicion['tension_picos'].reshape(-1,1))
    # tensiones.append(medicion['tension'].reshape(-1,1))

tensiones = np.concatenate(tensiones, axis = 0)

# # Ruido
# tensiones_ruido = []
# carpeta = '/poisson(10ms)/ruido_2v/'
# for f in os.listdir(os.path.join(input_path + carpeta)):
#     medicion = load_dict(fname = os.path.join(input_path + carpeta + f))
#     tensiones_ruido.append(medicion['tension_picos'].reshape(-1,1))
# tensiones_ruido = np.concatenate(tensiones_ruido, axis = 0)

plt.figure()
# plt.hist(tensiones_ruido,
#           bins = bins,
#           label = "Ruido",
#           histtype = "step", 
#           color = "green")
# plt.vlines(umbral, 0, 80000, color = 'red', label = f'umbral: {umbral}')

plt.hist(tensiones,
          bins = bins,
          label = "Laser prendido",
          histtype = "step", 
          color = "blue")
plt.legend()
plt.xlabel('Tensión [V]')
plt.ylabel('Número de eventos')
plt.grid(visible = True, alpha=0.3)
plt.yscale('log')
plt.show(block = False)

# Hacemos las cuentas habiendo fijado el umbral
umbral = 2 #tensiones_ruido.mean() - 3*np.std(tensiones_ruido)
ocurrencias = []
for f in os.listdir(os.path.join(input_path + carpeta)):
    medicion = load_dict(fname = os.path.join(input_path + carpeta + f))

    # Sacamos los indices   
    indice_picos = find_peaks(-medicion['tension'], height = umbral, distance = 10)[0]
    tension_picos = medicion['tension'][indice_picos]
    ocurrencia = len(tension_picos[tension_picos<-umbral])

    # if ocurrencia>0:
    #     plt.figure()
    #     plt.plot(medicion['tiempo'], medicion['tension'],'o-', markersize =1)
    #     plt.scatter(medicion['tiempo'][indice_picos], medicion['tension'][indice_picos], color = 'red')
    #     plt.show(block = False)
    ocurrencias.append(ocurrencia)

# ocurrencias = np.array(ocurrencias)
# cuentas, bordes = np.histogram(ocurrencias)
cuentas, frecuencia = np.unique(ocurrencias, return_counts = True)
frecuencia = frecuencia

plt.figure()
# plt.stairs(cuentas, bordes, fill=True)
# plt.vlines(bordes, 0, cuentas.max(), colors='w')
plt.bar(cuentas, frecuencia)
plt.xlabel('Número de fotones')
plt.ylabel('Ocurrencia')
plt.grid(visible = True, alpha=0.3)
plt.show(block = False)

# ==================
# Distribución Bose
# ==================

# Definición del umbral
paso = 2*.005*10/256 # medicion['unidades'].split('CH1: ')[1].split(';')[0]
maximo_altura = .021
bins = np.arange(-maximo_altura, 0, paso)

# Laser
tensiones = []
carpeta = '/bose(50ns)/laser_2v/'

for f in os.listdir(os.path.join(input_path + carpeta)):
    medicion = load_dict(fname = os.path.join(input_path + carpeta + f))
    # medicion['unidades'] = 'Reescribimos el umbral en 0.' +medicion['unidades']
    # medicion['indice_picos'] = find_peaks(-medicion['tension'], height = 0)[0]#, distance = 50)[0]
    # medicion['indice_picos'] = medicion['indice_picos'][medicion['tension'][medicion['indice_picos']]<0]
    # medicion['tiempo_picos'] = medicion['tiempo'][medicion['indice_picos']]
    # medicion['tension_picos'] = medicion['tension'][medicion['indice_picos']]
    
    # save_dict(os.path.join(input_path + carpeta + f), medicion, rewrite = True)
    tensiones.append(medicion['tension_picos'].reshape(-1,1))
    # tensiones.append(medicion['tension'].reshape(-1,1))

tensiones = np.concatenate(tensiones, axis = 0)

# Ruido
tensiones_ruido = []
carpeta = '/bose(50ns)/ruido_2v/'
for f in os.listdir(os.path.join(input_path + carpeta)):
    medicion = load_dict(fname = os.path.join(input_path + carpeta + f))
    # Sacamos los indices   
    indice_picos = find_peaks(-medicion['tension'], height = umbral, distance = 10)[0]
    tension_picos = medicion['tension'][indice_picos]
    tensiones_ruido.append(medicion['tension_picos'].reshape(-1,1))
tensiones_ruido = np.concatenate(tensiones_ruido, axis = 0)

plt.figure()
plt.hist(tensiones_ruido,
          bins = bins,
          label = "Ruido",
          histtype = "step", 
          color = "green")

plt.hist(tensiones,
          bins = bins,
          label = "Laser prendido",
          histtype = "step", 
          color = "blue")
plt.legend()
plt.xlabel('Tensión [V]')
plt.ylabel('Número de eventos')
plt.grid(visible = True, alpha=0.3)
plt.yscale('log')
plt.show(block = False)

# Hacemos las cuentas habiendo fijado el umbral
carpeta = '/bose(50ns)/laser_2v/'
umbral = .0039 #tensiones_ruido.mean() - 3*np.std(tensiones_ruido)
ocurrencias = []
for f in os.listdir(os.path.join(input_path + carpeta)):
    medicion = load_dict(fname = os.path.join(input_path + carpeta + f))

    # Sacamos los indices   
    indice_picos = find_peaks(-medicion['tension'], height = umbral, distance = 10)[0]
    tension_picos = medicion['tension'][indice_picos]
    ocurrencia = len(tension_picos[tension_picos<-umbral])

    # if ocurrencia>0:
    #     plt.figure()
    #     plt.plot(medicion['tiempo'], medicion['tension'],'o-', markersize =1)
    #     plt.scatter(medicion['tiempo'][indice_picos], medicion['tension'][indice_picos], color = 'red')
    #     plt.show(block = False)
    ocurrencias.append(ocurrencia)

# Ajuste
cuentas, apariciones = np.unique(ocurrencias, return_counts = True)
error_suma_apariciones = np.sqrt(np.sum(apariciones))
frecuencia = apariciones/np.sum(apariciones)
error_frecuencia = np.sqrt((np.sqrt(apariciones)/np.sum(apariciones))**2 + 
        (apariciones*error_suma_apariciones/(np.sum(apariciones)**2))**2
        )

# Valor medio y error
media = np.array([cuenta*frec for cuenta, frec in zip(cuentas, frecuencia)]).sum()
media_std = np.sqrt(np.sum([(cuenta*e_frec)**2 for cuenta, e_frec in zip(cuentas, error_frecuencia)]))

formula_bose = 'a_0**x/((1+a_0)**(1+x))'
def formula_bose_f(a_0, x):
    return a_0**x/((1+a_0)**(1+x))

franja_error = Propagacion_errores(
        variables = [('a_0', media)], 
        errores = np.array([media_std]).reshape(-1,1), 
        formula = formula_bose, 
        dominio = cuentas
        ).fit()[1]


plt.figure()

plt.plot(cuentas, formula_bose_f(media, cuentas), color = 'C0')
plt.errorbar(cuentas, frecuencia, yerr = error_frecuencia, fmt = 'o', capsize = 1.5, color = 'C1', label = 'Experimental')
# plt.scatter(cuentas, frecuencia, marker = '.', fmt = 'None', capsize = 1.5, color = 'black', label = 'Experimental')
plt.bar(cuentas, frecuencia, color = 'C2')

plt.errorbar(cuentas, formula_bose_f(media, cuentas), yerr = franja_error, marker = 'o', capsize = 1.5, color = 'C0', label = 'Ajuste')
# plt.plot(cuentas, formula_bose_f(media, cuentas), 'r.-', label = 'Ajuste', alpha = .5)
# plt.plot(cuentas, formula_bose_f(media, cuentas) + franja_error, '--', color = 'green', label = 'Error del ajuste')
# plt.plot(cuentas, formula_bose_f(media, cuentas) - franja_error, '--', color = 'green')
# plt.fill_between(cuentas, formula_bose_f(media, cuentas) - franja_error, formula_bose_f(media, cuentas) + franja_error, facecolor = "gray", alpha = 0.3)

plt.xlabel('Número de fotones')
plt.ylabel('Ocurrencia')
plt.grid(visible = True, alpha=0.3)
plt.legend()
plt.show(block = False)

# Tiempo de coherencia
color = {20:'C0',35:'C1',60:'C2'}
plt.figure()   
for l in [20,35,60]:
    carpeta = f'/correlacion_frecuencia/correlacion_{l}e-1v_0_ohms/'
    correlaciones = []
    normalizaciones = []
    for f in os.listdir(os.path.join(input_path + carpeta)):
        medicion = load_dict(fname = os.path.join(input_path + carpeta + f))
        tension = -medicion['tension'] + (medicion['tension']).max()
        correlacion = np.correlate(tension, tension, mode ='same')
        normalizaciones.append(np.correlate(tension, tension))
        correlaciones.append(correlacion.reshape(-1,1))

    normalizacion = np.mean(normalizaciones)
    correlacion_promedio = np.mean(np.array(correlaciones), axis =0)/normalizacion
    diferencia_temporal = medicion['diferencia_temporal']

    eje_temporal = np.arange(-1250*diferencia_temporal, 1250*diferencia_temporal, diferencia_temporal)

    t_c = np.round(np.abs(1000*eje_temporal[find_nearest_arg(correlacion_promedio, 0.5)]),5)

    plt.plot(eje_temporal, correlacion_promedio, color = color[l])
    plt.hlines(y = .5, xmin = 0, xmax = t_c/1000, color = color[l], linestyles = 'dashdot',
                label = f'Frecuencia = {np.round(definir_frecuencia(l/10),2)} Hz y ' +r'$\tau_c$' + f' = {t_c} ms')
    plt.xlim(0,0.005)
plt.ylabel('Correlación')
plt.xlabel(r'$\Delta \tau$ [s]')
plt.grid(visible = True)
plt.legend()
plt.show(block = False)






