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
cuentas, frecuencia = np.unique(ocurrencias, return_counts = True)
error_frecuencia = np.sqrt(frecuencia) + frecuencia*.001

formula_bose = 
ajuste = Ajuste(
    x = cuentas.reshape(-1,1),
    y = frecuencia.reshape(-1,1), 
    cov_y = error_frecuencia.reshape(-1,1)
)

ajuste.fit(formula)
plt.figure()
# plt.stairs(cuentas, bordes, fill=True)
# plt.vlines(bordes, 0, cuentas.max(), colors='w')
plt.errorbar(cuentas, frecuencia, yerr = error_frecuencia, marker = '.', fmt = 'None', capsize = 1.5, color = 'black')
plt.bar(cuentas, frecuencia, color = 'C2')
plt.xlabel('Número de fotones')
plt.ylabel('Ocurrencia')
plt.grid(visible = True, alpha=0.3)
plt.show(block = False)


# %%

# Tiempo de coherencia

carpeta = '/poisson(10ms)/laser_2v/'

mediciones = []
for f in os.listdir('C:/GRUPO 8/correlacion_frecuencia/correlacion_20e-1v_0_ohms'):
    medicion = load_dict(fname = os.path.join('C:/GRUPO 8/correlacion_frecuencia/correlacion_20e-1v_0_ohms'+ f))
    


# Levantamos la curva promediando
correlacion_promedio = np.mean([load_dict(f'C:/GRUPO 8/correlacion_frecuencia/{carpeta}/{i}')['correlacion'] for i in os.listdir(f'C:/GRUPO 8/correlacion_frecuencia/{carpeta}')], axis = 0)
diferencia_temporal = load_dict(fname)['diferencia_temporal']
eje_temporal = np.arange(-1250*diferencia_temporal, 1250*diferencia_temporal, diferencia_temporal)
plt.figure()   
plt.plot(eje_temporal, correlacion_promedio)
plt.grid(visible = True)
plt.show(block = False)

































