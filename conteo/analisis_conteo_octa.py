import os, matplotlib.pyplot as plt, numpy as np, pandas as pd
from matplotlib import rcParams as rc
from scipy.signal import find_peaks
from scipy.signal import savgol_filter
from labos.ajuste import Ajuste
from labos.propagacion import Propagacion_errores
from herramientas.config.config_builder import Parser, save_dict, load_dict

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

#%%
# ====================
# Distribución Poisson
# ====================

# Determinamos el umbral

# Definición del umbral
paso = 1.5*1*10/256
maximo_altura = 6
bins = np.arange(-maximo_altura, 0, paso)

# Laser
tensiones = []
carpeta = '/poisson(10ms)/laser_2v_bis/'

for f in os.listdir(os.path.join(glob_path + carpeta)):
    medicion = load_dict(fname = os.path.join(glob_path + carpeta + f))
    # medicion['unidades'] = 'Reescribimos el umbral en 0.' +medicion['unidades']
    # medicion['indice_picos'] = find_peaks(-medicion['tension'], height = 0)[0]#, distance = 50)[0]
    # medicion['indice_picos'] = medicion['indice_picos'][medicion['tension'][medicion['indice_picos']]<0]
    # medicion['tiempo_picos'] = medicion['tiempo'][medicion['indice_picos']]
    # medicion['tension_picos'] = medicion['tension'][medicion['indice_picos']]
    
    # save_dict(os.path.join(glob_path + carpeta + f), medicion, rewrite = True)
    tensiones.append(medicion['tension_picos'].reshape(-1,1))
tensiones = np.concatenate(tensiones, axis = 0)

# Ruido
tensiones_ruido = []
carpeta = '/poisson(10ms)/ruido_2v/'
for f in os.listdir(os.path.join(glob_path + carpeta)):
    medicion = load_dict(fname = os.path.join(glob_path + carpeta + f))
    tensiones_ruido.append(medicion['tension_picos'].reshape(-1,1))
tensiones_ruido = np.concatenate(tensiones_ruido, axis = 0)

# # Calculamos el umbral
# umbral = tensiones_ruido.mean() - 3*np.std(tensiones_ruido)
umbral = 1.20

plt.figure()
plt.hist(tensiones_ruido,
          bins = bins,
          label = "Ruido",
          histtype = "step", 
          color = "green")
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
plt.show()

# Encontramos los conteos habiendo fijado el umbral
umbral = 1.2
ocurrencias = []
carpeta = '/poisson(10ms)/laser_2v/'
for f in os.listdir(os.path.join(glob_path + carpeta)):
    medicion = load_dict(fname = os.path.join(glob_path + carpeta + f))

    # Sacamos los indices   
    indice_picos = find_peaks(-medicion['tension'], height = umbral)[0]#, distance = 50)[0]
    tiempo_picos = medicion['tiempo'][indice_picos]
    tension_picos = medicion['tension'][indice_picos]
    # plt.figure()
    # plt.plot(medicion['tiempo'], medicion['tension'],'o-', markersize =1)
    # plt.scatter(medicion['tiempo'][indice_picos], medicion['tension'][indice_picos], color = 'red')
    # plt.show()
    
    ocurrencia = len(medicion['tension_picos'][medicion['tension_picos']<-umbral])
    ocurrencia = len(tension_picos[tension_picos<-umbral])
    ocurrencias.append(ocurrencia)

# ocurrencias = np.array(ocurrencias)
# cuentas, bordes = np.histogram(ocurrencias)
cuentas, frecuencia = np.unique(ocurrencias, return_counts = True)
frecuencia = frecuencia/np.sum(frecuencia)

plt.figure()
# plt.stairs(cuentas, bordes, fill=True)
# plt.vlines(bordes, 0, cuentas.max(), colors='w')
plt.bar(cuentas, frecuencia)
plt.xlabel('Frecuencia')
plt.ylabel('Número de eventos')
plt.grid(visible = True, alpha=0.3)
plt.show()

# Mandamos un mensaje canchero al grupo
mensaje_tel(
    api_token = '5448153732:AAGhKraJQquEqMfpD3cb4rnTcrKB6U1ViMA',
    chat_id = '-693150998',
    mensaje = 'JUEGA BOOCA'
    )

# %%
# Determinar el umbral




## Distribución Bose

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
plt.show()

































