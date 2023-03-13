#La alternativa que me funciona
# from herramientas.config.config_builder import Parser
import sys
sys.path.append('./herramientas/config/')
from config_builder import Parser

import os
import numpy as np
import matplotlib.pyplot as plt
import pickle

# Importo los paths
glob_path = os.path.normpath(os.getcwd())
variables = Parser(configuration = 'conteo').config()

def load_dict(fname:str):
    '''
    Para cargar un diccionario en formato pickle. 
    '''
    isfile = os.path.isfile(fname)
    if not isfile:
        print(f'El archivo {fname} no existe')
        return
    try:
        with open(file = fname, mode = "rb") as archive:
            data = pickle.load(file = archive)
        return data
    except:
        print('Algo fallo')
    return


#GRAFICO FIGURA 4 PAPER

path_ruido = './input/conteo/bose(50ns)/ruido_2v'
tensiones_ruido = []
for filename in os.listdir(path_ruido):
    medicion = load_dict(fname = path_ruido + '/' + filename)
    tensiones_ruido.append(medicion['tension_picos'])
tensiones_ruido = np.concatenate(tensiones_ruido)

path = './input/conteo/bose(50ns)/laser_2v'
tensiones = []
for filename in os.listdir(path):
    medicion = load_dict(fname = path + '/' + filename)
    tensiones.append(medicion['tension_picos'])
tensiones = np.concatenate(tensiones)




paso = 0.00020
maximo_altura = 0.015
bins = np.arange(-maximo_altura, 0, paso)

umbral = -0.00422

fig, axs = plt.subplots(2, 1, sharex=True)

axs[0].plot(medicion['tension']*1e3, medicion['tiempo']*1e6, linewidth=0.7)
ylim = axs[0].get_ylim()
axs[0].vlines(umbral*1e3, -1, 1, linestyles='dashed', colors='black')
axs[0].set_ylim(ylim)
axs[0].set_ylabel('Tiempo [$\mu$s]')
axs[0].grid(visible = True, alpha=0.3)

axs[1].hist(tensiones*1e3,
          label = "Laser prendido",
          bins = bins*1e3,
        #   histtype = "step", 
          color = "green")
axs[1].hist(tensiones_ruido*1e3,
         bins = bins*1e3,
         label = "Ruido",
        #  histtype = "step", 
         color = "red")
axs[1].set_yscale('log')
ylim = axs[1].get_ylim()
axs[1].vlines(umbral*1e3, -10, 10e5, linestyles='dashed', colors='black')
axs[1].legend(loc='upper left')
axs[1].set_xlabel('Tensión [mV]')
axs[1].set_ylabel('Número de eventos')
axs[1].grid(visible = True, alpha=0.3)
axs[1].set_yscale('log')
axs[1].set_ylim(0, ylim[-1])
plt.savefig('./output/conteo/ruido-umbral-ej.png', dpi=400)
plt.show()