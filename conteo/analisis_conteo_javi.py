#La alternativa que me funciona
# from herramientas.config.config_builder import Parser
import sys
sys.path.append('./herramientas/config/')
from config_builder import Parser, load_dict

import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy.signal import find_peaks

# Importo los paths
glob_path = os.path.normpath(os.getcwd())
variables = Parser(configuration = 'conteo').config()

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

umbral = -0.003

fig, axs = plt.subplots(2, 1, sharex=True)

axs[0].plot(medicion['tension']*1e3, medicion['tiempo']*1e6, linewidth=0.7, label='Señal osciloscopio', color='C0')
ylim = axs[0].get_ylim()
axs[0].vlines(umbral*1e3, -1, 1, linestyles='dashed', colors='black', label='Umbral')
axs[0].set_ylim(ylim)
axs[0].legend(loc='upper left')
axs[0].set_ylabel('Tiempo [$\mu$s]')
axs[0].grid(visible = True, alpha=0.3)

axs[1].hist(tensiones*1e3,
          label = "Laser prendido",
          bins = bins*1e3,
        #   histtype = "step", 
          color = "C1")
axs[1].hist(tensiones_ruido*1e3,
         bins = bins*1e3,
         label = "Ruido",
        #  histtype = "step", 
         color = "C2")
axs[1].set_yscale('log')
ylim = axs[1].get_ylim()
axs[1].vlines(umbral*1e3, -10, 10e5, linestyles='dashed', colors='black')
axs[1].legend(loc='upper left')
axs[1].set_xlabel('Tensión [mV]')
axs[1].set_ylabel('Número de eventos')
axs[1].grid(visible = True, alpha=0.3)
axs[1].set_ylim(ylim)
# plt.savefig('./output/conteo/ruido-umbral-ej.png', dpi=400)
plt.show()

########################################
## P0, P1, P2
########################################

########################################
#BOSE
########################################
path_bose = 'input\\conteo\\bose(50ns)\\laser_2v' #Ventana de 500 ns

umbral_bose = 0.003
P0_bose_total = np.zeros(2500)
P1_bose_total = np.zeros(2500)
P2_bose_total = np.zeros(2500)
for filename in os.listdir(path_bose):
  data = load_dict(path_bose + '\\' + filename)
  P0_bose = np.zeros(2500)
  P1_bose = np.zeros(2500)
  P2_bose = np.zeros(2500)
  for i in range(2500):
    if i == 0:
      indice_picos = find_peaks(-data['tension'], height = umbral_bose)[0]#, distance = 50)[0]
      tiempo_picos = data['tiempo'][indice_picos]
      tension_picos = data['tension'][indice_picos]
    else:
      indice_picos = find_peaks(-data['tension'][:-i], height = umbral_bose)[0]#, distance = 50)[0]
      tiempo_picos = data['tiempo'][:-i][indice_picos]
      tension_picos = data['tension'][:-i][indice_picos]
  
    ocurrencia = len(tension_picos[tension_picos<-umbral_bose])
    if ocurrencia == 0:
      P0_bose[i] += 1
    elif ocurrencia == 1:
      P1_bose[i] += 1 
    elif ocurrencia == 2:
      P2_bose[i] += 1
  P0_bose_total += P0_bose
  P1_bose_total += P1_bose
  P2_bose_total += P2_bose 

# ocurrencias_total = np.concatenate(ocurrencias_total, axis=1)
# ocurrencias_total = np.concatenate([np.array(i).reshape(-1, 1) for i in ocurrencias_total], axis=1)
# occ = np.sum(ocurrencias_total, axis=1)

def Pn_teo_bose(t, n, nT, T):
  return (nT/T*t)**n/(1 + nT/T * t)**(n+1)

def Pn_teo_poisson(t, n, nT, T):
  return (nT/T*t)**n/np.math.factorial(n) * np.exp(-nT/T*t)

n_mean = 3 #A CHEQUEAR
T = 500 #ns

fig, axs = plt.subplots(3, sharex=True)
vent = np.linspace(5e-7, 0, 2500)*1e9 #ns
axs[0].plot(vent, P0_bose_total/np.max(P0_bose_total), 'o',label='Datos', markevery=70)
axs[0].plot(vent, Pn_teo_bose(vent, 0, n_mean, T), label='Bose Einstein')
axs[0].plot(vent, Pn_teo_poisson(vent, 0, n_mean, T), label='Poisson')
axs[0].set_ylabel('$P_0$')
axs[0].grid()
axs[0].legend()

axs[1].plot(vent, P1_bose_total/np.max(P1_bose_total), 'o', markevery=70)
axs[1].plot(vent, Pn_teo_bose(vent, 1, n_mean, T))
axs[1].plot(vent, Pn_teo_poisson(vent, 1, n_mean, T))
axs[1].set_ylabel('$P_1$')
axs[1].grid()

axs[2].plot(vent, P2_bose_total/np.max(P2_bose_total), 'o', markevery=70)
axs[2].plot(vent, Pn_teo_bose(vent, 2, n_mean, T))
axs[2].plot(vent, Pn_teo_poisson(vent, 2, n_mean, T))
axs[2].set_ylabel('$P_2$')
axs[2].grid()
axs[2].set_xlabel('Tamaño de ventana temporal [ns]')

plt.show()

########################################
#POISSON
########################################

path_poisson = 'input\\conteo\\poisson(10ms)\\laser_2v'

umbral_poisson = 0.003
P0_poisson_total = np.zeros(2500)
P1_poisson_total = np.zeros(2500)
P2_poisson_total = np.zeros(2500)
for filename in os.listdir(path_poisson):
  data = load_dict(path_poisson + '\\' + filename)
  P0_poisson = np.zeros(2500)
  P1_poisson = np.zeros(2500)
  P2_poisson = np.zeros(2500)
  for i in range(2500):
    if i == 0:
      indice_picos = find_peaks(-data['tension'], height = umbral_poisson)[0]#, distance = 50)[0]
      tiempo_picos = data['tiempo'][indice_picos]
      tension_picos = data['tension'][indice_picos]
    else:
      indice_picos = find_peaks(-data['tension'][:-i], height = umbral_poisson)[0]#, distance = 50)[0]
      tiempo_picos = data['tiempo'][:-i][indice_picos]
      tension_picos = data['tension'][:-i][indice_picos]
  
    ocurrencia = len(tension_picos[tension_picos<-umbral_poisson])
    if ocurrencia == 0:
      P0_poisson[i] += 1
    elif ocurrencia == 1:
      P1_poisson[i] += 1 
    elif ocurrencia == 2:
      P2_poisson[i] += 1
  P0_poisson_total += P0_poisson
  P1_poisson_total += P1_poisson
  P2_poisson_total += P2_poisson 


n_mean = 1 #A CHEQUEAR
T = 100 #ms

fig, axs = plt.subplots(3, sharex=True)
vent = np.linspace(T, 0, 2500) #ms
axs[0].plot(vent, P0_poisson_total/np.max(P0_poisson_total), 'o',label='Datos', markevery=70)
axs[0].plot(vent, Pn_teo_bose(vent, 0, n_mean, T), label='Bose Einstein')
axs[0].plot(vent, Pn_teo_poisson(vent, 0, n_mean, T), label='Poisson')
axs[0].set_ylabel('$P_0$')
axs[0].grid()
axs[0].legend()

axs[1].plot(vent, P1_poisson_total/np.max(P1_poisson_total), 'o', markevery=70)
axs[1].plot(vent, Pn_teo_bose(vent, 1, n_mean, T))
axs[1].plot(vent, Pn_teo_poisson(vent, 1, n_mean, T))
axs[1].set_ylabel('$P_1$')
axs[1].grid()

axs[2].plot(vent, P2_poisson_total/np.max(P2_poisson_total), 'o', markevery=70)
axs[2].plot(vent, Pn_teo_bose(vent, 2, n_mean, T))
axs[2].plot(vent, Pn_teo_poisson(vent, 2, n_mean, T))
axs[2].set_ylabel('$P_2$')
axs[2].grid()
axs[2].set_xlabel('Tamaño de ventana temporal [ms]')

plt.show()