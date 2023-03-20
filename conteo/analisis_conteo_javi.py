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
from labos.propagacion import Propagacion_errores

# Importo los paths
glob_path = os.path.normpath(os.getcwd())
variables = Parser(configuration = 'conteo').config()

#GRAFICO FIGURA 4 PAPER

tensiones_ruido = []
carpeta = 'input/conteo/bose(50ns)/ruido_2v/'
for f in os.listdir(os.path.join(carpeta)):
    medicion = load_dict(fname = os.path.join(carpeta + f))
    for t in medicion['tension_picos']:
        tensiones_ruido.append(t)
tensiones_ruido, cuentas_ruido = np.unique(tensiones_ruido, return_counts = True)

tensiones = []
carpeta = './input/conteo/bose(50ns)/laser_2v/'
for f in os.listdir(carpeta):
    medicion = load_dict(fname = os.path.join(carpeta + f))
    for t in medicion['tension_picos']:
        tensiones.append(t)
tensiones, cuentas = np.unique(tensiones, return_counts = True)



# paso = 0.00020
# maximo_altura = 0.015
# bins = np.arange(-maximo_altura, 0, paso)

umbral = -0.003

fig, axs = plt.subplots(2, 1, sharex=True)

axs[0].plot(medicion['tension']*1e3, medicion['tiempo']*1e6, linewidth=0.7, label='Señal osciloscopio', color='C0')
ylim = axs[0].get_ylim()
axs[0].vlines(umbral*1e3, -1, 1, linestyles='dashed', colors='C2', label='Umbral')
axs[0].set_ylim(ylim)
axs[0].legend(loc='upper left')
axs[0].set_ylabel('Tiempo [$\mu$s]')
axs[0].grid(visible = True, alpha=0.3)

axs[1].bar(tensiones*1e3,
        cuentas,
        color = 'C1', 
        width= .25,
        alpha = .75,
        label = 'Laser incidiendo sobre FM')
axs[1].bar(tensiones_ruido*1e3,
        cuentas_ruido,
        color = 'C0', 
        width= .25,
        alpha = 1,
        label = 'Ruido')
axs[1].set_yscale('log')
ylim = axs[1].get_ylim()
axs[1].vlines(umbral*1e3, -10, 10e5, linestyles='dashed', colors='C2')
axs[1].legend(loc='upper left')
axs[1].set_xlabel('Tensión [mV]')
axs[1].set_ylabel('Número de eventos')
axs[1].grid(visible = True, alpha=0.3)
axs[1].set_ylim(ylim)
plt.savefig('./output/conteo/presentacion/ruido-umbral-ej.png', dpi=400)
plt.show()

##############################################
#UMBRAL RUIDO BOSE
##############################################
plt.bar(tensiones*1e3,
        cuentas,
        color = 'C1', 
        width= .25,
        alpha = .75,
        label = 'Laser incidiendo sobre FM')
plt.bar(tensiones_ruido*1e3,
        cuentas_ruido,
        color = 'C0', 
        width= .25,
        alpha = 1,
        label = 'Ruido')
plt.yscale('log')
ylim = axs[1].get_ylim()
plt.vlines(umbral*1e3, -10, 10e5, linestyles='dashed', colors='C2')
plt.legend(loc='upper left')
plt.xlabel('Tensión [mV]')
plt.ylabel('Número de eventos')
plt.grid(visible = True, alpha=0.3)
plt.ylim(ylim)
plt.savefig('./output/conteo/presentacion/ruido-umbral-bose.png', dpi=400)
plt.show()

########################################
## EJEMPLO SUB-VENTANAS
########################################
plt.plot(medicion['tiempo']*1e6, medicion['tension']*1e3, linewidth=1, label='Señal osciloscopio', color='C0')
plt.vlines(medicion['tiempo'][::500]*1e6, min(medicion['tension']*1e3), max(medicion['tension']*1e3), colors='C1', linestyles='dashed', label='Sub-ventanas')
ylim = axs[0].get_ylim()
plt.set_ylim(ylim)
plt.legend(loc='upper right')
plt.xlabel('Tiempo [$\mu$s]')
plt.ylabel('Tensión [mV]')
plt.grid(visible = True, alpha=0.3)
plt.savefig('./output/conteo/presentacion/ejemplo-subventanas.png', dpi=400)
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
ocurrencias = np.zeros((2500, 1000))
for j, filename in enumerate(os.listdir(path_bose)):
  data = load_dict(path_bose + '\\' + filename)
  P0_bose = np.zeros(2500)
  P1_bose = np.zeros(2500)
  P2_bose = np.zeros(2500)
  ocurrencia_tau = np.zeros(2500)
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
    ocurrencia_tau[i] = ocurrencia
    if ocurrencia == 0:
      P0_bose[i] += 1
    elif ocurrencia == 1:
      P1_bose[i] += 1 
    elif ocurrencia == 2:
      P2_bose[i] += 1

  ocurrencias[:,j] = ocurrencia_tau
  
  P0_bose_total += P0_bose
  P1_bose_total += P1_bose
  P2_bose_total += P2_bose 

error_frecuencia0 = np.zeros(2500)
error_frecuencia1 = np.zeros(2500)
error_frecuencia2 = np.zeros(2500)
for i in range(2500):
  cuentas, apariciones = np.unique(ocurrencias[i,:], return_counts=True)
  error_suma_apariciones = np.sqrt(np.sum(apariciones))
  frecuencia = apariciones/np.sum(apariciones)
  try:
    error_frecuencia0[i], error_frecuencia1[i], error_frecuencia2[i], *_ = np.sqrt((np.sqrt(apariciones)/np.sum(apariciones))**2 + 
          (apariciones*error_suma_apariciones/(np.sum(apariciones)**2))**2)
  except:
    try:
      error_frecuencia0[i], error_frecuencia1[i], *_ = np.sqrt((np.sqrt(apariciones)/np.sum(apariciones))**2 + 
        (apariciones*error_suma_apariciones/(np.sum(apariciones)**2))**2)
    except:
      error_frecuencia0[i], *_ = np.sqrt((np.sqrt(apariciones)/np.sum(apariciones))**2 + 
      (apariciones*error_suma_apariciones/(np.sum(apariciones)**2))**2)

P0_bose_total = P0_bose_total/1000
P1_bose_total = P1_bose_total/1000
P2_bose_total = P2_bose_total/1000
# ocurrencias_total = np.concatenate(ocurrencias_total, axis=1)
# ocurrencias_total = np.concatenate([np.array(i).reshape(-1, 1) for i in ocurrencias_total], axis=1)
# occ = np.sum(ocurrencias_total, axis=1)

def Pn_teo_bose(t, n, nT, T):
  return (nT/T*t)**n / (1 + nT/T * t)**(n+1)

def Pn_teo_poisson(t, n, nT, T):
  return (nT/T*t)**n / np.math.factorial(n) * np.exp(-nT/T*t)

def error_formula_poisson(a_0, x, n, T):
    # return (n/(a_0*x/T) -1)*np.exp(-a_0*x/T)*(a_0*x/T)**n/np.math.factorial(n)
    return (np.exp(-a_0*x/T)*(a_0*x/T)**n * (n*T-a_0*x))/(T*a_0*np.math.factorial(n))



formula_bose_str0 = '1/(1 + a_0/T * x)'
formula_bose_str1 = '(a_0/T*x) / (1 + a_0/T * x)**2'
formula_bose_str2 = '(a_0/T*x)**2 / (1 + a_0/T * x)**(3)'
def formula_bose(a_0, x):
    return a_0**x/((1+a_0)**(1+x))




n_mean = 3.622 #A CHEQUEAR
n_std = 0.1714406
T = 500 #ns
vent = np.linspace(500, 0, 2500) #ns

franja_error0_bose = Propagacion_errores(
        variables = [('a_0', n_mean), ('T', T)], 
        errores = np.array([n_std, 0]).reshape(-1,1), 
        formula = formula_bose_str0, 
        dominio = vent
        ).fit()[1]
franja_error1_bose = Propagacion_errores(
        variables = [('a_0', n_mean), ('T', T)], 
        errores = np.array([n_std, 0]).reshape(-1,1), 
        formula = formula_bose_str1, 
        dominio = vent
        ).fit()[1]
franja_error2_bose = Propagacion_errores(
        variables = [('a_0', n_mean), ('T', T)], 
        errores = np.array([n_std, 0]).reshape(-1,1), 
        formula = formula_bose_str2, 
        dominio = vent
        ).fit()[1]
franja_error0_poisson = np.sqrt((error_formula_poisson(n_mean, vent, 0, T)*n_std)**2)
franja_error1_poisson = np.sqrt((error_formula_poisson(n_mean, vent, 1, T)*n_std)**2)
franja_error2_poisson = np.sqrt((error_formula_poisson(n_mean, vent, 2, T)*n_std)**2)

# fig, axs = plt.subplots(3, sharex=True)

plt.errorbar(vent, P0_bose_total, yerr=error_frecuencia0+P0_bose_total*0.01, fmt='.',label='Datos', markevery=70, errorevery=70, capsize=2, zorder=5)
plt.plot(vent, Pn_teo_bose(vent, 0, n_mean, T), label='Bose Einstein', color='C1')
plt.plot(vent, Pn_teo_bose(vent, 0, n_mean, T) + franja_error0_bose, '--', color = 'C1')
plt.plot(vent, Pn_teo_bose(vent, 0, n_mean, T) - franja_error0_bose, '--', color = 'C1')
plt.fill_between(vent, Pn_teo_bose(vent, 0, n_mean, T) - franja_error0_bose, Pn_teo_bose(vent, 0, n_mean, T) + franja_error0_bose, facecolor = "gray", alpha = 0.3)
plt.plot(vent, Pn_teo_poisson(vent, 0, n_mean, T), label='Poisson', color='C2')
plt.plot(vent, Pn_teo_poisson(vent, 0, n_mean, T) + franja_error0_poisson, '--', color = 'C2')
plt.plot(vent, Pn_teo_poisson(vent, 0, n_mean, T) - franja_error0_poisson, '--', color = 'C2')
plt.fill_between(vent, Pn_teo_poisson(vent, 0, n_mean, T) - franja_error0_poisson, Pn_teo_poisson(vent, 0, n_mean, T) + franja_error0_poisson, facecolor = "gray", alpha = 0.3)
plt.ylabel('Probablidad de 0-fotones')
plt.xlabel('Tamaño de ventana temporal [ns]')
plt.grid()
plt.legend()
plt.savefig('output\\conteo\\presentacion\\P0-bose.png', dpi=400)
plt.show()

plt.errorbar(vent, P1_bose_total, yerr=error_frecuencia1+P1_bose_total*0.01, fmt='.',label='Datos', markevery=70, errorevery=70, capsize=5, color='C0')
plt.plot(vent, Pn_teo_bose(vent, 1, n_mean, T), label='Bose Einstein', color='C1')
plt.plot(vent, Pn_teo_bose(vent, 1, n_mean, T) + franja_error1_bose, '--', color = 'C1')
plt.plot(vent, Pn_teo_bose(vent, 1, n_mean, T) - franja_error1_bose, '--', color = 'C1')
plt.fill_between(vent, Pn_teo_bose(vent, 1, n_mean, T) - franja_error1_bose, Pn_teo_bose(vent, 1, n_mean, T) + franja_error1_bose, facecolor = "gray", alpha = 0.3)
plt.plot(vent, Pn_teo_poisson(vent, 1, n_mean, T), label='Poisson', color='C2')
plt.plot(vent, Pn_teo_poisson(vent, 1, n_mean, T) + franja_error1_poisson, '--', color = 'C2')
plt.plot(vent, Pn_teo_poisson(vent, 1, n_mean, T) - franja_error1_poisson, '--', color = 'C2')
plt.fill_between(vent, Pn_teo_poisson(vent, 1, n_mean, T) - franja_error1_poisson, Pn_teo_poisson(vent, 1, n_mean, T) + franja_error1_poisson, facecolor = "gray", alpha = 0.3)
plt.ylabel('Probablidad de 1-fotón')
plt.xlabel('Tamaño de ventana temporal [ns]')
plt.grid()
plt.legend()
plt.savefig('output\\conteo\\presentacion\\P1-bose.png', dpi=400)
plt.show()

plt.errorbar(vent, P2_bose_total, yerr=error_frecuencia2+P2_bose_total*0.01, fmt='.',label='Datos', markevery=70, errorevery=70, capsize=5, color='C0')
plt.plot(vent, Pn_teo_bose(vent, 2, n_mean, T), label='Bose Einstein', color='C1')
plt.plot(vent, Pn_teo_bose(vent, 2, n_mean, T) + franja_error2_bose, '--', color = 'C1')
plt.plot(vent, Pn_teo_bose(vent, 2, n_mean, T) - franja_error2_bose, '--', color = 'C1')
plt.fill_between(vent, Pn_teo_bose(vent, 2, n_mean, T) - franja_error2_bose, Pn_teo_bose(vent, 2, n_mean, T) + franja_error2_bose, facecolor = "gray", alpha = 0.3)
plt.plot(vent, Pn_teo_poisson(vent, 2, n_mean, T), label='Poisson', color='C2')
plt.plot(vent, Pn_teo_poisson(vent, 2, n_mean, T) + franja_error2_poisson, '--', color = 'C2')
plt.plot(vent, Pn_teo_poisson(vent, 2, n_mean, T) - franja_error2_poisson, '--', color = 'C2')
plt.fill_between(vent, Pn_teo_poisson(vent, 2, n_mean, T) - franja_error2_poisson, Pn_teo_poisson(vent, 2, n_mean, T) + franja_error2_poisson, facecolor = "gray", alpha = 0.3)
plt.ylabel('Probablidad de 2-fotones')
plt.xlabel('Tamaño de ventana temporal [ns]')
plt.grid()
plt.legend()
plt.savefig('output\\conteo\\presentacion\\P2-bose.png', dpi=400)
plt.show()


########################################
#POISSON
########################################

path_poisson = 'input\\conteo\\poisson(10ms)\\laser_2v_bis_2' #Ventana de 500 ns

umbral_poisson = 3.1 #1
P0_poisson_total = np.zeros(2500)
P1_poisson_total = np.zeros(2500)
P2_poisson_total = np.zeros(2500)
ocurrencias = np.zeros((2500, 1000))
for j, filename in enumerate(os.listdir(path_poisson)):
  data = load_dict(path_poisson + '\\' + filename)
  P0_poisson = np.zeros(2500)
  P1_poisson = np.zeros(2500)
  P2_poisson = np.zeros(2500)
  ocurrencia_tau = np.zeros(2500)
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
    ocurrencia_tau[i] = ocurrencia
    if ocurrencia == 0:
      P0_poisson[i] += 1
    elif ocurrencia == 1:
      P1_poisson[i] += 1 
    elif ocurrencia == 2:
      P2_poisson[i] += 1

  ocurrencias[:,j] = ocurrencia_tau
  
  P0_poisson_total += P0_poisson
  P1_poisson_total += P1_poisson
  P2_poisson_total += P2_poisson 

error_frecuencia0 = np.zeros(2500)
error_frecuencia1 = np.zeros(2500)
error_frecuencia2 = np.zeros(2500)
for i in range(2500):
  cuentas, apariciones = np.unique(ocurrencias[i,:], return_counts=True)
  error_suma_apariciones = np.sqrt(np.sum(apariciones))
  frecuencia = apariciones/np.sum(apariciones)
  try:
    error_frecuencia0[i], error_frecuencia1[i], error_frecuencia2[i], *_ = np.sqrt((np.sqrt(apariciones)/np.sum(apariciones))**2 + 
          (apariciones*error_suma_apariciones/(np.sum(apariciones)**2))**2)
  except:
    try:
      error_frecuencia0[i], error_frecuencia1[i], *_ = np.sqrt((np.sqrt(apariciones)/np.sum(apariciones))**2 + 
        (apariciones*error_suma_apariciones/(np.sum(apariciones)**2))**2)
    except:
      error_frecuencia0[i], *_ = np.sqrt((np.sqrt(apariciones)/np.sum(apariciones))**2 + 
      (apariciones*error_suma_apariciones/(np.sum(apariciones)**2))**2)

P0_poisson_total = P0_poisson_total/1000
P1_poisson_total = P1_poisson_total/1000
P2_poisson_total = P2_poisson_total/1000

n_mean = 0.8021739 #20.06304 
n_std = 0.05725898 #0.9861
T = 100 #ms
vent = np.linspace(100, 0, 2500) #ms

franja_error0_bose = Propagacion_errores(
        variables = [('a_0', n_mean), ('T', T)], 
        errores = np.array([n_std, 0]).reshape(-1,1), 
        formula = formula_bose_str0, 
        dominio = vent
        ).fit()[1]
franja_error1_bose = Propagacion_errores(
        variables = [('a_0', n_mean), ('T', T)], 
        errores = np.array([n_std, 0]).reshape(-1,1), 
        formula = formula_bose_str1, 
        dominio = vent
        ).fit()[1]
franja_error2_bose = Propagacion_errores(
        variables = [('a_0', n_mean), ('T', T)], 
        errores = np.array([n_std, 0]).reshape(-1,1), 
        formula = formula_bose_str2, 
        dominio = vent
        ).fit()[1]
franja_error0_poisson = np.sqrt((error_formula_poisson(n_mean, vent, 0, T)*n_std)**2)
franja_error1_poisson = np.sqrt((error_formula_poisson(n_mean, vent, 1, T)*n_std)**2)
franja_error2_poisson = np.sqrt((error_formula_poisson(n_mean, vent, 2, T)*n_std)**2)



plt.errorbar(vent, P0_poisson_total, yerr=error_frecuencia0+P0_poisson_total*0.01, fmt='.',label='Datos', markevery=70, errorevery=70, capsize=2, zorder=5)
plt.plot(vent, Pn_teo_bose(vent, 0, n_mean, T), label='Bose Einstein', color='C1')
plt.plot(vent, Pn_teo_bose(vent, 0, n_mean, T) + franja_error0_bose, '--', color = 'C1')
plt.plot(vent, Pn_teo_bose(vent, 0, n_mean, T) - franja_error0_bose, '--', color = 'C1')
plt.fill_between(vent, Pn_teo_bose(vent, 0, n_mean, T) - franja_error0_bose, Pn_teo_bose(vent, 0, n_mean, T) + franja_error0_bose, facecolor = "gray", alpha = 0.3)
plt.plot(vent, Pn_teo_poisson(vent, 0, n_mean, T), label='Poisson', color='C2')
plt.plot(vent, Pn_teo_poisson(vent, 0, n_mean, T) + franja_error0_poisson, '--', color = 'C2')
plt.plot(vent, Pn_teo_poisson(vent, 0, n_mean, T) - franja_error0_poisson, '--', color = 'C2')
plt.fill_between(vent, Pn_teo_poisson(vent, 0, n_mean, T) - franja_error0_poisson, Pn_teo_poisson(vent, 0, n_mean, T) + franja_error0_poisson, facecolor = "gray", alpha = 0.3)
plt.ylabel('Probablidad de 0-fotones')
plt.xlabel('Tamaño de ventana temporal [ms]')
plt.grid()
plt.legend()
plt.savefig('output\\conteo\\presentacion\\P0-poisson-umbral-grande.png', dpi=400)
plt.show()

plt.errorbar(vent, P1_poisson_total, yerr=error_frecuencia1+P1_poisson_total*0.01, fmt='.',label='Datos', markevery=70, errorevery=70, capsize=5, color='C0')
plt.plot(vent, Pn_teo_bose(vent, 1, n_mean, T), label='Bose Einstein', color='C1')
plt.plot(vent, Pn_teo_bose(vent, 1, n_mean, T) + franja_error1_bose, '--', color = 'C1')
plt.plot(vent, Pn_teo_bose(vent, 1, n_mean, T) - franja_error1_bose, '--', color = 'C1')
plt.fill_between(vent, Pn_teo_bose(vent, 1, n_mean, T) - franja_error1_bose, Pn_teo_bose(vent, 1, n_mean, T) + franja_error1_bose, facecolor = "gray", alpha = 0.3)
plt.plot(vent, Pn_teo_poisson(vent, 1, n_mean, T), label='Poisson', color='C2')
plt.plot(vent, Pn_teo_poisson(vent, 1, n_mean, T) + franja_error1_poisson, '--', color = 'C2')
plt.plot(vent, Pn_teo_poisson(vent, 1, n_mean, T) - franja_error1_poisson, '--', color = 'C2')
plt.fill_between(vent, Pn_teo_poisson(vent, 1, n_mean, T) - franja_error1_poisson, Pn_teo_poisson(vent, 1, n_mean, T) + franja_error1_poisson, facecolor = "gray", alpha = 0.3)
plt.ylabel('Probablidad de 1-fotón')
plt.xlabel('Tamaño de ventana temporal [ms]')
plt.grid()
plt.legend()
plt.savefig('output\\conteo\\presentacion\\P1-poisson-umbral-grande.png', dpi=400)
plt.show()

plt.errorbar(vent, P2_poisson_total, yerr=error_frecuencia2+P2_bose_total*0.01, fmt='.',label='Datos', markevery=70, errorevery=70, capsize=5, color='C0')
plt.plot(vent, Pn_teo_bose(vent, 2, n_mean, T), label='Bose Einstein', color='C1')
plt.plot(vent, Pn_teo_bose(vent, 2, n_mean, T) + franja_error2_bose, '--', color = 'C1')
plt.plot(vent, Pn_teo_bose(vent, 2, n_mean, T) - franja_error2_bose, '--', color = 'C1')
plt.fill_between(vent, Pn_teo_bose(vent, 2, n_mean, T) - franja_error2_bose, Pn_teo_bose(vent, 2, n_mean, T) + franja_error2_bose, facecolor = "gray", alpha = 0.3)
plt.plot(vent, Pn_teo_poisson(vent, 2, n_mean, T), label='Poisson', color='C2')
plt.plot(vent, Pn_teo_poisson(vent, 2, n_mean, T) + franja_error2_poisson, '--', color = 'C2')
plt.plot(vent, Pn_teo_poisson(vent, 2, n_mean, T) - franja_error2_poisson, '--', color = 'C2')
plt.fill_between(vent, Pn_teo_poisson(vent, 2, n_mean, T) - franja_error2_poisson, Pn_teo_poisson(vent, 2, n_mean, T) + franja_error2_poisson, facecolor = "gray", alpha = 0.3)
plt.ylabel('Probablidad de 2-fotones')
plt.xlabel('Tamaño de ventana temporal [ms]')
plt.grid()
plt.legend()
plt.savefig('output\\conteo\\presentacion\\P2-poisson-umbral-grande.png', dpi=400)
plt.show()

###############################
#GRAFICOS INTRODUCCION PRESENTACION
###############################

x = np.linspace(0, 3, 1000)
y1 = np.sin(2*np.pi*x)
y2 = -y1

plt.plot(x, y1)
plt.plot(x, y2)
plt.xticks([0, 1, 2, 3], ['0', '$T$', '$2T$', '$3T$'])
plt.xlabel('Tiempo')
plt.ylabel('Amplitud')
plt.grid()
plt.savefig('output\\conteo\\presentacion\\grafico-intro-1.png', dpi=400)
plt.show()

x = np.linspace(0, 10, 1000)
y1 = np.sin(2*np.pi*x+1)
y2 = np.sin(2*np.pi*x+x*np.pi/4)
plt.plot(x, y1)
plt.plot(x, y2)
plt.vlines(6.71, -1, 1)
plt.xlabel('Tiempo')
plt.ylabel('Amplitud')
plt.grid()
plt.show()

def formula_poisson(a_0, x):
    funcion = []
    for n in x:
        funcion.append(np.exp(-a_0)*a_0**n/np.math.factorial(n))
    return np.array(funcion)

n = np.linspace(0, 10, 100)
x, y = np.unique(formula_poisson(20, n), return_counts=True)

plt.bar(x, y)
plt.plot()


###############################
#Caracterizacion resistencias-ancho de picos
###############################

path = 'input\\conteo\\ancho_resistencia_picos'

for filename in os.listdir(path):
  data = load_dict(path + '\\' + filename)
  plt.figure(figsize=(6.4*2, 4.8*2))
  V = data['tension']
  t = data['tiempo']
  R = filename.split('_')[2]
  plt.plot(t*1e9, V*1e3, '.-', label='R = '+R+'$\\Omega$', markevery=10)
  plt.xlabel('Tiempo [ns]', fontsize=30)
  plt.ylabel('Voltaje [mV]', fontsize=30)
  plt.xticks(fontsize=25)
  plt.yticks(fontsize=25)
  plt.grid()
  # plt.legend(loc='lower left', prop={'size': 15})
  # plt.savefig(f'output/conteo/ancho_resistencia_{R}.svg')
  plt.show()

#Grafico 50 y 22k juntos

data_50 = load_dict(path + '\\ancho_resistencia_50_ohms.pkl')
data_22k = load_dict(path + '\\ancho_resistencia_22k_ohms.pkl')

V_50 = data_50['tension']
t_50 = data_50['tiempo']

V_22k = data_22k['tension']
t_22k = data_22k['tiempo']

fig, axs = plt.subplots(2,1)

axs[0].plot(t_50*1e9, V_50*1e3, '.-', color='C0', label='R = 50 $\\Omega$', markevery=10)
axs[0].set_xlabel('Tiempo [ns]')
axs[0].set_ylabel('Voltaje [mV]')
axs[0].legend()
axs[0].grid()

axs[1].plot(t_22k*1e6, V_22k*1e3, '.-', color='C1', label='R = 22 k$\\Omega$', markevery=10)
axs[1].set_xlabel('Tiempo [ms]')
axs[1].set_ylabel('Voltaje [mV]')
axs[1].legend()
axs[1].grid()

plt.tight_layout()
plt.savefig('output/conteo/ancho_resistencia_50_22k.svg')
plt.show()

###############################
#Tiempo de coherencia
###############################

path = 'input\\conteo\\correlacion_frecuencia'

t_cs = []
frecs = []
for l in range(20, 65, 5):
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

    t_c = np.round(np.abs(1000*eje_temporal[find_nearest_arg(correlacion_promedio, 0.5)]), 5)
    frecs.append(definir_frecuencia(l/10))
    t_cs.append(t_c)

plt.plot(frecs, t_cs, 'o')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Tiempo de coherencia [ms]')
plt.grid()
plt.savefig('output/conteo/frecuencia-vs-coherencia.svg')
plt.show()