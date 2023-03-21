import os, matplotlib.pyplot as plt, numpy as np
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

def formula_poisson(a_0, x):
    funcion = []
    for n in x:
        funcion.append(np.exp(-a_0)*a_0**n/np.math.factorial(n))
    return np.array(funcion)

def error_formula_poisson(a_0, x):
    funcion = []
    for n in x:
        # funcion.append((n/a_0 -1)*np.exp(-a_0)*a_0**n/np.math.factorial(n))
        funcion.append((a_0**(-1 + n)*np.exp(-a_0)*(-a_0 + n))/(np.math.factorial(n)))
    return np.array(funcion)

formula_bose_str = 'a_0**x/((1+a_0)**(1+x))'
def formula_bose(a_0, x):
    return a_0**x/((1+a_0)**(1+x))

# ====================
# Distribución Poisson
# ====================

# Definición del umbral mediante método visual

# Laser
tensiones = []
carpeta = '/poisson(10ms)/laser_2v_bis_2/'
for f in os.listdir(os.path.join(input_path + carpeta)):
    medicion = load_dict(fname = os.path.join(input_path + carpeta + f))
    for t in medicion['tension_picos']:
        tensiones.append(t)

tensiones, cuentas = np.unique(tensiones, return_counts = True)

# Ruido
tensiones_ruido = []
carpeta = '/poisson(10ms)/ruido_2v/'
for f in os.listdir(os.path.join(input_path + carpeta)):
    medicion = load_dict(fname = os.path.join(input_path + carpeta + f))
    for t in medicion['tension_picos']:
        tensiones_ruido.append(t)
tensiones_ruido, cuentas_ruido = np.unique(tensiones_ruido, return_counts = True)

# Graficamos definiendo un umbral
umbral1 = 1
umbral2 = 3.1
plt.figure()

# plt.vlines(x = -umbral1, ymin = 0, ymax =100000, color = 'C2', label = 'Umbral', linestyles='dashed')
plt.vlines(x = -umbral2, ymin = 0, ymax =100000, color = 'C2', label = 'Umbral', linestyles='dashed')
plt.bar(tensiones,
        cuentas,
        color = 'C1',
        width= .05,
        alpha = .75,
        label = 'Laser incidiendo sobre FM')
plt.bar(tensiones_ruido,
        cuentas_ruido,
        color = 'C0',
        width= .05,
        alpha = 1,
        label = 'Ruido')
# plt.ylim(0,cuentas_ruido.max())
plt.xlabel('Tensión [V]')
plt.ylabel('Número de eventos')
plt.grid(visible = True, alpha=0.3)
plt.yscale('log')
plt.legend()
# plt.show(block = False)
# plt.savefig(os.path.join(output_path + os.path.normpath('/presentacion/umbral_chico.png')))
# plt.savefig(os.path.join(output_path + os.path.normpath('/presentacion/umbral_grande.png')))

# Levantamos la estadística habiendo fijado el umbral
umbral = umbral1
# umbral = umbral2

carpeta = '/poisson(10ms)/laser_2v_bis_2/'
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

# Hacemos un ajuste
cuentas, apariciones = np.unique(ocurrencias, return_counts = True)
error_suma_apariciones = np.sqrt(np.sum(apariciones))
frecuencia = apariciones/np.sum(apariciones)
error_frecuencia = np.sqrt((np.sqrt(apariciones)/np.sum(apariciones))**2 +
        (apariciones*error_suma_apariciones/(np.sum(apariciones)**2))**2
        )

# Valor medio y error
media = np.array([cuenta*frec for cuenta, frec in zip(cuentas, frecuencia)]).sum()
media_std = np.sqrt(np.sum([(cuenta*e_frec)**2 for cuenta, e_frec in zip(cuentas, error_frecuencia)]))

franja_error = np.sqrt((error_formula_poisson(media, cuentas)*media_std)**2)

# Graficamos
plt.figure()
# Datos
plt.bar(cuentas,
        frecuencia,
        color = 'C0',
        width = .5,
        alpha = 1)
plt.errorbar(cuentas, frecuencia, yerr = error_frecuencia, fmt = 'o', capsize = 1.5, color = 'C1', label = 'Experimental')

# Ajuste
plt.plot(cuentas, formula_poisson(media, cuentas), color = 'C2')
plt.errorbar(cuentas, formula_poisson(media, cuentas), yerr = franja_error, marker = 'o', capsize = 1.5, color = 'C2', label = 'Ajuste')
plt.xticks(cuentas)
plt.xlabel('Número de fotones')
plt.ylabel('Probabilidad')
plt.grid(visible = True, alpha=0.3)
plt.legend()
plt.show(block = False)
# plt.savefig(os.path.join(output_path + os.path.normpath('/presentacion/distribucion_poisson_umbral_chico.png')))
# plt.savefig(os.path.join(output_path + os.path.normpath('/presentacion/distribucion_poisson_umbral_grande.png')))

# Coeficiente de determinación 1 - sigma_r**2/sigma_y**2
sigma_r = frecuencia - formula_bose(media, cuentas)
ss_res = np.sum(a = sigma_r**2)
ss_tot = np.sum(a = (frecuencia - np.mean(frecuencia))**2)
R2 = 1 - (ss_res / ss_tot)

# Chi^2
chi_2 = np.sum(((frecuencia - formula_bose(media, cuentas))/error_frecuencia)**2)
expected_chi_2 = len(frecuencia) - 1
variance_chi_2 = np.sqrt(2*expected_chi_2)
reduced_chi_2 = -chi_2/(len(frecuencia)-1)

# ==================
# Distribución Bose
# ==================

# Definición del umbral mediante método visual

# Laser
tensiones = []
carpeta = '/bose(50ns)/laser_2v/'
for f in os.listdir(os.path.join(input_path + carpeta)):
    medicion = load_dict(fname = os.path.join(input_path + carpeta + f))
    for t in medicion['tension_picos']:
        tensiones.append(t)

tensiones, cuentas = np.unique(tensiones, return_counts = True)

# Ruido
tensiones_ruido = []
carpeta = '/bose(50ns)/ruido_2v/'
for f in os.listdir(os.path.join(input_path + carpeta)):
    medicion = load_dict(fname = os.path.join(input_path + carpeta + f))
    medicion['unidades'] = 'Reescribimos el umbral en 0.' +medicion['unidades']
    medicion['indice_picos'] = find_peaks(-medicion['tension'], height = 0)[0]#, distance = 50)[0]
    medicion['indice_picos'] = medicion['indice_picos'][medicion['tension'][medicion['indice_picos']]<0]
    medicion['tiempo_picos'] = medicion['tiempo'][medicion['indice_picos']]
    medicion['tension_picos'] = medicion['tension'][medicion['indice_picos']]
    # save_dict(os.path.join(input_path + carpeta + f), medicion, rewrite = True)
    for t in medicion['tension_picos']:
        tensiones_ruido.append(t)
tensiones_ruido, cuentas_ruido = np.unique(tensiones_ruido, return_counts = True)


# Graficamos definiendo un umbral
umbral = 0.003 #tensiones_ruido.mean() - 3*np.std(tensiones_ruido)

plt.figure()
plt.vlines(x = -umbral, ymin = 0, ymax =1000, label = f'Umbral:{-umbral} V', color = 'C2', linestyles='dashed')
plt.bar(tensiones,
        cuentas,
        color = 'C1',
        width =.00015,
        alpha = 1,
        label = 'Laser incidiendo sobre FM')
plt.bar(tensiones_ruido,
        cuentas_ruido,
        color = 'C0',
        width =.00015,
        alpha = 1,
        label = 'Ruido')
# plt.ylim(0,cuentas_ruido.max())
plt.xlabel('Tensión [V]')
plt.ylabel('Número de eventos')
plt.grid(visible = True, alpha=0.3)
plt.yscale('log')
plt.legend()
plt.show(block = False)

# Levantamos la estadística habiendo fijado el umbral
carpeta = '/bose(50ns)/laser_2v/'
ocurrencias = []
for f in os.listdir(os.path.join(input_path + carpeta)):
    medicion = load_dict(fname = os.path.join(input_path + carpeta + f))

    # Sacamos los indices
    indice_picos = find_peaks(-medicion['tension'], height = umbral, distance = 10)[0]
    tension_picos = medicion['tension'][indice_picos]
    ocurrencia = len(tension_picos[tension_picos<-umbral])
    ocurrencias.append(ocurrencia)

# Hacemos un ajuste
cuentas, apariciones = np.unique(ocurrencias, return_counts = True)
error_suma_apariciones = np.sqrt(np.sum(apariciones))
frecuencia = apariciones/np.sum(apariciones)
error_frecuencia = np.sqrt((np.sqrt(apariciones)/np.sum(apariciones))**2 +
        (apariciones*error_suma_apariciones/(np.sum(apariciones)**2))**2
        )

# Valor medio y error
media = np.array([cuenta*frec for cuenta, frec in zip(cuentas, frecuencia)]).sum()
media_std = np.sqrt(np.sum([(cuenta*e_frec)**2 for cuenta, e_frec in zip(cuentas, error_frecuencia)]))

franja_error = Propagacion_errores(
        variables = [('a_0', media)],
        errores = np.array([media_std]).reshape(-1,1),
        formula = formula_bose_str,
        dominio = cuentas
        ).fit()[1]

# Graficamos
plt.figure()
# Datos
plt.bar(cuentas,
        frecuencia,
        color = 'C0',
        width = .5,
        alpha = 1)
plt.errorbar(cuentas, frecuencia, yerr = error_frecuencia + frecuencia*0.01, fmt = 'o', capsize = 1.5, color = 'C1', label = 'Experimental')

# Ajuste
plt.plot(cuentas, formula_bose(media, cuentas), color = 'C2')
plt.errorbar(cuentas, formula_bose(media, cuentas), yerr = franja_error, marker = 'o', capsize = 1.5, color = 'C2', label = 'Ajuste')

plt.xticks(cuentas[:,5])
plt.xlabel('Número de fotones')
plt.ylabel('Probabilidad')
plt.grid(visible = True, alpha=0.3)
plt.legend()
plt.show(block = False)
# plt.savefig(os.path.join(output_path + os.path.normpath('/presentacion/distribucion_bose.png')))


# Coeficiente de determinación 1 - sigma_r**2/sigma_y**2
sigma_r = frecuencia - formula_bose(media, cuentas)
ss_res = np.sum(a = sigma_r**2)
ss_tot = np.sum(a = (frecuencia - np.mean(frecuencia))**2)
R2 = 1 - (ss_res / ss_tot)

# Chi^2
chi_2 = np.sum(((frecuencia - formula_bose(media, cuentas))/error_frecuencia)**2)
expected_chi_2 = len(frecuencia) - 1
variance_chi_2 = np.sqrt(2*expected_chi_2)
reduced_chi_2 = -chi_2/(len(frecuencia)-1)

# ====================
# Tiempo de coherencia
# ====================

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

    t_c = np.round(np.abs(1000*eje_temporal[find_nearest_arg(correlacion_promedio, 0.5)]), 5)

    plt.plot(eje_temporal*1000, correlacion_promedio, color = color[l],
                label = f'Frecuencia = {np.round(definir_frecuencia(l/10),2)} Hz y ' +r'$\tau_c$' + f' = {t_c} ms')
    if l ==20:
        plt.hlines(y = .5, xmin = 0, xmax = t_c, color = color[l], linestyles = 'dashdot')
    plt.xlim(0,5)
plt.ylabel('Correlación')
plt.xlabel(r'$\Delta \tau$ [ms]')
plt.grid(visible = True)
plt.legend()
plt.show(block = False)
# plt.savefig(os.path.join(output_path + os.path.normpath('/presentacion/coherencia_temporal.png')))

#==============================================
# Probabilidad de medir 0, 1 y 2 fotones: Bosse
#==============================================
path_bose = os.path.join(input_path + '\\bose(50ns)\\laser_2v') #Ventana de 500 ns
umbral_bose = 0.003

# Defino las probabilidades
ocurrencias = np.zeros((2500, 1000))

for j, filename in enumerate(os.listdir(path_bose)):
  data = load_dict(path_bose + '\\' + filename)
  for i in range(2500):
    indice_picos = find_peaks(-data['tension'][i:], height = umbral_bose)[0]
    tension_picos = data['tension'][i:][indice_picos]
    ocurrencia = len(tension_picos[tension_picos<-umbral_bose])
    ocurrencias[i,j] = ocurrencia

P0_bose = np.sum(ocurrencias == 0, axis = 1)
P1_bose = np.sum(ocurrencias == 1, axis = 1)
P2_bose = np.sum(ocurrencias == 2, axis = 1)
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

# ==========================================================================
# Grafico umbrales de detección con señal para Bose y sin señal para Poisson
# ==========================================================================

# Bose

# Levanto los datos del ruido y del laser
tensiones_ruido = []
carpeta = os.path.normpath('/bose(50ns)/ruido_2v/')
for f in os.listdir(os.path.join(input_path + carpeta)):
    medicion = load_dict(fname = os.path.join(input_path + carpeta + '/' + f))
    for t in medicion['tension_picos']:
        tensiones_ruido.append(t)
tensiones_ruido, cuentas_ruido = np.unique(tensiones_ruido, return_counts = True)

tensiones = []
carpeta = os.path.normpath('/bose(50ns)/laser_2v/')
for f in os.listdir(os.path.join(input_path + carpeta)):
    medicion = load_dict(fname = os.path.join(input_path + carpeta + '/' + f))
    for t in medicion['tension_picos']:
        tensiones.append(t)
tensiones, cuentas = np.unique(tensiones, return_counts = True)

# Fijo el umbral y grafico
umbral = -0.003

fig, axs = plt.subplots(nrows = 2, ncols = 1, sharex=True)

axs[0].plot(medicion['tension']*1e3, 
            medicion['tiempo']*1e6, 
            linewidth=0.7, 
            label='Señal osciloscopio', 
            color='C0')
axs[0].vlines(umbral*1e3, 
              ymin =  -1, 
              ymax = 1, 
              linestyles='dashed', 
              colors='C2', 
              label='Umbral: -3 mV')
# ylim = axs[0].get_ylim()
axs[0].set_ylim((medicion['tiempo']*1e6)[0], (medicion['tiempo']*1e6)[-1])
axs[0].legend(loc='upper left')
axs[0].set_ylabel('Tiempo [$\mu$s]')
axs[0].grid(visible = True, alpha=0.3)

axs[1].bar(tensiones*1e3,
        cuentas,
        color = 'C1', 
        width= .15,
        alpha = .75,
        label = 'Laser incidiendo sobre FM')
axs[1].bar(tensiones_ruido*1e3,
        cuentas_ruido,
        color = 'C0', 
        width= .15,
        alpha = 1,
        label = 'Ruido')
axs[1].vlines(umbral*1e3, 
              ymin = -10, 
              ymax = 10e5, 
              linestyles='dashed', 
              colors='C2')
axs[1].legend(loc='upper left')
axs[1].set_yscale('log')
axs[1].set_xlabel('Tensión [mV]')
axs[1].set_ylabel('Número de eventos')
axs[1].grid(visible = True, alpha=0.3)
axs[1].set_ylim(0, cuentas_ruido.max())
# fig.savefig(os.path.join(output_path + os.path.normpath('/umbral_bose.svg')))
fig.show(block = False)

# Poisson

# Levanto los datos del ruido y del laser
tensiones_ruido = []
carpeta = os.path.normpath('/poisson(10ms)/ruido_2v/')
for f in os.listdir(os.path.join(input_path + carpeta)):
    medicion = load_dict(fname = os.path.join(input_path + carpeta + '/' + f))
    for t in medicion['tension_picos']:
        tensiones_ruido.append(t)
tensiones_ruido, cuentas_ruido = np.unique(tensiones_ruido, return_counts = True)

tensiones = []
carpeta = os.path.normpath('/poisson(10ms)/laser_2v_bis_2/')
for f in os.listdir(os.path.join(input_path + carpeta)):
    medicion = load_dict(fname = os.path.join(input_path + carpeta + '/' + f))
    for t in medicion['tension_picos']:
        tensiones.append(t)
tensiones, cuentas = np.unique(tensiones, return_counts = True)

# Fijo el umbral y grafico
umbral = -0.003

plt.figure()
plt.plot()

plt.bar(tensiones,
        cuentas,
        color = 'C1', 
        width= .045,
        alpha = 1,
        label = 'Laser incidiendo sobre FM',
        zorder = 0,
        # edgecolor = 'black'
        )
plt.bar(tensiones_ruido,
        cuentas_ruido,
        color = 'C0', 
        width= .045,
        alpha = 1,
        label = 'Ruido',
        zorder = 1
        )
plt.vlines(-1, 
              ymin = -10, 
              ymax = 10e5, 
              linestyles='dashed', 
              colors='C2',
              label='Umbral: -1 V')
plt.vlines(-3.1, 
              ymin =  -10, 
              ymax = 10e5, 
              linestyles='dashed', 
              colors='C3', 
              label='Umbral: -3,1 V')
plt.legend(loc='upper left')
plt.yscale('log')
plt.xlabel('Tensión [V]')
plt.ylabel('Número de eventos')
plt.grid(visible = True, alpha=0.3)
plt.ylim(0, cuentas_ruido.max())
# plt.savefig(os.path.join(output_path + os.path.normpath('/umbral_poisson.svg')))
plt.show(block = False)
