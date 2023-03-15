# Importo paquetes
import time, numpy as np, pickle, os
from matplotlib import pyplot as plt
from scipy.signal import find_peaks
from labos.notificacion_bot import mensaje_tel
from labos.ajuste import Ajuste

import pyvisa # VISA (Virtual instrument software architecture)

def guardar_csv(*data, filename, root='.\\', delimiter=',', header='', rewrite=False):
    '''
    Para guardar un archivo en csv
    '''
    isfile = os.path.isfile(root+filename+'.csv')
    if not rewrite and isfile:
        print('Este archivo ya existe, para sobreescribirlo usar el argumento rewrite')
        return
    try:
        np.savetxt(root+filename+'.csv', np.transpose(np.array([*data])), header=header, delimiter=delimiter)
        if isfile:
            print('ATENCIÓN: SE SOBREESCRIBIÓ EL ARCHIVO')
    except:
        print('Algo falló cuando se guardaba')
    return

def save_dict(fname:str, dic:dict, rewrite: bool = False):
    '''
    Para salvar un diccionario en formato pickle. 
    '''
    isfile = os.path.isfile(fname)
    if isfile:
        if not rewrite:
            print('Este archivo ya existe, para sobreescribirlo usar el argumento rewrite = True.')
            return
    try:
        with open(file = fname, mode = "wb") as archive:
            pickle.dump(file = archive, obj=dic)
        texto = f'Se guardo en: {fname}.'
        if isfile:
            texto += f'Atención: se reescribió el archivo {fname}'
    except:
        print('Algo fallo cuando se guardaba')
    return

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
definir_frecuencia(2)
# =============================================================================
# Chequeo los instrumentos que están conectados por USB
# =============================================================================
rm = pyvisa.ResourceManager()
instrumentos = rm.list_resources()  

# =============================================================================
# Printear la variable instrumentos para chequear que están enchufados en el mi
# smo orden. Sino hay que rechequear la numeración de gen y osc.
# =============================================================================
osc = rm.open_resource(instrumentos[0])

# Chequeo que estén bien etiquetados
osc.query('*IDN?')


# =============================================================================
# OSCLIOSCOPIO: configuro la curva
# =============================================================================

# Modo de transmision: Binario positivo. 
osc.write('DATa:ENC RPB') 

# Un byte de dato. Con RPB son 256 bits.
osc.write('DATa:WIDth 1')

# La curva mandada inicia en el primer dato y termina en el último dato, sin importar el modo de adquisición
osc.write("DATa:STARt 1") 
osc.write("DATa:STOP 2500") 


# Modo de adquisición. Si se usa AVE, después se corre 'ACQuire:NUMAVg <NR1>' ó 'ACQuire:NUMAVg?' (<Nr1>: 4,16,64,128)
osc.write("ACQuire:MODe SAMP") # Puede ser PEAKdetect o AVErage 

# =============================================================================
# OSCLIOSCOPIO: seteo los canales
# =============================================================================

# La escala vertical admite: 2e-3,5e-3, 10e-3, 20e-3, 50e-3, 100e-3, 200e-3, 500e-3, .1e1, .2e1, .5e1. 
escala_CH1, escala_CH2 = .5e1,.5e1
cero_CH1, cero_CH2 = 0, 0 
osc.write(f"CH1:SCAle {escala_CH1}")
osc.write(f"CH2:SCAle {escala_CH2}")
osc.write(f"CH1:POSition {cero_CH1}")
osc.write(f"CH2:POSition {cero_CH2}")

# La escala horizontal admite varios valores, pero sea cual sea que setees pone el valor más cercano
# periodo = 1/2*np.pi*freq
# escala_t = numero_de_periodos_en_pantalla/(10*periodo) #hay 10 divs horizontales
escala_t, cero_t = .5e-3, 0
osc.write(f"HORizontal:SCAle {escala_t}")
osc.write(f"HORizontal:POSition {cero_t}")

# =============================================================================
# OSCLIOSCOPIO: si vas a repetir la adquisicion muchas veces sin cambiar la esc
# ala es util definir una funcion que mida y haga las cuentas. 'WFMPRE': wavefo
# rm preamble.
# =============================================================================

# Función para sacarle una foto a la pantalla del osciloscopio por canal
def medir(inst, channel:int = 1):
    """
    Adquiere los datos del canal canal especificado.
    WFMPRE:XZEro? query the time of first data point in waveform
          :XINcr? query the horizontal sampling interval
          :YZEro? query the waveform conversion factor
          :YMUlt? query the vertical scale factor
          :YOFf?  query the vertical position
    INPUT: 
    -inst:objeto de pyvisa: el nombre de la instancia.
    -channel:int: canal del osciloscopio que se quiere medir.
    
    OUTPUT:
    Tupla de dos arrays de numpy: el tiempo y la tensión.
    """
    if channel:
        inst.write('DATa:SOUrce CH' + str(channel)) 
    xze, xin, yze, ymu, yoff = inst.query_ascii_values('WFMPRE:XZE?;XIN?;YZE?;YMU?;YOFF?', separator = ';')
    datos = inst.query_binary_values('CURV?', datatype = 'B', container = np.array)
    data = (datos - yoff)*ymu + yze
    tiempo = xze + np.arange(len(data)) * xin
    return tiempo, data

# =============================================================================
# Encontrar el umbral
# =============================================================================
numero_de_mediciones = 1000
tension_fuente = 2
# carpeta = 'ruido_2v_bis'
carpeta = 'laser_2v_bis_3'
os.mkdir(f'C:/GRUPO 8/datos/poisson(10ms)/{carpeta}')
umbral =  0
t_0 = time.time()
for i in range(numero_de_mediciones):
    print(f'Así de ansiosos estamos: {i+1}/{numero_de_mediciones}')
    # Sacamos una foto al osciloscopio (ambos canales)
    tiempo, tension = medir(osc, 1)
    
    # Sacamos los indices   
    indice_picos = find_peaks(-tension, height = umbral)[0]
    indice_picos = indice_picos[tension[indice_picos]<0]
    tiempo_picos = tiempo[indice_picos]
    tension_picos = tension[indice_picos]
    
    # Creamos un diccionario para organizar y documentar los datos
    fname = f'C:/GRUPO 8/datos/poisson(10ms)/{carpeta}/medicion_{i}.pkl'
    
    tension_PMT, escala_v, escala_h, angulo_polarizador_1, angulo_polarizador_2, resistencia = '900 V', '10 mV', '10 ms', '32 º', '270 º', '1k Ohms'
    medicion = {
    'unidades': f'-Numero de mediciones {numero_de_mediciones} y umbral en {umbral} V.-RES: {resistencia} -OSC: VERT: CH1: {escala_v}; HOR: {escala_h}. -PMT: {tension_PMT}. -Polarizador1: {angulo_polarizador_1}. -Polarizador2: {angulo_polarizador_2}',
    'tiempo': tiempo,
    'tension': tension,
    'indice_picos':indice_picos,
    'tiempo_picos':tiempo_picos,
    'tension_picos':tension_picos,
    'tension_fuente': tension_fuente
     }
    
    # Graficar un dato
    # plt.figure()
    # plt.plot(medicion['tiempo'], medicion['tension'], color = 'red')
    # # plt.scatter(medicion['tiempo_picos'], medicion['tension_picos'])
    # plt.grid(visible = True)
    # plt.xlabel('Tiempo [s]')
    # plt.ylabel('Tensión [V]')
    # plt.show(block = False)

    # Los guardamos
    save_dict(fname = fname, dic = medicion)
tardada = time.time()-t_0
print(tardada)

# Mandamos un mensaje canchero al grupo
mensaje_tel(
    api_token = '5448153732:AAGhKraJQquEqMfpD3cb4rnTcrKB6U1ViMA',
    chat_id = '-693150998',
    mensaje = f'JUEGA EL ROOOJJJO. Tardo {tardada/60}'
    )

# Histograma: poner aproximadamente este valor en el paso float(escala_v.split(' mV')[0])

paso = .5*10/256
paso_ruido = 0.0006 # 5002V .02/256
maximo_altura = 2.5
# maximo_altura_ruido  = 
bins = np.arange(-maximo_altura, 0, paso)

# bins_ruido = np.arange(-maximo_altura, maximo_altura + paso_ruido, paso_ruido)
# Laser
tensiones = []
for i, f in enumerate(os.listdir(f'C:/GRUPO 8/datos/poisson(10ms)/{carpeta}/')):
    medicion = load_dict(fname = f'C:/GRUPO 8/datos/poisson(10ms)/{carpeta}/' + f)
    tensiones.append(medicion['tension_picos'].reshape(-1,1))
tensiones = np.concatenate(tensiones, axis = 0)

# Ruido
tensiones_ruido = []
for i, f in enumerate(os.listdir('C:/GRUPO 8/datos/poisson(10ms)/ruido_2v/')):
    medicion = load_dict(fname = 'C:/GRUPO 8/datos/poisson(10ms)/ruido_2v/' + f)
    tensiones_ruido.append(medicion['tension_picos'].reshape(-1,1))
tensiones_ruido = np.concatenate(tensiones_ruido, axis = 0)

# Ruido oscuro
# tensiones_ruido_oscuro = []
# for i, f in enumerate(os.listdir('C:/GRUPO 8/datos/poisson(10ms)/ruido_oscuro_2v/')):
#     medicion = load_dict(fname = 'C:/GRUPO 8/datos/poisson(10ms)/ruido_oscuro_2v/' + f)
#     tensiones_ruido_oscuro.append(medicion['tension_picos'].reshape(-1,1))
# tensiones_ruido_oscuro = np.concatenate(tensiones_ruido_oscuro, axis = 0)


plt.figure()
plt.hist(tensiones_ruido,
          bins = bins,
          label = "Ruido",
          histtype = "step", 
          color = "green")

# plt.hist(tensiones_ruido_oscuro,
#           bins = bins,
#           label = "Ruido oscuro",
#           histtype = "step", 
#           color = "black")      

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

# =============================================================================
# Habiendo encontrado el umbral, hacemos estadistica
# =============================================================================
numero_de_mediciones = 1000
carpeta = 'laser_2v'
os.mkdir(f'C:/GRUPO 8/poisson(100ms)/{carpeta}')
umbral =  0.062
t_0 = time.time()
for i in range(numero_de_mediciones):
    print(f'Así de ansiosos estamos: {i+1}/{numero_de_mediciones}')
    if i == numero_de_mediciones//2:
        print('VAMO ROJO LOCOOO DALEEE QUE YA VA POR LA MITAD')
        
    # Sacamos una foto al osciloscopio (ambos canales)
    tiempo, tension = medir(osc, 1)
    
    # Sacamos los indices   
    indice_picos = find_peaks(-tension, height = umbral)[0]
    tiempo_picos = tiempo[indice_picos]
    tension_picos = tension[indice_picos]
    
    
    # Creamos un diccionario para organizar y documentar los datos
    fname = f'C:/GRUPO 8/{carpeta}/medicion_{i}.pkl'

    tension_PMT, escala_v, escala_h, angulo_polarizador, resistencia = '900 V', '5 mV', '100 ms', '0 º', '50 Ohms'
    medicion = {
    'unidades': f'-Numero de mediciones {numero_de_mediciones} y umbral en {umbral} V.-RES: {resistencia} -OSC: VERT: CH1: {escala_v}; HOR: {escala_h}. -PMT: {tension_PMT}. -Polarizador: {angulo_polarizador}',
    'tiempo': tiempo,
    'tension': tension,
    'indice_picos':indice_picos,
    'tiempo_picos':tiempo_picos,
    'tension_picos':tension_picos,
    'tension_fuente': tension_fuente
     }
        
    
    # Graficar un dato
    plt.figure()
    plt.plot(medicion['tiempo'], medicion['tension'], '-o', color = 'red')
    plt.scatter(medicion['tiempo_picos'], medicion['tension_picos'])
    plt.grid(visible = True)
    plt.xlabel('Tiempo [s]')
    plt.ylabel('Tensión [V]')
    plt.show(block = False)

    # Los guardamos
    save_dict(fname = fname, dic = medicion)
tardada = time.time()-t_0
print(tardada)
# Mandamos un mensaje canchero al grupo
mensaje_tel(
    api_token = '5448153732:AAGhKraJQquEqMfpD3cb4rnTcrKB6U1ViMA',
    chat_id = '-693150998',
    mensaje = f'JUEGA BOOCA. Tardo {tardada/60}'
    )


# Histograma: poner aproximadamente este valor en el paso float(escala_v.split(' mV')[0])/256
# paso = 0.00020
# maximo_altura = 0.015
# bins = np.arange(-maximo_altura, maximo_altura + paso, paso)

# Laser
carpeta = 'laser_2v'
umbral  = 0.062
ocurrencias = []
for i, f in enumerate(os.listdir(f'C:/GRUPO 8/poisson(10ms)/{carpeta}/')):
    medicion = load_dict(fname = f'C:/GRUPO 8/poisson(10ms)/{carpeta}/' + f)
    
    # Sacamos los indices   
    indice_picos = find_peaks(-medicion['tension'], height = umbral, distance = 50)[0]
    tiempo_picos = medicion['tiempo'][indice_picos]
    tension_picos = medicion['tension'][indice_picos]
    plt.figure()
    plt.scatter(medicion['tiempo'], medicion['tension'], s =2)
    plt.scatter(medicion['tiempo'][indice_picos], medicion['tension'][indice_picos])
    plt.show()
    
    # ocurrencia = len(medicion['tension_picos'][medicion['tension_picos']<-umbral])
    ocurrencia = len(tension_picos[tension_picos<-umbral])
    ocurrencias.append(ocurrencia)
ocurrencias = np.array(ocurrencias)
cuentas, frecuencia = np.unique(ocurrencias, return_counts = True)
frecuencia = frecuencia/np.sum(frecuencia)

plt.figure()

plt.bar(
    cuentas, 
    frecuencia,
    # label = medicion['escala_v'],
    color = "blue")
plt.legend()
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

# ======================================
# Correlación para distintas frecuencias
# ======================================
tension_fuente = 2
carpeta = 'correlacion_20e-1v_0_ohms'
os.mkdir(f'C:/GRUPO 8/correlacion_frecuencia/{carpeta}')
numero_de_mediciones = 11
# i=0
# for i in range(numero_de_mediciones):
i +=1
print(i)
fname = f'C:/GRUPO 8/correlacion_frecuencia/{carpeta}/medicion_{i}.pkl'

tiempo, tension = medir(osc, channel = 1)

# Restamos el offset a la tensión para que el integrando oscile entre positivos y negativos
tension = tension - tension.mean()

# Autocorrelación desde la mitad
correlacion = np.correlate(tension, tension, 'same')

# Diferencias temporales
diferencia_temporal = tiempo[1]-tiempo[0]
eje_temporal = np.arange(-1250*diferencia_temporal, 1250*diferencia_temporal, diferencia_temporal)

tension_PMT, escala_v, escala_h, angulo_polarizador, resistencia  = '900 V', '5 V', '1 ms', '315 º', '0 Ohms'
# print(definir_frecuencia(tension_fuente)
medicion = {
 'unidades': f'-Numero de mediciones {numero_de_mediciones}, la tensión de la fuente está en V.-RES: {resistencia} -OSC: VERT: CH1: {escala_v}; HOR: {escala_h}. -PMT: {tension_PMT}. -Polarizador: {angulo_polarizador}.',
'tension_fuente': tension_fuente,
'tiempo': tiempo,
'tension': tension,
'correlacion': correlacion,
'diferencia_temporal': diferencia_temporal,
'eje_temporal': eje_temporal
 }

save_dict(fname = fname, dic = medicion)

# Levantamos la curva promediando
correlacion_promedio = np.mean([load_dict(f'C:/GRUPO 8/correlacion_frecuencia/{carpeta}/{i}')['correlacion'] for i in os.listdir(f'C:/GRUPO 8/correlacion_frecuencia/{carpeta}')], axis = 0)
diferencia_temporal = load_dict(fname)['diferencia_temporal']
eje_temporal = np.arange(-1250*diferencia_temporal, 1250*diferencia_temporal, diferencia_temporal)
plt.figure()   
plt.plot(eje_temporal, correlacion_promedio)
plt.grid(visible = True)
plt.show()






















