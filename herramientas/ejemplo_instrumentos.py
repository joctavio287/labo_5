import time, numpy as np, pickle, os
from matplotlib import pyplot as plt
# Hay que correr:
# >> pip uninstall visa
# >> pip install pyvisa pyvisa-py pyUSB pySerial libusb PyVICP zeroconf psutil

# import nidaqmx # DAQ (Data Acquisition) 

import pyvisa # VISA (Virtual instrument software architecture)

# from gpib_ctypes import make_default_gpib
# make_default_gpib()


# =============================================================================
# Chequeo los instrumentos que están conectados por USB
# =============================================================================
rm = pyvisa.ResourceManager()
instrumentos = rm.list_resources()  

# =============================================================================
# Printear la variable instrumentos para chequear que están enchufados en el mi
# smo orden. Sino hay que rechequear la numeración de gen y osc.
# =============================================================================

gen = rm.open_resource(instrumentos[0])

osc = rm.open_resource(instrumentos[1])

# =============================================================================
# Si vas a repetir la adquisicion muchas veces sin cambiar la escala es util de
# finir una funcion que mida y haga las cuentas. 'WFMPRE': waveform preamble.
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
# Entonces ahora, cuando quiera ver la pantalla del osciloscopio (siempre y cua
# ndo esté bien seteado) se puede correr las siguientes lineas
# =============================================================================

N = 10 # número de secuencias de la pantalla
for n in range(N):
    tiempo, data = medir(inst = osc, channel = 1)
    plt.figure()
    plt.plot(tiempo, data);
    plt.xlabel('Tiempo [s]');
    plt.ylabel('Tensión [V]');
    time.sleep(1)

# PARA MEDIR EL FILTRO RC
# =============================================================================
# Seteo las frecuencias que quiero [Hz] y tomo mediciones percatándome de ajust
# ar el osciloscopio por c/ medicion. Para escala de tiempos por división:
# 
# HORizontal:MAIn:SCAle <escala>', donde <escala> está en segundos 
# <escala> puede ser: 1, 2.5 ó 5 con un exponente.
# Ej: 2.5E-3 son 2.5 ms
#       5E-6 son 5.0 us
# =============================================================================

channel_1 = []
channel_2 = []

# =============================================================================
# Estoy asumiendo que el generador está en High Z y que alimento con Vp2p.La si
# guiente lista está para fijar la tensión del osciloscopio más adelante, dentr
# o de la función.
# =============================================================================

graduacion_vertical = [50e-3, 100e-3, 200e-3, 500e-3, .1e1, .2e1, .5e1] # Volts

# Seteo la resolucion vertical de los dos canales, asumiendo que ch2 es la de r
# eferencia. Los valores posibles son {50mV, 100mV, 200mV, 500mV, 1V, 2V, 5V} y
# algunos más chicos o grandes que no creo que usemos.

# La escala vertical la seteo como para que en 6 cuadraditos entre las dos cres
# tas de la onda. Para esto creo un diccionario que tiene como values c/graduac
# ión posible de la escala vertical y como keys la distancias de dichas graduac
# iones a la deseada (6 cuadraditos entre las dos crestas). Elijo la graduación
# que tiene menor distancia a la deseada.
# """

for tension in np.arange(1, 4):
    gen.write(f'VOLT {tension}')
    
    time.sleep(1)

    aux = {}
    for grad in graduacion_vertical:
        aux[abs(grad-tension/6)] = grad
    escala = aux[min(aux.keys())]

    osc.write(f'CH{1}:SCAle {escala:3.1E}')    #https://docs.python.org/3/library/string.html; ctrl+f: Type
    osc.write(f'CH{2}:SCAle {escala:3.1E}')    
    time.sleep(1)

    for freq in np.logspace(start = 1, stop = 4, num = 20, base = 10):
        time.sleep(1/freq)
        gen.write(f'FREQ {freq}')
        """
        Seteo la escala temporal como para que entren 4 períodos (T = 1/freq) en el an
        cho completo de la pantalla (que tiene 10 cuadraditos). El valor al cual queda
        fijado la escala es el más cercano a los valores predefinidos.
        """
        escala_t = (4/10)*(1/freq)
        osc.write(f'HORizontal:MAIn:SCAle {escala_t:3.1E}')
        osc.write('MEASUrement:IMMed:SOU CH1; TYPe PK2k')
        channel_1.append(osc.query_ascii_values('MEASUrement:IMMed:VALue?')[0])
        osc.write('MEASUrement:IMMed:SOU CH2; TYPe PK2k')
        channel_2.append(osc.query_ascii_values('MEASUrement:IMMed:VALue?')[0])

# =============================================================================
# Para saber el ID de la placa conectada (DevX):
# =============================================================================

system = nidaqmx.system.System.local()
for device in system.devices:
    print(device)


# =============================================================================
# Para setear (y preguntar) el modo y rango de un canal analógico:
# 'nidaqmx.Task()' is a method to create a task.
#'task.ai_channels' gets the collection of analog input (a.i) channels for this
# task.
#"Dev1/ai1" is the input channel.
# =============================================================================

with nidaqmx.Task() as task:  
    ai_channel = task.ai_channels.add_ai_voltage_chan("Dev1/ai1", max_val = 10, min_val = -10)
    print(ai_channel.ai_term_cfg)  # specifies the terminal configuration for the channel.
    print(ai_channel.ai_max) # it returns the coerced maximum value that the device can measure with the current settings
    print(ai_channel.ai_min) # idem with minimum	
	

# =============================================================================
# Medicion por tiempo/samples de una sola vez:
# 'cfg_samp_clk_timing' sets the source of the Sample Clock, the rate of the Sa
# mple Clock and the number of samples to acquire or generate. Specifies the sa
# mpling rate in samples per channel per second.
# =============================================================================

def medir_daq(duracion, fs):
    cant_puntos = duracion*fs    
    with nidaqmx.Task() as task:
        task.ai_channels.add_ai_voltage_chan(
        physical_channel = "Dev1/ai1", 
        terminal_config = nidaqmx.constants.TerminalConfiguration.DIFFERENTIAL
        )
        
        task.timing.cfg_samp_clk_timing(
        rate = fs,
        samps_per_chan = cant_puntos,
        sample_mode = nidaqmx.constants.AcquisitionType.FINITE
        )

        datos = task.read(number_of_samples_per_channel = nidaqmx.constants.READ_ALL_AVAILABLE)
    datos = np.asarray(datos)    
    return datos


# =============================================================================
# DAQ en modo finito (en el cual ajustamos nosotres las frecuencia de sampleo y
# el tiempo de medición). Creo que va a ser más conveniente que el otro. El chi
# ste es que le ponemos un tiempo largo (10s por ej), le pegas a la barra y dej
# as que oscile. Cuando termine vas a tener una señal larga que va a tener el t
# iempo previo a la pegada que la descartaremos cuando analicemos los datos:
# =============================================================================

# Duración de la medición en segundos
duracion = 10 # si se cambia a lo largo de las mediciones hay que registrarlo para cuando procesemos los datos.

# Frecuencia de muestreo (Hz) del daq
fs = 250000  # tal vez conviene bajarla, dado que  la frecuencia mínima de sampleo es mucho menor y no necesitamos una transformada tan fina

datita = medir_daq(duracion, fs)

plt.figure(1)
plt.plot(np.linspace(0,duracion,len(datita)), datita)
plt.xlabel('Tiempo [s]')
plt.ylabel('Tensión [V]')
plt.show()

# Para guardar la datita no sabría como hacer porque no sé en qué formato viene. Probar type(datita),
# si es una lista, array o algo por el estilo se puede copiar el guardado de arriba. Tal vez, si son 
# demasiados datos, conviene guardar de otra forma que no sea .csv. También esta la opción de bajar la
# frecuencia de muestreo (no descartar, puede ser muy util, sobre todo para tomar mediciones preliminares)

# =============================================================================
# Otra opción: medición continua. La idea es la misma, la diferencia es que cua
# ndo vos frenas la task, va a dejar de medir. Es medio desprolijo, creo que co
# nviene usar lo de arriba. Además, de esta forma, no tenemos con precisión cuá
# l es la escala horizontal.
# =============================================================================

task = nidaqmx.Task()
modo = nidaqmx.constants.TerminalConfiguration.DIFFERENTIAL
task.ai_channels.add_ai_voltage_chan("Dev1/ai1", terminal_config = modo)
task.timing.cfg_samp_clk_timing(fs, sample_mode = nidaqmx.constants.AcquisitionType.CONTINUOUS)
task.start()
t0 = time.time()
total = 0
for i in range(10):
    time.sleep(0.1)
    datos = task.read(number_of_samples_per_channel = nidaqmx.constants.READ_ALL_AVAILABLE)           
    t1 = time.time() # time in seconds since the 'epoch': January 1, 1970, 00:00:00 (UTC)
    total = total + len(datos)
    print("%2.3fs %d %d %2.3f" % (t1-t0, len(datos), total, total/(t1-t0)))    
task.stop()
task.close()

##################################### EJEMPLO FERRO ################################################

# ========================================================================
#                         Funciones auxiliares
# ========================================================================
# Función para integrar señales
def funcion_integradora(x, y, offset = True):
    '''
    Realiza la integral numerica de un set de datos (x, y)
    INPUT: 
    -x--> np.array: el tiempo de la señal (de no tener inventarlo convenientemente)
    -y--> np.array: la señal.
    -offset--> Bool: si la señal está corrida del cero, el offset genera errores. 
    '''
    if offset:
        y -= y.mean()
    T = x.max() - x.min()
    return np.cumsum(y) * (T/len(x))  # HAY UN OFFSET SOLUCIONAR

# Hacemos la conversión de resistencia [Ohms] a Temperatura [C]
def funcion_conversora_temp(t):
    '''
    Conversor para Pt100 platinum resistor (http://www.madur.com/pdf/tools/en/Pt100_en.pdf)
    INPUT:
    t --> np.array: temperatura[Grados Celsius]. 
    OUTPUT:
    r --> np.array: resistencia[Ohms].
     '''
    R_0 = 100 # Ohms; resistencia a 0 grados Celsius
    A = 3.9083e-3 # grados a la menos 1
    B = -5.775e-7 # grados a la menos 2
    C = -4.183e-12 # grados a la menos 4
    return np.piecewise(t, [t < 0, t >= 0], [lambda t: R_0*(1+A*t+B*t**2+C*(t-100)*t**3), lambda t: R_0*(1+A*t+B*t**2)])

# ========================================================================
# Chequeamos los instrumentos que están conectados
# ========================================================================

rm = pyvisa.ResourceManager()
instrumentos = rm.list_resources()

# Hay que chequear el numero de la lista para asignarlo correctamente:
osc = rm.open_resource(instrumentos[0])
mult = rm.open_resource(instrumentos[1])
# ========================================================================
# Vamos a tomar mediciones indirectas de la temperatura del núcleo del 
# transformador, recopilando la impedancia de una resistencia adosada 
# al mismo. Además tomamos mediciones de la tensión del primario y el 
# secundario. 
# ========================================================================

# La graduación vertical (1 cuadradito) puede ser: [50e-3, 100e-3, 200e-3, 500e-3, .1e1, .2e1, .5e1]
# Medimos con diferente escalas los dos canales y estas cambiaron para el experimento con y sin integrador
escala = .2e1
osc.write(f'CH{1}:SCale {escala:3.1E}')
osc.write(f'CH{2}:SCale {escala:3.1E}')

# ========================================================================
# Setemos la escala horizontal teniendo en cuenta que la frecuencia del toma
# (220 V) es de 50 Hz.
# ========================================================================
freq = 50 # Hz
escala_t = 4/(10*2*np.pi*freq) # para que entren cuatro períodos en la pantalla (tiene diez cuadraditos, x eso el 10)
osc.write(f'HORizontal:MAin:SCale {escala_t:3.1E}')

# ========================================================================
# Como el código en Python es secuencial vamos a tener que medir en tiempos
# diferidos las tres magnitudes y después interpolar los datos para 'hacer
# que coincidan'.
# ========================================================================

# Las mediciones se van a efectuar cada 'intervalo_temporal'
intervalo_temporal = 4

# Hay que hacer una iteración para saber cuánto tarda y podamos asignar bien el intervalo temporal
t_auxiliar_1 = time.time()

# ESTA ES UNA ITERACION PELADA
t = time.time()
float(mult.query('MEASURE:FRES?'))
marca_resistencia = t + (time.time() - t)/2
marca_resistencia-t
t = time.time()
medir(osc, 1)
marca_tension_1 = t + (time.time() - t)/2
marca_tension_1 - t
t = time.time()
medir(osc, 2)
marca_tension_2 = t + (time.time() - t)/2
marca_tension_2 - t

t_auxiliar_2 = time.time()

# Actualizamos el valor del intervalo temporal (resto de manera tal que el tiempo del inervalo 
# resultante sea, en verdad, el especificado más arriba):
intervalo_temporal -= t_auxiliar_2 - t_auxiliar_1

# Asumimos que el fenómeno dura 3.5'= 210'', modificar de ser necesario
tiempo_total = 150

# Creo las iteraciones (tiempo_total/intervalo_temporal es el número de pasos que vamos a tomar)
iteraciones = np.arange(0, int(tiempo_total/intervalo_temporal), 1)

# Las tres magnitudes que vamos a medir
resistencia, tension_1, tension_2 = [], [], []

# Las marcas temporales de las mediciones
marca_temporal_resistencia, marca_temporal_tension_1, marca_temporal_tension_2 = [], [], []

# Tomamos la referencia del tiempo inicial
t_0 = time.time()

# Hacemos iteraciones
for i in iteraciones:

    # Medición de resistencia
    t = time.time()
    resistencia.append(float(mult.query('MEASURE:FRES?')))
    # La marca temporal es el promedio entre antes y después de medir
    marca_resistencia = t + (time.time() - t)/2
    # Appendeamos el tiempo respecto al t_0
    marca_temporal_resistencia.append(marca_resistencia-t_0)

    # Medición de tensión en el primario
    t = time.time()
    tension_1.append(medir(osc, 1))
    # La marca temporal es el promedio entre antes y después de medir
    marca_tension_1 = t + (time.time() - t)/2
    # Appendeamos el tiempo respecto al t_0
    marca_temporal_tension_1.append(marca_tension_1 - t_0)

    # Medición de tensión en el secundario
    t = time.time()
    tension_1.append(medir(osc, 2))
    # La marca temporal es el promedio entre antes y después de medir
    marca_tension_2 = t + (time.time() - t)/2
    # Appendeamos el tiempo respecto al t_0
    marca_temporal_tension_2.append(marca_tension_2 - t_0)
    
    # Intervalo temporal entre mediciones
    time.sleep(intervalo_temporal)

# ESTABA MAL# Corregimos desfasajes temporales entre la tensión del primario y el secundario
# for i in iteraciones:
#     # Este es el desfasaje entre que sacamos la captura del primario al secundario
#     desfasaje = marca_temporal_tension_2[i] - marca_temporal_tension_1[i]
#     # ATENCION ESTA INTERPOLACION ES INCORRECTA MOSTRARLE EL DIBU A SOFI
#     # Chequear que lo que viene del osc es arrays, dado el caso contrario cambiarlos
#     tension_2[i] = tension_1[i][0], np.interp(tension_1[i][0], tension_2[i][0] - desfasaje, tension_2[i][1])


# Convertimos en formato numpy los datos
resistencia = np.array(resistencia)
marca_temporal_resistencia,marca_temporal_tension_1,marca_temporal_tension_2 = np.array(marca_temporal_resistencia),np.array(marca_temporal_tension_1),np.array(marca_temporal_tension_2)

# Antes de analizar lo medido, guardamos los datos
datos = {
'resistencia' : resistencia,
'marca_temporal_resistencia' : marca_temporal_resistencia,
'tension_1': tension_1, 
'marca_temporal_tension_1' : marca_temporal_tension_1,
'tension_2': tension_2, 
'marca_temporal_tension_2' : marca_temporal_tension_2
}

# Modificar acorde a dónde guardemos en la compu del labo:
full_path = 'C:/repos/labo_5/output/ferromagnetismo/'

# Cambiar después de cada medición, sino los datos se van a reescribir
numero_de_medicion = 16

# Guardar datos
with open(file = os.path.join(os.path.normpath(full_path) + os.path.normpath('medicion_{}.pkl'.format(numero_de_medicion))), mode = "wb") as archive:
    pickle.dump(obj = datos, file = archive)

# Cargar datos
with open(file = os.path.join(os.path.normpath(full_path) + os.path.normpath('medicion_{}.pkl'.format(numero_de_medicion))), mode = "rb") as archive:
    datos_leidos = pickle.load(file = archive)
