# Importo los paths
from herramientas.config.config_builder import Parser
variables = Parser(configuration = 'espectroscopia').config()
glob_path = os.path.normpath(os.getcwd())
input_path = os.path.join(glob_path + os.path.normpath(variables['input']))
output_path = os.path.join(glob_path + os.path.normpath(variables['output']))

# Importo paquetes
import time, numpy as np, pickle, os
from matplotlib import pyplot as plt

import pyvisa # VISA (Virtual instrument software architecture)

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
osc = rm.open_resource(instrumentos[1])
las = rm.open_resource(instrumentos[2])

# Chequeo que estén bien etiquetados
osc.query('*IDN?')
gen.query('*IDN?')
las.query('*IDN?')

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
# GENERADOR: configuración inicial, asumiendo que usamos sólo el canal 1. Se pu
# ede usar el dos como output
# =============================================================================

# Seteamos la impedancia como high Z
gen.write('OUTPut1:IMPedance INFinity')

# Prendemos el canal 1
gen.write('OUTPut1:STATe on')

# Preguntamos el valor actual de la frecuencia y lo seteamos en 1 Hz. Se pueden hacer barridos
gen.query_ascii_values('FREQ?')
frecuencia = 1 # Hz
gen.write(f'FREQuency {frecuencia}')

# El tipo de señal: SINusoid|SQUare|PULSe|RAMP|StairDown|StairUp|Stair Up&Dwn|Trapezoid|RoundHalf|AbsSine|AbsHalfSine|ClippedSine|ChoppedSine|NegRamp|OscRise|OscDecay|CodedPulse|PosPulse|NegPulse|ExpRise|ExpDecay|Sinc|Tan|Cotan|SquareRoot|X^2|HaverSine|Lorentz|Ln(x)|X^3|CauchyDistr|BesselJ|BesselY|ErrorFunc|Airy|Rectangle|Gauss|Hamming|Hanning|Bartlett|Blackman|Laylight|Triangle|DC|Heart|Round|Chirp|Rhombus|Cardiac          
gen.write('SOURce1:FUNCtion:SHAPe SQUare')

# Seteamos un offset puede ser en mV o V
gen.write('SOURce1:VOLTage:LEVel:IMMediate:OFFSet 0mV')

# Seteamos la amplitud
gen.write('SOURce1:VOLTage:LEVel:IMMediate:AMPLitude .01mVpp')

# =============================================================================
# Las mediciones que hacemos fueron variando escalas tanto horizontales como ve
# rticales. Además se efectuaron cambios sobre la temperatura de la celda del R
# b y se quitaron (o pusieron) los imánes que generaban el campo magnético cons
# tante sobre la muestra de Rb. El láser se mantuvo siempre a la misma temperat
# ura (CHEQUEAR #TODO). CONFIGURACION DEL PIB controlador de T de la celda: P 2
# 50 I 80 D 75.
# =============================================================================

# Sacamos una foto al osciloscopio (ambos canales)
tiempo_1, tension_1 = medir(osc,1)
tiempo_2, tension_2 = medir(osc,2)

# Creamos un diccionario para organizar y documentar los datos
temperatura_celda, name = 27.4, 'medicion_triangular_pitalla'
medicion = {
'unidades':f'-OSC: VERT: CH1: 100mV, CH2: 10mV; HOR: 10ms. -RED: FREQ: 16.35hz; AMPL: 233mV pk2pk. -RB: {temperatura_celda}ºC. -LAS: SETPOINT: 0.9 A .',
'tiempo':tiempo_1,
'tension_1':tension_1,
#'tension_2':tension_2
}

# Los graficamos para ver que estén bien
fig, axs = plt.subplots(nrows = 3, ncols = 1, sharex = True)
axs.flatten()
axs[0].plot(medicion['tiempo'], medicion['tension_1'], label = 'Tensión del canal 1: plot')
axs[0].grid(visible = True)
axs[1].scatter(medicion['tiempo'], medicion['tension_1'], s = 2, label = 'Tensión del canal 1: scatter')
axs[1].grid(visible = True)
#axs[2].scatter(medicion['tiempo'], medicion['tension_2'], s = 2, color = 'black', label = 'Tensión del canal 2')
axs[2].grid(visible = True)
fig.legend()
fig.tight_layout()
fig.show()

# Los guardamos
fname = f'C:/Users/Publico/Desktop/GRUPO 8 COOLS/{name}.pkl'
with open(file = fname, mode = "wb") as archive:
   pickle.dump(obj = medicion, file = archive)     

# Los abrimos para ver que este todo bien
with open(file = fname, mode = "rb") as archive:
   datos = pickle.load(file = archive)   

# Los graficamos con el mismo propósito
fig, axs = plt.subplots(nrows = 3, ncols = 1, sharex = True)
axs.flatten()
axs[0].plot(datos['tiempo_1'], datos['tension_1'], label = 'Tensión del canal 1: plot')
axs[0].grid(visible = True)
axs[1].scatter(datos['tiempo_1'], datos['tension_1'], s = 2, label = 'Tensión del canal 1: scatter')
axs[1].grid(visible = True)
axs[2].scatter(datos['tiempo_2'], datos['tension_2'], s = 2, color = 'black', label = 'Tensión del canal 2')
axs[2].grid(visible = True)
fig.legend()
fig.tight_layout()
fig.show()

