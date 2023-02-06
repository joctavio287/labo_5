# # Importo los paths
# from herramientas.config.config_builder import Parser
# glob_path = os.path.normpath(os.getcwd())
# variables = Parser(configuration = 'espectroscopia').config()

# Importo paquetes
import time, numpy as np, pickle, os
from matplotlib import pyplot as plt

# # Hay que correr:
# # >> pip uninstall visa
# # >> pip install pyvisa pyvisa-py pyUSB pySerial libusb PyVICP zeroconf psutil
# # import nidaqmx # DAQ (Data Acquisition) 

# # Esto es por si falta el paquete para entrada gpib 
# from gpib_ctypes import make_default_gpib
# linux-gpib in linux
# make_default_gpib()

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
gen = rm.open_resource(instrumentos[1])
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
escala_t, cero_t = 2.5e-6,0
osc.write("HORizontal:SCAle {escala_t}")
osc.write("HORizontal:POSition {cero_t}")

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

# # Ajustamos la fase (por defecto está en radianes)
# gen.write('SOURce1:PHASe:ADJust MINimum')

# # Sincronizamos las fases de los dos outputs
# gen.write('SOURce1:PHASe:INITiate')

# Seteamos un offset puede ser en mV o V
gen.write('SOURce1:VOLTage:LEVel:IMMediate:OFFSet 0mV')

# Seteamos la amplitud
gen.write('SOURce1:VOLTage:LEVel:IMMediate:AMPLitude .01mVpp')


# =============================================================================
# LASER: configuración inicial. Hay distintas formas de usar el instrumento, po
# demos pedirle una dada configuración (medir corriente, temperatura, etc.), in
# iciar la medición y buscar el resultado que se guardo en memoria; podemos dar
# le una configuración y leer (que combina iniciar y buscar) ó (lo mejor) podem
# os darle una configuración y medir en el mismo comando. Checar cuando usar wr
# ite o query
# =============================================================================

# Para cortar una medición
las.write('ABORt')

# Para cambiar Current setpoint [A]
las.query('SOURce:CURRent? MAXimum')
las.write('SOURce:CURRent 1.0')

# Puede tener función CURRent o POWer (corriente o potencia)
las.query('FUNCtion:MODE?') 

# La forma funcional puede ser DC o PULSe (direct current o pulso -tiene banda de configs,chequear-)
las.query('SOURce:FUNCtion[:SHAPe]?')

# Seteo los dos valores previos
las.write('SOURce:FUNCtion:MODE CURRent;SHAPe DC')

# =============================================================================
# LASER: modulación interna o externa
# =============================================================================

# Chequeo el estado de la modulación. Dice: 'SOURce[1]:AM[:STATe]?' No entiendo si va lo del corchete
las.query('SOURce:AM:1?')

# Activo la modulación. 0 = OFF 1 = ON
las.write('SOURce:AM 1') 

# Chequeo en que setup está la modulación
las.query('SOURce:AM:SOURce?')

# Fijo modulación externa o interna (se pueden tener ambas, no entiendo pa qué)
# las.write('SOURce:AM:SOURCE EXTernal')
las.write('SOURce:AM:SOURCE INTernal')

# Chequeo el tipo de modulación interna y la defino por triangular
las.query('SOURce:AM:INTernal:SHApe?')
las.write('SOURce:AM:INTernal:SHApe TRI') #{SINusoid|SQUare|TRIangle}

# Chequeo la frecuencia de modulación interna y la defino por 1Hz
las.query('SOURce:AM:INTernal:FREQuency? DEF') # [{MIN|MAX|DEF}]
las.write('SOURce:AM:INTernal:FREQuency DEF 1') # [{MIN|MAX|DEF}] Hz

# PARA CHQUEAR LA PROFUNDIAD NO SE QUE ES # TODO
# SOURce[1]:AM:INTernal[:DEPTh] {MIN|MAX|DEF|<percent>}
# SOURce[1]:AM:INTernal[:DEPTh]? [{MIN|MAX|DEF}]

# =============================================================================
# LASER: PD Sense Commands (photodiode Sense commands # TODO COMENTAR VER EN LA
# DOCUMENTACION). TAMBIEN EL DE LA THERMOPILE TEC: thermoelectric cooler.
# =============================================================================

# Mido temperatura y corriente
las.write('MEASure:SCAlar:TEMPerature?')
las.write('MEASure:SCALar:CURRent1:DC?')



# =============================================================================
# Ejemplo para guardar y cargar mediciones un diccionario en formato pkl.
# =============================================================================

# corriente_setting= [1.5,1.4,1.1,1.0,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1] #mA
# corriente_reading = [1.3,1.2,.9,.8,.7,.6,.5,.4,.3,.2,.1,0,0] #ma
# tension = [11.7,11.7,11.7,8.3,.595,.488,.425,.393,.363,.343,.333,.333,.333] #v 
# datos = {
#     'unidades':'Las unidades son correspondientemente mA y V. La escala vertical del osciloscopio estaba seteada en 5 V. El láser a 22.6 y el rb a 26.1 ºC',
#     'corriente_reading':corriente_setting,
#     'corriente_setting':corriente_reading,
#     'tension':tension
# }
# # Guardar datos
# with open(file = 'C:/repos/labo_5/input/espectroscopia/medicion_sinrb.pkl', mode = "wb") as archive:
#     pickle.dump(obj = datos, file = archive)

# # Cargar datos
# with open(file = 'C:/repos/labo_5/input/espectroscopia/medicion_sinrb.pkl', mode = "rb") as archive:
#     datos_leidos = pickle.load(file = archive)