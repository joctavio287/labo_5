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

# # Ejemplo medicion

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