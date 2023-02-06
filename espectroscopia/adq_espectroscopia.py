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
# Configuro la curva
# =============================================================================

# Modo de transmision: Binario positivo. 
osc.write('DATa:ENC RPB') 

# Un byte de dato. Con RPB son 256 bits.
osc.write('DATa:WIDth 1')

# La curva mandada inicia en el primer dato y termina en el último dato, sin importar el modo de adquisición
osc.write("DATa:STARt 1") 
osc.write("DATa:STOP 2500") #La curva mandada finaliza en el último dato


#Adquisición por sampleo
self._osci.write("ACQ:MOD SAMP")

#Seteo de canal
self.setCanal(canal = 1, escala = 20e-3)
self.setCanal(canal = 2, escala = 20e-3)
self.setTiempo(escala = 1e-3, cero = 0)

#Bloquea el control del osciloscopio
self._osci.write("LOC")

def __del__(self):
self._osci.write("UNLOC") #Desbloquea el control del osciloscopio
self._osci.close()

def setCanal(self, canal, escala, cero = 0):
#if coup != "DC" or coup != "AC" or coup != "GND":
#coup = "DC"
#self._osci.write("CH{0}:COUP ".format(canal) + coup) #Acoplamiento DC
#self._osci.write("CH{0}:PROB 
print
self._osci.write("CH{0}:SCA {1}".format(canal,escala))
self._osci.write("CH{0}:POS {1}".format(canal,cero))

def getCanal(self,canal):
return self._osci.query("CH{0}?".format(canal))

def setTiempo(self, escala, cero = 0):
self._osci.write("HOR:SCA {0}".format(escala))
self._osci.write("HOR:POS {0}".format(cero))	

def getTiempo(self):
return self._osci.query("HOR?")

def getVentana(self,canal):
self._osci.write("SEL:CH{0} ON".format(canal)) #Hace aparecer el canal en pantalla. Por si no está habilitado
self._osci.write("DAT:SOU CH{0}".format(canal)) #Selecciona el canal
#xze primer punto de la waveform
#xin intervalo de sampleo
#ymu factor de escala vertical
#yoff offset vertical
xze, xin, yze, ymu, yoff = self._osci.query_ascii_values('WFMPRE:XZE?;XIN?;YZE?;YMU?;YOFF?;', 
                                                            separator=';') 
data = (self._osci.query_binary_values('CURV?', datatype='B', 
                                        container=np.array) - yoff) * ymu + yze        
tiempo = xze + np.arange(len(data)) * xin
return tiempo, data
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