import os
from herramientas.config.config_builder import Parser

# Importo los paths
glob_path = os.path.normpath(os.getcwd())
variables = Parser(configuration = 'glow').config()

# -*- coding: utf-8 -*-
"""
Editor de Spyder

Este es un archivo temporal.
"""
import pickle,numpy as np
import matplotlib.pyplot as plt
def save_dict(path:str, dic:dict):
    try:
        with open(file = path, mode = "wb") as archive:
            pickle.dump(file = archive, obj=dic)
        print(f'Se guardo en: {path}')
    except:
        print('Algo fallo')

def load_dict(path:str):
    try:
        with open(file = path, mode = "rb") as archive:
            data = pickle.load(file = archive)
        return data
    except:
        print('Algo fallo')

import pyvisa

rm = pyvisa.ResourceManager()
instrumentos = rm.list_resources()  

gen = rm.open_resource(instrumentos[2])
mult1 = rm.open_resource(instrumentos[3])
mult2 = rm.open_resource(instrumentos[4])

# Chequeo que estén bien etiquetados. Identificar los multimetros (17/2: arriba el 1)
gen.query('*IDN?')
mult1.query('*IDN?')
mult2.query('*IDN?')


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
# MULTIMETROS: 
# =============================================================================
tension_entrada = 300
tension_osci = 0.004*tension_entrada
tension_osci += 0.004*50
gen.write('SOURce1:VOLTage:LEVel:IMMediate:OFFSet {}V'.format(tension_osci))

tension_R0 = []
tension_R1 = []
tension_R0_v = []
tension_R1_v = []
tension_iter = np.linspace(300,720,100)
tension_osci = 0.004*tension_iter

for tension in tension_osci:
    gen.write(f'SOURce1:VOLTage:LEVel:IMMediate:OFFSet {tension}V')
    tension_R1.append(float(mult1.query('MEASURE:VOLTage:DC?'))) # V
    tension_R0.append(float(mult2.query('MEASURE:VOLTage:DC?'))) # V

tension_iter_v = np.linspace(720,300,100)
tension_osci_v = 0.004*tension_iter_v

for tension in tension_osci_v:
    gen.write(f'SOURce1:VOLTage:LEVel:IMMediate:OFFSet {tension}V')
    tension_R1_v.append(float(mult1.query('MEASURE:VOLTage:DC?'))) # V
    tension_R0_v.append(float(mult2.query('MEASURE:VOLTage:DC?'))) # V


tension_R0 = np.array(tension_R0)
tension_R1 = np.array(tension_R1)
corriente_1 = tension_R0/150
corriente_2 = tension_R1/56000
corriente_t = corriente_1 + corriente_2

tension_glow = tension_iter - corriente_1*(30000 +150) - corriente_2*30000

tension_R0_v = np.array(tension_R0_v)
tension_R1_v = np.array(tension_R1_v)
corriente_1_v = tension_R0_v/150
corriente_2_v = tension_R1_v/56000
corriente_t_v = corriente_1_v + corriente_2_v


tension_glow_v = tension_iter_v - corriente_1_v*(30000 +150) - corriente_2_v*30000


fig, ax = plt.subplots(nrows= 1,ncols = 1)
ax.scatter(corriente_t, tension_glow, color = 'red', label = 'ida')
ax.scatter(corriente_t_v, tension_glow_v, color = 'blue', label = 'vuelta')
ax.set_xlabel('Corriente [A]')
ax.set_ylabel('Tension [V]')
ax.grid()
fig.legend()
fig.show()


datos = { 
'unidades': 'Medimos un ciclo de histeresis. Pr: 0.38 mbar; L: (44.04 +- .01) mm ',
'tension_entrada': tension_iter.reshape(-1,1),
'tension_entrada_v':tension_iter_v.reshape(-1,1),
'tension_r0': tension_R0.reshape(-1,1),
'tension_r1': tension_R1.reshape(-1,1),
'tension_r0_v': tension_R0_v.reshape(-1,1),
'tension_r1_v': tension_R1_v.reshape(-1,1),
'tension_glow': tension_glow.reshape(-1,1),
'tension_glow_v': tension_glow_v.reshape(-1,1),
'corriente_1': corriente_1.reshape(-1,1),
'corriente_1_v': corriente_1_v.reshape(-1,1),
'corriente_2': corriente_2.reshape(-1,1),
'corriente_2_v': corriente_2_v.reshape(-1,1),
'corriente_t': corriente_t.reshape(-1,1),
'corriente_t_v': corriente_t_v.reshape(-1,1)
}

num = 4
save_dict(path = f'C:\GRUPO8\medicion_{num}.pkl', dic = datos)
datos_leidos = load_dict(path = f'C:\GRUPO8\medicion_{num}.pkl')


fig_2, ax_2 = plt.subplots(nrows= 1,ncols = 1)
ax_2.scatter(datos_leidos['corriente_t'].reshape(-1), datos_leidos['tension_glow'].reshape(-1), color = 'red', label = 'ida')
ax_2.scatter(datos_leidos['corriente_t_v'].reshape(-1), datos_leidos['tension_glow_v'].reshape(-1), color = 'blue', label = 'vuelta')
ax_2.set_xlabel('Corriente [A]')
ax_2.set_ylabel('Tension [V]')
fig_2.legend()
fig_2.show()

