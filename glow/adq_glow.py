import os, pickle, numpy as np, matplotlib.pyplot as plt
import pyvisa

# ====================
# Funciones auxiliares
# ====================
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

def conversor(obj, tension_fuente: float):
    '''
    Esta función escribe la tensión necesaria en el generador para que la fuente de continua alimente con 'tension_fuente'.
    Unidades V
    '''
    tension_osci = 0.004*tension_fuente
    try:
        respuesta = obj.write('SOURce1:VOLTage:LEVel:IMMediate:OFFSet {}V'.format(tension_osci))
    except:
        raise Exception(f'No se pudo setear. Correr el siguiente comando: {obj}.write("SOURce1:VOLTage:LEVel:IMMediate:OFFSet 0V")')
    return respuesta

# =============================
# Configuración de instrumentos
# =============================

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

# El tipo de señal
gen.write('SOURce1:FUNCtion:SHAPe DC')

# Seteamos un offset puede ser en mV o V
gen.write('SOURce1:VOLTage:LEVel:IMMediate:OFFSet 0mV')

# Seteamos la tensión de la fuente continua en 300 V
conversor(obj = gen, tension_fuente = 300)

# Medimos la caída de tensión en R0, R1 iterando por distintos valores en la tensión de la fuente de entrada
tension_R0 = []
tension_R1 = []
tension_R0_v = []
tension_R1_v = []
tension_iter = np.linspace(300,720,100)

for tension in tension_iter:
    conversor(obj = gen, tension_fuente = tension)
    tension_R1.append(float(mult1.query('MEASURE:VOLTage:DC?'))) # V
    tension_R0.append(float(mult2.query('MEASURE:VOLTage:DC?'))) # V

tension_iter_v = np.linspace(720,300,100)

for tension in tension_iter_v:
    conversor(obj = gen, tension_fuente = tension)
    tension_R1_v.append(float(mult1.query('MEASURE:VOLTage:DC?'))) # V
    tension_R0_v.append(float(mult2.query('MEASURE:VOLTage:DC?'))) # V

# Convertimos a corriente teniendo en cuenta el valor de las resistencias
tension_R0 = np.array(tension_R0)
tension_R1 = np.array(tension_R1)
corriente_1 = tension_R0/150 # A
corriente_2 = tension_R1/56000 # A
corriente_t = corriente_1 + corriente_2 # A

tension_R0_v = np.array(tension_R0_v)
tension_R1_v = np.array(tension_R1_v)
corriente_1_v = tension_R0_v/150 # A
corriente_2_v = tension_R1_v/56000 # A
corriente_t_v = corriente_1_v + corriente_2_v # A

# La caída de tensión entre el cátodo y el ánodo es la caida de tensión en la fuente menos la caída en las resistencias
tension_glow = tension_iter - corriente_1*(30000 + 150) - corriente_2*30000
tension_glow_v = tension_iter_v - corriente_1_v*(30000 +150) - corriente_2_v*30000

# Grafico para ver la curva I V
fig, ax = plt.subplots(nrows= 1,ncols = 1)
ax.scatter(corriente_t, tension_glow, color = 'red', label = 'ida')
ax.scatter(corriente_t_v, tension_glow_v, color = 'blue', label = 'vuelta')
ax.set_xlabel('Corriente [A]')
ax.set_ylabel('Tension [V]')
ax.grid()
fig.legend()
fig.show()

# Guardamos los datos

num = 4
Pr, L = 0.38 , 44.04
datos = { 
'unidades': f'Medimos un ciclo de histeresis. Pr: {Pr} mbar; L: ({L} +- .01) mm. La tensión está en V, la corriente en A',
'presion' : Pr,
'tension_entrada_i': tension_iter.reshape(-1,1),
'tension_entrada_v':tension_iter_v.reshape(-1,1),
'tension_r0_i': tension_R0.reshape(-1,1),
'tension_r1_i': tension_R1.reshape(-1,1),
'tension_r0_v': tension_R0_v.reshape(-1,1),
'tension_r1_v': tension_R1_v.reshape(-1,1),
'tension_glow_i': tension_glow.reshape(-1,1),
'tension_glow_v': tension_glow_v.reshape(-1,1),
'corriente_1_i': corriente_1.reshape(-1,1),
'corriente_1_v': corriente_1_v.reshape(-1,1),
'corriente_2_i': corriente_2.reshape(-1,1),
'corriente_2_v': corriente_2_v.reshape(-1,1),
'corriente_t_i': corriente_t.reshape(-1,1),
'corriente_t_v': corriente_t_v.reshape(-1,1),
'tension_entrada': np.concatenate((tension_iter, tension_iter_v)),
'tension_r0': np.concatenate((tension_R0, tension_R0_v)),
'tension_r1': np.concatenate((tension_R1, tension_R1_v)),
'tension_glow': np.concatenate((tension_glow, tension_glow_v)),
'corriente_1': np.concatenate((corriente_1, corriente_1_v)),
'corriente_2': np.concatenate((corriente_2, corriente_2_v)),
'corriente_t': np.concatenate((corriente_t, corriente_t_v)),
}
for i in range(1, 5):
    fname = f'C:/repos/labo_5/input/glow/medicion_{i}.pkl'
    datos = load_dict(fname = fname)
    datos['presion']  = float(datos['unidades'].split('Pr: ')[1].split(' mbar')[0])
    save_dict(fname = fname, dic = datos, rewrite = True)
fname = f'C:\GRUPO8\medicion_{num}.pkl'
save_dict(fname = fname, dic = datos)

# Volvemos a graficar para corroborar que estén los datos
datos_leidos = load_dict(fname = fname)
fig, ax = plt.subplots(nrows= 1,ncols = 1)
ax.scatter(datos_leidos['corriente_t'].reshape(-1), datos_leidos['tension_glow'].reshape(-1), color = 'red', label = 'ida')
ax.scatter(datos_leidos['corriente_t_v'].reshape(-1), datos_leidos['tension_glow_v'].reshape(-1), color = 'blue', label = 'vuelta')
ax.set_xlabel('Corriente [A]')
ax.set_ylabel('Tension [V]')
fig.legend()
fig.show()

# Graficamos todas las mediciones juntas
fig, ax = plt.subplots(nrows= 1, ncols = 1)
# for i in range(1, num + 1):
    # fname = f'C:\GRUPO8\medicion_{i}.pkl'
fname = f'C:/repos/labo_5/input/glow/medicion_{i}.pkl'
datos_leidos = load_dict(fname = fname)
ax.scatter(datos_leidos['corriente_t'], datos_leidos['tension_glow'], label = f'{datos_leidos['Pr']} mbar', s = 2)
ini_flechas_x, ini_flechas_y = np.roll(datos_leidos['corriente_t'], 1), np.roll(datos_leidos['tension_glow'], 1)
fin_flechas_x = np.concatenate((datos_leidos['corriente_t'][:-1],np.array([datos_leidos['corriente_t'][-2]])))
fin_flechas_y = np.concatenate((datos_leidos['tension_glow'][:-1],np.array([datos_leidos['tension_glow'][-2]]))) 
for X_f, Y_f, X_i, Y_i in zip(fin_flechas_x, fin_flechas_y, ini_flechas_x, ini_flechas_y):
    ax.annotate(text = "",
    xy = (X_f,Y_f), 
    xytext = (X_i,Y_i),
    arrowprops = {'arrowstyle': "->", 'color':f'{c}'},
    size = 7
    )
ax.set_xlabel('Corriente [A]')
ax.set_ylabel('Tension [V]')
fig.legend()
fig.show()

import pandas as pd
df = pd.DataFrame.from_dict({'x' : [0,3,8,7,5,3,2,1],
                             'y' : [0,1,3,5,9,8,7,5]})
xpuntas = df['x']
ypuntas = df['y']

# calculate position and direction vectors:

xinicio = np.roll(xpuntas, 1)
yinicio = np.roll(ypuntas, 1)
fig, ax = plt.subplots()
ax.scatter(x,y)
ax.plot(x,y)

# plot arrow on each line:
for X_f, Y_f, X_i, Y_i in zip(xpuntas, ypuntas, xinicio, yinicio):
    ax.annotate(text = "",
    xy = (X_f,Y_f), 
    xytext = (X_i,Y_i),
    arrowprops = dict(arrowstyle="->", color='k'),
    size = 20
    )
fig.show()
