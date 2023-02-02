import os
import numpy as np
from labos import ajuste
from herramientas.config.config_builder import Parser

from scipy import interpolate
from scipy.signal import savgol_filter

# Importo los paths
glob_path = os.path.normpath(os.getcwd())
variables = Parser(configuration = 'ferromagnetismo').config()

# La funciones auxiliares
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

# Escala auxiliar para hacer la transformación a temperatura
temperaturas_auxiliar = np.linspace(-300, 300, 100000) 

conversor_r_t = {r: t for r, t in zip(funcion_conversora_temp(temperaturas_auxiliar), temperaturas_auxiliar)}

# =========================================================================================
# Leo los datos, determiné viendo plots de todos los datos que las únicas mediciones buenas 
# son la 12, 13, 14, 15 y 16. A las tensiones les quito el offset y les aplico un filtro sa
# vgol.
# =========================================================================================

# Errores en las esclas de tensión acorde a c/ medición
errores = {
'medicion_12_c1':8*.5/256,
'medicion_13_c1':8*.5/256, 
'medicion_14_c1':8/256, 
'medicion_15_c1':8*2/256, 
'medicion_16_c1':8/256,#'medicion_16_c1':8*2/256,
'medicion_12_c2':8*.2/256, 
'medicion_13_c2':8*.2/256, 
'medicion_14_c2':8*.2/256, 
'medicion_15_c2':8*.5/256, 
'medicion_16_c2':8/256#'medicion_16_c2':8*.5/256
}

# Elijo la medición
medicion = 16

# Leo la data de la resistencia, transformo su valor a temperatura y guardo el tiempo en el que se hizo la medición
file_resistencias = os.path.join(glob_path + os.path.normpath(variables['input']) + os.path.normpath(f'/Medicion {medicion} - Resistencias.txt'))

resistencia = np.loadtxt(file_resistencias, delimiter = ',', dtype = float)

temperatura = np.array([conversor_r_t[min(conversor_r_t.keys(), key = lambda x:abs(x-r))]  + 273.15 for r in resistencia[0]] )

tiempo = np.array(resistencia[1])

# Armo un iterador. Cada valor es el índice de la medición
iterador = np.arange(1, len(temperatura) + 1)

# Defino un diccionario con las mediciones
globals()[f'medicion_{medicion}'] = {}

for j in iterador:
    CH1 = np.loadtxt(os.path.join(glob_path + os.path.normpath(variables['input']) + os.path.normpath(f'/Medicion {medicion} - CH1 - Resistencia {j}.0.txt')), delimiter = ',', dtype = float)
    CH2 = np.loadtxt(os.path.join(glob_path + os.path.normpath(variables['input']) + os.path.normpath(f'/Medicion {medicion} - CH2 - Resistencia {j}.0.txt')), delimiter = ',', dtype = float)

    # Cuando escribo los datos corrigo offsets, primero respecto al CH1 y en base a eso el CH2. Además agrego filtro sav
    window = 11
    globals()[f'medicion_{medicion}'][j] = {
        'tiempo_r':tiempo[j-1], # j-1 pq las mediciones arrancan en uno pero los valores como el tiempo en 0
        'tiempo_r':tiempo[j-1],
        'temperatura': temperatura[j-1],
        'tiempo_1':CH1[0,:],
        'tiempo_2': CH2[0,:],
        'tension_1': savgol_filter(
            x = CH1[1, :] - CH1[1, :].mean(), 
            window_length = window, 
            polyorder = 0),
        'tension_2': savgol_filter(
            x = (CH2[1, :]- CH1[1, :].mean())-(CH2[1, :]- CH1[1, :].mean()).mean(),
            window_length = window, 
            polyorder = 0)
        }
        
# # ========================================================
# # Levanto una curva de histéresis con los datos de tensión
# # ========================================================

# # Grafico todas las curvas de histeresis para c/medición. Hago un gráfico 3-D defino primero el mapa de colores
# cmap = plt.get_cmap('plasma')
# cmap_values = np.linspace(0., 1., len(temperatura))
# colors = cmap(cmap_values)
# colors_rgb = ['#{0:02x}{1:02x}{2:02x}'.format(int(255*a), int(255*b), int(255*c)) for a, b, c, _ in colors]

# # Hago el gráfico:
# fig, ax = plt.subplots(nrows = 1, ncols = 1, subplot_kw={'projection': '3d'}, figsize = (7,6))
# for i, c in zip(iterador, colors_rgb):
#     med = eval(f'medicion_{medicion}')[i]
#     ax.scatter(med['tension_1'], i, med['tension_2'], c = c, s = 2)
#     ax.set_xlabel(r'Tensión primario $\propto H$ [V]')
#     ax.set_ylabel('Temperatura [K]')
#     ax.set_yticklabels([50, 80, 120, 160, 200, 240, 280])
#     ax.set_zlabel(r'Tensión secundario $\propto B$ [V]')
# fig.show()

# # ========================================================================================
# # Levanto la remanencia para cada curva de histéresis con los datos de tensión que leí. Es
# # decir, la diferencia entre el máximo y mínimo de tensión en el canal 2, cuando la tensió
# # n en el canal 1 vale 0.
# # ========================================================================================

remanencia = []
for i in iterador:
    med = eval(f'medicion_{medicion}')[i]
    
    # Indices cercanos a los cruces de la tension_1 con el 0. Lo calculo viendo cuando cambia de signo
    indices = np.where(np.diff(np.sign(med['tension_1'])))[0]

    # Interpolo linealmente el 0 con veinte puntos delante y veinte detrás
    minimos_tension_2 = []
    for indice in indices:
        f = interpolate.interp1d(med['tension_1'][indice - 20:indice + 20], med['tiempo_1'][indice - 20:indice + 20])
        h = interpolate.interp1d(med['tiempo_2'], med['tension_2'])
        minimos_tension_2.append(h(f(0))) # Cuando el tiempo de 1 es tal que su tensión es 0, evalúo tensión 2
    
    # Agrego el punto de remanencia teniendo en cuenta un promedio de los puntos por arriba y por debajo
    remanencia.append((np.mean([m for m in minimos_tension_2 if m >= 0]) - np.mean([m for m in minimos_tension_2 if m < 0]))/2)

# Convierto la lista en np.ndarray
remanencia = np.array(remanencia)

# Estoy haciendo la resta entre dos valores del canal 2, entonces aparece el factor sqrt(2):
# error = np.full(len(remanencia), np.sqrt(2)*errores[f'medicion_{medicion}_c2'])
error = np.full(
shape = len(remanencia), 
fill_value = np.sqrt(errores[f'medicion_{medicion}_c2']**2 + errores[f'medicion_{medicion}_c1']**2)
)*.5

# Hago el ajuste
# formula = 'np.piecewise(x, [x < a_0, x >= a_0], [lambda x: a_1*np.abs(x- a_0)**(a_2) + a_3, a_3])'
formula = 'np.piecewise(x, [x < a_0, x >= a_0], [lambda x: a_1*np.abs(x- a_0)**(a_2) + a_3, a_3])'

# Initial guess
p_0 = [258,.05,.5,.04]
regr = ajuste.Ajuste(x = temperatura, y = remanencia, cov_y = error.reshape(-1,1))
regr.fit(formula = formula, p0 = p_0)

# TODO: el ajuste explota en la transición de la función partida
regr.graph(
estilo = 'ajuste_2',
label_x = 'Tempeatura [K]',
label_y = r'Tensión [$\propto$ V]',
save = True,
path = os.path.join(glob_path + os.path.normpath(variables['output']) + os.path.normpath('/Ajuste.png'))
)
regr.bondad()
regr.parametros