import os, matplotlib.pyplot as plt, numpy as np
from herramientas.config.config_builder import Parser, save_dict, load_dict
from labos.ajuste import Ajuste
from labos.propagacion import Propagacion_errores

# Importo los paths
glob_path = os.path.normpath(os.getcwd())
variables = Parser(configuration = 'glow').config()
input_path = os.path.join(glob_path + os.path.normpath(variables['input']))
output_path = os.path.join(glob_path + os.path.normpath(variables['output']))

# ====================================================
# Graficamos todas las mediciones juntas de histéresis
# ====================================================

# Ultima medicion efectuada
num = 4

# Graficamos todas las mediciones juntas de histéresis
cmap = plt.get_cmap('plasma')
cmap_values = np.linspace(0., 1., num)
colors = cmap(cmap_values)
colors_rgb = ['#{0:02x}{1:02x}{2:02x}'.format(int(255*a), int(255*b), int(255*c)) for a, b, c, _ in colors]

plt.figure()
dic = {1:4,2:4,3:4, 4:4}
for c, i in zip(colors_rgb, np.arange(1, num + 1 , 1)): # for i in range(1, num + 1):
    fname = os.path.join(input_path + os.path.normpath(f'/medicion_{i}.pkl'))
    datos_leidos = load_dict(fname = fname)
    pr = datos_leidos['presion']
    corriente = 1000*datos_leidos['corriente_t']
    plt.plot(corriente, datos_leidos['tension_glow'], '-.',label = f'{pr} mbar', color = c)
    ini_flechas_x = [corriente[len(corriente)//dic[i]],corriente[len(corriente)*2//3]]
    ini_flechas_y = [datos_leidos['tension_glow'][len(datos_leidos['tension_glow'])//dic[i]],datos_leidos['tension_glow'][len(datos_leidos['tension_glow'])*2//3]]
    fin_flechas_x = [corriente[len(corriente)//dic[i] + 3],corriente[len(corriente)*2//3+3]]
    fin_flechas_y = [datos_leidos['tension_glow'][len(datos_leidos['tension_glow'])//dic[i]+3],datos_leidos['tension_glow'][len(datos_leidos['tension_glow'])*2//3+3]]
    for X_f, Y_f, X_i, Y_i in zip(fin_flechas_x, fin_flechas_y, ini_flechas_x, ini_flechas_y):
        plt.annotate(text = "",
        xy = (X_f,Y_f), 
        xytext = (X_i,Y_i),
        arrowprops = dict(arrowstyle = "->"),#, color = c),
        size = 25
        )
plt.grid(visible = True)
plt.xlabel('Corriente [mA]')
plt.ylabel('Tension [V]')
plt.legend(loc = 'best')
plt.tight_layout()
plt.show(block = False)
# plt.savefig(fname = os.path.join(output_path + os.path.normpath('/mediciones_VI_dia1.png')))

# ====================================================
# ====================================================

# ================================================
# Graficamos la tensión de ruptura Paschen ejemplo 
# ================================================
i = 1
fname = os.path.join(input_path + os.path.normpath(f'/medicion_paschen_{i}.pkl'))
datos_leidos = load_dict(fname = fname)

plt.figure()
plt.hlines(y = datos_leidos['ruptura'], xmin = 0, xmax = datos_leidos['corriente_t'][-1]*1000, color = 'red', label = 'Tensión de ruptura')
# Transformo a mA
corriente = datos_leidos['corriente_t']*1000
plt.scatter(corriente, datos_leidos['tension_glow'], s = 2, color = 'blue', label = 'Curva IV')
plt.xlabel('Intensidad de corriente [mA]')
plt.ylabel('Tension entre electrodos [V]')
plt.grid(visible = True)
plt.legend(loc = 'lower left')
plt.show(block = False)
# plt.savefig(os.path.join(output_path + os.path.normpath('/tension_ruptura_ej.png')))

# ================================================
# ================================================

# =========================================
# Respuesta de la tensión al glow: cuaderno 
# =========================================
fname = os.path.join(input_path + os.path.normpath(f'/medicion_osciloscopio.pkl'))
datos_leidos = load_dict(fname = fname)

plt.figure()
plt.scatter(datos_leidos['tiempo'], datos_leidos['tension'], s = 2, color = 'blue', label = 'Respuesta en tensión al Glow')
plt.xlabel('Tiempo [s]')
plt.ylabel('Tension [V]')
plt.grid(visible = True)
plt.legend(loc = 'best')
plt.show(block = False)
# plt.savefig(os.path.join(output_path + os.path.normpath('/respuesta_osci.png')))
# =========================================
# =========================================

# =========================================
# Espectrograma de la luz del aire y del He 
# =========================================
helio_0_64 = 'espectro_helio_0.64mbar.csv'
helio_0_16 = 'espectro_helio_0.16mbar.csv'
aire = 'espectro_aire.csv'

# Aire 1.6 mbar
fname = os.path.join(input_path + os.path.normpath(f'/{aire}'))
datos_leidos = np.loadtxt(fname = fname, delimiter = ';', skiprows=33)
frecuencia, intensidad = datos_leidos[:,0].reshape(-1,1), datos_leidos[:,1].reshape(-1,1)
plt.figure()
plt.plot(frecuencia, intensidad, '-', label = 'Aire 1,6 mbar')
plt.xlabel('Tiempo [s]')
plt.ylabel('Tension [V]')
plt.grid(visible = True)
plt.legend(loc = 'best')
plt.show(block = False)
# plt.savefig(os.path.join(output_path + os.path.normpath('/espectro_aire.png')))

# Helio .64 mbar
fname = os.path.join(input_path + os.path.normpath(f'/{helio_0_64}'))
datos_leidos = np.loadtxt(fname = fname, delimiter = ',', skiprows=33)
frecuencia, intensidad = datos_leidos[:,0].reshape(-1,1), datos_leidos[:,1].reshape(-1,1)
plt.figure()
plt.plot(frecuencia, intensidad, '-', label = 'Helio 0,64 mbar')
plt.xlabel('Tiempo [s]')
plt.ylabel('Tension [V]')
plt.grid(visible = True)
plt.legend(loc = 'best')
plt.show(block = False)
# plt.savefig(os.path.join(output_path + os.path.normpath('/espectro_he_64.png')))

# Helio .16 mbar
fname = os.path.join(input_path + os.path.normpath(f'/{helio_0_16}'))
datos_leidos = np.loadtxt(fname = fname, delimiter = ',', skiprows=33)
frecuencia, intensidad = datos_leidos[:,0].reshape(-1,1), datos_leidos[:,1].reshape(-1,1)
plt.figure()
plt.plot(frecuencia, intensidad, '-', label = 'Helio 0,16 mbar')
plt.xlabel('Tiempo [s]')
plt.ylabel('Tension [V]')
plt.grid(visible = True)
plt.legend(loc = 'best')
plt.show(block = False)
# plt.savefig(os.path.join(output_path + os.path.normpath('/espectro__he_16.png')))
# =========================================
# =========================================


# =======================================================
# Graficamos todas las mediciones juntas de histéresis He
# =======================================================

# Ultima medicion efectuada
num = 4

# Graficamos todas las mediciones juntas de histéresis
cmap = plt.get_cmap('plasma')
cmap_values = np.linspace(0., 1., num)
colors = cmap(cmap_values)
colors_rgb = ['#{0:02x}{1:02x}{2:02x}'.format(int(255*a), int(255*b), int(255*c)) for a, b, c, _ in colors]

plt.figure()
dic = {1:4,2:3,3:3, 4:3}
for c, i in zip(colors_rgb, np.arange(1, num + 1 , 1)): # for i in range(1, num + 1):
    fname = os.path.join(input_path + os.path.normpath(f'/medicion_helio_{i}.pkl'))
    datos_leidos = load_dict(fname = fname)
    pr = datos_leidos['presion']
    corriente = 1000*datos_leidos['corriente_t']
    plt.plot(corriente, datos_leidos['tension_glow'], '-.',label = f'{pr} mbar', color = c)
    ini_flechas_x = [corriente[len(corriente)//dic[i]],corriente[len(corriente)*2//3]]
    ini_flechas_y = [datos_leidos['tension_glow'][len(datos_leidos['tension_glow'])//dic[i]],datos_leidos['tension_glow'][len(datos_leidos['tension_glow'])*2//3]]
    fin_flechas_x = [corriente[len(corriente)//dic[i] + 3],corriente[len(corriente)*2//3+3]]
    fin_flechas_y = [datos_leidos['tension_glow'][len(datos_leidos['tension_glow'])//dic[i]+3],datos_leidos['tension_glow'][len(datos_leidos['tension_glow'])*2//3+3]]
    for X_f, Y_f, X_i, Y_i in zip(fin_flechas_x, fin_flechas_y, ini_flechas_x, ini_flechas_y):
        plt.annotate(text = "",
        xy = (X_f,Y_f), 
        xytext = (X_i,Y_i),
        arrowprops = dict(arrowstyle = "->"),#, color = c),
        size = 25
        )
plt.grid(visible = True)
plt.xlabel('Corriente [mA]')
plt.ylabel('Tension [V]')
plt.legend(loc = 'best')
plt.tight_layout()
plt.show(block = False)
# plt.savefig(fname = os.path.join(output_path + os.path.normpath('/mediciones_VI_helio.png')))
# =======================================================
# =======================================================

# ===================================================
# Levantamos la curva de Paschen e intentamos ajustar
# ===================================================

# Armo dos arrays con los datos de presión por distancia y otro de tensión de ruptura
rupturas = []
pds = []
for i in range(1, 26):
    fname = os.path.join(input_path + os.path.normpath(f'/medicion_paschen_{i}.pkl'))
    datos_i = load_dict(fname = fname)
    rupturas.append(datos_i['ruptura'])
    pds.append(datos_i['pd']/10)

# Transformo a arrays
rupturas = np.array(rupturas).reshape(-1,1)
pds = np.array(pds).reshape(-1,1)

# Propago el error de las rupturas
# ERROR TODO

# Hago el ajuste
def func(x, a_0, a_1, a_2):
    return a_0*(x)/(np.log(a_1*x) - np.log(a_2))

formula_de_paschen = 'a_0*(x)/(np.log(a_1*x) + np.log(a_2))'
# (e *exp(a_2))/a_1 es el minimo que es aprox 0.63
# cuando tiende a infinito x, crece como a_0*x/a_1
ajuste = Ajuste(x = pds, y = rupturas, cov_y = np.full(shape = rupturas.shape, fill_value = 1))
ajuste.fit(formula = formula_de_paschen, p0 = [600, 3, .2])
a_0, a_1, a_2 = ajuste.parametros
(np.e*a_2)/a_1
# Ploteo
pds_auxiliar = np.linspace(pds[0][0],pds[-1][0],1000)

fig, ax = plt.subplots(nrows = 1, ncols = 1)
ax.scatter(pds, rupturas, color = 'blue', s = 10)
ax.plot(pds_auxiliar, func(pds_auxiliar, a_0, a_1, a_2))
ax.set_xlabel('Presión x distancia [mbar x cm]')
ax.set_ylabel('Tension de ruptura [V]')
ax.grid(visible = True)
fig.show()



# ===================================================
# ===================================================