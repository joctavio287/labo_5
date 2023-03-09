import os, matplotlib.pyplot as plt, numpy as np
from herramientas.config.config_builder import Parser, save_dict, load_dict
from labos.ajuste import Ajuste
from labos.propagacion import Propagacion_errores
from sympy import symbols, lambdify, latex, log
from matplotlib.widgets import Slider, Button
from scipy.optimize import curve_fit

# Importo los paths
glob_path = os.path.normpath(os.getcwd())
variables = Parser(configuration = 'glow').config()
input_path = os.path.join(glob_path + os.path.normpath(variables['input']))
output_path = os.path.join(glob_path + os.path.normpath(variables['output']))

# # # ARREGLANDO TENSION GLOW MEDICIONES AIRE
# R_0, e_R_0 = 149.2, 0.2  # TODO MEDIR DE NUEVO RESISTENCIAS
# R_1, e_R_1 = 55325, 6
# R_2, e_R_2 = 56100000, 458000
# R_3, e_R_3 = 29.932, 4
# formula_tension_glow = '-t_r_0 + (t_r_1)*(r3_3/r1_2 + 1)'
# t_0  = time.time()
# for i in range(1, 5):
#     fname = os.path.join(input_path + os.path.normpath(f'/medicion_{i}.pkl'))
#     datos_i = load_dict(fname = fname)
#     datos_i['tension_glow'] = -datos_i['tension_r0'] + datos_i['tension_r1']*(R_2/R_1 + 1)
#     datos_i['error_tension_r0'] = 0.000035*datos_i['tension_r0']+0.000005*10
#     datos_i['error_tension_r1'] = 0.000040*datos_i['tension_r1']+0.000007*1 
#     error_glow = []
#     for j in range(len(datos_i['tension_r0'])):
#         propagacion = Propagacion_errores(
#         formula = formula_tension_glow,
#         variables = [('t_r_0', datos_i['tension_r0'][j]), # tensión r_0
#                     ('t_r_1', datos_i['tension_r1'][j]), # tensión r_1
#                     ('r1_2', R_1), # r_1
#                     ('r3_3', R_2), # r_2
#                     ],
#         errores = np.array([
#             datos_i['error_tension_r0'][j],
#             datos_i['error_tension_r0'][j],
#             e_R_1,
#             e_R_2]).reshape(-1, 1)
#         )
#         error_glow.append(propagacion.fit()[1])
#     datos_i['error_glow'] = np.array(error_glow).reshape(-1,1)
#     save_dict(fname = fname, dic = datos_i, rewrite = True)

# for i in range(1, 5):
#     fname = os.path.join(input_path + os.path.normpath(f'/medicion_helio_{i}.pkl'))
#     datos_i = load_dict(fname = fname)
#     datos_i['tension_glow'] = -datos_i['tension_r0'] + datos_i['tension_r1']*(R_2/R_1 + 1)
#     datos_i['error_tension_r0'] = 0.000035*datos_i['tension_r0']+0.000005*10
#     datos_i['error_tension_r1'] = 0.000040*datos_i['tension_r1']+0.000007*1 
#     error_glow = []
#     for j in range(len(datos_i['tension_r0'])):
#         propagacion = Propagacion_errores(
#         formula = formula_tension_glow,
#         variables = [('t_r_0', datos_i['tension_r0'][j]), # tensión r_0
#                     ('t_r_1', datos_i['tension_r1'][j]), # tensión r_1
#                     ('r1_2', R_1), # r_1
#                     ('r3_3', R_2), # r_2
#                     ],
#         errores = np.array([
#             datos_i['error_tension_r0'][j],
#             datos_i['error_tension_r0'][j],
#             e_R_1,
#             e_R_2]).reshape(-1, 1)
#         )
#         error_glow.append(propagacion.fit()[1])
#     datos_i['error_glow'] = np.array(error_glow).reshape(-1,1)
#     save_dict(fname = fname, dic = datos_i, rewrite = True)
# for i in range(1, 26):
#     fname = os.path.join(input_path + os.path.normpath(f'/medicion_paschen_{i}.pkl'))
#     datos_i = load_dict(fname = fname)
#     datos_i['tension_glow'] = -datos_i['tension_r0'] + datos_i['tension_r1']*(R_2/R_1 + 1)
#     datos_i['error_tension_r0'] = 0.000035*datos_i['tension_r0']+0.000005*10
#     datos_i['error_tension_r1'] = 0.000040*datos_i['tension_r1']+0.000007*1 
#     error_glow = []
#     for j in range(len(datos_i['tension_r0'])):
#         propagacion = Propagacion_errores(
#         formula = formula_tension_glow,
#         variables = [('t_r_0', datos_i['tension_r0'][j,0]), # tensión r_0
#                     ('t_r_1', datos_i['tension_r1'][j,0]), # tensión r_1
#                     ('r1_2', R_1), # r_1
#                     ('r3_3', R_2), # r_2
#                     ],
#         errores = np.array([
#             datos_i['error_tension_r0'][j,0],
#             datos_i['error_tension_r0'][j,0],
#             e_R_1,
#             e_R_2]).reshape(-1, 1)
#         )
#         error_glow.append(propagacion.fit()[1])
#     datos_i['error_glow'] = np.array(error_glow).reshape(-1,1)
#     save_dict(fname = fname, dic = datos_i, rewrite = True)

# print(time.time()-t_0)
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
# Graficamos la tensión de ruptura Paschen ejemplo # TODO HACER IGUAL A ARRIBA PERO CON ZOOM A LA RUPTURA Y MARCAR CON FLECHAS LA  HISTERESIS
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
# # TODO ELEGIR UNA DEL HELIO Y USAR SIN ETIQUETAS DE PRESIÓN Y VER DE GRAFICARRLAS JUNTAS
# # TODO BUSCAR PICOS: CHEQUEAR CORRESPONDENCIA. AIRE: NITROGENO; HE: He.
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
errores_rupturas = []
R_0, e_R_0 = 149.2, 0.2  # TODO MEDIR DE NUEVO RESISTENCIAS
R_1, e_R_1 = 55325, 6
R_2, e_R_2 = 56100000, 458000
R_3, e_R_3 = 29.932, 4
formula_tension_glow = '-t_r_0 + (t_r_1)*(r3_3/r1_2 + 1)'
for i in range(1, 5):
    # fname = os.path.join(input_path + os.path.normpath(f'/medicion_paschen_{i}.pkl'))
    fname = os.path.join(input_path + os.path.normpath(f'/medicion_{i}.pkl'))
    datos_i = load_dict(fname = fname)
    for j in range(len(datos_i['tension_r0'])):
        # propagacion = Propagacion_errores(
        # formula = formula_tension_glow,
        # variables = [('t_r_0', datos_i['tension_r0'][j,0]), # tensión r_0
        #             ('t_r_1', datos_i['tension_r1'][j,0]), # tensión r_1
        #             ('r1_2', R_1), # r_1
        #             ('r3_3', R_2), # r_2
        #             ],
        # errores = np.array([
        #     datos_i['error_tension_r0'][j,0],
        #     datos_i['error_tension_r0'][j,0],
        #     e_R_1,
        #     e_R_2]).reshape(-1, 1)
        # )
        propagacion = Propagacion_errores(
        formula = formula_tension_glow,
        variables = [('t_r_0', datos_i['tension_r0'][j]), # tensión r_0
                    ('t_r_1', datos_i['tension_r1'][j]), # tensión r_1
                    ('r1_2', R_1), # r_1
                    ('r3_3', R_2), # r_2
                    ],
        errores = np.array([
            datos_i['error_tension_r0'][j],
            datos_i['error_tension_r0'][j],
            e_R_1,
            e_R_2]).reshape(-1, 1)
        )
        datos_i['error_glow'] = propagacion.fit()[1]
    rupturas.append(datos_i['ruptura'])
    pds.append(datos_i['pd']/10)

# Transformo a arrays
rupturas = np.array(rupturas).reshape(-1,1)
pds = np.array(pds).reshape(-1,1)

# Propago el error de las rupturas y de pds # TODO
R_0, e_R_0 = 149.2, 0.2  # TODO MEDIR DE NUEVO RESISTENCIAS
R_1, e_R_1 = 55325, 6
R_2, e_R_2 = 56100000, 458000
R_3, e_R_3 = 29.932, 4
tension_R_0, e_tension_R_0 = datos_i['tension_r0'], 0.000035*datos_i['tension_r0']+0.000005*10
tension_R_1, e_tension_R_1 = datos_i['tension_r1'], 0.000040*datos_i['tension_r1']+0.000007*1





# Defino la fórmula de Paschen:
def func(x, a_0, a_1, a_2):
    return a_0*(x)/(np.log(a_1*x) + a_2)

formula_de_paschen = f'a_0*(x)/(np.log(a_1*x) + a_2)'

# Para chequear en que valores diverge el método
# for a_1, a_2 in zip(np.linspace(-10,10,100), np.linspace(1,20, 100)):
#     a_1,a_2
#     try:
#         np.asarray_chkfinite(np.log(a_1*pds)-a_2)
#     except:
#         raise ValueError(f'{a_1, a_2}')
# Hago el ajuste
ajuste = Ajuste(x = pds, y = rupturas, cov_y = np.full(shape = rupturas.shape, fill_value = 10)) # TODO: definir bien el error
# p_a_0, p_a_1, p_a_2 = 699, 0.001, 12
p_a_0, p_a_1, p_a_2=555, .0000482, 11.458
# p_a_0, p_a_1, p_a_2 = 1,1,1

ajuste.fit(
formula = formula_de_paschen,
p0 = [p_a_0, p_a_1, p_a_2], 
# bounds = ([500,0.000001,0],[800,.0000484,np.inf]), 
bounds = ([554,.0000482,0],[800,.0000484,np.inf]), 

# bounds = ([200,0.000001,-np.inf],[np.inf,100,20]), 
method = 'dogbox'
)
ajuste.parametros, np.sqrt(np.diag(ajuste.cov_parametros))

# ajuste = Ajuste(x = pds_teo, y = tension_teo, cov_y = np.full(shape = tension_teo.shape, fill_value = 10)) # TODO: definir bien el error
# # p_a_0, p_a_1, p_a_2 = 699, 0.001, 12
# p_a_0, p_a_1, p_a_2=555, .0000482, 11.458
# # p_a_0, p_a_1, p_a_2 = 1,1,1

# ajuste.fit(
# formula = formula_de_paschen,
# p0 = [p_a_0, p_a_1, p_a_2], 
# # bounds = ([500,0.000001,0],[800,.0000484,np.inf]), 
# # bounds = ([554,.0000482,0],[800,.0000484,np.inf]), 
# bounds = ([0,0.000001,-20],[np.inf, 1,20]), 
# # method = 'dogbox'
# )
# ajuste.parametros, np.sqrt(np.diag(ajuste.cov_parametros))
# ajuste.graph(estilo = 'ajuste_2')


pds_auxiliar = np.linspace(pds[0], pds[-1],1000)

plt.figure()    
# plt.scatter(pds, rupturas, label = 'Datos')
# plt.plot(pds_auxiliar, func(pds_auxiliar,*ajuste.parametros), linewidth = 2)


# plt.xlim([0, 1.75])
# plt.ylim([300, 1000])
plt.xlabel('pd', fontsize = 16)
plt.ylabel('Tensión de ruptura [V]', fontsize = 16) #, rotation = 0)

# plt.xscale('log')
# plt.yscale('log')
plt.grid(visible = True)
plt.show(block = False)

ajuste.graph(estilo = 'ajuste_1')
