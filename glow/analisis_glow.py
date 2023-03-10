import os, matplotlib.pyplot as plt, numpy as np
from herramientas.config.config_builder import Parser, save_dict, load_dict
from labos.ajuste import Ajuste
from labos.propagacion import Propagacion_errores
from sympy import symbols, lambdify, latex, log
from matplotlib.widgets import Slider, Button
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from matplotlib.legend_handler import HandlerLine2D

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
    pr = '(' + str(datos_leidos['presion']).replace('.',',') + r' $\pm$' +' 0,02)'
    corriente = 1000*datos_leidos['corriente_t']
    plt.plot(corriente, datos_leidos['tension_glow'], '-.', color = c, label = f'{pr} mbar')
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
    plt.errorbar(
    corriente, 
    datos_leidos['tension_glow'], 
    yerr = datos_leidos['error_glow'].reshape(-1), 
    xerr = 1000*datos_leidos['error_corriente_t'].reshape(-1),
    marker = '.', 
    fmt = 'None', 
    # capsize = 1, 
    color = c)
plt.grid(visible = True)
plt.xlabel('Intensidad de corriente [mA]')
plt.ylabel('Tension [V]')
plt.legend(loc = 'best')
plt.tight_layout()
plt.show(block = False)
# plt.savefig(fname = os.path.join(output_path + os.path.normpath('/informe/mediciones_VI_aire.svg')))

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

espectro_aire_teorico = [337.1, 357.7, 380.5, 391.4, 399.5, 405.9, 427.8] 
#The removal of impurities from gray cotton fabric by atmospheric pressure plasma treatment and its characterization using ATR-FTIR spectroscopy (Hemen Dave, Lalita Ledwani, Nisha Chandwani, Narendrasinh Chauhan and S.K. Nema
fig, axs = plt.subplots(2, sharex=True, figsize=(7,10))
# Aire 1.6 mbar
fname = os.path.join(input_path + os.path.normpath(f'/{aire}'))
datos_leidos = np.loadtxt(fname = fname, delimiter = ';', skiprows=33)
frecuencia, intensidad = datos_leidos[:,0], datos_leidos[:,1]
pks = find_peaks(intensidad, distance=25, height=0.0731)[0][:8]
# plt.figure()
# axs[0].set_title('Espectrometría del aire', fontsize=20)
axs[0].plot(frecuencia, intensidad, 'b', label = 'Aire')
axs[0].errorbar(frecuencia[pks], intensidad[pks], xerr=2, fmt='ko', capsize=5)
axs[0].vlines(espectro_aire_teorico, min(intensidad), max(intensidad), linestyles='dashed', colors='green', zorder=0)
# axs[0].set_xlabel('Longitud de onda [nm]')
# axs[0].set_ylabel('Intensidad [u.a.]')
axs[0].grid(visible = True)
# plt.show(block = False)
# print(frecuencia[pks])
# plt.savefig(os.path.join(output_path + os.path.normpath('/espectro_aire.png')))

espectro_he_teorico = [393.5912, 447.148, 471.314, 492.193, 501.567, 587.562, 667.815, 706.5190, 728.1349]
#https://physics.nist.gov/cgi-bin/ASD/lines1.pl?spectra=He&limits_type=0&low_w=200&upp_w=1000&unit=1&submit=Retrieve+Data&de=0&I_scale_type=1&format=0&line_out=0&en_unit=0&output=0&bibrefs=1&page_size=15&show_obs_wl=1&order_out=0&max_low_enrg=&show_av=2&max_upp_enrg=&tsb_value=0&min_str=&A_out=0&intens_out=on&max_str=&allowed_out=1&forbid_out=1&min_accur=&min_intens=&conf_out=on&term_out=on&enrg_out=on&J_out=on

# Helio .64 mbar
fname = os.path.join(input_path + os.path.normpath(f'/{helio_0_64}'))
datos_leidos = np.loadtxt(fname = fname, delimiter = ',', skiprows=33)
frecuencia, intensidad = datos_leidos[:,0], datos_leidos[:,1]
pks = find_peaks(intensidad, distance=25, height=0.10)[0][[0, 2, 3, 4, 5, 6, 8, 9, 10]]
# plt.figure()
# axs[1].set_title('Espectrometría del helio', fontsize=20)
axs[1].plot(frecuencia, intensidad, 'r', label = 'Helio')
axs[1].vlines(espectro_he_teorico, min(intensidad), max(intensidad), linestyles='dashed', colors='green', label='Longitudes de onda de referencia', zorder=0)
axs[1].errorbar(frecuencia[pks], intensidad[pks], xerr=2, fmt='ko', capsize=5, label='Picos de intensidad')
axs[1].set_xlabel('Longitud de onda [nm]', fontsize=15)
fig.supylabel('Intensidad [u.a.]', fontsize=15)
axs[1].set_xlim(330, 735)
axs[1].grid(visible = True)
# axs[1].legend(loc = 'best')
handles, labels = [(a + b) for a, b in zip(axs[0].get_legend_handles_labels(), axs[1].get_legend_handles_labels())]
axs[0].legend(handles, labels, loc='upper right')
plt.show(block = False)
# print(frecuencia[pks])
# plt.savefig(os.path.join(output_path + os.path.normpath('/espectro_aire_he.svg')))

# Helio .16 mbar
# fname = os.path.join(input_path + os.path.normpath(f'/{helio_0_16}'))
# datos_leidos = np.loadtxt(fname = fname, delimiter = ',', skiprows=33)
# frecuencia, intensidad = datos_leidos[:,0], datos_leidos[:,1]
# pks = find_peaks(intensidad, distance=25, height=0.17)[0]
# # plt.figure()
# plt.plot(frecuencia, intensidad, '-', label = 'Helio 0,16 mbar')
# plt.plot(frecuencia[pks], intensidad[pks], 'x')
# plt.xlabel('Longitud de onda [nm]')
# plt.ylabel('Intensidad [u.a.]')
# plt.grid(visible = True)
# plt.legend(loc = 'best')
# plt.show(block = False)
# print(frecuencia[pks])
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
    datos_leidos['unidades']
    pr = '(' + str(datos_leidos['presion']).replace('.',',') + r' $\pm$' +' 0,02)'
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
    plt.errorbar(
    corriente, 
    datos_leidos['tension_glow'], 
    yerr = datos_leidos['error_glow'].reshape(-1), 
    xerr = 1000*datos_leidos['error_corriente_t'].reshape(-1), 
    marker = '.', 
    fmt = 'None', 
    # capsize = 1, 
    color = c)
plt.grid(visible = True)
plt.xlabel('Intensidad de corriente [mA]')

plt.ylabel('Tension [V]')
plt.legend(loc = 'best')
plt.tight_layout()
plt.show(block = False)
# plt.savefig(fname = os.path.join(output_path + os.path.normpath('/informe/mediciones_VI_helio.svg')))
# =======================================================
# =======================================================

# ===================================================
# Levantamos la curva de Paschen e intentamos ajustar
# ===================================================
# Defino la fórmula de Paschen y su derivada:
def func(x, a_0, a_1, a_2):
    return a_0*(x)/(np.log(a_1*x) + a_2)
formula_de_paschen = f'a_0*(x)/(np.log(a_1*x) + a_2)'
def dfunc(x, a_0, a_1, a_2):
    return(a_0*(np.log(a_1*x)+a_2-1))/(np.log(a_1*x)+a_2)**2

# Armo dos arrays con los datos de presión por distancia y otro de tensión de ruptura
rupturas = []
errores_rupturas = []
pds = []
errores_pds = []
for i in range(1, 26):
    fname = os.path.join(input_path + os.path.normpath(f'/n_medicion_paschen_{i}.pkl'))
    datos = load_dict(fname = fname)

    # Datos ruptura [V]
    rupturas.append(datos['ruptura'])
  
    # Datos pxd [cm x mbar]
    propaga_pd = Propagacion_errores(
    formula = 'p_0*d_1', 
    variables = [('p_0', datos['presion']),
                 ('d_1', datos['distancia'])],
    errores = np.array([.02,#datos['error_presion'],
                        datos['error_distancia']]).reshape(-1,1)
    )
    pd, e_pd = propaga_pd.fit()
    print(pd/10, e_pd/10)

    # En centímetros
    pds.append(pd/10)
    errores_pds.append(e_pd/10)

    # Agrego al error de la ruptura el error del eje
    errores_rupturas.append(datos['error_ruptura'])

# Transformo a arrays
rupturas = np.array(rupturas).reshape(-1,1)
errores_rupturas = np.array(errores_rupturas).reshape(-1,1)
pds = np.array(pds).reshape(-1,1)
errores_pds = np.array(errores_pds).reshape(-1,1)

# Corto los datos
d = 5
rupturas  = rupturas[d:]
errores_rupturas = errores_rupturas[d:]
pds = pds[d:]
errores_pds  = errores_pds[d:]

# Hago el ajuste
# p_a_0, p_a_1, p_a_2 = 699, 0.001, 12
p_a_0, p_a_1, p_a_2 = 500, .0000482, 11.458
# p_a_0, p_a_1, p_a_2 = 1,1,1
ajuste = Ajuste(x = pds, y = rupturas, cov_y = errores_rupturas) # TODO: definir bien el error
ajuste.fit(
formula = formula_de_paschen,
p0 = [p_a_0, p_a_1, p_a_2], 
bounds = ([500,0.000001,0],[800,.0000484,np.inf]), 
method = 'dogbox'
)
# Chequeo los parámetros y la bondad
ajuste.parametros, np.sqrt(np.diag(ajuste.cov_parametros))
ajuste.bondad()

# Tomo una tira auxiliar para graficar el ajuste y hago las bandas de error del mismo
pds_auxiliar = np.linspace(pds[0], pds[-1],1000)
franja_error = Propagacion_errores(
        variables = [('a_0', ajuste.parametros[0]), ('a_1', ajuste.parametros[1]), ('a_2', ajuste.parametros[1])], 
        errores = ajuste.cov_parametros, 
        formula = formula_de_paschen, 
        dominio = pds_auxiliar
        ).fit()[1]

plt.figure()
plt.scatter(x = pds, y = rupturas, s = 5, color = 'black', label = 'Datos')
plt.errorbar(
    pds.reshape(-1), 
    rupturas.reshape(-1), 
    yerr = errores_rupturas.reshape(-1), 
    xerr = errores_pds.reshape(-1),
    marker = '.', 
    fmt = 'None', 
    capsize = 1.5, 
    color = 'black', 
    label = 'Error de los datos')
plt.plot(pds_auxiliar, func(pds_auxiliar, *ajuste.parametros), 'r-', label = 'Ajuste', alpha = .5)
plt.plot(pds_auxiliar, func(pds_auxiliar, *ajuste.parametros) + franja_error, '--', color = 'green', label = 'Error del ajuste')
plt.plot(pds_auxiliar, func(pds_auxiliar, *ajuste.parametros) - franja_error, '--', color = 'green')
plt.fill_between(
    pds_auxiliar.reshape(-1), 
    func(pds_auxiliar, *ajuste.parametros).reshape(-1) - franja_error.reshape(-1),
    func(pds_auxiliar, *ajuste.parametros).reshape(-1) + franja_error.reshape(-1),
    facecolor = "gray", 
    alpha = 0.3)
plt.xlabel(xlabel = 'Presión x Distancia [cm x mbar]')
plt.ylabel(ylabel = 'Tensión de ruptura [V]')
plt.ylim([250, 600])
plt.grid(visible = True)
plt.legend()
plt.tight_layout()
plt.show(block = False)       

#  ####################################################################################################################################################################
#  ####################################################################################################################################################################
#  ####################################################################################################################################################################
# # ARREGLANDO TENSION GLOW MEDICIONES AIRE Y HE
# R_0, e_R_0 = 149.2, 0.2  # TODO MEDIR DE NUEVO RESISTENCIAS
# R_1, e_R_1 = 55325, 6
# R_2, e_R_2 = 56100000, 458000
# R_3, e_R_3 = 29.932, 4
# formula_tension_glow = '-t_r_0 + (t_r_1)*(r3_3/r1_2 + 1)'
# formula_corriente_t =  't_r_0/r0_2 +  t_r_1/r1_3'
# # t_0  = time.time()
# for i in range(1, 5):
#     fname = os.path.join(input_path + os.path.normpath(f'/medicion_{i}.pkl'))
#     datos_i = load_dict(fname = fname)
#     datos_i['tension_glow'] = -datos_i['tension_r0'] + datos_i['tension_r1']*(R_2/R_1 + 1)
#     datos_i['error_tension_r0'] = 0.000035*datos_i['tension_r0']+0.000005*10
#     datos_i['error_tension_r1'] = 0.000040*datos_i['tension_r1']+0.000007*1 
#     error_corriente_t = []
#     for j in range(len(datos_i['tension_r0'])):
#         propagacion = Propagacion_errores(
#         formula = formula_corriente_t,
#         variables = [
#                     ('t_r_0', datos_i['tension_r0'][j]), # tensión r_0
#                     ('t_r_1', datos_i['tension_r1'][j]), # tensión r_1
#                     ('r0_2', R_0), # r_1
#                     ('r1_3', R_1), # r_2
#                     ],
#         errores = np.array([
#             datos_i['error_tension_r0'][j],
#             datos_i['error_tension_r0'][j],
#             e_R_0,
#             e_R_1]).reshape(-1, 1)
#         )
#         datos_i['corriente_t'][j] = propagacion.fit()[0]
#         error_corriente_t.append(propagacion.fit()[1])
#     datos_i['error_corriente_t'] = np.array(error_corriente_t).reshape(-1,1)
#     # error_glow = []
#     # for j in range(len(datos_i['tension_r0'])):
#     #     propagacion = Propagacion_errores(
#     #     formula = formula_tension_glow,
#     #     variables = [('t_r_0', datos_i['tension_r0'][j]), # tensión r_0
#     #                 ('t_r_1', datos_i['tension_r1'][j]), # tensión r_1
#     #                 ('r1_2', R_1), # r_1
#     #                 ('r3_3', R_2), # r_2
#     #                 ],
#     #     errores = np.array([
#     #         datos_i['error_tension_r0'][j],
#     #         datos_i['error_tension_r0'][j],
#     #         e_R_1,
#     #         e_R_2]).reshape(-1, 1)
#     #     )
#     #     error_glow.append(propagacion.fit()[1])
#     # datos_i['error_glow'] = np.array(error_glow).reshape(-1,1)
#     save_dict(fname = fname, dic = datos_i, rewrite = True)

# for i in range(1, 5):
#     fname = os.path.join(input_path + os.path.normpath(f'/medicion_helio_{i}.pkl'))
#     datos_i = load_dict(fname = fname)
#     datos_i['tension_glow'] = -datos_i['tension_r0'] + datos_i['tension_r1']*(R_2/R_1 + 1)
#     datos_i['error_tension_r0'] = 0.000035*datos_i['tension_r0']+0.000005*10
#     datos_i['error_tension_r1'] = 0.000040*datos_i['tension_r1']+0.000007*1 
#     error_corriente_t = []
#     for j in range(len(datos_i['tension_r0'])):
#         propagacion = Propagacion_errores(
#         formula = formula_corriente_t,
#         variables = [
#                     ('t_r_0', datos_i['tension_r0'][j]), # tensión r_0
#                     ('t_r_1', datos_i['tension_r1'][j]), # tensión r_1
#                     ('r0_2', R_0), # r_1
#                     ('r1_3', R_1), # r_2
#                     ],
#         errores = np.array([
#             datos_i['error_tension_r0'][j],
#             datos_i['error_tension_r0'][j],
#             e_R_0,
#             e_R_1]).reshape(-1, 1)
#         )
#         datos_i['corriente_t'][j] = propagacion.fit()[0]
#         error_corriente_t.append(propagacion.fit()[1])
#     datos_i['error_corriente_t'] = np.array(error_corriente_t).reshape(-1,1)
#     # error_glow = []
#     # for j in range(len(datos_i['tension_r0'])):
#     #     propagacion = Propagacion_errores(
#     #     formula = formula_tension_glow,
#     #     variables = [('t_r_0', datos_i['tension_r0'][j]), # tensión r_0
#     #                 ('t_r_1', datos_i['tension_r1'][j]), # tensión r_1
#     #                 ('r1_2', R_1), # r_1
#     #                 ('r3_3', R_2), # r_2
#     #                 ],
#     #     errores = np.array([
#     #         datos_i['error_tension_r0'][j],
#     #         datos_i['error_tension_r0'][j],
#     #         e_R_1,
#     #         e_R_2]).reshape(-1, 1)
#     #     )
#     #     error_glow.append(propagacion.fit()[1])
#     # datos_i['error_glow'] = np.array(error_glow).reshape(-1,1)
#     save_dict(fname = fname, dic = datos_i, rewrite = True)
# for i in range(1, 26):
#     fname = os.path.join(input_path + os.path.normpath(f'/n_medicion_paschen_{i}.pkl'))
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
# # # ARREGLANDO PASCHEN
# # for i in range(1, 26):
# #     fname = os.path.join(input_path + os.path.normpath(f'/medicion_paschen_{i}.pkl'))
# #     datos_i = load_dict(fname = fname)
# #     datos_i['ruptura'] = datos_i['tension_glow'].max()
# #     datos_i['error_ruptura'] = datos_i['error_glow'][datos_i['tension_glow'].argmax()]
# #     save_dict(fname = fname, dic = datos_i, rewrite = True)

# =============================
# ESCALAS LOG
# =============================

fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (8,4))
dic = {1:(10,7),2:(19,9),3:(16,15), 4:(26,7)}
for c, i in zip(colors_rgb, np.arange(1, num + 1 , 1)): # for i in range(1, num + 1):
    fname = os.path.join(input_path + os.path.normpath(f'/medicion_{i}.pkl'))
    datos_leidos = load_dict(fname = fname)
    pr = '(' + str(datos_leidos['presion']).replace('.',',') + r' $\pm$' +' 0,02)'
    corriente = 1000*datos_leidos['corriente_t']
    ax[0].plot(corriente, datos_leidos['tension_glow'], '-.', color = c, label = f'{pr} mbar')
    ini_flechas_x = [corriente[dic[i][0]],corriente[len(corriente)-dic[i][1]]]
    ini_flechas_y = [datos_leidos['tension_glow'][dic[i][0]],datos_leidos['tension_glow'][len(datos_leidos['tension_glow'])-dic[i][1]]]
    fin_flechas_x = [corriente[dic[i][0] + 1],corriente[len(corriente)-dic[i][1] + 1]]
    fin_flechas_y = [datos_leidos['tension_glow'][dic[i][0] + 1],datos_leidos['tension_glow'][len(datos_leidos['tension_glow'])-dic[i][1] + 1]]
    for X_f, Y_f, X_i, Y_i in zip(fin_flechas_x, fin_flechas_y, ini_flechas_x, ini_flechas_y):
        ax[0].annotate(text = "",
        xy = (X_f,Y_f), 
        xytext = (X_i,Y_i),
        arrowprops = dict(arrowstyle = "->"), color = c,
        size = 25
        )
    ax[0].errorbar(
    corriente, 
    datos_leidos['tension_glow'], 
    yerr = datos_leidos['error_glow'].reshape(-1), 
    xerr = 1000*datos_leidos['error_corriente_t'].reshape(-1), 
    marker = '.', 
    fmt = 'None', 
    # capsize = 1, 
    color = c)
ax[0].grid(visible = True)
ax[0].set_xlabel('Intensidad de corriente [mA]')
ax[0].set_xscale('log')
ax[0].set_ylim(280,400)
ax[0].set_ylabel('Tension [V]')
ax[0].legend()
ax[0].set_title(label = 'Aire')

dic = {1:(24,12),2:(55,12),3:(66,16), 4:(53,12)}
for c, i in zip(colors_rgb, np.arange(1, num + 1 , 1)): # for i in range(1, num + 1):
    fname = os.path.join(input_path + os.path.normpath(f'/medicion_helio_{i}.pkl'))
    datos_leidos = load_dict(fname = fname)
    datos_leidos['unidades']
    pr = '(' + str(datos_leidos['presion']).replace('.',',') + r' $\pm$' +' 0,02)'
    corriente = 1000*datos_leidos['corriente_t']
    ax[1].plot(corriente, datos_leidos['tension_glow'], '-.',label = f'{pr} mbar', color = c)
    ini_flechas_x = [corriente[dic[i][0]],corriente[len(corriente)-dic[i][1]]]
    ini_flechas_y = [datos_leidos['tension_glow'][dic[i][0]],datos_leidos['tension_glow'][len(datos_leidos['tension_glow'])-dic[i][1]]]
    fin_flechas_x = [corriente[dic[i][0] + 1],corriente[len(corriente)-dic[i][1] + 1]]
    fin_flechas_y = [datos_leidos['tension_glow'][dic[i][0] + 1],datos_leidos['tension_glow'][len(datos_leidos['tension_glow'])-dic[i][1] + 1]]
    for X_f, Y_f, X_i, Y_i in zip(fin_flechas_x, fin_flechas_y, ini_flechas_x, ini_flechas_y):
        ax[1].annotate(text = "",
        xy = (X_f,Y_f), 
        xytext = (X_i,Y_i),
        arrowprops = dict(arrowstyle = "->"), color = c,
        size = 25
        )
    ax[1].errorbar(
    corriente, 
    datos_leidos['tension_glow'], 
    yerr = datos_leidos['error_glow'].reshape(-1), 
    xerr = 1000*datos_leidos['error_corriente_t'].reshape(-1), 
    marker = '.', 
    fmt = 'None', 
    # capsize = 1, 
    color = c)
ax[1].grid(visible = True)
ax[1].set_xlabel('Intensidad de corriente [mA]')
ax[1].set_xscale('log')
ax[1].set_ylabel('Tension [V]')
ax[1].legend()
ax[1].set_title(label = 'Helio')
fig.tight_layout()
fig.show()
# fig.savefig(fname = os.path.join(output_path + os.path.normpath('/informe/mediciones_VI_log.svg')))