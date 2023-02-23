import os, matplotlib.pyplot as plt, numpy as np
from herramientas.config.config_builder import Parser, save_dict, load_dict, guardar_csv

# Importo los paths
glob_path = os.path.normpath(os.getcwd())
variables = Parser(configuration = 'glow').config()
input_path = os.path.join(glob_path + os.path.normpath(variables['input']))
output_path = os.path.join(glob_path + os.path.normpath(variables['output']))


# ==================
# Figura 2 del paper
# ==================
dic = {2:4,6:3,10:6}
fig, ax = plt.subplots(nrows = 1, ncols = 1)
for i in [2,6,10]:
    datos_ida = np.loadtxt(fname = os.path.join(input_path + os.path.normpath(f'/figuras_paper/figura2-{i}k-forward.txt')), delimiter = ';')
    datos_vuelta = np.loadtxt(fname = os.path.join(input_path + os.path.normpath(f'/figuras_paper/figura2-{i}k-reverse.txt')), delimiter = ';')
    datos = np.concatenate((datos_ida, np.flip(m = datos_vuelta, axis = 0)), axis = 0)
    ax.scatter(datos[:, 0], datos[:, 1], s = 5, label = f' R = {i} '+ r'K$\Omega$')
    ini_flechas_x, ini_flechas_y = [datos[len(datos[:, 0])//dic[i], 0],datos[len(datos[:, 0])*2//3, 0]], [datos[len(datos[:, 0])//dic[i], 1],datos[len(datos[:, 0])*2//3, 1]]
    fin_flechas_x, fin_flechas_y = [datos[len(datos[:, 0])//dic[i] + 3, 0],datos[len(datos[:, 0])*2//3+3, 0]], [datos[len(datos[:, 0])//dic[i]+3, 1],datos[len(datos[:, 0])*2//3+3, 1]]
    for X_f, Y_f, X_i, Y_i in zip(fin_flechas_x, fin_flechas_y, ini_flechas_x, ini_flechas_y):
        ax.annotate(text = "",
        xy = (X_f,Y_f), 
        xytext = (X_i,Y_i),
        arrowprops = dict(arrowstyle = "->"),#, color = c),
        size = 15
        )
ax.grid(visible = True)
ax.set_xlabel('Intensidad de corriente en la descarga [mA]')
ax.set_ylabel('Tensión entre los electrodos [V]')
fig.legend(loc = 'center right')
fig.tight_layout()
# fig.savefig(fname = os.path.join(input_path + os.path.normpath('/figura2.png')))
fig.show()

# ==================
# Figura 3 del paper
# ==================
dic = {2:4,6:5,10:6}
fig, ax = plt.subplots(nrows = 1, ncols = 1)
for i in [2,6,10]:
    datos_ida = np.loadtxt(fname = os.path.join(input_path + os.path.normpath(f'/figuras_paper/figura3-{i}k-forward.txt')), delimiter = ';')
    datos_vuelta = np.loadtxt(fname = os.path.join(input_path + os.path.normpath(f'/figuras_paper/figura3-{i}k-reverse.txt')), delimiter = ';')
    datos = np.concatenate((datos_ida, np.flip(m = datos_vuelta, axis = 0)), axis = 0)
    ax.scatter(datos[:, 0], datos[:, 1], s = 5)
    ax.plot(datos[:, 0], datos[:, 1], label = f' R = {i} '+ r'K$\Omega$')
    ini_flechas_x, ini_flechas_y = [datos[len(datos[:, 0])//dic[i], 0],datos[len(datos[:, 0])*2//3, 0]], [datos[len(datos[:, 0])//dic[i], 1],datos[len(datos[:, 0])*2//3, 1]]
    fin_flechas_x, fin_flechas_y = [datos[len(datos[:, 0])//dic[i] + 3, 0],datos[len(datos[:, 0])*2//3+3, 0]], [datos[len(datos[:, 0])//dic[i]+3, 1],datos[len(datos[:, 0])*2//3+3, 1]]
    for X_f, Y_f, X_i, Y_i in zip(fin_flechas_x, fin_flechas_y, ini_flechas_x, ini_flechas_y):
        ax.annotate(text = "",
        xy = (X_f,Y_f), 
        xytext = (X_i,Y_i),
        arrowprops = dict(arrowstyle = "->"),#, color = c),
        size = 15
        )
ax.grid(visible = True)
ax.set_xlabel('Tensión entre los electrodos [V]')
ax.set_ylabel(r'Resistencia del gas [K$\Omega$]')
fig.legend(loc = 'upper right')
fig.tight_layout()
fig.show()
# fig.savefig(os.path.join(input_path + os.path.normpath('/figura3.png')))

# ==================
# Figura 4 del paper
# ==================
dic = {0.2:3,0.6:4,1.0:3}
fig, ax = plt.subplots(nrows = 1, ncols = 1)
for i in [0.2,0.6,1.0]:
    datos_ida = np.loadtxt(fname = os.path.join(input_path + os.path.normpath(f'/figuras_paper/figura4-{i}-forward.txt')), delimiter = ';')
    datos_vuelta = np.loadtxt(fname = os.path.join(input_path + os.path.normpath(f'/figuras_paper/figura4-{i}-reverse.txt')), delimiter = ';')
    datos = np.concatenate((datos_ida, np.flip(m = datos_vuelta, axis = 0)), axis = 0)
    ax.scatter(datos[:, 0], datos[:, 1], s = 5,  label = f' P = {i} mbar')
    ax.plot(datos[:, 0], datos[:, 1])
    ini_flechas_x, ini_flechas_y = [datos[len(datos[:, 0])//dic[i], 0],datos[len(datos[:, 0])*2//3, 0]], [datos[len(datos[:, 0])//dic[i], 1],datos[len(datos[:, 0])*2//3, 1]]
    fin_flechas_x, fin_flechas_y = [datos[len(datos[:, 0])//dic[i] + 3, 0],datos[len(datos[:, 0])*2//3+3, 0]], [datos[len(datos[:, 0])//dic[i]+3, 1],datos[len(datos[:, 0])*2//3+3, 1]]
    for X_f, Y_f, X_i, Y_i in zip(fin_flechas_x, fin_flechas_y, ini_flechas_x, ini_flechas_y):
        ax.annotate(text = "",
        xy = (X_f,Y_f), 
        xytext = (X_i,Y_i),
        arrowprops = dict(arrowstyle = "->"),#, color = c),
        size = 15
        )
ax.grid(visible = True)
ax.set_xlabel('Intensidad de corriente en la descarga [mA]')
ax.set_ylabel('Tensión entre los electrodos [V]')
fig.legend(loc = 'right')
fig.tight_layout()
fig.show()
# fig.savefig(os.path.join(input_path + os.path.normpath('/figura4.png')))

# ==================
# Figura 5 del paper
# ==================
fig, ax = plt.subplots(nrows = 1, ncols = 1)
for i in [0.2,0.6,1.0]:
    datos_ida = np.loadtxt(fname = os.path.join(input_path + os.path.normpath(f'/figuras_paper/figura5-{i}-forward.txt')), delimiter = ';')
    datos_vuelta = np.loadtxt(fname = os.path.join(input_path + os.path.normpath(f'/figuras_paper/figura5-{i}-reverse.txt')), delimiter = ';')
    datos = np.concatenate((datos_ida, np.flip(m = datos_vuelta, axis = 0)), axis = 0)
    # ax.scatter(datos_ida[:, 0], datos_ida[:, 1], s = 5, label = f'P = {i} mbar')
    # ax.plot(datos_ida[:, 0], datos_ida[:, 1])
    # ax.scatter(datos_vuelta[:, 0], datos_vuelta[:, 1], s = 5)
    # ax.plot(datos_vuelta[:, 0], datos_vuelta[:, 1])
    ax.scatter(datos[:, 0], datos[:, 1], s = 5, label = f'P = {i} mbar')
    ax.plot(datos[:, 0], datos[:, 1])

    ini_flechas_x, ini_flechas_y = [datos[len(datos[:, 0])//3, 0],datos[len(datos[:, 0])*2//3, 0]], [datos[len(datos[:, 0])//3, 1],datos[len(datos[:, 0])*2//3, 1]]
    fin_flechas_x, fin_flechas_y = [datos[len(datos[:, 0])//3+5, 0],datos[len(datos[:, 0])*2//3+5, 0]], [datos[len(datos[:, 0])//3+5, 1],datos[len(datos[:, 0])*2//3+5, 1]]
    for X_f, Y_f, X_i, Y_i in zip(fin_flechas_x, fin_flechas_y, ini_flechas_x, ini_flechas_y):
        ax.annotate(text = "",
        xy = (X_f,Y_f), 
        xytext = (X_i,Y_i),
        arrowprops = dict(arrowstyle = "->"),#, color = c),
        size = 15
        )
ax.grid(visible = True)
ax.set_xlabel('Tensión entre los electrodos [V]')
ax.set_ylabel(r'Resistencia del gas [K$\Omega$]')
fig.legend()#loc = 'right')
fig.tight_layout()
fig.show()
# fig.savefig(os.path.join(input_path + os.path.normpath('/figura5.png')))

# ==================
# Figura 6 del paper
# ==================
dic = {30:(3,3/2,0),40:(3,3/2,8),50:(4,4/2,10)}
fig, ax = plt.subplots(nrows = 1, ncols = 1)
for i in [30,40,50]:
    datos_ida = np.loadtxt(fname = os.path.join(input_path + os.path.normpath(f'/figuras_paper/figura6-{i}-forward.txt')), delimiter = ';')
    datos_vuelta = np.loadtxt(fname = os.path.join(input_path + os.path.normpath(f'/figuras_paper/figura6-{i}-reverse.txt')), delimiter = ';')
    datos = np.concatenate((datos_ida, np.flip(m = datos_vuelta, axis = 0)), axis = 0)
    ax.scatter(datos[:, 0], datos[:, 1], s = 5, label = f' d = {i} cm')
    # ax.plot(datos[:, 0], datos[:, 1])
    ini_flechas_x = [datos[len(datos[:, 0])//dic[i][0] + dic[i][2], 0], datos[int(len(datos[:, 0])//dic[i][1]) + dic[i][2], 0]]
    ini_flechas_y = [datos[len(datos[:, 0])//dic[i][0] + dic[i][2], 1], datos[int(len(datos[:, 0])//dic[i][1]) + dic[i][2], 1]]

    fin_flechas_x = [datos[int(len(datos[:, 0])//dic[i][0]) + dic[i][2] + 3, 0], datos[int(len(datos[:, 0])//dic[i][1]) +dic[i][2]+3, 0]]
    fin_flechas_y = [datos[int(len(datos[:, 0])//dic[i][0]) + dic[i][2] + 3, 1], datos[int(len(datos[:, 0])//dic[i][1]) +dic[i][2]+3, 1]]
    for X_f, Y_f, X_i, Y_i in zip(fin_flechas_x, fin_flechas_y, ini_flechas_x, ini_flechas_y):
        ax.annotate(text = "",
        xy = (X_f,Y_f), 
        xytext = (X_i,Y_i),
        arrowprops = dict(arrowstyle = "->"),#, color = c),
        size = 15
        )
ax.grid(visible = True)
ax.set_xlabel('Intensidad de corriente en la descarga [mA]')
ax.set_ylabel('Tensión entre los electrodos [V]')
fig.legend(loc = 'right')
fig.tight_layout()
fig.show()
# fig.savefig(os.path.join(input_path + os.path.normpath('/figura6.png')))


# ==================
# Figura 7 del paper
# ==================
fig, ax = plt.subplots(nrows = 1, ncols = 1)
for i in [30,40,50]:
    datos_ida = np.loadtxt(fname = os.path.join(input_path + os.path.normpath(f'/figuras_paper/figura7-{i}-forward.txt')), delimiter = ';')
    datos_vuelta = np.loadtxt(fname = os.path.join(input_path + os.path.normpath(f'/figuras_paper/figura7-{i}-reverse.txt')), delimiter = ';')
    datos = np.concatenate((datos_ida, np.flip(m = datos_vuelta, axis = 0)), axis = 0)
    # ax.scatter(datos_ida[:, 0], datos_ida[:, 1], s = 5, label = f'P = {i} mbar')
    # ax.plot(datos_ida[:, 0], datos_ida[:, 1])
    # ax.scatter(datos_vuelta[:, 0], datos_vuelta[:, 1], s = 5)
    # ax.plot(datos_vuelta[:, 0], datos_vuelta[:, 1])
    ax.scatter(datos[:, 0], datos[:, 1], s = 5, label = f'd = {i} cm')
    ax.plot(datos[:, 0], datos[:, 1])

    ini_flechas_x, ini_flechas_y = [datos[len(datos[:, 0])//3, 0],datos[len(datos[:, 0])*2//3, 0]], [datos[len(datos[:, 0])//3, 1],datos[len(datos[:, 0])*2//3, 1]]
    fin_flechas_x, fin_flechas_y = [datos[len(datos[:, 0])//3+5, 0],datos[len(datos[:, 0])*2//3+5, 0]], [datos[len(datos[:, 0])//3+5, 1],datos[len(datos[:, 0])*2//3+5, 1]]
    for X_f, Y_f, X_i, Y_i in zip(fin_flechas_x, fin_flechas_y, ini_flechas_x, ini_flechas_y):
        ax.annotate(text = "",
        xy = (X_f,Y_f), 
        xytext = (X_i,Y_i),
        arrowprops = dict(arrowstyle = "->"),#, color = c),
        size = 15
        )
ax.grid(visible = True)
ax.set_xlabel('Tensión entre los electrodos [V]')
ax.set_ylabel(r'Resistencia del gas [K$\Omega$]')
fig.legend()#loc = 'right')
fig.tight_layout()
fig.show()
# fig.savefig(os.path.join(input_path + os.path.normpath('/figura7.png')))    


# ==================
# Figura 8 del paper
# ==================
dic = {
1:' R = 10 '+r'K$\Omega$'+'\n P = 0.6 mbar\n d = 40 cm',
2:' R = 6 '+r'K$\Omega$'+'\n P = 0.4 mbar\n d = 40 cm',
3:' R = 6 '+r'K$\Omega$'+'\n P = 0.6 mbar\n d = 30 cm'
}
for i in range(1,4):
    fig, ax = plt.subplots(nrows = 1, ncols = 1)
    experimental = np.loadtxt(fname = os.path.join(input_path + os.path.normpath(f'/figuras_paper/figura8-{i}-experimental.txt')), delimiter = ';')
    fitted = np.loadtxt(fname = os.path.join(input_path + os.path.normpath(f'/figuras_paper/figura8-{i}-fitted.txt')), delimiter = ';')
    ax.plot(fitted[:, 0], fitted[:, 1], color = 'red', label = 'Ajuste')
    ax.scatter(experimental[:, 0], experimental[:, 1], s = 10, label = dic[i])

    ax.grid(visible = True)
    ax.set_xlabel('Tensión entre los electrodos [V]')
    ax.set_ylabel(r'Resistencia del gas [K$\Omega$]')
    fig.legend(loc = 'right')#loc = 'right')
    fig.tight_layout()
    fig.show()
    # fig.savefig(os.path.join(input_path + os.path.normpath(f'/figura8-{i}_sin_ajuste.png')))    
