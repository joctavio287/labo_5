from sklearn.preprocessing import PolynomialFeatures
import numpy as np, matplotlib.pyplot as plt
from scipy import interpolate
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit
import scipy.stats as st
from herramientas.errores.propagacion import Propagacion_errores

# Clase para hacer ajustes. A partir de esta clase vamos a heredar Regresion_Lineal y No_lineal:
class Ajuste:
    modelos = ['regresion_lineal', 'curve_fit']
    estilos_graficos = ['ajuste_1', 'ajuste_2', 'slider', 'errores']
    def __init__(self, x: np.array, y: np.array, cov_y: np.array = None) -> None:
        '''
        INPUT: 
        -x, y: np.arrays: son los datos para ajustar. Se asume que hay una sola variable independiente.
        -cov_y: np.array: matriz de covarianza de los datos en y. De no haber errores, por defecto se toma la identidad.
        -n: int: orden del ajuste lineal.
        -ordenada: bool: si queremos o no obtener la ordenada del ajuste.
        '''
        # Get righy dimensions
        if x.ndim == 1:
            self.x = x.reshape(-1,1)
        else:
            self.x = x
        if y.ndim == 1:
            self.y = y.reshape(-1,1)
        else:
            self.y = y
        if cov_y is not None and cov_y.ndim != 2:
            raise ValueError("Passed array is not of the right shape. It should be a (len(y), len(y)) shaped array.")
        self.cov_y = cov_y
        self.sigma_y = None
        self.y_modelo = None
        self.vander = None
        self.parametros = None
        self.cov_parametros = None
        self.r = None
        self.R2 = None
        self.chi_2 = None
        self.expr = None
    
    # @classmethod # Empleado.foramto_string('Juan-Castro-30000') crea la instancia en ese formato
    # def formato_string(cls, emp_str):
    #   nombre, apellido, sueldo = emp_str.split('-')
    #   sueldo = int(sueldo)
    #   return cls(nombre, apellido, sueldo)
    
    def __str__(self):
        texto  = f'Ajuste lineal:\n -Parámetros de ajuste: {self.parametros}.\n'
        texto += f' -Matriz de covarianza de los parámetros: {self.cov_parametros}.\n'
        texto += f' -Bondad del ajuste:\n  *Correlación lineal: {self.r}'
        print(texto)

    def fit(self, modelo:str, **kwargs):
        '''
        OUTPUT: 
        -actualiza los coeficientes del ajuste y su matriz de covarianza acorde al modelo.
        Si se utiliza regresión_lineal se deberá agregar (opcional) los hiperparámetros 'n' y 'ordenada'.
        El primero da el orden del ajuste lineal, el segundo si se quiere la ordenada o no.
        En caso de utilizar curve_fit se debera especificar el parametro f (la función a fitear):
        ejemplo: 
        def f(x, t_0, a, g, c):
            return np.array([lambda x: a*np.abs(x-t_0)**(g) + c])
        Y de querer los siguientes parámetros:
        bounds: tupla de dos elementos (lower y upper). Cada elemento es un array con la longitud de los parametros
        p0: el initial guess
        absolute_sigma = False: # usar sólo en caso de que este en las mismas unidades (es decir no hay repetidos valores sobre la misma medicion. GGOOGLEAR)
        check_finite = True: # chequea que no haya nans o infs
        '''        
        # Determina la matriz de covarianza y el sigma:       
        if self.cov_y is None:
            # es la identidad:
            self.cov_y = np.identity(len(self.y)) 
        self.sigma_y = np.sqrt(np.diag(self.cov_y).reshape(-1,1))

        # Chequea que el modelo especificado exista:
        if modelo in Ajuste.modelos:
            pass
        else:
            raise Exception(f"No existe el modelo {modelo}")

        if modelo == 'regresion_lineal':
            self.parametros, self.cov_parametros = self.regresion_lineal(
            x = self.x,
            y = self.y,
            cov_y = self.cov_y,
            **kwargs)

            # los datos predecidos por el modelo:
            self.y_modelo = np.dot(self.vander, self.parametros)

        elif modelo == 'curve_fit':
            # Define function from expression
            self.expr = kwargs['expr']
            func = Ajuste.define_func(self.expr)
            
            # In order to prevent from reading it at the kwargs of curve_fit
            kwargs.pop('expr')
            
            if self.x.shape[1]!=1:
                self.parametros, self.cov_parametros = curve_fit(
                f = func,
                xdata = self.x,
                ydata = self.y.reshape(-1),
                sigma = self.cov_y,
                **kwargs) # this could be cov matrix or sigma
            else:
                self.parametros, self.cov_parametros = curve_fit(
                f = func,
                xdata = self.x.reshape(-1),
                ydata = self.y.reshape(-1),
                sigma = self.cov_y, # this can be cov matrix or sigma
                **kwargs) 
            
            # los datos predecidos por el modelo:
            self.y_modelo = func(self.x, *self.parametros)
    
    def regresion_lineal(self, x:np.array, y:np.array, cov_y:np.array, n:int = 1, ordenada:bool = False):
        '''
        OUTPUT:
        parametros:np.array (shape...
        cov_parametros:np.array(shape...
        '''        
        # matriz de Vandermonde:
        pfeats = PolynomialFeatures(degree = n, include_bias = ordenada)
        vander = pfeats.fit_transform(x)
        self.vander = vander.copy()
        
        # calculos auxilares:
        inversa_cov = np.linalg.inv(cov_y)
        auxiliar = np.linalg.inv(np.dot(np.dot(vander.T, inversa_cov), vander))

        # parámetros [At.Cov-1. A]-1.At.Cov-1.y = [a_0, a_1, ..., a_n]t:
        parametros = np.dot(np.dot(np.dot(auxiliar, vander.T), inversa_cov), y) 

        # matriz de covarianza de los parámetros [At.Cov-1.A]-1:
        cov_parametros = np.linalg.inv(np.dot(vander.T, np.dot(inversa_cov, vander)))
           
        return parametros, cov_parametros
    
    def bondad(self):
        '''
        OUTPUT: 
        -actualiza los parámetros de bondad: r, R cuadrado, chi cuadrado y el chi cuadrado reducido.
        '''    
        # Matriz de correlación lineal de los datos:
        self.r = np.corrcoef(self.x.flatten(), self.y.flatten())

        # Coeficiente de determinación: 1 - sigma_r**2/sigma_y**2
        sigma_r = self.y - self.y_modelo
        # self.sigma_y = np.sqrt(np.diag(self.cov_y))
        self.R2 = float(1 - np.dot(sigma_r.T, sigma_r)/np.dot(self.sigma_y.T, self.sigma_y))

        # Chi^2:
        # El valor esperado para chi es len(y_data) - # de datos ajustados ± sqrt(len - #).
        # Un chi alto podría indicar error subestimado o que y_i != f(x_i)
        # Un chi bajo podría indicar error sobrestimado
        self.chi_2 = np.sum(((self.y - self.y_modelo)/self.sigma_y)**2, axis = 0)
        self.reduced_chi_2 = self.chi_2/(len(self.y)-len(self.parametros))

        # Printea la bondad:
        texto  = f'Bondad del ajuste:\n -Correlación lineal: {np.matrix(self.r)}.\n'
        texto += f' -R cuadrado: {self.R2}.\n'
        texto += f' -Chi cuadrado: {self.chi_2}.'
        texto += f' -Chi cuadrado reducido: {self.reduced_chi_2}.'
        print(texto)

    def graph(self, estilo:str, label_x: str, label_y: str, funcion:str):
        if estilo in Ajuste.estilos_graficos:
            pass
        else:
            var = ', '.join(Ajuste.estilos_graficos)
            raise Exception(f"No existe el gráfico {estilo}. Los disponibles son: {var}")

        if estilo == 'errores':
            # calculo los residuos
            residuos = self.y_modelo - self.y
            
            # creo la figura y grafico
            fig, ax = plt.subplots(nrows = 1, ncols = 1)
            eje_x = np.zeros(shape = self.y.shape)
            ax.scatter(x = self.x, y = residuos, s = 10, color = 'black', label = 'Resiudos')
            ax.plot(self.x, eje_x, color = 'black', linewidth = 0.4)
            ax.vlines(x = self.x, ymin = eje_x, ymax = eje_x + residuos, color = 'red', alpha = .8)
            ax.set_xlabel(xlabel = label_x)
            ax.set_ylabel(ylabel = label_y)
            ax.grid()
            ax.legend()
            fig.show()
        
        elif estilo == 'ajuste_1':
            #tira auxiliar para graficar
            x_auxiliar = np.linspace(self.x[0], self.x[-1], len(self.x)*10).reshape(-1)
            y_auxiliar = Ajuste.define_func(self.expr)(x_auxiliar, *self.parametros).reshape(-1)
            
            # calculo las franjas de error (sigma) teniendo en cuenta la Var(y_i, y_j) usando la formula suministrada:
            franja_error = []
            variables = [(f'a_{i}', self.parametros[i]) for i in range(self.expr.count('a_'))]
            for t in x_auxiliar: #TODO ALTAMENTE INEFICIENTE AK ES CUANDO TARDA EL CODIGO
                aux = self.expr.replace('x_', str(float(t)))
                # In order to sympy to understand
                aux = aux.replace('np.', '')
                franja_error.append(Propagacion_errores(variables = variables, errores = self.cov_parametros, formula = aux).fit()[1])

            fig, ax = plt.subplots(nrows = 1, ncols = 1)
            ax.scatter(x = self.x, y = self.y, s = 5, color = 'black', label = 'Datos')
            ax.errorbar(self.x, self.y, yerr = self.sigma_y.reshape(-1), marker = '.', fmt = 'None', capsize = 1.5, color = 'black', label = 'Error de los datos')
            ax.plot(x_auxiliar, y_auxiliar, 'r-', label = 'Ajuste', alpha = .5)
            ax.plot(x_auxiliar, y_auxiliar + franja_error, '--', color = 'green', label = 'Error del ajuste')
            ax.plot(x_auxiliar, y_auxiliar - franja_error, '--', color = 'green')
            ax.fill_between(x_auxiliar, y_auxiliar - franja_error, y_auxiliar + franja_error, facecolor = "gray", alpha = 0.3)
            ax.set_xlabel(xlabel = label_x)
            ax.set_ylabel(ylabel = label_y)
            ax.grid()
            ax.legend()
            fig.tight_layout()
            fig.show()

    @staticmethod
    def define_func(expr:str):
        '''
        Counts number of variables of the form 'a_i' and defines a function with 
        x (independent variable) and the 'a_i' coefficients.
        INPUT:
        expr:str: formula of function to be fitted
        '''
        n_vars = [f'a_{i}' for i in range(expr.count('a_'))]
        n_vars = ['x_'] + n_vars
        args = ', '.join(n_vars)
        expresion = expr
        exec(f'def func({args}):\n    import numpy as np\n    return {expresion}', globals())
        return globals()['func']

    # def calculadora_de_varianza(self, func, cov_dependientes):
    #     variables = [(globals()[f'a_{i}'], self.parameters[i]) for i in range(len(self.parameters))]
    #     valor, error  = Propagacion_errores(variables = self.parameters, formula = funcion, errores = self.cov_parametros).fit()

if __name__ == '__main__':
    # EJEMPLO CON DOMINIO 2D
    # x_1 = np.linspace(0, 100, 70).reshape(-1,1)
    # x_2 = np.linspace(0, 54, 70).reshape(-1,1)
    # X = np.concatenate([x_1, x_2], axis = 1)
    # pfeats = PolynomialFeatures(degree = 1, include_bias =True)
    # vander = pfeats.fit_transform(X)

    # y, sigma = (10 + X[:, 0]*2+ X[:, 1]*9).reshape(-1,1), .01
    # signal = np.random.normal(y, sigma, size = y.shape)
    # def func(x_:np.array, a:np.float64, b:np.float64, c:np.float64):
    #     return a*x_[:, 0] + b* x_[:, 1] + c
    # regr = Ajuste(x = X, y = y)   
    # regr.fit(modelo = 'curve_fit', f = func)
    # regr.parametros
    # regr.fit(modelo = 'regresion_lineal', ordenada = True)
    # result = np.dot(regr.vander, regr.parametros)
    # result = func(X, *regr.parametros)
    # regr.bondad()
    # final, dato = -1, 1
    # plt.scatter(X[:final,dato], signal[:final])
    # plt.plot(X[:final,dato], result[:final], color = 'red')
    # plt.show(block = False)    
    
    # EJEMPLO 1D:
    x = np.linspace(0,100,40)
    y = 1 + 2*np.sin(0.1*x)*np.exp(-.5*x)
    sigma = .01
    signal = np.random.normal(y, sigma, size = y.shape)
    # def func(x, b, a, w):
    #     return b + a*np.sin(x*w)
    expr = 'a_0 + a_1*np.sin(a_2*x_)*np.exp(a_3*x_)'
    # aux = expr
    # aux = aux.replace('x', str(float(t)))
    # variables = [(f'a_{i}', aj.parametros[i]) for i in range(aux.count('a_'))]
    # Propagacion_errores(variables = variables, errores = aj.cov_parametros, formula = aux).fit()[1]


    aj = Ajuste(x, signal)
    aj.fit(modelo='curve_fit', expr = expr, p0 = [1.6, 2.6, 6/50,-.7])
    # aj.parametros, aj.cov_parametros
    aj.bondad()
    aj.graph(estilo = 'ajuste_1', label_x = 'Tiempo [s]', label_y = r'Tension [$\propto V$]', funcion = expr)
