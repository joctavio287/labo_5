from sklearn.preprocessing import PolynomialFeatures
from scipy.optimize import curve_fit
from herramientas.errores.propagacion import Propagacion_errores
import numpy as np, matplotlib.pyplot as plt

class Ajuste:
    modelos = ['regresion_lineal', 'curve_fit']
    estilos_graficos = ['ajuste_1', 'ajuste_2', 'slider', 'errores'] #TODO: slider no implementado

    def __init__(self, x: np.array, y: np.array, cov_y: np.array = None) -> None:
        '''
        INPUT: 
        -x, y: np.arrays: son los datos para ajustar. Si hay más de una variable independiente deben
        disponerse en unsolo array cuya dimensión deberá ser X.shape = (numero_de_datos, numero_de_v
        ariables).

        -cov_y: np.array: matriz de covarianza de los datos y. También es posible pasar un array con
        las desviaciones estándar de cada dato. Si este es el caso, la matriz de covarianza se const
        ruira automáticamente. De no haber errores, por defecto se toma la identidad.
        '''
        # Transforma arrays flattens en vectores columna
        if x.ndim == 1:
            self.x = x.reshape(-1,1)
        else:
            self.x = x
        if y.ndim == 1:
            self.y = y.reshape(-1,1)
        else:
            self.y = y
        
        # Determina la matriz de covarianza y el sigma (desviaciones estándar)       
        if cov_y is None:
            # si no se pasa nada, por defecto queda la identidad
            self.cov_y = np.identity(len(self.y))
            self.sigma_y = np.sqrt(np.diag(cov_y))

        elif cov_y.ndim != 2:
            raise ValueError("El array que se paso no tiene las dimensiones requeridas.\n Debería tener dimensiones (len(y), len(y)) ó (len(y), 1) si se pasan desviaciones estándar.")
        
        if cov_y.shape[1] == 1:
            self.cov_y = np.diag(cov_y.reshape(-1)**2)
            self.sigma_y = cov_y
        else:
            self.cov_y = cov_y
            self.sigma_y = np.sqrt(np.diag(cov_y))

        # Hiperparámetros que se definen dentro de la clase
        self.y_modelo = None
        self.vander = None
        self.parametros = None
        self.cov_parametros = None
        self.r = None
        self.R2 = None
        self.chi_2 = None
        self.expr = None
    
    # TODO: implementar distinta forma de llamar 
    # @classmethod # Empleado.foramto_string('Juan-Castro-30000') crea la instancia en ese formato
    # def formato_string(cls, emp_str):
    #   nombre, apellido, sueldo = emp_str.split('-')
    #   sueldo = int(sueldo)
    #   return cls(nombre, apellido, sueldo)
    
    def __str__(self):
        '''
        Si se ejecuta 'print(instancia de la clase)' ó 'str(instancia)' se correrá esta función. 
        Sirve para dar información respecto a lo que ya corrio la instancia.
        '''
        texto  = f'Ajuste lineal:\n -Parámetros de ajuste: {self.parametros}.\n'
        texto += f' -Matriz de covarianza de los parámetros: {self.cov_parametros}.\n'
        self.bondad()
        return texto

    def fit(self, modelo:str, **kwargs):
        '''
        Actualiza los coeficientes del ajuste y su matriz de covarianza acorde al modelo especificad
        o.
        INPUT:
        modelo: str: se deberá elegir entre 'regresion_lineal' ó 'curve_fit' de la librería scipy. S
        i se elige la última, se deberá incluir, además, el siguiente parámetro.
        
        expr: str: función de ajuste expresada como str. Es necesario que los parámetros tomen la fo
        rma 'a_i' con i = 0, 1, 2, 3... Se debe respetar el orden en las varibles y para funciones e
        speciales usar la libreria numpy. Ejemplo:
        >> expr = 'a_0 + a_1*x**2 + np.sin(a_2*x)**3'

        **kwargs
        REGRESIÓN LINEAL:
        n: int: orden de la regresión.
        ordenada: bool: si incluir o no la ordenada al origen como parámetro a determinar. Si ordena
        da=False entonces se asume que es nula.

        CURVE FIT:https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html
        p0: np.array: valores iniciales de los parámetros. Deben incluirse todos, de no pasar este p
        arámetro se tomara np.ones(shape = self.parametros.shape).
        
        bounds: 2-tuple or array_like: una tupla en la que en el primer valor se incluye un np.array 
        con la misma dimensión que los parámetros en la que irán las cotas inferiores; en el segundo
        valor, las superiores. Para que no haya cota usar, respectivamente, -np.inf y np.inf.
        
        method: {'lm', 'trf', 'dogbox'}: método usado para optimizar. Default es lm para problemas s
        in restricciones y trf si se proveen bounds. lm no funcionará si el numero de observaciones 
        es menor que el número de variables. En ese caso usar trf o dogbox.
        
        absolute_sigma: bool: si es True, la desviación estándar es usada en sentido absoluto y la m
        atriz de covarianza de los parámetros (self.cov_parametros) refleja valores absolutos. Si es
        False (default) sólo valores relativos de la desviación importan, en ese caso la matriz de c
        ovarianza se basa en un escaleo por un factor constante. 
        This constant is set by demanding that the reduced chisq for the optimal parameters popt whe
        n using the scaled sigma equals unity. In other words, sigma is scaled to match the sample v
        ariance of the residuals after the fit. Default is False. Mathematically, pcov(absolute_sigm
        a=False) = pcov(absolute_sigma=True) * chisq(popt)/(M-N).
        '''        
        # Chequea que el modelo especificado exista
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

            # Los datos predecidos por el modelo
            self.y_modelo = np.dot(self.vander, self.parametros)

        elif modelo == 'curve_fit':
            self.expr = kwargs['expr']

            # Define una función en base a la expresión especificada en los parámetros
            func = Ajuste.define_func(self.expr)
            
            # Se elimina la variable para prevenir que la lea curve_fit (no es un parámetro)
            kwargs.pop('expr')
            
            if self.x.shape[1]!=1:
                self.parametros, self.cov_parametros = curve_fit(
                f = func,
                xdata = self.x,
                ydata = self.y.reshape(-1),
                sigma = self.cov_y, # se puede pasar sigma en lugar de cov_y
                **kwargs)
            else:
                self.parametros, self.cov_parametros = curve_fit(
                f = func,
                xdata = self.x.reshape(-1),
                ydata = self.y.reshape(-1),
                sigma = self.cov_y, # se puede pasar sigma en lugar de cov_y
                **kwargs) 
            
            # Los datos predecidos por el modelo
            self.y_modelo = func(self.x, *self.parametros)
    
    def regresion_lineal(self, n:int = 1, ordenada:bool = False):
        '''
        Realiza una regresión lineal y actualiza la matriz de Vandermonde
        INPUT:
        n: int: orden de la regresión.
        ordenada: bool: si incluir o no la ordenada al origen como parámetro a determinar. Si ordena
        da=False entonces se asume que es nula.

        OUTPUT:
        parametros: np.array: los parámetros del ajuste.
        cov_parametros: np.array: la matriz de covarianza de los parámetros del ajuste.
        '''        
        # Matriz de Vandermonde:
        pfeats = PolynomialFeatures(degree = n, include_bias = ordenada)
        vander = pfeats.fit_transform(self.x)
        self.vander = vander.copy()
        
        # Calculos auxilares:
        inversa_cov = np.linalg.inv(self.cov_y)
        auxiliar = np.linalg.inv(np.dot(np.dot(vander.T, inversa_cov), vander))

        # Parámetros [At.Cov-1. A]-1.At.Cov-1.y = [a_0, a_1, ..., a_n]t
        parametros = np.dot(np.dot(np.dot(auxiliar, vander.T), inversa_cov), self.y) 

        # Matriz de covarianza de los parámetros [At.Cov-1.A]-1
        cov_parametros = np.linalg.inv(np.dot(vander.T, np.dot(inversa_cov, vander)))
           
        return parametros, cov_parametros
    
    def bondad(self):
        '''
        Calcula coeficientes para determinar la bondad del ajuste: r, R^2, Xi^2 y Xi^2 reducido.

        - La correlación lineal (r) me habla de qué tan determinadas están unas variables respecto a
        las otras. r(x,y) := cov(x,y)/sigma(x)*sigma(y) y -1 < r < 1. 
        Si r = 1 las variables están correlacionadas (i.e: cuando una crece, la otra también), si
        r = -1 las variables están anticorrelacionadas (i.e: cuando una crece, la otra decrece); 
        si r=0 no están correlacionadas.
        Lo que devuelve este método es la matriz de correlación entre las variables x e y. Es decir
        r(x,x), r(x,y) = r(y,x) y r(y,y). Obviamente el dato interesante es el simétrico (es el que se
        reporta).

        - El coeficiente de determinación (R^2) determina la variación de los resultados que puede explicarse
        por el modelo. Básicamente da una noción de si el ajuste cae o no dentro del error de los datos.
        En el caso de un ajuste lineal, el r(x,y) coincide con el R^2.
        
        - El estadístico Chi cuadrado sirve para testear que los datos siguen una distribución gaussiana con
        media en los datos adquiridos. La idea es contrastar su valor con su esperanza. Un valor alto podría 
        indicar errores subestimados o que los datos no siguen la distribución especificada. Mientras que un
        valor bajo podría indicar que los errores están sobre estimados. Se tiene que 
        E(Xi^2) = (# de datos - # parametros ajustados) 
        Sigma(Xi^2) = sqrt(2(# de datos - # parametros ajustados)).
        
        - La idea con el Chi cuadrado reducido es la misma, pero su valor es más acotado, puesto que va norma
        lizado por su esperanza
        E(Xi_r^2) ~ 1
        Sigma(Xi_r^2) = sqrt(2)/(# de datos - # parametros ajustados)**(3/2).
        '''    
        # Matriz de correlación lineal de los datos
        self.r = np.corrcoef(self.x.flatten(), self.y.flatten())

        # Coeficiente de determinación 1 - sigma_r**2/sigma_y**2
        sigma_r = self.y - self.y_modelo
        self.R2 = float(1 - np.dot(sigma_r.T, sigma_r)/np.dot(self.sigma_y.T, self.sigma_y))

        # Chi^2
        self.chi_2 = np.sum(((self.y - self.y_modelo)/self.sigma_y)**2, axis = 0)
        expected_chi_2 = len(self.y) - len(self.parametros)
        variance_chi_2 = np.sqrt(2*expected_chi_2)
        self.reduced_chi_2 = self.chi_2/(len(self.y)-len(self.parametros))

        # Printea la bondad
        texto  = f'Bondad del ajuste:\n -Correlación lineal: {np.matrix(self.r)}.\n'
        texto += f' -R cuadrado: {self.R2}.\n'
        texto += f' -Chi cuadrado: {self.chi_2}. Expected {expected_chi_2} ± {variance_chi_2}'
        texto += f' -Chi cuadrado reducido: {self.reduced_chi_2}.'
        print(texto)

    def graph(self, estilo:str, label_x: str, label_y: str):
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
            fig.tight_layout()
            fig.show()

        elif estilo == 'ajuste_1':
            #tira auxiliar para graficar
            x_auxiliar = np.linspace(self.x[0], self.x[-1], len(self.x)*1000).reshape(-1)
            y_auxiliar = Ajuste.define_func(self.expr)(x_auxiliar, *self.parametros).reshape(-1)
            fig, ax = plt.subplots(nrows = 1, ncols = 1)
            ax.scatter(x = self.x, y = self.y, s = 5, color = 'black', label = 'Datos')
            ax.errorbar(self.x, self.y, yerr = self.sigma_y.reshape(-1), marker = '.', fmt = 'None', capsize = 1.5, color = 'black', label = 'Error de los datos')
            ax.plot(x_auxiliar, y_auxiliar, 'r-', label = 'Ajuste', alpha = .5)
            ax.set_xlabel(xlabel = label_x)
            ax.set_ylabel(ylabel = label_y)
            ax.grid()
            ax.legend()
            fig.tight_layout()
            fig.show()

        elif estilo == 'ajuste_2':            
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
        Cuenta el numero de variables de la forma 'a_i' y defune una función con 'x' como 
        variable independiente y los 'a_i' como coeficientes.
        INPUT:
        expr: str: formula de la función a fittear.
        '''
        n_vars = [f'a_{i}' for i in range(expr.count('a_'))]
        n_vars = ['x_'] + n_vars
        args = ', '.join(n_vars)
        expresion = expr
        exec(f'def func({args}):\n    import numpy as np\n    return {expresion}', locals())
        return locals()['func']

if __name__ == '__main__':

    # EJEMPLO 1D:
    x = np.linspace(0,100,40)
    y = 1 + 2*np.sin(0.1*x)*np.exp(-.5*x)
    sigma = .005
    signal = np.random.normal(y, sigma, size = y.shape)
    expr = 'a_0 + a_1*np.sin(.1*x_)*np.exp(a_2*x_)'
    aj = Ajuste(x, signal, cov_y = np.array([sigma for i in x]).reshape(-1,1))
    aj.fit(modelo='curve_fit', expr = expr, bounds =([-10, 0, -np.inf] , [10, 10, 0]))
    print(aj)
    aj.graph(estilo = 'ajuste_1', label_x = 'Tiempo [s]', label_y = r'Tension [$\propto V$]')
    aj.graph(estilo = 'errores', label_x = 'Tiempo [s]', label_y = r'Tension [$\propto V$]')


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
    