import numpy as np, re
from sympy import symbols, lambdify, diff

# Sí, una lista enorme de funciones.
from sympy import log, sin, exp, cos, tan, sqrt, pi, atan, asin, acos, Abs, cot, sec, csc, sinc, acot, asec, acsc, atan2, sinh, cosh, tanh, coth, sech, csch, asinh, acosh, atanh, acoth, asech, root, Piecewise

class Propagacion_errores:

    def __init__(self, formula:str, variables: list, errores: np.array, dominio: np.array = None) -> None:
        '''
        INPUT:
        formula: str: la fórmula de la variable cuyo error se quiere propagar. Esta debe usar exclu
        sivamente la librería de numpy o sympy para funciones especiales (np.sin, cos, etc.).
        
        variables: list: es una lista de tuplas de dos elementos. Cada tupla esta compuesta por dos
        valores. El primer valor es un str cuyo símbolo representa a cada variable (se debe respeta
        r las mismas variables usadas en la fórmula).
        
        errores: np.array: array con los errores de cada variable o matriz de covarianza de las var
        iables. Se debe respetar el orden pasado en 'variables' y 'formula'. Las dimensiones del ar
        ray deben ser (cantidad_de_var, 1) o (cantidad_de_var, cantidad_de_var) si se pasa la matri
        z de covarianza.
        '''
        # Variables importantes e hiperparámetros usados dentro de la clase
        self.formula = formula
        self.variables = variables
        self.dominio = dominio
        self.valor = None
        self.error = None
        
        # Determina la matriz de covarianza:
        if errores.ndim != 2:
            raise ValueError("El array que se paso no tiene las dimensiones requeridas.\n Debería tener dimensiones (len(variables), len(variables)) ó (len(variables), 1) si se pasan desviaciones estándar.")

        if errores.shape[1] == 1:
            self.covarianza = np.diag(errores.reshape(-1)**2)
            self.errores = errores
        else:
            self.covarianza = errores
            self.errores = np.sqrt(np.diag(errores))

    def __str__(self):
        '''
        Si se ejecuta 'print(instancia de la clase)' ó 'str(instancia)' se correrá esta función. Sir
        ve para dar información respecto a lo que ya corrio la instancia.
        '''
        texto  = f'El valor obtenido es: ({self.valor} ± {self.error})'
        print(texto)
        return texto
    
    # @classmethod 
    # def without_variables(cls, dominio: np.array, formula:str, variables: list, errores: np.array)):
    #     '''
    #     Esta forma de llamar a propagación está implementada para evaluar el error de una función en
    #     una tira de datos. Por ejemplo: f(x_) = a_0*x_**2. En este caso la variable x_ está fija, pe
    #     ro corre a lo largo de muchos puntos. Mientras que a_0 es un parametro con error.
    #     # FIXME TODO: documentar bien
    #     En este tipo de llamado se reserva la variable 'x_' para ser evaluada en 'dominio'.
    #     '''
        
    #     return cls(variables = variables, errores = errores, formula = formula)

    def fit(self):
        '''
        Calcula el error propagado y el valor de la magnitud de interés.
        
        OUTPUT: tupla con dos valores.
        self.valor: np.float64: valor de la cantidad pasada en la fórmula.
        
        self.error: np.float64: error (raíz cuadrada de varianza) propagado de la cantidad pasada e
        n la fórmula.
        '''
        simbolos = [i for i,j in self.variables]
        valores = [j for i,j in self.variables]
        
        # Defino como símbolos las variables de las cuales depende la expresión:
        for sim in simbolos:
            globals()[sim] = symbols(sim, real = True)

        # Si se pasa dominio se agrega la variable x_ que es muda
        if isinstance(self.dominio, np.ndarray):
            x_ = symbols('x_', real = True)
        
        # Defino la expresión simbólica:
        self.formula = Propagacion_errores.traductor_numpy_sympy(self.formula)
        formula = eval(self.formula)
        
        # Calculo la aproximación lineal de la covarianza de el i-ésimo dato con el j-ésimo y sumo:
        derivadas_parciales = [diff(formula, eval(sim)) for sim in simbolos]
        covarianza_resultado = 0
        for i in range(len(simbolos)):
            for j in range(len(simbolos)):
                covarianza_resultado += derivadas_parciales[i]*self.covarianza[i, j]*derivadas_parciales[j]
        
        # Fórmula del error simbólico
        error_simbolico = sqrt(covarianza_resultado)

        # Convierto la expresión simbólica en un módulo numérico (numpy) para poder reemplazar:
        if isinstance(self.dominio, np.ndarray):
            lambd_err = lambdify(['x_'] + simbolos, error_simbolico, modules = ['numpy'])
            lambd_val = lambdify(['x_'] + simbolos, formula, modules = ['numpy'])
            valores = [self.dominio] + valores
            self.valor, self.error = lambd_val(*valores), lambd_err(*valores)

            # Borro las variables auxiliares que se definieron en globals()
            for sim in simbolos:
                del globals()[sim]

            return self.valor, self.error, lambd_err

    @staticmethod
    def traductor_numpy_sympy(expr:str):
        '''
        Traductor de fórmulas escritas en strings de numpy a sympy.

        INPUT:
        expr: str: expresión a traducir desde el paquete numpy al sympy.
        
        OUTPUT: 
        expr: str: misma expresión, pero escrita en formato sympy
        '''
        # Por si toma una función partida 
        if 'np.piecewise' in expr:
            
            # Extraigo rangos
            start = expr.find('[') + 1
            end = expr.find(']')
            list_of_ranges = expr[start:end].replace(' ', '').split(',')
            
            # Extraigo funciones
            start = expr.rfind('[') + 1
            end = expr.rfind(']')
            list_of_funcs_aux = expr[start:end].replace(' ', '').split(',')
            list_of_funcs = []
            for fun in list_of_funcs_aux:
                if 'lambda' in fun:
                    list_of_funcs.append(fun.split(':')[1])
                else: 
                    list_of_funcs.append(fun)
            
            # Creo los argumentos de la funcion de Sympy
            args = ''
            for range, func  in zip(list_of_ranges, list_of_funcs):
                args+=f'({func}, {range}),'
            expr = f'Piecewise({args})'
        
        expr = expr.replace('np.', '')
        dic = {
            'arctan':'atan',
            'arctan2':'atan2',
            'arcsin':'asin',
            'arccos':'acos',
            'abs': 'Abs',
            'arcsinh': 'asinh',
            'arccosh': 'acosh',
            'arctanh': 'atanh',
        }
        for key in dic.keys():
            if key in expr:
                expr = expr.replace(key, dic[key])
        return expr

# Extraigo rangos
if __name__ == '__main__':
    # # Prueba regresión lineal
    # # INPUT
    # expr = 'A*cos(f*t) + C'    
    # variables = [
    #     ('f', 100), # Hz
    #     ('A', 2), # Volts
    #     ('t', 1), # s
    #     ('C', .5), # Volts
    #     ]
    # errores = np.array([0.0005, .0001, 0, .0001]).reshape(-1,1) # Notar que al tiempo no le asigne error
    # propaga = Propagacion_errores(formula = expr, variables = variables, errores = errores)
    # valor, error = propaga.fit()
    # print(valor, error)

    remanencia = [0.6617640000000007, 0.648, 0.6463663999999998, 0.644, 0.6446468571428572, 0.6313968000000003, 0.6101840000000002, 0.6201240000000001, 0.5880179999999999, 0.5917687999999999, 0.5733999999999997, 0.5831424000000001, 0.5679799999999995, 0.5564127999999998, 0.5430304000000001, 0.5362119999999997, 0.5105864, 0.517164, 0.4968688, 0.48230079999999953, 0.4683824, 0.46571199999999935, 0.4647747999999996, 0.4464924000000006, 0.42020479999999993, 0.4440120000000002, 0.4033028000000004, 0.4176367999999996, 0.38584560000000045, 0.3707120000000005, 0.38518199999999947, 0.36424320000000016, 0.33886479999999963, 0.3343535999999997, 0.3384912000000001, 0.3169108571428574, 0.2907544000000004, 0.2953548000000006, 0.2602840000000004, 0.2610116, 0.20596000000000017, 0.22832840000000082, 0.21134320000000062, 0.19457520000000011, 0.18572639999999896, 0.18100000000000027, 0.11647039999999967, 0.12683519999999993, 0.11390000000000014, 0.1225975999999997, 0.0967327999999996, 0.09915520000000044, 0.08739759999999902, 0.08858000000000059, 0.06923520000000026, 0.04971599999999991, 0.04452320000000015, 0.04711520000000016, 0.03800000000000004, 0.016878399999999738, 0.006257599999999929, 0.019350400000000274, 0.014008000000000338, 0.011264999999999938, 0.017362799999999897, 0.010928399999999842, 0.011608000000000238, 0.013312000000000067, 0.0080819999999997, 0.013582666666666762, 0.012048333333332904, 0.00451599999999988, 0.008423619047618899, 0.009185523809523857, 0.00942560000000025, 0.00749600000000034, 0.009920400000000086, 0.010071333333333219, 0.0107526666666668, 0.0051593333333333205, 0.009077600000000165, 0.008663111111111118, 0.008011047619047704, 0.011247999999999411, 0.009370000000000028, 0.00923333333333337, 0.010202000000000128, 0.008650399999999874, 0.007635600000000099, 0.008488833333333144, 0.007483333333333452, 0.010556571428571316, 0.008483142857142578, 0.008123809523809191, 0.008267200000000186, 0.008483200000000425, 0.007842400000000128, 0.0086675000000002, 0.004449600000000178, 0.005907000000000015]
    def ajuste(t, t_0, a, g, c):
        return np.piecewise(t, [t < t_0, t >= t_0], [lambda t: a*np.abs(t-t_0)**(g) + c, c])
        # return np.piecewise(t, [t < t_0, t >= t_0], [lambda t: a*(t-t_0)**(g) + c, c])

    # Errores en las esclas de tensión acorde a c/ medicion:
    errores = {'medicion_12_c1':8*.5/256, 'medicion_13_c1':8*.5/256, 'medicion_14_c1':8/256, 'medicion_15_c1':8*2/256, 'medicion_16_c1':8*2/256,'medicion_12_c2':8*.2/256, 'medicion_13_c2':8*.2/256, 'medicion_14_c2':8*.2/256, 'medicion_15_c2':8*.5/256, 'medicion_16_c2':8*.5/256}

    # Estoy haciendo la resta entre dos valores del canal 2, entonces aparece el factor sqrt(2):
    error = np.full(len(remanencia), 2.77680184e-08)

    # Hago el ajuste
    formula = 'np.piecewise(x_, [x_ < a_0, x_ >= a_0], [lambda x_: a_1*np.abs(x_- a_0)**(a_2) + a_3, a_3])'
    formula = formula.replace('x_', str(float(3)))
    variables = [('a_0',2.48909942e+02), ('a_1',6.70276581e-02),('a_2',4.48016420e-01),('a_3',1.42274642e-02)]
    Propagacion_errores(formula = formula, variables = variables, errores = error.reshape(-1,1)).fit()