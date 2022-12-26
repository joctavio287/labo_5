import numpy as np
from sympy import symbols, lambdify, diff

# Sí, una lista enorme de funciones.
from sympy import log, sin, exp, cos, tan, sqrt, pi, atan, asin, acos, Abs, cot, sec, csc, sinc, acot, asec, acsc, atan2, sinh, cosh, tanh, coth, sech, csch, asinh, acosh, atanh, acoth, asech, root

class Propagacion_errores:

    def __init__(self, formula:str, variables: list, errores: np.array) -> None:
        '''
        INPUT: 
        -formula: str: la fórmula de la variable cuyo error se quiere propagar. Esta debe usar exclu
        sivamente la librería de sympy para funciones especiales (sin, cos, etc.). A continuación de
        jo una lista con las funciones incluidas: pi, log, sin, exp, cos, tan, sqrt, atan, asin, aco
        s, Abs, cot, sec, csc, sinc, acot, asec, acsc, atan2, sinh, cosh, tanh, coth, sech, csch, as
        inh, acosh, atanh, acoth, asech, root. De querer usar otras que estén incluidas dentro de la
        librería agregarlas manualmente en el import.
        
        -variables: list: es una lista de tuplas de dos elementos. Cada tupla esta compuesta por dos
        valores. El primer valor es un str cuyo símbolo representa a cada variable (se debe respetar
        las mismas variables usadas en la fórmula).

        -cov_y: np.array: matriz de covarianza de las variables. También es posible pasar un array c
        on las desviaciones estándar de cada dato. Si este es el caso, la matriz de covarianza se co
        nstruira automáticamente.
        '''
        # Variables importantes e hiperparámetros usados dentro de la clase
        self.formula = formula
        self.variables = variables
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
    
    # @classmethod #TODO: implementar distinta forma de llamar
    # def without_variables(cls, variables_value, formula, errores):
    #     variables = [(f'a_{i}', variables_value[i]) for i in range(len(variables_value))]
    #     return cls(variables = variables, errores = errores, formula = formula)

    def fit(self):
        '''
        Calcula el error propagado y el valor de la magnitud de interés.
        
        OUTPUT: tupla con dos valores.
        -self.valor: np.float64: valor de la cantidad pasada en la fórmula.
        
        -self.error: np.float64: error (raíz cuadrada de varianza) propagado de la cantidad pasada e
        n la fórmula.
        '''
        simbolos = [i for i,j in self.variables]
        valores = [j for i,j in self.variables]
        
        # Defino como símbolos las variables de las cuales depende la expresión:
        for sim in simbolos:
            locals()[sim] = symbols(sim, real = True)

        # Defino la expresión simbólica:
        formula = eval(self.formula)
        
        # Calculo la aproximación lineal de la covarianza de el i-ésimo dato con el j-ésimo y sumo:
        derivadas_parciales = [diff(formula, eval(sim)) for sim in simbolos]
        covarianza_resultado = 0
        for i in range(len(simbolos)):
            for j in range(len(simbolos)):
                covarianza_resultado += derivadas_parciales[i]*self.covarianza[i, j]*derivadas_parciales[j]
        error_simbolico = sqrt(covarianza_resultado)

        # Convierto la expresión simbólica en un módulo numérico (numpy) para poder reemplazar:
        lambd_err = lambdify(simbolos, error_simbolico, modules = ['numpy'])
        lambd_val = lambdify(simbolos, formula, modules = ['numpy'])
        self.valor, self.error = lambd_val(*valores), lambd_err(*valores)
        return self.valor, self.error

if __name__ == '__main__':
    # Prueba regresión lineal
    # INPUT
    expr = 'A*cos(f*t) + C'    
    variables = [
        ('f', 100), # Hz
        ('A', 2), # Volts
        ('t', 1), # s
        ('C', .5), # Volts
        ]
    errores = np.array([0.0005, .0001, 0, .0001]).reshape(-1,1) # Notar que al tiempo no le asigne error
    propaga = Propagacion_errores(formula = expr, variables = variables, errores = errores)
    valor, error = propaga.fit()
    print(valor, error)


