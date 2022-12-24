import numpy as np
from sympy import symbols,lambdify,diff,sqrt,sin,exp,cos

class Propagacion_errores:

    def __init__(self, variables: list, errores: np.array, formula:str) -> None:
        '''
        INPUT: 
        -data --> dict: es un diccionario con dos keys: 
        
            *'variables'--> list de tuples de tres elementos (es necesario pasar 
            tres elementos aunque no haya errores para dicha variable): (simbolo de la variable, valor).
        
            *'expr'--> str: formula
        -covarianza --> np.array of shape n x n, donde n es la cantidad de variables
        '''
        self.variables = variables

        # Si se pasa un array de errores, armo la matriz de covarianza. En el caso contrario, armo los errores:
        if errores.ndim != 2:
            raise ValueError("Passed array is not the right shape. It should be a (len(variables), 1) shaped array if sigma of variables was provided or a (len(variables), len(variables)) if a covariance matrix is passed.")

        if errores.shape[1] == 1:
            self.covarianza = np.diag(errores.reshape(-1)**2)
            self.errores = errores
        else:
            self.covarianza = errores
            self.errores = np.sqrt(np.diag(errores))
        self.formula = formula
        self.valor = None
        self.error = None
  

    def __str__(self):
        texto  = f'El valor obtenido es: ({self.valor} ± {self.error})'
        print(texto)
        return texto
    
    # @classmethod
    # def without_variables(cls, variables_value, formula, errores):
    #     variables = [(f'a_{i}', variables_value[i]) for i in range(len(variables_value))]
    #     return cls(variables = variables, errores = errores, formula = formula)

    def fit(self):
        '''
        OUTPUT:
        Actualiza los valores de self.valor y self.error.
        '''
        simbolos = [i for i,j in self.variables]
        valores = [j for i,j in self.variables]
        
        # Defino como símbolos las variables de las cuales depende la expresión:
        for sim in simbolos:
            globals()[sim] = symbols(sim, real = True)

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
    variables = [
        ('f', 14.78), 
        ('a',-.052710), 
        ('d',5/1000),
        ('m',88.85/1000), 
        ('l',.36)]
    errores = np.array([0,.00009, 0.05/1000, .01/1000, 1/1000]).reshape(-1,1)
    expr = '((f**2)*4*np.pi**2+a**2)/((((np.pi*(d)**4)/64)/(m/l)*4.934484391**4))'    

    propaga = Propagacion_errores(variables = variables, formula = expr, errores = errores)
    propaga.fit()


