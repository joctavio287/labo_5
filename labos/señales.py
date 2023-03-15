import numpy as np

def fft(tiempo = None, señal = None):
    '''
    Realiza la transformada Fourier de una señal y devuelve los arrays necesarios para grafic
    ar.

    Tener en cuenta que la mayor frecuencia que se puede registrar es la mitad de la frequenc
    ia de sampleo (Teorema de Nyquist). Si, por ejemplo, el sampleo fuese por lo menos dos ve
    ces más lento que la frecuencia de la señal veríamos aliasing. Si tomasemos todo el rango
    espectral, el algoritmo lo completaría espejando el resultado respecto a la mitad de la f
    recuencia de sampleo.

    INPUT:
    tiempo[opt.]-->np.array: eje x de la función que queremos transformar.
    señal-->np.array: eje y de la función que queremos transformar.

    OUTPUT:
    touple de np.arrays(): frecuencias, amplitud espectral.
    '''
    if señal is None or tiempo is None:
        raise ValueError('Se deben proveer la señal y el tiempo en format np.ndarray')
    
    # Frecuencia de sampleo
    tstep = (tiempo.max()-tiempo.min())/len(tiempo)
    fsamp = 1/tstep  
    
    # Realizo la transformada
    señal_fft, N = np.fft.fft(señal), len(señal)
    
    return np.linspace(0, fsamp/2, int(N/2)), 2*np.abs(señal_fft[:N//2])/N
    

    