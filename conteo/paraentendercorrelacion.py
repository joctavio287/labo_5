import numpy as np
import matplotlib.pyplot as plt
T = 1
paso = T/2500

# Tomo el intervalo [0,T] con pasos T/2500: 25001 pasos
largo = 10*T
t = np.arange(0,largo + paso, paso)
signal = np.sin(t*2*np.pi/(T))
# signal_d = np.sin((t)**2*2*np.pi/(T) )
signal_d = np.sin(t*100*2*np.pi/(T) )*np.sin(t*2*np.pi/(T) )


correla = np.correlate(signal, signal, mode = 'same')
correla_d = np.correlate(signal_d, signal_d, mode = 'same')

desfasaje = np.arange(-np.pi, np.pi + paso, paso)
plt.figure()
plt.plot(t-largo/2, correla/np.max(correla), label = 'coherente')
plt.plot(t-largo/2, correla_d/np.max(correla_d), label = 'incoherente')
# plt.plot(t,signal)
# plt.plot(t, signal_d)
# plt.plot(t+T/2,signal)
plt.legend()
plt.grid(visible = True)
plt.show(block = False)

plt.figure()

# plt.plot(t,signal)
plt.plot(t, signal_d)
# plt.plot(t+T/2,signal)
plt.legend()
plt.grid(visible = True)
plt.show(block = False)