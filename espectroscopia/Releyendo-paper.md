# Informe III: conteo de fotones

## Teoría
### Contexto
En un experimento usual de conteo de fotones, un TFM (tubo fotomultiplicador) se utiliza para convertir luz en electrones, que son amplificados en pulsos eléctricos que son enviados a circuitos contadores. En experimentos de baja iluminación cada pulso se asigna a un solo fotón que llega al fotocátodo y se registra como una cuenta. Obteniendo un número grande de cuentas, la estadística del número de fotones por unidad de tiempo se puede medir, dando información valiosa sobre la naturaleza de la fuente de luz. Por ejemplo, una fuente clásica monocromática con un fotodetector perfecto daría la misma cuenta sobre intervalos de tiempo iguales (en un enfoque semiclásico las cuentas surgen de la naturaleza cuántica de los electrones). Por otro lado, un campo fluctuante como aquel generado por una fuente térmica, daría diferentes cuentas cada vez que se repita el experimento, evidenciando características estadísticas distintivas que evidencian el comportamiento aleatorio de las fuentes térmicas.

### Tiempos de muestreo menores al tiempo de coherencia
El TFM se usa para reunir información sobre el número de fotones que llegan en un intervalo arbitrario T. Para una fuente de luz en equilibrio térmico, la emisión no es constante en el tiempo, sino que se caracteriza por tener fluctuaciones. Un experimento repetido dentro del mismo intervalo temporal no va a dar siempre los mismos resultados, lo cual muestra la naturaleza estadística de la fuente de luz. El campo electromagnético oscilante en una frecuencia $\omega$ tiene energías 
$$E_{n} = (n + \frac{1}{2})\, \hbar\, \omega,$$
con n el número de fotones o cuantos de radiación electromagnética. A temperatura $\theta$ la probabilidad de encontrar el estado excitado con energía $E_n$ viene dada por la distribución de Boltzmann 
$$P(n) = \frac{e^{\frac{-E_n}{k_B\, \theta}}}{\sum_{n} e^{\frac{-E_n}{k_B\, \theta}}}.$$
Por otro lado, el valor medio de fotones emitidos puede ser calculado explicitamente (para eso se usa que la función de partición es una serie geométrica)
$$  \langle n\rangle = \frac{1}{e^{\frac{-\hbar\ \omega }{k_B\, \theta}}-1}.$$
De esta forma se puede expresar la probabilidad de encontrar el estado excitado con energía $E_n$ en función del número medio de fotones:
$$P(n) = \frac{\langle n \rangle ^{n}}{(1 + \langle n \rangle)^{1+n}}.$$
En la práctica se mide el número de fotones en un intervalo temporal $T$, es decir, aquellos fotones que están en el volumen de longitud $L = c\,T$, ya que aquellos que se encuentren más lejos no lograran llegar al fotodetector a tiempo.
Repetir el experimento significa tomar el mismo intervalo temporal ($T$), pero comenzando en diferentes instántes t (asumiendo siempre que la fuente de luz es estacionaria). Las fluctuaciones en la medición apareceran como fluctuaciones en la potencia de la luz de la fuente.
En los resultados previos se asumió que sólo se mide un modo del campo electromagnético. Es decir, que la fuente de luz se mantiene monocromática durante el intervalo de medición. Esto es
$$
\begin{equation}
T \ll T_C
\end{equation}
$$
con  $T_C$ el tiempo de coherencia.
El tiempo de coherencia es la inversa del ancho de banda y puede ser determinado observando el tiempo característico de fluctuación de la señal, ya sea directamente (si el detector es lo suficientemente rápido) u observando el contraste de un patrón de interferencia como función del tiempo. Dado que el aparato experimental requiere intervalos temporales de varios microsegundos, se necesitan fuentes de luz extremadamante coherentes para satisfacer ($1$). Por lo tanto, se utiliza un láser monocromático de He-Ne y el carácter térmico de la fuente se impone utilizando fluctuaciones aleatorias artificiales mediante el uso de un vidrio esmerilado que gira y genera decoherencia espacial. 
En este caso, la probabilidad de medir n fotones en un interavalo T queda
$$P(n) = \frac{\langle n \rangle ^{n}}{(1 + \langle n \rangle)^{1+n}}, \quad (T\ll T_c)
$$
con $\langle n \rangle$ una función del tiempo.
Desde el punto de vista semiclásico, la intensidad de la luz fluctua aleatoriamente con una duración típica del orden de $T_c$. Y el campo es medido en intervalos temporales T mucho menores que las duraciones de dichas fluctuaciones. Durante el tiempo T, la potencia puede considerarse constante, pero entre una medición y otra ocurren fluctuaciones aleatorias. Fotones detectados dentro del intervalo T están correlacionados, lo cual significa que no son estadísticamente independientes. Es decir, si un fotón se detecta en un intervalo $\Delta t$ dentro de T, es más probable que se detecte otro fotón en otros intervalos $\Delta t$ dentro de T.

### Tiempos de muestreo mayores al tiempo de coherencia
En la ausencia de fluctuaciones entre mediciones (entre intervalos T) la detección de fotones no debería estar correlacionadas. Es decir que haber detectado un fotón en un intervalo $\Delta t$ no debería influenciar la probabilidad de detectar otro fotón cerca del intervalo $\Delta t$. En este límite, la probabilidad de detectar un fotón en un intervalo $\Delta t$ debería ser proporcional al intervalo y, a su vez, debería ser independiente de cualquier otro evento que ocurra en otro intervalo. En dicho caso, la probabilidad de detectar n fotones en un intervalo T debería seguir la distribución de Poisson
$$
P_{n}(T) = \frac{\langle n\rangle^{n}\, e^{-\langle n \rangle}}{n!}. \quad (T\ll T_c)
$$
Fotones no correlacionados tienen una distribución de Poisson, pero también tienen otras características estadísticas distintivas.

### Probabilidad de tiempo de llegada
Una de estas características puede verse en la probabilidad del tiempo de llegada $P_{at}(T)$, que es la probabilidad de medir dos fotones consecutivos en un intervalo de tiempo T. Esta probabilidad se puede expresar como el producto de dos probabilidades, la de contar 0 fotones en un intervalo [t, t+T] y la de contar 1 fotón en el tiempo t+T. Luego de promediar estadísticamente sobre las fluctuaciones de intensidad, $P_{at}(T)$ puede ser expresada como
$$
P_{at}(T) =\langle P_0(t,T)\, p(t+T)\rangle.
$$
Para eventos no correlacionados, $p(t)$ toma un valor constante y esta ecuación toma la forma 
$$
P_{at}(T) =\zeta \,P_0(t,T),
$$
con $\zeta$ una constante. Sino, $P_{at}(T)$ no es proporcional a $P_{0}(T)$.

Cuando se utiliza luz térmica, la correlación de los fotones se pierde para tiempos mucho mayores a $T_c$. Este fenómeno puede ser entendido notando que si el intervalo temporal T es muy largo suceden muchas fluctuaciones y, entonces, estamos midiendo un promedio de las fluctuaciones y no la fluctuación en sí. Cuanto más largo es el intervalo, el valor medido se acercará más al valor medio.
Como consecuencia, las estadísticas medidas se acercan a la distribución de Poisson no correlacionada. Otra forma de ver este problema es notar que si $ T\gg T_c$ la luz no puede ser percibida como monocromática y, por lo tanto, muchos modos son necesarios para describir la luz. Los fotones de un mismo modo están correlacionados por la distribución de Bose-Einstein, pero fotones de diferentes modos están no-correlacionados. En consecuencia, cuando se cuentan fotones dentro de un intervalo temporal largo, aquellos que pertenecen a diferentes modos son detectados y la correlación se pierde.
Para una descripción más formal, se puede considerar lo siguiente. En la teoría semiclásica de detección óptica, el campo electromagnético es tratado clásicamente y el TFM (tubo fotomultiplicador) convierte intensidad clásica y continua ($I$) en una sucesión discreta de cuentas.
Teniendo en cuenta la hipótesis de que la probabilidad $p(t)$ por unidad de tiempo de tener una sola cuenta a tiempo t es proporcional a la intensidad $\bar{I}(t)$, se puede obtener la fórmula de Mandel
$$
P_{n}(T) = \langle \frac{[\xi\, \hat{I}(t,T)\, T]^{n}}{n!}\, exp[-\xi\, \hat{I}(t,T)\, T]\rangle,
$$
donde $\xi$ es la eficiencia (numero de detecciones/numero de emisiones) del detector. La distribución se obtiene como un promedio estadístico sobre las fluctuaciones de intensidad $\bar{I}(t,T)$. Es difícil de encontrar una expresión general para el promedio estadístico de la función dependiente del tiempo, pero en los dos casos discutidos las expresiones explicitadas se pueden derivar exactamente.
En fin, hemos obtenido el comportamiento de la fuente térmica en dos casos extremos. Sin embargo, es casi imposible trabajar con fuentes térmicas reales bajo la condición $T\ll T_c$ debido a que el tiempo de coherencia de las fuentes térmicas es del orden de $10^{-8}s$. Por esta razón, construimos una fuente pseudo-térmica cuyo tiempo de coherencia $T_c$ se puede elegir para satisfacer las condiciones requeridas.