# Final Matematica IV

## Teoremas

---

- Condiciones de Cauchy-Riemann: demostración, holomorfía, etc.

- Series de potencias (funciones analíticas), región de validez (radio de convergencia), derivadas definidas.

- Fórmula de Cauchy: relación de $f(z_0)$ con integral sobre curvas, condiciones y demostración, fórmula generalizada (Teorema de Taylor).

- Desarrollo de Laurent: demostración, holomorfía en una corona, regiones de validez, coeficientes.

- Clasificación de singularidades: aisladas y no aisladas, desarrollo de Laurent (evitable/polo/esencial), propiedades, singularidades en el $\infty$.

- Teorema de Residuos: definición, teorema (de la suma), residuo en el $\infty$.

- Series de Fourier: sistemas ortonormales (ortonormalidad, p.i, e.v),  definición, convergencia en media cuadrática, formas trigonométricas y exponenciales compleja, convergencia puntual y uniforme.

- Transformada de Fourier: definición, propiedades (en particular, convolución), lema de Plancherel-Parseval ($L^2$), teoremas de inversión.
 
- Transformada de Laplace: definición, orden exponencial, propiedades, forma compleja, inversión


## Respuestas

---

### **Cauchy-Riemann**

 **Derivada de una función compleja**
 
 Sea  $A \subset \mathbb{C}$ un conjunto abierto y $f: A \rightarrow \mathbb{C}$, se dice que $f$ es derivable en un punto $a\in A$ si existe el límite $$\lim_{z\to a} \frac{f(z)-f(a)}{z-a} = \lim_{h\to 0}\frac{f(a+h)-f(a)}{h}\in \mathbb{C}.$$
En ese caso, el valor del límite se representa por $f'(a)$ y se llama la derivada de $f$ en $a$.

**Observación**: tener en cuenta que el límite puede ser acercandose por cualquier curva del plano complejo. Lo cual hace de esta derivada, una definición mucho más fuerte que aquella definida para funciones reales.

Vale que la suma, resta, composición y división de funciones derivables, es derivable si se tienen en cuenta dominios apropiados (es decir, que en la división de funciones no se anule el denominador, que el dominio de la función que se compone sea el codoninio de la función con la cual componemos, etc).

**Condición de Cauchy-Riemann como consecuencia de la definición de derivada**

Sean $f = u + i\,v$ y $z_0 = x_0 + i\, y_0$, con $u$ y $v$ y $x_0$ e $y_0$ las partes reales e imaginarias de $f$ y $z_0$ respectivamente. Entonces, si existe $f'(z_0)$ los siguientes límites deben coincidir,

$$
f'(z_0) = \lim_{(x,y_0)\to(x_0,y_0)} \frac{f(z)-f(z_0)}{z-z_0} =
\lim_{(x_0,y)\to(x_0,y_0)} \frac{f(z)-f(z_0)}{z-z_0}.
$$

Desarrollando el término dentro del primer límite se tiene que

$$
\frac{f(z)-f(z_0)}{z-z_0} = \frac{u(x,y_0)+i\, v(x,y_0)-u(x_0,y_0)-i\, v(x_0,y_0)}{x-x_0} = 
$$

$$
 = \frac{u(x,y_0)-u(x_0,y_0)}{x-x_0} + i\, \frac{v(x,y_0)-v(x_0,y_0)}{x-x_0}.
$$

Por lo que al tomar el límite $x\to x_0$ se tiene que $f'(z_0) = \frac{\partial u}{\partial x}(x_0,y_0)+ i\, \frac{\partial v}{\partial x}(x_0,y_0)$. Por otro lado, al desarrollar el segundo térimno se tiene

$$
\frac{f(z)-f(z_0)}{z-z_0} = \frac{u(x_0,y)+i\, v(x_0,y)-u(x_0,y_0)-i\, v(x_0,y_0)}{i\, (y-y_0)} = 
$$

$$
 = \frac{u(x_0,y)-u(x_0,y_0)}{i\, (y-y_0)} + \frac{v(x_0,y)-v(x_0,y_0)}{ (y-y_0)}.
$$

Y tomando el límite $y\to y_0$ se tiene que $f'(z_0) = -i\, \frac{\partial u}{\partial y}(x_0,y_0)+\frac{\partial v}{\partial y}(x_0,y_0)$. De aquí, igualando parte real e imaginaria de $f'(z_0)$ se deduce que

$$
\frac{\partial u}{\partial x}(x_0,y_0) = \frac{\partial v}{\partial y}(x_0,y_0)$$
y
$$
\frac{\partial u}{\partial y}(z_0) = -\frac{\partial v}{\partial x}(x_0,y_0)
$$
o, equivalentmente,
$$
f'(z_0) = \frac{\partial f}{\partial x}(z_0) = -i\, \frac{\partial f}{\partial y}(z_0)
$$

**Teorema de Cauchy Riemann**

Una función $f:U\subseteq\mathbb{C}\rightarrow\mathbb{C}$ es derivable en $z_0 = x_0 + i\, y_0\in U$ con $U$ abierto $\Leftrightarrow$ $f$ es diferenciable en $(x_0,y_0)$ y sus derivadas parciales satisfacen la condición de C-R.

**Demostración**

$\Rightarrow$

Defino $\Delta f = f(x_0 + h_x, y_0 +h_y)-f(x_0,y_0)$. Luego, como $f$ es derivable en $z_0$, es conveniente definir 
$$
\alpha_0(h_x,h_y) = \frac{\Delta f(h_x,h_y)-f'(z_0)\, (h_x + i\,h_y)}{h_x+i\, h_y},
$$
que tiende a 0 cuando $(h_x,h_y)\rightarrow(0,0)$. De esta forma, $\Delta f = \,\alpha_0\, (h_x +i\, h_y) +f'(z_0)\, (h_x + i\, h_y)$ que permite definir $\alpha(h_x, h_y) := \alpha_0\, (h_x +i\, h_y)$ que satisface

$$
\lim_{(h_x,h_y)\to(0,0)}\frac{\alpha(h_x, h_y)}{\|(h_x,h_y)\|}=0,
$$
por lo tanto, f es diferenciable y $f'(z_0)=\frac{\partial f}{\partial x}(z_0)= \frac{1}{i}\, \frac{\partial f}{\partial y}(z_0)$.

$\Leftarrow$

Sea $\Delta f = \frac{\partial f}{\partial x}(z_0)\, h_x + \frac{\partial f}{\partial y}(z_0)\, h_y + \alpha(h_x, h_y)$. Usando Cauchy-Riemann ($\frac{\partial f}{\partial y}(z_0) = i\, \frac{\partial f}{\partial x}(z_0)$) tenemos que

$$
\Delta f = \frac{\partial f}{\partial x}(z_0)\, (h_x + i\,h_y) + \alpha(h_x, h_y)
$$
con

$$
\lim_{(h_x,h_y)\to(0,0)}\frac{\alpha(h_x, h_y)}{\|(h_x,h_y)\|}=0
$$
por hipótesis.
$$ \blacksquare$$

**Funciones holomórfas**

Sea $f:A\subseteq\mathbb{C}\rightarrow\mathbb{C}$ con $A$ abierto. Se dice que $f$ es holomorfa en $z_0\in A$ si $\exists \, r>0:f$  es derivable en el disco $D(z_0,r)$. Si $f$ es holomorfa en cada $z\in A$ se dice que $f$ es holomorfa en $A$.

**Ejemplos**

- Si $f(z) = z^2$ entonces $u(x,y) =x^2-y^2$ y $v(x,y)= 2\,x\, y$. Entonces, es inmediato ver que se satisaface C-R para todo $(x,y)\in\mathbb{R}^2$. Además, al ser $f(x,y)$ suma y multiplicación de funciones diferenciables, es diferenciable para todo $(x,y)\in\mathbb{R}^2$. Entonces es derivable y holomorfa en todo $\mathbb{C}$.

- Si  $f(x,y) = x^2 + y^2 - 2\, x\, y$, C-R se satisface sólo para $A = \{z\in\mathbb{C}:Re(z) = 0\}$, a pesar de ser diferenciable en todo $\mathbb{R}^{2}$. Por lo tanto, sólo es derivable en $A$ y no es holomorfa para ningún punto ya que no A no tiene puntos interiores.

**Funciones armónicas**

Una función real  $f:A\subseteq\mathbb{R}^2\rightarrow\mathbb{R}$ se dice armónica en $U\subset\mathbb{R}^2$ abierto si sus derivadas parciales de primer y segundo orden son continuas y satisfacen la ecuación

$$
\Delta f := f_{xx} + f_{yy} = 0
$$

que se conoce como ecuación de Laplace.

**Teorema para funciones holomorfas**

Si una función a valores complejos es holomorfa y su parte real e imaginaria es $C^{2}(D\subseteq\mathbb{R}^2)$ abierto, dichas componentes son armónicas en D.

**Demostración**

Se toman las ecuaciones de C-R y se derivan respecto a x por un lado, y respecto a y, por el otro . Luego, usando Clairout-Schwarz (derivadas cruzadas coinciden) se suman ambos  pares de ecuaciones y se obtiene lo desesado.
$$ \blacksquare$$

**Funciones armónicas conjugadas**

Si dos funciones dadas $u$ y $v$ son armónicas en un abierto $D$ y sus derivadas parciales de primer orden satisfacen C-R en $D$, se dice que $v$ es armónica conjugada de $u$.

**Observación**: no vale el recíproco. Además si $v$ es armónica conjugada de $u$, entonces $f = u + i\,v$ es holomorfa. Por último, si $v$ es a.c de $u$, entonces $v+c$ también lo es $\forall c\in \mathbb{R}$.

**Ejemplo**

- Ya que $f(z) = z^{2} = x^{2} + y^{2} + i\, 2\ x\,$ y $g(z) = e^{x}\, [cos(y) + i\, sen(y))]$ son holomorfas en todo el plano, también lo es su producto. Entonces por el primer teorema,
$
Re[f(z)\, g(z)] = e^{x}[(x^{2}-y^{2})\,cos(y)-2\,x\,y\,sen(y)]
$
es armónica en todo el plano.

### **Series de potencia**

**Convergencia puntual de una sucesión de funciones**

Sea $f_{n}:A\rightarrow\mathbb{C}$ una sucesión de funciones, todas ellas definidas en el conjunto $A$. Se dice que la sucesión converge puntualmente si, para cada $z\in A$, la sucesión $f_{n}(z)$ converge. El límite define una nueva función en $A$. Es decir, si para cada $\epsilon >0, \exists\,  n_0(z): \forall\, n\geq n_0(z)$, $|f_{n}(z)-f(z)|<\epsilon$. 

**Convergencia uniforme de una sucesión de funciones**

Sea $f_{n}:A\rightarrow\mathbb{C}$ una sucesión de funciones, todas ellas definidas en el conjunto $A$. Se dice que la sucesión converge uniformemente en $A$ si para cada $\epsilon >0, \exists\,  n_0: \forall\, n\geq n_0$, $|f_{n}(z)-f(z)|<\epsilon$. 

**Observación:** la diferencia entre este dos tipos de convergencia radica en que la puntual, el $n_0$ depende del $z$ que analicemos. Mientras que en la uniforme tomado $n_0$ la cota debe valer para todo $z$.

**Ejemplos**

- $f_n(z) = z^{n}$ en $|z|<1$. 

Como $z^{n}\to 0$ para cada $z\in\mathbb{C}$ tal que $|z|<1$. La convergencia no es uniforme pues dado $\epsilon>0$, 
$$
|z^{n}-0|<\epsilon \Rightarrow \quad
\begin{aligned}
& n\, log|z|<log|\epsilon|\quad  \text{si}\quad \epsilon>1\\
& n\, log|z|>log\frac{1}{|\epsilon|}\quad  \text{si}\quad \epsilon>1
\end{aligned} 
$$
$$
\Rightarrow 
\begin{aligned}
& n_0 = 1 \quad  \text{si}\quad \epsilon>1\\
& n_0 = \frac{log\frac{1}{|\epsilon|}}{log|z|}\quad  \text{si}\quad 0<\epsilon\leq1.
\end{aligned} 
$$
Es decir, $n_0 = n_0(\epsilon, z)$.

- $f_n(x) = x^{n}$ en $[0,1)$ sucede lo mismo. 

Sin embargo, si consideramos cualquier subintervalo acotado $B = [0,r]$ con $r<1$ hay convergencia uniforme. Basta con tomar, $\hat{n}_0 = n_0(\epsilon, r)$, con $n_0$ la función definida arriba.

**Observación:** para analizar la convergencia de series la idea es la misma, pero se considera la sucesión de sumas parciales $S_{k}(Z) = \sum_{n=1}^{h} f_{n}(z)$.

**Criterio de Weierstrass**

Sean $f_n:A\subset \mathbb{C}\rightarrow\mathbb{C}$. Supongamos que existen $M_n\geq0$ tales que
$$
\begin{aligned}
&i)\quad |f_{n}(z)|\leq M_n\, \forall z\in A,\quad \forall n\in N\\
&ii)\quad \sum_{n = 1}^{\infty} M_n\quad \text{converge}.
\end{aligned}
$$

Entonces $\sum^{\infty}_{n = 1} f_n$ converge absoluta y uniformemente en A.

**Demostración**

Dado $z\in A$, como 
$$
f_{n}(z)|\leq M_n\, \forall z\in A,\quad \forall n\in N
$$
y
$$
\sum_{n = 1}^{\infty} M_n\quad \text{converge},
$$

$$
\Rightarrow \sum^{\infty}_{n=1}|f_n(z)|\quad \text{converge por el criterio de comparación.}
$$

Es decir, que la serie $\sum^{\infty}_{n=1}f_n(z)$ converge absolutamente en A. Esto muestra que la sucesión de sumas parciales 
$$
S_{k}(z) = \sum_{n=1}^{k} f_{n}(z)
$$ 
converge puntualmente a una función $S(z)$.

Para mostrar que la serie converge uniformemente, notemos que si $z\in A$ y $k<m$,

$$
|S_m(z)-S_k(z)| = |\sum_{n=k+1}^{m} f_{n}(z)|\leq\sum_{n=k+1}^{m} |f_{n}(z)|\leq \sum_{n =k+1}^{m} M_n
$$

entonces si  $m\to \infty$

$$
|S(z)-S_k(z)|\leq M-M_k<\epsilon
$$
para $k>k_0$ con $k_0$ independiente de z.
$$ \blacksquare$$

**Serie de potencias**

Una serie de potencias es una serie de la forma 
$$
\sum^{\infty}_{n=0} a_n\, (z-z_0)^{n}
$$
con $a_n$ y $z_0\in\mathbb{C}$ fijos.

**Lema de Abel-Weierstrass**

Suponiendo que $|a_n|\, r^{n}_0\leq M \, \forall n$ con $M\in \mathbb{R}$ y $r_0>0$. Entonces para $r<r_0$
$$
\sum^{\infty}_{n = 0}a_n\, (z-z_0)^{n}
$$ 
converge uniforme y absolutamente en el disco cerrado $A_r = \{z: |z-z_0|\leq r\}$.

**Demostración**

Para $z\in A_r$ se tiene que 
$$
|a_n|\leq \frac{M}{r^{n}_0}\implies|a_n\, (z-z_0)^n|\leq |a_n|\, r^{n}\leq M\, (\frac{r}{r_0})^{n}.
$$
Entonces, sea $M_n = M\, (\frac{r}{r_0})^n$, como $\frac{r}{r_0}<1$, la serie 
$$
\sum^{\infty}_{n=0}M_n
$$
converge. Así por el criterio de Weierstrass la serie converge uniforme y absolutamente en $A_r = \{z: |z-z_0|\leq r\}$.
$$ \blacksquare$$
**Teorema de convergencia para series de potencias**

Sea una serie de potencias
$$
\sum^{\infty}_{n = 0}a_n\, (z-z_0)^{n}, \quad \exists!\, R\geq0\in \mathbb{R}
$$ 
 tal que si $|z-z_0|<R$, la serie converge y si $|z-z_0|>R$, diverge. Este número se llama *radio de convergencia*. La convergencia es uniforme y absoluta en cualquier disco cerrado contenido en $A = \{z\in \mathbb{C}: |z-z_0|<R\}$. No se puede hacer un enunciado general sobre el radio de convergencia.

**Demostración**

Sea 
$$
R = sup\, \{r\geq0:  \sum^{\infty}_{n = 0} |a_n|\, r^n\quad \text{converge}\},
$$
veamos que $R$ tiene las propiedades deseadas. Notemos $A_r$ al disco cerrado de radio $r$ y centro $z_0$. Sea $r_0<R$, por la definición de $R$, existe un $ r_1\in (r_0,R]$ tal que la serie 
$$
\sum^{\infty}_{n=0} |a_n|\,r_1^{n}
$$
converge. Por lo tanto, por el criterio de comparación, la serie
$$
\sum^{\infty}_{n=0} |a_n|\,r_0^{n}
$$
también lo hace y los términos de su serie están acotados y tienden a 0. Por lo tanto, por el lema de Abel-Weierstrass, la serie converge uniforme y absolutamente en $A_r$. Y puesto que siempre se puede elegir $r_0\in (r,R)$, tenemos la convergencia absoluta en $|z-z_0|<R$.

Si ahora suponemos que $|z_1-z_0|>R$ y que 
$$
\sum^{\infty}_{n=0} a_n\,(z_1-z_0)^{n}
$$
converge, llegaremos a una contradicción.

Bajo la hipótesis de que la serie convege, sus términos están acotados y tienden a 0. Así, por el lema de Abel-Weierstrass, si $r\in (R, |z_1-z_0|)$ entonces 
$$
\sum^{\infty}_{n=0} a_n\,(\tilde{z}_1-z_0)^{n}
$$
converge absolutamente en $A_r$. Por lo tanto, 
$$
\sum^{\infty}_{n=0} a_n\, r^{n}
$$
converge. Pero esto significaría, por la definición de $R$, que $R<R$, lo cual es absurdo.
$$ \blacksquare$$

**Criterios**

Consideremos la serie 
$$
\sum^{\infty}_{n=0} a_n\,(z_1-z_0)^{n}
$$

1. Si existe
$$\rho = \lim_{n\to\infty}|\frac{a_{n+1}}{a_n}|,$$
entonces R = $\frac{1}{\rho}$ es el radio de convergencia. $R = 0$ si $\rho = \infty$ y $R=\infty$ si $\rho =0$.

**Demostración:**

Aplicamos el criterio de D'Alambert a la serie
$$
\lim_{n\to\infty}|\frac{a_{n+1}\,(z-z_0)^{n+1}}{a_n\,(z-z_0)^n}|=\lim_{n\to\infty}|\frac{a_{n+1}\,(z-z_0)}{a_n}|=\rho\, |z-z_0|.
$$
Luego, para que la serie converga $\rho\,|z-z_0|$ debe ser menor que uno. En el caso que sea mayor que uno la serie diverge. Por lo tanto, $R$ es el radio de convergencia.
$$\blacksquare$$

2. Si existe
$$\rho = \lim_{n\to\infty}\sqrt[n]{a_n},$$
entonces R = $\frac{1}{\rho}$ es el radio de convergencia. 

**Demostración**

Análoga, pero utilizando el criterio de la raíz de Cauchy.

3. Fórmula de Hadamar

Sea 
$$
\rho = \lim_{n\to\infty}sup\sqrt[n]{a_n}
$$
que siempre existe. Entonces $R = \frac{1}{\rho}$. El tema es que calcular ese límite no es trivial me imagino.

**Propiedades de las series de potencia**

Consideremos la serie de potencias
$$
\sum_{n=0}^{\infty}a_n\, (z-z_0)^n
$$
con radio de convergencia $R>0$. 

1. Sea 
$$
f(z) = \sum_{n=0}^{\infty}a_n\, (z-z_0)^n
$$
es una función continua en $D(z_0, R)$ por ser límite uniforme de una sucesión de funciones continuas en un disco cerrado $\bar{D}(z_0,R)$ con $r<R$. En particular, es holomorfa en el mismo dominio.

2. Se tiene que 

$$
f'(z) = \sum_{n=1}^{\infty}n\,a_n\, (z-z_0)^{n-1}
$$
en $D(z_0, R)$ y además el radio de convergencia es el mismo.

**Demostración**
 
Usamos la fórmula de Hadamard para $f(z)$,
$$
R = \frac{1}{\lim_{n\to\infty}sup\sqrt[n]{a_n}}.
$$

Y, por otro lado, como 

$$
\lim_{n\to\infty}sup\sqrt[n]{n\, a_n}= \lim_{n\to\infty}sup\sqrt[n]{a_n}
$$
el radio de convergencia es el mismo. Esto pasa porque $\sqrt[n]{n}\to 1$

**Observación:**
aplicando el método anterior iterativamente notamos que 
$$
f^{(k)}(z)=\sum_{n=k}^\infty n\, \dots (n-k+1)\, a_n\,(z-z_0)^{n-k}\quad\forall k\in \mathbb{N}.
$$
Si evaluamos en $z_0$, se tiene que 
$$
f^{(k)}(z_0) =a_k\, k!\implies a_k = \frac{f^{(k)}(z_0)}{k!}.
$$

Por lo tanto,  
$$
f(z) = \sum_{n=0}^{\infty}\frac{f^{(n)}(z_0)}{n!}\,(z-z_0)^{n}
$$
en $D(z_0,R)$.

**Funciones enteras**

Decimos que una función es entera si su serie de potencias tiene radio infinito.

**Funciones analíticas**

Dada $f:A\subset\mathbb{C}\to\mathbb{C}$ con $A$ abierto. Decimos que $f$ es analítica en $z_0\in A$ si existe $\delta>0$ tal que
$$
f(z) = \sum_{n=0}^{\infty}a_n\, (z-z_0)^n
$$
en $D(z_0,\delta)\subset A$. Y se llama analítica en $A$ si lo es para todo sus puntos.

**FALTAN EJEMPLOS**

### **Integrales complejas: fórmula de Cauchy**

**Ejemplo fundamental**

Calculemos 
$$
\int_{C_r^+(z_0)}(z-z_0)^n\, dz,\quad n\in\mathbb{Z}.
$$
Podemos parametrizar la curva $C_r^+(z_0)$ como $z(t) = z_0 + r\, e^{it}$ con $t\in(-\pi, \pi]$. De esta forma se tiene

$$
\int_{-\pi}^{\pi}r^{n+1}\, e^{i\, (n+1)\, t}\,i\, dt = r^{n+1}\, i\, \int_{-\pi}^{\pi} e^{i\, (n+1)\, t}\, dt = 
$$

$$
r^{n+1}\, i\,[\frac{e^{i\, (n+1)\, t}}{i\, (n+1)\,t}]^{\pi}_{-\pi} = r^{n+1}\, 2\, \frac{sen[(n+1)\, \pi]}{\pi\, (n+1)} = 0\quad \forall n\in\mathbb{Z}\setminus\{-1\}
$$
En el caso de que $n = -1$, la integral da $2\, \pi\, i$.

**Teorema de independencia del camino**

Sea $f:D\subset\mathbb{C}\to\mathbb{C}$ continua en $D$ abierto y conexo entonces son equivalentes las siguiente afirmaciones.

1. Para toda curva cerrada $\Gamma\in D$ $$\int_{\Gamma}f(z)\,dz = 0.$$
2. Fijados $z_0,z_1\in D$, si $\Gamma\in D$ es una curva que va de $z_0$ a $z_1$ $$\implies  \int_{\Gamma}f(z)\,dz $$ depende únicamente de los puntos final e inicial, pero no de la curva.
3. La función $f$ tiene una primitiva holomorfa en $D$, es decir, existe $F$ holomorfa tal que $F'(z) = f(z)$ para todo $z\in D$.

**Demostración**

$1\implies2)$ Sean $\Gamma_1$ y $\Gamma_2$ dos curvas cuales quieras que están dentro de $D$ y unen los puntos $z_0$ y $z_1$. Entonces consideremos la curva cerrada $\Gamma = \Gamma_1-\Gamma_2$. Luego por $1$ se tiene que 
$$
0 =\int_\Gamma f(z)\, dz = \int_{\Gamma_1^{+}} f(z)\, dz +\int_{\Gamma_2^-} f(z)\, dz = 
$$

$$
\int_{\Gamma_1} f(z)\, dz  -\int_{\Gamma_2} f(z)\,dz\implies \int_{\Gamma_1} f(z)\, dz = \int_{\Gamma_2} f(z)\,dz
$$
$2\implies3)$
Si fijamos $z_0\in D$ y tomamos $\Gamma_{z_0\to z_1}$ entonces definimos
$$
F(z) = \int_{\Gamma_{z_0 \to z_1}}f(t)\, dt 
$$
esta bien definida ya que el valor de la integral no depende de la curva. Ahora podemos considerar un segmento $\sigma$ cuya longitud sea $|h|$ con $h\in\mathbb{C}$ que una $z$ con $z+h$. Luego $\Gamma_{z_0\to z_1}+\sigma$ une $z_0$ con $z +h.$ De esta forma,
$$
F(z+h) = \int_{\Gamma_{z_0 \to z_1+\sigma}}f(t)\, dt =\int_{\Gamma_{z_0 \to z_1}}f(t)\, dt + \int_{\sigma}f(t)\, dt 
$$
$$
F(z+h) - F(z)=  \int_{\sigma}f(t)\, dt 
$$
parametrizando $\sigma$ con $w:[0,1]\to\mathbb{C}: w(t) = z + t\, h$ tenemos
$$
\int_{\sigma}dw = \int^{1}_0h\, dt = h\implies \frac{1}{h}\, \int_{\sigma} dw = 1.
$$
Volviendo a la expresión de arriba tenemos

$$
\frac{F(z+h)-F(z)}{h} - f(z) = \frac{1}{h}\,\int_{\sigma}f(t)\, dt-f(z)\,  \frac{1}{h}\, \int_{\sigma} dt=
$$

$$
\frac{1}{h}\,\int_{\sigma}f(t)-f(z)\, dt.
$$

Por lo tanto, 
$$
|\frac{F(z+h)-F(z)}{h} - f(z)| = \frac{1}{|h|}\, |\int_{\sigma}f(t)-f(z)\, dt|\leq
$$
$$
\frac{1}{|h|}\, sup|f(t)-f(z)|\, Long(\sigma) = sup|f(t)-f(z)|\to 0
$$
cuando $h\to 0$. Esto prueba que $F'(z) = f(z)$.

$3\implies1)$

Sea $\Gamma$ una curva cerrada y sea $z:[a,b]\to\mathbb{C}$ una parametrización de $\Gamma$. Como se cumple 3), entonces existe $F(z)$ holomorfa tal que $f(z) = F'(z)$. Luego,
$$
\int_\Gamma f\, dz = \int^b_af(z(t))\,  z'(t)\, dt = \int^b_a \frac{d\, F(z(t))}{d\, t}\, dt$$
$$\implies F(z(b))-F(z(a))=0$$
porque $\Gamma$ es cerrada.
$$\blacksquare$$

**Regla de Barrow**

Sea $f:D\subset\mathbb{C}\to\mathbb{C}$ continua en $D$ conexo y abierto. Si $f$ admite una primitiva $F$ en $D$ entonces dados $z_0$ y $z_1$ en $D$ se tiene 
$$
\int_{z_0}^{z_1}f(z)\, dz = F(z_1)-F(z_0).
$$

**Demostración**

Sea $\Gamma$ una curva en $D$ que conecta $z_0$ y $z_1$ parametrizada por $z:[a, b]\to\mathbb{C}$. Luego, como existe $F$ holomorfa tal que $F'(z)= f(z)$, se tiene que 
$$
\int_{z_0}^{z_1}f(z)\, dz =\int_a^bf(z(t))\, z'(t)\, dt = \int_a^bF'(z(t))\, z'(t)\, dt = 
$$

$$
\int^b_a\frac{d\, F(z(t))}{d\, t} = F(z(b))-F(z(a)) = F(z_1) -F(z_0).$$
$$\blacksquare$$

**Relación con integrales curvilineas en $\mathbb{R}^2$**

Sea $f:D\subset\mathbb{C}\to\mathbb{C}$ continua en $D$ conexo y abierto y sea $\Gamma$ una curva en $D$ parametrizada por $z:[a, b]\to\mathbb{C}$, entonces si $f = u +i\, v$ y $z =x +i\, y$ se tiene que 

$$
\int_\Gamma f(z)\, dz = \int_a^b [u(x(t), y(t)) + i\, v(x(t), y(t))]\, [x'(t) +i\, y'(t)]\, dt=
$$

$$
\int_a^b(u\, x'(t) - v\,y'(t)) \, dt + i\,\int_a^b (u\, y'(t) +v\, x'(t))\, dt =
$$

$$
\int_{\Gamma}(u\, dx -v\, dy) +i\, \int_{\Gamma}(v\,dx+ u\, dy)
$$

**Teorema de Green**

Sea $(P, Q)$ un campo vectorial en $C^1(\Omega\subset\mathbb{R}^2$) con $\Omega$ abierto. Sea $\zeta$ una curva cerrada simple que encierra una región $D$, donde $\zeta\cup D\subset \Omega$. Entonces,
$$
\int_{\zeta^{+}}(P\, dx +Q\, dy) =\int\int_{D}(\frac{\partial Q}{\partial x}-\frac{\partial P}{\partial y})dx\, dy
$$

**Teorema de Cauchy**

Sea $f:A\subset\mathbb{C}\to\mathbb{C}$ con $A$ abierto y $f = u + i\,v $ holomorfa en $A$. Sea $\Gamma\subset A$ una curva simple cerrada tal que la región encerrada por $\Gamma$ está contenida en $A$. Supongamos además que $u$ y $v$ son $C^{1}(A)$. Entonces 
$$
\int_{\Gamma}f(z)\, dz =0.
$$

**Demostración**

Sea $D$ la región encerrada por $\Gamma$. Entonces, por el Teorema de Green,
$$
\int_{\Gamma^{+}}f(z)\, dz = \int_{\Gamma^{+}}(u\, dx - v\, dy) + i\, \int_{\Gamma^{+}}(v\, dx + u\, dy) =
$$
$$
-\int\int_{D}(\frac{\partial v}{\partial x}+\frac{\partial u}{\partial y})dx\, dy +i\, \int\int_{D}(\frac{\partial u}{\partial x}-\frac{\partial v}{\partial y})dx\, dy.
$$
Y dado que $f$ es holomorfa ambas integrales se anulan porque $u$ y $v$ satisfacen las condiciones de C-R.
$$\blacksquare$$

Se puede prescindir de que $u$ y $v$ sean $C^{1}$ y que la curva que encierra la región $D$ sea simple.

**Teorema de Cauchy-Goursat**

Sea $f:A\subset\mathbb{C}\to\mathbb{C}$ holomorfa en $A$ abierto y sea $\Gamma\subset A$ una curva cerrada tal que la región encerrada por $\Gamma$ está contenida en $A$. Entonces 
$$
\int_{\Gamma}f(z)\, dz =0.
$$

**Fórmula de Cauchy**

Sea $f:A\subset\mathbb{C}\to\mathbb{C}$ holomorfa en $A$ abierto y sea $\Gamma$ una curva simple cerrada tal que la región que encierra está contenida en $A$; si $z_0$ es un punto interior a $\Gamma$, entonces
$$
f(z_0)=\frac{1}{2\, \pi\, i}\int_{\Gamma^{+}}\frac{f(z)}{z-z_0}\, dz.
$$

**Demostración**

Sea $\tilde{\Gamma} = \Gamma^{+} + \sigma + C(z_0,r)^{-}-\sigma$, donde $C(z_0,r)^{-}$ es la circunferencia de centro $z_0$ y radio r chico y $\sigma$ es un segmento que une $\Gamma$ y la circunferencia. Entonces, como $\tilde{\Gamma}$ es una curva cerrada y $\frac{f(z)}{z-z_0}$ es holomorfa en su interior, pues esquivamos la singularidad; por el Teorema de Cauchy-Gourzat

$$
\int_{\tilde\Gamma}\frac{f(z)}{z-z_0}\, dz=0.
$$
Por lo tanto si tenemos en cuenta que $\int_{\sigma} = -\int_{\sigma^-}$, entonces se llega a que
$$
\int_{\Gamma^{+}}\frac{f(z)}{z-z_0}\, dz = \int_{C(z_0,r)^{+}}\frac{f(z)}{z-z_0}\, dz.
$$
Pero,

$$
\int_{C(z_0,r)^{+}}\frac{f(z)}{z-z_0}\, dz = \int_{C(z_0,r)^{+}}\frac{f(z)-f(z_0)}{z-z_0}\, dz + \int_{C(z_0,r)^{+}}\frac{f(z_0)}{z-z_0}\, dz=
$$

$$
\int_{C(z_0,r)^{+}}\frac{f(z)}{z-z_0}\, dz = \int_{C(z_0,r)^{+}}\frac{f(z)-f(z_0)}{z-z_0}\, dz + f(z_0)\, 2\, \pi\, i.
$$

Por lo que si mostramos que la primer integral después de la igualdad se anula acuando $r\to0$, terminamos. Para eso, recordando que estamos integrando sobre la circunferencia de radio $r$ centrada en $z_0$
$$
|\int_{C(z_0,r)^{+}}\frac{f(z)-f(z_0)}{z-z_0}dz|\leq \underset{z\in \mathbb{C}}{sup}\, |\frac{f(z)-f(z_0)}{z-z_0}|\, 2\, \pi\, r = 
$$
$$
\underset{z\in \mathbb{C}}{sup}\, |\frac{f(z)-f(z_0)}{r}|\, 2\, \pi\, r = \underset{z\in \mathbb{C}}{sup}\, |f(z)-f(z_0)|\, 2\, \pi \underset{r\to0}{\to 0}
$$
porque $f$ es continua en $z_0$. Finalmente,
$$
f(z_0)\, 2\, \pi\, i=\int_{\Gamma^{+}}\frac{f(z)}{z-z_0}\, dz.
$$
$$\blacksquare$$

**Paso al límite bajo la integral**

Consideremos una sucesión de funciones $f_n: A\subset\mathbb{C}\to\mathbb{C}$, con $f_n$ continua en $A\quad \forall n\in \mathbb{N}$. Supongamos que $f_n\to f$ uniformemente en $A$. Entonces, dada una curva $\Gamma\subset A$,

$$
\lim_{n\to\infty}\int_{\Gamma}f_{n}(z)\, dz = \int_{\Gamma}f(z)\, dz.
$$

**Demostación**
$$
|\int_{\Gamma}f_{n}(z)\, dz - \int_{\Gamma}f(z)\, dz| = |\int_\Gamma (f_{n}(z)-f(z))\, dz|\leq
$$
$$
\underset{z\in\Gamma}{sup}|f_{n}(z)-f(z)|\, Long(\Gamma)\underset{n \to\infty}{\rightarrow} 0
$$
porque $f_n\to f$ uniformemente en $A$.

**Teorema de Taylor**

Sea $f:A\subset\mathbb{C}\to\mathbb{C}$ holomorfa en $A$ abierto. Entonces $f$ es analítica en $A$ y su serie de Taylor centrada en $z_0\in A$ converge hacia $f$ en cada disco $D(z_0, R)\subset A$.

**Demostración**

Sea $z_0$ tal que $D(z_0, R)\subset A$ y sea $z$ tal que $|z-z_0|<R$. Entonces consideremos $r$ y $r_0$ tal que $0<|z-z_0|<r_0<r<R$. Entonces si consideramos $C_r =\{w: |w-z_0| = r\}$ la circunferencia de radio $r$ centrada en $z_0$ por la fórmula de Cauchy tenemos que
$$
f(z)= \frac{1}{2\, \pi\, i}\, \int_{C_r^{+}}\frac{f(w)}{w-z}\, dw.
$$
Usando que $|z-z_0|<r = |w-z_0|$ podemos escribir
$$
\frac{1}{w-z} = \frac{1}{(w-z_0)-(z-z_0)}= \frac{1}{w-z_0}\, \frac{1}{1-\frac{z-z_0}{w-z_0}} =
$$
$$
\frac{1}{w-z_0}\, \sum^{\infty}_{n=0}(\frac{z-z_0}{w-z_0})^{n} = \sum^{\infty}_{n=0}\frac{(z-z_0)^{n}}{(w-z_0)^{n+1}}
$$
$$
\implies \frac{1}{w-z} = \sum^{\infty}_{n=0}\frac{(z-z_0)^{n}}{(w-z_0)^{n+1}},
$$
Por el criterio de Weierstrass, la serie converge uniformemente para $w\in C_r$ porque
$$
|\frac{(z-z_0)^{n}}{(w-z_0)^{n+1}}| = \frac{1}{r}\, (\frac{|z-z_0|}{r})^{n}\leq \frac{1}{r}\, (\frac{r_0}{r})^{n} = M_n
$$
y $0<\frac{r_0}{r}<1$, lo cual permite notar que la serie $\sum_n M_n$ converge.

Usando la proposición del paso al límite
$$
f(z)=\frac{1}{2\, \pi\, i}\, \int_{C_r^{+}}\frac{f(w)}{w-z}\, dw = \frac{1}{2\, \pi\, i}\int_{C_r^{+}}f(w)\, \sum^{\infty}_{n=0}\frac{(z-z_0)^{n}}{(w-z_0)^{n+1}}\, dw=
$$
$$
\frac{1}{2\, \pi\, i}\sum^{\infty}_{n=0}\int_{C_r^{+}}f(w)\, \frac{(z-z_0)^{n}}{(w-z_0)^{n+1}}\, dw=
$$
$$
\sum^{\infty}_{n=0}\left(\frac{1}{2\, \pi\, i}\,\int_{C_r^{+}}\frac{f(w)}{(w-z_0)^{n+1}}\, dw \right) (z-z_0)^{n} 
$$
Es decir, probamos que $f$ es analítica,
$$
f(z)=\sum^{\infty}_{n=0}a_n\, (z-z_0)^{n}
$$
con
$$
a_n = \frac{1}{2\, \pi \, i}\int_{C_r^{+}}\frac{f(w)}{(w-z_0)^{n+1}}\, dw.
$$
Como el integrando es holomorfo en $D(z_0,R)\setminus\{z_0\}$, por el Teorema de Cauchy-Goursat, $a_n$ da el mismo valor para cualquier $0<r<R$.
Finalmente, recordando el resultado visto para series de potencias, $f$ tiene derivadas de todos los órdenes en $D(z_0,R)$ y se tiene que 
$$
a_{n} = \frac{f^{n}(z_0)}{n!}.
$$
Por lo tanto, $f$ coincide con su serie de Taylor centrada en $z_0$ dentro del disco $D(z_0,R)$.
$$\blacksquare$$

**Fórmula de Cauchy generalizada**

Sea $f:A\subset\mathbb{C}\to\mathbb{C}$ holomorfa en $A$ abierto y sea $\Gamma\subset A$ una curva cerrada simple, tal que la región encerrada por $\Gamma$ está contenida en A. Si $z_0$ es un punto interior, entonces

$$
f^{(n)}(z_0) = \frac{n!}{2\, \pi\, i}\int_{\Gamma^{+}}\frac{f(w)}{(w-z_0)^{n+1}}\ dz\quad \text{con}\quad n\geq 0.
$$

**Demostración**

Por el Teorema de Taylor tenemos que 
$$
\frac{f^{n}(z_0)}{n\, !}=a_n=\frac{1}{2\, \pi\,i}\int_{C_r^{+}}\frac{f(w)}{(w-z_0)^{n+1}}\, dw
$$
con $C_r =\{w:|w-z_0| = r\}$. Ahora, por el Teorema de Cauchy-Goursat,
$$
\int_{C_r^{+}}\frac{f(w)}{(w-z_0)^{n+1}}\, dw = \int_{\Gamma^{+}}\frac{f(w)}{(w-z_0)^{n+1}}\, dw
$$
ya que podemos unir la curva $\Gamma$ con $C_r^{-}$ mediante un segmento para la ida y la vuelta que se termina anulando. La integral total da 0, porque allí $f$ es holomorfa, entonces se llega al resultado de arriba.
$$\blacksquare$$

**Observaciones**

El Teorema de Taylor prueba que si $f$ es holomorfa en $z_0$, entonces $f$ es analítica en $z_0$ y su radio de convergencia es igual a la distancia más próxima a una singularidad. En particular si $f$ es holomorfa en $\mathbb{C}$ entonces $f$ es entera (analítica con radio $\infty$). 

En fin, si $A$ es abierto, entonces 
$$
f\quad \text{es holomorfa en}\quad A \iff f\quad \text{es analítica en}\quad A.
$$

**Teorema de Morera**

Sea $f:A\subset\mathbb{C}\to\mathbb{C}$ continua en $A$ abierto y conexo. Si suponemos que 
$$
\int_{\Gamma} f(z)\,dz = 0 
$$
para toda curva cerrada $\Gamma\subset A$. Entonces $f$ es holomorfa en $A$.

**Demostración**

Probamos que bajo estas hipótesis, $f$ admite una primitiva $F$ holomorfa en 
$A$. Pero al ser $F$  holomorfa en $A$, por el Teorema de Taylor, tiene derivadas de todos los órdenes en $A$. Entonces $f = F'$ es holomorfa en $A$.
$$\blacksquare$$

**Desigualdad de Cauchy**
Si $f$ es analítica en $A$ abierto y sea $C_R$ la circunferencia de radio $R$ y centro $z_0$ con $C_R\subset A$ entonces 

$$
|f(z)|\leq M\quad\forall z\in\mathbb{C}\implies|f^{k}(z_0)|\leq\frac{k!\, M}{R^{k}}\quad\forall k\in\mathbb{N}.
$$

**Demostración**

Por la fórmula de Cauchy generalizada
$$
f^{(k)}(z_0)=\frac{k!}{2\, \pi\, i}\int_{C^{+}_R}\frac{f(w)}{(w-z_0)^{k+1}}dw.
$$
Entonces, como $|w-z_0|=R$
$$
|f^{k}(z_0)| = \frac{k!}{|2\, \pi\, i|}\, |\int_{C^{+}_R}\frac{f(w)}{(w-z_0)^{k+1}}\, dw|\leq
$$
$$
\frac{k!}{2\, \pi}\, \int_{C^{+}_R}|\frac{f(w)}{(w-z_0)^{k+1}}\, dw|\leq\frac{k!}{2\, \pi}\, \frac{M}{R^{k+1}}\int_{C^{+}_R}dw =
$$
$$
\frac{k!\, M\, Long(C^{+}_R)}{2\, \pi\ R^{k+1}} = \frac{M\, k!}{R^{k}}
$$
$$\blacksquare$$

**Teorema de Liouville**

Si $f$ es entera y $\exists M>0$ tal que $|f(z)|<M \quad \forall z\in \mathbb{C}\implies f$  es constante.

**Demostración**

Para cualquier $z_0\in\mathbb{C}$, por la Desigualdad de Cauchy
$$
|f'(z_0)|\leq \frac{M}{R}\underset{R\to\infty}{\to}0.
$$
Luego, $f(z)$ es constante.

**Principio del módulo máximo**

Sea $f:A\subset\mathbb{C}\to\mathbb{C}$ holomorfa en $A$ abierto y conexo. Supongamos que existe $z_0\in A$ tal que $|f|$ alcanza su máximo en $z_0$. Entonces $f$ es constante en $A$. Es decir, $f(z)\equiv f(z_0)$ para todo $z\in A$.

**FALTAN EJEMPLOS**

###  **Desarrollo de Laurent**

**Teorema de Laurent**

Sea $f$ una función holomorfa en la corona $D(z_0, R_1 ,R_2) =\{z\in\mathbb{C}: R_1<|z-z_0|<R_2\}$ para ciertos $0\leq R_1<R_2 \leq + \infty$. Entonces
$$
f(z)=\sum_{n=0}^{\infty}a_n\, (z-z_0)^{n} + \sum_{n=1}^{\infty}a_{-n}\, (z-z_0)^{n}= \sum_{n=-\infty}^{\infty}a_n\, (z-z_0)^{n},
$$
donde 
$$
a_n = \frac{1}{2\, \pi\, i}\, \int_{C^{+}_r}\frac{f(w)}{(w-z_0)^{n+1}}\, dw\quad n\in\mathbb{Z}
$$

y $C_r$ es la circunferencia de centro $z_0$ y radio $R_1<r<R_2$.

**Demostración**

Fijemos $r_1$ y $r_2$ tales que $R_1<r_1<r_2<R_2$. Veremos que existe un desarrollo de Laurent uniformemente convergente en el anillo $r_1<|z-z_0|<r_2$. Luego, como dicho desarrollo está univocamente determinado por $f$, el mismo no dependerá de los $r_1$ y $r_2$ elegidos. De este modo el desarrollo será válido para $D(z_0,R_1,R_2)$. 
Tomemos $\rho_1$ y $\rho_2$ tales que
$$
R_1<\rho_1<r_1<r_2<\rho_2<R_2.
$$
Y ahora tomemos $\tilde{\Gamma} = C^{+}_{\rho_1} + \sigma + C^{-}_{\rho_2} - \sigma$ con $C_{\rho_1}$ y $C_{\rho_2}$ dos circunferencias de centro $z_0$ y radios $\rho_1$ y $\rho_2$ respectivamente y $\sigma$ un segmento que une ambas circunferencias. La curva recién definida es una curva simple y cerrada y la función $\frac{f(z)}{z-z_0}$ es holomorfa en su interior. Entonces, por la fórmula de Cauchy,
$$
f(z) =\frac{1}{2\, \pi\, i} \int_{\tilde{\Gamma}}\frac{f(w)}{w-z}\, dw = \frac{1}{2\, \pi\, i} \int_{C_{\rho_1}^{+}}\frac{f(w)}{w-z}\, dw + \frac{1}{2\, \pi\, i} \int_{C_{\rho_2}^{-}}\frac{f(w)}{w-z}\, dw
$$
porque las integrales sobre $\sigma$ se anulan.
$$
\implies f(z) = \frac{1}{2\, \pi\, i} \int_{C_{\rho_1}^{+}}\frac{f(w)}{w-z}\, dw + \frac{1}{2\, \pi\, i} \int_{C_{\rho_2}^{+}}\frac{f(w)}{z-w}\, dw.
$$

Ahora hay que separar en dos casos. Si $w\in C_{\rho_2}$, usando que $|z-z_0|<\rho_2 = |w-z_0|$, notamos que
$$
\frac{1}{w-z} = \frac{1}{w-z_0 - (z - z_0)}= \frac{1}{w-z_0} \, \frac{1}{1 - \frac{z - z_0}{w-z_0}} = \sum_{n=0}^{\infty} \frac{(z - z_0)^{n}}{(w-z_0)^{n+1}}.
$$
En cambio, si $w\in C_{\rho_1}$, usando que $|w-z_0|=\rho_1<|z-z_0|$, notamos que
$$
\frac{1}{w-z} = \frac{1}{w-z_0 - (z - z_0)}= \frac{1}{z-z_0} \, \frac{-1}{1 - \frac{w - z_0}{z-z_0}} = - \sum_{n=0}^{\infty} \frac{(w - z_0)^{n}}{(z-z_0)^{n+1}}.
$$
En ambos casos, la serie converge uniformemente por el criterio de Weierstrass. En el primer caso,
$$
|\frac{(z-z_0)^{n}}{(w-z_0)^{n+1}}| = \frac{1}{\rho_2}\, (\frac{|z-z_0|}{\rho_2})^{n}\leq \frac{1}{\rho_2}\, (\frac{r_2}{\rho_2})^{n} = M_n
$$
y $0<\frac{r_2}{\rho_2}<1$, lo cual permite determinar que la serie $\sum_n M_n$ converge. En el segundo,
$$
|\frac{(w-z_0)^{n}}{(z-z_0)^{n+1}}| = \frac{1}{|z-z_0|}\, (\frac{\rho_1}{|z-z_0|})^{n}\leq \frac{1}{r_1}\, (\frac{\rho_1}{r_1})^{n} = \tilde{M}_n
$$
y $0<\frac{\rho_1}{r_1}<1$, lo cual permite determinar que la serie $\sum_n \tilde{M}_n$ converge.

Por lo tanto, reemplazando ambos resultados, llegamos a que
$$
f(z) = \frac{1}{2\, \pi\, i}\int_{C_{\rho_2}^{+}}\frac{f(w)}{w-z}\, dw-\frac{1}{2\, \pi\, i}\int_{C_{\rho_1}^{+}}\frac{f(w)}{w-z}\, dw=
$$
$$
\frac{1}{2\, \pi\, i}\int_{C_{\rho_2}^{+}}f(w)\sum_{n=0}^{\infty}\frac{(z-z_0)^{n}}{(w-z_0)^{n+1}}\, dw+\frac{1}{2\, \pi\, i}\int_{C_{\rho_1}^{+}}f(w)\sum_{n=0}^{\infty}\frac{(w-z_0)^{n}}{(z-z_0)^{n+1}}\, dw
$$
Usando la proposición de paso a límite bajo la integral tenemos que
$$
f(z) = \frac{1}{2\, \pi\, i}\sum^{\infty}_{n=0}\int_{C^{+}_{\rho_2}}f(w)\,\frac{(z-z_0)^{n}}{(w-z_0)^{n+1}}\, dw + \frac{1}{2\, \pi\, i}\sum^{\infty}_{n=0}\int_{C^{+}_{\rho_1}}f(w)\,\frac{(w-z_0)^{n}}{(z-z_0)^{n+1}}\, dw = 
$$
$$
\sum^{\infty}_{n=0} \left( \frac{1}{2\, \pi\, i} \int_{C^{+}_{\rho_2}}\frac{f(w)}{(w-z_0)^{n+1}}\, dw\right)\, (z-z_0)^{n} +\sum^{\infty}_{n=1} \left( \frac{1}{2\, \pi\, i} \int_{C^{+}_{\rho_1}}\frac{f(w)}{(w-z_0)^{-n+1}}\, dw\right)\, (z-z_0)^{-n}. 
$$
Como los integrandos son holomorfos en $D(z_0, R_1, R_2)$, por el teorema de Cauchy-Goursat, para el calculo de $a_n$ podemos integrar sobre $C_r$ para cualquier $R_1<r<R_2$.
$$
\implies f(z) = \sum_{n\in\mathbb{Z}} \left( \frac{1}{2\, \pi\, i} \int_{C^{+}_{r}}\frac{f(w)}{(w-z_0)^{n+1}}\, dw\right)\, (z-z_0)^{n}
$$

$$\blacksquare$$

**Observaciones** 

El desarrollo de Laurent es único. Además, los desarrollos de Taylor y Laurent coinciden. Para ver esto consideremos una función $f$ holomorfa en $|z-z_0|<R_2$ que entonces admite un desarrollo de Taylor. Pero $f$ también es holomorfa en $R_1<|z-z_0|<R_2$ y entonces también admite un desarrollo de Laurent en dicha corona. De hecho, como $f$ es holomorfa en el disco $|z-z_0|<R_2$, los coeficientes $a_{-n}$ con $n\geq 1$ del desarrollo de Laurent satisfacen

$$
a_{-n}=\frac{1}{2\, \pi\, i}\int_{C^{+}_r}\frac{f(w)}{(w-z_0)^{-n+1}}\, dw = \frac{1}{2\, \pi\, i}\int_{C^{+}_r}f(w)(w-z_0)^{n-1}\, dw = 0,
$$
por el teorema de Cauchy-Goursat.

**EJEMPLOS**

### Clasificación de singularidades
**¿Qué es una singularidad??**

Es un punto donde la función tiene algún problema de definición. En particular, se dice que $f$ es singular en $z_0$ si no es holomorfa en ese punto.

**Singularidades aisladas y no aisladas**

Se dice que $z_0$ es una singularidad aislada de $f$ si existe $R>0$ tal que $f$ es holomorfa en el disco reducido (disco menos el centro) $0<|z-z_0|<R$. Y es no aislada si no existe ningún disco reducido alrededor de $z_0$ en el cual $f$ es holomorfa.

En el caso de tener una singularidad aislada el disco reducido $0<|z-z_0|<R$ que es una corona, es una región donde $f$ admite un desarrollo en series de Laurent, puesto que $f$ es holomorfa allí. Los coeficientes $a_{-n}$ con $n\geq 1$ determinan la parte singular o principal del desarrollo (potencias negativas) que permite clasificar las singularidades.

**Singularidad aislada: evitable**

Diremos que $z_0$ es una singularidad evitable si $a_{-n} = 0\quad \forall n\geq 1$. Es decir, no hay potencias negativas en el desarrollo (parte singular nula).

**Singularidad aislada: polo de orden m**

Diremos que $z_0$ es un polo de orden $m\geq1$ si $a_{-m} \neq 0$ y $a_{-n}=0 \quad\forall n>m$. Es decir, hay finitas potencias negativas en su desarrollo, la última de ellas indica el orden del polo.

**Singularidad aislada: esencial**

Diremos que $z_0$ es una singularidad esencial si existen infinitos $n\geq 1$ tales que $a_{-n}\neq 0$. Osea, infinitas potencias en el desarrollo.

**Proposicones sobre el comportamiento de los polos**

Además se pueden clasificar las singularidades acorde a cómo se comportan cerca del punto singular.

**Proposición: comportamiento cerca de singularidades evitables**

Sea $z_0$ una singularidad aislada de $f$ es equivalente que

1. $z_0$ es una singularidad evitable de f.

2. Existe el límite 
$$
\lim_{z\to z_0}f(z) = w_0\in \mathbb{C}.
$$

3. $|f|$ está acotada en $0<|z-z_0|<\delta$ para algún $\delta>0$.

**Demostración**

**$1\implies2)$**

Si $z_0$ es una singularidad evitable de $f$, en $0<|z-z_0|<R$ el desarrollo de Laurent es
$$
f(z)=\sum^{\infty}_{n=0}a_{n}\, (z-z_0)^{n}.
$$
Entonces $f$ es continua en $z_0$ y existe
$$
\lim_{z\to z_0}f(z)=w_0\in\mathbb{C}.
$$

**$2\implies 3)$**

Si existe el límite de arriba, dado $\epsilon>0,\quad \exists \delta>0: |f(z)-w_0|<\epsilon$ si $0<|z-z_0|<\delta$.

Entonces, 
$$
|f(z)|\leq|f(z)-w_0| + |w_0|<\epsilon +|w_0|,
$$
si $0<|z-z_0|<\delta$.

**$3\implies 1)$**

Supongamos que $|f(z)|\leq M$ si $0<|z-z_0|<\delta$, con $M>0$.

Sabemos que 
$$
f(z) = \sum_{n\in \mathbb{Z}}a_n\, (z-z_0)^{n}
$$
en $0<|z-z_0|<R$, con

$$
a_n = \frac{1}{2\, \pi\, i}\int_{C_r^{+}}\frac{f(w)}{(w-z_0)^{n+1}}\ dw, n\in\mathbb{Z},
$$
donde $C_r$ es la circunferencia de radio $0<r<R$ y centro $z_0$.
Para $n\geq 1$, si $0<r<\delta$, se tiene que 
$$|a_{-n}| = |\frac{1}{2\, \pi\, i}\int_{C_r^{+}}\frac{f(w)}{(w-z_0)^{-n+1}}| =
$$

$$|\frac{1}{2\, \pi\, i}\int_{C_r^{+}}f(w)\, (w-z_0)^{n-1}\, dw| \leq
$$

$$
\frac{1}{2\, \pi}\underset{w\in C_r}{sup}|f(w)\, (w-z_0)^{n-1}|\, 2\, \pi\, r\leq
$$
$$
 M\, r^{n}\to 0
$$
cuando $r\to 0$. Es decir, $a_{-n} = 0$ para todo $n\leq 1$ y por lo tanto $z_0$ es una singularidad evitable.
$$\blacksquare$$

**Proposición: comportamiento cerca de singularidades polos**

Sea $z_0$ una singularidad aislada de $f$ es equivalente que

1. $z_0$ es un polo de $f$.

2. El límite da
$$
\lim_{z\to z_0}f(z) = \infty.
$$

3. La funcion $\frac{1}{f}$, completada con el valor 0 en $z = z_0$, es holomorfa en $z_0$.

**Demostración**

**$1\implies2)$**

Si $z_0$ es un polo de orden $m\geq 1$, el desarrollo de Laurent en la corona $0<|z-z_|<R$ es 
$$
f(z)=\sum^{\infty}_{n=-m}a_n\, (z-z_0)^{n} = \sum^{\infty}_{k=0}a_{k-m}\, (z-z_0)^{k-m}=
$$
$$
(z-z_0)^{-m} \sum^{\infty}_{k=0}a_{k-m}\, (z-z_0)^{k}=(z-z_0)^{-m}\, \eta(z).
$$
Es decir, que 
$$
f(z) = (z-z_0)^{-m}\, \eta(z)\quad \text{y}\quad \eta(z)=\sum^{\infty}_{k=0}a_{k-m}\, (z-z_0)^{k},
$$
con $\eta(z)$ una función continua en $z_0$ y, además, $\eta(z_0)=a_{-m}\neq 0$ puesto que $z_0$ es un polo de orden $m.$ Por lo tanto,
$$
\lim_{z\to z_0} f(z) = \infty.
$$
**$2\implies3)$**

Partimos de que $f(z)\underset{z\to z_0}\to\infty$, entonces 
$$
\exists \,\delta : 0<|z-z_0|<\delta  \implies |f(z)|\geq 1,
$$
es decir que no se anula, por lo que la función $1/f$ resulta holomorfa en $D(z_0,\delta)$ y, además,

$$
\lim_{z\to z_0}\frac{1}{f(z)}=0.
$$

Por la proposición del comportamiento cerca de singularidades evitables, sabemos que $1/f$ tiene una singularidad evitable en $z_0$. Entonces,

$$
g(z) = \begin{cases}
 & \frac{1}{f}\quad\text{si } z\in D(z_0,\delta) \\
 & 0\quad \text{si } z = z_0 
\end{cases} 
$$
es holomorfa en $z = z_0$.

**$3\implies1)$**

Haciendo uso de la definición de $g(z)$ dada más arriba, como g(z_0) = 0 y $g \not\equiv 0$, al ser holomorfa, existe $m\geq 1$ tal que $g^{(m)}(z_0) \neq 0$ y $g^{(k)}(z_0) = 0$ para $0\leq k <m$ (la letra $m$ no se elijió pq si jeje).

Entonces, en $|z-z_0|<\delta$, por lo visto en el ítem anterior vale que

$$
g(z) = \sum^{\infty}_{n=0}a_n\,(z-z_0)^{n}=\sum^{\infty}_{n=m}a_n\,(z-z_0)^{n}=
$$

$$
= \sum^{\infty}_{k=0}a_{k+m}\,(z-z_0)^{k+m}=(z-z_0)^{m}\sum^{\infty}_{k=0}a_{k+m}\,(z-z_0)^{k}=
$$

$$
(z-z_0)^{m}\, \varphi(z)
$$
con $\varphi$ holomorfa en $z_0$ y $\varphi(z_0) = a_m\neq 0$. Por lo tanto,
$$
f(z) = (z-z_0)^{-m}\, \frac{1}{\varphi(z)}= (z-z_0)^{-m}\, \eta(z) \quad\text{en}\quad D(z_0,\delta)
$$
con $\eta(z)$ holomorfa en $z_0$ y $\eta(z_0)\neq0$. Es decir, que $z_0$ es un polo de $f$ de orden $m$.

$$\blacksquare$$

**Observaciones**

Si $f$ es holomorfa en $z_0$ y $f \not \equiv 0$ diremos que $z_0$ es un cero de $f$ de orden $m\geq 1$ si $f^{(k)}(z_0)=0$ para $0\leq k<m$ y $f^{(m)}(z_0)\neq 0$. Se tiene
$$
z_0\quad \text{polo de $f$ de orden $m$}\Longleftrightarrow z_0\quad \text{cero de $\frac{1}{f}$ de orden $m$}.
$$
También se deduce que si $z_0$ es una singularidad aislada de $f$ 
$$
z_0\quad \text{polo de $f$ de orden $m$}\Longleftrightarrow \lim_{z\to z_0}(z-z_0)^{m}\, f(z) = w_0\in \mathbb{C},\quad w_0\neq0.
$$

**Corolario: comportamiento cerca de singularidades esenciales**

Sea $z_0$ una singularidad aislada de $f$, resulta equivalente que

1. $z_0$ es una singularidad esencial de $f$.
2. No existe $\lim_{z\to z_0}f(z)$-

**Demostración**

Es una singularidad y si sucede cualquiera de las dos hipótesis se rompen las hipótesis de las proposiciones de arriba. Entonces como $z_0$ es una singularidad aislada y no es evitable o un polo, es esencial.

$$\blacksquare$$

**Holomorfía en el infinito**

Diremos que la función $f$ es holomorfa en el infinito si la función $F(w) := f(\frac{1}{w})$ es holomorfa en $w=0.$

**Singularidad aislada en el infinito**

Diremos que la función $f$ tiene una singularidad en el infinito, si la función $F(w) := f(\frac{1}{w})$ tiene una singularidad aislada en $w=0.$
La singularidad será evitable, un polo o esencial en el infinito si $F(w)$ tiene el mismo tipo de singularidad, pero en $w= 0.$


En otras palabras, $f$ tiene una singularidad aislada en el infinito si $f$ es holomorfa en $|z|>R$ para algun $R>0$ (admite un desarrollo de Laurent en la corona $|z|>R$).

**EJEMPLOS**

### **Residuos**

**Un ejemplo fundamental**

Si $z_0$ es una singularidad aislada de $f$, entonces holomorfa en el disco reducido $0<|z-z_0|<R$. Es decir que admite un desarrollo de Laurent cuyos coeficientes serán
$$
a_n = \frac{1}{2\, \pi\, i}\int_{C_r^{+}}\frac{f(w)}{(w-z_0)^{(n+1)}}\, dw,\quad n\in\mathbb{Z}.
$$

En particular,
$$
a_{-1} = \frac{1}{2\, \pi\, i}\int_{C_r^{+}}f(w)\, dw.
$$

Esto tiene mucho sentido, pues si hacemos el cálculo explícito tenemos que

$$
\int_{C_r^{+}}f(z)\, dz = \int_{C_r^{+}}\sum_{n\in\mathbb{Z}}a_{n}\,(z-z_0)^{n} dz= \sum_{n\in\mathbb{Z}}a_{n}\, \int_{C_r^{+}}(z-z_0)^{n} dz=
$$

$$\sum_{n\in\mathbb{Z}}a_{n}\,\cdot
\begin{cases}
 &  2\, \pi\, i \quad\text{si } n= -1\\
 & 0\quad \text{si } n \neq0
\end{cases}= a_{-1}\, 2\, \pi\, i.
$$

**Residuo**

Se llama residuo de $f$ en la singularidad $z_0$ al coeficiente $a_{-1}$ en su desarrollo de Laurento alrededor del punto singular. Es decir,
$$
Res(f, z_0)=a_{-1} = \frac{1}{2\, \pi\, i}\int_{C_{r}^{+}}f(w)\, dw
$$
con $C_r$ la circunferencia de radio $r>0$ centrada en $z_0$ que no contenga otra singularidad de $f$ en su interior.

**Proposicion**

Si $z_0$ es un polo de orden $m$ de $f$ y definimos $g(z)=(z-z_0)^{m}\, f(z)$ entonces
$$
Res(f, z_0) = \frac{1}{(m-1)!}\, g^{(m-1)}(z_0).
$$

**Demostración**

Para algún $R>0$ se tiene en el disco reducido $0<|z-z_0|<R$ que
$$
f(z) =\sum_{n =-m}^{\infty} a_{n}\, (z-z_0)^{n}=\sum_{k = 0}^{\infty} a_{k-m}\, (z-z_0)^{k-m}=
$$
$$
(z-z_0)^{-m}\, \sum_{k =0}^{\infty} a_{k-m}\, (z-z_0)^{k}= (z-z_0)^{-m}\, g(z).
$$

Es decir, 
$$
g(z) = f(z)\, (z-z_0)^{m}\quad\text{en}\quad 0<|z-z_0|<R
$$
con 
$$
g(z) = \sum_{k =0}^{\infty} a_{k-m}\, (z-z_0)^{k}.
$$
Notemos que $g$ es holomorfa en $|z-z_0|<R$ y, por lo tanto, por las propiedades de la serie de Taylor, tenemos que 
$$
\frac{g^{(k)}(z_0)}{k!} = a_{k-m} \quad \forall k\geq0.
$$
En particular,
$$
\frac{g^{(m-1)}(z_0)}{(m-1)!} = a_{-1}= Res(f, z_0).
$$
$$\blacksquare$$

**Observación**

Si $z_0$ es un polo simple, la proposición anterior toma la forma
$$
Res(f,z_0) = \lim_{z\to z_0} (z-z_0)\, f(z).
$$

**Teorema de los residuos**

Sea $f$ una función holomorfa en $D$ salvo un número finito de singularidades aisladas y sea $\Gamma$ una curva cerrada simple tal que tanto ella como su interior están contenidas en $D$ y no pasa por ninguna singularidad. Entonces, si $z_1,\dots, z_k$ son los puntos singulares dentro de $\Gamma$ se tiene que 

$$
\int_{\Gamma^{+}}f(z)\, dz = 2\, \pi\, i\sum_{j=1}^{k}Res(f, z_j)
$$

**Demostración**

Para $1\leq j\leq k$ consideramos $C_j$ una circunferencia centrada en $z_j$ contenida en el interior de $\Gamma$ y que no contenga otra singularidad más que $z_j$. Si unimos cada $C_j$ con $\Gamma$ mediante segmentos podemos construir una curva cerrada $\tilde{\Gamma}$, de modo que $f$ es holomorfa en la región limitada por $\tilde{\Gamma}$. Luego, por el Teorema de Cauchy-Goursat

$$
0 = \int_{\tilde{\Gamma}} f(z)\, dz = \int_{\Gamma^{+}} f(z)\, dz -\sum^{k}_{j=1}\int_{C_j^{+}} f(z)\, dz 
$$
$$
\implies \int_{\Gamma^{+}} f(z)\, dz =2\, \pi\, i\, \sum^{k}_{j=1}Res(f, z_j)
$$

$$\blacksquare$$

**Otro ejemplo importante**

Sea $f$ holomorfa en $|z|>R$, entonces $f$ admite un desarrollo de Laurent en esta corona con coeficientes
$$
a_n = \frac{1}{2\, \pi\, i}\int_{C_r^{+}}\frac{f(w)}{w^{n+1}}\, dw,\quad n\in\mathbb{Z},
$$
con $C_r$ la circunferencia de radio $r>R$. En particular
$$
a_{-1} = \frac{1}{2\, \pi\, i}\int_{C_r^{+}}f(w)\, dw.
$$

**Residuo en el infinito**

Sea $f$ holomorfa en $|z|>R$. Consideremos el coeficiente $a_{-1}$ en su desarrollo de Laurent en esta corona. Definimos el residuo de $f$ en el infinito como

$$
Res(f,\infty) = - a_{-1} = \frac{1}{2\ \pi\, i}\int_{C_r^{-}}f(w)\, dw
$$
con $C_r$ la circunferencia de radio $r>R$ y centro 0 (importante notar que estamos afuera de $R$ por eso el cambio de signo).

**Observación importante para el cálculo** 

Sea $f$ holomorfa en $|z|>R$ entonces
$$
f(z) = \sum^{\infty}_{n=0}a_n\, z^{n} + \sum^{\infty}_{n=1}a_{-n}\, z^{-n}.
$$
Notemos que 
$$
f(\frac{1}{z}) = \sum^{\infty}_{n=0}a_n\, z^{-n} + \sum^{\infty}_{n=1}a_{n}\, z^{-n}.
$$
si $0<|z|<\frac{1}{R}$. Entonces
$$
f(\frac{1}{z})\frac{1}{z^{2}}= \sum^{\infty}_{n=0}a_n\, z^{-n-2} + \sum^{\infty}_{n=1}a_{-n}\, z^{n-2}.
$$
Por lo tanto, 
$$
Res(f, \infty) = -a_{-1} = -Res(f(\frac{1}{z})\, \frac{1}{z^{2}},0).
$$

**Corolario: suma total de residuos**

Si $f$  tiene sólo un número finito de singularidades en $\mathbb{C}_\infty$ las sumas de los residuos en ese punto es nula.

**Demostración**

Sea $R>0$ tal que en $|z|<R$ están todas las singularidades $z_1,\dots,z_k\in\mathbb{C}$. El teorema de los residuos dice que
$$
\int_{C^{+}_R}f(z)\, dz = 2\, \pi\, i\, \sum_{j=1}^{k} Res(f, z_j),
$$
donde $C_R$ es la circunferencia de radio $R$ y centro 0. Pero,
$$
Res(f, \infty) = \frac{1}{2\, \pi\, i}\int_{C_R^{-}}f(z)\, dz= -\frac{1}{2\, \pi\, i}\int_{C_R^{+}}f(z)\, dz.
$$
En consecuencia,
$$
Res(f, \infty) +\sum_{j=1}^{k} Res(f, z_j) =0.
$$
$$\blacksquare$$

**EJEMPLOS** 

### **Series de Fourier**

**Pregunta de Fourier**

Fourier se preguntó cuándo es que podemos representar una función $f:\mathbb{R}\to\mathbb{R}$ como
$$
f(x) = \frac{a_0}{2} + \sum^{\infty}_{n=1}(a_n\, cos(n\pi\,x)+b_n\, sen(n\pi\,x)),
$$
bajo qué hipótesis sobre $f$, cómo es que se eligen los coeficientes y en qué sentido deberíamos interpretar la convergencia si es que existe.


**Proposición: ortogonalidad de senos, cosenos y 1**

Las funciones $\{1, cos(n\, \pi \, x), sen(n\, \pi \, x)\}, \quad n\in\mathbb{N}$ satisfacen, para $n$, $m \in\mathbb{Z}$

$$
\int^{1}_{-1}cos(n\, \pi \, x)\, cos(m\, \pi \, x)\, dx = \begin{cases}
 &  0 \quad\text{si } n\neq m\\
 &  1 \quad\text{si } n=m\neq 0\\
 &  2 \quad\text{si } n=m=0
\end{cases};
$$

$$
\int^{1}_{-1}sen(n\, \pi \, x)\, sen(m\, \pi \, x)\, dx = \begin{cases}
 &  0 \quad\text{si } n\neq m\\
 &  1 \quad\text{si } n=m\neq0
\end{cases};
$$

$$
\int^{1}_{-1}sen(n\, \pi \, x)\, cos(m\, \pi \, x)\, dx = 0\quad\forall\quad n,m.
$$

**Demostración**

Sale por medio de ralaciones trigonométricas deducibles por medio de escribir las funciones seno y coseno en términos de exponenciales complejas.

$$\blacksquare$$

**Producto escalar sobre espacios vectoriales**

Sea $\mathbb{V}$ un espacio vectorial sobre $\mathbb{R}$, entonces un producto escalar (o interno) sobre $\mathbb{V}$ es una aplicación $\langle·,·\rangle: \mathbb{V}\times \mathbb{V}\to \mathbb{R}$ que satisface

1. $\langle v,v\rangle\geq0\quad\forall v\in \mathbb{V}$,

2. $\langle v,v\rangle = 0 \Longleftrightarrow v= 0$,

3. $\langle a\, u +b\, v, w\rangle = a\, \langle u,w \rangle + b\, \langle v,w \rangle\quad \forall u, v, w \in \mathbb{V}\quad \text{y}\quad \forall a,b\in \mathbb{R}$,

4. $\langle u,v\rangle = \langle v,u\rangle\quad \forall u, v\in \mathbb{V}$.

En particular, $\mathbb{V} = \{ f:[a,b]\to \mathbb{R}: \text{f es continua a trozos}\}$  con
$$
\langle f, g \rangle = \int^{b}_{a}f(t)\, g(t)\, dt
$$
definen un espacio vectorial con producto escalar.


**Norma**

Sea $\mathbb{V}$ un espacio vectorial sobre $\mathbb{R}$, una norma sobre $\mathbb{V}$ es una aplicación $\|\cdot\|:\mathbb{V}\to\mathbb{R}$ que satisface

1. $\|v\|\geq 0\quad \forall v\in \mathbb{V},$

2. $\|v\| = 0\quad \forall \Longleftrightarrow v= 0,$

3. $\|a\, v\| = |a|\,\|v\| \quad \forall v \in \mathbb{V}\quad \text{y}\quad \forall a\in \mathbb{R},$

4. $\|u+v\|\leq \|u\| +\|v\|\quad \forall u,v\in \mathbb{V}.$


**Distancia**

Si $\mathbb{V}$ es un espacio vectorial y $\|\cdot\|$ es una norma sobre $\mathbb{V}$, definimos la función distancia como

$$
dist(u, w) = \|v-w\|\quad \text{con}\quad v,w\in \mathbb{V}.
$$

**Norma inducida por producto escalar**

Si $\mathbb{V}$ es un espacio vectorial con producto escalar $\langle\cdot, \cdot\rangle$ entonces

$$
\|v\| = \sqrt{\langle v, v\rangle }
$$

define una norma sobre $\mathbb{v}$.

**Sistema ortonormal**

Sea $\mathbb{V}$ un espacio vectorial con producto escalar $\langle\cdot,\cdot\rangle$ y sea $\{v_j\}_{j\in\mathbb{N}}$ una sucesión de elementos de $\mathbb{V}$. Diremos que $\{v_j\}_{j\in\mathbb{N}}$ es un sistema ortogonal si se cumple que 

$$
\langle v_j, v_k\rangle = 0 \quad\text{si}\quad j\neq k\quad \text{y}\quad \langle v_j, v_k\rangle \neq 0 \quad\forall j.
$$

Diremos que el sistema es ortonormal si además se tiene

$$
\langle v_j, v_j\rangle = 1 \quad \forall j.
$$

**Convergencia en media cuadrática en sucesión de funciones**

Sea $\{f_n\}_{n\in\mathbb{N}}$ una sucesión de funciones con $f_n:[a,b]\to \mathbb{R}$, decimos que la sucesión converge a $f$ en media cuadrática en $[a,b]$ si 
$$
\lim_{n\to \infty}\sqrt{\int_a^{b}(f_{n}(t)-f(t))\, dt} = 0;
$$
es decir, 
$$
\lim_{n\to \infty}\| f_n- f\| =0
$$
para la norma inducida por el producto escalar definido más arriba.

**Observación**

La convergencia en series es igual que para sucesiones de funciones, pero en este caso la sucesión está formada por la sucesión de sumas parciales (que son también funciones).

**Proposición: mejor aproximación en media cuadrática**

Sea $\mathbb{V}$ el espacio vectorial de las funciones continuas a trozos en el intervalo $[a,b]$ con el producto interno definido más arriba y la norma definida por éste producto. Entonces buscamos los coeficientes $\{c_n\}_{n\in \mathbb{N}}$ de modo que para cada $N\geq 1$
$$
\sum_{n=1}^{N} c_n\, \phi_n(x) 
$$
sea la mejor aproximación a $f$ en media cuadrática. Es decir que buscamos los coeficientes que para cada $N\geq 1$ minimicen 
$$
\| S_N- f\| = \| \sum_{n=1}^{N}  c_n\, \phi_n -f\|.
$$

Se puede demostrar que dichos coeficientes son 
$$
c_n = \langle f, \phi_n\rangle.
$$

**Demostración**

Fijado $N\geq 1$, buscamos $\{c_n\}_{n\geq 1}$ tales que 

$$
\| \sum_{n=1}^{N}  c_n\, \phi_n -f\|
$$
se minimiza. Podemos desarrollar esta expresión 

$$
\| \sum_{n=1}^{N}  c_n\, \phi_n -f\|^{2} = \langle \sum_{n=1}^{N}c_n\, \phi_n -f,\sum_{n=1}^{N}c_n\, \phi_n -f \rangle=
$$
$$
\langle \sum_{n=1}^{N}c_n\,\phi_n, \sum_{n=1}^{N}c_n\, \phi_n \rangle-2\, \langle f,\sum_{n=1}^{N}c_n\,\phi_n\rangle + \langle f, f\rangle = 
$$
$$
\|f\|^{2} - 2\, \sum_{n=1}^{N}c_n\langle f, \phi_n \rangle + \sum_{n=1}^{N}c_n^{2},
$$
donde se usó la ortonormalidad de las funciones $\phi_n$. Completando cuadrados obtenemos que 
$$
\| \sum_{n=1}^{N} c_n\, \phi_n -f\|^{2} = \|f\|^{2} + \sum_{n=1}^{N} (c_n - \langle f, \phi_n \rangle)^{2} - \sum_{n=1}^{N}\langle f, \phi_n\rangle^{2},
$$
donde queda en evidencia que el valor mínimo se alcanza para 
$$
c_n = \langle f, \phi_n\rangle \quad \forall n.
$$
En dicho caso queda 
$$
\|f - \sum^{N}_i = 1 c_n\, \phi_n\|^{2} = \|f\|^{2} - \sum_{n=1}^{N}c_n^{2}
$$
$$\blacksquare$$

**Corolario: desigualdad de Bessel**

Bajo las hipótesis de la proposición anterior, notando que $c_n = \langle f,\phi_n\rangle$, se tiene

1. la desigualdad de Bessel, $$\sum_{n=1}^{N}c^{2}_n\leq \|f\|^{2},$$

2. la serie $\sum_{n=1}^{\infty} c^{2}_n$ es convergente,

3. $c_n = \langle f, \phi_n\rangle\to 0$.

**Demostración**

El resultado se obtiene observando que
$$
0\leq \|f- \sum_{n=1}^{N}c_n\, \phi_n \|^{2} =\|f\|^{2} - \sum_{n=1}^{N}c_{n}^{2}
$$

$$\blacksquare$$

**Serie de Fourier**

Sea $\mathbb{V}$ un espacio vectorial con producto escalar $\langle \cdot, \cdot\rangle$ y sea $\{\phi_n \}_{n\geq 1}$ un sistema ortonormal en $\mathbb{V}$ y sea $f\in \mathbb{V}$. Se llama serie de Fourier en el sistema $\{\phi_n \}_{n\geq 1}$ a la serie
$$
\sum_{n=1}^{\infty}c_n\, \phi_n\quad\text{con}\quad c_n = \langle f,\phi_n\rangle.
$$
Se nota 
$$f\sim\sum_{n=1}^{\infty}c_n\, \phi_n.$$


**Serie de Fourier Trigonométrica en [-L,L]**

**Serie de Fourier exponencial**

**Observación**

Utilización del intervalo [-1,1] y que es l mismo para cualquier L

**Lema de Riemann Lebesgue**

**Demostración**

$$\blacksquare$$

**Corolario**

**Demostración**

$$\blacksquare$$

**Derivada lateral**

**Reescritura de la sucesión de sumas parciales**

**Núcleo de Dirichlet: propiedades**

**Demostración**

4

$$\blacksquare$$

**Teorema de convergencia puntual**


**Demostración**


$$\blacksquare$$

**Desigualdad de Cauchy-Scharz en $\mathbb{R}^{n}$**

**Observación sobre la derivada de la serie de Fourier**

**Teorema de convergencia uniforme**

**Demostración**


$$\blacksquare$$

**Completitud de un sistema ortonormal**

**Igualdad de Parseval**

**Demostración**

$$\blacksquare$$

**Completitud del sistema trigonométrico**

**EJEMPLOS**