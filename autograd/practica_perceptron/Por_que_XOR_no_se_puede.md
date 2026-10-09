# ¿Por qué un solo perceptrón no puede resolver XOR?

## 1. Qué calcula un perceptrón

Un perceptrón con dos entradas hace siempre lo mismo:

```
z = w0·x0 + w1·x1 + b        ← parte lineal
ŷ = f(z)                      ← función de activación (escalón, sigmoide, swish, ...)
```

Lo importante: **la activación se aplica sobre `z`, y `z` es una función lineal de las entradas**.
Geométricamente, la ecuación `w0·x0 + w1·x1 + b = 0` es una **recta** en el plano (x0, x1).
El perceptrón solo puede decir "de un lado de la recta" o "del otro lado".

## 2. La tabla de XOR

| x0 | x1 | XOR |
|----|----|-----|
| 0  | 0  | 0   |
| 0  | 1  | 1   |
| 1  | 0  | 1   |
| 1  | 1  | 0   |

Si lo dibujamos en el plano:

```
 x1
  1 |  (0,1)=1        (1,1)=0
    |
    |
  0 |  (0,0)=0        (1,0)=1
    +-------------------------- x0
       0               1
```

Los **1** están en una diagonal y los **0** en la otra.

## 3. Intuición geométrica

Probá trazar **una sola recta** que deje los dos 1 de un lado y los dos 0 del otro: no existe.
Cualquier recta que separe (0,1) de (0,0) termina dejando (1,0) o (1,1) del lado equivocado.

Comparalo con AND, OR o NAND (los otros notebooks de esta carpeta):

```
        AND                         XOR
 1 | 0      1               1 | 1      0
   |      /                   |    ¿¿??
 0 | 0   / 0                0 | 0      1
   +----/------               +-----------
```

En AND basta una recta que deje (1,1) solo de un lado. Por eso AND, OR y NAND se pueden resolver
con un perceptrón y XOR no. A los problemas que **sí** se pueden separar con una recta se los llama
**linealmente separables**. XOR **no** es linealmente separable.

## 4. Demostración algebraica (con función escalón)

Supongamos que el perceptrón da 1 cuando `z > 0` y 0 cuando `z < 0`. Escribamos `z` para cada entrada:

```
z00 = b
z01 = w1 + b
z10 = w0 + b
z11 = w0 + w1 + b
```

Fijate que siempre se cumple esta identidad:

```
z01 + z10 = z00 + z11        (las dos dan w0 + w1 + 2b)
```

Para resolver XOR necesitaríamos:

- `z01 > 0` y `z10 > 0`  →  `z01 + z10 > 0`
- `z00 < 0` y `z11 < 0`  →  `z00 + z11 < 0`

Pero las dos sumas son **el mismo número**: no puede ser positivo y negativo a la vez.
**Contradicción** ⇒ no existen `w0, w1, b` que resuelvan XOR.

## 5. ¿Y si uso sigmoide o swish en vez de escalón?

El argumento de arriba sigue valiendo, porque **cambiar la activación no cambia que `z` sea lineal**.

- **Sigmoide, ReLU, tanh** (funciones que solo crecen): un valor más grande de `z` da una salida
  más grande o igual. Para que (0,1) y (1,0) den más que (0,0) y (1,1) harían falta las mismas
  desigualdades del punto 4, que ya vimos que son imposibles.
- **Swish** (`f(z) = z·sigmoid(z)`): no es estrictamente creciente, porque tiene un pequeño "pozo"
  en z ≈ -1.28. Igual no alcanza:
  - para que la salida sea ≈ 1 hace falta `z ≈ 1.28`, así que `z01 + z10 ≈ 2.56`;
  - para que la salida sea ≈ 0 hace falta `z ≈ 0` o `z` muy negativo, así que `z00 + z11 ≤ ~0`.
  - Por la identidad `z01 + z10 = z00 + z11`, eso es imposible.

> Aclaración: existen activaciones "raras" (por ejemplo una campana gaussiana, que es alta en el
> medio y baja en los extremos) con las que una sola neurona sí podría imitar XOR. Las activaciones
> que usamos normalmente (escalón, sigmoide, ReLU, swish) no pueden.

## 6. Lo que pasó en `Perceptron_XOR.ipynb`

Al entrenar, la pérdida se quedó clavada en **0.25** y el modelo devolvía **0.5** para todas las entradas.

Como no hay forma de acertar, el descenso por gradiente encuentra lo "menos malo":

- lleva `w0 = w1 = 0`, o sea que ignora las entradas;
- ajusta el sesgo para que la salida sea 0.5 siempre (con swish, `b ≈ 0.739`);
- el error en cada punto es `(0.5)² = 0.25`, y el promedio es **0.25**.

Con `w0 = w1 = 0` el gradiente también da 0, así que el entrenamiento se queda quieto ahí para siempre.
Esto no es un bug del código: es el **límite del modelo**.

## 7. La solución: agregar una capa oculta

XOR se puede escribir combinando compuertas que **sí** son linealmente separables:

```
XOR(x0, x1) = AND( OR(x0, x1), NAND(x0, x1) )
```

Es decir:

```
 x0 ──┬──► [ OR   ] ──┐
      │               ├──► [ AND ] ──► XOR
 x1 ──┼──► [ NAND ] ──┘
```

Cada neurona oculta traza **su propia recta**, y la neurona de salida combina las dos.
Con dos rectas sí se puede encerrar la diagonal de los 1:

- **Recta A** (neurona OR): `x0 + x1 = 0.5`. Deja (0,0) de un lado y los otros tres del otro.
- **Recta B** (neurona NAND): `x0 + x1 = 1.5`. Deja (1,1) de un lado y los otros tres del otro.
- La **franja entre las dos rectas** contiene exactamente (0,1) y (1,0), que son los puntos donde XOR vale 1.
  La neurona de salida (AND) responde "1 solo si estoy por arriba de A **y** por debajo de B".

Esto es una **red multicapa (MLP)**: entrada → capa oculta (2 neuronas) → salida.
Probando con michigrad y swish, una red 2-2-1 llega a `L ≈ 0` y predice `[0, 1, 1, 0]`.

## Resumen

1. Un perceptrón = activación aplicada a **una recta**.
2. XOR necesita separar dos diagonales, y **una recta no alcanza** (no es linealmente separable).
3. Cambiar la activación (sigmoide, swish, ...) no lo arregla, porque la parte lineal sigue siendo una sola recta.
4. Hace falta **al menos una capa oculta**: varias rectas combinadas pueden formar regiones más complejas.

> Dato histórico: Minsky y Papert señalaron esta limitación en el libro *Perceptrons* (1969), y eso
> enfrió la investigación en redes neuronales por años. Se superó con redes multicapa entrenadas con
> **backpropagation**, que es justamente lo que hace el autograd de michigrad.
