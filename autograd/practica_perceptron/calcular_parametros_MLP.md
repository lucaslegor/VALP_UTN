# Cómo calcular cuántos pesos (w) y sesgos (b) tiene un MLP

## La regla

- **Cada neurona tiene un `w` por cada entrada que recibe.**
  Lo hace `Neuron(nin)`: `self.w = [Value(...) for _ in range(nin)]`.
- **Cada neurona tiene un solo `b`.**
- **Las entradas de una capa son las salidas de la capa anterior.**

Para cada capa:

```
w de la capa = entradas × neuronas
b de la capa = neuronas
```

Y el total del MLP es la suma de todas las capas.

## Cómo se arman las capas en el código

```python
class MLP:
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]
```

`sz` es la lista de tamaños. Cada par de números consecutivos es una capa
`Layer(entradas, neuronas)`.

## Ejemplo: `xor = MLP(2, [3, 3, 1])`

```python
sz = [2] + [3, 3, 1]   # [2, 3, 3, 1]
```

| Capa | `Layer(nin, nout)` | Neuronas | `w` por neurona | `w` de la capa | `b` de la capa |
|---|---|---|---|---|---|
| 1 (oculta) | `Layer(2, 3)` | 3 | 2 | 3 × 2 = **6** | 3 |
| 2 (oculta) | `Layer(3, 3)` | 3 | 3 | 3 × 3 = **9** | 3 |
| 3 (salida) | `Layer(3, 1)` | 1 | 3 | 1 × 3 = **3** | 1 |
| **Total** | | **7** | | **18** | **7** |

- Pesos: (2·3) + (3·3) + (3·1) = 6 + 9 + 3 = **18**
- Sesgos: 3 + 3 + 1 = **7**
- Parámetros entrenables: 18 + 7 = **25**

```
x0 ─┐       ┌─ n ─┐       ┌─ n ─┐
    ├─ 6w ─┼─ n ─┼─ 9w ──┼─ n ─┼─ 3w ── n ── ŷ
x1 ─┘       └─ n ─┘       └─ n ─┘
 2 entradas   capa 1        capa 2       salida
```

## Fórmula general

Si `sz = [n0, n1, n2, ..., nk]`:

```
total w = n0·n1 + n1·n2 + ... + n(k-1)·nk
total b = n1 + n2 + ... + nk
```

## Comprobarlo en el notebook

```python
total_w = sum(len(n.w) for l in xor.layers for n in l.neurons)   # 18
total_b = sum(1 for l in xor.layers for n in l.neurons)          # 7
print(total_w, total_b, total_w + total_b)                       # 18 7 25
```

## Comparación con el perceptrón simple

El perceptrón de una sola neurona tenía 2 pesos (`w0`, `w1`) y 1 sesgo
(3 parámetros). El MLP tiene 25. Esa capacidad extra es la que le permite
resolver XOR.
