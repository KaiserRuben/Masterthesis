# Formulas for PowerPoint Equation Editor

Copy each formula below into PowerPoint's equation tool (Insert → Equation, or Alt+=).
Use the **LaTeX input mode** (click the small down-arrow on the Equation tab → "LaTeX").

---

## Slide 5 — Context determines the decision space

```
M = \sigma(C) \subseteq U
```

```
S_j \in \{ \mathbb{R}, \mathbb{Z}, \{c_1, \ldots, c_k\} \}
```

```
U = \prod_j S_j
```

---

## Slide 6 — Predictability / Thickness

```
\text{Thickness} = \frac{|g|}{\|\Delta g\|}
```

---

## Slide 7 — Controllability / Direction

```
\text{Direction} = \arg\max_i |\Delta_i g|
```

---

## Slide 8 — Evaluability / Margin

```
\text{Margin} = |g(m)|
```

---

## Slide 9 — Three facets summary (same formulas as 6–8, in the cards)

---

## Slide 11 — The contrast function g

Main definition:
```
g_{jk}(m) = P(y_j \mid m) - P(y_k \mid m)
```

Sign interpretation:
```
g > 0 \text{: system prefers } y_j \qquad g < 0 \text{: system prefers } y_k \qquad g = 0 \text{: boundary}
```

Antisymmetry:
```
g_{jk} = -g_{kj}
```

Boundary as derived object:
```
B_{jk} = g^{-1}(0) = \{ m \in M \mid g_{jk}(m) = 0 \}
```

---

## Slide 12 — Four geometric quantities (table cells)

| Property    | Formula |
|-------------|---------|
| Margin      | `\|g_{jk}(m)\|` |
| Sensitivity | `\|\Delta g(m)\| = \max_i \|\Delta_i g\|` |
| Thickness   | `\frac{\|g\|}{\|\Delta g\|}` |
| Direction   | `\arg\max_i \|\Delta_i g(m)\|` |

Lattice-first definition:
```
\Delta_i g(x) = g(x + e_i) - g(x)
```

Genotype lattice:
```
G \subset \mathbb{Z}^d
```

---

## Slide 13 — Bridge Claim (right-side boxes)

```
\text{Predictability} \rightarrow \text{Thickness} = \frac{|g|}{\|\Delta g\|}
```

```
\text{Controllability} \rightarrow \text{Direction} = \arg\max_i |\Delta_i g|
```

```
\text{Evaluability} \rightarrow \text{Margin} = |g(m)|
```

---

## Slide 14 — Boundary Testing definition

```
BT(f) := \text{systematic empirical approximation of } g_{jk}
```

---

## Slide 17 — SMOO objectives (TargetedBalance card)

```
|P(A) - P(B)| \rightarrow 0
```

---

## Slide 18 — PDQ formula

```
PDQ(a, b) = \frac{d_o(P(a), P(b))}{d_i(a, b)}
```

---

## Slide 4 — f : M → Y (if you want it formal)

```
f : M \rightarrow Y
```
