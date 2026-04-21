# ARTIFICIAL HYDROCARBON NETWORKS — ESTUDIO COMPARATIVO DE ROBUSTEZ
## Especificación Técnica Completa v1.0

**Scripts implementados:** `CRv2.4.py`, `Exov2.4.py`  
**Fecha:** Marzo 2026

---

## ÍNDICE

1. [Resumen del Estudio](#1-resumen-del-estudio)
2. [Fundamentos Matemáticos de AHN](#2-fundamentos-matemáticos-de-ahn)
3. [Algoritmo de Entrenamiento](#3-algoritmo-de-entrenamiento)
4. [Correcciones de Implementación v2.1](#4-correcciones-de-implementación-v21)
5. [Mejoras de Robustez del Optimizador](#5-mejoras-de-robustez-del-optimizador)
6. [Calibración de Probabilidades — Platt Scaling](#6-calibración-de-probabilidades--platt-scaling)
7. [Datasets y Preprocesamiento](#7-datasets-y-preprocesamiento)
8. [Modelos Baseline](#8-modelos-baseline)
9. [Diseño Experimental — Experimento Principal](#9-diseño-experimental--experimento-principal)
10. [Diseño Experimental — Robustez](#10-diseño-experimental--robustez)
11. [Métricas de Evaluación](#11-métricas-de-evaluación)
12. [Hiperparámetros de Configuración](#12-hiperparámetros-de-configuración)
13. [Tabla de Archivos Generados](#13-tabla-de-archivos-generados)

---

## 1. RESUMEN DEL ESTUDIO

Este estudio implementa y evalúa Artificial Hydrocarbon Networks (AHN) para clasificación binaria, comparándolo contra SVM, Random Forest y MLP en dos dominios:

| Dominio | Dataset | n | Features | Balance |
|---|---|---|---|---|
| Riesgo crediticio | German Credit Risk (HuggingFace) | 1000 | 48 (one-hot) | 70% good / 30% bad |
| Astrofísica | NASA KOI Exoplanet Archive | ~9500 | 10 físicas | ~55% FP / 45% confirmed |

Para cada dominio se ejecutan **4 experimentos**:

- **Experimento principal**: Comparación completa AHN vs. baselines con K-Fold CV
- **EXP 1 — Data Scarcity**: Degradación con train reducido al 5–100%
- **EXP 2 — Feature Noise**: Degradación con ruido Gaussiano en inferencia σ ∈ {0.0…1.0}
- **EXP 3 — Label Noise**: Degradación con etiquetas corruptas p ∈ {0%…20%}

---

## 2. FUNDAMENTOS MATEMÁTICOS DE AHN

### 2.1. Jerarquía de Componentes

```
AHNMolecule  →  AHNCompound  →  AHNMixture
   φ_k(x)          ψ(x)            S(x)
```

### 2.2. Molécula CH_k — Ecuación (1)

La molécula de orden k evalúa una función no lineal sobre x ∈ ℝⁿ:

$$\varphi_k(x) = \sum_{r=1}^{n} \sigma_r \prod_{i=1}^{k} (x_r - H_{i,r}) + b$$

donde:
- `n` — número de features
- `σ_r ∈ ℝ` — valor del carbono para el feature r (entrenado)
- `H_{i,r} ∈ ℝ` — posición del i-ésimo hidrógeno en la dimensión r (entrenado)
- `b ∈ ℝ` — bias opcional, activo solo cuando `k ≥ 2` y `use_bias=True`
- `k` — orden de la molécula: CH₃ = cúbico (k=3), CH₂ = cuadrático (k=2)

**Interpretación química:**
- k=3 (CH₃) — molécula terminal, capacidad expresiva cúbica por feature
- k=2 (CH₂) — molécula interior, cuadrática por feature
- k=1 (CH) — lineal (solo para n_molecules=1 en regresión simple)

**Nota sobre bias:** Para k=1 el bias es redundante porque `Σ σ_r · H_{1,r}` ya actúa como offset global. Para k≥2 el término `φ_k(x)=0` cuando `x_r = H_{i,r}` para algún r, creando un punto muerto real; el bias lo elimina.

### 2.3. Parámetros por Molécula

| Símbolo | Dimensión | Descripción |
|---|---|---|
| `σ` | (n,) | Valores de carbono por feature |
| `H` | (k, n) | Posiciones de hidrógenos |
| `b` | escalar | Bias (solo k≥2 y use_bias=True) |

Total de parámetros por molécula: `n·(k+1) + [1 si use_bias]`

### 2.4. Estructura del Compuesto — Cadena Lineal Saturada

Un compuesto de m moléculas sigue la estructura química CH₃–(CH₂)_(m-2)–CH₃:

| m | Estructura | k_orders |
|---|---|---|
| 1 | CH₃ | [3] |
| 2 | CH₃–CH₃ | [3, 3] |
| 3 | CH₃–CH₂–CH₃ | [3, 2, 3] |
| 4 | CH₃–CH₂–CH₂–CH₃ | [3, 2, 2, 3] |

### 2.5. Límites y Regiones de Partición — Ecuación (2)

Los límites del compuesto se construyen recursivamente sobre `L_min, L_max ∈ ℝⁿ` (min/max por feature en X_train):

$$L_0 = L_{\min}, \quad L_j = L_{j-1} + r_j \quad \text{para } j=1,\ldots,m-1, \quad L_m = L_{\max}$$

Los centros implícitos de cada región:

$$\mu_j = \frac{L_{j-1} + L_j}{2} \in \mathbb{R}^n, \quad j = 0, 1, \ldots, m-1$$

Las distancias `r_j ∈ ℝⁿ` son los parámetros de partición entrenables.

### 2.6. Particionamiento — Ecuación (A1)

Cada muestra x se asigna a la molécula j* cuyo centro proyectado 1D esté más cercano:

$$j^*(x) = \arg\min_{j \in \{0,\ldots,m-1\}} \left| (x \cdot v_1) - (\mu_j \cdot v_1) \right|$$

donde `v_1 ∈ ℝⁿ` es el primer componente principal del conjunto de entrenamiento (eje de cadena). Esta corrección v2.1 reemplaza la distancia L2 n-dimensional que colapsa en datasets de alta dimensionalidad (ver §4).

La partición j* define la región Σ_j:

$$\Sigma_j = \{x \in X_{\text{train}} : j^*(x) = j\}$$

### 2.7. Salida del Compuesto — Ecuación (A2)

El compuesto evalúa la molécula de la región asignada:

$$\psi(x) = \varphi_{j^*(x)}(x)$$

### 2.8. Mezcla de Compuestos — Ecuación (4)

Con c compuestos {ψ₁,...,ψ_c}, la salida de la mezcla es:

$$S(x) = \sum_{i=1}^{c} \alpha_i \, \psi_i(x)$$

Los coeficientes α_i se calculan por mínimos cuadrados (LSE) sobre X_train:

$$\boldsymbol{\alpha} = \arg\min_{\alpha} \left\| \mathbf{y} - \Psi \boldsymbol{\alpha} \right\|^2$$

donde Ψ ∈ ℝ^{N×c}, Ψ_{ni} = ψ_i(x_n). Resuelto con `scipy.linalg.lstsq`.

**Para c=1:** α = [1.0], S(x) = ψ₁(x).

### 2.9. Clasificación Binaria — Ecuaciones (A5) y (A6)

**Predicción — Ec.(A5):**

$$\hat{y}(x) = \mathbb{1}[S(x) \geq \tau]$$

donde τ = `threshold` (por defecto 0.5). Equivalente a `round(S(x))` cuando τ=0.5.

**Error de clasificación en entrenamiento — Ec.(A6):**

$$E_j = \frac{1}{2|\Sigma_j|} \sum_{(x,y) \in \Sigma_j} \left(y - \text{round}\!\left(\text{clip}(\varphi_j(x),\, -2,\, 3)\right)\right)^2$$

El `clip(-2, 3)` estabiliza el gradiente numérico sin afectar la frontera de decisión.

### 2.10. Error Global y Criterio de Parada

$$E_{\text{global}} = \sum_{j=0}^{m-1} E_j$$

El entrenamiento se detiene cuando `E_global ≤ ε` (tolerancia), o al alcanzar `max_iterations`.

---

## 3. ALGORITMO DE ENTRENAMIENTO

### Algoritmo 1 (paper AHN, con mejoras v2.1)

```
Entrada: X_train, y_train, η (learning rate), ε (tolerancia), max_iter, patience
Salida:  molecules* (mejor estado), r* (mejor partición)

1. Inicializar moléculas {φ_j} con σ ~ N(0, 0.01), H ~ N(0, 0.1)
2. Inicializar r_j con _init_bounds(X_train):
   a. Calcular v₁ = PC1 de X_train centrado   [corrección C1, §4]
   b. Proyectar X_proj = X_train @ v₁
   c. q_j = quantile(X_proj, j/m) para j=0,...,m
   d. Añadir ruido U(-0.1, 0.1)·(p_max-p_min)/m, anclar extremos
   e. Reconstruir L_j ∈ ℝⁿ desde proyecciones 1D  [corrección C3, §4]
   f. r_j = L_j - L_{j-1}

3. best_E ← ∞,  best_snap ← snapshot()
4. Para t = 1, ..., max_iter:
   a. Asignar j*(x) por dist. 1D proyectada  [corrección C2, §4]
   b. Para j=0,...,m-1:
      - Ajustar φ_j sobre Σ_j con L-BFGS-B minimizando Ec.(A3)
      - Calcular E_j con round() — Ec.(A6)
   c. E_global ← Σ E_j
   d. Si E_global < best_E: guardar best_snap, no_improve ← 0
      Si no: no_improve ← no_improve + 1
   e. Si E_global ≤ ε: break
   f. Actualizar r_j — Ec.(3):
      Δr_j = -η·(E_{j-1} - E_j), con E_{-1} = 0
      Aplicar _clip_r()
   g. Si no_improve ≥ patience:
      Reinicializar _init_bounds(X_train)  [mejora §5.1]
      no_improve ← 0

5. Restaurar best_snap → self.molecules, self.r
6. Almacenar self.best_E_ = best_E
```

### 3.1. Optimización de Molécula — L-BFGS-B

Cada molécula se optimiza con `scipy.optimize.minimize`, método `L-BFGS-B`, con gradientes analíticos:

$$\frac{\partial \mathcal{L}}{\partial \sigma_r} = -\frac{1}{N_j} \sum_{n} \text{residual}_n \cdot \prod_{i=1}^k (x_{n,r} - H_{i,r})$$

$$\frac{\partial \mathcal{L}}{\partial H_{i,r}} = \frac{1}{N_j} \sum_{n} \text{residual}_n \cdot \sigma_r \cdot \frac{\prod_{i'} (x_{n,r} - H_{i',r})}{x_{n,r} - H_{i,r}}$$

$$\frac{\partial \mathcal{L}}{\partial b} = -\frac{1}{N_j} \sum_{n} \text{residual}_n$$

donde `residual_n = y_n - φ_j(x_n)` y `N_j = |Σ_j|`.

### 3.2. Actualización de Límites — Ecuación (3)

$$r_j^{(t+1)} = r_j^{(t)} - \eta \left(E_{j-1}^{(t)} - E_j^{(t)}\right), \quad E_{-1} = 0$$

Las regiones con error alto relativo al anterior reciben un límite mayor (más espacio); las de error bajo lo reducen. La función `_clip_r()` garantiza r_j ≥ 2% del rango por feature y que la suma no exceda el 98% del rango total.

### 3.3. Multi-Restart

El entrenamiento completo se repite `n_restarts` veces con seeds distintas. Se conserva el compuesto con menor `best_E_`.

---

## 4. CORRECCIONES DE IMPLEMENTACIÓN v2.1

Tres problemas detectados en la implementación canónica del paper con datasets de alta dimensionalidad (n_features >> m):

### C1 — Inicialización Cuantílica con Ruido

**Problema:** Inicialización aleatoria pura de r_j → colapso de partición (todas las muestras en Σ₀).  
**Corrección:** Inicialización anclada en cuantiles del dataset proyectado sobre PC1, más perturbación `U(-δ, δ)` con δ = 0.10:

$$p_j = \text{quantile}(X \cdot v_1,\, j/m) + \mathcal{U}(-\delta, \delta) \cdot \frac{p_{\max} - p_{\min}}{m}$$

### C2 — Particionamiento 1D sobre PC1

**Problema:** Distancia L2 en ℝⁿ sufre concentración de medida para n >> m → todas las muestras equidistan de todos los centros → asignación arbitraria.  
**Corrección:** Proyectar X sobre v₁ (PC1) y calcular distancia escalar:

$$j^*(x) = \arg\min_j \left| (x \cdot v_1) - c_j \right|, \quad c_j = \frac{p_j + p_{j+1}}{2}$$

Los features se siguen evaluando en ℝⁿ dentro de φ_k(x); solo el mecanismo de asignación usa 1D.

### C3 — Anclaje de Proyecciones en _compute_bounds

**Problema:** `L_min · v₁ << min(X · v₁)` → centros proyectados fuera del soporte del dataset → partición 0 siempre vacía.  
**Corrección:** Anclar explícitamente los extremos del array de proyecciones:

$$p^{(1D)}_0 = \min(X_{\text{train}} \cdot v_1), \quad p^{(1D)}_m = \max(X_{\text{train}} \cdot v_1)$$

Los límites intermedios L[1]…L[m-1] se proyectan normalmente.

**Resultado tras correcciones** (German Credit Risk, n=600, n_features=48):

```
m=2 → particiones=[341, 259]   vacías=0  ✓
m=3 → particiones=[138, 264, 198]  vacías=0  ✓
```

---

## 5. MEJORAS DE ROBUSTEZ DEL OPTIMIZADOR

Implementadas sin modificar ninguna ecuación del paper:

### 5.1. Reinicialización en Estancamiento

Si `E_global` no mejora en `patience` iteraciones consecutivas, se llama de nuevo a `_init_bounds(X_train)` generando una partición cuantílica completamente nueva con el estado actual del RNG (entropía acumulada garantiza diversidad).

Esto reemplaza la perturbación de ruido pequeña (`perturb_scale`) que resultó inefectiva: el gradiente `Δr = -η(E_{j-1}-E_j)` neutraliza el ruido pequeño en 2–3 iteraciones, volviendo al mismo atractor.

### 5.2. Best-State Tracking

En cada iteración que mejora `E_global`, se guarda un snapshot profundo:

```python
best_snap = (deepcopy(self.molecules), self.r.copy())
```

Al finalizar el entrenamiento se restaura siempre el mejor estado, independientemente de cuántas iteraciones de exploración posteriores haya habido.

El mínimo real se expone como `comp.best_E_` para que el selector de multi-restart use el valor correcto.

### 5.3. Función `_clip_r()`

Garantiza integridad de los límites después de cada actualización:

```
Para cada feature f:
  r_j[f] = max(r_j[f],  0.02 · (L_max[f] - L_min[f]))  # mínimo 2% del rango
  Si Σ_j r_j[f] > 0.98 · (L_max[f] - L_min[f]):
    escalar r_j[f] proporcionalmente                    # no exceder 98%
```

---

## 6. CALIBRACIÓN DE PROBABILIDADES — PLATT SCALING

### 6.1. Motivación

`S(x)` es un score arbitrario, no una probabilidad calibrada. Para calcular ROC-AUC se requiere que `predict_proba()` devuelva estimaciones de probabilidad monótonas con la clase positiva.

### 6.2. Formulación

Se ajusta una regresión logística post-hoc sobre un validation set separado:

$$P(y=1 \mid x) = \sigma(a \cdot S(x) + b) = \frac{1}{1 + e^{-(a \cdot S(x) + b)}}$$

Los parámetros `a, b` se estiman minimizando la log-verosimilitud sobre `(X_val, y_val)` con `sklearn.linear_model.LogisticRegression(C=1e6)`.

### 6.3. Separación estricta de datos

| Uso | Conjunto |
|---|---|
| Entrenamiento AHN | X_train |
| Platt Scaling | X_val (nunca X_train) |
| Evaluación final | X_test (nunca visto en entrenamiento ni Platt) |

En K-Fold CV: Platt se ajusta sobre el 20% final del train fold, no sobre el held-out.

### 6.4. Separación predict() vs predict_proba()

```
predict(x)       → S(x) >= threshold   (NUNCA pasa por Platt)
predict_proba(x) → σ(a·S(x) + b)      (usa Platt si fue calibrado)
```

Esta separación garantiza que `predict()` sea determinista y no dependa de los datos de validación.

---

## 7. DATASETS Y PREPROCESAMIENTO

### 7.1. German Credit Risk

| Aspecto | Detalle |
|---|---|
| Fuente | HuggingFace: `inGeniia/german-credit-risk_credit-scoring_mlp` |
| Fallback 1 | `german_credit_risk.csv` local |
| Fallback 2 | `make_classification(n=1000, n_feat=48, weights=[0.7,0.3])` |
| Target | `Risk`: good→0, bad→1 |
| Features | Todas las columnas; `pd.get_dummies(drop_first=True)` |
| n_features | 48 (tras one-hot encoding) |
| Balance | ~70% good, ~30% bad |

### 7.2. NASA Exoplanet Archive KOI

| Aspecto | Detalle |
|---|---|
| Fuente | `https://exoplanetarchive.ipac.caltech.edu/cgi-bin/nstedAPI/nph-nstedAPI?table=cumulative&format=csv` |
| Fallback 1 | `cumulative_koi.csv` local |
| Fallback 2 | `make_classification(n=2000, n_feat=10, weights=[0.45,0.55])` |
| Target | `koi_disposition`: CONFIRMED→1, FALSE POSITIVE→0 |
| Features | 10 físicas del tránsito (ver tabla abajo) |
| Balance | ~55% FP, ~45% Confirmed |

**Features KOI seleccionadas:**

| Feature | Descripción | Unidad |
|---|---|---|
| `koi_period` | Período orbital | días |
| `koi_time0bk` | Tiempo de referencia del tránsito | días BJD |
| `koi_impact` | Parámetro de impacto | adimensional |
| `koi_duration` | Duración del tránsito | horas |
| `koi_depth` | Profundidad del tránsito | ppm |
| `koi_prad` | Radio del planeta | R_Tierra |
| `koi_teq` | Temperatura de equilibrio | K |
| `koi_insol` | Insolación | F_Tierra |
| `koi_model_snr` | Señal-ruido del modelo | adimensional |
| `koi_steff` | Temperatura efectiva estelar | K |

### 7.3. Split y Escalado

**División estratificada 60/20/20** (misma en ambos dominios):

```python
X_temp, X_test  = train_test_split(X, test_size=0.20, stratify=y, random_state=42)
X_train, X_val  = train_test_split(X_temp, test_size=0.25, stratify=y_temp, random_state=42)
# → train=60%, val=20%, test=20%
```

**Escalado MinMax a [-1, 1]:**

```python
scaler = MinMaxScaler(feature_range=(-1, 1))
X_train = scaler.fit_transform(X_train)   # fit SOLO sobre train
X_val   = scaler.transform(X_val)
X_test  = scaler.transform(X_test)
```

El rango `(-1, 1)` garantiza que `H_{i,r}` inicializado en `N(0, 0.1)` esté dentro del soporte de los datos.

---

## 8. MODELOS BASELINE

Todos los baselines usan los mismos conjuntos de train/val/test que AHN.

### 8.1. SVM

```python
# Credit Risk
SVC(kernel='rbf', probability=True, random_state=42)

# Exoplanetas
SVC(kernel='rbf', probability=True, class_weight='balanced', random_state=42)
```

`class_weight='balanced'` en Exoplanetas compensa el desbalance de clases en el subset KOI sin `CANDIDATE`.

### 8.2. Random Forest

```python
RandomForestClassifier(
    n_estimators=100, max_depth=10, min_samples_split=5, random_state=42
)
```

### 8.3. MLP

```python
MLPClassifier(
    hidden_layer_sizes=(64, 32), activation='relu', solver='adam',
    alpha=0.001, learning_rate_init=0.001, max_iter=300,
    early_stopping=True, validation_fraction=0.15, random_state=42
)
```

---

## 9. DISEÑO EXPERIMENTAL — EXPERIMENTO PRINCIPAL

### 9.1. Entrenamiento y Evaluación

1. `AHNMixture.fit(X_train, y_train)` — entrena n_restarts compuestos
2. `AHNMixture.fit_platt(X_val, y_val)` — calibra probabilidades
3. Evaluar sobre X_test: `predict()` para Acc/F1, `predict_proba()` para AUC
4. Evaluar baselines con `fit(X_train)`, `predict/predict_proba(X_test)`

### 9.2. K-Fold Cross Validation

**K = 5, StratifiedKFold**, aplicado sobre `X_cv = X_train ∪ X_val`:

```
Para cada fold (tr_idx, val_idx):
  1. Reservar 20% del train fold como X_platt (anti-leakage)
  2. AHN.fit(X_tr_pure, y_tr_pure)
  3. AHN.fit_platt(X_platt, y_platt)
  4. Evaluar: AUC, ACC, F1 sobre X_ho (held-out)
```

El modelo final (entrenado sobre X_train completo) no se modifica. K-Fold solo estima la variabilidad.

### 9.3. Métricas Reportadas

| Métrica | Función | Sobre |
|---|---|---|
| Train Accuracy | `accuracy_score(y_train, predict(X_train))` | X_train |
| Val Accuracy | `accuracy_score(y_val, predict(X_val))` | X_val |
| Test Accuracy | `accuracy_score(y_test, predict(X_test))` | X_test |
| Precision | `precision_score(y_test, predict(X_test))` | X_test |
| Recall | `recall_score(y_test, predict(X_test))` | X_test |
| F1-Score | `f1_score(y_test, predict(X_test))` | X_test |
| ROC-AUC | `roc_auc_score(y_test, predict_proba(X_test)[:,1])` | X_test |

### 9.4. Score Global Ponderado

Para determinar el mejor modelo general:

$$\text{Overall} = 0.3 \cdot \text{Acc} + 0.3 \cdot F_1 + 0.4 \cdot \text{AUC}$$

---

## 10. DISEÑO EXPERIMENTAL — ROBUSTEZ

Los tres experimentos de robustez usan una **configuración AHN reducida** para los sweeps (el mínimo se alcanza en iteraciones 1–5 según análisis de convergencia; best-state tracking lo preserva):

```python
_AHN_SWEEP = {
    **AHN_CONFIG,
    'n_restarts':     1,   # 1 solo restart
    'max_iterations': 40,  # suficiente para capturar best_E
    'patience':       10,  # reinit más rápido
}
```

**Punto de referencia:** Los resultados en condición limpia (frac=100%, σ=0, flip=0%) son comparables con el experimento principal (pequeñas diferencias por n_restarts=1 vs. 3).

### 10.1. EXP 1 — Data Scarcity

**Pregunta:** ¿Cuántos datos de entrenamiento necesita cada modelo para ser competitivo?

**Diseño:**

```
Fracciones: f ∈ {5%, 10%, 20%, 30%, 50%, 75%, 100%}

Para cada f:
  1. Submuestreo estratificado de X_train:
     idx = stratified_sample(X_train, n = max(int(N_train · f), 10))
  2. Entrenar AHN y baselines sobre X_train[idx]
  3. Platt AHN sobre X_val (fijo, nunca submuestreado)
  4. Evaluar sobre X_test (fijo, nunca modificado)
  5. Registrar: Acc, Precision, Recall, F1, AUC
```

**Hipótesis:** AHN degrada más suavemente que MLP en pocos datos (estructura química inductiva vs. pesos libres). SVM debería ser el más robusto por el margen máximo.

**Referencia:** punto f=100% (condición limpia).

### 10.2. EXP 2 — Feature Noise

**Pregunta:** ¿Resisten los modelos a ruido en las features durante inferencia?  
Simula sensores ruidosos o errores de medición en producción.

**Diseño:**

```
σ ∈ {0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0}   (relativo a escala [-1,1])

1. Entrenar AHN y baselines UNA VEZ sobre X_train limpio
2. Para cada σ:
   X_test_noisy = X_test + ε,  ε ~ N(0, σ²·I)
   (ruido iid por feature, nueva muestra por σ, seed fijo=0)
3. Inferencia con modelos ya entrenados (sin re-entrenamiento)
4. Evaluar sobre (X_test_noisy, y_test)
```

**Nota metodológica:** El ruido se añade **solo en inferencia**, no en entrenamiento. Esto mide robustez del modelo ante distribución shift en X_test. La variante opuesta (ruido en train) mediría data augmentation, que es un experimento distinto.

**Hipótesis:** AHN podría ser más robusto por el particionamiento hard sobre PC1 — una perturbación isotrópica en ℝⁿ tiene proyección menor sobre un eje específico que sobre todos los features independientemente.

**Referencia:** punto σ=0 (condición limpia).

### 10.3. EXP 3 — Label Noise

**Pregunta:** ¿Resisten los modelos a etiquetas de entrenamiento corruptas?  
Simula errores de anotación humana o etiquetado automático imperfecto.

**Diseño:**

```
p ∈ {0%, 5%, 10%, 15%, 20%}

Para cada p:
  1. Generar y_noisy:
     flip_mask ~ Bernoulli(p) sobre y_train
     y_noisy[flip_mask] = 1 - y_train[flip_mask]   # flip simétrico
  2. Entrenar AHN y baselines sobre (X_train, y_noisy)
  3. Platt AHN sobre X_val con y_val LIMPIO
  4. Evaluar sobre (X_test, y_test LIMPIO)
```

**Flip simétrico:** tanto positivos (1→0) como negativos (0→1) se corrompen con igual probabilidad p. Para p=20% y balance 70/30 en GCR, ~126 de 600 etiquetas son corruptas.

**Hipótesis:** El `round()` en Ec.(A6) crea una zona muerta alrededor de 0.5: errores de entrenamiento de muestras flip se amplifican menos porque la función de pérdida ya aplica umbral. RF con bagging debería ser el más robusto por promediado de árboles.

**Referencia:** punto p=0% (condición limpia).

---

## 11. MÉTRICAS DE EVALUACIÓN

### 11.1. Métricas de Clasificación

| Métrica | Fórmula | Rango | Interpretación |
|---|---|---|---|
| Accuracy | (TP+TN)/(TP+TN+FP+FN) | [0,1] | Fracción de predicciones correctas |
| Precision | TP/(TP+FP) | [0,1] | De los predichos positivos, ¿cuántos son reales? |
| Recall | TP/(TP+FN) | [0,1] | De los positivos reales, ¿cuántos se detectaron? |
| F1-Score | 2·P·R/(P+R) | [0,1] | Media armónica Precision-Recall |
| ROC-AUC | área bajo curva ROC | [0,1] | Capacidad discriminativa ranking |

`zero_division=0` en todos los casos para manejar particiones vacías.


### 11.2. Métricas de Convergencia AHN

| Variable | Descripción |
|---|---|
| `E_j` | Error de la molécula j — Ec.(A6) |
| `E_global` | Suma de errores: Σ E_j |
| `best_E_` | Mínimo histórico de E_global (restaurado al final) |
| `history` | Lista de E_global por iteración |
| `particiones` | Número de muestras asignadas a cada molécula |

---

## 12. HIPERPARÁMETROS DE CONFIGURACIÓN

### 12.1. Configuración Principal (experimento comparativo)

```python
AHN_CONFIG = dict(
    n_compounds    = 1,      # un solo compuesto
    n_molecules    = 2,      # CH3-CH3 (del paper [8])
    learning_rate  = 0.1,    # η — tasa de actualización de r_j
    tolerance      = 0.01,   # ε — umbral de parada E_global ≤ ε
    max_iterations = 100,    # iteraciones máximas del Algoritmo 1
    random_state   = 42,     # semilla base
    use_bias       = True,   # bias entrenable para k≥2
    use_bce        = False,  # alphas por LSE (True = BCE, solo c>1)
    threshold      = 0.5,    # τ — umbral de clasificación Ec.(A5)
    patience       = 20,     # iters sin mejora antes de reinit bounds
    n_restarts     = 3,      # repeticiones con seeds distintas
)
```

### 12.2. Configuración Reducida (sweeps de robustez)

```python
_AHN_SWEEP = {
    **AHN_CONFIG,
    'n_restarts':     1,   # 1 restart para velocidad en sweeps
    'max_iterations': 40,  # best_E siempre se captura en iters 1-5
    'patience':       10,
}
```

### 12.3. Tabla de Hiperparámetros — Significado y Origen

| Parámetro | Default | Origen | Descripción |
|---|---|---|---|
| `n_molecules` (m) | 2 | Paper [8] | Número de moléculas en el compuesto |
| `learning_rate` (η) | 0.1 | Paper [8] | Tasa de actualización Ec.(3) |
| `tolerance` (ε) | 0.01 | Paper [8] | Umbral de convergencia E_global |
| `threshold` (τ) | 0.5 | Spec §10.7 | Umbral de clasificación Ec.(A5) |
| `max_iterations` | 100 | Implementación | Límite de iteraciones del Alg. 1 |
| `use_bias` | True | Implementación | Bias en φ_k para k≥2 |
| `patience` | 20 | Mejora v2.1 | Iters sin mejora antes de reinit |
| `n_restarts` | 3 | Mejora v2.1 | Repeticiones multi-restart |
| `init_noise` | 0.10 | Spec §10.5 | Factor de perturbación cuantílica |
| `clip_range` | (-2, 3) | Spec §10.7 | Clipping antes de round() en E_j |

---

## 13. TABLA DE ARCHIVOS GENERADOS

### 13.1. Credit Risk (`ahn_outputs/`)

| # | Archivo | Contenido |
|---|---|---|
| 1 | `final_comparison_table.csv` | Métricas completas todos los modelos |
| 2 | `final_radar_comparison.png` | Gráfica radar AHN vs baselines |
| 3 | `final_roc_curves_complete.png` | Curvas ROC superpuestas |
| 4 | `final_metrics_comparison.png` | Barras Acc/F1/AUC/Prec/Rec |
| 5 | `ahn_confusion_matrices.png` | Matrices de confusión (4 modelos) |
| 6 | `ahn_internal_structure.png` | Particionamiento + convergencia AHN |
| 7 | `ahn_comparison_data.pkl` | Objeto pickle con todos los resultados |
| 8 | `robustness_scarcity.png/.csv` | EXP 1 — Data Scarcity |
| 9 | `robustness_feature_noise.png/.csv` | EXP 2 — Feature Noise |
| 10 | `robustness_label_noise.png/.csv` | EXP 3 — Label Noise |

### 13.2. Exoplanetas (`exo_ahn_outputs/`)

Misma estructura que Credit Risk, más:

| Feature | Diferencia |
|---|---|
| Fuente | NASA KOI |
| Labels en confusion | FP / Confirmed |
| SVM | `class_weight='balanced'` |
| Directorio | `exo_ahn_outputs/` |

### 13.3. Scripts Python

| Script | Descripción |
|---|---|
| `CRv2.4.py` | Experimento completo German Credit Risk (bloques 1–12) |
| `Exov2.4.py` | Experimento completo NASA KOI (bloques 1–12) |

---
