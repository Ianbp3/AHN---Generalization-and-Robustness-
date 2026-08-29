import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from math import pi
from scipy.optimize import minimize
from scipy.linalg import lstsq
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, roc_curve, confusion_matrix,
    classification_report
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import warnings
import os
import pickle

warnings.filterwarnings('ignore')

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "exo_ahn_outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def out(filename):
    return os.path.join(OUTPUT_DIR, filename)


# ══════════════════════════════════════════════════════════════════════════════
#  BLOQUE 1 — IMPLEMENTACIÓN AHN
# ══════════════════════════════════════════════════════════════════════════════

class AHNMolecule:

    def __init__(self, k, n_features, rng, use_bias=False):
        self.k = k
        self.n_features = n_features
        self.use_bias = use_bias and (k >= 2)
        self.sigma = rng.standard_normal(n_features) * 0.01
        self.H     = rng.standard_normal((k, n_features)) * 0.1
        self.bias  = 0.0   # siempre existe; solo se entrena si use_bias=True

    def evaluate_batch(self, X):
        result = np.zeros(len(X))
        for r in range(self.n_features):
            prod = np.ones(len(X))
            for i in range(self.k):
                prod *= (X[:, r] - self.H[i, r])
            result += self.sigma[r] * prod
        return result + self.bias

    def get_params(self):
        base = np.concatenate([self.sigma, self.H.ravel()])
        return np.append(base, self.bias) if self.use_bias else base

    def set_params(self, params):
        n = self.n_features
        self.sigma = params[:n].copy()
        self.H     = params[n : n + self.k * n].reshape(self.k, n).copy()
        if self.use_bias:
            self.bias = float(params[-1])


class AHNCompound:

    def __init__(self, n_molecules=3, n_features=None, learning_rate=0.1,
                 tolerance=0.01, max_iterations=80, random_state=42,
                 use_bias=False, threshold=0.5, patience=20):
        self.m             = n_molecules
        self.n_feat        = n_features
        self.eta           = learning_rate
        self.epsilon       = tolerance
        self.max_iter      = max_iterations
        self.use_bias      = use_bias
        self.threshold     = threshold
        self.patience      = patience    # iters sin mejora antes de reinit bounds
        self.rng           = np.random.default_rng(random_state)

        if   n_molecules == 1: self.k_orders = [3]
        elif n_molecules == 2: self.k_orders = [3, 3]
        else:                  self.k_orders = [3] + [2] * (n_molecules - 2) + [3]

        self.molecules = []
        self.L = self.r = self.L_min = self.L_max = self.centers = None
        self.history = []
        self.pc1_explained_variance_ratio_ = None   # set in _init_bounds when m > 1

    # ── Inicialización de bounds ───────────────────────────────────────────────

    def _init_bounds(self, X):
        self.L_min = X.min(axis=0)
        self.L_max = X.max(axis=0)

        if self.m > 1:
            X_c = X - X.mean(axis=0)

            try:
                _, S, Vt = np.linalg.svd(X_c, full_matrices=False)
                self._chain_axis = Vt[0]
                total_var = float(np.sum(S ** 2))
                self.pc1_explained_variance_ratio_ = (
                    float(S[0] ** 2 / total_var) if total_var > 0 else 0.0
                )
            except np.linalg.LinAlgError:
                self._chain_axis = np.zeros(self.n_feat)
                self._chain_axis[np.argmax(X.var(axis=0))] = 1.0
                self.pc1_explained_variance_ratio_ = None

            X_proj = X @ self._chain_axis
            p_min, p_max = X_proj.min(), X_proj.max()
            self._p_min, self._p_max = p_min, p_max

            q_steps  = np.linspace(0.0, 1.0, self.m + 1)
            p_bounds = np.quantile(X_proj, q_steps)
            step     = (p_max - p_min) / self.m
            noise    = self.rng.uniform(-0.10, 0.10, self.m + 1) * step
            noise[[0, -1]] = 0
            p_bounds = np.sort(p_bounds + noise)
            p_bounds[0], p_bounds[-1] = p_min, p_max
            self._proj_bounds  = p_bounds
            self._proj_centers = (p_bounds[:-1] + p_bounds[1:]) / 2

            X_mean = X.mean(axis=0)
            self.L  = np.zeros((self.m + 1, self.n_feat))
            self.L[0] = self.L_min
            self.L[-1] = self.L_max
            for j in range(1, self.m):
                self.L[j] = np.clip(
                    X_mean + p_bounds[j] * self._chain_axis,
                    self.L_min + 1e-9, self.L_max - 1e-9
                )

            self.r = np.diff(self.L, axis=0)[:-1]
            self._clip_r()
        else:
            self._chain_axis   = None
            self._proj_bounds  = None
            self._proj_centers = None
            self.r = np.zeros((0, self.n_feat))

        self._compute_bounds()

    def _clip_r(self):
        ranges  = self.L_max - self.L_min
        min_val = np.maximum(ranges * 0.02, 1e-8)
        for j in range(len(self.r)):
            self.r[j] = np.maximum(self.r[j], min_val)
        for f in range(self.n_feat):
            total = self.r[:, f].sum()
            avail = ranges[f] * 0.98
            if total > avail and avail > 0:
                self.r[:, f] *= avail / total

    def _compute_bounds(self):
        self.L = np.zeros((self.m + 1, self.n_feat))
        self.L[0] = self.L_min
        for j in range(1, self.m):
            self.L[j] = np.minimum(self.L[j-1] + self.r[j-1], self.L_max - 1e-9)
        self.L[self.m] = self.L_max
        self.centers = np.array([(self.L[j] + self.L[j+1]) / 2 for j in range(self.m)])

        if getattr(self, '_chain_axis', None) is not None:
            p = ([self._p_min]
                 + [float(np.dot(self.L[j], self._chain_axis)) for j in range(1, self.m)]
                 + [self._p_max])
            self._proj_centers = np.array([(p[j] + p[j+1]) / 2 for j in range(self.m)])

    def _partition(self, X):
        if getattr(self, '_chain_axis', None) is not None:
            X_proj = X @ self._chain_axis
            dists  = np.abs(X_proj[:, None] - self._proj_centers[None, :])
        else:
            dists = np.stack([
                np.linalg.norm(X - self.centers[j], axis=1)
                for j in range(self.m)
            ], axis=1)
        return np.argmin(dists, axis=1)

    def _fit_molecule(self, mol, X_p, y_p):
        if len(X_p) == 0:
            return 0.0

        n, k = self.n_feat, mol.k

        def objective_and_grad(params):
            sigma = params[:n]
            H     = params[n : n + k * n].reshape(k, n)
            bias  = float(params[-1]) if mol.use_bias else 0.0

            terms  = np.ones((len(X_p), n))
            factor = np.ones((k, len(X_p), n))
            for i in range(k):
                factor[i] = X_p - H[i]
                terms     *= factor[i]

            phi      = (sigma * terms).sum(axis=1) + bias
            residual = y_p - phi
            loss     = 0.5 * np.mean(residual ** 2)

            g_sigma = -np.mean(residual[:, None] * terms, axis=0)
            g_H = np.zeros((k, n))
            for i in range(k):
                with np.errstate(divide='ignore', invalid='ignore'):
                    cofactor = np.where(factor[i] != 0, terms / factor[i], 0.0)
                g_H[i] = np.mean(residual[:, None] * sigma * cofactor, axis=0)

            grad = np.concatenate([g_sigma, g_H.ravel()])
            if mol.use_bias:
                grad = np.append(grad, -np.mean(residual))
            return loss, grad

        res = minimize(objective_and_grad, mol.get_params(), method='L-BFGS-B',
                       jac=True, options={'maxiter': 150, 'ftol': 1e-10, 'gtol': 1e-7})
        mol.set_params(res.x)

        preds = mol.evaluate_batch(X_p)
        E_j   = 0.5 * np.mean((y_p - np.round(np.clip(preds, -2.0, 3.0))) ** 2)
        return E_j

    # ── Entrenamiento — Algoritmo 1 ────────────────────────────────────────────

    def _snapshot(self):
        import copy
        return (copy.deepcopy(self.molecules), self.r.copy())

    def _restore(self, snapshot):
        self.molecules, self.r = snapshot
        self._compute_bounds()

    def fit(self, X, y, verbose=True):
        self.n_feat    = X.shape[1]
        self.molecules = [AHNMolecule(k, self.n_feat, self.rng, use_bias=self.use_bias)
                          for k in self.k_orders]
        self._init_bounds(X)
        self.history = []

        best_E       = np.inf
        best_snap    = self._snapshot()
        no_improve   = 0

        for it in range(self.max_iter):
            assignments = self._partition(X)
            errors = []
            for j in range(self.m):
                mask = assignments == j
                errors.append(
                    self._fit_molecule(self.molecules[j], X[mask], y[mask])
                    if mask.sum() > 0 else 0.0
                )

            E_global = sum(errors)
            self.history.append(E_global)

            # ── Best-state tracking ───────────────────────────────────────────
            if E_global < best_E:
                best_E    = E_global
                best_snap = self._snapshot()
                no_improve = 0
            else:
                no_improve += 1

            if verbose and (it % 10 == 0 or it < 3):
                sizes = [(assignments == j).sum() for j in range(self.m)]
                star  = '*' if E_global == best_E else ' '
                print(f"  Iter {it+1:3d}{star}| E_global={E_global:.6f} | partitions={sizes}")

            if E_global <= self.epsilon:
                if verbose:
                    print(f"  Converged at iter {it+1}  (E={E_global:.6f} <= {self.epsilon})")
                break

            if self.m > 1:
                E_ext = [0.0] + errors
                for j in range(self.m - 1):
                    self.r[j] += -self.eta * (E_ext[j] - E_ext[j+1])
                self._clip_r()
                self._compute_bounds()

                if no_improve >= self.patience and self.m > 1:
                    self._init_bounds(X)
                    no_improve = 0
                    if verbose:
                        print(f"  ↺ Reinit bounds at iter {it+1}  (no improvement for {self.patience} iters)")

        self._restore(best_snap)
        self.best_E_ = best_E
        if verbose:
            print(f"  ✓ Best-state restored  (E_best={best_E:.6f})")
        return self

    def predict_raw(self, X):
        assignments = self._partition(X)
        result = np.zeros(len(X))
        for j in range(self.m):
            mask = assignments == j
            if mask.sum() > 0:
                result[mask] = self.molecules[j].evaluate_batch(X[mask])
        return result

    def predict(self, X):
        return (self.predict_raw(X) >= self.threshold).astype(int)

    def predict_proba(self, X):
        raw  = self.predict_raw(X)
        prob = 1.0 / (1.0 + np.exp(-raw))
        return np.column_stack([1 - prob, prob])


class AHNMixture:

    def __init__(self, n_compounds=1, n_molecules=3, learning_rate=0.1,
                 tolerance=0.01, max_iterations=80, random_state=42,
                 use_bias=False, use_bce=False,
                 threshold=0.5, patience=20, n_restarts=1):
        self.c             = n_compounds
        self.m             = n_molecules
        self.eta           = learning_rate
        self.epsilon       = tolerance
        self.max_iter      = max_iterations
        self.rs            = random_state
        self.use_bias      = use_bias
        self.use_bce       = use_bce
        self.threshold     = threshold
        self.patience      = patience       # iters sin mejora para reinit bounds
        self.n_restarts    = n_restarts     # entrena N veces, queda con el mejor
        self.compounds     = []
        self.alphas        = None
        self.platt_a       = None
        self.platt_b       = None

    def _make_compound(self, seed):
        return AHNCompound(
            n_molecules=self.m, n_features=None,
            learning_rate=self.eta, tolerance=self.epsilon,
            max_iterations=self.max_iter, random_state=seed,
            use_bias=self.use_bias, threshold=self.threshold,
            patience=self.patience,
        )

    def fit(self, X, y, verbose=True):
        best_compounds = None
        best_E_total   = np.inf

        for restart in range(self.n_restarts):
            if verbose and self.n_restarts > 1:
                print(f"\n  [Restart {restart+1}/{self.n_restarts}]")

            compounds_try = []
            E_total = 0.0
            for i in range(self.c):
                if verbose:
                    mid = 'CH2-' * (self.m - 2)
                    print(f"\n  Compound {i+1}/{self.c}  (CH3-{mid}CH3):")
                seed = self.rs + i + restart * self.c
                comp = self._make_compound(seed)
                comp.fit(X, y, verbose=verbose)
                E_total += getattr(comp, 'best_E_', np.inf)
                compounds_try.append(comp)

            if E_total < best_E_total:
                best_E_total   = E_total
                best_compounds = compounds_try
                if verbose and self.n_restarts > 1:
                    print(f"  ★ New best  E={E_total:.6f}  (restart {restart+1})")

        self.compounds = best_compounds

        if self.c > 1:
            Psi = np.column_stack([c.predict_raw(X) for c in self.compounds])
            if self.use_bce:
                from sklearn.linear_model import LogisticRegression
                lr = LogisticRegression(C=1e4, solver='lbfgs',
                                        max_iter=1000, fit_intercept=False)
                lr.fit(Psi, y.astype(int))
                self.alphas = lr.coef_[0]
            else:
                self.alphas, *_ = lstsq(Psi, y.astype(float))
        else:
            self.alphas = np.array([1.0])
        return self

    def fit_platt(self, X_val, y_val):
        from sklearn.linear_model import LogisticRegression
        scores = self.predict_raw(X_val).reshape(-1, 1)
        lr = LogisticRegression(C=1e6, solver='lbfgs', max_iter=1000)
        lr.fit(scores, y_val)
        self.platt_a = float(lr.coef_[0, 0])
        self.platt_b = float(lr.intercept_[0])
        return self

    def predict_raw(self, X):
        Psi = np.array([c.predict_raw(X) for c in self.compounds])
        return self.alphas @ Psi

    def predict_proba(self, X):
        raw   = self.predict_raw(X)
        logit = (self.platt_a * raw + self.platt_b) if self.platt_a is not None else raw
        prob  = 1.0 / (1.0 + np.exp(-logit))
        return np.column_stack([1 - prob, prob])

    def predict(self, X):
        # Uses the calibrated probability once fit_platt() has been called, so
        # AHN's hard labels (and therefore Accuracy/Precision/Recall/F1) are
        # consistent with its own stated calibration step, and directly
        # comparable to CalibratedBaseline.predict() below. Falls back to the
        # raw uncalibrated score only if fit_platt() was never called.
        if self.platt_a is not None:
            prob = self.predict_proba(X)[:, 1]
            return (prob >= self.threshold).astype(int)
        return (self.predict_raw(X) >= self.threshold).astype(int)


class CalibratedBaseline:
    """
    Wraps a fitted sklearn classifier (SVM, RF, MLP) with a post-hoc Platt/
    logistic calibrator fit on the model's own raw decision score, using the
    exact same protocol as AHNMixture.fit_platt: a 1-D logistic regression of
    y_val on the raw score, evaluated only on the held-out validation set.

    "Raw score" = decision_function(X) when available (SVM's signed margin),
    otherwise the model's native predict_proba(X)[:, 1] (RF, MLP), which plays
    the same role AHN's predict_raw(X) plays for AHNMixture.

    predict() thresholds the CALIBRATED probability at 0.5 (not the model's
    own internal .predict(), which for some sklearn estimators — notably SVC —
    can use a different internal decision rule than predict_proba). This gives
    every model in the comparison the same calibration procedure and the same
    definition of "0.5 threshold" for Accuracy/Precision/Recall/F1.
    """

    def __init__(self, base_model, threshold=0.5):
        self.base_model = base_model
        self.threshold  = threshold
        self.platt_a    = None
        self.platt_b    = None

    def _raw_score(self, X):
        if hasattr(self.base_model, 'decision_function'):
            return self.base_model.decision_function(X)
        return self.base_model.predict_proba(X)[:, 1]

    def fit_platt(self, X_val, y_val):
        raw = self._raw_score(X_val).reshape(-1, 1)
        lr  = LogisticRegression(C=1e6, solver='lbfgs', max_iter=1000)
        lr.fit(raw, y_val)
        self.platt_a = float(lr.coef_[0, 0])
        self.platt_b = float(lr.intercept_[0])
        return self

    def predict_proba(self, X):
        raw   = self._raw_score(X)
        logit = (self.platt_a * raw + self.platt_b) if self.platt_a is not None else raw
        prob  = 1.0 / (1.0 + np.exp(-logit))
        return np.column_stack([1 - prob, prob])

    def predict(self, X):
        prob = self.predict_proba(X)[:, 1]
        return (prob >= self.threshold).astype(int)


# ══════════════════════════════════════════════════════════════════════════════
#  BLOQUE 2 — CARGAR Y PREPARAR DATOS
# ══════════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("FULL COMPARISON: AHN vs Baseline Models — NASA Exoplanets KOI")
print("=" * 70)

NASA_URL = ("https://exoplanetarchive.ipac.caltech.edu/cgi-bin/nstedAPI/"
            "nph-nstedAPI?table=cumulative&format=csv")

FEATURES = [
    'koi_period',    # Período orbital [días]
    'koi_time0bk',   # Tiempo de referencia del tránsito [días BJD]
    'koi_impact',    # Parámetro de impacto
    'koi_duration',  # Duración del tránsito [horas]
    'koi_depth',     # Profundidad del tránsito [ppm]
    'koi_prad',      # Radio del planeta [R_Tierra]
    'koi_teq',       # Temperatura de equilibrio [K]
    'koi_insol',     # Insolación [F_Tierra]
    'koi_model_snr', # Señal-ruido del modelo
    'koi_steff',     # Temperatura efectiva estelar [K]
]

try:
    df = pd.read_csv(NASA_URL)
    print(f"Dataset downloaded from NASA Exoplanet Archive: {len(df)} rows")
    df_sel = df[FEATURES + ['koi_disposition']].dropna()
    df_bin = df_sel[
        df_sel['koi_disposition'].isin(['CONFIRMED', 'FALSE POSITIVE'])
    ].copy()
    df_bin['label'] = df_bin['koi_disposition'].map({'CONFIRMED': 1, 'FALSE POSITIVE': 0})
    X_array = df_bin[FEATURES].values.astype(float)
    y       = df_bin['label'].values
    DATA_SOURCE = "NASA Exoplanet Archive KOI"
    _df_full_for_errors = df   # kept only to look up *_err1/*_err2 columns below
except Exception:
    try:
        df = pd.read_csv("cumulative_koi.csv")
        print("Dataset loaded from local file")
        df_sel = df[FEATURES + ['koi_disposition']].dropna()
        df_bin = df_sel[
            df_sel['koi_disposition'].isin(['CONFIRMED', 'FALSE POSITIVE'])
        ].copy()
        df_bin['label'] = df_bin['koi_disposition'].map({'CONFIRMED': 1, 'FALSE POSITIVE': 0})
        X_array = df_bin[FEATURES].values.astype(float)
        y       = df_bin['label'].values
        DATA_SOURCE = "NASA KOI (local)"
        _df_full_for_errors = df
    except Exception:
        print("⚠  Real dataset unavailable — using synthetic data that replicates")
        print("   the characteristics of the KOI catalog:")
        print("   2000 samples · 10 features · 55/45 imbalance · transit signal")
        from sklearn.datasets import make_classification
        X_array, y = make_classification(
            n_samples=2000, n_features=10, n_informative=6, n_redundant=2,
            n_clusters_per_class=2, weights=[0.45, 0.55],
            flip_y=0.04, random_state=42
        )
        DATA_SOURCE = "Synthetic (KOI replica)"
        _df_full_for_errors = None   # no archive columns to pull from

# ── Incertidumbre real por objeto/feature (Reviewer #1, punto de menor
# prioridad): la tabla KOI publica columnas de error por objeto para las
# cantidades físicas medidas, con la convención {feature}_err1 (superior) /
# {feature}_err2 (inferior). Las recolectamos aquí, alineadas fila-a-fila con
# X_array, para poder usarlas más adelante como una alternativa físicamente
# fundamentada al ruido gaussiano de sigma plano en el experimento de Feature
# Noise. Si una feature no tiene columnas de error en este pull del archivo
# (p.ej. koi_model_snr, que es un diagnóstico derivado sin incertidumbre de
# medición propia, o datos sintéticos), esa columna se marca sin cobertura y
# el generador de ruido físico recae automáticamente en el esquema de sigma
# plano SOLO para esa feature — nunca en un fallback global silencioso.
_ERR_SUFFIXES = ('_err1', '_err2')
_feat_has_err_cols = {feat: False for feat in FEATURES}
U_raw_all = None

if _df_full_for_errors is not None:
    U_df = pd.DataFrame(index=df_bin.index, columns=FEATURES, dtype=float)
    for feat in FEATURES:
        c1, c2 = f'{feat}{_ERR_SUFFIXES[0]}', f'{feat}{_ERR_SUFFIXES[1]}'
        if c1 in _df_full_for_errors.columns and c2 in _df_full_for_errors.columns:
            e1 = _df_full_for_errors.loc[df_bin.index, c1].abs()
            e2 = _df_full_for_errors.loc[df_bin.index, c2].abs()
            u  = (e1 + e2) / 2.0
            u  = u.replace(0.0, np.nan)   # a reported 0 is a placeholder, not real precision
            if u.notna().sum() > 0:
                U_df[feat] = u
                _feat_has_err_cols[feat] = True
    if any(_feat_has_err_cols.values()):
        for feat in FEATURES:
            if _feat_has_err_cols[feat]:
                U_df[feat] = U_df[feat].fillna(U_df[feat].median())
        U_raw_all = U_df.values   # same row order as X_array — positional indexing matches


idx_all = np.arange(len(X_array))
idx_temp, idx_test = train_test_split(
    idx_all, test_size=0.2, stratify=y, random_state=42
)
idx_train, idx_val = train_test_split(
    idx_temp, test_size=0.25, stratify=y[idx_temp], random_state=42
)
# Identical partition to splitting X_array/y directly with the same
# random_state — index-based only so U_raw_all can be sliced the same way.
X_train, X_val, X_test = X_array[idx_train], X_array[idx_val], X_array[idx_test]
y_train, y_val, y_test = y[idx_train],       y[idx_val],       y[idx_test]

scaler  = MinMaxScaler(feature_range=(-1, 1))
X_train = scaler.fit_transform(X_train)
X_val   = scaler.transform(X_val)
X_test  = scaler.transform(X_test)

# Incertidumbre real por objeto, para el test set, en el mismo espacio escalado
# [-1,1] donde se inyecta el ruido: una incertidumbre es una diferencia, así
# que solo aplica la parte multiplicativa del escalado Min-Max (el offset
# aditivo se cancela). Features sin cobertura real quedan en 1.0, que
# reproduce exactamente el ruido plano sigma*N(0,1) original para esa feature.
U_test_scaled = np.ones((len(idx_test), len(FEATURES)))
if U_raw_all is not None:
    for _j, _feat in enumerate(FEATURES):
        if _feat_has_err_cols[_feat]:
            U_test_scaled[:, _j] = U_raw_all[idx_test, _j] * scaler.scale_[_j]

print(f"Features: {X_train.shape[1]}")
print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
print(f"Test balance — Confirmed: {(y_test==1).sum()}, False Positive: {(y_test==0).sum()}")

# ── RESUMEN DEL DATASET  (Reviewer #1: N, class ratio, feature names) ────────
print("\n" + "=" * 70)
print("DATASET SUMMARY  (for Section III-A of the paper)")
print("=" * 70)
print(f"  Data source            : {DATA_SOURCE}")
print(f"  N total (after dropna) : {len(X_array)}")
print(f"  Overall balance        : Confirmed={int((y==1).sum())} ({(y==1).mean():.1%})   "
      f"False Positive={int((y==0).sum())} ({(y==0).mean():.1%})")
print(f"  N train / val / test   : {len(X_train)} / {len(X_val)} / {len(X_test)}")
print(f"  Train balance          : Confirmed={int((y_train==1).sum())} ({(y_train==1).mean():.1%})   "
      f"False Positive={int((y_train==0).sum())} ({(y_train==0).mean():.1%})")
print(f"  Test balance           : Confirmed={int((y_test==1).sum())} ({(y_test==1).mean():.1%})   "
      f"False Positive={int((y_test==0).sum())} ({(y_test==0).mean():.1%})")
print(f"  Features ({len(FEATURES)})       : {', '.join(FEATURES)}")

_n_with_err = sum(_feat_has_err_cols.values())
if U_raw_all is not None:
    print(f"  Real uncertainty (err1/err2) available for {_n_with_err}/{len(FEATURES)} features:")
    for _feat in FEATURES:
        if _feat_has_err_cols[_feat]:
            _rel = np.median(U_raw_all[:, FEATURES.index(_feat)]) / (np.abs(X_array[:, FEATURES.index(_feat)]).mean() + 1e-12)
            print(f"    {_feat:<15} yes  (median relative uncertainty ≈ {_rel:.2%})")
        else:
            print(f"    {_feat:<15} no  (no err1/err2 columns in this pull — will use flat sigma)")
else:
    print(f"  Real uncertainty (err1/err2): not available (source = {DATA_SOURCE}) — "
          f"Exp. 2b (physical noise) will use flat sigma as in Exp. 2")

pd.DataFrame([{
    'data_source': DATA_SOURCE, 'n_total': len(X_array),
    'n_confirmed': int((y==1).sum()), 'n_false_positive': int((y==0).sum()),
    'n_train': len(X_train), 'n_val': len(X_val), 'n_test': len(X_test),
    'n_features': len(FEATURES), 'features': ';'.join(FEATURES),
}]).to_csv(out('dataset_summary.csv'), index=False)
print("  Saved: dataset_summary.csv")

# ── CONTROL: clasificador trivial "todo positivo"  (Reviewer #1) ─────────────
# Distingue una ventaja de robustez genuina de un modelo que simplemente
# degenera hacia la clase positiva. Sus métricas dependen solo del balance de
# y_test/y_train/y_val (nunca de X), por lo que son idénticas en el
# experimento principal y en los tres escenarios de robustez: ninguno de ellos
# modifica el balance de clases de test.
def _all_positive_metrics(y_true):
    pred = np.ones_like(y_true)
    return {
        'acc':       accuracy_score(y_true, pred),
        'precision': precision_score(y_true, pred, zero_division=0),
        'recall':    recall_score(y_true, pred, zero_division=0),   # = 1.0 por construcción
        'f1':        f1_score(y_true, pred, zero_division=0),
        'ap':        float((y_true == 1).mean()),  # AP de un score constante = prevalencia positiva
        'auc':       0.5,                           # ROC-AUC de un score constante = 0.5 por definición
    }

ALL_POSITIVE_CONTROL = {
    'train_acc': _all_positive_metrics(y_train)['acc'],
    'val_acc':   _all_positive_metrics(y_val)['acc'],
    **_all_positive_metrics(y_test),
}
print(f"\nControl (All-Positive): Acc={ALL_POSITIVE_CONTROL['acc']:.4f}, "
      f"Precision={ALL_POSITIVE_CONTROL['precision']:.4f}, Recall={ALL_POSITIVE_CONTROL['recall']:.4f}, "
      f"F1={ALL_POSITIVE_CONTROL['f1']:.4f}  "
      f"(fixed reference: ignores X, unaffected by noise/scarcity)")


# ══════════════════════════════════════════════════════════════════════════════
#  BLOQUE 3 — ENTRENAR AHN
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("Training AHN (Artificial Hydrocarbon Networks)...")
print("=" * 70)


AHN_CONFIG = dict(
    n_compounds    = 1,
    n_molecules    = 2,
    learning_rate  = 0.3,   # eta
    tolerance      = 0.1,  # epsilon
    max_iterations = 100,
    random_state   = 42,
    use_bias       = True,
    use_bce        = False,
    threshold      = 0.5,
    patience       = 20,    # iters sin mejora antes de reinit bounds
    n_restarts     = 1,     # Ablation (20 seeds, n_restarts=1 vs 3): AUC/AP
                             # identical to 3-4 decimals either way; F1 variance
                             # roughly halves with 3 restarts (0.0050→0.0026) but
                             # its mean doesn't improve — best-state tracking
                             # already converges to nearly the same result
                             # regardless of initialization, so restarts weren't
                             # earning their 3x training cost on this data.
)
m = AHN_CONFIG['n_molecules']
if   m == 1: _struct = "CH3"
elif m == 2: _struct = "CH3-CH3"
else:        _struct = "CH3-" + "CH2-" * (m - 2) + "CH3"
print(f"Structure: {_struct}  |  eta={AHN_CONFIG['learning_rate']}  "
      f"|  eps={AHN_CONFIG['tolerance']}  |  max_iter={AHN_CONFIG['max_iterations']}  "
      f"|  bias={AHN_CONFIG['use_bias']}  |  bce={AHN_CONFIG['use_bce']}")

# ── Construcción unificada y semillada de baselines ───────────────────────────
# Un único punto de verdad para los hiperparámetros de SVM/RF/MLP, usado por el
# experimento principal, la CV y los tres barridos de robustez (antes había
# tres copias separadas que podían divergir). Aceptar `seed` es lo que permite
# las corridas multi-semilla del Bloque 8 en adelante.
#
# Manejo de desbalance de clases (Reviewer #2 — "unificar o justificar"):
#   Ningún modelo recibe tratamiento especial por desbalance de clases. La
#   versión original solo pesaba SVM (class_weight='balanced'); una versión
#   intermedia intentó "igualar" el tratamiento dándole a RF su propio
#   class_weight='balanced' nativo y a MLP un oversampling por frecuencia
#   inversa como sustituto (MLPClassifier.fit() no acepta sample_weight ni
#   class_weight). Se descartó esa vía: el oversampling de MLP no es en
#   realidad el mismo mecanismo que el reweighting de SVM/RF, así que
#   "igualar" mecanismos distintos seguía dejando una asimetría real, solo
#   menos visible. La opción más simple y más defendible es la otra rama que
#   ofrece el reviewer — unificar quitándolo en vez de añadiéndolo — de modo
#   que los cuatro modelos (incluido AHN, que nunca tuvo ningún mecanismo de
#   este tipo) se entrenan exactamente igual frente al desbalance de clases:
#   ninguno.
def make_baselines(seed=42):
    return {
        'SVM': SVC(kernel='rbf', probability=True, random_state=seed),
        'Random Forest': RandomForestClassifier(
            n_estimators=100, max_depth=10, min_samples_split=5,
            random_state=seed),
        'MLP': MLPClassifier(
            hidden_layer_sizes=(64, 32), activation='relu', solver='adam',
            alpha=0.001, learning_rate_init=0.001, max_iter=300,
            early_stopping=True, validation_fraction=0.15, random_state=seed),
    }

def fit_baseline(name, model, X, y, rng=None):
    """Fits a baseline. Kept as the single entry point used by every call site
    (main experiment, CV, all three robustness sweeps) so nothing else needs
    to change if a per-model mechanism is ever reintroduced. `rng` is accepted
    for call-site compatibility but currently unused — no model gets special
    imbalance handling."""
    model.fit(X, y)
    return model

ahn = AHNMixture(**AHN_CONFIG)
ahn.fit(X_train, y_train, verbose=True)

_pc1_var = ahn.compounds[0].pc1_explained_variance_ratio_
if _pc1_var is not None:
    print(f"\nVariance explained by the 1st principal component "
          f"(PCA, quantile partitioning): {_pc1_var:.4f}  ({_pc1_var*100:.2f}%)")

ahn.fit_platt(X_val, y_val)
print(f"Platt Scaling fitted:  a={ahn.platt_a:.4f}  b={ahn.platt_b:.4f}")

y_pred_ahn  = ahn.predict(X_test)
y_proba_ahn = ahn.predict_proba(X_test)[:, 1]

ahn_results = {
    'train_acc': accuracy_score(y_train, ahn.predict(X_train)),
    'val_acc':   accuracy_score(y_val,   ahn.predict(X_val)),
    'test_acc':  accuracy_score(y_test,  y_pred_ahn),
    'precision': precision_score(y_test, y_pred_ahn, zero_division=0),
    'recall':    recall_score(y_test,    y_pred_ahn, zero_division=0),
    'f1':        f1_score(y_test,        y_pred_ahn, zero_division=0),
    'roc_auc':   roc_auc_score(y_test,   y_proba_ahn),
    'avg_precision': average_precision_score(y_test, y_proba_ahn),
    'y_pred':    y_pred_ahn,
    'y_proba':   y_proba_ahn,
    'confusion_matrix': confusion_matrix(y_test, y_pred_ahn),
}
print(f"\nAHN: Acc={ahn_results['test_acc']:.4f}, "
      f"F1={ahn_results['f1']:.4f}, "
      f"AUC-PR={ahn_results['avg_precision']:.4f}, "
      f"ROC-AUC={ahn_results['roc_auc']:.4f}")


# ── BLOQUE 3b — K-FOLD CROSS VALIDATION  (LOS 4 MODELOS) ─────────────────────
# Reviewer #2: reportar CV con media y desviación estándar por modelo y
# métrica, como evidencia de que AHN, SVM, RF y MLP son comparables.
#
# Los 4 modelos comparten exactamente los mismos folds de StratifiedKFold, y
# cada fold reserva un 20% interno de su porción de entrenamiento solo para
# calibración Platt (nunca visto por el ajuste del modelo ni por el fold de
# evaluación) — la misma disciplina anti-leakage ya usada para AHN, ahora
# aplicada de forma uniforme a los tres baselines también.
# ─────────────────────────────────────────────────────────────────────────────

from sklearn.model_selection import StratifiedKFold

K_FOLDS  = 10   # Kohavi (1995): 10-fold stratified CV is the standard recommendation
                 # for accuracy estimation "even if computation power allows using
                 # more folds." Our train+val union (~5,860 rows on real KOI data)
                 # is well past the n>1000 threshold where 10-fold is standard
                 # practice (vs. 5-fold for smaller datasets), and comfortably
                 # affordable computationally.
kf       = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
X_cv     = np.vstack([X_train, X_val])   # usamos todo lo que no es test
y_cv     = np.concatenate([y_train, y_val])

_CV_MODELS  = ['AHN', 'SVM', 'Random Forest', 'MLP']
_CV_METRICS = ['auc', 'ap', 'acc', 'f1']   # must match the printed header order: AUC | AUC-PR | ACC | F1
cv_scores   = {m: {met: [] for met in _CV_METRICS} for m in _CV_MODELS}

print(f"\n{'─'*78}")
print(f"K-FOLD CROSS VALIDATION  (K={K_FOLDS}, StratifiedKFold, data=train+val, 4 models)")
print(f"{'─'*78}")
print(f"  {'Fold':>5}  {'Model':<16}  {'AUC':>7}  {'AUC-PR':>7}  {'ACC':>7}  {'F1':>7}")

for fold, (tr_idx, val_idx) in enumerate(kf.split(X_cv, y_cv), 1):
    X_tr, X_ho = X_cv[tr_idx], X_cv[val_idx]
    y_tr, y_ho = y_cv[tr_idx], y_cv[val_idx]

    # Split interno para Platt — 20% del train fold, nunca del held-out
    n_platt   = max(20, int(0.20 * len(X_tr)))
    X_platt   = X_tr[-n_platt:]
    y_platt   = y_tr[-n_platt:]
    X_tr_pure = X_tr[:-n_platt]
    y_tr_pure = y_tr[:-n_platt]
    _rng_cv   = np.random.default_rng(42 + fold)

    fold_scores = {}

    cfg_fold = {**AHN_CONFIG, 'random_state': 42 + fold}
    m_ahn = AHNMixture(**cfg_fold)
    m_ahn.fit(X_tr_pure, y_tr_pure, verbose=False)
    m_ahn.fit_platt(X_platt, y_platt)
    yp, ypr = m_ahn.predict(X_ho), m_ahn.predict_proba(X_ho)[:, 1]
    fold_scores['AHN'] = dict(acc=accuracy_score(y_ho, yp), f1=f1_score(y_ho, yp, zero_division=0),
                               ap=average_precision_score(y_ho, ypr), auc=roc_auc_score(y_ho, ypr))

    for name, bm in make_baselines(42 + fold).items():
        fit_baseline(name, bm, X_tr_pure, y_tr_pure, _rng_cv)
        cal = CalibratedBaseline(bm).fit_platt(X_platt, y_platt)
        yp, ypr = cal.predict(X_ho), cal.predict_proba(X_ho)[:, 1]
        fold_scores[name] = dict(acc=accuracy_score(y_ho, yp), f1=f1_score(y_ho, yp, zero_division=0),
                                  ap=average_precision_score(y_ho, ypr), auc=roc_auc_score(y_ho, ypr))

    for name in _CV_MODELS:
        for met in _CV_METRICS:
            cv_scores[name][met].append(fold_scores[name][met])
        s = fold_scores[name]
        print(f"  {fold:>5}  {name:<16}  {s['auc']:.4f}   {s['ap']:.4f}   {s['acc']:.4f}   {s['f1']:.4f}")

print(f"{'─'*78}")
print(f"  {'Model':<16}  {'AUC':>15}  {'AUC-PR':>15}  {'ACC':>15}  {'F1':>15}")
for name in _CV_MODELS:
    row = f"  {name:<16}"
    for met in _CV_METRICS:
        vals = cv_scores[name][met]
        row += f"  {np.mean(vals):.4f}±{np.std(vals):.4f}"
    print(row)
print(f"\n  [Actual AHN test: "
      f"AUC={ahn_results['roc_auc']:.4f}  "
      f"AUC-PR={ahn_results['avg_precision']:.4f}  "
      f"ACC={ahn_results['test_acc']:.4f}  "
      f"F1={ahn_results['f1']:.4f}]")

pd.DataFrame([
    {'model': name, 'metric': met, 'fold': i + 1, 'value': v}
    for name in _CV_MODELS for met in _CV_METRICS
    for i, v in enumerate(cv_scores[name][met])
]).to_csv(out('cv_results_all_models.csv'), index=False)
print("  Saved: cv_results_all_models.csv")


# ══════════════════════════════════════════════════════════════════════════════
#  BLOQUE 4 — ENTRENAR MODELOS BASELINE
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("Training baseline models...")
print("=" * 70)

baseline_models       = make_baselines(seed=42)
baseline_results      = {}
calibrated_baselines  = {}   # kept for reuse in robustness experiments if needed
_rng_main = np.random.default_rng(42)
for name, model in baseline_models.items():
    fit_baseline(name, model, X_train, y_train, _rng_main)

    # Uniform calibration: same Platt procedure as AHN, fit on the same
    # validation set. Hard labels now come from the calibrated probability
    # for every model, not from each estimator's own internal decision rule.
    cal = CalibratedBaseline(model).fit_platt(X_val, y_val)
    calibrated_baselines[name] = cal

    y_pred  = cal.predict(X_test)
    y_proba = cal.predict_proba(X_test)[:, 1]
    baseline_results[name] = {
        'train_acc': accuracy_score(y_train, cal.predict(X_train)),
        'val_acc':   accuracy_score(y_val,   cal.predict(X_val)),
        'test_acc':  accuracy_score(y_test,  y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall':    recall_score(y_test,    y_pred, zero_division=0),
        'f1':        f1_score(y_test,        y_pred, zero_division=0),
        'roc_auc':   roc_auc_score(y_test,   y_proba),
        'avg_precision': average_precision_score(y_test, y_proba),
        'y_pred':    y_pred,
        'y_proba':   y_proba,
        'confusion_matrix': confusion_matrix(y_test, y_pred),
    }
    print(f"  {name}: Acc={baseline_results[name]['test_acc']:.4f}, "
          f"F1={baseline_results[name]['f1']:.4f}, "
          f"AUC-PR={baseline_results[name]['avg_precision']:.4f}, "
          f"ROC-AUC={baseline_results[name]['roc_auc']:.4f}  "
          f"(Platt a={cal.platt_a:.3f}, b={cal.platt_b:.3f})")


# ══════════════════════════════════════════════════════════════════════════════
#  BLOQUE 5 — TABLA COMPARATIVA
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("FULL COMPARISON TABLE")
print("=" * 70)

rows = []
for name, res in baseline_results.items():
    rows.append({'Model': name, 'Type': 'Baseline',
                 'Train Acc': res['train_acc'], 'Val Acc': res['val_acc'],
                 'Test Acc':  res['test_acc'],  'Precision': res['precision'],
                 'Recall':    res['recall'],    'F1-Score':  res['f1'],
                 'AUC-PR': res['avg_precision'],
                 'ROC-AUC':   res['roc_auc']})
rows.append({'Model': 'AHN', 'Type': 'AHN',
             'Train Acc': ahn_results['train_acc'], 'Val Acc': ahn_results['val_acc'],
             'Test Acc':  ahn_results['test_acc'],  'Precision': ahn_results['precision'],
             'Recall':    ahn_results['recall'],    'F1-Score':  ahn_results['f1'],
             'AUC-PR': ahn_results['avg_precision'],
             'ROC-AUC':   ahn_results['roc_auc']})
rows.append({'Model': 'All-Positive (Control)', 'Type': 'Control',
             'Train Acc': ALL_POSITIVE_CONTROL['train_acc'],
             'Val Acc':   ALL_POSITIVE_CONTROL['val_acc'],
             'Test Acc':  ALL_POSITIVE_CONTROL['acc'],
             'Precision': ALL_POSITIVE_CONTROL['precision'],
             'Recall':    ALL_POSITIVE_CONTROL['recall'],
             'F1-Score':  ALL_POSITIVE_CONTROL['f1'],
             'AUC-PR': ALL_POSITIVE_CONTROL['ap'],
             'ROC-AUC':   ALL_POSITIVE_CONTROL['auc']})

comparison_df = (pd.DataFrame(rows)
                   .sort_values('ROC-AUC', ascending=False)
                   .reset_index(drop=True))

print(comparison_df.drop(columns='Type').to_string(index=False))
comparison_df.to_csv(out('final_comparison_table.csv'), index=False)
print("\nTable saved: final_comparison_table.csv")

# Classification reports
print("\n" + "=" * 70)
print("CLASSIFICATION REPORTS")
print("=" * 70)
for name, res in {'AHN': ahn_results, **baseline_results}.items():
    print(f"\n{name}:")
    print(classification_report(y_test, res['y_pred'],
                                target_names=['False Positive', 'Confirmed'], digits=4))


# ══════════════════════════════════════════════════════════════════════════════
#  BLOQUE 6 — VISUALIZACIONES
# ══════════════════════════════════════════════════════════════════════════════

print("\nGenerating visualizations...")

plt.style.use('seaborn-v0_8-darkgrid')
colors_baseline = sns.color_palette("husl", len(baseline_results))
color_ahn = '#FF6B6B'

# ── FIGURA 1: Radar Chart ─────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(projection='polar'))
categories = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
N      = len(categories)
angles = [n / N * 2 * pi for n in range(N)] + [0]

for (name, res), color in zip(baseline_results.items(), colors_baseline):
    vals = [res['test_acc'], res['precision'], res['recall'],
            res['f1'], res['roc_auc'], res['test_acc']]
    ax.plot(angles, vals, 'o-', lw=2, label=name, color=color, alpha=0.7)
    ax.fill(angles, vals, alpha=0.1, color=color)

ahn_vals = [ahn_results['test_acc'], ahn_results['precision'], ahn_results['recall'],
            ahn_results['f1'], ahn_results['roc_auc'], ahn_results['test_acc']]
ax.plot(angles, ahn_vals, 'o-', lw=3, label='AHN', color=color_ahn)
ax.fill(angles, ahn_vals, alpha=0.2, color=color_ahn)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, size=12)
ax.set_ylim(0, 1)
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1), fontsize=10)
plt.title('Radar Chart: Comparison of All Metrics',
          size=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(out('final_radar_comparison.png'), dpi=300, bbox_inches='tight')
print("  Saved: final_radar_comparison.png")
plt.close()

# ── FIGURA 2: Curvas ROC ──────────────────────────────────────────────────────
plt.figure(figsize=(10, 8))
for (name, res), color in zip(baseline_results.items(), colors_baseline):
    fpr, tpr, _ = roc_curve(y_test, res['y_proba'])
    plt.plot(fpr, tpr, color=color, lw=2, alpha=0.7,
             label=f'{name} (AUC={res["roc_auc"]:.4f})')

fpr_a, tpr_a, _ = roc_curve(y_test, ahn_results['y_proba'])
plt.plot(fpr_a, tpr_a, color=color_ahn, lw=3,
         label=f'AHN (AUC={ahn_results["roc_auc"]:.4f})')
plt.fill_between(fpr_a, tpr_a, alpha=0.08, color=color_ahn)
plt.plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.5, label='Random (AUC=0.5000)')
plt.xlim([0, 1]); plt.ylim([0, 1.05])
plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
plt.ylabel('True Positive Rate (Sensitivity)', fontsize=12)
plt.title('Curvas ROC — AHN vs Baseline Models', fontsize=14, fontweight='bold')
plt.legend(loc='lower right', fontsize=10, framealpha=0.95)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(out('final_roc_curves_complete.png'), dpi=300, bbox_inches='tight')
print("  Saved: final_roc_curves_complete.png")
plt.close()

# ── FIGURA 3: Barras comparativas ─────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(16, 11))
fig.suptitle('Quantitative Comparison: AHN vs Baseline Models',
             fontsize=16, fontweight='bold')

models_order = list(baseline_results.keys()) + ['AHN']
x_pos      = np.arange(len(models_order))
colors_all = list(colors_baseline) + [color_ahn]
width = 0.35

def add_labels(ax, bars):
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.003,
                f'{h:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

# Accuracy
ax = axes[0, 0]
accs = [baseline_results[m]['test_acc'] for m in baseline_results] + [ahn_results['test_acc']]
add_labels(ax, ax.bar(x_pos, accs, color=colors_all, alpha=0.8, edgecolor='black', lw=1.2))
ax.set_title('Test Accuracy', fontsize=13, fontweight='bold')
ax.set_xticks(x_pos); ax.set_xticklabels(models_order, rotation=35, ha='right')
ax.set_ylim([max(0, min(accs) - 0.1), 1.0]); ax.set_ylabel('Accuracy'); ax.grid(axis='y', alpha=0.3)

# F1-Score
ax = axes[0, 1]
f1s = [baseline_results[m]['f1'] for m in baseline_results] + [ahn_results['f1']]
add_labels(ax, ax.bar(x_pos, f1s, color=colors_all, alpha=0.8, edgecolor='black', lw=1.2))
ax.set_title('F1-Score', fontsize=13, fontweight='bold')
ax.set_xticks(x_pos); ax.set_xticklabels(models_order, rotation=35, ha='right')
ax.set_ylim([max(0, min(f1s) - 0.1), 1.0]); ax.set_ylabel('F1'); ax.grid(axis='y', alpha=0.3)

# ROC-AUC
ax = axes[1, 0]
rocs = [baseline_results[m]['roc_auc'] for m in baseline_results] + [ahn_results['roc_auc']]
add_labels(ax, ax.bar(x_pos, rocs, color=colors_all, alpha=0.8, edgecolor='black', lw=1.2))
ax.set_title('ROC-AUC', fontsize=13, fontweight='bold')
ax.set_xticks(x_pos); ax.set_xticklabels(models_order, rotation=35, ha='right')
ax.set_ylim([max(0, min(rocs) - 0.1), 1.0]); ax.set_ylabel('AUC'); ax.grid(axis='y', alpha=0.3)

# Precision vs Recall
ax = axes[1, 1]
pres = [baseline_results[m]['precision'] for m in baseline_results] + [ahn_results['precision']]
recs = [baseline_results[m]['recall']    for m in baseline_results] + [ahn_results['recall']]
b1 = ax.bar(x_pos - width/2, pres, width, label='Precision',
            alpha=0.8, edgecolor='black', lw=1.2, color=colors_all)
b2 = ax.bar(x_pos + width/2, recs, width, label='Recall',
            alpha=0.8, edgecolor='black', lw=1.2, color=colors_all)
for bar in list(b1) + list(b2):
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + 0.003,
            f'{h:.2f}', ha='center', va='bottom', fontsize=8)
ax.set_title('Precision vs Recall', fontsize=13, fontweight='bold')
ax.set_xticks(x_pos); ax.set_xticklabels(models_order, rotation=35, ha='right')
ax.set_ylim([0, 1.05]); ax.legend(); ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(out('final_metrics_comparison.png'), dpi=300, bbox_inches='tight')
print("  Saved: final_metrics_comparison.png")
plt.close()

# ── FIGURA 4: Matrices de confusion ──────────────────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))
fig.suptitle('Confusion Matrices — AHN vs Baseline  (NASA Exoplanets KOI)',
             fontsize=14, fontweight='bold')

plot_items = [('AHN', ahn_results, color_ahn)] + \
             [(n, r, c) for (n, r), c in zip(baseline_results.items(), colors_baseline)]

for ax, (name, res, col) in zip(axes, plot_items):
    sns.heatmap(res['confusion_matrix'], annot=True, fmt='d', cmap='Blues',
                ax=ax, cbar=False,
                xticklabels=['FP', 'Confirmed'], yticklabels=['FP', 'Confirmed'])
    marker = '* ' if name == 'AHN' else ''
    ax.set_title(f'{marker}{name}\nAcc={res["test_acc"]:.4f}  F1={res["f1"]:.4f}',
                 fontsize=10, fontweight='bold', color=col)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual' if name == 'AHN' else '')

plt.tight_layout()
plt.savefig(out('ahn_confusion_matrices.png'), dpi=300, bbox_inches='tight')
print("  Saved: ahn_confusion_matrices.png")
plt.close()

# ── FIGURA 5: Convergencia AHN ────────────────────────────────────────────────
comp = ahn.compounds[0]
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Internal Structure of the AHN Compound', fontsize=13, fontweight='bold')

ax = axes[0]
assignments = comp._partition(X_train)
counts = [(assignments == j).sum() for j in range(comp.m)]
mol_lbl = ['CH3\n(k=3)' if j == 0 or j == comp.m-1 else 'CH2\n(k=2)'
           for j in range(comp.m)]
bar_c = [color_ahn, '#3498db', '#2ecc71'][:comp.m]
bars  = ax.bar([f'Mol {j+1}\n{mol_lbl[j]}' for j in range(comp.m)],
               counts, color=bar_c, alpha=0.85, edgecolor='white', lw=1.5)
for bar, n in zip(bars, counts):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
            str(n), ha='center', fontweight='bold')
ax.set_ylabel('Samples assigned (train)')
ax.set_title(f'Sigma_j Partitioning  (m={comp.m} molecules)')

ax = axes[1]
hist = comp.history
ax.plot(range(1, len(hist)+1), hist, '-o', color=color_ahn, lw=2,
        markersize=5, label='E_global = sum(E_j)')
ax.axhline(AHN_CONFIG['tolerance'], color='gray', ls='--', lw=1.5,
           label=f'eps = {AHN_CONFIG["tolerance"]}')
ax.set_xlabel('Iteration'); ax.set_ylabel('Global error')
ax.set_title('Convergence — Algorithm 1')
ax.legend()
if min(hist) > 0:
    ax.set_yscale('log')
plt.tight_layout()
plt.savefig(out('ahn_internal_structure.png'), dpi=300, bbox_inches='tight')
print("  Saved: ahn_internal_structure.png")
plt.close()


# ══════════════════════════════════════════════════════════════════════════════
#  BLOQUE 7 — ANALISIS ESTADISTICO Y RESUMEN
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("STATISTICAL ANALYSIS")
print("=" * 70)

print("\nRANKING BY METRIC:")
print("-" * 70)
for metric in ['Test Acc', 'F1-Score', 'AUC-PR', 'ROC-AUC']:
    ranked = comparison_df.sort_values(metric, ascending=False)
    print(f"\n{metric}:")
    for rank, (_, row) in enumerate(ranked.iterrows(), 1):
        print(f"  {rank}. {row['Model']:20s} {row[metric]:.4f}")

# Score ponderado (el control se excluye de la elección de "mejor modelo":
# no es un candidato real, es una referencia de degeneración de clase)
comparison_df['Overall Score'] = (
    0.3 * comparison_df['Test Acc'] +
    0.3 * comparison_df['F1-Score'] +
    0.4 * comparison_df['ROC-AUC']
)
_real_df = comparison_df[comparison_df['Type'] != 'Control']
best = _real_df.loc[_real_df['Overall Score'].idxmax()]

print("\n" + "=" * 70)
print("BEST OVERALL MODEL  (score = 0.3*Acc + 0.3*F1 + 0.4*AUC)")
print("=" * 70)
print(f"\n  Winner   : {best['Model']}")
print(f"  Overall  : {best['Overall Score']:.4f}")
print(f"  Accuracy : {best['Test Acc']:.4f}")
print(f"  F1-Score : {best['F1-Score']:.4f}")
print(f"  ROC-AUC  : {best['ROC-AUC']:.4f}")

b_df  = comparison_df[comparison_df['Type'] == 'Baseline']
b_acc = b_df['Test Acc'].mean()
b_f1  = b_df['F1-Score'].mean()
b_ap  = b_df['AUC-PR'].mean()
b_roc = b_df['ROC-AUC'].mean()

print("\n" + "=" * 70)
print("AHN vs BASELINE AVERAGE")
print("=" * 70)

print(f"\n  Accuracy : AHN {ahn_results['test_acc']:.4f}  vs  Baseline {b_acc:.4f}  "
      f"({'UP' if ahn_results['test_acc'] >= b_acc else 'DOWN'} {abs(ahn_results['test_acc']-b_acc):.4f})")
print(f"  F1-Score : AHN {ahn_results['f1']:.4f}  vs  Baseline {b_f1:.4f}  "
      f"({'UP' if ahn_results['f1'] >= b_f1 else 'DOWN'} {abs(ahn_results['f1']-b_f1):.4f})")
print(f"  AUC-PR   : AHN {ahn_results['avg_precision']:.4f}  vs  Baseline {b_ap:.4f}  "
      f"({'UP' if ahn_results['avg_precision'] >= b_ap else 'DOWN'} {abs(ahn_results['avg_precision']-b_ap):.4f})")
print(f"  ROC-AUC  : AHN {ahn_results['roc_auc']:.4f}  vs  Baseline {b_roc:.4f}  "
      f"({'UP' if ahn_results['roc_auc'] >= b_roc else 'DOWN'} {abs(ahn_results['roc_auc']-b_roc):.4f})")
print(f"\n  [Control All-Positive F1={ALL_POSITIVE_CONTROL['f1']:.4f}  "
      f"(Recall=1.0000 por construcción) — AHN F1={ahn_results['f1']:.4f} "
      f"está {'por encima' if ahn_results['f1'] > ALL_POSITIVE_CONTROL['f1'] else 'por debajo'} de esta referencia]")

# Guardar pickle
with open(out('ahn_comparison_data.pkl'), 'wb') as f:
    pickle.dump({'ahn': ahn, 'ahn_results': ahn_results,
                 'baseline_results': baseline_results,
                 'comparison_df': comparison_df}, f)

print("\n" + "=" * 70)
print("FULL ANALYSIS FINISHED")
print("=" * 70)
print(f"\nFiles generated in: {OUTPUT_DIR}")
print("  1. final_comparison_table.csv")
print("  2. final_radar_comparison.png")
print("  3. final_roc_curves_complete.png")
print("  4. final_metrics_comparison.png")
print("  5. ahn_confusion_matrices.png")
print("  6. ahn_internal_structure.png")
print("  7. ahn_comparison_data.pkl")
print("=" * 70)


# ══════════════════════════════════════════════════════════════════════════════
#  BLOQUE 8 — EXPERIMENTOS DE ROBUSTEZ: CONFIGURACIÓN COMÚN
# ══════════════════════════════════════════════════════════════════════════════

from sklearn.model_selection import train_test_split as _tts

_PALETTE = {
    'AHN': '#E74C3C',
    'SVM': '#3498DB',
    'Random Forest': '#2ECC71',
    'MLP': '#9B59B6',
}
_MARKERS = {'AHN': 'o', 'SVM': 's', 'Random Forest': '^', 'MLP': 'D'}
_LW      = {'AHN': 2.5, 'SVM': 1.8, 'Random Forest': 1.8, 'MLP': 1.8}
_MODELS  = ['AHN', 'SVM', 'Random Forest', 'MLP']
_METRICS = ['acc', 'precision', 'recall', 'f1', 'ap', 'auc']
_MLABELS = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-PR', 'ROC-AUC']

# Reviewer #2: repetir los tres escenarios de robustez con múltiples semillas y
# reportar bandas de varianza, en vez de una sola corrida — la oscilación de
# MLP en la Fig. 1 original no bastaba para concluir robustez con una corrida.
ROBUSTNESS_SEEDS = [0, 1, 2, 3, 4]

# NOTA: una versión anterior de este script entrenaba AHN con un presupuesto
# reducido (n_restarts=1, max_iterations=40, patience=10) específicamente en
# los tres barridos de robustez, documentado solo en la sección de limitaciones
# del paper. Eso introduce un confusor real: si AHN parte de un óptimo peor
# solo en estos experimentos, cualquier "robustez" observada podría deberse a
# que hay menos rendimiento que perder, no a una propiedad estructural del
# modelo — exactamente el tipo de artefacto que un revisor escéptico señalaría.
# Se eliminó esa reducción: AHN usa aquí la MISMA configuración completa que en
# el experimento principal (AHN_CONFIG, sin modificar), igual que los tres
# baselines nunca tuvieron su presupuesto de entrenamiento reducido en ningún
# experimento.

def _eval_all(X_tr, y_tr, X_te, y_te, seed=42):
    """Fits AHN + all three baselines with one shared seed (controls AHN's own
    randomness AND, via make_baselines(seed), every baseline's randomness), all
    calibrated on the fixed X_val/y_val, and returns metrics for all four."""
    results = {}
    _rng = np.random.default_rng(seed)

    cfg = {**AHN_CONFIG, 'random_state': seed}
    _ahn = AHNMixture(**cfg)
    _ahn.fit(X_tr, y_tr, verbose=False)
    _ahn.fit_platt(X_val, y_val)          # calibrated on the fixed validation set
    yp  = _ahn.predict(X_te)              # now uses the calibrated probability
    ypr = _ahn.predict_proba(X_te)[:, 1]
    results['AHN'] = {
        'acc':       accuracy_score(y_te, yp),
        'precision': precision_score(y_te, yp, zero_division=0),
        'recall':    recall_score(y_te, yp, zero_division=0),
        'f1':        f1_score(y_te, yp, zero_division=0),
        'ap':        average_precision_score(y_te, ypr),
        'auc':       roc_auc_score(y_te, ypr),
    }

    for name, model in make_baselines(seed).items():
        fit_baseline(name, model, X_tr, y_tr, _rng)
        # Same Platt protocol as AHN, fit on the same fixed validation set —
        # ensures every model in the sweep shares one calibration procedure
        # and one definition of the 0.5 operating point.
        cal = CalibratedBaseline(model).fit_platt(X_val, y_val)
        yp  = cal.predict(X_te)
        ypr = cal.predict_proba(X_te)[:, 1]
        results[name] = {
            'acc':       accuracy_score(y_te, yp),
            'precision': precision_score(y_te, yp, zero_division=0),
            'recall':    recall_score(y_te, yp, zero_division=0),
            'f1':        f1_score(y_te, yp, zero_division=0),
            'ap':        average_precision_score(y_te, ypr),
            'auc':       roc_auc_score(y_te, ypr),
        }
    return results

def _aggregate(df_raw, x_col):
    """Collapses a long-format (seed, x_col, model, *metrics) DataFrame into
    per-(model, x_col) mean and std across seeds."""
    agg = df_raw.groupby(['model', x_col])[_METRICS].agg(['mean', 'std']).reset_index()
    agg.columns = ['model', x_col] + [f'{met}_{stat}' for met, stat in agg.columns.tolist()[2:]]
    std_cols = [f'{met}_std' for met in _METRICS]
    agg[std_cols] = agg[std_cols].fillna(0.0)   # single-seed edge case
    return agg

def _plot_robustness(x_vals, agg_df, x_col, x_labels, xlabel, title, fname,
                     highlight_ref=None):
    """Plots mean ± std (error bars) across ROBUSTNESS_SEEDS for each metric,
    styled to match the paper's existing Figs. 1-3 (white background, light
    gridlines, serif font) rather than the darkgrid style used by the
    diagnostic-only plots elsewhere in this script — scoped locally via
    plt.style.context so it doesn't affect anything else.

    Font sizes are calibrated for the PRINTED size, not the on-screen size:
    this figure is natively ~24in wide, but a paper will scale it down to
    roughly 7in (double-column width) — a ~3.5x shrink. No title is baked
    into the image — it prints to console for the LaTeX \\caption{} instead,
    since a redundant in-image title just eats space a reader has to squint
    past to reach the panels. Only one shared y-axis label ('Score') is
    shown, on the leftmost panel — repeating it on all six panels was
    redundant with the panel titles directly above each one and stole space
    that's better spent making the titles themselves bigger.
    """
    print(f"  Figure caption reference: {title}  (mean ± std, {len(ROBUSTNESS_SEEDS)} seeds)")

    with plt.style.context('seaborn-v0_8-whitegrid'):
        plt.rcParams['font.family'] = 'serif'
        fig, axes = plt.subplots(1, 6, figsize=(24, 4.8), sharey=True)

        handles, labels = None, None
        for i, (ax, met, mlbl) in enumerate(zip(axes, _METRICS, _MLABELS)):
            for m in _MODELS:
                sub = agg_df[agg_df.model == m].set_index(x_col).loc[x_vals]
                means = sub[f'{met}_mean'].values
                stds  = sub[f'{met}_std'].values
                xs    = range(len(x_vals))
                # Error bars first, at a higher z-order, so caps always show
                # even when std is tiny — otherwise the marker (drawn on top
                # in a single errorbar() call) can fully hide a short cap,
                # which is exactly what happened under Label Noise, where
                # std is often smaller than the marker's own radius.
                ax.errorbar(xs, means, yerr=stds, fmt='none', ecolor=_PALETTE[m],
                            capsize=5, elinewidth=1.6, capthick=1.6,
                            alpha=0.9, zorder=5)
                ax.plot(xs, means, marker=_MARKERS[m], color=_PALETTE[m],
                        lw=_LW[m], markersize=7, label=m, alpha=0.9, zorder=3)
            if highlight_ref is not None:
                ax.axvline(highlight_ref, color='gray', lw=1.3, ls='--',
                           alpha=0.6, label='Clean ref.')
            if handles is None:
                handles, labels = ax.get_legend_handles_labels()   # collect once, shared legend below
            ax.set_xticks(range(len(x_vals)))
            _rot = 0 if any('\n' in lbl for lbl in x_labels) else 30
            ax.set_xticklabels(x_labels, fontsize=12, rotation=_rot,
                               ha=('center' if _rot == 0 else 'right'))
            ax.set_title(mlbl, fontsize=22, fontweight='bold', pad=10)
            ax.set_ylim([0.0, 1.05])
            ax.tick_params(axis='y', labelsize=14, labelleft=(i == 0))
            ax.grid(axis='y', alpha=0.3)

        axes[0].set_ylabel('Score', fontsize=18)
        fig.text(0.5, -0.02, xlabel, ha='center', fontsize=16)

        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.10),
                   ncol=len(labels), fontsize=17, frameon=False, columnspacing=1.8)
        plt.tight_layout(rect=[0, 0.02, 1, 0.92])
        plt.savefig(out(fname), dpi=300, bbox_inches='tight')
    print(f"  Saved: {fname}")
    plt.close()

def _print_table(x_vals, x_col_label, agg_df, x_labels, x_col):
    header = f"  {'':>18}" + "".join(f"  {lbl:>15}" for lbl in _MLABELS)
    sep    = "  " + "-" * (18 + 17 * len(_MLABELS))
    for xlbl, xv in zip(x_labels, x_vals):
        print(f"\n  {x_col_label}={xlbl}")
        print(sep); print(header); print(sep)
        for m in _MODELS:
            row = f"  {m:>18}"
            sub = agg_df[(agg_df.model == m) & (agg_df[x_col] == xv)]
            for met in _METRICS:
                row += f"  {sub[f'{met}_mean'].values[0]:.3f}±{sub[f'{met}_std'].values[0]:.3f}"
            print(row)


# ══════════════════════════════════════════════════════════════════════════════
#  BLOQUE 9 — EXP 1: DATA SCARCITY
# ══════════════════════════════════════════════════════════════════════════════

SCARCITY_FRACTIONS = [0.05, 0.10, 0.20, 0.30, 0.50, 0.75, 1.00]

print("\n" + "=" * 70)
print("EXP 1 — DATA SCARCITY  [Exoplanets KOI]")
print("=" * 70)
print("  Stratified subsampling of X_train  |  X_val, X_test fixed")
print(f"  Fractions: {[f'{f:.0%}' for f in SCARCITY_FRACTIONS]}")
print(f"  Seeds: {ROBUSTNESS_SEEDS}  (mean ± std over {len(ROBUSTNESS_SEEDS)} runs)\n")

sc_rows  = []
sc_sizes = {}

for frac in SCARCITY_FRACTIONS:
    n = max(int(len(X_train) * frac), 10)
    sc_sizes[frac] = n
    for seed in ROBUSTNESS_SEEDS:
        if frac < 1.0:
            idx, _ = _tts(np.arange(len(X_train)), train_size=n,
                          stratify=y_train, random_state=seed)
        else:
            idx = np.arange(len(X_train))
        X_sub, y_sub = X_train[idx], y_train[idx]
        res = _eval_all(X_sub, y_sub, X_test, y_test, seed=seed)
        for m in _MODELS:
            sc_rows.append({'model': m, 'fraction': frac, 'seed': seed, **res[m]})

    mean_auc = {m: np.mean([r['auc'] for r in sc_rows
                             if r['fraction'] == frac and r['model'] == m]) for m in _MODELS}
    print(f"  frac={frac:.0%}  n≈{sc_sizes[frac]:5d}  |  "
          + "  ".join(f"{m} AUC={mean_auc[m]:.3f}" for m in _MODELS))

sc_raw_df = pd.DataFrame(sc_rows)
sc_raw_df.to_csv(out('robustness_scarcity_raw.csv'), index=False)   # per-seed, for transparency
sc_df = _aggregate(sc_raw_df, 'fraction')
sc_df.to_csv(out('robustness_scarcity.csv'), index=False)           # mean ± std, used by plots/tables

_print_table(SCARCITY_FRACTIONS, 'frac', sc_df,
             [f"{f:.0%} (n={sc_sizes[f]})" for f in SCARCITY_FRACTIONS], 'fraction')

_plot_robustness(
    SCARCITY_FRACTIONS, sc_df, 'fraction',
    [f"{f:.0%}" for f in SCARCITY_FRACTIONS],
    'Training set fraction',
    f'EXP 1 — Data Scarcity [Exoplanets]: Metrics vs. training set size '
    f'(n={sc_sizes[0.05]} at 5% to n={sc_sizes[1.00]} at 100%)',
    'robustness_scarcity.png',
    highlight_ref=len(SCARCITY_FRACTIONS) - 1,
)
print("  Saved: robustness_scarcity.csv (aggregated)  /  robustness_scarcity_raw.csv (per seed)")


# ══════════════════════════════════════════════════════════════════════════════
#  BLOQUE 10 — EXP 2: FEATURE NOISE
# ══════════════════════════════════════════════════════════════════════════════

NOISE_SIGMAS = [0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0]

print("\n" + "=" * 70)
print("EXP 2 — FEATURE NOISE  [Exoplanets KOI]")
print("=" * 70)
print("  Train on clean X_train  |  Noise N(0,σ) added to X_test")
print(f"  σ: {NOISE_SIGMAS}  (scaled range [-1,1])")
print(f"  Seeds: {ROBUSTNESS_SEEDS}  (mean ± std over {len(ROBUSTNESS_SEEDS)} runs)\n")

fn_rows = []

for seed in ROBUSTNESS_SEEDS:
    # Entrenar y calibrar una sola vez por semilla (en limpio), luego evaluar
    # en todos los niveles de sigma — misma disciplina que antes, repetida
    # ahora sobre varias semillas para poder reportar media ± std.
    _rng = np.random.default_rng(seed)

    cfg_fn = {**AHN_CONFIG, 'random_state': seed}
    _ahn_fn = AHNMixture(**cfg_fn)
    _ahn_fn.fit(X_train, y_train, verbose=False)
    _ahn_fn.fit_platt(X_val, y_val)

    _bl_fn = {}
    for name, model in make_baselines(seed).items():
        fit_baseline(name, model, X_train, y_train, _rng)
        _bl_fn[name] = CalibratedBaseline(model).fit_platt(X_val, y_val)

    _rng_noise = np.random.default_rng(2000 + seed)
    for sigma in NOISE_SIGMAS:
        X_te_n = X_test + (_rng_noise.normal(0, sigma, X_test.shape) if sigma > 0 else 0)

        yp  = _ahn_fn.predict(X_te_n)
        ypr = _ahn_fn.predict_proba(X_te_n)[:, 1]
        fn_rows.append({'model': 'AHN', 'sigma': sigma, 'seed': seed,
            'acc':       accuracy_score(y_test, yp),
            'precision': precision_score(y_test, yp, zero_division=0),
            'recall':    recall_score(y_test, yp, zero_division=0),
            'f1':        f1_score(y_test, yp, zero_division=0),
            'ap':        average_precision_score(y_test, ypr),
            'auc':       roc_auc_score(y_test, ypr),
        })
        for name, cal in _bl_fn.items():
            yp  = cal.predict(X_te_n)
            ypr = cal.predict_proba(X_te_n)[:, 1]
            fn_rows.append({'model': name, 'sigma': sigma, 'seed': seed,
                'acc':       accuracy_score(y_test, yp),
                'precision': precision_score(y_test, yp, zero_division=0),
                'recall':    recall_score(y_test, yp, zero_division=0),
                'f1':        f1_score(y_test, yp, zero_division=0),
                'ap':        average_precision_score(y_test, ypr),
                'auc':       roc_auc_score(y_test, ypr),
            })

    mean_auc0 = {m: np.mean([r['auc'] for r in fn_rows
                              if r['sigma'] == 0.0 and r['seed'] == seed and r['model'] == m])
                 for m in _MODELS}
    print(f"  seed={seed}  σ=0.00  |  " + "  ".join(f"{m} AUC={mean_auc0[m]:.3f}" for m in _MODELS))

fn_raw_df = pd.DataFrame(fn_rows)
fn_raw_df.to_csv(out('robustness_feature_noise_raw.csv'), index=False)
fn_df = _aggregate(fn_raw_df, 'sigma')
fn_df.to_csv(out('robustness_feature_noise.csv'), index=False)

_print_table(NOISE_SIGMAS, 'σ', fn_df, [f"σ={s}" for s in NOISE_SIGMAS], 'sigma')

_plot_robustness(
    NOISE_SIGMAS, fn_df, 'sigma',
    [f"σ={s}" for s in NOISE_SIGMAS],
    'σ Gaussian noise (scale [-1,1])',
    'EXP 2 — Feature Noise [Exoplanets]: Metrics under increasing noise at inference',
    'robustness_feature_noise.png',
    highlight_ref=0,
)
print("  Saved: robustness_feature_noise.csv (aggregated)  /  robustness_feature_noise_raw.csv (per seed)")


# ══════════════════════════════════════════════════════════════════════════════
#  BLOQUE 10b — EXP 2b: FEATURE NOISE (MUESTREADO DESDE INCERTIDUMBRE REAL)
# ══════════════════════════════════════════════════════════════════════════════
#
# Reviewer #1 (punto de menor prioridad, no bloqueante): "the KOI table
# publishes per-object error columns. Sampling from those instead of a flat
# sigma would make the noise scenario a physical one."
#
# Diferencia frente al Exp. 2 (Bloque 10) — ver también RESUMEN DEL DATASET:
#   Exp. 2  (sigma plano):  ruido = N(0, sigma), IDÉNTICO para las 10 features
#                           y para todos los objetos del test set. sigma=1.0
#                           es "una unidad del rango escalado [-1,1]" — una
#                           cantidad arbitraria, sin relación con qué tan bien
#                           se midió cada parámetro para cada candidato.
#   Exp. 2b (físico):       ruido = N(0, sigma * U_ij), donde U_ij es la
#                           incertidumbre de medición REAL reportada por el
#                           pipeline de Kepler para el objeto i y la feature j
#                           (promedio de |err1| y |err2|, columnas publicadas
#                           por el NASA Exoplanet Archive), convertida al mismo
#                           espacio escalado [-1,1]. sigma sigue siendo el
#                           multiplicador de severidad (mismos valores de
#                           barrido que en Exp. 2, para comparabilidad directa
#                           del eje x), pero ahora la magnitud relativa del
#                           ruido entre features y entre objetos refleja qué
#                           tan bien fue medido cada uno, en vez de ser
#                           uniforme: un candidato con koi_period muy preciso
#                           recibe poco ruido en esa feature; uno con
#                           koi_depth mal determinado recibe más.
#   Para features sin columnas de error disponibles en este pull del archivo,
#   Exp. 2b recae automáticamente en el mismo ruido plano que Exp. 2 SOLO para
#   esa feature — nunca se inventa una incertidumbre. Con datos sintéticos
#   (sin columnas reales), Exp. 2b es idéntico a Exp. 2 por construcción.

print("\n" + "=" * 70)
print("EXP 2b — FEATURE NOISE (SAMPLED FROM REAL PER-OBJECT UNCERTAINTY)  [Exoplanets KOI]")
print("=" * 70)
if U_raw_all is not None:
    print(f"  Real uncertainty available for {sum(_feat_has_err_cols.values())}/{len(FEATURES)} features "
          f"(remaining features use flat sigma — see DATASET SUMMARY above)")
else:
    print(f"  ⚠  No real uncertainty columns available (source = {DATA_SOURCE}) — "
          f"this experiment is identical to Exp. 2 with this data source")
print(f"  σ (severity multiplier): {NOISE_SIGMAS}")
print(f"  Seeds: {ROBUSTNESS_SEEDS}  (mean ± std over {len(ROBUSTNESS_SEEDS)} runs)\n")

fn2_rows = []

for seed in ROBUSTNESS_SEEDS:
    cfg_fn2 = {**AHN_CONFIG, 'random_state': seed}
    _ahn_fn2 = AHNMixture(**cfg_fn2)
    _ahn_fn2.fit(X_train, y_train, verbose=False)
    _ahn_fn2.fit_platt(X_val, y_val)

    _bl_fn2   = {}
    _rng_fit2 = np.random.default_rng(seed)
    for name, model in make_baselines(seed).items():
        fit_baseline(name, model, X_train, y_train, _rng_fit2)
        _bl_fn2[name] = CalibratedBaseline(model).fit_platt(X_val, y_val)

    _rng_noise2 = np.random.default_rng(3000 + seed)
    for sigma in NOISE_SIGMAS:
        noise    = _rng_noise2.normal(0, 1, X_test.shape) * sigma * U_test_scaled if sigma > 0 else 0
        X_te_n2  = X_test + noise

        yp  = _ahn_fn2.predict(X_te_n2)
        ypr = _ahn_fn2.predict_proba(X_te_n2)[:, 1]
        fn2_rows.append({'model': 'AHN', 'sigma': sigma, 'seed': seed,
            'acc':       accuracy_score(y_test, yp),
            'precision': precision_score(y_test, yp, zero_division=0),
            'recall':    recall_score(y_test, yp, zero_division=0),
            'f1':        f1_score(y_test, yp, zero_division=0),
            'ap':        average_precision_score(y_test, ypr),
            'auc':       roc_auc_score(y_test, ypr),
        })
        for name, cal in _bl_fn2.items():
            yp  = cal.predict(X_te_n2)
            ypr = cal.predict_proba(X_te_n2)[:, 1]
            fn2_rows.append({'model': name, 'sigma': sigma, 'seed': seed,
                'acc':       accuracy_score(y_test, yp),
                'precision': precision_score(y_test, yp, zero_division=0),
                'recall':    recall_score(y_test, yp, zero_division=0),
                'f1':        f1_score(y_test, yp, zero_division=0),
                'ap':        average_precision_score(y_test, ypr),
                'auc':       roc_auc_score(y_test, ypr),
            })

    mean_auc0 = {m: np.mean([r['auc'] for r in fn2_rows
                              if r['sigma'] == 0.0 and r['seed'] == seed and r['model'] == m])
                 for m in _MODELS}
    print(f"  seed={seed}  σ=0.00  |  " + "  ".join(f"{m} AUC={mean_auc0[m]:.3f}" for m in _MODELS))

fn2_raw_df = pd.DataFrame(fn2_rows)
fn2_raw_df.to_csv(out('robustness_feature_noise_physical_raw.csv'), index=False)
fn2_df = _aggregate(fn2_raw_df, 'sigma')
fn2_df.to_csv(out('robustness_feature_noise_physical.csv'), index=False)

_print_table(NOISE_SIGMAS, 'σ', fn2_df, [f"σ={s}" for s in NOISE_SIGMAS], 'sigma')

_plot_robustness(
    NOISE_SIGMAS, fn2_df, 'sigma',
    [f"σ={s}" for s in NOISE_SIGMAS],
    'σ (multiplier over real per-object uncertainty)',
    'EXP 2b — Feature Noise (physical) [Exoplanets]: noise sampled from real measurement uncertainty',
    'robustness_feature_noise_physical.png',
    highlight_ref=0,
)
print("  Saved: robustness_feature_noise_physical.csv (aggregated)  /  robustness_feature_noise_physical_raw.csv (per seed)")

# ── Comparación directa: Exp. 2 (sigma plano) vs Exp. 2b (físico) ───────────
def _lookup(df, m, met, val):
    return df[(df.model == m) & (df.sigma == val)][f'{met}_mean'].values[0]

print(f"\n  Comparison at σ=1.0 (most severe point of the sweep)")
print(f"  {'Model':<16}  {'F1 flat':>9}  {'F1 physical':>10}  {'AUC-PR flat':>13}  {'AUC-PR physical':>14}")
print("  " + "-" * 72)
for m in _MODELS:
    f1_plano  = _lookup(fn_df,  m, 'f1', 1.0)
    f1_fisico = _lookup(fn2_df, m, 'f1', 1.0)
    ap_plano  = _lookup(fn_df,  m, 'ap', 1.0)
    ap_fisico = _lookup(fn2_df, m, 'ap', 1.0)
    print(f"  {m:<16}  {f1_plano:>9.4f}  {f1_fisico:>10.4f}  {ap_plano:>13.4f}  {ap_fisico:>14.4f}")


# ══════════════════════════════════════════════════════════════════════════════
#  BLOQUE 11 — EXP 3: LABEL NOISE
# ══════════════════════════════════════════════════════════════════════════════

FLIP_RATES = [0.00, 0.05, 0.10, 0.15, 0.20]

print("\n" + "=" * 70)
print("EXP 3 — LABEL NOISE  [Exoplanets KOI]")
print("=" * 70)
print("  Random flip of p% of y_train  |  y_test always clean")
print(f"  Rates: {[f'{p:.0%}' for p in FLIP_RATES]}")
print(f"  Seeds: {ROBUSTNESS_SEEDS}  (mean ± std over {len(ROBUSTNESS_SEEDS)} runs)\n")

ln_rows = []

for flip_rate in FLIP_RATES:
    for seed in ROBUSTNESS_SEEDS:
        _rng_ln = np.random.default_rng(1000 + seed)
        y_noisy = y_train.copy()
        if flip_rate > 0:
            flip_mask = _rng_ln.random(len(y_noisy)) < flip_rate
            y_noisy[flip_mask] = 1 - y_noisy[flip_mask]
        res = _eval_all(X_train, y_noisy, X_test, y_test, seed=seed)
        for m in _MODELS:
            ln_rows.append({'model': m, 'flip_rate': flip_rate, 'seed': seed, **res[m]})

    mean_auc = {m: np.mean([r['auc'] for r in ln_rows
                             if r['flip_rate'] == flip_rate and r['model'] == m]) for m in _MODELS}
    print(f"  flip={flip_rate:.0%}  |  " + "  ".join(f"{m} AUC={mean_auc[m]:.3f}" for m in _MODELS))

ln_raw_df = pd.DataFrame(ln_rows)
ln_raw_df.to_csv(out('robustness_label_noise_raw.csv'), index=False)
ln_df = _aggregate(ln_raw_df, 'flip_rate')
ln_df.to_csv(out('robustness_label_noise.csv'), index=False)

_print_table(FLIP_RATES, 'flip', ln_df, [f"{p:.0%}" for p in FLIP_RATES], 'flip_rate')

_plot_robustness(
    FLIP_RATES, ln_df, 'flip_rate',
    [f"{p:.0%}" for p in FLIP_RATES],
    'Label flip rate (train)',
    'EXP 3 — Label Noise [Exoplanets]: Metrics under corrupted training labels',
    'robustness_label_noise.png',
    highlight_ref=0,
)
print("  Saved: robustness_label_noise.csv (aggregated)  /  robustness_label_noise_raw.csv (per seed)")


# ══════════════════════════════════════════════════════════════════════════════
#  BLOQUE 12 — RESUMEN DE ROBUSTEZ
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("ROBUSTNESS SUMMARY — AUC drop (clean condition → most extreme)")
print("=" * 70)

def _get(df, m, met, x_col, x_val):
    return df[(df.model == m) & (df[x_col] == x_val)][f'{met}_mean'].values[0]

def _get_std(df, m, met, x_col, x_val):
    return df[(df.model == m) & (df[x_col] == x_val)][f'{met}_std'].values[0]

print(f"\n  {'Experiment':<24}  {'Model':<16}  {'Clean AUC':>10}  {'Extreme AUC (±seed std)':>28}  {'ΔAUC':>7}"
      f"  {'Clean AP':>9}  {'Extreme AP (±seed std)':>27}  {'ΔAP':>7}")
print("  " + "-" * 130)

for m in _MODELS:
    a_c     = _get(sc_df, m, 'auc', 'fraction', 1.00)
    a_e     = _get(sc_df, m, 'auc', 'fraction', 0.05)
    a_e_std = _get_std(sc_df, m, 'auc', 'fraction', 0.05)
    p_c     = _get(sc_df, m, 'ap', 'fraction', 1.00)
    p_e     = _get(sc_df, m, 'ap', 'fraction', 0.05)
    p_e_std = _get_std(sc_df, m, 'ap', 'fraction', 0.05)
    print(f"  {'Scarcity (5%→100%)':<24}  {m:<16}  {a_c:>10.4f}  {a_e:.4f} ± {a_e_std:<20.4f}  {a_e-a_c:>+7.4f}"
          f"  {p_c:>9.4f}  {p_e:.4f} ± {p_e_std:<19.4f}  {p_e-p_c:>+7.4f}")

print()
for m in _MODELS:
    a_c     = _get(fn_df, m, 'auc', 'sigma', 0.0)
    a_e     = _get(fn_df, m, 'auc', 'sigma', 1.0)
    a_e_std = _get_std(fn_df, m, 'auc', 'sigma', 1.0)
    p_c     = _get(fn_df, m, 'ap', 'sigma', 0.0)
    p_e     = _get(fn_df, m, 'ap', 'sigma', 1.0)
    p_e_std = _get_std(fn_df, m, 'ap', 'sigma', 1.0)
    print(f"  {'Feature Noise (σ=1.0)':<24}  {m:<16}  {a_c:>10.4f}  {a_e:.4f} ± {a_e_std:<20.4f}  {a_e-a_c:>+7.4f}"
          f"  {p_c:>9.4f}  {p_e:.4f} ± {p_e_std:<19.4f}  {p_e-p_c:>+7.4f}")

print()
for m in _MODELS:
    a_c     = _get(ln_df, m, 'auc', 'flip_rate', 0.00)
    a_e     = _get(ln_df, m, 'auc', 'flip_rate', 0.20)
    a_e_std = _get_std(ln_df, m, 'auc', 'flip_rate', 0.20)
    p_c     = _get(ln_df, m, 'ap', 'flip_rate', 0.00)
    p_e     = _get(ln_df, m, 'ap', 'flip_rate', 0.20)
    p_e_std = _get_std(ln_df, m, 'ap', 'flip_rate', 0.20)
    print(f"  {'Label Noise (20% flip)':<24}  {m:<16}  {a_c:>10.4f}  {a_e:.4f} ± {a_e_std:<20.4f}  {a_e-a_c:>+7.4f}"
          f"  {p_c:>9.4f}  {p_e:.4f} ± {p_e_std:<19.4f}  {p_e-p_c:>+7.4f}")

print("\n" + "=" * 70)
print("ROBUSTNESS EXPERIMENTS FINISHED  [Exoplanets KOI]")
print("=" * 70)
print(f"\nAdditional files in: {OUTPUT_DIR}")
print("   8. robustness_scarcity.png  / robustness_scarcity.csv")
print("   9. robustness_feature_noise.png  / robustness_feature_noise.csv")
print("  10. robustness_label_noise.png  / robustness_label_noise.csv")
print("=" * 70)
