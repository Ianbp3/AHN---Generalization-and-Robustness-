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
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import warnings
import os
import pickle

warnings.filterwarnings('ignore')

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ahn_outputs")
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
        self.patience      = patience
        self.rng           = np.random.default_rng(random_state)

        if   n_molecules == 1: self.k_orders = [3]
        elif n_molecules == 2: self.k_orders = [3, 3]
        else:                  self.k_orders = [3] + [2] * (n_molecules - 2) + [3]

        self.molecules = []
        self.L = self.r = self.L_min = self.L_max = self.centers = None
        self.history = []

    # ── Inicialización de bounds ───────────────────────────────────────────────

    def _init_bounds(self, X):
        self.L_min = X.min(axis=0)
        self.L_max = X.max(axis=0)

        if self.m > 1:
            X_c = X - X.mean(axis=0)

            try:
                _, _, Vt = np.linalg.svd(X_c, full_matrices=False)
                self._chain_axis = Vt[0]
            except np.linalg.LinAlgError:
                self._chain_axis = np.zeros(self.n_feat)
                self._chain_axis[np.argmax(X.var(axis=0))] = 1.0

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

            if E_global < best_E:
                best_E    = E_global
                best_snap = self._snapshot()
                no_improve = 0
            else:
                no_improve += 1

            if verbose and (it % 10 == 0 or it < 3):
                sizes = [(assignments == j).sum() for j in range(self.m)]
                star  = '*' if E_global == best_E else ' '
                print(f"  Iter {it+1:3d}{star}| E_global={E_global:.6f} | particiones={sizes}")

            if E_global <= self.epsilon:
                if verbose:
                    print(f"  Convergido en iter {it+1}  (E={E_global:.6f} <= {self.epsilon})")
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
                        print(f"  ↺ Reinit bounds en iter {it+1}  (sin mejora por {self.patience} iters)")

        self._restore(best_snap)
        self.best_E_ = best_E
        if verbose:
            print(f"  ✓ Best-state restaurado  (E_best={best_E:.6f})")
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
                    print(f"\n  Compuesto {i+1}/{self.c}  (CH3-{mid}CH3):")
                seed = self.rs + i + restart * self.c
                comp = self._make_compound(seed)
                comp.fit(X, y, verbose=verbose)
                E_total += getattr(comp, 'best_E_', np.inf)
                compounds_try.append(comp)

            if E_total < best_E_total:
                best_E_total   = E_total
                best_compounds = compounds_try
                if verbose and self.n_restarts > 1:
                    print(f"  ★ Nuevo mejor  E={E_total:.6f}  (restart {restart+1})")

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
        return (self.predict_raw(X) >= self.threshold).astype(int)


# ══════════════════════════════════════════════════════════════════════════════
#  BLOQUE 2 — CARGAR Y PREPARAR DATOS
# ══════════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("COMPARACION COMPLETA: AHN vs Baseline Models")
print("=" * 70)

try:
    df = pd.read_csv(
        "hf://datasets/inGeniia/german-credit-risk_credit-scoring_mlp/"
        "german_credit_risk.csv"
    )
    print("Dataset cargado desde HuggingFace")
    # Preparar datos
    y       = df["Risk"].map({"good": 0, "bad": 1}).values
    X_raw   = df.drop(columns=["Risk"])
    X_array = pd.get_dummies(X_raw, drop_first=True).values.astype(float)
    DATA_SOURCE = "German Credit Risk (HuggingFace)"
except Exception:
    try:
        df = pd.read_csv("german_credit_risk.csv")
        print("Dataset cargado desde archivo local")
        y       = df["Risk"].map({"good": 0, "bad": 1}).values
        X_raw   = df.drop(columns=["Risk"])
        X_array = pd.get_dummies(X_raw, drop_first=True).values.astype(float)
        DATA_SOURCE = "German Credit Risk (local)"
    except Exception:
        print("⚠  Dataset real no disponible — usando datos sintéticos que replican")
        print("   las características de German Credit Risk:")
        print("   1000 muestras · 48 features · desbalance 70/30 · 5 clústeres/clase")
        from sklearn.datasets import make_classification
        X_array, y = make_classification(
            n_samples=1000, n_features=48, n_informative=12, n_redundant=10,
            n_clusters_per_class=3, weights=[0.7, 0.3],
            flip_y=0.05, random_state=42
        )
        DATA_SOURCE = "Sintético (réplica GCR)"

X_temp, X_test,  y_temp, y_test  = train_test_split(
    X_array, y, test_size=0.2, stratify=y, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp,  y_temp, test_size=0.25, stratify=y_temp, random_state=42
)

scaler  = MinMaxScaler(feature_range=(-1, 1))
X_train = scaler.fit_transform(X_train)
X_val   = scaler.transform(X_val)
X_test  = scaler.transform(X_test)

print(f"Features: {X_train.shape[1]}")
print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
print(f"Balance test — Good: {(y_test==0).sum()}, Bad: {(y_test==1).sum()}")


# ══════════════════════════════════════════════════════════════════════════════
#  BLOQUE 3 — ENTRENAR AHN
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("Entrenando AHN (Artificial Hydrocarbon Networks)...")
print("=" * 70)

AHN_CONFIG = dict(
    n_compounds    = 1,
    n_molecules    = 2,
    learning_rate  = 0.3,   # eta
    tolerance      = 0.1,  # epsilon
    max_iterations = 500,
    random_state   = 42,
    use_bias       = True,
    use_bce        = False,
    threshold      = 0.5,
    patience       = 20,    # iters sin mejora antes de reinit bounds
    n_restarts     = 3,     # 3 restarts → queda con el de menor E_best
)
m = AHN_CONFIG['n_molecules']
if   m == 1: _struct = "CH3"
elif m == 2: _struct = "CH3-CH3"
else:        _struct = "CH3-" + "CH2-" * (m - 2) + "CH3"
print(f"Estructura: {_struct}  |  eta={AHN_CONFIG['learning_rate']}  "
      f"|  eps={AHN_CONFIG['tolerance']}  |  max_iter={AHN_CONFIG['max_iterations']}  "
      f"|  bias={AHN_CONFIG['use_bias']}  |  bce={AHN_CONFIG['use_bce']}")

ahn = AHNMixture(**AHN_CONFIG)
ahn.fit(X_train, y_train, verbose=True)

ahn.fit_platt(X_val, y_val)
print(f"Platt Scaling ajustado:  a={ahn.platt_a:.4f}  b={ahn.platt_b:.4f}")

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
    'y_pred':    y_pred_ahn,
    'y_proba':   y_proba_ahn,
    'confusion_matrix': confusion_matrix(y_test, y_pred_ahn),
}
print(f"\nAHN: Acc={ahn_results['test_acc']:.4f}, "
      f"F1={ahn_results['f1']:.4f}, "
      f"ROC-AUC={ahn_results['roc_auc']:.4f}")


# ── BLOQUE 3b — K-FOLD CROSS VALIDATION  ─────────────────────────────────────
# Evalúa la estabilidad del modelo sobre distintos splits de X_train.
# En cada fold:
#   1. AHN.fit()       → train fold  (K-1 partes)
#   2. fit_platt()     → 20% interno del train fold  (anti-leakage estricto)
#   3. Evaluación      → held-out fold
# El modelo final (entrenado arriba sobre X_train completo) no cambia.
# ─────────────────────────────────────────────────────────────────────────────

from sklearn.model_selection import StratifiedKFold

K_FOLDS  = 5
kf       = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
X_cv     = np.vstack([X_train, X_val])   # usamos todo lo que no es test
y_cv     = np.concatenate([y_train, y_val])

cv_aucs, cv_accs, cv_f1s = [], [], []

print(f"\n{'─'*70}")
print(f"K-FOLD CROSS VALIDATION  (K={K_FOLDS}, StratifiedKFold, datos=train+val)")
print(f"{'─'*70}")
print(f"  {'Fold':>5}  {'AUC':>7}  {'ACC':>7}  {'F1':>7}  Platt (a, b)")

for fold, (tr_idx, val_idx) in enumerate(kf.split(X_cv, y_cv), 1):
    X_tr, X_ho = X_cv[tr_idx], X_cv[val_idx]
    y_tr, y_ho = y_cv[tr_idx], y_cv[val_idx]

    # Split interno para Platt — 20% del train fold, nunca del held-out
    n_platt      = max(20, int(0.20 * len(X_tr)))
    X_platt      = X_tr[-n_platt:]
    y_platt      = y_tr[-n_platt:]
    X_tr_pure    = X_tr[:-n_platt]
    y_tr_pure    = y_tr[:-n_platt]

    cfg_fold = {k: v for k, v in AHN_CONFIG.items()}
    cfg_fold['random_state'] = 42 + fold
    m = AHNMixture(**cfg_fold)
    m.fit(X_tr_pure, y_tr_pure, verbose=False)
    m.fit_platt(X_platt, y_platt)

    yp   = m.predict(X_ho)
    ypr  = m.predict_proba(X_ho)[:, 1]
    auc  = roc_auc_score(y_ho, ypr)
    acc  = accuracy_score(y_ho, yp)
    f1   = f1_score(y_ho, yp, zero_division=0)
    cv_aucs.append(auc); cv_accs.append(acc); cv_f1s.append(f1)
    print(f"  {fold:>5}  {auc:.4f}   {acc:.4f}   {f1:.4f}   "
          f"a={m.platt_a:.3f}  b={m.platt_b:.3f}")

print(f"{'─'*70}")
print(f"  {'Media':>5}  {np.mean(cv_aucs):.4f}   {np.mean(cv_accs):.4f}   {np.mean(cv_f1s):.4f}")
print(f"  {'±std':>5}  {np.std(cv_aucs):.4f}   {np.std(cv_accs):.4f}   {np.std(cv_f1s):.4f}")
print(f"  [Test real AHN final: "
      f"AUC={ahn_results['roc_auc']:.4f}  "
      f"ACC={ahn_results['test_acc']:.4f}  "
      f"F1={ahn_results['f1']:.4f}]")


# ══════════════════════════════════════════════════════════════════════════════
#  BLOQUE 4 — ENTRENAR MODELOS BASELINE
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("Entrenando modelos baseline...")
print("=" * 70)

baseline_models = {
    'SVM': SVC(kernel='rbf', probability=True, random_state=42),
    'Random Forest': RandomForestClassifier(
        n_estimators=100, max_depth=10, min_samples_split=5, random_state=42
    ),
    'MLP': MLPClassifier(
        hidden_layer_sizes=(64, 32), activation='relu', solver='adam',
        alpha=0.001, learning_rate_init=0.001, max_iter=300,
        early_stopping=True, validation_fraction=0.15, random_state=42
    ),
}

baseline_results = {}
for name, model in baseline_models.items():
    model.fit(X_train, y_train)
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    baseline_results[name] = {
        'train_acc': accuracy_score(y_train, model.predict(X_train)),
        'val_acc':   accuracy_score(y_val,   model.predict(X_val)),
        'test_acc':  accuracy_score(y_test,  y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall':    recall_score(y_test,    y_pred, zero_division=0),
        'f1':        f1_score(y_test,        y_pred, zero_division=0),
        'roc_auc':   roc_auc_score(y_test,   y_proba),
        'y_pred':    y_pred,
        'y_proba':   y_proba,
        'confusion_matrix': confusion_matrix(y_test, y_pred),
    }
    print(f"  {name}: Acc={baseline_results[name]['test_acc']:.4f}, "
          f"F1={baseline_results[name]['f1']:.4f}, "
          f"ROC-AUC={baseline_results[name]['roc_auc']:.4f}")


# ══════════════════════════════════════════════════════════════════════════════
#  BLOQUE 5 — TABLA COMPARATIVA
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("TABLA COMPARATIVA COMPLETA")
print("=" * 70)

rows = []
for name, res in baseline_results.items():
    rows.append({'Model': name, 'Type': 'Baseline',
                 'Train Acc': res['train_acc'], 'Val Acc': res['val_acc'],
                 'Test Acc':  res['test_acc'],  'Precision': res['precision'],
                 'Recall':    res['recall'],    'F1-Score':  res['f1'],
                 'ROC-AUC':   res['roc_auc']})
rows.append({'Model': 'AHN', 'Type': 'AHN',
             'Train Acc': ahn_results['train_acc'], 'Val Acc': ahn_results['val_acc'],
             'Test Acc':  ahn_results['test_acc'],  'Precision': ahn_results['precision'],
             'Recall':    ahn_results['recall'],    'F1-Score':  ahn_results['f1'],
             'ROC-AUC':   ahn_results['roc_auc']})

comparison_df = (pd.DataFrame(rows)
                   .sort_values('ROC-AUC', ascending=False)
                   .reset_index(drop=True))

print(comparison_df.drop(columns='Type').to_string(index=False))
comparison_df.to_csv(out('final_comparison_table.csv'), index=False)
print("\nTabla guardada: final_comparison_table.csv")

# Classification reports
print("\n" + "=" * 70)
print("CLASSIFICATION REPORTS")
print("=" * 70)
for name, res in {'AHN': ahn_results, **baseline_results}.items():
    print(f"\n{name}:")
    print(classification_report(y_test, res['y_pred'],
                                target_names=['Good', 'Bad'], digits=4))


# ══════════════════════════════════════════════════════════════════════════════
#  BLOQUE 6 — VISUALIZACIONES
# ══════════════════════════════════════════════════════════════════════════════

print("\nGenerando visualizaciones...")

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
plt.title('Radar Chart: Comparación de Todas las Métricas',
          size=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(out('final_radar_comparison.png'), dpi=300, bbox_inches='tight')
print("  Guardado: final_radar_comparison.png")
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
print("  Guardado: final_roc_curves_complete.png")
plt.close()

# ── FIGURA 3: Barras comparativas ─────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(16, 11))
fig.suptitle('Comparación Cuantitativa: AHN vs Baseline Models',
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
print("  Guardado: final_metrics_comparison.png")
plt.close()

# ── FIGURA 4: Matrices de confusion ──────────────────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))
fig.suptitle('Matrices de Confusion — AHN vs Baseline', fontsize=14, fontweight='bold')

plot_items = [('AHN', ahn_results, color_ahn)] + \
             [(n, r, c) for (n, r), c in zip(baseline_results.items(), colors_baseline)]

for ax, (name, res, col) in zip(axes, plot_items):
    sns.heatmap(res['confusion_matrix'], annot=True, fmt='d', cmap='Blues',
                ax=ax, cbar=False,
                xticklabels=['Good', 'Bad'], yticklabels=['Good', 'Bad'])
    marker = '* ' if name == 'AHN' else ''
    ax.set_title(f'{marker}{name}\nAcc={res["test_acc"]:.4f}  F1={res["f1"]:.4f}',
                 fontsize=10, fontweight='bold', color=col)
    ax.set_xlabel('Predicho')
    ax.set_ylabel('Real' if name == 'AHN' else '')

plt.tight_layout()
plt.savefig(out('ahn_confusion_matrices.png'), dpi=300, bbox_inches='tight')
print("  Guardado: ahn_confusion_matrices.png")
plt.close()

# ── FIGURA 5: Convergencia AHN ────────────────────────────────────────────────
comp = ahn.compounds[0]
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Estructura Interna del Compuesto AHN', fontsize=13, fontweight='bold')

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
ax.set_ylabel('Muestras asignadas (train)')
ax.set_title(f'Particionamiento Sigma_j  (m={comp.m} moleculas)')

ax = axes[1]
hist = comp.history
ax.plot(range(1, len(hist)+1), hist, '-o', color=color_ahn, lw=2,
        markersize=5, label='E_global = sum(E_j)')
ax.axhline(AHN_CONFIG['tolerance'], color='gray', ls='--', lw=1.5,
           label=f'eps = {AHN_CONFIG["tolerance"]}')
ax.set_xlabel('Iteracion'); ax.set_ylabel('Error global')
ax.set_title('Convergencia — Algoritmo 1')
ax.legend()
if min(hist) > 0:
    ax.set_yscale('log')
plt.tight_layout()
plt.savefig(out('ahn_internal_structure.png'), dpi=300, bbox_inches='tight')
print("  Guardado: ahn_internal_structure.png")
plt.close()


# ══════════════════════════════════════════════════════════════════════════════
#  BLOQUE 7 — ANALISIS ESTADISTICO Y RESUMEN
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("ANALISIS ESTADISTICO")
print("=" * 70)

print("\nRANKING POR METRICA:")
print("-" * 70)
for metric in ['Test Acc', 'F1-Score', 'ROC-AUC']:
    ranked = comparison_df.sort_values(metric, ascending=False)
    print(f"\n{metric}:")
    for rank, (_, row) in enumerate(ranked.iterrows(), 1):
        print(f"  {rank}. {row['Model']:20s} {row[metric]:.4f}")

# Score ponderado
comparison_df['Overall Score'] = (
    0.3 * comparison_df['Test Acc'] +
    0.3 * comparison_df['F1-Score'] +
    0.4 * comparison_df['ROC-AUC']
)
best = comparison_df.loc[comparison_df['Overall Score'].idxmax()]

print("\n" + "=" * 70)
print("MEJOR MODELO GENERAL  (score = 0.3*Acc + 0.3*F1 + 0.4*AUC)")
print("=" * 70)
print(f"\n  Ganador  : {best['Model']}")
print(f"  Overall  : {best['Overall Score']:.4f}")
print(f"  Accuracy : {best['Test Acc']:.4f}")
print(f"  F1-Score : {best['F1-Score']:.4f}")
print(f"  ROC-AUC  : {best['ROC-AUC']:.4f}")

b_df  = comparison_df[comparison_df['Type'] == 'Baseline']
b_acc = b_df['Test Acc'].mean()
b_f1  = b_df['F1-Score'].mean()
b_roc = b_df['ROC-AUC'].mean()

print("\n" + "=" * 70)
print("AHN vs BASELINE PROMEDIO")
print("=" * 70)

print(f"\n  Accuracy : AHN {ahn_results['test_acc']:.4f}  vs  Baseline {b_acc:.4f}  "
      f"({'UP' if ahn_results['test_acc'] >= b_acc else 'DOWN'} {abs(ahn_results['test_acc']-b_acc):.4f})")
print(f"  F1-Score : AHN {ahn_results['f1']:.4f}  vs  Baseline {b_f1:.4f}  "
      f"({'UP' if ahn_results['f1'] >= b_f1 else 'DOWN'} {abs(ahn_results['f1']-b_f1):.4f})")
print(f"  ROC-AUC  : AHN {ahn_results['roc_auc']:.4f}  vs  Baseline {b_roc:.4f}  "
      f"({'UP' if ahn_results['roc_auc'] >= b_roc else 'DOWN'} {abs(ahn_results['roc_auc']-b_roc):.4f})")

# Guardar pickle
with open(out('ahn_comparison_data.pkl'), 'wb') as f:
    pickle.dump({'ahn': ahn, 'ahn_results': ahn_results,
                 'baseline_results': baseline_results,
                 'comparison_df': comparison_df}, f)

print("\n" + "=" * 70)
print("ANALISIS COMPLETO FINALIZADO")
print("=" * 70)
print(f"\nArchivos generados en: {OUTPUT_DIR}")
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
_METRICS = ['acc', 'precision', 'recall', 'f1', 'auc']
_MLABELS = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']

_AHN_SWEEP = {k: v for k, v in AHN_CONFIG.items()}
_AHN_SWEEP['n_restarts']    = 1
_AHN_SWEEP['max_iterations'] = 40
_AHN_SWEEP['patience']      = 10

def _fresh_baselines():
    return {
        'SVM': SVC(kernel='rbf', probability=True, random_state=42),
        'Random Forest': RandomForestClassifier(
            n_estimators=100, max_depth=10, min_samples_split=5, random_state=42),
        'MLP': MLPClassifier(
            hidden_layer_sizes=(64, 32), activation='relu', solver='adam',
            alpha=0.001, learning_rate_init=0.001, max_iter=300,
            early_stopping=True, validation_fraction=0.15, random_state=42),
    }

def _eval_all(X_tr, y_tr, X_te, y_te, ahn_seed=42):
    results = {}

    # AHN
    cfg = {**_AHN_SWEEP, 'random_state': ahn_seed}
    _ahn = AHNMixture(**cfg)
    _ahn.fit(X_tr, y_tr)
    _ahn.fit_platt(X_val, y_val)
    yp  = _ahn.predict(X_te)
    ypr = _ahn.predict_proba(X_te)[:, 1]
    results['AHN'] = {
        'acc':       accuracy_score(y_te, yp),
        'precision': precision_score(y_te, yp, zero_division=0),
        'recall':    recall_score(y_te, yp, zero_division=0),
        'f1':        f1_score(y_te, yp, zero_division=0),
        'auc':       roc_auc_score(y_te, ypr),
    }

    # Baselines
    for name, model in _fresh_baselines().items():
        model.fit(X_tr, y_tr)
        yp  = model.predict(X_te)
        ypr = model.predict_proba(X_te)[:, 1]
        results[name] = {
            'acc':       accuracy_score(y_te, yp),
            'precision': precision_score(y_te, yp, zero_division=0),
            'recall':    recall_score(y_te, yp, zero_division=0),
            'f1':        f1_score(y_te, yp, zero_division=0),
            'auc':       roc_auc_score(y_te, ypr),
        }
    return results

def _plot_robustness(x_vals, all_results, x_labels, xlabel, title, fname,
                     highlight_ref=None):
    fig, axes = plt.subplots(1, 5, figsize=(22, 5))
    fig.suptitle(title, fontsize=13, fontweight='bold', y=1.01)

    for ax, met, mlbl in zip(axes, _METRICS, _MLABELS):
        for m in _MODELS:
            vals = [r[m][met] for r in all_results]
            ax.plot(range(len(x_vals)), vals,
                    marker=_MARKERS[m], color=_PALETTE[m],
                    lw=_LW[m], markersize=7, label=m, alpha=0.9)

        if highlight_ref is not None:
            ax.axvline(highlight_ref, color='gray', lw=1.2, ls='--', alpha=0.6)

        ax.set_xticks(range(len(x_vals)))
        ax.set_xticklabels(x_labels, fontsize=8, rotation=30, ha='right')
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel(mlbl, fontsize=9)
        ax.set_title(mlbl, fontsize=10, fontweight='bold')
        ax.set_ylim([0.0, 1.05])
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(out(fname), dpi=300, bbox_inches='tight')
    print(f"  Guardado: {fname}")
    plt.close()

def _print_table(x_vals, x_col, all_results, x_labels):
    header = f"  {'':>18}" + "".join(f"  {lbl:>9}" for lbl in _MLABELS)
    sep    = "  " + "-" * (18 + 11 * len(_MLABELS))
    for xi, (xv, xlbl, res) in enumerate(zip(x_vals, x_labels, all_results)):
        print(f"\n  {x_col}={xlbl}")
        print(sep)
        print(header)
        print(sep)
        for m in _MODELS:
            row = f"  {m:>18}"
            for met in _METRICS:
                row += f"  {res[m][met]:>9.4f}"
            print(row)


# ══════════════════════════════════════════════════════════════════════════════
#  BLOQUE 9 — EXP 1: DATA SCARCITY
# ══════════════════════════════════════════════════════════════════════════════

SCARCITY_FRACTIONS = [0.05, 0.10, 0.20, 0.30, 0.50, 0.75, 1.00]

print("\n" + "=" * 70)
print("EXP 1 — DATA SCARCITY")
print("=" * 70)
print("  Submuestreo estratificado de X_train  |  X_val, X_test fijos")
print(f"  Fracciones: {[f'{f:.0%}' for f in SCARCITY_FRACTIONS]}\n")

sc_results  = []
sc_labels   = []
sc_sizes    = []

for frac in SCARCITY_FRACTIONS:
    n = max(int(len(X_train) * frac), 10)
    if frac < 1.0:
        idx, _ = _tts(np.arange(len(X_train)), train_size=n,
                      stratify=y_train, random_state=42)
    else:
        idx = np.arange(len(X_train))

    X_sub, y_sub = X_train[idx], y_train[idx]
    sc_sizes.append(len(X_sub))
    res = _eval_all(X_sub, y_sub, X_test, y_test)
    sc_results.append(res)

    lbl = f"{frac:.0%}\n(n={len(X_sub)})"
    sc_labels.append(lbl)
    print(f"  frac={frac:.0%}  n={len(X_sub):4d}  |  "
          + "  ".join(f"{m} AUC={res[m]['auc']:.3f}" for m in _MODELS))

_print_table(SCARCITY_FRACTIONS, 'frac', sc_results,
             [f"{f:.0%} (n={n})" for f, n in zip(SCARCITY_FRACTIONS, sc_sizes)])

_plot_robustness(
    SCARCITY_FRACTIONS, sc_results,
    [f"{f:.0%}\n(n={n})" for f, n in zip(SCARCITY_FRACTIONS, sc_sizes)],
    xlabel="Fracción del train set",
    title="EXP 1 — Data Scarcity: Comportamiento de métricas según tamaño del train set",
    fname="robustness_scarcity.png",
    highlight_ref=len(SCARCITY_FRACTIONS) - 1,
)

pd.DataFrame([
    {'model': m, 'fraction': f, 'n_train': n,
     **{met: sc_results[i][m][met] for met in _METRICS}}
    for i, (f, n) in enumerate(zip(SCARCITY_FRACTIONS, sc_sizes))
    for m in _MODELS
]).to_csv(out('robustness_scarcity.csv'), index=False)
print("  Guardado: robustness_scarcity.csv")


# ══════════════════════════════════════════════════════════════════════════════
#  BLOQUE 10 — EXP 2: FEATURE NOISE
# ══════════════════════════════════════════════════════════════════════════════

NOISE_SIGMAS = [0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0]

print("\n" + "=" * 70)
print("EXP 2 — FEATURE NOISE")
print("=" * 70)
print("  Entrenar con X_train limpio  |  Ruido N(0,σ) añadido a X_test")
print(f"  σ: {NOISE_SIGMAS}  (escala escalada [-1,1])\n")

_ahn_fn = AHNMixture(**{**_AHN_SWEEP})
_ahn_fn.fit(X_train, y_train)
_ahn_fn.fit_platt(X_val, y_val)

_bl_fn = _fresh_baselines()
for _m in _bl_fn.values():
    _m.fit(X_train, y_train)

fn_results = []
fn_labels  = []
_rng_fn    = np.random.default_rng(0)

for sigma in NOISE_SIGMAS:
    X_te_n = X_test + (_rng_fn.normal(0, sigma, X_test.shape) if sigma > 0 else 0)

    res = {}
    # AHN
    yp  = _ahn_fn.predict(X_te_n)
    ypr = _ahn_fn.predict_proba(X_te_n)[:, 1]
    res['AHN'] = {
        'acc':       accuracy_score(y_test, yp),
        'precision': precision_score(y_test, yp, zero_division=0),
        'recall':    recall_score(y_test, yp, zero_division=0),
        'f1':        f1_score(y_test, yp, zero_division=0),
        'auc':       roc_auc_score(y_test, ypr),
    }
    # Baselines
    for name, model in _bl_fn.items():
        yp  = model.predict(X_te_n)
        ypr = model.predict_proba(X_te_n)[:, 1]
        res[name] = {
            'acc':       accuracy_score(y_test, yp),
            'precision': precision_score(y_test, yp, zero_division=0),
            'recall':    recall_score(y_test, yp, zero_division=0),
            'f1':        f1_score(y_test, yp, zero_division=0),
            'auc':       roc_auc_score(y_test, ypr),
        }

    fn_results.append(res)
    fn_labels.append(f"σ={sigma:.2f}")
    print(f"  σ={sigma:.2f}  |  "
          + "  ".join(f"{m} AUC={res[m]['auc']:.3f}" for m in _MODELS))

_print_table(NOISE_SIGMAS, 'σ', fn_results, [f"σ={s}" for s in NOISE_SIGMAS])

_plot_robustness(
    NOISE_SIGMAS, fn_results,
    [f"σ={s}" for s in NOISE_SIGMAS],
    xlabel="σ ruido Gaussiano (escala [-1,1])",
    title="EXP 2 — Feature Noise: Comportamiento de métricas con ruido creciente en inferencia",
    fname="robustness_feature_noise.png",
    highlight_ref=0,
)

pd.DataFrame([
    {'model': m, 'sigma': s,
     **{met: fn_results[i][m][met] for met in _METRICS}}
    for i, s in enumerate(NOISE_SIGMAS)
    for m in _MODELS
]).to_csv(out('robustness_feature_noise.csv'), index=False)
print("  Guardado: robustness_feature_noise.csv")


# ══════════════════════════════════════════════════════════════════════════════
#  BLOQUE 11 — EXP 3: LABEL NOISE
# ══════════════════════════════════════════════════════════════════════════════

FLIP_RATES = [0.00, 0.05, 0.10, 0.15, 0.20]

print("\n" + "=" * 70)
print("EXP 3 — LABEL NOISE")
print("=" * 70)
print("  Flip aleatorio de p% de y_train  |  y_test siempre limpio")
print(f"  Tasas: {[f'{p:.0%}' for p in FLIP_RATES]}\n")

ln_results = []
ln_labels  = []
_rng_ln    = np.random.default_rng(1)

for flip_rate in FLIP_RATES:
    y_noisy = y_train.copy()
    if flip_rate > 0:
        flip_mask = _rng_ln.random(len(y_noisy)) < flip_rate
        y_noisy[flip_mask] = 1 - y_noisy[flip_mask]
    n_flipped = (y_noisy != y_train).sum()

    res = _eval_all(X_train, y_noisy, X_test, y_test)
    ln_results.append(res)
    ln_labels.append(f"{flip_rate:.0%}\n({n_flipped} flip)")

    print(f"  flip={flip_rate:.0%}  ({n_flipped:3d} etiq. corruptas)  |  "
          + "  ".join(f"{m} AUC={res[m]['auc']:.3f}" for m in _MODELS))

_print_table(FLIP_RATES, 'flip', ln_results,
             [f"{p:.0%} ({(y_train.copy()).shape[0]})" for p in FLIP_RATES])

_plot_robustness(
    FLIP_RATES, ln_results,
    [f"{p:.0%}" for p in FLIP_RATES],
    xlabel="Tasa de flip de etiquetas (train)",
    title="EXP 3 — Label Noise: Comportamiento de métricas con etiquetas de entrenamiento corruptas",
    fname="robustness_label_noise.png",
    highlight_ref=0, 
)

pd.DataFrame([
    {'model': m, 'flip_rate': fp,
     **{met: ln_results[i][m][met] for met in _METRICS}}
    for i, fp in enumerate(FLIP_RATES)
    for m in _MODELS
]).to_csv(out('robustness_label_noise.csv'), index=False)
print("  Guardado: robustness_label_noise.csv")


# ══════════════════════════════════════════════════════════════════════════════
#  BLOQUE 12 — RESUMEN DE ROBUSTEZ
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("RESUMEN DE ROBUSTEZ — Caída de AUC (condición limpia → más extrema)")
print("=" * 70)

print(f"\n  {'Experimento':<22}  {'Modelo':<16}  {'AUC limpio':>10}  {'AUC extremo':>11}  {'ΔAUC':>7}")
print("  " + "-" * 72)

# Scarcity: limpio = 100%, extremo = 5%
for m in _MODELS:
    a_clean  = sc_results[-1][m]['auc']
    a_ext    = sc_results[0][m]['auc']
    print(f"  {'Scarcity (5%→100%)':<22}  {m:<16}  {a_clean:>10.4f}  {a_ext:>11.4f}  {a_ext-a_clean:>+7.4f}")

print()
# Feature noise: limpio = σ=0, extremo = σ=1.0
for m in _MODELS:
    a_clean = fn_results[0][m]['auc']
    a_ext   = fn_results[-1][m]['auc']
    print(f"  {'Feature Noise (σ=1.0)':<22}  {m:<16}  {a_clean:>10.4f}  {a_ext:>11.4f}  {a_ext-a_clean:>+7.4f}")

print()
# Label noise: limpio = 0%, extremo = 20%
for m in _MODELS:
    a_clean = ln_results[0][m]['auc']
    a_ext   = ln_results[-1][m]['auc']
    print(f"  {'Label Noise (20% flip)':<22}  {m:<16}  {a_clean:>10.4f}  {a_ext:>11.4f}  {a_ext-a_clean:>+7.4f}")

print("\n" + "=" * 70)
print("EXPERIMENTOS DE ROBUSTEZ FINALIZADOS")
print("=" * 70)
print(f"\nArchivos adicionales en: {OUTPUT_DIR}")
print("   8. robustness_scarcity.png  / robustness_scarcity.csv")
print("   9. robustness_feature_noise.png  / robustness_feature_noise.csv")
print("  10. robustness_label_noise.png  / robustness_label_noise.csv")
print("=" * 70)