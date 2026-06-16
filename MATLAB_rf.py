"""
MATLAB_rf.py  —  Random Forest Clásico (Clasificación y Regresión)
Tab para MATHABS: RFC / RFR con sklearn.ensemble
"""
import copy
import gc
import threading
import traceback
import time

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, StringVar, BooleanVar, IntVar, DoubleVar
from tkinter import scrolledtext

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score, confusion_matrix,
    classification_report, r2_score, mean_squared_error, mean_absolute_error,
)
from sklearn.inspection import permutation_importance

from MATLAB_filter_component import FilterComponent

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _coerce_int(v, default=10, minimum=1):
    try:
        return max(minimum, int(float(str(v).strip())))
    except Exception:
        return default


def _coerce_float(v, default=0.25, minimum=0.0, maximum=1.0):
    try:
        return max(minimum, min(maximum, float(str(v).strip())))
    except Exception:
        return default


def _fmt(v, decimals=4):
    if v is None:
        return "N/A"
    try:
        return f"{float(v):.{decimals}f}"
    except Exception:
        return str(v)


# ─────────────────────────────────────────────────────────────────────────────
# Main Tab Class
# ─────────────────────────────────────────────────────────────────────────────

class RFTab(ttk.Frame):
    """Random Forest Clasificación / Regresión — tab MATHABS."""

    _NON_RF_KEYS = {
        "test_size", "task_type", "target_col", "drop_first",
        "stratify", "missing_strategy",
    }

    def __init__(self, parent, main_app=None, **kwargs):
        super().__init__(parent, **kwargs)
        self.main_app = main_app
        self.data: pd.DataFrame | None = None

        # State
        self.model = None
        self.results: dict = {}
        self.feature_importance_df: pd.DataFrame = pd.DataFrame()
        self.latest_fit_dataframe: pd.DataFrame | None = None
        self.latest_target_col: str = ""
        self.latest_covariates: list = []
        self.latest_encoded_columns: list = []
        self.latest_report_text: str = ""
        self.latest_task_type: str = "auto"   # "classification" | "regression" | "auto"
        self.saved_models: list = []
        self.active_saved_model_index: int | None = None

        # Vars
        self.target_var = StringVar()
        self.task_type_var = StringVar(value="auto")
        self.test_size_var = StringVar(value="0.25")
        self.random_state_var = StringVar(value="42")
        self.n_estimators_var = StringVar(value="300")
        self.max_depth_var = StringVar(value="")
        self.min_samples_leaf_var = StringVar(value="4")
        self.min_samples_split_var = StringVar(value="8")
        self.max_features_var = StringVar(value="sqrt")
        self.n_jobs_var = IntVar(value=-1)
        self.oob_score_var = BooleanVar(value=True)
        self.class_weight_var = StringVar(value="balanced")
        self.missing_strategy_var = StringVar(value="Imputar mediana/moda")
        self.drop_first_var = BooleanVar(value=True)
        self.stratify_var = BooleanVar(value=True)
        self.status_var = StringVar(value="Listo.")

        self._build_ui()

    # ──────────────────────────────────────────────────────────────────────
    # UI
    # ──────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        self._build_preprocessing_tab()
        self._build_model_tab()
        self._build_results_tab()
        self._build_charts_tab()
        self._build_history_tab()

        # Status bar
        ttk.Label(self, textvariable=self.status_var, anchor="w",
                  relief="sunken").pack(fill=tk.X, side=tk.BOTTOM)

    # ── Preprocessing ──────────────────────────────────────────────────────

    def _build_preprocessing_tab(self):
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="1. Preprocesamiento")

        top = ttk.Frame(frame)
        top.pack(fill=tk.X, padx=10, pady=8)

        # Target
        ttk.Label(top, text="Variable objetivo (Y):").grid(row=0, column=0, sticky="w", padx=5)
        self.target_combo = ttk.Combobox(top, textvariable=self.target_var, width=28, state="readonly")
        self.target_combo.grid(row=0, column=1, padx=5, pady=3)

        # Task type
        ttk.Label(top, text="Tipo de tarea:").grid(row=0, column=2, sticky="w", padx=10)
        for i, (label, val) in enumerate([("Auto", "auto"), ("Clasificación", "classification"), ("Regresión", "regression")]):
            ttk.Radiobutton(top, text=label, variable=self.task_type_var, value=val).grid(
                row=0, column=3 + i, padx=3)

        # Split
        ttk.Label(top, text="Test %:").grid(row=1, column=0, sticky="w", padx=5)
        ttk.Entry(top, textvariable=self.test_size_var, width=8).grid(row=1, column=1, sticky="w", padx=5, pady=3)
        ttk.Label(top, text="Semilla:").grid(row=1, column=2, sticky="w", padx=10)
        ttk.Entry(top, textvariable=self.random_state_var, width=8).grid(row=1, column=3, sticky="w", padx=5)
        ttk.Checkbutton(top, text="Estratificar", variable=self.stratify_var).grid(row=1, column=4, padx=5)

        # Missing
        ttk.Label(top, text="Datos faltantes:").grid(row=2, column=0, sticky="w", padx=5)
        ttk.Combobox(top, textvariable=self.missing_strategy_var, width=24, state="readonly",
                     values=["Imputar mediana/moda", "Eliminar filas"]).grid(row=2, column=1, padx=5, pady=3)
        ttk.Checkbutton(top, text="Dummy drop_first", variable=self.drop_first_var).grid(row=2, column=2, padx=10)

        # Covariates
        mid = ttk.Frame(frame)
        mid.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        ttk.Label(mid, text="Covariables (X):").pack(anchor="w")
        lb_frame = ttk.Frame(mid)
        lb_frame.pack(fill=tk.BOTH, expand=True)
        scrollbar = ttk.Scrollbar(lb_frame, orient=tk.VERTICAL)
        self.cov_listbox = tk.Listbox(lb_frame, selectmode=tk.EXTENDED,
                                      yscrollcommand=scrollbar.set, height=10)
        scrollbar.config(command=self.cov_listbox.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.cov_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        btn_row = ttk.Frame(frame)
        btn_row.pack(fill=tk.X, padx=10, pady=5)
        ttk.Button(btn_row, text="Actualizar columnas", command=self._refresh_columns).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_row, text="Seleccionar todas", command=self._select_all_covariates).pack(side=tk.LEFT, padx=5)

    # ── Model ──────────────────────────────────────────────────────────────

    def _build_model_tab(self):
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="2. Modelo RF")

        hp = ttk.LabelFrame(frame, text="Hiperparámetros Random Forest")
        hp.pack(fill=tk.X, padx=10, pady=8)

        params = [
            ("Árboles (n_estimators):", self.n_estimators_var),
            ("Max profundidad (vacío=ilimitado):", self.max_depth_var),
            ("min_samples_leaf:", self.min_samples_leaf_var),
            ("min_samples_split:", self.min_samples_split_var),
            ("max_features:", self.max_features_var),
        ]
        for row_i, (lbl, var) in enumerate(params):
            ttk.Label(hp, text=lbl).grid(row=row_i, column=0, sticky="w", padx=8, pady=3)
            ttk.Entry(hp, textvariable=var, width=18).grid(row=row_i, column=1, sticky="w", padx=5)

        ttk.Label(hp, text="n_jobs:").grid(row=0, column=2, sticky="w", padx=15)
        ttk.Entry(hp, textvariable=self.n_jobs_var, width=6).grid(row=0, column=3, sticky="w", padx=5)
        ttk.Checkbutton(hp, text="OOB score", variable=self.oob_score_var).grid(row=1, column=2, columnspan=2, sticky="w", padx=15)
        ttk.Label(hp, text="Class weight (RFC):").grid(row=2, column=2, sticky="w", padx=15)
        ttk.Combobox(hp, textvariable=self.class_weight_var, width=14, state="readonly",
                     values=["balanced", "balanced_subsample", "none"]).grid(row=2, column=3, sticky="w", padx=5)

        action_row = ttk.Frame(frame)
        action_row.pack(fill=tk.X, padx=10, pady=8)
        ttk.Button(action_row, text="▶ Ejecutar RF", command=self.run_model).pack(side=tk.LEFT, padx=5)
        ttk.Button(action_row, text="⚡ Tuning automático", command=self.run_auto_tuning).pack(side=tk.LEFT, padx=5)

        # Tuning manual grid
        grid_frame = ttk.LabelFrame(frame, text="Tuning manual (listas separadas por coma)")
        grid_frame.pack(fill=tk.X, padx=10, pady=5)
        self.tune_n_estimators_var = StringVar(value="100,200,300,500")
        self.tune_max_depth_var = StringVar(value="None,5,10,20")
        self.tune_min_leaf_var = StringVar(value="2,4,8")
        self.tune_max_features_var = StringVar(value="sqrt,log2")
        grid_params = [
            ("n_estimators:", self.tune_n_estimators_var),
            ("max_depth:", self.tune_max_depth_var),
            ("min_samples_leaf:", self.tune_min_leaf_var),
            ("max_features:", self.tune_max_features_var),
        ]
        for col_i, (lbl, var) in enumerate(grid_params):
            ttk.Label(grid_frame, text=lbl).grid(row=0, column=col_i * 2, sticky="w", padx=5, pady=3)
            ttk.Entry(grid_frame, textvariable=var, width=22).grid(row=0, column=col_i * 2 + 1, padx=5)

    # ── Results ────────────────────────────────────────────────────────────

    def _build_results_tab(self):
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="3. Resultados")
        self.results_text = scrolledtext.ScrolledText(frame, wrap=tk.WORD, font=("Consolas", 9))
        self.results_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    # ── Charts ─────────────────────────────────────────────────────────────

    def _build_charts_tab(self):
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="4. Gráficas")

        ctrl = ttk.Frame(frame)
        ctrl.pack(fill=tk.X, padx=5, pady=4)
        ttk.Button(ctrl, text="Importancia variables", command=self.plot_feature_importance).pack(side=tk.LEFT, padx=4)
        ttk.Button(ctrl, text="ROC / Residuales", command=self.plot_roc_or_residuals).pack(side=tk.LEFT, padx=4)
        ttk.Button(ctrl, text="Matriz confusión", command=self.plot_confusion_matrix).pack(side=tk.LEFT, padx=4)
        ttk.Button(ctrl, text="Real vs Predicho", command=self.plot_actual_vs_predicted).pack(side=tk.LEFT, padx=4)

        self.chart_frame = ttk.Frame(frame)
        self.chart_frame.pack(fill=tk.BOTH, expand=True)
        self._chart_fig = None
        self._chart_canvas = None

    def _get_chart_ax(self, figsize=(8, 5)):
        for w in self.chart_frame.winfo_children():
            w.destroy()
        fig, ax = plt.subplots(figsize=figsize)
        canvas = FigureCanvasTkAgg(fig, master=self.chart_frame)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self._chart_fig = fig
        self._chart_canvas = canvas
        return fig, ax

    # ── History ────────────────────────────────────────────────────────────

    def _build_history_tab(self):
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="5. Historial")

        cols = ("id", "tipo", "objetivo", "arboles", "acc_auc_r2", "f1_rmse", "oob", "cv")
        self.history_tree = ttk.Treeview(frame, columns=cols, show="headings", height=14)
        headers = {"id": ("ID", 40), "tipo": ("Tipo", 80), "objetivo": ("Objetivo", 120),
                   "arboles": ("Árboles", 65), "acc_auc_r2": ("Acc/AUC/R²", 90),
                   "f1_rmse": ("F1/RMSE", 80), "oob": ("OOB", 65), "cv": ("CV", 75)}
        for col, (heading, width) in headers.items():
            self.history_tree.heading(col, text=heading)
            self.history_tree.column(col, width=width, minwidth=30)

        vsb = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=self.history_tree.yview)
        self.history_tree.configure(yscrollcommand=vsb.set)
        self.history_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)

        btn_row = ttk.Frame(frame)
        btn_row.pack(fill=tk.X, padx=5, pady=5)
        ttk.Button(btn_row, text="Cargar seleccionado", command=self._load_selected_model).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_row, text="Eliminar seleccionado", command=self._delete_selected_model).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_row, text="Limpiar lista", command=self._clear_history).pack(side=tk.LEFT, padx=5)

        self.history_tree.bind("<Double-1>", lambda e: self._load_selected_model())

    # ──────────────────────────────────────────────────────────────────────
    # Data
    # ──────────────────────────────────────────────────────────────────────

    def set_data(self, df: pd.DataFrame):
        self.data = df
        self._refresh_columns()

    def _get_shared_data(self):
        if self.data is not None:
            return self.data
        if self.main_app is not None:
            for attr in ("shared_dataframe", "data", "df"):
                df = getattr(self.main_app, attr, None)
                if isinstance(df, pd.DataFrame) and not df.empty:
                    return df
        return None

    def _refresh_columns(self):
        df = self._get_shared_data()
        if df is None:
            return
        cols = list(df.columns)
        self.target_combo["values"] = cols
        if not self.target_var.get() and cols:
            self.target_var.set(cols[-1])
        self.cov_listbox.delete(0, tk.END)
        for c in cols:
            self.cov_listbox.insert(tk.END, c)

    def _select_all_covariates(self):
        self.cov_listbox.selection_set(0, tk.END)

    # ──────────────────────────────────────────────────────────────────────
    # Prepare data
    # ──────────────────────────────────────────────────────────────────────

    def _prepare_data(self, df, target_col, covariates):
        """Returns (X_encoded, y, task_type, encoded_columns, warnings_list)."""
        warnings = []

        # Impute / drop missing
        strategy = self.missing_strategy_var.get()
        work = df[covariates + [target_col]].copy()
        if strategy == "Eliminar filas":
            before = len(work)
            work = work.dropna()
            if len(work) < before:
                warnings.append(f"Se eliminaron {before - len(work)} filas con datos faltantes.")
        else:
            for col in covariates:
                if work[col].dtype in (np.float64, np.int64, float, int) or pd.api.types.is_numeric_dtype(work[col]):
                    work[col] = work[col].fillna(work[col].median())
                else:
                    work[col] = work[col].fillna(work[col].mode().iloc[0] if not work[col].mode().empty else "")
            # target: drop rows where target is null
            before = len(work)
            work = work.dropna(subset=[target_col])
            if len(work) < before:
                warnings.append(f"Se eliminaron {before - len(work)} filas donde el objetivo era nulo.")

        if work.empty:
            raise ValueError("Sin datos tras preprocesar.")

        # Detect task type
        task = self.task_type_var.get()
        y_raw = work[target_col]
        if task == "auto":
            n_unique = y_raw.nunique()
            is_numeric = pd.api.types.is_numeric_dtype(y_raw)
            if not is_numeric or n_unique <= 20:
                task = "classification"
            else:
                task = "regression"
            warnings.append(f"Tarea detectada automáticamente: {task} ({n_unique} valores únicos en '{target_col}').")

        # Encode target
        if task == "classification":
            le = LabelEncoder()
            y = le.fit_transform(y_raw.astype(str))
            self._label_encoder = le
            self._target_classes = le.classes_
        else:
            y = pd.to_numeric(y_raw, errors="coerce").values
            self._label_encoder = None
            self._target_classes = None
            if np.isnan(y).any():
                mask = ~np.isnan(y)
                work = work[mask]
                y = y[mask]
                warnings.append("Se descartaron filas donde el objetivo no era numérico.")

        # Encode covariates
        cov_df = work[covariates].copy()
        cat_cols = [c for c in covariates if not pd.api.types.is_numeric_dtype(cov_df[c])]
        if cat_cols:
            cov_df = pd.get_dummies(cov_df, columns=cat_cols,
                                    drop_first=bool(self.drop_first_var.get()))

        encoded_columns = list(cov_df.columns)
        return cov_df.astype(float), y, task, encoded_columns, warnings

    # ──────────────────────────────────────────────────────────────────────
    # Build RF params from UI
    # ──────────────────────────────────────────────────────────────────────

    def _get_rf_params(self):
        params: dict = {}
        params["n_estimators"] = _coerce_int(self.n_estimators_var.get(), 300, minimum=10)
        _md = str(self.max_depth_var.get()).strip()
        params["max_depth"] = None if _md in ("", "None", "none") else _coerce_int(_md, None, minimum=1)
        params["min_samples_leaf"] = _coerce_int(self.min_samples_leaf_var.get(), 4, minimum=1)
        params["min_samples_split"] = _coerce_int(self.min_samples_split_var.get(), 8, minimum=2)
        _mf = str(self.max_features_var.get()).strip().lower()
        if _mf in ("sqrt", "log2"):
            params["max_features"] = _mf
        elif _mf in ("none", "all", ""):
            params["max_features"] = None
        else:
            try:
                params["max_features"] = int(float(_mf)) if float(_mf) >= 1 else float(_mf)
            except Exception:
                params["max_features"] = "sqrt"
        params["oob_score"] = bool(self.oob_score_var.get())
        params["n_jobs"] = _coerce_int(self.n_jobs_var.get(), -1, minimum=-1) or -1
        params["random_state"] = _coerce_int(self.random_state_var.get(), 42)
        return params

    def _build_rf_model(self, params, task_type, class_weight=None):
        """Instantiate RFC or RFR, stripping non-sklearn keys."""
        clean = {k: v for k, v in params.items() if k not in self._NON_RF_KEYS}
        if not clean.get("oob_score"):
            clean["oob_score"] = False
        if task_type == "classification":
            cw = class_weight or (self.class_weight_var.get() if self.class_weight_var.get() != "none" else None)
            return RandomForestClassifier(class_weight=cw, **{k: v for k, v in clean.items() if k != "class_weight"})
        else:
            clean.pop("class_weight", None)
            return RandomForestRegressor(**clean)

    # ──────────────────────────────────────────────────────────────────────
    # Thread-safe fit
    # ──────────────────────────────────────────────────────────────────────

    def _fit_with_ui_pump(self, model, X, y):
        """Fit in background thread, pump UI every 100ms."""
        _exc = [None]
        orig_jobs = getattr(model, "n_jobs", 1)
        try:
            model.set_params(n_jobs=1)
        except Exception:
            pass

        def _worker():
            try:
                model.fit(X, y)
            except Exception as e:
                _exc[0] = e

        t = threading.Thread(target=_worker, daemon=True)
        t.start()
        while t.is_alive():
            try:
                self.update_idletasks()
            except Exception:
                pass
            t.join(timeout=0.1)
        t.join()
        try:
            model.set_params(n_jobs=orig_jobs)
        except Exception:
            pass
        if _exc[0] is not None:
            raise _exc[0]

    # ──────────────────────────────────────────────────────────────────────
    # Compute metrics
    # ──────────────────────────────────────────────────────────────────────

    def _compute_metrics(self, model, X_train, X_test, y_train, y_test, task_type, rf_params):
        metrics = {}
        metrics["task_type"] = task_type
        metrics["oob_score"] = getattr(model, "oob_score_", None)

        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test) if len(X_test) > 0 else np.array([])

        if task_type == "classification":
            metrics["accuracy_train"] = accuracy_score(y_train, y_pred_train)
            metrics["accuracy_test"] = accuracy_score(y_test, y_pred_test) if len(y_test) > 0 else None
            metrics["f1_train"] = f1_score(y_train, y_pred_train, average="weighted", zero_division=0)
            metrics["f1_test"] = f1_score(y_test, y_pred_test, average="weighted", zero_division=0) if len(y_test) > 0 else None
            # AUC
            n_classes = len(np.unique(y_train))
            if n_classes == 2 and len(X_test) > 0 and hasattr(model, "predict_proba"):
                try:
                    proba = model.predict_proba(X_test)[:, 1]
                    metrics["auc_test"] = roc_auc_score(y_test, proba)
                except Exception:
                    metrics["auc_test"] = None
            elif n_classes > 2 and len(X_test) > 0 and hasattr(model, "predict_proba"):
                try:
                    proba = model.predict_proba(X_test)
                    metrics["auc_test"] = roc_auc_score(y_test, proba, multi_class="ovr", average="weighted")
                except Exception:
                    metrics["auc_test"] = None
            else:
                metrics["auc_test"] = None
            # CV
            try:
                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc" if n_classes == 2 else "f1_weighted", n_jobs=1)
                metrics["cv_mean"] = float(np.mean(cv_scores))
                metrics["cv_std"] = float(np.std(cv_scores))
            except Exception:
                metrics["cv_mean"] = None
                metrics["cv_std"] = None

        else:  # regression
            metrics["r2_train"] = r2_score(y_train, y_pred_train)
            metrics["r2_test"] = r2_score(y_test, y_pred_test) if len(y_test) > 0 else None
            metrics["rmse_train"] = float(np.sqrt(mean_squared_error(y_train, y_pred_train)))
            metrics["rmse_test"] = float(np.sqrt(mean_squared_error(y_test, y_pred_test))) if len(y_test) > 0 else None
            metrics["mae_test"] = float(mean_absolute_error(y_test, y_pred_test)) if len(y_test) > 0 else None
            try:
                cv = KFold(n_splits=5, shuffle=True, random_state=42)
                cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="r2", n_jobs=1)
                metrics["cv_mean"] = float(np.mean(cv_scores))
                metrics["cv_std"] = float(np.std(cv_scores))
            except Exception:
                metrics["cv_mean"] = None
                metrics["cv_std"] = None

        return metrics

    def _compute_feature_importance(self, model, X_train, y_train, encoded_cols):
        try:
            base_imp = model.feature_importances_
            df = pd.DataFrame({
                "variable": encoded_cols,
                "importance_mdi": base_imp,
            })
            df = df.sort_values("importance_mdi", ascending=False).reset_index(drop=True)
            return df
        except Exception:
            return pd.DataFrame()

    # ──────────────────────────────────────────────────────────────────────
    # Report
    # ──────────────────────────────────────────────────────────────────────

    def _build_report(self, metrics, rf_params, target_col, covariates, task_type,
                      encoded_cols, n_train, n_test, warnings):
        lines = []
        lines.append("=" * 60)
        lines.append(f"  Random Forest {'Clasificación' if task_type == 'classification' else 'Regresión'}")
        lines.append("=" * 60)
        lines.append(f"Objetivo       : {target_col}")
        lines.append(f"Covariables    : {len(covariates)} originales → {len(encoded_cols)} codificadas")
        lines.append(f"Train / Test   : {n_train} / {n_test}")
        lines.append(f"Árboles        : {rf_params.get('n_estimators')}")
        lines.append(f"max_depth      : {rf_params.get('max_depth', 'None')}")
        lines.append(f"min_leaf       : {rf_params.get('min_samples_leaf')}")
        lines.append(f"max_features   : {rf_params.get('max_features')}")
        lines.append("")

        if task_type == "classification":
            lines.append("── Métricas ─────────────────────────────────")
            lines.append(f"Accuracy  train : {_fmt(metrics.get('accuracy_train'), 4)}")
            lines.append(f"Accuracy  test  : {_fmt(metrics.get('accuracy_test'), 4)}")
            lines.append(f"F1 (ponderado) train : {_fmt(metrics.get('f1_train'), 4)}")
            lines.append(f"F1 (ponderado) test  : {_fmt(metrics.get('f1_test'), 4)}")
            lines.append(f"AUC-ROC   test  : {_fmt(metrics.get('auc_test'), 4)}")
            lines.append(f"OOB score       : {_fmt(metrics.get('oob_score'), 4)}")
            lines.append(f"CV 5-fold (media±std): {_fmt(metrics.get('cv_mean'), 4)} ± {_fmt(metrics.get('cv_std'), 4)}")
        else:
            lines.append("── Métricas ─────────────────────────────────")
            lines.append(f"R²        train : {_fmt(metrics.get('r2_train'), 4)}")
            lines.append(f"R²        test  : {_fmt(metrics.get('r2_test'), 4)}")
            lines.append(f"RMSE      train : {_fmt(metrics.get('rmse_train'), 4)}")
            lines.append(f"RMSE      test  : {_fmt(metrics.get('rmse_test'), 4)}")
            lines.append(f"MAE       test  : {_fmt(metrics.get('mae_test'), 4)}")
            lines.append(f"OOB score       : {_fmt(metrics.get('oob_score'), 4)}")
            lines.append(f"CV R² 5-fold (media±std): {_fmt(metrics.get('cv_mean'), 4)} ± {_fmt(metrics.get('cv_std'), 4)}")

        if warnings:
            lines.append("")
            lines.append("── Advertencias ─────────────────────────────")
            for w in warnings:
                lines.append(f"  ⚠ {w}")

        lines.append("")
        lines.append("── Variables más importantes (MDI) ──────────")
        if not self.feature_importance_df.empty:
            for _, row in self.feature_importance_df.head(20).iterrows():
                bar = "█" * int(row["importance_mdi"] * 40)
                lines.append(f"  {row['variable']:<35} {row['importance_mdi']:.4f}  {bar}")

        return "\n".join(lines)

    # ──────────────────────────────────────────────────────────────────────
    # Run model
    # ──────────────────────────────────────────────────────────────────────

    def run_model(self, params_override=None, store_snapshot=True,
                  covariates_override=None, target_override=None):
        df = self._get_shared_data()
        if df is None or df.empty:
            messagebox.showerror("RF", "No hay datos disponibles.")
            return

        target_col = target_override or self.target_var.get().strip()
        if not target_col or target_col not in df.columns:
            messagebox.showerror("RF", "Selecciona una variable objetivo válida.")
            return

        selected_indices = self.cov_listbox.curselection()
        covariates = covariates_override or [self.cov_listbox.get(i) for i in selected_indices]
        covariates = [c for c in covariates if c != target_col]
        if not covariates:
            messagebox.showerror("RF", "Selecciona al menos una covariable.")
            return

        self.status_var.set("Preparando datos...")
        self.update_idletasks()

        try:
            X_enc, y, task_type, encoded_cols, warnings = self._prepare_data(df, target_col, covariates)
        except Exception as exc:
            messagebox.showerror("Error RF", f"Error en preprocesamiento:\n{exc}")
            self.status_var.set("Error en preprocesamiento.")
            return

        test_size = _coerce_float(self.test_size_var.get(), 0.25)
        random_state = _coerce_int(self.random_state_var.get(), 42)

        if test_size > 0:
            try:
                stratify_arg = y if (bool(self.stratify_var.get()) and task_type == "classification") else None
                X_train, X_test, y_train, y_test = train_test_split(
                    X_enc, y, test_size=test_size,
                    random_state=random_state, stratify=stratify_arg)
            except Exception:
                X_train, X_test, y_train, y_test = train_test_split(
                    X_enc, y, test_size=test_size, random_state=random_state)
        else:
            X_train, X_test, y_train, y_test = X_enc, X_enc.iloc[0:0], y, y[:0]

        # Build params
        if isinstance(params_override, dict) and params_override:
            rf_params = copy.deepcopy(params_override)
            for k in list(rf_params.keys()):
                if k in self._NON_RF_KEYS:
                    rf_params.pop(k)
        else:
            rf_params = self._get_rf_params()

        self.status_var.set(f"Entrenando {task_type} RF con {rf_params.get('n_estimators')} árboles...")
        self.update_idletasks()

        try:
            model = self._build_rf_model(rf_params, task_type)
            self._fit_with_ui_pump(model, X_train, y_train)
        except Exception as exc:
            messagebox.showerror("Error RF", f"Error al entrenar:\n{exc}")
            self.status_var.set("Error al entrenar.")
            return

        self.status_var.set("Calculando métricas...")
        self.update_idletasks()

        metrics = self._compute_metrics(model, X_train, X_test, y_train, y_test, task_type, rf_params)
        self.feature_importance_df = self._compute_feature_importance(model, X_train, y_train, encoded_cols)

        report = self._build_report(metrics, rf_params, target_col, covariates, task_type,
                                     encoded_cols, len(X_train), len(X_test), warnings)

        # Save state
        self.model = model
        self.results = metrics
        self.latest_fit_dataframe = X_enc
        self.latest_target_col = target_col
        self.latest_covariates = list(covariates)
        self.latest_encoded_columns = list(encoded_cols)
        self.latest_report_text = report
        self.latest_task_type = task_type
        self._last_X_train = X_train
        self._last_X_test = X_test
        self._last_y_train = y_train
        self._last_y_test = y_test

        # Display report
        self.results_text.delete("1.0", tk.END)
        self.results_text.insert(tk.END, report)

        if store_snapshot:
            self._store_snapshot(rf_params, metrics, report, task_type,
                                  target_col, covariates, encoded_cols, X_enc)

        _main_metric = _fmt(metrics.get("cv_mean"), 3)
        self.status_var.set(
            f"Listo. Tipo: {task_type}  |  CV={_main_metric}  |  "
            f"OOB={_fmt(metrics.get('oob_score'), 3)}"
        )
        self.plot_feature_importance()

    # ──────────────────────────────────────────────────────────────────────
    # Auto tuning
    # ──────────────────────────────────────────────────────────────────────

    def run_auto_tuning(self):
        df = self._get_shared_data()
        if df is None or df.empty:
            messagebox.showerror("RF Tuning", "No hay datos disponibles.")
            return
        target_col = self.target_var.get().strip()
        selected_indices = self.cov_listbox.curselection()
        covariates = [self.cov_listbox.get(i) for i in selected_indices]
        covariates = [c for c in covariates if c != target_col]
        if not target_col or not covariates:
            messagebox.showerror("RF Tuning", "Selecciona objetivo y covariables.")
            return

        # Parse manual tuning grids
        def _parse_list(s, typ="int"):
            result = []
            for part in str(s).split(","):
                part = part.strip()
                if part.lower() in ("none", ""):
                    result.append(None)
                    continue
                try:
                    result.append(int(part) if typ == "int" else part)
                except Exception:
                    result.append(part)
            return result or [None]

        n_estimators_grid = _parse_list(self.tune_n_estimators_var.get(), "int")
        max_depth_grid = _parse_list(self.tune_max_depth_var.get(), "int")
        min_leaf_grid = _parse_list(self.tune_min_leaf_var.get(), "int")
        max_features_grid = _parse_list(self.tune_max_features_var.get(), "str")

        candidates = []
        for ne in n_estimators_grid:
            for md in max_depth_grid:
                for ml in min_leaf_grid:
                    for mf in max_features_grid:
                        p = copy.deepcopy(self._get_rf_params())
                        if ne is not None:
                            p["n_estimators"] = int(ne)
                        if md is not None:
                            try:
                                p["max_depth"] = int(md)
                            except Exception:
                                p["max_depth"] = None
                        else:
                            p["max_depth"] = None
                        if ml is not None:
                            try:
                                p["min_samples_leaf"] = int(ml)
                                p["min_samples_split"] = max(2, int(ml) * 2)
                            except Exception:
                                pass
                        if mf is not None:
                            _mf_s = str(mf).strip().lower()
                            p["max_features"] = None if _mf_s in ("none", "") else _mf_s
                        candidates.append(p)

        total = len(candidates)
        if total > 200:
            if not messagebox.askyesno("RF Tuning",
                    f"Se evaluarán {total} combinaciones. ¿Continuar?"):
                return

        # Progress dialog
        prog_win = tk.Toplevel(self)
        prog_win.title("Tuning RF en progreso")
        prog_win.geometry("460x140")
        prog_win.grab_set()
        prog_var = StringVar(value="Iniciando...")
        ttk.Label(prog_win, textvariable=prog_var, wraplength=440).pack(padx=10, pady=8)
        pb = ttk.Progressbar(prog_win, maximum=total, mode="determinate")
        pb.pack(fill=tk.X, padx=10)
        cancel_flag = [False]
        ttk.Button(prog_win, text="Cancelar", command=lambda: cancel_flag.__setitem__(0, True)).pack(pady=8)
        prog_win.update()

        best_score = None
        best_params = None
        best_metrics = None
        results_list = []

        try:
            X_enc, y, task_type, encoded_cols, warnings = self._prepare_data(df, target_col, covariates)
        except Exception as exc:
            prog_win.destroy()
            messagebox.showerror("RF Tuning", f"Error en datos:\n{exc}")
            return

        test_size = _coerce_float(self.test_size_var.get(), 0.25)
        random_state = _coerce_int(self.random_state_var.get(), 42)
        try:
            stratify_arg = y if (bool(self.stratify_var.get()) and task_type == "classification") else None
            X_train, X_test, y_train, y_test = train_test_split(
                X_enc, y, test_size=test_size,
                random_state=random_state, stratify=stratify_arg)
        except Exception:
            X_train, X_test, y_train, y_test = train_test_split(
                X_enc, y, test_size=test_size, random_state=random_state)

        _exc_outer = [None]

        def _tuning_worker():
            try:
                for i, params in enumerate(candidates):
                    if cancel_flag[0]:
                        break
                    _ne = params.get("n_estimators", 300)
                    prog_var.set(f"Candidato {i + 1}/{total} — {_ne} árboles, "
                                 f"depth={params.get('max_depth')}, leaf={params.get('min_samples_leaf')}")
                    pb["value"] = i + 1
                    try:
                        prog_win.update()
                    except Exception:
                        break

                    try:
                        model = self._build_rf_model(params, task_type)
                        model.set_params(n_jobs=1)
                        model.fit(X_train, y_train)
                        metrics = self._compute_metrics(model, X_train, X_test, y_train, y_test,
                                                        task_type, params)
                        score = metrics.get("cv_mean")
                        if score is None:
                            score = (metrics.get("accuracy_test") or metrics.get("r2_test") or float("-inf"))
                        results_list.append({"params": params, "metrics": metrics, "score": float(score)})

                        nonlocal best_score, best_params, best_metrics
                        if best_score is None or float(score) > float(best_score or float("-inf")):
                            best_score = float(score)
                            best_params = copy.deepcopy(params)
                            best_metrics = copy.deepcopy(metrics)
                    except Exception:
                        pass

                    model = None
                    if i % 10 == 0:
                        gc.collect()
            except Exception as e:
                _exc_outer[0] = e

        t = threading.Thread(target=_tuning_worker, daemon=True)
        t.start()
        while t.is_alive():
            try:
                prog_win.update()
            except Exception:
                break
            t.join(timeout=0.15)
        t.join()

        try:
            prog_win.destroy()
        except Exception:
            pass

        if _exc_outer[0] is not None:
            messagebox.showerror("RF Tuning", f"Error durante tuning:\n{_exc_outer[0]}")
            return

        if best_params is None:
            messagebox.showwarning("RF Tuning", "No se pudo evaluar ninguna configuración.")
            return

        messagebox.showinfo("RF Tuning",
            f"Tuning completo: {len(results_list)}/{total} evaluados.\n"
            f"Mejor CV: {_fmt(best_score, 4)}\n"
            f"Parámetros: n={best_params.get('n_estimators')}, "
            f"depth={best_params.get('max_depth')}, leaf={best_params.get('min_samples_leaf')}, "
            f"mf={best_params.get('max_features')}\n\n"
            f"Se entrena y carga el mejor modelo.")

        self.run_model(params_override=best_params, covariates_override=covariates,
                       target_override=target_col, store_snapshot=True)

    # ──────────────────────────────────────────────────────────────────────
    # Snapshots / History
    # ──────────────────────────────────────────────────────────────────────

    def _store_snapshot(self, rf_params, metrics, report, task_type,
                         target_col, covariates, encoded_cols, fit_df):
        label = f"RF #{len(self.saved_models) + 1} ({task_type[:3].upper()})"
        snap = {
            "label": label,
            "params": copy.deepcopy(rf_params),
            "metrics": copy.deepcopy(metrics),
            "report_text": report,
            "model": self.model,
            "feature_importance_df": self.feature_importance_df.copy(deep=True)
                if isinstance(self.feature_importance_df, pd.DataFrame) else pd.DataFrame(),
            "task_type": task_type,
            "target_col": target_col,
            "covariates": list(covariates),
            "encoded_cols": list(encoded_cols),
            "fit_dataframe": fit_df.copy(deep=True)
                if isinstance(fit_df, pd.DataFrame) else None,
            "label_encoder": getattr(self, "_label_encoder", None),
            "target_classes": getattr(self, "_target_classes", None),
        }
        self.saved_models.append(snap)
        self.active_saved_model_index = len(self.saved_models) - 1
        self._refresh_history_tree()

    def _refresh_history_tree(self):
        self.history_tree.delete(*self.history_tree.get_children())
        for i, snap in enumerate(self.saved_models):
            m = snap.get("metrics", {})
            task = snap.get("task_type", "?")
            if task == "classification":
                main_metric = _fmt(m.get("auc_test") or m.get("accuracy_test"), 3)
                sec_metric = _fmt(m.get("f1_test"), 3)
            else:
                main_metric = _fmt(m.get("r2_test"), 3)
                sec_metric = _fmt(m.get("rmse_test"), 3)
            self.history_tree.insert("", tk.END, iid=str(i), values=(
                i + 1,
                task[:4].upper(),
                snap.get("target_col", ""),
                snap.get("params", {}).get("n_estimators", "?"),
                main_metric,
                sec_metric,
                _fmt(m.get("oob_score"), 3),
                _fmt(m.get("cv_mean"), 3),
            ))

    def _load_selected_model(self):
        selected = self.history_tree.selection()
        if not selected:
            return
        idx = int(selected[0])
        if idx < 0 or idx >= len(self.saved_models):
            return
        snap = self.saved_models[idx]
        self.active_saved_model_index = idx

        # If snapshot has full model, restore directly
        if snap.get("model") is not None:
            self.model = snap["model"]
            self.results = copy.deepcopy(snap.get("metrics", {}))
            self.feature_importance_df = snap["feature_importance_df"].copy(deep=True) \
                if isinstance(snap.get("feature_importance_df"), pd.DataFrame) else pd.DataFrame()
            self.latest_task_type = snap.get("task_type", "classification")
            self.latest_target_col = snap.get("target_col", "")
            self.latest_covariates = list(snap.get("covariates", []))
            self.latest_encoded_columns = list(snap.get("encoded_cols", []))
            self.latest_report_text = snap.get("report_text", "")
            self._label_encoder = snap.get("label_encoder")
            self._target_classes = snap.get("target_classes")
            self.results_text.delete("1.0", tk.END)
            self.results_text.insert(tk.END, self.latest_report_text)
            self.status_var.set(f"Modelo cargado: {snap.get('label')}")
            self.plot_feature_importance()
        else:
            # Retrain
            self.run_model(
                params_override=snap.get("params"),
                covariates_override=snap.get("covariates"),
                target_override=snap.get("target_col"),
                store_snapshot=False,
            )
            snap["model"] = self.model
            snap["feature_importance_df"] = self.feature_importance_df.copy(deep=True) \
                if isinstance(self.feature_importance_df, pd.DataFrame) else pd.DataFrame()

    def _delete_selected_model(self):
        selected = self.history_tree.selection()
        if not selected:
            return
        idx = int(selected[0])
        if 0 <= idx < len(self.saved_models):
            self.saved_models.pop(idx)
            self.active_saved_model_index = None
            self._refresh_history_tree()

    def _clear_history(self):
        if messagebox.askyesno("Limpiar historial", "¿Eliminar todos los modelos guardados?"):
            self.saved_models.clear()
            self.active_saved_model_index = None
            self._refresh_history_tree()

    # ──────────────────────────────────────────────────────────────────────
    # Plots
    # ──────────────────────────────────────────────────────────────────────

    def plot_feature_importance(self):
        if self.model is None or self.feature_importance_df.empty:
            return
        df = self.feature_importance_df.head(25)
        fig, ax = self._get_chart_ax(figsize=(9, max(4, len(df) * 0.3 + 1)))
        colors = ["#E53935" if i == 0 else "#1565C0" for i in range(len(df))]
        ax.barh(df["variable"][::-1], df["importance_mdi"][::-1], color=colors[::-1])
        ax.set_xlabel("Importancia MDI", fontsize=9)
        ax.set_title("Importancia de variables (MDI)", fontsize=11, fontweight="bold")
        ax.tick_params(labelsize=8)
        fig.tight_layout()
        self._chart_canvas.draw_idle()

    def plot_roc_or_residuals(self):
        if self.model is None:
            messagebox.showwarning("RF", "Ejecuta un modelo primero.")
            return
        task = self.latest_task_type
        if task == "classification":
            self._plot_roc()
        else:
            self._plot_residuals()

    def _plot_roc(self):
        from sklearn.metrics import roc_curve
        if not hasattr(self, "_last_X_test") or self._last_X_test is None:
            return
        X_test = self._last_X_test
        y_test = self._last_y_test
        if len(X_test) == 0:
            messagebox.showwarning("RF", "No hay datos de test para la ROC.")
            return

        classes = getattr(self, "_target_classes", None)
        n_classes = len(np.unique(y_test))

        if n_classes == 2 and hasattr(self.model, "predict_proba"):
            fig, ax = self._get_chart_ax()
            proba = self.model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, proba)
            auc = roc_auc_score(y_test, proba)
            ax.plot(fpr, tpr, color="#1565C0", lw=2, label=f"AUC = {auc:.4f}")
            ax.plot([0, 1], [0, 1], "k--", lw=1)
            ax.set_xlabel("1 - Especificidad (FPR)", fontsize=9)
            ax.set_ylabel("Sensibilidad (TPR)", fontsize=9)
            ax.set_title("Curva ROC", fontsize=11, fontweight="bold")
            ax.legend(fontsize=9)
            fig.tight_layout()
            self._chart_canvas.draw_idle()
        else:
            messagebox.showinfo("RF", "ROC multi-clase: use la métrica AUC en el reporte.")

    def _plot_residuals(self):
        if not hasattr(self, "_last_X_test") or self._last_X_test is None:
            return
        X_test = self._last_X_test
        y_test = self._last_y_test
        if len(X_test) == 0:
            messagebox.showwarning("RF", "No hay datos de test para residuales.")
            return
        y_pred = self.model.predict(X_test)
        residuals = y_test - y_pred
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for w in self.chart_frame.winfo_children():
            w.destroy()
        canvas = FigureCanvasTkAgg(fig, master=self.chart_frame)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self._chart_fig = fig
        self._chart_canvas = canvas

        axes[0].scatter(y_pred, residuals, alpha=0.5, s=20, color="#1565C0")
        axes[0].axhline(0, color="red", lw=1.5, ls="--")
        axes[0].set_xlabel("Predicho", fontsize=9)
        axes[0].set_ylabel("Residual", fontsize=9)
        axes[0].set_title("Residuales vs Predicho", fontsize=10)

        axes[1].hist(residuals, bins=30, color="#1565C0", alpha=0.8, edgecolor="k")
        axes[1].set_xlabel("Residual", fontsize=9)
        axes[1].set_title("Distribución residuales", fontsize=10)

        fig.tight_layout()
        canvas.draw_idle()

    def plot_confusion_matrix(self):
        if self.model is None or self.latest_task_type != "classification":
            messagebox.showwarning("RF", "Solo disponible para clasificación.")
            return
        if not hasattr(self, "_last_X_test") or len(self._last_X_test) == 0:
            messagebox.showwarning("RF", "No hay datos de test.")
            return
        y_pred = self.model.predict(self._last_X_test)
        cm = confusion_matrix(self._last_y_test, y_pred)
        classes = getattr(self, "_target_classes", None)
        labels = [str(c) for c in classes] if classes is not None else [str(i) for i in range(cm.shape[0])]

        fig, ax = self._get_chart_ax(figsize=(max(5, len(labels)), max(4, len(labels))))
        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        fig.colorbar(im, ax=ax)
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(labels, fontsize=8)
        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black", fontsize=9)
        ax.set_xlabel("Predicho", fontsize=9)
        ax.set_ylabel("Real", fontsize=9)
        ax.set_title("Matriz de Confusión", fontsize=11, fontweight="bold")
        fig.tight_layout()
        self._chart_canvas.draw_idle()

    def plot_actual_vs_predicted(self):
        if self.model is None or self.latest_task_type != "regression":
            messagebox.showwarning("RF", "Solo disponible para regresión.")
            return
        if not hasattr(self, "_last_X_test") or len(self._last_X_test) == 0:
            messagebox.showwarning("RF", "No hay datos de test.")
            return
        y_pred = self.model.predict(self._last_X_test)
        y_real = self._last_y_test
        fig, ax = self._get_chart_ax()
        ax.scatter(y_real, y_pred, alpha=0.5, s=20, color="#1565C0")
        mn = min(y_real.min(), y_pred.min())
        mx = max(y_real.max(), y_pred.max())
        ax.plot([mn, mx], [mn, mx], "r--", lw=1.5, label="y=x (perfecto)")
        ax.set_xlabel("Real", fontsize=9)
        ax.set_ylabel("Predicho", fontsize=9)
        ax.set_title(f"Real vs Predicho  —  R²={_fmt(self.results.get('r2_test'), 4)}", fontsize=11)
        ax.legend(fontsize=9)
        fig.tight_layout()
        self._chart_canvas.draw_idle()
