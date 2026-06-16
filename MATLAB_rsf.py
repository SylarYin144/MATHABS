import os
import re
import copy
import pickle
import traceback
import importlib
import inspect
import time
from statistics import NormalDist
from typing import Any
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog, StringVar, BooleanVar, DoubleVar, IntVar
from tkinter import scrolledtext

# ── Suppress harmless "main thread is not in main loop" errors ──────────
# Python's GC may finalize tkinter Variables/Images from a background thread,
# causing RuntimeError in their __del__. This is cosmetic and does not affect
# functionality, but it spams the console.  Patch __del__ to silence it.
_orig_variable_del = tk.Variable.__del__

def _safe_variable_del(self):
    try:
        _orig_variable_del(self)
    except RuntimeError:
        pass

tk.Variable.__del__ = _safe_variable_del

try:
    _orig_image_del = tk.Image.__del__

    def _safe_image_del(self):
        try:
            _orig_image_del(self)
        except RuntimeError:
            pass

    tk.Image.__del__ = _safe_image_del
except AttributeError:
    pass  # Some tkinter builds don't expose Image.__del__
# ────────────────────────────────────────────────────────────────────────

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker

if not hasattr(np, "trapezoid") and hasattr(np, "trapz"):
    setattr(np, "trapezoid", np.trapz)
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.tree import plot_tree
from lifelines import KaplanMeierFitter

from MATLAB_filter_component import FilterComponent

RandomSurvivalForest: Any = None
concordance_index_censored: Any = None
concordance_index_ipcw: Any = None
cumulative_dynamic_auc: Any = None
brier_score: Any = None
integrated_brier_score: Any = None
Surv: Any = None

try:
    RandomSurvivalForest = getattr(importlib.import_module("sksurv.ensemble"), "RandomSurvivalForest")
    sksurv_metrics = importlib.import_module("sksurv.metrics")
    concordance_index_censored = getattr(sksurv_metrics, "concordance_index_censored")
    concordance_index_ipcw = getattr(sksurv_metrics, "concordance_index_ipcw", None)
    cumulative_dynamic_auc = getattr(sksurv_metrics, "cumulative_dynamic_auc", None)
    brier_score = getattr(sksurv_metrics, "brier_score", None)
    integrated_brier_score = getattr(sksurv_metrics, "integrated_brier_score", None)
    Surv = getattr(importlib.import_module("sksurv.util"), "Surv")
    SKSURV_AVAILABLE = True
    SKSURV_IMPORT_ERROR = ""
except Exception as exc:
    SKSURV_AVAILABLE = False
    SKSURV_IMPORT_ERROR = str(exc)


class ScrolledFrame(ttk.Frame):
    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.canvas = tk.Canvas(self, highlightthickness=0, bd=0)
        self.interior = ttk.Frame(self.canvas)
        self.v_scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.h_scrollbar = ttk.Scrollbar(self, orient="horizontal", command=self.canvas.xview)
        self.canvas.configure(yscrollcommand=self.v_scrollbar.set, xscrollcommand=self.h_scrollbar.set)
        self.v_scrollbar.pack(side="right", fill="y")
        self.h_scrollbar.pack(side="bottom", fill="x")
        self.canvas.pack(side="left", fill="both", expand=True)
        self.canvas.create_window((0, 0), window=self.interior, anchor="nw")
        self.interior.bind("<Configure>", self._on_interior_configure)

    def _on_interior_configure(self, _event=None):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))


class RSFTab(ttk.Frame):
    def __init__(self, notebook, main_app_instance=None):
        super().__init__(notebook)
        self.main_app_instance = main_app_instance
        self.data = None
        self.base_data = None
        self.model = None
        self.results = None
        self.feature_importance_df = pd.DataFrame()
        self.latest_prediction_df = None
        self.latest_survival_profiles = []
        self.latest_calibration_df = pd.DataFrame()
        self.latest_brier_df = pd.DataFrame()
        self.latest_eval_time = None
        self.latest_fit_dataframe = None
        self.latest_duration_col = None
        self.latest_event_col = None
        self.latest_covariates = []
        self.latest_encoded_columns = []
        self.latest_report_text = ""
        self.latest_tuning_summary = ""
        self.shared_metadata = {}
        self.current_shared_filter_summary = []
        self.using_shared_dataset = True
        self.saved_models = []
        self.active_saved_model_index = None
        self._plot_text_overrides = {}
        self.variable_configs = {}
        self._tuning_progress_dialog = None
        self._loading_model_in_progress = False
        self._auto_tuning_in_progress = False
        self._tuning_progress_var = None
        self._tuning_progress_note_var = None
        self._tuning_progress_bar = None
        self._tuning_cancel_button = None
        self._tuning_pause_button = None
        self._tuning_skip_button = None
        self._tuning_zoom_button = None
        self._tuning_current_model_var = None
        self._tuning_best_metric_var = None
        self._tuning_best_row_var = None
        self._tuning_autoscroll_var = None
        self._tuning_history_text = None
        self._tuning_history_meta_var = None
        self._tuning_cancel_requested = False
        self._tuning_pause_requested = False
        self._tuning_skip_scope_requested = False
        self._scope_color_map = {}
        self._scope_palette = [
            "#2563EB", "#16A34A", "#DC2626", "#D97706", "#7C3AED",
            "#0891B2", "#BE123C", "#4D7C0F", "#0F766E", "#B45309",
        ]
        self._persistent_covariates_pending = []
        self._persistent_duration_pending = ""
        self._persistent_event_pending = ""
        self._max_cancel_autotune_snapshots = 80
        self._live_scatter_refresh_every = 1
        self._live_scatter_last_refresh_completed = -1
        self._live_scatter_fast_mode_threshold = 450
        self._live_scatter_max_draw_groups = 24
        self._live_scatter_hover_limit = 600
        self._live_scatter_filter_cb = None
        self._live_scatter_marker_mode_var = None
        self._live_scatter_heatmap_var = None
        self._live_scatter_heat_bins_var = None
        self._live_scatter_apply_gap_filters_var = None
        self._live_scatter_use_global_gap_var = None
        self._live_scatter_gap_global_var = None
        self._live_scatter_gap_oob_cv_var = None
        self._live_scatter_gap_oob_test_var = None
        self._live_scatter_gap_cv_test_var = None
        self._live_scatter_popout_dialog = None
        self._live_scatter_popout_fig = None
        self._live_scatter_popout_canvas = None
        self._live_scatter_popout_closing = False
        self._live_scatter_advanced_filter_fields = []
        self._live_scatter_advanced_filters = {}
        self._live_scatter_filter_values = []
        self._live_scatter_all_records = []
        self._live_scatter_pulse_seen_uids = set()
        self._live_scatter_marker_symbols = [
            "o", "s", "^", "D", "P", "v", "<", ">", "h", "8", "p"
        ]
        self.categorical_compare_display_map = {
            "all": "Cada categoría vs referencia",
            "one_vs_rest": "Dicotómica: elegida vs resto",
            "quantitative": "Mantener como cuantitativa",
        }
        self.categorical_compare_reverse_map = {
            display: internal for internal, display in self.categorical_compare_display_map.items()
        }

        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.preproc_tab = ttk.Frame(self.notebook)
        self.modeling_tab = ttk.Frame(self.notebook)
        self.results_tab = ttk.Frame(self.notebook)
        self.graphs_tab = ttk.Frame(self.notebook)

        self.notebook.add(self.preproc_tab, text="1. Preprocesamiento")
        self.notebook.add(self.modeling_tab, text="2. Modelado RSF")
        self.notebook.add(self.results_tab, text="3. Resultados")
        self.notebook.add(self.graphs_tab, text="4. Gráficas")

        self._create_preproc_widgets()
        self._create_modeling_widgets()
        self._create_results_widgets()
        self._create_graphs_widgets()

        if not SKSURV_AVAILABLE:
            self.results_text.insert(
                tk.END,
                "RSF disponible en modo de preparación.\n\n"
                "Para entrenar el modelo necesitas instalar `scikit-survival`.\n"
                f"Detalle de importación: {SKSURV_IMPORT_ERROR}\n",
            )

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------
    def _create_preproc_widgets(self):
        load_frame = ttk.LabelFrame(self.preproc_tab, text="Cargar Datos")
        load_frame.pack(fill=tk.X, padx=10, pady=10)

        ttk.Button(load_frame, text="Cargar Archivo (Excel/CSV)", command=self.load_data).pack(
            side=tk.LEFT, padx=5, pady=5
        )
        self.file_label = ttk.Label(load_frame, text="Esperando dataset compartido o archivo local.")
        self.file_label.pack(side=tk.LEFT, padx=5, pady=5)

        info_frame = ttk.LabelFrame(self.preproc_tab, text="Estado RSF")
        info_frame.pack(fill=tk.X, padx=10, pady=(0, 10))

        availability_text = (
            "`scikit-survival` disponible ✅" if SKSURV_AVAILABLE else f"`scikit-survival` no disponible ⚠️ ({SKSURV_IMPORT_ERROR})"
        )
        ttk.Label(info_frame, text=availability_text, foreground=("darkgreen" if SKSURV_AVAILABLE else "darkorange")).pack(
            anchor="w", padx=8, pady=6
        )

        filter_frame = ttk.LabelFrame(self.preproc_tab, text="Filtros")
        filter_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.filter_component = FilterComponent(filter_frame)
        self.filter_component.pack(fill=tk.BOTH, expand=True)

    def _create_modeling_widgets(self):
        container = ScrolledFrame(self.modeling_tab)
        container.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)
        parent = container.interior

        vars_frame = ttk.LabelFrame(parent, text="Selección de Variables")
        vars_frame.pack(fill=tk.X, padx=8, pady=8)
        vars_frame.columnconfigure(1, weight=1)

        ttk.Label(vars_frame, text="Variable de tiempo:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.duration_var = StringVar()
        self.duration_combo = ttk.Combobox(vars_frame, textvariable=self.duration_var, state="readonly")
        self.duration_combo.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

        ttk.Label(vars_frame, text="Variable de evento:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.event_var = StringVar()
        self.event_combo = ttk.Combobox(vars_frame, textvariable=self.event_var, state="readonly")
        self.event_combo.grid(row=1, column=1, padx=5, pady=5, sticky="ew")

        ttk.Label(vars_frame, text="Covariables:").grid(row=2, column=0, padx=5, pady=5, sticky="nw")
        self.covariates_listbox = tk.Listbox(vars_frame, selectmode=tk.MULTIPLE, height=8, exportselection=False)
        self.covariates_listbox.grid(row=2, column=1, padx=5, pady=5, sticky="ew")
        ttk.Label(
            vars_frame,
            text="Clic simple para marcar/desmarcar covariables.",
            foreground="#666666",
        ).grid(row=3, column=1, padx=5, pady=(0, 2), sticky="w")
        ttk.Button(vars_frame, text="Configurar categorías...", command=self.open_categorical_config_dialog).grid(
            row=4, column=1, padx=5, pady=(0, 8), sticky="e"
        )

        prep_frame = ttk.LabelFrame(parent, text="Opciones de Preprocesamiento")
        prep_frame.pack(fill=tk.X, padx=8, pady=8)
        for col_idx in range(4):
            prep_frame.columnconfigure(col_idx, weight=1)

        self.missing_strategy_var = StringVar(value="Imputar mediana/moda")
        self.drop_first_var = BooleanVar(value=True)
        self.test_size_var = DoubleVar(value=0.25)
        self.stratify_event_var = BooleanVar(value=True)

        ttk.Label(prep_frame, text="Faltantes:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        ttk.Combobox(
            prep_frame,
            textvariable=self.missing_strategy_var,
            values=["Imputar mediana/moda", "Eliminar filas incompletas"],
            state="readonly",
        ).grid(row=0, column=1, padx=5, pady=5, sticky="ew")

        ttk.Checkbutton(prep_frame, text="One-hot drop_first", variable=self.drop_first_var).grid(
            row=0, column=2, padx=5, pady=5, sticky="w"
        )
        ttk.Checkbutton(prep_frame, text="Estratificar por evento", variable=self.stratify_event_var).grid(
            row=0, column=3, padx=5, pady=5, sticky="w"
        )

        ttk.Label(prep_frame, text="Proporción test:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        ttk.Entry(prep_frame, textvariable=self.test_size_var, width=10).grid(row=1, column=1, padx=5, pady=5, sticky="w")
        ttk.Label(prep_frame, text="Usa 0 para entrenar con todos los casos", foreground="#555555").grid(
            row=1, column=2, columnspan=2, padx=5, pady=5, sticky="w"
        )

        ttk.Label(prep_frame, text="Métrica optimización:").grid(row=2, column=0, padx=5, pady=5, sticky="w")
        self.optimization_metric_var = StringVar(value="Harrell C-index")
        self.optimization_metric_combo = ttk.Combobox(
            prep_frame,
            textvariable=self.optimization_metric_var,
            values=["Harrell C-index", "Uno C-index", "IBS", "Brier Score"],
            state="readonly",
            width=18,
        )
        self.optimization_metric_combo.grid(row=2, column=1, padx=5, pady=5, sticky="w")
        self.optimization_metric_combo.bind("<<ComboboxSelected>>", lambda _event: self._on_optimization_metric_change())

        self.tau_mode_var = StringVar(value="Percentil 90")
        self.tau_manual_var = StringVar(value="")
        self.tau_label_widget = ttk.Label(prep_frame, text="τ (tau) Uno:")
        self.tau_mode_combo = ttk.Combobox(
            prep_frame,
            textvariable=self.tau_mode_var,
            values=["Percentil 90", "Último caso", "Manual"],
            state="readonly",
            width=16,
        )
        self.tau_mode_combo.bind("<<ComboboxSelected>>", lambda _event: self._on_tau_mode_change())
        self.tau_manual_entry = ttk.Entry(prep_frame, textvariable=self.tau_manual_var, width=10)
        self.tau_help_widget = ttk.Label(prep_frame, text="(Solo aplica a Uno C-index)", foreground="#555555")

        self.tau_label_widget.grid(row=3, column=0, padx=5, pady=5, sticky="w")
        self.tau_mode_combo.grid(row=3, column=1, padx=5, pady=5, sticky="w")
        self.tau_manual_entry.grid(row=3, column=2, padx=5, pady=5, sticky="w")
        self.tau_help_widget.grid(row=3, column=3, padx=5, pady=5, sticky="w")
        self._on_optimization_metric_change()

        hyper_frame = ttk.LabelFrame(parent, text="Hiperparámetros RSF")
        hyper_frame.pack(fill=tk.X, padx=8, pady=8)
        for col_idx in range(4):
            hyper_frame.columnconfigure(col_idx, weight=1)

        self.preset_var = StringVar(value="Balanceado")
        self.n_estimators_var = IntVar(value=300)
        self.max_features_var = StringVar(value="sqrt")
        self.max_features_manual_var = StringVar(value="")
        self.min_samples_split_var = IntVar(value=10)
        self.min_samples_leaf_var = IntVar(value=5)
        self.max_depth_var = StringVar(value="")
        self.max_leaf_nodes_var = StringVar(value="")
        self.bootstrap_var = BooleanVar(value=True)
        self.max_samples_var = StringVar(value="")
        self.oob_score_var = BooleanVar(value=True)
        self.n_jobs_var = IntVar(value=-1)
        self.random_state_var = StringVar(value="42")
        self.cv_enabled_var = BooleanVar(value=True)
        self.cv_folds_var = IntVar(value=5)

        ttk.Label(hyper_frame, text="Preset:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        ttk.Combobox(
            hyper_frame,
            textvariable=self.preset_var,
            values=["Rápido", "Balanceado", "Robusto"],
            state="readonly",
            width=16,
        ).grid(row=0, column=1, padx=5, pady=5, sticky="w")
        ttk.Button(hyper_frame, text="Aplicar preset", command=self._apply_preset).grid(row=0, column=2, padx=5, pady=5, sticky="w")

        ttk.Label(hyper_frame, text="n_estimators:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        ttk.Entry(hyper_frame, textvariable=self.n_estimators_var, width=12).grid(row=1, column=1, padx=5, pady=5, sticky="w")

        ttk.Label(hyper_frame, text="max_features:").grid(row=1, column=2, padx=5, pady=5, sticky="w")
        ttk.Combobox(
            hyper_frame,
            textvariable=self.max_features_var,
            values=["sqrt", "log2", "all", "0.5", "manual"],
            state="readonly",
            width=12,
        ).grid(row=1, column=3, padx=5, pady=5, sticky="w")

        ttk.Label(hyper_frame, text="manual max_features:").grid(row=2, column=0, padx=5, pady=5, sticky="w")
        ttk.Entry(hyper_frame, textvariable=self.max_features_manual_var, width=12).grid(row=2, column=1, padx=5, pady=5, sticky="w")

        ttk.Label(hyper_frame, text="min_samples_split:").grid(row=2, column=2, padx=5, pady=5, sticky="w")
        ttk.Entry(hyper_frame, textvariable=self.min_samples_split_var, width=12).grid(row=2, column=3, padx=5, pady=5, sticky="w")

        ttk.Label(hyper_frame, text="min_samples_leaf:").grid(row=3, column=0, padx=5, pady=5, sticky="w")
        ttk.Entry(hyper_frame, textvariable=self.min_samples_leaf_var, width=12).grid(row=3, column=1, padx=5, pady=5, sticky="w")

        ttk.Label(hyper_frame, text="max_depth:").grid(row=3, column=2, padx=5, pady=5, sticky="w")
        ttk.Entry(hyper_frame, textvariable=self.max_depth_var, width=12).grid(row=3, column=3, padx=5, pady=5, sticky="w")

        ttk.Label(hyper_frame, text="max_leaf_nodes:").grid(row=4, column=0, padx=5, pady=5, sticky="w")
        ttk.Entry(hyper_frame, textvariable=self.max_leaf_nodes_var, width=12).grid(row=4, column=1, padx=5, pady=5, sticky="w")

        ttk.Label(hyper_frame, text="max_samples:").grid(row=4, column=2, padx=5, pady=5, sticky="w")
        ttk.Entry(hyper_frame, textvariable=self.max_samples_var, width=12).grid(row=4, column=3, padx=5, pady=5, sticky="w")

        ttk.Checkbutton(hyper_frame, text="bootstrap", variable=self.bootstrap_var).grid(row=5, column=0, padx=5, pady=5, sticky="w")
        ttk.Checkbutton(hyper_frame, text="oob_score", variable=self.oob_score_var).grid(row=5, column=1, padx=5, pady=5, sticky="w")
        ttk.Checkbutton(hyper_frame, text="Calcular CV C-index", variable=self.cv_enabled_var).grid(row=5, column=2, padx=5, pady=5, sticky="w")

        ttk.Label(hyper_frame, text="n_jobs:").grid(row=6, column=0, padx=5, pady=5, sticky="w")
        ttk.Entry(hyper_frame, textvariable=self.n_jobs_var, width=12).grid(row=6, column=1, padx=5, pady=5, sticky="w")

        ttk.Label(hyper_frame, text="Semillas (ej: 42,7,99):").grid(row=6, column=2, padx=5, pady=5, sticky="w")
        ttk.Entry(hyper_frame, textvariable=self.random_state_var, width=16).grid(row=6, column=3, padx=5, pady=5, sticky="w")

        ttk.Label(hyper_frame, text="CV folds:").grid(row=7, column=0, padx=5, pady=5, sticky="w")
        ttk.Entry(hyper_frame, textvariable=self.cv_folds_var, width=12).grid(row=7, column=1, padx=5, pady=5, sticky="w")
        self.cv_metric_var = StringVar(value="C-Uno (IPCW)")
        ttk.Label(hyper_frame, text="Métrica CV:").grid(row=7, column=2, padx=5, pady=5, sticky="w")
        ttk.Combobox(
            hyper_frame,
            textvariable=self.cv_metric_var,
            values=["C clásico", "C-Uno (IPCW)", "C-Antolini (Ctd)"],
            state="readonly",
            width=16,
        ).grid(row=7, column=3, padx=5, pady=5, sticky="w")
        self.oob_cindex_var = StringVar(value="Harrell (nativo)")
        ttk.Label(hyper_frame, text="C-index OOB:").grid(row=8, column=0, padx=5, pady=5, sticky="w")
        ttk.Combobox(
            hyper_frame,
            textvariable=self.oob_cindex_var,
            values=["Harrell (nativo)", "Uno (IPCW)", "Antolini (Ctd)"],
            state="readonly",
            width=16,
        ).grid(row=8, column=1, padx=5, pady=5, sticky="w")
        ttk.Label(
            hyper_frame,
            text="Regla: min_samples_split = int(min_samples_leaf × M), M ≥ 2.0",
            foreground="#666666",
        ).grid(row=9, column=0, columnspan=4, padx=5, pady=(0, 6), sticky="w")

        actions_frame = ttk.Frame(parent)
        actions_frame.pack(fill=tk.X, padx=8, pady=(4, 10))

        ttk.Label(actions_frame, text="Perfil tuning:").pack(side=tk.LEFT, padx=(5, 4))
        self.tuning_profile_var = StringVar(value="General")
        self.tuning_profile_combo = ttk.Combobox(
            actions_frame,
            textvariable=self.tuning_profile_var,
            values=["General", "Pocos datos (<200)", "Mediano (200-500)", "Pesado (>500)", "Manual"],
            state="readonly",
            width=22,
        )
        self.tuning_profile_combo.pack(side=tk.LEFT, padx=(0, 8))

        self.robust_vimp_mode_var = StringVar(value="Permisivo (IC95% sup > 0)")
        ttk.Label(actions_frame, text="Limpieza VIMP:").pack(side=tk.LEFT, padx=(4, 4))
        ttk.Combobox(
            actions_frame,
            textvariable=self.robust_vimp_mode_var,
            values=[
                "Permisivo (IC95% sup > 0)",
                "Estricto (IC95% inf > 0)",
                "Híbrido (2 de 3)",
            ],
            state="readonly",
            width=24,
        ).pack(side=tk.LEFT, padx=(0, 6))

        self.corr_prune_threshold_var = StringVar(value="0.85")
        
        # Nuevas variables para autotuning árboles
        self.auto_tree_start_var = StringVar(value="100")
        self.auto_tree_step_var = StringVar(value="100")
        self.auto_tree_max_var = StringVar(value="1200")
        
        # Nuevas variables para screening de semillas
        self.early_stopping_enabled_var = BooleanVar(value=False)
        self.early_stopping_seeds_var = StringVar(value="2")
        self.early_stopping_gap_var = StringVar(value="0.01")
        self.early_stopping_pass_stops_var = BooleanVar(value=False)  # parar al pasar (semiincompleto)
        self.early_stopping_impossible_stops_var = BooleanVar(value=False)  # parar si ya es inalcanzable
        
        ttk.Label(actions_frame, text="Poda correlación:").pack(side=tk.LEFT, padx=(2, 4))
        ttk.Combobox(
            actions_frame,
            textvariable=self.corr_prune_threshold_var,
            values=["Sin poda", "0.85", "0.90", "0.95"],
            state="readonly",
            width=10,
        ).pack(side=tk.LEFT, padx=(0, 6))

        # UI para los límites automáticos de árboles
        ttk.Label(actions_frame, text=" | Auto Árboles (Ini:").pack(side=tk.LEFT)
        ttk.Entry(actions_frame, textvariable=self.auto_tree_start_var, width=5).pack(side=tk.LEFT)
        ttk.Label(actions_frame, text="Paso:").pack(side=tk.LEFT)
        ttk.Entry(actions_frame, textvariable=self.auto_tree_step_var, width=4).pack(side=tk.LEFT)
        ttk.Label(actions_frame, text="Máx):").pack(side=tk.LEFT)
        ttk.Entry(actions_frame, textvariable=self.auto_tree_max_var, width=5).pack(side=tk.LEFT, padx=(0, 6))

        # --- Screening de Semillas Frame ---
        screening_frame = ttk.LabelFrame(parent, text="Screening de Semillas (Early Stopping)")
        screening_frame.pack(fill=tk.X, padx=8, pady=(4, 10))

        def _toggle_screening():
            state = "normal" if self.early_stopping_enabled_var.get() else "disabled"
            _es_seeds_entry.configure(state=state)
            _es_gap_entry.configure(state=state)
            _es_pass_stops_cb.configure(state=state)
            _es_impossible_cb.configure(state=state)

        ttk.Checkbutton(
            screening_frame,
            text="Activar Screening",
            variable=self.early_stopping_enabled_var,
            command=_toggle_screening
        ).pack(side=tk.LEFT, padx=(5, 10))

        ttk.Label(screening_frame, text="Semillas:").pack(side=tk.LEFT)
        _es_seeds_entry = ttk.Entry(screening_frame, textvariable=self.early_stopping_seeds_var, width=5, state="disabled")
        _es_seeds_entry.pack(side=tk.LEFT, padx=(2, 10))

        ttk.Label(screening_frame, text="Dif. máx.:").pack(side=tk.LEFT)
        _es_gap_entry = ttk.Entry(screening_frame, textvariable=self.early_stopping_gap_var, width=6, state="disabled")
        _es_gap_entry.pack(side=tk.LEFT, padx=(2, 10))

        _es_pass_stops_cb = ttk.Checkbutton(
            screening_frame,
            text="Parar al pasar umbral (semiincompleto)",
            variable=self.early_stopping_pass_stops_var,
            state="disabled",
        )
        _es_pass_stops_cb.pack(side=tk.LEFT, padx=(0, 5))

        _es_impossible_cb = ttk.Checkbutton(
            screening_frame,
            text="Parar si inalcanzable (en cada semilla)",
            variable=self.early_stopping_impossible_stops_var,
            state="disabled",
        )
        _es_impossible_cb.pack(side=tk.LEFT, padx=(0, 5))
        _toggle_screening()
        # ------------------------------------

        self.manual_trees_var = StringVar(value="100,200,300,500")
        self.manual_max_features_grid_var = StringVar(value="sqrt,log2,0.5,all")
        self.manual_min_leaf_grid_var = StringVar(value="2,3,4,5,6,7,8,9,10")
        self.manual_split_mult_grid_var = StringVar(value="2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22")
        self.manual_max_depth_grid_var = StringVar(value="none,5,10")
        self.manual_max_leaf_nodes_grid_var = StringVar(value="none,15,30")
        self.manual_max_samples_grid_var = StringVar(value="")
        self.tuning_models_mode_var = StringVar(value="Multivariado")
        self.tvt_min_covariates_var = StringVar(value="1")
        self.tvt_max_covariates_var = StringVar(value="")
        self.tvt_max_combinations_var = StringVar(value="120")
        self.tvt_required_covariates_var = StringVar(value="")

        manual_tuning_frame = ttk.LabelFrame(parent, text="Tuning manual RSF (valores personalizados)")
        manual_tuning_frame.pack(fill=tk.X, padx=8, pady=(0, 8))
        for col_idx in range(4):
            manual_tuning_frame.columnconfigure(col_idx, weight=1)

        ttk.Label(manual_tuning_frame, text="Árboles:").grid(row=0, column=0, padx=5, pady=4, sticky="w")
        
        _trees_frame = ttk.Frame(manual_tuning_frame)
        _trees_frame.grid(row=0, column=1, padx=5, pady=4, sticky="ew")
        
        ttk.Entry(_trees_frame, textvariable=self.manual_trees_var, width=12).pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        ttk.Label(_trees_frame, text=" Mín:").pack(side=tk.LEFT)
        _trees_min = ttk.Entry(_trees_frame, width=4)
        _trees_min.insert(0, "100")
        _trees_min.pack(side=tk.LEFT)
        
        ttk.Label(_trees_frame, text=" Máx:").pack(side=tk.LEFT)
        _trees_max = ttk.Entry(_trees_frame, width=4)
        _trees_max.insert(0, "1200")
        _trees_max.pack(side=tk.LEFT)
        
        ttk.Label(_trees_frame, text=" Paso:").pack(side=tk.LEFT)
        _trees_step = ttk.Entry(_trees_frame, width=4)
        _trees_step.insert(0, "100")
        _trees_step.pack(side=tk.LEFT)
        
        def _gen_trees():
            try:
                vmin = int(_trees_min.get())
                vmax = int(_trees_max.get())
                vstep = int(_trees_step.get())
                if vstep > 0 and vmax >= vmin:
                    vals = list(range(vmin, vmax + 1, vstep))
                    self.manual_trees_var.set(",".join(map(str, vals)))
            except Exception:
                pass
                
        ttk.Button(_trees_frame, text="▶", width=2, command=_gen_trees).pack(side=tk.LEFT, padx=(2, 0))
        ttk.Label(manual_tuning_frame, text="max_features:").grid(row=0, column=2, padx=5, pady=4, sticky="w")
        ttk.Entry(manual_tuning_frame, textvariable=self.manual_max_features_grid_var, width=24).grid(row=0, column=3, padx=5, pady=4, sticky="ew")

        ttk.Label(manual_tuning_frame, text="min_leaf:").grid(row=1, column=0, padx=5, pady=4, sticky="w")
        ttk.Entry(manual_tuning_frame, textvariable=self.manual_min_leaf_grid_var, width=24).grid(row=1, column=1, padx=5, pady=4, sticky="ew")
        ttk.Label(manual_tuning_frame, text="Mult. split (M):").grid(row=1, column=2, padx=5, pady=4, sticky="w")
        ttk.Entry(manual_tuning_frame, textvariable=self.manual_split_mult_grid_var, width=24).grid(row=1, column=3, padx=5, pady=4, sticky="ew")

        ttk.Label(manual_tuning_frame, text="max_depth:").grid(row=2, column=0, padx=5, pady=4, sticky="w")
        ttk.Entry(manual_tuning_frame, textvariable=self.manual_max_depth_grid_var, width=24).grid(row=2, column=1, padx=5, pady=4, sticky="ew")
        ttk.Label(manual_tuning_frame, text="max_leaf_nodes:").grid(row=2, column=2, padx=5, pady=4, sticky="w")
        ttk.Entry(manual_tuning_frame, textvariable=self.manual_max_leaf_nodes_grid_var, width=24).grid(row=2, column=3, padx=5, pady=4, sticky="ew")

        ttk.Label(manual_tuning_frame, text="max_samples:").grid(row=3, column=0, padx=5, pady=4, sticky="w")
        ttk.Entry(manual_tuning_frame, textvariable=self.manual_max_samples_grid_var, width=24).grid(row=3, column=1, padx=5, pady=4, sticky="ew")
        ttk.Button(
            manual_tuning_frame,
            text="Seleccionar con Ctrl/Shift...",
            command=self._open_manual_tuning_selector_dialog,
        ).grid(row=3, column=2, columnspan=2, padx=5, pady=4, sticky="e")

        ttk.Label(manual_tuning_frame, text="Modelos a evaluar:").grid(row=4, column=0, padx=5, pady=4, sticky="w")
        ttk.Combobox(
            manual_tuning_frame,
            textvariable=self.tuning_models_mode_var,
            values=["Multivariado", "Univariado", "Ambos", "Todos contra todos"],
            state="readonly",
            width=24,
        ).grid(row=4, column=1, padx=5, pady=4, sticky="w")

        self.tuning_progress_metric_var = StringVar(value="C-Uno (IPCW)")
        ttk.Label(manual_tuning_frame, text="Métrica en progreso:").grid(row=4, column=2, padx=5, pady=4, sticky="w")
        ttk.Combobox(
            manual_tuning_frame,
            textvariable=self.tuning_progress_metric_var,
            values=["CV", "C-Uno (IPCW)", "C-test", "OOB", "BSS", "IBS"],
            state="readonly",
            width=24,
        ).grid(row=4, column=3, padx=5, pady=4, sticky="w")

        ttk.Label(manual_tuning_frame, text="Todos vs Todos - mín. covariables:").grid(row=5, column=0, padx=5, pady=4, sticky="w")
        ttk.Combobox(
            manual_tuning_frame,
            textvariable=self.tvt_min_covariates_var,
            values=["1", "2", "3", "4", "5", "6", "7", "8", "10", "12", "15", "20"],
            state="readonly",
            width=24,
        ).grid(row=5, column=1, padx=5, pady=4, sticky="w")

        ttk.Label(manual_tuning_frame, text="Obligatorias: A,B=Y; (A|B)=O:").grid(row=5, column=2, padx=5, pady=4, sticky="w")
        ttk.Entry(manual_tuning_frame, textvariable=self.tvt_required_covariates_var, width=24).grid(row=5, column=3, padx=5, pady=4, sticky="ew")

        ttk.Label(manual_tuning_frame, text="Todos vs Todos - máx. covariables:").grid(row=6, column=0, padx=5, pady=4, sticky="w")
        ttk.Combobox(
            manual_tuning_frame,
            textvariable=self.tvt_max_covariates_var,
            values=["", "2", "3", "4", "5", "6", "7", "8", "10", "12", "15", "20"],
            state="readonly",
            width=24,
        ).grid(row=6, column=1, padx=5, pady=4, sticky="w")

        ttk.Label(manual_tuning_frame, text="Máx. combinaciones a generar:").grid(row=6, column=2, padx=5, pady=4, sticky="w")
        ttk.Entry(manual_tuning_frame, textvariable=self.tvt_max_combinations_var, width=24).grid(row=6, column=3, padx=5, pady=4, sticky="ew")

        ttk.Button(
            manual_tuning_frame,
            text="Usar selección de X",
            command=self._set_tvt_required_covariates_from_selection,
        ).grid(row=7, column=2, padx=5, pady=4, sticky="e")
        ttk.Button(
            manual_tuning_frame,
            text="Limpiar obligatorias",
            command=lambda: self.tvt_required_covariates_var.set(""),
        ).grid(row=7, column=3, padx=5, pady=4, sticky="w")

        ttk.Label(
            manual_tuning_frame,
            text="Obligatorias: coma=todos-obligatorios (A,B), pipe=OR (A|B)=al-menos-uno, se combinan: A|B,C = uno de A|B Y también C. Paréntesis opcionales. Ej: EIMA|RepetidosCAG,ecog",
            foreground="#666666",
        ).grid(row=8, column=0, columnspan=4, padx=5, pady=(2, 4), sticky="w")

        self.auto_tuning_status_var = StringVar(value="")
        ttk.Button(actions_frame, text="Correlaciones", command=self.show_correlation_diagnostics).pack(side=tk.RIGHT, padx=5)
        ttk.Button(actions_frame, text="Optimización Robusta", command=self.run_robust_optimization).pack(side=tk.RIGHT, padx=5)
        ttk.Button(actions_frame, text="Opt. Robusta (Nested CV)", command=self.run_nested_cv_optimization).pack(side=tk.RIGHT, padx=5)
        ttk.Button(actions_frame, text="Tuning automático", command=self.run_auto_tuning).pack(side=tk.RIGHT, padx=5)
        ttk.Button(actions_frame, text="Ejecutar RSF", command=self.run_model).pack(side=tk.RIGHT, padx=5)

        history_frame = ttk.LabelFrame(parent, text="Modelos RSF temporales en memoria (comparación)")
        history_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=(4, 8))

        ttk.Label(
            history_frame,
            text="Nota: esta lista es temporal. Solo se guarda a disco si presionas 'Guardar a archivo...'.",
            foreground="#666666",
        ).pack(anchor="w", padx=6, pady=(4, 2))
        ttk.Label(
            history_frame,
            text="Tip: en la tabla de modelos sí puedes usar Ctrl/Shift para selección múltiple.",
            foreground="#666666",
        ).pack(anchor="w", padx=6, pady=(0, 4))

        history_columns = ("id", "apto", "seed", "scope", "mode", "trees", "features", "leaf", "split", "max_depth", "max_leaf_nodes", "max_samples", "test_prop", "cv", "c_train", "c_test", "oob",
                           "c_uno", "c_antolini", "tau", "ibs", "ibs_km", "bss",
                           "c_q25", "c_q50", "c_q75", "brier_q25", "brier_q50", "brier_q75",
                           "auroc_q25", "auroc_q50", "auroc_q75")
        self.saved_models_tree = ttk.Treeview(history_frame, columns=history_columns, show="headings", height=7, selectmode="extended")
        _rsf_col_headings = {
            "id": "ID", "apto": "Apto", "seed": "Seed", "scope": "Scope", "mode": "Modo", "trees": "Árboles", "features": "max_features", "leaf": "min_leaf",
            "split": "min_split", "max_depth": "max_depth", "max_leaf_nodes": "max_leaf_nodes", "max_samples": "max_samples", "test_prop": "Test %", "cv": "CV", "c_train": "C-train",
            "c_test": "C-test", "oob": "OOB", "c_uno": "C-Uno (IPCW)",
            "c_antolini": "C-Antolini (Ctd)", "tau": "τ (tau)", "ibs": "IBS",
            "ibs_km": "IBS KM (nulo)", "bss": "BSS",
            "c_q25": "C@Q25", "c_q50": "C@Q50", "c_q75": "C@Q75",
            "brier_q25": "Brier@Q25", "brier_q50": "Brier@Q50", "brier_q75": "Brier@Q75",
            "auroc_q25": "AUC@Q25", "auroc_q50": "AUC@Q50", "auroc_q75": "AUC@Q75",
        }
        for col_id, heading_text in _rsf_col_headings.items():
            self.saved_models_tree.heading(col_id, text=heading_text, command=lambda c=col_id: self._sort_saved_models_tree(c))

        _rsf_col_widths = {"id": 50, "apto": 45, "seed": 60, "scope": 280, "mode": 95, "trees": 70, "features": 100, "leaf": 70, "split": 75, "max_depth": 85, "max_leaf_nodes": 110, "max_samples": 95, "test_prop": 75,
                           "cv": 170, "c_train": 170, "c_test": 170, "oob": 80,
                           "c_uno": 110, "c_antolini": 120, "tau": 80, "ibs": 160,
                           "ibs_km": 95, "bss": 160,
                           "c_q25": 80, "c_q50": 80, "c_q75": 80,
                           "brier_q25": 85, "brier_q50": 85, "brier_q75": 85,
                           "auroc_q25": 80, "auroc_q50": 80, "auroc_q75": 80}
        for col_name in history_columns:
            w = _rsf_col_widths[col_name]
            self.saved_models_tree.column(col_name, width=w, anchor="center", stretch=(col_name in ("features", "scope")))

        self._tree_column_config = {col: {"visible": True, "width": _rsf_col_widths[col], "heading": _rsf_col_headings[col]} for col in history_columns}
        self._restore_saved_layout("rsf_saved_models", self._tree_column_config)
        self._apply_tree_column_layout()
        self._register_saved_layout("rsf_saved_models", self._tree_column_config, self.saved_models_tree)
        self.saved_models_tree.pack(fill=tk.BOTH, expand=True, side=tk.TOP)
        self.saved_models_tree.bind("<Double-1>", self._on_saved_models_tree_double_click)
        self.saved_models_tree.bind("<Button-3>", self._show_tree_column_menu)

        history_buttons = ttk.Frame(history_frame)
        history_buttons.pack(fill=tk.X, pady=(8, 0))

        self.saved_models_status_var = StringVar(value="Modelos en memoria: 0 | Activo: ninguno")
        ttk.Label(history_buttons, textvariable=self.saved_models_status_var, foreground="navy").pack(side=tk.LEFT, padx=5)
        ttk.Button(history_buttons, text="Guardar TODOS...", command=self._save_all_models_to_file).pack(side=tk.RIGHT, padx=5)
        ttk.Button(history_buttons, text="Guardar a archivo...", command=self._save_selected_model_to_file).pack(side=tk.RIGHT, padx=5)
        ttk.Button(history_buttons, text="Importar archivo...", command=self._import_saved_model_from_file).pack(side=tk.RIGHT, padx=5)
        ttk.Button(history_buttons, text="Cargar seleccionado", command=self._load_selected_saved_model).pack(side=tk.RIGHT, padx=5)
        ttk.Button(history_buttons, text="Eliminar seleccionado", command=self._delete_selected_saved_model).pack(side=tk.RIGHT, padx=5)

        history_buttons2 = ttk.Frame(history_frame)
        history_buttons2.pack(fill=tk.X, pady=(4, 0))
        ttk.Button(history_buttons2, text="🔍 Explorar modelos", command=self._open_model_explorer).pack(side=tk.LEFT, padx=5)
        ttk.Button(history_buttons2, text="Borrar no seleccionados", command=self._delete_non_selected_models).pack(side=tk.LEFT, padx=5)
        ttk.Button(history_buttons2, text="Borrar no aptos", command=self._delete_non_apt_models).pack(side=tk.LEFT, padx=5)
        ttk.Button(history_buttons2, text="Limpiar lista", command=self._clear_saved_models).pack(side=tk.LEFT, padx=5)

    def _create_results_widgets(self):
        self.results_text = scrolledtext.ScrolledText(self.results_tab, wrap=tk.WORD, height=20)
        self.results_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    def _create_graphs_widgets(self):
        ttk.Label(
            self.graphs_tab,
            text="Tip: haz clic en título, nombres de ejes, categorías o leyenda para editar el texto.",
            foreground="#666666",
        ).pack(anchor="w", padx=12, pady=(6, 0))

        self.graphs_notebook = ttk.Notebook(self.graphs_tab)
        self.graphs_notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.importance_tab = ttk.Frame(self.graphs_notebook)
        self.km_tab = ttk.Frame(self.graphs_notebook)
        self.profile_tab = ttk.Frame(self.graphs_notebook)
        self.impact_tab = ttk.Frame(self.graphs_notebook)
        self.calibration_tab = ttk.Frame(self.graphs_notebook)
        self.brier_tab = ttk.Frame(self.graphs_notebook)
        self.tree_tab = ttk.Frame(self.graphs_notebook)
        self.mindepth_tab = ttk.Frame(self.graphs_notebook)
        self.shap_tab = ttk.Frame(self.graphs_notebook)
        self.pdp_tab = ttk.Frame(self.graphs_notebook)
        self.proximity_tab = ttk.Frame(self.graphs_notebook)
        self.graphs_notebook.add(self.importance_tab, text="Importancia de Variables")
        self.graphs_notebook.add(self.km_tab, text="Kaplan-Meier por Riesgo")
        self.graphs_notebook.add(self.profile_tab, text="Efecto vs Tiempo")
        self.graphs_notebook.add(self.impact_tab, text="Riesgo en t")
        self.graphs_notebook.add(self.calibration_tab, text="Calibración")
        self.graphs_notebook.add(self.brier_tab, text="Brier / IBS")
        self.graphs_notebook.add(self.tree_tab, text="Árbol Individual")
        self.graphs_notebook.add(self.mindepth_tab, text="Prof. Mínima")
        self.graphs_notebook.add(self.shap_tab, text="SHAP")
        self.graphs_notebook.add(self.pdp_tab, text="PDP")
        self.graphs_notebook.add(self.proximity_tab, text="Proximidad")

        self.importance_fig = plt.figure(figsize=(6, 4))
        self.importance_canvas = FigureCanvasTkAgg(self.importance_fig, master=self.importance_tab)
        self.importance_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self._register_editable_plot_canvas(self.importance_canvas, "importance")

        self.km_fig = plt.figure(figsize=(6, 4))
        self.km_canvas = FigureCanvasTkAgg(self.km_fig, master=self.km_tab)
        self.km_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self._register_editable_plot_canvas(self.km_canvas, "km")

        profile_controls = ttk.Frame(self.profile_tab)
        profile_controls.pack(fill=tk.X, padx=8, pady=(8, 0))
        ttk.Label(profile_controls, text="Covariable:").pack(side=tk.LEFT, padx=(0, 6))
        self.profile_covariate_var = StringVar()
        self.profile_covariate_combo = ttk.Combobox(profile_controls, textvariable=self.profile_covariate_var, state="readonly", width=22)
        self.profile_covariate_combo.pack(side=tk.LEFT, padx=(0, 8))
        self.profile_covariate_combo.bind("<<ComboboxSelected>>", lambda _event: self.plot_survival_profiles())

        ttk.Label(profile_controls, text="Valores:").pack(side=tk.LEFT, padx=(6, 6))
        self.profile_values_var = StringVar()
        self.profile_values_entry = ttk.Entry(profile_controls, textvariable=self.profile_values_var, width=18)
        self.profile_values_entry.pack(side=tk.LEFT, padx=(0, 8))
        self.profile_values_entry.bind("<Return>", lambda _event: self.plot_survival_profiles())
        self.profile_values_entry.bind("<FocusOut>", lambda _event: self.plot_survival_profiles())

        ttk.Label(profile_controls, text="Otras vars:").pack(side=tk.LEFT, padx=(6, 6))
        self.profile_baseline_var = StringVar()
        self.profile_baseline_entry = ttk.Entry(profile_controls, textvariable=self.profile_baseline_var, width=34)
        self.profile_baseline_entry.pack(side=tk.LEFT, padx=(0, 8))
        self.profile_baseline_entry.bind("<Return>", lambda _event: self.plot_survival_profiles())
        self.profile_baseline_entry.bind("<FocusOut>", lambda _event: self.plot_survival_profiles())

        self.profile_show_ci_var = BooleanVar(value=True)
        ttk.Checkbutton(
            profile_controls,
            text="Mostrar IC",
            variable=self.profile_show_ci_var,
            command=self.plot_survival_profiles,
        ).pack(side=tk.LEFT, padx=(4, 0))
        ttk.Button(profile_controls, text="Actualizar", command=self.plot_survival_profiles).pack(side=tk.LEFT, padx=(4, 0))
        ttk.Label(
            self.profile_tab,
            text="Formato Valores: 10,20,30 | Otras vars: Variable:valor, Variable2:valor2",
            foreground="#666666",
        ).pack(anchor="w", padx=12, pady=(2, 4))

        self.profile_fig = plt.figure(figsize=(6, 4))
        self.profile_canvas = FigureCanvasTkAgg(self.profile_fig, master=self.profile_tab)
        self.profile_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self._register_editable_plot_canvas(self.profile_canvas, "profile")

        impact_controls = ttk.Frame(self.impact_tab)
        impact_controls.pack(fill=tk.X, padx=8, pady=(8, 0))
        ttk.Label(impact_controls, text="Covariable:").pack(side=tk.LEFT, padx=(0, 6))
        self.impact_covariate_var = StringVar()
        self.impact_covariate_combo = ttk.Combobox(impact_controls, textvariable=self.impact_covariate_var, state="readonly", width=22)
        self.impact_covariate_combo.pack(side=tk.LEFT, padx=(0, 8))
        self.impact_covariate_combo.bind("<<ComboboxSelected>>", lambda _event: self.plot_variable_impact())

        ttk.Label(impact_controls, text="Tiempo(s) t:").pack(side=tk.LEFT, padx=(6, 6))
        self.impact_time_var = StringVar()
        self.impact_time_entry = ttk.Entry(impact_controls, textvariable=self.impact_time_var, width=16)
        self.impact_time_entry.pack(side=tk.LEFT, padx=(0, 8))
        self.impact_time_entry.bind("<Return>", lambda _event: self.plot_variable_impact())
        self.impact_time_entry.bind("<FocusOut>", lambda _event: self.plot_variable_impact())

        ttk.Label(impact_controls, text="Valores:").pack(side=tk.LEFT, padx=(6, 6))
        self.impact_values_var = StringVar()
        self.impact_values_entry = ttk.Entry(impact_controls, textvariable=self.impact_values_var, width=16)
        self.impact_values_entry.pack(side=tk.LEFT, padx=(0, 8))
        self.impact_values_entry.bind("<Return>", lambda _event: self.plot_variable_impact())
        self.impact_values_entry.bind("<FocusOut>", lambda _event: self.plot_variable_impact())

        ttk.Label(impact_controls, text="Otras vars:").pack(side=tk.LEFT, padx=(6, 6))
        self.impact_baseline_var = StringVar()
        self.impact_baseline_entry = ttk.Entry(impact_controls, textvariable=self.impact_baseline_var, width=28)
        self.impact_baseline_entry.pack(side=tk.LEFT, padx=(0, 8))
        self.impact_baseline_entry.bind("<Return>", lambda _event: self.plot_variable_impact())
        self.impact_baseline_entry.bind("<FocusOut>", lambda _event: self.plot_variable_impact())

        self.impact_show_ci_var = BooleanVar(value=True)
        ttk.Checkbutton(
            impact_controls,
            text="Mostrar IC",
            variable=self.impact_show_ci_var,
            command=self.plot_variable_impact,
        ).pack(side=tk.LEFT, padx=(4, 0))

        self.impact_all_times_var = BooleanVar(value=False)
        ttk.Checkbutton(
            impact_controls,
            text="Todos los tiempos (degradado)",
            variable=self.impact_all_times_var,
            command=self.plot_variable_impact,
        ).pack(side=tk.LEFT, padx=(8, 0))

        ttk.Button(impact_controls, text="Actualizar", command=self.plot_variable_impact).pack(side=tk.LEFT, padx=(4, 0))
        ttk.Label(
            self.impact_tab,
            text="Tiempo(s): 12,24,36 | Valores: 10,20,30 | Otras vars: Variable:valor | 'Todos los tiempos' usa el grid completo con degradado de color",
            foreground="#666666",
        ).pack(anchor="w", padx=12, pady=(2, 4))

        self.impact_fig = plt.figure(figsize=(6, 4))
        self.impact_canvas = FigureCanvasTkAgg(self.impact_fig, master=self.impact_tab)
        self.impact_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self._register_editable_plot_canvas(self.impact_canvas, "impact")

        self.calibration_fig = plt.figure(figsize=(6, 4))
        self.calibration_canvas = FigureCanvasTkAgg(self.calibration_fig, master=self.calibration_tab)
        self.calibration_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self._register_editable_plot_canvas(self.calibration_canvas, "calibration")

        self.brier_fig = plt.figure(figsize=(6, 4))
        self.brier_canvas = FigureCanvasTkAgg(self.brier_fig, master=self.brier_tab)
        self.brier_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self._register_editable_plot_canvas(self.brier_canvas, "brier")

        # ── Tree tab controls ──
        tree_controls = ttk.Frame(self.tree_tab)
        tree_controls.pack(fill=tk.X, padx=8, pady=(8, 0))
        ttk.Label(tree_controls, text="Árbol #:").pack(side=tk.LEFT, padx=(0, 4))
        self.tree_index_var = IntVar(value=0)
        self.tree_index_spin = ttk.Spinbox(
            tree_controls, textvariable=self.tree_index_var,
            from_=0, to=0, width=6, command=self.plot_single_tree)
        self.tree_index_spin.pack(side=tk.LEFT, padx=(0, 8))
        self.tree_index_spin.bind("<Return>", lambda _e: self.plot_single_tree())

        ttk.Label(tree_controls, text="Prof. máx:").pack(side=tk.LEFT, padx=(6, 4))
        self.tree_max_depth_var = IntVar(value=3)
        tree_depth_spin = ttk.Spinbox(
            tree_controls, textvariable=self.tree_max_depth_var,
            from_=1, to=10, width=4, command=self.plot_single_tree)
        tree_depth_spin.pack(side=tk.LEFT, padx=(0, 8))
        tree_depth_spin.bind("<Return>", lambda _e: self.plot_single_tree())

        self.tree_show_impurity_var = BooleanVar(value=False)
        ttk.Checkbutton(
            tree_controls, text="Mostrar impureza",
            variable=self.tree_show_impurity_var,
            command=self.plot_single_tree,
        ).pack(side=tk.LEFT, padx=(4, 0))

        ttk.Button(tree_controls, text="Actualizar", command=self.plot_single_tree).pack(side=tk.LEFT, padx=(8, 0))

        ttk.Label(
            self.tree_tab,
            text="Muestra un árbol individual del bosque con nodos, reglas de split, hojas y nº de casos.",
            foreground="#666666",
        ).pack(anchor="w", padx=12, pady=(2, 4))

        self.tree_fig = plt.figure(figsize=(12, 6))
        
        self.tree_scroll_container = ttk.Frame(self.tree_tab)
        self.tree_scroll_container.pack(fill=tk.BOTH, expand=True)
        
        self.tree_scroll_canvas = tk.Canvas(self.tree_scroll_container, bg="white")
        tree_v_scroll = ttk.Scrollbar(self.tree_scroll_container, orient=tk.VERTICAL, command=self.tree_scroll_canvas.yview)
        tree_h_scroll = ttk.Scrollbar(self.tree_scroll_container, orient=tk.HORIZONTAL, command=self.tree_scroll_canvas.xview)
        self.tree_scroll_canvas.configure(yscrollcommand=tree_v_scroll.set, xscrollcommand=tree_h_scroll.set)
        
        tree_v_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        tree_h_scroll.pack(side=tk.BOTTOM, fill=tk.X)
        self.tree_scroll_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.tree_canvas = FigureCanvasTkAgg(self.tree_fig, master=self.tree_scroll_canvas)
        self.tree_canvas_widget = self.tree_canvas.get_tk_widget()
        self.tree_scroll_window = self.tree_scroll_canvas.create_window((0, 0), window=self.tree_canvas_widget, anchor="nw")
        
        def _on_tree_fig_resize(*args):
            w, h = self.tree_canvas.get_width_height()
            self.tree_canvas_widget.config(width=w, height=h)
            self.tree_scroll_canvas.configure(scrollregion=(0, 0, w, h))
            self.tree_scroll_canvas.itemconfig(self.tree_scroll_window, width=w, height=h)
            
        self.tree_canvas.callbacks.connect('resize_event', _on_tree_fig_resize)
        self._update_tree_scroll = _on_tree_fig_resize
        
        self._register_editable_plot_canvas(self.tree_canvas, "tree")

        # ── Prof. Mínima tab ──────────────────────────────────────────
        mindepth_controls = ttk.Frame(self.mindepth_tab)
        mindepth_controls.pack(fill=tk.X, padx=8, pady=(8, 0))
        self.mindepth_ci_var = StringVar(value="95")
        ttk.Label(mindepth_controls, text="Nivel de Confianza (%):").pack(side=tk.LEFT, padx=(0, 4))
        mindepth_ci_combo = ttk.Combobox(
            mindepth_controls, textvariable=self.mindepth_ci_var,
            values=["90", "95", "99"], state="readonly", width=5,
        )
        mindepth_ci_combo.pack(side=tk.LEFT, padx=(0, 8))
        mindepth_ci_combo.bind("<<ComboboxSelected>>", lambda _e: self.plot_minimal_depth())
        ttk.Button(mindepth_controls, text="Recalcular", command=self.plot_minimal_depth).pack(side=tk.LEFT, padx=(4, 0))
        self.mindepth_fig = plt.figure(figsize=(6, 4))
        self.mindepth_canvas = FigureCanvasTkAgg(self.mindepth_fig, master=self.mindepth_tab)
        self.mindepth_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        ttk.Label(
            self.mindepth_tab,
            text="Profundidad Mínima Media: cuánto más pequeño, más predictiva es la variable.",
            foreground="#666666",
        ).pack(anchor="w", padx=12, pady=(0, 4))

        # ── SHAP tab ──────────────────────────────────────────────────
        shap_controls = ttk.Frame(self.shap_tab)
        shap_controls.pack(fill=tk.X, padx=8, pady=(8, 0))
        ttk.Label(shap_controls, text="Muestra máx:").pack(side=tk.LEFT, padx=(0, 4))
        self.shap_n_var = IntVar(value=200)
        shap_spin = ttk.Spinbox(shap_controls, textvariable=self.shap_n_var, from_=20, to=2000, width=6,
                                command=self.plot_shap_summary)
        shap_spin.pack(side=tk.LEFT, padx=(0, 8))
        shap_spin.bind("<Return>", lambda _e: self.plot_shap_summary())
        ttk.Button(shap_controls, text="Calcular SHAP", command=self.plot_shap_summary).pack(side=tk.LEFT, padx=(4, 0))
        # ── SHAP Dependence Plot controls ──
        shap_dep_controls = ttk.Frame(self.shap_tab)
        shap_dep_controls.pack(fill=tk.X, padx=8, pady=(4, 0))
        ttk.Label(shap_dep_controls, text="Dependence Plot — Variable X:").pack(side=tk.LEFT, padx=(0, 4))
        self.shap_dep_x_var = StringVar()
        self.shap_dep_x_combo = ttk.Combobox(shap_dep_controls, textvariable=self.shap_dep_x_var,
                                              state="readonly", width=18)
        self.shap_dep_x_combo.pack(side=tk.LEFT, padx=(0, 8))
        ttk.Label(shap_dep_controls, text="Color/Interacción:").pack(side=tk.LEFT, padx=(0, 4))
        self.shap_dep_color_var = StringVar()
        self.shap_dep_color_combo = ttk.Combobox(shap_dep_controls, textvariable=self.shap_dep_color_var,
                                                  state="readonly", width=18)
        self.shap_dep_color_combo.pack(side=tk.LEFT, padx=(0, 8))
        # ── Representation options for the interaction variable ──
        sep = ttk.Separator(shap_dep_controls, orient=tk.VERTICAL)
        sep.pack(side=tk.LEFT, fill=tk.Y, padx=(4, 6), pady=2)
        ttk.Label(shap_dep_controls, text="Mostrar como:").pack(side=tk.LEFT, padx=(0, 3))
        self.shap_dep_show_color_var = BooleanVar(value=True)
        ttk.Checkbutton(shap_dep_controls, text="Color",
                         variable=self.shap_dep_show_color_var).pack(side=tk.LEFT, padx=(0, 2))
        self.shap_dep_show_size_var = BooleanVar(value=False)
        ttk.Checkbutton(shap_dep_controls, text="Tamaño",
                         variable=self.shap_dep_show_size_var).pack(side=tk.LEFT, padx=(0, 2))
        self.shap_dep_show_value_var = BooleanVar(value=False)
        ttk.Checkbutton(shap_dep_controls, text="Valor",
                         variable=self.shap_dep_show_value_var).pack(side=tk.LEFT, padx=(0, 4))
        ttk.Button(shap_dep_controls, text="Generar Dependence Plot",
                   command=self.plot_shap_dependence).pack(side=tk.LEFT, padx=(4, 0))
        ttk.Label(
            self.shap_tab,
            text="Requiere: pip install shap  |  Puede tardar unos segundos.",
            foreground="#666666",
        ).pack(anchor="w", padx=12, pady=(0, 4))
        self.shap_fig = plt.figure(figsize=(7, 5))
        self.shap_canvas = FigureCanvasTkAgg(self.shap_fig, master=self.shap_tab)
        self.shap_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # ── PDP tab ───────────────────────────────────────────────────
        pdp_controls = ttk.Frame(self.pdp_tab)
        pdp_controls.pack(fill=tk.X, padx=8, pady=(8, 0))

        pdp_row1 = ttk.Frame(pdp_controls)
        pdp_row1.pack(fill=tk.X, pady=2)
        pdp_row2 = ttk.Frame(pdp_controls)
        pdp_row2.pack(fill=tk.X, pady=2)

        ttk.Label(pdp_row1, text="Variable:").pack(side=tk.LEFT, padx=(0, 4))
        self.pdp_covariate_var = StringVar()
        self.pdp_covariate_combo = ttk.Combobox(pdp_row1, textvariable=self.pdp_covariate_var,
                                                 state="readonly", width=22)
        self.pdp_covariate_combo.pack(side=tk.LEFT, padx=(0, 8))
        self.pdp_covariate_combo.bind("<<ComboboxSelected>>", lambda _e: self.plot_pdp())
        ttk.Label(pdp_row1, text="Puntos:").pack(side=tk.LEFT, padx=(6, 4))
        self.pdp_grid_var = IntVar(value=40)
        pdp_grid_spin = ttk.Spinbox(pdp_row1, textvariable=self.pdp_grid_var, from_=10, to=200,
                                    width=5, command=self.plot_pdp)
        pdp_grid_spin.pack(side=tk.LEFT, padx=(0, 8))
        pdp_grid_spin.bind("<Return>", lambda _e: self.plot_pdp())
        self.pdp_ice_var = BooleanVar(value=False)
        ttk.Checkbutton(pdp_row1, text="Mostrar ICE", variable=self.pdp_ice_var,
                         command=self.plot_pdp).pack(side=tk.LEFT, padx=(4, 0))
        ttk.Button(pdp_row1, text="Actualizar", command=self.plot_pdp).pack(side=tk.LEFT, padx=(8, 0))

        ttk.Label(pdp_row2, text="Colorear ICE por:").pack(side=tk.LEFT, padx=(0, 4))
        self.pdp_ice_color_var = StringVar(value="(Ninguna)")
        self.pdp_ice_color_combo = ttk.Combobox(pdp_row2, textvariable=self.pdp_ice_color_var,
                                                 state="readonly", width=18)
        self.pdp_ice_color_combo.pack(side=tk.LEFT, padx=(0, 4))
        self.pdp_ice_color_combo.bind("<<ComboboxSelected>>", lambda _e: self.plot_pdp())
        ttk.Label(
            self.pdp_tab,
            text="PDP: efecto marginal de la variable sobre el riesgo predicho (ICE = líneas individuales).",
            foreground="#666666",
        ).pack(anchor="w", padx=12, pady=(2, 4))
        self.pdp_fig = plt.figure(figsize=(6, 4))
        self.pdp_canvas = FigureCanvasTkAgg(self.pdp_fig, master=self.pdp_tab)
        self.pdp_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # ── Proximidad tab ────────────────────────────────────────────
        prox_controls = ttk.Frame(self.proximity_tab)
        prox_controls.pack(fill=tk.X, padx=8, pady=(8, 0))
        ttk.Label(prox_controls, text="Muestra máx:").pack(side=tk.LEFT, padx=(0, 4))
        self.proximity_n_var = IntVar(value=150)
        prox_spin = ttk.Spinbox(prox_controls, textvariable=self.proximity_n_var, from_=20, to=500,
                                width=6, command=self.plot_proximity_matrix)
        prox_spin.pack(side=tk.LEFT, padx=(0, 8))
        prox_spin.bind("<Return>", lambda _e: self.plot_proximity_matrix())
        self.proximity_cluster_var = BooleanVar(value=True)
        ttk.Checkbutton(prox_controls, text="Reordenar por clustering", variable=self.proximity_cluster_var,
                         command=self.plot_proximity_matrix).pack(side=tk.LEFT, padx=(4, 0))
        ttk.Button(prox_controls, text="Calcular", command=self.plot_proximity_matrix).pack(side=tk.LEFT, padx=(8, 0))
        # ── Clustering y Cruce Clínico ─────────────────────────────
        cluster_frame = ttk.LabelFrame(self.proximity_tab, text="Clustering Jerárquico y Cruce Clínico")
        cluster_frame.pack(fill=tk.X, padx=8, pady=(6, 0))
        cluster_row1 = ttk.Frame(cluster_frame)
        cluster_row1.pack(fill=tk.X, padx=6, pady=(4, 2))
        ttk.Label(cluster_row1, text="Nº Clústeres:").pack(side=tk.LEFT, padx=(0, 4))
        self.proximity_n_clusters_var = IntVar(value=4)
        ttk.Spinbox(cluster_row1, textvariable=self.proximity_n_clusters_var,
                     from_=2, to=10, width=4).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Label(cluster_row1, text="Método:").pack(side=tk.LEFT, padx=(0, 4))
        self.proximity_linkage_var = StringVar(value="ward")
        ttk.Combobox(cluster_row1, textvariable=self.proximity_linkage_var,
                     values=["ward", "average", "complete", "single"],
                     state="readonly", width=10).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(cluster_row1, text="Generar Grupo_RSF",
                   command=self._generate_proximity_clusters).pack(side=tk.LEFT, padx=(8, 0))
        self.proximity_cluster_status_var = StringVar(value="Sin grupos asignados")
        ttk.Label(cluster_row1, textvariable=self.proximity_cluster_status_var,
                  foreground="navy").pack(side=tk.LEFT, padx=(12, 0))
        cluster_row2 = ttk.Frame(cluster_frame)
        cluster_row2.pack(fill=tk.X, padx=6, pady=(2, 6))
        ttk.Label(cluster_row2, text="Variable a cruzar:").pack(side=tk.LEFT, padx=(0, 4))
        self.proximity_cross_var = StringVar()
        self.proximity_cross_combo = ttk.Combobox(cluster_row2, textvariable=self.proximity_cross_var,
                                                   state="readonly", width=24)
        self.proximity_cross_combo.pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(cluster_row2, text="Análisis Bivariado (Cruce Clínico)",
                   command=self._run_cluster_bivariate_analysis).pack(side=tk.LEFT, padx=(4, 0))
        ttk.Button(cluster_row2, text="KM por Grupo_RSF",
                   command=self._run_cluster_km_analysis).pack(side=tk.LEFT, padx=(4, 0))
        ttk.Label(
            self.proximity_tab,
            text="Fracción de árboles en que dos pacientes caen en el mismo nodo hoja. Alto = comportamiento similar.",
            foreground="#666666",
        ).pack(anchor="w", padx=12, pady=(2, 4))
        self.proximity_fig = plt.figure(figsize=(6, 5))
        self.proximity_canvas = FigureCanvasTkAgg(self.proximity_fig, master=self.proximity_tab)
        self.proximity_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------
    def _apply_preset(self):
        preset = self.preset_var.get()
        if preset == "Rápido":
            self.n_estimators_var.set(100)
            self.max_features_var.set("sqrt")
            self.min_samples_leaf_var.set(3)
            self.min_samples_split_var.set(6)
        elif preset == "Robusto":
            self.n_estimators_var.set(500)
            self.max_features_var.set("log2")
            self.min_samples_leaf_var.set(10)
            self.min_samples_split_var.set(20)
        else:
            self.n_estimators_var.set(300)
            self.max_features_var.set("sqrt")
            self.min_samples_leaf_var.set(5)
            self.min_samples_split_var.set(10)

    def _coerce_int(self, raw_value, default, minimum=None):
        try:
            value = int(raw_value)
        except Exception:
            value = default
        if minimum is not None and value < minimum:
            value = minimum
        return value

    def _coerce_float(self, raw_value, default, minimum=None, maximum=None):
        try:
            value = float(raw_value)
        except Exception:
            value = default
        if minimum is not None and value < minimum:
            value = minimum
        if maximum is not None and value > maximum:
            value = maximum
        return value

    # -- Identity-based snapshot lookup (avoids DataFrame == comparison) --
    def _find_snapshot_identity(self, target, snapshot_list):
        """Return the index of *target* in *snapshot_list* using ``is``.

        Using ``in`` or ``list.index()`` with dicts that contain DataFrames
        triggers ``ValueError: The truth value of a DataFrame is ambiguous``
        because Python falls back to ``==`` element-wise comparison.
        """
        if target is None:
            return None
        for idx, item in enumerate(snapshot_list):
            if item is target:
                return idx
        return None

    def _resolve_holdout_split_settings(self, data, event_col, requested_test_size, min_train_rows=5, min_test_rows=2, prefer_stratify=True, context_label="RSF"):
        if data is None or len(data) == 0:
            raise ValueError("No hay datos suficientes para crear la partición train/test.")

        total_rows = int(len(data))
        requested = self._coerce_float(requested_test_size, 0.25, minimum=0.0, maximum=0.95)
        if requested <= 0:
            return 0.0, None, []

        min_train_rows = max(int(min_train_rows or 0), 2)
        min_test_rows = max(int(min_test_rows or 0), 1)
        warnings = []
        stratify_values = None

        if prefer_stratify and event_col in data.columns:
            event_series = (pd.to_numeric(data[event_col], errors='coerce').fillna(0) > 0).astype(int)
            n_classes = int(event_series.nunique())
            if n_classes > 1 and event_series.value_counts().min() >= 2:
                stratify_values = event_series
                min_train_rows = max(min_train_rows, n_classes)
                min_test_rows = max(min_test_rows, n_classes)
            elif n_classes > 1:
                warnings.append(
                    "No se pudo estratificar por evento porque alguna clase tiene muy pocos casos; se usó partición aleatoria simple."
                )

        if total_rows <= (min_train_rows + min_test_rows):
            fallback_train = max(2, min(total_rows - min_test_rows, min_train_rows))
            if total_rows <= fallback_train:
                raise ValueError(
                    f"No hay suficientes observaciones ({total_rows}) para separar entrenamiento/prueba con al menos {min_train_rows} filas de entrenamiento."
                )
            min_train_rows = fallback_train

        max_test_size = (total_rows - min_train_rows) / float(total_rows)
        min_test_size = min_test_rows / float(total_rows)
        adjusted = requested

        if adjusted > max_test_size:
            adjusted = max_test_size
            warnings.append(
                f"Proporción test ajustada de {requested:.2f} a {adjusted:.2f} para dejar al menos {min_train_rows} casos en entrenamiento."
            )
        if adjusted < min_test_size:
            adjusted = min_test_size
            if adjusted > requested + 1e-12:
                warnings.append(
                    f"Proporción test ajustada de {requested:.2f} a {adjusted:.2f} para dejar al menos {min_test_rows} casos en prueba."
                )

        adjusted = min(max(adjusted, 0.05), 0.95)
        test_count = int(np.ceil(adjusted * total_rows))
        train_count = total_rows - test_count
        if train_count < min_train_rows or test_count < min_test_rows:
            raise ValueError(
                f"No se pudo crear una partición válida con {total_rows} filas (train={train_count}, test={test_count})."
            )

        return float(adjusted), stratify_values, warnings

    def _parse_optional_int(self, raw_value):
        text = str(raw_value).strip()
        if not text:
            return None
        try:
            return int(float(text))
        except Exception:
            return None

    def _parse_optional_float(self, raw_value):
        text = str(raw_value).strip()
        if not text:
            return None
        try:
            return float(text)
        except Exception:
            return None

    def _required_min_samples_split(self, min_samples_leaf):
        min_leaf = self._coerce_int(min_samples_leaf, 1, minimum=1)
        return max(2, 2 * min_leaf)

    def _align_split_leaf_params(self, min_samples_leaf, min_samples_split):
        min_leaf = self._coerce_int(min_samples_leaf, 10, minimum=1)
        min_split = self._coerce_int(min_samples_split, 10, minimum=2)
        required_split = self._required_min_samples_split(min_leaf)
        if min_split < required_split:
            raise ValueError(
                "Configuración RSF incoherente: "
                f"min_samples_split={min_split} y min_samples_leaf={min_leaf}. "
                "Debe cumplirse min_samples_split >= 2 * min_samples_leaf "
                f"(mínimo requerido: {required_split})."
            )
        return min_leaf, min_split

    def _resolve_max_features(self, max_features_value, n_features):
        raw_value = str(max_features_value).strip().lower()
        if raw_value == "all":
            return None
        if raw_value in {"sqrt", "log2"}:
            return raw_value
        if raw_value == "manual":
            manual_value = str(self.max_features_manual_var.get()).strip()
            if not manual_value:
                return "sqrt"
            raw_value = manual_value

        try:
            if "." in raw_value:
                value = float(raw_value)
                if 0 < value <= 1:
                    return value
            value = int(float(raw_value))
            return max(1, min(value, max(1, n_features)))
        except Exception:
            return "sqrt"

    def _resolve_max_features_effective_count(self, resolved_max_features, n_features):
        """Return the effective number of variables considered per split.

        This mirrors sklearn semantics and is used to avoid tuning duplicate
        candidates that are textually different but statistically equivalent
        (e.g. sqrt/log2/0.5 when n_features=2 all resolve to 1).
        """
        try:
            n_total = max(1, int(n_features))
        except Exception:
            n_total = 1

        if resolved_max_features is None:
            return n_total

        if isinstance(resolved_max_features, (int, np.integer)):
            return max(1, min(int(resolved_max_features), n_total))

        if isinstance(resolved_max_features, (float, np.floating)):
            value = float(resolved_max_features)
            if not np.isfinite(value):
                return n_total
            if 0.0 < value <= 1.0:
                return max(1, min(int(value * n_total), n_total))
            return max(1, min(int(value), n_total))

        token = str(resolved_max_features).strip().lower()
        if token == "sqrt":
            return max(1, min(int(np.sqrt(n_total)), n_total))
        if token == "log2":
            return max(1, min(int(np.log2(n_total)), n_total))
        if token in {"all", "none", ""}:
            return n_total

        try:
            value = float(token)
            if 0.0 < value <= 1.0:
                return max(1, min(int(value * n_total), n_total))
            return max(1, min(int(value), n_total))
        except Exception:
            return n_total

    def _format_metric(self, value, decimals=4):
        if value is None or (isinstance(value, float) and not np.isfinite(value)):
            return "N/D"
        try:
            return f"{float(value):.{decimals}f}"
        except Exception:
            return str(value)

    def _format_c_index_display(self, value, ci=None, decimals=3):
        base_text = self._format_metric(value, decimals=decimals)
        if base_text == "N/D":
            return "N/D"
        if isinstance(ci, (list, tuple)) and len(ci) == 2:
            try:
                lower = float(ci[0])
                upper = float(ci[1])
            except (TypeError, ValueError):
                return base_text
            if np.isfinite(lower) and np.isfinite(upper):
                if lower > upper:
                    lower, upper = upper, lower
                return f"{base_text} ({lower:.{decimals}f}, {upper:.{decimals}f})"
        return base_text

    def _format_max_features_display(self, params, snapshot=None):
        """Format max_features showing the resolved count, e.g. 'sqrt (=4)'."""
        raw = params.get("max_features", "-")
        if raw is None or str(raw).strip() in ("", "-"):
            return "-"
        raw_str = str(raw).strip()

        # Determine the number of features from snapshot or current state
        n_features = None
        if snapshot is not None:
            enc_cols = snapshot.get("latest_encoded_columns", [])
            if enc_cols:
                n_features = len(enc_cols)
            else:
                covs = snapshot.get("latest_covariates", [])
                if covs:
                    n_features = len(covs)
        if n_features is None:
            if self.latest_encoded_columns:
                n_features = len(self.latest_encoded_columns)
            elif self.latest_covariates:
                n_features = len(self.latest_covariates)

        if n_features is not None and n_features > 0:
            try:
                resolved = self._resolve_max_features(raw_str, n_features)
                effective = self._resolve_max_features_effective_count(resolved, n_features)
                raw_lower = raw_str.lower()
                if raw_lower in ("sqrt", "log2") or (
                    "." in raw_str and 0 < float(raw_str) <= 1
                ):
                    return f"{raw_str} (={effective})"
                elif raw_lower == "all":
                    return f"all (={n_features})"
                else:
                    return str(raw)
            except Exception:
                return str(raw)
        return str(raw)

    def _annotate_max_features_text(self, max_features_value):
        """Return a text annotation like 'sqrt (=4 de 10)' for reports."""
        if max_features_value is None:
            return "None"
        raw_str = str(max_features_value).strip()
        if raw_str in ("", "-"):
            return raw_str

        # Get the total number of encoded features
        n_features = None
        if hasattr(self, "latest_encoded_columns") and self.latest_encoded_columns:
            n_features = len(self.latest_encoded_columns)
        elif hasattr(self, "latest_covariates") and self.latest_covariates:
            n_features = len(self.latest_covariates)

        if n_features is not None and n_features > 0:
            try:
                resolved = self._resolve_max_features(raw_str, n_features)
                effective = self._resolve_max_features_effective_count(resolved, n_features)
                raw_lower = raw_str.lower()
                if raw_lower in ("sqrt", "log2"):
                    return f"{raw_str} (={effective} de {n_features})"
                elif "." in raw_str:
                    try:
                        fval = float(raw_str)
                        if 0 < fval <= 1:
                            return f"{raw_str} (={effective} de {n_features})"
                    except ValueError:
                        pass
                elif raw_lower in ("all", "none"):
                    return f"all (={n_features})"
            except Exception:
                pass
        return raw_str

    def _describe_shared_source(self, metadata):
        if not isinstance(metadata, dict):
            return "Archivo de Trabajo"
        for key in ("source_name", "file_name", "tab_name", "module_name", "display_name"):
            value = metadata.get(key)
            if value:
                return str(value)
        return "Archivo de Trabajo"

    def _quote_report_value(self, value):
        if isinstance(value, float):
            return f"{value:.4g}"
        return str(value)

    def _get_categorical_compare_display_value(self, internal_mode):
        return self.categorical_compare_display_map.get(internal_mode, self.categorical_compare_display_map["all"])

    def _get_default_categorical_compare_display(self):
        return self._get_categorical_compare_display_value("all")

    def _get_categorical_compare_internal_mode(self, display_value):
        return self.categorical_compare_reverse_map.get(display_value, "all")

    def open_categorical_config_dialog(self):
        if self.data is None or self.data.empty:
            messagebox.showwarning("Sin datos", "Carga datos y selecciona covariables primero.")
            return

        selected_indices = self.covariates_listbox.curselection() if hasattr(self, 'covariates_listbox') else ()
        if not selected_indices:
            messagebox.showwarning("Sin selección", "Selecciona una o más covariables para configurar sus categorías.")
            return

        selected_vars = [self.covariates_listbox.get(i) for i in selected_indices]
        categorical_vars = []
        for var_name in selected_vars:
            if var_name not in self.data.columns:
                continue
            series = self.data[var_name]
            unique_count = int(series.nunique(dropna=True)) if hasattr(series, 'nunique') else 0
            if (not pd.api.types.is_numeric_dtype(series)) or unique_count <= 10:
                categorical_vars.append(var_name)

        if not categorical_vars:
            messagebox.showwarning(
                "Sin variables configurables",
                "Las covariables seleccionadas parecen numéricas continuas. Selecciona variables categóricas o numéricas con pocos niveles.",
            )
            return

        RSFCategoricalConfigDialog(self, self, categorical_vars)

    def _is_covariate_categorical(self, covariate):
        """Central authority: should this covariate be treated as categorical?

        Priority order:
        1. User explicitly configured via 'Configurar categorías' dialog
           → obey variable_configs['treat_as'] / ['compare_mode'] 100%
        2. Not configured → fall back to dtype of the raw data:
           - Non-numeric dtype (object/string/category) → True (categorical)
           - Numeric dtype → False (continuous)

        This function is the SINGLE source of truth used by all plots,
        reports and calculations. Never use pd.api.types.is_numeric_dtype
        on user covariates directly — call this instead.
        """
        # --- 1. Check explicit user config ---
        cfg = {}
        if isinstance(getattr(self, 'variable_configs', {}), dict):
            cfg = self.variable_configs.get(covariate, {})

        treat_as = str(cfg.get('treat_as', cfg.get('compare_mode', ''))).strip().lower()

        if treat_as in {'categorical', 'all', 'one_vs_rest'}:
            return True
        if treat_as in {'quantitative', 'numeric', 'continuous', 'continuo', 'cuantitativa'}:
            return False

        # --- 2. Not configured: use dtype as fallback ---
        data = (self.latest_fit_dataframe
                if isinstance(self.latest_fit_dataframe, pd.DataFrame)
                else getattr(self, 'data', None))
        if data is not None and covariate in data.columns:
            return not pd.api.types.is_numeric_dtype(data[covariate])
        return False

    def _apply_categorical_configurations(self, df, covariates, warnings_list=None):
        if df is None or df.empty:
            return df

        transformed_df = df.copy()
        for cov in covariates:
            if cov not in transformed_df.columns:
                continue

            config = self.variable_configs.get(cov, {}) if isinstance(getattr(self, 'variable_configs', {}), dict) else {}
            series = transformed_df[cov]
            is_numeric_series = pd.api.types.is_numeric_dtype(series)
            compare_mode = str(config.get('compare_mode', 'all')).strip().lower()
            treatment_mode = str(config.get('treat_as', compare_mode)).strip().lower()
            treat_as_categorical = (not is_numeric_series)
            if treatment_mode in {'quantitative', 'numeric', 'continuous', 'continuo', 'cuantitativa'}:
                treat_as_categorical = False
            elif compare_mode in {'all', 'one_vs_rest'} and bool(config):
                treat_as_categorical = True
            if not treat_as_categorical:
                continue

            non_missing = series.dropna()
            if non_missing.empty:
                continue

            normalized_series = series.apply(lambda value: np.nan if pd.isna(value) else str(value))
            unique_values = [str(val) for val in pd.Series(normalized_series.dropna()).unique().tolist()]
            if not unique_values:
                continue

            compare_mode = str(config.get('compare_mode', 'all')).strip().lower()
            chosen_category = config.get('selected_cat', config.get('ref_cat', unique_values[0]))
            chosen_category = str(chosen_category) if chosen_category not in (None, '') else unique_values[0]

            if compare_mode == 'one_vs_rest':
                rest_label = f"Resto ({chosen_category})"
                transformed_series = normalized_series.apply(
                    lambda value: np.nan if pd.isna(value) else (chosen_category if str(value) == chosen_category else rest_label)
                )
                transformed_df[cov] = pd.Categorical(transformed_series, categories=[rest_label, chosen_category])
                if warnings_list is not None:
                    note = f"'{cov}' se codificó como dicotómica: {chosen_category} vs resto."
                    if note not in warnings_list:
                        warnings_list.append(note)
            else:
                if chosen_category in unique_values:
                    ordered_categories = [chosen_category] + [value for value in unique_values if value != chosen_category]
                    transformed_df[cov] = pd.Categorical(normalized_series, categories=ordered_categories)
                    if warnings_list is not None:
                        note = f"'{cov}' usa '{chosen_category}' como categoría de referencia."
                        if note not in warnings_list:
                            warnings_list.append(note)
                else:
                    transformed_df[cov] = normalized_series

        return transformed_df

    def _get_plot_covariate_candidates(self):
        if self.latest_fit_dataframe is None or not self.latest_covariates:
            return []
        return [cov for cov in self.latest_covariates if cov in self.latest_fit_dataframe.columns]

    def _coerce_plot_value_for_column(self, column_name, raw_value):
        raw_text = "" if raw_value is None else str(raw_value).strip()
        if raw_text == "":
            return None

        data = self.latest_fit_dataframe
        if data is not None and column_name in data.columns and pd.api.types.is_numeric_dtype(data[column_name]):
            return float(raw_text)

        return raw_text

    def _parse_plot_values(self, column_name, raw_value):
        if raw_value is None:
            return []

        parts = [part.strip() for part in re.split(r'[,;]', str(raw_value)) if part.strip()]
        parsed_values = []
        for part in parts:
            try:
                coerced = self._coerce_plot_value_for_column(column_name, part)
            except (TypeError, ValueError):
                continue
            if coerced is not None:
                parsed_values.append(coerced)
        return parsed_values

    def _parse_time_points_input(self, raw_value):
        if raw_value is None:
            return []

        parts = [part.strip() for part in re.split(r'[,;]', str(raw_value)) if part.strip()]
        parsed_times = []
        seen_values = set()
        for part in parts:
            try:
                time_value = float(part)
            except (TypeError, ValueError):
                continue
            if not np.isfinite(time_value) or time_value <= 0:
                continue
            normalized = float(time_value)
            if normalized in seen_values:
                continue
            parsed_times.append(normalized)
            seen_values.add(normalized)
        return parsed_times

    def _get_fixed_time_palette(self, time_points):
        palette_colors = [
            "#1f77b4",  # azul
            "#d62728",  # rojo
            "#2ca02c",  # verde
            "#9467bd",  # morado
            "#ff7f0e",  # naranja
            "#17becf",  # cian
            "#8c564b",  # café
            "#e377c2",  # rosa
        ]
        line_styles = ['-', '--', '-.', ':']
        marker_styles = ['o', 's', '^', 'D', 'v', 'P', 'X', '*']
        hatch_styles = ['', '//', '\\\\', 'xx', '..', '++', '--', 'oo']

        normalized_times = []
        seen_values = set()
        for raw_value in time_points or []:
            try:
                normalized = float(raw_value)
            except (TypeError, ValueError):
                continue
            if not np.isfinite(normalized):
                continue
            if normalized in seen_values:
                continue
            normalized_times.append(normalized)
            seen_values.add(normalized)

        style_map = {}
        for idx, time_value in enumerate(sorted(normalized_times)):
            style_map[time_value] = {
                "color": palette_colors[idx % len(palette_colors)],
                "line_style": line_styles[idx % len(line_styles)],
                "marker": marker_styles[idx % len(marker_styles)],
                "hatch": hatch_styles[idx % len(hatch_styles)],
            }
        return style_map

    def _parse_plot_baseline_overrides(self, raw_value, exclude_covariate=None):
        overrides = {}
        if raw_value is None:
            return overrides

        raw_text = str(raw_value).strip()
        if not raw_text:
            return overrides

        normalized_chunks = [chunk.strip() for chunk in re.split(r'[;,\n]+', raw_text) if chunk.strip()]
        for piece in normalized_chunks:
            if '=' in piece:
                key, value = piece.split('=', 1)
            elif ':' in piece:
                key, value = piece.split(':', 1)
            else:
                continue
            key = key.strip()
            value = value.strip()
            if not key or key == exclude_covariate or key not in self.latest_covariates:
                continue
            try:
                coerced = self._coerce_plot_value_for_column(key, value)
            except (TypeError, ValueError):
                continue
            if coerced is not None:
                overrides[key] = coerced
        return overrides

    def _build_plot_reference_row(self, focal_covariate=None, overrides=None):
        overrides = overrides or {}
        base_row = {}
        data = self.latest_fit_dataframe if self.latest_fit_dataframe is not None else pd.DataFrame()

        for col in self.latest_covariates:
            if col == focal_covariate or col not in data.columns:
                continue
            if col in overrides:
                base_row[col] = overrides[col]
                continue

            col_series = data[col].dropna()
            if col_series.empty:
                base_row[col] = 0.0 if pd.api.types.is_numeric_dtype(data[col]) else ""
                continue

            if pd.api.types.is_numeric_dtype(col_series):
                mean_val = pd.to_numeric(col_series, errors='coerce').mean()
                if pd.isna(mean_val):
                    mean_val = pd.to_numeric(col_series, errors='coerce').median()
                if pd.isna(mean_val):
                    mean_val = pd.to_numeric(col_series, errors='coerce').dropna().iloc[0]
                base_row[col] = float(mean_val)
            else:
                modes = col_series.astype(str).mode()
                base_row[col] = modes.iloc[0] if not modes.empty else str(col_series.astype(str).iloc[0])

        return base_row

    def _format_plot_value_label(self, value):
        if isinstance(value, (int, float, np.integer, np.floating)) and np.isfinite(value):
            return f"{float(value):.3g}"
        return str(value)

    def _get_plot_categorical_values(self, covariate, cov_series):
        if cov_series is None:
            return []

        series_non_null = cov_series.dropna()
        if series_non_null.empty:
            return []

        try:
            dtype = self.latest_fit_dataframe[covariate].dtype if (
                isinstance(self.latest_fit_dataframe, pd.DataFrame)
                and covariate in self.latest_fit_dataframe.columns
            ) else None
        except Exception:
            dtype = None

        if isinstance(dtype, pd.CategoricalDtype):
            ordered_values = []
            observed = set(series_non_null.astype(str).tolist())
            for category_value in dtype.categories:
                category_label = str(category_value)
                if category_label in observed:
                    ordered_values.append(category_label)
            if ordered_values:
                return ordered_values

        return list(dict.fromkeys(series_non_null.astype(str).tolist()))

    def _encode_prediction_frame(self, predict_df):
        if predict_df is None or predict_df.empty:
            return pd.DataFrame(columns=self.latest_encoded_columns)

        prepared_predict_df = self._apply_categorical_configurations(
            predict_df.copy(),
            self.latest_covariates,
            warnings_list=None,
        )
        encoded_df = pd.get_dummies(
            prepared_predict_df,
            drop_first=bool(getattr(self, 'latest_drop_first', True)),
            dummy_na=False,
        )
        return encoded_df.reindex(columns=self.latest_encoded_columns, fill_value=0)

    def _build_loaded_snapshot_eval_payload(self, encoded_full):
        if not isinstance(self.latest_fit_dataframe, pd.DataFrame):
            return None, None
        if not self.latest_duration_col or not self.latest_event_col:
            return None, None
        if self.latest_duration_col not in self.latest_fit_dataframe.columns or self.latest_event_col not in self.latest_fit_dataframe.columns:
            return None, None

        try:
            duration_values = pd.to_numeric(self.latest_fit_dataframe[self.latest_duration_col], errors='coerce')
            event_values = pd.to_numeric(self.latest_fit_dataframe[self.latest_event_col], errors='coerce')
        except Exception:
            return None, None

        valid_mask = duration_values.notna() & (duration_values > 0) & event_values.notna()
        if int(valid_mask.sum()) == 0:
            return None, None

        try:
            encoded_eval = encoded_full.loc[valid_mask].copy()
        except Exception:
            try:
                valid_idx = np.flatnonzero(np.asarray(valid_mask, dtype=bool))
                encoded_eval = encoded_full.iloc[valid_idx].copy()
            except Exception:
                return None, None

        y_eval = np.empty(int(valid_mask.sum()), dtype=[("event", bool), ("time", float)])
        y_eval["event"] = event_values.loc[valid_mask].to_numpy(dtype=float) > 0
        y_eval["time"] = duration_values.loc[valid_mask].to_numpy(dtype=float)
        return encoded_eval, y_eval

    def _resolve_loaded_snapshot_eval_time(self, y_eval):
        candidate_values = [self.latest_eval_time]
        if isinstance(self.results, dict):
            candidate_values.append(self.results.get("brier_eval_time"))

        for raw_value in candidate_values:
            try:
                time_value = float(raw_value)
            except (TypeError, ValueError):
                continue
            if np.isfinite(time_value) and time_value > 0:
                return float(time_value)

        try:
            observed_times = np.asarray(y_eval["time"], dtype=float)
            observed_times = observed_times[np.isfinite(observed_times) & (observed_times > 0)]
        except Exception:
            observed_times = np.asarray([], dtype=float)

        if observed_times.size:
            return float(np.nanmedian(observed_times))
        return None

    def _build_brier_fallback_dataframe(self):
        if not isinstance(self.results, dict):
            return pd.DataFrame()

        raw_time = self.results.get("brier_eval_time")
        raw_score = self.results.get("brier_at_eval_time")
        try:
            time_value = float(raw_time)
            score_value = float(raw_score)
        except (TypeError, ValueError):
            return pd.DataFrame()

        if not np.isfinite(time_value) or time_value <= 0 or not np.isfinite(score_value):
            return pd.DataFrame()

        return pd.DataFrame({"time": [time_value], "brier_score": [score_value]})

    def _recalculate_brier_from_model(self):
        """Recalculate Brier curve using model + full data when holdout data is missing."""
        try:
            if self.model is None or not callable(brier_score):
                return pd.DataFrame()
            X_enc = self._get_X_encoded()
            if X_enc is None or X_enc.empty:
                return pd.DataFrame()
            if not isinstance(self.latest_fit_dataframe, pd.DataFrame):
                return pd.DataFrame()
            if not self.latest_duration_col or not self.latest_event_col:
                return pd.DataFrame()

            duration_col = self.latest_duration_col
            event_col = self.latest_event_col
            df = self.latest_fit_dataframe
            durations = pd.to_numeric(df[duration_col], errors="coerce")
            events = (pd.to_numeric(df[event_col], errors="coerce").fillna(0) > 0).astype(bool)
            valid = durations.notna() & np.isfinite(durations)
            if valid.sum() < 10:
                return pd.DataFrame()

            y_structured = Surv.from_arrays(events[valid].values, durations[valid].values) if Surv is not None else None
            if y_structured is None:
                return pd.DataFrame()

            X_valid = X_enc.loc[valid]
            times_arr = durations[valid].values
            t_min = float(np.percentile(times_arr[times_arr > 0], 5))
            t_max = float(np.percentile(times_arr, 95))
            if t_max <= t_min:
                return pd.DataFrame()
            eval_times = np.linspace(t_min, t_max, num=30)

            surv_fns = self.model.predict_survival_function(X_valid)
            surv_matrix = np.asarray([fn(eval_times) for fn in surv_fns], dtype=float)
            _, brier_values = brier_score(y_structured, y_structured, surv_matrix, eval_times)
            result_df = pd.DataFrame({"time": eval_times, "brier_score": brier_values})
            self.latest_brier_df = result_df
            if self.latest_eval_time is None:
                self.latest_eval_time = float(eval_times[len(eval_times) // 2])
            return result_df
        except Exception:
            return pd.DataFrame()

    def _recalculate_calibration_from_model(self):
        """Recalculate calibration using model + full data when holdout data is missing."""
        try:
            if self.model is None:
                return
            X_enc = self._get_X_encoded()
            if X_enc is None or X_enc.empty:
                return
            if not isinstance(self.latest_fit_dataframe, pd.DataFrame):
                return
            if not self.latest_duration_col or not self.latest_event_col:
                return

            duration_col = self.latest_duration_col
            event_col = self.latest_event_col
            df = self.latest_fit_dataframe
            durations = pd.to_numeric(df[duration_col], errors="coerce")
            events = (pd.to_numeric(df[event_col], errors="coerce").fillna(0) > 0).astype(bool)
            valid = durations.notna() & np.isfinite(durations)
            if valid.sum() < 10:
                return

            y_structured = Surv.from_arrays(events[valid].values, durations[valid].values) if Surv is not None else None
            if y_structured is None:
                return

            X_valid = X_enc.loc[valid]
            eval_time = self.latest_eval_time
            if eval_time is None or not np.isfinite(eval_time):
                eval_time = float(np.percentile(durations[valid].values, 50))
                self.latest_eval_time = eval_time

            surv_fns = self.model.predict_survival_function(X_valid)
            predicted_surv = np.asarray([fn(eval_time) for fn in surv_fns], dtype=float)
            self.latest_calibration_df = self._summarize_calibration(y_structured, predicted_surv, eval_time)
        except Exception:
            pass


    def _sync_plot_covariate_selectors(self):
        candidates = self._get_plot_covariate_candidates()

        default_time = np.nan
        if self.latest_fit_dataframe is not None and self.latest_duration_col in self.latest_fit_dataframe.columns:
            try:
                default_time = float(np.nanmedian(pd.to_numeric(self.latest_fit_dataframe[self.latest_duration_col], errors='coerce')))
            except Exception:
                default_time = np.nan

        if hasattr(self, 'impact_time_var') and (not self.impact_time_var.get().strip()) and np.isfinite(default_time) and default_time > 0:
            self.impact_time_var.set(f"{default_time:.2f}")

        if hasattr(self, 'profile_covariate_combo'):
            self.profile_covariate_combo['values'] = candidates
            current_profile = self.profile_covariate_var.get() if hasattr(self, 'profile_covariate_var') else ''
            if candidates:
                if current_profile not in candidates:
                    self.profile_covariate_var.set(candidates[0])
            elif hasattr(self, 'profile_covariate_var'):
                self.profile_covariate_var.set('')

        if hasattr(self, 'impact_covariate_combo'):
            self.impact_covariate_combo['values'] = candidates
            current_impact = self.impact_covariate_var.get() if hasattr(self, 'impact_covariate_var') else ''
            if candidates:
                if current_impact not in candidates:
                    self.impact_covariate_var.set(candidates[0])
            elif hasattr(self, 'impact_covariate_var'):
                self.impact_covariate_var.set('')

    def _sync_pdp_covariate_selector(self):
        """Populate the PDP variable selector with the current encoded columns."""
        if not hasattr(self, "pdp_covariate_combo"):
            return
        X_enc = self._get_X_encoded()
        candidates = list(X_enc.columns) if X_enc is not None else []
        self.pdp_covariate_combo["values"] = candidates
        current = self.pdp_covariate_var.get() if hasattr(self, "pdp_covariate_var") else ""
        if candidates and current not in candidates:
            self.pdp_covariate_var.set(candidates[0])
        # Sync ICE color-by selector
        if hasattr(self, "pdp_ice_color_combo"):
            color_candidates = ["(Ninguna)"] + candidates
            self.pdp_ice_color_combo["values"] = color_candidates
            cur_color = self.pdp_ice_color_var.get() if hasattr(self, "pdp_ice_color_var") else ""
            if cur_color not in color_candidates:
                self.pdp_ice_color_var.set("(Ninguna)")

    def _sync_shap_dep_selectors(self):
        """Populate the SHAP Dependence Plot variable selectors."""
        if not hasattr(self, "shap_dep_x_combo"):
            return
        X_enc = self._get_X_encoded()
        enc_cols = list(X_enc.columns) if X_enc is not None else []
        self.shap_dep_x_combo["values"] = enc_cols
        target_df = self.latest_fit_dataframe if isinstance(self.latest_fit_dataframe, pd.DataFrame) else self.data
        if target_df is not None:
            color_opts = [col for col in target_df.columns if col not in (self.latest_duration_col, self.latest_event_col)]
        else:
            color_opts = list(enc_cols)
        self.shap_dep_color_combo["values"] = color_opts
        # Also sync proximity cross-variable combo
        if hasattr(self, "proximity_cross_combo") and target_df is not None:
            candidates = []
            if self.latest_duration_col:
                candidates.append(f"[Supervivencia] {self.latest_duration_col}")
            for col in target_df.columns:
                if col not in ("Grupo_RSF",) and col != self.latest_duration_col and col != self.latest_event_col:
                    candidates.append(col)
            self.proximity_cross_combo["values"] = candidates
            if candidates and not self.proximity_cross_var.get():
                self.proximity_cross_var.set(candidates[0])

    # ------------------------------------------------------------------
    # Data I/O and sync
    # ------------------------------------------------------------------
    def load_data(self):
        filepath = filedialog.askopenfilename(filetypes=[("Excel files", "*.xlsx *.xls"), ("CSV files", "*.csv")])
        if not filepath:
            return

        try:
            if filepath.lower().endswith(".csv"):
                self.data = pd.read_csv(filepath)
            else:
                self.data = pd.read_excel(filepath)

            self.base_data = self.data.copy(deep=True)
            self.variable_configs = {}
            self.file_label.config(text=os.path.basename(filepath))
            self.filter_component.set_dataframe(self.data)
            self._update_variable_comboboxes()
            self._reset_results_view()
            self.using_shared_dataset = False
            self.shared_metadata = {}
            self.current_shared_filter_summary = []
            messagebox.showinfo("Éxito", "Datos cargados correctamente en RSF.")
        except Exception as exc:
            messagebox.showerror("Error", f"No se pudo cargar el archivo: {exc}")
            self.data = None
            self.base_data = None

    def _update_variable_comboboxes(self):
        columns = self.data.columns.tolist() if isinstance(self.data, pd.DataFrame) else []

        pending_duration = str(getattr(self, "_persistent_duration_pending", "") or "").strip()
        pending_event = str(getattr(self, "_persistent_event_pending", "") or "").strip()
        current_duration = pending_duration or (self.duration_var.get().strip() if hasattr(self, "duration_var") else "")
        current_event = pending_event or (self.event_var.get().strip() if hasattr(self, "event_var") else "")
        selected_covariates = []
        if hasattr(self, "covariates_listbox"):
            try:
                selected_covariates = [
                    self.covariates_listbox.get(i) for i in self.covariates_listbox.curselection()
                ]
            except Exception:
                selected_covariates = []
        pending_covariates = [
            str(c).strip() for c in list(getattr(self, "_persistent_covariates_pending", []) or [])
            if str(c).strip()
        ]
        if pending_covariates:
            selected_covariates = list(dict.fromkeys(pending_covariates))

        self.duration_combo["values"] = columns
        self.event_combo["values"] = [""] + columns

        if hasattr(self, "duration_var"):
            resolved_duration = current_duration if current_duration in columns else ""
            self.duration_var.set(resolved_duration)
            if pending_duration and resolved_duration == pending_duration:
                self._persistent_duration_pending = ""
        if hasattr(self, "event_var"):
            resolved_event = current_event if current_event in columns else ""
            self.event_var.set(resolved_event)
            if pending_event and resolved_event == pending_event:
                self._persistent_event_pending = ""

        self.covariates_listbox.delete(0, tk.END)
        applied_covariates = []
        for idx, col in enumerate(columns):
            self.covariates_listbox.insert(tk.END, col)
            if col in selected_covariates:
                try:
                    self.covariates_listbox.selection_set(idx)
                    applied_covariates.append(col)
                except Exception:
                    pass

        if pending_covariates:
            self._persistent_covariates_pending = [
                c for c in pending_covariates if c not in set(applied_covariates)
            ]

    def update_variable_selectors(self):
        self._update_variable_comboboxes()

    def receive_shared_dataset(self, *, dataset, filtered_dataset=None, filter_summary=None, metadata=None, source_widget=None):
        if not self.using_shared_dataset:
            return
        if source_widget is self:
            return

        self.shared_metadata = dict(metadata or {})
        self.current_shared_filter_summary = list(filter_summary or [])

        if dataset is None:
            self.base_data = None
            self.data = None
            self.variable_configs = {}
            self._reset_results_view()
            try:
                self.filter_component.set_dataframe(pd.DataFrame())
            except Exception:
                pass
            self._update_variable_comboboxes()
            self.file_label.config(text="Sin archivo compartido.")
            return

        try:
            base_df = dataset.copy(deep=True)
        except Exception:
            base_df = dataset

        if isinstance(filtered_dataset, pd.DataFrame):
            try:
                active_df = filtered_dataset.copy(deep=True)
            except Exception:
                active_df = filtered_dataset
        else:
            try:
                active_df = base_df.copy(deep=True)
            except Exception:
                active_df = base_df

        self.base_data = base_df
        self.data = active_df
        self.variable_configs = {}

        try:
            self.filter_component.set_dataframe(self.data)
        except Exception:
            pass

        self._update_variable_comboboxes()
        self._reset_results_view()

        try:
            rows, cols = self.data.shape
            filter_suffix = f" | Filtros: {len(self.current_shared_filter_summary)}" if self.current_shared_filter_summary else ""
            source_name = self._describe_shared_source(self.shared_metadata)
            self.file_label.config(text=f"Compartido: {source_name} ({rows} filas, {cols} cols){filter_suffix}")
        except Exception:
            self.file_label.config(text="Dataset compartido disponible")

    def get_persistent_state(self):
        """Return a JSON-serializable snapshot of RSF UI configuration."""

        def _safe_get(var_name, default=None):
            var_obj = getattr(self, var_name, None)
            if var_obj is None:
                return default
            try:
                return var_obj.get()
            except Exception:
                return default

        selected_covariates = []
        if hasattr(self, "covariates_listbox"):
            try:
                selected_covariates = [
                    str(self.covariates_listbox.get(i)).strip()
                    for i in self.covariates_listbox.curselection()
                    if str(self.covariates_listbox.get(i)).strip()
                ]
            except Exception:
                selected_covariates = []

        state = {
            "version": 1,
            "duration_col": str(_safe_get("duration_var", "") or "").strip(),
            "event_col": str(_safe_get("event_var", "") or "").strip(),
            "selected_covariates": list(dict.fromkeys(selected_covariates)),
            "missing_strategy": str(_safe_get("missing_strategy_var", "Imputar mediana/moda") or "Imputar mediana/moda"),
            "drop_first": bool(_safe_get("drop_first_var", True)),
            "test_size": float(self._coerce_float(_safe_get("test_size_var", 0.25), 0.25, minimum=0.0, maximum=0.95)),
            "stratify_event": bool(_safe_get("stratify_event_var", True)),
            "optimization_metric": str(_safe_get("optimization_metric_var", "Harrell C-index") or "Harrell C-index"),
            "tau_mode": str(_safe_get("tau_mode_var", "Percentil 90") or "Percentil 90"),
            "tau_manual": str(_safe_get("tau_manual_var", "") or ""),
            "preset": str(_safe_get("preset_var", "Balanceado") or "Balanceado"),
            "n_estimators": int(self._coerce_int(_safe_get("n_estimators_var", 300), 300, minimum=10)),
            "max_features": str(_safe_get("max_features_var", "sqrt") or "sqrt"),
            "max_features_manual": str(_safe_get("max_features_manual_var", "") or ""),
            "min_samples_split": int(self._coerce_int(_safe_get("min_samples_split_var", 10), 10, minimum=2)),
            "min_samples_leaf": int(self._coerce_int(_safe_get("min_samples_leaf_var", 5), 5, minimum=1)),
            "max_depth": str(_safe_get("max_depth_var", "") or ""),
            "max_leaf_nodes": str(_safe_get("max_leaf_nodes_var", "") or ""),
            "bootstrap": bool(_safe_get("bootstrap_var", True)),
            "max_samples": str(_safe_get("max_samples_var", "") or ""),
            "oob_score": bool(_safe_get("oob_score_var", True)),
            "n_jobs": int(self._coerce_int(_safe_get("n_jobs_var", -1), -1)),
            "random_state": str(_safe_get("random_state_var", "42") or "42"),
            "cv_enabled": bool(_safe_get("cv_enabled_var", True)),
            "cv_folds": int(self._coerce_int(_safe_get("cv_folds_var", 5), 5, minimum=2)),
            "cv_metric": str(_safe_get("cv_metric_var", "C-Uno (IPCW)") or "C-Uno (IPCW)"),
            "tuning_profile": str(_safe_get("tuning_profile_var", "General") or "General"),
            "robust_vimp_mode": str(_safe_get("robust_vimp_mode_var", "Permisivo (IC95% sup > 0)") or "Permisivo (IC95% sup > 0)"),
            "corr_prune_threshold": str(_safe_get("corr_prune_threshold_var", "0.85") or "0.85"),
            "manual_trees": str(_safe_get("manual_trees_var", "") or ""),
            "manual_max_features": str(_safe_get("manual_max_features_grid_var", "") or ""),
            "manual_min_leaf": str(_safe_get("manual_min_leaf_grid_var", "") or ""),
            "manual_split_mult": str(_safe_get("manual_split_mult_grid_var", "") or ""),
            "manual_max_depth": str(_safe_get("manual_max_depth_grid_var", "") or ""),
            "manual_max_leaf_nodes": str(_safe_get("manual_max_leaf_nodes_grid_var", "") or ""),
            "manual_max_samples": str(_safe_get("manual_max_samples_grid_var", "") or ""),
            "tuning_models_mode": str(_safe_get("tuning_models_mode_var", "Multivariado") or "Multivariado"),
            "tuning_progress_metric": str(_safe_get("tuning_progress_metric_var", "C-Uno (IPCW)") or "C-Uno (IPCW)"),
            "tvt_min_covariates": str(_safe_get("tvt_min_covariates_var", "1") or "1"),
            "tvt_max_covariates": str(_safe_get("tvt_max_covariates_var", "") or ""),
            "tvt_max_combinations": str(_safe_get("tvt_max_combinations_var", "120") or "120"),
            "tvt_required_covariates": str(_safe_get("tvt_required_covariates_var", "") or ""),
            "oob_cindex": str(_safe_get("oob_cindex_var", "Harrell (nativo)") or "Harrell (nativo)"),
            "variable_configs": copy.deepcopy(getattr(self, "variable_configs", {})),
            "auto_tree_start": str(_safe_get("auto_tree_start_var", "100") or "100"),
            "auto_tree_step": str(_safe_get("auto_tree_step_var", "100") or "100"),
            "auto_tree_max": str(_safe_get("auto_tree_max_var", "1200") or "1200"),
            "early_stopping_enabled": bool(_safe_get("early_stopping_enabled_var", False)),
            "early_stopping_seeds": str(_safe_get("early_stopping_seeds_var", "2") or "2"),
            "early_stopping_gap": str(_safe_get("early_stopping_gap_var", "0.01") or "0.01"),
            "early_stopping_pass_stops": bool(_safe_get("early_stopping_pass_stops_var", False)),
            "early_stopping_impossible_stops": bool(_safe_get("early_stopping_impossible_stops_var", False)),
        }
        return state

    def load_persistent_state(self, state):
        """Restore RSF UI configuration previously returned by get_persistent_state."""
        if not isinstance(state, dict):
            return

        def _set_var(var_name, value):
            var_obj = getattr(self, var_name, None)
            if var_obj is None:
                return
            try:
                var_obj.set(value)
            except Exception:
                pass

        _duration = str(state.get("duration_col", "") or "").strip()
        _event = str(state.get("event_col", "") or "").strip()
        _set_var("duration_var", _duration)
        _set_var("event_var", _event)
        self._persistent_duration_pending = _duration
        self._persistent_event_pending = _event

        selected_covariates = [
            str(c).strip()
            for c in list(state.get("selected_covariates", []) or [])
            if str(c).strip()
        ]
        self._persistent_covariates_pending = list(dict.fromkeys(selected_covariates))

        _set_var("missing_strategy_var", str(state.get("missing_strategy", "Imputar mediana/moda") or "Imputar mediana/moda"))
        _set_var("drop_first_var", bool(state.get("drop_first", True)))
        _set_var("test_size_var", self._coerce_float(state.get("test_size", 0.25), 0.25, minimum=0.0, maximum=0.95))
        _set_var("stratify_event_var", bool(state.get("stratify_event", True)))
        _set_var("optimization_metric_var", str(state.get("optimization_metric", "Harrell C-index") or "Harrell C-index"))
        _set_var("tau_mode_var", str(state.get("tau_mode", "Percentil 90") or "Percentil 90"))
        _set_var("tau_manual_var", str(state.get("tau_manual", "") or ""))

        _set_var("preset_var", str(state.get("preset", "Balanceado") or "Balanceado"))
        _set_var("n_estimators_var", self._coerce_int(state.get("n_estimators", 300), 300, minimum=10))
        _set_var("max_features_var", str(state.get("max_features", "sqrt") or "sqrt"))
        _set_var("max_features_manual_var", str(state.get("max_features_manual", "") or ""))
        _set_var("min_samples_split_var", self._coerce_int(state.get("min_samples_split", 10), 10, minimum=2))
        _set_var("min_samples_leaf_var", self._coerce_int(state.get("min_samples_leaf", 5), 5, minimum=1))
        _set_var("max_depth_var", str(state.get("max_depth", "") or ""))
        _set_var("max_leaf_nodes_var", str(state.get("max_leaf_nodes", "") or ""))
        _set_var("bootstrap_var", bool(state.get("bootstrap", True)))
        _set_var("max_samples_var", str(state.get("max_samples", "") or ""))
        _set_var("oob_score_var", bool(state.get("oob_score", True)))
        _set_var("n_jobs_var", self._coerce_int(state.get("n_jobs", -1), -1))
        _set_var("random_state_var", str(state.get("random_state", "42") or "42"))
        _set_var("cv_enabled_var", bool(state.get("cv_enabled", True)))
        _set_var("cv_folds_var", self._coerce_int(state.get("cv_folds", 5), 5, minimum=2))
        _set_var("cv_metric_var", str(state.get("cv_metric", "C-Uno (IPCW)") or "C-Uno (IPCW)"))

        _set_var("tuning_profile_var", str(state.get("tuning_profile", "General") or "General"))
        _set_var("robust_vimp_mode_var", str(state.get("robust_vimp_mode", "Permisivo (IC95% sup > 0)") or "Permisivo (IC95% sup > 0)"))
        _set_var("corr_prune_threshold_var", str(state.get("corr_prune_threshold", "0.85") or "0.85"))
        _set_var("manual_trees_var", str(state.get("manual_trees", "") or ""))
        _set_var("manual_max_features_grid_var", str(state.get("manual_max_features", "") or ""))
        _set_var("manual_min_leaf_grid_var", str(state.get("manual_min_leaf", "") or ""))
        _set_var("manual_split_mult_grid_var", str(state.get("manual_split_mult", "") or ""))
        _set_var("manual_max_depth_grid_var", str(state.get("manual_max_depth", "") or ""))
        _set_var("manual_max_leaf_nodes_grid_var", str(state.get("manual_max_leaf_nodes", "") or ""))
        _set_var("manual_max_samples_grid_var", str(state.get("manual_max_samples", "") or ""))
        _set_var("tuning_models_mode_var", str(state.get("tuning_models_mode", "Multivariado") or "Multivariado"))
        _set_var("tuning_progress_metric_var", str(state.get("tuning_progress_metric", "C-Uno (IPCW)") or "C-Uno (IPCW)"))
        _set_var("tvt_min_covariates_var", str(state.get("tvt_min_covariates", "1") or "1"))
        _set_var("tvt_max_covariates_var", str(state.get("tvt_max_covariates", "") or ""))
        _set_var("tvt_max_combinations_var", str(state.get("tvt_max_combinations", "120") or "120"))
        _set_var("tvt_required_covariates_var", str(state.get("tvt_required_covariates", "") or ""))
        _set_var("oob_cindex_var", str(state.get("oob_cindex", "Harrell (nativo)") or "Harrell (nativo)"))

        if "variable_configs" in state:
            self.variable_configs = copy.deepcopy(state.get("variable_configs", {}))
            
        _set_var("auto_tree_start_var", str(state.get("auto_tree_start", "100") or "100"))
        _set_var("auto_tree_step_var", str(state.get("auto_tree_step", "100") or "100"))
        _set_var("auto_tree_max_var", str(state.get("auto_tree_max", "1200") or "1200"))
        _set_var("early_stopping_enabled_var", bool(state.get("early_stopping_enabled", False)))
        _set_var("early_stopping_seeds_var", str(state.get("early_stopping_seeds", "2") or "2"))
        _set_var("early_stopping_gap_var", str(state.get("early_stopping_gap", "0.01") or "0.01"))
        _set_var("early_stopping_pass_stops_var", bool(state.get("early_stopping_pass_stops", False)))
        _set_var("early_stopping_impossible_stops_var", bool(state.get("early_stopping_impossible_stops", False)))

        if hasattr(self, "_on_optimization_metric_change"):
            try:
                self._on_optimization_metric_change()
            except Exception:
                pass
        if hasattr(self, "_on_tau_mode_change"):
            try:
                self._on_tau_mode_change()
            except Exception:
                pass

        if isinstance(getattr(self, "data", None), pd.DataFrame):
            self._update_variable_comboboxes()

    # ------------------------------------------------------------------
    # Modeling
    # ------------------------------------------------------------------
    def _prepare_dataframe_for_rsf(self, input_df, duration_col, event_col, covariates):
        if input_df is None or input_df.empty:
            raise ValueError("No hay datos disponibles después del filtrado.")

        model_cols = [duration_col, event_col] + list(covariates)
        working_df = input_df[model_cols].copy()
        warnings_list = []

        working_df[duration_col] = pd.to_numeric(working_df[duration_col], errors="coerce")
        working_df[event_col] = pd.to_numeric(working_df[event_col], errors="coerce")

        invalid_time = working_df[duration_col].isna() | (working_df[duration_col] <= 0)
        if invalid_time.any():
            removed = int(invalid_time.sum())
            working_df = working_df.loc[~invalid_time].copy()
            warnings_list.append(f"Se descartaron {removed} filas por tiempo inválido o ≤ 0.")

        invalid_event = working_df[event_col].isna()
        if invalid_event.any():
            removed = int(invalid_event.sum())
            working_df = working_df.loc[~invalid_event].copy()
            warnings_list.append(f"Se descartaron {removed} filas por evento vacío/no numérico.")

        if working_df.empty:
            raise ValueError("No quedaron filas válidas luego de limpiar tiempo y evento.")

        strategy = self.missing_strategy_var.get()
        if strategy == "Eliminar filas incompletas":
            missing_mask = working_df[covariates].isna().any(axis=1)
            if missing_mask.any():
                removed = int(missing_mask.sum())
                working_df = working_df.loc[~missing_mask].copy()
                warnings_list.append(f"Se descartaron {removed} filas por faltantes en covariables.")
        else:
            for cov in covariates:
                series = working_df[cov]
                # Use _is_covariate_categorical: respects user config, falls back to dtype
                if not self._is_covariate_categorical(cov):
                    fill_value = series.median()
                    if pd.isna(fill_value):
                        fill_value = 0.0
                    working_df[cov] = series.fillna(fill_value)
                else:
                    modes = series.dropna().mode()
                    fill_value = modes.iloc[0] if not modes.empty else "(Vacío)"
                    working_df[cov] = series.fillna(fill_value)

        if working_df.empty:
            raise ValueError("No hay filas disponibles para entrenar el RSF.")

        event_bool = working_df[event_col].astype(float).fillna(0).clip(lower=0)
        event_bool = event_bool > 0
        if event_bool.nunique() < 2:
            raise ValueError("La variable de evento necesita al menos eventos y censuras para ajustar RSF.")

        X_raw = self._apply_categorical_configurations(working_df[covariates].copy(), covariates, warnings_list=warnings_list)
        X_encoded = pd.get_dummies(X_raw, drop_first=bool(self.drop_first_var.get()), dummy_na=False)

        if X_encoded.empty:
            raise ValueError("La codificación produjo una matriz vacía. Revise las covariables seleccionadas.")

        X_encoded = X_encoded.replace([np.inf, -np.inf], np.nan)
        if X_encoded.isna().any().any():
            X_encoded = X_encoded.fillna(0)
            warnings_list.append("Se reemplazaron valores no finitos en la matriz codificada por 0.")

        y_structured = Surv.from_arrays(event=event_bool.to_numpy(dtype=bool), time=working_df[duration_col].to_numpy(dtype=float))
        return working_df, X_encoded, y_structured, warnings_list

    def _compute_c_index(self, y_structured, risk_scores):
        try:
            return float(concordance_index_censored(y_structured["event"], y_structured["time"], risk_scores)[0])
        except Exception:
            return None

    def _resolve_optimization_metric_choice(self):
        raw_metric = getattr(self, "optimization_metric_var", None)
        raw_metric = raw_metric.get() if raw_metric else "Harrell C-index"
        raw_metric = str(raw_metric).strip().lower()
        if "uno" in raw_metric:
            return "uno", "Uno C-index", True
        if "ibs" in raw_metric:
            return "ibs", "IBS", False
        if "brier" in raw_metric:
            return "brier", "Brier Score", False
        return "harrell", "Harrell C-index", True

    def _on_optimization_metric_change(self):
        metric_code, _metric_label, _higher_better = self._resolve_optimization_metric_choice()
        show_tau = (metric_code == "uno")

        for widget in (
            getattr(self, "tau_label_widget", None),
            getattr(self, "tau_mode_combo", None),
            getattr(self, "tau_manual_entry", None),
            getattr(self, "tau_help_widget", None),
        ):
            if widget is None:
                continue
            if show_tau:
                widget.grid()
            else:
                widget.grid_remove()

        if show_tau and hasattr(self, "tau_mode_var") and not str(self.tau_mode_var.get()).strip():
            self.tau_mode_var.set("Percentil 90")

        self._on_tau_mode_change()

    def _on_tau_mode_change(self):
        tau_var = getattr(self, "tau_mode_var", None)
        tau_value = tau_var.get() if tau_var is not None and hasattr(tau_var, "get") else ""
        manual_mode = str(tau_value).strip().lower() == "manual"
        entry = getattr(self, "tau_manual_entry", None)
        if entry is not None:
            try:
                entry.configure(state=("normal" if manual_mode else "disabled"))
            except Exception:
                pass

    def _normalize_tau_mode(self, raw_mode):
        mode = str(raw_mode or "").strip().lower()
        if mode in {"percentil 90", "auto (p90)", "p90", "auto"}:
            return "Percentil 90"
        if mode in {"último caso", "ultimo caso", "último evento", "ultimo evento"}:
            return "Último caso"
        if mode == "manual":
            return "Manual"
        return "Percentil 90"

    def _resolve_tau_for_uno(self, y_train, y_test=None, raise_on_error=False):
        tau_mode_var = getattr(self, "tau_mode_var", None)
        tau_mode_value = tau_mode_var.get() if tau_mode_var is not None and hasattr(tau_mode_var, "get") else "Percentil 90"
        mode = self._normalize_tau_mode(tau_mode_value)

        train_times = np.asarray(y_train["time"], dtype=float)
        train_events = np.asarray(y_train["event"], dtype=bool)
        finite_train = train_times[np.isfinite(train_times)]
        event_times = train_times[train_events & np.isfinite(train_times)]

        if mode == "Manual":
            tau_manual_var = getattr(self, "tau_manual_var", None)
            raw_manual = tau_manual_var.get() if tau_manual_var is not None and hasattr(tau_manual_var, "get") else ""
            try:
                manual_tau = float(raw_manual)
            except Exception as exc:
                if raise_on_error:
                    raise ValueError("Tau manual debe ser numérico.") from exc
                return None

            if not np.isfinite(manual_tau) or manual_tau <= 0:
                if raise_on_error:
                    raise ValueError("Tau manual debe ser positivo.")
                return None

            if finite_train.size > 0:
                max_observed = float(np.nanmax(finite_train))
                if manual_tau > max_observed:
                    if raise_on_error:
                        raise ValueError(
                            f"Tau manual ({manual_tau:.4g}) excede el máximo tiempo observado ({max_observed:.4g})."
                        )
                    return None

            return float(manual_tau)

        if event_times.size == 0:
            if raise_on_error:
                raise ValueError(
                    "No hay tiempos con evento para calcular tau automático. "
                    "Use tau manual o verifique la variable de evento."
                )
            return None

        if mode == "Último caso":
            return float(np.nanmax(event_times))

        return float(np.nanpercentile(event_times, 90))

    def _resolve_cv_metric_choice(self):
        raw_metric = getattr(self, "cv_metric_var", None)
        raw_metric = raw_metric.get() if raw_metric else "C-Uno (IPCW)"
        raw_metric = str(raw_metric).strip()
        if "Antolini" in raw_metric:
            return "antolini", "C-Antolini (Ctd)"
        if "clásico" in raw_metric or raw_metric.lower() == "c clasico":
            return "harrell", "C clásico"
        return "uno", "C-Uno (IPCW)"

    def _resolve_oob_cindex_choice(self):
        """Return 'harrell', 'uno', or 'antolini' based on oob_cindex_var UI selection."""
        raw = getattr(self, "oob_cindex_var", None)
        raw = raw.get() if raw else "Harrell (nativo)"
        raw = str(raw).strip()
        if "Uno" in raw or "IPCW" in raw:
            return "uno"
        if "Antolini" in raw:
            return "antolini"
        return "harrell"

    def _compute_oob_cindex(self, model, y_train):
        """Compute OOB C-index using the method chosen in oob_cindex_var.

        Falls back to the native Harrell oob_score_ if OOB predictions are
        not available or the computation fails.
        """
        harrell_fallback = getattr(model, "oob_score_", None)
        oob_mode = self._resolve_oob_cindex_choice()
        if oob_mode == "harrell":
            return harrell_fallback

        # Need oob_prediction_ (set by sklearn/sksurv when oob_score=True)
        oob_preds = getattr(model, "oob_prediction_", None)
        if oob_preds is None or not hasattr(oob_preds, "__len__") or len(oob_preds) == 0:
            return harrell_fallback

        try:
            oob_arr = np.asarray(oob_preds, dtype=float).ravel()
            if len(oob_arr) != len(y_train):
                return harrell_fallback
            # scikit-survival sets 0 for samples with no OOB trees
            valid = np.isfinite(oob_arr) & (oob_arr != 0.0)
            if valid.sum() < 10:
                return harrell_fallback
            y_oob = y_train[valid]
            p_oob = oob_arr[valid]

            if oob_mode == "uno" and callable(concordance_index_ipcw):
                t_oob = np.asarray(y_oob["time"], dtype=float)
                tau = float(np.nanpercentile(t_oob, 80))
                result = concordance_index_ipcw(y_oob, y_oob, p_oob, tau=tau)
                return float(np.asarray(result).reshape(-1)[0])

            if oob_mode == "antolini" and callable(cumulative_dynamic_auc):
                t_oob = np.asarray(y_oob["time"], dtype=float)
                eval_times = np.nanpercentile(t_oob, [25, 50, 75])
                t_min, t_max = float(np.nanmin(t_oob)), float(np.nanmax(t_oob))
                eval_times = np.unique(eval_times[(eval_times > t_min) & (eval_times < t_max)])
                if len(eval_times) == 0:
                    return harrell_fallback
                _, mean_auc = cumulative_dynamic_auc(y_oob, y_oob, p_oob, eval_times)
                return float(mean_auc)
        except Exception:
            pass

        return harrell_fallback

    # ------------------------------------------------------------------
    # Multi-seed helpers
    # ------------------------------------------------------------------
    def _parse_random_seeds(self):
        """Parse random_state_var which may be one int or comma-separated ints.
        Returns a list of unique ints, e.g. [42, 7, 99].
        """
        raw = ""
        if hasattr(self, "random_state_var"):
            try:
                raw = str(self.random_state_var.get()).strip()
            except Exception:
                raw = "42"
        if not raw:
            return [42]
        seeds = []
        seen = set()
        for part in raw.replace(";", ",").split(","):
            part = part.strip()
            if not part:
                continue
            try:
                s = int(float(part))
                if s not in seen:
                    seeds.append(s)
                    seen.add(s)
            except Exception:
                pass
        return seeds if seeds else [42]

    def _average_seed_metrics(self, metrics_list):
        """Average scalar numeric metrics across multiple seed runs.
        Non-numeric / structural fields are taken from the first dict.
        """
        if not metrics_list:
            return {}
        if len(metrics_list) == 1:
            return dict(metrics_list[0])

        _SCALAR_KEYS = {
            "c_index_train", "c_index_test", "c_index_uno", "c_index_antolini",
            "oob_score", "c_index_cv_mean", "c_index_cv_std",
            "ibs", "ibs_km", "bss", "brier_at_eval_time",
            "brier_q25", "brier_q50", "brier_q75",
            "auroc_q25", "auroc_q50", "auroc_q75",
            "c_harrell_q25", "c_harrell_q50", "c_harrell_q75",
        }

        result = dict(metrics_list[0])
        for key in _SCALAR_KEYS:
            vals = []
            for m in metrics_list:
                v = m.get(key)
                if v is not None:
                    try:
                        vf = float(v)
                        if not np.isnan(vf):
                            vals.append(vf)
                    except (TypeError, ValueError):
                        pass
            result[key] = float(np.mean(vals)) if vals else None

        result["n_seeds"] = len(metrics_list)
        # Nullify CI fields when averaging (they are no longer single-run CIs)
        for ci_key in ("c_index_train_ci", "c_index_test_ci", "c_index_cv_ci",
                       "c_index_uno_ci", "c_index_antolini_ci", "ibs_ci", "bss_ci"):
            result[ci_key] = None
        return result

    def _fit_and_score_seed(self, seed, X_encoded, y_structured, rsf_params_base,
                             test_size, stratify_values, cv_metric_code,
                             resolved_tau_hint, eval_times_hint,
                             precomputed_split=None):
        """Fit RSF with a single seed, return (model, X_test, y_test, metrics_partial).
        Used for multi-seed averaging in run_model / run_auto_tuning.

        precomputed_split: optional (X_train, X_test, y_train, y_test) tuple.
        When provided, the function skips the internal train_test_split and uses
        these pre-split datasets directly. Only the RSF random_state changes between
        seeds. This ensures all screening-chart seeds are evaluated on identical data,
        making their metrics directly comparable to seed 1 from the main tuning path.
        """
        try:
            rsf_params = copy.deepcopy(rsf_params_base)
            rsf_params["random_state"] = int(seed)

            if precomputed_split is not None:
                X_train, X_test, y_train, y_test = precomputed_split
            elif test_size > 0 and len(X_encoded) > 2:
                X_train, X_test, y_train, y_test = train_test_split(
                    X_encoded, y_structured,
                    test_size=test_size,
                    random_state=int(seed),
                    stratify=stratify_values,
                )
            else:
                X_train = X_encoded.copy()
                X_test = X_encoded.iloc[0:0].copy()
                y_train = y_structured
                y_test = y_structured[:0]

            model = RandomSurvivalForest(**rsf_params)
            self._fit_model_with_ui_pump(model, X_train, y_train)

            train_preds = model.predict(X_train)
            test_preds = model.predict(X_test) if len(X_test) > 0 else np.asarray([], dtype=float)

            m = {}
            m["c_index_train"] = self._compute_c_index(y_train, train_preds)
            m["c_index_test"] = None
            m["c_index_uno"] = None
            m["c_index_antolini"] = None

            if cv_metric_code == "harrell" and len(X_test) > 0:
                m["c_index_test"] = self._compute_c_index(y_test, test_preds)
            if cv_metric_code == "uno" and len(X_test) > 0 and callable(concordance_index_ipcw):
                try:
                    r = concordance_index_ipcw(y_train, y_test, test_preds, tau=resolved_tau_hint)
                    m["c_index_uno"] = float(np.asarray(r).reshape(-1)[0])
                    m["c_index_test"] = m["c_index_uno"]
                except Exception:
                    pass
            if cv_metric_code == "antolini" and len(X_test) > 0:
                antolini = self._compute_c_antolini_score(
                    model, X_train, y_train, X_test, y_test,
                    eval_times=eval_times_hint, tau=resolved_tau_hint)
                m["c_index_antolini"] = antolini
                m["c_index_test"] = antolini

            m["oob_score"] = self._compute_oob_cindex(model, y_train)

            # CV C-index — force_enabled=True para que siempre se calcule
            # (igual que en el path principal de seed 1), independientemente
            # de si el usuario tiene CV desactivado en la UI.
            try:
                tev = np.asarray(y_train["event"], dtype=bool).astype(int)
                cv_r = self._compute_cv_cindex(X_train, tev, y_train, rsf_params, return_values=True, force_enabled=True)
                if isinstance(cv_r, tuple) and len(cv_r) == 3:
                    m["c_index_cv_mean"], m["c_index_cv_std"], _ = cv_r
                elif isinstance(cv_r, tuple) and len(cv_r) == 2:
                    m["c_index_cv_mean"], m["c_index_cv_std"] = cv_r
                else:
                    m["c_index_cv_mean"] = m["c_index_cv_std"] = None
            except Exception:
                m["c_index_cv_mean"] = m["c_index_cv_std"] = None

            # IBS / BSS
            m["ibs"] = m["ibs_km"] = m["bss"] = None
            if len(X_test) > 0 and eval_times_hint is not None:
                try:
                    _ibs, _ikm, _bss = self._compute_ibs_and_bss(
                        model, X_train, y_train, X_test, y_test, eval_times_hint)
                    m["ibs"] = _ibs
                    m["ibs_km"] = _ikm
                    m["bss"] = _bss
                except Exception:
                    pass
            # Fallback OOB-based IBS/BSS cuando no hay test set o falla el test-based
            if m.get("bss") is None:
                try:
                    _oi, _ok, _ob = self._compute_ibs_and_bss_oob(model, X_train, y_train)
                    if _ob is not None:
                        m["ibs"] = m["ibs"] or _oi
                        m["ibs_km"] = m["ibs_km"] or _ok
                        m["bss"] = _ob
                except Exception:
                    pass

            # Brier / AUROC / C quantiles
            for k in ("brier_q25", "brier_q50", "brier_q75",
                      "auroc_q25", "auroc_q50", "auroc_q75",
                      "c_harrell_q25", "c_harrell_q50", "c_harrell_q75",
                      "brier_at_eval_time"):
                m[k] = None

            if len(X_test) > 0 and eval_times_hint is not None and callable(brier_score):
                try:
                    _tt = np.asarray(y_train["time"], dtype=float)
                    _te = np.asarray(y_test["time"], dtype=float)
                    _max_t = float(np.nanmax(_tt[np.isfinite(_tt)]))
                    _valid = _te < _max_t
                    Xbs = X_test[_valid] if _valid.sum() >= 2 else X_test
                    ybs = y_test[_valid] if _valid.sum() >= 2 else y_test
                    _et = np.asarray(eval_times_hint, dtype=float)
                    _tbs = np.asarray(ybs["time"], dtype=float)
                    _ets = np.unique(_et[(_et >= float(np.nanmin(_tbs))) & (_et < float(np.nanmax(_tbs)))])
                    if _ets.size < 2:
                        _ets = eval_times_hint
                    sfns = model.predict_survival_function(Xbs)
                    smat = np.asarray([fn(_ets) for fn in sfns], dtype=float)
                    _, bv = brier_score(y_train, ybs, smat, _ets)
                    m["brier_at_eval_time"] = float(np.asarray(bv, dtype=float)[len(_ets) // 2])
                    ev_mask = ybs["event"].astype(bool)
                    etq = np.asarray(ybs["time"], dtype=float)[ev_mask]
                    if etq.size >= 4:
                        q25 = float(np.percentile(etq, 25))
                        q50 = float(np.percentile(etq, 50))
                        q75 = float(np.percentile(etq, 75))
                        qtimes = np.array([q25, q50, q75])
                        try:
                            sq = np.asarray([[fn(t) for t in qtimes] for fn in sfns], dtype=float)
                            _, bq = brier_score(y_train, ybs, sq, qtimes)
                            bq = np.asarray(bq, dtype=float)
                            m["brier_q25"] = float(bq[0])
                            m["brier_q50"] = float(bq[1])
                            m["brier_q75"] = float(bq[2])
                        except Exception:
                            pass
                        if callable(cumulative_dynamic_auc):
                            try:
                                rq = 1.0 - np.asarray([[fn(t) for t in qtimes] for fn in sfns], dtype=float)
                                aq, _ = cumulative_dynamic_auc(y_train, ybs, rq, qtimes)
                                aq = np.asarray(aq, dtype=float)
                                m["auroc_q25"] = float(aq[0])
                                m["auroc_q50"] = float(aq[1])
                                m["auroc_q75"] = float(aq[2])
                            except Exception:
                                pass
                        if callable(concordance_index_ipcw):
                            for tq, kk in [(q25, "c_harrell_q25"), (q50, "c_harrell_q50"), (q75, "c_harrell_q75")]:
                                try:
                                    r = concordance_index_ipcw(y_train, y_test, test_preds, tau=tq)
                                    m[kk] = float(np.asarray(r).reshape(-1)[0])
                                except Exception:
                                    pass
                except Exception:
                    pass

            return model, X_test, y_test, m
        except Exception:
            return None, None, None, {}

    def _resolve_cv_metric_labels(self):
        """Return explicit labels for CV mean and its equivalent test metric."""
        metric_code, metric_label = self._resolve_cv_metric_choice()
        cv_label = f"CV ({metric_label} folds)"
        if metric_code == "uno":
            test_eq_label = "Test C-Uno (IPCW)"
        elif metric_code == "antolini":
            test_eq_label = "Test C-Antolini (Ctd)"
        else:
            test_eq_label = "Test C-clásico"
        return cv_label, test_eq_label

    def _get_metric_display_label(self, key):
        """Human label for plot/report metrics with explicit CV family naming."""
        cv_label, _test_eq_label = self._resolve_cv_metric_labels()
        overrides = {
            "c_index_cv_mean": cv_label,
            "c_index_uno": "C-Uno (IPCW)",
            "c_index_test": "C-test (Harrell)",
            "c_index_train": "C-train (Harrell)",
            "c_index_antolini": "C-Antolini (Ctd)",
        }
        if key in overrides:
            return overrides[key]
        for k, lbl in self._EXPLORER_METRICS:
            if k == key:
                return lbl
        return key

    def _is_c_index_metric_key(self, key):
        return str(key or "").strip().startswith("c_index")

    def _is_c_like_metric_key(self, key):
        k = str(key or "").strip()
        return self._is_c_index_metric_key(k) or k == "oob_score"

    def _should_draw_coherence_diagonal(self, x_key, y_key):
        """Draw y=x when comparing C-index family with C-index/OOB scales."""
        xk = str(x_key or "").strip()
        yk = str(y_key or "").strip()
        return (
            self._is_c_like_metric_key(xk)
            and self._is_c_like_metric_key(yk)
            and (self._is_c_index_metric_key(xk) or self._is_c_index_metric_key(yk))
        )

    def _resolve_viability_metric_pair_for_plot(self, x_key, y_key):
        """Use chart axes for viability when both axes are C-index metrics."""
        xk = str(x_key or "").strip()
        yk = str(y_key or "").strip()
        if self._is_c_index_metric_key(xk) and self._is_c_index_metric_key(yk) and xk != yk:
            return xk, yk, self._get_metric_display_label(xk), self._get_metric_display_label(yk)
        cv_key, test_key, cv_lbl, test_lbl = self._resolve_clinical_stability_metric_keys()
        return cv_key, test_key, cv_lbl, test_lbl

    def _get_metric_ci_bounds(self, metrics, key):
        """Return (lower, upper) CI bounds for a metric key, or None."""
        if not isinstance(metrics, dict):
            return None

        key = str(key or "")
        ci_key_map = {
            "c_index_train": "c_index_train_ci",
            "c_index_test": "c_index_test_ci",
            "c_index_cv_mean": "c_index_cv_ci",
            "c_index_uno": "c_index_uno_ci",
            "c_index_antolini": "c_index_antolini_ci",
            "bss": "bss_ci",
            "ibs": "ibs_ci",
        }
        ci_value = metrics.get(ci_key_map.get(key, ""))
        if ci_value is None and key in ("c_index_uno", "c_index_antolini"):
            # Backward compatibility for snapshots where these CIs are mirrored
            # from test CI.
            ci_value = metrics.get("c_index_test_ci")

        if not isinstance(ci_value, (tuple, list, np.ndarray)) or len(ci_value) < 2:
            return None
        try:
            lo = float(ci_value[0])
            hi = float(ci_value[1])
        except Exception:
            return None
        if not (np.isfinite(lo) and np.isfinite(hi)):
            return None
        if lo > hi:
            lo, hi = hi, lo
        return (lo, hi)

    def _get_metric_ci_half_width(self, metrics, key):
        """Return a CI half-width for a metric, using C-like proxies when needed."""
        if not isinstance(metrics, dict):
            return None

        direct_bounds = self._get_metric_ci_bounds(metrics, key)
        if direct_bounds is not None:
            half_width = 0.5 * abs(float(direct_bounds[1]) - float(direct_bounds[0]))
            if np.isfinite(half_width) and half_width > 0.0:
                return float(half_width)

        key = str(key or "")
        if not (key == "oob_score" or self._is_c_index_metric_key(key)):
            return None

        proxy_ci_keys = (
            "c_index_cv_ci",
            "c_index_test_ci",
            "c_index_uno_ci",
            "c_index_antolini_ci",
            "c_index_train_ci",
        )
        widths = []
        for proxy_key in proxy_ci_keys:
            proxy_ci = metrics.get(proxy_key)
            if not isinstance(proxy_ci, (tuple, list, np.ndarray)) or len(proxy_ci) < 2:
                continue
            try:
                lo = float(proxy_ci[0])
                hi = float(proxy_ci[1])
            except Exception:
                continue
            if not (np.isfinite(lo) and np.isfinite(hi)):
                continue
            if lo > hi:
                lo, hi = hi, lo
            half_width = 0.5 * abs(hi - lo)
            if np.isfinite(half_width) and half_width > 0.0:
                widths.append(float(half_width))

        if not widths:
            return None
        return float(np.median(np.asarray(widths, dtype=float)))

    def _draw_ci_cross_for_point(self, ax, x_value, y_value, metrics, x_key, y_key,
                                 color="#475569", alpha=0.35, linewidth=0.9, zorder=1.6):
        """Draw CI as cross (horizontal + vertical whiskers) for one point."""
        drawn = False
        x_ci = self._get_metric_ci_bounds(metrics, x_key)
        y_ci = self._get_metric_ci_bounds(metrics, y_key)

        if x_ci is None:
            x_half = self._get_metric_ci_half_width(metrics, x_key)
            try:
                x_center = float(x_value)
            except Exception:
                x_center = None
            if x_half is not None and x_center is not None and np.isfinite(x_center):
                lo = float(x_center - x_half)
                hi = float(x_center + x_half)
                if self._is_c_like_metric_key(x_key):
                    lo = max(0.0, lo)
                    hi = min(1.0, hi)
                if hi > lo:
                    x_ci = (lo, hi)

        if y_ci is None:
            y_half = self._get_metric_ci_half_width(metrics, y_key)
            try:
                y_center = float(y_value)
            except Exception:
                y_center = None
            if y_half is not None and y_center is not None and np.isfinite(y_center):
                lo = float(y_center - y_half)
                hi = float(y_center + y_half)
                if self._is_c_like_metric_key(y_key):
                    lo = max(0.0, lo)
                    hi = min(1.0, hi)
                if hi > lo:
                    y_ci = (lo, hi)

        if x_ci is not None:
            ax.hlines(float(y_value), float(x_ci[0]), float(x_ci[1]),
                      colors=color, alpha=alpha, linewidth=linewidth, zorder=zorder)
            drawn = True
        if y_ci is not None:
            ax.vlines(float(x_value), float(y_ci[0]), float(y_ci[1]),
                      colors=color, alpha=alpha, linewidth=linewidth, zorder=zorder)
            drawn = True
        return drawn

    def _compute_cv_metric_value(self, model, X_train, y_train, X_test, y_test, metric_code=None):
        metric_code = metric_code or self._resolve_cv_metric_choice()[0]
        try:
            preds = model.predict(X_test)
        except Exception:
            return None

        if metric_code == "harrell":
            return self._compute_c_index(y_test, preds)

        if metric_code == "uno":
            if not callable(concordance_index_ipcw):
                return None
            tau_cv = self._resolve_tau(y_train, y_test)
            if tau_cv is None:
                return None
            try:
                result = concordance_index_ipcw(y_train, y_test, preds, tau=tau_cv)
                return float(np.asarray(result).reshape(-1)[0])
            except Exception:
                return None

        if metric_code == "antolini":
            return self._compute_c_antolini_score(
                model, X_train, y_train, X_test, y_test,
                eval_times=self._build_evaluation_time_grid(y_train, y_test, tau=self._resolve_tau(y_train, y_test)),
                tau=self._resolve_tau(y_train, y_test),
            )

        return None

    def _compute_c_index_ci(self, y_structured, risk_scores, n_bootstrap=120, random_state=42):
        if y_structured is None or risk_scores is None:
            return None

        risk_scores = np.asarray(risk_scores, dtype=float).reshape(-1)
        if risk_scores.size < 5:
            return None

        try:
            event_values = np.asarray(y_structured["event"], dtype=bool)
        except Exception:
            return None
        if np.unique(event_values).size < 2:
            return None

        rng = np.random.default_rng(random_state)
        population_idx = np.arange(risk_scores.size)
        sampled_scores = []
        for _ in range(int(max(20, n_bootstrap))):
            bootstrap_idx = rng.choice(population_idx, size=risk_scores.size, replace=True)
            y_sample = y_structured[bootstrap_idx]
            if np.unique(np.asarray(y_sample["event"], dtype=bool)).size < 2:
                continue
            c_value = self._compute_c_index(y_sample, risk_scores[bootstrap_idx])
            if c_value is not None and np.isfinite(c_value):
                sampled_scores.append(c_value)

        if len(sampled_scores) < 10:
            return None

        alpha = 0.05
        lower = float(np.nanquantile(sampled_scores, alpha / 2.0))
        upper = float(np.nanquantile(sampled_scores, 1.0 - (alpha / 2.0)))
        lower = float(np.clip(lower, 0.0, 1.0))
        upper = float(np.clip(upper, 0.0, 1.0))
        return (min(lower, upper), max(lower, upper))

    def _compute_cv_cindex(self, X, event_values, y_structured, rsf_params, force_enabled=False, folds_override=None, return_values=False):
        if not (force_enabled or bool(self.cv_enabled_var.get())):
            return (None, None, []) if return_values else (None, None)

        raw_folds = self.cv_folds_var.get() if folds_override is None else folds_override
        folds_requested = self._coerce_int(raw_folds, 5, minimum=2)
        event_int = pd.Series(event_values).astype(int)
        min_class_size = int(event_int.value_counts().min()) if not event_int.empty else 0
        folds = min(folds_requested, min_class_size) if min_class_size >= 2 else 0
        if folds < 2:
            return (None, None, []) if return_values else (None, None)

        cindex_values = []
        metric_code, _metric_label = self._resolve_cv_metric_choice()
        cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=rsf_params.get("random_state", 42))
        # Siempre usar el pump de UI para mantener la ventana responsiva (resize, Cancel)
        # durante los folds de validación cruzada, tanto en carga manual como en autotuning.
        for train_idx, test_idx in cv.split(X, event_int):
            if bool(getattr(self, "_auto_tuning_in_progress", False)) and bool(getattr(self, "_tuning_cancel_requested", False)):
                raise InterruptedError("Cancelado por el usuario.")
            model_cv = RandomSurvivalForest(**rsf_params)
            X_train_cv = X.iloc[train_idx]
            X_test_cv = X.iloc[test_idx]
            y_train_cv = y_structured[train_idx]
            y_test_cv = y_structured[test_idx]
            self._fit_model_with_ui_pump(model_cv, X_train_cv, y_train_cv)
            cindex = self._compute_cv_metric_value(
                model_cv, X_train_cv, y_train_cv, X_test_cv, y_test_cv, metric_code=metric_code)
            if cindex is not None:
                cindex_values.append(cindex)

        if not cindex_values:
            return (None, None, []) if return_values else (None, None)

        mean_value = float(np.mean(cindex_values))
        std_value = float(np.std(cindex_values))
        return (mean_value, std_value, list(cindex_values)) if return_values else (mean_value, std_value)

    def _get_current_rsf_params(self, n_features):
        min_samples_leaf, min_samples_split = self._align_split_leaf_params(
            self.min_samples_leaf_var.get(),
            self.min_samples_split_var.get(),
        )
        self.min_samples_leaf_var.set(min_samples_leaf)
        self.min_samples_split_var.set(min_samples_split)

        rsf_params = {
            "n_estimators": self._coerce_int(self.n_estimators_var.get(), 300, minimum=10),
            "max_features": self._resolve_max_features(self.max_features_var.get(), n_features),
            "min_samples_split": min_samples_split,
            "min_samples_leaf": min_samples_leaf,
            "max_depth": self._parse_optional_int(self.max_depth_var.get()),
            "max_leaf_nodes": self._parse_optional_int(self.max_leaf_nodes_var.get()),
            "bootstrap": bool(self.bootstrap_var.get()),
            "oob_score": bool(self.oob_score_var.get()) and bool(self.bootstrap_var.get()),
            "n_jobs": self._coerce_int(self.n_jobs_var.get(), -1),
            "random_state": self._parse_random_seeds()[0],
        }

        max_samples = self._parse_optional_float(self.max_samples_var.get())
        if max_samples is not None:
            rsf_params["max_samples"] = max_samples

        return {key: value for key, value in rsf_params.items() if value is not None}

    def _suggest_tuning_profile(self, n_rows):
        if int(n_rows) < 200:
            return "Pocos datos (<200)"
        if int(n_rows) <= 500:
            return "Mediano (200-500)"
        return "Pesado (>500)"

    def _format_manual_tuning_selection(self, values):
        cleaned_values = []
        for value in values or []:
            text = str(value).strip()
            if text:
                cleaned_values.append(text)
        return ",".join(cleaned_values)

    def _get_manual_tuning_selector_specs(self):
        return {
            "trees": {
                "label": "Árboles",
                "variable": getattr(self, "manual_trees_var", None),
                "options": ["50", "100", "150", "200", "300", "400", "500", "700", "1000"],
            },
            "max_features": {
                "label": "max_features",
                "variable": getattr(self, "manual_max_features_grid_var", None),
                "options": ["sqrt", "log2", "0.1", "0.3", "0.5", "0.7", "all"],
            },
            "min_leaf": {
                "label": "min_leaf",
                "variable": getattr(self, "manual_min_leaf_grid_var", None),
                "options": ["2", "3", "4", "5", "6", "7", "8", "9", "10"],
            },
            "split_mult": {
                "label": "Mult. split (M)",
                "variable": getattr(self, "manual_split_mult_grid_var", None),
                "options": ["2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22"],
            },
            "max_depth": {
                "label": "max_depth",
                "variable": getattr(self, "manual_max_depth_grid_var", None),
                "options": ["none", "3", "5", "8", "10", "15", "20"],
            },
            "max_leaf_nodes": {
                "label": "max_leaf_nodes",
                "variable": getattr(self, "manual_max_leaf_nodes_grid_var", None),
                "options": ["none", "10", "15", "20", "30", "50", "100"],
            },
            "max_samples": {
                "label": "max_samples",
                "variable": getattr(self, "manual_max_samples_grid_var", None),
                "options": ["none", "0.3", "0.5", "0.7", "0.8", "0.9", "1.0"],
            },
        }

    def _open_manual_tuning_selector_dialog(self):
        selector_specs = self._get_manual_tuning_selector_specs()
        dialog = tk.Toplevel(self.winfo_toplevel())
        dialog.title("Selector múltiple - tuning manual RSF")
        dialog.transient(self.winfo_toplevel())
        dialog.grab_set()
        dialog.resizable(True, True)

        ttk.Label(
            dialog,
            text="Usa Ctrl o Shift para seleccionar varios valores en cada lista.",
            foreground="#555555",
        ).pack(anchor="w", padx=10, pady=(10, 6))

        grid_frame = ttk.Frame(dialog)
        grid_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        for col_idx in range(2):
            grid_frame.columnconfigure(col_idx, weight=1)

        selector_widgets = {}
        for idx, (key, spec) in enumerate(selector_specs.items()):
            frame = ttk.LabelFrame(grid_frame, text=spec["label"])
            frame.grid(row=idx // 2, column=idx % 2, padx=6, pady=6, sticky="nsew")
            frame.columnconfigure(0, weight=1)

            listbox = tk.Listbox(frame, selectmode=tk.EXTENDED, exportselection=False, height=min(8, len(spec["options"])))
            listbox.grid(row=0, column=0, padx=4, pady=4, sticky="nsew")
            for option in spec["options"]:
                listbox.insert(tk.END, option)

            raw_current = spec["variable"].get() if spec.get("variable") is not None else ""
            current_values = {text.strip().lower() for text in str(raw_current).split(",") if text.strip()}
            for option_index, option_value in enumerate(spec["options"]):
                if str(option_value).strip().lower() in current_values:
                    listbox.selection_set(option_index)

            selector_widgets[key] = listbox

        buttons_frame = ttk.Frame(dialog)
        buttons_frame.pack(fill=tk.X, padx=10, pady=(0, 10))

        def apply_selection():
            for key, spec in selector_specs.items():
                variable = spec.get("variable")
                if variable is None:
                    continue
                selected_values = [selector_widgets[key].get(item_index) for item_index in selector_widgets[key].curselection()]
                if selected_values:
                    variable.set(self._format_manual_tuning_selection(selected_values))
            dialog.destroy()

        ttk.Button(buttons_frame, text="Aplicar", command=apply_selection).pack(side=tk.RIGHT, padx=4)
        ttk.Button(buttons_frame, text="Cancelar", command=dialog.destroy).pack(side=tk.RIGHT, padx=4)
        dialog.wait_window()

    def _parse_manual_tuning_values(self, raw_text, cast_type="int"):
        values = []
        for part in str(raw_text).split(','):
            text = part.strip()
            if not text:
                continue
            try:
                if cast_type == "int":
                    values.append(int(float(text)))
                elif cast_type == "float":
                    values.append(float(text))
                else:
                    values.append(text)
            except Exception:
                continue
        return values

    def _parse_optional_manual_values(self, raw_text, value_kind="int"):
        values = []
        for part in str(raw_text).split(','):
            text = part.strip()
            if not text:
                continue
            if text.lower() in {"none", "null", "na", "n/a", ""}:
                values.append(None)
                continue
            try:
                if value_kind == "int":
                    values.append(int(float(text)))
                elif value_kind == "float":
                    values.append(float(text))
                else:
                    values.append(text)
            except Exception:
                continue
        return values

    def _set_tvt_required_covariates_from_selection(self):
        if not hasattr(self, "tvt_required_covariates_var"):
            return
        if not hasattr(self, "covariates_listbox"):
            return

        selected_covariates = []
        try:
            selected_indices = list(self.covariates_listbox.curselection())
        except Exception:
            selected_indices = []

        for idx in selected_indices:
            try:
                cov_text = str(self.covariates_listbox.get(idx)).strip()
            except Exception:
                cov_text = ""
            if cov_text and cov_text not in selected_covariates:
                selected_covariates.append(cov_text)

        self.tvt_required_covariates_var.set(",".join(selected_covariates))

    def _get_tvt_required_covariates(self, available_covariates):
        """Returns list of OR-groups (each a list of canonical covariate names).
        Syntax:
          A,B,C   → [[A],[B],[C]]  all required (AND)
          (A|B),C → [[A,B],[C]]   one of A or B, AND C
          (A|B|C) → [[A,B,C]]    at least one of A, B, or C
        """
        available = [str(c).strip() for c in list(available_covariates or []) if str(c).strip()]
        if not available:
            return []

        raw_text = self.tvt_required_covariates_var.get() if hasattr(self, "tvt_required_covariates_var") else ""
        if not str(raw_text).strip():
            return []

        available_by_lower = {name.lower(): name for name in available}
        # Tokenize at top-level commas (respecting parentheses)
        tokens = []
        depth = 0
        current: list = []
        for ch in str(raw_text).strip():
            if ch == '(':
                depth += 1
                current.append(ch)
            elif ch == ')':
                depth = max(0, depth - 1)
                current.append(ch)
            elif ch == ',' and depth == 0:
                tokens.append(''.join(current).strip())
                current = []
            else:
                current.append(ch)
        if current:
            tokens.append(''.join(current).strip())

        groups = []
        for token in tokens:
            token = token.strip()
            if not token:
                continue
            if token.startswith('(') and token.endswith(')'):
                # OR group explicit: (A|B|C)
                inner = token[1:-1]
                parts = [p.strip() for p in inner.split('|')]
                group = []
                for part in parts:
                    canonical = available_by_lower.get(part.lower())
                    if canonical and canonical not in group:
                        group.append(canonical)
                if group:
                    groups.append(group)
            elif '|' in token:
                # OR group bare: A|B|C  (sin paréntesis — igual se interpreta como OR)
                parts = [p.strip() for p in token.split('|')]
                group = []
                for part in parts:
                    canonical = available_by_lower.get(part.lower())
                    if canonical and canonical not in group:
                        group.append(canonical)
                if group:
                    groups.append(group)
            else:
                # Single name (AND term)
                canonical = available_by_lower.get(token.lower())
                if canonical:
                    groups.append([canonical])
        return groups

    def _get_tvt_min_covariates(self, n_available, n_required=0):
        n_available = max(1, int(n_available or 1))
        raw_min = self.tvt_min_covariates_var.get() if hasattr(self, "tvt_min_covariates_var") else 1
        min_covariates = self._coerce_int(raw_min, 1, minimum=1)
        min_covariates = max(int(min_covariates), int(max(0, n_required)))
        return min(int(min_covariates), int(n_available))

    def _get_tvt_max_covariates(self, n_available, min_covariates=1):
        n_available = max(1, int(n_available or 1))
        min_covariates = max(1, int(min_covariates or 1))
        raw_max = ""
        if hasattr(self, "tvt_max_covariates_var"):
            try:
                raw_max = str(self.tvt_max_covariates_var.get() or "").strip()
            except Exception:
                raw_max = ""

        if not raw_max:
            max_covariates = n_available
        else:
            max_covariates = self._coerce_int(raw_max, n_available, minimum=min_covariates)

        max_covariates = max(int(max_covariates), int(min_covariates))
        return min(int(max_covariates), int(n_available))

    def _get_tvt_max_combinations(self):
        raw_max = ""
        if hasattr(self, "tvt_max_combinations_var"):
            try:
                raw_max = str(self.tvt_max_combinations_var.get() or "").strip()
            except Exception:
                raw_max = ""
        if not raw_max:
            return 120
        return self._coerce_int(raw_max, 120, minimum=1)

    def _build_manual_profile_config(self):
        trees = [value for value in self._parse_manual_tuning_values(self.manual_trees_var.get(), "int") if value >= 10]
        max_features = self._parse_manual_tuning_values(self.manual_max_features_grid_var.get(), "text")
        min_leaf = [value for value in self._parse_manual_tuning_values(self.manual_min_leaf_grid_var.get(), "int") if value >= 1]
        split_mults = [value for value in self._parse_manual_tuning_values(self.manual_split_mult_grid_var.get(), "float") if value >= 2.0]
        max_depth = self._parse_optional_manual_values(self.manual_max_depth_grid_var.get(), "int")
        max_leaf_nodes = self._parse_optional_manual_values(self.manual_max_leaf_nodes_grid_var.get(), "int")
        max_samples = self._parse_optional_manual_values(self.manual_max_samples_grid_var.get(), "float")

        if not trees:
            trees = [100, 300, 500]
        if not max_features:
            max_features = ["sqrt", "log2", "0.5", "all"]
        if not min_leaf:
            min_leaf = [3, 5, 10]
        if not split_mults:
            split_mults = [2.0, 2.5, 3.0]
        if not max_depth:
            max_depth = [None]
        if not max_leaf_nodes:
            max_leaf_nodes = [None]
        if not max_samples:
            max_samples = [None]

        return {
            "trees": sorted(set(int(x) for x in trees)),
            "max_features": [str(x).strip() for x in max_features if str(x).strip()],
            "min_leaf": sorted(set(int(x) for x in min_leaf)),
            "split_multipliers": sorted(set(float(x) for x in split_mults)),
            "max_depth": list(dict.fromkeys(max_depth)),
            "max_leaf_nodes": list(dict.fromkeys(max_leaf_nodes)),
            "max_samples": list(dict.fromkeys(max_samples)),
        }

    def _resolve_tuning_evaluation_mode(self):
        raw_mode = str(self.tuning_models_mode_var.get()).strip().lower() if hasattr(self, "tuning_models_mode_var") else "multivariado"
        if raw_mode == "univariado":
            return "Univariado"
        if raw_mode == "ambos":
            return "Ambos"
        if raw_mode == "todos contra todos":
            return "TodosContraTodos"
        return "Multivariado"

    def _build_tuning_feature_sets(self, covariates, evaluation_mode):
        from itertools import combinations as _combinations

        def _scope_label_from_covariates(cov_list):
            cov_text = [str(c).strip() for c in list(cov_list or []) if str(c).strip()]
            return ", ".join(cov_text) if cov_text else "(sin covariables)"

        unique_covariates = []
        for cov in list(covariates or []):
            cov_text = str(cov).strip()
            if cov_text and cov_text not in unique_covariates:
                unique_covariates.append(cov_text)

        if not unique_covariates:
            return []

        if evaluation_mode == "Univariado":
            return [
                {"scope_id": f"uni::{cov}", "scope_label": _scope_label_from_covariates([cov]), "covariates": [cov]}
                for cov in unique_covariates
            ]

        if evaluation_mode == "Ambos":
            feature_sets = [{
                "scope_id": "multi::all",
                "scope_label": _scope_label_from_covariates(unique_covariates),
                "covariates": list(unique_covariates),
            }]
            feature_sets.extend(
                {"scope_id": f"uni::{cov}", "scope_label": _scope_label_from_covariates([cov]), "covariates": [cov]}
                for cov in unique_covariates
            )
            return feature_sets

        if evaluation_mode == "TodosContraTodos":
            n = len(unique_covariates)
            feature_sets = []
            required_groups = self._get_tvt_required_covariates(unique_covariates)
            min_covariates = self._get_tvt_min_covariates(n, n_required=len(required_groups))
            max_covariates = self._get_tvt_max_covariates(n, min_covariates=min_covariates)
            max_combinations = self._get_tvt_max_combinations()

            for size in range(int(min_covariates), int(max_covariates) + 1):
                for combo in _combinations(unique_covariates, size):
                    combo_list = list(combo)
                    if required_groups:
                        combo_set = set(combo_list)
                        if not all(any(c in combo_set for c in grp) for grp in required_groups):
                            continue
                    if size == 1:
                        label = _scope_label_from_covariates(combo_list)
                        sid = f"uni::{combo_list[0]}"
                    elif size == n:
                        label = _scope_label_from_covariates(combo_list)
                        sid = "multi::all"
                    else:
                        label = _scope_label_from_covariates(combo_list)
                        sid = f"combo::{'|'.join(combo_list)}"
                    feature_sets.append({
                        "scope_id": sid,
                        "scope_label": label,
                        "covariates": combo_list,
                    })
                    if len(feature_sets) >= int(max_combinations):
                        break
                if len(feature_sets) >= int(max_combinations):
                    break
            return feature_sets

        return [{
            "scope_id": "multi::all",
            "scope_label": _scope_label_from_covariates(unique_covariates),
            "covariates": list(unique_covariates),
        }]

    def _generate_tuning_candidates(self, n_features, n_rows, n_train=None, auto_trees=False):
        selected_profile = self.tuning_profile_var.get().strip() if hasattr(self, "tuning_profile_var") else "General"
        if not selected_profile:
            selected_profile = "General"

        profile_configs = {
            "General": {
                "trees": [100, 200, 300, 400, 500],
                "max_features": ["sqrt", "log2", "0.5", "all"],
                "min_leaf": [5, 10, 15],
                "split_multipliers": [2.0, 2.5, 3.0],
            },
            "Pocos datos (<200)": {
                "trees": [100, 200, 300, 400],
                "max_features": ["sqrt", "log2", "0.5", "all"],
                "min_leaf": [5, 10, 15],
                "split_multipliers": [2.5, 3.0, 4.0],
            },
            "Mediano (200-500)": {
                "trees": [200, 300, 400, 500, 700],
                "max_features": ["sqrt", "log2", "0.5", "0.7", "all"],
                "min_leaf": [3, 5, 10],
                "split_multipliers": [2.0, 2.5, 3.0],
            },
            "Pesado (>500)": {
                "trees": [300, 500, 800, 1000],
                "max_features": ["sqrt", "log2", "0.3", "0.5", "0.7"],
                "min_leaf": [3, 5, 10],
                "split_multipliers": [2.0, 3.0, 5.0],
            },
        }

        if selected_profile == "Manual":
            config = self._build_manual_profile_config()
        else:
            config = profile_configs.get(selected_profile, profile_configs["General"])
        recommended_profile = self._suggest_tuning_profile(n_rows)
        grouped_candidates = {str(key): [] for key in config["max_features"]}
        seen = set()

        max_depth_values = config.get("max_depth", [None])
        max_leaf_nodes_values = config.get("max_leaf_nodes", [None])
        max_samples_values = config.get("max_samples", [None])

        _split_cap = (n_train // 2) if n_train and n_train > 0 else None

        tree_values = list(config["trees"])
        if bool(auto_trees):
            try:
                _ui_trees = self._coerce_int(self.n_estimators_var.get(), self._AUTO_TREE_MAX, minimum=10)
            except Exception:
                _ui_trees = self._AUTO_TREE_MAX
            if int(_ui_trees) <= int(self._AUTO_TREE_START):
                _ui_trees = int(self._AUTO_TREE_MAX)
            _start_auto = min(self._coerce_int(self._AUTO_TREE_START, 100, minimum=10), int(_ui_trees))
            tree_values = [max(10, int(_start_auto))]

        for n_estimators in tree_values:
            for min_samples_leaf in config["min_leaf"]:
                for split_mult in config["split_multipliers"]:
                    min_samples_split = max(2, int(min_samples_leaf * split_mult))
                    if _split_cap is not None and min_samples_split > _split_cap:
                        continue
                    if selected_profile == "Pocos datos (<200)" and min_samples_leaf < 5:
                        continue
                    for max_depth in max_depth_values:
                        for max_leaf_nodes in max_leaf_nodes_values:
                            for max_samples in max_samples_values:
                                for max_features in config["max_features"]:
                                    display_max_features = str(max_features)
                                    if selected_profile == "Pocos datos (<200)" and display_max_features == "all" and min_samples_leaf <= 5 and n_estimators >= 300:
                                        continue
                                    if selected_profile == "Mediano (200-500)" and display_max_features == "all" and min_samples_leaf <= 3:
                                        continue
                                    if selected_profile == "Pesado (>500)" and display_max_features == "0.7" and min_samples_leaf <= 3 and n_estimators >= 800:
                                        continue

                                    resolved_max_features = self._resolve_max_features(display_max_features, n_features)
                                    effective_max_features = self._resolve_max_features_effective_count(
                                        resolved_max_features,
                                        n_features,
                                    )
                                    key = (
                                        int(n_estimators),
                                        int(effective_max_features),
                                        int(min_samples_leaf),
                                        int(min_samples_split),
                                        str(max_depth),
                                        str(max_leaf_nodes),
                                        str(max_samples),
                                    )
                                    if key in seen:
                                        continue
                                    seen.add(key)

                                    grouped_candidates.setdefault(display_max_features, []).append(
                                        {
                                            "display": {
                                                "n_estimators": int(n_estimators),
                                                "max_features": display_max_features,
                                                "min_samples_leaf": int(min_samples_leaf),
                                                "min_samples_split": int(min_samples_split),
                                                "split_multiplier": round(split_mult, 2),
                                                "max_depth": max_depth,
                                                "max_leaf_nodes": max_leaf_nodes,
                                                "max_samples": max_samples,
                                            },
                                            "params": {
                                                "n_estimators": int(n_estimators),
                                                "max_features": resolved_max_features,
                                                "min_samples_leaf": int(min_samples_leaf),
                                                "min_samples_split": int(min_samples_split),
                                                "max_depth": max_depth,
                                                "max_leaf_nodes": max_leaf_nodes,
                                                "max_samples": max_samples,
                                            },
                                        }
                                    )

        candidates = []
        ordered_keys = [str(value) for value in config["max_features"]]
        max_candidates = float('inf') if selected_profile == "Manual" else 30
        while len(candidates) < max_candidates and any(grouped_candidates.get(key) for key in ordered_keys):
            for key in ordered_keys:
                bucket = grouped_candidates.get(key, [])
                if bucket:
                    candidates.append(bucket.pop(0))
                    if len(candidates) >= max_candidates:
                        break

        return candidates, selected_profile, recommended_profile

    def _compute_tuning_growth_metric(self, model, X_train, y_train, X_test, y_test, cv_metric_code,
                                      eval_times_tuning=None, resolved_tau_tuning=None):
        """Metric used to detect tree-growth plateau during auto-tuning."""
        oob_val = getattr(model, "oob_score_", None)
        if oob_val is not None and np.isfinite(oob_val):
            return float(oob_val), "OOB"

        if len(X_test) > 0:
            try:
                test_preds = model.predict(X_test)
            except Exception:
                test_preds = None

            if test_preds is not None:
                if cv_metric_code == "uno" and callable(concordance_index_ipcw):
                    try:
                        uno_result = concordance_index_ipcw(y_train, y_test, test_preds, tau=resolved_tau_tuning)
                        return float(np.asarray(uno_result).reshape(-1)[0]), "C-Uno"
                    except Exception:
                        pass
                if cv_metric_code == "antolini":
                    antolini = self._compute_c_antolini_score(
                        model,
                        X_train,
                        y_train,
                        X_test,
                        y_test,
                        eval_times=eval_times_tuning,
                        tau=resolved_tau_tuning,
                    )
                    if antolini is not None and np.isfinite(antolini):
                        return float(antolini), "C-Antolini"
                c_test = self._compute_c_index(y_test, test_preds)
                if c_test is not None and np.isfinite(c_test):
                    return float(c_test), "C-test"

        try:
            train_preds = model.predict(X_train)
            c_train = self._compute_c_index(y_train, train_preds)
            if c_train is not None and np.isfinite(c_train):
                return float(c_train), "C-train"
        except Exception:
            pass

        return None, "N/D"

    def _fit_rsf_until_plateau(self, candidate_params, X_train, y_train, X_test, y_test,
                               cv_metric_code, eval_times_tuning=None, resolved_tau_tuning=None,
                               warm_start_supported=True, max_trees_limit=None):
        """Grow trees incrementally and stop when improvements plateau."""
        start_trees = self._coerce_int(candidate_params.get("n_estimators", self._AUTO_TREE_START), self._AUTO_TREE_START, minimum=10)
        step = self._coerce_int(self._AUTO_TREE_STEP, 100, minimum=10)
        min_trees = max(self._coerce_int(self._AUTO_TREE_MIN, 200, minimum=10), start_trees)
        _configured_max = self._coerce_int(self._AUTO_TREE_MAX, 1200, minimum=min_trees)
        if max_trees_limit is not None:
            try:
                _configured_max = min(int(_configured_max), int(max_trees_limit))
            except Exception:
                pass
        max_trees = max(int(_configured_max), min_trees)
        delta_min = float(max(self._AUTO_TREE_DELTA_MIN, 0.0))
        patience = self._coerce_int(self._AUTO_TREE_PATIENCE, 3, minimum=1)

        tree_values = list(range(start_trees, max_trees + 1, step))
        if not tree_values:
            tree_values = [start_trees]
        if tree_values[-1] != max_trees:
            tree_values.append(max_trees)

        model = None
        last_trees = None
        best_trees = None
        best_metric = None
        no_improve_steps = 0
        metric_label = "N/D"
        metric_history = []

        for n_trees in tree_values:
            if bool(getattr(self, "_tuning_cancel_requested", False)):
                raise InterruptedError("Cancelado por el usuario.")
            if model is None or not warm_start_supported:
                fit_params = dict(candidate_params)
                fit_params["n_estimators"] = int(n_trees)
                if warm_start_supported:
                    fit_params["warm_start"] = True
                model = RandomSurvivalForest(**fit_params)
            else:
                model.set_params(warm_start=True, n_estimators=int(n_trees))

            self._fit_model_with_ui_pump(model, X_train, y_train)
            last_trees = int(n_trees)

            metric_value, metric_label = self._compute_tuning_growth_metric(
                model,
                X_train,
                y_train,
                X_test,
                y_test,
                cv_metric_code,
                eval_times_tuning=eval_times_tuning,
                resolved_tau_tuning=resolved_tau_tuning,
            )
            metric_history.append((int(n_trees), metric_value))

            improved = False
            if metric_value is not None and np.isfinite(metric_value):
                if best_metric is None or float(metric_value) > float(best_metric) + delta_min:
                    best_metric = float(metric_value)
                    best_trees = int(n_trees)
                    no_improve_steps = 0
                    improved = True

            if not improved:
                no_improve_steps += 1

            if int(n_trees) >= int(min_trees) and no_improve_steps >= patience:
                break

        if best_trees is None:
            best_trees = int(last_trees or start_trees)

        if last_trees is None:
            final_params = dict(candidate_params)
            final_params["n_estimators"] = int(best_trees)
            final_model = RandomSurvivalForest(**final_params)
            self._fit_model_with_ui_pump(final_model, X_train, y_train)
            return final_model, int(best_trees), metric_history, metric_label

        if int(best_trees) == int(last_trees):
            return model, int(best_trees), metric_history, metric_label

        final_params = dict(candidate_params)
        final_params["n_estimators"] = int(best_trees)
        final_model = RandomSurvivalForest(**final_params)
        self._fit_model_with_ui_pump(final_model, X_train, y_train)
        return final_model, int(best_trees), metric_history, metric_label

    def _apply_best_params_to_controls(self, display_params):
        if not isinstance(display_params, dict):
            return

        if "n_estimators" in display_params:
            self.n_estimators_var.set(int(display_params["n_estimators"]))
        if "min_samples_leaf" in display_params:
            self.min_samples_leaf_var.set(int(display_params["min_samples_leaf"]))
        if "min_samples_split" in display_params:
            self.min_samples_split_var.set(int(display_params["min_samples_split"]))

        max_features_value = str(display_params.get("max_features", "sqrt")).strip()
        if max_features_value in {"sqrt", "log2", "all", "0.5"}:
            self.max_features_var.set(max_features_value)
            if max_features_value != "manual":
                self.max_features_manual_var.set("")
        else:
            self.max_features_var.set("manual")
            self.max_features_manual_var.set(max_features_value)

    def _build_tuning_summary(
        self,
        tuning_results,
        best_result,
        profile_name="General",
        n_rows=None,
        recommended_profile=None,
        test_size=None,
        duration_col=None,
        event_col=None,
        covariates=None,
        tuning_mode=None,
    ):
        lines = []
        cv_metric_label = self._resolve_cv_metric_choice()[1]
        cv_display_label, test_eq_label = self._resolve_cv_metric_labels()
        lines.append("=== Tuning automático RSF ===")
        lines.append(f"Perfil usado: {profile_name}")
        if tuning_mode:
            lines.append(f"Modelos evaluados: {tuning_mode}")
        if n_rows is not None:
            lines.append(f"Observaciones para tuning: {int(n_rows)}")
        if recommended_profile:
            lines.append(f"Perfil sugerido por tamaño: {recommended_profile}")
        if test_size is not None:
            if float(test_size) <= 0:
                lines.append("Proporción test usada: 0.00 (sin holdout; todo entrenamiento)")
            else:
                lines.append(f"Proporción test usada: {float(test_size):.2f}")
        lines.append(f"Combinaciones evaluadas: {len(tuning_results)}")
        lines.append(f"Criterio principal: {cv_display_label} (desempate por {test_eq_label} y OOB).")

        duration_used = duration_col or getattr(self, "latest_duration_col", None) or (self.duration_var.get().strip() if hasattr(self, "duration_var") else "")
        event_used = event_col or getattr(self, "latest_event_col", None) or (self.event_var.get().strip() if hasattr(self, "event_var") else "")
        covariates_used = list(covariates) if isinstance(covariates, (list, tuple)) else list(getattr(self, "latest_covariates", []))
        if duration_used or event_used or covariates_used:
            lines.append("")
            lines.append("Variables usadas para tuning:")
            if duration_used:
                lines.append(f"- Tiempo: {duration_used}")
            if event_used:
                lines.append(f"- Evento: {event_used}")
            if covariates_used:
                lines.append(f"- Covariables ({len(covariates_used)}): {', '.join(str(c) for c in covariates_used)}")

        if str(profile_name).strip().lower() == "manual":
            lines.append("")
            lines.append("Espacio de búsqueda manual usado:")
            lines.append(f"- Árboles: {self.manual_trees_var.get() if hasattr(self, 'manual_trees_var') else '-'}")
            lines.append(f"- max_features: {self.manual_max_features_grid_var.get() if hasattr(self, 'manual_max_features_grid_var') else '-'}")
            lines.append(f"- min_leaf: {self.manual_min_leaf_grid_var.get() if hasattr(self, 'manual_min_leaf_grid_var') else '-'}")
            lines.append(f"- Mult. split (M): {self.manual_split_mult_grid_var.get() if hasattr(self, 'manual_split_mult_grid_var') else '-'}")
            lines.append(f"- max_depth: {self.manual_max_depth_grid_var.get() if hasattr(self, 'manual_max_depth_grid_var') else '-'}")
            lines.append(f"- max_leaf_nodes: {self.manual_max_leaf_nodes_grid_var.get() if hasattr(self, 'manual_max_leaf_nodes_grid_var') else '-'}")
            lines.append(f"- max_samples: {self.manual_max_samples_grid_var.get() if hasattr(self, 'manual_max_samples_grid_var') else '-'}")
        lines.append("")
        lines.append("Top configuraciones:")

        for idx, result in enumerate(tuning_results[:5], start=1):
            display = result.get("display", {})
            metrics = result.get("metrics", {})
            scope_label = str(display.get("scope", "")).strip()
            scope_prefix = f"{scope_label} | " if scope_label else ""
            lines.append(
                f"{idx}. {scope_prefix}árboles={display.get('n_estimators', '-')}, "
                f"max_features={display.get('max_features', '-')}, "
                f"min_leaf={display.get('min_samples_leaf', '-')}, "
                f"min_split={display.get('min_samples_split', '-')} (M={display.get('split_multiplier', '-')}), "
                f"{cv_display_label}={self._format_c_index_display(metrics.get('c_index_cv_mean'), metrics.get('c_index_cv_ci'), decimals=3)}, "
                f"{test_eq_label}={self._format_c_index_display(metrics.get(self._resolve_clinical_stability_metric_keys()[1]), metrics.get('c_index_test_ci'), decimals=3)}, "
                f"OOB={self._format_metric(metrics.get('oob_score'))}"
            )

        if best_result:
            best_display = best_result.get("display", {})
            best_metrics = best_result.get("metrics", {})
            lines.append("")
            lines.append("Mejor configuración detectada:")
            if best_display.get("scope"):
                lines.append(f"- Tipo de modelo: {best_display.get('scope')}")
            lines.append(f"- n_estimators: {best_display.get('n_estimators', '-')}")
            lines.append(f"- max_features: {best_display.get('max_features', '-')}")
            lines.append(f"- min_samples_leaf: {best_display.get('min_samples_leaf', '-')}")
            lines.append(f"- min_samples_split: {best_display.get('min_samples_split', '-')} (M={best_display.get('split_multiplier', '-')})")
            lines.append(
                f"- {cv_display_label}: {self._format_c_index_display(best_metrics.get('c_index_cv_mean'), best_metrics.get('c_index_cv_ci'), decimals=4)}"
            )
            lines.append(
                f"- {test_eq_label}: {self._format_c_index_display(best_metrics.get(self._resolve_clinical_stability_metric_keys()[1]), best_metrics.get('c_index_test_ci'), decimals=4)}"
            )
            lines.append(f"- OOB: {self._format_metric(best_metrics.get('oob_score'))}")

        return "\n".join(lines)

    def _format_elapsed_time(self, total_seconds):
        try:
            seconds = max(0, int(round(float(total_seconds))))
        except (TypeError, ValueError):
            seconds = 0
        hours, remainder = divmod(seconds, 3600)
        minutes, secs = divmod(remainder, 60)
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        return f"{minutes:02d}:{secs:02d}"

    def _build_auto_tuning_progress_message(self, profile_name, n_rows, completed, total, elapsed_seconds, cancel_requested=False,
                                              trees_completed=0, trees_total=0):
        total_count = max(int(total or 0), 1)
        completed_count = min(max(int(completed or 0), 0), total_count)
        percent = (completed_count / total_count) * 100.0 if total_count else 0.0
        elapsed_value = max(float(elapsed_seconds or 0.0), 0.0)
        # Use tree-weighted progress for more accurate ETA when available
        _tw_done = int(trees_completed or 0)
        _tw_total = int(trees_total or 0)
        if _tw_done > 0 and _tw_done < _tw_total:
            remaining_seconds = (elapsed_value / _tw_done) * (_tw_total - _tw_done)
            remaining_text = self._format_elapsed_time(remaining_seconds)
        elif completed_count > 0 and completed_count < total_count:
            remaining_seconds = (elapsed_value / completed_count) * (total_count - completed_count)
            remaining_text = self._format_elapsed_time(remaining_seconds)
        elif completed_count >= total_count:
            remaining_text = "00:00"
        else:
            remaining_text = "calculando..."

        prefix = "Cancelando tuning RSF" if cancel_requested else f"Perfil {profile_name}"
        return (
            f"{prefix} | {int(n_rows)} casos | avance {completed_count}/{total_count} ({percent:.1f}%) | "
            f"transcurrido {self._format_elapsed_time(elapsed_value)} | faltan aprox. {remaining_text}"
        )

    def _toggle_window_zoom(self, window):
        """Toggle maximize/restore state in a cross-platform safe way."""
        if window is None:
            return
        try:
            current_state = str(window.state()).lower()
            if current_state == "zoomed":
                window.state("normal")
            else:
                window.state("zoomed")
        except Exception:
            try:
                current_zoomed = bool(window.attributes("-zoomed"))
                window.attributes("-zoomed", not current_zoomed)
            except Exception:
                pass
        self._update_zoom_button_text(window)

    def _update_zoom_button_text(self, window):
        """Update the zoom toggle button label to reflect current state."""
        btn = getattr(self, "_tuning_zoom_button", None)
        if btn is None:
            return
        try:
            is_zoomed = str(window.state()).lower() == "zoomed"
        except Exception:
            is_zoomed = False
        try:
            btn.configure(text="Restaurar" if is_zoomed else "Maximizar")
        except Exception:
            pass

    def _request_auto_tuning_cancel(self):
        self._tuning_cancel_requested = True
        progress_note_var = getattr(self, "_tuning_progress_note_var", None)
        if progress_note_var is not None:
            progress_note_var.set(
                "Cancelando... se conservarán las configuraciones ya evaluadas al terminar la combinación actual."
            )
        cancel_button = getattr(self, "_tuning_cancel_button", None)
        if cancel_button is not None:
            try:
                cancel_button.configure(state="disabled", text="Cancelando...")
            except Exception:
                pass
        pause_button = getattr(self, "_tuning_pause_button", None)
        if pause_button is not None:
            try:
                pause_button.configure(state="disabled")
            except Exception:
                pass
        skip_button = getattr(self, "_tuning_skip_button", None)
        if skip_button is not None:
            try:
                skip_button.configure(state="disabled")
            except Exception:
                pass

    def _toggle_auto_tuning_pause(self):
        if bool(getattr(self, "_tuning_cancel_requested", False)):
            return
        self._tuning_pause_requested = not bool(getattr(self, "_tuning_pause_requested", False))
        is_paused = bool(self._tuning_pause_requested)

        pause_button = getattr(self, "_tuning_pause_button", None)
        if pause_button is not None:
            try:
                pause_button.configure(text=("Reanudar" if is_paused else "Pausar"))
            except Exception:
                pass

        progress_note_var = getattr(self, "_tuning_progress_note_var", None)
        if progress_note_var is not None:
            if is_paused:
                progress_note_var.set(
                    "Tuning en pausa. Puedes revisar la tabla de modelos RSF y explorar los parciales; pulsa Reanudar para continuar."
                )
            else:
                progress_note_var.set("Tuning en curso. Puedes cancelar y se conservarán los resultados parciales ya calculados.")

    def _request_auto_tuning_skip_scope(self):
        if bool(getattr(self, "_tuning_cancel_requested", False)):
            return
        self._tuning_skip_scope_requested = True
        skip_button = getattr(self, "_tuning_skip_button", None)
        if skip_button is not None:
            try:
                skip_button.configure(state="disabled", text="Saltando...")
            except Exception:
                pass
        progress_note_var = getattr(self, "_tuning_progress_note_var", None)
        if progress_note_var is not None:
            progress_note_var.set(
                "Se solicitó saltar la combinación de variables actual; se avanzará al siguiente scope al terminar este fit."
            )

    def _reset_auto_tuning_skip_button(self):
        skip_button = getattr(self, "_tuning_skip_button", None)
        if skip_button is not None and not bool(getattr(self, "_tuning_cancel_requested", False)):
            try:
                skip_button.configure(state="normal", text="Saltar combinación")
            except Exception:
                pass

    def _coerce_gap_threshold(self, raw_value, default=0.05, minimum=0.0, maximum=0.3):
        value = self._coerce_float(raw_value, default, minimum=minimum, maximum=maximum)
        return float(value)

    def _passes_metric_gap_thresholds(self, metrics, oob_cv_threshold, oob_test_threshold, cv_test_threshold):
        if not isinstance(metrics, dict):
            return True, {}

        def _finite_value(key_name):
            raw = metrics.get(key_name)
            if raw is None:
                return None
            try:
                value = float(raw)
            except (TypeError, ValueError):
                return None
            return value if np.isfinite(value) else None

        oob_val = _finite_value("oob_score")
        cv_val = _finite_value("c_index_cv_mean")

        test_metric_key = "c_index_test"
        test_val = _finite_value(test_metric_key)
        if test_val is None:
            for alt_key in ("c_index_uno", "c_index_antolini"):
                alt_val = _finite_value(alt_key)
                if alt_val is not None:
                    test_metric_key = alt_key
                    test_val = alt_val
                    break

        checks = [
            ("oob_cv", oob_val, cv_val, float(oob_cv_threshold)),
            ("oob_test", oob_val, test_val, float(oob_test_threshold)),
            ("cv_test", cv_val, test_val, float(cv_test_threshold)),
        ]
        diagnostics = {}
        all_ok = True
        for diag_key, left_val, right_val, threshold in checks:
            if left_val is None or right_val is None:
                diagnostics[diag_key] = {"ok": True, "gap": None, "threshold": threshold}
                if diag_key in {"oob_test", "cv_test"}:
                    diagnostics[diag_key]["test_metric"] = test_metric_key
                continue
            gap = abs(float(left_val) - float(right_val))
            is_ok = gap <= float(threshold)
            diagnostics[diag_key] = {"ok": bool(is_ok), "gap": float(gap), "threshold": float(threshold)}
            if diag_key in {"oob_test", "cv_test"}:
                diagnostics[diag_key]["test_metric"] = test_metric_key
            all_ok = bool(all_ok and is_ok)
        return bool(all_ok), diagnostics

    def _build_autotune_live_snapshot(
        self,
        result,
        row_index,
        profile_name,
        recommended_profile,
        requested_test_size,
        duration_col,
        event_col,
        fallback_covariates,
        tuning_mode,
    ):
        snapshot_params = copy.deepcopy((result or {}).get("params", {}))
        display_max_features = str((result or {}).get("display", {}).get("max_features", "")).strip().lower()
        if display_max_features == "all":
            snapshot_params.pop("max_features", None)
        elif display_max_features:
            snapshot_params["max_features"] = (result or {}).get("display", {}).get("max_features")

        snapshot_params["test_size"] = (result or {}).get("test_size", requested_test_size)
        snapshot_params["tau_mode"] = self.tau_mode_var.get() if hasattr(self, "tau_mode_var") else "Percentil 90"
        snapshot_params["tau_manual"] = self.tau_manual_var.get() if hasattr(self, "tau_manual_var") else ""
        snapshot_params["optimization_metric"] = self.optimization_metric_var.get() if hasattr(self, "optimization_metric_var") else "Harrell C-index"
        snapshot_params["tuning_scope"] = (result or {}).get("scope", (result or {}).get("display", {}).get("scope", "-"))
        snapshot_params["tuning_mode"] = (result or {}).get("mode", tuning_mode)

        report_text = self._build_tuning_summary(
            [result],
            result,
            profile_name=profile_name,
            n_rows=len((result or {}).get("fit_dataframe", self.data if isinstance(self.data, pd.DataFrame) else pd.DataFrame())),
            recommended_profile=recommended_profile,
            test_size=(result or {}).get("test_size", requested_test_size),
            duration_col=duration_col,
            event_col=event_col,
            covariates=(result or {}).get("covariates", fallback_covariates),
            tuning_mode=tuning_mode,
        )

        return {
            "label": f"AutoTune parcial #{int(row_index)}",
            "autotune_live": True,
            "autotune_partial": True,
            "params": snapshot_params,
            "metrics": copy.deepcopy((result or {}).get("metrics", {})),
            "scope": (result or {}).get("scope", (result or {}).get("display", {}).get("scope", "-")),
            "mode": (result or {}).get("mode", tuning_mode),
            "report_text": report_text,
            "latest_fit_dataframe": (result or {}).get("fit_dataframe", self.data).copy(deep=True) if isinstance((result or {}).get("fit_dataframe", self.data), pd.DataFrame) else None,
            "latest_duration_col": duration_col,
            "latest_event_col": event_col,
            "latest_covariates": list((result or {}).get("covariates", fallback_covariates)),
            "latest_encoded_columns": list((result or {}).get("encoded_columns", [])),
            "latest_drop_first": bool(self.drop_first_var.get()) if hasattr(self, "drop_first_var") else True,
            "variable_configs": copy.deepcopy(getattr(self, "variable_configs", {})),
            "tuning_summary": self.latest_tuning_summary,
        }

    def _sync_live_autotune_snapshots(
        self,
        tuning_results,
        profile_name,
        recommended_profile,
        requested_test_size,
        duration_col,
        event_col,
        covariates,
        tuning_mode,
    ):
        if not isinstance(tuning_results, list):
            return

        active_snapshot = None
        if isinstance(self.active_saved_model_index, int) and 0 <= self.active_saved_model_index < len(self.saved_models):
            active_snapshot = self.saved_models[self.active_saved_model_index]

        preserved_snapshots = [
            snapshot for snapshot in list(self.saved_models)
            if not bool(snapshot.get("autotune_live", False))
        ]

        max_keep = int(getattr(self, "_max_cancel_autotune_snapshots", 180) or 180)
        max_keep = max(30, max_keep)
        publish_results = list(tuning_results[-max_keep:])

        live_snapshots = []
        for row_idx, result in enumerate(publish_results, start=1):
            try:
                live_snapshot = self._build_autotune_live_snapshot(
                    result=result,
                    row_index=row_idx,
                    profile_name=profile_name,
                    recommended_profile=recommended_profile,
                    requested_test_size=requested_test_size,
                    duration_col=duration_col,
                    event_col=event_col,
                    fallback_covariates=covariates,
                    tuning_mode=tuning_mode,
                )
                live_snapshots.append(live_snapshot)
            except Exception:
                continue

        self.saved_models = preserved_snapshots + live_snapshots

        _found_idx = self._find_snapshot_identity(active_snapshot, self.saved_models)
        if _found_idx is not None:
            self.active_saved_model_index = _found_idx
        elif live_snapshots:
            self.active_saved_model_index = len(self.saved_models) - 1
        elif self.saved_models:
            self.active_saved_model_index = len(self.saved_models) - 1
        else:
            self.active_saved_model_index = None

        self._refresh_saved_models_tree(rerank=False)

    def _wait_if_auto_tuning_paused(
        self,
        tuning_results=None,
        profile_name="General",
        recommended_profile=None,
        requested_test_size=0.25,
        duration_col="",
        event_col="",
        covariates=None,
        tuning_mode="Multivariado",
    ):
        if not bool(getattr(self, "_tuning_pause_requested", False)):
            return

        synced_once = False
        while bool(getattr(self, "_tuning_pause_requested", False)) and not bool(getattr(self, "_tuning_cancel_requested", False)):
            if not synced_once and isinstance(tuning_results, list) and tuning_results:
                self._sync_live_autotune_snapshots(
                    tuning_results=tuning_results,
                    profile_name=profile_name,
                    recommended_profile=recommended_profile,
                    requested_test_size=requested_test_size,
                    duration_col=duration_col,
                    event_col=event_col,
                    covariates=covariates,
                    tuning_mode=tuning_mode,
                )
                synced_once = True

            progress_note_var = getattr(self, "_tuning_progress_note_var", None)
            if progress_note_var is not None:
                progress_note_var.set(
                    "Tuning en pausa. Puedes explorar los modelos parciales en la tabla RSF."
                )

            dialog = getattr(self, "_tuning_progress_dialog", None)
            if dialog is not None:
                try:
                    dialog.update()
                except tk.TclError:
                    self._tuning_cancel_requested = True
                    break
            else:
                try:
                    self.update_idletasks()
                except Exception:
                    pass

            time.sleep(0.12)

        if not bool(getattr(self, "_tuning_cancel_requested", False)):
            progress_note_var = getattr(self, "_tuning_progress_note_var", None)
            if progress_note_var is not None:
                progress_note_var.set("Reanudando tuning RSF...")

    def _resolve_tuning_progress_metric_key(self):
        """Map the UI label to the metrics dict key and whether higher is better."""
        label = self.tuning_progress_metric_var.get() if hasattr(self, "tuning_progress_metric_var") else "C-Uno (IPCW)"
        mapping = {
            "CV": ("c_index_cv_mean", True),
            "C-Uno (IPCW)": ("c_index_uno", True),
            "C-test": ("c_index_test", True),
            "OOB": ("oob_score", True),
            "BSS": ("bss", True),
            "IBS": ("ibs", False),
        }
        return mapping.get(label, ("c_index_uno", True))

    # ------------------------------------------------------------------
    _BALANCED_METRICS = [
        ("c_index_uno",    True),
        ("c_index_cv_mean", True),
        ("c_index_test",   True),
        ("oob_score",      True),
        ("bss",            True),
    ]

    # Thresholds for the 5-phase clinical ranking
    _RANK_BSS_MIN = -0.05          # BSS must be > -0.05 (only purge truly awful calibration)
    _RANK_OVERFIT_GAP_MAX = 0.10   # |C-Test - CV mean| tolerance
    _RANK_CUNO_TIE_EPS = 0.005     # C-Uno difference considered a tie
    _RANK_CV_TIE_EPS = 0.005       # CV difference considered a tie
    _PLOT_GAP_OOB_CV_DEFAULT = 0.05
    _PLOT_GAP_OOB_TEST_DEFAULT = 0.08
    _PLOT_GAP_CV_TEST_DEFAULT = 0.03
    _PLOT_VIABILITY_GAP_DEFAULT = 0.03
    _PLOT_VIABILITY_GAP_MAX = 0.30

    # Automatic tree-growth controls (used by run_auto_tuning)
    @property
    def _AUTO_TREE_START(self):
        try: return int(self.auto_tree_start_var.get())
        except Exception: return 100

    @property
    def _AUTO_TREE_STEP(self):
        try: return int(self.auto_tree_step_var.get())
        except Exception: return 100

    @property
    def _AUTO_TREE_MIN(self):
        try: return max(int(self.auto_tree_start_var.get()), 100)
        except Exception: return 200

    @property
    def _AUTO_TREE_MAX(self):
        try: return int(self.auto_tree_max_var.get())
        except Exception: return 1200

    _AUTO_TREE_DELTA_MIN = 0.001
    _AUTO_TREE_PATIENCE = 3

    def _resolve_clinical_stability_metric_keys(self):
        """Return (cv_key, test_equivalent_key, cv_label, test_label) for clinical stability checks."""
        metric_code, metric_label = self._resolve_cv_metric_choice()
        if metric_code == "uno":
            return "c_index_cv_mean", "c_index_uno", metric_label, "C-Uno (test)"
        if metric_code == "antolini":
            return "c_index_cv_mean", "c_index_antolini", metric_label, "C-Antolini (test)"
        return "c_index_cv_mean", "c_index_test", metric_label, "C-clásico (test)"

    def _rank_tuning_results(self, tuning_results):
        """Rank auto-tuning results using 5-phase clinical logic.

        Phase 1 – Purge: discard models whose BSS <= -0.05 (calibration gate).
           Phase 2 – Stability: discard models where |Holdout_clinico - CV_mean| > 0.10.
           Phase 3 – Selection: sort survivors by the primary clinical metric (desc).
           Phase 4 – Tiebreak-1: prefer higher CV mean.
               Tiebreak-2: prefer smaller |CV - Holdout_clinico| gap.
        Phase 5 – Tiebreak-2 (Occam): fewer trees, then higher min_split.

        If all models are purged the full list is returned sorted by the
        original score (graceful fallback so we never lose all results).

        Returns the sorted list *in-place* and a dict with purge stats.
        """
        if not tuning_results:
            return tuning_results, {}

        metric_code = self._resolve_cv_metric_choice()[0]
        cv_key = "c_index_cv_mean"
        if metric_code == "uno":
            primary_key = "c_index_uno"
            holdout_key = "c_index_test"
        elif metric_code == "antolini":
            primary_key = "c_index_antolini"
            holdout_key = "c_index_antolini"
        else:
            primary_key = "c_index_test"
            holdout_key = "c_index_test"

        def _g(item, key):
            """Get a numeric metric value or None."""
            v = item.get("metrics", {}).get(key)
            if v is not None and np.isfinite(v):
                return float(v)
            return None

        # ── Phase 1: BSS gate ──────────────────────────────────────────
        purged_bss = []
        survivors = []
        for item in tuning_results:
            bss = _g(item, "bss")
            if bss is not None and bss <= self._RANK_BSS_MIN:
                purged_bss.append(item)
            else:
                survivors.append(item)

        # ── Phase 2: Overfitting gap gate ──────────────────────────────
        purged_gap = []
        stable = []
        for item in survivors:
            cv = _g(item, cv_key)
            ct = _g(item, holdout_key)
            if cv is not None and ct is not None and abs(ct - cv) > self._RANK_OVERFIT_GAP_MAX:
                purged_gap.append(item)
            else:
                stable.append(item)

        stats = {
            "total": len(tuning_results),
            "purged_bss": len(purged_bss),
            "purged_gap": len(purged_gap),
            "survivors": len(stable),
        }

        # Fallback: if every model was purged, keep them all
        if not stable:
            stable = list(tuning_results)
            purged_bss.clear()
            purged_gap.clear()
            stats["fallback"] = True

        # ── Phases 3-5: Composite sort key ─────────────────────────────
        def _sort_key(item):
            cv_sel = _g(item, cv_key)
            primary_metric = _g(item, primary_key)
            holdout_metric = _g(item, holdout_key)
            primary_for_sort = primary_metric if primary_metric is not None else (cv_sel if cv_sel is not None else float("-inf"))
            cv_for_sort = cv_sel if cv_sel is not None else float("-inf")
            holdout_for_gap = holdout_metric if holdout_metric is not None else primary_for_sort
            gap = abs(holdout_for_gap - cv_for_sort) if np.isfinite(cv_for_sort) and np.isfinite(holdout_for_gap) else float("inf")
            p    = item.get("params", {}) or item.get("display", {})
            trees = p.get("n_estimators", 9999)
            split = p.get("min_samples_split", 0)

            # Diagonal-proximity score: min(primary, cv) ensures the model
            # is high on BOTH axes (close to y=x AND high). A model that is
            # very high on one axis but low on the other is penalised heavily.
            if np.isfinite(primary_for_sort) and np.isfinite(cv_for_sort):
                diagonal_score = min(primary_for_sort, cv_for_sort)
                avg_score = (primary_for_sort + cv_for_sort) / 2.0
            else:
                diagonal_score = float("-inf")
                avg_score = float("-inf")

            return (
                diagonal_score,    # Phase 3: balanced – closest to diagonal AND high
                avg_score,         # Phase 4: tiebreak by average performance
                -gap,              # Phase 4b: smaller CV-holdout gap is better
                -trees,            # Phase 5a: fewer trees = better
                split,             # Phase 5b: higher split = better
            )

        stable.sort(key=_sort_key, reverse=True)

        # Rebuild the full list: survivors first, then purged (sorted too)
        purged_all = purged_bss + purged_gap
        purged_all.sort(key=_sort_key, reverse=True)
        tuning_results[:] = stable + purged_all
        return tuning_results, stats

    def _format_candidate_params_short(self, candidate_params):
        """Build a short description of the candidate hyperparams."""
        parts = []
        n = candidate_params.get("n_estimators")
        if n is not None:
            parts.append(f"árboles={n}")
        mf = candidate_params.get("max_features")
        if mf is not None:
            parts.append(f"feat={self._annotate_max_features_text(mf)}")
        ml = candidate_params.get("min_samples_leaf")
        if ml is not None:
            parts.append(f"leaf={ml}")
        ms = candidate_params.get("min_samples_split")
        if ms is not None:
            sm = candidate_params.get("split_multiplier")
            if sm is not None:
                parts.append(f"split={ms} (M={sm})")
            else:
                parts.append(f"split={ms}")
        md = candidate_params.get("max_depth")
        if md is not None:
            parts.append(f"depth={md}")
        return ", ".join(parts) if parts else "—"

    def _format_candidate_params_full(self, candidate_params, scope=None, covariates=None):
        """Full description: ALL hyperparams + seeds + scope + covariates (may span two lines)."""
        parts = []
        n = candidate_params.get("n_estimators")
        if n is not None:
            parts.append(f"árboles={n}")
        mf = candidate_params.get("max_features")
        if mf is not None:
            parts.append(f"feat={mf}")
        ml = candidate_params.get("min_samples_leaf")
        if ml is not None:
            parts.append(f"leaf={ml}")
        ms = candidate_params.get("min_samples_split")
        if ms is not None:
            sm = candidate_params.get("split_multiplier")
            if sm is not None:
                parts.append(f"split={ms} (M={sm})")
            else:
                parts.append(f"split={ms}")
        md = candidate_params.get("max_depth")
        if md is not None:
            parts.append(f"depth={md}")
        bootstrap = candidate_params.get("bootstrap")
        if bootstrap is not None and not bool(bootstrap):
            parts.append("bootstrap=False")
        max_samples = candidate_params.get("max_samples")
        if max_samples is not None:
            parts.append(f"max_samples={max_samples}")
        try:
            seeds = self._parse_random_seeds()
            if len(seeds) == 1:
                parts.append(f"semilla={seeds[0]}")
            else:
                parts.append(f"semillas={','.join(str(s) for s in seeds)}")
        except Exception:
            rs = candidate_params.get("random_state")
            if rs is not None:
                parts.append(f"semilla={rs}")
        params_line = ", ".join(parts) if parts else "—"
        if covariates:
            cov_list = [str(c).strip() for c in list(covariates) if str(c).strip()]
            cov_text = ", ".join(cov_list) if cov_list else "(sin covariables)"
            params_line = f"{params_line} | vars[{len(cov_list)}]: {cov_text}"
        elif scope:
            params_line = f"scope: {scope} | {params_line}"
        return params_line

    def _format_candidate_scope_short(self, candidate_display):
        scope_text = str((candidate_display or {}).get("scope", "")).strip()
        return scope_text if scope_text else "—"

    def _format_covariates_short(self, covariates, max_items=None):
        covs = [str(c).strip() for c in list(covariates or []) if str(c).strip()]
        if not covs:
            return "(sin covariables)"
        if max_items is None or max_items <= 0 or len(covs) <= max_items:
            return ", ".join(covs)
        return f"{', '.join(covs[:max_items])}, ... (+{len(covs) - max_items})"

    def _get_scope_color(self, scope_label):
        scope_key = str(scope_label).strip() or "General"
        color_map = getattr(self, "_scope_color_map", None)
        if not isinstance(color_map, dict):
            self._scope_color_map = {}
            color_map = self._scope_color_map
        if scope_key in color_map:
            return color_map[scope_key]
        palette = getattr(self, "_scope_palette", None) or ["#2563EB"]
        next_color = palette[len(color_map) % len(palette)]
        color_map[scope_key] = next_color
        return next_color

    def _set_live_scatter_info_line(self, text, max_chars=320):
        info_var = getattr(self, "_live_scatter_info_var", None)
        if info_var is None:
            return
        raw_text = str(text or "").strip()
        lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
        line = "\n".join(lines) if lines else ""
        if len(line) > max_chars:
            line = line[: max_chars - 1].rstrip() + "…"
        info_var.set(line)

    def _update_tuning_status_text_widget(self):
        """Refresh the compact scrollable status pane in the tuning progress dialog."""
        tw = getattr(self, "_tuning_status_text_widget", None)
        if tw is None:
            return
        parts = []
        for attr in (
            "_tuning_progress_var",
            "_tuning_current_model_var",
            "_tuning_best_metric_var",
            "_tuning_best_row_var",
            "_tuning_balanced_var",
        ):
            v = getattr(self, attr, None)
            if v is not None:
                val = v.get().strip()
                if val:
                    parts.append(val)
        extra = getattr(self, "_tuning_vars_summary_text", "")
        if extra:
            parts.append(extra)
        try:
            autoscroll_var = getattr(self, "_tuning_autoscroll_var", None)
            do_autoscroll = (autoscroll_var is None or bool(autoscroll_var.get()))
            # Save scroll position so the user's view is preserved
            _prev_yview = tw.yview() if not do_autoscroll else None
            tw.configure(state="normal")
            tw.delete("1.0", tk.END)
            tw.insert(tk.END, "\n".join(parts))
            if do_autoscroll:
                tw.see(tk.END)
            elif _prev_yview is not None:
                tw.yview_moveto(_prev_yview[0])
            tw.configure(state="disabled")
        except Exception:
            pass

    def _sort_live_scatter_filter_values(self, values):
        def _key_fn(raw):
            txt = str(raw)
            try:
                return (0, float(txt))
            except Exception:
                return (1, txt.lower())

        return sorted([str(v) for v in values], key=_key_fn)

    def _update_live_scatter_filter_options(self, records):
        base_values = ["Todos", "Solo aptos", "Solo no aptos"]
        profile_values = set()
        for rec in list(records or []):
            cov_sig = str((rec or {}).get("cov_sig", "")).strip()
            if cov_sig:
                profile_values.add(f"Perfil: {cov_sig}")

        values = base_values + self._sort_live_scatter_filter_values(profile_values)
        self._live_scatter_filter_values = list(values)

        filter_cb = getattr(self, "_live_scatter_filter_cb", None)
        if filter_cb is not None:
            try:
                filter_cb.configure(values=values)
            except Exception:
                pass

        filter_var = getattr(self, "_live_scatter_filter_var", None)
        if filter_var is not None:
            current = str(filter_var.get() or "Todos")
            if current not in values:
                filter_var.set("Todos")

    def _open_live_scatter_advanced_filters_dialog(self):
        parent_dialog = getattr(self, "_tuning_progress_dialog", None)
        if parent_dialog is None:
            return
        try:
            if not parent_dialog.winfo_exists():
                return
        except Exception:
            return

        fields = list(getattr(self, "_live_scatter_advanced_filter_fields", []) or [])
        if not fields:
            fields = [
                ("cov_sig", "Combinación covariables"),
                ("scope", "Scope"),
                ("mode", "Modo"),
                ("trees", "Árboles"),
                ("max_features", "max_features"),
                ("min_leaf", "min_leaf"),
                ("min_split", "min_split"),
            ]
            self._live_scatter_advanced_filter_fields = list(fields)

        current_filters = getattr(self, "_live_scatter_advanced_filters", None)
        if not isinstance(current_filters, dict):
            current_filters = {k: set() for k, _ in fields}
            self._live_scatter_advanced_filters = current_filters
        for field_key, _ in fields:
            current_filters.setdefault(field_key, set())

        records = list(getattr(self, "_live_scatter_all_records", []) or [])
        options_map = {k: set() for k, _ in fields}
        for rec in records:
            for field_key in options_map:
                options_map[field_key].add(str((rec or {}).get(field_key, "-")))
        options_map = {k: self._sort_live_scatter_filter_values(v) for k, v in options_map.items()}

        popup = tk.Toplevel(parent_dialog)
        popup.title("Filtros avanzados (tuning en vivo)")
        popup.geometry("980x620")
        popup.transient(parent_dialog)
        popup.grab_set()

        ttk.Label(
            popup,
            text="Selecciona una o varias opciones por campo. Si dejas un campo vacío, se consideran todos.",
            foreground="#334155",
            wraplength=940,
            justify="left",
        ).pack(fill=tk.X, padx=10, pady=(10, 6))

        host = ttk.Frame(popup)
        host.pack(fill=tk.BOTH, expand=True, padx=10, pady=6)

        listboxes = {}
        for i, (field_key, field_label) in enumerate(fields):
            panel = ttk.LabelFrame(host, text=field_label, padding=6)
            panel.grid(row=i // 3, column=i % 3, sticky="nsew", padx=6, pady=6)

            lb = tk.Listbox(panel, selectmode=tk.EXTENDED, exportselection=False, height=8)
            sb = ttk.Scrollbar(panel, orient=tk.VERTICAL, command=lb.yview)
            lb.configure(yscrollcommand=sb.set)
            lb.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            sb.pack(side=tk.RIGHT, fill=tk.Y)

            values = list(options_map.get(field_key, []))
            for item in values:
                lb.insert(tk.END, item)

            selected_now = current_filters.get(field_key) or set()
            for pos, item in enumerate(values):
                if item in selected_now:
                    lb.selection_set(pos)

            listboxes[field_key] = (lb, values)

        for col in range(3):
            host.columnconfigure(col, weight=1)
        for row in range(max(1, (len(fields) + 2) // 3)):
            host.rowconfigure(row, weight=1)

        btns = ttk.Frame(popup)
        btns.pack(fill=tk.X, padx=10, pady=(0, 10))

        def _apply_filters_from_dialog():
            for field_key, (lb, values) in listboxes.items():
                current_filters[field_key] = {values[i] for i in lb.curselection()}
            popup.destroy()
            self._refresh_live_scatter()

        def _clear_filters_from_dialog():
            for field_key, _ in fields:
                current_filters[field_key] = set()
            popup.destroy()
            self._refresh_live_scatter()

        ttk.Button(btns, text="Aplicar", command=_apply_filters_from_dialog).pack(side=tk.RIGHT, padx=4)
        ttk.Button(btns, text="Cancelar", command=popup.destroy).pack(side=tk.RIGHT, padx=4)
        ttk.Button(btns, text="Limpiar filtros", command=_clear_filters_from_dialog).pack(side=tk.LEFT, padx=4)

    def _open_live_scatter_metric_filters_dialog(self):
        """Popup to set min/max thresholds per metric for the live scatter."""
        parent_dialog = getattr(self, "_tuning_progress_dialog", None)
        if parent_dialog is None:
            return
        try:
            if not parent_dialog.winfo_exists():
                return
        except Exception:
            return

        _metric_filters = getattr(self, "_live_scatter_metric_filters", None)
        if not isinstance(_metric_filters, dict):
            _metric_filters = {}
            self._live_scatter_metric_filters = _metric_filters

        # Metrics available (key, label, higher-is-better, typical range hint)
        _available = [
            ("bss",              "BSS",              True,  "0.0 – 1.0"),
            ("ibs",              "IBS",              False, "0.0 – ∞  (menor = mejor)"),
            ("oob_score",        "OOB C-index",      True,  "0.0 – 1.0"),
            ("c_index_uno",      "C-Uno (IPCW)",     True,  "0.0 – 1.0"),
            ("c_index_cv_mean",  "C-index CV media", True,  "0.0 – 1.0"),
            ("c_index_test",     "C-test",           True,  "0.0 – 1.0"),
            ("c_index_train",    "C-train",          True,  "0.0 – 1.0"),
            ("c_index_antolini", "C-Antolini",       True,  "0.0 – 1.0"),
        ]

        popup = tk.Toplevel(parent_dialog)
        popup.title("Filtros de métrica — tuning en vivo")
        popup.geometry("560x480")
        popup.resizable(True, True)
        popup.transient(parent_dialog)
        popup.grab_set()

        ttk.Label(
            popup,
            text=(
                "Define umbrales mínimos y/o máximos por métrica.\n"
                "Los modelos que no cumplan algún umbral activo se excluyen de la gráfica.\n"
                "Deja en blanco para desactivar el filtro de esa métrica."
            ),
            foreground="#334155",
            justify="left",
            wraplength=520,
        ).pack(fill=tk.X, padx=12, pady=(10, 6))

        scroll_frame_outer = ttk.Frame(popup)
        scroll_frame_outer.pack(fill=tk.BOTH, expand=True, padx=12, pady=4)

        canvas_inner = tk.Canvas(scroll_frame_outer, highlightthickness=0)
        scrollbar_inner = ttk.Scrollbar(scroll_frame_outer, orient="vertical", command=canvas_inner.yview)
        canvas_inner.configure(yscrollcommand=scrollbar_inner.set)
        scrollbar_inner.pack(side=tk.RIGHT, fill=tk.Y)
        canvas_inner.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        inner_frame = ttk.Frame(canvas_inner)
        inner_window = canvas_inner.create_window((0, 0), window=inner_frame, anchor="nw")

        def _on_inner_configure(evt=None):
            canvas_inner.configure(scrollregion=canvas_inner.bbox("all"))

        def _on_canvas_configure(evt=None):
            canvas_inner.itemconfig(inner_window, width=canvas_inner.winfo_width())

        inner_frame.bind("<Configure>", _on_inner_configure)
        canvas_inner.bind("<Configure>", _on_canvas_configure)

        # Header row
        header = ttk.Frame(inner_frame)
        header.pack(fill=tk.X, pady=(0, 4))
        ttk.Label(header, text="Métrica", width=22, anchor="w", font=("Segoe UI", 9, "bold")).grid(row=0, column=0, padx=4)
        ttk.Label(header, text="Mín",     width=10, anchor="w", font=("Segoe UI", 9, "bold")).grid(row=0, column=1, padx=4)
        ttk.Label(header, text="Máx",     width=10, anchor="w", font=("Segoe UI", 9, "bold")).grid(row=0, column=2, padx=4)
        ttk.Label(header, text="Rango típico",  width=20, anchor="w", font=("Segoe UI", 8), foreground="#6b7280").grid(row=0, column=3, padx=4)

        ttk.Separator(inner_frame, orient="horizontal").pack(fill=tk.X, pady=(0, 6))

        entry_vars = {}
        for mk, mlbl, _higher, _hint in _available:
            row_f = ttk.Frame(inner_frame)
            row_f.pack(fill=tk.X, pady=2)

            current = _metric_filters.get(mk, {})
            mn_str = str(current.get("min", "")) if current.get("min") is not None else ""
            mx_str = str(current.get("max", "")) if current.get("max") is not None else ""

            var_min = StringVar(value=mn_str)
            var_max = StringVar(value=mx_str)
            entry_vars[mk] = (var_min, var_max)

            ttk.Label(row_f, text=mlbl, width=22, anchor="w").grid(row=0, column=0, padx=4)
            ent_min = ttk.Entry(row_f, textvariable=var_min, width=10)
            ent_min.grid(row=0, column=1, padx=4)
            ent_max = ttk.Entry(row_f, textvariable=var_max, width=10)
            ent_max.grid(row=0, column=2, padx=4)
            ttk.Label(row_f, text=_hint, foreground="#6b7280", font=("Segoe UI", 8)).grid(row=0, column=3, padx=4)

        btns = ttk.Frame(popup)
        btns.pack(fill=tk.X, padx=12, pady=(4, 10))

        def _apply():
            new_filters = {}
            parse_errors = []
            for mk, (var_min, var_max) in entry_vars.items():
                mn_raw = var_min.get().strip().replace(",", ".")
                mx_raw = var_max.get().strip().replace(",", ".")
                mn_val = None
                mx_val = None
                if mn_raw:
                    try:
                        mn_val = float(mn_raw)
                    except ValueError:
                        parse_errors.append(f"'{mn_raw}' no es un número válido (Mín de {mk})")
                if mx_raw:
                    try:
                        mx_val = float(mx_raw)
                    except ValueError:
                        parse_errors.append(f"'{mx_raw}' no es un número válido (Máx de {mk})")
                if mn_val is not None or mx_val is not None:
                    new_filters[mk] = {"min": mn_val, "max": mx_val}
            if parse_errors:
                messagebox.showerror("Error en filtros", "\n".join(parse_errors), parent=popup)
                return
            self._live_scatter_metric_filters = new_filters
            popup.destroy()
            self._refresh_live_scatter()

        def _clear_all():
            for var_min, var_max in entry_vars.values():
                var_min.set("")
                var_max.set("")

        ttk.Button(btns, text="Aplicar",        command=_apply).pack(side=tk.RIGHT, padx=4)
        ttk.Button(btns, text="Cancelar",       command=popup.destroy).pack(side=tk.RIGHT, padx=4)
        ttk.Button(btns, text="Limpiar todo",   command=_clear_all).pack(side=tk.LEFT, padx=4)
        ttk.Button(btns, text="Limpiar y aplicar",
                   command=lambda: [_clear_all(), _apply()]).pack(side=tk.LEFT, padx=4)

        # Active filters summary label
        _active = {mk: b for mk, b in _metric_filters.items()
                   if b.get("min") is not None or b.get("max") is not None}
        if _active:
            _summary_parts = []
            for mk, b in _active.items():
                _lbl = next((l for k, l, *_ in _available if k == mk), mk)
                _parts = []
                if b.get("min") is not None:
                    _parts.append(f"≥{b['min']}")
                if b.get("max") is not None:
                    _parts.append(f"≤{b['max']}")
                _summary_parts.append(f"{_lbl}: {', '.join(_parts)}")
            ttk.Label(
                popup,
                text="Filtros activos: " + "  |  ".join(_summary_parts),
                foreground="#0055AA",
                font=("Segoe UI", 8),
                wraplength=520,
            ).pack(padx=12, pady=(0, 6))

    def _append_auto_tuning_history_row(
        self,
        row_index,
        status_label,
        metric_label,
        metric_value,
        best_value,
        candidate_display,
        candidate_params,
        duration_col,
        event_col,
        covariates,
        metrics=None,
        extra_note="",
    ):
        text_widget = getattr(self, "_tuning_history_text", None)
        if text_widget is None:
            return

        scope_short = self._format_candidate_scope_short(candidate_display)
        cov_text = self._format_covariates_short(covariates)
        metric_text = self._format_metric(metric_value) if metric_value is not None else "—"
        best_text = self._format_metric(best_value) if best_value is not None else "—"
        params_text = self._format_candidate_params_short({
            **candidate_params,
            "split_multiplier": (candidate_display or {}).get("split_multiplier"),
        })
        detail_line = (
            f"{int(row_index):04d} | {status_label:<8} | {scope_short:<22} | "
            f"{metric_label}={metric_text:<8} | mejor={best_text:<8} | {params_text}"
        )
        if extra_note:
            detail_line = f"{detail_line} | {extra_note}"

        vars_line = (
            f"       vars[{len(list(covariates or []))}] tiempo={duration_col or '-'} "
            f"evento={event_col or '-'} | {cov_text}"
        )
        if isinstance(metrics, dict):
            ibs_display = self._format_c_index_display(metrics.get('ibs'), metrics.get('ibs_ci'), decimals=4) if metrics.get('ibs_ci') else self._format_metric(metrics.get('ibs'))
            bss_display = self._format_c_index_display(metrics.get('bss'), metrics.get('bss_ci'), decimals=4) if metrics.get('bss_ci') else self._format_metric(metrics.get('bss'))
            vars_line = (
                f"{vars_line} | OOB={self._format_metric(metrics.get('oob_score'))}"
                f" | IBS={ibs_display}"
                f" | BSS={bss_display}"
            )

        try:
            text_widget.configure(state="normal")
            scope_color = self._get_scope_color(scope_short)
            scope_tag = f"scope_{re.sub(r'[^0-9A-Za-z_]+', '_', scope_short) or 'General'}"
            text_widget.tag_configure(scope_tag, foreground=scope_color)
            text_widget.insert(tk.END, detail_line + "\n", (scope_tag,))
            text_widget.insert(tk.END, vars_line + "\n", (scope_tag,))
            autoscroll_var = getattr(self, "_tuning_autoscroll_var", None)
            if autoscroll_var is None or bool(autoscroll_var.get()):
                text_widget.see(tk.END)
            text_widget.configure(state="disabled")
        except Exception:
            pass

        if status_label == "GANADOR":
            best_row_var = getattr(self, "_tuning_best_row_var", None)
            if best_row_var is not None:
                cv_key, test_key, cv_lbl, test_lbl = self._resolve_clinical_stability_metric_keys()
                cv_txt = self._format_metric((metrics or {}).get(cv_key)) if isinstance(metrics, dict) else "-"
                test_txt = self._format_metric((metrics or {}).get(test_key)) if isinstance(metrics, dict) else "-"
                oob_txt = self._format_metric((metrics or {}).get("oob_score")) if isinstance(metrics, dict) else "-"
                best_row_var.set(
                    f"MEJOR ACTUAL -> {metric_label}={metric_text} | {cv_lbl}={cv_txt} | {test_lbl}={test_txt} | OOB={oob_txt} | {params_text} | vars[{len(list(covariates or []))}]: {cov_text}"
                )

    def _open_auto_tuning_progress_dialog(self, profile_name, n_rows, total_candidates, duration_col=None, event_col=None, covariates=None):
        self._close_auto_tuning_progress_dialog()
        self._tuning_cancel_requested = False
        self._tuning_pause_requested = True   # abre pausado para que el usuario configure el popup
        self._tuning_skip_scope_requested = False

        parent_window = self.winfo_toplevel() if hasattr(self, "winfo_toplevel") else None
        dialog = tk.Toplevel(parent_window if parent_window is not None else self)
        dialog.title("Progreso del tuning automático RSF")
        dialog.transient(parent_window)
        dialog.resizable(True, True)
        try:
            dialog.minsize(1200, 520)
        except Exception:
            pass
        dialog.protocol("WM_DELETE_WINDOW", self._request_auto_tuning_cancel)

        container = ttk.Frame(dialog, padding=12)
        container.pack(fill=tk.BOTH, expand=True)

        header_row = ttk.Frame(container)
        header_row.pack(fill=tk.X, pady=(0, 6))

        ttk.Label(
            header_row,
            text="Tuning automático RSF en progreso",
            font=("Segoe UI", 10, "bold"),
        ).pack(side=tk.LEFT, anchor="w")

        header_actions = ttk.Frame(header_row)
        header_actions.pack(side=tk.RIGHT)

        self._tuning_zoom_button = ttk.Button(
            header_actions,
            text="Maximizar",
            command=lambda w=dialog: self._toggle_window_zoom(w),
        )
        self._tuning_zoom_button.pack(side=tk.LEFT, padx=(0, 8))
        self._tuning_pause_button = ttk.Button(header_actions, text="▶ Iniciar", command=self._toggle_auto_tuning_pause)
        self._tuning_pause_button.pack(side=tk.LEFT, padx=(0, 8))
        self._tuning_skip_button = ttk.Button(header_actions, text="Saltar combinación", command=self._request_auto_tuning_skip_scope)
        self._tuning_skip_button.pack(side=tk.LEFT, padx=(0, 8))
        self._tuning_cancel_button = ttk.Button(header_actions, text="Cancelar", command=self._request_auto_tuning_cancel)
        self._tuning_cancel_button.pack(side=tk.LEFT)

        self._tuning_progress_var = StringVar(
            value=self._build_auto_tuning_progress_message(profile_name, n_rows, 0, total_candidates, 0.0)
        )

        # Current model being evaluated
        self._tuning_current_model_var = StringVar(value="Evaluando: preparando primer candidato...")

        # Best metric so far
        metric_label = self.tuning_progress_metric_var.get() if hasattr(self, "tuning_progress_metric_var") else "C-Uno (IPCW)"
        if metric_label == "CV":
            metric_label = self._resolve_cv_metric_labels()[0]
        self._tuning_best_metric_var = StringVar(value=f"Mejor {metric_label}: \u2014 (0 modelos evaluados)")

        self._tuning_balanced_var = StringVar(value="M\u00e1s equilibrado: \u2014 (esperando \u22653 modelos)")

        self._tuning_best_row_var = StringVar(value="MEJOR ACTUAL -> esperando evaluaci\u00f3n...")

        self._tuning_vars_summary_text = (
            f"Variables activas: tiempo={duration_col or '-'} | evento={event_col or '-'} | "
            f"covariables={len(list(covariates or []))}"
        )
        self._tuning_history_meta_var = StringVar(
            value="Tabla en vivo: cada fila muestra ganador/vencido, m\u00e9trica comparada, mejor acumulado y par\u00e1metros."
        )

        # \u2500\u2500 Compact scrollable status pane (replaces individual stacked labels) \u2500\u2500\u2500\u2500
        _status_outer = ttk.Frame(container)
        _status_outer.pack(fill=tk.X, pady=(0, 4))
        _status_yscroll = ttk.Scrollbar(_status_outer, orient="vertical")
        _status_text_widget = tk.Text(
            _status_outer,
            height=8,
            wrap="word",
            yscrollcommand=_status_yscroll.set,
            font=("Segoe UI", 7),
            background="#F3F6FA",
            state="disabled",
            borderwidth=1,
            relief="solid",
            cursor="arrow",
        )
        _status_yscroll.configure(command=_status_text_widget.yview)
        _status_yscroll.pack(side=tk.RIGHT, fill=tk.Y)
        _status_text_widget.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self._tuning_status_text_widget = _status_text_widget
        self._update_tuning_status_text_widget()

        history_frame = ttk.Frame(container)
        history_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 8))

        # ── PanedWindow: history (left) + live 3D scatter (right) ──
        paned = ttk.PanedWindow(history_frame, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)

        # Left pane – history text
        history_left = ttk.Frame(paned)
        paned.add(history_left, weight=3)

        history_scroll = ttk.Scrollbar(history_left, orient="vertical")
        history_scroll_x = ttk.Scrollbar(history_left, orient="horizontal")
        history_text = tk.Text(
            history_left,
            height=12,
            wrap="none",
            yscrollcommand=history_scroll.set,
            xscrollcommand=history_scroll_x.set,
            font=("Consolas", 9),
            background="#FAFAFA",
        )
        history_scroll.configure(command=history_text.yview)
        history_scroll_x.configure(command=history_text.xview)
        history_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        history_scroll_x.pack(side=tk.BOTTOM, fill=tk.X)
        history_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        history_header = (
            "idx  | estado   | tipo modelo            | metrica   | mejor    | parametros\n"
            "--------------------------------------------------------------------------------"
        )
        history_text.insert(tk.END, history_header + "\n")
        history_text.configure(state="disabled")
        self._tuning_history_text = history_text

        # Right pane – Notebook with two tabs (Dispersión / Screening)
        charts_nb = ttk.Notebook(paned)
        paned.add(charts_nb, weight=2)
        scatter_right = ttk.Frame(charts_nb)
        charts_nb.add(scatter_right, text="Dispersión")

        scatter_ctrl = ttk.Frame(scatter_right)
        scatter_ctrl.pack(fill=tk.X, padx=2, pady=(0, 2))

        scatter_ctrl_row1 = ttk.Frame(scatter_ctrl)
        scatter_ctrl_row1.pack(fill=tk.X, pady=(0, 1))
        scatter_ctrl_row2 = ttk.Frame(scatter_ctrl)
        scatter_ctrl_row2.pack(fill=tk.X, pady=(0, 1))
        scatter_ctrl_row3 = ttk.Frame(scatter_ctrl)
        scatter_ctrl_row3.pack(fill=tk.X)

        _live_metric_keys = [k for k, _ in self._EXPLORER_METRICS]
        _live_defaults = ["c_index_cv_mean", "c_index_oob_mean", "bss"]
        self._live_scatter_vars = []
        for i, axis_name in enumerate(("X:", "Y:", "Color:")):
            ttk.Label(scatter_ctrl_row1, text=axis_name, font=("Segoe UI", 8)).pack(side=tk.LEFT, padx=(4 if i else 0, 1))
            var = StringVar(value=_live_defaults[i])
            cb = ttk.Combobox(scatter_ctrl_row1, textvariable=var, values=_live_metric_keys, state="readonly", width=12)
            cb.pack(side=tk.LEFT, padx=(0, 4))
            self._live_scatter_vars.append(var)

        self._live_scatter_filter_var = StringVar(value="Todos")
        ttk.Label(scatter_ctrl_row2, text="Mostrar:", font=("Segoe UI", 8)).pack(side=tk.LEFT, padx=(4, 1))
        filter_cb = ttk.Combobox(
            scatter_ctrl_row2,
            textvariable=self._live_scatter_filter_var,
            values=["Todos", "Solo aptos", "Solo no aptos"],
            state="readonly",
            width=18,
        )
        filter_cb.pack(side=tk.LEFT, padx=(0, 4))
        self._live_scatter_filter_cb = filter_cb

        self._live_scatter_point_size_var = StringVar(value="55")
        ttk.Label(scatter_ctrl_row2, text="Tamaño:", font=("Segoe UI", 8)).pack(side=tk.LEFT, padx=(4, 1))
        size_cb = ttk.Combobox(
            scatter_ctrl_row2,
            textvariable=self._live_scatter_point_size_var,
            values=["25", "35", "45", "55", "70", "90", "120"],
            state="readonly",
            width=5,
        )
        size_cb.pack(side=tk.LEFT, padx=(0, 4))

        self._live_scatter_marker_mode_var = StringVar(value="Combinación")
        ttk.Label(scatter_ctrl_row2, text="Símbolo:", font=("Segoe UI", 8)).pack(side=tk.LEFT, padx=(4, 1))
        _cb_ls_marker_mode = ttk.Combobox(
            scatter_ctrl_row2,
            textvariable=self._live_scatter_marker_mode_var,
            values=["Combinación", "Scope", "Modo", "max_features", "min_leaf", "min_split",
                    "Completo/Semiinc./Incompleto"],
            state="readonly",
            width=18,
        )
        _cb_ls_marker_mode.pack(side=tk.LEFT, padx=(0, 4))
        _cb_ls_marker_mode.bind("<<ComboboxSelected>>", lambda _: self._refresh_live_scatter())

        self._live_scatter_heatmap_var = BooleanVar(value=False)
        ttk.Checkbutton(
            scatter_ctrl_row3,
            text="Mapa calor",
            variable=self._live_scatter_heatmap_var,
            command=self._refresh_live_scatter,
        ).pack(side=tk.LEFT, padx=(4, 2))

        self._live_scatter_heat_bins_var = StringVar(value="35")
        ttk.Label(scatter_ctrl_row3, text="Bins:", font=("Segoe UI", 8)).pack(side=tk.LEFT, padx=(2, 1))
        ttk.Combobox(
            scatter_ctrl_row3,
            textvariable=self._live_scatter_heat_bins_var,
            values=["20", "25", "30", "35", "45", "60"],
            state="readonly",
            width=4,
        ).pack(side=tk.LEFT, padx=(0, 4))

        self._live_scatter_show_apt_var = BooleanVar(value=False)
        ttk.Checkbutton(
            scatter_ctrl_row3,
            text="Mostrar aptos",
            variable=self._live_scatter_show_apt_var,
            command=self._refresh_live_scatter,
        ).pack(side=tk.LEFT, padx=(4, 2))

        self._live_scatter_show_completed_only_var = StringVar(value="Todos")
        ttk.Combobox(
            scatter_ctrl_row3,
            textvariable=self._live_scatter_show_completed_only_var,
            values=["Completados", "Todos"],
            state="readonly",
            width=12,
        ).pack(side=tk.LEFT, padx=(2, 4))
        self._live_scatter_show_completed_only_var.trace_add("write", lambda *_: self._refresh_live_scatter())

        ttk.Button(
            scatter_ctrl_row3,
            text="Filtros avanzados...",
            command=self._open_live_scatter_advanced_filters_dialog,
        ).pack(side=tk.LEFT, padx=4)

        ttk.Button(
            scatter_ctrl_row3,
            text="Filtros métrica...",
            command=self._open_live_scatter_metric_filters_dialog,
        ).pack(side=tk.LEFT, padx=4)

        ttk.Button(scatter_ctrl_row3, text="Actualizar", command=self._refresh_live_scatter).pack(side=tk.LEFT, padx=4)
        ttk.Button(scatter_ctrl_row3, text="Separar gráfica", command=self._open_live_scatter_popout).pack(side=tk.LEFT, padx=4)

        self._live_scatter_advanced_filter_fields = [
            ("cov_sig", "Combinación covariables"),
            ("scope", "Scope"),
            ("mode", "Modo"),
            ("trees", "Árboles"),
            ("max_features", "max_features"),
            ("min_leaf", "min_leaf"),
            ("min_split", "min_split"),
        ]
        self._live_scatter_advanced_filters = {
            k: set() for k, _ in self._live_scatter_advanced_filter_fields
        }
        # Dict of {metric_key: {"min": float|None, "max": float|None}}
        self._live_scatter_metric_filters: dict = {}
        self._live_scatter_filter_values = ["Todos", "Solo aptos", "Solo no aptos"]
        self._live_scatter_all_records = []
        self._live_scatter_pulse_seen_uids = set()

        # Second control row: three sensitivity thresholds + display toggles
        scatter_ctrl2 = ttk.Frame(scatter_right)
        scatter_ctrl2.pack(fill=tk.X, padx=2, pady=(0, 1))

        self._live_scatter_apply_gap_filters_var = BooleanVar(value=True)
        self._live_scatter_use_global_gap_var = BooleanVar(value=True)
        self._live_scatter_gap_global_var = StringVar(value=f"{self._PLOT_VIABILITY_GAP_DEFAULT:.2f}")
        self._live_scatter_gap_oob_cv_var = StringVar(value=f"{self._PLOT_GAP_OOB_CV_DEFAULT:.2f}")
        self._live_scatter_gap_oob_test_var = StringVar(value=f"{self._PLOT_GAP_OOB_TEST_DEFAULT:.2f}")
        self._live_scatter_gap_cv_test_var = StringVar(value=f"{self._PLOT_GAP_CV_TEST_DEFAULT:.2f}")
        self._live_scatter_show_ids_var = BooleanVar(value=False)
        self._live_scatter_show_ci_var = BooleanVar(value=True)

        threshold_values = ["0.00", "0.01", "0.02", "0.03", "0.05", "0.07", "0.10", "0.15", "0.20", "0.25", "0.30"]

        def _bind_threshold_refresh(cb):
            cb.bind("<<ComboboxSelected>>", lambda _event: self._refresh_live_scatter())
            cb.bind("<Return>", lambda _event: self._refresh_live_scatter())
            cb.bind("<FocusOut>", lambda _event: self._refresh_live_scatter())

        ttk.Checkbutton(
            scatter_ctrl2,
            text="Aplicar umbrales",
            variable=self._live_scatter_apply_gap_filters_var,
            command=self._refresh_live_scatter,
        ).pack(side=tk.LEFT, padx=(2, 6))

        _threshold_specific_boxes = []

        def _toggle_live_gap_mode():
            use_global_var = getattr(self, "_live_scatter_use_global_gap_var", None)
            use_global = bool(use_global_var.get()) if use_global_var is not None else False
            state = "disabled" if use_global else "normal"
            for cb in _threshold_specific_boxes:
                try:
                    cb.configure(state=state)
                except Exception:
                    pass
            self._refresh_live_scatter()

        ttk.Checkbutton(
            scatter_ctrl2,
            text="Δ único CV/Test/OOB",
            variable=self._live_scatter_use_global_gap_var,
            command=_toggle_live_gap_mode,
        ).pack(side=tk.LEFT, padx=(0, 4))

        ttk.Label(scatter_ctrl2, text="Δ<=", font=("Segoe UI", 8)).pack(side=tk.LEFT, padx=(0, 1))
        _cb_gap_global = ttk.Combobox(
            scatter_ctrl2,
            textvariable=self._live_scatter_gap_global_var,
            values=threshold_values,
            state="normal",
            width=5,
        )
        _cb_gap_global.pack(side=tk.LEFT, padx=(0, 6))
        _bind_threshold_refresh(_cb_gap_global)

        ttk.Label(scatter_ctrl2, text="|OOB-CV|<=", font=("Segoe UI", 8)).pack(side=tk.LEFT, padx=(2, 1))
        _cb_gap_oob_cv = ttk.Combobox(
            scatter_ctrl2,
            textvariable=self._live_scatter_gap_oob_cv_var,
            values=threshold_values,
            state="normal",
            width=5,
        )
        _cb_gap_oob_cv.pack(side=tk.LEFT, padx=(0, 5))
        _bind_threshold_refresh(_cb_gap_oob_cv)
        _threshold_specific_boxes.append(_cb_gap_oob_cv)

        ttk.Label(scatter_ctrl2, text="|OOB-C-test|<=", font=("Segoe UI", 8)).pack(side=tk.LEFT, padx=(2, 1))
        _cb_gap_oob_test = ttk.Combobox(
            scatter_ctrl2,
            textvariable=self._live_scatter_gap_oob_test_var,
            values=threshold_values,
            state="normal",
            width=5,
        )
        _cb_gap_oob_test.pack(side=tk.LEFT, padx=(0, 5))
        _bind_threshold_refresh(_cb_gap_oob_test)
        _threshold_specific_boxes.append(_cb_gap_oob_test)

        ttk.Label(scatter_ctrl2, text="|CV-C-test|<=", font=("Segoe UI", 8)).pack(side=tk.LEFT, padx=(2, 1))
        _cb_gap_cv_test = ttk.Combobox(
            scatter_ctrl2,
            textvariable=self._live_scatter_gap_cv_test_var,
            values=threshold_values,
            state="normal",
            width=5,
        )
        _cb_gap_cv_test.pack(side=tk.LEFT, padx=(0, 6))
        _bind_threshold_refresh(_cb_gap_cv_test)
        _threshold_specific_boxes.append(_cb_gap_cv_test)

        _toggle_live_gap_mode()

        ttk.Checkbutton(
            scatter_ctrl2,
            text="Etiquetas ID",
            variable=self._live_scatter_show_ids_var,
            command=self._refresh_live_scatter,
        ).pack(side=tk.LEFT, padx=(2, 4))

        ttk.Checkbutton(
            scatter_ctrl2,
            text="IC en cruz",
            variable=self._live_scatter_show_ci_var,
            command=self._refresh_live_scatter,
        ).pack(side=tk.LEFT, padx=(2, 4))

        self._live_scatter_info_var = StringVar(value="Esperando modelos...")
        ttk.Label(scatter_right, textvariable=self._live_scatter_info_var,
                  foreground="navy", font=("Segoe UI", 8),
                  justify="left").pack(fill=tk.X, padx=4, pady=(0, 2))

        live_fig = plt.figure(figsize=(4.5, 3.2))
        live_canvas = FigureCanvasTkAgg(live_fig, master=scatter_right)
        live_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=2, pady=(0, 2))

        self._live_scatter_fig = live_fig
        self._live_scatter_canvas = live_canvas
        self._live_scatter_data = {}

        def _on_live_hover(event):
            if event.inaxes is None:
                return
            sd = self._live_scatter_data
            ax = sd.get("ax")
            if ax is None or not sd.get("all_x"):
                return
            _hover_limit = int(getattr(self, "_live_scatter_hover_limit", 600) or 600)
            if len(sd.get("all_x", [])) > _hover_limit:
                return
            best_dist, best_i = float("inf"), None
            for i in range(len(sd["all_x"])):
                try:
                    coords = ax.transData.transform((sd["all_x"][i], sd["all_y"][i]))
                    dist = ((event.x - coords[0]) ** 2 + (event.y - coords[1]) ** 2) ** 0.5
                    if dist < best_dist:
                        best_dist = dist
                        best_i = i
                except Exception:
                    continue
            if best_i is not None and best_dist < 30:
                res = sd["results"][best_i]
                model_id = int(sd.get("all_ids", [best_i + 1])[best_i]) if sd.get("all_ids") else (best_i + 1)
                p = res.get("params", {})
                m = res.get("metrics", {})
                kx, ky, kz = sd.get("kx", ""), sd.get("ky", ""), sd.get("kz", "")
                rec = None
                recs = sd.get("records", [])
                if best_i < len(recs):
                    rec = recs[best_i]
                parts = [
                    f"Modelo #{model_id} {res.get('scope','')}",
                    self._format_candidate_params_short(p),
                ]
                if isinstance(rec, dict):
                    cov_sig = str(rec.get("cov_sig", "")).strip()
                    if cov_sig:
                        parts.append(f"Perfil: {cov_sig}")
                    mode_txt = str(rec.get("mode", "")).strip()
                    if mode_txt:
                        parts.append(f"Modo: {mode_txt}")
                for mk in (kx, ky, kz):
                    for k2, lbl2 in self._EXPLORER_METRICS:
                        if k2 == mk:
                            val = m.get(mk)
                            parts.append(f"{lbl2}={self._format_metric(val)}")
                            break
                tags = []
                if best_i == sd.get("best_metric_idx"):
                    tags.append("MEJOR METRICA")
                if best_i == sd.get("clinical_best_idx"):
                    tags.append("MEJOR CLINICO")
                if best_i == sd.get("last_visible_idx"):
                    tags.append("ULTIMO (NEGRO)")
                if tags:
                    parts.append("[" + ", ".join(tags) + "]")
                info_var = getattr(self, "_live_scatter_info_var", None)
                if info_var is not None:
                    cov_line = self._format_covariates_short(res.get("covariates", []), max_items=99)
                    self._set_live_scatter_info_line("  |  ".join(parts) + f"\nCovariables: {cov_line}")

        def _on_live_click(event):
            if event.inaxes is None:
                return
            sd = self._live_scatter_data
            ax = sd.get("ax")
            if ax is None or not sd.get("all_x"):
                return
            best_dist, best_i = float("inf"), None
            for i in range(len(sd["all_x"])):
                try:
                    coords = ax.transData.transform((sd["all_x"][i], sd["all_y"][i]))
                    dist = ((event.x - coords[0]) ** 2 + (event.y - coords[1]) ** 2) ** 0.5
                    if dist < best_dist:
                        best_dist = dist
                        best_i = i
                except Exception:
                    continue
            if best_i is None or best_dist >= 30:
                return

            res = sd["results"][best_i]
            recs = sd.get("records", [])
            rec = recs[best_i] if best_i < len(recs) else None
            # Store persistent selection UID
            if rec is not None:
                self._live_scatter_selected_uid = rec.get("uid")
            model_id = int(sd.get("all_ids", [best_i + 1])[best_i]) if sd.get("all_ids") else (best_i + 1)
            p = res.get("params", {})
            m = res.get("metrics", {})
            cov_line = self._format_covariates_short(res.get("covariates", []), max_items=99)
            n_seeds = int(m.get("n_seeds", 1) or 1)
            seeds_tag = f" (prom. {n_seeds} semillas)" if n_seeds > 1 else ""
            detail = (
                f"Modelo #{model_id} | {self._format_candidate_params_short(p)}"
                f" | C-Uno={self._format_metric(m.get('c_index_uno'))}{seeds_tag}"
                f" | CV={self._format_metric(m.get('c_index_cv_mean'))}{seeds_tag}"
                f" | BSS={self._format_metric(m.get('bss'))}{seeds_tag}"
                f" | OOB={self._format_metric(m.get('oob_score'))}{seeds_tag}"
            )
            self._live_scatter_selected_info = detail + f"\nCovariables: {cov_line}"
            self._set_live_scatter_info_line(self._live_scatter_selected_info)

        live_canvas.mpl_connect("motion_notify_event", _on_live_hover)
        live_canvas.mpl_connect("button_press_event", _on_live_click)

        # ── Screening live pane (tab 2) ────────────────────────────────────────
        vimp_right = ttk.Frame(charts_nb)
        charts_nb.add(vimp_right, text="Screening")

        vimp_ctrl = ttk.Frame(vimp_right)
        vimp_ctrl.pack(fill=tk.X, padx=2, pady=(2, 2))

        ttk.Label(vimp_ctrl, text="Screening por semilla", font=("Segoe UI", 8, "bold")).pack(side=tk.LEFT, padx=(2, 8))

        # Ejes X e Y seleccionables (igual que la gráfica de dispersión principal)
        _screen_opts = ["OOB", "C-CV", "C-Test", "BSS"]
        ttk.Label(vimp_ctrl, text="X:", font=("Segoe UI", 8)).pack(side=tk.LEFT, padx=(0, 1))
        self._autotune_screen_x_var = StringVar(value="OOB")
        _cb_sx = ttk.Combobox(vimp_ctrl, textvariable=self._autotune_screen_x_var,
                               values=_screen_opts, state="readonly", width=7)
        _cb_sx.pack(side=tk.LEFT, padx=(0, 4))
        _cb_sx.bind("<<ComboboxSelected>>", lambda _e: self._update_autotune_vimp_chart())

        ttk.Label(vimp_ctrl, text="Y:", font=("Segoe UI", 8)).pack(side=tk.LEFT, padx=(0, 1))
        self._autotune_screen_y_var = StringVar(value="C-Test")
        _cb_sy = ttk.Combobox(vimp_ctrl, textvariable=self._autotune_screen_y_var,
                               values=_screen_opts, state="readonly", width=7)
        _cb_sy.pack(side=tk.LEFT, padx=(0, 4))
        _cb_sy.bind("<<ComboboxSelected>>", lambda _e: self._update_autotune_vimp_chart())

        ttk.Label(vimp_ctrl, text="IC:", font=("Segoe UI", 8)).pack(side=tk.LEFT, padx=(6, 1))
        self._autotune_screen_ci_var = StringVar(value="95%")
        _cb_ci = ttk.Combobox(vimp_ctrl, textvariable=self._autotune_screen_ci_var,
                               values=["80%", "90%", "95%", "99%"], state="readonly", width=5)
        _cb_ci.pack(side=tk.LEFT, padx=(0, 4))
        _cb_ci.bind("<<ComboboxSelected>>", lambda _e: self._update_autotune_vimp_chart())

        # Point size
        ttk.Label(vimp_ctrl, text="Pts:", font=("Segoe UI", 8)).pack(side=tk.LEFT, padx=(6, 1))
        self._autotune_screen_pt_size_var = StringVar(value="62")
        _sb_pts = ttk.Spinbox(vimp_ctrl, textvariable=self._autotune_screen_pt_size_var,
                               from_=10, to=300, increment=10, width=4)
        _sb_pts.pack(side=tk.LEFT, padx=(0, 4))
        _sb_pts.bind("<FocusOut>", lambda _e: self._update_autotune_vimp_chart())
        _sb_pts.bind("<Return>", lambda _e: self._update_autotune_vimp_chart())

        # Show/hide seed label (S1, S2...) and value annotation
        self._autotune_screen_show_seed_label_var = BooleanVar(value=True)
        ttk.Checkbutton(vimp_ctrl, text="S#", variable=self._autotune_screen_show_seed_label_var,
                        command=self._update_autotune_vimp_chart).pack(side=tk.LEFT, padx=(2, 0))
        self._autotune_screen_show_vals_var = BooleanVar(value=True)
        ttk.Checkbutton(vimp_ctrl, text="vals", variable=self._autotune_screen_show_vals_var,
                        command=self._update_autotune_vimp_chart).pack(side=tk.LEFT, padx=(2, 0))
        self._autotune_screen_show_line_var = BooleanVar(value=True)
        ttk.Checkbutton(vimp_ctrl, text="unión", variable=self._autotune_screen_show_line_var,
                        command=self._update_autotune_vimp_chart).pack(side=tk.LEFT, padx=(2, 0))
        self._autotune_screen_show_traj_var = BooleanVar(value=True)
        ttk.Checkbutton(vimp_ctrl, text="tray", variable=self._autotune_screen_show_traj_var,
                        command=self._update_autotune_vimp_chart).pack(side=tk.LEFT, padx=(2, 0))

        # Legacy vars kept as None so cleanup code doesn't crash
        self._autotune_screen_metric_vars = {}
        self._autotune_vimp_ci_var = None
        self._autotune_vimp_topn_var = None
        self._autotune_vimp_info_var = StringVar(value="Esperando modelos...")
        ttk.Label(vimp_right, textvariable=self._autotune_vimp_info_var,
                  foreground="navy", font=("Segoe UI", 8),
                  justify="left").pack(fill=tk.X, padx=4, pady=(0, 2))

        vimp_fig = plt.figure(figsize=(3.5, 4.0))
        vimp_canvas = FigureCanvasTkAgg(vimp_fig, master=vimp_right)
        vimp_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=2, pady=(0, 2))

        self._autotune_vimp_fig = vimp_fig
        self._autotune_vimp_canvas = vimp_canvas
        self._autotune_vimp_records = []
        self._autotune_metrics_records = []
        # Per-seed tracking (se reinicia con cada candidato)
        self._autotune_current_seed_metrics = []   # [{seed, oob, cv, ctest, bss}, ...]
        self._autotune_current_vimp = {}            # {feature: normalized_importance}
        self._autotune_current_candidate_n = 0
        self._autotune_current_trees_done = 0      # árboles terminados en la semilla actual
        self._autotune_current_trees_total = 0     # total de árboles del candidato actual
        self._autotune_current_candidate_params = {}  # parámetros del candidato en curso

        # Draw empty placeholder
        _vimp_ax0 = vimp_fig.add_subplot(111)
        _vimp_ax0.text(0.5, 0.5, "Esperando modelos...", ha="center", va="center",
                       fontsize=9, color="#9ca3af")
        _vimp_ax0.set_axis_off()
        vimp_canvas.draw_idle()
        # ── end VIMP pane ──────────────────────────────────────────────────────

        self._tuning_autoscroll_var = BooleanVar(value=False)
        ttk.Checkbutton(
            container,
            text="Auto-scroll en tabla",
            variable=self._tuning_autoscroll_var,
        ).pack(anchor="w", pady=(0, 8))

        progress_bar = ttk.Progressbar(
            container,
            orient="horizontal",
            mode="determinate",
            maximum=max(int(total_candidates or 1), 1),
            value=0,
            length=480,
        )
        progress_bar.pack(fill=tk.X, pady=(0, 8))

        self._tuning_progress_note_var = StringVar(
            value="Configura el tamaño del popup y las opciones de la gráfica, luego pulsa ▶ Iniciar para comenzar el tuning."
        )
        ttk.Label(
            container,
            textvariable=self._tuning_progress_note_var,
            justify="left",
            foreground="#555555",
            wraplength=500,
        ).pack(anchor="w", pady=(0, 10))

        # Keep zoom label in sync if user maximizes/restores from title bar.
        dialog.bind("<Configure>", lambda _e, w=dialog: self._update_zoom_button_text(w))
        self._update_zoom_button_text(dialog)

        dialog.update_idletasks()
        dialog.lift()

        self._tuning_progress_dialog = dialog
        self._tuning_progress_bar = progress_bar
        self._live_scatter_last_refresh_completed = -1

    def _update_auto_tuning_progress_dialog(self, profile_name, n_rows, completed, total, elapsed_seconds,
                                              cancel_requested=False, current_params=None, best_metric_value=None,
                                              best_metric_params=None, n_evaluated=0,
                                              balanced_text=None, trees_completed=0, trees_total=0,
                                              current_tree_text=None,
                                              current_covariates=None, current_scope=None):
        progress_message = self._build_auto_tuning_progress_message(
            profile_name=profile_name,
            n_rows=n_rows,
            completed=completed,
            total=total,
            elapsed_seconds=elapsed_seconds,
            cancel_requested=cancel_requested,
            trees_completed=trees_completed,
            trees_total=trees_total,
        )
        self.auto_tuning_status_var.set(progress_message)

        progress_var = getattr(self, "_tuning_progress_var", None)
        if progress_var is not None:
            progress_var.set(progress_message)

        # Update current model being evaluated
        current_model_var = getattr(self, "_tuning_current_model_var", None)
        if current_model_var is not None and current_params is not None:
            _full_desc = self._format_candidate_params_full(
                current_params, scope=current_scope, covariates=current_covariates
            )
            _n_seeds_ui = len(self._parse_random_seeds()) if hasattr(self, "_parse_random_seeds") else 1
            _seed_info = f" | semilla 1/{_n_seeds_ui}" if _n_seeds_ui > 1 else ""
            _base_text = f"Evaluando #{completed + 1}/{total}{_seed_info}: {_full_desc}"
            if current_tree_text:
                _base_text = f"{_base_text}  |  {current_tree_text}"
            current_model_var.set(_base_text)

        # Update best metric so far
        best_metric_var = getattr(self, "_tuning_best_metric_var", None)
        if best_metric_var is not None:
            metric_label = self.tuning_progress_metric_var.get() if hasattr(self, "tuning_progress_metric_var") else "C-Uno (IPCW)"
            if metric_label == "CV":
                metric_label = self._resolve_cv_metric_labels()[0]
            if best_metric_value is not None and n_evaluated > 0:
                best_params_text = self._format_candidate_params_short(best_metric_params) if best_metric_params else ""
                best_metric_var.set(
                    f"Mejor {metric_label}: {best_metric_value:.4f}  ({best_params_text})  —  {n_evaluated} evaluados"
                )
            else:
                best_metric_var.set(
                    f"Mejor {metric_label}: —  ({n_evaluated} evaluados)"
                )

        # Update balanced best
        balanced_var = getattr(self, "_tuning_balanced_var", None)
        if balanced_var is not None and balanced_text is not None:
            balanced_var.set(balanced_text)

        progress_note_var = getattr(self, "_tuning_progress_note_var", None)
        if progress_note_var is not None and not cancel_requested:
            remaining = max(int(total or 0) - int(completed or 0), 0)
            progress_note_var.set(
                f"Restan {remaining} combinaciones. Si cancelas, se guardará lo que ya alcanzó a evaluarse."
            )
        progress_bar = getattr(self, "_tuning_progress_bar", None)
        if progress_bar is not None:
            try:
                progress_bar.configure(maximum=max(int(total or 1), 1))
                progress_bar["value"] = min(max(int(completed or 0), 0), max(int(total or 1), 1))
            except Exception:
                pass

        # Refresh live scatter only on the first models, periodically, and on cancel/end.
        _completed_now = int(completed or 0)
        _last_scatter_refresh = int(getattr(self, "_live_scatter_last_refresh_completed", -1) or -1)
        _scatter_refresh_step = max(1, int(getattr(self, "_live_scatter_refresh_every", 4) or 4))
        _should_refresh_scatter = (
            _completed_now <= 3 or
            cancel_requested or
            _completed_now >= int(total or 0) or
            (_completed_now - _last_scatter_refresh) >= _scatter_refresh_step
        )
        if _should_refresh_scatter:
            if not getattr(self, "_scatter_refresh_in_progress", False):
                self._scatter_refresh_in_progress = True
                try:
                    self._refresh_live_scatter()
                finally:
                    self._scatter_refresh_in_progress = False
            self._live_scatter_last_refresh_completed = _completed_now

        self._update_tuning_status_text_widget()

        dialog = getattr(self, "_tuning_progress_dialog", None)
        if dialog is not None:
            try:
                dialog.update_idletasks()
            except tk.TclError:
                self._tuning_cancel_requested = True

    def _close_live_scatter_popout(self):
        if bool(getattr(self, "_live_scatter_popout_closing", False)):
            return
        self._live_scatter_popout_closing = True

        popout_dialog = getattr(self, "_live_scatter_popout_dialog", None)
        popout_canvas = getattr(self, "_live_scatter_popout_canvas", None)
        popout_fig = getattr(self, "_live_scatter_popout_fig", None)

        # Drop references first so concurrent refreshes become no-ops.
        self._live_scatter_popout_dialog = None
        self._live_scatter_popout_fig = None
        self._live_scatter_popout_canvas = None

        # TkAgg can freeze if we close/destroy while draw is active;
        # schedule cleanup on idle and avoid explicit plt.close here.
        if popout_dialog is not None:
            try:
                if popout_dialog.winfo_exists():
                    popout_dialog.after_idle(popout_dialog.destroy)
            except Exception:
                pass

        if popout_canvas is not None:
            try:
                canvas_widget = popout_canvas.get_tk_widget()
                if canvas_widget is not None and int(canvas_widget.winfo_exists()) == 1:
                    canvas_widget.after_idle(canvas_widget.destroy)
            except Exception:
                pass

        if popout_fig is not None:
            try:
                popout_fig.clear()
            except Exception:
                pass

        self._live_scatter_popout_closing = False

    def _refresh_live_scatter_popout(self):
        if bool(getattr(self, "_live_scatter_popout_closing", False)):
            return

        popout_dialog = getattr(self, "_live_scatter_popout_dialog", None)
        popout_fig = getattr(self, "_live_scatter_popout_fig", None)
        popout_canvas = getattr(self, "_live_scatter_popout_canvas", None)
        if popout_dialog is None or popout_fig is None or popout_canvas is None:
            return
        try:
            if not popout_dialog.winfo_exists():
                self._close_live_scatter_popout()
                return
        except Exception:
            self._close_live_scatter_popout()
            return

        source_canvas = getattr(self, "_live_scatter_canvas", None)
        popout_fig.clear()
        ax = popout_fig.add_subplot(111)

        if source_canvas is None:
            ax.text(0.5, 0.5, "No hay gráfica en vivo disponible.", ha="center", va="center", fontsize=11, color="#6b7280")
            ax.set_axis_off()
            popout_canvas.draw_idle()
            return

        try:
            source_canvas.draw()
            frame_rgba = np.asarray(source_canvas.buffer_rgba())
        except Exception:
            frame_rgba = None

        if frame_rgba is None or frame_rgba.size <= 0:
            ax.text(0.5, 0.5, "No se pudo actualizar la vista separada.", ha="center", va="center", fontsize=11, color="#6b7280")
            ax.set_axis_off()
        else:
            ax.imshow(frame_rgba)
            ax.set_axis_off()
            ax.set_title("RSF en vivo (vista separada)", fontsize=10)

        try:
            popout_fig.tight_layout()
        except Exception:
            pass
        popout_canvas.draw_idle()

    def _open_live_scatter_popout(self):
        popout_dialog = getattr(self, "_live_scatter_popout_dialog", None)
        if popout_dialog is not None:
            try:
                if popout_dialog.winfo_exists():
                    popout_dialog.lift()
                    popout_dialog.focus_force()
                    self._refresh_live_scatter_popout()
                    return
            except Exception:
                pass

        parent_window = self.winfo_toplevel() if hasattr(self, "winfo_toplevel") else None
        popout_dialog = tk.Toplevel(parent_window if parent_window is not None else self)
        popout_dialog.title("RSF en vivo - Gráfica separada")
        popout_dialog.geometry("1280x820")
        popout_dialog.minsize(900, 560)
        if parent_window is not None:
            popout_dialog.transient(parent_window)

        header = ttk.Frame(popout_dialog, padding=(10, 8, 10, 4))
        header.pack(fill=tk.X)
        ttk.Label(
            header,
            text="Vista separada y redimensionable de la gráfica en vivo.",
            foreground="#334155",
        ).pack(side=tk.LEFT)
        ttk.Button(header, text="Actualizar", command=self._refresh_live_scatter_popout).pack(side=tk.RIGHT, padx=(6, 0))
        ttk.Button(header, text="Cerrar", command=self._close_live_scatter_popout).pack(side=tk.RIGHT)

        popout_fig = plt.figure(figsize=(12, 7))
        popout_canvas = FigureCanvasTkAgg(popout_fig, master=popout_dialog)
        popout_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))

        self._live_scatter_popout_dialog = popout_dialog
        self._live_scatter_popout_fig = popout_fig
        self._live_scatter_popout_canvas = popout_canvas
        self._live_scatter_popout_closing = False
        popout_dialog.protocol("WM_DELETE_WINDOW", self._close_live_scatter_popout)
        self._refresh_live_scatter_popout()

    def _update_autotune_vimp_chart(self):
        """Screening scatter: un punto por semilla, ejes X e Y seleccionables.
        Dibuja la cruz del promedio con sus IC al porcentaje elegido.
        Si ambos ejes son metricas C-index dibuja la referencia y=x."""
        import numpy as _np_v
        try:
            from scipy import stats as _sp_stats
            _HAS_SCIPY = True
        except ImportError:
            _HAS_SCIPY = False
        fig = getattr(self, '_autotune_vimp_fig', None)
        canvas = getattr(self, '_autotune_vimp_canvas', None)
        seed_recs = list(getattr(self, '_autotune_current_seed_metrics', None) or [])
        cand_n = getattr(self, '_autotune_current_candidate_n', 0)
        info_var = getattr(self, '_autotune_vimp_info_var', None)
        if fig is None or canvas is None:
            return

        _metric_key_map = {'OOB': 'oob', 'C-CV': 'cv', 'C-Test': 'ctest', 'BSS': 'bss'}
        _metric_label_map = {'oob': 'OOB', 'cv': 'C-CV', 'ctest': 'C-Test', 'bss': 'BSS'}
        _c_index_metrics = {'oob', 'cv', 'ctest'}

        x_var = getattr(self, '_autotune_screen_x_var', None)
        y_var = getattr(self, '_autotune_screen_y_var', None)
        ci_var = getattr(self, '_autotune_screen_ci_var', None)
        pt_size_var = getattr(self, '_autotune_screen_pt_size_var', None)
        show_seed_label_var = getattr(self, '_autotune_screen_show_seed_label_var', None)
        show_vals_var = getattr(self, '_autotune_screen_show_vals_var', None)
        x_label = x_var.get() if x_var is not None else 'OOB'
        y_label = y_var.get() if y_var is not None else 'C-Test'
        ci_str  = ci_var.get() if ci_var is not None else '95%'
        try:
            pt_size = max(10, int(float(pt_size_var.get()))) if pt_size_var is not None else 62
        except (ValueError, TypeError):
            pt_size = 62
        show_seed_label = show_seed_label_var.get() if show_seed_label_var is not None else True
        show_vals = show_vals_var.get() if show_vals_var is not None else True
        show_line_var = getattr(self, '_autotune_screen_show_line_var', None)
        show_traj_var = getattr(self, '_autotune_screen_show_traj_var', None)
        show_line = show_line_var.get() if show_line_var is not None else True
        show_traj = show_traj_var.get() if show_traj_var is not None else True
        x_key = _metric_key_map.get(x_label, 'oob')
        y_key = _metric_key_map.get(y_label, 'ctest')
        try:
            ci_level = float(ci_str.replace('%', '').strip()) / 100.0
        except (ValueError, AttributeError):
            ci_level = 0.95

        fig.clear()

        if not seed_recs:
            ax0 = fig.add_subplot(111)
            ax0.text(0.5, 0.5, 'Esperando semillas...', ha='center', va='center',
                     fontsize=9, color='#9ca3af')
            ax0.set_axis_off()
            canvas.draw_idle()
            if info_var is not None:
                try:
                    info_var.set('Esperando modelos...')
                except Exception:
                    pass
            return

        from matplotlib.gridspec import GridSpec as _GridSpec

        trees_done  = int(getattr(self, '_autotune_current_trees_done',  0))
        trees_total = int(getattr(self, '_autotune_current_trees_total', 0))

        # Layout: tiny progress bar on top, main scatter below
        _has_progress = trees_total > 0
        if _has_progress:
            gs = _GridSpec(2, 1, figure=fig,
                           height_ratios=[0.07, 0.93],
                           hspace=0.06)
            ax_bar = fig.add_subplot(gs[0])
            ax     = fig.add_subplot(gs[1])
        else:
            ax = fig.add_subplot(111)
            ax_bar = None

        # ── Trees progress bar ────────────────────────────────────────────────
        if ax_bar is not None:
            _pct = min(1.0, trees_done / trees_total) if trees_total > 0 else 0.0
            ax_bar.barh([0], [_pct], height=0.8, color='#3b82f6', alpha=0.82, left=0)
            ax_bar.barh([0], [1.0 - _pct], height=0.8, color='#e5e7eb', alpha=0.55, left=_pct)
            _n_seeds_done = len(seed_recs)
            _n_seeds_total = max(1, round(trees_total / max(1, trees_done / max(1, _n_seeds_done))))
            ax_bar.text(0.5, 0, f'{trees_done:,} / {trees_total:,} árboles  '
                                f'({_n_seeds_done}/{_n_seeds_total} semillas)  '
                                f'{_pct*100:.0f}%',
                        transform=ax_bar.transAxes, ha='center', va='center',
                        fontsize=7, fontweight='bold', color='#1e3a8a')
            ax_bar.set_xlim(0, 1)
            ax_bar.set_axis_off()
        # ─────────────────────────────────────────────────────────────────────

        n_s = len(seed_recs)

        xs, ys, labels = [], [], []
        for i, r in enumerate(seed_recs):
            xv = r.get(x_key)
            yv = r.get(y_key)
            if xv is not None and yv is not None:
                try:
                    xs.append(float(xv))
                    ys.append(float(yv))
                    labels.append(f'S{i+1}')
                except (TypeError, ValueError):
                    pass

        if not xs:
            ax.text(0.5, 0.5, 'Sin datos para los ejes seleccionados',
                    ha='center', va='center', fontsize=8, color='#9ca3af')
            ax.set_axis_off()
        else:
            _seed_colors = ['#2563eb', '#16a34a', '#dc2626', '#9333ea',
                            '#f59e0b', '#0891b2', '#be185d', '#65a30d']
            
            # Encontrar el índice de la semilla más cercana a la media
            closest_idx = -1
            if len(xs) > 0:
                mean_x, mean_y = _np_v.mean(xs), _np_v.mean(ys)
                range_x = max(1e-9, max(xs) - min(xs))
                range_y = max(1e-9, max(ys) - min(ys))
                dist_sq = [((x - mean_x) / range_x)**2 + ((y - mean_y) / range_y)**2 for x, y in zip(xs, ys)]
                closest_idx = dist_sq.index(min(dist_sq))

            for i, (xv, yv, lbl) in enumerate(zip(xs, ys, labels)):
                col = _seed_colors[i % len(_seed_colors)]
                marker = '*' if i == closest_idx else 'o'
                size = pt_size * 2.5 if i == closest_idx else pt_size
                ax.scatter([xv], [yv], color=col, marker=marker, s=size, zorder=4, alpha=0.88)
                if show_seed_label:
                    ax.annotate(lbl, (xv, yv), textcoords='offset points',
                                xytext=(5, 4), fontsize=7, color=col, fontweight='bold')
                if show_vals:
                    ax.annotate(f'{xv:.3f},{yv:.3f}', (xv, yv), textcoords='offset points',
                                xytext=(5, -8), fontsize=6, color='#4b5563')

            # Conectar puntos en orden de llegada con linea tenue
            if len(xs) > 1 and getattr(self, '_autotune_screen_show_line_var', None) and self._autotune_screen_show_line_var.get():
                ax.plot(xs, ys, color='#d1d5db', linewidth=0.8, zorder=2, alpha=0.5, linestyle='--')

            # ── Trayectoria de la media acumulada ────────────────────────────
            if len(xs) > 1 and getattr(self, '_autotune_screen_show_traj_var', None) and self._autotune_screen_show_traj_var.get():
                traj_mx = [_np_v.mean(xs[:k+1]) for k in range(len(xs))]
                traj_my = [_np_v.mean(ys[:k+1]) for k in range(len(ys))]
                ax.plot(traj_mx, traj_my,
                        color='#b45309', linewidth=1.8, zorder=3,
                        alpha=0.85, solid_capstyle='round',
                        label='Trayectoria media')
                # Puntitos en cada posición de la media acumulada
                ax.scatter(traj_mx[:-1], traj_my[:-1],
                           marker='o', s=18, color='#b45309', alpha=0.55, zorder=3)
            # ─────────────────────────────────────────────────────────────────

            # --- Cruz del promedio con IC ---
            n_pts = len(xs)
            mean_x = _np_v.mean(xs)
            mean_y = _np_v.mean(ys)
            ci_x_lo, ci_x_hi = mean_x, mean_x
            ci_y_lo, ci_y_hi = mean_y, mean_y
            _ci_label = f'{int(round(ci_level*100))}%'
            if n_pts >= 2 and _HAS_SCIPY:
                sem_x = _np_v.std(xs, ddof=1) / _np_v.sqrt(n_pts)
                sem_y = _np_v.std(ys, ddof=1) / _np_v.sqrt(n_pts)
                if sem_x > 0:
                    ci_x_lo, ci_x_hi = _sp_stats.t.interval(
                        ci_level, df=n_pts - 1, loc=mean_x, scale=sem_x)
                if sem_y > 0:
                    ci_y_lo, ci_y_hi = _sp_stats.t.interval(
                        ci_level, df=n_pts - 1, loc=mean_y, scale=sem_y)
            elif n_pts >= 2 and not _HAS_SCIPY:
                # IC aproximado sin scipy: z*sem (normal, conservador)
                _z = {0.80: 1.282, 0.90: 1.645, 0.95: 1.960, 0.99: 2.576}.get(ci_level, 1.960)
                sem_x = _np_v.std(xs, ddof=1) / _np_v.sqrt(n_pts)
                sem_y = _np_v.std(ys, ddof=1) / _np_v.sqrt(n_pts)
                ci_x_lo, ci_x_hi = mean_x - _z * sem_x, mean_x + _z * sem_x
                ci_y_lo, ci_y_hi = mean_y - _z * sem_y, mean_y + _z * sem_y

            # Dibujar la cruz: primero los brazos del IC
            ax.plot([ci_x_lo, ci_x_hi], [mean_y, mean_y],
                    color='#b45309', linewidth=2.0, zorder=5, alpha=0.85, solid_capstyle='round')
            ax.plot([mean_x, mean_x], [ci_y_lo, ci_y_hi],
                    color='#b45309', linewidth=2.0, zorder=5, alpha=0.85, solid_capstyle='round')
            # Capuchones en los extremos del IC
            _cap_kw = dict(color='#b45309', linewidth=1.5, zorder=5, alpha=0.85)
            _xr = ax.get_xlim() if ax.get_xlim() != (0.0, 1.0) else (min(xs)-0.01, max(xs)+0.01)
            _yr = ax.get_ylim() if ax.get_ylim() != (0.0, 1.0) else (min(ys)-0.01, max(ys)+0.01)
            _cap_h = ((_xr[1] - _xr[0]) if _xr[1] != _xr[0] else 0.05) * 0.02
            _cap_v = ((_yr[1] - _yr[0]) if _yr[1] != _yr[0] else 0.05) * 0.02
            for _cx in [ci_x_lo, ci_x_hi]:
                ax.plot([_cx, _cx], [mean_y - _cap_v, mean_y + _cap_v], **_cap_kw)
            for _cy in [ci_y_lo, ci_y_hi]:
                ax.plot([mean_x - _cap_h, mean_x + _cap_h], [_cy, _cy], **_cap_kw)
            # Punto central del promedio (cruz naranja gruesa)
            ax.scatter([mean_x], [mean_y], marker='+', s=200, color='#b45309',
                       linewidths=2.5, zorder=6, label=f'Prom±IC {_ci_label}')
            ax.annotate(f'ø({mean_x:.3f},{mean_y:.3f})', (mean_x, mean_y),
                        textcoords='offset points', xytext=(-4, 7),
                        fontsize=6, color='#b45309', fontweight='bold')

            # Referencia y=x cuando ambos ejes son tipo C-index
            if x_key in _c_index_metrics and y_key in _c_index_metrics:
                _all = list(xs) + list(ys)
                _lo = max(0.0, min(_all) - 0.05)
                _hi = min(1.0, max(_all) + 0.05)
                ax.plot([_lo, _hi], [_lo, _hi], color='#94a3b8', linewidth=1.1,
                        linestyle='--', alpha=0.7, label='y = x', zorder=1)

            ax.legend(fontsize=7.5, loc='lower right', framealpha=0.65)
            ax.set_xlabel(_metric_label_map.get(x_key, x_label), fontsize=8)
            ax.set_ylabel(_metric_label_map.get(y_key, y_label), fontsize=8)
            # ── Footer: model info instead of title (avoids overlap with progress bar) ──
            _cand_params = getattr(self, '_autotune_current_candidate_params', {})
            try:
                _footer_str = self._format_candidate_params_short(_cand_params) if _cand_params else ''
            except Exception:
                _footer_str = ''
            _footer_label = _metric_label_map.get(x_key, x_label)
            if _footer_str:
                ax.set_xlabel(f"{_footer_label}\n"
                              f"Cand. #{cand_n}  |  {n_s} semilla(s)  |  {_footer_str}",
                              fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(True, linewidth=0.4, alpha=0.35)

            # ── Text overlay: promedios e IC de cada eje ──────────────────
            _all_metric_keys = list(_metric_key_map.values())  # oob, cv, ctest, bss
            _overlay_lines = []
            for _mk in _all_metric_keys:
                _vals_m = [float(r[_mk]) for r in seed_recs
                           if r.get(_mk) is not None and _np_v.isfinite(float(r[_mk]))]
                if not _vals_m:
                    continue
                _mean_m = _np_v.mean(_vals_m)
                _n_m = len(_vals_m)
                if _n_m >= 2 and _HAS_SCIPY:
                    _sem_m = _np_v.std(_vals_m, ddof=1) / _np_v.sqrt(_n_m)
                    _lo_m, _hi_m = _sp_stats.t.interval(ci_level, df=_n_m - 1, loc=_mean_m, scale=_sem_m) if _sem_m > 0 else (_mean_m, _mean_m)
                elif _n_m >= 2:
                    _z = {0.80: 1.282, 0.90: 1.645, 0.95: 1.960, 0.99: 2.576}.get(ci_level, 1.960)
                    _sem_m = _np_v.std(_vals_m, ddof=1) / _np_v.sqrt(_n_m)
                    _lo_m, _hi_m = _mean_m - _z * _sem_m, _mean_m + _z * _sem_m
                else:
                    _lo_m, _hi_m = _mean_m, _mean_m
                _lbl_m = _metric_label_map.get(_mk, _mk)
                if _n_m >= 2:
                    _overlay_lines.append(f'{_lbl_m}: {_mean_m:.3f} [{_lo_m:.3f}–{_hi_m:.3f}]')
                else:
                    _overlay_lines.append(f'{_lbl_m}: {_mean_m:.3f}')
            if _overlay_lines:
                _overlay_text = f'IC {_ci_label}\n' + '\n'.join(_overlay_lines)
                # Tamaño de fuente adaptado al número de líneas: más líneas → fuente más pequeña
                _n_overlay_lines = len(_overlay_lines) + 1  # +1 por el encabezado IC
                _overlay_fs = max(6.5, min(9.5, 46.0 / max(_n_overlay_lines, 1)))
                ax.text(0.01, 0.99, _overlay_text, transform=ax.transAxes,
                        fontsize=_overlay_fs, va='top', ha='left', color='#1e3a5f',
                        bbox=dict(boxstyle='round,pad=0.35', facecolor='#f0f4ff',
                                  edgecolor='#93c5fd', alpha=0.82),
                        zorder=7)
            # ─────────────────────────────────────────────────────────────

        fig.tight_layout(pad=0.7)
        try:
            canvas.draw()
        except Exception:
            canvas.draw_idle()

        if info_var is not None:
            try:
                info_var.set(f'Candidato #{cand_n} | {n_s} semilla(s) | X={x_label} Y={y_label} | IC={ci_str}')
            except Exception:
                pass


    def _refresh_live_scatter(self):
        fig = getattr(self, "_live_scatter_fig", None)
        canvas = getattr(self, "_live_scatter_canvas", None)
        results = getattr(self, "_live_tuning_results", None)
        if fig is None or canvas is None or results is None:
            return

        scatter_vars = getattr(self, "_live_scatter_vars", None)
        if not scatter_vars or len(scatter_vars) < 3:
            return

        kx, ky, kz = scatter_vars[0].get(), scatter_vars[1].get(), scatter_vars[2].get()

        def _lbl(key):
            return self._get_metric_display_label(key)

        def _valid_metric_value(key, value):
            if value is None:
                return False
            try:
                v = float(value)
            except (TypeError, ValueError):
                return False
            if not np.isfinite(v):
                return False
            if str(key).startswith("c_index") or str(key) == "oob_score":
                return 0.0 <= v <= 1.0
            if str(key) == "ibs":
                return v >= 0.0
            if str(key) == "tau":
                return v > 0.0
            return True

        if not results:
            fig.clear()
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "Esperando modelos evaluados...", ha="center", va="center", fontsize=11, color="#6b7280")
            ax.set_axis_off()
            self._update_live_scatter_filter_options([])
            self._live_scatter_all_records = []
            self._live_scatter_data = {
                "all_x": [],
                "all_y": [],
                "all_z": [],
                "all_ids": [],
                "results": [],
                "all_apt": [],
                "ax": ax,
                "kx": kx,
                "ky": ky,
                "kz": kz,
                "best_metric_idx": None,
                "clinical_best_idx": None,
            }
            self._set_live_scatter_info_line("Esperando modelos...")
            canvas.draw_idle()
            self._refresh_live_scatter_popout()
            return

        point_size_var = getattr(self, "_live_scatter_point_size_var", None)
        try:
            base_size = float(point_size_var.get()) if point_size_var is not None else 50.0
        except Exception:
            base_size = 50.0
        base_size = max(10.0, min(base_size, 250.0))

        show_apt_var = getattr(self, "_live_scatter_show_apt_var", None)
        show_only_apt = bool(show_apt_var.get()) if show_apt_var is not None else False

        apply_gap_filters_var = getattr(self, "_live_scatter_apply_gap_filters_var", None)
        apply_gap_filters = bool(apply_gap_filters_var.get()) if apply_gap_filters_var is not None else True

        use_global_gap_var = getattr(self, "_live_scatter_use_global_gap_var", None)
        use_global_gap = bool(use_global_gap_var.get()) if use_global_gap_var is not None else False

        gap_global_var = getattr(self, "_live_scatter_gap_global_var", None)
        gap_global = self._coerce_gap_threshold(
            gap_global_var.get() if gap_global_var is not None else self._PLOT_VIABILITY_GAP_DEFAULT,
            self._PLOT_VIABILITY_GAP_DEFAULT,
            maximum=self._PLOT_VIABILITY_GAP_MAX,
        )

        gap_oob_cv_var = getattr(self, "_live_scatter_gap_oob_cv_var", None)
        gap_oob_test_var = getattr(self, "_live_scatter_gap_oob_test_var", None)
        gap_cv_test_var = getattr(self, "_live_scatter_gap_cv_test_var", None)
        gap_oob_cv = self._coerce_gap_threshold(
            gap_oob_cv_var.get() if gap_oob_cv_var is not None else self._PLOT_GAP_OOB_CV_DEFAULT,
            self._PLOT_GAP_OOB_CV_DEFAULT,
            maximum=self._PLOT_VIABILITY_GAP_MAX,
        )
        gap_oob_test = self._coerce_gap_threshold(
            gap_oob_test_var.get() if gap_oob_test_var is not None else self._PLOT_GAP_OOB_TEST_DEFAULT,
            self._PLOT_GAP_OOB_TEST_DEFAULT,
            maximum=self._PLOT_VIABILITY_GAP_MAX,
        )
        gap_cv_test = self._coerce_gap_threshold(
            gap_cv_test_var.get() if gap_cv_test_var is not None else self._PLOT_GAP_CV_TEST_DEFAULT,
            self._PLOT_GAP_CV_TEST_DEFAULT,
            maximum=self._PLOT_VIABILITY_GAP_MAX,
        )
        if use_global_gap:
            gap_oob_cv = float(gap_global)
            gap_oob_test = float(gap_global)
            gap_cv_test = float(gap_global)

        show_ids_var = getattr(self, "_live_scatter_show_ids_var", None)
        _show_ids_live = bool(show_ids_var.get()) if show_ids_var is not None else False

        show_ci_var = getattr(self, "_live_scatter_show_ci_var", None)
        _show_ci_live = bool(show_ci_var.get()) if show_ci_var is not None else True

        heatmap_var = getattr(self, "_live_scatter_heatmap_var", None)
        use_heatmap = bool(heatmap_var.get()) if heatmap_var is not None else False

        heat_bins_var = getattr(self, "_live_scatter_heat_bins_var", None)
        try:
            heat_bins = max(8, int(float(str(heat_bins_var.get()).strip()))) if heat_bins_var is not None else 35
        except Exception:
            heat_bins = 35

        marker_mode_var = getattr(self, "_live_scatter_marker_mode_var", None)
        marker_mode = str(marker_mode_var.get() if marker_mode_var is not None else "Combinación")

        from collections import Counter

        def _extract_cov_signature(record):
            covs = [str(c).strip() for c in record.get("covariates", []) if str(c).strip()]
            if covs:
                return " + ".join(covs)
            scope = record.get("scope") or (record.get("display", {}) or {}).get("scope")
            return str(scope or "Sin covariables")

        def _extract_mode(record):
            mode_text = str(record.get("mode") or "").strip()
            if mode_text:
                return mode_text
            covs = [str(c).strip() for c in record.get("covariates", []) if str(c).strip()]
            if not covs:
                return "-"
            if len(covs) == 1:
                return "1 var"
            return f"{len(covs)} vars"

        def _build_record(idx, record):
            params = record.get("params", {}) if isinstance(record, dict) else {}
            metrics = record.get("metrics", {}) if isinstance(record, dict) else {}
            try:
                model_id = int(record.get("model_id"))
            except Exception:
                model_id = int(idx) + 1
            scope_txt = str(record.get("scope") or (record.get("display", {}) or {}).get("scope") or "General")
            return {
                "uid": id(record),
                "index": int(idx),
                "model_id": model_id,
                "snapshot": record,
                "params": params,
                "metrics": metrics,
                "cov_sig": _extract_cov_signature(record),
                "scope": scope_txt,
                "mode": _extract_mode(record),
                "trees": str(params.get("n_estimators", "-")),
                "max_features": str(params.get("max_features", "all")),
                "min_leaf": str(params.get("min_samples_leaf", "-")),
                "min_split": str(params.get("min_samples_split", "-")),
            }

        marker_complete_sym = "o"
        marker_incomplete_sym = "X"

        def _resolve_marker_category(record):
            if marker_mode == "Scope":
                return str(record.get("scope", "-"))
            if marker_mode == "Modo":
                return str(record.get("mode", "-"))
            if marker_mode == "max_features":
                return str(record.get("max_features", "-"))
            if marker_mode == "min_leaf":
                return str(record.get("min_leaf", "-"))
            if marker_mode == "min_split":
                return str(record.get("min_split", "-"))
            if marker_mode == "Completo/Semiinc./Incompleto":
                snap = record.get("snapshot", record)
                if isinstance(snap, dict):
                    if bool(snap.get("discarded_by_screening", False)):
                        return "Incompleto (Screening)"
                    if bool(snap.get("semi_complete", False)):
                        return "Semiincompleto"
                return "Completado"
            return str(record.get("cov_sig", "-"))

        advanced_filter_fields = list(getattr(self, "_live_scatter_advanced_filter_fields", []) or [])
        if not advanced_filter_fields:
            advanced_filter_fields = [
                ("cov_sig", "Combinación covariables"),
                ("scope", "Scope"),
                ("mode", "Modo"),
                ("trees", "Árboles"),
                ("max_features", "max_features"),
                ("min_leaf", "min_leaf"),
                ("min_split", "min_split"),
            ]
            self._live_scatter_advanced_filter_fields = list(advanced_filter_fields)

        advanced_filters = getattr(self, "_live_scatter_advanced_filters", None)
        if not isinstance(advanced_filters, dict):
            advanced_filters = {k: set() for k, _ in advanced_filter_fields}
            self._live_scatter_advanced_filters = advanced_filters
        for field_key, _ in advanced_filter_fields:
            advanced_filters.setdefault(field_key, set())

        # ── Metric min/max filters ──────────────────────────────────────
        _metric_filters = getattr(self, "_live_scatter_metric_filters", None)
        if not isinstance(_metric_filters, dict):
            _metric_filters = {}
            self._live_scatter_metric_filters = _metric_filters

        def _passes_metric_filters(metrics_dict):
            for _mk, _bounds in _metric_filters.items():
                _val = metrics_dict.get(_mk)
                if _val is None:
                    continue
                try:
                    _fv = float(_val)
                except (TypeError, ValueError):
                    continue
                _mn = _bounds.get("min")
                _mx = _bounds.get("max")
                if _mn is not None and _fv < _mn:
                    return False
                if _mx is not None and _fv > _mx:
                    return False
            return True

        def _passes_advanced_filters(record):
            for field_key, _ in advanced_filter_fields:
                selected = advanced_filters.get(field_key) or set()
                if selected and str(record.get(field_key, "-")) not in selected:
                    return False
            return True

        all_records = []
        _show_completed_var = getattr(self, "_live_scatter_show_completed_only_var", None)
        _filter_completed = _show_completed_var.get() == "Completados" if _show_completed_var else False

        for idx, raw in enumerate(results):
            if not isinstance(raw, dict):
                continue
            if _filter_completed and raw.get("discarded_by_screening", False):
                continue
            rec = _build_record(idx, raw)
            metrics = rec.get("metrics", {})

            vx, vy, vz = metrics.get(kx), metrics.get(ky), metrics.get(kz)
            if not (_valid_metric_value(kx, vx) and _valid_metric_value(ky, vy)):
                continue
            # Color (Z) is optional: plotted in gray if missing
            vz_valid = _valid_metric_value(kz, vz)

            if not _passes_metric_filters(metrics):
                continue

            vx = float(vx)
            vy = float(vy)
            vz = float(vz) if vz_valid else None

            bss = metrics.get("bss")
            gap_ok, gap_diag = self._passes_metric_gap_thresholds(
                metrics,
                oob_cv_threshold=gap_oob_cv,
                oob_test_threshold=gap_oob_test,
                cv_test_threshold=gap_cv_test,
            )
            bss_ok = (bss is None or not _valid_metric_value("bss", bss) or float(bss) > float(self._RANK_BSS_MIN))
            apt = bool(bss_ok and gap_ok)

            rec["x"] = vx
            rec["y"] = vy
            rec["c"] = vz
            rec["apt"] = apt
            rec["gap_ok"] = bool(gap_ok)
            rec["gap_diag"] = gap_diag
            all_records.append(rec)

        self._live_scatter_all_records = list(all_records)
        self._update_live_scatter_filter_options(all_records)

        filter_var = getattr(self, "_live_scatter_filter_var", None)
        show_filter = str((filter_var.get() if filter_var is not None else "Todos") or "Todos")
        profile_filter = show_filter[len("Perfil: "):].strip() if show_filter.startswith("Perfil: ") else ""

        pulse_seen = getattr(self, "_live_scatter_pulse_seen_uids", None)
        if not isinstance(pulse_seen, set):
            pulse_seen = set()
            self._live_scatter_pulse_seen_uids = pulse_seen
        new_records = [r for r in all_records if r.get("uid") not in pulse_seen]
        for rec in all_records:
            pulse_seen.add(rec.get("uid"))

        visible_records = []
        for rec in all_records:
            apt = bool(rec.get("apt"))
            if show_only_apt:
                if not apt:
                    continue
            else:
                if show_filter == "Solo aptos" and not apt:
                    continue
                if show_filter == "Solo no aptos" and apt:
                    continue
                if profile_filter and str(rec.get("cov_sig", "")).strip() != profile_filter:
                    continue

            if apply_gap_filters and not bool(rec.get("gap_ok", True)):
                continue

            if not _passes_advanced_filters(rec):
                continue

            visible_records.append(rec)

        fig.clear()
        ax = fig.add_subplot(111)

        all_x = [r.get("x") for r in visible_records]
        all_y = [r.get("y") for r in visible_records]
        all_z = [r.get("c") for r in visible_records]
        all_ids = [r.get("model_id") for r in visible_records]
        all_res = [r.get("snapshot") for r in visible_records]
        all_apt = [bool(r.get("apt")) for r in visible_records]
        last_visible_idx = None
        if visible_records and all_records:
            latest_uid = all_records[-1].get("uid")
            for i, rec in enumerate(visible_records):
                if rec.get("uid") == latest_uid:
                    last_visible_idx = i
                    break

        # Split records into those with valid Z-color and those without
        _colored_recs = [r for r in visible_records if r.get("c") is not None]
        _gray_recs    = [r for r in visible_records if r.get("c") is None]

        c_arr_colored = np.asarray([r["c"] for r in _colored_recs], dtype=float) if _colored_recs else np.asarray([], dtype=float)
        norm = None
        mappable = None
        if c_arr_colored.size > 1 and len(np.unique(np.round(c_arr_colored, 8))) > 1:
            norm = mcolors.Normalize(vmin=float(np.min(c_arr_colored)), vmax=float(np.max(c_arr_colored)))

        render_mode_note = ""
        hidden_group_points = 0
        marker_group_count = 0
        if visible_records:
            _fast_threshold = int(getattr(self, "_live_scatter_fast_mode_threshold", 450) or 450)
            fast_mode = bool(self._auto_tuning_in_progress and len(visible_records) >= _fast_threshold and not use_heatmap)
            if use_heatmap and _colored_recs:
                _hx = [r["x"] for r in _colored_recs]
                _hy = [r["y"] for r in _colored_recs]
                _hz = [r["c"] for r in _colored_recs]
                mappable = ax.hexbin(
                    _hx,
                    _hy,
                    C=_hz,
                    reduce_C_function=np.mean,
                    gridsize=heat_bins,
                    cmap="viridis",
                    mincnt=1,
                    linewidths=0.0,
                    alpha=0.92,
                    zorder=1,
                )
                if _gray_recs:
                    ax.scatter(
                        [r["x"] for r in _gray_recs],
                        [r["y"] for r in _gray_recs],
                        c="#9ca3af", marker="o", s=base_size * 0.7,
                        alpha=0.65, edgecolors="none", linewidths=0.0, zorder=1,
                        label=f"Sin {_lbl(kz)} ({len(_gray_recs)})",
                    )
                render_mode_note = "mapa de calor"
            elif fast_mode:
                if _colored_recs:
                    _cx = [r["x"] for r in _colored_recs]
                    _cy = [r["y"] for r in _colored_recs]
                    _cz = [r["c"] for r in _colored_recs]
                    sc = ax.scatter(
                        _cx, _cy,
                        c=_cz if norm is not None else "#3b82f6",
                        cmap="viridis" if norm is not None else None,
                        norm=norm, marker="o", s=base_size,
                        alpha=0.9, edgecolors="none", linewidths=0.0, zorder=2,
                        label=f"Con {_lbl(kz)} ({len(_colored_recs)})",
                    )
                    if norm is not None:
                        mappable = sc
                if _gray_recs:
                    ax.scatter(
                        [r["x"] for r in _gray_recs],
                        [r["y"] for r in _gray_recs],
                        c="#9ca3af", marker="o", s=base_size * 0.7,
                        alpha=0.65, edgecolors="none", linewidths=0.0, zorder=2,
                        label=f"Sin {_lbl(kz)} ({len(_gray_recs)})",
                    )
                render_mode_note = "modo rápido"
            else:
                marker_categories = [_resolve_marker_category(r) for r in visible_records]
                category_buckets = {}
                for rec, cat in zip(visible_records, marker_categories):
                    category_buckets.setdefault(cat, []).append(rec)

                ordered_groups = sorted(category_buckets.keys(), key=lambda _g: len(category_buckets[_g]), reverse=True)
                marker_group_count = len(ordered_groups)
                max_draw_groups = int(getattr(self, "_live_scatter_max_draw_groups", 24) or 24)
                draw_groups = ordered_groups[:max_draw_groups]
                hidden_groups = ordered_groups[max_draw_groups:]
                hidden_group_points = sum(len(category_buckets[g]) for g in hidden_groups)

                marker_symbols = list(getattr(self, "_live_scatter_marker_symbols", []) or [])
                if not marker_symbols:
                    marker_symbols = ["o", "s", "^", "D", "P", "v", "<", ">", "h", "8", "p"]
                marker_map = {
                    g: marker_symbols[i % len(marker_symbols)]
                    for i, g in enumerate(draw_groups)
                }

                max_legend_groups = 10
                legend_groups = set(draw_groups[:max_legend_groups])

                for group in draw_groups:
                    group_points = category_buckets.get(group, [])
                    gx = [p.get("x") for p in group_points]
                    gy = [p.get("y") for p in group_points]
                    # Split colored vs gray within group
                    gx_c = [p["x"] for p in group_points if p.get("c") is not None]
                    gy_c = [p["y"] for p in group_points if p.get("c") is not None]
                    gz_c = [p["c"] for p in group_points if p.get("c") is not None]
                    gx_g = [p["x"] for p in group_points if p.get("c") is None]
                    gy_g = [p["y"] for p in group_points if p.get("c") is None]
                    label = f"{group} ({len(group_points)})" if (len(draw_groups) > 1 and group in legend_groups) else None
                    if marker_mode == "Completo/Semiinc./Incompleto":
                        if group == "Completado":
                            _mk = marker_complete_sym
                            _edge = "none"
                            _lw = 0.0
                            _alpha_c = 0.9
                        elif group == "Semiincompleto":
                            _mk = "D"
                            _edge = "#f59e0b"
                            _lw = 1.4
                            _alpha_c = 0.82
                        else:  # Incompleto (Screening)
                            _mk = marker_incomplete_sym
                            _edge = "#475569"
                            _lw = 1.2
                            _alpha_c = 0.75
                    else:
                        _mk = marker_map.get(group, "o")
                        _edge = "none"
                        _lw = 0.0
                        _alpha_c = 0.9
                    if gx_c:
                        sc = ax.scatter(
                            gx_c, gy_c,
                            c=gz_c if norm is not None else "#3b82f6",
                            cmap="viridis" if norm is not None else None,
                            norm=norm,
                            marker=_mk,
                            s=base_size, alpha=_alpha_c, edgecolors=_edge, linewidths=_lw, zorder=2,
                            label=label,
                        )
                        if norm is not None and mappable is None:
                            mappable = sc
                        label = None  # don't duplicate legend for gray sub-group
                    if gx_g:
                        ax.scatter(
                            gx_g, gy_g,
                            c="#9ca3af",
                            marker=_mk,
                            s=base_size * 0.7, alpha=0.65, edgecolors="none", linewidths=0.0, zorder=2,
                            label=label,
                        )

                if hidden_group_points > 0:
                    other_points = []
                    for group in hidden_groups:
                        other_points.extend(category_buckets.get(group, []))
                    ox_c = [p["x"] for p in other_points if p.get("c") is not None]
                    oy_c = [p["y"] for p in other_points if p.get("c") is not None]
                    oc   = [p["c"] for p in other_points if p.get("c") is not None]
                    ox_g = [p["x"] for p in other_points if p.get("c") is None]
                    oy_g = [p["y"] for p in other_points if p.get("c") is None]
                    if ox_c:
                        sc2 = ax.scatter(
                            ox_c, oy_c,
                            c=oc if norm is not None else "#6b7280",
                            cmap="viridis" if norm is not None else None,
                            norm=norm, marker="o",
                            s=max(18.0, base_size * 0.65), alpha=0.72,
                            edgecolors="none", linewidths=0.0,
                            label=f"Otros tipos ({hidden_group_points})",
                        )
                        if norm is not None and mappable is None and oc:
                            mappable = sc2
                    if ox_g:
                        ax.scatter(
                            ox_g, oy_g,
                            c="#9ca3af", marker="o",
                            s=max(18.0, base_size * 0.65), alpha=0.55,
                            edgecolors="none", linewidths=0.0,
                            label=None,
                        )

                if marker_group_count > 0:
                    render_mode_note = f"símbolos ({marker_group_count} grupos)"

        # No overlay visual for apt/non-apt: filtering controls already handle visibility.

        if _show_ci_live and visible_records:
            ci_count = 0
            for rec in visible_records:
                drawn = self._draw_ci_cross_for_point(
                    ax,
                    rec.get("x"),
                    rec.get("y"),
                    rec.get("metrics", {}),
                    kx,
                    ky,
                    color="#475569",
                    alpha=0.33,
                    linewidth=0.9,
                    zorder=1.7,
                )
                if drawn:
                    ci_count += 1
            if ci_count > 0:
                ax.plot([], [], color="#475569", alpha=0.45, lw=1.0, label=f"IC cruz ({ci_count})")

        if new_records:
            nx = [r.get("x") for r in new_records]
            ny = [r.get("y") for r in new_records]
            bg_rgb = np.asarray(ax.get_facecolor()[:3], dtype=float)
            inv_rgb = np.clip(1.0 - bg_rgb, 0.0, 1.0)
            pulse_edge = mcolors.to_hex(inv_rgb)
            pulse_fill = "#ffffff"

            ax.scatter(
                nx,
                ny,
                facecolors="none",
                edgecolors=pulse_edge,
                s=base_size * 3.5,
                linewidths=2.6,
                alpha=0.92,
                zorder=11,
            )
            ax.scatter(
                nx,
                ny,
                facecolors=pulse_fill,
                edgecolors=pulse_edge,
                s=base_size * 2.6,
                linewidths=1.2,
                alpha=0.90,
                zorder=12,
                label=f"Nuevos ({len(new_records)})",
            )

        if last_visible_idx is not None and all_x and all_y and last_visible_idx < len(all_x):
            pass  # "Último modelo" removed — selection persists via _live_scatter_selected_uid

        # ── Persistent selection highlight ─────────────────────────────
        _sel_uid = getattr(self, "_live_scatter_selected_uid", None)
        if _sel_uid is not None:
            for i, rec in enumerate(visible_records):
                if rec.get("uid") == _sel_uid:
                    ax.scatter(
                        [all_x[i]], [all_y[i]],
                        c="none", s=base_size * 5,
                        marker="o",
                        edgecolors="#000000",
                        linewidths=2.5,
                        label="Seleccionado",
                        zorder=16,
                    )
                    break

        if mappable is not None and c_arr_colored.size > 0:
            cbar = fig.colorbar(mappable, ax=ax, pad=0.02, fraction=0.05)
            cbar.set_label(_lbl(kz), fontsize=7)
            c_min = float(np.min(c_arr_colored))
            c_max = float(np.max(c_arr_colored))
            c_unique = len(np.unique(np.round(c_arr_colored, 8)))
            tick_count = int(max(3, min(7, c_unique)))
            cbar.set_ticks(np.linspace(c_min, c_max, num=tick_count))
            if str(kz) in {"bss", "ibs", "c_index_cv_std"}:
                cbar.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.4f"))
            elif str(kz) == "tau":
                cbar.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
            else:
                cbar.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
            cbar.update_ticks()
            cbar.ax.tick_params(labelsize=7)

        if all_x and all_y and self._should_draw_coherence_diagonal(kx, ky):
            minv = min(float(np.min(all_x)), float(np.min(all_y)))
            maxv = max(float(np.max(all_x)), float(np.max(all_y)))
            ax.plot([minv, maxv], [minv, maxv], color="#FFB300", lw=1.8, ls="--", label="Diagonal y=x (equilibrio C-index/OOB)")

        axis_records = list(visible_records) + list(new_records)
        if axis_records:
            x_vals = np.asarray([r.get("x") for r in axis_records], dtype=float)
            y_vals = np.asarray([r.get("y") for r in axis_records], dtype=float)
            if x_vals.size > 0 and y_vals.size > 0:
                x_span = float(np.max(x_vals) - np.min(x_vals))
                y_span = float(np.max(y_vals) - np.min(y_vals))
                x_pad = max(0.01, 0.06 * x_span) if x_span > 0 else max(0.01, abs(float(np.mean(x_vals))) * 0.03)
                y_pad = max(0.01, 0.06 * y_span) if y_span > 0 else max(0.01, abs(float(np.mean(y_vals))) * 0.03)
                ax.set_xlim(float(np.min(x_vals) - x_pad), float(np.max(x_vals) + x_pad))
                ax.set_ylim(float(np.min(y_vals) - y_pad), float(np.max(y_vals) + y_pad))

        metric_key, metric_higher_better = self._resolve_tuning_progress_metric_key()
        best_metric_idx = None
        best_metric_val = None
        for i, rec in enumerate(visible_records):
            mv = rec.get("metrics", {}).get(metric_key)
            if not _valid_metric_value(metric_key, mv):
                continue
            mvf = float(mv)
            if best_metric_idx is None:
                best_metric_idx = i
                best_metric_val = mvf
                continue
            ref = float(best_metric_val)
            is_better = (mvf > ref) if metric_higher_better else (mvf < ref)
            if is_better:
                best_metric_idx = i
                best_metric_val = mvf
        if best_metric_idx is not None and all_x and all_y:
            pass  # marcador quitado por solicitud del usuario

        clinical_best_idx = None
        try:
            apt_snaps = [rec.get("snapshot") for rec in visible_records if bool(rec.get("apt"))]
            if not apt_snaps:
                apt_snaps = [rec.get("snapshot") for rec in visible_records]
            if apt_snaps:
                ranked_apt, _ = self._rank_tuning_results(list(apt_snaps))
                if ranked_apt:
                    best_clinical = ranked_apt[0]
                    for i, rec in enumerate(visible_records):
                        if rec.get("snapshot") is best_clinical:
                            clinical_best_idx = i
                            break
        except Exception:
            clinical_best_idx = None
        if clinical_best_idx is not None and all_x and all_y:
            pass  # marcador quitado por solicitud del usuario

        if all_x and all_y and _show_ids_live:
            x_span = max(float(np.max(all_x) - np.min(all_x)), 1e-9)
            y_span = max(float(np.max(all_y) - np.min(all_y)), 1e-9)
            dx = 0.006 * x_span
            dy = 0.008 * y_span
            for idx_pos, (px, py) in enumerate(zip(all_x, all_y)):
                mid = all_ids[idx_pos] if idx_pos < len(all_ids) else (idx_pos + 1)
                jitter_x = (((idx_pos % 5) - 2) * 0.35) * dx
                jitter_y = (((idx_pos % 7) - 3) * 0.35) * dy
                ax.text(
                    px + jitter_x,
                    py + jitter_y,
                    str(mid),
                    fontsize=6,
                    color="#1F2937",
                    alpha=0.85,
                    ha="left",
                    va="bottom",
                    zorder=6,
                )

        if not visible_records and not new_records:
            ax.text(0.5, 0.5, "No hay modelos visibles con X e Y válidas.", ha="center", va="center", fontsize=11, color="#999")

        ax.set_xlabel(_lbl(kx), fontsize=10)
        ax.set_ylabel(_lbl(ky), fontsize=10)
        ax.set_title("Explorador de modelos RSF", fontsize=11, fontweight="bold")
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3, fontsize=8, framealpha=0.92)
        ax.grid(True, alpha=0.15)

        sd = getattr(self, "_live_scatter_data", {})
        sd["all_x"] = all_x
        sd["all_y"] = all_y
        sd["all_z"] = all_z
        sd["all_ids"] = all_ids
        sd["results"] = all_res
        sd["records"] = visible_records
        sd["ax"] = ax
        sd["kx"] = kx
        sd["ky"] = ky
        sd["kz"] = kz
        sd["all_apt"] = all_apt
        sd["gap_oob_cv"] = gap_oob_cv
        sd["gap_oob_test"] = gap_oob_test
        sd["gap_cv_test"] = gap_cv_test
        sd["use_global_gap"] = use_global_gap
        sd["gap_global"] = gap_global
        sd["apply_gap_filters"] = apply_gap_filters
        sd["best_metric_idx"] = best_metric_idx
        sd["clinical_best_idx"] = clinical_best_idx
        sd["last_visible_idx"] = last_visible_idx
        self._live_scatter_data = sd

        summary_parts = [f"{len(all_x)} modelos visibles"]
        if show_only_apt:
            summary_parts.append("filtro: solo aptos")
        elif show_filter != "Todos":
            summary_parts.append(f"filtro: {show_filter}")
        n_adv = sum(1 for values in advanced_filters.values() if values)
        if n_adv:
            summary_parts.append(f"filtros avanzados: {n_adv} activos")
        if apply_gap_filters:
            if use_global_gap:
                summary_parts.append(
                    f"umbrales ON | Δ único(CV/Test/OOB)≤{gap_global:.2f}"
                )
            else:
                summary_parts.append(
                    f"umbrales ON | OOB-CV≤{gap_oob_cv:.2f} | OOB-C-test≤{gap_oob_test:.2f} | CV-C-test≤{gap_cv_test:.2f}"
                )
        else:
            summary_parts.append("umbrales OFF")
        if use_heatmap:
            summary_parts.append(f"vista: mapa de calor (bins={heat_bins})")
        elif render_mode_note:
            summary_parts.append(f"vista: {render_mode_note}")
        if hidden_group_points > 0:
            summary_parts.append(f"tipos agrupados: {hidden_group_points}")
        if new_records:
            summary_parts.append(f"nuevos: {len(new_records)}")
        if best_metric_idx is not None and best_metric_idx < len(all_ids):
            summary_parts.append(f"Mejor métrica: #{all_ids[best_metric_idx]}")
        if clinical_best_idx is not None and clinical_best_idx < len(all_ids):
            summary_parts.append(f"Mejor clínico: #{all_ids[clinical_best_idx]}")
        # "Último" removed — replaced by persistent selection
        _sel_uid_now = getattr(self, "_live_scatter_selected_uid", None)
        if _sel_uid_now is not None:
            for _r in visible_records:
                if _r.get("uid") == _sel_uid_now:
                    summary_parts.append(f"Seleccionado: #{_r.get('model_id','?')}")
                    break
        summary_parts.append("clic en punto = ver características")
        # Restore selected info line if there's an active selection; otherwise show summary
        _sel_info = getattr(self, "_live_scatter_selected_info", None)
        if _sel_uid_now is not None and _sel_info:
            self._set_live_scatter_info_line(_sel_info)
        else:
            self._set_live_scatter_info_line("  |  ".join(summary_parts))

        try:
            fig.tight_layout(rect=(0.0, 0.08, 1.0, 1.0))
        except Exception:
            pass
        canvas.draw_idle()
        self._refresh_live_scatter_popout()

    def _flush_tuning_dialog_events(self):
        """Force Tkinter to process pending events (e.g. Cancel click) during long-running loops."""
        dialog = getattr(self, "_tuning_progress_dialog", None)
        if dialog is not None:
            try:
                # Procesar eventos de entrada (clics, teclado) para que el botón Cancelar
                # responda entre iteraciones, sin arriesgar re-entrada al loop de tuning.
                dialog.update()
            except tk.TclError:
                self._tuning_cancel_requested = True

    def _fit_model_with_ui_pump(self, model, X_train, y_train):
        """Run model.fit() in a background thread while keeping the UI responsive.

        scikit-survival's RandomSurvivalForest releases the GIL during its C/numpy
        operations, so a background thread allows Tkinter to keep processing events
        (including the Cancel button) every ~100 ms instead of freezing.

        Note: on Windows, joblib's loky backend can deadlock when spawn is initiated
        from a non-main thread.  To avoid this, we force n_jobs=1 inside the thread
        so the fit uses a single process (still multi-threaded via numpy/OpenBLAS).
        The main training speed is dominated by tree-building which is CPU-bound in C,
        so n_jobs=1 per-candidate is acceptable when candidates run sequentially.
        """
        import threading as _threading
        _exc = [None]

        # Snapshot original n_jobs and temporarily set to 1 to avoid joblib
        # multiprocessing issues when called from a background thread on Windows.
        _orig_n_jobs = getattr(model, "n_jobs", 1)
        try:
            model.set_params(n_jobs=1)
        except Exception:
            pass

        def _worker():
            try:
                model.fit(X_train, y_train)
            except Exception as _e:
                _exc[0] = _e

        _t = _threading.Thread(target=_worker, daemon=True)
        _t.start()

        _dialog = getattr(self, "_tuning_progress_dialog", None)
        _interrupted = False
        while _t.is_alive():
            if _dialog is not None:
                try:
                    # update() (en lugar de update_idletasks) mantiene la UI responsiva:
                    # el usuario puede ver gráficas, mover ventanas y cambiar parámetros.
                    # La re-entrada al loop de tuning está bloqueada por _auto_tuning_in_progress.
                    _dialog.update()
                except tk.TclError:
                    self._tuning_cancel_requested = True
                    _interrupted = True
                    break
            else:
                # Sin diálogo de tuning abierto (p.ej. carga manual de modelo):
                # bombear la ventana principal para no congelar la UI.
                try:
                    self.update_idletasks()
                except Exception:
                    pass
            if bool(getattr(self, "_tuning_cancel_requested", False)):
                _interrupted = True
                break
            _t.join(timeout=0.1)

        if _t.is_alive():
            # Do not block forever here. If fit thread is stuck, returning control
            # allows the caller to abort autotuning instead of hanging in a cycle.
            if _interrupted:
                raise InterruptedError("Cancelado por el usuario.")
            raise RuntimeError("Entrenamiento RSF sin respuesta: abortado para evitar ciclo.")

        _t.join()  # thread already finished; this returns immediately

        # Restore original n_jobs
        try:
            model.set_params(n_jobs=_orig_n_jobs)
        except Exception:
            pass

        if _exc[0] is not None:
            raise _exc[0]

    def _close_auto_tuning_progress_dialog(self):
        dialog = getattr(self, "_tuning_progress_dialog", None)
        if dialog is not None:
            try:
                if dialog.winfo_exists():
                    dialog.destroy()
            except Exception:
                pass
        self._tuning_progress_dialog = None
        self._tuning_progress_var = None
        self._tuning_progress_note_var = None
        self._tuning_progress_bar = None
        self._tuning_cancel_button = None
        self._tuning_pause_button = None
        self._tuning_skip_button = None
        self._tuning_zoom_button = None
        self._tuning_current_model_var = None
        self._tuning_best_metric_var = None
        self._tuning_best_row_var = None
        self._tuning_autoscroll_var = None
        self._tuning_history_text = None
        self._tuning_history_meta_var = None
        self._tuning_pause_requested = False
        self._tuning_skip_scope_requested = False
        self._close_live_scatter_popout()
        # Live scatter cleanup
        live_fig = getattr(self, "_live_scatter_fig", None)
        if live_fig is not None:
            try:
                plt.close(live_fig)
            except Exception:
                pass
        self._live_scatter_fig = None
        self._live_scatter_canvas = None
        self._live_scatter_vars = None
        self._live_scatter_filter_var = None
        self._live_scatter_filter_cb = None
        self._live_scatter_filter_values = []
        self._live_scatter_point_size_var = None
        self._live_scatter_marker_mode_var = None
        self._live_scatter_heatmap_var = None
        self._live_scatter_heat_bins_var = None
        self._live_scatter_show_apt_var = None
        self._live_scatter_apply_gap_filters_var = None
        self._live_scatter_use_global_gap_var = None
        self._live_scatter_gap_global_var = None
        self._live_scatter_gap_oob_cv_var = None
        self._live_scatter_gap_oob_test_var = None
        self._live_scatter_gap_cv_test_var = None
        self._live_scatter_show_ids_var = None
        self._live_scatter_show_ci_var = None
        self._live_scatter_advanced_filter_fields = []
        self._live_scatter_advanced_filters = {}
        self._live_scatter_all_records = []
        self._live_scatter_pulse_seen_uids = set()
        self._live_scatter_info_var = None
        self._live_scatter_data = {}
        self._live_tuning_results = None
        self._live_scatter_last_refresh_completed = -1
        # VIMP chart cleanup
        vimp_fig2 = getattr(self, "_autotune_vimp_fig", None)
        if vimp_fig2 is not None:
            try:
                plt.close(vimp_fig2)
            except Exception:
                pass
        self._autotune_vimp_fig = None
        self._autotune_vimp_canvas = None
        self._autotune_vimp_records = []
        self._autotune_metrics_records = []
        self._autotune_vimp_ci_var = None
        self._autotune_vimp_topn_var = None
        self._autotune_vimp_info_var = None
        self._autotune_screen_x_var = None
        self._autotune_screen_y_var = None
        self._autotune_screen_ci_var = None
        self._autotune_screen_pt_size_var = None
        self._autotune_screen_show_seed_label_var = None
        self._autotune_screen_show_vals_var = None
        self._autotune_current_seed_metrics = []
        self._autotune_current_vimp = {}
        self._autotune_current_candidate_n = 0
        self._autotune_current_trees_done = 0
        self._autotune_current_trees_total = 0
        self._autotune_current_candidate_params = {}

    def _resolve_tau(self, y_train, y_test=None):
        """Backward-compatible tau resolver for code paths that still call _resolve_tau."""
        return self._resolve_tau_for_uno(y_train, y_test=y_test, raise_on_error=False)

    def _build_evaluation_time_grid(self, y_train, y_test, tau=None):
        try:
            train_times = np.asarray(y_train["time"], dtype=float)
            test_times = np.asarray(y_test["time"], dtype=float)
        except Exception:
            return None

        train_times = train_times[np.isfinite(train_times)]
        test_times = test_times[np.isfinite(test_times)]
        if train_times.size < 5 or test_times.size < 3:
            return None

        lower_bound = max(float(np.nanpercentile(train_times, 10)), float(np.nanmin(test_times)))
        upper_bound = min(float(np.nanpercentile(train_times, 90)), float(np.nanmax(test_times)))
        if tau is not None and np.isfinite(tau) and tau > 0:
            upper_bound = min(upper_bound, float(tau))
        if not np.isfinite(lower_bound) or not np.isfinite(upper_bound) or lower_bound >= upper_bound:
            return None

        eval_times = np.unique(np.linspace(lower_bound, upper_bound, num=12))
        eval_times = eval_times[np.isfinite(eval_times)]
        return eval_times if eval_times.size >= 2 else None

    def _build_fallback_auc_time_grid(self, y_train, y_test, tau=None):
        """Build a more permissive time grid for cumulative_dynamic_auc when the default grid fails."""
        try:
            train_times = np.asarray(y_train["time"], dtype=float)
            test_times = np.asarray(y_test["time"], dtype=float)
            test_events = np.asarray(y_test["event"], dtype=bool)
        except Exception:
            return None

        train_times = train_times[np.isfinite(train_times)]
        test_times = test_times[np.isfinite(test_times)]
        if train_times.size < 3 or test_times.size < 2:
            return None

        event_times = test_times[test_events[:len(test_times)]] if test_events.size == test_times.size else np.array([], dtype=float)
        event_times = event_times[np.isfinite(event_times)]
        candidate_source = event_times if event_times.size >= 2 else test_times

        lower_bound = max(float(np.nanmin(train_times)), float(np.nanmin(candidate_source)))
        upper_bound = min(float(np.nanmax(train_times)), float(np.nanmax(candidate_source)))
        if tau is not None and np.isfinite(tau) and tau > 0:
            upper_bound = min(upper_bound, float(tau))
        if not np.isfinite(lower_bound) or not np.isfinite(upper_bound) or lower_bound >= upper_bound:
            return None

        if candidate_source.size >= 4:
            grid = np.unique(np.percentile(candidate_source, [25, 50, 75]))
        else:
            grid = np.unique(np.linspace(lower_bound, upper_bound, num=min(5, max(2, candidate_source.size + 1))))
        grid = np.asarray([t for t in grid if np.isfinite(t) and lower_bound <= float(t) <= upper_bound], dtype=float)
        return grid if grid.size >= 1 else None

    def _compute_c_antolini_score(self, model, X_train, y_train, X_test, y_test, eval_times=None, tau=None):
        """Robust computation of time-dependent AUC mean used as C-Antolini proxy."""
        if len(X_test) == 0 or not callable(cumulative_dynamic_auc):
            return None

        candidate_grids = []
        if eval_times is not None:
            try:
                eval_arr = np.asarray(eval_times, dtype=float)
                eval_arr = eval_arr[np.isfinite(eval_arr)]
                if eval_arr.size >= 1:
                    candidate_grids.append(np.unique(eval_arr))
            except Exception:
                pass

        fallback_grid = self._build_fallback_auc_time_grid(y_train, y_test, tau=tau)
        if fallback_grid is not None and fallback_grid.size >= 1:
            candidate_grids.append(np.unique(fallback_grid))

        if not candidate_grids:
            return None

        for time_grid in candidate_grids:
            try:
                surv_fns = model.predict_survival_function(X_test)
                risk_matrix = 1.0 - np.asarray([fn(time_grid) for fn in surv_fns], dtype=float)
                _, mean_auc = cumulative_dynamic_auc(y_train, y_test, risk_matrix, time_grid)
                return float(np.asarray([mean_auc], dtype=float).reshape(-1)[0])
            except Exception:
                pass

            try:
                test_preds = model.predict(X_test)
                _, mean_auc = cumulative_dynamic_auc(y_train, y_test, test_preds, time_grid)
                return float(np.asarray([mean_auc], dtype=float).reshape(-1)[0])
            except Exception:
                pass

        return None

    def _get_confidence_z_value(self, confidence_level=0.95):
        try:
            level = float(confidence_level)
        except (TypeError, ValueError):
            level = 0.95
        level = float(np.clip(level, 0.50, 0.999))
        return float(NormalDist().inv_cdf(0.5 + (level / 2.0)))

    def _compute_mean_confidence_interval(self, values, confidence_level=0.95, clip_min=None, clip_max=None):
        arr = np.asarray(values, dtype=float).reshape(-1)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return np.nan, np.nan

        center = float(np.nanmean(arr))
        if arr.size == 1:
            lower = upper = center
        else:
            stderr = float(np.nanstd(arr, ddof=1) / np.sqrt(arr.size))
            margin = self._get_confidence_z_value(confidence_level) * stderr if np.isfinite(stderr) else 0.0
            lower = center - margin
            upper = center + margin

        if clip_min is not None:
            lower = max(lower, clip_min)
            upper = max(upper, clip_min)
        if clip_max is not None:
            lower = min(lower, clip_max)
            upper = min(upper, clip_max)
        if lower > upper:
            lower, upper = upper, lower
        return float(lower), float(upper)

    def _build_asymmetric_errorbars(self, center_values, lower_values, upper_values):
        if center_values is None or lower_values is None or upper_values is None:
            return None

        center = np.asarray(center_values, dtype=float)
        lower = np.asarray(lower_values, dtype=float)
        upper = np.asarray(upper_values, dtype=float)
        if center.shape != lower.shape or center.shape != upper.shape:
            return None

        if center.ndim == 0:
            center = center.reshape(1)
            lower = lower.reshape(1)
            upper = upper.reshape(1)

        lower_err = np.clip(center - lower, 0.0, None)
        upper_err = np.clip(upper - center, 0.0, None)
        combined = np.vstack([lower_err, upper_err])
        return combined if np.isfinite(combined).any() else None

    def _evaluate_step_function(self, step_fn, eval_times):
        eval_times = np.asarray(eval_times, dtype=float)
        if eval_times.size == 0:
            return np.asarray([], dtype=float)
        if step_fn is None:
            return np.full(eval_times.shape, np.nan, dtype=float)

        try:
            values = np.asarray(step_fn(eval_times), dtype=float)
            if values.shape == eval_times.shape:
                return values
        except Exception:
            pass

        x_values = np.asarray(getattr(step_fn, "x", []), dtype=float)
        y_values = np.asarray(getattr(step_fn, "y", []), dtype=float)
        if x_values.size == 0 or y_values.size == 0:
            return np.full(eval_times.shape, np.nan, dtype=float)

        indices = np.searchsorted(x_values, eval_times, side="right") - 1
        indices = np.clip(indices, 0, y_values.size - 1)
        return np.asarray(y_values[indices], dtype=float)

    def _compute_survival_curve_confidence_bands(self, model, predict_encoded, confidence_level=0.95, max_estimators=80):
        if model is None or predict_encoded is None or len(predict_encoded) == 0:
            return None

        try:
            survival_functions = list(model.predict_survival_function(predict_encoded))
        except Exception:
            return None

        if not survival_functions:
            return None

        time_grid = np.asarray(getattr(survival_functions[0], "x", []), dtype=float)
        if time_grid.size == 0:
            return None

        point_matrix = np.asarray(
            [self._evaluate_step_function(step_fn, time_grid) for step_fn in survival_functions],
            dtype=float,
        )
        point_matrix = np.clip(point_matrix, 0.0, 1.0)

        estimators = list(getattr(model, "estimators_", []) or [])
        if max_estimators is not None and len(estimators) > max_estimators:
            sampled_idx = np.linspace(0, len(estimators) - 1, num=max_estimators, dtype=int)
            estimators = [estimators[idx] for idx in np.unique(sampled_idx)]

        tree_curves = []
        for estimator in estimators:
            try:
                estimator_functions = list(estimator.predict_survival_function(predict_encoded))
            except Exception:
                continue
            if len(estimator_functions) != point_matrix.shape[0]:
                continue
            estimator_matrix = np.asarray(
                [self._evaluate_step_function(step_fn, time_grid) for step_fn in estimator_functions],
                dtype=float,
            )
            if estimator_matrix.shape == point_matrix.shape:
                tree_curves.append(estimator_matrix)

        if len(tree_curves) >= 2:
            tree_array = np.clip(np.asarray(tree_curves, dtype=float), 0.0, 1.0)
            alpha = float(np.clip(1.0 - float(confidence_level), 0.001, 0.50))
            lower = np.nanquantile(tree_array, alpha / 2.0, axis=0)
            upper = np.nanquantile(tree_array, 1.0 - (alpha / 2.0), axis=0)
            lower = np.clip(np.minimum(lower, point_matrix), 0.0, 1.0)
            upper = np.clip(np.maximum(upper, point_matrix), 0.0, 1.0)
            ci_available = True
        else:
            lower = point_matrix.copy()
            upper = point_matrix.copy()
            ci_available = False

        return {
            "times": time_grid,
            "point_estimate": point_matrix,
            "lower": lower,
            "upper": upper,
            "ci_available": ci_available,
            "n_estimators_used": len(tree_curves),
            "confidence_level": float(confidence_level),
        }

    def _extract_curve_values_at_time(self, curve_payload, eval_time, output_type="survival"):
        if not isinstance(curve_payload, dict):
            return None

        times = np.asarray(curve_payload.get("times", []), dtype=float)
        point_matrix = np.asarray(curve_payload.get("point_estimate", []), dtype=float)
        if times.size == 0 or point_matrix.size == 0:
            return None

        try:
            requested_time = float(eval_time)
        except (TypeError, ValueError):
            requested_time = np.nan
        if not np.isfinite(requested_time):
            requested_time = float(times[len(times) // 2])
        eval_time_clipped = float(np.clip(requested_time, times[0], times[-1]))

        point_values = np.asarray([np.interp(eval_time_clipped, times, row) for row in point_matrix], dtype=float)
        lower_matrix = curve_payload.get("lower")
        upper_matrix = curve_payload.get("upper")
        ci_available = bool(curve_payload.get("ci_available") and lower_matrix is not None and upper_matrix is not None)

        lower_values = None
        upper_values = None
        if ci_available:
            lower_matrix = np.asarray(lower_matrix, dtype=float)
            upper_matrix = np.asarray(upper_matrix, dtype=float)
            lower_values = np.asarray([np.interp(eval_time_clipped, times, row) for row in lower_matrix], dtype=float)
            upper_values = np.asarray([np.interp(eval_time_clipped, times, row) for row in upper_matrix], dtype=float)

        metric_key = str(output_type).strip().lower()
        if metric_key in {"risk", "1-s", "1 - s", "evento", "event"}:
            point_values = np.clip(1.0 - point_values, 0.0, 1.0)
            if ci_available and lower_values is not None and upper_values is not None:
                risk_lower = 1.0 - upper_values
                risk_upper = 1.0 - lower_values
                lower_values = np.clip(risk_lower, 0.0, 1.0)
                upper_values = np.clip(risk_upper, 0.0, 1.0)
            output_type = "risk"
        else:
            point_values = np.clip(point_values, 0.0, 1.0)
            if ci_available and lower_values is not None and upper_values is not None:
                lower_values = np.clip(lower_values, 0.0, 1.0)
                upper_values = np.clip(upper_values, 0.0, 1.0)
            output_type = "survival"

        return {
            "eval_time": eval_time_clipped,
            "point_estimate": point_values,
            "lower": lower_values,
            "upper": upper_values,
            "ci_available": ci_available,
            "output_type": output_type,
        }

    def _extract_km_confidence_limits(self, kmf, eval_time):
        if kmf is None:
            return np.nan, np.nan

        try:
            ci_df = getattr(kmf, "confidence_interval_survival_function_", None)
            if ci_df is None or ci_df.empty:
                ci_df = getattr(kmf, "confidence_interval_", None)
            if ci_df is None or ci_df.empty:
                return np.nan, np.nan

            lower_col = next((col for col in ci_df.columns if "lower" in str(col).lower()), ci_df.columns[0])
            upper_col = next((col for col in ci_df.columns if "upper" in str(col).lower()), ci_df.columns[-1])
            ci_times = np.asarray(ci_df.index, dtype=float)
            if ci_times.size == 0:
                return np.nan, np.nan

            eval_time_clipped = float(np.clip(float(eval_time), ci_times[0], ci_times[-1]))
            lower_values = np.asarray(ci_df[lower_col], dtype=float)
            upper_values = np.asarray(ci_df[upper_col], dtype=float)
            idx = int(np.searchsorted(ci_times, eval_time_clipped, side="right") - 1)
            idx = max(0, min(idx, len(ci_times) - 1))
            lower = float(lower_values[idx])
            upper = float(upper_values[idx])
            return float(np.clip(lower, 0.0, 1.0)), float(np.clip(upper, 0.0, 1.0))
        except Exception:
            return np.nan, np.nan

    def _summarize_calibration(self, y_test, predicted_survival, eval_time):
        if predicted_survival is None:
            return pd.DataFrame()

        try:
            calibration_df = pd.DataFrame(
                {
                    "predicted_survival": np.asarray(predicted_survival, dtype=float),
                    "time": np.asarray(y_test["time"], dtype=float),
                    "event": np.asarray(y_test["event"], dtype=bool),
                }
            )
        except Exception:
            return pd.DataFrame()

        calibration_df = calibration_df.replace([np.inf, -np.inf], np.nan).dropna()
        if calibration_df.empty:
            return pd.DataFrame()

        try:
            n_groups = min(4, max(2, calibration_df["predicted_survival"].nunique()))
            group_labels = [f"Q{i + 1}" for i in range(n_groups)]
            calibration_df["group"] = pd.qcut(
                calibration_df["predicted_survival"].rank(method="first"),
                q=n_groups,
                labels=group_labels,
                duplicates="drop",
            )
        except Exception:
            calibration_df["group"] = "Grupo único"

        kmf = KaplanMeierFitter()
        rows = []
        for group_name, group_df in calibration_df.groupby("group", dropna=False, observed=False):
            if group_df.empty:
                continue

            eval_time_scalar = float(np.asarray([eval_time], dtype=float)[0])
            predicted_center = float(group_df["predicted_survival"].mean())
            predicted_lower, predicted_upper = self._compute_mean_confidence_interval(
                group_df["predicted_survival"],
                confidence_level=0.95,
                clip_min=0.0,
                clip_max=1.0,
            )

            observed_survival = np.nan
            observed_lower = np.nan
            observed_upper = np.nan
            try:
                kmf.fit(group_df["time"], event_observed=group_df["event"], label=str(group_name))
                observed_survival = float(np.asarray(kmf.predict(eval_time_scalar)).reshape(-1)[0])
                observed_lower, observed_upper = self._extract_km_confidence_limits(kmf, eval_time_scalar)
            except Exception:
                observed_survival = np.nan
                observed_lower = np.nan
                observed_upper = np.nan

            rows.append(
                {
                    "group": str(group_name),
                    "predicted_survival": predicted_center,
                    "predicted_survival_lower": predicted_lower,
                    "predicted_survival_upper": predicted_upper,
                    "observed_survival": observed_survival,
                    "observed_survival_lower": observed_lower,
                    "observed_survival_upper": observed_upper,
                    "n": int(len(group_df)),
                }
            )

        return pd.DataFrame(rows)

    def _compute_permutation_importance(self, model, X_test, y_test, random_state):
        try:
            n_repeats = 10
            importance = permutation_importance(
                model,
                X_test,
                y_test,
                n_repeats=n_repeats,
                random_state=random_state,
                n_jobs=1,
            )
            importances_mean = getattr(importance, "importances_mean", None)
            importances_std = getattr(importance, "importances_std", None)
            if importances_mean is None or importances_std is None:
                return pd.DataFrame(columns=["feature", "importance", "importance_std", "importance_lower", "importance_upper"])

            std_err = np.asarray(importances_std, dtype=float) / np.sqrt(max(n_repeats, 1))
            ci_margin = self._get_confidence_z_value(0.95) * std_err
            importance_df = pd.DataFrame(
                {
                    "feature": X_test.columns,
                    "importance": importances_mean,
                    "importance_std": importances_std,
                    "importance_lower": np.asarray(importances_mean, dtype=float) - ci_margin,
                    "importance_upper": np.asarray(importances_mean, dtype=float) + ci_margin,
                }
            ).sort_values("importance", ascending=False)
            return importance_df.reset_index(drop=True)
        except Exception:
            return pd.DataFrame(columns=["feature", "importance", "importance_std", "importance_lower", "importance_upper"])

    def _build_survival_profiles(self, model, X_encoded, risk_scores):
        profiles = []
        if X_encoded is None or X_encoded.empty:
            return profiles

        try:
            ordered_positions = np.argsort(np.asarray(risk_scores, dtype=float))
            chosen_positions = sorted(set([int(ordered_positions[0]), int(ordered_positions[len(ordered_positions) // 2]), int(ordered_positions[-1])]))
            profile_labels = ["Riesgo bajo", "Riesgo medio", "Riesgo alto"]
            selected_rows = X_encoded.iloc[chosen_positions]
            survival_functions = model.predict_survival_function(selected_rows)

            for idx, surv_fn in enumerate(survival_functions):
                label = profile_labels[min(idx, len(profile_labels) - 1)]
                profiles.append(
                    {
                        "label": label,
                        "x": np.asarray(surv_fn.x, dtype=float),
                        "y": np.asarray(surv_fn.y, dtype=float),
                    }
                )
        except Exception:
            return []

        return profiles

    def show_correlation_diagnostics(self):
        """Show Pearson/Spearman correlation matrices for the currently selected RSF predictors."""
        if self.data is None:
            messagebox.showerror("Correlaciones RSF", "Cargue datos primero.")
            return

        filtered_data = self.filter_component.apply_filters()
        if filtered_data is None or filtered_data.empty:
            messagebox.showerror("Correlaciones RSF", "No hay datos disponibles tras aplicar filtros.")
            return

        duration_col = self.duration_var.get().strip()
        event_col = self.event_var.get().strip()
        selected_indices = self.covariates_listbox.curselection()
        covariates = [self.covariates_listbox.get(i) for i in selected_indices]
        if not duration_col or not event_col or not covariates:
            messagebox.showerror("Correlaciones RSF", "Seleccione tiempo, evento y al menos una covariable.")
            return

        # Solo variables cuantitativas — respeta la configuración del usuario
        num_covariates = [c for c in covariates if not self._is_covariate_categorical(c)]
        excluded = [c for c in covariates if c not in num_covariates]

        if len(num_covariates) < 2:
            messagebox.showinfo(
                "Correlaciones RSF",
                "Se necesitan al menos 2 variables cuantitativas para calcular correlaciones.\n"
                + (f"Excluidas (no numéricas): {', '.join(excluded)}" if excluded else ""),
            )
            return

        try:
            X_corr = filtered_data[num_covariates].apply(pd.to_numeric, errors="coerce").dropna()
            pearson_corr = X_corr.corr(method="pearson")
            spearman_corr = X_corr.corr(method="spearman")
        except Exception as exc:
            messagebox.showerror("Correlaciones RSF", f"No se pudieron calcular las matrices de correlación:\n{exc}")
            return

        warnings_list = []
        if excluded:
            warnings_list.append(f"Variables no numéricas omitidas: {', '.join(excluded)}")

        popup = tk.Toplevel(self.winfo_toplevel())
        popup.title("Correlaciones de Variables — RSF")
        popup.geometry("860x620")
        popup.transient(self.winfo_toplevel())
        popup.grab_set()

        header = ttk.Frame(popup, padding=(12, 10, 12, 4))
        header.pack(fill=tk.X)
        summary_lines = [
            f"Variables cuantitativas: {len(num_covariates)} | filas útiles: {len(X_corr)}",
            "Las correlaciones se calculan solo sobre variables numéricas originales.",
        ]
        if warnings_list:
            summary_lines.append("Avisos: " + " | ".join(str(w) for w in warnings_list[:3]))
        ttk.Label(header, text="\n".join(summary_lines), justify="left", wraplength=820).pack(anchor="w")

        top_pairs = self._extract_top_correlation_pairs(pearson_corr, threshold=0.70, max_pairs=12)
        pairs_box = scrolledtext.ScrolledText(popup, height=6, wrap=tk.WORD, font=("Consolas", 9))
        pairs_box.pack(fill=tk.X, padx=12, pady=(0, 8))
        pairs_box.insert(tk.END, "Pares con |r| alto (Pearson):\n")
        if top_pairs:
            for left, right, value in top_pairs:
                pairs_box.insert(tk.END, f"- {left} vs {right}: r={value:.3f}\n")
        else:
            pairs_box.insert(tk.END, "- No se detectaron pares con |r| >= 0.70.\n")
        pairs_box.configure(state=tk.DISABLED)

        notebook = ttk.Notebook(popup)
        notebook.pack(fill=tk.BOTH, expand=True, padx=12, pady=(0, 10))

        pearson_tab = ttk.Frame(notebook)
        notebook.add(pearson_tab, text="Correlación Pearson")
        self._populate_correlation_matrix_tab(pearson_tab, pearson_corr, "Pearson")

        spearman_tab = ttk.Frame(notebook)
        notebook.add(spearman_tab, text="Correlación Spearman")
        self._populate_correlation_matrix_tab(spearman_tab, spearman_corr, "Spearman")

        ttk.Button(popup, text="Cerrar", command=popup.destroy).pack(pady=(0, 10))

    def _extract_top_correlation_pairs(self, corr_matrix, threshold=0.70, max_pairs=12):
        if not isinstance(corr_matrix, pd.DataFrame) or corr_matrix.empty:
            return []

        pairs = []
        cols = list(corr_matrix.columns)
        for i, left in enumerate(cols):
            for j in range(i + 1, len(cols)):
                right = cols[j]
                try:
                    val = float(corr_matrix.iloc[i, j])
                except Exception:
                    continue
                if np.isfinite(val) and abs(val) >= float(threshold):
                    pairs.append((left, right, val))
        pairs.sort(key=lambda item: abs(item[2]), reverse=True)
        return pairs[:max_pairs]

    def _populate_correlation_matrix_tab(self, tab, corr_matrix, corr_type):
        """Create and populate a correlation-matrix Treeview for RSF."""
        label = ttk.Label(
            tab,
            text=f"Matriz de correlación de {corr_type}. Filas con alguna |r| > 0.70 se resaltan.",
            wraplength=760,
            justify="left",
        )
        label.pack(pady=10, padx=10, anchor="w")

        tree_frame = ttk.Frame(tab)
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        columns = ["Variable"] + list(corr_matrix.columns)
        tree = ttk.Treeview(tree_frame, columns=columns, show="headings")
        tree.heading("Variable", text="Variable")
        tree.column("Variable", width=180, anchor="w", stretch=False)
        for col in corr_matrix.columns:
            tree.heading(col, text=col)
            tree.column(col, width=90, anchor="e", stretch=True)

        tree.tag_configure("high_corr", background="salmon")
        tree.tag_configure("perfect_corr", background="lightgrey")

        for index, row in corr_matrix.iterrows():
            values = [index] + [f"{float(val):.3f}" if pd.notna(val) else "-" for val in row]
            has_high_corr = any(abs(float(val)) > 0.70 and abs(float(val)) < 0.999999 for val in row if pd.notna(val))
            has_only_diag = all((not pd.notna(val)) or abs(float(val)) >= 0.999999 or abs(float(val)) < 0.70 for val in row)
            tags = ("high_corr",) if has_high_corr else (("perfect_corr",) if has_only_diag else ())
            tree.insert("", "end", values=values, tags=tags)

        ysb = ttk.Scrollbar(tree_frame, orient="vertical", command=tree.yview)
        xsb = ttk.Scrollbar(tree_frame, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=ysb.set, xscrollcommand=xsb.set)
        ysb.pack(side="right", fill="y")
        xsb.pack(side="bottom", fill="x")
        tree.pack(fill=tk.BOTH, expand=True)

    def _build_results_report(self, data, rsf_params, metrics, warnings_list, test_size=None, model=None):
        lines = []
        cv_metric_label = self._resolve_cv_metric_choice()[1]
        latest_duration_col = getattr(self, "latest_duration_col", None)
        latest_event_col = getattr(self, "latest_event_col", None)
        latest_covariates = list(getattr(self, "latest_covariates", []) or [])
        lines.append("=== Resumen del Random Survival Forest (RSF) ===")
        lines.append(f"Observaciones utilizadas: {len(data)}")

        if latest_event_col and latest_event_col in data.columns:
            try:
                events = int((pd.to_numeric(data[latest_event_col], errors='coerce').fillna(0) > 0).sum())
                censored = len(data) - events
                lines.append(f"Eventos observados: {events}")
                lines.append(f"Censurados: {censored}")
            except Exception:
                pass

        if isinstance(metrics, dict):
            n_raw = metrics.get("n_rows_filtered")
            n_clean = metrics.get("n_rows_clean")
            n_removed = metrics.get("n_rows_removed")
            n_train = metrics.get("n_train")
            n_train_events = metrics.get("n_train_events")
            n_train_censored = metrics.get("n_train_censored")
            n_test = metrics.get("n_test")
            n_test_events = metrics.get("n_test_events")
            n_test_censored = metrics.get("n_test_censored")
            removed_case_ids = metrics.get("removed_case_ids") or []

            lines.append("")
            lines.append("=== Desglose de datos usados (limpieza y partición) ===")
            if n_raw is not None:
                lines.append(f"- Casos tras filtros previos: {int(n_raw)}")
            if n_clean is not None:
                lines.append(f"- Casos válidos para modelado: {int(n_clean)}")
            if n_removed is not None:
                lines.append(f"- Casos eliminados en preparación: {int(n_removed)}")
            if removed_case_ids:
                preview = ", ".join(str(item) for item in removed_case_ids[:20])
                suffix = "" if len(removed_case_ids) <= 20 else f" ... (+{len(removed_case_ids) - 20} más)"
                lines.append(f"- Índices/IDs eliminados: {preview}{suffix}")

            if n_train is not None or n_test is not None:
                lines.append("- Partición train/test:")
                if n_train is not None:
                    lines.append(
                        f"  train={int(n_train)} | eventos={int(n_train_events or 0)} | censurados={int(n_train_censored or 0)}"
                    )
                if n_test is not None:
                    lines.append(
                        f"  test={int(n_test)} | eventos={int(n_test_events or 0)} | censurados={int(n_test_censored or 0)}"
                    )

        if latest_duration_col or latest_event_col or latest_covariates:
            lines.append("")
            lines.append("=== Variables del modelo ===")
            if latest_duration_col:
                lines.append(f"- Variable de tiempo: {latest_duration_col}")
            if latest_event_col:
                lines.append(f"- Variable de evento: {latest_event_col}")
            if latest_covariates:
                lines.append(f"- Covariables ({len(latest_covariates)}):")
                for cov in latest_covariates:
                    lines.append(f"  - {cov}")

        lines.append("")
        lines.append("=== Parámetros del modelo y validación ===")
        for key, value in rsf_params.items():
            display_val = self._quote_report_value(value)
            if key == "max_features":
                display_val = self._annotate_max_features_text(value)
            lines.append(f"- {key}: {display_val}")

        if model is not None and hasattr(model, "estimators_") and model.estimators_:
            import numpy as np
            depths = [est.tree_.max_depth for est in model.estimators_ if hasattr(est, "tree_")]
            if depths:
                mean_depth = float(np.mean(depths))
                max_depth = int(np.max(depths))
                lines.append(f"- profundidad de los árboles (depth): media={mean_depth:.1f}, máx={max_depth}")

        # Mostrar siempre estos hiperparámetros opcionales para poder replicar la configuración.
        optional_ui_fields = [
            ("max_depth", getattr(self, "max_depth_var", None)),
            ("max_leaf_nodes", getattr(self, "max_leaf_nodes_var", None)),
            ("max_samples", getattr(self, "max_samples_var", None)),
        ]
        for field_name, ui_var in optional_ui_fields:
            if field_name in rsf_params:
                continue
            raw_text = ""
            try:
                raw_text = ui_var.get() if ui_var is not None else ""
            except Exception:
                raw_text = ""
            raw_text = str(raw_text).strip()
            lines.append(f"- {field_name}: {self._quote_report_value(raw_text if raw_text else 'none')}")

        if hasattr(self, 'missing_strategy_var'):
            lines.append(f"- missing_strategy: {self._quote_report_value(self.missing_strategy_var.get())}")
        if hasattr(self, 'drop_first_var'):
            lines.append(f"- drop_first: {self._quote_report_value(bool(self.drop_first_var.get()))}")
        if hasattr(self, 'stratify_event_var'):
            lines.append(f"- stratify_event: {self._quote_report_value(bool(self.stratify_event_var.get()))}")
        if hasattr(self, 'cv_enabled_var'):
            lines.append(f"- cv_enabled: {self._quote_report_value(bool(self.cv_enabled_var.get()))}")
        if hasattr(self, 'cv_folds_var'):
            lines.append(f"- cv_folds: {self._quote_report_value(self.cv_folds_var.get())}")
        if test_size is not None:
            if float(test_size) <= 0:
                lines.append("- test_size: 0.00 (sin holdout; todo el dataset se usa para entrenamiento)")
            else:
                lines.append(f"- test_size: {float(test_size):.2f}")

        lines.append("")
        lines.append("=== Desempeño e interpretación ===")
        lines.append(
            f"- C-index entrenamiento = {self._format_c_index_display(metrics.get('c_index_train'), metrics.get('c_index_train_ci'), decimals=4)}"
        )
        lines.append(
            f"- C-index prueba = {self._format_c_index_display(metrics.get('c_index_test'), metrics.get('c_index_test_ci'), decimals=4)}"
        )
        lines.append(
            f"- C-index Uno IPCW = {self._format_c_index_display(metrics.get('c_index_uno'), metrics.get('c_index_uno_ci'), decimals=4)} "
            f"(IPCW = Inverse Probability of Censoring Weighting; mayor es mejor y corrige por censura)."
        )
        lines.append(
            f"- C-index Antolini (Ctd) = {self._format_c_index_display(metrics.get('c_index_antolini'), metrics.get('c_index_antolini_ci'), decimals=4)} "
            f"(AUC dinámica media; generaliza el C-index para predicciones dependientes del tiempo)."
        )
        tau_val = metrics.get('tau')
        if tau_val is not None and np.isfinite(tau_val):
            lines.append(f"- τ (tau) utilizado = {tau_val:.2f} (truncamiento IPCW para Uno, Antolini y Brier).")
        lines.append(
            f"- {cv_metric_label} CV media = {self._format_c_index_display(metrics.get('c_index_cv_mean'), metrics.get('c_index_cv_ci'), decimals=4)}"
        )
        lines.append(
            f"- C-index CV DE = {self._format_metric(metrics.get('c_index_cv_std'))} "
            f"(DE = desviación estándar entre folds; menor indica más estabilidad)."
        )
        lines.append(
            f"- OOB score = {self._format_metric(metrics.get('oob_score'))} "
            f"(desempeño out-of-bag interno; mayor es mejor)."
        )
        lines.append(
            f"- Integrated Brier Score (IBS) = {self._format_c_index_display(metrics.get('ibs'), metrics.get('ibs_ci'), decimals=4)} "
            f"(promedio del error de predicción a lo largo del tiempo; menor es mejor)."
        )
        lines.append(
            f"- IBS Kaplan-Meier (nulo) = {self._format_metric(metrics.get('ibs_km'))} "
            f"(IBS del modelo nulo KM; referencia para el BSS)."
        )
        bss_val = metrics.get('bss')
        lines.append(
            f"- Brier Skill Score (BSS) = {self._format_c_index_display(bss_val, metrics.get('bss_ci'), decimals=4)} "
            f"(1 - IBS_modelo/IBS_KM; positivo = mejor que KM nulo)."
        )
        if bss_val is not None:
            if bss_val > 0.1:
                lines.append("  → Excelente: el modelo supera significativamente al KM nulo.")
            elif bss_val > 0:
                lines.append("  → El modelo supera al KM nulo.")
            elif bss_val > -0.05:
                lines.append("  → El modelo es similar al KM nulo.")
            else:
                lines.append("  → El modelo es peor que el KM nulo. Revisar calibración.")
        brier_time = metrics.get('brier_eval_time')
        if brier_time is not None:
            lines.append(
                f"- Brier score en t={float(brier_time):.3g} = {self._format_metric(metrics.get('brier_at_eval_time'))} "
                f"(error puntual de supervivencia en ese tiempo; menor es mejor)."
            )

        importance_df = pd.DataFrame()
        if isinstance(self.feature_importance_df, pd.DataFrame) and not self.feature_importance_df.empty and "importance" in self.feature_importance_df.columns:
            importance_df = self.feature_importance_df.sort_values("importance", ascending=False).copy()
        if not importance_df.empty:
            lines.append("")
            lines.append("=== Importancia de variables (VIMP = Variable Importance por permutación) ===")
            lines.append("Valores positivos más altos indican mayor influencia; valores cercanos a 0 o negativos sugieren poca o nula influencia.")
            for _, row in importance_df.iterrows():
                feature_name = row.get('feature', 'variable')
                importance_value = self._format_metric(row.get('importance'))
                std_text = self._format_metric(row.get('importance_std')) if 'importance_std' in row else 'N/D'
                lower_text = self._format_metric(row.get('importance_lower')) if 'importance_lower' in row else 'N/D'
                upper_text = self._format_metric(row.get('importance_upper')) if 'importance_upper' in row else 'N/D'
                lines.append(
                    f"- {feature_name}: VIMP={importance_value} | DE={std_text} | IC95%=({lower_text}, {upper_text})"
                )

        if warnings_list:
            lines.append("")
            lines.append("=== Avisos de preparación ===")
            for item in warnings_list:
                lines.append(f"- {item}")

        if self.latest_covariates:
            lines.append("")
            lines.append("=== Covariables incluidas ===")
            lines.append(", ".join(self.latest_covariates))

        return "\n".join(lines)

    def run_model(self, tuning_summary=None, store_snapshot=True, params_override=None, covariates_override=None):
        if not SKSURV_AVAILABLE:
            messagebox.showerror(
                "RSF no disponible",
                "No se pudo importar `scikit-survival`. Instálalo para entrenar la pestaña RSF."
                f"\n\nDetalle: {SKSURV_IMPORT_ERROR}",
            )
            return

        if self.data is None:
            messagebox.showerror("Error", "Cargue datos o use el dataset compartido primero.")
            return

        filtered_data = self.filter_component.apply_filters()
        if filtered_data is None or filtered_data.empty:
            messagebox.showerror("Error", "No hay datos disponibles tras aplicar filtros.")
            return

        duration_col = self.duration_var.get().strip()
        event_col = self.event_var.get().strip()
        if isinstance(covariates_override, (list, tuple)) and covariates_override:
            covariates = [str(c).strip() for c in covariates_override if str(c).strip()]
        else:
            selected_indices = self.covariates_listbox.curselection()
            covariates = [self.covariates_listbox.get(i) for i in selected_indices]

        if not duration_col or not event_col or not covariates:
            messagebox.showerror("Error", "Seleccione tiempo, evento y al menos una covariable para RSF.")
            return

        try:
            clean_data, X_encoded, y_structured, warnings_list = self._prepare_dataframe_for_rsf(
                filtered_data, duration_col, event_col, covariates
            )

            self.latest_fit_dataframe = clean_data.copy()
            self.latest_duration_col = duration_col
            self.latest_event_col = event_col
            self.latest_covariates = list(covariates)
            self.latest_encoded_columns = list(X_encoded.columns)
            self.latest_drop_first = bool(self.drop_first_var.get())
            self._sync_plot_covariate_selectors()

            requested_test_size = self._coerce_float(self.test_size_var.get(), 0.25, minimum=0.0, maximum=0.95)
            # If reconstructing from a saved snapshot that recorded seeds, reuse them
            _saved_seeds = None
            if isinstance(params_override, dict):
                _sv = params_override.get("seeds_used")
                if isinstance(_sv, (list, tuple)) and len(_sv) > 0:
                    try:
                        _saved_seeds = [int(s) for s in _sv]
                    except Exception:
                        _saved_seeds = None
            _all_seeds = _saved_seeds if _saved_seeds else self._parse_random_seeds()
            random_state = _all_seeds[0]
            event_values = (pd.to_numeric(clean_data[event_col], errors="coerce").fillna(0) > 0).astype(int)

            test_size, stratify_values, split_warnings = self._resolve_holdout_split_settings(
                clean_data,
                event_col,
                requested_test_size,
                min_train_rows=max(8, int(X_encoded.shape[1]) + 2),
                min_test_rows=2,
                prefer_stratify=bool(self.stratify_event_var.get()),
                context_label="RSF",
            )
            warnings_list.extend(split_warnings)
            if test_size <= 0:
                X_train = X_encoded.copy()
                X_test = X_encoded.iloc[0:0].copy()
                y_train = y_structured
                y_test = y_structured[:0]
            else:
                X_train, X_test, y_train, y_test = train_test_split(
                    X_encoded,
                    y_structured,
                    test_size=test_size,
                    random_state=random_state,
                    stratify=stratify_values,
                )

            if isinstance(params_override, dict) and params_override:
                rsf_params = copy.deepcopy(params_override)
                # Eliminar claves que no son parámetros de RandomSurvivalForest
                for _non_rsf_key in (
                    "test_size", "tau_mode", "tau_manual",
                    "tuning_scope", "tuning_mode", "scope", "mode",
                    "optimization_metric", "drop_first", "stratify_event",
                    "missing_strategy", "seed_used", "seeds_used",
                ):
                    rsf_params.pop(_non_rsf_key, None)
                # Convertir max_features al tipo correcto
                _mf = rsf_params.get("max_features")
                if _mf is not None:
                    _mf_str = str(_mf).strip().lower()
                    if _mf_str in ("sqrt", "log2"):
                        rsf_params["max_features"] = _mf_str
                    elif _mf_str in ("none", "all", ""):
                        rsf_params.pop("max_features", None)
                    else:
                        try:
                            _mf_num = float(_mf)
                            rsf_params["max_features"] = int(_mf_num) if _mf_num == int(_mf_num) and _mf_num >= 1 else _mf_num
                        except (ValueError, TypeError):
                            rsf_params.pop("max_features", None)
            else:
                rsf_params = self._get_current_rsf_params(X_train.shape[1])

            validated_leaf, validated_split = self._align_split_leaf_params(
                rsf_params.get("min_samples_leaf", self.min_samples_leaf_var.get()),
                rsf_params.get("min_samples_split", self.min_samples_split_var.get()),
            )
            rsf_params["min_samples_leaf"] = validated_leaf
            rsf_params["min_samples_split"] = validated_split
            _split_cap_run = len(X_train) // 2
            if validated_split > _split_cap_run > 0:
                rsf_params["min_samples_split"] = _split_cap_run
                validated_split = _split_cap_run
            rsf_params["random_state"] = self._coerce_int(rsf_params.get("random_state", random_state), random_state)

            model = RandomSurvivalForest(**rsf_params)
            self._fit_model_with_ui_pump(model, X_train, y_train)

            self.model = model
            train_preds = model.predict(X_train)
            test_preds = model.predict(X_test) if len(X_test) > 0 else np.asarray([], dtype=float)
            full_preds = model.predict(X_encoded)

            cv_metric_code, _cv_metric_label = self._resolve_cv_metric_choice()
            c_index_train = None
            c_index_train_ci = None
            c_index_test = None
            c_index_test_ci = None
            # Always compute Harrell train/test (cheap, used for reporting)
            c_index_train = self._compute_c_index(y_train, train_preds)
            c_index_train_ci = self._compute_c_index_ci(y_train, train_preds)
            if cv_metric_code == "harrell":
                c_index_test = self._compute_c_index(y_test, test_preds) if len(X_test) > 0 else None
                c_index_test_ci = self._compute_c_index_ci(y_test, test_preds) if len(X_test) > 0 else None
            oob_score = self._compute_oob_cindex(model, y_train)
            train_event_values = (np.asarray(y_train["event"], dtype=bool).astype(int)) if len(y_train) > 0 else pd.Series(dtype=int)
            cv_result = self._compute_cv_cindex(
                X_train,
                train_event_values,
                y_train,
                rsf_params,
                return_values=True,
            )
            if isinstance(cv_result, tuple) and len(cv_result) == 3:
                c_index_cv_mean, c_index_cv_std, cv_values = cv_result
            else:
                c_index_cv_mean, c_index_cv_std = cv_result
                cv_values = []
            c_index_cv_ci = self._compute_mean_confidence_interval(cv_values, clip_min=0.0, clip_max=1.0) if cv_values else None

            c_index_uno = None
            c_index_antolini = None
            ibs_value = None
            ibs_km_value = None
            ibs_ci = None
            bss_value = None
            bss_ci = None
            brier_at_eval = None
            eval_time = None
            resolved_tau = None
            brier_q25 = None; brier_q50 = None; brier_q75 = None
            auroc_q25 = None; auroc_q50 = None; auroc_q75 = None
            c_harrell_q25 = None; c_harrell_q50 = None; c_harrell_q75 = None
            time_q25 = None; time_q50 = None; time_q75 = None
            self.latest_brier_df = pd.DataFrame()
            self.latest_calibration_df = pd.DataFrame()

            if len(X_test) > 0:
                resolved_tau = self._resolve_tau(y_train, y_test)

            eval_times = self._build_evaluation_time_grid(y_train, y_test, tau=resolved_tau) if len(X_test) > 0 else None
            
            # ─────────────────────────────────────────────────────────────────
            # C-Uno (IPCW) — solo necesita concordance_index_ipcw
            # ─────────────────────────────────────────────────────────────────
            if cv_metric_code == "uno" and len(X_test) > 0 and callable(concordance_index_ipcw):
                try:
                    ipcw_result = concordance_index_ipcw(y_train, y_test, test_preds, tau=resolved_tau)
                    c_index_uno = float(np.asarray(ipcw_result).reshape(-1)[0])
                except Exception:
                    c_index_uno = None

            # ─────────────────────────────────────────────────────────────────
            # C-Antolini (Ctd) via cumulative_dynamic_auc — solo necesita eval_times y cumulative_dynamic_auc
            # ─────────────────────────────────────────────────────────────────
            if cv_metric_code == "antolini" and len(X_test) > 0:
                c_index_antolini = self._compute_c_antolini_score(
                    model, X_train, y_train, X_test, y_test, eval_times=eval_times, tau=resolved_tau)

            # ── C-test: siempre refleja la métrica elegida ──────────────────
            if c_index_test is None and len(X_test) > 0:
                if cv_metric_code == "uno" and c_index_uno is not None:
                    c_index_test = c_index_uno
                    c_index_test_ci = None
                elif cv_metric_code == "antolini" and c_index_antolini is not None:
                    c_index_test = c_index_antolini
                    c_index_test_ci = None

            # ─────────────────────────────────────────────────────────────────
            # Brier score, IBS, calibration, quantile metrics — requieren survival_matrix
            # ─────────────────────────────────────────────────────────────────
            if len(X_test) > 0 and eval_times is not None and callable(brier_score):
                try:
                    # Filtrar test para que tiempos < max(train) — IPCW lo requiere
                    _train_t = np.asarray(y_train["time"], dtype=float)
                    _test_t = np.asarray(y_test["time"], dtype=float)
                    _max_train = float(np.nanmax(_train_t[np.isfinite(_train_t)]))
                    _valid_test = _test_t < _max_train
                    if _valid_test.sum() >= 2:
                        X_test_bs = X_test[_valid_test]
                        y_test_bs = y_test[_valid_test]
                    else:
                        X_test_bs = X_test
                        y_test_bs = y_test

                    # Ajustar eval_times al rango válido del test filtrado
                    _et = np.asarray(eval_times, dtype=float)
                    _test_bs_t = np.asarray(y_test_bs["time"], dtype=float)
                    _et_safe = _et[(_et >= float(np.nanmin(_test_bs_t))) & (_et < float(np.nanmax(_test_bs_t)))]
                    _et_safe = np.unique(_et_safe)
                    if _et_safe.size < 2:
                        _et_safe = eval_times  # fallback, dejará que falle y se atrape

                    survival_functions_test = model.predict_survival_function(X_test_bs)
                    survival_matrix = np.asarray([fn(_et_safe) for fn in survival_functions_test], dtype=float)
                    brier_result = brier_score(y_train, y_test_bs, survival_matrix, _et_safe)
                    if isinstance(brier_result, tuple) and len(brier_result) >= 2:
                        _, brier_values = brier_result
                    else:
                        raise ValueError("`brier_score` no devolvió la tupla esperada.")
                    
                    # IBS solo se calcula si integrated_brier_score está disponible
                    if callable(integrated_brier_score):
                        try:
                            ibs_raw = integrated_brier_score(y_train, y_test_bs, survival_matrix, _et_safe)
                            ibs_value = float(np.asarray([ibs_raw], dtype=float).reshape(-1)[0])
                        except Exception:
                            ibs_value = None

                    # Fallback robusto: IBS por integración trapezoidal del Brier(t)
                    if ibs_value is None:
                        try:
                            _t = np.asarray(_et_safe, dtype=float)
                            _b = np.asarray(brier_values, dtype=float)
                            if _t.size >= 2 and _b.size == _t.size and np.isfinite(_t).all() and np.isfinite(_b).all():
                                _span = float(_t[-1] - _t[0])
                                if _span > 0:
                                    ibs_value = float(np.trapz(_b, _t) / _span)
                        except Exception:
                            ibs_value = None
                    
                    mid_idx = len(_et_safe) // 2
                    eval_time = float(_et_safe[mid_idx])
                    brier_at_eval = float(np.asarray(brier_values, dtype=float)[mid_idx])
                    self.latest_brier_df = pd.DataFrame({"time": _et_safe, "brier_score": brier_values})
                    self.latest_calibration_df = self._summarize_calibration(y_test_bs, survival_matrix[:, mid_idx], eval_time)
                    self.latest_eval_time = eval_time

                    # ── IBS KM (null model) and BSS ──
                    ibs_km_value = None
                    bss_value = None
                    ibs_ci = None
                    bss_ci = None
                    try:
                        _ibs_m, _ibs_km, _bss = self._compute_ibs_and_bss(
                            model, X_train, y_train, X_test, y_test, eval_times)
                        if ibs_value is None and _ibs_m is not None:
                            ibs_value = _ibs_m
                        ibs_km_value = _ibs_km
                        bss_value = _bss
                        ibs_ci, bss_ci = self._compute_ibs_bss_ci(
                            model, X_train, y_train, X_test, y_test, eval_times, random_state=random_state)
                    except Exception:
                        pass

                    # ── Quantile-based metrics (Q25, Q50, Q75 of event times) ──
                    event_mask_q = y_test_bs["event"].astype(bool)
                    event_times_q = np.asarray(y_test_bs["time"], dtype=float)[event_mask_q]
                    if event_times_q.size >= 4:
                        time_q25 = float(np.percentile(event_times_q, 25))
                        time_q50 = float(np.percentile(event_times_q, 50))
                        time_q75 = float(np.percentile(event_times_q, 75))
                        quantile_times = np.array([time_q25, time_q50, time_q75])

                        # Brier at quartiles
                        try:
                            surv_at_q = np.asarray([[fn(t) for t in quantile_times] for fn in survival_functions_test], dtype=float)
                            _, brier_q_vals = brier_score(y_train, y_test_bs, surv_at_q, quantile_times)
                            brier_q_vals = np.asarray(brier_q_vals, dtype=float)
                            brier_q25 = float(brier_q_vals[0]); brier_q50 = float(brier_q_vals[1]); brier_q75 = float(brier_q_vals[2])
                        except Exception:
                            pass

                        # AUROC at quartiles
                        if callable(cumulative_dynamic_auc):
                            try:
                                risk_matrix_q = 1.0 - np.asarray([[fn(t) for t in quantile_times] for fn in survival_functions_test], dtype=float)
                                auc_vals, _ = cumulative_dynamic_auc(y_train, y_test_bs, risk_matrix_q, quantile_times)
                                auc_vals = np.asarray(auc_vals, dtype=float)
                                auroc_q25 = float(auc_vals[0]); auroc_q50 = float(auc_vals[1]); auroc_q75 = float(auc_vals[2])
                            except Exception:
                                pass

                        # Harrell's C at quartiles (IPCW with tau=quartile)
                        for tau_q, attr_name in [(time_q25, 'c_harrell_q25'), (time_q50, 'c_harrell_q50'), (time_q75, 'c_harrell_q75')]:
                            try:
                                r = concordance_index_ipcw(y_train, y_test, test_preds, tau=tau_q)
                                locals()[attr_name] = float(np.asarray(r).reshape(-1)[0])
                            except Exception:
                                pass
                        c_harrell_q25 = locals().get('c_harrell_q25'); c_harrell_q50 = locals().get('c_harrell_q50'); c_harrell_q75 = locals().get('c_harrell_q75')

                except Exception as exc_metric:
                    warnings_list.append(f"No se pudieron calcular todas las métricas de Brier/IBS: {exc_metric}")
                    if self.latest_brier_df.empty:
                        self.latest_brier_df = pd.DataFrame()
                        self.latest_eval_time = None
                    self.latest_calibration_df = pd.DataFrame()

            imp_df = self._compute_permutation_importance(model, X_test, y_test, random_state) if len(X_test) > 0 else pd.DataFrame(columns=["feature", "importance", "importance_std", "importance_lower", "importance_upper"])
            if imp_df.empty and len(X_train) > 0:
                imp_df = self._compute_permutation_importance(model, X_train, y_train, random_state)
            self.feature_importance_df = imp_df
            self.latest_survival_profiles = self._build_survival_profiles(model, X_encoded, full_preds)

            prediction_df = clean_data[[duration_col, event_col]].copy()
            prediction_df["risk_score"] = np.asarray(full_preds, dtype=float)
            try:
                labels = ["Q1 bajo", "Q2 medio-bajo", "Q3 medio-alto", "Q4 alto"]
                prediction_df["risk_group"] = pd.qcut(
                    prediction_df["risk_score"].rank(method="first"),
                    q=min(4, max(2, prediction_df["risk_score"].nunique())),
                    labels=labels[: min(4, max(2, prediction_df["risk_score"].nunique()))],
                    duplicates="drop",
                )
            except Exception:
                prediction_df["risk_group"] = "Grupo único"
            self.latest_prediction_df = prediction_df

            removed_index = filtered_data.index.difference(clean_data.index)
            removed_case_ids = [str(idx) for idx in list(removed_index)]

            train_events_count = int(np.asarray(y_train["event"], dtype=bool).sum()) if len(y_train) > 0 else 0
            train_total_count = int(len(y_train))
            train_censored_count = max(train_total_count - train_events_count, 0)
            test_events_count = int(np.asarray(y_test["event"], dtype=bool).sum()) if len(y_test) > 0 else 0
            test_total_count = int(len(y_test))
            test_censored_count = max(test_total_count - test_events_count, 0)

            metrics = {
                "c_index_train": c_index_train,
                "c_index_train_ci": c_index_train_ci,
                "c_index_test": c_index_test,
                "c_index_test_ci": c_index_test_ci,
                "c_index_uno": c_index_uno,
                "c_index_uno_ci": c_index_test_ci if c_index_uno is not None else None,
                "c_index_antolini": c_index_antolini,
                "c_index_antolini_ci": c_index_test_ci if c_index_antolini is not None else None,
                "tau": resolved_tau,
                "c_index_cv_mean": c_index_cv_mean,
                "c_index_cv_ci": c_index_cv_ci,
                "c_index_cv_std": c_index_cv_std,
                "oob_score": oob_score,
                "ibs": ibs_value,
                "ibs_ci": ibs_ci,
                "ibs_km": ibs_km_value,
                "bss": bss_value,
                "bss_ci": bss_ci,
                "brier_at_eval_time": brier_at_eval,
                "brier_eval_time": eval_time,
                "time_q25": time_q25, "time_q50": time_q50, "time_q75": time_q75,
                "brier_q25": brier_q25, "brier_q50": brier_q50, "brier_q75": brier_q75,
                "auroc_q25": auroc_q25, "auroc_q50": auroc_q50, "auroc_q75": auroc_q75,
                "c_harrell_q25": c_harrell_q25, "c_harrell_q50": c_harrell_q50, "c_harrell_q75": c_harrell_q75,
                "n_rows_filtered": int(len(filtered_data)),
                "n_rows_clean": int(len(clean_data)),
                "n_rows_removed": int(len(removed_case_ids)),
                "removed_case_ids": removed_case_ids,
                "n_train": train_total_count,
                "n_train_events": train_events_count,
                "n_train_censored": train_censored_count,
                "n_test": test_total_count,
                "n_test_events": test_events_count,
                "n_test_censored": test_censored_count,
            }

            # ── Multi-seed: one snapshot per seed ────────────────────────
            if len(_all_seeds) > 1:
                _structural_fields = {
                    _sk: metrics.get(_sk) for _sk in (
                        "tau", "c_index_cv_ci", "brier_eval_time",
                        "time_q25", "time_q50", "time_q75",
                        "n_rows_filtered", "n_rows_clean", "n_rows_removed",
                        "removed_case_ids", "n_train", "n_train_events",
                        "n_train_censored", "n_test", "n_test_events", "n_test_censored",
                    )
                }
                # First seed is already in (model, X_test, y_test, metrics)
                _seed_runs = [(model, X_test, y_test, metrics, _all_seeds[0])]
                for _extra_seed in _all_seeds[1:]:
                    _emodel, _eXtest, _eytest, _em = self._fit_and_score_seed(
                        _extra_seed, X_encoded, y_structured, rsf_params,
                        test_size, stratify_values, cv_metric_code,
                        resolved_tau, eval_times,
                    )
                    if _emodel is not None and _em:
                        for _sk, _sv in _structural_fields.items():
                            _em.setdefault(_sk, _sv)
                        _seed_runs.append((_emodel, _eXtest, _eytest, _em, _extra_seed))

                if store_snapshot:
                    for _smodel, _sXtest, _sytest, _smet, _sval in _seed_runs:
                        _sp = copy.deepcopy(rsf_params)
                        _sp["random_state"] = int(_sval)
                        _sp["seed_used"] = int(_sval)
                        _sp["seeds_used"] = [int(_sval)]
                        _sfull = np.asarray(_smodel.predict(X_encoded), dtype=float)
                        _simp = self._compute_permutation_importance(_smodel, _sXtest, _sytest, int(_sval)) if len(_sXtest) > 0 else pd.DataFrame(columns=["feature", "importance", "importance_std", "importance_lower", "importance_upper"])
                        _spred = prediction_df.copy()
                        _spred["risk_score"] = _sfull
                        _sprof = self._build_survival_profiles(_smodel, X_encoded, _sfull)
                        _srpt = self._build_results_report(clean_data, _sp, _smet, warnings_list, test_size=test_size, model=_smodel)
                        if tuning_summary:
                            _srpt = f"{_srpt}\n\n{tuning_summary}"
                        self._store_snapshot_direct(
                            _sp, _srpt, test_size,
                            _smet, _smodel, _simp, _spred, _sprof,
                        )
                    store_snapshot = False  # already stored per-seed above

                # Use the representative seed (closest to average) for active display;
                # fall back to last seed if not specified.
                _preferred_seed = None
                if isinstance(params_override, dict):
                    _ps = params_override.get("seed_used")
                    if _ps is not None:
                        try:
                            _preferred_seed = int(_ps)
                        except (TypeError, ValueError):
                            _preferred_seed = None
                _active_run = None
                if _preferred_seed is not None:
                    for _sr in _seed_runs:
                        if int(_sr[4]) == _preferred_seed:
                            _active_run = _sr
                            break
                if _active_run is None:
                    _active_run = _seed_runs[-1]
                model, X_test, y_test, metrics = _active_run[0], _active_run[1], _active_run[2], _active_run[3]
                _active_seed_used = int(_active_run[4])
                full_preds = np.asarray(model.predict(X_encoded), dtype=float)
                self.model = model
                imp_df = self._compute_permutation_importance(model, X_test, y_test, _active_seed_used) if len(X_test) > 0 else pd.DataFrame(columns=["feature", "importance", "importance_std", "importance_lower", "importance_upper"])
                if imp_df.empty:
                    try:
                        if test_size is None or test_size <= 0:
                            X_train_last = X_encoded
                            y_train_last = y_structured
                        else:
                            X_train_last, _, y_train_last, _ = train_test_split(
                                X_encoded, y_structured, test_size=test_size, random_state=_active_seed_used, stratify=stratify_values
                            )
                        if len(X_train_last) > 0:
                            imp_df = self._compute_permutation_importance(model, X_train_last, y_train_last, _active_seed_used)
                    except Exception:
                        pass
                self.feature_importance_df = imp_df
                self.latest_survival_profiles = self._build_survival_profiles(model, X_encoded, full_preds)
                prediction_df["risk_score"] = full_preds

            self.results = metrics

            report_text = self._build_results_report(clean_data, rsf_params, metrics, warnings_list, test_size=test_size, model=self.model)
            if tuning_summary:
                report_text = f"{report_text}\n\n{tuning_summary}"
            self.latest_report_text = report_text
            self.latest_tuning_summary = tuning_summary or ""
            self.results_text.delete("1.0", tk.END)
            self.results_text.insert(tk.END, report_text)

            if store_snapshot:
                self._store_current_model_snapshot(rsf_params, report_text, test_size=test_size)
            self.plot_feature_importance()
            self.plot_risk_groups_km()
            self.plot_survival_profiles()
            self.plot_variable_impact()
            self.plot_calibration()
            self.plot_brier_curve()
            self.plot_single_tree()
            self.plot_minimal_depth()
            self._sync_pdp_covariate_selector()
            self._sync_shap_dep_selectors()
            self.plot_pdp()

            if warnings_list:
                messagebox.showwarning("Avisos de preparación", "\n".join(warnings_list))
        except Exception as exc:
            traceback_text = traceback.format_exc(limit=3)
            self.results_text.delete("1.0", tk.END)
            self.results_text.insert(tk.END, f"No se pudo ajustar el RSF:\n{exc}\n\n{traceback_text}")
            messagebox.showerror("Error en RSF", f"Ocurrió un error al ajustar el modelo:\n{exc}")

    # ==================================================================
    # Robust Optimization Protocol
    # ==================================================================

    def _open_robust_progress_dialog(self, total_phases=8):
        """Open a multi-phase progress dialog for robust optimization."""
        parent_window = self.winfo_toplevel()
        dialog = tk.Toplevel(parent_window)
        dialog.title("Optimización Robusta RSF")
        dialog.geometry("600x520")
        dialog.transient(parent_window)
        dialog.resizable(True, True)
        try:
            dialog.minsize(600, 520)
        except Exception:
            pass
        dialog.protocol("WM_DELETE_WINDOW", self._request_auto_tuning_cancel)

        container = ttk.Frame(dialog, padding=14)
        container.pack(fill=tk.BOTH, expand=True)

        ttk.Label(container, text="Optimización Robusta RSF", font=("Segoe UI", 12, "bold")).pack(anchor="w", pady=(0, 8))

        # ── 8 phase indicator squares ──
        phase_names = [
            "1.Base", "2.VIMP", "3.Search", "4.Fijo",
            "5.Depth", "6.Elim", "7.Grid", "8.Valid"
        ]
        phases_frame = ttk.Frame(container)
        phases_frame.pack(fill=tk.X, pady=(0, 10))
        self._robust_phase_labels = []
        for i, name in enumerate(phase_names):
            lbl = tk.Label(phases_frame, text=name, width=8, height=1,
                           bg="#d4d4d4", fg="#666666", relief="raised", bd=1,
                           font=("Segoe UI", 8, "bold"))
            lbl.grid(row=0, column=i, padx=2, pady=2)
            self._robust_phase_labels.append(lbl)

        # ── Current phase / detail text ──
        self._robust_phase_var = StringVar(value="Inicializando...")
        ttk.Label(container, textvariable=self._robust_phase_var, justify="left", wraplength=560,
                  font=("Segoe UI", 9, "bold")).pack(anchor="w", pady=(0, 2))

        self._robust_detail_var = StringVar(value="")
        ttk.Label(container, textvariable=self._robust_detail_var, justify="left", wraplength=560,
                  foreground="#333333").pack(anchor="w", pady=(0, 8))

        # ── 4 macro-phase progress bars ──
        macro_phases = [
            ("Fase I — Limpieza Inicial", "Pasos 1-2"),
            ("Fase II — Afinación Primaria", "Pasos 3-4"),
            ("Fase III — Resolución Correlaciones", "Pasos 5-6"),
            ("Fase IV — Consolidación Final", "Pasos 7-8"),
        ]
        self._robust_macro_bars = []
        self._robust_macro_labels = []
        for title, sub in macro_phases:
            row_frame = ttk.Frame(container)
            row_frame.pack(fill=tk.X, pady=(1, 3))
            lbl_var = StringVar(value=f"{title} ({sub})")
            ttk.Label(row_frame, textvariable=lbl_var, font=("Segoe UI", 8)).pack(anchor="w")
            bar = ttk.Progressbar(row_frame, orient="horizontal", mode="determinate",
                                  maximum=100, value=0, length=560)
            bar.pack(fill=tk.X, pady=(1, 0))
            self._robust_macro_bars.append(bar)
            self._robust_macro_labels.append(lbl_var)

        # ── Sub-step progress (current step detail) ──
        step_frame = ttk.Frame(container)
        step_frame.pack(fill=tk.X, pady=(6, 4))
        ttk.Label(step_frame, text="Paso actual:", font=("Segoe UI", 8)).pack(anchor="w")
        self._robust_step_bar = ttk.Progressbar(step_frame, orient="horizontal", mode="determinate",
                                                  maximum=100, value=0, length=560)
        self._robust_step_bar.pack(fill=tk.X, pady=(1, 0))

        self._robust_elapsed_var = StringVar(value="Tiempo transcurrido: 00:00:00")
        ttk.Label(container, textvariable=self._robust_elapsed_var, foreground="#555555").pack(anchor="w", pady=(6, 6))

        buttons = ttk.Frame(container)
        buttons.pack(fill=tk.X)
        self._tuning_cancel_button = ttk.Button(buttons, text="Cancelar", command=self._request_auto_tuning_cancel)
        self._tuning_cancel_button.pack(side=tk.RIGHT)

        # Keep a reference for the global phase bar (compatibility)
        self._robust_phase_bar = None

        dialog.update_idletasks()
        dialog.lift()
        self._tuning_progress_dialog = dialog

    def _update_robust_progress(self, phase_num, phase_text, detail="", step_cur=0, step_max=100, elapsed=0.0):
        """Update the robust optimization progress dialog."""
        # ── Update phase indicator squares ──
        phase_colors = {
            "completed": {"bg": "#22c55e", "fg": "#ffffff"},   # green
            "active":    {"bg": "#3b82f6", "fg": "#ffffff"},   # blue
            "pending":   {"bg": "#d4d4d4", "fg": "#666666"},   # gray
        }
        for i, lbl in enumerate(getattr(self, "_robust_phase_labels", [])):
            try:
                phase_idx = i + 1
                if phase_idx < phase_num:
                    lbl.configure(**phase_colors["completed"])
                elif phase_idx == phase_num:
                    lbl.configure(**phase_colors["active"])
                else:
                    lbl.configure(**phase_colors["pending"])
            except Exception:
                pass

        # ── Update text labels ──
        try:
            self._robust_phase_var.set(f"Fase {phase_num}/8: {phase_text}")
        except Exception:
            pass
        try:
            self._robust_detail_var.set(detail)
        except Exception:
            pass

        # ── Update macro-phase bars ──
        # Macro I=phases 1-2, II=3-4, III=5-6, IV=7-8
        macro_mapping = {1: 0, 2: 0, 3: 1, 4: 1, 5: 2, 6: 2, 7: 3, 8: 3}
        macro_idx = macro_mapping.get(phase_num, 0)
        is_first_in_macro = phase_num in (1, 3, 5, 7)  # first sub-phase of macro

        for i, bar in enumerate(getattr(self, "_robust_macro_bars", [])):
            try:
                if i < macro_idx:
                    # Completed macro phase
                    bar["maximum"] = 100
                    bar["value"] = 100
                elif i == macro_idx:
                    # Current macro phase: map sub-step progress
                    if is_first_in_macro:
                        # First half of macro: map step progress to 0-50%
                        pct = (step_cur / max(step_max, 1)) * 50
                    else:
                        # Second half of macro: 50% + map step progress to 50-100%
                        pct = 50 + (step_cur / max(step_max, 1)) * 50
                    bar["maximum"] = 100
                    bar["value"] = min(int(pct), 100)
                else:
                    # Future macro phase
                    bar["maximum"] = 100
                    bar["value"] = 0
            except Exception:
                pass

        # ── Update step bar ──
        try:
            self._robust_step_bar["maximum"] = max(int(step_max), 1)
            self._robust_step_bar["value"] = min(int(step_cur), int(step_max))
        except Exception:
            pass

        # ── Update elapsed time ──
        try:
            h, rem = divmod(int(elapsed), 3600)
            m, s = divmod(rem, 60)
            self._robust_elapsed_var.set(f"Tiempo transcurrido: {h:02d}:{m:02d}:{s:02d}")
        except Exception:
            pass

        dialog = getattr(self, "_tuning_progress_dialog", None)
        if dialog is not None:
            try:
                dialog.update_idletasks()
                dialog.update()
            except tk.TclError:
                self._tuning_cancel_requested = True

    def _compute_vimp_bootstrap(self, model, X, y, n_bootstrap=100, n_repeats=5, random_state=42,
                                   progress_callback=None):
        """Compute VIMP with bootstrap CIs. Returns DataFrame with columns:
        feature, importance, importance_lower, importance_upper.
        progress_callback(completed, total) is called after each iteration."""
        rng = np.random.default_rng(random_state)
        n_samples = len(X)
        all_importances = []  # shape: (n_bootstrap, n_features)

        for b in range(n_bootstrap):
            # Check for cancellation
            if self._tuning_cancel_requested:
                break
            idx = rng.choice(n_samples, size=n_samples, replace=True)
            X_b = X.iloc[idx].reset_index(drop=True)
            y_b = y[idx]
            try:
                imp = permutation_importance(model, X_b, y_b, n_repeats=n_repeats,
                                             random_state=random_state + b, n_jobs=1)
                all_importances.append(imp.importances_mean)
            except Exception:
                continue
            # Update progress
            if progress_callback is not None:
                progress_callback(b + 1, n_bootstrap)

        if self._tuning_cancel_requested:
            raise InterruptedError("Cancelado por el usuario.")

        if not all_importances:
            return pd.DataFrame(columns=["feature", "importance", "importance_lower", "importance_upper"])

        imp_matrix = np.array(all_importances)  # (B, p)
        mean_imp = np.mean(imp_matrix, axis=0)
        lower = np.percentile(imp_matrix, 2.5, axis=0)
        upper = np.percentile(imp_matrix, 97.5, axis=0)

        df = pd.DataFrame({
            "feature": X.columns,
            "importance": mean_imp,
            "importance_lower": lower,
            "importance_upper": upper,
        }).sort_values("importance", ascending=False).reset_index(drop=True)
        return df

    def _compute_minimal_depth(self, model, feature_names):
        """Compute average minimal depth for each feature across all trees in the RSF.
        Lower depth = more important (splits closer to root)."""
        n_features = len(feature_names)
        depth_sums = np.zeros(n_features, dtype=float)
        depth_counts = np.zeros(n_features, dtype=float)
        n_trees = len(model.estimators_)

        for tree in model.estimators_:
            tree_obj = tree.tree_
            n_nodes = tree_obj.node_count
            # Compute depth of each node via BFS
            node_depth = np.zeros(n_nodes, dtype=int)
            stack = [0]
            while stack:
                node_id = stack.pop()
                left = tree_obj.children_left[node_id]
                right = tree_obj.children_right[node_id]
                if left != -1:
                    node_depth[left] = node_depth[node_id] + 1
                    stack.append(left)
                if right != -1:
                    node_depth[right] = node_depth[node_id] + 1
                    stack.append(right)

            # For each feature, find minimal (shallowest) depth where it's used
            feature_indices = tree_obj.feature
            for feat_idx in range(n_features):
                nodes_using_feat = np.where(feature_indices == feat_idx)[0]
                if len(nodes_using_feat) > 0:
                    min_depth = int(np.min(node_depth[nodes_using_feat]))
                    depth_sums[feat_idx] += min_depth
                    depth_counts[feat_idx] += 1
                else:
                    # Feature not used in this tree -> assign max depth + 1
                    max_tree_depth = int(np.max(node_depth)) + 1
                    depth_sums[feat_idx] += max_tree_depth
                    depth_counts[feat_idx] += 1

        avg_depth = np.where(depth_counts > 0, depth_sums / depth_counts, np.inf)
        df = pd.DataFrame({
            "feature": list(feature_names),
            "avg_minimal_depth": avg_depth,
            "trees_used_in": depth_counts.astype(int),
            "usage_pct": (depth_counts / max(n_trees, 1) * 100).round(1),
        }).sort_values("avg_minimal_depth").reset_index(drop=True)
        return df

    def _resolve_vimp_cleaning_mode(self):
        raw_mode = getattr(self, "robust_vimp_mode_var", None)
        raw_mode = raw_mode.get() if raw_mode else "Permisivo (IC95% sup > 0)"
        if "Estricto" in str(raw_mode):
            return "strict", "Estricto (IC95% inf > 0)"
        if "Híbrido" in str(raw_mode):
            return "hybrid", "Híbrido (2 de 3)"
        return "permissive", "Permisivo (IC95% sup > 0)"

    def _resolve_correlation_pruning_threshold(self):
        raw_value = getattr(self, "corr_prune_threshold_var", None)
        raw_value = raw_value.get() if raw_value else "0.85"
        raw_text = str(raw_value).strip()
        if raw_text.lower() in {"sin poda", "none", "off", "no"}:
            return None, "Sin poda"
        threshold = self._coerce_float(raw_text, 0.85, minimum=0.50, maximum=0.999)
        return float(threshold), f"|rho| >= {float(threshold):.2f}"

    def _resolve_robust_tree_candidates(self, n_rows):
        """Return candidate tree counts for robust RSF protocols based on selected profile/manual grid."""
        selected_profile = self.tuning_profile_var.get().strip() if hasattr(self, "tuning_profile_var") else "General"
        if not selected_profile:
            selected_profile = "General"

        profile_trees = {
            "General": [100, 200, 300, 400, 500],
            "Pocos datos (<200)": [100, 200, 300, 400],
            "Mediano (200-500)": [200, 300, 400, 500, 700],
            "Pesado (>500)": [300, 500, 800, 1000],
        }

        if selected_profile == "Manual":
            try:
                trees = list(self._build_manual_profile_config().get("trees", []))
            except Exception:
                trees = []
        else:
            trees = list(profile_trees.get(selected_profile, profile_trees["General"]))

        recommended_profile = self._suggest_tuning_profile(int(max(n_rows, 1)))
        trees.extend(profile_trees.get(recommended_profile, []))

        try:
            trees.append(self._coerce_int(self.n_estimators_var.get(), 300, minimum=10))
        except Exception:
            pass

        cleaned = sorted({int(t) for t in trees if str(t).strip() and int(float(t)) >= 50 and int(float(t)) <= 3000})
        if not cleaned:
            cleaned = [100, 200, 300, 500]
        return cleaned

    def _select_vimp_features(self, vimp_df, mode_code):
        """Return (retained_features, removed_features, criterion_text) based on selected VIMP cleaning mode."""
        if not isinstance(vimp_df, pd.DataFrame) or vimp_df.empty:
            return [], [], "Sin VIMP disponible"

        if mode_code == "strict":
            retained_mask = vimp_df["importance_lower"] > 0
            criterion_text = "eliminar si IC95% inferior del VIMP <= 0"
        else:
            # permissive and hybrid start with the same gate in phase 2;
            # hybrid applies an additional 2/3 filter after minimal-depth + RFE.
            retained_mask = vimp_df["importance_upper"] > 0
            criterion_text = "eliminar si IC95% superior del VIMP <= 0"

        retained_features = vimp_df.loc[retained_mask, "feature"].tolist()
        removed_features = vimp_df.loc[~retained_mask, "feature"].tolist()

        if not retained_features:
            retained_features = vimp_df.head(3)["feature"].tolist()
            removed_features = [f for f in vimp_df["feature"] if f not in retained_features]

        return retained_features, removed_features, criterion_text

    def _apply_hybrid_signature_filter(self, current_features, vimp_df, mdepth_df, min_keep=2):
        """Hybrid 2-of-3 filter on the current signature:
        1) VIMP IC95% lower > 0
        2) VIMP mean importance > 0
        3) Minimal depth in top 80% (lower depth is better)
        Returns (filtered_features, diagnostics)."""
        if not current_features:
            return [], {}

        vimp_map = {}
        if isinstance(vimp_df, pd.DataFrame) and not vimp_df.empty:
            for _, row in vimp_df.iterrows():
                vimp_map[row["feature"]] = {
                    "importance": row.get("importance"),
                    "importance_lower": row.get("importance_lower"),
                }

        md_rank = []
        if isinstance(mdepth_df, pd.DataFrame) and not mdepth_df.empty and "feature" in mdepth_df:
            md_rank = [f for f in mdepth_df["feature"].tolist() if f in current_features]
        if not md_rank:
            md_rank = list(current_features)

        top_count = max(int(np.ceil(len(md_rank) * 0.80)), min_keep)
        md_top_set = set(md_rank[:top_count])

        scored = []
        for feat in current_features:
            info = vimp_map.get(feat, {})
            c1 = (info.get("importance_lower") is not None and pd.notna(info.get("importance_lower"))
                  and float(info.get("importance_lower")) > 0)
            c2 = (info.get("importance") is not None and pd.notna(info.get("importance"))
                  and float(info.get("importance")) > 0)
            c3 = feat in md_top_set
            score = int(c1) + int(c2) + int(c3)
            scored.append((feat, score, c1, c2, c3, float(info.get("importance") or 0.0)))

        keep = [feat for feat, score, *_rest in scored if score >= 2]
        if len(keep) < min_keep:
            scored_sorted = sorted(scored, key=lambda x: (x[1], x[5]), reverse=True)
            keep = [x[0] for x in scored_sorted[:max(min_keep, min(len(scored_sorted), min_keep))]]

        diagnostics = {
            "top80_depth_count": top_count,
            "kept": keep,
            "details": [
                {
                    "feature": feat,
                    "score": score,
                    "c1_ic_lower_pos": c1,
                    "c2_vimp_pos": c2,
                    "c3_depth_top80": c3,
                }
                for feat, score, c1, c2, c3, _ in scored
            ],
        }
        return keep, diagnostics

    def _prune_correlated_features_by_depth(self, X, mdepth_df, threshold=0.85, min_keep=2):
        """Drop redundant highly correlated features, keeping the shallower minimal-depth feature."""
        if X is None or X.empty or len(X.columns) <= min_keep:
            return list(getattr(X, "columns", [])), []
        if not isinstance(mdepth_df, pd.DataFrame) or mdepth_df.empty:
            return list(X.columns), []

        try:
            corr_matrix = X.corr(method="spearman").abs()
        except Exception:
            return list(X.columns), []

        depth_rank = {feat: idx for idx, feat in enumerate(mdepth_df["feature"].tolist())}
        kept = list(X.columns)
        removed_pairs = []

        for i, left in enumerate(list(X.columns)):
            for j in range(i + 1, len(X.columns)):
                right = X.columns[j]
                if left not in kept or right not in kept:
                    continue
                try:
                    corr_val = float(corr_matrix.loc[left, right])
                except Exception:
                    continue
                if not np.isfinite(corr_val) or corr_val < float(threshold):
                    continue
                if len(kept) <= min_keep:
                    continue

                left_rank = depth_rank.get(left, 10**9)
                right_rank = depth_rank.get(right, 10**9)
                drop_feat = right if left_rank <= right_rank else left
                keep_feat = left if drop_feat == right else right
                if drop_feat in kept and len(kept) > min_keep:
                    kept.remove(drop_feat)
                    removed_pairs.append((keep_feat, drop_feat, corr_val))

        return kept, removed_pairs

    def _compute_c_uno_score(self, model, X_train, y_train, X_test, y_test, tau):
        """Compute C-index Uno (IPCW) for a given model and test set."""
        if len(X_test) == 0 or not callable(concordance_index_ipcw):
            return None
        try:
            test_preds = model.predict(X_test)
            result = concordance_index_ipcw(y_train, y_test, test_preds, tau=tau)
            return float(np.asarray(result).reshape(-1)[0])
        except Exception:
            return None

    def _compute_c_uno_internal_cv(self, X, y, params, n_folds=None, metric_code=None):
        """Compute internal CV score for robust optimization using the selected metric."""
        n = len(X)
        if n < 10:
            return None
        if n_folds is None:
            n_folds = 3 if n < 150 else 5
        event_ints = np.asarray(y["event"], dtype=bool).astype(int)
        if metric_code is None:
            metric_code = self._resolve_optimization_metric_choice()[0]
        try:
            cv = StratifiedKFold(n_splits=n_folds, shuffle=True,
                                 random_state=params.get("random_state", 42))
            scores = []
            for tr_idx, val_idx in cv.split(X, event_ints):
                X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
                y_tr, y_val = y[tr_idx], y[val_idx]
                mdl = RandomSurvivalForest(**params)
                self._fit_model_with_ui_pump(mdl, X_tr, y_tr)
                self._flush_tuning_dialog_events()
                if self._tuning_cancel_requested:
                    return None

                if metric_code in {"harrell", "uno"}:
                    score = self._compute_cv_metric_value(mdl, X_tr, y_tr, X_val, y_val, metric_code=metric_code)
                elif metric_code == "ibs":
                    eval_times = self._build_evaluation_time_grid(y_tr, y_val, tau=None)
                    score, _ibs_km, _bss = self._compute_ibs_and_bss(mdl, X_tr, y_tr, X_val, y_val, eval_times)
                elif metric_code == "brier":
                    score = None
                    eval_times = self._build_evaluation_time_grid(y_tr, y_val, tau=None)
                    if len(X_val) > 0 and eval_times is not None and callable(brier_score):
                        try:
                            surv_fns = mdl.predict_survival_function(X_val)
                            surv_matrix = np.asarray([fn(eval_times) for fn in surv_fns], dtype=float)
                            brier_out = brier_score(y_tr, y_val, surv_matrix, eval_times)
                            if isinstance(brier_out, tuple) and len(brier_out) >= 2:
                                _, brier_vals = brier_out
                                brier_arr = np.asarray(brier_vals, dtype=float)
                                if brier_arr.size > 0 and np.isfinite(brier_arr).any():
                                    score = float(brier_arr[len(brier_arr) // 2])
                        except Exception:
                            score = None
                else:
                    score = None

                if score is not None:
                    scores.append(score)
            return float(np.mean(scores)) if scores else None
        except Exception:
            return None

    def _compute_ibs_and_bss_oob(self, model, X_train, y_train):
        """Compute IBS and BSS using a held-out subset of the training data.
        Usa predict_survival_function en un subset del train como proxy de test.
        Returns (ibs_model, ibs_km, bss) or (None, None, None) on failure.
        """
        try:
            _bs_fn = brier_score if callable(brier_score) else None
            _ibs_fn = integrated_brier_score if callable(integrated_brier_score) else None
            if _bs_fn is None and _ibs_fn is None:
                return None, None, None
            if not hasattr(model, "predict_survival_function"):
                return None, None, None

            train_times = np.asarray(y_train["time"], dtype=float)
            train_events = np.asarray(y_train["event"], dtype=bool)
            n = len(train_times)
            if n < 8:
                return None, None, None

            max_t = float(np.nanmax(train_times[np.isfinite(train_times)]))

            # Usar una fracción del train como "pseudo-test"
            # (las muestras con tiempo < max_t para cumplir el requisito de sksurv)
            _valid = train_times < max_t
            if _valid.sum() < 4:
                _cap = float(np.percentile(train_times[np.isfinite(train_times)], 99))
                _valid = train_times <= _cap
            if _valid.sum() < 4:
                return None, None, None

            X_sub = X_train[_valid]
            y_sub = y_train[_valid]
            sub_times = np.asarray(y_sub["time"], dtype=float)

            # Grilla de tiempos de evaluación dentro del rango del subset
            et_min = float(np.percentile(sub_times[sub_times > 0], 10)) if (sub_times > 0).sum() > 2 else float(np.nanmin(sub_times[sub_times > 0]))
            et_max = float(np.percentile(sub_times, 90))
            if et_max <= et_min:
                et_max = float(np.nanmax(sub_times))
            if et_max <= et_min:
                return None, None, None

            # Usar event_times_ del modelo como candidatos si están disponibles
            if hasattr(model, "event_times_"):
                ev_t = np.asarray(model.event_times_, dtype=float)
                et = ev_t[(ev_t >= et_min) & (ev_t < et_max)]
            else:
                et = None

            if et is None or len(et) < 2:
                et = np.linspace(et_min, et_max * 0.99, num=min(30, max(5, int(_valid.sum()) // 3)))
            et = np.unique(et)
            if len(et) < 2:
                return None, None, None

            # Predecir funciones de supervivencia para el subset
            surv_fns = model.predict_survival_function(X_sub)
            surv_matrix = np.asarray([[fn(t) for t in et] for fn in surv_fns], dtype=float)

            # ── IBS del modelo ──────────────────────────────────────────────
            ibs_model = None
            if _ibs_fn is not None:
                try:
                    ibs_model = float(_ibs_fn(y_train, y_sub, surv_matrix, et))
                except Exception:
                    pass
            if ibs_model is None and _bs_fn is not None:
                try:
                    _, bv = _bs_fn(y_train, y_sub, surv_matrix, et)
                    _t = np.asarray(et, dtype=float)
                    span = float(_t[-1] - _t[0])
                    if span > 0:
                        ibs_model = float(np.trapz(np.asarray(bv, dtype=float), _t) / span)
                except Exception:
                    pass
            if ibs_model is None:
                return None, None, None

            # ── IBS del modelo nulo KM ──────────────────────────────────────
            ibs_km = None
            try:
                from lifelines import KaplanMeierFitter
                kmf = KaplanMeierFitter()
                kmf.fit(train_times, event_observed=train_events)
                km_surv = np.asarray([float(kmf.predict(t)) for t in et], dtype=float)
            except Exception:
                # KM empírico sin lifelines
                try:
                    _sort_idx = np.argsort(train_times)
                    _st = train_times[_sort_idx]
                    _se = train_events[_sort_idx].astype(float)
                    _n_t = len(_st)
                    km_surv = np.ones(len(et))
                    _s = 1.0
                    _j = 0
                    for _i_t, _tau in enumerate(et):
                        while _j < _n_t and _st[_j] <= _tau:
                            if _se[_j] > 0:
                                _s *= (1.0 - 1.0 / max(_n_t - _j, 1))
                            _j += 1
                        km_surv[_i_t] = _s
                except Exception:
                    km_surv = None

            if km_surv is not None:
                km_mat = np.tile(km_surv, (len(y_sub), 1))
                if _ibs_fn is not None:
                    try:
                        ibs_km = float(_ibs_fn(y_train, y_sub, km_mat, et))
                    except Exception:
                        pass
                if ibs_km is None and _bs_fn is not None:
                    try:
                        _, bvk = _bs_fn(y_train, y_sub, km_mat, et)
                        _t = np.asarray(et, dtype=float)
                        span = float(_t[-1] - _t[0])
                        if span > 0:
                            ibs_km = float(np.trapz(np.asarray(bvk, dtype=float), _t) / span)
                    except Exception:
                        pass

            bss = None
            if ibs_km is not None and ibs_km > 0:
                bss = float(1.0 - ibs_model / ibs_km)

            return ibs_model, ibs_km, bss
        except Exception:
            return None, None, None

    def _compute_ibs_and_bss(self, model, X_train, y_train, X_test, y_test, eval_times):
        """Compute IBS and Brier Skill Score (BSS) comparing model vs KM null model.
        Returns (ibs_model, ibs_km, bss) or (None, None, None) on failure."""
        if len(X_test) == 0 or eval_times is None:
            return None, None, None
        try:
            # ── Filtrar test para que tiempos estén dentro del rango de train ──
            # sksurv IPCW requiere que todos los tiempos de test < max(train time)
            train_times_arr = np.asarray(y_train["time"], dtype=float)
            test_times_arr = np.asarray(y_test["time"], dtype=float)
            max_train_t = float(np.nanmax(train_times_arr[np.isfinite(train_times_arr)]))
            min_train_t = float(np.nanmin(train_times_arr[np.isfinite(train_times_arr)]))
            valid_mask = test_times_arr < max_train_t
            if valid_mask.sum() < 2:
                return None, None, None

            X_test_f = X_test[valid_mask]
            y_test_f = y_test[valid_mask]

            # Ajustar eval_times para que estén dentro de [min_test_filtered, max_test_filtered)
            et = np.asarray(eval_times, dtype=float)
            test_f_times = np.asarray(y_test_f["time"], dtype=float)
            et = et[(et >= float(np.nanmin(test_f_times))) & (et < float(np.nanmax(test_f_times)))]
            et = np.unique(et)
            if et.size < 2:
                return None, None, None

            surv_fns = model.predict_survival_function(X_test_f)
            surv_matrix = np.asarray([fn(et) for fn in surv_fns], dtype=float)

            # Model IBS
            ibs_model = None
            if callable(integrated_brier_score):
                try:
                    ibs_model = float(integrated_brier_score(y_train, y_test_f, surv_matrix, et))
                except Exception:
                    pass
            if ibs_model is None:
                _, brier_vals = brier_score(y_train, y_test_f, surv_matrix, et)
                _t = np.asarray(et, dtype=float)
                _b = np.asarray(brier_vals, dtype=float)
                span = float(_t[-1] - _t[0])
                if span > 0 and _t.size >= 2:
                    ibs_model = float(np.trapz(_b, _t) / span)

            # KM null model IBS
            kmf = KaplanMeierFitter()
            train_events = np.asarray(y_train["event"], dtype=bool)
            kmf.fit(train_times_arr, event_observed=train_events)
            km_surv_at_times = np.asarray([kmf.predict(t) for t in et], dtype=float)
            km_surv_matrix = np.tile(km_surv_at_times, (len(X_test_f), 1))

            ibs_km = None
            if callable(integrated_brier_score):
                try:
                    ibs_km = float(integrated_brier_score(y_train, y_test_f, km_surv_matrix, et))
                except Exception:
                    pass
            if ibs_km is None and callable(brier_score):
                try:
                    _, brier_km = brier_score(y_train, y_test_f, km_surv_matrix, et)
                    _t = np.asarray(et, dtype=float)
                    _bk = np.asarray(brier_km, dtype=float)
                    span = float(_t[-1] - _t[0])
                    if span > 0:
                        ibs_km = float(np.trapz(_bk, _t) / span)
                except Exception:
                    pass

            bss = None
            if ibs_model is not None and ibs_km is not None and ibs_km > 0:
                bss = 1.0 - (ibs_model / ibs_km)

            return ibs_model, ibs_km, bss
        except Exception:
            return None, None, None

    def _compute_ibs_bss_ci(self, model, X_train, y_train, X_test, y_test, eval_times, n_bootstrap=120, random_state=42):
        """Compute bootstrap 95% CI for IBS and BSS.
        Returns (ibs_ci, bss_ci) where each is (lower, upper) or None."""
        if X_test is None or len(X_test) < 5 or eval_times is None:
            return None, None
        rng = np.random.default_rng(random_state)
        idx_pool = np.arange(len(X_test))
        ibs_samples = []
        bss_samples = []
        # Pump UI en carga/manual; evitarlo durante autotuning masivo.
        _pump_ui = bool(
            getattr(self, "_loading_model_in_progress", False)
            or not bool(getattr(self, "_auto_tuning_in_progress", False))
        )
        for _boot_i in range(int(max(20, n_bootstrap))):
            if _pump_ui and _boot_i % 20 == 0:
                try:
                    self.update_idletasks()
                except Exception:
                    pass
            boot_idx = rng.choice(idx_pool, size=len(idx_pool), replace=True)
            X_b = X_test.iloc[boot_idx]
            y_b = y_test[boot_idx]
            if np.unique(np.asarray(y_b["event"], dtype=bool)).size < 2:
                continue
            try:
                ibs_m, ibs_km, bss_v = self._compute_ibs_and_bss(model, X_train, y_train, X_b, y_b, eval_times)
                if ibs_m is not None and np.isfinite(ibs_m):
                    ibs_samples.append(ibs_m)
                if bss_v is not None and np.isfinite(bss_v):
                    bss_samples.append(bss_v)
            except Exception:
                continue
        ibs_ci = None
        if len(ibs_samples) >= 10:
            lo = float(np.nanquantile(ibs_samples, 0.025))
            hi = float(np.nanquantile(ibs_samples, 0.975))
            ibs_ci = (min(lo, hi), max(lo, hi))
        bss_ci = None
        if len(bss_samples) >= 10:
            lo = float(np.nanquantile(bss_samples, 0.025))
            hi = float(np.nanquantile(bss_samples, 0.975))
            bss_ci = (min(lo, hi), max(lo, hi))
        return ibs_ci, bss_ci

    def run_robust_optimization(self):
        """Execute the full Robust RSF Optimization Protocol (8 phases)."""
        if not SKSURV_AVAILABLE:
            messagebox.showerror("RSF no disponible",
                                 f"Instala scikit-survival para usar esta función.\n\n{SKSURV_IMPORT_ERROR}")
            return
        if self.data is None:
            messagebox.showerror("Error", "Cargue datos o use el dataset compartido primero.")
            return

        filtered_data = self.filter_component.apply_filters()
        if filtered_data is None or filtered_data.empty:
            messagebox.showerror("Error", "No hay datos disponibles tras aplicar filtros.")
            return

        duration_col = self.duration_var.get().strip()
        event_col = self.event_var.get().strip()
        selected_indices = self.covariates_listbox.curselection()
        covariates = [self.covariates_listbox.get(i) for i in selected_indices]
        if not duration_col or not event_col or not covariates:
            messagebox.showerror("Error", "Seleccione tiempo, evento y al menos una covariable.")
            return

        self._tuning_cancel_requested = False
        start_time = time.perf_counter()
        self._open_robust_progress_dialog(total_phases=8)
        report_lines = ["=" * 60, "PROTOCOLO DE OPTIMIZACIÓN ROBUSTA RSF", "=" * 60, ""]
        phase_log = {}
        vimp_mode_code, vimp_mode_label = self._resolve_vimp_cleaning_mode()
        opt_metric_code, opt_metric_label, opt_metric_higher_better = self._resolve_optimization_metric_choice()
        corr_threshold, corr_threshold_label = self._resolve_correlation_pruning_threshold()

        try:
            # ── PHASE 1: Baseline model ──────────────────────────────
            self._update_robust_progress(1, "Entrenamiento Base (Baseline)",
                                         "Paso 1/4: Preparando datos y limpiando covariables...",
                                         0, 4, time.perf_counter() - start_time)
            if self._tuning_cancel_requested:
                raise InterruptedError("Cancelado por el usuario.")

            clean_data, X_encoded, y_structured, warnings_list = self._prepare_dataframe_for_rsf(
                filtered_data, duration_col, event_col, covariates)

            random_state = self._parse_random_seeds()[0]
            n_features_orig = X_encoded.shape[1]
            tree_candidates = self._resolve_robust_tree_candidates(len(clean_data))
            baseline_tree_count = self._coerce_int(self.n_estimators_var.get(), 300, minimum=10)

            # Use defaults for baseline
            baseline_params = {
                "n_estimators": baseline_tree_count,
                "max_features": "sqrt",
                "min_samples_leaf": 10,
                "min_samples_split": 20,
                "bootstrap": True,
                "oob_score": True,
                "n_jobs": self._coerce_int(self.n_jobs_var.get(), -1),
                "random_state": random_state,
            }

            self._update_robust_progress(1, "Entrenamiento Base (Baseline)",
                                         f"Paso 2/4: Partición train/test ({n_features_orig} variables, {len(clean_data)} obs)...",
                                         1, 4, time.perf_counter() - start_time)

            # Holdout split
            requested_test_size = self._coerce_float(self.test_size_var.get(), 0.25, minimum=0.05, maximum=0.5)
            event_values_int = (pd.to_numeric(clean_data[event_col], errors="coerce").fillna(0) > 0).astype(int)
            test_size, stratify_values, split_warnings = self._resolve_holdout_split_settings(
                clean_data, event_col, requested_test_size,
                min_train_rows=max(8, n_features_orig + 2), min_test_rows=2,
                prefer_stratify=bool(self.stratify_event_var.get()), context_label="Robust optimization")
            warnings_list.extend(split_warnings)

            X_train, X_test, y_train, y_test = train_test_split(
                X_encoded, y_structured, test_size=test_size, random_state=random_state, stratify=stratify_values)

            self._update_robust_progress(1, "Entrenamiento Base (Baseline)",
                                         f"Paso 3/4: Entrenando RSF baseline (300 árboles, {len(X_train)} obs train)...",
                                         2, 4, time.perf_counter() - start_time)

            baseline_model = RandomSurvivalForest(**baseline_params)
            baseline_model.fit(X_train, y_train)
            self._flush_tuning_dialog_events()

            self._update_robust_progress(1, "Entrenamiento Base (Baseline)",
                                         f"Paso 4/4: Calculando {opt_metric_label} baseline...",
                                         3, 4, time.perf_counter() - start_time)

            tau = None
            if opt_metric_code == "uno":
                tau = self._resolve_tau_for_uno(y_train, y_test=y_test, raise_on_error=True)
            eval_times = self._build_evaluation_time_grid(y_train, y_test, tau=(tau if opt_metric_code == "uno" else None))

            baseline_metric = None
            if opt_metric_code == "harrell":
                baseline_metric = self._compute_c_index(y_test, baseline_model.predict(X_test)) if len(X_test) > 0 else None
            elif opt_metric_code == "uno":
                baseline_metric = self._compute_c_uno_score(baseline_model, X_train, y_train, X_test, y_test, tau)
            elif opt_metric_code == "ibs":
                baseline_metric, _ibs_km_base, _bss_base = self._compute_ibs_and_bss(
                    baseline_model, X_train, y_train, X_test, y_test, eval_times)
            elif opt_metric_code == "brier":
                if len(X_test) > 0 and eval_times is not None and callable(brier_score):
                    try:
                        _surv_fns = baseline_model.predict_survival_function(X_test)
                        _surv_mtx = np.asarray([fn(eval_times) for fn in _surv_fns], dtype=float)
                        _brier_out = brier_score(y_train, y_test, _surv_mtx, eval_times)
                        if isinstance(_brier_out, tuple) and len(_brier_out) >= 2:
                            _, _brier_vals = _brier_out
                            _brier_arr = np.asarray(_brier_vals, dtype=float)
                            if _brier_arr.size > 0 and np.isfinite(_brier_arr).any():
                                baseline_metric = float(_brier_arr[len(_brier_arr) // 2])
                    except Exception:
                        baseline_metric = None

            report_lines.append("FASE 1: Entrenamiento Base")
            report_lines.append("-" * 40)
            report_lines.append(f"  Modo limpieza VIMP: {vimp_mode_label}")
            report_lines.append(f"  Métrica de optimización robusta: {opt_metric_label}")
            report_lines.append(f"  Poda por correlación: {corr_threshold_label}")
            report_lines.append(f"  Árboles candidatos: {', '.join(str(x) for x in tree_candidates)}")
            report_lines.append(f"  Variables iniciales: {n_features_orig} ({len(covariates)} covariables originales)")
            report_lines.append(f"  Observaciones: {len(clean_data)} (train={len(X_train)}, test={len(X_test)})")
            report_lines.append(f"  {opt_metric_label} baseline: {self._format_metric(baseline_metric)}")
            report_lines.append(f"  OOB baseline: {self._format_metric(getattr(baseline_model, 'oob_score_', None))}")
            report_lines.append("")
            phase_log["phase1_metric"] = baseline_metric
            phase_log["vimp_mode"] = vimp_mode_label
            phase_log["opt_metric_label"] = opt_metric_label

            self._update_robust_progress(1, "Entrenamiento Base",
                                         f"Baseline completado. {opt_metric_label}={self._format_metric(baseline_metric)}, "
                                         f"OOB={self._format_metric(getattr(baseline_model, 'oob_score_', None))}",
                                         4, 4, time.perf_counter() - start_time)

            # ── PHASE 2: VIMP Bootstrap noise elimination ─────────────
            if self._tuning_cancel_requested:
                raise InterruptedError("Cancelado por el usuario.")
            n_vimp_bootstrap = 60
            self._update_robust_progress(2, "Eliminación de Ruido (VIMP Bootstrap)",
                                         f"Calculando VIMP con {n_vimp_bootstrap} iteraciones bootstrap sobre {n_features_orig} variables...",
                                         0, n_vimp_bootstrap, time.perf_counter() - start_time)

            def _vimp_progress(completed, total):
                self._update_robust_progress(
                    2, "Eliminación de Ruido (VIMP Bootstrap)",
                    f"Bootstrap {completed}/{total} completado ({completed * 100 // total}%)",
                    completed, total, time.perf_counter() - start_time)

            vimp_df = self._compute_vimp_bootstrap(baseline_model, X_train, y_train,
                                                    n_bootstrap=n_vimp_bootstrap, n_repeats=5,
                                                    random_state=random_state,
                                                    progress_callback=_vimp_progress)

            retained_features, removed_vimp, phase2_criterion = self._select_vimp_features(vimp_df, vimp_mode_code)

            report_lines.append("FASE 2: Eliminación de Ruido (VIMP Bootstrap IC95%)")
            report_lines.append("-" * 40)
            report_lines.append(f"  Criterio ({vimp_mode_label}): {phase2_criterion}")
            report_lines.append(f"  Variables evaluadas: {len(vimp_df)}")
            report_lines.append(f"  Variables eliminadas ({len(removed_vimp)}): {', '.join(removed_vimp) if removed_vimp else 'ninguna'}")
            report_lines.append(f"  Variables retenidas ({len(retained_features)}): {', '.join(retained_features)}")
            for _, row in vimp_df.iterrows():
                status = "✓" if row["feature"] in retained_features else "✗"
                report_lines.append(
                    f"    {status} {row['feature']}: VIMP={row['importance']:.4f} "
                    f"IC95%=({row['importance_lower']:.4f}, {row['importance_upper']:.4f})"
                )
            report_lines.append("")
            phase_log["phase2_removed"] = removed_vimp
            phase_log["phase2_retained"] = retained_features

            self._update_robust_progress(2, "Eliminación de Ruido", f"{len(removed_vimp)} variables eliminadas.",
                                         100, 100, time.perf_counter() - start_time)

            # Filter X to retained features
            X_train_f = X_train[retained_features].copy()
            X_test_f = X_test[retained_features].copy()

            # ── PHASE 3: Hyperparameter Search (mtry + nodesize + trees) ─────
            if self._tuning_cancel_requested:
                raise InterruptedError("Cancelado por el usuario.")

            n_feat = X_train_f.shape[1]
            # Generate random search candidates for mtry and nodesize
            rng = np.random.default_rng(random_state)
            n_search = min(80, max(30, n_feat * 4))
            mtry_options = sorted(set([
                max(1, int(np.sqrt(n_feat))),
                max(1, int(np.log2(max(n_feat, 2)))),
                max(1, n_feat // 3),
                max(1, n_feat // 2),
                n_feat,
            ]))
            nodesize_options = [1, 3, 5, 8, 10, 15, 20, 30]
            nodesize_options = [ns for ns in nodesize_options if ns < len(X_train_f) // 3]
            if not nodesize_options:
                nodesize_options = [1, 3, 5]

            split_mult_options = [2.0, 2.5, 3.0, 4.0]
            _split_cap_ph3 = len(X_train_f) // 2
            search_candidates = []
            for _ in range(n_search):
                mtry = int(rng.choice(mtry_options))
                nodesize = int(rng.choice(nodesize_options))
                trees = int(rng.choice(tree_candidates))
                sm = float(rng.choice(split_mult_options))
                if max(2, int(nodesize * sm)) > _split_cap_ph3:
                    continue
                search_candidates.append({
                    "max_features": mtry,
                    "min_samples_leaf": nodesize,
                    "n_estimators": trees,
                    "split_multiplier": sm,
                })
            # Remove duplicates
            seen = set()
            unique_candidates = []
            for c in search_candidates:
                key = (c["max_features"], c["min_samples_leaf"], c["n_estimators"], c["split_multiplier"])
                if key not in seen:
                    seen.add(key)
                    unique_candidates.append(c)
            search_candidates = unique_candidates

            self._update_robust_progress(3, "Búsqueda de Hiperparámetros (Random Search)",
                                         f"Evaluando {len(search_candidates)} combinaciones de mtry × nodesize...",
                                         0, len(search_candidates), time.perf_counter() - start_time)

            phase3_results = []
            _p3_total_trees = sum(c["n_estimators"] for c in search_candidates)
            _p3_trees_done = 0
            for idx, cand in enumerate(search_candidates):
                if self._tuning_cancel_requested:
                    raise InterruptedError("Cancelado por el usuario.")
                try:
                    params = {**baseline_params, "n_estimators": cand["n_estimators"], "max_features": cand["max_features"],
                              "min_samples_leaf": cand["min_samples_leaf"],
                              "min_samples_split": max(2, int(cand["min_samples_leaf"] * cand["split_multiplier"]))}
                    cv_score = self._compute_c_uno_internal_cv(X_train_f, y_train, params, metric_code=opt_metric_code)
                    if cv_score is None:
                        cv_score = float("-inf") if opt_metric_higher_better else float("inf")
                    phase3_results.append({"params": params, "score": cv_score, "model": None})
                except Exception:
                    pass

                _p3_trees_done += cand["n_estimators"]
                _p3_elapsed = time.perf_counter() - start_time
                _p3_eta = ""
                if _p3_trees_done > 0 and _p3_trees_done < _p3_total_trees:
                    _p3_rem = (_p3_elapsed / _p3_trees_done) * (_p3_total_trees - _p3_trees_done)
                    _p3_eta = f" | ETA {self._format_elapsed_time(_p3_rem)}"
                _cand_mf = self._annotate_max_features_text(cand['max_features'])
                self._update_robust_progress(3, "Búsqueda de Hiperparámetros",
                                             f"Candidato {idx + 1}/{len(search_candidates)}: mtry={_cand_mf}, nodesize={cand['min_samples_leaf']}, trees={cand['n_estimators']}{_p3_eta}",
                                             idx + 1, len(search_candidates), _p3_elapsed)

            if not phase3_results:
                raise ValueError("No se pudo evaluar ningún candidato en la búsqueda de hiperparámetros.")

            phase3_results.sort(key=lambda r: r["score"], reverse=bool(opt_metric_higher_better))
            best_phase3 = phase3_results[0]

            report_lines.append(f"FASE 3: Búsqueda de Hiperparámetros ({opt_metric_label})")
            report_lines.append("-" * 40)
            report_lines.append(f"  Combinaciones evaluadas: {len(phase3_results)}")
            report_lines.append(f"  Métrica: {opt_metric_label}")
            for i, r in enumerate(phase3_results[:5]):
                report_lines.append(
                    f"    #{i + 1} trees={r['params']['n_estimators']}, mtry={self._annotate_max_features_text(r['params']['max_features'])}, "
                    f"nodesize={r['params']['min_samples_leaf']}, "
                    f"score={self._format_metric(r['score'])}")
            report_lines.append(f"  ▶ Mejor: trees={best_phase3['params']['n_estimators']}, mtry={self._annotate_max_features_text(best_phase3['params']['max_features'])}, "
                                f"nodesize={best_phase3['params']['min_samples_leaf']}, "
                                f"score={self._format_metric(best_phase3['score'])}")
            report_lines.append("")
            phase_log["phase3_best_trees"] = best_phase3["params"]["n_estimators"]
            phase_log["phase3_best_mtry"] = best_phase3["params"]["max_features"]
            phase_log["phase3_best_nodesize"] = best_phase3["params"]["min_samples_leaf"]
            phase_log["phase3_metric"] = best_phase3["score"]

            # ── PHASE 4: Fix transitional model ─────────────────────
            if self._tuning_cancel_requested:
                raise InterruptedError("Cancelado por el usuario.")
            _trans_mf = self._annotate_max_features_text(best_phase3['params']['max_features'])
            self._update_robust_progress(4, "Selección del Modelo Transitorio",
                                         f"Fijando mtry={_trans_mf}, nodesize={best_phase3['params']['min_samples_leaf']}...",
                                         0, 1, time.perf_counter() - start_time)

            transitional_params = copy.deepcopy(best_phase3["params"])
            # Retrain transitional model on full training set with best params
            transitional_model = RandomSurvivalForest(**transitional_params)
            transitional_model.fit(X_train_f, y_train)
            self._flush_tuning_dialog_events()

            report_lines.append("FASE 4: Modelo Transitorio")
            report_lines.append("-" * 40)
            report_lines.append(f"  árboles fijados: {transitional_params['n_estimators']}")
            report_lines.append(f"  mtry fijado: {self._annotate_max_features_text(transitional_params['max_features'])}")
            report_lines.append(f"  nodesize fijado: {transitional_params['min_samples_leaf']}")
            report_lines.append(f"  {opt_metric_label} transitorio: {self._format_metric(best_phase3['score'])}")
            report_lines.append("")

            _fixed_mf = self._annotate_max_features_text(transitional_params['max_features'])
            self._update_robust_progress(4, "Modelo Transitorio",
                                         f"Fijados: mtry={_fixed_mf}, nodesize={transitional_params['min_samples_leaf']}, trees={transitional_params['n_estimators']}",
                                         1, 1, time.perf_counter() - start_time)

            # ── PHASE 5: Minimal Depth Analysis ──────────────────────
            if self._tuning_cancel_requested:
                raise InterruptedError("Cancelado por el usuario.")
            n_retained = len(retained_features)
            self._update_robust_progress(5, "Análisis de Profundidad Mínima",
                                         f"Paso 1/3: Reentrenando modelo con {n_retained} variables y parámetros óptimos...",
                                         0, 3, time.perf_counter() - start_time)

            # Retrain with transitional params on all retained features
            mdepth_model = RandomSurvivalForest(**transitional_params)
            mdepth_model.fit(X_train_f, y_train)
            self._flush_tuning_dialog_events()

            self._update_robust_progress(5, "Análisis de Profundidad Mínima",
                                         f"Paso 2/3: Recorriendo {len(mdepth_model.estimators_)} árboles para calcular profundidad mínima...",
                                         1, 3, time.perf_counter() - start_time)

            mdepth_df = self._compute_minimal_depth(mdepth_model, list(X_train_f.columns))
            if corr_threshold is None:
                corr_pruned_features = list(mdepth_df["feature"].tolist())
                corr_removed_pairs = []
            else:
                corr_pruned_features, corr_removed_pairs = self._prune_correlated_features_by_depth(
                    X_train_f[mdepth_df["feature"].tolist()], mdepth_df, threshold=corr_threshold, min_keep=2)

            self._update_robust_progress(5, "Análisis de Profundidad Mínima",
                                         f"Paso 3/3: Ranking generado para {len(mdepth_df)} variables.",
                                         2, 3, time.perf_counter() - start_time)

            report_lines.append("FASE 5: Profundidad Mínima (Minimal Depth)")
            report_lines.append("-" * 40)
            report_lines.append(f"  Variables analizadas: {len(mdepth_df)}")
            report_lines.append(f"  Ranking por profundidad (menor = más importante):")
            for _, row in mdepth_df.iterrows():
                report_lines.append(
                    f"    {row['feature']}: prof_media={row['avg_minimal_depth']:.2f}, "
                    f"usado_en={row['trees_used_in']}/{len(mdepth_model.estimators_)} árboles ({row['usage_pct']}%)"
                )
            if corr_removed_pairs:
                report_lines.append(f"  Poda previa por correlación alta ({corr_threshold_label}):")
                for keep_feat, drop_feat, corr_val in corr_removed_pairs:
                    report_lines.append(f"    conservar {keep_feat} | eliminar {drop_feat} | rho={corr_val:.3f}")
            else:
                report_lines.append(f"  Poda por correlación: sin eliminaciones ({corr_threshold_label}).")
            report_lines.append("")

            best_feat = mdepth_df.iloc[0]['feature'] if len(mdepth_df) > 0 else '?'
            worst_feat = mdepth_df.iloc[-1]['feature'] if len(mdepth_df) > 0 else '?'
            self._update_robust_progress(5, "Profundidad Mínima",
                                         f"Completado. Mejor: {best_feat}, Peor: {worst_feat}",
                                         3, 3, time.perf_counter() - start_time)

            # ── PHASE 6: Adaptive Recursive Elimination ──────────────
            if self._tuning_cancel_requested:
                raise InterruptedError("Cancelado por el usuario.")

            # Order features by minimal depth (best first)
            ordered_features = [feat for feat in mdepth_df["feature"].tolist() if feat in corr_pruned_features]
            p = len(ordered_features)

            # Adaptive elimination block size
            if p > 100:
                block_pct = 0.05
            elif p > 20:
                block_pct = 0.10
            else:
                block_pct = None  # 1-by-1

            current_features = list(ordered_features)
            elimination_history = []

            # Compute initial C-Uno with all retained features
            c_uno_init = self._compute_c_uno_internal_cv(X_train_f[current_features], y_train,
                                                           transitional_params, metric_code=opt_metric_code)
            elimination_history.append({"n_features": len(current_features), "score": c_uno_init,
                                         "removed": [], "features": list(current_features)})

            max_elimination_steps = max(p - 2, 1)  # Keep at least 2 features
            self._update_robust_progress(6, "Eliminación Recursiva Adaptativa",
                                         f"Eliminando variables desde {p} (bloque adaptativo)...",
                                         0, max_elimination_steps, time.perf_counter() - start_time)

            best_c_uno_elim = c_uno_init if c_uno_init is not None else (float("-inf") if opt_metric_higher_better else float("inf"))
            best_feature_set = list(current_features)
            consecutive_drops = 0
            step = 0

            while len(current_features) > 2:
                if self._tuning_cancel_requested:
                    raise InterruptedError("Cancelado por el usuario.")

                if block_pct is not None:
                    n_remove = max(1, int(len(current_features) * block_pct))
                else:
                    n_remove = 1

                n_remove = min(n_remove, len(current_features) - 2)
                if n_remove <= 0:
                    break

                # Remove worst features (highest minimal depth = last in ordered list)
                to_remove = current_features[-n_remove:]
                candidate_features = [f for f in current_features if f not in to_remove]

                try:
                    c_uno_elim = self._compute_c_uno_internal_cv(
                        X_train_f[candidate_features], y_train, transitional_params, metric_code=opt_metric_code)
                except Exception:
                    c_uno_elim = None

                elimination_history.append({
                    "n_features": len(candidate_features),
                    "score": c_uno_elim,
                    "removed": to_remove,
                    "features": list(candidate_features),
                })

                step += 1
                self._update_robust_progress(6, "Eliminación Recursiva",
                                             f"{len(candidate_features)} variables restantes, "
                                             f"{opt_metric_label}={self._format_metric(c_uno_elim)}",
                                             step, max_elimination_steps, time.perf_counter() - start_time)

                c_val = c_uno_elim if c_uno_elim is not None else (float("-inf") if opt_metric_higher_better else float("inf"))
                better_or_equal = (c_val >= (best_c_uno_elim - 0.005)) if opt_metric_higher_better else (c_val <= (best_c_uno_elim + 0.005))
                if better_or_equal:  # tolerance
                    if opt_metric_higher_better:
                        best_c_uno_elim = max(best_c_uno_elim, c_val)
                    else:
                        best_c_uno_elim = min(best_c_uno_elim, c_val)
                    best_feature_set = list(candidate_features)
                    consecutive_drops = 0
                else:
                    consecutive_drops += 1

                current_features = list(candidate_features)

                # Stop if performance drops consistently
                if consecutive_drops >= 3:
                    break

            # Use one step back if the last step dropped
            final_features = list(best_feature_set)

            if vimp_mode_code == "hybrid":
                hybrid_features, hybrid_diag = self._apply_hybrid_signature_filter(
                    final_features, vimp_df, mdepth_df, min_keep=2)
                if hybrid_features:
                    final_features = list(hybrid_features)
                    report_lines.append("  Filtro híbrido 2/3 aplicado (IC95% inf>0, VIMP>0, depth top80%).")
                    report_lines.append(f"  Variables tras híbrido ({len(final_features)}): {', '.join(final_features)}")
                    phase_log["phase6_hybrid_details"] = hybrid_diag

            report_lines.append("FASE 6: Eliminación Recursiva Adaptativa")
            report_lines.append("-" * 40)
            report_lines.append(f"  Bloque de eliminación: {'1 a 1' if block_pct is None else f'{block_pct * 100:.0f}% por paso'}")
            report_lines.append(f"  Pasos de eliminación: {len(elimination_history) - 1}")
            report_lines.append(f"  Criterio de parada: ≥3 deterioros consecutivos >0.005 en {opt_metric_label}")
            report_lines.append(f"  Historial:")
            for h in elimination_history:
                removed_str = ", ".join(h["removed"]) if h["removed"] else "(inicio)"
                report_lines.append(f"    p={h['n_features']}: score={self._format_metric(h['score'])} | eliminadas: {removed_str}")
            report_lines.append(f"  ▶ Firma predictiva final ({len(final_features)} variables): {', '.join(final_features)}")
            report_lines.append("")
            phase_log["phase6_final_features"] = final_features
            phase_log["phase6_best_metric"] = best_c_uno_elim

            # Rebuild X with final features
            X_train_final = X_train_f[final_features].copy()
            X_test_final = X_test_f[final_features].copy()

            # ── PHASE 7: Fine Grid Search ────────────────────────────
            if self._tuning_cancel_requested:
                raise InterruptedError("Cancelado por el usuario.")

            n_feat_final = len(final_features)
            # Fine grid around best values
            best_trees = transitional_params["n_estimators"]
            best_mtry = transitional_params["max_features"]
            best_nodesize = transitional_params["min_samples_leaf"]

            tree_grid = sorted(set(tree_candidates + [
                max(50, best_trees - 150),
                max(50, best_trees - 75),
                best_trees,
                best_trees + 75,
                best_trees + 150,
            ]))
            tree_grid = sorted(tree_grid, key=lambda val: abs(val - best_trees))[:5]
            tree_grid = sorted(set(int(v) for v in tree_grid))

            mtry_grid = sorted(set([
                max(1, best_mtry - 2), max(1, best_mtry - 1), best_mtry,
                min(n_feat_final, best_mtry + 1), min(n_feat_final, best_mtry + 2),
                max(1, int(np.sqrt(n_feat_final))), max(1, int(np.log2(max(n_feat_final, 2)))),
                n_feat_final,
            ]))
            nodesize_grid = sorted(set([
                max(1, best_nodesize - 3), max(1, best_nodesize - 1), best_nodesize,
                best_nodesize + 2, best_nodesize + 5,
            ]))

            grid_mult = [2.0, 3.0]
            _split_cap_ph7 = len(X_train_final) // 2
            grid_candidates = []
            for nt in tree_grid:
                for mt in mtry_grid:
                    for ns in nodesize_grid:
                        for sm in grid_mult:
                            if max(2, int(ns * sm)) > _split_cap_ph7:
                                continue
                            grid_candidates.append({"n_estimators": nt, "max_features": mt, "min_samples_leaf": ns, "split_multiplier": sm})

            self._update_robust_progress(7, "Re-Optimización Final (Grid Search)",
                                         f"Evaluando {len(grid_candidates)} combinaciones finas...",
                                         0, len(grid_candidates), time.perf_counter() - start_time)

            phase7_results = []
            _p7_total_trees = sum(c["n_estimators"] for c in grid_candidates)
            _p7_trees_done = 0
            for idx, cand in enumerate(grid_candidates):
                if self._tuning_cancel_requested:
                    raise InterruptedError("Cancelado por el usuario.")
                try:
                    params = {**baseline_params, "n_estimators": cand["n_estimators"], "max_features": cand["max_features"],
                              "min_samples_leaf": cand["min_samples_leaf"],
                              "min_samples_split": max(2, int(cand["min_samples_leaf"] * cand["split_multiplier"]))}
                    c_uno = self._compute_c_uno_internal_cv(X_train_final, y_train, params, metric_code=opt_metric_code)
                    if c_uno is None:
                        c_uno = float("-inf") if opt_metric_higher_better else float("inf")
                    phase7_results.append({"params": params, "score": c_uno, "model": None})
                except Exception:
                    pass

                _p7_trees_done += cand["n_estimators"]
                _p7_elapsed = time.perf_counter() - start_time
                _p7_eta = ""
                if _p7_trees_done > 0 and _p7_trees_done < _p7_total_trees:
                    _p7_rem = (_p7_elapsed / _p7_trees_done) * (_p7_total_trees - _p7_trees_done)
                    _p7_eta = f" | ETA {self._format_elapsed_time(_p7_rem)}"
                _cand7_mf = self._annotate_max_features_text(cand['max_features'])
                self._update_robust_progress(7, "Grid Search Final",
                                             f"Candidato {idx + 1}/{len(grid_candidates)}: mtry={_cand7_mf}, nodesize={cand['min_samples_leaf']}, trees={cand['n_estimators']}{_p7_eta}",
                                             idx + 1, len(grid_candidates), _p7_elapsed)

            if not phase7_results:
                raise ValueError("No se pudo evaluar ningún candidato en la afinación final.")

            phase7_results.sort(key=lambda r: r["score"], reverse=bool(opt_metric_higher_better))
            best_final = phase7_results[0]
            final_params = best_final["params"]
            # Retrain final model on full training set with best params
            final_model = RandomSurvivalForest(**final_params)
            final_model.fit(X_train_final, y_train)
            self._flush_tuning_dialog_events()

            report_lines.append("FASE 7: Re-Optimización Final (Grid Search)")
            report_lines.append("-" * 40)
            report_lines.append(f"  Combinaciones evaluadas: {len(phase7_results)}")
            for i, r in enumerate(phase7_results[:5]):
                report_lines.append(
                    f"    #{i + 1} trees={r['params']['n_estimators']}, mtry={self._annotate_max_features_text(r['params']['max_features'])}, "
                    f"nodesize={r['params']['min_samples_leaf']}, "
                    f"score={self._format_metric(r['score'])}"
                )
            report_lines.append(f"  ▶ Mejor final: trees={final_params['n_estimators']}, mtry={self._annotate_max_features_text(final_params['max_features'])}, "
                                f"nodesize={final_params['min_samples_leaf']}, "
                                f"score={self._format_metric(best_final['score'])}")
            report_lines.append("")
            phase_log["phase7_metric"] = best_final["score"]
            phase_log["phase7_trees"] = final_params["n_estimators"]
            phase_log["phase7_mtry"] = final_params["max_features"]
            phase_log["phase7_nodesize"] = final_params["min_samples_leaf"]

            # ── PHASE 8: Final Validation & Calibration ──────────────
            if self._tuning_cancel_requested:
                raise InterruptedError("Cancelado por el usuario.")
            self._update_robust_progress(8, "Validación Final y Calibración",
                                         "Paso 1/4: Calculando C-Uno IPCW del modelo final...",
                                         0, 4, time.perf_counter() - start_time)

            # C-Uno final (informativo; no siempre es la métrica objetivo)
            c_uno_final = self._compute_c_uno_score(final_model, X_train_final, y_train, X_test_final, y_test, tau)
            self._update_robust_progress(8, "Validación Final",
                                         f"Paso 1/4 completado. C-Uno={self._format_metric(c_uno_final)}",
                                         1, 4, time.perf_counter() - start_time)

            # IBS and BSS
            ibs_model, ibs_km, bss = self._compute_ibs_and_bss(final_model, X_train_final, y_train,
                                                                 X_test_final, y_test, eval_times)
            self._update_robust_progress(8, "Validación Final",
                                         f"Paso 2/4 completado. IBS={self._format_metric(ibs_model)}, BSS={self._format_metric(bss)}",
                                         2, 4, time.perf_counter() - start_time)

            # Cross-validated C-index (final)
            train_event_values = (np.asarray(y_train["event"], dtype=bool).astype(int))
            cv_result = self._compute_cv_cindex(X_train_final, train_event_values, y_train, final_params,
                                                 force_enabled=True, return_values=True)
            cv_mean, cv_std, cv_values = (None, None, [])
            if isinstance(cv_result, tuple):
                if len(cv_result) == 3:
                    cv_mean, cv_std, cv_values = cv_result
                elif len(cv_result) == 2:
                    cv_mean, cv_std = cv_result
            cv_ci = self._compute_mean_confidence_interval(cv_values, clip_min=0.0, clip_max=1.0) if cv_values else None
            self._update_robust_progress(8, "Validación Final",
                                         f"Paso 3/4 completado. CV C-index={self._format_metric(cv_mean)}",
                                         3, 4, time.perf_counter() - start_time)

            # C-Antolini
            c_antolini = None
            if len(X_test_final) > 0:
                c_antolini = self._compute_c_antolini_score(
                    final_model, X_train_final, y_train, X_test_final, y_test, eval_times=eval_times, tau=tau)
            self._update_robust_progress(8, "Validación Final",
                                         f"Paso 4/4 completado. C-Antolini={self._format_metric(c_antolini)}",
                                         4, 4, time.perf_counter() - start_time)

            brier_mid_final = None
            if len(X_test_final) > 0 and eval_times is not None and callable(brier_score):
                try:
                    _surv_fns_fin = final_model.predict_survival_function(X_test_final)
                    _surv_mtx_fin = np.asarray([fn(eval_times) for fn in _surv_fns_fin], dtype=float)
                    _brier_fin = brier_score(y_train, y_test, _surv_mtx_fin, eval_times)
                    if isinstance(_brier_fin, tuple) and len(_brier_fin) >= 2:
                        _, _bvals_fin = _brier_fin
                        _barr_fin = np.asarray(_bvals_fin, dtype=float)
                        if _barr_fin.size > 0 and np.isfinite(_barr_fin).any():
                            brier_mid_final = float(_barr_fin[len(_barr_fin) // 2])
                except Exception:
                    brier_mid_final = None

            if opt_metric_code == "harrell":
                final_opt_metric_value = self._compute_c_index(y_test, final_model.predict(X_test_final)) if len(X_test_final) > 0 else None
            elif opt_metric_code == "uno":
                final_opt_metric_value = c_uno_final
            elif opt_metric_code == "ibs":
                final_opt_metric_value = ibs_model
            else:
                final_opt_metric_value = brier_mid_final

            total_elapsed = time.perf_counter() - start_time

            report_lines.append("FASE 8: Validación Final y Calibración")
            report_lines.append("-" * 40)
            report_lines.append(f"  === Discriminación ===")
            report_lines.append(f"  C-Uno (IPCW) holdout:   {self._format_metric(c_uno_final)}")
            report_lines.append(f"  C-Antolini (Ctd) medio: {self._format_metric(c_antolini)}")
            report_lines.append(f"  C-index CV medio:       {self._format_c_index_display(cv_mean, cv_ci, decimals=4)}")
            report_lines.append(f"  C-index CV DE:          {self._format_metric(cv_std)}")
            report_lines.append(f"  OOB score:              {self._format_metric(getattr(final_model, 'oob_score_', None))}")
            report_lines.append(f"  τ (tau):                {self._format_metric(tau)}")
            report_lines.append("")
            report_lines.append(f"  === Calibración ===")
            report_lines.append(f"  IBS modelo:             {self._format_metric(ibs_model)}")
            report_lines.append(f"  IBS Kaplan-Meier (nulo):{self._format_metric(ibs_km)}")
            report_lines.append(f"  Brier Skill Score (BSS):{self._format_metric(bss)}")

            robust_ibs_ci, robust_bss_ci = None, None
            try:
                robust_ibs_ci, robust_bss_ci = self._compute_ibs_bss_ci(
                    final_model, X_train_final, y_train, X_test_final, y_test, eval_times, random_state=random_state)
            except Exception:
                pass
            if robust_ibs_ci is not None:
                report_lines.append(f"  IBS 95% CI:             ({robust_ibs_ci[0]:.4f}, {robust_ibs_ci[1]:.4f})")
            if robust_bss_ci is not None:
                report_lines.append(f"  BSS 95% CI:             ({robust_bss_ci[0]:.4f}, {robust_bss_ci[1]:.4f})")

            if bss is not None:
                if bss > 0:
                    report_lines.append(f"  → El modelo predice MEJOR que asignar el riesgo promedio KM de la cohorte.")
                elif bss == 0:
                    report_lines.append(f"  → El modelo es equivalente al modelo nulo KM.")
                else:
                    report_lines.append(f"  → El modelo es PEOR que el modelo nulo KM. Requiere calibración externa.")
            report_lines.append("")

            # ── Summary ──────────────────────────────────────────────
            report_lines.append("=" * 60)
            report_lines.append("RESUMEN DE LA OPTIMIZACIÓN ROBUSTA")
            report_lines.append("=" * 60)
            report_lines.append(f"  Variables iniciales:     {n_features_orig}")
            report_lines.append(f"  Tras VIMP bootstrap:     {len(phase_log.get('phase2_retained', []))}")
            report_lines.append(f"  Firma predictiva final:  {len(final_features)}")
            report_lines.append(f"  Variables finales:       {', '.join(final_features)}")
            report_lines.append("")
            report_lines.append(f"  Hiperparámetros finales:")
            report_lines.append(f"    n_estimators:     {final_params.get('n_estimators', 300)}")
            report_lines.append(f"    max_features:     {self._annotate_max_features_text(final_params.get('max_features'))}")
            report_lines.append(f"    min_samples_leaf: {final_params.get('min_samples_leaf')}")
            report_lines.append(f"    min_samples_split:{final_params.get('min_samples_split')}")
            report_lines.append("")
            report_lines.append(
                f"  {opt_metric_label} baseline → final: "
                f"{self._format_metric(phase_log.get('phase1_metric'))} → {self._format_metric(final_opt_metric_value)}"
            )
            report_lines.append(f"  BSS: {self._format_metric(bss)}")
            h, rem = divmod(int(total_elapsed), 3600)
            m, s = divmod(rem, 60)
            report_lines.append(f"  Tiempo total: {h:02d}:{m:02d}:{s:02d}")
            report_lines.append("")
            if warnings_list:
                report_lines.append("Avisos de preparación:")
                for w in warnings_list:
                    report_lines.append(f"  - {w}")

            full_report = "\n".join(report_lines)

            # Store as model snapshot
            final_metrics = {
                "c_index_train": self._compute_c_index(y_train, final_model.predict(X_train_final)),
                "c_index_test": self._compute_c_index(y_test, final_model.predict(X_test_final)) if len(X_test_final) > 0 else None,
                "c_index_uno": c_uno_final,
                "c_index_antolini": c_antolini,
                "optimization_metric_code": opt_metric_code,
                "optimization_metric_label": opt_metric_label,
                "optimization_metric_value": final_opt_metric_value,
                "brier_mid": brier_mid_final,
                "c_index_cv_mean": cv_mean,
                "c_index_cv_std": cv_std,
                "c_index_cv_ci": cv_ci,
                "tau": tau,
                "oob_score": self._compute_oob_cindex(final_model, y_train),
                "ibs": ibs_model,
                "ibs_ci": robust_ibs_ci,
                "ibs_km": ibs_km,
                "bss": bss,
                "bss_ci": robust_bss_ci,
                "n_train": len(X_train_final),
                "n_test": len(X_test_final),
            }

            latest_brier_df = pd.DataFrame()
            latest_calibration_df = pd.DataFrame()
            latest_eval_time = None
            if len(X_test_final) > 0 and eval_times is not None and callable(brier_score):
                try:
                    surv_fns_final = final_model.predict_survival_function(X_test_final)
                    surv_matrix_final = np.asarray([fn(eval_times) for fn in surv_fns_final], dtype=float)
                    brier_result_final = brier_score(y_train, y_test, surv_matrix_final, eval_times)
                    if isinstance(brier_result_final, tuple) and len(brier_result_final) >= 2:
                        _, brier_values_final = brier_result_final
                        latest_brier_df = pd.DataFrame({"time": eval_times, "brier_score": brier_values_final})
                        mid_idx_final = len(eval_times) // 2
                        latest_eval_time = float(eval_times[mid_idx_final])
                        latest_calibration_df = self._summarize_calibration(
                            y_test, surv_matrix_final[:, mid_idx_final], latest_eval_time)
                except Exception:
                    pass

            full_preds = np.asarray(final_model.predict(X_encoded[final_features]), dtype=float)
            latest_survival_profiles = self._build_survival_profiles(final_model, X_encoded[final_features], full_preds)
            latest_prediction_df = clean_data[[duration_col, event_col]].copy()
            latest_prediction_df["risk_score"] = full_preds
            try:
                labels = ["Q1 bajo", "Q2 medio-bajo", "Q3 medio-alto", "Q4 alto"]
                qn = min(4, max(2, latest_prediction_df["risk_score"].nunique()))
                latest_prediction_df["risk_group"] = pd.qcut(
                    latest_prediction_df["risk_score"].rank(method="first"),
                    q=qn,
                    labels=labels[:qn],
                    duplicates="drop",
                )
            except Exception:
                latest_prediction_df["risk_group"] = "Grupo único"

            snapshot = {
                "label": f"RobustOpt #{len(self.saved_models) + 1}",
                "params": {
                    **copy.deepcopy(final_params),
                    "optimization_metric": opt_metric_label,
                    "tau_mode": self.tau_mode_var.get() if hasattr(self, "tau_mode_var") else "Percentil 90",
                    "tau_manual": self.tau_manual_var.get() if hasattr(self, "tau_manual_var") else "",
                },
                "metrics": final_metrics,
                "report_text": full_report,
                "model": final_model,
                "feature_importance_df": self._compute_permutation_importance(
                    final_model, X_test_final, y_test, random_state),
                "latest_prediction_df": latest_prediction_df.copy(deep=True),
                "latest_survival_profiles": copy.deepcopy(latest_survival_profiles),
                "latest_calibration_df": latest_calibration_df.copy(deep=True) if isinstance(latest_calibration_df, pd.DataFrame) else pd.DataFrame(),
                "latest_brier_df": latest_brier_df.copy(deep=True) if isinstance(latest_brier_df, pd.DataFrame) else pd.DataFrame(),
                "latest_eval_time": latest_eval_time,
                "latest_fit_dataframe": clean_data.copy(deep=True),
                "latest_duration_col": duration_col,
                "latest_event_col": event_col,
                "latest_covariates": list(covariates),
                "latest_encoded_columns": list(final_features),
                "latest_drop_first": bool(self.drop_first_var.get()),
                "variable_configs": copy.deepcopy(getattr(self, "variable_configs", {})),
            }
            self.saved_models.append(snapshot)
            self.active_saved_model_index = len(self.saved_models) - 1
            self._refresh_saved_models_tree()

            # Set current model state
            self.model = final_model
            self.feature_importance_df = snapshot["feature_importance_df"]
            self.latest_prediction_df = latest_prediction_df
            self.latest_survival_profiles = latest_survival_profiles
            self.latest_calibration_df = latest_calibration_df
            self.latest_brier_df = latest_brier_df
            self.latest_eval_time = latest_eval_time
            self.latest_fit_dataframe = clean_data
            self.latest_duration_col = duration_col
            self.latest_event_col = event_col
            self.latest_covariates = list(covariates)
            self.latest_encoded_columns = list(final_features)
            self.latest_report_text = full_report

            # Display report
            self.results_text.delete("1.0", tk.END)
            self.results_text.insert(tk.END, full_report)
            self.auto_tuning_status_var.set(
                f"Optimización Robusta: {opt_metric_label}={self._format_metric(final_opt_metric_value)}, "
                f"BSS={self._format_metric(bss)}, {len(final_features)} variables"
            )

            # Show summary popup
            self._show_robust_optimization_summary(phase_log, final_params, final_metrics, final_features,
                                                    total_elapsed, n_features_orig, elimination_history, vimp_df, mdepth_df)

        except InterruptedError:
            self.auto_tuning_status_var.set("Optimización Robusta cancelada por el usuario.")
            messagebox.showinfo("Cancelado", "La optimización robusta fue cancelada por el usuario.")
        except Exception as exc:
            self.auto_tuning_status_var.set(f"Optimización Robusta falló: {exc}")
            traceback_text = traceback.format_exc(limit=5)
            self.results_text.delete("1.0", tk.END)
            self.results_text.insert(tk.END, f"Error en Optimización Robusta:\n{exc}\n\n{traceback_text}")
            messagebox.showerror("Error", f"Error en la Optimización Robusta:\n{exc}")
        finally:
            self._close_auto_tuning_progress_dialog()
            self._tuning_cancel_requested = False

    def _show_robust_optimization_summary(self, phase_log, final_params, final_metrics,
                                           final_features, total_elapsed, n_features_orig,
                                           elimination_history, vimp_df, mdepth_df):
        """Show a comprehensive popup summarizing the robust optimization results."""
        parent_window = self.winfo_toplevel()
        popup = tk.Toplevel(parent_window)
        popup.title("Resumen — Optimización Robusta RSF")
        popup.geometry("700x620")
        popup.transient(parent_window)
        popup.resizable(True, True)

        container = ttk.Frame(popup, padding=14)
        container.pack(fill=tk.BOTH, expand=True)

        ttk.Label(container, text="Resumen de Optimización Robusta RSF",
                  font=("Segoe UI", 12, "bold")).pack(anchor="w", pady=(0, 8))

        text_widget = scrolledtext.ScrolledText(container, wrap=tk.WORD, font=("Consolas", 9), height=30)
        text_widget.pack(fill=tk.BOTH, expand=True, pady=(0, 8))

        lines = []
        h, rem = divmod(int(total_elapsed), 3600)
        m, s = divmod(rem, 60)
        lines.append(f"Tiempo total: {h:02d}:{m:02d}:{s:02d}")
        lines.append("")

        # Feature journey
        lines.append("═══ Recorrido de variables ═══")
        lines.append(f"  Iniciales:          {n_features_orig}")
        n_after_vimp = len(phase_log.get("phase2_retained", []))
        lines.append(f"  Tras VIMP bootstrap: {n_after_vimp} (eliminadas: {len(phase_log.get('phase2_removed', []))})")
        if phase_log.get("phase2_removed"):
            lines.append(f"    Eliminadas: {', '.join(phase_log['phase2_removed'])}")
        lines.append(f"  Firma final:         {len(final_features)}")
        lines.append(f"    → {', '.join(final_features)}")
        lines.append("")

        opt_metric_label = str(phase_log.get("opt_metric_label", "Métrica"))

        # Elimination curve
        lines.append("═══ Curva de eliminación adaptativa ═══")
        for h_entry in elimination_history:
            marker = " ◀ óptimo" if h_entry["features"] == final_features else ""
            lines.append(f"  p={h_entry['n_features']:3d} | {opt_metric_label}={self._format_metric(h_entry.get('score'))}{marker}")
        lines.append("")

        # Hyperparameters
        lines.append("═══ Hiperparámetros finales ═══")
        lines.append(f"  n_estimators:     {final_params.get('n_estimators', 300)}")
        lines.append(f"  max_features:     {self._annotate_max_features_text(final_params.get('max_features'))}")
        lines.append(f"  min_samples_leaf: {final_params.get('min_samples_leaf')}")
        lines.append(f"  min_samples_split:{final_params.get('min_samples_split')}")
        lines.append("")

        # Performance
        lines.append("═══ Rendimiento final ═══")
        lines.append(f"  C-Uno (IPCW):           {self._format_metric(final_metrics.get('c_index_uno'))}")
        lines.append(f"  C-Antolini (Ctd):       {self._format_metric(final_metrics.get('c_index_antolini'))}")
        cv_ci = final_metrics.get("c_index_cv_ci")
        lines.append(f"  C-index CV medio:       {self._format_c_index_display(final_metrics.get('c_index_cv_mean'), cv_ci, decimals=4)}")
        lines.append(f"  OOB score:              {self._format_metric(final_metrics.get('oob_score'))}")
        lines.append("")

        # Calibration
        lines.append("═══ Calibración ═══")
        lines.append(f"  IBS modelo:             {self._format_c_index_display(final_metrics.get('ibs'), final_metrics.get('ibs_ci'), decimals=4)}")
        lines.append(f"  IBS Kaplan-Meier:       {self._format_metric(final_metrics.get('ibs_km'))}")
        bss = final_metrics.get("bss")
        lines.append(f"  BSS (1 - IBS/IBS_KM):  {self._format_c_index_display(bss, final_metrics.get('bss_ci'), decimals=4)}")
        if bss is not None:
            if bss > 0.1:
                lines.append(f"  → Excelente: el modelo supera significativamente al KM nulo.")
            elif bss > 0:
                lines.append(f"  → El modelo supera al KM nulo (mejora marginal).")
            elif bss > -0.05:
                lines.append(f"  → El modelo es similar al KM nulo. Considerar más datos o recalibración.")
            else:
                lines.append(f"  → El modelo es peor que el KM nulo. La calibración debe revisarse.")
        lines.append("")

        # C-Uno evolution
        lines.append(f"═══ Evolución de {opt_metric_label} por fase ═══")
        lines.append(f"  Fase 1 (baseline):      {self._format_metric(phase_log.get('phase1_metric'))}")
        lines.append(f"  Fase 3 (random search):  {self._format_metric(phase_log.get('phase3_metric'))}")
        lines.append(f"  Fase 6 (tras eliminación):{self._format_metric(phase_log.get('phase6_best_metric'))}")
        lines.append(f"  Fase 7 (grid final):     {self._format_metric(phase_log.get('phase7_metric'))}")

        text_widget.insert(tk.END, "\n".join(lines))
        text_widget.configure(state=tk.DISABLED)

        btn_frame = ttk.Frame(container)
        btn_frame.pack(fill=tk.X)
        ttk.Button(btn_frame, text="Cerrar", command=popup.destroy).pack(side=tk.RIGHT, padx=5)
        ttk.Button(btn_frame, text="Copiar al portapapeles",
                   command=lambda: (popup.clipboard_clear(), popup.clipboard_append("\n".join(lines)))).pack(side=tk.RIGHT, padx=5)

    # ═══════════════════════════════════════════════════════════
    #  NESTED CV ROBUST OPTIMIZATION (Architecture B)
    # ═══════════════════════════════════════════════════════════

    def run_nested_cv_optimization(self):
        """Execute Nested CV Robust RSF Optimization (Architecture B).
        Outer K-fold loop with full 8-phase pipeline per fold.
        Final consensus model trained on ALL data with variables selected in ≥60% of folds."""
        if not SKSURV_AVAILABLE:
            messagebox.showerror("RSF no disponible",
                                 f"Instala scikit-survival.\n\n{SKSURV_IMPORT_ERROR}")
            return
        if self.data is None:
            messagebox.showerror("Error", "Cargue datos o use el dataset compartido primero.")
            return

        filtered_data = self.filter_component.apply_filters()
        if filtered_data is None or filtered_data.empty:
            messagebox.showerror("Error", "No hay datos disponibles tras aplicar filtros.")
            return

        duration_col = self.duration_var.get().strip()
        event_col = self.event_var.get().strip()
        selected_indices = self.covariates_listbox.curselection()
        covariates = [self.covariates_listbox.get(i) for i in selected_indices]
        if not duration_col or not event_col or not covariates:
            messagebox.showerror("Error", "Seleccione tiempo, evento y al menos una covariable.")
            return

        self._tuning_cancel_requested = False
        start_time = time.perf_counter()
        self._open_robust_progress_dialog(total_phases=8)

        K_OUTER = 5
        n_vimp_bootstrap = 60
        vimp_mode_code, vimp_mode_label = self._resolve_vimp_cleaning_mode()
        cv_metric_code, cv_metric_label = self._resolve_cv_metric_choice()
        corr_threshold, corr_threshold_label = self._resolve_correlation_pruning_threshold()

        try:
            # ── Prepare data ──
            clean_data, X_encoded, y_structured, warnings_list = self._prepare_dataframe_for_rsf(
                filtered_data, duration_col, event_col, covariates)

            random_state = self._parse_random_seeds()[0]
            n_features_orig = X_encoded.shape[1]
            event_ints = np.asarray(y_structured["event"], dtype=bool).astype(int)

            tree_candidates = self._resolve_robust_tree_candidates(len(clean_data))
            baseline_tree_count = self._coerce_int(self.n_estimators_var.get(), 300, minimum=10)

            baseline_params = {
                "n_estimators": baseline_tree_count, "max_features": "sqrt",
                "min_samples_leaf": 10, "min_samples_split": 20,
                "bootstrap": True, "oob_score": True,
                "n_jobs": self._coerce_int(self.n_jobs_var.get(), -1),
                "random_state": random_state,
            }

            outer_cv = StratifiedKFold(n_splits=K_OUTER, shuffle=True, random_state=random_state)
            fold_results = []
            all_fold_features = []

            for fold_idx, (train_ext_idx, test_ext_idx) in enumerate(outer_cv.split(X_encoded, event_ints)):
                if self._tuning_cancel_requested:
                    raise InterruptedError("Cancelado por el usuario.")

                fl = f"[Fold {fold_idx + 1}/{K_OUTER}]"
                X_tr_ext = X_encoded.iloc[train_ext_idx].reset_index(drop=True)
                X_te_ext = X_encoded.iloc[test_ext_idx].reset_index(drop=True)
                y_tr_ext = y_structured[train_ext_idx]
                y_te_ext = y_structured[test_ext_idx]

                # ── Phase 1: Baseline ──
                self._update_robust_progress(1, f"{fl} Baseline",
                    f"Entrenando baseline ({len(X_tr_ext)} train, {len(X_te_ext)} test)...",
                    0, 1, time.perf_counter() - start_time)

                bl_model = RandomSurvivalForest(**baseline_params)
                bl_model.fit(X_tr_ext, y_tr_ext)
                self._flush_tuning_dialog_events()
                tau = self._resolve_tau(y_tr_ext, y_te_ext)
                eval_times = self._build_evaluation_time_grid(y_tr_ext, y_te_ext, tau=tau)

                self._update_robust_progress(1, f"{fl} Baseline", "Baseline completado.",
                    1, 1, time.perf_counter() - start_time)

                # ── Phase 2: VIMP bootstrap ──
                if self._tuning_cancel_requested:
                    raise InterruptedError("Cancelado por el usuario.")

                def _vimp_prog(completed, total, _fl=fl):
                    self._update_robust_progress(2, f"{_fl} VIMP Bootstrap",
                        f"Bootstrap {completed}/{total}", completed, total,
                        time.perf_counter() - start_time)

                vimp_df = self._compute_vimp_bootstrap(bl_model, X_tr_ext, y_tr_ext,
                    n_bootstrap=n_vimp_bootstrap, n_repeats=5,
                    random_state=random_state + fold_idx,
                    progress_callback=_vimp_prog)

                retained_features, _, _ = self._select_vimp_features(vimp_df, vimp_mode_code)

                X_tr_f = X_tr_ext[retained_features].copy()

                # ── Phase 3: Random Search (internal CV) ──
                if self._tuning_cancel_requested:
                    raise InterruptedError("Cancelado por el usuario.")

                n_feat = len(retained_features)
                rng = np.random.default_rng(random_state + fold_idx)
                n_search = min(80, max(30, n_feat * 4))
                mtry_opts = sorted(set([max(1, int(np.sqrt(n_feat))),
                                        max(1, int(np.log2(max(n_feat, 2)))),
                                        max(1, n_feat // 3), max(1, n_feat // 2), n_feat]))
                ns_opts = [ns for ns in [1, 3, 5, 8, 10, 15, 20, 30] if ns < len(X_tr_f) // 3]
                if not ns_opts:
                    ns_opts = [1, 3, 5]

                split_mult_opts = [2.0, 2.5, 3.0, 4.0]
                search_cands = []
                seen_combos = set()
                for _ in range(n_search):
                    mt = int(rng.choice(mtry_opts))
                    ns = int(rng.choice(ns_opts))
                    nt = int(rng.choice(tree_candidates))
                    sm = float(rng.choice(split_mult_opts))
                    if (nt, mt, ns, sm) not in seen_combos:
                        seen_combos.add((nt, mt, ns, sm))
                        search_cands.append({"n_estimators": nt, "max_features": mt, "min_samples_leaf": ns, "split_multiplier": sm})

                p3_results = []
                _np3_total_trees = sum(c["n_estimators"] for c in search_cands)
                _np3_trees_done = 0
                for idx, cand in enumerate(search_cands):
                    if self._tuning_cancel_requested:
                        raise InterruptedError("Cancelado por el usuario.")
                    try:
                        params = {**baseline_params, "n_estimators": cand["n_estimators"], "max_features": cand["max_features"],
                                  "min_samples_leaf": cand["min_samples_leaf"],
                                  "min_samples_split": max(2, int(cand["min_samples_leaf"] * cand["split_multiplier"]))}
                        c_uno = self._compute_c_uno_internal_cv(X_tr_f, y_tr_ext, params)
                        p3_results.append({"params": params, "c_uno": c_uno if c_uno is not None else -1.0})
                    except Exception:
                        pass
                    _np3_trees_done += cand["n_estimators"]
                    _np3_elapsed = time.perf_counter() - start_time
                    _np3_eta = ""
                    if _np3_trees_done > 0 and _np3_trees_done < _np3_total_trees:
                        _np3_rem = (_np3_elapsed / _np3_trees_done) * (_np3_total_trees - _np3_trees_done)
                        _np3_eta = f" | ETA {self._format_elapsed_time(_np3_rem)}"
                    _ncand_mf = self._annotate_max_features_text(cand['max_features'])
                    self._update_robust_progress(3, f"{fl} Random Search",
                        f"Candidato {idx + 1}/{len(search_cands)}: mtry={_ncand_mf}, nodesize={cand['min_samples_leaf']}, trees={cand['n_estimators']}{_np3_eta}",
                        idx + 1, len(search_cands), _np3_elapsed)

                if not p3_results:
                    p3_results = [{"params": baseline_params, "c_uno": -1.0}]
                p3_results.sort(key=lambda r: r["c_uno"], reverse=True)
                trans_params = copy.deepcopy(p3_results[0]["params"])

                # ── Phase 4: Fix transitional ──
                self._update_robust_progress(4, f"{fl} Modelo Transitorio",
                    "Fijando parámetros...", 1, 1, time.perf_counter() - start_time)

                # ── Phase 5: Minimal Depth ──
                if self._tuning_cancel_requested:
                    raise InterruptedError("Cancelado por el usuario.")
                self._update_robust_progress(5, f"{fl} Profundidad Mínima",
                    "Calculando...", 0, 1, time.perf_counter() - start_time)

                mdepth_mdl = RandomSurvivalForest(**trans_params)
                mdepth_mdl.fit(X_tr_f, y_tr_ext)
                self._flush_tuning_dialog_events()
                mdepth_df = self._compute_minimal_depth(mdepth_mdl, list(X_tr_f.columns))
                if corr_threshold is None:
                    corr_pruned_features = list(mdepth_df["feature"].tolist())
                else:
                    corr_pruned_features, _corr_removed_pairs = self._prune_correlated_features_by_depth(
                        X_tr_f[mdepth_df["feature"].tolist()], mdepth_df, threshold=corr_threshold, min_keep=2)

                self._update_robust_progress(5, f"{fl} Profundidad Mínima",
                    "Completado.", 1, 1, time.perf_counter() - start_time)

                # ── Phase 6: Recursive Elimination ──
                if self._tuning_cancel_requested:
                    raise InterruptedError("Cancelado por el usuario.")

                ordered_feats = [feat for feat in mdepth_df["feature"].tolist() if feat in corr_pruned_features]
                p = len(ordered_feats)
                block_pct = 0.05 if p > 100 else (0.10 if p > 20 else None)
                cur_feats = list(ordered_feats)
                max_elim = max(p - 2, 1)

                c_init = self._compute_c_uno_internal_cv(X_tr_f[cur_feats], y_tr_ext, trans_params)
                best_c_elim = c_init if c_init is not None else -1.0
                best_feat_set = list(cur_feats)
                consec_drops = 0
                elim_step = 0

                while len(cur_feats) > 2:
                    if self._tuning_cancel_requested:
                        raise InterruptedError("Cancelado por el usuario.")
                    n_rem = max(1, int(len(cur_feats) * block_pct)) if block_pct else 1
                    n_rem = min(n_rem, len(cur_feats) - 2)
                    if n_rem <= 0:
                        break
                    to_rem = cur_feats[-n_rem:]
                    cand_feats = [f for f in cur_feats if f not in to_rem]
                    try:
                        c_elim = self._compute_c_uno_internal_cv(
                            X_tr_f[cand_feats], y_tr_ext, trans_params)
                    except Exception:
                        c_elim = None
                    elim_step += 1
                    self._update_robust_progress(6, f"{fl} Eliminación",
                        f"{len(cand_feats)} vars, C-Uno={self._format_metric(c_elim)}",
                        elim_step, max_elim, time.perf_counter() - start_time)
                    c_val = c_elim if c_elim is not None else -1.0
                    if c_val >= best_c_elim - 0.005:
                        best_c_elim = max(best_c_elim, c_val)
                        best_feat_set = list(cand_feats)
                        consec_drops = 0
                    else:
                        consec_drops += 1
                    cur_feats = list(cand_feats)
                    if consec_drops >= 3:
                        break

                fold_final_features = list(best_feat_set)
                if vimp_mode_code == "hybrid":
                    hybrid_features, _ = self._apply_hybrid_signature_filter(
                        fold_final_features, vimp_df, mdepth_df, min_keep=2)
                    if hybrid_features:
                        fold_final_features = list(hybrid_features)
                X_tr_final = X_tr_f[fold_final_features].copy()

                # ── Phase 7: Grid Search ──
                if self._tuning_cancel_requested:
                    raise InterruptedError("Cancelado por el usuario.")

                nff = len(fold_final_features)
                b_trees = trans_params["n_estimators"]
                b_mtry = trans_params["max_features"]
                b_ns = trans_params["min_samples_leaf"]
                tree_g = sorted(set(tree_candidates + [
                    max(50, b_trees - 150),
                    max(50, b_trees - 75),
                    b_trees,
                    b_trees + 75,
                    b_trees + 150,
                ]))
                tree_g = sorted(tree_g, key=lambda val: abs(val - b_trees))[:5]
                tree_g = sorted(set(int(v) for v in tree_g))
                mtry_g = sorted(set([max(1, b_mtry - 2), max(1, b_mtry - 1), b_mtry,
                                     min(nff, b_mtry + 1), min(nff, b_mtry + 2),
                                     max(1, int(np.sqrt(nff))), max(1, int(np.log2(max(nff, 2)))), nff]))
                ns_g = sorted(set([max(1, b_ns - 3), max(1, b_ns - 1), b_ns, b_ns + 2, b_ns + 5]))
                grid_mult = [2.0, 3.0]
                grid_cands = [{"n_estimators": nt, "max_features": mt, "min_samples_leaf": ns, "split_multiplier": sm}
                              for nt in tree_g for mt in mtry_g for ns in ns_g for sm in grid_mult]

                p7_results = []
                _np7_total_trees = sum(c["n_estimators"] for c in grid_cands)
                _np7_trees_done = 0
                for idx, cand in enumerate(grid_cands):
                    if self._tuning_cancel_requested:
                        raise InterruptedError("Cancelado por el usuario.")
                    try:
                        params = {**baseline_params, "n_estimators": cand["n_estimators"], "max_features": cand["max_features"],
                                  "min_samples_leaf": cand["min_samples_leaf"],
                                  "min_samples_split": max(2, int(cand["min_samples_leaf"] * cand["split_multiplier"]))}
                        c_uno = self._compute_c_uno_internal_cv(X_tr_final, y_tr_ext, params)
                        p7_results.append({"params": params, "c_uno": c_uno if c_uno is not None else -1.0})
                    except Exception:
                        pass
                    _np7_trees_done += cand["n_estimators"]
                    _np7_elapsed = time.perf_counter() - start_time
                    _np7_eta = ""
                    if _np7_trees_done > 0 and _np7_trees_done < _np7_total_trees:
                        _np7_rem = (_np7_elapsed / _np7_trees_done) * (_np7_total_trees - _np7_trees_done)
                        _np7_eta = f" | ETA {self._format_elapsed_time(_np7_rem)}"
                    _ncand7_mf = self._annotate_max_features_text(cand['max_features'])
                    self._update_robust_progress(7, f"{fl} Grid Search",
                        f"Candidato {idx + 1}/{len(grid_cands)}: mtry={_ncand7_mf}, nodesize={cand['min_samples_leaf']}, trees={cand['n_estimators']}{_np7_eta}",
                        idx + 1, len(grid_cands), _np7_elapsed)

                if not p7_results:
                    p7_results = [{"params": trans_params, "c_uno": -1.0}]
                p7_results.sort(key=lambda r: r["c_uno"], reverse=True)
                fold_final_params = p7_results[0]["params"]

                # ── Phase 8: Validate on external test fold ──
                if self._tuning_cancel_requested:
                    raise InterruptedError("Cancelado por el usuario.")
                self._update_robust_progress(8, f"{fl} Validación Externa",
                    "Entrenando y evaluando en fold externo...",
                    0, 3, time.perf_counter() - start_time)

                fold_model = RandomSurvivalForest(**fold_final_params)
                fold_model.fit(X_tr_final, y_tr_ext)
                self._flush_tuning_dialog_events()
                X_te_final = X_te_ext[fold_final_features].copy()

                c_uno_final = self._compute_c_uno_score(fold_model, X_tr_final, y_tr_ext,
                                                         X_te_final, y_te_ext, tau)
                self._update_robust_progress(8, f"{fl} Validación",
                    f"C-Uno={self._format_metric(c_uno_final)}",
                    1, 3, time.perf_counter() - start_time)

                ibs_model, ibs_km, bss = self._compute_ibs_and_bss(
                    fold_model, X_tr_final, y_tr_ext, X_te_final, y_te_ext, eval_times)
                self._update_robust_progress(8, f"{fl} Validación",
                    f"BSS={self._format_metric(bss)}",
                    2, 3, time.perf_counter() - start_time)

                c_antolini = self._compute_c_antolini_score(
                    fold_model, X_tr_final, y_tr_ext, X_te_final, y_te_ext, eval_times=eval_times, tau=tau)

                self._update_robust_progress(8, f"{fl} Validación",
                    f"Fold {fold_idx + 1} completado.", 3, 3, time.perf_counter() - start_time)

                fold_results.append({
                    "fold": fold_idx + 1, "c_uno": c_uno_final, "c_antolini": c_antolini,
                    "ibs": ibs_model, "ibs_km": ibs_km, "bss": bss,
                    "features": fold_final_features, "n_features": len(fold_final_features),
                    "params": copy.deepcopy(fold_final_params),
                    "n_train": len(X_tr_ext), "n_test": len(X_te_ext),
                    "vimp_removed": len(vimp_df) - len(retained_features),
                })
                all_fold_features.append(set(fold_final_features))

            # ═══ Consensus Analysis ═══
            total_elapsed = time.perf_counter() - start_time

            all_vars = set()
            for fs in all_fold_features:
                all_vars.update(fs)

            var_freq = {}
            for v in sorted(all_vars):
                var_freq[v] = sum(1 for fs in all_fold_features if v in fs)

            consensus_threshold = max(1, int(np.ceil(K_OUTER * 0.6)))
            consensus_vars = sorted([v for v, c in var_freq.items() if c >= consensus_threshold],
                                    key=lambda v: -var_freq[v])
            intersection_vars = sorted(set.intersection(*all_fold_features)) if all_fold_features else []

            c_uno_vals = [r["c_uno"] for r in fold_results if r["c_uno"] is not None]
            ibs_vals = [r["ibs"] for r in fold_results if r["ibs"] is not None]
            bss_vals = [r["bss"] for r in fold_results if r["bss"] is not None]
            c_ant_vals = [r["c_antolini"] for r in fold_results if r["c_antolini"] is not None]

            if not consensus_vars:
                consensus_vars = sorted(all_vars)[:max(3, len(all_vars))]

            # Train consensus model on ALL data
            X_consensus = X_encoded[consensus_vars].copy()
            best_fold = max(fold_results, key=lambda r: r["c_uno"] if r["c_uno"] is not None else -1)
            consensus_params = copy.deepcopy(best_fold["params"])
            if isinstance(consensus_params.get("max_features"), int):
                consensus_params["max_features"] = min(consensus_params["max_features"], len(consensus_vars))

            consensus_model = RandomSurvivalForest(**consensus_params)
            consensus_model.fit(X_consensus, y_structured)

            # ── Build report ──
            report_lines = ["=" * 60, "OPTIMIZACIÓN ROBUSTA RSF — NESTED CV (Arquitectura B)", "=" * 60, ""]
            report_lines.append(f"Folds externos: {K_OUTER}")
            report_lines.append(f"Modo limpieza VIMP: {vimp_mode_label}")
            report_lines.append(f"Métrica CV interna: {cv_metric_label}")
            report_lines.append(f"Poda por correlación: {corr_threshold_label}")
            report_lines.append(f"Árboles candidatos: {', '.join(str(x) for x in tree_candidates)}")
            report_lines.append(f"Observaciones totales: {len(X_encoded)}")
            report_lines.append(f"Variables iniciales: {n_features_orig}")
            report_lines.append("")
            for r in fold_results:
                report_lines.append(
                    f"  Fold {r['fold']}: C-Uno={self._format_metric(r['c_uno'])}, "
                    f"IBS={self._format_metric(r['ibs'])}, BSS={self._format_metric(r['bss'])}, "
                    f"vars={r['n_features']}, VIMP_elim={r['vimp_removed']}")
            report_lines.append("")

            report_lines.append("MÉTRICAS AGREGADAS")
            report_lines.append("-" * 40)
            if c_uno_vals:
                report_lines.append(f"  C-Uno:      {np.mean(c_uno_vals):.4f} ± {np.std(c_uno_vals):.4f}")
            if c_ant_vals:
                report_lines.append(f"  C-Antolini: {np.mean(c_ant_vals):.4f} ± {np.std(c_ant_vals):.4f}")
            if ibs_vals:
                report_lines.append(f"  IBS:        {np.mean(ibs_vals):.4f} ± {np.std(ibs_vals):.4f}")
            if bss_vals:
                report_lines.append(f"  BSS:        {np.mean(bss_vals):.4f} ± {np.std(bss_vals):.4f}")
            report_lines.append("")

            report_lines.append("ESTABILIDAD DE SELECCIÓN DE VARIABLES")
            report_lines.append("-" * 40)
            report_lines.append(f"  Umbral consenso: ≥{consensus_threshold}/{K_OUTER} folds (≥60%)")
            report_lines.append(f"  Intersección (todos los folds): {len(intersection_vars)}")
            if intersection_vars:
                report_lines.append(f"    {', '.join(intersection_vars)}")
            report_lines.append(f"  Variables consenso (≥{consensus_threshold} folds): {len(consensus_vars)}")
            report_lines.append(f"  Frecuencia por variable:")
            for v, c in sorted(var_freq.items(), key=lambda x: -x[1]):
                bar = "█" * c + "░" * (K_OUTER - c)
                status = "✓" if c >= consensus_threshold else "·"
                report_lines.append(f"    {status} {v}: {bar} ({c}/{K_OUTER})")
            report_lines.append("")

            report_lines.append("MODELO CONSENSO FINAL")
            report_lines.append("-" * 40)
            report_lines.append(f"  Variables ({len(consensus_vars)}): {', '.join(consensus_vars)}")
            report_lines.append(f"  Entrenado con TODOS los datos ({len(X_consensus)} obs)")
            report_lines.append(f"  OOB score: {self._format_metric(getattr(consensus_model, 'oob_score_', None))}")
            report_lines.append(f"  trees={consensus_params.get('n_estimators')}, mtry={self._annotate_max_features_text(consensus_params.get('max_features'))}, "
                                f"nodesize={consensus_params.get('min_samples_leaf')}")

            h, rem = divmod(int(total_elapsed), 3600)
            m, s = divmod(rem, 60)
            report_lines.append(f"  Tiempo total: {h:02d}:{m:02d}:{s:02d}")
            report_lines.append("")
            if warnings_list:
                report_lines.append("Avisos:")
                for w in warnings_list:
                    report_lines.append(f"  - {w}")

            full_report = "\n".join(report_lines)

            # ── Store as snapshot ──
            ibs_km_vals = [r["ibs_km"] for r in fold_results if r["ibs_km"] is not None]
            final_metrics = {
                "c_index_train": self._compute_c_index(y_structured, consensus_model.predict(X_consensus)),
                "c_index_test": None,
                "c_index_uno": float(np.mean(c_uno_vals)) if c_uno_vals else None,
                "c_index_antolini": float(np.mean(c_ant_vals)) if c_ant_vals else None,
                "c_index_cv_mean": float(np.mean(c_uno_vals)) if c_uno_vals else None,
                "c_index_cv_std": float(np.std(c_uno_vals)) if c_uno_vals else None,
                "c_index_cv_ci": None,
                "tau": None,
                "oob_score": self._compute_oob_cindex(consensus_model, y_structured),
                "ibs": float(np.mean(ibs_vals)) if ibs_vals else None,
                "ibs_ci": self._compute_mean_confidence_interval(ibs_vals) if len(ibs_vals) >= 2 else None,
                "ibs_km": float(np.mean(ibs_km_vals)) if ibs_km_vals else None,
                "bss": float(np.mean(bss_vals)) if bss_vals else None,
                "bss_ci": self._compute_mean_confidence_interval(bss_vals) if len(bss_vals) >= 2 else None,
                "n_train": len(X_consensus),
                "n_test": 0,
                "nested_cv_folds": K_OUTER,
                "var_frequency": var_freq,
            }

            full_preds = np.asarray(consensus_model.predict(X_consensus), dtype=float)
            latest_survival_profiles = self._build_survival_profiles(consensus_model, X_consensus, full_preds)
            latest_prediction_df = clean_data[[duration_col, event_col]].copy()
            latest_prediction_df["risk_score"] = full_preds
            try:
                labels = ["Q1 bajo", "Q2 medio-bajo", "Q3 medio-alto", "Q4 alto"]
                qn = min(4, max(2, latest_prediction_df["risk_score"].nunique()))
                latest_prediction_df["risk_group"] = pd.qcut(
                    latest_prediction_df["risk_score"].rank(method="first"),
                    q=qn,
                    labels=labels[:qn],
                    duplicates="drop",
                )
            except Exception:
                latest_prediction_df["risk_group"] = "Grupo único"

            snapshot = {
                "label": f"NestedCV #{len(self.saved_models) + 1}",
                "params": copy.deepcopy(consensus_params),
                "metrics": final_metrics,
                "report_text": full_report,
                "model": consensus_model,
                "feature_importance_df": self._compute_permutation_importance(
                    consensus_model, X_consensus, y_structured, random_state),
                "latest_prediction_df": latest_prediction_df.copy(deep=True),
                "latest_survival_profiles": copy.deepcopy(latest_survival_profiles),
                "latest_calibration_df": pd.DataFrame(),
                "latest_brier_df": pd.DataFrame(),
                "latest_eval_time": None,
                "latest_fit_dataframe": clean_data.copy(deep=True),
                "latest_duration_col": duration_col,
                "latest_event_col": event_col,
                "latest_covariates": list(covariates),
                "latest_encoded_columns": list(consensus_vars),
                "latest_drop_first": bool(self.drop_first_var.get()),
                "variable_configs": copy.deepcopy(getattr(self, "variable_configs", {})),
            }
            self.saved_models.append(snapshot)
            self.active_saved_model_index = len(self.saved_models) - 1
            self._refresh_saved_models_tree()

            self.model = consensus_model
            self.feature_importance_df = snapshot["feature_importance_df"]
            self.latest_prediction_df = latest_prediction_df
            self.latest_survival_profiles = latest_survival_profiles
            self.latest_calibration_df = pd.DataFrame()
            self.latest_brier_df = pd.DataFrame()
            self.latest_eval_time = None
            self.latest_fit_dataframe = clean_data
            self.latest_duration_col = duration_col
            self.latest_event_col = event_col
            self.latest_covariates = list(covariates)
            self.latest_encoded_columns = list(consensus_vars)
            self.latest_report_text = full_report

            self.results_text.delete("1.0", tk.END)
            self.results_text.insert(tk.END, full_report)
            if c_uno_vals:
                self.auto_tuning_status_var.set(
                    f"Nested CV: C-Uno={np.mean(c_uno_vals):.4f}±{np.std(c_uno_vals):.4f}, "
                    f"{len(consensus_vars)} vars consenso")
            else:
                self.auto_tuning_status_var.set("Nested CV completado")

            # Show summary popup
            self._show_nested_cv_summary(fold_results, var_freq, consensus_vars, intersection_vars,
                                          consensus_params, final_metrics, total_elapsed,
                                          n_features_orig, K_OUTER)

        except InterruptedError:
            self.auto_tuning_status_var.set("Nested CV cancelada por el usuario.")
            messagebox.showinfo("Cancelado", "La optimización Nested CV fue cancelada.")
        except Exception as exc:
            self.auto_tuning_status_var.set(f"Nested CV falló: {exc}")
            traceback_text = traceback.format_exc(limit=5)
            self.results_text.delete("1.0", tk.END)
            self.results_text.insert(tk.END, f"Error en Nested CV:\n{exc}\n\n{traceback_text}")
            messagebox.showerror("Error", f"Error en Nested CV:\n{exc}")
        finally:
            self._close_auto_tuning_progress_dialog()
            self._tuning_cancel_requested = False

    def _show_nested_cv_summary(self, fold_results, var_freq, consensus_vars, intersection_vars,
                                 consensus_params, final_metrics, total_elapsed, n_features_orig, k_outer):
        """Show summary popup for Nested CV optimization."""
        parent_window = self.winfo_toplevel()
        popup = tk.Toplevel(parent_window)
        popup.title("Resumen — Nested CV RSF")
        popup.geometry("700x620")
        popup.transient(parent_window)
        popup.resizable(True, True)

        container = ttk.Frame(popup, padding=14)
        container.pack(fill=tk.BOTH, expand=True)

        ttk.Label(container, text="Resumen: Optimización Robusta RSF — Nested CV",
                  font=("Segoe UI", 12, "bold")).pack(anchor="w", pady=(0, 8))

        text_widget = scrolledtext.ScrolledText(container, wrap=tk.WORD, font=("Consolas", 9), height=30)
        text_widget.pack(fill=tk.BOTH, expand=True, pady=(0, 8))

        lines = []
        h, rem = divmod(int(total_elapsed), 3600)
        m, s = divmod(rem, 60)
        lines.append(f"Tiempo total: {h:02d}:{m:02d}:{s:02d}")
        lines.append(f"Folds externos: {k_outer}")
        lines.append("")

        lines.append("═══ Resultados por Fold ═══")
        c_uno_vals = []
        for r in fold_results:
            c = r["c_uno"]
            if c is not None:
                c_uno_vals.append(c)
            lines.append(f"  Fold {r['fold']}: C-Uno={self._format_metric(c)}, "
                         f"BSS={self._format_metric(r.get('bss'))}, vars={r['n_features']}")
        lines.append("")

        lines.append("═══ Métricas Agregadas ═══")
        if c_uno_vals:
            lines.append(f"  C-Uno:  {np.mean(c_uno_vals):.4f} ± {np.std(c_uno_vals):.4f}")
        bss_vals = [r["bss"] for r in fold_results if r["bss"] is not None]
        if bss_vals:
            lines.append(f"  BSS:    {np.mean(bss_vals):.4f} ± {np.std(bss_vals):.4f}")
        lines.append(f"  OOB (consenso): {self._format_metric(final_metrics.get('oob_score'))}")
        lines.append("")

        lines.append("═══ Estabilidad de Variables ═══")
        consensus_threshold = max(1, int(np.ceil(k_outer * 0.6)))
        lines.append(f"  Variables iniciales: {n_features_orig}")
        lines.append(f"  Intersección (todos los folds): {len(intersection_vars)}")
        lines.append(f"  Consenso (≥{consensus_threshold}/{k_outer}): {len(consensus_vars)}")
        lines.append("")
        for v, c in sorted(var_freq.items(), key=lambda x: -x[1]):
            bar = "█" * c + "░" * (k_outer - c)
            status = "✓" if c >= consensus_threshold else "·"
            lines.append(f"    {status} {v}: {bar} ({c}/{k_outer})")
        lines.append("")

        lines.append("═══ Modelo Consenso ═══")
        lines.append(f"  Variables ({len(consensus_vars)}): {', '.join(consensus_vars)}")
        lines.append(f"  mtry={consensus_params.get('max_features')}, "
                     f"nodesize={consensus_params.get('min_samples_leaf')}")

        text_widget.insert(tk.END, "\n".join(lines))
        text_widget.configure(state=tk.DISABLED)

        btn_frame = ttk.Frame(container)
        btn_frame.pack(fill=tk.X)
        ttk.Button(btn_frame, text="Cerrar", command=popup.destroy).pack(side=tk.RIGHT, padx=5)
        ttk.Button(btn_frame, text="Copiar al portapapeles",
                   command=lambda: (popup.clipboard_clear(), popup.clipboard_append("\n".join(lines)))).pack(
            side=tk.RIGHT, padx=5)

    def run_auto_tuning(self):
        if getattr(self, '_auto_tuning_in_progress', False):
            messagebox.showinfo(
                "En progreso",
                "Ya hay un tuning en ejecución. Espera a que termine o cancélalo.",
                parent=self.winfo_toplevel(),
            )
            return

        if not SKSURV_AVAILABLE:
            messagebox.showerror(
                "RSF no disponible",
                "No se pudo importar `scikit-survival`. Instálalo para ejecutar tuning automático."
                f"\n\nDetalle: {SKSURV_IMPORT_ERROR}",
            )
            return

        if self.data is None:
            messagebox.showerror("Error", "Cargue datos o use el dataset compartido primero.")
            return

        filtered_data = self.filter_component.apply_filters()
        if filtered_data is None or filtered_data.empty:
            messagebox.showerror("Error", "No hay datos disponibles tras aplicar filtros.")
            return

        duration_col = self.duration_var.get().strip()
        event_col = self.event_var.get().strip()
        selected_indices = self.covariates_listbox.curselection()
        covariates = [self.covariates_listbox.get(i) for i in selected_indices]

        if not duration_col or not event_col or not covariates:
            messagebox.showerror("Error", "Seleccione tiempo, evento y al menos una covariable para el tuning RSF.")
            return

        self.auto_tuning_status_var.set("Tuning automático: preparando datos...")
        self.update_idletasks()

        tuning_results = []
        selected_profile = "General"
        recommended_profile = None
        tuning_mode = self._resolve_tuning_evaluation_mode()
        feature_sets = self._build_tuning_feature_sets(covariates, tuning_mode)

        if tuning_mode == "TodosContraTodos" and len(feature_sets) > 10:
            from itertools import combinations as _comb
            n_vars = len([c for c in covariates if str(c).strip()])
            total_possible = sum(1 for r in range(1, n_vars + 1) for _ in _comb(range(n_vars), r))
            required_covs_preview = self._get_tvt_required_covariates(covariates)
            min_cov_preview = self._get_tvt_min_covariates(n_vars, n_required=len(required_covs_preview))
            max_cov_preview = self._get_tvt_max_covariates(n_vars, min_covariates=min_cov_preview)
            max_combo_preview = self._get_tvt_max_combinations()
            shown = len(feature_sets)
            msg = (f"Se generarán {shown} combinaciones de variables "
                   f"(de {total_possible} posibles con {n_vars} variables).\n"
                   f"Límites activos: mín={min_cov_preview}, máx={max_cov_preview}, tope combinaciones={max_combo_preview}.\n\n"
                   f"Esto puede tomar mucho tiempo. ¿Continuar?")
            if not messagebox.askyesno("Todos contra todos", msg):
                self.auto_tuning_status_var.set("Tuning cancelado por el usuario.")
                return

        auto_trees_mode = True
        _ui_tree_cap = self._coerce_int(self.n_estimators_var.get(), self._AUTO_TREE_MAX, minimum=10)
        # El campo manual de árboles no debe limitar accidentalmente el auto-tuning
        # por debajo del tope automático por defecto.
        auto_tree_max_limit = max(int(self._AUTO_TREE_MAX), int(_ui_tree_cap))
        if int(auto_tree_max_limit) <= int(self._AUTO_TREE_START):
            auto_tree_max_limit = int(self._AUTO_TREE_MAX)
        total_candidates = 0
        cancel_requested = False
        tuning_start_time = time.perf_counter()

        # Clear stale live autotuning snapshots from previous runs.
        if self.saved_models:
            _prev_active_snapshot = None
            if isinstance(self.active_saved_model_index, int) and 0 <= self.active_saved_model_index < len(self.saved_models):
                _prev_active_snapshot = self.saved_models[self.active_saved_model_index]
            self.saved_models[:] = [
                snapshot for snapshot in self.saved_models
                if not bool(snapshot.get("autotune_live", False))
            ]
            _found_idx = self._find_snapshot_identity(_prev_active_snapshot, self.saved_models)
            if _found_idx is not None:
                self.active_saved_model_index = _found_idx
            elif self.saved_models:
                self.active_saved_model_index = len(self.saved_models) - 1
            else:
                self.active_saved_model_index = None
            self._refresh_saved_models_tree(rerank=False)

        try:
            requested_test_size = self._coerce_float(self.test_size_var.get(), 0.25, minimum=0.0, maximum=0.95)
            _all_seeds_tuning = self._parse_random_seeds()
            random_state = _all_seeds_tuning[0]
            if not feature_sets:
                raise ValueError("No se pudieron construir conjuntos de variables para el modo de evaluación seleccionado.")

            warnings_list = []
            scope_contexts = []

            for feature_set in feature_sets:
                scope_label = feature_set.get("scope_label", "Multivariado")
                scope_covariates = list(feature_set.get("covariates", []))
                if not scope_covariates:
                    warnings_list.append(f"{scope_label}: sin covariables válidas, se omite.")
                    continue

                clean_data_scope, X_encoded_scope, y_structured_scope, scope_warnings = self._prepare_dataframe_for_rsf(
                    filtered_data, duration_col, event_col, scope_covariates
                )
                warnings_list.extend([f"{scope_label}: {item}" for item in scope_warnings])

                event_values_scope = (pd.to_numeric(clean_data_scope[event_col], errors="coerce").fillna(0) > 0).astype(int)
                test_size_scope, stratify_values_scope, split_warnings_scope = self._resolve_holdout_split_settings(
                    clean_data_scope,
                    event_col,
                    requested_test_size,
                    min_train_rows=max(8, int(X_encoded_scope.shape[1]) + 2),
                    min_test_rows=2,
                    prefer_stratify=bool(self.stratify_event_var.get()),
                    context_label=f"RSF tuning ({scope_label})",
                )
                warnings_list.extend([f"{scope_label}: {item}" for item in split_warnings_scope])

                if test_size_scope <= 0:
                    X_train_scope = X_encoded_scope.copy()
                    X_test_scope = X_encoded_scope.iloc[0:0].copy()
                    y_train_scope = y_structured_scope
                    y_test_scope = y_structured_scope[:0]
                else:
                    X_train_scope, X_test_scope, y_train_scope, y_test_scope = train_test_split(
                        X_encoded_scope,
                        y_structured_scope,
                        test_size=test_size_scope,
                        random_state=random_state,
                        stratify=stratify_values_scope,
                    )

                resolved_tau_scope = self._resolve_tau(y_train_scope, y_test_scope) if len(X_test_scope) > 0 else None
                eval_times_scope = self._build_evaluation_time_grid(y_train_scope, y_test_scope, tau=resolved_tau_scope) if len(X_test_scope) > 0 else None

                base_params_scope = self._get_current_rsf_params(X_train_scope.shape[1])
                tuning_candidates_scope, scope_profile, scope_recommended_profile = self._generate_tuning_candidates(
                    X_train_scope.shape[1], len(clean_data_scope), n_train=len(X_train_scope), auto_trees=auto_trees_mode
                )
                if not tuning_candidates_scope:
                    warnings_list.append(f"{scope_label}: no se generaron combinaciones para tuning.")
                    continue

                if selected_profile == "General" or selected_profile == scope_profile:
                    selected_profile = scope_profile
                if recommended_profile is None:
                    recommended_profile = scope_recommended_profile

                scope_contexts.append({
                    "scope_label": scope_label,
                    "scope_covariates": scope_covariates,
                    "clean_data": clean_data_scope,
                    "X_encoded": X_encoded_scope,
                    "y_structured": y_structured_scope,
                    "test_size": test_size_scope,
                    "X_train": X_train_scope,
                    "X_test": X_test_scope,
                    "y_train": y_train_scope,
                    "y_test": y_test_scope,
                    "resolved_tau": resolved_tau_scope,
                    "eval_times": eval_times_scope,
                    "base_params": base_params_scope,
                    "tuning_candidates": tuning_candidates_scope,
                })

            if not scope_contexts:
                raise ValueError("No fue posible preparar ningún conjunto de variables para tuning (Univariado/Multivariado).")

            # Normalize candidate order and remove exact duplicates before computing progress totals.
            for _scope_ctx in scope_contexts:
                _base = _scope_ctx.get("base_params", {})
                _raw_candidates = list(_scope_ctx.get("tuning_candidates", []) or [])

                def _candidate_sort_key(_cand):
                    _p = dict((_cand or {}).get("params", {}))
                    _n = int(_p.pop("n_estimators", _base.get("n_estimators", 300)))
                    _rest = tuple((k, repr(_p.get(k))) for k in sorted(_p.keys()))
                    return (_rest, _n)

                try:
                    _sorted_candidates = sorted(_raw_candidates, key=_candidate_sort_key)
                except Exception:
                    _sorted_candidates = _raw_candidates

                _seen_candidate_keys = set()
                _unique_candidates = []
                for _cand in _sorted_candidates:
                    _params = dict((_cand or {}).get("params", {}))
                    _params["n_estimators"] = int(_params.get("n_estimators", _base.get("n_estimators", 300)))
                    _ckey = tuple((k, repr(_params.get(k))) for k in sorted(_params.keys()))
                    if _ckey in _seen_candidate_keys:
                        continue
                    _seen_candidate_keys.add(_ckey)
                    _unique_candidates.append(_cand)

                _scope_ctx["tuning_candidates"] = _unique_candidates

            total_candidates = sum(len(ctx.get("tuning_candidates", [])) for ctx in scope_contexts)
            # Pre-compute total tree weight for tree-weighted ETA
            _total_trees = 0
            for _sc in scope_contexts:
                _bp = _sc.get("base_params", {})
                for _tc in _sc.get("tuning_candidates", []):
                    if auto_trees_mode:
                        _total_trees += int(max(self._AUTO_TREE_MIN, auto_tree_max_limit))
                    else:
                        _total_trees += _tc.get("params", {}).get("n_estimators", _bp.get("n_estimators", 300))
            _trees_completed = 0
            tuning_start_time = time.perf_counter()
            progress_metric_key, progress_metric_higher_better = self._resolve_tuning_progress_metric_key()
            _best_progress_value = None
            _best_progress_params = None
            _n_evaluated = 0
            _sync_publish_every = 1
            self._open_auto_tuning_progress_dialog(
                selected_profile,
                len(scope_contexts[0]["clean_data"]),
                total_candidates,
                duration_col=duration_col,
                event_col=event_col,
                covariates=covariates,
            )
            self._live_tuning_results = tuning_results  # shared ref for live scatter (must be AFTER dialog open)
            self._update_auto_tuning_progress_dialog(
                profile_name=selected_profile,
                n_rows=len(scope_contexts[0]["clean_data"]),
                completed=0,
                total=total_candidates,
                elapsed_seconds=0.0,
                cancel_requested=False,
            )

            _completed_global = 0
            try:
                _warm_start_supported = "warm_start" in inspect.signature(RandomSurvivalForest).parameters
            except Exception:
                _warm_start_supported = False
            self._auto_tuning_in_progress = True
            for scope_context in scope_contexts:
                scope_label = scope_context["scope_label"]
                scope_covariates = scope_context["scope_covariates"]
                clean_data_scope = scope_context["clean_data"]
                X_encoded_scope = scope_context["X_encoded"]
                test_size_scope = scope_context["test_size"]
                X_train = scope_context["X_train"]
                X_test = scope_context["X_test"]
                y_train = scope_context["y_train"]
                y_test = scope_context["y_test"]
                resolved_tau_tuning = scope_context["resolved_tau"]
                eval_times_tuning = scope_context["eval_times"]
                base_params = scope_context["base_params"]
                tuning_candidates = scope_context["tuning_candidates"]
                _scope_warm_cache = {}

                self._reset_auto_tuning_skip_button()
                self._wait_if_auto_tuning_paused(
                    tuning_results=tuning_results,
                    profile_name=selected_profile,
                    recommended_profile=recommended_profile,
                    requested_test_size=requested_test_size,
                    duration_col=duration_col,
                    event_col=event_col,
                    covariates=covariates,
                    tuning_mode=tuning_mode,
                )
                if bool(self._tuning_cancel_requested):
                    cancel_requested = True
                    break

                for candidate in tuning_candidates:
                    if bool(self._tuning_skip_scope_requested):
                        warnings_list.append(
                            f"{scope_label}: combinación de variables saltada por el usuario."
                        )
                        self._tuning_skip_scope_requested = False
                        self._reset_auto_tuning_skip_button()
                        break

                    self._wait_if_auto_tuning_paused(
                        tuning_results=tuning_results,
                        profile_name=selected_profile,
                        recommended_profile=recommended_profile,
                        requested_test_size=requested_test_size,
                        duration_col=duration_col,
                        event_col=event_col,
                        covariates=covariates,
                        tuning_mode=tuning_mode,
                    )
                    if bool(self._tuning_cancel_requested):
                        cancel_requested = True
                        break
                    if bool(self._tuning_skip_scope_requested):
                        warnings_list.append(
                            f"{scope_label}: combinación de variables saltada por el usuario."
                        )
                        self._tuning_skip_scope_requested = False
                        self._reset_auto_tuning_skip_button()
                        break

                    _completed_global += 1
                    _cand_trees = candidate.get("params", {}).get("n_estimators", base_params.get("n_estimators", 300))
                    _trees_used_candidate = int(_cand_trees)
                    _current_tree_text = None
                    if auto_trees_mode:
                        _current_tree_text = f"árboles auto: inicio={self._AUTO_TREE_START} | tope={auto_tree_max_limit} | elegido=calculando..."
                    elapsed_seconds = time.perf_counter() - tuning_start_time
                    _cand_params_display = {**base_params, **candidate.get("params", {}), **{k: v for k, v in candidate.get("display", {}).items() if k == "split_multiplier"}}
                    self._update_auto_tuning_progress_dialog(
                        profile_name=selected_profile,
                        n_rows=len(clean_data_scope),
                        completed=_completed_global - 1,
                        total=total_candidates,
                        elapsed_seconds=elapsed_seconds,
                        cancel_requested=bool(self._tuning_cancel_requested),
                        current_params=_cand_params_display,
                        best_metric_value=_best_progress_value,
                        best_metric_params=_best_progress_params,
                        n_evaluated=_n_evaluated,
                        trees_completed=_trees_completed,
                        trees_total=_total_trees,
                        current_tree_text=_current_tree_text,
                        current_covariates=list(scope_covariates),
                        current_scope=scope_label,
                    )
                    if bool(self._tuning_cancel_requested):
                        cancel_requested = True
                        break

                    candidate_params = {**base_params, **candidate.get("params", {}), "random_state": random_state}
                    if not bool(candidate_params.get("bootstrap", True)):
                        candidate_params["oob_score"] = False

                    if len(_all_seeds_tuning) > 1:
                        _cur_var = getattr(self, "_tuning_current_model_var", None)
                        if _cur_var is not None:
                            try:
                                _cur_var.set(
                                    f"Evaluando #{_completed_global}/{total_candidates}"
                                    f" — semilla 1/{len(_all_seeds_tuning)}: {_all_seeds_tuning[0]}"
                                    f" | {self._format_candidate_params_short(_cand_params_display)}"
                                )
                                _d = getattr(self, "_tuning_progress_dialog", None)
                                if _d:
                                    self._update_tuning_status_text_widget()
                                    _d.update_idletasks()
                            except Exception:
                                pass

                    try:
                        cv_metric_code, _cv_metric_label = self._resolve_cv_metric_choice()
                        _cache_key = None

                        if auto_trees_mode:
                            model, _selected_trees, _tree_growth_history, _growth_metric_label = self._fit_rsf_until_plateau(
                                candidate_params,
                                X_train,
                                y_train,
                                X_test,
                                y_test,
                                cv_metric_code,
                                eval_times_tuning=eval_times_tuning,
                                resolved_tau_tuning=resolved_tau_tuning,
                                warm_start_supported=_warm_start_supported,
                                max_trees_limit=auto_tree_max_limit,
                            )
                            candidate_params["n_estimators"] = int(_selected_trees)
                            _trees_used_candidate = int(_selected_trees)
                            _cand_params_display["n_estimators"] = int(_selected_trees)
                            _tree_start_used = int(_tree_growth_history[0][0]) if _tree_growth_history else int(self._AUTO_TREE_START)
                            _current_tree_text = (
                                f"árboles auto: inicio={_tree_start_used} | tope={auto_tree_max_limit} | "
                                f"elegido={_selected_trees} ({_growth_metric_label})"
                            )
                        else:
                            model = None
                            _cache_key = None
                            _target_trees = int(candidate_params.get("n_estimators", base_params.get("n_estimators", 300)))

                            if _warm_start_supported:
                                _cache_parts = []
                                for _k in sorted(candidate_params.keys()):
                                    if _k in ("n_estimators", "warm_start"):
                                        continue
                                    _cache_parts.append((_k, repr(candidate_params.get(_k))))
                                _cache_key = tuple(_cache_parts)
                                _cached_payload = _scope_warm_cache.get(_cache_key)
                                if _cached_payload is not None:
                                    _cached_model = _cached_payload.get("model")
                                    _cached_trees = int(_cached_payload.get("n_estimators", 0) or 0)
                                    if _cached_model is not None and _target_trees == _cached_trees:
                                        model = _cached_model
                                    elif _cached_model is not None and _target_trees > _cached_trees:
                                        try:
                                            _cached_model.set_params(warm_start=True, n_estimators=_target_trees)
                                            self._fit_model_with_ui_pump(_cached_model, X_train, y_train)
                                            model = _cached_model
                                            _scope_warm_cache[_cache_key] = {"model": model, "n_estimators": _target_trees}
                                        except Exception:
                                            model = None

                            if model is None:
                                _fit_params = dict(candidate_params)
                                # Eliminar claves de metadatos que no son parámetros de RandomSurvivalForest
                                for _meta_k in ("seed_used", "seeds_used"):
                                    _fit_params.pop(_meta_k, None)
                                if _warm_start_supported:
                                    _fit_params["warm_start"] = True
                                model = RandomSurvivalForest(**_fit_params)
                                self._fit_model_with_ui_pump(model, X_train, y_train)
                                if _warm_start_supported and _cache_key is not None:
                                    _scope_warm_cache[_cache_key] = {"model": model, "n_estimators": _target_trees}

                        self._flush_tuning_dialog_events()
                        if self._tuning_cancel_requested:
                            cancel_requested = True
                            break

                        if model is None:
                            raise RuntimeError("No se pudo entrenar el modelo RSF para esta configuración.")

                        train_preds = model.predict(X_train)
                        test_preds = model.predict(X_test) if len(X_test) > 0 else np.asarray([], dtype=float)

                        metrics = {
                            "c_index_train": self._compute_c_index(y_train, train_preds),
                            "c_index_train_ci": self._compute_c_index_ci(y_train, train_preds),
                            "c_index_test": self._compute_c_index(y_test, test_preds) if (len(X_test) > 0 and cv_metric_code == "harrell") else None,
                            "c_index_test_ci": self._compute_c_index_ci(y_test, test_preds) if (len(X_test) > 0 and cv_metric_code == "harrell") else None,
                            "tau": resolved_tau_tuning,
                            "oob_score": self._compute_oob_cindex(model, y_train),
                        }

                        if cv_metric_code == "uno" and len(X_test) > 0 and callable(concordance_index_ipcw):
                            try:
                                ipcw_result = concordance_index_ipcw(y_train, y_test, test_preds, tau=resolved_tau_tuning)
                                metrics["c_index_uno"] = float(np.asarray(ipcw_result).reshape(-1)[0])
                            except Exception:
                                metrics["c_index_uno"] = None
                        else:
                            metrics["c_index_uno"] = None

                        if cv_metric_code == "antolini" and len(X_test) > 0:
                            metrics["c_index_antolini"] = self._compute_c_antolini_score(
                                model, X_train, y_train, X_test, y_test,
                                eval_times=eval_times_tuning, tau=resolved_tau_tuning)
                        else:
                            metrics["c_index_antolini"] = None

                        # C-test siempre refleja la métrica elegida
                        if metrics["c_index_test"] is None and len(X_test) > 0:
                            if cv_metric_code == "uno" and metrics.get("c_index_uno") is not None:
                                metrics["c_index_test"] = metrics["c_index_uno"]
                            elif cv_metric_code == "antolini" and metrics.get("c_index_antolini") is not None:
                                metrics["c_index_test"] = metrics["c_index_antolini"]

                        metrics["c_index_uno_ci"] = metrics.get("c_index_test_ci") if metrics.get("c_index_uno") is not None else None
                        metrics["c_index_antolini_ci"] = metrics.get("c_index_test_ci") if metrics.get("c_index_antolini") is not None else None

                        if len(X_test) > 0 and eval_times_tuning is not None:
                            try:
                                _ibs_m, _ibs_km, _bss = self._compute_ibs_and_bss(
                                    model, X_train, y_train, X_test, y_test, eval_times_tuning)
                                metrics["ibs"] = _ibs_m
                                metrics["ibs_km"] = _ibs_km
                                metrics["bss"] = _bss
                            except Exception:
                                metrics["ibs"] = None
                                metrics["ibs_km"] = None
                                metrics["bss"] = None
                        else:
                            # Fallback: compute IBS/BSS from OOB predictions (no test set needed)
                            try:
                                _ibs_m, _ibs_km, _bss = self._compute_ibs_and_bss_oob(model, X_train, y_train)
                                metrics["ibs"] = _ibs_m
                                metrics["ibs_km"] = _ibs_km
                                metrics["bss"] = _bss
                            except Exception:
                                metrics["ibs"] = None
                                metrics["ibs_km"] = None
                                metrics["bss"] = None
                        # Always try OOB IBS/BSS as fallback if test-based gave None
                        if metrics.get("bss") is None:
                            try:
                                _oi, _ok, _ob = self._compute_ibs_and_bss_oob(model, X_train, y_train)
                                if _ob is not None:
                                    metrics["ibs"] = metrics.get("ibs") or _oi
                                    metrics["ibs_km"] = metrics.get("ibs_km") or _ok
                                    metrics["bss"] = _ob
                            except Exception:
                                pass

                        # Permitir que el botón Cancelar responda tras el cálculo de IBS
                        self._flush_tuning_dialog_events()
                        if bool(self._tuning_cancel_requested):
                            cancel_requested = True
                            break

                        train_event_values = (np.asarray(y_train["event"], dtype=bool).astype(int)) if len(y_train) > 0 else pd.Series(dtype=int)
                        cv_result = self._compute_cv_cindex(
                            X_train,
                            train_event_values,
                            y_train,
                            candidate_params,
                            force_enabled=True,
                            return_values=True,
                        )
                        if isinstance(cv_result, tuple) and len(cv_result) == 3:
                            cv_mean, cv_std, cv_values = cv_result
                        else:
                            cv_mean, cv_std = cv_result
                            cv_values = []
                        metrics["c_index_cv_mean"] = cv_mean
                        metrics["c_index_cv_std"] = cv_std
                        metrics["c_index_cv_ci"] = self._compute_mean_confidence_interval(cv_values, clip_min=0.0, clip_max=1.0) if cv_values else None

                        # Permitir que el botón Cancelar responda tras la validación cruzada
                        self._flush_tuning_dialog_events()
                        if bool(self._tuning_cancel_requested):
                            cancel_requested = True
                            break

                        # ── Capturar semilla 1 para gráfica live (reinicia el candidato) ──
                        try:
                            _csd = getattr(self, "_autotune_current_seed_metrics", None)
                            if _csd is not None:
                                _csd.clear()
                                _cur_vimp_d = getattr(self, "_autotune_current_vimp", None)
                                if isinstance(_cur_vimp_d, dict):
                                    _cur_vimp_d.clear()
                                self._autotune_current_candidate_n = int(_completed_global)
                                # Record n_estimators for progress bar
                                _n_est_cand = int(candidate_params.get("n_estimators", 100))
                                _n_seeds_cand = len(_all_seeds_tuning) if _all_seeds_tuning else 1
                                self._autotune_current_trees_total = _n_est_cand * _n_seeds_cand
                                self._autotune_current_trees_done = _n_est_cand  # first seed done
                                # Save params for chart footer
                                self._autotune_current_candidate_params = dict(candidate_params)
                                _csd.append({
                                    "seed": int(_all_seeds_tuning[0]),
                                    "oob":   metrics.get("oob_score"),
                                    "cv":    metrics.get("c_index_cv_mean"),
                                    "ctest": metrics.get("c_index_test"),
                                    "bss":   metrics.get("bss"),
                                })
                                try:
                                    self._update_autotune_vimp_chart()
                                    _d = getattr(self, "_tuning_progress_dialog", None)
                                    if _d:
                                        _d.update()
                                except Exception:
                                    pass
                        except Exception:
                            pass
                        # ──────────────────────────────────────────────────────────────────

                        # ── Multi-seed averaging for tuning candidate ──────
                        _discarded_by_screening = False
                        _semi_complete = False
                        _last_screen_val = None
                        _last_screen_detail = {}
                        # Siempre inicializar (fallback 1 semilla)
                        _seed_metrics_list = [metrics]
                        _seed_vals_list = [int(_all_seeds_tuning[0]) if _all_seeds_tuning else int(candidate_params.get("random_state", 42))]
                        if len(_all_seeds_tuning) > 1:
                            _seed_metrics_list = [metrics]
                            _seed_vals_list = [int(_all_seeds_tuning[0])]

                            _es_enabled = getattr(self, "early_stopping_enabled_var", None)
                            _es_enabled = bool(_es_enabled.get()) if _es_enabled else False
                            _es_seeds_limit = 2
                            _es_gap_limit = 0.01
                            _es_impossible_mode = False
                            if _es_enabled:
                                try: _es_seeds_limit = max(1, int(self.early_stopping_seeds_var.get()))
                                except Exception: _es_seeds_limit = 2
                                try: _es_gap_limit = float(self.early_stopping_gap_var.get())
                                except Exception: _es_gap_limit = 0.01
                                _impossible_var = getattr(self, "early_stopping_impossible_stops_var", None)
                                _es_impossible_mode = bool(_impossible_var.get()) if _impossible_var else False

                            _n_total = len(_all_seeds_tuning)
                            _n_screen = _es_seeds_limit  # default; capped below when enabled
                            if _es_enabled and _n_total > 1:
                                _n_screen = min(_n_total, _es_seeds_limit)
                                _u_screen = _es_gap_limit * (1.0 + ((_n_total - _n_screen) / float(_n_total)) * 0.3125)
                            else:
                                _u_screen = _es_gap_limit

                            def _evaluate_screening(m_list):
                                # Criterio POR PAR: para cada par (CV-test, CV-OOB, test-OOB) se
                                # acumula la suma de diferencias absolutas a lo largo de las semillas.
                                # El "running_sum" reportado es la suma del PAR PEOR (el que tiene mayor promedio).
                                # Check imposible: si la suma de CUALQUIER par > umbral × n_total,
                                # ese par ya no puede bajar → descartar.
                                # Retorna (avg_peor_par, worst_pair_dict, running_sum_peor_par, n_valid).
                                _metric_keys = ["c_index_cv_mean", "c_index_test", "oob_score"]
                                _pair_labels = {
                                    ("c_index_cv_mean", "c_index_test"): "CV-test",
                                    ("c_index_cv_mean", "oob_score"): "CV-OOB",
                                    ("c_index_test", "oob_score"): "test-OOB",
                                }
                                _pair_sums = {}    # par -> suma acumulada de diffs
                                _pair_counts = {}  # par -> cantidad de veces que se pudo calcular
                                
                                for _m_dict in m_list:
                                    _vals = {}
                                    for _mk in _metric_keys:
                                        _v = _m_dict.get(_mk)
                                        if _v is not None:
                                            try:
                                                _fv = float(_v)
                                                if np.isfinite(_fv):
                                                    _vals[_mk] = _fv
                                            except (TypeError, ValueError):
                                                pass
                                    if len(_vals) < 2:
                                        continue
                                    _kl = list(_vals.keys())
                                    for _pi in range(len(_kl)):
                                        for _pj in range(_pi + 1, len(_kl)):
                                            _k1, _k2 = _kl[_pi], _kl[_pj]
                                            _diff = abs(_vals[_k1] - _vals[_k2])
                                            _pname = _pair_labels.get((_k1, _k2)) or _pair_labels.get((_k2, _k1)) or f"{_k1[:2]}-{_k2[:2]}"
                                            _pair_sums[_pname] = _pair_sums.get(_pname, 0.0) + _diff
                                            _pair_counts[_pname] = _pair_counts.get(_pname, 0) + 1
                                            
                                # El par con mayor promedio determina el resultado
                                if not _pair_sums:
                                    return 0.0, {}, 0.0, 0
                                _pair_avgs = {_p: _pair_sums[_p] / _pair_counts[_p] for _p in _pair_sums}
                                _worst_pair_name = max(_pair_avgs, key=lambda _p: _pair_avgs[_p])
                                _avg = _pair_avgs[_worst_pair_name]
                                _running_sum = _pair_sums[_worst_pair_name]
                                _n_valid = _pair_counts[_worst_pair_name]
                                
                                # worst_per_pair: suma acumulada de cada par (para mostrar en UI)
                                _worst_per_pair = {_p: round(_pair_sums[_p], 5) for _p in _pair_sums}
                                return _avg, _worst_per_pair, _running_sum, _n_valid

                            _last_budget_str = ""
                            if not _discarded_by_screening:
                                for _si, _eseed in enumerate(_all_seeds_tuning[1:], 2):
                                    # Update progress label to show current seed
                                    _cur_var = getattr(self, "_tuning_current_model_var", None)
                                    if _cur_var is not None:
                                        try:
                                            _budget_prefix = f" | {_last_budget_str}" if _last_budget_str else ""
                                            _cur_var.set(
                                                f"Evaluando #{_completed_global}/{total_candidates}"
                                                f" — semilla {_si}/{len(_all_seeds_tuning)}: {_eseed}"
                                                f"{_budget_prefix}"
                                                f" | {self._format_candidate_params_short(candidate_params)}"
                                            )
                                            _d = getattr(self, "_tuning_progress_dialog", None)
                                            if _d:
                                                self._update_tuning_status_text_widget()
                                                _d.update_idletasks()
                                        except Exception:
                                            pass
                                    _, _, _, _em = self._fit_and_score_seed(
                                        _eseed, X_encoded_scope, y_structured_scope, candidate_params,
                                        test_size_scope, stratify_values_scope,
                                        cv_metric_code, resolved_tau_tuning, eval_times_tuning,
                                        precomputed_split=(X_train, X_test, y_train, y_test),
                                    )
                                    if _em:
                                        _seed_metrics_list.append(_em)
                                        _seed_vals_list.append(int(_eseed))
                                        # ── Actualizar gráfica con métricas de esta semilla ──
                                        try:
                                            _csd2 = getattr(self, "_autotune_current_seed_metrics", None)
                                            if _csd2 is not None:
                                                _csd2.append({
                                                    "seed": int(_eseed),
                                                    "oob":   _em.get("oob_score"),
                                                    "cv":    _em.get("c_index_cv_mean"),
                                                    "ctest": _em.get("c_index_test"),
                                                    "bss":   _em.get("bss"),
                                                })
                                                # Increment tree progress
                                                _n_est_done = int(candidate_params.get("n_estimators", 100))
                                                self._autotune_current_trees_done = min(
                                                    getattr(self, "_autotune_current_trees_done", 0) + _n_est_done,
                                                    getattr(self, "_autotune_current_trees_total", 1)
                                                )
                                                try:
                                                    self._update_autotune_vimp_chart()
                                                    _d2 = getattr(self, "_tuning_progress_dialog", None)
                                                    if _d2:
                                                        _d2.update()
                                                except Exception:
                                                    pass
                                        except Exception:
                                            pass
                                        # ─────────────────────────────────────────────────────

                                    # Actualizar budget acumulado tras cada semilla (para el label "Evaluando")
                                    if _es_enabled and len(_seed_metrics_list) >= 1:
                                        try:
                                            _bv, _bdetail, _bsum, _bn = _evaluate_screening(_seed_metrics_list)
                                            _b_imp = _es_gap_limit * _n_total
                                            if not _es_impossible_mode:
                                                _pairs_str = f"prom={_bv:.4f}/max_prom={_u_screen:.4f}"
                                            else:
                                                _pairs_str = " | ".join(
                                                    f"{_pn}={_pv:.4f}/{_b_imp:.4f} ({(_pv/_b_imp*100) if _b_imp>0 else 0:.0f}%)"
                                                    for _pn, _pv in _bdetail.items()
                                                ) if _bdetail else f"suma={_bsum:.4f}/{_b_imp:.4f}"
                                            _last_budget_str = _pairs_str
                                        except Exception:
                                            pass

                                    # Determinar si hay que evaluar screening en esta semilla
                                    _impossible_check_now = _es_impossible_mode and len(_seed_metrics_list) >= 2
                                    _fixed_check_now = (not _es_impossible_mode) and _si == _n_screen
                                    if _es_enabled and (_fixed_check_now or _impossible_check_now):
                                        _screen_val, _screen_detail, _screen_sum, _screen_n = _evaluate_screening(_seed_metrics_list)
                                        # Dos presupuestos distintos:
                                        # - Imposible: ¿puede el promedio de TODAS las n_total semillas ser ≤ threshold?
                                        #   Budget = threshold × n_total  (diffs ≥0, suma solo puede crecer)
                                        # - Check fijo en n_screen: ¿el promedio de las n_screen semillas ≤ u_screen?
                                        #   Budget = u_screen × n_screen  (umbral ligeramente relajado si n_screen < n_total)
                                        _budget_impossible = _es_gap_limit * _n_total
                                        _budget_fixed = _u_screen * _n_screen
                                        _remaining = max(0.0, _budget_impossible - _screen_sum)
                                        _pct_used = (_screen_sum / _budget_impossible * 100) if _budget_impossible > 0 else 0.0
                                        # Desglose del peor seed (CV-test, CV-OOB, test-OOB)
                                        _detail_str = " | ".join(
                                            f"{_pn}={_pv:.4f}" if _pv is not None else f"{_pn}=n/a"
                                            for _pn, _pv in _screen_detail.items()
                                        ) or "n/a"
                                        
                                        if not _es_impossible_mode:
                                            _budget_str = (
                                                f"prom={_screen_val:.4f}/max_prom={_u_screen:.4f} "
                                                f"(evaluado en s={_si})"
                                            )
                                        else:
                                            _budget_str = (
                                                f"prom={_screen_val:.4f} | suma={_screen_sum:.4f}/{_budget_impossible:.4f}"
                                                f" ({_pct_used:.0f}% usado, rest={_remaining:.4f})"
                                            )
                                        
                                        _last_budget_str = _budget_str
                                        # Imposible: suma ya supera el presupuesto total de n_total semillas
                                        _is_impossible = _impossible_check_now and (_screen_sum > _budget_impossible)
                                        # Check fijo: promedio de las n_screen semillas supera el umbral relajado
                                        _is_fail_fixed = _fixed_check_now and (_screen_val > _u_screen)
                                        if _is_impossible or _is_fail_fixed:
                                            _discarded_by_screening = True
                                            _last_screen_val = _screen_val
                                            _last_screen_detail = _screen_detail
                                            if _cur_var is not None:
                                                try:
                                                    if _is_impossible and _si != _n_screen:
                                                        _label_prefix = f"[INALCANZABLE ✗] semilla {_si}/{len(_all_seeds_tuning)}"
                                                    else:
                                                        _label_prefix = f"[SCREENING ✗] semilla {_si}/{len(_all_seeds_tuning)}"
                                                    _cur_var.set(
                                                        f"{_label_prefix} — #{_completed_global}/{total_candidates}"
                                                        f" DESCARTADO — {_budget_str}"
                                                        f" | peor-seed: {_detail_str}"
                                                        f" | {self._format_candidate_params_short(candidate_params)}"
                                                    )
                                                    _d = getattr(self, "_tuning_progress_dialog", None)
                                                    if _d:
                                                        self._update_tuning_status_text_widget()
                                                        _d.update_idletasks()
                                                except Exception:
                                                    pass
                                            break
                                        elif _si == _n_screen:
                                            # Pasó el screening
                                            _last_screen_val = _screen_val
                                            _last_screen_detail = _screen_detail
                                            _es_pass_stops = getattr(self, "early_stopping_pass_stops_var", None)
                                            _pass_stops_mode = bool(_es_pass_stops.get()) if _es_pass_stops else False
                                            if _pass_stops_mode:
                                                _semi_complete = True
                                                if _cur_var is not None:
                                                    try:
                                                        _cur_var.set(
                                                            f"[SCREENING ✓ SEMIINCOMPLETO] #{_completed_global}/{total_candidates}"
                                                            f" — {_budget_str}"
                                                            f" | parado en semilla {_si}/{len(_all_seeds_tuning)}"
                                                        )
                                                        _d = getattr(self, "_tuning_progress_dialog", None)
                                                        if _d:
                                                            self._update_tuning_status_text_widget()
                                                            _d.update_idletasks()
                                                    except Exception:
                                                        pass
                                                break
                                            else:
                                                # continuar con semillas restantes
                                                if _cur_var is not None:
                                                    try:
                                                        _cur_var.set(
                                                            f"[SCREENING ✓] #{_completed_global}/{total_candidates}"
                                                            f" aprobado — {_budget_str}"
                                                            f" | continuando semillas restantes..."
                                                        )
                                                        _d = getattr(self, "_tuning_progress_dialog", None)
                                                        if _d:
                                                            self._update_tuning_status_text_widget()
                                                            _d.update_idletasks()
                                                    except Exception:
                                                        pass
                                        else:
                                            # Semilla intermedia: mostrar budget acumulado en tiempo real
                                            if _cur_var is not None:
                                                try:
                                                    _cur_var.set(
                                                        f"[s{_si}/{len(_all_seeds_tuning)}] #{_completed_global}/{total_candidates}"
                                                        f" — {_budget_str}"
                                                        f" | peor: {_detail_str}"
                                                        f" | {self._format_candidate_params_short(candidate_params)}"
                                                    )
                                                    _d = getattr(self, "_tuning_progress_dialog", None)
                                                    if _d:
                                                        self._update_tuning_status_text_widget()
                                                        _d.update_idletasks()
                                                except Exception:
                                                    pass

                            if len(_seed_metrics_list) > 1:
                                _avg = self._average_seed_metrics(_seed_metrics_list)
                                # Preserve structural / non-scalar fields from original
                                for _sk in ("tau", "c_index_cv_ci", "c_index_uno_ci",
                                            "c_index_antolini_ci", "c_index_test_ci", "c_index_train_ci"):
                                    _avg.setdefault(_sk, metrics.get(_sk))
                                metrics = _avg

                                # ── Semilla representativa: la más cercana al promedio ──
                                _rep_keys = ["oob_score", "c_index_cv_mean", "c_index_test", "bss"]
                                _avg_vals = {_rk: float(_avg.get(_rk) or 0.0) for _rk in _rep_keys}
                                _best_rep_seed = _seed_vals_list[0]
                                _best_rep_dist = float("inf")
                                for _ri, (_rmet, _rseed) in enumerate(zip(_seed_metrics_list, _seed_vals_list)):
                                    _rdist = sum(
                                        abs(float(_rmet.get(_rk) or 0.0) - _avg_vals[_rk])
                                        for _rk in _rep_keys
                                    )
                                    if _rdist < _best_rep_dist:
                                        _best_rep_dist = _rdist
                                        _best_rep_seed = _rseed
                                # Guardar en candidate_params para el snapshot
                                candidate_params["seed_used"] = int(_best_rep_seed)
                                candidate_params["seeds_used"] = list(_seed_vals_list)
                                # ─────────────────────────────────────────────────────
                            else:
                                # Solo 1 semilla superviviente en el multi-seed
                                candidate_params["seed_used"] = int(_seed_vals_list[0])
                                candidate_params["seeds_used"] = list(_seed_vals_list)

                        # Caso 1 semilla total (sin multi-seed)
                        if "seed_used" not in candidate_params:
                            candidate_params["seed_used"] = int(_all_seeds_tuning[0]) if _all_seeds_tuning else int(candidate_params.get("random_state", 42))
                        if "seeds_used" not in candidate_params:
                            candidate_params["seeds_used"] = [candidate_params["seed_used"]]

                        score = metrics["c_index_cv_mean"]
                        if score is None:
                            score = metrics.get("c_index_uno")
                        if score is None:
                            score = metrics["c_index_test"]
                        if score is None:
                            score = float("-inf")

                        candidate_display = copy.deepcopy(candidate.get("display", {}))
                        candidate_display["n_estimators"] = int(candidate_params.get("n_estimators", candidate_display.get("n_estimators", _cand_trees)))
                        candidate_display["scope"] = scope_label

                        # ── Capturar VIMP del candidato actual y actualizar gráfica ──
                        try:
                            _cur_vimp2 = getattr(self, "_autotune_current_vimp", None)
                            if isinstance(_cur_vimp2, dict) and model is not None and hasattr(model, "feature_importances_"):
                                _fi2 = model.feature_importances_
                                _feat2 = list(X_encoded_scope.columns)
                                _max2 = float(max(_fi2)) if len(_fi2) > 0 and float(max(_fi2)) > 0 else 1.0
                                _cur_vimp2.clear()
                                _cur_vimp2.update({_feat2[_kk]: float(_fi2[_kk]) / _max2 for _kk in range(len(_feat2))})
                            # Actualizar gráfica al terminar el candidato completo
                            try:
                                self._update_autotune_vimp_chart()
                                _d3 = getattr(self, "_tuning_progress_dialog", None)
                                if _d3:
                                    _d3.update_idletasks()
                            except Exception:
                                pass
                        except Exception:
                            pass
                        # ────────────────────────────────────────────────────────────

                        tuning_results.append(
                            {
                                "model_id": int(_completed_global),
                                "display": candidate_display,
                                "params": copy.deepcopy(candidate_params),
                                "metrics": metrics,
                                "score": float(score),
                                "covariates": list(scope_covariates),
                                "scope": scope_label,
                                "mode": tuning_mode,
                                "fit_dataframe": clean_data_scope.copy(deep=True),
                                "encoded_columns": list(X_encoded_scope.columns),
                                "test_size": test_size_scope,
                                "discarded_by_screening": _discarded_by_screening,
                                "semi_complete": _semi_complete,
                                "screening_range": _last_screen_val,
                                "screening_detail": _last_screen_detail,
                            }                        )

                        _n_evaluated += 1
                        _was_new_best = False
                        _cur_metric_raw = metrics.get(progress_metric_key)
                        if _cur_metric_raw is not None:
                            try:
                                _cur_metric_val = float(_cur_metric_raw)
                            except (TypeError, ValueError):
                                _cur_metric_val = None
                            if _cur_metric_val is not None and np.isfinite(_cur_metric_val):
                                if _best_progress_value is None or (
                                    (progress_metric_higher_better and _cur_metric_val > _best_progress_value) or
                                    (not progress_metric_higher_better and _cur_metric_val < _best_progress_value)
                                ):
                                    _best_progress_value = _cur_metric_val
                                    _best_progress_params = copy.deepcopy(candidate_params)
                                    _was_new_best = True

                        # ── Log per-seed data al registro de historial ──
                        try:
                            _text_w = getattr(self, "_tuning_history_text", None)
                            if _text_w is not None and len(_seed_vals_list) > 1:
                                _rep_seed_log = candidate_params.get("seed_used", _seed_vals_list[0])
                                _seed_lines = []
                                for _log_i, (_log_s, _log_m) in enumerate(zip(_seed_vals_list, _seed_metrics_list)):
                                    _is_rep = (_log_s == _rep_seed_log)
                                    _star = " ★" if _is_rep else ""
                                    _seed_lines.append(
                                        f"       S{_log_i+1}(seed={_log_s}){_star}"
                                        f" OOB={self._format_metric(_log_m.get('oob_score'))}"
                                        f" CV={self._format_metric(_log_m.get('c_index_cv_mean'))}"
                                        f" Test={self._format_metric(_log_m.get('c_index_test'))}"
                                        f" BSS={self._format_metric(_log_m.get('bss'))}"
                                    )
                                _per_seed_block = "\n".join(_seed_lines)
                                _text_w.configure(state="normal")
                                _text_w.insert(tk.END, _per_seed_block + "\n")
                                _text_w.configure(state="disabled")
                        except Exception:
                            pass
                        # ─────────────────────────────────────────────────

                        self._append_auto_tuning_history_row(
                            row_index=_n_evaluated,
                            status_label=("GANADOR" if _was_new_best else "vencido"),
                            metric_label=(self.tuning_progress_metric_var.get() if hasattr(self, "tuning_progress_metric_var") else "C-Uno (IPCW)"),
                            metric_value=_cur_metric_val,
                            best_value=_best_progress_value,
                            candidate_display=candidate_display,
                            candidate_params=candidate_params,
                            duration_col=duration_col,
                            event_col=event_col,
                            covariates=scope_covariates,
                            metrics=metrics,
                            extra_note=(
                                "Incompleto/Descartado (Screening)" if _discarded_by_screening else
                                "Semiincompleto (pasó umbral, parado temprano)" if _semi_complete else
                                (
                                    f"árboles auto {(_tree_growth_history[0][0] if _tree_growth_history else candidate_params.get('n_estimators'))}"
                                    f"->{candidate_params.get('n_estimators')} ({_growth_metric_label})"
                                )
                                if auto_trees_mode else
                                ("se queda" if _was_new_best else "no supera al mejor")
                            ),
                        )

                        if tuning_results and (
                            _n_evaluated <= 3
                            or (_n_evaluated % _sync_publish_every) == 0
                            or bool(self._tuning_pause_requested)
                        ):
                            self._sync_live_autotune_snapshots(
                                tuning_results=tuning_results,
                                profile_name=selected_profile,
                                recommended_profile=recommended_profile,
                                requested_test_size=requested_test_size,
                                duration_col=duration_col,
                                event_col=event_col,
                                covariates=covariates,
                                tuning_mode=tuning_mode,
                            )

                        # --- Liberar memoria después de cada candidato ---
                        # Los modelos NO se guardan en tuning_results; liberarlos
                        # de inmediato evita que la RAM se sature con decenas de
                        # modelos de 200-500 MB cada uno.
                        model = None
                        if _cache_key is not None and len(_scope_warm_cache) > 1:
                            # Warm-cache: conservar solo la entrada más reciente
                            for _stale_k in [k for k in list(_scope_warm_cache) if k != _cache_key]:
                                del _scope_warm_cache[_stale_k]
                        # Recolección de basura cada 8 candidatos
                        if _n_evaluated % 8 == 0:
                            import gc as _gc
                            _gc.collect()
                        # -------------------------------------------------

                    except InterruptedError:
                        cancel_requested = True
                        break
                    except RuntimeError as candidate_exc:
                        _err_text = str(candidate_exc or "").lower()
                        if "sin respuesta" in _err_text or "evitar ciclo" in _err_text:
                            cancel_requested = True
                            self._tuning_cancel_requested = True
                            warnings_list.append(
                                f"{scope_label} | configuración {_completed_global}/{total_candidates} abortada: {candidate_exc}"
                            )
                            break
                        _n_evaluated += 1
                        candidate_display = copy.deepcopy(candidate.get("display", {}))
                        candidate_display["scope"] = scope_label
                        self._append_auto_tuning_history_row(
                            row_index=_n_evaluated,
                            status_label="error",
                            metric_label=(self.tuning_progress_metric_var.get() if hasattr(self, "tuning_progress_metric_var") else "C-Uno (IPCW)"),
                            metric_value=None,
                            best_value=_best_progress_value,
                            candidate_display=candidate_display,
                            candidate_params=candidate_params,
                            duration_col=duration_col,
                            event_col=event_col,
                            covariates=scope_covariates,
                            extra_note=f"omitido: {candidate_exc}",
                        )
                        warnings_list.append(
                            f"{scope_label} | configuración {_completed_global}/{total_candidates} omitida: {candidate_exc}"
                        )
                    except Exception as candidate_exc:
                        _n_evaluated += 1
                        candidate_display = copy.deepcopy(candidate.get("display", {}))
                        candidate_display["scope"] = scope_label
                        self._append_auto_tuning_history_row(
                            row_index=_n_evaluated,
                            status_label="error",
                            metric_label=(self.tuning_progress_metric_var.get() if hasattr(self, "tuning_progress_metric_var") else "C-Uno (IPCW)"),
                            metric_value=None,
                            best_value=_best_progress_value,
                            candidate_display=candidate_display,
                            candidate_params=candidate_params,
                            duration_col=duration_col,
                            event_col=event_col,
                            covariates=scope_covariates,
                            extra_note=f"omitido: {candidate_exc}",
                        )
                        warnings_list.append(
                            f"{scope_label} | configuración {_completed_global}/{total_candidates} omitida: {candidate_exc}"
                        )

                    elapsed_seconds = time.perf_counter() - tuning_start_time
                    _trees_completed += int(_trees_used_candidate)
                    # Live preview of 5-phase clinical ranking
                    _balanced_text = None
                    if len(tuning_results) >= 3:
                        _preview = list(tuning_results)
                        _preview, _pstats = self._rank_tuning_results(_preview)
                        _bal_m = _preview[0].get("metrics", {})
                        _bal_p = _preview[0].get("params", {})
                        _bal_parts = []
                        for _mk, _mlbl in [("c_index_cv_mean", "CV"), ("c_index_uno", "C-Uno"),
                                            ("c_index_test", "C-test"), ("oob_score", "OOB"), ("bss", "BSS")]:
                            _mv = _bal_m.get(_mk)
                            if _mv is not None and np.isfinite(_mv):
                                _bal_parts.append(f"{_mlbl}={_mv:.4f}")
                        _surv = _pstats.get("survivors", len(_preview))
                        _bal_scope = _preview[0].get("scope", "")
                        _bal_covs = _preview[0].get("covariates", [])
                        _balanced_text = (
                            f"Mejor clínico ({_surv} aptos de {_pstats.get('total', len(_preview))}): "
                            f"{self._format_candidate_params_full(_bal_p, scope=_bal_scope, covariates=_bal_covs)} | {', '.join(_bal_parts)}"
                        )
                    self._update_auto_tuning_progress_dialog(
                        profile_name=selected_profile,
                        n_rows=len(clean_data_scope),
                        completed=_completed_global,
                        total=total_candidates,
                        elapsed_seconds=elapsed_seconds,
                        cancel_requested=bool(self._tuning_cancel_requested),
                        current_params=_cand_params_display,
                        best_metric_value=_best_progress_value,
                        best_metric_params=_best_progress_params,
                        n_evaluated=_n_evaluated,
                        balanced_text=_balanced_text,
                        trees_completed=_trees_completed,
                        trees_total=_total_trees,
                        current_tree_text=_current_tree_text,
                        current_covariates=list(scope_covariates),
                        current_scope=scope_label,
                    )
                    if bool(self._tuning_skip_scope_requested):
                        warnings_list.append(
                            f"{scope_label}: combinación de variables saltada por el usuario."
                        )
                        self._tuning_skip_scope_requested = False
                        self._reset_auto_tuning_skip_button()
                        break
                    if bool(self._tuning_cancel_requested):
                        cancel_requested = True
                        break

                if cancel_requested:
                    break

            self._auto_tuning_in_progress = False

            if cancel_requested and not tuning_results:
                cancel_message = (
                    "El tuning automático RSF fue cancelado antes de completar una configuración.\n"
                    f"Casos analizados: {len(scope_contexts[0]['clean_data'])}\n"
                    f"Combinaciones preparadas: {total_candidates}"
                )
                self.auto_tuning_status_var.set("Tuning automático cancelado por el usuario.")
                self.results_text.delete("1.0", tk.END)
                self.results_text.insert(tk.END, cancel_message)
                messagebox.showinfo("Tuning RSF cancelado", cancel_message)
                return

            if not tuning_results:
                raise ValueError("No se pudieron evaluar configuraciones de RSF.")

            _previous_active_saved_index = self.active_saved_model_index if isinstance(self.active_saved_model_index, int) else None

            # ── Finalising: close dialog early so the UI is responsive ──
            progress_note_var = getattr(self, "_tuning_progress_note_var", None)
            if progress_note_var is not None:
                try:
                    progress_note_var.set(
                        "Finalizando: guardando resultados parciales..."
                        if cancel_requested else
                        "Finalizando: guardando resultados y aplicando mejor modelo..."
                    )
                except Exception:
                    pass
            self._flush_tuning_dialog_events()
            self._close_auto_tuning_progress_dialog()

            tuning_results, rank_stats = self._rank_tuning_results(tuning_results)

            best_result = tuning_results[0]
            tuning_summary = self._build_tuning_summary(
                tuning_results,
                best_result,
                profile_name=selected_profile,
                n_rows=len(best_result.get("fit_dataframe", filtered_data)),
                recommended_profile=recommended_profile,
                test_size=best_result.get("test_size", requested_test_size),
                duration_col=duration_col,
                event_col=event_col,
                covariates=best_result.get("covariates", covariates),
                tuning_mode=tuning_mode,
            )
            # Append ranking stats to summary
            if rank_stats:
                _purged_bss = rank_stats.get("purged_bss", 0)
                _purged_gap = rank_stats.get("purged_gap", 0)
                _survivors = rank_stats.get("survivors", len(tuning_results))
                if _purged_bss or _purged_gap:
                    tuning_summary += (
                        f"\n\n── Filtro clínico de calidad ──\n"
                        f"  Modelos evaluados: {rank_stats.get('total', len(tuning_results))}\n"
                    )
                    if _purged_bss:
                        tuning_summary += f"  Descartados por BSS ≤ {self._RANK_BSS_MIN} (calibración muy pobre): {_purged_bss}\n"
                    if _purged_gap:
                        _cv_lbl, _test_lbl = self._resolve_cv_metric_labels()
                        tuning_summary += f"  Descartados por brecha |{_test_lbl} − {_cv_lbl}| > {self._RANK_OVERFIT_GAP_MAX} (inestabilidad): {_purged_gap}\n"
                    tuning_summary += f"  Sobrevivientes para ranking final: {_survivors}\n"
                    if rank_stats.get("fallback"):
                        tuning_summary += "  ⚠ Todos fueron filtrados; se conservó la lista completa como respaldo.\n"
            if cancel_requested:
                tuning_summary += (
                    f"\n\nNota: el usuario canceló el proceso. Se conservaron {len(tuning_results)} "
                    f"de {total_candidates} configuraciones evaluadas."
                )
            if warnings_list:
                tuning_summary += "\n\nAvisos de preparación:\n" + "\n".join(f"- {item}" for item in warnings_list)
            if auto_trees_mode:
                tuning_summary += (
                    "\n\nÁrboles (n_estimators): selección automática por meseta "
                    f"(inicio={self._AUTO_TREE_START}, paso={self._AUTO_TREE_STEP}, "
                    f"mín={self._AUTO_TREE_MIN}, máx={auto_tree_max_limit}, "
                    f"delta={self._AUTO_TREE_DELTA_MIN}, paciencia={self._AUTO_TREE_PATIENCE})."
                )

            results_to_store = list(tuning_results)
            if cancel_requested:
                max_keep = int(getattr(self, "_max_cancel_autotune_snapshots", 180) or 180)
                max_keep = max(20, max_keep)
                if len(results_to_store) > max_keep:
                    results_to_store = list(results_to_store[:max_keep])
                    if best_result not in results_to_store:
                        results_to_store[-1] = best_result
                    tuning_summary += (
                        f"\n\nNota de rendimiento: se guardaron {len(results_to_store)} snapshots "
                        f"(de {len(tuning_results)} evaluados) para mantener fluida la UI tras cancelar."
                    )

            if cancel_requested and self.saved_models:
                _previous_active_snapshot = None
                if _previous_active_saved_index is not None and 0 <= _previous_active_saved_index < len(self.saved_models):
                    _previous_active_snapshot = self.saved_models[_previous_active_saved_index]
                self.saved_models[:] = [
                    snapshot for snapshot in self.saved_models
                    if not bool(snapshot.get("autotune_partial", False))
                ]
                if _previous_active_snapshot is not None:
                    _found_idx = self._find_snapshot_identity(_previous_active_snapshot, self.saved_models)
                    if _found_idx is not None:
                        self.active_saved_model_index = _found_idx
                    else:
                        self.active_saved_model_index = None

            if self.saved_models:
                _active_snapshot = None
                if isinstance(self.active_saved_model_index, int) and 0 <= self.active_saved_model_index < len(self.saved_models):
                    _active_snapshot = self.saved_models[self.active_saved_model_index]
                self.saved_models[:] = [
                    snapshot for snapshot in self.saved_models
                    if not bool(snapshot.get("autotune_live", False))
                ]
                _found_active_idx = self._find_snapshot_identity(_active_snapshot, self.saved_models)
                if _found_active_idx is not None:
                    self.active_saved_model_index = _found_active_idx
                elif self.saved_models:
                    self.active_saved_model_index = len(self.saved_models) - 1
                else:
                    self.active_saved_model_index = None

            best_saved_index = None
            for result in results_to_store:
                snapshot_params = copy.deepcopy(result.get("params", {}))
                display_max_features = str(result.get("display", {}).get("max_features", "")).strip().lower()
                if display_max_features == "all":
                    snapshot_params.pop("max_features", None)
                elif display_max_features:
                    snapshot_params["max_features"] = result.get("display", {}).get("max_features")

                snapshot_params["test_size"] = result.get("test_size", requested_test_size)
                snapshot_params["tau_mode"] = self.tau_mode_var.get() if hasattr(self, "tau_mode_var") else "Percentil 90"
                snapshot_params["tau_manual"] = self.tau_manual_var.get() if hasattr(self, "tau_manual_var") else ""
                snapshot_params["optimization_metric"] = self.optimization_metric_var.get() if hasattr(self, "optimization_metric_var") else "Harrell C-index"
                snapshot_params["tuning_scope"] = result.get("scope", result.get("display", {}).get("scope", "-"))
                snapshot_params["tuning_mode"] = result.get("mode", tuning_mode)
                snapshot = {
                    "label": f"AutoTune #{len(self.saved_models) + 1}",
                    "autotune_partial": bool(cancel_requested),
                    "params": snapshot_params,
                    "metrics": copy.deepcopy(result.get("metrics", {})),
                    "scope": result.get("scope", result.get("display", {}).get("scope", "-")),
                    "mode": result.get("mode", tuning_mode),
                    "report_text": self._build_tuning_summary(
                        [result],
                        result,
                        profile_name=selected_profile,
                        n_rows=len(result.get("fit_dataframe", filtered_data)),
                        recommended_profile=recommended_profile,
                        test_size=result.get("test_size", requested_test_size),
                        duration_col=duration_col,
                        event_col=event_col,
                        covariates=result.get("covariates", covariates),
                        tuning_mode=tuning_mode,
                    ),
                    "latest_fit_dataframe": result.get("fit_dataframe", filtered_data).copy(deep=True),
                    "latest_duration_col": duration_col,
                    "latest_event_col": event_col,
                    "latest_covariates": list(result.get("covariates", covariates)),
                    "latest_encoded_columns": list(result.get("encoded_columns", [])),
                    "latest_drop_first": bool(self.drop_first_var.get()) if hasattr(self, "drop_first_var") else True,
                    "variable_configs": copy.deepcopy(getattr(self, "variable_configs", {})),
                }
                self.saved_models.append(snapshot)
                if result is best_result:
                    best_saved_index = len(self.saved_models) - 1

            if cancel_requested and _previous_active_saved_index is not None and 0 <= _previous_active_saved_index < len(self.saved_models):
                self.active_saved_model_index = _previous_active_saved_index
            else:
                self.active_saved_model_index = best_saved_index if best_saved_index is not None else (len(self.saved_models) - 1 if self.saved_models else None)
            self._refresh_saved_models_tree(rerank=False)

            status_prefix = "Tuning parcial listo" if cancel_requested else f"Perfil {selected_profile} listo"
            _cv_lbl, _test_lbl = self._resolve_cv_metric_labels()
            self.auto_tuning_status_var.set(
                f"{status_prefix} ({tuning_mode}): "
                f"{_cv_lbl}={self._format_c_index_display(best_result.get('metrics', {}).get('c_index_cv_mean'), best_result.get('metrics', {}).get('c_index_cv_ci'), decimals=3)}, "
                f"{_test_lbl}={self._format_c_index_display(best_result.get('metrics', {}).get(self._resolve_clinical_stability_metric_keys()[1]), best_result.get('metrics', {}).get('c_index_test_ci'), decimals=3)}"
            )
            if cancel_requested:
                self.results_text.delete("1.0", tk.END)
                self.results_text.insert(
                    tk.END,
                    tuning_summary + "\n\nCancelación segura: se conservaron los modelos parciales y se mantuvo el modelo activo previo."
                )
                messagebox.showinfo(
                    "Tuning RSF cancelado",
                    f"Se canceló el tuning y se conservaron {len(tuning_results)} de {total_candidates} configuraciones ya evaluadas.",
                )
            else:
                self.results_text.delete("1.0", tk.END)
                self.results_text.insert(
                    tk.END,
                    tuning_summary + "\n\nTuning completado exitosamente.\n\nLos mejores modelos (incluyendo multi-semilla) han sido añadidos a la lista lateral de 'Modelos guardados'.\n\n➜ Para visualizar un modelo, selecciónalo en la lista y presiona 'Cargar modelo' (o haz doble clic sobre él)."
                )
                messagebox.showinfo(
                    "Tuning RSF completado",
                    f"Se completó el tuning de manera exitosa.\n\nSe evaluaron {len(tuning_results)} configuraciones.\nLos modelos candidatos han sido listados en el panel izquierdo (Modelos Guardados).\n\nHaz doble clic en cualquiera de ellos para cargarlo rápidamente con su semilla representativa.",
                )
                if hasattr(self, "notebook") and hasattr(self, "results_tab"):
                    try:
                        self.notebook.select(self.results_tab)
                    except Exception:
                        pass
        except Exception as exc:
            self.auto_tuning_status_var.set(f"Tuning automático falló: {exc}")
            traceback_text = traceback.format_exc(limit=3)
            self.results_text.delete("1.0", tk.END)
            self.results_text.insert(tk.END, f"No se pudo completar el tuning automático RSF:\n{exc}\n\n{traceback_text}")
            messagebox.showerror("Error en tuning RSF", f"Ocurrió un error durante el tuning automático:\n{exc}")
        finally:
            self._auto_tuning_in_progress = False
            self._close_auto_tuning_progress_dialog()
            self._tuning_cancel_requested = False
            self._tuning_pause_requested = False
            self._tuning_skip_scope_requested = False

    # ------------------------------------------------------------------
    # Model history
    # ------------------------------------------------------------------
    def _apply_snapshot_params(self, snapshot_params):
        if not isinstance(snapshot_params, dict):
            return

        if "n_estimators" in snapshot_params:
            self.n_estimators_var.set(self._coerce_int(snapshot_params.get("n_estimators"), 300, minimum=10))
        if "min_samples_leaf" in snapshot_params:
            self.min_samples_leaf_var.set(self._coerce_int(snapshot_params.get("min_samples_leaf"), 10, minimum=1))
        if "min_samples_split" in snapshot_params:
            self.min_samples_split_var.set(self._coerce_int(snapshot_params.get("min_samples_split"), 10, minimum=2))

        max_features_value = snapshot_params.get("max_features", None)
        if max_features_value in (None, "-", "", "None"):
            self.max_features_var.set("all")
            self.max_features_manual_var.set("")
        else:
            max_features_text = str(max_features_value).strip()
            if max_features_text in {"sqrt", "log2", "all", "0.3", "0.5", "0.7"}:
                self.max_features_var.set(max_features_text)
                self.max_features_manual_var.set("")
            else:
                self.max_features_var.set("manual")
                self.max_features_manual_var.set(max_features_text)

        self.max_depth_var.set("" if snapshot_params.get("max_depth") in (None, "", "-") else str(snapshot_params.get("max_depth")))
        self.max_leaf_nodes_var.set("" if snapshot_params.get("max_leaf_nodes") in (None, "", "-") else str(snapshot_params.get("max_leaf_nodes")))
        self.max_samples_var.set("" if snapshot_params.get("max_samples") in (None, "", "-") else str(snapshot_params.get("max_samples")))

        if "bootstrap" in snapshot_params:
            self.bootstrap_var.set(bool(snapshot_params.get("bootstrap")))
        if "oob_score" in snapshot_params:
            self.oob_score_var.set(bool(snapshot_params.get("oob_score")))
        if "n_jobs" in snapshot_params:
            self.n_jobs_var.set(self._coerce_int(snapshot_params.get("n_jobs"), -1))
        if "seeds_used" in snapshot_params and isinstance(snapshot_params["seeds_used"], list) and snapshot_params["seeds_used"]:
            self.random_state_var.set(", ".join(map(str, snapshot_params["seeds_used"])))
        elif "random_state" in snapshot_params:
            self.random_state_var.set(str(snapshot_params.get("random_state")))
        if "test_size" in snapshot_params:
            self.test_size_var.set(self._coerce_float(snapshot_params.get("test_size"), 0.25, minimum=0.0, maximum=0.95))
        if "optimization_metric" in snapshot_params and hasattr(self, "optimization_metric_var"):
            try:
                self.optimization_metric_var.set(str(snapshot_params.get("optimization_metric") or "Harrell C-index"))
            except Exception:
                self.optimization_metric_var.set("Harrell C-index")
            self._on_optimization_metric_change()
        if "drop_first" in snapshot_params and hasattr(self, "drop_first_var"):
            self.drop_first_var.set(bool(snapshot_params.get("drop_first")))
        if "stratify_event" in snapshot_params and hasattr(self, "stratify_event_var"):
            self.stratify_event_var.set(bool(snapshot_params.get("stratify_event")))
        if "missing_strategy" in snapshot_params and hasattr(self, "missing_strategy_var"):
            self.missing_strategy_var.set(str(snapshot_params.get("missing_strategy") or "Imputar mediana/moda"))
        if "tau_mode" in snapshot_params and hasattr(self, "tau_mode_var"):
            self.tau_mode_var.set(self._normalize_tau_mode(snapshot_params.get("tau_mode") or "Percentil 90"))
            self._on_tau_mode_change()
        if "tau_manual" in snapshot_params and hasattr(self, "tau_manual_var"):
            self.tau_manual_var.set("" if snapshot_params.get("tau_manual") in (None, "None") else str(snapshot_params.get("tau_manual")))

    def _restore_saved_snapshot_ui_state(self):
        columns = []
        for df_source in (getattr(self, "data", None), getattr(self, "latest_fit_dataframe", None)):
            if isinstance(df_source, pd.DataFrame):
                for col_name in df_source.columns.tolist():
                    if col_name not in columns:
                        columns.append(col_name)
        for cov_name in getattr(self, "latest_covariates", []):
            if cov_name not in columns:
                columns.append(cov_name)

        if hasattr(self, "duration_combo"):
            self.duration_combo["values"] = columns
        if hasattr(self, "event_combo"):
            self.event_combo["values"] = [""] + columns
        if hasattr(self, "duration_var"):
            duration_value = self.latest_duration_col or ""
            self.duration_var.set(duration_value if (not columns or duration_value in columns) else "")
        if hasattr(self, "event_var"):
            event_value = self.latest_event_col or ""
            self.event_var.set(event_value if (not columns or event_value in columns) else "")
        if hasattr(self, "drop_first_var"):
            self.drop_first_var.set(bool(getattr(self, "latest_drop_first", bool(self.drop_first_var.get()))))

        if hasattr(self, "covariates_listbox"):
            self.covariates_listbox.delete(0, tk.END)
            for idx, col_name in enumerate(columns):
                self.covariates_listbox.insert(tk.END, col_name)
                if col_name in getattr(self, "latest_covariates", []):
                    try:
                        self.covariates_listbox.selection_set(idx)
                    except Exception:
                        pass

    def _rebuild_loaded_snapshot_plot_state(self, snapshot=None):
        """Rebuild missing chart payloads for legacy snapshots when possible."""
        if self.model is None or not isinstance(self.latest_fit_dataframe, pd.DataFrame):
            return
        if not isinstance(self.latest_covariates, list) or not self.latest_covariates:
            return
        if any(col not in self.latest_fit_dataframe.columns for col in self.latest_covariates):
            return

        try:
            encoded_full = self._encode_prediction_frame(self.latest_fit_dataframe[self.latest_covariates].copy())
        except Exception:
            return
        if encoded_full is None or encoded_full.empty:
            return

        try:
            full_preds = np.asarray(self.model.predict(encoded_full), dtype=float)
        except Exception:
            return

        if self.latest_prediction_df is None or getattr(self.latest_prediction_df, "empty", True):
            try:
                if self.latest_duration_col in self.latest_fit_dataframe.columns and self.latest_event_col in self.latest_fit_dataframe.columns:
                    prediction_df = self.latest_fit_dataframe[[self.latest_duration_col, self.latest_event_col]].copy()
                    prediction_df["risk_score"] = full_preds
                    try:
                        labels = ["Q1 bajo", "Q2 medio-bajo", "Q3 medio-alto", "Q4 alto"]
                        q = min(4, max(2, prediction_df["risk_score"].nunique()))
                        prediction_df["risk_group"] = pd.qcut(
                            prediction_df["risk_score"].rank(method="first"),
                            q=q,
                            labels=labels[:q],
                            duplicates="drop",
                        )
                    except Exception:
                        prediction_df["risk_group"] = "Grupo único"
                    self.latest_prediction_df = prediction_df
                    if isinstance(snapshot, dict):
                        snapshot["latest_prediction_df"] = prediction_df.copy(deep=True)
            except Exception:
                pass

        if not self.latest_survival_profiles:
            try:
                self.latest_survival_profiles = self._build_survival_profiles(self.model, encoded_full, full_preds)
                if isinstance(snapshot, dict):
                    snapshot["latest_survival_profiles"] = copy.deepcopy(self.latest_survival_profiles)
            except Exception:
                pass

        needs_calibration = not isinstance(self.latest_calibration_df, pd.DataFrame) or self.latest_calibration_df.empty
        needs_brier = not isinstance(self.latest_brier_df, pd.DataFrame) or self.latest_brier_df.empty
        if not (needs_calibration or needs_brier):
            return

        encoded_eval, y_eval = self._build_loaded_snapshot_eval_payload(encoded_full)
        if encoded_eval is None or y_eval is None or len(encoded_eval) == 0 or len(y_eval) == 0:
            return

        resolved_eval_time = self._resolve_loaded_snapshot_eval_time(y_eval)
        if resolved_eval_time is not None and (self.latest_eval_time is None or not np.isfinite(self.latest_eval_time)):
            self.latest_eval_time = float(resolved_eval_time)
            if isinstance(snapshot, dict):
                snapshot["latest_eval_time"] = self.latest_eval_time

        if needs_calibration and resolved_eval_time is not None:
            try:
                curve_payload = self._compute_survival_curve_confidence_bands(self.model, encoded_eval)
                calibration_payload = self._extract_curve_values_at_time(curve_payload, resolved_eval_time, output_type="survival")
                if calibration_payload:
                    effective_eval_time = float(calibration_payload.get("eval_time", resolved_eval_time))
                    predicted_survival = np.asarray(calibration_payload.get("point_estimate", []), dtype=float)
                    calibration_df = self._summarize_calibration(y_eval, predicted_survival, effective_eval_time)
                    if isinstance(calibration_df, pd.DataFrame) and not calibration_df.empty:
                        self.latest_calibration_df = calibration_df
                        self.latest_eval_time = effective_eval_time
                        if isinstance(snapshot, dict):
                            snapshot["latest_calibration_df"] = calibration_df.copy(deep=True)
                            snapshot["latest_eval_time"] = self.latest_eval_time
            except Exception:
                pass

        if needs_brier:
            rebuilt_brier_df = pd.DataFrame()
            try:
                eval_times = self._build_evaluation_time_grid(y_eval, y_eval, tau=None)
                if callable(brier_score) and eval_times is not None:
                    survival_functions = list(self.model.predict_survival_function(encoded_eval))
                    survival_matrix = np.asarray(
                        [self._evaluate_step_function(step_fn, eval_times) for step_fn in survival_functions],
                        dtype=float,
                    )
                    brier_result = brier_score(y_eval, y_eval, survival_matrix, eval_times)
                    if isinstance(brier_result, tuple) and len(brier_result) >= 2:
                        _, brier_values = brier_result
                        brier_values = np.asarray(brier_values, dtype=float)
                        if brier_values.size == len(eval_times):
                            rebuilt_brier_df = pd.DataFrame({"time": eval_times, "brier_score": brier_values})
                            if self.latest_eval_time is None and len(eval_times) > 0:
                                self.latest_eval_time = float(eval_times[len(eval_times) // 2])
            except Exception:
                rebuilt_brier_df = pd.DataFrame()

            if rebuilt_brier_df.empty:
                rebuilt_brier_df = self._build_brier_fallback_dataframe()
                if self.latest_eval_time is None and not rebuilt_brier_df.empty:
                    try:
                        self.latest_eval_time = float(rebuilt_brier_df.iloc[0]["time"])
                    except Exception:
                        pass

            if not rebuilt_brier_df.empty:
                self.latest_brier_df = rebuilt_brier_df
                if isinstance(snapshot, dict):
                    snapshot["latest_brier_df"] = rebuilt_brier_df.copy(deep=True)
                    snapshot["latest_eval_time"] = self.latest_eval_time

    def _restore_saved_snapshot_state(self, snapshot):
        if not isinstance(snapshot, dict):
            return False

        feature_importance_df = snapshot.get("feature_importance_df")
        latest_prediction_df = snapshot.get("latest_prediction_df")
        latest_calibration_df = snapshot.get("latest_calibration_df")
        latest_brier_df = snapshot.get("latest_brier_df")
        latest_fit_dataframe = snapshot.get("latest_fit_dataframe")
        report_text = snapshot.get("report_text", "Sin reporte guardado.")

        # A snapshot is restorable only if it has actual model outputs to display
        # (model object, feature importances, or predictions). Snapshots that only
        # contain params + metrics + fit_dataframe (e.g. AutoTune candidates) are
        # NOT restorable and must trigger a retrain via the fallback path in
        # _load_selected_saved_model.
        has_model_outputs = (
            snapshot.get("model") is not None
            or (isinstance(feature_importance_df, pd.DataFrame) and len(feature_importance_df) > 0)
            or (isinstance(latest_prediction_df, pd.DataFrame) and len(latest_prediction_df) > 0)
        )
        has_saved_view_state = has_model_outputs and any([
            isinstance(latest_fit_dataframe, pd.DataFrame),
            bool(snapshot.get("metrics")),
            bool(str(report_text).strip()),
        ])
        if not has_saved_view_state:
            return False

        self.model = snapshot.get("model")
        self.results = copy.deepcopy(snapshot.get("metrics", {}))
        self.feature_importance_df = feature_importance_df.copy(deep=True) if isinstance(feature_importance_df, pd.DataFrame) else pd.DataFrame()
        self.latest_prediction_df = latest_prediction_df.copy(deep=True) if isinstance(latest_prediction_df, pd.DataFrame) else None
        self.latest_survival_profiles = copy.deepcopy(snapshot.get("latest_survival_profiles", []))
        self.latest_calibration_df = latest_calibration_df.copy(deep=True) if isinstance(latest_calibration_df, pd.DataFrame) else pd.DataFrame()
        self.latest_brier_df = latest_brier_df.copy(deep=True) if isinstance(latest_brier_df, pd.DataFrame) else pd.DataFrame()
        self.latest_eval_time = snapshot.get("latest_eval_time")
        self.latest_fit_dataframe = latest_fit_dataframe.copy(deep=True) if isinstance(latest_fit_dataframe, pd.DataFrame) else None
        self.latest_duration_col = snapshot.get("latest_duration_col")
        self.latest_event_col = snapshot.get("latest_event_col")
        self.latest_covariates = list(snapshot.get("latest_covariates", []))
        self.latest_encoded_columns = list(snapshot.get("latest_encoded_columns", []))
        self.latest_drop_first = bool(snapshot.get("latest_drop_first", bool(self.drop_first_var.get())))
        self.latest_tuning_summary = snapshot.get("tuning_summary", "")
        self.latest_report_text = str(report_text)
        self.variable_configs = copy.deepcopy(snapshot.get("variable_configs", {}))

        # Solo intentar reconstrucciones que dependan del modelo cuando exista.
        if self.model is not None and isinstance(self.latest_fit_dataframe, pd.DataFrame):
            self._rebuild_loaded_snapshot_plot_state(snapshot)

        self._restore_saved_snapshot_ui_state()
        self._sync_plot_covariate_selectors()
        if hasattr(self, "results_text"):
            self.results_text.delete("1.0", tk.END)
            self.results_text.insert(tk.END, self.latest_report_text)
        self.plot_feature_importance()
        self.plot_risk_groups_km()
        self.plot_survival_profiles()
        self.plot_variable_impact()
        self.plot_calibration()
        self.plot_brier_curve()
        if hasattr(self, "tree_fig") and hasattr(self, "tree_canvas"):
            self.plot_single_tree()
        self.plot_minimal_depth()
        self._sync_pdp_covariate_selector()
        self.plot_pdp()
        return True

    def _toggle_covariate_on_click(self, event):
        if not hasattr(self, "covariates_listbox"):
            return None
        lb = self.covariates_listbox
        idx = lb.nearest(event.y)
        if idx < 0 or idx >= lb.size():
            return "break"

        if lb.selection_includes(idx):
            lb.selection_clear(idx)
        else:
            lb.selection_set(idx)
        lb.activate(idx)
        return "break"

    def _get_selected_saved_model_index(self):
        selected_items = self.saved_models_tree.selection() if hasattr(self, "saved_models_tree") else ()
        if selected_items:
            try:
                return int(selected_items[0])
            except Exception:
                return None
        if self.active_saved_model_index is not None and 0 <= self.active_saved_model_index < len(self.saved_models):
            return int(self.active_saved_model_index)
        if self.saved_models:
            return len(self.saved_models) - 1
        return None

    def _save_selected_model_to_file(self):
        selected_index = self._get_selected_saved_model_index()
        if selected_index is None or not self.saved_models:
            messagebox.showwarning("Guardar modelo RSF", "Selecciona un modelo de la lista para guardarlo en disco.")
            return

        snapshot = self.saved_models[selected_index]
        raw_name = str(snapshot.get("label", f"RSF_{selected_index + 1}"))
        safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", raw_name).strip("._") or f"RSF_{selected_index + 1}"
        file_path = filedialog.asksaveasfilename(
            title="Guardar modelo RSF",
            defaultextension=".rsf.pkl",
            initialfile=f"{safe_name}.rsf.pkl",
            filetypes=[("Modelo RSF de Mathabs", "*.rsf.pkl"), ("Archivo Pickle", "*.pkl"), ("Todos los archivos", "*.*")],
        )
        if not file_path:
            return

        try:
            payload = {
                "matabs_type": "RSFSnapshot",
                "version": 1,
                "snapshot": copy.deepcopy(snapshot),
            }
            with open(file_path, "wb") as handle:
                pickle.dump(payload, handle)
            messagebox.showinfo("Modelo RSF guardado", f"Se guardó el modelo en:\n{file_path}")
        except Exception as exc:
            messagebox.showerror("Error al guardar RSF", f"No se pudo guardar el modelo:\n{exc}")

    def _save_all_models_to_file(self):
        if not self.saved_models:
            messagebox.showwarning("Guardar modelos RSF", "No hay modelos en memoria para guardar.")
            return

        file_path = filedialog.asksaveasfilename(
            title="Guardar TODOS los modelos RSF",
            defaultextension=".rsf.bundle.pkl",
            initialfile="RSF_modelos_todos.rsf.bundle.pkl",
            filetypes=[
                ("Paquete de modelos RSF", "*.rsf.bundle.pkl"),
                ("Archivo Pickle", "*.pkl"),
                ("Todos los archivos", "*.*"),
            ],
        )
        if not file_path:
            return

        try:
            payload = {
                "matabs_type": "RSFSnapshotList",
                "version": 1,
                "snapshots": copy.deepcopy(self.saved_models),
            }
            with open(file_path, "wb") as handle:
                pickle.dump(payload, handle)
            messagebox.showinfo(
                "Modelos RSF guardados",
                f"Se guardaron {len(self.saved_models)} modelos en:\n{file_path}",
            )
        except Exception as exc:
            messagebox.showerror("Error al guardar RSF", f"No se pudieron guardar los modelos:\n{exc}")

    def _import_saved_model_from_file(self):
        file_path = filedialog.askopenfilename(
            title="Importar modelo RSF guardado",
            filetypes=[
                ("Modelo RSF de Mathabs", "*.rsf.pkl"),
                ("Paquete de modelos RSF", "*.rsf.bundle.pkl"),
                ("Archivo Pickle", "*.pkl"),
                ("Todos los archivos", "*.*"),
            ],
        )
        if not file_path:
            return

        try:
            with open(file_path, "rb") as handle:
                payload = pickle.load(handle)

            snapshots_batch = None
            if isinstance(payload, dict) and payload.get("matabs_type") == "RSFSnapshotList":
                snapshots_batch = payload.get("snapshots", [])
            elif isinstance(payload, dict) and isinstance(payload.get("snapshots"), list):
                snapshots_batch = payload.get("snapshots", [])

            if isinstance(snapshots_batch, list):
                imported_count = 0
                for batch_idx, raw_snapshot in enumerate(snapshots_batch, start=1):
                    if not isinstance(raw_snapshot, dict):
                        continue
                    imported_snapshot = copy.deepcopy(raw_snapshot)
                    if not imported_snapshot.get("label"):
                        imported_snapshot["label"] = f"{os.path.splitext(os.path.basename(file_path))[0]} #{batch_idx}"
                    self.saved_models.append(imported_snapshot)
                    imported_count += 1

                if imported_count <= 0:
                    raise ValueError("El archivo no contiene snapshots RSF válidos.")

                self.active_saved_model_index = len(self.saved_models) - 1
                self._refresh_saved_models_tree()
                if hasattr(self, "saved_models_tree") and self.active_saved_model_index is not None:
                    self.saved_models_tree.selection_set(str(self.active_saved_model_index))
                self._load_selected_saved_model()
                messagebox.showinfo(
                    "Modelos RSF importados",
                    f"Se importaron {imported_count} modelos desde:\n{file_path}",
                )
                return

            snapshot = payload.get("snapshot") if isinstance(payload, dict) and "snapshot" in payload else payload
            if not isinstance(snapshot, dict):
                raise ValueError("El archivo no contiene un modelo RSF válido de Mathabs.")

            imported_snapshot = copy.deepcopy(snapshot)
            if not imported_snapshot.get("label"):
                imported_snapshot["label"] = os.path.splitext(os.path.basename(file_path))[0]

            self.saved_models.append(imported_snapshot)
            self.active_saved_model_index = len(self.saved_models) - 1
            self._refresh_saved_models_tree()
            if hasattr(self, "saved_models_tree"):
                self.saved_models_tree.selection_set(str(self.active_saved_model_index))
            self._load_selected_saved_model()
            messagebox.showinfo("Modelo RSF importado", f"Se cargó el modelo desde:\n{file_path}")
        except Exception as exc:
            messagebox.showerror("Error al importar RSF", f"No se pudo importar el modelo:\n{exc}")

    def _store_snapshot_direct(self, rsf_params, report_text, test_size,
                                metrics, model, imp_df, pred_df, profiles,
                                calibration_df=None, brier_df=None, eval_time=None, fit_df=None):
        """Store a model snapshot without relying on self.results / self.model.
        Used when saving per-seed snapshots during multi-seed runs."""
        snapshot_params = copy.deepcopy(rsf_params)
        if test_size is not None:
            snapshot_params["test_size"] = test_size
        snapshot_params["drop_first"] = bool(self.drop_first_var.get()) if hasattr(self, "drop_first_var") else True
        snapshot_params["stratify_event"] = bool(self.stratify_event_var.get()) if hasattr(self, "stratify_event_var") else True
        snapshot_params["missing_strategy"] = self.missing_strategy_var.get() if hasattr(self, "missing_strategy_var") else "Imputar mediana/moda"
        snapshot_params["tau_mode"] = self.tau_mode_var.get() if hasattr(self, "tau_mode_var") else "Percentil 90"
        snapshot_params["tau_manual"] = self.tau_manual_var.get() if hasattr(self, "tau_manual_var") else ""
        snapshot_params["optimization_metric"] = self.optimization_metric_var.get() if hasattr(self, "optimization_metric_var") else "Harrell C-index"
        if "seed_used" not in snapshot_params:
            snapshot_params["seed_used"] = snapshot_params.get("random_state", 42)
        if "seeds_used" not in snapshot_params:
            snapshot_params["seeds_used"] = [snapshot_params.get("random_state", 42)]
        _cal = calibration_df if isinstance(calibration_df, pd.DataFrame) else self.latest_calibration_df
        _brier = brier_df if isinstance(brier_df, pd.DataFrame) else self.latest_brier_df
        _etime = eval_time if eval_time is not None else self.latest_eval_time
        _fdf = fit_df if isinstance(fit_df, pd.DataFrame) else self.latest_fit_dataframe
        snapshot = {
            "label": f"RSF #{len(self.saved_models) + 1}",
            "params": snapshot_params,
            "metrics": copy.deepcopy(metrics),
            "report_text": report_text,
            "model": model,
            "feature_importance_df": imp_df.copy(deep=True) if isinstance(imp_df, pd.DataFrame) else pd.DataFrame(),
            "latest_prediction_df": pred_df.copy(deep=True) if isinstance(pred_df, pd.DataFrame) else None,
            "latest_survival_profiles": copy.deepcopy(profiles),
            "latest_calibration_df": _cal.copy(deep=True) if isinstance(_cal, pd.DataFrame) else pd.DataFrame(),
            "latest_brier_df": _brier.copy(deep=True) if isinstance(_brier, pd.DataFrame) else pd.DataFrame(),
            "latest_eval_time": _etime,
            "latest_fit_dataframe": _fdf.copy(deep=True) if isinstance(_fdf, pd.DataFrame) else None,
            "latest_duration_col": self.latest_duration_col,
            "latest_event_col": self.latest_event_col,
            "latest_covariates": list(self.latest_covariates),
            "latest_encoded_columns": list(self.latest_encoded_columns),
            "latest_drop_first": bool(getattr(self, "latest_drop_first", True)),
            "tuning_summary": self.latest_tuning_summary,
            "variable_configs": copy.deepcopy(getattr(self, "variable_configs", {})),
        }
        self.saved_models.append(snapshot)
        self.active_saved_model_index = len(self.saved_models) - 1
        self._refresh_saved_models_tree()

    def _store_current_model_snapshot(self, rsf_params, report_text, test_size=None):
        metrics = self.results or {}
        snapshot_params = copy.deepcopy(rsf_params)
        if test_size is not None:
            snapshot_params["test_size"] = test_size
        snapshot_params["drop_first"] = bool(self.drop_first_var.get()) if hasattr(self, "drop_first_var") else True
        snapshot_params["stratify_event"] = bool(self.stratify_event_var.get()) if hasattr(self, "stratify_event_var") else True
        snapshot_params["missing_strategy"] = self.missing_strategy_var.get() if hasattr(self, "missing_strategy_var") else "Imputar mediana/moda"
        snapshot_params["tau_mode"] = self.tau_mode_var.get() if hasattr(self, "tau_mode_var") else "Percentil 90"
        snapshot_params["tau_manual"] = self.tau_manual_var.get() if hasattr(self, "tau_manual_var") else ""
        snapshot_params["optimization_metric"] = self.optimization_metric_var.get() if hasattr(self, "optimization_metric_var") else "Harrell C-index"
        # Save the seeds used so reconstruction can reproduce the same multi-seed average
        try:
            snapshot_params["seeds_used"] = self._parse_random_seeds()
        except Exception:
            snapshot_params["seeds_used"] = [int(rsf_params.get("random_state", 42))]
        # Store the single seed used (first of the list) for display in the table
        snapshot_params.setdefault("seed_used", snapshot_params["seeds_used"][0] if snapshot_params["seeds_used"] else rsf_params.get("random_state", 42))
        snapshot = {
            "label": f"RSF #{len(self.saved_models) + 1}",
            "params": snapshot_params,
            "metrics": copy.deepcopy(metrics),
            "report_text": report_text,
            "model": self.model,
            "feature_importance_df": self.feature_importance_df.copy(deep=True) if isinstance(self.feature_importance_df, pd.DataFrame) else pd.DataFrame(),
            "latest_prediction_df": self.latest_prediction_df.copy(deep=True) if isinstance(self.latest_prediction_df, pd.DataFrame) else None,
            "latest_survival_profiles": copy.deepcopy(self.latest_survival_profiles),
            "latest_calibration_df": self.latest_calibration_df.copy(deep=True) if isinstance(self.latest_calibration_df, pd.DataFrame) else pd.DataFrame(),
            "latest_brier_df": self.latest_brier_df.copy(deep=True) if isinstance(self.latest_brier_df, pd.DataFrame) else pd.DataFrame(),
            "latest_eval_time": self.latest_eval_time,
            "latest_fit_dataframe": self.latest_fit_dataframe.copy(deep=True) if isinstance(self.latest_fit_dataframe, pd.DataFrame) else None,
            "latest_duration_col": self.latest_duration_col,
            "latest_event_col": self.latest_event_col,
            "latest_covariates": list(self.latest_covariates),
            "latest_encoded_columns": list(self.latest_encoded_columns),
            "latest_drop_first": bool(getattr(self, "latest_drop_first", bool(self.drop_first_var.get()))),
            "tuning_summary": self.latest_tuning_summary,
            "variable_configs": copy.deepcopy(getattr(self, "variable_configs", {})),
        }
        self.saved_models.append(snapshot)
        self.active_saved_model_index = len(self.saved_models) - 1
        self._refresh_saved_models_tree()

    def _get_layout_host(self):
        host_app = self.__dict__.get("master")
        if self.__dict__.get("tk") is not None:
            try:
                top_level = self.winfo_toplevel()
                if top_level is not None:
                    host_app = top_level
            except Exception:
                pass
        return host_app

    def _restore_saved_layout(self, layout_key, col_config):
        host_app = self._get_layout_host()
        apply_layout = getattr(host_app, "apply_saved_table_layout", None) if host_app is not None else None
        if callable(apply_layout):
            apply_layout(layout_key, col_config)

    def _register_saved_layout(self, layout_key, col_config, treeview):
        host_app = self._get_layout_host()
        register_layout = getattr(host_app, "register_table_layout_source", None) if host_app is not None else None
        if callable(register_layout):
            register_layout(layout_key, col_config, treeview)

    def _persist_saved_layout(self, layout_key, col_config, treeview):
        host_app = self._get_layout_host()
        persist_layout = getattr(host_app, "persist_table_layout", None) if host_app is not None else None
        if callable(persist_layout):
            persist_layout(layout_key, col_config, treeview)

    def _persist_saved_models_tree_layout(self):
        self._persist_saved_layout("rsf_saved_models", self._tree_column_config, getattr(self, "saved_models_tree", None))

    def save_table_layouts(self):
        self._persist_saved_models_tree_layout()

    def _show_tree_column_menu(self, event):
        """Right-click context menu to toggle column visibility."""
        menu = tk.Menu(self.saved_models_tree, tearoff=0)
        menu.add_command(label="── Columnas visibles ──", state="disabled")
        menu.add_separator()
        for col_id, cfg in self._tree_column_config.items():
            label = f"✓ {cfg['heading']}" if cfg["visible"] else f"   {cfg['heading']}"
            menu.add_command(label=label, command=lambda c=col_id: self._toggle_tree_column(c))
        menu.add_separator()
        menu.add_command(label="Mostrar todas", command=self._show_all_tree_columns)
        menu.add_command(label="Ocultar todas", command=self._hide_all_tree_columns)
        menu.tk_popup(event.x_root, event.y_root)

    def _toggle_tree_column(self, col_id):
        cfg = self._tree_column_config[col_id]
        try:
            current_width = int(self.saved_models_tree.column(col_id, option="width"))
            if current_width > 0:
                cfg["width"] = current_width
        except Exception:
            pass
        cfg["visible"] = not cfg["visible"]
        self._apply_tree_column_layout()
        self._persist_saved_models_tree_layout()
        self._persist_saved_layout("rsf_saved_models", self._tree_column_config, self.saved_models_tree)

    def _show_all_tree_columns(self):
        for col_id, cfg in self._tree_column_config.items():
            cfg["visible"] = True
        self._apply_tree_column_layout()
        self._persist_saved_models_tree_layout()
        self._persist_saved_layout("rsf_saved_models", self._tree_column_config, self.saved_models_tree)

    def _hide_all_tree_columns(self):
        for col_id, cfg in self._tree_column_config.items():
            cfg["visible"] = col_id == "id"  # keep at least 'id' visible
        self._apply_tree_column_layout()
        self._persist_saved_models_tree_layout()
        self._persist_saved_layout("rsf_saved_models", self._tree_column_config, self.saved_models_tree)

    def _apply_tree_column_layout(self):
        visible_cols = [col_id for col_id, cfg in self._tree_column_config.items() if cfg.get("visible", True)]
        try:
            self.saved_models_tree.configure(displaycolumns=visible_cols if visible_cols else ())
        except Exception:
            pass
        for col_id, cfg in self._tree_column_config.items():
            if cfg.get("visible", True):
                self.saved_models_tree.column(col_id, width=cfg["width"], minwidth=24, stretch=False)
            else:
                self.saved_models_tree.column(col_id, width=0, minwidth=0, stretch=False)

    def _get_tree_sort_value(self, column_name, raw_value):
        numeric_columns = {"id", "trees", "leaf", "split", "test_prop", "cv", "c_train", "c_test", "oob",
                           "c_uno", "c_antolini", "tau", "ibs", "ibs_km", "bss",
                           "c_q25", "c_q50", "c_q75", "brier_q25", "brier_q50", "brier_q75",
                           "auroc_q25", "auroc_q50", "auroc_q75"}
        if column_name in numeric_columns:
            try:
                match = re.search(r"[-+]?\d*\.?\d+", str(raw_value))
                return float(match.group(0)) if match else float("-inf")
            except Exception:
                return float("-inf")
        return str(raw_value).strip().lower()

    def _sort_saved_models_tree(self, column_name):
        if not hasattr(self, "saved_models_tree"):
            return

        # Columns where higher = better (default sort descending first click)
        metric_columns = {"cv", "c_train", "c_test", "oob",
                          "c_uno", "c_antolini", "bss",
                          "c_q25", "c_q50", "c_q75",
                          "auroc_q25", "auroc_q50", "auroc_q75"}
        if not hasattr(self, "_saved_models_sort_state"):
            self._saved_models_sort_state = {}

        previous_reverse = self._saved_models_sort_state.get(column_name)
        if previous_reverse is None:
            reverse = column_name in metric_columns
        else:
            reverse = not bool(previous_reverse)

        items = []
        for item_id in self.saved_models_tree.get_children(""):
            raw_value = self.saved_models_tree.set(item_id, column_name)
            items.append((self._get_tree_sort_value(column_name, raw_value), item_id))

        items.sort(key=lambda pair: pair[0], reverse=reverse)
        for position, (_, item_id) in enumerate(items):
            self.saved_models_tree.move(item_id, "", position)

        self._saved_models_sort_state[column_name] = reverse

    def _on_saved_models_tree_double_click(self, event=None):
        if not hasattr(self, "saved_models_tree"):
            return None

        event = event or tk.Event()
        region = self.saved_models_tree.identify_region(getattr(event, "x", 0), getattr(event, "y", 0))
        if region == "heading":
            col_token = self.saved_models_tree.identify_column(getattr(event, "x", 0))
            if not col_token or not str(col_token).startswith("#"):
                return "break"
            try:
                col_idx = int(str(col_token)[1:]) - 1
            except Exception:
                return "break"

            columns = list(self.saved_models_tree["columns"])
            if not (0 <= col_idx < len(columns)):
                return "break"

            col_name = columns[col_idx]
            metric_columns = {"cv", "c_train", "c_test", "oob"}
            reverse = getattr(self, "_saved_models_sort_state", {}).get(col_name)
            if reverse is None:
                reverse = col_name in metric_columns
            self._show_best_sorted_models_popup(col_name, bool(reverse))
            return "break"

        return self._load_selected_saved_model(event)

    def _parse_table_sort_value(self, raw_value):
        text = str(raw_value).strip()
        if text in {"", "-", "N/A", "nan", "None"}:
            return ("str", "")
        match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
        if match:
            try:
                return ("num", float(match.group(0)))
            except Exception:
                pass
        return ("str", text.lower())

    def _sort_keys_equal(self, left_key, right_key, tol=1e-10):
        if left_key[0] != right_key[0]:
            return False
        if left_key[0] == "num":
            return abs(left_key[1] - right_key[1]) <= tol
        return left_key[1] == right_key[1]

    def _show_best_sorted_models_popup(self, column_name, reverse):
        if not hasattr(self, "saved_models_tree"):
            return
        children = self.saved_models_tree.get_children("")
        if not children:
            return
        columns = list(self.saved_models_tree["columns"])
        if column_name not in columns:
            return

        from collections import Counter as _Counter

        col_idx = columns.index(column_name)
        metric_title = str(self._tree_column_config.get(column_name, {}).get("heading", column_name) or column_name)

        # Collect ALL saved model rows (not just tied ones)
        all_rows = [self.saved_models_tree.item(iid, "values") for iid in children]
        if not all_rows:
            return

        # Parse the metric column for every row
        metric_keys = [
            self._parse_table_sort_value(row[col_idx] if col_idx < len(row) else "-")
            for row in all_rows
        ]
        numeric_metric_indices = [i for i, k in enumerate(metric_keys) if k[0] == "num"]

        if numeric_metric_indices:
            sorted_by_metric = sorted(numeric_metric_indices, key=lambda i: metric_keys[i][1], reverse=True)
            mayor_idx = sorted_by_metric[0]   # highest metric value
            menor_idx = sorted_by_metric[-1]  # lowest metric value
            metric_nums = [float(metric_keys[i][1]) for i in numeric_metric_indices]
            mayor_metric_str = f"{float(metric_keys[mayor_idx][1]):.4f}"
            menor_metric_str = f"{metric_keys[menor_idx][1]:.4f}"
            avg_metric_str   = f"{np.mean(metric_nums):.4f}"
        else:
            mayor_idx = 0
            menor_idx = len(all_rows) - 1
            m_val = str(all_rows[0][col_idx] if col_idx < len(all_rows[0]) else "-")
            mayor_metric_str = menor_metric_str = avg_metric_str = m_val

        # Parameter column definitions (key in treeview → display label)
        param_defs = [
            ("scope",     "Scope"),
            ("mode",      "Modo"),
            ("trees",     "Árboles"),
            ("features",  "max_features"),
            ("leaf",      "min_leaf"),
            ("split",     "min_split"),
            ("max_depth", "max_depth"),
            ("max_leaf_nodes", "max_leaf_nodes"),
            ("max_samples", "max_samples"),
            ("test_prop", "Test %"),
        ]
        param_columns = [
            (key, label, columns.index(key))
            for key, label in param_defs
            if key in columns
        ]

        # ── Build popup ──────────────────────────────────────────────
        popup = tk.Toplevel(self)
        popup.title("Mejor valor en modelos RSF")
        popup.resizable(True, True)
        popup.transient(self.winfo_toplevel())
        popup.grab_set()

        sort_order_text = "descendente" if reverse else "ascendente"
        hdr = ttk.Frame(popup, padding=(12, 8, 12, 4))
        hdr.pack(fill=tk.X)
        ttk.Label(
            hdr,
            text=(
                f"Columna: {metric_title}  |  Orden aplicado: {sort_order_text}"
                f"  |  Mejores modelos: {len(all_rows)}"
            ),
            foreground="navy",
            font=("TkDefaultFont", 9, "bold"),
        ).pack(anchor="w")

        # Diversity warning when all models share the same max_features
        if "features" in columns:
            feat_idx = columns.index("features")
            unique_feats = {str(r[feat_idx] if feat_idx < len(r) else "-") for r in all_rows}
            if len(unique_feats) == 1:
                mf_val = next(iter(unique_feats))
                wf = ttk.Frame(popup, relief="solid", padding=6)
                wf.pack(fill=tk.X, padx=12, pady=(0, 6))
                ttk.Label(
                    wf,
                    text=(
                        f"⚠  Todos los modelos usan max_features={mf_val}. Con un único "
                        f"valor de max_features los árboles producen los mismos splits, "
                        f"dando métricas idénticas. Varía max_features (sqrt, log2, 0.5…) "
                        f"para obtener diversidad real."
                    ),
                    foreground="#8B4513",
                    wraplength=660,
                    justify="left",
                ).pack(anchor="w")

        # ── Sub-tables grid ───────────────────────────────────────────
        grid_frame = ttk.Frame(popup, padding=(12, 0, 12, 4))
        grid_frame.pack(fill=tk.BOTH, expand=True)

        for grid_pos, (col_key, col_label, p_idx) in enumerate(param_columns):
            row_g = grid_pos // 2
            col_g = grid_pos % 2

            mayor_param = str(all_rows[mayor_idx][p_idx] if p_idx < len(all_rows[mayor_idx]) else "-")
            menor_param = str(all_rows[menor_idx][p_idx] if p_idx < len(all_rows[menor_idx]) else "-")

            param_nums = []
            param_strs = []
            for row in all_rows:
                rv = row[p_idx] if p_idx < len(row) else "-"
                pk = self._parse_table_sort_value(rv)
                if pk[0] == "num":
                    param_nums.append(float(pk[1]))
                else:
                    param_strs.append(str(rv))

            if param_nums:
                avg_param_str = f"{np.mean(param_nums):.1f}"
            elif param_strs:
                mc = _Counter(param_strs).most_common(1)
                avg_param_str = f"(más común) {mc[0][0]}" if mc else "-"
            else:
                avg_param_str = "-"

            sub = ttk.LabelFrame(grid_frame, text=col_label, padding=6)
            sub.grid(row=row_g, column=col_g, padx=6, pady=4, sticky="nsew")

            mini = ttk.Treeview(sub, columns=("label", "param", "metric"),
                                show="headings", height=3, selectmode="none")
            mini.heading("label",  text="")
            mini.heading("param",  text=col_label)
            mini.heading("metric", text=metric_title)
            mini.column("label",  width=80,  anchor="e",      stretch=False)
            mini.column("param",  width=130, anchor="center", stretch=True)
            mini.column("metric", width=140, anchor="center", stretch=False)
            mini.pack(fill=tk.X, expand=True)

            mini.insert("", "end", values=("Mayor",    mayor_param,   mayor_metric_str))
            mini.insert("", "end", values=("Menor",    menor_param,   menor_metric_str))
            mini.insert("", "end", values=("Promedio", avg_param_str, avg_metric_str))

        for col_g in range(2):
            grid_frame.columnconfigure(col_g, weight=1)

        ttk.Button(popup, text="Cerrar", command=popup.destroy).pack(pady=(4, 10))

        popup.update_idletasks()
        w = min(860, max(580, popup.winfo_reqwidth() + 30))
        h = min(740, max(360, popup.winfo_reqheight() + 20))
        popup.geometry(f"{w}x{h}")

    def _refresh_saved_models_tree(self, rerank=True):
        if not hasattr(self, "saved_models_tree"):
            return

        # ── Always re-rank saved_models by the 5-phase clinical ranking ──
        if rerank and len(self.saved_models) > 1:
            # Track the currently active snapshot by identity
            active_snapshot = None
            if (self.active_saved_model_index is not None
                    and 0 <= self.active_saved_model_index < len(self.saved_models)):
                active_snapshot = self.saved_models[self.active_saved_model_index]

            self._rank_tuning_results(self.saved_models)

            # Restore active index after re-ordering
            if active_snapshot is not None:
                _found_idx = self._find_snapshot_identity(active_snapshot, self.saved_models)
                if _found_idx is not None:
                    self.active_saved_model_index = _found_idx
                else:
                    self.active_saved_model_index = 0

        for item in self.saved_models_tree.get_children():
            self.saved_models_tree.delete(item)

        active_text = "ninguno"
        for idx, snapshot in enumerate(self.saved_models):
            params = snapshot.get("params", {})
            metrics = snapshot.get("metrics", {})
            raw_test_size = params.get("test_size", None)
            test_display = "-"
            if raw_test_size is not None:
                try:
                    test_display = f"{float(raw_test_size):.2f}"
                except Exception:
                    test_display = str(raw_test_size)
            c_uno_v = metrics.get("c_index_uno")
            c_ant_v = metrics.get("c_index_antolini")
            c_uno_ci = metrics.get("c_index_uno_ci")
            c_ant_ci = metrics.get("c_index_antolini_ci")
            tau_v = metrics.get("tau")
            ibs_v = metrics.get("ibs")
            _fmt4 = lambda v: f"{v:.4f}" if v is not None and pd.notna(v) else "-"
            ibs_km_v = metrics.get("ibs_km")
            bss_v = metrics.get("bss")
            ibs_ci = metrics.get("ibs_ci")
            bss_ci = metrics.get("bss_ci")
            model_covariates = list(snapshot.get("latest_covariates", []))
            model_n_vars = len([c for c in model_covariates if str(c).strip()])
            if model_n_vars <= 0:
                mode_display = "-"
            elif model_n_vars == 1:
                mode_display = "1 var"
            else:
                mode_display = f"{model_n_vars} vars"
            # Seed display
            _seeds_used_list = params.get("seeds_used", [])
            if isinstance(_seeds_used_list, (list, tuple)) and len(_seeds_used_list) > 1:
                seed_display = "MS"
            else:
                seed_display = params.get("seed_used", None)
                if seed_display is None:
                    seed_display = _seeds_used_list[0] if _seeds_used_list else params.get("random_state", "-")
                seed_display = str(seed_display) if seed_display is not None else "-"
            values = (
                idx + 1,
                "★" if self._is_model_clinically_apt(snapshot) else "",
                seed_display,
                params.get("tuning_scope", snapshot.get("scope", "-")),
                mode_display,
                params.get("n_estimators", "-"),
                self._format_max_features_display(params, snapshot),
                params.get("min_samples_leaf", "-"),
                params.get("min_samples_split", "-"),
                params.get("max_depth", "-"),
                params.get("max_leaf_nodes", "-"),
                params.get("max_samples", "-"),
                test_display,
                self._format_c_index_display(metrics.get("c_index_cv_mean"), metrics.get("c_index_cv_ci"), decimals=3),
                self._format_c_index_display(metrics.get("c_index_train"), metrics.get("c_index_train_ci"), decimals=3),
                self._format_c_index_display(metrics.get("c_index_test"), metrics.get("c_index_test_ci"), decimals=3),
                self._format_metric(metrics.get("oob_score")),
                self._format_c_index_display(c_uno_v, c_uno_ci, decimals=3),
                self._format_c_index_display(c_ant_v, c_ant_ci, decimals=3),
                f"{tau_v:.1f}" if tau_v is not None and pd.notna(tau_v) else "-",
                self._format_c_index_display(ibs_v, ibs_ci, decimals=4),
                _fmt4(ibs_km_v),
                self._format_c_index_display(bss_v, bss_ci, decimals=4),
                _fmt4(metrics.get("c_harrell_q25")), _fmt4(metrics.get("c_harrell_q50")), _fmt4(metrics.get("c_harrell_q75")),
                _fmt4(metrics.get("brier_q25")), _fmt4(metrics.get("brier_q50")), _fmt4(metrics.get("brier_q75")),
                _fmt4(metrics.get("auroc_q25")), _fmt4(metrics.get("auroc_q50")), _fmt4(metrics.get("auroc_q75")),
            )
            self.saved_models_tree.insert("", "end", iid=str(idx), values=values)

        if self.saved_models and self.active_saved_model_index is not None and 0 <= self.active_saved_model_index < len(self.saved_models):
            active_text = self.saved_models[self.active_saved_model_index].get("label", f"RSF #{self.active_saved_model_index + 1}")
            self.saved_models_tree.selection_set(str(self.active_saved_model_index))

        if hasattr(self, "saved_models_status_var"):
            self.saved_models_status_var.set(f"Modelos en memoria: {len(self.saved_models)} | Activo: {active_text}")

    def _load_selected_saved_model(self, _event=None):
        if not self.saved_models:
            return
        # Guard against re-entrancy (user double-clicks while loading)
        if getattr(self, "_loading_model_in_progress", False):
            return
        selected_items = self.saved_models_tree.selection() if hasattr(self, "saved_models_tree") else ()
        if not selected_items:
            return
        selected_index = int(selected_items[0])
        snapshot = self.saved_models[selected_index]
        self.active_saved_model_index = selected_index
        self._apply_snapshot_params(snapshot.get("params", {}))

        restored = self._restore_saved_snapshot_state(snapshot)
        if not restored:
            # Para snapshots incompletos (del tuning), restaurar el UI primero
            if isinstance(snapshot.get("latest_fit_dataframe"), pd.DataFrame):
                self.latest_fit_dataframe = snapshot.get("latest_fit_dataframe").copy(deep=True)
            if snapshot.get("latest_duration_col"):
                self.latest_duration_col = snapshot.get("latest_duration_col")
            if snapshot.get("latest_event_col"):
                self.latest_event_col = snapshot.get("latest_event_col")
            if isinstance(snapshot.get("latest_covariates"), list) and snapshot.get("latest_covariates"):
                self.latest_covariates = list(snapshot.get("latest_covariates"))
            if isinstance(snapshot.get("latest_encoded_columns"), list) and snapshot.get("latest_encoded_columns"):
                self.latest_encoded_columns = list(snapshot.get("latest_encoded_columns"))
            self._restore_saved_snapshot_ui_state()
            snapshot_report = snapshot.get("report_text", "")
            tuning_summary = snapshot_report if "=== Tuning automático RSF ===" in str(snapshot_report) else None
            # Pasar params y covariables explícitamente para no depender del estado del UI
            snapshot_covariates = snapshot.get("latest_covariates")
            snapshot_params = snapshot.get("params")

            # ── Optimización: al cargar solo re-entrenar con la semilla representativa ──
            # El snapshot guarda seeds_used=[42,123,456] y seed_used=123 (la más cercana al promedio).
            # No tiene sentido re-entrenar con TODAS las semillas solo para mostrar el modelo.
            # Solo reentrenamos con la representativa para que la carga sea ~N× más rápida.
            if isinstance(snapshot_params, dict):
                snapshot_params = copy.deepcopy(snapshot_params)
                original_seeds = snapshot_params.get("seeds_used", [])
                representative = snapshot_params.get("seed_used")
                if isinstance(original_seeds, (list, tuple)) and len(original_seeds) > 1 and representative is not None:
                    try:
                        snapshot_params["seeds_used"] = [int(representative)]
                        snapshot_params["seed_used"] = int(representative)
                        snapshot_params["random_state"] = int(representative)
                    except (TypeError, ValueError):
                        pass

            # Abrir un dialog de progreso para que _fit_model_with_ui_pump pueda
            # bombear eventos de la ventana principal y evitar que Windows marque
            # la app como "No responde" durante el recálculo del modelo.
            _prev_dialog = getattr(self, "_tuning_progress_dialog", None)
            _load_dialog = None
            try:
                _parent = self.winfo_toplevel()
                _load_dialog = tk.Toplevel(_parent)
                _load_dialog.title("RSF — Cargando modelo")
                _load_dialog.geometry("380x90")
                _load_dialog.resizable(False, False)
                _load_dialog.transient(_parent)
                _load_dialog.protocol("WM_DELETE_WINDOW", lambda: None)
                tk.Label(
                    _load_dialog,
                    text="Recalculando modelo RSF seleccionado…\nEspere, esto puede tardar unos momentos.",
                    padx=18, pady=18, justify="left",
                ).pack()
                try:
                    _load_dialog.grab_set()
                except Exception:
                    pass
                _load_dialog.update()
                self._tuning_progress_dialog = _load_dialog
            except Exception:
                _load_dialog = None

            self._loading_model_in_progress = True
            try:
                self.run_model(
                    tuning_summary=tuning_summary,
                    store_snapshot=False,
                    params_override=snapshot_params if isinstance(snapshot_params, dict) else None,
                    covariates_override=snapshot_covariates if isinstance(snapshot_covariates, list) else None,
                )
            except Exception as load_exc:
                messagebox.showerror("Error al cargar modelo", f"No se pudo recalcular el modelo:\n{load_exc}")
                return
            finally:
                self._loading_model_in_progress = False
                # Cerrar el dialog de carga y restaurar el estado previo
                if _load_dialog is not None:
                    try:
                        _load_dialog.destroy()
                    except Exception:
                        pass
                if getattr(self, "_tuning_progress_dialog", None) is _load_dialog:
                    self._tuning_progress_dialog = _prev_dialog

            # Sincronizar el snapshot seleccionado con el estado recién recalculado.
            snapshot["metrics"] = copy.deepcopy(self.results or {})
            snapshot["report_text"] = str(getattr(self, "latest_report_text", snapshot.get("report_text", "")))
            snapshot["model"] = self.model
            snapshot["feature_importance_df"] = self.feature_importance_df.copy(deep=True) if isinstance(self.feature_importance_df, pd.DataFrame) else pd.DataFrame()
            snapshot["latest_prediction_df"] = self.latest_prediction_df.copy(deep=True) if isinstance(self.latest_prediction_df, pd.DataFrame) else None
            snapshot["latest_survival_profiles"] = copy.deepcopy(self.latest_survival_profiles)
            snapshot["latest_calibration_df"] = self.latest_calibration_df.copy(deep=True) if isinstance(self.latest_calibration_df, pd.DataFrame) else pd.DataFrame()
            snapshot["latest_brier_df"] = self.latest_brier_df.copy(deep=True) if isinstance(self.latest_brier_df, pd.DataFrame) else pd.DataFrame()
            snapshot["latest_eval_time"] = self.latest_eval_time
            snapshot["latest_fit_dataframe"] = self.latest_fit_dataframe.copy(deep=True) if isinstance(self.latest_fit_dataframe, pd.DataFrame) else None
            snapshot["latest_duration_col"] = self.latest_duration_col
            snapshot["latest_event_col"] = self.latest_event_col
            snapshot["latest_covariates"] = list(self.latest_covariates)
            snapshot["latest_encoded_columns"] = list(self.latest_encoded_columns)
            snapshot["latest_drop_first"] = bool(getattr(self, "latest_drop_first", bool(self.drop_first_var.get())))
            snapshot["tuning_summary"] = getattr(self, "latest_tuning_summary", "")
            snapshot["variable_configs"] = copy.deepcopy(getattr(self, "variable_configs", {}))

        self._refresh_saved_models_tree()
        self.notebook.select(self.results_tab)

    def _delete_selected_saved_model(self):
        if not self.saved_models:
            return
        selected_items = self.saved_models_tree.selection() if hasattr(self, "saved_models_tree") else ()
        if not selected_items:
            return

        selected_indices = sorted({int(item) for item in selected_items})
        previous_active = self.active_saved_model_index
        deleted_was_active = previous_active in selected_indices if previous_active is not None else False

        for idx in sorted(selected_indices, reverse=True):
            if 0 <= idx < len(self.saved_models):
                del self.saved_models[idx]

        if not self.saved_models:
            self.active_saved_model_index = None
            self._refresh_saved_models_tree()
            self._reset_results_view()
            return

        anchor_index = selected_indices[0] if selected_indices else 0
        if previous_active is None:
            self.active_saved_model_index = min(anchor_index, len(self.saved_models) - 1)
        elif deleted_was_active:
            self.active_saved_model_index = min(anchor_index, len(self.saved_models) - 1)
        else:
            shift_left = sum(1 for idx in selected_indices if idx < previous_active)
            shifted_active = previous_active - shift_left
            self.active_saved_model_index = max(0, min(shifted_active, len(self.saved_models) - 1))

        self._refresh_saved_models_tree()
        if self.active_saved_model_index is not None and hasattr(self, "saved_models_tree"):
            self.saved_models_tree.selection_set(str(self.active_saved_model_index))
        if deleted_was_active and self.active_saved_model_index is not None and hasattr(self, "saved_models_tree"):
            self._load_selected_saved_model()

    def _is_model_clinically_apt(self, snapshot):
        """Return True if the model passes BSS and overfitting-gap gates."""
        metrics = snapshot.get("metrics", {})
        bss = metrics.get("bss")
        if bss is not None and np.isfinite(bss) and bss <= self._RANK_BSS_MIN:
            return False
        cv_key, test_key, _cv_lbl, _test_lbl = self._resolve_clinical_stability_metric_keys()
        cv = metrics.get(cv_key)
        ct = metrics.get(test_key)
        if (cv is not None and np.isfinite(cv)
                and ct is not None and np.isfinite(ct)
                and abs(ct - cv) > self._RANK_OVERFIT_GAP_MAX):
            return False
        return True

    def _delete_non_selected_models(self):
        """Delete all models that are NOT currently selected in the tree."""
        if not self.saved_models:
            return
        selected_items = self.saved_models_tree.selection() if hasattr(self, "saved_models_tree") else ()
        if not selected_items:
            messagebox.showwarning("Sin selección", "Selecciona al menos un modelo para conservar.")
            return
        keep_indices = sorted({int(item) for item in selected_items})
        keep_models = [self.saved_models[i] for i in keep_indices if 0 <= i < len(self.saved_models)]
        n_delete = len(self.saved_models) - len(keep_models)
        if n_delete == 0:
            return
        if not messagebox.askyesno("Confirmar", f"Se eliminarán {n_delete} modelo(s) no seleccionado(s).\n¿Continuar?"):
            return

        active_snapshot = None
        if (self.active_saved_model_index is not None
                and 0 <= self.active_saved_model_index < len(self.saved_models)):
            active_snapshot = self.saved_models[self.active_saved_model_index]

        self.saved_models[:] = keep_models

        _found_idx = self._find_snapshot_identity(active_snapshot, self.saved_models)
        if _found_idx is not None:
            self.active_saved_model_index = _found_idx
        elif self.saved_models:
            self.active_saved_model_index = 0
        else:
            self.active_saved_model_index = None

        self._refresh_saved_models_tree()

    def _delete_non_apt_models(self):
        """Delete all models that do NOT pass the clinical aptitude gates."""
        if not self.saved_models:
            return
        apt_models = [s for s in self.saved_models if self._is_model_clinically_apt(s)]
        n_delete = len(self.saved_models) - len(apt_models)
        if n_delete == 0:
            messagebox.showinfo("Sin cambios", "Todos los modelos son clínicamente aptos.")
            return
        if not messagebox.askyesno("Confirmar", f"Se eliminarán {n_delete} modelo(s) no apto(s).\n¿Continuar?"):
            return

        active_snapshot = None
        if (self.active_saved_model_index is not None
                and 0 <= self.active_saved_model_index < len(self.saved_models)):
            active_snapshot = self.saved_models[self.active_saved_model_index]

        self.saved_models[:] = apt_models

        _found_idx = self._find_snapshot_identity(active_snapshot, self.saved_models)
        if _found_idx is not None:
            self.active_saved_model_index = _found_idx
        elif self.saved_models:
            self.active_saved_model_index = 0
        else:
            self.active_saved_model_index = None
            self._reset_results_view()

        self._refresh_saved_models_tree()

    # ------------------------------------------------------------------
    # Model Explorer (2D scatter + colormap)
    # ------------------------------------------------------------------
    _EXPLORER_METRICS = [
        ("c_index_uno",      "C-Uno (IPCW)"),
        ("c_index_cv_mean",  "CV (métrica seleccionada)"),
        ("c_index_test",     "C-test"),
        ("c_index_train",    "C-train"),
        ("oob_score",        "OOB"),
        ("bss",              "BSS"),
        ("ibs",              "IBS"),
        ("tau",              "τ (tau)"),
        ("c_index_antolini", "C-Antolini"),
    ]

    def _open_model_explorer(self):
        if not self.saved_models or len(self.saved_models) < 2:
            messagebox.showinfo("Explorar modelos", f"Se necesitan al menos 2 modelos en memoria. Actualmente hay {len(self.saved_models) if self.saved_models else 0}.")
            return

        try:
            from collections import Counter

            parent_window = self.winfo_toplevel() if hasattr(self, "winfo_toplevel") else None
            dlg = tk.Toplevel(parent_window if parent_window is not None else self)
            dlg.title("Explorador de modelos RSF")
            dlg.geometry("1120x740")
            if parent_window is not None:
                dlg.transient(parent_window)

            # ── Fila 1: ejes + filtros principales ──
            ctrl = ttk.Frame(dlg)
            ctrl.pack(fill=tk.X, padx=8, pady=(8, 2))

            metric_keys = [k for k, _ in self._EXPLORER_METRICS]

            defaults = ["c_index_uno", "c_index_cv_mean", "bss"]
            vars_combo = []
            for i, axis_name in enumerate(("Eje X:", "Eje Y:", "Color:")):
                ttk.Label(ctrl, text=axis_name).pack(side=tk.LEFT, padx=(8 if i else 0, 2))
                var = StringVar(value=defaults[i])
                cb = ttk.Combobox(ctrl, textvariable=var, values=metric_keys, state="readonly", width=16)
                cb.pack(side=tk.LEFT, padx=(0, 6))
                vars_combo.append(var)

            show_var = StringVar(value="Todos")
            ttk.Label(ctrl, text="Aptos:").pack(side=tk.LEFT, padx=(10, 2))
            filter_cb = ttk.Combobox(ctrl, textvariable=show_var, values=["Todos", "Solo aptos", "Solo no aptos"], state="readonly", width=13)
            filter_cb.pack(side=tk.LEFT, padx=(0, 6))

            # Filtro de completados (3 estados)
            show_completed_var = StringVar(value="Todos")
            ttk.Label(ctrl, text="Semillas:").pack(side=tk.LEFT, padx=(8, 2))
            ttk.Combobox(
                ctrl,
                textvariable=show_completed_var,
                values=["Todos", "Solo completados", "Solo incompletos"],
                state="readonly",
                width=17,
            ).pack(side=tk.LEFT, padx=(0, 6))
            show_completed_var.trace_add("write", lambda *_: _redraw())

            show_apt_only_var = BooleanVar(value=False)
            ttk.Checkbutton(
                ctrl,
                text="Solo aptos",
                variable=show_apt_only_var,
                command=lambda: _redraw(),
            ).pack(side=tk.LEFT, padx=(6, 4))

            ttk.Button(ctrl, text="Filtros avanzados...", command=lambda: _open_advanced_filters_dialog()).pack(side=tk.RIGHT, padx=4)
            ttk.Button(ctrl, text="Actualizar", command=lambda: _redraw()).pack(side=tk.RIGHT, padx=4)

            # ── Fila 2: opciones visuales ──
            ctrl2 = ttk.Frame(dlg)
            ctrl2.pack(fill=tk.X, padx=8, pady=(2, 2))

            point_size_var = StringVar(value="55")
            ttk.Label(ctrl2, text="Tamaño:").pack(side=tk.LEFT, padx=(0, 2))
            ttk.Combobox(
                ctrl2,
                textvariable=point_size_var,
                values=["25", "35", "45", "55", "70", "90", "120"],
                state="readonly",
                width=5,
            ).pack(side=tk.LEFT, padx=(0, 8))

            marker_mode_var = StringVar(value="Combinación")
            ttk.Label(ctrl2, text="Símbolo por:").pack(side=tk.LEFT, padx=(0, 2))
            ttk.Combobox(
                ctrl2,
                textvariable=marker_mode_var,
                values=["Combinación", "Scope", "Modo", "max_features", "min_leaf", "min_split",
                        "Completo/Semiinc./Incompleto"],
                state="readonly",
                width=20,
            ).pack(side=tk.LEFT, padx=(0, 4))

            _marker_shape_options = ["o", "s", "^", "D", "P", "X", "v", "<", ">", "h", "*", "p"]
            marker_complete_var = StringVar(value="o")
            marker_incomplete_var = StringVar(value="X")
            ttk.Label(ctrl2, text="✓:").pack(side=tk.LEFT, padx=(4, 1))
            _cb_marker_complete = ttk.Combobox(
                ctrl2, textvariable=marker_complete_var,
                values=_marker_shape_options, state="readonly", width=3,
            )
            _cb_marker_complete.pack(side=tk.LEFT, padx=(0, 2))
            _cb_marker_complete.bind("<<ComboboxSelected>>", lambda _: _redraw())
            ttk.Label(ctrl2, text="✗:").pack(side=tk.LEFT, padx=(2, 1))
            _cb_marker_incomplete = ttk.Combobox(
                ctrl2, textvariable=marker_incomplete_var,
                values=_marker_shape_options, state="readonly", width=3,
            )
            _cb_marker_incomplete.pack(side=tk.LEFT, padx=(0, 8))
            _cb_marker_incomplete.bind("<<ComboboxSelected>>", lambda _: _redraw())

            heatmap_var = BooleanVar(value=False)
            ttk.Checkbutton(
                ctrl2,
                text="Mapa de calor",
                variable=heatmap_var,
                command=lambda: _redraw(),
            ).pack(side=tk.LEFT, padx=(0, 4))

            heat_bins_var = StringVar(value="35")
            ttk.Label(ctrl2, text="Bins:").pack(side=tk.LEFT, padx=(0, 2))
            ttk.Combobox(
                ctrl2,
                textvariable=heat_bins_var,
                values=["20", "25", "30", "35", "45", "60"],
                state="readonly",
                width=5,
            ).pack(side=tk.LEFT, padx=(0, 10))

            show_ids_var = BooleanVar(value=False)
            ttk.Checkbutton(ctrl2, text="Etiquetas ID",
                            variable=show_ids_var,
                            command=lambda: _redraw()).pack(side=tk.LEFT, padx=(0, 8))
            show_ci_var = BooleanVar(value=True)
            ttk.Checkbutton(ctrl2, text="IC en cruz",
                            variable=show_ci_var,
                            command=lambda: _redraw()).pack(side=tk.LEFT, padx=(0, 8))

            # ── Fila 3: umbrales de calidad ──
            ctrl3 = ttk.Frame(dlg)
            ctrl3.pack(fill=tk.X, padx=8, pady=(2, 4))

            apply_gap_filters_var = BooleanVar(value=True)
            use_global_gap_var = BooleanVar(value=True)
            gap_global_var = StringVar(value=f"{self._PLOT_VIABILITY_GAP_DEFAULT:.2f}")
            gap_oob_cv_var = StringVar(value=f"{self._PLOT_GAP_OOB_CV_DEFAULT:.2f}")
            gap_oob_test_var = StringVar(value=f"{self._PLOT_GAP_OOB_TEST_DEFAULT:.2f}")
            gap_cv_test_var = StringVar(value=f"{self._PLOT_GAP_CV_TEST_DEFAULT:.2f}")

            threshold_values = ["0.00", "0.01", "0.02", "0.03", "0.05", "0.07", "0.10", "0.15", "0.20", "0.25", "0.30"]

            def _bind_threshold_refresh(cb):
                cb.bind("<<ComboboxSelected>>", lambda _event: _redraw())
                cb.bind("<Return>", lambda _event: _redraw())
                cb.bind("<FocusOut>", lambda _event: _redraw())

            ttk.Checkbutton(
                ctrl3,
                text="Aplicar umbrales",
                variable=apply_gap_filters_var,
                command=lambda: _redraw(),
            ).pack(side=tk.LEFT, padx=(0, 8))

            _threshold_specific_boxes = []

            def _toggle_explorer_gap_mode(refresh_plot=True):
                use_global = bool(use_global_gap_var.get())
                state = "disabled" if use_global else "normal"
                for cb in _threshold_specific_boxes:
                    try:
                        cb.configure(state=state)
                    except Exception:
                        pass
                if refresh_plot:
                    try:
                        _redraw()
                    except NameError:
                        pass

            ttk.Checkbutton(
                ctrl3,
                text="Δ único CV/Test/OOB",
                variable=use_global_gap_var,
                command=_toggle_explorer_gap_mode,
            ).pack(side=tk.LEFT, padx=(0, 4))

            ttk.Label(ctrl3, text="Δ<=").pack(side=tk.LEFT, padx=(0, 2))
            _cb_gap_global = ttk.Combobox(ctrl3, textvariable=gap_global_var, values=threshold_values, state="normal", width=5)
            _cb_gap_global.pack(side=tk.LEFT, padx=(0, 6))
            _bind_threshold_refresh(_cb_gap_global)

            ttk.Label(ctrl3, text="|OOB-CV|<=").pack(side=tk.LEFT, padx=(0, 2))
            _cb_gap_oob_cv = ttk.Combobox(ctrl3, textvariable=gap_oob_cv_var, values=threshold_values, state="normal", width=5)
            _cb_gap_oob_cv.pack(side=tk.LEFT, padx=(0, 6))
            _bind_threshold_refresh(_cb_gap_oob_cv)
            _threshold_specific_boxes.append(_cb_gap_oob_cv)

            ttk.Label(ctrl3, text="|OOB-C-test|<=").pack(side=tk.LEFT, padx=(0, 2))
            _cb_gap_oob_test = ttk.Combobox(ctrl3, textvariable=gap_oob_test_var, values=threshold_values, state="normal", width=5)
            _cb_gap_oob_test.pack(side=tk.LEFT, padx=(0, 6))
            _bind_threshold_refresh(_cb_gap_oob_test)
            _threshold_specific_boxes.append(_cb_gap_oob_test)

            ttk.Label(ctrl3, text="|CV-C-test|<=").pack(side=tk.LEFT, padx=(0, 2))
            _cb_gap_cv_test = ttk.Combobox(ctrl3, textvariable=gap_cv_test_var, values=threshold_values, state="normal", width=5)
            _cb_gap_cv_test.pack(side=tk.LEFT, padx=(0, 12))
            _bind_threshold_refresh(_cb_gap_cv_test)
            _threshold_specific_boxes.append(_cb_gap_cv_test)

            _toggle_explorer_gap_mode(refresh_plot=False)

            # ── Info label ──
            info_var = StringVar(value="")
            ttk.Label(dlg, textvariable=info_var, foreground="navy", wraplength=920, justify="left").pack(fill=tk.X, padx=10, pady=(0, 2))

            # ── Figure ──
            fig, ax_holder = plt.subplots(1, 1, figsize=(9, 5.5))
            plt.close(fig)  # detach from pyplot
            canvas = FigureCanvasTkAgg(fig, master=dlg)
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=6, pady=(0, 6))

            _scatter_data = {}
            _pulse_seen_uids = set(id(s) for s in self.saved_models)
            _marker_symbols = ["o", "s", "^", "D", "P", "v", "<", ">", "h", "8", "p"]

            _advanced_filter_fields = [
                ("cov_sig", "Combinación covariables"),
                ("scope", "Scope"),
                ("mode", "Modo"),
                ("trees", "Árboles"),
                ("max_features", "max_features"),
                ("min_leaf", "min_leaf"),
                ("min_split", "min_split"),
            ]
            _advanced_filters = {k: set() for k, _ in _advanced_filter_fields}

            def _get_metric_label(key):
                return self._get_metric_display_label(key)

            def _sort_filter_values(values):
                def _key_fn(raw):
                    txt = str(raw)
                    try:
                        return (0, float(txt))
                    except Exception:
                        return (1, txt.lower())

                return sorted(values, key=_key_fn)

            def _extract_cov_signature(snapshot):
                covs = [str(c).strip() for c in snapshot.get("latest_covariates", []) if str(c).strip()]
                if covs:
                    return " + ".join(covs)
                params = snapshot.get("params", {}) if isinstance(snapshot, dict) else {}
                scope = params.get("tuning_scope", snapshot.get("scope", "-")) if isinstance(snapshot, dict) else "-"
                return str(scope or "Sin covariables")

            def _extract_mode(snapshot):
                params = snapshot.get("params", {}) if isinstance(snapshot, dict) else {}
                mode_value = params.get("tuning_mode", snapshot.get("mode", "")) if isinstance(snapshot, dict) else ""
                mode_text = str(mode_value).strip()
                if mode_text:
                    return mode_text
                n_cov = len([c for c in snapshot.get("latest_covariates", []) if str(c).strip()]) if isinstance(snapshot, dict) else 0
                if n_cov <= 0:
                    return "-"
                if n_cov == 1:
                    return "1 var"
                return f"{n_cov} vars"

            def _build_record(idx, snapshot):
                params = snapshot.get("params", {}) if isinstance(snapshot, dict) else {}
                metrics = snapshot.get("metrics", {}) if isinstance(snapshot, dict) else {}
                return {
                    "uid": id(snapshot),
                    "index": int(idx),
                    "model_id": int(idx) + 1,
                    "snapshot": snapshot,
                    "params": params,
                    "metrics": metrics,
                    "cov_sig": _extract_cov_signature(snapshot),
                    "scope": str(params.get("tuning_scope", snapshot.get("scope", "-"))) if isinstance(snapshot, dict) else "-",
                    "mode": _extract_mode(snapshot),
                    "trees": str(params.get("n_estimators", "-")),
                    "max_features": str(params.get("max_features", "all")),
                    "min_leaf": str(params.get("min_samples_leaf", "-")),
                    "min_split": str(params.get("min_samples_split", "-")),
                }

            def _collect_filter_options():
                options = {k: set() for k, _ in _advanced_filter_fields}
                for idx, snapshot in enumerate(self.saved_models):
                    rec = _build_record(idx, snapshot)
                    for key in options:
                        options[key].add(str(rec.get(key, "-")))
                return {k: _sort_filter_values(v) for k, v in options.items()}

            def _passes_advanced_filters(record):
                for key, _ in _advanced_filter_fields:
                    selected = _advanced_filters.get(key) or set()
                    if selected and str(record.get(key, "-")) not in selected:
                        return False
                return True

            def _resolve_marker_category(record):
                mode_name = str(marker_mode_var.get() or "Combinación")
                if mode_name == "Scope":
                    return str(record.get("scope", "-"))
                if mode_name == "Modo":
                    return str(record.get("mode", "-"))
                if mode_name == "max_features":
                    return str(record.get("max_features", "-"))
                if mode_name == "min_leaf":
                    return str(record.get("min_leaf", "-"))
                if mode_name == "min_split":
                    return str(record.get("min_split", "-"))
                if mode_name == "Completo/Semiinc./Incompleto":
                    snap = record.get("snapshot", {})
                    if isinstance(snap, dict):
                        if bool(snap.get("discarded_by_screening", False)):
                            return "Incompleto (Screening)"
                        if bool(snap.get("semi_complete", False)):
                            return "Semiincompleto"
                    return "Completado"
                return str(record.get("cov_sig", "-"))

            def _open_advanced_filters_dialog():
                popup = tk.Toplevel(dlg)
                popup.title("Filtros avanzados del explorador")
                popup.geometry("980x620")
                popup.transient(dlg)
                popup.grab_set()

                ttk.Label(
                    popup,
                    text="Selecciona una o varias opciones por campo. Si dejas un campo vacío, se consideran todos.",
                    foreground="#334155",
                    wraplength=940,
                    justify="left",
                ).pack(fill=tk.X, padx=10, pady=(10, 6))

                options_map = _collect_filter_options()
                host = ttk.Frame(popup)
                host.pack(fill=tk.BOTH, expand=True, padx=10, pady=6)

                listboxes = {}
                for i, (field_key, field_label) in enumerate(_advanced_filter_fields):
                    panel = ttk.LabelFrame(host, text=field_label, padding=6)
                    panel.grid(row=i // 3, column=i % 3, sticky="nsew", padx=6, pady=6)

                    lb = tk.Listbox(panel, selectmode=tk.EXTENDED, exportselection=False, height=8)
                    sb = ttk.Scrollbar(panel, orient=tk.VERTICAL, command=lb.yview)
                    lb.configure(yscrollcommand=sb.set)
                    lb.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
                    sb.pack(side=tk.RIGHT, fill=tk.Y)

                    values = list(options_map.get(field_key, []))
                    for item in values:
                        lb.insert(tk.END, item)

                    selected_now = _advanced_filters.get(field_key) or set()
                    for pos, item in enumerate(values):
                        if item in selected_now:
                            lb.selection_set(pos)

                    listboxes[field_key] = (lb, values)

                for col in range(3):
                    host.columnconfigure(col, weight=1)
                for row in range(max(1, (len(_advanced_filter_fields) + 2) // 3)):
                    host.rowconfigure(row, weight=1)

                btns = ttk.Frame(popup)
                btns.pack(fill=tk.X, padx=10, pady=(0, 10))

                def _apply_filters_from_dialog():
                    for field_key, (_lb, values) in listboxes.items():
                        _advanced_filters[field_key] = {values[i] for i in _lb.curselection()}
                    popup.destroy()
                    _redraw()

                def _clear_filters_from_dialog():
                    for field_key, _ in _advanced_filter_fields:
                        _advanced_filters[field_key] = set()
                    popup.destroy()
                    _redraw()

                ttk.Button(btns, text="Aplicar", command=_apply_filters_from_dialog).pack(side=tk.RIGHT, padx=4)
                ttk.Button(btns, text="Cancelar", command=popup.destroy).pack(side=tk.RIGHT, padx=4)
                ttk.Button(btns, text="Limpiar filtros", command=_clear_filters_from_dialog).pack(side=tk.LEFT, padx=4)

            def _redraw():
                fig.clear()
                ax = fig.add_subplot(111)

                kx, ky, kc = vars_combo[0].get(), vars_combo[1].get(), vars_combo[2].get()
                lx, ly, lc = _get_metric_label(kx), _get_metric_label(ky), _get_metric_label(kc)
                show_filter = show_var.get()
                show_only_apt = bool(show_apt_only_var.get())
                use_heatmap = bool(heatmap_var.get())
                try:
                    base_size = float(point_size_var.get())
                except Exception:
                    base_size = 55.0
                base_size = max(12.0, min(base_size, 250.0))

                try:
                    heat_bins = max(8, int(float(str(heat_bins_var.get()).strip())))
                except Exception:
                    heat_bins = 35

                all_records = []
                apply_gap_filters = bool(apply_gap_filters_var.get())
                use_global_gap = bool(use_global_gap_var.get())
                gap_global = self._coerce_gap_threshold(gap_global_var.get(), self._PLOT_VIABILITY_GAP_DEFAULT, maximum=self._PLOT_VIABILITY_GAP_MAX)
                gap_oob_cv = self._coerce_gap_threshold(gap_oob_cv_var.get(), self._PLOT_GAP_OOB_CV_DEFAULT, maximum=self._PLOT_VIABILITY_GAP_MAX)
                gap_oob_test = self._coerce_gap_threshold(gap_oob_test_var.get(), self._PLOT_GAP_OOB_TEST_DEFAULT, maximum=self._PLOT_VIABILITY_GAP_MAX)
                gap_cv_test = self._coerce_gap_threshold(gap_cv_test_var.get(), self._PLOT_GAP_CV_TEST_DEFAULT, maximum=self._PLOT_VIABILITY_GAP_MAX)
                if use_global_gap:
                    gap_oob_cv = float(gap_global)
                    gap_oob_test = float(gap_global)
                    gap_cv_test = float(gap_global)

                def _valid_metric_value(key, value):
                    if value is None:
                        return False
                    try:
                        v = float(value)
                    except (TypeError, ValueError):
                        return False
                    if not np.isfinite(v):
                        return False
                    if str(key).startswith("c_index") or str(key) == "oob_score":
                        return 0.0 <= v <= 1.0
                    if str(key) == "ibs":
                        return v >= 0.0
                    if str(key) == "tau":
                        return v > 0.0
                    return True

                _completion_filter = show_completed_var.get()

                for idx, snap in enumerate(self.saved_models):
                    _is_discarded = bool(snap.get("discarded_by_screening", False))
                    if _completion_filter == "Solo completados" and _is_discarded:
                        continue
                    if _completion_filter == "Solo incompletos" and not _is_discarded:
                        continue
                    rec = _build_record(idx, snap)
                    m = rec.get("metrics", {}) if isinstance(rec, dict) else {}
                    vx, vy, vc = m.get(kx), m.get(ky), m.get(kc)
                    if not (_valid_metric_value(kx, vx) and _valid_metric_value(ky, vy) and _valid_metric_value(kc, vc)):
                        continue
                    vx, vy, vc = float(vx), float(vy), float(vc)

                    _bss_v = m.get("bss")
                    _gaps_ok, _gap_diag = self._passes_metric_gap_thresholds(
                        m,
                        oob_cv_threshold=gap_oob_cv,
                        oob_test_threshold=gap_oob_test,
                        cv_test_threshold=gap_cv_test,
                    )
                    _bss_ok = _bss_v is None or not np.isfinite(float(_bss_v)) or float(_bss_v) > -0.05
                    apt = bool(_bss_ok and _gaps_ok)

                    rec["x"] = vx
                    rec["y"] = vy
                    rec["c"] = vc
                    rec["apt"] = apt
                    rec["gap_ok"] = bool(_gaps_ok)
                    rec["gap_diag"] = _gap_diag
                    all_records.append(rec)

                new_records = [r for r in all_records if r.get("uid") not in _pulse_seen_uids]
                for r in all_records:
                    _pulse_seen_uids.add(r.get("uid"))

                visible_records = []
                for rec in all_records:
                    apt = bool(rec.get("apt"))

                    if show_only_apt:
                        if not apt:
                            continue
                    else:
                        if show_filter == "Solo aptos" and not apt:
                            continue
                        if show_filter == "Solo no aptos" and apt:
                            continue

                    if apply_gap_filters and not bool(rec.get("gap_ok", True)):
                        continue

                    if not _passes_advanced_filters(rec):
                        continue

                    visible_records.append(rec)

                if not visible_records and not new_records:
                    ax.text(0.5, 0.5, "No hay modelos con datos suficientes para graficar.",
                            ha="center", va="center", fontsize=11, color="#999")
                    canvas.draw()
                    return

                all_x = [r.get("x") for r in visible_records]
                all_y = [r.get("y") for r in visible_records]
                all_c = [r.get("c") for r in visible_records]
                all_idx = [r.get("index") for r in visible_records]
                all_apt = [bool(r.get("apt")) for r in visible_records]
                all_snap = [r.get("snapshot") for r in visible_records]
                marker_categories = [_resolve_marker_category(r) for r in visible_records]
                last_visible_idx = None
                if visible_records and all_records:
                    latest_uid = all_records[-1].get("uid")
                    for i, rec in enumerate(visible_records):
                        if rec.get("uid") == latest_uid:
                            last_visible_idx = i
                            break

                norm = None
                mappable = None
                c_arr = np.asarray(all_c, dtype=float) if all_c else np.asarray([], dtype=float)

                if c_arr.size > 0 and len(np.unique(np.round(c_arr, 8))) > 1:
                    norm = mcolors.Normalize(vmin=float(np.min(c_arr)), vmax=float(np.max(c_arr)))

                if visible_records:
                    if use_heatmap and len(visible_records) >= 2:
                        hb = ax.hexbin(
                            all_x,
                            all_y,
                            C=all_c,
                            reduce_C_function=np.mean,
                            gridsize=heat_bins,
                            cmap="viridis",
                            mincnt=1,
                            linewidths=0.0,
                            alpha=0.92,
                            zorder=1,
                        )
                        mappable = hb
                    else:
                        group_counts = Counter(marker_categories)
                        ordered_groups = [g for g, _ in group_counts.most_common()]
                        marker_map = {g: _marker_symbols[i % len(_marker_symbols)] for i, g in enumerate(ordered_groups)}
                        max_legend_groups = 10
                        legend_groups = set(ordered_groups[:max_legend_groups])
                        hidden_group_points = sum(group_counts[g] for g in ordered_groups[max_legend_groups:])

                        for group in ordered_groups:
                            group_points = [rec for rec, cat in zip(visible_records, marker_categories) if cat == group]
                            gx = [p.get("x") for p in group_points]
                            gy = [p.get("y") for p in group_points]
                            gc = [p.get("c") for p in group_points]
                            label = f"{group} ({len(group_points)})" if (len(ordered_groups) > 1 and group in legend_groups) else None
                            _use_compl_mode = (marker_mode_var.get() == "Completo/Semiinc./Incompleto")
                            if _use_compl_mode:
                                if group == "Completado":
                                    _mk = marker_complete_var.get()
                                    _edge = "none"
                                    _lw = 0.0
                                    _alpha = 0.9
                                elif group == "Semiincompleto":
                                    _mk = "D"
                                    _edge = "#f59e0b"
                                    _lw = 1.4
                                    _alpha = 0.82
                                else:  # Incompleto (Screening)
                                    _mk = marker_incomplete_var.get()
                                    _edge = "#475569"
                                    _lw = 1.2
                                    _alpha = 0.75
                            else:
                                _mk = marker_map.get(group, "o")
                                _edge = "none"
                                _lw = 0.0
                                _alpha = 0.9
                            sc = ax.scatter(
                                gx,
                                gy,
                                c=gc if norm is not None else "#3b82f6",
                                cmap="viridis" if norm is not None else None,
                                norm=norm,
                                marker=_mk,
                                s=base_size,
                                alpha=_alpha,
                                edgecolors=_edge,
                                linewidths=_lw,
                                zorder=2,
                                label=label,
                            )
                            if norm is not None and mappable is None:
                                mappable = sc

                        if hidden_group_points > 0:
                            ax.scatter(
                                [],
                                [],
                                marker="o",
                                s=max(18.0, base_size * 0.65),
                                facecolors="none",
                                edgecolors="#64748b",
                                linewidths=1.0,
                                label=f"Otros tipos ({hidden_group_points})",
                            )

                # IC por punto en forma de cruz para ejes X/Y (si existen CIs)
                if bool(show_ci_var.get()) and visible_records:
                    ci_count = 0
                    for i, snap in enumerate(all_snap):
                        metrics_i = snap.get("metrics", {}) if isinstance(snap, dict) else {}
                        drawn = self._draw_ci_cross_for_point(
                            ax,
                            all_x[i],
                            all_y[i],
                            metrics_i,
                            kx,
                            ky,
                            color="#475569",
                            alpha=0.33,
                            linewidth=0.9,
                            zorder=1.7,
                        )
                        if drawn:
                            ci_count += 1
                    if ci_count > 0:
                        ax.plot([], [], color="#475569", alpha=0.45, lw=1.0, label=f"IC cruz ({ci_count})")

                # Pulso blanco temporal para puntos nuevos, aunque no entren al filtro actual.
                if new_records:
                    nx = [r.get("x") for r in new_records]
                    ny = [r.get("y") for r in new_records]
                    bg_rgb = np.asarray(ax.get_facecolor()[:3], dtype=float)
                    inv_rgb = np.clip(1.0 - bg_rgb, 0.0, 1.0)
                    pulse_edge = mcolors.to_hex(inv_rgb)
                    luminance = float(0.2126 * bg_rgb[0] + 0.7152 * bg_rgb[1] + 0.0722 * bg_rgb[2])
                    pulse_fill = "#ffffff" if luminance < 0.55 else "#111827"

                    ax.scatter(nx, ny, facecolors="none", edgecolors=pulse_edge, s=base_size * 3.5,
                               linewidths=2.6, alpha=0.92, zorder=11)
                    ax.scatter(nx, ny, facecolors=pulse_fill, edgecolors=pulse_edge, s=base_size * 2.6,
                               linewidths=1.2, alpha=0.90, zorder=12, label=f"Nuevos ({len(new_records)})")

                # Colorbar
                if mappable is not None and all_c:
                    cbar = fig.colorbar(mappable, ax=ax, pad=0.02, fraction=0.05)
                    cbar.set_label(lc, fontsize=9)
                    c_min = float(np.min(c_arr))
                    c_max = float(np.max(c_arr))
                    c_unique = len(np.unique(np.round(np.asarray(all_c, dtype=float), 8)))
                    tick_count = int(max(3, min(7, c_unique)))
                    cbar.set_ticks(np.linspace(c_min, c_max, num=tick_count))
                    if str(kc) in {"bss", "ibs", "c_index_cv_std"}:
                        cbar.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.4f"))
                    elif str(kc) == "tau":
                        cbar.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
                    else:
                        cbar.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
                    cbar.update_ticks()
                    cbar.ax.tick_params(labelsize=8)

                # Coherence diagonal when both axes are C-index family
                if all_x and all_y and self._should_draw_coherence_diagonal(kx, ky):
                    minv = min(float(np.min(all_x)), float(np.min(all_y)))
                    maxv = max(float(np.max(all_x)), float(np.max(all_y)))
                    ax.plot([minv, maxv], [minv, maxv], color="#f59e0b", lw=1.6, ls="--", label="Diagonal y=x (C-index/OOB)")

                # Forzar límites para incluir también el pulso de nuevos modelos fuera del filtro.
                axis_records = list(visible_records) + list(new_records)
                if axis_records:
                    x_vals = np.asarray([r.get("x") for r in axis_records], dtype=float)
                    y_vals = np.asarray([r.get("y") for r in axis_records], dtype=float)
                    if x_vals.size > 0 and y_vals.size > 0:
                        x_span = float(np.max(x_vals) - np.min(x_vals))
                        y_span = float(np.max(y_vals) - np.min(y_vals))
                        x_pad = max(0.01, 0.06 * x_span) if x_span > 0 else max(0.01, abs(float(np.mean(x_vals))) * 0.03)
                        y_pad = max(0.01, 0.06 * y_span) if y_span > 0 else max(0.01, abs(float(np.mean(y_vals))) * 0.03)
                        ax.set_xlim(float(np.min(x_vals) - x_pad), float(np.max(x_vals) + x_pad))
                        ax.set_ylim(float(np.min(y_vals) - y_pad), float(np.max(y_vals) + y_pad))

                # Best by selected progress metric
                metric_key, metric_higher_better = self._resolve_tuning_progress_metric_key()
                best_metric_idx = None
                best_metric_val = None
                for i, snap in enumerate(all_snap):
                    mv = snap.get("metrics", {}).get(metric_key)
                    if not _valid_metric_value(metric_key, mv):
                        continue
                    mvf = float(mv)
                    if best_metric_idx is None:
                        best_metric_idx = i
                        best_metric_val = mvf
                        continue
                    if best_metric_val is None:
                        best_metric_idx = i
                        best_metric_val = mvf
                        continue
                    ref = float(best_metric_val)
                    is_better = (mvf > ref) if metric_higher_better else (mvf < ref)
                    if is_better:
                        best_metric_idx = i
                        best_metric_val = mvf
                if best_metric_idx is not None and all_x and all_y:
                    pass  # marcador quitado por solicitud del usuario

                # Best clinical among apt models
                clinical_best_idx = None
                try:
                    apt_snaps = [snap for snap, is_apt in zip(all_snap, all_apt) if is_apt]
                    if not apt_snaps:
                        apt_snaps = list(all_snap)
                    if apt_snaps:
                        ranked_apt, _ = self._rank_tuning_results(list(apt_snaps))
                        if ranked_apt:
                            best_clinical_snap = ranked_apt[0]
                            for i, snap in enumerate(all_snap):
                                if snap is best_clinical_snap:
                                    clinical_best_idx = i
                                    break
                except Exception:
                    clinical_best_idx = None
                if clinical_best_idx is not None and all_x and all_y:
                    pass  # marcador quitado por solicitud del usuario

                if last_visible_idx is not None and all_x and all_y and last_visible_idx < len(all_x):
                    ax.scatter([all_x[last_visible_idx]], [all_y[last_visible_idx]], c="#000000", marker="o", s=base_size * 2.2,
                               edgecolors="#ffffff", linewidths=1.6, zorder=14,
                               label="Último modelo")

                # Highlight active model currently loaded (if visible)
                active_idx = getattr(self, "active_saved_model_index", None)
                if active_idx is not None and active_idx in all_idx and all_x and all_y:
                    pos_active = all_idx.index(active_idx)
                    ax.scatter([all_x[pos_active]], [all_y[pos_active]], c="#39ff14", marker="o", s=base_size * 2.6,
                               edgecolors="black", linewidths=2.0, zorder=11,
                               label="Modelo activo")

                # ID labels: shown only when user toggles the checkbox
                if bool(show_ids_var.get()) and all_x and all_y:
                    x_span = max(float(np.max(all_x) - np.min(all_x)), 1e-9)
                    y_span = max(float(np.max(all_y) - np.min(all_y)), 1e-9)
                    dx = 0.006 * x_span
                    dy = 0.008 * y_span
                    for i, (px, py) in enumerate(zip(all_x, all_y)):
                        mid = all_idx[i] + 1 if i < len(all_idx) else (i + 1)
                        jitter_x = (((i % 5) - 2) * 0.35) * dx
                        jitter_y = (((i % 7) - 3) * 0.35) * dy
                        ax.text(px + jitter_x, py + jitter_y, str(mid),
                                fontsize=6, color="#1F2937", alpha=0.85,
                                ha="left", va="bottom", zorder=6)

                ax.set_xlabel(lx, fontsize=10)
                ax.set_ylabel(ly, fontsize=10)
                ax.set_title("Explorador de modelos RSF", fontsize=11, fontweight="bold")
                handles, labels = ax.get_legend_handles_labels()
                if handles:
                    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3, fontsize=8, framealpha=0.92)
                ax.grid(True, alpha=0.15)

                _scatter_data["all_x"] = all_x
                _scatter_data["all_y"] = all_y
                _scatter_data["all_c"] = all_c
                _scatter_data["all_idx"] = all_idx
                _scatter_data["all_apt"] = all_apt
                _scatter_data["records"] = visible_records
                _scatter_data["best_metric_idx"] = best_metric_idx
                _scatter_data["clinical_best_idx"] = clinical_best_idx
                _scatter_data["ax"] = ax
                _scatter_data["kx"] = kx
                _scatter_data["ky"] = ky
                _scatter_data["kc"] = kc
                _scatter_data["apply_gap_filters"] = apply_gap_filters
                _scatter_data["gap_oob_cv"] = gap_oob_cv
                _scatter_data["gap_oob_test"] = gap_oob_test
                _scatter_data["gap_cv_test"] = gap_cv_test
                _scatter_data["use_global_gap"] = use_global_gap
                _scatter_data["gap_global"] = gap_global
                _scatter_data["last_visible_idx"] = last_visible_idx

                info_parts = [f"Modelos visibles: {len(all_x)}"]
                if show_only_apt:
                    info_parts.append("Filtro activo: solo aptos")
                n_adv = sum(1 for values in _advanced_filters.values() if values)
                if n_adv:
                    info_parts.append(f"Filtros avanzados: {n_adv} activos")
                if apply_gap_filters:
                    if use_global_gap:
                        info_parts.append(
                            f"Umbrales ON | Δ único(CV/Test/OOB)≤{gap_global:.2f}"
                        )
                    else:
                        info_parts.append(
                            f"Umbrales ON | OOB-CV≤{gap_oob_cv:.2f} | OOB-C-test≤{gap_oob_test:.2f} | CV-C-test≤{gap_cv_test:.2f}"
                        )
                else:
                    info_parts.append("Umbrales OFF")
                if use_heatmap:
                    info_parts.append(f"Vista: mapa de calor (bins={heat_bins})")
                if new_records:
                    info_parts.append(f"Nuevos: {len(new_records)} (pulso blanco)")
                if clinical_best_idx is not None:
                    bc_snap = self.saved_models[all_idx[clinical_best_idx]]
                    bc_p = bc_snap.get("params", {})
                    info_parts.append(f"Mejor clínico: #{all_idx[clinical_best_idx]+1} ({self._format_candidate_params_short(bc_p)})")
                if best_metric_idx is not None:
                    info_parts.append(f"Mejor métrica: #{all_idx[best_metric_idx]+1}")
                if last_visible_idx is not None and last_visible_idx < len(all_idx):
                    info_parts.append(f"Último: #{all_idx[last_visible_idx]+1} (negro)")
                info_var.set("  |  ".join(info_parts))

                fig.tight_layout(rect=(0.0, 0.08, 1.0, 1.0))
                canvas.draw()

            def _on_hover(event):
                if event.inaxes is None:
                    return
                ax = _scatter_data.get("ax")
                if ax is None or not _scatter_data.get("all_x"):
                    return
                all_x = _scatter_data["all_x"]
                all_y = _scatter_data["all_y"]
                all_idx = _scatter_data["all_idx"]
                all_records_visible = _scatter_data.get("records", [])

                # Simple 2D distance
                best_dist, best_i = float("inf"), None
                for i in range(len(all_x)):
                    try:
                        px, py = ax.transData.transform((all_x[i], all_y[i]))
                        dist = ((event.x - px) ** 2 + (event.y - py) ** 2) ** 0.5
                        if dist < best_dist:
                            best_dist = dist
                            best_i = i
                    except Exception:
                        continue

                if best_i is not None and best_dist < 30:
                    rec = all_records_visible[best_i] if best_i < len(all_records_visible) else None
                    snap_idx = all_idx[best_i]
                    snap = rec.get("snapshot") if isinstance(rec, dict) else self.saved_models[snap_idx]
                    p = snap.get("params", {}) if isinstance(snap, dict) else {}
                    m = snap.get("metrics", {}) if isinstance(snap, dict) else {}
                    kx = _scatter_data.get("kx", "")
                    ky = _scatter_data.get("ky", "")
                    kc = _scatter_data.get("kc", "")
                    parts = [
                        f"Modelo #{snap_idx + 1}",
                        self._format_candidate_params_short(p),
                        f"{_get_metric_label(kx)}={self._format_metric(m.get(kx))}",
                        f"{_get_metric_label(ky)}={self._format_metric(m.get(ky))}",
                        f"{_get_metric_label(kc)}={self._format_metric(m.get(kc))}",
                    ]
                    for mk, mlbl in [("c_index_uno", "C-Uno"), ("c_index_cv_mean", "CV"),
                                      ("c_index_test", "C-test"), ("oob_score", "OOB"), ("bss", "BSS")]:
                        if mk not in (kx, ky, kc):
                            val = m.get(mk)
                            if val is not None and np.isfinite(val):
                                parts.append(f"{mlbl}={self._format_metric(val)}")
                    tags = []
                    if best_i == _scatter_data.get("best_metric_idx"):
                        tags.append("MEJOR METRICA")
                    if best_i == _scatter_data.get("clinical_best_idx"):
                        tags.append("MEJOR CLINICO")
                    if best_i == _scatter_data.get("last_visible_idx"):
                        tags.append("ULTIMO (NEGRO)")
                    if tags:
                        parts.append("[" + ", ".join(tags) + "]")
                    cov_line = self._format_covariates_short(snap.get("latest_covariates", []), max_items=99) if isinstance(snap, dict) else "(sin covariables)"
                    info_var.set("  |  ".join(parts) + f"\nCovariables: {cov_line}")
                else:
                    n_total = len(_scatter_data.get("all_idx", []))
                    info_parts = [f"Modelos visibles: {n_total}"]
                    if _scatter_data.get("apply_gap_filters"):
                        if _scatter_data.get("use_global_gap"):
                            info_parts.append(
                                f"Umbrales ON | Δ único(CV/Test/OOB)≤{_scatter_data.get('gap_global', self._PLOT_VIABILITY_GAP_DEFAULT):.2f}"
                            )
                        else:
                            info_parts.append(
                                f"Umbrales ON | OOB-CV≤{_scatter_data.get('gap_oob_cv', self._PLOT_GAP_OOB_CV_DEFAULT):.2f}"
                                f" | OOB-C-test≤{_scatter_data.get('gap_oob_test', self._PLOT_GAP_OOB_TEST_DEFAULT):.2f}"
                                f" | CV-C-test≤{_scatter_data.get('gap_cv_test', self._PLOT_GAP_CV_TEST_DEFAULT):.2f}"
                            )
                    info_var.set("  |  ".join(info_parts))

            canvas.mpl_connect("motion_notify_event", _on_hover)

            _redraw()
        except Exception as explorer_exc:
            messagebox.showerror("Error en explorador", f"No se pudo abrir el explorador de modelos:\n{explorer_exc}")
            import traceback; traceback.print_exc()

    def _clear_saved_models(self):
        self.saved_models = []
        self.active_saved_model_index = None
        self._refresh_saved_models_tree()
        self._reset_results_view()

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    def plot_feature_importance(self):
        self.importance_fig.clear()
        ax = self.importance_fig.add_subplot(111)

        if not isinstance(self.feature_importance_df, pd.DataFrame) or self.feature_importance_df.empty:
            ax.text(0.5, 0.5, "Ajusta un RSF para ver la importancia de variables.", ha="center", va="center")
            self._apply_plot_text_overrides(ax, "importance")
            self.importance_canvas.draw()
            return

        top_df = self.feature_importance_df.sort_values("importance", ascending=True).copy()
        self.importance_fig.set_size_inches(7, max(4.2, 0.33 * len(top_df) + 1.4))
        xerr = None
        if {"importance_lower", "importance_upper"}.issubset(top_df.columns):
            xerr = self._build_asymmetric_errorbars(
                top_df["importance"].to_numpy(dtype=float),
                top_df["importance_lower"].to_numpy(dtype=float),
                top_df["importance_upper"].to_numpy(dtype=float),
            )

        ax.barh(
            top_df["feature"],
            top_df["importance"],
            color="#1f77b4",
            alpha=0.85,
            xerr=xerr,
            error_kw={"ecolor": "#08306b", "elinewidth": 1.1, "capsize": 4, "alpha": 0.9},
        )
        ax.set_title("Importancia por permutación (RSF)")
        ax.set_xlabel("VIMP = disminución media del desempeño (DE/IC muestran variabilidad)")
        ax.set_ylabel("Variable")
        ax.grid(True, axis="x", alpha=0.2)
        if xerr is not None:
            ax.text(
                0.98,
                0.02,
                "Barras = media ± IC 95%",
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=8,
                color="#334155",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="#cbd5e1"),
            )

        self._apply_plot_text_overrides(ax, "importance")
        self.importance_fig.tight_layout()
        self.importance_canvas.draw()

    def plot_risk_groups_km(self):
        self.km_fig.clear()
        ax = self.km_fig.add_subplot(111)

        if self.latest_prediction_df is None or self.latest_prediction_df.empty:
            ax.text(0.5, 0.5, "Ajusta un RSF para ver Kaplan-Meier por grupos de riesgo.", ha="center", va="center")
            self._apply_plot_text_overrides(ax, "km")
            self.km_canvas.draw()
            return

        try:
            kmf = KaplanMeierFitter()
            df = self.latest_prediction_df.copy()
            for group_name, group_df in df.groupby("risk_group", dropna=False, observed=False):
                label = str(group_name)
                durations = pd.to_numeric(group_df[self.latest_duration_col], errors="coerce")
                events = pd.to_numeric(group_df[self.latest_event_col], errors="coerce").fillna(0) > 0
                kmf.fit(durations, event_observed=events, label=label)
                kmf.plot_survival_function(ax=ax, ci_show=True, ci_alpha=0.16)

            # Comparación global: KM observado de datos vs supervivencia promedio del modelo.
            durations_all = pd.to_numeric(df[self.latest_duration_col], errors="coerce")
            events_all = pd.to_numeric(df[self.latest_event_col], errors="coerce").fillna(0) > 0
            valid_mask = durations_all.notna()
            if valid_mask.any():
                kmf.fit(durations_all[valid_mask], event_observed=events_all[valid_mask], label="KM observado (global)")
                km_obs = kmf.survival_function_.copy()
                ax.step(
                    km_obs.index.to_numpy(dtype=float),
                    km_obs.iloc[:, 0].to_numpy(dtype=float),
                    where="post",
                    linewidth=2.6,
                    color="#111827",
                    linestyle="-",
                    alpha=0.95,
                    label="KM observado (global)",
                    zorder=5,
                )

            if self.model is not None and self.latest_fit_dataframe is not None and self.latest_covariates:
                try:
                    cohort_df = self.latest_fit_dataframe[self.latest_covariates].copy()
                    cohort_encoded = self._encode_prediction_frame(cohort_df)
                    if cohort_encoded is not None and not cohort_encoded.empty:
                        surv_fns = self.model.predict_survival_function(cohort_encoded)
                        surv_fns = list(surv_fns)
                        if surv_fns:
                            model_times = np.asarray(surv_fns[0].x, dtype=float)
                            if model_times.size >= 2:
                                model_matrix = np.asarray([fn(model_times) for fn in surv_fns], dtype=float)
                                model_mean = np.nanmean(model_matrix, axis=0)
                                ax.step(
                                    model_times,
                                    model_mean,
                                    where="post",
                                    linewidth=2.2,
                                    color="#6d28d9",
                                    linestyle="--",
                                    alpha=0.95,
                                    label="RSF promedio (cohorte)",
                                    zorder=6,
                                )
                except Exception:
                    pass

            ax.set_title("Kaplan-Meier RSF: cuartiles + comparación datos vs modelo")
            ax.set_xlabel("Tiempo")
            ax.set_ylabel("Probabilidad de supervivencia")
            ax.grid(True, alpha=0.2)
            ax.text(
                0.98,
                0.02,
                "Sombras = IC 95% KM",
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=8,
                color="#334155",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="#cbd5e1"),
            )
        except Exception as exc:
            ax.text(0.5, 0.5, f"No se pudo generar KM por riesgo:\n{exc}", ha="center", va="center")

        self._apply_plot_text_overrides(ax, "km")
        self.km_fig.tight_layout()
        self.km_canvas.draw()

    def plot_calibration(self):
        self.calibration_fig.clear()
        ax = self.calibration_fig.add_subplot(111)

        if not isinstance(self.latest_calibration_df, pd.DataFrame) or self.latest_calibration_df.empty:
            # Attempt to recalculate calibration on-demand
            if self.model is not None:
                self._recalculate_calibration_from_model()
        if not isinstance(self.latest_calibration_df, pd.DataFrame) or self.latest_calibration_df.empty:
            ax.text(0.5, 0.5, "Ajusta un RSF para ver la calibración temporal.", ha="center", va="center")
            self._apply_plot_text_overrides(ax, "calibration")
            self.calibration_canvas.draw()
            return

        calibration_df = self.latest_calibration_df.dropna(subset=["predicted_survival", "observed_survival"]).copy()
        if calibration_df.empty:
            ax.text(0.5, 0.5, "No hay suficientes datos para la gráfica de calibración.", ha="center", va="center")
            self._apply_plot_text_overrides(ax, "calibration")
            self.calibration_canvas.draw()
            return

        max_axis = max(
            float(calibration_df["predicted_survival"].max()),
            float(calibration_df["observed_survival"].max()),
            1.0,
        )
        sizes = np.clip(calibration_df["n"].to_numpy(dtype=float) * 8.0, 40.0, 220.0)
        xerr = None
        yerr = None
        if {"predicted_survival_lower", "predicted_survival_upper"}.issubset(calibration_df.columns):
            xerr = self._build_asymmetric_errorbars(
                calibration_df["predicted_survival"].to_numpy(dtype=float),
                calibration_df["predicted_survival_lower"].to_numpy(dtype=float),
                calibration_df["predicted_survival_upper"].to_numpy(dtype=float),
            )
        if {"observed_survival_lower", "observed_survival_upper"}.issubset(calibration_df.columns):
            yerr = self._build_asymmetric_errorbars(
                calibration_df["observed_survival"].to_numpy(dtype=float),
                calibration_df["observed_survival_lower"].to_numpy(dtype=float),
                calibration_df["observed_survival_upper"].to_numpy(dtype=float),
            )

        ax.errorbar(
            calibration_df["predicted_survival"],
            calibration_df["observed_survival"],
            xerr=xerr,
            yerr=yerr,
            fmt="o",
            markersize=6,
            color="#d62728",
            ecolor="#7f1d1d",
            elinewidth=1.0,
            capsize=4,
            alpha=0.85,
            label="Grupo observado ± IC 95%",
        )
        ax.scatter(
            calibration_df["predicted_survival"],
            calibration_df["observed_survival"],
            s=sizes,
            color="#d62728",
            alpha=0.25,
        )
        for _, row in calibration_df.iterrows():
            ax.annotate(str(row["group"]), (row["predicted_survival"], row["observed_survival"]), xytext=(4, 4), textcoords="offset points", fontsize=8)

        ax.plot([0, max_axis], [0, max_axis], linestyle="--", color="#1f77b4", linewidth=1.5, label="Calibración ideal")
        horizon_text = f" a t={self.latest_eval_time:.3g}" if self.latest_eval_time is not None else ""
        ax.set_title(f"Calibración RSF{horizon_text}")
        ax.set_xlabel("Supervivencia predicha")
        ax.set_ylabel("Supervivencia observada (KM)")
        ax.set_xlim(0, max_axis)
        ax.set_ylim(0, max_axis)
        ax.grid(True, alpha=0.2)
        ax.legend(loc="best", fontsize=8)

        self._apply_plot_text_overrides(ax, "calibration")
        self.calibration_fig.tight_layout()
        self.calibration_canvas.draw()

    def plot_brier_curve(self):
        self.brier_fig.clear()
        ax = self.brier_fig.add_subplot(111)

        ibs_val = None
        if isinstance(self.results, dict):
            ibs_val = self.results.get("ibs")
        if ibs_val is None and self.saved_models and self.active_saved_model_index is not None and 0 <= self.active_saved_model_index < len(self.saved_models):
            ibs_val = self.saved_models[self.active_saved_model_index].get("metrics", {}).get("ibs")

        brier_df = self.latest_brier_df.copy(deep=True) if isinstance(self.latest_brier_df, pd.DataFrame) else pd.DataFrame()
        if brier_df.empty:
            brier_df = self._build_brier_fallback_dataframe()
        # Fallback: recalculate from model + full data if still empty
        if (not isinstance(brier_df, pd.DataFrame) or brier_df.empty) and self.model is not None and callable(brier_score):
            brier_df = self._recalculate_brier_from_model()

        if not isinstance(brier_df, pd.DataFrame) or brier_df.empty:
            if ibs_val is not None and pd.notna(ibs_val):
                ax.text(0.5, 0.58, "No hay curva de Brier guardada para este modelo.", ha="center", va="center")
                ax.text(0.5, 0.44, f"IBS disponible: {float(ibs_val):.4f}", ha="center", va="center", fontsize=11)
                ax.set_title(f"Curva de Brier RSF | IBS={float(ibs_val):.4f}")
            else:
                ax.text(0.5, 0.5, "Activa holdout para ver la curva de Brier / IBS del RSF.", ha="center", va="center")
            self._apply_plot_text_overrides(ax, "brier")
            self.brier_canvas.draw()
            return

        brier_df = brier_df.dropna(subset=["time", "brier_score"]).copy()
        if brier_df.empty:
            ax.text(0.5, 0.5, "No hay suficientes puntos para graficar Brier / IBS.", ha="center", va="center")
            self._apply_plot_text_overrides(ax, "brier")
            self.brier_canvas.draw()
            return

        brier_df = brier_df.sort_values("time")
        times = brier_df["time"].to_numpy(dtype=float)
        scores = brier_df["brier_score"].to_numpy(dtype=float)
        if len(times) == 1:
            ax.plot(times, scores, color="#8b5cf6", linewidth=2.0, marker="o", markersize=7, label="Brier(t)")
        else:
            ax.plot(times, scores, color="#8b5cf6", linewidth=2.2, label="Brier(t)")
            ax.fill_between(times, scores, 0, color="#8b5cf6", alpha=0.16)

        eval_time_marker = self.latest_eval_time
        if (eval_time_marker is None or not np.isfinite(eval_time_marker)) and len(times) == 1:
            eval_time_marker = float(times[0])
        if eval_time_marker is not None and np.isfinite(eval_time_marker):
            ax.axvline(float(eval_time_marker), color="#475569", linestyle="--", linewidth=1.2, label=f"t≈{float(eval_time_marker):.2f}")

        title = "Curva de Brier RSF"
        if ibs_val is not None and pd.notna(ibs_val):
            title += f" | IBS={float(ibs_val):.4f}"
        ax.set_title(title)
        ax.set_xlabel("Tiempo")
        ax.set_ylabel("Brier score")
        ax.grid(True, alpha=0.2)
        ax.legend(loc="best", fontsize=8)

        self._apply_plot_text_overrides(ax, "brier")
        self.brier_fig.tight_layout()
        self.brier_canvas.draw()

    # ------------------------------------------------------------------
    # Helper: reconstruct X_encoded from stored snapshot data
    # ------------------------------------------------------------------
    def _get_X_encoded(self):
        """Return X_encoded DataFrame reconstructed from latest_fit_dataframe.
        Returns None if the data or model is missing."""
        if not isinstance(self.latest_fit_dataframe, pd.DataFrame) or self.latest_fit_dataframe.empty:
            return None
        if not self.latest_covariates:
            return None
        try:
            available_covs = [c for c in self.latest_covariates
                              if c in self.latest_fit_dataframe.columns]
            if not available_covs:
                return None
            X_raw = self.latest_fit_dataframe[available_covs].copy()
            drop_first = bool(getattr(self, "latest_drop_first", True))
            X_enc = pd.get_dummies(X_raw, drop_first=drop_first, dummy_na=False)
            X_enc = X_enc.replace([np.inf, -np.inf], np.nan).fillna(0)
            # Force all columns to float64 — pd.get_dummies produces uint8/bool
            # which causes shap / np.isfinite to crash with "ufunc isfinite not
            # supported for the input types".
            X_enc = X_enc.astype(np.float64)
            # Align to encoded columns used when model was trained
            if self.latest_encoded_columns:
                for col in self.latest_encoded_columns:
                    if col not in X_enc.columns:
                        X_enc[col] = 0.0
                X_enc = X_enc[[c for c in self.latest_encoded_columns if c in X_enc.columns]]
            return X_enc
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Profundidad Mínima (Minimal Depth)
    # ------------------------------------------------------------------
    def plot_minimal_depth(self):
        self.mindepth_fig.clear()
        ax = self.mindepth_fig.add_subplot(111)

        if self.model is None or not hasattr(self.model, "estimators_") or not self.model.estimators_:
            ax.text(0.5, 0.5, "Ajusta un RSF para ver la Profundidad Mínima.",
                    ha="center", va="center")
            self.mindepth_canvas.draw()
            return

        X_enc = self._get_X_encoded()
        if X_enc is None:
            ax.text(0.5, 0.5, "No hay datos disponibles para calcular la Profundidad Mínima.",
                    ha="center", va="center")
            self.mindepth_canvas.draw()
            return

        # Read confidence level from UI
        try:
            ci_level = float(self.mindepth_ci_var.get()) / 100.0
        except Exception:
            ci_level = 0.95
        ci_level = float(np.clip(ci_level, 0.50, 0.999))

        feature_names = list(X_enc.columns)
        n_features = len(feature_names)
        n_estimators = len(self.model.estimators_)

        # Collect per-tree minimal depths for each feature
        depths_per_tree = {i: [] for i in range(n_features)}

        for estimator in self.model.estimators_:
            t = estimator.tree_
            node_depth = np.zeros(t.node_count, dtype=int)
            stack = [(0, 0)]
            while stack:
                node_id, depth = stack.pop()
                node_depth[node_id] = depth
                left = t.children_left[node_id]
                right = t.children_right[node_id]
                if left != -1:  # internal node
                    stack.append((left, depth + 1))
                    stack.append((right, depth + 1))
            # For each feature find minimum split depth in this tree
            feat_min = {}
            for node_id in range(t.node_count):
                feat = t.feature[node_id]
                if 0 <= feat < n_features:
                    d = node_depth[node_id]
                    if feat not in feat_min or d < feat_min[feat]:
                        feat_min[feat] = d
            for feat_idx, min_d in feat_min.items():
                depths_per_tree[feat_idx].append(min_d)

        # Compute mean, std, and CI for each feature
        mean_depths = np.full(n_features, np.nan)
        ci_lower = np.full(n_features, np.nan)
        ci_upper = np.full(n_features, np.nan)
        z_val = self._get_confidence_z_value(ci_level)

        for feat_idx in range(n_features):
            arr = np.asarray(depths_per_tree[feat_idx], dtype=float)
            if arr.size > 0:
                m = float(np.mean(arr))
                mean_depths[feat_idx] = m
                if arr.size >= 2:
                    se = float(np.std(arr, ddof=1)) / np.sqrt(arr.size)
                    ci_lower[feat_idx] = m - z_val * se
                    ci_upper[feat_idx] = m + z_val * se
                else:
                    ci_lower[feat_idx] = m
                    ci_upper[feat_idx] = m

        # Sort ascending (lower depth = more important)
        valid_mask = np.isfinite(mean_depths)
        sort_idx = np.argsort(np.where(valid_mask, mean_depths, np.inf))
        sorted_names = [feature_names[i] for i in sort_idx]
        sorted_depths = np.array([mean_depths[i] for i in sort_idx])
        sorted_lower = np.array([ci_lower[i] for i in sort_idx])
        sorted_upper = np.array([ci_upper[i] for i in sort_idx])

        self.mindepth_fig.set_size_inches(7, max(4, 0.35 * n_features + 1.2))
        colors = ["#0077BB" if np.isfinite(d) else "#cccccc" for d in sorted_depths]
        clean_depths = np.array([d if np.isfinite(d) else 0 for d in sorted_depths])

        # Build error bars
        xerr = None
        has_ci = np.isfinite(sorted_lower).any() and np.isfinite(sorted_upper).any()
        if has_ci:
            err_lo = np.clip(clean_depths - np.where(np.isfinite(sorted_lower), sorted_lower, clean_depths), 0, None)
            err_hi = np.clip(np.where(np.isfinite(sorted_upper), sorted_upper, clean_depths) - clean_depths, 0, None)
            xerr = np.array([err_lo, err_hi])

        ax.barh(sorted_names, clean_depths, color=colors, alpha=0.85,
                xerr=xerr,
                error_kw={"ecolor": "#08306b", "elinewidth": 1.1, "capsize": 4, "alpha": 0.9})
        ax.set_title("Profundidad Mínima Media (RSF)")
        ax.set_xlabel("Profundidad mínima media (menor = más importante)")
        ax.set_ylabel("Variable")
        ax.grid(True, axis="x", alpha=0.2)
        ci_pct = int(round(ci_level * 100))
        note_text = f"Barras = media ± IC {ci_pct}%  |  Gris = variable nunca usada"
        ax.text(
            0.98, 0.02,
            note_text,
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=8, color="#334155",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="#cbd5e1"),
        )
        self.mindepth_fig.tight_layout()
        self.mindepth_canvas.draw()

    # ------------------------------------------------------------------
    # SHAP Summary Plot
    # ------------------------------------------------------------------
    def plot_shap_summary(self):
        self.shap_fig.clear()
        ax = self.shap_fig.add_subplot(111)

        if self.model is None:
            ax.text(0.5, 0.5, "Ajusta un RSF para ver los valores SHAP.",
                    ha="center", va="center")
            self.shap_canvas.draw()
            return

        try:
            import shap as _shap
        except ImportError:
            ax.text(0.5, 0.5,
                    "La librería 'shap' no está instalada.\n"
                    "Instálala con:  pip install shap",
                    ha="center", va="center", fontsize=10,
                    bbox=dict(boxstyle="round", facecolor="#fef3c7", edgecolor="#d97706"))
            self.shap_canvas.draw()
            return

        X_enc = self._get_X_encoded()
        if X_enc is None or X_enc.empty:
            ax.text(0.5, 0.5, "No hay datos disponibles para calcular SHAP.",
                    ha="center", va="center")
            self.shap_canvas.draw()
            return

        n_max = max(20, int(self.shap_n_var.get()))
        if len(X_enc) > n_max:
            X_sample = X_enc.sample(n_max, random_state=42)
        else:
            X_sample = X_enc.copy()

        ax.text(0.5, 0.5, "Calculando valores SHAP…\n(puede tardar varios segundos)",
                ha="center", va="center", fontsize=11, color="#1e3a5f")
        self.shap_canvas.draw()
        self.shap_fig.canvas.flush_events()
        try:
            self.update_idletasks()
        except Exception:
            pass

        try:
            # RSF's TreeExplainer crashes internally with 'ufunc isfinite'
            # on object arrays.  Use permutation-based Explainer(model.predict)
            # which always returns clean float arrays.
            bg = _shap.sample(X_enc, min(50, len(X_enc)), random_state=42)
            explainer = _shap.Explainer(self.model.predict, bg)
            sv_obj = explainer(X_sample)
            shap_values = np.asarray(sv_obj.values, dtype=float)

            if shap_values is None or shap_values.shape[1] == 0:
                raise ValueError("SHAP returned empty values.")

            # ── Beeswarm-style summary plot (manual) ──────────────────
            self.shap_fig.clear()
            feature_names = list(X_sample.columns)
            n_feat = len(feature_names)
            mean_abs = np.abs(shap_values).mean(axis=0)
            order = np.argsort(mean_abs)  # ascending for horizontal bar

            self.shap_fig.set_size_inches(8, max(4.5, 0.38 * n_feat + 1.5))
            ax = self.shap_fig.add_subplot(111)

            for row_idx, feat_idx in enumerate(order):
                sv = shap_values[:, feat_idx].astype(float)
                fv = pd.to_numeric(X_sample.iloc[:, feat_idx], errors="coerce").to_numpy(dtype=float)
                fv_finite = np.where(np.isfinite(fv), fv, 0.0)
                ptp = np.ptp(fv_finite)
                fv_norm = (fv_finite - fv_finite.min()) / (ptp + 1e-12)
                # jitter y slightly
                rng = np.random.default_rng(42)
                jitter = rng.uniform(-0.3, 0.3, size=len(sv))
                ax.scatter(sv, row_idx + jitter, c=fv_norm, cmap="cividis",
                           s=8, alpha=0.65, vmin=0, vmax=1)

            ax.set_yticks(range(n_feat))
            ax.set_yticklabels([feature_names[i] for i in order], fontsize=9)
            ax.axvline(0, color="black", lw=0.8, linestyle="--")
            ax.set_xlabel("Valor SHAP (impacto en la predicción del modelo)")
            ax.set_title(f"SHAP Summary Plot — RSF  (n={len(X_sample)})")
            ax.grid(True, axis="x", alpha=0.15)

            import matplotlib.cm as _cm
            import matplotlib.colors as _mcolors
            sm = _cm.ScalarMappable(cmap="cividis", norm=_mcolors.Normalize(0, 1))
            sm.set_array([])
            cb = self.shap_fig.colorbar(sm, ax=ax, pad=0.01, fraction=0.03)
            cb.set_label("Valor de la variable\n(oscuro=bajo, claro=alto)", fontsize=8)

            self.shap_fig.tight_layout()
            self.shap_canvas.draw()
        except Exception as exc:
            self.shap_fig.clear()
            ax = self.shap_fig.add_subplot(111)
            ax.text(0.5, 0.5, f"Error calculando SHAP:\n{exc}",
                    ha="center", va="center", fontsize=9, color="red")
            self.shap_canvas.draw()

        # Sync SHAP dependence combos after computing
        if hasattr(self, "shap_dep_x_combo") and X_enc is not None:
            enc_cols = list(X_enc.columns)
            self.shap_dep_x_combo["values"] = enc_cols
            color_opts = list(enc_cols)
            target_df = self.latest_fit_dataframe if isinstance(self.latest_fit_dataframe, pd.DataFrame) else self.data
            if target_df is not None and "Grupo_RSF" in target_df.columns:
                color_opts.append("Grupo_RSF")
            self.shap_dep_color_combo["values"] = color_opts

    # ------------------------------------------------------------------
    # SHAP Dependence Plot
    # ------------------------------------------------------------------
    def plot_shap_dependence(self):
        """Generate a SHAP Dependence Plot to explore non-linearity and interactions."""
        if self.model is None:
            messagebox.showerror("SHAP Dependence", "Ajusta un RSF primero.")
            return

        x_var = self.shap_dep_x_var.get().strip()
        color_var = self.shap_dep_color_var.get().strip()
        if not x_var:
            messagebox.showerror("SHAP Dependence", "Selecciona una Variable X.")
            return

        try:
            import shap as _shap
        except ImportError:
            messagebox.showerror("SHAP Dependence", "La librería 'shap' no está instalada.\nInstálala con: pip install shap")
            return

        X_enc = self._get_X_encoded()
        if X_enc is None or X_enc.empty:
            messagebox.showerror("SHAP Dependence", "No hay datos disponibles.")
            return

        if x_var not in X_enc.columns:
            messagebox.showerror("SHAP Dependence", f"La variable '{x_var}' no existe en los datos codificados.")
            return

        n_max = max(20, int(self.shap_n_var.get()))
        if len(X_enc) > n_max:
            try:
                unique_x = X_enc[x_var].nunique()
                if 1 < unique_x <= 10:
                    def _sample_group(g):
                        k = max(1, n_max // unique_x)
                        return g.sample(min(len(g), k), random_state=42)
                    X_sample = X_enc.groupby(x_var, group_keys=False).apply(_sample_group)
                    if len(X_sample) < n_max:
                        rem = X_enc.drop(X_sample.index)
                        X_sample = pd.concat([X_sample, rem.sample(min(len(rem), n_max - len(X_sample)), random_state=42)])
                else:
                    X_sample = X_enc.sample(n_max, random_state=42)
            except Exception:
                X_sample = X_enc.sample(n_max, random_state=42)
            sample_idx = X_sample.index
        else:
            X_sample = X_enc.copy()
            sample_idx = X_sample.index

        try:
            # RSF's TreeExplainer crashes internally with 'ufunc isfinite'
            # on object arrays.  Use permutation-based Explainer(model.predict)
            # which always returns clean float arrays.
            bg = _shap.sample(X_enc, min(50, len(X_enc)), random_state=42)
            explainer = _shap.Explainer(self.model.predict, bg)
            sv_obj = explainer(X_sample)
            shap_values = np.asarray(sv_obj.values, dtype=float)

            if shap_values is None or shap_values.shape[1] == 0:
                raise ValueError("SHAP returned empty values.")

            feat_idx = list(X_sample.columns).index(x_var)
            shap_x = shap_values[:, feat_idx].astype(float)

            # ---------------------------------------------------------------
            # Determine if x_var should be treated as categorical.
            # RULE: respect ONLY what the user explicitly configured in
            # variable_configs via the "Configurar categorías" dialog.
            # No automatic heuristics — a numeric 0/1 stays numeric unless
            # the user said otherwise.
            # ---------------------------------------------------------------
            raw_x_col = X_sample[x_var]
            target_df_orig = (self.latest_fit_dataframe
                              if isinstance(self.latest_fit_dataframe, pd.DataFrame)
                              else self.data)

            # Find the original covariate name (x_var may be a dummy column
            # named "ORIGINAL_Category" after pd.get_dummies).
            _orig_cov_name = None
            for cov in (self.latest_covariates or []):
                if cov == x_var or x_var.startswith(cov + "_"):
                    _orig_cov_name = cov
                    break

            # Check if the user explicitly configured this covariate as categorical
            _var_cfg = {}
            if _orig_cov_name and isinstance(getattr(self, "variable_configs", {}), dict):
                _var_cfg = self.variable_configs.get(_orig_cov_name, {})

            _treat_as = str(_var_cfg.get("treat_as", _var_cfg.get("compare_mode", ""))).strip().lower()
            _user_said_categorical = _treat_as not in {"quantitative", "numeric", "continuous",
                                                       "continuo", "cuantitativa", ""}

            if _user_said_categorical and _orig_cov_name and target_df_orig is not None \
                    and _orig_cov_name in target_df_orig.columns:
                # Use the original (pre-encoding) column labels from the fitted dataframe
                orig_series = target_df_orig.loc[sample_idx, _orig_cov_name]
                x_display_labels = orig_series.astype(str).to_numpy()
                # Map each unique label to an integer position for the axis
                unique_cats = list(dict.fromkeys(x_display_labels))  # ordered, deduplicated
                cat_to_pos = {lbl: i for i, lbl in enumerate(unique_cats)}
                feature_x = np.array([cat_to_pos[lbl] for lbl in x_display_labels], dtype=float)
                x_axis_is_categorical = True
            else:
                # Treat as continuous numeric — convert directly
                feature_x = pd.to_numeric(raw_x_col, errors="coerce").to_numpy(dtype=float)
                x_display_labels = None
                unique_cats = None
                x_axis_is_categorical = False

            # ---------------------------------------------------------------
            # Build color mapping — also respects variable_configs for the
            # color variable (same rule as X: user config, not heuristics).
            # ---------------------------------------------------------------
            color_text_vals = None
            _color_is_discrete = False   # whether to draw legend instead of colorbar

            def _color_cov_name(col_name):
                """Find the original covariate name for a (possibly dummy) column."""
                for cov in (self.latest_covariates or []):
                    if cov == col_name or col_name.startswith(cov + "_"):
                        return cov
                return col_name

            def _user_configured_as_categorical(col_name):
                """Return True only if the user explicitly set treat_as=categorical."""
                orig = _color_cov_name(col_name)
                cfg = {}
                if orig and isinstance(getattr(self, "variable_configs", {}), dict):
                    cfg = self.variable_configs.get(orig, {})
                ta = str(cfg.get("treat_as", cfg.get("compare_mode", ""))).strip().lower()
                return ta not in {"quantitative", "numeric", "continuous",
                                  "continuo", "cuantitativa", ""}

            def _resolve_color_series(raw_c, col_name):
                """Return (color_vals, color_text_vals, cmap, is_discrete).
                Discrete only when user explicitly configured the variable as categorical."""
                is_cat_by_config = _user_configured_as_categorical(col_name)
                if is_cat_by_config or pd.api.types.is_bool_dtype(raw_c) \
                        or not pd.api.types.is_numeric_dtype(raw_c):
                    cat_obj = pd.Categorical(raw_c.astype(str))
                    return cat_obj.codes.astype(float), np.array(cat_obj).astype(str), "tab10", True
                # Purely numeric, user did not configure as categorical → continuous colormap
                vals = pd.to_numeric(raw_c, errors="coerce").to_numpy(dtype=float)
                return vals, None, "cividis", False

            if color_var == "Grupo_RSF":
                if target_df_orig is not None and "Grupo_RSF" in target_df_orig.columns:
                    raw_vals = target_df_orig.loc[sample_idx, "Grupo_RSF"]
                    cat_obj = pd.Categorical(raw_vals)
                    color_vals = cat_obj.codes.astype(float)
                    color_text_vals = np.array(cat_obj).astype(str)
                    color_label = "Grupo_RSF"
                    cmap = "tab10"
                    _color_is_discrete = True
                else:
                    color_vals = feature_x.copy()
                    color_label = x_var
                    cmap = "cividis"
            elif color_var and color_var in X_sample.columns:
                color_vals, color_text_vals, cmap, _color_is_discrete = _resolve_color_series(X_sample[color_var], color_var)
                color_label = color_var
            elif color_var and isinstance(target_df_orig, pd.DataFrame) and color_var in target_df_orig.columns:
                raw_c = target_df_orig.loc[sample_idx, color_var]
                color_vals, color_text_vals, cmap, _color_is_discrete = _resolve_color_series(raw_c, color_var)
                color_label = color_var
            else:
                color_vals = feature_x.copy()
                color_label = x_var
                cmap = "cividis"

            # ---------------------------------------------------------------
            # Filter out NaN/Inf — for categorical X, feature_x is integer
            # positions (always finite), so only filter on shap and color.
            # ---------------------------------------------------------------
            if x_axis_is_categorical:
                valid_mask = np.isfinite(shap_x) & np.isfinite(color_vals)
            else:
                valid_mask = np.isfinite(shap_x) & np.isfinite(feature_x) & np.isfinite(color_vals)

            shap_x = shap_x[valid_mask]
            feature_x = feature_x[valid_mask]
            color_vals = color_vals[valid_mask]
            if x_display_labels is not None:
                x_display_labels = x_display_labels[valid_mask]
            if color_text_vals is not None:
                color_text_vals = color_text_vals[valid_mask]

            if len(shap_x) == 0:
                raise ValueError("Todos los valores resultaron NaN/Inf tras la conversión.")

            # ---------------------------------------------------------------
            # Analytical Report — adapted for categorical vs continuous X
            # ---------------------------------------------------------------
            report_lines = []
            report_lines.append(f"Análisis Numérico de SHAP Dependence\nVariable: {x_var}")
            report_lines.append("="*45)

            if x_axis_is_categorical and unique_cats is not None:
                # Group-based report for categorical X
                report_lines.append(f"\nTipo de variable: CATEGÓRICA ({len(unique_cats)} categorías)")
                report_lines.append(f"\n1. SHAP Medio por Categoría")
                group_stats = []
                for cat in unique_cats:
                    mask_g = (x_display_labels == cat) & np.isfinite(shap_x)
                    if mask_g.sum() > 0:
                        g_shap = shap_x[mask_g]
                        g_mean = float(g_shap.mean())
                        g_sd = float(g_shap.std()) if len(g_shap) > 1 else 0.0
                        group_stats.append((cat, g_mean, g_sd, int(mask_g.sum())))
                        direction_g = "↑ riesgo" if g_mean > 0 else "↓ riesgo"
                        report_lines.append(f"   {cat}: media={g_mean:+.4f} ± {g_sd:.4f}  (n={mask_g.sum()})  {direction_g}")

                if group_stats:
                    best_cat = max(group_stats, key=lambda t: t[1])
                    worst_cat = min(group_stats, key=lambda t: t[1])
                    report_lines.append(f"\n2. Categoría de Mayor Riesgo")
                    report_lines.append(f"   '{best_cat[0]}' → SHAP medio = {best_cat[1]:+.4f}")
                    report_lines.append(f"\n3. Categoría de Menor Riesgo (protectora)")
                    report_lines.append(f"   '{worst_cat[0]}' → SHAP medio = {worst_cat[1]:+.4f}")
                    report_lines.append(f"\n4. Diferencia Máxima entre Categorías")
                    delta = best_cat[1] - worst_cat[1]
                    report_lines.append(f"   Δ SHAP = {delta:.4f}  ({'diferencia importante' if abs(delta) > 0.02 else 'diferencia pequeña'})")

                if color_label != x_var and np.ptp(color_vals) > 0 and len(np.unique(feature_x)) > 1:
                    try:
                        residuals = shap_x - np.array([
                            shap_x[feature_x == pos].mean() if (feature_x == pos).sum() > 0 else 0.0
                            for pos in feature_x
                        ])
                        if np.std(residuals) > 0 and np.std(color_vals) > 0:
                            corr_resid = float(np.corrcoef(color_vals, residuals)[0, 1])
                            report_lines.append(f"\n5. Interacción con '{color_label}'")
                            report_lines.append(f"   r residual = {corr_resid:.2f}")
                            if abs(corr_resid) > 0.2:
                                direction = "AUMENTA" if corr_resid > 0 else "REDUCE"
                                report_lines.append(f"   Evidencia de interacción: '{color_label}' tiende a {direction} el riesgo dentro de cada categoría.")
                            else:
                                report_lines.append("   Sin evidencia de interacción significativa.")
                    except Exception:
                        pass
            elif len(feature_x) > 2 and np.ptp(feature_x) > 0:
                # Continuous X report
                corr_x = np.corrcoef(feature_x, shap_x)[0, 1]
                if corr_x > 0.3:
                    trend_txt = "Incremento de riesgo. Valores mayores de la variable aumentan el riesgo/mortalidad."
                elif corr_x < -0.3:
                    trend_txt = "Efecto protector. Valores mayores de la variable reducen el riesgo/mortalidad."
                else:
                    trend_txt = "Efecto no lineal o débil. No hay una tendencia monotónica clara."
                report_lines.append(f"\n1. Comportamiento de la Pendiente")
                report_lines.append(f"   Correlación lineal (r) = {corr_x:.2f}")
                report_lines.append(f"   Interpretación: {trend_txt}")

                try:
                    p3 = np.polyfit(feature_x, shap_x, 3)
                    poly3 = np.poly1d(p3)
                    x_grid = np.linspace(feature_x.min(), feature_x.max(), 500)
                    y_grid = poly3(x_grid)
                    roots = []
                    for i in range(len(y_grid)-1):
                        if (y_grid[i] <= 0 and y_grid[i+1] > 0) or (y_grid[i] >= 0 and y_grid[i+1] < 0):
                            roots.append(x_grid[i])
                    report_lines.append(f"\n2. El Umbral Crítico (Cruce SHAP = 0)")
                    if roots:
                        rt_str = ", ".join([f"{r:.2f}" for r in roots[:2]])
                        report_lines.append(f"   El impacto cambia aproximadamente en X ≈ {rt_str}")
                        report_lines.append(f"   (Estimado suavizado usando regresión polinomial)")
                    else:
                        if y_grid.mean() > 0:
                            report_lines.append("   Efecto consistentemente positivo (aumenta riesgo) en todo el rango.")
                        else:
                            report_lines.append("   Efecto consistentemente negativo (protector) en todo el rango.")
                except Exception:
                    report_lines.append("\n2. El Umbral Crítico (Cruce SHAP = 0)\n   No se pudo estimar el umbral suavizado.")

                if color_label != x_var and np.ptp(color_vals) > 0:
                    try:
                        residuals = shap_x - np.polyval(np.polyfit(feature_x, shap_x, 2), feature_x)
                        corr_resid = np.corrcoef(color_vals, residuals)[0, 1]
                        report_lines.append(f"\n3. Identificación de Interacción\n   Variable Secundaria: {color_label}")
                        report_lines.append(f"   Correlación parcial (dispersión vertical) r = {corr_resid:.2f}")
                        if abs(corr_resid) > 0.2:
                            direction = "AUMENTA" if corr_resid > 0 else "REDUCE"
                            report_lines.append(f"   Interpretación: Hay evidencia de interacción. Para un mismo nivel de {x_var}, valores más altos en {color_label} tienden a {direction} el riesgo.")
                        else:
                            report_lines.append(f"   Interpretación: Interacción débil o nula.")
                    except Exception:
                        pass
            else:
                report_lines.append("\nNo hay suficiente variabilidad en X para realizar el análisis numérico.")

            report_text = "\n".join(report_lines)

            # Create pop-up dialog
            dialog = tk.Toplevel(self.winfo_toplevel())
            dialog.title(f"SHAP Dependence & Analysis: {x_var}")
            dialog.geometry("1100x700")
            dialog.transient(self.winfo_toplevel())

            main_pw = ttk.PanedWindow(dialog, orient=tk.HORIZONTAL)
            main_pw.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

            fig_frame = tk.Frame(main_pw)
            main_pw.add(fig_frame, weight=3)

            text_frame = tk.Frame(main_pw, bg="#f8fafc", bd=1, relief=tk.SUNKEN)
            main_pw.add(text_frame, weight=1)

            lbl_header = tk.Label(text_frame, text="Auditoría Numérica", font=("Segoe UI", 11, "bold"), bg="#f8fafc", fg="#0f172a")
            lbl_header.pack(fill=tk.X, pady=(5, 0))

            txt_report = tk.Text(text_frame, wrap=tk.WORD, font=("Consolas", 10), bg="#f8fafc", fg="#1e293b", bd=0, padx=10, pady=10)
            txt_report.pack(fill=tk.BOTH, expand=True)
            txt_report.insert("1.0", report_text)
            txt_report.config(state=tk.DISABLED)

            fig = plt.figure(figsize=(8, 5.5))
            canvas = FigureCanvasTkAgg(fig, master=fig_frame)
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

            ax = fig.add_subplot(111)

            # Read representation flags
            use_color = bool(self.shap_dep_show_color_var.get()) if hasattr(self, "shap_dep_show_color_var") else True
            use_size = bool(self.shap_dep_show_size_var.get()) if hasattr(self, "shap_dep_show_size_var") else False
            use_value = bool(self.shap_dep_show_value_var.get()) if hasattr(self, "shap_dep_show_value_var") else False

            # ---------------------------------------------------------------
            # Jitter for categorical X (avoids all points stacking on integer pos)
            # ---------------------------------------------------------------
            rng_jitter = np.random.default_rng(42)
            if x_axis_is_categorical and unique_cats is not None and len(unique_cats) > 0:
                jitter_x_scatter = rng_jitter.uniform(-0.28, 0.28, size=len(feature_x))
            else:
                jitter_x_scatter = np.zeros(len(feature_x))

            # --- Size mapping (compute early, used in both value-text and dot modes) ---
            if use_size:
                cv_finite_s = color_vals[np.isfinite(color_vals)]
                s_min, s_max = 12, 180
                if len(cv_finite_s) > 0 and np.ptp(cv_finite_s) > 0:
                    size_norm = (color_vals - cv_finite_s.min()) / np.ptp(cv_finite_s)
                else:
                    size_norm = np.full_like(color_vals, 0.5)
                sizes = s_min + size_norm * (s_max - s_min)
            else:
                sizes = 25

            # ---------------------------------------------------------------
            # Helper: draw discrete color legend for categorical color vars
            # ---------------------------------------------------------------
            def _add_discrete_legend(ax_obj, c_vals, c_texts, c_label, c_cmap_name):
                import matplotlib.cm as _lcm
                import matplotlib.patches as _lpatch
                if c_texts is None:
                    return
                unique_codes = sorted(set(c_vals[np.isfinite(c_vals)].astype(int)))
                # Build a mapping code → label string
                code_label_map = {}
                for code, txt in zip(c_vals.astype(int), c_texts):
                    code_label_map[code] = txt
                try:
                    lut = _lcm.get_cmap(c_cmap_name)
                    n_u = max(len(unique_codes), 1)
                    handles = []
                    for code in unique_codes:
                        lbl = code_label_map.get(code, str(code))
                        color_i = lut(code / max(n_u - 1, 1))
                        handles.append(_lpatch.Patch(color=color_i, label=lbl))
                    ax_obj.legend(handles=handles, title=c_label, fontsize=7,
                                  title_fontsize=7, loc="upper right", framealpha=0.85)
                except Exception:
                    pass

            # --- Render points ---
            if use_value:
                # The number IS the marker — no scatter dots
                import matplotlib.cm as _val_cm
                import matplotlib.colors as _val_mcolors
                cv_finite = color_vals[np.isfinite(color_vals)]
                if use_color and len(cv_finite) > 0 and np.ptp(cv_finite) > 0:
                    val_norm = _val_mcolors.Normalize(vmin=float(cv_finite.min()),
                                                      vmax=float(cv_finite.max()))
                    val_cmap = _val_cm.get_cmap(cmap) if isinstance(cmap, str) else cmap
                else:
                    val_norm = None
                    val_cmap = None

                # IMPORTANT: ax.text() does NOT update axis limits.
                # Plot invisible scatter to force correct auto-scaling.
                ax.scatter(feature_x + jitter_x_scatter, shap_x, s=0.1, alpha=0.01)

                # Add small Y-jitter to prevent identical positions from stacking
                y_range = np.ptp(shap_x) if np.ptp(shap_x) > 0 else 1.0
                jitter_y = rng_jitter.uniform(-0.012, 0.012, size=len(shap_x)) * y_range

                # Font size scaled to point count
                n_pts = len(feature_x)
                if n_pts <= 50:
                    fsize = 8
                elif n_pts <= 150:
                    fsize = 6.5
                else:
                    fsize = 5

                for i in range(n_pts):
                    if color_text_vals is not None:
                        val_txt = color_text_vals[i]
                    else:
                        val_txt = f"{color_vals[i]:.1f}"
                    if use_size:
                        fsize_i = fsize * (0.6 + 0.8 * ((sizes[i] - 12) / max(168, 1)))
                    else:
                        fsize_i = fsize
                    if val_cmap is not None and val_norm is not None:
                        rgba = list(val_cmap(val_norm(color_vals[i])))
                        rgba[3] = 0.85
                        txt_color = tuple(rgba)
                    else:
                        txt_color = "#0077BB"
                    ax.text(feature_x[i] + jitter_x_scatter[i], shap_x[i] + jitter_y[i], val_txt,
                            fontsize=fsize_i, fontweight="bold",
                            ha="center", va="center", color=txt_color)

                # Add colorbar/legend when using color
                if use_color and val_cmap is not None and val_norm is not None:
                    if _color_is_discrete and color_text_vals is not None:
                        _add_discrete_legend(ax, color_vals, color_text_vals, color_label, cmap)
                    else:
                        sm = _val_cm.ScalarMappable(cmap=val_cmap, norm=val_norm)
                        sm.set_array([])
                        cb = fig.colorbar(sm, ax=ax, pad=0.02, fraction=0.04)
                        cb.set_label(f"{color_label} (color)", fontsize=9)
            else:
                # Standard scatter dots (sizes already computed above)
                if use_color:
                    sc = ax.scatter(feature_x + jitter_x_scatter, shap_x, c=color_vals, cmap=cmap,
                                   s=sizes, alpha=0.75, edgecolors="#33415555", linewidths=0.4)
                    if _color_is_discrete and color_text_vals is not None:
                        _add_discrete_legend(ax, color_vals, color_text_vals, color_label, cmap)
                    else:
                        cb = fig.colorbar(sc, ax=ax, pad=0.02, fraction=0.04)
                        cb.set_label(f"{color_label} (color)", fontsize=9)
                else:
                    sc = ax.scatter(feature_x + jitter_x_scatter, shap_x, color="#0077BB",
                                   s=sizes, alpha=0.65, edgecolors="#33415555", linewidths=0.4)

            ax.axhline(0, color="black", lw=1.0, linestyle="--", alpha=0.6)

            # ---------------------------------------------------------------
            # X-axis labels: use category names for categorical X
            # ---------------------------------------------------------------
            if x_axis_is_categorical and unique_cats is not None:
                ax.set_xticks(range(len(unique_cats)))
                ax.set_xticklabels(unique_cats, rotation=30, ha="right", fontsize=9)
                ax.set_xlim(-0.6, len(unique_cats) - 0.4)
                ax.set_xlabel(f"Categoría de {x_var}", fontsize=11)
            else:
                ax.set_xlabel(f"Valor de {x_var}", fontsize=11)
            ax.set_ylabel(f"Valor SHAP (Impacto en Riesgo)", fontsize=11)

            # Build title with active representations
            repr_parts = []
            if use_color:
                repr_parts.append("color")
            if use_size:
                repr_parts.append("tamaño")
            if use_value:
                repr_parts.append("valor")
            repr_str = ", ".join(repr_parts) if repr_parts else "ninguna"
            ax.set_title(f"SHAP Dependence Plot: {x_var}\n(Interacción: {color_label} → {repr_str})", fontsize=12)
            ax.grid(True, alpha=0.15)

            # Add size legend if using size (and no discrete legend already added)
            if use_size and not (_color_is_discrete and use_color):
                import matplotlib.lines as mlines
                cv_finite = color_vals[np.isfinite(color_vals)]
                if len(cv_finite) > 0 and np.ptp(cv_finite) > 0:
                    lo_val, hi_val = float(cv_finite.min()), float(cv_finite.max())
                    small_h = mlines.Line2D([], [], marker='o', color='gray', markersize=4,
                                            linestyle='None', label=f'{color_label}={lo_val:.1f}')
                    big_h = mlines.Line2D([], [], marker='o', color='gray', markersize=11,
                                          linestyle='None', label=f'{color_label}={hi_val:.1f}')
                    ax.legend(handles=[small_h, big_h], loc='upper left', fontsize=7,
                              title='Tamaño', title_fontsize=7, framealpha=0.8)

            fig.tight_layout()
            canvas.draw()
        except Exception as exc:
            messagebox.showerror("Error SHAP Dependence", f"Error:\n{exc}")

    # ------------------------------------------------------------------
    # PDP — Partial Dependence Plot
    # ------------------------------------------------------------------
    def plot_pdp(self):
        self.pdp_fig.clear()
        ax = self.pdp_fig.add_subplot(111)

        if self.model is None:
            ax.text(0.5, 0.5, "Ajusta un RSF para ver el PDP.",
                    ha="center", va="center")
            self.pdp_canvas.draw()
            return

        X_enc = self._get_X_encoded()
        if X_enc is None or X_enc.empty:
            ax.text(0.5, 0.5, "No hay datos disponibles para calcular el PDP.",
                    ha="center", va="center")
            self.pdp_canvas.draw()
            return

        feature = self.pdp_covariate_var.get().strip()
        if not feature or feature not in X_enc.columns:
            ax.text(0.5, 0.5, "Selecciona una variable en el selector de arriba.",
                    ha="center", va="center")
            self.pdp_canvas.draw()
            return

        try:
            n_grid = max(10, int(self.pdp_grid_var.get()))
        except Exception:
            n_grid = 40

        try:
            col_vals = X_enc[feature].values.astype(float)
            unique_vals = np.unique(col_vals)
            if len(unique_vals) <= n_grid:
                grid = unique_vals
            else:
                plo = np.percentile(col_vals, 2)
                phi = np.percentile(col_vals, 98)
                grid = np.linspace(plo, phi, n_grid)

            # For ICE, sample at most 200 rows
            n_ice_max = 200
            if len(X_enc) > n_ice_max:
                X_base = X_enc.sample(n_ice_max, random_state=42).reset_index(drop=True)
            else:
                X_base = X_enc.copy().reset_index(drop=True)

            # Compute predictions for each grid value
            ice_matrix = np.zeros((len(X_base), len(grid)), dtype=float)
            for j, gv in enumerate(grid):
                X_copy = X_base.copy()
                X_copy[feature] = gv
                ice_matrix[:, j] = self.model.predict(X_copy).astype(float)

            pdp_line = ice_matrix.mean(axis=0)

            # Normalize ICE to start at 0 (centered ICE)
            show_ice = bool(self.pdp_ice_var.get())

            # Determine color-by variable (always define, used in title)
            color_by = self.pdp_ice_color_var.get().strip() if hasattr(self, "pdp_ice_color_var") else "(Ninguna)"
            use_colormap = (show_ice and color_by and color_by != "(Ninguna)"
                            and color_by in X_base.columns and color_by != feature)

            if show_ice:
                ice_centered = ice_matrix - ice_matrix[:, 0:1]
                if use_colormap:
                    import matplotlib.cm as _cm
                    import matplotlib.colors as _mcolors
                    color_vals = pd.to_numeric(X_base[color_by], errors="coerce").to_numpy(dtype=float)
                    color_vals = np.where(np.isfinite(color_vals), color_vals, 0.0)
                    c_min, c_max = float(np.nanmin(color_vals)), float(np.nanmax(color_vals))
                    norm = _mcolors.Normalize(vmin=c_min, vmax=c_max)
                    cmap = _cm.get_cmap("cividis")
                    for i, row in enumerate(ice_centered):
                        rgba = cmap(norm(color_vals[i]))
                        ax.plot(grid, row, color=rgba, alpha=0.45, linewidth=1.2)
                    # Add colorbar
                    sm = _cm.ScalarMappable(cmap=cmap, norm=norm)
                    sm.set_array([])
                    cb = self.pdp_fig.colorbar(sm, ax=ax, pad=0.02, fraction=0.04)
                    cb.set_label(color_by, fontsize=9)
                else:
                    for row in ice_centered:
                        ax.plot(grid, row, color="#0077BB", alpha=0.15, linewidth=1.0)

            ax.plot(grid, pdp_line, color="#EE7733", linewidth=2.2, label="PDP (media)")
            ax.fill_between(grid,
                            np.percentile(ice_matrix, 25, axis=0),
                            np.percentile(ice_matrix, 75, axis=0),
                            color="#EE7733", alpha=0.12, label="IQR 25–75%")

            ax.set_xlabel(feature)
            ax.set_ylabel("Riesgo predicho (score RSF)")
            if show_ice and use_colormap:
                ax.set_title(f"Partial Dependence Plot — {feature}\n(ICE coloreado por {color_by})")
            else:
                ax.set_title(f"Partial Dependence Plot — {feature}")
            ax.grid(True, alpha=0.2)
            ax.legend(fontsize=8)
            self.pdp_fig.tight_layout()
            self.pdp_canvas.draw()
        except Exception as exc:
            ax.text(0.5, 0.5, f"Error calculando PDP:\n{exc}",
                    ha="center", va="center", fontsize=9, color="red")
            self.pdp_canvas.draw()

    # ------------------------------------------------------------------
    # Proximity Matrix
    # ------------------------------------------------------------------
    def plot_proximity_matrix(self):
        self.proximity_fig.clear()
        ax = self.proximity_fig.add_subplot(111)

        if self.model is None or not hasattr(self.model, "apply"):
            ax.text(0.5, 0.5, "Ajusta un RSF para ver la Matriz de Proximidad.",
                    ha="center", va="center")
            self.proximity_canvas.draw()
            return

        X_enc = self._get_X_encoded()
        if X_enc is None or X_enc.empty:
            ax.text(0.5, 0.5, "No hay datos disponibles para calcular la Proximidad.",
                    ha="center", va="center")
            self.proximity_canvas.draw()
            return

        try:
            n_max = max(20, int(self.proximity_n_var.get()))
        except Exception:
            n_max = 150

        try:
            if len(X_enc) > n_max:
                X_sub = X_enc.sample(n_max, random_state=42).reset_index(drop=True)
            else:
                X_sub = X_enc.copy().reset_index(drop=True)

            # leaf_indices: shape (n_samples, n_trees)
            leaf_indices = self.model.apply(X_sub)
            n_samples = len(X_sub)
            n_trees = leaf_indices.shape[1]

            # Vectorized proximity: prox[i,j] = fraction of trees where same leaf
            prox = np.zeros((n_samples, n_samples), dtype=float)
            for t in range(n_trees):
                leaves = leaf_indices[:, t]
                same = (leaves[:, None] == leaves[None, :]).astype(float)
                prox += same
            prox /= n_trees

            # Optional hierarchical clustering reorder
            do_cluster = bool(self.proximity_cluster_var.get())
            if do_cluster and n_samples > 2:
                try:
                    from scipy.cluster.hierarchy import linkage, leaves_list as _ll
                    dist = 1.0 - prox
                    dist = np.clip(dist, 0, 1)
                    # Make symmetric and zero diagonal
                    dist = (dist + dist.T) / 2
                    np.fill_diagonal(dist, 0)
                    condensed = dist[np.triu_indices(n_samples, k=1)]
                    Z = linkage(condensed, method="average")
                    order = _ll(Z)
                    prox = prox[np.ix_(order, order)]
                except Exception:
                    pass

            # Get risk scores for annotations
            risk_scores = None
            if self.latest_prediction_df is not None and "risk_score" in self.latest_prediction_df.columns:
                try:
                    if len(X_enc) > n_max:
                        risk_scores = np.array([float(v) for v in
                                                self.model.predict(X_sub).tolist()])
                    else:
                        risk_scores = self.latest_prediction_df["risk_score"].values[:n_samples]
                except Exception:
                    pass

            im = ax.imshow(prox, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1,
                           interpolation="nearest")
            self.proximity_fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02,
                                         label="Proximidad (fracción)")
            ax.set_title(f"Matriz de Proximidad RSF  (n={n_samples})")
            ax.set_xlabel("Paciente")
            ax.set_ylabel("Paciente")
            tick_step = max(1, n_samples // 10)
            ax.set_xticks(range(0, n_samples, tick_step))
            ax.set_yticks(range(0, n_samples, tick_step))
            ax.tick_params(labelsize=7)

            self.proximity_fig.tight_layout()
            self.proximity_canvas.draw()
        except Exception as exc:
            ax.text(0.5, 0.5, f"Error calculando Proximidad:\n{exc}",
                    ha="center", va="center", fontsize=9, color="red")
            self.proximity_canvas.draw()

    # ------------------------------------------------------------------
    # Clustering Jerárquico desde Proximidad + Cruce Clínico
    # ------------------------------------------------------------------
    def _generate_proximity_clusters(self):
        """Assign Grupo_RSF to all patients using hierarchical clustering on full proximity matrix."""
        if self.model is None or not hasattr(self.model, "apply"):
            messagebox.showerror("Clustering RSF", "Ajusta un RSF primero.")
            return
        X_enc = self._get_X_encoded()
        if X_enc is None or X_enc.empty:
            messagebox.showerror("Clustering RSF", "No hay datos codificados disponibles.")
            return

        try:
            from scipy.cluster.hierarchy import linkage, fcluster
            from scipy.spatial.distance import squareform

            n_clusters = max(2, int(self.proximity_n_clusters_var.get()))
            method = self.proximity_linkage_var.get() or "ward"

            # Full proximity matrix for ALL patients
            leaf_indices = self.model.apply(X_enc)
            n_samples = len(X_enc)
            n_trees = leaf_indices.shape[1]
            prox = np.zeros((n_samples, n_samples), dtype=float)
            for t in range(n_trees):
                leaves = leaf_indices[:, t]
                same = (leaves[:, None] == leaves[None, :]).astype(float)
                prox += same
            prox /= n_trees

            # Distance matrix
            dist = 1.0 - prox
            dist = np.clip(dist, 0, 1)
            dist = (dist + dist.T) / 2
            np.fill_diagonal(dist, 0)

            condensed = squareform(dist, checks=False)
            Z = linkage(condensed, method=method)
            labels = fcluster(Z, t=n_clusters, criterion='maxclust')

            # Assign to fit dataframe
            if isinstance(self.latest_fit_dataframe, pd.DataFrame) and len(self.latest_fit_dataframe) == n_samples:
                self.latest_fit_dataframe["Grupo_RSF"] = labels.astype(int)
            elif self.data is not None and len(self.data) == n_samples:
                self.data["Grupo_RSF"] = labels.astype(int)
            if isinstance(self.latest_fit_dataframe, pd.DataFrame):
                target_df = self.latest_fit_dataframe
            elif self.data is not None:
                target_df = self.data
            else:
                target_df = None

            if target_df is not None and "Grupo_RSF" in target_df.columns:
                counts = target_df["Grupo_RSF"].value_counts().sort_index()
                desc = ", ".join([f"G{k}={v}" for k, v in counts.items()])
                self.proximity_cluster_status_var.set(f"✅ {n_clusters} grupos asignados: {desc}")

                # Populate cross-variable combo
                candidates = []
                if self.latest_duration_col:
                    candidates.append(f"[Supervivencia] {self.latest_duration_col}")
                for col in target_df.columns:
                    if col not in ("Grupo_RSF",) and col != self.latest_duration_col and col != self.latest_event_col:
                        candidates.append(col)
                self.proximity_cross_combo["values"] = candidates
                if candidates and not self.proximity_cross_var.get():
                    self.proximity_cross_var.set(candidates[0])
                # Also populate SHAP dependence combos
                if hasattr(self, "shap_dep_x_combo"):
                    enc_cols = list(X_enc.columns)
                    self.shap_dep_x_combo["values"] = enc_cols
                    color_opts = enc_cols + ["Grupo_RSF"]
                    self.shap_dep_color_combo["values"] = color_opts

            messagebox.showinfo("Clustering RSF",
                                f"Se asignaron {n_clusters} grupos (Grupo_RSF) a {n_samples} pacientes.\n"
                                f"Método de enlace: {method}")
        except Exception as exc:
            messagebox.showerror("Error en Clustering", f"No se pudo generar clústeres:\n{exc}")

    def _run_cluster_bivariate_analysis(self):
        """Run bivariate analysis: boxplot + Kruskal-Wallis or bar + Chi² depending on variable type."""
        target_df = self.latest_fit_dataframe if isinstance(self.latest_fit_dataframe, pd.DataFrame) else self.data
        if target_df is None or "Grupo_RSF" not in target_df.columns:
            messagebox.showerror("Cruce Clínico", "Primero genera los grupos con 'Generar Grupo_RSF'.")
            return
        cross_var = self.proximity_cross_var.get().strip()
        if not cross_var:
            messagebox.showerror("Cruce Clínico", "Selecciona una variable a cruzar.")
            return

        # Check if it's survival
        if cross_var.startswith("[Supervivencia]"):
            self._run_cluster_km_analysis()
            return

        if cross_var not in target_df.columns:
            messagebox.showerror("Cruce Clínico", f"La variable '{cross_var}' no existe en los datos.")
            return

        try:
            from scipy import stats
            dialog = tk.Toplevel(self.winfo_toplevel())
            dialog.title(f"Cruce Clínico: Grupo_RSF × {cross_var}")
            dialog.geometry("900x650")
            dialog.transient(self.winfo_toplevel())

            fig = plt.figure(figsize=(8, 5))
            canvas = FigureCanvasTkAgg(fig, master=dialog)
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

            results_text = scrolledtext.ScrolledText(dialog, wrap=tk.WORD, height=8)
            results_text.pack(fill=tk.X, padx=8, pady=(4, 8))

            ax = fig.add_subplot(111)
            series = pd.to_numeric(target_df[cross_var], errors="coerce")
            is_numeric = series.notna().sum() >= max(10, len(series) * 0.5)

            if is_numeric:
                # Boxplot + Kruskal-Wallis
                groups = []
                group_labels = []
                for g, gdf in target_df.groupby("Grupo_RSF", observed=False):
                    vals = pd.to_numeric(gdf[cross_var], errors="coerce").dropna()
                    if len(vals) > 0:
                        groups.append(vals.values)
                        group_labels.append(f"G{g} (n={len(vals)})")

                if len(groups) >= 2:
                    ax.boxplot(groups, labels=group_labels, patch_artist=True,
                               boxprops=dict(facecolor="#a5b4fc", alpha=0.7),
                               medianprops=dict(color="#1e3a5f", linewidth=2))
                    ax.set_title(f"Distribución de {cross_var} por Grupo_RSF")
                    ax.set_xlabel("Grupo RSF")
                    ax.set_ylabel(cross_var)
                    ax.grid(True, axis="y", alpha=0.2)

                    # Kruskal-Wallis
                    stat, p_value = stats.kruskal(*groups)
                    results_text.insert(tk.END, f"═══ Análisis Bivariado: {cross_var} × Grupo_RSF ═══\n\n")
                    results_text.insert(tk.END, f"Prueba: Kruskal-Wallis\n")
                    results_text.insert(tk.END, f"Estadístico H = {stat:.4f}\n")
                    results_text.insert(tk.END, f"Valor p = {p_value:.6f}\n")
                    results_text.insert(tk.END, f"{'✅ Diferencia significativa (p < 0.05)' if p_value < 0.05 else '⚠️ Sin diferencia significativa (p ≥ 0.05)'}\n\n")

                    # Group medians
                    results_text.insert(tk.END, "Medianas por grupo:\n")
                    for label, vals in zip(group_labels, groups):
                        results_text.insert(tk.END, f"  {label}: mediana={np.median(vals):.3f}, media={np.mean(vals):.3f}\n")

                    # ANOVA as secondary
                    try:
                        f_stat, p_anova = stats.f_oneway(*groups)
                        results_text.insert(tk.END, f"\nANOVA (paramétrica): F={f_stat:.4f}, p={p_anova:.6f}\n")
                    except Exception:
                        pass
                else:
                    ax.text(0.5, 0.5, "Se necesitan al menos 2 grupos con datos.", ha="center", va="center")
            else:
                # Categorical: bar chart + Chi²
                ct = pd.crosstab(target_df["Grupo_RSF"], target_df[cross_var])
                ct.plot(kind="bar", ax=ax, alpha=0.8, edgecolor="#334155", linewidth=0.5)
                ax.set_title(f"Distribución de {cross_var} por Grupo_RSF")
                ax.set_xlabel("Grupo RSF")
                ax.set_ylabel("Frecuencia")
                ax.legend(title=cross_var, fontsize=8)
                ax.grid(True, axis="y", alpha=0.2)

                chi2, p_value, dof, expected = stats.chi2_contingency(ct)
                results_text.insert(tk.END, f"═══ Análisis Bivariado: {cross_var} × Grupo_RSF ═══\n\n")
                results_text.insert(tk.END, f"Prueba: Chi-cuadrada\n")
                results_text.insert(tk.END, f"χ² = {chi2:.4f}, gl = {dof}\n")
                results_text.insert(tk.END, f"Valor p = {p_value:.6f}\n")
                results_text.insert(tk.END, f"{'✅ Asociación significativa (p < 0.05)' if p_value < 0.05 else '⚠️ Sin asociación significativa (p ≥ 0.05)'}\n\n")
                results_text.insert(tk.END, "Tabla de contingencia:\n")
                results_text.insert(tk.END, ct.to_string() + "\n")

            fig.tight_layout()
            canvas.draw()
        except Exception as exc:
            messagebox.showerror("Error en Cruce Clínico", f"Error:\n{exc}")

    def _run_cluster_km_analysis(self):
        """Generate Kaplan-Meier curves stratified by Grupo_RSF with Log-Rank test."""
        target_df = self.latest_fit_dataframe if isinstance(self.latest_fit_dataframe, pd.DataFrame) else self.data
        if target_df is None or "Grupo_RSF" not in target_df.columns:
            messagebox.showerror("KM por Grupo_RSF", "Primero genera los grupos con 'Generar Grupo_RSF'.")
            return
        duration_col = self.latest_duration_col
        event_col = self.latest_event_col
        if not duration_col or not event_col:
            messagebox.showerror("KM por Grupo_RSF", "No se encontraron las variables de tiempo/evento del modelo.")
            return
        if duration_col not in target_df.columns or event_col not in target_df.columns:
            messagebox.showerror("KM por Grupo_RSF", f"Variables {duration_col}/{event_col} no están en los datos.")
            return

        try:
            dialog = tk.Toplevel(self.winfo_toplevel())
            dialog.title("Kaplan-Meier por Grupo_RSF")
            dialog.geometry("900x650")
            dialog.transient(self.winfo_toplevel())

            fig = plt.figure(figsize=(8, 5))
            canvas = FigureCanvasTkAgg(fig, master=dialog)
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

            results_text = scrolledtext.ScrolledText(dialog, wrap=tk.WORD, height=8)
            results_text.pack(fill=tk.X, padx=8, pady=(4, 8))

            ax = fig.add_subplot(111)
            kmf = KaplanMeierFitter()
            group_data = {}
            colors = ["#2563EB", "#DC2626", "#16A34A", "#D97706", "#7C3AED",
                      "#0891B2", "#BE123C", "#4D7C0F", "#0F766E", "#B45309"]

            for idx, (g, gdf) in enumerate(target_df.groupby("Grupo_RSF", observed=False)):
                durations = pd.to_numeric(gdf[duration_col], errors="coerce")
                events = pd.to_numeric(gdf[event_col], errors="coerce").fillna(0) > 0
                valid = durations.notna()
                if valid.sum() < 2:
                    continue
                label = f"Grupo {g} (n={valid.sum()})"
                kmf.fit(durations[valid], event_observed=events[valid], label=label)
                kmf.plot_survival_function(ax=ax, ci_show=True, ci_alpha=0.15,
                                           color=colors[idx % len(colors)])
                group_data[g] = (durations[valid].values, events[valid].values)

            ax.set_title("Kaplan-Meier por Grupo_RSF (Subgrupos de Proximidad)")
            ax.set_xlabel(duration_col)
            ax.set_ylabel("Supervivencia estimada")
            ax.grid(True, alpha=0.2)
            ax.legend(loc="best", fontsize=9)

            # Log-Rank test
            results_text.insert(tk.END, "═══ Supervivencia por Grupo_RSF ═══\n\n")
            if len(group_data) >= 2:
                try:
                    from lifelines.statistics import logrank_test as _logrank_test
                    group_keys = sorted(group_data.keys())
                    # Pairwise log-rank
                    for i in range(len(group_keys)):
                        for j in range(i + 1, len(group_keys)):
                            gi, gj = group_keys[i], group_keys[j]
                            d1, e1 = group_data[gi]
                            d2, e2 = group_data[gj]
                            lr = _logrank_test(d1, d2, event_observed_A=e1, event_observed_B=e2)
                            results_text.insert(tk.END,
                                f"Grupo {gi} vs Grupo {gj}: χ²={lr.test_statistic:.4f}, p={lr.p_value:.6f}"
                                f" {'✅' if lr.p_value < 0.05 else '⚠️'}\n")
                    # Global multivariate log-rank
                    from lifelines.statistics import multivariate_logrank_test as _ml_test
                    all_durations = np.concatenate([group_data[k][0] for k in group_keys])
                    all_events = np.concatenate([group_data[k][1] for k in group_keys])
                    all_groups = np.concatenate([np.full(len(group_data[k][0]), k) for k in group_keys])
                    mlr = _ml_test(all_durations, all_groups, all_events)
                    results_text.insert(tk.END,
                        f"\nLog-Rank Global: χ²={mlr.test_statistic:.4f}, p={mlr.p_value:.6f}"
                        f" {'✅ Diferencia global significativa' if mlr.p_value < 0.05 else '⚠️ Sin diferencia global significativa'}\n")
                except ImportError:
                    results_text.insert(tk.END, "Para Log-Rank se necesita lifelines.statistics.\n")
                except Exception as exc_lr:
                    results_text.insert(tk.END, f"Error en Log-Rank: {exc_lr}\n")

            fig.tight_layout()
            canvas.draw()
        except Exception as exc:
            messagebox.showerror("Error KM", f"Error:\n{exc}")


    def plot_single_tree(self):
        self.tree_fig.clear()

        if self.model is None or not hasattr(self.model, "estimators_") or len(self.model.estimators_) == 0:
            ax = self.tree_fig.add_subplot(111)
            ax.text(0.5, 0.5, "Ajusta un RSF para visualizar un árbol individual.",
                    ha="center", va="center", fontsize=11)
            self._apply_plot_text_overrides(ax, "tree")
            self.tree_canvas.draw()
            if hasattr(self, '_update_tree_scroll'): self._update_tree_scroll()
            return

        try:
            self._plot_single_tree_impl()
        except Exception as exc:
            self.tree_fig.clear()
            ax = self.tree_fig.add_subplot(111)
            ax.text(0.5, 0.5, f"No se pudo dibujar el árbol:\n{exc}",
                    ha="center", va="center", fontsize=10, color="#cc0000")
            self.tree_canvas.draw()
            if hasattr(self, '_update_tree_scroll'): self._update_tree_scroll()

    def _plot_single_tree_impl(self):
        from sklearn.tree import DecisionTreeRegressor as _DTR

        n_trees = len(self.model.estimators_)
        self.tree_index_spin.configure(to=n_trees - 1)
        tree_idx_raw = self._coerce_int(self.tree_index_var.get(), 0, minimum=0)
        tree_idx = max(0, min(tree_idx_raw, n_trees - 1))
        max_depth_raw = self._coerce_int(self.tree_max_depth_var.get(), 3, minimum=1)
        max_depth = max(1, min(max_depth_raw, 10))
        show_impurity = self.tree_show_impurity_var.get()

        estimator = self.model.estimators_[tree_idx]

        feature_names = None
        if isinstance(getattr(self, "latest_encoded_columns", None), list) and self.latest_encoded_columns:
            feature_names = list(self.latest_encoded_columns)
        elif self.latest_covariates:
            feature_names = list(self.latest_covariates)

        try:
            n_leaves = estimator.tree_.n_leaves if hasattr(estimator.tree_, 'n_leaves') else int(np.sum(estimator.tree_.children_left == -1))
        except Exception:
            n_leaves = "?"
        fig_h = max(5, 1.2 * (2 ** max_depth))
        fig_w = max(10, 1.5 * (2 ** max_depth))
        self.tree_fig.set_size_inches(fig_w, fig_h)

        # Wrap SurvivalTree as DecisionTreeRegressor so plot_tree accepts it
        wrapper = _DTR.__new__(_DTR)
        wrapper.tree_ = estimator.tree_
        # Algunos atributos de SurvivalTree pueden venir como NaN; sanitizarlos
        # evita errores internos de sklearn.plot_tree al convertir a entero.
        _raw_n_features = getattr(estimator, "n_features_in_", getattr(estimator.tree_, "n_features", 0))
        try:
            _n_features_in = int(_raw_n_features)
        except Exception:
            _n_features_in = int(len(feature_names)) if feature_names else 1
        if _n_features_in <= 0:
            _n_features_in = int(len(feature_names)) if feature_names else 1

        wrapper.n_features_in_ = int(_n_features_in)
        wrapper.n_outputs_ = 1
        _raw_max_features = getattr(estimator, "max_features_", _n_features_in)
        try:
            if _raw_max_features is None or (isinstance(_raw_max_features, (float, np.floating)) and not np.isfinite(_raw_max_features)):
                raise ValueError("max_features_ inválido")
            _max_features_safe = int(_raw_max_features)
        except Exception:
            _max_features_safe = int(_n_features_in)
        if _max_features_safe < 1:
            _max_features_safe = int(_n_features_in)
        wrapper.max_features_ = int(_max_features_safe)

        if feature_names and len(feature_names) != int(_n_features_in):
            feature_names = None
        # Copy all attributes plot_tree may inspect in newer sklearn versions
        for _attr, _default in [
            ("splitter", "best"),
            ("criterion", "squared_error"),
            ("max_depth", None),
            ("min_samples_split", 2),
            ("min_samples_leaf", 1),
            ("min_weight_fraction_leaf", 0.0),
            ("max_leaf_nodes", None),
            ("min_impurity_decrease", 0.0),
            ("class_weight", None),
            ("ccp_alpha", 0.0),
        ]:
            _value = getattr(estimator, _attr, _default)
            if isinstance(_value, (float, np.floating)) and not np.isfinite(_value):
                _value = _default
            setattr(wrapper, _attr, _value)

        ax = self.tree_fig.add_subplot(111)
        _fallback_note = ""
        _plot_kwargs = {
            "ax": ax,
            "max_depth": max_depth,
            "feature_names": feature_names,
            "impurity": show_impurity,
            "rounded": True,
            "fontsize": 8,
            "proportion": True,
        }
        _is_survival_tree = (
            "sksurv" in str(type(estimator).__module__).lower()
            or "survivaltree" in str(type(estimator).__name__).lower()
        )
        
        texts = None
        if not _is_survival_tree:
            try:
                texts = plot_tree(
                    wrapper,
                    filled=True,
                    **_plot_kwargs,
                )
            except Exception as _plot_exc:
                _plot_err = str(_plot_exc or "").lower()
                if "cannot convert float nan to integer" not in _plot_err and "nan" not in _plot_err:
                    raise
                _is_survival_tree = True  # Trigger manual filling fallback

        if _is_survival_tree or texts is None:
            ax.clear()
            texts = plot_tree(
                wrapper,
                filled=False,
                **_plot_kwargs,
            )
            
            # Aplicar relleno (filled) manualmente para árboles de supervivencia
            import re
            import matplotlib.pyplot as plt
            import matplotlib.colors as mcolors
            
            node_values = []
            for t in texts:
                m = re.search(r'value\s*=\s*\[*\s*([-\d\.]+)', t.get_text())
                if m:
                    node_values.append(float(m.group(1)))
            
            if node_values:
                vmin, vmax = min(node_values), max(node_values)
                cmap = plt.get_cmap("Oranges")
                if vmax > vmin:
                    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
                else:
                    norm = mcolors.Normalize(vmin=vmin - 1, vmax=vmax + 1)
                
                for t in texts:
                    old_text = t.get_text()
                    m = re.search(r'value\s*=\s*\[*\s*([-\d\.]+)', old_text)
                    if m:
                        val = float(m.group(1))
                        color = cmap(norm(val))
                        # Alpha para que el texto sea legible
                        color_with_alpha = (color[0], color[1], color[2], 0.7)
                        patch = t.get_bbox_patch()
                        patch.set_facecolor(color_with_alpha)
                        patch.set_fill(True)
                        
                        # Simplificamos el texto para no incluir el arreglo gigante de 'value'
                        new_text = re.sub(r'\n*value\s*=\s*\[[^\]]+\]', f'\nriesgo = {val:.4g}', old_text, flags=re.DOTALL)
                        t.set_text(new_text)

        total_samples = estimator.tree_.n_node_samples[0]
        ax.set_title(
            f"Árbol #{tree_idx} de {n_trees}  |  "
            f"hojas={n_leaves}  |  casos raíz={total_samples}  |  "
            f"prof. mostrada={max_depth}{_fallback_note}",
            fontsize=10,
        )

        self._apply_plot_text_overrides(ax, "tree")
        self.tree_fig.tight_layout()
        self.tree_canvas.draw()
        if hasattr(self, '_update_tree_scroll'): self._update_tree_scroll()

    def plot_survival_profiles(self):
        self.profile_fig.clear()
        ax = self.profile_fig.add_subplot(111)

        if self.model is None or self.latest_fit_dataframe is None or not self.latest_covariates:
            ax.text(0.5, 0.5, "Ajusta un RSF para ver el efecto parcial de una covariable.", ha="center", va="center")
            self._apply_plot_text_overrides(ax, "profile")
            self.profile_canvas.draw()
            return

        try:
            available_covariates = self._get_plot_covariate_candidates()
            covariate = self.profile_covariate_var.get().strip() if hasattr(self, 'profile_covariate_var') else ''
            if covariate not in available_covariates:
                covariate = available_covariates[0] if available_covariates else None
                if covariate and hasattr(self, 'profile_covariate_var'):
                    self.profile_covariate_var.set(covariate)

            if not covariate:
                raise ValueError("No se encontró covariable válida para graficar.")

            data = self.latest_fit_dataframe
            cov_series = data[covariate].dropna()
            if cov_series.empty:
                raise ValueError("La covariable seleccionada no tiene valores válidos.")

            manual_values_text = self.profile_values_var.get() if hasattr(self, 'profile_values_var') else ''
            manual_values = self._parse_plot_values(covariate, manual_values_text)
            if manual_values:
                values = manual_values
            elif not self._is_covariate_categorical(covariate):
                # Continuous variable: show P10, P50, P90
                numeric_values = pd.to_numeric(cov_series, errors='coerce').dropna()
                if numeric_values.empty:
                    raise ValueError("No hay valores numéricos válidos para la covariable seleccionada.")
                if numeric_values.nunique() > 4:
                    values = [
                        float(np.nanpercentile(numeric_values, 10)),
                        float(np.nanpercentile(numeric_values, 50)),
                        float(np.nanpercentile(numeric_values, 90)),
                    ]
                else:
                    values = sorted(numeric_values.unique().tolist())
            else:
                # Categorical variable: use original labels
                values = self._get_plot_categorical_values(covariate, cov_series)

            if not values:
                raise ValueError("No se pudieron determinar valores para la covariable seleccionada.")

            baseline_text = self.profile_baseline_var.get() if hasattr(self, 'profile_baseline_var') else ''
            baseline_overrides = self._parse_plot_baseline_overrides(baseline_text, exclude_covariate=covariate)
            base_row = self._build_plot_reference_row(focal_covariate=covariate, overrides=baseline_overrides)

            predict_rows = []
            labels = []
            for value in values:
                row = base_row.copy()
                row[covariate] = value
                predict_rows.append(row)
                labels.append(self._format_plot_value_label(value))

            predict_df = pd.DataFrame(predict_rows, columns=self.latest_covariates)
            predict_encoded = self._encode_prediction_frame(predict_df)
            curve_payload = self._compute_survival_curve_confidence_bands(self.model, predict_encoded)
            if not curve_payload:
                raise ValueError("No se pudieron obtener curvas de supervivencia para la covariable seleccionada.")

            times = np.asarray(curve_payload["times"], dtype=float)
            point_estimate = np.asarray(curve_payload["point_estimate"], dtype=float)
            lower = np.asarray(curve_payload["lower"], dtype=float) if curve_payload.get("lower") is not None else None
            upper = np.asarray(curve_payload["upper"], dtype=float) if curve_payload.get("upper") is not None else None
            ci_available = bool(curve_payload.get("ci_available") and lower is not None and upper is not None)
            show_ci = bool(self.profile_show_ci_var.get()) if hasattr(self, 'profile_show_ci_var') else True
            lower_band = lower if lower is not None else point_estimate
            upper_band = upper if upper is not None else point_estimate

            for idx, label_text in enumerate(labels):
                y_values = point_estimate[idx]
                ax.step(times, y_values, where="post", linewidth=2, label=f"{covariate}: {label_text}")
                if show_ci and ci_available:
                    ax.fill_between(
                        times,
                        lower_band[idx],
                        upper_band[idx],
                        step="post",
                        alpha=0.12,
                    )

            ax.set_title(f"Efecto de '{covariate}' sobre la supervivencia (RSF)")
            ax.set_xlabel("Tiempo")
            ax.set_ylabel("Probabilidad de supervivencia")
            ax.set_ylim(0, 1.05)
            ax.grid(True, alpha=0.2)
            if show_ci and ci_available:
                ax.text(
                    0.98,
                    0.02,
                    "Bandas = IC 95% aprox.",
                    transform=ax.transAxes,
                    ha="right",
                    va="bottom",
                    fontsize=8,
                    color="#334155",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="#cbd5e1"),
                )
            legend_ncols = 1
            if len(labels) >= 10:
                legend_ncols = 3
            elif len(labels) >= 6:
                legend_ncols = 2
            ax.legend(loc="best", fontsize=8, ncol=legend_ncols)

            if baseline_overrides:
                overrides_text = ', '.join(f"{key}: {self._format_plot_value_label(val)}" for key, val in baseline_overrides.items())
                ax.text(
                    0.02,
                    0.02,
                    f"Otras vars fijas: {overrides_text}",
                    transform=ax.transAxes,
                    ha='left',
                    va='bottom',
                    fontsize=8,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.75, edgecolor='#bbbbbb'),
                )
        except Exception as exc:
            ax.text(0.5, 0.5, f"No se pudo generar el efecto parcial:\n{exc}", ha="center", va="center")

        self._apply_plot_text_overrides(ax, "profile")
        self.profile_fig.tight_layout()
        self.profile_canvas.draw()

    def plot_variable_impact(self):
        self.impact_fig.clear()
        ax = self.impact_fig.add_subplot(111)

        if self.model is None or self.latest_fit_dataframe is None or not self.latest_covariates:
            ax.text(0.5, 0.5, "Ajusta un RSF para ver el impacto de una covariable en un tiempo t.", ha="center", va="center")
            self._apply_plot_text_overrides(ax, "impact")
            self.impact_canvas.draw()
            return

        try:
            available_covariates = self._get_plot_covariate_candidates()
            covariate = self.impact_covariate_var.get().strip() if hasattr(self, 'impact_covariate_var') else ''
            if covariate not in available_covariates:
                covariate = available_covariates[0] if available_covariates else None
                if covariate and hasattr(self, 'impact_covariate_var'):
                    self.impact_covariate_var.set(covariate)

            if not covariate:
                raise ValueError("No se encontró una covariable para evaluar el impacto.")

            data = self.latest_fit_dataframe
            cov_series = data[covariate].dropna()
            if cov_series.empty:
                raise ValueError("La covariable seleccionada no tiene valores disponibles.")

            raw_time_text = self.impact_time_var.get() if hasattr(self, 'impact_time_var') else ''
            all_times_mode = bool(self.impact_all_times_var.get()) if hasattr(self, 'impact_all_times_var') else False

            if all_times_mode:
                # Build a dense time grid from the model's event times
                duration_col = self.latest_duration_col
                event_col = self.latest_event_col
                if duration_col and event_col and duration_col in data.columns and event_col in data.columns:
                    all_event_times = np.asarray(
                        pd.to_numeric(
                            data.loc[pd.to_numeric(data[event_col], errors='coerce').fillna(0) > 0, duration_col],
                            errors='coerce',
                        ).dropna(),
                        dtype=float,
                    )
                    if all_event_times.size == 0:
                        all_event_times = np.asarray(
                            pd.to_numeric(data[duration_col], errors='coerce').dropna(),
                            dtype=float,
                        )
                else:
                    all_event_times = np.array([], dtype=float)

                if all_event_times.size >= 2:
                    t_min = float(np.nanpercentile(all_event_times, 5))
                    t_max = float(np.nanpercentile(all_event_times, 95))
                    n_steps = max(8, min(20, int(all_event_times.size // 3)))
                    eval_times = list(np.unique(np.linspace(t_min, t_max, num=n_steps)))
                else:
                    eval_times = [float(np.nanmedian(pd.to_numeric(data[self.latest_duration_col], errors='coerce')))]
            else:
                eval_times = self._parse_time_points_input(raw_time_text)

            if not eval_times:
                default_time = float(np.nanmedian(pd.to_numeric(data[self.latest_duration_col], errors='coerce')))
                if not np.isfinite(default_time) or default_time <= 0:
                    default_time = 1.0
                eval_times = [default_time]
                if hasattr(self, 'impact_time_var') and not str(raw_time_text).strip():
                    self.impact_time_var.set(f"{default_time:.2f}")

            manual_values_text = self.impact_values_var.get() if hasattr(self, 'impact_values_var') else ''
            manual_values = self._parse_plot_values(covariate, manual_values_text)

            is_numeric = not self._is_covariate_categorical(covariate)
            if manual_values:
                values = manual_values
            elif is_numeric:
                numeric_values = pd.to_numeric(cov_series, errors='coerce').dropna()
                values = np.linspace(float(numeric_values.min()), float(numeric_values.max()), num=60)
                values = np.unique(values)
            else:
                values = self._get_plot_categorical_values(covariate, cov_series)

            baseline_text = self.impact_baseline_var.get() if hasattr(self, 'impact_baseline_var') else ''
            baseline_overrides = self._parse_plot_baseline_overrides(baseline_text, exclude_covariate=covariate)
            base_row = self._build_plot_reference_row(focal_covariate=covariate, overrides=baseline_overrides)

            predict_rows = []
            labels = []
            for value in values:
                row = base_row.copy()
                row[covariate] = value
                predict_rows.append(row)
                labels.append(value)

            predict_df = pd.DataFrame(predict_rows, columns=self.latest_covariates)
            predict_encoded = self._encode_prediction_frame(predict_df)
            curve_payload = self._compute_survival_curve_confidence_bands(self.model, predict_encoded)
            if not curve_payload:
                raise ValueError("No se pudieron calcular los riesgos a partir del modelo RSF.")

            show_ci = bool(self.impact_show_ci_var.get()) if hasattr(self, 'impact_show_ci_var') else True
            all_times_mode = bool(self.impact_all_times_var.get()) if hasattr(self, 'impact_all_times_var') else False
            plotted_time_labels = []
            any_ci_available = False
            max_risk_value = 0.0
            time_style_map = self._get_fixed_time_palette(eval_times)

            # Build gradient palette for all-times mode
            import matplotlib.cm as _cm
            import matplotlib.colors as _mcolors
            _gradient_color = None  # type: ignore[assignment]
            if all_times_mode and len(eval_times) >= 2:
                _cmap = _cm.get_cmap("coolwarm")
                _norm = _mcolors.Normalize(vmin=float(eval_times[0]), vmax=float(eval_times[-1]))
                def _gradient_color(t):  # type: ignore[assignment]
                    return _cmap(_norm(float(t)))

            if is_numeric:
                x_values = np.asarray(values, dtype=float)
                order = np.argsort(x_values)
                x_sorted = x_values[order]

                for idx_time, requested_time in enumerate(eval_times):
                    risk_payload = self._extract_curve_values_at_time(curve_payload, requested_time, output_type="risk")
                    if not risk_payload:
                        continue

                    eval_time_clipped = float(risk_payload["eval_time"])
                    plotted_time_labels.append(eval_time_clipped)
                    risk_scores = np.asarray(risk_payload["point_estimate"], dtype=float)
                    risk_lower = np.asarray(risk_payload["lower"], dtype=float) if risk_payload.get("lower") is not None else None
                    risk_upper = np.asarray(risk_payload["upper"], dtype=float) if risk_payload.get("upper") is not None else None
                    ci_available = bool(risk_payload.get("ci_available") and risk_lower is not None and risk_upper is not None)
                    any_ci_available = any_ci_available or ci_available
                    risk_lower_band = risk_lower if risk_lower is not None else risk_scores
                    risk_upper_band = risk_upper if risk_upper is not None else risk_scores
                    risk_sorted = risk_scores[order]
                    max_risk_value = max(max_risk_value, float(np.nanmax(risk_sorted)))

                    if _gradient_color is not None:
                        color = _gradient_color(requested_time)
                        line_style = "-"
                        marker_style = None
                        line_label = None  # labels go on the right side as text
                    else:
                        line_label = f"t={eval_time_clipped:.2f}" if len(eval_times) > 1 else "Riesgo estimado"
                        style_info = time_style_map.get(float(requested_time), {})
                        color = style_info.get("color", "#1f77b4")
                        line_style = style_info.get("line_style", "-")
                        marker_style = style_info.get("marker", "o")

                    mark_step = max(1, len(x_sorted) // 8)
                    ax.plot(
                        x_sorted,
                        risk_sorted,
                        color=color,
                        linewidth=2.0 if _gradient_color is not None else 2.2,
                        linestyle=line_style,
                        marker=marker_style if (len(eval_times) > 1 and _gradient_color is None) else None,
                        markevery=mark_step if (len(eval_times) > 1 and _gradient_color is None) else None,
                        markersize=4.5,
                        alpha=0.88 if _gradient_color is not None else 1.0,
                        label=line_label,
                    )
                    if show_ci and ci_available:
                        ax.fill_between(
                            x_sorted,
                            risk_lower_band[order],
                            risk_upper_band[order],
                            color=color,
                            alpha=0.10 if _gradient_color is not None else 0.14,
                        )
                    elif show_ci and len(eval_times) == 1 and not ci_available:
                        ax.fill_between(x_sorted, risk_sorted, color=color, alpha=0.12)

                    # Left-side label for gradient mode
                    if _gradient_color is not None and len(x_sorted) > 0:
                        first_x = float(x_sorted[0])
                        first_y = float(risk_sorted[0])
                        ax.annotate(
                            f"t={eval_time_clipped:.1f}",
                            xy=(first_x, first_y),
                            xytext=(-6, 0),
                            textcoords="offset points",
                            fontsize=7,
                            color=color,
                            va="center",
                            ha="right",
                            clip_on=False,
                        )

                # Colorbar for gradient mode
                if _gradient_color is not None and len(eval_times) >= 2:
                    import matplotlib.cm as _cm2
                    import matplotlib.colors as _mc2
                    _sm = _cm2.ScalarMappable(
                        cmap=_cm2.get_cmap("coolwarm"),
                        norm=_mc2.Normalize(vmin=float(eval_times[0]), vmax=float(eval_times[-1])),
                    )
                    _sm.set_array([])
                    _cbar = self.impact_fig.colorbar(_sm, ax=ax, fraction=0.035, pad=0.01)
                    _cbar.set_label("Tiempo t", fontsize=8)
                    _cbar.ax.tick_params(labelsize=7)

                ax.set_xlabel(covariate)
            else:
                x_positions = np.arange(len(labels), dtype=float)
                width = min(0.35, 0.8 / max(1, len(eval_times)))
                offsets = (np.arange(len(eval_times), dtype=float) - ((len(eval_times) - 1) / 2.0)) * width

                for idx_time, requested_time in enumerate(eval_times):
                    risk_payload = self._extract_curve_values_at_time(curve_payload, requested_time, output_type="risk")
                    if not risk_payload:
                        continue

                    eval_time_clipped = float(risk_payload["eval_time"])
                    plotted_time_labels.append(eval_time_clipped)
                    risk_scores = np.asarray(risk_payload["point_estimate"], dtype=float)
                    risk_lower = np.asarray(risk_payload["lower"], dtype=float) if risk_payload.get("lower") is not None else None
                    risk_upper = np.asarray(risk_payload["upper"], dtype=float) if risk_payload.get("upper") is not None else None
                    ci_available = bool(risk_payload.get("ci_available") and risk_lower is not None and risk_upper is not None)
                    any_ci_available = any_ci_available or ci_available
                    risk_lower_band = risk_lower if risk_lower is not None else risk_scores
                    risk_upper_band = risk_upper if risk_upper is not None else risk_scores
                    max_risk_value = max(max_risk_value, float(np.nanmax(risk_scores)))
                    yerr = self._build_asymmetric_errorbars(risk_scores, risk_lower_band, risk_upper_band) if (show_ci and ci_available) else None

                    style_info = time_style_map.get(float(requested_time), {})
                    ax.bar(
                        x_positions + offsets[idx_time],
                        risk_scores,
                        width=width,
                        color=style_info.get("color", "#1f77b4"),
                        edgecolor="#334155",
                        linewidth=0.7,
                        hatch=style_info.get("hatch", '') if len(eval_times) > 1 else '',
                        alpha=0.82,
                        yerr=yerr,
                        error_kw={"ecolor": "#7f1d1d", "elinewidth": 1.0, "capsize": 4, "alpha": 0.9},
                        label=(f"t={eval_time_clipped:.2f}" if len(eval_times) > 1 else "Riesgo estimado"),
                    )

                label_rotation = 15
                if len(labels) >= 14:
                    label_rotation = 45
                elif len(labels) >= 9:
                    label_rotation = 30
                ax.set_xticks(x_positions)
                ax.set_xticklabels([str(self._format_plot_value_label(val)) for val in labels], rotation=label_rotation)
                ax.set_xlabel(covariate)

            if not plotted_time_labels:
                raise ValueError("No se pudieron graficar los tiempos solicitados.")

            time_label_text = ", ".join(f"{time_val:.2f}" for time_val in plotted_time_labels)
            ax.set_ylabel("Riesgo estimado (1 - S(t))")
            if all_times_mode:
                t_lo = min(plotted_time_labels)
                t_hi = max(plotted_time_labels)
                ax.set_title(
                    f"Impacto de '{covariate}' sobre el riesgo — todos los tiempos "
                    f"(t={t_lo:.1f} a t={t_hi:.1f}, n={len(plotted_time_labels)} curvas)"
                )
            else:
                ax.set_title(f"Impacto de '{covariate}' sobre el riesgo para t={time_label_text}")
            ax.set_ylim(0, min(1.05, max(1.0, max_risk_value + 0.05)))
            ax.grid(True, alpha=0.2)
            if all_times_mode:
                # Left-margin space for annotations
                try:
                    x_lims = ax.get_xlim()
                    x_range = x_lims[1] - x_lims[0]
                    ax.set_xlim(x_lims[0] - x_range * 0.10, x_lims[1])
                except Exception:
                    pass
            elif len(plotted_time_labels) > 1:
                ax.text(
                    0.02,
                    0.02,
                    "Cada color/estilo representa un tiempo distinto",
                    transform=ax.transAxes,
                    ha="left",
                    va="bottom",
                    fontsize=8,
                    color="#334155",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.78, edgecolor="#cbd5e1"),
                )
            if show_ci and any_ci_available:
                ax.text(
                    0.98,
                    0.02,
                    "IC 95% aprox. visible",
                    transform=ax.transAxes,
                    ha="right",
                    va="bottom",
                    fontsize=8,
                    color="#334155",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="#cbd5e1"),
                )
            if not all_times_mode:
                handles, legend_labels = ax.get_legend_handles_labels()
                if legend_labels:
                    legend_title = "Tiempo evaluado" if len(plotted_time_labels) > 1 else None
                    ax.legend(loc="best", fontsize=8, title=legend_title, title_fontsize=9, ncol=(2 if len(plotted_time_labels) >= 4 else 1))

            if baseline_overrides:
                overrides_text = ', '.join(f"{key}: {self._format_plot_value_label(val)}" for key, val in baseline_overrides.items())
                ax.text(
                    0.02,
                    0.98,
                    f"Otras vars fijas: {overrides_text}",
                    transform=ax.transAxes,
                    ha='left',
                    va='top',
                    fontsize=8,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.75, edgecolor='#bbbbbb'),
                )
        except Exception as exc:
            ax.text(0.5, 0.5, f"No se pudo calcular el impacto de la covariable:\n{exc}", ha="center", va="center")

        self._apply_plot_text_overrides(ax, "impact")
        self.impact_fig.tight_layout()
        self.impact_canvas.draw()

    # ------------------------------------------------------------------
    # Editable plot text (aligned with AFT behavior)
    # ------------------------------------------------------------------
    def _register_editable_plot_canvas(self, canvas, plot_key):
        if canvas is None:
            return
        setattr(canvas, "_matabs_plot_key", plot_key)
        if getattr(canvas, "_matabs_text_edit_cid", None) is not None:
            return
        cid = canvas.mpl_connect(
            "button_press_event",
            lambda event, current_canvas=canvas: self._on_plot_text_click(event, current_canvas),
        )
        setattr(canvas, "_matabs_text_edit_cid", cid)

    def _iter_editable_plot_texts(self, ax):
        if ax is None:
            return []
        editable_items = [
            ("title", ax.title, "título"),
            ("xlabel", ax.xaxis.label, "eje X"),
            ("ylabel", ax.yaxis.label, "eje Y"),
        ]
        for idx, tick_text in enumerate(ax.get_xticklabels()):
            if str(tick_text.get_text()).strip():
                editable_items.append((f"xtick_label_{idx}", tick_text, f"categoría eje X {idx + 1}"))
        for idx, tick_text in enumerate(ax.get_yticklabels()):
            if str(tick_text.get_text()).strip():
                editable_items.append((f"ytick_label_{idx}", tick_text, f"categoría eje Y {idx + 1}"))
        legend = ax.get_legend()
        if legend is not None:
            legend_title = legend.get_title()
            if legend_title is not None and str(legend_title.get_text()).strip():
                editable_items.append(("legend_title", legend_title, "título de la leyenda"))
            for idx, legend_text in enumerate(legend.get_texts()):
                editable_items.append((f"legend_label_{idx}", legend_text, f"leyenda {idx + 1}"))
        return editable_items

    def _text_contains_click(self, text_artist, event, renderer):
        if text_artist is None or renderer is None or event is None or event.x is None or event.y is None:
            return False
        current_text = text_artist.get_text() if hasattr(text_artist, "get_text") else ""
        if not str(current_text).strip():
            return False
        try:
            bbox = text_artist.get_window_extent(renderer=renderer)
        except Exception:
            return False
        if bbox is None or bbox.width <= 0 or bbox.height <= 0:
            return False
        return bbox.expanded(1.15, 1.25).contains(event.x, event.y)

    def _update_plot_text_override(self, plot_key, target_key, new_text, ax=None):
        if not plot_key:
            return
        overrides = self._plot_text_overrides.setdefault(plot_key, {})
        if target_key in {"title", "xlabel", "ylabel", "legend_title"}:
            overrides[target_key] = new_text
            return

        target_mappings = {
            "legend_label_": ("legend_labels", (lambda current_ax: [txt.get_text() for txt in current_ax.get_legend().get_texts()]) if ax is not None and ax.get_legend() is not None else None),
            "xtick_label_": ("xtick_labels", (lambda current_ax: [txt.get_text() for txt in current_ax.get_xticklabels()]) if ax is not None else None),
            "ytick_label_": ("ytick_labels", (lambda current_ax: [txt.get_text() for txt in current_ax.get_yticklabels()]) if ax is not None else None),
        }
        for prefix, (storage_key, extractor) in target_mappings.items():
            if not (isinstance(target_key, str) and target_key.startswith(prefix)):
                continue
            try:
                label_index = int(target_key.rsplit("_", 1)[1])
            except Exception:
                return
            current_labels = extractor(ax) if callable(extractor) and ax is not None else []
            stored_labels = list(overrides.get(storage_key, current_labels))
            while len(stored_labels) <= label_index:
                stored_labels.append("")
            stored_labels[label_index] = new_text
            overrides[storage_key] = stored_labels
            return

    def _apply_plot_text_overrides(self, ax, plot_key):
        if ax is None:
            return
        overrides = self._plot_text_overrides.get(plot_key, {}) if isinstance(self._plot_text_overrides, dict) else {}
        if not overrides:
            return
        if "title" in overrides:
            ax.set_title(overrides["title"])
        if "xlabel" in overrides:
            ax.set_xlabel(overrides["xlabel"])
        if "ylabel" in overrides:
            ax.set_ylabel(overrides["ylabel"])

        if isinstance(overrides.get("xtick_labels"), list):
            xticks = ax.get_xticks()
            current_xticklabels = ax.get_xticklabels()
            x_rotation = current_xticklabels[0].get_rotation() if current_xticklabels else 0
            merged_xlabels = [tick.get_text() for tick in current_xticklabels]
            if len(merged_xlabels) < len(xticks):
                merged_xlabels.extend([""] * (len(xticks) - len(merged_xlabels)))
            for idx, new_label in enumerate(overrides["xtick_labels"]):
                if idx < len(merged_xlabels):
                    merged_xlabels[idx] = str(new_label)
            if len(xticks) == len(merged_xlabels):
                ax.set_xticks(xticks)
                ax.set_xticklabels(merged_xlabels, rotation=x_rotation)

        if isinstance(overrides.get("ytick_labels"), list):
            yticks = ax.get_yticks()
            current_yticklabels = ax.get_yticklabels()
            y_rotation = current_yticklabels[0].get_rotation() if current_yticklabels else 0
            merged_ylabels = [tick.get_text() for tick in current_yticklabels]
            if len(merged_ylabels) < len(yticks):
                merged_ylabels.extend([""] * (len(yticks) - len(merged_ylabels)))
            for idx, new_label in enumerate(overrides["ytick_labels"]):
                if idx < len(merged_ylabels):
                    merged_ylabels[idx] = str(new_label)
            if len(yticks) == len(merged_ylabels):
                ax.set_yticks(yticks)
                ax.set_yticklabels(merged_ylabels, rotation=y_rotation)

        legend = ax.get_legend()
        if legend is not None:
            if "legend_title" in overrides:
                legend.set_title(overrides["legend_title"])
            if isinstance(overrides.get("legend_labels"), list):
                for text_obj, new_label in zip(legend.get_texts(), overrides["legend_labels"]):
                    text_obj.set_text(str(new_label))

    def _on_plot_text_click(self, event, canvas):
        if event is None or canvas is None:
            return
        toolbar = getattr(canvas, "toolbar", None)
        if toolbar is not None and getattr(toolbar, "mode", ""):
            return
        figure = getattr(canvas, "figure", None)
        if figure is None:
            return
        try:
            renderer = canvas.get_renderer()
        except Exception:
            renderer = getattr(figure.canvas, "get_renderer", lambda: None)()
        plot_key = getattr(canvas, "_matabs_plot_key", None)
        parent_window = self.winfo_toplevel() if hasattr(self, "winfo_toplevel") else None
        for ax in figure.axes:
            for target_key, text_artist, label_name in self._iter_editable_plot_texts(ax):
                if not self._text_contains_click(text_artist, event, renderer):
                    continue
                current_text = text_artist.get_text() if hasattr(text_artist, "get_text") else ""
                new_text = simpledialog.askstring(
                    "Editar texto del gráfico",
                    f"Nuevo texto para {label_name}:",
                    initialvalue=current_text,
                    parent=parent_window,
                )
                if new_text is None:
                    return
                text_artist.set_text(new_text)
                self._update_plot_text_override(plot_key, target_key, new_text, ax=ax)
                canvas.draw_idle()
                return

    # ------------------------------------------------------------------
    # Reset state
    # ------------------------------------------------------------------
    def _reset_results_view(self):
        self.model = None
        self.results = None
        self.feature_importance_df = pd.DataFrame()
        self.latest_prediction_df = None
        self.latest_survival_profiles = []
        self.latest_calibration_df = pd.DataFrame()
        self.latest_brier_df = pd.DataFrame()
        self.latest_eval_time = None
        self.latest_fit_dataframe = None
        self.latest_duration_col = None
        self.latest_event_col = None
        self.latest_covariates = []
        self.latest_encoded_columns = []
        self.latest_report_text = ""
        self.latest_tuning_summary = ""
        if hasattr(self, "results_text"):
            self.results_text.delete("1.0", tk.END)
        for string_attr in (
            'profile_covariate_var', 'profile_values_var', 'profile_baseline_var',
            'impact_covariate_var', 'impact_values_var', 'impact_baseline_var', 'impact_time_var'
        ):
            if hasattr(self, string_attr):
                getattr(self, string_attr).set('')

        for fig_attr, canvas_attr in (
            ("importance_fig", "importance_canvas"),
            ("km_fig", "km_canvas"),
            ("profile_fig", "profile_canvas"),
            ("impact_fig", "impact_canvas"),
            ("calibration_fig", "calibration_canvas"),
            ("brier_fig", "brier_canvas"),
        ):
            if hasattr(self, fig_attr) and hasattr(self, canvas_attr):
                fig = getattr(self, fig_attr)
                canvas = getattr(self, canvas_attr)
                fig.clear()
                canvas.draw()

class RSFCategoricalConfigDialog(tk.Toplevel):
    def __init__(self, parent, app_instance, selected_vars):
        super().__init__(parent)
        self.transient(parent)
        self.grab_set()
        self.title("Configuración categórica RSF")
        self.app_instance = app_instance
        self.selected_vars = list(selected_vars)
        self.row_configs = {}

        main_frame = ttk.Frame(self, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(
            main_frame,
            text="Elige la categoría de referencia o una dicotomía elegida vs resto para cada variable.",
            foreground="#666666",
        ).pack(anchor="w", pady=(0, 8))

        for var_name in self.selected_vars:
            row_frame = ttk.LabelFrame(main_frame, text=var_name, padding="8")
            row_frame.pack(fill=tk.X, expand=True, pady=4)
            row_frame.columnconfigure(1, weight=1)

            current_config = self.app_instance.variable_configs.get(var_name, {})
            if self.app_instance.data is not None and var_name in self.app_instance.data.columns:
                unique_values = sorted(self.app_instance.data[var_name].dropna().astype(str).unique().tolist())
            else:
                unique_values = []

            default_compare_mode = current_config.get("compare_mode")
            if not default_compare_mode:
                default_compare_mode = "quantitative" if (
                    self.app_instance.data is not None
                    and var_name in self.app_instance.data.columns
                    and pd.api.types.is_numeric_dtype(self.app_instance.data[var_name])
                ) else "all"
            compare_mode_var = tk.StringVar(
                value=self.app_instance._get_categorical_compare_display_value(default_compare_mode)
            )
            category_var = tk.StringVar(value=str(current_config.get("selected_cat", current_config.get("ref_cat", unique_values[0] if unique_values else ""))))

            ttk.Label(row_frame, text="Modo:").grid(row=0, column=0, padx=5, pady=2, sticky="w")
            ttk.Combobox(
                row_frame,
                textvariable=compare_mode_var,
                values=list(self.app_instance.categorical_compare_display_map.values()),
                state="readonly",
                width=28,
            ).grid(row=0, column=1, padx=5, pady=2, sticky="ew")

            ttk.Label(row_frame, text="Categoría elegida:").grid(row=1, column=0, padx=5, pady=2, sticky="w")
            ttk.Combobox(
                row_frame,
                textvariable=category_var,
                values=unique_values,
                state="readonly",
                width=24,
            ).grid(row=1, column=1, padx=5, pady=2, sticky="ew")

            ttk.Label(
                row_frame,
                text="• 'Cada categoría vs referencia' usa esta categoría como base si está activo drop_first.\n"
                     "• 'Dicotómica' convierte la variable en elegida vs todas las demás.\n"
                     "• 'Mantener como cuantitativa' evita forzar variables numéricas con pocos niveles como categóricas.",
                foreground="#666666",
            ).grid(row=2, column=0, columnspan=2, padx=5, pady=(2, 0), sticky="w")

            self.row_configs[var_name] = {
                "compare_mode_var": compare_mode_var,
                "category_var": category_var,
            }

        buttons_frame = ttk.Frame(main_frame)
        buttons_frame.pack(fill=tk.X, pady=(10, 0))
        ttk.Button(buttons_frame, text="Aplicar", command=self.apply_configurations).pack(side=tk.RIGHT, padx=5)
        ttk.Button(buttons_frame, text="Cancelar", command=self.destroy).pack(side=tk.RIGHT)

    def apply_configurations(self):
        for var_name, row_config in self.row_configs.items():
            category_value = row_config["category_var"].get().strip()
            compare_mode = self.app_instance._get_categorical_compare_internal_mode(row_config["compare_mode_var"].get())
            self.app_instance.variable_configs[var_name] = {
                "compare_mode": compare_mode,
                "treat_as": ("quantitative" if compare_mode == "quantitative" else "categorical"),
                "ref_cat": category_value,
                "selected_cat": category_value,
            }
        self.destroy()


