import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog, StringVar, BooleanVar, DoubleVar, IntVar, Listbox, MULTIPLE, SINGLE, BROWSE, Toplevel, Frame, Label, Entry, Button, Checkbutton, Radiobutton
from tkinter import scrolledtext
import pandas as pd
import numpy as np
from lifelines import WeibullAFTFitter, LogNormalAFTFitter, LogLogisticAFTFitter, KaplanMeierFitter, NelsonAalenFitter
from lifelines.utils import datetimes_to_durations, concordance_index
from sklearn.model_selection import train_test_split

# --- Uno's C-index (IPCW) + Antolini Ctd + Brier from scikit-survival ---
try:
    from sksurv.metrics import concordance_index_ipcw as _concordance_index_ipcw_aft
    from sksurv.metrics import cumulative_dynamic_auc as _cumulative_dynamic_auc_aft
    from sksurv.metrics import brier_score as _brier_score_aft
    from sksurv.metrics import integrated_brier_score as _integrated_brier_score_aft
except Exception:
    _concordance_index_ipcw_aft = None
    _cumulative_dynamic_auc_aft = None
    _brier_score_aft = None
    _integrated_brier_score_aft = None
import os
import json
import re
import copy
from statistics import NormalDist
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from MATLAB_filter_component import FilterComponent

class ScrolledFrame(ttk.Frame):
    def __init__(self, parent, *args, **kw):
        super().__init__(parent, *args, **kw)
        self.canvas = tk.Canvas(self, highlightthickness=0, bd=0)
        self.interior = ttk.Frame(self.canvas)
        self.v_scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.h_scrollbar = ttk.Scrollbar(self, orient="horizontal", command=self.canvas.xview)
        self.canvas.configure(yscrollcommand=self.v_scrollbar.set, xscrollcommand=self.h_scrollbar.set)
        self.v_scrollbar.pack(side="right", fill="y")
        self.h_scrollbar.pack(side="bottom", fill="x")
        self.canvas.pack(side="left", fill="both", expand=True)
        self.interior_id = self.canvas.create_window(0, 0, window=self.interior, anchor="nw")
        self.interior.bind('<Configure>', self._on_interior_configure)
        self.canvas.bind('<Configure>', self._on_canvas_configure)
        self.canvas.bind('<Enter>', self._bind_mousewheel_events)
        self.canvas.bind('<Leave>', self._unbind_mousewheel_events)

    def _on_interior_configure(self, event=None):
        bbox = self.canvas.bbox("all")
        if bbox is not None:
            self.canvas.configure(scrollregion=bbox)
        interior_width = self.interior.winfo_reqwidth()
        canvas_width = self.canvas.winfo_width()
        target_width = max(interior_width, canvas_width)
        if target_width > 1:
            self.canvas.itemconfigure(self.interior_id, width=target_width)

    def _on_canvas_configure(self, event):
        interior_width = self.interior.winfo_reqwidth()
        target_width = max(event.width, interior_width)
        if target_width > 1:
            self.canvas.itemconfigure(self.interior_id, width=target_width)
        self._on_interior_configure()

    def _bind_mousewheel_events(self, event):
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind_all("<Button-4>", self._on_mousewheel)
        self.canvas.bind_all("<Button-5>", self._on_mousewheel)

    def _unbind_mousewheel_events(self, event):
        self.canvas.unbind_all("<MouseWheel>")
        self.canvas.unbind_all("<Button-4>")
        self.canvas.unbind_all("<Button-5>")

    def _on_mousewheel(self, event):
        if self.canvas.winfo_containing(event.x_root, event.y_root) == self.canvas:
            if event.num == 4:
                self.canvas.yview_scroll(-1, "units")
            elif event.num == 5:
                self.canvas.yview_scroll(1, "units")
            else:
                self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

class AFTsTab(ttk.Frame):
    def __init__(self, notebook, main_app_instance):
        super().__init__(notebook)
        self.main_app_instance = main_app_instance
        self.data = None
        self.base_data = None
        self.results = None
        self.fitter = None
        self.variable_configs = {}
        self.shared_metadata = {}
        self.current_shared_filter_summary = []
        self.using_shared_dataset = True
        self.latest_fit_dataframe = None
        self.latest_duration_col = None
        self.latest_event_col = None
        self.latest_covariates = []
        self.latest_formula = None
        self.latest_brier_df = pd.DataFrame()
        self.latest_eval_time = None
        self.saved_models = []
        self.active_saved_model_index = None
        self._plot_text_overrides = {}
        self.categorical_compare_display_map = {
            "all": "Todos los grupos vs referencia",
            "one_vs_rest": "Dicotómica: elegida vs resto",
        }
        self.categorical_compare_reverse_map = {
            display: internal for internal, display in self.categorical_compare_display_map.items()
        }
        self.spline_type_display_map = {
            "B-spline": "No restringido (B-spline)",
            "Natural": "Restringido (Natural)",
        }
        self.spline_type_reverse_map = {display: internal for internal, display in self.spline_type_display_map.items()}

        # --- UI ---
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # --- Preprocessing Tab ---
        self.preproc_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.preproc_tab, text="1. Preprocesamiento")

        # --- Modeling Tab ---
        self.modeling_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.modeling_tab, text="2. Modelado AFT")

        # --- Results Tab ---
        self.results_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.results_tab, text="3. Resultados")

        # --- Graphs Tab ---
        self.graphs_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.graphs_tab, text="4. Gráficas")

        self._create_preproc_widgets()
        self._create_modeling_widgets()
        self._create_results_widgets()
        self._create_graphs_widgets()

    def _create_preproc_widgets(self):
        # --- File Loading ---
        load_frame = ttk.LabelFrame(self.preproc_tab, text="Cargar Datos")
        load_frame.pack(fill=tk.X, padx=10, pady=10)

        ttk.Button(load_frame, text="Cargar Archivo (Excel/CSV)", command=self.load_data).pack(side=tk.LEFT, padx=5, pady=5)
        self.file_label = ttk.Label(load_frame, text="Ningún archivo cargado.")
        self.file_label.pack(side=tk.LEFT, padx=5, pady=5)

        # --- Filtering ---
        filter_frame = ttk.LabelFrame(self.preproc_tab, text="Filtros")
        filter_frame.pack(fill=tk.X, padx=10, pady=10)
        self.filter_component = FilterComponent(filter_frame)
        self.filter_component.pack(fill=tk.X, expand=True)

    def _create_modeling_widgets(self):
        # Scrollable container for the modeling tab
        self._modeling_scroll = ScrolledFrame(self.modeling_tab)
        self._modeling_scroll.pack(fill=tk.BOTH, expand=True)
        modeling_content = self._modeling_scroll.interior

        # --- Model Selection ---
        model_frame = ttk.LabelFrame(modeling_content, text="Configuración del Modelo AFT")
        model_frame.pack(fill=tk.X, padx=10, pady=10)
        model_frame.columnconfigure(1, weight=1)

        ttk.Label(model_frame, text="Modelo AFT:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.model_var = StringVar(value="Weibull")
        model_combo = ttk.Combobox(model_frame, textvariable=self.model_var, values=["Weibull", "Log-Normal", "Log-Logistic"], state="readonly")
        model_combo.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

        self.evaluate_all_models_var = BooleanVar(value=False)
        ttk.Checkbutton(
            model_frame,
            text="Evaluar los tres modelos AFT",
            variable=self.evaluate_all_models_var,
        ).grid(row=1, column=0, columnspan=2, padx=5, pady=(0, 5), sticky="w")

        # --- Holdout Train/Test ---
        holdout_frame = ttk.LabelFrame(modeling_content, text="Validaci\u00f3n Holdout (Train/Test)")
        holdout_frame.pack(fill=tk.X, padx=10, pady=(0, 5))
        holdout_frame.columnconfigure(1, weight=1)

        self.calculate_test_cindex_var = BooleanVar(value=False)
        ttk.Checkbutton(
            holdout_frame,
            text="Calcular C-Index en holdout (train/test)",
            variable=self.calculate_test_cindex_var,
        ).grid(row=0, column=0, columnspan=4, padx=5, pady=5, sticky="w")

        ttk.Label(holdout_frame, text="Proporci\u00f3n test:").grid(row=1, column=0, padx=5, pady=2, sticky="w")
        self.test_size_var = StringVar(value="0.20")
        ttk.Entry(holdout_frame, textvariable=self.test_size_var, width=8).grid(row=1, column=1, padx=5, pady=2, sticky="w")

        ttk.Label(holdout_frame, text="Semilla:").grid(row=1, column=2, padx=5, pady=2, sticky="w")
        self.test_random_seed_var = IntVar(value=42)
        ttk.Entry(holdout_frame, textvariable=self.test_random_seed_var, width=8).grid(row=1, column=3, padx=5, pady=2, sticky="w")

        ttk.Label(holdout_frame, text="τ (tau) IPCW:").grid(row=2, column=0, padx=5, pady=2, sticky="w")
        self.tau_mode_var = StringVar(value="Auto (P90)")
        ttk.Combobox(holdout_frame, textvariable=self.tau_mode_var,
                     values=["Auto (P90)", "Último evento", "Manual"],
                     state="readonly", width=16).grid(row=2, column=1, padx=5, pady=2, sticky="w")
        self.tau_manual_var = StringVar(value="")
        ttk.Entry(holdout_frame, textvariable=self.tau_manual_var, width=8).grid(row=2, column=2, padx=5, pady=2, sticky="w")
        ttk.Label(holdout_frame, text="Uno, Antolini, Brier/IBS", foreground="#555555").grid(row=2, column=3, padx=5, pady=2, sticky="w")

        # --- Variable Selection ---
        vars_frame = ttk.LabelFrame(modeling_content, text="Selección de Variables")
        vars_frame.pack(fill=tk.X, padx=10, pady=10)
        vars_frame.columnconfigure(1, weight=1)

        ttk.Label(vars_frame, text="Variable de Duración:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.duration_var = StringVar()
        self.duration_combo = ttk.Combobox(vars_frame, textvariable=self.duration_var, state="readonly")
        self.duration_combo.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

        ttk.Label(vars_frame, text="Variable de Evento (Opcional):").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.event_var = StringVar()
        self.event_combo = ttk.Combobox(vars_frame, textvariable=self.event_var, state="readonly")
        self.event_combo.grid(row=1, column=1, padx=5, pady=5, sticky="ew")

        ttk.Label(vars_frame, text="Covariables:").grid(row=2, column=0, padx=5, pady=5, sticky="w")
        self.covariates_listbox = tk.Listbox(vars_frame, selectmode=tk.MULTIPLE, height=6, exportselection=False)
        self.covariates_listbox.grid(row=2, column=1, padx=5, pady=5, sticky="ew")

        ttk.Button(vars_frame, text="Configurar Covariables...", command=self.open_variable_config_dialog).grid(row=3, column=1, padx=5, pady=10, sticky="e")

        ttk.Button(modeling_content, text="Ejecutar Modelo", command=self.run_model).pack(pady=(0, 10))

        history_frame = ttk.LabelFrame(modeling_content, text="Modelos AFT guardados para comparación")
        history_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        history_columns = ("id", "modelo", "parametros", "aic", "bic", "c_index", "c_index_test",
                           "c_uno", "c_antolini", "tau", "ibs",
                           "c_q25", "c_q50", "c_q75",
                           "brier_q25", "brier_q50", "brier_q75",
                           "auroc_q25", "auroc_q50", "auroc_q75",
                           "test_pct", "lr_stat", "lr_p", "rho", "rho_inv", "cs_slope", "cs_dist")
        self.saved_models_tree = ttk.Treeview(history_frame, columns=history_columns, show="headings", height=8, selectmode="browse")
        self._aft_tree_headings = {
            "id": "ID", "modelo": "Modelo", "parametros": "Parámetros", "aic": "AIC", "bic": "BIC",
            "c_index": "C-Index (Train)", "c_index_test": "C-Index (Test)",
            "c_uno": "C-Uno (IPCW)", "c_antolini": "C-Antolini (Ctd)", "tau": "τ (tau)", "ibs": "IBS",
            "c_q25": "C@Q25", "c_q50": "C@Q50", "c_q75": "C@Q75",
            "brier_q25": "Brier@Q25", "brier_q50": "Brier@Q50", "brier_q75": "Brier@Q75",
            "auroc_q25": "AUC@Q25", "auroc_q50": "AUC@Q50", "auroc_q75": "AUC@Q75",
            "test_pct": "Test %", "lr_stat": "LR chi2", "lr_p": "LR p",
            "rho": "Rho/Forma", "rho_inv": "1/Rho", "cs_slope": "Pend.CS", "cs_dist": "|CS-1|"
        }
        for col_id, heading_text in self._aft_tree_headings.items():
            self.saved_models_tree.heading(col_id, text=heading_text, command=lambda c=col_id: self._sort_saved_models_tree(c))

        self.saved_models_tree.column("id", width=60, anchor="center", stretch=False)
        self.saved_models_tree.column("modelo", width=120, anchor="center", stretch=False)
        self.saved_models_tree.column("parametros", width=320, anchor="w", stretch=True)
        self.saved_models_tree.column("aic", width=90, anchor="center", stretch=False)
        self.saved_models_tree.column("bic", width=90, anchor="center", stretch=False)
        self.saved_models_tree.column("c_index", width=165, anchor="center", stretch=False)
        self.saved_models_tree.column("c_index_test", width=165, anchor="center", stretch=False)
        self.saved_models_tree.column("c_uno", width=110, anchor="center", stretch=False)
        self.saved_models_tree.column("c_antolini", width=120, anchor="center", stretch=False)
        self.saved_models_tree.column("tau", width=80, anchor="center", stretch=False)
        self.saved_models_tree.column("ibs", width=80, anchor="center", stretch=False)
        self.saved_models_tree.column("c_q25", width=85, anchor="center", stretch=False)
        self.saved_models_tree.column("c_q50", width=85, anchor="center", stretch=False)
        self.saved_models_tree.column("c_q75", width=85, anchor="center", stretch=False)
        self.saved_models_tree.column("brier_q25", width=85, anchor="center", stretch=False)
        self.saved_models_tree.column("brier_q50", width=85, anchor="center", stretch=False)
        self.saved_models_tree.column("brier_q75", width=85, anchor="center", stretch=False)
        self.saved_models_tree.column("auroc_q25", width=85, anchor="center", stretch=False)
        self.saved_models_tree.column("auroc_q50", width=85, anchor="center", stretch=False)
        self.saved_models_tree.column("auroc_q75", width=85, anchor="center", stretch=False)
        self.saved_models_tree.column("test_pct", width=60, anchor="center", stretch=False)
        self.saved_models_tree.column("lr_stat", width=90, anchor="center", stretch=False)
        self.saved_models_tree.column("lr_p", width=80, anchor="center", stretch=False)
        self.saved_models_tree.column("rho", width=85, anchor="center", stretch=False)
        self.saved_models_tree.column("rho_inv", width=85, anchor="center", stretch=False)
        self.saved_models_tree.column("cs_slope", width=80, anchor="center", stretch=False)
        self.saved_models_tree.column("cs_dist", width=70, anchor="center", stretch=False)

        tree_y_scroll = ttk.Scrollbar(history_frame, orient="vertical", command=self.saved_models_tree.yview)
        tree_x_scroll = ttk.Scrollbar(history_frame, orient="horizontal", command=self.saved_models_tree.xview)
        self.saved_models_tree.configure(yscrollcommand=tree_y_scroll.set, xscrollcommand=tree_x_scroll.set)

        self.saved_models_tree.pack(fill=tk.BOTH, expand=True, side=tk.TOP)
        tree_y_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        tree_x_scroll.pack(side=tk.BOTTOM, fill=tk.X)
        self.saved_models_tree.bind("<Double-1>", self._load_selected_saved_model)
        self.saved_models_tree.bind("<Button-3>", self._show_aft_tree_column_menu)
        self._aft_tree_col_widths = {"id": 60, "modelo": 120, "parametros": 320, "aic": 90, "bic": 90,
                                      "c_index": 165, "c_index_test": 165,
                                      "c_uno": 110, "c_antolini": 120, "tau": 80, "ibs": 80,
                                      "c_q25": 85, "c_q50": 85, "c_q75": 85,
                                      "brier_q25": 85, "brier_q50": 85, "brier_q75": 85,
                                      "auroc_q25": 85, "auroc_q50": 85, "auroc_q75": 85,
                                      "test_pct": 60, "lr_stat": 90, "lr_p": 80,
                                      "rho": 85, "rho_inv": 85, "cs_slope": 80, "cs_dist": 70}
        self._aft_tree_col_config = {c: {"visible": True, "width": self._aft_tree_col_widths[c], "heading": self._aft_tree_headings[c]} for c in history_columns}
        self._restore_saved_layout("aft_saved_models", self._aft_tree_col_config)
        self._aft_tree_sort_reversed = {}
        self._apply_aft_tree_column_layout()
        self._register_saved_layout("aft_saved_models", self._aft_tree_col_config, self.saved_models_tree)

        history_buttons = ttk.Frame(history_frame)
        history_buttons.pack(fill=tk.X, pady=(8, 0))

        self.saved_models_status_var = StringVar(value="Modelos guardados: 0 | Activo: ninguno")
        ttk.Label(history_buttons, textvariable=self.saved_models_status_var, foreground="navy").pack(side=tk.LEFT, padx=5)
        ttk.Button(history_buttons, text="Cargar seleccionado", command=self._load_selected_saved_model).pack(side=tk.RIGHT, padx=5)
        ttk.Button(history_buttons, text="Eliminar seleccionado", command=self._delete_selected_saved_model).pack(side=tk.RIGHT, padx=5)
        ttk.Button(history_buttons, text="Limpiar lista", command=self._clear_saved_models).pack(side=tk.RIGHT, padx=5)

        self._refresh_saved_models_tree()

    def open_variable_config_dialog(self):
        selected_indices = self.covariates_listbox.curselection()
        if not selected_indices:
            messagebox.showwarning("Sin Selección", "Seleccione una o más covariables para configurar.")
            return
        
        selected_vars = [self.covariates_listbox.get(i) for i in selected_indices]
        VariableConfigDialog(self, self, selected_vars)

    def _create_results_widgets(self):
        self.results_text = tk.Text(self.results_tab, wrap=tk.WORD, height=20)
        self.results_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    def _create_graphs_widgets(self):
        ttk.Label(
            self.graphs_tab,
            text="Tip: haz clic en el título, nombres de ejes, categorías o leyenda para editar el texto.",
            foreground="#666666",
        ).pack(anchor="w", padx=12, pady=(6, 0))

        self.graphs_notebook = ttk.Notebook(self.graphs_tab)
        self.graphs_notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.survival_plot_tab = ttk.Frame(self.graphs_notebook)
        self.graphs_notebook.add(self.survival_plot_tab, text="Supervivencia: Observado vs Modelo")

        self.partial_effect_tab = ttk.Frame(self.graphs_notebook)
        self.graphs_notebook.add(self.partial_effect_tab, text="Efectos de Covariables")

        self.risk_effect_tab = ttk.Frame(self.graphs_notebook)
        self.graphs_notebook.add(self.risk_effect_tab, text="Riesgo vs Covariable")

        self.cumulative_hazard_tab = ttk.Frame(self.graphs_notebook)
        self.graphs_notebook.add(self.cumulative_hazard_tab, text="Validación Cox-Snell")

        self.forest_plot_tab = ttk.Frame(self.graphs_notebook)
        self.graphs_notebook.add(self.forest_plot_tab, text="Forest Plot")

        self.brier_tab = ttk.Frame(self.graphs_notebook)
        self.graphs_notebook.add(self.brier_tab, text="Brier / IBS")

        self.survival_fig = plt.figure(figsize=(6, 4))
        self.survival_canvas = FigureCanvasTkAgg(self.survival_fig, master=self.survival_plot_tab)
        self.survival_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self._register_editable_plot_canvas(self.survival_canvas, "survival")

        partial_controls = ttk.Frame(self.partial_effect_tab)
        partial_controls.pack(fill=tk.X, padx=8, pady=(8, 0))
        ttk.Label(partial_controls, text="Covariable:").pack(side=tk.LEFT, padx=(0, 6))
        self.partial_effect_var = StringVar()
        self.partial_effect_combo = ttk.Combobox(partial_controls, textvariable=self.partial_effect_var, state="readonly", width=22)
        self.partial_effect_combo.pack(side=tk.LEFT, padx=(0, 8))
        self.partial_effect_combo.bind("<<ComboboxSelected>>", lambda _event: self.plot_partial_effects())

        ttk.Label(partial_controls, text="Valores:").pack(side=tk.LEFT, padx=(6, 6))
        self.partial_effect_values_var = StringVar()
        self.partial_effect_values_entry = ttk.Entry(partial_controls, textvariable=self.partial_effect_values_var, width=18)
        self.partial_effect_values_entry.pack(side=tk.LEFT, padx=(0, 8))
        self.partial_effect_values_entry.bind("<Return>", lambda _event: self.plot_partial_effects())
        self.partial_effect_values_entry.bind("<FocusOut>", lambda _event: self.plot_partial_effects())

        ttk.Label(partial_controls, text="Otras vars:").pack(side=tk.LEFT, padx=(6, 6))
        self.partial_effect_baseline_var = StringVar()
        self.partial_effect_baseline_entry = ttk.Entry(partial_controls, textvariable=self.partial_effect_baseline_var, width=34)
        self.partial_effect_baseline_entry.pack(side=tk.LEFT, padx=(0, 8))
        self.partial_effect_baseline_entry.bind("<Return>", lambda _event: self.plot_partial_effects())
        self.partial_effect_baseline_entry.bind("<FocusOut>", lambda _event: self.plot_partial_effects())

        ttk.Button(partial_controls, text="Actualizar", command=self.plot_partial_effects).pack(side=tk.LEFT, padx=(4, 0))
        ttk.Label(
            self.partial_effect_tab,
            text="Formato de 'Otras vars': Variable:valor, Variable2:valor2",
            foreground="#666666",
        ).pack(anchor="w", padx=12, pady=(2, 4))

        self.partial_effect_fig = plt.figure(figsize=(6, 4))
        self.partial_effect_canvas = FigureCanvasTkAgg(self.partial_effect_fig, master=self.partial_effect_tab)
        self.partial_effect_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self._register_editable_plot_canvas(self.partial_effect_canvas, "partial_effect")

        risk_controls = ttk.Frame(self.risk_effect_tab)
        risk_controls.pack(fill=tk.X, padx=8, pady=(8, 0))
        ttk.Label(risk_controls, text="Covariable:").pack(side=tk.LEFT, padx=(0, 6))
        self.risk_effect_var = StringVar()
        self.risk_effect_combo = ttk.Combobox(risk_controls, textvariable=self.risk_effect_var, state="readonly", width=28)
        self.risk_effect_combo.pack(side=tk.LEFT, padx=(0, 8))
        self.risk_effect_combo.bind("<<ComboboxSelected>>", lambda _event: self.plot_risk_vs_covariate())

        ttk.Label(risk_controls, text="Tiempo t:").pack(side=tk.LEFT, padx=(8, 6))
        self.risk_time_var = StringVar()
        self.risk_time_entry = ttk.Entry(risk_controls, textvariable=self.risk_time_var, width=10)
        self.risk_time_entry.pack(side=tk.LEFT, padx=(0, 6))
        self.risk_time_entry.bind("<Return>", lambda _event: self.plot_risk_vs_covariate())
        self.risk_time_entry.bind("<FocusOut>", lambda _event: self.plot_risk_vs_covariate())

        self.risk_effect_fig = plt.figure(figsize=(6, 4))
        self.risk_effect_canvas = FigureCanvasTkAgg(self.risk_effect_fig, master=self.risk_effect_tab)
        self.risk_effect_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self._register_editable_plot_canvas(self.risk_effect_canvas, "risk_effect")

        self.hazard_fig = plt.figure(figsize=(6, 4))
        self.hazard_canvas = FigureCanvasTkAgg(self.hazard_fig, master=self.cumulative_hazard_tab)
        self.hazard_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self._register_editable_plot_canvas(self.hazard_canvas, "cumulative_hazard")

        self.forest_fig = plt.figure(figsize=(6, 4))
        self.forest_canvas = FigureCanvasTkAgg(self.forest_fig, master=self.forest_plot_tab)
        self.forest_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self._register_editable_plot_canvas(self.forest_canvas, "forest")

        self.brier_fig = plt.figure(figsize=(6, 4))
        self.brier_canvas = FigureCanvasTkAgg(self.brier_fig, master=self.brier_tab)
        self.brier_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self._register_editable_plot_canvas(self.brier_canvas, "brier")

    def _register_editable_plot_canvas(self, canvas, plot_key):
        if canvas is None:
            return

        setattr(canvas, '_matabs_plot_key', plot_key)
        if getattr(canvas, '_matabs_text_edit_cid', None) is not None:
            return

        cid = canvas.mpl_connect(
            'button_press_event',
            lambda event, current_canvas=canvas: self._on_plot_text_click(event, current_canvas),
        )
        setattr(canvas, '_matabs_text_edit_cid', cid)

    def _iter_editable_plot_texts(self, ax):
        if ax is None:
            return []

        editable_items = [
            ('title', ax.title, 'título'),
            ('xlabel', ax.xaxis.label, 'eje X'),
            ('ylabel', ax.yaxis.label, 'eje Y'),
        ]

        for idx, tick_text in enumerate(ax.get_xticklabels()):
            if str(tick_text.get_text()).strip():
                editable_items.append((f'xtick_label_{idx}', tick_text, f'categoría eje X {idx + 1}'))

        for idx, tick_text in enumerate(ax.get_yticklabels()):
            if str(tick_text.get_text()).strip():
                editable_items.append((f'ytick_label_{idx}', tick_text, f'categoría eje Y {idx + 1}'))

        legend = ax.get_legend()
        if legend is not None:
            legend_title = legend.get_title()
            if legend_title is not None and str(legend_title.get_text()).strip():
                editable_items.append(('legend_title', legend_title, 'título de la leyenda'))
            for idx, legend_text in enumerate(legend.get_texts()):
                editable_items.append((f'legend_label_{idx}', legend_text, f'leyenda {idx + 1}'))

        return editable_items

    def _text_contains_click(self, text_artist, event, renderer):
        if text_artist is None or renderer is None or event is None or event.x is None or event.y is None:
            return False

        current_text = text_artist.get_text() if hasattr(text_artist, 'get_text') else ''
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
        if target_key in {'title', 'xlabel', 'ylabel', 'legend_title'}:
            overrides[target_key] = new_text
            return

        target_mappings = {
            'legend_label_': ('legend_labels', (lambda current_ax: [txt.get_text() for txt in current_ax.get_legend().get_texts()]) if ax is not None and ax.get_legend() is not None else None),
            'xtick_label_': ('xtick_labels', (lambda current_ax: [txt.get_text() for txt in current_ax.get_xticklabels()]) if ax is not None else None),
            'ytick_label_': ('ytick_labels', (lambda current_ax: [txt.get_text() for txt in current_ax.get_yticklabels()]) if ax is not None else None),
        }

        for prefix, (storage_key, extractor) in target_mappings.items():
            if not (isinstance(target_key, str) and target_key.startswith(prefix)):
                continue
            try:
                label_index = int(target_key.rsplit('_', 1)[1])
            except (TypeError, ValueError):
                return

            current_labels = extractor(ax) if callable(extractor) and ax is not None else []
            stored_labels = list(overrides.get(storage_key, current_labels))
            while len(stored_labels) <= label_index:
                stored_labels.append('')
            stored_labels[label_index] = new_text
            overrides[storage_key] = stored_labels
            return

    def _apply_plot_text_overrides(self, ax, plot_key):
        if ax is None:
            return

        overrides = self._plot_text_overrides.get(plot_key, {}) if isinstance(self._plot_text_overrides, dict) else {}
        if not overrides:
            return

        if 'title' in overrides:
            ax.set_title(overrides['title'])
        if 'xlabel' in overrides:
            ax.set_xlabel(overrides['xlabel'])
        if 'ylabel' in overrides:
            ax.set_ylabel(overrides['ylabel'])

        if isinstance(overrides.get('xtick_labels'), list):
            xticks = ax.get_xticks()
            current_xticklabels = ax.get_xticklabels()
            x_rotation = current_xticklabels[0].get_rotation() if current_xticklabels else 0
            merged_xlabels = [tick.get_text() for tick in current_xticklabels]
            if len(merged_xlabels) < len(xticks):
                merged_xlabels.extend([''] * (len(xticks) - len(merged_xlabels)))
            for idx, new_label in enumerate(overrides['xtick_labels']):
                if idx < len(merged_xlabels):
                    merged_xlabels[idx] = str(new_label)
            if len(xticks) == len(merged_xlabels):
                ax.set_xticks(xticks)
                ax.set_xticklabels(merged_xlabels, rotation=x_rotation)

        if isinstance(overrides.get('ytick_labels'), list):
            yticks = ax.get_yticks()
            current_yticklabels = ax.get_yticklabels()
            y_rotation = current_yticklabels[0].get_rotation() if current_yticklabels else 0
            merged_ylabels = [tick.get_text() for tick in current_yticklabels]
            if len(merged_ylabels) < len(yticks):
                merged_ylabels.extend([''] * (len(yticks) - len(merged_ylabels)))
            for idx, new_label in enumerate(overrides['ytick_labels']):
                if idx < len(merged_ylabels):
                    merged_ylabels[idx] = str(new_label)
            if len(yticks) == len(merged_ylabels):
                ax.set_yticks(yticks)
                ax.set_yticklabels(merged_ylabels, rotation=y_rotation)

        legend = ax.get_legend()
        if legend is not None:
            if 'legend_title' in overrides:
                legend.set_title(overrides['legend_title'])
            if isinstance(overrides.get('legend_labels'), list):
                for text_obj, new_label in zip(legend.get_texts(), overrides['legend_labels']):
                    text_obj.set_text(str(new_label))

    def _on_plot_text_click(self, event, canvas):
        if event is None or canvas is None:
            return

        toolbar = getattr(canvas, 'toolbar', None)
        if toolbar is not None and getattr(toolbar, 'mode', ''):
            return

        figure = getattr(canvas, 'figure', None)
        if figure is None:
            return

        try:
            renderer = canvas.get_renderer()
        except Exception:
            renderer = getattr(figure.canvas, 'get_renderer', lambda: None)()

        plot_key = getattr(canvas, '_matabs_plot_key', None)
        parent_window = self.winfo_toplevel() if hasattr(self, 'winfo_toplevel') else None

        for ax in figure.axes:
            for target_key, text_artist, label_name in self._iter_editable_plot_texts(ax):
                if not self._text_contains_click(text_artist, event, renderer):
                    continue

                current_text = text_artist.get_text() if hasattr(text_artist, 'get_text') else ''
                new_text = simpledialog.askstring(
                    'Editar texto del gráfico',
                    f'Nuevo texto para {label_name}:',
                    initialvalue=current_text,
                    parent=parent_window,
                )
                if new_text is None:
                    return

                text_artist.set_text(new_text)
                self._update_plot_text_override(plot_key, target_key, new_text, ax=ax)
                canvas.draw_idle()
                return

    def load_data(self):
        filepath = filedialog.askopenfilename(filetypes=[("Excel files", "*.xlsx *.xls"), ("CSV files", "*.csv")])
        if not filepath:
            return
        try:
            if filepath.endswith('.csv'):
                self.data = pd.read_csv(filepath)
            else:
                self.data = pd.read_excel(filepath)
            
            self.base_data = self.data.copy(deep=True)
            self.file_label.config(text=os.path.basename(filepath))
            if self.filter_component:
                self.filter_component.set_dataframe(self.data)
            self._update_variable_comboboxes()
            self._reset_results_view()
            self.using_shared_dataset = False
            self.shared_metadata = {}
            self.current_shared_filter_summary = []
            messagebox.showinfo("Éxito", "Datos cargados correctamente.")
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo cargar el archivo: {e}")
            self.data = None
            self.base_data = None

    def _update_variable_comboboxes(self):
        if self.data is not None:
            columns = self.data.columns.tolist()
            self.duration_combo['values'] = columns
            self.event_combo['values'] = [""] + columns
            self.covariates_listbox.delete(0, tk.END)
            for col in columns:
                self.covariates_listbox.insert(tk.END, col)

    def _quote_identifier(self, column_name):
        return f"Q({json.dumps(str(column_name))})"

    def _get_spline_display_value(self, internal_type):
        return self.spline_type_display_map.get(internal_type, internal_type)

    def _get_spline_internal_type(self, display_value):
        return self.spline_type_reverse_map.get(display_value, display_value)

    def _get_default_spline_display(self):
        return self._get_spline_display_value("B-spline")

    def _get_categorical_compare_display_value(self, internal_mode):
        return self.categorical_compare_display_map.get(internal_mode, self.categorical_compare_display_map["all"])

    def _get_categorical_compare_internal_mode(self, display_value):
        return self.categorical_compare_reverse_map.get(display_value, "all")

    def _get_default_categorical_compare_display(self):
        return self._get_categorical_compare_display_value("all")

    def _prepare_categorical_series_for_model(self, series, ref_cat=None, compare_mode="all"):
        if series is None:
            return series

        prepared = series.copy().astype(object)
        non_null_mask = prepared.notna()
        if non_null_mask.any():
            prepared.loc[non_null_mask] = prepared.loc[non_null_mask].astype(str)

        ref_cat_str = None if ref_cat is None or str(ref_cat).strip() == "" else str(ref_cat)
        if ref_cat_str and compare_mode == "one_vs_rest":
            prepared.loc[non_null_mask] = prepared.loc[non_null_mask].apply(
                lambda value: ref_cat_str if str(value) == ref_cat_str else "RESTO"
            )

        return prepared

    def _parse_manual_knots(self, raw_value):
        if raw_value is None:
            return []
        if isinstance(raw_value, (list, tuple, set)):
            candidates = list(raw_value)
        else:
            candidates = re.split(r"[,;]", str(raw_value))

        clean_knots = []
        for candidate in candidates:
            candidate_str = str(candidate).strip()
            if not candidate_str:
                continue
            try:
                clean_knots.append(float(candidate_str))
            except ValueError:
                continue
        return sorted(set(clean_knots))

    def _resolve_holdout_split_settings(self, data, event_col, requested_test_size,
                                        min_train_rows=5, min_test_rows=2,
                                        prefer_stratify=True, context_label="holdout"):
        if data is None or len(data) == 0:
            raise ValueError("No hay datos suficientes para crear la partición train/test.")
        total_rows = int(len(data))
        try:
            requested = float(requested_test_size)
        except (TypeError, ValueError):
            requested = 0.25
        requested = min(max(requested, 0.05), 0.95)
        min_train_rows = max(int(min_train_rows or 0), 2)
        min_test_rows = max(int(min_test_rows or 0), 1)
        warnings = []
        stratify_values = None
        if prefer_stratify and event_col and event_col in data.columns:
            event_series = pd.to_numeric(data[event_col], errors='coerce').fillna(0).astype(int)
            n_classes = int(event_series.nunique())
            if n_classes > 1 and event_series.value_counts().min() >= 2:
                stratify_values = event_series
                min_train_rows = max(min_train_rows, n_classes)
                min_test_rows = max(min_test_rows, n_classes)
            elif n_classes > 1:
                warnings.append("No se pudo estratificar por evento; se usó partición aleatoria simple.")
        if total_rows <= (min_train_rows + min_test_rows):
            fallback_train = max(2, min(total_rows - min_test_rows, min_train_rows))
            if total_rows <= fallback_train:
                raise ValueError(f"No hay suficientes observaciones ({total_rows}) para holdout.")
            min_train_rows = fallback_train
        max_test_size = (total_rows - min_train_rows) / float(total_rows)
        min_test_size = min_test_rows / float(total_rows)
        adjusted = requested
        if adjusted > max_test_size:
            adjusted = max_test_size
            warnings.append(f"Proporción test ajustada de {requested:.2f} a {adjusted:.2f} para dejar al menos {min_train_rows} casos en entrenamiento.")
        if adjusted < min_test_size:
            adjusted = min_test_size
        adjusted = min(max(adjusted, 0.05), 0.95)
        test_count = int(np.ceil(adjusted * total_rows))
        train_count = total_rows - test_count
        if train_count < min_train_rows or test_count < min_test_rows:
            raise ValueError(f"No se pudo crear partición válida ({total_rows} filas, train={train_count}, test={test_count}).")
        return float(adjusted), stratify_values, warnings

    def _derive_spline_df(self, spline_type, spline_degree, spline_num_knots, custom_knots=None):
        try:
            knots_from_config = len(custom_knots) if custom_knots else 0
        except TypeError:
            knots_from_config = 0

        effective_knots = knots_from_config if knots_from_config > 0 else max(0, int(spline_num_knots or 0))
        effective_degree = max(1, int(spline_degree or 1))

        if spline_type == "B-spline":
            if effective_knots > 0:
                derived_df = effective_knots + effective_degree
            else:
                derived_df = effective_degree
            return max(derived_df, 1)

        if effective_knots > 0:
            derived_df = effective_knots + 1
        else:
            derived_df = 4
        return max(derived_df, 1)

    def _build_degree_only_polynomial_formula(self, column_name, degree):
        safe_term = self._quote_identifier(column_name)
        effective_degree = max(1, int(degree or 1))
        formula_terms = [safe_term]
        for power in range(2, effective_degree + 1):
            formula_terms.append(f"I({safe_term} ** {power})")
        return " + ".join(formula_terms)

    def _get_aft_lambda_rows(self, results_df=None):
        source_results = results_df if results_df is not None else self.results
        if source_results is None or source_results.empty:
            return pd.DataFrame()

        summary_df = source_results.reset_index()
        if not {'param', 'covariate'}.issubset(summary_df.columns):
            if 'covariate' in summary_df.columns:
                return summary_df[summary_df['covariate'] != 'Intercept'].copy()
            return summary_df.copy()

        available_params = [str(param) for param in summary_df['param'].dropna().unique().tolist()]
        preferred_order = ['lambda_', 'mu_', 'alpha_']

        selected_param = next((param for param in preferred_order if param in available_params), None)
        if selected_param is None:
            non_intercept_counts = (
                summary_df[summary_df['covariate'] != 'Intercept']
                .groupby('param')
                .size()
                .sort_values(ascending=False)
            )
            if not non_intercept_counts.empty:
                selected_param = str(non_intercept_counts.index[0])
            elif available_params:
                selected_param = available_params[0]

        if selected_param is None:
            return pd.DataFrame()

        return summary_df[
            (summary_df['param'] == selected_param) & (summary_df['covariate'] != 'Intercept')
        ].copy()

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

        config = self.variable_configs.get(column_name, {}) if isinstance(self.variable_configs, dict) else {}
        if config.get('type') == 'Cualitativa' and config.get('compare_mode') == 'one_vs_rest':
            ref_cat = str(config.get('ref_cat', '')).strip()
            return ref_cat if ref_cat and raw_text == ref_cat else 'RESTO'

        return raw_text

    def _parse_partial_effect_values(self, column_name, raw_value):
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

    def _parse_partial_effect_baseline_overrides(self, raw_value, exclude_covariate=None):
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

    def _get_spline_knots_for_display(self, covariate_name):
        config = self.variable_configs.get(covariate_name, {}) if isinstance(self.variable_configs, dict) else {}
        if not config.get('spline', False):
            return [], None

        data = self.latest_fit_dataframe
        if data is None or covariate_name not in data.columns:
            return [], None

        numeric_series = pd.to_numeric(data[covariate_name], errors='coerce').dropna()
        if numeric_series.empty or numeric_series.nunique() <= 1:
            return [], None

        custom_knots = self._parse_manual_knots(config.get('custom_knots', []))
        min_val = float(numeric_series.min())
        max_val = float(numeric_series.max())

        if custom_knots:
            exact_knots = [float(knot) for knot in custom_knots if np.isfinite(float(knot)) and min_val <= float(knot) <= max_val]
            return sorted(set(exact_knots)), 'exactos'

        try:
            num_knots = max(0, int(config.get('num_knots', 0) or 0))
        except (TypeError, ValueError):
            num_knots = 0

        if num_knots <= 0:
            return [], None

        quantiles = np.linspace(0, 1, num_knots + 2)[1:-1]
        auto_knots = np.quantile(numeric_series, quantiles)
        auto_knots = np.atleast_1d(auto_knots).astype(float)
        cleaned_knots = [float(knot) for knot in auto_knots if np.isfinite(knot) and min_val < float(knot) < max_val]
        return sorted(set(cleaned_knots)), 'auto'

    def _sync_plot_covariate_selectors(self):
        candidates = self._get_plot_covariate_candidates()

        if hasattr(self, 'risk_time_var'):
            default_time = np.nan
            if self.latest_fit_dataframe is not None and self.latest_duration_col in self.latest_fit_dataframe.columns:
                try:
                    default_time = float(np.nanmedian(pd.to_numeric(self.latest_fit_dataframe[self.latest_duration_col], errors='coerce')))
                except Exception:
                    default_time = np.nan
            if not self.risk_time_var.get().strip() and np.isfinite(default_time) and default_time > 0:
                self.risk_time_var.set(f"{default_time:.2f}")

        if hasattr(self, 'partial_effect_combo'):
            self.partial_effect_combo['values'] = candidates
            if candidates:
                current_partial = self.partial_effect_var.get()
                if current_partial not in candidates:
                    self.partial_effect_var.set(candidates[0])
            else:
                self.partial_effect_var.set('')

        if hasattr(self, 'risk_effect_combo'):
            self.risk_effect_combo['values'] = candidates
            if candidates:
                current_risk = self.risk_effect_var.get()
                if current_risk not in candidates:
                    self.risk_effect_var.set(candidates[0])
            else:
                self.risk_effect_var.set('')

    def _compute_cox_snell_diagnostics(self):
        if self.fitter is None or self.latest_fit_dataframe is None:
            return None

        data = self.latest_fit_dataframe.copy()
        duration_col = self.latest_duration_col
        event_col = self.latest_event_col

        if not duration_col or duration_col not in data.columns:
            return None

        covariate_df = data[self.latest_covariates] if self.latest_covariates else pd.DataFrame(index=data.index)
        survival_df = self.fitter.predict_survival_function(covariate_df)
        if survival_df is None or survival_df.empty:
            return None

        time_grid = survival_df.index.to_numpy(dtype=float)
        if time_grid.size == 0:
            return None

        residuals = []
        for col_position, (_, row) in enumerate(data.iterrows()):
            observed_time = float(row[duration_col])
            curve_values = survival_df.iloc[:, col_position].to_numpy(dtype=float)
            survival_at_time = float(np.interp(observed_time, time_grid, curve_values))
            survival_at_time = float(np.clip(survival_at_time, 1e-12, 1.0))
            residuals.append(-np.log(survival_at_time))

        residuals = np.asarray(residuals, dtype=float)
        valid_mask = np.isfinite(residuals) & (residuals >= 0)
        if event_col and event_col in data.columns:
            event_observed = pd.to_numeric(data[event_col], errors='coerce').fillna(0).to_numpy(dtype=float)
            valid_mask &= np.isfinite(event_observed)
            event_observed = event_observed[valid_mask]
        else:
            event_observed = None

        residuals = residuals[valid_mask]
        if residuals.size < 2:
            return None

        naf = NelsonAalenFitter()
        if event_observed is not None:
            naf.fit(residuals, event_observed=event_observed, label='Nelson-Aalen (Cox-Snell)')
        else:
            naf.fit(residuals, label='Nelson-Aalen (Cox-Snell)')

        cumulative_df = naf.cumulative_hazard_.reset_index()
        x_values = cumulative_df.iloc[:, 0].to_numpy(dtype=float)
        y_values = cumulative_df.iloc[:, 1].to_numpy(dtype=float)

        fit_mask = np.isfinite(x_values) & np.isfinite(y_values)
        if fit_mask.sum() >= 2:
            slope, intercept = np.polyfit(x_values[fit_mask], y_values[fit_mask], 1)
            mean_abs_deviation = float(np.mean(np.abs(y_values[fit_mask] - x_values[fit_mask])))
        else:
            slope, intercept, mean_abs_deviation = np.nan, np.nan, np.nan

        if pd.notna(mean_abs_deviation) and abs(slope - 1.0) <= 0.15 and mean_abs_deviation <= 0.10:
            interpretation = 'Buen ajuste: la curva de Cox-Snell sigue de cerca la diagonal.'
        elif pd.notna(mean_abs_deviation) and abs(slope - 1.0) <= 0.35 and mean_abs_deviation <= 0.25:
            interpretation = 'Ajuste aceptable: hay desviación moderada frente a la diagonal.'
        else:
            interpretation = 'Posible desajuste: la curva se aleja de la diagonal; compare otra distribución AFT.'

        return {
            'residuals': residuals,
            'x': x_values,
            'y': y_values,
            'slope': slope,
            'intercept': intercept,
            'mae': mean_abs_deviation,
            'interpretation': interpretation,
        }

    def _format_metric_value(self, value, decimals=3):
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            return "-"
        if not np.isfinite(numeric_value):
            return "-"
        return f"{numeric_value:.{decimals}f}"

    def _get_confidence_z_value(self, confidence_level=0.95):
        try:
            level = float(confidence_level)
        except (TypeError, ValueError):
            level = 0.95
        level = float(np.clip(level, 0.50, 0.999))
        return float(NormalDist().inv_cdf(0.5 + (level / 2.0)))

    def _format_c_index_display(self, value, ci=None, decimals=3):
        base_text = self._format_metric_value(value, decimals)
        if base_text == "-":
            return "-"
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

    def _resolve_tau(self, times, events):
        """Resolve τ truncation time for IPCW metrics."""
        mode = getattr(self, 'tau_mode_var', None)
        mode = mode.get() if mode else "Auto (P90)"
        times_arr = np.asarray(times, dtype=float)
        events_arr = np.asarray(events, dtype=bool)
        event_times = times_arr[events_arr & np.isfinite(times_arr)]
        if event_times.size == 0:
            event_times = times_arr[np.isfinite(times_arr)]
        if mode == "Manual":
            try:
                val = float(self.tau_manual_var.get())
                if np.isfinite(val) and val > 0:
                    return val
            except Exception:
                pass
        if mode == "Último evento":
            return float(np.max(event_times)) if event_times.size > 0 else None
        return float(np.percentile(event_times, 90)) if event_times.size > 0 else None

    def _build_evaluation_time_grid_aft(self, train_times, test_times, tau=None):
        """Build evaluation time grid for Antolini's Ctd AUC."""
        train_t = np.asarray(train_times, dtype=float)
        test_t = np.asarray(test_times, dtype=float)
        train_t = train_t[np.isfinite(train_t)]
        test_t = test_t[np.isfinite(test_t)]
        if train_t.size < 5 or test_t.size < 3:
            return None
        lower = max(float(np.nanpercentile(train_t, 10)), float(np.nanmin(test_t)))
        upper = min(float(np.nanpercentile(train_t, 90)), float(np.nanmax(test_t)))
        if tau is not None and np.isfinite(tau) and tau > 0:
            upper = min(upper, float(tau))
        if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
            return None
        grid = np.unique(np.linspace(lower, upper, num=12))
        return grid if grid.size >= 2 else None

    def _bootstrap_model_c_index_ci(self, fitter, data, event_col=None, n_bootstrap=120, random_state=42):
        if fitter is None or not isinstance(data, pd.DataFrame) or data.empty or len(data) < 5:
            return None

        event_name = event_col or getattr(self, 'latest_event_col', None)
        if event_name and event_name in data.columns:
            event_values = pd.to_numeric(data[event_name], errors='coerce').fillna(0) > 0
            if event_values.nunique() < 2:
                return None

        rng = np.random.default_rng(random_state)
        population_idx = np.arange(len(data))
        sampled_scores = []
        for _ in range(int(max(20, n_bootstrap))):
            bootstrap_idx = rng.choice(population_idx, size=len(population_idx), replace=True)
            sample_df = data.iloc[bootstrap_idx]
            if event_name and event_name in sample_df.columns:
                sample_events = pd.to_numeric(sample_df[event_name], errors='coerce').fillna(0) > 0
                if sample_events.nunique() < 2:
                    continue
            try:
                c_value = float(fitter.score(sample_df, scoring_method="concordance_index"))
            except Exception:
                continue
            if np.isfinite(c_value):
                sampled_scores.append(c_value)

        if len(sampled_scores) < 10:
            return None

        alpha = 0.05
        lower = float(np.nanquantile(sampled_scores, alpha / 2.0))
        upper = float(np.nanquantile(sampled_scores, 1.0 - (alpha / 2.0)))
        lower = float(np.clip(lower, 0.0, 1.0))
        upper = float(np.clip(upper, 0.0, 1.0))
        return (min(lower, upper), max(lower, upper))

    def _get_shape_parameter_info(self, results_df=None):
        source_results = results_df if results_df is not None else self.results
        if source_results is None or source_results.empty:
            return None, np.nan, np.nan

        try:
            summary_df = source_results.reset_index()
            for shape_param in ('rho_', 'beta_', 'sigma_'):
                shape_rows = summary_df[(summary_df['param'] == shape_param) & (summary_df['covariate'] == 'Intercept')]
                if shape_rows.empty:
                    continue
                shape_row = shape_rows.iloc[0]
                if 'exp(coef)' in shape_row and pd.notna(shape_row['exp(coef)']):
                    shape_value = float(shape_row['exp(coef)'])
                elif 'coef' in shape_row and pd.notna(shape_row['coef']):
                    shape_value = float(np.exp(float(shape_row['coef'])))
                else:
                    continue

                inverse_value = np.nan
                if np.isfinite(shape_value) and shape_value != 0:
                    inverse_value = 1.0 / float(shape_value)
                return shape_param, shape_value, inverse_value
        except Exception:
            pass

        return None, np.nan, np.nan

    def _compute_current_model_registry_metrics(self, holdout_info=None):
        metrics = {
            'aic': getattr(self.fitter, 'AIC_', np.nan),
            'bic': getattr(self.fitter, 'BIC_', np.nan),
            'c_index': np.nan,
            'c_index_ci': None,
            'c_index_test': np.nan,
            'c_index_test_ci': None,
            'c_index_uno': None,
            'c_index_antolini': None,
            'tau': None,
            'ibs': None,
            'brier_q25': None, 'brier_q50': None, 'brier_q75': None,
            'auroc_q25': None, 'auroc_q50': None, 'auroc_q75': None,
            'c_harrell_q25': None, 'c_harrell_q50': None, 'c_harrell_q75': None,
            'time_q25': None, 'time_q50': None, 'time_q75': None,
            'test_proportion': np.nan,
            'lr_stat': np.nan,
            'lr_p': np.nan,
            'rho_value': np.nan,
            'rho_inverse': np.nan,
            'cox_snell_slope': np.nan,
            'cox_snell_distance': np.nan,
        }

        try:
            fit_df = self.latest_fit_dataframe if isinstance(self.latest_fit_dataframe, pd.DataFrame) else None
            if self.fitter is not None and fit_df is not None and not fit_df.empty:
                metrics['c_index'] = float(self.fitter.score(fit_df, scoring_method="concordance_index"))
                metrics['c_index_ci'] = self._bootstrap_model_c_index_ci(self.fitter, fit_df, event_col=self.latest_event_col)
        except Exception:
            pass

        if holdout_info and holdout_info.get('active'):
            metrics['c_index'] = holdout_info.get('c_index_train', np.nan)
            metrics['c_index_ci'] = holdout_info.get('c_index_train_ci')
            metrics['c_index_test'] = holdout_info.get('c_index_test', np.nan)
            metrics['c_index_test_ci'] = holdout_info.get('c_index_test_ci')
            metrics['test_proportion'] = holdout_info.get('test_proportion', np.nan)
            metrics['c_index_uno'] = holdout_info.get('c_index_uno')
            metrics['c_index_antolini'] = holdout_info.get('c_index_antolini')
            metrics['tau'] = holdout_info.get('tau')
            metrics['ibs'] = holdout_info.get('ibs')
            for _qk in ('brier_q25','brier_q50','brier_q75','auroc_q25','auroc_q50','auroc_q75',
                         'c_harrell_q25','c_harrell_q50','c_harrell_q75','time_q25','time_q50','time_q75'):
                metrics[_qk] = holdout_info.get(_qk)

        try:
            lr_test = self.fitter.log_likelihood_ratio_test()
            if lr_test is not None:
                metrics['lr_stat'] = float(lr_test.test_statistic)
                metrics['lr_p'] = float(lr_test.p_value)
        except Exception:
            pass

        _shape_label, shape_value, shape_inverse = self._get_shape_parameter_info(self.results)
        if pd.notna(shape_value):
            metrics['rho_value'] = float(shape_value)
        if pd.notna(shape_inverse):
            metrics['rho_inverse'] = float(shape_inverse)

        cox_snell = self._compute_cox_snell_diagnostics()
        if cox_snell:
            metrics['cox_snell_slope'] = float(cox_snell.get('slope', np.nan))
            if pd.notna(metrics['cox_snell_slope']):
                metrics['cox_snell_distance'] = abs(metrics['cox_snell_slope'] - 1.0)

        return metrics

    def _store_current_model_snapshot(self, model_label, report_text, holdout_info=None):
        model_number = len(self.saved_models) + 1
        covariates_short = ", ".join(self.latest_covariates[:4])
        if len(self.latest_covariates) > 4:
            covariates_short += f" +{len(self.latest_covariates) - 4} más"
        parameter_text = f"dur={self.latest_duration_col}; evento={self.latest_event_col or 'N/A'}; covs={covariates_short or 'N/A'}"

        snapshot = {
            'label': f"M{model_number}",
            'timestamp': datetime.now().strftime("%H:%M:%S"),
            'model_type': model_label,
            'duration_col': self.latest_duration_col,
            'event_col': self.latest_event_col,
            'covariates': list(self.latest_covariates),
            'formula': self.latest_formula,
            'fit_dataframe': self.latest_fit_dataframe.copy(deep=True) if isinstance(self.latest_fit_dataframe, pd.DataFrame) else self.latest_fit_dataframe,
            'results': self.results.copy(deep=True) if isinstance(self.results, pd.DataFrame) else self.results,
            'fitter': self.fitter,
            'report_text': report_text,
            'variable_configs': copy.deepcopy(self.variable_configs),
            'snapshot_params': {
                'test_size': self.test_size_var.get() if hasattr(self, 'test_size_var') else None,
                'tau_mode': self.tau_mode_var.get() if hasattr(self, 'tau_mode_var') else None,
                'tau_manual': self.tau_manual_var.get() if hasattr(self, 'tau_manual_var') else None,
            },
            'parameter_text': parameter_text,
            'latest_brier_df': self.latest_brier_df.copy(deep=True) if isinstance(self.latest_brier_df, pd.DataFrame) else pd.DataFrame(),
            'latest_eval_time': self.latest_eval_time,
            'metrics': self._compute_current_model_registry_metrics(holdout_info=holdout_info),
        }
        self.saved_models.append(snapshot)
        self.active_saved_model_index = len(self.saved_models) - 1
        self._refresh_saved_models_tree()
        return snapshot

    # ── Column toggle + sorting for saved_models_tree ─────────────────
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
        self._persist_saved_layout("aft_saved_models", self._aft_tree_col_config, getattr(self, "saved_models_tree", None))

    def save_table_layouts(self):
        self._persist_saved_models_tree_layout()

    def _show_aft_tree_column_menu(self, event):
        if not hasattr(self, 'saved_models_tree') or self.saved_models_tree is None:
            return
        menu = tk.Menu(self.saved_models_tree, tearoff=0)
        menu.add_command(label="── Columnas visibles ──", state="disabled")
        menu.add_separator()
        for col_id, cfg in self._aft_tree_col_config.items():
            label = ("✓ " if cfg["visible"] else "   ") + cfg["heading"]
            menu.add_command(label=label, command=lambda c=col_id: self._toggle_aft_tree_column(c))
        menu.add_separator()
        menu.add_command(label="Mostrar todas", command=self._show_all_aft_tree_columns)
        try:
            menu.tk_popup(event.x_root, event.y_root)
        finally:
            menu.grab_release()

    def _toggle_aft_tree_column(self, col_id):
        cfg = self._aft_tree_col_config[col_id]
        try:
            current_width = int(self.saved_models_tree.column(col_id, option="width"))
            if current_width > 0:
                cfg["width"] = current_width
        except Exception:
            pass
        cfg["visible"] = not cfg["visible"]
        self._apply_aft_tree_column_layout()
        self._persist_saved_models_tree_layout()
        self._persist_saved_layout("aft_saved_models", self._aft_tree_col_config, self.saved_models_tree)

    def _show_all_aft_tree_columns(self):
        for col_id, cfg in self._aft_tree_col_config.items():
            cfg["visible"] = True
        self._apply_aft_tree_column_layout()
        self._persist_saved_models_tree_layout()
        self._persist_saved_layout("aft_saved_models", self._aft_tree_col_config, self.saved_models_tree)

    def _apply_aft_tree_column_layout(self):
        visible_cols = [col_id for col_id, cfg in self._aft_tree_col_config.items() if cfg.get("visible", True)]
        try:
            self.saved_models_tree.configure(displaycolumns=visible_cols if visible_cols else ())
        except Exception:
            pass
        for col_id, cfg in self._aft_tree_col_config.items():
            if cfg.get("visible", True):
                self.saved_models_tree.column(col_id, width=cfg["width"], minwidth=24, stretch=False)
            else:
                self.saved_models_tree.column(col_id, width=0, minwidth=0, stretch=False)

    def _sort_saved_models_tree(self, col_name):
        """Ordena saved_models por la columna especificada y repuebla el tree."""
        if not self.saved_models:
            return
        try:
            col_idx = list(self._aft_tree_col_config.keys()).index(col_name)
        except ValueError:
            return
        reverse = self._aft_tree_sort_reversed.get(col_name, False)

        def sort_key(snapshot):
            metrics = snapshot.get('metrics', {})
            mapping = {
                'id': snapshot.get('label', ''),
                'modelo': snapshot.get('model_type', ''),
                'parametros': snapshot.get('parameter_text', ''),
                'aic': metrics.get('aic', np.nan),
                'bic': metrics.get('bic', np.nan),
                'c_index': metrics.get('c_index', np.nan),
                'c_index_test': metrics.get('c_index_test', np.nan),
                'c_uno': metrics.get('c_index_uno', np.nan),
                'c_antolini': metrics.get('c_index_antolini', np.nan),
                'tau': metrics.get('tau', np.nan),
                'ibs': metrics.get('ibs', np.nan),
                'c_q25': metrics.get('c_harrell_q25', np.nan),
                'c_q50': metrics.get('c_harrell_q50', np.nan),
                'c_q75': metrics.get('c_harrell_q75', np.nan),
                'brier_q25': metrics.get('brier_q25', np.nan),
                'brier_q50': metrics.get('brier_q50', np.nan),
                'brier_q75': metrics.get('brier_q75', np.nan),
                'auroc_q25': metrics.get('auroc_q25', np.nan),
                'auroc_q50': metrics.get('auroc_q50', np.nan),
                'auroc_q75': metrics.get('auroc_q75', np.nan),
                'test_pct': metrics.get('test_proportion', np.nan),
                'lr_stat': metrics.get('lr_stat', np.nan),
                'lr_p': metrics.get('lr_p', np.nan),
                'rho': metrics.get('rho_value', np.nan),
                'rho_inv': metrics.get('rho_inverse', np.nan),
                'cs_slope': metrics.get('cox_snell_slope', np.nan),
                'cs_dist': metrics.get('cox_snell_distance', np.nan),
            }
            val = mapping.get(col_name, '')
            try:
                v = float(val)
                return (0, v if not np.isnan(v) else float('-inf'))
            except (ValueError, TypeError):
                return (1, str(val).lower())

        self.saved_models.sort(key=sort_key, reverse=reverse)
        self._aft_tree_sort_reversed[col_name] = not reverse
        if self.active_saved_model_index is not None:
            self.active_saved_model_index = None  # reset – user should re-select
        self._refresh_saved_models_tree()
        self._show_best_sorted_models_popup(col_name, reverse)

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

    def _show_best_sorted_models_popup(self, col_name, reverse):
        if not hasattr(self, "saved_models_tree"):
            return
        children = self.saved_models_tree.get_children("")
        if not children:
            return
        columns = list(self.saved_models_tree["columns"])
        if col_name not in columns:
            return

        col_idx = columns.index(col_name)
        id_idx = columns.index("id") if "id" in columns else None
        modelo_idx = columns.index("modelo") if "modelo" in columns else None
        params_idx = columns.index("parametros") if "parametros" in columns else None

        first_values = self.saved_models_tree.item(children[0], "values")
        best_key = self._parse_table_sort_value(first_values[col_idx] if col_idx < len(first_values) else "")

        all_top_rows = []
        numeric_values = []
        for item_id in children:
            values = self.saved_models_tree.item(item_id, "values")
            current_key = self._parse_table_sort_value(values[col_idx] if col_idx < len(values) else "")
            if not self._sort_keys_equal(current_key, best_key):
                break

            model_id = values[id_idx] if id_idx is not None and id_idx < len(values) else "-"
            model_type = values[modelo_idx] if modelo_idx is not None and modelo_idx < len(values) else "-"
            params_text = values[params_idx] if params_idx is not None and params_idx < len(values) else "-"
            metric_text = values[col_idx] if col_idx < len(values) else "-"
            all_top_rows.append((model_id, params_text, metric_text, str(model_type)))
            if current_key[0] == "num":
                numeric_values.append(float(current_key[1]))

        if not all_top_rows:
            return

        unique_models = set(r[3] for r in all_top_rows)
        diversity_warning = None
        if len(all_top_rows) >= 5 and len(unique_models) == 1:
            diversity_warning = (
                f"⚠  Todos los modelos empatados usan el mismo tipo ({next(iter(unique_models))}). "
                f"Si los parámetros también son idénticos, los resultados convergerán al mismo valor. "
                f"Prueba diferentes distribuciones AFT o distintas covariables para obtener diferenciación real."
            )

        MAX_DISPLAY = 10
        top_rows = all_top_rows[:MAX_DISPLAY]
        hidden_count = len(all_top_rows) - MAX_DISPLAY

        metric_title = str(self._aft_tree_headings.get(col_name, col_name) or col_name)
        popup = tk.Toplevel(self)
        popup.title("Mejor valor en modelos AFT")
        popup.geometry("980x480")
        popup.transient(self.winfo_toplevel())
        popup.grab_set()

        sort_order_text = "descendente" if reverse else "ascendente"
        total_tied = len(all_top_rows)
        title_text = (
            f"Columna: {metric_title} | Orden: {sort_order_text} | "
            f"Modelos empatados en el mejor valor: {total_tied}"
            + (f" (mostrando los primeros {MAX_DISPLAY})" if hidden_count > 0 else "")
        )
        ttk.Label(popup, text=title_text, foreground="navy").pack(anchor="w", padx=12, pady=(10, 4))

        if numeric_values and len(numeric_values) > 1:
            avg_text = f"Promedio entre empatados: {float(np.mean(numeric_values)):.6f}"
            ttk.Label(popup, text=avg_text, foreground="#555555").pack(anchor="w", padx=12, pady=(0, 4))

        if diversity_warning:
            warn_frame = ttk.Frame(popup, relief="solid", padding=6)
            warn_frame.pack(fill=tk.X, padx=12, pady=(0, 6))
            ttk.Label(warn_frame, text=diversity_warning, foreground="#8B4513",
                      wraplength=940, justify="left").pack(anchor="w")

        tree = ttk.Treeview(popup, columns=("modelo", "parametros", "valor"), show="headings", height=10)
        tree.heading("modelo", text="Modelo")
        tree.heading("parametros", text="Parámetros")
        tree.heading("valor", text=metric_title)
        tree.column("modelo", width=90, anchor="center", stretch=False)
        tree.column("parametros", width=700, anchor="w", stretch=True)
        tree.column("valor", width=150, anchor="center", stretch=False)
        tree.pack(fill=tk.BOTH, expand=True, padx=12, pady=(0, 4))

        for row in top_rows:
            tree.insert("", "end", values=(row[0], row[1], row[2]))

        if hidden_count > 0:
            ttk.Label(popup, text=f"+ {hidden_count} modelos más con el mismo valor (no mostrados).",
                      foreground="#777777").pack(anchor="w", padx=12, pady=(0, 4))

        ttk.Button(popup, text="Cerrar", command=popup.destroy).pack(pady=(0, 10))

    def _refresh_saved_models_tree(self):
        if not hasattr(self, 'saved_models_tree'):
            return

        for item in self.saved_models_tree.get_children():
            self.saved_models_tree.delete(item)

        for idx, snapshot in enumerate(self.saved_models):
            metrics = snapshot.get('metrics', {})
            tp = metrics.get('test_proportion', np.nan)
            tp_str = f"{tp:.0%}" if pd.notna(tp) and tp > 0 else "-"
            lr_p_val = metrics.get('lr_p', np.nan)
            lr_p_str = f"{lr_p_val:.4g}" if pd.notna(lr_p_val) else "-"
            c_uno_v = metrics.get('c_index_uno')
            c_ant_v = metrics.get('c_index_antolini')
            tau_v = metrics.get('tau')
            cs_dist_v = metrics.get('cox_snell_distance', np.nan)
            _f4 = lambda v: f"{v:.4f}" if v is not None and pd.notna(v) else "-"
            values = (
                snapshot.get('label', f"M{idx + 1}"),
                snapshot.get('model_type', '-'),
                snapshot.get('parameter_text', '-'),
                self._format_metric_value(metrics.get('aic'), 2),
                self._format_metric_value(metrics.get('bic'), 2),
                self._format_c_index_display(metrics.get('c_index'), metrics.get('c_index_ci'), decimals=3),
                self._format_c_index_display(metrics.get('c_index_test'), metrics.get('c_index_test_ci'), decimals=3),
                _f4(c_uno_v),
                _f4(c_ant_v),
                f"{tau_v:.1f}" if tau_v is not None and pd.notna(tau_v) else "-",
                _f4(metrics.get('ibs')),
                _f4(metrics.get('c_harrell_q25')),
                _f4(metrics.get('c_harrell_q50')),
                _f4(metrics.get('c_harrell_q75')),
                _f4(metrics.get('brier_q25')),
                _f4(metrics.get('brier_q50')),
                _f4(metrics.get('brier_q75')),
                _f4(metrics.get('auroc_q25')),
                _f4(metrics.get('auroc_q50')),
                _f4(metrics.get('auroc_q75')),
                tp_str,
                self._format_metric_value(metrics.get('lr_stat'), 2),
                lr_p_str,
                self._format_metric_value(metrics.get('rho_value'), 3),
                self._format_metric_value(metrics.get('rho_inverse'), 3),
                self._format_metric_value(metrics.get('cox_snell_slope'), 3),
                f"{cs_dist_v:.3f}" if pd.notna(cs_dist_v) else "-",
            )
            self.saved_models_tree.insert('', 'end', iid=str(idx), values=values)

        active_text = "ninguno"
        if self.saved_models and self.active_saved_model_index is not None and 0 <= self.active_saved_model_index < len(self.saved_models):
            active_snapshot = self.saved_models[self.active_saved_model_index]
            active_text = f"{active_snapshot.get('label', '')} - {active_snapshot.get('model_type', '')}"
            self.saved_models_tree.selection_set(str(self.active_saved_model_index))

        if hasattr(self, 'saved_models_status_var'):
            self.saved_models_status_var.set(f"Modelos guardados: {len(self.saved_models)} | Activo: {active_text}")

    def _apply_snapshot_params(self, snapshot_params):
        if not isinstance(snapshot_params, dict):
            return

        if hasattr(self, 'test_size_var') and 'test_size' in snapshot_params:
            test_size_value = snapshot_params.get('test_size')
            self.test_size_var.set('' if test_size_value in (None, 'None') else str(test_size_value))
        if hasattr(self, 'tau_mode_var') and snapshot_params.get('tau_mode'):
            self.tau_mode_var.set(str(snapshot_params.get('tau_mode') or 'Auto (P90)'))
        if hasattr(self, 'tau_manual_var') and 'tau_manual' in snapshot_params:
            tau_manual_value = snapshot_params.get('tau_manual')
            self.tau_manual_var.set('' if tau_manual_value in (None, 'None') else str(tau_manual_value))

    def _restore_saved_snapshot_ui_state(self):
        columns = []
        for df_source in (getattr(self, 'data', None), getattr(self, 'latest_fit_dataframe', None)):
            if isinstance(df_source, pd.DataFrame):
                for col_name in df_source.columns.tolist():
                    if col_name not in columns:
                        columns.append(col_name)
        for cov_name in getattr(self, 'latest_covariates', []):
            if cov_name not in columns:
                columns.append(cov_name)

        if hasattr(self, 'duration_combo'):
            self.duration_combo['values'] = columns
        if hasattr(self, 'event_combo'):
            self.event_combo['values'] = [''] + columns
        if hasattr(self, 'duration_var'):
            duration_value = self.latest_duration_col or ''
            self.duration_var.set(duration_value if (not columns or duration_value in columns) else '')
        if hasattr(self, 'event_var'):
            event_value = self.latest_event_col or ''
            self.event_var.set(event_value if (not columns or event_value in columns) else '')
        if hasattr(self, 'covariates_listbox'):
            self.covariates_listbox.delete(0, tk.END)
            for idx, col_name in enumerate(columns):
                self.covariates_listbox.insert(tk.END, col_name)
                if col_name in getattr(self, 'latest_covariates', []):
                    try:
                        self.covariates_listbox.selection_set(idx)
                    except Exception:
                        pass

    def _apply_saved_model_snapshot(self, snapshot, index=None):
        self.fitter = snapshot.get('fitter')
        saved_results = snapshot.get('results')
        self.results = saved_results.copy(deep=True) if isinstance(saved_results, pd.DataFrame) else saved_results
        saved_df = snapshot.get('fit_dataframe')
        self.latest_fit_dataframe = saved_df.copy(deep=True) if isinstance(saved_df, pd.DataFrame) else saved_df
        self.latest_duration_col = snapshot.get('duration_col')
        self.latest_event_col = snapshot.get('event_col')
        self.latest_covariates = list(snapshot.get('covariates', []))
        self.latest_formula = snapshot.get('formula')
        self.latest_brier_df = snapshot.get('latest_brier_df').copy(deep=True) if isinstance(snapshot.get('latest_brier_df'), pd.DataFrame) else pd.DataFrame()
        self.latest_eval_time = snapshot.get('latest_eval_time')
        self.variable_configs = copy.deepcopy(snapshot.get('variable_configs', {}))
        self._apply_snapshot_params(snapshot.get('snapshot_params', {}))

        if index is not None:
            self.active_saved_model_index = index

        if hasattr(self, 'model_var'):
            self.model_var.set(snapshot.get('model_type', 'Weibull'))
        self._restore_saved_snapshot_ui_state()

        if hasattr(self, 'results_text'):
            self.results_text.delete('1.0', tk.END)
            self.results_text.insert(tk.END, snapshot.get('report_text', ''))

        self._sync_plot_covariate_selectors()
        self.plot_survival_comparison()
        self.plot_partial_effects()
        self.plot_risk_vs_covariate()
        self.plot_cumulative_hazard()
        self.plot_forest()
        self.plot_brier_curve()
        self._refresh_saved_models_tree()

    def _load_selected_saved_model(self, event=None):
        if not self.saved_models:
            messagebox.showinfo('Sin modelos', 'Aún no hay modelos AFT guardados para cargar.')
            return
        selected_items = self.saved_models_tree.selection() if hasattr(self, 'saved_models_tree') else ()
        if not selected_items:
            messagebox.showwarning('Sin selección', 'Seleccione un modelo guardado para cargarlo.')
            return
        selected_index = int(selected_items[0])
        snapshot = self.saved_models[selected_index]
        self._apply_saved_model_snapshot(snapshot, index=selected_index)
        self.notebook.select(self.results_tab)

    def _delete_selected_saved_model(self):
        if not self.saved_models:
            return
        selected_items = self.saved_models_tree.selection() if hasattr(self, 'saved_models_tree') else ()
        if not selected_items:
            messagebox.showwarning('Sin selección', 'Seleccione un modelo guardado para eliminarlo.')
            return

        selected_index = int(selected_items[0])
        del self.saved_models[selected_index]

        if not self.saved_models:
            self.active_saved_model_index = None
            self._reset_results_view()
            self._refresh_saved_models_tree()
            return

        self.active_saved_model_index = min(selected_index, len(self.saved_models) - 1)
        self._apply_saved_model_snapshot(self.saved_models[self.active_saved_model_index], index=self.active_saved_model_index)

    def _clear_saved_models(self):
        if not self.saved_models:
            return
        if not messagebox.askyesno('Limpiar lista', '¿Desea eliminar todos los modelos AFT guardados de la lista?'):
            return
        self.saved_models = []
        self.active_saved_model_index = None
        self._refresh_saved_models_tree()
        self._reset_results_view()

    def run_model(self):
        if self.data is None:
            messagebox.showerror("Error", "Cargue los datos primero.")
            return

        filtered_data = self.filter_component.apply_filters()
        if filtered_data is None:
            messagebox.showerror("Error", "No se pudieron aplicar los filtros al dataset actual.")
            return

        duration_col = self.duration_var.get()
        event_col = self.event_var.get() or None

        selected_indices = self.covariates_listbox.curselection()
        covariates = [self.covariates_listbox.get(i) for i in selected_indices]

        if not duration_col or not covariates:
            messagebox.showerror("Error", "Debe seleccionar la duración y al menos una covariable.")
            return

        formula_parts = []
        for cov in covariates:
            config = self.variable_configs.get(cov, {})
            quoted_cov = self._quote_identifier(cov)
            config_type = config.get('type')

            if config.get('spline', False):
                spline_type = config.get('spline_type', 'B-spline')
                spline_degree = max(1, int(config.get('spline_degree', 3)))
                custom_knots = self._parse_manual_knots(config.get('custom_knots', []))
                spline_num_knots = len(custom_knots) if custom_knots else max(0, int(config.get('num_knots', 0)))
                df_value = max(
                    1,
                    int(
                        config.get(
                            'spline_df',
                            self._derive_spline_df(spline_type, spline_degree, spline_num_knots, custom_knots)
                        )
                    )
                )

                if spline_type == 'Natural':
                    if custom_knots:
                        formula_parts.append(f"cr({quoted_cov}, knots={tuple(custom_knots)})")
                    else:
                        formula_parts.append(f"cr({quoted_cov}, df={df_value})")
                else:
                    if custom_knots:
                        formula_parts.append(
                            f"bs({quoted_cov}, knots={tuple(custom_knots)}, degree={spline_degree}, include_intercept=False)"
                        )
                    elif spline_num_knots == 0:
                        formula_parts.append(self._build_degree_only_polynomial_formula(cov, spline_degree))
                    else:
                        formula_parts.append(
                            f"bs({quoted_cov}, df={df_value}, degree={spline_degree}, include_intercept=False)"
                        )
            elif config_type == 'Cualitativa':
                ref_cat = config.get('ref_cat')
                if ref_cat:
                    formula_parts.append(
                        f"C({quoted_cov}, Treatment(reference={json.dumps(str(ref_cat))}))"
                    )
                else:
                    formula_parts.append(f"C({quoted_cov})")
            else:
                formula_parts.append(quoted_cov)

        formula = " + ".join(formula_parts)
        self.latest_formula = formula

        model_type = self.model_var.get()
        if model_type == "Weibull":
            self.fitter = WeibullAFTFitter()
        elif model_type == "Log-Normal":
            self.fitter = LogNormalAFTFitter()
        elif model_type == "Log-Logistic":
            self.fitter = LogLogisticAFTFitter()

        try:
            ancillary = filtered_data.copy()
            duration_series = pd.to_numeric(ancillary[duration_col], errors="coerce")
            if duration_series.isna().any():
                messagebox.showerror(
                    "Duración inválida",
                    f"La columna '{duration_col}' contiene valores no numéricos o vacíos tras la conversión.",
                )
                return

            adjustment_note = None
            min_duration = duration_series.min()
            if pd.notna(min_duration) and min_duration <= 0:
                epsilon = 1e-6
                shift_amount = abs(min_duration) + epsilon
                duration_series = duration_series + shift_amount
                adjustment_note = (
                    f"Se detectaron duraciones ≤ 0 en '{duration_col}'. "
                    f"Se agregó un desplazamiento de {shift_amount:.6g} para hacerlas positivas."
                )

            ancillary[duration_col] = duration_series

            warnings_list = []

            for cov in covariates:
                config = self.variable_configs.get(cov, {})
                if config.get('type') == 'Cualitativa' and cov in ancillary.columns:
                    ancillary[cov] = self._prepare_categorical_series_for_model(
                        ancillary[cov],
                        ref_cat=config.get('ref_cat'),
                        compare_mode=config.get('compare_mode', 'all'),
                    )

            if event_col:
                event_series = pd.to_numeric(ancillary[event_col], errors="coerce")
                invalid_events = event_series.isna()
                if invalid_events.any():
                    dropped = int(invalid_events.sum())
                    ancillary = ancillary.loc[~invalid_events].copy()
                    event_series = event_series.loc[~invalid_events]
                    warnings_list.append(
                        f"Se descartaron {dropped} filas por eventos no numéricos o vacíos en '{event_col}'."
                    )
                ancillary[event_col] = event_series

            required_cols = [duration_col] + covariates
            if event_col:
                required_cols.append(event_col)

            missing_mask = ancillary[required_cols].isna().any(axis=1)
            if missing_mask.any():
                dropped = int(missing_mask.sum())
                ancillary = ancillary.loc[~missing_mask].copy()
                warnings_list.append(
                    f"Se descartaron {dropped} filas por valores faltantes en variables del modelo."
                )

            if ancillary.empty:
                messagebox.showerror(
                    "Datos insuficientes",
                    "No hay filas válidas después de limpiar duraciones, eventos y covariables."
                )
                return

            non_positive_mask = ancillary[duration_col] <= 0
            if non_positive_mask.any():
                raise ValueError(
                    "This model does not allow for non-positive durations. "
                    "Revise los datos de duración."
                )

            clean_data = ancillary.copy()

            self.latest_fit_dataframe = clean_data
            self.latest_duration_col = duration_col
            self.latest_event_col = event_col
            self.latest_covariates = list(covariates)

            model_definitions = [
                ("Weibull", WeibullAFTFitter),
                ("Log-Normal", LogNormalAFTFitter),
                ("Log-Logistic", LogLogisticAFTFitter),
            ]
            model_class_map = {name: cls for name, cls in model_definitions}

            comparison_table_text = None
            best_model_name = None

            evaluate_all = bool(self.evaluate_all_models_var.get())
            fitted_models_map = {}

            # ── Holdout train/test (move BEFORE model comparison) ───────
            holdout_info = {}
            holdout_active = (
                getattr(self, 'calculate_test_cindex_var', None) is not None
                and self.calculate_test_cindex_var.get()
                and event_col
            )
            df_train = None
            df_test = None
            if holdout_active:
                try:
                    requested_ts = float(self.test_size_var.get())
                    seed_val = int(self.test_random_seed_var.get())
                    min_train = max(8, len(covariates) + 2)
                    prefer_strat = getattr(self, 'stratify_holdout_var', None)
                    prefer_strat = prefer_strat.get() if prefer_strat is not None else True
                    ts, stratify_vals, ho_warnings = self._resolve_holdout_split_settings(
                        clean_data, event_col, requested_ts,
                        min_train_rows=min_train, min_test_rows=2,
                        prefer_stratify=prefer_strat, context_label="AFT holdout",
                    )
                    for hw in ho_warnings:
                        warnings_list.append(f"[Holdout] {hw}")

                    idx_train, idx_test = train_test_split(
                        clean_data.index, test_size=ts,
                        random_state=seed_val, stratify=stratify_vals,
                    )
                    df_train = clean_data.loc[idx_train].copy()
                    df_test = clean_data.loc[idx_test].copy()
                except Exception as e_ho:
                    warnings_list.append(f"[Holdout] Error al dividir: {e_ho}")
                    holdout_active = False

            # Data used for fitting / comparison: train-only when holdout is active
            fit_data = df_train if holdout_active and df_train is not None else clean_data

            if evaluate_all:
                comparison_records = []
                for name, cls in model_definitions:
                    fitter_candidate = cls()
                    try:
                        fitter_candidate.fit(
                            fit_data,
                            duration_col=duration_col,
                            event_col=event_col,
                            formula=formula,
                        )
                        log_likelihood = getattr(fitter_candidate, "log_likelihood_", np.nan)
                        aic_value = getattr(fitter_candidate, "AIC_", np.nan)
                        bic_value = getattr(fitter_candidate, "BIC_", np.nan)
                        try:
                            concordance_value = fitter_candidate.score(
                                fit_data,
                                scoring_method="concordance_index",
                            )
                            concordance_ci = self._bootstrap_model_c_index_ci(
                                fitter_candidate,
                                fit_data,
                                event_col=event_col,
                            )
                        except Exception:
                            concordance_value = np.nan
                            concordance_ci = None

                        try:
                            lr_test = fitter_candidate.log_likelihood_ratio_test()
                            lr_stat = float(lr_test.test_statistic)
                            lr_p = float(lr_test.p_value)
                        except Exception:
                            lr_stat = np.nan
                            lr_p = np.nan

                        comparison_records.append(
                            {
                                "Modelo": name,
                                "Log-Likelihood": log_likelihood,
                                "AIC": aic_value,
                                "BIC": bic_value,
                                "LR chi2": lr_stat,
                                "LR p": lr_p,
                                "C-Index": concordance_value,
                                "C-Index CI": concordance_ci,
                                "Estado": "OK",
                            }
                        )
                        fitted_models_map[name] = fitter_candidate
                    except Exception as model_err:
                        comparison_records.append(
                            {
                                "Modelo": name,
                                "Log-Likelihood": np.nan,
                                "AIC": np.nan,
                                "BIC": np.nan,
                                "LR chi2": np.nan,
                                "LR p": np.nan,
                                "C-Index": np.nan,
                                "C-Index CI": None,
                                "Estado": f"Error: {str(model_err)[:120]}",
                            }
                        )

                ok_records = [r for r in comparison_records if r["Estado"] == "OK" and pd.notna(r["AIC"])]
                if ok_records:
                    best_model_name = min(ok_records, key=lambda r: r["AIC"])["Modelo"]

                comparison_df = pd.DataFrame(comparison_records)
                if not comparison_df.empty:
                    def _format_numeric(val):
                        return f"{val:.4f}" if pd.notna(val) else "-"

                    for column in ["Log-Likelihood", "AIC", "BIC", "LR chi2", "LR p"]:
                        if column in comparison_df.columns:
                            comparison_df[column] = comparison_df[column].apply(_format_numeric)

                    if "C-Index" in comparison_df.columns:
                        comparison_df["C-Index"] = comparison_df.apply(
                            lambda row: self._format_c_index_display(row.get("C-Index"), row.get("C-Index CI"), decimals=4),
                            axis=1,
                        )
                        comparison_df = comparison_df.drop(columns=["C-Index CI"], errors='ignore')

                    comparison_table_text = comparison_df.to_string(index=False)

                selected_fitter = fitted_models_map.get(model_type)
                if selected_fitter is None:
                    messagebox.showerror(
                        "Error en el Modelo",
                        f"No se pudo ajustar el modelo '{model_type}' dentro del grid comparativo. Revise los datos o seleccione otro tipo.",
                    )
                    return

                self.fitter = selected_fitter
            else:
                fitter_class = model_class_map.get(model_type)
                if fitter_class is None:
                    messagebox.showerror("Modelo inválido", f"Modelo AFT desconocido: {model_type}")
                    return

                self.fitter = fitter_class()
                self.fitter.fit(
                    fit_data,
                    duration_col=duration_col,
                    event_col=event_col,
                    formula=formula,
                )

            self.results = self.fitter.summary
            self.latest_brier_df = pd.DataFrame()
            self.latest_eval_time = None
            if holdout_active and df_train is not None:
                self.latest_fit_dataframe = df_train

            # ── Holdout C-index scoring ─────────────────────────────────
            if holdout_active and df_train is not None and df_test is not None:
                try:
                    c_train = float(self.fitter.score(df_train, scoring_method="concordance_index"))
                    c_test = float(self.fitter.score(df_test, scoring_method="concordance_index"))

                    holdout_info = {
                        'active': True,
                        'test_proportion': float(ts),
                        'c_index_train': c_train,
                        'c_index_train_ci': self._bootstrap_model_c_index_ci(self.fitter, df_train, event_col=event_col),
                        'c_index_test': c_test,
                        'c_index_test_ci': self._bootstrap_model_c_index_ci(self.fitter, df_test, event_col=event_col),
                        'n_train': len(df_train),
                        'n_test': len(df_test),
                        'gap': c_test - c_train,
                    }

                    # --- Uno's C-index (IPCW) + Antolini Ctd ---
                    resolved_tau = self._resolve_tau(df_train[duration_col], df_train[event_col])
                    holdout_info['tau'] = resolved_tau
                    if callable(_concordance_index_ipcw_aft):
                        try:
                            y_train_struct = np.array(
                                [(bool(e), float(t)) for e, t in zip(df_train[event_col], df_train[duration_col])],
                                dtype=[('event', bool), ('time', float)],
                            )
                            y_test_struct = np.array(
                                [(bool(e), float(t)) for e, t in zip(df_test[event_col], df_test[duration_col])],
                                dtype=[('event', bool), ('time', float)],
                            )
                            # AFT predict_median: higher median = lower risk → negate for concordance
                            pred_median_test = self.fitter.predict_median(df_test)
                            risk_scores = -np.asarray(pred_median_test, dtype=float).reshape(-1)
                            ipcw_result = _concordance_index_ipcw_aft(y_train_struct, y_test_struct, risk_scores, tau=resolved_tau)
                            holdout_info['c_index_uno'] = float(np.asarray(ipcw_result).reshape(-1)[0])
                        except Exception:
                            holdout_info['c_index_uno'] = None

                        # Antolini's Ctd
                        if callable(_cumulative_dynamic_auc_aft) and y_train_struct is not None:
                            try:
                                eval_grid = self._build_evaluation_time_grid_aft(
                                    df_train[duration_col], df_test[duration_col], tau=resolved_tau)
                                if eval_grid is not None:
                                    _, mean_auc = _cumulative_dynamic_auc_aft(
                                        y_train_struct, y_test_struct, risk_scores, eval_grid)
                                    holdout_info['c_index_antolini'] = float(mean_auc)
                                else:
                                    holdout_info['c_index_antolini'] = None
                            except Exception:
                                holdout_info['c_index_antolini'] = None
                        else:
                            holdout_info['c_index_antolini'] = None

                        # ── IBS + Brier/AUROC/C at quartiles ──────────────
                        if y_train_struct is not None and y_test_struct is not None and eval_grid is not None:
                            try:
                                surv_df_aft = self.fitter.predict_survival_function(df_test, times=eval_grid)
                                surv_matrix_aft = surv_df_aft.values.T  # (n_test, n_times)

                                if callable(_brier_score_aft) and callable(_integrated_brier_score_aft):
                                    _, brier_vals_aft = _brier_score_aft(y_train_struct, y_test_struct, surv_matrix_aft, eval_grid)
                                    brier_vals_aft = np.asarray(brier_vals_aft, dtype=float)
                                    self.latest_brier_df = pd.DataFrame({"time": np.asarray(eval_grid, dtype=float), "brier_score": brier_vals_aft})
                                    if len(eval_grid) > 0:
                                        self.latest_eval_time = float(np.asarray(eval_grid, dtype=float)[min(len(eval_grid) - 1, len(eval_grid) // 2)])
                                    holdout_info['ibs'] = float(_integrated_brier_score_aft(y_train_struct, y_test_struct, surv_matrix_aft, eval_grid))

                                ev_mask_aft = y_test_struct["event"].astype(bool)
                                ev_times_aft = y_test_struct["time"][ev_mask_aft]
                                if ev_times_aft.size >= 4:
                                    tq25 = float(np.percentile(ev_times_aft, 25))
                                    tq50 = float(np.percentile(ev_times_aft, 50))
                                    tq75 = float(np.percentile(ev_times_aft, 75))
                                    holdout_info['time_q25'] = tq25; holdout_info['time_q50'] = tq50; holdout_info['time_q75'] = tq75
                                    q_times_aft = np.array([tq25, tq50, tq75])

                                    # Brier at quartiles
                                    try:
                                        surv_q_aft = self.fitter.predict_survival_function(df_test, times=q_times_aft).values.T
                                        _, brier_qv_aft = _brier_score_aft(y_train_struct, y_test_struct, surv_q_aft, q_times_aft)
                                        brier_qv_aft = np.asarray(brier_qv_aft, dtype=float)
                                        holdout_info['brier_q25'] = float(brier_qv_aft[0])
                                        holdout_info['brier_q50'] = float(brier_qv_aft[1])
                                        holdout_info['brier_q75'] = float(brier_qv_aft[2])
                                    except Exception:
                                        pass

                                    # AUROC at quartiles
                                    if callable(_cumulative_dynamic_auc_aft):
                                        try:
                                            auc_vals_aft, _ = _cumulative_dynamic_auc_aft(y_train_struct, y_test_struct, risk_scores, q_times_aft)
                                            auc_vals_aft = np.asarray(auc_vals_aft, dtype=float)
                                            holdout_info['auroc_q25'] = float(auc_vals_aft[0])
                                            holdout_info['auroc_q50'] = float(auc_vals_aft[1])
                                            holdout_info['auroc_q75'] = float(auc_vals_aft[2])
                                        except Exception:
                                            pass

                                    # C at quartiles (IPCW with tau=quartile)
                                    for tau_q, key in [(tq25, 'c_harrell_q25'), (tq50, 'c_harrell_q50'), (tq75, 'c_harrell_q75')]:
                                        try:
                                            r = _concordance_index_ipcw_aft(y_train_struct, y_test_struct, risk_scores, tau=tau_q)
                                            holdout_info[key] = float(np.asarray(r).reshape(-1)[0])
                                        except Exception:
                                            pass
                            except Exception:
                                pass

                    else:
                        holdout_info['c_index_uno'] = None
                        holdout_info['c_index_antolini'] = None
                except Exception as e_ho:
                    warnings_list.append(f"[Holdout] Error al evaluar: {e_ho}")

            report_text = self._build_results_report(
                self.latest_fit_dataframe if holdout_info.get('active') else clean_data,
                duration_col,
                event_col,
                model_type,
                adjustment_note,
                warnings_list,
                comparison_table_text=comparison_table_text,
                best_model_name=best_model_name,
                holdout_info=holdout_info,
            )

            self.results_text.delete("1.0", tk.END)
            self.results_text.insert(tk.END, report_text)
            self._store_current_model_snapshot(model_type, report_text, holdout_info=holdout_info)
            self._sync_plot_covariate_selectors()

            self.plot_survival_comparison()
            self.plot_partial_effects()
            self.plot_risk_vs_covariate()
            self.plot_cumulative_hazard()
            self.plot_forest()
            self.plot_brier_curve()

            if adjustment_note:
                messagebox.showwarning("Duraciones ajustadas", adjustment_note)
            if warnings_list:
                messagebox.showwarning("Filas descartadas", "\n".join(warnings_list))

            self.notebook.select(self.results_tab)
        except Exception as e:
            messagebox.showerror("Error en el Modelo", f"Ocurrió un error al ajustar el modelo: {e}")
            self._reset_results_view()

    def _build_results_report(
        self,
        data,
        duration_col,
        event_col,
        model_label,
        adjustment_note,
        warnings_list,
        comparison_table_text=None,
        best_model_name=None,
        holdout_info=None,
    ):
        lines = []

        if comparison_table_text:
            lines.append("=== Comparativa de Modelos AFT ===")
            lines.append(comparison_table_text)
            if best_model_name:
                lines.append(f"Mejor AIC en el grid: {best_model_name}")
            lines.append("")

        lines.append("=== Resumen del Modelo AFT ===")
        lines.append(f"Modelo seleccionado: {model_label}")
        lines.append(f"Observaciones utilizadas: {len(data)}")

        if event_col:
            try:
                events = int(np.nan_to_num(data[event_col]).sum())
                censored = len(data) - events
                lines.append(f"Eventos observados: {events}")
                lines.append(f"Censurados: {censored}")
            except Exception:
                lines.append(f"Eventos observados: N/D (no numérico)")

        try:
            median_duration = data[duration_col].median()
            lines.append(f"Mediana de duración: {median_duration:.4g}")
        except Exception:
            lines.append("Mediana de duración: N/D")

        log_likelihood = getattr(self.fitter, "log_likelihood_", None)
        if log_likelihood is not None and pd.notna(log_likelihood):
            lines.append(f"Log-Likelihood: {log_likelihood:.4f}")

        for attr_name, label in (("AIC_", "AIC"), ("BIC_", "BIC")):
            value = getattr(self.fitter, attr_name, None)
            if value is not None and pd.notna(value):
                lines.append(f"{label}: {value:.4f}")

        if holdout_info and holdout_info.get('active'):
            lines.append(f"")
            lines.append(f"=== Validación Holdout Train/Test ===")
            lines.append(f"Proporción test: {holdout_info['test_proportion']:.0%}")
            lines.append(f"Observaciones entrenamiento: {holdout_info['n_train']}")
            lines.append(f"Observaciones prueba: {holdout_info['n_test']}")
            lines.append(
                f"C-Index (Train): {self._format_c_index_display(holdout_info['c_index_train'], holdout_info.get('c_index_train_ci'), decimals=4)}"
            )
            lines.append(
                f"C-Index (Test):  {self._format_c_index_display(holdout_info['c_index_test'], holdout_info.get('c_index_test_ci'), decimals=4)}"
            )
            c_uno_val = holdout_info.get('c_index_uno')
            if c_uno_val is not None and pd.notna(c_uno_val):
                lines.append(f"C-Index Uno (IPCW): {c_uno_val:.4f} (IPCW = Inverse Probability of Censoring Weighting; corrige por censura)")
            else:
                lines.append("C-Index Uno (IPCW): N/A")
            c_ant_val = holdout_info.get('c_index_antolini')
            if c_ant_val is not None and pd.notna(c_ant_val):
                lines.append(f"C-Index Antolini (Ctd): {c_ant_val:.4f} (AUC dinámica media; generaliza C-index para predicciones dependientes del tiempo)")
            else:
                lines.append("C-Index Antolini (Ctd): N/A")
            tau_val = holdout_info.get('tau')
            if tau_val is not None and np.isfinite(tau_val):
                lines.append(f"τ (tau) utilizado: {tau_val:.2f} (truncamiento IPCW para Uno, Antolini y Brier/IBS)")
            ibs_val = holdout_info.get('ibs')
            if ibs_val is not None and pd.notna(ibs_val):
                lines.append(f"IBS (Integrated Brier Score): {ibs_val:.4f}")
            lines.append(f"Δ (Test - Train): {holdout_info['gap']:.4f}")
            lines.append("Nota: todas las métricas (AIC, BIC, Log-Likelihood, Wald, Cox-Snell) provienen del modelo entrenado solo con el subset de entrenamiento.")
        else:
            try:
                concordance = self.fitter.score(data, scoring_method="concordance_index")
                concordance_ci = self._bootstrap_model_c_index_ci(self.fitter, data, event_col=event_col)
                if concordance is not None and pd.notna(concordance):
                    lines.append(
                        f"C-Index (concordancia, ajuste): {self._format_c_index_display(concordance, concordance_ci, decimals=4)}"
                    )
                    lines.append("Nota: este C-Index se calcula sobre los mismos datos usados para ajustar el modelo; no corresponde a un holdout train/test separado.")
            except Exception:
                lines.append("C-Index (concordancia, ajuste): N/D")

        lines.append("")
        lines.append("=== Validación del ajuste AFT ===")
        lines.append("- AIC / BIC: menor es mejor al comparar distribuciones AFT.")

        try:
            lrt = self.fitter.log_likelihood_ratio_test()
            if lrt is not None:
                lines.append(
                    f"- Likelihood Ratio Test vs modelo nulo: chi2={float(lrt.test_statistic):.4f}, gl={int(lrt.degrees_freedom)}, p={float(lrt.p_value):.4g}"
                )
        except Exception:
            lines.append("- Likelihood Ratio Test vs modelo nulo: N/D")

        shape_label, shape_value, shape_inverse = self._get_shape_parameter_info(self.results)
        if shape_label and pd.notna(shape_value):
            lines.append(
                f"- Parámetro de forma ({shape_label}): {shape_value:.4f}; inverso={shape_inverse:.4f}"
            )

        effect_rows = self._get_aft_lambda_rows()
        if not effect_rows.empty and 'p' in effect_rows.columns:
            effect_param = str(effect_rows['param'].iloc[0]) if 'param' in effect_rows.columns and not effect_rows.empty else 'principal'
            significant_rows = effect_rows[pd.to_numeric(effect_rows['p'], errors='coerce') < 0.05].copy()
            if not significant_rows.empty:
                wald_terms = [
                    f"{row['covariate']} (p={float(row['p']):.4g})"
                    for _, row in significant_rows.head(8).iterrows()
                    if pd.notna(row.get('p'))
                ]
                if wald_terms:
                    lines.append(f"- Wald (p < 0.05 en {effect_param}): " + ", ".join(wald_terms))
                else:
                    lines.append("- Wald: hay covariables significativas, revise la tabla de coeficientes para el detalle.")
            else:
                lines.append(f"- Wald: ninguna covariable {effect_param} alcanzó p < 0.05.")
        else:
            lines.append("- Wald: revise la tabla de coeficientes para los p-valores individuales.")

        cox_snell_info = self._compute_cox_snell_diagnostics()
        if cox_snell_info:
            slope = cox_snell_info.get('slope', np.nan)
            intercept = cox_snell_info.get('intercept', np.nan)
            mae = cox_snell_info.get('mae', np.nan)
            lines.append(
                f"- Cox-Snell: pendiente={slope:.3f}, intercepto={intercept:.3f}, desviación media={mae:.3f}. {cox_snell_info.get('interpretation', '')}"
            )
        else:
            lines.append("- Cox-Snell: no se pudo calcular la validación gráfica con el ajuste actual.")

        if adjustment_note:
            lines.append("")
            lines.append(f"[Aviso] {adjustment_note}")

        for warn in warnings_list:
            lines.append(f"[Aviso] {warn}")

        if self.latest_covariates:
            lines.append("")
            cov_text = ", ".join(self.latest_covariates)
            lines.append(f"Covariables incluidas: {cov_text}")

            if self.latest_formula:
                lines.append(f"Fórmula usada: {self.latest_formula}")

            lines.append("Configuración de covariables:")
            for cov_name in self.latest_covariates:
                cfg = self.variable_configs.get(cov_name, {})
                var_type = cfg.get('type', 'Cuantitativa')
                if var_type == 'Cualitativa':
                    ref_cat = cfg.get('ref_cat', 'N/A')
                    compare_mode = self._get_categorical_compare_display_value(cfg.get('compare_mode', 'all'))
                    lines.append(f"  - {cov_name}: Cualitativa (ref: {ref_cat}; comparación: {compare_mode})")
                elif cfg.get('spline', False):
                    spline_type = cfg.get('spline_type', 'B-spline')
                    spline_degree = cfg.get('spline_degree', 3)
                    spline_num_knots = cfg.get('num_knots', 0)
                    spline_df = cfg.get('spline_df', 'auto')
                    custom_knots = cfg.get('custom_knots', [])
                    extra = f", nodos exactos={custom_knots}" if custom_knots else ""
                    lines.append(
                        f"  - {cov_name}: Cuantitativa con spline ({spline_type}, grado={spline_degree}, nodos={spline_num_knots}, df auto={spline_df}{extra})"
                    )
                else:
                    lines.append(f"  - {cov_name}: Cuantitativa")

        lines.append("")
        lines.append("=== Coeficientes del modelo ===")
        if self.results is not None and not self.results.empty:
            display_df = self.results.copy()
            desired_cols = [c for c in ['exp(coef)', 'exp(coef) lower 95%', 'exp(coef) upper 95%', 'p'] if c in display_df.columns]
            if desired_cols:
                display_df = display_df[desired_cols]
            lines.append(display_df.to_string())
        else:
            lines.append("No hay estimaciones disponibles.")

        return "\n".join(lines)

    def plot_survival_comparison(self):
        self.survival_fig.clear()
        ax = self.survival_fig.add_subplot(111)

        if self.results is None or self.latest_fit_dataframe is None:
            ax.text(0.5, 0.5, "Ajuste un modelo para visualizar las curvas.", ha="center", va="center")
            self.survival_canvas.draw()
            return

        try:
            data = self.latest_fit_dataframe
            duration_col = self.latest_duration_col
            event_col = self.latest_event_col

            kmf = KaplanMeierFitter()
            if event_col:
                kmf.fit(data[duration_col], event_observed=data[event_col], label="Kaplan-Meier observado")
            else:
                kmf.fit(data[duration_col], label="Kaplan-Meier observado")
            kmf.plot(ax=ax, ci_show=False, color="#1f77b4")
        except Exception as exc:
            ax.text(0.5, 0.5, f"No se pudo calcular la curva KM:\n{exc}", ha="center", va="center")
            self.survival_canvas.draw()
            return

        try:
            covariate_data = data[self.latest_covariates]
            surv_df = self.fitter.predict_survival_function(covariate_data)
            avg_surv = surv_df.mean(axis=1)
            ax.plot(avg_surv.index, avg_surv.values, color="#ff7f0e", label="Supervivencia AFT (promedio)")
        except Exception as exc:
            ax.text(0.5, 0.15, f"Predicción promedio no disponible:\n{exc}", transform=ax.transAxes, ha="center", va="center", fontsize=8)

        ax.set_title("Supervivencia observada vs modelo AFT")
        ax.set_xlabel("Tiempo")
        ax.set_ylabel("Probabilidad de supervivencia")
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.2)
        ax.legend(loc="best")
        self._apply_plot_text_overrides(ax, "survival")
        self.survival_canvas.draw()

    def plot_partial_effects(self):
        self.partial_effect_fig.clear()
        ax = self.partial_effect_fig.add_subplot(111)

        if self.results is None or self.latest_fit_dataframe is None or not self.latest_covariates:
            ax.text(0.5, 0.5, "Ajuste un modelo con covariables válidas para ver efectos parciales.", ha="center", va="center")
            self.partial_effect_canvas.draw()
            return

        try:
            available_covariates = self._get_plot_covariate_candidates()
            covariate = self.partial_effect_var.get().strip() if hasattr(self, 'partial_effect_var') else ''
            if covariate not in available_covariates:
                covariate = available_covariates[0] if available_covariates else None
                if covariate and hasattr(self, 'partial_effect_var'):
                    self.partial_effect_var.set(covariate)

            if not covariate:
                raise ValueError("No se encontró covariable válida para graficar.")

            data = self.latest_fit_dataframe
            cov_series = data[covariate].dropna()
            if cov_series.empty:
                raise ValueError("Sin valores válidos para la covariable seleccionada.")

            manual_values_text = self.partial_effect_values_var.get() if hasattr(self, 'partial_effect_values_var') else ''
            manual_values = self._parse_partial_effect_values(covariate, manual_values_text)

            if manual_values:
                values = manual_values
            else:
                if pd.api.types.is_numeric_dtype(cov_series):
                    values = sorted(pd.to_numeric(cov_series, errors='coerce').dropna().unique().tolist())
                    if len(values) > 4:
                        values = [values[0], values[len(values)//2], values[-1]]
                else:
                    values = sorted(cov_series.astype(str).unique().tolist())

            if not values:
                raise ValueError("No se pudieron determinar valores para la covariable seleccionada.")

            baseline_text = self.partial_effect_baseline_var.get() if hasattr(self, 'partial_effect_baseline_var') else ''
            baseline_overrides = self._parse_partial_effect_baseline_overrides(baseline_text, exclude_covariate=covariate)
            base_row = self._build_plot_reference_row(focal_covariate=covariate, overrides=baseline_overrides)

            predict_rows = []
            labels = []
            for value in values:
                row = base_row.copy()
                row[covariate] = value
                predict_rows.append(row)
                labels.append(self._format_plot_value_label(value))

            predict_df = pd.DataFrame(predict_rows, columns=self.latest_covariates)
            survival_df = self.fitter.predict_survival_function(predict_df)
            if survival_df is None or survival_df.empty:
                raise ValueError("El modelo no devolvió curvas de supervivencia para los valores solicitados.")

            for idx in range(survival_df.shape[1]):
                curve_label = f"{covariate}: {labels[idx]}"
                ax.plot(survival_df.index, survival_df.iloc[:, idx], linewidth=2, label=curve_label)

            ax.set_title(f"Efecto parcial de '{covariate}' en la supervivencia")
            ax.set_xlabel("Tiempo")
            ax.set_ylabel("Probabilidad de supervivencia")
            ax.set_ylim(0, 1.05)
            ax.grid(True, alpha=0.2)
            ax.legend(loc='best', fontsize=8)

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

        self._apply_plot_text_overrides(ax, "partial_effect")
        self.partial_effect_canvas.draw()

    def plot_risk_vs_covariate(self):
        self.risk_effect_fig.clear()
        ax = self.risk_effect_fig.add_subplot(111)

        if self.results is None or self.latest_fit_dataframe is None or not self.latest_covariates:
            ax.text(0.5, 0.5, "Ajuste un modelo con covariables válidas para ver el riesgo.", ha="center", va="center")
            self.risk_effect_canvas.draw()
            return

        data = self.latest_fit_dataframe
        duration_col = self.latest_duration_col

        try:
            available_covariates = self._get_plot_covariate_candidates()
            covariate = self.risk_effect_var.get().strip() if hasattr(self, 'risk_effect_var') else ''
            if covariate not in available_covariates:
                covariate = available_covariates[0] if available_covariates else None
                if covariate and hasattr(self, 'risk_effect_var'):
                    self.risk_effect_var.set(covariate)

            if not covariate:
                raise ValueError("No se encontró una covariable para evaluar el riesgo.")

            cov_series = data[covariate].dropna()
            if cov_series.empty:
                raise ValueError("La covariable seleccionada no tiene valores disponibles.")

            is_numeric = pd.api.types.is_numeric_dtype(cov_series)
            if is_numeric:
                cov_values = np.linspace(float(cov_series.min()), float(cov_series.max()), num=60)
                cov_values = np.unique(cov_values)
                if cov_values.size < 2:
                    val = float(cov_series.iloc[0]) if not cov_series.empty else 0.0
                    cov_values = np.array([val - 1e-6, val + 1e-6])
            else:
                cov_values = cov_series.astype(str).unique()

            base_row = {}
            for col in self.latest_covariates:
                col_series = data[col]
                if col == covariate:
                    continue
                if pd.api.types.is_numeric_dtype(col_series):
                    mean_val = col_series.mean()
                    if pd.isna(mean_val):
                        mean_val = col_series.median()
                    if pd.isna(mean_val):
                        mean_val = col_series.dropna().iloc[0] if not col_series.dropna().empty else 0.0
                    base_row[col] = float(mean_val)
                else:
                    modes = col_series.dropna().mode()
                    if not modes.empty:
                        base_row[col] = modes.iloc[0]
                    else:
                        non_null = col_series.dropna()
                        base_row[col] = non_null.iloc[0] if not non_null.empty else ""

            predict_rows = []
            labels = []
            for value in cov_values:
                row = base_row.copy()
                row[covariate] = value
                predict_rows.append(row)
                labels.append(value)

            predict_df = pd.DataFrame(predict_rows, columns=self.latest_covariates)

            eval_time = np.nan
            if hasattr(self, 'risk_time_var'):
                try:
                    eval_time = float(str(self.risk_time_var.get()).strip())
                except (TypeError, ValueError):
                    eval_time = np.nan

            if not np.isfinite(eval_time) or eval_time <= 0:
                eval_time = float(np.nanmedian(data[duration_col])) if duration_col else np.nan
            if not np.isfinite(eval_time) or eval_time <= 0:
                eval_time = float(np.nanmean(data[duration_col])) if duration_col else np.nan
            if not np.isfinite(eval_time) or eval_time <= 0:
                eval_time = float(np.nanmax(data[duration_col])) if duration_col else 1.0
            if not np.isfinite(eval_time) or eval_time <= 0:
                eval_time = 1.0

            if hasattr(self, 'risk_time_var'):
                self.risk_time_var.set(f"{eval_time:.2f}")

            survival_df = self.fitter.predict_survival_function(predict_df)
            if survival_df.empty:
                raise ValueError("El modelo no devolvió curvas de supervivencia para la covariable seleccionada.")

            times = survival_df.index.to_numpy(dtype=float)
            eval_time = float(np.clip(eval_time, times[0], times[-1]))

            survival_at_t = []
            for idx in range(survival_df.shape[1]):
                column_values = survival_df.iloc[:, idx].to_numpy(dtype=float)
                survival_at_t.append(np.interp(eval_time, times, column_values))

            survival_at_t = np.asarray(survival_at_t, dtype=float)
            risk_scores = 1.0 - survival_at_t

            if is_numeric:
                ax.plot(cov_values, risk_scores, color="#d62728", linewidth=2, label="Riesgo estimado")
                ax.fill_between(cov_values, risk_scores, color="#d62728", alpha=0.15)
                ax.set_xlabel(covariate)

                knot_values, knot_source = self._get_spline_knots_for_display(covariate)
                if knot_values:
                    knot_positions = np.asarray(knot_values, dtype=float)
                    knot_risk = np.interp(knot_positions, cov_values.astype(float), risk_scores.astype(float))
                    for idx, knot in enumerate(knot_positions):
                        ax.axvline(
                            knot,
                            color="#9467bd",
                            linestyle="--",
                            linewidth=1.2,
                            alpha=0.85,
                            label="Nodos spline" if idx == 0 else None,
                        )
                    ax.scatter(knot_positions, knot_risk, color="#9467bd", s=40, zorder=4)
                    knot_text = ", ".join(f"{float(knot):.3g}" for knot in knot_positions)
                    source_text = "exactos" if knot_source == 'exactos' else "auto"
                    ax.text(
                        0.02,
                        0.98,
                        f"Nodos {source_text}: {knot_text}",
                        transform=ax.transAxes,
                        ha="left",
                        va="top",
                        fontsize=8,
                        bbox=dict(boxstyle="round", facecolor="white", alpha=0.75, edgecolor="#bbbbbb"),
                    )
            else:
                x_positions = np.arange(len(labels))
                ax.bar(x_positions, risk_scores, color="#d62728", alpha=0.8)
                ax.set_xticks(x_positions)
                ax.set_xticklabels([str(val) for val in labels], rotation=15)
                ax.set_xlabel(covariate)

            ax.set_ylabel(f"Riesgo estimado (1 - S(t)) con t={eval_time:.2f}")
            ax.set_title(f"Impacto de '{covariate}' sobre el riesgo acumulado")
            ax.set_ylim(0, min(1.05, max(1.0, risk_scores.max() + 0.05)))
            ax.grid(True, alpha=0.2)
            handles, legend_labels = ax.get_legend_handles_labels()
            if legend_labels:
                ax.legend(loc="best", fontsize=8)
        except Exception as exc:
            ax.text(0.5, 0.5, f"No se pudo calcular el riesgo por covariable:\n{exc}", ha="center", va="center")

        self._apply_plot_text_overrides(ax, "risk_effect")
        self.risk_effect_canvas.draw()

    def plot_cumulative_hazard(self):
        self.hazard_fig.clear()
        ax = self.hazard_fig.add_subplot(111)

        if self.results is None or self.latest_fit_dataframe is None:
            ax.text(0.5, 0.5, "Ajuste un modelo para ver la validación Cox-Snell.", ha="center", va="center")
            self.hazard_canvas.draw()
            return

        try:
            diagnostic = self._compute_cox_snell_diagnostics()
            if not diagnostic:
                raise ValueError("No fue posible derivar residuos de Cox-Snell con este ajuste.")

            x_values = diagnostic['x']
            y_values = diagnostic['y']
            max_axis = max(float(np.nanmax(x_values)), float(np.nanmax(y_values)), 1.0)

            ax.plot(x_values, y_values, color="#d62728", linewidth=2, label="Nelson-Aalen de residuos")
            ax.plot([0, max_axis], [0, max_axis], linestyle="--", color="#1f77b4", label="Diagonal ideal 45°")
            ax.set_title("Validación Cox-Snell del modelo AFT")
            ax.set_xlabel("Residuo de Cox-Snell")
            ax.set_ylabel("Riesgo acumulado estimado")
            ax.grid(True, alpha=0.25)
            ax.legend(loc="best")

            slope = diagnostic.get('slope', np.nan)
            mae = diagnostic.get('mae', np.nan)
            interpretation = diagnostic.get('interpretation', '')
            if pd.notna(slope) and pd.notna(mae):
                ax.text(
                    0.02,
                    0.98,
                    f"Pendiente~{slope:.2f}\nDesv.media~{mae:.2f}\n{interpretation}",
                    transform=ax.transAxes,
                    va="top",
                    ha="left",
                    fontsize=8,
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="#bbbbbb"),
                )
        except Exception as exc:
            ax.text(0.5, 0.5, f"No se pudo calcular la validación Cox-Snell:\n{exc}", ha="center", va="center")

        self._apply_plot_text_overrides(ax, "cumulative_hazard")
        self.hazard_canvas.draw()

    def plot_brier_curve(self):
        self.brier_fig.clear()
        ax = self.brier_fig.add_subplot(111)

        if not isinstance(self.latest_brier_df, pd.DataFrame) or self.latest_brier_df.empty:
            ax.text(0.5, 0.5, "Activa holdout para ver la curva de Brier / IBS del modelo AFT.", ha="center", va="center")
            self._apply_plot_text_overrides(ax, "brier")
            self.brier_canvas.draw()
            return

        brier_df = self.latest_brier_df.dropna(subset=["time", "brier_score"]).copy()
        if brier_df.empty:
            ax.text(0.5, 0.5, "No hay suficientes puntos para graficar Brier / IBS.", ha="center", va="center")
            self._apply_plot_text_overrides(ax, "brier")
            self.brier_canvas.draw()
            return

        brier_df = brier_df.sort_values("time")
        times = brier_df["time"].to_numpy(dtype=float)
        scores = brier_df["brier_score"].to_numpy(dtype=float)
        ax.plot(times, scores, color="#8b5cf6", linewidth=2.2, label="Brier(t)")
        ax.fill_between(times, scores, 0, color="#8b5cf6", alpha=0.16)

        if self.latest_eval_time is not None and np.isfinite(self.latest_eval_time):
            ax.axvline(float(self.latest_eval_time), color="#475569", linestyle="--", linewidth=1.2, label=f"t≈{float(self.latest_eval_time):.2f}")

        ibs_val = None
        if self.saved_models and self.active_saved_model_index is not None and 0 <= self.active_saved_model_index < len(self.saved_models):
            ibs_val = self.saved_models[self.active_saved_model_index].get('metrics', {}).get('ibs')

        title = "Curva de Brier AFT"
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

    def plot_forest(self):
        self.forest_fig.clear()
        ax = self.forest_fig.add_subplot(111)

        if self.results is None or self.results.empty:
            ax.text(0.5, 0.5, "Ajuste un modelo para ver el Forest Plot.", ha="center", va="center")
            self.forest_canvas.draw()
            return

        try:
            effect_rows = self._get_aft_lambda_rows()

            if effect_rows.empty:
                raise ValueError("No hay coeficientes principales distintos al intercepto.")

            effect_rows = effect_rows.sort_values('exp(coef)')
            y_pos = np.arange(len(effect_rows))
            hr = effect_rows['exp(coef)'].to_numpy(dtype=float)
            lower = effect_rows['exp(coef) lower 95%'].to_numpy(dtype=float)
            upper = effect_rows['exp(coef) upper 95%'].to_numpy(dtype=float)
            param_label = str(effect_rows['param'].iloc[0]) if 'param' in effect_rows.columns and not effect_rows.empty else 'principal'

            ax.errorbar(hr, y_pos, xerr=[hr - lower, upper - hr], fmt='o', capsize=4, color='black', ecolor='#4d4d4d')
            ax.set_yticks(y_pos)
            ax.set_yticklabels(effect_rows['covariate'])
            ax.axvline(1.0, color='gray', linestyle='--', linewidth=1)
            ax.invert_yaxis()
            ax.set_xlabel('Exp(coef) (ratio de tiempo)')
            ax.set_title(f'Forest Plot - Parámetro {param_label}')
            ax.grid(True, axis='x', alpha=0.2)
        except Exception as exc:
            ax.text(0.5, 0.5, f"No se pudo generar el Forest Plot:\n{exc}", ha='center', va='center')

        self._apply_plot_text_overrides(ax, "forest")
        self.forest_canvas.draw()

    def _reset_results_view(self):
        self.results = None
        self.fitter = None
        self.latest_fit_dataframe = None
        self.latest_duration_col = None
        self.latest_event_col = None
        self.latest_covariates = []
        self.latest_formula = None
        self.latest_brier_df = pd.DataFrame()
        self.latest_eval_time = None
        self.active_saved_model_index = None
        self._plot_text_overrides = {}
        if hasattr(self, 'partial_effect_var'):
            self.partial_effect_var.set('')
        if hasattr(self, 'partial_effect_values_var'):
            self.partial_effect_values_var.set('')
        if hasattr(self, 'partial_effect_baseline_var'):
            self.partial_effect_baseline_var.set('')
        if hasattr(self, 'risk_effect_var'):
            self.risk_effect_var.set('')
        if hasattr(self, 'risk_time_var'):
            self.risk_time_var.set('')
        if hasattr(self, 'saved_models_status_var'):
            self.saved_models_status_var.set(f"Modelos guardados: {len(self.saved_models)} | Activo: ninguno")
        if hasattr(self, 'results_text'):
            self.results_text.delete("1.0", tk.END)
        if hasattr(self, 'survival_fig'):
            self.survival_fig.clear()
            self.survival_canvas.draw()
        if hasattr(self, 'partial_effect_fig'):
            self.partial_effect_fig.clear()
            self.partial_effect_canvas.draw()
        if hasattr(self, 'risk_effect_fig'):
            self.risk_effect_fig.clear()
            self.risk_effect_canvas.draw()
        if hasattr(self, 'hazard_fig'):
            self.hazard_fig.clear()
            self.hazard_canvas.draw()
        if hasattr(self, 'forest_fig'):
            self.forest_fig.clear()
            self.forest_canvas.draw()
        if hasattr(self, 'brier_fig'):
            self.brier_fig.clear()
            self.brier_canvas.draw()

    def _describe_shared_source(self, metadata):
        if metadata and metadata.get('source_path'):
            try:
                return os.path.basename(metadata['source_path'])
            except Exception:
                return metadata.get('source_path') or "Archivo compartido"
        return "Archivo compartido"

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
            self._reset_results_view()
            if self.filter_component:
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

        if self.filter_component:
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

class VariableConfigDialog(Toplevel):
    def __init__(self, parent, app_instance, selected_vars):
        super().__init__(parent)
        self.transient(parent)
        self.grab_set()
        self.title("Configuración Detallada de Variables")
        self.app_instance = app_instance
        self.selected_vars = selected_vars
        self.row_configs = {}

        main_frame = ttk.Frame(self, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        scrolled_frame = ScrolledFrame(main_frame)
        scrolled_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        self.controls_frame = scrolled_frame.interior

        for var_name in self.selected_vars:
            self.row_configs[var_name] = {}
            
            row_labelframe = ttk.LabelFrame(self.controls_frame, text=var_name, padding="10")
            row_labelframe.pack(fill=tk.X, pady=5, padx=5)

            # Variable Type
            ttk.Label(row_labelframe, text="Tipo:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
            type_var = tk.StringVar(value="Cuantitativa")
            self.row_configs[var_name]['type_var'] = type_var
            
            rb_cuant = ttk.Radiobutton(row_labelframe, text="Cuantitativa", variable=type_var, value="Cuantitativa", command=lambda v=var_name: self._toggle_controls(v))
            rb_cuant.grid(row=0, column=1, sticky=tk.W, padx=2)
            
            rb_cual = ttk.Radiobutton(row_labelframe, text="Cualitativa", variable=type_var, value="Cualitativa", command=lambda v=var_name: self._toggle_controls(v))
            rb_cual.grid(row=0, column=2, sticky=tk.W, padx=2)

            # Reference Category
            ttk.Label(row_labelframe, text="Ref. Cat.:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
            ref_combo = ttk.Combobox(row_labelframe, state="disabled", width=15)
            ref_combo.grid(row=1, column=1, columnspan=2, sticky=tk.EW, padx=5)
            self.row_configs[var_name]['ref_combo'] = ref_combo

            ttk.Label(row_labelframe, text="Comp. categórica:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
            cat_compare_mode_var = tk.StringVar(value=self.app_instance._get_default_categorical_compare_display())
            self.row_configs[var_name]['cat_compare_mode_var'] = cat_compare_mode_var
            cat_compare_mode_combo = ttk.Combobox(
                row_labelframe,
                textvariable=cat_compare_mode_var,
                values=list(self.app_instance.categorical_compare_display_map.values()),
                state="disabled",
                width=28,
            )
            cat_compare_mode_combo.grid(row=2, column=1, columnspan=2, sticky=tk.EW, padx=5)
            self.row_configs[var_name]['cat_compare_mode_combo'] = cat_compare_mode_combo

            # Spline
            ttk.Label(row_labelframe, text="Spline:").grid(row=3, column=0, sticky=tk.W, padx=5, pady=2)
            spline_var = tk.BooleanVar(value=False)
            self.row_configs[var_name]['spline_var'] = spline_var
            cb_spline = ttk.Checkbutton(row_labelframe, text="Usar", variable=spline_var, command=lambda v=var_name: self._toggle_controls(v))
            cb_spline.grid(row=3, column=1, sticky=tk.W, padx=5)

            ttk.Label(row_labelframe, text="  Tipo spline:").grid(row=4, column=0, sticky=tk.W, padx=15, pady=2)
            spline_type_var = tk.StringVar(value=self.app_instance._get_default_spline_display())
            self.row_configs[var_name]['spline_type_var'] = spline_type_var
            spline_type_combo = ttk.Combobox(
                row_labelframe,
                textvariable=spline_type_var,
                values=list(self.app_instance.spline_type_display_map.values()),
                state="disabled",
                width=24,
            )
            spline_type_combo.grid(row=4, column=1, columnspan=2, sticky=tk.EW, padx=5)
            spline_type_combo.bind("<<ComboboxSelected>>", lambda _event, v=var_name: self._toggle_controls(v))
            self.row_configs[var_name]['spline_type_combo'] = spline_type_combo

            ttk.Label(row_labelframe, text="  Nodos internos (0 = polinómico):").grid(row=5, column=0, sticky=tk.W, padx=15, pady=2)
            spline_knots_var = tk.IntVar(value=0)
            self.row_configs[var_name]['spline_knots_var'] = spline_knots_var
            spline_knots_spinbox = ttk.Spinbox(row_labelframe, from_=0, to=15, textvariable=spline_knots_var, width=5, state="disabled")
            spline_knots_spinbox.grid(row=5, column=1, sticky=tk.W, padx=5)
            self.row_configs[var_name]['spline_knots_spinbox'] = spline_knots_spinbox

            ttk.Label(row_labelframe, text="  Grado (1=recta, 2=cuadrática, 3=cúbica):").grid(row=6, column=0, sticky=tk.W, padx=15, pady=2)
            spline_degree_var = tk.IntVar(value=3)
            self.row_configs[var_name]['spline_degree_var'] = spline_degree_var
            spline_degree_spinbox = ttk.Spinbox(row_labelframe, from_=1, to=5, textvariable=spline_degree_var, width=5, state="disabled")
            spline_degree_spinbox.grid(row=6, column=1, sticky=tk.W, padx=5)
            self.row_configs[var_name]['spline_degree_spinbox'] = spline_degree_spinbox

            ttk.Label(row_labelframe, text="  Nodos manuales exactos (coma):").grid(row=7, column=0, sticky=tk.W, padx=15, pady=2)
            spline_custom_knots_var = tk.StringVar(value="")
            self.row_configs[var_name]['spline_custom_knots_var'] = spline_custom_knots_var
            spline_custom_knots_entry = ttk.Entry(row_labelframe, textvariable=spline_custom_knots_var, width=26, state="disabled")
            spline_custom_knots_entry.grid(row=7, column=1, columnspan=2, sticky=tk.EW, padx=5)
            self.row_configs[var_name]['spline_custom_knots_entry'] = spline_custom_knots_entry

            ttk.Label(row_labelframe, text="  DF automático:").grid(row=8, column=0, sticky=tk.W, padx=15, pady=2)
            spline_df_display_var = tk.StringVar(value="4")
            self.row_configs[var_name]['spline_df_display_var'] = spline_df_display_var
            ttk.Label(row_labelframe, textvariable=spline_df_display_var, foreground="navy").grid(row=8, column=1, sticky=tk.W, padx=5)

            for tk_var in (spline_knots_var, spline_degree_var, spline_custom_knots_var):
                tk_var.trace_add("write", lambda *_args, v=var_name: self._update_auto_df(v))

            # Load existing config
            config = self.app_instance.variable_configs.get(var_name, {})
            if self.app_instance.data is not None and var_name in self.app_instance.data.columns:
                inferred_type = "Cuantitativa" if pd.api.types.is_numeric_dtype(self.app_instance.data[var_name]) else "Cualitativa"
            else:
                inferred_type = "Cuantitativa"

            type_var.set(config.get('type', inferred_type))
            cat_compare_mode_var.set(
                self.app_instance._get_categorical_compare_display_value(config.get('compare_mode', 'all'))
            )
            if type_var.get() == 'Cualitativa' and self.app_instance.data is not None:
                unique_vals = sorted(self.app_instance.data[var_name].astype(str).unique().tolist())
                ref_combo['values'] = unique_vals
                ref_combo.set(config.get('ref_cat', unique_vals[0] if unique_vals else ''))

            if config.get('spline', False):
                spline_var.set(True)
                spline_type_var.set(self.app_instance._get_spline_display_value(config.get('spline_type', 'B-spline')))
                spline_knots_var.set(config.get('num_knots', 0))
                spline_degree_var.set(config.get('spline_degree', 3))
                existing_custom_knots = config.get('custom_knots') or []
                if existing_custom_knots:
                    spline_custom_knots_var.set(", ".join(f"{val:g}" for val in existing_custom_knots))

            self._toggle_controls(var_name)

        buttons_frame = ttk.Frame(main_frame)
        buttons_frame.pack(fill=tk.X, pady=(10,0))
        ttk.Button(buttons_frame, text="OK/Aplicar", command=self.apply_configurations).pack(side=tk.RIGHT, padx=5)
        ttk.Button(buttons_frame, text="Cancelar", command=self.destroy).pack(side=tk.RIGHT)

    def _update_auto_df(self, var_name):
        config = self.row_configs.get(var_name)
        if not config:
            return

        var_is_quantitative = config['type_var'].get() == "Cuantitativa"
        if not var_is_quantitative:
            config['spline_df_display_var'].set("N/A")
            return

        spline_type = self.app_instance._get_spline_internal_type(config['spline_type_var'].get())

        try:
            spline_degree = max(1, int(config['spline_degree_var'].get() or 1))
        except (ValueError, tk.TclError):
            spline_degree = 1

        try:
            spline_num_knots = max(0, int(config['spline_knots_var'].get() or 0))
        except (ValueError, tk.TclError):
            spline_num_knots = 0

        custom_knots = self.app_instance._parse_manual_knots(config['spline_custom_knots_var'].get())
        derived_df = self.app_instance._derive_spline_df(
            spline_type,
            spline_degree,
            spline_num_knots,
            custom_knots,
        )
        config['spline_df_display_var'].set(str(derived_df))

    def _toggle_controls(self, var_name):
        config = self.row_configs[var_name]
        var_is_quantitative = config['type_var'].get() == "Cuantitativa"

        config['ref_combo'].config(state="readonly" if not var_is_quantitative else "disabled")
        config['cat_compare_mode_combo'].config(state="readonly" if not var_is_quantitative else "disabled")
        if var_is_quantitative:
            config['ref_combo'].set("")
        else:
            if self.app_instance.data is not None:
                unique_vals = sorted(self.app_instance.data[var_name].astype(str).unique().tolist())
                config['ref_combo']['values'] = unique_vals
                if self.app_instance.variable_configs.get(var_name, {}).get('ref_cat') in unique_vals:
                    config['ref_combo'].set(self.app_instance.variable_configs[var_name]['ref_cat'])
                elif unique_vals:
                    config['ref_combo'].set(unique_vals[0])

        if not var_is_quantitative:
            config['spline_var'].set(False)

        use_spline = var_is_quantitative and bool(config['spline_var'].get())
        spline_type = self.app_instance._get_spline_internal_type(config['spline_type_var'].get())
        degree_enabled = use_spline and spline_type == "B-spline"

        config['spline_type_combo'].config(state="readonly" if use_spline else "disabled")
        config['spline_knots_spinbox'].config(state="normal" if use_spline else "disabled")
        config['spline_degree_spinbox'].config(state="normal" if degree_enabled else "disabled")
        config['spline_custom_knots_entry'].config(state="normal" if use_spline else "disabled")

        self._update_auto_df(var_name)

    def apply_configurations(self):
        for var_name, config_widgets in self.row_configs.items():
            var_type = config_widgets['type_var'].get()
            new_config = {'type': var_type}

            if var_type == 'Cualitativa':
                ref_cat = config_widgets['ref_combo'].get().strip()
                if ref_cat:
                    new_config['ref_cat'] = ref_cat
                new_config['compare_mode'] = self.app_instance._get_categorical_compare_internal_mode(
                    config_widgets['cat_compare_mode_var'].get()
                )
                new_config['spline'] = False
            else:
                use_spline = bool(config_widgets['spline_var'].get())
                new_config['spline'] = use_spline

                if use_spline:
                    spline_type = self.app_instance._get_spline_internal_type(config_widgets['spline_type_var'].get())
                    try:
                        spline_degree = max(1, int(config_widgets['spline_degree_var'].get() or 1))
                    except (ValueError, tk.TclError):
                        spline_degree = 1

                    try:
                        requested_knots = max(0, int(config_widgets['spline_knots_var'].get() or 0))
                    except (ValueError, tk.TclError):
                        requested_knots = 0

                    custom_knots = self.app_instance._parse_manual_knots(config_widgets['spline_custom_knots_var'].get())
                    effective_num_knots = len(custom_knots) if custom_knots else requested_knots
                    derived_df = self.app_instance._derive_spline_df(
                        spline_type,
                        spline_degree,
                        effective_num_knots,
                        custom_knots,
                    )

                    new_config.update({
                        'spline': True,
                        'spline_type': spline_type,
                        'num_knots': effective_num_knots,
                        'spline_degree': spline_degree,
                        'custom_knots': custom_knots,
                        'spline_df': derived_df,
                    })

            self.app_instance.variable_configs[var_name] = new_config

        self.destroy()