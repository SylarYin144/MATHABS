#!/usr/bin/env python
# -*- coding: utf-8 -*-

# --- Standard Python Imports ---
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog, StringVar, BooleanVar, DoubleVar, IntVar, Listbox, MULTIPLE, SINGLE, BROWSE, Toplevel, Frame, Label, Entry, Button, Checkbutton, Radiobutton
from tkinter import scrolledtext
import os
import pickle
import warnings
import traceback
import csv
import json
import re
import math
import sys

# --- Third-party Library Imports ---
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import scipy.stats
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from patsy import dmatrices, dmatrix, bs
from patsy.builtins import Q, Treatment

# --- Custom Module Imports ---
try:
    from MATLAB_filter_component import FilterComponent
    FILTER_COMPONENT_AVAILABLE = True
except ImportError:
    FilterComponent = None
    FILTER_COMPONENT_AVAILABLE = False
    print("INFO: MATLAB_filter_component not found. Advanced filters may be unavailable.")

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# --- Auxiliary Functions ---

def format_p_value(p_val, threshold=0.0001):
    if pd.isna(p_val) or not isinstance(p_val, (float, np.floating, int)):
        return "N/A"
    if p_val < threshold:
        return f"{p_val:.2e}"
    else:
        return f"{p_val:.4f}"

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
        self.canvas.bind('<Enter>', self._bind_mousewheel_events)
        self.canvas.bind('<Leave>', self._unbind_mousewheel_events)

    def _on_interior_configure(self, event):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

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

# --- Main Application Class ---

class LinearRegressionApp(ttk.Frame):
    def __init__(self, parent_notebook_tab, main_app_instance=None):
        super().__init__(parent_notebook_tab)
        self.pack(fill=tk.BOTH, expand=True)
        self.main_app = main_app_instance
        self.parent_for_dialogs = self.winfo_toplevel()

        # Data and configuration variables
        self.raw_data = None
        self.data = None
        self.shared_dataset_active = False
        self.dependent_var = ""
        self.independent_vars = []
        self.results = None
        self.variable_configs = {}

        # UI Control Variables
        self.var_selection_method_var = StringVar(value="Ninguno (usar todas)")
        self.p_enter_var = DoubleVar(value=0.05)
        self.p_remove_var = DoubleVar(value=0.05)
        self.penalization_method_var = StringVar(value="Ninguna")
        self.penalizer_strength_var = DoubleVar(value=0.1)
        self.l1_ratio_for_elasticnet_var = DoubleVar(value=0.5)
        self.covariate_scaling_method_var = StringVar(value="Ninguna")

        # Notebook for tabs
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)

        # Tab 1: Load, Filter, and Preprocess
        self.tab_frame_preproc = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(self.tab_frame_preproc, text='  1. Carga y Preprocesamiento  ')
        self.tab_frame_preproc_content = ScrolledFrame(self.tab_frame_preproc)
        self.tab_frame_preproc_content.pack(fill=tk.BOTH, expand=True)

        # Tab 2: Linear Modeling
        self.tab_frame_modeling = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(self.tab_frame_modeling, text='  2. Modelado Lineal  ')
        self.tab_frame_modeling_content = ScrolledFrame(self.tab_frame_modeling)
        self.tab_frame_modeling_content.pack(fill=tk.BOTH, expand=True)

        # Tab 3: Results
        self.tab_frame_results = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(self.tab_frame_results, text='  3. Resultados  ')
        self.tab_frame_results_content = ScrolledFrame(self.tab_frame_results)
        self.tab_frame_results_content.pack(fill=tk.BOTH, expand=True)

        # Tab 4: Graphs
        self.tab_frame_graphs = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(self.tab_frame_graphs, text='  4. Gráficas  ')
        self.tab_frame_graphs_content = ScrolledFrame(self.tab_frame_graphs)
        self.tab_frame_graphs_content.pack(fill=tk.BOTH, expand=True)

        # Tab 5: Log
        self.tab_frame_log = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(self.tab_frame_log, text='  Log  ')
        self.log_text_widget = scrolledtext.ScrolledText(self.tab_frame_log, wrap=tk.WORD, height=10, state=tk.DISABLED, font=("Courier New", 9))
        self.log_text_widget.pack(fill=tk.BOTH, expand=True)
        self.log_text_widget.tag_config("INFO", foreground="black")
        self.log_text_widget.tag_config("DEBUG", foreground="gray")
        self.log_text_widget.tag_config("WARN", foreground="orange")
        self.log_text_widget.tag_config("ERROR", foreground="red")
        self.log_text_widget.tag_config("SUCCESS", foreground="green")

        self.create_preproc_controls()
        self.create_modeling_controls()
        self.create_results_controls()
        self.create_graphs_controls()

        self.log("Interfaz de LinearRegressionApp inicializada.", "INFO")

    def log(self, message, level="INFO"):
        self.log_text_widget.config(state=tk.NORMAL)
        self.log_text_widget.insert(tk.END, f"[{level}] {message}\n", level)
        self.log_text_widget.config(state=tk.DISABLED)
        self.log_text_widget.see(tk.END)

    def create_preproc_controls(self):
        p_content = self.tab_frame_preproc_content.interior
        
        # File Loading
        frame_load = ttk.LabelFrame(p_content, text="Carga de Archivo")
        frame_load.pack(fill=tk.X, padx=10, pady=10)
        btn_load = ttk.Button(frame_load, text="Seleccionar Archivo (.xlsx, .csv)", command=self.load_file)
        btn_load.pack(side=tk.LEFT, padx=10, pady=10)
        self.label_file_info = ttk.Label(frame_load, text="Ningún archivo cargado.")
        self.label_file_info.pack(side=tk.LEFT, padx=10, pady=10)
        self.data_source_status_var = StringVar(value="Origen: sin datos")
        self.label_data_source_status = ttk.Label(frame_load, textvariable=self.data_source_status_var, foreground="#8B5E00")
        self.label_data_source_status.pack(side=tk.LEFT, padx=10, pady=10)

        # Filters
        if FILTER_COMPONENT_AVAILABLE:
            frame_filters = ttk.LabelFrame(p_content, text="Filtros de Datos")
            frame_filters.pack(fill=tk.X, padx=10, pady=10)
            self.filter_component = FilterComponent(frame_filters, log_callback=self.log)
            self.filter_component.pack(fill=tk.BOTH, expand=True)


        # Variable Selection
        frame_vars = ttk.LabelFrame(p_content, text="Selección de Variables")
        frame_vars.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Label(frame_vars, text="Variable Dependiente (Y):").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.combo_dep_var = ttk.Combobox(frame_vars, state="readonly")
        self.combo_dep_var.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

        ttk.Label(frame_vars, text="Variables Independientes (X):").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.listbox_indep_vars = Listbox(frame_vars, selectmode=MULTIPLE, height=8)
        self.listbox_indep_vars.grid(row=1, column=1, padx=5, pady=5, sticky="ew")
        
        btn_config_vars = ttk.Button(frame_vars, text="Configurar Variables Seleccionadas...", command=self.open_variable_config_dialog)
        btn_config_vars.grid(row=2, column=1, padx=5, pady=10, sticky="e")

        frame_vars.columnconfigure(1, weight=1)

    def open_variable_config_dialog(self):
        selected_indices = self.listbox_indep_vars.curselection()
        if not selected_indices:
            messagebox.showwarning("Sin Selección", "Seleccione una o más variables para configurar.")
            return
        
        selected_vars = [self.listbox_indep_vars.get(i) for i in selected_indices]
        VariableConfigDialog(self, self, selected_vars)

    def create_modeling_controls(self):
        m_content = self.tab_frame_modeling_content.interior
        
        frame_model_options = ttk.LabelFrame(m_content, text="Opciones de Modelado")
        frame_model_options.pack(fill=tk.X, padx=10, pady=10)

        # Variable Selection Method
        ttk.Label(frame_model_options, text="Selección de Variables:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        combo_var_selection = ttk.Combobox(frame_model_options, textvariable=self.var_selection_method_var,
                                           values=["Ninguno (usar todas)", "Forward", "Backward", "Stepwise"],
                                           state="readonly")
        combo_var_selection.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

        # P-value thresholds
        ttk.Label(frame_model_options, text="P-valor para entrar:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        entry_p_enter = ttk.Entry(frame_model_options, textvariable=self.p_enter_var, width=10)
        entry_p_enter.grid(row=1, column=1, padx=5, pady=5, sticky="w")

        ttk.Label(frame_model_options, text="P-valor para salir:").grid(row=2, column=0, padx=5, pady=5, sticky="w")
        entry_p_remove = ttk.Entry(frame_model_options, textvariable=self.p_remove_var, width=10)
        entry_p_remove.grid(row=2, column=1, padx=5, pady=5, sticky="w")

        # Penalization
        ttk.Label(frame_model_options, text="Penalización:").grid(row=3, column=0, padx=5, pady=5, sticky="w")
        combo_penalization = ttk.Combobox(frame_model_options, textvariable=self.penalization_method_var,
                                            values=["Ninguna", "L1 (Lasso)", "L2 (Ridge)", "ElasticNet"],
                                            state="readonly")
        combo_penalization.grid(row=3, column=1, padx=5, pady=5, sticky="ew")

        ttk.Label(frame_model_options, text="Fuerza de Penalización (alpha):").grid(row=4, column=0, padx=5, pady=5, sticky="w")
        entry_alpha = ttk.Entry(frame_model_options, textvariable=self.penalizer_strength_var, width=10)
        entry_alpha.grid(row=4, column=1, padx=5, pady=5, sticky="w")

        ttk.Label(frame_model_options, text="Ratio L1 (ElasticNet):").grid(row=5, column=0, padx=5, pady=5, sticky="w")
        entry_l1_ratio = ttk.Entry(frame_model_options, textvariable=self.l1_ratio_for_elasticnet_var, width=10)
        entry_l1_ratio.grid(row=5, column=1, padx=5, pady=5, sticky="w")

        # Scaling
        ttk.Label(frame_model_options, text="Escalado de Covariables:").grid(row=6, column=0, padx=5, pady=5, sticky="w")
        combo_scaling = ttk.Combobox(frame_model_options, textvariable=self.covariate_scaling_method_var,
                                       values=["Ninguna", "StandardScaler", "MinMaxScaler"],
                                       state="readonly")
        combo_scaling.grid(row=6, column=1, padx=5, pady=5, sticky="ew")

        # Univariate Analysis Checkbox
        self.univariate_analysis_var = BooleanVar(value=False)
        chk_univariate = ttk.Checkbutton(frame_model_options, text="Análisis Univariado (genera Forest Plot)", variable=self.univariate_analysis_var)
        chk_univariate.grid(row=7, column=0, columnspan=2, padx=5, pady=10, sticky="w")

        # Run Model Button
        btn_run_model = ttk.Button(m_content, text="Ejecutar Modelo de Regresión Lineal", command=self.run_linear_regression)
        btn_run_model.pack(pady=20, padx=10, fill=tk.X)

    def create_results_controls(self):
        r_content = self.tab_frame_results_content.interior
        
        # Summary Text
        self.results_text = scrolledtext.ScrolledText(r_content, wrap=tk.WORD, height=20, state=tk.DISABLED)
        self.results_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    def create_graphs_controls(self):
        g_content = self.tab_frame_graphs_content.interior
        
        graphs_notebook = ttk.Notebook(g_content)
        graphs_notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Diagnostics Tab
        diagnostics_frame = ttk.Frame(graphs_notebook)
        graphs_notebook.add(diagnostics_frame, text="Diagnósticos del Modelo")

        # --- Plot Selection Controls ---
        plot_controls_frame = ttk.Frame(diagnostics_frame)
        plot_controls_frame.pack(pady=5, padx=10, fill=tk.X)

        # Radio buttons for plot type
        self.plot_choice_var = StringVar(value="residuals")
        ttk.Radiobutton(plot_controls_frame, text="Residuos vs. Ajustados", variable=self.plot_choice_var, value="residuals", command=self.update_diagnostic_plot).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(plot_controls_frame, text="Gráfico de Influencia", variable=self.plot_choice_var, value="influence", command=self.update_diagnostic_plot).pack(side=tk.LEFT, padx=5)

        # Combobox for influence plot labels
        self.influence_id_var = StringVar()
        influence_id_label = ttk.Label(plot_controls_frame, text="ID para Gráfico de Influencia:")
        influence_id_label.pack(side=tk.LEFT, padx=(20, 5))
        self.influence_id_combo = ttk.Combobox(plot_controls_frame, textvariable=self.influence_id_var, state="readonly", width=20)
        self.influence_id_combo.pack(side=tk.LEFT, padx=5)
        self.influence_id_combo.bind("<<ComboboxSelected>>", self.on_influence_id_selected)


        # Canvas for the plot
        self.diag_fig = plt.figure(figsize=(8, 6))
        self.diag_canvas = FigureCanvasTkAgg(self.diag_fig, master=diagnostics_frame)
        self.diag_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.diag_toolbar = NavigationToolbar2Tk(self.diag_canvas, diagnostics_frame)
        self.diag_toolbar.update()

        # Forest Plot Tab
        self.forest_plot_frame = ttk.Frame(graphs_notebook)
        graphs_notebook.add(self.forest_plot_frame, text="Forest Plot")
        
        self.forest_fig = plt.figure(figsize=(8, 6))
        self.forest_canvas = FigureCanvasTkAgg(self.forest_fig, master=self.forest_plot_frame)
        self.forest_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.forest_toolbar = NavigationToolbar2Tk(self.forest_canvas, self.forest_plot_frame)
        self.forest_toolbar.update()

    def on_influence_id_selected(self, event=None):
        self.plot_choice_var.set("influence")
        self.update_diagnostic_plot()

    def update_diagnostic_plot(self):
        if self.results is None:
            return

        self.diag_fig.clear()
        
        plot_choice = self.plot_choice_var.get()
        if plot_choice == "residuals":
            self.plot_residuals()
        elif plot_choice == "influence":
            self.plot_influence()
        
        self.diag_canvas.draw()


    def load_file(self):
        file_path = filedialog.askopenfilename(
            title="Seleccionar Archivo de Datos",
            filetypes=[("Archivos de Datos", "*.csv *.xlsx *.xls"), ("Todos los Archivos", "*.*")]
        )
        if not file_path:
            self.log("Carga de archivo cancelada.", "INFO")
            return

        try:
            if file_path.lower().endswith('.csv'):
                self.raw_data = pd.read_csv(file_path)
            else:
                self.raw_data = pd.read_excel(file_path)
            
            self.data = self.raw_data.copy()
            self.shared_dataset_active = False
            if hasattr(self, 'filter_component'):
                self.filter_component.set_dataframe(self.data)
            self.log(f"Archivo '{os.path.basename(file_path)}' cargado. Filas: {self.data.shape[0]}, Columnas: {self.data.shape[1]}", "SUCCESS")
            self.label_file_info.config(text=f"Cargado: {os.path.basename(file_path)}")
            if hasattr(self, 'data_source_status_var'):
                self.data_source_status_var.set("Origen: archivo local")
            self.update_variable_selectors()
        except Exception as e:
            self.log(f"Error al cargar archivo: {e}", "ERROR")
            messagebox.showerror("Error de Carga", f"No se pudo cargar el archivo:\n{e}")

    def receive_shared_dataset(self, *, dataset, filtered_dataset=None, filter_summary=None, metadata=None, source_widget=None):
        """Recibe el dataset compartido desde Archivo de Trabajo/MainApp."""
        if source_widget is self:
            return

        if dataset is None:
            self.raw_data = None
            self.data = None
            self.shared_dataset_active = False
            if hasattr(self, 'filter_component'):
                try:
                    self.filter_component.set_dataframe(pd.DataFrame())
                except Exception:
                    pass
            if hasattr(self, 'label_file_info'):
                self.label_file_info.config(text="Sin archivo compartido.")
            if hasattr(self, 'data_source_status_var'):
                self.data_source_status_var.set("Origen: sin datos")
            self.update_variable_selectors()
            self.results = None
            self.log("Dataset compartido limpiado en Regresión Lineal.", "INFO")
            return

        try:
            base_df = dataset.copy(deep=True)
        except Exception:
            base_df = dataset

        try:
            active_df = filtered_dataset.copy(deep=True) if isinstance(filtered_dataset, pd.DataFrame) else base_df.copy(deep=True)
        except Exception:
            active_df = filtered_dataset if isinstance(filtered_dataset, pd.DataFrame) else base_df

        self.raw_data = base_df
        self.data = active_df
        self.shared_dataset_active = True

        if hasattr(self, 'filter_component'):
            try:
                self.filter_component.set_dataframe(self.data)
            except Exception:
                pass

        source_name = "Dataset compartido"
        if isinstance(metadata, dict) and metadata.get('source_path'):
            try:
                source_name = os.path.basename(metadata['source_path'])
            except Exception:
                source_name = str(metadata.get('source_path'))

        try:
            rows_all, cols_all = self.raw_data.shape
            rows_active = self.data.shape[0]
            summary_txt = f"Compartido: {source_name} ({rows_active}/{rows_all} filas, {cols_all} cols)"
            if filter_summary:
                summary_txt += f" | filtros={len(list(filter_summary))}"
        except Exception:
            summary_txt = f"Compartido: {source_name}"

        if hasattr(self, 'label_file_info'):
            self.label_file_info.config(text=summary_txt)
        if hasattr(self, 'data_source_status_var'):
            self.data_source_status_var.set("Origen: dataset compartido")

        self.update_variable_selectors()
        self.results = None
        self.log(f"Dataset compartido recibido en Regresión Lineal: {summary_txt}", "SUCCESS")

    def update_variable_selectors(self):
        if self.data is None:
            self.combo_dep_var['values'] = []
            self.combo_dep_var.set("")
            self.listbox_indep_vars.delete(0, tk.END)
            if hasattr(self, 'influence_id_combo'):
                self.influence_id_combo['values'] = []
                self.influence_id_combo.set("")
            return
        
        cols = sorted(self.data.columns.tolist())
        numeric_cols = sorted(self.data.select_dtypes(include=np.number).columns.tolist())

        self.combo_dep_var['values'] = numeric_cols
        if numeric_cols:
            self.combo_dep_var.set(numeric_cols[0])

        self.listbox_indep_vars.delete(0, tk.END)
        for col in cols:
            self.listbox_indep_vars.insert(tk.END, col)

        # Populate the influence plot ID combobox
        if hasattr(self, 'influence_id_combo'):
            self.influence_id_combo['values'] = ["Índice"] + cols
            self.influence_id_combo.set("Índice")

    def _is_regularized_result(self):
        return bool(getattr(self, "current_penalization_method", "Ninguna") != "Ninguna")

    def _get_result_params(self):
        params = getattr(self.results, "params", None)
        if isinstance(params, pd.Series):
            return params.copy()
        if params is None:
            return pd.Series(dtype=float)
        try:
            return pd.Series(params, index=getattr(self, "model_feature_names", None))
        except Exception:
            return pd.Series(dtype=float)

    def _get_prediction_residuals(self):
        if self.results is None:
            return None, None

        fitted_vals = getattr(self.results, "fittedvalues", None)
        if fitted_vals is None:
            return None, None

        try:
            fitted_series = pd.Series(np.asarray(fitted_vals).reshape(-1), index=self.model_response.index)
        except Exception:
            fitted_series = pd.Series(np.asarray(fitted_vals).reshape(-1))

        residuals = getattr(self.results, "resid", None)
        if residuals is None and hasattr(self, "model_response"):
            try:
                residuals = np.asarray(self.model_response).reshape(-1) - np.asarray(fitted_series).reshape(-1)
            except Exception:
                residuals = None
        try:
            residual_series = pd.Series(np.asarray(residuals).reshape(-1), index=fitted_series.index) if residuals is not None else None
        except Exception:
            residual_series = None
        return fitted_series, residual_series

    def _quote_formula_name(self, column_name):
        return f'Q({json.dumps(str(column_name))})'

    def _build_variable_term(self, variable_name):
        config = self.variable_configs.get(variable_name, {}) if isinstance(self.variable_configs, dict) else {}
        quoted_name = self._quote_formula_name(variable_name)
        var_type = str(config.get('type', 'Cuantitativa') or 'Cuantitativa').strip()
        use_spline = bool(config.get('spline', False)) and var_type == 'Cuantitativa'

        if var_type == 'Cualitativa':
            ref_cat = config.get('ref_cat')
            if ref_cat not in (None, ''):
                return f"C({quoted_name}, Treatment(reference={repr(ref_cat)}))"
            return f"C({quoted_name})"

        if use_spline:
            spline_df = max(2, int(config.get('spline_df', 4) or 4))
            return f"bs({quoted_name}, df={spline_df}, include_intercept=False)"

        return quoted_name

    def _uses_configured_transformations(self, variable_names):
        for variable_name in variable_names:
            config = self.variable_configs.get(variable_name, {}) if isinstance(self.variable_configs, dict) else {}
            var_type = str(config.get('type', 'Cuantitativa') or 'Cuantitativa').strip()
            if var_type == 'Cualitativa' or bool(config.get('spline', False)):
                return True
        return False

    def _build_model_formula(self, dependent_var, variable_names):
        lhs = self._quote_formula_name(dependent_var)
        rhs_terms = [self._build_variable_term(var_name) for var_name in variable_names if str(var_name).strip()]
        if not rhs_terms:
            return f"{lhs} ~ 1"
        return f"{lhs} ~ {' + '.join(rhs_terms)}"

    def _build_model_results_text(self):
        if self.results is None:
            return "Sin resultados."

        fitted_vals, residuals = self._get_prediction_residuals()
        y_true = getattr(self, "model_response", None)
        y_true_arr = np.asarray(y_true).reshape(-1) if y_true is not None else np.asarray([], dtype=float)
        y_pred_arr = np.asarray(fitted_vals).reshape(-1) if fitted_vals is not None else np.asarray([], dtype=float)

        metric_lines = []
        if y_true_arr.size > 0 and y_pred_arr.size == y_true_arr.size:
            try:
                metric_lines.append(f"R²: {r2_score(y_true_arr, y_pred_arr):.4f}")
            except Exception:
                pass
            try:
                rmse = float(np.sqrt(mean_squared_error(y_true_arr, y_pred_arr)))
                metric_lines.append(f"RMSE: {rmse:.4f}")
            except Exception:
                pass
            try:
                mae = float(np.mean(np.abs(y_true_arr - y_pred_arr)))
                metric_lines.append(f"MAE: {mae:.4f}")
            except Exception:
                pass
            try:
                mse = float(mean_squared_error(y_true_arr, y_pred_arr))
                metric_lines.append(f"MSE: {mse:.4f}")
            except Exception:
                pass

        params = self._get_result_params()
        header = [
            "=== Regresión Lineal ===",
            f"Método: {getattr(self, 'current_penalization_method', 'Ninguna')}",
            f"Variable dependiente: {self.dependent_var}",
            f"Variables independientes finales ({len(self.independent_vars)}): {', '.join(self.independent_vars)}",
        ]
        if metric_lines:
            header.append("Métricas: " + " | ".join(metric_lines))

        if not self._is_regularized_result() and hasattr(self.results, "summary"):
            try:
                return "\n".join(header) + "\n\n" + str(self.results.summary())
            except Exception:
                pass

        lines = header + ["", "Coeficientes:"]
        if not params.empty:
            for name, value in params.items():
                if pd.isna(value):
                    continue
                lines.append(f"- {name}: {float(value):.6f}")
        else:
            lines.append("- No disponibles")

        if residuals is None:
            lines.append("")
            lines.append("Nota: este resultado no expone residuos completos para diagnósticos avanzados.")

        return "\n".join(lines)

    def _perform_variable_selection(self, y, X, selection_method):
        p_enter = self.p_enter_var.get()
        p_remove = self.p_remove_var.get()
        
        if selection_method == "Forward":
            return self._forward_selection(y, X, p_enter)
        elif selection_method == "Backward":
            return self._backward_elimination(y, X, p_remove)
        elif selection_method == "Stepwise":
            return self._stepwise_selection(y, X, p_enter, p_remove)
        else:
            return X.columns.tolist()

    def _forward_selection(self, y, X, p_enter):
        initial_list = []
        included = list(initial_list)
        while True:
            changed=False
            excluded = list(set(X.columns)-set(included))
            new_pval = pd.Series(index=excluded, dtype='float64')
            for new_column in excluded:
                model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included+[new_column]]))).fit()
                new_pval[new_column] = model.pvalues[new_column]
            best_pval = new_pval.min()
            if best_pval < p_enter:
                best_feature = new_pval.idxmin()
                included.append(best_feature)
                changed=True
            if not changed:
                break
        return included

    def _backward_elimination(self, y, X, p_remove):
        included = X.columns.tolist()
        while True:
            changed=False
            model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included]))).fit()
            pvalues = model.pvalues.iloc[1:]
            worst_pval = pvalues.max()
            if worst_pval > p_remove:
                changed=True
                worst_feature = pvalues.idxmax()
                included.remove(worst_feature)
            if not changed:
                break
        return included

    def _stepwise_selection(self, y, X, p_enter, p_remove):
        included = []
        while True:
            changed=False
            # forward step
            excluded = list(set(X.columns)-set(included))
            new_pval = pd.Series(index=excluded, dtype='float64')
            for new_column in excluded:
                model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included+[new_column]]))).fit()
                new_pval[new_column] = model.pvalues[new_column]
            best_pval = new_pval.min()
            if best_pval < p_enter:
                best_feature = new_pval.idxmin()
                included.append(best_feature)
                changed=True

            # backward step
            model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included]))).fit()
            pvalues = model.pvalues.iloc[1:]
            worst_pval = pvalues.max() 
            if worst_pval > p_remove:
                worst_feature = pvalues.idxmax()
                included.remove(worst_feature)
                changed=True
            
            if not changed:
                break
        return included

    def run_linear_regression(self):
        if self.data is None:
            messagebox.showerror("Error", "No hay datos cargados.")
            return

        if self.univariate_analysis_var.get():
            self._run_univariate_analysis()
            return

        if hasattr(self, 'filter_component'):
            filtered_data = self.filter_component.apply_filters()
            if filtered_data is None:
                return # Error message is shown by the component
        else:
            filtered_data = self.data.copy()

        self.dependent_var = self.combo_dep_var.get()
        selected_indices = self.listbox_indep_vars.curselection()
        initial_independent_vars = [self.listbox_indep_vars.get(i) for i in selected_indices]

        if not self.dependent_var or not initial_independent_vars:
            messagebox.showerror("Error", "Debe seleccionar una variable dependiente y al menos una independiente.")
            return

        selection_method = self.var_selection_method_var.get()
        scaling_method = self.covariate_scaling_method_var.get()
        penalization_method = self.penalization_method_var.get()
        self.current_penalization_method = penalization_method
        alpha = self.penalizer_strength_var.get()
        l1_ratio = self.l1_ratio_for_elasticnet_var.get()

        y = filtered_data[self.dependent_var]
        X = filtered_data[initial_independent_vars]

        if scaling_method != "Ninguna":
            self.log(f"Aplicando escalado: {scaling_method}", "INFO")
            if scaling_method == "StandardScaler":
                scaler = StandardScaler()
            else: # MinMaxScaler
                scaler = MinMaxScaler()
            X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)

        effective_selection_method = selection_method
        if selection_method != "Ninguno (usar todas)" and self._uses_configured_transformations(initial_independent_vars):
            warning_msg = (
                "La selección automática (Forward/Backward/Stepwise) con variables categóricas o splines "
                "aún no se soporta de forma segura en esta pestaña. Se usará 'Ninguno (usar todas)'."
            )
            self.log(warning_msg, "WARN")
            effective_selection_method = "Ninguno (usar todas)"

        final_vars = self._perform_variable_selection(y, X, effective_selection_method)
        
        if not final_vars:
            self.log("La selección de variables no resultó en ninguna variable seleccionada.", "WARN")
            messagebox.showwarning("Advertencia", "La selección de variables no resultó en ninguna variable seleccionada.")
            return

        self.independent_vars = final_vars
        
        formula = self._build_model_formula(self.dependent_var, self.independent_vars)
        self.log(f"Fórmula de regresión final: {formula}", "INFO")

        try:
            y, X = dmatrices(formula, data=filtered_data, return_type='dataframe')

            if scaling_method != "Ninguna":
                scale_columns = [col for col in X.columns if str(col).lower() != "intercept"]
                if scale_columns:
                    if scaling_method == "StandardScaler":
                        scaler = StandardScaler()
                    else:
                        scaler = MinMaxScaler()
                    X.loc[:, scale_columns] = scaler.fit_transform(X[scale_columns])

            # Store the data used for the model, aligned with the model's index, for later use in plotting.
            self.model_data = filtered_data.loc[y.index].copy()
            self.model_response = y.iloc[:, 0].copy() if isinstance(y, pd.DataFrame) else pd.Series(np.asarray(y).reshape(-1))
            self.model_feature_names = list(X.columns)
            
            model = sm.OLS(y, X)
            
            if penalization_method == "Ninguna":
                self.results = model.fit()
            else:
                if penalization_method == "L1 (Lasso)":
                    self.results = model.fit_regularized(method='elastic_net', alpha=alpha, L1_wt=1.0)
                elif penalization_method == "L2 (Ridge)":
                    self.results = model.fit_regularized(method='elastic_net', alpha=alpha, L1_wt=0.0)
                else: # ElasticNet
                    self.results = model.fit_regularized(method='elastic_net', alpha=alpha, L1_wt=l1_ratio)
            
            self.results_text.config(state=tk.NORMAL)
            self.results_text.delete('1.0', tk.END)
            self.results_text.insert(tk.END, self._build_model_results_text())
            self.results_text.config(state=tk.DISABLED)
            
            self.log("Regresión lineal completada exitosamente.", "SUCCESS")
            if not self.univariate_analysis_var.get():
                params = self._get_result_params()
                params = params.drop('Intercept', errors='ignore')
                conf_int = self.results.conf_int().drop('Intercept', errors='ignore') if hasattr(self.results, 'conf_int') else None
                multivariate_results = []
                for var in params.index:
                    if conf_int is not None and var in conf_int.index:
                        current_conf_int = conf_int.loc[var].tolist()
                    else:
                        current_conf_int = [params[var], params[var]]
                    multivariate_results.append({
                        'variable': var,
                        'coef': params[var],
                        'conf_int': current_conf_int
                    })
                self.plot_forest_plot(multivariate_results, "Forest Plot de Análisis Multivariado")
            self.notebook.select(self.tab_frame_results)

        except Exception as e:
            self.log(f"Error al ejecutar la regresión: {e}", "ERROR")
            self.log(traceback.format_exc(), "ERROR")
            if hasattr(self, 'tab_frame_log'):
                try:
                    self.notebook.select(self.tab_frame_log)
                except Exception:
                    pass
            messagebox.showerror("Error de Regresión", f"Ocurrió un error:\n{e}")

    def _run_univariate_analysis(self):
        if self.data is None:
            messagebox.showerror("Error", "No hay datos cargados.")
            return

        if hasattr(self, 'filter_component'):
            filtered_data = self.filter_component.apply_filters()
            if filtered_data is None:
                return # Error message is shown by the component
        else:
            filtered_data = self.data.copy()

        dependent_var = self.combo_dep_var.get()
        selected_indices = self.listbox_indep_vars.curselection()
        independent_vars = [self.listbox_indep_vars.get(i) for i in selected_indices]

        if not dependent_var or not independent_vars:
            messagebox.showerror("Error", "Debe seleccionar una variable dependiente y al menos una independiente.")
            return

        univariate_results = []
        summary_text = "Análisis Univariado\n" + "="*40 + "\n\n"

        for var in independent_vars:
            formula = self._build_model_formula(dependent_var, [var])
            try:
                y, X = dmatrices(formula, data=filtered_data, return_type='dataframe')
                model = sm.OLS(y, X)
                results = model.fit()

                params = results.params.drop('Intercept', errors='ignore')
                if params.empty:
                    continue
                coef_name = params.abs().idxmax()
                coef = float(params.loc[coef_name])
                conf_table = results.conf_int().drop('Intercept', errors='ignore')
                conf_int = conf_table.loc[coef_name].tolist() if coef_name in conf_table.index else [coef, coef]
                p_value = results.pvalues.get(coef_name)
                
                univariate_results.append({
                    'variable': var,
                    'coef': coef,
                    'conf_int': conf_int,
                    'p_value': p_value
                })
                
                summary_text += f"Variable: {var}\n"
                summary_text += f"Término principal: {coef_name}\n"
                summary_text += str(results.summary()) + "\n\n"

            except Exception as e:
                self.log(f"Error en análisis univariado para '{var}': {e}", "ERROR")

        self.results_text.config(state=tk.NORMAL)
        self.results_text.delete('1.0', tk.END)
        self.results_text.insert(tk.END, summary_text)
        self.results_text.config(state=tk.DISABLED)

        if univariate_results:
            self.plot_forest_plot(univariate_results, "Forest Plot de Análisis Univariado")
            self.notebook.select(self.tab_frame_graphs)



    def plot_forest_plot(self, results, title):
        self.forest_fig.clear()
        ax = self.forest_fig.add_subplot(111)

        variables = [res['variable'] for res in results]
        coefs = [res['coef'] for res in results]
        conf_ints = [res['conf_int'] for res in results]
        
        # Dynamic height adjustment
        num_vars = len(variables)
        base_height = 4
        height_per_var = 0.5
        fig_height = base_height + (num_vars * height_per_var)
        self.forest_fig.set_size_inches(8, fig_height)

        errors = [(ci[1] - ci[0]) / 2 for ci in conf_ints]
        
        ax.errorbar(coefs, range(len(variables)), xerr=errors, fmt='o', capsize=5)
        ax.axvline(x=0, color='r', linestyle='--')
        ax.set_yticks(range(len(variables)))
        ax.set_yticklabels(variables)
        ax.set_xlabel("Coeficiente")
        ax.set_title(title)
        
        self.forest_fig.tight_layout()
        self.forest_canvas.draw()

    def plot_residuals(self):
        if self.results is None:
            messagebox.showerror("Error", "No hay resultados de regresión para graficar.")
            return

        ax = self.diag_fig.add_subplot(111)
        fitted_vals, residuals = self._get_prediction_residuals()
        if fitted_vals is None or residuals is None:
            ax.text(0.5, 0.5, "No hay residuos disponibles para este modelo.", ha="center", va="center")
            ax.set_axis_off()
            return
        
        ax.scatter(fitted_vals, residuals)
        ax.axhline(y=0, color='r', linestyle='--')
        ax.set_xlabel("Valores Ajustados")
        ax.set_ylabel("Residuos")
        ax.set_title("Residuos vs. Valores Ajustados")

    def plot_influence(self):
        if self.results is None or not hasattr(self, 'model_data'):
            messagebox.showerror("Error", "No hay resultados de regresión para graficar. Ejecute un modelo primero.")
            return

        if self._is_regularized_result() or not hasattr(self.results, 'get_influence'):
            ax = self.diag_fig.add_subplot(111)
            ax.text(0.5, 0.5, "El gráfico de influencia solo está disponible para OLS sin penalización.", ha="center", va="center", wrap=True)
            ax.set_axis_off()
            return

        ax = self.diag_fig.add_subplot(111)
        id_var = self.influence_id_var.get()

        # The influence_plot uses the dataframe index for labels.
        # We temporarily change the index of the model's internal dataframe for plotting.
        original_exog_index = self.results.model.exog.index
        
        try:
            if id_var != "Índice" and id_var in self.model_data.columns:
                # Use the selected column for labels.
                # The model's data (exog) is aligned with model_data by index.
                new_labels = self.model_data[id_var]
                
                # If there are duplicate labels, append the original index to ensure uniqueness.
                if new_labels.duplicated().any():
                    self.log(f"Se encontraron IDs duplicados en la columna '{id_var}'. Se añadirán los índices originales para desambiguar.", "WARN")
                    new_labels = new_labels.astype(str) + " (idx:" + original_exog_index.astype(str) + ")"
                
                # Temporarily set the index on the model's exogenous variable dataframe
                self.results.model.exog.index = new_labels

            sm.graphics.influence_plot(self.results, criterion="cooks", ax=ax)
            ax.set_title("Gráfico de Influencia (Leverage vs. Residuos Estandarizados)")

        except Exception as e:
            self.log(f"Error al generar el gráfico de influencia: {e}", "ERROR")
            messagebox.showerror("Error de Gráfico", f"No se pudo generar el gráfico de influencia:\n{e}")
        finally:
            # IMPORTANT: Always restore the original index to avoid side effects.
            if hasattr(self.results.model.exog, 'index'):
                 self.results.model.exog.index = original_exog_index



if __name__ == '__main__':
    root = tk.Tk()
    root.title("Aplicación de Regresión Lineal")
    root.geometry("1000x700")
    app = LinearRegressionApp(root)
    root.mainloop()

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

            # Spline
            ttk.Label(row_labelframe, text="Spline:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
            spline_var = tk.BooleanVar(value=False)
            self.row_configs[var_name]['spline_var'] = spline_var
            cb_spline = ttk.Checkbutton(row_labelframe, text="Usar", variable=spline_var, command=lambda v=var_name: self._toggle_controls(v))
            cb_spline.grid(row=2, column=1, sticky=tk.W, padx=5)
            
            ttk.Label(row_labelframe, text="Spline DF:").grid(row=3, column=0, sticky=tk.W, padx=15, pady=2)
            spline_df_var = tk.IntVar(value=4)
            self.row_configs[var_name]['spline_df_var'] = spline_df_var
            spline_df_spinbox = ttk.Spinbox(row_labelframe, from_=2, to=10, textvariable=spline_df_var, width=5, state="disabled")
            spline_df_spinbox.grid(row=3, column=1, sticky=tk.W, padx=5)
            self.row_configs[var_name]['spline_df_spinbox'] = spline_df_spinbox

            # Load existing config
            if var_name in self.app_instance.variable_configs:
                config = self.app_instance.variable_configs[var_name]
                type_var.set(config.get('type', 'Cuantitativa'))
                if type_var.get() == 'Cualitativa':
                    unique_vals = sorted(self.app_instance.data[var_name].astype(str).unique().tolist())
                    ref_combo['values'] = unique_vals
                    ref_combo.set(config.get('ref_cat', unique_vals[0] if unique_vals else ''))
                if config.get('spline', False):
                    spline_var.set(True)
                    spline_df_var.set(config.get('spline_df', 4))
            
            self._toggle_controls(var_name)

        buttons_frame = ttk.Frame(main_frame)
        buttons_frame.pack(fill=tk.X, pady=(10,0))
        ttk.Button(buttons_frame, text="OK/Aplicar", command=self.apply_configurations).pack(side=tk.RIGHT, padx=5)
        ttk.Button(buttons_frame, text="Cancelar", command=self.destroy).pack(side=tk.RIGHT)

    def _toggle_controls(self, var_name):
        config = self.row_configs[var_name]
        var_is_quantitative = config['type_var'].get() == "Cuantitativa"
        
        config['ref_combo'].config(state="readonly" if not var_is_quantitative else "disabled")
        if var_is_quantitative:
            config['ref_combo'].set("")

        config['spline_df_spinbox'].config(state=tk.NORMAL if var_is_quantitative and config['spline_var'].get() else tk.DISABLED)
        if not var_is_quantitative:
            config['spline_var'].set(False)

    def apply_configurations(self):
        for var_name, config_widgets in self.row_configs.items():
            var_type = config_widgets['type_var'].get()
            self.app_instance.variable_configs[var_name] = {'type': var_type}
            
            if var_type == 'Cualitativa':
                self.app_instance.variable_configs[var_name]['ref_cat'] = config_widgets['ref_combo'].get()
            
            if var_type == 'Cuantitativa' and config_widgets['spline_var'].get():
                self.app_instance.variable_configs[var_name]['spline'] = True
                self.app_instance.variable_configs[var_name]['spline_df'] = config_widgets['spline_df_var'].get()
            else:
                self.app_instance.variable_configs[var_name]['spline'] = False

        self.destroy()
