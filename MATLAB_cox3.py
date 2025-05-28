#!/usr/bin/env python
# -*- coding: utf-8 -*-

# --- Importaciones Estándar de Python ---
from sklearn.model_selection import KFold
from lifelines.utils import concordance_index
from lifelines import CoxPHFitter, KaplanMeierFitter
# Usar este para evitar problemas con LogFormatter
from matplotlib.ticker import ScalarFormatter
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.pyplot as plt
import os
import pickle
import warnings
import traceback
import csv
import json
import re
import math
import sys # Añadido para manipulación de sys.path

# --- Importaciones de Tkinter ---
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog, StringVar, BooleanVar, DoubleVar, IntVar, Listbox, MULTIPLE, SINGLE, BROWSE, Toplevel, Frame, Label, Entry, Button, Checkbutton, Radiobutton
from tkinter import scrolledtext

# --- Importaciones de Librerías de Terceros (Data Science y Plotting) ---
import pandas as pd
import numpy as np
import scipy.stats

import matplotlib
matplotlib.use('TkAgg')  # Backend para Tkinter

# --- Importaciones de Lifelines ---
# from lifelines.statistics import proportional_hazard_test #
# check_assumptions lo reemplaza en gran medida

try:
    from lifelines.scoring import brier_score
    LIFELINES_BRIER_SCORE_AVAILABLE = True
except ImportError:
    LIFELINES_BRIER_SCORE_AVAILABLE = False
    print("ADVERTENCIA: La función 'brier_score' no pudo ser importada desde 'lifelines.scoring'.")

try:
    from lifelines.calibration import survival_probability_calibration_plot
    LIFELINES_CALIBRATION_AVAILABLE = True
except ImportError:
    LIFELINES_CALIBRATION_AVAILABLE = False
    print("ADVERTENCIA: 'survival_probability_calibration_plot' no pudo ser importada. Gráfico de Calibración no disponible.")


try:
    from patsy import dmatrix
    PATSY_AVAILABLE = True
except ImportError:
    dmatrix = None
    PATSY_AVAILABLE = False
    print("ADVERTENCIA: 'patsy' no instalada. Funciones de Spline y manejo avanzado de categóricas limitadas.")

# Bloque de importación del componente de filtro con manejo de sys.path
filter_component_path = r"D:\APPS\MATABS"
if filter_component_path not in sys.path:
    sys.path.insert(0, filter_component_path)
try:
    from MATLAB_filter_component import FilterComponent
    FILTER_COMPONENT_AVAILABLE = True
except ImportError:
    FilterComponent = None
    FILTER_COMPONENT_AVAILABLE = False
    print(f"ERROR: No se pudo importar MATLAB_filter_component desde '{filter_component_path}'. Filtros avanzados no disponibles.")
except Exception as e:
    FilterComponent = None
    FILTER_COMPONENT_AVAILABLE = False
    print(f"ERROR inesperado al importar MATLAB_filter_component: {e}. Filtros avanzados no disponibles.")
    traceback.print_exc(limit=1)

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning, module='lifelines')
warnings.filterwarnings("ignore", category=FutureWarning)

# --- FUNCIONES AUXILIARES GLOBALES ---


def format_p_value(p_val, threshold=0.0001):
    if pd.isna(p_val) or not isinstance(p_val, (float, np.floating, int)):
        return "N/A"
    if p_val < threshold:
        return f"{p_val:.2e}"
    else:
        return f"{p_val:.4f}"


def compute_model_metrics(model, X_design, y_data, time_col, event_col,
                          c_index_cv_mean=None, c_index_cv_std=None,
                          schoenfeld_results_df=None, loglik_null=None, log_func=print):
    metrics = {}
    n_obs = y_data.shape[0] if y_data is not None and not y_data.empty else 0
    num_params = 0
    if hasattr(
            model,
            'params_') and model.params_ is not None and not model.params_.empty:
        num_params = len(model.params_)
    elif X_design is not None and not X_design.empty:
        num_params = X_design.shape[1]

    log_likelihood_val = getattr(model, "log_likelihood_", None)
    metrics["Log-Likelihood"] = log_likelihood_val
    metrics["-2 Log-Likelihood"] = -2.0 * \
        log_likelihood_val if log_likelihood_val is not None and pd.notna(
            log_likelihood_val) else None

    schoenfeld_p_global = None
    if schoenfeld_results_df is not None and isinstance(
            schoenfeld_results_df,
            pd.DataFrame) and not schoenfeld_results_df.empty:
        if 'p' in schoenfeld_results_df.columns:
            # Intentos para encontrar el p-valor global
            if 'global' in schoenfeld_results_df.index.map(str).str.lower():
                try:
                    schoenfeld_p_global = schoenfeld_results_df.loc[schoenfeld_results_df.index.map(str).str.lower() == 'global', 'p'].iloc[0]
                except IndexError:
                    pass
            elif isinstance(schoenfeld_results_df.index, pd.MultiIndex) and ('all', '-') in schoenfeld_results_df.index:
                schoenfeld_p_global = schoenfeld_results_df.loc[(
                    'all', '-'), 'p']
            elif any(idx in schoenfeld_results_df.index.map(str).str.lower() for idx in ['test_statistic', 'global_test', 'overall']):
                try:
                    schoenfeld_p_global = schoenfeld_results_df[schoenfeld_results_df.index.map(str).str.lower().isin(['test_statistic', 'global_test', 'overall'])]['p'].iloc[0]
                except IndexError:
                    pass
        metrics["Schoenfeld details"] = schoenfeld_results_df.to_dict('index')
    metrics["Schoenfeld p-value (global)"] = schoenfeld_p_global

    if hasattr(
            model,
            "summary") and model.summary is not None and not model.summary.empty:
        summary_df = model.summary.copy()
        if 'z' in summary_df.columns:
            z_scores = summary_df['z'].dropna()
            if not z_scores.empty:
                wald_stat = float(
                    (z_scores ** 2).sum())
                df_wald = len(z_scores)
                metrics["Wald p-value (global approx)"] = scipy.stats.chi2.sf(
                    wald_stat, df_wald) if df_wald > 0 else None
        if 'p' in summary_df.columns:
            metrics["Wald p-values (individual)"] = summary_df["p"].dropna().to_dict()
        if 'exp(coef)' in summary_df.columns:
            metrics["HR (individual)"] = summary_df['exp(coef)'].to_dict()
        if 'exp(coef) lower 95%' in summary_df.columns and 'exp(coef) upper 95%' in summary_df.columns:
            metrics["HR_CI (individual)"] = {
                str(idx): {
                    'HR': r.get('exp(coef)'),
                    'lower_95': r.get('exp(coef) lower 95%'),
                    'upper_95': r.get('exp(coef) upper 95%')} for idx,
                r in summary_df.iterrows()}
        metrics["summary_df"] = summary_df

    metrics["C-Index (Training)"] = getattr(model, "concordance_index_", None)
    metrics["C-Index (CV Mean)"] = c_index_cv_mean
    metrics["C-Index (CV Std)"] = c_index_cv_std

    if log_likelihood_val is not None and pd.notna(
            log_likelihood_val) and n_obs > 0 and num_params >= 0:
        metrics["AIC"] = -2 * log_likelihood_val + 2 * num_params
        if n_obs > num_params + 1:
            metrics["BIC"] = -2 * log_likelihood_val + np.log(n_obs) * num_params
        else:
            metrics["BIC"] = None
    else:
        metrics["AIC"] = None
        metrics["BIC"] = None

    if log_likelihood_val is not None and pd.notna(log_likelihood_val) and \
       loglik_null is not None and pd.notna(loglik_null) and num_params > 0:
        lr_stat = -2 * (loglik_null - log_likelihood_val)
        if lr_stat < 0:
            lr_stat = 0.0
        metrics["Global LR Test p-value"] = scipy.stats.chi2.sf(
            lr_stat, num_params) if num_params > 0 else None
    else:
        metrics["Global LR Test p-value"] = None
    return metrics

# --- CLASES AUXILIARES PARA LA UI ---


class PlotOptionsDialog(Toplevel):
    def __init__(self, parent, current_options=None, apply_callback=None):
        super().__init__(parent)
        self.transient(parent)
        self.grab_set()
        self.title("Opciones de Gráfico")
        self.parent = parent
        self.result = {}
        self.apply_callback = apply_callback
        self.current_options = current_options if current_options else {}

        fields_setup = {
            "title": ("Título del Gráfico:", self.current_options.get('title', '')),
            "xlabel": ("Etiqueta Eje X:", self.current_options.get('xlabel', '')),
            "ylabel": ("Etiqueta Eje Y:", self.current_options.get('ylabel', '')),
            "linewidth": ("Ancho de Línea:", str(self.current_options.get('linewidth', '1.5'))),
            "markersize": ("Tamaño de Marcador:", str(self.current_options.get('markersize', '5'))),
            "xlim_min": ("Límite X Mínimo:", str(self.current_options.get('xlim_min', ''))),
            "xlim_max": ("Límite X Máximo:", str(self.current_options.get('xlim_max', ''))),
            "ylim_min": ("Límite Y Mínimo:", str(self.current_options.get('ylim_min', ''))),
            "ylim_max": ("Límite Y Máximo:", str(self.current_options.get('ylim_max', '')))
        }

        self.vars_entries = {}
        main_dialog_frame = ttk.Frame(self, padding="10")
        main_dialog_frame.pack(fill=tk.BOTH, expand=True)

        current_row_idx = 0
        for key, (label_text_val, default_val_str) in fields_setup.items():
            ttk.Label(main_dialog_frame, text=label_text_val).grid(
                row=current_row_idx, column=0, sticky=tk.W, padx=5, pady=3)
            entry_var = StringVar(self, value=default_val_str)
            self.vars_entries[key] = entry_var
            ttk.Entry(main_dialog_frame, textvariable=entry_var, width=40).grid(
                row=current_row_idx, column=1, sticky=tk.EW, padx=5, pady=3)
            current_row_idx += 1

        main_dialog_frame.columnconfigure(1, weight=1)

        ttk.Label(main_dialog_frame, text="Escala Eje X:").grid(
            row=current_row_idx, column=0, sticky=tk.W, padx=5, pady=3)
        self.xscale_var_tk = StringVar(self, value=self.current_options.get('xscale', 'linear'))
        ttk.Combobox(main_dialog_frame, textvariable=self.xscale_var_tk,
                     values=["linear", "log"], state="readonly", width=10).grid(
                         row=current_row_idx, column=1, sticky=tk.W, padx=5, pady=3)
        current_row_idx += 1

        ttk.Label(main_dialog_frame, text="Escala Eje Y:").grid(
            row=current_row_idx, column=0, sticky=tk.W, padx=5, pady=3)
        self.yscale_var_tk = StringVar(self, value=self.current_options.get('yscale', 'linear'))
        ttk.Combobox(main_dialog_frame, textvariable=self.yscale_var_tk,
                     values=["linear", "log"], state="readonly", width=10).grid(
                         row=current_row_idx, column=1, sticky=tk.W, padx=5, pady=3)
        current_row_idx += 1

        ttk.Label(main_dialog_frame, text="Paleta de Colores:").grid(
            row=current_row_idx, column=0, sticky=tk.W, padx=5, pady=3)
        self.cmap_var_tk = StringVar(self, value=self.current_options.get('cmap', 'viridis'))
        available_colormaps = [
            'viridis', 'plasma', 'inferno', 'magma', 'cividis',
            'Blues', 'Greens', 'Oranges', 'Reds', 'Purples', 'Greys',
            'YlOrRd', 'YlGnBu', 'PuBuGn', 'BuGn', 'GnBu', 'Pastel1', 'Set1'
        ]
        self.cmap_combobox = ttk.Combobox(main_dialog_frame, textvariable=self.cmap_var_tk,
                                           values=available_colormaps, state="readonly", width=15)
        self.cmap_combobox.grid(
            row=current_row_idx, column=1, sticky=tk.W, padx=5, pady=3)
        current_row_idx += 1

        ttk.Label(main_dialog_frame, text="Orden de Forest Plot:").grid(
            row=current_row_idx, column=0, sticky=tk.W, padx=5, pady=3)
        sort_order_options = [
            "original", "hr_asc", "hr_desc", "p_asc", "p_desc", "name_asc", "name_desc"
        ]
        self.sort_order_var_tk = StringVar(self, value=self.current_options.get('sort_order', 'original'))
        self.sort_order_combobox = ttk.Combobox(main_dialog_frame, textvariable=self.sort_order_var_tk,
                                                 values=sort_order_options, state="readonly", width=15)
        self.sort_order_combobox.grid(
            row=current_row_idx, column=1, sticky=tk.W, padx=5, pady=3)
        current_row_idx += 1

        self.grid_on_var_tk = BooleanVar(self, value=self.current_options.get('grid', True))
        ttk.Checkbutton(main_dialog_frame, text="Mostrar Rejilla (Grid)",
                        variable=self.grid_on_var_tk).grid(
                            row=current_row_idx, column=0, columnspan=2, sticky=tk.W, padx=5, pady=5)
        current_row_idx += 1

        buttons_frame = ttk.Frame(main_dialog_frame)
        buttons_frame.grid(row=current_row_idx, column=0, columnspan=2, pady=15)
        ttk.Button(buttons_frame, text="Aplicar Opciones", command=self._on_apply_options).pack(side=tk.LEFT, padx=10)
        ttk.Button(buttons_frame, text="Cancelar", command=self.destroy).pack(side=tk.LEFT, padx=10)
        self.wait_window(self)

    def _on_apply_options(self):
        for key, str_var_obj in self.vars_entries.items():
            val_str_ui = str_var_obj.get().strip()
            try:
                if key in ['linewidth', 'markersize', 'xlim_min', 'xlim_max', 'ylim_min', 'ylim_max']:
                    self.result[key] = float(val_str_ui) if val_str_ui else None
                else:
                    self.result[key] = val_str_ui if val_str_ui else None
            except ValueError:
                self.result[key] = None
                if self.parent and hasattr(self.parent, 'log'):
                    self.parent.log(f"Valor inválido '{val_str_ui}' para opción '{key}'.", "WARN")
        self.result['xscale'] = self.xscale_var_tk.get()
        self.result['yscale'] = self.yscale_var_tk.get()
        self.result['grid'] = self.grid_on_var_tk.get()
        self.result['cmap'] = self.cmap_var_tk.get()
        self.result['sort_order'] = self.sort_order_var_tk.get()
        if self.apply_callback:
            self.apply_callback(self.result)
        self.destroy()


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


def apply_plot_options(ax, options_dict, log_func=print):
    if not options_dict or ax is None:
        return
    try:
        if options_dict.get('title') is not None:
            ax.set_title(options_dict['title'])
        if options_dict.get('xlabel') is not None:
            ax.set_xlabel(options_dict['xlabel'])
        if options_dict.get('ylabel') is not None:
            ax.set_ylabel(options_dict['ylabel'])
        xmin_opt, xmax_opt = options_dict.get(
            'xlim_min'), options_dict.get('xlim_max')
        ymin_opt, ymax_opt = options_dict.get(
            'ylim_min'), options_dict.get('ylim_max')
        current_ax_xlim = ax.get_xlim()
        final_ax_xmin = xmin_opt if xmin_opt is not None and pd.notna(
            xmin_opt) else current_ax_xlim[0]
        final_ax_xmax = xmax_opt if xmax_opt is not None and pd.notna(
            xmax_opt) else current_ax_xlim[1]
        if final_ax_xmin is not None and final_ax_xmax is not None and final_ax_xmin < final_ax_xmax:
            ax.set_xlim(final_ax_xmin, final_ax_xmax)
        current_ax_ylim = ax.get_ylim()
        final_ax_ymin = ymin_opt if ymin_opt is not None and pd.notna(
            ymin_opt) else current_ax_ylim[0]
        final_ax_ymax = ymax_opt if ymax_opt is not None and pd.notna(
            ymax_opt) else current_ax_ylim[1]
        if final_ax_ymin is not None and final_ax_ymax is not None and final_ax_ymin < final_ax_ymax:
            ax.set_ylim(final_ax_ymin, final_ax_ymax)
        if options_dict.get('xscale') == 'log':
            if ax.get_xlim()[0] > 0:
                ax.set_xscale('log')
                ax.xaxis.set_major_formatter(ScalarFormatter())
                ax.xaxis.get_major_formatter().set_scientific(False)
                ax.xaxis.get_major_formatter().set_useOffset(False)
            else:
                log_func("Advertencia: Límite X <= 0, no se puede aplicar escala log.", "WARN")
        elif options_dict.get('xscale') == 'linear':
            ax.set_xscale('linear')
            ax.xaxis.set_major_formatter(ScalarFormatter())
        if options_dict.get('yscale') == 'log':
            if ax.get_ylim()[0] > 0:
                ax.set_yscale('log')
                ax.yaxis.set_major_formatter(ScalarFormatter())
                ax.yaxis.get_major_formatter().set_scientific(False)
                ax.yaxis.get_major_formatter().set_useOffset(False)
            else:
                log_func("Advertencia: Límite Y <= 0, no se puede aplicar escala log.", "WARN")
        elif options_dict.get('yscale') == 'linear':
            ax.set_yscale('linear')
            ax.yaxis.set_major_formatter(ScalarFormatter())
        if options_dict.get('grid') is not None:
            ax.grid(options_dict.get('grid'), linestyle=':', alpha=0.6)
        linewidth_opt_val = options_dict.get('linewidth')
        if linewidth_opt_val is not None and pd.notna(linewidth_opt_val):
            for line_obj in ax.get_lines():
                line_obj.set_linewidth(linewidth_opt_val)
        markersize_opt_val = options_dict.get('markersize')
        if markersize_opt_val is not None and pd.notna(markersize_opt_val):
            for line_obj in ax.get_lines():
                if line_obj.get_marker() not in ['None', None, '']:
                    line_obj.set_markersize(markersize_opt_val)
        if ax.get_legend() is not None:
            ax.legend()
        if hasattr(ax.figure, 'canvas') and ax.figure.canvas:
            ax.figure.canvas.draw_idle()
    except Exception as e_apply_plot:
        log_func(f"Error aplicando opciones de gráfico: {e_apply_plot}", "ERROR")
        traceback.print_exc(limit=1)


class ModelSummaryWindow(Toplevel):
    def __init__(self, parent, title="Resumen del Modelo", summary_text=""):
        super().__init__(parent)
        self.title(title)
        self.geometry("750x600")
        self.transient(parent)
        self.grab_set()
        main_summary_frame = ttk.Frame(self, padding="10")
        main_summary_frame.pack(fill=tk.BOTH, expand=True)
        self.summary_text_widget = scrolledtext.ScrolledText(
            main_summary_frame, wrap=tk.WORD, height=25, width=85, font=("Courier New", 9))
        self.summary_text_widget.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.summary_text_widget.insert(tk.END, summary_text)
        self.summary_text_widget.config(state=tk.DISABLED)
        summary_buttons_frame = ttk.Frame(main_summary_frame)
        summary_buttons_frame.pack(fill=tk.X, pady=(10, 0))
        ttk.Button(summary_buttons_frame, text="Copiar Todo al Portapapeles",
                   command=self._copy_summary_to_clipboard).pack(side=tk.LEFT, padx=5)
        ttk.Button(summary_buttons_frame, text="Cerrar Ventana",
                   command=self.destroy).pack(side=tk.RIGHT, padx=5)
        self.protocol("WM_DELETE_WINDOW", self.destroy)
        self.summary_text_widget.focus_set()
        self.wait_window(self)

    def _copy_summary_to_clipboard(self):
        try:
            self.clipboard_clear()
            self.clipboard_append(self.summary_text_widget.get("1.0", tk.END))
            messagebox.showinfo("Resumen Copiado", "Contenido copiado al portapapeles.", parent=self)
        except tk.TclError:
            messagebox.showwarning("Error al Copiar", "No se pudo acceder al portapapeles.", parent=self)
        except Exception as e_copy:
            messagebox.showerror("Error", f"Error al copiar: {e_copy}", parent=self)


# --- CLASE PRINCIPAL DE LA APLICACIÓN ---
class CoxModelingApp(ttk.Frame):
    def __init__(self, parent_notebook_tab):
        super().__init__(parent_notebook_tab)
        self.pack(fill=tk.BOTH, expand=True)
        self.parent_for_dialogs = self.winfo_toplevel()

        # Variables para datos y configuración
        self.raw_data = None
        self.data = None
        self.time_col_original_name = ""
        self.event_col_original_name = ""
        self.selected_covariables_from_ui = []
        self.covariables_type_config = {}  # {var_name: "Cuantitativa" | "Cualitativa"}
        self.ref_categories_config = {}  # {cual_var_name: "ref_category_value"}
        # {cuant_var_name: {'type': 'Natural'|'B-spline', 'df': int}}
        self.spline_config_details = {}
        self.current_plot_options = {}  # Diccionario para guardar opciones de gráficos

        # Variables para modelos
        # Lista de diccionarios, cada uno con datos de un modelo
        self.generated_models_data = []
        # Diccionario del modelo seleccionado en la Treeview
        self.selected_model_in_treeview = None

        # Variables de control para la UI (Pestaña 2: Modelado)
        self.cox_model_type_var = StringVar(value="Multivariado")  # "Univariado" | "Multivariado"
        # "Ninguno (usar todas)" | "Backward" | "Forward" | "Stepwise (Fwd luego Bwd)"
        self.var_selection_method_var = StringVar(value="Ninguno (usar todas)")
        # Umbral p-value para entrar (Forward/Stepwise)
        self.p_enter_var = DoubleVar(value=0.05)
        # Umbral p-value para salir (Backward/Stepwise)
        self.p_remove_var = DoubleVar(value=0.05)
        # "Ninguna" | "L2 (Ridge)" | "L1 (Lasso)" | "ElasticNet"
        self.penalization_method_var = StringVar(value="Ninguna")
        self.penalizer_strength_var = DoubleVar(value=0.1)  # Valor de lambda (alpha en lifelines)
        self.l1_ratio_for_elasticnet_var = DoubleVar(value=0.5)  # Ratio L1 para ElasticNet (0=Ridge, 1=Lasso)
        self.tie_handling_method_var = StringVar(value="efron")  # "efron" | "breslow" | "exact"
        self.calculate_cv_cindex_var = BooleanVar(value=False)  # Calcular C-Index por CV
        self.cv_num_kfolds_var = IntVar(value=5)  # Número de folds para CV
        self.cv_random_seed_var = IntVar(value=42)  # Semilla aleatoria para CV

        # Crear Notebook (pestañas)
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)

        # Pestaña 1: Carga, Filtros y Preproceso
        self.tab_frame_preproc = ttk.Frame(self.notebook, padding="10")
        self.tab_frame_preproc.pack(fill=tk.BOTH, expand=True)
        self.notebook.add(self.tab_frame_preproc, text='  1. Carga y Preprocesamiento de Datos  ')

        # Usar ScrolledFrame para el contenido de la pestaña 1
        self.tab_frame_preproc_content = ScrolledFrame(self.tab_frame_preproc)
        self.tab_frame_preproc_content.pack(fill=tk.BOTH, expand=True)

        # Pestaña 2: Modelado Cox
        self.tab_frame_modeling = ttk.Frame(self.notebook, padding="10")
        self.tab_frame_modeling.pack(fill=tk.BOTH, expand=True)
        self.notebook.add(self.tab_frame_modeling, text='  2. Modelado Cox  ')

        # Usar ScrolledFrame para el contenido de la pestaña 2
        self.tab_frame_modeling_content = ScrolledFrame(self.tab_frame_modeling)
        self.tab_frame_modeling_content.pack(fill=tk.BOTH, expand=True)

        # Pestaña 3: Visualización y Reportes
        self.tab_frame_results = ttk.Frame(self.notebook, padding="10")
        self.tab_frame_results.pack(fill=tk.BOTH, expand=True)
        self.notebook.add(self.tab_frame_results, text='  3. Resultados y Visualización  ')

        # Usar ScrolledFrame para el contenido de la pestaña 3
        self.tab_frame_results_content = ScrolledFrame(self.tab_frame_results)
        self.tab_frame_results_content.pack(fill=tk.BOTH, expand=True)

        # Pestaña 4: Log
        self.tab_frame_log = ttk.Frame(self.notebook, padding="10")
        self.tab_frame_log.pack(fill=tk.BOTH, expand=True)
        self.notebook.add(self.tab_frame_log, text='  Log  ')

        # Controles de Log
        self.log_text_widget = scrolledtext.ScrolledText(
            self.tab_frame_log, wrap=tk.WORD, height=10, state=tk.DISABLED, font=("Courier New", 9))
        self.log_text_widget.pack(fill=tk.BOTH, expand=True)
        self.log_text_widget.tag_config("INFO", foreground="black")
        self.log_text_widget.tag_config("DEBUG", foreground="gray")
        self.log_text_widget.tag_config("WARN", foreground="orange")
        self.log_text_widget.tag_config("ERROR", foreground="red")
        self.log_text_widget.tag_config("SUCCESS", foreground="green")
        self.log_text_widget.tag_config("HEADER", foreground="blue", font=("Courier New", 9, "bold"))
        self.log_text_widget.tag_config("SUBHEADER", foreground="purple", font=("Courier New", 9, "bold"))
        self.log_text_widget.tag_config("CONFIG", foreground="darkgreen")

        # Inicializar controles de cada pestaña
        self.create_preproc_controls()
        self.create_grid_controls()
        self.create_results_controls()

        self.log("Interfaz de CoxModelingApp inicializada y controles creados.", "INFO")

    def log(self, message_text, level_str="INFO"):
        if not hasattr(self, 'log_text_widget'):
            print(f"FALLBACK LOG: [{level_str.upper()}] {message_text}")
            return
        try:
            current_timestamp = pd.Timestamp.now().strftime('%H:%M:%S.%f')[:-3]
            self.log_text_widget.config(state=tk.NORMAL)
            self.log_text_widget.insert(tk.END, f"[{current_timestamp}] [{level_str.upper()}] {message_text}\n", level_str.upper())
            self.log_text_widget.config(state=tk.DISABLED)
            self.log_text_widget.see(tk.END)
            if level_str.upper() in ["ERROR", "WARN"] and self.parent_for_dialogs and self.parent_for_dialogs.winfo_exists():
                self.parent_for_dialogs.bell()
        except Exception as e_logging:
            print(f"ERROR EN LOGGER: {e_logging}")

    # --- MÉTODOS PARA PESTAÑA 1: CARGA, FILTROS Y PREPROCESO ---

    def create_preproc_controls(self):
        p_content = self.tab_frame_preproc_content.interior
        self.log("Creando controles para la pestaña de Preprocesamiento...", "DEBUG")
        frame_carga_archivo = ttk.LabelFrame(p_content, text="Carga de Archivo de Datos")
        frame_carga_archivo.pack(fill=tk.X, padx=10, pady=10, ipady=5)
        btn_cargar = ttk.Button(frame_carga_archivo, text="Seleccionar y Cargar Archivo (.xlsx, .xls, .csv)", command=self.cargar_archivo)
        btn_cargar.pack(side=tk.LEFT, padx=10, pady=10)
        self.label_archivo_cargado_info = ttk.Label(frame_carga_archivo, text="Ningún archivo cargado.", width=60, anchor="w")
        self.label_archivo_cargado_info.pack(side=tk.LEFT, padx=10, pady=10, fill=tk.X, expand=True)

        if FILTER_COMPONENT_AVAILABLE and FilterComponent is not None:
            frame_filtros_avanzados = ttk.LabelFrame(p_content, text="Filtros Avanzados sobre Datos Cargados")
            frame_filtros_avanzados.pack(fill=tk.BOTH, expand=True, padx=10, pady=10, ipady=5)
            self.custom_filter_component_instance = FilterComponent(frame_filtros_avanzados, log_callback=self.log)
            self.custom_filter_component_instance.pack(fill="both", expand=True, padx=5, pady=5)
            btn_aplicar_filtros_avanzados = ttk.Button(frame_filtros_avanzados, text="Aplicar Filtros Avanzados al Dataset Principal", command=self._apply_fc_filters_to_main_data)
            btn_aplicar_filtros_avanzados.pack(pady=10, padx=5)
        else:
            frame_filtros_avanzados_disabled = ttk.LabelFrame(p_content, text="Filtros Avanzados")
            frame_filtros_avanzados_disabled.pack(fill=tk.X, padx=10, pady=10)
            ttk.Label(frame_filtros_avanzados_disabled, text="Funcionalidad de Filtros Avanzados no disponible (MATLAB_filter_component no cargado).", wraplength=400, justify=tk.LEFT).pack(padx=10, pady=10)

        frame_definicion_modelo = ttk.LabelFrame(p_content, text="Definición del Modelo de Supervivencia y Configuración de Variables")
        frame_definicion_modelo.pack(fill=tk.BOTH, expand=True, padx=10, pady=10, ipady=5)

        subframe_tiempo_evento = ttk.Frame(frame_definicion_modelo, padding=5)
        subframe_tiempo_evento.pack(fill=tk.X, pady=5)
        ttk.Label(subframe_tiempo_evento, text="Columna de Tiempo:").grid(row=0, column=0, padx=5, pady=3, sticky="e")
        self.combo_col_tiempo = ttk.Combobox(subframe_tiempo_evento, state="readonly", width=25)
        self.combo_col_tiempo.grid(row=0, column=1, padx=5, pady=3, sticky="ew")
        ttk.Label(subframe_tiempo_evento, text="Renombrar Tiempo a (opcional):").grid(row=0, column=2, padx=5, pady=3, sticky="e")
        self.entry_renombrar_col_tiempo = ttk.Entry(subframe_tiempo_evento, width=20)
        self.entry_renombrar_col_tiempo.grid(row=0, column=3, padx=5, pady=3, sticky="ew")

        ttk.Label(subframe_tiempo_evento, text="Columna de Evento (0/1):").grid(row=1, column=0, padx=5, pady=3, sticky="e")
        self.combo_col_evento = ttk.Combobox(subframe_tiempo_evento, state="readonly", width=25)
        self.combo_col_evento.grid(row=1, column=1, padx=5, pady=3, sticky="ew")
        ttk.Label(subframe_tiempo_evento, text="Renombrar Evento a (opcional):").grid(row=1, column=2, padx=5, pady=3, sticky="e")
        self.entry_renombrar_col_evento = ttk.Entry(subframe_tiempo_evento, width=20)
        self.entry_renombrar_col_evento.grid(row=1, column=3, padx=5, pady=3, sticky="ew")
        subframe_tiempo_evento.columnconfigure(1, weight=1)
        subframe_tiempo_evento.columnconfigure(3, weight=1)

        paned_covariables_config = ttk.PanedWindow(frame_definicion_modelo, orient=tk.HORIZONTAL)
        paned_covariables_config.pack(fill=tk.BOTH, expand=True, pady=10)

        frame_lista_covariables = ttk.LabelFrame(paned_covariables_config, text="Selección de Covariables para el Modelo")
        paned_covariables_config.add(frame_lista_covariables, weight=1)
        ttk.Label(frame_lista_covariables, text="Variables disponibles (seleccione para incluir en modelos y/o configurar):").pack(anchor="w", padx=5, pady=(5, 2))

        frame_botones_lista_covs = ttk.Frame(frame_lista_covariables)
        frame_botones_lista_covs.pack(fill=tk.X, padx=5)
        ttk.Button(frame_botones_lista_covs, text="Sel. Todas", width=10,
                   command=lambda: self.listbox_covariables_disponibles.selection_set(0, tk.END)
                   if hasattr(self, 'listbox_covariables_disponibles') else None).pack(side=tk.LEFT, padx=2, pady=2)
        ttk.Button(frame_botones_lista_covs, text="Desel. Todas", width=10,
                   command=lambda: self.listbox_covariables_disponibles.selection_clear(0, tk.END)
                   if hasattr(self, 'listbox_covariables_disponibles') else None).pack(side=tk.LEFT, padx=2, pady=2)

        frame_listbox_con_scroll = ttk.Frame(frame_lista_covariables)
        frame_listbox_con_scroll.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.listbox_covariables_disponibles = tk.Listbox(frame_listbox_con_scroll, selectmode=tk.MULTIPLE, height=12, exportselection=False)
        scrollbar_y_lista_covs = ttk.Scrollbar(frame_listbox_con_scroll, orient=tk.VERTICAL, command=self.listbox_covariables_disponibles.yview)
        scrollbar_x_lista_covs = ttk.Scrollbar(frame_listbox_con_scroll, orient=tk.HORIZONTAL, command=self.listbox_covariables_disponibles.xview)
        self.listbox_covariables_disponibles.config(yscrollcommand=scrollbar_y_lista_covs.set, xscrollcommand=scrollbar_x_lista_covs.set)
        scrollbar_y_lista_covs.pack(side=tk.RIGHT, fill=tk.Y)
        scrollbar_x_lista_covs.pack(side=tk.BOTTOM, fill=tk.X)
        self.listbox_covariables_disponibles.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.listbox_covariables_disponibles.bind("<<ListboxSelect>>", self.on_covariate_select_for_config)

        frame_config_covariable_seleccionada = ttk.LabelFrame(paned_covariables_config, text="Configurar Variable(s) Seleccionada(s)")
        paned_covariables_config.add(frame_config_covariable_seleccionada, weight=1)

        self.label_cov_seleccionada_nombre = ttk.Label(frame_config_covariable_seleccionada, text="Ninguna Seleccionada", font=("TkDefaultFont", 10, "bold"), wraplength=200, justify=tk.CENTER)
        self.label_cov_seleccionada_nombre.pack(pady=(10, 10), padx=5)

        subframe_tipo_variable = ttk.Frame(frame_config_covariable_seleccionada)
        subframe_tipo_variable.pack(fill=tk.X, pady=5, padx=10)
        ttk.Label(subframe_tipo_variable, text="Tipo de Variable:").pack(side=tk.LEFT, padx=(0, 10))
        self.var_tipo_covariable_seleccionada = StringVar(value="Cuantitativa")
        self.radio_cuantitativa = ttk.Radiobutton(subframe_tipo_variable, text="Cuantitativa", variable=self.var_tipo_covariable_seleccionada, value="Cuantitativa", command=self._toggle_spline_and_refcat_controls, state=tk.DISABLED)
        self.radio_cuantitativa.pack(side=tk.LEFT)
        self.radio_cualitativa = ttk.Radiobutton(subframe_tipo_variable, text="Cualitativa", variable=self.var_tipo_covariable_seleccionada, value="Cualitativa", command=self._toggle_spline_and_refcat_controls, state=tk.DISABLED)
        self.radio_cualitativa.pack(side=tk.LEFT, padx=(10, 0))

        subframe_ref_categoria = ttk.Frame(frame_config_covariable_seleccionada)
        subframe_ref_categoria.pack(fill=tk.X, pady=5, padx=10)
        ttk.Label(subframe_ref_categoria, text="Categoría de Referencia (si Cualitativa y una seleccionada):").pack(side=tk.LEFT, anchor='w')
        self.combo_ref_categoria_seleccionada = ttk.Combobox(subframe_ref_categoria, state="disabled", width=20)
        self.combo_ref_categoria_seleccionada.pack(side=tk.LEFT, padx=5, pady=2, fill=tk.X, expand=True)

        subframe_spline_check = ttk.Frame(frame_config_covariable_seleccionada)
        subframe_spline_check.pack(fill=tk.X, pady=5, padx=10)
        self.var_usar_spline_seleccionada = BooleanVar(value=False)
        self.checkbutton_usar_spline = ttk.Checkbutton(subframe_spline_check, text="Usar Spline (si Cuantitativa(s))", variable=self.var_usar_spline_seleccionada, command=self._toggle_spline_and_refcat_controls, state=tk.DISABLED)
        self.checkbutton_usar_spline.pack(side=tk.LEFT, anchor='w')

        subframe_spline_detalles = ttk.Frame(frame_config_covariable_seleccionada)
        subframe_spline_detalles.pack(fill=tk.X, pady=5, padx=10)
        ttk.Label(subframe_spline_detalles, text="  Tipo de Spline:").pack(side=tk.LEFT, padx=(15, 5))
        self.combo_tipo_spline_seleccionada = ttk.Combobox(subframe_spline_detalles, values=["Natural", "B-spline"], state="disabled", width=12)
        self.combo_tipo_spline_seleccionada.set("Natural")
        self.combo_tipo_spline_seleccionada.pack(side=tk.LEFT, padx=5)

        # Spinbox para grados de libertad (df) del spline
        ttk.Label(subframe_spline_detalles, text="  Grados de Libertad (df):").pack(side=tk.LEFT, padx=(15, 5))
        self.var_df_spline_seleccionada = IntVar(value=4)
        self.spinbox_df_spline = ttk.Spinbox(subframe_spline_detalles, from_=2, to=10, textvariable=self.var_df_spline_seleccionada, width=5, state="disabled")
        self.spinbox_df_spline.pack(side=tk.LEFT, padx=5)

        # Botón para aplicar configuración de covariables
        ttk.Button(frame_config_covariable_seleccionada, text="Aplicar Configuración a Seleccionadas", command=self.apply_covariate_config_to_selected).pack(pady=10)

        # Controles para transformaciones de variables
        frame_transformaciones = ttk.LabelFrame(p_content, text="Transformaciones de Variables")
        frame_transformaciones.pack(fill=tk.X, padx=10, pady=10, ipady=5)

        subframe_log_transform = ttk.Frame(frame_transformaciones, padding=5)
        subframe_log_transform.pack(fill=tk.X, pady=5)
        ttk.Label(subframe_log_transform, text="Variable para Transformación Log:").pack(side=tk.LEFT, padx=5)
        self.combo_var_para_log = ttk.Combobox(subframe_log_transform, state="readonly", width=25)
        self.combo_var_para_log.pack(side=tk.LEFT, padx=5)
        ttk.Label(subframe_log_transform, text="Base:").pack(side=tk.LEFT, padx=5)
        self.combo_base_log = ttk.Combobox(subframe_log_transform, values=["e", "10", "2"], state="readonly", width=5)
        self.combo_base_log.set("e")
        self.combo_base_log.pack(side=tk.LEFT, padx=5)
        ttk.Button(subframe_log_transform, text="Aplicar Log", command=self.convert_to_log_transform).pack(side=tk.LEFT, padx=5)

        subframe_formula_var = ttk.Frame(frame_transformaciones, padding=5)
        subframe_formula_var.pack(fill=tk.X, pady=5)
        ttk.Button(subframe_formula_var, text="Crear Nueva Variable por Fórmula", command=self.create_variable_by_formula).pack(side=tk.LEFT, padx=5)

        self.log("Controles de Preprocesamiento creados.", "DEBUG")
        self.actualizar_controles_preproc() # Llamada inicial para poblar combos

    def actualizar_controles_preproc(self):
        """Actualiza los comboboxes y listboxes con las columnas del DataFrame actual."""
        if self.data is None:
            cols = []
        else:
            cols = sorted(self.data.columns.tolist())

        # Actualizar comboboxes de tiempo y evento
        current_time_val = self.combo_col_tiempo.get()
        current_event_val = self.combo_col_evento.get()
        self.combo_col_tiempo['values'] = cols
        self.combo_col_evento['values'] = cols
        if current_time_val in cols:
            self.combo_col_tiempo.set(current_time_val)
        elif cols:
            self.combo_col_tiempo.set(cols[0])
        else:
            self.combo_col_tiempo.set("")

        if current_event_val in cols:
            self.combo_col_evento.set(current_event_val)
        elif len(cols) > 1: # Intentar seleccionar una diferente a tiempo si hay más de una
            if cols[0] == self.combo_col_tiempo.get() and len(cols) > 1:
                self.combo_col_evento.set(cols[1])
            else:
                self.combo_col_evento.set(cols[0]) # Fallback a la primera si la segunda también es igual o solo hay una
        elif cols: # Si solo queda una columna y no es la de tiempo (ya se asignó arriba)
            self.combo_col_evento.set(cols[0])
        else:
            self.combo_col_evento.set("")


        # Actualizar listbox de covariables
        self.listbox_covariables_disponibles.delete(0, tk.END)
        # Excluir la columna de tiempo y evento de las covariables disponibles
        time_sel = self.combo_col_tiempo.get()
        event_sel = self.combo_col_evento.get()
        cov_cols = [c for c in cols if c not in [time_sel, event_sel]]
        for col in cov_cols:
            self.listbox_covariables_disponibles.insert(tk.END, col)
        
        # Actualizar combobox de variable para transformación log
        current_log_var = self.combo_var_para_log.get()
        numeric_cols = []
        if self.data is not None:
             numeric_cols = [c for c in cols if pd.api.types.is_numeric_dtype(self.data[c])]
        
        self.combo_var_para_log['values'] = numeric_cols
        if current_log_var in numeric_cols:
            self.combo_var_para_log.set(current_log_var)
        elif numeric_cols:
            self.combo_var_para_log.set(numeric_cols[0])
        else:
            self.combo_var_para_log.set("")

        self.on_covariate_select_for_config() # Actualizar UI de configuración de covariable

    def cargar_archivo(self):
        """Permite al usuario seleccionar y cargar un archivo de datos (CSV o Excel)."""
        file_path = filedialog.askopenfilename(
            title="Seleccionar Archivo de Datos",
            filetypes=[("Archivos de Datos", "*.csv *.xlsx *.xls"), ("Todos los Archivos", "*.*")]
        )
        if not file_path:
            self.log("Carga de archivo cancelada.", "INFO")
            return

        self.log(f"Intentando cargar archivo: {file_path}", "INFO")
        try:
            if file_path.lower().endswith('.csv'):
                self.raw_data = pd.read_csv(file_path)
            elif file_path.lower().endswith(('.xlsx', '.xls')):
                self.raw_data = pd.read_excel(file_path)
            else:
                messagebox.showerror("Formato No Soportado", "El archivo seleccionado no es un CSV o Excel válido.", parent=self.parent_for_dialogs)
                self.log(f"Formato de archivo no soportado: {file_path}", "ERROR")
                return

            self.data = self.raw_data.copy() # Trabajar con una copia
            self.log(f"Archivo '{os.path.basename(file_path)}' cargado exitosamente. Filas: {self.data.shape[0]}, Columnas: {self.data.shape[1]}", "SUCCESS")
            self.label_archivo_cargado_info.config(text=f"Cargado: {os.path.basename(file_path)} ({self.data.shape[0]} filas, {self.data.shape[1]} cols)")
            
            # Resetear configuraciones de modelo previas
            self.covariables_type_config = {}
            self.ref_categories_config = {}
            self.spline_config_details = {}
            self.generated_models_data = [] # Limpiar modelos de datos anteriores
            self.selected_model_in_treeview = None
            if hasattr(self, 'treeview_lista_modelos'): self._update_models_treeview()


            self.actualizar_controles_preproc()
            messagebox.showinfo("Carga Exitosa", f"Archivo '{os.path.basename(file_path)}' cargado.", parent=self.parent_for_dialogs)

        except Exception as e:
            messagebox.showerror("Error de Carga", f"No se pudo cargar el archivo:\n{e}", parent=self.parent_for_dialogs)
            self.log(f"Error al cargar archivo '{file_path}': {e}", "ERROR")
            self.raw_data = None
            self.data = None
            self.label_archivo_cargado_info.config(text="Error al cargar archivo.")
            traceback.print_exc(limit=3)

    def _apply_fc_filters_to_main_data(self):
        """Aplica los filtros definidos en FilterComponent al DataFrame principal."""
        if self.data is None:
            messagebox.showwarning("Sin Datos", "Cargue datos primero para aplicar filtros.", parent=self.parent_for_dialogs)
            return
        if not FILTER_COMPONENT_AVAILABLE or self.custom_filter_component_instance is None:
            messagebox.showerror("Error", "Componente de filtro no disponible.", parent=self.parent_for_dialogs)
            return

        self.log("Aplicando filtros avanzados al dataset principal...", "INFO")
        try:
            original_rows = self.data.shape[0]
            # Pasar el DataFrame actual para que el FilterComponent lo use
            # Asumimos que FilterComponent tiene un método que toma el df y devuelve el filtrado
            # o modifica el que tiene internamente y lo podemos obtener.
            # Por el nombre `apply_filters`, asumimos que retorna el filtrado.
            filtered_data = self.custom_filter_component_instance.apply_filters_to_df(self.data.copy()) # Pasar una copia

            if filtered_data is not None:
                if filtered_data.empty and original_rows > 0:
                    if not messagebox.askyesno("Dataset Vacío", "La aplicación de filtros resultó en un dataset vacío. ¿Desea continuar con el dataset vacío o revertir?", parent=self.parent_for_dialogs):
                        self.log("Aplicación de filtros que resultó en dataset vacío fue revertida por el usuario.", "INFO")
                        return # No modificar self.data

                self.data = filtered_data # Actualizar el DataFrame principal
                rows_after_filter = self.data.shape[0]
                self.log(f"Filtros aplicados. Filas originales: {original_rows}, Filas después de filtro: {rows_after_filter} (Eliminadas: {original_rows - rows_after_filter})", "SUCCESS")
                
                # Actualizar etiqueta del archivo cargado
                current_label_text = self.label_archivo_cargado_info.cget("text")
                base_filename_match = re.match(r"Cargado: ([^(\s]+)", current_label_text)
                base_filename = base_filename_match.group(1) if base_filename_match else "Archivo"
                
                self.label_archivo_cargado_info.config(text=f"Cargado: {base_filename} ({self.data.shape[0]} filas, {self.data.shape[1]} cols) [FILTRADO]")
                
                self.actualizar_controles_preproc() # Actualizar comboboxes, etc.
                messagebox.showinfo("Filtros Aplicados", f"Filtros aplicados exitosamente. {original_rows - rows_after_filter} filas eliminadas.", parent=self.parent_for_dialogs)
            else:
                self.log("La aplicación de filtros no retornó datos (None). No se modificó el dataset.", "WARN")
                messagebox.showwarning("Filtros No Aplicados", "La operación de filtro no resultó en un DataFrame válido (retornó None).", parent=self.parent_for_dialogs)

        except Exception as e:
            self.log(f"Error al aplicar filtros avanzados: {e}", "ERROR")
            messagebox.showerror("Error de Filtro", f"No se pudieron aplicar los filtros:\n{e}", parent=self.parent_for_dialogs)
            traceback.print_exc(limit=3)

    def on_covariate_select_for_config(self, event=None):
        """Actualiza la UI de configuración de covariables cuando se selecciona una en la listbox."""
        sel_idx = self.listbox_covariables_disponibles.curselection()
        if not hasattr(self, 'label_cov_seleccionada_nombre'): # UI no completamente creada
            return

        if len(sel_idx) == 1:
            var_name = self.listbox_covariables_disponibles.get(sel_idx[0])
            self.update_cov_config_ui_for_var(var_name, multiple_selected=False)
        elif len(sel_idx) > 1:
            # Múltiples seleccionados, solo permitir cambiar tipo y spline común
            self.update_cov_config_ui_for_var(None, multiple_selected=True)
        else: # Ninguno seleccionado
            self.update_cov_config_ui_for_var(None, multiple_selected=False)

    def update_cov_config_ui_for_var(self, var_name_cfg, multiple_selected=False):
        """
        Actualiza los controles de configuración de una covariable específica
        o los adapta si hay múltiples/ninguna seleccionada.
        """
        # Lista de atributos de UI esperados para la configuración
        ui_attrs_expected = [
            'label_cov_seleccionada_nombre', 'var_tipo_covariable_seleccionada',
            'radio_cuantitativa', 'radio_cualitativa', 
            'combo_ref_categoria_seleccionada', 'var_usar_spline_seleccionada', 
            'checkbutton_usar_spline', 'combo_tipo_spline_seleccionada', 
            'var_df_spline_seleccionada', 'spinbox_df_spline'
        ]
        if not all(hasattr(self, attr) for attr in ui_attrs_expected):
            self.log("Advertencia: Faltan atributos de UI para configurar covariables. UI puede estar incompleta.", "WARN")
            return

        is_single_selection = (var_name_cfg is not None) and (not multiple_selected)
        
        if multiple_selected:
            self.label_cov_seleccionada_nombre.config(text=f"{len(self.listbox_covariables_disponibles.curselection())} Variables Seleccionadas")
            # Habilitar cambio de tipo y spline si todas son compatibles
            # Por ahora, permitir cambiar tipo. Spline se habilita/deshabilita en _toggle.
            self.radio_cuantitativa.config(state=tk.NORMAL)
            self.radio_cualitativa.config(state=tk.NORMAL)
            # Ref categoría deshabilitada para múltiple selección
            self.combo_ref_categoria_seleccionada.set("")
            self.combo_ref_categoria_seleccionada.config(state="disabled", values=[])
            # Spline: se maneja en _toggle_spline_and_refcat_controls
        elif is_single_selection and self.data is not None and var_name_cfg in self.data.columns:
            self.label_cov_seleccionada_nombre.config(text=var_name_cfg)
            self.radio_cuantitativa.config(state=tk.NORMAL)
            self.radio_cualitativa.config(state=tk.NORMAL)

            # Inferir tipo de datos si no está configurado
            inferred_dtype = "Cuantitativa" if pd.api.types.is_numeric_dtype(self.data[var_name_cfg]) else "Cualitativa"
            current_var_type = self.covariables_type_config.get(var_name_cfg, inferred_dtype)
            self.var_tipo_covariable_seleccionada.set(current_var_type)

            if current_var_type == "Cualitativa":
                unique_cats = sorted(list(self.data[var_name_cfg].astype(str).unique()))
                self.combo_ref_categoria_seleccionada['values'] = unique_cats
                current_ref_cat = self.ref_categories_config.get(var_name_cfg)
                if current_ref_cat in unique_cats:
                    self.combo_ref_categoria_seleccionada.set(current_ref_cat)
                elif unique_cats: # Default a la primera si no hay config o la config no es válida
                    self.combo_ref_categoria_seleccionada.set(unique_cats[0])
                else: # Sin categorías
                    self.combo_ref_categoria_seleccionada.set("")
            else: # Cuantitativa
                self.combo_ref_categoria_seleccionada.set("")
                self.combo_ref_categoria_seleccionada.config(state="disabled", values=[])
            
            # Configuración de Spline
            if current_var_type == "Cuantitativa" and var_name_cfg in self.spline_config_details:
                self.var_usar_spline_seleccionada.set(True)
                spl_conf = self.spline_config_details[var_name_cfg]
                self.combo_tipo_spline_seleccionada.set(spl_conf.get('type', 'Natural'))
                self.var_df_spline_seleccionada.set(spl_conf.get('df', 4))
            elif current_var_type == "Cuantitativa": # Es cuantitativa pero sin config de spline
                 self.var_usar_spline_seleccionada.set(False) # Asegurar que esté desactivado
                 self.combo_tipo_spline_seleccionada.set('Natural') # Default
                 self.var_df_spline_seleccionada.set(4) # Default
            else: # Cualitativa, spline no aplica
                self.var_usar_spline_seleccionada.set(False)

        else: # Ninguna seleccionada o error
            self.label_cov_seleccionada_nombre.config(text="Ninguna Seleccionada")
            self.radio_cuantitativa.config(state=tk.DISABLED)
            self.radio_cualitativa.config(state=tk.DISABLED)
            self.var_tipo_covariable_seleccionada.set("Cuantitativa") # Reset a default
            self.combo_ref_categoria_seleccionada.set("")
            self.combo_ref_categoria_seleccionada.config(state="disabled", values=[])
            self.var_usar_spline_seleccionada.set(False)
            # Los demás (checkbutton_usar_spline, etc.) se manejan en _toggle

        self._toggle_spline_and_refcat_controls()


    def _toggle_spline_and_refcat_controls(self, event=None):
        """Habilita/deshabilita controles de spline y categoría de referencia."""
        sel_indices = self.listbox_covariables_disponibles.curselection()
        num_selected = len(sel_indices)
        
        # Estado base de los radios de tipo de variable
        type_radio_state = tk.NORMAL if num_selected > 0 else tk.DISABLED
        self.radio_cuantitativa.config(state=type_radio_state)
        self.radio_cualitativa.config(state=type_radio_state)

        current_type_choice = self.var_tipo_covariable_seleccionada.get()

        # Categoría de Referencia: solo para 1 cualitativa seleccionada
        if num_selected == 1 and current_type_choice == "Cualitativa":
            self.combo_ref_categoria_seleccionada.config(state="readonly")
        else:
            self.combo_ref_categoria_seleccionada.config(state="disabled")
            if num_selected != 1: # Limpiar si no es selección única
                 self.combo_ref_categoria_seleccionada.set("")
                 self.combo_ref_categoria_seleccionada['values'] = []


        # Spline: solo para cuantitativas (1 o más)
        can_use_spline = (num_selected > 0 and current_type_choice == "Cuantitativa")
        self.checkbutton_usar_spline.config(state=tk.NORMAL if can_use_spline else tk.DISABLED)
        if not can_use_spline: # Si no se puede usar spline, desactivar el check
            self.var_usar_spline_seleccionada.set(False)
        
        # Detalles de Spline: si se marca "Usar Spline" y es aplicable
        spline_details_state = "readonly" if self.var_usar_spline_seleccionada.get() and can_use_spline else "disabled"
        self.combo_tipo_spline_seleccionada.config(state=spline_details_state)
        self.spinbox_df_spline.config(state=spline_details_state)
        # Asegurar que los valores de spline no se mantengan si se cambia de tipo o se desmarca
        if spline_details_state == "disabled":
            self.combo_tipo_spline_seleccionada.set("Natural") # Reset
            self.var_df_spline_seleccionada.set(4) # Reset


    def apply_covariate_config_to_selected(self):
        """Aplica la configuración de tipo, spline o categoría de referencia a las covariables seleccionadas."""
        sel_indices = self.listbox_covariables_disponibles.curselection()
        if not sel_indices:
            messagebox.showwarning("Sin Selección", "Seleccione una o más covariables para aplicar la configuración.", parent=self.parent_for_dialogs)
            return

        selected_var_names = [self.listbox_covariables_disponibles.get(i) for i in sel_indices]
        
        new_var_type = self.var_tipo_covariable_seleccionada.get()
        use_spline_config = self.var_usar_spline_seleccionada.get()
        spline_type_config = self.combo_tipo_spline_seleccionada.get()
        spline_df_config = self.var_df_spline_seleccionada.get()
        
        ref_category_config = None
        if len(selected_var_names) == 1 and new_var_type == "Cualitativa" and self.combo_ref_categoria_seleccionada.cget('state') != 'disabled':
            ref_category_config = self.combo_ref_categoria_seleccionada.get()
            if not ref_category_config: # Si está vacío pero debería haber uno
                messagebox.showwarning("Ref. Vacía", "Seleccione una categoría de referencia para la variable cualitativa.", parent=self.parent_for_dialogs)
                return


        num_applied = 0
        for var_name_apply in selected_var_names:
            log_msgs_for_var = [f"Aplicando config a '{var_name_apply}':"]
            
            # Aplicar Tipo
            self.covariables_type_config[var_name_apply] = new_var_type
            log_msgs_for_var.append(f"Tipo='{new_var_type}'")

            if new_var_type == "Cualitativa":
                # Limpiar config de spline si existía
                if var_name_apply in self.spline_config_details:
                    del self.spline_config_details[var_name_apply]
                    log_msgs_for_var.append("Spline eliminado.")
                
                # Aplicar categoría de referencia (solo si es la única seleccionada y se configuró)
                if len(selected_var_names) == 1 and ref_category_config is not None:
                    self.ref_categories_config[var_name_apply] = ref_category_config
                    log_msgs_for_var.append(f"Ref.Cat.='{ref_category_config}'")
                elif var_name_apply not in self.ref_categories_config and self.data is not None and var_name_apply in self.data.columns:
                    # Si no hay ref explícita y no es selección única, asignar default (primera alfabéticamente)
                    # Esto asegura que todas las cualitativas tengan una ref.
                    cats_apply = sorted(list(self.data[var_name_apply].astype(str).unique()))
                    if cats_apply:
                        self.ref_categories_config[var_name_apply] = cats_apply[0]
                        log_msgs_for_var.append(f"Ref.Cat.(default)='{cats_apply[0]}'")
            
            elif new_var_type == "Cuantitativa":
                # Limpiar config de ref. categoría si existía
                if var_name_apply in self.ref_categories_config:
                    del self.ref_categories_config[var_name_apply]
                    log_msgs_for_var.append("Ref.Cat. eliminada.")
                
                # Aplicar o limpiar config de Spline
                if use_spline_config:
                    self.spline_config_details[var_name_apply] = {'type': spline_type_config, 'df': spline_df_config}
                    log_msgs_for_var.append(f"Spline: Tipo='{spline_type_config}', DF={spline_df_config}")
                elif var_name_apply in self.spline_config_details: # No usar spline, pero existía config
                    del self.spline_config_details[var_name_apply]
                    log_msgs_for_var.append("Spline eliminado.")
            
            self.log(" ".join(log_msgs_for_var), "CONFIG")
            num_applied += 1

        if num_applied > 0:
            messagebox.showinfo("Configuración Aplicada", f"Configuración aplicada a {num_applied} variable(s).", parent=self.parent_for_dialogs)
        
        # Re-actualizar la UI de configuración para reflejar los cambios,
        # especialmente si la selección actual es una de las modificadas.
        self.on_covariate_select_for_config()


    def convert_to_log_transform(self):
        """Realiza una transformación logarítmica en la columna numérica seleccionada."""
        if self.data is None:
            messagebox.showerror("Error", "No hay datos cargados para realizar la transformación.", parent=self.parent_for_dialogs)
            return

        v_name = self.combo_var_para_log.get().strip()
        base = self.combo_base_log.get().strip()

        if not v_name:
            messagebox.showwarning("Selección Requerida", "Seleccione una variable para aplicar la transformación logarítmica.", parent=self.parent_for_dialogs)
            return
        if v_name not in self.data.columns:
            messagebox.showerror("Error", f"La variable '{v_name}' no se encontró en el dataset actual.", parent=self.parent_for_dialogs)
            return
        if not pd.api.types.is_numeric_dtype(self.data[v_name]):
            messagebox.showerror("Error", f"La variable '{v_name}' no es numérica y no puede ser transformada logarítmicamente.", parent=self.parent_for_dialogs)
            return

        new_col_name = f"{v_name}_log{base if base != 'e' else ''}"
        if new_col_name in self.data.columns:
            if not messagebox.askyesno("Sobrescribir Columna", f"La columna '{new_col_name}' ya existe. ¿Desea sobrescribirla?", parent=self.parent_for_dialogs):
                self.log("Transformación logarítmica cancelada por el usuario (columna existente).", "INFO")
                return

        series_to_transform = self.data[v_name].copy()
        non_positive_values = (series_to_transform <= 0)
        if non_positive_values.any():
            self.log(f"Advertencia: La variable '{v_name}' contiene {non_positive_values.sum()} valor(es) no positivo(s) (<= 0). Estos valores se convertirán a NaN antes de la transformación logarítmica.", "WARN")
            series_to_transform[non_positive_values] = np.nan

        try:
            if base == "e":
                self.data[new_col_name] = np.log(series_to_transform)
            elif base == "10":
                self.data[new_col_name] = np.log10(series_to_transform)
            elif base == "2":
                self.data[new_col_name] = np.log2(series_to_transform)
            else:
                messagebox.showerror("Error", f"Base logarítmica '{base}' no reconocida. Use 'e', '10' o '2'.", parent=self.parent_for_dialogs)
                self.log(f"Error: Base logarítmica no válida '{base}'.", "ERROR")
                return

            self.log(f"Variable '{v_name}' transformada logarítmicamente (base {base}) a una nueva columna '{new_col_name}'.", "SUCCESS")
            self.actualizar_controles_preproc()
            messagebox.showinfo("Transformación Exitosa", f"La variable '{v_name}' ha sido transformada a '{new_col_name}'.", parent=self.parent_for_dialogs)
        except Exception as e:
            messagebox.showerror("Error de Transformación", f"Fallo al aplicar la transformación logarítmica:\n{e}", parent=self.parent_for_dialogs)
            self.log(f"Error al aplicar transformación logarítmica para '{v_name}': {e}", "ERROR")
            traceback.print_exc(limit=3)


    def create_variable_by_formula(self):
        """Crea una nueva variable en el DataFrame usando una fórmula de Pandas eval."""
        if self.data is None:
            messagebox.showerror("Error", "No hay datos cargados para crear una variable.", parent=self.parent_for_dialogs)
            return

        new_var_name = simpledialog.askstring(
            "Crear Nueva Variable",
            "Ingrese el nombre para la nueva variable:",
            parent=self.parent_for_dialogs)

        if not new_var_name or not new_var_name.strip():
            self.log("Creación de variable por fórmula cancelada (sin nombre).", "INFO")
            return
        new_var_name = new_var_name.strip()
        # Validar nombre de variable (simple check)
        if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", new_var_name):
            messagebox.showerror("Nombre Inválido", "El nombre de la variable debe ser un identificador Python válido (letras, números, guion bajo, no empezar con número).", parent=self.parent_for_dialogs)
            return


        if new_var_name in self.data.columns:
            messagebox.showerror("Error", f"La variable '{new_var_name}' ya existe.", parent=self.parent_for_dialogs)
            return

        available_cols_example = ", ".join(self.data.columns[:min(5, len(self.data.columns))])
        if len(self.data.columns) > 5:
            available_cols_example += "..."

        formula_str = simpledialog.askstring(
            "Ingresar Fórmula",
            f"Columnas disponibles (ejemplo): {available_cols_example}\n"
            "Use nombres de columna directamente. Si tienen espacios o caracteres especiales, "
            "use Q('nombre de columna con espacios') o renombre la columna previamente.\n"
            "Funciones NumPy (np.), math (math.), Pandas (pd.) están disponibles.\n"
            "Ejemplo: `col1 * 2 + np.log(Q('otra columna'))`",
            parent=self.parent_for_dialogs)

        if not formula_str:
            self.log("Creación de variable por fórmula cancelada (sin fórmula).", "INFO")
            return

        try:
            env = {'np': np, 'math': math, 'pd': pd}
            temp_df_for_eval = self.data.copy() # Evaluar sobre una copia
            
            # pandas.eval es más seguro y a menudo más rápido que Python eval()
            # Para acceder a columnas con espacios o caracteres especiales, Patsy usa Q()
            # pandas.eval usa `@` para variables locales o `df[]` notación.
            # Dado que el prompt sugiere Q(), es mejor si el usuario limpia los nombres de columna
            # o si usamos un motor que entienda Q() (como Patsy, pero eso es para design matrices).
            # Aquí, para df.eval, el usuario debe usar nombres de columna válidos o backticks ``.
            
            # Para simplificar, asumimos que el usuario usará nombres de columna válidos o
            # que `pandas.eval` con `engine='python'` puede manejarlo si se pasan las columnas
            # como `local_dict`.

            # Usar local_dict para que las columnas sean accesibles directamente.
            # `DataFrame.eval` con `engine='python'` y `local_dict` es una opción.
            local_env_for_eval = {**env, **{col_name: temp_df_for_eval[col_name] for col_name in temp_df_for_eval.columns}}

            # Intentar parsear la fórmula para reemplazar Q('col name') con un formato que eval entienda
            # o instruir al usuario. Por ahora, asumimos que los nombres son "limpios" o que usan backticks.
            # Para Q(), necesitaríamos un pre-procesamiento de `formula_str`.
            # Un simple reemplazo si Q() solo envuelve nombres de columna:
            # formula_for_eval = re.sub(r"Q\('(.*?)'\)", r"\1", formula_str) # Simplista
            # O, mejor, que el usuario use `df['col name with space']` si es necesario.
            # El ejemplo actual de `col1 * 2 + np.log(Q('otra columna'))` no funcionará directamente
            # con `df.eval` si 'otra columna' tiene espacios.
            # Le diremos al usuario que use `\`nombre de columna con espacios\``.

            # Con `engine='python'`, podemos usar los nombres de columna directamente si están en `local_dict`.
            # Y `@variable` para las del `env`.
            # Para la fórmula dada, `np.log` es de `env`. `col1` y `Q('otra columna')` deberían estar en `local_dict`.
            # Si `Q('otra columna')` es una columna llamada literalmente "Q('otra columna')", necesita estar en `local_dict`.
            # Si es una forma de referenciar 'otra columna', necesitamos procesarlo.
            
            # La forma más robusta con `engine='python'` es que `formula_str` use los nombres
            # de las columnas como están en el DataFrame.
            # `pd.eval` es la función global, `df.eval` es el método.
            
            # Si el usuario escribe "col1 + col2", y col1, col2 son columnas:
            result_series = temp_df_for_eval.eval(formula_str, engine='python', local_dict=env)
            
            if not isinstance(result_series, pd.Series):
                raise ValueError(f"La fórmula no resultó en una Serie de Pandas (obtenido: {type(result_series)}).")
            if len(result_series) != len(self.data):
                 raise ValueError(f"La serie resultante ({len(result_series)} elementos) no coincide en longitud con el DataFrame ({len(self.data)} elementos).")

            self.data[new_var_name] = result_series.values 
            
            self.log(f"Nueva variable '{new_var_name}' creada con fórmula: '{formula_str}'.", "SUCCESS")
            self.actualizar_controles_preproc()
            messagebox.showinfo("Variable Creada", f"La variable '{new_var_name}' ha sido creada y añadida al dataset.", parent=self.parent_for_dialogs)

        except Exception as e:
            err_details = traceback.format_exc()
            messagebox.showerror(
                "Error de Fórmula",
                f"Error al evaluar la fórmula para '{new_var_name}':\n{e}\n\n"
                "Asegúrese que los nombres de columna con espacios o caracteres especiales estén entre acentos graves (backticks), ej: \`nombre con espacio\`.\n"
                f"Detalles del error:\n{err_details[:600]}...",
                parent=self.parent_for_dialogs)
            self.log(f"Error al crear variable '{new_var_name}' con fórmula '{formula_str}': {e}\n{err_details}", "ERROR")

    # --- MÉTODOS PARA PESTAÑA 2: MODELADO COX ---
    def create_grid_controls(self):
        g_content = self.tab_frame_modeling_content.interior 
        self.log("Creando controles para la pestaña de Modelado Cox...", "DEBUG")
        
        # Configuración General
        frame_config_general = ttk.LabelFrame(g_content, text="Configuración General del Modelo Cox", padding=10)
        frame_config_general.pack(fill=tk.X, padx=10, pady=10)

        # PanedWindow para dividir en dos columnas
        paned_config = ttk.PanedWindow(frame_config_general, orient=tk.HORIZONTAL)
        paned_config.pack(fill=tk.BOTH, expand=True)

        # --- Columna Izquierda ---
        left_col_frame = ttk.Frame(paned_config, padding=5)
        paned_config.add(left_col_frame, weight=1)

        # Tipo de Modelado
        frame_tipo_modelado = ttk.Frame(left_col_frame)
        frame_tipo_modelado.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(frame_tipo_modelado, text="Tipo de Modelado:").pack(side=tk.LEFT, padx=(0,5))
        ttk.Radiobutton(frame_tipo_modelado, text="Multivariado", variable=self.cox_model_type_var, value="Multivariado").pack(side=tk.LEFT, padx=3)
        ttk.Radiobutton(frame_tipo_modelado, text="Univariado", variable=self.cox_model_type_var, value="Univariado").pack(side=tk.LEFT, padx=3)

        # Selección de Variables
        frame_sel_vars = ttk.LabelFrame(left_col_frame, text="Selección de Variables (para Multivariado)")
        frame_sel_vars.pack(fill=tk.X, expand=True, pady=(0,10))
        
        grid_sel_vars = ttk.Frame(frame_sel_vars, padding=5)
        grid_sel_vars.pack(fill=tk.X)
        ttk.Label(grid_sel_vars, text="Método:").grid(row=0, column=0, padx=5, pady=3, sticky=tk.W)
        metodos_sel = ["Ninguno (usar todas)", "Backward", "Forward", "Stepwise (Fwd luego Bwd)"]
        self.combo_metodo_seleccion_vars = ttk.Combobox(grid_sel_vars, textvariable=self.var_selection_method_var, values=metodos_sel, state="readonly", width=25)
        self.combo_metodo_seleccion_vars.grid(row=0, column=1, columnspan=3, padx=5, pady=3, sticky=tk.EW)
        self.var_selection_method_var.set("Ninguno (usar todas)") # Default
        
        ttk.Label(grid_sel_vars, text="P para Entrar:").grid(row=1, column=0, padx=5, pady=3, sticky=tk.W)
        ttk.Entry(grid_sel_vars, textvariable=self.p_enter_var, width=8).grid(row=1, column=1, padx=5, pady=3, sticky=tk.W)
        ttk.Label(grid_sel_vars, text="P para Salir:").grid(row=1, column=2, padx=5, pady=3, sticky=tk.W)
        ttk.Entry(grid_sel_vars, textvariable=self.p_remove_var, width=8).grid(row=1, column=3, padx=5, pady=3, sticky=tk.W)
        grid_sel_vars.columnconfigure(1, weight=1); grid_sel_vars.columnconfigure(3, weight=1)

        # --- Columna Derecha ---
        right_col_frame = ttk.Frame(paned_config, padding=5)
        paned_config.add(right_col_frame, weight=1)

        # Regularización
        frame_reg = ttk.LabelFrame(right_col_frame, text="Regularización (Penalización)")
        frame_reg.pack(fill=tk.X, expand=True, pady=(0,10))
        grid_reg = ttk.Frame(frame_reg, padding=5); grid_reg.pack(fill=tk.X)
        
        ttk.Label(grid_reg, text="Tipo:").grid(row=0, column=0, padx=5, pady=3, sticky=tk.W)
        tipos_pen = ["Ninguna", "L2 (Ridge)", "L1 (Lasso)", "ElasticNet"]
        self.combo_tipo_penalizacion = ttk.Combobox(grid_reg, textvariable=self.penalization_method_var, values=tipos_pen, state="readonly", width=18)
        self.combo_tipo_penalizacion.grid(row=0, column=1, padx=5, pady=3, sticky=tk.EW)
        self.penalization_method_var.set("Ninguna") # Default
        self.combo_tipo_penalizacion.bind("<<ComboboxSelected>>", self._toggle_penalization_params_ui_state)
        
        ttk.Label(grid_reg, text="Valor Penalización:").grid(row=1, column=0, padx=5, pady=3, sticky=tk.W)
        self.entry_valor_penalizacion = ttk.Entry(grid_reg, textvariable=self.penalizer_strength_var, width=10, state=tk.DISABLED)
        self.entry_valor_penalizacion.grid(row=1, column=1, padx=5, pady=3, sticky=tk.W)
        
        ttk.Label(grid_reg, text="Ratio L1 (ElasticNet):").grid(row=2, column=0, padx=5, pady=3, sticky=tk.W)
        self.entry_ratio_l1 = ttk.Entry(grid_reg, textvariable=self.l1_ratio_for_elasticnet_var, width=8, state=tk.DISABLED)
        self.entry_ratio_l1.grid(row=2, column=1, padx=5, pady=3, sticky=tk.W)
        grid_reg.columnconfigure(1, weight=1)

        # Manejo de Empates
        frame_ties = ttk.LabelFrame(right_col_frame, text="Manejo de Empates en Tiempos")
        frame_ties.pack(fill=tk.X, expand=True, pady=(0,10))
        grid_ties = ttk.Frame(frame_ties, padding=5); grid_ties.pack(fill=tk.X)
        ttk.Label(grid_ties, text="Método:").grid(row=0, column=0, padx=5, pady=3, sticky=tk.W)
        metodos_ties = ["efron", "breslow", "exact"]
        self.combo_metodo_empates = ttk.Combobox(grid_ties, textvariable=self.tie_handling_method_var, values=metodos_ties, state="readonly", width=18)
        self.combo_metodo_empates.grid(row=0, column=1, padx=5, pady=3, sticky=tk.EW)
        self.tie_handling_method_var.set("efron") # Default
        grid_ties.columnconfigure(1, weight=1)
        
        # Validación Cruzada C-Index
        frame_cv = ttk.LabelFrame(g_content, text="C-Index por Validación Cruzada (Opcional)")
        frame_cv.pack(fill=tk.X, padx=10, pady=10)
        grid_cv_ui = ttk.Frame(frame_cv, padding=5); grid_cv_ui.pack(fill=tk.X)
        
        ttk.Checkbutton(grid_cv_ui, text="Calcular C-Index con CV", variable=self.calculate_cv_cindex_var).grid(row=0, column=0, padx=5, pady=3, sticky=tk.W)
        ttk.Label(grid_cv_ui, text="Num. Folds (K):").grid(row=0, column=1, padx=15, pady=3, sticky=tk.W)
        ttk.Entry(grid_cv_ui, textvariable=self.cv_num_kfolds_var, width=5).grid(row=0, column=2, padx=5, pady=3, sticky=tk.W)
        ttk.Label(grid_cv_ui, text="Semilla Aleatoria:").grid(row=0, column=3, padx=15, pady=3, sticky=tk.W)
        ttk.Entry(grid_cv_ui, textvariable=self.cv_random_seed_var, width=7).grid(row=0, column=4, padx=5, pady=3, sticky=tk.W)

        # Botón Ejecutar
        frame_ejecutar = ttk.Frame(g_content); frame_ejecutar.pack(fill=tk.X, pady=(15, 10))
        btn_ejecutar = ttk.Button(frame_ejecutar, text="▶ Ejecutar Modelado Cox", command=self._execute_cox_modeling_orchestrator)
        btn_ejecutar.pack(padx=10, pady=5, ipady=5)

        # Treeview para Modelos Generados
        self.frame_modelos_generados_display = ttk.LabelFrame(g_content, text="Modelos Cox Generados en esta Sesión")
        self.frame_modelos_generados_display.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        cols_tv = ("#", "Nombre Modelo", "Covariables (Términos)", "C-Index (Train)", "AIC", "LogLik", "Schoenfeld (p global)")
        self.treeview_lista_modelos = ttk.Treeview(self.frame_modelos_generados_display, columns=cols_tv, show="headings", height=7)
        
        col_widths = {"#": 40, "Nombre Modelo": 200, "Covariables (Términos)": 280, "C-Index (Train)": 100, "AIC": 100, "LogLik": 100, "Schoenfeld (p global)": 140}
        col_anchors = {"#": tk.CENTER, "C-Index (Train)": tk.E, "AIC": tk.E, "LogLik": tk.E, "Schoenfeld (p global)": tk.E}
        for col in cols_tv:
            self.treeview_lista_modelos.heading(col, text=col)
            self.treeview_lista_modelos.column(col, width=col_widths.get(col, 120), anchor=col_anchors.get(col, tk.W), minwidth=max(40, col_widths.get(col, 60)//2))
            
        ysb_tv = ttk.Scrollbar(self.frame_modelos_generados_display, orient=tk.VERTICAL, command=self.treeview_lista_modelos.yview)
        xsb_tv = ttk.Scrollbar(self.frame_modelos_generados_display, orient=tk.HORIZONTAL, command=self.treeview_lista_modelos.xview)
        self.treeview_lista_modelos.configure(yscrollcommand=ysb_tv.set, xscrollcommand=xsb_tv.set)
        ysb_tv.pack(side=tk.RIGHT, fill=tk.Y); xsb_tv.pack(side=tk.BOTTOM, fill=tk.X)
        self.treeview_lista_modelos.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.treeview_lista_modelos.bind("<<TreeviewSelect>>", self._on_model_select_from_treeview)

        # Botones de Acción para Modelo Seleccionado
        frame_acciones = ttk.Frame(self.frame_modelos_generados_display, padding=(0,5,0,0))
        frame_acciones.pack(fill=tk.X, pady=5)
        
        acciones_config_btns = [
            ("Ver Resumen", self.show_selected_model_summary), ("Gráf. Schoenfeld", self.show_schoenfeld),
            ("Sup. Base", self.show_baseline_survival), ("Riesgo Acum. Base", self.show_baseline_hazard),
            ("Forest Plot", self.generar_forest_plot), ("Gráf. Calibración", self.generate_calibration_plot),
            ("Predicción", self.realizar_prediccion), ("Exportar Resumen", self.export_model_summary),
            ("Guardar Modelo", self.save_model), ("Cargar Modelo", self.load_model_from_file),
            ("Reporte Metod.", self.show_methodological_report)
        ]
        
        # Layout dinámico para botones de acción
        max_btns_per_row = 6 
        current_row_frame_acciones = None
        for i, (text, cmd) in enumerate(acciones_config_btns):
            if i % max_btns_per_row == 0:
                current_row_frame_acciones = ttk.Frame(frame_acciones)
                current_row_frame_acciones.pack(fill=tk.X, pady=1)
            ttk.Button(current_row_frame_acciones, text=text, command=cmd).pack(side=tk.LEFT, padx=3, pady=2, fill=tk.X, expand=True)

        self.log("Controles de Modelado Cox creados.", "DEBUG")
        self._toggle_penalization_params_ui_state() # Estado inicial de UI de penalización

    def _toggle_penalization_params_ui_state(self, event=None):
        pen_method = self.penalization_method_var.get()
        
        # Estado para valor de penalización
        if pen_method != "Ninguna":
            self.entry_valor_penalizacion.config(state=tk.NORMAL)
        else:
            self.entry_valor_penalizacion.config(state=tk.DISABLED)
            self.penalizer_strength_var.set(0.0) # Resetear valor si no hay penalización

        # Estado para L1 ratio
        if pen_method == "ElasticNet":
            self.entry_ratio_l1.config(state=tk.NORMAL)
        else:
            self.entry_ratio_l1.config(state=tk.DISABLED)
            if pen_method == "L1 (Lasso)": self.l1_ratio_for_elasticnet_var.set(1.0)
            elif pen_method == "L2 (Ridge)": self.l1_ratio_for_elasticnet_var.set(0.0)
            else: self.l1_ratio_for_elasticnet_var.set(0.5) # Default si no es relevante

    # --- MÉTODOS PARA PESTAÑA 2: MODELADO COX --- (Continuación Lógica)

    def _preparar_datos_para_modelado(self):
        if self.data is None or self.data.empty:
            self.log("No hay datos cargados.", "WARN"); messagebox.showwarning("Sin Datos", "Cargue datos primero.", parent=self.parent_for_dialogs); return None
        
        time_col_ui = self.combo_col_tiempo.get().strip()
        event_col_ui = self.combo_col_evento.get().strip()
        rename_time_to = self.entry_renombrar_col_tiempo.get().strip()
        rename_event_to = self.entry_renombrar_col_evento.get().strip()

        if not time_col_ui or not event_col_ui:
            self.log("Columnas T/E no seleccionadas.", "WARN"); messagebox.showwarning("Variables Faltantes", "Seleccione Tiempo y Evento.", parent=self.parent_for_dialogs); return None
        if time_col_ui not in self.data.columns or event_col_ui not in self.data.columns:
            self.log(f"Columnas T/E ('{time_col_ui}', '{event_col_ui}') no en datos.", "ERROR"); messagebox.showerror("Columnas Inválidas", "Columnas T/E no existen.", parent=self.parent_for_dialogs); return None

        sel_cov_indices = self.listbox_covariables_disponibles.curselection()
        selected_covs_orig_names = [self.listbox_covariables_disponibles.get(i) for i in sel_cov_indices if self.listbox_covariables_disponibles.get(i) not in [time_col_ui, event_col_ui]]

        if self.cox_model_type_var.get() == "Multivariado" and not selected_covs_orig_names:
             self.log("Multivariado sin covariables -> modelo nulo.", "INFO")
        elif self.cox_model_type_var.get() == "Univariado" and not selected_covs_orig_names:
             self.log("Univariado sin covariables. Abortando.", "WARN"); messagebox.showwarning("Sin Covariables", "Seleccione covariables para univariado.", parent=self.parent_for_dialogs); return None

        df_model_prep = self.data[[time_col_ui, event_col_ui] + selected_covs_orig_names].copy()

        final_t_col = rename_time_to if rename_time_to and rename_time_to != time_col_ui else time_col_ui
        final_e_col = rename_event_to if rename_event_to and rename_event_to != event_col_ui else event_col_ui
        
        renames = {}
        if final_t_col != time_col_ui:
            if final_t_col in df_model_prep.columns and final_t_col != time_col_ui : # Evitar renombrar a sí misma o a una existente accidentalmente
                 self.log(f"Advertencia: Nombre renombrado para Tiempo '{final_t_col}' ya existe o es el original. No se renombrará.", "WARN"); final_t_col = time_col_ui
            else: renames[time_col_ui] = final_t_col
        if final_e_col != event_col_ui:
            if final_e_col in df_model_prep.columns and final_e_col != event_col_ui:
                 self.log(f"Advertencia: Nombre renombrado para Evento '{final_e_col}' ya existe o es el original. No se renombrará.", "WARN"); final_e_col = event_col_ui
            else: renames[event_col_ui] = final_e_col
        
        if renames: df_model_prep.rename(columns=renames, inplace=True); self.log(f"Columnas renombradas: {renames}", "INFO")

        initial_rows_prep = len(df_model_prep)
        df_model_prep.dropna(subset=[final_t_col, final_e_col], inplace=True)
        if len(df_model_prep) < initial_rows_prep: self.log(f"Eliminadas {initial_rows_prep - len(df_model_prep)} filas con NaN en T/E.", "WARN")
        if df_model_prep.empty: self.log("DF vacío post-NaN en T/E.", "ERROR"); messagebox.showerror("Datos Insuficientes", "No quedan datos post-NaN en T/E.", parent=self.parent_for_dialogs); return None

        # df_model_prep ahora tiene T/E con nombres finales, y las covariables originales seleccionadas.
        # build_design_matrix tomará esto, las covariables originales, y los nombres finales de T/E.
        df_filtered_patsy, X_design_patsy, formula_patsy_gen, terms_patsy_display = self.build_design_matrix(
            df_model_prep, selected_covs_orig_names, final_t_col, final_e_col
        )

        if X_design_patsy is None: self.log("Falló build_design_matrix.", "ERROR"); return None
        
        y_survival_patsy = df_filtered_patsy[[final_t_col, final_e_col]]

        self.log(f"Preparación de datos: DF ({df_filtered_patsy.shape}), X ({X_design_patsy.shape}), y ({y_survival_patsy.shape})", "INFO")
        self.log(f"Fórmula Patsy: {formula_patsy_gen}", "DEBUG")
        self.log(f"Términos Patsy: {terms_patsy_display}", "DEBUG")
        
        return (df_filtered_patsy, X_design_patsy, y_survival_patsy, formula_patsy_gen, terms_patsy_display, final_t_col, final_e_col)


    def _get_patsy_safe_var_name(self, var_name): # No se usa actualmente, Patsy Q() maneja nombres.
        if not isinstance(var_name, str): return str(var_name)
        safe_name = re.sub(r'[^a-zA-Z0-9_]', '_', var_name)
        if re.match(r'^\d', safe_name): safe_name = '_' + safe_name
        return safe_name

    def build_design_matrix(self, df_input_bd, selected_covs_orig_names_bd, time_col_name_bd, event_col_name_bd):
        if not PATSY_AVAILABLE or dmatrix is None:
            self.log("'patsy' no disponible.", "ERROR"); messagebox.showerror("Error Patsy", "'patsy' no instalada.", parent=self.parent_for_dialogs); return None, None, None, None

        df_for_patsy_bd = df_input_bd.copy() # df_input_bd ya tiene T/E con nombres finales y vars originales

        if not selected_covs_orig_names_bd: # Modelo nulo
            formula_patsy_bd = "0"
            try:
                if df_for_patsy_bd.empty: X_design_bd = pd.DataFrame(index=df_for_patsy_bd.index) # X vacía con índice
                else: X_design_bd = dmatrix(formula_patsy_bd, df_for_patsy_bd, return_type="dataframe")
                self.log("Matriz de diseño para modelo nulo (sin covariables).", "INFO")
                return df_for_patsy_bd, X_design_bd, formula_patsy_bd, []
            except Exception as e_patsy_null:
                self.log(f"Error Patsy (modelo nulo): {e_patsy_null}", "ERROR"); traceback.print_exc(limit=3); return None, None, None, None
        
        formula_parts_bd = []
        for orig_cov_name_bd in selected_covs_orig_names_bd:
            if orig_cov_name_bd not in df_for_patsy_bd.columns:
                self.log(f"Advertencia: Cov. original '{orig_cov_name_bd}' no en DF para Patsy. Saltando.", "WARN"); continue

            config_type_bd = self.covariables_type_config.get(orig_cov_name_bd, "Cuantitativa" if pd.api.types.is_numeric_dtype(df_for_patsy_bd[orig_cov_name_bd]) else "Cualitativa")
            
            term_syntax_bd = f"Q('{orig_cov_name_bd}')" # Default
            if config_type_bd == "Cuantitativa":
                if orig_cov_name_bd in self.spline_config_details:
                    spl_cfg_bd = self.spline_config_details[orig_cov_name_bd]
                    patsy_func_bd = 'cr' if spl_cfg_bd.get('type', 'Natural') == 'Natural' else 'bs'
                    term_syntax_bd = f"{patsy_func_bd}(Q('{orig_cov_name_bd}'), df={spl_cfg_bd.get('df', 4)})"
            else: # Cualitativa
                # Asegurar que la columna sea tratada como categórica por Patsy si no lo es ya
                if not pd.api.types.is_categorical_dtype(df_for_patsy_bd[orig_cov_name_bd].dtype) and \
                   not pd.api.types.is_string_dtype(df_for_patsy_bd[orig_cov_name_bd].dtype) and \
                   not pd.api.types.is_object_dtype(df_for_patsy_bd[orig_cov_name_bd].dtype): # Si es bool, int, etc.
                     df_for_patsy_bd[orig_cov_name_bd] = df_for_patsy_bd[orig_cov_name_bd].astype(str)

                ref_cat_bd = self.ref_categories_config.get(orig_cov_name_bd)
                if ref_cat_bd and str(ref_cat_bd).strip():
                    ref_cat_str_bd = str(ref_cat_bd)
                    # Comprobar si la ref_cat existe en la columna (convertida a str para comparación)
                    if ref_cat_str_bd in df_for_patsy_bd[orig_cov_name_bd].astype(str).unique():
                        term_syntax_bd = f"C(Q('{orig_cov_name_bd}'), Treatment(Q('{ref_cat_str_bd}')))"
                    else:
                        self.log(f"Advertencia: Ref.Cat. '{ref_cat_str_bd}' para '{orig_cov_name_bd}' no en datos. Usando default Patsy.", "WARN")
                        term_syntax_bd = f"C(Q('{orig_cov_name_bd}'))"
                else:
                    term_syntax_bd = f"C(Q('{orig_cov_name_bd}'))" # Patsy default ref
            formula_parts_bd.append(term_syntax_bd)

        formula_patsy_bd = "0 + " + " + ".join(formula_parts_bd) if formula_parts_bd else "0"
        self.log(f"Fórmula Patsy generada: {formula_patsy_bd}", "DEBUG")

        try:
            if df_for_patsy_bd.empty and formula_patsy_bd != "0":
                 self.log("DF entrada Patsy vacío con fórmula no nula.", "ERROR"); return None,None,None,None
            
            X_design_bd = dmatrix(formula_patsy_bd, df_for_patsy_bd, return_type="dataframe")
            
            # df_input_bd es el que tiene T/E y las vars originales.
            # X_design_bd tiene índice de las filas retenidas por Patsy (NaN drop).
            # Alinear df_input_bd con X_design_bd.
            df_filtered_by_patsy_idx_bd = df_input_bd.loc[X_design_bd.index].copy()
            
            final_terms_display_bd = list(X_design_bd.columns)
            self.log(f"Patsy: X_design ({X_design_bd.shape}), DF filtrado ({df_filtered_by_patsy_idx_bd.shape})", "INFO")
            self.log(f"Términos Patsy finales: {final_terms_display_bd}", "DEBUG")
            return df_filtered_by_patsy_idx_bd, X_design_bd, formula_patsy_bd, final_terms_display_bd
        except Exception as e_patsy_build:
            self.log(f"Error Patsy (build matriz): {e_patsy_build}", "ERROR"); traceback.print_exc(limit=messagebox.showerror("Error Patsy", f"Error construyendo matriz de diseño:\n{e_patsy_build}", parent=self.parent_for_dialogs)
            return None, None, None, None


    def _perform_variable_selection(self, df_aligned_orig_vs, X_design_initial_vs, time_col_vs, event_col_vs, formula_initial_vs, terms_initial_vs):
        method_vs = self.var_selection_method_var.get()
        
        if method_vs == "Ninguno (usar todas)":
            self.log("Selección Variables: 'Ninguno'. Usando todas.", "INFO")
            return df_aligned_orig_vs, X_design_initial_vs, formula_initial_vs, terms_initial_vs

        if X_design_initial_vs.empty:
            self.log("X_design inicial vacía. No se puede seleccionar variables.", "WARN")
            return df_aligned_orig_vs, X_design_initial_vs, "0", [] # Modelo nulo

        try:
            p_enter_vs = float(self.p_enter_var.get())
            p_remove_vs = float(self.p_remove_var.get())
            if not (0 <= p_enter_vs <= 1 and 0 <= p_remove_vs <= 1): raise ValueError("P fuera de [0,1]")
        except ValueError:
            self.log(f"P-valores inválidos para selección. Usando todas.", "ERROR"); messagebox.showerror("Error Paráms", "P-valores para selección deben ser numéricos entre 0 y 1.", parent=self.parent_for_dialogs)
            return df_aligned_orig_vs, X_design_initial_vs, formula_initial_vs, terms_initial_vs

        self.log(f"Selección Vars: Método='{method_vs}', P_Entrar={p_enter_vs}, P_Salir={p_remove_vs}", "INFO")

        try:
            # from lifelines.utils import forward_selection, backward_selection # Si fueran de utils
            # En versiones recientes, son métodos de CPHFitter
            
            df_for_selection_vs = X_design_initial_vs.copy()
            df_for_selection_vs[time_col_vs] = df_aligned_orig_vs[time_col_vs]
            df_for_selection_vs[event_col_vs] = df_aligned_orig_vs[event_col_vs]
            
            candidate_terms_vs = list(X_design_initial_vs.columns) # = terms_initial_vs
            if not candidate_terms_vs: self.log("No hay términos candidatos para selección.", "WARN"); return df_aligned_orig_vs, X_design_initial_vs, "0", []

            cph_selector = CoxPHFitter(penalizer=0.0) # Modelo limpio para selección
            selected_cph_after_selection = None

            if method_vs == "Forward":
                cph_selector.forward_select(df_for_selection_vs, duration_col=time_col_vs, event_col=event_col_vs, features_to_include=candidate_terms_vs, p_enter=p_enter_vs, scoring_method="log_likelihood")
                selected_cph_after_selection = cph_selector
            elif method_vs == "Backward":
                # Backward necesita un modelo ajustado con todas las vars primero
                if candidate_terms_vs : cph_selector.fit(df_for_selection_vs, duration_col=time_col_vs, event_col=event_col_vs, formula=" + ".join(candidate_terms_vs))
                else: cph_selector.fit(df_for_selection_vs, duration_col=time_col_vs, event_col=event_col_vs, formula="0") # Modelo nulo si no hay vars
                cph_selector.backward_select(df_for_selection_vs, duration_col=time_col_vs, event_col=event_col_vs, features_to_remove=candidate_terms_vs, p_remove=p_remove_vs, scoring_method="log_likelihood")
                selected_cph_after_selection = cph_selector
            elif method_vs == "Stepwise (Fwd luego Bwd)":
                cph_selector.stepwise_select(df_for_selection_vs, duration_col=time_col_vs, event_col=event_col_vs, features_to_include=candidate_terms_vs, p_enter=p_enter_vs, p_remove=p_remove_vs, scoring_method="log_likelihood")
                selected_cph_after_selection = cph_selector

            if selected_cph_after_selection is None or not hasattr(selected_cph_after_selection, 'params_') or selected_cph_after_selection.params_.empty:
                self.log(f"Selección '{method_vs}' no resultó en covariables. Modelo nulo.", "WARN")
                return df_aligned_orig_vs, pd.DataFrame(index=X_design_initial_vs.index), "0", []

            final_selected_terms_vs = [term for term in list(selected_cph_after_selection.params_.index) if term in X_design_initial_vs.columns] # Asegurar que son de X_initial
            if not final_selected_terms_vs: self.log(f"'{method_vs}' no retuvo covariables. Modelo nulo.", "WARN"); return df_aligned_orig_vs, pd.DataFrame(index=X_design_initial_vs.index), "0", []

            self.log(f"Selección completada. Términos Patsy seleccionados: {final_selected_terms_vs}", "INFO")
            X_design_selected_vs = X_design_initial_vs[final_selected_terms_vs].copy()
            formula_selected_vs = "0 + " + " + ".join(final_selected_terms_vs) if final_selected_terms_vs else "0"
            # df_aligned_orig_vs ya está alineado con X_design_initial_vs, y por tanto con X_design_selected_vs
            return df_aligned_orig_vs, X_design_selected_vs, formula_selected_vs, final_selected_terms_vs
        except ImportError:
            self.log("Error importando funciones de selección de Lifelines. ¿Versión antigua?", "ERROR"); messagebox.showerror("Error Lifelines", "Funciones de selección no encontradas. Verifique versión.", parent=self.parent_for_dialogs)
        except AttributeError as ae: # Ej. si forward_select no es método
            self.log(f"Error de atributo en selección (posiblemente versión Lifelines): {ae}", "ERROR"); messagebox.showerror("Error Lifelines", f"Error en selección (¿versión Lifelines incompatible?):\n{ae}", parent=self.parent_for_dialogs)
        except Exception as e_vs:
            self.log(f"Error en selección de variables: {e_vs}", "ERROR"); traceback.print_exc(limit=5)
            messagebox.showerror("Error Selección", f"Error en selección de variables:\n{e_vs}", parent=self.parent_for_dialogs)
        return df_aligned_orig_vs, X_design_initial_vs, formula_initial_vs, terms_initial_vs # Fallback a todas

    def _run_model_and_get_metrics(self, df_lifelines_rm, X_design_rm, y_survival_rm, time_col_rm, event_col_rm, formula_patsy_rm, model_name_rm, covariates_display_terms_rm, penalizer_val_rm=0.0, l1_ratio_val_rm=0.0):
        self.log(f"Ajustando modelo Cox: '{model_name_rm}'...", "INFO")
        model_data_rm = {
            "model_name": model_name_rm, "time_col_for_model": time_col_rm, "event_col_for_model": event_col_rm,
            "formula_patsy": formula_patsy_rm, "covariates_processed": covariates_display_terms_rm,
            "df_used_for_fit": df_lifelines_rm.copy(), "X_design_used_for_fit": X_design_rm.copy(), 
            "y_survival_used_for_fit": y_survival_rm.copy(),
            "penalizer_value": penalizer_val_rm, "l1_ratio_value": l1_ratio_val_rm,
            "tie_method_used": self.tie_handling_method_var.get(),
            "metrics": {}, "schoenfeld_results": None, "model": None, "loglik_null": None,
            "c_index_cv_mean": None, "c_index_cv_std": None
        }

        try:
            # Modelo Nulo (para LogLik nulo)
            cph_null_rm = CoxPHFitter(penalizer=0.0, tie_method=model_data_rm["tie_method_used"])
            # Usar y_survival_rm que está alineado con X_design_rm para el nulo
            df_for_null_fit_rm = pd.DataFrame({time_col_rm: y_survival_rm[time_col_rm], event_col_rm: y_survival_rm[event_col_rm]}, index=y_survival_rm.index)
            cph_null_rm.fit(df_for_null_fit_rm, duration_col=time_col_rm, event_col=event_col_rm, formula="0")
            model_data_rm["loglik_null"] = cph_null_rm.log_likelihood_

            # Modelo Principal
            cph_main_rm = CoxPHFitter(penalizer=penalizer_val_rm, l1_ratio=l1_ratio_val_rm, tie_method=model_data_rm["tie_method_used"])
            
            # X_design_rm y y_survival_rm deberían estar alineados.
            if X_design_rm.empty and not covariates_display_terms_rm: # Modelo nulo explícito
                cph_main_rm.fit(df_for_null_fit_rm, duration_col=time_col_rm, event_col=event_col_rm, formula="0")
            elif X_design_rm.empty and covariates_display_terms_rm: # Error
                self.log("Error: X_design vacía pero se esperaban covariables.", "ERROR"); return None
            else: # Modelo con covariables
                cph_main_rm.fit(X_design_rm, durations=y_survival_rm[time_col_rm], event_observed=y_survival_rm[event_col_rm])
            
            model_data_rm["model"] = cph_main_rm
            self.log(f"Modelo '{model_name_rm}' ajustado.", "SUCCESS")

            # Test de Schoenfeld
            if not X_design_rm.empty:
                try:
                    from lifelines.statistics import proportional_hazard_test
                    df_check_ph = X_design_rm.copy()
                    df_check_ph[time_col_rm] = y_survival_rm[time_col_rm]
                    df_check_ph[event_col_rm] = y_survival_rm[event_col_rm]
                    ph_test_obj = proportional_hazard_test(cph_main_rm, df_check_ph, time_transform='log')
                    model_data_rm["schoenfeld_results"] = ph_test_obj.summary
                    self.log("Test Schoenfeld completado.", "INFO")
                except Exception as e_sch: self.log(f"Error Test Schoenfeld: {e_sch}", "ERROR"); model_data_rm["schoenfeld_results"] = None
            else: self.log("Modelo nulo, no Test Schoenfeld.", "INFO")

            # C-Index CV
            if self.calculate_cv_cindex_var.get() and not X_design_rm.empty:
                try:
                    kf_cv = KFold(n_splits=self.cv_num_kfolds_var.get(), shuffle=True, random_state=self.cv_random_seed_var.get())
                    c_indices_cv_list = []
                    for train_idx, test_idx in kf_cv.split(X_design_rm):
                        X_tr, y_tr = X_design_rm.iloc[train_idx], y_survival_rm.iloc[train_idx]
                        X_te, y_te = X_design_rm.iloc[test_idx], y_survival_rm.iloc[test_idx]
                        if X_tr.empty or y_tr.empty or X_te.empty or y_te.empty: continue
                        cph_fold = CoxPHFitter(penalizer=penalizer_val_rm, l1_ratio=l1_ratio_val_rm, tie_method=model_data_rm["tie_method_used"])
                        cph_fold.fit(X_tr, durations=y_tr[time_col_rm], event_observed=y_tr[event_col_rm])
                        preds_te_fold = cph_fold.predict_partial_hazard(X_te)
                        c_idx_fold = concordance_index(y_te[time_col_rm], -preds_te_fold, y_te[event_col_rm])
                        c_indices_cv_list.append(c_idx_fold)
                    if c_indices_cv_list: model_data_rm["c_index_cv_mean"] = np.mean(c_indices_cv_list); model_data_rm["c_index_cv_std"] = np.std(c_indices_cv_list)
                    self.log(f"C-Index CV: Media={model_data_rm['c_index_cv_mean']:.3f} (DE={model_data_rm['c_index_cv_std']:.3f})", "INFO")
                except Exception as e_cv_rm: self.log(f"Error C-Index CV: {e_cv_rm}", "ERROR")
            elif self.calculate_cv_cindex_var.get(): self.log("C-Index CV no calculado (modelo nulo).", "INFO")

        except Exception as e_fit_main:
            self.log(f"Error ajuste modelo/métricas '{model_name_rm}': {e_fit_main}", "ERROR"); traceback.print_exc(limit=5); return None

        model_data_rm["metrics"] = compute_model_metrics(
            cph_main_rm, X_design_rm, y_survival_rm, time_col_rm, event_col_rm,
            model_data_rm["c_index_cv_mean"], model_data_rm["c_index_cv_std"],
            model_data_rm["schoenfeld_results"], model_data_rm["loglik_null"], self.log
        )
        return model_data_rm

    def _update_models_treeview(self):
        self.treeview_lista_modelos.delete(*self.treeview_lista_modelos.get_children())
        if not self.generated_models_data: self.log("No hay modelos para mostrar.", "INFO"); return

        for i, md_tv in enumerate(self.generated_models_data):
            name_tv = md_tv.get('model_name', f"Modelo {i+1}")
            covs_tv_terms = md_tv.get('covariates_processed', [])
            covs_str_tv = ", ".join(covs_tv_terms) if covs_tv_terms else "(Nulo)"
            
            metrics_tv = md_tv.get('metrics', {})
            c_idx_tr_tv = metrics_tv.get('C-Index (Training)')
            aic_tv = metrics_tv.get('AIC')
            loglik_tv = metrics_tv.get('Log-Likelihood')
            sch_p_glob_tv = metrics_tv.get('Schoenfeld p-value (global)')

            vals_tv = (
                i + 1, name_tv, covs_str_tv,
                f"{c_idx_tr_tv:.3f}" if pd.notna(c_idx_tr_tv) else "N/A",
                f"{aic_tv:.2f}" if pd.notna(aic_tv) else "N/A",
                f"{loglik_tv:.2f}" if pd.notna(loglik_tv) else "N/A",
                format_p_value(sch_p_glob_tv) if pd.notna(sch_p_glob_tv) else "N/A"
            )
            self.treeview_lista_modelos.insert("", tk.END, iid=str(i), values=vals_tv)
        self.log(f"Treeview actualizada con {len(self.generated_models_data)} modelos.", "INFO")


    def _execute_cox_modeling_orchestrator(self):
        self.log("*"*35 + " INICIO MODELADO COX " + "*"*35, "HEADER")
        temp_models_list_orch = [] # Lista para esta ejecución

        prep_res = self._preparar_datos_para_modelado()
        if prep_res is None:
            self.log("Falló preparación de datos. Abortando.", "ERROR"); self.log("*"*35 + " FIN MODELADO (ERRORES) " + "*"*35, "HEADER")
            self.generated_models_data = temp_models_list_orch; self._update_models_treeview(); return
        
        (df_init_full, X_init_full, y_init_data, formula_init, terms_init_display, t_col_final, e_col_final) = prep_res

        if df_init_full is None or df_init_full.empty:
            self.log("DF inicial vacío post-preparación. Abortando.", "ERROR"); self.log("*"*35 + " FIN MODELADO (ERRORES) " + "*"*35, "HEADER")
            self.generated_models_data = temp_models_list_orch; self._update_models_treeview(); return

        pen_meth = self.penalization_method_var.get(); pen_val = 0.0; l1_r = 0.0
        if pen_meth != "Ninguna":
            try: pen_val = float(self.penalizer_strength_var.get()); assert pen_val >= 0
            except: self.log("Valor penalización inválido. Usando 0.", "ERROR"); pen_val = 0.0
            if pen_val > 0:
                if pen_meth == "L1 (Lasso)": l1_r = 1.0
                elif pen_meth == "L2 (Ridge)": l1_r = 0.0
                elif pen_meth == "ElasticNet":
                    try: l1_r = float(self.l1_ratio_for_elasticnet_var.get()); assert 0 <= l1_r <= 1
                    except: self.log("Ratio L1 inválido. Usando 0.5.", "ERROR"); l1_r = 0.5
                self.log(f"Penalización: Tipo='{pen_meth}', Valor={pen_val:.4g}, L1_Ratio={l1_r:.2f}", "CONFIG")
            else: self.log(f"Penalización '{pen_meth}' con valor <=0. Sin penalización efectiva.", "INFO"); pen_val = 0.0
        else: self.log("Sin penalización.", "CONFIG")

        model_type_ui = self.cox_model_type_var.get()
        if model_type_ui == "Univariado":
            self.log("Iniciando modelado Univariado...", "INFO")
            orig_covs_ui = [self.listbox_covariables_disponibles.get(i) for i in self.listbox_covariables_disponibles.curselection() if self.listbox_covariables_disponibles.get(i) not in [self.combo_col_tiempo.get(), self.combo_col_evento.get()]]
            if not orig_covs_ui: self.log("No hay covariables originales para univariado.", "WARN")
            else:
                for orig_cov_uni in orig_covs_ui:
                    self.log(f"--- Univariado para: {orig_cov_uni} ---", "SUBHEADER")
                    df_uni_f, X_uni_d, formula_uni, terms_uni = self.build_design_matrix(df_init_full, [orig_cov_uni], t_col_final, e_col_final)
                    if X_uni_d is None or df_uni_f is None or df_uni_f.empty: self.log(f"Fallo build_design_matrix para '{orig_cov_uni}'.", "WARN"); continue
                    y_uni_s = df_uni_f[[t_col_final, e_col_final]]
                    if X_uni_d.empty and not terms_uni: self.log(f"X_design vacía para '{orig_cov_uni}'.", "WARN"); continue
                    name_uni = f"Univariado: {orig_cov_uni}" + (f" (Términos: {', '.join(terms_uni)})" if terms_uni != [orig_cov_uni] else "")
                    md_uni = self._run_model_and_get_metrics(df_uni_f, X_uni_d, y_uni_s, t_col_final, e_col_final, formula_uni, name_uni, terms_uni, pen_val, l1_r)
                    if md_uni: temp_models_list_orch.append(md_uni)
        
        elif model_type_ui == "Multivariado":
            self.log("Iniciando modelado Multivariado...", "INFO")
            df_multi, X_multi, formula_multi, terms_multi = df_init_full, X_init_full, formula_init, terms_init_display
            y_multi = y_init_data
            sel_meth_ui = self.var_selection_method_var.get(); suffix_multi = " (Todas las Variables)"
            if sel_meth_ui != "Ninguno (usar todas)":
                if X_init_full is None or X_init_full.empty: self.log("X_design inicial vacío. No se puede seleccionar variables.", "WARN")
                else:
                    self.log(f"Selección de variables: {sel_meth_ui}", "INFO")
                    df_multi, X_multi, formula_multi, terms_multi = self._perform_variable_selection(df_init_full, X_init_full, t_col_final, e_col_final, formula_init, terms_init_display)
                    y_multi = df_multi[[t_col_final, e_col_final]] # Re-alinear y
                    suffix_multi = f" ({sel_meth_ui})"
            if X_multi is None or df_multi is None or df_multi.empty: self.log("DF o X_design multivariado vacío. Abortando.", "ERROR")
            else:
                if X_multi.empty and not terms_multi: suffix_multi += " (Nulo)"
                name_multi = f"Multivariado{suffix_multi}"
                md_multi = self._run_model_and_get_metrics(df_multi, X_multi, y_multi, t_col_final, e_col_final, formula_multi, name_multi, terms_multi, pen_val, l1_r)
                if md_multi: temp_models_list_orch.append(md_multi)

        self.generated_models_data = temp_models_list_orch
        self._update_models_treeview()
        msg_fin = f"Modelado completado. {len(self.generated_models_data)} modelo(s) generado(s)." if self.generated_models_data else "No se generó ningún modelo."
        self.log(msg_fin, "SUCCESS" if self.generated_models_data else "WARN")
        messagebox.showinfo("Modelado Terminado", msg_fin, parent=self.parent_for_dialogs)
        self.log("*"*35 + " FIN PROCESO DE MODELADO COX " + "*"*35, "HEADER")


    def _on_model_select_from_treeview(self, event=None):
        sel_item = self.treeview_lista_modelos.focus()
        if sel_item:
            try:
                idx = int(sel_item)
                if 0 <= idx < len(self.generated_models_data):
                    self.selected_model_in_treeview = self.generated_models_data[idx]
                    self.log(f"Modelo '{self.selected_model_in_treeview.get('model_name')}' seleccionado.", "INFO")
                else: self.selected_model_in_treeview = None; self.log("Índice modelo fuera de rango.", "WARN")
            except ValueError: self.selected_model_in_treeview = None; self.log("Error obteniendo índice modelo.", "WARN")
        else: self.selected_model_in_treeview = None; self.log("Ningún modelo seleccionado.", "INFO")
        self._update_results_buttons_state()

    def show_selected_model_summary(self):
        if not self._check_model_selected_and_valid(): return
        model_dict_sum = self.selected_model_in_treeview
        text_sum = self._generate_text_summary_for_model(model_dict_sum) # Usar helper
        ModelSummaryWindow(self.parent_for_dialogs, f"Resumen: {model_dict_sum.get('model_name', 'N/A')}", text_sum)
        self.log(f"Mostrando resumen para '{model_dict_sum.get('model_name', 'N/A')}'.", "INFO")

    def _create_plot_window(self, fig, title="Gráfico", is_single_plot=True):
        plot_win = Toplevel(self.parent_for_dialogs); plot_win.title(title)
        fig_w_px, fig_h_px = fig.get_figwidth() * fig.dpi, fig.get_figheight() * fig.dpi
        win_w = int(fig_w_px + 60); win_h = int(fig_h_px + (120 if is_single_plot else 80))
        max_w, max_h = int(plot_win.winfo_screenwidth()*0.85), int(plot_win.winfo_screenheight()*0.8)
        plot_win.geometry(f"{min(win_w,max_w)}x{min(win_h,max_h)}")

        frame_main_plot = ttk.Frame(plot_win); frame_main_plot.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        canvas_plot = FigureCanvasTkAgg(fig, master=frame_main_plot)
        canvas_plot.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        toolbar_plot_frame = ttk.Frame(frame_main_plot)
        toolbar_plot_frame.pack(fill=tk.X, pady=(5,0))
        NavigationToolbar2Tk(canvas_plot, toolbar_plot_frame).update()

        btns_plot_bottom = ttk.Frame(plot_win); btns_plot_bottom.pack(fill=tk.X, pady=5, padx=5)
        if is_single_plot: ttk.Button(btns_plot_bottom, text="Opciones...", command=lambda f=fig: self._open_plot_options_for_figure(f)).pack(side=tk.LEFT, padx=5)
        ttk.Button(btns_plot_bottom, text="Cerrar", command=lambda w=plot_win, f_close=fig: self._on_plot_window_close(w, f_close)).pack(side=tk.RIGHT, padx=5)
        plot_win.protocol("WM_DELETE_WINDOW", lambda w=plot_win, f_close=fig: self._on_plot_window_close(w, f_close))

    def _on_plot_window_close(self, window_ref, fig_ref):
        plt.close(fig_ref) 
        window_ref.destroy()

    def _open_plot_options_for_figure(self, fig_opt):
        if fig_opt is None or not fig_opt.get_axes(): self.log("Figura no válida para opciones.", "WARN"); return
        self._active_figure_for_options = fig_opt # Guardar referencia
        PlotOptionsDialog(self.parent_for_dialogs, self.current_plot_options.copy(), self._apply_options_to_active_figure)
    
    def _apply_options_to_active_figure(self, new_opts_plot):
        if hasattr(self, '_active_figure_for_options') and self._active_figure_for_options:
            fig_to_reconfig = self._active_figure_for_options
            if fig_to_reconfig.get_axes():
                for ax_reconfig in fig_to_reconfig.get_axes():
                    try: apply_plot_options(ax_reconfig, new_opts_plot, self.log)
                    except Exception as e_apply_reconfig: self.log(f"Error aplicando opciones a eje: {e_apply_reconfig}", "ERROR")
                if hasattr(fig_to_reconfig, 'canvas') and fig_to_reconfig.canvas: fig_to_reconfig.canvas.draw_idle()
                self.current_plot_options = new_opts_plot.copy() # Actualizar globales
                self.log("Opciones de gráfico aplicadas a figura activa.", "INFO")
            else: self.log("Figura activa sin ejes para opciones.", "WARN")
            del self._active_figure_for_options # Limpiar referencia


    def show_schoenfeld(self):
        if not self._check_model_selected_and_valid(check_params=True): return
        md_sch = self.selected_model_in_treeview
        cph_sch = md_sch.get('model'); name_sch = md_sch.get('model_name', 'N/A')
        try:
            self.log(f"Generando Gráf. Schoenfeld para '{name_sch}'...", "INFO")
            df_check_sch = md_sch.get('X_design_used_for_fit').copy()
            df_check_sch[md_sch.get('time_col_for_model')] = md_sch.get('y_survival_used_for_fit')[md_sch.get('time_col_for_model')]
            df_check_sch[md_sch.get('event_col_for_model')] = md_sch.get('y_survival_used_for_fit')[md_sch.get('event_col_for_model')]

            num_params_sch = len(cph_sch.params_)
            ncols_s = min(2, num_params_sch); nrows_s = math.ceil(num_params_sch / ncols_s)
            fig_s, axes_s_flat = plt.subplots(nrows_s, ncols_s, figsize=(12 if ncols_s > 1 else 7, 4*nrows_s), sharex=False, squeeze=False) # squeeze=False
            axes_s_flat = axes_s_flat.flatten()

            for idx, cov_name_s in enumerate(list(cph_sch.params_.index)):
                if idx < len(axes_s_flat):
                    ax_s_curr = axes_s_flat[idx]
                    try:
                        cph_sch.plot_residuals(training_df=df_check_sch, kind="schoenfeld", columns=[cov_name_s], ax=ax_s_curr)
                        ax_s_curr.set_title(f"Schoenfeld: {cov_name_s}") # Sobrescribir título por defecto si es necesario
                        apply_plot_options(ax_s_curr, self.current_plot_options, self.log) # Aplicar opciones globales
                    except Exception as e_plot_s_cov: self.log(f"Error plot Schoenfeld para {cov_name_s}: {e_plot_s_cov}", "ERROR")
            for i_empty_s in range(num_params_sch, len(axes_s_flat)): axes_s_flat[i_empty_s].set_visible(False) # Ocultar no usados
            fig_s.suptitle(f"Residuos de Schoenfeld ({name_sch})", fontsize=14); plt.tight_layout(rect=[0,0,1,0.96])
            self._create_plot_window(fig_s, f"Schoenfeld: {name_sch}", is_single_plot=False)
        except Exception as e_s_main: self.log(f"Error Gráf. Schoenfeld '{name_sch}': {e_s_main}", "ERROR"); traceback.print_exc(limit=3); messagebox.showerror("Error Gráfico", f"Error Schoenfeld:\n{e_s_main}", parent=self.parent_for_dialogs)

    def show_baseline_survival(self):
        if not self._check_model_selected_and_valid(): return
        md_bs = self.selected_model_in_treeview; cph_bs = md_bs.get('model'); name_bs = md_bs.get('model_name', 'N/A')
        try:
            fig_bs, ax_bs = plt.subplots(figsize=(10,6)); cph_bs.baseline_survival_.plot(ax=ax_bs, legend=False)
            opts_bs = self.current_plot_options.copy()
            opts_bs['title'] = opts_bs.get('title') or f"Supervivencia Base S0(t) ({name_bs})"
            opts_bs['xlabel'] = opts_bs.get('xlabel') or f"Tiempo ({md_bs.get('time_col_for_model','T')})"
            opts_bs['ylabel'] = opts_bs.get('ylabel') or "S0(t)"
            apply_plot_options(ax_bs, opts_bs, self.log)
            self._create_plot_window(fig_bs, f"Sup. Base: {name_bs}")
        except Exception as e_bs: self.log(f"Error Sup.Base '{name_bs}': {e_bs}", "ERROR"); messagebox.showerror("Error Gráfico", f"Error Sup.Base:\n{e_bs}", parent=self.parent_for_dialogs)

    def show_baseline_hazard(self):
        if not self._check_model_selected_and_valid(): return
        md_bh = self.selected_model_in_treeview; cph_bh = md_bh.get('model'); name_bh = md_bh.get('model_name', 'N/A')
        try:
            fig_bh, ax_bh = plt.subplots(figsize=(10,6)); cph_bh.baseline_hazard_.plot(ax=ax_bh, legend=False)
            opts_bh = self.current_plot_options.copy()
            opts_bh['title'] = opts_bh.get('title') or f"Riesgo Acumulado Base H0(t) ({name_bh})"
            opts_bh['xlabel'] = opts_bh.get('xlabel') or f"Tiempo ({md_bh.get('time_col_for_model','T')})"
            opts_bh['ylabel'] = opts_bh.get('ylabel') or "H0(t)"
            apply_plot_options(ax_bh, opts_bh, self.log)
            self._create_plot_window(fig_bh, f"Riesgo Acum. Base: {name_bh}")
        except Exception as e_bh: self.log(f"Error Riesgo Acum.Base '{name_bh}': {e_bh}", "ERROR"); messagebox.showerror("Error Gráfico", f"Error Riesgo Acum.Base:\n{e_bh}", parent=self.parent_for_dialogs)

    def generar_forest_plot(self):
        if not self._check_model_selected_and_valid(check_params=True): return
        md_fp = self.selected_model_in_treeview; name_fp = md_fp.get('model_name', 'N/A')
        sum_df_fp = md_fp.get('metrics',{}).get('summary_df')
        if sum_df_fp is None or sum_df_fp.empty or 'exp(coef)' not in sum_df_fp.columns:
            self.log("No hay datos para Forest Plot.", "INFO"); messagebox.showinfo("Forest Plot","No hay HRs para mostrar.",parent=self.parent_for_dialogs); return
        try:
            plot_df_fp = sum_df_fp.copy()
            sort_fp = self.current_plot_options.get('sort_order', 'original')
            if sort_fp == "hr_asc": plot_df_fp.sort_values('exp(coef)', inplace=True)
            elif sort_fp == "hr_desc": plot_df_fp.sort_values('exp(coef)', ascending=False, inplace=True)
            # ... otros ordenamientos ...
            
            fig_fp, ax_fp = plt.subplots(figsize=(10, max(4, len(plot_df_fp)*0.5)))
            y_pos_fp = np.arange(len(plot_df_fp))
            hrs_fp, low_ci_fp, upp_ci_fp = plot_df_fp['exp(coef)'], plot_df_fp['exp(coef) lower 95%'], plot_df_fp['exp(coef) upper 95%']
            ax_fp.errorbar(hrs_fp, y_pos_fp, xerr=[hrs_fp-low_ci_fp, upp_ci_fp-hrs_fp], fmt='o', capsize=5, color='k', ms=5, elw=1.2)
            ax_fp.set_yticks(y_pos_fp); ax_fp.set_yticklabels(plot_df_fp.index); ax_fp.invert_yaxis()
            ax_fp.axvline(1.0, color='gray', ls='--', lw=0.8)
            
            opts_fp = self.current_plot_options.copy(); opts_fp['title'] = opts_fp.get('title') or f"Forest Plot HRs ({name_fp})"; opts_fp['xlabel'] = opts_fp.get('xlabel') or "Hazard Ratio (HR) con IC 95%"; apply_plot_options(ax_fp, opts_fp, self.log); plt.tight_layout()
            self._create_plot_window(fig_fp, f"Forest Plot: {name_fp}")
        except Exception as e_fp: self.log(f"Error Forest Plot '{name_fp}': {e_fp}", "ERROR"); traceback.print_exc(limit=3); messagebox.showerror("Error Gráfico", f"Error Forest Plot:\n{e_fp}", parent=self.parent_for_dialogs)

    def realizar_prediccion(self):
        if not self._check_model_selected_and_valid(check_params=True): return # check_params=True si modelo nulo no tiene sentido
        md_pred = self.selected_model_in_treeview; cph_pred = md_pred.get('model'); name_pred = md_pred.get('model_name', 'N/A')
        
        orig_vars_ask_pred = []
        if hasattr(cph_pred, 'design_info_') and cph_pred.design_info_:
            orig_vars_ask_pred = [fn for fn in cph_pred.design_info_.factor_infos.keys() if fn != 'Intercept']
        if not orig_vars_ask_pred: self.log("No se pudieron obtener vars originales para predicción.", "WARN"); messagebox.showwarning("Predicción","No se pudieron determinar vars de entrada.",parent=self.parent_for_dialogs); return

        pred_diag = Toplevel(self.parent_for_dialogs); pred_diag.title(f"Predicción: {name_pred}"); pred_diag.transient(self.parent_for_dialogs)
        entries_pred = {}; frame_main_pred_diag = ttk.Frame(pred_diag, padding=10); frame_main_pred_diag.pack(fill=tk.BOTH, expand=True)
        ttk.Label(frame_main_pred_diag, text="Valores para covariables originales:", font=("TkDefaultFont",10,"bold")).pack(pady=(0,10),anchor='w')
        frame_vars_pred = ttk.Frame(frame_main_pred_diag); frame_vars_pred.pack(fill=tk.X, pady=5)
        for i, var_n in enumerate(orig_vars_ask_pred):
            ttk.Label(frame_vars_pred, text=f"{var_n}:").grid(row=i,column=0,padx=5,pady=3,sticky=tk.E)
            svar_pred = StringVar(); entries_pred[var_n] = svar_pred
            # Default con media/moda si se puede
            if self.data is not None and var_n in self.data:
                try: svar_pred.set(f"{self.data[var_n].mean():.2f}" if pd.api.types.is_numeric_dtype(self.data[var_n]) else str(self.data[var_n].mode()[0]))
                except: pass
            ttk.Entry(frame_vars_pred, textvariable=svar_pred, width=25).grid(row=i,column=1,padx=5,pady=3,sticky=tk.EW)
        frame_vars_pred.columnconfigure(1,weight=1)
        
        frame_opts_pred = ttk.Frame(frame_main_pred_diag); frame_opts_pred.pack(fill=tk.X,pady=10)
        ttk.Label(frame_opts_pred,text="Tipo Predicción:").grid(row=0,column=0,padx=5,pady=3,sticky=tk.W)
        type_var_pred_ui = StringVar(value="Supervivencia")
        ttk.Radiobutton(frame_opts_pred,text="Prob.Supervivencia",variable=type_var_pred_ui,value="Supervivencia").grid(row=0,column=1,padx=5,pady=3,sticky=tk.W)
        ttk.Radiobutton(frame_opts_pred,text="Riesgo Acumulado",variable=type_var_pred_ui,value="Riesgo").grid(row=0,column=2,padx=5,pady=3,sticky=tk.W)
        ttk.Label(frame_opts_pred,text="Tiempo(s) (ej: 100 o 50,100):").grid(row=1,column=0,padx=5,pady=3,sticky=tk.W)
        times_str_var_pred_ui = StringVar(value="100")
        ttk.Entry(frame_opts_pred,textvariable=times_str_var_pred_ui,width=30).grid(row=1,column=1,columnspan=2,padx=5,pady=3,sticky=tk.EW)
        
        frame_btns_pred_diag = ttk.Frame(frame_main_pred_diag,padding=(0,10,0,0)); frame_btns_pred_diag.pack(fill=tk.X)
        ttk.Button(frame_btns_pred_diag,text="Predecir y Mostrar Curva",command=lambda: self._perform_prediction_and_plot(pred_diag,md_pred,entries_pred,type_var_pred_ui.get(),times_str_var_pred_ui.get())).pack(side=tk.LEFT,padx=10)
        ttk.Button(frame_btns_pred_diag,text="Cancelar",command=pred_diag.destroy).pack(side=tk.RIGHT,padx=10)

    def _perform_prediction_and_plot(self, dialog_pred_ref, md_dict_for_pred, entries_dict_for_pred, type_ui_pred, times_str_ui_pred):
        cph_model_for_pred = md_dict_for_pred.get('model'); name_for_pred = md_dict_for_pred.get('model_name', 'N/A')
        try:
            times_list_pred = [float(t.strip()) for t in times_str_ui_pred.split(',') if t.strip()]
            if not times_list_pred or any(t <= 0 for t in times_list_pred): raise ValueError("Tiempos inválidos")
            times_list_pred = sorted(list(set(times_list_pred)))
        except ValueError: messagebox.showerror("Error Tiempos","Tiempos inválidos.",parent=dialog_pred_ref); return
        
        input_data_patsy_pred = {}
        for var_k, svar_obj in entries_dict_for_pred.items():
            val_entry = svar_obj.get().strip()
            if not val_entry: messagebox.showerror("Valor Faltante",f"Valor faltante para '{var_k}'.",parent=dialog_pred_ref); return
            # Simplificado: intentar float, luego string. Para categóricas, Patsy necesita el string.
            try: input_data_patsy_pred[var_k] = float(val_entry)
            except ValueError: input_data_patsy_pred[var_k] = str(val_entry)
        df_patsy_input_pred = pd.DataFrame([input_data_patsy_pred])

        try:
            design_info_pred = getattr(cph_model_for_pred, 'design_info_', None)
            if design_info_pred: X_patsy_pred_final = dmatrix(design_info_pred, df_patsy_input_pred, return_type="dataframe")
            else: self.log("No design_info_ en modelo para predicción. Usando fórmula (menos robusto).", "WARN"); X_patsy_pred_final = dmatrix(md_dict_for_pred.get('formula_patsy','0'), df_patsy_input_pred, return_type="dataframe")
        except Exception as e_patsy_pred_final: self.log(f"Error Patsy en predicción: {e_patsy_pred_final}","ERROR"); messagebox.showerror("Error Patsy Pred.","Error transformando entradas.",parent=dialog_pred_ref); return

        try:
            fig_curve_pred, ax_curve_pred = plt.subplots(figsize=(10,6)); results_text_pred = []
            if type_ui_pred == "Supervivencia":
                surv_df_pred = cph_model_for_pred.predict_survival_function(X_patsy_pred_final); surv_df_pred.plot(ax=ax_curve_pred, legend=False)
                ax_curve_pred.set_ylabel("S(t|X)"); title_curve_pred = f"Pred. Supervivencia ({name_for_pred})"
                for t_val in times_list_pred: val_plot = surv_df_pred.interpolate(method='index').loc[t_val].iloc[0] if t_val not in surv_df_pred.index else surv_df_pred.loc[t_val].iloc[0]; results_text_pred.append(f"S(t={t_val}|X) = {val_plot:.3f}"); ax_curve_pred.scatter([t_val],[val_plot],marker='o',color='r',s=50,zorder=5,label=f't={t_val}' if t_val==times_list_pred[0] else None)
            else: # Riesgo
                cumhaz_df_pred = cph_model_for_pred.predict_cumulative_hazard(X_patsy_pred_final); cumhaz_df_pred.plot(ax=ax_curve_pred, legend=False)
                ax_curve_pred.set_ylabel("H(t|X)"); title_curve_pred = f"Pred. Riesgo Acum. ({name_for_pred})"
                for t_val in times_list_pred: val_plot = cumhaz_df_pred.interpolate(method='index').loc[t_val].iloc[0] if t_val not in cumhaz_df_pred.index else cumhaz_df_pred.loc[t_val].iloc[0]; results_text_pred.append(f"H(t={t_val}|X) = {val_plot:.3f}"); ax_curve_pred.scatter([t_val],[val_plot],marker='o',color='r',s=50,zorder=5,label=f't={t_val}' if t_val==times_list_pred[0] else None)
            
            opts_curve_pred = self.current_plot_options.copy(); opts_curve_pred['title'] = opts_curve_pred.get('title') or title_curve_pred; opts_curve_pred['xlabel'] = opts_curve_pred.get('xlabel') or f"Tiempo ({md_dict_for_pred.get('time_col_for_model','T')})"; apply_plot_options(ax_curve_pred, opts_curve_pred, self.log)
            if results_text_pred: ax_curve_pred.legend()
            self._create_plot_window(fig_curve_pred, title_curve_pred)
            messagebox.showinfo("Resultados Predicción", "Resultados en tiempos especificados:\n" + "\n".join(results_text_pred), parent=dialog_pred_ref)
        except Exception as e_curve_pred: self.log(f"Error pred/plot: {e_curve_pred}","ERROR"); messagebox.showerror("Error Pred/Plot",f"Error al predecir/plotear:\n{e_curve_pred}",parent=dialog_pred_ref)


    def generate_calibration_plot(self):
        if not self._check_model_selected_and_valid(): return
        if not LIFELINES_CALIBRATION_AVAILABLE: messagebox.showwarning("No Disponible","Gráf. Calibración no disponible.",parent=self.parent_for_dialogs); return
        md_cal = self.selected_model_in_treeview; cph_cal = md_cal.get('model'); name_cal = md_cal.get('model_name','N/A')
        X_dsgn_cal = md_cal.get('X_design_used_for_fit'); y_surv_cal = md_cal.get('y_survival_used_for_fit')
        t_col_cal, e_col_cal = md_cal.get('time_col_for_model'), md_cal.get('event_col_for_model')
        if X_dsgn_cal is None or y_surv_cal is None: messagebox.showerror("Error Datos","Faltan datos para calibración.",parent=self.parent_for_dialogs); return
        
        t0_cal_str = simpledialog.askstring("Tiempo Calibración","Ingrese t0 para calibración:",parent=self.parent_for_dialogs)
        if not t0_cal_str: return
        try: t0_val_cal = float(t0_cal_str); assert t0_val_cal > 0
        except: messagebox.showerror("Tiempo Inválido",f"Tiempo t0 inválido.",parent=self.parent_for_dialogs); return
        try:
            fig_cal, ax_cal = plt.subplots(figsize=(8,8))
            survival_probability_calibration_plot(cph_cal, X_dsgn_cal, T=y_surv_cal[t_col_cal], E=y_surv_cal[e_col_cal], t0=t0_val_cal, ax=ax_cal)
            opts_cal = self.current_plot_options.copy(); opts_cal['title'] = opts_cal.get('title') or f"Calibración en t0={t0_val_cal} ({name_cal})"; apply_plot_options(ax_cal, opts_cal, self.log)
            self._create_plot_window(fig_cal, f"Calibración t0={t0_val_cal}: {name_cal}")
        except Exception as e_cal: self.log(f"Error gráf.calibración '{name_cal}': {e_cal}","ERROR"); traceback.print_exc(limit=3); messagebox.showerror("Error Gráfico",f"Error gráf.calibración:\n{e_cal}",parent=self.parent_for_dialogs)

    def export_model_summary(self):
        if not self._check_model_selected_and_valid(): return
        md_exp = self.selected_model_in_treeview; name_exp = md_exp.get('model_name','Modelo_Exportado')
        summary_txt_exp = self._generate_text_summary_for_model(md_exp)
        if not summary_txt_exp: self.log("No se pudo generar resumen para exportar.", "ERROR"); return
        fpath_exp = filedialog.asksaveasfilename(title="Guardar Resumen Como...",defaultextension=".txt",initialfile=f"Resumen_{name_exp.replace(' ','_').replace(':','')}.txt",filetypes=[("Texto","*.txt"),("Todos","*.*")])
        if not fpath_exp: self.log("Exportación cancelada.", "INFO"); return
        try:
            with open(fpath_exp, "w", encoding="utf-8") as f_exp: f_exp.write(summary_txt_exp)
            self.log(f"Resumen '{name_exp}' exportado a: {fpath_exp}", "SUCCESS"); messagebox.showinfo("Exportación Exitosa",f"Resumen guardado en:\n{fpath_exp}",parent=self.parent_for_dialogs)
        except Exception as e_exp: self.log(f"Error exportando resumen: {e_exp}","ERROR"); messagebox.showerror("Error Exportación",f"No se pudo guardar:\n{e_exp}",parent=self.parent_for_dialogs)

    def _generate_text_summary_for_model(self, model_dict_gst):
        name_gst = model_dict_gst.get('model_name', 'N/A'); s_txt_gst = f"--- Resumen Modelo: {name_gst} ---\n"; s_txt_gst += f"Generado: {pd.Timestamp.now():%Y-%m-%d %H:%M:%S}\n\n"
        s_txt_gst += "Configuración Ajuste:\n"; s_txt_gst += f"  Tiempo: {model_dict_gst.get('time_col_for_model','N/A')}\n  Evento: {model_dict_gst.get('event_col_for_model','N/A')}\n"
        s_txt_gst += f"  Fórmula Patsy: {model_dict_gst.get('formula_patsy','N/A')}\n  Términos Modelo: {', '.join(model_dict_gst.get('covariates_processed',[]))}\n"
        s_txt_gst += f"  Penalización: {model_dict_gst.get('penalizer_value',0.0):.4g} (L1 Ratio: {model_dict_gst.get('l1_ratio_value',0.0):.2f})\n  Manejo Empates: {model_dict_gst.get('tie_method_used','N/A')}\n\n"
        
        s_txt_gst += "Coeficientes (Resumen Lifelines):\n"
        sum_df_gst = model_dict_gst.get('metrics',{}).get('summary_df')
        s_txt_gst += (sum_df_gst.to_string() + "\n\n") if sum_df_gst is not None and not sum_df_gst.empty else "  (No disponibles)\n\n"
        
        s_txt_gst += "Métricas Evaluación:\n"
        metrics_gst = model_dict_gst.get('metrics',{})
        for k,v in metrics_gst.items():
            if k in ["summary_df","schoenfeld_details","HR (individual)","HR_CI (individual)","Wald p-values (individual)"]: continue
            if isinstance(v,pd.DataFrame): continue
            s_txt_gst += f"  {k}: {f'{v:.4f}' if isinstance(v,(float,np.floating)) else (str(v)[:200] if pd.notna(v) else 'N/A')}\n"
        s_txt_gst += "\nTest Schoenfeld (Riesgos Proporcionales):\n"
        sch_df_gst = metrics_gst.get("schoenfeld_results"); sch_p_g_gst = metrics_gst.get('Schoenfeld p-value (global)')
        if sch_df_gst is not None and not sch_df_gst.empty: s_txt_gst += f"  P-Global: {format_p_value(sch_p_g_gst)}\n{sch_df_gst.to_string()}\n"
        elif pd.notna(sch_p_g_gst): s_txt_gst += f"  P-Global: {format_p_value(sch_p_g_gst)}\n"
        else: s_txt_gst += "  (No calculado/disponible o no aplica)\n"
        s_txt_gst += "\n--- Fin Resumen ---\n"; return s_txt_gst

    def save_model(self):
        if not self._check_model_selected_and_valid(): return
        md_save = self.selected_model_in_treeview; name_save = md_save.get('model_name','Modelo_Guardado')
        fpath_save = filedialog.asksaveasfilename(title="Guardar Modelo Como...",defaultextension=".pkl",initialfile=f"{name_save.replace(' ','_').replace(':','')}.pkl",filetypes=[("Pickle","*.pkl"),("Todos","*.*")])
        if not fpath_save: self.log("Guardado cancelado.", "INFO"); return
        try:
            with open(fpath_save, "wb") as f_save: pickle.dump(md_save, f_save)
            self.log(f"Modelo '{name_save}' guardado en: {fpath_save}", "SUCCESS"); messagebox.showinfo("Modelo Guardado",f"Modelo guardado en:\n{fpath_save}",parent=self.parent_for_dialogs)
        except Exception as e_save: self.log(f"Error guardando modelo: {e_save}","ERROR"); messagebox.showerror("Error Guardando",f"No se pudo guardar:\n{e_save}",parent=self.parent_for_dialogs)

    def load_model_from_file(self):
        fpath_load = filedialog.askopenfilename(title="Cargar Modelo Pickle",filetypes=[("Pickle","*.pkl"),("Todos","*.*")])
        if not fpath_load: self.log("Carga cancelada.", "INFO"); return
        try:
            with open(fpath_load, "rb") as f_load: loaded_md = pickle.load(f_load)
            if not (isinstance(loaded_md,dict) and 'model' in loaded_md and 'model_name' in loaded_md and isinstance(loaded_md.get('model'),CoxPHFitter)):
                raise ValueError("Archivo no contiene un modelo CoxPHFitter válido en el formato esperado.")
            self.generated_models_data.append(loaded_md); self._update_models_treeview()
            new_idx_load = len(self.generated_models_data)-1
            self.treeview_lista_modelos.selection_set(str(new_idx_load)); self.treeview_lista_modelos.focus(str(new_idx_load)); self._on_model_select_from_treeview()
            self.log(f"Modelo '{loaded_md.get('model_name')}' cargado desde: {fpath_load}", "SUCCESS"); messagebox.showinfo("Modelo Cargado",f"Modelo '{loaded_md.get('model_name')}' cargado.",parent=self.parent_for_dialogs)
        except (pickle.UnpicklingError, ValueError) as e_load_val: self.log(f"Error carga/formato modelo: {e_load_val}","ERROR"); messagebox.showerror("Error Carga/Formato",f"Error al cargar o formato inválido:\n{e_load_val}",parent=self.parent_for_dialogs)
        except Exception as e_load_gen: self.log(f"Error general cargando modelo: {e_load_gen}","ERROR"); traceback.print_exc(limit=3); messagebox.showerror("Error Carga",f"No se pudo cargar:\n{e_load_gen}",parent=self.parent_for_dialogs)

    def _check_model_selected_and_valid(self, check_params=False):
        if not self.selected_model_in_treeview: messagebox.showwarning("Sin Modelo","Seleccione modelo.",parent=self.parent_for_dialogs); return False
        md_obj_chk = self.selected_model_in_treeview.get('model')
        if not (md_obj_chk and isinstance(md_obj_chk, CoxPHFitter)): messagebox.showerror("Error Modelo","Objeto modelo no válido.",parent=self.parent_for_dialogs); self.log(f"Modelo '{self.selected_model_in_treeview.get('model_name','N/A')}' sin CPH válido.","ERROR"); return False
        if check_params and (not hasattr(md_obj_chk,'params_') or md_obj_chk.params_ is None or md_obj_chk.params_.empty):
            messagebox.showinfo("Modelo Nulo","Modelo sin covariables. Función requiere covariables.",parent=self.parent_for_dialogs); self.log(f"Función requiere covariables, modelo '{self.selected_model_in_treeview.get('model_name')}' nulo.","INFO"); return False
        return True

    def create_results_controls(self):
        r_content_rc = self.tab_frame_results_content.interior
        self.log("Creando controles Pestaña Resultados...", "DEBUG")
        frame_opts_plot_global = ttk.LabelFrame(r_content_rc, text="Opciones Globales de Gráficos")
        frame_opts_plot_global.pack(fill=tk.X, padx=10, pady=10, ipady=5)
        ttk.Button(frame_opts_plot_global, text="Configurar Opciones Gráfico Predeterminadas...", command=self._open_global_plot_options_dialog).pack(side=tk.LEFT, padx=10, pady=5)
        
        self.results_display_area_rc = ttk.Frame(r_content_rc, padding=10)
        self.results_display_area_rc.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        ttk.Label(self.results_display_area_rc, text="Seleccione modelo en Pestaña 2 y use botones de acción para ver resultados.", wraplength=600, justify=tk.CENTER, font=("TkDefaultFont",10,"italic")).pack(pady=20,padx=10)
        self.log("Controles Resultados creados.", "DEBUG")

    def _open_global_plot_options_dialog(self):
        PlotOptionsDialog(self.parent_for_dialogs, self.current_plot_options.copy(), self._update_global_plot_options)

    def _update_global_plot_options(self, new_opts_gpo):
        self.current_plot_options = new_opts_gpo.copy()
        self.log("Opciones de gráfico globales actualizadas.", "CONFIG"); messagebox.showinfo("Opciones Actualizadas","Opciones de gráfico predeterminadas actualizadas.",parent=self.parent_for_dialogs)

    def _update_results_buttons_state(self): # Placeholder, botones en Tab 2
        pass 
    
    def show_methodological_report(self):
        if not self._check_model_selected_and_valid(): return
        md_rep = self.selected_model_in_treeview; name_rep = md_rep.get('model_name','N/A')
        text_summary_rep = self._generate_text_summary_for_model(md_rep)
        report_full = f"--- Reporte Metodológico: {name_rep} ---\n\n"
        report_full += "1. Objetivo Modelo:\n   Estimar relación covariables y tiempo-hasta-evento con Modelo Cox.\n\n"
        df_fit_rep_meth = md_rep.get('df_used_for_fit'); num_obs_rep_meth = len(df_fit_rep_meth) if df_fit_rep_meth is not None else 'N/A'
        event_col_rep_meth = md_rep.get('event_col_for_model','N/A'); num_events_rep_meth = 'N/A'
        if df_fit_rep_meth is not None and event_col_rep_meth in df_fit_rep_meth:
            try: num_events_rep_meth = int(df_fit_rep_meth[event_col_rep_meth].sum())
            except: pass
        report_full += f"2. Datos Usados (post-preparación para este modelo):\n   - Observaciones: {num_obs_rep_meth}\n   - Eventos: {num_events_rep_meth}\n\n"
        report_full += "3. Contenido Resumen Técnico (ver abajo):\n"
        report_full += "   - Configuración ajuste.\n   - Coeficientes (HRs, ICs).\n   - Métricas ajuste/evaluación.\n   - Test Supuestos (Schoenfeld).\n\n"
        report_full += text_summary_rep
        report_full += "\n\n4. Limitaciones y Consideraciones (Placeholder):\n   [Describa limitaciones y generalizabilidad.]\n\n"
        report_full += "5. Conclusión General (Placeholder):\n   [Interprete hallazgos en contexto.]\n"
        ModelSummaryWindow(self.parent_for_dialogs, f"Reporte Metodológico: {name_rep}", report_full)
        self.log(f"Mostrando reporte metodológico para '{name_rep}'.", "INFO")

# --- Fin de la clase CoxModelingApp ---

if __name__ == "__main__":
    root = tk.Tk()
    root.title(f"Software Modelos de Supervivencia de Cox v1.2.26") # Sincronizar con log abajo
    
    screen_w = root.winfo_screenwidth(); screen_h = root.winfo_screenheight()
    app_w = int(screen_w * 0.90); app_h = int(screen_h * 0.88)
    center_x = max(0, (screen_w - app_w) // 2); center_y = max(0, (screen_h - app_h) // 2)
    root.geometry(f"{app_w}x{app_h}+{center_x}+{center_y}"); root.minsize(1050, 720)
    
    style = ttk.Style()
    themes = style.theme_names()
    preferred_themes = ['clam', 'alt', 'default', 'classic']
    if os.name == 'nt': preferred_themes = ['vista', 'xpnative'] + preferred_themes
    
    chosen_theme = style.theme_use() # Get default first
    for theme_name in preferred_themes:
        if theme_name in themes:
            try: style.theme_use(theme_name); chosen_theme = theme_name; break
            except tk.TclError: pass
    print(f"INFO: Tema UI: '{chosen_theme}'")

    app = CoxModelingApp(root)
    
    app_version = "1.2.26" # Mantener sincronizado con el título
    app.log("*"*80, "HEADER"); app.log(f"  Software Modelado Cox (v{app_version}) Iniciado  ", "HEADER")
    app.log(f"  Tema UI: {chosen_theme}", "CONFIG"); app.log("*"*80, "HEADER")

    # Chequeos de dependencias
    if not PATSY_AVAILABLE: app.log("ERROR CRÍTICO: 'patsy' NO encontrada. Funciones esenciales deshabilitadas. Instale 'patsy'.", "ERROR")
    else: app.log("'patsy' cargada.", "INFO")
    if not FILTER_COMPONENT_AVAILABLE: app.log("ADVERTENCIA: 'MATLAB_filter_component' NO importado. Filtros avanzados no disponibles.", "WARN")
    else: app.log("'MATLAB_filter_component' cargado.", "INFO")
    if LIFELINES_CALIBRATION_AVAILABLE: app.log("'survival_probability_calibration_plot' disponible.", "INFO")
    else: app.log("ADVERTENCIA: 'survival_probability_calibration_plot' NO disponible.", "WARN")
    if LIFELINES_BRIER_SCORE_AVAILABLE: app.log("'brier_score' disponible.", "INFO")
    else: app.log("ADVERTENCIA: 'brier_score' NO disponible.", "WARN")
    
    root.mainloop()
