#!/usr/bin/env python
# -*- coding: utf-8 -*-

# --- Importaciones Estándar de Python ---
from sklearn.model_selection import KFold
from lifelines.utils import concordance_index
from lifelines import CoxPHFitter, KaplanMeierFitter
import lifelines # Importar lifelines directamente para verificar la versión
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
# from scipy.signal import savgol_filter # noqa - Commented out, replaced by LOESS for h0(t)
from statsmodels.nonparametric.smoothers_lowess import lowess

import matplotlib
matplotlib.use('TkAgg')  # Backend para Tkinter

# --- Importaciones de Lifelines ---
# check_assumptions lo reemplaza en gran medida

try:
    from lifelines.scoring import brier_score
    LIFELINES_BRIER_SCORE_AVAILABLE = True
except ImportError:
    LIFELINES_BRIER_SCORE_AVAILABLE = False
    print("ADVERTENCIA: La función 'brier_score' no pudo ser importada desde 'lifelines.scoring'.")

try:
    from lifelines.calibration import survival_probability_calibration
    LIFELINES_CALIBRATION_AVAILABLE = True
except ImportError:
    LIFELINES_CALIBRATION_AVAILABLE = False
    print("ADVERTENCIA: 'survival_probability_calibration' no pudo ser importada. Gráfico de Calibración no disponible.")


try:
    from patsy import dmatrix
    PATSY_AVAILABLE = True
except ImportError:
    dmatrix = None
    PATSY_AVAILABLE = False
    print("ADVERTENCIA: 'patsy' no instalada. Funciones de Spline y manejo avanzado de categóricas limitadas.")

try:
    from MATLAB_filter_component import FilterComponent
    FILTER_COMPONENT_AVAILABLE = True
except ImportError:
    FilterComponent = None
    FILTER_COMPONENT_AVAILABLE = False
    print(f"INFO: MATLAB_filter_component not found in standard Python paths. Advanced filters may be unavailable.")
except Exception as e:
    FilterComponent = None
    FILTER_COMPONENT_AVAILABLE = False
    print(f"ERROR inesperado al importar MATLAB_filter_component: {e}. Filtros avanzados no disponibles.")
    traceback.print_exc(limit=None)

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
    if log_func: # Ensure log_func is provided
        log_func(f"DEBUG: compute_model_metrics: Received schoenfeld_results_df (type: {type(schoenfeld_results_df)}).", "DEBUG")
        if isinstance(schoenfeld_results_df, pd.DataFrame):
            log_func(f"DEBUG: compute_model_metrics: schoenfeld_results_df is DataFrame. Empty: {schoenfeld_results_df.empty}. Shape: {schoenfeld_results_df.shape}", "DEBUG")
            if not schoenfeld_results_df.empty:
                log_func(f"DEBUG: compute_model_metrics: schoenfeld_results_df head:\n{schoenfeld_results_df.head().to_string()}", "DEBUG")
        else:
            log_func(f"DEBUG: compute_model_metrics: schoenfeld_results_df is not DataFrame.", "DEBUG")

    if schoenfeld_results_df is not None and isinstance(schoenfeld_results_df, pd.DataFrame) and not schoenfeld_results_df.empty:
        if 'p' in schoenfeld_results_df.columns:
            found_global_p = False
            index_as_str_lower = []
            if hasattr(schoenfeld_results_df, 'index') and schoenfeld_results_df.index is not None:
                try:
                    # Convert index to lowercase strings for comparison safely
                    index_as_str_lower = [str(x).lower() for x in schoenfeld_results_df.index]
                except Exception as e_map_idx:
                    if log_func: log_func(f"DEBUG: Could not map schoenfeld_results_df.index to lowercased strings: {e_map_idx}", "WARN")
                    # index_as_str_lower remains empty

            if index_as_str_lower: # Proceed only if index mapping was successful and yielded content
                for idx_name_candidate in ['global', 'test_statistic', 'global_test', 'overall']:
                    if idx_name_candidate in index_as_str_lower:
                        try:
                            # Find original index labels that match the candidate (case-insensitive)
                            original_matching_indices = [
                                idx_val for idx_val, str_idx_lower_val 
                                in zip(schoenfeld_results_df.index, index_as_str_lower) 
                                if str_idx_lower_val == idx_name_candidate
                            ]
                            if original_matching_indices:
                                # Take the first match
                                p_val_candidate = schoenfeld_results_df.loc[original_matching_indices[0], 'p']
                                # If .loc returns a Series (e.g. due to non-unique index), take the first element
                                if isinstance(p_val_candidate, pd.Series):
                                    schoenfeld_p_global = p_val_candidate.iloc[0]
                                else:
                                    schoenfeld_p_global = p_val_candidate
                                found_global_p = True
                                if log_func: log_func(f"DEBUG: Found Schoenfeld global p-value using index key '{idx_name_candidate}'. Value: {schoenfeld_p_global}", "DEBUG")
                                break 
                        except (IndexError, KeyError) as e_access: 
                            if log_func: log_func(f"DEBUG: Error accessing p-value for index key '{idx_name_candidate}': {e_access}", "DEBUG")
                            pass # Continue to next candidate
            
            if not found_global_p and isinstance(schoenfeld_results_df.index, pd.MultiIndex):
                multi_index_candidates = [('all', '-'), ('GLOBAL', ''), ('Global', ''), ('global', '')] 
                for mi_candidate in multi_index_candidates:
                    if mi_candidate in schoenfeld_results_df.index:
                        try:
                            p_val_candidate = schoenfeld_results_df.loc[mi_candidate, 'p']
                            if isinstance(p_val_candidate, pd.Series): schoenfeld_p_global = p_val_candidate.iloc[0]
                            elif isinstance(p_val_candidate, pd.DataFrame): schoenfeld_p_global = p_val_candidate['p'].iloc[0]
                            else: schoenfeld_p_global = p_val_candidate
                            found_global_p = True
                            if log_func: log_func(f"DEBUG: Found Schoenfeld global p-value using MultiIndex key {mi_candidate}. Value: {schoenfeld_p_global}", "DEBUG")
                            break
                        except (IndexError, KeyError) as e_access_mi:
                            if log_func: log_func(f"DEBUG: Error accessing p-value for MultiIndex key {mi_candidate}: {e_access_mi}", "DEBUG")
                            pass
            
            if not found_global_p and log_func:
                log_func("DEBUG: Global Schoenfeld p-value not found through common keys.", "DEBUG")

        else: # 'p' column not in schoenfeld_results_df
            if log_func: log_func("WARN: 'p' column not found in schoenfeld_results_df. Cannot extract global p-value.", "WARN")
        
        metrics["Schoenfeld details"] = schoenfeld_results_df.to_dict('index') 
    metrics["Schoenfeld p-value (global)"] = schoenfeld_p_global

    # Check for model summary and safely access its properties
    model_summary_df = getattr(model, "summary", None)
    if model_summary_df is not None and isinstance(model_summary_df, pd.DataFrame) and not model_summary_df.empty:
        summary_df = model_summary_df.copy() # Work with a copy
        if 'z' in summary_df.columns:
            z_scores = summary_df['z'].dropna() # z_scores is a Series
            if not z_scores.empty: # Correct check for a Series
                try:
                    wald_stat = float((z_scores ** 2).sum()) # Should be scalar
                    df_wald = len(z_scores) # Should be scalar
                    if df_wald > 0: # Ensure df_wald is positive for chi2.sf
                        metrics["Wald p-value (global approx)"] = scipy.stats.chi2.sf(wald_stat, df_wald)
                    else:
                        metrics["Wald p-value (global approx)"] = None
                except Exception as e_wald:
                    if log_func: log_func(f"DEBUG: Error calculating Wald p-value: {e_wald}", "WARN")
                    metrics["Wald p-value (global approx)"] = None
            else: # z_scores is empty
                 metrics["Wald p-value (global approx)"] = None
        else: # 'z' column not in summary_df
            metrics["Wald p-value (global approx)"] = None

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
        metrics["summary_df"] = summary_df # Store the original copied summary
    else: # model.summary was None, not a DataFrame, or empty
        if log_func: log_func(f"DEBUG: Model summary not available or empty. Type: {type(model_summary_df)}", "DEBUG")
        metrics["Wald p-value (global approx)"] = None
        metrics["summary_df"] = pd.DataFrame() # Ensure it's an empty DF

    # Concordance Index
    metrics["C-Index (Training)"] = getattr(model, "concordance_index_", None) # This is usually a scalar
    metrics["C-Index (CV Mean)"] = c_index_cv_mean # Scalar or None
    metrics["C-Index (CV Std)"] = c_index_cv_std # Scalar or None

    # AIC and BIC calculations
    # Ensure log_likelihood_val, n_obs, num_params are scalars
    if pd.notna(log_likelihood_val) and isinstance(log_likelihood_val, (int, float)) and \
       isinstance(n_obs, (int, float)) and n_obs > 0 and \
       isinstance(num_params, (int, float)) and num_params >= 0:
        metrics["AIC"] = -2 * log_likelihood_val + 2 * num_params
        if n_obs > num_params + 1: # Ensure n_obs is sufficiently larger than num_params for BIC
            metrics["BIC"] = -2 * log_likelihood_val + np.log(n_obs) * num_params
        else:
            metrics["BIC"] = None 
    else:
        if log_func: log_func(f"DEBUG: AIC/BIC not calculated due to invalid inputs: LL={log_likelihood_val}, N={n_obs}, params={num_params}", "DEBUG")
        metrics["AIC"] = None
        metrics["BIC"] = None

    # Global Likelihood Ratio Test
    # Ensure loglik_null is also a scalar
    if pd.notna(log_likelihood_val) and isinstance(log_likelihood_val, (int, float)) and \
       pd.notna(loglik_null) and isinstance(loglik_null, (int, float)) and \
       isinstance(num_params, (int, float)) and num_params > 0:
        lr_stat = -2 * (loglik_null - log_likelihood_val) # lr_stat should be scalar
        if lr_stat < 0: 
            lr_stat = 0.0 # Ensure lr_stat is non-negative
        metrics["Global LR Test p-value"] = scipy.stats.chi2.sf(lr_stat, num_params)
    else:
        if log_func: log_func(f"DEBUG: Global LR Test not calculated due to invalid inputs: LL={log_likelihood_val}, LL_null={loglik_null}, params={num_params}", "DEBUG")
        metrics["Global LR Test p-value"] = None
        
    return metrics

# --- CLASES AUXILIARES PARA LA UI ---

class DetailedCovariateConfigDialog(tk.Toplevel):
    def __init__(self, parent, app_instance, selected_covariates):
        super().__init__(parent)
        self.transient(parent)
        self.grab_set()
        self.title("Configuración Detallada de Covariables")
        self.app_instance = app_instance
        self.selected_covariates = selected_covariates
        self.row_configs = {}

        # Main frame
        main_frame = ttk.Frame(self, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Scrolled Frame
        scrolled_frame = ScrolledFrame(main_frame)
        scrolled_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        self.controls_frame = scrolled_frame.interior

        # Dynamically create configuration rows
        for cov_name in self.selected_covariates:
            self.row_configs[cov_name] = {}

            row_labelframe = ttk.LabelFrame(self.controls_frame, text=cov_name, padding="10")
            row_labelframe.pack(fill=tk.X, pady=5, padx=5)

            # Tipo Variable
            ttk.Label(row_labelframe, text="Tipo:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
            type_var = tk.StringVar(value="Cuantitativa") # Default, will be refined
            self.row_configs[cov_name]['type_var'] = type_var

            rb_cuant = ttk.Radiobutton(row_labelframe, text="Cuantitativa", variable=type_var, value="Cuantitativa",
                                       command=lambda c=cov_name: self._toggle_row_controls_state(c))
            rb_cuant.grid(row=0, column=1, sticky=tk.W, padx=2)
            self.row_configs[cov_name]['rb_cuant'] = rb_cuant

            rb_cual = ttk.Radiobutton(row_labelframe, text="Cualitativa", variable=type_var, value="Cualitativa",
                                      command=lambda c=cov_name: self._toggle_row_controls_state(c))
            rb_cual.grid(row=0, column=2, sticky=tk.W, padx=2)
            self.row_configs[cov_name]['rb_cual'] = rb_cual

            # Ref. Cat.
            ttk.Label(row_labelframe, text="Ref. Cat.:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
            ref_combo = ttk.Combobox(row_labelframe, state="disabled", width=15)
            ref_combo.grid(row=1, column=1, columnspan=2, sticky=tk.EW, padx=5)
            self.row_configs[cov_name]['ref_combo'] = ref_combo

            # Usar Spline
            ttk.Label(row_labelframe, text="Spline:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
            spline_var = tk.BooleanVar(value=False)
            self.row_configs[cov_name]['spline_var'] = spline_var
            cb_spline = ttk.Checkbutton(row_labelframe, text="Usar", variable=spline_var,
                                        command=lambda c=cov_name: self._toggle_row_controls_state(c))
            cb_spline.grid(row=2, column=1, sticky=tk.W, padx=5)
            self.row_configs[cov_name]['cb_spline'] = cb_spline

            # Spline Tipo
            ttk.Label(row_labelframe, text="  Tipo Spline:").grid(row=3, column=0, sticky=tk.W, padx=15, pady=2)
            spline_type_combo = ttk.Combobox(row_labelframe, values=["Natural", "B-spline"], state="disabled", width=10)
            spline_type_combo.set("Natural")
            spline_type_combo.grid(row=3, column=1, columnspan=2, sticky=tk.EW, padx=5)
            self.row_configs[cov_name]['spline_type_combo'] = spline_type_combo

            # Spline DF
            ttk.Label(row_labelframe, text="  Spline DF:").grid(row=4, column=0, sticky=tk.W, padx=15, pady=2)
            spline_df_var = tk.IntVar(value=4)
            self.row_configs[cov_name]['spline_df_var'] = spline_df_var
            spline_df_spinbox = ttk.Spinbox(row_labelframe, from_=2, to=10, textvariable=spline_df_var, width=5, state="disabled")
            spline_df_spinbox.grid(row=4, column=1, sticky=tk.W, padx=5)
            self.row_configs[cov_name]['spline_df_spinbox'] = spline_df_spinbox

            # --- Load existing or inferred configuration for the row ---
            # Type
            current_type = self.app_instance.covariables_type_config.get(cov_name)
            if not current_type and self.app_instance.data is not None and cov_name in self.app_instance.data:
                current_type = "Cuantitativa" if pd.api.types.is_numeric_dtype(self.app_instance.data[cov_name]) else "Cualitativa"
            else: # Fallback if data is somehow not available or column not present (should be caught by caller)
                current_type = "Cuantitativa"
            type_var.set(current_type)

            # Ref. Cat. (if Cualitativa)
            if current_type == "Cualitativa":
                if self.app_instance.data is not None and cov_name in self.app_instance.data:
                    unique_vals = sorted(self.app_instance.data[cov_name].astype(str).unique().tolist())
                    ref_combo['values'] = unique_vals
                    stored_ref_cat = self.app_instance.ref_categories_config.get(cov_name)
                    if stored_ref_cat in unique_vals:
                        ref_combo.set(stored_ref_cat)
                    elif unique_vals:
                        ref_combo.set(unique_vals[0]) # Default to first if not set or invalid

            # Spline (if Cuantitativa)
            if current_type == "Cuantitativa":
                if cov_name in self.app_instance.spline_config_details:
                    spline_var.set(True)
                    spl_conf = self.app_instance.spline_config_details[cov_name]
                    spline_type_combo.set(spl_conf.get('type', 'Natural'))
                    spline_df_var.set(spl_conf.get('df', 4))
                else:
                    spline_var.set(False)

            # Update control states based on loaded/inferred config
            self._toggle_row_controls_state(cov_name)


        # OK/Cancel Buttons
        buttons_frame = ttk.Frame(main_frame)
        buttons_frame.pack(fill=tk.X, pady=(10,0))
        ttk.Button(buttons_frame, text="OK/Aplicar", command=self.apply_configurations).pack(side=tk.RIGHT, padx=5)
        ttk.Button(buttons_frame, text="Cancelar", command=self.destroy).pack(side=tk.RIGHT)

        self.protocol("WM_DELETE_WINDOW", self.destroy)
        self.wait_window(self)

    def _toggle_row_controls_state(self, cov_name):
        if not cov_name in self.row_configs: return

        config = self.row_configs[cov_name]
        var_type = config['type_var'].get()
        use_spline = config['spline_var'].get()

        # Ref Cat Combo
        if var_type == "Cualitativa":
            config['ref_combo'].config(state="readonly")
            # TODO: Populate ref_combo with unique values from self.app_instance.data[cov_name]
            # For now, keeping it simple as per subtask instructions.
            # Example: if self.app_instance.data is not None and cov_name in self.app_instance.data:
            #    unique_vals = sorted(self.app_instance.data[cov_name].astype(str).unique().tolist())
            #    config['ref_combo']['values'] = unique_vals
            #    if unique_vals: config['ref_combo'].set(unique_vals[0])
        else:
            config['ref_combo'].config(state="disabled")
            config['ref_combo'].set("")

        # Spline Checkbutton (itself)
        config['cb_spline'].config(state=tk.NORMAL if var_type == "Cuantitativa" else tk.DISABLED)
        if var_type == "Cualitativa": # If type is changed to Cualitativa, uncheck "Usar Spline"
            config['spline_var'].set(False)
            use_spline = False # Update local var for subsequent logic

        # Spline Type and DF
        spline_details_state = tk.NORMAL if (var_type == "Cuantitativa" and use_spline) else tk.DISABLED
        config['spline_type_combo'].config(state=spline_details_state)
        config['spline_df_spinbox'].config(state=spline_details_state)
        if spline_details_state == tk.DISABLED:
            config['spline_type_combo'].set("Natural")
            config['spline_df_var'].set(4)

    def apply_configurations(self):
        self.app_instance.log("Aplicando configuraciones detalladas de covariables...", "INFO")
        for cov_name, config_widgets in self.row_configs.items():
            new_type = config_widgets['type_var'].get()
            self.app_instance.covariables_type_config[cov_name] = new_type

            if new_type == "Cualitativa":
                selected_ref_cat = config_widgets['ref_combo'].get()
                if selected_ref_cat: # Ensure a selection was made if combobox is active
                    self.app_instance.ref_categories_config[cov_name] = selected_ref_cat
                else: # Handle case where combobox might be empty but type is Cualitativa
                    self.app_instance.log(f"Advertencia: No se seleccionó categoría de referencia para '{cov_name}' (tipo Cualitativa). Se podría usar default de Patsy.", "WARN")
                    if cov_name in self.app_instance.ref_categories_config: # Remove if it was set and now is invalid
                        del self.app_instance.ref_categories_config[cov_name]

                if cov_name in self.app_instance.spline_config_details:
                    del self.app_instance.spline_config_details[cov_name]
                    self.app_instance.log(f"Config. spline eliminada para '{cov_name}' (cambiado a Cualitativa).", "DEBUG")

            elif new_type == "Cuantitativa":
                use_spline = config_widgets['spline_var'].get()
                if use_spline:
                    spline_type = config_widgets['spline_type_combo'].get()
                    spline_df = config_widgets['spline_df_var'].get()
                    self.app_instance.spline_config_details[cov_name] = {'type': spline_type, 'df': spline_df}
                    self.app_instance.log(f"Config. spline aplicada para '{cov_name}': Tipo={spline_type}, DF={spline_df}.", "DEBUG")
                else:
                    if cov_name in self.app_instance.spline_config_details:
                        del self.app_instance.spline_config_details[cov_name]
                        self.app_instance.log(f"Config. spline eliminada para '{cov_name}' (desmarcado).", "DEBUG")

                if cov_name in self.app_instance.ref_categories_config:
                    del self.app_instance.ref_categories_config[cov_name]
                    self.app_instance.log(f"Config. ref.cat. eliminada para '{cov_name}' (cambiado a Cuantitativa).", "DEBUG")

            self.app_instance.log(f"Configuración para '{cov_name}' actualizada: Tipo='{new_type}'.", "CONFIG")

        self.app_instance.log("Todas las configuraciones detalladas han sido procesadas.", "INFO")

        # Refresh the main UI's simple config panel if any of the configured variables are currently selected there
        # This is a simple way to trigger a refresh, assuming on_covariate_select_for_config handles it.
        current_main_selections = self.app_instance.listbox_covariables_disponibles.curselection()
        if current_main_selections:
            # Get the name of the first selected item in the main listbox to trigger its config UI update
            first_selected_idx_main = current_main_selections[0]
            # Check if listbox is not empty and index is valid
            if self.app_instance.listbox_covariables_disponibles.size() > 0 and first_selected_idx_main < self.app_instance.listbox_covariables_disponibles.size() :
                 self.app_instance.on_covariate_select_for_config()
            else: #If selection is somehow invalid or listbox empty, call with no specific event
                 self.app_instance.on_covariate_select_for_config()
        else: #If nothing selected in main listbox, still call it to reset the simple panel
            self.app_instance.on_covariate_select_for_config()

        self.destroy()


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
        self.sort_order_var_tk = StringVar(self, value=self.current_options.get('sort_order', 'original')) # ADD THIS LINE
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

        # Botón para configuración detallada
        btn_config_detallada = ttk.Button(frame_lista_covariables, text="Configurar Seleccionadas Detalladamente...",
                                          command=self.open_detailed_configuration_dialog)
        btn_config_detallada.pack(pady=5, padx=5, fill=tk.X)


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
 
        # Preserve current selection
        previously_selected_covs = []
        for i in self.listbox_covariables_disponibles.curselection():
            previously_selected_covs.append(self.listbox_covariables_disponibles.get(i))
 
        # Actualizar listbox de covariables
        self.listbox_covariables_disponibles.delete(0, tk.END)
        # Excluir la columna de tiempo y evento de las covariables disponibles
        time_sel = self.combo_col_tiempo.get()
        event_sel = self.combo_col_evento.get()
        cov_cols = [c for c in cols if c not in [time_sel, event_sel]]
        for col in cov_cols:
            self.listbox_covariables_disponibles.insert(tk.END, col)
        
        # Re-select previously selected items
        for i, col in enumerate(cov_cols):
            if col in previously_selected_covs:
                self.listbox_covariables_disponibles.selection_set(i)
        
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
        self.log(f"DEBUG: Columnas disponibles en self.data: {self.data.columns.tolist() if self.data is not None else 'N/A'}", "DEBUG")

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
            if self.custom_filter_component_instance:
                self.custom_filter_component_instance.set_dataframe(self.data)
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
            filtered_data = self.custom_filter_component_instance.apply_filters()

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
        
        new_var_type_bulk = self.var_tipo_covariable_seleccionada.get() # Type from main panel
        use_spline_bulk = self.var_usar_spline_seleccionada.get()      # Spline use from main panel
        spline_type_bulk = self.combo_tipo_spline_seleccionada.get()  # Spline type from main panel
        spline_df_bulk = self.var_df_spline_seleccionada.get()        # Spline DF from main panel
        
        ref_category_for_single_selection = None
        if len(selected_var_names) == 1 and new_var_type_bulk == "Cualitativa" and self.combo_ref_categoria_seleccionada.cget('state') != 'disabled':
            ref_category_for_single_selection = self.combo_ref_categoria_seleccionada.get()
            if not ref_category_for_single_selection:
                messagebox.showwarning("Ref. Vacía",
                                       f"Para '{selected_var_names[0]}', seleccione una categoría de referencia o use el diálogo detallado.",
                                       parent=self.parent_for_dialogs)
                return

        num_applied = 0
        for var_name_apply in selected_var_names:
            log_msgs_for_var = [f"Aplicando config (panel simple) a '{var_name_apply}':"]
            
            # 1. Set the new type
            self.covariables_type_config[var_name_apply] = new_var_type_bulk
            log_msgs_for_var.append(f"Tipo='{new_var_type_bulk}'")

            # 2. Clean up and set defaults based on the new type
            if new_var_type_bulk == "Cualitativa":
                # Remove spline config if it exists
                if var_name_apply in self.spline_config_details:
                    del self.spline_config_details[var_name_apply]
                    log_msgs_for_var.append("Config. spline eliminada (tipo cambiado a Cualitativa).")
                
                # Set reference category
                if len(selected_var_names) == 1 and ref_category_for_single_selection is not None:
                    # This case is for single selection where ref_cat is taken from main panel
                    self.ref_categories_config[var_name_apply] = ref_category_for_single_selection
                    log_msgs_for_var.append(f"Ref.Cat.='{ref_category_for_single_selection}'")
                elif var_name_apply not in self.ref_categories_config:
                    # For multiple selections, or single if ref_cat wasn't set from panel,
                    # set a default if data is available and no prior ref_cat exists
                    if self.data is not None and var_name_apply in self.data.columns:
                        try:
                            unique_cats = sorted(list(self.data[var_name_apply].astype(str).unique()))
                            if unique_cats:
                                self.ref_categories_config[var_name_apply] = unique_cats[0]
                                log_msgs_for_var.append(f"Ref.Cat.(default)='{unique_cats[0]}'")
                            else:
                                log_msgs_for_var.append("No hay valores únicos para Ref.Cat.(default).")
                        except Exception as e_unique_cats:
                             log_msgs_for_var.append(f"Error obteniendo Ref.Cat.(default): {e_unique_cats}")
                    else:
                        log_msgs_for_var.append("No hay datos para determinar Ref.Cat.(default).")
                # If var_name_apply IS in ref_categories_config and it's a multiple selection, we keep the existing one.
            
            elif new_var_type_bulk == "Cuantitativa":
                # Remove ref category config if it exists
                if var_name_apply in self.ref_categories_config:
                    del self.ref_categories_config[var_name_apply]
                    log_msgs_for_var.append("Config. Ref.Cat. eliminada (tipo cambiado a Cuantitativa).")
                
                # Apply or remove spline config based on main panel's "Usar Spline"
                if use_spline_bulk:
                    self.spline_config_details[var_name_apply] = {'type': spline_type_bulk, 'df': spline_df_bulk}
                    log_msgs_for_var.append(f"Spline: Tipo='{spline_type_bulk}', DF={spline_df_bulk}")
                else: # Not using spline via main panel
                    if var_name_apply in self.spline_config_details:
                        del self.spline_config_details[var_name_apply]
                        log_msgs_for_var.append("Config. spline eliminada (desmarcado en panel simple).")
            
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
                "Asegúrese que los nombres de columna con espacios o caracteres especiales estén entre acentos graves (backticks), ej: \\`nombre con espacio\\`.\n"
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
        metodos_ties_ui = ["efron", "breslow", "exact"] 
        self.combo_metodo_empates = ttk.Combobox(grid_ties, textvariable=self.tie_handling_method_var, values=metodos_ties_ui, state="readonly", width=18)
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
            ("Gráf. Impacto Variable", self.show_variable_impact_plot), # Modified button text
            ("Predicción", self.realizar_prediccion), ("Exportar Resumen", self.export_model_summary),
            ("Guardar Modelo", self.save_model), ("Cargar Modelo", self.load_model_from_file),
            ("Reporte Metod.", self.show_methodological_report),
            ("Riesgo Inst. Base", self.show_baseline_instantaneous_hazard) # New button
        ]
        
        # Layout dinámico para botones de acción
        max_btns_per_row = 7 # Adjusted for the new button
        current_row_frame_acciones = None
        for i, (text, cmd) in enumerate(acciones_config_btns):
            if i % max_btns_per_row == 0:
                current_row_frame_acciones = ttk.Frame(frame_acciones)
                current_row_frame_acciones.pack(fill=tk.X, pady=1)
            ttk.Button(current_row_frame_acciones, text=text, command=cmd).pack(side=tk.LEFT, padx=3, pady=2, fill=tk.X, expand=True)

        # Add the new button row for clear models
        clear_models_frame = ttk.Frame(frame_acciones)
        clear_models_frame.pack(fill=tk.X, pady=5)
        ttk.Button(clear_models_frame, text="Limpiar Todos los Modelos", command=self._clear_all_generated_models).pack(side=tk.RIGHT, padx=5)

        # Add the new button row for clear models

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
            if final_t_col in df_model_prep.columns and final_t_col != time_col_ui : 
                 self.log(f"Advertencia: Nombre renombrado para Tiempo '{final_t_col}' ya existe o es el original. No se renombrará.", "WARN"); final_t_col = time_col_ui
            else: renames[time_col_ui] = final_t_col
        if final_e_col != event_col_ui:
            if final_e_col in df_model_prep.columns and final_e_col != event_col_ui:
                 self.log(f"Advertencia: Nombre renombrado para Evento '{final_e_col}' ya existe o es el original. No se renombrará.", "WARN"); final_e_col = event_col_ui
            else: renames[event_col_ui] = final_e_col
        
        if renames: df_model_prep.rename(columns=renames, inplace=True); self.log(f"Columnas renombradas: {renames}", "INFO")
 
        self.log(f"DEBUG: df_model_prep columns before T/E conversion/dropna: {df_model_prep.columns.tolist()}", "DEBUG")
        self.log(f"DEBUG: selected_covs_orig_names: {selected_covs_orig_names}", "DEBUG")
 
        # Validación y conversión de tipos para Tiempo y Evento
        try:
            df_model_prep[final_t_col] = pd.to_numeric(df_model_prep[final_t_col])
            if df_model_prep[final_t_col].min() <= 0 and not (df_model_prep[final_t_col] == 0).all() : # Permitir si todos son cero (caso raro)
                self.log(f"Advertencia: Columna Tiempo '{final_t_col}' contiene valores no positivos (<=0). Estos pueden causar problemas.", "WARN")
        except ValueError as e_t:
            self.log(f"Error convirtiendo columna Tiempo '{final_t_col}' a numérico: {e_t}", "ERROR")
            messagebox.showerror("Error de Tipo", f"Columna Tiempo '{final_t_col}' no puede ser convertida a numérica.", parent=self.parent_for_dialogs)
            return None
        
        try:
            df_model_prep[final_e_col] = pd.to_numeric(df_model_prep[final_e_col])
            df_model_prep.dropna(subset=[final_t_col, final_e_col], inplace=True)
            if df_model_prep.empty:
                 self.log(f"Dataset vacío después de convertir T/E a numérico y eliminar NaNs en T/E.", "ERROR")
                 messagebox.showerror("Datos Insuficientes", "No quedan datos válidos para T/E después de la conversión a numérico y eliminación de NaNs.", parent=self.parent_for_dialogs)
                 return None
 
            if not df_model_prep[final_e_col].isin([0, 1]).all():
                 num_invalid_events = df_model_prep[~df_model_prep[final_e_col].isin([0, 1])].shape[0]
                 self.log(f"Columna Evento '{final_e_col}' tiene {num_invalid_events} valor(es) que no son 0 o 1 después de conversión y dropna.", "ERROR")
                 messagebox.showerror("Error de Tipo", f"Columna Evento '{final_e_col}' debe contener solo valores 0 o 1.", parent=self.parent_for_dialogs)
                 return None
            df_model_prep[final_e_col] = df_model_prep[final_e_col].astype(int)
        except ValueError as e_e: # Si to_numeric falla completamente
            self.log(f"Error convirtiendo columna Evento '{final_e_col}' a numérico 0/1: {e_e}", "ERROR")
            messagebox.showerror("Error de Tipo", f"Columna Evento '{final_e_col}' no puede ser convertida a numérica (0/1).", parent=self.parent_for_dialogs)
            return None
 
        initial_rows_prep = len(df_model_prep)
        df_model_prep.dropna(subset=[final_t_col, final_e_col], inplace=True)
        if len(df_model_prep) < initial_rows_prep: self.log(f"Eliminadas {initial_rows_prep - len(df_model_prep)} filas con NaN en T/E (posiblemente de conversión).", "WARN")
        if df_model_prep.empty: self.log("DF vacío post-NaN en T/E.", "ERROR"); messagebox.showerror("Datos Insuficientes", "No quedan datos post-NaN en T/E.", parent=self.parent_for_dialogs); return None
 
        df_filtered_patsy, X_design_patsy, formula_patsy_gen, terms_patsy_display = self.build_design_matrix(
            df_model_prep, selected_covs_orig_names, final_t_col, final_e_col
        )
        self.log(f"DEBUG: df_filtered_patsy columns after build_design_matrix: {df_filtered_patsy.columns.tolist() if df_filtered_patsy is not None else 'N/A'}", "DEBUG")

        if X_design_patsy is None or df_filtered_patsy is None: 
             self.log("Falló build_design_matrix.", "ERROR"); return None
        
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

        df_for_patsy_bd = df_input_bd.copy() 

        if not selected_covs_orig_names_bd: 
            formula_patsy_bd = "0" 
            try:
                if df_for_patsy_bd.empty:
                    X_design_bd = pd.DataFrame(index=df_for_patsy_bd.index) 
                else:
                    X_design_bd = dmatrix(formula_patsy_bd, df_for_patsy_bd, return_type="dataframe")
                
                self.log("Matriz de diseño para modelo nulo (sin covariables).", "INFO")
                return df_for_patsy_bd, X_design_bd, formula_patsy_bd, []
            except Exception as e_patsy_null:
                self.log(f"Error Patsy (modelo nulo): {e_patsy_null}", "ERROR"); traceback.print_exc(limit=3); return None, None, None, None
        
        formula_parts_bd = []
        for orig_cov_name_bd in selected_covs_orig_names_bd:
            if orig_cov_name_bd not in df_for_patsy_bd.columns:
                self.log(f"Advertencia: Cov. original '{orig_cov_name_bd}' no en DF para Patsy. Saltando.", "WARN"); continue

            config_type_bd = self.covariables_type_config.get(orig_cov_name_bd, "Cuantitativa" if pd.api.types.is_numeric_dtype(df_for_patsy_bd[orig_cov_name_bd]) else "Cualitativa")
            
            term_syntax_bd = f"Q('{orig_cov_name_bd}')" 
            if config_type_bd == "Cuantitativa":
                if orig_cov_name_bd in self.spline_config_details:
                    spl_cfg_bd = self.spline_config_details[orig_cov_name_bd]
                    patsy_func_bd = 'cr' if spl_cfg_bd.get('type', 'Natural') == 'Natural' else 'bs'
                    term_syntax_bd = f"{patsy_func_bd}(Q('{orig_cov_name_bd}'), df={spl_cfg_bd.get('df', 4)})"
            else: 
                if not pd.api.types.is_categorical_dtype(df_for_patsy_bd[orig_cov_name_bd].dtype) and \
                   not pd.api.types.is_string_dtype(df_for_patsy_bd[orig_cov_name_bd].dtype) and \
                   not pd.api.types.is_object_dtype(df_for_patsy_bd[orig_cov_name_bd].dtype): 
                     df_for_patsy_bd[orig_cov_name_bd] = df_for_patsy_bd[orig_cov_name_bd].astype(str)

                ref_cat_bd = self.ref_categories_config.get(orig_cov_name_bd)
                if ref_cat_bd and str(ref_cat_bd).strip():
                    ref_cat_str_bd = str(ref_cat_bd)
                    if ref_cat_str_bd in df_for_patsy_bd[orig_cov_name_bd].astype(str).unique():
                        # For string literals like 'F', Patsy expects Treatment('F')
                        # If ref_cat_str_bd could be numeric, further type checking might be needed,
                        # but for now, assuming string reference categories are common.
                        # Enclosing ref_cat_str_bd in single quotes within the f-string if it's not purely numeric.
                        if re.match(r"^-?\d+(\.\d+)?$", ref_cat_str_bd): # Check if it looks like a number
                             term_syntax_bd = f"C(Q('{orig_cov_name_bd}'), Treatment({ref_cat_str_bd}))"
                        else: # Assume string, enclose in quotes for Patsy
                             term_syntax_bd = f"C(Q('{orig_cov_name_bd}'), Treatment('{ref_cat_str_bd}'))"
                    else:
                        self.log(f"Advertencia: Ref.Cat. '{ref_cat_str_bd}' para '{orig_cov_name_bd}' no en datos. Usando default Patsy.", "WARN")
                        term_syntax_bd = f"C(Q('{orig_cov_name_bd}'))"
                else:
                    term_syntax_bd = f"C(Q('{orig_cov_name_bd}'))" 
            formula_parts_bd.append(term_syntax_bd)

        formula_patsy_bd = "0 + " + " + ".join(formula_parts_bd) if formula_parts_bd else "0"
        self.log(f"Fórmula Patsy generada: {formula_patsy_bd}", "DEBUG")

        try:
            if df_for_patsy_bd.empty and formula_patsy_bd != "0": 
                 self.log("DF entrada Patsy vacío con fórmula no nula.", "ERROR"); return None,None,None,None
            
            X_design_bd = dmatrix(formula_patsy_bd, df_for_patsy_bd, return_type="dataframe")
            df_filtered_by_patsy_idx_bd = df_input_bd.loc[X_design_bd.index].copy()
            
            final_terms_display_bd = list(X_design_bd.columns)
            self.log(f"Patsy: X_design ({X_design_bd.shape}), DF filtrado ({df_filtered_by_patsy_idx_bd.shape})", "INFO")
            self.log(f"Términos Patsy finales: {final_terms_display_bd}", "DEBUG")
            return df_filtered_by_patsy_idx_bd, X_design_bd, formula_patsy_bd, final_terms_display_bd
        except Exception as e_patsy_build:
            self.log(f"Error Patsy (build matriz): {e_patsy_build}", "ERROR"); traceback.print_exc(limit=5)
            messagebox.showerror("Error Patsy", f"Error construyendo matriz de diseño:\n{e_patsy_build}", parent=self.parent_for_dialogs)
            return None, None, None, None


    def _perform_variable_selection(self, df_aligned_orig_vs, X_design_initial_vs, time_col_vs, event_col_vs, formula_initial_vs, terms_initial_vs):
        method_vs = self.var_selection_method_var.get()

        selected_covs_orig_names = []
        # Regex to extract original variable names from Q('var_name') in Patsy terms
        regex_q_var = re.compile(r"Q\('([^']+)'\)")

        if terms_initial_vs is None: # Ensure terms_initial_vs is iterable
            terms_initial_vs = []

        # Extract all potential original covariate names from the initial full model
        all_initial_orig_covs = []
        for term in terms_initial_vs:
            matches = regex_q_var.findall(term)
            for original_var_name in matches:
                if original_var_name not in all_initial_orig_covs:
                    all_initial_orig_covs.append(original_var_name)

        if method_vs == "Ninguno (usar todas)":
            self.log("Selección Variables: 'Ninguno (usar todas)'. Usando todas las covariables iniciales.", "INFO")
            selected_covs_orig_names = all_initial_orig_covs
        
        elif method_vs in ["Backward", "Forward", "Stepwise (Fwd luego Bwd)"]:
            self.log(f"Selección Variables: '{method_vs}' no está soportado directamente en esta versión de Lifelines. "
                       f"Se procederá usando todas las covariables iniciales, similar a 'Ninguno (usar todas)'.", "WARN")
            messagebox.showwarning("Método No Soportado",
                                   f"El método de selección de variables '{method_vs}' no está directamente disponible "
                                   f"en la versión actual de la librería 'lifelines'.\n\n"
                                   f"El modelo se ajustará utilizando todas las covariables seleccionadas inicialmente.",
                                   parent=self.parent_for_dialogs)
            selected_covs_orig_names = all_initial_orig_covs # Default to using all variables

        else: # Should not happen given UI choices
            self.log(f"Método de selección desconocido: {method_vs}. Usando todas las covariables.", "ERROR")
            selected_covs_orig_names = all_initial_orig_covs

        if not selected_covs_orig_names and all_initial_orig_covs:
             # This case might occur if logic changes, but generally if all_initial_orig_covs is not empty,
             # selected_covs_orig_names should also not be empty for the above paths.
             self.log("Advertencia: No se seleccionaron covariables finales, pero había covariables iniciales. Esto podría ser un error.", "WARN")

        # This function must return a list of original covariate names.
        # The calling function _execute_cox_modeling_orchestrator will then use these names
        # to call build_design_matrix again.
        self.log(f"Covariables originales seleccionadas/pasadas para reconstrucción: {selected_covs_orig_names}", "DEBUG")
        return selected_covs_orig_names

    def _run_model_and_get_metrics(self, df_lifelines_rm, X_design_rm, y_survival_rm, time_col_rm, event_col_rm, 
                                   formula_patsy_rm, model_name_rm, covariates_display_terms_rm, 
                                   full_patsy_formula_for_new_data_transform_arg, 
                                   penalizer_val_rm=0.0, l1_ratio_val_rm=0.0, 
                                   model_type_for_fit_logic="Multivariado"): 
        self.log(f"Ajustando modelo Cox: '{model_name_rm}'...", "INFO")
        
        ui_selected_tie_method = self.tie_handling_method_var.get() # Para registro
        
        model_data_rm = {
            "model_name": model_name_rm, "time_col_for_model": time_col_rm, "event_col_for_model": event_col_rm,
            "formula_patsy": formula_patsy_rm, 
            "full_patsy_formula_for_new_data_transform": full_patsy_formula_for_new_data_transform_arg, 
            "covariates_processed": covariates_display_terms_rm,
            "df_used_for_fit": self.data.copy(), # Almacenar el DataFrame completo original (después de filtros iniciales)
            "X_design_used_for_fit": X_design_rm.copy(),
            "y_survival_used_for_fit": y_survival_rm.copy(),
            "penalizer_value": penalizer_val_rm, "l1_ratio_value": l1_ratio_val_rm,
            "tie_method_used": ui_selected_tie_method,
            "metrics": {}, "schoenfeld_results": None, "model": None, "loglik_null": None,
            "c_index_cv_mean": None, "c_index_cv_std": None,
            "schoenfeld_status_message": None, # Initialize status message
            "proportional_hazard_test_summary": None # Initialize new key
        }
 
        try:
            # Modelo Nulo (para LogLik nulo)
            cph_null_rm = CoxPHFitter(penalizer=0.0)
            # Para el modelo nulo, solo necesitamos las columnas de tiempo y evento del df_lifelines_rm
            df_for_null_fit_rm = df_lifelines_rm[[time_col_rm, event_col_rm]].copy()
            cph_null_rm.fit(df_for_null_fit_rm, duration_col=time_col_rm, event_col=event_col_rm, formula="0")
            model_data_rm["loglik_null"] = cph_null_rm.log_likelihood_
 
            # Modelo Principal
            cph_main_rm = CoxPHFitter(penalizer=penalizer_val_rm, l1_ratio=l1_ratio_val_rm)
            
            # Siempre usar el DataFrame original (df_lifelines_rm) y la fórmula de Patsy (formula_patsy_rm)
            # lifelines.fit() se encargará de construir la matriz de diseño internamente.
            df_for_fit_main = df_lifelines_rm.copy()
            model_data_rm["df_final_fit_shape"] = df_for_fit_main.shape
            actual_formula_for_fit = formula_patsy_rm
            
            self.log(f"DEBUG: df_for_fit_main columns before fit: {df_for_fit_main.columns.tolist()}", "DEBUG")
            self.log(f"DEBUG: actual_formula_for_fit before fit: {actual_formula_for_fit}", "DEBUG")
            
            cph_main_rm.fit(df_for_fit_main, duration_col=time_col_rm, event_col=event_col_rm, formula=actual_formula_for_fit)
            
            model_data_rm["model"] = cph_main_rm
            self.log(f"Modelo '{model_name_rm}' ajustado.", "SUCCESS")

            # Test de Schoenfeld
            if not X_design_rm.empty:
                if hasattr(cph_main_rm, 'params_') and not cph_main_rm.params_.empty:
                    self.log(f"--- Iniciando Test de Schoenfeld para Modelo: '{model_name_rm}' ---", "INFO")
                    self.log(f"Pre-check_assumptions: df_for_fit_main shape: {df_for_fit_main.shape}", "DEBUG")
                    self.log(f"Pre-check_assumptions: df_for_fit_main dtypes:\n{df_for_fit_main.dtypes.to_string()}", "DEBUG")
                    self.log(f"Pre-check_assumptions: df_for_fit_main columns: {df_for_fit_main.columns.tolist()}", "DEBUG")
                    self.log(f"Pre-check_assumptions: duration_col: {time_col_rm}, event_col: {event_col_rm}", "DEBUG")
                    self.log(f"Pre-check_assumptions: formula for cph_main_rm.fit: {actual_formula_for_fit}", "DEBUG")
                    self.log(f"Pre-check_assumptions: X_design_rm is empty: {X_design_rm.empty}", "DEBUG")
                    params_empty = not hasattr(cph_main_rm, 'params_') or cph_main_rm.params_ is None or cph_main_rm.params_.empty
                    self.log(f"Pre-check_assumptions: cph_main_rm.params_ is empty: {params_empty}", "DEBUG")

                    results_check_assumptions = None # Initialize
                    model_data_rm["schoenfeld_results"] = pd.DataFrame() # Ensure initialized as DataFrame
                    model_data_rm["schoenfeld_status_message"] = "Test de Schoenfeld no se ejecutó o no aplicó inicialmente." # Default status

                    try:
                        self.log("Calling cph_main_rm.check_assumptions(df_for_fit_main)...", "DEBUG")
                        results_check_assumptions = cph_main_rm.check_assumptions(df_for_fit_main)
                        model_data_rm["check_assumptions_results_raw"] = results_check_assumptions # Store raw results
                        self.log("Post-check_assumptions: Call completed.", "DEBUG")
                        self.log(f"DEBUG: Post-check_assumptions: Type of results_check_assumptions: {type(results_check_assumptions)}", "DEBUG")

                        schoenfeld_df_candidate = None
                        found_schoenfeld_results = False # Renamed for clarity from the previous version's logic

                        if isinstance(results_check_assumptions, list):
                            self.log(f"DEBUG: `check_assumptions` returned a list with {len(results_check_assumptions)} elements.", "DEBUG")
                            if not results_check_assumptions: # Empty list
                                self.log("INFO: `check_assumptions` devolvió una lista vacía.", "INFO")
                                model_data_rm["schoenfeld_status_message"] = "Test de Schoenfeld no arrojó datos detallados (lista vacía de check_assumptions)."
                            else: # Iterate through the list
                                for i, result_candidate_item in enumerate(results_check_assumptions): # Renamed item to result_candidate_item
                                    self.log(f"DEBUG: Checking element {i} in results_check_assumptions (type: {type(result_candidate_item)}).", "DEBUG")
                                    # Primary way: check if the item itself is the DataFrame we want
                                    if isinstance(result_candidate_item, pd.DataFrame) and not result_candidate_item.empty:
                                        # Check for columns that indicate it's the Schoenfeld summary
                                        # Using 'test_statistic' and 'p' as generally reliable indicators.
                                        if all(col in result_candidate_item.columns for col in ['test_statistic', 'p']):
                                            self.log(f"DEBUG: Found candidate Schoenfeld DataFrame in results_check_assumptions list (direct DataFrame check at index {i}). Shape: {result_candidate_item.shape}", "DEBUG")
                                            schoenfeld_df_candidate = result_candidate_item
                                            found_schoenfeld_results = True
                                            break
                                    # Secondary way: check if it's an object with a .summary DataFrame
                                    elif hasattr(result_candidate_item, 'summary') and isinstance(result_candidate_item.summary, pd.DataFrame) and not result_candidate_item.summary.empty:
                                        # Also check if this summary DataFrame has the expected columns
                                        summary_df_check = result_candidate_item.summary
                                        if all(col in summary_df_check.columns for col in ['test_statistic', 'p']):
                                            self.log(f"DEBUG: Found candidate Schoenfeld DataFrame in results_check_assumptions list (via .summary attribute at index {i}). Shape: {summary_df_check.shape}", "DEBUG")
                                            schoenfeld_df_candidate = summary_df_check
                                            found_schoenfeld_results = True
                                            break
                                # Fallback if no suitable DataFrame found by iterating, but the list had a known structure (e.g., result at index 1)
                                # This part retains a bit of the original logic if the list structure is [bool, DataFrame, ...]
                                if not found_schoenfeld_results and len(results_check_assumptions) >= 2 and isinstance(results_check_assumptions[1], pd.DataFrame):
                                    self.log("WARN: No specific Schoenfeld DataFrame found by primary checks. Falling back to checking results_check_assumptions[1] if it's a DataFrame.", "WARN")
                                    potential_fallback_df = results_check_assumptions[1]
                                    if not potential_fallback_df.empty and all(col in potential_fallback_df.columns for col in ['test_statistic', 'p']):
                                        schoenfeld_df_candidate = potential_fallback_df
                                        found_schoenfeld_results = True
                                        self.log(f"DEBUG: Using fallback: results_check_assumptions[1] as Schoenfeld DataFrame. Shape: {schoenfeld_df_candidate.shape}", "DEBUG")


                        else: # Not a list
                            self.log("WARN: `check_assumptions` no devolvió una lista.", "WARN")
                            model_data_rm["schoenfeld_status_message"] = "Resultados del Test de Schoenfeld no tuvieron el formato esperado (check_assumptions no devolvió lista)."

                        # Process the candidate DataFrame
                        if found_schoenfeld_results and schoenfeld_df_candidate is not None: # schoenfeld_df_candidate should be a DataFrame here
                            if schoenfeld_df_candidate.empty: # This check is now redundant if loops check for non-empty, but safe
                                model_data_rm["schoenfeld_results"] = pd.DataFrame() # Ensure it's an empty DF, not the empty one found
                                self.log("INFO: Test de Schoenfeld devolvió un DataFrame vacío (o candidate was empty).", "INFO")
                                model_data_rm["schoenfeld_status_message"] = "Test de Schoenfeld no arrojó datos detallados (DataFrame vacío de check_assumptions)."
                            else:
                                model_data_rm["schoenfeld_results"] = schoenfeld_df_candidate.copy() # Use .copy()
                                self.log("INFO: Test de Schoenfeld calculado exitosamente y DataFrame (schoenfeld_results) no está vacío.", "INFO")
                                model_data_rm["schoenfeld_status_message"] = "Test de Schoenfeld (check_assumptions) calculado exitosamente."
                                self.log(f"DEBUG: Schoenfeld DataFrame shape: {model_data_rm['schoenfeld_results'].shape}", "DEBUG")
                                self.log(f"DEBUG: Schoenfeld DataFrame columns: {model_data_rm['schoenfeld_results'].columns.tolist()}", "DEBUG")
                                self.log(f"DEBUG: Schoenfeld DataFrame head:\n{model_data_rm['schoenfeld_results'].head().to_string()}", "DEBUG")
                        # If not found_schoenfeld_results but it was a list and not empty (e.g. list of bools, or DFs not matching criteria)
                        elif not found_schoenfeld_results and isinstance(results_check_assumptions, list) and results_check_assumptions:
                             self.log("WARN: No se encontró un DataFrame de Schoenfeld adecuado en la lista de `check_assumptions` (elementos no eran DF esperados o eran vacíos).", "WARN")
                             model_data_rm["schoenfeld_status_message"] = "Schoenfeld: resultados detallados no encontrados o en formato inesperado en la lista de check_assumptions."
                        # If results_check_assumptions was not a list, or was an empty list, status messages are already set.
                        # model_data_rm["schoenfeld_results"] remains an empty DataFrame if no candidate was found or processed.

                    except Exception as e_sch_detailed:
                        model_data_rm["schoenfeld_status_message"] = "Error durante procesamiento de resultados del Test de Schoenfeld (check_assumptions)."
                        detailed_tb = traceback.format_exc()
                        self.log(f"CRITICAL ERROR during Test Schoenfeld (cph_main_rm.check_assumptions call): {e_sch_detailed}\nTraceback:\n{detailed_tb}", "ERROR")
                        # model_data_rm["schoenfeld_results"] remains an empty DataFrame (initialized before try block)

                    self.log(f"--- Test de Schoenfeld para Modelo: '{model_name_rm}' Finalizado ---", "INFO")
                else: # No model parameters
                    self.log("INFO: Modelo sin parámetros (covariables). Test de Schoenfeld no aplicable.", "INFO")
                    model_data_rm["schoenfeld_status_message"] = "Test de Schoenfeld no aplicable (modelo sin covariables)."
                    # model_data_rm["schoenfeld_results"] is already an empty DataFrame
            else: # No X_design (null model)
                self.log("INFO: Modelo nulo (X_design_rm está vacío). Test de Schoenfeld no aplicable.", "INFO")
                model_data_rm["schoenfeld_status_message"] = "Test de Schoenfeld no aplicable (modelo nulo)."
                # model_data_rm["schoenfeld_results"] is already an empty DataFrame

            # Fallback or alternative: proportional_hazard_test
            # Decide whether to try proportional_hazard_test based on the status and content of schoenfeld_results
            current_schoenfeld_df = model_data_rm.get("schoenfeld_results")
            current_schoenfeld_status = model_data_rm.get("schoenfeld_status_message", "")
            
            # Try proportional_hazard_test if check_assumptions didn't yield a non-empty DataFrame of residuals,
            # or if check_assumptions itself had issues that didn't prevent trying an alternative.
            should_try_ph_test = False
            if isinstance(current_schoenfeld_df, pd.DataFrame) and current_schoenfeld_df.empty:
                if "calculado exitosamente" not in current_schoenfeld_status: # e.g. if it was "DataFrame vacío de check_assumptions"
                    should_try_ph_test = True
            elif "Error durante cálculo" in current_schoenfeld_status or \
                 "no tuvieron el formato esperado" in current_schoenfeld_status or \
                 "no arrojó datos detallados" in current_schoenfeld_status or \
                 "Schoenfeld: resultados detallados no encontrados" in current_schoenfeld_status :
                should_try_ph_test = True
            
            # Ensure model has params for ph_test if we are to try it
            if should_try_ph_test and not (hasattr(cph_main_rm, 'params_') and cph_main_rm.params_ is not None and not cph_main_rm.params_.empty):
                self.log("INFO: `proportional_hazard_test` no se intentará porque el modelo no tiene parámetros (similar a check_assumptions).", "INFO")
                should_try_ph_test = False
            
            if should_try_ph_test:
                self.log("INFO: `check_assumptions` no proporcionó resultados detallados de Schoenfeld. Intentando `proportional_hazard_test` por separado.", "INFO")
                try:
                    from lifelines.statistics import proportional_hazard_test

                    # df_for_fit_main is the DataFrame used for cph_main_rm.fit()
                    # cph_main_rm is the fitted model object
                    ph_test_results_obj = proportional_hazard_test(cph_main_rm, df_for_fit_main, time_transform='log')

                    if ph_test_results_obj is not None and hasattr(ph_test_results_obj, 'summary') and isinstance(ph_test_results_obj.summary, pd.DataFrame) and not ph_test_results_obj.summary.empty:
                        model_data_rm["proportional_hazard_test_summary"] = ph_test_results_obj.summary
                        self.log("INFO: `proportional_hazard_test` ejecutado exitosamente y su resumen ha sido almacenado.", "INFO")
                        self.log(f"DEBUG: Resumen de proportional_hazard_test (shape: {ph_test_results_obj.summary.shape}):\n{ph_test_results_obj.summary.head().to_string()}", "DEBUG")

                        # Update status message to reflect this new information
                        # Update status message to reflect this new information, building on previous status
                        prev_status = model_data_rm["schoenfeld_status_message"]
                        if "no arrojó datos detallados" in prev_status or "DataFrame vacío" in prev_status or "formato inesperado" in prev_status or "no encontrados" in prev_status:
                             model_data_rm["schoenfeld_status_message"] = f"{prev_status.replace('.', '')}, pero se obtuvieron resultados de proportional_hazard_test."
                        elif "Error durante cálculo" in prev_status: # If check_assumptions had an error
                             model_data_rm["schoenfeld_status_message"] = f"{prev_status.replace('.', '')}; adicionalmente, proportional_hazard_test también fue ejecutado y proporcionó un resumen."
                        else: # Generic addition
                             model_data_rm["schoenfeld_status_message"] = f"{prev_status.replace('.', '')}. Adicionalmente, se obtuvieron resultados de proportional_hazard_test."
                    else: # proportional_hazard_test did not return a valid summary
                        self.log("WARN: `proportional_hazard_test` no devolvió un resumen válido (DataFrame no vacío).", "WARN")
                        prev_status = model_data_rm["schoenfeld_status_message"]
                        ph_test_fail_suffix = " Adicionalmente, el intento con proportional_hazard_test tampoco arrojó un resumen."
                        # Append suffix if the original status was about missing data or format issues from check_assumptions
                        if any(phrase in prev_status for phrase in ["no arrojó datos detallados", "DataFrame vacío", "formato inesperado", "no encontrados", "Error durante cálculo"]):
                            model_data_rm["schoenfeld_status_message"] = prev_status.replace(".","") + ph_test_fail_suffix
                        # else, if it was "calculado exitosamente" from check_assumptions, that message is probably fine.

                except ImportError:
                    self.log("ERROR: No se pudo importar `proportional_hazard_test` desde `lifelines.statistics`.", "ERROR")
                    model_data_rm["schoenfeld_status_message"] = model_data_rm["schoenfeld_status_message"].replace(".","") + " (Error al importar proportional_hazard_test)."
                except Exception as e_ph_test:
                    self.log(f"ERROR: Excepción durante la llamada a `proportional_hazard_test`: {e_ph_test}", "ERROR")
                    # self.log(traceback.format_exc(), "DEBUG")
                    model_data_rm["schoenfeld_status_message"] = model_data_rm["schoenfeld_status_message"].replace(".","") + f" (Error al ejecutar proportional_hazard_test: {str(e_ph_test)[:50]}...)."

            # C-Index CV
            if self.calculate_cv_cindex_var.get() and not X_design_rm.empty:
                try:
                    kf_cv = KFold(n_splits=self.cv_num_kfolds_var.get(), shuffle=True, random_state=self.cv_random_seed_var.get())
                    c_indices_cv_list = []
                    for train_idx, test_idx in kf_cv.split(df_lifelines_rm): # Usar df_lifelines_rm para split
                        
                        df_fold_for_fit_cv = df_lifelines_rm.iloc[train_idx].copy() # Siempre usar el DF original
                        
                        # X_te_cv ya no se necesita para fit, solo para predict_partial_hazard
                        # y_te_cv se usa para concordance_index
                        y_te_cv = y_survival_rm.iloc[test_idx]

                        if df_fold_for_fit_cv.empty or y_te_cv.empty: continue
                        
                        cph_fold = CoxPHFitter(penalizer=penalizer_val_rm, l1_ratio=l1_ratio_val_rm)
                        # Usar la misma fórmula que el modelo principal
                        cph_fold.fit(df_fold_for_fit_cv, duration_col=time_col_rm, event_col=event_col_rm, formula=actual_formula_for_fit)
                        
                        # Para predict_partial_hazard, lifelines también espera el DataFrame original
                        preds_te_fold = cph_fold.predict_partial_hazard(df_lifelines_rm.iloc[test_idx])
                        c_idx_fold = concordance_index(y_te_cv[time_col_rm], -preds_te_fold, y_te_cv[event_col_rm])
                        c_indices_cv_list.append(c_idx_fold)

                    if c_indices_cv_list: model_data_rm["c_index_cv_mean"] = np.mean(c_indices_cv_list); model_data_rm["c_index_cv_std"] = np.std(c_indices_cv_list)
                    self.log(f"C-Index CV: Media={model_data_rm['c_index_cv_mean']:.3f} (DE={model_data_rm['c_index_cv_std']:.3f})", "INFO")
                except Exception as e_cv_rm: self.log(f"Error C-Index CV: {e_cv_rm}", "ERROR"); traceback.print_exc(limit=3)
            elif self.calculate_cv_cindex_var.get(): self.log("C-Index CV no calculado (modelo nulo o sin X_design).", "INFO")

        except Exception as e_fit_main:
            self.log(f"Error ajuste modelo/métricas '{model_name_rm}': {e_fit_main}", "ERROR"); traceback.print_exc(limit=5); return None

        model_data_rm["_df_for_fit_main_INTERNAL_USE"] = df_lifelines_rm.copy()
        model_data_rm["_X_design_rm_INTERNAL_USE"] = X_design_rm.copy() 
        model_data_rm["_y_survival_rm_INTERNAL_USE"] = y_survival_rm.copy() 

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
        temp_models_list_orch = [] 

        prep_res = self._preparar_datos_para_modelado()
        if prep_res is None:
            self.log("Falló preparación de datos. Abortando.", "ERROR"); self.log("*"*35 + " FIN MODELADO (ERRORES) " + "*"*35, "HEADER")
            return
        
        (df_init_full, X_init_full, y_init_data, formula_init_patsy_full, terms_init_display, t_col_final, e_col_final) = prep_res

        if df_init_full is None or df_init_full.empty:
            self.log("DF inicial vacío post-preparación. Abortando.", "ERROR"); self.log("*"*35 + " FIN MODELADO (ERRORES) " + "*"*35, "HEADER")
            return

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
                    df_uni_f, X_uni_d, formula_uni_patsy, terms_uni = self.build_design_matrix(df_init_full, [orig_cov_uni], t_col_final, e_col_final)
                    if X_uni_d is None or df_uni_f is None or df_uni_f.empty: self.log(f"Fallo build_design_matrix para '{orig_cov_uni}'.", "WARN"); continue
                    
                    y_uni_s = df_uni_f[[t_col_final, e_col_final]]
                    if X_uni_d.empty and not terms_uni: 
                        self.log(f"X_design vacía para '{orig_cov_uni}' (modelo nulo para esta var).", "WARN")
                        continue 
                    
                    name_uni = f"Univariado: {orig_cov_uni}" + (f" (Términos: {', '.join(terms_uni)})" if terms_uni != [orig_cov_uni] and terms_uni else "")
                    md_uni = self._run_model_and_get_metrics(df_uni_f, X_uni_d, y_uni_s, t_col_final, e_col_final, 
                                                             formula_uni_patsy, name_uni, terms_uni, formula_uni_patsy,
                                                             pen_val, l1_r, model_type_for_fit_logic="Univariado")
                    if md_uni: temp_models_list_orch.append(md_uni)
        
        elif model_type_ui == "Multivariado":
            self.log("Iniciando modelado Multivariado...", "INFO")
            df_multi_current = df_init_full 
            X_multi_current = X_init_full
            formula_multi_current = formula_init_patsy_full 
            terms_multi_current = terms_init_display 

            sel_meth_ui = self.var_selection_method_var.get(); suffix_multi = " (Todas las Variables)"
            if sel_meth_ui != "Ninguno (usar todas)":
                if X_init_full is None or X_init_full.empty:
                    self.log("X_design inicial vacío. No se puede seleccionar variables.", "WARN")
                    # Si no hay covariables iniciales, el modelo multivariado con selección será nulo
                    selected_orig_covs_after_selection = []
                else:
                    self.log(f"Selección de variables: {sel_meth_ui}", "INFO")
                    # _perform_variable_selection ahora devuelve solo la lista de nombres de covariables originales
                    selected_orig_covs_after_selection = self._perform_variable_selection(
                        df_init_full, X_init_full, t_col_final, e_col_final,
                        formula_init_patsy_full, terms_init_display
                    )
                    if selected_orig_covs_after_selection is None: # Fallo en selección
                        self.log("Fallo en selección de variables. Abortando modelado multivariado.", "ERROR")
                        self.log("*"*35 + " FIN MODELADO (ERRORES) " + "*"*35, "HEADER")
                        return
                    
                # Reconstruir X_design y formula_patsy con las covariables seleccionadas
                # df_init_full es el DataFrame original alineado y limpio
                df_multi_current, X_multi_current, formula_multi_current, terms_multi_current = \
                    self.build_design_matrix(df_init_full, selected_orig_covs_after_selection, t_col_final, e_col_final)
                
                if X_multi_current is None or df_multi_current is None or df_multi_current.empty:
                    self.log("Fallo al reconstruir matriz de diseño después de selección de variables. Abortando.", "ERROR")
                    self.log("*"*35 + " FIN MODELADO (ERRORES) " + "*"*35, "HEADER")
                    return

                suffix_multi = f" ({sel_meth_ui})"
            
            # Si no se hizo selección, o si la selección resultó en un modelo nulo, usar los iniciales
            if X_multi_current is None or df_multi_current is None or df_multi_current.empty:
                df_multi_current = df_init_full
                X_multi_current = X_init_full
                formula_multi_current = formula_init_patsy_full
                terms_multi_current = terms_init_display
                if sel_meth_ui != "Ninguno (usar todas)": # Si se intentó selección pero falló o resultó nula
                    suffix_multi += " (Nulo/Fallo Selección)"
                else: # Si no se intentó selección
                    suffix_multi = " (Todas las Variables)"

            y_multi = df_multi_current[[t_col_final, e_col_final]]
            
            if X_multi_current.empty and not terms_multi_current: suffix_multi += " (Nulo)"
            name_multi = f"Multivariado{suffix_multi}"
            
            md_multi = self._run_model_and_get_metrics(df_multi_current, X_multi_current, y_multi,
                                                       t_col_final, e_col_final, formula_multi_current,
                                                       name_multi, terms_multi_current, formula_init_patsy_full,
                                                       pen_val, l1_r, model_type_for_fit_logic="Multivariado")
            if md_multi: temp_models_list_orch.append(md_multi)
 
        # Añadir los modelos generados a la lista existente, no sobrescribir
        self.generated_models_data.extend(temp_models_list_orch)
        self._update_models_treeview()
        msg_fin = f"Modelado completado. {len(temp_models_list_orch)} modelo(s) generado(s) y añadido(s)." if temp_models_list_orch else "No se generó ningún modelo nuevo."
        self.log(msg_fin, "SUCCESS" if temp_models_list_orch else "WARN")
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
            if hasattr(self, '_active_figure_for_options'): 
                 del self._active_figure_for_options 
        else: self.log("No hay figura activa para opciones.", "WARN")


    def show_schoenfeld(self):
        if not self._check_model_selected_and_valid(check_params=True): return
        md_sch = self.selected_model_in_treeview
        cph_sch = md_sch.get('model')
        name_sch = md_sch.get('model_name', 'N/A')
        schoenfeld_status_msg = md_sch.get("schoenfeld_status_message", "Estado del test estadístico de Schoenfeld no especificado o test no aplicable.")
        
        self.log(f"Estado del test estadístico de Schoenfeld para '{name_sch}': {schoenfeld_status_msg}", "INFO")
        
        # df_for_schoenfeld is still needed if we compute residuals directly from the fitter
        df_for_schoenfeld = md_sch.get('_df_for_fit_main_INTERNAL_USE')
        if df_for_schoenfeld is None or df_for_schoenfeld.empty:
            self.log(f"DataFrame de ajuste ('_df_for_fit_main_INTERNAL_USE') no disponible o vacío para modelo '{name_sch}'. No se puede graficar Schoenfeld.", "ERROR")
            messagebox.showerror("Error Datos", "Datos de ajuste para gráfico Schoenfeld no disponibles en el modelo.", parent=self.parent_for_dialogs)
            return

        if not hasattr(cph_sch, 'params_') or cph_sch.params_ is None or cph_sch.params_.empty:
            self.log(f"Modelo '{name_sch}' no tiene parámetros (covariables). Gráfico de Schoenfeld no aplicable.", "INFO")
            messagebox.showinfo("No Aplicable", "Modelo no tiene covariables para mostrar gráfico de Schoenfeld.", parent=self.parent_for_dialogs)
            return

        self.log(f"Generando gráfico de residuos de Schoenfeld escalados para '{name_sch}' manualmente...", "INFO")
        fig_s = None 
        try:
            # Compute scaled Schoenfeld residuals.
            # This assumes cph_sch (the fitter object) knows the dataframe it was fitted on if training_df is not provided.
            # For lifelines, CoxPHFitter stores the training_df if it was passed to fit() directly.
            # If fit was called with formula and data separately, it constructs design matrix internally.
            # compute_residuals should ideally use the same data context as fit.
            # The previous version explicitly passed training_df=df_for_schoenfeld.
            # The subtask asks to remove it, relying on the fitter's internal state.
            scaled_residuals = cph_sch.compute_residuals(training_dataframe=df_for_schoenfeld, kind='scaled_schoenfeld')

            if scaled_residuals.empty:
                self.log(f"Residuos de Schoenfeld escalados vacíos para '{name_sch}'.", "WARN")
                messagebox.showwarning("Gráfico No Disponible", 
                                       "No se pudieron calcular los residuos de Schoenfeld escalados (DataFrame vacío).",
                                       parent=self.parent_for_dialogs)
                return

            covariate_names = scaled_residuals.columns
            num_params_sch = len(covariate_names)

            if num_params_sch == 0: 
                self.log(f"No hay covariables en los residuos de Schoenfeld para graficar para '{name_sch}'.", "INFO")
                messagebox.showinfo("Info", "No hay covariables en los residuos de Schoenfeld para graficar.", parent=self.parent_for_dialogs)
                return

            ncols_s = min(2, num_params_sch)
            nrows_s = math.ceil(num_params_sch / ncols_s)
            
            fig_s, axes_s_flat_tuple = plt.subplots(nrows_s, ncols_s, 
                                             figsize=(12 if ncols_s > 1 else 7, 4 * nrows_s), 
                                             sharex=True, squeeze=False)
            axes_s_flat = axes_s_flat_tuple.flatten()

            for idx, cov_name_s in enumerate(covariate_names):
                if idx < len(axes_s_flat):
                    ax_s_curr = axes_s_flat[idx]
                    ax_s_curr.plot(scaled_residuals.index, scaled_residuals[cov_name_s], 
                                   linestyle='none', marker='o', markersize=3, alpha=0.6)
                    ax_s_curr.axhline(0, color='grey', linestyle='--', lw=0.8)
                    ax_s_curr.set_title(f"Schoenfeld: {cov_name_s}", fontsize=10)
                    ax_s_curr.set_ylabel("Scaled Residual", fontsize=8)
                    
                    # Determine if the current subplot is in the bottom row of visible plots
                    is_in_last_visible_row = False
                    if nrows_s == 1: # Only one row
                        is_in_last_visible_row = True
                    elif (idx // ncols_s) == (nrows_s - 1): # It's in the actual last row
                        is_in_last_visible_row = True
                    elif (idx // ncols_s) == (nrows_s - 2) and (idx + ncols_s >= num_params_sch) : # It's in row above last, and last row is incomplete
                        is_in_last_visible_row = True
                        
                    if is_in_last_visible_row:
                         ax_s_curr.set_xlabel("Time", fontsize=8)
            
            for i_empty_s in range(num_params_sch, len(axes_s_flat)):
                axes_s_flat[i_empty_s].set_visible(False)

            fig_s.suptitle(f"Scaled Schoenfeld Residuals ({name_sch})", fontsize=14)
            plt.tight_layout(rect=[0, 0, 1, 0.96]) 
            
            self._create_plot_window(fig_s, f"Schoenfeld: {name_sch}", is_single_plot=True)

        except Exception as e_plot:
            self.log(f"Error al generar gráfico manual de residuos de Schoenfeld para '{name_sch}': {e_plot}", "ERROR")
            traceback.print_exc(limit=3)
            if fig_s is not None: 
                plt.close(fig_s) 
            messagebox.showerror("Error de Gráfico", 
                               f"No se pudo generar el gráfico de residuos de Schoenfeld:\n{e_plot}",
                               parent=self.parent_for_dialogs)

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
            ax_fp.errorbar(hrs_fp, y_pos_fp, xerr=[hrs_fp-low_ci_fp, upp_ci_fp-hrs_fp], fmt='o', capsize=5, color='k', ms=5, elinewidth=1.2)
            ax_fp.set_yticks(y_pos_fp); ax_fp.set_yticklabels(plot_df_fp.index); ax_fp.invert_yaxis()
            ax_fp.axvline(1.0, color='gray', ls='--', lw=0.8)
            
            opts_fp = self.current_plot_options.copy()
            opts_fp['title'] = opts_fp.get('title') or f"Forest Plot HRs ({name_fp})"
            opts_fp['xlabel'] = opts_fp.get('xlabel') or "Hazard Ratio (HR) con IC 95%"
            apply_plot_options(ax_fp, opts_fp, self.log); plt.tight_layout()
            self._create_plot_window(fig_fp, f"Forest Plot: {name_fp}")
        except Exception as e_fp: self.log(f"Error Forest Plot '{name_fp}': {e_fp}", "ERROR"); traceback.print_exc(limit=3); messagebox.showerror("Error Gráfico", f"Error Forest Plot:\n{e_fp}", parent=self.parent_for_dialogs)

    def realizar_prediccion(self):
        if not self._check_model_selected_and_valid(): return 
        md_pred = self.selected_model_in_treeview; name_pred = md_pred.get('model_name', 'N/A')
        
        full_patsy_formula = md_pred.get("full_patsy_formula_for_new_data_transform")
        if not full_patsy_formula:
            self.log("Fórmula de Patsy original no encontrada en modelo. Predicción puede ser limitada o fallar.", "WARN")
            messagebox.showwarning("Predicción", "Fórmula de Patsy original no encontrada. Se intentará con la información disponible.", parent=self.parent_for_dialogs)
            if not md_pred.get('covariates_processed', []): 
                 orig_vars_ask_pred = []
            else: 
                messagebox.showerror("Error Predicción", "No se puede determinar qué variables originales se necesitan para la predicción sin la fórmula de Patsy completa.", parent=self.parent_for_dialogs)
                return
        else: 
            orig_vars_ask_pred = sorted(list(set(re.findall(r"Q\('([^']+)'\)", full_patsy_formula))))

        if not orig_vars_ask_pred and md_pred.get('covariates_processed', []):
            self.log("No se pudieron determinar variables originales de Q() en fórmula, pero hay covariables procesadas. UI de predicción puede ser incompleta.", "WARN")

        pred_diag = Toplevel(self.parent_for_dialogs); pred_diag.title(f"Predicción: {name_pred}"); pred_diag.transient(self.parent_for_dialogs)
        entries_pred = {}; frame_main_pred_diag = ttk.Frame(pred_diag, padding=10); frame_main_pred_diag.pack(fill=tk.BOTH, expand=True)
        
        if orig_vars_ask_pred:
            ttk.Label(frame_main_pred_diag, text="Valores para covariables originales:", font=("TkDefaultFont",10,"bold")).pack(pady=(0,10),anchor='w')
            frame_vars_pred = ttk.Frame(frame_main_pred_diag); frame_vars_pred.pack(fill=tk.X, pady=5)
            for i, var_n in enumerate(orig_vars_ask_pred):
                ttk.Label(frame_vars_pred, text=f"{var_n}:").grid(row=i,column=0,padx=5,pady=3,sticky=tk.E)
                svar_pred = StringVar(); entries_pred[var_n] = svar_pred
                if self.data is not None and var_n in self.data:
                    try: svar_pred.set(f"{self.data[var_n].mean():.2f}" if pd.api.types.is_numeric_dtype(self.data[var_n]) else str(self.data[var_n].mode(dropna=True)[0]))
                    except: pass
                ttk.Entry(frame_vars_pred, textvariable=svar_pred, width=25).grid(row=i,column=1,padx=5,pady=3,sticky=tk.EW)
            frame_vars_pred.columnconfigure(1,weight=1)
        else:
            ttk.Label(frame_main_pred_diag, text="Modelo nulo o sin covariables originales identificables por Q(). Se predecirá línea base.", font=("TkDefaultFont",10,"italic")).pack(pady=(0,10),anchor='w')

        
        frame_opts_pred = ttk.Frame(frame_main_pred_diag); frame_opts_pred.pack(fill=tk.X,pady=10)
        
        ttk.Label(frame_opts_pred,text="Tipo Predicción:").grid(row=0,column=0,padx=5,pady=3,sticky=tk.W)
        type_var_pred_ui = StringVar(value="Supervivencia")
        ttk.Radiobutton(frame_opts_pred,text="Prob.Supervivencia",variable=type_var_pred_ui,value="Supervivencia").grid(row=0,column=1,padx=5,pady=3,sticky=tk.W)
        ttk.Radiobutton(frame_opts_pred,text="Riesgo Acumulado",variable=type_var_pred_ui,value="Riesgo").grid(row=0,column=2,padx=5,pady=3,sticky=tk.W)
        ttk.Radiobutton(frame_opts_pred,text="Prob. Evento Acum. (1-S(t))",variable=type_var_pred_ui,value="ProbEventoAcum").grid(row=0,column=3,padx=5,pady=3,sticky=tk.W)
        ttk.Radiobutton(frame_opts_pred,text="Densidad Evento (f(t|x))",variable=type_var_pred_ui,value="DensidadEvento").grid(row=0,column=4,padx=5,pady=3,sticky=tk.W)
        
        ttk.Label(frame_opts_pred,text="Tiempo(s) (ej: 100 o 50,100):").grid(row=1,column=0,padx=5,pady=3,sticky=tk.W)
        times_str_var_pred_ui = StringVar(value="")
        if self.data is not None and md_pred.get('time_col_for_model') in self.data:
            try:
                median_time = self.data[md_pred.get('time_col_for_model')].median()
                if pd.notna(median_time):
                    times_str_var_pred_ui.set(f"{median_time:.1f}")
            except: pass
        ttk.Entry(frame_opts_pred,textvariable=times_str_var_pred_ui,width=30).grid(row=1,column=1,columnspan=3,padx=5,pady=3,sticky=tk.EW)
        
        frame_btns_pred_diag = ttk.Frame(frame_main_pred_diag,padding=(0,10,0,0)); frame_btns_pred_diag.pack(fill=tk.X)
        ttk.Button(frame_btns_pred_diag,text="Predecir y Mostrar Curva",command=lambda: self._perform_prediction_and_plot(pred_diag,md_pred,entries_pred,type_var_pred_ui.get(),times_str_var_pred_ui.get())).pack(side=tk.LEFT,padx=10)
        ttk.Button(frame_btns_pred_diag,text="Cancelar",command=pred_diag.destroy).pack(side=tk.RIGHT,padx=10)

    def _perform_prediction_and_plot(self, dialog_pred_ref, md_dict_for_pred, entries_dict_for_pred, type_ui_pred, times_str_ui_pred):
        cph_model_for_pred = md_dict_for_pred.get('model'); name_for_pred = md_dict_for_pred.get('model_name', 'N/A')
        times_list_pred = []
        if times_str_ui_pred.strip():
            try:
                times_list_pred = [float(t.strip()) for t in times_str_ui_pred.split(',') if t.strip()]
                if any(t < 0 for t in times_list_pred): raise ValueError("Tiempos negativos no permitidos.")
                times_list_pred = sorted(list(set(times_list_pred)))
            except ValueError:
                messagebox.showerror("Error Tiempos","Tiempos inválidos. Ingrese números separados por comas o déjelo vacío para la curva completa.",parent=dialog_pred_ref); return
        
        input_data_dict_pred = {}
        for var_k, svar_obj in entries_dict_for_pred.items():
            val_entry = svar_obj.get().strip()
            if not val_entry: messagebox.showerror("Valor Faltante",f"Valor faltante para '{var_k}'.",parent=dialog_pred_ref); return
            try: input_data_dict_pred[var_k] = float(val_entry)
            except ValueError: input_data_dict_pred[var_k] = str(val_entry)
        
        df_patsy_input_pred = pd.DataFrame([input_data_dict_pred]) if input_data_dict_pred else pd.DataFrame([{}])


        try:
            full_formula_for_transform = md_dict_for_pred.get("full_patsy_formula_for_new_data_transform")
            final_model_terms = md_dict_for_pred.get('covariates_processed', [])

            X_patsy_pred_final: pd.DataFrame

            if not final_model_terms:
                X_patsy_pred_final = dmatrix("0", df_patsy_input_pred, return_type="dataframe")
            elif not full_formula_for_transform:
                self.log("Error crítico: Fórmula completa de Patsy no disponible para transformar datos para predicción.", "ERROR")
                messagebox.showerror("Error Predicción", "No se pudo determinar la fórmula de Patsy para transformar nuevos datos.", parent=dialog_pred_ref)
                return
            else:
                X_temp_full_design = dmatrix(full_formula_for_transform, df_patsy_input_pred, return_type="dataframe")
                
                if set(final_model_terms).issubset(set(X_temp_full_design.columns)):
                    X_patsy_pred_final = X_temp_full_design[final_model_terms]
                else:
                    missing_terms = set(final_model_terms) - set(X_temp_full_design.columns)
                    self.log(f"Error: Términos del modelo {missing_terms} no encontrados en X transformada para predicción.", "ERROR")
                    messagebox.showerror("Error Predicción", f"Discrepancia en términos para predicción. Faltan: {missing_terms}", parent=dialog_pred_ref)
                    return
        except Exception as e_patsy_pred_final:
            self.log(f"Error Patsy en predicción: {e_patsy_pred_final}","ERROR"); traceback.print_exc(limit=3);
            messagebox.showerror("Error Patsy Pred.","Error transformando entradas para predicción.",parent=dialog_pred_ref); return

        try:
            fig_curve_pred, ax_curve_pred = plt.subplots(figsize=(10,6)); results_text_pred = []
            pred_col_name_for_times = None # To store the name of the main prediction column for point predictions

            if type_ui_pred == "Supervivencia":
                title_curve_pred = f"Pred. Prob. Supervivencia ({name_for_pred})"
                label_prefix = "S"
                ax_curve_pred.set_ylabel("S(t|X)")

                try:
                    # Request confidence intervals
                    surv_func_and_ci_df = cph_model_for_pred.predict_survival_function(
                        df_patsy_input_pred, return_confidence_intervals=True
                    )

                    # Dynamically find column names for a single prediction scenario
                    # Assuming df_patsy_input_pred represents one individual, so one set of pred/CI cols
                    base_col_name = str(df_patsy_input_pred.index[0]) # Usually '0' if not otherwise specified

                    pred_col_name = f"{base_col_name}_pred"
                    lower_ci_col_name = f"{base_col_name}_lower_0.95"
                    upper_ci_col_name = f"{base_col_name}_upper_0.95"

                    if pred_col_name in surv_func_and_ci_df.columns and \
                       lower_ci_col_name in surv_func_and_ci_df.columns and \
                       upper_ci_col_name in surv_func_and_ci_df.columns:

                        ax_curve_pred.plot(surv_func_and_ci_df.index, surv_func_and_ci_df[pred_col_name], label='S(t|X) Estimada')
                        ax_curve_pred.fill_between(surv_func_and_ci_df.index,
                                                   surv_func_and_ci_df[lower_ci_col_name],
                                                   surv_func_and_ci_df[upper_ci_col_name],
                                                   alpha=0.2, label='IC 95%')
                        ax_curve_pred.legend()
                        pred_col_name_for_times = pred_col_name # Save for point predictions
                        # For interp, use the main prediction series from surv_func_and_ci_df
                        pred_df_for_times = surv_func_and_ci_df[[pred_col_name_for_times]]
                        self.log("Gráfico de supervivencia individual con IC 95% generado.", "INFO")
                    else: # Fallback if CI columns are not as expected
                        self.log(f"Advertencia: Columnas de IC no encontradas ({pred_col_name}, {lower_ci_col_name}, {upper_ci_col_name}). "
                                   f"Graficando supervivencia sin IC. Columnas disponibles: {surv_func_and_ci_df.columns.tolist()}", "WARN")
                        # Plot just the first column as prediction if CI parts are missing
                        # This assumes the first column is the prediction if CI parts are missing
                        pred_df = surv_func_and_ci_df.iloc[:, [0]]
                        pred_df.plot(ax=ax_curve_pred, legend=False, label='S(t|X) Estimada (sin IC)')
                        ax_curve_pred.legend()
                        pred_col_name_for_times = pred_df.columns[0]
                        pred_df_for_times = pred_df # Use this for interp
                except Exception as e_ci_plot:
                    self.log(f"Error al intentar graficar supervivencia con IC: {e_ci_plot}. Fallback a graficar sin IC.", "ERROR")
                    traceback.print_exc(limit=2)
                    # Fallback to original plotting method if CI method fails
                    pred_df = cph_model_for_pred.predict_survival_function(df_patsy_input_pred)
                    pred_df.plot(ax=ax_curve_pred, legend=False, label='S(t|X) Estimada (sin IC - error)')
                    ax_curve_pred.legend()
                    pred_col_name_for_times = pred_df.columns[0] if not pred_df.empty else None
                    pred_df_for_times = pred_df

            elif type_ui_pred == "Riesgo":
                pred_df = cph_model_for_pred.predict_cumulative_hazard(df_patsy_input_pred)
                pred_df.plot(ax=ax_curve_pred, legend=False)
                ax_curve_pred.set_ylabel("H(t|X)")
                title_curve_pred = f"Pred. Riesgo Acumulado ({name_for_pred})"
                label_prefix = "H"
                pred_col_name_for_times = pred_df.columns[0] if not pred_df.empty else None
                pred_df_for_times = pred_df

            elif type_ui_pred == "ProbEventoAcum":
                surv_df_temp = cph_model_for_pred.predict_survival_function(df_patsy_input_pred)
                pred_df = 1 - surv_df_temp
                pred_df.plot(ax=ax_curve_pred, legend=False)
                ax_curve_pred.set_ylabel("1 - S(t|X)")
                title_curve_pred = f"Pred. Prob. Evento Acumulado (1-S(t)) ({name_for_pred})"
                label_prefix = "1-S"
                pred_col_name_for_times = pred_df.columns[0] if not pred_df.empty else None
                pred_df_for_times = pred_df
            
            elif type_ui_pred == "DensidadEvento":
                title_curve_pred = f"Pred. Densidad de Evento Individual f(t|x) ({name_for_pred})"
                label_prefix = "f"
                ax_curve_pred.set_ylabel("f(t|X) - Densidad de Probabilidad de Evento")

                survival_df_ftx = cph_model_for_pred.predict_survival_function(df_patsy_input_pred)
                cum_hazard_df_ftx = cph_model_for_pred.predict_cumulative_hazard(df_patsy_input_pred)

                if survival_df_ftx.empty or cum_hazard_df_ftx.empty or len(survival_df_ftx.index) < 2:
                    self.log(f"Datos insuficientes para calcular f(t|x) para '{name_for_pred}'.", "ERROR")
                    messagebox.showerror("Error de Cálculo", "No hay suficientes datos para calcular la densidad del evento.", parent=dialog_pred_ref)
                    plt.close(fig_curve_pred)
                    return

                s_tx = survival_df_ftx.iloc[:, 0]
                H_tx = cum_hazard_df_ftx.iloc[:, 0]
                time_points_ftx = survival_df_ftx.index.to_numpy()

                dt_ftx = np.diff(time_points_ftx)
                dH_tx = np.diff(H_tx.to_numpy())

                valid_dt_mask = dt_ftx > 1e-8 # Use a small epsilon to avoid issues with float precision

                time_points_for_h_f = time_points_ftx[1:][valid_dt_mask]
                dH_tx_valid = dH_tx[valid_dt_mask]
                dt_ftx_valid = dt_ftx[valid_dt_mask]

                if len(time_points_for_h_f) < 1:
                    self.log(f"No hay suficientes puntos de tiempo válidos (dt > 0) para calcular h(t|x) para f(t|x) en '{name_for_pred}'.", "ERROR")
                    messagebox.showerror("Error de Cálculo", "Intervalos de tiempo insuficientes para calcular la tasa de riesgo instantánea.", parent=dialog_pred_ref)
                    plt.close(fig_curve_pred)
                    return

                h_tx_raw = dH_tx_valid / dt_ftx_valid

                # Align s_tx with h_tx_raw and time_points_for_h_f
                # s_tx corresponds to time_points_ftx. We need s_tx at time_points_for_h_f for the multiplication.
                # Interpolation is safer if time_points_for_h_f is not perfectly time_points_ftx[1:] due to valid_dt_mask
                s_tx_aligned_for_f = np.interp(time_points_for_h_f, time_points_ftx, s_tx.to_numpy())

                f_tx_raw = h_tx_raw * s_tx_aligned_for_f
                pred_df_for_times = pd.DataFrame({f'{df_patsy_input_pred.index[0]}_pred': f_tx_raw}, index=time_points_for_h_f)
                pred_col_name_for_times = pred_df_for_times.columns[0]


                ax_curve_pred.plot(time_points_for_h_f, f_tx_raw, label='f(t|x) Bruta Estimada', alpha=0.7, linestyle='--')

                if len(f_tx_raw) > 5: # LOESS needs a minimum number of points
                    try:
                        # Ensure time points are strictly increasing for LOESS if it requires it (some versions might)
                        # However, lowess from statsmodels should handle sorted x by default.
                        # frac is the fraction of data used for smoothing each point.
                        # it is the number of robustifying iterations.
                        f_tx_smoothed = lowess(f_tx_raw, time_points_for_h_f, frac=0.3, it=1, return_sorted=False)
                        ax_curve_pred.plot(time_points_for_h_f, f_tx_smoothed, label='f(t|x) Suavizada (LOESS)', color='green')
                        # For point predictions, use smoothed data if available
                        pred_df_for_times = pd.DataFrame({f'{df_patsy_input_pred.index[0]}_pred': f_tx_smoothed}, index=time_points_for_h_f)
                        pred_col_name_for_times = pred_df_for_times.columns[0]
                        self.log(f"LOESS smoothing aplicado a f(t|x) para '{name_for_pred}'.", "INFO")
                    except Exception as e_loess:
                        self.log(f"Error durante LOESS smoothing para f(t|x) en '{name_for_pred}': {e_loess}. Mostrando solo datos brutos.", "ERROR")
                        traceback.print_exc(limit=2)
                else:
                    self.log(f"No hay suficientes puntos ({len(f_tx_raw)}) para LOESS smoothing en f(t|x) para '{name_for_pred}'.", "WARN")
                ax_curve_pred.legend()


            # Logic for specific time point predictions
            if times_list_pred and pred_col_name_for_times and pred_df_for_times is not None and not pred_df_for_times.empty:
                current_legend_items = len(ax_curve_pred.get_legend_handles_labels()[0]) if ax_curve_pred.get_legend() else 0
                show_time_legend = (current_legend_items == 0) # Show legend only if no other legend items from curve itself

                for t_val in times_list_pred:
                    if t_val < pred_df_for_times.index.min() or t_val > pred_df_for_times.index.max():
                        results_text_pred.append(f"{label_prefix}(t={t_val}|X) = N/A (fuera de rango)");
                        self.log(f"Advertencia: Tiempo de predicción {t_val} fuera del rango de la curva.", "WARN")
                    else:
                        # Use the specifically identified prediction column for interpolation
                        val_plot = np.interp(t_val, pred_df_for_times.index, pred_df_for_times[pred_col_name_for_times])
                        results_text_pred.append(f"{label_prefix}(t={t_val}|X) = {val_plot:.3f}");
                        ax_curve_pred.scatter([t_val],[val_plot],marker='o',color='r',s=50,zorder=5,
                                              label=f'Pred. en t={t_val}' if show_time_legend and t_val==times_list_pred[0] else None)
                if show_time_legend and any(l is not None for l in ax_curve_pred.get_legend_handles_labels()[1]): # if any label was set for scatter
                     ax_curve_pred.legend()
            elif times_list_pred :
                 results_text_pred.append("No se pudieron obtener predicciones en tiempos específicos (columna de predicción no identificada o datos vacíos).")


            opts_curve_pred = self.current_plot_options.copy()
            opts_curve_pred['title'] = opts_curve_pred.get('title') or title_curve_pred
            opts_curve_pred['xlabel'] = opts_curve_pred.get('xlabel') or f"Tiempo ({md_dict_for_pred.get('time_col_for_model','T')})"
            apply_plot_options(ax_curve_pred, opts_curve_pred, self.log)
             # Ensure legend is present if any labels were added
            if not ax_curve_pred.get_legend() and any(l is not None for l in ax_curve_pred.get_legend_handles_labels()[1]):
                ax_curve_pred.legend()

            self._create_plot_window(fig_curve_pred, title_curve_pred)
            
            if results_text_pred: # Show results/info whether it's point predictions or just "curve shown"
                messagebox.showinfo("Resultados Predicción", "Resultados en tiempos especificados:\n" + "\n".join(results_text_pred), parent=dialog_pred_ref)
            else:
                messagebox.showinfo("Resultados Predicción", "Curva de predicción completa generada.", parent=dialog_pred_ref)
        except Exception as e_curve_pred: self.log(f"Error pred/plot: {e_curve_pred}","ERROR"); traceback.print_exc(limit=3); messagebox.showerror("Error Pred/Plot",f"Error al predecir/plotear:\n{e_curve_pred}",parent=dialog_pred_ref)


    def generate_calibration_plot(self):
        if not self._check_model_selected_and_valid(): return
        if not LIFELINES_CALIBRATION_AVAILABLE: messagebox.showwarning("No Disponible","Gráf. Calibración no disponible.",parent=self.parent_for_dialogs); return
        md_cal = self.selected_model_in_treeview; cph_cal = md_cal.get('model'); name_cal = md_cal.get('model_name','N/A')
        
        X_dsgn_cal = md_cal.get('_X_design_rm_INTERNAL_USE')
        y_surv_cal_dict = md_cal.get('_y_survival_rm_INTERNAL_USE') 
        
        if X_dsgn_cal is None or y_surv_cal_dict is None:
            self.log(f"Datos (X o y) para calibración no disponibles en modelo '{name_cal}'.", "ERROR")
            messagebox.showerror("Error Datos","Faltan datos X_design o y_survival para calibración en el modelo guardado.",parent=self.parent_for_dialogs); return
        if X_dsgn_cal.empty and md_cal.get('covariates_processed',[]): 
            self.log(f"X_design para calibración está vacía pero el modelo tiene covariables ('{name_cal}').", "WARN")
            messagebox.showwarning("Datos Inconsistentes","X_design para calibración está vacía. Gráfico puede no ser significativo.", parent=self.parent_for_dialogs)

        t_col_cal, e_col_cal = md_cal.get('time_col_for_model'), md_cal.get('event_col_for_model')
        if not t_col_cal or not e_col_cal:
             self.log(f"Nombres de columnas T/E no encontrados en modelo '{name_cal}'.", "ERROR")
             messagebox.showerror("Error Interno", "Nombres T/E faltan en modelo.", parent=self.parent_for_dialogs); return

        T_cal_series = y_surv_cal_dict[t_col_cal]
        E_cal_series = y_surv_cal_dict[e_col_cal]

        t0_cal_str = simpledialog.askstring("Tiempo Calibración","Ingrese t0 para calibración:",parent=self.parent_for_dialogs)
        if not t0_cal_str: return
        try: t0_val_cal = float(t0_cal_str); assert t0_val_cal > 0
        except: messagebox.showerror("Tiempo Inválido",f"Tiempo t0 inválido.",parent=self.parent_for_dialogs); return
        try:
            fig_cal, ax_cal = plt.subplots(figsize=(8,8))
            if X_dsgn_cal.empty and not md_cal.get('covariates_processed',[]): 
                 self.log("Calibración no es aplicable a un modelo nulo sin covariables.", "INFO")
                 messagebox.showinfo("No Aplicable", "Gráfico de calibración no aplica a modelos nulos.", parent=self.parent_for_dialogs)
                 plt.close(fig_cal) 
                 return

            df_actually_used_for_fit = md_cal.get('_df_for_fit_main_INTERNAL_USE')
            if df_actually_used_for_fit is None:
                self.log(f"DataFrame used for fitting ('_df_for_fit_main_INTERNAL_USE') not found in model data for calibration model '{name_cal}'. Calibration plot cannot be generated.", "ERROR")
                messagebox.showerror("Error Calibración", f"Datos de ajuste no encontrados en el modelo '{name_cal}' para generar el gráfico de calibración.", parent=self.parent_for_dialogs)
                plt.close(fig_cal) # Close the figure if data is missing
                return

            survival_probability_calibration(cph_cal, df_actually_used_for_fit, t0=t0_val_cal, ax=ax_cal)
            opts_cal = self.current_plot_options.copy(); opts_cal['title'] = opts_cal.get('title') or f"Calibración en t0={t0_val_cal} ({name_cal})"; apply_plot_options(ax_cal, opts_cal, self.log)
            self._create_plot_window(fig_cal, f"Calibración t0={t0_val_cal}: {name_cal}")
        except Exception as e_cal:
            self.log(f"Error gráf.calibración '{name_cal}': {e_cal}","ERROR")
            import traceback # Explicitly import traceback here
            self.log(traceback.format_exc(), "DEBUG") # Log the full traceback for debugging this specific error
            messagebox.showerror("Error Gráfico",f"Error al generar gráfico de calibración para '{name_cal}':\n{e_cal}",parent=self.parent_for_dialogs)

            # Ensure figure created by plt.subplots is closed if an error occurs
            # before it's properly managed by _create_plot_window or if _create_plot_window itself fails.
            if 'fig_cal' in locals():
                try:
                    # Check if the window for this figure still exists or was ever created.
                    # This is a bit tricky as canvas might not be there if plt.subplots() itself failed,
                    # or if _create_plot_window was never reached.
                    # A simpler plt.close(fig_cal) is often sufficient.
                    plt.close(fig_cal)
                except Exception as e_close_fig:
                    self.log(f"DEBUG: Minor error trying to close fig_cal during calibration error handling: {e_close_fig}", "DEBUG")

    def show_variable_impact_plot(self):
        if not self._check_model_selected_and_valid(check_params=True):
            return

        md_vip = self.selected_model_in_treeview
        cph_model_vip = md_vip.get('model')
        model_name_vip = md_vip.get('model_name', 'N/A')

        if not hasattr(cph_model_vip, 'params_') or cph_model_vip.params_.empty:
            messagebox.showinfo("Sin Parámetros", "El modelo seleccionado no tiene covariables (parámetros) para analizar.", parent=self.parent_for_dialogs)
            return

        available_covariates = list(cph_model_vip.params_.index)
        if not available_covariates:
            messagebox.showinfo("Sin Covariables", "No se encontraron covariables en los parámetros del modelo.", parent=self.parent_for_dialogs)
            return

        # Create a simple dialog to choose the covariate
        # For simplicity, using simpledialog.askstring to list choices.
        # A more complex dialog could use a Combobox.
        choice_prompt = "Seleccione la covariable para analizar su impacto (escriba el nombre exacto):\n\n" + "\n".join(available_covariates)

        # Use a Toplevel dialog for better control if simpledialog is too basic or problematic with many vars
        dialog = Toplevel(self.parent_for_dialogs)
        dialog.title("Seleccionar Covariable")
        dialog.geometry("400x350") # Adjust size as needed

        ttk.Label(dialog, text="Seleccione la covariable para el gráfico de impacto:", wraplength=380).pack(pady=10, padx=10)

        covariate_var = StringVar()
        combo_covs_widget = None # Initialize reference
        listbox_covs_widget = None # Initialize reference

        # Populate combobox if there are few items, otherwise Listbox might be better
        if len(available_covariates) < 20: # Arbitrary threshold
            combo_covs_widget = ttk.Combobox(dialog, textvariable=covariate_var, values=available_covariates, state="readonly", width=40)
            if available_covariates:
                combo_covs_widget.set(available_covariates[0])
            combo_covs_widget.pack(pady=5, padx=10)
        else: # Use Listbox for many covariates
            ttk.Label(dialog, text="Covariables disponibles:").pack(pady=(5,0))
            listbox_frame = ttk.Frame(dialog)
            listbox_frame.pack(pady=5, padx=10, fill=tk.BOTH, expand=True)
            listbox_covs_widget = Listbox(listbox_frame, selectmode=SINGLE, exportselection=False, height=8)
            for cov_name_lb in available_covariates:
                listbox_covs_widget.insert(tk.END, cov_name_lb)
            if available_covariates:
                listbox_covs_widget.selection_set(0) # Pre-select first

            scrollbar_y_covs = ttk.Scrollbar(listbox_frame, orient=tk.VERTICAL, command=listbox_covs_widget.yview)
            listbox_covs_widget.config(yscrollcommand=scrollbar_y_covs.set)
            scrollbar_y_covs.pack(side=tk.RIGHT, fill=tk.Y)
            listbox_covs_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)


        chosen_covariate = None

        def on_ok():
            nonlocal chosen_covariate
            selected_value = None
            if listbox_covs_widget is not None and listbox_covs_widget.winfo_exists():
                if listbox_covs_widget.curselection():
                    selected_value = listbox_covs_widget.get(listbox_covs_widget.curselection()[0])
            elif combo_covs_widget is not None and combo_covs_widget.winfo_exists():
                selected_value = covariate_var.get()

            if selected_value and selected_value.strip():
                chosen_covariate = selected_value
                dialog.destroy()
            else:
                 messagebox.showwarning("Selección Requerida", "Debe seleccionar una covariable de la lista.", parent=dialog)
                 # chosen_covariate remains None

        def on_cancel():
            nonlocal chosen_covariate # Ensure chosen_covariate can be set to signal cancellation
            chosen_covariate = "##CANCELLED##" # Special value to indicate cancellation
            dialog.destroy()

        button_frame = ttk.Frame(dialog)
        button_frame.pack(pady=10)
        ttk.Button(button_frame, text="Aceptar", command=on_ok).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Cancelar", command=on_cancel).pack(side=tk.RIGHT, padx=5)

        dialog.transient(self.parent_for_dialogs)
        dialog.grab_set()
        self.parent_for_dialogs.wait_window(dialog)

        if not chosen_covariate or chosen_covariate == "##CANCELLED##":
            self.log("Selección de covariable para gráfico de impacto cancelada o no realizada.", "INFO")
            return

        # Ask user for the scale
        response = messagebox.askyesnocancel(
            "Escala del Gráfico de Impacto",
            "Presione 'Sí' para escala Hazard Ratio (HR).\n"
            "Presione 'No' para escala Log(Hazard Ratio) (Log-HR).",
            parent=self.parent_for_dialogs
        )

        if response is None: # Cancelled
            self.log("Selección de escala para gráfico de impacto cancelada.", "INFO")
            return

        plot_as_hr = response # True for HR, False for Log(HR)

        self.log(f"Generando gráfico de impacto para covariable: '{chosen_covariate}' del modelo '{model_name_vip}'. Escala: {'HR' if plot_as_hr else 'Log(HR)'}", "INFO")

        covariate_for_plot = chosen_covariate
        match = re.match(r"Q\('([^']+)'\)", chosen_covariate)
        original_name_from_q = None
        if match:
            original_name_from_q = match.group(1)
            covariate_for_plot = original_name_from_q
            self.log(f"Extracted original name '{covariate_for_plot}' from Patsy term '{chosen_covariate}' for plotting.", "DEBUG")
        else:
            self.log(f"Using covariate name '{covariate_for_plot}' directly (not a Q-encoded term).", "DEBUG")

        fig_vip = None
        ax_vip = None
        current_opts_vip = self.current_plot_options.copy()

        try:
            if plot_as_hr:
                # Get data for HR plot
                effects_df = cph_model_vip.plot_partial_effects_on_outcome(
                    covariate_for_plot,
                    values=None,
                    plot_baseline=False,
                    plot=False # Get DataFrame instead of plotting
                )
                if effects_df is None or effects_df.empty:
                    messagebox.showerror("Error de Datos", f"No se pudieron obtener los datos de efectos para '{covariate_for_plot}'.", parent=self.parent_for_dialogs)
                    self.log(f"plot_partial_effects_on_outcome devolvió None o DataFrame vacío para '{covariate_for_plot}' al intentar obtener datos para HR.", "ERROR")
                    return

                hr_estimate = np.exp(effects_df['coef'])
                hr_lower_ci = np.exp(effects_df['coef_lower_0.95'])
                hr_upper_ci = np.exp(effects_df['coef_upper_0.95'])

                fig_vip, ax_vip = plt.subplots(figsize=(10, 6))

                # Logging for main HR path
                self.log(f"DEBUG: HR Plot X-values (index): {effects_df.index}", "DEBUG")
                self.log(f"DEBUG: HR Plot Y-estimate (hr_estimate): {hr_estimate}", "DEBUG")
                self.log(f"DEBUG: HR Plot Y-CI-lower (hr_lower_ci): {hr_lower_ci}", "DEBUG")
                self.log(f"DEBUG: HR Plot Y-CI-upper (hr_upper_ci): {hr_upper_ci}", "DEBUG")

                ax_vip.plot(effects_df.index, hr_estimate, label='HR Estimado')
                ax_vip.fill_between(effects_df.index, hr_lower_ci, hr_upper_ci, alpha=0.2, label='IC 95% HR')
                ax_vip.axhline(1.0, color='grey', linestyle='--', lw=0.8)

                plot_title_str = f"Impacto de '{covariate_for_plot}' sobre Hazard Ratio (HR)\nModelo: {model_name_vip}"
                ax_vip.set_title(plot_title_str) # Set title directly
                ax_vip.set_ylabel("Hazard Ratio (HR)")
                ax_vip.set_xlabel(f"Valor de {covariate_for_plot}")
                ax_vip.legend() # Call legend before apply_plot_options

                # Update current_opts_vip for apply_plot_options, but core elements are already set
                current_opts_vip['title'] = current_opts_vip.get('title', plot_title_str) # Keep this for consistency if options dialog is used
                current_opts_vip['ylabel'] = current_opts_vip.get('ylabel', "Hazard Ratio (HR)")
                current_opts_vip['xlabel'] = current_opts_vip.get('xlabel', f"Valor de {covariate_for_plot}")

            # This is the primary plotting attempt for Log(HR) scale
            elif not plot_as_hr: # Plot as Log(HR)
                fig_vip, ax_vip = plt.subplots(figsize=(10, 6))
                cph_model_vip.plot_partial_effects_on_outcome(
                    covariate_for_plot,
                    values=None,
                    plot_baseline=False,
                    ax=ax_vip
                )
                plot_title_str = f"Impacto de '{covariate_for_plot}' sobre Log(Hazard Ratio)\nModelo: {model_name_vip}"
                ax_vip.set_title(plot_title_str)
                ax_vip.set_ylabel(f"Log(Hazard Ratio) para {covariate_for_plot}")
                ax_vip.set_xlabel(f"Valor de {covariate_for_plot}")

                current_opts_vip['title'] = current_opts_vip.get('title', plot_title_str)
                current_opts_vip['ylabel'] = current_opts_vip.get('ylabel', f"Log(Hazard Ratio) para {covariate_for_plot}")
                current_opts_vip['xlabel'] = current_opts_vip.get('xlabel', f"Valor de {covariate_for_plot}")

            # apply_plot_options with try-except wrapper (common for both HR and LogHR main paths)
            if fig_vip and ax_vip : # Ensure figure and axis were created
                try: # 16 spaces
                    opts_to_apply = current_opts_vip if isinstance(current_opts_vip, dict) else self.current_plot_options.copy() # 20 spaces
                    apply_plot_options(ax_vip, opts_to_apply, self.log) # 20 spaces
                except TypeError as e_apply_opts:
                    log_msg_prefix = f"Covariate: '{chosen_covariate}', Scale: {'HR' if plot_as_hr else 'Log(HR)'}."
                    if "got an unexpected keyword argument 'plot'" in str(e_apply_opts):
                        self.log(f"WARNING: {log_msg_prefix} apply_plot_options failed with 'plot' keyword error: {e_apply_opts}. Displaying with minimal styling.", "WARN")
                        if hasattr(ax_vip, 'grid'):
                            grid_option = opts_to_apply.get('grid', True) if 'opts_to_apply' in locals() and isinstance(opts_to_apply, dict) else True
                            if grid_option: ax_vip.grid(True, linestyle=':', alpha=0.6)
                        if hasattr(ax_vip, 'get_legend_handles_labels') and plot_as_hr:
                            handles, labels = ax_vip.get_legend_handles_labels()
                            if handles and labels and not ax_vip.get_legend():
                                ax_vip.legend()
                    else:
                        self.log(f"ERROR: {log_msg_prefix} Unexpected TypeError during apply_plot_options: {e_apply_opts}", "ERROR")
                except Exception as e_general_apply_opts:
                    self.log(f"ERROR: {log_msg_prefix} General error during apply_plot_options: {e_general_apply_opts}", "ERROR")

                plt.tight_layout()
                self._create_plot_window(fig_vip, f"Impacto Variable: {chosen_covariate} ({model_name_vip}) - Escala {'HR' if plot_as_hr else 'Log(HR)'}")

        except IndexError as e_vip_idx: # This IndexError fallback is now primarily for Log(HR) plotting issues
            tb_str_vip = traceback.format_exc()
            self.log(f"IndexError during initial plot attempt for '{chosen_covariate}' (Escala: {'HR' if plot_as_hr else 'Log(HR)'}): {e_vip_idx}", "ERROR")
            self.log(tb_str_vip, "DEBUG")

            if "tuple index out of range" in str(e_vip_idx) and "values.shape[1]" in tb_str_vip:
                self.log(f"Specific IndexError detected for '{chosen_covariate}'. Attempting fallback plotting strategy.", "WARN")

                df_fit = md_vip.get('_df_for_fit_main_INTERNAL_USE')
                if df_fit is None or df_fit.empty:
                    self.log("DataFrame de ajuste no disponible para fallback.", "ERROR")
                    messagebox.showerror("Error de Gráfico (IndexError)", f"Error (IndexError) y datos de ajuste no disponibles para '{chosen_covariate}'.", parent=self.parent_for_dialogs)
                    if fig_vip and plt.fignum_exists(fig_vip.number): plt.close(fig_vip) # Close if open
                    return

                series_for_fallback = df_fit.get(covariate_for_plot)
                if series_for_fallback is None or series_for_fallback.empty or series_for_fallback.isna().all():
                    self.log(f"Serie de datos para '{covariate_for_plot}' no encontrada o vacía en df_fit para fallback.", "ERROR")
                    messagebox.showerror("Error de Gráfico (IndexError)", f"Error (IndexError) y datos de la covariable '{covariate_for_plot}' no encontrados para fallback.", parent=self.parent_for_dialogs)
                    if fig_vip and plt.fignum_exists(fig_vip.number): plt.close(fig_vip)
                    return

                is_quantitative_fallback = self.covariables_type_config.get(original_name_from_q if match else chosen_covariate) == "Cuantitativa"
                if (original_name_from_q if match else chosen_covariate) not in self.covariables_type_config:
                    is_quantitative_fallback = pd.api.types.is_numeric_dtype(series_for_fallback.dtype)

                if is_quantitative_fallback:
                    min_val_fb, max_val_fb = series_for_fallback.min(), series_for_fallback.max()
                    if min_val_fb == max_val_fb or pd.isna(min_val_fb) or pd.isna(max_val_fb):
                        self.log(f"Rango no válido para fallback de '{covariate_for_plot}'.", "WARN")
                        messagebox.showerror("Error de Gráfico (IndexError)", f"Error (IndexError) y rango de datos inválido para fallback de '{covariate_for_plot}'.", parent=self.parent_for_dialogs)
                        if fig_vip and plt.fignum_exists(fig_vip.number): plt.close(fig_vip)
                        return

                    manual_values_1d_fb = np.linspace(min_val_fb, max_val_fb, 150)
                    manual_values_2d_fb = manual_values_1d_fb.reshape(-1, 1)

                    try:
                        self.log(f"Intentando fallback plot con valores manuales para '{covariate_for_plot}'. Escala: {'HR' if plot_as_hr else 'Log(HR)'}", "INFO")

                        if fig_vip and plt.fignum_exists(fig_vip.number): plt.close(fig_vip)
                        fig_vip, ax_vip = plt.subplots(figsize=(10, 6))

                        if plot_as_hr: # Fallback for HR scale
                            # This path should ideally not be hit often if data retrieval for HR is robustly handled before this main try-except.
                            # However, keeping it for completeness of the original fallback structure if a plotting-related IndexError occurred after successful HR data retrieval.
                            effects_df_fb = cph_model_vip.plot_partial_effects_on_outcome(
                                covariate_for_plot, values=manual_values_2d_fb, plot_baseline=False, plot=False
                            )
                            if effects_df_fb is None or effects_df_fb.empty:
                                raise ValueError("Fallback data retrieval for HR scale failed within IndexError fallback.")

                            hr_e_fb = np.exp(effects_df_fb['coef'])
                            hr_l_fb = np.exp(effects_df_fb['coef_lower_0.95'])
                            hr_u_fb = np.exp(effects_df_fb['coef_upper_0.95'])

                            self.log(f"DEBUG: HR Fallback Plot X-values (manual_values_1d_fb): {manual_values_1d_fb}", "DEBUG")
                            self.log(f"DEBUG: HR Fallback Plot Y-estimate (hr_e_fb): {hr_e_fb}", "DEBUG")
                            self.log(f"DEBUG: HR Fallback Plot Y-CI-lower (hr_l_fb): {hr_l_fb}", "DEBUG")
                            self.log(f"DEBUG: HR Fallback Plot Y-CI-upper (hr_u_fb): {hr_u_fb}", "DEBUG")

                            ax_vip.plot(effects_df_fb.index, hr_e_fb, label='HR Estimado (Fallback)')
                            ax_vip.fill_between(effects_df_fb.index, hr_l_fb, hr_u_fb, alpha=0.2, label='IC 95% HR (Fallback)')
                            ax_vip.axhline(1.0, color='grey', linestyle='--', lw=0.8)

                            fallback_plot_title_str = f"Impacto de '{covariate_for_plot}' sobre HR (Fallback)"
                            ax_vip.set_title(fallback_plot_title_str)
                            ax_vip.set_ylabel("Hazard Ratio (HR)")
                            ax_vip.set_xlabel(f"Valor de {covariate_for_plot}")
                            ax_vip.legend()

                            current_opts_vip['title'] = fallback_plot_title_str
                            current_opts_vip['ylabel'] = "Hazard Ratio (HR)"
                            current_opts_vip['xlabel'] = f"Valor de {covariate_for_plot}"

                        else: # Fallback for Log(HR) scale
                            cph_model_vip.plot_partial_effects_on_outcome(
                                covariate_for_plot,
                                values=manual_values_2d_fb,
                                plot_baseline=False,
                                ax=ax_vip
                            )
                            fallback_plot_title_str = f"Impacto de '{covariate_for_plot}' sobre Log(HR) (Fallback)"
                            ax_vip.set_title(fallback_plot_title_str)
                            ax_vip.set_ylabel(f"Log(Hazard Ratio) para {covariate_for_plot}")
                            ax_vip.set_xlabel(f"Valor de {covariate_for_plot}")

                            current_opts_vip['title'] = fallback_plot_title_str
                            current_opts_vip['ylabel'] = f"Log(Hazard Ratio) para {covariate_for_plot}"
                            current_opts_vip['xlabel'] = f"Valor de {covariate_for_plot}"

                        # apply_plot_options with try-except wrapper for fallback
                        try: # 24 spaces
                            opts_to_apply_fb = current_opts_vip if isinstance(current_opts_vip, dict) else self.current_plot_options.copy() # 28 spaces
                            apply_plot_options(ax_vip, opts_to_apply_fb, self.log) # 28 spaces
                        except TypeError as e_apply_opts_fb:
                            log_msg_prefix_fb = f"Fallback Covariate: '{chosen_covariate}', Scale: {'HR' if plot_as_hr else 'Log(HR)'}."
                            if "got an unexpected keyword argument 'plot'" in str(e_apply_opts_fb):
                                self.log(f"WARNING: {log_msg_prefix_fb} apply_plot_options failed with 'plot' keyword error: {e_apply_opts_fb}. Displaying with minimal styling.", "WARN")
                                if hasattr(ax_vip, 'grid'):
                                    # Ensure opts_to_apply_fb is in scope for the except block
                                    grid_option_fb = opts_to_apply_fb.get('grid', True) if 'opts_to_apply_fb' in locals() and isinstance(opts_to_apply_fb, dict) else True
                                    if grid_option_fb: ax_vip.grid(True, linestyle=':', alpha=0.6)
                                if hasattr(ax_vip, 'get_legend_handles_labels') and plot_as_hr:
                                    handles_fb, labels_fb = ax_vip.get_legend_handles_labels()
                                    if handles_fb and labels_fb and not ax_vip.get_legend():
                                        ax_vip.legend()
                            else:
                                self.log(f"ERROR: {log_msg_prefix_fb} Unexpected TypeError during apply_plot_options: {e_apply_opts_fb}", "ERROR")
                        except Exception as e_general_apply_opts_fb:
                            self.log(f"ERROR: {log_msg_prefix_fb} General error during apply_plot_options: {e_general_apply_opts_fb}", "ERROR")

                        plt.tight_layout()
                        self._create_plot_window(fig_vip, f"Impacto Variable (Fallback): {chosen_covariate} ({model_name_vip}) - Escala {'HR' if plot_as_hr else 'Log(HR)'}")
                        self.log(f"Fallback plot para '{chosen_covariate}' generado exitosamente.", "SUCCESS")
                        return
                    except Exception as e_fallback_plot:
                        self.log(f"Error durante fallback plot para '{chosen_covariate}' (Escala: {'HR' if plot_as_hr else 'Log(HR)'}): {e_fallback_plot}", "ERROR")
                        self.log(traceback.format_exc(), "DEBUG")
                        messagebox.showerror("Error de Gráfico (IndexError)", f"Error conocido (IndexError) y el intento de fallback también falló para '{chosen_covariate}'.\nDetalles: {e_fallback_plot}", parent=self.parent_for_dialogs)
                        if fig_vip and plt.fignum_exists(fig_vip.number): plt.close(fig_vip)
                        return
                else: # Not quantitative, fallback with manual values not applicable
                    self.log(f"'{covariate_for_plot}' no es cuantitativa, fallback con linspace no aplicable para el IndexError específico.", "INFO")

            # If not the specific IndexError or if fallback was not applicable/failed for quantitative
            messagebox.showerror("Error de Gráfico (IndexError)", f"Se produjo un IndexError al generar el gráfico para '{chosen_covariate}':\n{e_vip_idx}", parent=self.parent_for_dialogs)
            if fig_vip and plt.fignum_exists(fig_vip.number): plt.close(fig_vip)

        except Exception as e_vip: # General errors not caught by specific IndexErorr or data retrieval for HR
            log_final_error_prefix = f"Error general al generar gráfico de impacto para '{chosen_covariate}' (Escala: {'HR' if plot_as_hr else 'Log(HR)'}):"
            self.log(f"{log_final_error_prefix} {e_vip}", "ERROR")
            self.log(traceback.format_exc(), "DEBUG")
            messagebox.showerror("Error Gráfico", f"No se pudo generar el gráfico de impacto para '{chosen_covariate}':\n{e_vip}", parent=self.parent_for_dialogs)
            if fig_vip and plt.fignum_exists(fig_vip.number): plt.close(fig_vip)
                opts_to_apply = current_opts_vip if isinstance(current_opts_vip, dict) else self.current_plot_options.copy()
                apply_plot_options(ax_vip, opts_to_apply, self.log)
            except TypeError as e_apply_opts:
                log_msg_prefix = f"Covariate: '{chosen_covariate}', Scale: {'HR' if plot_as_hr else 'Log(HR)'}."
                if "got an unexpected keyword argument 'plot'" in str(e_apply_opts):
                    self.log(f"WARNING: {log_msg_prefix} apply_plot_options failed with 'plot' keyword error: {e_apply_opts}. Displaying with minimal styling.", "WARN")
                    # Apply minimal styling if possible
                    if hasattr(ax_vip, 'grid'):
                        grid_option = opts_to_apply.get('grid', True) if 'opts_to_apply' in locals() and isinstance(opts_to_apply, dict) else True
                        if grid_option: ax_vip.grid(True, linestyle=':', alpha=0.6)
                    if hasattr(ax_vip, 'get_legend_handles_labels') and plot_as_hr: # Only force legend for HR if it was set manually
                        handles, labels = ax_vip.get_legend_handles_labels()
                        if handles and labels and not ax_vip.get_legend():
                            ax_vip.legend()
                else:
                    self.log(f"ERROR: {log_msg_prefix} Unexpected TypeError during apply_plot_options: {e_apply_opts}", "ERROR")
            except Exception as e_general_apply_opts:
                self.log(f"ERROR: {log_msg_prefix} General error during apply_plot_options: {e_general_apply_opts}", "ERROR")

            plt.tight_layout()
            self._create_plot_window(fig_vip, f"Impacto Variable: {chosen_covariate} ({model_name_vip}) - Escala {'HR' if plot_as_hr else 'Log(HR)'}")

        except IndexError as e_vip_idx:
            tb_str_vip = traceback.format_exc()
            self.log(f"IndexError al generar gráfico de impacto para '{chosen_covariate}' (Escala: {'HR' if plot_as_hr else 'Log(HR)'}): {e_vip_idx}", "ERROR")
            self.log(tb_str_vip, "DEBUG")

            # Fallback logic for the "tuple index out of range" error, now adapted for both scales
            if "tuple index out of range" in str(e_vip_idx) and "values.shape[1]" in tb_str_vip:
                self.log(f"Detectado IndexError conocido para '{chosen_covariate}'. Intentando fallback plotting strategy.", "WARN")

                df_fit = md_vip.get('_df_for_fit_main_INTERNAL_USE')
                if df_fit is None or df_fit.empty:
                    self.log("DataFrame de ajuste no disponible para fallback.", "ERROR")
                    messagebox.showerror("Error de Gráfico (IndexError)", f"Error conocido (IndexError) y datos de ajuste no disponibles para '{chosen_covariate}'.", parent=self.parent_for_dialogs)
                    if fig_vip: plt.close(fig_vip)
                    return

                series_for_fallback = df_fit.get(covariate_for_plot)
                if series_for_fallback is None or series_for_fallback.empty or series_for_fallback.isna().all():
                    self.log(f"Serie de datos para '{covariate_for_plot}' no encontrada o vacía en df_fit para fallback.", "ERROR")
                    messagebox.showerror("Error de Gráfico (IndexError)", f"Error conocido (IndexError) y datos de la covariable '{covariate_for_plot}' no encontrados para fallback.", parent=self.parent_for_dialogs)
                    if fig_vip: plt.close(fig_vip)
                    return

                # Determine if quantitative for fallback value generation
                is_quantitative_fallback = self.covariables_type_config.get(original_name_from_q if match else chosen_covariate) == "Cuantitativa"
                if (original_name_from_q if match else chosen_covariate) not in self.covariables_type_config:
                    is_quantitative_fallback = pd.api.types.is_numeric_dtype(series_for_fallback.dtype)

                if is_quantitative_fallback:
                    min_val_fb, max_val_fb = series_for_fallback.min(), series_for_fallback.max()
                    if min_val_fb == max_val_fb or pd.isna(min_val_fb) or pd.isna(max_val_fb):
                        self.log(f"Rango no válido para fallback de '{covariate_for_plot}'.", "WARN")
                        messagebox.showerror("Error de Gráfico (IndexError)", f"Error conocido (IndexError) y rango de datos inválido para fallback de '{covariate_for_plot}'.", parent=self.parent_for_dialogs)
                        if fig_vip: plt.close(fig_vip)
                        return

                    manual_values_1d_fb = np.linspace(min_val_fb, max_val_fb, 150)
                    manual_values_2d_fb = manual_values_1d_fb.reshape(-1, 1)

                    try:
                        self.log(f"Intentando fallback plot con valores manuales para '{covariate_for_plot}'. Escala: {'HR' if plot_as_hr else 'Log(HR)'}", "INFO")

                        # Ensure fig_vip and ax_vip are from the outer scope if not re-creating, or re-create them if necessary.
                        # For this fallback, it's safer to ensure ax_vip is clean or correctly targeted.
                        # The original code structure re-created fig_vip, ax_vip here for the fallback.
                        if fig_vip: plt.close(fig_vip)
                        fig_vip, ax_vip = plt.subplots(figsize=(10, 6))

                        if plot_as_hr: # Fallback for HR scale
                            effects_df_fb = cph_model_vip.plot_partial_effects_on_outcome(
                                covariate_for_plot, values=manual_values_2d_fb, plot_baseline=False, plot=False
                            )
                            if effects_df_fb is None or effects_df_fb.empty:
                                raise ValueError("Fallback data retrieval for HR scale failed.")
                            hr_e_fb = np.exp(effects_df_fb['coef'])
                            hr_l_fb = np.exp(effects_df_fb['coef_lower_0.95'])
                            hr_u_fb = np.exp(effects_df_fb['coef_upper_0.95'])
                            ax_vip.plot(effects_df_fb.index, hr_e_fb, label='HR Estimado (Fallback)')
                            ax_vip.fill_between(effects_df_fb.index, hr_l_fb, hr_u_fb, alpha=0.2, label='IC 95% HR (Fallback)')
                            ax_vip.axhline(1.0, color='grey', linestyle='--', lw=0.8)
                            ax_vip.legend()
                            current_opts_vip['title'] = f"Impacto de '{covariate_for_plot}' sobre HR (Fallback)"
                            current_opts_vip['ylabel'] = "Hazard Ratio (HR)"
                        else: # Fallback for Log(HR) scale
                            # This is the critical path for the bug: ensure it plots Log(HR)
                            cph_model_vip.plot_partial_effects_on_outcome(
                                covariate_for_plot,
                                values=manual_values_2d_fb, # Use manually generated values
                                plot_baseline=False,
                                ax=ax_vip # Plot directly on the new ax_vip for fallback
                            )
                            # plot_partial_effects_on_outcome for Log(HR) should add its own y=0 line if applicable.
                            current_opts_vip['title'] = f"Impacto de '{covariate_for_plot}' sobre Log(HR) (Fallback)"
                            current_opts_vip['ylabel'] = f"Log(Hazard Ratio) para {covariate_for_plot}"

                        current_opts_vip['xlabel'] = f"Valor de {covariate_for_plot}"
                        apply_plot_options(ax_vip, current_opts_vip, self.log)
                        plt.tight_layout()
                        self._create_plot_window(fig_vip, f"Impacto Variable (Fallback): {chosen_covariate} ({model_name_vip}) - Escala {'HR' if plot_as_hr else 'Log(HR)'}")
                        self.log(f"Fallback plot para '{chosen_covariate}' generado exitosamente.", "SUCCESS")
                        return
                    except Exception as e_fallback_plot:
                        self.log(f"Error durante fallback plot para '{chosen_covariate}' (Escala: {'HR' if plot_as_hr else 'Log(HR)'}): {e_fallback_plot}", "ERROR")
                        self.log(traceback.format_exc(), "DEBUG")
                        messagebox.showerror("Error de Gráfico (IndexError)", f"Error conocido (IndexError) y el intento de fallback también falló para '{chosen_covariate}'.\nDetalles: {e_fallback_plot}", parent=self.parent_for_dialogs)
                        if fig_vip: plt.close(fig_vip)
                        return
                else: # Not quantitative, fallback with manual values not applicable
                    self.log(f"'{covariate_for_plot}' no es cuantitativa, fallback con linspace no aplicable.", "INFO")

            # If not the specific IndexError or if fallback was not applicable/failed
            messagebox.showerror("Error de Gráfico (IndexError)", f"Se produjo un IndexError al generar el gráfico para '{chosen_covariate}':\n{e_vip_idx}", parent=self.parent_for_dialogs)
            if fig_vip: plt.close(fig_vip)

        except Exception as e_vip:
            self.log(f"Error general al generar gráfico de impacto para '{chosen_covariate}' (Escala: {'HR' if plot_as_hr else 'Log(HR)'}): {e_vip}", "ERROR")
            self.log(traceback.format_exc(), "DEBUG")
            messagebox.showerror("Error Gráfico", f"No se pudo generar el gráfico de impacto para '{chosen_covariate}':\n{e_vip}", parent=self.parent_for_dialogs)
            if fig_vip: plt.close(fig_vip)

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
        s_txt_gst += f"  Fórmula Patsy (usada en fit): {model_dict_gst.get('formula_patsy','N/A')}\n"
        s_txt_gst += f"  Fórmula Patsy (original completa para transformar nuevos datos): {model_dict_gst.get('full_patsy_formula_for_new_data_transform','N/A')}\n"
        s_txt_gst += f"  Términos Modelo (columnas en X_design): {', '.join(model_dict_gst.get('covariates_processed',[]))}\n"
        s_txt_gst += f"  Penalización: {model_dict_gst.get('penalizer_value',0.0):.4g} (L1 Ratio: {model_dict_gst.get('l1_ratio_value',0.0):.2f})\n  Manejo Empates (UI): {model_dict_gst.get('tie_method_used','N/A')} (Lifelines usará su default: efron)\n\n" 
        
        s_txt_gst += "Coeficientes (Resumen Lifelines):\n"
        sum_df_gst = model_dict_gst.get('metrics',{}).get('summary_df')
        s_txt_gst += (sum_df_gst.to_string() + "\n\n") if sum_df_gst is not None and not sum_df_gst.empty else "  (No disponibles o modelo nulo)\n\n"
        
        s_txt_gst += "Métricas Evaluación:\n"
        metrics_gst = model_dict_gst.get('metrics',{})
        for k,v in metrics_gst.items():
            if k in ["summary_df","schoenfeld_details","HR (individual)","HR_CI (individual)","Wald p-values (individual)"]: continue
            if isinstance(v,pd.DataFrame): continue
            s_txt_gst += f"  {k}: {f'{v:.4f}' if isinstance(v,(float,np.floating)) else (str(v)[:200] if pd.notna(v) else 'N/A')}\n"
        s_txt_gst += "\nTest de Supuesto de Riesgos Proporcionales:\n"
        sch_df_detailed_residuals = model_dict_gst.get("schoenfeld_results") # This is now the primary source from the new logic
        ph_test_summary_df = model_dict_gst.get("proportional_hazard_test_summary") # This is the fallback/alternative
        schoenfeld_status_msg = model_dict_gst.get("schoenfeld_status_message", "Estado del test no especificado.")
        sch_p_g_gst = metrics_gst.get('Schoenfeld p-value (global)') # This is derived by compute_model_metrics from schoenfeld_results

        self.log(f"DEBUG (_generate_text_summary): sch_df_detailed_residuals (model_dict_gst['schoenfeld_results']) type: {type(sch_df_detailed_residuals)}, is_df: {isinstance(sch_df_detailed_residuals, pd.DataFrame)}, empty: {sch_df_detailed_residuals.empty if isinstance(sch_df_detailed_residuals, pd.DataFrame) else 'N/A'}", "DEBUG")
        self.log(f"DEBUG (_generate_text_summary): ph_test_summary_df type: {type(ph_test_summary_df)}, is_df: {isinstance(ph_test_summary_df, pd.DataFrame)}, empty: {ph_test_summary_df.empty if isinstance(ph_test_summary_df, pd.DataFrame) else 'N/A'}", "DEBUG")
        self.log(f"DEBUG (_generate_text_summary): sch_p_g_gst (from metrics): {sch_p_g_gst}", "DEBUG")
        self.log(f"DEBUG (_generate_text_summary): schoenfeld_status_msg: '{schoenfeld_status_msg}'", "DEBUG")

        has_displayed_schoenfeld_details = False
        if sch_df_detailed_residuals is not None and isinstance(sch_df_detailed_residuals, pd.DataFrame) and not sch_df_detailed_residuals.empty:
            s_txt_gst += "  Resultados Detallados de Residuos de Schoenfeld (de `check_assumptions` o su procesamiento):\n"
            if pd.notna(sch_p_g_gst):
                 s_txt_gst += f"    P-Global (derivado de estos residuos): {format_p_value(sch_p_g_gst)}\n"
            s_txt_gst += f"{sch_df_detailed_residuals.to_string()}\n"
            has_displayed_schoenfeld_details = True
        
        # Display proportional_hazard_test summary if it exists AND either
        # 1. schoenfeld_results (detailed residuals) were not available/empty OR
        # 2. It's explicitly mentioned in the status that ph_test was also run (covers cases where both might have info)
        if ph_test_summary_df is not None and isinstance(ph_test_summary_df, pd.DataFrame) and not ph_test_summary_df.empty:
            if not has_displayed_schoenfeld_details or "proportional_hazard_test" in schoenfeld_status_msg:
                 s_txt_gst += "  Resultados del Test de Proporcionalidad de Riesgos (de `proportional_hazard_test`):\n"
                 s_txt_gst += f"{ph_test_summary_df.to_string()}\n"
        
        # If no detailed residuals were displayed from either source, but a global p-value exists from compute_model_metrics
        # (which would have used schoenfeld_results, even if empty, to try and get a global p), display it.
        if not has_displayed_schoenfeld_details and \
           (ph_test_summary_df is None or (isinstance(ph_test_summary_df, pd.DataFrame) and ph_test_summary_df.empty)) and \
           pd.notna(sch_p_g_gst):
            s_txt_gst += f"  P-Global del Test de Schoenfeld (detalles de residuos no disponibles o vacíos): {format_p_value(sch_p_g_gst)}\n"

        s_txt_gst += f"  Estado General del Test (interpretación del proceso): {schoenfeld_status_msg}\n"
        s_txt_gst += "\n--- Fin Resumen ---\n"; return s_txt_gst

    def save_model(self):
        if not self._check_model_selected_and_valid(): return
        md_save = self.selected_model_in_treeview; name_save = md_save.get('model_name','Modelo_Guardado')
        
        model_dict_to_save = md_save.copy()

        fpath_save = filedialog.asksaveasfilename(title="Guardar Modelo Como...",defaultextension=".pkl",initialfile=f"{name_save.replace(' ','_').replace(':','')}.pkl",filetypes=[("Pickle","*.pkl"),("Todos","*.*")])
        if not fpath_save: self.log("Guardado cancelado.", "INFO"); return
        try:
            with open(fpath_save, "wb") as f_save: pickle.dump(model_dict_to_save, f_save)
            self.log(f"Modelo '{name_save}' guardado en: {fpath_save}", "SUCCESS"); messagebox.showinfo("Modelo Guardado",f"Modelo guardado en:\n{fpath_save}",parent=self.parent_for_dialogs)
        except Exception as e_save: self.log(f"Error guardando modelo: {e_save}","ERROR"); messagebox.showerror("Error Guardando",f"No se pudo guardar:\n{e_save}",parent=self.parent_for_dialogs)

    def load_model_from_file(self):
        fpath_load = filedialog.askopenfilename(title="Cargar Modelo Pickle",filetypes=[("Pickle","*.pkl"),("Todos","*.*")])
        if not fpath_load: self.log("Carga cancelada.", "INFO"); return
        try:
            with open(fpath_load, "rb") as f_load: loaded_md = pickle.load(f_load)
            if not (isinstance(loaded_md,dict) and 'model' in loaded_md and 'model_name' in loaded_md and isinstance(loaded_md.get('model'),CoxPHFitter)):
                raise ValueError("Archivo no contiene un modelo CoxPHFitter válido en el formato esperado.")
            if '_df_for_fit_main_INTERNAL_USE' not in loaded_md or \
               '_X_design_rm_INTERNAL_USE' not in loaded_md or \
               '_y_survival_rm_INTERNAL_USE' not in loaded_md:
                self.log(f"Advertencia: Modelo '{loaded_md.get('model_name')}' cargado sin DataFrames internos. Algunas funciones de visualización (Schoenfeld, Calibración) pueden no funcionar.", "WARN")
                messagebox.showwarning("Datos Faltantes en Modelo", "El modelo cargado no contiene los DataFrames internos necesarios para todos los gráficos (ej. Schoenfeld, Calibración). Estos gráficos podrían no funcionar.", parent=self.parent_for_dialogs)

            self.generated_models_data.append(loaded_md); self._update_models_treeview()
            new_idx_load = len(self.generated_models_data)-1
            self.treeview_lista_modelos.selection_set(str(new_idx_load)); self.treeview_lista_modelos.focus(str(new_idx_load)); self._on_model_select_from_treeview()
            self.log(f"Modelo '{loaded_md.get('model_name')}' cargado desde: {fpath_load}", "SUCCESS"); messagebox.showinfo("Modelo Cargado",f"Modelo '{loaded_md.get('model_name')}' cargado.",parent=self.parent_for_dialogs)
        except (pickle.UnpicklingError, ValueError) as e_load_val: self.log(f"Error carga/formato modelo: {e_load_val}","ERROR"); messagebox.showerror("Error Carga/Formato",f"Error al cargar o formato inválido:\n{e_load_val}",parent=self.parent_for_dialogs)
        except Exception as e_load_gen: self.log(f"Error general cargando modelo: {e_load_gen}","ERROR"); traceback.print_exc(limit=3); messagebox.showerror("Error Carga",f"No se pudo cargar:\n{e_load_gen}",parent=self.parent_for_dialogs)

    def _clear_all_generated_models(self):
        """Elimina todos los modelos generados de la lista y actualiza la Treeview."""
        if messagebox.askyesno("Confirmar Limpieza", "¿Está seguro de que desea eliminar todos los modelos generados?", parent=self.parent_for_dialogs):
            self.generated_models_data = []
            self._update_models_treeview()
            self.selected_model_in_treeview = None
            self._update_results_buttons_state() # Deshabilitar botones de resultados
            self.log("Todos los modelos generados han sido eliminados.", "INFO")

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

    def open_detailed_configuration_dialog(self):
        selected_indices = self.listbox_covariables_disponibles.curselection()
        if not selected_indices:
            messagebox.showinfo("Sin Selección", "Seleccione una o más covariables de la lista para configurar detalladamente.", parent=self.parent_for_dialogs)
            return

        selected_covs = [self.listbox_covariables_disponibles.get(i) for i in selected_indices]

        # Check if data is loaded, as it's needed by the dialog for context (e.g., populating ref categories)
        if self.data is None:
            messagebox.showerror("Error de Datos", "No hay datos cargados. Cargue un archivo de datos primero.", parent=self.parent_for_dialogs)
            self.log("Intento de abrir diálogo de config detallada sin datos cargados.", "WARN")
            return

        DetailedCovariateConfigDialog(self.parent_for_dialogs, self, selected_covs)

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
        
        df_final_shape = md_rep.get('df_final_fit_shape')
        if df_final_shape and isinstance(df_final_shape, tuple) and len(df_final_shape) >= 1:
            num_obs_rep_meth = df_final_shape[0]
        else:
            num_obs_rep_meth = 'N/A'

        num_events_rep_meth = 'N/A'
        if md_rep.get('model') and hasattr(md_rep['model'], 'event_observed'):
            try: num_events_rep_meth = int(md_rep['model'].event_observed.sum())
            except: pass
        elif md_rep.get('_y_survival_rm_INTERNAL_USE') and md_rep.get('event_col_for_model') in md_rep['_y_survival_rm_INTERNAL_USE']:
            try: num_events_rep_meth = int(md_rep['_y_survival_rm_INTERNAL_USE'][md_rep.get('event_col_for_model')].sum())
            except: pass

        report_full += f"2. Datos Usados (post-preparación para este modelo):\n   - Observaciones: {num_obs_rep_meth}\n   - Eventos: {num_events_rep_meth}\n\n"
        report_full += "3. Contenido Resumen Técnico (ver abajo):\n"
        report_full += "   - Configuración ajuste.\n   - Coeficientes (HRs, ICs).\n   - Métricas ajuste/evaluación.\n   - Test Supuestos (Schoenfeld).\n\n"
        report_full += text_summary_rep
        report_full += "\n\n4. Limitaciones y Consideraciones (Placeholder):\n   [Describa limitaciones y generalizabilidad.]\n\n"
        report_full += "5. Conclusión General (Placeholder):\n   [Interprete hallazgos en contexto.]\n"
        ModelSummaryWindow(self.parent_for_dialogs, f"Reporte Metodológico: {name_rep}", report_full)
        self.log(f"Mostrando reporte metodológico para '{name_rep}'.", "INFO")

    def show_baseline_instantaneous_hazard(self):
        if not self._check_model_selected_and_valid():
            return

        md_bih = self.selected_model_in_treeview
        cph_bih = md_bih.get('model')
        name_bih = md_bih.get('model_name', 'N/A')
        time_col_name_bih = md_bih.get('time_col_for_model', 'T')

        self.log(f"Generando gráfico de Riesgo Instantáneo Base h₀(t) para '{name_bih}'...", "INFO")

        try:
            baseline_hazard_df = cph_bih.baseline_hazard_
            if baseline_hazard_df.empty or 'baseline hazard' not in baseline_hazard_df.columns:
                self.log(f"DataFrame de riesgo base ('baseline_hazard_') vacío o sin columna 'baseline hazard' para '{name_bih}'.", "WARN")
                messagebox.showwarning("Datos No Disponibles",
                                       "El DataFrame de riesgo base está vacío o no contiene la columna 'baseline hazard'.",
                                       parent=self.parent_for_dialogs)
                return

            t = baseline_hazard_df.index.values
            H0_t = baseline_hazard_df['baseline hazard'].values

            if len(t) < 2:
                self.log(f"No hay suficientes puntos de datos en baseline_hazard_ para calcular h₀(t) para '{name_bih}'. Se requieren al menos 2.", "WARN")
                messagebox.showwarning("Datos Insuficientes",
                                       "No hay suficientes puntos de datos (<2) para calcular el riesgo instantáneo base.",
                                       parent=self.parent_for_dialogs)
                return

            dt = np.diff(t)
            dH0_t = np.diff(H0_t)

            # Handle potential division by zero if dt has zero values (unlikely for distinct time points)
            # dt can be zero if time points are not unique, filter those out
            valid_diffs = dt > 0
            if not np.all(valid_diffs):
                self.log(f"Advertencia: Se encontraron intervalos de tiempo (dt) no positivos en '{name_bih}'. Se filtrarán.", "WARN")
                t_plot = t[1:][valid_diffs]
                h0_t_raw = dH0_t[valid_diffs] / dt[valid_diffs]
                if len(t_plot) < 1: # After filtering, not enough points left
                    self.log(f"No quedan suficientes puntos de datos válidos después de filtrar dt <= 0 para '{name_bih}'.", "WARN")
                    messagebox.showwarning("Datos Insuficientes", "No quedan puntos válidos después de filtrar dt no positivos.", parent=self.parent_for_dialogs)
                    return
            else:
                t_plot = t[1:]
                h0_t_raw = dH0_t / dt

            if len(h0_t_raw) == 0: # Should be caught by len(t_plot) < 1 already
                 self.log(f"h0_t_raw está vacío después del cálculo para '{name_bih}'.", "WARN")
                 messagebox.showwarning("Cálculo Fallido", "El cálculo del riesgo instantáneo base resultó en un conjunto vacío.", parent=self.parent_for_dialogs)
                 return

            fig_bih, ax_bih = plt.subplots(figsize=(10, 6))
            ax_bih.step(t_plot, h0_t_raw, where='post', label='h₀(t) Bruta (Estimada)', alpha=0.7, linestyle='--')

            h0_t_smoothed = None
            if len(h0_t_raw) > 5: # LOESS typically needs a reasonable number of points
                try:
                    # Apply LOESS smoothing
                    # frac controls the smoothness (fraction of data used for each local regression)
                    # it controls the number of robustifying iterations
                    h0_t_smoothed_values = lowess(h0_t_raw, t_plot, frac=0.3, it=1, return_sorted=False)
                    # lowess with return_sorted=False returns an array of y-values, ensure t_plot aligns
                    ax_bih.plot(t_plot, h0_t_smoothed_values, label='h₀(t) Suavizada (LOESS, frac=0.3)', color='red')
                    self.log(f"Suavizado LOESS aplicado a h₀(t) para '{name_bih}' (frac=0.3, it=1).", "INFO")
                except Exception as e_loess:
                    self.log(f"Error durante suavizado LOESS para h₀(t) en '{name_bih}': {e_loess}. Mostrando solo datos brutos.", "ERROR")
                    traceback.print_exc(limit=2)
                    # h0_t_smoothed remains None implicitly if error, or explicitly set if needed for other logic
            else:
                self.log(f"No hay suficientes puntos en h₀(t) ({len(h0_t_raw)}) para suavizado LOESS para '{name_bih}'. Se requieren más de 5. Mostrando solo datos brutos.", "WARN")

            opts_bih = self.current_plot_options.copy()
            opts_bih['title'] = opts_bih.get('title') or f"Riesgo Instantáneo Base h₀(t) ({name_bih})"
            opts_bih['xlabel'] = opts_bih.get('xlabel') or f"Tiempo ({time_col_name_bih})"
            opts_bih['ylabel'] = opts_bih.get('ylabel') or "h₀(t) - Tasa de Riesgo Instantáneo Base"

            apply_plot_options(ax_bih, opts_bih, self.log)
            if ax_bih.get_legend() is None or not ax_bih.get_legend().get_texts(): # Ensure legend if not already by apply_plot_options
                ax_bih.legend()

            plt.tight_layout()
            self._create_plot_window(fig_bih, f"Riesgo Instant. Base: {name_bih}")
            self.log(f"Gráfico de Riesgo Instantáneo Base h₀(t) para '{name_bih}' generado y mostrado.", "SUCCESS")

        except Exception as e_bih:
            self.log(f"Error al generar gráfico de Riesgo Instantáneo Base h₀(t) para '{name_bih}': {e_bih}", "ERROR")
            traceback.print_exc(limit=3)
            messagebox.showerror("Error de Gráfico",
                                 f"No se pudo generar el gráfico de Riesgo Instantáneo Base h₀(t):\n{e_bih}",
                                 parent=self.parent_for_dialogs)
            # Ensure figure is closed if it was created before the error
            if 'fig_bih' in locals() and fig_bih is not None:
                plt.close(fig_bih)


# --- Fin de la clase CoxModelingApp ---

if __name__ == "__main__":
    root = tk.Tk()
    root.title(f"Software Modelos de Supervivencia de Cox v1.2.26")
    
    screen_w = root.winfo_screenwidth(); screen_h = root.winfo_screenheight()
    app_w = int(screen_w * 0.90); app_h = int(screen_h * 0.88)
    center_x = max(0, (screen_w - app_w) // 2); center_y = max(0, (screen_h - app_h) // 2)
    root.geometry(f"{app_w}x{app_h}+{center_x}+{center_y}"); root.minsize(1050, 720)
    
    style = ttk.Style()
    themes = style.theme_names()
    preferred_themes = ['clam', 'alt', 'default', 'classic']
    if os.name == 'nt': preferred_themes = ['vista', 'xpnative'] + preferred_themes
    
    chosen_theme = style.theme_use()
    for theme_name in preferred_themes:
        if theme_name in themes:
            try: style.theme_use(theme_name); chosen_theme = theme_name; break
            except tk.TclError: pass
    print(f"INFO: Tema UI: '{chosen_theme}'")

    app = CoxModelingApp(root)
    
    app_version = "1.2.26"
    app.log("*"*80, "HEADER"); app.log(f"  Software Modelado Cox (v{app_version}) Iniciado  ", "HEADER")
    app.log(f"  Tema UI: {chosen_theme}", "CONFIG"); app.log("*"*80, "HEADER")

    if not PATSY_AVAILABLE: app.log("ERROR CRÍTICO: 'patsy' NO encontrada. Funciones esenciales deshabilitadas. Instale 'patsy'.", "ERROR")
    else: app.log("'patsy' cargada.", "INFO")
    if not FILTER_COMPONENT_AVAILABLE: app.log("ADVERTENCIA: 'MATLAB_filter_component' NO importado. Filtros avanzados no disponibles.", "WARN")
    else: app.log("'MATLAB_filter_component' cargado.", "INFO")
    if LIFELINES_CALIBRATION_AVAILABLE: app.log("'survival_probability_calibration' disponible.", "INFO")
    else: app.log("ADVERTENCIA: 'survival_probability_calibration' NO disponible.", "WARN")
    if LIFELINES_BRIER_SCORE_AVAILABLE: app.log("'brier_score' disponible.", "INFO")
    else: app.log("ADVERTENCIA: 'brier_score' NO disponible.", "WARN")
    
    root.mainloop()
