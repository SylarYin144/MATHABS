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
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tkinter import scrolledtext

# --- Importaciones de Librerías de Terceros (Data Science y Plotting) ---
import pandas as pd
import numpy as np
import scipy.stats

import matplotlib
matplotlib.use('TkAgg')  # Backend para Tkinter

# --- Importaciones de Lifelines ---
from lifelines.exceptions import ConvergenceError
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

def parse_time_horizon_string(time_str: str, log_func=print) -> list[float]:
    """
    Parsea una cadena de horizontes de tiempo.
    La cadena puede contener números individuales, rangos (ej. "10-15"),
    o combinaciones separadas por comas.
    Retorna una lista ordenada y única de puntos de tiempo como floats.
    """
    if not time_str or time_str.isspace():
        return []

    all_time_points = []
    parts = time_str.split(',')

    for part_raw in parts:
        part = part_raw.strip()
        if not part:
            continue

        if '-' in part:
            range_components = part.split('-', 1) # Solo dividir en el primer guion
            if len(range_components) == 2:
                start_str, end_str = range_components
                try:
                    start_val = int(start_str.strip())
                    end_val = int(end_str.strip())
                    if start_val <= end_val:
                        all_time_points.extend(float(i) for i in range(start_val, end_val + 1))
                    else:
                        log_func(f"Advertencia: Rango inválido '{part}': El inicio ({start_val}) es mayor que el fin ({end_val}). Saltando.", "WARN")
                except ValueError:
                    log_func(f"Advertencia: Rango inválido '{part}': No se pudieron convertir los límites a enteros. Saltando.", "WARN")
            else: # Más de un guion o guion al inicio/final de forma incorrecta
                log_func(f"Advertencia: Formato de rango inválido '{part}'. Use 'inicio-fin'. Saltando.", "WARN")
        else:
            try:
                time_point = float(part)
                all_time_points.append(time_point)
            except ValueError:
                log_func(f"Advertencia: Parte inválida en la cadena de tiempo '{part}': No se pudo convertir a número. Saltando.", "WARN")

    if not all_time_points:
        return []

    return sorted(list(set(all_time_points)))


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
            spline_type_combo = ttk.Combobox(row_labelframe, values=["Natural", "B-spline"], state="disabled", width=10,
                                             command=lambda c=cov_name: self._toggle_row_controls_state(c)) # Añadido command
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

            # Spline Degree (Nuevo para B-Splines)
            ttk.Label(row_labelframe, text="  Spline Grado:").grid(row=5, column=0, sticky=tk.W, padx=15, pady=2)
            spline_degree_var = tk.IntVar(value=3) # Default cúbico
            self.row_configs[cov_name]['spline_degree_var'] = spline_degree_var
            # Grados comunes: 1 (lineal), 2 (cuadrático), 3 (cúbico)
            spline_degree_spinbox = ttk.Spinbox(row_labelframe, from_=1, to=5, textvariable=spline_degree_var, width=5, state="disabled")
            spline_degree_spinbox.grid(row=5, column=1, sticky=tk.W, padx=5)
            self.row_configs[cov_name]['spline_degree_spinbox'] = spline_degree_spinbox


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
                    spline_degree_var.set(spl_conf.get('degree', 3)) # Cargar grado, default 3
                else:
                    spline_var.set(False)
                    # Asegurar defaults también para grado si no hay config de spline
                    spline_degree_var.set(3)

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
        if not cov_name in self.row_configs:
            # self.app_instance.log(f"DEBUG: _toggle_row_controls_state - cov_name '{cov_name}' not in self.row_configs. Saliendo.", "DEBUG")
            return

        config = self.row_configs[cov_name]
        var_is_quantitative = config['type_var'].get() == "Cuantitativa"

        # self.app_instance.log(f"DEBUG: _toggle_row_controls_state para '{cov_name}': var_is_quantitative={var_is_quantitative}", "DEBUG")

        # Configurar ComboBox de Categoría de Referencia
        config['ref_combo'].config(state="readonly" if not var_is_quantitative else "disabled")
        if var_is_quantitative:
            config['ref_combo'].set("")

        # Configurar CheckBox "Usar Spline"
        config['cb_spline'].config(state=tk.NORMAL if var_is_quantitative else tk.DISABLED)
        if not var_is_quantitative:
            config['spline_var'].set(False) # Forzar desmarcado si no es cuantitativa

        # Obtener el estado actual del checkbox "Usar Spline" DESPUÉS de cualquier posible cambio
        spline_is_active = config['spline_var'].get()
        # self.app_instance.log(f"DEBUG: _toggle_row_controls_state para '{cov_name}': spline_is_active={spline_is_active}", "DEBUG")

        # Configurar "Tipo Spline" y "Spline DF"
        # Habilitados si la variable es cuantitativa Y el checkbox "Usar Spline" está marcado.
        can_configure_spline_type_and_df = var_is_quantitative and spline_is_active
        spline_type_df_control_state = tk.NORMAL if can_configure_spline_type_and_df else tk.DISABLED

        config['spline_type_combo'].config(state=spline_type_df_control_state)
        config['spline_df_spinbox'].config(state=spline_type_df_control_state)
        # self.app_instance.log(f"DEBUG: _toggle_row_controls_state para '{cov_name}': spline_type_df_control_state='{spline_type_df_control_state}'", "DEBUG")

        # Configurar "Spline Grado"
        # Habilitado si los detalles del spline (tipo/df) están habilitados Y el tipo es "B-spline".
        selected_spline_type = config['spline_type_combo'].get()
        can_configure_degree = can_configure_spline_type_and_df and (selected_spline_type == "B-spline")
        spline_degree_control_state = tk.NORMAL if can_configure_degree else tk.DISABLED
        config['spline_degree_spinbox'].config(state=spline_degree_control_state)
        # self.app_instance.log(f"DEBUG: _toggle_row_controls_state para '{cov_name}': selected_spline_type='{selected_spline_type}', spline_degree_control_state='{spline_degree_control_state}'", "DEBUG")

        # Resetear valores si los controles correspondientes están deshabilitados
        if not can_configure_spline_type_and_df:
            config['spline_type_combo'].set("Natural")
            config['spline_df_var'].set(4)
            config['spline_degree_var'].set(3) # Grado también se resetea

        if not can_configure_degree:
            # Si el grado no es configurable (pero tipo/df sí podrían serlo, ej. para Natural spline),
            # reseteamos la variable de grado a 3.
            # Esto es importante si se cambia de B-spline a Natural.
            config['spline_degree_var'].set(3)

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
                    spline_degree = config_widgets['spline_degree_var'].get()

                    spline_config_data = {'type': spline_type, 'df': spline_df}
                    if spline_type == "B-spline":
                        spline_config_data['degree'] = spline_degree

                    self.app_instance.spline_config_details[cov_name] = spline_config_data

                    log_message = f"Config. spline aplicada para '{cov_name}': Tipo={spline_type}, DF={spline_df}"
                    if spline_type == "B-spline":
                        log_message += f", Grado={spline_degree}"
                    log_message += "."
                    self.app_instance.log(log_message, "DEBUG")
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




class CoxGraphSelectionDialog(tk.Toplevel):
    def __init__(self, parent, title, graph_options_callbacks, apply_callback):
        super().__init__(parent)
        self.transient(parent)
        self.grab_set()
        self.title(title)
        self.parent = parent
        self.graph_options_callbacks = graph_options_callbacks # Dict: {"Graph Name": callback_function}
        self.apply_callback = apply_callback
        self.selected_graphs = {} # Dict: {"Graph Name": tk.BooleanVar}

        main_dialog_frame = ttk.Frame(self, padding="10")
        main_dialog_frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(main_dialog_frame, text="Seleccione los gráficos que desea generar:", font=("TkDefaultFont", 10, "bold")).pack(pady=(0,10), anchor='w')

        # Scrolled Frame for Checkbuttons
        scrolled_content_frame = ScrolledFrame(main_dialog_frame) # Assuming ScrolledFrame is available
        scrolled_content_frame.pack(fill=tk.BOTH, expand=True, pady=(0,10))
        checkbox_frame = scrolled_content_frame.interior

        for graph_name in self.graph_options_callbacks.keys():
            var = tk.BooleanVar(value=False)
            self.selected_graphs[graph_name] = var
            cb = ttk.Checkbutton(checkbox_frame, text=graph_name, variable=var)
            cb.pack(anchor=tk.W, padx=5, pady=2)

        buttons_frame = ttk.Frame(main_dialog_frame)
        buttons_frame.pack(fill=tk.X, pady=(10,0))

        ttk.Button(buttons_frame, text="Generar Seleccionados", command=self._on_apply).pack(side=tk.LEFT, padx=5)
        ttk.Button(buttons_frame, text="Seleccionar Todos", command=self._select_all).pack(side=tk.LEFT, padx=5)
        ttk.Button(buttons_frame, text="Deseleccionar Todos", command=self._deselect_all).pack(side=tk.LEFT, padx=5)
        ttk.Button(buttons_frame, text="Cancelar", command=self.destroy).pack(side=tk.RIGHT, padx=5)

        self.protocol("WM_DELETE_WINDOW", self.destroy)
        self.wait_window(self)

    def _on_apply(self):
        graphs_to_generate = []
        for graph_name, var in self.selected_graphs.items():
            if var.get():
                graphs_to_generate.append(graph_name)

        if not graphs_to_generate:
            messagebox.showwarning("Sin Selección", "No se seleccionó ningún gráfico para generar.", parent=self)
            return

        if self.apply_callback:
            self.apply_callback(graphs_to_generate)
        self.destroy()

    def _select_all(self):
        for var in self.selected_graphs.values():
            var.set(True)

    def _deselect_all(self):
        for var in self.selected_graphs.values():
            var.set(False)

class CalibrationPlotOptionsDialog(tk.Toplevel):
    def __init__(self, parent, available_strat_vars, log_func=print):
        super().__init__(parent)
        self.transient(parent)
        self.grab_set()
        self.title("Opciones de Gráficos OOS") # Title changed to be more generic
        self.parent_app = parent
        self.log = log_func
        self.available_strat_vars = available_strat_vars
        self.result = None

        main_frame = ttk.Frame(self, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # --- Overall OOS Plot Type Selection ---
        oos_plot_choice_frame = ttk.LabelFrame(main_frame, text="Tipo de Gráfico OOS Principal", padding="5")
        oos_plot_choice_frame.pack(fill=tk.X, pady=5)
        self.oos_plot_choice_var = tk.StringVar(value="calibration") # Default

        ttk.Radiobutton(oos_plot_choice_frame, text="Gráfico de Calibración OOS",
                        variable=self.oos_plot_choice_var, value="calibration",
                        command=self._toggle_oos_plot_sections).pack(anchor=tk.W, padx=5, pady=2)
        ttk.Radiobutton(oos_plot_choice_frame, text="Gráfico de Correlación vs. Tiempo",
                        variable=self.oos_plot_choice_var, value="correlation_time",
                        command=self._toggle_oos_plot_sections).pack(anchor=tk.W, padx=5, pady=2)

        # --- Time Horizon Entry (now more generic) ---
        time_frame = ttk.Frame(main_frame)
        time_frame.pack(fill=tk.X, pady=5)
        # Label updated to be more generic
        ttk.Label(time_frame, text="Horizontes de Tiempo (ej: 10, 20-25, 30):").pack(side=tk.LEFT, padx=(0,5))
        self.time_horizon_var = tk.StringVar()
        self.time_horizon_entry = ttk.Entry(time_frame, textvariable=self.time_horizon_var, width=20) # Increased width
        self.time_horizon_entry.pack(side=tk.LEFT)

        # --- Frame for Calibration Plot Specific Settings ---
        self.calibration_plot_settings_frame = ttk.Frame(main_frame)
        # Packed/unpacked by _toggle_oos_plot_sections

        # Encapsulated existing "Tipo de Gráfico de Calibración" into calibration_plot_settings_frame
        plot_type_frame = ttk.LabelFrame(self.calibration_plot_settings_frame, text="Opciones Específicas de Calibración", padding="5")
        plot_type_frame.pack(fill=tk.X, pady=5)

        self.plot_type_var = tk.StringVar(value="decile") # For decile/stratified calibration

        rb_decile = ttk.Radiobutton(plot_type_frame, text="Por Deciles de Riesgo",
                                    variable=self.plot_type_var, value="decile", command=self._toggle_strat_var_combo)
        rb_decile.pack(anchor=tk.W)

        rb_stratified = ttk.Radiobutton(plot_type_frame, text="Estratificado por Variable",
                                        variable=self.plot_type_var, value="stratified", command=self._toggle_strat_var_combo)
        rb_stratified.pack(anchor=tk.W)

        self.strat_var_frame = ttk.Frame(plot_type_frame)
        self.strat_var_frame.pack(fill=tk.X, padx=20, pady=(5,0))
        strat_label = ttk.Label(self.strat_var_frame, text="Variable de Estratificación:")
        strat_label.grid(row=0, column=0, sticky=tk.W, padx=(0,5), pady=2)
        self.strat_var_combo = ttk.Combobox(self.strat_var_frame, state="disabled", values=self.available_strat_vars, width=25)
        if self.available_strat_vars:
            self.strat_var_combo.set(self.available_strat_vars[0])
        self.strat_var_combo.grid(row=0, column=1, sticky=tk.EW, padx=(0,5), pady=2)
        self.group_quantitative_by_deciles_var = tk.BooleanVar(value=True)
        self.cb_group_by_deciles = ttk.Checkbutton(self.strat_var_frame,
                                                   text="Agrupar var. cuantitativa por deciles/cuantiles",
                                                   variable=self.group_quantitative_by_deciles_var)
        self.cb_group_by_deciles.grid(row=0, column=2, sticky=tk.W, padx=(10,0), pady=2)
        self.strat_var_frame.columnconfigure(1, weight=1)

        # --- Frame for Correlation over Time Plot Specific Settings ---
        self.correlation_time_plot_settings_frame = ttk.Frame(main_frame)
        # Packed/unpacked by _toggle_oos_plot_sections

        self.pearson_corr_var = tk.BooleanVar(value=True)
        cb_pearson = ttk.Checkbutton(self.correlation_time_plot_settings_frame,
                                     text="Correlación de Pearson",
                                     variable=self.pearson_corr_var)
        cb_pearson.pack(anchor=tk.W, padx=5, pady=2)

        self.spearman_corr_var = tk.BooleanVar(value=True)
        cb_spearman = ttk.Checkbutton(self.correlation_time_plot_settings_frame,
                                      text="Correlación de Spearman",
                                      variable=self.spearman_corr_var)
        cb_spearman.pack(anchor=tk.W, padx=5, pady=2)

        # OK/Cancel Buttons
        buttons_frame = ttk.Frame(main_frame)
        buttons_frame.pack(fill=tk.X, pady=(10,0))
        ttk.Button(buttons_frame, text="Generar Gráfico(s)", command=self._on_ok).pack(side=tk.RIGHT, padx=5) # Text updated
        ttk.Button(buttons_frame, text="Cancelar", command=self.destroy).pack(side=tk.RIGHT)

        self.protocol("WM_DELETE_WINDOW", self.destroy)
        self.time_horizon_entry.focus_set()

        # Initial call to set visibility and states
        self._toggle_oos_plot_sections()
        # self._toggle_strat_var_combo() # This will be called by _toggle_oos_plot_sections if needed

        self.wait_window(self)

    def _toggle_oos_plot_sections(self):
        chosen_oos_plot = self.oos_plot_choice_var.get()

        if chosen_oos_plot == "calibration":
            self.calibration_plot_settings_frame.pack(fill=tk.X, pady=5)
            if hasattr(self, 'correlation_time_plot_settings_frame'): # Check if exists before trying to unpack
                 self.correlation_time_plot_settings_frame.pack_forget()
            self._toggle_strat_var_combo() # Update states within calibration section
        elif chosen_oos_plot == "correlation_time":
            self.correlation_time_plot_settings_frame.pack(fill=tk.X, pady=5)
            self.calibration_plot_settings_frame.pack_forget()
            # Ensure calibration-specific controls are disabled when this section is hidden
            if hasattr(self, 'strat_var_combo'): self.strat_var_combo.config(state="disabled")
            if hasattr(self, 'cb_group_by_deciles'): self.cb_group_by_deciles.config(state="disabled")
        else: # Should not happen
            if hasattr(self, 'calibration_plot_settings_frame'): self.calibration_plot_settings_frame.pack_forget()
            if hasattr(self, 'correlation_time_plot_settings_frame'): self.correlation_time_plot_settings_frame.pack_forget()

    def _toggle_strat_var_combo(self):
        # This method now only cares about controls within the calibration section.
        # It should not error if its parent frame (calibration_plot_settings_frame) is hidden.
        if not hasattr(self, 'plot_type_var') or not hasattr(self, 'strat_var_combo') or not hasattr(self, 'cb_group_by_deciles'):
            return # Dialog not fully initialized or elements missing.

        if self.oos_plot_choice_var.get() != "calibration": # Only relevant if calibration plot is chosen
            self.strat_var_combo.config(state="disabled")
            self.cb_group_by_deciles.config(state="disabled")
            return

        if self.plot_type_var.get() == "stratified" and self.strat_var_combo.get():
            self.strat_var_combo.config(state="readonly" if self.available_strat_vars else "disabled")
            self.cb_group_by_deciles.config(state=tk.NORMAL)
        else:
            self.strat_var_combo.config(state="disabled")
            self.cb_group_by_deciles.config(state="disabled")

    def _on_ok(self):
        # Time horizon validation (now generic, could be single float or comma-separated list/ranges)
        time_horizon_str_val = self.time_horizon_var.get().strip()
        if not time_horizon_str_val: # Required for both plot types
            messagebox.showerror("Valor Requerido",
                               "Por favor, ingrese al menos un horizonte de tiempo.",
                               parent=self)
            return

        # Basic validation for "calibration" plot needing a single float for t_horizon
        current_oos_plot_choice = self.oos_plot_choice_var.get()
        if current_oos_plot_choice == "calibration":
            try:
                t_horizon_calib = float(time_horizon_str_val)
                if t_horizon_calib <= 0:
                    raise ValueError("El horizonte de tiempo para calibración debe ser positivo.")
            except ValueError:
                messagebox.showerror("Valor Inválido para Calibración",
                                   "El gráfico de calibración requiere un único valor numérico positivo para el horizonte de tiempo.",
                                   parent=self)
                return
        # For "correlation_time", time_horizon_str_val can be more complex, validation will be in the plotting function.

        self.result = {
            'time_horizon_str': time_horizon_str_val,
            'oos_plot_choice': current_oos_plot_choice
        }

        if current_oos_plot_choice == "calibration":
            self.result['plot_type'] = self.plot_type_var.get() # decile/stratified
            if self.plot_type_var.get() == "stratified":
                strat_var_name = self.strat_var_combo.get()
                if not strat_var_name and self.available_strat_vars:
                    messagebox.showwarning("Selección Requerida",
                                       "Por favor, seleccione una variable de estratificación para calibración.",
                                       parent=self)
                    self.result = None # Invalidate result
                    return
                elif not self.available_strat_vars and not strat_var_name :
                     messagebox.showerror("Error",
                                       "No hay variables disponibles para estratificación y ninguna seleccionada.",
                                       parent=self)
                     self.result = None # Invalidate result
                     return
                self.result['strat_var'] = strat_var_name
                self.result['group_by_deciles'] = self.group_quantitative_by_deciles_var.get()
            else: # decile plot
                self.result['strat_var'] = None
                self.result['group_by_deciles'] = False # Not applicable

        elif current_oos_plot_choice == "correlation_time":
            show_pearson = self.pearson_corr_var.get()
            show_spearman = self.spearman_corr_var.get()

            if not show_pearson and not show_spearman:
                messagebox.showwarning("Selección Requerida",
                                       "Debe seleccionar al menos un tipo de correlación (Pearson o Spearman) para este gráfico.",
                                       parent=self)
                return # Do not close dialog

            self.result['show_pearson'] = show_pearson
            self.result['show_spearman'] = show_spearman
            self.result['plot_type'] = None # Not applicable
            self.result['strat_var'] = None # Not applicable
            self.result['group_by_deciles'] = False # Not applicable

        self.log(f"Opciones de gráfico OOS seleccionadas: {self.result}", "DEBUG")
        self.destroy()

# --- CLASE PRINCIPAL DE LA APLICACIÓN ---

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
        # {cuant_var_name: {'type': 'Natural'|'B-spline', 'df': int, 'degree': int}}
        self.spline_config_details = {}
        self.current_plot_options = {}  # Diccionario para guardar opciones de gráficos

        # Variables para modelos
        self.generated_models_data = [] # Lista de diccionarios, cada uno con datos de un modelo
        self.selected_model_in_treeview = None # Diccionario del modelo seleccionado en la Treeview
        self.btn_oos_calibration = None

        # Variables de control para la UI (Pestaña 2: Modelado)
        self.cox_model_type_var = StringVar(value="Multivariado")
        self.var_selection_method_var = StringVar(value="Ninguno (usar todas)")
        self.p_enter_var = DoubleVar(value=0.05)
        self.p_remove_var = DoubleVar(value=0.05)
        self.penalization_method_var = StringVar(value="Ninguna")
        self.penalizer_strength_var = DoubleVar(value=0.1)
        self.l1_ratio_for_elasticnet_var = DoubleVar(value=0.5)
        self.tie_handling_method_var = StringVar(value="efron")
        self.calculate_cv_cindex_var = BooleanVar(value=True)
        self.cv_num_kfolds_var = IntVar(value=5)
        self.cv_random_seed_var = IntVar(value=42)
        self.covariate_scaling_method_var = StringVar(value="Ninguna")

        # Atributos para el ordenamiento del Treeview de modelos
        self.last_sort_col = None
        self.last_sort_reverse = False

        # Crear Notebook (pestañas)
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)

        # Pestaña 1: Carga, Filtros y Preproceso
        self.tab_frame_preproc = ttk.Frame(self.notebook, padding="10")
        self.tab_frame_preproc.pack(fill=tk.BOTH, expand=True)
        self.notebook.add(self.tab_frame_preproc, text='  1. Carga y Preprocesamiento de Datos  ')
        self.tab_frame_preproc_content = ScrolledFrame(self.tab_frame_preproc)
        self.tab_frame_preproc_content.pack(fill=tk.BOTH, expand=True)

        # Pestaña 2: Modelado Cox
        self.tab_frame_modeling = ttk.Frame(self.notebook, padding="10")
        self.tab_frame_modeling.pack(fill=tk.BOTH, expand=True)
        self.notebook.add(self.tab_frame_modeling, text='  2. Modelado Cox  ')
        self.tab_frame_modeling_content = ScrolledFrame(self.tab_frame_modeling)
        self.tab_frame_modeling_content.pack(fill=tk.BOTH, expand=True)

        # Pestaña 3: Visualización y Reportes
        self.tab_frame_results = ttk.Frame(self.notebook, padding="10")
        self.tab_frame_results.pack(fill=tk.BOTH, expand=True)
        self.notebook.add(self.tab_frame_results, text='  3. Resultados y Visualización  ')
        self.tab_frame_results_content = ScrolledFrame(self.tab_frame_results)
        self.tab_frame_results_content.pack(fill=tk.BOTH, expand=True)

        # Pestaña 4: Log
        self.tab_frame_log = ttk.Frame(self.notebook, padding="10")
        self.tab_frame_log.pack(fill=tk.BOTH, expand=True)
        self.notebook.add(self.tab_frame_log, text='  Log  ')
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

        # NUEVO: Spinbox para grado del B-spline en el panel simple
        ttk.Label(subframe_spline_detalles, text="  Grado (B-spline):").pack(side=tk.LEFT, padx=(15, 5))
        self.var_degree_spline_seleccionada = IntVar(value=3) # Default cúbico
        self.spinbox_degree_spline = ttk.Spinbox(subframe_spline_detalles, from_=1, to=5, textvariable=self.var_degree_spline_seleccionada, width=5, state="disabled")
        self.spinbox_degree_spline.pack(side=tk.LEFT, padx=5)

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
 
        # >>> INICIO DE LA MODIFICACIÓN PROPUESTA <<<
        # Después de poblar self.listbox_covariables_disponibles y antes de llamar a on_covariate_select_for_config

        # Asegurar que todas las covariables en la lista tengan una configuración base si aún no la tienen
        if self.data is not None:
            # cov_cols ya está definido como: cov_cols = [c for c in cols if c not in [time_sel, event_sel]]
            # cols es: cols = sorted(self.data.columns.tolist())
            # time_sel es: time_sel = self.combo_col_tiempo.get()
            # event_sel es: event_sel = self.combo_col_evento.get()

            # Es importante que cov_cols se calcule aquí con los valores actualizados de tiempo y evento
            current_cols_in_df = sorted(self.data.columns.tolist())
            current_time_selection = self.combo_col_tiempo.get()
            current_event_selection = self.combo_col_evento.get()
            actual_cov_cols = [c for c in current_cols_in_df if c not in [current_time_selection, current_event_selection]]

            for cov_name in actual_cov_cols:
                if cov_name not in self.covariables_type_config:
                    # Inferir tipo y almacenar
                    try:
                        is_numeric = pd.api.types.is_numeric_dtype(self.data[cov_name])
                        inferred_type = "Cuantitativa" if is_numeric else "Cualitativa"
                        self.covariables_type_config[cov_name] = inferred_type
                        self.log(f"Configuración de tipo inferida y almacenada para '{cov_name}': {inferred_type}", "DEBUG")

                        if inferred_type == "Cualitativa":
                            # Asignar categoría de referencia por defecto si no existe
                            if cov_name not in self.ref_categories_config:
                                unique_cats = sorted(list(self.data[cov_name].astype(str).unique()))
                                if unique_cats:
                                    self.ref_categories_config[cov_name] = unique_cats[0]
                                    self.log(f"Categoría de referencia por defecto almacenada para '{cov_name}': {unique_cats[0]}", "DEBUG")
                        elif inferred_type == "Cuantitativa":
                            # Asegurar que no haya config de ref.cat. para cuantitativas
                            if cov_name in self.ref_categories_config:
                                del self.ref_categories_config[cov_name]

                        # Si se cambia a Cualitativa y tenía config de spline, limpiarla
                        if inferred_type == "Cualitativa" and cov_name in self.spline_config_details:
                            del self.spline_config_details[cov_name]
                            self.log(f"Configuración de spline eliminada para '{cov_name}' debido a tipo Cualitativo inferido.", "DEBUG")
                    except KeyError:
                        self.log(f"Advertencia: La covariable '{cov_name}' no se encontró en self.data al intentar inferir tipo por defecto. Esto puede ocurrir si las columnas cambian rápidamente.", "WARN")
                    except Exception as e_infer:
                        self.log(f"Error al inferir configuración por defecto para '{cov_name}': {e_infer}", "ERROR")

        # >>> FIN DE LA MODIFICACIÓN PROPUESTA <<<

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

    def open_detailed_configuration_dialog(self):
        sel_indices = self.listbox_covariables_disponibles.curselection()
        if not sel_indices:
            messagebox.showwarning("Sin Selección", "Seleccione una o más covariables de la lista para configurar detalladamente.", parent=self.parent_for_dialogs)
            return

        selected_covs = [self.listbox_covariables_disponibles.get(i) for i in sel_indices]

        DetailedCovariateConfigDialog(self.parent_for_dialogs, self, selected_covs)
        self.log(f"Abierto diálogo de configuración detallada para: {selected_covs}", "INFO")

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
            'var_df_spline_seleccionada', 'spinbox_df_spline',
            'var_degree_spline_seleccionada', 'spinbox_degree_spline' # Añadidos nuevos widgets
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
                self.var_degree_spline_seleccionada.set(spl_conf.get('degree', 3)) # Cargar grado
            elif current_var_type == "Cuantitativa": # Es cuantitativa pero sin config de spline
                 self.var_usar_spline_seleccionada.set(False) # Asegurar que esté desactivado
                 self.combo_tipo_spline_seleccionada.set('Natural') # Default
                 self.var_df_spline_seleccionada.set(4) # Default
                 self.var_degree_spline_seleccionada.set(3) # Default grado
            else: # Cualitativa, spline no aplica
                self.var_usar_spline_seleccionada.set(False)
                self.var_degree_spline_seleccionada.set(3) # Resetear grado también

        else: # Ninguna seleccionada o error
            self.label_cov_seleccionada_nombre.config(text="Ninguna Seleccionada")
            self.radio_cuantitativa.config(state=tk.DISABLED)
            self.radio_cualitativa.config(state=tk.DISABLED)
            self.var_tipo_covariable_seleccionada.set("Cuantitativa") # Reset a default
            self.combo_ref_categoria_seleccionada.set("")
            self.combo_ref_categoria_seleccionada.config(state="disabled", values=[])
            self.var_usar_spline_seleccionada.set(False)
            self.var_degree_spline_seleccionada.set(3) # Reset grado
            # Los demás (checkbutton_usar_spline, etc.) se manejan en _toggle

        self._toggle_spline_and_refcat_controls()


    def _toggle_spline_and_refcat_controls(self, event=None):
        """Habilita/deshabilita controles de spline y categoría de referencia."""
        # Asegurarse que todos los widgets existen antes de intentar configurarlos
        expected_widgets_for_toggle = [
            'radio_cuantitativa', 'radio_cualitativa', 'combo_ref_categoria_seleccionada',
            'checkbutton_usar_spline', 'combo_tipo_spline_seleccionada',
            'spinbox_df_spline', 'spinbox_degree_spline' # Añadido spinbox_degree_spline
        ]
        if not all(hasattr(self, attr) for attr in expected_widgets_for_toggle):
            self.log("DEBUG: _toggle_spline_and_refcat_controls - Faltan widgets, UI no completamente inicializada.", "DEBUG")
            return

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
        spline_general_details_state = "readonly" if self.var_usar_spline_seleccionada.get() and can_use_spline else "disabled"
        self.combo_tipo_spline_seleccionada.config(state=spline_general_details_state)
        self.spinbox_df_spline.config(state=spline_general_details_state)

        # El grado solo tiene sentido para B-spline
        spline_type_selected_simple = self.combo_tipo_spline_seleccionada.get()
        spline_degree_state_simple = "readonly" if (spline_general_details_state == "readonly" and spline_type_selected_simple == "B-spline") else "disabled"
        self.spinbox_degree_spline.config(state=spline_degree_state_simple)


        # Asegurar que los valores de spline no se mantengan si se cambia de tipo o se desmarca
        if spline_general_details_state == "disabled":
            self.combo_tipo_spline_seleccionada.set("Natural") # Reset
            self.var_df_spline_seleccionada.set(4) # Reset
            self.var_degree_spline_seleccionada.set(3) # Reset grado
        elif spline_degree_state_simple == "disabled" and spline_type_selected_simple == "Natural":
             # Si es Natural Spline, el grado no es aplicable, resetear/fijar a 3 (aunque no se use directamente)
            self.var_degree_spline_seleccionada.set(3)


    def apply_covariate_config_to_selected(self):
        """Aplica la configuración de tipo, spline o categoría de referencia a las covariables seleccionadas."""
        sel_indices = self.listbox_covariables_disponibles.curselection()
        if not sel_indices:
            messagebox.showwarning("Sin Selección", "Seleccione una o más covariables para aplicar la configuración.", parent=self.parent_for_dialogs)
            return

        selected_var_names = [self.listbox_covariables_disponibles.get(i) for i in sel_indices]
        
        new_var_type_bulk = self.var_tipo_covariable_seleccionada.get()
        use_spline_bulk = self.var_usar_spline_seleccionada.get()
        spline_type_bulk = self.combo_tipo_spline_seleccionada.get()
        spline_df_bulk = self.var_df_spline_seleccionada.get()
        spline_degree_bulk = self.var_degree_spline_seleccionada.get() # NUEVO: Leer grado del panel simple
        
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
                    current_spline_config = {'type': spline_type_bulk, 'df': spline_df_bulk}
                    log_spline_parts = [f"Tipo='{spline_type_bulk}'", f"DF={spline_df_bulk}"]
                    if spline_type_bulk == "B-spline":
                        current_spline_config['degree'] = spline_degree_bulk
                        log_spline_parts.append(f"Grado={spline_degree_bulk}")

                    self.spline_config_details[var_name_apply] = current_spline_config
                    log_msgs_for_var.append(f"Spline: {', '.join(log_spline_parts)}")
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

        # Escalado de Covariables Numéricas
        frame_scaling = ttk.LabelFrame(right_col_frame, text="Preprocesamiento de Covariables Numéricas")
        frame_scaling.pack(fill=tk.X, expand=True, pady=(10,0))
        grid_scaling = ttk.Frame(frame_scaling, padding=5)
        grid_scaling.pack(fill=tk.X)

        ttk.Label(grid_scaling, text="Método de Escalado:").grid(row=0, column=0, padx=5, pady=3, sticky=tk.W)
        scaling_options = ["Ninguna", "Estandarización (Z-score)", "Normalización (Min-Max)"]
        self.combo_scaling_method = ttk.Combobox(grid_scaling,
                                                    textvariable=self.covariate_scaling_method_var,
                                                    values=scaling_options,
                                                    state="readonly",
                                                    width=25)
        self.combo_scaling_method.grid(row=0, column=1, padx=5, pady=3, sticky=tk.EW)
        # self.covariate_scaling_method_var is already set to "Ninguna" in __init__
        grid_scaling.columnconfigure(1, weight=1)
        
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
        
        # --- MODIFICACIÓN DE COLUMNAS ---
        cols_tv = (
            "#", "Nombre Modelo", "Variables y Splines",
            "AIC", "-2 LogLik", "C-Index (Train)", "C-Index (CV)",
            "Schoenfeld (p min)", "Wald (p max)" # Columna renombrada
        )
        self.treeview_lista_modelos = ttk.Treeview(self.frame_modelos_generados_display, columns=cols_tv, show="headings", height=7)
        
        col_widths = {
            "#": 40,
            "Nombre Modelo": 220,
            "Variables y Splines": 350,
            "AIC": 90,
            "-2 LogLik": 100,
            "C-Index (Train)": 100,
            "C-Index (CV)": 100,
            "Schoenfeld (p min)": 120,
            "Wald (p max)": 100 # Etiqueta actualizada
        }

        col_anchors = {
            "#": tk.CENTER,
            "AIC": tk.E,
            "-2 LogLik": tk.E,
            "C-Index (Train)": tk.E,
            "C-Index (CV)": tk.E,
            "Schoenfeld (p min)": tk.E,
            "Wald (p max)": tk.E, # Etiqueta actualizada
            "Variables y Splines": tk.W,
            "Nombre Modelo": tk.W
        }
        # --- FIN MODIFICACIÓN DE COLUMNAS ---

        for col in cols_tv:
            self.treeview_lista_modelos.heading(
                col,
                text=col,
                command=lambda c=col: self._sort_models_by_column(c) # Usar lambda para pasar el nombre de la columna
            )
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
            ("Ver Resumen", self.show_selected_model_summary),
            ("Editar Nombre/Notas", self._open_edit_model_details_dialog),
            ("Generar Gráficos Cox", self.open_graph_selection_dialog),
            ("Calibración OOS (CV)", self.show_new_calibration_plots),
            ("Predicción", self.realizar_prediccion),
            ("Exportar Resumen", self.export_model_summary),
            ("Guardar Modelo", self.save_model),
            ("Cargar Modelo", self.load_model_from_file),
            ("Reporte Metod.", self.show_methodological_report)
        ]
        
        # Layout dinámico para botones de acción
        max_btns_per_row = 6 # Adjust as needed, maybe 4 or 5 now with more buttons
        current_row_frame_acciones = None
        for i, (text, cmd) in enumerate(acciones_config_btns):
            if i % max_btns_per_row == 0:
                current_row_frame_acciones = ttk.Frame(frame_acciones)
                current_row_frame_acciones.pack(fill=tk.X, pady=1)

            button_widget = ttk.Button(current_row_frame_acciones, text=text, command=cmd)
            button_widget.pack(side=tk.LEFT, padx=3, pady=2, fill=tk.X, expand=True)

            if text == "Calibración OOS (CV)":
                self.btn_oos_calibration = button_widget

        if self.btn_oos_calibration:
            self.btn_oos_calibration.config(state=tk.DISABLED)

        # Add the new button row for clear models
        clear_models_frame = ttk.Frame(frame_acciones) # Este frame ya existe, añadir el botón aquí
        # clear_models_frame.pack(fill=tk.X, pady=5) # No re-pack si ya está
        ttk.Button(clear_models_frame, text="Limpiar Todos los Modelos", command=self._clear_all_generated_models).pack(side=tk.RIGHT, padx=5)


        self.log("Controles de Modelado Cox creados.", "DEBUG")
        self._toggle_penalization_params_ui_state() # Estado inicial de UI de penalización

    def _open_edit_model_details_dialog(self):
        if not self._check_model_selected_and_valid():
            return
        EditModelDetailsDialog(self.parent_for_dialogs, self.selected_model_in_treeview, self)

    def _on_model_select_from_treeview(self, event=None):
        self.log("Selección en Treeview de Modelos cambió.", "DEBUG")
        selected_item_id = self.treeview_lista_modelos.focus()

        if not selected_item_id:
            self.selected_model_in_treeview = None
            self.log("Ningún modelo seleccionado en Treeview.", "DEBUG")
            if hasattr(self, 'btn_oos_calibration') and self.btn_oos_calibration:
                self.btn_oos_calibration.config(state=tk.DISABLED)
            return

        try:
            item_values = self.treeview_lista_modelos.item(selected_item_id, 'values')
            if not item_values:
                self.selected_model_in_treeview = None
                self.log("Error: Item seleccionado en Treeview no tiene valores.", "ERROR")
                if hasattr(self, 'btn_oos_calibration') and self.btn_oos_calibration:
                     self.btn_oos_calibration.config(state=tk.DISABLED)
                return

            model_idx_str = item_values[0]
            model_idx = int(model_idx_str) - 1

            if 0 <= model_idx < len(self.generated_models_data):
                self.selected_model_in_treeview = self.generated_models_data[model_idx]
                self.log(f"Modelo seleccionado: '{self.selected_model_in_treeview.get('model_name', 'N/A')}'", "INFO")

                if hasattr(self, 'btn_oos_calibration') and self.btn_oos_calibration:
                    oos_preds = self.selected_model_in_treeview.get("oos_predictions")
                    c_index_cv = self.selected_model_in_treeview.get("metrics", {}).get("C-Index (CV Mean)")

                    if oos_preds is not None and pd.notna(c_index_cv):
                        self.btn_oos_calibration.config(state=tk.NORMAL)
                        self.log("Botón OOS Calibración HABILITADO.", "DEBUG")
                    else:
                        self.btn_oos_calibration.config(state=tk.DISABLED)
                        self.log(f"Botón OOS Calibración DESHABILITADO (oos_preds: {oos_preds is not None}, c_index_cv: {c_index_cv}).", "DEBUG")
            else:
                self.selected_model_in_treeview = None
                self.log(f"Error: Índice de modelo '{model_idx_str}' fuera de rango.", "ERROR")
                if hasattr(self, 'btn_oos_calibration') and self.btn_oos_calibration:
                     self.btn_oos_calibration.config(state=tk.DISABLED)

        except ValueError:
            self.selected_model_in_treeview = None
            self.log(f"Error: No se pudo convertir el ID del item '{selected_item_id}' a índice numérico.", "ERROR")
            if hasattr(self, 'btn_oos_calibration') and self.btn_oos_calibration:
                self.btn_oos_calibration.config(state=tk.DISABLED)
        except Exception as e:
            self.selected_model_in_treeview = None
            self.log(f"Error al seleccionar modelo de Treeview: {e}", "ERROR")
            traceback.print_exc(limit=2)
            if hasattr(self, 'btn_oos_calibration') and self.btn_oos_calibration:
                self.btn_oos_calibration.config(state=tk.DISABLED)

        # Explicitly call after selection logic to update UI based on (potentially) new selection
        if hasattr(self, '_update_ui_after_model_selection'):
            self._update_ui_after_model_selection()
        elif hasattr(self, 'btn_oos_calibration') and not self.selected_model_in_treeview :
             if self.btn_oos_calibration: self.btn_oos_calibration.config(state=tk.DISABLED)

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

    def _execute_cox_modeling_orchestrator(self):
        self.log("--- Iniciando Orquestador de Modelado Cox ---", "HEADER")
        if self.data is None or self.data.empty:
            messagebox.showerror("Error de Datos", "No hay datos cargados para modelar.", parent=self.parent_for_dialogs)
            self.log("Orquestador: Intento de modelar sin datos.", "ERROR")
            return

        (df_patsy_processed, X_design_initial, y_survival_initial,
         formula_patsy_initial, terms_patsy_initial,
         time_col_final, event_col_final,
         scaling_method_applied, fitted_scaler_obj, scaled_columns_info) = self._preparar_datos_para_modelado()

        if df_patsy_processed is None or X_design_initial is None or y_survival_initial is None:
            self.log("Orquestador: Falló la preparación inicial de datos. Abortando modelado.", "ERROR")
            # _preparar_datos_para_modelado ya muestra mensajes de error
            return

        if y_survival_initial.empty:
            messagebox.showerror("Error de Datos", "No quedan datos después del preprocesamiento inicial (NaNs en T/E o columnas vacías).", parent=self.parent_for_dialogs)
            self.log("Orquestador: DataFrame de supervivencia vacío después de preparación inicial.", "ERROR")
            return

        num_events = y_survival_initial[event_col_final].sum()
        if num_events == 0:
            messagebox.showwarning("Sin Eventos",
                                 "La columna de evento no contiene ningún evento (todos los valores son 0) después del preprocesamiento. "
                                 "No se pueden ajustar modelos de Cox.", parent=self.parent_for_dialogs)
            self.log("Orquestador: No hay eventos en los datos preparados. Modelado Cox no posible.", "WARN")
            return
        elif num_events < 5: # Umbral arbitrario, pero muy pocos eventos son problemáticos
             self.log(f"Orquestador: Advertencia - Muy pocos eventos ({num_events}) en los datos. Los resultados del modelo pueden ser inestables.", "WARN")


        model_type_selected = self.cox_model_type_var.get()
        penalizer_val = self.penalizer_strength_var.get() if self.penalization_method_var.get() != "Ninguna" else 0.0
        l1_ratio_val = self.l1_ratio_for_elasticnet_var.get() if self.penalization_method_var.get() == "ElasticNet" else 0.0
        if self.penalization_method_var.get() == "L1 (Lasso)":
            l1_ratio_val = 1.0
        elif self.penalization_method_var.get() == "L2 (Ridge)":
            l1_ratio_val = 0.0

        sel_cov_indices_ui = self.listbox_covariables_disponibles.curselection()
        all_selected_covs_orig_names_from_ui = [self.listbox_covariables_disponibles.get(i) for i in sel_cov_indices_ui]


        if model_type_selected == "Univariado":
            self.log("Orquestador: Iniciando modelado univariado...", "INFO")
            if not all_selected_covs_orig_names_from_ui:
                messagebox.showwarning("Sin Covariables", "Seleccione al menos una covariable para el modelado univariado.", parent=self.parent_for_dialogs)
                self.log("Orquestador: No hay covariables seleccionadas para univariado.", "WARN")
                return

            temp_models_data_univar = []
            for i_univar, cov_name_univar in enumerate(all_selected_covs_orig_names_from_ui):
                self.log(f"Ajustando modelo univariado para: {cov_name_univar} ({i_univar+1}/{len(all_selected_covs_orig_names_from_ui)})", "SUBHEADER")

                df_filtered_univar, X_design_univar, formula_patsy_univar, terms_patsy_univar = self.build_design_matrix(
                    df_patsy_processed, [cov_name_univar], time_col_final, event_col_final
                )

                if X_design_univar is None or df_filtered_univar is None:
                    self.log(f"Orquestador (Univar): Falló build_design_matrix para '{cov_name_univar}'. Saltando.", "ERROR")
                    continue

                y_survival_univar = df_filtered_univar[[time_col_final, event_col_final]]

                if X_design_univar.empty and formula_patsy_univar != "0":
                     self.log(f"Orquestador (Univar): X_design para '{cov_name_univar}' vacío y fórmula no nula. Saltando.", "WARN")
                     continue
                if y_survival_univar.empty:
                     self.log(f"Orquestador (Univar): y_survival para '{cov_name_univar}' vacío. Saltando.", "WARN")
                     continue

                model_name_univar = f"Univar_{cov_name_univar.replace(' ', '_')}"
                model_result_univar = self._run_model_and_get_metrics(
                    df_filtered_univar, X_design_univar, y_survival_univar,
                    time_col_final, event_col_final,
                    formula_patsy_univar, model_name_univar,
                    terms_patsy_univar,
                    formula_patsy_univar,
                    penalizer_val, l1_ratio_val,
                    model_type_for_fit_logic=model_type_selected,
                    scaling_method_applied=scaling_method_applied,
                    fitted_scaler_obj=fitted_scaler_obj,
                    scaled_columns_info=scaled_columns_info
                )
                if model_result_univar and model_result_univar.get("model"):
                    temp_models_data_univar.append(model_result_univar)
                else:
                    self.log(f"Orquestador (Univar): Modelo para '{cov_name_univar}' no se ajustó correctamente o no retornó resultados.", "WARN")

            self.generated_models_data.extend(temp_models_data_univar)

        elif model_type_selected == "Multivariado":
            self.log("Orquestador: Iniciando modelado multivariado...", "INFO")

            final_covs_for_multivar_model_orig_names = self._perform_variable_selection(
                df_patsy_processed, X_design_initial, time_col_final, event_col_final, formula_patsy_initial, terms_patsy_initial
            )

            if not final_covs_for_multivar_model_orig_names and not (X_design_initial.empty and formula_patsy_initial == "0"):
                self.log("Orquestador (Multivar): No quedaron covariables después de la selección de variables. Ajustando modelo nulo.", "INFO")
                final_covs_for_multivar_model_orig_names = []

            df_filtered_final_multi, X_design_final_multi, formula_patsy_final_multi, terms_patsy_final_multi = self.build_design_matrix(
                df_patsy_processed, final_covs_for_multivar_model_orig_names, time_col_final, event_col_final
            )

            if X_design_final_multi is None or df_filtered_final_multi is None:
                self.log("Orquestador (Multivar): Falló la reconstrucción de la matriz de diseño final. Abortando.", "ERROR")
                return

            y_survival_final_multi = df_filtered_final_multi[[time_col_final, event_col_final]]

            if X_design_final_multi.empty and formula_patsy_final_multi != "0":
                 self.log("Orquestador (Multivar): X_design final vacío y fórmula no nula. Abortando.", "ERROR")
                 return
            if y_survival_final_multi.empty:
                 self.log("Orquestador (Multivar): y_survival final vacío. Abortando.", "ERROR")
                 return

            model_name_multivar = "Multivariado_Full"
            if not final_covs_for_multivar_model_orig_names: model_name_multivar = "Multivariado_Nulo"

            model_result_multivar = self._run_model_and_get_metrics(
                df_filtered_final_multi, X_design_final_multi, y_survival_final_multi,
                time_col_final, event_col_final,
                formula_patsy_final_multi, model_name_multivar,
                terms_patsy_final_multi,
                formula_patsy_final_multi,
                penalizer_val, l1_ratio_val,
                model_type_for_fit_logic=model_type_selected,
                scaling_method_applied=scaling_method_applied,
                fitted_scaler_obj=fitted_scaler_obj,
                scaled_columns_info=scaled_columns_info
            )
            if model_result_multivar and model_result_multivar.get("model"):
                self.generated_models_data.append(model_result_multivar)
            else:
                self.log("Orquestador (Multivar): Modelo multivariado no se ajustó correctamente o no retornó resultados.", "WARN")

        else:
            messagebox.showerror("Error Interno", f"Tipo de modelo '{model_type_selected}' no reconocido.", parent=self.parent_for_dialogs)
            self.log(f"Orquestador: Tipo de modelo desconocido '{model_type_selected}'.", "ERROR")
            return

        self._update_models_treeview()
        if self.generated_models_data:
            self.log(f"Orquestador: {len(self.generated_models_data)} modelo(s) generado(s) y añadido(s) a la lista.", "SUCCESS")
            if hasattr(self, 'treeview_lista_modelos') and self.treeview_lista_modelos.get_children():
                last_item_id = self.treeview_lista_modelos.get_children()[-1]
                self.treeview_lista_modelos.selection_set(last_item_id)
                self.treeview_lista_modelos.focus(last_item_id)
                self.treeview_lista_modelos.see(last_item_id)
        else:
            self.log("Orquestador: No se generaron modelos válidos.", "WARN")

        self.log("--- Finalizado Orquestador de Modelado Cox ---", "HEADER")

    def _preparar_datos_para_modelado(self):
        if self.data is None or self.data.empty:
            self.log("No hay datos cargados.", "WARN"); messagebox.showwarning("Sin Datos", "Cargue datos primero.", parent=self.parent_for_dialogs); return None, None, None, None, None, None, None, "Ninguna", None, []
        
        time_col_ui = self.combo_col_tiempo.get().strip()
        event_col_ui = self.combo_col_evento.get().strip()
        rename_time_to = self.entry_renombrar_col_tiempo.get().strip()
        rename_event_to = self.entry_renombrar_col_evento.get().strip()

        if not time_col_ui or not event_col_ui:
            self.log("Columnas T/E no seleccionadas.", "WARN"); messagebox.showwarning("Variables Faltantes", "Seleccione Tiempo y Evento.", parent=self.parent_for_dialogs); return None, None, None, None, None, None, None, "Ninguna", None, []
        if time_col_ui not in self.data.columns or event_col_ui not in self.data.columns:
            self.log(f"Columnas T/E ('{time_col_ui}', '{event_col_ui}') no en datos.", "ERROR"); messagebox.showerror("Columnas Inválidas", "Columnas T/E no existen.", parent=self.parent_for_dialogs); return None, None, None, None, None, None, None, "Ninguna", None, []

        sel_cov_indices = self.listbox_covariables_disponibles.curselection()
        selected_covs_orig_names = [self.listbox_covariables_disponibles.get(i) for i in sel_cov_indices if self.listbox_covariables_disponibles.get(i) not in [time_col_ui, event_col_ui]]

        if self.cox_model_type_var.get() == "Multivariado" and not selected_covs_orig_names:
             self.log("Multivariado sin covariables -> modelo nulo.", "INFO")
        elif self.cox_model_type_var.get() == "Univariado" and not selected_covs_orig_names:
             self.log("Univariado sin covariables. Abortando.", "WARN"); messagebox.showwarning("Sin Covariables", "Seleccione covariables para univariado.", parent=self.parent_for_dialogs); return None, None, None, None, None, None, None, "Ninguna", None, []

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
            return None, None, None, None, None, None, None, "Ninguna", None, []
        
        try:
            df_model_prep[final_e_col] = pd.to_numeric(df_model_prep[final_e_col])
            df_model_prep.dropna(subset=[final_t_col, final_e_col], inplace=True) # Drop NaNs in T/E cols *before* astype(int)
            if df_model_prep.empty:
                 self.log(f"Dataset vacío después de convertir T/E a numérico y eliminar NaNs en T/E.", "ERROR")
                 messagebox.showerror("Datos Insuficientes", "No quedan datos válidos para T/E después de la conversión a numérico y eliminación de NaNs.", parent=self.parent_for_dialogs)
                 return None, None, None, None, None, None, None, "Ninguna", None, []
 
            if not df_model_prep[final_e_col].isin([0, 1]).all():
                 num_invalid_events = df_model_prep[~df_model_prep[final_e_col].isin([0, 1])].shape[0]
                 self.log(f"Columna Evento '{final_e_col}' tiene {num_invalid_events} valor(es) que no son 0 o 1 después de conversión y dropna.", "ERROR")
                 messagebox.showerror("Error de Tipo", f"Columna Evento '{final_e_col}' debe contener solo valores 0 o 1.", parent=self.parent_for_dialogs)
                 return None, None, None, None, None, None, None, "Ninguna", None, []
            df_model_prep[final_e_col] = df_model_prep[final_e_col].astype(int)
        except ValueError as e_e: # Si to_numeric falla completamente
            self.log(f"Error convirtiendo columna Evento '{final_e_col}' a numérico 0/1: {e_e}", "ERROR")
            messagebox.showerror("Error de Tipo", f"Columna Evento '{final_e_col}' no puede ser convertida a numérica (0/1).", parent=self.parent_for_dialogs)
            return None, None, None, None, None, None, None, "Ninguna", None, []
 
        initial_rows_prep = len(df_model_prep)
        # Note: NaNs in T/E already handled above. This second dropna might be redundant for T/E but kept for safety.
        df_model_prep.dropna(subset=[final_t_col, final_e_col], inplace=True)
        if len(df_model_prep) < initial_rows_prep: self.log(f"Eliminadas {initial_rows_prep - len(df_model_prep)} filas con NaN en T/E (posiblemente de conversión).", "WARN")
        if df_model_prep.empty: self.log("DF vacío post-NaN en T/E.", "ERROR"); messagebox.showerror("Datos Insuficientes", "No quedan datos post-NaN en T/E.", parent=self.parent_for_dialogs); return None, None, None, None, None, None, None, "Ninguna", None, []

        # --- Aplicar Escalado de Covariables ---
        scaling_method = self.covariate_scaling_method_var.get()
        fitted_scaler = None
        scaled_column_names = []

        numeric_cols_to_scale = [
            col for col in selected_covs_orig_names
            if col in df_model_prep.columns and pd.api.types.is_numeric_dtype(df_model_prep[col])
        ]

        if not numeric_cols_to_scale and scaling_method != "Ninguna":
            self.log(f"Método de escalado '{scaling_method}' seleccionado, pero no hay covariables numéricas seleccionadas/disponibles para escalar. No se aplicará escalado.", "WARN")
            scaling_method = "Ninguna"
        elif not numeric_cols_to_scale and scaling_method == "Ninguna":
             self.log("No hay covariables numéricas seleccionadas para escalar, y el método es 'Ninguna'.", "INFO")


        if scaling_method == "Estandarización (Z-score)" and numeric_cols_to_scale:
            scaler = StandardScaler()
            df_model_prep[numeric_cols_to_scale] = scaler.fit_transform(df_model_prep[numeric_cols_to_scale])
            fitted_scaler = scaler
            scaled_column_names = numeric_cols_to_scale.copy()
            self.log(f"Covariables estandarizadas (Z-score): {scaled_column_names}", "INFO")
        elif scaling_method == "Normalización (Min-Max)" and numeric_cols_to_scale:
            scaler = MinMaxScaler()
            df_model_prep[numeric_cols_to_scale] = scaler.fit_transform(df_model_prep[numeric_cols_to_scale])
            fitted_scaler = scaler
            scaled_column_names = numeric_cols_to_scale.copy()
            self.log(f"Covariables normalizadas (Min-Max): {scaled_column_names}", "INFO")
        elif scaling_method == "Ninguna":
            self.log("No se aplicó escalado de covariables.", "INFO")
        # --- Fin Escalado ---

        df_filtered_patsy, X_design_patsy, formula_patsy_gen, terms_patsy_display = self.build_design_matrix(
            df_model_prep, selected_covs_orig_names, final_t_col, final_e_col
        )
        self.log(f"DEBUG: df_filtered_patsy columns after build_design_matrix: {df_filtered_patsy.columns.tolist() if df_filtered_patsy is not None else 'N/A'}", "DEBUG")

        if X_design_patsy is None or df_filtered_patsy is None: 
             self.log("Falló build_design_matrix.", "ERROR"); return None, None, None, None, None, None, None, "Ninguna", None, []
        
        y_survival_patsy = df_filtered_patsy[[final_t_col, final_e_col]]

        self.log(f"Preparación de datos: DF ({df_filtered_patsy.shape}), X ({X_design_patsy.shape}), y ({y_survival_patsy.shape})", "INFO")
        self.log(f"Fórmula Patsy: {formula_patsy_gen}", "DEBUG")
        self.log(f"Términos Patsy: {terms_patsy_display}", "DEBUG")
        
        return (df_filtered_patsy, X_design_patsy, y_survival_patsy,
                formula_patsy_gen, terms_patsy_display,
                final_t_col, final_e_col,
                scaling_method, fitted_scaler, scaled_column_names)


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
            
            term_syntax_bd = f"Q('{orig_cov_name_bd}')" # Default term

            if config_type_bd == "Cuantitativa":
                is_spline_candidate = orig_cov_name_bd in self.spline_config_details

                # >>> INICIO DE LA MODIFICACIÓN PROPUESTA PARA SPLINES <<<
                if is_spline_candidate:
                    try:
                        if orig_cov_name_bd not in df_for_patsy_bd.columns:
                            self.log(f"Advertencia: La columna '{orig_cov_name_bd}' no se encontró en df_for_patsy_bd al preparar para spline. Se omitirá.", "WARN")
                            continue # Saltar esta covariable

                        # Asegurar que la columna sea numérica para operaciones de spline
                        df_for_patsy_bd[orig_cov_name_bd] = pd.to_numeric(df_for_patsy_bd[orig_cov_name_bd], errors='coerce')

                        if df_for_patsy_bd[orig_cov_name_bd].isnull().all():
                            self.log(f"Advertencia: La columna '{orig_cov_name_bd}' es completamente NaN después de pd.to_numeric. No se puede usar para spline. Se omitirá.", "WARN")
                            continue # Saltar esta covariable

                        # Proceder con la configuración del spline
                        spl_cfg_bd = self.spline_config_details[orig_cov_name_bd]
                        spline_type = spl_cfg_bd.get('type', 'Natural')
                        spline_df = spl_cfg_bd.get('df', 4)

                        if spline_type == 'Natural':
                            patsy_func_bd = 'cr'
                            term_syntax_bd = f"{patsy_func_bd}(Q('{orig_cov_name_bd}'), df={spline_df})"
                        elif spline_type == 'B-spline':
                            patsy_func_bd = 'bs'
                            spline_degree = spl_cfg_bd.get('degree', 3)
                            term_syntax_bd = f"{patsy_func_bd}(Q('{orig_cov_name_bd}'), df={spline_df}, degree={spline_degree})"
                        else: # Fallback si el tipo de spline no es reconocido
                            term_syntax_bd = f"Q('{orig_cov_name_bd}')"
                            self.log(f"Advertencia: Tipo de spline '{spline_type}' no reconocido para '{orig_cov_name_bd}'. Usando término cuantitativo estándar.", "WARN")

                    except Exception as e_spline_prep:
                        self.log(f"Error al preparar '{orig_cov_name_bd}' para spline: {e_spline_prep}. Se usará como término cuantitativo estándar.", "ERROR")
                        term_syntax_bd = f"Q('{orig_cov_name_bd}')" # Fallback a término normal si hay error en la preparación del spline
                else: # No es candidato a spline, usar término cuantitativo estándar
                    term_syntax_bd = f"Q('{orig_cov_name_bd}')"
                # >>> FIN DE LA MODIFICACIÓN PROPUESTA PARA SPLINES <<<

            else: # Cualitativa
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

    def _run_model_and_get_metrics(self, df_lifelines_rm, X_design_rm, y_survival_rm,
                                   time_col_rm, event_col_rm,
                                   formula_patsy_rm, model_name_rm,
                                   covariates_display_terms_rm,
                                   full_patsy_formula_for_new_data_transform_arg, 
                                   penalizer_val_rm=0.0, l1_ratio_val_rm=0.0, 
                                   model_type_for_fit_logic="Multivariado",
                                   scaling_method_applied="Ninguna",
                                   fitted_scaler_obj=None,
                                   scaled_columns_info=None):
        self.log(f"Ajustando modelo Cox: '{model_name_rm}'...", "INFO")
        
        ui_selected_tie_method = self.tie_handling_method_var.get()

        model_data_rm = {
            "model_name": model_name_rm, "time_col_for_model": time_col_rm, "event_col_for_model": event_col_rm,
            "formula_patsy": formula_patsy_rm,
            "full_patsy_formula_for_new_data_transform": full_patsy_formula_for_new_data_transform_arg,
            "covariates_processed": covariates_display_terms_rm,
            "df_used_for_fit": self.data.copy(),
            "X_design_used_for_fit": X_design_rm.copy(),
            "y_survival_used_for_fit": y_survival_rm.copy(),
            "penalizer_value": penalizer_val_rm, "l1_ratio_value": l1_ratio_val_rm,
            "tie_method_used": ui_selected_tie_method,
            "metrics": {}, "schoenfeld_results": pd.DataFrame(), "model": None, "loglik_null": None,
            "c_index_cv_mean": None, "c_index_cv_std": None,
            "schoenfeld_status_message": "Test de Schoenfeld no ejecutado o no aplicable inicialmente.",
            "proportional_hazard_test_summary": None,
            "oos_predictions": None,
            "scaling_method_applied": scaling_method_applied,
            "fitted_scaler_object": fitted_scaler_obj,
            "scaled_columns_info": scaled_columns_info if scaled_columns_info is not None else []
        }
        model_data_rm['custom_model_name'] = None
        model_data_rm['model_notes'] = ''

        # 1. Fit Null Model
        try:
            cph_null_rm = CoxPHFitter(penalizer=0.0)
            df_for_null_fit_rm = df_lifelines_rm[[time_col_rm, event_col_rm]].copy()
            cph_null_rm.fit(df_for_null_fit_rm, duration_col=time_col_rm, event_col=event_col_rm, formula="0")
            model_data_rm["loglik_null"] = cph_null_rm.log_likelihood_
        except Exception as e_null_fit:
            self.log(f"Error ajustando modelo nulo para '{model_name_rm}': {e_null_fit}", "WARN")
            model_data_rm["loglik_null"] = None

        # 2. Prepare for Main Model Fit
        cph_main_rm_instance = CoxPHFitter(penalizer=penalizer_val_rm, l1_ratio=l1_ratio_val_rm)
        df_for_fit_main = df_lifelines_rm.copy()
        model_data_rm["df_final_fit_shape"] = df_for_fit_main.shape
        actual_formula_for_fit = formula_patsy_rm

        self.log(f"DEBUG: Attempting to fit main model '{model_name_rm}'. DF shape: {df_for_fit_main.shape}, Formula: '{actual_formula_for_fit}'", "DEBUG")

        # 3. Main Model Fit with Detailed Error Handling
        fitted_cph_model = None
        if df_for_fit_main.empty:
            self.log(f"FALLO DE AJUSTE DEL MODELO: '{model_name_rm}'. El DataFrame para el ajuste está vacío.", "ERROR")
        elif X_design_rm.empty and actual_formula_for_fit != "0":
            self.log(f"FALLO DE AJUSTE DEL MODELO: '{model_name_rm}'. X_design (columnas de patsy) está vacío pero la fórmula no es nula ('{actual_formula_for_fit}').", "ERROR")
        else:
            try:
                cph_main_rm_instance.fit(df_for_fit_main, duration_col=time_col_rm, event_col=event_col_rm, formula=actual_formula_for_fit)
                model_data_rm["model"] = cph_main_rm_instance
                fitted_cph_model = cph_main_rm_instance
                self.log(f"Modelo '{model_name_rm}' ajustado exitosamente.", "SUCCESS")
            except ConvergenceError as e_conv:
                num_obs_fail = df_for_fit_main.shape[0]
                num_events_fail = df_for_fit_main[event_col_rm].sum() if event_col_rm in df_for_fit_main.columns else 'N/A'
                self.log(f"FALLO DE AJUSTE DEL MODELO (ConvergenceError): '{model_name_rm}'", "ERROR")
                self.log(f"  Error específico: {e_conv}", "ERROR")
                self.log(f"  Observaciones usadas: {num_obs_fail}, Eventos: {num_events_fail}", "ERROR")
                if "cr(" in actual_formula_for_fit:
                    self.log("  ADVERTENCIA ADICIONAL: El modelo incluía splines naturales (cr()). Estos pueden ser numéricamente inestables. Considere usar B-splines (bs()) o reducir los grados de libertad (df).", "WARN")
                traceback.print_exc(limit=2)
            except np.linalg.LinAlgError as e_linalg:
                num_obs_fail = df_for_fit_main.shape[0]
                num_events_fail = df_for_fit_main[event_col_rm].sum() if event_col_rm in df_for_fit_main.columns else 'N/A'
                self.log(f"FALLO DE AJUSTE DEL MODELO (LinAlgError - ej. Matriz Singular): '{model_name_rm}'", "ERROR")
                self.log(f"  Error específico: {e_linalg}", "ERROR")
                self.log(f"  Observaciones usadas: {num_obs_fail}, Eventos: {num_events_fail}", "ERROR")
                if "cr(" in actual_formula_for_fit:
                    self.log("  ADVERTENCIA ADICIONAL: El modelo incluía splines naturales (cr()). Estos pueden causar problemas de colinealidad. Considere usar B-splines (bs()) o reducir los grados de libertad (df).", "WARN")
                traceback.print_exc(limit=2)
            except Exception as e_fit_main:
                num_obs_fail = df_for_fit_main.shape[0] if isinstance(df_for_fit_main, pd.DataFrame) else 'N/A'
                num_events_fail = (df_for_fit_main[event_col_rm].sum() if isinstance(df_for_fit_main, pd.DataFrame) and event_col_rm in df_for_fit_main.columns else 'N/A')
                self.log(f"FALLO DE AJUSTE DEL MODELO (Error General e Inesperado): '{model_name_rm}'", "ERROR")
                self.log(f"  Error específico: {e_fit_main}", "ERROR")
                if num_obs_fail != 'N/A':
                    self.log(f"  Observaciones (si disponibles): {num_obs_fail}, Eventos (si disponibles): {num_events_fail}", "ERROR")
                traceback.print_exc(limit=3)

        fitted_cph_model = model_data_rm.get("model")

        if fitted_cph_model:
            if not X_design_rm.empty:
                if hasattr(fitted_cph_model, 'params_') and fitted_cph_model.params_ is not None and not fitted_cph_model.params_.empty:
                    self.log(f"--- Iniciando Test de Schoenfeld para Modelo: '{model_name_rm}' ---", "INFO")
                    try:
                        results_check_assumptions = fitted_cph_model.check_assumptions(df_for_fit_main)
                        model_data_rm["check_assumptions_results_raw"] = results_check_assumptions
                        schoenfeld_df_candidate = None
                        found_schoenfeld_results = False
                        if isinstance(results_check_assumptions, list) and results_check_assumptions:
                            for i_item, item_ca in enumerate(results_check_assumptions):
                                if isinstance(item_ca, pd.DataFrame) and not item_ca.empty and all(col_ca in item_ca.columns for col_ca in ['test_statistic', 'p']):
                                    schoenfeld_df_candidate = item_ca
                                    found_schoenfeld_results = True
                                    break
                                elif hasattr(item_ca, 'summary') and isinstance(item_ca.summary, pd.DataFrame) and not item_ca.summary.empty and all(col_ca in item_ca.summary.columns for col_ca in ['test_statistic', 'p']):
                                    schoenfeld_df_candidate = item_ca.summary
                                    found_schoenfeld_results = True
                                    break
                            if not found_schoenfeld_results and len(results_check_assumptions) >= 2 and isinstance(results_check_assumptions[1], pd.DataFrame) and all(col_ca in results_check_assumptions[1].columns for col_ca in ['test_statistic', 'p']):
                                 schoenfeld_df_candidate = results_check_assumptions[1]
                                 found_schoenfeld_results = True

                        if found_schoenfeld_results and schoenfeld_df_candidate is not None:
                            model_data_rm["schoenfeld_results"] = schoenfeld_df_candidate.copy()
                            model_data_rm["schoenfeld_status_message"] = "Test de Schoenfeld (check_assumptions) calculado exitosamente."
                        else:
                            model_data_rm["schoenfeld_status_message"] = "Schoenfeld: resultados detallados no encontrados o en formato inesperado."
                    except Exception as e_sch_detailed:
                        model_data_rm["schoenfeld_status_message"] = "Error durante Test de Schoenfeld (check_assumptions)."
                        self.log(f"ERROR en Test Schoenfeld para '{model_name_rm}': {e_sch_detailed}\\n{traceback.format_exc()}", "ERROR")
                    self.log(f"--- Test de Schoenfeld para Modelo: '{model_name_rm}' Finalizado. Status: {model_data_rm['schoenfeld_status_message']} ---", "INFO")
                else:
                    self.log(f"INFO: Modelo '{model_name_rm}' sin parámetros. Test de Schoenfeld no aplicable.", "INFO")
                    model_data_rm["schoenfeld_status_message"] = "Test de Schoenfeld no aplicable (modelo sin covariables)."
            else:
                self.log(f"INFO: Modelo nulo '{model_name_rm}' (X_design_rm vacío). Test de Schoenfeld no aplicable.", "INFO")
                model_data_rm["schoenfeld_status_message"] = "Test de Schoenfeld no aplicable (modelo nulo)."

            schoenfeld_df_current = model_data_rm.get("schoenfeld_results")
            status_msg_current = model_data_rm.get("schoenfeld_status_message", "")
            should_try_ph_test_fallback = (schoenfeld_df_current is None or schoenfeld_df_current.empty) and \
                                          ("calculado exitosamente" not in status_msg_current)

            if should_try_ph_test_fallback and hasattr(fitted_cph_model, 'params_') and fitted_cph_model.params_ is not None and not fitted_cph_model.params_.empty:
                self.log(f"INFO: Intentando `proportional_hazard_test` para '{model_name_rm}'.", "INFO")
                try:
                    from lifelines.statistics import proportional_hazard_test
                    ph_test_results = proportional_hazard_test(fitted_cph_model, df_for_fit_main, time_transform='log')
                    if ph_test_results is not None and hasattr(ph_test_results, 'summary') and isinstance(ph_test_results.summary, pd.DataFrame) and not ph_test_results.summary.empty:
                        model_data_rm["proportional_hazard_test_summary"] = ph_test_results.summary
                        model_data_rm["schoenfeld_status_message"] += " Adicionalmente, proportional_hazard_test proporcionó un resumen."
                        self.log(f"INFO: `proportional_hazard_test` para '{model_name_rm}' exitoso.", "INFO")
                    else:
                        model_data_rm["schoenfeld_status_message"] += " Adicionalmente, proportional_hazard_test no arrojó resumen."
                except Exception as e_ph_test_fallback:
                    self.log(f"ERROR en `proportional_hazard_test` para '{model_name_rm}': {e_ph_test_fallback}", "ERROR")
                    model_data_rm["schoenfeld_status_message"] += f" (Error en proportional_hazard_test: {str(e_ph_test_fallback)[:30]}...)."

            if self.calculate_cv_cindex_var.get() and not X_design_rm.empty:
                self.log(f"Iniciando cálculo de C-Index CV para '{model_name_rm}'.", "INFO")
                try:
                    kf_cv = KFold(n_splits=self.cv_num_kfolds_var.get(), shuffle=True, random_state=self.cv_random_seed_var.get())
                    c_indices_cv_list = []
                    all_oos_predictions_data = []
                    for train_idx, test_idx in kf_cv.split(df_for_fit_main):
                        df_fold_for_fit_cv = df_for_fit_main.iloc[train_idx].copy()
                        df_fold_for_pred_cv = df_for_fit_main.iloc[test_idx].copy()
                        y_te_cv_for_cindex = df_fold_for_pred_cv[[time_col_rm, event_col_rm]]
                        if df_fold_for_fit_cv.empty or y_te_cv_for_cindex.empty : continue
                        cph_fold_cv = CoxPHFitter(penalizer=penalizer_val_rm, l1_ratio=l1_ratio_val_rm)
                        cph_fold_cv.fit(df_fold_for_fit_cv, duration_col=time_col_rm, event_col=event_col_rm, formula=actual_formula_for_fit)
                        preds_te_fold_cv = cph_fold_cv.predict_partial_hazard(df_fold_for_pred_cv)
                        c_idx_fold_cv = concordance_index(y_te_cv_for_cindex[time_col_rm], -preds_te_fold_cv, y_te_cv_for_cindex[event_col_rm])
                        c_indices_cv_list.append(c_idx_fold_cv)
                        try:
                            oos_sf_fold_cv = cph_fold_cv.predict_survival_function(df_fold_for_pred_cv)
                            for subj_orig_idx_cv in df_fold_for_pred_cv.index:
                                all_oos_predictions_data.append({
                                    "subject_id": subj_orig_idx_cv,
                                    "true_time": y_te_cv_for_cindex.loc[subj_orig_idx_cv, time_col_rm],
                                    "true_event": y_te_cv_for_cindex.loc[subj_orig_idx_cv, event_col_rm],
                                    "predicted_survival_function": oos_sf_fold_cv[subj_orig_idx_cv]
                                })
                        except Exception as e_pred_sf_cv_loop:
                            self.log(f"Error prediciendo OOS SF en CV para '{model_name_rm}': {e_pred_sf_cv_loop}", "WARN")
                    if c_indices_cv_list:
                        model_data_rm["c_index_cv_mean"] = np.mean(c_indices_cv_list)
                        model_data_rm["c_index_cv_std"] = np.std(c_indices_cv_list)
                        self.log(f"C-Index CV para '{model_name_rm}': Media={model_data_rm['c_index_cv_mean']:.3f} (DE={model_data_rm['c_index_cv_std']:.3f})", "INFO")
                    if all_oos_predictions_data:
                        model_data_rm["oos_predictions"] = all_oos_predictions_data
                        self.log(f"Almacenadas {len(all_oos_predictions_data)} predicciones OOS de CV para '{model_name_rm}'.", "INFO")
                except Exception as e_cv_rm_main:
                    self.log(f"Error general en C-Index CV para '{model_name_rm}': {e_cv_rm_main}", "ERROR")
                    traceback.print_exc(limit=3)
            elif self.calculate_cv_cindex_var.get():
                 self.log(f"C-Index CV no calculado para '{model_name_rm}' (modelo nulo o X_design vacío).", "INFO")
        else:
            self.log(f"Ajuste del modelo '{model_name_rm}' falló. Omitiendo tests de Schoenfeld y C-Index CV.", "WARN")
            model_data_rm["schoenfeld_status_message"] = "No aplicable (fallo en ajuste de modelo)."
            model_data_rm["c_index_cv_mean"] = None
            model_data_rm["c_index_cv_std"] = None
            model_data_rm["oos_predictions"] = None

        original_vars_for_this_model = _extract_original_var_names_from_patsy_formula(full_patsy_formula_for_new_data_transform_arg)

        model_data_rm["active_spline_configs"] = {
            var: self.spline_config_details[var].copy()
            for var in original_vars_for_this_model
            if var in self.spline_config_details
        }
        model_data_rm["active_type_configs"] = {
            var: self.covariables_type_config[var]
            for var in original_vars_for_this_model
            if var in self.covariables_type_config
        }
        model_data_rm["original_vars_in_model_list"] = original_vars_for_this_model

        model_data_rm["_df_for_fit_main_INTERNAL_USE"] = df_lifelines_rm.copy()
        model_data_rm["_X_design_rm_INTERNAL_USE"] = X_design_rm.copy()
        model_data_rm["_y_survival_rm_INTERNAL_USE"] = y_survival_rm.copy()

        model_data_rm["metrics"] = compute_model_metrics(
            fitted_cph_model,
            X_design_rm, y_survival_rm, time_col_rm, event_col_rm,
            model_data_rm.get("c_index_cv_mean"),
            model_data_rm.get("c_index_cv_std"),
            model_data_rm.get("schoenfeld_results"),
            model_data_rm.get("loglik_null"),
            self.log
        )
        return model_data_rm

def _extract_original_var_names_from_patsy_formula(patsy_formula_string):
    """
    Extrae los nombres de las variables originales de una cadena de fórmula Patsy
    que usa Q('var_name') para los nombres de las variables.
    """
    if not patsy_formula_string or not isinstance(patsy_formula_string, str):
        return []

    # Regex para encontrar Q('variable_name')
    # Captura lo que está dentro de las comillas simples
    pattern = re.compile(r"Q\('([^']+)'\)")
    matches = pattern.findall(patsy_formula_string)

    # Devolver una lista de nombres únicos
    return sorted(list(set(matches)))


class EditModelDetailsDialog(tk.Toplevel):
    def __init__(self, parent, model_data, app_instance):
        super().__init__(parent)
        self.transient(parent)
        self.grab_set()
        self.title("Editar Nombre y Notas del Modelo")
        self.model_data_ref = model_data # Referencia al diccionario del modelo
        self.app_instance_ref = app_instance # Referencia a la app principal para actualizar Treeview

        main_frame = ttk.Frame(self, padding="15")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Nombre del Modelo
        ttk.Label(main_frame, text="Nombre del Modelo Personalizado:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=(0,2))
        self.model_name_var = tk.StringVar(value=self.model_data_ref.get('custom_model_name', '') or self.model_data_ref.get('model_name', ''))
        self.name_entry = ttk.Entry(main_frame, textvariable=self.model_name_var, width=60)
        self.name_entry.grid(row=1, column=0, sticky=tk.EW, padx=5, pady=(0,10))

        # Notas del Modelo
        ttk.Label(main_frame, text="Notas Adicionales:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=(0,2))
        self.notes_text = scrolledtext.ScrolledText(main_frame, wrap=tk.WORD, height=8, width=60, font=("TkDefaultFont", 9))
        self.notes_text.insert(tk.END, self.model_data_ref.get('model_notes', ''))
        self.notes_text.grid(row=3, column=0, sticky=tk.NSEW, padx=5, pady=(0,10))

        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(3, weight=1)


        # Botones
        buttons_frame = ttk.Frame(main_frame)
        buttons_frame.grid(row=4, column=0, sticky=tk.E, pady=(10,0))
        ttk.Button(buttons_frame, text="Guardar Cambios", command=self._save_changes).pack(side=tk.LEFT, padx=5)
        ttk.Button(buttons_frame, text="Cancelar", command=self.destroy).pack(side=tk.LEFT)

        self.protocol("WM_DELETE_WINDOW", self.destroy)
        self.name_entry.focus_set()
        self.wait_window(self)

    def _save_changes(self):
        new_name = self.model_name_var.get().strip()
        new_notes = self.notes_text.get("1.0", tk.END).strip()

        if not new_name: # Permitir notas vacías, pero no nombre vacío.
            messagebox.showwarning("Nombre Requerido", "El nombre del modelo no puede estar vacío.", parent=self)
            return

        self.model_data_ref['custom_model_name'] = new_name
        self.model_data_ref['model_notes'] = new_notes

        self.app_instance_ref.log(f"Detalles del modelo '{self.model_data_ref.get('model_name')}' actualizados: Nuevo nombre='{new_name}'.", "INFO")

        # Actualizar el Treeview en la aplicación principal
        self.app_instance_ref._update_models_treeview() # Asume que este método refresca usando generated_models_data

        # Re-seleccionar el item editado en el Treeview si es posible
        # Esto requiere encontrar el item_id de nuevo o pasarlo
        # Para simplificar, no se re-seleccionará aquí, pero _update_models_treeview
        # podría manejar la selección si se pasa el model_name original.

        self.destroy()


# --- MÉTODOS PARA PESTAÑA 3: RESULTADOS Y VISUALIZACIÓN ---

class ResultsTabControls:
    def __init__(self, parent_frame, app_instance):
        self.parent_frame = parent_frame
        self.app = app_instance # Referencia a la instancia principal de CoxModelingApp
        self.log = self.app.log # Usar el logger de la app principal

        self.plot_canvas_widget = None
        self.plot_toolbar = None
        self.current_fig = None
        self.current_ax = None
        self.plot_options_for_current_type = {} # Opciones específicas por tipo de gráfico

        self.create_results_display_area()
        self.log("Controles de la Pestaña de Resultados creados.", "DEBUG")

    def create_results_display_area(self):
        content = self.parent_frame # El ScrolledFrame.interior ya es el parent_frame

        # Frame para el área de gráficos
        self.graph_area_frame = ttk.LabelFrame(content, text="Área de Visualización de Gráficos")
        self.graph_area_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Botón para opciones de gráfico (se activa cuando hay un gráfico)
        self.btn_plot_options = ttk.Button(self.graph_area_frame, text="Configurar Opciones del Gráfico Actual...",
                                           command=self._open_plot_options_dialog, state=tk.DISABLED)
        self.btn_plot_options.pack(pady=(5,0), padx=5, anchor=tk.NE)

        # Placeholder para el canvas del gráfico
        self.canvas_placeholder = ttk.Frame(self.graph_area_frame, height=400) # Altura inicial
        self.canvas_placeholder.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        ttk.Label(self.canvas_placeholder, text="Aquí se mostrarán los gráficos generados.", style="Italic.TLabel").pack(padx=10, pady=10, expand=True)

        # Estilo para el texto itálico
        style = ttk.Style()
        style.configure("Italic.TLabel", font=("TkDefaultFont", 9, "italic"))

    def _open_plot_options_dialog(self):
        if self.current_fig is None or self.current_ax is None:
            self.log("No hay gráfico activo para configurar opciones.", "WARN")
            messagebox.showwarning("Sin Gráfico", "No hay ningún gráfico activo para configurar.", parent=self.app.parent_for_dialogs)
            return

        # Determinar el tipo de gráfico actual para cargar/guardar opciones específicas
        # Esto es un placeholder, se necesitará una forma de saber el tipo de gráfico actual
        current_plot_type_key = getattr(self.current_fig, '_plot_type_key', 'default_plot')

        dialog = PlotOptionsDialog(
            self.app.parent_for_dialogs,
            current_options=self.plot_options_for_current_type.get(current_plot_type_key, {}),
            apply_callback=lambda opts: self._apply_and_save_plot_options(opts, current_plot_type_key)
        )

    def _apply_and_save_plot_options(self, options, plot_type_key):
        if self.current_ax and self.current_fig:
            apply_plot_options(self.current_ax, options, log_func=self.log)
            self.plot_options_for_current_type[plot_type_key] = options.copy() # Guardar opciones
            self.log(f"Opciones de gráfico aplicadas y guardadas para tipo '{plot_type_key}'.", "INFO")
            if self.plot_canvas_widget:
                self.plot_canvas_widget.draw_idle()
        else:
            self.log("Intento de aplicar opciones sin gráfico activo.", "WARN")

    def display_figure(self, fig, plot_type_key='default_plot'):
        if fig is None:
            self.log("Intento de mostrar figura nula.", "WARN")
            self.clear_plot_area()
            ttk.Label(self.canvas_placeholder, text="Error: La figura a mostrar es nula.", style="Error.TLabel").pack(padx=10, pady=10, expand=True)
            ttk.Style().configure("Error.TLabel", foreground="red")
            return

        self.clear_plot_area() # Limpiar área antes de mostrar nueva figura
        self.current_fig = fig

        # Asumir que la figura tiene al menos un eje, o el primero es el relevante
        self.current_ax = fig.get_axes()[0] if fig.get_axes() else None

        # Guardar el tipo de gráfico para opciones persistentes
        # Esto es un atributo ad-hoc, podría ser mejor gestionado si se conoce el origen del gráfico
        setattr(self.current_fig, '_plot_type_key', plot_type_key)

        self.plot_canvas_widget = FigureCanvasTkAgg(fig, master=self.canvas_placeholder)
        canvas_native_widget = self.plot_canvas_widget.get_tk_widget()
        canvas_native_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Crear barra de herramientas para el nuevo canvas
        # Destruir la anterior si existe para evitar duplicados o errores
        if self.plot_toolbar:
            self.plot_toolbar.destroy()

        self.plot_toolbar = NavigationToolbar2Tk(self.plot_canvas_widget, self.canvas_placeholder, pack_toolbar=False)
        self.plot_toolbar.update()
        self.plot_toolbar.pack(side=tk.BOTTOM, fill=tk.X)

        self.plot_canvas_widget.draw()
        self.btn_plot_options.config(state=tk.NORMAL) # Habilitar botón de opciones
        self.log(f"Figura mostrada en la pestaña de resultados (Tipo: {plot_type_key}).", "INFO")

    def clear_plot_area(self):
        # Limpiar la figura y el canvas actual
        if self.current_fig:
            plt.close(self.current_fig) # Cerrar figura de matplotlib para liberar memoria
            self.current_fig = None
            self.current_ax = None

        if self.plot_canvas_widget:
            self.plot_canvas_widget.get_tk_widget().destroy()
            self.plot_canvas_widget = None

        if self.plot_toolbar:
            self.plot_toolbar.destroy()
            self.plot_toolbar = None

        # Limpiar cualquier widget hijo del placeholder (como etiquetas de error/info)
        for widget in self.canvas_placeholder.winfo_children():
            if widget != self.btn_plot_options : # No destruir el botón de opciones si está dentro del placeholder
                 widget.destroy()

        self.btn_plot_options.config(state=tk.DISABLED) # Deshabilitar botón de opciones
        self.log("Área de gráficos limpiada.", "DEBUG")

    def update_for_new_results(self):
        # Este método podría ser llamado cuando se genera un nuevo modelo o se selecciona uno
        # para limpiar el área de gráficos si es necesario.
        # Por ahora, la limpieza se hace en display_figure.
        pass


    def create_results_controls(self):
        """Crea los controles para la Pestaña 3: Resultados y Visualización."""
        # El ScrolledFrame.interior ya está asignado a self.parent_frame_content
        # y los controles se añaden directamente allí por ResultsTabControls.
        # Esta función es un stub si la creación principal se hace en __init__ de ResultsTabControls.
        self.app.results_tab_manager = ResultsTabControls(self.app.tab_frame_results_content.interior, self.app)
        self.log("Gestor de controles de la Pestaña de Resultados (ResultsTabControls) instanciado.", "DEBUG")


# --- INICIO DE LA APLICACIÓN ---
def main():
    # Asegurar que el script se ejecuta desde el directorio correcto
    # para que las importaciones relativas funcionen si es necesario.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)

    # Intentar importar MATLAB_filter_component aquí también si es parte del paquete
    # y no se encuentra en el path global de Python.
    # (Esto ya se maneja con try-except en las importaciones globales)

    root = tk.Tk()
    root.title("Herramienta Avanzada de Modelado Cox y Supervivencia")

    # Configurar tamaño inicial y permitir redimensionamiento
    root.geometry("1150x850")
    root.minsize(900, 700)

    # Estilo ttk
    style = ttk.Style(root)
    try:
        # Intentar usar un tema más moderno si está disponible
        available_themes = style.theme_names()
        # Preferencias de temas (pueden variar por OS)
        preferred_themes = ['clam', 'alt', 'default', 'vista', 'xpnative']
        for theme in preferred_themes:
            if theme in available_themes:
                style.theme_use(theme)
                break
    except tk.TclError:
        print("INFO: No se pudo aplicar un tema ttk preferido, usando default del sistema.")

    # Crear una instancia de la aplicación principal
    app = CoxModelingApp(root)

    # Configurar el cierre de la ventana
    root.protocol("WM_DELETE_WINDOW", lambda: on_closing(root, app))

    root.mainloop()

def on_closing(root_window, app_instance):
    if messagebox.askokcancel("Salir", "¿Está seguro que desea salir de la aplicación?", parent=root_window):
        app_instance.log("Cerrando la aplicación...", "INFO")
        # Aquí se podrían añadir acciones de limpieza si fueran necesarias
        root_window.destroy()

if __name__ == "__main__":
    main()
