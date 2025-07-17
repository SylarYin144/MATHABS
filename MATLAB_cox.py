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
from statsmodels.stats.outliers_influence import variance_inflation_factor
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
        # {cuant_var_name: {'type': 'Natural'|'B-spline', 'df': int}}
        self.spline_config_details = {}
        self.current_plot_options = {}  # Diccionario para guardar opciones de gráficos

        # Variables para modelos
        # Lista de diccionarios, cada uno con datos de un modelo
        self.generated_models_data = []
        # Diccionario del modelo seleccionado en la Treeview
        self.selected_model_in_treeview = None
        self.btn_oos_calibration = None
        self.btn_collinearity_diag = None # <-- NUEVO
        self.entry_custom_model_name = None # Placeholder for custom name Entry
        self.text_custom_model_notes = None # Placeholder for custom notes Text

        # Variables de control para la UI (Pestaña 2: Modelado)
        self.cox_model_type_var = StringVar(value="Multivariado")  # "Univariado" | "Multivariado"
        self.generate_univariate_forest_plot_var = BooleanVar(value=True) # <-- NUEVO
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
        self.calculate_cv_cindex_var = BooleanVar(value=True)  # Calcular C-Index por CV
        self.cv_num_kfolds_var = IntVar(value=5)  # Número de folds para CV
        self.cv_random_seed_var = IntVar(value=42)  # Semilla aleatoria para CV
        self.covariate_scaling_method_var = StringVar(value="Ninguna")

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
        self.log(f"Current spline_config_details after apply: {self.spline_config_details}", "DEBUG")


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
        ttk.Label(frame_tipo_modelado, text="Tipo de Modelado:").grid(row=0, column=0, padx=(0,5), pady=3, sticky=tk.W)

        radio_multi = ttk.Radiobutton(frame_tipo_modelado, text="Multivariado", variable=self.cox_model_type_var, value="Multivariado", command=self._toggle_univariate_forest_plot_cb)
        radio_multi.grid(row=0, column=1, padx=3, pady=3, sticky=tk.W)

        radio_uni = ttk.Radiobutton(frame_tipo_modelado, text="Univariado", variable=self.cox_model_type_var, value="Univariado", command=self._toggle_univariate_forest_plot_cb)
        radio_uni.grid(row=0, column=2, padx=3, pady=3, sticky=tk.W)

        # Checkbox para Forest Plot univariado
        self.cb_univariate_forest_plot = ttk.Checkbutton(frame_tipo_modelado, text="Generar Forest Plot de Univariados", variable=self.generate_univariate_forest_plot_var)
        self.cb_univariate_forest_plot.grid(row=1, column=1, columnspan=2, padx=20, pady=(5,0), sticky=tk.W)


        # Selección de Variables
        self.frame_sel_vars = ttk.LabelFrame(left_col_frame, text="Selección de Variables (para Multivariado)")
        self.frame_sel_vars.pack(fill=tk.X, expand=True, pady=(0,10))
        
        grid_sel_vars = ttk.Frame(self.frame_sel_vars, padding=5)
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
        
        cols_tv = ("#", "Nombre Modelo", "Variables y Splines", "AIC", "-2 LogLik",
                   "C-Index (Train)", "C-Index (CV)", "Schoenfeld (p min)", "Wald (p max)")
        self.treeview_lista_modelos = ttk.Treeview(self.frame_modelos_generados_display, columns=cols_tv, show="headings", height=7)
        self.treeview_sort_reversed = {} # Para guardar el estado de ordenamiento por columna

        col_widths = {
            "#": 30,
            "Nombre Modelo": 180,
            "Variables y Splines": 250,
            "AIC": 80,
            "-2 LogLik": 80,
            "C-Index (Train)": 100,
            "C-Index (CV)": 90,
            "Schoenfeld (p min)": 110,
            "Wald (p max)": 90
        }
        col_anchors = {
            "#": tk.CENTER,
            "AIC": tk.E,
            "-2 LogLik": tk.E,
            "C-Index (Train)": tk.E,
            "C-Index (CV)": tk.E,
            "Schoenfeld (p min)": tk.E,
            "Wald (p max)": tk.E
        }
        for col in cols_tv:
            # Usamos una función lambda que captura el valor de `col` en el momento de la definición
            self.treeview_lista_modelos.heading(col, text=col, command=lambda c=col: self._sort_treeview_column(c))
            self.treeview_lista_modelos.column(col, width=col_widths.get(col, 120),
                                               anchor=col_anchors.get(col, tk.W),
                                               minwidth=max(30, col_widths.get(col, 50)//2))
            self.treeview_sort_reversed[col] = False
            
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
            ("Generar Gráficos Cox", self.open_graph_selection_dialog),
            ("Calibración OOS (CV)", self.show_new_calibration_plots),
            ("Diagnóstico de Colinealidad", self._calculate_and_show_vif), # <-- NUEVO
            ("Predicción", self.realizar_prediccion),
            ("Exportar Resumen", self.export_model_summary),
            ("Guardar Modelo", self.save_model),
            ("Cargar Modelo", self.load_model_from_file),
            ("Reporte Metod.", self.show_methodological_report)
        ]
        
        # Layout dinámico para botones de acción
        max_btns_per_row = 5 # Ajustado para mejor layout
        current_row_frame_acciones = None
        for i, (text, cmd) in enumerate(acciones_config_btns):
            if i % max_btns_per_row == 0:
                current_row_frame_acciones = ttk.Frame(frame_acciones)
                current_row_frame_acciones.pack(fill=tk.X, pady=1)

            button_widget = ttk.Button(current_row_frame_acciones, text=text, command=cmd)
            button_widget.pack(side=tk.LEFT, padx=3, pady=2, fill=tk.X, expand=True)

            if text == "Calibración OOS (CV)":
                self.btn_oos_calibration = button_widget
            elif text == "Diagnóstico de Colinealidad": # <-- NUEVO
                self.btn_collinearity_diag = button_widget

        if self.btn_oos_calibration:
            self.btn_oos_calibration.config(state=tk.DISABLED)
        if self.btn_collinearity_diag: # <-- NUEVO
            self.btn_collinearity_diag.config(state=tk.DISABLED)

        # Add the new button row for clear models
        clear_models_frame = ttk.Frame(frame_acciones)
        clear_models_frame.pack(fill=tk.X, pady=5)
        ttk.Button(clear_models_frame, text="Limpiar Todos los Modelos", command=self._clear_all_generated_models).pack(side=tk.RIGHT, padx=5)

        # UI para Nombre Personalizado y Notas del Modelo Seleccionado
        frame_custom_details = ttk.LabelFrame(self.frame_modelos_generados_display, text="Detalles Personalizados del Modelo Seleccionado", padding=10)
        frame_custom_details.pack(fill=tk.X, padx=5, pady=(10,5))

        # Nombre Personalizado
        ttk.Label(frame_custom_details, text="Nombre Personalizado:").grid(row=0, column=0, padx=5, pady=3, sticky=tk.W)
        self.entry_custom_model_name_var = StringVar()
        self.entry_custom_model_name = ttk.Entry(frame_custom_details, textvariable=self.entry_custom_model_name_var, width=50)
        self.entry_custom_model_name.grid(row=0, column=1, padx=5, pady=3, sticky=tk.EW)

        # Notas
        ttk.Label(frame_custom_details, text="Notas:").grid(row=1, column=0, padx=5, pady=3, sticky=tk.NW)
        self.text_custom_model_notes = scrolledtext.ScrolledText(frame_custom_details, width=60, height=3, wrap=tk.WORD, font=("TkDefaultFont", 9))
        self.text_custom_model_notes.grid(row=1, column=1, padx=5, pady=3, sticky=tk.EW)

        frame_custom_details.columnconfigure(1, weight=1)

        ttk.Button(frame_custom_details, text="Guardar Nombre/Notas", command=self._save_custom_model_details).grid(row=2, column=1, padx=5, pady=5, sticky=tk.E)


        self.log("Controles de Modelado Cox creados.", "DEBUG")
        self._toggle_penalization_params_ui_state() # Estado inicial de UI de penalización
        self._toggle_univariate_forest_plot_cb() # Estado inicial del checkbox de Forest Plot

    def _toggle_univariate_forest_plot_cb(self):
        """Habilita o deshabilita el checkbox de Forest Plot univariado."""
        if hasattr(self, 'cb_univariate_forest_plot'):
            is_univariate = self.cox_model_type_var.get() == "Univariado"
            self.cb_univariate_forest_plot.config(state=tk.NORMAL if is_univariate else tk.DISABLED)
            # También controla la visibilidad del frame de selección de variables
            if hasattr(self, 'frame_sel_vars'):
                 self.frame_sel_vars.config(text="Selección de Variables" if is_univariate else "Selección de Variables (para Multivariado)")


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
            
            term_syntax_bd = f"Q('{orig_cov_name_bd}')" 
            if config_type_bd == "Cuantitativa":
                if orig_cov_name_bd in self.spline_config_details:
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
                    else: # Fallback for unknown spline type
                        term_syntax_bd = f"Q('{orig_cov_name_bd}')"
                        self.log(f"WARN: Tipo de spline desconocido '{spline_type}' para '{orig_cov_name_bd}'. Tratada como cuantitativa normal.", "WARN")
                else: # No spline config for this quantitative var
                    term_syntax_bd = f"Q('{orig_cov_name_bd}')"
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
        
        ui_selected_tie_method = self.tie_handling_method_var.get() # Para registro
        
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
            "scaled_columns_info": scaled_columns_info if scaled_columns_info is not None else [],
            "custom_model_name": model_name_rm, # Inicializar con el nombre generado
            "custom_model_notes": "" # Inicializar notas vacías
        }

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
        if df_for_fit_main.empty:
            self.log(f"FALLO DE AJUSTE DEL MODELO: '{model_name_rm}'. El DataFrame para el ajuste está vacío.", "ERROR")
            # model_data_rm["model"] is already None
        elif X_design_rm.empty and actual_formula_for_fit != "0":
            self.log(f"FALLO DE AJUSTE DEL MODELO: '{model_name_rm}'. X_design está vacío pero la fórmula no es nula ('{actual_formula_for_fit}').", "ERROR")
            # model_data_rm["model"] is already None
        else:
            try:
                cph_main_rm_instance.fit(df_for_fit_main, duration_col=time_col_rm, event_col=event_col_rm, formula=actual_formula_for_fit)
                model_data_rm["model"] = cph_main_rm_instance
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
                # model_data_rm["model"] remains None
            except np.linalg.LinAlgError as e_linalg:
                num_obs_fail = df_for_fit_main.shape[0]
                num_events_fail = df_for_fit_main[event_col_rm].sum() if event_col_rm in df_for_fit_main.columns else 'N/A'
                self.log(f"FALLO DE AJUSTE DEL MODELO (LinAlgError - ej. Matriz Singular): '{model_name_rm}'", "ERROR")
                self.log(f"  Error específico: {e_linalg}", "ERROR")
                self.log(f"  Observaciones usadas: {num_obs_fail}, Eventos: {num_events_fail}", "ERROR")
                if "cr(" in actual_formula_for_fit:
                    self.log("  ADVERTENCIA ADICIONAL: El modelo incluía splines naturales (cr()). Estos pueden causar problemas de colinealidad. Considere usar B-splines (bs()) o reducir los grados de libertad (df).", "WARN")
                traceback.print_exc(limit=2)
                # model_data_rm["model"] remains None
            except Exception as e_fit_main:
                num_obs_fail = df_for_fit_main.shape[0] if isinstance(df_for_fit_main, pd.DataFrame) else 'N/A'
                num_events_fail = (df_for_fit_main[event_col_rm].sum() if isinstance(df_for_fit_main, pd.DataFrame) and event_col_rm in df_for_fit_main.columns else 'N/A')
                self.log(f"FALLO DE AJUSTE DEL MODELO (Error General e Inesperado): '{model_name_rm}'", "ERROR")
                self.log(f"  Error específico: {e_fit_main}", "ERROR")
                if num_obs_fail != 'N/A':
                    self.log(f"  Observaciones (si disponibles): {num_obs_fail}, Eventos (si disponibles): {num_events_fail}", "ERROR")
                traceback.print_exc(limit=3)
                # model_data_rm["model"] remains None

        # 4. Post-Fit Operations
        fitted_cph_model = model_data_rm.get("model")

        if fitted_cph_model:
            # Test de Schoenfeld
            if not X_design_rm.empty:
                if hasattr(fitted_cph_model, 'params_') and fitted_cph_model.params_ is not None and not fitted_cph_model.params_.empty:
                    self.log(f"--- Iniciando Test de Schoenfeld para Modelo: '{model_name_rm}' ---", "INFO")
                    try:
                        results_check_assumptions = fitted_cph_model.check_assumptions(df_for_fit_main)
                        model_data_rm["check_assumptions_results_raw"] = results_check_assumptions
                        schoenfeld_df_candidate = None
                        found_schoenfeld_results = False
                        if isinstance(results_check_assumptions, list) and results_check_assumptions:
                            for i, item in enumerate(results_check_assumptions):
                                if isinstance(item, pd.DataFrame) and not item.empty and all(col in item.columns for col in ['test_statistic', 'p']):
                                    schoenfeld_df_candidate = item
                                    found_schoenfeld_results = True
                                    break
                                elif hasattr(item, 'summary') and isinstance(item.summary, pd.DataFrame) and not item.summary.empty and all(col in item.summary.columns for col in ['test_statistic', 'p']):
                                    schoenfeld_df_candidate = item.summary
                                    found_schoenfeld_results = True
                                    break
                            if not found_schoenfeld_results and len(results_check_assumptions) >= 2 and isinstance(results_check_assumptions[1], pd.DataFrame) and all(col in results_check_assumptions[1].columns for col in ['test_statistic', 'p']):
                                 schoenfeld_df_candidate = results_check_assumptions[1]
                                 found_schoenfeld_results = True

                        if found_schoenfeld_results and schoenfeld_df_candidate is not None:
                            model_data_rm["schoenfeld_results"] = schoenfeld_df_candidate.copy()
                            model_data_rm["schoenfeld_status_message"] = "Schoenfeld (check_assumptions) calculado exitosamente."
                            self.log(f"Schoenfeld data for '{model_name_rm}' obtained from check_assumptions.", "DEBUG")
                        else:
                            model_data_rm["schoenfeld_status_message"] = "Schoenfeld (check_assumptions): resultados detallados no encontrados o en formato inesperado."
                            self.log(f"Schoenfeld data for '{model_name_rm}' from check_assumptions was empty or not found.", "DEBUG")
                    except Exception as e_sch_detailed:
                        model_data_rm["schoenfeld_status_message"] = "Error durante Test de Schoenfeld (check_assumptions)."
                        self.log(f"ERROR en Test Schoenfeld (check_assumptions) para '{model_name_rm}': {e_sch_detailed}\n{traceback.format_exc()}", "ERROR")
                    self.log(f"--- Test de Schoenfeld (check_assumptions) para Modelo: '{model_name_rm}' Finalizado. Status: {model_data_rm['schoenfeld_status_message']} ---", "INFO")
                else: # No parameters in model
                    self.log(f"INFO: Modelo '{model_name_rm}' sin parámetros. Test de Schoenfeld (check_assumptions) no aplicable.", "INFO")
                    model_data_rm["schoenfeld_status_message"] = "Test de Schoenfeld (check_assumptions) no aplicable (modelo sin covariables)."
            else: # X_design_rm is empty (null model)
                self.log(f"INFO: Modelo nulo '{model_name_rm}' (X_design_rm vacío). Test de Schoenfeld (check_assumptions) no aplicable.", "INFO")
                model_data_rm["schoenfeld_status_message"] = "Test de Schoenfeld (check_assumptions) no aplicable (modelo nulo)."

            # Fallback or supplement with proportional_hazard_test
            # This test provides a summary table which might be what `schoenfeld_results` is expected to be
            # if check_assumptions doesn't yield the detailed residuals table in the expected format.
            schoenfeld_df_from_check_assumptions = model_data_rm.get("schoenfeld_results")

            # Condition to try proportional_hazard_test:
            # 1. Model has parameters
            # 2. EITHER schoenfeld_results from check_assumptions is empty/None
            #    OR it doesn't seem to contain individual p-values (e.g. only global test or wrong format)
            # For simplicity, we'll try it if check_assumptions didn't yield a non-empty DataFrame with a 'p' column.
            should_try_ph_test = False
            if hasattr(fitted_cph_model, 'params_') and fitted_cph_model.params_ is not None and not fitted_cph_model.params_.empty:
                if schoenfeld_df_from_check_assumptions is None or schoenfeld_df_from_check_assumptions.empty or 'p' not in schoenfeld_df_from_check_assumptions.columns:
                    should_try_ph_test = True

            if should_try_ph_test:
                self.log(f"INFO: `schoenfeld_results` de `check_assumptions` para '{model_name_rm}' está vacío o no tiene columna 'p'. Intentando `proportional_hazard_test` como fuente alternativa/suplementaria.", "INFO")
                try:
                    from lifelines.statistics import proportional_hazard_test
                    ph_test_results_obj = proportional_hazard_test(fitted_cph_model, df_for_fit_main, time_transform='log')

                    if ph_test_results_obj is not None and hasattr(ph_test_results_obj, 'summary') and \
                       isinstance(ph_test_results_obj.summary, pd.DataFrame) and not ph_test_results_obj.summary.empty and \
                       'p' in ph_test_results_obj.summary.columns:

                        # Store the summary from proportional_hazard_test
                        model_data_rm["proportional_hazard_test_summary"] = ph_test_results_obj.summary.copy()
                        self.log(f"INFO: `proportional_hazard_test` para '{model_name_rm}' proporcionó un resumen con p-valores.", "INFO")

                        # If original schoenfeld_results was empty/missing 'p', replace it with this summary
                        if schoenfeld_df_from_check_assumptions is None or schoenfeld_df_from_check_assumptions.empty or 'p' not in schoenfeld_df_from_check_assumptions.columns:
                            model_data_rm["schoenfeld_results"] = ph_test_results_obj.summary.copy()
                            model_data_rm["schoenfeld_status_message"] = "Resultados de Schoenfeld obtenidos de proportional_hazard_test (summary)."
                            self.log(f"INFO: `schoenfeld_results` para '{model_name_rm}' ahora utiliza el resumen de `proportional_hazard_test`.", "INFO")
                        else:
                            model_data_rm["schoenfeld_status_message"] += " `proportional_hazard_test` también proporcionó un resumen."
                    else:
                        model_data_rm["schoenfeld_status_message"] += " `proportional_hazard_test` no arrojó un resumen con p-valores utilizables."
                        self.log(f"INFO: `proportional_hazard_test` para '{model_name_rm}' no produjo un resumen con p-valores.", "INFO")
                except Exception as e_ph_test_fallback:
                    self.log(f"ERROR en `proportional_hazard_test` para '{model_name_rm}': {e_ph_test_fallback}", "ERROR")
                    model_data_rm["schoenfeld_status_message"] += f" (Error en proportional_hazard_test: {str(e_ph_test_fallback)[:50]}...)."

            # C-Index CV
            if self.calculate_cv_cindex_var.get() and not X_design_rm.empty:
                self.log(f"Iniciando cálculo de C-Index CV para '{model_name_rm}'.", "INFO")
                try:
                    kf_cv = KFold(n_splits=self.cv_num_kfolds_var.get(), shuffle=True, random_state=self.cv_random_seed_var.get())
                    c_indices_cv_list = []
                    all_oos_predictions_data_cv = []

                    # --- Prepare formula for CV with explicit knots for B-splines ---
                    cv_patsy_formula = actual_formula_for_fit # Start with the main model's formula

                    # Find B-spline terms: bs(Q('var'), df=X, degree=Y)
                    # We need original variable names to get data for knot calculation from df_lifelines_rm
                    # and the df/degree to calculate knots.

                    # Regex to find bs terms and capture var, df, degree
                    # Example: bs(Q('Rep_CAG'), df=3, degree=3)
                    # Need to handle spaces carefully.
                    bs_pattern = re.compile(r"bs\(Q\('([^']+)'\),\s*df=(\d+),\s*degree=(\d+)\)")

                    modified_bs_terms = {} # Store original term -> new term with knots

                    for match in bs_pattern.finditer(actual_formula_for_fit):
                        original_term = match.group(0)
                        var_name = match.group(1)
                        df = int(match.group(2))
                        degree = int(match.group(3))

                        self.log(f"CV C-Index: Found B-spline term for '{var_name}': {original_term}", "DEBUG")

                        if var_name in df_lifelines_rm.columns:
                            data_series = df_lifelines_rm[var_name].dropna()
                            if len(data_series) < (degree + 1) or data_series.nunique() < 2 : # Not enough data or unique values for knots
                                self.log(f"CV C-Index: Insufficient data or unique values for '{var_name}' to define B-spline knots. Skipping knot modification for this term.", "WARN")
                                continue

                            # Calculate number of interior knots: df - degree - 1 (Patsy's default for df interpretation)
                            # However, an easier way to ensure consistency is to specify enough knots for 'df' basis functions.
                            # A common interpretation: df = number of knots (including boundary) + degree - 1
                            # Or, if df is number of basis functions, number of knots = df - degree + 1 (for B-splines, including boundary knots)
                            # Patsy's `df` for `bs` means "produce a basis matrix with this many columns".
                            # The number of interior knots is `df - degree - 1`.
                            num_interior_knots = df - degree - 1

                            knots = []
                            if num_interior_knots >= 0:
                                # Define percentiles for interior knots if num_interior_knots > 0
                                if num_interior_knots > 0:
                                    percentiles = np.linspace(0, 100, num_interior_knots + 2)[1:-1]
                                    knots = np.percentile(data_series, percentiles).tolist()
                                    self.log(f"CV C-Index: Calculated {len(knots)} interior knots for '{var_name}' (df={df}, degree={degree}): {knots}", "DEBUG")
                                else: # num_interior_knots == 0
                                    self.log(f"CV C-Index: Using boundary knots only for '{var_name}' (df={df}, degree={degree}, num_interior_knots=0).", "DEBUG")
                                    knots = [] # Patsy will use boundary knots

                                # Construct new bs term with explicit knots
                                new_bs_term = f"bs(Q('{var_name}'), knots={knots}, degree={degree}, include_intercept=False)"
                                modified_bs_terms[original_term] = new_bs_term
                            else: # df <= degree
                                self.log(f"CV C-Index: Configuration for bs(Q('{var_name}'), df={df}, degree={degree}) has df <= degree. Skipping explicit knot generation. Patsy will handle it.", "WARN")
                        else:
                            self.log(f"CV C-Index: Variable '{var_name}' for B-spline not found in df_lifelines_rm. Cannot calculate knots.", "WARN")

                    # Replace terms in the CV formula
                    if modified_bs_terms:
                        temp_cv_formula = cv_patsy_formula
                        for old, new in modified_bs_terms.items():
                            temp_cv_formula = temp_cv_formula.replace(old, new)
                        cv_patsy_formula = temp_cv_formula
                        self.log(f"CV C-Index: Modified formula for CV folds with explicit knots: {cv_patsy_formula}", "INFO")
                    else:
                        self.log(f"CV C-Index: No B-spline terms were modified with explicit knots for CV formula. Using original: {cv_patsy_formula}", "INFO")
                    # --- End of CV formula preparation ---


                    self.log(f"CV C-Index for '{model_name_rm}': Starting KFold loop. Formula for folds: {cv_patsy_formula}", "DEBUG")

                    for i_fold, (train_idx, test_idx) in enumerate(kf_cv.split(df_lifelines_rm)):
                        self.log(f"CV C-Index Fold {i_fold+1}/{kf_cv.get_n_splits()}: Processing...", "DEBUG")
                        df_fold_for_fit_cv = df_lifelines_rm.iloc[train_idx].copy()
                        df_fold_for_pred_cv = df_lifelines_rm.iloc[test_idx].copy() # Data for prediction
                        y_te_cv = y_survival_rm.iloc[test_idx] # True outcomes for test set

                        if df_fold_for_fit_cv.empty or y_te_cv.empty or df_fold_for_pred_cv.empty:
                            self.log(f"CV C-Index Fold {i_fold+1}: Data empty for train or test. Skipping fold.", "WARN")
                            continue
                        
                        self.log(f"CV C-Index Fold {i_fold+1}: Training data shape: {df_fold_for_fit_cv.shape}", "DEBUG")
                        self.log(f"CV C-Index Fold {i_fold+1}: Test data shape for prediction: {df_fold_for_pred_cv.shape}", "DEBUG")

                        cph_fold_cv = CoxPHFitter(penalizer=penalizer_val_rm, l1_ratio=l1_ratio_val_rm)
                        try:
                            self.log(f"CV C-Index Fold {i_fold+1}: Attempting to fit with formula: {cv_patsy_formula}", "DEBUG") # Use cv_patsy_formula
                            cph_fold_cv.fit(df_fold_for_fit_cv, duration_col=time_col_rm, event_col=event_col_rm, formula=cv_patsy_formula) # Use cv_patsy_formula
                            self.log(f"CV C-Index Fold {i_fold+1}: Fit successful.", "DEBUG")

                            preds_te_fold_cv = cph_fold_cv.predict_partial_hazard(df_fold_for_pred_cv)
                            c_idx_fold_cv = concordance_index(y_te_cv[time_col_rm], -preds_te_fold_cv, y_te_cv[event_col_rm])
                            c_indices_cv_list.append(c_idx_fold_cv)
                            self.log(f"CV C-Index Fold {i_fold+1}: Calculated C-Index: {c_idx_fold_cv:.4f}", "DEBUG")

                            # OOS predictions for this fold (if needed later for calibration etc.)
                            try:
                                oos_sf_fold_cv = cph_fold_cv.predict_survival_function(df_fold_for_pred_cv)
                                for subj_orig_idx_cv in df_fold_for_pred_cv.index:
                                    all_oos_predictions_data_cv.append({
                                        "subject_id": subj_orig_idx_cv, # This is original index from df_lifelines_rm
                                        "true_time": y_te_cv.loc[subj_orig_idx_cv, time_col_rm],
                                        "true_event": y_te_cv.loc[subj_orig_idx_cv, event_col_rm],
                                        "predicted_survival_function": oos_sf_fold_cv[subj_orig_idx_cv]
                                    })
                            except Exception as e_pred_sf_cv_loop:
                                self.log(f"CV C-Index Fold {i_fold+1}: Error predicting OOS SF: {e_pred_sf_cv_loop}", "WARN")

                        except ConvergenceError as e_conv_fold:
                            self.log(f"CV C-Index Fold {i_fold+1}: FIT FAILED (ConvergenceError). Error: {e_conv_fold}", "ERROR")
                            self.log(f"CV C-Index Fold {i_fold+1}: Traceback:\n{traceback.format_exc(limit=2)}", "DEBUG")
                        except Exception as e_fold_fit:
                            self.log(f"CV C-Index Fold {i_fold+1}: FIT FAILED (Other Error). Error: {e_fold_fit}", "ERROR")
                            self.log(f"CV C-Index Fold {i_fold+1}: Traceback:\n{traceback.format_exc(limit=2)}", "DEBUG")

                    self.log(f"CV C-Index for '{model_name_rm}': Completed KFold loop. Collected C-Indices: {c_indices_cv_list}", "DEBUG")
                    if c_indices_cv_list:
                        model_data_rm["c_index_cv_mean"] = np.mean(c_indices_cv_list)
                        model_data_rm["c_index_cv_std"] = np.std(c_indices_cv_list)
                        self.log(f"C-Index CV for '{model_name_rm}' ({len(c_indices_cv_list)}/{kf_cv.get_n_splits()} folds successful): Mean={model_data_rm['c_index_cv_mean']:.3f} (DE={model_data_rm['c_index_cv_std']:.3f})", "INFO")
                    else:
                        self.log(f"C-Index CV for '{model_name_rm}': No C-Indices calculated from any fold.", "WARN")
                        model_data_rm["c_index_cv_mean"] = None # Ensure it's None if list is empty
                        model_data_rm["c_index_cv_std"] = None

                    if all_oos_predictions_data_cv: # Check if list is populated
                        model_data_rm["oos_predictions"] = all_oos_predictions_data_cv # Assign to the correct key
                        self.log(f"Almacenadas {len(all_oos_predictions_data_cv)} predicciones OOS de CV para '{model_name_rm}'.", "INFO")
                    else:
                        model_data_rm["oos_predictions"] = None # Ensure it's None if list is empty

                except Exception as e_cv_rm_main: # Catch errors in KFold setup or outer loop logic
                    self.log(f"Error general en C-Index CV (fuera del bucle de folds) para '{model_name_rm}': {e_cv_rm_main}", "ERROR")
                    traceback.print_exc(limit=3)
            elif self.calculate_cv_cindex_var.get(): # This means X_design_rm was empty (null model)
                 self.log(f"C-Index CV no calculado para '{model_name_rm}' (modelo nulo o X_design vacío).", "INFO")
        else: # model_data_rm["model"] is None (fit failed)
            self.log(f"Ajuste del modelo '{model_name_rm}' falló. Omitiendo tests de Schoenfeld y C-Index CV.", "WARN")
            model_data_rm["schoenfeld_status_message"] = "No aplicable (fallo en ajuste de modelo)."
            model_data_rm["c_index_cv_mean"] = None
            model_data_rm["c_index_cv_std"] = None
            model_data_rm["oos_predictions"] = None

        # 5. Store internal data copies
        model_data_rm["_df_for_fit_main_INTERNAL_USE"] = df_lifelines_rm.copy()
        model_data_rm["_X_design_rm_INTERNAL_USE"] = X_design_rm.copy() 
        model_data_rm["_y_survival_rm_INTERNAL_USE"] = y_survival_rm.copy() 

        # 6. Calculate and Store Final Metrics
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

    def _update_models_treeview(self):
        self.treeview_lista_modelos.delete(*self.treeview_lista_modelos.get_children())
        if not self.generated_models_data:
            self.log("No hay modelos para mostrar.", "INFO")
            return

        for i, md_tv in enumerate(self.generated_models_data):
            # Nombre Modelo (custom o original)
            name_tv = md_tv.get('custom_model_name', md_tv.get('model_name', f"Modelo {i+1}"))

            # Variables y Splines
            covs_processed = md_tv.get('covariates_processed', []) # Estos son los términos de Patsy
            original_covs_in_model = []
            # Extraer nombres originales de Q('var_name') o términos directos
            for term in covs_processed:
                match_q = re.search(r"Q\('([^']+)'\)", term)
                if match_q:
                    original_covs_in_model.append(match_q.group(1))
                elif not any(x in term for x in ['cr(', 'bs(', 'C(', ' ट्रीटमेंट(', ' Treatment(', 'Intercept']): # Evitar funciones de Patsy
                    original_covs_in_model.append(term)

            original_covs_in_model = sorted(list(set(original_covs_in_model))) # Únicos y ordenados

            vars_splines_display_list = []
            model_spline_configs = md_tv.get('spline_config_details', {}) # Spline configs usadas por este modelo

            # Recuperar la configuración de splines que estaba vigente AL MOMENTO del ajuste del modelo.
            # Esta información debería estar almacenada en el diccionario del modelo `md_tv`.
            # Asumimos que `md_tv` contiene una clave como `spline_config_details_at_fit_time`
            # o que `md_tv.get('formula_patsy')` ya refleja la configuración de splines.
            # Para simplificar, vamos a iterar sobre `original_covs_in_model` y buscar su config
            # en `self.spline_config_details` (config global actual) O en una clave específica del modelo si existiera.
            # El plan indica usar `spline_config_details` de `md_tv`.
            # El `formula_patsy` en `md_tv` ya tiene la info de splines incorporada en su sintaxis.

            # Vamos a intentar reconstruir la info de splines a partir de la fórmula patsy del modelo
            # y/o de `covariates_processed` que son los términos finales.

            # Si `covariates_processed` tiene términos como "cr(Q('var'), df=4)" o "bs(Q('var'), df=3, degree=2)"
            # podemos parsearlos.

            # Alternativa: Usar `full_patsy_formula_for_new_data_transform` que se guarda en el modelo,
            # ya que esta fórmula se construye con la configuración de splines.

            patsy_formula_for_splines = md_tv.get('formula_patsy', '') # Usar la fórmula del modelo

            for orig_var_name in original_covs_in_model:
                display_str = orig_var_name
                # Buscar configuración de spline para esta variable original en la fórmula del modelo
                # Ejemplo: cr(Q('AGE'), df=4) -> AGE (Natural, df=4)
                # Ejemplo: bs(Q('BMI'), df=3, degree=2) -> BMI (B-spline, df=3, deg=2)

                # Regex para splines naturales (cr)
                cr_match = re.search(rf"cr\(Q\('{re.escape(orig_var_name)}'\),\s*df=(\d+)\)", patsy_formula_for_splines)
                if cr_match:
                    df = cr_match.group(1)
                    display_str += f" (Natural, df={df})"
                else:
                    # Regex para B-splines (bs)
                    bs_match = re.search(rf"bs\(Q\('{re.escape(orig_var_name)}'\),\s*df=(\d+)(?:,\s*degree=(\d+))?\)", patsy_formula_for_splines)
                    if bs_match:
                        df = bs_match.group(1)
                        degree = bs_match.group(2) if bs_match.group(2) else '3' # Default degree 3 si no se especifica
                        display_str += f" (B-spline, df={df}, deg={degree})"

                vars_splines_display_list.append(display_str)

            vars_splines_str = ", ".join(vars_splines_display_list) if vars_splines_display_list else "(Nulo)"
            if not vars_splines_display_list and covs_processed: # Si no pudimos parsear splines pero hay términos
                 vars_splines_str = ", ".join(covs_processed) # Fallback a los términos procesados

            metrics_tv = md_tv.get('metrics', {})

            # AIC
            aic_tv = metrics_tv.get('AIC')

            # -2 LogLik
            minus_2_loglik_tv = metrics_tv.get('-2 Log-Likelihood') # Ya debería estar en metrics
            if minus_2_loglik_tv is None and pd.notna(metrics_tv.get('Log-Likelihood')): # Calcular si no está
                minus_2_loglik_tv = -2.0 * metrics_tv.get('Log-Likelihood')

            # C-Index (Train)
            c_idx_tr_tv = metrics_tv.get('C-Index (Training)')

            # C-Index (CV)
            c_idx_cv_tv = md_tv.get('c_index_cv_mean') # Directamente del diccionario del modelo

            # Schoenfeld (p min)
            schoenfeld_df_results = md_tv.get("schoenfeld_results") # This might now come from proportional_hazard_test summary
            schoenfeld_p_min_tv = "N/A"
            self.log(f"DEBUG Treeview: Model '{name_tv}', schoenfeld_results type: {type(schoenfeld_df_results)}, empty: {schoenfeld_df_results.empty if isinstance(schoenfeld_df_results, pd.DataFrame) else 'N/A'}", "DEBUG")

            if isinstance(schoenfeld_df_results, pd.DataFrame) and not schoenfeld_df_results.empty and 'p' in schoenfeld_df_results.columns:
                # Filter out known global/summary rows by index name before taking min
                # Common index names for global tests in lifelines: 'global', 'GLOBAL', '- globales -', 'Overall', 'Test Statistic'
                # Making it case-insensitive and flexible
                global_test_indices = ['global', 'overall', 'test statistic']

                # Ensure index is string for filtering, handle MultiIndex if present
                if isinstance(schoenfeld_df_results.index, pd.MultiIndex):
                    # For MultiIndex, we might need a more specific way to identify global rows,
                    # or assume that individual terms are in the first level of the index.
                    # For now, we'll try to convert the first level to string and check.
                    try:
                        idx_to_check = schoenfeld_df_results.index.get_level_values(0).astype(str).str.lower()
                    except Exception as e_multi_idx:
                        self.log(f"DEBUG Treeview: Error processing MultiIndex for Schoenfeld: {e_multi_idx}", "WARN")
                        idx_to_check = pd.Series([]) # Empty series if error
                else: # Single index
                    idx_to_check = schoenfeld_df_results.index.astype(str).str.lower()

                individual_terms_schoenfeld_df = schoenfeld_df_results[
                    ~idx_to_check.isin(global_test_indices)
                ]

                self.log(f"DEBUG Treeview: Model '{name_tv}', N individual Schoenfeld terms after filtering: {len(individual_terms_schoenfeld_df)}", "DEBUG")
                if not individual_terms_schoenfeld_df.empty and 'p' in individual_terms_schoenfeld_df.columns:
                    valid_p_values_schoenfeld = individual_terms_schoenfeld_df['p'].dropna()
                    if not valid_p_values_schoenfeld.empty:
                        schoenfeld_p_min_tv = valid_p_values_schoenfeld.min()
                        self.log(f"DEBUG Treeview: Model '{name_tv}', Schoenfeld p min: {schoenfeld_p_min_tv} from {len(valid_p_values_schoenfeld)} values.", "DEBUG")
                    else:
                        self.log(f"DEBUG Treeview: Model '{name_tv}', No valid non-NaN Schoenfeld p-values for individual terms.", "DEBUG")
                else:
                    self.log(f"DEBUG Treeview: Model '{name_tv}', No individual Schoenfeld terms left after filtering or 'p' column missing.", "DEBUG")
            else:
                self.log(f"DEBUG Treeview: Model '{name_tv}', Schoenfeld results not DataFrame, empty, or no 'p' column.", "DEBUG")

            # Wald (p max)
            summary_df_tv = metrics_tv.get('summary_df') # This is CoxPHFitter.summary
            wald_p_max_tv = "N/A"
            self.log(f"DEBUG Treeview: Model '{name_tv}', summary_df type: {type(summary_df_tv)}, empty: {summary_df_tv.empty if isinstance(summary_df_tv, pd.DataFrame) else 'N/A'}", "DEBUG")

            if isinstance(summary_df_tv, pd.DataFrame) and not summary_df_tv.empty and 'p' in summary_df_tv.columns:
                # summary_df from CoxPHFitter directly lists individual covariates/spline components.
                # No need to filter out 'global' rows here as it doesn't typically have them.
                valid_p_values_wald = summary_df_tv['p'].dropna()
                if not valid_p_values_wald.empty:
                    wald_p_max_tv = valid_p_values_wald.max()
                    self.log(f"DEBUG Treeview: Model '{name_tv}', Wald p max: {wald_p_max_tv} from {len(valid_p_values_wald)} values.", "DEBUG")
                else:
                    self.log(f"DEBUG Treeview: Model '{name_tv}', No valid non-NaN Wald p-values.", "DEBUG")
            else:
                 self.log(f"DEBUG Treeview: Model '{name_tv}', Wald summary_df not DataFrame, empty, or no 'p' column.", "DEBUG")


            vals_tv = (
                i + 1,                                      # #
                name_tv,                                    # Nombre Modelo
                vars_splines_str,                           # Variables y Splines
                f"{aic_tv:.2f}" if pd.notna(aic_tv) else "N/A", # AIC
                f"{minus_2_loglik_tv:.2f}" if pd.notna(minus_2_loglik_tv) else "N/A", # -2 LogLik
                f"{c_idx_tr_tv:.3f}" if pd.notna(c_idx_tr_tv) else "N/A", # C-Index (Train)
                f"{c_idx_cv_tv:.3f}" if pd.notna(c_idx_cv_tv) else "N/A",   # C-Index (CV)
                format_p_value(schoenfeld_p_min_tv) if pd.notna(schoenfeld_p_min_tv) else "N/A", # Schoenfeld (p min)
                format_p_value(wald_p_max_tv) if pd.notna(wald_p_max_tv) else "N/A" # Wald (p max)
            )
            self.treeview_lista_modelos.insert("", tk.END, iid=str(i), values=vals_tv)
        self.log(f"Treeview actualizada con {len(self.generated_models_data)} modelos.", "INFO")

    def _sort_treeview_column(self, col_name):
        """Ordena los datos del Treeview por la columna especificada."""
        if not self.generated_models_data:
            return

        # Determinar el índice de la columna a partir de su nombre
        try:
            col_idx = self.treeview_lista_modelos["columns"].index(col_name)
        except ValueError:
            self.log(f"Error: Columna '{col_name}' no encontrada para ordenamiento.", "ERROR")
            return

        # Obtener los datos a ordenar (lista de tuplas de valores como se muestran en el Treeview)
        # Es mejor ordenar self.generated_models_data directamente y luego repoblar.

        # Función de clave para el ordenamiento
        def get_sort_key(model_dict_item):
            # Re-extraer el valor específico para la columna de ordenamiento
            # Esto debe coincidir con cómo se generan los valores en _update_models_treeview
            metrics = model_dict_item.get('metrics', {})

            if col_name == "#":
                # Necesitamos el índice original antes de ordenar, así que esto es un poco circular.
                # Guardaremos el índice original en el diccionario del modelo si no está ya.
                # O, más simple, ordenamos los datos y _update_models_treeview asigna nuevos #.
                # Para ordenar por '#', necesitamos el valor numérico.
                # Buscamos el modelo en self.generated_models_data para obtener su índice original.
                try:
                    original_index = self.generated_models_data.index(model_dict_item)
                    return original_index
                except ValueError:
                    return -1 # Fallback
            elif col_name == "Nombre Modelo":
                return model_dict_item.get('custom_model_name', model_dict_item.get('model_name', ''))
            elif col_name == "Variables y Splines":
                # Similar a _update_models_treeview para generar esta cadena
                covs_processed = model_dict_item.get('covariates_processed', [])
                original_covs_in_model = []
                for term in covs_processed:
                    match_q = re.search(r"Q\('([^']+)'\)", term)
                    if match_q: original_covs_in_model.append(match_q.group(1))
                    elif not any(x in term for x in ['cr(', 'bs(', 'C(', 'Intercept']):
                        original_covs_in_model.append(term)
                original_covs_in_model = sorted(list(set(original_covs_in_model)))
                patsy_formula_for_splines = model_dict_item.get('formula_patsy', '')
                vars_splines_display_list = []
                for orig_var_name in original_covs_in_model:
                    display_str = orig_var_name
                    cr_match = re.search(rf"cr\(Q\('{re.escape(orig_var_name)}'\),\s*df=(\d+)\)", patsy_formula_for_splines)
                    if cr_match: display_str += f" (Natural, df={cr_match.group(1)})"
                    else:
                        bs_match = re.search(rf"bs\(Q\('{re.escape(orig_var_name)}'\),\s*df=(\d+)(?:,\s*degree=(\d+))?\)", patsy_formula_for_splines)
                        if bs_match: display_str += f" (B-spline, df={bs_match.group(1)}, deg={bs_match.group(2) or '3'})"
                    vars_splines_display_list.append(display_str)
                return ", ".join(vars_splines_display_list) if vars_splines_display_list else "(Nulo)"
            elif col_name == "AIC":
                val = metrics.get('AIC')
                return float(val) if pd.notna(val) else float('-inf') # Tratar N/A como muy pequeño
            elif col_name == "-2 LogLik":
                val = metrics.get('-2 Log-Likelihood')
                if val is None and pd.notna(metrics.get('Log-Likelihood')):
                     val = -2.0 * metrics.get('Log-Likelihood')
                return float(val) if pd.notna(val) else float('-inf')
            elif col_name == "C-Index (Train)":
                val = metrics.get('C-Index (Training)')
                return float(val) if pd.notna(val) else float('-inf')
            elif col_name == "C-Index (CV)":
                val = model_dict_item.get('c_index_cv_mean')
                return float(val) if pd.notna(val) else float('-inf')
            elif col_name == "Schoenfeld (p min)":
                schoenfeld_df = model_dict_item.get("schoenfeld_results")
                if schoenfeld_df is not None and isinstance(schoenfeld_df, pd.DataFrame) and not schoenfeld_df.empty and 'p' in schoenfeld_df.columns:
                    sch_df_no_global = schoenfeld_df[~schoenfeld_df.index.astype(str).str.lower().isin(['global', 'test_statistic', 'global_test', 'overall'])]
                    if not sch_df_no_global['p'].dropna().empty:
                        return float(sch_df_no_global['p'].dropna().min())
                return float('inf') # Tratar N/A como muy grande para p-values (queremos los pequeños primero)
            elif col_name == "Wald (p max)":
                summary_df = metrics.get('summary_df')
                if summary_df is not None and isinstance(summary_df, pd.DataFrame) and not summary_df.empty and 'p' in summary_df.columns:
                    if not summary_df['p'].dropna().empty:
                        return float(summary_df['p'].dropna().max())
                return float('-inf') # Tratar N/A como muy pequeño para p-values (queremos los grandes primero)
            return "" # Fallback para otras columnas o si el valor no se encuentra

        current_reverse_order = self.treeview_sort_reversed.get(col_name, False)

        try:
            self.generated_models_data.sort(key=get_sort_key, reverse=current_reverse_order)
        except TypeError as e_sort:
            self.log(f"Error de tipo al ordenar por '{col_name}': {e_sort}. Asegúrese que todos los valores son comparables.", "ERROR")
            # Podría intentar convertir a string como fallback, pero puede no ser el orden deseado.
            # For now, just log and don't sort if types are mixed unexpectedly.
            return


        self.treeview_sort_reversed[col_name] = not current_reverse_order

        # Actualizar el Treeview
        self._update_models_treeview()
        self.log(f"Treeview ordenado por '{col_name}', descendente={current_reverse_order}.", "INFO")


    def _execute_cox_modeling_orchestrator(self):
        self.log("*"*35 + " INICIO MODELADO COX " + "*"*35, "HEADER")
        successful_fits = 0
        failed_fits = 0
        temp_models_list_orch = []

        prep_res = self._preparar_datos_para_modelado()
        if prep_res is None:
            self.log("Falló preparación de datos. Abortando.", "ERROR"); self.log("*"*35 + " FIN MODELADO (ERRORES) " + "*"*35, "HEADER")
            return
        
        (df_init_full, X_init_full, y_init_data,
         formula_init_patsy_full, terms_init_display,
         t_col_final, e_col_final,
         scaling_method_used, scaler_object, scaled_cols_list) = prep_res

        if df_init_full is None or df_init_full.empty: # df_init_full is now df_filtered_patsy
            self.log("DF inicial (post-patsy) vacío post-preparación. Abortando.", "ERROR"); self.log("*"*35 + " FIN MODELADO (ERRORES) " + "*"*35, "HEADER")
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
                                                             pen_val, l1_r, model_type_for_fit_logic="Univariado",
                                                             scaling_method_applied=scaling_method_used,
                                                             fitted_scaler_obj=scaler_object,
                                                             scaled_columns_info=scaled_cols_list)
                    if md_uni:
                        temp_models_list_orch.append(md_uni)
                        if md_uni.get("model") is not None:
                            successful_fits += 1
                        else:
                            failed_fits += 1

            if self.generate_univariate_forest_plot_var.get() and temp_models_list_orch:
                self.log("Generando Forest Plot para todos los modelos univariados...", "INFO")
                # Llamar a una nueva función que se encargará de generar el gráfico
                self._generate_univariate_forest_plot(temp_models_list_orch)

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
                                                       name_multi, terms_multi_current, formula_init_patsy_full, # formula_init_patsy_full is for new data transform
                                                       pen_val, l1_r, model_type_for_fit_logic="Multivariado",
                                                       scaling_method_applied=scaling_method_used,
                                                       fitted_scaler_obj=scaler_object,
                                                       scaled_columns_info=scaled_cols_list)
            if md_multi:
                temp_models_list_orch.append(md_multi)
                if md_multi.get("model") is not None:
                    successful_fits += 1
                else:
                    failed_fits += 1
 
        # Añadir los modelos generados a la lista existente, no sobrescribir
        self.generated_models_data.extend(temp_models_list_orch)
        self._update_models_treeview()
        msg_fin = f"Modelado completado. {len(temp_models_list_orch)} modelo(s) generado(s) y añadido(s)." if temp_models_list_orch else "No se generó ningún modelo nuevo."
        self.log(msg_fin, "SUCCESS" if temp_models_list_orch else "WARN")
        messagebox.showinfo("Modelado Terminado", msg_fin, parent=self.parent_for_dialogs)

        total_models_attempted = successful_fits + failed_fits
        self.log(f"Resumen de Convergencia de Modelos:", "SUBHEADER")
        self.log(f"  Modelos Totales Intentados: {total_models_attempted}", "INFO")
        self.log(f"  Ajustes Exitosos: {successful_fits}", "SUCCESS" if successful_fits > 0 else "INFO")
        self.log(f"  Ajustes Fallidos: {failed_fits}", "ERROR" if failed_fits > 0 else "INFO")

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

        # Actualizar UI de nombre/notas personalizados
        if self.selected_model_in_treeview:
            custom_name = self.selected_model_in_treeview.get('custom_model_name', self.selected_model_in_treeview.get('model_name', ''))
            custom_notes = self.selected_model_in_treeview.get('custom_model_notes', '')
            if self.entry_custom_model_name: # Verificar que el widget exista
                self.entry_custom_model_name_var.set(custom_name)
            if self.text_custom_model_notes: # Verificar que el widget exista
                self.text_custom_model_notes.config(state=tk.NORMAL)
                self.text_custom_model_notes.delete("1.0", tk.END)
                self.text_custom_model_notes.insert("1.0", custom_notes)
                self.text_custom_model_notes.config(state=tk.DISABLED if not self.selected_model_in_treeview else tk.NORMAL) # Permitir edición si hay modelo
        else: # No model selected
            if self.entry_custom_model_name: self.entry_custom_model_name_var.set("")
            if self.text_custom_model_notes:
                self.text_custom_model_notes.config(state=tk.NORMAL)
                self.text_custom_model_notes.delete("1.0", tk.END)
                self.text_custom_model_notes.config(state=tk.DISABLED)


        if self.btn_oos_calibration: # Check if button exists
            if self.selected_model_in_treeview and self.selected_model_in_treeview.get("oos_predictions"):
                self.btn_oos_calibration.config(state=tk.NORMAL)
            else:
                self.btn_oos_calibration.config(state=tk.DISABLED)

        if self.btn_collinearity_diag: # <-- NUEVO
            can_run_vif = False
            if self.selected_model_in_treeview:
                x_design = self.selected_model_in_treeview.get("X_design_used_for_fit")
                if x_design is not None and isinstance(x_design, pd.DataFrame) and x_design.shape[1] > 1:
                    can_run_vif = True
            self.btn_collinearity_diag.config(state=tk.NORMAL if can_run_vif else tk.DISABLED)

        self._update_results_buttons_state()

    def _save_custom_model_details(self):
        if not self.selected_model_in_treeview:
            messagebox.showwarning("Sin Modelo", "Seleccione un modelo del Treeview primero.", parent=self.parent_for_dialogs)
            return

        if not self.entry_custom_model_name or not self.text_custom_model_notes:
             self.log("Error: Widgets de nombre/notas personalizados no inicializados.", "ERROR")
             messagebox.showerror("Error UI", "Los campos para nombre/notas no están listos.", parent=self.parent_for_dialogs)
             return

        new_custom_name = self.entry_custom_model_name_var.get().strip()
        new_custom_notes = self.text_custom_model_notes.get("1.0", tk.END).strip()

        if not new_custom_name: # Permitir notas vacías, pero no nombre vacío (usará el original en ese caso)
            # Revertir al nombre original si el usuario borra el nombre personalizado
            new_custom_name = self.selected_model_in_treeview.get('model_name', 'Modelo Desconocido')
            self.entry_custom_model_name_var.set(new_custom_name) # Actualizar UI

        self.selected_model_in_treeview['custom_model_name'] = new_custom_name
        self.selected_model_in_treeview['custom_model_notes'] = new_custom_notes

        self.log(f"Detalles personalizados guardados para '{self.selected_model_in_treeview.get('model_name')}': Nombre='{new_custom_name}', Notas='{new_custom_notes[:30]}...'", "INFO")
        messagebox.showinfo("Guardado", "Nombre y notas personalizados guardados para el modelo seleccionado.", parent=self.parent_for_dialogs)

        # Actualizar el Treeview para reflejar el nuevo nombre personalizado
        self._update_models_treeview()

        # Re-seleccionar el item en el treeview si es posible
        # Esto es un poco más complejo porque el iid es el índice numérico, que puede cambiar si el orden cambia.
        # Por ahora, solo actualizamos. El usuario puede necesitar re-seleccionar si el orden cambia.
        # Si el orden no cambia, la selección debería persistir.


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

                    # Retrieve and format p-value for the current covariate
                    p_val_str = "N/A"
                    sch_results_df = md_sch.get("schoenfeld_results")
                    ph_test_summary_df = md_sch.get("proportional_hazard_test_summary")

                    # Try schoenfeld_results first
                    if sch_results_df is not None and isinstance(sch_results_df, pd.DataFrame) and not sch_results_df.empty and 'p' in sch_results_df.columns:
                        if cov_name_s in sch_results_df.index:
                            p_val = sch_results_df.loc[cov_name_s, 'p']
                            p_val_str = format_p_value(p_val)

                    # If p-value still "N/A", try ph_test_summary_df
                    if p_val_str == "N/A" and ph_test_summary_df is not None and isinstance(ph_test_summary_df, pd.DataFrame) and not ph_test_summary_df.empty and 'p' in ph_test_summary_df.columns:
                        if cov_name_s in ph_test_summary_df.index:
                            p_val = ph_test_summary_df.loc[cov_name_s, 'p']
                            p_val_str = format_p_value(p_val)

                    ax_s_curr.set_title(f"Schoenfeld: {cov_name_s}\nPH Test p: {p_val_str}", fontsize=9)
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

    def show_baseline_cumulative_incidence(self):
        if not self._check_model_selected_and_valid():
            return

        model_dict = self.selected_model_in_treeview
        cph_model = model_dict.get('model')
        model_name = model_dict.get('model_name', 'N/A')
        time_col_name = model_dict.get('time_col_for_model', 'Tiempo')

        if not hasattr(cph_model, 'baseline_survival_'):
            self.log(f"baseline_survival_ no encontrado en el modelo '{model_name}'. No se puede calcular F₀(t).", "ERROR")
            messagebox.showerror("Error de Datos",
                                 f"Atributo baseline_survival_ no disponible en el modelo '{model_name}'.",
                                 parent=self.parent_for_dialogs)
            return

        try:
            baseline_survival = cph_model.baseline_survival_
            baseline_cumulative_incidence = 1 - baseline_survival

            fig, ax = plt.subplots(figsize=(10, 6))
            baseline_cumulative_incidence.plot(ax=ax, legend=False, drawstyle='steps-post') # Often plotted as step function

            # Prepare plot options
            plot_opts = self.current_plot_options.copy()
            plot_opts['title'] = plot_opts.get('title') or f"Incidencia Acumulada Base F₀(t) ({model_name})"
            plot_opts['xlabel'] = plot_opts.get('xlabel') or f"{time_col_name}"
            plot_opts['ylabel'] = plot_opts.get('ylabel') or "F₀(t) (Incidencia Acumulada)"

            # Ensure y-axis starts at 0, and potentially goes up to 1 or slightly above if data dictates
            current_ylim = ax.get_ylim()
            final_ymin = 0
            final_ymax = max(1.0, current_ylim[1]) # Ensure it at least goes to 1.0
            if 'ylim_min' not in plot_opts or plot_opts['ylim_min'] is None: # User hasn't specified a min
                 plot_opts['ylim_min'] = final_ymin
            if 'ylim_max' not in plot_opts or plot_opts['ylim_max'] is None: # User hasn't specified a max
                 plot_opts['ylim_max'] = final_ymax


            apply_plot_options(ax, plot_opts, self.log)

            self._create_plot_window(fig, f"Incidencia Acum. Base: {model_name}")
            self.log(f"Gráfico de Incidencia Acumulada Base F₀(t) para '{model_name}' generado.", "INFO")

        except Exception as e:
            self.log(f"Error al generar gráfico de Incidencia Acumulada Base para '{model_name}': {e}", "ERROR")
            messagebox.showerror("Error de Gráfico",
                                 f"No se pudo generar el gráfico de Incidencia Acumulada Base:\n{e}",
                                 parent=self.parent_for_dialogs)
            if 'fig' in locals() and fig: # Ensure figure is closed if an error occurs after creation
                plt.close(fig)
            traceback.print_exc(limit=3)

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
        
        ttk.Label(frame_opts_pred,text="Tiempo(s) (ej: 100 o 50,100):").grid(row=1,column=0,padx=5,pady=3,sticky=tk.W)
        times_str_var_pred_ui = StringVar(value="") # Default a vacío para que sea opcional
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
            if type_ui_pred == "Supervivencia":
                pred_df = cph_model_for_pred.predict_survival_function(df_patsy_input_pred)
                pred_df.plot(ax=ax_curve_pred, legend=False)
                ax_curve_pred.set_ylabel("S(t|X)")
                title_curve_pred = f"Pred. Prob. Supervivencia ({name_for_pred})"
                label_prefix = "S"
            elif type_ui_pred == "Riesgo":
                pred_df = cph_model_for_pred.predict_cumulative_hazard(df_patsy_input_pred)
                pred_df.plot(ax=ax_curve_pred, legend=False)
                ax_curve_pred.set_ylabel("H(t|X)")
                title_curve_pred = f"Pred. Riesgo Acumulado ({name_for_pred})"
                label_prefix = "H"
            elif type_ui_pred == "ProbEventoAcum":
                surv_df_temp = cph_model_for_pred.predict_survival_function(df_patsy_input_pred)
                pred_df = 1 - surv_df_temp
                pred_df.plot(ax=ax_curve_pred, legend=False)
                ax_curve_pred.set_ylabel("1 - S(t|X)")
                title_curve_pred = f"Pred. Prob. Evento Acumulado (1-S(t)) ({name_for_pred})"
                label_prefix = "1-S"
            
            if times_list_pred: # Solo si se especificaron tiempos
                for t_val in times_list_pred:
                    if t_val < pred_df.index.min() or t_val > pred_df.index.max():
                        results_text_pred.append(f"{label_prefix}(t={t_val}|X) = N/A (fuera de rango de curva)");
                        self.log(f"Advertencia: Tiempo de predicción {t_val} fuera del rango de la curva de predicción.", "WARN")
                    else:
                        val_plot = np.interp(t_val, pred_df.index, pred_df.iloc[:,0])
                        results_text_pred.append(f"{label_prefix}(t={t_val}|X) = {val_plot:.3f}");
                        ax_curve_pred.scatter([t_val],[val_plot],marker='o',color='r',s=50,zorder=5,label=f't={t_val}' if t_val==times_list_pred[0] else None)
                if results_text_pred: ax_curve_pred.legend()
            else: # Si no se especificaron tiempos, no mostrar resultados puntuales ni scatter
                results_text_pred.append("Curva completa mostrada (no se especificaron tiempos puntuales).")
                # ax_curve_pred.legend() # La leyenda de la curva ya se maneja por plot() si hay múltiples líneas, pero aquí solo hay una.

            opts_curve_pred = self.current_plot_options.copy()
            opts_curve_pred['title'] = opts_curve_pred.get('title') or title_curve_pred
            opts_curve_pred['xlabel'] = opts_curve_pred.get('xlabel') or f"Tiempo ({md_dict_for_pred.get('time_col_for_model','T')})"
            apply_plot_options(ax_curve_pred, opts_curve_pred, self.log)
            
            self._create_plot_window(fig_curve_pred, title_curve_pred)
            
            if times_list_pred:
                messagebox.showinfo("Resultados Predicción", "Resultados en tiempos especificados:\n" + "\n".join(results_text_pred), parent=dialog_pred_ref)
            else:
                messagebox.showinfo("Resultados Predicción", "Curva de predicción completa generada.", parent=dialog_pred_ref)
        except Exception as e_curve_pred: self.log(f"Error pred/plot: {e_curve_pred}","ERROR"); traceback.print_exc(limit=3); messagebox.showerror("Error Pred/Plot",f"Error al predecir/plotear:\n{e_curve_pred}",parent=dialog_pred_ref)


    def generate_calibration_plot(self):
        # 1. Verificar que hay un modelo seleccionado y válido
        if not self._check_model_selected_and_valid():
            return

        # 2. Verificar si la función de calibración está disponible
        if not LIFELINES_CALIBRATION_AVAILABLE:
            messagebox.showwarning("Función No Disponible",
                                   "La gráfica de calibración no está disponible. "
                                   "Asegúrese de que su versión de 'lifelines' es reciente.",
                                   parent=self.parent_for_dialogs)
            return

        # 3. Obtener los objetos necesarios del modelo guardado
        md_cal = self.selected_model_in_treeview
        cph_model = md_cal.get('model')
        model_name = md_cal.get('model_name', 'N/A')

        # Es crucial usar los mismos datos con los que se ajustó el modelo
        # _df_for_fit_main_INTERNAL_USE was the original full df used for patsy design matrix creation
        # _X_design_rm_INTERNAL_USE was the design matrix
        # _y_survival_rm_INTERNAL_USE was the T,E outcome df
        # The survival_probability_calibration function expects the dataframe that was used to fit the model,
        # which means the one that includes the original columns before patsy transformation,
        # as lifelines will handle the formula application internally if the model was fit with a formula.
        training_data = md_cal.get('_df_for_fit_main_INTERNAL_USE') # This should be the correct one.

        if training_data is None:
            self.log(f"DataFrame de ajuste ('_df_for_fit_main_INTERNAL_USE') no disponible en el modelo '{model_name}'. No se puede generar gráfico de calibración.", "ERROR")
            messagebox.showerror("Error de Datos",
                               "Los datos de ajuste no se encontraron en el modelo guardado. "
                               "No se puede generar el gráfico de calibración.",
                               parent=self.parent_for_dialogs)
            return

        # Ensure the training_data still contains the necessary columns as per the model's formula
        # This is a sanity check, as the model fitting process itself would have required these.
        # CoxPHFitter stores the formula if fitted that way.
        if hasattr(cph_model, 'formula') and cph_model.formula:
            try:
                # Attempt a quick dmatrix creation with a single row to check for column presence
                # This is an indirect way to check if training_data is suitable for the model's formula
                # Note: This might be slow for very wide data, but generally okay.
                # Consider if there's a more direct way to get required columns from formula.
                dmatrix(cph_model.formula, training_data.head(1), return_type='dataframe')
            except Exception as e_col_check:
                self.log(f"Error al verificar columnas en training_data para calibración (modelo '{model_name}'): {e_col_check}. Los datos podrían no ser adecuados para la fórmula del modelo.", "WARN")
                # Proceed with caution, or could even error out here if strictness is required.
                # For now, let lifelines handle potential errors during calibration call.


        # 4. Pedir al usuario el tiempo de calibración t₀
        t0_str = simpledialog.askstring("Tiempo de Calibración",
                                        "Ingrese el punto de tiempo (t₀) para evaluar la calibración:",
                                        parent=self.parent_for_dialogs)
        if not t0_str:
            self.log("Generación de gráfico de calibración cancelada.", "INFO")
            return
        try:
            t0_value = float(t0_str)
            if t0_value <= 0:
                raise ValueError("El tiempo debe ser positivo.")
        except ValueError:
            messagebox.showerror("Valor Inválido",
                               f"'{t0_str}' no es un tiempo válido.",
                               parent=self.parent_for_dialogs)
            return

        # 5. Generar la gráfica
        fig_cal = None
        try:
            # Crear la figura y los ejes
            fig_cal, ax_cal = plt.subplots(figsize=(8, 8))

            # Llamar a la función de lifelines
            # survival_probability_calibration uses the fitted model (cph_model)
            # and the original training_data. It will internally use the model's
            # formula and duration/event columns specified during the model's fit.
            survival_probability_calibration(
                cph_model,      # The fitted CoxPHFitter object
                training_data,  # The DataFrame used to fit the model
                t0=t0_value,    # The specific time point for calibration
                ax=ax_cal       # The matplotlib axes to plot on
            )

            # Personalizar y mostrar la gráfica
            plot_title = f"Gráfico de Calibración para t₀={t0_value}\nModelo: {model_name}"

            # Use a fresh plot_opts dictionary for this specific plot
            # to avoid interference from self.current_plot_options for these specific labels
            current_opts_cal = self.current_plot_options.copy()
            current_opts_cal['title'] = current_opts_cal.get('title', plot_title)
            current_opts_cal['xlabel'] = current_opts_cal.get('xlabel', "Predicción (Probabilidad de Supervivencia Estimada)")
            current_opts_cal['ylabel'] = current_opts_cal.get('ylabel', "Observación (Proporción Real de Supervivencia)")

            apply_plot_options(ax_cal, current_opts_cal, self.log) # Apply combined/defaulted options

            # Crear la ventana para mostrar el gráfico
            self._create_plot_window(fig_cal, f"Calibración: {model_name} (t₀={t0_value})")

            self.log(f"Gráfico de calibración generado para t0={t0_value} para el modelo '{model_name}'.", "SUCCESS")

        except Exception as e:
            # Cerrar la figura si se creó pero hubo un error
            if fig_cal is not None:
                plt.close(fig_cal)

            self.log(f"Error al generar gráfico de calibración: {e}", "ERROR")
            self.log(traceback.format_exc(), "DEBUG") # Log full traceback for detailed debugging
            messagebox.showerror("Error de Gráfico",
                               f"No se pudo generar el gráfico de calibración:\n{e}",
                               parent=self.parent_for_dialogs)

    def show_variable_impact_plot(self):
        if not self._check_model_selected_and_valid(check_params=True):
            return

        md_vip = self.selected_model_in_treeview
        cph_model_vip = md_vip.get('model')
        model_name_vip = md_vip.get('model_name', 'N/A')

        original_training_data = md_vip.get('_df_for_fit_main_INTERNAL_USE')
        if original_training_data is None or original_training_data.empty:
            messagebox.showerror("Error de Datos",
                               "Los datos de ajuste originales ('_df_for_fit_main_INTERNAL_USE') no se encontraron en el modelo. "
                               "No se puede generar el gráfico de efecto de variable.",
                               parent=self.parent_for_dialogs)
            self.log(f"Datos de entrenamiento originales no encontrados para el modelo '{model_name_vip}'.", "ERROR")
            return

        candidate_vars_for_plot = set()
        if hasattr(cph_model_vip, 'formula') and cph_model_vip.formula:
            formula_terms = re.findall(r"Q\('([^']+)'\)|([a-zA-Z_][a-zA-Z0-9_]*)", cph_model_vip.formula)
            for q_term, raw_term in formula_terms:
                term_to_add = q_term if q_term else raw_term
                if term_to_add and term_to_add not in ['Intercept', '0', '1'] and not any(func in term_to_add for func in ['cr(', 'bs(', 'C(']):
                    if term_to_add in original_training_data.columns:
                        candidate_vars_for_plot.add(term_to_add)

        for col in original_training_data.select_dtypes(include=np.number).columns:
            if col not in [md_vip.get('time_col_for_model'), md_vip.get('event_col_for_model')]:
                 candidate_vars_for_plot.add(col)

        if not candidate_vars_for_plot:
            messagebox.showinfo("Sin Covariables Adecuadas",
                                "No se pudieron identificar covariables numéricas adecuadas para este gráfico.",
                                parent=self.parent_for_dialogs)
            self.log(f"No hay covariables numéricas adecuadas para el gráfico de efecto en el modelo '{model_name_vip}'.", "WARN")
            return

        dialog = Toplevel(self.parent_for_dialogs)
        dialog.title("Seleccionar Covariable para Gráfico de Efecto")
        dialog.geometry("400x400") # Adjusted height for radio buttons
        ttk.Label(dialog, text="Seleccione la covariable (preferiblemente continua) para visualizar su efecto:", wraplength=380).pack(pady=10, padx=10)

        covariate_var = StringVar() # This is for Listbox selection, not used directly if selection is fetched by index
        sorted_candidates = sorted(list(candidate_vars_for_plot))

        listbox_frame = ttk.Frame(dialog)
        listbox_frame.pack(pady=5, padx=10, fill=tk.BOTH, expand=True)
        # Changed selectmode to tk.EXTENDED
        listbox_covs_widget = Listbox(listbox_frame, selectmode=tk.EXTENDED, exportselection=False, height=8)
        for cov_name_lb in sorted_candidates:
            listbox_covs_widget.insert(tk.END, cov_name_lb)
        if sorted_candidates: # Pre-select the first item if list is not empty
            listbox_covs_widget.selection_set(0)
            # covariate_var.set(sorted_candidates[0]) # Not strictly needed if we fetch by index

        scrollbar_y_covs = ttk.Scrollbar(listbox_frame, orient=tk.VERTICAL, command=listbox_covs_widget.yview)
        listbox_covs_widget.config(yscrollcommand=scrollbar_y_covs.set)
        scrollbar_y_covs.pack(side=tk.RIGHT, fill=tk.Y)
        listbox_covs_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Frame for Y-axis scale options
        scale_frame = ttk.Frame(dialog)
        scale_frame.pack(pady=5, padx=10, fill=tk.X)
        ttk.Label(scale_frame, text="Escala Eje Y:").pack(side=tk.LEFT, padx=(0,5))

        y_scale_choice_var = StringVar(value="log_hr") # Default to Log(HR)

        rb_log_hr = ttk.Radiobutton(scale_frame, text="Log(Hazard Ratio)", variable=y_scale_choice_var, value="log_hr")
        rb_log_hr.pack(side=tk.LEFT)
        rb_hr = ttk.Radiobutton(scale_frame, text="Hazard Ratio", variable=y_scale_choice_var, value="hr")
        rb_hr.pack(side=tk.LEFT, padx=(5,0))

        chosen_covariates_for_effect = [] # Now a list
        chosen_y_scale = "log_hr" # Default, will be updated by on_ok

        def on_ok():
            nonlocal chosen_covariates_for_effect, chosen_y_scale # Make sure to declare nonlocal
            selections = listbox_covs_widget.curselection() # Get tuple of selected indices
            if selections: # Check if any item is selected
                chosen_covariates_for_effect = [listbox_covs_widget.get(i) for i in selections]
                chosen_y_scale = y_scale_choice_var.get() # Get the scale choice
                dialog.destroy()
            else:
                 messagebox.showwarning("Selección Requerida", "Debe seleccionar al menos una covariable de la lista.", parent=dialog)

        def on_cancel():
            dialog.destroy()

        button_frame = ttk.Frame(dialog)
        button_frame.pack(pady=10)
        ttk.Button(button_frame, text="Aceptar", command=on_ok).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Cancelar", command=on_cancel).pack(side=tk.RIGHT, padx=5)

        dialog.transient(self.parent_for_dialogs)
        dialog.grab_set()
        self.parent_for_dialogs.wait_window(dialog)

        if not chosen_covariates_for_effect: # Check if list is empty
            self.log("Selección de covariable(s) para gráfico de efecto cancelada o vacía.", "INFO")
            return

        self.log(f"Generando datos para gráfico de efecto. Covariables: {', '.join(chosen_covariates_for_effect)}, Escala Y: {chosen_y_scale}", "INFO")

        plot_data_list = [] # To store data for each line to be plotted

        # Determine if normalization is needed (more than one variable selected for plotting)
        apply_normalization = len(chosen_covariates_for_effect) > 1

        # Get all columns that were part of the model's formula (original features before patsy)
        # These are the columns that need to be present in the DataFrame passed to predict_log_partial_hazard
        all_original_model_vars = [
            col for col in original_training_data.columns
            if col not in [md_vip.get('time_col_for_model'), md_vip.get('event_col_for_model')]
        ]

        fig_effect, ax_effect = plt.subplots(figsize=(10, 6)) # Create figure once, before the loop

        for current_cov_to_plot in chosen_covariates_for_effect:
            self.log(f"Preparando datos para: {current_cov_to_plot}", "DEBUG")

            if current_cov_to_plot not in original_training_data.columns:
                self.log(f"Advertencia: La covariable '{current_cov_to_plot}' no está en los datos de entrenamiento originales. Saltando.", "WARN")
                messagebox.showwarning("Variable no Encontrada",
                                       f"La covariable '{current_cov_to_plot}' no se encontró en los datos originales del modelo.",
                                       parent=self.parent_for_dialogs)
                continue

            # Create sequence of values for the current_cov_to_plot
            min_val = original_training_data[current_cov_to_plot].min()
            max_val = original_training_data[current_cov_to_plot].max()
            is_numeric_cov = pd.api.types.is_numeric_dtype(original_training_data[current_cov_to_plot])

            if not is_numeric_cov and apply_normalization :
                 self.log(f"Advertencia: Normalización no aplicable a variable no numérica '{current_cov_to_plot}' en gráfico multivariable. Se usará sin normalizar si es la única, o se omitirá.", "WARN")
                 if len(chosen_covariates_for_effect) > 1: # Skip if multi-plot and non-numeric
                     messagebox.showwarning("Variable No Numérica", f"La variable '{current_cov_to_plot}' no es numérica y no puede normalizarse para el gráfico multivariable. Será omitida.", parent=self.parent_for_dialogs)
                     continue

            x_plot_values_actual = [] # Actual values of the covariate for the x-axis of this line
            x_axis_display_values = [] # Values to use for plotting on X (could be original or normalized)
            normalization_info = None # For legend: "Var (0=min, 1=max)"

            if pd.isna(min_val) or pd.isna(max_val) or (is_numeric_cov and min_val == max_val):
                if is_numeric_cov and min_val == max_val and pd.notna(min_val):
                    self.log(f"Variable '{current_cov_to_plot}' tiene un único valor numérico ({min_val}). Usando pequeño rango.", "DEBUG")
                    delta = abs(min_val * 0.05) if min_val != 0 else 0.05
                    if delta == 0: delta = 0.05
                    x_plot_values_actual = np.linspace(min_val - delta, max_val + delta, 100)
                elif not is_numeric_cov: # Categorical with one level or all NaN after filtering
                     unique_vals = original_training_data[current_cov_to_plot].unique()
                     if len(unique_vals) == 1 and pd.notna(unique_vals[0]):
                         x_plot_values_actual = [unique_vals[0]] * 2 # Plot as a point/short line
                         self.log(f"Variable '{current_cov_to_plot}' tiene un único valor categórico ('{unique_vals[0]}').", "DEBUG")
                     else:
                         self.log(f"No se pudo determinar rango para '{current_cov_to_plot}'. Saltando.", "WARN")
                         continue
                else: # Numeric but problematic range
                    self.log(f"No se pudo determinar rango para '{current_cov_to_plot}'. Saltando.", "WARN")
                    continue
            else: # Standard case for numeric with a range
                x_plot_values_actual = np.linspace(min_val, max_val, 100)

            # Normalization if needed
            if apply_normalization and is_numeric_cov:
                if max_val == min_val : # Avoid division by zero if somehow missed above
                    x_axis_display_values = np.zeros_like(x_plot_values_actual) if min_val == 0 else np.full_like(x_plot_values_actual, 0.5) # Or handle as single point
                else:
                    x_axis_display_values = (x_plot_values_actual - min_val) / (max_val - min_val)
                normalization_info = f"{current_cov_to_plot} (0={min_val:.2g}, 1={max_val:.2g})"
            else:
                x_axis_display_values = x_plot_values_actual # Use original scale for single var plot or non-numeric

            # Create the prediction DataFrame grid
            predict_df_list_for_current_cov = []
            for current_x_val in x_plot_values_actual: # Iterate using actual values
                row = {}
                row[current_cov_to_plot] = current_x_val # Set current plotting variable to its sequence value

                for other_col in all_original_model_vars:
                    if other_col == current_cov_to_plot:
                        continue # Already set

                    # If this other_col is ALSO one of the chosen_covariates_for_effect (but not current_cov_to_plot)
                    # it should be held at its mean/mode for the current_cov_to_plot's line.
                    # All other non-plotting model variables also held at mean/mode.
                    if pd.api.types.is_numeric_dtype(original_training_data[other_col]):
                        row[other_col] = original_training_data[other_col].mean()
                    else:
                        row[other_col] = original_training_data[other_col].mode(dropna=True)[0] if not original_training_data[other_col].mode(dropna=True).empty else None
                predict_df_list_for_current_cov.append(row)

            predict_df_current_cov = pd.DataFrame(predict_df_list_for_current_cov)
            # Reindex to ensure all necessary columns for the model formula are present, in correct order.
            predict_df_current_cov = predict_df_current_cov.reindex(columns=all_original_model_vars, fill_value=np.nan)
            # Note: fill_value for missing columns might need more thought if a variable was entirely missing
            # from all_original_model_vars but was in the formula (unlikely if all_original_model_vars is derived correctly).

            # Predict log-partial hazard
            log_ph_preds = cph_model_vip.predict_log_partial_hazard(predict_df_current_cov)

            y_values_for_plot = log_ph_preds
            if chosen_y_scale == "hr":
                y_values_for_plot = np.exp(log_ph_preds)

            # CI Calculation (attempt)
            ci_lower_plot, ci_upper_plot = None, None
            ci_available_for_this_line = False
            if PATSY_AVAILABLE and hasattr(cph_model_vip, 'formula') and hasattr(cph_model_vip, 'variance_matrix_') and not predict_df_current_cov.empty:
                try:
                    # Ensure predict_df_current_cov is suitable for dmatrix.
                    design_matrix_pred_current_cov = dmatrix(cph_model_vip.formula, predict_df_current_cov, return_type='dataframe')

                    # Align columns of design_matrix_pred_current_cov with cph_model_vip.params_.index before matrix multiplication
                    # This is a common point of failure if names/orders don't match.
                    # A robust way is to reindex design_matrix_pred_current_cov by model's parameter names, filling missing with 0.
                    params_cols = cph_model_vip.params_.index
                    design_matrix_pred_aligned = design_matrix_pred_current_cov.reindex(columns=params_cols, fill_value=0)

                    variance_pred = np.diag(design_matrix_pred_aligned @ cph_model_vip.variance_matrix_ @ design_matrix_pred_aligned.T)
                    se_pred = np.sqrt(variance_pred)

                    if chosen_y_scale == "log_hr":
                        ci_lower_plot = log_ph_preds - 1.96 * se_pred
                        ci_upper_plot = log_ph_preds + 1.96 * se_pred
                    else: # HR scale
                        ci_lower_plot = np.exp(log_ph_preds - 1.96 * se_pred)
                        ci_upper_plot = np.exp(log_ph_preds + 1.96 * se_pred)
                    ci_available_for_this_line = True
                except Exception as e_ci_loop:
                    self.log(f"Error calculating CI for {current_cov_to_plot}: {e_ci_loop}. CI will not be shown for this line.", "WARN")

            plot_data_list.append({
                "cov_name": current_cov_to_plot,
                "x_values_for_plot_axis": x_axis_display_values, # This is what's plotted on X
                "y_values_for_plot": y_values_for_plot,    # This is what's plotted on Y
                "ci_lower": ci_lower_plot,
                "ci_upper": ci_upper_plot,
                "ci_available": ci_available_for_this_line,
                "normalization_label": normalization_info if normalization_info else current_cov_to_plot
            })
        # End loop for chosen_covariates_for_effect

        # --- Plotting logic will start here, using plot_data_list ---
        if not plot_data_list:
            messagebox.showerror("Error de Datos", "No se pudieron generar datos para graficar.", parent=self.parent_for_dialogs)
            self.log("plot_data_list vacío, no se puede graficar.", "ERROR")
            if fig_effect: plt.close(fig_effect)
            return

        # --- New Plotting Logic Starts Here ---
        try:
            num_lines = len(plot_data_list)
            y_scale_name_for_legend = "Log(HR)" if chosen_y_scale == "log_hr" else "HR"

            for i, line_data in enumerate(plot_data_list):
                # Cycle through default matplotlib colors if more than one line
                color = plt.cm.get_cmap('viridis')(i / max(1, num_lines -1)) if num_lines > 1 else 'blue'

                ax_effect.plot(line_data["x_values_for_plot_axis"],
                               line_data["y_values_for_plot"],
                               label=line_data["normalization_label"], # This contains cov_name and norm info
                               color=color)
                if line_data["ci_available"]:
                    ax_effect.fill_between(line_data["x_values_for_plot_axis"],
                                           line_data["ci_lower"],
                                           line_data["ci_upper"],
                                           alpha=0.2,
                                           color=color)

            # Set common plot properties
            title_text = ""
            xlabel_text = ""
            ylabel_text = f"{y_scale_name_for_legend} (Efecto Parcial Ajustado)"

            if num_lines == 1:
                single_line_data = plot_data_list[0]
                title_text = f"Efecto Ajustado de '{single_line_data['cov_name']}' sobre {y_scale_name_for_legend}\nModelo: {model_name_vip}"
                # If not normalized (single plot), x-axis is actual value
                if single_line_data["normalization_label"] == single_line_data["cov_name"]:
                    xlabel_text = f"Valor de {single_line_data['cov_name']}"
                else: # Was normalized even for single plot (e.g. if logic changes) or to show range
                    xlabel_text = f"Valor Normalizado de {single_line_data['cov_name']} (0-1)"

            else: # Multiple lines
                title_text = f"Efecto Ajustado de Múltiples Covariables sobre {y_scale_name_for_legend}\nModelo: {model_name_vip}"
                xlabel_text = "Valor Normalizado de Covariable (0-1)"

            if chosen_y_scale == "hr":
                ax_effect.axhline(1, color='grey', linestyle='--', linewidth=0.8)
            else: # log_hr scale
                ax_effect.axhline(0, color='grey', linestyle='--', linewidth=0.8)

            current_opts_effect = self.current_plot_options.copy()
            current_opts_effect['title'] = current_opts_effect.get('title', title_text)
            current_opts_effect['xlabel'] = xlabel_text # Always use the specific one for this plot
            current_opts_effect['ylabel'] = ylabel_text # Always use the specific one

            apply_plot_options(ax_effect, current_opts_effect, self.log)

            if num_lines > 0 and ax_effect.has_data(): # Check if any data was actually plotted
                ax_effect.legend(fontsize='small')

            plt.tight_layout()
            self._create_plot_window(fig_effect, f"Efecto Ajustado de Covariable(s) ({model_name_vip})")
            self.log(f"Gráfico de efecto ajustado para {num_lines} covariable(s) ({chosen_y_scale}) generado.", "SUCCESS")

        except Exception as e_plot_final:
            self.log(f"Error final al graficar efectos ajustados: {e_plot_final}", "ERROR")
            if fig_effect: plt.close(fig_effect) # Ensure figure is closed on error
            traceback.print_exc(limit=5)
            messagebox.showerror("Error de Gráfico Final",
                               f"No se pudo generar el gráfico de efectos ajustados:\n{e_plot_final}",
                               parent=self.parent_for_dialogs)
        # --- End of New Plotting Logic ---

    def show_variable_impact_plot_hr_scale(self):
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

        dialog = Toplevel(self.parent_for_dialogs)
        dialog.title("Seleccionar Covariable para Gráfico HR")
        dialog.geometry("400x350")
        ttk.Label(dialog, text="Seleccione la covariable para el gráfico de impacto (escala HR):", wraplength=380).pack(pady=10, padx=10)
        covariate_var = StringVar()
        combo_covs_widget = None
        listbox_covs_widget = None

        if len(available_covariates) < 20:
            combo_covs_widget = ttk.Combobox(dialog, textvariable=covariate_var, values=available_covariates, state="readonly", width=40)
            if available_covariates:
                combo_covs_widget.set(available_covariates[0])
            combo_covs_widget.pack(pady=5, padx=10)
        else:
            ttk.Label(dialog, text="Covariables disponibles:").pack(pady=(5,0))
            listbox_frame = ttk.Frame(dialog)
            listbox_frame.pack(pady=5, padx=10, fill=tk.BOTH, expand=True)
            listbox_covs_widget = Listbox(listbox_frame, selectmode=SINGLE, exportselection=False, height=8)
            for cov_name_lb in available_covariates:
                listbox_covs_widget.insert(tk.END, cov_name_lb)
            if available_covariates:
                listbox_covs_widget.selection_set(0)
            scrollbar_y_covs = ttk.Scrollbar(listbox_frame, orient=tk.VERTICAL, command=listbox_covs_widget.yview)
            listbox_covs_widget.config(yscrollcommand=scrollbar_y_covs.set)
            scrollbar_y_covs.pack(side=tk.RIGHT, fill=tk.Y)
            listbox_covs_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        chosen_covariate = None
        def on_ok():
            nonlocal chosen_covariate
            selected_value = None
            if listbox_covs_widget and listbox_covs_widget.winfo_exists():
                if listbox_covs_widget.curselection():
                    selected_value = listbox_covs_widget.get(listbox_covs_widget.curselection()[0])
            elif combo_covs_widget and combo_covs_widget.winfo_exists():
                selected_value = covariate_var.get()
            if selected_value and selected_value.strip():
                chosen_covariate = selected_value
                dialog.destroy()
            else:
                 messagebox.showwarning("Selección Requerida", "Debe seleccionar una covariable.", parent=dialog)

        def on_cancel():
            dialog.destroy()

        button_frame = ttk.Frame(dialog)
        button_frame.pack(pady=10)
        ttk.Button(button_frame, text="Aceptar", command=on_ok).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Cancelar", command=on_cancel).pack(side=tk.RIGHT, padx=5)
        dialog.transient(self.parent_for_dialogs)
        dialog.grab_set()
        self.parent_for_dialogs.wait_window(dialog)

        if not chosen_covariate:
            self.log("Selección de covariable para gráfico de impacto HR cancelada.", "INFO")
            return

        covariate_for_plot = chosen_covariate
        match = re.match(r"Q\('([^']+)'\)", chosen_covariate)
        if match:
            covariate_for_plot = match.group(1)

        fig_vip_hr = None
        try:
            fig_vip_hr, ax_vip_hr = plt.subplots(figsize=(10, 6))

            # Lifelines plots on log-hazard scale by default
            cph_model_vip.plot_partial_effects_on_outcome(
                covariate_for_plot,
                values=None,
                plot_baseline=False,
                ax=ax_vip_hr
            )

            # Transform y-axis ticks to HR scale
            current_yticks = ax_vip_hr.get_yticks()
            ax_vip_hr.set_yticklabels([f"{np.exp(y_tick):.2f}" for y_tick in current_yticks])

            # Add a horizontal line at HR = 1 (which is log(HR) = 0 on the original scale)
            # The plot_partial_effects_on_outcome already draws a line at y=0 (log(HR)=0)
            # We just need to make sure this line is understood as HR=1.
            # If we want to be explicit or if the line style needs changing:
            # ax_vip_hr.axhline(0, color='grey', linestyle=':', linewidth=1, label="HR = 1")
            # This line is already at log(HR)=0. The tick transformation handles the perception.


            plot_title = f"Impacto de '{covariate_for_plot}' sobre Hazard Ratio (HR)"
            plot_title += f"\nModelo: {model_name_vip}"

            current_opts_vip_hr = self.current_plot_options.copy()
            current_opts_vip_hr['title'] = current_opts_vip_hr.get('title', plot_title)
            current_opts_vip_hr['xlabel'] = current_opts_vip_hr.get('xlabel', f"Valor de {covariate_for_plot}")
            # Explicitly set ylabel for this specific graph
            current_opts_vip_hr['ylabel'] = f"Hazard Ratio (HR) para {covariate_for_plot}"

            apply_plot_options(ax_vip_hr, current_opts_vip_hr, self.log)

            # Adjust layout if y-tick labels are too wide
            plt.tight_layout()
            self._create_plot_window(fig_vip_hr, f"Impacto Variable (HR): {chosen_covariate} ({model_name_vip})")
            self.log(f"Gráfico de impacto (escala HR) para '{chosen_covariate}' generado.", "SUCCESS")

        except IndexError as e_vip_idx:
            # Handle known IndexError from lifelines if necessary (as in show_variable_impact_plot)
            # For brevity, this specific fallback is omitted here but could be added if needed.
            tb_str_vip = traceback.format_exc()
            self.log(f"IndexError al generar gráfico de impacto HR para '{chosen_covariate}': {e_vip_idx}", "ERROR")
            self.log(tb_str_vip, "DEBUG")
            messagebox.showerror("Error de Gráfico (IndexError)",
                                 f"Se produjo un IndexError al generar el gráfico de impacto HR para '{chosen_covariate}':\n{e_vip_idx}\n\n"
                                 "Esto puede ser un problema con la librería 'lifelines' o la covariable seleccionada. "
                                 "Consulte el log para más detalles.",
                                 parent=self.parent_for_dialogs)
            if fig_vip_hr: plt.close(fig_vip_hr)
        except Exception as e_vip:
            self.log(f"Error general al generar gráfico de impacto HR para '{chosen_covariate}': {e_vip}", "ERROR")
            self.log(traceback.format_exc(), "DEBUG")
            messagebox.showerror("Error Gráfico",
                               f"No se pudo generar el gráfico de impacto HR para '{chosen_covariate}':\n{e_vip}",
                               parent=self.parent_for_dialogs)
            if fig_vip_hr:
                plt.close(fig_vip_hr)

    def show_variable_impact_plot_hr_scale(self):
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

        dialog = Toplevel(self.parent_for_dialogs)
        dialog.title("Seleccionar Covariable para Gráfico HR")
        dialog.geometry("400x350")
        ttk.Label(dialog, text="Seleccione la covariable para el gráfico de impacto (escala HR):", wraplength=380).pack(pady=10, padx=10)
        covariate_var = StringVar()
        combo_covs_widget = None
        listbox_covs_widget = None

        if len(available_covariates) < 20:
            combo_covs_widget = ttk.Combobox(dialog, textvariable=covariate_var, values=available_covariates, state="readonly", width=40)
            if available_covariates:
                combo_covs_widget.set(available_covariates[0])
            combo_covs_widget.pack(pady=5, padx=10)
        else:
            ttk.Label(dialog, text="Covariables disponibles:").pack(pady=(5,0))
            listbox_frame = ttk.Frame(dialog)
            listbox_frame.pack(pady=5, padx=10, fill=tk.BOTH, expand=True)
            listbox_covs_widget = Listbox(listbox_frame, selectmode=SINGLE, exportselection=False, height=8)
            for cov_name_lb in available_covariates:
                listbox_covs_widget.insert(tk.END, cov_name_lb)
            if available_covariates:
                listbox_covs_widget.selection_set(0)
            scrollbar_y_covs = ttk.Scrollbar(listbox_frame, orient=tk.VERTICAL, command=listbox_covs_widget.yview)
            listbox_covs_widget.config(yscrollcommand=scrollbar_y_covs.set)
            scrollbar_y_covs.pack(side=tk.RIGHT, fill=tk.Y)
            listbox_covs_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        chosen_covariate = None
        def on_ok():
            nonlocal chosen_covariate
            selected_value = None
            if listbox_covs_widget and listbox_covs_widget.winfo_exists():
                if listbox_covs_widget.curselection():
                    selected_value = listbox_covs_widget.get(listbox_covs_widget.curselection()[0])
            elif combo_covs_widget and combo_covs_widget.winfo_exists():
                selected_value = covariate_var.get()
            if selected_value and selected_value.strip():
                chosen_covariate = selected_value
                dialog.destroy()
            else:
                 messagebox.showwarning("Selección Requerida", "Debe seleccionar una covariable.", parent=dialog)

        def on_cancel():
            dialog.destroy()

        button_frame = ttk.Frame(dialog)
        button_frame.pack(pady=10)
        ttk.Button(button_frame, text="Aceptar", command=on_ok).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Cancelar", command=on_cancel).pack(side=tk.RIGHT, padx=5)
        dialog.transient(self.parent_for_dialogs)
        dialog.grab_set()
        self.parent_for_dialogs.wait_window(dialog)

        if not chosen_covariate:
            self.log("Selección de covariable para gráfico de impacto HR cancelada.", "INFO")
            return

        covariate_for_plot = chosen_covariate
        match = re.match(r"Q\('([^']+)'\)", chosen_covariate)
        if match:
            covariate_for_plot = match.group(1)

        fig_vip_hr = None
        try:
            fig_vip_hr, ax_vip_hr = plt.subplots(figsize=(10, 6))

            # Lifelines plots on log-hazard scale by default
            cph_model_vip.plot_partial_effects_on_outcome(
                covariate_for_plot,
                values=None,
                plot_baseline=False,
                ax=ax_vip_hr
            )

            # Transform y-axis ticks to HR scale
            current_yticks = ax_vip_hr.get_yticks()
            ax_vip_hr.set_yticklabels([f"{np.exp(y_tick):.2f}" for y_tick in current_yticks])

            # Add a horizontal line at HR = 1 (which is log(HR) = 0 on the original scale)
            # The plot_partial_effects_on_outcome already draws a line at y=0 (log(HR)=0)
            # We just need to make sure this line is understood as HR=1.
            # If we want to be explicit or if the line style needs changing:
            # ax_vip_hr.axhline(0, color='grey', linestyle=':', linewidth=1, label="HR = 1")
            # This line is already at log(HR)=0. The tick transformation handles the perception.


            plot_title = f"Impacto de '{covariate_for_plot}' sobre Hazard Ratio (HR)"
            plot_title += f"\nModelo: {model_name_vip}"

            current_opts_vip_hr = self.current_plot_options.copy()
            current_opts_vip_hr['title'] = current_opts_vip_hr.get('title', plot_title)
            current_opts_vip_hr['xlabel'] = current_opts_vip_hr.get('xlabel', f"Valor de {covariate_for_plot}")
            # Explicitly set ylabel for this specific graph
            current_opts_vip_hr['ylabel'] = f"Hazard Ratio (HR) para {covariate_for_plot}"

            apply_plot_options(ax_vip_hr, current_opts_vip_hr, self.log)

            # Adjust layout if y-tick labels are too wide
            plt.tight_layout()
            self._create_plot_window(fig_vip_hr, f"Impacto Variable (HR): {chosen_covariate} ({model_name_vip})")
            self.log(f"Gráfico de impacto (escala HR) para '{chosen_covariate}' generado.", "SUCCESS")

        except IndexError as e_vip_idx:
            # Handle known IndexError from lifelines if necessary (as in show_variable_impact_plot)
            # For brevity, this specific fallback is omitted here but could be added if needed.
            tb_str_vip = traceback.format_exc()
            self.log(f"IndexError al generar gráfico de impacto HR para '{chosen_covariate}': {e_vip_idx}", "ERROR")
            self.log(tb_str_vip, "DEBUG")
            messagebox.showerror("Error de Gráfico (IndexError)",
                                 f"Se produjo un IndexError al generar el gráfico de impacto HR para '{chosen_covariate}':\n{e_vip_idx}\n\n"
                                 "Esto puede ser un problema con la librería 'lifelines' o la covariable seleccionada. "
                                 "Consulte el log para más detalles.",
                                 parent=self.parent_for_dialogs)
            if fig_vip_hr: plt.close(fig_vip_hr)
        except Exception as e_vip:
            self.log(f"Error general al generar gráfico de impacto HR para '{chosen_covariate}': {e_vip}", "ERROR")
            self.log(traceback.format_exc(), "DEBUG")
            messagebox.showerror("Error Gráfico",
                               f"No se pudo generar el gráfico de impacto HR para '{chosen_covariate}':\n{e_vip}",
                               parent=self.parent_for_dialogs)
            if fig_vip_hr:
                plt.close(fig_vip_hr)

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
        original_name_gst = model_dict_gst.get('model_name', 'N/A')
        custom_name_gst = model_dict_gst.get('custom_model_name', original_name_gst)
        custom_notes_gst = model_dict_gst.get('custom_model_notes', '')

        s_txt_gst = f"--- Resumen Modelo: {custom_name_gst} ---\n"
        if custom_name_gst != original_name_gst:
            s_txt_gst += f"(Nombre Original: {original_name_gst})\n"
        s_txt_gst += f"Generado: {pd.Timestamp.now():%Y-%m-%d %H:%M:%S}\n"

        if custom_notes_gst:
            s_txt_gst += f"\nNotas Personalizadas:\n{custom_notes_gst}\n"

        s_txt_gst += "\nConfiguración Ajuste:\n"; s_txt_gst += f"  Tiempo: {model_dict_gst.get('time_col_for_model','N/A')}\n  Evento: {model_dict_gst.get('event_col_for_model','N/A')}\n"
        s_txt_gst += f"  Fórmula Patsy (usada en fit): {model_dict_gst.get('formula_patsy','N/A')}\n"
        s_txt_gst += f"  Fórmula Patsy (original completa para transformar nuevos datos): {model_dict_gst.get('full_patsy_formula_for_new_data_transform','N/A')}\n"
        s_txt_gst += f"  Términos Modelo (columnas en X_design): {', '.join(model_dict_gst.get('covariates_processed',[]))}\n"
        s_txt_gst += f"  Penalización: {model_dict_gst.get('penalizer_value',0.0):.4g} (L1 Ratio: {model_dict_gst.get('l1_ratio_value',0.0):.2f})\n  Manejo Empates (UI): {model_dict_gst.get('tie_method_used','N/A')} (Lifelines usará su default: efron)\n"
        
        # Información de Escalado
        scaling_method = model_dict_gst.get('scaling_method_applied', 'Ninguna')
        scaled_cols = model_dict_gst.get('scaled_columns_info', [])
        s_txt_gst += "\nPreprocesamiento de Covariables Numéricas:\n"
        s_txt_gst += f"  Método de Escalado Aplicado: {scaling_method}\n"
        if scaling_method != "Ninguna" and scaled_cols:
            s_txt_gst += f"  Columnas Escaladas: {', '.join(scaled_cols)}\n"
        elif scaling_method != "Ninguna" and not scaled_cols:
            s_txt_gst += "  (Método de escalado seleccionado, pero no se escalaron columnas numéricas.)\n"
        s_txt_gst += "\n" # Add a newline for separation

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
            if self.btn_oos_calibration:
                self.btn_oos_calibration.config(state=tk.DISABLED)
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

        # Controles de fuente
        ttk.Label(frame_opts_plot_global, text="Fuente:").pack(side=tk.LEFT, padx=(20, 5), pady=5)
        self.font_family_var = StringVar(value="sans-serif")
        font_families = ["serif", "sans-serif", "monospace", "Arial", "Times New Roman", "Courier New", "Palatino Linotype"]
        self.font_family_combo = ttk.Combobox(frame_opts_plot_global, textvariable=self.font_family_var, values=font_families, state="readonly", width=15)
        self.font_family_combo.pack(side=tk.LEFT, padx=5, pady=5)
        self.font_family_combo.bind("<<ComboboxSelected>>", self.on_font_change)

        ttk.Label(frame_opts_plot_global, text="Tamaño:").pack(side=tk.LEFT, padx=(10, 5), pady=5)
        self.font_size_var = IntVar(value=10)
        self.font_size_spinbox = ttk.Spinbox(frame_opts_plot_global, from_=6, to=20, textvariable=self.font_size_var, width=5, command=self.on_font_change)
        self.font_size_spinbox.pack(side=tk.LEFT, padx=5, pady=5)
        
        self.results_display_area_rc = ttk.Frame(r_content_rc, padding=10)
        self.results_display_area_rc.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        ttk.Label(self.results_display_area_rc, text="Seleccione modelo en Pestaña 2 y use botones de acción para ver resultados.", wraplength=600, justify=tk.CENTER, font=("TkDefaultFont",10,"italic")).pack(pady=20,padx=10)
        self.log("Controles Resultados creados.", "DEBUG")

    def on_font_change(self, event=None):
        """Se llama cuando la fuente o el tamaño de la fuente cambian."""
        font_family = self.font_family_var.get()
        font_size = self.font_size_var.get()

        # Actualizar la configuración global de matplotlib
        plt.rcParams.update({
            'font.family': font_family,
            'font.size': font_size,
            'axes.titlesize': font_size + 2,
            'axes.labelsize': font_size,
            'xtick.labelsize': font_size,
            'ytick.labelsize': font_size,
            'legend.fontsize': font_size,
            'figure.titlesize': font_size + 4
        })
        self.log(f"Fuente global de gráficos actualizada a: {font_family}, Tamaño: {font_size}", "CONFIG")

    def update_font_styles(self, font_family, font_size):
        """Actualiza la fuente en los widgets de texto de esta pestaña."""
        if hasattr(self, 'log_text_widget'):
            self.log_text_widget.config(font=(font_family, font_size))
        if hasattr(self, 'text_custom_model_notes'):
            self.text_custom_model_notes.config(font=(font_family, font_size))

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

    def open_graph_selection_dialog(self):
        if not self._check_model_selected_and_valid():
            # _check_model_selected_and_valid already shows a message if no model or invalid
            return

        # Define graph names and their corresponding methods
        # Using the user-approved names where applicable
        graph_callbacks = {
            "Riesgo Acumulado Base H₀(t)": self.show_baseline_hazard,
            "Gráf. Schoenfeld": self.show_schoenfeld,
            "Supervivencia Base S₀(t)": self.show_baseline_survival,
            "Incidencia Acumulada Base F₀(t)": self.show_baseline_cumulative_incidence, # New entry
            "Forest Plot (HRs)": self.generar_forest_plot,
            "Gráf. Calibración": self.generate_calibration_plot,
            "Análisis de Efecto de Covariable(s)": self.show_variable_impact_plot
            # Add more graph types here if needed in the future
        }

        # Callback function to be executed when "Generar Seleccionados" is clicked in the dialog
        def generate_selected_graphs(selected_graph_names):
            if not selected_graph_names:
                # This case should ideally be handled by the dialog itself, but as a safeguard:
                self.log("Ningún gráfico seleccionado para generar desde el diálogo.", "INFO")
                return

            self.log(f"Generando gráficos seleccionados: {', '.join(selected_graph_names)}", "INFO")
            for graph_name in selected_graph_names:
                callback_method = graph_callbacks.get(graph_name)
                if callback_method:
                    try:
                        # Check specific requirements for certain plots before calling
                        if graph_name == "Gráf. Schoenfeld" or                            graph_name == "Forest Plot (HRs)" or                            graph_name == "Efecto Variable sobre Log(HR)":
                            if not self._check_model_selected_and_valid(check_params=True):
                                self.log(f"Modelo no válido o sin parámetros para '{graph_name}'. Saltando.", "WARN")
                                continue # Skip this graph if model doesn't have params

                        # For calibration plot, it has its own internal checks for LIFELINES_CALIBRATION_AVAILABLE
                        # and prompts for t0.

                        callback_method()
                        self.log(f"Gráfico '{graph_name}' solicitado.", "DEBUG")
                    except Exception as e_graph_gen:
                        self.log(f"Error al generar gráfico '{graph_name}': {e_graph_gen}", "ERROR")
                        messagebox.showerror("Error de Gráfico",
                                             f"No se pudo generar el gráfico '{graph_name}':\n{e_graph_gen}",
                                             parent=self.parent_for_dialogs) # Assuming self.parent_for_dialogs is accessible
                        traceback.print_exc(limit=3)
                else:
                    self.log(f"No se encontró el método callback para el gráfico: {graph_name}", "WARN")

        # Instantiate and show the dialog
        CoxGraphSelectionDialog(
            parent=self.parent_for_dialogs,
            title="Seleccionar Gráficos Cox",
            graph_options_callbacks=graph_callbacks,
            apply_callback=generate_selected_graphs
        )
    
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

    def _calculate_and_show_vif(self):
        """Calcula y muestra el Factor de Inflación de Varianza (VIF) para el modelo seleccionado."""
        if not self._check_model_selected_and_valid():
            return

        model_dict = self.selected_model_in_treeview
        model_name = model_dict.get('model_name', 'N/A')
        self.log(f"Iniciando cálculo de VIF para modelo: '{model_name}'", "INFO")

        X_design = model_dict.get("X_design_used_for_fit")

        if X_design is None or not isinstance(X_design, pd.DataFrame) or X_design.shape[1] <= 1:
            messagebox.showinfo("No Aplicable",
                                "El diagnóstico de colinealidad (VIF) solo es aplicable a modelos multivariados con más de una variable.",
                                parent=self.parent_for_dialogs)
            self.log("Cálculo de VIF no aplicable (no es multivariado > 1 var).", "INFO")
            return

        # Calcular VIF
        try:
            vif_data = pd.DataFrame()
            vif_data["feature"] = X_design.columns
            vif_data["VIF"] = [variance_inflation_factor(X_design.values, i) for i in range(X_design.shape[1])]
            vif_data.sort_values("VIF", ascending=False, inplace=True)
            self.log(f"VIF calculado para {len(vif_data)} características.", "DEBUG")

            # Calcular matrices de correlación
            pearson_corr = X_design.corr(method='pearson')
            spearman_corr = X_design.corr(method='spearman')
            self.log("Matrices de correlación de Pearson y Spearman calculadas.", "DEBUG")

        except Exception as e:
            self.log(f"Error calculando diagnósticos de colinealidad: {e}", "ERROR")
            messagebox.showerror("Error de Cálculo", f"No se pudo completar el diagnóstico de colinealidad:\n{e}", parent=self.parent_for_dialogs)
            return

        # Crear ventana emergente para mostrar los resultados
        popup = Toplevel(self.parent_for_dialogs)
        popup.title(f"Diagnóstico de Colinealidad - {model_name}")
        popup.geometry("700x550")
        popup.transient(self.parent_for_dialogs)
        popup.grab_set()

        # Crear Notebook para las pestañas
        notebook = ttk.Notebook(popup)
        notebook.pack(fill="both", expand=True, padx=10, pady=10)

        # --- Pestaña VIF ---
        vif_tab = ttk.Frame(notebook)
        notebook.add(vif_tab, text="VIF")

        # Explicación VIF
        explanation_text = ("El Factor de Inflación de la Varianza (VIF) mide la multicolinealidad entre las variables predictoras en un modelo de regresión.\n\n"
                            "Interpretación general:\n"
                            " • VIF = 1: No hay correlación.\n"
                            " • 1 < VIF < 5: Correlación moderada.\n"
                            " • VIF > 5 ó 10: Correlación alta, puede ser problemática.")
        ttk.Label(vif_tab, text=explanation_text, wraplength=650, justify="left").pack(pady=10, padx=10, anchor="w")

        # Tabla VIF
        vif_tree_frame = ttk.Frame(vif_tab)
        vif_tree_frame.pack(fill="both", expand=True, padx=10, pady=5)

        vif_cols = ("Variable", "VIF")
        vif_tree = ttk.Treeview(vif_tree_frame, columns=vif_cols, show="headings")
        vif_tree.heading("Variable", text="Variable del Modelo")
        vif_tree.heading("VIF", text="Valor VIF")
        vif_tree.column("Variable", width=400)
        vif_tree.column("VIF", width=100, anchor="e")

        vif_tree.tag_configure('high_corr', background='salmon')
        vif_tree.tag_configure('mod_corr', background='khaki')

        for index, row in vif_data.iterrows():
            vif_value = f"{row['VIF']:.3f}"
            tags = ()
            if row['VIF'] > 10:
                tags = ('high_corr',)
            elif row['VIF'] > 5:
                tags = ('mod_corr',)
            vif_tree.insert("", "end", values=(row["feature"], vif_value), tags=tags)

        vif_ysb = ttk.Scrollbar(vif_tree_frame, orient="vertical", command=vif_tree.yview)
        vif_xsb = ttk.Scrollbar(vif_tree_frame, orient="horizontal", command=vif_tree.xview)
        vif_tree.configure(yscrollcommand=vif_ysb.set, xscrollcommand=vif_xsb.set)
        vif_ysb.pack(side="right", fill="y")
        vif_xsb.pack(side="bottom", fill="x")
        vif_tree.pack(fill="both", expand=True)

        # --- Pestaña Correlación Pearson ---
        pearson_tab = ttk.Frame(notebook)
        notebook.add(pearson_tab, text="Correlación Pearson")

        # --- Pestaña Correlación Pearson ---
        pearson_tab = ttk.Frame(notebook)
        notebook.add(pearson_tab, text="Correlación Pearson")
        self._populate_correlation_matrix_tab(pearson_tab, pearson_corr, "Pearson")

        # --- Pestaña Correlación Spearman ---
        spearman_tab = ttk.Frame(notebook)
        notebook.add(spearman_tab, text="Correlación Spearman")
        self._populate_correlation_matrix_tab(spearman_tab, spearman_corr, "Spearman")


        ttk.Button(popup, text="Cerrar", command=popup.destroy).pack(pady=10)

    def _populate_correlation_matrix_tab(self, tab, corr_matrix, corr_type):
        """Crea y llena un Treeview para una matriz de correlación en una pestaña dada."""
        label = ttk.Label(tab, text=f"Matriz de Correlación de {corr_type}. Valores |r| > 0.7 resaltados.", wraplength=650)
        label.pack(pady=10, padx=10)

        tree_frame = ttk.Frame(tab)
        tree_frame.pack(fill="both", expand=True, padx=10, pady=5)

        columns = ["Variable"] + list(corr_matrix.columns)
        tree = ttk.Treeview(tree_frame, columns=columns, show="headings")

        tree.heading("Variable", text="Variable")
        tree.column("Variable", width=120, anchor="w", stretch=False)
        for col in corr_matrix.columns:
            tree.heading(col, text=col)
            tree.column(col, width=80, anchor="e", stretch=True)

        # Tags para coloreado
        tree.tag_configure('high_corr', background='salmon')
        tree.tag_configure('perfect_corr', background='lightgrey')

        for index, row in corr_matrix.iterrows():
            values = [index] + [f"{val:.3f}" for val in row]
            # Determinar si alguna correlación en la fila es alta (excluyendo la diagonal)
            has_high_corr = any(abs(val) > 0.7 and abs(val) < 1.0 for val in row)
            tags = ('high_corr',) if has_high_corr else ()
            tree.insert("", "end", values=values, tags=tags)

        ysb = ttk.Scrollbar(tree_frame, orient="vertical", command=tree.yview)
        xsb = ttk.Scrollbar(tree_frame, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=ysb.set, xscrollcommand=xsb.set)
        ysb.pack(side="right", fill="y")
        xsb.pack(side="bottom", fill="x")
        tree.pack(fill="both", expand=True)


    def _generate_univariate_forest_plot(self, univariate_models_data):
        """Genera un Forest Plot a partir de una lista de modelos univariados."""
        if not univariate_models_data:
            self.log("No hay datos de modelos univariados para generar el Forest Plot.", "WARN")
            return

        plot_data = []
        for model_dict in univariate_models_data:
            summary_df = model_dict.get('metrics', {}).get('summary_df')
            if summary_df is None or summary_df.empty:
                continue

            # Para univariados, el summary_df debería tener una sola fila.
            # El nombre de la variable está en el índice.
            var_name = summary_df.index[0]
            hr = summary_df['exp(coef)'].iloc[0]
            lower_ci = summary_df['exp(coef) lower 95%'].iloc[0]
            upper_ci = summary_df['exp(coef) upper 95%'].iloc[0]
            p_val = summary_df['p'].iloc[0]

            plot_data.append({
                'var_name': var_name,
                'hr': hr,
                'lower_ci': lower_ci,
                'upper_ci': upper_ci,
                'p_val': p_val
            })

        if not plot_data:
            self.log("No se pudieron extraer datos de HR para el Forest Plot de univariados.", "WARN")
            messagebox.showwarning("Sin Datos para Gráfico", "No se encontraron datos de Hazard Ratio en los modelos univariados para generar el gráfico.", parent=self.parent_for_dialogs)
            return

        df_plot = pd.DataFrame(plot_data)
        df_plot.sort_values('hr', inplace=True) # Ordenar por HR ascendente por defecto

        try:
            fig, ax = plt.subplots(figsize=(10, max(4, len(df_plot) * 0.4)))
            y_pos = np.arange(len(df_plot))

            ax.errorbar(df_plot['hr'], y_pos, xerr=[df_plot['hr'] - df_plot['lower_ci'], df_plot['upper_ci'] - df_plot['hr']],
                        fmt='o', capsize=5, color='k', ms=5, elinewidth=1.2)

            ax.set_yticks(y_pos)
            ax.set_yticklabels(df_plot['var_name'])
            ax.invert_yaxis()
            ax.axvline(1.0, color='gray', ls='--', lw=0.8)

            opts = self.current_plot_options.copy()
            opts['title'] = opts.get('title') or "Forest Plot de Análisis Univariado"
            opts['xlabel'] = opts.get('xlabel') or "Hazard Ratio (HR) con IC 95%"

            apply_plot_options(ax, opts, self.log)
            plt.tight_layout()

            self._create_plot_window(fig, "Forest Plot Univariado")
            self.log("Forest Plot de modelos univariados generado exitosamente.", "SUCCESS")

        except Exception as e:
            self.log(f"Error generando Forest Plot de univariados: {e}", "ERROR")
            traceback.print_exc(limit=3)
            messagebox.showerror("Error de Gráfico", f"No se pudo generar el Forest Plot de univariados:\n{e}", parent=self.parent_for_dialogs)


    def show_new_calibration_plots(self):
        self.log("Attempting to show new calibration plots...", "INFO")
        if not self._check_model_selected_and_valid():
            return # Message already shown by _check_model_selected_and_valid

        model_dict = self.selected_model_in_treeview
        model_name = model_dict.get('model_name', 'N/A')

        oos_predictions_data = model_dict.get("oos_predictions")

        if not oos_predictions_data:
            messagebox.showwarning("Datos No Disponibles",
                                   f"No se encontraron predicciones Out-of-Sample para el modelo '{model_name}'.\n"
                                   "Asegúrese de que la Validación Cruzada ('Calcular C-Index con CV') se ejecutó al crear este modelo.",
                                   parent=self.parent_for_dialogs)
            self.log(f"No hay datos OOS para calibración en modelo '{model_name}'.", "WARN")
            return

        self.log(f"Datos OOS encontrados para el modelo '{model_name}'. {len(oos_predictions_data)} sujetos.", "INFO")

        # --- Replace placeholder with this new logic ---

        # Determine available stratification variables from self.data
        available_vars_for_strat = []
        if self.data is not None:
            excluded_cols = [self.combo_col_tiempo.get(), self.combo_col_evento.get()]
            # For simplicity, allow all non-time/event columns. Can be refined later.
            available_vars_for_strat = [col for col in self.data.columns if col not in excluded_cols]
        else:
            self.log("self.data is None, no stratification variables available.", "WARN")
            # Dialog will handle empty list if self.data is None

        dialog = CalibrationPlotOptionsDialog(parent=self, # Pass app instance as parent
                                              available_strat_vars=sorted(list(set(available_vars_for_strat))),
                                              log_func=self.log)

        if dialog.result is None:
            self.log("Opciones de calibración canceladas por el usuario.", "INFO")
            return

        user_choices = dialog.result # This now contains 'time_horizon_str' and 'oos_plot_choice'
        oos_plot_type_selected = user_choices['oos_plot_choice']
        plot_specific_title = "" # Initialize

        fig_cal_oos, ax_cal_oos = plt.subplots(figsize=(8, 8)) # Create figure once

        try:
            if oos_plot_type_selected == 'calibration':
                time_h_calib = float(user_choices['time_horizon_str']) # Validated in dialog for this type
                plot_t_calib = user_choices['plot_type']
                strat_v_calib = user_choices['strat_var']
                group_by_deciles_choice_calib = user_choices.get('group_by_deciles', False)

                if plot_t_calib == 'decile':
                    self._generate_decile_calibration_plot_oos(oos_predictions_data, time_h_calib, ax_cal_oos)
                    plot_specific_title = f"Calibración OOS por Deciles (t={time_h_calib:.2f})"
                elif plot_t_calib == 'stratified':
                    if not strat_v_calib:
                        messagebox.showerror("Error", "No se seleccionó variable de estratificación.", parent=self.parent_for_dialogs)
                        self.log("Calibración estratificada sin variable de estratificación.", "ERROR")
                        plt.close(fig_cal_oos); return
                    self._generate_stratified_calibration_plot_oos(oos_predictions_data, time_h_calib, strat_v_calib, ax_cal_oos, group_quantitative_by_deciles=group_by_deciles_choice_calib)
                    plot_specific_title = f"Calibración OOS por '{strat_v_calib}' (t={time_h_calib:.2f})"
                else:
                    self.log(f"Tipo de gráfico de calibración desconocido: {plot_t_calib}", "ERROR")
                    messagebox.showerror("Error", f"Tipo de gráfico de calibración desconocido: {plot_t_calib}", parent=self.parent_for_dialogs)
                    plt.close(fig_cal_oos); return

            elif oos_plot_type_selected == 'correlation_time':
                time_horizon_str = user_choices['time_horizon_str']
                show_pearson = user_choices.get('show_pearson', False)
                show_spearman = user_choices.get('show_spearman', False)
                self.log(f"Solicitado gráfico de correlación vs tiempo. Pearson: {show_pearson}, Spearman: {show_spearman}, Tiempos: '{time_horizon_str}'", "INFO")

                model_time_col = model_dict.get('time_col_for_model', 'Tiempo')

                correlation_data = self._calculate_calibration_correlations_over_time(
                    model_dict,
                    time_horizon_str,
                    show_pearson,
                    show_spearman
                )

                if not correlation_data:
                    self.log("No se generaron datos de correlación.", "WARN")
                    messagebox.showwarning("Sin Datos",
                                           "No se pudieron calcular datos de correlación para los tiempos especificados.",
                                           parent=self.parent_for_dialogs)
                    if fig_cal_oos: plt.close(fig_cal_oos)
                    return

                ax_cal_oos.clear()

                self._generate_correlation_over_time_plot(
                    correlation_data,
                    model_time_col,
                    show_pearson,
                    show_spearman,
                    model_name,
                    ax_cal_oos
                )
                # Title for the window will be based on what _generate_correlation_over_time_plot sets on the axes
                plot_specific_title = ax_cal_oos.get_title() if ax_cal_oos.get_title() else f"Correlación vs Tiempo: {model_name}"

            else:
                self.log(f"Tipo de gráfico OOS desconocido: {oos_plot_type_selected}", "ERROR")
                messagebox.showerror("Error", f"Tipo de gráfico OOS desconocido: {oos_plot_type_selected}", parent=self.parent_for_dialogs)
                plt.close(fig_cal_oos); return

            if not ax_cal_oos.has_data():
                 self.log("El método de generación de gráfico no añadió datos al eje. No se mostrará la ventana.", "WARN")
                 plt.close(fig_cal_oos)
                 # Optionally show a messagebox to the user
                 messagebox.showwarning("Gráfico Vacío", "No se generaron datos válidos para el gráfico seleccionado.", parent=self.parent_for_dialogs)
                 return

            # Use plot_specific_title which is now correctly set for both calibration and correlation plots
            self._create_plot_window(fig_cal_oos, f"{plot_specific_title} - Modelo: {model_name}")

        except Exception as e_cal_main:
            self.log(f"Error al generar o mostrar el gráfico OOS: {e_cal_main}", "ERROR")
            if fig_cal_oos: plt.close(fig_cal_oos)
            traceback.print_exc(limit=3)
            messagebox.showerror("Error de Gráfico",
                               f"No se pudo generar el gráfico de calibración OOS:\n{e_cal_main}",
                               parent=self.parent_for_dialogs)
        # --- End of new logic ---


    def _generate_decile_calibration_plot_oos(self, oos_predictions_list, time_horizon_t, ax):
        self.log(f"Generando gráfico de calibración por deciles para t={time_horizon_t}...", "INFO")

        if not oos_predictions_list:
            self.log("No OOS prediction data provided for decile calibration plot.", "ERROR")
            ax.text(0.5, 0.5, "No hay datos OOS para generar el gráfico.", ha='center', va='center')
            return

        # 1. Data Preparation: Convert list of dicts to DataFrame and get predicted event prob at time_horizon_t
        subject_data = []
        for item in oos_predictions_list:
            pred_sf_series = item["predicted_survival_function"]
            # Interpolate survival probability at time_horizon_t
            # Ensure time_horizon_t is within the bounds of the series' index (time points)
            # Use .get(key, default) for series if time_horizon_t might not be exact index

            # Create a common time grid for interpolation if necessary, or interpolate directly
            # For simplicity, using direct interpolation and handling bounds.
            min_time_pred = pred_sf_series.index.min()
            max_time_pred = pred_sf_series.index.max()

            if time_horizon_t < min_time_pred:
                predicted_s_at_t = 1.0
            elif time_horizon_t > max_time_pred:
                predicted_s_at_t = pred_sf_series.iloc[-1] # Survival at last predicted time
            else:
                # Interpolate (linear should be fine for SF)
                predicted_s_at_t = np.interp(time_horizon_t, pred_sf_series.index, pred_sf_series.values)

            predicted_event_prob_at_t = 1.0 - predicted_s_at_t

            subject_data.append({
                "subject_id": item["subject_id"],
                "true_time": item["true_time"],
                "true_event": item["true_event"],
                "predicted_event_prob": predicted_event_prob_at_t
            })

        if not subject_data:
            self.log("No subject data after processing predictions for decile calibration.", "ERROR")
            ax.text(0.5, 0.5, "No se pudieron procesar las predicciones.", ha='center', va='center')
            return

        oos_df = pd.DataFrame(subject_data)
        oos_df.dropna(subset=["predicted_event_prob"], inplace=True) # Should not happen if handled above

        if oos_df.empty:
            self.log("DataFrame OOS vacío después de calcular probabilidades de evento predichas.", "ERROR")
            ax.text(0.5, 0.5, "DataFrame OOS vacío.", ha='center', va='center')
            return

        # 2. Decile Grouping
        try:
            # Ensure at least 10 unique prediction values for qcut to work well, or handle fewer.
            # If fewer than 10 unique values, qcut might create fewer than 10 bins or error.
            num_unique_preds = oos_df["predicted_event_prob"].nunique()
            n_quantiles = min(10, num_unique_preds) if num_unique_preds > 1 else 1 # Avoid error if only 1 unique value

            if n_quantiles <= 1 : # Not enough diversity for deciles
                 self.log(f"No hay suficientes valores predichos únicos ({num_unique_preds}) para crear deciles significativos. Se mostrará un solo punto si es posible.", "WARN")
                 # Create a single group if n_quantiles is 1
                 oos_df["decile"] = 0
            else:
                 oos_df["decile"] = pd.qcut(oos_df["predicted_event_prob"], q=n_quantiles, labels=False, duplicates='drop')

        except ValueError as e_qcut:
            self.log(f"Error al crear deciles con pd.qcut: {e_qcut}. Puede haber muy pocos puntos de datos o valores no únicos. Intentando agrupar por un solo grupo.", "WARN")
            oos_df["decile"] = 0 # Fallback to a single group

        # 3. Calculate X and Y Coordinates for Each Decile
        calibration_points = []
        for i, group_df in oos_df.groupby("decile"):
            if group_df.empty:
                continue

            mean_predicted_prob = group_df["predicted_event_prob"].mean()

            kmf_decile = KaplanMeierFitter()
            kmf_decile.fit(group_df["true_time"], event_observed=group_df["true_event"])

            # Get survival probability S(t) at time_horizon_t for the decile
            # This also needs interpolation or careful handling if t is not an event time
            survival_at_t_decile_df = kmf_decile.survival_function_at_times([time_horizon_t])
            self.log(f"Decile group {i}: survival_at_t_decile_df type: {type(survival_at_t_decile_df)}, shape: {survival_at_t_decile_df.shape if isinstance(survival_at_t_decile_df, pd.DataFrame) else 'N/A'}, empty: {survival_at_t_decile_df.empty if isinstance(survival_at_t_decile_df, pd.DataFrame) else 'N/A'}", "DEBUG")
            self.log(f"Decile group {i}: survival_at_t_decile_df head:\n{survival_at_t_decile_df.head().to_string() if isinstance(survival_at_t_decile_df, pd.DataFrame) and not survival_at_t_decile_df.empty else 'N/A'}", "DEBUG")

            if survival_at_t_decile_df is not None and not survival_at_t_decile_df.empty:
                if isinstance(survival_at_t_decile_df, pd.DataFrame):
                    observed_s_at_t_decile = survival_at_t_decile_df.iloc[0,0]
                    self.log(f"Decile group {i}: Accessed observed_s_at_t_decile from DataFrame.", "DEBUG")
                elif isinstance(survival_at_t_decile_df, pd.Series):
                    observed_s_at_t_decile = survival_at_t_decile_df.iloc[0]
                    self.log(f"Decile group {i}: Accessed observed_s_at_t_decile from Series.", "DEBUG")
                else:
                    observed_s_at_t_decile = 1.0 # Fallback
                    self.log(f"Decile group {i}: survival_at_t_decile_df is not DataFrame or Series (Type: {type(survival_at_t_decile_df)}). Defaulting observed_s_at_t_decile to 1.0.", "WARN")
            else:
                observed_s_at_t_decile = 1.0
                self.log(f"Decile group {i}: survival_at_t_decile_df was None or empty. Defaulting observed_s_at_t_decile to 1.0.", "WARN")

            observed_event_incidence_decile = 1.0 - observed_s_at_t_decile

            # Confidence Interval for observed incidence
            kmf_ci_sf_decile = kmf_decile.confidence_interval_survival_function_ # This is a DataFrame
            self.log(f"Decile group {i}: kmf_ci_sf_decile type: {type(kmf_ci_sf_decile)}, shape: {kmf_ci_sf_decile.shape if isinstance(kmf_ci_sf_decile, pd.DataFrame) else 'N/A'}, empty: {kmf_ci_sf_decile.empty if isinstance(kmf_ci_sf_decile, pd.DataFrame) else 'N/A'}", "DEBUG")
            if kmf_ci_sf_decile is None or (isinstance(kmf_ci_sf_decile, pd.DataFrame) and kmf_ci_sf_decile.empty):
                self.log(f"Decile group {i}: kmf_ci_sf_decile is None or empty. Skipping CI calculation for this group.", "WARN")
                y_error_lower = 0.0
                y_error_upper = 0.0
            else:
                # CI for S(t) is [S_lower, S_upper]. So CI for P(T<=t) = 1-S(t) is [1-S_upper, 1-S_lower]
                # Find CI for time_horizon_t (may need interpolation or selection of closest time)
                # For simplicity, find closest available time point in CI index
                ci_idx_time = kmf_ci_sf_decile.index.get_indexer([time_horizon_t], method='nearest')[0]
                s_lower_at_t = kmf_ci_sf_decile.iloc[ci_idx_time, 0] # Lower CI for S(t)
                s_upper_at_t = kmf_ci_sf_decile.iloc[ci_idx_time, 1] # Upper CI for S(t)

                ci_observed_incidence_lower = 1.0 - s_upper_at_t
                ci_observed_incidence_upper = 1.0 - s_lower_at_t

                # Ensure error magnitudes are positive for errorbar
                y_error_lower = abs(observed_event_incidence_decile - ci_observed_incidence_lower)
                y_error_upper = abs(ci_observed_incidence_upper - observed_event_incidence_decile)

            calibration_points.append({
                "x_pred": mean_predicted_prob,
                "y_obs": observed_event_incidence_decile,
                "y_err_lower": y_error_lower,
                "y_err_upper": y_error_upper
            })

        if not calibration_points:
            self.log("No se generaron puntos de calibración.", "ERROR")
            ax.text(0.5, 0.5, "No se pudieron generar los puntos de calibración.", ha='center', va='center')
            return

        cal_df = pd.DataFrame(calibration_points)

        # 4. Plotting
        ax.plot(cal_df["x_pred"], cal_df["y_obs"], marker='o', linestyle='-', label="Calibración por Deciles")

        # Construct yerr appropriately for errorbar
        y_errors_for_plot = [cal_df["y_err_lower"].values, cal_df["y_err_upper"].values]
        ax.errorbar(cal_df["x_pred"], cal_df["y_obs"],
                    yerr=y_errors_for_plot,
                    fmt='none', ecolor='gray', capsize=3, elinewidth=1)

        ax.plot([0, 1], [0, 1], linestyle='--', color='red', label="Calibración Perfecta")

        ax.set_xlabel("Probabilidad Predicha de Evento P(T <= t)")
        ax.set_ylabel("Probabilidad Observada de Evento (Kaplan-Meier)")
        ax.set_title(f"Calibración OOS por Deciles (t={time_horizon_t:.2f})")
        ax.legend()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(True, linestyle=':', alpha=0.7)

        # --- Calcular y loguear Correlaciones para el gráfico de deciles ---
        pearson_r_cal = np.nan
        spearman_rho_cal = np.nan

        if cal_df is not None and len(cal_df) >= 2:
            x_pred_cal = cal_df['x_pred']
            y_obs_cal = cal_df['y_obs']

            # Pearson
            if np.std(x_pred_cal) < 1e-6 or np.std(y_obs_cal) < 1e-6:
                self.log("Advertencia: Varianza cero o muy baja en datos para Pearson en gráfico de deciles. Correlación será NaN.", "WARN")
            else:
                try:
                    pearson_r_cal, _ = scipy.stats.pearsonr(x_pred_cal, y_obs_cal)
                except ValueError as e_pearson:
                    self.log(f"Error calculando Pearson en gráfico de deciles: {e_pearson}. Correlación será NaN.", "ERROR")
                except Exception as e_gen_pearson:
                    self.log(f"Error general calculando Pearson en gráfico de deciles: {e_gen_pearson}. Correlación será NaN.", "ERROR")

            # Spearman
            # Spearman es más robusto a la varianza, pero puede dar NaN si los datos son perfectamente constantes o hay muy pocos puntos.
            if np.std(x_pred_cal) < 1e-6 or np.std(y_obs_cal) < 1e-6: # Similar check for consistency
                 self.log("Advertencia: Varianza cero o muy baja en datos para Spearman en gráfico de deciles. Correlación será NaN.", "WARN")
            else:
                try:
                    spearman_rho_cal, _ = scipy.stats.spearmanr(x_pred_cal, y_obs_cal)
                except ValueError as e_spearman:
                    self.log(f"Error calculando Spearman en gráfico de deciles: {e_spearman}. Correlación será NaN.", "ERROR")
                except Exception as e_gen_spearman:
                    self.log(f"Error general calculando Spearman en gráfico de deciles: {e_gen_spearman}. Correlación será NaN.", "ERROR")

            pearson_str_cal = f"{pearson_r_cal:.3f}" if pd.notna(pearson_r_cal) else "N/A"
            spearman_str_cal = f"{spearman_rho_cal:.3f}" if pd.notna(spearman_rho_cal) else "N/A"
            self.log(f"Correlaciones para gráfico de calibración por deciles (t={time_horizon_t}): Pearson={pearson_str_cal}, Spearman={spearman_str_cal}, N Puntos={len(cal_df)}", "INFO")

            # Display correlations on the plot
            corr_text_parts = []
            if pd.notna(pearson_r_cal):
                corr_text_parts.append(f"Pearson r: {pearson_r_cal:.2f}")
            if pd.notna(spearman_rho_cal):
                corr_text_parts.append(f"Spearman ρ: {spearman_rho_cal:.2f}")

            if corr_text_parts:
                corr_display_text = "\n".join(corr_text_parts)
                ax.text(0.95, 0.05, corr_display_text,
                        transform=ax.transAxes,
                        fontsize=9,
                        verticalalignment='bottom',
                        horizontalalignment='right',
                        bbox=dict(boxstyle='round,pad=0.3', fc='wheat', alpha=0.5))
        else:
            self.log(f"No se pueden calcular correlaciones para gráfico de deciles (t={time_horizon_t}): Menos de 2 puntos de datos en cal_df (N={len(cal_df) if cal_df is not None else 0}).", "WARN")

        self.log("Gráfico de calibración por deciles OOS generado.", "SUCCESS")


    def _generate_stratified_calibration_plot_oos(self, oos_predictions_list, time_horizon_t, stratification_variable_name, ax, group_quantitative_by_deciles: bool):
        self.log(f"Generando gráfico de calibración estratificado por '{stratification_variable_name}' para t={time_horizon_t} (Agrupar Cuantitativas: {group_quantitative_by_deciles})...", "INFO")

        if not oos_predictions_list:
            self.log("No OOS prediction data provided for stratified calibration plot.", "ERROR")
            ax.text(0.5, 0.5, "No hay datos OOS para generar el gráfico.", ha='center', va='center')
            return

        if self.data is None or self.data.empty: # More explicit check for DataFrame
            self.log("Dataset original (self.data) no cargado o vacío. No se puede obtener variable de estratificación.", "ERROR")
            messagebox.showerror("Error de Datos", "Dataset original no disponible para obtener la variable de estratificación.", parent=self.parent_for_dialogs)
            ax.text(0.5, 0.5, "Dataset original no disponible.", ha='center', va='center')
            return

        if stratification_variable_name not in self.data.columns:
            self.log(f"Variable de estratificación '{stratification_variable_name}' no encontrada en self.data.", "ERROR")
            messagebox.showerror("Error de Variable", f"Variable de estratificación '{stratification_variable_name}' no encontrada.", parent=self.parent_for_dialogs)
            ax.text(0.5, 0.5, f"Variable '{stratification_variable_name}' no encontrada.", ha='center', va='center')
            return

        # 1. Data Preparation
        subject_data_for_strat_plot = []
        for item in oos_predictions_list:
            pred_sf_series = item["predicted_survival_function"]
            min_time_pred = pred_sf_series.index.min()
            max_time_pred = pred_sf_series.index.max()

            if time_horizon_t < min_time_pred: predicted_s_at_t = 1.0
            elif time_horizon_t > max_time_pred: predicted_s_at_t = pred_sf_series.iloc[-1]
            else: predicted_s_at_t = np.interp(time_horizon_t, pred_sf_series.index, pred_sf_series.values)

            predicted_event_prob_at_t = 1.0 - predicted_s_at_t

            subject_data_for_strat_plot.append({
                "subject_id": item["subject_id"], # This is the original index
                "true_time": item["true_time"],
                "true_event": item["true_event"],
                "predicted_event_prob": predicted_event_prob_at_t
            })

        if not subject_data_for_strat_plot:
            self.log("No subject data after processing OOS predictions for stratified calibration.", "ERROR")
            ax.text(0.5, 0.5, "No se pudieron procesar las predicciones OOS.", ha='center', va='center')
            return

        oos_df = pd.DataFrame(subject_data_for_strat_plot)
        oos_df.dropna(subset=["predicted_event_prob"], inplace=True)

        self.log(f"Stratified Plot: Preparing for merge. oos_df columns: {list(oos_df.columns)}", "DEBUG")
        self.log(f"Stratified Plot: stratification_variable_name: '{stratification_variable_name}'", "DEBUG")
        if self.data is not None:
            self.log(f"Stratified Plot: self.data is present. Index name: {self.data.index.name}", "DEBUG")
            self.log(f"Stratified Plot: self.data columns: {list(self.data.columns)}", "DEBUG")
            self.log(f"Stratified Plot: self.data head (first 3 rows):\n{self.data.head(3).to_string()}", "DEBUG")
            if 'subject_id' in oos_df.columns:
                self.log(f"Stratified Plot: oos_df['subject_id'] head (first 3):\n{oos_df['subject_id'].head(3).to_string()}", "DEBUG")
            else:
                self.log(f"Stratified Plot: 'subject_id' not in oos_df columns before merge.", "WARN")
        else:
            self.log(f"Stratified Plot: self.data is None. Cannot proceed with merge.", "ERROR")
            # It's already checked earlier, but as a safeguard for this specific logging context
            ax.text(0.5, 0.5, "Error: self.data es None.", ha='center', va='center')
            return

        self.log(f"Stratified Plot: oos_df 'subject_id' head:\n{oos_df['subject_id'].head().to_string()}", "DEBUG")
        self.log(f"Stratified Plot: self.data.index head:\n{self.data.index.to_series().head().to_string()}", "DEBUG")
        self.log(f"Stratified Plot: Attempting merge with stratification variable: '{stratification_variable_name}'", "DEBUG")

        if stratification_variable_name not in self.data.columns:
            self.log(f"Critical Error: Stratification variable '{stratification_variable_name}' is not a column in self.data. Available columns: {self.data.columns.tolist()}", "ERROR")
            messagebox.showerror("Error Interno", f"La variable de estratificación '{stratification_variable_name}' no se encontró en las columnas de los datos principales.", parent=self.parent_for_dialogs if hasattr(self, 'parent_for_dialogs') else None)
            if ax: ax.text(0.5, 0.5, f"Error: Variable '{stratification_variable_name}' no en datos.", ha='center', va='center')
            return

        try:
            # Merge oos_df (left) with the selected stratification variable from self.data (right).
            # 'subject_id' in oos_df contains original index values that should align with self.data.index.
            oos_df = oos_df.merge(
                self.data[[stratification_variable_name]], # Select only the necessary column from self.data
                left_on='subject_id',      # Use the 'subject_id' column from oos_df (which has original index values)
                right_index=True,          # Match with the index of self.data
                how='left',                # Keep all oos_df rows
                suffixes=('_oos', '_originaldata') # Suffixes in case 'stratification_variable_name' was somehow 'subject_id' (though unlikely here)
            )
            self.log(f"Stratified Plot: Merge successful. oos_df columns after merge: {list(oos_df.columns)}", "DEBUG")
            if stratification_variable_name in oos_df.columns and not oos_df[stratification_variable_name].isnull().all():
                self.log(f"Stratified Plot: '{stratification_variable_name}' column head after merge:\n{oos_df[stratification_variable_name].head().to_string()}", "DEBUG")
            elif stratification_variable_name not in oos_df.columns:
                 self.log(f"Stratified Plot: WARNING - '{stratification_variable_name}' column NOT FOUND after merge. This is unexpected.", "WARN")
            else:
                self.log(f"Stratified Plot: '{stratification_variable_name}' column is all NaN after merge. Check if 'subject_id' values in OOS data match indices in main data or if column in main data is all NaN.", "WARN")

        except Exception as e_merge:
            self.log(f"Error durante el merge para la estratificación: {e_merge}", "ERROR")
            self.log(traceback.format_exc(), "DEBUG")
            messagebox.showerror("Error de Merge", f"No se pudo realizar el cruce de datos para la estratificación: {e_merge}", parent=self.parent_for_dialogs if hasattr(self, 'parent_for_dialogs') else None)
            if ax: ax.text(0.5, 0.5, "Error en cruce de datos.", ha='center', va='center')
            return

        # The existing logging and try-except blocks for isnull/dropna and groupby should follow this new merge logic.
        self.log(f"Stratified Plot: oos_df columns before isnull check: {list(oos_df.columns)}", "DEBUG")
        self.log(f"Stratified Plot: stratification_variable_name: '{stratification_variable_name}'", "DEBUG")
        try:
            if oos_df[stratification_variable_name].isnull().any():
                self.log(f"Algunos sujetos OOS no tienen valor para la variable de estratificación '{stratification_variable_name}'. Serán excluidos.", "WARN")
                oos_df.dropna(subset=[stratification_variable_name], inplace=True)
        except KeyError as e_key_strat_check:
            self.log(f"KeyError al verificar/eliminar NaNs para '{stratification_variable_name}' en oos_df. Columns: {list(oos_df.columns)}", "ERROR")
            self.log(f"Error: {e_key_strat_check}", "ERROR")
            ax.text(0.5, 0.5, f"Error interno: Clave '{stratification_variable_name}' no encontrada post-merge.", ha='center', va='center')
            return

        if oos_df.empty: # This check remains outside, after potential dropna
            self.log("DataFrame OOS vacío después de merge/dropna para estratificación.", "ERROR")
            ax.text(0.5, 0.5, "No hay datos para estratificar.", ha='center', va='center')
            return

        # 2. Determine Variable Type and Create Groups for Stratification
        group_by_column_name = stratification_variable_name
        stratum_name_prefix = ""
        # Ensure self.data and the stratification_variable_name column exist before checking dtype
        if self.data is None or stratification_variable_name not in self.data.columns:
            self.log(f"Error: self.data no está disponible o '{stratification_variable_name}' no es una columna válida.", "ERROR")
            ax.text(0.5, 0.5, "Error de datos para estratificación.", ha='center', va='center')
            return
        is_numeric_strat_var = pd.api.types.is_numeric_dtype(self.data[stratification_variable_name])

        perform_deciling = is_numeric_strat_var and group_quantitative_by_deciles
        plot_title_detail = "" # Will be set below

        if perform_deciling:
            self.log(f"Variable '{stratification_variable_name}' es numérico y se solicitó agrupar por deciles/cuantiles.", "INFO")
            try:
                oos_df['strat_group_numeric_deciles'] = pd.qcut(oos_df[stratification_variable_name], q=10, labels=False, duplicates='drop')
                group_by_column_name = 'strat_group_numeric_deciles'
                stratum_name_prefix = "Decil "
                plot_title_detail = f"por Deciles de '{stratification_variable_name}'"
                self.log("Deciles (q=10) creados para la variable numérica de estratificación.", "INFO")
            except ValueError:
                self.log(f"Error al crear 10 deciles para '{stratification_variable_name}'. Intentando 5 cuantiles.", "WARN")
                try:
                    oos_df['strat_group_numeric_deciles'] = pd.qcut(oos_df[stratification_variable_name], q=5, labels=False, duplicates='drop')
                    group_by_column_name = 'strat_group_numeric_deciles'
                    stratum_name_prefix = "Quintil "
                    plot_title_detail = f"por Quintiles de '{stratification_variable_name}'"
                    self.log("Quintiles (q=5) creados para la variable numérica de estratificación.", "INFO")
                except ValueError:
                    self.log(f"Error al crear 5 cuantiles para '{stratification_variable_name}'. Agrupando como única categoría.", "WARN")
                    oos_df['strat_group_numeric_deciles'] = 0 # Fallback to a single group
                    group_by_column_name = 'strat_group_numeric_deciles'
                    stratum_name_prefix = "Grupo "
                    plot_title_detail = f"para '{stratification_variable_name}' (agrupado)"
        else: # Categorical OR (Numeric AND checkbox for deciling is OFF)
            if is_numeric_strat_var: # Numeric but deciling checkbox was off
                self.log(f"Variable de estratificación '{stratification_variable_name}' es numérica, pero no se solicitó agrupar por deciles. Usando valores únicos.", "INFO")
                plot_title_detail = f"por Valores Únicos de '{stratification_variable_name}'"
                # Ensure the column is treated as categorical for grouping if it's numeric but not deciled
                if pd.api.types.is_numeric_dtype(oos_df[stratification_variable_name]):
                     oos_df[stratification_variable_name] = oos_df[stratification_variable_name].astype(str)
            else: # Categorical
                self.log(f"Variable de estratificación '{stratification_variable_name}' es categórica. Usando valores únicos.", "INFO")
                plot_title_detail = f"por Categorías de '{stratification_variable_name}'"

            group_by_column_name = stratification_variable_name
            stratum_name_prefix = ""


        calibration_points_strat = []
        self.log(f"Stratified Plot: Grouping by column '{group_by_column_name}'.", "DEBUG")

        try:
            for strat_value, group_df in oos_df.groupby(group_by_column_name):
                if group_df.empty or len(group_df) < 2:
                    self.log(f"Estrato '{strat_value}' (col: {group_by_column_name}) tiene muy pocos datos ({len(group_df)}). Saltando.", "WARN")
                    continue

                mean_predicted_prob_strat = group_df["predicted_event_prob"].mean()

                kmf_strat = KaplanMeierFitter()
                kmf_strat.fit(group_df["true_time"], event_observed=group_df["true_event"])

                survival_at_t_strat_df = kmf_strat.survival_function_at_times([time_horizon_t])
                self.log(f"Stratum '{strat_value}': survival_at_t_strat_df type: {type(survival_at_t_strat_df)}, shape: {survival_at_t_strat_df.shape if isinstance(survival_at_t_strat_df, pd.DataFrame) else 'N/A'}, empty: {survival_at_t_strat_df.empty if isinstance(survival_at_t_strat_df, pd.DataFrame) else 'N/A'}", "DEBUG")

                if survival_at_t_strat_df is not None and not survival_at_t_strat_df.empty:
                    if isinstance(survival_at_t_strat_df, pd.DataFrame):
                        observed_s_at_t_strat = survival_at_t_strat_df.iloc[0,0]
                    elif isinstance(survival_at_t_strat_df, pd.Series):
                        observed_s_at_t_strat = survival_at_t_strat_df.iloc[0]
                    else: observed_s_at_t_strat = 1.0
                else: observed_s_at_t_strat = 1.0

                observed_event_incidence_strat = 1.0 - observed_s_at_t_strat

                y_err_lower_strat, y_err_upper_strat = 0.0, 0.0
                try:
                    kmf_ci_sf_strat = kmf_strat.confidence_interval_survival_function_
                    self.log(f"Stratum '{strat_value}': kmf_ci_sf_strat type: {type(kmf_ci_sf_strat)}, shape: {kmf_ci_sf_strat.shape if isinstance(kmf_ci_sf_strat, pd.DataFrame) else 'N/A'}, empty: {kmf_ci_sf_strat.empty if isinstance(kmf_ci_sf_strat, pd.DataFrame) else 'N/A'}", "DEBUG")

                    if kmf_ci_sf_strat is not None and not kmf_ci_sf_strat.empty and not kmf_ci_sf_strat.index.empty:
                        ci_idx_time_strat = kmf_ci_sf_strat.index.get_indexer([time_horizon_t], method='nearest')[0]
                        s_lower_at_t_strat = kmf_ci_sf_strat.iloc[ci_idx_time_strat, 0]
                        s_upper_at_t_strat = kmf_ci_sf_strat.iloc[ci_idx_time_strat, 1]
                        ci_observed_incidence_lower_strat = 1.0 - s_upper_at_t_strat
                        ci_observed_incidence_upper_strat = 1.0 - s_lower_at_t_strat
                        y_err_lower_strat = abs(observed_event_incidence_strat - ci_observed_incidence_lower_strat)
                        y_err_upper_strat = abs(ci_observed_incidence_upper_strat - observed_event_incidence_strat)
                    else:
                        self.log(f"Stratum '{strat_value}' (col: {group_by_column_name}): kmf_ci_sf_strat is None, empty, or has empty index. CI for this stratum will be zero.", "WARN")
                except Exception as e_ci_strat:
                    self.log(f"Stratum '{strat_value}' (col: {group_by_column_name}): Error calculating CI: {e_ci_strat}. CI for this stratum will be zero.", "ERROR")

                # Determine stratum name for legend
                if perform_deciling: # If deciling was performed, strat_value is an integer group number
                    current_stratum_name = f"{stratum_name_prefix}{int(strat_value)}"
                else: # Categorical, or numeric not deciled: strat_value is the actual category/value
                    current_stratum_name = str(strat_value)

                calibration_points_strat.append({
                    "stratum_name": current_stratum_name,
                    "x_pred": mean_predicted_prob_strat,
                    "y_obs": observed_event_incidence_strat,
                    "y_err": [[y_err_lower_strat], [y_err_upper_strat]]
                })
        except KeyError as e_key_strat_group:
            self.log(f"KeyError en groupby para '{group_by_column_name}' en oos_df. Columns: {list(oos_df.columns)}", "ERROR")
            self.log(f"Error: {e_key_strat_group}", "ERROR")
            ax.text(0.5, 0.5, f"Error interno: Clave '{group_by_column_name}' no encontrada para groupby.", ha='center', va='center')
            return

        if not calibration_points_strat:
            self.log("No se generaron puntos de calibración estratificados.", "ERROR")
            ax.text(0.5, 0.5, "No se pudieron generar puntos de calibración estratificados.", ha='center', va='center')
            return

        # --- Calcular Correlaciones para el gráfico estratificado (basado en los puntos de los estratos) ---
        pearson_r_strat = np.nan
        spearman_rho_strat = np.nan
        cal_df_strat = pd.DataFrame(calibration_points_strat)

        if not cal_df_strat.empty and len(cal_df_strat) >= 2:
            x_pred_strat = cal_df_strat['x_pred']
            y_obs_strat = cal_df_strat['y_obs']

            # Pearson
            if np.std(x_pred_strat) < 1e-6 or np.std(y_obs_strat) < 1e-6:
                self.log("Advertencia: Varianza cero o muy baja en datos para Pearson en gráfico estratificado. Correlación será NaN.", "WARN")
            else:
                try:
                    pearson_r_strat, _ = scipy.stats.pearsonr(x_pred_strat, y_obs_strat)
                except ValueError as e_pearson:
                    self.log(f"Error calculando Pearson en gráfico estratificado: {e_pearson}. Correlación será NaN.", "ERROR")
                except Exception as e_gen_pearson:
                    self.log(f"Error general calculando Pearson en gráfico estratificado: {e_gen_pearson}. Correlación será NaN.", "ERROR")

            # Spearman
            if np.std(x_pred_strat) < 1e-6 or np.std(y_obs_strat) < 1e-6:
                self.log("Advertencia: Varianza cero o muy baja en datos para Spearman en gráfico estratificado. Correlación será NaN.", "WARN")
            else:
                try:
                    spearman_rho_strat, _ = scipy.stats.spearmanr(x_pred_strat, y_obs_strat)
                except ValueError as e_spearman:
                    self.log(f"Error calculando Spearman en gráfico estratificado: {e_spearman}. Correlación será NaN.", "ERROR")
                except Exception as e_gen_spearman:
                    self.log(f"Error general calculando Spearman en gráfico estratificado: {e_gen_spearman}. Correlación será NaN.", "ERROR")

            pearson_str_display = f"{pearson_r_strat:.2f}" if pd.notna(pearson_r_strat) else "N/A"
            spearman_str_display = f"{spearman_rho_strat:.2f}" if pd.notna(spearman_rho_strat) else "N/A"
            self.log(f"Correlaciones para gráfico estratificado (basadas en {len(cal_df_strat)} puntos de estratos): Pearson={pearson_str_display}, Spearman={spearman_str_display}", "INFO")

        elif not cal_df_strat.empty and len(cal_df_strat) < 2 :
             self.log(f"No se pueden calcular correlaciones para gráfico estratificado: Menos de 2 puntos de estratos (N Puntos={len(cal_df_strat)}).", "WARN")

        # 3. Plotting
        for point_data in calibration_points_strat: # calibration_points_strat is the original list of dicts
            ax.errorbar(point_data["x_pred"], point_data["y_obs"],
                        yerr=np.array(point_data["y_err"]).reshape(2,-1),
                        fmt='o', label=point_data["stratum_name"], capsize=3, elinewidth=1, markersize=6)

        ax.plot([0, 1], [0, 1], linestyle='--', color='red', label="Calibración Perfecta")

        ax.set_xlabel("Probabilidad Predicha de Evento P(T <= t)")
        ax.set_ylabel("Probabilidad Observada de Evento (Kaplan-Meier)")
        ax.set_title(f"Calibración OOS {plot_title_detail} (t={time_horizon_t:.2f})")
        ax.legend(title=f"{stratification_variable_name}{' (Grupos)' if perform_deciling else ''}", fontsize='small')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(True, linestyle=':', alpha=0.7)

        # --- Display Correlations on Stratified Plot ---
        # pearson_r_strat and spearman_rho_strat are calculated earlier in this method
        if 'cal_df_strat' in locals() and isinstance(cal_df_strat, pd.DataFrame) and len(cal_df_strat) >=2: # Check if cal_df_strat was created and had enough points
            corr_text_parts_strat = []
            if pd.notna(pearson_r_strat):
                corr_text_parts_strat.append(f"Pearson r (estratos): {pearson_r_strat:.2f}")
            if pd.notna(spearman_rho_strat):
                corr_text_parts_strat.append(f"Spearman ρ (estratos): {spearman_rho_strat:.2f}")

            if corr_text_parts_strat:
                corr_display_text_strat = "\n".join(corr_text_parts_strat)
                ax.text(0.95, 0.05, corr_display_text_strat,
                        transform=ax.transAxes,
                        fontsize=9,
                        verticalalignment='bottom',
                        horizontalalignment='right',
                        bbox=dict(boxstyle='round,pad=0.3', fc='lightskyblue', alpha=0.5))

        self.log(f"Gráfico de calibración OOS estratificado por '{stratification_variable_name}' generado.", "SUCCESS")

    def _calculate_calibration_correlations_over_time(self,
                                                     model_dict: dict,
                                                     time_horizon_str: str,
                                                     calculate_pearson: bool,
                                                     calculate_spearman: bool) -> list:
        self.log(f"Calculando correlaciones de calibración OOS. Tiempos: '{time_horizon_str}', Pearson: {calculate_pearson}, Spearman: {calculate_spearman}", "INFO")

        time_points = parse_time_horizon_string(time_horizon_str, self.log)
        if not time_points:
            self.log("No se especificaron puntos de tiempo válidos para el cálculo de correlación. Retornando lista vacía.", "WARN")
            return []

        oos_predictions = model_dict.get("oos_predictions")
        if not oos_predictions:
            self.log("No se encontraron datos de 'oos_predictions' en el modelo. No se pueden calcular correlaciones.", "ERROR")
            messagebox.showerror("Error de Datos", "Predicciones Out-of-Sample no encontradas en el modelo.", parent=self.parent_for_dialogs)
            return []

        correlation_results = []

        for t_horizon in time_points:
            self.log(f"Procesando correlaciones para t_horizon = {t_horizon}", "DEBUG")
            predicted_probs_at_t = []
            observed_status_at_t = []

            for pred_item in oos_predictions:
                true_time = pred_item["true_time"]
                true_event = pred_item["true_event"]
                pred_sf_series = pred_item["predicted_survival_function"]

                # Interpolar S(t) y calcular P_pred(evento) = 1 - S(t)
                s_at_t = 1.0 # Default si t_horizon está antes del inicio de la curva SF
                if not pred_sf_series.empty:
                    min_time_pred = pred_sf_series.index.min()
                    max_time_pred = pred_sf_series.index.max()
                    if t_horizon < min_time_pred:
                        s_at_t = 1.0
                    elif t_horizon > max_time_pred:
                        s_at_t = pred_sf_series.iloc[-1]
                    else:
                        s_at_t = np.interp(t_horizon, pred_sf_series.index, pred_sf_series.values)
                p_pred_at_t = 1.0 - s_at_t

                # Determinar estado observado en t_horizon
                # Excluir sujetos censurados en o antes de t_horizon sin haber experimentado el evento
                if true_time <= t_horizon and true_event == 0:
                    continue # Excluir para este t_horizon

                obs_at_t = 1 if (true_time <= t_horizon and true_event == 1) else 0

                predicted_probs_at_t.append(p_pred_at_t)
                observed_status_at_t.append(obs_at_t)

            if len(predicted_probs_at_t) < 2:
                self.log(f"No hay suficientes datos ({len(predicted_probs_at_t)}) para calcular correlación en t={t_horizon}. Saltando.", "WARN")
                correlation_results.append({'time': t_horizon, 'pearson': np.nan, 'spearman': np.nan, 'n_obs': len(predicted_probs_at_t)})
                continue

            pearson_r, spearman_rho = np.nan, np.nan # Usar np.nan como default

            if calculate_pearson:
                try:
                    # Verificar si hay varianza en los datos antes de llamar a pearsonr
                    if np.std(predicted_probs_at_t) == 0 or np.std(observed_status_at_t) == 0:
                        self.log(f"Varianza cero en datos para Pearson en t={t_horizon}. Pearson será NaN.", "WARN")
                        pearson_r = np.nan # O podría ser 0.0 si se considera apropiado
                    else:
                        pearson_r, _ = scipy.stats.pearsonr(predicted_probs_at_t, observed_status_at_t)
                except ValueError as e_pearson:
                    self.log(f"Error calculando Pearson en t={t_horizon}: {e_pearson}. Pearson será NaN.", "ERROR")
                    pearson_r = np.nan
                except Exception as e_gen_pearson: # Captura más general
                    self.log(f"Error general calculando Pearson en t={t_horizon}: {e_gen_pearson}. Pearson será NaN.", "ERROR")
                    pearson_r = np.nan


            if calculate_spearman:
                try:
                    # Spearman también puede fallar con varianza cero, aunque es más robusto a distribuciones.
                    if np.std(predicted_probs_at_t) == 0 or np.std(observed_status_at_t) == 0:
                         self.log(f"Varianza cero en datos para Spearman en t={t_horizon}. Spearman será NaN.", "WARN")
                         spearman_rho = np.nan
                    else:
                        spearman_rho, _ = scipy.stats.spearmanr(predicted_probs_at_t, observed_status_at_t)
                except ValueError as e_spearman:
                    self.log(f"Error calculando Spearman en t={t_horizon}: {e_spearman}. Spearman será NaN.", "ERROR")
                    spearman_rho = np.nan
                except Exception as e_gen_spearman:
                    self.log(f"Error general calculando Spearman en t={t_horizon}: {e_gen_spearman}. Spearman será NaN.", "ERROR")
                    spearman_rho = np.nan

            correlation_results.append({
                'time': t_horizon,
                'pearson': pearson_r,
                'spearman': spearman_rho,
                'n_obs': len(predicted_probs_at_t) # Guardar N para este tiempo
            })
            pearson_str = f'{pearson_r:.3f}' if pd.notna(pearson_r) else 'N/A'
            spearman_str = f'{spearman_rho:.3f}' if pd.notna(spearman_rho) else 'N/A'
            self.log(f"Resultados correlación t={t_horizon}: Pearson={pearson_str}, Spearman={spearman_str}, N={len(predicted_probs_at_t)}", "DEBUG")

        return correlation_results

    def _generate_correlation_over_time_plot(self,
                                             correlation_results: list,
                                             time_col_name_for_axis_label: str,
                                             plot_pearson: bool,
                                             plot_spearman: bool,
                                             model_name_for_title: str,
                                             ax: plt.Axes):
        if not correlation_results:
            self.log("No hay datos de correlación para graficar.", "WARN")
            ax.text(0.5, 0.5, "No hay datos de correlación para graficar.",
                    ha='center', va='center', fontsize=12, color='grey')
            return

        df_corr = pd.DataFrame(correlation_results)
        df_corr.sort_values(by='time', inplace=True)

        lines_plotted = 0
        if plot_pearson and 'pearson' in df_corr.columns and df_corr['pearson'].notna().any():
            ax.plot(df_corr['time'], df_corr['pearson'], marker='o', linestyle='-', label="Pearson")
            lines_plotted += 1

        if plot_spearman and 'spearman' in df_corr.columns and df_corr['spearman'].notna().any():
            ax.plot(df_corr['time'], df_corr['spearman'], marker='x', linestyle='--', label="Spearman")
            lines_plotted +=1

        ax.set_xlabel(f"Horizonte de Tiempo ({time_col_name_for_axis_label})")
        ax.set_ylabel("Coeficiente de Correlación")
        ax.set_title(f"Evolución de Correlación (Predicha vs. Observada) vs. Tiempo\nModelo: {model_name_for_title}")
        ax.set_ylim(-1.05, 1.05)
        ax.axhline(0, color='grey', linestyle='--', linewidth=0.8)

        if lines_plotted > 0:
            ax.legend()
        else:
            self.log("No se graficaron líneas de correlación (Pearson/Spearman no seleccionados o sin datos válidos).", "INFO")
            ax.text(0.5, 0.5, "No hay datos válidos de correlación para los tipos seleccionados.",
                    ha='center', va='center', fontsize=10, color='grey')

        ax.grid(True, linestyle=':', alpha=0.7)
        self.log(f"Gráfico de correlación vs tiempo generado para modelo '{model_name_for_title}'.", "INFO")


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

