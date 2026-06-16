#!/usr/bin/env python
# -*- coding: utf-8 -*-

# --- Importaciones Estándar de Python ---
from sklearn.model_selection import KFold, train_test_split
from lifelines.utils import concordance_index
from lifelines import CoxPHFitter, KaplanMeierFitter
import lifelines # Importar lifelines directamente para verificar la versión

# --- Uno's C-index (IPCW) + Antolini Ctd + Brier from scikit-survival ---
try:
    from sksurv.metrics import concordance_index_ipcw as _concordance_index_ipcw
    from sksurv.metrics import cumulative_dynamic_auc as _cumulative_dynamic_auc
    from sksurv.metrics import brier_score as _brier_score_cox
    from sksurv.metrics import integrated_brier_score as _integrated_brier_score_cox
except Exception:
    _concordance_index_ipcw = None
    _cumulative_dynamic_auc = None
    _brier_score_cox = None
    _integrated_brier_score_cox = None
# Usar este para evitar problemas con LogFormatter
from matplotlib.ticker import ScalarFormatter
from matplotlib import transforms as mtransforms
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
import copy
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


def format_c_index_display(value, ci=None, decimals=3, na_text="N/A"):
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return na_text
    if not np.isfinite(numeric_value):
        return na_text

    base_text = f"{numeric_value:.{decimals}f}"
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


def compute_mean_confidence_interval_from_samples(values, confidence_level=0.95, clip_min=None, clip_max=None):
    arr = np.asarray(values, dtype=float).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None

    center = float(np.nanmean(arr))
    if arr.size == 1:
        lower = upper = center
    else:
        stderr = float(np.nanstd(arr, ddof=1) / np.sqrt(arr.size))
        z_value = float(scipy.stats.norm.ppf(0.5 + (float(np.clip(confidence_level, 0.50, 0.999)) / 2.0)))
        margin = z_value * stderr if np.isfinite(stderr) else 0.0
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
    return (float(lower), float(upper))


def bootstrap_concordance_ci_from_scores(y_data, time_col, event_col, score_values, confidence_level=0.95, n_bootstrap=120, random_state=42):
    if not isinstance(y_data, pd.DataFrame) or y_data.empty or time_col not in y_data.columns or event_col not in y_data.columns:
        return None

    scores = np.asarray(score_values, dtype=float).reshape(-1)
    if scores.size != len(y_data) or scores.size < 5:
        return None

    times = pd.to_numeric(y_data[time_col], errors='coerce').to_numpy(dtype=float)
    events = pd.to_numeric(y_data[event_col], errors='coerce').fillna(0).to_numpy(dtype=float)
    valid_mask = np.isfinite(times) & np.isfinite(scores)
    if valid_mask.sum() < 5:
        return None

    times = times[valid_mask]
    scores = scores[valid_mask]
    events = events[valid_mask]
    if np.unique(events > 0).size < 2:
        return None

    rng = np.random.default_rng(random_state)
    population_idx = np.arange(times.size)
    sampled_cindices = []
    alpha = 1.0 - float(np.clip(confidence_level, 0.50, 0.999))

    for _ in range(int(max(20, n_bootstrap))):
        bootstrap_idx = rng.choice(population_idx, size=population_idx.size, replace=True)
        sampled_events = events[bootstrap_idx]
        if np.unique(sampled_events > 0).size < 2:
            continue
        try:
            c_value = float(concordance_index(times[bootstrap_idx], scores[bootstrap_idx], sampled_events))
        except Exception:
            continue
        if np.isfinite(c_value):
            sampled_cindices.append(c_value)

    if len(sampled_cindices) < 10:
        return None

    lower = float(np.nanquantile(sampled_cindices, alpha / 2.0))
    upper = float(np.nanquantile(sampled_cindices, 1.0 - (alpha / 2.0)))
    lower = float(np.clip(lower, 0.0, 1.0))
    upper = float(np.clip(upper, 0.0, 1.0))
    return (min(lower, upper), max(lower, upper))


def coerce_bool_option(value, default=None):
    """Interpret assorted truthy/falsey representations from UI settings."""
    if value is None:
        return default
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y", "on", "si", "sí"}:
            return True
        if normalized in {"false", "0", "no", "off"}:
            return False
        return default
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    try:
        return bool(int(value))
    except (TypeError, ValueError):
        return default


def normalize_schoenfeld_label(label):
    """Produce a normalized string representation for Schoenfeld-related labels/index keys."""
    if label is None:
        return ""
    if isinstance(label, tuple):
        parts = [normalize_schoenfeld_label(part) for part in label if part is not None]
        return " | ".join(part for part in parts if part)
    label_str = str(label)
    label_str = re.sub(r"\s+", " ", label_str)
    return label_str.strip()


def normalize_schoenfeld_dataframe(df):
    """Return a copy of the Schoenfeld results dataframe with standardized column names."""
    if df is None or not isinstance(df, pd.DataFrame):
        return df
    normalized_df = df.copy()
    rename_map = {}
    for col in normalized_df.columns:
        col_norm = normalize_schoenfeld_label(col).lower()
        if col_norm in {"p", "p value", "p-value", "p_value", "pvalues", "p-values"}:
            rename_map[col] = 'p'
        elif col_norm in {"test statistic", "test_statistic", "test-statistic", "chi2", "chisq", "chi squared", "chi-squared"}:
            rename_map[col] = 'test_statistic'
        elif col_norm in {"-log2(p)", "-log2 p", "-log2p"}:
            rename_map[col] = '-log2(p)'
    if rename_map:
        normalized_df = normalized_df.rename(columns=rename_map)
    return normalized_df


def extract_schoenfeld_p_value(df, term_label):
    """Retrieve the p-value associated with a specific term from a Schoenfeld results dataframe."""
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return None
    df_norm = normalize_schoenfeld_dataframe(df)
    if 'p' not in df_norm.columns:
        return None

    candidates = []
    if term_label in df_norm.index:
        candidates.append(term_label)

    term_norm = normalize_schoenfeld_label(term_label)
    if not candidates:
        for idx in df_norm.index:
            if normalize_schoenfeld_label(idx) == term_norm:
                candidates.append(idx)
                break

    for idx in candidates:
        try:
            value = df_norm.loc[idx, 'p']
        except KeyError:
            continue
        if isinstance(value, pd.Series):
            value = value.iloc[0]
        elif isinstance(value, pd.DataFrame):
            value = value.iloc[0, 0]
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None

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
                          schoenfeld_results_df=None, loglik_null=None, log_func=print,
                          c_index_test=None, test_proportion=None, c_index_gap=None,
                          c_index_train_ci=None, c_index_test_ci=None, c_index_cv_ci=None,
                          c_index_uno=None, c_index_antolini=None, tau=None,
                          ibs=None, brier_q25=None, brier_q50=None, brier_q75=None,
                          auroc_q25=None, auroc_q50=None, auroc_q75=None,
                          c_harrell_q25=None, c_harrell_q50=None, c_harrell_q75=None,
                          time_q25=None, time_q50=None, time_q75=None):
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

    schoenfeld_results_df = normalize_schoenfeld_dataframe(schoenfeld_results_df) if isinstance(schoenfeld_results_df, pd.DataFrame) else schoenfeld_results_df
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
        wald_global_p = None
        try:
            params_series = getattr(model, 'params_', None)
            variance_matrix_obj = getattr(model, 'variance_matrix_', None)
            if params_series is not None and variance_matrix_obj is not None and not params_series.empty:
                params_vec = params_series.to_numpy(dtype=float)
                if isinstance(variance_matrix_obj, pd.DataFrame):
                    variance_matrix_obj = variance_matrix_obj.reindex(index=params_series.index, columns=params_series.index, fill_value=0.0)
                    cov_matrix = variance_matrix_obj.to_numpy(dtype=float, copy=False)
                else:
                    cov_matrix = np.asarray(variance_matrix_obj, dtype=float)

                # Usar pseudo-inversa por estabilidad numérica (cov puede ser singular con penalización)
                cov_pinv = np.linalg.pinv(cov_matrix)
                wald_stat = float(params_vec.T @ cov_pinv @ params_vec)
                df_wald = params_vec.size
                if df_wald > 0 and np.isfinite(wald_stat) and wald_stat >= 0:
                    wald_global_p = scipy.stats.chi2.sf(wald_stat, df_wald)
        except Exception as e_wald_global:
            if log_func: log_func(f"DEBUG: Error calculating global Wald via covariance: {e_wald_global}", "WARN")
            wald_global_p = None

        metrics["Wald p-value (global approx)"] = wald_global_p

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
    metrics["C-Index (Training) CI"] = c_index_train_ci
    metrics["C-Index (Test)"] = c_index_test
    metrics["C-Index (Test) CI"] = c_index_test_ci
    metrics["C-Index Uno (IPCW)"] = c_index_uno
    metrics["C-Index Antolini (Ctd)"] = c_index_antolini
    metrics["τ (tau)"] = tau
    metrics["IBS"] = ibs
    metrics["Brier@Q25"] = brier_q25; metrics["Brier@Q50"] = brier_q50; metrics["Brier@Q75"] = brier_q75
    metrics["AUC@Q25"] = auroc_q25; metrics["AUC@Q50"] = auroc_q50; metrics["AUC@Q75"] = auroc_q75
    metrics["C@Q25"] = c_harrell_q25; metrics["C@Q50"] = c_harrell_q50; metrics["C@Q75"] = c_harrell_q75
    metrics["t(Q25)"] = time_q25; metrics["t(Q50)"] = time_q50; metrics["t(Q75)"] = time_q75
    metrics["Test Proportion"] = test_proportion
    metrics["C-Index Gap (Test-Train)"] = c_index_gap
    metrics["C-Index (CV Mean)"] = c_index_cv_mean # Scalar or None
    metrics["C-Index (CV Mean) CI"] = c_index_cv_ci
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

            # Comparación categórica
            ttk.Label(row_labelframe, text="Comparación:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
            compare_display_values = list(self.app_instance.categorical_compare_display_map.values())
            cat_compare_mode_combo = ttk.Combobox(row_labelframe, values=compare_display_values, state="disabled", width=30)
            cat_compare_mode_combo.set(self.app_instance._get_default_categorical_compare_display())
            cat_compare_mode_combo.grid(row=2, column=1, columnspan=2, sticky=tk.EW, padx=5)
            cat_compare_mode_combo.bind("<<ComboboxSelected>>", lambda _event, c=cov_name: self._toggle_row_controls_state(c))
            self.row_configs[cov_name]['cat_compare_mode_combo'] = cat_compare_mode_combo

            ttk.Label(row_labelframe, text="  Grupos a comparar (coma):").grid(row=3, column=0, sticky=tk.W, padx=15, pady=2)
            cat_compare_groups_var = tk.StringVar(value="")
            self.row_configs[cov_name]['cat_compare_groups_var'] = cat_compare_groups_var
            cat_compare_groups_entry = ttk.Entry(row_labelframe, textvariable=cat_compare_groups_var, width=28, state="disabled")
            cat_compare_groups_entry.grid(row=3, column=1, columnspan=2, sticky=tk.EW, padx=5)
            self.row_configs[cov_name]['cat_compare_groups_entry'] = cat_compare_groups_entry

            # Usar Spline
            ttk.Label(row_labelframe, text="Spline:").grid(row=4, column=0, sticky=tk.W, padx=5, pady=2)
            spline_var = tk.BooleanVar(value=False)
            self.row_configs[cov_name]['spline_var'] = spline_var
            cb_spline = ttk.Checkbutton(row_labelframe, text="Usar", variable=spline_var,
                                        command=lambda c=cov_name: self._toggle_row_controls_state(c))
            cb_spline.grid(row=4, column=1, sticky=tk.W, padx=5)
            self.row_configs[cov_name]['cb_spline'] = cb_spline

            # Spline Tipo
            ttk.Label(row_labelframe, text="  Tipo Spline:").grid(row=5, column=0, sticky=tk.W, padx=15, pady=2)
            display_values = list(self.app_instance.spline_type_display_map.values())
            spline_type_combo = ttk.Combobox(row_labelframe, values=display_values, state="disabled", width=24)
            spline_type_combo.set(self.app_instance._get_default_spline_display())
            spline_type_combo.grid(row=5, column=1, columnspan=2, sticky=tk.EW, padx=5)
            spline_type_combo.bind("<<ComboboxSelected>>", lambda _event, c=cov_name: self._toggle_row_controls_state(c))
            self.row_configs[cov_name]['spline_type_combo'] = spline_type_combo

            ttk.Label(row_labelframe, text="  Nodos Internos (0 = polinómico):").grid(row=6, column=0, sticky=tk.W, padx=15, pady=2)
            spline_knots_var = tk.IntVar(value=0)
            self.row_configs[cov_name]['spline_knots_var'] = spline_knots_var
            spline_knots_spinbox = ttk.Spinbox(row_labelframe, from_=0, to=15, textvariable=spline_knots_var, width=5, state="disabled")
            spline_knots_spinbox.grid(row=6, column=1, sticky=tk.W, padx=5)
            self.row_configs[cov_name]['spline_knots_spinbox'] = spline_knots_spinbox

            # Spline Degree (Nuevo para B-Splines)
            ttk.Label(row_labelframe, text="  Grado (1=recta, 2=cuadrática, 3=cúbica):").grid(row=7, column=0, sticky=tk.W, padx=15, pady=2)
            spline_degree_var = tk.IntVar(value=3) # Default cúbico
            self.row_configs[cov_name]['spline_degree_var'] = spline_degree_var
            # Grados comunes: 1 (lineal), 2 (cuadrático), 3 (cúbico)
            spline_degree_spinbox = ttk.Spinbox(row_labelframe, from_=1, to=5, textvariable=spline_degree_var, width=5, state="disabled")
            spline_degree_spinbox.grid(row=7, column=1, sticky=tk.W, padx=5)
            self.row_configs[cov_name]['spline_degree_spinbox'] = spline_degree_spinbox

            ttk.Label(row_labelframe, text="  Nodos Manuales Exactos (coma):").grid(row=8, column=0, sticky=tk.W, padx=15, pady=2)
            spline_custom_knots_var = tk.StringVar(value="")
            self.row_configs[cov_name]['spline_custom_knots_var'] = spline_custom_knots_var
            spline_custom_knots_entry = ttk.Entry(row_labelframe, textvariable=spline_custom_knots_var, width=24, state="disabled")
            spline_custom_knots_entry.grid(row=8, column=1, columnspan=2, sticky=tk.EW, padx=5)
            self.row_configs[cov_name]['spline_custom_knots_entry'] = spline_custom_knots_entry


            # --- Load existing or inferred configuration for the row ---
            # Type
            current_type = self.app_instance.covariables_type_config.get(cov_name)
            if not current_type and self.app_instance.data is not None and cov_name in self.app_instance.data:
                current_type = "Cuantitativa" if pd.api.types.is_numeric_dtype(self.app_instance.data[cov_name]) else "Cualitativa"
            else: # Fallback if data is somehow not available or column not present (should be caught by caller)
                current_type = "Cuantitativa"
            type_var.set(current_type)

            # Ref. Cat. y comparación categórica (if Cualitativa)
            if current_type == "Cualitativa":
                unique_vals = self.app_instance._get_reference_category_values(cov_name)
                ref_combo['values'] = unique_vals
                stored_ref_cat = self.app_instance.ref_categories_config.get(cov_name)
                if stored_ref_cat in unique_vals:
                    ref_combo.set(stored_ref_cat)
                elif unique_vals:
                    ref_combo.set(unique_vals[0]) # Default to first if not set or invalid

                effective_ref_cat = ref_combo.get()
                compare_cfg = self.app_instance._get_categorical_compare_config(cov_name, effective_ref_cat)
                cat_compare_mode_combo.set(
                    self.app_instance._get_categorical_compare_display_value(compare_cfg.get('mode', 'all'))
                )
                cat_compare_groups_var.set(
                    self.app_instance._format_categorical_compare_groups(compare_cfg.get('selected_groups', []))
                )

            # Spline (if Cuantitativa)
            if current_type == "Cuantitativa":
                if cov_name in self.app_instance.spline_config_details:
                    spline_var.set(True)
                    spl_conf = self.app_instance.spline_config_details[cov_name]
                    internal_type_existing = spl_conf.get('type', 'Natural')
                    spline_type_combo.set(self.app_instance._get_spline_display_value(internal_type_existing))
                    spline_knots_var.set(spl_conf.get('num_knots', 0))
                    spline_degree_var.set(spl_conf.get('degree', 3)) # Cargar grado, default 3
                    existing_manual_knots = spl_conf.get('custom_knots') or []
                    if existing_manual_knots:
                        spline_custom_knots_var.set(
                            ", ".join(f"{val:g}" for val in existing_manual_knots)
                        )
                else:
                    spline_var.set(False)
                    # Asegurar defaults también para grado si no hay config de spline
                    spline_knots_var.set(0)
                    spline_degree_var.set(3)
                    spline_type_combo.set(self.app_instance._get_default_spline_display())

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

        # Configurar ComboBox de Categoría de Referencia y comparación categórica
        if var_is_quantitative:
            config['ref_combo'].config(state="disabled", values=[])
            config['ref_combo'].set("")
            config['cat_compare_mode_combo'].set(self.app_instance._get_default_categorical_compare_display())
            config['cat_compare_mode_combo'].config(state="disabled")
            config['cat_compare_groups_entry'].config(state="disabled")
        else:
            unique_vals = self.app_instance._get_reference_category_values(cov_name)
            config['ref_combo']['values'] = unique_vals
            stored_ref_cat = self.app_instance.ref_categories_config.get(cov_name)
            if stored_ref_cat in unique_vals:
                config['ref_combo'].set(stored_ref_cat)
            elif unique_vals and config['ref_combo'].get() not in unique_vals:
                config['ref_combo'].set(unique_vals[0])
            elif not unique_vals:
                config['ref_combo'].set("")
            config['ref_combo'].config(state="readonly" if unique_vals else "disabled")

            if config['cat_compare_mode_combo'].get() not in self.app_instance.categorical_compare_reverse_map:
                compare_cfg = self.app_instance._get_categorical_compare_config(cov_name, config['ref_combo'].get())
                config['cat_compare_mode_combo'].set(
                    self.app_instance._get_categorical_compare_display_value(compare_cfg.get('mode', 'all'))
                )
                if compare_cfg.get('selected_groups') and not config['cat_compare_groups_var'].get().strip():
                    config['cat_compare_groups_var'].set(
                        self.app_instance._format_categorical_compare_groups(compare_cfg.get('selected_groups', []))
                    )

            config['cat_compare_mode_combo'].config(state="readonly" if unique_vals else "disabled")
            selected_compare_mode = self.app_instance._get_categorical_compare_internal_mode(config['cat_compare_mode_combo'].get())
            groups_entry_state = tk.NORMAL if (selected_compare_mode == 'selected' and unique_vals) else tk.DISABLED
            config['cat_compare_groups_entry'].config(state=groups_entry_state)

        # Configurar CheckBox "Usar Spline"
        config['cb_spline'].config(state=tk.NORMAL if var_is_quantitative else tk.DISABLED)
        if not var_is_quantitative:
            config['spline_var'].set(False) # Forzar desmarcado si no es cuantitativa

        # Obtener el estado actual del checkbox "Usar Spline" DESPUÉS de cualquier posible cambio
        spline_is_active = config['spline_var'].get()
        # self.app_instance.log(f"DEBUG: _toggle_row_controls_state para '{cov_name}': spline_is_active={spline_is_active}", "DEBUG")

        # Configurar "Tipo Spline"
        # Habilitado si la variable es cuantitativa Y el checkbox "Usar Spline" está marcado.
        can_configure_spline_options = var_is_quantitative and spline_is_active
        spline_type_control_state = tk.NORMAL if can_configure_spline_options else tk.DISABLED

        config['spline_type_combo'].config(state=spline_type_control_state)
        selected_spline_display = config['spline_type_combo'].get()
        selected_spline_type = self.app_instance._get_spline_internal_type(selected_spline_display)

        if spline_type_control_state == tk.NORMAL and selected_spline_type in ("B-spline", "Natural"):
            knots_state = tk.NORMAL
        else:
            knots_state = tk.DISABLED
        config['spline_knots_spinbox'].config(state=knots_state)
        if knots_state == tk.DISABLED:
            config['spline_knots_var'].set(0)
        # self.app_instance.log(f"DEBUG: _toggle_row_controls_state para '{cov_name}': spline_type_df_control_state='{spline_type_df_control_state}'", "DEBUG")

        # Configurar "Spline Grado"
        # Habilitado si los detalles del spline están habilitados y el tipo es "B-spline".
        can_configure_degree = can_configure_spline_options and (selected_spline_type == "B-spline")
        spline_degree_control_state = tk.NORMAL if can_configure_degree else tk.DISABLED
        config['spline_degree_spinbox'].config(state=spline_degree_control_state)

        # Los nodos manuales deben poder fijarse tanto para B-spline como para Natural.
        custom_knots_state = tk.NORMAL if (can_configure_spline_options and selected_spline_type in ("B-spline", "Natural")) else tk.DISABLED
        config['spline_custom_knots_entry'].config(state=custom_knots_state)
        # self.app_instance.log(f"DEBUG: _toggle_row_controls_state para '{cov_name}': selected_spline_type='{selected_spline_type}', spline_degree_control_state='{spline_degree_control_state}'", "DEBUG")

        # Resetear valores si los controles correspondientes están deshabilitados
        if not can_configure_spline_options:
            config['spline_type_combo'].set(self.app_instance._get_default_spline_display())
            config['spline_knots_var'].set(0)
            config['spline_degree_var'].set(3) # Grado también se resetea
            config['spline_custom_knots_var'].set("")

        if not can_configure_degree:
            # Si el grado no es configurable (pero tipo/df sí podrían serlo, ej. para Natural spline),
            # reseteamos la variable de grado a 3.
            # Esto es importante si se cambia de B-spline a Natural.
            config['spline_degree_var'].set(3)

        if custom_knots_state == tk.DISABLED:
            config['spline_custom_knots_var'].set("")

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

                compare_mode_display = config_widgets['cat_compare_mode_combo'].get()
                compare_mode_internal = self.app_instance._get_categorical_compare_internal_mode(compare_mode_display)
                compare_groups_raw = config_widgets['cat_compare_groups_var'].get()
                compare_cfg = self.app_instance._build_categorical_compare_config(
                    cov_name,
                    ref_category=selected_ref_cat,
                    mode=compare_mode_internal,
                    selected_groups=compare_groups_raw
                )
                self.app_instance.categorical_compare_config[cov_name] = compare_cfg
                config_widgets['cat_compare_groups_var'].set(
                    self.app_instance._format_categorical_compare_groups(compare_cfg.get('selected_groups', []))
                )
                if compare_cfg.get('mode') == 'selected' and not compare_cfg.get('selected_groups'):
                    self.app_instance.log(
                        f"'{cov_name}': modo 'Comparar solo contra grupos elegidos' sin grupos válidos. Se usará el comportamiento estándar hasta que elijas grupos.",
                        "WARN"
                    )

                if cov_name in self.app_instance.spline_config_details:
                    del self.app_instance.spline_config_details[cov_name]
                    self.app_instance.log(f"Config. spline eliminada para '{cov_name}' (cambiado a Cualitativa).", "DEBUG")

            elif new_type == "Cuantitativa":
                use_spline = config_widgets['spline_var'].get()
                if use_spline:
                    spline_type_display_row = config_widgets['spline_type_combo'].get()
                    spline_type_internal_row = self.app_instance._get_spline_internal_type(spline_type_display_row)
                    spline_num_knots = self.app_instance._coerce_int_value(
                        config_widgets['spline_knots_spinbox'].get(),
                        fallback=0,
                        field_name=f"Nodos internos para '{cov_name}'",
                        min_value=0
                    )
                    config_widgets['spline_knots_var'].set(spline_num_knots)

                    custom_knots_raw = config_widgets['spline_custom_knots_var'].get().strip()
                    custom_knots_list = []
                    if custom_knots_raw:
                        custom_knots_candidates = re.split(r"[,;]", custom_knots_raw)
                        for candidate in custom_knots_candidates:
                            candidate_clean = candidate.strip()
                            if not candidate_clean:
                                continue
                            try:
                                custom_knots_list.append(float(candidate_clean))
                            except ValueError:
                                self.app_instance.log(f"NODO manual inválido '{candidate_clean}' para '{cov_name}'. Entrada ignorada.", "WARN")
                        if custom_knots_list:
                            custom_knots_list = sorted(set(custom_knots_list))
                            config_widgets['spline_custom_knots_var'].set(
                                ", ".join(f"{val:g}" for val in custom_knots_list)
                            )
                            spline_num_knots = len(custom_knots_list)
                            config_widgets['spline_knots_var'].set(spline_num_knots)
                        else:
                            config_widgets['spline_custom_knots_var'].set("")

                    spline_degree = self.app_instance._coerce_int_value(
                        config_widgets['spline_degree_spinbox'].get(),
                        fallback=3,
                        field_name=f"Grado de spline para '{cov_name}'",
                        min_value=1
                    )
                    config_widgets['spline_degree_var'].set(spline_degree)

                    if spline_type_internal_row == "Natural":
                        spline_degree = 3
                        config_widgets['spline_degree_var'].set(3)

                    spline_df = self.app_instance._derive_spline_df(
                        spline_type_internal_row,
                        spline_degree,
                        spline_num_knots,
                        custom_knots_list
                    )

                    spline_config_data = {
                        'type': spline_type_internal_row,
                        'df': spline_df,
                        'num_knots': spline_num_knots,
                        'restricted': spline_type_internal_row == "Natural",
                        'degree': spline_degree if spline_type_internal_row == "B-spline" else 3,
                        'custom_knots': custom_knots_list
                    }

                    self.app_instance.spline_config_details[cov_name] = spline_config_data

                    log_message = f"Config. spline aplicada para '{cov_name}': Tipo={spline_type_display_row}, DF(auto)={spline_df}, Nodos={spline_num_knots}"
                    if spline_type_internal_row == "B-spline":
                        log_message += f", Grado={spline_degree}"
                        if spline_num_knots == 0 and not custom_knots_list:
                            log_message += " (0 nodos => forma polinómica)"
                    else:
                        log_message += ", Grado=3 (natural)"
                    if custom_knots_list:
                        log_message += f", NodosManualExactos={custom_knots_list}"
                    log_message += "."
                    self.app_instance.log(log_message, "DEBUG")
                else:
                    if cov_name in self.app_instance.spline_config_details:
                        del self.app_instance.spline_config_details[cov_name]
                        self.app_instance.log(f"Config. spline eliminada para '{cov_name}' (desmarcado).", "DEBUG")

                if cov_name in self.app_instance.ref_categories_config:
                    del self.app_instance.ref_categories_config[cov_name]
                    self.app_instance.log(f"Config. ref.cat. eliminada para '{cov_name}' (cambiado a Cuantitativa).", "DEBUG")
                if cov_name in self.app_instance.categorical_compare_config:
                    del self.app_instance.categorical_compare_config[cov_name]
                    self.app_instance.log(f"Config. comparación categórica eliminada para '{cov_name}' (cambiado a Cuantitativa).", "DEBUG")

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


class EditPredictionLegendsDialog(tk.Toplevel):
    def __init__(self, parent, curves_to_plot):
        super().__init__(parent)
        self.transient(parent)
        self.grab_set()
        self.title("Editar Leyendas de Curvas de Predicción")

        self.result = None
        self.entries = {}

        main_frame = ttk.Frame(self, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(main_frame, text="Edite las leyendas para cada curva:", font=("TkDefaultFont", 10, "bold")).pack(pady=(0,10), anchor='w')

        scrolled_frame = ScrolledFrame(main_frame)
        scrolled_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        self.controls_frame = scrolled_frame.interior

        for i, (original_label, curve_df) in enumerate(curves_to_plot.items()):
            row_frame = ttk.Frame(self.controls_frame)
            row_frame.pack(fill=tk.X, pady=2, padx=5)

            ttk.Label(row_frame, text=f"Curva {i+1}:", width=10).pack(side=tk.LEFT, padx=(0,5))

            entry_var = tk.StringVar(value=original_label)
            entry = ttk.Entry(row_frame, textvariable=entry_var, width=80)
            entry.pack(side=tk.LEFT, fill=tk.X, expand=True)

            self.entries[original_label] = entry_var

        buttons_frame = ttk.Frame(main_frame)
        buttons_frame.pack(fill=tk.X, pady=(10,0))
        ttk.Button(buttons_frame, text="OK/Generar Gráfico", command=self.on_ok).pack(side=tk.RIGHT, padx=5)
        ttk.Button(buttons_frame, text="Cancelar", command=self.destroy).pack(side=tk.RIGHT)

        self.protocol("WM_DELETE_WINDOW", self.destroy)
        self.wait_window(self)

    def on_ok(self):
        self.result = {original_label: var.get() for original_label, var in self.entries.items()}
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
    def __init__(self, parent_notebook_tab, main_app_instance=None, **kwargs):
        super().__init__(parent_notebook_tab, **kwargs)
        self.pack(fill=tk.BOTH, expand=True)
        self.main_app = main_app_instance
        self.using_shared_dataset = True
        self.shared_dataset_metadata = {}
        self.current_shared_filter_summary = []
        self.parent_for_dialogs = self.winfo_toplevel()

        # Variables para datos y configuración
        self.raw_data = None
        self.data = None
        self.time_col_original_name = ""
        self.event_col_original_name = ""
        self.selected_covariables_from_ui = []
        self.covariables_type_config = {}  # {var_name: "Cuantitativa" | "Cualitativa"}
        self.ref_categories_config = {}  # {cual_var_name: "ref_category_value"}
        self.categorical_compare_config = {}  # {cual_var_name: {'mode': 'all'|'selected'|'one_vs_rest', 'selected_groups': []}}
        # {cuant_var_name: {'type': 'Natural'|'B-spline', 'df': int (auto), 'num_knots': int, 'degree': int, 'restricted': bool}}
        self.spline_config_details = {}
        # Cache with the most recent spline metadata generated while building the design matrix
        self._last_spline_basis_metadata = {}
        self.spline_type_display_map = {
            "Natural": "Restringido (Natural)",
            "B-spline": "No restringido (B-spline)"
        }
        self.spline_type_reverse_map = {display: internal for internal, display in self.spline_type_display_map.items()}
        self.categorical_compare_display_map = {
            "all": "Comparar contra todos los grupos",
            "selected": "Comparar solo contra grupos elegidos",
            "one_vs_rest": "Dicotómica: elegida vs resto"
        }
        self.categorical_compare_reverse_map = {display: internal for internal, display in self.categorical_compare_display_map.items()}
        self.current_plot_options = {}  # Diccionario para guardar opciones de gráficos
        self.nomogram_default_time_points = [5.0, 10.0, 15.0]  # Tiempo base para ejes de supervivencia en nomogramas

        # Variables para modelos
        # Lista de diccionarios, cada uno con datos de un modelo
        self.generated_models_data = []
        # Diccionario del modelo seleccionado en la Treeview
        self.selected_model_in_treeview = None
        self.selected_models_in_treeview = []
        self.btn_oos_calibration = None
        self.btn_collinearity_diag = None # <-- NUEVO
        self.btn_nomogram = None
        self.btn_delete_model = None
        self.entry_custom_model_name = None # Placeholder for custom name Entry
        self.text_custom_model_notes = None # Placeholder for custom notes Text

    # Grid de escenarios para experimentación con configuraciones
        # Grid de escenarios para experimentación con configuraciones
        self.model_grid_entries = []
        self.model_grid_counter = 0
        self.model_grid_tree = None
        self.frame_model_grid = None

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
        self.calculate_test_cindex_var = BooleanVar(value=False)  # Calcular C-Index con holdout train/test
        self.test_size_var = DoubleVar(value=0.25)  # Proporción para holdout test
        self.test_random_seed_var = IntVar(value=42)  # Semilla aleatoria para holdout
        self.stratify_holdout_var = BooleanVar(value=True)  # Estratificar por evento en holdout
        self.covariate_scaling_method_var = StringVar(value="Ninguna")
        self._manual_holdout_config_dirty = False
        self._install_holdout_dirty_tracking()

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
        self._install_holdout_dirty_tracking()

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

    def _install_holdout_dirty_tracking(self):
        tracked_var_names = (
            'calculate_test_cindex_var',
            'test_size_var',
            'test_random_seed_var',
            'stratify_holdout_var',
            'tau_mode_var',
            'tau_manual_var',
        )
        for var_name in tracked_var_names:
            variable = getattr(self, var_name, None)
            if variable is None or not hasattr(variable, 'trace_add'):
                continue
            try:
                variable.trace_add("write", self._mark_holdout_config_dirty)
            except Exception:
                continue

    def _mark_holdout_config_dirty(self, *_args):
        self._manual_holdout_config_dirty = True

    def _clear_holdout_config_dirty(self):
        self._manual_holdout_config_dirty = False

    def _get_spline_display_value(self, internal_type):
        return self.spline_type_display_map.get(internal_type, internal_type)

    def _get_spline_internal_type(self, display_value):
        return self.spline_type_reverse_map.get(display_value, display_value)

    def _get_default_spline_display(self):
        return self._get_spline_display_value("Natural")

    def _format_c_index_display(self, value, ci=None, decimals=3):
        return format_c_index_display(value, ci=ci, decimals=decimals)

    def _get_categorical_compare_display_value(self, internal_mode):
        return self.categorical_compare_display_map.get(
            internal_mode,
            self.categorical_compare_display_map.get("all", "Comparar contra todos los grupos")
        )

    def _get_categorical_compare_internal_mode(self, display_value):
        return self.categorical_compare_reverse_map.get(
            display_value,
            display_value if display_value in self.categorical_compare_display_map else "all"
        )

    def _get_default_categorical_compare_display(self):
        return self.categorical_compare_display_map.get("all", "Comparar contra todos los grupos")

    def _parse_categorical_compare_groups(self, raw_value):
        if raw_value is None:
            return []
        if isinstance(raw_value, (list, tuple, set)):
            raw_items = list(raw_value)
        else:
            raw_items = re.split(r"[,;\n]+", str(raw_value))

        clean_items = []
        seen = set()
        for item in raw_items:
            item_str = str(item).strip()
            if not item_str or item_str in seen:
                continue
            clean_items.append(item_str)
            seen.add(item_str)
        return clean_items

    def _format_categorical_compare_groups(self, groups):
        return ", ".join(str(item).strip() for item in (groups or []) if str(item).strip())

    def _coerce_plot_value_for_column(self, column_name, raw_value, data=None):
        raw_text = "" if raw_value is None else str(raw_value).strip()
        if raw_text == "":
            return None

        data = data if isinstance(data, pd.DataFrame) else getattr(self, 'latest_fit_dataframe', None)
        if isinstance(data, pd.DataFrame) and column_name in data.columns and pd.api.types.is_numeric_dtype(data[column_name]):
            return float(raw_text)

        lower_text = raw_text.lower()
        if lower_text in {"nan", "none", "null"}:
            return np.nan

        return raw_text

    def _parse_partial_effect_values(self, column_name, raw_value, data=None):
        if raw_value is None:
            return []

        parts = [part.strip() for part in re.split(r'[,;]', str(raw_value)) if part.strip()]
        parsed_values = []
        for part in parts:
            try:
                coerced = self._coerce_plot_value_for_column(column_name, part, data=data)
            except (TypeError, ValueError):
                continue
            if coerced is not None:
                parsed_values.append(coerced)
        return parsed_values

    def _parse_partial_effect_baseline_overrides(self, raw_value, exclude_covariate=None, data=None, available_covariates=None):
        overrides = {}
        if raw_value is None:
            return overrides

        raw_text = str(raw_value).strip()
        if not raw_text:
            return overrides

        data = data if isinstance(data, pd.DataFrame) else getattr(self, 'latest_fit_dataframe', None)
        valid_covariates = available_covariates
        if valid_covariates is None:
            valid_covariates = list(getattr(self, 'latest_covariates', []) or [])
            if not valid_covariates and isinstance(data, pd.DataFrame):
                valid_covariates = list(data.columns)

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
            if not key or key == exclude_covariate or key not in valid_covariates:
                continue
            try:
                coerced = self._coerce_plot_value_for_column(key, value, data=data)
            except (TypeError, ValueError):
                continue
            if coerced is not None:
                overrides[key] = coerced
        return overrides

    def _build_plot_reference_row(self, focal_covariate=None, overrides=None, data=None, covariates=None):
        overrides = overrides or {}
        base_row = {}
        data = data if isinstance(data, pd.DataFrame) else getattr(self, 'latest_fit_dataframe', None)
        if data is None:
            data = pd.DataFrame()

        if covariates is None:
            covariates = list(getattr(self, 'latest_covariates', []) or [])
            if not covariates and isinstance(data, pd.DataFrame):
                covariates = list(data.columns)

        for col in covariates:
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

    def _build_categorical_compare_config(self, var_name, ref_category=None, mode="all", selected_groups=None):
        internal_mode = self._get_categorical_compare_internal_mode(mode)
        available_values = self._get_reference_category_values(var_name)
        available_lookup = {str(value): str(value) for value in available_values}
        ref_value = str(ref_category).strip() if ref_category not in (None, "") else ""

        parsed_groups = self._parse_categorical_compare_groups(selected_groups)
        clean_groups = []
        seen = set()
        for group in parsed_groups:
            if group == ref_value or group in seen:
                continue
            if available_lookup and group not in available_lookup:
                continue
            clean_groups.append(group)
            seen.add(group)

        return {
            "mode": internal_mode if internal_mode in self.categorical_compare_display_map else "all",
            "selected_groups": clean_groups
        }

    def _get_categorical_compare_config(self, var_name, ref_category=None):
        stored = self.categorical_compare_config.get(var_name, {})
        if not isinstance(stored, dict):
            stored = {}
        return self._build_categorical_compare_config(
            var_name,
            ref_category=ref_category,
            mode=stored.get("mode", "all"),
            selected_groups=stored.get("selected_groups", [])
        )

    def _coerce_int_value(self, raw_value, fallback, field_name, min_value=None):
        """Sanitize integer inputs coming from Tk widgets or raw strings."""
        extracted = raw_value
        if hasattr(raw_value, "get"):
            try:
                extracted = raw_value.get()
            except (tk.TclError, ValueError, TypeError):
                extracted = ""

        display_val = extracted if extracted not in ("", None) else "(vacío)"

        try:
            coerced = int(extracted)
        except (TypeError, ValueError):
            try:
                coerced = int(float(extracted))
            except (TypeError, ValueError):
                adjusted_fallback = fallback
                if min_value is not None and adjusted_fallback < min_value:
                    adjusted_fallback = min_value
                self.log(f"{field_name}: entrada inválida '{display_val}'. Se usa {adjusted_fallback}.", "WARN")
                return adjusted_fallback

        if min_value is not None and coerced < min_value:
            self.log(f"{field_name}: valor {coerced} menor que {min_value}. Ajustado a {min_value}.", "WARN")
            coerced = min_value

        return coerced

    def _coerce_float_value(self, raw_value, fallback, field_name, min_value=None):
        """Sanitize float inputs coming from Tk widgets or raw strings."""
        extracted = raw_value
        if hasattr(raw_value, "get"):
            try:
                extracted = raw_value.get()
            except (tk.TclError, ValueError, TypeError):
                extracted = ""

        display_val = extracted if extracted not in ("", None) else "(vacío)"

        try:
            coerced = float(extracted)
        except (TypeError, ValueError):
            adjusted_fallback = fallback
            if min_value is not None and adjusted_fallback < min_value:
                adjusted_fallback = min_value
            self.log(f"{field_name}: entrada inválida '{display_val}'. Se usa {adjusted_fallback}.", "WARN")
            return adjusted_fallback

        if min_value is not None and coerced < min_value:
            self.log(f"{field_name}: valor {coerced} menor que {min_value}. Ajustado a {min_value}.", "WARN")
            coerced = min_value

        return coerced

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

    def _build_evaluation_time_grid_cox(self, train_times, test_times, tau=None):
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

    def _resolve_holdout_split_settings(self, data, event_col, requested_test_size, min_train_rows=5, min_test_rows=2, prefer_stratify=True, context_label="holdout"):
        if data is None or len(data) == 0:
            raise ValueError("No hay datos suficientes para crear la partición train/test.")

        total_rows = int(len(data))
        requested = self._coerce_float_value(requested_test_size, 0.25, f"Proporción test ({context_label})", min_value=0.05)
        requested = min(max(requested, 0.05), 0.95)

        min_train_rows = max(int(min_train_rows or 0), 2)
        min_test_rows = max(int(min_test_rows or 0), 1)
        warnings = []
        stratify_values = None

        if prefer_stratify and event_col in data.columns:
            event_series = pd.to_numeric(data[event_col], errors='coerce').fillna(0).astype(int)
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

    def _derive_spline_df(self, spline_type, spline_degree, spline_num_knots, custom_knots=None):
        """Infer a reasonable df for spline settings when the user only specifies grado/nodos."""
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

        # Natural (restricted) splines default to cúbicos
        if effective_knots > 0:
            derived_df = effective_knots + 1
        else:
            derived_df = 4
        return max(derived_df, 1)

    def _build_degree_only_polynomial_formula(self, var_name, degree):
        """Create a pure polynomial formula for the chosen degree, without internal spline knots."""
        safe_term = f"Q('{var_name}')"
        effective_degree = max(1, int(degree or 1))
        formula_terms = [safe_term]
        for power in range(2, effective_degree + 1):
            formula_terms.append(f"I({safe_term} ** {power})")
        return " + ".join(formula_terms)

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
        self.combo_ref_categoria_seleccionada.bind("<<ComboboxSelected>>", self._on_categorical_panel_change)

        subframe_comparacion_categoria = ttk.Frame(frame_config_covariable_seleccionada)
        subframe_comparacion_categoria.pack(fill=tk.X, pady=5, padx=10)
        ttk.Label(subframe_comparacion_categoria, text="Modo de comparación categórica:").grid(row=0, column=0, padx=5, pady=2, sticky="w")
        compare_mode_values_simple = list(self.categorical_compare_display_map.values())
        self.combo_modo_comparacion_categoria = ttk.Combobox(subframe_comparacion_categoria, values=compare_mode_values_simple, state="disabled", width=34)
        self.combo_modo_comparacion_categoria.set(self._get_default_categorical_compare_display())
        self.combo_modo_comparacion_categoria.grid(row=0, column=1, padx=5, pady=2, sticky="ew")
        self.combo_modo_comparacion_categoria.bind("<<ComboboxSelected>>", self._on_categorical_panel_change)

        ttk.Label(subframe_comparacion_categoria, text="Grupos a comparar (coma):").grid(row=1, column=0, padx=5, pady=2, sticky="w")
        self.var_grupos_comparacion_categoria = StringVar(value="")
        self.entry_grupos_comparacion_categoria = ttk.Entry(subframe_comparacion_categoria, textvariable=self.var_grupos_comparacion_categoria, state="disabled")
        self.entry_grupos_comparacion_categoria.grid(row=1, column=1, padx=5, pady=2, sticky="ew")
        self.entry_grupos_comparacion_categoria.bind("<FocusOut>", self._on_categorical_panel_change)
        self.entry_grupos_comparacion_categoria.bind("<Return>", self._on_categorical_panel_change)
        subframe_comparacion_categoria.columnconfigure(1, weight=1)

        subframe_spline_check = ttk.Frame(frame_config_covariable_seleccionada)
        subframe_spline_check.pack(fill=tk.X, pady=5, padx=10)
        self.var_usar_spline_seleccionada = BooleanVar(value=False)
        self.checkbutton_usar_spline = ttk.Checkbutton(subframe_spline_check, text="Usar Spline (si Cuantitativa(s))", variable=self.var_usar_spline_seleccionada, command=self._toggle_spline_and_refcat_controls, state=tk.DISABLED)
        self.checkbutton_usar_spline.pack(side=tk.LEFT, anchor='w')

        subframe_spline_detalles = ttk.Frame(frame_config_covariable_seleccionada)
        subframe_spline_detalles.pack(fill=tk.X, pady=5, padx=10)
        ttk.Label(subframe_spline_detalles, text="  Tipo de Spline:").pack(side=tk.LEFT, padx=(15, 5))
        spline_type_values_simple = list(self.spline_type_display_map.values())
        self.combo_tipo_spline_seleccionada = ttk.Combobox(subframe_spline_detalles, values=spline_type_values_simple, state="disabled", width=24)
        self.combo_tipo_spline_seleccionada.set(self._get_default_spline_display())
        self.combo_tipo_spline_seleccionada.pack(side=tk.LEFT, padx=5)

        ttk.Label(subframe_spline_detalles, text="  Nodos internos (0 = polinómico):").pack(side=tk.LEFT, padx=(15, 5))
        self.var_knots_spline_seleccionada = IntVar(value=0)
        self.spinbox_knots_spline = ttk.Spinbox(subframe_spline_detalles, from_=0, to=15, textvariable=self.var_knots_spline_seleccionada, width=5, state="disabled")
        self.spinbox_knots_spline.pack(side=tk.LEFT, padx=5)

        # NUEVO: Spinbox para grado del spline en el panel simple
        ttk.Label(subframe_spline_detalles, text="  Grado (1=recta, 2=cuadrática, 3=cúbica):").pack(side=tk.LEFT, padx=(15, 5))
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
                            if cov_name not in self.categorical_compare_config:
                                self.categorical_compare_config[cov_name] = {"mode": "all", "selected_groups": []}
                        elif inferred_type == "Cuantitativa":
                            # Asegurar que no haya config de ref.cat. para cuantitativas
                            if cov_name in self.ref_categories_config:
                                del self.ref_categories_config[cov_name]
                            if cov_name in self.categorical_compare_config:
                                del self.categorical_compare_config[cov_name]

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
            
            self.using_shared_dataset = False
            self.shared_dataset_metadata = {}
            self.current_shared_filter_summary = []

            self._prepare_new_dataset()
            self.actualizar_controles_preproc()
            messagebox.showinfo("Carga Exitosa", f"Archivo '{os.path.basename(file_path)}' cargado.", parent=self.parent_for_dialogs)

        except Exception as e:
            messagebox.showerror("Error de Carga", f"No se pudo cargar el archivo:\n{e}", parent=self.parent_for_dialogs)
            self.log(f"Error al cargar archivo '{file_path}': {e}", "ERROR")
            self.raw_data = None
            self.data = None
            self.label_archivo_cargado_info.config(text="Error al cargar archivo.")
            traceback.print_exc(limit=3)

    def _prepare_new_dataset(self):
        self.covariables_type_config = {}
        self.ref_categories_config = {}
        self.categorical_compare_config = {}
        self.spline_config_details = {}
        self.generated_models_data = []
        self.selected_model_in_treeview = None
        if hasattr(self, 'treeview_lista_modelos'):
            try:
                self._update_models_treeview()
            except Exception:
                pass

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

        previous_source_path = None
        if hasattr(self, "shared_dataset_metadata") and isinstance(self.shared_dataset_metadata, dict):
            previous_source_path = self.shared_dataset_metadata.get("source_path")
        had_data_before = isinstance(getattr(self, "data", None), pd.DataFrame)

        new_metadata = metadata or {}
        new_source_path = new_metadata.get("source_path")

        self.shared_dataset_metadata = new_metadata
        self.current_shared_filter_summary = list(filter_summary or [])

        if dataset is None:
            self.raw_data = None
            self.data = None
            if getattr(self, 'custom_filter_component_instance', None):
                try:
                    self.custom_filter_component_instance.set_dataframe(pd.DataFrame())
                except Exception:
                    pass
            if hasattr(self, 'label_archivo_cargado_info'):
                self.label_archivo_cargado_info.config(text="Sin archivo compartido.")
            self._prepare_new_dataset()
            try:
                self.actualizar_controles_preproc()
            except Exception:
                pass
            return

        try:
            base_df = dataset.copy(deep=True)
        except Exception:
            base_df = dataset

        try:
            if filtered_dataset is not None and isinstance(filtered_dataset, pd.DataFrame):
                active_df = filtered_dataset.copy(deep=True)
            else:
                active_df = base_df.copy(deep=True)
        except Exception:
            active_df = filtered_dataset if isinstance(filtered_dataset, pd.DataFrame) else base_df

        self.raw_data = base_df
        self.data = active_df

        if getattr(self, 'custom_filter_component_instance', None):
            try:
                self.custom_filter_component_instance.set_dataframe(self.data)
            except Exception:
                pass

        info_suffix = f" | Filtros: {len(self.current_shared_filter_summary)}" if self.current_shared_filter_summary else ""
        source_name = self._describe_shared_source(self.shared_dataset_metadata)
        if hasattr(self, 'label_archivo_cargado_info'):
            try:
                rows, cols = self.data.shape
                self.label_archivo_cargado_info.config(text=f"Compartido: {source_name} ({rows} filas, {cols} cols){info_suffix}")
            except Exception:
                self.label_archivo_cargado_info.config(text=f"Compartido: {source_name}{info_suffix}")

        should_reset_models = False
        if not had_data_before:
            should_reset_models = True
        elif previous_source_path and new_source_path and previous_source_path != new_source_path:
            should_reset_models = True
        elif not previous_source_path and new_source_path:
            should_reset_models = True
        elif self.raw_data is None:
            should_reset_models = True

        if should_reset_models:
            self._prepare_new_dataset()
        else:
            self.log("Se actualizó el dataset compartido; los modelos existentes se conservan.", "INFO")
        try:
            self.actualizar_controles_preproc()
        except Exception:
            pass

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

    def _get_reference_category_values(self, var_name):
        """Obtiene categorías de referencia únicas, limpias y en orden de aparición."""
        if self.data is None or not var_name or var_name not in self.data.columns:
            return []

        try:
            series = self.data[var_name].dropna()
            if series.empty:
                return []
            return [str(value) for value in pd.unique(series.astype(str))]
        except Exception as exc:
            self.log(f"No se pudieron leer categorías para '{var_name}': {exc}", "WARN")
            return []

    def _persist_single_selected_categorical_config_from_panel(self, quiet=False):
        """Guarda al vuelo la configuración categórica del panel simple para la variable seleccionada."""
        if not all(hasattr(self, attr) for attr in [
            'listbox_covariables_disponibles', 'var_tipo_covariable_seleccionada',
            'combo_ref_categoria_seleccionada', 'combo_modo_comparacion_categoria',
            'var_grupos_comparacion_categoria'
        ]):
            return False

        sel_idx = self.listbox_covariables_disponibles.curselection()
        if len(sel_idx) != 1:
            return False
        if self.var_tipo_covariable_seleccionada.get() != "Cualitativa":
            return False

        var_name = self.listbox_covariables_disponibles.get(sel_idx[0])
        available_cats = self._get_reference_category_values(var_name)
        if not available_cats:
            return False

        ref_cat = (self.combo_ref_categoria_seleccionada.get() or "").strip()
        if ref_cat not in available_cats:
            ref_cat = available_cats[0]
            self.combo_ref_categoria_seleccionada.set(ref_cat)

        mode_internal = self._get_categorical_compare_internal_mode(
            self.combo_modo_comparacion_categoria.get()
        )
        compare_cfg = self._build_categorical_compare_config(
            var_name,
            ref_category=ref_cat,
            mode=mode_internal,
            selected_groups=self.var_grupos_comparacion_categoria.get()
        )

        self.covariables_type_config[var_name] = "Cualitativa"
        self.ref_categories_config[var_name] = ref_cat
        self.categorical_compare_config[var_name] = compare_cfg

        formatted_groups = self._format_categorical_compare_groups(compare_cfg.get('selected_groups', []))
        if self.var_grupos_comparacion_categoria.get() != formatted_groups:
            self.var_grupos_comparacion_categoria.set(formatted_groups)

        if not quiet:
            mode_label = self._get_categorical_compare_display_value(compare_cfg.get('mode', 'all'))
            self.log(f"Config. categórica guardada al vuelo para '{var_name}': Ref='{ref_cat}', Modo='{mode_label}'.", "DEBUG")

        return True

    def _on_categorical_panel_change(self, event=None):
        self._persist_single_selected_categorical_config_from_panel(quiet=True)
        self._toggle_spline_and_refcat_controls()

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
            'combo_ref_categoria_seleccionada', 'combo_modo_comparacion_categoria',
            'var_grupos_comparacion_categoria', 'entry_grupos_comparacion_categoria',
            'var_usar_spline_seleccionada', 'checkbutton_usar_spline',
            'combo_tipo_spline_seleccionada', 'var_knots_spline_seleccionada',
            'spinbox_knots_spline', 'var_degree_spline_seleccionada',
            'spinbox_degree_spline' # Añadidos nuevos widgets
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
            # Ref categoría y comparación categórica deshabilitadas para múltiple selección
            self.combo_ref_categoria_seleccionada.set("")
            self.combo_ref_categoria_seleccionada.config(state="disabled", values=[])
            self.combo_modo_comparacion_categoria.set(self._get_default_categorical_compare_display())
            self.combo_modo_comparacion_categoria.config(state="disabled")
            self.var_grupos_comparacion_categoria.set("")
            self.entry_grupos_comparacion_categoria.config(state="disabled")
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
                unique_cats = self._get_reference_category_values(var_name_cfg)
                self.combo_ref_categoria_seleccionada['values'] = unique_cats
                current_ref_cat = self.ref_categories_config.get(var_name_cfg)
                if current_ref_cat in unique_cats:
                    self.combo_ref_categoria_seleccionada.set(current_ref_cat)
                elif unique_cats: # Default a la primera si no hay config o la config no es válida
                    self.combo_ref_categoria_seleccionada.set(unique_cats[0])
                else: # Sin categorías
                    self.combo_ref_categoria_seleccionada.set("")

                effective_ref_cat = self.combo_ref_categoria_seleccionada.get()
                compare_cfg = self._get_categorical_compare_config(var_name_cfg, effective_ref_cat)
                self.combo_modo_comparacion_categoria.set(
                    self._get_categorical_compare_display_value(compare_cfg.get("mode", "all"))
                )
                self.var_grupos_comparacion_categoria.set(
                    self._format_categorical_compare_groups(compare_cfg.get("selected_groups", []))
                )
            else: # Cuantitativa
                self.combo_ref_categoria_seleccionada.set("")
                self.combo_ref_categoria_seleccionada.config(state="disabled", values=[])
                self.combo_modo_comparacion_categoria.set(self._get_default_categorical_compare_display())
                self.var_grupos_comparacion_categoria.set("")
            
            # Configuración de Spline
            if current_var_type == "Cuantitativa" and var_name_cfg in self.spline_config_details:
                self.var_usar_spline_seleccionada.set(True)
                spl_conf = self.spline_config_details[var_name_cfg]
                internal_type_cfg = spl_conf.get('type', 'Natural')
                self.combo_tipo_spline_seleccionada.set(self._get_spline_display_value(internal_type_cfg))
                self.var_knots_spline_seleccionada.set(spl_conf.get('num_knots', 0))
                self.var_degree_spline_seleccionada.set(spl_conf.get('degree', 3)) # Cargar grado
            elif current_var_type == "Cuantitativa": # Es cuantitativa pero sin config de spline
                 self.var_usar_spline_seleccionada.set(False) # Asegurar que esté desactivado
                 self.combo_tipo_spline_seleccionada.set(self._get_default_spline_display()) # Default
                 self.var_knots_spline_seleccionada.set(0)
                 self.var_degree_spline_seleccionada.set(3) # Default grado
            else: # Cualitativa, spline no aplica
                self.var_usar_spline_seleccionada.set(False)
                self.var_knots_spline_seleccionada.set(0)
                self.var_degree_spline_seleccionada.set(3) # Resetear grado también

        else: # Ninguna seleccionada o error
            self.label_cov_seleccionada_nombre.config(text="Ninguna Seleccionada")
            self.radio_cuantitativa.config(state=tk.DISABLED)
            self.radio_cualitativa.config(state=tk.DISABLED)
            self.var_tipo_covariable_seleccionada.set("Cuantitativa") # Reset a default
            self.combo_ref_categoria_seleccionada.set("")
            self.combo_ref_categoria_seleccionada.config(state="disabled", values=[])
            self.combo_modo_comparacion_categoria.set(self._get_default_categorical_compare_display())
            self.combo_modo_comparacion_categoria.config(state="disabled")
            self.var_grupos_comparacion_categoria.set("")
            self.entry_grupos_comparacion_categoria.config(state="disabled")
            self.var_usar_spline_seleccionada.set(False)
            self.var_knots_spline_seleccionada.set(0)
            self.var_degree_spline_seleccionada.set(3) # Reset grado
            if hasattr(self, 'combo_tipo_spline_seleccionada'):
                self.combo_tipo_spline_seleccionada.set(self._get_default_spline_display())
            # Los demás (checkbutton_usar_spline, etc.) se manejan en _toggle

        self._toggle_spline_and_refcat_controls()


    def _toggle_spline_and_refcat_controls(self, event=None):
        """Habilita/deshabilita controles de spline y categoría de referencia."""
        # Asegurarse que todos los widgets existen antes de intentar configurarlos
        expected_widgets_for_toggle = [
            'radio_cuantitativa', 'radio_cualitativa', 'combo_ref_categoria_seleccionada',
            'combo_modo_comparacion_categoria', 'entry_grupos_comparacion_categoria',
            'checkbutton_usar_spline', 'combo_tipo_spline_seleccionada',
            'spinbox_knots_spline', 'spinbox_degree_spline' # Añadido spinbox_degree_spline
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

        # Categoría de Referencia y comparación categórica: solo para 1 cualitativa seleccionada
        if num_selected == 1 and current_type_choice == "Cualitativa":
            selected_var_name = self.listbox_covariables_disponibles.get(sel_indices[0])
            unique_cats = self._get_reference_category_values(selected_var_name)
            self.combo_ref_categoria_seleccionada['values'] = unique_cats

            widget_ref_cat = (self.combo_ref_categoria_seleccionada.get() or "").strip()
            stored_ref_cat = self.ref_categories_config.get(selected_var_name)
            if widget_ref_cat in unique_cats:
                current_ref_cat = widget_ref_cat
            elif stored_ref_cat in unique_cats:
                current_ref_cat = stored_ref_cat
            elif unique_cats:
                current_ref_cat = unique_cats[0]
            else:
                current_ref_cat = ""

            self.combo_ref_categoria_seleccionada.set(current_ref_cat)
            self.combo_ref_categoria_seleccionada.config(state="readonly" if unique_cats else "disabled")

            if self.combo_modo_comparacion_categoria.get() not in self.categorical_compare_reverse_map:
                compare_cfg = self._get_categorical_compare_config(selected_var_name, self.combo_ref_categoria_seleccionada.get())
                self.combo_modo_comparacion_categoria.set(
                    self._get_categorical_compare_display_value(compare_cfg.get("mode", "all"))
                )
                if compare_cfg.get("selected_groups") and not self.var_grupos_comparacion_categoria.get().strip():
                    self.var_grupos_comparacion_categoria.set(
                        self._format_categorical_compare_groups(compare_cfg.get("selected_groups", []))
                    )

            self.combo_modo_comparacion_categoria.config(state="readonly" if unique_cats else "disabled")
            selected_compare_mode = self._get_categorical_compare_internal_mode(self.combo_modo_comparacion_categoria.get())
            groups_entry_state = tk.NORMAL if (selected_compare_mode == "selected" and unique_cats) else tk.DISABLED
            self.entry_grupos_comparacion_categoria.config(state=groups_entry_state)
            if unique_cats:
                self._persist_single_selected_categorical_config_from_panel(quiet=True)
        else:
            self.combo_ref_categoria_seleccionada.config(state="disabled")
            self.combo_modo_comparacion_categoria.config(state="disabled")
            self.entry_grupos_comparacion_categoria.config(state="disabled")
            if num_selected != 1 or current_type_choice != "Cualitativa":
                 self.combo_ref_categoria_seleccionada.set("")
                 self.combo_ref_categoria_seleccionada['values'] = []
                 self.combo_modo_comparacion_categoria.set(self._get_default_categorical_compare_display())
                 self.var_grupos_comparacion_categoria.set("")


        # Spline: solo para cuantitativas (1 o más)
        can_use_spline = (num_selected > 0 and current_type_choice == "Cuantitativa")
        self.checkbutton_usar_spline.config(state=tk.NORMAL if can_use_spline else tk.DISABLED)
        if not can_use_spline: # Si no se puede usar spline, desactivar el check
            self.var_usar_spline_seleccionada.set(False)
        
        # Detalles de Spline: si se marca "Usar Spline" y es aplicable
        spline_general_details_state = "normal" if self.var_usar_spline_seleccionada.get() and can_use_spline else "disabled"
        self.combo_tipo_spline_seleccionada.config(state=spline_general_details_state)
        # Ajustar estado de nodos según tipo de spline
        spline_type_selected_display = self.combo_tipo_spline_seleccionada.get()
        spline_type_selected_simple = self._get_spline_internal_type(spline_type_selected_display)

        if spline_general_details_state == "normal" and spline_type_selected_simple in ("B-spline", "Natural"):
            knots_state = "normal"
        else:
            knots_state = "disabled"
        self.spinbox_knots_spline.config(state=knots_state)
        if knots_state == "disabled":
            self.var_knots_spline_seleccionada.set(0)

        # El grado solo tiene sentido para B-spline
        spline_degree_state_simple = "normal" if (spline_general_details_state == "normal" and spline_type_selected_simple == "B-spline") else "disabled"
        self.spinbox_degree_spline.config(state=spline_degree_state_simple)


        # Asegurar que los valores de spline no se mantengan si se cambia de tipo o se desmarca
        if spline_general_details_state == "disabled":
            self.combo_tipo_spline_seleccionada.set(self._get_default_spline_display()) # Reset
            self.var_knots_spline_seleccionada.set(0)
            self.var_degree_spline_seleccionada.set(3) # Reset grado
        elif spline_degree_state_simple == "disabled" and spline_type_selected_simple == "Natural":
            # Si es Natural Spline, el grado no es aplicable, resetear/fijar a 3 (aunque no se use directamente)
            self.var_degree_spline_seleccionada.set(3)

    def apply_covariate_config_to_selected(self):
        """Aplica la configuración de tipo, spline o categoría de referencia a las covariables seleccionadas."""
        sel_indices = self.listbox_covariables_disponibles.curselection()
        if not sel_indices:
            messagebox.showwarning("Sin Selección", "Seleccione una o más covariables para aplicar la configuración.", parent=self.parent_for_dialogs)
            return False

        selected_var_names = [self.listbox_covariables_disponibles.get(i) for i in sel_indices]
        
        new_var_type_bulk = self.var_tipo_covariable_seleccionada.get()
        use_spline_bulk = self.var_usar_spline_seleccionada.get()
        spline_type_display_bulk = self.combo_tipo_spline_seleccionada.get()
        spline_type_internal_bulk = self._get_spline_internal_type(spline_type_display_bulk)

        spline_degree_bulk = self._coerce_int_value(
            self.spinbox_degree_spline.get(),
            fallback=3,
            field_name="Grado de spline (panel simple)",
            min_value=1
        )
        self.var_degree_spline_seleccionada.set(spline_degree_bulk) # Mantener UI sincronizada

        spline_num_knots_bulk = self._coerce_int_value(
            self.spinbox_knots_spline.get(),
            fallback=0,
            field_name="Nodos internos (panel simple)",
            min_value=0
        )
        self.var_knots_spline_seleccionada.set(spline_num_knots_bulk)
        if spline_type_internal_bulk not in {"B-spline", "Natural"}:
            spline_num_knots_bulk = 0
        
        ref_category_for_single_selection = None
        categorical_compare_mode_bulk = "all"
        comparison_groups_for_single_selection = []
        if len(selected_var_names) == 1 and new_var_type_bulk == "Cualitativa" and self.combo_ref_categoria_seleccionada.cget('state') != 'disabled':
            ref_category_for_single_selection = self.combo_ref_categoria_seleccionada.get()
            if not ref_category_for_single_selection:
                messagebox.showwarning("Ref. Vacía",
                                       f"Para '{selected_var_names[0]}', seleccione una categoría de referencia o use el diálogo detallado.",
                                       parent=self.parent_for_dialogs)
                return False

            categorical_compare_mode_bulk = self._get_categorical_compare_internal_mode(
                self.combo_modo_comparacion_categoria.get()
            )
            comparison_groups_for_single_selection = self._parse_categorical_compare_groups(
                self.var_grupos_comparacion_categoria.get()
            )
            if categorical_compare_mode_bulk == "selected" and not comparison_groups_for_single_selection:
                messagebox.showwarning(
                    "Grupos requeridos",
                    f"Para '{selected_var_names[0]}', escribe una o más categorías en 'Grupos a comparar (coma)'.",
                    parent=self.parent_for_dialogs
                )
                return False

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
                if len(selected_var_names) == 1:
                    compare_cfg_for_var = self._build_categorical_compare_config(
                        var_name_apply,
                        ref_category=self.ref_categories_config.get(var_name_apply),
                        mode=categorical_compare_mode_bulk,
                        selected_groups=comparison_groups_for_single_selection
                    )
                    self.categorical_compare_config[var_name_apply] = compare_cfg_for_var
                    log_msgs_for_var.append(
                        f"ModoCat='{self._get_categorical_compare_display_value(compare_cfg_for_var.get('mode', 'all'))}'"
                    )
                    if compare_cfg_for_var.get('selected_groups'):
                        log_msgs_for_var.append(f"GruposCat={compare_cfg_for_var['selected_groups']}")
                elif var_name_apply not in self.categorical_compare_config:
                    self.categorical_compare_config[var_name_apply] = {"mode": "all", "selected_groups": []}
            
            elif new_var_type_bulk == "Cuantitativa":
                # Remove ref category config if it exists
                if var_name_apply in self.ref_categories_config:
                    del self.ref_categories_config[var_name_apply]
                    log_msgs_for_var.append("Config. Ref.Cat. eliminada (tipo cambiado a Cuantitativa).")
                if var_name_apply in self.categorical_compare_config:
                    del self.categorical_compare_config[var_name_apply]
                    log_msgs_for_var.append("Config. comparación categórica eliminada (tipo cambiado a Cuantitativa).")
                
                # Apply or remove spline config based on main panel's "Usar Spline"
                if use_spline_bulk:
                    if spline_type_internal_bulk == "Natural":
                        spline_degree_bulk = 3
                        self.var_degree_spline_seleccionada.set(3)
                        self.log(
                            f"Spline natural para '{var_name_apply}' fijado a cúbico; los naturales requieren grado 3.",
                            "INFO"
                        )

                    spline_df_bulk = self._derive_spline_df(
                        spline_type_internal_bulk,
                        spline_degree_bulk,
                        spline_num_knots_bulk
                    )
                    current_spline_config = {
                        'type': spline_type_internal_bulk,
                        'df': spline_df_bulk,
                        'num_knots': spline_num_knots_bulk,
                        'restricted': spline_type_internal_bulk == "Natural",
                        'degree': spline_degree_bulk if spline_type_internal_bulk == "B-spline" else 3
                    }
                    log_spline_parts = [f"Tipo='{spline_type_display_bulk}'", f"DF(auto)={spline_df_bulk}", f"Nodos={spline_num_knots_bulk}"]
                    if spline_type_internal_bulk == "B-spline":
                        log_spline_parts.append(f"Grado={spline_degree_bulk}")
                        if spline_num_knots_bulk == 0:
                            log_spline_parts.append("0 nodos => forma polinómica")
                    else:
                        log_spline_parts.append("Grado=3 (natural)")

                    self.spline_config_details[var_name_apply] = current_spline_config
                    log_msgs_for_var.append(f"Spline: {', '.join(log_spline_parts)}")
                else: # Not using spline via main panel
                    if var_name_apply in self.spline_config_details:
                        del self.spline_config_details[var_name_apply]
                        log_msgs_for_var.append("Config. spline eliminada (desmarcado en panel simple).")
            
            self.log(" ".join(log_msgs_for_var), "CONFIG")
            num_applied += 1

        if num_applied > 0:
            self.log(f"Configuración aplicada a {num_applied} variable(s) desde el panel simple.", "INFO")
        
        # Re-actualizar la UI de configuración para reflejar los cambios,
        # especialmente si la selección actual es una de las modificadas.
        self.on_covariate_select_for_config()
        self.log(f"Current spline_config_details after apply: {self.spline_config_details}", "DEBUG")
        return num_applied > 0


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

        # Contenedor responsivo en dos columnas para que el scroll capture todo el contenido.
        two_column_container = ttk.Frame(frame_config_general)
        two_column_container.pack(fill=tk.BOTH, expand=True)
        two_column_container.columnconfigure(0, weight=1, uniform="cox_model_cols")
        two_column_container.columnconfigure(1, weight=1, uniform="cox_model_cols")

        # --- Columna Izquierda ---
        left_col_frame = ttk.Frame(two_column_container, padding=5)
        left_col_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 8))

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
        right_col_frame = ttk.Frame(two_column_container, padding=5)
        right_col_frame.grid(row=0, column=1, sticky="nsew", padx=(8, 0))

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

        ttk.Checkbutton(grid_cv_ui, text="Calcular C-Index con holdout train/test", variable=self.calculate_test_cindex_var).grid(row=1, column=0, padx=5, pady=3, sticky=tk.W)
        ttk.Label(grid_cv_ui, text="Proporción test:").grid(row=1, column=1, padx=15, pady=3, sticky=tk.W)
        ttk.Entry(grid_cv_ui, textvariable=self.test_size_var, width=6).grid(row=1, column=2, padx=5, pady=3, sticky=tk.W)
        ttk.Label(grid_cv_ui, text="Semilla test:").grid(row=1, column=3, padx=15, pady=3, sticky=tk.W)
        ttk.Entry(grid_cv_ui, textvariable=self.test_random_seed_var, width=7).grid(row=1, column=4, padx=5, pady=3, sticky=tk.W)
        ttk.Checkbutton(grid_cv_ui, text="Estratificar por evento", variable=self.stratify_holdout_var).grid(row=2, column=0, padx=5, pady=3, sticky=tk.W)

        ttk.Label(grid_cv_ui, text="τ (tau) IPCW:").grid(row=3, column=0, padx=5, pady=3, sticky=tk.W)
        self.tau_mode_var = StringVar(value="Auto (P90)")
        ttk.Combobox(grid_cv_ui, textvariable=self.tau_mode_var,
                     values=["Auto (P90)", "Último evento", "Manual"],
                     state="readonly", width=16).grid(row=3, column=1, padx=5, pady=3, sticky=tk.W)
        self.tau_manual_var = StringVar(value="")
        ttk.Entry(grid_cv_ui, textvariable=self.tau_manual_var, width=7).grid(row=3, column=2, padx=5, pady=3, sticky=tk.W)
        ttk.Label(grid_cv_ui, text="(Uno, Antolini, Brier/IBS)", foreground="#555555").grid(row=3, column=3, padx=5, pady=3, sticky=tk.W)

        # Grid y modelos en contenedor apilado para que el scroll vertical funcione con todo el contenido.
        grid_and_results_container = ttk.Frame(g_content)
        grid_and_results_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        grid_container = ttk.Frame(grid_and_results_container)
        grid_container.pack(fill=tk.BOTH, expand=True)

        # Grid de experimentos de modelos
        self.frame_model_grid = ttk.LabelFrame(grid_container, text="Grid de escenarios (experimentos de modelos)", padding=10)
        self.frame_model_grid.pack(fill=tk.BOTH, expand=True)

        grid_columns = ("#", "Escenario", "Tipo", "Covariables", "Penalización", "Escalado", "Ties", "Splines", "Notas")
        self.model_grid_tree = ttk.Treeview(
            self.frame_model_grid,
            columns=grid_columns,
            show="headings",
            height=6,
            selectmode="extended"
        )

        column_widths = {
            "#": 30,
            "Escenario": 170,
            "Tipo": 100,
            "Covariables": 220,
            "Penalización": 130,
            "Escalado": 110,
            "Ties": 90,
            "Splines": 200,
            "Notas": 160
        }

        column_anchors = {
            "#": tk.CENTER,
            "Tipo": tk.CENTER,
            "Penalización": tk.W,
            "Escalado": tk.W,
            "Ties": tk.CENTER,
            "Escenario": tk.W,
            "Covariables": tk.W,
            "Splines": tk.W,
            "Notas": tk.W
        }

        for col in grid_columns:
            self.model_grid_tree.heading(col, text=col, command=lambda c=col: self._sort_grid_tv_column(c))
            self.model_grid_tree.column(col,
                                        width=column_widths.get(col, 120),
                                        minwidth=max(40, column_widths.get(col, 80)//2),
                                        anchor=column_anchors.get(col, tk.W))

        grid_tree_scroll_y = ttk.Scrollbar(self.frame_model_grid, orient=tk.VERTICAL, command=self.model_grid_tree.yview)
        grid_tree_scroll_x = ttk.Scrollbar(self.frame_model_grid, orient=tk.HORIZONTAL, command=self.model_grid_tree.xview)
        self.model_grid_tree.configure(yscrollcommand=grid_tree_scroll_y.set, xscrollcommand=grid_tree_scroll_x.set)

        grid_tree_scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
        grid_tree_scroll_x.pack(side=tk.BOTTOM, fill=tk.X)
        self.model_grid_tree.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.model_grid_tree.bind("<Double-1>", self._on_grid_entry_double_click)
        self.model_grid_tree.bind("<Button-3>", self._show_grid_tv_column_menu)
        self._grid_tv_col_config = {c: {"visible": True, "width": column_widths.get(c, 120), "heading": c} for c in grid_columns}
        self._restore_saved_layout("cox_model_grid", self._grid_tv_col_config)
        self._grid_sort_reversed = {}
        self._apply_treeview_column_layout(self.model_grid_tree, self._grid_tv_col_config)
        self._register_saved_layout("cox_model_grid", self._grid_tv_col_config, self.model_grid_tree)

        grid_button_frame = ttk.Frame(self.frame_model_grid)
        grid_button_frame.pack(fill=tk.X, padx=5, pady=(5, 0))

        ttk.Button(grid_button_frame, text="Agregar escenario actual", command=self._add_current_config_to_grid).pack(side=tk.LEFT, padx=4, pady=2)
        ttk.Button(grid_button_frame, text="Variaciones spline y penalización...", command=self._open_spline_penalization_variations_dialog).pack(side=tk.LEFT, padx=4, pady=2)
        ttk.Button(grid_button_frame, text="Ejecutar modelo", command=self._execute_selected_grid_entries).pack(side=tk.LEFT, padx=4, pady=2)
        ttk.Button(grid_button_frame, text="Eliminar selección", command=self._remove_selected_grid_entries).pack(side=tk.RIGHT, padx=4, pady=2)

        # Botón Ejecutar
        frame_ejecutar = ttk.Frame(grid_container)
        frame_ejecutar.pack(fill=tk.X, pady=(10, 10))
        btn_ejecutar = ttk.Button(frame_ejecutar, text="▶ Ejecutar Modelado Cox", command=self._execute_cox_modeling_orchestrator)
        btn_ejecutar.pack(padx=10, pady=5, ipady=5)

        # Treeview para Modelos Generados
        self.frame_modelos_generados_display = ttk.LabelFrame(grid_and_results_container, text="Modelos Cox Generados en esta Sesión", padding=10)
        self.frame_modelos_generados_display.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        cols_tv = ("#", "Nombre Modelo", "Variables y Splines", "Test %", "AIC", "BIC", "-2 LogLik",
                   "C-Index (Train)", "C-Index (Test)", "C-Index (CV/Test)", "ΔTest-Train",
                   "C-Uno (IPCW)", "C-Antolini (Ctd)", "τ (tau)", "IBS",
                   "C@Q25", "C@Q50", "C@Q75", "Brier@Q25", "Brier@Q50", "Brier@Q75",
                   "AUC@Q25", "AUC@Q50", "AUC@Q75",
                   "LR p (global)", "Wald p (global)",
                   "Schoenfeld (p min)", "Wald (p max)")
        self.treeview_lista_modelos = ttk.Treeview(
            self.frame_modelos_generados_display,
            columns=cols_tv,
            show="headings",
            height=12,
            selectmode="extended"
        )
        self.treeview_sort_reversed = {}

        col_widths = {
            "#": 30, "Nombre Modelo": 180, "Variables y Splines": 250, "Test %": 70,
            "AIC": 80, "BIC": 80, "-2 LogLik": 80,
            "C-Index (Train)": 165, "C-Index (Test)": 165, "C-Index (CV/Test)": 175, "ΔTest-Train": 95,
            "C-Uno (IPCW)": 110, "C-Antolini (Ctd)": 120, "τ (tau)": 80, "IBS": 80,
            "C@Q25": 80, "C@Q50": 80, "C@Q75": 80,
            "Brier@Q25": 85, "Brier@Q50": 85, "Brier@Q75": 85,
            "AUC@Q25": 80, "AUC@Q50": 80, "AUC@Q75": 80,
            "LR p (global)": 100, "Wald p (global)": 100,
            "Schoenfeld (p min)": 110, "Wald (p max)": 90
        }
        col_anchors = {
            "#": tk.CENTER, "Test %": tk.E, "AIC": tk.E, "BIC": tk.E, "-2 LogLik": tk.E,
            "C-Index (Train)": tk.E, "C-Index (Test)": tk.E, "C-Index (CV/Test)": tk.E,
            "ΔTest-Train": tk.E, "C-Uno (IPCW)": tk.E, "C-Antolini (Ctd)": tk.E, "τ (tau)": tk.E,
            "IBS": tk.E, "C@Q25": tk.E, "C@Q50": tk.E, "C@Q75": tk.E,
            "Brier@Q25": tk.E, "Brier@Q50": tk.E, "Brier@Q75": tk.E,
            "AUC@Q25": tk.E, "AUC@Q50": tk.E, "AUC@Q75": tk.E,
            "LR p (global)": tk.E, "Wald p (global)": tk.E,
            "Schoenfeld (p min)": tk.E, "Wald (p max)": tk.E
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
        self.treeview_lista_modelos.bind("<Button-3>", self._show_models_tv_column_menu)
        self._models_tv_col_config = {c: {"visible": True, "width": col_widths.get(c, 120), "heading": c} for c in cols_tv}
        self._restore_saved_layout("cox_generated_models", self._models_tv_col_config)
        self._apply_treeview_column_layout(self.treeview_lista_modelos, self._models_tv_col_config)
        self._register_saved_layout("cox_generated_models", self._models_tv_col_config, self.treeview_lista_modelos)

        # Botones de Acción para Modelo Seleccionado
        frame_acciones = ttk.Frame(self.frame_modelos_generados_display, padding=(0,5,0,0))
        frame_acciones.pack(fill=tk.X, pady=5)
        
        acciones_config_btns = [
            ("Ver Resumen", self.show_selected_model_summary),
            ("Generar Gráficos Cox", self.open_graph_selection_dialog),
            ("Nomograma", self.generate_nomogram_for_selected_model),
            ("Calibración OOS (CV)", self.show_new_calibration_plots),
            ("Diagnóstico de Colinealidad", self._calculate_and_show_vif), # <-- NUEVO
            ("Predicción", self.realizar_prediccion),
            ("Exportar Resumen", self.export_model_summary),
            ("Guardar Modelo", self.save_model),
            ("Cargar Modelo", self.load_model_from_file),
            ("Reporte Metod.", self.show_methodological_report)
        ]
        
        # Layout dinámico para botones de acción
        max_btns_per_row = 4  # Menos botones por fila para evitar que se compriman o desaparezcan.
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
            elif text == "Nomograma":
                self.btn_nomogram = button_widget

        if self.btn_oos_calibration:
            self.btn_oos_calibration.config(state=tk.DISABLED)
        if self.btn_collinearity_diag: # <-- NUEVO
            self.btn_collinearity_diag.config(state=tk.DISABLED)
        if self.btn_nomogram:
            self.btn_nomogram.config(state=tk.DISABLED)

        # Add the new button row for clear models
        clear_models_frame = ttk.Frame(frame_acciones)
        clear_models_frame.pack(fill=tk.X, pady=5)
        self.btn_delete_model = ttk.Button(clear_models_frame, text="Eliminar modelos seleccionados", command=self._delete_selected_model)
        self.btn_delete_model.pack(side=tk.LEFT, padx=5)
        ttk.Button(clear_models_frame, text="Limpiar Todos los Modelos", command=self._clear_all_generated_models).pack(side=tk.RIGHT, padx=5)
        if self.btn_delete_model:
            self.btn_delete_model.config(state=tk.DISABLED)

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

    # --- Grid de escenarios para exploración ---

    def _capture_current_model_settings(self):
        try:
            selected_covs = []
            if hasattr(self, 'listbox_covariables_disponibles'):
                listbox_size = self.listbox_covariables_disponibles.size()
                selected_indices = set(self.listbox_covariables_disponibles.curselection())
                time_col = self.combo_col_tiempo.get().strip()
                event_col = self.combo_col_evento.get().strip()
                for idx in range(listbox_size):
                    if idx not in selected_indices:
                        continue
                    item_value = self.listbox_covariables_disponibles.get(idx)
                    if item_value in (time_col, event_col):
                        continue
                    selected_covs.append(item_value)
                selected_covs = list(dict.fromkeys(selected_covs))
        except Exception as exc:
            self.log(f"No se pudieron capturar las covariables seleccionadas: {exc}", "WARN")
            selected_covs = []

        model_type = self.cox_model_type_var.get() if hasattr(self, 'cox_model_type_var') else "Multivariado"
        selection_method = self.var_selection_method_var.get() if hasattr(self, 'var_selection_method_var') else "Ninguno (usar todas)"

        p_enter_val = self._coerce_float_value(self.p_enter_var, 0.05, "P-valor de entrada", min_value=0.0)
        p_remove_val = self._coerce_float_value(self.p_remove_var, 0.10, "P-valor de salida", min_value=0.0)

        pen_method = self.penalization_method_var.get() if hasattr(self, 'penalization_method_var') else "Ninguna"
        pen_value = self._coerce_float_value(self.penalizer_strength_var, 0.0, "Penalización (λ)", min_value=0.0)
        pen_alpha = self._coerce_float_value(self.l1_ratio_for_elasticnet_var, 0.5, "Penalización α", min_value=0.0)
        if pen_alpha > 1.0:
            self.log(f"Penalización α: valor {pen_alpha} mayor a 1.0. Ajustado a 1.0.", "WARN")
            pen_alpha = 1.0

        scaling_method = self.covariate_scaling_method_var.get() if hasattr(self, 'covariate_scaling_method_var') else "Ninguna"
        tie_method = self.tie_handling_method_var.get() if hasattr(self, 'tie_handling_method_var') else "efron"

        calculate_cv = bool(self.calculate_cv_cindex_var.get()) if hasattr(self, 'calculate_cv_cindex_var') else False
        cv_kfolds = self._coerce_int_value(self.cv_num_kfolds_var, 5, "CV K-folds", min_value=2) if hasattr(self, 'cv_num_kfolds_var') else 5
        cv_seed = self._coerce_int_value(self.cv_random_seed_var, 42, "CV seed") if hasattr(self, 'cv_random_seed_var') else 42
        calculate_test_holdout = bool(self.calculate_test_cindex_var.get()) if hasattr(self, 'calculate_test_cindex_var') else False
        test_size = self._coerce_float_value(self.test_size_var, 0.25, "Proporción test", min_value=0.05) if hasattr(self, 'test_size_var') else 0.25
        if test_size >= 0.95:
            self.log(f"Proporción test: valor {test_size} mayor o igual a 0.95. Ajustado a 0.95.", "WARN")
            test_size = 0.95
        test_seed = self._coerce_int_value(self.test_random_seed_var, 42, "Semilla test") if hasattr(self, 'test_random_seed_var') else 42

        time_column = self.combo_col_tiempo.get().strip() if hasattr(self, 'combo_col_tiempo') else ""
        event_column = self.combo_col_evento.get().strip() if hasattr(self, 'combo_col_evento') else ""

        dataset_metadata = copy.deepcopy(self.shared_dataset_metadata) if hasattr(self, 'shared_dataset_metadata') else {}
        filter_summary = list(self.current_shared_filter_summary) if hasattr(self, 'current_shared_filter_summary') else []

        config_snapshot = {
            "model_type": model_type,
            "selected_covariables": selected_covs,
            "var_selection_method": selection_method,
            "p_enter": p_enter_val,
            "p_remove": p_remove_val,
            "penalization": {
                "method": pen_method,
                "value": pen_value,
                "l1_ratio": pen_alpha
            },
            "scaling_method": scaling_method,
            "tie_method": tie_method,
            "calculate_cv": calculate_cv,
            "cv_kfolds": cv_kfolds,
            "cv_seed": cv_seed,
            "calculate_test_holdout": calculate_test_holdout,
            "test_size": test_size,
            "test_seed": test_seed,
            "categorical_configs": copy.deepcopy(self.categorical_compare_config),
            "spline_configs": copy.deepcopy(self.spline_config_details),
            "generate_univariate_forest_plot": bool(self.generate_univariate_forest_plot_var.get()) if hasattr(self, 'generate_univariate_forest_plot_var') else True,
            "time_column": time_column,
            "event_column": event_column,
            "dataset_metadata": dataset_metadata,
            "filter_summary": filter_summary,
            "notes": ""
        }

        return config_snapshot

    def _add_current_config_to_grid(self):
        config_snapshot = self._capture_current_model_settings()
        if config_snapshot is None:
            messagebox.showwarning("Sin configuración", "No se pudo capturar la configuración actual del modelo.", parent=self.parent_for_dialogs)
            return

        default_name = f"Escenario {self.model_grid_counter + 1}"
        scenario_name = simpledialog.askstring(
            "Nuevo escenario",
            "Nombre para el escenario:",
            initialvalue=default_name,
            parent=self.parent_for_dialogs
        )

        if scenario_name is None:
            self.log("Alta de escenario cancelada por el usuario.", "INFO")
            return

        scenario_name = scenario_name.strip() or default_name

        entry = {
            "id": self.model_grid_counter,
            "name": scenario_name,
            "config": config_snapshot,
            "status": "pendiente",
            "notes": "",
            "origin": "manual"
        }

        self.model_grid_entries.append(entry)
        self.model_grid_counter += 1
        self._refresh_model_grid_tree()
        self.log(f"Escenario '{scenario_name}' agregado al grid con {len(config_snapshot['selected_covariables'])} covariable(s).", "SUCCESS")

    def _parse_optional_int_list(self, raw_text):
        values = []
        if not raw_text:
            return values
        for token in raw_text.split(','):
            token = token.strip()
            if not token:
                continue
            lower = token.lower()
            if lower in {"auto", "", "none", "-"}:
                values.append(None)
                continue
            try:
                values.append(int(token))
            except ValueError:
                raise ValueError(f"'{token}' no es un entero válido.")
        return values

    def _add_spline_variations_to_grid(self):
        self._open_spline_penalization_variations_dialog(default_tab="spline", section_overrides={"spline": True, "penal": False})

    def _add_penalization_variations_to_grid(self):
        self._open_spline_penalization_variations_dialog(default_tab="penal", section_overrides={"spline": False, "penal": True})

    def _open_spline_penalization_variations_dialog(self, default_tab="combined", section_overrides=None):
        base_snapshot = self._capture_current_model_settings()
        if base_snapshot is None:
            messagebox.showwarning("Sin configuración", "No se pudo capturar la configuración actual del modelo.", parent=self.parent_for_dialogs)
            return

        current_df = None
        if isinstance(self.data, pd.DataFrame) and not self.data.empty:
            current_df = self.data
        elif isinstance(self.raw_data, pd.DataFrame) and not self.raw_data.empty:
            current_df = self.raw_data

        numeric_cols = []
        if current_df is not None and not current_df.empty:
            numeric_cols = [
                col for col in current_df.columns
                if pd.api.types.is_numeric_dtype(current_df[col])
            ]

        default_spline_enabled = bool(numeric_cols)
        default_penal_enabled = True

        if section_overrides:
            if "spline" in section_overrides:
                default_spline_enabled = section_overrides["spline"] and bool(numeric_cols)
            if "penal" in section_overrides:
                default_penal_enabled = section_overrides["penal"]

        dialog = tk.Toplevel(self.parent_for_dialogs)
        dialog.title("Variaciones de spline y penalización")
        dialog.transient(self.parent_for_dialogs)
        dialog.grab_set()

        notebook = ttk.Notebook(dialog)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        spline_tab = ttk.Frame(notebook)
        penal_tab = ttk.Frame(notebook)
        notebook.add(spline_tab, text="Splines")
        notebook.add(penal_tab, text="Penalización")

        # --- Tab Splines ---
        spline_enabled_var = tk.BooleanVar(value=default_spline_enabled)

        spline_header = ttk.Frame(spline_tab)
        spline_header.pack(fill=tk.X, pady=(0, 5))
        spline_toggle_btn = ttk.Checkbutton(
            spline_header,
            text="Generar variaciones de spline",
            variable=spline_enabled_var
        )
        spline_toggle_btn.pack(side=tk.LEFT, anchor=tk.W)

        spline_form = ttk.Frame(spline_tab)
        spline_form.pack(fill=tk.X, pady=5)

        ttk.Label(spline_form, text="Covariable:").grid(row=0, column=0, sticky=tk.W, pady=2)
        spline_var_combo = ttk.Combobox(spline_form, values=sorted(numeric_cols), state="readonly")
        if numeric_cols:
            spline_var_combo.set(sorted(numeric_cols)[0])
        spline_var_combo.grid(row=0, column=1, sticky=tk.EW, pady=2)

        ttk.Label(spline_form, text="Tipo de spline:").grid(row=1, column=0, sticky=tk.W, pady=2)
        spline_type_values = list(self.spline_type_display_map.values())
        spline_type_combo = ttk.Combobox(spline_form, values=spline_type_values, state="readonly")
        spline_type_combo.set(self._get_default_spline_display())
        spline_type_combo.grid(row=1, column=1, sticky=tk.EW, pady=2)

        ttk.Label(spline_form, text="DF (lista, ej. 3,4):").grid(row=2, column=0, sticky=tk.W, pady=2)
        spline_df_entry = ttk.Entry(spline_form)
        spline_df_entry.insert(0, "3,4,5")
        spline_df_entry.grid(row=2, column=1, sticky=tk.EW, pady=2)

        ttk.Label(spline_form, text="Nodos internos (lista, opcional):").grid(row=3, column=0, sticky=tk.W, pady=2)
        spline_knots_entry = ttk.Entry(spline_form)
        spline_knots_entry.insert(0, "auto")
        spline_knots_entry.grid(row=3, column=1, sticky=tk.EW, pady=2)

        ttk.Label(spline_form, text="Grado (solo B-spline):").grid(row=4, column=0, sticky=tk.W, pady=2)
        spline_degree_entry = ttk.Entry(spline_form)
        spline_degree_entry.insert(0, "3")
        spline_degree_entry.grid(row=4, column=1, sticky=tk.EW, pady=2)

        ttk.Label(spline_form, text="Nombre base para escenarios:").grid(row=5, column=0, sticky=tk.W, pady=(8, 2))
        spline_base_name_entry = ttk.Entry(spline_form)
        spline_base_name_entry.insert(0, f"Escenario spline {self.model_grid_counter + 1}")
        spline_base_name_entry.grid(row=5, column=1, sticky=tk.EW, pady=(8, 2))

        ttk.Label(spline_form, text="Notas (opcional):").grid(row=6, column=0, sticky=tk.W, pady=2)
        spline_notes_entry = ttk.Entry(spline_form)
        spline_notes_entry.grid(row=6, column=1, sticky=tk.EW, pady=2)

        spline_form.columnconfigure(1, weight=1)

        if not numeric_cols:
            ttk.Label(
                spline_tab,
                text="No se encontraron covariables numéricas en el dataset actual.",
                foreground="gray"
            ).pack(pady=(0, 5), anchor="w")

        # --- Tab Penalización ---
        penal_enabled_var = tk.BooleanVar(value=default_penal_enabled)

        penal_header = ttk.Frame(penal_tab)
        penal_header.pack(fill=tk.X, pady=(0, 5))
        ttk.Checkbutton(
            penal_header,
            text="Generar variaciones de penalización",
            variable=penal_enabled_var
        ).pack(side=tk.LEFT, anchor=tk.W)

        penal_form = ttk.Frame(penal_tab)
        penal_form.pack(fill=tk.X, pady=5)

        ttk.Label(penal_form, text="Métodos disponibles:").grid(row=0, column=0, sticky=tk.W, pady=2)
        methods_available = ["Ninguna", "L2 (Ridge)", "L1 (Lasso)", "ElasticNet"]
        current_method = base_snapshot.get("penalization", {}).get("method", "Ninguna") or "Ninguna"
        penal_methods_frame = ttk.Frame(penal_form)
        penal_methods_frame.grid(row=0, column=1, sticky=tk.W)
        penal_method_vars = {}
        penal_method_checkbuttons = []
        for idx, method_name in enumerate(methods_available):
            var = tk.BooleanVar(value=(method_name == current_method))
            penal_method_vars[method_name] = var
            chk = ttk.Checkbutton(penal_methods_frame, text=method_name, variable=var)
            chk.grid(row=idx, column=0, sticky=tk.W, pady=2)
            penal_method_checkbuttons.append(chk)

        ttk.Label(penal_form, text="Valores λ (separados por coma):").grid(row=1, column=0, sticky=tk.W, pady=2)
        penal_lambda_entry = ttk.Entry(penal_form)
        penal_lambda_entry.insert(0, "0.01,0.1,0.5")
        penal_lambda_entry.grid(row=1, column=1, sticky=tk.EW, pady=2, padx=5)

        ttk.Label(penal_form, text="Ratios α ElasticNet (0-1, coma):").grid(row=2, column=0, sticky=tk.W, pady=2)
        penal_ratio_entry = ttk.Entry(penal_form)
        penal_ratio_entry.insert(0, "0.25,0.5,0.75")
        penal_ratio_entry.grid(row=2, column=1, sticky=tk.EW, pady=2, padx=5)

        ttk.Label(penal_form, text="Nombre base para escenarios:").grid(row=3, column=0, sticky=tk.W, pady=(8, 2))
        penal_base_name_entry = ttk.Entry(penal_form)
        penal_base_name_entry.insert(0, f"Escenario penalización {self.model_grid_counter + 1}")
        penal_base_name_entry.grid(row=3, column=1, sticky=tk.EW, pady=(8, 2), padx=5)

        ttk.Label(penal_form, text="Notas (opcional):").grid(row=4, column=0, sticky=tk.W, pady=2)
        penal_notes_entry = ttk.Entry(penal_form)
        penal_notes_entry.insert(0, base_snapshot.get("notes", ""))
        penal_notes_entry.grid(row=4, column=1, sticky=tk.EW, pady=2, padx=5)

        penal_form.columnconfigure(1, weight=1)

        # --- Control de estado de widgets ---
        penal_entry_widgets = [
            penal_lambda_entry,
            penal_ratio_entry,
            penal_base_name_entry,
            penal_notes_entry
        ]

        def update_spline_state(*_):
            state = "readonly" if spline_enabled_var.get() else "disabled"
            entry_state = tk.NORMAL if spline_enabled_var.get() else tk.DISABLED
            spline_var_combo.configure(state=state)
            spline_type_combo.configure(state=state)
            for widget in [spline_df_entry, spline_knots_entry, spline_degree_entry, spline_base_name_entry, spline_notes_entry]:
                widget.configure(state=entry_state)

        def update_penal_state(*_):
            state = tk.NORMAL if penal_enabled_var.get() else tk.DISABLED
            for widget in penal_entry_widgets:
                widget.configure(state=state)
            for chk in penal_method_checkbuttons:
                if penal_enabled_var.get():
                    chk.state(["!disabled"])
                else:
                    chk.state(["disabled"])

        spline_enabled_var.trace_add("write", update_spline_state)
        penal_enabled_var.trace_add("write", update_penal_state)

        update_spline_state()
        update_penal_state()

        if not numeric_cols:
            spline_toggle_btn.state(["disabled"])

        # --- Botones ---
        button_frame = ttk.Frame(dialog)
        button_frame.pack(pady=(0, 10))

        def parse_float_list(raw_text, *, allow_zero=False):
            clean_values = []
            if not raw_text:
                return clean_values
            for token in raw_text.split(','):
                stripped = token.strip()
                if not stripped:
                    continue
                try:
                    value = float(stripped)
                except ValueError as err:
                    raise ValueError(f"'{stripped}' no es un número válido") from err
                if allow_zero:
                    if value < 0:
                        raise ValueError(f"'{stripped}' debe ser mayor o igual a 0")
                else:
                    if value <= 0:
                        raise ValueError(f"'{stripped}' debe ser mayor que 0")
                clean_values.append(value)
            return clean_values

        def on_accept():
            if not spline_enabled_var.get() and not penal_enabled_var.get():
                messagebox.showinfo("Sin selección", "Activa al menos una sección (Splines o Penalización) para generar variaciones.", parent=dialog)
                return

            def merge_notes(*notes_parts):
                merged = []
                for part in notes_parts:
                    if part:
                        clean = part.strip()
                        if clean and clean not in merged:
                            merged.append(clean)
                return " | ".join(merged)

            def apply_spline_variant_to_config(config_obj, variant_data):
                target_var = variant_data["target_var"]
                spline_details = copy.deepcopy(variant_data["spline_details"])
                selected_covs = config_obj.setdefault("selected_covariables", [])
                if target_var not in selected_covs:
                    selected_covs.append(target_var)
                config_obj["selected_covariables"] = list(dict.fromkeys(selected_covs))
                spline_configs = config_obj.setdefault("spline_configs", {})
                spline_configs[target_var] = spline_details

            def apply_penal_variant_to_config(config_obj, variant_data):
                config_obj["penalization"] = {
                    "method": variant_data["method"],
                    "value": variant_data["lambda_value"],
                    "l1_ratio": variant_data["l1_ratio"]
                }

            spline_variants = []
            penal_variants = []

            if spline_enabled_var.get():
                if not numeric_cols:
                    messagebox.showwarning("Sin covariables", "No hay covariables numéricas disponibles para generar variaciones de spline.", parent=dialog)
                    return

                target_var = spline_var_combo.get().strip()
                if not target_var:
                    messagebox.showwarning("Covariable requerida", "Selecciona una covariable numérica.", parent=dialog)
                    return

                type_display = spline_type_combo.get().strip()
                if not type_display:
                    messagebox.showwarning("Tipo de spline", "Selecciona el tipo de spline.", parent=dialog)
                    return

                try:
                    df_values = self._parse_optional_int_list(spline_df_entry.get().strip()) or [None]
                except ValueError as err_int:
                    messagebox.showerror("DF inválidos", str(err_int), parent=dialog)
                    return

                try:
                    knot_values = self._parse_optional_int_list(spline_knots_entry.get().strip())
                except ValueError as err_knots:
                    messagebox.showerror("Nodos inválidos", str(err_knots), parent=dialog)
                    return

                knot_values = knot_values if knot_values else [None]

                try:
                    degree_values = self._parse_optional_int_list(spline_degree_entry.get().strip())
                except ValueError as err_deg:
                    messagebox.showerror("Grado inválido", str(err_deg), parent=dialog)
                    return

                if not degree_values:
                    degree_values = [3]

                base_label_spline = spline_base_name_entry.get().strip() or f"Escenario spline {self.model_grid_counter + 1}"
                notes_spline = spline_notes_entry.get().strip()
                internal_type = self.spline_type_reverse_map.get(type_display, "Natural")

                for df_val in df_values:
                    for knot_val in knot_values:
                        degrees_iter = degree_values if internal_type == "B-spline" else [3]
                        for degree_val in degrees_iter:
                            spline_details = {
                                "type": internal_type,
                                "df": df_val,
                                "num_knots": knot_val,
                                "restricted": internal_type == "Natural",
                                "degree": degree_val if internal_type == "B-spline" else 3
                            }

                            detail_fragments = []
                            if df_val is not None:
                                detail_fragments.append(f"df={df_val}")
                            if knot_val is not None:
                                detail_fragments.append(f"nudos={knot_val}")
                            if internal_type == "B-spline" and degree_val is not None:
                                detail_fragments.append(f"grado={degree_val}")

                            label = base_label_spline
                            if detail_fragments:
                                label += " | " + ", ".join(detail_fragments)

                            spline_variants.append({
                                "target_var": target_var,
                                "spline_details": spline_details,
                                "label": label,
                                "notes": notes_spline
                            })

                if not spline_variants:
                    messagebox.showinfo("Sin combinaciones", "No se generó ninguna variación válida de spline.", parent=dialog)
                    return

            if penal_enabled_var.get():
                selected_methods = [method for method, var in penal_method_vars.items() if var.get()]
                if not selected_methods:
                    messagebox.showwarning("Selección requerida", "Selecciona al menos un método de penalización.", parent=dialog)
                    return

                try:
                    lambda_values = parse_float_list(penal_lambda_entry.get().strip(), allow_zero=True)
                except ValueError as err_lambda:
                    messagebox.showerror("Valores λ inválidos", str(err_lambda), parent=dialog)
                    return

                try:
                    ratio_values = parse_float_list(penal_ratio_entry.get().strip(), allow_zero=True)
                except ValueError as err_ratio:
                    messagebox.showerror("Ratios α inválidos", str(err_ratio), parent=dialog)
                    return

                if not lambda_values:
                    fallback_lambda = base_snapshot.get("penalization", {}).get("value", 0.1)
                    lambda_values = [val for val in [fallback_lambda, 0.1] if val > 0]
                    if not lambda_values:
                        lambda_values = [0.1]

                ratio_values = [min(max(val, 0.0), 1.0) for val in ratio_values if 0.0 <= val <= 1.0]
                if not ratio_values:
                    ratio_values = [base_snapshot.get("penalization", {}).get("l1_ratio", 0.5) or 0.5]

                base_label_penal = penal_base_name_entry.get().strip() or f"Escenario penalización {self.model_grid_counter + 1}"
                notes_penal = penal_notes_entry.get().strip()

                for method in selected_methods:
                    if method == "Ninguna":
                        penal_variants.append({
                            "method": "Ninguna",
                            "lambda_value": 0.0,
                            "l1_ratio": 0.0,
                            "label": f"{base_label_penal} | {method}",
                            "notes": notes_penal
                        })
                        continue

                    usable_lambdas = [val for val in lambda_values if val > 0]
                    if not usable_lambdas:
                        continue

                    if method == "ElasticNet":
                        ratios_to_apply = ratio_values or [0.5]
                        for lambda_val in usable_lambdas:
                            for ratio_val in ratios_to_apply:
                                ratio_clipped = min(max(ratio_val, 0.0), 1.0)
                                penal_variants.append({
                                    "method": method,
                                    "lambda_value": lambda_val,
                                    "l1_ratio": ratio_clipped,
                                    "label": f"{base_label_penal} | {method} λ={lambda_val:.4g}, α={ratio_clipped:.3g}",
                                    "notes": notes_penal
                                })
                    else:
                        for lambda_val in usable_lambdas:
                            l1_ratio = 1.0 if method == "L1 (Lasso)" else 0.0
                            penal_variants.append({
                                "method": method,
                                "lambda_value": lambda_val,
                                "l1_ratio": l1_ratio,
                                "label": f"{base_label_penal} | {method} λ={lambda_val:.4g}",
                                "notes": notes_penal
                            })

                if penal_enabled_var.get() and not penal_variants:
                    messagebox.showinfo("Sin combinaciones", "No se generó ninguna variación válida de penalización.", parent=dialog)
                    return

            entries_to_add = []
            count_spline_only = 0
            count_penal_only = 0
            count_combined = 0

            if spline_variants and penal_variants:
                for spline_variant in spline_variants:
                    for penal_variant in penal_variants:
                        config_variant = copy.deepcopy(base_snapshot)
                        apply_spline_variant_to_config(config_variant, spline_variant)
                        apply_penal_variant_to_config(config_variant, penal_variant)
                        config_variant["notes"] = merge_notes(spline_variant.get("notes"), penal_variant.get("notes"))
                        entry_label = " | ".join([part for part in [spline_variant.get("label"), penal_variant.get("label")] if part])
                        entries_to_add.append({
                            "config": config_variant,
                            "label": entry_label,
                            "notes": config_variant.get("notes", ""),
                            "origin": "spline_penal_combination"
                        })
                        count_combined += 1
            elif spline_variants:
                for spline_variant in spline_variants:
                    config_variant = copy.deepcopy(base_snapshot)
                    apply_spline_variant_to_config(config_variant, spline_variant)
                    config_variant["notes"] = merge_notes(spline_variant.get("notes"))
                    entries_to_add.append({
                        "config": config_variant,
                        "label": spline_variant.get("label"),
                        "notes": config_variant.get("notes", ""),
                        "origin": "spline_variation"
                    })
                    count_spline_only += 1
            elif penal_variants:
                for penal_variant in penal_variants:
                    config_variant = copy.deepcopy(base_snapshot)
                    apply_penal_variant_to_config(config_variant, penal_variant)
                    config_variant["notes"] = merge_notes(penal_variant.get("notes"))
                    entries_to_add.append({
                        "config": config_variant,
                        "label": penal_variant.get("label"),
                        "notes": config_variant.get("notes", ""),
                        "origin": "penalization_variation"
                    })
                    count_penal_only += 1

            total_added = len(entries_to_add)
            if total_added == 0:
                messagebox.showinfo("Sin combinaciones", "No se generó ninguna variación válida.", parent=dialog)
                return

            for entry_data in entries_to_add:
                entry = {
                    "id": self.model_grid_counter,
                    "name": entry_data["label"] or f"Escenario {self.model_grid_counter}",
                    "config": entry_data["config"],
                    "status": "pendiente",
                    "notes": entry_data["notes"],
                    "origin": entry_data["origin"]
                }
                self.model_grid_entries.append(entry)
                self.model_grid_counter += 1

            dialog.destroy()
            self._refresh_model_grid_tree()

            log_parts = []
            if count_combined:
                log_parts.append(f"{count_combined} combinaciones spline+penalización")
            if count_spline_only:
                log_parts.append(f"{count_spline_only} variaciones solo spline")
            if count_penal_only:
                log_parts.append(f"{count_penal_only} variaciones solo penalización")
            self.log("Se agregaron " + " y ".join(log_parts) + " al grid.", "SUCCESS")

            summary_lines = ["Variaciones generadas:"]
            summary_lines.append(f"Combinadas spline+penalización: {count_combined}")
            summary_lines.append(f"Solo spline: {count_spline_only}")
            summary_lines.append(f"Solo penalización: {count_penal_only}")
            messagebox.showinfo("Variaciones agregadas", "\n".join(summary_lines), parent=self.parent_for_dialogs)

        def on_cancel():
            dialog.destroy()

        ttk.Button(button_frame, text="Generar", command=on_accept).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Cancelar", command=on_cancel).pack(side=tk.RIGHT, padx=5)

        if default_tab == "spline":
            notebook.select(spline_tab)
        elif default_tab == "penal":
            notebook.select(penal_tab)

        self.parent_for_dialogs.wait_window(dialog)

    def _remove_selected_grid_entries(self):
        if not self.model_grid_tree:
            return

        selected = self.model_grid_tree.selection()
        if not selected:
            messagebox.showinfo("Grid", "Selecciona uno o más escenarios para eliminarlos.", parent=self.parent_for_dialogs)
            return

        ids_to_remove = set()
        for iid in selected:
            try:
                ids_to_remove.add(int(iid))
            except ValueError:
                continue

        before = len(self.model_grid_entries)
        self.model_grid_entries = [entry for entry in self.model_grid_entries if entry.get("id") not in ids_to_remove]
        removed = before - len(self.model_grid_entries)
        self._refresh_model_grid_tree()
        self.log(f"Se eliminaron {removed} escenario(s) del grid.", "INFO")

    def _remove_grid_entry_by_id(self, entry_id):
        if entry_id is None:
            return

        entry = self._find_grid_entry_by_id(entry_id)
        if not entry:
            return

        self.model_grid_entries = [item for item in self.model_grid_entries if item.get("id") != entry_id]
        self._refresh_model_grid_tree()
        entry_name = entry.get("name") or entry_id
        self.log(f"Escenario '{entry_name}' se eliminó del grid tras cargarse en el panel.", "INFO")

    def _get_grid_entry_display_values(self, entry, display_index):
        config = entry.get("config", {})
        covariables = config.get("selected_covariables", [])
        selection_method = config.get("var_selection_method", "")

        if not covariables:
            cov_text = "(auto/selección)"
        elif len(covariables) <= 6:
            cov_text = ", ".join(covariables)
        else:
            cov_text = f"{len(covariables)} covariables"

        if selection_method and selection_method != "Ninguno (usar todas)":
            cov_text += f" | Sel.: {selection_method.split()[0]}"

        pen_conf = config.get("penalization", {}) or {}
        pen_method = pen_conf.get("method", "Ninguna") or "Ninguna"
        if pen_method == "Ninguna":
            pen_text = "Ninguna"
        else:
            pen_text = pen_method
            pen_value = pen_conf.get("value")
            if pen_value is not None:
                pen_text += f" (λ={pen_value:.3g})"
            if pen_method == "ElasticNet":
                pen_text += f", α={pen_conf.get('l1_ratio', 0.5):.2f}"

        spline_configs = config.get("spline_configs", {}) or {}
        spline_text = "—"
        if spline_configs:
            details = []
            for var_name, detail in spline_configs.items():
                if not isinstance(detail, dict):
                    details.append(f"{var_name}: (sin detalles)")
                    continue

                type_internal = detail.get('type', '')
                display_label = self._get_spline_display_value(type_internal) if isinstance(type_internal, str) else 'Spline'
                if not display_label:
                    display_label = 'Spline'

                parts = [display_label]
                if detail.get('df') is not None:
                    parts.append(f"df={detail['df']}")
                if detail.get('num_knots') is not None:
                    parts.append(f"nudos={detail['num_knots']}")

                is_b_spline = isinstance(type_internal, str) and type_internal.lower() == 'b-spline'
                if is_b_spline and detail.get('degree') is not None:
                    parts.append(f"grado={detail['degree']}")

                details.append(f"{var_name}: {' '.join(parts)}")
            if details:
                spline_text = "; ".join(details)

        notes_text = entry.get("notes") or config.get("notes") or ""

        values = (
            display_index,
            entry.get("name", f"Escenario {display_index}"),
            config.get("model_type", ""),
            cov_text,
            pen_text,
            config.get("scaling_method", "Ninguna"),
            config.get("tie_method", "efron"),
            spline_text,
            notes_text
        )
        return values

    def _refresh_model_grid_tree(self):
        if not self.model_grid_tree:
            return

        selected_ids = set(self.model_grid_tree.selection())
        for iid in self.model_grid_tree.get_children():
            self.model_grid_tree.delete(iid)

        for display_index, entry in enumerate(self.model_grid_entries, start=1):
            iid = str(entry.get("id"))
            values = self._get_grid_entry_display_values(entry, display_index)
            self.model_grid_tree.insert("", tk.END, iid=iid, values=values)
            if iid in selected_ids:
                self.model_grid_tree.selection_add(iid)

    def _find_grid_entry_by_id(self, entry_id):
        for entry in self.model_grid_entries:
            if entry.get("id") == entry_id:
                return entry
        return None

    def _execute_grid_entries(self, entries):
        executed_count = 0
        for entry in entries:
            if not entry:
                continue
            try:
                applied_ok = self._apply_grid_entry_to_ui(entry)
                if not applied_ok:
                    continue
                self._execute_cox_modeling_orchestrator()
                executed_count += 1
            except Exception as exc:
                entry_name = entry.get("name", "Escenario")
                self.log(f"Error al ejecutar automáticamente el escenario '{entry_name}': {exc}", "ERROR")

        if executed_count > 0:
            self.log(f"Se ejecutaron {executed_count} escenario(s) desde el grid.", "INFO")

    def _on_grid_entry_double_click(self, event):
        if not self.model_grid_tree:
            return
        item_id = self.model_grid_tree.identify_row(event.y)
        if not item_id:
            return
        try:
            entry_id = int(item_id)
        except ValueError:
            return
        entry = self._find_grid_entry_by_id(entry_id)
        if entry:
            self._execute_grid_entries([entry])

    def _execute_selected_grid_entries(self):
        if not self.model_grid_tree:
            return
        selected = self.model_grid_tree.selection()
        if not selected:
            messagebox.showinfo("Grid", "Selecciona un escenario y vuelve a intentar.", parent=self.parent_for_dialogs)
            return
        entries_to_run = []
        for iid in selected:
            try:
                entry_id = int(iid)
            except ValueError:
                self.log(f"Identificador de escenario no válido en la selección: {iid}", "WARN")
                continue
            entry = self._find_grid_entry_by_id(entry_id)
            if entry:
                entries_to_run.append(entry)
            else:
                self.log(f"No se encontró el escenario con id {entry_id} al intentar ejecutarlo.", "WARN")

        if not entries_to_run:
            messagebox.showwarning("Grid", "No se encontraron escenarios válidos para ejecutar.", parent=self.parent_for_dialogs)
            return

        self._execute_grid_entries(entries_to_run)

    def _apply_grid_entry_to_ui(self, entry):
        config = entry.get("config", {})
        entry_id = entry.get("id")
        apply_success = False

        self.cox_model_type_var.set(config.get("model_type", "Multivariado"))
        self._toggle_univariate_forest_plot_cb()

        selection_method = config.get("var_selection_method", "Ninguno (usar todas)")
        self.var_selection_method_var.set(selection_method)
        if hasattr(self, 'combo_metodo_seleccion_vars'):
            available_methods = set(self.combo_metodo_seleccion_vars.cget("values"))
            if selection_method in available_methods:
                self.combo_metodo_seleccion_vars.set(selection_method)
            else:
                fallback = "Ninguno (usar todas)"
                self.combo_metodo_seleccion_vars.set(fallback)
                self.var_selection_method_var.set(fallback)

        self.p_enter_var.set(config.get("p_enter", 0.05))
        self.p_remove_var.set(config.get("p_remove", 0.10))

        pen_conf = config.get("penalization", {}) or {}
        pen_method = pen_conf.get("method", "Ninguna") or "Ninguna"
        self.penalization_method_var.set(pen_method)
        if hasattr(self, 'combo_tipo_penalizacion'):
            available_pen = set(self.combo_tipo_penalizacion.cget("values"))
            if pen_method in available_pen:
                self.combo_tipo_penalizacion.set(pen_method)
            else:
                self.combo_tipo_penalizacion.set("Ninguna")
                self.penalization_method_var.set("Ninguna")

        self.penalizer_strength_var.set(pen_conf.get("value", 0.0))
        self.l1_ratio_for_elasticnet_var.set(pen_conf.get("l1_ratio", 0.5))
        self._toggle_penalization_params_ui_state()

        scaling_method = config.get("scaling_method", "Ninguna")
        self.covariate_scaling_method_var.set(scaling_method)
        if hasattr(self, 'combo_scaling_method'):
            available_scaling = set(self.combo_scaling_method.cget("values"))
            if scaling_method in available_scaling:
                self.combo_scaling_method.set(scaling_method)
            else:
                self.combo_scaling_method.set("Ninguna")
                self.covariate_scaling_method_var.set("Ninguna")

        tie_method = config.get("tie_method", "efron")
        self.tie_handling_method_var.set(tie_method)
        if hasattr(self, 'combo_metodo_empates'):
            available_ties = set(self.combo_metodo_empates.cget("values"))
            if tie_method in available_ties:
                self.combo_metodo_empates.set(tie_method)
            else:
                self.combo_metodo_empates.set("efron")
                self.tie_handling_method_var.set("efron")

        self.calculate_cv_cindex_var.set(config.get("calculate_cv", bool(self.calculate_cv_cindex_var.get())))
        self.cv_num_kfolds_var.set(config.get("cv_kfolds", self.cv_num_kfolds_var.get()))
        self.cv_random_seed_var.set(config.get("cv_seed", self.cv_random_seed_var.get()))
        self.calculate_test_cindex_var.set(config.get("calculate_test_holdout", bool(self.calculate_test_cindex_var.get())))
        self.test_size_var.set(config.get("test_size", self.test_size_var.get()))
        self.test_random_seed_var.set(config.get("test_seed", self.test_random_seed_var.get()))

        self.generate_univariate_forest_plot_var.set(config.get("generate_univariate_forest_plot", True))
        self._toggle_univariate_forest_plot_cb()
        self._clear_holdout_config_dirty()

        selected_covs = config.get("selected_covariables", [])
        if hasattr(self, 'listbox_covariables_disponibles'):
            self.listbox_covariables_disponibles.selection_clear(0, tk.END)
            available_items = [self.listbox_covariables_disponibles.get(i) for i in range(self.listbox_covariables_disponibles.size())]
            for cov_name in selected_covs:
                if cov_name in available_items:
                    idx = available_items.index(cov_name)
                    self.listbox_covariables_disponibles.selection_set(idx)
                else:
                    self.log(f"Covariable '{cov_name}' no está disponible en la lista actual al aplicar escenario.", "WARN")

        self.categorical_compare_config = copy.deepcopy(config.get("categorical_configs", {}))
        self.spline_config_details = copy.deepcopy(config.get("spline_configs", {}))
        self.on_covariate_select_for_config()

        notes_text = entry.get("notes") or config.get("notes") or ""
        if notes_text:
            self.log(f"Escenario '{entry.get('name', 'Escenario')}' cargado. Nota: {notes_text}", "INFO")

        self.log(f"Escenario '{entry.get('name', 'Escenario')}' cargado en el panel de modelado.", "INFO")

        try:
            applied_ok = self.apply_covariate_config_to_selected()
            if applied_ok:
                self.log("Configuración aplicada automáticamente al cargar el escenario desde el grid.", "INFO")
                if entry_id is not None:
                    self._remove_grid_entry_by_id(entry_id)
                apply_success = True
            else:
                self.log("No se pudo aplicar automáticamente la configuración del escenario; revisa los detalles en la pestaña.", "WARN")
        except Exception as exc:
            self.log(f"Error al aplicar automáticamente la configuración del escenario: {exc}", "ERROR")
            apply_success = False

        return apply_success

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
        self.selected_covariables_from_ui = list(selected_covs_orig_names)

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
            # Convertir a numérico, forzando errores a NaN
            numeric_event_col = pd.to_numeric(df_model_prep[final_e_col], errors='coerce')

            # Comprobar si todos los valores son 0, 1, o NaN
            if not numeric_event_col.dropna().isin([0, 1]).all():
                self.log(f"Columna Evento '{final_e_col}' contiene valores que no son 0 o 1.", "ERROR")
                messagebox.showerror("Error de Tipo", f"La columna de Evento ('{final_e_col}') debe contener solo valores 0 y 1.", parent=self.parent_for_dialogs)
                return None, None, None, None, None, None, None, "Ninguna", None, []

            df_model_prep[final_e_col] = numeric_event_col
            df_model_prep.dropna(subset=[final_t_col, final_e_col], inplace=True)

            if df_model_prep.empty:
                self.log("Dataset vacío después de procesar y eliminar NaNs en columnas de Tiempo/Evento.", "ERROR")
                messagebox.showerror("Datos Insuficientes", "No quedan datos válidos después de procesar las columnas de Tiempo y Evento.", parent=self.parent_for_dialogs)
                return None, None, None, None, None, None, None, "Ninguna", None, []

            df_model_prep[final_e_col] = df_model_prep[final_e_col].astype(int)
        except Exception as e_e:
            self.log(f"Error procesando la columna de Evento '{final_e_col}': {e_e}", "ERROR")
            messagebox.showerror("Error de Tipo", f"Error al procesar la columna de Evento '{final_e_col}'. Asegúrese de que sea binaria (0/1).", parent=self.parent_for_dialogs)
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

    def _calculate_internal_knots_for_series(self, series, desired_internal_knots, var_name):
        """Compute internal knot positions for a numeric series using quantile spacing."""
        metadata = {
            "requested_internal_knots": int(desired_internal_knots),
            "used_internal_knots": 0,
            "internal_knots": [],
            "boundary_knots": [],
            "series_min": None,
            "series_max": None,
            "knot_strategy": "quantile"
        }

        if desired_internal_knots <= 0:
            numeric_series = pd.to_numeric(series, errors='coerce').dropna()
            if numeric_series.empty:
                self.log(f"No se calcularon nodos para '{var_name}' (serie vacía tras limpiar).", "DEBUG")
                return metadata
            unique_vals = np.unique(numeric_series)
            if unique_vals.size < 2:
                self.log(f"No se calcularon nodos para '{var_name}' (menos de dos valores únicos).", "WARN")
                return metadata
            metadata["boundary_knots"] = [float(unique_vals[0]), float(unique_vals[-1])]
            metadata["series_min"] = float(np.min(numeric_series))
            metadata["series_max"] = float(np.max(numeric_series))
            return metadata

        numeric_series = pd.to_numeric(series, errors='coerce').dropna()
        if numeric_series.empty:
            self.log(f"No se calcularon nodos para '{var_name}': serie vacía tras convertir a numérica.", "WARN")
            return metadata

        unique_vals = np.unique(numeric_series)
        if unique_vals.size < 2:
            self.log(f"No se calcularon nodos para '{var_name}': se requieren al menos dos valores únicos.", "WARN")
            metadata["boundary_knots"] = [float(unique_vals[0])] * 2
            metadata["series_min"] = float(unique_vals[0])
            metadata["series_max"] = float(unique_vals[0])
            return metadata

        metadata["boundary_knots"] = [float(unique_vals[0]), float(unique_vals[-1])]
        metadata["series_min"] = float(np.min(numeric_series))
        metadata["series_max"] = float(np.max(numeric_series))

        max_possible_internal = max(0, unique_vals.size - 2)
        if max_possible_internal <= 0:
            self.log(f"'{var_name}' no admite nodos internos adicionales (solo valores mínimo y máximo disponibles).", "WARN")
            return metadata

        used_internal_knots = int(min(desired_internal_knots, max_possible_internal))
        if used_internal_knots < desired_internal_knots:
            self.log(f"'{var_name}': nodos internos solicitados={desired_internal_knots}, máximos posibles={max_possible_internal}. Se ajusta a {used_internal_knots}.", "WARN")

        if used_internal_knots <= 0:
            return metadata

        quantile_positions = np.linspace(0, 1, used_internal_knots + 2)[1:-1]
        try:
            knots = np.quantile(numeric_series, quantile_positions)
        except Exception as err_quantile:
            self.log(f"Fallo al calcular quantiles para nodos de '{var_name}': {err_quantile}", "ERROR")
            return metadata

        deduped_knots = []
        for knot_val in np.atleast_1d(knots):
            knot_float = float(knot_val)
            if not deduped_knots or abs(knot_float - deduped_knots[-1]) > 1e-8:
                deduped_knots.append(knot_float)

        if len(deduped_knots) < used_internal_knots:
            self.log(f"'{var_name}': nodos internos únicos={len(deduped_knots)} menores al solicitado. Se usarán los disponibles.", "WARN")
            used_internal_knots = len(deduped_knots)

        metadata["internal_knots"] = deduped_knots
        metadata["used_internal_knots"] = used_internal_knots
        return metadata


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
        self._last_spline_basis_metadata = {}
        for orig_cov_name_bd in selected_covs_orig_names_bd:
            if orig_cov_name_bd not in df_for_patsy_bd.columns:
                self.log(f"Advertencia: Cov. original '{orig_cov_name_bd}' no en DF para Patsy. Saltando.", "WARN"); continue

            config_type_bd = self.covariables_type_config.get(orig_cov_name_bd, "Cuantitativa" if pd.api.types.is_numeric_dtype(df_for_patsy_bd[orig_cov_name_bd]) else "Cualitativa")
            
            term_syntax_bd = f"Q('{orig_cov_name_bd}')" 
            if config_type_bd == "Cuantitativa":
                if orig_cov_name_bd in self.spline_config_details:
                    spl_cfg_bd = self.spline_config_details[orig_cov_name_bd]
                    spline_type = spl_cfg_bd.get('type', 'Natural')
                    spline_df_requested_raw = spl_cfg_bd.get('df', 4)
                    spline_df = spline_df_requested_raw
                    spline_num_knots = max(0, spl_cfg_bd.get('num_knots', 0))
                    spline_degree_cfg = spl_cfg_bd.get('degree', 3)
                    spline_restricted = spl_cfg_bd.get('restricted', spline_type == 'Natural')
                    custom_knots_cfg = spl_cfg_bd.get('custom_knots') or []

                    if spline_df in (None, "", "auto"):
                        spline_df = self._derive_spline_df(
                            spline_type,
                            spline_degree_cfg,
                            spline_num_knots,
                            custom_knots_cfg
                        )

                    try:
                        numeric_values_for_df = pd.to_numeric(df_for_patsy_bd[orig_cov_name_bd], errors='coerce').dropna()
                    except Exception as err_numeric:
                        numeric_values_for_df = df_for_patsy_bd[orig_cov_name_bd].dropna()
                        self.log(f"'{orig_cov_name_bd}': no se pudo convertir a numérico para validar DF ({err_numeric}). Se usa la serie original.", "WARN")

                    unique_count_for_df = numeric_values_for_df.nunique()

                    try:
                        spline_df_adjusted = int(spline_df)
                    except (TypeError, ValueError):
                        derived_df_fallback = self._derive_spline_df(
                            spline_type,
                            spline_degree_cfg,
                            spline_num_knots,
                            custom_knots_cfg
                        )
                        self.log(
                            f"'{orig_cov_name_bd}': DF inválido '{spline_df}'. Se utiliza derivado={derived_df_fallback}.",
                            "WARN"
                        )
                        spline_df_adjusted = int(derived_df_fallback)
                        spline_df = spline_df_adjusted

                    if unique_count_for_df > 0 and spline_df_adjusted > unique_count_for_df:
                        self.log(f"'{orig_cov_name_bd}': df solicitado ({spline_df_adjusted}) excede valores únicos ({unique_count_for_df}). Ajustado a {unique_count_for_df}.", "WARN")
                        spline_df_adjusted = int(unique_count_for_df)

                    if spline_type == 'Natural':
                        if unique_count_for_df >= 3 and spline_df_adjusted < 3:
                            self.log(f"'{orig_cov_name_bd}': df {spline_df_adjusted} incrementado a 3 para spline restringido estable.", "WARN")
                            spline_df_adjusted = 3
                        elif unique_count_for_df < 3:
                            fallback_df = int(unique_count_for_df) if unique_count_for_df > 0 else 1
                            if spline_df_adjusted != fallback_df:
                                self.log(f"'{orig_cov_name_bd}': df ajustado a {fallback_df} por valores únicos limitados en spline restringido.", "WARN")
                            spline_df_adjusted = fallback_df
                    elif spline_type == 'B-spline':
                        effective_degree = max(1, int(spline_degree_cfg))
                        if unique_count_for_df > 0 and unique_count_for_df <= effective_degree:
                            adjusted_degree = max(1, unique_count_for_df - 1)
                            if adjusted_degree < effective_degree:
                                self.log(
                                    f"'{orig_cov_name_bd}': grado {effective_degree} excede valores únicos ({unique_count_for_df}). Ajustado a {adjusted_degree}.",
                                    "WARN"
                                )
                                effective_degree = adjusted_degree
                                spline_degree_cfg = effective_degree

                        explicit_zero_knots_bs_local = (spline_num_knots == 0 and not custom_knots_cfg)
                        if custom_knots_cfg:
                            min_df_bs = len(custom_knots_cfg) + effective_degree
                        elif explicit_zero_knots_bs_local:
                            min_df_bs = effective_degree
                        elif spline_num_knots > 0:
                            min_df_bs = spline_num_knots + effective_degree
                        else:
                            min_df_bs = effective_degree + 1

                        if unique_count_for_df > 0:
                            min_df_bs = min(max(min_df_bs, effective_degree), unique_count_for_df)

                        if spline_df_adjusted < min_df_bs:
                            self.log(
                                f"'{orig_cov_name_bd}': df {spline_df_adjusted} incrementado a {min_df_bs} para compatibilidad con B-spline grado {effective_degree}.",
                                "WARN"
                            )
                            spline_df_adjusted = int(min_df_bs)
                            spline_df = spline_df_adjusted

                    if spline_df_adjusted < 1:
                        spline_df_adjusted = 1

                    if spline_df_adjusted != spline_df:
                        self.log(f"'{orig_cov_name_bd}': df efectivo utilizado = {spline_df_adjusted}.", "INFO")
                        spline_df = spline_df_adjusted

                    internal_knots_to_calculate = spline_num_knots
                    knots_source_descriptor = "solicitados"
                    explicit_zero_knots = (spline_num_knots == 0 and not custom_knots_cfg)

                    if spline_type == 'Natural':
                        if internal_knots_to_calculate <= 0 and spline_df > 1:
                            internal_knots_to_calculate = max(0, int(spline_df) - 1)
                            knots_source_descriptor = "derivados_por_df"
                    elif spline_type == 'B-spline':
                        degree_for_bs = max(1, int(spline_degree_cfg))
                        if explicit_zero_knots:
                            internal_knots_to_calculate = 0
                            knots_source_descriptor = "cero_explicito"
                        elif internal_knots_to_calculate <= 0 and spline_df > degree_for_bs:
                            internal_knots_to_calculate = max(0, int(spline_df) - degree_for_bs)
                            knots_source_descriptor = "derivados_por_df"

                    manual_knots_clean = []
                    if custom_knots_cfg:
                        for knot_val in custom_knots_cfg:
                            try:
                                manual_knots_clean.append(float(knot_val))
                            except (TypeError, ValueError):
                                self.log(f"'{orig_cov_name_bd}': nodo manual '{knot_val}' inválido. Se ignora.", "WARN")
                        manual_knots_clean = sorted(set(manual_knots_clean))

                    if manual_knots_clean and spline_type not in {'B-spline', 'Natural'}:
                        self.log(f"'{orig_cov_name_bd}': nodos manuales proporcionados, pero el tipo de spline '{spline_type}' no los admite. Se ignorarán.", "WARN")
                        manual_knots_clean = []

                    if manual_knots_clean:
                        manual_min = float(numeric_values_for_df.min()) if not numeric_values_for_df.empty else None
                        manual_max = float(numeric_values_for_df.max()) if not numeric_values_for_df.empty else None
                        out_of_bounds = []
                        if manual_min is not None and manual_max is not None:
                            for knot_val in manual_knots_clean:
                                if knot_val < manual_min or knot_val > manual_max:
                                    out_of_bounds.append(knot_val)
                        if out_of_bounds:
                            self.log(f"'{orig_cov_name_bd}': nodos manuales fuera de rango ({out_of_bounds}). Revise que los knots correspondan al rango real de los datos.", "WARN")

                        knot_metadata = {
                            "requested_internal_knots": len(manual_knots_clean),
                            "used_internal_knots": len(manual_knots_clean),
                            "internal_knots": manual_knots_clean,
                            "boundary_knots": [manual_min, manual_max] if manual_min is not None and manual_max is not None else [],
                            "series_min": manual_min,
                            "series_max": manual_max,
                            "knot_strategy": "manual",
                            "internal_knots_source": "manual",
                            "spline_type": spline_type,
                            "df_requested": spline_df_requested_raw,
                            "df_applied": spline_df,
                            "restricted": spline_restricted,
                            "num_knots_requested": len(manual_knots_clean),
                            "num_knots_derived_from_df": None,
                            "custom_knots": manual_knots_clean
                        }
                    else:
                        knot_metadata = self._calculate_internal_knots_for_series(
                            df_for_patsy_bd[orig_cov_name_bd],
                            internal_knots_to_calculate,
                            orig_cov_name_bd
                        )
                        knot_metadata['internal_knots_source'] = knots_source_descriptor
                        if knots_source_descriptor == "derivados_por_df":
                            knot_metadata['num_knots_derived_from_df'] = internal_knots_to_calculate
                        else:
                            knot_metadata['num_knots_derived_from_df'] = None
                        knot_metadata['spline_type'] = spline_type
                        knot_metadata['df_requested'] = spline_df_requested_raw
                        knot_metadata['df_applied'] = spline_df
                        knot_metadata['restricted'] = spline_restricted
                        knot_metadata['num_knots_requested'] = spline_num_knots
                        knot_metadata['custom_knots'] = manual_knots_clean

                    if spline_type == 'Natural':
                        patsy_func_bd = 'cr'
                        knot_metadata['degree'] = 3
                        knot_metadata['num_knots_requested'] = spline_num_knots
                        if knot_metadata.get('internal_knots'):
                            knots_tuple = tuple(knot_metadata['internal_knots'])
                            term_syntax_bd = f"{patsy_func_bd}(Q('{orig_cov_name_bd}'), knots={knots_tuple})"
                            self.log(f"'{orig_cov_name_bd}' (Natural Spline): nodos internos usados={len(knot_metadata['internal_knots'])}, límites={knot_metadata.get('boundary_knots', [])}", "DEBUG")
                            knot_metadata['df_effective'] = len(knot_metadata['internal_knots']) + 1
                            knot_metadata['df_applied'] = knot_metadata['df_effective']
                        else:
                            term_syntax_bd = f"{patsy_func_bd}(Q('{orig_cov_name_bd}'), df={spline_df})"
                            knot_metadata['df_effective'] = spline_df
                            knot_metadata['df_applied'] = spline_df
                    elif spline_type == 'B-spline':
                        patsy_func_bd = 'bs'
                        spline_degree = max(1, int(spline_degree_cfg))
                        knot_metadata['degree'] = spline_degree
                        if manual_knots_clean:
                            knot_metadata['num_knots_requested'] = len(manual_knots_clean)
                        else:
                            knot_metadata['num_knots_requested'] = spline_num_knots

                        if explicit_zero_knots and not manual_knots_clean:
                            term_syntax_bd = self._build_degree_only_polynomial_formula(orig_cov_name_bd, spline_degree)
                            knot_metadata['df_effective'] = spline_degree
                            knot_metadata['df_applied'] = spline_degree
                            knot_metadata['basis_mode'] = 'polynomial_zero_knots'
                            knot_metadata['internal_knots'] = []
                            knot_metadata['used_internal_knots'] = 0
                            self.log(
                                f"'{orig_cov_name_bd}': B-spline con 0 nodos => polinomio grado {spline_degree} sin knots internos.",
                                "INFO"
                            )
                        elif knot_metadata.get('internal_knots'):
                            knots_tuple = tuple(knot_metadata['internal_knots'])
                            term_syntax_bd = f"{patsy_func_bd}(Q('{orig_cov_name_bd}'), knots={knots_tuple}, degree={spline_degree}, include_intercept=False)"
                            self.log(f"'{orig_cov_name_bd}' (B-spline grado {spline_degree}): nodos internos usados={len(knot_metadata['internal_knots'])}, límites={knot_metadata.get('boundary_knots', [])}", "DEBUG")
                            knot_metadata['df_effective'] = len(knot_metadata['internal_knots']) + spline_degree
                            knot_metadata['df_applied'] = knot_metadata['df_effective']
                            spline_df = knot_metadata['df_effective']
                        else:
                            term_syntax_bd = f"{patsy_func_bd}(Q('{orig_cov_name_bd}'), df={spline_df}, degree={spline_degree}, include_intercept=False)"
                            knot_metadata['df_effective'] = spline_df
                            knot_metadata['df_applied'] = spline_df
                    else: # Fallback for unknown spline type
                        term_syntax_bd = f"Q('{orig_cov_name_bd}')"
                        knot_metadata['spline_type'] = 'Ninguna'
                        self.log(f"WARN: Tipo de spline desconocido '{spline_type}' para '{orig_cov_name_bd}'. Tratada como cuantitativa normal.", "WARN")

                    knot_metadata['num_knots_used'] = knot_metadata.get('used_internal_knots', 0)
                    self._last_spline_basis_metadata[orig_cov_name_bd] = knot_metadata
                    cfg_entry_existing = self.spline_config_details.get(orig_cov_name_bd)
                    if cfg_entry_existing:
                        try:
                            if knot_metadata.get('df_applied') is not None:
                                cfg_entry_existing['df'] = int(knot_metadata['df_applied'])
                            if knot_metadata.get('num_knots_used') is not None:
                                cfg_entry_existing['num_knots'] = int(knot_metadata['num_knots_used'])
                            if spline_type == 'B-spline':
                                cfg_entry_existing['degree'] = spline_degree
                                cfg_entry_existing['custom_knots'] = manual_knots_clean if manual_knots_clean else []
                        except Exception as err_update_cfg:
                            self.log(f"No se pudo actualizar config de spline para '{orig_cov_name_bd}': {err_update_cfg}", "WARN")
                else: # No spline config for this quantitative var
                    term_syntax_bd = f"Q('{orig_cov_name_bd}')"
            else:
                source_series_cat = df_for_patsy_bd[orig_cov_name_bd]
                try:
                    df_for_patsy_bd[orig_cov_name_bd] = source_series_cat.where(
                        source_series_cat.isna(),
                        source_series_cat.astype(str)
                    ).astype(object)
                except Exception:
                    if not pd.api.types.is_categorical_dtype(df_for_patsy_bd[orig_cov_name_bd].dtype) and \
                       not pd.api.types.is_string_dtype(df_for_patsy_bd[orig_cov_name_bd].dtype) and \
                       not pd.api.types.is_object_dtype(df_for_patsy_bd[orig_cov_name_bd].dtype):
                        df_for_patsy_bd[orig_cov_name_bd] = df_for_patsy_bd[orig_cov_name_bd].astype(str)

                ref_cat_bd = self.ref_categories_config.get(orig_cov_name_bd)
                available_ref_values = self._get_reference_category_values(orig_cov_name_bd)
                ref_cat_str_bd = str(ref_cat_bd).strip() if ref_cat_bd not in (None, "") else ""
                if not ref_cat_str_bd and available_ref_values:
                    ref_cat_str_bd = str(available_ref_values[0]).strip()

                compare_cfg_bd = self._get_categorical_compare_config(orig_cov_name_bd, ref_cat_str_bd)
                compare_mode_bd = compare_cfg_bd.get("mode", "all")
                selected_compare_groups_bd = compare_cfg_bd.get("selected_groups", [])

                if compare_mode_bd == "selected":
                    if not ref_cat_str_bd:
                        self.log(
                            f"'{orig_cov_name_bd}': el modo categórico seleccionado requiere una categoría de referencia válida. Se usará la codificación estándar.",
                            "WARN"
                        )
                    elif not selected_compare_groups_bd:
                        self.log(
                            f"'{orig_cov_name_bd}': modo 'comparar solo contra grupos elegidos' sin grupos válidos. Se usa comparación contra todos.",
                            "WARN"
                        )
                    else:
                        allowed_groups_bd = {ref_cat_str_bd, *selected_compare_groups_bd}
                        series_as_str = df_for_patsy_bd[orig_cov_name_bd].apply(
                            lambda val: str(val).strip() if pd.notna(val) else np.nan
                        )
                        outside_mask = series_as_str.notna() & ~series_as_str.isin(list(allowed_groups_bd))
                        excluded_count = int(outside_mask.sum())
                        if excluded_count > 0:
                            df_for_patsy_bd.loc[outside_mask, orig_cov_name_bd] = np.nan
                        self.log(
                            f"'{orig_cov_name_bd}': comparación categórica limitada a {sorted(allowed_groups_bd)}; se excluirán {excluded_count} fila(s) con otras categorías.",
                            "INFO"
                        )
                elif compare_mode_bd == "one_vs_rest":
                    if not ref_cat_str_bd:
                        self.log(
                            f"'{orig_cov_name_bd}': no se pudo aplicar la dicotomización porque falta la categoría elegida de referencia.",
                            "WARN"
                        )
                    else:
                        series_as_str = df_for_patsy_bd[orig_cov_name_bd].apply(
                            lambda val: str(val).strip() if pd.notna(val) else np.nan
                        )
                        non_null_mask = series_as_str.notna()
                        df_for_patsy_bd.loc[non_null_mask, orig_cov_name_bd] = series_as_str.where(
                            series_as_str == ref_cat_str_bd,
                            other="RESTO"
                        )
                        self.log(
                            f"'{orig_cov_name_bd}': comparación dicotómica aplicada ({ref_cat_str_bd} vs RESTO).",
                            "INFO"
                        )

                if ref_cat_str_bd:
                    current_unique_vals = df_for_patsy_bd[orig_cov_name_bd].dropna().astype(str).unique()
                    if ref_cat_str_bd in current_unique_vals:
                        term_syntax_bd = f"C(Q('{orig_cov_name_bd}'), Treatment('{ref_cat_str_bd}'))"
                    else:
                        self.log(
                            f"Advertencia: Ref.Cat. '{ref_cat_str_bd}' para '{orig_cov_name_bd}' no está en los datos efectivos. Usando el default de Patsy.",
                            "WARN"
                        )
                        term_syntax_bd = f"C(Q('{orig_cov_name_bd}'))"
                else:
                    term_syntax_bd = f"C(Q('{orig_cov_name_bd}'))"
            formula_parts_bd.append(term_syntax_bd)

        formula_patsy_bd = " + ".join(formula_parts_bd) if formula_parts_bd else "1"
        self.log(f"Fórmula Patsy generada: {formula_patsy_bd}", "DEBUG")

        try:
            if df_for_patsy_bd.empty and formula_patsy_bd != "0": 
                 self.log("DF entrada Patsy vacío con fórmula no nula.", "ERROR"); return None,None,None,None
            
            X_design_bd = dmatrix(formula_patsy_bd, df_for_patsy_bd, return_type="dataframe")
            # IMPORTANTE: devolver el DataFrame transformado usado por Patsy para que el ajuste Cox
            # respete recodificaciones categóricas como 'elegida vs RESTO' o 'grupos elegidos'.
            df_filtered_by_patsy_idx_bd = df_for_patsy_bd.loc[X_design_bd.index].copy()
            
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

        if terms_initial_vs is None:
            terms_initial_vs = []

        def infer_original_var_from_term(term_text):
            if not term_text:
                return None
            if ':' in term_text:
                inferred = [infer_original_var_from_term(part.strip()) for part in term_text.split(':')]
                inferred = [value for value in inferred if value]
                if not inferred:
                    return None
                if len(set(inferred)) == 1:
                    return inferred[0]
                return ':'.join(sorted(set(inferred)))
            match_q = re.search(r"Q\('([^']+)'\)", term_text)
            if match_q:
                return match_q.group(1)
            match_c = re.search(r"C\('([^']+)'\)", term_text)
            if match_c:
                return match_c.group(1)
            return None

        all_initial_orig_covs = []
        for term in terms_initial_vs:
            original_name = infer_original_var_from_term(str(term))
            if original_name and original_name not in all_initial_orig_covs:
                all_initial_orig_covs.append(original_name)

        if not all_initial_orig_covs:
            self.log("Selección Variables: no se detectaron covariables en la matriz inicial.", "WARN")
            return []

        if method_vs == "Ninguno (usar todas)":
            self.log("Selección Variables: 'Ninguno (usar todas)'. Usando todas las covariables iniciales.", "INFO")
            return all_initial_orig_covs

        p_enter_value = self._coerce_float_value(self.p_enter_var, 0.05, "P-valor de entrada", min_value=0.0)
        p_remove_value = self._coerce_float_value(self.p_remove_var, 0.10, "P-valor de salida", min_value=0.0)

        fit_successful = False

        def attempt_selection_fit(candidate_covariates):
            nonlocal fit_successful
            unique_covs = [cov for cov in dict.fromkeys(candidate_covariates) if cov]
            if not unique_covs:
                return None

            design_res = self.build_design_matrix(df_aligned_orig_vs, unique_covs, time_col_vs, event_col_vs)
            if not design_res or design_res[0] is None or design_res[0].empty:
                return None

            df_candidate, _, formula_candidate, _ = design_res
            try:
                selector_cph = CoxPHFitter(penalizer=0.0)
                selector_cph.fit(df_candidate, duration_col=time_col_vs, event_col=event_col_vs, formula=formula_candidate)
            except Exception as exc_sel:
                self.log(f"Selección '{method_vs}': fallo al ajustar con covariables {unique_covs}: {exc_sel}", "WARN")
                return None

            fit_successful = True
            return selector_cph.summary.copy()

        def aggregate_p_values(summary_df):
            if summary_df is None or summary_df.empty:
                return {}
            aggregated = {}
            for term_name, row in summary_df.iterrows():
                cov_key = infer_original_var_from_term(str(term_name))
                if not cov_key:
                    continue
                try:
                    p_val = float(row.get('p'))
                except (TypeError, ValueError):
                    continue
                if not np.isfinite(p_val):
                    continue
                aggregated.setdefault(cov_key, []).append(p_val)
            return {cov: max(values) for cov, values in aggregated.items() if values}

        selected_covariables = []

        if method_vs == "Backward":
            current_covs = list(all_initial_orig_covs)
            max_iterations = max(10, len(current_covs) * 5)
            iterations = 0
            while current_covs and iterations < max_iterations:
                iterations += 1
                summary_df = attempt_selection_fit(current_covs)
                if summary_df is None:
                    self.log("Backward: no se pudo ajustar el modelo con el conjunto actual; se detiene la selección.", "WARN")
                    break
                p_values_map = aggregate_p_values(summary_df)
                if not p_values_map:
                    break
                worst_var, worst_p = max(p_values_map.items(), key=lambda item: item[1])
                if worst_p > p_remove_value and worst_var in current_covs:
                    current_covs.remove(worst_var)
                    self.log(f"Backward: se remueve '{worst_var}' (p={worst_p:.4g} > {p_remove_value:.4g}).", "INFO")
                    continue
                break
            if iterations >= max_iterations:
                self.log("Backward: se alcanzó el límite de iteraciones durante la selección.", "WARN")
            selected_covariables = current_covs

        elif method_vs == "Forward":
            remaining_covs = list(all_initial_orig_covs)
            selected_covs = []
            max_iterations = max(10, len(remaining_covs) * 5)
            iterations = 0
            while remaining_covs and iterations < max_iterations:
                iterations += 1
                best_candidate = None
                best_candidate_p = None
                for candidate in list(remaining_covs):
                    summary_df = attempt_selection_fit(selected_covs + [candidate])
                    if summary_df is None:
                        continue
                    p_values_map = aggregate_p_values(summary_df)
                    candidate_p = p_values_map.get(candidate)
                    if candidate_p is None:
                        continue
                    if candidate_p <= p_enter_value and (best_candidate is None or candidate_p < best_candidate_p):
                        best_candidate = candidate
                        best_candidate_p = candidate_p
                if best_candidate is None:
                    break
                selected_covs.append(best_candidate)
                remaining_covs.remove(best_candidate)
                self.log(f"Forward: se añade '{best_candidate}' (p={best_candidate_p:.4g} ≤ {p_enter_value:.4g}).", "INFO")
            if iterations >= max_iterations:
                self.log("Forward: se alcanzó el límite de iteraciones durante la selección.", "WARN")
            selected_covariables = selected_covs

        elif method_vs == "Stepwise (Fwd luego Bwd)":
            remaining_covs = list(all_initial_orig_covs)
            selected_covs = []
            max_iterations = max(10, len(remaining_covs) * 6)
            iterations = 0
            while remaining_covs and iterations < max_iterations:
                iterations += 1
                best_candidate = None
                best_candidate_p = None
                cached_summary = None
                for candidate in list(remaining_covs):
                    summary_df = attempt_selection_fit(selected_covs + [candidate])
                    if summary_df is None:
                        continue
                    p_values_map = aggregate_p_values(summary_df)
                    candidate_p = p_values_map.get(candidate)
                    if candidate_p is None:
                        continue
                    if candidate_p <= p_enter_value and (best_candidate is None or candidate_p < best_candidate_p):
                        best_candidate = candidate
                        best_candidate_p = candidate_p
                        cached_summary = summary_df
                if best_candidate is None:
                    break
                selected_covs.append(best_candidate)
                remaining_covs.remove(best_candidate)
                self.log(f"Stepwise: se añade '{best_candidate}' (p={best_candidate_p:.4g} ≤ {p_enter_value:.4g}).", "INFO")

                changes = True
                while changes and selected_covs:
                    changes = False
                    summary_for_removal = cached_summary
                    if summary_for_removal is None or aggregate_p_values(summary_for_removal).keys() != set(selected_covs):
                        summary_for_removal = attempt_selection_fit(selected_covs)
                    if summary_for_removal is None:
                        break
                    p_values_map = aggregate_p_values(summary_for_removal)
                    if not p_values_map:
                        break
                    worst_var, worst_p = max(p_values_map.items(), key=lambda item: item[1])
                    if worst_p > p_remove_value and worst_var in selected_covs:
                        selected_covs.remove(worst_var)
                        if worst_var not in remaining_covs:
                            remaining_covs.append(worst_var)
                        self.log(f"Stepwise: se remueve '{worst_var}' (p={worst_p:.4g} > {p_remove_value:.4g}).", "INFO")
                        cached_summary = None
                        changes = True
            if iterations >= max_iterations:
                self.log("Stepwise: se alcanzó el límite de iteraciones durante la selección.", "WARN")
            selected_covariables = selected_covs

        else:
            self.log(f"Método de selección desconocido: {method_vs}. Usando todas las covariables iniciales.", "ERROR")
            return all_initial_orig_covs

        selected_covariables = list(dict.fromkeys(selected_covariables))

        if not selected_covariables:
            if not fit_successful and all_initial_orig_covs:
                self.log(f"Selección ({method_vs}): no se pudo ajustar ningún modelo. Se usarán todas las covariables originales.", "WARN")
                return all_initial_orig_covs
            self.log(f"Selección ({method_vs}): sin covariables que cumplan los umbrales; el modelo será nulo.", "WARN")
        else:
            self.log(f"Selección ({method_vs}): covariables finales {selected_covariables}", "INFO")

        return selected_covariables

    def _run_model_and_get_metrics(self, df_lifelines_rm, X_design_rm, y_survival_rm,
                                   time_col_rm, event_col_rm,
                                   formula_patsy_rm, model_name_rm,
                                   covariates_display_terms_rm,
                                   full_patsy_formula_for_new_data_transform_arg, 
                                   penalizer_val_rm=0.0, l1_ratio_val_rm=0.0, 
                                   model_type_for_fit_logic="Multivariado",
                                   scaling_method_applied="Ninguna",
                                   fitted_scaler_obj=None,
                                   scaled_columns_info=None,
                                   selected_covariates_original=None):
        self.log(f"Ajustando modelo Cox: '{model_name_rm}'...", "INFO")
        
        ui_selected_tie_method = self.tie_handling_method_var.get() # Para registro
        selected_covariates_original = list(dict.fromkeys(selected_covariates_original or []))
        ui_config_snapshot = {
            "model_type": model_type_for_fit_logic,
            "test_size": float(self.test_size_var.get()) if hasattr(self, 'test_size_var') else None,
            "stratify_holdout": bool(self.stratify_holdout_var.get()) if hasattr(self, 'stratify_holdout_var') else None,
            "tau_mode": self.tau_mode_var.get() if hasattr(self, 'tau_mode_var') else None,
            "tau_manual": self.tau_manual_var.get() if hasattr(self, 'tau_manual_var') else None,
        }
        
        model_data_rm = {
            "model_name": model_name_rm, "time_col_for_model": time_col_rm, "event_col_for_model": event_col_rm,
            "formula_patsy": formula_patsy_rm, 
            "full_patsy_formula_for_new_data_transform": full_patsy_formula_for_new_data_transform_arg, 
            "covariates_processed": covariates_display_terms_rm,
            "selected_covariables_original": selected_covariates_original,
            "ui_config_snapshot": ui_config_snapshot,
            "df_used_for_fit": self.data.copy(),
            "X_design_used_for_fit": X_design_rm.copy(),
            "y_survival_used_for_fit": y_survival_rm.copy(),
            "penalizer_value": penalizer_val_rm, "l1_ratio_value": l1_ratio_val_rm,
            "tie_method_used": ui_selected_tie_method,
            "metrics": {}, "schoenfeld_results": pd.DataFrame(), "model": None, "loglik_null": None,
            "c_index_cv_mean": None, "c_index_cv_std": None, "c_index_cv_ci": None,
            "c_index_test": None, "c_index_test_ci": None, "c_index_train_ci": None,
            "test_proportion": None, "c_index_gap": None,
            "schoenfeld_status_message": "Test de Schoenfeld no ejecutado o no aplicable inicialmente.",
            "proportional_hazard_test_summary": None,
            "oos_predictions": None,
            "scaling_method_applied": scaling_method_applied,
            "fitted_scaler_object": fitted_scaler_obj,
            "scaled_columns_info": scaled_columns_info if scaled_columns_info is not None else [],
            "custom_model_name": model_name_rm, # Inicializar con el nombre generado
            "custom_model_notes": "", # Inicializar notas vacías
            "design_info": None, # Placeholder for design_info
            "spline_basis_metadata": {}
        }

        if hasattr(X_design_rm, "design_info"):
            model_data_rm["design_info"] = X_design_rm.design_info

        if hasattr(self, "_last_spline_basis_metadata"):
            try:
                model_data_rm["spline_basis_metadata"] = copy.deepcopy(self._last_spline_basis_metadata)
            except Exception as err_copy_spline:
                self.log(f"No se pudo copiar metadata de splines: {err_copy_spline}", "WARN")

        try:
            model_data_rm["spline_config_details"] = copy.deepcopy(self.spline_config_details)
            model_data_rm["categorical_compare_config"] = copy.deepcopy(self.categorical_compare_config)
            model_data_rm["ref_categories_config"] = copy.deepcopy(self.ref_categories_config)
        except Exception as err_copy_cfg:
            self.log(f"No se pudo copiar configuración de splines/categóricas: {err_copy_cfg}", "WARN")

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

                # Check for low variance in the design matrix
                if X_design_rm is not None and not X_design_rm.empty:
                    low_variance_cols = [col for col in X_design_rm.columns if X_design_rm[col].var() < 1e-5]
                    if low_variance_cols:
                        self.log("  POSIBLE CAUSA: Se detectaron variables con varianza muy baja (casi constantes) en la matriz de diseño:", "WARN")
                        self.log(f"    - {', '.join(low_variance_cols)}", "WARN")
                        self.log("    - Esto puede causar inestabilidad numérica. Considere revisar o eliminar estas variables.", "WARN")

                if "cr(" in actual_formula_for_fit:
                    self.log("  ADVERTENCIA ADICIONAL: El modelo incluía splines naturales (cr()). Estos pueden ser numéricamente inestables. Considere usar B-splines (bs()) o reducir los grados de libertad (df).", "WARN")

                messagebox.showerror("Error de Convergencia",
                                     f"El modelo '{model_name_rm}' no pudo converger.\n\n"
                                     "Posibles Causas:\n"
                                     "1. Colinealidad Alta: Dos o más variables están altamente correlacionadas.\n"
                                     "2. Varianza Cero/Baja: Una variable tiene el mismo valor (o casi) para todos los sujetos.\n"
                                     "3. Separación de Datos: Una variable predice perfectamente el resultado en un subgrupo.\n\n"
                                     "Sugerencias:\n"
                                     "- Use el 'Diagnóstico de Colinealidad' en el modelo anterior si es posible.\n"
                                     "- Revise las variables en el modelo, especialmente las categóricas con pocos casos en algún nivel.\n"
                                     "- Si usa splines, intente con menos grados de libertad (df) o use B-splines en lugar de Natural.\n"
                                     "- Considere aplicar regularización (L1 o L2).",
                                     parent=self.parent_for_dialogs)
                traceback.print_exc(limit=2)
            except np.linalg.LinAlgError as e_linalg:
                num_obs_fail = df_for_fit_main.shape[0]
                num_events_fail = df_for_fit_main[event_col_rm].sum() if event_col_rm in df_for_fit_main.columns else 'N/A'
                self.log(f"FALLO DE AJUSTE DEL MODELO (LinAlgError - ej. Matriz Singular): '{model_name_rm}'", "ERROR")
                self.log(f"  Error específico: {e_linalg}", "ERROR")
                self.log(f"  Observaciones usadas: {num_obs_fail}, Eventos: {num_events_fail}", "ERROR")

                if X_design_rm is not None and not X_design_rm.empty:
                    low_variance_cols = [col for col in X_design_rm.columns if X_design_rm[col].var() < 1e-5]
                    if low_variance_cols:
                        self.log("  POSIBLE CAUSA: Se detectaron variables con varianza muy baja (casi constantes) en la matriz de diseño:", "WARN")
                        self.log(f"    - {', '.join(low_variance_cols)}", "WARN")

                if "cr(" in actual_formula_for_fit:
                    self.log("  ADVERTENCIA ADICIONAL: El modelo incluía splines naturales (cr()). Estos pueden causar problemas de colinealidad. Considere usar B-splines (bs()) o reducir los grados de libertad (df).", "WARN")

                messagebox.showerror("Error de Álgebra Lineal (Matriz Singular)",
                                     f"El modelo '{model_name_rm}' falló debido a un problema numérico (a menudo una 'matriz singular').\n\n"
                                     "Esto es frecuentemente causado por colinealidad perfecta o cuasi-perfecta.\n\n"
                                     "Sugerencias:\n"
                                     "- Revise si una variable categórica tiene un nivel con cero (o muy pocos) eventos.\n"
                                     "- Verifique si una variable es una combinación lineal de otras (ej: var3 = var1 + var2).\n"
                                     "- Use el 'Diagnóstico de Colinealidad' para investigar.",
                                     parent=self.parent_for_dialogs)
                traceback.print_exc(limit=2)
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

        # ── Determine diagnostics scope ──────────────────────────────────
        # When holdout is active, ALL reported metrics (AIC, LogLik, Wald,
        # Schoenfeld, C-Index Train, CV) come from a model trained *only*
        # on the training subset.  The full-data model stored in
        # model_data_rm["model"] is kept for predictions / plots.
        _holdout_active = False
        model_for_diagnostics = fitted_cph_model
        df_for_diagnostics = df_for_fit_main
        y_for_diagnostics = y_survival_rm

        if (fitted_cph_model
                and not X_design_rm.empty
                and getattr(self, 'calculate_test_cindex_var', None) is not None
                and self.calculate_test_cindex_var.get()):
            try:
                requested_test_size_holdout = float(self.test_size_var.get())
                test_seed_holdout = int(self.test_random_seed_var.get())
                min_train_rows_holdout = max(8, int(X_design_rm.shape[1]) + 2)
                prefer_strat_holdout = getattr(self, 'stratify_holdout_var', None)
                prefer_strat_holdout = prefer_strat_holdout.get() if prefer_strat_holdout is not None else True
                test_size_holdout, stratify_holdout, holdout_warnings = self._resolve_holdout_split_settings(
                    df_lifelines_rm, event_col_rm, requested_test_size_holdout,
                    min_train_rows=min_train_rows_holdout, min_test_rows=2,
                    prefer_stratify=prefer_strat_holdout, context_label=model_name_rm,
                )
                for hw in holdout_warnings:
                    self.log(f"{model_name_rm}: {hw}", "WARN")

                idx_train_holdout, idx_test_holdout = train_test_split(
                    df_lifelines_rm.index,
                    test_size=test_size_holdout,
                    random_state=test_seed_holdout,
                    stratify=stratify_holdout,
                )
                df_train_holdout = df_lifelines_rm.loc[idx_train_holdout].copy()
                df_test_holdout = df_lifelines_rm.loc[idx_test_holdout].copy()
                y_train_holdout = y_survival_rm.loc[idx_train_holdout]
                y_test_holdout = y_survival_rm.loc[idx_test_holdout]

                if not df_train_holdout.empty and not df_test_holdout.empty:
                    cph_holdout = CoxPHFitter(penalizer=penalizer_val_rm, l1_ratio=l1_ratio_val_rm)
                    cph_holdout.fit(df_train_holdout, duration_col=time_col_rm,
                                   event_col=event_col_rm, formula=actual_formula_for_fit)
                    preds_test_ho = cph_holdout.predict_partial_hazard(df_test_holdout)
                    preds_train_ho = cph_holdout.predict_partial_hazard(df_train_holdout)

                    c_test_ho = concordance_index(y_test_holdout[time_col_rm], -preds_test_ho, y_test_holdout[event_col_rm])
                    c_train_ho = concordance_index(y_train_holdout[time_col_rm], -preds_train_ho, y_train_holdout[event_col_rm])

                    # Only store holdout results after full success
                    model_data_rm["test_proportion"] = float(test_size_holdout)
                    model_data_rm["c_index_test"] = float(c_test_ho)
                    model_data_rm["c_index_gap"] = float(c_test_ho - c_train_ho)
                    model_data_rm["c_index_train_ci"] = bootstrap_concordance_ci_from_scores(
                        y_train_holdout,
                        time_col_rm,
                        event_col_rm,
                        -np.asarray(preds_train_ho, dtype=float).reshape(-1),
                    )
                    model_data_rm["c_index_test_ci"] = bootstrap_concordance_ci_from_scores(
                        y_test_holdout,
                        time_col_rm,
                        event_col_rm,
                        -np.asarray(preds_test_ho, dtype=float).reshape(-1),
                    )
                    # --- Uno's C-index (IPCW) + Antolini Ctd ---
                    c_index_uno_ho = None
                    c_index_antolini_ho = None
                    resolved_tau_ho = None
                    if callable(_concordance_index_ipcw):
                        try:
                            y_train_struct = np.array(
                                [(bool(e), float(t)) for e, t in zip(y_train_holdout[event_col_rm], y_train_holdout[time_col_rm])],
                                dtype=[('event', bool), ('time', float)],
                            )
                            y_test_struct = np.array(
                                [(bool(e), float(t)) for e, t in zip(y_test_holdout[event_col_rm], y_test_holdout[time_col_rm])],
                                dtype=[('event', bool), ('time', float)],
                            )
                            resolved_tau_ho = self._resolve_tau(y_train_holdout[time_col_rm], y_train_holdout[event_col_rm])
                            risk_scores_ho = np.asarray(preds_test_ho, dtype=float).reshape(-1)
                            ipcw_result = _concordance_index_ipcw(y_train_struct, y_test_struct,
                                                                  risk_scores_ho, tau=resolved_tau_ho)
                            c_index_uno_ho = float(np.asarray(ipcw_result).reshape(-1)[0])
                        except Exception:
                            c_index_uno_ho = None

                        # Antolini's Ctd
                        if callable(_cumulative_dynamic_auc) and y_train_struct is not None:
                            try:
                                eval_grid_ho = self._build_evaluation_time_grid_cox(
                                    y_train_holdout[time_col_rm], y_test_holdout[time_col_rm], tau=resolved_tau_ho)
                                if eval_grid_ho is not None:
                                    _, mean_auc = _cumulative_dynamic_auc(
                                        y_train_struct, y_test_struct, risk_scores_ho, eval_grid_ho)
                                    c_index_antolini_ho = float(mean_auc)
                            except Exception:
                                c_index_antolini_ho = None

                    model_data_rm["c_index_uno"] = c_index_uno_ho
                    model_data_rm["c_index_antolini"] = c_index_antolini_ho
                    model_data_rm["tau"] = resolved_tau_ho

                    # ── IBS + Brier/AUROC/C at quartiles ───────────────────
                    ibs_ho = None; brier_q25_ho = None; brier_q50_ho = None; brier_q75_ho = None
                    auroc_q25_ho = None; auroc_q50_ho = None; auroc_q75_ho = None
                    c_q25_ho = None; c_q50_ho = None; c_q75_ho = None
                    time_q25_ho = None; time_q50_ho = None; time_q75_ho = None
                    brier_curve_df_ho = None; brier_eval_time_ho = None
                    if y_train_struct is not None and y_test_struct is not None and eval_grid_ho is not None:
                        # Survival function matrix from lifelines
                        try:
                            surv_df_ho = cph_holdout.predict_survival_function(df_test_holdout, times=eval_grid_ho)
                            surv_matrix_ho = surv_df_ho.values.T  # (n_test, n_times)

                            if callable(_brier_score_cox) and callable(_integrated_brier_score_cox):
                                _, brier_vals_ho = _brier_score_cox(y_train_struct, y_test_struct, surv_matrix_ho, eval_grid_ho)
                                brier_vals_ho = np.asarray(brier_vals_ho, dtype=float)
                                brier_curve_df_ho = pd.DataFrame({"time": np.asarray(eval_grid_ho, dtype=float), "brier_score": brier_vals_ho})
                                if len(eval_grid_ho) > 0:
                                    brier_eval_time_ho = float(np.asarray(eval_grid_ho, dtype=float)[min(len(eval_grid_ho) - 1, len(eval_grid_ho) // 2)])
                                ibs_ho = float(_integrated_brier_score_cox(y_train_struct, y_test_struct, surv_matrix_ho, eval_grid_ho))

                            # Event time quartiles
                            ev_mask_ho = y_test_struct["event"].astype(bool)
                            ev_times_ho = y_test_struct["time"][ev_mask_ho]
                            if ev_times_ho.size >= 4:
                                time_q25_ho = float(np.percentile(ev_times_ho, 25))
                                time_q50_ho = float(np.percentile(ev_times_ho, 50))
                                time_q75_ho = float(np.percentile(ev_times_ho, 75))
                                q_times_ho = np.array([time_q25_ho, time_q50_ho, time_q75_ho])

                                # Brier at quartiles
                                try:
                                    surv_q_ho = cph_holdout.predict_survival_function(df_test_holdout, times=q_times_ho).values.T
                                    _, brier_qv = _brier_score_cox(y_train_struct, y_test_struct, surv_q_ho, q_times_ho)
                                    brier_qv = np.asarray(brier_qv, dtype=float)
                                    brier_q25_ho = float(brier_qv[0]); brier_q50_ho = float(brier_qv[1]); brier_q75_ho = float(brier_qv[2])
                                except Exception:
                                    pass

                                # AUROC at quartiles
                                if callable(_cumulative_dynamic_auc):
                                    try:
                                        auc_vals_ho, _ = _cumulative_dynamic_auc(y_train_struct, y_test_struct, risk_scores_ho, q_times_ho)
                                        auc_vals_ho = np.asarray(auc_vals_ho, dtype=float)
                                        auroc_q25_ho = float(auc_vals_ho[0]); auroc_q50_ho = float(auc_vals_ho[1]); auroc_q75_ho = float(auc_vals_ho[2])
                                    except Exception:
                                        pass

                                # C at quartiles (IPCW with tau=quartile)
                                for tau_q, target in [(time_q25_ho, 'q25'), (time_q50_ho, 'q50'), (time_q75_ho, 'q75')]:
                                    try:
                                        r = _concordance_index_ipcw(y_train_struct, y_test_struct, risk_scores_ho, tau=tau_q)
                                        v = float(np.asarray(r).reshape(-1)[0])
                                        if target == 'q25': c_q25_ho = v
                                        elif target == 'q50': c_q50_ho = v
                                        else: c_q75_ho = v
                                    except Exception:
                                        pass
                        except Exception as e_qm:
                            self.log(f"Métricas cuantiles holdout: {e_qm}", "WARN")

                    model_data_rm["ibs"] = ibs_ho
                    model_data_rm["brier_curve_df"] = brier_curve_df_ho
                    model_data_rm["brier_eval_time"] = brier_eval_time_ho
                    model_data_rm["brier_q25"] = brier_q25_ho; model_data_rm["brier_q50"] = brier_q50_ho; model_data_rm["brier_q75"] = brier_q75_ho
                    model_data_rm["auroc_q25"] = auroc_q25_ho; model_data_rm["auroc_q50"] = auroc_q50_ho; model_data_rm["auroc_q75"] = auroc_q75_ho
                    model_data_rm["c_harrell_q25"] = c_q25_ho; model_data_rm["c_harrell_q50"] = c_q50_ho; model_data_rm["c_harrell_q75"] = c_q75_ho
                    model_data_rm["time_q25"] = time_q25_ho; model_data_rm["time_q50"] = time_q50_ho; model_data_rm["time_q75"] = time_q75_ho

                    self.log(
                        f"C-Index holdout '{model_name_rm}': Train={c_train_ho:.3f}, "
                        f"Test={c_test_ho:.3f}, Δ={c_test_ho - c_train_ho:.3f}"
                        f"{f', Uno={c_index_uno_ho:.3f}' if c_index_uno_ho is not None else ''}"
                        f"{f', IBS={ibs_ho:.4f}' if ibs_ho is not None else ''}", "INFO")

                    # All diagnostics now use the holdout model
                    model_for_diagnostics = cph_holdout
                    df_for_diagnostics = df_train_holdout
                    y_for_diagnostics = y_train_holdout
                    _holdout_active = True

                    # Recompute null-model LogLik on training data
                    try:
                        cph_null_ho = CoxPHFitter(penalizer=0.0)
                        cph_null_ho.fit(
                            df_train_holdout[[time_col_rm, event_col_rm]].copy(),
                            duration_col=time_col_rm, event_col=event_col_rm, formula="0")
                        model_data_rm["loglik_null"] = cph_null_ho.log_likelihood_
                    except Exception as e_null_ho:
                        self.log(f"Error modelo nulo holdout: {e_null_ho}", "WARN")

                    self.log(
                        f"Holdout activo: métricas (AIC, Wald, Schoenfeld, CV) provienen "
                        f"del modelo entrenado en {len(df_train_holdout)} obs.", "INFO")
            except Exception as e_holdout_setup:
                self.log(f"Error configurando holdout: {e_holdout_setup}\n{traceback.format_exc(limit=3)}", "ERROR")

        if fitted_cph_model:
            # Test de Schoenfeld
            if not X_design_rm.empty:
                if hasattr(model_for_diagnostics, 'params_') and model_for_diagnostics.params_ is not None and not model_for_diagnostics.params_.empty:
                    self.log(f"--- Iniciando Test de Schoenfeld para Modelo: '{model_name_rm}' ---", "INFO")
                    try:
                        results_check_assumptions = model_for_diagnostics.check_assumptions(df_for_diagnostics)
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
                            model_data_rm["schoenfeld_results"] = normalize_schoenfeld_dataframe(schoenfeld_df_candidate)
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
            if hasattr(model_for_diagnostics, 'params_') and model_for_diagnostics.params_ is not None and not model_for_diagnostics.params_.empty:
                if schoenfeld_df_from_check_assumptions is None or schoenfeld_df_from_check_assumptions.empty or 'p' not in schoenfeld_df_from_check_assumptions.columns:
                    should_try_ph_test = True

            if should_try_ph_test:
                self.log(f"INFO: `schoenfeld_results` de `check_assumptions` para '{model_name_rm}' está vacío o no tiene columna 'p'. Intentando `proportional_hazard_test` como fuente alternativa/suplementaria.", "INFO")
                try:
                    from lifelines.statistics import proportional_hazard_test
                    ph_test_results_obj = proportional_hazard_test(model_for_diagnostics, df_for_diagnostics, time_transform='log')
                    if ph_test_results_obj is not None and hasattr(ph_test_results_obj, 'summary') and \
                       isinstance(ph_test_results_obj.summary, pd.DataFrame) and not ph_test_results_obj.summary.empty:

                        ph_summary_normalized = normalize_schoenfeld_dataframe(ph_test_results_obj.summary)
                        # Store the summary from proportional_hazard_test
                        model_data_rm["proportional_hazard_test_summary"] = ph_summary_normalized
                        if 'p' in ph_summary_normalized.columns:
                            self.log(f"INFO: `proportional_hazard_test` para '{model_name_rm}' proporcionó un resumen con p-valores.", "INFO")
                        else:
                            self.log(f"INFO: `proportional_hazard_test` para '{model_name_rm}' no incluye columna 'p' explícita tras normalización.", "INFO")

                        # If original schoenfeld_results was empty/missing 'p', replace it with this summary
                        if schoenfeld_df_from_check_assumptions is None or schoenfeld_df_from_check_assumptions.empty or 'p' not in schoenfeld_df_from_check_assumptions.columns:
                            model_data_rm["schoenfeld_results"] = ph_summary_normalized
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

                        if var_name in df_for_diagnostics.columns:
                            data_series = df_for_diagnostics[var_name].dropna()
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
                            self.log(f"CV C-Index: Variable '{var_name}' for B-spline not found in df_for_diagnostics. Cannot calculate knots.", "WARN")

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

                    for i_fold, (train_idx, test_idx) in enumerate(kf_cv.split(df_for_diagnostics)):
                        self.log(f"CV C-Index Fold {i_fold+1}/{kf_cv.get_n_splits()}: Processing...", "DEBUG")
                        df_fold_for_fit_cv = df_for_diagnostics.iloc[train_idx].copy()
                        df_fold_for_pred_cv = df_for_diagnostics.iloc[test_idx].copy() # Data for prediction
                        y_te_cv = y_for_diagnostics.iloc[test_idx] # True outcomes for test set

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
                        model_data_rm["c_index_cv_ci"] = compute_mean_confidence_interval_from_samples(
                            c_indices_cv_list,
                            clip_min=0.0,
                            clip_max=1.0,
                        )
                        self.log(f"C-Index CV for '{model_name_rm}' ({len(c_indices_cv_list)}/{kf_cv.get_n_splits()} folds successful): Mean={model_data_rm['c_index_cv_mean']:.3f} (DE={model_data_rm['c_index_cv_std']:.3f})", "INFO")
                    else:
                        self.log(f"C-Index CV for '{model_name_rm}': No C-Indices calculated from any fold.", "WARN")
                        model_data_rm["c_index_cv_mean"] = None # Ensure it's None if list is empty
                        model_data_rm["c_index_cv_std"] = None
                        model_data_rm["c_index_cv_ci"] = None

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
            model_data_rm["c_index_cv_ci"] = None
            model_data_rm["c_index_test"] = None
            model_data_rm["c_index_test_ci"] = None
            model_data_rm["c_index_train_ci"] = None
            model_data_rm["test_proportion"] = None
            model_data_rm["c_index_gap"] = None
            model_data_rm["oos_predictions"] = None

        if model_for_diagnostics is not None and model_data_rm.get("c_index_train_ci") is None:
            try:
                preds_train_diag = model_for_diagnostics.predict_partial_hazard(df_for_diagnostics)
                model_data_rm["c_index_train_ci"] = bootstrap_concordance_ci_from_scores(
                    y_for_diagnostics,
                    time_col_rm,
                    event_col_rm,
                    -np.asarray(preds_train_diag, dtype=float).reshape(-1),
                )
            except Exception as err_cindex_ci:
                self.log(f"No se pudo calcular IC del C-Index train para '{model_name_rm}': {err_cindex_ci}", "WARN")

        # 5. Store internal data copies
        model_data_rm["_df_for_fit_main_INTERNAL_USE"] = df_lifelines_rm.copy()
        model_data_rm["_X_design_rm_INTERNAL_USE"] = X_design_rm.copy() 
        model_data_rm["_y_survival_rm_INTERNAL_USE"] = y_survival_rm.copy() 

        # 6. Calculate and Store Final Metrics
        model_data_rm["metrics"] = compute_model_metrics(
            model_for_diagnostics,
            X_design_rm, y_for_diagnostics, time_col_rm, event_col_rm,
            model_data_rm.get("c_index_cv_mean"),
            model_data_rm.get("c_index_cv_std"),
            model_data_rm.get("schoenfeld_results"),
            model_data_rm.get("loglik_null"),
            self.log,
            model_data_rm.get("c_index_test"),
            model_data_rm.get("test_proportion"),
            model_data_rm.get("c_index_gap"),
            model_data_rm.get("c_index_train_ci"),
            model_data_rm.get("c_index_test_ci"),
            model_data_rm.get("c_index_cv_ci"),
            model_data_rm.get("c_index_uno"),
            model_data_rm.get("c_index_antolini"),
            model_data_rm.get("tau"),
            ibs=model_data_rm.get("ibs"),
            brier_q25=model_data_rm.get("brier_q25"), brier_q50=model_data_rm.get("brier_q50"), brier_q75=model_data_rm.get("brier_q75"),
            auroc_q25=model_data_rm.get("auroc_q25"), auroc_q50=model_data_rm.get("auroc_q50"), auroc_q75=model_data_rm.get("auroc_q75"),
            c_harrell_q25=model_data_rm.get("c_harrell_q25"), c_harrell_q50=model_data_rm.get("c_harrell_q50"), c_harrell_q75=model_data_rm.get("c_harrell_q75"),
            time_q25=model_data_rm.get("time_q25"), time_q50=model_data_rm.get("time_q50"), time_q75=model_data_rm.get("time_q75"),
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
            spline_metadata_snapshot = md_tv.get('spline_basis_metadata', {}) or {}
            patsy_formula_for_splines = md_tv.get('formula_patsy', '')

            handled_covariates = set()

            if spline_metadata_snapshot:
                for cov_name_meta in sorted(spline_metadata_snapshot.keys()):
                    meta_details = spline_metadata_snapshot.get(cov_name_meta) or {}
                    detail_parts = [cov_name_meta]

                    method_label = meta_details.get('spline_type')
                    if method_label:
                        detail_parts.append(f"método={method_label}")

                    degree_val = meta_details.get('degree')
                    if degree_val is not None:
                        detail_parts.append("cúbico" if degree_val == 3 else f"grado={degree_val}")

                    df_requested_val = meta_details.get('df_requested')
                    df_applied_val = meta_details.get('df_applied')
                    df_effective_val = meta_details.get('df_effective')

                    if df_requested_val is not None and df_applied_val is not None and df_requested_val != df_applied_val:
                        detail_parts.append(f"df pedido={df_requested_val}")
                        detail_parts.append(f"df usado={df_applied_val}")
                    elif df_applied_val is not None:
                        detail_parts.append(f"df={df_applied_val}")
                    elif df_requested_val is not None:
                        detail_parts.append(f"df={df_requested_val}")

                    if df_effective_val is not None and df_effective_val != df_applied_val:
                        detail_parts.append(f"df efectivo={df_effective_val}")

                    num_knots_req = meta_details.get('num_knots_requested')
                    num_knots_auto = meta_details.get('num_knots_derived_from_df')
                    num_knots_used = meta_details.get('num_knots_used')

                    if num_knots_req is not None:
                        detail_parts.append(f"nodos pedidos={num_knots_req}")
                    if num_knots_auto is not None and (num_knots_req is None or num_knots_auto != num_knots_req):
                        detail_parts.append(f"nodos df={num_knots_auto}")
                    if num_knots_used is not None:
                        detail_parts.append(f"nodos usados={num_knots_used}")

                    knots_source = meta_details.get('internal_knots_source')
                    if knots_source:
                        detail_parts.append(f"origen_nodos={knots_source}")

                    boundary_vals = meta_details.get('boundary_knots') or []
                    if boundary_vals:
                        boundary_preview = ", ".join(f"{val:.2g}" for val in boundary_vals[:4])
                        detail_parts.append(f"límites=[{boundary_preview}]")

                    knot_values = meta_details.get('internal_knots') or []
                    if knot_values:
                        preview_vals = ", ".join(f"{val:.2g}" for val in knot_values[:4])
                        if len(knot_values) > 4:
                            preview_vals += ", ..."
                        detail_parts.append(f"nodos=[{preview_vals}]")

                    vars_splines_display_list.append(" | ".join(detail_parts))
                    handled_covariates.add(cov_name_meta)

            # Añadir covariables restantes (sin spline o sin metadata) usando la fórmula como fallback
            for orig_var_name in original_covs_in_model:
                if orig_var_name in handled_covariates:
                    continue

                display_str = orig_var_name

                cr_match = re.search(rf"cr\(Q\('{re.escape(orig_var_name)}'\),\s*df=(\d+)\)", patsy_formula_for_splines)
                if cr_match:
                    df = cr_match.group(1)
                    display_str += f" (Natural, df={df})"
                else:
                    bs_match = re.search(rf"bs\(Q\('{re.escape(orig_var_name)}'\),\s*df=(\d+)(?:,\s*degree=(\d+))?\)", patsy_formula_for_splines)
                    if bs_match:
                        df = bs_match.group(1)
                        degree = bs_match.group(2) if bs_match.group(2) else '3'
                        display_str += f" (B-spline, df={df}, deg={degree})"

                vars_splines_display_list.append(display_str)

            penalizer_value = md_tv.get('penalizer_value', 0.0)
            l1_ratio_value = md_tv.get('l1_ratio_value', 0.0)
            penalization_info = f"penalización={penalizer_value:.4g} (l1={l1_ratio_value:.2f})"

            if vars_splines_display_list:
                vars_splines_display_list.insert(0, penalization_info)
            elif covs_processed:
                vars_splines_display_list = [penalization_info] + covs_processed
            else:
                vars_splines_display_list = [penalization_info]

            vars_splines_str = ", ".join(vars_splines_display_list)

            metrics_tv = md_tv.get('metrics', {})

            # Test %
            test_prop_tv = metrics_tv.get('Test Proportion')

            # AIC
            aic_tv = metrics_tv.get('AIC')

            # -2 LogLik
            minus_2_loglik_tv = metrics_tv.get('-2 Log-Likelihood') # Ya debería estar en metrics
            if minus_2_loglik_tv is None and pd.notna(metrics_tv.get('Log-Likelihood')): # Calcular si no está
                minus_2_loglik_tv = -2.0 * metrics_tv.get('Log-Likelihood')

            # C-Index (Train)
            c_idx_tr_tv = metrics_tv.get('C-Index (Training)')

            # C-Index (Test)
            c_idx_test_tv = metrics_tv.get('C-Index (Test)')

            # C-Index (CV/Test)
            c_idx_cv_tv = md_tv.get('c_index_cv_mean') # Directamente del diccionario del modelo

            # Gap Test-Train
            c_idx_gap_tv = metrics_tv.get('C-Index Gap (Test-Train)')

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


            # BIC
            bic_tv = metrics_tv.get('BIC')
            _f4 = lambda v: f"{v:.4f}" if pd.notna(v) else "N/A"

            vals_tv = (
                i + 1,                                      # #
                name_tv,                                    # Nombre Modelo
                vars_splines_str,                           # Variables y Splines
                f"{test_prop_tv:.2f}" if pd.notna(test_prop_tv) else "N/A", # Test %
                f"{aic_tv:.2f}" if pd.notna(aic_tv) else "N/A", # AIC
                f"{bic_tv:.2f}" if pd.notna(bic_tv) else "N/A", # BIC
                f"{minus_2_loglik_tv:.2f}" if pd.notna(minus_2_loglik_tv) else "N/A", # -2 LogLik
                self._format_c_index_display(c_idx_tr_tv, metrics_tv.get('C-Index (Training) CI'), decimals=3), # C-Index (Train)
                self._format_c_index_display(c_idx_test_tv, metrics_tv.get('C-Index (Test) CI'), decimals=3), # C-Index (Test)
                self._format_c_index_display(c_idx_cv_tv, metrics_tv.get('C-Index (CV Mean) CI'), decimals=3),   # C-Index (CV/Test)
                f"{c_idx_gap_tv:.3f}" if pd.notna(c_idx_gap_tv) else "N/A", # ΔTest-Train
                _f4(metrics_tv.get('C-Index Uno (IPCW)')),  # C-Uno
                _f4(metrics_tv.get('C-Index Antolini (Ctd)')),  # C-Antolini
                f"{metrics_tv.get('τ (tau)'):.1f}" if pd.notna(metrics_tv.get('τ (tau)')) else "N/A",  # τ
                _f4(metrics_tv.get('IBS')),  # IBS
                _f4(metrics_tv.get('C@Q25')), _f4(metrics_tv.get('C@Q50')), _f4(metrics_tv.get('C@Q75')),
                _f4(metrics_tv.get('Brier@Q25')), _f4(metrics_tv.get('Brier@Q50')), _f4(metrics_tv.get('Brier@Q75')),
                _f4(metrics_tv.get('AUC@Q25')), _f4(metrics_tv.get('AUC@Q50')), _f4(metrics_tv.get('AUC@Q75')),
                format_p_value(metrics_tv.get('Global LR Test p-value')) if pd.notna(metrics_tv.get('Global LR Test p-value')) else "N/A",  # LR p global
                format_p_value(metrics_tv.get('Wald p-value (global approx)')) if pd.notna(metrics_tv.get('Wald p-value (global approx)')) else "N/A",  # Wald p global
                format_p_value(schoenfeld_p_min_tv) if pd.notna(schoenfeld_p_min_tv) else "N/A", # Schoenfeld (p min)
                format_p_value(wald_p_max_tv) if pd.notna(wald_p_max_tv) else "N/A" # Wald (p max)
            )
            self.treeview_lista_modelos.insert("", tk.END, iid=str(i), values=vals_tv)
        self.log(f"Treeview actualizada con {len(self.generated_models_data)} modelos.", "INFO")

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

    # ── Column toggle for treeview_lista_modelos ──────────────────────────
    def _persist_models_tv_layout(self):
        self._persist_saved_layout("cox_generated_models", self._models_tv_col_config, getattr(self, "treeview_lista_modelos", None))

    def _persist_grid_tv_layout(self):
        self._persist_saved_layout("cox_model_grid", self._grid_tv_col_config, getattr(self, "model_grid_tree", None))

    def save_table_layouts(self):
        self._persist_models_tv_layout()
        self._persist_grid_tv_layout()

    def _show_models_tv_column_menu(self, event):
        if not hasattr(self, 'treeview_lista_modelos') or self.treeview_lista_modelos is None:
            return
        menu = tk.Menu(self.treeview_lista_modelos, tearoff=0)
        menu.add_command(label="── Columnas visibles ──", state="disabled")
        menu.add_separator()
        for col_id, cfg in self._models_tv_col_config.items():
            label = ("✓ " if cfg["visible"] else "   ") + cfg["heading"]
            menu.add_command(label=label, command=lambda c=col_id: self._toggle_models_tv_column(c))
        menu.add_separator()
        menu.add_command(label="Mostrar todas", command=self._show_all_models_tv_columns)
        try:
            menu.tk_popup(event.x_root, event.y_root)
        finally:
            menu.grab_release()

    def _toggle_models_tv_column(self, col_id):
        cfg = self._models_tv_col_config[col_id]
        try:
            current_width = int(self.treeview_lista_modelos.column(col_id, option="width"))
            if current_width > 0:
                cfg["width"] = current_width
        except Exception:
            pass
        cfg["visible"] = not cfg["visible"]
        self._apply_treeview_column_layout(self.treeview_lista_modelos, self._models_tv_col_config)
        self._persist_models_tv_layout()
        self._persist_saved_layout("cox_generated_models", self._models_tv_col_config, self.treeview_lista_modelos)

    def _show_all_models_tv_columns(self):
        for col_id, cfg in self._models_tv_col_config.items():
            cfg["visible"] = True
        self._apply_treeview_column_layout(self.treeview_lista_modelos, self._models_tv_col_config)
        self._persist_models_tv_layout()
        self._persist_saved_layout("cox_generated_models", self._models_tv_col_config, self.treeview_lista_modelos)

    def _apply_treeview_column_layout(self, treeview, col_config):
        visible_cols = [col_id for col_id, cfg in col_config.items() if cfg.get("visible", True)]
        try:
            treeview.configure(displaycolumns=visible_cols if visible_cols else ())
        except Exception:
            pass
        for col_id, cfg in col_config.items():
            if cfg.get("visible", True):
                treeview.column(col_id, width=cfg["width"], minwidth=24, stretch=False)
            else:
                treeview.column(col_id, width=0, minwidth=0, stretch=False)

    # ── Column toggle for model_grid_tree ─────────────────────────────────
    def _show_grid_tv_column_menu(self, event):
        if not hasattr(self, 'model_grid_tree') or self.model_grid_tree is None:
            return
        menu = tk.Menu(self.model_grid_tree, tearoff=0)
        menu.add_command(label="── Columnas visibles ──", state="disabled")
        menu.add_separator()
        for col_id, cfg in self._grid_tv_col_config.items():
            label = ("✓ " if cfg["visible"] else "   ") + cfg["heading"]
            menu.add_command(label=label, command=lambda c=col_id: self._toggle_grid_tv_column(c))
        menu.add_separator()
        menu.add_command(label="Mostrar todas", command=self._show_all_grid_tv_columns)
        try:
            menu.tk_popup(event.x_root, event.y_root)
        finally:
            menu.grab_release()

    def _toggle_grid_tv_column(self, col_id):
        cfg = self._grid_tv_col_config[col_id]
        try:
            current_width = int(self.model_grid_tree.column(col_id, option="width"))
            if current_width > 0:
                cfg["width"] = current_width
        except Exception:
            pass
        cfg["visible"] = not cfg["visible"]
        self._apply_treeview_column_layout(self.model_grid_tree, self._grid_tv_col_config)
        self._persist_grid_tv_layout()
        self._persist_saved_layout("cox_model_grid", self._grid_tv_col_config, self.model_grid_tree)

    def _show_all_grid_tv_columns(self):
        for col_id, cfg in self._grid_tv_col_config.items():
            cfg["visible"] = True
        self._apply_treeview_column_layout(self.model_grid_tree, self._grid_tv_col_config)
        self._persist_grid_tv_layout()
        self._persist_saved_layout("cox_model_grid", self._grid_tv_col_config, self.model_grid_tree)

    def _sort_grid_tv_column(self, col_name):
        """Ordena el model_grid_tree por la columna especificada."""
        if not self.model_grid_tree:
            return
        try:
            col_idx = self.model_grid_tree["columns"].index(col_name)
        except ValueError:
            return
        items = [(self.model_grid_tree.item(iid, "values"), iid) for iid in self.model_grid_tree.get_children()]
        reverse = self._grid_sort_reversed.get(col_name, False)
        def sort_key(pair):
            val = pair[0][col_idx] if col_idx < len(pair[0]) else ""
            try:
                return (0, float(val))
            except (ValueError, TypeError):
                return (1, str(val).lower())
        items.sort(key=sort_key, reverse=reverse)
        for idx, (vals, iid) in enumerate(items):
            self.model_grid_tree.move(iid, "", idx)
        self._grid_sort_reversed[col_name] = not reverse

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
            elif col_name == "Test %":
                val = metrics.get('Test Proportion')
                return float(val) if pd.notna(val) else float('-inf')
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
            elif col_name == "C-Index (Test)":
                val = metrics.get('C-Index (Test)')
                return float(val) if pd.notna(val) else float('-inf')
            elif col_name in ("C-Index (CV)", "C-Index (CV/Test)"):
                val = model_dict_item.get('c_index_cv_mean')
                return float(val) if pd.notna(val) else float('-inf')
            elif col_name == "ΔTest-Train":
                val = metrics.get('C-Index Gap (Test-Train)')
                return float(val) if pd.notna(val) else float('-inf')
            elif col_name == "BIC":
                val = metrics.get('BIC')
                return float(val) if pd.notna(val) else float('-inf')
            elif col_name == "C-Uno (IPCW)":
                val = metrics.get('C-Index Uno (IPCW)')
                return float(val) if pd.notna(val) else float('-inf')
            elif col_name == "C-Antolini (Ctd)":
                val = metrics.get('C-Index Antolini (Ctd)')
                return float(val) if pd.notna(val) else float('-inf')
            elif col_name == "τ (tau)":
                val = metrics.get('τ (tau)')
                return float(val) if pd.notna(val) else float('-inf')
            elif col_name == "IBS":
                val = metrics.get('IBS')
                return float(val) if pd.notna(val) else float('-inf')
            elif col_name in ("C@Q25", "C@Q50", "C@Q75", "Brier@Q25", "Brier@Q50", "Brier@Q75",
                              "AUC@Q25", "AUC@Q50", "AUC@Q75"):
                val = metrics.get(col_name)
                return float(val) if pd.notna(val) else float('-inf')
            elif col_name == "LR p (global)":
                val = metrics.get('Global LR Test p-value')
                return float(val) if pd.notna(val) else float('inf')
            elif col_name == "Wald p (global)":
                val = metrics.get('Wald p-value (global approx)')
                return float(val) if pd.notna(val) else float('inf')
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
        self._show_best_sorted_models_popup(col_name, current_reverse_order)
        self.log(f"Treeview ordenado por '{col_name}', descendente={current_reverse_order}.", "INFO")

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
        if not hasattr(self, "treeview_lista_modelos"):
            return
        children = self.treeview_lista_modelos.get_children("")
        if not children:
            return
        columns = list(self.treeview_lista_modelos["columns"])
        if col_name not in columns:
            return

        col_idx = columns.index(col_name)
        model_idx = columns.index("Nombre Modelo") if "Nombre Modelo" in columns else None
        params_idx = columns.index("Variables y Splines") if "Variables y Splines" in columns else None

        first_values = self.treeview_lista_modelos.item(children[0], "values")
        best_key = self._parse_table_sort_value(first_values[col_idx] if col_idx < len(first_values) else "")

        all_top_rows = []
        numeric_values = []
        for item_id in children:
            values = self.treeview_lista_modelos.item(item_id, "values")
            current_key = self._parse_table_sort_value(values[col_idx] if col_idx < len(values) else "")
            if not self._sort_keys_equal(current_key, best_key):
                break

            model_name = values[model_idx] if model_idx is not None and model_idx < len(values) else "-"
            params_text = values[params_idx] if params_idx is not None and params_idx < len(values) else "-"
            metric_text = values[col_idx] if col_idx < len(values) else "-"
            all_top_rows.append((model_name, params_text, metric_text))
            if current_key[0] == "num":
                numeric_values.append(float(current_key[1]))

        if not all_top_rows:
            return

        # Detectar si todos los empatados comparten las mismas covariables
        unique_params = set(r[1] for r in all_top_rows)
        diversity_warning = None
        if len(all_top_rows) >= 5 and len(unique_params) == 1:
            diversity_warning = (
                "⚠  Todos los modelos empatados tienen exactamente las mismas covariables y splines. "
                "Esto significa que comparten la misma estructura y el empate es estructural, no casual. "
                "Prueba diferentes especificaciones de variables o splines para obtener modelos distintos."
            )

        MAX_DISPLAY = 10
        top_rows = all_top_rows[:MAX_DISPLAY]
        hidden_count = len(all_top_rows) - MAX_DISPLAY

        popup = tk.Toplevel(self)
        popup.title("Mejor valor en modelos Cox")
        popup.geometry("1040x480")
        popup.transient(self.winfo_toplevel())
        popup.grab_set()

        sort_order_text = "descendente" if reverse else "ascendente"
        total_tied = len(all_top_rows)
        title_text = (
            f"Columna: {col_name} | Orden: {sort_order_text} | "
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
                      wraplength=1000, justify="left").pack(anchor="w")

        tree = ttk.Treeview(popup, columns=("modelo", "parametros", "valor"), show="headings", height=10)
        tree.heading("modelo", text="Modelo")
        tree.heading("parametros", text="Variables y Splines")
        tree.heading("valor", text=col_name)
        tree.column("modelo", width=180, anchor="w", stretch=False)
        tree.column("parametros", width=680, anchor="w", stretch=True)
        tree.column("valor", width=150, anchor="center", stretch=False)
        tree.pack(fill=tk.BOTH, expand=True, padx=12, pady=(0, 4))

        for row in top_rows:
            tree.insert("", "end", values=row)

        if hidden_count > 0:
            ttk.Label(popup, text=f"+ {hidden_count} modelos más con el mismo valor (no mostrados).",
                      foreground="#777777").pack(anchor="w", padx=12, pady=(0, 4))

        ttk.Button(popup, text="Cerrar", command=popup.destroy).pack(pady=(0, 10))


    def _execute_cox_modeling_orchestrator(self):
        self._persist_single_selected_categorical_config_from_panel(quiet=True)
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
                                                             scaled_columns_info=scaled_cols_list,
                                                             selected_covariates_original=[orig_cov_uni])
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
            
            selected_covs_for_snapshot = (
                list(selected_orig_covs_after_selection)
                if sel_meth_ui != "Ninguno (usar todas)"
                else list(getattr(self, 'selected_covariables_from_ui', []))
            )

            md_multi = self._run_model_and_get_metrics(df_multi_current, X_multi_current, y_multi,
                                                       t_col_final, e_col_final, formula_multi_current,
                                                       name_multi, terms_multi_current, formula_init_patsy_full, # formula_init_patsy_full is for new data transform
                                                       pen_val, l1_r, model_type_for_fit_logic="Multivariado",
                                                       scaling_method_applied=scaling_method_used,
                                                       fitted_scaler_obj=scaler_object,
                                                       scaled_columns_info=scaled_cols_list,
                                                       selected_covariates_original=selected_covs_for_snapshot)
            if md_multi:
                temp_models_list_orch.append(md_multi)
                if md_multi.get("model") is not None:
                    successful_fits += 1
                else:
                    failed_fits += 1
 
        # Añadir los modelos generados a la lista existente, no sobrescribir
        self.generated_models_data.extend(temp_models_list_orch)
        self._update_models_treeview()

        if temp_models_list_orch and hasattr(self, 'treeview_lista_modelos'):
            try:
                last_idx = len(self.generated_models_data) - 1
                last_iid = str(last_idx)
                self.treeview_lista_modelos.selection_set(last_iid)
                self.treeview_lista_modelos.focus(last_iid)
                self.treeview_lista_modelos.see(last_iid)
                self._on_model_select_from_treeview()
                self._clear_holdout_config_dirty()
                self.log(f"Selección automática activada para el último modelo generado (índice {last_idx + 1}).", "INFO")
            except Exception as err_select_last:
                self.log(f"No se pudo auto-seleccionar el último modelo generado: {err_select_last}", "WARN")

        msg_fin = f"Modelado completado. {len(temp_models_list_orch)} modelo(s) generado(s) y añadido(s)." if temp_models_list_orch else "No se generó ningún modelo nuevo."
        self.log(msg_fin, "SUCCESS" if temp_models_list_orch else "WARN")

        total_models_attempted = successful_fits + failed_fits
        self.log(f"Resumen de Convergencia de Modelos:", "SUBHEADER")
        self.log(f"  Modelos Totales Intentados: {total_models_attempted}", "INFO")
        self.log(f"  Ajustes Exitosos: {successful_fits}", "SUCCESS" if successful_fits > 0 else "INFO")
        self.log(f"  Ajustes Fallidos: {failed_fits}", "ERROR" if failed_fits > 0 else "INFO")

        self.log("*"*35 + " FIN PROCESO DE MODELADO COX " + "*"*35, "HEADER")


    def _restore_selected_model_ui_state(self, model_dict, restore_holdout_settings=None):
        if not isinstance(model_dict, dict):
            return

        columns = []
        for df_source in (
            getattr(self, 'data', None),
            model_dict.get('df_used_for_fit'),
            model_dict.get('_df_for_fit_main_INTERNAL_USE'),
        ):
            if isinstance(df_source, pd.DataFrame):
                for col_name in df_source.columns.tolist():
                    if col_name not in columns:
                        columns.append(col_name)

        time_col = model_dict.get('time_col_for_model') or ""
        event_col = model_dict.get('event_col_for_model') or ""
        selected_covs = list(model_dict.get('selected_covariables_original') or [])
        if not selected_covs:
            for cov_name in model_dict.get('covariates_processed', []) or []:
                if cov_name in columns and cov_name not in selected_covs and cov_name not in [time_col, event_col]:
                    selected_covs.append(cov_name)

        for col_name in [time_col, event_col] + selected_covs:
            if col_name and col_name not in columns:
                columns.append(col_name)

        if hasattr(self, 'combo_col_tiempo'):
            self.combo_col_tiempo['values'] = columns
            self.combo_col_tiempo.set(time_col if (not columns or time_col in columns) else "")
        if hasattr(self, 'combo_col_evento'):
            self.combo_col_evento['values'] = columns
            self.combo_col_evento.set(event_col if (not columns or event_col in columns) else "")
        if hasattr(self, 'listbox_covariables_disponibles'):
            self.listbox_covariables_disponibles.delete(0, tk.END)
            cov_columns = [c for c in columns if c not in [time_col, event_col]]
            for idx, col_name in enumerate(cov_columns):
                self.listbox_covariables_disponibles.insert(tk.END, col_name)
                if col_name in selected_covs:
                    try:
                        self.listbox_covariables_disponibles.selection_set(idx)
                    except Exception:
                        pass

        config_snapshot = model_dict.get('ui_config_snapshot') or {}
        if hasattr(self, 'cox_model_type_var') and config_snapshot.get('model_type'):
            self.cox_model_type_var.set(str(config_snapshot.get('model_type')))

        should_restore_holdout = restore_holdout_settings
        if should_restore_holdout is None:
            should_restore_holdout = not bool(getattr(self, '_manual_holdout_config_dirty', False))

        if should_restore_holdout:
            if hasattr(self, 'test_size_var') and 'test_size' in config_snapshot and config_snapshot.get('test_size') is not None:
                self.test_size_var.set(config_snapshot.get('test_size'))
            if hasattr(self, 'stratify_holdout_var') and 'stratify_holdout' in config_snapshot and config_snapshot.get('stratify_holdout') is not None:
                self.stratify_holdout_var.set(bool(config_snapshot.get('stratify_holdout')))
            if hasattr(self, 'tau_mode_var') and config_snapshot.get('tau_mode'):
                self.tau_mode_var.set(str(config_snapshot.get('tau_mode')))
            if hasattr(self, 'tau_manual_var') and 'tau_manual' in config_snapshot:
                tau_manual_value = config_snapshot.get('tau_manual')
                self.tau_manual_var.set("" if tau_manual_value in (None, "None") else str(tau_manual_value))
            self._clear_holdout_config_dirty()


    def _on_model_select_from_treeview(self, event=None):
        selected_iids = self.treeview_lista_modelos.selection()
        selected_models = []

        if selected_iids:
            for iid in selected_iids:
                try:
                    idx = int(iid)
                except ValueError:
                    self.log(f"Ítem seleccionado '{iid}' no es un índice válido.", "WARN")
                    continue

                if 0 <= idx < len(self.generated_models_data):
                    selected_models.append(self.generated_models_data[idx])
                else:
                    self.log(f"Índice de modelo fuera de rango: {idx}.", "WARN")

        self.selected_models_in_treeview = selected_models
        self.selected_model_in_treeview = selected_models[0] if selected_models else None

        if self.selected_model_in_treeview:
            self.log(
                f"Modelos seleccionados: {', '.join([md.get('model_name', 'N/A') for md in selected_models])}",
                "INFO"
            )
            try:
                self._restore_selected_model_ui_state(self.selected_model_in_treeview)
            except Exception as err_restore_ui:
                self.log(f"No se pudo restaurar la UI del modelo seleccionado: {err_restore_ui}", "WARN")
        else:
            self.log("Ningún modelo seleccionado.", "INFO")

        # Actualizar UI de nombre/notas personalizados usando el primer modelo seleccionado
        if self.selected_model_in_treeview:
            custom_name = self.selected_model_in_treeview.get('custom_model_name', self.selected_model_in_treeview.get('model_name', ''))
            custom_notes = self.selected_model_in_treeview.get('custom_model_notes', '')
            if self.entry_custom_model_name:
                self.entry_custom_model_name_var.set(custom_name)
            if self.text_custom_model_notes:
                self.text_custom_model_notes.config(state=tk.NORMAL)
                self.text_custom_model_notes.delete("1.0", tk.END)
                self.text_custom_model_notes.insert("1.0", custom_notes)
                self.text_custom_model_notes.config(state=tk.NORMAL)
        else:
            if self.entry_custom_model_name:
                self.entry_custom_model_name_var.set("")
            if self.text_custom_model_notes:
                self.text_custom_model_notes.config(state=tk.NORMAL)
                self.text_custom_model_notes.delete("1.0", tk.END)
                self.text_custom_model_notes.config(state=tk.DISABLED)

        if self.btn_oos_calibration:
            enabled = any(md.get("oos_predictions") for md in selected_models)
            self.btn_oos_calibration.config(state=tk.NORMAL if enabled else tk.DISABLED)

        if self.btn_collinearity_diag:
            can_run_vif = False
            if self.selected_model_in_treeview:
                x_design = self.selected_model_in_treeview.get("X_design_used_for_fit")
                if x_design is not None and isinstance(x_design, pd.DataFrame) and x_design.shape[1] > 1:
                    can_run_vif = True
            self.btn_collinearity_diag.config(state=tk.NORMAL if can_run_vif else tk.DISABLED)

        if self.btn_nomogram:
            can_generate_nomogram = False
            if self.selected_model_in_treeview:
                model_obj = self.selected_model_in_treeview.get("model")
                x_design_sel = self.selected_model_in_treeview.get("X_design_used_for_fit")
                if (
                    model_obj
                    and hasattr(model_obj, 'params_')
                    and model_obj.params_ is not None
                    and not model_obj.params_.empty
                    and isinstance(x_design_sel, pd.DataFrame)
                    and not x_design_sel.empty
                ):
                    can_generate_nomogram = True
            self.btn_nomogram.config(state=tk.NORMAL if can_generate_nomogram else tk.DISABLED)

        if self.btn_delete_model:
            self.btn_delete_model.config(state=tk.NORMAL if self.selected_model_in_treeview else tk.DISABLED)

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

    def _format_term_for_display(self, term_raw):
        """Convierte un término de Patsy a una etiqueta más legible para resúmenes y gráficos."""
        if not term_raw:
            return ""

        term = str(term_raw)

        # Interacciones separadas por ':'
        if ':' in term:
            parts = [self._format_term_for_display(part.strip()) for part in term.split(':')]
            return " × ".join(parts)

        # Potencias explícitas: I(Q('x') ** 2)
        match_power = re.match(r"I\(Q\('([^']+)'\) \*\* (\d+)\)", term)
        if match_power:
            base_name, power = match_power.groups()
            power_int = int(power)
            superscript_map = {2: '²', 3: '³', 4: '⁴', 5: '⁵'}
            if power_int in superscript_map:
                return f"{base_name}{superscript_map[power_int]}"
            return f"{base_name}^{power_int}"

        # B-splines bs(Q('var'), ...)[i]
        match_bs = re.match(r"bs\(Q\('([^']+)'\),[^)]*\)\[(\d+)\]", term)
        if match_bs:
            base_name = match_bs.group(1)
            basis_idx = int(match_bs.group(2)) + 1
            return f"{base_name} (spline {basis_idx})"

        # Natural splines cr(Q('var'), ...)[i]
        match_cr = re.match(r"cr\(Q\('([^']+)'\),[^)]*\)\[(\d+)\]", term)
        if match_cr:
            base_name = match_cr.group(1)
            basis_idx = int(match_cr.group(2)) + 1
            return f"{base_name} (natural spline {basis_idx})"

        # C(Q('var'), Treatment('ref'))[T.level]
        match_cat_treatment = re.match(r"C\(Q\('([^']+)'\),\s*Treatment\('([^']+)'\)\)\[T\.([^\]]+)\]", term)
        if match_cat_treatment:
            base, ref_level, level = match_cat_treatment.groups()
            if str(level).strip().upper() == "RESTO":
                return f"{base}: RESTO vs {ref_level}"
            return f"{base}: {level} vs {ref_level}"

        # Q('var') wrapper
        match_q = re.match(r"Q\('([^']+)'\)$", term)
        if match_q:
            return match_q.group(1)

        # C('var')[T.level] para dummies
        match_cat = re.match(r"C\('([^']+)'\)\[T\.([^\]]+)\]", term)
        if match_cat:
            base, level = match_cat.groups()
            return f"{base} = {level}"

        # Transformaciones comunes
        if term.startswith("np.log(") and term.endswith(")"):
            inner = term[len("np.log("):-1]
            return f"log({inner})"

        if term.startswith("np.log1p(") and term.endswith(")"):
            inner = term[len("np.log1p("):-1]
            return f"log1p({inner})"

        return term

    def _format_term_for_nomogram(self, term_raw):
        """Convierte un término de Patsy a una etiqueta legible para el nomograma."""
        return self._format_term_for_display(term_raw)

    def generate_nomogram_for_selected_model(self):
        if not self._check_model_selected_and_valid(check_params=True):
            return

        model_data = self.selected_model_in_treeview
        model_obj = model_data.get('model')
        x_design = model_data.get('X_design_used_for_fit')
        model_name = model_data.get('custom_model_name', model_data.get('model_name', 'Modelo Cox'))

        if not isinstance(x_design, pd.DataFrame) or x_design.empty:
            messagebox.showinfo("Nomograma No Disponible", "La matriz de diseño del modelo está vacía; no se puede construir el nomograma.", parent=self.parent_for_dialogs)
            self.log("Nomograma: matriz de diseño vacía o no disponible.", "WARN")
            return

        params = getattr(model_obj, 'params_', None)
        if params is None or params.empty:
            messagebox.showinfo("Nomograma No Disponible", "El modelo no tiene coeficientes estimados.", parent=self.parent_for_dialogs)
            self.log("Nomograma: modelo sin coeficientes estimados.", "WARN")
            return

        def _parse_float_list(input_str):
            if input_str is None:
                return None
            values = []
            for raw_part in input_str.split(','):
                part = raw_part.strip()
                if not part:
                    continue
                try:
                    values.append(float(part))
                except ValueError:
                    self.log(f"Valor no numérico ignorado en lista: '{part}'", "WARN")
            return values

        default_percentiles = [2.5, 50, 75, 97]
        custom_tick_values_global = []
        percentiles_for_ticks = default_percentiles.copy()

        custom_values_input = simpledialog.askstring(
            "Valores para Nomograma",
            "Ingrese valores específicos (ej. 1.0, 2.5, 5) para marcadores en cada línea.\nDeje vacío para omitir.",
            parent=self.parent_for_dialogs
        )
        if custom_values_input is not None:
            parsed_custom = _parse_float_list(custom_values_input)
            if parsed_custom:
                custom_tick_values_global = parsed_custom
                self.log(f"Nomograma: valores personalizados para ticks: {custom_tick_values_global}", "INFO")

        percentiles_input = simpledialog.askstring(
            "Percentiles para Nomograma",
            "Ingrese percentiles (0-100) separados por coma (ej. 2.5, 50, 75, 97).\nDeje en blanco para usar el valor predeterminado.",
            initialvalue=", ".join(str(p) for p in default_percentiles),
            parent=self.parent_for_dialogs
        )
        if percentiles_input is not None:
            parsed_percentiles = _parse_float_list(percentiles_input)
            if parsed_percentiles:
                percentiles_filtered = [p for p in parsed_percentiles if 0 <= p <= 100]
                if percentiles_filtered:
                    percentiles_for_ticks = percentiles_filtered
                    self.log(f"Nomograma: percentiles personalizados: {percentiles_for_ticks}", "INFO")
                else:
                    self.log("Nomograma: percentiles ingresados fuera de rango. Se usan predeterminados.", "WARN")

        default_time_list = getattr(self, "nomogram_default_time_points", [5.0, 10.0, 15.0])
        survival_time_points = default_time_list.copy()
        time_points_input = simpledialog.askstring(
            "Tiempos para Supervivencia",
            "Ingrese tiempos (misma unidad que el tiempo del modelo) separados por coma.\n"
            "Ejemplo: 5, 10, 15",
            initialvalue=", ".join(f"{t:g}" for t in default_time_list) if default_time_list else "",
            parent=self.parent_for_dialogs
        )

        if time_points_input is not None:
            parsed_times = _parse_float_list(time_points_input)
            if parsed_times:
                filtered_times = sorted({t for t in parsed_times if t > 0})
                if filtered_times:
                    survival_time_points = filtered_times
                    self.nomogram_default_time_points = survival_time_points.copy()
                    self.log(f"Nomograma: tiempos personalizados para supervivencia: {survival_time_points}", "INFO")
                else:
                    self.log("Nomograma: tiempos de supervivencia ingresados inválidos. Se usan predeterminados.", "WARN")
            elif time_points_input.strip():
                self.log("Nomograma: entrada de tiempos sin valores numéricos válidos. Se usan predeterminados.", "WARN")

        survival_time_points = sorted({t for t in survival_time_points if t > 0})

        def _add_tick_entry(ticks_list, value, label=None):
            if value is None:
                return
            try:
                value_f = float(value)
            except (TypeError, ValueError):
                return
            if math.isnan(value_f) or math.isinf(value_f):
                return
            for entry in ticks_list:
                if math.isclose(entry["value"], value_f, rel_tol=1e-9, abs_tol=1e-9):
                    if label and label not in entry["labels"]:
                        entry["labels"].append(label)
                    return
            ticks_list.append({"value": value_f, "labels": [label] if label else []})

        original_training_df = model_data.get('_df_for_fit_main_INTERNAL_USE')
        if isinstance(original_training_df, pd.DataFrame):
            try:
                original_training_df = original_training_df.loc[x_design.index]
            except Exception:
                original_training_df = original_training_df.copy()
        else:
            original_training_df = None

        term_details = []
        spline_groups = {}

        def _add_standard_term_detail(term_name, coef_value, series_full):
            series_numeric = pd.to_numeric(series_full, errors='coerce').dropna()
            if series_numeric.empty:
                self.log(f"Nomograma: término '{term_name}' sin datos numéricos válidos. Se omite.", "DEBUG")
                return False

            col_min = float(series_numeric.min())
            col_max = float(series_numeric.max())
            if math.isclose(col_min, col_max, rel_tol=1e-9, abs_tol=1e-9):
                self.log(f"Nomograma: término '{term_name}' tiene rango nulo y se omite.", "INFO")
                return False

            effect_range = abs(float(coef_value)) * (col_max - col_min)
            ticks_info = []
            _add_tick_entry(ticks_info, col_min)
            _add_tick_entry(ticks_info, col_max)

            if custom_tick_values_global:
                for custom_val in custom_tick_values_global:
                    if col_min - 1e-12 <= custom_val <= col_max + 1e-12:
                        _add_tick_entry(ticks_info, custom_val)

            if percentiles_for_ticks:
                for pct in percentiles_for_ticks:
                    try:
                        q_val = float(series_numeric.quantile(pct / 100.0))
                    except Exception:
                        continue
                    if math.isnan(q_val):
                        continue
                    if col_min - 1e-12 <= q_val <= col_max + 1e-12:
                        _add_tick_entry(ticks_info, q_val, f"P{pct:g}")

            if len(ticks_info) < 3:
                auto_ticks = np.linspace(col_min, col_max, num=5)
                for auto_val in auto_ticks:
                    _add_tick_entry(ticks_info, auto_val)

            ticks_info.sort(key=lambda entry: entry["value"])

            term_details.append({
                "term": term_name,
                "coef": float(coef_value),
                "min": col_min,
                "max": col_max,
                "series": series_numeric,
                "effect_range": effect_range,
                "is_spline_group": False,
                "ticks_info": ticks_info
            })
            return True

        def _build_spline_group_detail(base_name, group_info):
            entries = group_info.get("entries", [])
            if not entries:
                return None

            spline_func = group_info.get("spline_func", "bs")
            spline_label = "B-spline" if spline_func == "bs" else "Natural spline"
            basis_columns = {}
            coefs = []
            for idx_entry, entry in enumerate(entries):
                series_full = entry.get("series_full")
                if series_full is None:
                    continue
                series_clean = pd.to_numeric(series_full, errors='coerce')
                basis_columns[f"basis_{idx_entry}"] = series_clean.fillna(0.0)
                coefs.append(float(entry.get("coef", 0.0)))

            if not basis_columns or not coefs:
                return None

            basis_df = pd.DataFrame(basis_columns)
            if basis_df.empty:
                return None

            effect_series = basis_df.values @ np.array(coefs, dtype=float)
            effect_series = pd.Series(effect_series, index=basis_df.index)

            raw_series = None
            if isinstance(original_training_df, pd.DataFrame) and base_name in original_training_df.columns:
                try:
                    raw_series = pd.to_numeric(original_training_df[base_name], errors='coerce')
                except Exception:
                    raw_series = None

            if raw_series is not None:
                if len(raw_series) != len(effect_series):
                    try:
                        raw_series = raw_series.reindex(effect_series.index)
                    except Exception:
                        raw_series = pd.Series(raw_series.values, index=effect_series.index[:len(raw_series)])
                else:
                    raw_series = pd.Series(raw_series.values, index=effect_series.index)

            if raw_series is None or raw_series.dropna().empty:
                self.log(f"Nomograma: no se pudo reconstruir valores originales para spline '{base_name}'.", "WARN")
                return None

            raw_series_clean = raw_series.dropna()
            combined_df = pd.DataFrame({"value": raw_series_clean, "effect": effect_series.loc[raw_series_clean.index]}).dropna()
            if combined_df.empty:
                self.log(f"Nomograma: datos combinados vacíos para spline '{base_name}'.", "WARN")
                return None

            aggregated = combined_df.groupby("value", as_index=False)["effect"].mean().sort_values("value")
            if aggregated.shape[0] > 200:
                sample_idx = np.linspace(0, aggregated.shape[0] - 1, 200, dtype=int)
                aggregated = aggregated.iloc[sample_idx]

            value_min = float(aggregated["value"].min())
            value_max = float(aggregated["value"].max())
            if math.isclose(value_min, value_max, rel_tol=1e-9, abs_tol=1e-9):
                self.log(f"Nomograma: spline '{base_name}' tiene rango mínimo en los datos originales.", "INFO")
                return None

            effect_min = float(aggregated["effect"].min())
            effect_max = float(aggregated["effect"].max())
            effect_range = abs(effect_max - effect_min)
            if math.isclose(effect_range, 0.0, rel_tol=1e-12, abs_tol=1e-12):
                self.log(f"Nomograma: efecto combinado de spline '{base_name}' es cercano a cero.", "INFO")
                return None

            ticks_info = []
            _add_tick_entry(ticks_info, value_min)
            _add_tick_entry(ticks_info, value_max)

            if custom_tick_values_global:
                for custom_val in custom_tick_values_global:
                    if value_min - 1e-12 <= custom_val <= value_max + 1e-12:
                        _add_tick_entry(ticks_info, custom_val)

            if percentiles_for_ticks and not raw_series_clean.empty:
                for pct in percentiles_for_ticks:
                    try:
                        q_val = float(raw_series_clean.quantile(pct / 100.0))
                    except Exception:
                        continue
                    if math.isnan(q_val):
                        continue
                    if value_min - 1e-12 <= q_val <= value_max + 1e-12:
                        _add_tick_entry(ticks_info, q_val, f"P{pct:g}")

            if len(ticks_info) < 3 and aggregated.shape[0] >= 2:
                sample_values = np.linspace(value_min, value_max, num=min(5, aggregated.shape[0]))
                for s_val in sample_values:
                    _add_tick_entry(ticks_info, s_val)

            ticks_info.sort(key=lambda entry: entry["value"])

            spline_detail = {
                "term": base_name,
                "coef": float(np.mean(coefs)),
                "min": value_min,
                "max": value_max,
                "series": None,
                "effect_range": effect_range,
                "is_spline_group": True,
                "effect_map_values": aggregated["value"].to_numpy(),
                "effect_map_effects": aggregated["effect"].to_numpy(),
                "effect_min": effect_min,
                "effect_max": effect_max,
                "num_basis": len(entries),
                "basis_terms": [entry.get("term") for entry in entries],
                "coefs": np.array(coefs, dtype=float),
                "spline_label": spline_label,
                "spline_func": spline_func,
                "ticks_info": ticks_info
            }
            return spline_detail

        for term, coef in params.items():
            if term not in x_design.columns:
                self.log(f"Nomograma: término '{term}' no encontrado en la matriz de diseño. Se omite.", "DEBUG")
                continue

            series_full = x_design[term]
            match_spline_term = re.match(r"(bs|cr)\(Q\('([^']+)'\),[^)]*\)\[(\d+)\]", term)
            if match_spline_term:
                spline_func = match_spline_term.group(1)
                base_name_spline = match_spline_term.group(2)
                spline_group = spline_groups.setdefault((spline_func, base_name_spline), {
                    "entries": [],
                    "spline_func": spline_func,
                    "base_name": base_name_spline
                })
                spline_group["entries"].append({
                    "term": term,
                    "coef": float(coef),
                    "series_full": series_full
                })
                continue

            _add_standard_term_detail(term, float(coef), series_full)

        for group_key, group_info in spline_groups.items():
            base_name_for_group = group_info.get("base_name")
            if base_name_for_group is None:
                if isinstance(group_key, tuple) and len(group_key) == 2:
                    base_name_for_group = group_key[1]
                else:
                    base_name_for_group = group_key
            spline_detail = _build_spline_group_detail(base_name_for_group, group_info)
            if spline_detail is not None:
                term_details.append(spline_detail)
            else:
                for entry in group_info.get("entries", []):
                    _add_standard_term_detail(entry.get("term"), entry.get("coef"), entry.get("series_full"))

        if not term_details:
            messagebox.showinfo("Nomograma No Disponible", "No se encontraron covariables numéricas con rango para construir el nomograma.", parent=self.parent_for_dialogs)
            self.log("Nomograma: sin términos válidos tras aplicar filtros.", "WARN")
            return

        global_max_effect = max(detail["effect_range"] for detail in term_details)
        total_effect_range = sum(detail["effect_range"] for detail in term_details)

        if math.isclose(global_max_effect, 0.0, rel_tol=1e-12, abs_tol=1e-12) or \
           math.isclose(total_effect_range, 0.0, rel_tol=1e-12, abs_tol=1e-12):
            messagebox.showinfo("Nomograma No Disponible", "Los coeficientes efectivos del modelo son cero; el nomograma no es informativo.", parent=self.parent_for_dialogs)
            self.log("Nomograma: efecto máximo global cero.", "WARN")
            return

        max_points_scale = 100.0
        points_factor = max_points_scale / global_max_effect

        total_points_scaled = 0.0
        lp_baseline = 0.0

        for detail in term_details:
            detail["points_max"] = detail["effect_range"] * points_factor
            total_points_scaled += detail["points_max"]

            if detail.get("is_spline_group"):
                values = detail.get("effect_map_values")
                effects = detail.get("effect_map_effects")
                if values is not None and effects is not None and len(values) >= 2:
                    effect_at_min_val = float(np.interp(detail["min"], values, effects))
                    effect_at_max_val = float(np.interp(detail["max"], values, effects))
                    diff_effect = effect_at_max_val - effect_at_min_val
                    if math.isclose(diff_effect, 0.0, abs_tol=1e-9):
                        detail["direction"] = 0
                        detail["direction_label"] = "Riesgo sin tendencia global"
                    elif diff_effect > 0:
                        detail["direction"] = 1
                        detail["direction_label"] = "Riesgo ↑ al subir (global)"
                    else:
                        detail["direction"] = -1
                        detail["direction_label"] = "Riesgo ↓ al subir (global)"
                else:
                    detail["direction"] = 0
                    detail["direction_label"] = "Riesgo no evaluado"

                effect_min_val = float(detail.get("effect_min", 0.0))
                effect_max_val = float(detail.get("effect_max", effect_min_val + detail["effect_range"]))
                try:
                    detail["hr_range"] = math.exp(effect_max_val - effect_min_val) if detail["effect_range"] > 0 else float('nan')
                except Exception:
                    detail["hr_range"] = float('nan')
            else:
                detail["direction"] = 1 if detail["coef"] >= 0 else -1
                detail["hr_unit"] = math.exp(detail["coef"])
                if detail["direction"] >= 0:
                    baseline_val = detail["min"]
                    extreme_val = detail["max"]
                else:
                    baseline_val = detail["max"]
                    extreme_val = detail["min"]
                effect_min_val = float(detail["coef"] * baseline_val)
                effect_max_val = float(detail["coef"] * extreme_val)

            detail["effect_baseline"] = float(effect_min_val)
            detail["effect_extreme"] = float(effect_max_val)
            lp_baseline += float(effect_min_val)

        total_effect_range = float(total_effect_range)
        x_limit = max(max_points_scale, total_points_scaled) * 1.05
        num_terms = len(term_details)

        try:
            lp_values_array = np.dot(x_design.values, params.values)
        except Exception:
            lp_values_array = None

        lp_axis_min = lp_baseline
        lp_axis_max = lp_baseline + total_effect_range

        if lp_values_array is not None and len(lp_values_array) > 0:
            lp_axis_min = min(lp_axis_min, float(np.nanmin(lp_values_array)))
            lp_axis_max = max(lp_axis_max, float(np.nanmax(lp_values_array)))

        if lp_axis_min < lp_baseline - 1e-9:
            self.log("Nomograma: se detectaron predictores menores a la referencia mínima; se recorta la escala inferior.", "WARN")
            lp_axis_min = lp_baseline

        if math.isclose(lp_axis_min, lp_axis_max, abs_tol=1e-9):
            lp_axis_max = lp_axis_min + 1.0

        def _lp_to_points(lp_value):
            return (lp_value - lp_baseline) * points_factor

        baseline_survival_df = getattr(model_obj, "baseline_survival_", None)
        baseline_times = None
        baseline_surv_values = None
        if isinstance(baseline_survival_df, pd.DataFrame) and not baseline_survival_df.empty:
            try:
                idx_numeric = pd.to_numeric(baseline_survival_df.index, errors='coerce')
                surv_values = baseline_survival_df.iloc[:, 0].astype(float)
                valid_mask = idx_numeric.notna() & surv_values.notna()
                idx_numeric = idx_numeric[valid_mask]
                surv_values = surv_values[valid_mask]
                if not idx_numeric.empty:
                    order = np.argsort(idx_numeric.to_numpy(dtype=float))
                    baseline_times = idx_numeric.to_numpy(dtype=float)[order]
                    baseline_surv_values = surv_values.to_numpy(dtype=float)[order]
            except Exception as surv_err:
                baseline_times = None
                baseline_surv_values = None
                self.log(f"Nomograma: error procesando supervivencia base: {surv_err}", "WARN")

        def _baseline_survival_at(time_value):
            if baseline_times is None or baseline_surv_values is None or len(baseline_times) == 0:
                return None
            if time_value <= baseline_times[0]:
                return float(baseline_surv_values[0])
            if time_value >= baseline_times[-1]:
                return float(baseline_surv_values[-1])
            return float(np.interp(time_value, baseline_times, baseline_surv_values))

        def _survival_from_lp(lp_value, s0_value):
            try:
                return float(np.clip(s0_value ** math.exp(lp_value), 0.0, 1.0))
            except Exception:
                return None

        def _points_for_probability(prob_value, s0_value):
            if prob_value <= 0.0 or prob_value >= 1.0:
                return None
            if s0_value is None or s0_value <= 0.0 or s0_value >= 1.0:
                return None
            ratio = math.log(prob_value) / math.log(s0_value)
            if ratio <= 0.0:
                return None
            lp_val = math.log(ratio)
            return _lp_to_points(lp_val)

        def fmt_num(value):
            if value is None or (isinstance(value, float) and math.isnan(value)):
                return "NA"
            abs_val = abs(value)
            if abs_val >= 1000 or (abs_val > 0 and abs_val < 0.01):
                return f"{value:.2e}"
            if abs_val >= 100:
                return f"{value:.0f}"
            if abs_val >= 10:
                return f"{value:.1f}"
            return f"{value:.2f}"

        def fmt_prob(prob):
            if prob is None:
                return "NA"
            if prob >= 0.995:
                return f"{prob:.3f}"
            if prob >= 0.1:
                return f"{prob:.2f}"
            return f"{prob:.2f}"

        survival_axes_data = []
        survival_candidates = [0.99, 0.97, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.5, 0.25, 0.1, 0.05, 0.03, 0.01]
        if survival_time_points and baseline_times is not None and baseline_surv_values is not None:
            for time_val in survival_time_points:
                s0_val = _baseline_survival_at(time_val)
                if s0_val is None or s0_val <= 0.0 or s0_val >= 1.0:
                    self.log(f"Nomograma: supervivencia base no válida para t={time_val:g}.", "WARN")
                    continue
                surv_high = _survival_from_lp(lp_baseline, s0_val)
                surv_low = _survival_from_lp(lp_axis_max, s0_val)
                if surv_high is None or surv_low is None:
                    continue
                p_max = max(surv_high, surv_low)
                p_min = min(surv_high, surv_low)
                if math.isclose(p_max, p_min, rel_tol=1e-6, abs_tol=1e-6):
                    self.log(f"Nomograma: rango de supervivencia muy estrecho para t={time_val:g}.", "INFO")
                    continue
                tick_probs = [p for p in survival_candidates if p_min - 1e-6 <= p <= p_max + 1e-6]
                tick_probs.extend([p_min, p_max])
                tick_probs = sorted({round(p, 6) for p in tick_probs}, reverse=True)
                tick_entries = []
                for prob_value in tick_probs:
                    points_val = _points_for_probability(prob_value, s0_val)
                    if points_val is None:
                        continue
                    if points_val < -0.5 or points_val > x_limit + 0.5:
                        continue
                    tick_entries.append({"prob": prob_value, "points": points_val})
                if len(tick_entries) >= 2:
                    survival_axes_data.append({
                        "time": time_val,
                        "ticks": tick_entries
                    })
        else:
            if survival_time_points:
                self.log("Nomograma: no se pudo generar ejes de supervivencia por falta de supervivencia base.", "WARN")

        row_count = 1 + num_terms + 1 + 1 + len(survival_axes_data)
        row_spacing = 1.1
        height_units = (row_count + 2) * row_spacing
        fig_height = max(6.0, height_units * 0.85)
        fig, ax = plt.subplots(figsize=(10, fig_height))
        fig.subplots_adjust(left=0.32, right=0.94, top=0.9, bottom=0.08)
        ax.set_xlim(0, x_limit)
        ax.set_ylim(-2 * row_spacing, height_units)
        ax.axis('off')

        label_transform = mtransforms.blended_transform_factory(ax.transAxes, ax.transData)

        current_y = height_units - row_spacing

        # Eje superior de puntos
        ax.hlines(current_y, 0, max_points_scale, color='black', linewidth=1.3)
        for tick in np.linspace(0, max_points_scale, 6):
            ax.vlines(tick, current_y, current_y + 0.2, color='black', linewidth=1)
            ax.text(tick, current_y + 0.32, f"{tick:.0f}", ha='center', va='bottom', fontsize=9)
        ax.text(max_points_scale / 2.0, current_y + 0.6, "Puntos", ha='center', va='bottom', fontsize=11, fontweight='bold')
        current_y -= row_spacing

        # Dibujar cada término
        for detail in term_details:
            y_pos = current_y
            current_y -= row_spacing

            points_max = detail["points_max"]
            ax.hlines(y_pos, 0, x_limit, color='#d9d9d9', linewidth=0.8)
            ax.hlines(y_pos, 0, points_max, color='#1f77b4', linewidth=2.0)

            label_main = self._format_term_for_nomogram(detail["term"])
            ax.text(-0.05, y_pos + 0.05, label_main, transform=label_transform, ha='right', va='center', fontweight='bold')

            tick_entries = detail.get("ticks_info") or []
            if not tick_entries:
                if detail.get("is_spline_group"):
                    fallback_values = np.linspace(detail["min"], detail["max"], num=5)
                else:
                    series_vals = detail.get("series")
                    if series_vals is not None:
                        unique_vals = np.unique(series_vals.values)
                        if unique_vals.size <= 2:
                            fallback_values = unique_vals
                        else:
                            fallback_values = np.linspace(detail["min"], detail["max"], num=5)
                    else:
                        fallback_values = np.linspace(detail["min"], detail["max"], num=5)
                tick_entries = [{"value": float(val), "labels": []} for val in fallback_values]

            value_range = detail["max"] - detail["min"]
            for tick_entry in tick_entries:
                tick_val = tick_entry.get("value")
                if tick_val is None:
                    continue

                if detail.get("is_spline_group"):
                    values = detail.get("effect_map_values")
                    effects = detail.get("effect_map_effects")
                    if values is None or effects is None or len(values) == 0:
                        continue
                    if math.isclose(detail["effect_range"], 0.0, abs_tol=1e-12):
                        continue
                    interpolated_effect = float(np.interp(tick_val, values, effects))
                    tick_point = ((interpolated_effect - detail.get("effect_min", 0.0)) / detail["effect_range"]) * points_max
                else:
                    if math.isclose(value_range, 0.0, abs_tol=1e-12):
                        continue
                    direction = detail.get("direction", 1)
                    if direction < 0:
                        frac = (detail["max"] - tick_val) / value_range
                    else:
                        frac = (tick_val - detail["min"]) / value_range
                    frac = np.clip(frac, 0.0, 1.0)
                    tick_point = frac * points_max

                tick_point = float(np.clip(tick_point, 0.0, points_max))
                ax.vlines(tick_point, y_pos - 0.15, y_pos + 0.15, color='#1f77b4', linewidth=1)

                label_lines = [fmt_num(tick_val)]
                extra_labels = tick_entry.get("labels", [])
                if extra_labels:
                    label_lines.append("/".join(extra_labels))
                label_text = "\n".join(label_lines)
                ax.text(tick_point, y_pos - 0.38, label_text, ha='center', va='top', fontsize=8)

        # Eje de puntos totales
        total_y = current_y
        current_y -= row_spacing
        ax.hlines(total_y, 0, total_points_scaled, color='black', linewidth=1.3)
        ax.text(-0.05, total_y + 0.05, "Total de puntos", transform=label_transform, ha='right', va='center', fontweight='bold')
        if total_points_scaled > 0:
            for tick in np.linspace(0, total_points_scaled, 6):
                ax.vlines(tick, total_y, total_y - 0.2, color='black', linewidth=1)
                ax.text(tick, total_y - 0.32, f"{tick:.0f}", ha='center', va='top', fontsize=8)

        # Eje de predictor lineal
        linear_y = current_y
        current_y -= row_spacing
        lp_line_start = max(0.0, _lp_to_points(lp_axis_min))
        lp_line_end = _lp_to_points(lp_axis_max)
        ax.hlines(linear_y, lp_line_start, min(lp_line_end, x_limit), color='black', linewidth=1.3)
        ax.text(-0.05, linear_y + 0.05, "Predictor lineal", transform=label_transform, ha='right', va='center', fontweight='bold')
        for lp_val in np.linspace(lp_axis_min, lp_axis_max, 6):
            tick_point = _lp_to_points(lp_val)
            if tick_point < -0.5 or tick_point > x_limit + 0.5:
                continue
            ax.vlines(tick_point, linear_y, linear_y - 0.2, color='black', linewidth=1)
            ax.text(tick_point, linear_y - 0.32, f"{lp_val:.2f}", ha='center', va='top', fontsize=8)

        # Ejes de probabilidades de supervivencia
        for surviv_axis in survival_axes_data:
            surv_y = current_y
            current_y -= row_spacing
            ax.hlines(surv_y, 0, total_points_scaled, color='black', linewidth=1.2)
            ax.text(-0.05, surv_y + 0.05, f"Probabilidad de supervivencia a {fmt_num(surviv_axis['time'])}",
                    transform=label_transform, ha='right', va='center', fontweight='bold')

            for entry in surviv_axis['ticks']:
                tick_point = entry['points']
                if tick_point < -0.5 or tick_point > x_limit + 0.5:
                    continue
                ax.vlines(tick_point, surv_y, surv_y + 0.2, color='black', linewidth=1)
                ax.text(tick_point, surv_y + 0.32, fmt_prob(entry['prob']), ha='center', va='bottom', fontsize=8)

        fig.suptitle(f"Nomograma - {model_name}", fontsize=13, fontweight='bold')
        fig.tight_layout(rect=[0, 0.02, 1, 0.94])

        self._create_plot_window(fig, f"Nomograma: {model_name}")
        self.log(f"Nomograma generado para '{model_name}' con {num_terms} término(s).", "SUCCESS")

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
            sch_results_df = md_sch.get("schoenfeld_results")
            ph_test_summary_df = md_sch.get("proportional_hazard_test_summary")

            for idx, cov_name_s in enumerate(covariate_names):
                if idx < len(axes_s_flat):
                    ax_s_curr = axes_s_flat[idx]
                    ax_s_curr.plot(scaled_residuals.index, scaled_residuals[cov_name_s], 
                                   linestyle='none', marker='o', markersize=3, alpha=0.6)
                    ax_s_curr.axhline(0, color='grey', linestyle='--', lw=0.8)

                    # Retrieve and format per-term p-value using normalized helpers
                    p_val = extract_schoenfeld_p_value(sch_results_df, cov_name_s)
                    if p_val is None:
                        p_val = extract_schoenfeld_p_value(ph_test_summary_df, cov_name_s)
                    p_val_str = format_p_value(p_val) if p_val is not None else "N/A"

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

    def show_cumulative_baseline_hazard(self):
        if not self._check_model_selected_and_valid(): return
        md_bh = self.selected_model_in_treeview; cph_bh = md_bh.get('model'); name_bh = md_bh.get('model_name', 'N/A')
        try:
            fig_bh, ax_bh = plt.subplots(figsize=(10,6));
            cph_bh.baseline_cumulative_hazard_.plot(ax=ax_bh, legend=False)
            opts_bh = self.current_plot_options.copy()
            opts_bh['title'] = opts_bh.get('title') or f"Riesgo Acumulado Base H0(t) ({name_bh})"
            opts_bh['xlabel'] = opts_bh.get('xlabel') or f"Tiempo ({md_bh.get('time_col_for_model','T')})"
            opts_bh['ylabel'] = opts_bh.get('ylabel') or "H0(t)"
            apply_plot_options(ax_bh, opts_bh, self.log)
            self._create_plot_window(fig_bh, f"Riesgo Acum. Base: {name_bh}")
        except Exception as e_bh: self.log(f"Error Riesgo Acum. Base '{name_bh}': {e_bh}", "ERROR"); messagebox.showerror("Error Gráfico", f"Error Riesgo Acum. Base:\n{e_bh}", parent=self.parent_for_dialogs)

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
            formatted_labels_fp = [self._format_term_for_display(idx) for idx in plot_df_fp.index]
            ax_fp.set_yticks(y_pos_fp); ax_fp.set_yticklabels(formatted_labels_fp); ax_fp.invert_yaxis()
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
            if full_patsy_formula:
                # Use a specific regex to find only the original variable names inside Q('')
                orig_vars_ask_pred = sorted(list(set(re.findall(r"Q\('([^']+)'\)", full_patsy_formula))))
                self.log(f"Variables para predicción extraídas de Q(): {orig_vars_ask_pred}", "INFO")
            else:
                orig_vars_ask_pred = []

        if not orig_vars_ask_pred and md_pred.get('covariates_processed', []):
            self.log("No se pudieron determinar variables originales. UI de predicción puede ser incompleta o fallar.", "WARN")

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
        ttk.Button(frame_btns_pred_diag,text="Predecir y Mostrar Curva",command=lambda: self._perform_prediction_and_plot(pred_diag,md_pred,{k: v for k, v in entries_pred.items()},type_var_pred_ui.get(),times_str_var_pred_ui.get())).pack(side=tk.LEFT,padx=10)
        ttk.Button(frame_btns_pred_diag,text="Cancelar",command=pred_diag.destroy).pack(side=tk.RIGHT,padx=10)

    def _parse_prediction_input_and_generate_scenarios(self, entries_dict, original_data_context):
        from itertools import product

        parsed_vars = {}
        range_vars = {} # To store which variables are ranges, e.g., {'VarA': '1-3'}

        for var_name, str_var_obj in entries_dict.items():
            val_str = str_var_obj.get().strip()
            if not val_str:
                self.log(f"Valor faltante para '{var_name}'.", "ERROR")
                messagebox.showerror("Valor Faltante", f"El valor para '{var_name}' no puede estar vacío.")
                return None, None

            is_numeric = pd.api.types.is_numeric_dtype(original_data_context.get(var_name))

            if is_numeric and '-' in val_str and ',' not in val_str:
                try:
                    start, end = map(float, val_str.split('-'))
                    if start > end: start, end = end, start
                    if start != int(start) or end != int(end):
                        raise ValueError("Los rangos solo se soportan para números enteros.")

                    parsed_vars[var_name] = list(np.arange(int(start), int(end) + 1))
                    range_vars[var_name] = val_str # Mark as a range variable
                    self.log(f"Variable '{var_name}' procesada como rango: {parsed_vars[var_name]}", "DEBUG")
                except ValueError as e:
                    self.log(f"Rango inválido para '{var_name}': {e}. Tratado como literal.", "WARN")
                    parsed_vars[var_name] = [val_str]

            elif ',' in val_str:
                values = [v.strip() for v in val_str.split(',')]
                if is_numeric:
                    try:
                        parsed_vars[var_name] = [float(v) for v in values]
                    except ValueError:
                        self.log(f"Mezcla de tipos en lista para '{var_name}'. Tratados como texto.", "WARN")
                        parsed_vars[var_name] = values
                else:
                    parsed_vars[var_name] = values

            else:
                if is_numeric:
                    try:
                        parsed_vars[var_name] = [float(val_str)]
                    except ValueError:
                        parsed_vars[var_name] = [val_str]
                else:
                    parsed_vars[var_name] = [val_str]

        # --- Generate Scenarios ---
        if not parsed_vars:
            return [], {}

        keys = list(parsed_vars.keys())
        value_lists = list(parsed_vars.values())

        scenarios = [dict(zip(keys, combo)) for combo in product(*value_lists)]

        self.log(f"Generados {len(scenarios)} escenarios de predicción. Variables de rango: {range_vars}", "INFO")
        return scenarios, range_vars

    def _perform_prediction_and_plot(self, dialog_pred_ref, md_dict_for_pred, entries_dict_for_pred, type_ui_pred, times_str_ui_pred):
        cph_model_for_pred = md_dict_for_pred.get('model')
        name_for_pred = md_dict_for_pred.get('model_name', 'N/A')

        scenarios, range_vars = self._parse_prediction_input_and_generate_scenarios(entries_dict_for_pred, self.data)
        if scenarios is None or not scenarios:
            self.log("No se generaron escenarios para la predicción.", "WARN")
            if scenarios is not None:
                messagebox.showwarning("Sin Escenarios", "No se generaron escenarios válidos.", parent=dialog_pred_ref)
            return

        times_list_pred = []
        if times_str_ui_pred.strip():
            try:
                times_list_pred = [float(t.strip()) for t in times_str_ui_pred.split(',') if t.strip()]
            except ValueError:
                messagebox.showerror("Error Tiempos", "Tiempos inválidos.", parent=dialog_pred_ref)
                return

        fig_curve_pred, ax_curve_pred = plt.subplots(figsize=(10, 6))

        # 1. Calculate all individual curves
        all_curves = {}
        self.log(f"Calculando curvas para {len(scenarios)} escenarios...", "DEBUG")
        for scenario in scenarios:
            df_input = pd.DataFrame([scenario])
            try:
                # Scaling logic
                fitted_scaler = md_dict_for_pred.get("fitted_scaler_object")
                scaled_columns = md_dict_for_pred.get("scaled_columns_info", [])
                if fitted_scaler and scaled_columns:
                    numeric_cols = [col for col in scaled_columns if col in df_input.columns]
                    if numeric_cols:
                        df_input[numeric_cols] = df_input[numeric_cols].apply(pd.to_numeric, errors='coerce')
                        df_input[numeric_cols] = fitted_scaler.transform(df_input[numeric_cols])

                # Prediction
                pred_df = None
                if type_ui_pred == "Supervivencia":
                    pred_df = cph_model_for_pred.predict_survival_function(df_input)
                elif type_ui_pred == "Riesgo":
                    pred_df = cph_model_for_pred.predict_cumulative_hazard(df_input)
                else: # ProbEventoAcum
                    pred_df = 1 - cph_model_for_pred.predict_survival_function(df_input)

                if pred_df is not None and not pred_df.empty:
                    self.log(f"Curva calculada para escenario: {scenario}", "DEBUG")
                    all_curves[tuple(sorted(scenario.items()))] = pred_df
                else:
                    self.log(f"pred_df fue None o vacío para escenario: {scenario}", "WARN")
            except Exception as e:
                self.log(f"Error en predicción para escenario {scenario}: {e}", "ERROR")
                traceback.print_exc(limit=2)
                messagebox.showerror("Error en Predicción", f"Falló para:\n{scenario}\n\nError: {e}", parent=dialog_pred_ref)
                return
        self.log(f"Se calcularon {len(all_curves)} curvas individuales.", "DEBUG")

        # 2. Group scenarios and prepare curves for plotting
        final_curves_to_plot = {}
        scenarios_in_groups = set()

        for range_var, range_str in range_vars.items():
            def get_base_scenario_tuple(s_dict):
                s_copy = s_dict.copy()
                if range_var in s_copy: del s_copy[range_var]
                return tuple(sorted(s_copy.items()))

            grouped_by_base = {}
            for scen_dict in scenarios:
                if range_var in scen_dict:
                    base_tuple = get_base_scenario_tuple(scen_dict)
                    if base_tuple not in grouped_by_base:
                        grouped_by_base[base_tuple] = []
                    grouped_by_base[base_tuple].append(scen_dict)

            for base_tuple, group_scenarios in grouped_by_base.items():
                curves_to_avg = [all_curves[tuple(sorted(s.items()))] for s in group_scenarios if tuple(sorted(s.items())) in all_curves]

                if not curves_to_avg: continue

                for s in group_scenarios: scenarios_in_groups.add(tuple(sorted(s.items())))

                all_indices = pd.concat(curves_to_avg).index.unique().sort_values()
                fill_val = 1.0 if type_ui_pred == "Supervivencia" else 0.0
                reindexed = [c.reindex(all_indices, method='ffill').fillna(fill_val) for c in curves_to_avg]
                avg_curve = pd.concat(reindexed).groupby(level=0).mean()

                label_dict = dict(base_tuple)
                label_dict[range_var] = f"Grupo({range_str})"
                label = ", ".join([f"{k}={v}" for k, v in sorted(label_dict.items())])
                final_curves_to_plot[label] = avg_curve

        for scenario_tuple, curve in all_curves.items():
            if scenario_tuple not in scenarios_in_groups:
                label = ", ".join([f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}" for k,v in sorted(scenario_tuple)])
                final_curves_to_plot[label] = curve

        # 3. Plotting
        self.log(f"Total de curvas a graficar (individuales + grupos): {len(final_curves_to_plot)}", "DEBUG")
        if not final_curves_to_plot:
            self.log("No se generaron curvas finales para graficar.", "WARN")
            messagebox.showwarning("Sin Gráficos", "No se pudieron generar curvas de predicción.", parent=dialog_pred_ref)
            return

        dialog_legends = EditPredictionLegendsDialog(dialog_pred_ref, final_curves_to_plot)
        if dialog_legends.result is None:
            self.log("Edición de leyendas cancelada por el usuario.", "INFO")
            return

        new_labels = dialog_legends.result

        # Rebuild the final_curves_to_plot with the new labels
        final_curves_to_plot_labeled = {}
        for original_label, curve in final_curves_to_plot.items():
            new_label = new_labels.get(original_label, original_label) # Fallback to original if something goes wrong
            final_curves_to_plot_labeled[new_label] = curve

        opts_curve_pred = self.current_plot_options.copy()
        cmap_name = opts_curve_pred.get('cmap', 'viridis')
        try:
            cmap_obj = plt.get_cmap(cmap_name)
        except ValueError:
            self.log(f"Paleta '{cmap_name}' no reconocida. Se usará 'viridis'.", "WARN")
            cmap_obj = plt.get_cmap('viridis')
        colors = cmap_obj(np.linspace(0, 1, len(final_curves_to_plot_labeled)))
        results_text_pred_list = []

        for i, (label, pred_df) in enumerate(final_curves_to_plot_labeled.items()):
            # Ensure pred_df is a DataFrame and has a name for the legend
            if isinstance(pred_df, pd.Series):
                pred_df = pred_df.to_frame(name=label)
            elif isinstance(pred_df, pd.DataFrame) and pred_df.columns[0] != label:
                pred_df.columns = [label]

            pred_df.plot(ax=ax_curve_pred, legend=False, drawstyle='steps-post', color=colors[i], label=label)
            if times_list_pred:
                label_prefix = {"Supervivencia": "S", "Riesgo": "H", "ProbEventoAcum": "1-S"}[type_ui_pred]
                results_for_scenario = [f"Curva: {label}"]
                for t_val in times_list_pred:
                    val_str = f"{np.interp(t_val, pred_df.index, pred_df.iloc[:,0]):.3f}" if t_val >= pred_df.index.min() and t_val <= pred_df.index.max() else "N/A"
                    if val_str != "N/A":
                        ax_curve_pred.scatter([t_val], [float(val_str)], marker='o', color=colors[i], s=40, zorder=5)
                    results_for_scenario.append(f"  {label_prefix}(t={t_val}|X) = {val_str}")
                results_text_pred_list.append("\n".join(results_for_scenario))

        # 4. Finalize plot
        title_map = {"Supervivencia": "Pred. Prob. Supervivencia", "Riesgo": "Pred. Riesgo Acumulado", "ProbEventoAcum": "Pred. Prob. Evento Acumulado"}
        title_curve_pred = f"{title_map.get(type_ui_pred)} ({name_for_pred})"
        ylabel_map = {"Supervivencia": "S(t|X)", "Riesgo": "H(t|X)", "ProbEventoAcum": "1 - S(t|X)"}
        ax_curve_pred.set_ylabel(ylabel_map.get(type_ui_pred))

        opts_curve_pred['title'] = opts_curve_pred.get('title') or title_curve_pred
        opts_curve_pred['xlabel'] = opts_curve_pred.get('xlabel') or f"Tiempo ({md_dict_for_pred.get('time_col_for_model','T')})"
        grid_setting = opts_curve_pred.get('grid')
        apply_plot_options(ax_curve_pred, opts_curve_pred, self.log)
        grid_active = coerce_bool_option(grid_setting, default=False)
        ax_curve_pred.set_axisbelow(True)
        ax_curve_pred.grid(grid_active, which='both', linestyle=':', alpha=0.6)

        ax_curve_pred.legend(fontsize='small')

        self._create_plot_window(fig_curve_pred, title_curve_pred)

        if results_text_pred_list:
            ModelSummaryWindow(dialog_pred_ref, "Resultados de Predicción Puntual", "\n\n".join(results_text_pred_list))
        elif not times_list_pred:
            messagebox.showinfo("Predicción Exitosa", "Curva(s) de predicción generada(s).", parent=dialog_pred_ref)


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

    def generate_brier_plot(self):
        if not self.selected_model_in_treeview:
            messagebox.showwarning("Sin Modelo", "Seleccione un modelo para generar la gráfica de Brier / IBS.", parent=self.parent_for_dialogs)
            return

        md_brier = self.selected_model_in_treeview
        model_name = md_brier.get('model_name', 'N/A')
        metrics_brier = md_brier.get('metrics', {}) or {}
        brier_curve_df = md_brier.get('brier_curve_df')
        fig_brier = None

        try:
            fig_brier, ax_brier = plt.subplots(figsize=(8, 5))
            plotted = False

            if isinstance(brier_curve_df, pd.DataFrame) and not brier_curve_df.empty and {'time', 'brier_score'}.issubset(brier_curve_df.columns):
                curve_df = brier_curve_df.dropna(subset=['time', 'brier_score']).copy().sort_values('time')
                if not curve_df.empty:
                    times = curve_df['time'].to_numpy(dtype=float)
                    scores = curve_df['brier_score'].to_numpy(dtype=float)
                    ax_brier.plot(times, scores, color="#8b5cf6", linewidth=2.2, label="Brier(t)")
                    ax_brier.fill_between(times, scores, 0, color="#8b5cf6", alpha=0.16)
                    eval_time = md_brier.get('brier_eval_time')
                    if eval_time is not None and np.isfinite(eval_time):
                        ax_brier.axvline(float(eval_time), color="#475569", linestyle="--", linewidth=1.2, label=f"t≈{float(eval_time):.2f}")
                    plotted = True

            if not plotted:
                fallback_points = []
                for lbl in ("Q25", "Q50", "Q75"):
                    val = metrics_brier.get(f"Brier@{lbl}")
                    if val is not None and pd.notna(val):
                        fallback_points.append((lbl, float(val)))
                if not fallback_points:
                    raise ValueError("El modelo seleccionado no tiene datos Brier / IBS disponibles. Ejecuta holdout primero.")
                labels = [item[0] for item in fallback_points]
                values = [item[1] for item in fallback_points]
                ax_brier.bar(labels, values, color="#8b5cf6", alpha=0.8, edgecolor="#4c1d95", label="Brier@Q")
                ax_brier.set_xlabel("Horizonte temporal")
                plotted = True

            ibs_val = metrics_brier.get('IBS')
            title_txt = f"Curva Brier / IBS - {model_name}"
            if ibs_val is not None and pd.notna(ibs_val):
                title_txt += f"\nIBS={float(ibs_val):.4f}"

            plot_opts_brier = self.current_plot_options.copy()
            plot_opts_brier['title'] = plot_opts_brier.get('title', title_txt)
            plot_opts_brier['xlabel'] = plot_opts_brier.get('xlabel', ax_brier.get_xlabel() or 'Tiempo')
            plot_opts_brier['ylabel'] = plot_opts_brier.get('ylabel', 'Brier score')
            apply_plot_options(ax_brier, plot_opts_brier, self.log)
            ax_brier.grid(True, alpha=0.2)
            handles_brier, labels_brier = ax_brier.get_legend_handles_labels()
            if handles_brier:
                ax_brier.legend(loc='best', fontsize=8)

            self._create_plot_window(fig_brier, f"Brier / IBS: {model_name}")
            self.log(f"Gráfico Brier / IBS generado para '{model_name}'.", "SUCCESS")
        except Exception as e_brier_plot:
            if fig_brier is not None:
                plt.close(fig_brier)
            self.log(f"Error al generar gráfico Brier / IBS: {e_brier_plot}", "ERROR")
            messagebox.showerror("Error de Gráfico", f"No se pudo generar la gráfica Brier / IBS:\n{e_brier_plot}", parent=self.parent_for_dialogs)

    def show_variable_impact_plot(self):
        selected_models = self.selected_models_in_treeview or ([] if self.selected_model_in_treeview is None else [self.selected_model_in_treeview])
        if not selected_models:
            messagebox.showinfo("Sin Modelo", "Seleccione uno o más modelos para generar el gráfico de efecto.", parent=self.parent_for_dialogs)
            return

        def _collect_candidate_vars(model_dict, training_df):
            candidates = set()
            model_obj_local = model_dict.get('model')
            time_col_local = model_dict.get('time_col_for_model')
            event_col_local = model_dict.get('event_col_for_model')

            if hasattr(model_obj_local, 'formula') and model_obj_local.formula:
                formula_terms_local = re.findall(r"Q\('([^']+)'\)|([a-zA-Z_][a-zA-Z0-9_]*)", model_obj_local.formula)
                for q_term, raw_term in formula_terms_local:
                    term_to_add = q_term if q_term else raw_term
                    if term_to_add and term_to_add not in ['Intercept', '0', '1'] and not any(func in term_to_add for func in ['cr(', 'bs(', 'C(']):
                        if term_to_add in training_df.columns:
                            candidates.add(term_to_add)

            for col_local in training_df.select_dtypes(include=np.number).columns:
                if col_local not in [time_col_local, event_col_local]:
                    candidates.add(col_local)

            return candidates

        model_wrappers = []
        combined_options = []

        for md in selected_models:
            model_obj = md.get('model')
            if not (model_obj and isinstance(model_obj, CoxPHFitter)):
                self.log(f"Modelo seleccionado sin objeto Cox válido: {md.get('model_name', 'N/A')}.", "WARN")
                continue

            original_training_data = md.get('_df_for_fit_main_INTERNAL_USE')
            if original_training_data is None or original_training_data.empty:
                self.log(f"Datos de entrenamiento no disponibles para '{md.get('model_name', 'N/A')}'. Se omite en comparación.", "WARN")
                continue

            candidate_vars = _collect_candidate_vars(md, original_training_data)
            if not candidate_vars:
                self.log(f"Modelo '{md.get('model_name', 'N/A')}' sin covariables numéricas adecuadas para graficar.", "WARN")
                continue

            wrapper = {
                "model_dict": md,
                "model_obj": model_obj,
                "training_df": original_training_data,
                "time_col": md.get('time_col_for_model'),
                "event_col": md.get('event_col_for_model'),
                "spline_metadata": md.get('spline_basis_metadata', {}) or {},
                "display_name": md.get('custom_model_name', md.get('model_name', 'Modelo Cox')),
            }
            model_wrappers.append(wrapper)

            for var_name in sorted(candidate_vars):
                combined_options.append({
                    "wrapper": wrapper,
                    "covariate": var_name,
                    "label": f"{wrapper['display_name']} :: {var_name}"
                })

        if not combined_options:
            messagebox.showinfo(
                "Sin Covariables",
                "Ninguno de los modelos seleccionados cuenta con covariables numéricas aptas para este gráfico.",
                parent=self.parent_for_dialogs
            )
            return

        dialog = Toplevel(self.parent_for_dialogs)
        dialog.title("Seleccionar Covariable(s) y Modelo(s) para Gráfico de Efecto")
        dialog.geometry("440x420")
        ttk.Label(
            dialog,
            text="Seleccione una o más combinaciones Modelo::Variable para visualizar su efecto:",
            wraplength=420
        ).pack(pady=10, padx=10)

        listbox_frame = ttk.Frame(dialog)
        listbox_frame.pack(pady=5, padx=10, fill=tk.BOTH, expand=True)
        listbox_covs_widget = Listbox(listbox_frame, selectmode=tk.EXTENDED, exportselection=False, height=10)
        for option in combined_options:
            listbox_covs_widget.insert(tk.END, option["label"])
        if combined_options:
            listbox_covs_widget.selection_set(0)

        scrollbar_y_covs = ttk.Scrollbar(listbox_frame, orient=tk.VERTICAL, command=listbox_covs_widget.yview)
        listbox_covs_widget.config(yscrollcommand=scrollbar_y_covs.set)
        scrollbar_y_covs.pack(side=tk.RIGHT, fill=tk.Y)
        listbox_covs_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scale_frame = ttk.Frame(dialog)
        scale_frame.pack(pady=5, padx=10, fill=tk.X)
        ttk.Label(scale_frame, text="Escala Eje Y:").pack(side=tk.LEFT, padx=(0,5))
        y_scale_choice_var = StringVar(value="log_hr")
        ttk.Radiobutton(scale_frame, text="Log(Hazard Ratio)", variable=y_scale_choice_var, value="log_hr").pack(side=tk.LEFT)
        ttk.Radiobutton(scale_frame, text="Hazard Ratio", variable=y_scale_choice_var, value="hr").pack(side=tk.LEFT, padx=(5,0))

        show_knots_var = BooleanVar(value=True)
        show_ci_var = BooleanVar(value=True)
        normalize_axes_var = BooleanVar(value=len(combined_options) > 1)

        options_frame = ttk.LabelFrame(dialog, text="Opciones de visualización")
        options_frame.pack(pady=5, padx=10, fill=tk.X)
        ttk.Checkbutton(options_frame, text="Mostrar nodos del spline", variable=show_knots_var).pack(anchor='w', padx=5, pady=2)
        ttk.Checkbutton(options_frame, text="Mostrar banda de IC 95%", variable=show_ci_var).pack(anchor='w', padx=5, pady=2)
        ttk.Checkbutton(
            options_frame,
            text="Normalizar eje X (escala 0-1 para comparar)",
            variable=normalize_axes_var
        ).pack(anchor='w', padx=5, pady=2)

        chosen_items = []
        chosen_y_scale = "log_hr"
        plot_show_knots = True
        plot_show_ci = True
        normalize_axes = normalize_axes_var.get()

        def on_ok():
            nonlocal chosen_items, chosen_y_scale, plot_show_knots, plot_show_ci, normalize_axes
            selections = listbox_covs_widget.curselection()
            if not selections:
                messagebox.showwarning("Selección Requerida", "Seleccione al menos una combinación Modelo::Variable.", parent=dialog)
                return
            chosen_items.extend(combined_options[i] for i in selections)
            chosen_y_scale = y_scale_choice_var.get()
            plot_show_knots = bool(show_knots_var.get())
            plot_show_ci = bool(show_ci_var.get())
            normalize_axes = bool(normalize_axes_var.get())
            dialog.destroy()

        def on_cancel():
            dialog.destroy()

        button_frame = ttk.Frame(dialog)
        button_frame.pack(pady=10)
        ttk.Button(button_frame, text="Aceptar", command=on_ok).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Cancelar", command=on_cancel).pack(side=tk.RIGHT, padx=5)

        dialog.transient(self.parent_for_dialogs)
        dialog.grab_set()
        self.parent_for_dialogs.wait_window(dialog)

        if not chosen_items:
            self.log("Selección de covariable(s) para gráfico de efecto cancelada o vacía.", "INFO")
            return

        label_list = [item["label"] for item in chosen_items]
        self.log(
            f"Generando datos para gráfico de efecto. Selecciones: {', '.join(label_list)}, Escala Y: {chosen_y_scale}, Mostrar nodos={plot_show_knots}, Mostrar IC={plot_show_ci}",
            "INFO"
        )

        plot_data_list = []
        apply_normalization = bool(normalize_axes and len(chosen_items) > 1)
        self.log(f"Normalización de eje X habilitada: {apply_normalization}", "DEBUG")
        fig_effect, ax_effect = plt.subplots(figsize=(10, 6))

        for selection in chosen_items:
            wrapper = selection["wrapper"]
            current_cov_to_plot = selection["covariate"]
            current_model_display = wrapper["display_name"]
            cph_model_vip = wrapper["model_obj"]
            original_training_data = wrapper["training_df"]
            spline_metadata_for_model = wrapper["spline_metadata"]

            self.log(f"Preparando datos para: {current_model_display} :: {current_cov_to_plot}", "DEBUG")

            if current_cov_to_plot not in original_training_data.columns:
                self.log(f"Advertencia: La covariable '{current_cov_to_plot}' no está en los datos de entrenamiento originales. Saltando.", "WARN")
                messagebox.showwarning(
                    "Variable no Encontrada",
                    f"La covariable '{current_cov_to_plot}' no se encontró en los datos originales del modelo '{current_model_display}'.",
                    parent=self.parent_for_dialogs
                )
                continue

            spline_info_current = spline_metadata_for_model.get(current_cov_to_plot, {}) or {}
            spline_type_current = (spline_info_current.get('spline_type') or "").lower()

            min_val = original_training_data[current_cov_to_plot].min()
            max_val = original_training_data[current_cov_to_plot].max()
            is_numeric_cov = pd.api.types.is_numeric_dtype(original_training_data[current_cov_to_plot])

            if not is_numeric_cov and apply_normalization:
                self.log(f"Advertencia: Normalización no aplicable a variable no numérica '{current_cov_to_plot}' en gráfico multivariable. Se omite para comparación.", "WARN")
                messagebox.showwarning(
                    "Variable No Numérica",
                    f"La variable '{current_cov_to_plot}' no es numérica y no puede normalizarse junto con otras. Se omitirá.",
                    parent=self.parent_for_dialogs
                )
                continue

            x_plot_values_actual = []
            x_axis_display_values = []
            normalization_info = None

            if pd.isna(min_val) or pd.isna(max_val) or (is_numeric_cov and min_val == max_val):
                if is_numeric_cov and min_val == max_val and pd.notna(min_val):
                    self.log(f"Variable '{current_cov_to_plot}' tiene un único valor numérico ({min_val}). Usando pequeño rango.", "DEBUG")
                    delta = abs(min_val * 0.05) if min_val != 0 else 0.05
                    if delta == 0:
                        delta = 0.05
                    x_plot_values_actual = np.linspace(min_val - delta, max_val + delta, 100)
                elif not is_numeric_cov:
                    unique_vals = original_training_data[current_cov_to_plot].unique()
                    if len(unique_vals) == 1 and pd.notna(unique_vals[0]):
                        x_plot_values_actual = [unique_vals[0]] * 2
                        self.log(f"Variable '{current_cov_to_plot}' tiene un único valor categórico ('{unique_vals[0]}').", "DEBUG")
                    else:
                        self.log(f"No se pudo determinar rango para '{current_cov_to_plot}'. Saltando.", "WARN")
                        continue
                else:
                    self.log(f"No se pudo determinar rango para '{current_cov_to_plot}'. Saltando.", "WARN")
                    continue
            else:
                x_plot_values_actual = np.linspace(min_val, max_val, 100)

            extension_info = None
            uses_natural_spline = is_numeric_cov and spline_type_current == "natural"
            if uses_natural_spline and not apply_normalization and len(x_plot_values_actual) >= 2:
                boundary_knots = spline_info_current.get('boundary_knots') or []
                trained_min = min(min_val, min(boundary_knots)) if boundary_knots else min_val
                trained_max = max(max_val, max(boundary_knots)) if boundary_knots else max_val
                trained_range = trained_max - trained_min
                if trained_range > 0:
                    extension_fraction = 0.1
                    extension_amount = trained_range * extension_fraction
                    extended_min = trained_min - extension_amount
                    extended_max = trained_max + extension_amount

                    if trained_min > 0 and extended_min <= 0:
                        smallest_positive = original_training_data[current_cov_to_plot][original_training_data[current_cov_to_plot] > 0]
                        if not smallest_positive.empty:
                            floor_val = float(smallest_positive.min()) * 0.5
                            extended_min = max(floor_val, 1e-8)
                        else:
                            extended_min = max(trained_min * 0.5, 1e-8)
                    elif trained_min == 0 and extended_min < 0:
                        extended_min = 0.0

                    if extended_max <= trained_max:
                        extended_max = trained_max

                    if extended_max > extended_min:
                        x_plot_values_actual = np.linspace(extended_min, extended_max, 140)
                        extension_info = {
                            "extended_min": float(extended_min),
                            "extended_max": float(extended_max),
                            "trained_min": float(trained_min),
                            "trained_max": float(trained_max)
                        }
                        self.log(
                            f"RCS '{current_cov_to_plot}': rango de graficación extendido a [{extended_min:.3g}, {extended_max:.3g}] para resaltar colas lineales.",
                            "DEBUG"
                        )

            if apply_normalization and is_numeric_cov:
                if max_val == min_val:
                    x_axis_display_values = np.zeros_like(x_plot_values_actual) if min_val == 0 else np.full_like(x_plot_values_actual, 0.5)
                else:
                    x_axis_display_values = (x_plot_values_actual - min_val) / (max_val - min_val)
                normalization_info = f"{current_cov_to_plot} (0={min_val:.2g}, 1={max_val:.2g})"
            else:
                x_axis_display_values = x_plot_values_actual

            all_original_model_vars = [
                col for col in original_training_data.columns
                if col not in [wrapper["time_col"], wrapper["event_col"]]
            ]

            predict_df_list_for_current_cov = []
            for current_x_val in x_plot_values_actual:
                row = {current_cov_to_plot: current_x_val}
                for other_col in all_original_model_vars:
                    if other_col == current_cov_to_plot:
                        continue
                    if pd.api.types.is_numeric_dtype(original_training_data[other_col]):
                        row[other_col] = original_training_data[other_col].mean()
                    else:
                        modes = original_training_data[other_col].mode(dropna=True)
                        row[other_col] = modes.iloc[0] if not modes.empty else None
                predict_df_list_for_current_cov.append(row)

            predict_df_current_cov = pd.DataFrame(predict_df_list_for_current_cov)
            predict_df_current_cov = predict_df_current_cov.reindex(columns=all_original_model_vars, fill_value=np.nan)

            try:
                log_ph_preds = cph_model_vip.predict_log_partial_hazard(predict_df_current_cov)
            except Exception as pred_err:
                self.log(f"Error prediciendo efecto parcial para '{current_cov_to_plot}' en '{current_model_display}': {pred_err}", "ERROR")
                continue

            y_values_for_plot = log_ph_preds
            if chosen_y_scale == "hr":
                y_values_for_plot = np.exp(log_ph_preds)

            ci_lower_plot, ci_upper_plot = None, None
            ci_available_for_this_line = False
            design_matrix_pred_aligned = None
            params_cols = cph_model_vip.params_.index if hasattr(cph_model_vip, 'params_') else []

            if hasattr(cph_model_vip, 'regressors') and cph_model_vip.regressors is not None and not predict_df_current_cov.empty:
                try:
                    transformed_design = cph_model_vip.regressors.transform_df(predict_df_current_cov)
                    if isinstance(transformed_design.columns, pd.MultiIndex):
                        try:
                            design_matrix_pred_aligned = transformed_design.xs('beta_', axis=1, level='param')
                        except Exception:
                            design_matrix_pred_aligned = transformed_design.copy()
                    else:
                        design_matrix_pred_aligned = transformed_design.copy()
                    if params_cols is not None and len(params_cols) > 0:
                        design_matrix_pred_aligned = design_matrix_pred_aligned.reindex(columns=params_cols, fill_value=0.0)
                    design_matrix_pred_aligned = design_matrix_pred_aligned.astype(float)
                except Exception as e_transform_df:
                    design_matrix_pred_aligned = None
                    self.log(f"Advertencia: transform_df falló para {current_cov_to_plot} en '{current_model_display}': {e_transform_df}. Se intentará método alterno.", "WARN")

            if design_matrix_pred_aligned is None and PATSY_AVAILABLE and hasattr(cph_model_vip, 'formula') and not predict_df_current_cov.empty:
                try:
                    design_matrix_pred_current_cov = dmatrix(cph_model_vip.formula, predict_df_current_cov, return_type='dataframe')
                    design_matrix_pred_aligned = design_matrix_pred_current_cov.reindex(columns=params_cols, fill_value=0.0).fillna(0.0)
                except Exception as e_ci_loop:
                    self.log(f"Error calculating CI (dmatrix) para {current_cov_to_plot} en '{current_model_display}': {e_ci_loop}. Se intentará método alterno.", "WARN")
                    design_matrix_pred_aligned = None

            if design_matrix_pred_aligned is not None and hasattr(cph_model_vip, 'variance_matrix_'):
                try:
                    variance_matrix_df = cph_model_vip.variance_matrix_
                    if isinstance(variance_matrix_df, pd.DataFrame):
                        variance_matrix_df = variance_matrix_df.reindex(index=params_cols, columns=params_cols, fill_value=0.0)
                        variance_matrix_np = variance_matrix_df.to_numpy(dtype=float, copy=False)
                    else:
                        variance_matrix_np = np.asarray(variance_matrix_df, dtype=float)

                    design_matrix_np = design_matrix_pred_aligned.to_numpy(dtype=float, copy=False)
                    variance_pred = np.einsum('ij,jk,ik->i', design_matrix_np, variance_matrix_np, design_matrix_np, optimize=True)
                    variance_pred = np.clip(variance_pred, a_min=0.0, a_max=None)
                    se_pred = np.sqrt(variance_pred)

                    if chosen_y_scale == "log_hr":
                        ci_lower_plot = log_ph_preds - 1.96 * se_pred
                        ci_upper_plot = log_ph_preds + 1.96 * se_pred
                    else:
                        ci_lower_plot = np.exp(log_ph_preds - 1.96 * se_pred)
                        ci_upper_plot = np.exp(log_ph_preds + 1.96 * se_pred)
                    ci_available_for_this_line = True
                except Exception as e_ci_variance:
                    self.log(f"Error calculating CI (var-matrix) para {current_cov_to_plot} en '{current_model_display}': {e_ci_variance}.", "WARN")

            if not ci_available_for_this_line and hasattr(cph_model_vip, '_compute_pointwise_statistics'):
                try:
                    stats_df = cph_model_vip._compute_pointwise_statistics(
                        predict_df_current_cov,
                        predict_function=cph_model_vip.predict_partial_hazard,
                        predict_function_kwargs={}
                    )
                    if isinstance(stats_df, pd.DataFrame) and {'estimate', 'lower', 'upper'}.issubset(stats_df.columns):
                        estimate_vals = stats_df['estimate'].to_numpy(dtype=float)
                        lower_vals = stats_df['lower'].to_numpy(dtype=float)
                        upper_vals = stats_df['upper'].to_numpy(dtype=float)
                        if chosen_y_scale == "log_hr":
                            y_values_for_plot = np.log(np.clip(estimate_vals, a_min=1e-12, a_max=None))
                            ci_lower_plot = np.log(np.clip(lower_vals, a_min=1e-12, a_max=None))
                            ci_upper_plot = np.log(np.clip(upper_vals, a_min=1e-12, a_max=None))
                        else:
                            y_values_for_plot = estimate_vals
                            ci_lower_plot = lower_vals
                            ci_upper_plot = upper_vals
                        ci_available_for_this_line = True
                except Exception as e_ci_alt:
                    self.log(f"Fallback CI (pointwise stats) falló para {current_cov_to_plot} en '{current_model_display}': {e_ci_alt}.", "WARN")

            knot_info_for_var = spline_metadata_for_model.get(current_cov_to_plot, {})
            knots_axis_values = []
            knots_y_values = []
            knots_actual_vals = knot_info_for_var.get('internal_knots', []) or []
            boundary_axis_values = []
            boundary_actual_vals = knot_info_for_var.get('boundary_knots', []) or []

            if knots_actual_vals and is_numeric_cov:
                axis_min_val = float(np.min(x_axis_display_values)) if len(x_axis_display_values) else None
                axis_max_val = float(np.max(x_axis_display_values)) if len(x_axis_display_values) else None

                if axis_min_val is not None and axis_max_val is not None and axis_max_val != axis_min_val:
                    for knot_actual in knots_actual_vals:
                        axis_val = (knot_actual - min_val) / (max_val - min_val) if (apply_normalization and max_val != min_val) else knot_actual
                        if axis_val < axis_min_val - 1e-9 or axis_val > axis_max_val + 1e-9:
                            continue
                        knots_axis_values.append(axis_val)
                        try:
                            knots_y_values.append(float(np.interp(axis_val, x_axis_display_values, y_values_for_plot)))
                        except Exception:
                            knots_y_values.append(None)

                if knots_actual_vals:
                    formatted_knots = ", ".join(f"{val:.4g}" for val in knots_actual_vals)
                    self.log(f"Nodos internos para '{current_cov_to_plot}' en '{current_model_display}': {formatted_knots}", "INFO")

            if boundary_actual_vals and is_numeric_cov and len(x_axis_display_values):
                axis_min_val = float(np.min(x_axis_display_values)) if len(x_axis_display_values) else None
                axis_max_val = float(np.max(x_axis_display_values)) if len(x_axis_display_values) else None
                if axis_min_val is not None and axis_max_val is not None and axis_max_val != axis_min_val:
                    for boundary_value in boundary_actual_vals:
                        axis_val = (boundary_value - min_val) / (max_val - min_val) if (apply_normalization and max_val != min_val) else boundary_value
                        boundary_axis_values.append(axis_val)

            base_label = f"{current_model_display} :: {current_cov_to_plot}" if len(model_wrappers) > 1 else current_cov_to_plot
            if normalization_info:
                legend_label = normalization_info.replace(current_cov_to_plot, base_label)
            else:
                legend_label = base_label

            plot_data_list.append({
                "covariate_name": current_cov_to_plot,
                "model_display_name": current_model_display,
                "legend_label": legend_label,
                "x_values_for_plot_axis": x_axis_display_values,
                "y_values_for_plot": y_values_for_plot,
                "ci_lower": ci_lower_plot,
                "ci_upper": ci_upper_plot,
                "ci_available": ci_available_for_this_line,
                "knot_info": {
                    "axis_positions": knots_axis_values,
                    "y_positions": knots_y_values,
                    "actual_positions": knots_actual_vals,
                    "metadata": knot_info_for_var,
                    "boundary_axis_positions": boundary_axis_values,
                    "boundary_actual_positions": boundary_actual_vals
                },
                "extension_info": extension_info
            })

        if not plot_data_list:
            messagebox.showerror("Error de Datos", "No se pudieron generar datos para graficar.", parent=self.parent_for_dialogs)
            self.log("plot_data_list vacío, no se puede graficar.", "ERROR")
            if fig_effect:
                plt.close(fig_effect)
            return

        try:
            num_lines = len(plot_data_list)
            y_scale_name_for_legend = "Log(HR)" if chosen_y_scale == "log_hr" else "HR"

            shaded_training_range_done = False

            for i, line_data in enumerate(plot_data_list):
                color = plt.cm.get_cmap('viridis')(i / max(1, num_lines - 1)) if num_lines > 1 else 'blue'

                ax_effect.plot(
                    line_data["x_values_for_plot_axis"],
                    line_data["y_values_for_plot"],
                    label=line_data["legend_label"],
                    color=color
                )

                if plot_show_ci and line_data["ci_available"]:
                    ax_effect.fill_between(
                        line_data["x_values_for_plot_axis"],
                        line_data["ci_lower"],
                        line_data["ci_upper"],
                        alpha=0.2,
                        color=color
                    )
                elif plot_show_ci and not line_data["ci_available"]:
                    self.log(f"IC 95% no disponible para '{line_data['covariate_name']}' en '{line_data['model_display_name']}'.", "WARN")

                knot_info_line = line_data.get("knot_info", {})
                knot_axis_positions = knot_info_line.get("axis_positions") or []
                knot_y_positions = knot_info_line.get("y_positions") or []
                boundary_axis_positions = knot_info_line.get("boundary_axis_positions") or []

                if plot_show_knots and knot_axis_positions and knot_y_positions:
                    first_marker = True
                    for xk, yk in zip(knot_axis_positions, knot_y_positions):
                        ax_effect.axvline(xk, color=color, linestyle=':', linewidth=0.9, alpha=0.45)
                        if yk is not None:
                            marker_label = f"Nodos {line_data['model_display_name']}::{line_data['covariate_name']}" if first_marker else "_nolegend_"
                            ax_effect.scatter([xk], [yk], color=color, marker='D', edgecolors='black', s=40, zorder=6, label=marker_label)
                        first_marker = False

                if boundary_axis_positions:
                    first_boundary = True
                    for xb in boundary_axis_positions:
                        boundary_label = f"Límites {line_data['model_display_name']}::{line_data['covariate_name']}" if first_boundary else "_nolegend_"
                        ax_effect.axvline(xb, color=color, linestyle='--', linewidth=0.8, alpha=0.35, label=boundary_label)
                        first_boundary = False

                extension_meta = line_data.get("extension_info")
                if (
                    extension_meta and not shaded_training_range_done and
                    extension_meta.get("trained_min") is not None and
                    extension_meta.get("trained_max") is not None and
                    extension_meta["trained_max"] > extension_meta["trained_min"]
                ):
                    ax_effect.axvspan(
                        extension_meta["trained_min"],
                        extension_meta["trained_max"],
                        color='grey',
                        alpha=0.08,
                        zorder=-5,
                        label="Rango entrenado"
                    )
                    shaded_training_range_done = True

            models_present = sorted({entry["model_display_name"] for entry in plot_data_list})

            title_text = ""
            xlabel_text = ""
            ylabel_text = f"{y_scale_name_for_legend} (Efecto Parcial Ajustado)"

            if num_lines == 1:
                single_line_data = plot_data_list[0]
                single_model_name = single_line_data["model_display_name"]
                single_cov_name = single_line_data["covariate_name"]
                title_text = f"Efecto Ajustado de '{single_cov_name}' sobre {y_scale_name_for_legend}\nModelo: {single_model_name}"
                if "(::" in single_line_data["legend_label"]:
                    base_cov_text = single_line_data["legend_label"].split(" (::", 1)[0]
                else:
                    base_cov_text = single_line_data["legend_label"]
                if apply_normalization:
                    if base_cov_text == single_cov_name:
                        xlabel_text = f"Valor Normalizado de {single_cov_name} (0-1)"
                    else:
                        xlabel_text = f"Valor Normalizado de {base_cov_text} (0-1)"
                else:
                    xlabel_text = f"Valor de {single_cov_name}"

                knot_info_single = single_line_data.get("knot_info", {})
                if plot_show_knots and knot_info_single and knot_info_single.get("actual_positions"):
                    formatted_knots = ", ".join(f"{val:.4g}" for val in knot_info_single.get("actual_positions", []))
                    ax_effect.text(
                        0.98,
                        0.02,
                        f"Nodos: {formatted_knots}",
                        transform=ax_effect.transAxes,
                        ha='right',
                        va='bottom',
                        fontsize=9,
                        color='dimgray'
                    )
            else:
                if len(models_present) == 1:
                    title_text = f"Efecto Ajustado sobre {y_scale_name_for_legend}\nModelo: {models_present[0]}"
                else:
                    title_text = f"Efecto Ajustado sobre {y_scale_name_for_legend}\nModelos: {', '.join(models_present)}"
                if apply_normalization:
                    xlabel_text = "Valor Normalizado de Covariable (0-1)"
                else:
                    xlabel_text = "Valor de Covariable"

            if chosen_y_scale == "hr":
                ax_effect.axhline(1, color='grey', linestyle='--', linewidth=0.8)
            else:
                ax_effect.axhline(0, color='grey', linestyle='--', linewidth=0.8)

            current_opts_effect = self.current_plot_options.copy()
            current_opts_effect['title'] = current_opts_effect.get('title', title_text)
            current_opts_effect['xlabel'] = xlabel_text
            current_opts_effect['ylabel'] = ylabel_text

            apply_plot_options(ax_effect, current_opts_effect, self.log)

            if num_lines > 0 and ax_effect.has_data():
                ax_effect.legend(fontsize='small')

            plt.tight_layout()
            self._create_plot_window(fig_effect, "Efecto Ajustado de Covariable(s)")
            self.log(f"Gráfico de efecto ajustado para {num_lines} línea(s) generado.", "SUCCESS")

        except Exception as e_plot_final:
            self.log(f"Error final al graficar efectos ajustados: {e_plot_final}", "ERROR")
            if fig_effect:
                plt.close(fig_effect)
            traceback.print_exc(limit=5)
            messagebox.showerror(
                "Error de Gráfico Final",
                f"No se pudo generar el gráfico de efectos ajustados:\n{e_plot_final}",
                parent=self.parent_for_dialogs
            )

    def _compute_time_selected_impact_data(self, model_dict, covariate, eval_time=None, output_type="risk", manual_values_text="", baseline_text=""):
        if not isinstance(model_dict, dict):
            raise ValueError("No se encontró un modelo Cox válido.")

        cph_model = model_dict.get('model')
        if not (cph_model and isinstance(cph_model, CoxPHFitter)):
            raise ValueError("El modelo seleccionado no es un Cox válido.")

        data = model_dict.get('_df_for_fit_main_INTERNAL_USE')
        if data is None or data.empty:
            raise ValueError("No hay datos de entrenamiento disponibles para el modelo seleccionado.")

        time_col = model_dict.get('time_col_for_model')
        event_col = model_dict.get('event_col_for_model')
        available_covariates = [col for col in data.columns if col not in [time_col, event_col]]
        if covariate not in available_covariates:
            raise ValueError(f"La covariable '{covariate}' no está disponible en el modelo.")

        cov_series = data[covariate].dropna()
        if cov_series.empty:
            raise ValueError("La covariable seleccionada no tiene valores disponibles.")

        is_numeric = pd.api.types.is_numeric_dtype(cov_series)
        manual_values = self._parse_partial_effect_values(covariate, manual_values_text, data=data)

        if manual_values:
            values = manual_values
        elif is_numeric:
            numeric_values = pd.to_numeric(cov_series, errors='coerce').dropna()
            values = np.linspace(float(numeric_values.min()), float(numeric_values.max()), num=60)
            values = np.unique(values)
            if values.size < 2:
                center_val = float(numeric_values.iloc[0]) if not numeric_values.empty else 0.0
                delta = abs(center_val * 0.05) if center_val != 0 else 0.05
                values = np.array([center_val - delta, center_val + delta], dtype=float)
        else:
            values = sorted(cov_series.astype(str).unique().tolist())[:8]

        if len(values) == 0:
            raise ValueError("No se pudieron generar valores para la covariable seleccionada.")

        baseline_overrides = self._parse_partial_effect_baseline_overrides(
            baseline_text,
            exclude_covariate=covariate,
            data=data,
            available_covariates=available_covariates,
        )
        base_row = self._build_plot_reference_row(
            focal_covariate=covariate,
            overrides=baseline_overrides,
            data=data,
            covariates=available_covariates,
        )

        predict_rows = []
        labels = []
        for value in values:
            row = base_row.copy()
            row[covariate] = value
            predict_rows.append(row)
            labels.append(self._format_plot_value_label(value))

        predict_df = pd.DataFrame(predict_rows, columns=available_covariates)
        survival_df = cph_model.predict_survival_function(predict_df)
        if survival_df is None or survival_df.empty:
            raise ValueError("El modelo no devolvió curvas de supervivencia para la covariable seleccionada.")

        times = survival_df.index.to_numpy(dtype=float)
        if times.size == 0:
            raise ValueError("No se encontraron tiempos válidos en la predicción del modelo.")

        try:
            requested_time = float(eval_time)
        except (TypeError, ValueError):
            requested_time = np.nan

        if not np.isfinite(requested_time) or requested_time <= 0:
            if time_col and time_col in data.columns:
                requested_time = float(np.nanmedian(pd.to_numeric(data[time_col], errors='coerce')))
        if not np.isfinite(requested_time) or requested_time <= 0:
            requested_time = float(times[-1]) if times.size else 1.0
        if not np.isfinite(requested_time) or requested_time <= 0:
            requested_time = 1.0

        eval_time_clipped = float(np.clip(requested_time, times[0], times[-1]))

        survival_at_t = []
        for idx in range(survival_df.shape[1]):
            curve_values = survival_df.iloc[:, idx].to_numpy(dtype=float)
            survival_at_t.append(float(np.interp(eval_time_clipped, times, curve_values)))

        survival_at_t = np.asarray(survival_at_t, dtype=float)
        y_values = survival_at_t.copy()
        ci_lower = None
        ci_upper = None
        ci_available = False

        try:
            log_partial_hazard = cph_model.predict_log_partial_hazard(predict_df)
            lp_values = np.asarray(log_partial_hazard, dtype=float).reshape(-1)
            params_cols = cph_model.params_.index if hasattr(cph_model, 'params_') else []
            design_matrix_pred_aligned = None

            if hasattr(cph_model, 'regressors') and cph_model.regressors is not None and not predict_df.empty:
                transformed_design = cph_model.regressors.transform_df(predict_df)
                if isinstance(transformed_design.columns, pd.MultiIndex):
                    try:
                        design_matrix_pred_aligned = transformed_design.xs('beta_', axis=1, level='param')
                    except Exception:
                        design_matrix_pred_aligned = transformed_design.copy()
                else:
                    design_matrix_pred_aligned = transformed_design.copy()

            if design_matrix_pred_aligned is not None and len(params_cols) > 0:
                design_matrix_pred_aligned = design_matrix_pred_aligned.reindex(columns=params_cols, fill_value=0.0)
                design_matrix_pred_aligned = design_matrix_pred_aligned.astype(float)

                variance_matrix_df = getattr(cph_model, 'variance_matrix_', None)
                baseline_ch_df = getattr(cph_model, 'baseline_cumulative_hazard_', None)
                if isinstance(variance_matrix_df, pd.DataFrame) and isinstance(baseline_ch_df, pd.DataFrame) and not baseline_ch_df.empty:
                    variance_matrix_df = variance_matrix_df.reindex(index=params_cols, columns=params_cols, fill_value=0.0)
                    variance_matrix_np = variance_matrix_df.to_numpy(dtype=float, copy=False)
                    design_matrix_np = design_matrix_pred_aligned.to_numpy(dtype=float, copy=False)
                    variance_pred = np.einsum('ij,jk,ik->i', design_matrix_np, variance_matrix_np, design_matrix_np, optimize=True)
                    variance_pred = np.clip(variance_pred, a_min=0.0, a_max=None)
                    se_lp = np.sqrt(variance_pred)

                    baseline_times = baseline_ch_df.index.to_numpy(dtype=float)
                    baseline_values = baseline_ch_df.iloc[:, 0].to_numpy(dtype=float)
                    h0_t = float(np.interp(eval_time_clipped, baseline_times, baseline_values))
                    h0_t = max(h0_t, 0.0)

                    lp_lower = lp_values - 1.96 * se_lp
                    lp_upper = lp_values + 1.96 * se_lp
                    survival_lower = np.exp(-h0_t * np.exp(lp_upper))
                    survival_upper = np.exp(-h0_t * np.exp(lp_lower))
                    ci_lower = np.clip(survival_lower, 0.0, 1.0)
                    ci_upper = np.clip(survival_upper, 0.0, 1.0)
                    ci_available = True
        except Exception as ci_exc:
            self.log(f"IC 95% no disponible para impacto en t de '{covariate}': {ci_exc}", "WARN")

        metric_key = str(output_type).strip().lower()
        if metric_key in {"risk", "1-s", "1 - s", "evento", "event"}:
            y_values = 1.0 - survival_at_t
            output_type = "risk"
            if ci_available and ci_lower is not None and ci_upper is not None:
                risk_lower = 1.0 - ci_upper
                risk_upper = 1.0 - ci_lower
                ci_lower = np.clip(risk_lower, 0.0, 1.0)
                ci_upper = np.clip(risk_upper, 0.0, 1.0)
        else:
            y_values = survival_at_t
            output_type = "survival"

        spline_metadata = model_dict.get('spline_basis_metadata', {}) or {}
        spline_info = spline_metadata.get(covariate, {}) if isinstance(spline_metadata, dict) else {}
        knot_values = spline_info.get('internal_knots', []) or []

        x_numeric = np.asarray(values, dtype=float) if is_numeric else np.arange(len(labels), dtype=float)
        return {
            "covariate": covariate,
            "is_numeric": is_numeric,
            "x_numeric": x_numeric,
            "raw_values": list(values),
            "labels": labels,
            "y_values": np.clip(np.asarray(y_values, dtype=float), 0.0, 1.0),
            "ci_lower": np.asarray(ci_lower, dtype=float) if ci_lower is not None else None,
            "ci_upper": np.asarray(ci_upper, dtype=float) if ci_upper is not None else None,
            "ci_available": bool(ci_available and ci_lower is not None and ci_upper is not None),
            "baseline_overrides": baseline_overrides,
            "requested_time": float(requested_time),
            "eval_time": float(eval_time_clipped),
            "output_type": output_type,
            "knot_values": knot_values,
            "model_name": model_dict.get('custom_model_name', model_dict.get('model_name', 'Modelo Cox')),
        }

    def show_variable_impact_at_time_plot(self):
        if not self._check_model_selected_and_valid(check_params=True):
            return

        md_vit = self.selected_model_in_treeview
        training_df = md_vit.get('_df_for_fit_main_INTERNAL_USE')

        if training_df is None or training_df.empty:
            messagebox.showinfo("Sin Datos", "El modelo seleccionado no tiene datos de entrenamiento disponibles.", parent=self.parent_for_dialogs)
            return

        time_col = md_vit.get('time_col_for_model')
        event_col = md_vit.get('event_col_for_model')
        available_covariates = [col for col in training_df.columns if col not in [time_col, event_col]]
        if not available_covariates:
            messagebox.showinfo("Sin Covariables", "No hay covariables disponibles para estimar el impacto en el tiempo.", parent=self.parent_for_dialogs)
            return

        default_time = np.nan
        if time_col and time_col in training_df.columns:
            default_time = float(np.nanmedian(pd.to_numeric(training_df[time_col], errors='coerce')))
        if not np.isfinite(default_time) or default_time <= 0:
            default_time = 1.0

        dialog = Toplevel(self.parent_for_dialogs)
        dialog.title("Impacto en tiempo elegido")
        dialog.geometry("560x320")

        ttk.Label(
            dialog,
            text="Seleccione la covariable y el tiempo t para estimar S(t) o 1-S(t) como en AFT/RSF.",
            wraplength=520
        ).pack(pady=10, padx=12)

        form_frame = ttk.Frame(dialog, padding=(12, 0, 12, 0))
        form_frame.pack(fill=tk.BOTH, expand=True)
        form_frame.columnconfigure(1, weight=1)

        covariate_var = StringVar(value=available_covariates[0])
        output_type_var = StringVar(value="risk")
        time_var = StringVar(value=f"{default_time:.2f}")
        manual_values_var = StringVar()
        baseline_var = StringVar()
        show_ci_var = BooleanVar(value=True)

        ttk.Label(form_frame, text="Covariable:").grid(row=0, column=0, sticky="w", padx=(0, 8), pady=4)
        ttk.Combobox(form_frame, textvariable=covariate_var, values=available_covariates, state="readonly", width=34).grid(row=0, column=1, sticky="ew", pady=4)

        ttk.Label(form_frame, text="Salida:").grid(row=1, column=0, sticky="w", padx=(0, 8), pady=4)
        output_frame = ttk.Frame(form_frame)
        output_frame.grid(row=1, column=1, sticky="w", pady=4)
        ttk.Radiobutton(output_frame, text="Riesgo 1-S(t)", variable=output_type_var, value="risk").pack(side=tk.LEFT)
        ttk.Radiobutton(output_frame, text="Supervivencia S(t)", variable=output_type_var, value="survival").pack(side=tk.LEFT, padx=(10, 0))

        ttk.Label(form_frame, text="Tiempo t:").grid(row=2, column=0, sticky="w", padx=(0, 8), pady=4)
        ttk.Entry(form_frame, textvariable=time_var, width=16).grid(row=2, column=1, sticky="w", pady=4)

        ttk.Label(form_frame, text="Valores (opcional):").grid(row=3, column=0, sticky="w", padx=(0, 8), pady=4)
        ttk.Entry(form_frame, textvariable=manual_values_var, width=42).grid(row=3, column=1, sticky="ew", pady=4)

        ttk.Label(form_frame, text="Otras vars fijas:").grid(row=4, column=0, sticky="w", padx=(0, 8), pady=4)
        ttk.Entry(form_frame, textvariable=baseline_var, width=42).grid(row=4, column=1, sticky="ew", pady=4)

        ttk.Checkbutton(form_frame, text="Mostrar IC 95%", variable=show_ci_var).grid(row=5, column=1, sticky="w", pady=(2, 2))

        ttk.Label(
            form_frame,
            text="Formato sugerido: `Edad=60; Sexo=M` o `Edad:60, Sexo:F`.",
            foreground="#555555",
            wraplength=420
        ).grid(row=6, column=1, sticky="w", pady=(0, 6))

        chosen_options = {}

        def on_ok():
            selected_covariate = covariate_var.get().strip()
            if not selected_covariate:
                messagebox.showwarning("Selección requerida", "Debe elegir una covariable.", parent=dialog)
                return
            chosen_options.update({
                "covariate": selected_covariate,
                "output_type": output_type_var.get().strip() or "risk",
                "time": time_var.get().strip(),
                "manual_values_text": manual_values_var.get(),
                "baseline_text": baseline_var.get(),
                "show_ci": bool(show_ci_var.get()),
            })
            dialog.destroy()

        def on_cancel():
            dialog.destroy()

        button_frame = ttk.Frame(dialog)
        button_frame.pack(pady=10)
        ttk.Button(button_frame, text="Aceptar", command=on_ok).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Cancelar", command=on_cancel).pack(side=tk.RIGHT, padx=5)

        dialog.transient(self.parent_for_dialogs)
        dialog.grab_set()
        self.parent_for_dialogs.wait_window(dialog)

        if not chosen_options:
            self.log("Selección para gráfico de impacto en tiempo cancelada.", "INFO")
            return

        fig_vit = None
        try:
            plot_payload = self._compute_time_selected_impact_data(
                md_vit,
                chosen_options["covariate"],
                eval_time=chosen_options.get("time"),
                output_type=chosen_options.get("output_type", "risk"),
                manual_values_text=chosen_options.get("manual_values_text", ""),
                baseline_text=chosen_options.get("baseline_text", ""),
            )

            fig_vit, ax_vit = plt.subplots(figsize=(10, 6))
            color = "#d62728" if plot_payload["output_type"] == "risk" else "#1f77b4"
            x_numeric = np.asarray(plot_payload["x_numeric"], dtype=float)
            y_values = np.asarray(plot_payload["y_values"], dtype=float)

            show_ci = bool(chosen_options.get("show_ci", True))
            ci_available = bool(plot_payload.get("ci_available"))
            ci_lower = np.asarray(plot_payload.get("ci_lower"), dtype=float) if plot_payload.get("ci_lower") is not None else None
            ci_upper = np.asarray(plot_payload.get("ci_upper"), dtype=float) if plot_payload.get("ci_upper") is not None else None

            if plot_payload["is_numeric"]:
                order = np.argsort(x_numeric)
                x_plot = x_numeric[order]
                y_plot = y_values[order]
                ax_vit.plot(x_plot, y_plot, color=color, linewidth=2, label=plot_payload['model_name'])
                if show_ci and ci_available and ci_lower is not None and ci_upper is not None:
                    ax_vit.fill_between(x_plot, ci_lower[order], ci_upper[order], color=color, alpha=0.18, label="IC 95%")
                else:
                    ax_vit.fill_between(x_plot, y_plot, color=color, alpha=0.12)
                ax_vit.set_xlabel(plot_payload["covariate"])

                knot_values = plot_payload.get("knot_values") or []
                if knot_values:
                    knot_positions = np.asarray(knot_values, dtype=float)
                    knot_y = np.interp(knot_positions, x_plot, y_plot)
                    for idx, knot in enumerate(knot_positions):
                        ax_vit.axvline(
                            knot,
                            color="#9467bd",
                            linestyle="--",
                            linewidth=1.1,
                            alpha=0.8,
                            label="Nodos spline" if idx == 0 else None,
                        )
                    ax_vit.scatter(knot_positions, knot_y, color="#9467bd", s=35, zorder=4)
            else:
                x_positions = np.arange(len(plot_payload["labels"]))
                ax_vit.bar(x_positions, y_values, color=color, alpha=0.82)
                if show_ci and ci_available and ci_lower is not None and ci_upper is not None:
                    lower_err = np.clip(y_values - ci_lower, 0.0, None)
                    upper_err = np.clip(ci_upper - y_values, 0.0, None)
                    ax_vit.errorbar(
                        x_positions,
                        y_values,
                        yerr=np.vstack([lower_err, upper_err]),
                        fmt='none',
                        ecolor='black',
                        elinewidth=1.0,
                        capsize=4,
                        label="IC 95%",
                    )
                ax_vit.set_xticks(x_positions)
                ax_vit.set_xticklabels(plot_payload["labels"], rotation=15)
                ax_vit.set_xlabel(plot_payload["covariate"])

            if show_ci and not ci_available:
                self.log(f"IC 95% no disponible para la gráfica en t de '{plot_payload['covariate']}'.", "WARN")

            if plot_payload["output_type"] == "risk":
                ylabel = f"Riesgo estimado (1 - S(t)) con t={plot_payload['eval_time']:.2f}"
                title = f"Impacto de '{plot_payload['covariate']}' sobre el riesgo acumulado\nModelo: {plot_payload['model_name']}"
            else:
                ylabel = f"Supervivencia estimada S(t) con t={plot_payload['eval_time']:.2f}"
                title = f"Impacto de '{plot_payload['covariate']}' sobre la supervivencia\nModelo: {plot_payload['model_name']}"

            ax_vit.set_ylabel(ylabel)
            ax_vit.set_title(title)
            ax_vit.set_ylim(0, 1.05)
            ax_vit.grid(True, alpha=0.2)
            handles, legend_labels = ax_vit.get_legend_handles_labels()
            if legend_labels:
                ax_vit.legend(loc="best", fontsize=8)

            if plot_payload["baseline_overrides"]:
                overrides_text = ', '.join(
                    f"{key}: {self._format_plot_value_label(val)}"
                    for key, val in plot_payload["baseline_overrides"].items()
                )
                ax_vit.text(
                    0.02,
                    0.98,
                    f"Otras vars fijas: {overrides_text}",
                    transform=ax_vit.transAxes,
                    ha='left',
                    va='top',
                    fontsize=8,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.75, edgecolor='#bbbbbb'),
                )

            if abs(plot_payload["requested_time"] - plot_payload["eval_time"]) > 1e-9:
                ax_vit.text(
                    0.98,
                    0.02,
                    f"t solicitado={plot_payload['requested_time']:.2f}; ajustado al rango={plot_payload['eval_time']:.2f}",
                    transform=ax_vit.transAxes,
                    ha='right',
                    va='bottom',
                    fontsize=8,
                    color='dimgray',
                )

            current_opts_vit = self.current_plot_options.copy()
            current_opts_vit['title'] = current_opts_vit.get('title') or title
            current_opts_vit['xlabel'] = current_opts_vit.get('xlabel') or ax_vit.get_xlabel()
            current_opts_vit['ylabel'] = current_opts_vit.get('ylabel') or ylabel
            apply_plot_options(ax_vit, current_opts_vit, self.log)

            plt.tight_layout()
            self._create_plot_window(fig_vit, f"Impacto en t: {plot_payload['covariate']} ({plot_payload['model_name']})")
            self.log(
                f"Gráfico de impacto en tiempo generado para '{plot_payload['covariate']}' con t={plot_payload['eval_time']:.2f}.",
                "SUCCESS"
            )
        except Exception as e_vit:
            self.log(f"Error al generar gráfico de impacto en tiempo: {e_vit}", "ERROR")
            self.log(traceback.format_exc(), "DEBUG")
            messagebox.showerror(
                "Error Gráfico",
                f"No se pudo generar el gráfico de impacto en tiempo:\n{e_vit}",
                parent=self.parent_for_dialogs,
            )
            if fig_vit:
                plt.close(fig_vit)

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

        formatted_terms_gst = [self._format_term_for_display(term) for term in model_dict_gst.get('covariates_processed', [])]
        s_txt_gst += "\nConfiguración Ajuste:\n"; s_txt_gst += f"  Tiempo: {model_dict_gst.get('time_col_for_model','N/A')}\n  Evento: {model_dict_gst.get('event_col_for_model','N/A')}\n"
        s_txt_gst += f"  Fórmula Patsy (usada en fit): {model_dict_gst.get('formula_patsy','N/A')}\n"
        s_txt_gst += f"  Fórmula Patsy (original completa para transformar nuevos datos): {model_dict_gst.get('full_patsy_formula_for_new_data_transform','N/A')}\n"
        s_txt_gst += f"  Términos Modelo (columnas en X_design): {', '.join(formatted_terms_gst)}\n"
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

        categorical_cfg_snapshot = model_dict_gst.get('categorical_compare_config', {}) or {}
        ref_cfg_snapshot = model_dict_gst.get('ref_categories_config', {}) or {}
        if categorical_cfg_snapshot or ref_cfg_snapshot:
            s_txt_gst += "Configuración de Variables Cualitativas:\n"
            relevant_cat_vars = set()
            for raw_term in model_dict_gst.get('covariates_processed', []) or []:
                raw_term_str = str(raw_term)
                match_cat_var = re.search(r"C\(Q\('([^']+)'\)", raw_term_str)
                if match_cat_var:
                    relevant_cat_vars.add(match_cat_var.group(1))
            if not relevant_cat_vars:
                relevant_cat_vars = set(ref_cfg_snapshot.keys()) | set(categorical_cfg_snapshot.keys())

            for cat_var in sorted(relevant_cat_vars):
                ref_val = ref_cfg_snapshot.get(cat_var, 'N/A')
                compare_cfg = categorical_cfg_snapshot.get(cat_var, {}) or {}
                compare_mode = compare_cfg.get('mode', 'all')
                selected_groups = compare_cfg.get('selected_groups', []) or []
                if compare_mode == 'one_vs_rest':
                    desc_mode = f"Dicotómica: {ref_val} vs RESTO"
                elif compare_mode == 'selected':
                    desc_mode = f"Solo grupos elegidos vs {ref_val}: {', '.join(map(str, selected_groups)) if selected_groups else '(sin grupos)'}"
                else:
                    desc_mode = f"Todos los grupos vs {ref_val}"
                s_txt_gst += f"  - {cat_var}: {desc_mode}\n"
            s_txt_gst += "\n"

        spline_meta_snapshot = model_dict_gst.get('spline_basis_metadata', {}) or {}
        penalizer_val_summary = model_dict_gst.get('penalizer_value', 0.0)
        l1_ratio_val_summary = model_dict_gst.get('l1_ratio_value', 0.0)
        s_txt_gst += "Configuración de Splines:\n"
        if spline_meta_snapshot:
            for cov_name_meta, meta_details in sorted(spline_meta_snapshot.items()):
                meta_details = meta_details or {}
                detail_parts = []

                method_label = meta_details.get('spline_type', 'N/A')
                detail_parts.append(f"método={method_label}")

                detail_parts.append(f"penalización={penalizer_val_summary:.4g} (l1={l1_ratio_val_summary:.2f})")

                degree_val = meta_details.get('degree')
                if degree_val is not None:
                    detail_parts.append("cúbico" if degree_val == 3 else f"grado={degree_val}")

                df_requested_val = meta_details.get('df_requested')
                df_applied_val = meta_details.get('df_applied')
                df_effective_val = meta_details.get('df_effective')

                if df_requested_val is not None:
                    detail_parts.append(f"df pedido={df_requested_val}")
                if df_applied_val is not None:
                    detail_parts.append(f"df usado={df_applied_val}")
                if df_effective_val is not None and df_effective_val != df_applied_val:
                    detail_parts.append(f"df efectivo={df_effective_val}")

                num_knots_req = meta_details.get('num_knots_requested')
                num_knots_auto = meta_details.get('num_knots_derived_from_df')
                num_knots_used = meta_details.get('num_knots_used')
                if num_knots_req is not None:
                    detail_parts.append(f"nodos pedidos={num_knots_req}")
                if num_knots_auto is not None and (num_knots_req is None or num_knots_auto != num_knots_req):
                    detail_parts.append(f"nodos df={num_knots_auto}")
                if num_knots_used is not None:
                    detail_parts.append(f"nodos usados={num_knots_used}")

                knots_source = meta_details.get('internal_knots_source')
                if knots_source:
                    detail_parts.append(f"origen_nodos={knots_source}")

                boundary_vals = meta_details.get('boundary_knots') or []
                if boundary_vals:
                    boundary_str = ", ".join(f"{val:.4g}" for val in boundary_vals)
                    detail_parts.append(f"límites=[{boundary_str}]")

                knot_values = meta_details.get('internal_knots') or []
                if knot_values:
                    knots_str = ", ".join(f"{val:.4g}" for val in knot_values)
                    detail_parts.append(f"nodos=[{knots_str}]")

                s_txt_gst += f"  - {cov_name_meta}: " + "; ".join(detail_parts) + "\n"
        else:
            s_txt_gst += "  (No se aplicaron splines a este modelo.)\n"
        s_txt_gst += "\n"

        s_txt_gst += "Coeficientes (Resumen Lifelines):\n"
        sum_df_gst = model_dict_gst.get('metrics',{}).get('summary_df')
        if sum_df_gst is not None and not sum_df_gst.empty:
            sum_df_display = sum_df_gst.copy()
            sum_df_display.index = [self._format_term_for_display(idx) for idx in sum_df_display.index]
            # Show only HR (exp(coef)), its 95% CI, and p-value
            display_cols = [c for c in ['exp(coef)', 'exp(coef) lower 95%', 'exp(coef) upper 95%', 'p'] if c in sum_df_display.columns]
            if display_cols:
                sum_df_display = sum_df_display[display_cols]
            s_txt_gst += sum_df_display.to_string() + "\n\n"
        else:
            s_txt_gst += "  (No disponibles o modelo nulo)\n\n"
        
        s_txt_gst += "Métricas Evaluación:\n"
        metrics_gst = model_dict_gst.get('metrics',{})
        cindex_ci_keys = {"C-Index (Training) CI", "C-Index (Test) CI", "C-Index (CV Mean) CI"}
        for k,v in metrics_gst.items():
            if k in ["summary_df","schoenfeld_details","HR (individual)","HR_CI (individual)","Wald p-values (individual)"] or k in cindex_ci_keys:
                continue
            if isinstance(v,pd.DataFrame):
                continue
            if k == "C-Index (Training)":
                s_txt_gst += f"  {k}: {self._format_c_index_display(v, metrics_gst.get('C-Index (Training) CI'), decimals=4)}\n"
                continue
            if k == "C-Index (Test)":
                s_txt_gst += f"  {k}: {self._format_c_index_display(v, metrics_gst.get('C-Index (Test) CI'), decimals=4)}\n"
                continue
            if k == "C-Index Uno (IPCW)":
                if v is not None and pd.notna(v):
                    s_txt_gst += f"  {k}: {v:.4f} (IPCW = Inverse Probability of Censoring Weighting; corrige por censura)\n"
                else:
                    s_txt_gst += f"  {k}: N/A\n"
                continue
            if k == "C-Index Antolini (Ctd)":
                if v is not None and pd.notna(v):
                    s_txt_gst += f"  {k}: {v:.4f} (AUC dinámica media; generaliza C-index para predicciones dependientes del tiempo)\n"
                else:
                    s_txt_gst += f"  {k}: N/A\n"
                continue
            if k == "τ (tau)":
                if v is not None and pd.notna(v):
                    s_txt_gst += f"  {k}: {v:.2f} (truncamiento IPCW para Uno y Antolini)\n"
                continue
            if k == "C-Index (CV Mean)":
                s_txt_gst += f"  {k}: {self._format_c_index_display(v, metrics_gst.get('C-Index (CV Mean) CI'), decimals=4)}\n"
                continue
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

    def _build_pickle_safe_model_copy(self, model_dict):
        """Create a pickle-safe copy of a Cox model snapshot for export to disk."""
        if not isinstance(model_dict, dict):
            raise TypeError("El modelo a guardar debe ser un diccionario.")

        safe_copy = {}
        skipped_keys = []

        for key, value in model_dict.items():
            if key == "design_info":
                safe_copy[key] = None
                skipped_keys.append(key)
                continue

            try:
                pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
                safe_copy[key] = value
            except Exception as exc:
                skipped_keys.append(key)
                self.log(
                    f"Guardado de modelo: se omitió '{key}' porque no es serializable ({exc}).",
                    "WARN"
                )

        if skipped_keys:
            safe_copy["_serialization_skipped_keys"] = skipped_keys
            safe_copy.setdefault(
                "_serialization_warning",
                "Algunos campos auxiliares no serializables se omitieron al guardar el modelo."
            )

        return safe_copy

    def _sanitize_model_filename(self, raw_name):
        safe_name = str(raw_name if raw_name is not None else 'Modelo_Guardado').strip()
        safe_name = re.sub(r'[<>:"/\\|?*]+', '_', safe_name)
        safe_name = re.sub(r'\s+', '_', safe_name)
        safe_name = safe_name.strip('._')
        return safe_name or 'Modelo_Guardado'

    def save_model(self):
        if not self.selected_model_in_treeview and self.generated_models_data:
            try:
                fallback_idx = len(self.generated_models_data) - 1
                fallback_iid = str(fallback_idx)
                if hasattr(self, 'treeview_lista_modelos'):
                    self.treeview_lista_modelos.selection_set(fallback_iid)
                    self.treeview_lista_modelos.focus(fallback_iid)
                    self.treeview_lista_modelos.see(fallback_iid)
                self._on_model_select_from_treeview()
            except Exception as err_autoselect:
                self.log(f"No se pudo auto-seleccionar un modelo antes de guardar: {err_autoselect}", "WARN")

        if not self._check_model_selected_and_valid(): return
        md_save = self.selected_model_in_treeview
        name_save = md_save.get('custom_model_name', md_save.get('model_name', 'Modelo_Guardado'))
        safe_initial_name = self._sanitize_model_filename(name_save)

        model_dict_to_save = self._build_pickle_safe_model_copy(md_save)

        fpath_save = filedialog.asksaveasfilename(
            title="Guardar Modelo Como...",
            defaultextension=".pkl",
            initialfile=f"{safe_initial_name}.pkl",
            filetypes=[("Pickle", "*.pkl"), ("Todos", "*.*")]
        )
        if not fpath_save:
            self.log("Guardado cancelado.", "INFO")
            return

        temp_save_path = fpath_save + ".tmp"
        try:
            with open(temp_save_path, "wb") as f_save:
                pickle.dump(model_dict_to_save, f_save, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(temp_save_path, fpath_save)
            self.log(f"Modelo '{name_save}' guardado en: {fpath_save}", "SUCCESS")
            messagebox.showinfo("Modelo Guardado", f"Modelo guardado en:\n{fpath_save}", parent=self.parent_for_dialogs)
        except Exception as e_save:
            try:
                if os.path.exists(temp_save_path):
                    os.remove(temp_save_path)
            except Exception:
                pass
            self.log(f"Error guardando modelo: {e_save}", "ERROR")
            messagebox.showerror("Error Guardando", f"No se pudo guardar:\n{e_save}", parent=self.parent_for_dialogs)

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

    def _delete_selected_model(self):
        selected_items = self.treeview_lista_modelos.selection()
        if not selected_items:
            messagebox.showwarning("Sin Selección", "Seleccione uno o más modelos para eliminar.", parent=self.parent_for_dialogs)
            return

        selected_map = {}
        for iid in selected_items:
            try:
                idx = int(iid)
            except ValueError:
                self.log(f"Eliminar modelo: ítem '{iid}' no convertible a índice.", "ERROR")
                continue

            if idx < 0 or idx >= len(self.generated_models_data):
                self.log(f"Eliminar modelo: índice {idx} fuera de rango.", "ERROR")
                continue

            md_to_remove = self.generated_models_data[idx]
            model_display_name = md_to_remove.get('custom_model_name', md_to_remove.get('model_name', f"Modelo {idx+1}"))
            selected_map[idx] = model_display_name

        if not selected_map:
            messagebox.showerror("Selección Inválida", "No se encontraron modelos válidos para eliminar.", parent=self.parent_for_dialogs)
            return

        sorted_indices_asc = sorted(selected_map.keys())
        sorted_indices_desc = sorted(selected_map.keys(), reverse=True)

        if len(sorted_indices_asc) == 1:
            prompt = f"¿Desea eliminar el modelo '{selected_map[sorted_indices_asc[0]]}'?"
        else:
            preview_names = ", ".join(selected_map[idx] for idx in sorted_indices_asc[:3])
            if len(sorted_indices_asc) > 3:
                preview_names += ", ..."
            prompt = (
                f"¿Desea eliminar los {len(sorted_indices_asc)} modelos seleccionados?\n"
                f"{preview_names}"
            )

        if not messagebox.askyesno("Confirmar Eliminación", prompt, parent=self.parent_for_dialogs):
            return

        for idx in sorted_indices_desc:
            self.generated_models_data.pop(idx)

        removed_names = [selected_map[idx] for idx in sorted_indices_asc]

        self.selected_model_in_treeview = None
        self.selected_models_in_treeview = []
        self._update_models_treeview()

        if self.generated_models_data:
            next_index = min(sorted_indices_asc[0], len(self.generated_models_data) - 1)
            if next_index >= 0:
                self.treeview_lista_modelos.selection_set(str(next_index))
                self.treeview_lista_modelos.focus(str(next_index))
                self._on_model_select_from_treeview()
        else:
            remaining_items = self.treeview_lista_modelos.get_children()
            if remaining_items:
                self.treeview_lista_modelos.selection_remove(*remaining_items)
            if self.btn_oos_calibration:
                self.btn_oos_calibration.config(state=tk.DISABLED)
            if self.btn_collinearity_diag:
                self.btn_collinearity_diag.config(state=tk.DISABLED)
            if self.btn_nomogram:
                self.btn_nomogram.config(state=tk.DISABLED)
            if self.btn_delete_model:
                self.btn_delete_model.config(state=tk.DISABLED)
            if self.entry_custom_model_name:
                self.entry_custom_model_name_var.set("")
            if self.text_custom_model_notes:
                self.text_custom_model_notes.config(state=tk.NORMAL)
                self.text_custom_model_notes.delete("1.0", tk.END)
                self.text_custom_model_notes.config(state=tk.DISABLED)
            self._update_results_buttons_state()

        self.log(f"Modelos eliminados: {', '.join(removed_names)}", "INFO")

    def _clear_all_generated_models(self):
        """Elimina todos los modelos generados de la lista y actualiza la Treeview."""
        if messagebox.askyesno("Confirmar Limpieza", "¿Está seguro de que desea eliminar todos los modelos generados?", parent=self.parent_for_dialogs):
            self.generated_models_data = []
            self._update_models_treeview()
            self.selected_model_in_treeview = None
            self.selected_models_in_treeview = []
            if self.btn_oos_calibration:
                self.btn_oos_calibration.config(state=tk.DISABLED)
            if self.btn_delete_model:
                self.btn_delete_model.config(state=tk.DISABLED)
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
            "Riesgo Acumulado Base H₀(t)": self.show_cumulative_baseline_hazard,
            "Gráf. Schoenfeld": self.show_schoenfeld,
            "Incidencia Acumulada Base F₀(t)": self.show_baseline_cumulative_incidence, # New entry
            "Forest Plot (HRs)": self.generar_forest_plot,
            "Gráf. Calibración": self.generate_calibration_plot,
            "Gráf. Brier / IBS": self.generate_brier_plot,
            "Análisis de Efecto de Covariable(s)": self.show_variable_impact_plot,
            "Impacto en tiempo elegido S(t) / 1-S(t)": self.show_variable_impact_at_time_plot
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

        report_full += f"2. Datos Usados (post-preparación para este modelo):\n   - Observaciones totales: {num_obs_rep_meth}\n   - Eventos: {num_events_rep_meth}\n"

        # Holdout info
        test_prop_rep = md_rep.get('test_proportion') or md_rep.get('metrics', {}).get('Test Proportion')
        if test_prop_rep and pd.notna(test_prop_rep) and float(test_prop_rep) > 0:
            n_total = int(num_obs_rep_meth) if isinstance(num_obs_rep_meth, (int, float)) else 0
            n_test = int(round(n_total * float(test_prop_rep))) if n_total > 0 else 'N/A'
            n_train = (n_total - n_test) if isinstance(n_test, int) else 'N/A'
            report_full += f"   - Proporción test: {float(test_prop_rep):.0%}\n"
            report_full += f"   - Casos entrenamiento: {n_train}\n"
            report_full += f"   - Casos prueba (test): {n_test}\n"
            c_test_val = md_rep.get('c_index_test') or md_rep.get('metrics', {}).get('C-Index (Test)')
            c_test_ci = md_rep.get('c_index_test_ci') or md_rep.get('metrics', {}).get('C-Index (Test) CI')
            c_gap_val = md_rep.get('c_index_gap') or md_rep.get('metrics', {}).get('C-Index Gap (Test-Train)')
            if c_test_val and pd.notna(c_test_val):
                report_full += f"   - C-Index (Test): {self._format_c_index_display(c_test_val, c_test_ci, decimals=4)}\n"
            if c_gap_val and pd.notna(c_gap_val):
                report_full += f"   - Δ C-Index (Test-Train): {float(c_gap_val):.4f}\n"
            report_full += "   Nota: todas las métricas (AIC, Wald, Schoenfeld) provienen del modelo entrenado solo con el subset de entrenamiento.\n"
        report_full += "\n"
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
    
    root.mainloop()

