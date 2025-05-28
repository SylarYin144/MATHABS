#!/usr/bin/env python
# -*- coding: utf-8 -*-

# --- Importaciones Estándar de Python ---
from sklearn.model_selection import KFold
from lifelines.utils import concordance_index
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
from lifelines import CoxPHFitter, KaplanMeierFitter, WeibullAFTFitter, ExponentialFitter, LogNormalAFTFitter, LogLogisticAFTFitter
# from lifelines.fitters import GompertzFitter # Comentado debido a ImportError. Considerar GeneralizedGompertzRegressionFitter o GompertzRegressionFitter si es necesario.
from lifelines.statistics import proportional_hazard_test

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
    from patsy import dmatrix, DesignInfo, Treatment # Añadido DesignInfo y Treatment
    PATSY_AVAILABLE = True
except ImportError:
    dmatrix = None
    DesignInfo = None
    Treatment = None
    PATSY_AVAILABLE = False
    print("ADVERTENCIA: 'patsy' no instalada. Funciones de Spline y manejo avanzado de categóricas limitadas.")

try:
    from MATLAB_filter_component import FilterComponent
    FILTER_COMPONENT_AVAILABLE = True
except ImportError:
    FilterComponent = None
    FILTER_COMPONENT_AVAILABLE = False
    print("ERROR: No se pudo importar MATLAB_filter_component. Filtros avanzados no disponibles.")

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
    # Asegurar que la columna de evento sea 0/1
    unique_events = df_processed[final_event_col].dropna().unique()
    if not all(val in [0, 1] for val in unique_events):
        print(f"Warning: Event column '{final_event_col}' contains values other than 0 and 1: {unique_events}. Coercing to 0/1. Verify logic.")
        df_processed[final_event_col] = df_processed[final_event_col].apply(lambda x: 1 if pd.notnull(x) and x != 0 else (0 if pd.notnull(x) and x == 0 else np.nan))

    print(f"Using time column: '{final_time_col}', event column: '{final_event_col}'.")

    # V. Preprocesamiento y Manejo de Covariables
    formula_parts = []
    original_vars_in_formula = set() # Para rastrear las variables originales usadas en la fórmula

    for cov_name in covariates:
        config = cov_configs.get(cov_name, {})
        original_vars_in_formula.add(cov_name)

        if config.get('type') == 'Qualitative':
            if not PATSY_AVAILABLE:
                raise ImportError("Patsy is required for qualitative covariates. Please install it.")
            ref_cat = config.get('refCategory')
            if ref_cat:
                # Asegurar que ref_cat esté entre comillas si contiene caracteres especiales para patsy
                safe_ref_cat = f"'{ref_cat}'" if not str(ref_cat).isalnum() else ref_cat
                formula_parts.append(f"C({cov_name}, Treatment(reference={safe_ref_cat}))")
            else:
                formula_parts.append(f"C({cov_name})") # Patsy elige la referencia
        elif config.get('splineConfig'):
            if not PATSY_AVAILABLE:
                raise ImportError("Patsy is required for spline covariates. Please install it.")
            sc = config['splineConfig']
            spline_func = 'bs' if sc.get('type') == 'B-spline' else 'ns' # ns para Natural, bs para B-spline
            degree_str = f", degree={sc['degree']}" if spline_func == 'bs' and 'degree' in sc else ""
            formula_parts.append(f"{spline_func}({cov_name}, df={sc['df']}{degree_str})")
        else: # Cuantitativa sin splines, o tipo no especificado (tratar como está)
            df_processed[cov_name] = pd.to_numeric(df_processed[cov_name], errors='coerce') # Asegurar numérico
            formula_parts.append(cov_name)
        
        if config.get('isTDC'):
            print(f"Note: {cov_name} marked as TDC. Specific formula adjustment might be needed for time-varying effects.")

    # Construir la fórmula completa para patsy
    if not formula_parts:
        full_formula_str = "1" # Modelo nulo si no hay covariables
    else:
        full_formula_str = ' + '.join(formula_parts)

    print(f"Patsy formula to be used: {full_formula_str}")

    design_matrix = pd.DataFrame()
    processed_formula_terms = []

    if PATSY_AVAILABLE:
        try:
            # Subconjunto de df_processed a solo las columnas necesarias para dmatrix para evitar problemas con otros NaNs
            required_cols_for_patsy = list(original_vars_in_formula)
            # Asegurarse de que las columnas requeridas existan antes de subconjuntar
            required_cols_for_patsy = [col for col in required_cols_for_patsy if col in df_processed.columns]
            df_for_patsy = df_processed[required_cols_for_patsy].copy()

            # Usando 'return_type="dataframe"' da nombres de columna que lifelines puede analizar.
            # NA_action='drop' (predeterminado) o 'raise'. Si es 'drop', los índices podrían no alinearse con el df original.
            design_matrix = dmatrix(full_formula_str, df_for_patsy, return_type='dataframe', NA_action='drop')
            
            # La columna 'Intercept' de patsy a menudo no es necesaria si lifelines añade la suya propia,
            # o para los modelos de Cox (ya que la función de riesgo base la absorbe).
            if 'Intercept' in design_matrix.columns and full_formula_str != "1":
                design_matrix = design_matrix.drop(columns=['Intercept'])
            
            processed_formula_terms = design_matrix.columns.tolist()
            
            # Alinear el df original con la design_matrix (debido a la eliminación de NA de patsy)
            # df_processed será la concatenación de la matriz de diseño y las columnas esenciales (tiempo, evento, pesos)
            # del df original, alineadas por índice.
            df_essential_cols = df_processed[[final_time_col, final_event_col] + ([weight_col] if weight_col else [])].loc[design_matrix.index]
            df_processed = pd.concat([df_essential_cols, design_matrix], axis=1)

            print(f"Design matrix created with {df_processed.shape[0]} rows after patsy NA handling for covariates.")
            print(f"Processed formula terms for lifelines: {processed_formula_terms}")

        except Exception as e_patsy:
            print(f"Error creating design matrix with patsy: {e_patsy}")
            print("Consider checking covariate types, NaNs, or formula syntax. Falling back to raw covariates.")
            # Fallback si patsy falla: usar covariables seleccionadas en bruto (menos robusto)
            processed_formula_terms = [c for c in covariates if c in df_processed.columns] # Asegurar que existan
            df_processed = df_processed[[final_time_col, final_event_col] + ([weight_col] if weight_col else []) + processed_formula_terms].copy()
            # Aplicar eliminación por lista para estas columnas específicas si patsy falló
            df_processed.dropna(subset=[final_time_col, final_event_col] + processed_formula_terms, inplace=True)
            print(f"Patsy fallback: Using raw covariates. {df_processed.shape[0]} rows after basic NA drop.")
    else:
        print("Patsy not available. Using raw covariates and basic NA handling.")
        processed_formula_terms = [c for c in covariates if c in df_processed.columns]
        df_processed = df_processed[[final_time_col, final_event_col] + ([weight_col] if weight_col else []) + processed_formula_terms].copy()
        # Asegurar que las columnas de covariables sean numéricas
        for col in processed_formula_terms:
            df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')


    # VI. Manejo de Datos Faltantes (para Tiempo, Evento, Pesos y covariables no procesadas por patsy)
    essential_cols_for_na_check = [final_time_col, final_event_col]
    if weight_col and weight_col in df_processed.columns:
        essential_cols_for_na_check.append(weight_col)
    
    # Añadir términos de covariables que ahora son columnas en df_processed
    essential_cols_for_na_check.extend(processed_formula_terms)

    # Asegurar que todas las columnas para la verificación de NA existan realmente en df_processed
    essential_cols_for_na_check = [col for col in essential_cols_for_na_check if col in df_processed.columns]

    initial_rows = df_processed.shape[0]
    if missing_strategy == 'ListwiseDeletion':
        df_processed.dropna(subset=essential_cols_for_na_check, inplace=True)
        print(f"Listwise deletion applied. Rows removed: {initial_rows - df_processed.shape[0]}. Final rows: {df_processed.shape[0]}")
    elif missing_strategy in ['MeanImputation', 'MedianImputation']:
        # Lógica de imputación para columnas de COVARIABLES especificadas (solo numéricas)
        # Esto es complejo si patsy ya creó términos dummy/spline.
        # Generalmente, NA_action='drop' de patsy maneja la falta de datos en las variables *usadas en la fórmula*.
        # Esta MISSING_DATA_STRATEGY podría ser más sobre las columnas originales de tiempo/evento/peso
        # o si se tomó una ruta de procesamiento de covariables sin patsy.
        # Para simplificar aquí, asumiendo que el manejo de NA de patsy es primario para las covariables.
        # Si no se usa patsy, entonces iterar sobre processed_formula_terms:
        for term in processed_formula_terms:
            if pd.api.types.is_numeric_dtype(df_processed[term]):
                if df_processed[term].isnull().any():
                    fill_value = df_processed[term].mean() if missing_strategy == 'MeanImputation' else df_processed[term].median()
                    df_processed[term].fillna(fill_value, inplace=True)
                    print(f"Imputed NaNs in '{term}' with {missing_strategy.lower().replace('imputation','')}.")
        # Después de la imputación, aún podría ser necesario eliminar filas si tiempo/evento/peso son NA:
        df_processed.dropna(subset=[final_time_col, final_event_col] + ([weight_col] if weight_col else []), inplace=True)
        print(f"{missing_strategy} applied. Final rows: {df_processed.shape[0]}")


    if df_processed.empty:
        raise ValueError("DataFrame is empty after preprocessing and NA handling. Cannot proceed to modeling.")

    return df_processed, processed_formula_terms, final_time_col, final_event_col

def fit_survival_model(df_processed, final_time_col, final_event_col, processed_formula_terms, model_config_params):
    """
    Crea y ajusta modelos de supervivencia (CoxPH o modelos paramétricos especificados) a los datos preprocesados.
    """
    model_type = model_config_params.get('parametricModelType', 'CoxPH')
    fitter_instance = None

    # I. Instanciación y Ajuste del Modelo
    if model_type == 'CoxPH':
        pen_type = model_config_params.get('penalizationType', 'None')
        pen_strength = model_config_params.get('penalizer_strength', 0.1) if pen_type != 'None' else 0.0
        l1_r = 0.0
        if pen_type == 'Lasso': l1_r = 1.0
        elif pen_type == 'ElasticNet': l1_r = model_config_params.get('l1_ratio_for_elasticnet', 0.5)
        
        fitter_instance = CoxPHFitter(
            penalizer=pen_strength,
            l1_ratio=l1_r,
            tie_method=model_config_params.get('tieHandlingMethod', 'efron')
        )
    elif model_type == 'Weibull':
        fitter_instance = WeibullAFTFitter()
    elif model_type == 'Exponential':
        fitter_instance = ExponentialFitter()
    elif model_type == 'LogNormal':
        fitter_instance = LogNormalAFTFitter()
    elif model_type == 'LogLogistic':
        fitter_instance = LogLogisticAFTFitter()
    elif model_type == 'Gompertz':
        print("Warning: GompertzFitter not directly available. Skipping Gompertz model. Consider using GeneralizedGompertzRegressionFitter if available in your lifelines version.")
        return None # No se puede instanciar el fitter, retornar None
    # elif model_type == 'GeneralizedGamma':
    #     fitter_instance = GeneralizedGammaRegressionFitter() # Si está disponible y es el tipo correcto
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    formula_to_fit = " + ".join(processed_formula_terms) if processed_formula_terms else "1"

    try:
        fitter_instance.fit(
            df_processed,
            duration_col=final_time_col,
            event_col=final_event_col,
            formula=formula_to_fit,
            strata=model_config_params.get('stratificationColumns'),
            weights_col=model_config_params.get('weightColumn'),
            robust=(model_type == 'CoxPH') # Errores estándar robustos principalmente para Cox
        )
        print(f"Successfully fitted {model_type} model.")
        return fitter_instance
    except Exception as e:
        print(f"Error fitting {model_type} model: {e}")
        return None

def extract_model_metrics(fitted_model, model_config_params):
    """
    Extrae métricas clave de rendimiento del modelo y estadísticas de resumen del modelo ajustado.
    """
    metrics = {}
    if fitted_model is None:
        return metrics

    # II. Extracción de Métricas Centrales
    summary_df_raw = fitted_model.summary.copy()

    # Formatear valores p en el resumen para el almacenamiento si se replica el formato de cadena exacto de la UI
    if 'p' in summary_df_raw.columns:
        summary_df_raw['p_display'] = summary_df_raw['p'].apply(format_p_value)

    metrics['summaryDf'] = summary_df_raw.reset_index().to_dict('records')

    metrics['cIndex'] = getattr(fitted_model, 'concordance_index_', None)
    metrics['logLikelihood'] = getattr(fitted_model, 'log_likelihood_', None)
    metrics['aic'] = getattr(fitted_model, 'AIC_', None)

    if hasattr(fitted_model, 'log_likelihood_ratio_test'):
        metrics['waldPValueGlobal'] = fitted_model.log_likelihood_ratio_test().p_value
    elif hasattr(fitted_model, 'LRT_p_value_'):
         metrics['waldPValueGlobal'] = fitted_model.LRT_p_value_

    if hasattr(fitted_model, 'durations'):
        metrics['totalObservations'] = fitted_model.durations.shape[0]
        metrics['numberOfEvents'] = fitted_model.event_observed.sum()
        metrics['numberOfCensures'] = metrics['totalObservations'] - metrics['numberOfEvents']
        # Si se usaron pesos:
        if model_config_params.get('weightColumn') and hasattr(fitted_model, 'weights'):
            metrics['totalObservations'] = fitted_model.weights.sum()
            metrics['numberOfEvents'] = fitted_model.weights[fitted_model.event_observed].sum()

    return metrics

def check_ph_assumption(fitted_model, df_processed):
    """
    Evalúa la suposición de riesgos proporcionales (PH) utilizando los residuos de Schoenfeld.
    """
    schoenfeld_results = {}
    if isinstance(fitted_model, CoxPHFitter) and hasattr(fitted_model, 'params_') and not fitted_model.params_.empty:
        try:
            # Usar proportional_hazard_test para obtener p-valores individuales
            ph_test_results = proportional_hazard_test(fitted_model, df_processed, time_transform='log')
            
            schoenfeld_p_values_individual = {}
            if not ph_test_results.summary.empty:
                for cov_name, row_data in ph_test_results.summary.iterrows():
                    schoenfeld_p_values_individual[str(cov_name)] = row_data['p']
            
            schoenfeld_results['schoenfeldPValuesIndividual'] = schoenfeld_p_values_individual
            
            # Para el p-valor global, si proportional_hazard_test lo proporciona directamente
            if hasattr(ph_test_results, 'global_test_result') and hasattr(ph_test_results.global_test_result, 'p_value'):
                schoenfeld_results['schoenfeldPValueGlobal'] = ph_test_results.global_test_result.p_value
            else:
                # Si no hay un p-valor global directo, se puede simular con el mínimo de los individuales
                # Esto NO es estadísticamente riguroso, es una simplificación para la simulación de la UI
                if schoenfeld_p_values_individual:
                    schoenfeld_results['schoenfeldPValueGlobal'] = min(schoenfeld_p_values_individual.values())
                else:
                    schoenfeld_results['schoenfeldPValueGlobal'] = None

        except Exception as e_ph:
            print(f"PH assumption check failed: {e_ph}")
            schoenfeld_results['schoenfeldPValueGlobal'] = None
            schoenfeld_results['schoenfeldPValuesIndividual'] = {}
    else:
        print("PH assumption check skipped: Model is not CoxPHFitter or not fitted.")
        schoenfeld_results['schoenfeldPValueGlobal'] = None
        schoenfeld_results['schoenfeldPValuesIndividual'] = {}
    return schoenfeld_results

def perform_cindex_cross_validation(fitter_class, fitter_params, df_processed, final_time_col, final_event_col, processed_formula_terms, model_config_params):
    """
    Estima el rendimiento del modelo fuera de la muestra utilizando la validación cruzada k-fold para el índice C.
    """
    cindex_cv_score = None
    k_val = model_config_params.get('kFolds')
    model_type = model_config_params.get('parametricModelType', 'CoxPH')

    if k_val and k_val > 1:
        try:
            formula_to_fit = " + ".join(processed_formula_terms) if processed_formula_terms else "1"
            cv_scores = k_fold_cross_validation(
                fitter_class(**fitter_params), # Pasar argumentos del constructor para el fitter
                df_processed,
                duration_col=final_time_col,
                event_col=final_event_col,
                formula=formula_to_fit,
                k=int(k_val),
                scoring_method="concordance_index",
                fitter_kwargs={ # Argumentos pasados al método .fit() dentro de cada fold
                    'strata': model_config_params.get('stratificationColumns'),
                    'weights_col': model_config_params.get('weightColumn'),
                    'robust': (model_type == 'CoxPH')
                }
            )
            cindex_cv_score = np.mean(cv_scores)
            print(f"Cross-validation C-Index: {cindex_cv_score:.4f}")
        except Exception as e_cv:
            print(f"Cross-validation failed: {e_cv}")
    else:
        print("Cross-validation skipped: kFolds not specified or less than 2.")
    return cindex_cv_score

def generate_plot_data(fitted_model, df_processed, final_time_col, final_event_col, model_config_params):
    """
    Genera los datos necesarios para varias parcelas de diagnóstico y resultados.
    """
    plot_data = {}
    if fitted_model is None:
        return plot_data

    # V. Generación de Datos para Gráficos
    # 1. Supervivencia/Riesgo Base
    if hasattr(fitted_model, 'baseline_survival_'):
        plot_data['baselineSurvival'] = fitted_model.baseline_survival_.reset_index().to_dict('records')
    if hasattr(fitted_model, 'baseline_hazard_'):
        plot_data['baselineHazard'] = fitted_model.baseline_hazard_.reset_index().to_dict('records')

    # 2. Residuos de Schoenfeld (para CoxPH)
    if isinstance(fitted_model, CoxPHFitter):
        try:
            schoenfeld_df = fitted_model.compute_residuals(df_processed, kind='schoenfeld')
            schoenfeld_plot_data = []
            for col in schoenfeld_df.columns:
                schoenfeld_plot_data.append({
                    'variable': col,
                    'data': [{'time': t, 'residual': r} for t, r in schoenfeld_df[col].items()]
                })
            plot_data['schoenfeldResiduals'] = schoenfeld_plot_data
        except Exception as e:
            print(f"Error computing Schoenfeld residuals: {e}")

    # 3. Datos del Gráfico de Bosque (Forest Plot)
    if hasattr(fitted_model, 'summary'):
        forest_df = fitted_model.summary[['exp(coef)', 'exp(coef) lower 95%', 'exp(coef) upper 95%', 'p']].copy()
        forest_df = forest_df.reset_index().rename(columns={
            'index': 'name',
            'exp(coef)': 'hr',
            'exp(coef) lower 95%': 'lower',
            'exp(coef) upper 95%': 'upper',
            'p': 'pValue'
        })
        forest_df['pValue_formatted'] = forest_df['pValue'].apply(format_p_value)
        plot_data['forestPlot'] = forest_df.to_dict('records')

    # 4. Gráfico Log-Menos-Log (para CoxPH, típicamente con estratos o covariables categóricas)
    if isinstance(fitted_model, CoxPHFitter):
        strat_cols = model_config_params.get('stratificationColumns', [])
        if strat_cols: # Solo si hay columnas de estratificación
            log_minus_log_data_groups = []
            # Usar la primera columna de estratificación para el ejemplo
            strat_col = strat_cols[0]
            if strat_col in df_processed.columns:
                for group_val, group_df in df_processed.groupby(strat_col):
                    kmf_group = KaplanMeierFitter().fit(group_df[final_time_col], group_df[final_event_col])
                    valid_survival = kmf_group.survival_function_[ (kmf_group.survival_function_.iloc[:,0] > 0) & (kmf_group.survival_function_.iloc[:,0] < 1) ]
                    if not valid_survival.empty:
                        log_time = np.log(valid_survival.index)
                        log_neg_log_survival = np.log(-np.log(valid_survival.iloc[:,0]))
                        log_minus_log_data_groups.append({
                            'group': str(group_val), # Convertir a cadena para JSON
                            'data': [{'logTime': lt, 'logMinusLogSurvival': lnls} for lt, lnls in zip(log_time, log_neg_log_survival)]
                        })
                plot_data['logMinusLogPlot'] = log_minus_log_data_groups
            else:
                print(f"Warning: Stratification column '{strat_col}' not found for Log-Minus-Log plot.")

    # 5. Residuos de Martingala (para CoxPH)
    if isinstance(fitted_model, CoxPHFitter):
        try:
            martingale_res = fitted_model.compute_residuals(df_processed, kind='martingale')
            plot_data['martingaleResiduals'] = martingale_res.reset_index().rename(columns={'index': 'observation_id', 0: 'residual'}).to_dict('records')
        except Exception as e:
            print(f"Error computing Martingale residuals: {e}")

    # 6. Residuos DFBETA (para CoxPH)
    if isinstance(fitted_model, CoxPHFitter):
        try:
            dfbeta_res = fitted_model.compute_residuals(df_processed, kind='dfbeta')
            dfbeta_plot_data = []
            for col in dfbeta_res.columns:
                dfbeta_plot_data.append({
                    'variable': col,
                    'data': [{'observation_id': idx, 'residual': r} for idx, r in dfbeta_res[col].items()]
                })
            plot_data['dfbetaResiduals'] = dfbeta_plot_data
        except Exception as e:
            print(f"Error computing DFBETA residuals: {e}")

    # 7. Datos del Gráfico de Calibración (Punteros)
    # La generación de datos para el gráfico de calibración es más compleja y a menudo implica
    # la predicción de probabilidades de supervivencia y la comparación con la supervivencia observada
    # en grupos de riesgo. lifelines.calibration.survival_probability_calibration_plot
    # es para graficar directamente, no para extraer datos fácilmente en un formato genérico.
    # Se omite la implementación detallada aquí por su complejidad y la necesidad de definir
    # tiempos específicos para la predicción y la agrupación.

    return plot_data

def run_survival_analysis(file_path, time_column, event_column, selected_covariates, covariate_configs,
                          missing_data_strategy, model_parameters,
                          renamed_time_column=None, renamed_event_column=None,
                          weight_column=None, advanced_filters_config=None):
    """
    Función principal para ejecutar el análisis de supervivencia de principio a fin.
    """
    print("\n--- Starting Survival Analysis ---")
    
    # 1. Carga de datos
    raw_df = load_data(file_path)
    if raw_df is None or raw_df.empty:
        print("Failed to load data or data is empty. Exiting.")
        return {"status": "error", "message": "Failed to load data or data is empty."}

    # 2. Aplicar filtros avanzados (opcional)
    filtered_df = apply_advanced_filters(raw_df, advanced_filters_config if advanced_filters_config else {})

    # 3. Preprocesamiento de datos
    try:
        df_processed, processed_formula_terms, final_time_col, final_event_col = \
            preprocess_data(filtered_df, time_column, event_column, selected_covariates, covariate_configs,
                            weight_column, missing_data_strategy, renamed_time_column, renamed_event_column)
    except ValueError as ve:
        print(f"Data preprocessing error: {ve}")
        return {"status": "error", "message": f"Data preprocessing error: {ve}"}
    except ImportError as ie:
        print(f"Missing dependency for preprocessing: {ie}")
        return {"status": "error", "message": f"Missing dependency for preprocessing: {ie}"}

    if df_processed.empty:
        print("DataFrame is empty after preprocessing. Cannot proceed to modeling.")
        return {"status": "error", "message": "DataFrame is empty after preprocessing."}

    # 4. Ajuste del modelo
    fitted_model = fit_survival_model(df_processed, final_time_col, final_event_col, processed_formula_terms, model_parameters)
    if fitted_model is None:
        print("Failed to fit survival model. Exiting.")
        return {"status": "error", "message": "Failed to fit survival model."}

    # 5. Extracción de métricas
    model_metrics = extract_model_metrics(fitted_model, model_parameters)

    # 6. Verificación de la suposición de riesgos proporcionales (solo para CoxPH)
    ph_assumption_results = check_ph_assumption(fitted_model, df_processed)
    model_metrics.update(ph_assumption_results)

    # 7. Validación cruzada para C-Index
    fitter_class_for_cv = type(fitted_model) # Obtener la clase del fitter ajustado
    fitter_params_for_cv = fitted_model.params # Obtener los parámetros del constructor del fitter
    cindex_cv = perform_cindex_cross_validation(fitter_class_for_cv, fitter_params_for_cv,
                                                df_processed, final_time_col, final_event_col,
                                                processed_formula_terms, model_parameters)
    if cindex_cv is not None:
        model_metrics['cIndexCV'] = cindex_cv

    # 8. Generación de datos para gráficos
    plot_data = generate_plot_data(fitted_model, df_processed, final_time_col, final_event_col, model_parameters)

    # 9. Finalizar el objeto ModelResult y el texto de resumen metodológico
    # Aquí se ensamblaría el objeto ModelResult final.
    # Para este ejemplo, solo devolvemos un diccionario con los resultados clave.
    
    # Generar un resumen textual del modelo
    full_summary_text = ""
    if fitted_model:
        try:
            # Capturar la salida de print_summary
            import io
            from contextlib import redirect_stdout
            f = io.StringIO()
            with redirect_stdout(f):
                fitted_model.print_summary(decimals=4, style="ascii")
            full_summary_text = f.getvalue()
        except Exception as e:
            full_summary_text = f"Could not generate full summary text: {e}"

    model_result = {
        "status": "success",
        "message": "Survival analysis completed successfully.",
        "metrics": model_metrics,
        "plotData": plot_data,
        "fullSummaryText": full_summary_text,
        "fittedModel": fitted_model # Opcional: devolver el objeto del modelo ajustado si es necesario para uso posterior
    }

    print("\n--- Survival Analysis Completed ---")
    return model_result

# --- Main script execution flow (example) ---
if __name__ == "__main__":
    # --- Global Configuration Variables (mirrors UI selections) ---
    # Estas se establecerían en función de la entrada del usuario en un script real
    # Para la demostración, usaremos valores de ejemplo.
    # Asegúrate de tener un archivo CSV o XLSX de ejemplo en la ruta especificada.
    # Por ejemplo, puedes crear un 'survival_data.csv' simple:
    # time,event,age,sex,treatment
    # 10,1,60,Male,A
    # 12,0,65,Female,B
    # 8,1,55,Male,A
    # 15,1,70,Female,B
    # 7,0,50,Male,A
    # 20,1,75,Female,A
    # 11,0,62,Male,B
    # 14,1,68,Female,A
    # 9,1,58,Male,B
    # 18,0,72,Female,B

    # Puedes cambiar esta ruta a un archivo de prueba real
    EXAMPLE_FILE_PATH = "survival_data.csv"
    # Crea un archivo survival_data.csv en la raíz de d:/APPS o ajusta la ruta
    # con el contenido de ejemplo de arriba.

    # Configuración de ejemplo (simulando la entrada de la UI)
    TIME_COLUMN = "time"
    EVENT_COLUMN = "event"
    RENAMED_TIME_COLUMN = "duration_months" # Opcional
    RENAMED_EVENT_COLUMN = "outcome_event" # Opcional
    SELECTED_COVARIATES = ["age", "sex", "treatment"] # Lista de nombres de covariables originales
    COVARIATE_CONFIGS = { # Diccionario: {'cov_name': {'type': 'Quantitative'/'Qualitative', 'refCategory': '...', 'splineConfig': {...}, 'isTDC': False}}
        'age': {'type': 'Quantitative', 'splineConfig': {'type': 'Natural', 'df': 3}},
        'sex': {'type': 'Qualitative', 'refCategory': 'Male'},
        'treatment': {'type': 'Qualitative', 'refCategory': 'A'}
    }
    WEIGHT_COLUMN = None # Opcional
    MISSING_DATA_STRATEGY = 'ListwiseDeletion' # 'MeanImputation', 'MedianImputation'
    ADVANCED_FILTERS_CONFIG = { # Diccionario: {'col_name': 'filter_rule_string'}
        'age': '50-70',
        'sex': 'Female'
    }
    MODEL_PARAMETERS = {
        'parametricModelType': 'CoxPH', # 'CoxPH', 'Weibull', 'Exponential', etc.
        'penalizationType': 'None', # 'Ridge', 'Lasso', 'ElasticNet', 'None'
        'penalizer_strength': 0.1, # Solo si penalizationType no es 'None'
        'l1_ratio_for_elasticnet': 0.5, # Solo si penalizationType es 'ElasticNet'
        'stratificationColumns': [], # Lista de columnas para estratificación
        'weightColumn': None, # Nombre de la columna de pesos
        'tieHandlingMethod': 'efron', # 'efron', 'breslow', 'exact'
        'kFolds': 5 # Para validación cruzada
    }

    # Ejecutar el análisis
    results = run_survival_analysis(
        file_path=EXAMPLE_FILE_PATH,
        time_column=TIME_COLUMN,
        event_column=EVENT_COLUMN,
        selected_covariates=SELECTED_COVARIATES,
        covariate_configs=COVARIATE_CONFIGS,
        missing_data_strategy=MISSING_DATA_STRATEGY,
        model_parameters=MODEL_PARAMETERS,
        renamed_time_column=RENAMED_TIME_COLUMN,
        renamed_event_column=RENAMED_EVENT_COLUMN,
        weight_column=WEIGHT_COLUMN,
        advanced_filters_config=ADVANCED_FILTERS_CONFIG
    )

    if results["status"] == "success":
        print("\n--- Analysis Results Summary ---")
        print(f"Model Type: {MODEL_PARAMETERS['parametricModelType']}")
        print(f"C-Index: {results['metrics'].get('cIndex'):.4f}")
        if 'cIndexCV' in results['metrics']:
            print(f"Cross-validated C-Index: {results['metrics']['cIndexCV']:.4f}")
        print(f"Total Observations: {results['metrics'].get('totalObservations')}")
        print(f"Number of Events: {results['metrics'].get('numberOfEvents')}")
        print("\n--- Full Model Summary ---")
        print(results['fullSummaryText'])
        
        print("\n--- Plot Data Keys ---")
        for key in results['plotData'].keys():
            print(f"- {key}")

        # --- Generación de Gráficos de Ejemplo ---
        print("\n--- Generating Example Plots ---")

        # 1. Gráfico de Supervivencia Base
        if 'baselineSurvival' in results['plotData'] and results['plotData']['baselineSurvival']:
            df_bs = pd.DataFrame(results['plotData']['baselineSurvival'])
            if not df_bs.empty and 'timeline' in df_bs.columns and 'baseline survival' in df_bs.columns:
                plt.figure(figsize=(8, 6))
                plt.plot(df_bs['timeline'], df_bs['baseline survival'])
                plt.title('Baseline Survival Function')
                plt.xlabel('Time')
                plt.ylabel('Survival Probability')
                plt.grid(True)
                plt.show()
            else:
                print("Could not plot Baseline Survival: Data is empty or missing required columns.")

        # 2. Gráfico de Riesgo Base
        if 'baselineHazard' in results['plotData'] and results['plotData']['baselineHazard']:
            df_bh = pd.DataFrame(results['plotData']['baselineHazard'])
            if not df_bh.empty and 'timeline' in df_bh.columns and 'baseline hazard' in df_bh.columns:
                plt.figure(figsize=(8, 6))
                plt.plot(df_bh['timeline'], df_bh['baseline hazard'])
                plt.title('Baseline Hazard Function')
                plt.xlabel('Time')
                plt.ylabel('Hazard Rate')
                plt.grid(True)
                plt.show()
            else:
                print("Could not plot Baseline Hazard: Data is empty or missing required columns.")

        # 3. Gráfico de Bosque (Forest Plot)
        if 'forestPlot' in results['plotData'] and results['plotData']['forestPlot']:
            forest_data = results['plotData']['forestPlot']
            if forest_data:
                names = [d['name'] for d in forest_data]
                hrs = [d['hr'] for d in forest_data]
                lowers = [d['lower'] for d in forest_data]
                uppers = [d['upper'] for d in forest_data]

                plt.figure(figsize=(10, len(names) * 0.6))
                y_pos = np.arange(len(names))
                
                # Plot HRs
                plt.scatter(hrs, y_pos, color='blue', zorder=3)
                
                # Plot CIs
                for i, (lower, upper) in enumerate(zip(lowers, uppers)):
                    plt.plot([lower, upper], [y_pos[i], y_pos[i]], color='gray', linestyle='-', linewidth=1, zorder=2)
                
                plt.axvline(x=1, color='red', linestyle='--', linewidth=0.8, label='HR = 1 (No Effect)')
                plt.yticks(y_pos, names)
                plt.xlabel('Hazard Ratio (exp(coef))')
                plt.title('Forest Plot of Hazard Ratios')
                plt.xscale('log') # HRs are often on a log scale
                plt.grid(True, which="both", ls="--", c='0.7')
                plt.legend()
                plt.tight_layout()
                plt.show()
            else:
                print("Could not plot Forest Plot: Data is empty.")

        # 4. Gráfico Log-Menos-Log (si hay datos de estratificación)
        if 'logMinusLogPlot' in results['plotData'] and results['plotData']['logMinusLogPlot']:
            log_minus_log_groups = results['plotData']['logMinusLogPlot']
            if log_minus_log_groups:
                plt.figure(figsize=(8, 6))
                for group_data in log_minus_log_groups:
                    group_name = group_data['group']
                    data_points = group_data['data']
                    if data_points:
                        log_times = [dp['logTime'] for dp in data_points]
                        log_minus_log_survivals = [dp['logMinusLogSurvival'] for dp in data_points]
                        plt.plot(log_times, log_minus_log_survivals, label=f'Group: {group_name}')
                
                plt.title('Log-Minus-Log Plot (for PH Assumption)')
                plt.xlabel('Log(Time)')
                plt.ylabel('Log(-Log(Survival Probability))')
                plt.grid(True)
                plt.legend()
                plt.show()
            else:
                print("Could not plot Log-Minus-Log Plot: Data is empty.")

    else:
        print(f"\nAnalysis failed: {results['message']}")
