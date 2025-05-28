#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import traceback
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report, roc_auc_score
from statsmodels.graphics.gofplots import ProbPlot # Para Hosmer-Lemeshow gráfico si es necesario, o usar cálculo manual.
# Considerar una función directa para Hosmer-Lemeshow si existe o implementarla.

# FilterComponent ha sido eliminado.
FilterComponent = None # Mantener para evitar errores si alguna lógica residual lo verifica.

class LogisticRegressionTab(ttk.Frame):
    """
    Pestaña para realizar análisis de Regresión Logística.
    """
    def __init__(self, master):
        super().__init__(master)
        self.df_original = None
        self.df_filtered = None
        self.model_results = None

        # Variables de control Tkinter
        self.filepath_var = tk.StringVar()
        self.dependent_var = tk.StringVar()
        self.independent_vars = [] # Lista de nombres de variables independientes seleccionadas

        # Variables para Filtros Generales (hasta 2 filtros) - AÑADIDO
        self.filter_active_1_var = tk.BooleanVar(value=False)
        self.filter_col_1_var = tk.StringVar()
        self.filter_op_1_var = tk.StringVar()
        self.filter_val_1_var = tk.StringVar()

        self.filter_active_2_var = tk.BooleanVar(value=False)
        self.filter_col_2_var = tk.StringVar()
        self.filter_op_2_var = tk.StringVar()
        self.filter_val_2_var = tk.StringVar()
        
        self.general_filter_operators = ["==", "!=", ">", "<", ">=", "<=", "contiene", "no contiene", "es NaN", "no es NaN"]

        self._build_ui()

    def _build_ui(self):
        # --- Layout Principal (PanedWindow) ---
        main_pane = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        main_pane.pack(fill=tk.BOTH, expand=True)

        # --- Panel Izquierdo: Controles ---
        left_pane = ttk.Frame(main_pane)
        main_pane.add(left_pane, weight=1)

        # --- Panel Derecho: Resultados ---
        right_pane = ttk.Frame(main_pane)
        main_pane.add(right_pane, weight=2)

        # --- Controles en Panel Izquierdo ---
        controls_frame = ttk.LabelFrame(left_pane, text="Configuración Regresión Logística")
        controls_frame.pack(padx=10, pady=10, fill="both", expand=True)

        # 1. Carga de Archivo
        file_frame = ttk.LabelFrame(controls_frame, text="1. Carga de Datos")
        file_frame.pack(fill="x", padx=5, pady=5)
        path_frame = ttk.Frame(file_frame)
        path_frame.pack(fill="x", pady=2)
        ttk.Label(path_frame, text="Archivo:").pack(side="left", padx=5)
        ttk.Entry(path_frame, textvariable=self.filepath_var, width=40, state="readonly").pack(side="left", padx=5, expand=True, fill="x")
        ttk.Button(path_frame, text="Buscar", command=self._load_file).pack(side="left", padx=5)

        # 2. Filtros Generales (Implementación directa)
        frm_filters_general = ttk.LabelFrame(controls_frame, text="2. Filtros Generales (Opcional)")
        frm_filters_general.pack(fill="x", padx=5, pady=5)

        # Filtro 1
        f1_frame = ttk.Frame(frm_filters_general)
        f1_frame.pack(fill="x", pady=2)
        ttk.Checkbutton(f1_frame, text="Activar Filtro 1:", variable=self.filter_active_1_var).grid(row=0, column=0, padx=2, sticky="w")
        self.filter_col_1_combo = ttk.Combobox(f1_frame, textvariable=self.filter_col_1_var, state="readonly", width=15)
        self.filter_col_1_combo.grid(row=0, column=1, padx=2)
        self.filter_op_1_combo = ttk.Combobox(f1_frame, textvariable=self.filter_op_1_var, values=self.general_filter_operators, state="readonly", width=10)
        self.filter_op_1_combo.grid(row=0, column=2, padx=2)
        self.filter_op_1_combo.set("==")
        ttk.Entry(f1_frame, textvariable=self.filter_val_1_var, width=15).grid(row=0, column=3, padx=2)
        
        # Filtro 2
        f2_frame = ttk.Frame(frm_filters_general)
        f2_frame.pack(fill="x", pady=2)
        ttk.Checkbutton(f2_frame, text="Activar Filtro 2:", variable=self.filter_active_2_var).grid(row=0, column=0, padx=2, sticky="w")
        self.filter_col_2_combo = ttk.Combobox(f2_frame, textvariable=self.filter_col_2_var, state="readonly", width=15)
        self.filter_col_2_combo.grid(row=0, column=1, padx=2)
        self.filter_op_2_combo = ttk.Combobox(f2_frame, textvariable=self.filter_op_2_var, values=self.general_filter_operators, state="readonly", width=10)
        self.filter_op_2_combo.grid(row=0, column=2, padx=2)
        self.filter_op_2_combo.set("==")
        ttk.Entry(f2_frame, textvariable=self.filter_val_2_var, width=15).grid(row=0, column=3, padx=2)

        # 3. Selección de Variables
        vars_frame = ttk.LabelFrame(controls_frame, text="3. Selección de Variables")
        vars_frame.pack(fill="x", padx=5, pady=5)
        # Variable Dependiente (Binaria)
        ttk.Label(vars_frame, text="Dependiente (Binaria):").grid(row=0, column=0, padx=5, pady=2, sticky="w")
        self.dep_var_combo = ttk.Combobox(vars_frame, textvariable=self.dependent_var, state="readonly", width=25)
        self.dep_var_combo.grid(row=0, column=1, padx=5, pady=2, sticky="ew")
        self.dep_var_combo.bind("<<ComboboxSelected>>", self._validate_dependent_var)
        # Variables Independientes
        ttk.Label(vars_frame, text="Independientes:").grid(row=1, column=0, padx=5, pady=2, sticky="nw")
        list_frame = ttk.Frame(vars_frame)
        list_frame.grid(row=1, column=1, padx=5, pady=2, sticky="nsew")
        self.indep_var_listbox = tk.Listbox(list_frame, selectmode=tk.MULTIPLE, height=6, exportselection=False)
        indep_scroll = ttk.Scrollbar(list_frame, orient="vertical", command=self.indep_var_listbox.yview)
        self.indep_var_listbox.configure(yscrollcommand=indep_scroll.set)
        indep_scroll.pack(side="right", fill="y")
        self.indep_var_listbox.pack(side="left", fill="both", expand=True)
        vars_frame.columnconfigure(1, weight=1)
        vars_frame.rowconfigure(1, weight=1)

        # 4. Botón de Ejecución
        run_button = ttk.Button(controls_frame, text="Ejecutar Regresión Logística", command=self._run_logistic_regression)
        run_button.pack(pady=10)

        # --- Resultados en Panel Derecho (con Notebook para texto y gráficos) ---
        self.results_notebook = ttk.Notebook(right_pane)
        self.results_notebook.pack(padx=10, pady=10, fill="both", expand=True)

        # Pestaña para Resultados de Texto
        results_text_frame = ttk.Frame(self.results_notebook)
        self.results_notebook.add(results_text_frame, text="Resumen del Modelo")
        self.results_text = tk.Text(results_text_frame, wrap="none", height=25)
        results_v_scroll = ttk.Scrollbar(results_text_frame, orient="vertical", command=self.results_text.yview)
        results_h_scroll = ttk.Scrollbar(results_text_frame, orient="horizontal", command=self.results_text.xview)
        self.results_text.configure(yscrollcommand=results_v_scroll.set, xscrollcommand=results_h_scroll.set)
        results_v_scroll.pack(side="right", fill="y")
        results_h_scroll.pack(side="bottom", fill="x")
        self.results_text.pack(fill="both", expand=True, padx=5, pady=5)
        self.results_text.config(state="disabled")

        # Pestaña para Gráfico ROC (se añadirá dinámicamente)
        self.roc_plot_frame = None # Se creará al generar el gráfico

    def _load_file(self):
        """Carga un archivo CSV o Excel y actualiza los controles."""
        filepath = filedialog.askopenfilename(
            title="Seleccionar archivo de datos",
            filetypes=(("Archivos CSV", "*.csv"), ("Archivos Excel", "*.xls *.xlsx"), ("Todos los archivos", "*.*"))
        )
        if not filepath: return

        ext = os.path.splitext(filepath)[1].lower()
        try:
            if ext == ".csv": self.df_original = pd.read_csv(filepath, sep=",")
            elif ext in [".xls", ".xlsx"]: self.df_original = pd.read_excel(filepath)
            else: messagebox.showerror("Error de Archivo", "Tipo de archivo no soportado."); return

            self.filepath_var.set(filepath)
            self._update_variable_selectors()
            # FilterComponent removido.
            # Actualizar combos de filtros generales
            cols = self.df_original.columns.tolist() if self.df_original is not None else []
            filter_cols_options = [''] + cols
            if hasattr(self, 'filter_col_1_combo'): # Verificar si los widgets ya existen
                self.filter_col_1_combo['values'] = filter_cols_options
                if not self.filter_col_1_var.get() and cols: self.filter_col_1_var.set('')
            if hasattr(self, 'filter_col_2_combo'):
                self.filter_col_2_combo['values'] = filter_cols_options
                if not self.filter_col_2_var.get() and cols: self.filter_col_2_var.set('')
            
            messagebox.showinfo("Archivo Cargado", f"Archivo '{os.path.basename(filepath)}' cargado.")
            self.log(f"Datos cargados: {self.df_original.shape}", "INFO")

        except Exception as e:
            messagebox.showerror("Error al Leer Archivo", f"No se pudo leer el archivo:\n{e}")
            self.df_original = None; self.filepath_var.set(""); self._update_variable_selectors()
            # FilterComponent removido.
            # Limpiar combos de filtros generales en caso de error
            if hasattr(self, 'filter_col_1_combo'): self.filter_col_1_combo['values'] = ['']
            if hasattr(self, 'filter_col_2_combo'): self.filter_col_2_combo['values'] = ['']
            self.log(f"Error cargando archivo: {e}", "ERROR")

    def _update_variable_selectors(self):
        """Actualiza los combobox y listbox de selección de variables."""
        cols = sorted(self.df_original.columns.tolist()) if self.df_original is not None else []
        binary_cols = []
        if self.df_original is not None:
            for col in cols:
                # Considerar binaria si tiene 2 valores únicos y es numérica o booleana
                unique_vals = self.df_original[col].dropna().unique()
                if len(unique_vals) == 2:
                     # Chequear si son 0/1, True/False, o dos números cualesquiera
                     is_numeric_or_bool = pd.api.types.is_numeric_dtype(self.df_original[col]) or pd.api.types.is_bool_dtype(self.df_original[col])
                     # Podríamos ser más estrictos y requerir 0/1
                     # is_01 = all(v in [0, 1] for v in unique_vals)
                     if is_numeric_or_bool:
                         binary_cols.append(col)

        self.dep_var_combo['values'] = [""] + binary_cols
        self.dependent_var.set("")

        self.indep_var_listbox.delete(0, tk.END)
        for col in cols:
            self.indep_var_listbox.insert(tk.END, col)

        self.log("Selectores de variables actualizados.", "DEBUG")

    def _validate_dependent_var(self, event=None):
        """Valida que la variable dependiente seleccionada sea binaria (0/1)."""
        dep_var_name = self.dependent_var.get()
        if not dep_var_name or self.df_original is None:
            return

        unique_vals = self.df_original[dep_var_name].dropna().unique()
        # Ser estricto: debe contener solo 0 y 1
        if not all(v in [0, 1] for v in unique_vals) or len(unique_vals) != 2:
            messagebox.showwarning("Variable Dependiente Inválida",
                                   f"La variable dependiente '{dep_var_name}' debe ser binaria y contener solo los valores 0 y 1.")
            self.dependent_var.set("") # Limpiar selección

    def _run_logistic_regression(self):
        """Ejecuta el análisis de regresión logística."""
        self.results_text.config(state="normal")
        self.results_text.delete("1.0", tk.END)
        self.log("Iniciando análisis de Regresión Logística...", "INFO")

        # 1. FilterComponent ha sido removido. Se trabaja directamente con una copia de self.df_original.
        if self.df_original is not None:
            df_initial = self.df_original.copy()
            self.log("Usando DataFrame original para regresión logística (antes de filtros generales).", "INFO")

            # Aplicar filtros generales definidos en la UI
            self.df_filtered = self._apply_general_filters(df_initial)

            if self.df_filtered is None or self.df_filtered.empty:
                self.log("DataFrame vacío después de aplicar filtros generales.", "WARN")
                self.results_text.insert(tk.END, "DataFrame vacío después de aplicar filtros generales.")
                self.results_text.config(state="disabled")
                return
        else:
            self.log("No hay datos cargados.", "ERROR")
            self.results_text.insert(tk.END, "No hay datos cargados.")
            self.results_text.config(state="disabled")
            return

        # 2. Obtener variables seleccionadas
        dep_var = self.dependent_var.get()
        selected_indices = self.indep_var_listbox.curselection()
        indep_vars = [self.indep_var_listbox.get(i) for i in selected_indices]

        if not dep_var:
            self.log("Variable dependiente no seleccionada.", "ERROR")
            messagebox.showerror("Error", "Seleccione la variable dependiente.")
            self.results_text.config(state="disabled")
            return
        if not indep_vars:
            self.log("Variables independientes no seleccionadas.", "ERROR")
            messagebox.showerror("Error", "Seleccione al menos una variable independiente.")
            self.results_text.config(state="disabled")
            return
        if dep_var in indep_vars:
            self.log("Variable dependiente no puede ser independiente.", "ERROR")
            messagebox.showerror("Error", "La variable dependiente no puede estar en la lista de independientes.")
            self.results_text.config(state="disabled")
            return

        # 3. Preparar datos para statsmodels (manejar NaNs)
        cols_to_use = [dep_var] + indep_vars
        df_analysis = self.df_filtered[cols_to_use].dropna()

        if df_analysis.empty or len(df_analysis) < len(cols_to_use) + 1:
             self.log(f"Datos insuficientes después de eliminar NaNs (Filas: {len(df_analysis)}, Vars: {len(cols_to_use)}).", "ERROR")
             self.results_text.insert(tk.END, f"Datos insuficientes después de eliminar NaNs (Filas: {len(df_analysis)}). Se necesitan más filas que variables.")
             self.results_text.config(state="disabled")
             return

        # Validar nuevamente la variable dependiente en los datos filtrados/limpios
        unique_deps = df_analysis[dep_var].unique()
        if not all(v in [0, 1] for v in unique_deps) or len(unique_deps) != 2:
             self.log(f"Variable dependiente '{dep_var}' no es binaria (0/1) en los datos de análisis.", "ERROR")
             self.results_text.insert(tk.END, f"Error: La variable dependiente '{dep_var}' no es binaria (0/1) en los datos usados para el análisis.")
             self.results_text.config(state="disabled")
             return

        self.log(f"Variables para modelo: Dep={dep_var}, Indep={indep_vars}", "INFO")
        self.log(f"Dimensiones datos análisis: {df_analysis.shape}", "INFO")

        # 4. Construir fórmula y ajustar modelo
        try:
            # Crear fórmula para statsmodels (maneja variables categóricas automáticamente con C())
            # Asegurarse de que los nombres de variables sean válidos para fórmulas
            clean_indep_vars = [f"`{v}`" if not v.isidentifier() else v for v in indep_vars]
            formula = f"`{dep_var}` ~ {' + '.join(clean_indep_vars)}"
            self.log(f"Fórmula: {formula}", "DEBUG")

            # Usar Logit para regresión logística binaria
            logit_model = smf.logit(formula, data=df_analysis)
            self.model_results = logit_model.fit()

            # 5. Mostrar resultados básicos
            summary_str = str(self.model_results.summary())
            self.results_text.insert(tk.END, summary_str)
            self.log("Modelo ajustado. Mostrando resumen básico.", "SUCCESS")

            # Calcular y mostrar Odds Ratios
            try:
                odds_ratios = pd.DataFrame({
                    "Odds Ratio": np.exp(self.model_results.params),
                    "IC 95% Inferior": np.exp(self.model_results.conf_int()[0]),
                    "IC 95% Superior": np.exp(self.model_results.conf_int()[1]),
                    "p-valor": self.model_results.pvalues
                })
                # Excluir intercepto si se desea
                # odds_ratios = odds_ratios.drop(index='Intercept')
                self.results_text.insert(tk.END, "\n\n--- Odds Ratios (exp(coef)) ---\n")
                self.results_text.insert(tk.END, odds_ratios.to_string(float_format="%.4f"))
                self.log("Odds Ratios calculados.", "INFO")
            except Exception as e_or:
                self.log(f"Error calculando Odds Ratios: {e_or}", "WARN")
                self.results_text.insert(tk.END, f"\n\nError calculando Odds Ratios: {e_or}")

            # --- Métricas de Evaluación Adicionales ---
            self.results_text.insert(tk.END, f"\n\n--- Métricas de Evaluación del Modelo ---\n")
            y_true = df_analysis[dep_var]
            y_pred_prob = self.model_results.predict(df_analysis[indep_vars]) # Probabilidades predichas

            # Pseudo R-cuadrado (McFadden)
            try:
                ll_full = self.model_results.llf
                ll_null = smf.logit(f"`{dep_var}` ~ 1", data=df_analysis).fit(disp=0).llf
                pseudo_r2_mcfadden = 1 - (ll_full / ll_null)
                self.results_text.insert(tk.END, f"Pseudo R-cuadrado (McFadden): {pseudo_r2_mcfadden:.4f}\n")
                self.log(f"Pseudo R2 (McFadden): {pseudo_r2_mcfadden:.4f}", "INFO")
            except Exception as e_r2:
                self.log(f"Error calculando Pseudo R2: {e_r2}", "WARN")
                self.results_text.insert(tk.END, f"Pseudo R-cuadrado (McFadden): Error ({e_r2})\n")

            # Curva ROC y AUC
            try:
                fpr, tpr, thresholds_roc = roc_curve(y_true, y_pred_prob)
                roc_auc = auc(fpr, tpr)
                self.results_text.insert(tk.END, f"AUC (Area Under ROC Curve): {roc_auc:.4f}\n")
                self.log(f"AUC: {roc_auc:.4f}", "INFO")
                self._plot_roc_curve(fpr, tpr, roc_auc) # Mostrar gráfico ROC
            except Exception as e_roc:
                self.log(f"Error calculando/plotando ROC/AUC: {e_roc}", "WARN")
                self.results_text.insert(tk.END, f"AUC: Error ({e_roc})\n")

            # Tabla de Clasificación y Métricas Relacionadas (usando umbral 0.5)
            try:
                threshold = 0.5
                y_pred_class = (y_pred_prob >= threshold).astype(int)
                cm = confusion_matrix(y_true, y_pred_class)
                self.results_text.insert(tk.END, f"\n--- Tabla de Clasificación (umbral={threshold}) ---\n")
                self.results_text.insert(tk.END, f"Verdaderos Negativos (VN): {cm[0,0]}\n")
                self.results_text.insert(tk.END, f"Falsos Positivos   (FP): {cm[0,1]}\n")
                self.results_text.insert(tk.END, f"Falsos Negativos   (FN): {cm[1,0]}\n")
                self.results_text.insert(tk.END, f"Verdaderos Positivos (VP): {cm[1,1]}\n\n")

                report = classification_report(y_true, y_pred_class, target_names=['Clase 0', 'Clase 1'], zero_division=0)
                self.results_text.insert(tk.END, "Reporte de Clasificación:\n")
                self.results_text.insert(tk.END, report + "\n")
                self.log("Tabla de clasificación y reporte generados.", "INFO")
            except Exception as e_cm:
                self.log(f"Error generando tabla de clasificación: {e_cm}", "WARN")
                self.results_text.insert(tk.END, f"Tabla de Clasificación: Error ({e_cm})\n")

            # Prueba de Hosmer-Lemeshow (Cálculo manual simplificado)
            try:
                hl_stat, hl_p_value = self._hosmer_lemeshow_test(y_true, y_pred_prob)
                self.results_text.insert(tk.END, f"\n--- Prueba de Hosmer-Lemeshow ---\n")
                self.results_text.insert(tk.END, f"Estadístico H-L: {hl_stat:.4f}\n")
                self.results_text.insert(tk.END, f"p-valor H-L: {hl_p_value:.4f}\n")
                interpretation = "Buen ajuste (p > 0.05)" if hl_p_value > 0.05 else "Pobre ajuste (p <= 0.05)"
                self.results_text.insert(tk.END, f"Interpretación: {interpretation}\n")
                self.log(f"Hosmer-Lemeshow: stat={hl_stat:.4f}, p={hl_p_value:.4f}", "INFO")
            except Exception as e_hl:
                self.log(f"Error calculando Hosmer-Lemeshow: {e_hl}", "WARN")
                self.results_text.insert(tk.END, f"Prueba de Hosmer-Lemeshow: Error ({e_hl})\n")

        except Exception as e:
            self.log(f"Error al ajustar el modelo logístico: {e}", "ERROR")
            self.results_text.insert(tk.END, f"Error al ajustar el modelo:\n{e}\n{traceback.format_exc()}")
            messagebox.showerror("Error de Modelo", f"Error al ajustar el modelo logístico:\n{e}")

        finally:
            self.results_text.config(state="disabled")

    def log(self, message, level="INFO"):
        """Placeholder para logging."""
        print(f"[{level}] LogisticRegTab: {message}")

    def _hosmer_lemeshow_test(self, y_true, y_pred_prob, g=10):
        """
        Calcula la prueba de Hosmer-Lemeshow.
        :param y_true: Serie de valores verdaderos (0 o 1).
        :param y_pred_prob: Serie de probabilidades predichas.
        :param g: Número de grupos (deciles por defecto).
        :return: (estadístico chi-cuadrado, p-valor)
        """
        y_true = np.array(y_true)
        y_pred_prob = np.array(y_pred_prob)

        # Ordenar por probabilidad predicha
        sorted_indices = np.argsort(y_pred_prob)
        y_true_sorted = y_true[sorted_indices]
        y_pred_prob_sorted = y_pred_prob[sorted_indices]

        # Crear grupos
        groups = np.array_split(np.arange(len(y_true_sorted)), g)
        
        observed_1 = []
        expected_1 = []
        observed_0 = []
        expected_0 = []
        group_size = []

        for group_indices in groups:
            if len(group_indices) == 0:
                continue
            
            obs_1_g = np.sum(y_true_sorted[group_indices])
            exp_1_g = np.sum(y_pred_prob_sorted[group_indices])
            
            obs_0_g = len(group_indices) - obs_1_g
            exp_0_g = len(group_indices) - exp_1_g
            
            # Evitar división por cero si un grupo esperado es 0 (aunque raro con probs)
            if exp_1_g == 0 and obs_1_g != 0: exp_1_g = 1e-9 # Pequeño valor para evitar error
            if exp_0_g == 0 and obs_0_g != 0: exp_0_g = 1e-9

            observed_1.append(obs_1_g)
            expected_1.append(exp_1_g)
            observed_0.append(obs_0_g)
            expected_0.append(exp_0_g)
            group_size.append(len(group_indices))

        # Calcular estadístico Chi-cuadrado
        hl_stat = 0
        for i in range(len(observed_1)):
            if expected_1[i] > 0: # Evitar división por cero
                 hl_stat += ((observed_1[i] - expected_1[i])**2) / expected_1[i]
            if expected_0[i] > 0: # Evitar división por cero
                 hl_stat += ((observed_0[i] - expected_0[i])**2) / expected_0[i]
        
        # Grados de libertad (g - 2 es común, pero puede variar)
        # Para una prueba más robusta, se podría usar g - k donde k es el número de
        # patrones de covarianza distintos si se agrupa por ellos, o g-2 si se agrupa por deciles de riesgo.
        # Aquí usamos g-2 como aproximación común para deciles.
        df_hl = max(1, g - 2)
        p_value = 1 - sm.stats.chisqprob(hl_stat, df_hl) # statsmodels.stats.chisqprob es scipy.stats.chi2.sf

        return hl_stat, p_value

    def _plot_roc_curve(self, fpr, tpr, roc_auc):
        """Genera y muestra el gráfico de la curva ROC en una nueva pestaña del Notebook."""
        if self.roc_plot_frame:
            for widget in self.roc_plot_frame.winfo_children():
                widget.destroy()
        else:
            self.roc_plot_frame = ttk.Frame(self.results_notebook)
            self.results_notebook.add(self.roc_plot_frame, text="Curva ROC")

        fig, ax = plt.subplots(figsize=(6, 5)) # Ajustar tamaño según sea necesario
        ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'Curva ROC (AUC = {roc_auc:.2f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Tasa de Falsos Positivos (1 - Especificidad)')
        ax.set_ylabel('Tasa de Verdaderos Positivos (Sensibilidad)')
        ax.set_title('Curva ROC')
        ax.legend(loc="lower right")
        ax.grid(True, linestyle=':', alpha=0.7)
        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=self.roc_plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        # Asegurarse de que la pestaña ROC sea visible
        try:
            self.results_notebook.select(self.roc_plot_frame)
        except tk.TclError: # Puede ocurrir si la pestaña ya está seleccionada o el notebook no está visible
            pass
        self.log("Curva ROC generada.", "INFO")


# --- Ejemplo de uso ---
if __name__ == '__main__':
    root = tk.Tk()
    root.title("Test Logistic Regression Tab")
    root.geometry("900x600")

    # Crear datos de ejemplo para logística
    np.random.seed(42)
    n_samples = 200
    X1 = np.random.rand(n_samples) * 10
    X2 = np.random.choice(['G1', 'G2', 'G3'], n_samples)
    X3_cont = np.random.randn(n_samples) * 5 + 20
    # Crear log-odds lineal
    log_odds = -2 + 0.5 * X1 + (X2 == 'G2') * 1.0 + (X2 == 'G3') * 1.5 - 0.1 * X3_cont
    # Convertir a probabilidad
    prob = 1 / (1 + np.exp(-log_odds))
    # Generar variable dependiente binaria
    Y = (np.random.rand(n_samples) < prob).astype(int)

    sample_df_log = pd.DataFrame({
        'Target': Y,
        'PredictorContinuo': X1,
        'PredictorCategorico': X2,
        'OtraNumerica': X3_cont,
        'ID_Sujeto': range(n_samples)
    })
    # Introducir algunos NaNs
    sample_df_log.loc[::15, 'PredictorContinuo'] = np.nan
    sample_df_log.loc[::20, 'PredictorCategorico'] = np.nan

    # Frame principal
    main_frame = ttk.Frame(root, padding="10")
    main_frame.pack(fill="both", expand=True)

    # Instanciar la pestaña
    logreg_tab = LogisticRegressionTab(main_frame)
    logreg_tab.pack(fill="both", expand=True)

    # Cargar datos de ejemplo en la pestaña (simulando clic de botón)
    logreg_tab.df_original = sample_df_log
    logreg_tab.filepath_var.set("sample_logistic_data.df") # Simular carga
    logreg_tab._update_variable_selectors()
    if hasattr(logreg_tab, 'filter_component') and logreg_tab.filter_component:
        logreg_tab.filter_component.set_dataframe(sample_df_log)

    root.mainloop()