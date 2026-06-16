#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import pandas as pd
import numpy as np
import matplotlib
# matplotlib.use("TkAgg") # Moved to main app
import matplotlib.pyplot as plt

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.stats import (
    shapiro, kstest, ttest_ind, mannwhitneyu,
    f_oneway, kruskal, probplot, chi2_contingency, fisher_exact
)
import traceback # Añadido para logging

def create_scrollable_frame(container):
    """
    Crea un frame con scroll vertical para ubicar muchos controles.
    """
    canvas = tk.Canvas(container)
    scrollbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
    scrollable_frame = ttk.Frame(canvas)
    scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)
    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")
    return scrollable_frame

def check_normality(data, mode):
    """
    Aplica la prueba de normalidad según 'mode'.
    """
    n = len(data)
    if n < 3:
        return ("Sin datos", 0, 1, False)
    if mode == "Automático":
        if n < 50:
            test_name = "Shapiro-Wilk"
            stat, p_val = shapiro(data)
        else:
            test_name = "Kolmogorov-Smirnov"
            data_std = (data - np.mean(data)) / np.std(data)
            stat, p_val = kstest(data_std, 'norm')
    elif mode == "Shapiro-Wilk":
        test_name = "Shapiro-Wilk"
        stat, p_val = shapiro(data)
    else:
        test_name = "Kolmogorov-Smirnov"
        data_std = (data - np.mean(data)) / np.std(data)
        stat, p_val = kstest(data_std, 'norm')
    is_normal = (p_val > 0.05)
    return (test_name, stat, p_val, is_normal)

class GraficaQQ(ttk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.data = None
        self.remove_blanks_var1 = tk.BooleanVar(value=True)
        self.remove_blanks_var2 = tk.BooleanVar(value=True)
        self.remove_non_numeric_var2 = tk.BooleanVar(value=True)

        # Variables para el filtro general
        self.general_filter_var = tk.StringVar()
        self.general_filter_op = tk.StringVar()
        self.general_filter_val = tk.StringVar()
        self.general_filter_active = tk.BooleanVar(value=False)

        self.create_widgets()

    def create_widgets(self):
        main_pane = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        main_pane.pack(fill=tk.BOTH, expand=True)

        left_frame = ttk.Frame(main_pane)
        main_pane.add(left_frame, weight=0)
        frm_result = ttk.LabelFrame(left_frame, text="Resultados")
        frm_result.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.txt_output = tk.Text(frm_result, height=25, wrap="none")
        self.txt_output.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        btn_clear = ttk.Button(left_frame, text="Borrar", command=self.clear_output)
        btn_clear.pack(padx=5, pady=5, anchor="e")

        right_frame = ttk.Frame(main_pane)
        main_pane.add(right_frame, weight=1)
        vertical_pane = ttk.PanedWindow(right_frame, orient=tk.VERTICAL)
        vertical_pane.pack(fill=tk.BOTH, expand=True)

        scroll_container = ttk.Frame(vertical_pane)
        vertical_pane.add(scroll_container, weight=0)
        self.scrollable_options = create_scrollable_frame(scroll_container)

        figure_frame = ttk.Frame(vertical_pane)
        vertical_pane.add(figure_frame, weight=1)
        self.figure = plt.Figure(figsize=(6, 4), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, master=figure_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # --- CONTROLES DE OPCIONES ---
        notebook = ttk.Notebook(self.scrollable_options)
        notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        tab_main = ttk.Frame(notebook)
        tab_filters = ttk.Frame(notebook)
        tab_analysis_opts = ttk.Frame(notebook)
        tab_output_opts = ttk.Frame(notebook)

        notebook.add(tab_main, text="Principal")
        notebook.add(tab_filters, text="Filtros")
        notebook.add(tab_analysis_opts, text="Opciones de Análisis")
        notebook.add(tab_output_opts, text="Opciones de Salida")

        # == Tab Principal ==
        frm_load = ttk.LabelFrame(tab_main, text="Cargar Datos (Excel/CSV)")
        frm_load.pack(fill=tk.X, padx=5, pady=5)
        btn_load = ttk.Button(frm_load, text="Cargar Archivo", command=self.load_data)
        btn_load.pack(side=tk.LEFT, padx=5, pady=5)

        frm_analysis = ttk.LabelFrame(tab_main, text="Tipo de Análisis")
        frm_analysis.pack(fill=tk.X, padx=5, pady=5)
        self.analysis_type = tk.StringVar(value="Continuo (Q-Q)")
        rb_cont = ttk.Radiobutton(frm_analysis, text="Continuo (Q-Q)", variable=self.analysis_type, value="Continuo (Q-Q)")
        rb_cat = ttk.Radiobutton(frm_analysis, text="Categórico (Chi-cuadrado)", variable=self.analysis_type, value="Categórico (Chi-cuadrado)")
        rb_cont.pack(side=tk.LEFT, padx=5, pady=5)
        rb_cat.pack(side=tk.LEFT, padx=5, pady=5)

        frm_vars = ttk.LabelFrame(tab_main, text="Selección de Variables")
        frm_vars.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(frm_vars, text="Variable 1 (Agrupadora):").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.cmb_var1 = ttk.Combobox(frm_vars, values=[], state="readonly")
        self.cmb_var1.grid(row=0, column=1, padx=5, pady=5, sticky="we")
        ttk.Label(frm_vars, text="Filtro/Etiquetas para Var 1:").grid(row=0, column=2, padx=5, pady=5, sticky="e")
        self.entry_var1_filter = ttk.Entry(frm_vars, width=40)
        self.entry_var1_filter.grid(row=0, column=3, padx=5, pady=5, sticky="we")

        ttk.Label(frm_vars, text="Variables 2 (a Analizar):").grid(row=1, column=0, padx=5, pady=5, sticky="nw")
        self.var2_listbox_frame = ttk.Frame(frm_vars)
        self.var2_listbox_frame.grid(row=1, column=1, padx=5, pady=5, sticky="we")
        self.var2_listbox = tk.Listbox(self.var2_listbox_frame, selectmode=tk.EXTENDED, exportselection=False, height=5)
        var2_scrollbar = ttk.Scrollbar(self.var2_listbox_frame, orient="vertical", command=self.var2_listbox.yview)
        self.var2_listbox.configure(yscrollcommand=var2_scrollbar.set)
        self.var2_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        var2_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        ttk.Label(frm_vars, text="Filtro/Etiquetas para Var 2:").grid(row=1, column=2, padx=5, pady=5, sticky="ne")
        self.entry_var2_filter = ttk.Entry(frm_vars, width=40)
        self.entry_var2_filter.grid(row=1, column=3, padx=5, pady=5, sticky="we")
        
        frm_vars.columnconfigure(1, weight=1)
        frm_vars.columnconfigure(3, weight=1)

        frm_run = ttk.Frame(tab_main)
        frm_run.pack(fill=tk.X, padx=5, pady=10)
        btn_run = ttk.Button(frm_run, text="Mostrar Análisis", command=self.generate_analysis)
        btn_run.pack(side=tk.LEFT, padx=10)
        btn_save = ttk.Button(frm_run, text="Guardar Gráfico", command=self.save_qq_plot)
        btn_save.pack(side=tk.LEFT, padx=10)

        # == Tab Filtros ==
        frm_general_filter = ttk.LabelFrame(tab_filters, text="Filtro General Adicional")
        frm_general_filter.pack(fill=tk.X, padx=5, pady=5)

        ttk.Checkbutton(frm_general_filter, text="Activar Filtro", variable=self.general_filter_active).grid(row=0, column=0, columnspan=2, padx=5, pady=2, sticky="w")
        
        ttk.Label(frm_general_filter, text="Variable:").grid(row=1, column=0, padx=5, pady=2, sticky="w")
        self.cmb_general_filter_var = ttk.Combobox(frm_general_filter, textvariable=self.general_filter_var, state="readonly")
        self.cmb_general_filter_var.grid(row=1, column=1, padx=5, pady=2, sticky="we")

        ttk.Label(frm_general_filter, text="Operador:").grid(row=2, column=0, padx=5, pady=2, sticky="w")
        self.cmb_general_filter_op = ttk.Combobox(frm_general_filter, textvariable=self.general_filter_op, values=["==", "!=", ">", "<", ">=", "<=", "contiene", "no contiene", "es NaN", "no es NaN"], state="readonly")
        self.cmb_general_filter_op.grid(row=2, column=1, padx=5, pady=2, sticky="we")
        self.cmb_general_filter_op.current(0)

        ttk.Label(frm_general_filter, text="Valor:").grid(row=3, column=0, padx=5, pady=2, sticky="w")
        self.entry_general_filter_val = ttk.Entry(frm_general_filter, textvariable=self.general_filter_val)
        self.entry_general_filter_val.grid(row=3, column=1, padx=5, pady=2, sticky="we")
        
        frm_general_filter.columnconfigure(1, weight=1)

        # == Tab Opciones de Salida ==
        frm_output_opts = ttk.LabelFrame(tab_output_opts, text="Opciones de Salida")
        frm_output_opts.pack(fill=tk.X, padx=5, pady=5)
        
        self.output_mode = tk.StringVar(value="Analítico")
        ttk.Radiobutton(frm_output_opts, text="Analítico", variable=self.output_mode, value="Analítico").pack(side=tk.LEFT, padx=5, pady=2)
        ttk.Radiobutton(frm_output_opts, text="Resumen", variable=self.output_mode, value="Resumen").pack(side=tk.LEFT, padx=5, pady=2)

        self.show_qq_popup = tk.BooleanVar(value=False)
        ttk.Checkbutton(frm_output_opts, text="Mostrar Q-Q plots por categoría (en ventana nueva)", variable=self.show_qq_popup).pack(side=tk.LEFT, padx=20, pady=2)

        frm_clean = ttk.LabelFrame(tab_output_opts, text="Opciones de Limpieza")
        frm_clean.pack(fill=tk.X, padx=5, pady=5)
        ttk.Checkbutton(frm_clean, text="Eliminar blancos en Variable 1", variable=self.remove_blanks_var1).pack(anchor=tk.W, padx=5, pady=2)
        ttk.Checkbutton(frm_clean, text="Eliminar blancos en Variable 2", variable=self.remove_blanks_var2).pack(anchor=tk.W, padx=5, pady=2)
        ttk.Checkbutton(frm_clean, text="Eliminar no numéricos en Variable 2", variable=self.remove_non_numeric_var2).pack(anchor=tk.W, padx=5, pady=2)

        # == Tab Opciones de Análisis ==
        frm_normal = ttk.LabelFrame(tab_analysis_opts, text="Prueba de Normalidad (Análisis Continuo)")
        frm_normal.pack(fill=tk.X, padx=5, pady=5)
        self.norm_mode = tk.StringVar(value="Automático")
        ttk.Radiobutton(frm_normal, text="Automático", variable=self.norm_mode, value="Automático").pack(side=tk.LEFT, padx=5, pady=2)
        ttk.Radiobutton(frm_normal, text="Shapiro-Wilk", variable=self.norm_mode, value="Shapiro-Wilk").pack(side=tk.LEFT, padx=5, pady=2)
        ttk.Radiobutton(frm_normal, text="Kolmogorov-Smirnov", variable=self.norm_mode, value="Kolmogorov-Smirnov").pack(side=tk.LEFT, padx=5, pady=2)

        frm_comp = ttk.LabelFrame(tab_analysis_opts, text="Prueba de Diferencias (Análisis Continuo)")
        frm_comp.pack(fill=tk.X, padx=5, pady=5)
        self.comp_mode = tk.StringVar(value="Automático")
        ttk.Radiobutton(frm_comp, text="Automático", variable=self.comp_mode, value="Automático").pack(side=tk.LEFT, padx=5, pady=2)
        ttk.Radiobutton(frm_comp, text="Paramétrico", variable=self.comp_mode, value="Paramétrico").pack(side=tk.LEFT, padx=5, pady=2)
        ttk.Radiobutton(frm_comp, text="No paramétrico", variable=self.comp_mode, value="No paramétrico").pack(side=tk.LEFT, padx=5, pady=2)

        frm_cat_opts = ttk.LabelFrame(tab_analysis_opts, text="Opciones Análisis Categórico")
        frm_cat_opts.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(frm_cat_opts, text="Test Categórico:").grid(row=0, column=0, padx=5, pady=2, sticky="w")
        self.cat_test_mode = tk.StringVar(value="Automático")
        ttk.Radiobutton(frm_cat_opts, text="Automático", variable=self.cat_test_mode, value="Automático").grid(row=0, column=1, padx=5, pady=2, sticky="w")
        ttk.Radiobutton(frm_cat_opts, text="Chi-cuadrado", variable=self.cat_test_mode, value="Chi-cuadrado").grid(row=0, column=2, padx=5, pady=2, sticky="w")
        ttk.Radiobutton(frm_cat_opts, text="Fisher Exact", variable=self.cat_test_mode, value="Fisher Exact").grid(row=0, column=3, padx=5, pady=2, sticky="w")
        ttk.Label(frm_cat_opts, text="Formato Salida:").grid(row=1, column=0, padx=5, pady=2, sticky="w")
        self.cat_format = tk.StringVar(value="Números")
        ttk.Radiobutton(frm_cat_opts, text="Números", variable=self.cat_format, value="Números").grid(row=1, column=1, padx=5, pady=2, sticky="w")
        ttk.Radiobutton(frm_cat_opts, text="Porcentajes", variable=self.cat_format, value="Porcentajes").grid(row=1, column=2, padx=5, pady=2, sticky="w")

    def clear_output(self):
        self.txt_output.delete("1.0", tk.END)

    def load_data(self):
        file_path = filedialog.askopenfilename(
            title="Selecciona archivo Excel o CSV",
            filetypes=[("Excel files", "*.xlsx *.xls"), ("CSV files", "*.csv"), ("Todos los archivos", "*.*")]
        )
        if not file_path:
            return
        try:
            if file_path.lower().endswith(".csv"):
                self.data = pd.read_csv(file_path)
            else:
                self.data = pd.read_excel(file_path)
            messagebox.showinfo("Éxito", f"Datos cargados correctamente.\nFilas: {self.data.shape[0]}, Columnas: {self.data.shape[1]}")
            cols = list(self.data.columns)
            self.cmb_var1['values'] = cols
            self.cmb_general_filter_var['values'] = cols
            
            self.var2_listbox.delete(0, tk.END)
            for col in cols:
                self.var2_listbox.insert(tk.END, col)

            if cols:
                self.cmb_var1.current(0)
                self.cmb_general_filter_var.current(0)
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo cargar el archivo:\n{e}")

    def _apply_specific_filter(self, series, filter_string):
        if not filter_string.strip():
            return series
        s_str = series.astype(str)
        if ":" in filter_string:
            mapping = {}
            keys_to_keep = []
            parts = [p.strip() for p in filter_string.split(",") if p.strip()]
            for part in parts:
                if ":" in part:
                    key, label = part.split(":", 1)
                    mapping[key.strip()] = label.strip()
                    keys_to_keep.append(key.strip())
            filtered = series[s_str.isin(keys_to_keep)]
            return filtered.map(mapping)
        elif "-" in filter_string and "," not in filter_string:
            parts = filter_string.split("-")
            if len(parts) == 2:
                try:
                    low = float(parts[0].strip()); high = float(parts[1].strip())
                    s_num = pd.to_numeric(series, errors="coerce")
                    return series[(s_num >= low) & (s_num <= high)]
                except ValueError: return series
        elif "," in filter_string:
            vals = [p.strip() for p in filter_string.split(",") if p.strip()]
            if pd.api.types.is_numeric_dtype(series):
                num_vals = []
                for v_str in vals:
                    try: num_vals.append(float(v_str))
                    except ValueError: num_vals.append(v_str)
                return series[series.isin(num_vals)]
            else:
                return series[s_str.isin(vals)]
        else:
            try:
                val_num = float(filter_string)
                if pd.api.types.is_numeric_dtype(series):
                    return series[pd.to_numeric(series, errors='coerce') == val_num]
                else:
                    return series[s_str == filter_string]
            except ValueError:
                return series[s_str == filter_string]
        return series

    def _apply_general_filters(self, df):
        df_filtered = df.copy()
        
        if self.general_filter_active.get():
            col_name = self.general_filter_var.get()
            op = self.general_filter_op.get()
            val_str = self.general_filter_val.get()

            if not col_name or col_name not in df_filtered.columns:
                return df_filtered

            try:
                col_series = df_filtered[col_name]
                
                if op == "es NaN":
                    df_filtered = df_filtered[col_series.isna()]
                elif op == "no es NaN":
                    df_filtered = df_filtered[col_series.notna()]
                else:
                    if val_str == "":
                         return df_filtered

                    if pd.api.types.is_numeric_dtype(col_series) and op in ["==", "!=", ">", "<", ">=", "<="]:
                        try:
                            val_num = float(val_str)
                            if op == "==": df_filtered = df_filtered[col_series == val_num]
                            elif op == "!=": df_filtered = df_filtered[col_series != val_num]
                            elif op == ">": df_filtered = df_filtered[col_series > val_num]
                            elif op == "<": df_filtered = df_filtered[col_series < val_num]
                            elif op == ">=": df_filtered = df_filtered[col_series >= val_num]
                            elif op == "<=": df_filtered = df_filtered[col_series <= val_num]
                        except ValueError:
                            pass # No se pudo convertir a número, no se filtra
                    elif pd.api.types.is_string_dtype(col_series) or col_series.dtype == 'object':
                        col_series_str = col_series.astype(str)
                        if op == "==": df_filtered = df_filtered[col_series_str == val_str]
                        elif op == "!=": df_filtered = df_filtered[col_series_str != val_str]
                        elif op == "contiene": df_filtered = df_filtered[col_series_str.str.contains(val_str, case=False, na=False)]
                        elif op == "no contiene": df_filtered = df_filtered[~col_series_str.str.contains(val_str, case=False, na=False)]
                        else:
                            if op == ">": df_filtered = df_filtered[col_series_str > val_str]
                            elif op == "<": df_filtered = df_filtered[col_series_str < val_str]
                            elif op == ">=": df_filtered = df_filtered[col_series_str >= val_str]
                            elif op == "<=": df_filtered = df_filtered[col_series_str <= val_str]
            except Exception as e:
                self.log(f"Error aplicando filtro general: {e}", "ERROR")
        return df_filtered

    def get_data_for_variable_pair(self, df, var1_name, var2_name, analysis_mode=None):
        df_filtered = df.copy()

        if df_filtered is None or df_filtered.empty or var1_name not in df_filtered.columns or var2_name not in df_filtered.columns:
            return None, None

        s_var1 = df_filtered[var1_name].copy()
        s_var2 = df_filtered[var2_name].copy()

        filter_var1_str = self.entry_var1_filter.get().strip()
        if filter_var1_str:
            s_var1 = self._apply_specific_filter(s_var1, filter_var1_str)

        filter_var2_str = self.entry_var2_filter.get().strip()
        if filter_var2_str:
            s_var2 = self._apply_specific_filter(s_var2, filter_var2_str)

        pair_df = pd.DataFrame({var1_name: s_var1, var2_name: s_var2})

        if self.remove_blanks_var1.get():
            pair_df[var1_name] = pair_df[var1_name].replace(r'^\s*$', np.nan, regex=True)
            pair_df.dropna(subset=[var1_name], inplace=True)
        
        if self.remove_blanks_var2.get():
            pair_df[var2_name] = pair_df[var2_name].replace(r'^\s*$', np.nan, regex=True)
            pair_df.dropna(subset=[var2_name], inplace=True)

        mode_to_use = analysis_mode or self.analysis_type.get()
        if mode_to_use == "Continuo (Q-Q)" and self.remove_non_numeric_var2.get():
            pair_df[var2_name] = pd.to_numeric(pair_df[var2_name], errors='coerce')
            pair_df.dropna(subset=[var2_name], inplace=True)

        return pair_df[var1_name], pair_df[var2_name]

    def _perform_categorical_analysis(self, var1_series, var2_series, var1_name, var2_name):
        results = []
        summary_results = []
        cat1 = var1_series.astype(str)
        cat2 = var2_series.astype(str)
        df_cat = pd.DataFrame({var1_name: cat1, var2_name: cat2}).dropna()

        if df_cat.empty:
            return [f"No quedan datos para el par ({var1_name}, {var2_name}) tras filtros.\n"], []

        contingency = pd.crosstab(df_cat[var1_name], df_cat[var2_name])
        chi2 = p = dof = None
        expected = None
        chi2_error_msg = None
        try:
            chi2, p, dof, expected = chi2_contingency(contingency)
        except ValueError as err:
            chi2_error_msg = str(err)

        test_type = self.cat_test_mode.get()
        fisher_justification = ""
        is_2x2 = contingency.shape == (2, 2)

        if test_type == "Automático":
            if is_2x2:
                if expected is not None and (expected < 5).any():
                    test_type = "Fisher Exact"
                    fisher_justification = "(Automático: tabla 2x2 con frec. esperada < 5)"
                elif expected is None:
                    test_type = "Fisher Exact"
                    fisher_justification = "(Automático: tabla 2x2 con frec. esperadas inválidas)"
                else:
                    test_type = "Chi-cuadrado"
                    fisher_justification = "(Automático: tabla 2x2 sin frec. esperada < 5)"
            else:
                test_type = "Chi-cuadrado"
                fisher_justification = "(Automático: tabla no es 2x2)"

        fisher_error_msg = None
        if test_type == "Fisher Exact":
            if is_2x2:
                try:
                    _, p = fisher_exact(contingency)
                    test_used = f"Fisher Exact {fisher_justification}".strip()
                except Exception as fisher_exc:
                    fisher_error_msg = str(fisher_exc)
                    test_used = "Fisher Exact (no disponible)"
                    p = None
            else:
                results.append("ADVERTENCIA: Test de Fisher solo aplica a tablas 2x2. Se usará Chi-cuadrado en su lugar.\n")
                test_type = "Chi-cuadrado"

        if test_type == "Chi-cuadrado":
            if expected is not None:
                test_used = f"Chi-cuadrado {fisher_justification}".strip()
            else:
                test_used = "Chi-cuadrado (no disponible)"
                p = None

        output_format = self.cat_format.get()
        table_str = contingency.to_string()
        if output_format == "Porcentajes":
            contingency_pct = contingency.div(contingency.sum().sum()) * 100
            table_str += "\n\nPorcentajes (%):\n" + contingency_pct.to_string(float_format="%.2f")

        results.append(f"--- Análisis Categórico: {var1_name} vs {var2_name} ---\n")
        results.append("Tabla de Contingencia:\n")
        results.append(table_str + "\n")
        results.append(f"\nTest usado: {test_used}\n")

        if p is not None:
            if "Chi-cuadrado" in test_used:
                results.append(f"Estadístico Chi-cuadrado = {chi2:.4f}, p = {p:.4f}, dof = {dof}\n")
            else:
                results.append(f"p-valor (Fisher) = {p:.4f}\n")
        else:
            if chi2_error_msg:
                results.append(f"No se pudo calcular el test de Chi-cuadrado: {chi2_error_msg}\n")
            elif fisher_error_msg:
                results.append(f"No se pudo calcular el test de Fisher: {fisher_error_msg}\n")
            else:
                results.append("No se pudo calcular el p-valor para la tabla de contingencia.\n")

        conclusion = f"Conclusión para [{var1_name} vs {var2_name}]: "
        if p is None:
            conclusion += "No se pudo evaluar la significancia (p no disponible).\n"
        elif p < 0.05:
            conclusion += "Existe asociación significativa entre las variables (p < 0.05).\n"
        else:
            conclusion += "No se encontró asociación significativa (p >= 0.05).\n"
        results.append(conclusion)
        summary_results.append(conclusion)

        return results, summary_results

    def _perform_continuous_analysis(self, var1_series, var2_series, var1_name, var2_name):
        results = []
        summary_results = []
        numeric_data = pd.to_numeric(var2_series, errors='coerce').dropna()
        if len(numeric_data) < 3:
            return [f"La variable '{var2_name}' no tiene suficientes datos numéricos (n<3) tras conversión.\n"], [], None

        cat_data = var1_series.loc[numeric_data.index].astype(str)
        norm_mode = self.norm_mode.get()
        test_name_g, stat_g, p_g, global_normal = check_normality(numeric_data, norm_mode)

        results.append(f"--- Análisis Continuo: {var2_name} (agrupado por {var1_name}) ---\n")
        results.append(f"--- Prueba de Normalidad Global para '{var2_name}' (n = {len(numeric_data)}) ---\n")
        results.append(f"Modo Normalidad: {norm_mode}, Prueba usada: {test_name_g}\n")
        results.append(f"Estadístico = {stat_g:.4f}, p = {p_g:.4f}\n")

        if global_normal:
            results.append("=> Distribución global: NORMAL (p > 0.05)\n")
        else:
            results.append("=> Distribución global: NO NORMAL (p <= 0.05)\n")
        
        results.append(f"Media = {numeric_data.mean():.4f}, Desv.Est. = {numeric_data.std():.4f}\n")
        results.append(f"Mediana = {numeric_data.median():.4f}\n\n")

        unique_groups = cat_data.unique()
        group_normal_dict = {}
        if len(unique_groups) > 1:
            results.append(f"--- Pruebas de Normalidad por Grupo ---\n")
            group_arrays, group_names = [], []
            for g in sorted(unique_groups):
                d = numeric_data[cat_data == g]
                if len(d) < 3: continue
                tn, st, pv, isnorm = check_normality(d, norm_mode)
                group_normal_dict[g] = isnorm
                desc = f"Media={d.mean():.4f}" if isnorm else f"Mediana={d.median():.4f}"
                results.append(f"  Grupo {g} (n={len(d)}): {tn}, p={pv:.4f}, Normal={isnorm}; {desc}\n")
                group_arrays.append(d.values)
                group_names.append(g)

            if len(group_arrays) >= 2:
                comp_mode = self.comp_mode.get()
                all_groups_normal = all(group_normal_dict.values())

                if comp_mode == "Paramétrico" or (comp_mode == "Automático" and all_groups_normal):
                    if len(group_arrays) == 2:
                        stat_diff, p_diff = ttest_ind(group_arrays[0], group_arrays[1], nan_policy='omit')
                        test_used = "t-test"
                    else:
                        stat_diff, p_diff = f_oneway(*group_arrays)
                        test_used = "ANOVA"
                else:
                    stat_diff, p_diff = kruskal(*group_arrays)
                    test_used = "Kruskal-Wallis"
                
                results.append(f"\n--- Prueba Global de Diferencias entre Grupos ({test_used}) ---\n")
                results.append(f"Estadístico = {stat_diff:.4f}, p = {p_diff:.4f}\n")
                
                conclusion = f"Conclusión para [{var2_name}]: "
                if p_diff < 0.05:
                    conclusion += "Hay diferencias significativas entre los grupos (p < 0.05).\n"
                else:
                    conclusion += "No hay diferencias significativas entre los grupos (p >= 0.05).\n"
                results.append(conclusion)
                summary_results.append(conclusion)

                results.append("\n--- Comparaciones Pareadas entre Grupos ---\n")
                for i in range(len(group_names)):
                    for j in range(i + 1, len(group_names)):
                        g1, g2 = group_names[i], group_names[j]
                        d1, d2 = numeric_data[cat_data == g1], numeric_data[cat_data == g2]
                        is_normal1, is_normal2 = group_normal_dict.get(g1, False), group_normal_dict.get(g2, False)

                        if comp_mode == "Paramétrico" or (comp_mode == "Automático" and is_normal1 and is_normal2):
                            st_p, p_p = ttest_ind(d1, d2, nan_policy='omit')
                            pair_test = "t-test"
                        else:
                            st_p, p_p = mannwhitneyu(d1, d2, alternative='two-sided')
                            pair_test = "Mann-Whitney U"

                        line = f"  {g1} vs {g2} ({pair_test}): p = {p_p:.4f} -> "
                        line += "Diferencia" if p_p < 0.05 else "No diferencia"
                        results.append(line + "\n")
        else:
            results.append(f"'{var1_name}' no tiene más de un grupo; no se realizaron pruebas de comparación.\n")
        
        return results, summary_results, (numeric_data, cat_data)

    def _create_qq_popup(self, plot_data_list):
        popup = tk.Toplevel(self)
        popup.title("Gráficos Q-Q por Categoría")
        popup.geometry("900x700")

        n_plots = sum(len(cat_data.unique()) for _, cat_data, _ in plot_data_list)
        if n_plots == 0:
            ttk.Label(popup, text="No hay datos suficientes para generar gráficos Q-Q.").pack()
            return
            
        cols = int(np.ceil(np.sqrt(n_plots)))
        rows = int(np.ceil(n_plots / cols))
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
        axes = np.array(axes).flatten()
        ax_idx = 0

        for numeric_data, cat_data, var2_name in plot_data_list:
            unique_groups = sorted(cat_data.unique())
            for group in unique_groups:
                if ax_idx >= len(axes): break
                ax = axes[ax_idx]
                group_data = numeric_data[cat_data == group]
                if len(group_data) < 3: continue

                probplot(group_data, dist="norm", plot=ax)
                mean_val = group_data.mean()
                median_val = group_data.median()
                ax.axvline(mean_val, color='red', linestyle='--', linewidth=1, label=f'Media: {mean_val:.2f}')
                ax.axvline(median_val, color='green', linestyle=':', linewidth=1, label=f'Mediana: {median_val:.2f}')
                ax.legend(fontsize=8)
                ax.set_title(f"{var2_name} - Grupo: {group} (n={len(group_data)})", fontsize=10)
                ax.set_xlabel("Teórico", fontsize=8)
                ax.set_ylabel("Muestra", fontsize=8)
                ax_idx += 1

        for i in range(ax_idx, len(axes)):
            axes[i].set_visible(False)

        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=popup)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def generate_analysis(self):
        self.txt_output.delete("1.0", tk.END)
        if self.data is None:
            messagebox.showwarning("Aviso", "Carga los datos primero.")
            return

        try:
            filtered_data = self._apply_general_filters(self.data)
            if filtered_data.empty:
                messagebox.showwarning("Aviso", "El filtro general no ha devuelto ningún dato.")
                return
        except Exception as e:
            messagebox.showerror("Error de Filtro", f"Error al aplicar el filtro general:\n{e}")
            return

        var1_name = self.cmb_var1.get().strip()
        selected_indices = self.var2_listbox.curselection()
        var2_names = [self.var2_listbox.get(i) for i in selected_indices]

        if not var1_name or not var2_names:
            messagebox.showwarning("Aviso", "Debes seleccionar la Variable 1 y al menos una Variable 2.")
            return

        analysis_mode = self.analysis_type.get()
        output_mode = self.output_mode.get()
        self.ax.clear()
        
        all_results, all_summary_results, qq_plot_data_list = [], [], []
        first_plot_done = False

        for var2_name in var2_names:
            var1_series, var2_series = self.get_data_for_variable_pair(filtered_data, var1_name, var2_name, analysis_mode)

            if var1_series is None or var2_series is None or var1_series.empty or var2_series.empty:
                all_results.append(f"--- No se pudo procesar el par: {var1_name} vs {var2_name} (datos insuficientes tras filtros) ---\n\n")
                continue

            if analysis_mode == "Categórico (Chi-cuadrado)":
                results, summary = self._perform_categorical_analysis(var1_series, var2_series, var1_name, var2_name)
                all_results.extend(results)
                all_summary_results.extend(summary)
            
            elif analysis_mode == "Continuo (Q-Q)":
                results, summary, plot_data = self._perform_continuous_analysis(var1_series, var2_series, var1_name, var2_name)
                all_results.extend(results)
                all_summary_results.extend(summary)

                if plot_data:
                    if self.show_qq_popup.get():
                        qq_plot_data_list.append((*plot_data, var2_name))
                    elif not first_plot_done:
                        numeric_data, _, = plot_data
                        probplot(numeric_data, dist="norm", plot=self.ax)
                        mean_val = numeric_data.mean()
                        median_val = numeric_data.median()
                        self.ax.axvline(mean_val, color='red', linestyle='--', linewidth=1, label=f'Media: {mean_val:.2f}')
                        self.ax.axvline(median_val, color='green', linestyle=':', linewidth=1, label=f'Mediana: {median_val:.2f}')
                        self.ax.legend()
                        self.ax.set_title(f"Q-Q Plot Global de {var2_name}")
                        first_plot_done = True

            all_results.append("\n" + "="*50 + "\n\n")

        if output_mode == "Resumen":
            self.txt_output.insert(tk.END, "".join(all_summary_results) or "No se generaron conclusiones resumidas.")
        else:
            self.txt_output.insert(tk.END, "".join(all_results))

        if self.show_qq_popup.get() and qq_plot_data_list:
            self._create_qq_popup(qq_plot_data_list)
        
        self.canvas.draw()

    def save_qq_plot(self):
        if self.data is None:
            messagebox.showwarning("Aviso", "No hay datos cargados.")
            return
        
        if not self.ax.get_lines():
            messagebox.showwarning("Aviso", "No hay ningún gráfico en el panel principal para guardar.")
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("PDF files", "*.pdf"), ("SVG files", "*.svg")]
        )
        if not file_path:
            return
        try:
            dpi = float(self.qq_dpi.get())
            width_px = float(self.qq_width.get())
            height_px = float(self.qq_height.get())
            width_inch = width_px / dpi
            height_inch = height_px / dpi
            self.figure.set_size_inches(width_inch, height_inch)
            self.figure.savefig(file_path, dpi=dpi, bbox_inches="tight")
            messagebox.showinfo("Guardado", f"Gráfico guardado en: {file_path}")
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo guardar el gráfico:\n{e}")

    def show_summary_popup(self):
        summary_content = self.txt_output.get("1.0", tk.END).strip()
        if not summary_content:
            messagebox.showwarning("Aviso", "No hay resumen para mostrar.")
            return
        popup = tk.Toplevel(self)
        popup.title("Resumen de Resultados")
        text_frame = ttk.Frame(popup)
        text_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        txt = tk.Text(text_frame, wrap="none", width=100, height=30)
        vsb = ttk.Scrollbar(text_frame, orient="vertical", command=txt.yview)
        hsb = ttk.Scrollbar(text_frame, orient="horizontal", command=txt.xview)
        txt.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        hsb.pack(side=tk.BOTTOM, fill=tk.X)
        txt.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        txt.insert(tk.END, summary_content)
        txt.config(state="disabled")
        btn_copy = ttk.Button(popup, text="Copiar Resumen", command=lambda: self._copy_to_clipboard(popup, summary_content))
        btn_copy.pack(pady=5)

    def _copy_to_clipboard(self, window, text_to_copy):
        try:
            window.clipboard_clear()
            window.clipboard_append(text_to_copy)
            messagebox.showinfo("Copiado", "Resultados copiados al portapapeles.", parent=window)
        except tk.TclError:
            messagebox.showwarning("Error Portapapeles", "No se pudo acceder al portapapeles.", parent=window)
        except Exception as e:
             messagebox.showerror("Error", f"Error inesperado al copiar:\n{e}", parent=window)

    def log(self, message, level="INFO"):
        print(f"[{level}] GraficaQQ: {message}")


if __name__ == "__main__":
    root = tk.Tk()
    root.title("Análisis Estadístico Integrado (Q-Q / Chi-cuadrado) con Filtros Avanzados")
    root.geometry("1200x700")
    app = GraficaQQ(root)
    app.pack(fill=tk.BOTH, expand=True)
    root.mainloop()
