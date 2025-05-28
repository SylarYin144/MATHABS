#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.stats import (
    shapiro, kstest, ttest_ind, mannwhitneyu,
    f_oneway, kruskal, probplot, chi2_contingency, fisher_exact
)
import traceback # Añadido para logging

# FilterComponent ha sido eliminado.
FilterComponent = None # Mantener para evitar errores si alguna lógica residual lo verifica.

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
    Aplica la prueba de normalidad según 'mode':
      - "Automático": Shapiro-Wilk si n < 50, sino Kolmogorov-Smirnov (sobre datos estandarizados).
      - "Shapiro-Wilk": siempre Shapiro-Wilk.
      - "Kolmogorov-Smirnov": siempre KS (sobre datos estandarizados).
    Retorna (nombre_test, estadístico, p_valor, es_normal).
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
    """
    Pestaña de Análisis Estadístico Integrado:
      - Permite cargar datos (Excel/CSV).
      - La variable 1 se trata siempre como categórica.
      - La variable 2 puede ser numérica o categórica.
      - Se puede elegir entre análisis Continuo (Q-Q Plot y pruebas) o Categórico (Chi-cuadrado).
      - Se incluyen filtros avanzados para Var1, Var2 y hasta 3 filtros adicionales.
      - La eliminación de blancos es configurable de forma independiente para Var1 y Var2.
      - La eliminación de valores no numéricos se aplica solo a Var2.
      - La gráfica se previsualiza embebida y se puede guardar con dimensiones y DPI definidos.
    """
    def __init__(self, master):
        super().__init__(master)
        self.data = None

        # Opciones de limpieza para cada variable
        self.remove_blanks_var1 = tk.BooleanVar(value=True)
        self.remove_blanks_var2 = tk.BooleanVar(value=True)
        self.remove_non_numeric_var2 = tk.BooleanVar(value=True)

        # Lista para filtros adicionales: se crearán 3 pares (combobox y entry)
        self.extra_filters = []

        # Variables para Filtros Generales (hasta 2 filtros)
        self.filter_active_1_var = tk.BooleanVar(value=False)
        self.filter_col_1_var = tk.StringVar()
        self.filter_op_1_var = tk.StringVar()
        self.filter_val_1_var = tk.StringVar()

        self.filter_active_2_var = tk.BooleanVar(value=False)
        self.filter_col_2_var = tk.StringVar()
        self.filter_op_2_var = tk.StringVar()
        self.filter_val_2_var = tk.StringVar()
        
        self.general_filter_operators = ["==", "!=", ">", "<", ">=", "<=", "contiene", "no contiene", "es NaN", "no es NaN"]

        self.create_widgets()

    def create_widgets(self):
        # Panel principal dividido en Resultados y Opciones+Gráfica
        main_pane = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        main_pane.pack(fill=tk.BOTH, expand=True)

        # --- Panel Izquierdo: Resultados ---
        left_frame = ttk.Frame(main_pane)
        main_pane.add(left_frame, weight=0)
        frm_result = ttk.LabelFrame(left_frame, text="Resultados")
        frm_result.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.txt_output = tk.Text(frm_result, height=25, wrap="none")
        self.txt_output.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        btn_clear = ttk.Button(left_frame, text="Borrar", command=self.clear_output)
        btn_clear.pack(padx=5, pady=5, anchor="e")

        # --- Panel Derecho: Opciones (scroll) y Gráfica ---
        right_frame = ttk.Frame(main_pane)
        main_pane.add(right_frame, weight=1)
        vertical_pane = ttk.PanedWindow(right_frame, orient=tk.VERTICAL)
        vertical_pane.pack(fill=tk.BOTH, expand=True)

        # Frame de opciones con scroll
        scroll_container = ttk.Frame(vertical_pane)
        vertical_pane.add(scroll_container, weight=0)
        self.scrollable_options = create_scrollable_frame(scroll_container)

        # Frame para la gráfica
        figure_frame = ttk.Frame(vertical_pane)
        vertical_pane.add(figure_frame, weight=1)
        self.figure = plt.Figure(figsize=(6, 4), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, master=figure_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # --- CONTROLES DE OPCIONES ---
        # 1. Cargar Datos
        frm_load = ttk.LabelFrame(self.scrollable_options, text="Cargar Datos (Excel/CSV)")
        frm_load.pack(fill=tk.X, padx=5, pady=5)
        btn_load = ttk.Button(frm_load, text="Cargar Archivo", command=self.load_data)
        btn_load.pack(side=tk.LEFT, padx=5, pady=5)

        # 2. Tipo de Análisis
        frm_analysis = ttk.LabelFrame(self.scrollable_options, text="Tipo de Análisis")
        frm_analysis.pack(fill=tk.X, padx=5, pady=5)
        self.analysis_type = tk.StringVar(value="Continuo (Q-Q)")
        rb_cont = ttk.Radiobutton(frm_analysis, text="Continuo (Q-Q)", variable=self.analysis_type, value="Continuo (Q-Q)")
        rb_cat = ttk.Radiobutton(frm_analysis, text="Categórico (Chi-cuadrado)", variable=self.analysis_type, value="Categórico (Chi-cuadrado)")
        rb_cont.pack(side=tk.LEFT, padx=5, pady=5)
        rb_cat.pack(side=tk.LEFT, padx=5, pady=5)

        # 3. Selección de Variables (Var1 y Var2)
        frm_vars = ttk.LabelFrame(self.scrollable_options, text="Selección de Variables")
        frm_vars.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(frm_vars, text="Variable 1:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.cmb_var1 = ttk.Combobox(frm_vars, values=[], state="readonly")
        self.cmb_var1.grid(row=0, column=1, padx=5, pady=5, sticky="we")
        ttk.Label(frm_vars, text="Filtro/Etiquetas para Variable 1:").grid(row=0, column=2, padx=5, pady=5, sticky="e")
        self.entry_var1_filter = ttk.Entry(frm_vars, width=40)
        self.entry_var1_filter.grid(row=0, column=3, padx=5, pady=5, sticky="we")
        ttk.Label(frm_vars, text="Variable 2:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.cmb_var2 = ttk.Combobox(frm_vars, values=[], state="readonly")
        self.cmb_var2.grid(row=1, column=1, padx=5, pady=5, sticky="we")
        ttk.Label(frm_vars, text="Filtro/Etiquetas para Variable 2:").grid(row=1, column=2, padx=5, pady=5, sticky="e")
        self.entry_var2_filter = ttk.Entry(frm_vars, width=40)
        self.entry_var2_filter.grid(row=1, column=3, padx=5, pady=5, sticky="we")
        frm_vars.columnconfigure(1, weight=1)
        frm_vars.columnconfigure(3, weight=1)

        # 4. Filtros Generales (Sección eliminada ya que FilterComponent fue removido)

        # 5. Opciones de Limpieza de Datos - Ajustar número
        frm_clean = ttk.LabelFrame(self.scrollable_options, text="Opciones de Limpieza (Aplicado a Var1/Var2 después de filtros)")
        frm_clean.pack(fill=tk.X, padx=5, pady=5)
        chk_blanks_var1 = ttk.Checkbutton(frm_clean, text="Eliminar blancos en Variable 1", variable=self.remove_blanks_var1)
        chk_blanks_var1.pack(anchor=tk.W, padx=5, pady=2)
        chk_blanks_var2 = ttk.Checkbutton(frm_clean, text="Eliminar blancos en Variable 2", variable=self.remove_blanks_var2)
        chk_blanks_var2.pack(anchor=tk.W, padx=5, pady=2)
        chk_non_numeric_var2 = ttk.Checkbutton(frm_clean, text="Eliminar valores no numéricos en Variable 2", variable=self.remove_non_numeric_var2)
        chk_non_numeric_var2.pack(anchor=tk.W, padx=5, pady=2)

        # 6. Prueba de Normalidad - Ajustar número
        frm_normal = ttk.LabelFrame(self.scrollable_options, text="Prueba de Normalidad (Var2 Continua)")
        frm_normal.pack(fill=tk.X, padx=5, pady=5)
        self.norm_mode = tk.StringVar(value="Automático")
        rb_norm_auto = ttk.Radiobutton(frm_normal, text="Automático", variable=self.norm_mode, value="Automático")
        rb_norm_shapiro = ttk.Radiobutton(frm_normal, text="Shapiro-Wilk", variable=self.norm_mode, value="Shapiro-Wilk")
        rb_norm_ks = ttk.Radiobutton(frm_normal, text="Kolmogorov-Smirnov", variable=self.norm_mode, value="Kolmogorov-Smirnov")
        rb_norm_auto.pack(side=tk.LEFT, padx=5, pady=2)
        rb_norm_shapiro.pack(side=tk.LEFT, padx=5, pady=2)
        rb_norm_ks.pack(side=tk.LEFT, padx=5, pady=2)

        # 7. Prueba de Diferencias (Var2 Continua vs Var1 Categórica) - Ajustar número
        frm_comp = ttk.LabelFrame(self.scrollable_options, text="Prueba de Diferencias (Var2 vs Var1)")
        frm_comp.pack(fill=tk.X, padx=5, pady=5)
        self.comp_mode = tk.StringVar(value="Automático")
        rb_comp_auto = ttk.Radiobutton(frm_comp, text="Automático", variable=self.comp_mode, value="Automático")
        rb_comp_param = ttk.Radiobutton(frm_comp, text="Paramétrico", variable=self.comp_mode, value="Paramétrico")
        rb_comp_nonparam = ttk.Radiobutton(frm_comp, text="No paramétrico", variable=self.comp_mode, value="No paramétrico")
        rb_comp_auto.pack(side=tk.LEFT, padx=5, pady=2)
        rb_comp_param.pack(side=tk.LEFT, padx=5, pady=2)
        rb_comp_nonparam.pack(side=tk.LEFT, padx=5, pady=2)

        # 8. Opciones de Análisis Categórico (Var1 vs Var2) - Ajustar número
        frm_cat_opts = ttk.LabelFrame(self.scrollable_options, text="Opciones Análisis Categórico (Var1 vs Var2)")
        frm_cat_opts.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(frm_cat_opts, text="Test Categórico:").grid(row=0, column=0, padx=5, pady=2, sticky="w")
        self.cat_test_mode = tk.StringVar(value="Automático")
        rb_cat_auto = ttk.Radiobutton(frm_cat_opts, text="Automático", variable=self.cat_test_mode, value="Automático")
        rb_cat_chi = ttk.Radiobutton(frm_cat_opts, text="Chi-cuadrado", variable=self.cat_test_mode, value="Chi-cuadrado")
        rb_cat_fisher = ttk.Radiobutton(frm_cat_opts, text="Fisher Exact", variable=self.cat_test_mode, value="Fisher Exact")
        rb_cat_auto.grid(row=0, column=1, padx=5, pady=2, sticky="w")
        rb_cat_chi.grid(row=0, column=2, padx=5, pady=2, sticky="w")
        rb_cat_fisher.grid(row=0, column=3, padx=5, pady=2, sticky="w")
        ttk.Label(frm_cat_opts, text="Formato Salida:").grid(row=1, column=0, padx=5, pady=2, sticky="w")
        self.cat_format = tk.StringVar(value="Números")
        rb_format_num = ttk.Radiobutton(frm_cat_opts, text="Números", variable=self.cat_format, value="Números")
        rb_format_pct = ttk.Radiobutton(frm_cat_opts, text="Porcentajes", variable=self.cat_format, value="Porcentajes")
        rb_format_num.grid(row=1, column=1, padx=5, pady=2, sticky="w")
        rb_format_pct.grid(row=1, column=2, padx=5, pady=2, sticky="w")

        # 9. Opciones Gráficas Q-Q Plot (Personalización) - Ajustar número
        frm_graph_opts = ttk.LabelFrame(self.scrollable_options, text="Opciones Gráficas (Q-Q Plot - Var2 Continua)")
        frm_graph_opts.pack(fill=tk.X, padx=5, pady=5)
        color_list = ["blue","red","green","black","orange","purple","cyan","magenta","yellow","brown","pink","gray"]
        ttk.Label(frm_graph_opts, text="Color puntos:").grid(row=0, column=0, padx=5, pady=2, sticky="w")
        self.cmb_marker_face = ttk.Combobox(frm_graph_opts, values=color_list, state="readonly", width=8)
        self.cmb_marker_face.grid(row=0, column=1, padx=5, pady=2, sticky="w")
        self.cmb_marker_face.set("blue")
        ttk.Label(frm_graph_opts, text="Borde puntos:").grid(row=0, column=2, padx=5, pady=2, sticky="w")
        self.cmb_marker_edge = ttk.Combobox(frm_graph_opts, values=color_list, state="readonly", width=8)
        self.cmb_marker_edge.grid(row=0, column=3, padx=5, pady=2, sticky="w")
        self.cmb_marker_edge.set("black")
        ttk.Label(frm_graph_opts, text="Tamaño puntos:").grid(row=0, column=4, padx=5, pady=2, sticky="w")
        self.entry_marker_size = ttk.Entry(frm_graph_opts, width=5)
        self.entry_marker_size.grid(row=0, column=5, padx=5, pady=2, sticky="w")
        self.entry_marker_size.insert(0, "6")
        ttk.Label(frm_graph_opts, text="Color línea ref:").grid(row=1, column=0, padx=5, pady=2, sticky="w")
        self.cmb_line_color = ttk.Combobox(frm_graph_opts, values=color_list, state="readonly", width=8)
        self.cmb_line_color.grid(row=1, column=1, padx=5, pady=2, sticky="w")
        self.cmb_line_color.set("red")
        ttk.Label(frm_graph_opts, text="Grosor línea ref:").grid(row=1, column=2, padx=5, pady=2, sticky="w")
        self.entry_line_width = ttk.Entry(frm_graph_opts, width=5)
        self.entry_line_width.grid(row=1, column=3, padx=5, pady=2, sticky="w")
        self.entry_line_width.insert(0, "2")
        ttk.Label(frm_graph_opts, text="Título:").grid(row=2, column=0, padx=5, pady=2, sticky="w")
        self.entry_title = ttk.Entry(frm_graph_opts, width=20)
        self.entry_title.grid(row=2, column=1, padx=5, pady=2, sticky="w")
        ttk.Label(frm_graph_opts, text="Fuente Título:").grid(row=2, column=2, padx=5, pady=2, sticky="w")
        self.entry_title_fontsize = ttk.Entry(frm_graph_opts, width=5)
        self.entry_title_fontsize.grid(row=2, column=3, padx=5, pady=2, sticky="w")
        self.entry_title_fontsize.insert(0, "12")
        ttk.Label(frm_graph_opts, text="Etiqueta X:").grid(row=3, column=0, padx=5, pady=2, sticky="w")
        self.entry_xlabel = ttk.Entry(frm_graph_opts, width=20)
        self.entry_xlabel.grid(row=3, column=1, padx=5, pady=2, sticky="w")
        ttk.Label(frm_graph_opts, text="Etiqueta Y:").grid(row=3, column=2, padx=5, pady=2, sticky="w")
        self.entry_ylabel = ttk.Entry(frm_graph_opts, width=20)
        self.entry_ylabel.grid(row=3, column=3, padx=5, pady=2, sticky="w")
        ttk.Label(frm_graph_opts, text="Fuente Ejes:").grid(row=3, column=4, padx=5, pady=2, sticky="w")
        self.entry_label_fontsize = ttk.Entry(frm_graph_opts, width=5)
        self.entry_label_fontsize.grid(row=3, column=5, padx=5, pady=2, sticky="w")
        self.entry_label_fontsize.insert(0, "10")
        ttk.Label(frm_graph_opts, text="Ancho (px):").grid(row=4, column=0, padx=5, pady=2, sticky="w")
        self.qq_width = ttk.Entry(frm_graph_opts, width=8)
        self.qq_width.grid(row=4, column=1, padx=5, pady=2, sticky="w")
        self.qq_width.insert(0, "800")
        ttk.Label(frm_graph_opts, text="Alto (px):").grid(row=4, column=2, padx=5, pady=2, sticky="w")
        self.qq_height = ttk.Entry(frm_graph_opts, width=8)
        self.qq_height.grid(row=4, column=3, padx=5, pady=2, sticky="w")
        self.qq_height.insert(0, "600")
        ttk.Label(frm_graph_opts, text="DPI:").grid(row=4, column=4, padx=5, pady=2, sticky="w")
        self.qq_dpi = ttk.Entry(frm_graph_opts, width=5)
        self.qq_dpi.grid(row=4, column=5, padx=5, pady=2, sticky="w")
        self.qq_dpi.insert(0, "100")

        # 10. Botones de Acción - Ajustar número
        frm_run = ttk.Frame(self.scrollable_options)
        frm_run.pack(fill=tk.X, padx=5, pady=10)
        btn_run = ttk.Button(frm_run, text="Mostrar Análisis", command=self.generate_analysis)
        btn_run.pack(side=tk.LEFT, padx=10)
        btn_save = ttk.Button(frm_run, text="Guardar Q-Q Plot", command=self.save_qq_plot)
        btn_save.pack(side=tk.LEFT, padx=10)

    # ------------------- Funciones Auxiliares -------------------
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
            self.cmb_var2['values'] = [""] + cols
            if cols:
                self.cmb_var1.current(0)
                self.cmb_var2.set("")
            # FilterComponent removido.
            # Actualizar combos de filtros generales
            filter_cols_options = [''] + cols
            if hasattr(self, 'filter_col_1_combo'): # Verificar si los widgets ya existen
                self.filter_col_1_combo['values'] = filter_cols_options
                if not self.filter_col_1_var.get() and cols: self.filter_col_1_var.set('')
            if hasattr(self, 'filter_col_2_combo'):
                self.filter_col_2_combo['values'] = filter_cols_options
                if not self.filter_col_2_var.get() and cols: self.filter_col_2_var.set('')
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo cargar el archivo:\n{e}")
            # FilterComponent removido.
            # Limpiar combos de filtros generales en caso de error
            if hasattr(self, 'filter_col_1_combo'): self.filter_col_1_combo['values'] = ['']
            if hasattr(self, 'filter_col_2_combo'): self.filter_col_2_combo['values'] = ['']

    # Se elimina apply_filter_criteria
        """
        Aplica filtrado según:
          - "1:GrupoA,2:GrupoB" para mapeo.
          - "1-5" para rango.
          - "1,2,3" para lista.
          - "1" para valor único.
        """
        if not filter_string.strip():
            return series
        s_str = series.astype(str)
        if ":" in filter_string:
            mapping = {}
            parts = [p.strip() for p in filter_string.split(",") if p.strip()]
            for part in parts:
                if ":" in part:
                    key, label = part.split(":", 1)
                    mapping[key.strip()] = label.strip()
            filtered = s_str[s_str.isin(list(mapping.keys()))]
            return filtered.map(mapping)
        if "-" in filter_string and "," not in filter_string:
            parts = filter_string.split("-")
            if len(parts) == 2:
                try:
                    low = float(parts[0].strip())
                    high = float(parts[1].strip())
                    s_num = pd.to_numeric(series, errors="coerce")
                    return s_num[(s_num >= low) & (s_num <= high)].dropna()
                except:
                    return series.dropna()
        if "," in filter_string:
            if pd.api.types.is_numeric_dtype(series):
                try:
                    numeric_vals = [float(p.strip()) for p in filter_string.split(",") if p.strip()]
                    return series[series.isin(numeric_vals)]
                except:
                    pass
            else:
                vals = [p.strip() for p in filter_string.split(",") if p.strip()]
                return series[series.isin(vals)]
        try:
            single_val = float(filter_string)
            return series[pd.to_numeric(series, errors="coerce") == single_val]
        except:
            return series[s_str == filter_string]

    def _apply_general_filters(self, df):
        """Aplica los filtros generales configurados en la UI al DataFrame."""
        df_filtered = df.copy()
        self.log("Aplicando filtros generales...", "INFO")

        filter_configs = [
            (self.filter_active_1_var, self.filter_col_1_var, self.filter_op_1_var, self.filter_val_1_var),
            (self.filter_active_2_var, self.filter_col_2_var, self.filter_op_2_var, self.filter_val_2_var)
        ]

        for i, (active_var, col_var, op_var, val_var) in enumerate(filter_configs, 1):
            if active_var.get():
                col_name = col_var.get()
                op = op_var.get()
                val_str = val_var.get()

                if not col_name:
                    self.log(f"Filtro {i}: Columna no seleccionada, omitiendo.", "WARN")
                    continue
                if col_name not in df_filtered.columns:
                    self.log(f"Filtro {i}: Columna '{col_name}' no encontrada en los datos, omitiendo.", "ERROR")
                    continue
                
                self.log(f"Filtro {i}: Col='{col_name}', Op='{op}', Val='{val_str}'", "DEBUG")

                try:
                    col_series = df_filtered[col_name]
                    
                    if op == "es NaN":
                        df_filtered = df_filtered[col_series.isna()]
                    elif op == "no es NaN":
                        df_filtered = df_filtered[col_series.notna()]
                    else:
                        if val_str == "" and op not in ["es NaN", "no es NaN"]:
                             self.log(f"Filtro {i}: Valor vacío para operador '{op}', omitiendo.", "WARN")
                             continue

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
                                self.log(f"Filtro {i}: Valor '{val_str}' no es numérico para columna numérica '{col_name}'. Omitiendo filtro.", "WARN")
                                continue
                        elif pd.api.types.is_string_dtype(col_series) or col_series.dtype == 'object':
                            if op == "==": df_filtered = df_filtered[col_series.astype(str) == val_str]
                            elif op == "!=": df_filtered = df_filtered[col_series.astype(str) != val_str]
                            elif op == "contiene": df_filtered = df_filtered[col_series.astype(str).str.contains(val_str, case=False, na=False)]
                            elif op == "no contiene": df_filtered = df_filtered[~col_series.astype(str).str.contains(val_str, case=False, na=False)]
                            else:
                                self.log(f"Filtro {i}: Operador '{op}' no recomendado para columna de texto '{col_name}'. Intentando comparación directa.", "WARN")
                                if op == ">": df_filtered = df_filtered[col_series.astype(str) > val_str]
                                elif op == "<": df_filtered = df_filtered[col_series.astype(str) < val_str]
                                elif op == ">=": df_filtered = df_filtered[col_series.astype(str) >= val_str]
                                elif op == "<=": df_filtered = df_filtered[col_series.astype(str) <= val_str]
                        else:
                             self.log(f"Filtro {i}: Tipo de columna '{col_series.dtype}' no manejado explícitamente para operador '{op}'. Intentando comparación directa.", "WARN")
                             if op == "==": df_filtered = df_filtered[col_series == val_str]
                             elif op == "!=": df_filtered = df_filtered[col_series != val_str]
                except Exception as e:
                    self.log(f"Filtro {i}: Error aplicando filtro Col='{col_name}', Op='{op}', Val='{val_str}': {e}", "ERROR")
                    traceback.print_exc()
        
        self.log(f"Datos después de filtros generales: {df_filtered.shape[0]} filas.", "INFO")
        return df_filtered

    def get_filtered_data(self):
        """Obtiene los datos filtrados usando FilterComponent y aplica filtros/limpieza específicos de Var1/Var2."""
        if self.data is None:
            self.log("No hay datos originales cargados.", "WARN")
            return None, None

        # 1. FilterComponent ha sido removido. Se trabaja directamente con una copia de self.data.
        if self.data is None:
            self.log("No hay datos originales cargados en get_filtered_data.", "WARN")
            return None, None
        df_filtered = self.data.copy()
        self.log("Usando datos originales para Q-Q (antes de filtros generales).", "INFO")

        # Aplicar filtros generales definidos en la UI
        df_filtered = self._apply_general_filters(df_filtered)
        
        if df_filtered is None or df_filtered.empty:
            self.log("DataFrame vacío después de aplicar filtros generales.", "WARN")
            # No es necesario messagebox aquí, get_filtered_data puede devolver None
            return None, None

        # 2. Extraer Var1 y Var2 del DataFrame filtrado
        var1_name = self.cmb_var1.get().strip()
        if not var1_name or var1_name not in df_filtered.columns:
            self.log(f"Variable 1 '{var1_name}' no encontrada en datos filtrados.", "ERROR")
            return None, None
        s_var1 = df_filtered[var1_name].copy()
        var2_name = self.cmb_var2.get().strip()
        s_var2 = None
        if var2_name and var2_name in df_filtered.columns:
            s_var2 = df_filtered[var2_name].copy()

        # 3. Aplicar filtros/etiquetas específicos de Var1 y Var2 (si se ingresaron)
        # Nota: Esta lógica podría refactorizarse si se desea que el FilterComponent maneje todo.
        # Por ahora, mantenemos la lógica original de apply_filter_criteria para estos campos.
        filter_var1_str = self.entry_var1_filter.get().strip()
        filter_var2_str = self.entry_var2_filter.get().strip()
        if filter_var1_str:
            s_var1 = self._apply_specific_filter(s_var1, filter_var1_str)
        if s_var2 is not None and filter_var2_str:
            s_var2 = self._apply_specific_filter(s_var2, filter_var2_str)

        # 4. Aplicar limpieza específica de Var1/Var2
        if self.remove_blanks_var1.get():
            s_var1 = s_var1.replace(r'^\s*$', np.nan, regex=True).dropna()
        if s_var2 is not None and self.remove_blanks_var2.get():
            s_var2 = s_var2.replace(r'^\s*$', np.nan, regex=True).dropna()
        if s_var2 is not None and self.remove_non_numeric_var2.get():
            s_numeric = pd.to_numeric(s_var2, errors='coerce')
            s_var2 = s_var2[~s_numeric.isna()]
        # Alinear índices
        common_idx = s_var1.dropna().index
        if s_var2 is not None:
            common_idx = common_idx.intersection(s_var2.dropna().index)
        s_var1 = s_var1.loc[common_idx]
        if s_var2 is not None:
            s_var2 = s_var2.loc[common_idx]
        return s_var1, s_var2

        # 5. Alinear índices
        common_idx = s_var1.index # Empezar con los índices de var1 (después de su limpieza/filtrado)
        if s_var2 is not None:
            common_idx = common_idx.intersection(s_var2.index) # Intersección con los de var2

        s_var1 = s_var1.loc[common_idx]
        if s_var2 is not None:
            s_var2 = s_var2.loc[common_idx]

        return s_var1, s_var2

    def _apply_specific_filter(self, series, filter_string):
        """Aplica filtro/etiquetado de Var1/Var2 (lógica similar a apply_filter_criteria original)."""
        if not filter_string.strip():
            return series
        s_str = series.astype(str)
        if ":" in filter_string: # Mapeo/Etiquetado
            mapping = {}
            keys_to_keep = []
            parts = [p.strip() for p in filter_string.split(",") if p.strip()]
            for part in parts:
                if ":" in part:
                    key, label = part.split(":", 1)
                    mapping[key.strip()] = label.strip()
                    keys_to_keep.append(key.strip())
            # Filtrar por las claves originales y luego mapear
            filtered = series[s_str.isin(keys_to_keep)]
            return filtered.map(mapping) # Devuelve la serie mapeada
        elif "-" in filter_string and "," not in filter_string: # Rango numérico
            parts = filter_string.split("-")
            if len(parts) == 2:
                try:
                    low = float(parts[0].strip()); high = float(parts[1].strip())
                    s_num = pd.to_numeric(series, errors="coerce")
                    return series[(s_num >= low) & (s_num <= high)] # Devuelve la serie filtrada
                except ValueError: return series # Ignorar si el rango no es válido
        elif "," in filter_string: # Lista de valores
            vals = [p.strip() for p in filter_string.split(",") if p.strip()]
            if pd.api.types.is_numeric_dtype(series):
                num_vals = []
                for v_str in vals:
                    try: num_vals.append(float(v_str))
                    except ValueError: num_vals.append(v_str) # Mantener como string si falla
                return series[series.isin(num_vals)]
            else:
                return series[s_str.isin(vals)]
        else: # Valor único
            try: # Intentar como número
                val_num = float(filter_string)
                if pd.api.types.is_numeric_dtype(series):
                    return series[pd.to_numeric(series, errors='coerce') == val_num]
                else: # Comparar como string si la columna no es numérica
                    return series[s_str == filter_string]
            except ValueError: # Comparar como string si la conversión falla
                return series[s_str == filter_string]
        return series # Devolver original si no se aplicó filtro

    def generate_analysis(self):
        self.txt_output.delete("1.0", tk.END)
        if self.data is None:
            messagebox.showwarning("Aviso", "Carga los datos primero.")
            return
        var1_series, var2_series = self.get_filtered_data()
        # Obtener los nombres reales de las variables
        var1_name = self.cmb_var1.get().strip()
        var2_name = self.cmb_var2.get().strip() if self.cmb_var2.get().strip() else "Sin selección"
        if var1_series is None or len(var1_series) < 1:
            self.txt_output.insert(tk.END, f"No hay datos válidos en la variable '{var1_name}' tras aplicar filtros.\n")
            return
        analysis_mode = self.analysis_type.get()
        self.ax.clear()
        # Si es análisis categórico, se muestran los nombres usados
        if analysis_mode == "Categórico (Chi-cuadrado)":
            if var2_series is None or len(var2_series) < 1:
                self.txt_output.insert(tk.END, f"No hay datos válidos en la variable '{var2_name}' para análisis categórico.\n")
                return
            cat1 = var1_series.astype(str)
            cat2 = var2_series.astype(str)
            # Usar los nombres reales de las variables en el DataFrame
            df_cat = pd.DataFrame({var1_name: cat1, var2_name: cat2}).dropna()
            if df_cat.empty:
                self.txt_output.insert(tk.END, "No quedan datos tras filtros.\n")
                self.figure.tight_layout()
                self.canvas.draw()
                return
            contingency = pd.crosstab(df_cat[var1_name], df_cat[var2_name])
            chi2, p, dof, expected = chi2_contingency(contingency)
            test_type = self.cat_test_mode.get()
            output_format = self.cat_format.get()
            if contingency.shape == (2,2) and test_type == "Automático":
                if (expected < 5).any():
                    test_type = "Fisher Exact"
                else:
                    test_type = "Chi-cuadrado"
            if test_type == "Fisher Exact" and contingency.shape == (2,2):
                _, p = fisher_exact(contingency)
                test_used = "Fisher Exact"
            elif test_type == "Fisher Exact" and contingency.shape != (2,2):
                self.txt_output.insert(tk.END, "Fisher Exact solo aplica a tablas 2x2; se usará Chi-cuadrado.\n")
                test_used = "Chi-cuadrado"
            else:
                test_used = "Chi-cuadrado"
            if output_format == "Porcentajes":
                contingency_pct = contingency.div(contingency.sum().sum()) * 100
                table_str = contingency.to_string() + "\n\nPorcentajes (%):\n" + contingency_pct.to_string()
            else:
                table_str = contingency.to_string()
            results = []
            results.append(f"--- Análisis Categórico para variables: {var1_name} y {var2_name} ---\n")
            results.append("Tabla de Contingencia:\n")
            results.append(table_str + "\n")
            results.append(f"\nTest usado: {test_used}\n")
            if test_used == "Chi-cuadrado":
                results.append(f"Estadístico Chi-cuadrado = {chi2:.4f}, p = {p:.4f}, dof = {dof}\n")
            else:
                results.append(f"p (Fisher) = {p:.4f}\n")
            if p < 0.05:
                results.append("Conclusión: Existe asociación entre las variables (p < 0.05).\n")
            else:
                results.append("Conclusión: No se encontró asociación (p >= 0.05).\n")
            self.txt_output.insert(tk.END, "".join(results))
            self.figure.tight_layout()
            self.canvas.draw()
            return

        # Modo Continuo (Q-Q Plot)
        if var2_series is None or len(var2_series) < 3:
            self.txt_output.insert(tk.END, f"No hay suficientes datos numéricos en la variable '{var2_name}' para análisis continuo.\n")
            return
        numeric_data = pd.to_numeric(var2_series, errors="coerce").dropna()
        if len(numeric_data) < 3:
            self.txt_output.insert(tk.END, f"La variable '{var2_name}' no tiene suficientes datos numéricos tras conversión.\n")
            return
        cat_data = var1_series.astype(str)
        probplot(numeric_data, dist="norm", plot=self.ax)
        lines = self.ax.get_lines()
        marker_face = self.cmb_marker_face.get()
        marker_edge = self.cmb_marker_edge.get()
        try:
            marker_size = float(self.entry_marker_size.get())
        except:
            marker_size = 6
        line_color = self.cmb_line_color.get()
        try:
            line_width = float(self.entry_line_width.get())
        except:
            line_width = 2
        if len(lines) >= 2:
            points = lines[0]
            points.set_markerfacecolor(marker_face)
            points.set_markeredgecolor(marker_edge)
            points.set_markersize(marker_size)
            ref_line = lines[1]
            ref_line.set_color(line_color)
            ref_line.set_linewidth(line_width)
        title_str = self.entry_title.get().strip() or f"Q-Q Plot de {var2_name}"
        try:
            title_fontsize = int(self.entry_title_fontsize.get())
        except:
            title_fontsize = 12
        xlabel_str = self.entry_xlabel.get().strip() or "Teórico (Distribución Normal)"
        ylabel_str = self.entry_ylabel.get().strip() or "Datos Muestrales"
        try:
            label_fontsize = int(self.entry_label_fontsize.get())
        except:
            label_fontsize = 10
        self.ax.set_title(title_str, fontsize=title_fontsize)
        self.ax.set_xlabel(xlabel_str, fontsize=label_fontsize)
        self.ax.set_ylabel(ylabel_str, fontsize=label_fontsize)
        self.figure.tight_layout()
        self.canvas.draw()
        norm_mode = self.norm_mode.get()
        test_name_g, stat_g, p_g, global_normal = check_normality(numeric_data, norm_mode)
        results = []
        results.append(f"--- Prueba de Normalidad Global para '{var2_name}' (n = {len(numeric_data)}) ---\n")
        results.append(f"Modo Normalidad: {norm_mode}\n")
        results.append(f"Prueba usada: {test_name_g}\n")
        results.append(f"Estadístico = {stat_g:.4f}, p = {p_g:.4f}\n")
        if global_normal:
            results.append("=> Distribución global: NORMAL (p > 0.05)\n")
            overall_stat = f"Media = {numeric_data.mean():.4f}, Desv.Est. = {numeric_data.std():.4f}, n = {len(numeric_data)}\n"
        else:
            results.append("=> Distribución global: NO NORMAL (p <= 0.05)\n")
            overall_stat = f"Mediana = {numeric_data.median():.4f}, n = {len(numeric_data)}\n"
        results.append(overall_stat + "\n")
        # Análisis por grupos (usando Var1 para agrupar)
        unique_groups = cat_data.unique()
        group_arrays = []
        group_names = []
        group_normal_dict = {}
        if len(unique_groups) > 1:
            results.append(f"--- Pruebas de Normalidad por Grupo (agrupados por '{var1_name}') ---\n")
            for g in sorted(unique_groups):
                d = numeric_data[cat_data == g]
                if len(d) < 3:
                    results.append(f"  Grupo {g}: n = {len(d)} -> insuficiente.\n")
                    continue
                tn, st, pv, isnorm = check_normality(d, norm_mode)
                group_normal_dict[g] = isnorm
                if isnorm:
                    desc = f"Media = {d.mean():.4f}, Std = {d.std():.4f}"
                else:
                    desc = f"Mediana = {d.median():.4f}"
                results.append(f"  Grupo {g} (n = {len(d)}): {tn}, estadístico = {st:.4f}, p = {pv:.4f}, normal = {isnorm}; {desc}\n")
                group_arrays.append(d.values)
                group_names.append(g)
            if len(group_arrays) >= 2:
                comp_mode = self.comp_mode.get()
                if comp_mode == "Paramétrico":
                    stat_diff, p_diff = f_oneway(*group_arrays)
                    test_used = "ANOVA (paramétrico)"
                elif comp_mode == "No paramétrico":
                    stat_diff, p_diff = kruskal(*group_arrays)
                    test_used = "Kruskal-Wallis (no param)"
                else:
                    if global_normal:
                        stat_diff, p_diff = f_oneway(*group_arrays)
                        test_used = "ANOVA (auto)"
                    else:
                        stat_diff, p_diff = kruskal(*group_arrays)
                        test_used = "Kruskal-Wallis (auto)"
                results.append("\n--- Prueba Global de Diferencias entre Grupos ---\n")
                results.append(f"  {test_used}: estadístico = {stat_diff:.4f}, p = {p_diff:.4f}\n")
                if p_diff < 0.05:
                    results.append("  => Hay diferencias significativas (p < 0.05)\n")
                else:
                    results.append("  => No hay diferencias significativas (p >= 0.05)\n")
                results.append("\n--- Comparaciones Pareadas entre Grupos ---\n")
                differences_found = []
                no_differences = []
                for i in range(len(group_names)):
                    for j in range(i+1, len(group_names)):
                        g1 = group_names[i]
                        g2 = group_names[j]
                        d1 = numeric_data[cat_data == g1]
                        d2 = numeric_data[cat_data == g2]
                        if comp_mode == "Paramétrico":
                            st_p, p_p = ttest_ind(d1, d2, nan_policy='omit')
                            pair_test = "t-test"
                        elif comp_mode == "No paramétrico":
                            st_p, p_p = mannwhitneyu(d1, d2, alternative='two-sided')
                            pair_test = "Mann-Whitney"
                        else:
                            if group_normal_dict[g1] and group_normal_dict[g2]:
                                st_p, p_p = ttest_ind(d1, d2, nan_policy='omit')
                                pair_test = "t-test (auto)"
                            else:
                                st_p, p_p = mannwhitneyu(d1, d2, alternative='two-sided')
                                pair_test = "Mann-Whitney (auto)"
                        line = f"  {g1} vs {g2} ({pair_test}): estadístico = {st_p:.4f}, p = {p_p:.4f} -> "
                        if p_p < 0.05:
                            line += "Diferencia"
                            differences_found.append(f"{g1} vs {g2}")
                        else:
                            line += "No diferencia"
                            no_differences.append(f"{g1} vs {g2}")
                        results.append(line + "\n")
                results.append("\nResumen Pareado:\n")
                if differences_found:
                    results.append("  Con diferencias: " + ", ".join(differences_found) + "\n")
                else:
                    results.append("  No se encontraron diferencias significativas.\n")
                if no_differences:
                    results.append("  Sin diferencias: " + ", ".join(no_differences) + "\n")
        else:
            results.append(f"'{var1_name}' no tiene más de un grupo; se hizo solo análisis global.\n")
        self.txt_output.insert(tk.END, "".join(results))

    def save_qq_plot(self):
        if self.data is None:
            messagebox.showwarning("Aviso", "No hay datos cargados.")
            return
        if self.analysis_type.get() == "Categórico (Chi-cuadrado)":
            messagebox.showwarning("Aviso", "En modo categórico no se genera Q-Q plot.")
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
        txt.config(state="disabled") # Hacerlo no editable
        btn_copy = ttk.Button(popup, text="Copiar Resumen", command=lambda: self._copy_to_clipboard(popup, summary_content))
        btn_copy.pack(pady=5)

    def _copy_to_clipboard(self, window, text_to_copy):
        """Copia texto al portapapeles."""
        try:
            window.clipboard_clear()
            window.clipboard_append(text_to_copy)
            messagebox.showinfo("Copiado", "Resultados copiados al portapapeles.", parent=window)
        except tk.TclError:
            messagebox.showwarning("Error Portapapeles", "No se pudo acceder al portapapeles.", parent=window)
        except Exception as e:
             messagebox.showerror("Error", f"Error inesperado al copiar:\n{e}", parent=window)

    def log(self, message, level="INFO"):
        """Placeholder para logging."""
        print(f"[{level}] GraficaQQ: {message}")


if __name__ == "__main__":
    # Envolver la aplicación completa en un canvas con scrollbar vertical para toda la pantalla
    root = tk.Tk()
    root.title("Análisis Estadístico Integrado (Q-Q / Chi-cuadrado) con Filtros Avanzados")
    root.geometry("1200x700")
    main_canvas = tk.Canvas(root)
    main_canvas.pack(side="left", fill="both", expand=True)
    v_scroll = ttk.Scrollbar(root, orient="vertical", command=main_canvas.yview)
    v_scroll.pack(side="right", fill="y")
    main_canvas.configure(yscrollcommand=v_scroll.set)
    main_frame = ttk.Frame(main_canvas)
    main_canvas.create_window((0,0), window=main_frame, anchor="nw")
    main_frame.bind("<Configure>", lambda e: main_canvas.configure(scrollregion=main_canvas.bbox("all")))
    app = GraficaQQ(main_frame)
    app.pack(fill=tk.BOTH, expand=True)
    root.mainloop()
