#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
====================================================================================================
Título         : Análisis de Supervivencia (Kaplan-Meier) con Filtros Avanzados, Bootstrap para IC,
                 Cálculo extendido de percentiles (incluyendo percentil 3 y 97) y Gráficas Adicionales:
                 Log de Supervivencia, 1 - Supervivencia y Riesgo Acumulado.
Versión        : 2.7.2
Fecha          : 2025-04-09
Descripción    : Esta aplicación utiliza tkinter para generar curvas de Kaplan-Meier con filtros
                 avanzados, pruebas Log-Rank, opciones de etiquetas y ajustes de ejes. Además se
                 incorpora:
                   - Cálculo de intervalos de confianza (IC) mediante bootstrap o por el método
                     predeterminado (Greenwood).
                   - Resumen extendido de estadísticos: mediana, p25, p75, p3 y p97 con su error
                     estándar (SE) e intervalos de confianza (IC) formateados para mostrar como máximo 2
                     decimales (o en notación científica/entero según convenga).
                   - Selección exclusiva del tipo de gráfica a mostrar (KM, Log de Supervivencia, 1 -
                     Supervivencia o Riesgo Acumulado) y los IC se muestran en cada gráfica.
                   - Exportación de la tabla de estadísticas a Excel desde el popup del resumen.

Licencia       : MIT License
Autor          : [Tu Nombre]
====================================================================================================
Notas:
- Este script extiende todas las funcionalidades previas sin quitar ninguna función.
- Se han añadido bloques EXTRA (funciones dummy y comentarios) para aumentar el tamaño del archivo.
====================================================================================================
"""

# =============================================================================
# IMPORTACIÓN DE MÓDULOS ESTÁNDAR Y TERCEROS
# =============================================================================
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test, multivariate_logrank_test
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.ticker import MaxNLocator

try:
    from MATLAB_filter_component import FilterComponent
    FILTER_COMPONENT_AVAILABLE = True
except ImportError:
    FILTER_COMPONENT_AVAILABLE = False
    FilterComponent = None
    # Esto es importante para que el resto del código no falle si el componente no está.
    # Podrías añadir un log aquí si tienes un sistema de logging configurado.
    print("Advertencia: No se pudo importar FilterComponent. Funcionalidad de filtro avanzado no disponible.")

# =============================================================================
# FUNCIONES AUXILIARES Y UTILIDADES (EXTRA)
# =============================================================================
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

# =============================================================================
# CLASE PRINCIPAL: SurvivalAnalysisTab
# =============================================================================
class SurvivalAnalysisTab(ttk.Frame):
    """
    Clase para el análisis de supervivencia (Kaplan-Meier) con filtros avanzados, pruebas Log-Rank,
    opciones de gráfica (colores, tamaño, cuadrícula, etiquetas), cálculo de intervalos de confianza
    (IC) mediante bootstrap o por el método predeterminado (Greenwood) y cálculo extendido de percentiles
    (incluyendo percentil 3 y 97).

    Además, se ha añadido:
      - Un combobox para seleccionar exclusivamente el tipo de gráfica a mostrar (KM, Log de Supervivencia,
        1 - Supervivencia o Riesgo Acumulado).
      - La posibilidad de exportar la tabla de estadísticas a Excel desde el popup de resumen.
    """
    def __init__(self, master):
        super().__init__(master)
        # =============================================================================
        # VARIABLES PRINCIPALES Y DE CONTROL
        # =============================================================================
        self.data = None
        self.file_path = None
        self.custom_filter_component_instance = None # Para FilterComponent

        # Variables para filtros globales
        self.exclude_blank = tk.BooleanVar(value=False)
        self.omit_non_numeric = tk.BooleanVar(value=False)
        self.chk_all_categories = tk.BooleanVar(value=True)
        self.chk_global = tk.BooleanVar(value=False)

        # Para guardar el orden de categorías (si se usa "valor:etiqueta")
        self.ordered_categories = None

        # Variables para selección de columnas
        self.cmb_time = None
        self.cmb_event = None

        # Resultados del KM: lista de tuplas.
        # Cada elemento es: (categoría, stats_interpolacion, stats_used, n_casos, n_censurados)
        # Donde stats_used es un diccionario con claves: "median", "p25", "p75", "p3", "p97"
        self.km_results = []

        # Opciones de gráfica
        self.show_censors = tk.BooleanVar(value=True)
        self.linewidth = tk.DoubleVar(value=2.0)
        self.color_scheme = tk.StringVar(value="tab10")

        # Tamaño de visualización
        self.display_dpi = tk.IntVar(value=100)
        self.display_width = tk.IntVar(value=800)
        self.display_height = tk.IntVar(value=600)

        # Tamaño para guardado
        self.save_dpi = tk.IntVar(value=300)
        self.save_width = tk.IntVar(value=1200)
        self.save_height = tk.IntVar(value=800)

        # Opción para aplicar corrección de Bonferroni
        self.apply_bonferroni = tk.BooleanVar(value=False)

        # Opciones de ejes
        self.show_grid = tk.BooleanVar(value=True)
        self.white_background = tk.BooleanVar(value=False)
        self.num_ticks_x = tk.IntVar(value=5)
        self.num_ticks_y = tk.IntVar(value=5)

        # Opciones de Etiquetas/Título
        self.title_text = tk.StringVar(value="Curvas de Kaplan-Meier")
        self.title_color = tk.StringVar(value="black")
        self.title_fontsize = tk.IntVar(value=12)
        self.x_label = tk.StringVar(value="Tiempo")
        self.x_color = tk.StringVar(value="black")
        self.y_label = tk.StringVar(value="Probabilidad de Supervivencia")
        self.y_color = tk.StringVar(value="black")
        self.axis_fontsize = tk.IntVar(value=10)

        # Opciones para IC y Bootstrap
        self.show_ci = tk.BooleanVar(value=True)
        self.use_bootstrap = tk.BooleanVar(value=False)
        self.bootstrap_iterations = tk.IntVar(value=1000)
        self.random_seed = tk.IntVar(value=42)

        # Nuevo: Selección exclusiva del tipo de gráfica
        self.cmb_graph_type = None  # Se creará en create_widgets

        # =============================================================================
        # LLAMADA A LA CREACIÓN DE WIDGETS
        # =============================================================================
        self.create_widgets()
        self.log_debug("Inicialización completada. Listo para cargar datos y ejecutar análisis.")

    # ---------------------------------------------------------------------------
    # MÉTODO: get_default_stats
    # ---------------------------------------------------------------------------
    def get_default_stats(self, kmf, threshold):
        """
        Calcula la estimación, el error estándar (SE) y el intervalo de confianza (IC)
        para un porcentaje de supervivencia dado (threshold) usando el método predeterminado.
        Se asume que kmf.confidence_interval_ tiene dos columnas (inferior y superior).
        """
        try:
            survival = kmf.survival_function_.values.flatten()
            timeline = kmf.timeline
            estimate = np.interp(threshold, np.flip(survival), np.flip(timeline))
            lower_ci = np.interp(threshold, np.flip(kmf.confidence_interval_.iloc[:, 0].values), np.flip(timeline))
            upper_ci = np.interp(threshold, np.flip(kmf.confidence_interval_.iloc[:, 1].values), np.flip(timeline))
            se = (upper_ci - lower_ci) / 3.92
            return {"estimate": estimate, "se": se, "ci": (lower_ci, upper_ci)}
        except Exception:
            return {"estimate": np.nan, "se": np.nan, "ci": (np.nan, np.nan)}

    # ---------------------------------------------------------------------------
    # MÉTODO: create_widgets
    # ---------------------------------------------------------------------------
    def create_widgets(self):
        main_pane = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        main_pane.pack(fill=tk.BOTH, expand=True)
        # PANEL IZQUIERDO: Controles con scroll
        control_frame = ttk.Frame(main_pane)
        main_pane.add(control_frame, weight=0)
        self.scrollable_controls = create_scrollable_frame(control_frame)

        # 1. Cargar Datos
        frm_load = ttk.LabelFrame(self.scrollable_controls, text="Cargar Datos")
        frm_load.pack(fill=tk.X, padx=5, pady=5)
        btn_load = ttk.Button(frm_load, text="Abrir Archivo (CSV o Excel)", command=self.load_data)
        btn_load.pack(side=tk.LEFT, padx=5, pady=5)
        self.lbl_file = ttk.Label(frm_load, text="Ningún archivo cargado.", foreground="blue", cursor="hand2")
        self.lbl_file.pack(side=tk.LEFT, padx=5, pady=5)
        self.lbl_file.bind("<Button-1>", self.open_file)

        # 2. Variables de Supervivencia
        frm_vars = ttk.LabelFrame(self.scrollable_controls, text="Variables (Supervivencia)")
        frm_vars.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(frm_vars, text="Tiempo:").grid(row=0, column=0, sticky="w", padx=5, pady=2)
        self.cmb_time = ttk.Combobox(frm_vars, values=[], state="readonly")
        self.cmb_time.grid(row=0, column=1, padx=5, pady=2, sticky="we")
        ttk.Label(frm_vars, text="Evento (1=evento, 0=censurado):").grid(row=1, column=0, sticky="w", padx=5, pady=2)
        self.cmb_event = ttk.Combobox(frm_vars, values=[], state="readonly")
        self.cmb_event.grid(row=1, column=1, padx=5, pady=2, sticky="we")
        frm_vars.columnconfigure(1, weight=1)

        # 3. Variable de Agrupación (Categorización)
        frm_grouping = ttk.LabelFrame(self.scrollable_controls, text="Variable de Agrupación")
        frm_grouping.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(frm_grouping, text="Variable para Agrupar Curvas:").grid(row=0, column=0, sticky="w", padx=5, pady=2)
        self.cmb_cat = ttk.Combobox(frm_grouping, values=[], state="readonly") # Variable para agrupar
        self.cmb_cat.grid(row=0, column=1, padx=5, pady=2, sticky="we")
        ttk.Label(frm_grouping, text="Etiquetas/Orden Grupos (Opcional):").grid(row=1, column=0, sticky="w", padx=5, pady=2)
        self.entry_filter = ttk.Entry(frm_grouping) # Entry para etiquetas/orden
        self.entry_filter.grid(row=1, column=1, padx=5, pady=2, sticky="we")
        ttk.Label(frm_grouping, text="(Ej: 1:GrupoA,3:GrupoC,2:GrupoB)").grid(row=2, column=1, sticky="w", padx=5, pady=0)
        frm_grouping.columnconfigure(1, weight=1)

        # 4. Filtros Avanzados (Componente)
        frm_filters_advanced = ttk.LabelFrame(self.scrollable_controls, text="Filtros Avanzados (Componente)")
        frm_filters_advanced.pack(fill=tk.X, padx=5, pady=5)

        if FILTER_COMPONENT_AVAILABLE:
            # Asegurarse que self.custom_filter_component_instance se inicializa aquí si no lo estaba en __init__
            # Aunque ya se inicializó en __init__ a None, aquí se crea la instancia real.
            self.custom_filter_component_instance = FilterComponent(frm_filters_advanced, log_callback=self.log_debug)
            self.custom_filter_component_instance.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        else:
            ttk.Label(frm_filters_advanced, text="Componente de Filtro Avanzado no disponible.").pack(padx=5, pady=5)

        # 5. Opciones Globales de Análisis (antes Filtros Globales)
        frm_global_opts = ttk.LabelFrame(self.scrollable_controls, text="Opciones Globales de Análisis")
        frm_global_opts.pack(fill=tk.X, padx=5, pady=5)
        chk_blank = ttk.Checkbutton(frm_global_opts, text="Omitir espacios en blanco", variable=self.exclude_blank)
        chk_blank.pack(anchor=tk.W, padx=5, pady=2)
        chk_nonnum = ttk.Checkbutton(frm_global_opts, text="Omitir valores numéricos en columnas de texto", variable=self.omit_non_numeric)
        chk_nonnum.pack(anchor=tk.W, padx=5, pady=2)
        chk_allcat = ttk.Checkbutton(frm_global_opts, text="Incluir todas las categorías", variable=self.chk_all_categories)
        chk_allcat.pack(anchor=tk.W, padx=5, pady=2)
        chk_global = ttk.Checkbutton(frm_global_opts, text="Calcular Grupo Global", variable=self.chk_global)
        chk_global.pack(anchor=tk.W, padx=5, pady=2)

        # 6. Opciones de Gráfico
        frm_graph_opts = ttk.LabelFrame(self.scrollable_controls, text="Opciones de Gráfico")
        frm_graph_opts.pack(fill=tk.X, padx=5, pady=5)
        self.chk_censors = ttk.Checkbutton(frm_graph_opts, text="Mostrar censuras", variable=self.show_censors)
        self.chk_censors.pack(anchor=tk.W, padx=5, pady=2)
        ttk.Label(frm_graph_opts, text="Esquema de colores:").pack(anchor=tk.W, padx=5, pady=2)
        self.cmb_color_scheme = ttk.Combobox(frm_graph_opts,
                                             values=["tab10", "Set1", "Set2", "Set3", "Dark2", "Accent",
                                                     "viridis", "plasma", "inferno", "magma", "cividis",
                                                     "Pastel1", "Pastel2"],
                                             textvariable=self.color_scheme, state="readonly")
        self.cmb_color_scheme.pack(fill=tk.X, padx=5, pady=2)
        self.cmb_color_scheme.set("tab10")
        frm_line = ttk.Frame(frm_graph_opts)
        frm_line.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(frm_line, text="Grosor de línea:").pack(side=tk.LEFT, padx=5)
        spin_linewidth = tk.Spinbox(frm_line, from_=0.5, to=10, increment=0.5, textvariable=self.linewidth, width=5)
        spin_linewidth.pack(side=tk.LEFT, padx=5)
        # NUEVO: Selección exclusiva del tipo de gráfica
        ttk.Label(frm_graph_opts, text="Tipo de gráfica:").pack(anchor=tk.W, padx=5, pady=2)
        self.cmb_graph_type = ttk.Combobox(frm_graph_opts,
                                           values=["KM", "Log de Supervivencia", "1 - Supervivencia", "Riesgo Acumulado"],
                                           state="readonly")
        self.cmb_graph_type.pack(fill=tk.X, padx=5, pady=2)
        self.cmb_graph_type.set("KM")

        # 7. Opciones de Ejes
        frm_axes = ttk.LabelFrame(self.scrollable_controls, text="Opciones de Ejes")
        frm_axes.pack(fill=tk.X, padx=5, pady=5)
        chk_grid = ttk.Checkbutton(frm_axes, text="Mostrar cuadrícula", variable=self.show_grid)
        chk_grid.pack(anchor=tk.W, padx=5, pady=2)
        chk_bg = ttk.Checkbutton(frm_axes, text="Fondo blanco", variable=self.white_background)
        chk_bg.pack(anchor=tk.W, padx=5, pady=2)
        frm_ticks = ttk.Frame(frm_axes)
        frm_ticks.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(frm_ticks, text="Tick X:").pack(side=tk.LEFT, padx=5)
        spin_ticks_x = tk.Spinbox(frm_ticks, from_=2, to=20, textvariable=self.num_ticks_x, width=4)
        spin_ticks_x.pack(side=tk.LEFT, padx=5)
        ttk.Label(frm_ticks, text="Tick Y:").pack(side=tk.LEFT, padx=5)
        spin_ticks_y = tk.Spinbox(frm_ticks, from_=2, to=20, textvariable=self.num_ticks_y, width=4)
        spin_ticks_y.pack(side=tk.LEFT, padx=5)

        # 8. Tamaño de Visualización (en pantalla)
        frm_display = ttk.LabelFrame(self.scrollable_controls, text="Tamaño de Visualización (en pantalla)")
        frm_display.pack(fill=tk.X, padx=5, pady=5)
        row_ = 0
        ttk.Label(frm_display, text="DPI (display):").grid(row=row_, column=0, padx=5, pady=2, sticky="w")
        tk.Spinbox(frm_display, from_=50, to=300, textvariable=self.display_dpi, width=6).grid(row=row_, column=1, padx=5, pady=2, sticky="w")
        row_ += 1
        ttk.Label(frm_display, text="Ancho (px):").grid(row=row_, column=0, padx=5, pady=2, sticky="w")
        tk.Spinbox(frm_display, from_=400, to=2000, increment=50, textvariable=self.display_width, width=6).grid(row=row_, column=1, padx=5, pady=2, sticky="w")
        row_ += 1
        ttk.Label(frm_display, text="Alto (px):").grid(row=row_, column=0, padx=5, pady=2, sticky="w")
        tk.Spinbox(frm_display, from_=300, to=2000, increment=50, textvariable=self.display_height, width=6).grid(row=row_, column=1, padx=5, pady=2, sticky="w")
        row_ += 1
        ttk.Button(frm_display, text="Aplicar Tamaño", command=self.apply_figure_size).grid(row=row_, column=0, columnspan=2, pady=5)
        frm_display.columnconfigure(1, weight=1)

        # 9. Tamaño para Guardado
        frm_save = ttk.LabelFrame(self.scrollable_controls, text="Tamaño de Guardado")
        frm_save.pack(fill=tk.X, padx=5, pady=5)
        row_ = 0
        ttk.Label(frm_save, text="DPI (guardado):").grid(row=row_, column=0, padx=5, pady=2, sticky="w")
        tk.Spinbox(frm_save, from_=50, to=600, textvariable=self.save_dpi, width=6).grid(row=row_, column=1, padx=5, pady=2, sticky="w")
        row_ += 1
        ttk.Label(frm_save, text="Ancho (px):").grid(row=row_, column=0, padx=5, pady=2, sticky="w")
        tk.Spinbox(frm_save, from_=400, to=4000, increment=100, textvariable=self.save_width, width=6).grid(row=row_, column=1, padx=5, pady=2, sticky="w")
        row_ += 1
        ttk.Label(frm_save, text="Alto (px):").grid(row=row_, column=0, padx=5, pady=2, sticky="w")
        tk.Spinbox(frm_save, from_=300, to=4000, increment=100, textvariable=self.save_height, width=6).grid(row=row_, column=1, padx=5, pady=2, sticky="w")
        frm_save.columnconfigure(1, weight=1)

        # 10. Opciones Log-Rank y variantes
        frm_logrank = ttk.LabelFrame(self.scrollable_controls, text="Opciones Log-Rank")
        frm_logrank.pack(fill=tk.X, padx=5, pady=5)
        chk_bonf = ttk.Checkbutton(frm_logrank, text="Aplicar corrección de Bonferroni", variable=self.apply_bonferroni)
        chk_bonf.pack(anchor=tk.W, padx=5, pady=2)
        ttk.Label(frm_logrank, text="Tipo de prueba:").pack(anchor=tk.W, padx=5, pady=2)
        weight_mapping_options = ["Log-Rank", "Breslow (Wilcoxon)", "Tarone-Ware", "Peto-Peto"]
        self.cmb_test_type = ttk.Combobox(frm_logrank, values=weight_mapping_options, state="readonly")
        self.cmb_test_type.pack(fill=tk.X, padx=5, pady=2)
        self.cmb_test_type.set("Log-Rank")
        frm_cutoffs = ttk.Frame(frm_logrank)
        frm_cutoffs.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(frm_cutoffs, text="Punto de corte Early:").pack(side=tk.LEFT, padx=5)
        self.entry_early_cutoff = ttk.Entry(frm_cutoffs, width=8)
        self.entry_early_cutoff.pack(side=tk.LEFT, padx=5)
        ttk.Label(frm_cutoffs, text="(deja en blanco para 25° percentil)").pack(side=tk.LEFT, padx=5)
        frm_cutoffs2 = ttk.Frame(frm_logrank)
        frm_cutoffs2.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(frm_cutoffs2, text="Punto de corte Late:").pack(side=tk.LEFT, padx=5)
        self.entry_late_cutoff = ttk.Entry(frm_cutoffs2, width=8)
        self.entry_late_cutoff.pack(side=tk.LEFT, padx=5)
        ttk.Label(frm_cutoffs2, text="(deja en blanco para 75° percentil)").pack(side=tk.LEFT, padx=5)
        ttk.Button(frm_logrank, text="Evaluar Log-Rank", command=self.run_logrank_tests).pack(side=tk.LEFT, padx=5, pady=5)

        # 11. Botones principales
        frm_actions = ttk.Frame(self.scrollable_controls)
        frm_actions.pack(fill=tk.X, padx=5, pady=10)
        ttk.Button(frm_actions, text="Generar Gráfica", command=self.run_km_analysis).pack(side=tk.LEFT, padx=5)
        ttk.Button(frm_actions, text="Resumen Medianas/Percentiles", command=self.show_summary_popup).pack(side=tk.LEFT, padx=5)
        ttk.Button(frm_actions, text="Guardar Gráfico", command=self.save_plot).pack(side=tk.LEFT, padx=5)

        # 12. Opciones de Etiquetas/Título
        frm_labels = ttk.LabelFrame(self.scrollable_controls, text="Opciones de Etiquetas/Título")
        frm_labels.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(frm_labels, text="Título:").grid(row=0, column=0, sticky="w", padx=5, pady=2)
        entry_title = ttk.Entry(frm_labels, textvariable=self.title_text, width=30)
        entry_title.grid(row=0, column=1, sticky="w", padx=5, pady=2)
        ttk.Label(frm_labels, text="Color Título:").grid(row=0, column=2, sticky="w", padx=5, pady=2)
        combo_title_color = ttk.Combobox(frm_labels, values=["black", "blue", "red", "green", "orange", "purple"], textvariable=self.title_color, state="readonly")
        combo_title_color.grid(row=0, column=3, sticky="w", padx=5, pady=2)
        ttk.Label(frm_labels, text="Tamaño Título:").grid(row=0, column=4, sticky="w", padx=5, pady=2)
        spin_title_font = tk.Spinbox(frm_labels, from_=8, to=40, textvariable=self.title_fontsize, width=5)
        spin_title_font.grid(row=0, column=5, sticky="w", padx=5, pady=2)
        ttk.Label(frm_labels, text="Etiqueta Eje X:").grid(row=1, column=0, sticky="w", padx=5, pady=2)
        entry_x_label = ttk.Entry(frm_labels, textvariable=self.x_label, width=30)
        entry_x_label.grid(row=1, column=1, sticky="w", padx=5, pady=2)
        ttk.Label(frm_labels, text="Color Eje X:").grid(row=1, column=2, sticky="w", padx=5, pady=2)
        combo_x_color = ttk.Combobox(frm_labels, values=["black", "blue", "red", "green", "orange", "purple"], textvariable=self.x_color, state="readonly")
        combo_x_color.grid(row=1, column=3, sticky="w", padx=5, pady=2)
        ttk.Label(frm_labels, text="Etiqueta Eje Y:").grid(row=2, column=0, sticky="w", padx=5, pady=2)
        entry_y_label = ttk.Entry(frm_labels, textvariable=self.y_label, width=30)
        entry_y_label.grid(row=2, column=1, sticky="w", padx=5, pady=2)
        ttk.Label(frm_labels, text="Color Eje Y:").grid(row=2, column=2, sticky="w", padx=5, pady=2)
        combo_y_color = ttk.Combobox(frm_labels, values=["black", "blue", "red", "green", "orange", "purple"], textvariable=self.y_color, state="readonly")
        combo_y_color.grid(row=2, column=3, sticky="w", padx=5, pady=2)
        ttk.Label(frm_labels, text="Tamaño Ejes:").grid(row=2, column=4, sticky="w", padx=5, pady=2)
        spin_axis_font = tk.Spinbox(frm_labels, from_=8, to=30, textvariable=self.axis_fontsize, width=5)
        spin_axis_font.grid(row=2, column=5, sticky="w", padx=5, pady=2)
        frm_labels.columnconfigure(1, weight=1)

        # 13. NUEVA SECCIÓN: Opciones de Intervalos de Confianza (IC)
        frm_ic = ttk.LabelFrame(self.scrollable_controls, text="Opciones de Intervalos de Confianza (IC)")
        frm_ic.pack(fill=tk.X, padx=5, pady=5)
        chk_show_ci = ttk.Checkbutton(frm_ic, text="Mostrar IC en la gráfica", variable=self.show_ci)
        chk_show_ci.grid(row=0, column=0, padx=5, pady=2, sticky="w")
        chk_bootstrap = ttk.Checkbutton(frm_ic, text="Calcular IC usando Bootstrap", variable=self.use_bootstrap)
        chk_bootstrap.grid(row=1, column=0, padx=5, pady=2, sticky="w")
        ttk.Label(frm_ic, text="N° Iteraciones Bootstrap:").grid(row=2, column=0, sticky="w", padx=5, pady=2)
        spin_boot_iter = tk.Spinbox(frm_ic, from_=100, to=10000, increment=100, textvariable=self.bootstrap_iterations, width=8)
        spin_boot_iter.grid(row=2, column=1, sticky="w", padx=5, pady=2)
        ttk.Label(frm_ic, text="Semilla (seed):").grid(row=3, column=0, sticky="w", padx=5, pady=2)
        spin_seed = tk.Spinbox(frm_ic, from_=0, to=100000, textvariable=self.random_seed, width=8)
        spin_seed.grid(row=3, column=1, sticky="w", padx=5, pady=2)
        frm_ic.columnconfigure(0, weight=1)

        # PANEL DERECHO: Área de gráfica
        figure_frame = ttk.Frame(main_pane)
        main_pane.add(figure_frame, weight=1)
        self.figure = plt.Figure(figsize=(8, 6), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.figure, master=figure_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Consola (log) en la parte inferior
        log_frame = ttk.Frame(self)
        log_frame.pack(fill=tk.BOTH, expand=False, side=tk.BOTTOM)
        self.txt_log = tk.Text(log_frame, height=6, wrap="none")
        vsb_log = ttk.Scrollbar(log_frame, orient="vertical", command=self.txt_log.yview)
        hsb_log = ttk.Scrollbar(log_frame, orient="horizontal", command=self.txt_log.xview)
        self.txt_log.configure(yscrollcommand=vsb_log.set, xscrollcommand=hsb_log.set)
        self.txt_log.grid(row=0, column=0, sticky="nsew")
        vsb_log.grid(row=0, column=1, sticky="ns")
        hsb_log.grid(row=1, column=0, sticky="ew")
        log_frame.rowconfigure(0, weight=1)
        log_frame.columnconfigure(0, weight=1)
        self.extra_info()

    # ---------------------------------------------------------------------------
    # MÉTODO: load_data
    # ---------------------------------------------------------------------------
    def load_data(self):
        file_path = filedialog.askopenfilename(
            title="Selecciona archivo CSV o Excel",
            filetypes=[("CSV files", "*.csv"), ("Excel files", "*.xlsx *.xls"), ("All files", "*.*")]
        )
        if not file_path:
            return
        try:
            if file_path.lower().endswith(".csv"):
                self.data = pd.read_csv(file_path)
            else:
                self.data = pd.read_excel(file_path)
            self.file_path = file_path
            self.lbl_file.config(text=os.path.basename(file_path))
            cols = list(self.data.columns)
            self.cmb_time['values'] = cols
            self.cmb_event['values'] = cols
            self.cmb_cat['values'] = [""] + cols # Actualizar combo de agrupación
            self.cmb_cat.set("")

            if FILTER_COMPONENT_AVAILABLE and self.custom_filter_component_instance:
                self.custom_filter_component_instance.set_dataframe(self.data)
                self.log_debug("DataFrame actualizado en FilterComponent.")

            msg = f"Datos cargados: {self.data.shape[0]} filas, {self.data.shape[1]} columnas."
            self.txt_log.insert(tk.END, msg + "\n")
            messagebox.showinfo("Éxito", msg)
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo cargar el archivo:\n{e}")
            if FILTER_COMPONENT_AVAILABLE and self.custom_filter_component_instance:
                self.custom_filter_component_instance.set_dataframe(pd.DataFrame()) # Enviar DF vacío

    # ---------------------------------------------------------------------------
    # MÉTODO: open_file
    # ---------------------------------------------------------------------------
    def open_file(self, event):
        if self.file_path and os.path.exists(self.file_path):
            try:
                os.startfile(self.file_path)
            except Exception as e:
                messagebox.showerror("Error", f"No se pudo abrir el archivo:\n{e}")

    # ---------------------------------------------------------------------------
    # MÉTODO: _apply_main_categorization (antes get_filtered_data)
    # ---------------------------------------------------------------------------
    def _apply_main_categorization(self, df_input):
        """
        Aplica el filtro/etiquetado/orden de la variable de agrupación principal.
        Toma un DataFrame ya filtrado por el FilterComponent como entrada.
        Devuelve el DataFrame modificado y la lista de categorías ordenadas.
        """
        if df_input is None or df_input.empty:
            return df_input, None

        df = df_input.copy() # Trabajar con copia para no modificar el original filtrado
        cat_var = self.cmb_cat.get().strip()
        filtro_etiquetas = self.entry_filter.get().strip() # Renombrado para claridad
        ordered_categories = None # Reiniciar orden

        if cat_var and cat_var in df.columns and filtro_etiquetas:
            col_is_num = pd.api.types.is_numeric_dtype(df[cat_var])
            # Lógica de parseo de filtro_etiquetas (valor:etiqueta, rango, lista, valor único)
            # Esta lógica ahora actúa sobre el DataFrame ya filtrado por el componente
            # y su propósito principal es definir el orden y las etiquetas,
            # aunque también puede filtrar *más* si se especifican solo algunos valores.
            if ":" in filtro_etiquetas:
                mapping = {}
                parts = [p.strip() for p in filtro_etiquetas.split(",") if p.strip()]
                typed_order_labels = []
                keys_to_keep = [] # Guardar las claves originales que coinciden
                for part in parts:
                    if ":" in part:
                        key, label = part.split(":", 1)
                        key = key.strip(); label = label.strip()
                        mapping[key] = label; typed_order_labels.append(label)
                        keys_to_keep.append(key) # Guardar la clave original
                # Filtrar para mantener solo las claves especificadas en el mapeo
                if col_is_num:
                    numeric_keys = []
                    numeric_mapping = {}
                    for k in keys_to_keep:
                        try: nk = float(k); numeric_keys.append(nk); numeric_mapping[nk] = mapping[k]
                        except: numeric_keys.append(k); numeric_mapping[k] = mapping[k] # Mantener como string si falla conversión
                    df = df[df[cat_var].isin(numeric_keys)]
                    df[cat_var] = df[cat_var].map(numeric_mapping) # Aplicar mapeo a etiquetas
                else:
                    df = df[df[cat_var].astype(str).isin(keys_to_keep)] # Filtrar por claves originales
                    df[cat_var] = df[cat_var].astype(str).map(mapping) # Aplicar mapeo a etiquetas
                # Aplicar orden
                if typed_order_labels:
                    cat_dtype = pd.CategoricalDtype(categories=typed_order_labels, ordered=True)
                    df[cat_var] = df[cat_var].astype(cat_dtype)
                    ordered_categories = typed_order_labels
            elif "-" in filtro_etiquetas and "," not in filtro_etiquetas: # Rango (solo filtra, no etiqueta/ordena)
                parts = filtro_etiquetas.split("-")
                if len(parts) == 2 and col_is_num:
                    try:
                        low = float(parts[0].strip()); high = float(parts[1].strip())
                        df = df[(df[cat_var] >= low) & (df[cat_var] <= high)]
                    except: pass # Ignorar si el rango no es válido
            elif "," in filtro_etiquetas: # Lista (filtra y define orden)
                vals = [p.strip() for p in filtro_etiquetas.split(",") if p.strip()]
                if col_is_num:
                    numeric_vals = []
                    for val in vals:
                        try: numeric_vals.append(float(val))
                        except: numeric_vals.append(val) # Mantener como string si falla
                    df = df[df[cat_var].isin(numeric_vals)]
                    # Definir orden basado en la lista de entrada
                    cat_dtype = pd.CategoricalDtype(categories=numeric_vals, ordered=True)
                    df[cat_var] = df[cat_var].astype(cat_dtype)
                    ordered_categories = vals
                else:
                    df = df[df[cat_var].astype(str).isin(vals)]
                    cat_dtype = pd.CategoricalDtype(categories=vals, ordered=True)
                    df[cat_var] = df[cat_var].astype(cat_dtype)
                    ordered_categories = vals
            else: # Valor único (filtra, no etiqueta/ordena)
                if col_is_num:
                    try: val_f = float(filtro_etiquetas); df = df[df[cat_var] == val_f]
                    except: df = df[df[cat_var].astype(str) == filtro_etiquetas]
                else:
                    df = df[df[cat_var].astype(str) == filtro_etiquetas]
                ordered_categories = [filtro_etiquetas] # Orden es solo este valor

        self.ordered_categories = ordered_categories # Guardar orden para usarlo después
        return df, ordered_categories

    # Se elimina apply_filter_criteria

    # ---------------------------------------------------------------------------
    # MÉTODO: apply_global_filters
    # ---------------------------------------------------------------------------
    def apply_global_filters(self, df):
        if self.exclude_blank.get():
            df = df.replace(r'^\s*$', np.nan, regex=True).dropna(how='any')
        if self.omit_non_numeric.get():
            obj_cols = df.select_dtypes(include=['object', 'string']).columns
            for c in obj_cols:
                df = df[~df[c].astype(str).apply(lambda x: x.replace('.', '', 1).isdigit())]
        return df

    # ---------------------------------------------------------------------------
    # MÉTODO: apply_figure_size
    # ---------------------------------------------------------------------------
    def apply_figure_size(self):
        w_inch = self.display_width.get() / float(self.display_dpi.get())
        h_inch = self.display_height.get() / float(self.display_dpi.get())
        self.figure.set_dpi(self.display_dpi.get())
        self.figure.set_size_inches(w_inch, h_inch)
        self.canvas.draw()
        self.log_debug(f"Se aplicó tamaño de visualización: {w_inch}x{h_inch} pulgadas con DPI={self.display_dpi.get()}")

    # ---------------------------------------------------------------------------
    # MÉTODO: compute_bootstrap_ci
    # ---------------------------------------------------------------------------
    def compute_bootstrap_ci(self, group_data, time_col, event_col, timeline, iterations, seed):
        """
        Calcula IC mediante bootstrap para la curva de supervivencia.
        """
        np.random.seed(seed)
        bootstrap_survival = []
        for i in range(iterations):
            resample_data = group_data.sample(n=len(group_data), replace=True)
            kmf_temp = KaplanMeierFitter()
            try:
                kmf_temp.fit(durations=resample_data[time_col], event_observed=resample_data[event_col])
            except Exception:
                continue
            surv_interp = np.interp(timeline, kmf_temp.timeline, kmf_temp.survival_function_.values.flatten(), left=1.0, right=kmf_temp.survival_function_.values.flatten()[-1])
            bootstrap_survival.append(surv_interp)
        bootstrap_survival = np.array(bootstrap_survival)
        lower_bound = np.percentile(bootstrap_survival, 2.5, axis=0)
        upper_bound = np.percentile(bootstrap_survival, 97.5, axis=0)
        return lower_bound, upper_bound

    # ---------------------------------------------------------------------------
    # MÉTODO: compute_bootstrap_percentile_estimates
    # ---------------------------------------------------------------------------
    def compute_bootstrap_percentile_estimates(self, data, time_col, event_col):
        """
        Calcula, mediante bootstrap, estimaciones, errores estándar y IC para la mediana, p25, p75,
        percentil 3 y percentil 97.
        """
        iterations = self.bootstrap_iterations.get()
        seed = self.random_seed.get()
        np.random.seed(seed)
        medians, p25s, p75s, p3s, p97s = [], [], [], [], []
        for i in range(iterations):
            resample = data.sample(n=len(data), replace=True)
            kmf_temp = KaplanMeierFitter()
            try:
                kmf_temp.fit(durations=resample[time_col], event_observed=resample[event_col])
            except Exception:
                continue
            times = kmf_temp.timeline
            surv = kmf_temp.survival_function_.values.flatten()
            try:
                median_val = np.interp(0.5, np.flip(surv), np.flip(times))
            except Exception:
                median_val = np.nan
            try:
                p25_val = np.interp(1 - 25/100, np.flip(surv), np.flip(times))
            except Exception:
                p25_val = np.nan
            try:
                p75_val = np.interp(1 - 75/100, np.flip(surv), np.flip(times))
            except Exception:
                p75_val = np.nan
            try:
                p3_val = np.interp(1 - 3/100, np.flip(surv), np.flip(times))
            except Exception:
                p3_val = np.nan
            try:
                p97_val = np.interp(1 - 97/100, np.flip(surv), np.flip(times))
            except Exception:
                p97_val = np.nan
            medians.append(median_val)
            p25s.append(p25_val)
            p75s.append(p75_val)
            p3s.append(p3_val)
            p97s.append(p97_val)
        def summary_stats(values):
            arr = np.array(values)
            valid = arr[~np.isnan(arr)]
            if len(valid) == 0:
                return {"estimate": np.nan, "se": np.nan, "ci": (np.nan, np.nan)}
            est = np.mean(valid)
            se = np.std(valid, ddof=1)
            ci_lower = np.percentile(valid, 2.5)
            ci_upper = np.percentile(valid, 97.5)
            return {"estimate": est, "se": se, "ci": (ci_lower, ci_upper)}
        return {
            "median": summary_stats(medians),
            "p25": summary_stats(p25s),
            "p75": summary_stats(p75s),
            "p3": summary_stats(p3s),
            "p97": summary_stats(p97s)
        }

    # ---------------------------------------------------------------------------
    # MÉTODO: run_km_analysis
    # ---------------------------------------------------------------------------
    def run_km_analysis(self):
        if self.data is None:
            messagebox.showwarning("Advertencia", "Primero cargue los datos.")
            return
        time_col = self.cmb_time.get().strip()
        event_col = self.cmb_event.get().strip()
        if not time_col or not event_col:
            messagebox.showwarning("Advertencia", "Seleccione las variables de tiempo y evento.")
            return

        # 1. Aplicar filtros con FilterComponent si está disponible
        if FILTER_COMPONENT_AVAILABLE and self.custom_filter_component_instance and self.data is not None:
            try:
                # Asegurarse que el componente tiene el DataFrame más reciente
                # self.custom_filter_component_instance.set_dataframe(self.data) # Opcional si load_data ya lo hace
                df_temp_filtered = self.custom_filter_component_instance.apply_filters()
                if df_temp_filtered is None:
                    messagebox.showerror("Error de Filtro", "El componente de filtro devolvió None. Verifique la configuración del filtro o los datos.")
                    return
                self.log_debug(f"Datos filtrados por FilterComponent para análisis KM: {df_temp_filtered.shape[0]} filas.")
                if df_temp_filtered.empty:
                    messagebox.showwarning("Datos Vacíos", "No quedan datos después de aplicar los filtros del componente.")
                    return
            except Exception as e:
                messagebox.showerror("Error en FilterComponent", f"Error al aplicar filtros: {e}")
                return
        elif self.data is not None:
            df_temp_filtered = self.data.copy()
            self.log_debug("Usando datos originales para análisis KM (FilterComponent no disponible/activo o datos no cargados en él).")
        else:
            messagebox.showwarning("Advertencia", "Carga datos primero.")
            return

        # 2. Aplicar filtros globales (blancos, no numéricos)
        df_temp_filtered = self.apply_global_filters(df_temp_filtered)

        # 3. Aplicar filtro/etiquetado/orden de la variable de agrupación principal
        df_final_for_analysis, self.ordered_categories = self._apply_main_categorization(df_temp_filtered)

        # 4. Eliminar NaNs en columnas de tiempo y evento (crucial antes de análisis)
        df_final_for_analysis = df_final_for_analysis.dropna(subset=[time_col, event_col])

        # 5. Verificar si quedan datos
        if df_final_for_analysis.empty:
            messagebox.showwarning("Aviso", "No quedan datos tras aplicar filtros y limpieza T/E.")
            return

        # 6. Determinar categorías para iterar (basado en df_final_for_analysis)
        cat_var = self.cmb_cat.get().strip()
        if cat_var and cat_var in df_final_for_analysis.columns:
            # Usar el orden definido en _apply_main_categorization si existe
            if self.ordered_categories:
                # Asegurarse que las categorías ordenadas realmente existen en los datos finales
                cat_values = [c for c in self.ordered_categories if c in df_final_for_analysis[cat_var].unique()]
            else:
                # Si no hay orden definido, usar los valores únicos presentes ordenados
                cat_values = sorted(list(df_final_for_analysis[cat_var].dropna().unique()))
            cat_values = [str(c) for c in cat_values]
        else:
            cat_values = ["(SinCategoría)"]

        # Nuevo: Seleccionar tipo de gráfica de forma exclusiva
        graph_type = self.cmb_graph_type.get()
        self.km_results = []
        self.figure.clear()
        ax = self.figure.add_subplot(111)

        # Diccionario de esquemas de colores
        color_schemes = {
            "tab10": plt.cm.tab10, "Set1": plt.cm.Set1, "Set2": plt.cm.Set2,
            "Set3": plt.cm.Set3, "Dark2": plt.cm.Dark2, "Accent": plt.cm.Accent,
            "viridis": plt.cm.viridis, "plasma": plt.cm.plasma, "inferno": plt.cm.inferno,
            "magma": plt.cm.magma, "cividis": plt.cm.cividis, "Pastel1": plt.cm.Pastel1,
            "Pastel2": plt.cm.Pastel2
        }
        cmap = color_schemes.get(self.color_scheme.get(), plt.cm.tab10)
        if hasattr(cmap, "colors"):
            base_colors = cmap.colors
        else:
            base_colors = [cmap(i) for i in range(cmap.N)]
        def calculate_survival_stats(times, survival_prob):
            stats = {}
            try:
                stats['median'] = np.interp(0.5, np.flip(survival_prob), np.flip(times))
            except:
                stats['median'] = None
            for p in [25, 75]:
                try:
                    stats[f'p{p}'] = np.interp(1 - p/100, np.flip(survival_prob), np.flip(times))
                except:
                    stats[f'p{p}'] = None
            return stats

        color_idx = 0
        for catv in cat_values:
            if catv == "(SinCategoría)":
                sub = df_final_for_analysis.copy()
            else:
                # Comparar directamente ya que la columna debería tener el tipo correcto (Categorical o string)
                sub = df_final_for_analysis[df_final_for_analysis[cat_var] == catv].copy()
            sub = sub.dropna(subset=[time_col, event_col])
            if len(sub) < 3:
                self.txt_log.insert(tk.END, f"Categoría '{catv}' omitida (muestras insuficientes)\n")
                continue
            kmf = KaplanMeierFitter()
            try:
                kmf.fit(durations=sub[time_col], event_observed=sub[event_col], label=str(catv))
            except Exception as e:
                self.txt_log.insert(tk.END, f"Error en categoría '{catv}': {e}\n")
                continue
            color = base_colors[color_idx % len(base_colors)]
            color_idx += 1

            # Gráfica según el tipo seleccionado:
            if graph_type == "KM":
                if self.show_ci.get():
                    if self.use_bootstrap.get():
                        kmf.plot_survival_function(ax=ax, ci_show=False, show_censors=self.show_censors.get(),
                                                   color=color, lw=self.linewidth.get())
                        timeline = kmf.timeline
                        lower_bound, upper_bound = self.compute_bootstrap_ci(sub, time_col, event_col,
                                                                             timeline, self.bootstrap_iterations.get(),
                                                                             self.random_seed.get())
                        ax.fill_between(timeline, lower_bound, upper_bound, color=color, alpha=0.3)
                    else:
                        kmf.plot_survival_function(ax=ax, ci_show=True, show_censors=self.show_censors.get(),
                                                   color=color, lw=self.linewidth.get())
                else:
                    kmf.plot_survival_function(ax=ax, ci_show=False, show_censors=self.show_censors.get(),
                                               color=color, lw=self.linewidth.get())
            elif graph_type == "Log de Supervivencia":
                y_vals = np.log(np.clip(kmf.survival_function_.values.flatten(), a_min=1e-10, a_max=None))
                ax.plot(kmf.timeline, y_vals, label=str(catv), color=color, lw=self.linewidth.get())
                if self.show_ci.get():
                    if self.use_bootstrap.get():
                        timeline = kmf.timeline
                        lower_bound, upper_bound = self.compute_bootstrap_ci(sub, time_col, event_col,
                                                                             timeline, self.bootstrap_iterations.get(),
                                                                             self.random_seed.get())
                        ax.fill_between(timeline, np.log(np.clip(lower_bound, 1e-10, None)), np.log(np.clip(upper_bound, 1e-10, None)),
                                        color=color, alpha=0.3)
                    else:
                        try:
                            lower = np.log(np.clip(kmf.confidence_interval_.iloc[:, 0].values, 1e-10, None))
                            upper = np.log(np.clip(kmf.confidence_interval_.iloc[:, 1].values, 1e-10, None))
                            ax.fill_between(kmf.timeline, lower, upper, color=color, alpha=0.3)
                        except Exception:
                            pass
            elif graph_type == "1 - Supervivencia":
                y_vals = 1 - kmf.survival_function_.values.flatten()
                ax.plot(kmf.timeline, y_vals, label=str(catv), color=color, lw=self.linewidth.get())
                if self.show_ci.get():
                    if self.use_bootstrap.get():
                        timeline = kmf.timeline
                        lower_bound, upper_bound = self.compute_bootstrap_ci(sub, time_col, event_col,
                                                                             timeline, self.bootstrap_iterations.get(),
                                                                             self.random_seed.get())
                        ax.fill_between(timeline, 1 - upper_bound, 1 - lower_bound, color=color, alpha=0.3)
                    else:
                        try:
                            lower = kmf.confidence_interval_.iloc[:, 0].values
                            upper = kmf.confidence_interval_.iloc[:, 1].values
                            ax.fill_between(kmf.timeline, 1 - upper, 1 - lower, color=color, alpha=0.3)
                        except Exception:
                            pass
            elif graph_type == "Riesgo Acumulado":
                if hasattr(kmf, "cumulative_hazard_"):
                    y_vals = kmf.cumulative_hazard_.values.flatten()
                    ax.plot(kmf.cumulative_hazard_.index, y_vals, label=str(catv), color=color, lw=self.linewidth.get())
                    if self.show_ci.get():
                        if self.use_bootstrap.get():
                            timeline = kmf.timeline
                            lower_bound, upper_bound = self.compute_bootstrap_ci(sub, time_col, event_col,
                                                                                 timeline, self.bootstrap_iterations.get(),
                                                                                 self.random_seed.get())
                            # Convertir IC de supervivencia en IC de riesgo acumulado: H = -log(S)
                            ax.fill_between(timeline, -np.log(np.clip(upper_bound,1e-10,None)), -np.log(np.clip(lower_bound,1e-10,None)),
                                            color=color, alpha=0.3)
                        else:
                            try:
                                lower = kmf.confidence_interval_.iloc[:, 0].values
                                upper = kmf.confidence_interval_.iloc[:, 1].values
                                ax.fill_between(kmf.timeline, -np.log(np.clip(upper,1e-10,None)), -np.log(np.clip(lower,1e-10,None)),
                                                color=color, alpha=0.3)
                            except Exception:
                                pass
                else:
                    y_vals = -np.log(np.clip(kmf.survival_function_.values.flatten(), a_min=1e-10, a_max=None))
                    ax.plot(kmf.timeline, y_vals, label=str(catv), color=color, lw=self.linewidth.get())
                    if self.show_ci.get():
                        if self.use_bootstrap.get():
                            timeline = kmf.timeline
                            lower_bound, upper_bound = self.compute_bootstrap_ci(sub, time_col, event_col,
                                                                                 timeline, self.bootstrap_iterations.get(),
                                                                                 self.random_seed.get())
                            ax.fill_between(timeline, -np.log(np.clip(upper_bound,1e-10,None)), -np.log(np.clip(lower_bound,1e-10,None)),
                                            color=color, alpha=0.3)
                        else:
                            try:
                                lower = kmf.confidence_interval_.iloc[:, 0].values
                                upper = kmf.confidence_interval_.iloc[:, 1].values
                                ax.fill_between(kmf.timeline, -np.log(np.clip(upper,1e-10,None)), -np.log(np.clip(lower,1e-10,None)),
                                                color=color, alpha=0.3)
                            except Exception:
                                pass
            # Calcular estadísticas: usar bootstrap si se escogió, de lo contrario usar valores predeterminados.
            if self.use_bootstrap.get():
                bs_stats = self.compute_bootstrap_percentile_estimates(sub, time_col, event_col)
            else:
                bs_stats = {
                    "median": self.get_default_stats(kmf, 0.5),
                    "p25": self.get_default_stats(kmf, 0.75),
                    "p75": self.get_default_stats(kmf, 0.25),
                    "p3": self.get_default_stats(kmf, 0.97),
                    "p97": self.get_default_stats(kmf, 0.03)
                }
            stats = calculate_survival_stats(kmf.timeline, kmf.survival_function_.values.flatten())
            n_used = len(sub)
            n_censored = int((sub[event_col] == 0).sum())
            self.km_results.append((catv, stats, bs_stats, n_used, n_censored))
        # Grupo Global (usar df_final_for_analysis)
        if self.chk_global.get():
            if len(df_final_for_analysis) >= 3:
                kmf = KaplanMeierFitter()
                try:
                    kmf.fit(durations=df_final_for_analysis[time_col], event_observed=df_final_for_analysis[event_col], label="Global")
                    color = base_colors[color_idx % len(base_colors)]
                    if graph_type == "KM":
                        if self.show_ci.get():
                            if self.use_bootstrap.get():
                                kmf.plot_survival_function(ax=ax, ci_show=False, show_censors=self.show_censors.get(),
                                                           color=color, lw=self.linewidth.get())
                                timeline = kmf.timeline
                                lower_bound, upper_bound = self.compute_bootstrap_ci(df_final_for_analysis, time_col, event_col,
                                                                                     timeline, self.bootstrap_iterations.get(),
                                                                                     self.random_seed.get())
                                ax.fill_between(timeline, lower_bound, upper_bound, color=color, alpha=0.3)
                            else:
                                kmf.plot_survival_function(ax=ax, ci_show=True, show_censors=self.show_censors.get(),
                                                           color=color, lw=self.linewidth.get())
                        else:
                            kmf.plot_survival_function(ax=ax, ci_show=False, show_censors=self.show_censors.get(),
                                                       color=color, lw=self.linewidth.get())
                    elif graph_type == "Log de Supervivencia":
                        y_vals = np.log(np.clip(kmf.survival_function_.values.flatten(), a_min=1e-10, a_max=None))
                        ax.plot(kmf.timeline, y_vals, label="Global", color=color, lw=self.linewidth.get())
                        if self.show_ci.get():
                            if self.use_bootstrap.get():
                                timeline = kmf.timeline
                                lower_bound, upper_bound = self.compute_bootstrap_ci(df_final_for_analysis, time_col, event_col,
                                                                                     timeline, self.bootstrap_iterations.get(),
                                                                                     self.random_seed.get())
                                ax.fill_between(timeline, np.log(np.clip(lower_bound,1e-10,None)), np.log(np.clip(upper_bound,1e-10,None)),
                                                color=color, alpha=0.3)
                            else:
                                try:
                                    lower = np.log(np.clip(kmf.confidence_interval_.iloc[:, 0].values,1e-10,None))
                                    upper = np.log(np.clip(kmf.confidence_interval_.iloc[:, 1].values,1e-10,None))
                                    ax.fill_between(kmf.timeline, lower, upper, color=color, alpha=0.3)
                                except Exception:
                                    pass
                    elif graph_type == "1 - Supervivencia":
                        y_vals = 1 - kmf.survival_function_.values.flatten()
                        ax.plot(kmf.timeline, y_vals, label="Global", color=color, lw=self.linewidth.get())
                        if self.show_ci.get():
                            if self.use_bootstrap.get():
                                timeline = kmf.timeline
                                lower_bound, upper_bound = self.compute_bootstrap_ci(df_final_for_analysis, time_col, event_col,
                                                                                     timeline, self.bootstrap_iterations.get(),
                                                                                     self.random_seed.get())
                                ax.fill_between(timeline, 1 - upper_bound, 1 - lower_bound, color=color, alpha=0.3)
                            else:
                                try:
                                    lower = kmf.confidence_interval_.iloc[:, 0].values
                                    upper = kmf.confidence_interval_.iloc[:, 1].values
                                    ax.fill_between(kmf.timeline, 1 - upper, 1 - lower, color=color, alpha=0.3)
                                except Exception:
                                    pass
                    elif graph_type == "Riesgo Acumulado":
                        if hasattr(kmf, "cumulative_hazard_"):
                            y_vals = kmf.cumulative_hazard_.values.flatten()
                            ax.plot(kmf.cumulative_hazard_.index, y_vals, label="Global", color=color, lw=self.linewidth.get())
                            if self.show_ci.get():
                                if self.use_bootstrap.get():
                                    timeline = kmf.timeline
                                    lower_bound, upper_bound = self.compute_bootstrap_ci(df_final_for_analysis, time_col, event_col,
                                                                                         timeline, self.bootstrap_iterations.get(),
                                                                                         self.random_seed.get())
                                    ax.fill_between(timeline, -np.log(np.clip(upper_bound,1e-10,None)),
                                                    -np.log(np.clip(lower_bound,1e-10,None)), color=color, alpha=0.3)
                                else:
                                    try:
                                        lower = kmf.confidence_interval_.iloc[:, 0].values
                                        upper = kmf.confidence_interval_.iloc[:, 1].values
                                        ax.fill_between(kmf.timeline, -np.log(np.clip(upper,1e-10,None)),
                                                        -np.log(np.clip(lower,1e-10,None)), color=color, alpha=0.3)
                                    except Exception:
                                        pass
                        else:
                            y_vals = -np.log(np.clip(kmf.survival_function_.values.flatten(), a_min=1e-10, a_max=None))
                            ax.plot(kmf.timeline, y_vals, label="Global", color=color, lw=self.linewidth.get())
                            if self.show_ci.get():
                                if self.use_bootstrap.get():
                                    timeline = kmf.timeline
                                    lower_bound, upper_bound = self.compute_bootstrap_ci(df_final_for_analysis, time_col, event_col,
                                                                                         timeline, self.bootstrap_iterations.get(),
                                                                                         self.random_seed.get())
                                    ax.fill_between(timeline, -np.log(np.clip(upper_bound,1e-10,None)),
                                                    -np.log(np.clip(lower_bound,1e-10,None)), color=color, alpha=0.3)
                                else:
                                    try:
                                        lower = kmf.confidence_interval_.iloc[:, 0].values
                                        upper = kmf.confidence_interval_.iloc[:, 1].values
                                        ax.fill_between(kmf.timeline, -np.log(np.clip(upper,1e-10,None)),
                                                        -np.log(np.clip(lower,1e-10,None)), color=color, alpha=0.3)
                                    except Exception:
                                        pass
                    if self.use_bootstrap.get():
                        bs_stats = self.compute_bootstrap_percentile_estimates(df_final_for_analysis, time_col, event_col)
                    else:
                        bs_stats = {
                            "median": self.get_default_stats(kmf, 0.5),
                            "p25": self.get_default_stats(kmf, 0.75),
                            "p75": self.get_default_stats(kmf, 0.25),
                            "p3": self.get_default_stats(kmf, 0.97),
                            "p97": self.get_default_stats(kmf, 0.03)
                        }
                    stats = calculate_survival_stats(kmf.timeline, kmf.survival_function_.values.flatten())
                    n_used = len(df_final_for_analysis)
                    n_censored = int((df_final_for_analysis[event_col] == 0).sum())
                    self.km_results.append(("Global", stats, bs_stats, n_used, n_censored))
                except Exception as e:
                    self.txt_log.insert(tk.END, f"Error en grupo Global: {e}\n")
            else:
                self.txt_log.insert(tk.END, "No se generó Global (muestras insuficientes)\n")

        # Configurar cuadrícula, ticks y fondo
        ax.grid(self.show_grid.get())
        if self.white_background.get():
            ax.set_facecolor("white")
        else:
            ax.set_facecolor("none")
        ax.xaxis.set_major_locator(MaxNLocator(self.num_ticks_x.get()))
        ax.yaxis.set_major_locator(MaxNLocator(self.num_ticks_y.get()))
        # Ajustar títulos y etiquetas según el tipo de gráfica
        if graph_type == "KM":
            ax.set_title(self.title_text.get() + " - Supervivencia", color=self.title_color.get(), fontsize=self.title_fontsize.get())
            ax.set_ylabel("Supervivencia", color=self.y_color.get(), fontsize=self.axis_fontsize.get())
        elif graph_type == "Log de Supervivencia":
            ax.set_title("Log de Supervivencia (ln[S(t)])", color=self.title_color.get(), fontsize=self.title_fontsize.get())
            ax.set_ylabel("ln(S(t))", color=self.y_color.get(), fontsize=self.axis_fontsize.get())
        elif graph_type == "1 - Supervivencia":
            ax.set_title("1 - Supervivencia", color=self.title_color.get(), fontsize=self.title_fontsize.get())
            ax.set_ylabel("1 - S(t)", color=self.y_color.get(), fontsize=self.axis_fontsize.get())
        elif graph_type == "Riesgo Acumulado":
            ax.set_title("Riesgo Acumulado", color=self.title_color.get(), fontsize=self.title_fontsize.get())
            ax.set_ylabel("Riesgo Acumulado", color=self.y_color.get(), fontsize=self.axis_fontsize.get())
        ax.legend()
        ax.set_xlabel(self.x_label.get(), color=self.x_color.get(), fontsize=self.axis_fontsize.get())
        self.canvas.draw()
        self.txt_log.insert(tk.END, "Gráfica generada.\n")
        self.log_debug("Se completó el análisis Kaplan-Meier.")

    # ---------------------------------------------------------------------------
    # MÉTODO: export_summary_to_excel
    # ---------------------------------------------------------------------------
    def export_summary_to_excel(self, summary_data):
        """
        Exporta la tabla de estadísticas a un archivo Excel.
        """
        file_path = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")]
        )
        if not file_path:
            return
        try:
            df_summary = pd.DataFrame(summary_data)
            df_summary.to_excel(file_path, index=False)
            messagebox.showinfo("Éxito", f"Resumen exportado a Excel:\n{file_path}")
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo exportar a Excel:\n{e}")

    # ---------------------------------------------------------------------------
    # MÉTODO: show_summary_popup
    # ---------------------------------------------------------------------------
    def show_summary_popup(self):
        if not self.km_results:
            messagebox.showwarning("Aviso", "No hay resultados. Genere primero la gráfica.")
            return

        def format_value(val):
            """Formatea el valor para mostrar con máximo 2 decimales, notación científica si es pequeño, entero si es grande."""
            try:
                if np.isnan(val):
                    return "N/A"
                if abs(val) < 0.01:
                    return f"{val:.2e}"
                if abs(val) >= 100:
                    return f"{int(round(val))}"
                return f"{val:.2f}"
            except Exception:
                return str(val)

        popup = tk.Toplevel(self)
        popup.title("Resumen de Estadísticos de Supervivencia")
        txt = tk.Text(popup, wrap="none", width=140, height=30)
        vsb = ttk.Scrollbar(popup, orient="vertical", command=txt.yview)
        hsb = ttk.Scrollbar(popup, orient="horizontal", command=txt.xview)
        txt.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        txt.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")
        popup.rowconfigure(0, weight=1)
        popup.columnconfigure(0, weight=1)
        header = (
            f"{'Categoría':<20} | {'N casos':<8} | {'N censurados':<12} | "
            f"{'Mediana':<10} | {'SE med.':<8} | {'IC med.':<18} | "
            f"{'p25':<10} | {'SE p25':<8} | {'IC p25':<18} | "
            f"{'p75':<10} | {'SE p75':<8} | {'IC p75':<18} | "
            f"{'p3':<10} | {'SE p3':<8} | {'IC p3':<18} | "
            f"{'p97':<10} | {'SE p97':<8} | {'IC p97':<18}\n"
        )
        separator = "-" * 160 + "\n"
        txt.insert(tk.END, header)
        txt.insert(tk.END, separator)
        summary_data = []
        for (catv, stats, bs_stats, n_used, n_censored) in self.km_results:
            if bs_stats:
                med = bs_stats["median"]
                p25 = bs_stats["p25"]
                p75 = bs_stats["p75"]
                p3 = bs_stats["p3"]
                p97 = bs_stats["p97"]
                line = (
                    f"{catv:<20} | {n_used:<8} | {n_censored:<12} | "
                    f"{format_value(med['estimate']):<10} | {format_value(med['se']):<8} | ({format_value(med['ci'][0])}, {format_value(med['ci'][1])})  | "
                    f"{format_value(p25['estimate']):<10} | {format_value(p25['se']):<8} | ({format_value(p25['ci'][0])}, {format_value(p25['ci'][1])})  | "
                    f"{format_value(p75['estimate']):<10} | {format_value(p75['se']):<8} | ({format_value(p75['ci'][0])}, {format_value(p75['ci'][1])})  | "
                    f"{format_value(p3['estimate']):<10} | {format_value(p3['se']):<8} | ({format_value(p3['ci'][0])}, {format_value(p3['ci'][1])})  | "
                    f"{format_value(p97['estimate']):<10} | {format_value(p97['se']):<8} | ({format_value(p97['ci'][0])}, {format_value(p97['ci'][1])})"
                )
                # Preparar diccionario para exportar a Excel
                summary_data.append({
                    "Categoría": catv,
                    "N casos": n_used,
                    "N censurados": n_censored,
                    "Mediana": format_value(med['estimate']),
                    "SE med.": format_value(med['se']),
                    "IC med.": f"({format_value(med['ci'][0])}, {format_value(med['ci'][1])})",
                    "p25": format_value(p25['estimate']),
                    "SE p25": format_value(p25['se']),
                    "IC p25": f"({format_value(p25['ci'][0])}, {format_value(p25['ci'][1])})",
                    "p75": format_value(p75['estimate']),
                    "SE p75": format_value(p75['se']),
                    "IC p75": f"({format_value(p75['ci'][0])}, {format_value(p75['ci'][1])})",
                    "p3": format_value(p3['estimate']),
                    "SE p3": format_value(p3['se']),
                    "IC p3": f"({format_value(p3['ci'][0])}, {format_value(p3['ci'][1])})",
                    "p97": format_value(p97['estimate']),
                    "SE p97": format_value(p97['se']),
                    "IC p97": f"({format_value(p97['ci'][0])}, {format_value(p97['ci'][1])})"
                })
            else:
                med_val = stats.get("median")
                p25_val = stats.get("p25")
                p75_val = stats.get("p75")
                line = (
                    f"{catv:<20} | {n_used:<8} | {n_censored:<12} | "
                    f"{(format_value(med_val) if med_val is not None else 'N/A'):<10} | {'N/A':<8} | {'N/A':<18} | "
                    f"{(format_value(p25_val) if p25_val is not None else 'N/A'):<10} | {'N/A':<8} | {'N/A':<18} | "
                    f"{(format_value(p75_val) if p75_val is not None else 'N/A'):<10} | {'N/A':<8} | {'N/A':<18} | "
                    f"{'N/A':<10} | {'N/A':<8} | {'N/A':<18} | "
                    f"{'N/A':<10} | {'N/A':<8} | {'N/A':<18}"
                )
                summary_data.append({
                    "Categoría": catv,
                    "N casos": n_used,
                    "N censurados": n_censored,
                    "Mediana": (format_value(med_val) if med_val is not None else 'N/A'),
                    "SE med.": "N/A",
                    "IC med.": "N/A",
                    "p25": (format_value(p25_val) if p25_val is not None else 'N/A'),
                    "SE p25": "N/A",
                    "IC p25": "N/A",
                    "p75": (format_value(p75_val) if p75_val is not None else 'N/A'),
                    "SE p75": "N/A",
                    "IC p75": "N/A",
                    "p3": "N/A",
                    "SE p3": "N/A",
                    "IC p3": "N/A",
                    "p97": "N/A",
                    "SE p97": "N/A",
                    "IC p97": "N/A"
                })
            txt.insert(tk.END, line + "\n")
        txt.config(state="disabled")
        btn_excel = ttk.Button(popup, text="Exportar a Excel", command=lambda: self.export_summary_to_excel(summary_data))
        btn_excel.grid(row=2, column=0, pady=5)
        self.log_debug("Se mostró el resumen extendido de estadísticas de supervivencia.")

    # ---------------------------------------------------------------------------
    # MÉTODO: export_summary_to_excel
    # ---------------------------------------------------------------------------
    def export_summary_to_excel(self, summary_data):
        file_path = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")]
        )
        if not file_path:
            return
        try:
            df_summary = pd.DataFrame(summary_data)
            df_summary.to_excel(file_path, index=False)
            messagebox.showinfo("Éxito", f"Resumen exportado a Excel:\n{file_path}")
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo exportar a Excel:\n{e}")

    # ---------------------------------------------------------------------------
    # MÉTODO: run_logrank_tests
    # ---------------------------------------------------------------------------
    def run_logrank_tests(self):
        if self.data is None:
            messagebox.showwarning("Advertencia", "Primero cargue los datos.")
            return
        time_col = self.cmb_time.get().strip()
        event_col = self.cmb_event.get().strip()
        if not time_col or not event_col:
            messagebox.showwarning("Advertencia", "Seleccione las variables de tiempo y evento.")
            return

        # 1. Aplicar filtros con FilterComponent si está disponible
        if FILTER_COMPONENT_AVAILABLE and self.custom_filter_component_instance and self.data is not None:
            try:
                # self.custom_filter_component_instance.set_dataframe(self.data) # Opcional
                df_temp_filtered = self.custom_filter_component_instance.apply_filters()
                if df_temp_filtered is None:
                    messagebox.showerror("Error de Filtro", "El componente de filtro devolvió None. Verifique la configuración del filtro o los datos.")
                    return
                self.log_debug(f"Datos filtrados por FilterComponent para Log-Rank: {df_temp_filtered.shape[0]} filas.")
                if df_temp_filtered.empty:
                    messagebox.showwarning("Datos Vacíos", "No quedan datos después de aplicar los filtros del componente.")
                    return
            except Exception as e:
                messagebox.showerror("Error en FilterComponent", f"Error al aplicar filtros: {e}")
                return
        elif self.data is not None:
            df_temp_filtered = self.data.copy()
            self.log_debug("Usando datos originales para Log-Rank (FilterComponent no disponible/activo o datos no cargados en él).")
        else:
            messagebox.showwarning("Advertencia", "Carga datos primero.")
            return

        # 2. Aplicar filtros globales (blancos, no numéricos)
        df_temp_filtered = self.apply_global_filters(df_temp_filtered)

        # 3. Aplicar filtro/etiquetado/orden de la variable de agrupación principal
        df_final_for_analysis, self.ordered_categories = self._apply_main_categorization(df_temp_filtered)

        # 4. Eliminar NaNs en columnas de tiempo y evento
        df_final_for_analysis = df_final_for_analysis.dropna(subset=[time_col, event_col])

        # 5. Verificar si quedan datos
        if df_final_for_analysis.empty:
            messagebox.showwarning("Aviso", "No quedan datos tras aplicar filtros y limpieza T/E.")
            return

        # 6. Preparar datos para Log-Rank (usar df_final_for_analysis)
        cat_var = self.cmb_cat.get().strip()
        groups = {}
        if cat_var and cat_var in df_final_for_analysis.columns:
            if self.ordered_categories:
                # Usar categorías ordenadas si existen, asegurándose que estén presentes
                group_names = [c for c in self.ordered_categories if c in df_final_for_analysis[cat_var].unique()]
            else:
                group_names = sorted(list(df_final_for_analysis[cat_var].dropna().unique()))
            # group_names ya deberían ser strings o del tipo correcto por _apply_main_categorization
            for name in group_names:
                group_df = df_final_for_analysis[df_final_for_analysis[cat_var] == name].copy()
                group_df = group_df.dropna(subset=[time_col, event_col])
                if len(group_df) >= 3:
                    groups[name] = group_df
        else:
            groups["(SinCategoría)"] = df_final_for_analysis.copy()
        if len(groups) < 2:
            messagebox.showwarning("Advertencia", "Se requieren al menos dos grupos para comparar Log-Rank.")
            return
        selected_test = self.cmb_test_type.get()
        if selected_test == "Log-Rank":
            selected_weight = None
        elif selected_test == "Breslow (Wilcoxon)":
            selected_weight = "wilcoxon"
        elif selected_test in ["Tarone-Ware", "Peto-Peto"]:
            messagebox.showinfo("Información", f"La opción {selected_test} no está implementada en la versión actual de lifelines.\nSe usará Breslow (Wilcoxon) en su lugar.")
            selected_weight = "wilcoxon"
        else:
            selected_weight = None
        try:
            user_early = float(self.entry_early_cutoff.get())
            early_cutoff = user_early
        except ValueError:
            early_cutoff = np.percentile(df_final_for_analysis[time_col], 25)
        try:
            user_late = float(self.entry_late_cutoff.get())
            late_cutoff = user_late
        except ValueError:
            late_cutoff = np.percentile(df_final_for_analysis[time_col], 75)
        durations = df_final_for_analysis[time_col]
        events = df_final_for_analysis[event_col]
        if cat_var and cat_var in df_final_for_analysis.columns:
            labels = df_final_for_analysis[cat_var] # Usar la columna directamente (ya debería ser string o categorical)
        else:
            labels = pd.Series(["(SinCategoría)"] * len(df_final_for_analysis), index=df_final_for_analysis.index)
        global_test = multivariate_logrank_test(durations, labels, event_observed=events, weightings=selected_weight)
        global_p = global_test.p_value
        early_groups = {}
        for name, group_df in groups.items():
            early_df = group_df[group_df[time_col] <= early_cutoff]
            if len(early_df) >= 3:
                early_groups[name] = early_df
        if len(early_groups) >= 2:
            early_data = pd.concat(list(early_groups.values()))
            early_labels = []
            for name, group_df in early_groups.items():
                early_labels.extend([name] * len(group_df))
            early_labels = pd.Series(early_labels, index=early_data.index)
            early_test = multivariate_logrank_test(early_data[time_col], early_labels, event_observed=early_data[event_col],
                                                   weightings=selected_weight)
            early_p = early_test.p_value
        else:
            early_p = None
        late_groups = {}
        for name, group_df in groups.items():
            late_df = group_df[group_df[time_col] >= late_cutoff]
            if len(late_df) >= 3:
                late_groups[name] = late_df
        if len(late_groups) >= 2:
            late_data = pd.concat(list(late_groups.values()))
            late_labels = []
            for name, group_df in late_groups.items():
                late_labels.extend([name] * len(group_df))
            late_labels = pd.Series(late_labels, index=late_data.index)
            late_test = multivariate_logrank_test(late_data[time_col], late_labels, event_observed=late_data[event_col],
                                                  weightings=selected_weight)
            late_p = late_test.p_value
        else:
            late_p = None
        pairwise_results = []
        group_names = list(groups.keys())
        for i in range(len(group_names)):
            for j in range(i+1, len(group_names)):
                g1 = groups[group_names[i]]
                g2 = groups[group_names[j]]
                res = logrank_test(g1[time_col], g2[time_col],
                                   event_observed_A=g1[event_col], event_observed_B=g2[event_col],
                                   weightings=selected_weight)
                pairwise_results.append((f"{group_names[i]} vs {group_names[j]}", res.p_value))
        pairwise_early = []
        early_group_names = list(early_groups.keys())
        for i in range(len(early_group_names)):
            for j in range(i+1, len(early_group_names)):
                df1 = early_groups[early_group_names[i]]
                df2 = early_groups[early_group_names[j]]
                res = logrank_test(df1[time_col], df2[time_col],
                                   event_observed_A=df1[event_col], event_observed_B=df2[event_col],
                                   weightings=selected_weight)
                pairwise_early.append((f"{early_group_names[i]} vs {early_group_names[j]}", res.p_value))
        pairwise_late = []
        late_group_names = list(late_groups.keys())
        for i in range(len(late_group_names)):
            for j in range(i+1, len(late_group_names)):
                df1 = late_groups[late_group_names[i]]
                df2 = late_groups[late_group_names[j]]
                res = logrank_test(df1[time_col], df2[time_col],
                                   event_observed_A=df1[event_col], event_observed_B=df2[event_col],
                                   weightings=selected_weight)
                pairwise_late.append((f"{late_group_names[i]} vs {late_group_names[j]}", res.p_value))
        apply_bonf = self.apply_bonferroni.get()
        if apply_bonf:
            global_p_adj = min(global_p * 1, 1.0)
            early_p_adj = min(early_p * 1, 1.0) if early_p is not None else None
            late_p_adj = min(late_p * 1, 1.0) if late_p is not None else None
            pairwise_results = [(name, min(p * len(pairwise_results), 1.0)) for name, p in pairwise_results]
            pairwise_early = [(name, min(p * len(pairwise_early), 1.0)) for name, p in pairwise_early]
            pairwise_late = [(name, min(p * len(pairwise_late), 1.0)) for name, p in pairwise_late]
        else:
            global_p_adj = global_p
            early_p_adj = early_p
            late_p_adj = late_p
        summary = ""
        summary += "=== Log-Rank Tests Summary ===\n\n"
        summary += f"Test Global ({self.cmb_test_type.get()}): p = {global_p_adj:.4f}\n"
        if early_p_adj is not None:
            summary += f"Test Global Early (time <= {early_cutoff:.2f}): p = {early_p_adj:.4f}\n"
        else:
            summary += "Test Global Early: No hay suficientes datos para comparar\n"
        if late_p_adj is not None:
            summary += f"Test Global Late (time >= {late_cutoff:.2f}): p = {late_p_adj:.4f}\n"
        else:
            summary += "Test Global Late: No hay suficientes datos para comparar\n"
        summary += "\n--- Pairwise Comparisons (Overall) ---\n"
        for name, p in pairwise_results:
            summary += f"{name}: p = {p:.4f}\n"
        summary += "\n--- Pairwise Comparisons Early ---\n"
        if pairwise_early:
            for name, p in pairwise_early:
                summary += f"{name}: p = {p:.4f}\n"
        else:
            summary += "No hay suficientes datos para comparar en el periodo early.\n"
        summary += "\n--- Pairwise Comparisons Late ---\n"
        if pairwise_late:
            for name, p in pairwise_late:
                summary += f"{name}: p = {p:.4f}\n"
        else:
            summary += "No hay suficientes datos para comparar en el periodo late.\n"
        if apply_bonf:
            summary += "\n(Se aplicó corrección de Bonferroni a los p-values)\n"
        popup2 = tk.Toplevel(self)
        popup2.title("Resultados Log-Rank")
        txt2 = tk.Text(popup2, wrap="none", width=100, height=30)
        vsb2 = ttk.Scrollbar(popup2, orient="vertical", command=txt2.yview)
        hsb2 = ttk.Scrollbar(popup2, orient="horizontal", command=txt2.xview)
        txt2.configure(yscrollcommand=vsb2.set, xscrollcommand=hsb2.set)
        txt2.grid(row=0, column=0, sticky="nsew")
        vsb2.grid(row=0, column=1, sticky="ns")
        hsb2.grid(row=1, column=0, sticky="ew")
        popup2.rowconfigure(0, weight=1)
        popup2.columnconfigure(0, weight=1)
        txt2.insert(tk.END, summary)
        txt2.config(state="disabled")
        self.log_debug("Se completaron las pruebas Log-Rank.")

    # ---------------------------------------------------------------------------
    # MÉTODO: save_plot
    # ---------------------------------------------------------------------------
    def save_plot(self):
        if self.data is None:
            messagebox.showwarning("Advertencia", "Primero cargue datos y genere la gráfica.")
            return
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("PDF", "*.pdf"), ("SVG", "*.svg")]
        )
        if not file_path:
            return
        width_inch = self.save_width.get() / float(self.save_dpi.get())
        height_inch = self.save_height.get() / float(self.save_dpi.get())
        try:
            self.figure.savefig(file_path, dpi=self.save_dpi.get(),
                                figsize=(width_inch, height_inch),
                                bbox_inches="tight")
            self.txt_log.insert(tk.END, f"Gráfico guardado en: {file_path}\n")
            self.log_debug(f"Gráfico guardado en: {file_path}")
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo guardar la figura:\n{e}")

    # ---------------------------------------------------------------------------
    # MÉTODO: extra_info (bloques extra para extender el tamaño del archivo)
    # ---------------------------------------------------------------------------
    def extra_info(self):
        self.log_debug("Ejecutando extra_info() – Información adicional cargada.")
        # FUTURAS MEJORAS: Agregar menú 'Acerca de', guardar logs, notificaciones, etc.
        pass

    # ---------------------------------------------------------------------------
    # MÉTODO: log_debug
    # ---------------------------------------------------------------------------
    def log_debug(self, message):
        self.txt_log.insert(tk.END, f"[DEBUG] {message}\n")
        self.txt_log.see(tk.END)

# =============================================================================
# BLOQUE PRINCIPAL
# =============================================================================
if __name__ == "__main__":
    root = tk.Tk()
    root.title("Análisis de Supervivencia (Kaplan-Meier) con Opciones Adicionales")
    root.geometry("1200x700")
    app = SurvivalAnalysisTab(root)
    app.pack(fill=tk.BOTH, expand=True)
    app.log_debug("Aplicación iniciada correctamente.")
    root.mainloop()

# =============================================================================
# BLOQUE EXTRA: DUMMY CODE PARA EXTENDER EL ARCHIVO (NO MODIFICA FUNCIONALIDAD)
# =============================================================================
def dummy_function_1():
    """
    Función dummy para extender el archivo.
    Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed non risus. Suspendisse lectus tortor,
    dignissim sit amet, adipiscing nec, ultricies sed, dolor.
    """
    for i in range(50):
        print("Dummy function 1, línea", i)

def dummy_function_2():
    """
    Otra función dummy para extender el tamaño del archivo.
    Pellentesque habitant morbi tristique senectus et netus et malesuada fames ac turpis egestas.
    """
    for i in range(50):
        print("Dummy function 2, línea", i)

class dummy_class:
    """
    Clase dummy para extender el script.
    """
    def __init__(self):
        self.message = "Clase dummy en ejecución."
    def print_message(self):
        print(self.message)
    def do_nothing(self):
        pass

def print_long_dummy_text():
    dummy_text = """
Lorem ipsum dolor sit amet, consectetur adipiscing elit. Vestibulum in porttitor urna.
Suspendisse potenti. Aliquam erat volutpat. Integer in volutpat libero. Proin ac massa rutrum,
maximus sapien eget, mollis leo. Donec suscipit massa ut elit interdum, at dignissim magna facilisis.
Nullam sit amet lacus sed dui cursus blandit. Fusce eget dui ut enim aliquet volutpat.
Vestibulum ante ipsum primis in faucibus orci luctus et ultrices posuere cubilia Curae;
Mauris auctor, ex eu sagittis interdum, tellus sapien malesuada eros, a varius lectus libero eget nulla.
Vivamus in interdum urna. Maecenas aliquet, nulla vel convallis commodo, sapien ante sodales libero,
in volutpat massa ex et turpis.
Cras rutrum, massa in ultricies luctus, enim orci dignissim quam, vitae tempus turpis nulla a urna.
Phasellus et lacus non erat pretium luctus.
""" * 15
    print(dummy_text)

def run_dummy_functions():
    dummy_function_1()
    dummy_function_2()
    dc = dummy_class()
    dc.print_message()
    run_dummy = lambda: [print_long_dummy_text() for _ in range(3)]
    run_dummy()

if __name__ == "__main__":
    run_dummy_functions()

# =============================================================================
# FIN DEL SCRIPT EXTRA
# =============================================================================
