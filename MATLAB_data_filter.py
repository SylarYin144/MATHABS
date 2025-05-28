#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DataFilterApp con Múltiples Gráficos Avanzados, Grid y Control de Anotación
--------------------------------------------------------------------------
Aplicación Tkinter para cargar datos de Excel, aplicar hasta 4 filtros encadenados,
generar estadísticas descriptivas y una variedad de gráficos personalizables
(Histograma, Barras, Countplot, Caja y Bigotes, Violín, Boxen, Swarm, KDE, Histplot,
Líneas, Dispersión con/sin regresión, Circular).
Permite mostrar puntos individuales en Box/Violin/Boxen plots, activar/desactivar la reja,
y controlar la anotación de filtros/n.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import numpy as np
import sys
import io
import importlib.util
import traceback
import seaborn as sns # Importar seaborn para gráficos estadísticos avanzados

from MATLAB_filter_component import FilterComponent

# --- Base para Plugins ---
class BasePlugin:
    """Clase base para plugins. Define la interfaz esperada."""
    name = "Plugin Desconocido"

    def run(self, df: pd.DataFrame, app_instance):
        """
        Método a implementar por cada plugin.
        Recibe el DataFrame filtrado y la instancia de la aplicación principal (DataFilterTab).
        """
        raise NotImplementedError("El método 'run' debe ser implementado por el plugin.")

# --- Pestaña de Filtros y Gráficos ---
class DataFilterTab(ttk.Frame):
    """Pestaña de filtros, estadísticas y gráficos."""

    def __init__(self, master):
        super().__init__(master)
        # Colores disponibles (40)
        self.color_options = [
            "blue", "green", "red", "skyblue", "orange", "purple", "black", "gray", "brown", "pink",
            "gold", "olive", "cyan", "magenta", "navy", "teal", "aqua", "maroon", "lime", "silver",
            "salmon", "plum", "turquoise", "coral", "indigo", "khaki", "lavender", "peru", "tomato", "chocolate",
            "darkgreen", "darkred", "darkblue", "darkorange", "lightblue", "lightgreen", "lightcoral",
            "violet", "orchid", "royalblue"
        ]
        self.data = None  # DataFrame cargado
        self.filtered_data = None # DataFrame después de aplicar filtros
        self.graph_path = None # Ruta de la última gráfica generada

        # Variables para Filtros Generales eliminadas
        
        self.general_filter_operators = ["==", "!=", ">", "<", ">=", "<=", "contiene", "no contiene", "es NaN", "no es NaN"] # Podría ser útil para FilterComponent si se extiende

        # ---------------- Layout principal ----------------
        self.paned = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        self.paned.pack(fill=tk.BOTH, expand=True)

        # Frame izquierdo = controles (con scroll)
        self.left_frame = ttk.Frame(self.paned)
        self.paned.add(self.left_frame, weight=1)

        self.scroll_canvas = tk.Canvas(self.left_frame)
        self.scroll_canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar = ttk.Scrollbar(self.left_frame, orient="vertical",
                                       command=self.scroll_canvas.yview)
        self.scrollbar.pack(side="right", fill="y")
        self.scroll_canvas.configure(yscrollcommand=self.scrollbar.set)

        self.scrollable_frame = ttk.Frame(self.scroll_canvas)
        self.scroll_canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.scroll_canvas.configure(scrollregion=self.scroll_canvas.bbox("all"))
        )

        # Frame derecho = resultados
        self.right_frame = ttk.Frame(self.paned)
        self.paned.add(self.right_frame, weight=2)

        # ---------------- Controles ----------------
        lbl_title = ttk.Label(self.scrollable_frame,
                              text="Carga de Datos y Configuración de Filtros/Estadísticas",
                              font=("Helvetica", 14))
        lbl_title.pack(pady=10)

        ttk.Button(self.scrollable_frame, text="Cargar Datos (Excel)",
                   command=self.load_data).pack(pady=5)
        self.lbl_file = ttk.Label(self.scrollable_frame, text="Ningún archivo cargado.")
        self.lbl_file.pack(pady=5)

        # -------- Filtros Generales (Reemplazado por FilterComponent) --------
        frm_filters_general = ttk.LabelFrame(self.scrollable_frame, text="Filtros Avanzados")
        frm_filters_general.pack(padx=10, pady=5, fill="x")

        self.filter_component = FilterComponent(frm_filters_general)
        self.filter_component.pack(fill="x", expand=True, padx=5, pady=5)

        # -------- Variable Principal y Opciones de Análisis --------
        frm_analysis_var = ttk.LabelFrame(self.scrollable_frame, text="Variable Principal y Opciones de Análisis")
        frm_analysis_var.pack(padx=10, pady=5, fill="x")

        ttk.Label(frm_analysis_var, text="Variable Principal para Análisis:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.cmb_variables = ttk.Combobox(frm_analysis_var, state="readonly") # Mantenido para seleccionar la variable a analizar
        self.cmb_variables.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

        ttk.Label(frm_analysis_var, text="Tipo de Variable Principal:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.cmb_tipo = ttk.Combobox(frm_analysis_var, values=["Cuantitativa", "Cualitativa"], state="readonly") # Mantenido
        self.cmb_tipo.grid(row=1, column=1, padx=5, pady=5, sticky="ew")
        self.cmb_tipo.set("Cuantitativa")

        ttk.Label(frm_analysis_var, text="Etiquetas y orden (para cualitativa):").grid(row=2, column=0, padx=5, pady=5, sticky="w")
        self.entry_etiquetas = ttk.Entry(frm_analysis_var) # Mantenido
        self.entry_etiquetas.grid(row=2, column=1, padx=5, pady=5, sticky="ew")
        ttk.Label(frm_analysis_var,
                  text="(Ej: 2:Alto,1:Medio,0:Bajo)").grid(row=3, column=0, columnspan=2,
                                                                                  padx=5, pady=2, sticky="w")

        self.exclude_blank = tk.BooleanVar(value=False) # Mantenido para análisis/estadísticas
        ttk.Checkbutton(frm_analysis_var, text="Excluir blancos/NaNs de la Variable Principal en Estadísticas/Gráficos",
                         variable=self.exclude_blank).grid(row=4, column=0, columnspan=2, padx=5, pady=5, sticky="w")

        frm_analysis_var.columnconfigure(1, weight=1) # Hacer que combobox/entry se expandan

        # -------- Parámetros de gráfica --------
        # (Se mantiene igual, se construye después)
        self._build_graph_params()

        # -------- Botones de acción --------
        ttk.Button(self.scrollable_frame, text="Aplicar Filtros/Estadísticas",
                   command=self.apply_filters).pack(pady=5)
        ttk.Button(self.scrollable_frame, text="Guardar Gráfica sin Ver",
                   command=self.save_graph_directly).pack(pady=5)

        # -------- Área de resultados --------
        self.txt_result = tk.Text(self.right_frame, height=30)
        self.txt_result.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

    # ---------------------------------------------------------------------
    # UI helpers (Se eliminan _build_extra_filters y _add_filter_row)
    # ---------------------------------------------------------------------

    def _build_graph_params(self):
        frm_graph = ttk.LabelFrame(self.scrollable_frame, text="Parámetros de Gráfica")
        frm_graph.pack(padx=10, pady=5, fill="x")

        # Tipo de Gráfico (Nuevo)
        ttk.Label(frm_graph, text="Tipo de Gráfico:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.cmb_plot_type = ttk.Combobox(frm_graph,
                                          values=["Histograma", "Gráfico de Barras", "Countplot",
                                                  "Caja y Bigotes", "Violín", "Boxen Plot", "Swarm Plot",
                                                  "KDE Plot", "Histplot (Hist+KDE)", "Gráfico Circular",
                                                  "Líneas", "Dispersión", "Dispersión con Regresión"],
                                          state="readonly")
        self.cmb_plot_type.grid(row=0, column=1, padx=5, pady=5, sticky="w")
        self.cmb_plot_type.set("Histograma") # Default

        # Segunda Variable (Nuevo, para gráficos 2D o agrupados)
        ttk.Label(frm_graph, text="Segunda Variable (Eje Y/Grupo/Color):").grid(row=0, column=2, padx=5, pady=5, sticky="w")
        self.cmb_variable_2d = ttk.Combobox(frm_graph, state="readonly")
        self.cmb_variable_2d.grid(row=0, column=3, padx=5, pady=5, sticky="w")

        # Mostrar puntos individuales (Nuevo)
        self.show_individual_points = tk.BooleanVar(value=False)
        ttk.Checkbutton(frm_graph, text="Mostrar puntos individuales",
                         variable=self.show_individual_points).grid(row=0, column=4, padx=5, pady=5, sticky="w")

        # Mostrar Reja (Nuevo)
        self.show_grid = tk.BooleanVar(value=False)
        ttk.Checkbutton(frm_graph, text="Mostrar reja (grid)",
                         variable=self.show_grid).grid(row=0, column=5, padx=5, pady=5, sticky="w")

        # Mostrar Anotación (Nuevo)
        self.show_annotation = tk.BooleanVar(value=True)
        ttk.Checkbutton(frm_graph, text="Mostrar anotación (filtros/n)",
                         variable=self.show_annotation).grid(row=0, column=6, padx=5, pady=5, sticky="w")


        # Tamaño / DPI
        self.entry_dpi, self.entry_width, self.entry_height = self._add_size_controls(frm_graph, start_row=1)

        # Colores principales
        self.cmb_color, self.cmb_edge = self._add_color_controls(frm_graph, start_row=2)

        # Orientación y rotación
        self.cmb_orient, self.cmb_tickrot = self._add_orientation_controls(frm_graph, start_row=2)

        # Mediana y colores de barras
        self.cmb_mediana, self.entry_bar_colors = self._add_bar_controls(frm_graph, start_row=3)

        # Texto (título, ejes)
        self.entry_custom_title, self.entry_x_label, self.entry_y_label = self._add_text_controls(frm_graph, start_row=4)

        # Escala / eje Y‑X
        self.cmb_scale, self.cmb_y_value = self._add_scale_controls(frm_graph, start_row=7) # Adjusted row

        # Límites
        (self.entry_xmin, self.entry_xmax,
         self.entry_ymin, self.entry_ymax) = self._add_limit_controls(frm_graph, start_row=8) # Adjusted row

        # Fuente
        self.cmb_font_color, self.entry_font_size = self._add_font_controls(frm_graph, start_row=10) # Adjusted row


    # ---- helpers (sub‑secciones de build_graph_params) ----
    def _add_size_controls(self, parent, start_row):
        ttk.Label(parent, text="DPI:").grid(row=start_row, column=0, padx=5, pady=5, sticky="w")
        entry_dpi = ttk.Entry(parent, width=10)
        entry_dpi.grid(row=start_row, column=1, padx=5, pady=5, sticky="w")
        entry_dpi.insert(0, "100")

        ttk.Label(parent, text="Ancho (px):").grid(row=start_row, column=2, padx=5, pady=5, sticky="w")
        entry_w = ttk.Entry(parent, width=10)
        entry_w.grid(row=start_row, column=3, padx=5, pady=5, sticky="w")
        entry_w.insert(0, "1000")

        ttk.Label(parent, text="Alto (px):").grid(row=start_row, column=4, padx=5, pady=5, sticky="w")
        entry_h = ttk.Entry(parent, width=10)
        entry_h.grid(row=start_row, column=5, padx=5, pady=5, sticky="w")
        entry_h.insert(0, "600")
        return entry_dpi, entry_w, entry_h

    def _add_color_controls(self, parent, start_row):
        ttk.Label(parent, text="Color gráfico:").grid(row=start_row, column=0, padx=5, pady=5, sticky="w")
        cmb_color = ttk.Combobox(parent, values=self.color_options, state="readonly", width=10)
        cmb_color.grid(row=start_row, column=1, padx=5, pady=5, sticky="w")
        cmb_color.set("skyblue")

        ttk.Label(parent, text="Color borde:").grid(row=start_row, column=2, padx=5, pady=5, sticky="w")
        cmb_edge = ttk.Combobox(parent, values=self.color_options, state="readonly", width=10)
        cmb_edge.grid(row=start_row, column=3, padx=5, pady=5, sticky="w")
        cmb_edge.set("black")
        return cmb_color, cmb_edge

    def _add_orientation_controls(self, parent, start_row):
        ttk.Label(parent, text="Orientación:").grid(row=start_row, column=4, padx=5, pady=5, sticky="w")
        cmb_orient = ttk.Combobox(parent, values=["Vertical", "Horizontal"], state="readonly", width=10)
        cmb_orient.grid(row=start_row, column=5, padx=5, pady=5, sticky="w")
        cmb_orient.set("Vertical")

        ttk.Label(parent, text="Rotación ticks:").grid(row=start_row, column=6, padx=5, pady=5, sticky="w")
        cmb_rot = ttk.Combobox(parent, values=["0", "45", "90"], state="readonly", width=5)
        cmb_rot.grid(row=start_row, column=7, padx=5, pady=5, sticky="w")
        cmb_rot.set("0")
        return cmb_orient, cmb_rot

    def _add_bar_controls(self, parent, start_row):
        ttk.Label(parent, text="Color mediana:").grid(row=start_row, column=0, padx=5, pady=5, sticky="w")
        cmb_mediana = ttk.Combobox(parent, values=self.color_options, state="readonly", width=10)
        cmb_mediana.grid(row=start_row, column=1, padx=5, pady=5, sticky="w")
        cmb_mediana.set("red")

        ttk.Label(parent, text="Colores de barras (coma):").grid(row=start_row, column=2, padx=5, pady=5, sticky="w")
        entry_bars = ttk.Entry(parent, width=30)
        entry_bars.grid(row=start_row, column=3, columnspan=2, padx=5, pady=5, sticky="w")
        return cmb_mediana, entry_bars

    def _add_text_controls(self, parent, start_row):
        ttk.Label(parent, text="Título del gráfico:").grid(row=start_row, column=0, padx=5, pady=5, sticky="w")
        entry_title = ttk.Entry(parent, width=30)
        entry_title.grid(row=start_row, column=1, columnspan=2, padx=5, pady=5, sticky="w")

        ttk.Label(parent, text="Etiqueta eje X:").grid(row=start_row+1, column=0, padx=5, pady=5, sticky="w")
        entry_x = ttk.Entry(parent, width=30)
        entry_x.grid(row=start_row+1, column=1, columnspan=2, padx=5, pady=5, sticky="w")

        ttk.Label(parent, text="Etiqueta eje Y:").grid(row=start_row+2, column=0, padx=5, pady=5, sticky="w")
        entry_y = ttk.Entry(parent, width=30)
        entry_y.grid(row=start_row+2, column=1, columnspan=2, padx=5, pady=5, sticky="w")
        return entry_title, entry_x, entry_y

    def _add_scale_controls(self, parent, start_row):
        ttk.Label(parent, text="Escala eje:").grid(row=start_row, column=0, padx=5, pady=5, sticky="w")
        cmb_scale = ttk.Combobox(parent, values=["linear", "log"], state="readonly", width=10)
        cmb_scale.grid(row=start_row, column=1, padx=5, pady=5, sticky="w")
        cmb_scale.set("linear")

        ttk.Label(parent, text="Valor eje Y/X:").grid(row=start_row+1, column=0, padx=5, pady=5, sticky="w")
        cmb_yval = ttk.Combobox(parent, values=["Count", "Frequency", "Density", "Percent", "Probability"], state="readonly", width=10)
        cmb_yval.grid(row=start_row+1, column=1, padx=5, pady=5, sticky="w")
        cmb_yval.set("Count")
        return cmb_scale, cmb_yval

    def _add_limit_controls(self, parent, start_row):
        ttk.Label(parent, text="X min:").grid(row=start_row, column=0, padx=5, pady=5, sticky="e")
        entry_xmin = ttk.Entry(parent, width=10)
        entry_xmin.grid(row=start_row, column=1, padx=5, pady=5, sticky="w")

        ttk.Label(parent, text="X max:").grid(row=start_row, column=2, padx=5, pady=5, sticky="e")
        entry_xmax = ttk.Entry(parent, width=10)
        entry_xmax.grid(row=start_row, column=3, padx=5, pady=5, sticky="w")

        ttk.Label(parent, text="Y min:").grid(row=start_row+1, column=0, padx=5, pady=5, sticky="e")
        entry_ymin = ttk.Entry(parent, width=10)
        entry_ymin.grid(row=start_row+1, column=1, padx=5, pady=5, sticky="w")

        ttk.Label(parent, text="Y max:").grid(row=start_row+1, column=2, padx=5, pady=5, sticky="e")
        entry_ymax = ttk.Entry(parent, width=10)
        entry_ymax.grid(row=start_row+1, column=3, padx=5, pady=5, sticky="w")
        return entry_xmin, entry_xmax, entry_ymin, entry_ymax

    def _add_font_controls(self, parent, start_row):
        ttk.Label(parent, text="Color fuente:").grid(row=start_row, column=0, padx=5, pady=5, sticky="w")
        cmb_font_color = ttk.Combobox(parent, values=self.color_options + ["black", "white"], state="readonly", width=10)
        cmb_font_color.grid(row=start_row, column=1, padx=5, pady=5, sticky="w")
        cmb_font_color.set("black")

        ttk.Label(parent, text="Tamaño fuente:").grid(row=start_row, column=2, padx=5, pady=5, sticky="w")
        entry_font_size = ttk.Entry(parent, width=5)
        entry_font_size.grid(row=start_row, column=3, padx=5, pady=5, sticky="w")
        entry_font_size.insert(0, "10")
        return cmb_font_color, entry_font_size

    # ---------------------------------------------------------------------
    # Lógica principal
    # ---------------------------------------------------------------------
    def load_data(self):
        """Carga datos de un archivo Excel."""
        filepath = filedialog.askopenfilename(filetypes=[("Excel files", "*.xlsx *.xls")])
        if not filepath:
            return
        try:
            self.data = pd.read_excel(filepath)
            self.lbl_file.config(text=os.path.basename(filepath))
            # Actualizar combobox de variable principal y secundaria
            self._update_analysis_variable_comboboxes()
            # Actualizar FilterComponent con el nuevo DataFrame
            self.filter_component.set_dataframe(self.data)
            messagebox.showinfo("Éxito", "Datos cargados correctamente.")
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo cargar el archivo: {e}")
            self.data = None
            self.lbl_file.config(text="Ningún archivo cargado.")
            self._update_analysis_variable_comboboxes()
            # Limpiar FilterComponent si hay error
            self.filter_component.set_dataframe(None)

    def _update_analysis_variable_comboboxes(self):
        """Actualiza los comboboxes de variables para análisis (principal y secundaria)."""
        variables = list(self.data.columns) if self.data is not None else []
        self.cmb_variables['values'] = variables # Para variable principal
        self.cmb_variable_2d['values'] = [""] + variables # Para variable secundaria de gráfico

        if variables:
            self.cmb_variables.set(variables[0]) # Seleccionar la primera por defecto
            self.cmb_variable_2d.set("") # Sin secundaria por defecto
        else:
            self.cmb_variables.set("")
            self.cmb_variable_2d.set("")

    # Se elimina _parse_filter_values ya que la lógica está en FilterComponent
    # Se elimina _apply_general_filters ya que la lógica está en FilterComponent

    def apply_filters(self):
        """Aplica los filtros del componente y genera estadísticas/gráfica para la variable principal."""
        if self.data is None:
            messagebox.showwarning("Aviso", "Carga datos primero.")
            return

        var_principal = self.cmb_variables.get()
        if not var_principal:
            messagebox.showwarning("Aviso", "Selecciona una variable principal.")
            return

        # 1. Obtener datos filtrados desde FilterComponent
        df = self.filter_component.apply_filters()

        if df is None: # Error durante el filtrado en el componente
            messagebox.showerror("Error de Filtro", "Ocurrió un error al aplicar los filtros desde el componente.")
            self.txt_result.delete("1.0", tk.END)
            if hasattr(self, 'graph_path') and self.graph_path and os.path.exists(self.graph_path):
                try: os.remove(self.graph_path)
                except: pass
            self.graph_path = None
            return
        
        if df.empty:
            messagebox.showinfo("Datos Vacíos", "No quedan datos después de aplicar los filtros.")
            self.txt_result.delete("1.0", tk.END)
            if hasattr(self, 'graph_path') and self.graph_path and os.path.exists(self.graph_path):
                try: os.remove(self.graph_path)
                except: pass
            self.graph_path = None
            return

        self.filtered_data = df # Guardar datos filtrados

        # 2. Obtener la variable principal y tipo para el análisis
        # var_principal ya se obtuvo
        if var_principal not in df.columns:
             messagebox.showerror("Error", f"La variable principal '{var_principal}' no se encuentra en los datos filtrados.")
             return

        tipo_variable_principal = self.cmb_tipo.get()

        # 3. Generar resumen y gráfica
        filters_info = [] # Omitir info de filtros por ahora, FilterComponent podría exponerla si es necesario

        # Generar resumen estadístico
        self._generate_summary(df, var_principal, tipo_variable_principal)

        # Generar gráfica
        plot_type = self.cmb_plot_type.get()
        var_secundaria = self.cmb_variable_2d.get() if self.cmb_variable_2d.get() else None
        show_points = self.show_individual_points.get()
        show_grid = self.show_grid.get()
        show_annotation = self.show_annotation.get()

        try:
            self._generate_plot(self.filtered_data, var_principal, var_secundaria, tipo_variable_principal, plot_type, show_points, show_grid, show_annotation, filters_info)
        except Exception as e:
            messagebox.showerror("Error al Graficar", f"Ocurrió un error al generar la gráfica:\n{e}\n\n{traceback.format_exc()}")


    def _generate_plot(self, df_orig: pd.DataFrame, var_principal: str, var_secundaria: str,
                       tipo_variable_principal: str, plot_type: str, show_points: bool, show_grid: bool, show_annotation: bool, filters_info: list):
        """Motor central para generar todos los tipos de gráficos."""
        if df_orig.empty:
            messagebox.showinfo("Información", "No hay datos para graficar después de aplicar los filtros.")
            if hasattr(self, 'graph_path') and self.graph_path and os.path.exists(self.graph_path):
                try: os.remove(self.graph_path)
                except: pass
            self.graph_path = None
            return

        df_plot = df_orig.copy()

        # Preprocesamiento de variables según el tipo seleccionado
        if tipo_variable_principal == "Cualitativa":
            df_plot[var_principal] = df_plot[var_principal].astype(str)
            if var_secundaria and var_secundaria in df_plot.columns:
                df_plot[var_secundaria] = df_plot[var_secundaria].astype(str)
        elif tipo_variable_principal == "Cuantitativa":
            if var_principal in df_plot.columns:
                 df_plot[var_principal] = pd.to_numeric(df_plot[var_principal], errors='coerce')
            if var_secundaria and var_secundaria in df_plot.columns:
                 # Solo convertir var_secundaria si es relevante para el tipo de gráfico
                 if plot_type in ["Líneas", "Dispersión", "Dispersión con Regresión", "Caja y Bigotes", "Violín", "Boxen Plot", "Swarm Plot"]:
                     df_plot[var_secundaria] = pd.to_numeric(df_plot[var_secundaria], errors='coerce')


        # Parámetros comunes de la gráfica
        dpi = int(self.entry_dpi.get())
        width_px = int(self.entry_width.get())
        height_px = int(self.entry_height.get())
        fig_width_in = width_px / dpi
        fig_height_in = height_px / dpi

        color = self.cmb_color.get()
        edge = self.cmb_edge.get()
        orient = self.cmb_orient.get()
        orient_mat = "horizontal" if orient == "Horizontal" else "vertical" # Para matplotlib
        rot = int(self.cmb_tickrot.get())
        med_color = self.cmb_mediana.get()
        custom_title = self.entry_custom_title.get()
        x_label = self.entry_x_label.get()
        y_label = self.entry_y_label.get()
        scale_mode = self.cmb_scale.get()
        y_value_mode = self.cmb_y_value.get() 

        x_min = self.entry_xmin.get()
        x_max = self.entry_xmax.get()
        y_min = self.entry_ymin.get()
        y_max = self.entry_ymax.get()

        font_color = self.cmb_font_color.get()
        font_size = int(self.entry_font_size.get())

        plt.style.use('seaborn-v0_8-whitegrid' if show_grid else 'seaborn-v0_8-white') 
        plt.rcParams.update({
            'font.size': font_size,
            'axes.labelcolor': font_color,
            'xtick.color': font_color,
            'ytick.color': font_color,
            'axes.edgecolor': font_color, 
            'axes.titlecolor': font_color 
        })

        plt.figure(figsize=(fig_width_in, fig_height_in), dpi=dpi)

        try:
            if plot_type == "Histograma":
                self._plot_histogram(df_plot, var_principal, orient_mat, color, edge, med_color, scale_mode, y_value_mode, x_label, y_label, x_min, x_max, y_min, y_max, rot)
            elif plot_type == "Gráfico de Barras":
                self._plot_bar_chart(df_plot, var_principal, orient_mat, edge, y_value_mode, x_label, y_label, x_min, x_max, y_min, y_max, rot)
            elif plot_type == "Countplot":
                self._plot_countplot(df_plot, var_principal, var_secundaria, orient_mat, color, edge, y_value_mode, x_label, y_label, x_min, x_max, y_min, y_max, rot)
            elif plot_type == "Caja y Bigotes":
                self._plot_boxplot(df_plot, var_principal, var_secundaria, orient_mat, color, edge, med_color, show_points, x_label, y_label, x_min, x_max, y_min, y_max, rot)
            elif plot_type == "Violín":
                self._plot_violinplot(df_plot, var_principal, var_secundaria, orient_mat, color, edge, med_color, show_points, x_label, y_label, x_min, x_max, y_min, y_max, rot)
            elif plot_type == "Boxen Plot":
                self._plot_boxenplot(df_plot, var_principal, var_secundaria, orient_mat, color, edge, show_points, x_label, y_label, x_min, x_max, y_min, y_max, rot)
            elif plot_type == "Swarm Plot":
                self._plot_swarmplot(df_plot, var_principal, var_secundaria, orient_mat, color, edge, x_label, y_label, x_min, x_max, y_min, y_max, rot)
            elif plot_type == "KDE Plot":
                self._plot_kdeplot(df_plot, var_principal, orient_mat, color, scale_mode, x_label, y_label, x_min, x_max, y_min, y_max)
            elif plot_type == "Histplot (Hist+KDE)":
                self._plot_histplot_kde(df_plot, var_principal, orient_mat, color, edge, scale_mode, y_value_mode, x_label, y_label, x_min, x_max, y_min, y_max)
            elif plot_type == "Gráfico Circular":
                self._plot_pie_chart(df_plot, var_principal, edge, x_label, y_label) # y_value_mode no es tan relevante aquí
            elif plot_type == "Líneas":
                self._plot_line_chart(df_plot, var_principal, var_secundaria, color, x_label, y_label, x_min, x_max, y_min, y_max)
            elif plot_type == "Dispersión":
                self._plot_scatter_plot(df_plot, var_principal, var_secundaria, color, edge, False, x_label, y_label, x_min, x_max, y_min, y_max)
            elif plot_type == "Dispersión con Regresión":
                self._plot_scatter_plot(df_plot, var_principal, var_secundaria, color, edge, True, x_label, y_label, x_min, x_max, y_min, y_max)
            else:
                messagebox.showwarning("Aviso", f"Tipo de gráfico '{plot_type}' no implementado.")
                plt.close() 
                return

            if custom_title:
                plt.title(custom_title, color=font_color)
            else:
                default_title = f"{plot_type} de {var_principal}"
                if var_secundaria and plot_type not in ["Gráfico Circular"]: # No añadir "por var_secundaria" a pie chart
                    default_title += f" por {var_secundaria}"
                plt.title(default_title, color=font_color)
            
            if show_annotation:
                n_filtered = len(df_plot) 
                series_for_plot = df_plot[var_principal]
                if self.exclude_blank.get() and plot_type not in ["Líneas", "Dispersión", "Dispersión con Regresión"]:
                    series_for_plot = series_for_plot.dropna()
                
                n_valid = series_for_plot.count()

                # Modificado: Omitir filtros_info por ahora
                annotation_text = (f"Total filas (n): {n_filtered}" +
                                   f"\nValores válidos en '{var_principal}' (gráfico): {n_valid}")

                plt.annotate(annotation_text, xy=(0.98, 0.98), xycoords="figure fraction",
                             fontsize=font_size-2, ha="right", va="top", 
                             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7, edgecolor=font_color))

            plt.tight_layout(rect=[0, 0, 0.95, 0.95]) 

            self.graph_path = os.path.join(os.getcwd(), "temp_graph.png")
            plt.savefig(self.graph_path)
            plt.close() 
            self._show_graph_popup(var_principal, plot_type)

        except Exception as e:
            plt.close() 
            raise e 

    def get_filtered_data(self):
        """Devuelve el DataFrame filtrado."""
        if self.filtered_data is None and self.data is not None:
            # Si no se han aplicado filtros explícitamente, pero hay datos cargados,
            # se podría devolver el df original o el resultado de aplicar filtros vacíos.
            # Por ahora, si filtered_data es None, es que no se ha llamado a apply_filters
            # o falló.
            return None 
        return self.filtered_data


    # ---- Helper for generating summary text ----
    def _generate_summary(self, df: pd.DataFrame, var_principal: str, tipo_variable_principal: str):
        self.txt_result.delete("1.0", tk.END)
        summary_sections = []

        summary_sections.append(f"Resumen de {var_principal} (después de filtros):")
        summary_sections.append(f"Total de filas filtradas (n): {len(df)}")

        series_orig = df[var_principal]
        
        # Aplicar tipo de variable antes de las estadísticas
        series_processed = series_orig.copy()
        if tipo_variable_principal == "Cualitativa":
            series_processed = series_processed.astype(str)
        elif tipo_variable_principal == "Cuantitativa":
            series_processed = pd.to_numeric(series_processed, errors='coerce')

        series_for_stats = series_processed.dropna() if self.exclude_blank.get() else series_processed

        valid_count = series_for_stats.count()
        blanks_in_original_selection = series_orig.isnull().sum() # Blancos antes de cualquier procesamiento de tipo o exclusión
        
        summary_sections.append(f"Valores válidos en '{var_principal}' (considerando tipo y 'Excluir en blanco'): {valid_count}")
        
        if self.exclude_blank.get():
            # Cuántos NaNs fueron excluidos activamente
            # Esto es la diferencia entre los NaNs en la serie original (después de coerción si es cuantitativa)
            # y los NaNs en la serie final para estadísticas (que ya tiene dropna si exclude_blank es True)
            series_after_type_coercion = series_orig.copy()
            if tipo_variable_principal == "Cuantitativa":
                 series_after_type_coercion = pd.to_numeric(series_after_type_coercion, errors='coerce')
            
            original_nans = series_after_type_coercion.isnull().sum()
            remaining_nans = series_for_stats.isnull().sum() # Debería ser 0 si exclude_blank y dropna funcionaron
            excluded_nans = original_nans - remaining_nans
            if excluded_nans > 0 :
                 summary_sections.append(f"Casillas en blanco/inválidas en '{var_principal}' (fueron {original_nans}, ahora excluidas): {excluded_nans}")
        else: # No se excluyen blancos, mostrar los que hay después de coerción de tipo
            series_after_type_coercion = series_orig.copy()
            if tipo_variable_principal == "Cuantitativa":
                 series_after_type_coercion = pd.to_numeric(series_after_type_coercion, errors='coerce')
            current_blanks = series_after_type_coercion.isnull().sum()
            if current_blanks > 0:
                 summary_sections.append(f"Casillas en blanco/inválidas en '{var_principal}' (incluidas en estadísticas): {current_blanks}")


        try:
             desc = series_for_stats.describe()
             summary_sections.append("  Estadísticas (calculadas" + (" excluyendo blancos/inválidos):" if self.exclude_blank.get() else " incluyendo blancos/inválidos si existen):"))
             summary_sections.append(desc.to_string().replace('\n', '\n    ')) 
        except Exception:
             summary_sections.append("  No se pudieron calcular estadísticas descriptivas.")

        freq_series_for_display = series_for_stats
        if tipo_variable_principal == "Cualitativa" and not self.exclude_blank.get():
            # Si es cualitativa y se incluyen blancos, value_counts debe manejar NaNs como una categoría si pd.NA o similar
            # Para consistencia, si son NaNs de numpy, value_counts(dropna=False) los cuenta.
            # Si ya se convirtieron a 'nan' string, se contarán normalmente.
            # Si queremos que los NaNs originales (no los 'nan' string) se muestren, necesitamos series_processed
            freq_series_for_display = series_processed.value_counts(dropna=False) # Muestra NaNs si existen
        else: # Cuantitativa o Cualitativa con exclude_blank=True
            freq_series_for_display = series_for_stats.value_counts(dropna=True)


        n_obs_freq = freq_series_for_display.sum() 
        summary_sections.append("  Frecuencia de etiquetas (Top 10" + (", excluyendo blancos/inválidos" if self.exclude_blank.get() else ", incluyendo blancos/inválidos") + "):")
        if not freq_series_for_display.empty:
            freq_df = pd.DataFrame({
                "Count": freq_series_for_display,
                "Percentage": (freq_series_for_display / n_obs_freq * 100).round(2) if n_obs_freq > 0 else 0
            })

            custom_text = self.entry_etiquetas.get().strip()
            if custom_text: # Aplicar etiquetas y orden independientemente del tipo, si se proveen
                try:
                    order, mapping = [], {}
                    for item in custom_text.split(","):
                        if ":" in item:
                            k, v = item.split(":", 1)
                            order.append(k.strip())
                            mapping[k.strip()] = v.strip()
                        else:
                            order.append(item.strip())
                    
                    current_index_str = freq_df.index.astype(str)
                    # Mapear primero si hay un mapeo, luego ordenar
                    mapped_index = [mapping.get(str(i), str(i)) for i in freq_df.index]
                    freq_df.index = mapped_index
                    
                    # Para el orden, necesitamos que 'order' se refiera a los valores originales o a los mapeados
                    # Si el usuario pone "1:Hombre" en etiquetas, y el orden es "Hombre", debe funcionar.
                    # La forma más segura es reindexar usando los valores *antes* del mapeo de visualización,
                    # y luego aplicar el mapeo de visualización.
                    # O, si el orden se refiere a las etiquetas ya mapeadas:
                    
                    # Reconstruir freq_df con índice original para ordenar, luego mapear
                    temp_freq_df_for_reorder = pd.DataFrame({
                        "Count": freq_series_for_display.values
                    }, index=freq_series_for_display.index.astype(str))

                    # Crear el nuevo índice ordenado: primero los de 'order', luego el resto
                    # Asegurarse que los items en 'order' existan en el índice actual
                    ordered_idx_keys = [o for o in order if o in temp_freq_df_for_reorder.index]
                    remaining_idx_keys = [i for i in temp_freq_df_for_reorder.index if i not in ordered_idx_keys]
                    final_order_keys = ordered_idx_keys + remaining_idx_keys
                    
                    temp_freq_df_for_reorder = temp_freq_df_for_reorder.reindex(final_order_keys)
                    
                    # Ahora aplicar el mapeo al índice de este df reordenado
                    temp_freq_df_for_reorder.index = [mapping.get(str(i), str(i)) for i in temp_freq_df_for_reorder.index]
                    freq_df = temp_freq_df_for_reorder

                except Exception as e:
                    summary_sections.append(f"  Advertencia: Error al aplicar etiquetas/orden personalizados: {e}\n  Usando orden por defecto.")

            summary_sections.append(freq_df.head(10).to_string().replace('\n', '\n    ')) 
        else:
            summary_sections.append("    No hay valores válidos para contar.")


        self.txt_result.insert(tk.END, "\n\n".join([str(s) for s in summary_sections if s is not None]))


    # ---- Plotting methods for each type ----

    def _plot_histogram(self, df, var, orient_mat, color, edge, med_color, scale_mode, y_value_mode, x_label, y_label, x_min, x_max, y_min, y_max, rot):
        """Genera un histograma."""
        # df ya tiene var_principal convertida a numérico si tipo es Cuantitativa
        if not pd.api.types.is_numeric_dtype(df[var]):
            messagebox.showwarning("Aviso", f"El histograma requiere una variable numérica. '{var}' no lo es (o no pudo ser convertida).")
            return

        data_for_hist = df[var].dropna() if self.exclude_blank.get() else df[var] 
        if data_for_hist.empty:
             plt.text(0.5, 0.5, "No hay datos válidos para el histograma", horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
             return

        if y_value_mode == "Porcentaje":
            weights = np.ones_like(data_for_hist) / len(data_for_hist) * 100 if len(data_for_hist) > 0 else None
        else:
            weights = None

        plt.hist(data_for_hist, bins=20, weights=weights, color=color, edgecolor=edge, orientation=orient_mat)

        default_ylabel = "Porcentaje (%)" if y_value_mode == "Porcentaje" else "Frecuencia"
        if orient_mat == "horizontal":
            plt.xlabel(x_label or default_ylabel)
            plt.ylabel(y_label or var)
            plt.xscale(scale_mode)
            if pd.api.types.is_numeric_dtype(data_for_hist) and not data_for_hist.empty:
                mediana = data_for_hist.median()
                plt.axhline(mediana, color=med_color, linestyle="dashed", linewidth=2, label=f"Mediana: {mediana:.2f}")
                plt.legend()
            plt.setp(plt.gca().get_yticklabels(), rotation=rot) 
        else: 
            plt.xlabel(x_label or var)
            plt.ylabel(y_label or default_ylabel)
            plt.yscale(scale_mode)
            if pd.api.types.is_numeric_dtype(data_for_hist) and not data_for_hist.empty:
                mediana = data_for_hist.median()
                plt.axvline(mediana, color=med_color, linestyle="dashed", linewidth=2, label=f"Mediana: {mediana:.2f}")
                plt.legend()
            plt.setp(plt.gca().get_xticklabels(), rotation=rot) 

        self._apply_limits(x_min, x_max, y_min, y_max)


    def _plot_bar_chart(self, df, var, orient_mat, edge, y_value_mode, x_label, y_label, x_min, x_max, y_min, y_max, rot):
        """Genera un gráfico de barras (usando matplotlib)."""
        # df[var] ya es string si tipo_variable_principal es Cualitativa
        categorical_series = df[var].astype(str) # Asegurar que sea string para value_counts
        freq_series = categorical_series.value_counts(dropna=self.exclude_blank.get()) 

        if len(freq_series) == 0:
             plt.text(0.5, 0.5, "No hay datos válidos para el gráfico de barras", horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
             return

        n_obs_freq = freq_series.sum() 
        freq_df = pd.DataFrame({
            "Count": freq_series,
            "Percentage": (freq_series / n_obs_freq * 100).round(2) if n_obs_freq > 0 else 0
        })

        custom_text = self.entry_etiquetas.get().strip()
        if custom_text:
            try:
                order, mapping = [], {}
                for item in custom_text.split(","):
                    if ":" in item:
                        k, v = item.split(":", 1)
                        order.append(k.strip())
                        mapping[k.strip()] = v.strip()
                    else:
                        order.append(item.strip())
                
                temp_freq_df_for_reorder = pd.DataFrame({
                    "Count": freq_series.values,
                    "Percentage": (freq_series.values / n_obs_freq * 100).round(2) if n_obs_freq > 0 else 0
                }, index=freq_series.index.astype(str))

                ordered_idx_keys = [o for o in order if o in temp_freq_df_for_reorder.index]
                remaining_idx_keys = [i for i in temp_freq_df_for_reorder.index if i not in ordered_idx_keys]
                final_order_keys = ordered_idx_keys + remaining_idx_keys
                
                freq_df = temp_freq_df_for_reorder.reindex(final_order_keys)
                freq_df.index = [mapping.get(str(i), str(i)) for i in freq_df.index]
            except Exception as e:
                messagebox.showwarning("Advertencia", f"Error al aplicar etiquetas/orden personalizados para Bar Chart: {e}\nUsando orden por defecto.")


        bar_colors = self._parse_bar_colors(len(freq_df))
        values = freq_df["Percentage"] if y_value_mode == "Porcentaje" else freq_df["Count"]
        default_ylabel = "Porcentaje (%)" if y_value_mode == "Porcentaje" else "Frecuencia"

        if orient_mat == "horizontal":
            y_pos = np.arange(len(freq_df))
            plt.barh(y_pos, values, color=bar_colors, edgecolor=edge)
            plt.yticks(y_pos, freq_df.index, rotation=rot)
            plt.xlabel(x_label or default_ylabel)
            plt.ylabel(y_label or var)
        else: 
            plt.bar(freq_df.index, values, color=bar_colors, edgecolor=edge)
            plt.xlabel(x_label or var)
            plt.ylabel(y_label or default_ylabel)
            plt.setp(plt.gca().get_xticklabels(), rotation=rot)

        if orient_mat == "horizontal":
             plt.xscale(self.cmb_scale.get())
        else:
             plt.yscale(self.cmb_scale.get())

        self._apply_limits(x_min, x_max, y_min, y_max)

    def _plot_countplot(self, df, var_principal, var_secundaria, orient_mat, color, edge, y_value_mode, x_label, y_label, x_min, x_max, y_min, y_max, rot):
        """Genera un gráfico de conteo (usando seaborn)."""
        # df[var_principal] y df[var_secundaria] ya son string si tipo_variable_principal es Cualitativa
        
        # Countplot espera que la variable principal sea categórica. Si no lo es después del preprocesamiento, advertir.
        if pd.api.types.is_numeric_dtype(df[var_principal]) and self.cmb_tipo.get() == "Cualitativa":
             # Esto puede pasar si el usuario marca como cualitativa una columna puramente numérica
             # y la conversión a str en _generate_plot la hizo string. Seaborn la tratará como categórica.
             pass # Es válido si el usuario la quiere tratar como cualitativa
        elif pd.api.types.is_numeric_dtype(df[var_principal]): # Si es numérica y el tipo es Cuantitativa
             messagebox.showwarning("Aviso", f"El Countplot es más adecuado para variables cualitativas. '{var_principal}' es numérica.")
             # No retornamos, seaborn podría intentar graficarla de todas formas.

        data_for_plot = df.copy() # Usar el df ya preprocesado
        if self.exclude_blank.get():
             data_for_plot = data_for_plot.dropna(subset=[var_principal])
             if var_secundaria and var_secundaria in data_for_plot.columns: # También para la secundaria si existe
                 data_for_plot = data_for_plot.dropna(subset=[var_secundaria])


        if data_for_plot.empty or data_for_plot[var_principal].empty:
             plt.text(0.5, 0.5, "No hay datos válidos para el Countplot", horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
             return
        
        # Asegurar que la variable principal sea string para el countplot, incluso si originalmente era numérica pero se marcó cualitativa
        data_for_plot[var_principal] = data_for_plot[var_principal].astype(str)
        if var_secundaria and var_secundaria in data_for_plot.columns:
             data_for_plot[var_secundaria] = data_for_plot[var_secundaria].astype(str)


        custom_text = self.entry_etiquetas.get().strip()
        order = None
        # El mapeo de etiquetas para countplot se maneja mejor cambiando los datos antes de graficar
        # o usando los ticks, ya que 'order' solo afecta el orden, no las etiquetas.
        # Por simplicidad, si hay mapeo, aplicaremos el orden de las *claves originales*
        # y el usuario debe asegurarse que las etiquetas en el gráfico coincidan.
        # O, podríamos intentar remapear los datos en data_for_plot[var_principal] ANTES de llamar a sns.countplot
        
        final_mapping = {}
        if custom_text:
            try:
                parsed_order = []
                temp_mapping = {}
                for item in custom_text.split(","):
                    if ":" in item:
                        k, v = item.split(":", 1)
                        parsed_order.append(k.strip())
                        temp_mapping[k.strip()] = v.strip()
                    else:
                        parsed_order.append(item.strip())
                
                # Aplicar mapeo a los datos si es necesario
                if temp_mapping:
                    # Crear una copia para no modificar data_for_plot[var_principal] si se usa en otro lado
                    mapped_series = data_for_plot[var_principal].astype(str).map(temp_mapping).fillna(data_for_plot[var_principal].astype(str))
                    data_for_plot[var_principal] = mapped_series
                    # El orden ahora debe referirse a los valores mapeados
                    order = [temp_mapping.get(o, o) for o in parsed_order if temp_mapping.get(o,o) in data_for_plot[var_principal].unique()]
                    order += [v for v in data_for_plot[var_principal].unique() if v not in order]

                else: # Solo orden, sin mapeo
                    order = [o for o in parsed_order if o in data_for_plot[var_principal].unique()]
                    order += [v for v in data_for_plot[var_principal].unique() if v not in order]


            except Exception as e:
                messagebox.showwarning("Advertencia", f"Error al aplicar etiquetas/orden para Countplot: {e}\nUsando orden por defecto.")
                order = None 


        if orient_mat == "horizontal":
            sns.countplot(y=var_principal, data=data_for_plot, color=color, edgecolor=edge, order=order, hue=var_secundaria if var_secundaria else None, ax=plt.gca())
            plt.xlabel(x_label or "Count")
            plt.ylabel(y_label or var_principal)
            plt.setp(plt.gca().get_yticklabels(), rotation=rot)
        else: 
            sns.countplot(x=var_principal, data=data_for_plot, color=color, edgecolor=edge, order=order, hue=var_secundaria if var_secundaria else None, ax=plt.gca())
            plt.xlabel(x_label or var_principal)
            plt.ylabel(y_label or "Count")
            plt.setp(plt.gca().get_xticklabels(), rotation=rot)

        if orient_mat == "horizontal":
             plt.xscale(self.cmb_scale.get())
        else:
             plt.yscale(self.cmb_scale.get())

        self._apply_limits(x_min, x_max, y_min, y_max)


    def _plot_boxplot(self, df, var_principal, var_secundaria, orient_mat, color, edge, med_color, show_points, x_label, y_label, x_min, x_max, y_min, y_max, rot):
        """Genera un gráfico de caja y bigotes."""
        if not pd.api.types.is_numeric_dtype(df[var_principal]):
            messagebox.showwarning("Aviso", f"El gráfico de Caja y Bigotes requiere que la variable principal '{var_principal}' sea numérica.")
            return
       
        data_for_plot = df.copy()
        if self.exclude_blank.get():
            data_for_plot.dropna(subset=[var_principal], inplace=True)

        if orient_mat == "horizontal":
            x_sns, y_sns = var_principal, var_secundaria
        else: 
            x_sns, y_sns = var_secundaria, var_principal
       
        final_plot_cols = [col for col in [x_sns, y_sns] if col and col in data_for_plot.columns]
        if final_plot_cols:
           data_for_plot.dropna(subset=final_plot_cols, inplace=True)

        if data_for_plot.empty or data_for_plot[var_principal].empty:
             plt.text(0.5, 0.5, "No hay datos válidos para el gráfico de Caja y Bigotes", horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
             return

        if var_secundaria and var_secundaria in data_for_plot.columns:
             data_for_plot[var_secundaria] = data_for_plot[var_secundaria].astype(str) # Asegurar que la secundaria sea categórica
             sns.boxplot(x=x_sns, y=y_sns, data=data_for_plot, color=color, linewidth=1.5, fliersize=5, ax=plt.gca())
             if orient_mat == "horizontal": plt.setp(plt.gca().get_yticklabels(), rotation=rot)
             else: plt.setp(plt.gca().get_xticklabels(), rotation=rot)
             if show_points: sns.stripplot(x=x_sns, y=y_sns, data=data_for_plot, color=".3", size=3, jitter=True, ax=plt.gca())
        else:
            if orient_mat == "horizontal": 
                 sns.boxplot(x=data_for_plot[var_principal], color=color, linewidth=1.5, fliersize=5, ax=plt.gca())
                 plt.yticks([])
                 if show_points: sns.stripplot(x=data_for_plot[var_principal], color=".3", size=3, jitter=True, ax=plt.gca())
            else: 
                 sns.boxplot(y=data_for_plot[var_principal], color=color, linewidth=1.5, fliersize=5, ax=plt.gca())
                 plt.xticks([])
                 if show_points: sns.stripplot(y=data_for_plot[var_principal], color=".3", size=3, jitter=True, ax=plt.gca())

        if orient_mat == "horizontal":
             plt.xlabel(x_label or var_principal); plt.ylabel(y_label or var_secundaria)
             plt.xscale(self.cmb_scale.get())
        else: 
             plt.xlabel(x_label or var_secundaria); plt.ylabel(y_label or var_principal)
             plt.yscale(self.cmb_scale.get())
        self._apply_limits(x_min, x_max, y_min, y_max)


    def _plot_violinplot(self, df, var_principal, var_secundaria, orient_mat, color, edge, med_color, show_points, x_label, y_label, x_min, x_max, y_min, y_max, rot):
        """Genera un gráfico de violín."""
        if not pd.api.types.is_numeric_dtype(df[var_principal]):
            messagebox.showwarning("Aviso", f"El gráfico de Violín requiere que la variable principal '{var_principal}' sea numérica.")
            return

        data_for_plot = df.copy()
        if self.exclude_blank.get(): data_for_plot.dropna(subset=[var_principal], inplace=True)

        if orient_mat == "horizontal": x_sns, y_sns = var_principal, var_secundaria
        else: x_sns, y_sns = var_secundaria, var_principal

        final_plot_cols = [col for col in [x_sns, y_sns] if col and col in data_for_plot.columns]
        if final_plot_cols: data_for_plot.dropna(subset=final_plot_cols, inplace=True)
        
        if data_for_plot.empty or data_for_plot[var_principal].empty:
            plt.text(0.5,0.5, "No hay datos válidos para Violín", ha='center', va='center', transform=plt.gca().transAxes); return

        if var_secundaria and var_secundaria in data_for_plot.columns:
            data_for_plot[var_secundaria] = data_for_plot[var_secundaria].astype(str)
            sns.violinplot(x=x_sns, y=y_sns, data=data_for_plot, color=color, inner="quartile", ax=plt.gca())
            if orient_mat == "horizontal": plt.setp(plt.gca().get_yticklabels(), rotation=rot)
            else: plt.setp(plt.gca().get_xticklabels(), rotation=rot)
            if show_points: sns.stripplot(x=x_sns, y=y_sns, data=data_for_plot, color=".3", size=3, jitter=True, ax=plt.gca())
        else:
            if orient_mat == "horizontal":
                sns.violinplot(x=data_for_plot[var_principal], color=color, inner="quartile", ax=plt.gca()); plt.yticks([])
                if show_points: sns.stripplot(x=data_for_plot[var_principal], color=".3", size=3, jitter=True, ax=plt.gca())
            else:
                sns.violinplot(y=data_for_plot[var_principal], color=color, inner="quartile", ax=plt.gca()); plt.xticks([])
                if show_points: sns.stripplot(y=data_for_plot[var_principal], color=".3", size=3, jitter=True, ax=plt.gca())
        
        if orient_mat == "horizontal": plt.xlabel(x_label or var_principal); plt.ylabel(y_label or var_secundaria); plt.xscale(self.cmb_scale.get())
        else: plt.xlabel(x_label or var_secundaria); plt.ylabel(y_label or var_principal); plt.yscale(self.cmb_scale.get())
        self._apply_limits(x_min, x_max, y_min, y_max)


    def _plot_boxenplot(self, df, var_principal, var_secundaria, orient_mat, color, edge, show_points, x_label, y_label, x_min, x_max, y_min, y_max, rot):
        """Genera un gráfico Boxen (Letter Value Box Plot)."""
        if not pd.api.types.is_numeric_dtype(df[var_principal]):
            messagebox.showwarning("Aviso", f"El gráfico Boxen requiere que la variable principal '{var_principal}' sea numérica.")
            return

        data_for_plot = df.copy()
        if self.exclude_blank.get(): data_for_plot.dropna(subset=[var_principal], inplace=True)

        if orient_mat == "horizontal": x_sns, y_sns = var_principal, var_secundaria
        else: x_sns, y_sns = var_secundaria, var_principal
       
        final_plot_cols = [col for col in [x_sns, y_sns] if col and col in data_for_plot.columns]
        if final_plot_cols: data_for_plot.dropna(subset=final_plot_cols, inplace=True)

        if data_for_plot.empty or data_for_plot[var_principal].empty:
            plt.text(0.5,0.5, "No hay datos válidos para Boxen", ha='center', va='center', transform=plt.gca().transAxes); return

        if var_secundaria and var_secundaria in data_for_plot.columns:
            data_for_plot[var_secundaria] = data_for_plot[var_secundaria].astype(str)
            sns.boxenplot(x=x_sns, y=y_sns, data=data_for_plot, color=color, linewidth=1.5, ax=plt.gca())
            if orient_mat == "horizontal": plt.setp(plt.gca().get_yticklabels(), rotation=rot)
            else: plt.setp(plt.gca().get_xticklabels(), rotation=rot)
            if show_points: sns.stripplot(x=x_sns, y=y_sns, data=data_for_plot, color=".3", size=3, jitter=True, ax=plt.gca())
        else:
            if orient_mat == "horizontal":
                sns.boxenplot(x=data_for_plot[var_principal], color=color, linewidth=1.5, ax=plt.gca()); plt.yticks([])
                if show_points: sns.stripplot(x=data_for_plot[var_principal], color=".3", size=3, jitter=True, ax=plt.gca())
            else:
                sns.boxenplot(y=data_for_plot[var_principal], color=color, linewidth=1.5, ax=plt.gca()); plt.xticks([])
                if show_points: sns.stripplot(y=data_for_plot[var_principal], color=".3", size=3, jitter=True, ax=plt.gca())

        if orient_mat == "horizontal": plt.xlabel(x_label or var_principal); plt.ylabel(y_label or var_secundaria); plt.xscale(self.cmb_scale.get())
        else: plt.xlabel(x_label or var_secundaria); plt.ylabel(y_label or var_principal); plt.yscale(self.cmb_scale.get())
        self._apply_limits(x_min, x_max, y_min, y_max)


    def _plot_swarmplot(self, df, var_principal, var_secundaria, orient_mat, color, edge, x_label, y_label, x_min, x_max, y_min, y_max, rot):
        """Genera un gráfico Swarm (puntos individuales sin solapamiento)."""
        if not pd.api.types.is_numeric_dtype(df[var_principal]):
            messagebox.showwarning("Aviso", f"El gráfico Swarm requiere que la variable principal '{var_principal}' sea numérica.")
            return
        if not var_secundaria or var_secundaria not in df.columns:
            messagebox.showwarning("Aviso", "El gráfico Swarm requiere seleccionar una Segunda Variable (Grupo/Categoría).")
            return

        data_for_plot = df.copy()
        if self.exclude_blank.get(): data_for_plot.dropna(subset=[var_principal], inplace=True)
       
        plot_cols_to_clean = [var_principal]
        if var_secundaria and var_secundaria in data_for_plot.columns:
             plot_cols_to_clean.append(var_secundaria)
        data_for_plot.dropna(subset=plot_cols_to_clean, inplace=True)


        if data_for_plot.empty or data_for_plot[var_principal].empty or data_for_plot[var_secundaria].empty :
             plt.text(0.5, 0.5, "No hay datos válidos para el gráfico Swarm", ha='center', va='center', transform=plt.gca().transAxes); return

        data_for_plot[var_secundaria] = data_for_plot[var_secundaria].astype(str)

        if orient_mat == "horizontal": x_sns, y_sns = var_principal, var_secundaria
        else: x_sns, y_sns = var_secundaria, var_principal

        sns.swarmplot(x=x_sns, y=y_sns, data=data_for_plot, color=color, edgecolor=edge, linewidth=0.5, ax=plt.gca())

        if orient_mat == "horizontal": plt.setp(plt.gca().get_yticklabels(), rotation=rot)
        else: plt.setp(plt.gca().get_xticklabels(), rotation=rot)

        if orient_mat == "horizontal": plt.xlabel(x_label or var_principal); plt.ylabel(y_label or var_secundaria); plt.xscale(self.cmb_scale.get())
        else: plt.xlabel(x_label or var_secundaria); plt.ylabel(y_label or var_principal); plt.yscale(self.cmb_scale.get())
        self._apply_limits(x_min, x_max, y_min, y_max)


    def _plot_kdeplot(self, df, var_principal, orient_mat, color, scale_mode, x_label, y_label, x_min, x_max, y_min, y_max):
        """Genera un gráfico de densidad (KDE)."""
        if not pd.api.types.is_numeric_dtype(df[var_principal]):
            messagebox.showwarning("Aviso", f"El gráfico KDE requiere una variable numérica. '{var_principal}' no lo es.")
            return

        data = df[var_principal].dropna() if self.exclude_blank.get() else df[var_principal] 
        if data.empty: plt.text(0.5,0.5, "No hay datos válidos para KDE", ha='center', va='center', transform=plt.gca().transAxes); return
        if len(data.unique()) < 2: plt.text(0.5,0.5, "Datos insuficientes/constantes para KDE", ha='center', va='center', transform=plt.gca().transAxes); return

        if orient_mat == "horizontal":
             sns.kdeplot(y=data, color=color, fill=True, ax=plt.gca())
             plt.xlabel(x_label or "Densidad"); plt.ylabel(y_label or var_principal)
             plt.xscale(scale_mode) 
        else: 
             sns.kdeplot(x=data, color=color, fill=True, ax=plt.gca())
             plt.xlabel(x_label or var_principal); plt.ylabel(y_label or "Densidad")
             plt.yscale(scale_mode) 
        self._apply_limits(x_min, x_max, y_min, y_max)

    def _plot_histplot_kde(self, df, var_principal, orient_mat, color, edge, scale_mode, y_value_mode, x_label, y_label, x_min, x_max, y_min, y_max):
        """Genera un Histplot (Histograma con KDE opcional)."""
        if not pd.api.types.is_numeric_dtype(df[var_principal]):
            messagebox.showwarning("Aviso", f"El Histplot requiere una variable numérica. '{var_principal}' no lo es.")
            return

        data = df[var_principal].dropna() if self.exclude_blank.get() else df[var_principal] 
        if data.empty: plt.text(0.5,0.5, "No hay datos válidos para Histplot", ha='center', va='center', transform=plt.gca().transAxes); return
        plot_kde = len(data.unique()) >= 2 

        if orient_mat == "horizontal":
             sns.histplot(y=data, color=color, edgecolor=edge, kde=plot_kde, stat=y_value_mode.lower(), ax=plt.gca())
             plt.xlabel(x_label or y_value_mode); plt.ylabel(y_label or var_principal)
             plt.xscale(scale_mode) 
        else: 
             sns.histplot(x=data, color=color, edgecolor=edge, kde=plot_kde, stat=y_value_mode.lower(), ax=plt.gca())
             plt.xlabel(x_label or var_principal); plt.ylabel(y_label or y_value_mode) 
             plt.yscale(scale_mode) 
        self._apply_limits(x_min, x_max, y_min, y_max)

    def _plot_pie_chart(self, df, var_principal, edge, x_label, y_label):
        """Genera un gráfico circular (Pie Chart)."""
        # La variable principal ya debería ser string si tipo_variable_principal es Cualitativa
        # o si el usuario la marcó como cualitativa.
        
        series_for_pie = df[var_principal].astype(str) # Asegurar que sea string para value_counts
        if self.exclude_blank.get():
            series_for_pie = series_for_pie[series_for_pie.notna() & (series_for_pie.str.strip() != '')]


        freq_series = series_for_pie.value_counts(dropna=self.exclude_blank.get())

        if freq_series.empty:
            plt.text(0.5, 0.5, "No hay datos válidos para el Gráfico Circular", ha='center', va='center', transform=plt.gca().transAxes)
            return

        labels = freq_series.index
        sizes = freq_series.values
        
        custom_text = self.entry_etiquetas.get().strip()
        if custom_text:
            try:
                order_keys, mapping = [], {}
                for item in custom_text.split(","):
                    if ":" in item:
                        k, v = item.split(":", 1)
                        order_keys.append(k.strip())
                        mapping[k.strip()] = v.strip()
                    else:
                        order_keys.append(item.strip())

                # Crear un DataFrame temporal para facilitar el ordenamiento y mapeo
                temp_df = pd.DataFrame({'sizes': sizes, 'original_labels': labels}, index=labels.astype(str))
                
                # Aplicar mapeo a las etiquetas originales para el orden
                # El orden se basa en las claves originales
                current_original_labels = temp_df.index.tolist()
                
                final_ordered_keys = [ok for ok in order_keys if ok in current_original_labels]
                remaining_keys = [lbl for lbl in current_original_labels if lbl not in final_ordered_keys]
                
                final_order_for_reindex = final_ordered_keys + remaining_keys
                temp_df = temp_df.reindex(final_order_for_reindex)
                
                # Aplicar el mapeo para las etiquetas finales del gráfico
                final_labels = [mapping.get(str(orig_label), str(orig_label)) for orig_label in temp_df.index]
                sizes = temp_df['sizes'].values
                labels = final_labels

            except Exception as e:
                messagebox.showwarning("Advertencia", f"Error al aplicar etiquetas/orden para Gráfico Circular: {e}\nUsando orden y etiquetas por defecto.")
                # Revertir a labels y sizes originales si hay error
                labels = freq_series.index
                sizes = freq_series.values


        pie_colors = self._parse_bar_colors(len(labels)) # Reutilizar la lógica de colores de barras

        plt.pie(sizes, labels=labels, colors=pie_colors, autopct='%1.1f%%', startangle=90, wedgeprops={'edgecolor': edge})
        plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        # No se aplican x_label, y_label, x_min, x_max, y_min, y_max directamente a pie charts


    def _plot_line_chart(self, df, var_principal, var_secundaria, color, x_label, y_label, x_min, x_max, y_min, y_max):
        """Genera un gráfico de líneas."""
        # df ya tiene var_principal y var_secundaria convertidas a numérico si tipo es Cuantitativa
        if not (var_secundaria and var_secundaria in df.columns):
            messagebox.showwarning("Aviso", "El gráfico de Líneas requiere seleccionar una Segunda Variable (Eje Y).")
            return
        if not pd.api.types.is_numeric_dtype(df[var_principal]) or not pd.api.types.is_numeric_dtype(df[var_secundaria]):
            messagebox.showwarning("Aviso", f"El gráfico de Líneas requiere que ambas variables ('{var_principal}' y '{var_secundaria}') sean numéricas.")
            return

        data_for_plot = df.copy()
        if self.exclude_blank.get(): # Excluir basado en var_principal primero
            data_for_plot.dropna(subset=[var_principal], inplace=True)
        # Luego, para líneas, ambas variables deben ser no-NaN para un punto
        data_for_plot.dropna(subset=[var_principal, var_secundaria], inplace=True)


        if data_for_plot.empty:
             plt.text(0.5, 0.5, "No hay datos válidos para el gráfico de Líneas", ha='center', va='center', transform=plt.gca().transAxes); return

        data_for_plot = data_for_plot.sort_values(by=var_principal)
        plt.plot(data_for_plot[var_principal], data_for_plot[var_secundaria], color=color)
        plt.xlabel(x_label or var_principal); plt.ylabel(y_label or var_secundaria)
        plt.xscale(self.cmb_scale.get()) 
        self._apply_limits(x_min, x_max, y_min, y_max) 


    def _plot_scatter_plot(self, df, var_principal, var_secundaria, color, edge, include_regression, x_label, y_label, x_min, x_max, y_min, y_max):
        """Genera un gráfico de dispersión (con o sin regresión)."""
        if not (var_secundaria and var_secundaria in df.columns):
            messagebox.showwarning("Aviso", "El gráfico de Dispersión requiere seleccionar una Segunda Variable (Eje Y).")
            return
        if not pd.api.types.is_numeric_dtype(df[var_principal]) or not pd.api.types.is_numeric_dtype(df[var_secundaria]):
            messagebox.showwarning("Aviso", f"El gráfico de Dispersión requiere que ambas variables ('{var_principal}' y '{var_secundaria}') sean numéricas.")
            return

        data_for_plot = df.copy()
        if self.exclude_blank.get(): data_for_plot.dropna(subset=[var_principal], inplace=True)
        data_for_plot.dropna(subset=[var_principal, var_secundaria], inplace=True)

        if data_for_plot.empty:
             plt.text(0.5, 0.5, "No hay datos válidos para el gráfico de Dispersión", ha='center', va='center', transform=plt.gca().transAxes); return

        if include_regression:
             sns.regplot(x=var_principal, y=var_secundaria, data=data_for_plot, color=color, scatter_kws={'edgecolor': edge}, ax=plt.gca())
        else:
             plt.scatter(data_for_plot[var_principal], data_for_plot[var_secundaria], color=color, edgecolor=edge)

        plt.xlabel(x_label or var_principal); plt.ylabel(y_label or var_secundaria)
        plt.xscale(self.cmb_scale.get()) 
        self._apply_limits(x_min, x_max, y_min, y_max) 


    # ---- helper límites ----
    def _apply_limits(self, x_min: str, x_max: str, y_min: str, y_max: str):
        """Aplica límites a los ejes X e Y."""
        try:
            xmin_val = float(x_min) if x_min else None
            xmax_val = float(x_max) if x_max else None
            ymin_val = float(y_min) if y_min else None
            ymax_val = float(y_max) if y_max else None

            current_xlim = plt.xlim()
            current_ylim = plt.ylim()

            plt.xlim(xmin_val if xmin_val is not None else current_xlim[0],
                     xmax_val if xmax_val is not None else current_xlim[1])
            plt.ylim(ymin_val if ymin_val is not None else current_ylim[0],
                     ymax_val if ymax_val is not None else current_ylim[1])

        except ValueError as e:
            messagebox.showwarning("Advertencia", f"Límites de eje inválidos: {e}\nIgnorando límites manuales.")
        except Exception as e:
             messagebox.showwarning("Advertencia", f"Error al aplicar límites de eje: {e}\nIgnorando límites manuales.")


    # ---- helper colores barras cualitativas ----
    def _parse_bar_colors(self, n: int):
        """Parsea la cadena de colores para barras y devuelve una lista."""
        txt = self.entry_bar_colors.get().strip()
        default_color = self.cmb_color.get()
        if not txt:
            return [default_color] * n
        
        cols_input = [c.strip() for c in txt.split(",") if c.strip()]
        if not cols_input:
             return [default_color] * n

        # Validar que los colores sean reconocibles por matplotlib
        valid_colors = []
        for c_name in cols_input:
            try:
                mpl.colors.to_rgb(c_name) # Intenta convertir a RGB, falla si no es válido
                valid_colors.append(c_name)
            except ValueError:
                # Si no es un nombre de color válido, usar el color por defecto para esa posición
                valid_colors.append(default_color) 
                print(f"Advertencia: Color '{c_name}' no reconocido, usando color por defecto '{default_color}'.")


        from itertools import cycle, islice
        return list(islice(cycle(valid_colors if valid_colors else [default_color]), n))

    # ---- popup imagen ----
    def _show_graph_popup(self, var, plot_type):
        """Muestra la gráfica generada en una ventana emergente."""
        win = tk.Toplevel(self)
        win.title(f"{plot_type} de {var}")
        try:
            img = tk.PhotoImage(file=self.graph_path)
            lbl_img = tk.Label(win, image=img)
            lbl_img.image = img 
            lbl_img.pack()
        except Exception as e:
            ttk.Label(win, text=f"Error al mostrar la imagen:\n{e}").pack()
            messagebox.showerror("Error", f"No se pudo mostrar la gráfica:\n{e}")

        ttk.Button(win, text="Guardar Gráfica", command=lambda: self._save_graph_from_popup(win)).pack(pady=5)


    def _save_graph_from_popup(self, popup_window):
         """Saves the graph displayed in the popup."""
         if not hasattr(self, "graph_path") or not os.path.exists(self.graph_path):
             messagebox.showwarning("Aviso", "No hay gráfica generada para guardar.")
             return
         dest = filedialog.asksaveasfilename(initialfile="grafica.png", defaultextension=".png",
                                             filetypes=[("PNG files", "*.png")])
         if dest:
             try:
                 import shutil; shutil.copy2(self.graph_path, dest)
                 messagebox.showinfo("Guardado", f"Gráfica guardada en: {dest}")
                 popup_window.destroy() 
             except Exception as e:
                 messagebox.showerror("Error", f"No se pudo guardar la gráfica: {e}")


    # ---------------------------------------------------------------------
    # Guardar (directamente sin ver)
    # ---------------------------------------------------------------------
    def save_graph_directly(self):
        """Guarda la última gráfica generada sin mostrarla."""
        if not hasattr(self, "graph_path") or not os.path.exists(self.graph_path):
            messagebox.showwarning("Aviso", "No hay gráfica generada para guardar.")
            return
        dest = filedialog.asksaveasfilename(initialfile="grafica.png", defaultextension=".png",
                                            filetypes=[("PNG files", "*.png")])
        if dest:
            try:
                import shutil; shutil.copy2(self.graph_path, dest)
                messagebox.showinfo("Guardado", f"Gráfica guardada en: {dest}")
            except Exception as e:
                messagebox.showerror("Error", f"No se pudo guardar la gráfica: {e}")


# --- Pestaña de Scripts Externos ---
class ExternalScriptsTab(ttk.Frame):
    """Pestaña para cargar y ejecutar scripts externos (plugins)."""

    def __init__(self, master, data_tab_instance):
        super().__init__(master)
        self.data_tab = data_tab_instance 
        self.plugins_dir = "plugins" 
        self.available_plugins = {} 

        lbl_title = ttk.Label(self, text="Ejecutar Scripts Externos (Plugins)", font=("Helvetica", 14))
        lbl_title.pack(pady=10)

        frm_plugins = ttk.LabelFrame(self, text="Plugins Disponibles")
        frm_plugins.pack(padx=10, pady=5, fill="x")

        self.listbox_plugins = tk.Listbox(frm_plugins, height=10)
        self.listbox_plugins.pack(side="left", fill="both", expand=True, padx=5, pady=5)
        scrollbar = ttk.Scrollbar(frm_plugins, orient="vertical", command=self.listbox_plugins.yview)
        scrollbar.pack(side="right", fill="y")
        self.listbox_plugins.config(yscrollcommand=scrollbar.set)

        ttk.Button(self, text="Recargar Plugins", command=self.load_plugins).pack(pady=5)
        ttk.Button(self, text="Ejecutar Plugin Seleccionado", command=self.run_selected_plugin).pack(pady=5)

        frm_output = ttk.LabelFrame(self, text="Salida del Script")
        frm_output.pack(padx=10, pady=5, fill="both", expand=True)

        self.txt_output = tk.Text(frm_output, height=15)
        self.txt_output.pack(fill="both", expand=True, padx=5, pady=5)

        self.load_plugins()

    def load_plugins(self):
        """Descubre y carga clases de plugins de la carpeta 'plugins'."""
        self.available_plugins = {}
        self.listbox_plugins.delete(0, tk.END)
        self.txt_output.delete("1.0", tk.END)
        self.txt_output.insert(tk.END, f"Buscando plugins en: {os.path.abspath(self.plugins_dir)}\n")

        if not os.path.exists(self.plugins_dir):
            self.txt_output.insert(tk.END, f"Error: Directorio '{self.plugins_dir}' no encontrado.\n")
            return

        sys.path.insert(0, self.plugins_dir)

        for filename in os.listdir(self.plugins_dir):
            if filename.endswith(".py") and filename != "__init__.py":
                module_name = filename[:-3] 
                try:
                    if module_name in sys.modules:
                         importlib.reload(sys.modules[module_name])
                         module = sys.modules[module_name]
                    else:
                         module = importlib.import_module(module_name)

                    for item_name in dir(module):
                        item = getattr(module, item_name)
                        if isinstance(item, type) and hasattr(item, 'name') and hasattr(item, 'run'):
                            self.available_plugins[item.name] = item
                            self.listbox_plugins.insert(tk.END, item.name)
                            self.txt_output.insert(tk.END, f"Plugin '{item.name}' cargado desde {filename}\n")

                except Exception as e:
                    self.txt_output.insert(tk.END, f"Error cargando plugin {filename}: {e}\n")
                    traceback.print_exc() 

        sys.path.pop(0)

        if not self.available_plugins:
             self.txt_output.insert(tk.END, "No se encontraron plugins válidos.\n")

    def run_selected_plugin(self):
        """Ejecuta el plugin seleccionado en la lista."""
        selected_indices = self.listbox_plugins.curselection()
        if not selected_indices:
            messagebox.showwarning("Aviso", "Selecciona un plugin para ejecutar.")
            return

        selected_name = self.listbox_plugins.get(selected_indices[0])
        plugin_class = self.available_plugins.get(selected_name)

        if not plugin_class:
            messagebox.showerror("Error", f"Plugin '{selected_name}' no encontrado o no cargado correctamente.")
            return

        df = self.data_tab.get_filtered_data()

        if df is None:
            messagebox.showwarning("Aviso", "No hay datos cargados o no se pudieron aplicar los filtros.")
            self.txt_output.insert(tk.END, "\nNo hay datos cargados o filtrados para el plugin.\n")
            return

        old_stdout = sys.stdout
        old_stderr = sys.stderr
        redirected_output = io.StringIO()
        sys.stdout = redirected_output
        sys.stderr = redirected_output

        self.txt_output.insert(tk.END, f"\n--- Ejecutando '{selected_name}' ---\n")
        try:
            plugin_instance = plugin_class()
            plugin_instance.run(df, self.data_tab)
            self.txt_output.insert(tk.END, "\n--- Ejecución completada ---\n")
        except Exception as e:
            self.txt_output.insert(tk.END, f"\n--- Error durante la ejecución de '{selected_name}' ---\n")
            traceback.print_exc(file=redirected_output)
            self.txt_output.insert(tk.END, "\n--- Ejecución con errores ---\n")
            messagebox.showerror("Error de Plugin", f"Ocurrió un error al ejecutar el plugin:\n{e}")
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            output_text = redirected_output.getvalue()
            self.txt_output.insert(tk.END, output_text)
            self.txt_output.see(tk.END) 


# Export explícito para importaciones (aunque no se usan aquí directamente)
__all__ = ["DataFilterTab", "ExternalScriptsTab", "BasePlugin"]

# Se eliminan las funciones auxiliares de filtro: _build_extra_filters, _add_filter_row, _parse_filter_values
# Se elimina la función auxiliar _update_variable_comboboxes (reemplazada por _update_analysis_variable_comboboxes)

# -------------------------------------------------------------------------
# Ejecución directa
# -------------------------------------------------------------------------
if __name__ == "__main__":
    root = tk.Tk()
    root.title("Aplicación de Análisis de Datos con Plugins")

    notebook = ttk.Notebook(root)
    notebook.pack(fill=tk.BOTH, expand=True)

    data_tab_instance = DataFilterTab(notebook)
    external_scripts_tab_instance = ExternalScriptsTab(notebook, data_tab_instance)

    notebook.add(data_tab_instance, text="Filtros/Estadísticas/Gráficos")
    notebook.add(external_scripts_tab_instance, text="Scripts Externos")

    root.mainloop()