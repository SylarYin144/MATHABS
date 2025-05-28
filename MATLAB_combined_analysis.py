import tkinter as tk
from tkinter import ttk, filedialog, messagebox, StringVar
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns
import os
import csv
import numpy as np
import traceback
import matplotlib as mpl # Importar para colores
from itertools import cycle, islice # Para colores de barras

# Importar el componente de filtro
from MATLAB_filter_component import FilterComponent

class CombinedAnalysisTab(ttk.Frame):
    def __init__(self, master, main_app_instance=None):
        super().__init__(master)
        self.main_app = main_app_instance
        self.data = None # DataFrame cargado
        self.filtered_data = None # DataFrame después de aplicar filtros
        self.graph_path = None # Ruta de la última gráfica generada
        self.current_fig = None # Para guardar la figura actual

        # Colores disponibles (de DataFilterTab)
        self.color_options = [
            "blue", "green", "red", "skyblue", "orange", "purple", "black", "gray", "brown", "pink",
            "gold", "olive", "cyan", "magenta", "navy", "teal", "aqua", "maroon", "lime", "silver",
            "salmon", "plum", "turquoise", "coral", "indigo", "khaki", "lavender", "peru", "tomato", "chocolate",
            "darkgreen", "darkred", "darkblue", "darkorange", "lightblue", "lightgreen", "lightcoral",
            "violet", "orchid", "royalblue"
        ]

        # --- Layout principal ---
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

        # Frame derecho = resultados y gráficos
        self.right_frame = ttk.Frame(self.paned)
        self.paned.add(self.right_frame, weight=2)

        # --- Controles en el frame izquierdo ---
        self._build_controls()

        # --- Área de resultados y gráficos en el frame derecho ---
        self._build_display_area()

        self.log("Pestaña de Análisis Combinado inicializada.", "INFO")

    def _build_controls(self):
        # Sección de Carga de Datos
        frm_data_load = ttk.LabelFrame(self.scrollable_frame, text="Carga de Datos", padding="10")
        frm_data_load.pack(padx=10, pady=5, fill="x")
        ttk.Button(frm_data_load, text="Cargar Datos (Excel/CSV)", command=self.load_data).pack(pady=5)
        self.lbl_file = ttk.Label(frm_data_load, text="Ningún archivo cargado.")
        self.lbl_file.pack(pady=5)

        # Sección de Filtros Avanzados (usando FilterComponent)
        frm_filters_general = ttk.LabelFrame(self.scrollable_frame, text="Filtros Avanzados", padding="10")
        frm_filters_general.pack(padx=10, pady=5, fill="x")
        self.filter_component = FilterComponent(frm_filters_general)
        self.filter_component.pack(fill="x", expand=True, padx=5, pady=5)

        # Sección de Variable Principal y Opciones de Análisis
        frm_analysis_var = ttk.LabelFrame(self.scrollable_frame, text="Variable Principal y Opciones de Análisis", padding="10")
        frm_analysis_var.pack(padx=10, pady=5, fill="x")
        ttk.Label(frm_analysis_var, text="Variable Principal para Análisis:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.cmb_variables = ttk.Combobox(frm_analysis_var, state="readonly")
        self.cmb_variables.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        ttk.Label(frm_analysis_var, text="Tipo de Variable Principal:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.cmb_tipo = ttk.Combobox(frm_analysis_var, values=["Cuantitativa", "Cualitativa"], state="readonly")
        self.cmb_tipo.grid(row=1, column=1, padx=5, pady=5, sticky="ew")
        self.cmb_tipo.set("Cuantitativa")
        ttk.Label(frm_analysis_var, text="Etiquetas y orden (para cualitativa):").grid(row=2, column=0, padx=5, pady=5, sticky="w")
        self.entry_etiquetas = ttk.Entry(frm_analysis_var)
        self.entry_etiquetas.grid(row=2, column=1, padx=5, pady=5, sticky="ew")
        ttk.Label(frm_analysis_var, text="(Ej: 2:Alto,1:Medio,0:Bajo)").grid(row=3, column=0, columnspan=2, padx=5, pady=2, sticky="w")
        self.exclude_blank = tk.BooleanVar(value=False)
        ttk.Checkbutton(frm_analysis_var, text="Excluir blancos/NaNs de la Variable Principal en Estadísticas/Gráficos", variable=self.exclude_blank).grid(row=4, column=0, columnspan=2, padx=5, pady=5, sticky="w")
        frm_analysis_var.columnconfigure(1, weight=1)

        # Sección de Parámetros de Gráfica (combinando opciones de ambos)
        self._build_graph_params()

        # Botones de acción
        ttk.Button(self.scrollable_frame, text="Aplicar Filtros y Generar Análisis/Gráfico", command=self.apply_filters_and_generate_chart).pack(pady=10, fill="x")
        ttk.Button(self.scrollable_frame, text="Guardar Gráfica", command=self.save_current_graph).pack(pady=5, fill="x")

    def _build_graph_params(self):
        frm_graph = ttk.LabelFrame(self.scrollable_frame, text="Parámetros de Gráfica", padding="10")
        frm_graph.pack(padx=10, pady=5, fill="x")

        # Tipo de Gráfico (combinado de ambos)
        ttk.Label(frm_graph, text="Tipo de Gráfico:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.chart_type_var = StringVar()
        self.chart_types = [
            "Histograma", "Gráfico de Barras", "Countplot",
            "Caja y Bigotes", "Violín", "Boxen Plot", "Swarm Plot",
            "KDE Plot", "Histplot (Hist+KDE)", "Gráfico Circular",
            "Líneas", "Dispersión", "Dispersión con Regresión",
            # Gráficos de GeneralChartsApp que requieren más implementación
            "Trimap (Mosaico Jerárquico)", "Gráficos Q-Q", "Gráfico de Densidad Suave",
            "Curvas de Kaplan-Meier", "Mapas de Calor", "Polígonos de Frecuencia",
            "Diagrama de Flujo"
        ]
        self.cmb_plot_type = ttk.Combobox(frm_graph, textvariable=self.chart_type_var, values=self.chart_types, state="readonly")
        self.cmb_plot_type.grid(row=0, column=1, padx=5, pady=5, sticky="ew", columnspan=2)
        self.cmb_plot_type.set("Histograma") # Default
        self.cmb_plot_type.bind("<<ComboboxSelected>>", self._update_chart_specific_params)

        # Segunda Variable (para gráficos 2D o agrupados)
        ttk.Label(frm_graph, text="Segunda Variable (Eje Y/Grupo/Color):").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.cmb_variable_2d = ttk.Combobox(frm_graph, state="readonly")
        self.cmb_variable_2d.grid(row=1, column=1, padx=5, pady=5, sticky="ew", columnspan=2)

        # Controles específicos de gráficos (se actualizarán dinámicamente)
        self.chart_specific_params_frame = ttk.Frame(frm_graph)
        self.chart_specific_params_frame.grid(row=2, column=0, columnspan=4, sticky="ew")
        self._update_chart_specific_params() # Inicializar

        frm_graph.columnconfigure(1, weight=1) # Hacer que combobox/entry se expandan

    def _update_chart_specific_params(self, event=None):
        # Limpiar controles anteriores
        for widget in self.chart_specific_params_frame.winfo_children():
            widget.destroy()

        chart_type = self.chart_type_var.get()
        current_row = 0

        # Controles comunes a muchos gráficos (de DataFilterTab)
        # Mostrar puntos individuales (para Box/Violin/Boxen)
        self.show_individual_points = tk.BooleanVar(value=False)
        ttk.Checkbutton(self.chart_specific_params_frame, text="Mostrar puntos individuales", variable=self.show_individual_points).grid(row=current_row, column=0, padx=5, pady=5, sticky="w")
        
        # Mostrar Reja (grid)
        self.show_grid = tk.BooleanVar(value=False)
        ttk.Checkbutton(self.chart_specific_params_frame, text="Mostrar reja (grid)", variable=self.show_grid).grid(row=current_row, column=1, padx=5, pady=5, sticky="w")
        
        # Mostrar Anotación (filtros/n)
        self.show_annotation = tk.BooleanVar(value=True)
        ttk.Checkbutton(self.chart_specific_params_frame, text="Mostrar anotación (filtros/n)", variable=self.show_annotation).grid(row=current_row, column=2, padx=5, pady=5, sticky="w")
        current_row += 1

        # Tamaño / DPI
        self.entry_dpi, self.entry_width, self.entry_height = self._add_size_controls(self.chart_specific_params_frame, start_row=current_row)
        current_row += 1

        # Colores principales
        self.cmb_color, self.cmb_edge = self._add_color_controls(self.chart_specific_params_frame, start_row=current_row)
        current_row += 1

        # Orientación y rotación
        self.cmb_orient, self.cmb_tickrot = self._add_orientation_controls(self.chart_specific_params_frame, start_row=current_row)
        current_row += 1

        # Mediana y colores de barras
        self.cmb_mediana, self.entry_bar_colors = self._add_bar_controls(self.chart_specific_params_frame, start_row=current_row)
        current_row += 1

        # Texto (título, ejes)
        self.entry_custom_title, self.entry_x_label, self.entry_y_label = self._add_text_controls(self.chart_specific_params_frame, start_row=current_row)
        current_row += 3 # 3 filas para los labels

        # Escala / eje Y-X
        self.cmb_scale, self.cmb_y_value = self._add_scale_controls(self.chart_specific_params_frame, start_row=current_row)
        current_row += 2 # 2 filas para los labels

        # Límites
        (self.entry_xmin, self.entry_xmax,
         self.entry_ymin, self.entry_ymax) = self._add_limit_controls(self.chart_specific_params_frame, start_row=current_row)
        current_row += 2 # 2 filas para los labels

        # Fuente
        self.cmb_font_color, self.entry_font_size = self._add_font_controls(self.chart_specific_params_frame, start_row=current_row)
        current_row += 1

        # Controles específicos para ciertos tipos de gráficos (de GeneralChartsApp)
        if chart_type == "Diagrama de Dispersión":
            # Ya se manejan con cmb_variable_2d para Y, pero podríamos añadir más opciones de Plotly
            pass
        elif chart_type == "Gráfico de Pastel":
            # Ya se manejan con cmb_variables para valores y cmb_variable_2d para etiquetas
            pass
        # Añadir lógica para otros gráficos si tienen parámetros únicos no cubiertos por los comunes

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
        ttk.Label(parent, text="Orientación:").grid(row=start_row, column=0, padx=5, pady=5, sticky="w")
        cmb_orient = ttk.Combobox(parent, values=["Vertical", "Horizontal"], state="readonly", width=10)
        cmb_orient.grid(row=start_row, column=1, padx=5, pady=5, sticky="w")
        cmb_orient.set("Vertical")
        ttk.Label(parent, text="Rotación ticks:").grid(row=start_row, column=2, padx=5, pady=5, sticky="w")
        cmb_rot = ttk.Combobox(parent, values=["0", "45", "90"], state="readonly", width=5)
        cmb_rot.grid(row=start_row, column=3, padx=5, pady=5, sticky="w")
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

    def _build_display_area(self):
        # Área de texto para estadísticas y log
        self.txt_result = tk.Text(self.right_frame, height=15, wrap=tk.WORD, state=tk.DISABLED, font=("Courier New", 9))
        self.txt_result.pack(padx=10, pady=10, fill=tk.X, expand=False)
        self._configure_log_tags() # Configurar tags para el log

        # Área de visualización del gráfico
        self.chart_display_frame = ttk.LabelFrame(self.right_frame, text="Visualización del Gráfico", padding="5")
        self.chart_display_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

    def _configure_log_tags(self):
        self.txt_result.tag_config("INFO", foreground="black")
        self.txt_result.tag_config("DEBUG", foreground="gray")
        self.txt_result.tag_config("WARN", foreground="orange")
        self.txt_result.tag_config("ERROR", foreground="red", font=("Courier New", 9, "bold"))
        self.txt_result.tag_config("SUCCESS", foreground="green")
        self.txt_result.tag_config("DESC", foreground="navy", font=("Courier New", 9, "bold"))
        self.txt_result.tag_config("PARAMS", foreground="purple")
        self.txt_result.tag_config("RECOM", foreground="darkgreen")

    def log(self, message, level="INFO"):
        try:
            timestamp = pd.Timestamp.now().strftime('%H:%M:%S.%f')[:-3]
            self.txt_result.config(state=tk.NORMAL)
            self.txt_result.insert(tk.END, f"[{timestamp}] [{level.upper()}] {message}\n", level.upper())
            self.txt_result.config(state=tk.DISABLED)
            self.txt_result.see(tk.END)
        except Exception as e:
            print(f"Error en logger de CombinedAnalysisTab: {e}")

    def load_data(self):
        filepath = filedialog.askopenfilename(
            title="Seleccionar archivo de datos",
            filetypes=(("Archivos Excel", "*.xlsx *.xls"),
                       ("Archivos CSV", "*.csv"),
                       ("Todos los archivos", "*.*")),
            parent=self.master # Usar master para el diálogo
        )
        if not filepath:
            self.log("Carga de archivo cancelada.", "INFO")
            return

        try:
            filename = os.path.basename(filepath)
            if filepath.endswith(('.xlsx', '.xls')):
                self.data = pd.read_excel(filepath, engine='openpyxl')
            elif filepath.endswith('.csv'):
                try:
                    sniffer = csv.Sniffer()
                    with open(filepath, 'r', encoding='utf-8-sig') as f:
                        dialect = sniffer.sniff(f.read(1024))
                    self.data = pd.read_csv(filepath, sep=dialect.delimiter)
                    self.log(f"Archivo CSV '{filename}' cargado con separador '{dialect.delimiter}' detectado.", "INFO")
                except Exception:
                    try: self.data = pd.read_csv(filepath, sep=',')
                    except pd.errors.ParserError: self.data = pd.read_csv(filepath, sep=';')
            else:
                messagebox.showerror("Error de Archivo", f"Tipo de archivo no soportado: {filename}", parent=self.master)
                self.log(f"Tipo de archivo no soportado: {filename}", "ERROR")
                return
            
            self.lbl_file.config(text=f"{filename} ({self.data.shape[0]}x{self.data.shape[1]})")
            self.log(f"Datos cargados desde '{filename}'. Dimensiones: {self.data.shape}", "SUCCESS")
            self._update_analysis_variable_comboboxes()
            self.filter_component.set_dataframe(self.data)
            self.current_fig = None # Resetear figura actual

        except Exception as e:
            messagebox.showerror("Error de Lectura", f"No se pudo leer el archivo:\n{e}", parent=self.master)
            self.log(f"Error leyendo archivo '{filepath}': {e}", "ERROR")
            self.data = None
            self.lbl_file.config(text="Error al cargar.")
            self._update_analysis_variable_comboboxes()
            self.filter_component.set_dataframe(None)
            self.current_fig = None

    def _update_analysis_variable_comboboxes(self):
        variables = list(self.data.columns) if self.data is not None else []
        self.cmb_variables['values'] = variables
        self.cmb_variable_2d['values'] = [""] + variables

        if variables:
            self.cmb_variables.set(variables[0])
            self.cmb_variable_2d.set("")
        else:
            self.cmb_variables.set("")
            self.cmb_variable_2d.set("")

    def apply_filters_and_generate_chart(self):
        if self.data is None:
            messagebox.showwarning("Aviso", "Carga datos primero.", parent=self.master)
            return

        var_principal = self.cmb_variables.get()
        if not var_principal:
            messagebox.showwarning("Aviso", "Selecciona una variable principal.", parent=self.master)
            return

        # 1. Obtener datos filtrados desde FilterComponent
        df = self.filter_component.apply_filters()

        if df is None:
            messagebox.showerror("Error de Filtro", "Ocurrió un error al aplicar los filtros desde el componente.", parent=self.master)
            self.txt_result.config(state=tk.NORMAL)
            self.txt_result.delete("1.0", tk.END)
            self.txt_result.config(state=tk.DISABLED)
            self._clear_chart_display()
            self.current_fig = None
            return
        
        if df.empty:
            messagebox.showinfo("Datos Vacíos", "No quedan datos después de aplicar los filtros.", parent=self.master)
            self.txt_result.config(state=tk.NORMAL)
            self.txt_result.delete("1.0", tk.END)
            self.txt_result.config(state=tk.DISABLED)
            self._clear_chart_display()
            self.current_fig = None
            return

        self.filtered_data = df

        # 2. Obtener la variable principal y tipo para el análisis
        if var_principal not in df.columns:
             messagebox.showerror("Error", f"La variable principal '{var_principal}' no se encuentra en los datos filtrados.", parent=self.master)
             return

        tipo_variable_principal = self.cmb_tipo.get()

        # 3. Generar resumen estadístico
        self._generate_summary(df, var_principal, tipo_variable_principal)

        # 4. Generar gráfica
        plot_type = self.cmb_plot_type.get()
        var_secundaria = self.cmb_variable_2d.get() if self.cmb_variable_2d.get() else None
        show_points = self.show_individual_points.get()
        show_grid = self.show_grid.get()
        show_annotation = self.show_annotation.get()

        self._clear_chart_display() # Limpiar el área de gráfico antes de generar uno nuevo

        try:
            self._generate_plot(self.filtered_data, var_principal, var_secundaria, tipo_variable_principal, plot_type, show_points, show_grid, show_annotation)
        except Exception as e:
            messagebox.showerror("Error al Graficar", f"Ocurrió un error al generar la gráfica:\n{e}\n\n{traceback.format_exc()}", parent=self.master)
            self.log(f"Error al generar gráfico: {e}", "ERROR")
            self.current_fig = None

    def _clear_chart_display(self):
        for widget in self.chart_display_frame.winfo_children():
            widget.destroy()
        self.current_fig = None

    def _generate_summary(self, df: pd.DataFrame, var_principal: str, tipo_variable_principal: str):
        self.txt_result.config(state=tk.NORMAL)
        self.txt_result.delete("1.0", tk.END) # Limpiar antes de añadir nuevo resumen
        summary_sections = []

        summary_sections.append(f"Resumen de {var_principal} (después de filtros):")
        summary_sections.append(f"Total de filas filtradas (n): {len(df)}")

        series_orig = df[var_principal]
        
        series_processed = series_orig.copy()
        if tipo_variable_principal == "Cualitativa":
            series_processed = series_processed.astype(str)
        elif tipo_variable_principal == "Cuantitativa":
            series_processed = pd.to_numeric(series_processed, errors='coerce')

        series_for_stats = series_processed.dropna() if self.exclude_blank.get() else series_processed

        valid_count = series_for_stats.count()
        
        summary_sections.append(f"Valores válidos en '{var_principal}' (considerando tipo y 'Excluir en blanco'): {valid_count}")
        
        if self.exclude_blank.get():
            series_after_type_coercion = series_orig.copy()
            if tipo_variable_principal == "Cuantitativa":
                 series_after_type_coercion = pd.to_numeric(series_after_type_coercion, errors='coerce')
            
            original_nans = series_after_type_coercion.isnull().sum()
            # Corrección: El cálculo de excluded_nans debe ser la diferencia entre los NaNs originales y los que quedan después de procesar para estadísticas.
            # Si series_for_stats ya tiene dropna(), entonces los NaNs restantes en ella son 0.
            # Por lo tanto, excluded_nans es simplemente original_nans si se aplicó dropna.
            excluded_nans = original_nans if series_for_stats.count() < series_after_type_coercion.count() else 0
            if excluded_nans > 0 : # Mostrar solo si realmente se excluyeron NaNs
                 summary_sections.append(f"Casillas en blanco/inválidas en '{var_principal}' (fueron {original_nans}, ahora excluidas): {excluded_nans}")
        else: 
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
            freq_series_for_display = series_processed.value_counts(dropna=False) 
        else: 
            freq_series_for_display = series_for_stats.value_counts(dropna=True)

        n_obs_freq = freq_series_for_display.sum() 
        summary_sections.append("  Frecuencia de etiquetas (Top 10" + (", excluyendo blancos/inválidos" if self.exclude_blank.get() else ", incluyendo blancos/inválidos") + "):")
        if not freq_series_for_display.empty:
            freq_df = pd.DataFrame({
                "Count": freq_series_for_display,
                "Percentage": (freq_series_for_display / n_obs_freq * 100).round(2) if n_obs_freq > 0 else 0
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
                    
                    # Usar el índice original de freq_series_for_display para el reordenamiento
                    temp_freq_df_for_reorder = pd.DataFrame({
                        "Count": freq_series_for_display.values
                    }, index=freq_series_for_display.index.astype(str))


                    ordered_idx_keys = [o for o in order if o in temp_freq_df_for_reorder.index]
                    remaining_idx_keys = [i for i in temp_freq_df_for_reorder.index if i not in ordered_idx_keys]
                    final_order_keys = ordered_idx_keys + remaining_idx_keys
                    
                    # Reindexar usando las claves originales
                    freq_df_reordered = temp_freq_df_for_reorder.reindex(final_order_keys)
                    
                    # Aplicar mapeo al índice del DataFrame reordenado
                    freq_df_reordered.index = [mapping.get(str(i), str(i)) for i in freq_df_reordered.index]
                    
                    # Re-calculate percentage after reordering and mapping
                    freq_df_reordered["Percentage"] = (freq_df_reordered["Count"] / freq_df_reordered["Count"].sum() * 100).round(2) if freq_df_reordered["Count"].sum() > 0 else 0
                    freq_df = freq_df_reordered


                except Exception as e:
                    summary_sections.append(f"  Advertencia: Error al aplicar etiquetas/orden personalizados: {e}\n  Usando orden por defecto.")

            summary_sections.append(freq_df.head(10).to_string().replace('\n', '\n    ')) 
        else:
            summary_sections.append("    No hay valores válidos para contar.")

        for section in summary_sections:
            self.log(section, "INFO") # Usar el logger para mostrar en el widget de texto
        self.txt_result.config(state=tk.DISABLED)

    def _generate_plot(self, df_orig: pd.DataFrame, var_principal: str, var_secundaria: str,
                       tipo_variable_principal: str, plot_type: str, show_points: bool, show_grid: bool, show_annotation: bool):
        """Motor central para generar todos los tipos de gráficos."""
        if df_orig.empty:
            messagebox.showwarning("Información", "No hay datos para graficar después de aplicar los filtros.", parent=self.master)
            self.current_fig = None
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
        orient_mat = "horizontal" if orient == "Horizontal" else "vertical"
        rot = int(self.cmb_tickrot.get())
        med_color = self.cmb_mediana.get()
        custom_title = self.entry_custom_title.get()
        x_label = self.entry_x_label.get()
        y_label = self.entry_y_label.get()
        scale_mode = self.cmb_scale.get()
        y_value_mode = self.cmb_y_value.get() 

        x_min_str = self.entry_xmin.get()
        x_max_str = self.entry_xmax.get()
        y_min_str = self.entry_ymin.get()
        y_max_str = self.entry_ymax.get()

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

        fig, ax = plt.subplots(figsize=(fig_width_in, fig_height_in), dpi=dpi)
        self.current_fig = fig # Guardar referencia a la figura

        try:
            if plot_type == "Histograma":
                self._plot_histogram(ax, df_plot, var_principal, orient_mat, color, edge, med_color, scale_mode, y_value_mode, x_label, y_label, x_min_str, x_max_str, y_min_str, y_max_str, rot)
            elif plot_type == "Gráfico de Barras":
                self._plot_bar_chart(ax, df_plot, var_principal, orient_mat, edge, y_value_mode, x_label, y_label, x_min_str, x_max_str, y_min_str, y_max_str, rot)
            elif plot_type == "Countplot":
                self._plot_countplot(ax, df_plot, var_principal, var_secundaria, orient_mat, color, edge, y_value_mode, x_label, y_label, x_min_str, x_max_str, y_min_str, y_max_str, rot)
            elif plot_type == "Caja y Bigotes":
                self._plot_boxplot(ax, df_plot, var_principal, var_secundaria, orient_mat, color, edge, med_color, show_points, x_label, y_label, x_min_str, x_max_str, y_min_str, y_max_str, rot)
            elif plot_type == "Violín":
                self._plot_violinplot(ax, df_plot, var_principal, var_secundaria, orient_mat, color, edge, med_color, show_points, x_label, y_label, x_min_str, x_max_str, y_min_str, y_max_str, rot)
            elif plot_type == "Boxen Plot":
                self._plot_boxenplot(ax, df_plot, var_principal, var_secundaria, orient_mat, color, edge, show_points, x_label, y_label, x_min_str, x_max_str, y_min_str, y_max_str, rot)
            elif plot_type == "Swarm Plot":
                self._plot_swarmplot(ax, df_plot, var_principal, var_secundaria, orient_mat, color, edge, x_label, y_label, x_min_str, x_max_str, y_min_str, y_max_str, rot)
            elif plot_type == "KDE Plot":
                self._plot_kdeplot(ax, df_plot, var_principal, orient_mat, color, scale_mode, x_label, y_label, x_min_str, x_max_str, y_min_str, y_max_str)
            elif plot_type == "Histplot (Hist+KDE)":
                self._plot_histplot_kde(ax, df_plot, var_principal, orient_mat, color, edge, scale_mode, y_value_mode, x_label, y_label, x_min_str, x_max_str, y_min_str, y_max_str)
            elif plot_type == "Gráfico Circular":
                self._plot_pie_chart(ax, df_plot, var_principal, edge, x_label, y_label)
            elif plot_type == "Líneas":
                self._plot_line_chart(ax, df_plot, var_principal, var_secundaria, color, x_label, y_label, x_min_str, x_max_str, y_min_str, y_max_str)
            elif plot_type == "Dispersión":
                self._plot_scatter_plot(ax, df_plot, var_principal, var_secundaria, color, edge, False, x_label, y_label, x_min_str, x_max_str, y_min_str, y_max_str)
            elif plot_type == "Dispersión con Regresión":
                self._plot_scatter_plot(ax, df_plot, var_principal, var_secundaria, color, edge, True, x_label, y_label, x_min_str, x_max_str, y_min_str, y_max_str)
            else:
                messagebox.showwarning("Aviso", f"Tipo de gráfico '{plot_type}' no implementado completamente.", parent=self.master)
                plt.close(fig) 
                self.current_fig = None
                return

            if custom_title:
                ax.set_title(custom_title, color=font_color)
            else:
                default_title = f"{plot_type} de {var_principal}"
                if var_secundaria and plot_type not in ["Gráfico Circular"]:
                    default_title += f" por {var_secundaria}"
                ax.set_title(default_title, color=font_color)
            
            if show_annotation:
                n_filtered = len(df_plot) 
                series_for_plot = df_plot[var_principal]
                if self.exclude_blank.get() and plot_type not in ["Líneas", "Dispersión", "Dispersión con Regresión"]:
                    series_for_plot = series_for_plot.dropna()
                
                n_valid = series_for_plot.count()
                annotation_text = (f"Total filas (n): {n_filtered}" +
                                   f"\nValores válidos en '{var_principal}' (gráfico): {n_valid}")
                ax.annotate(annotation_text, xy=(0.98, 0.98), xycoords="axes fraction", # Cambiado a axes fraction
                             fontsize=font_size-2, ha="right", va="top", 
                             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7, edgecolor=font_color))

            fig.tight_layout(rect=[0, 0, 0.95, 0.95]) 
            
            canvas = FigureCanvasTkAgg(fig, master=self.chart_display_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
            self.log(f"{plot_type} generado.", "SUCCESS")

        except Exception as e:
            plt.close(fig) 
            self.current_fig = None
            raise e 

    def _apply_limits(self, ax, x_min_str: str, x_max_str: str, y_min_str: str, y_max_str: str):
        """Aplica límites a los ejes X e Y."""
        try:
            xmin_val = float(x_min_str) if x_min_str else None
            xmax_val = float(x_max_str) if x_max_str else None
            ymin_val = float(y_min_str) if y_min_str else None
            ymax_val = float(y_max_str) if y_max_str else None

            current_xlim = ax.get_xlim()
            current_ylim = ax.get_ylim()

            ax.set_xlim(xmin_val if xmin_val is not None else current_xlim[0],
                     xmax_val if xmax_val is not None else current_xlim[1])
            ax.set_ylim(ymin_val if ymin_val is not None else current_ylim[0],
                     ymax_val if ymax_val is not None else current_ylim[1])

        except ValueError as e:
            messagebox.showwarning("Advertencia", f"Límites de eje inválidos: {e}\nIgnorando límites manuales.", parent=self.master)
        except Exception as e:
             messagebox.showwarning("Advertencia", f"Error al aplicar límites de eje: {e}\nIgnorando límites manuales.", parent=self.master)

    def _parse_bar_colors(self, n: int):
        txt = self.entry_bar_colors.get().strip()
        default_color = self.cmb_color.get()
        if not txt:
            return [default_color] * n
        
        cols_input = [c.strip() for c in txt.split(",") if c.strip()]
        if not cols_input:
             return [default_color] * n

        valid_colors = []
        for c_name in cols_input:
            try:
                mpl.colors.to_rgb(c_name) 
                valid_colors.append(c_name)
            except ValueError:
                valid_colors.append(default_color) 
                self.log(f"Advertencia: Color '{c_name}' no reconocido, usando color por defecto '{default_color}'.", "WARN")
        
        return list(islice(cycle(valid_colors if valid_colors else [default_color]), n))

    def save_current_graph(self):
        if self.current_fig is None:
            messagebox.showwarning("Sin Gráfica", "No hay ninguna gráfica generada para guardar.", parent=self.master)
            return
        
        filepath = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("SVG files", "*.svg"), ("PDF files", "*.pdf"), ("All files", "*.*")],
            title="Guardar Gráfica Como",
            parent=self.master
        )
        if not filepath:
            return
        try:
            self.current_fig.savefig(filepath)
            messagebox.showinfo("Guardado", f"Gráfica guardada en:\n{filepath}", parent=self.master)
            self.log(f"Gráfica guardada en: {filepath}", "SUCCESS")
        except Exception as e:
            messagebox.showerror("Error al Guardar", f"No se pudo guardar la gráfica:\n{e}", parent=self.master)
            self.log(f"Error al guardar gráfica: {e}", "ERROR")

    # ---- Métodos de graficación específicos ----
    def _plot_histogram(self, ax, df, var, orient_mat, color, edge, med_color, scale_mode, y_value_mode, x_label, y_label, x_min_str, x_max_str, y_min_str, y_max_str, rot):
        if not pd.api.types.is_numeric_dtype(df[var]):
            messagebox.showwarning("Aviso", f"El histograma requiere una variable numérica. '{var}' no lo es.", parent=self.master)
            return
        data_for_hist = df[var].dropna() if self.exclude_blank.get() else df[var] 
        if data_for_hist.empty:
             ax.text(0.5, 0.5, "No hay datos válidos", ha='center', va='center', transform=ax.transAxes); return
        weights = np.ones_like(data_for_hist) / len(data_for_hist) * 100 if y_value_mode == "Percent" and len(data_for_hist) > 0 else None
        ax.hist(data_for_hist, bins=20, weights=weights, color=color, edgecolor=edge, orientation=orient_mat)
        default_ylabel = "Porcentaje (%)" if y_value_mode == "Percent" else "Frecuencia"
        if orient_mat == "horizontal":
            ax.set_xlabel(x_label or default_ylabel); ax.set_ylabel(y_label or var)
            ax.set_xscale(scale_mode)
            if pd.api.types.is_numeric_dtype(data_for_hist) and not data_for_hist.empty:
                mediana = data_for_hist.median()
                ax.axhline(mediana, color=med_color, linestyle="dashed", linewidth=2, label=f"Mediana: {mediana:.2f}")
                ax.legend()
            plt.setp(ax.get_yticklabels(), rotation=rot) 
        else: 
            ax.set_xlabel(x_label or var); ax.set_ylabel(y_label or default_ylabel)
            ax.set_yscale(scale_mode)
            if pd.api.types.is_numeric_dtype(data_for_hist) and not data_for_hist.empty:
                mediana = data_for_hist.median()
                ax.axvline(mediana, color=med_color, linestyle="dashed", linewidth=2, label=f"Mediana: {mediana:.2f}")
                ax.legend()
            plt.setp(ax.get_xticklabels(), rotation=rot) 
        self._apply_limits(ax, x_min_str, x_max_str, y_min_str, y_max_str)

    def _plot_bar_chart(self, ax, df, var, orient_mat, edge, y_value_mode, x_label, y_label, x_min_str, x_max_str, y_min_str, y_max_str, rot):
        categorical_series = df[var].astype(str)
        freq_series = categorical_series.value_counts(dropna=self.exclude_blank.get()) 
        if len(freq_series) == 0:
             ax.text(0.5, 0.5, "No hay datos válidos", ha='center', va='center', transform=ax.transAxes); return
        n_obs_freq = freq_series.sum() 
        freq_df = pd.DataFrame({"Count": freq_series, "Percentage": (freq_series / n_obs_freq * 100).round(2) if n_obs_freq > 0 else 0})
        custom_text = self.entry_etiquetas.get().strip()
        if custom_text: # Aplicar lógica de orden y mapeo de DataFilterTab
            try:
                order, mapping = [], {}
                for item in custom_text.split(","):
                    if ":" in item: k, v = item.split(":", 1); order.append(k.strip()); mapping[k.strip()] = v.strip()
                    else: order.append(item.strip())
                temp_df = pd.DataFrame({'Count': freq_series.values, 'Percentage': (freq_series.values / n_obs_freq * 100).round(2) if n_obs_freq > 0 else 0}, index=freq_series.index.astype(str))
                ordered_idx = [o for o in order if o in temp_df.index] + [i for i in temp_df.index if i not in order]
                freq_df = temp_df.reindex(ordered_idx)
                freq_df.index = [mapping.get(str(i), str(i)) for i in freq_df.index]
            except Exception as e: self.log(f"Error aplicando etiquetas/orden a Bar Chart: {e}", "WARN")
        bar_colors = self._parse_bar_colors(len(freq_df))
        values = freq_df["Percentage"] if y_value_mode == "Percent" else freq_df["Count"]
        default_ylabel = "Porcentaje (%)" if y_value_mode == "Percent" else "Frecuencia"
        if orient_mat == "horizontal":
            y_pos = np.arange(len(freq_df)); ax.barh(y_pos, values, color=bar_colors, edgecolor=edge)
            ax.set_yticks(y_pos); ax.set_yticklabels(freq_df.index, rotation=rot)
            ax.set_xlabel(x_label or default_ylabel); ax.set_ylabel(y_label or var)
            ax.set_xscale(self.cmb_scale.get())
        else: 
            ax.bar(freq_df.index, values, color=bar_colors, edgecolor=edge)
            ax.set_xlabel(x_label or var); ax.set_ylabel(y_label or default_ylabel)
            plt.setp(ax.get_xticklabels(), rotation=rot)
            ax.set_yscale(self.cmb_scale.get())
        self._apply_limits(ax, x_min_str, x_max_str, y_min_str, y_max_str)

    def _plot_countplot(self, ax, df, var_principal, var_secundaria, orient_mat, color, edge, y_value_mode, x_label, y_label, x_min_str, x_max_str, y_min_str, y_max_str, rot):
        data_for_plot = df.copy()
        if self.exclude_blank.get(): 
            data_for_plot = data_for_plot.dropna(subset=[var_principal])
            if var_secundaria and var_secundaria in data_for_plot.columns: data_for_plot = data_for_plot.dropna(subset=[var_secundaria])
        if data_for_plot.empty or data_for_plot[var_principal].empty:
             ax.text(0.5, 0.5, "No hay datos válidos", ha='center', va='center', transform=ax.transAxes); return
        data_for_plot[var_principal] = data_for_plot[var_principal].astype(str)
        if var_secundaria and var_secundaria in data_for_plot.columns: data_for_plot[var_secundaria] = data_for_plot[var_secundaria].astype(str)
        
        order, mapping = None, {}
        custom_text = self.entry_etiquetas.get().strip()
        if custom_text:
            try:
                parsed_order, temp_mapping = [], {}
                for item in custom_text.split(","):
                    if ":" in item: k, v = item.split(":", 1); parsed_order.append(k.strip()); temp_mapping[k.strip()] = v.strip()
                    else: parsed_order.append(item.strip())
                mapping = temp_mapping
                if mapping:
                    mapped_series = data_for_plot[var_principal].astype(str).map(mapping).fillna(data_for_plot[var_principal].astype(str))
                    data_for_plot[var_principal] = mapped_series
                    order = [mapping.get(o, o) for o in parsed_order if mapping.get(o,o) in data_for_plot[var_principal].unique()]
                    order += [v for v in data_for_plot[var_principal].unique() if v not in order]
                else:
                    order = [o for o in parsed_order if o in data_for_plot[var_principal].unique()]
                    order += [v for v in data_for_plot[var_principal].unique() if v not in order]
            except Exception as e: self.log(f"Error aplicando etiquetas/orden a Countplot: {e}", "WARN"); order = None

        if orient_mat == "horizontal":
            sns.countplot(y=var_principal, data=data_for_plot, color=color, edgecolor=edge, order=order, hue=var_secundaria if var_secundaria else None, ax=ax)
            ax.set_xlabel(x_label or "Count"); ax.set_ylabel(y_label or var_principal)
            plt.setp(ax.get_yticklabels(), rotation=rot)
            ax.set_xscale(self.cmb_scale.get())
        else: 
            sns.countplot(x=var_principal, data=data_for_plot, color=color, edgecolor=edge, order=order, hue=var_secundaria if var_secundaria else None, ax=ax)
            ax.set_xlabel(x_label or var_principal); ax.set_ylabel(y_label or "Count")
            plt.setp(ax.get_xticklabels(), rotation=rot)
            ax.set_yscale(self.cmb_scale.get())
        self._apply_limits(ax, x_min_str, x_max_str, y_min_str, y_max_str)

    def _plot_boxplot(self, ax, df, var_principal, var_secundaria, orient_mat, color, edge, med_color, show_points, x_label, y_label, x_min_str, x_max_str, y_min_str, y_max_str, rot):
        if not pd.api.types.is_numeric_dtype(df[var_principal]):
            messagebox.showwarning("Aviso", f"Boxplot requiere variable principal numérica: '{var_principal}'.", parent=self.master); return
        data_for_plot = df.copy()
        if self.exclude_blank.get(): data_for_plot.dropna(subset=[var_principal], inplace=True)
        x_sns, y_sns = (var_principal, var_secundaria) if orient_mat == "horizontal" else (var_secundaria, var_principal)
        final_plot_cols = [col for col in [x_sns, y_sns] if col and col in data_for_plot.columns]
        if final_plot_cols: data_for_plot.dropna(subset=final_plot_cols, inplace=True)
        if data_for_plot.empty or data_for_plot[var_principal].empty:
             ax.text(0.5, 0.5, "No hay datos válidos", ha='center', va='center', transform=ax.transAxes); return
        if var_secundaria and var_secundaria in data_for_plot.columns:
             data_for_plot[var_secundaria] = data_for_plot[var_secundaria].astype(str)
             sns.boxplot(x=x_sns, y=y_sns, data=data_for_plot, color=color, linewidth=1.5, fliersize=5, ax=ax, medianprops={'color': med_color})
             if orient_mat == "horizontal": plt.setp(ax.get_yticklabels(), rotation=rot)
             else: plt.setp(ax.get_xticklabels(), rotation=rot)
             if show_points: sns.stripplot(x=x_sns, y=y_sns, data=data_for_plot, color=".3", size=3, jitter=True, ax=ax)
        else:
            if orient_mat == "horizontal": 
                 sns.boxplot(x=data_for_plot[var_principal], color=color, linewidth=1.5, fliersize=5, ax=ax, medianprops={'color': med_color}); ax.set_yticks([])
                 if show_points: sns.stripplot(x=data_for_plot[var_principal], color=".3", size=3, jitter=True, ax=ax)
            else: 
                 sns.boxplot(y=data_for_plot[var_principal], color=color, linewidth=1.5, fliersize=5, ax=ax, medianprops={'color': med_color}); ax.set_xticks([])
                 if show_points: sns.stripplot(y=data_for_plot[var_principal], color=".3", size=3, jitter=True, ax=ax)
        if orient_mat == "horizontal": ax.set_xlabel(x_label or var_principal); ax.set_ylabel(y_label or var_secundaria); ax.set_xscale(self.cmb_scale.get())
        else: ax.set_xlabel(x_label or var_secundaria); ax.set_ylabel(y_label or var_principal); ax.set_yscale(self.cmb_scale.get())
        self._apply_limits(ax, x_min_str, x_max_str, y_min_str, y_max_str)

    def _plot_violinplot(self, ax, df, var_principal, var_secundaria, orient_mat, color, edge, med_color, show_points, x_label, y_label, x_min_str, x_max_str, y_min_str, y_max_str, rot):
        if not pd.api.types.is_numeric_dtype(df[var_principal]):
            messagebox.showwarning("Aviso", f"Violin plot requiere variable principal numérica: '{var_principal}'.", parent=self.master); return
        data_for_plot = df.copy()
        if self.exclude_blank.get(): data_for_plot.dropna(subset=[var_principal], inplace=True)
        x_sns, y_sns = (var_principal, var_secundaria) if orient_mat == "horizontal" else (var_secundaria, var_principal)
        final_plot_cols = [col for col in [x_sns, y_sns] if col and col in data_for_plot.columns]
        if final_plot_cols: data_for_plot.dropna(subset=final_plot_cols, inplace=True)
        if data_for_plot.empty or data_for_plot[var_principal].empty:
             ax.text(0.5,0.5, "No hay datos válidos", ha='center', va='center', transform=ax.transAxes); return
        if var_secundaria and var_secundaria in data_for_plot.columns:
            data_for_plot[var_secundaria] = data_for_plot[var_secundaria].astype(str)
            sns.violinplot(x=x_sns, y=y_sns, data=data_for_plot, color=color, inner="quartile", ax=ax)
            if orient_mat == "horizontal": plt.setp(ax.get_yticklabels(), rotation=rot)
            else: plt.setp(ax.get_xticklabels(), rotation=rot)
            if show_points: sns.stripplot(x=x_sns, y=y_sns, data=data_for_plot, color=".3", size=3, jitter=True, ax=ax)
        else:
            if orient_mat == "horizontal":
                sns.violinplot(x=data_for_plot[var_principal], color=color, inner="quartile", ax=ax); ax.set_yticks([])
                if show_points: sns.stripplot(x=data_for_plot[var_principal], color=".3", size=3, jitter=True, ax=ax)
            else:
                sns.violinplot(y=data_for_plot[var_principal], color=color, inner="quartile", ax=ax); ax.set_xticks([])
                if show_points: sns.stripplot(y=data_for_plot[var_principal], color=".3", size=3, jitter=True, ax=ax)
        if orient_mat == "horizontal": ax.set_xlabel(x_label or var_principal); ax.set_ylabel(y_label or var_secundaria); ax.set_xscale(self.cmb_scale.get())
        else: ax.set_xlabel(x_label or var_secundaria); ax.set_ylabel(y_label or var_principal); ax.set_yscale(self.cmb_scale.get())
        self._apply_limits(ax, x_min_str, x_max_str, y_min_str, y_max_str)

    def _plot_boxenplot(self, ax, df, var_principal, var_secundaria, orient_mat, color, edge, show_points, x_label, y_label, x_min_str, x_max_str, y_min_str, y_max_str, rot):
        if not pd.api.types.is_numeric_dtype(df[var_principal]):
            messagebox.showwarning("Aviso", f"Boxen plot requiere variable principal numérica: '{var_principal}'.", parent=self.master); return
        data_for_plot = df.copy()
        if self.exclude_blank.get(): data_for_plot.dropna(subset=[var_principal], inplace=True)
        x_sns, y_sns = (var_principal, var_secundaria) if orient_mat == "horizontal" else (var_secundaria, var_principal)
        final_plot_cols = [col for col in [x_sns, y_sns] if col and col in data_for_plot.columns]
        if final_plot_cols: data_for_plot.dropna(subset=final_plot_cols, inplace=True)
        if data_for_plot.empty or data_for_plot[var_principal].empty:
             ax.text(0.5,0.5, "No hay datos válidos", ha='center', va='center', transform=ax.transAxes); return
        if var_secundaria and var_secundaria in data_for_plot.columns:
            data_for_plot[var_secundaria] = data_for_plot[var_secundaria].astype(str)
            sns.boxenplot(x=x_sns, y=y_sns, data=data_for_plot, color=color, linewidth=1.5, ax=ax)
            if orient_mat == "horizontal": plt.setp(ax.get_yticklabels(), rotation=rot)
            else: plt.setp(ax.get_xticklabels(), rotation=rot)
            if show_points: sns.stripplot(x=x_sns, y=y_sns, data=data_for_plot, color=".3", size=3, jitter=True, ax=ax)
        else:
            if orient_mat == "horizontal":
                sns.boxenplot(x=data_for_plot[var_principal], color=color, linewidth=1.5, ax=ax); ax.set_yticks([])
                if show_points: sns.stripplot(x=data_for_plot[var_principal], color=".3", size=3, jitter=True, ax=ax)
            else:
                sns.boxenplot(y=data_for_plot[var_principal], color=color, linewidth=1.5, ax=ax); ax.set_xticks([])
                if show_points: sns.stripplot(y=data_for_plot[var_principal], color=".3", size=3, jitter=True, ax=ax)
        if orient_mat == "horizontal": ax.set_xlabel(x_label or var_principal); ax.set_ylabel(y_label or var_secundaria); ax.set_xscale(self.cmb_scale.get())
        else: ax.set_xlabel(x_label or var_secundaria); ax.set_ylabel(y_label or var_principal); ax.set_yscale(self.cmb_scale.get())
        self._apply_limits(ax, x_min_str, x_max_str, y_min_str, y_max_str)

    def _plot_swarmplot(self, ax, df, var_principal, var_secundaria, orient_mat, color, edge, x_label, y_label, x_min_str, x_max_str, y_min_str, y_max_str, rot):
        if not pd.api.types.is_numeric_dtype(df[var_principal]):
            messagebox.showwarning("Aviso", f"Swarm plot requiere variable principal numérica: '{var_principal}'.", parent=self.master); return
        if not (var_secundaria and var_secundaria in df.columns):
            messagebox.showwarning("Aviso", "Swarm plot requiere una Segunda Variable (Grupo/Categoría).", parent=self.master); return
        data_for_plot = df.copy()
        if self.exclude_blank.get(): data_for_plot.dropna(subset=[var_principal], inplace=True)
        plot_cols_to_clean = [var_principal, var_secundaria]
        data_for_plot.dropna(subset=plot_cols_to_clean, inplace=True)
        if data_for_plot.empty or data_for_plot[var_principal].empty or data_for_plot[var_secundaria].empty :
             ax.text(0.5, 0.5, "No hay datos válidos", ha='center', va='center', transform=ax.transAxes); return
        data_for_plot[var_secundaria] = data_for_plot[var_secundaria].astype(str)
        x_sns, y_sns = (var_principal, var_secundaria) if orient_mat == "horizontal" else (var_secundaria, var_principal)
        sns.swarmplot(x=x_sns, y=y_sns, data=data_for_plot, color=color, edgecolor=edge, linewidth=0.5, ax=ax)
        if orient_mat == "horizontal": plt.setp(ax.get_yticklabels(), rotation=rot)
        else: plt.setp(ax.get_xticklabels(), rotation=rot)
        if orient_mat == "horizontal": ax.set_xlabel(x_label or var_principal); ax.set_ylabel(y_label or var_secundaria); ax.set_xscale(self.cmb_scale.get())
        else: ax.set_xlabel(x_label or var_secundaria); ax.set_ylabel(y_label or var_principal); ax.set_yscale(self.cmb_scale.get())
        self._apply_limits(ax, x_min_str, x_max_str, y_min_str, y_max_str)

    def _plot_kdeplot(self, ax, df, var_principal, orient_mat, color, scale_mode, x_label, y_label, x_min_str, x_max_str, y_min_str, y_max_str):
        if not pd.api.types.is_numeric_dtype(df[var_principal]):
            messagebox.showwarning("Aviso", f"KDE plot requiere variable numérica: '{var_principal}'.", parent=self.master); return
        data = df[var_principal].dropna() if self.exclude_blank.get() else df[var_principal] 
        if data.empty: ax.text(0.5,0.5, "No hay datos válidos", ha='center', va='center', transform=ax.transAxes); return
        if len(data.unique()) < 2: ax.text(0.5,0.5, "Datos insuficientes/constantes", ha='center', va='center', transform=ax.transAxes); return
        if orient_mat == "horizontal":
             sns.kdeplot(y=data, color=color, fill=True, ax=ax)
             ax.set_xlabel(x_label or "Densidad"); ax.set_ylabel(y_label or var_principal)
             ax.set_xscale(scale_mode) 
        else: 
             sns.kdeplot(x=data, color=color, fill=True, ax=ax)
             ax.set_xlabel(x_label or var_principal); ax.set_ylabel(y_label or "Densidad")
             ax.set_yscale(scale_mode) 
        self._apply_limits(ax, x_min_str, x_max_str, y_min_str, y_max_str)

    def _plot_histplot_kde(self, ax, df, var_principal, orient_mat, color, edge, scale_mode, y_value_mode, x_label, y_label, x_min_str, x_max_str, y_min_str, y_max_str):
        if not pd.api.types.is_numeric_dtype(df[var_principal]):
            messagebox.showwarning("Aviso", f"Histplot requiere variable numérica: '{var_principal}'.", parent=self.master); return
        data = df[var_principal].dropna() if self.exclude_blank.get() else df[var_principal] 
        if data.empty: ax.text(0.5,0.5, "No hay datos válidos", ha='center', va='center', transform=ax.transAxes); return
        plot_kde = len(data.unique()) >= 2 
        stat_val = y_value_mode.lower() if y_value_mode.lower() in ["count", "frequency", "density", "percent", "probability"] else "count"
        if orient_mat == "horizontal":
             sns.histplot(y=data, color=color, edgecolor=edge, kde=plot_kde, stat=stat_val, ax=ax)
             ax.set_xlabel(x_label or y_value_mode); ax.set_ylabel(y_label or var_principal)
             ax.set_xscale(scale_mode) 
        else: 
             sns.histplot(x=data, color=color, edgecolor=edge, kde=plot_kde, stat=stat_val, ax=ax)
             ax.set_xlabel(x_label or var_principal); ax.set_ylabel(y_label or y_value_mode) 
             ax.set_yscale(scale_mode) 
        self._apply_limits(ax, x_min_str, x_max_str, y_min_str, y_max_str)

    def _plot_pie_chart(self, ax, df, var_principal, edge, x_label, y_label):
        series_for_pie = df[var_principal].astype(str)
        if self.exclude_blank.get():
            series_for_pie = series_for_pie[series_for_pie.notna() & (series_for_pie.str.strip() != '')]
        freq_series = series_for_pie.value_counts(dropna=self.exclude_blank.get())
        if freq_series.empty:
            ax.text(0.5, 0.5, "No hay datos válidos", ha='center', va='center', transform=ax.transAxes); return
        labels, sizes = freq_series.index, freq_series.values
        custom_text = self.entry_etiquetas.get().strip()
        if custom_text: # Aplicar lógica de orden y mapeo
            try:
                order_keys, mapping = [], {}
                for item in custom_text.split(","):
                    if ":" in item: k, v = item.split(":", 1); order_keys.append(k.strip()); mapping[k.strip()] = v.strip()
                    else: order_keys.append(item.strip())
                temp_df = pd.DataFrame({'sizes': sizes, 'original_labels': labels}, index=labels.astype(str))
                final_ordered_keys = [ok for ok in order_keys if ok in temp_df.index.tolist()] + [lbl for lbl in temp_df.index.tolist() if lbl not in order_keys]
                temp_df = temp_df.reindex(final_ordered_keys)
                final_labels = [mapping.get(str(orig_label), str(orig_label)) for orig_label in temp_df.index]
                sizes, labels = temp_df['sizes'].values, final_labels
            except Exception as e: self.log(f"Error aplicando etiquetas/orden a Pie Chart: {e}", "WARN")
        pie_colors = self._parse_bar_colors(len(labels))
        ax.pie(sizes, labels=labels, colors=pie_colors, autopct='%1.1f%%', startangle=90, wedgeprops={'edgecolor': edge})
        ax.axis('equal')

    def _plot_line_chart(self, ax, df, var_principal, var_secundaria, color, x_label, y_label, x_min_str, x_max_str, y_min_str, y_max_str):
        if not (var_secundaria and var_secundaria in df.columns):
            messagebox.showwarning("Aviso", "Gráfico de Líneas requiere una Segunda Variable (Eje Y).", parent=self.master); return
        if not pd.api.types.is_numeric_dtype(df[var_principal]) or not pd.api.types.is_numeric_dtype(df[var_secundaria]):
            messagebox.showwarning("Aviso", f"Gráfico de Líneas requiere variables numéricas: '{var_principal}', '{var_secundaria}'.", parent=self.master); return
        data_for_plot = df.copy()
        if self.exclude_blank.get(): data_for_plot.dropna(subset=[var_principal], inplace=True)
        data_for_plot.dropna(subset=[var_principal, var_secundaria], inplace=True)
        if data_for_plot.empty:
             ax.text(0.5, 0.5, "No hay datos válidos", ha='center', va='center', transform=ax.transAxes); return
        data_for_plot = data_for_plot.sort_values(by=var_principal)
        ax.plot(data_for_plot[var_principal], data_for_plot[var_secundaria], color=color)
        ax.set_xlabel(x_label or var_principal); ax.set_ylabel(y_label or var_secundaria)
        ax.set_xscale(self.cmb_scale.get()) 
        self._apply_limits(ax, x_min_str, x_max_str, y_min_str, y_max_str) 

    def _plot_scatter_plot(self, ax, df, var_principal, var_secundaria, color, edge, include_regression, x_label, y_label, x_min_str, x_max_str, y_min_str, y_max_str):
        if not (var_secundaria and var_secundaria in df.columns):
            messagebox.showwarning("Aviso", "Gráfico de Dispersión requiere una Segunda Variable (Eje Y).", parent=self.master); return
        if not pd.api.types.is_numeric_dtype(df[var_principal]) or not pd.api.types.is_numeric_dtype(df[var_secundaria]):
            messagebox.showwarning("Aviso", f"Gráfico de Dispersión requiere variables numéricas: '{var_principal}', '{var_secundaria}'.", parent=self.master); return
        data_for_plot = df.copy()
        if self.exclude_blank.get(): data_for_plot.dropna(subset=[var_principal], inplace=True)
        data_for_plot.dropna(subset=[var_principal, var_secundaria], inplace=True)
        if data_for_plot.empty:
             ax.text(0.5, 0.5, "No hay datos válidos", ha='center', va='center', transform=ax.transAxes); return
        if include_regression:
             sns.regplot(x=var_principal, y=var_secundaria, data=data_for_plot, color=color, scatter_kws={'edgecolor': edge}, ax=ax)
        else:
             ax.scatter(data_for_plot[var_principal], data_for_plot[var_secundaria], color=color, edgecolor=edge)
        ax.set_xlabel(x_label or var_principal); ax.set_ylabel(y_label or var_secundaria)
        ax.set_xscale(self.cmb_scale.get()) 
        self._apply_limits(ax, x_min_str, x_max_str, y_min_str, y_max_str) 

# Para pruebas directas
if __name__ == '__main__':
    root = tk.Tk()
    root.title("Prueba Pestaña Análisis Combinado")
    root.geometry("1200x800")
    
    app_tab = CombinedAnalysisTab(root)
    app_tab.pack(fill="both", expand=True)
    root.mainloop()