#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib
from matplotlib.colors import SymLogNorm, ListedColormap
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os # Añadido para manejo de archivos
import traceback # Añadido para logging de errores

# FilterComponent ha sido eliminado.
FilterComponent = None # Mantener para evitar errores si alguna lógica residual lo verifica.

# evitar ventanas externas
matplotlib.use("Agg")

class MapTab(ttk.Frame):
    def __init__(self, master):
        super().__init__(master)

        # Estado de la aplicación
        self.df_original = None # DataFrame cargado
        self.df_filtered = None # DataFrame después de filtros
        self.gdf_mex = None     # GeoDataFrame con geometrías
        self.fig_canvas = None  # Canvas para el gráfico matplotlib
        self.filepath_var = tk.StringVar() # Para mostrar ruta del archivo cargado

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

        # Variables de control Tkinter para la UI
        self.state_col_var = tk.StringVar()
        self.value_col_var = tk.StringVar()
        self.agg_method_var = tk.StringVar(value="count") # count, sum, mean
        self.show_labels = tk.BooleanVar(value=True)
        self.invert_cmap = tk.BooleanVar(value=False)

        self.create_widgets()

    def create_widgets(self):
        paned = ttk.Panedwindow(self, orient="horizontal")
        paned.pack(fill=tk.BOTH, expand=True)

        # --- Panel Izquierdo: Controles ---
        frm_left = ttk.Frame(paned); paned.add(frm_left, weight=1)

        # Frame para controles principales (carga, selección, filtros)
        ctrl_main = ttk.LabelFrame(frm_left, text="Configuración del Mapa")
        ctrl_main.pack(fill=tk.X, padx=5, pady=5)

        # Carga de Archivos
        file_frame = ttk.Frame(ctrl_main)
        file_frame.pack(fill=tk.X, pady=5)
        ttk.Button(file_frame, text="Cargar GeoJSON Estados", command=self.load_shapefile).pack(side=tk.LEFT, padx=5)
        ttk.Button(file_frame, text="Cargar Datos (CSV/Excel)", command=self._load_data).pack(side=tk.LEFT, padx=5)
        ttk.Entry(file_frame, textvariable=self.filepath_var, width=30, state="readonly").pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)

        # Selección de Columnas y Agregación
        select_frame = ttk.Frame(ctrl_main)
        select_frame.pack(fill=tk.X, pady=5)
        ttk.Label(select_frame, text="Columna Estado/Región:").grid(row=0, column=0, padx=5, pady=2, sticky="w")
        self.state_col_combo = ttk.Combobox(select_frame, textvariable=self.state_col_var, state="readonly", width=20)
        self.state_col_combo.grid(row=0, column=1, padx=5, pady=2, sticky="ew")
        ttk.Label(select_frame, text="Columna Valor:").grid(row=1, column=0, padx=5, pady=2, sticky="w")
        self.value_col_combo = ttk.Combobox(select_frame, textvariable=self.value_col_var, state="readonly", width=20)
        self.value_col_combo.grid(row=1, column=1, padx=5, pady=2, sticky="ew")
        ttk.Label(select_frame, text="Agregar por:").grid(row=2, column=0, padx=5, pady=2, sticky="w")
        self.agg_method_combo = ttk.Combobox(select_frame, textvariable=self.agg_method_var, values=["count", "sum", "mean"], state="readonly", width=10)
        self.agg_method_combo.grid(row=2, column=1, padx=5, pady=2, sticky="w")
        select_frame.columnconfigure(1, weight=1)

        # --- Filtros Generales (Implementación directa) ---
        frm_filters_general = ttk.LabelFrame(ctrl_main, text="Filtros Generales (Opcional)")
        frm_filters_general.pack(fill="x", pady=5) # Se añade al ctrl_main

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

        # Controles de Apariencia del Mapa (Mantenidos)
        ctrl_appearance = ttk.LabelFrame(frm_left, text="Apariencia del Mapa")
        ctrl_appearance.pack(fill=tk.X, padx=5, pady=5)

        # Paleta
        appearance_row1 = ttk.Frame(ctrl_appearance)
        appearance_row1.pack(fill=tk.X, pady=2)
        ttk.Label(appearance_row1, text="Paleta:").pack(side=tk.LEFT, padx=5)
        palettes = ["viridis","plasma","inferno","magma","cividis","turbo","Spectral","coolwarm","copper","winter","summer","autumn","spring","hot","bone"]
        self.cmb_palette = ttk.Combobox(appearance_row1, values=palettes, state="readonly", width=10); self.cmb_palette.pack(side=tk.LEFT, padx=5); self.cmb_palette.set("Spectral")
        ttk.Label(appearance_row1, text="N° colores:").pack(side=tk.LEFT, padx=5)
        self.ent_pal_n = ttk.Entry(appearance_row1, width=5); self.ent_pal_n.pack(side=tk.LEFT, padx=5); self.ent_pal_n.insert(0,"0")
        ttk.Checkbutton(appearance_row1, text="Invertir", variable=self.invert_cmap).pack(side=tk.LEFT, padx=5)

        # Escala y Rango
        appearance_row2 = ttk.Frame(ctrl_appearance)
        appearance_row2.pack(fill=tk.X, pady=2)
        ttk.Label(appearance_row2, text="Escala:").pack(side=tk.LEFT, padx=5)
        self.cmb_scale = ttk.Combobox(appearance_row2, values=["Regular","Logarítmica"], state="readonly", width=10); self.cmb_scale.pack(side=tk.LEFT, padx=5); self.cmb_scale.set("Logarítmica")
        ttk.Label(appearance_row2, text="Rango (min–max):").pack(side=tk.LEFT, padx=5)
        self.ent_vmin = ttk.Entry(appearance_row2, width=8); self.ent_vmin.pack(side=tk.LEFT, padx=5); self.ent_vmin.insert(0,"0")
        self.ent_vmax = ttk.Entry(appearance_row2, width=8); self.ent_vmax.pack(side=tk.LEFT, padx=5)

        # Ticks Colorbar
        appearance_row3 = ttk.Frame(ctrl_appearance)
        appearance_row3.pack(fill=tk.X, pady=2)
        ttk.Label(appearance_row3, text="N° Ticks CB:").pack(side=tk.LEFT, padx=5)
        self.ent_nt = ttk.Entry(appearance_row3,width=5); self.ent_nt.pack(side=tk.LEFT, padx=5); self.ent_nt.insert(0,"6")
        ttk.Label(appearance_row3, text="Valores CB (coma):").pack(side=tk.LEFT, padx=5)
        self.ent_vals = ttk.Entry(appearance_row3,width=15); self.ent_vals.pack(side=tk.LEFT, padx=5)

        # Etiquetas y DPI
        appearance_row4 = ttk.Frame(ctrl_appearance)
        appearance_row4.pack(fill=tk.X, pady=2)
        ttk.Checkbutton(appearance_row4, text="Mostrar etiquetas estados", variable=self.show_labels).pack(side=tk.LEFT, padx=5)
        ttk.Label(appearance_row4, text="DPI:").pack(side=tk.LEFT, padx=15)
        self.ent_dpi = ttk.Entry(appearance_row4,width=5); self.ent_dpi.pack(side=tk.LEFT, padx=5); self.ent_dpi.insert(0,"100")
        ttk.Label(appearance_row4, text="Grosor línea:").pack(side=tk.LEFT, padx=5)
        self.ent_lw = ttk.Entry(appearance_row4,width=5); self.ent_lw.pack(side=tk.LEFT, padx=5); self.ent_lw.insert(0,"1.0")

        # Títulos y Textos
        appearance_row5 = ttk.Frame(ctrl_appearance)
        appearance_row5.pack(fill=tk.X, pady=2)
        ttk.Label(appearance_row5, text="Título:").pack(side=tk.LEFT, padx=5)
        self.ent_title = ttk.Entry(appearance_row5,width=20); self.ent_title.pack(side=tk.LEFT, padx=5); self.ent_title.insert(0,"Mapa de México")
        ttk.Label(appearance_row5, text="Color:").pack(side=tk.LEFT, padx=5)
        self.ent_tcol = ttk.Entry(appearance_row5,width=8); self.ent_tcol.pack(side=tk.LEFT, padx=5); self.ent_tcol.insert(0,"black")
        ttk.Label(appearance_row5, text="Tamaño:").pack(side=tk.LEFT, padx=5)
        self.ent_tsz = ttk.Entry(appearance_row5,width=5); self.ent_tsz.pack(side=tk.LEFT, padx=5); self.ent_tsz.insert(0,"14")

        appearance_row6 = ttk.Frame(ctrl_appearance)
        appearance_row6.pack(fill=tk.X, pady=2)
        ttk.Label(appearance_row6, text="Subtítulo:").pack(side=tk.LEFT, padx=5)
        self.ent_sub = ttk.Entry(appearance_row6,width=20); self.ent_sub.pack(side=tk.LEFT, padx=5); self.ent_sub.insert(0,"(Valor Agregado)")
        ttk.Label(appearance_row6, text="Color:").pack(side=tk.LEFT, padx=5)
        self.ent_scol = ttk.Entry(appearance_row6,width=8); self.ent_scol.pack(side=tk.LEFT, padx=5); self.ent_scol.insert(0,"gray")
        ttk.Label(appearance_row6, text="Tamaño:").pack(side=tk.LEFT, padx=5)
        self.ent_ssz = ttk.Entry(appearance_row6,width=5); self.ent_ssz.pack(side=tk.LEFT, padx=5); self.ent_ssz.insert(0,"10")

        appearance_row7 = ttk.Frame(ctrl_appearance)
        appearance_row7.pack(fill=tk.X, pady=2)
        ttk.Label(appearance_row7, text="Título CB:").pack(side=tk.LEFT, padx=5)
        self.ent_cbt = ttk.Entry(appearance_row7,width=20); self.ent_cbt.pack(side=tk.LEFT, padx=5); self.ent_cbt.insert(0,"Valor")
        ttk.Label(appearance_row7, text="Color:").pack(side=tk.LEFT, padx=5)
        self.ent_cbtcol = ttk.Entry(appearance_row7,width=8); self.ent_cbtcol.pack(side=tk.LEFT, padx=5); self.ent_cbtcol.insert(0,"black")
        ttk.Label(appearance_row7, text="Tamaño:").pack(side=tk.LEFT, padx=5)
        self.ent_cbtsz = ttk.Entry(appearance_row7,width=5); self.ent_cbtsz.pack(side=tk.LEFT, padx=5); self.ent_cbtsz.insert(0,"10")

        appearance_row8 = ttk.Frame(ctrl_appearance)
        appearance_row8.pack(fill=tk.X, pady=2)
        ttk.Label(appearance_row8, text="Etiquetas CB Color:").pack(side=tk.LEFT, padx=5)
        self.ent_cbkcol = ttk.Entry(appearance_row8,width=8); self.ent_cbkcol.pack(side=tk.LEFT, padx=5); self.ent_cbkcol.insert(0,"black")
        ttk.Label(appearance_row8, text="Tamaño:").pack(side=tk.LEFT, padx=5)
        self.ent_cbksz = ttk.Entry(appearance_row8,width=5); self.ent_cbksz.pack(side=tk.LEFT, padx=5); self.ent_cbksz.insert(0,"8")

        # Botones Generar / Guardar
        action_frame = ttk.Frame(frm_left)
        action_frame.pack(fill=tk.X, padx=5, pady=10)
        ttk.Button(action_frame, text="Generar Mapa", command=self.show_map).pack(side=tk.LEFT, padx=10)
        ttk.Button(action_frame, text="Guardar Mapa", command=self.save_map).pack(side=tk.LEFT, padx=10)

        # Log Text (movido al final del panel izquierdo)
        self.txt_out = tk.Text(frm_left, height=4); self.txt_out.pack(fill=tk.BOTH, padx=5, pady=5, expand=True)

        # --- Panel Derecho: Mapa ---
        frm_right = ttk.Frame(paned); paned.add(frm_right, weight=2) # Aumentar peso para dar más espacio al mapa
        self.canvas_map = tk.Canvas(frm_right,bg="white"); self.canvas_map.pack(fill=tk.BOTH,expand=True)
        v2 = ttk.Scrollbar(frm_right,orient="vertical",command=self.canvas_map.yview); v2.pack(side=tk.RIGHT,fill=tk.Y)
        h2 = ttk.Scrollbar(frm_right,orient="horizontal",command=self.canvas_map.xview); h2.pack(side=tk.BOTTOM,fill=tk.X)
        self.canvas_map.configure(yscrollcommand=v2.set, xscrollcommand=h2.set)
        self.frm_map = ttk.Frame(self.canvas_map); self.canvas_map.create_window((0,0),window=self.frm_map,anchor="nw")
        self.frm_map.bind("<Configure>", lambda e: self.canvas_map.configure(scrollregion=self.canvas_map.bbox("all")))
        self.txt_out = tk.Text(frm_left,height=4); self.txt_out.pack(fill=tk.BOTH,padx=5,pady=5)

    def load_shapefile(self):
        path = filedialog.askopenfilename(title="GeoJSON", filetypes=[("GeoJSON","*.json"),("All","*.*")])
        if not path: return
        try:
            self.gdf_mex = gpd.read_file(path)
            self.txt_out.insert(tk.END, f"Shapefile: {path}\n")
            messagebox.showinfo("Éxito","Cargado shapefile")
        except Exception as e:
            messagebox.showerror("Error",str(e))

    # Se elimina load_population, la población debe venir de los datos principales si se usa Prevalencia
    # def load_population(self): ...

    def _load_data(self):
        """Carga datos principales (CSV/Excel) y actualiza controles."""
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
            self._update_column_selectors()
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
            self.log(f"Datos cargados: {self.df_original.shape}")

        except Exception as e:
            messagebox.showerror("Error al Leer Archivo", f"No se pudo leer el archivo:\n{e}")
            self.df_original = None; self.filepath_var.set(""); self._update_column_selectors()
            # FilterComponent removido.
            # Limpiar combos de filtros generales en caso de error
            if hasattr(self, 'filter_col_1_combo'): self.filter_col_1_combo['values'] = ['']
            if hasattr(self, 'filter_col_2_combo'): self.filter_col_2_combo['values'] = ['']
            self.log(f"Error cargando archivo: {e}")

    def _update_column_selectors(self):
        """Actualiza los comboboxes de selección de columnas."""
        cols = sorted(self.df_original.columns.tolist()) if self.df_original is not None else []
        numeric_cols = self.df_original.select_dtypes(include=np.number).columns.tolist() if self.df_original is not None else []

        self.state_col_combo['values'] = [""] + cols
        self.value_col_combo['values'] = [""] + numeric_cols # Para Sum/Mean, Count no necesita valor

        self.state_col_var.set("")
        self.value_col_var.set("")
        # Mantener método de agregación actual

    def _get_aggregated_data(self):
        """Aplica filtros y agrega los datos por estado/región."""
        if self.df_original is None:
            self.log("No hay datos originales cargados."); return None
        if not self.state_col_var.get():
            self.log("Seleccione la columna de Estado/Región."); return None
        state_col = self.state_col_var.get()
        agg_method = self.agg_method_var.get()
        value_col = self.value_col_var.get()

        if agg_method in ["sum", "mean"] and not value_col:
            self.log(f"Seleccione la columna de Valor para agregar por '{agg_method}'."); return None

        # 1. FilterComponent ha sido removido. Se trabaja directamente con una copia de self.df_original.
        if self.df_original is None: # Asegurarse que df_original existe
            self.log("df_original no está cargado en _get_aggregated_data.", "ERROR")
            return None
        
        df_initial = self.df_original.copy()
        self.log("Usando datos originales para agregación (antes de filtros generales).", "INFO")

        # Aplicar filtros generales definidos en la UI
        df_filtered = self._apply_general_filters(df_initial)

        if df_filtered is None or df_filtered.empty:
            self.log("No hay datos después de aplicar filtros generales en _get_aggregated_data."); return None

        # 2. Agrupar y agregar
        if state_col not in df_filtered.columns:
             self.log(f"Columna de estado '{state_col}' no encontrada en datos filtrados."); return None

        try:
            grouped = df_filtered.groupby(state_col)
            if agg_method == "count":
                aggregated_data = grouped.size().reset_index(name='Metric')
            elif agg_method == "sum":
                if value_col not in df_filtered.columns or not pd.api.types.is_numeric_dtype(df_filtered[value_col]):
                     self.log(f"Columna de valor '{value_col}' no es numérica o no existe."); return None
                aggregated_data = grouped[value_col].sum().reset_index(name='Metric')
            elif agg_method == "mean":
                if value_col not in df_filtered.columns or not pd.api.types.is_numeric_dtype(df_filtered[value_col]):
                     self.log(f"Columna de valor '{value_col}' no es numérica o no existe."); return None
                aggregated_data = grouped[value_col].mean().reset_index(name='Metric')
            else:
                self.log(f"Método de agregación desconocido: {agg_method}"); return None

            self.log(f"Datos agregados por '{state_col}' usando '{agg_method}'. {len(aggregated_data)} estados/regiones encontrados.")
            return aggregated_data

        except Exception as e:
            self.log(f"Error durante la agregación: {e}"); traceback.print_exc(); return None


    def show_map(self):
        fig = self.make_fig()
        if not fig: return
        if self.fig_canvas: self.fig_canvas.get_tk_widget().destroy()
        self.fig_canvas = FigureCanvasTkAgg(fig,master=self.frm_map)
        self.fig_canvas.draw(); self.fig_canvas.get_tk_widget().pack(fill=tk.BOTH,expand=True)
        self.canvas_map.config(scrollregion=self.canvas_map.bbox("all"))

    def make_fig(self):
        """Genera la figura del mapa basada en los datos agregados y filtrados."""
        df_agg = self._get_aggregated_data() # Obtener datos agregados
        state_col = self.state_col_var.get() # Nombre de la columna de estado en df_agg

        if df_agg is None or df_agg.empty:
            messagebox.showwarning("Aviso", "No se pudieron generar datos agregados para el mapa."); return None
        if self.gdf_mex is None:
             messagebox.showwarning("Aviso", "Cargue primero el archivo GeoJSON de estados."); return None
        if not state_col:
             messagebox.showwarning("Aviso", "Seleccione la columna de Estado/Región."); return None

        # --- Obtener parámetros de apariencia ---
        pal     = self.cmb_palette.get()
        ncol    = int(self.ent_pal_n.get()) if self.ent_pal_n.get().isdigit() else 0
        inv     = self.invert_cmap.get()
        scale   = self.cmb_scale.get()
        dpi     = int(self.ent_dpi.get())
        lw      = float(self.ent_lw.get())
        vmin    = float(self.ent_vmin.get())
        vmax_s  = self.ent_vmax.get().strip()
        vmax    = float(vmax_s) if vmax_s else None
        nt      = int(self.ent_nt.get())
        vals    = self.ent_vals.get().strip()

        # text styles with fallback
        title, tcol, tsz = self.ent_title.get().strip(), (self.ent_tcol.get().strip() or "black"), float(self.ent_tsz.get())
        subt, scol, ssz = self.ent_sub.get().strip(), (self.ent_scol.get().strip() or "gray"), float(self.ent_ssz.get())
        cbt, cbtcol, cbtsz = self.ent_cbt.get().strip(), (self.ent_cbtcol.get().strip() or "black"), float(self.ent_cbtsz.get())
        cbkcol, cbksz = (self.ent_cbkcol.get().strip() or "black"), float(self.ent_cbksz.get())

        # --- Preparar datos para el mapa ---
        col_to_plot = "Metric" # Columna generada por _get_aggregated_data
        if col_to_plot not in df_agg.columns:
             self.log(f"Columna '{col_to_plot}' no encontrada en datos agregados."); return None

        # Intentar determinar el nombre de la columna de estado en el GeoDataFrame
        # Comunes son 'NAME_1', 'NOMGEO', 'estado', etc. Intentaremos con 'NAME_1' primero.
        geojson_state_col = 'NAME_1'
        if geojson_state_col not in self.gdf_mex.columns:
            # Intentar encontrar una columna candidata
            possible_cols = [c for c in self.gdf_mex.columns if 'name' in c.lower() or 'nom' in c.lower() or 'estado' in c.lower()]
            if possible_cols:
                geojson_state_col = possible_cols[0]
                self.log(f"Usando columna '{geojson_state_col}' del GeoJSON para unir.", "INFO")
            else:
                messagebox.showerror("Error GeoJSON", "No se encontró una columna de nombre de estado adecuada en el GeoJSON (ej. 'NAME_1', 'NOMGEO').")
                return None

        # Merge GeoDataFrame con datos agregados
        gdf = self.gdf_mex.merge(df_agg, how="left", left_on=geojson_state_col, right_on=state_col)
        gdf[col_to_plot] = gdf[col_to_plot].fillna(0) # Rellenar NaNs con 0 para estados sin datos

        # Calcular vmax si no se especificó
        default_vmax = gdf[col_to_plot].max() if not gdf[col_to_plot].empty else 1
        if vmax is None: vmax = default_vmax
        # Determinar umbral para escala logarítmica (si se usa)
        thresh = 0.1 if vmax > 10 else 0.01 # Ajustar umbral basado en rango

        # Normalización
        if scale=="Logarítmica":
            norm = SymLogNorm(linthresh=thresh, linscale=1, vmin=vmin, vmax=vmax)
        else:
            norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)

        # colormap
        base = matplotlib.colormaps[pal]
        if ncol>0:
            colors = base(np.linspace(0,1,ncol))
            cmap = ListedColormap(colors)
        else:
            cmap = base
        if inv:
            cmap = cmap.reversed()

        # draw
        fig = Figure(figsize=(10,8), dpi=dpi); ax = fig.add_subplot(111)
        gdf.plot(column=col_to_plot, cmap=cmap, norm=norm, edgecolor="black", linewidth=lw, ax=ax, missing_kwds={'color': 'lightgrey', "hatch": "///", "label": "Sin datos"})
        ax.set_axis_off()
        # Usar nombre de columna de valor o método de agregación en título/subtítulo
        agg_method_display = self.agg_method_var.get()
        value_col_display = self.value_col_var.get()
        metric_display = f"{agg_method_display}({value_col_display})" if value_col_display and agg_method_display != 'count' else f"{agg_method_display}"
        default_title = title if title else f"Mapa de México - {metric_display}"
        default_subtitle = subt if subt else f"Agregado por {state_col}"
        ax.set_title(default_title, color=tcol, fontsize=tsz, pad=20)
        ax.text(0.5, 0.96, default_subtitle, transform=ax.transAxes, ha='center', color=scol, fontsize=ssz)

        # Colorbar
        sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap); sm._A=[]
        if vals:
            try: ticks = [float(x) for x in vals.split(",")]
            except: ticks = np.linspace(vmin,vmax,nt)
        else:
            ticks = np.linspace(vmin,vmax,nt)
        cbar = fig.colorbar(sm, ax=ax, ticks=ticks)
        for spine in cbar.ax.spines.values():
            spine.set_edgecolor(cbtcol); spine.set_linewidth(1)
        cbar.ax.tick_params(color=cbkcol, labelcolor=cbkcol, width=1)
        cbar.ax.set_yticklabels([f"{t:.2f}" for t in ticks], fontsize=cbksz, color=cbkcol)
        default_cbt = cbt if cbt else metric_display # Usar métrica como título default de CB
        cbar.set_label(default_cbt, fontsize=cbtsz, color=cbtcol)

        # Annotations
        if self.show_labels.get():
            for _,r in gdf.iterrows():
                if r.geometry is not None and pd.notnull(r[col_to_plot]):
                    pt = r.geometry.representative_point()
                    val_to_show = r[col_to_plot]
                    # Formatear texto según el valor
                    if abs(val_to_show) >= 1000: txt = f"{val_to_show:,.0f}"
                    elif abs(val_to_show) >= 10: txt = f"{val_to_show:,.1f}"
                    elif abs(val_to_show) >= 0.1: txt = f"{val_to_show:.2f}"
                    else: txt = f"{val_to_show:.2e}" # Notación científica para muy pequeños
                    ax.annotate(txt, xy=(pt.x,pt.y), ha='center', fontsize=cbksz, color=cbkcol)

        fig.tight_layout()
        return fig

    def save_map(self):
        path = filedialog.asksaveasfilename(defaultextension=".png",
            filetypes=[("PNG","*.png"),("JPG","*.jpg"),("All","*.*")])
        if not path: return
        fig = self.make_fig()
        if not fig: return
        try:
            fig.savefig(path)
            messagebox.showinfo("Éxito", f"Guardado en:\n{path}")
        except Exception as e:
            messagebox.showerror("Error", str(e))


    def log(self, message, level="INFO"):
        """Añade mensaje al área de texto de log."""
        try:
            self.txt_out.insert(tk.END, f"[{level}] {message}\n")
            self.txt_out.see(tk.END) # Auto-scroll
        except Exception as e:
            print(f"Error en log: {e}") # Fallback a consola

if __name__=="__main__":
    root = tk.Tk()
    root.title("Mapa Coroplético de México desde Datos")
    app = MapTab(root)
    app.pack(fill=tk.BOTH, expand=True)
    root.mainloop()


