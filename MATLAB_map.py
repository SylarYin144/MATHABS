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
        self.df_original = None
        self.df_filtered = None
        self.gdf_mex = None
        self.fig_canvas = None
        self.filepath_var = tk.StringVar()

        self.filter_active_1_var = tk.BooleanVar(value=False)
        self.filter_col_1_var = tk.StringVar()
        self.filter_op_1_var = tk.StringVar()
        self.filter_val_1_var = tk.StringVar()

        self.filter_active_2_var = tk.BooleanVar(value=False)
        self.filter_col_2_var = tk.StringVar()
        self.filter_op_2_var = tk.StringVar()
        self.filter_val_2_var = tk.StringVar()

        self.general_filter_operators = ["==", "!=", ">", "<", ">=", "<=", "contiene", "no contiene", "es NaN", "no es NaN"]

        self.state_col_var = tk.StringVar()
        self.value_col_var = tk.StringVar()
        self.agg_method_var = tk.StringVar(value="count")
        self.show_labels = tk.BooleanVar(value=True)
        self.invert_cmap = tk.BooleanVar(value=False)
        self.state_entries = {}

        self.MANUAL_STATE_DATA_OPTION = "(Usar Datos de Estado Manuales)"
        self.geojson_state_column_name = None

        self.default_population_data = {
            "AGUASCALIENTES": 1425607, "BAJA CALIFORNIA": 3769020, "BAJA CALIFORNIA SUR": 798447,
            "CAMPECHE": 928363, "CHIAPAS": 5543828, "CHIHUAHUA": 3741869,
            "COAHUILA DE ZARAGOZA": 3146771, "COLIMA": 731391, "CIUDAD DE MÉXICO": 9209944,
            "DISTRITO FEDERAL": 9209944, # Alias for CDMX
            "DURANGO": 1832650, "GUANAJUATO": 6166934, "GUERRERO": 3540685,
            "HIDALGO": 3082841, "JALISCO": 8348151, "MÉXICO": 16992418,
            "MICHOACÁN DE OCAMPO": 4748846, "MORELOS": 1971520, "NAYARIT": 1235456,
            "NUEVO LEÓN": 5784442, "OAXACA": 4132148, "PUEBLA": 6583278,
            "QUERÉTARO": 2368467, "QUINTANA ROO": 1857985, "SAN LUIS POTOSÍ": 2822255,
            "SINALOA": 3026943, "SONORA": 2944840, "TABASCO": 2402598,
            "TAMAULIPAS": 3527735, "TLAXCALA": 1342977, "VERACRUZ DE IGNACIO DE LA LLAVE": 8062579,
            "YUCATÁN": 2320898, "ZACATECAS": 1622138
        }
        self.create_widgets()

    def create_widgets(self):
        paned = ttk.Panedwindow(self, orient="horizontal")
        paned.pack(fill=tk.BOTH, expand=True)
        frm_left = ttk.Frame(paned); paned.add(frm_left, weight=1)

        ctrl_main = ttk.LabelFrame(frm_left, text="Configuración del Mapa")
        ctrl_main.pack(fill=tk.X, padx=5, pady=5)

        file_frame = ttk.Frame(ctrl_main)
        file_frame.pack(fill=tk.X, pady=5)
        ttk.Button(file_frame, text="Cargar GeoJSON Estados", command=self.load_shapefile).pack(side=tk.LEFT, padx=5)
        ttk.Button(file_frame, text="Cargar Datos (CSV/Excel)", command=self._load_data).pack(side=tk.LEFT, padx=5)
        ttk.Entry(file_frame, textvariable=self.filepath_var, width=30, state="readonly").pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)

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

        frm_filters_general = ttk.LabelFrame(ctrl_main, text="Filtros Generales (Opcional)")
        frm_filters_general.pack(fill="x", pady=5)
        f1_frame = ttk.Frame(frm_filters_general); f1_frame.pack(fill="x", pady=2)
        ttk.Checkbutton(f1_frame, text="Activar Filtro 1:", variable=self.filter_active_1_var).grid(row=0, column=0, padx=2, sticky="w")
        self.filter_col_1_combo = ttk.Combobox(f1_frame, textvariable=self.filter_col_1_var, state="readonly", width=15); self.filter_col_1_combo.grid(row=0, column=1, padx=2)
        self.filter_op_1_combo = ttk.Combobox(f1_frame, textvariable=self.filter_op_1_var, values=self.general_filter_operators, state="readonly", width=10); self.filter_op_1_combo.grid(row=0, column=2, padx=2); self.filter_op_1_combo.set("==")
        ttk.Entry(f1_frame, textvariable=self.filter_val_1_var, width=15).grid(row=0, column=3, padx=2)
        f2_frame = ttk.Frame(frm_filters_general); f2_frame.pack(fill="x", pady=2)
        ttk.Checkbutton(f2_frame, text="Activar Filtro 2:", variable=self.filter_active_2_var).grid(row=0, column=0, padx=2, sticky="w")
        self.filter_col_2_combo = ttk.Combobox(f2_frame, textvariable=self.filter_col_2_var, state="readonly", width=15); self.filter_col_2_combo.grid(row=0, column=1, padx=2)
        self.filter_op_2_combo = ttk.Combobox(f2_frame, textvariable=self.filter_op_2_var, values=self.general_filter_operators, state="readonly", width=10); self.filter_op_2_combo.grid(row=0, column=2, padx=2); self.filter_op_2_combo.set("==")
        ttk.Entry(f2_frame, textvariable=self.filter_val_2_var, width=15).grid(row=0, column=3, padx=2)

        state_data_lf = ttk.LabelFrame(frm_left, text="Datos por Estado")
        state_data_lf.pack(fill=tk.X, padx=5, pady=5)
        state_data_canvas_frame = ttk.Frame(state_data_lf); state_data_canvas_frame.pack(fill=tk.BOTH, expand=True)
        state_data_canvas = tk.Canvas(state_data_canvas_frame); state_data_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        state_data_scrollbar = ttk.Scrollbar(state_data_canvas_frame, orient=tk.VERTICAL, command=state_data_canvas.yview); state_data_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        state_data_canvas.configure(yscrollcommand=state_data_scrollbar.set)
        self.state_data_frame = ttk.Frame(state_data_canvas); state_data_canvas.create_window((0, 0), window=self.state_data_frame, anchor="nw")
        self.state_data_frame.bind("<Configure>", lambda e: state_data_canvas.configure(scrollregion=state_data_canvas.bbox("all")))

        restore_button = ttk.Button(state_data_lf, text="Restaurar Población por Defecto", command=self.restore_default_population)
        restore_button.pack(pady=(5,0)) # Add some padding top, none bottom

        # MODIFIED FOR STEP 5: Add Load Population from CSV button
        load_csv_button = ttk.Button(state_data_lf, text="Cargar Población desde CSV", command=self.load_population_from_csv)
        load_csv_button.pack(pady=(2,5)) # Add some padding top and bottom

        ctrl_appearance = ttk.LabelFrame(frm_left, text="Apariencia del Mapa"); ctrl_appearance.pack(fill=tk.X, padx=5, pady=5)
        appearance_row1 = ttk.Frame(ctrl_appearance); appearance_row1.pack(fill=tk.X, pady=2)
        ttk.Label(appearance_row1, text="Paleta:").pack(side=tk.LEFT, padx=5)
        palettes = ["viridis","plasma","inferno","magma","cividis","turbo","Spectral","coolwarm","copper","winter","summer","autumn","spring","hot","bone"]
        self.cmb_palette = ttk.Combobox(appearance_row1, values=palettes, state="readonly", width=10); self.cmb_palette.pack(side=tk.LEFT, padx=5); self.cmb_palette.set("Spectral")
        ttk.Label(appearance_row1, text="N° colores:").pack(side=tk.LEFT, padx=5)
        self.ent_pal_n = ttk.Entry(appearance_row1, width=5); self.ent_pal_n.pack(side=tk.LEFT, padx=5); self.ent_pal_n.insert(0,"0")
        ttk.Checkbutton(appearance_row1, text="Invertir", variable=self.invert_cmap).pack(side=tk.LEFT, padx=5)
        # ... (rest of create_widgets method remains largely the same) ...
        appearance_row2 = ttk.Frame(ctrl_appearance); appearance_row2.pack(fill=tk.X, pady=2)
        ttk.Label(appearance_row2, text="Escala:").pack(side=tk.LEFT, padx=5)
        self.cmb_scale = ttk.Combobox(appearance_row2, values=["Regular","Logarítmica"], state="readonly", width=10); self.cmb_scale.pack(side=tk.LEFT, padx=5); self.cmb_scale.set("Logarítmica")
        ttk.Label(appearance_row2, text="Rango (min–max):").pack(side=tk.LEFT, padx=5)
        self.ent_vmin = ttk.Entry(appearance_row2, width=8); self.ent_vmin.pack(side=tk.LEFT, padx=5); self.ent_vmin.insert(0,"0")
        self.ent_vmax = ttk.Entry(appearance_row2, width=8); self.ent_vmax.pack(side=tk.LEFT, padx=5)
        appearance_row3 = ttk.Frame(ctrl_appearance); appearance_row3.pack(fill=tk.X, pady=2)
        ttk.Label(appearance_row3, text="N° Ticks CB:").pack(side=tk.LEFT, padx=5)
        self.ent_nt = ttk.Entry(appearance_row3,width=5); self.ent_nt.pack(side=tk.LEFT, padx=5); self.ent_nt.insert(0,"6")
        ttk.Label(appearance_row3, text="Valores CB (coma):").pack(side=tk.LEFT, padx=5)
        self.ent_vals = ttk.Entry(appearance_row3,width=15); self.ent_vals.pack(side=tk.LEFT, padx=5)
        appearance_row4 = ttk.Frame(ctrl_appearance); appearance_row4.pack(fill=tk.X, pady=2)
        ttk.Checkbutton(appearance_row4, text="Mostrar etiquetas estados", variable=self.show_labels).pack(side=tk.LEFT, padx=5)
        ttk.Label(appearance_row4, text="DPI:").pack(side=tk.LEFT, padx=15)
        self.ent_dpi = ttk.Entry(appearance_row4,width=5); self.ent_dpi.pack(side=tk.LEFT, padx=5); self.ent_dpi.insert(0,"100")
        ttk.Label(appearance_row4, text="Grosor línea:").pack(side=tk.LEFT, padx=5)
        self.ent_lw = ttk.Entry(appearance_row4,width=5); self.ent_lw.pack(side=tk.LEFT, padx=5); self.ent_lw.insert(0,"1.0")
        appearance_row5 = ttk.Frame(ctrl_appearance); appearance_row5.pack(fill=tk.X, pady=2)
        ttk.Label(appearance_row5, text="Título:").pack(side=tk.LEFT, padx=5)
        self.ent_title = ttk.Entry(appearance_row5,width=20); self.ent_title.pack(side=tk.LEFT, padx=5); self.ent_title.insert(0,"Mapa de México")
        ttk.Label(appearance_row5, text="Color:").pack(side=tk.LEFT, padx=5)
        self.ent_tcol = ttk.Entry(appearance_row5,width=8); self.ent_tcol.pack(side=tk.LEFT, padx=5); self.ent_tcol.insert(0,"black")
        ttk.Label(appearance_row5, text="Tamaño:").pack(side=tk.LEFT, padx=5)
        self.ent_tsz = ttk.Entry(appearance_row5,width=5); self.ent_tsz.pack(side=tk.LEFT, padx=5); self.ent_tsz.insert(0,"14")
        appearance_row6 = ttk.Frame(ctrl_appearance); appearance_row6.pack(fill=tk.X, pady=2)
        ttk.Label(appearance_row6, text="Subtítulo:").pack(side=tk.LEFT, padx=5)
        self.ent_sub = ttk.Entry(appearance_row6,width=20); self.ent_sub.pack(side=tk.LEFT, padx=5); self.ent_sub.insert(0,"(Valor Agregado)")
        ttk.Label(appearance_row6, text="Color:").pack(side=tk.LEFT, padx=5)
        self.ent_scol = ttk.Entry(appearance_row6,width=8); self.ent_scol.pack(side=tk.LEFT, padx=5); self.ent_scol.insert(0,"gray")
        ttk.Label(appearance_row6, text="Tamaño:").pack(side=tk.LEFT, padx=5)
        self.ent_ssz = ttk.Entry(appearance_row6,width=5); self.ent_ssz.pack(side=tk.LEFT, padx=5); self.ent_ssz.insert(0,"10")
        appearance_row7 = ttk.Frame(ctrl_appearance); appearance_row7.pack(fill=tk.X, pady=2)
        ttk.Label(appearance_row7, text="Título CB:").pack(side=tk.LEFT, padx=5)
        self.ent_cbt = ttk.Entry(appearance_row7,width=20); self.ent_cbt.pack(side=tk.LEFT, padx=5); self.ent_cbt.insert(0,"Valor")
        ttk.Label(appearance_row7, text="Color:").pack(side=tk.LEFT, padx=5)
        self.ent_cbtcol = ttk.Entry(appearance_row7,width=8); self.ent_cbtcol.pack(side=tk.LEFT, padx=5); self.ent_cbtcol.insert(0,"black")
        ttk.Label(appearance_row7, text="Tamaño:").pack(side=tk.LEFT, padx=5)
        self.ent_cbtsz = ttk.Entry(appearance_row7,width=5); self.ent_cbtsz.pack(side=tk.LEFT, padx=5); self.ent_cbtsz.insert(0,"10")
        appearance_row8 = ttk.Frame(ctrl_appearance); appearance_row8.pack(fill=tk.X, pady=2)
        ttk.Label(appearance_row8, text="Etiquetas CB Color:").pack(side=tk.LEFT, padx=5)
        self.ent_cbkcol = ttk.Entry(appearance_row8,width=8); self.ent_cbkcol.pack(side=tk.LEFT, padx=5); self.ent_cbkcol.insert(0,"black")
        ttk.Label(appearance_row8, text="Tamaño:").pack(side=tk.LEFT, padx=5)
        self.ent_cbksz = ttk.Entry(appearance_row8,width=5); self.ent_cbksz.pack(side=tk.LEFT, padx=5); self.ent_cbksz.insert(0,"8")

        action_frame = ttk.Frame(frm_left); action_frame.pack(fill=tk.X, padx=5, pady=10)
        ttk.Button(action_frame, text="Generar Mapa", command=self.show_map).pack(side=tk.LEFT, padx=10)
        ttk.Button(action_frame, text="Guardar Mapa", command=self.save_map).pack(side=tk.LEFT, padx=10)
        self.txt_out = tk.Text(frm_left, height=4); self.txt_out.pack(fill=tk.BOTH, padx=5, pady=5, expand=True)
        frm_right = ttk.Frame(paned); paned.add(frm_right, weight=2)
        self.canvas_map = tk.Canvas(frm_right,bg="white"); self.canvas_map.pack(fill=tk.BOTH,expand=True)
        v2 = ttk.Scrollbar(frm_right,orient="vertical",command=self.canvas_map.yview); v2.pack(side=tk.RIGHT,fill=tk.Y)
        h2 = ttk.Scrollbar(frm_right,orient="horizontal",command=self.canvas_map.xview); h2.pack(side=tk.BOTTOM,fill=tk.X)
        self.canvas_map.configure(yscrollcommand=v2.set, xscrollcommand=h2.set)
        self.frm_map = ttk.Frame(self.canvas_map); self.canvas_map.create_window((0,0),window=self.frm_map,anchor="nw")
        self.frm_map.bind("<Configure>", lambda e: self.canvas_map.configure(scrollregion=self.canvas_map.bbox("all")))

    def load_population_from_csv(self):
        if not self.geojson_state_column_name:
            messagebox.showerror("Error", "Cargue primero un archivo GeoJSON para identificar los nombres de los estados.")
            self.log("Intento de cargar CSV de población sin GeoJSON cargado.", "ERROR")
            return

        if not self.state_entries:
            messagebox.showwarning("Advertencia", "No hay campos de estado para poblar. Asegúrese de que un GeoJSON esté cargado y procesado.")
            self.log("Intento de cargar CSV de población sin campos de estado en la UI.", "WARNING")
            return

        filepath = filedialog.askopenfilename(
            title="Seleccionar archivo CSV de población",
            filetypes=(("Archivos CSV", "*.csv"), ("Todos los archivos", "*.*"))
        )
        if not filepath:
            return

        try:
            df_pop_csv = pd.read_csv(filepath)
            self.log(f"CSV de población cargado: {filepath}", "INFO")

            csv_state_col = self.geojson_state_column_name

            potential_pop_cols = ["Poblacion", "POBLACION", "Population", "POP", "Población"]
            actual_population_col = None
            for col in potential_pop_cols:
                if col in df_pop_csv.columns:
                    actual_population_col = col
                    break

            if csv_state_col not in df_pop_csv.columns:
                messagebox.showerror("Error de CSV", f"El CSV debe contener una columna de nombres de estado llamada '{csv_state_col}' (coincidente con la columna del GeoJSON).")
                self.log(f"CSV de población no contiene la columna requerida para nombres de estado: {csv_state_col}", "ERROR")
                return

            if not actual_population_col:
                messagebox.showerror("Error de CSV", f"El CSV debe contener una columna de datos de población (ej: {', '.join(potential_pop_cols)}).")
                self.log(f"CSV de población no contiene una columna de población reconocida.", "ERROR")
                return

            updated_count = 0
            not_found_in_map = []

            for index, row in df_pop_csv.iterrows():
                try:
                    csv_state_name_original = row[csv_state_col]
                    population_val = row[actual_population_col]

                    if pd.isna(csv_state_name_original) or pd.isna(population_val):
                        self.log(f"Valor nulo encontrado en CSV para estado o población en fila {index + 2}, saltando.", "WARNING")
                        continue

                    csv_state_name_normalized = str(csv_state_name_original).upper()
                    matched_entry_widget = None
                    matched_original_geojson_state_name = None

                    for geojson_name_key_original, entry_widget_val in self.state_entries.items():
                        if str(geojson_name_key_original).upper() == csv_state_name_normalized:
                            matched_entry_widget = entry_widget_val
                            matched_original_geojson_state_name = geojson_name_key_original
                            break

                    if matched_entry_widget:
                        matched_entry_widget.delete(0, tk.END)
                        matched_entry_widget.insert(0, str(population_val))

                        if matched_original_geojson_state_name: # Should always be true if matched_entry_widget is true
                             self.default_population_data[matched_original_geojson_state_name] = population_val
                        updated_count += 1
                    else:
                        not_found_in_map.append(str(csv_state_name_original))

                except Exception as e_row:
                    self.log(f"Error procesando fila {index + 2} del CSV de población: {e_row}", "ERROR")
                    continue

            self.log(f"{updated_count} campos de población actualizados desde CSV.", "INFO")
            if not_found_in_map:
                self.log(f"Estados del CSV no encontrados en el mapa: {', '.join(not_found_in_map)}", "WARNING")
                messagebox.showwarning("Advertencia de Carga", f"{updated_count} estados actualizados.\n\nEstados del CSV no encontrados/coincidentes en el mapa (se ignoraron):\n{', '.join(not_found_in_map[:10])}{'...' if len(not_found_in_map) > 10 else ''}")
            else:
                messagebox.showinfo("Éxito", f"{updated_count} campos de población actualizados desde el archivo CSV.")

        except Exception as e_file:
            messagebox.showerror("Error al Leer CSV", f"No se pudo leer o procesar el archivo CSV de población:\n{e_file}")
            self.log(f"Error crítico al cargar/procesar CSV de población: {e_file}", "ERROR")
            traceback.print_exc()

    def load_shapefile(self):
        path = filedialog.askopenfilename(title="GeoJSON", filetypes=[("GeoJSON","*.json"),("All","*.*")])
        if not path: return
        try:
            if hasattr(self, 'state_data_frame') and self.state_data_frame.winfo_exists():
                for widget in self.state_data_frame.winfo_children():
                    widget.destroy()
            if hasattr(self, 'state_entries'):
                self.state_entries.clear()
            self.geojson_state_column_name = None

            self.gdf_mex = gpd.read_file(path)

            geojson_state_col_candidates = ['NAME_1', 'NOMGEO', 'estado', 'nombre']
            identified_geojson_col = None
            for col_name in geojson_state_col_candidates:
                if col_name in self.gdf_mex.columns:
                    identified_geojson_col = col_name
                    break

            if not identified_geojson_col:
                self.log("No standard state name column found in GeoJSON. Attempting to use first string column.", "WARNING")
                for col in self.gdf_mex.columns:
                    if self.gdf_mex[col].dtype == 'object':
                        identified_geojson_col = col
                        self.log(f"Using fallback GeoJSON state column: {identified_geojson_col}", "INFO")
                        break
                if not identified_geojson_col:
                    messagebox.showerror("Error GeoJSON", "No se pudo identificar la columna de nombres de estado en el GeoJSON.")
                    self.gdf_mex = None
                    return

            self.geojson_state_column_name = identified_geojson_col

            for index, row_data in self.gdf_mex.iterrows():
                state_name = row_data[self.geojson_state_column_name]

                if not isinstance(state_name, str):
                    state_name = str(state_name)

                lbl = ttk.Label(self.state_data_frame, text=f"{state_name}:")
                lbl.grid(row=index, column=0, padx=5, pady=2, sticky="w")

                entry = ttk.Entry(self.state_data_frame, width=15)
                entry.grid(row=index, column=1, padx=5, pady=2, sticky="ew")

                normalized_state_name = state_name.upper()
                default_pop = self.default_population_data.get(normalized_state_name, "0")
                entry.insert(0, str(default_pop))

                if hasattr(self, 'state_entries'):
                    self.state_entries[state_name] = entry

            if hasattr(self, 'state_data_frame') and self.state_data_frame.winfo_exists()):
                 self.state_data_frame.update_idletasks()

            self.txt_out.insert(tk.END, f"Shapefile: {os.path.basename(path)} (Columna Estado GeoJSON: {self.geojson_state_column_name})\n")
            messagebox.showinfo("Éxito", f"Cargado shapefile: {os.path.basename(path)}")

        except Exception as e:
            self.log(f"Error en load_shapefile: {e}", "ERROR")
            traceback.print_exc()
            messagebox.showerror("Error al cargar GeoJSON", f"Detalles: {str(e)}")
            self.gdf_mex = None
            self.geojson_state_column_name = None
            if hasattr(self, 'state_data_frame') and self.state_data_frame.winfo_exists():
                for widget in self.state_data_frame.winfo_children():
                    widget.destroy()
            if hasattr(self, 'state_entries'):
                self.state_entries.clear()

    def restore_default_population(self):
        if not self.state_entries:
            self.log("No hay campos de estado para restaurar. Cargue primero un GeoJSON.", "WARNING")
            messagebox.showwarning("Advertencia", "No hay campos de estado para restaurar. Cargue primero un archivo GeoJSON.")
            return

        num_restored = 0
        for state_name, entry_widget in self.state_entries.items():
            normalized_state_name_lookup = state_name.upper()
            default_pop = self.default_population_data.get(normalized_state_name_lookup, "0")

            entry_widget.delete(0, tk.END)
            entry_widget.insert(0, str(default_pop))
            num_restored += 1

        self.log(f"{num_restored} campos de población restaurados a valores por defecto.", "INFO")
        if num_restored > 0:
            messagebox.showinfo("Información", f"{num_restored} campos de población han sido restaurados a sus valores por defecto.")

    def _load_data(self):
        filepath = filedialog.askopenfilename(title="Seleccionar archivo de datos", filetypes=(("Archivos CSV", "*.csv"), ("Archivos Excel", "*.xls *.xlsx"), ("Todos los archivos", "*.*")))
        if not filepath: return
        ext = os.path.splitext(filepath)[1].lower()
        try:
            if ext == ".csv": self.df_original = pd.read_csv(filepath, sep=",")
            elif ext in [".xls", ".xlsx"]: self.df_original = pd.read_excel(filepath)
            else: messagebox.showerror("Error de Archivo", "Tipo de archivo no soportado."); return
            self.filepath_var.set(filepath)
            self._update_column_selectors()
            messagebox.showinfo("Archivo Cargado", f"Archivo '{os.path.basename(filepath)}' cargado.")
            self.log(f"Datos cargados: {self.df_original.shape}")
        except Exception as e:
            messagebox.showerror("Error al Leer Archivo", f"No se pudo leer el archivo:\n{e}")
            self.df_original = None; self.filepath_var.set(""); self._update_column_selectors()
            self.log(f"Error cargando archivo: {e}")

    def _update_column_selectors(self):
        cols = sorted(self.df_original.columns.tolist()) if self.df_original is not None else []
        self.state_col_combo['values'] = [""] + cols
        current_value_options = [""]
        if self.df_original is not None:
            numeric_cols = self.df_original.select_dtypes(include=np.number).columns.tolist()
            current_value_options.extend(numeric_cols)
        if self.MANUAL_STATE_DATA_OPTION not in current_value_options:
            current_value_options.append(self.MANUAL_STATE_DATA_OPTION)
        self.value_col_combo['values'] = current_value_options
        filter_cols_options = [''] + cols
        if hasattr(self, 'filter_col_1_combo'): self.filter_col_1_combo['values'] = filter_cols_options
        if hasattr(self, 'filter_col_2_combo'): self.filter_col_2_combo['values'] = filter_cols_options
        if not self.state_col_var.get() and cols: self.state_col_var.set("")
        if not self.value_col_var.get() and current_value_options: self.value_col_var.set("")
        # Ensure filter selections are preserved or reset gracefully
        if self.filter_col_1_var.get() not in filter_cols_options: self.filter_col_1_var.set('')
        if self.filter_col_2_var.get() not in filter_cols_options: self.filter_col_2_var.set('')


    def _get_aggregated_data(self):
        selected_value_col = self.value_col_var.get()
        if selected_value_col == self.MANUAL_STATE_DATA_OPTION:
            if not self.state_entries:
                self.log("No hay entradas de estado manuales. Cargue un GeoJSON.", "ERROR"); messagebox.showerror("Error", "No hay datos de estado manuales. Cargue GeoJSON."); return None, None
            if not self.geojson_state_column_name:
                self.log("Columna de estado GeoJSON no identificada.", "ERROR"); messagebox.showerror("Error", "Columna de estado GeoJSON no identificada. Recargue GeoJSON."); return None, None
            manual_data_list = []
            for state_name_key, entry_widget in self.state_entries.items():
                value_str = entry_widget.get(); value = 0.0
                try: value = float(value_str)
                except ValueError: self.log(f"Valor inválido '{value_str}' para {state_name_key}, usando 0.0", "WARNING")
                manual_data_list.append({self.geojson_state_column_name: state_name_key, "Metric": value})
            if not manual_data_list: self.log("Lista de datos manuales vacía.", "WARNING"); return None, None
            df_manual = pd.DataFrame(manual_data_list)
            self.log("Datos generados desde entradas manuales.", "INFO")
            return df_manual, self.geojson_state_column_name

        if self.df_original is None: self.log("Datos originales no cargados.", "ERROR"); return None, None
        state_col_from_datafile = self.state_col_var.get()
        if not state_col_from_datafile: self.log("Columna Estado/Región no seleccionada.", "ERROR"); return None, None
        agg_method = self.agg_method_var.get()
        if agg_method in ["sum", "mean"] and not selected_value_col: self.log(f"Columna Valor no seleccionada para '{agg_method}'.", "ERROR"); return None, None
        df_initial = self.df_original.copy(); df_filtered = self._apply_general_filters(df_initial)
        if df_filtered is None or df_filtered.empty: self.log("Sin datos tras filtros."); return None, None
        if state_col_from_datafile not in df_filtered.columns: self.log(f"Columna '{state_col_from_datafile}' no en datos filtrados."); return None, None
        try:
            grouped = df_filtered.groupby(state_col_from_datafile)
            if agg_method == "count": aggregated_data = grouped.size().reset_index(name='Metric')
            elif agg_method == "sum":
                if selected_value_col not in df_filtered.columns or not pd.api.types.is_numeric_dtype(df_filtered[selected_value_col]): self.log(f"Columna valor '{selected_value_col}' no numérica/existente."); return None,None
                aggregated_data = grouped[selected_value_col].sum().reset_index(name='Metric')
            elif agg_method == "mean":
                if selected_value_col not in df_filtered.columns or not pd.api.types.is_numeric_dtype(df_filtered[selected_value_col]): self.log(f"Columna valor '{selected_value_col}' no numérica/existente."); return None,None
                aggregated_data = grouped[selected_value_col].mean().reset_index(name='Metric')
            else: self.log(f"Método agregación desconocido: {agg_method}"); return None, None
            self.log(f"Datos agregados por '{state_col_from_datafile}' usando '{agg_method}'.")
            return aggregated_data, state_col_from_datafile
        except Exception as e: self.log(f"Error en agregación: {e}"); traceback.print_exc(); return None, None

    def _apply_general_filters(self, df):
        df_to_filter = df.copy()
        def apply_filter_row(df, active_var, col_var, op_var, val_var, filter_num_str): # Shortened for brevity
            if active_var.get() and col_var.get(): col = col_var.get(); op = op_var.get(); val_str = val_var.get() # ... (rest of filter logic)
            # (Assuming full filter logic from previous step is here)
            if active_var.get() and col_var.get():
                col = col_var.get(); op = op_var.get(); val_str = val_var.get()
                if col not in df.columns: self.log(f"Columna de filtro {filter_num_str} '{col}' no encontrada.", "WARNING"); return df
                original_col_dtype = df[col].dtype; val = val_str
                try:
                    if op not in ["es NaN", "no es NaN", "contiene", "no contiene"]:
                        if pd.api.types.is_numeric_dtype(original_col_dtype):
                            if val_str == '': self.log(f"Valor vacío para filtro numérico {filter_num_str} en '{col}'. No se aplicará.", "INFO"); return df
                            val = pd.to_numeric(val_str)
                        elif pd.api.types.is_datetime64_any_dtype(original_col_dtype):
                             if val_str == '': self.log(f"Valor vacío para filtro de fecha {filter_num_str} en '{col}'. No se aplicará.", "INFO"); return df
                             val = pd.to_datetime(val_str)
                        # else val remains str(val_str) - already handled
                except ValueError: self.log(f"No se pudo convertir '{val_str}' para filtro {filter_num_str} en '{col}'. Se usará como string.", "WARNING")
                # self.log(f"Aplicando filtro {filter_num_str}: {col} {op} {val if op not in ['es NaN', 'no es NaN'] else ''}", "INFO")
                try:
                    if op == "==": df = df[df[col] == val]
                    elif op == "!=": df = df[df[col] != val]
                    elif op == ">": df = df[df[col] > val]
                    elif op == "<": df = df[df[col] < val]
                    elif op == ">=": df = df[df[col] >= val]
                    elif op == "<=": df = df[df[col] <= val]
                    elif op == "contiene": df = df[df[col].astype(str).str.contains(str(val), case=False, na=False)]
                    elif op == "no contiene": df = df[~df[col].astype(str).str.contains(str(val), case=False, na=False)]
                    elif op == "es NaN": df = df[df[col].isnull()]
                    elif op == "no es NaN": df = df[df[col].notnull()]
                except Exception as filter_exc: self.log(f"Error aplicando filtro {filter_num_str} ({col} {op} {val_str}): {filter_exc}", "ERROR")
            return df
        df_to_filter = apply_filter_row(df_to_filter, self.filter_active_1_var, self.filter_col_1_var, self.filter_op_1_var, self.filter_val_1_var, "1")
        df_to_filter = apply_filter_row(df_to_filter, self.filter_active_2_var, self.filter_col_2_var, self.filter_op_2_var, self.filter_val_2_var, "2")
        return df_to_filter

    def show_map(self):
        fig = self.make_fig()
        if not fig: return
        if self.fig_canvas: self.fig_canvas.get_tk_widget().destroy()
        self.fig_canvas = FigureCanvasTkAgg(fig,master=self.frm_map); self.fig_canvas.draw(); self.fig_canvas.get_tk_widget().pack(fill=tk.BOTH,expand=True)
        self.canvas_map.config(scrollregion=self.canvas_map.bbox("all"))

    def make_fig(self):
        df_agg, data_state_col_for_merge = self._get_aggregated_data()
        if df_agg is None or df_agg.empty: messagebox.showwarning("Aviso", "No se pudieron generar datos agregados."); return None
        if self.gdf_mex is None: messagebox.showwarning("Aviso", "Cargue GeoJSON primero."); return None
        if not self.geojson_state_column_name: messagebox.showerror("Error", "Columna estado GeoJSON no disponible."); return None
        if not data_state_col_for_merge: messagebox.showerror("Error", "Columna estado de datos no disponible."); return None
        pal,ncol,inv,scale,dpi,lw,vmin,vmax_s,nt,vals = self.cmb_palette.get(),int(self.ent_pal_n.get()) if self.ent_pal_n.get().isdigit() else 0,self.invert_cmap.get(),self.cmb_scale.get(),int(self.ent_dpi.get()),float(self.ent_lw.get()),float(self.ent_vmin.get()),self.ent_vmax.get().strip(),int(self.ent_nt.get()),self.ent_vals.get().strip()
        vmax = float(vmax_s) if vmax_s else None
        title,tcol,tsz,subt,scol,ssz,cbt,cbtcol,cbtsz,cbkcol,cbksz = self.ent_title.get().strip(),self.ent_tcol.get().strip() or "black",float(self.ent_tsz.get()),self.ent_sub.get().strip(),self.ent_scol.get().strip() or "gray",float(self.ent_ssz.get()),self.ent_cbt.get().strip(),self.ent_cbtcol.get().strip() or "black",float(self.ent_cbtsz.get()),self.ent_cbkcol.get().strip() or "black",float(self.ent_cbksz.get())
        col_to_plot = "Metric"
        if col_to_plot not in df_agg.columns: self.log(f"Columna '{col_to_plot}' no en datos agregados."); return None
        gdf = self.gdf_mex.merge(df_agg, how="left", left_on=self.geojson_state_column_name, right_on=data_state_col_for_merge)
        gdf[col_to_plot] = gdf[col_to_plot].fillna(0)
        if vmax is None: vmax = gdf[col_to_plot].max() if not gdf[col_to_plot].empty else 1
        thresh = 0.1 if vmax > 10 else 0.01
        if scale=="Logarítmica": norm = SymLogNorm(linthresh=thresh, linscale=1, vmin=vmin, vmax=vmax)
        else: norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        base = matplotlib.colormaps[pal]
        if ncol>0: colors = base(np.linspace(0,1,ncol)); cmap = ListedColormap(colors)
        else: cmap = base
        if inv: cmap = cmap.reversed()
        fig = Figure(figsize=(10,8), dpi=dpi); ax = fig.add_subplot(111)
        gdf.plot(column=col_to_plot, cmap=cmap, norm=norm, edgecolor="black", linewidth=lw, ax=ax, missing_kwds={'color': 'lightgrey', "hatch": "///", "label": "Sin datos"})
        ax.set_axis_off()
        agg_method_display, value_col_display = self.agg_method_var.get(), self.value_col_var.get()
        metric_display = "Datos Manuales" if value_col_display == self.MANUAL_STATE_DATA_OPTION else f"{agg_method_display}({value_col_display})" if value_col_display and agg_method_display != 'count' else f"{agg_method_display}"
        subtitle_display_state_col = self.geojson_state_column_name if value_col_display == self.MANUAL_STATE_DATA_OPTION else data_state_col_for_merge
        default_title = title if title else f"Mapa de México - {metric_display}"
        default_subtitle = subt if subt else f"Agregado por {subtitle_display_state_col}"
        ax.set_title(default_title, color=tcol, fontsize=tsz, pad=20)
        ax.text(0.5, 0.96, default_subtitle, transform=ax.transAxes, ha='center', color=scol, fontsize=ssz)
        sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap); sm._A=[]
        if vals:
            try: ticks = [float(x) for x in vals.split(",")]
            except: ticks = np.linspace(vmin,vmax,nt)
        else: ticks = np.linspace(vmin,vmax,nt)
        cbar = fig.colorbar(sm, ax=ax, ticks=ticks)
        for spine in cbar.ax.spines.values(): spine.set_edgecolor(cbtcol); spine.set_linewidth(1)
        cbar.ax.tick_params(color=cbkcol, labelcolor=cbkcol, width=1)
        cbar.ax.set_yticklabels([f"{t:.2f}" for t in ticks], fontsize=cbksz, color=cbkcol)
        default_cbt = cbt if cbt else metric_display
        cbar.set_label(default_cbt, fontsize=cbtsz, color=cbtcol)
        if self.show_labels.get():
            for _,r in gdf.iterrows():
                if r.geometry is not None and pd.notnull(r[col_to_plot]):
                    pt = r.geometry.representative_point();
                    if pt.is_empty or not pt.is_valid: continue
                    val_to_show = r[col_to_plot]
                    if abs(val_to_show) >= 1000: txt = f"{val_to_show:,.0f}"
                    elif abs(val_to_show) >= 10: txt = f"{val_to_show:,.1f}"
                    elif abs(val_to_show) >= 0.1: txt = f"{val_to_show:.2f}"
                    else: txt = f"{val_to_show:.2e}"
                    ax.annotate(txt, xy=(pt.x,pt.y), ha='center', fontsize=cbksz, color=cbkcol)
        fig.tight_layout(); return fig

    def save_map(self):
        path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG","*.png"),("JPG","*.jpg"),("All","*.*")])
        if not path: return
        fig = self.make_fig()
        if not fig: return
        try: fig.savefig(path); messagebox.showinfo("Éxito", f"Guardado en:\n{path}")
        except Exception as e: messagebox.showerror("Error", str(e))

    def log(self, message, level="INFO"):
        try: self.txt_out.insert(tk.END, f"[{level}] {message}\n"); self.txt_out.see(tk.END)
        except Exception as e: print(f"Error en log: {e}")

if __name__=="__main__":
    root = tk.Tk()
    root.title("Mapa Coroplético de México desde Datos")
    app = MapTab(root)
    app.pack(fill=tk.BOTH, expand=True)
    root.mainloop()
