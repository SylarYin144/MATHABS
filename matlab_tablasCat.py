#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
import os
import traceback # Añadido para logging

# FilterComponent ha sido eliminado.
FilterComponent = None # Mantener para evitar errores si alguna lógica residual lo verifica.

def create_scrollable_frame(container):
    """Crea un frame con scroll vertical para poner muchos controles."""
    canvas = tk.Canvas(container)
    scrollbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
    scrollable_frame = ttk.Frame(canvas)
    scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)
    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")
    return scrollable_frame

class TablasCat(ttk.Frame):
    def __init__(self, master, log_callback=None):
        super().__init__(master)
        
        if log_callback:
            self.log_func = log_callback
        elif hasattr(master, 'log_func') and callable(master.log_func): # Prefer 'log_func' if available
            self.log_func = master.log_func
        elif hasattr(master, 'log') and callable(master.log): # Fallback to 'log'
            self.log_func = master.log
        else:
            # Basic print-based logger if no callback is provided
            def default_logger(message, level="INFO"): # Add level for compatibility
                print(f"[{level}] TablasCat: {message}") # Add class context to default print
            self.log_func = default_logger
            
        self.data = None
        self.file_path = None

        # Checkbuttons y opciones
        self.exclude_blank = tk.BooleanVar(value=False)         # Omitir filas con espacios en blanco
        self.omit_non_numeric_qual = tk.BooleanVar(value=False)   # Omitir filas con valores numéricos en columnas de texto
        self.chk_all_categories = tk.BooleanVar(value=True)       # Incluir todas las categorías
        self.chk_global = tk.BooleanVar(value=True)               # Calcular global

        # Estadísticos numéricos (cuantitativos)
        self.quant_stats = {}

        # Guardamos la tabla final para mostrar/ordenar/exportar
        self.df_final = pd.DataFrame()

        # Variable para conservar el orden de las categorías definido en el filtro principal
        self.ordered_categories = None

        self.create_widgets()

    def create_widgets(self):
        self.main_pane = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        self.main_pane.pack(fill=tk.BOTH, expand=True)

        # Panel izquierdo con scroll
        control_frame = ttk.Frame(self.main_pane)
        self.main_pane.add(control_frame, weight=1)
        self.scrollable_controls = create_scrollable_frame(control_frame)

        # 1. Cargar Datos
        frm_load = ttk.LabelFrame(self.scrollable_controls, text="Cargar Datos")
        frm_load.pack(fill=tk.X, padx=5, pady=5)
        btn_load = ttk.Button(frm_load, text="Abrir Archivo (CSV o Excel)", command=self.load_data)
        btn_load.pack(side=tk.LEFT, padx=5, pady=5)
        self.lbl_file = ttk.Label(frm_load, text="Ningún archivo cargado.", foreground="blue", cursor="hand2")
        self.lbl_file.pack(side=tk.LEFT, padx=5, pady=5)
        self.lbl_file.bind("<Button-1>", self.open_file)

        # 2. Variable de Agrupación
        frm_grouping = ttk.LabelFrame(self.scrollable_controls, text="Variable de Agrupación")
        frm_grouping.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(frm_grouping, text="Variable para Agrupar Tablas:").grid(row=0, column=0, padx=5, pady=2, sticky="w")
        self.cmb_cat = ttk.Combobox(frm_grouping, values=[], state="readonly") # Variable para agrupar
        self.cmb_cat.grid(row=0, column=1, padx=5, pady=2, sticky="we")
        ttk.Label(frm_grouping, text="Etiquetas/Orden Grupos (Opcional):").grid(row=1, column=0, padx=5, pady=2, sticky="w")
        self.entry_filter = ttk.Entry(frm_grouping) # Entry para etiquetas/orden
        self.entry_filter.grid(row=1, column=1, padx=5, pady=2, sticky="we")
        ttk.Label(frm_grouping, text="(Ej: 1:GrupoA,3:GrupoC,2:GrupoB)").grid(row=2, column=1, sticky="w", padx=5, pady=0)
        frm_grouping.columnconfigure(1, weight=1)

        # 3. Sección de Filtros Generales eliminada ya que FilterComponent fue removido.

        # 4. Formato (Orientación) - Ajustar número
        frm_format = ttk.LabelFrame(self.scrollable_controls, text="Formato de Tabla")
        frm_format.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(frm_format, text="Orientación de la Tabla:").pack(anchor="w", padx=5, pady=2)
        self.cmb_orientation = ttk.Combobox(frm_format, values=["Categorías en filas", "Categorías en columnas"], state="readonly")
        self.cmb_orientation.pack(fill=tk.X, padx=5, pady=2)
        self.cmb_orientation.set("Categorías en filas")

        # 5. Tabla Comparativa Cuantitativa - Ajustar número
        frm_quant = ttk.LabelFrame(self.scrollable_controls, text="Variables Cuantitativas")
        frm_quant.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(frm_quant, text="Selecciona Variables Numéricas:").pack(anchor="w", padx=5, pady=2)
        self.lst_quant = tk.Listbox(frm_quant, selectmode=tk.MULTIPLE, exportselection=False, height=5)
        self.lst_quant.pack(fill=tk.X, padx=5, pady=2)
        btn_clear_quant = ttk.Button(frm_quant, text="Borrar Selección", command=lambda: self.lst_quant.selection_clear(0, tk.END))
        btn_clear_quant.pack(padx=5, pady=2)

        frm_stats = ttk.LabelFrame(frm_quant, text="Estadísticos a Incluir")
        frm_stats.pack(fill=tk.X, padx=5, pady=5)
        stats_options = ["Número total", "Porcentaje", "Mín", "Máx", "Media", "Mediana", "Desviación Estándar", "Varianza"]
        for row in [stats_options[:4], stats_options[4:]]:
            frow = ttk.Frame(frm_stats)
            frow.pack(fill=tk.X, padx=5, pady=2)
            for opt in row:
                var_cb = tk.BooleanVar(value=True)
                chk = ttk.Checkbutton(frow, text=opt, variable=var_cb)
                chk.pack(side=tk.LEFT, padx=5)
                self.quant_stats[opt] = var_cb

        # 6. Tabla Comparativa Cualitativa - Ajustar número
        frm_qual = ttk.LabelFrame(self.scrollable_controls, text="Variables Cualitativas")
        frm_qual.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(frm_qual, text="Mostrar TODAS las variables (selecciona las que deseas tratar como cualitativas)").pack(anchor="w", padx=5, pady=2)
        self.lst_qual = tk.Listbox(frm_qual, selectmode=tk.MULTIPLE, exportselection=False, height=5)
        self.lst_qual.pack(fill=tk.X, padx=5, pady=2)
        btn_clear_qual = ttk.Button(frm_qual, text="Borrar Selección", command=lambda: self.lst_qual.selection_clear(0, tk.END))
        btn_clear_qual.pack(padx=5, pady=2)

        ttk.Label(frm_qual, text="Mostrar:").pack(anchor="w", padx=5, pady=2)
        self.cmb_qual_display = ttk.Combobox(frm_qual, values=["Total", "Porcentaje", "Ambos"], state="readonly")
        self.cmb_qual_display.pack(fill=tk.X, padx=5, pady=2)
        self.cmb_qual_display.set("Ambos")

        f_opts = ttk.Frame(frm_qual)
        f_opts.pack(fill=tk.X, padx=5, pady=2)
        # Estas dos checkbuttons se aplicarán como filtro global tras los filtros adicionales:
        self.chk_omit_blank = ttk.Checkbutton(f_opts, text="Omitir espacios en blanco", variable=self.exclude_blank)
        self.chk_omit_blank.pack(side=tk.LEFT, padx=5)

        self.chk_omit_nonnum = ttk.Checkbutton(f_opts, text="Omitir valores no numéricos", variable=self.omit_non_numeric_qual)
        self.chk_omit_nonnum.pack(side=tk.LEFT, padx=5)

        chk_all = ttk.Checkbutton(f_opts, text="Incluir todas las categorías", variable=self.chk_all_categories)
        chk_all.pack(side=tk.LEFT, padx=5)
        chk_global = ttk.Checkbutton(f_opts, text="Calcular Global", variable=self.chk_global)
        chk_global.pack(side=tk.LEFT, padx=5)

        # 7. Opciones de Tabla Compuesta - Ajustar número
        frm_composite = ttk.LabelFrame(self.scrollable_controls, text="Opciones Tabla Compuesta")
        frm_composite.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(frm_composite, text="Tipo de Tabla Compuesta:").grid(row=0, column=0, padx=5, pady=2, sticky="w")
        self.cmb_composite_type = ttk.Combobox(frm_composite, values=["Estadísticos Descriptivos", "Presencia/Ausencia", "Combinada"], state="readonly")
        self.cmb_composite_type.grid(row=0, column=1, padx=5, pady=2, sticky="we")
        self.cmb_composite_type.set("Combinada")

        ttk.Label(frm_composite, text="Valor(es) y Etiquetas para Cualitativa:").grid(row=1, column=0, padx=5, pady=2, sticky="w")
        self.entry_positive_value = ttk.Entry(frm_composite)
        self.entry_positive_value.grid(row=1, column=1, padx=5, pady=2, sticky="we")
        frm_composite.columnconfigure(1, weight=1)

        # 8. Botones de Control - Ajustar número
        frm_buttons = ttk.Frame(self.scrollable_controls)
        frm_buttons.pack(fill=tk.X, padx=5, pady=10)
        btn_resumen = ttk.Button(frm_buttons, text="Generar Resumen Descriptivo", command=self.generate_summary)
        btn_resumen.pack(side=tk.LEFT, padx=5)
        btn_composite = ttk.Button(frm_buttons, text="Generar Tabla Compuesta (Vista Previa)", command=self.generate_composite_table)
        btn_composite.pack(side=tk.LEFT, padx=5)

        # Panel de Salida (derecho)
        self.output_frame = ttk.Frame(self.main_pane)
        self.main_pane.add(self.output_frame, weight=2)
        self.txt_output = tk.Text(self.output_frame, wrap="none")
        vsb = ttk.Scrollbar(self.output_frame, orient="vertical", command=self.txt_output.yview)
        hsb = ttk.Scrollbar(self.output_frame, orient="horizontal", command=self.txt_output.xview)
        self.txt_output.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        self.txt_output.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")
        self.output_frame.rowconfigure(0, weight=1)
        self.output_frame.columnconfigure(0, weight=1)

    # --------------------------------------------------------------------------------
    # Carga de datos
    # --------------------------------------------------------------------------------
    def load_data(self):
        file_path = filedialog.askopenfilename(
            title="Selecciona archivo CSV o Excel",
            filetypes=[("CSV files","*.csv"), ("Excel files","*.xlsx *.xls"), ("All files","*.*")]
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

            # Actualizar combos y listboxes
            self.cmb_cat['values'] = [""] + cols # Actualizar combo de agrupación
            self.cmb_cat.set("")
            
            self.lst_quant.delete(0, tk.END)
            for c in cols:
                if pd.api.types.is_numeric_dtype(self.data[c]):
                    self.lst_quant.insert(tk.END, c)

            self.lst_qual.delete(0, tk.END)
            for c in cols:
                self.lst_qual.insert(tk.END, c)

            msg = f"Datos cargados: {self.data.shape[0]} filas, {self.data.shape[1]} columnas."
            messagebox.showinfo("Éxito", msg)
            self.txt_output.delete("1.0", tk.END)
            self.txt_output.insert(tk.END, msg)
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo cargar el archivo:\n{e}")
            if hasattr(self, 'filter_col_1_combo'): self.filter_col_1_combo['values'] = ['']
            if hasattr(self, 'filter_col_2_combo'): self.filter_col_2_combo['values'] = ['']

    def open_file(self, event):
        if self.file_path and os.path.exists(self.file_path):
            try:
                os.startfile(self.file_path)
            except Exception as e:
                messagebox.showerror("Error", f"No se pudo abrir el archivo:\n{e}")

    # --------------------------------------------------------------------------------
    # Aplicar filtros y categorización principal
    # --------------------------------------------------------------------------------
    def _apply_filters_and_categorization(self):
        """
        Aplica el filtro/etiquetado de la variable de agrupación.
        Devuelve el DataFrame resultante y las categorías ordenadas (si aplica).
        """
        if self.data is None:
            self.log_func("Advertencia: No hay datos originales cargados en _apply_filters_and_categorization.", "WARN")
            messagebox.showwarning("Datos no cargados", "Por favor, cargue un archivo de datos primero.")
            return None, None

        df_initial = self.data.copy()
        self.log_func("Info: Usando datos originales para categorización en _apply_filters_and_categorization.", "INFO")
        
        df = df_initial.copy() # Start with the initial data (copy for modification)

        if df.empty: 
            self.log_func("Advertencia: DataFrame vacío después de la copia inicial.", "WARN")
            messagebox.showwarning("Datos Vacíos", "El DataFrame está vacío inicialmente.")
            return df, None 
        
        # Aplicar filtro/etiquetado/orden de la variable de agrupación principal
        cat_var = self.cmb_cat.get().strip()
        filtro_etiquetas = self.entry_filter.get().strip()
        ordered_categories = None # Reiniciar orden

        if cat_var and cat_var in df.columns and filtro_etiquetas:
            col_is_num = pd.api.types.is_numeric_dtype(df[cat_var])
            
            if ":" in filtro_etiquetas:
                mapping = {}
                parts = [p.strip() for p in filtro_etiquetas.split(",") if p.strip()]
                typed_order_labels = []
                keys_to_keep = []
                for part in parts:
                    if ":" in part:
                        key, label = part.split(":", 1)
                        key = key.strip(); label = label.strip()
                        mapping[key] = label; typed_order_labels.append(label)
                        keys_to_keep.append(key)
                
                if col_is_num:
                    numeric_keys = []
                    numeric_mapping = {}
                    for k_str in keys_to_keep:
                        try: 
                            nk = float(k_str)
                            numeric_keys.append(nk)
                            numeric_mapping[nk] = mapping[k_str]
                        except ValueError: # Si la clave no es numérica, tratarla como string
                            numeric_keys.append(k_str) 
                            numeric_mapping[k_str] = mapping[k_str]

                    # Asegurar que la columna original es del tipo correcto antes de isin
                    if pd.api.types.is_numeric_dtype(df[cat_var].dtype):
                        df = df[df[cat_var].isin(numeric_keys)]
                        df[cat_var] = df[cat_var].map(numeric_mapping)
                    else: # Si la columna no es numérica pero se dieron claves numéricas
                        df = df[df[cat_var].astype(str).isin(map(str, numeric_keys))]
                        # Mapear usando strings para las claves originales
                        df[cat_var] = df[cat_var].astype(str).map({str(k):v for k,v in numeric_mapping.items()})
                else: # Columna no numérica
                    df = df[df[cat_var].astype(str).isin(keys_to_keep)]
                    df[cat_var] = df[cat_var].astype(str).map(mapping)
                
                if typed_order_labels:
                    cat_dtype = pd.CategoricalDtype(categories=typed_order_labels, ordered=True)
                    df[cat_var] = df[cat_var].astype(cat_dtype)
                    ordered_categories = typed_order_labels

            elif "-" in filtro_etiquetas and "," not in filtro_etiquetas: # Rango
                parts = filtro_etiquetas.split("-")
                if len(parts) == 2 and col_is_num:
                    try:
                        low = float(parts[0].strip()); high = float(parts[1].strip())
                        df = df[(df[cat_var] >= low) & (df[cat_var] <= high)]
                    except: pass
            elif "," in filtro_etiquetas: # Lista
                vals = [p.strip() for p in filtro_etiquetas.split(",") if p.strip()]
                if col_is_num:
                    numeric_vals = []
                    for val_str in vals:
                        try: numeric_vals.append(float(val_str))
                        except: numeric_vals.append(val_str) # Mantener como string si no es convertible
                    df = df[df[cat_var].isin(numeric_vals)]
                    # El orden se basa en la lista de valores si son numéricos
                    ordered_categories = [v for v in numeric_vals if pd.api.types.is_number(v)]
                else:
                    df = df[df[cat_var].astype(str).isin(vals)]
                    ordered_categories = vals # El orden es el de la lista de strings
                
                if ordered_categories:
                    cat_dtype = pd.CategoricalDtype(categories=ordered_categories, ordered=True)
                    df[cat_var] = df[cat_var].astype(cat_dtype)

            else: # Valor único
                if col_is_num:
                    try: val_f = float(filtro_etiquetas); df = df[df[cat_var] == val_f]
                    except: df = df[df[cat_var].astype(str) == filtro_etiquetas]
                else:
                    df = df[df[cat_var].astype(str) == filtro_etiquetas]
                ordered_categories = [filtro_etiquetas]

        self.ordered_categories = ordered_categories # Guardar orden para usarlo después
        return df, ordered_categories

    def apply_global_filters(self, df):
        """Aplica los filtros globales: excluir espacios en blanco y filas con valores numéricos en columnas de texto."""
        if self.exclude_blank.get():
            df = df.replace(r'^\s*$', np.nan, regex=True)
            df = df.dropna(how='any')
        if self.omit_non_numeric_qual.get():
            obj_cols = df.select_dtypes(include=['object', 'string']).columns
            for c in obj_cols:
                df = df[~df[c].astype(str).apply(lambda x: x.replace('.', '', 1).isdigit())]
        return df

    # --------------------------------------------------------------------------------
    # Resumen descriptivo general (texto)
    # --------------------------------------------------------------------------------
    def summarize_data(self):
        df, _ = self._apply_filters_and_categorization() # Usar la nueva función
        if df is None:
            return "Error al aplicar filtros o no hay datos."
        
        df = self.apply_global_filters(df) # Aplicar filtros globales después de la categorización
        if df.empty:
            return "No hay datos para resumir tras aplicar filtros."
        lines = []
        lines.append("Resumen del archivo (filtrado):")
        lines.append(f"  Número de variables (columnas): {df.shape[1]}")
        lines.append(f"  Número de registros (filas): {df.shape[0]}")
        lines.append("")
        for col in df.columns:
            series = df[col]
            lines.append(f"Variable: {col}")
            lines.append(f"  Tipo: {series.dtype}")
            valid = series.count()
            lines.append(f"  Valores válidos: {valid}")
            if pd.api.types.is_numeric_dtype(series):
                uniq = series.dropna().unique()
                if len(uniq) < 10 and np.all(np.mod(uniq, 1) == 0):
                    freq = series.value_counts(dropna=True)
                    freq_str = "; ".join([f"{c}: {cnt} ({cnt/valid*100:.1f}%)" for c, cnt in freq.items()])
                    lines.append("  (Interpretada como cualitativa por pocos valores únicos enteros)")
                    lines.append(f"  Frecuencia: {freq_str}")
                    lines.append(f"  Promedio: {series.mean():.2f}")
                    lines.append(f"  Mediana: {series.median():.2f}")
                else:
                    try:
                        lines.append(f"  Promedio: {series.mean():.2f}")
                        lines.append(f"  Mediana: {series.median():.2f}")
                        lines.append(f"  Desviación Est.: {series.std():.2f}")
                    except:
                        lines.append("  Error al calcular estadísticas numéricas.")
            else:
                freq = series.value_counts(dropna=True)
                lines.append("  Frecuencia (todas las categorías):")
                for lab, cnt in freq.items():
                    perc = cnt/valid*100 if valid > 0 else 0
                    lines.append(f"    {lab}: {cnt} ({perc:.1f}%)")
            lines.append("")
        return "\n".join(lines)

    def generate_summary(self):
        txt = self.summarize_data()
        self.txt_output.delete("1.0", tk.END)
        self.txt_output.insert(tk.END, txt)

    # Se elimina apply_filter_criteria ya que su lógica está ahora en FilterComponent

    # --------------------------------------------------------------------------------
    # Filtro + etiquetado para variable cualitativa
    # (Se usa en caso de "Valor(es) y Etiquetas para Cualitativa")
    # --------------------------------------------------------------------------------
    def parse_value_label_map(self, df, col, raw_filter):
        col_is_num = pd.api.types.is_numeric_dtype(df[col])
        if ":" in raw_filter:
            mapping = {}
            parts = [p.strip() for p in raw_filter.split(",") if p.strip()]
            for part in parts:
                if ":" in part:
                    key, label = part.split(":", 1)
                    key = key.strip()
                    label = label.strip()
                    mapping[key] = label
            if col_is_num:
                numeric_mapping = {}
                numeric_keys = []
                for k, v in mapping.items():
                    try:
                        nk = float(k)
                        numeric_keys.append(nk)
                        numeric_mapping[nk] = v
                    except:
                        numeric_keys.append(k)
                        numeric_mapping[k] = v
                df = df[df[col].isin(numeric_keys)]
                df[col] = df[col].map(numeric_mapping)
            else:
                df = df[df[col].isin(mapping.keys())]
                df[col] = df[col].map(mapping)
        elif "-" in raw_filter:
            parts = raw_filter.split("-")
            if len(parts) == 2:
                try:
                    low = float(parts[0].strip())
                    high = float(parts[1].strip())
                    df = df[(df[col] >= low) & (df[col] <= high)]
                except:
                    pass
        elif "," in raw_filter:
            vals = [p.strip() for p in raw_filter.split(",") if p.strip()]
            if col_is_num:
                numeric_vals = []
                for val in vals:
                    try:
                        numeric_vals.append(float(val))
                    except:
                        numeric_vals.append(val)
                df = df[df[col].isin(numeric_vals)]
            else:
                df = df[df[col].isin(vals)]
        else:
            if col_is_num:
                try:
                    val = float(raw_filter)
                    df = df[df[col] == val]
                except:
                    df = df[df[col] == raw_filter]
            else:
                df = df[df[col] == raw_filter]
        return df

    # --------------------------------------------------------------------------------
    # Generar Tabla Compuesta (Vista Previa)
    # --------------------------------------------------------------------------------
    def generate_composite_table(self):
        """Genera la tabla compuesta, la muestra en una ventana de vista previa y permite exportar a Excel."""
        if self.data is None:
            messagebox.showwarning("Aviso", "Primero carga los datos.")
            return

        # 1) Filtro principal y categorización
        df_filtered, ordered_categories_from_filter = self._apply_filters_and_categorization()
        # The variable 'ordered_categories_from_filter' will be used later if needed,
        # replacing the potentially unreliable self.ordered_categories for this specific run.
        
        if df_filtered is None: # self._apply_filters_and_categorization can return None
            messagebox.showwarning("Aviso", "No hay datos después de aplicar los filtros y la categorización inicial.")
            return
        
        if df_filtered.empty: # Adicionalmente chequear si está vacío post categorización
            messagebox.showwarning("Aviso", "No hay datos después de aplicar los filtros y la categorización inicial.")
            return

        # 2) Filtros adicionales (ELIMINADO)
        
        # 3) Aplicar los filtros globales (excluir blancos y no numéricos)
        df_filtered = self.apply_global_filters(df_filtered)
        if df_filtered.empty:
            messagebox.showwarning("Aviso", "No quedan datos tras aplicar los filtros globales.")
            return

        # 4) Selección de variables
        quant_indices = self.lst_quant.curselection()
        quant_vars = [self.lst_quant.get(i) for i in quant_indices]
        qual_indices = self.lst_qual.curselection()
        qual_vars = [self.lst_qual.get(i) for i in qual_indices]

        comp_type = self.cmb_composite_type.get()
        orientation = self.cmb_orientation.get()
        display_mode = self.cmb_qual_display.get()

        # 5) Determinar categorías (según dataset filtrado)
        cat_var = self.cmb_cat.get().strip()
        cat_values = ["(Sin categoría)"] # Default

        if cat_var and cat_var in df_filtered.columns:
            # Usar las categorías ordenadas obtenidas de _apply_filters_and_categorization
            if ordered_categories_from_filter: # Use the variable from the function call
                 # Asegurarse que las categorías ordenadas realmente existen en los datos finales
                 cat_values = [c for c in ordered_categories_from_filter if c in df_filtered[cat_var].unique()]
            elif self.chk_all_categories.get(): # Fallback to existing logic if no specific order came from the function
                 # Si se piden todas y no hay orden, tomar del DF original (antes de filtros de valor)
                 self.log_func("Usando todas las categorías del archivo original (orden alfabético).", "INFO")
                 cat_values = sorted(list(self.data[cat_var].dropna().astype(str).unique()))
            else:
                 # Tomar del DF filtrado y ordenar alfabéticamente
                 cat_values = sorted(list(df_filtered[cat_var].dropna().astype(str).unique()))
            
            if not cat_values: # Si la lista de categorías resultante está vacía
                cat_values = ["(Sin categoría / Datos filtrados)"]
        
        cat_values = list(dict.fromkeys(cat_values)) # Asegurar unicidad manteniendo orden

        big_list = []

        # 6) Recorremos cada categoría para generar filas
        for catv in cat_values:
            if cat_var and cat_var in df_filtered.columns and catv != "(Sin categoría)" and catv != "(Sin categoría / Datos filtrados)":
                # Comparar directamente (ya debería ser string o categorical)
                sub = df_filtered[df_filtered[cat_var] == catv]
            else:
                sub = df_filtered

            # CUANTITATIVAS
            for var in quant_vars:
                if var not in sub.columns:
                    continue
                s = sub[var].dropna()
                total_ = len(s)
                stats_vals = {}
                if display_mode in ["Total", "Ambos"] and self.quant_stats["Número total"].get():
                    stats_vals["Número total"] = total_
                if display_mode in ["Porcentaje", "Ambos"] and self.quant_stats["Porcentaje"].get():
                    stats_vals["Porcentaje"] = "100%" if total_ > 0 else "0%"
                if self.quant_stats["Mín"].get():
                    stats_vals["Mín"] = s.min() if total_ > 0 else ""
                if self.quant_stats["Máx"].get():
                    stats_vals["Máx"] = s.max() if total_ > 0 else ""
                if self.quant_stats["Media"].get():
                    stats_vals["Media"] = f"{s.mean():.2f}" if total_ > 0 else ""
                if self.quant_stats["Mediana"].get():
                    stats_vals["Mediana"] = f"{s.median():.2f}" if total_ > 0 else ""
                if self.quant_stats["Desviación Estándar"].get():
                    stats_vals["Desviación Estándar"] = f"{s.std():.2f}" if total_ > 0 else ""
                if self.quant_stats["Varianza"].get():
                    stats_vals["Varianza"] = f"{s.var():.2f}" if total_ > 0 else ""

                for st_name, st_val in stats_vals.items():
                    big_list.append({
                        "Categoria": catv,
                        "Variable": var,
                        "SubEstadistico": st_name,
                        "Valor": st_val
                    })

            # CUALITATIVAS
            if comp_type == "Presencia/Ausencia":
                pos_val = self.entry_positive_value.get().strip()
                if pos_val == "":
                    continue
                for var in qual_vars:
                    if var not in sub.columns:
                        continue
                    series = sub[var].dropna().astype(str)
                    if len(series) == 0:
                        continue
                    pos_count = (series == pos_val).sum()
                    absent_count = len(series) - pos_count
                    big_list.append({
                        "Categoria": catv,
                        "Variable": var,
                        "SubEstadistico": "Presencia",
                        "Valor": f"{pos_count} ({(pos_count/len(series)*100):.1f}%)"
                    })
                    big_list.append({
                        "Categoria": catv,
                        "Variable": var,
                        "SubEstadistico": "Ausencia",
                        "Valor": f"{absent_count} ({(absent_count/len(series)*100):.1f}%)"
                    })
            else:
                for var in qual_vars:
                    if var not in sub.columns:
                        continue
                    local_df = sub.copy()
                    qual_filter = self.entry_positive_value.get().strip()
                    if qual_filter:
                        local_df = self.parse_value_label_map(local_df, var, qual_filter)
                    s = local_df[var].dropna().astype(str)
                    if len(s) == 0:
                        continue
                    freq_filtered = s.value_counts()
                    if self.chk_all_categories.get():
                        col_full = self.data[var].dropna().astype(str)
                        total_col = len(col_full)
                        freq_dict = {}
                        for cat_ in freq_filtered.index:
                            count_ = freq_filtered[cat_]
                            pct_ = (count_ / total_col * 100) if total_col > 0 else 0
                            if self.cmb_qual_display.get() == "Total":
                                freq_dict[cat_] = f"{count_}"
                            elif self.cmb_qual_display.get() == "Porcentaje":
                                freq_dict[cat_] = f"{pct_:.1f}%"
                            else:
                                freq_dict[cat_] = f"{count_} ({pct_:.1f}%)"
                        desc = "; ".join([f"{k}: {v}" for k, v in freq_dict.items()])
                    else:
                        t_ = len(s)
                        freq_dict = {}
                        for cat_ in freq_filtered.index:
                            count_ = freq_filtered[cat_]
                            pct_ = (count_ / t_ * 100) if t_ > 0 else 0
                            if self.cmb_qual_display.get() == "Total":
                                freq_dict[cat_] = f"{count_}"
                            elif self.cmb_qual_display.get() == "Porcentaje":
                                freq_dict[cat_] = f"{pct_:.1f}%"
                            else:
                                freq_dict[cat_] = f"{count_} ({pct_:.1f}%)"
                        desc = "; ".join([f"{k}: {v}" for k, v in freq_dict.items()])

                    big_list.append({
                        "Categoria": catv,
                        "Variable": var,
                        "SubEstadistico": "Descriptivos",
                        "Valor": desc
                    })

        # 7) Calcular Global (opcional)
        if self.chk_global.get():
            for var in set(quant_vars + qual_vars):
                if var not in self.data.columns:
                    continue
                sub = df_filtered.copy()
                s = sub[var].dropna()
                if var in quant_vars:
                    total_ = len(s)
                    stats_vals = {}
                    if display_mode in ["Total", "Ambos"] and self.quant_stats["Número total"].get():
                        stats_vals["Número total"] = total_
                    if display_mode in ["Porcentaje", "Ambos"] and self.quant_stats["Porcentaje"].get():
                        stats_vals["Porcentaje"] = "100%" if total_ > 0 else "0%"
                    if self.quant_stats["Mín"].get():
                        stats_vals["Mín"] = s.min() if total_ > 0 else ""
                    if self.quant_stats["Máx"].get():
                        stats_vals["Máx"] = s.max() if total_ > 0 else ""
                    if self.quant_stats["Media"].get():
                        stats_vals["Media"] = f"{s.mean():.2f}" if total_ > 0 else ""
                    if self.quant_stats["Mediana"].get():
                        stats_vals["Mediana"] = f"{s.median():.2f}" if total_ > 0 else ""
                    if self.quant_stats["Desviación Estándar"].get():
                        stats_vals["Desviación Estándar"] = f"{s.std():.2f}" if total_ > 0 else ""
                    if self.quant_stats["Varianza"].get():
                        stats_vals["Varianza"] = f"{s.var():.2f}" if total_ > 0 else ""
                    for st_name, st_val in stats_vals.items():
                        big_list.append({
                            "Categoria": "Global",
                            "Variable": var,
                            "SubEstadistico": st_name,
                            "Valor": st_val
                        })
                else:
                    qual_filter = self.entry_positive_value.get().strip()
                    if qual_filter:
                        sub = self.parse_value_label_map(sub, var, qual_filter)
                    s = sub[var].dropna().astype(str)
                    if len(s) > 0:
                        freq_filtered = s.value_counts()
                        if self.chk_all_categories.get():
                            col_full = self.data[var].dropna().astype(str)
                            total_col = len(col_full)
                            freq_dict = {}
                            for cat_ in freq_filtered.index:
                                count_ = freq_filtered[cat_]
                                pct_ = (count_ / total_col * 100) if total_col > 0 else 0
                                if self.cmb_qual_display.get() == "Total":
                                    freq_dict[cat_] = f"{count_}"
                                elif self.cmb_qual_display.get() == "Porcentaje":
                                    freq_dict[cat_] = f"{pct_:.1f}%"
                                else:
                                    freq_dict[cat_] = f"{count_} ({pct_:.1f}%)"
                            desc = "; ".join([f"{k}: {v}" for k, v in freq_dict.items()])
                        else:
                            t_ = len(s)
                            freq_dict = {}
                            for cat_ in freq_filtered.index:
                                count_ = freq_filtered[cat_]
                                pct_ = (count_ / t_ * 100) if t_ > 0 else 0
                                if self.cmb_qual_display.get() == "Total":
                                    freq_dict[cat_] = f"{count_}"
                                elif self.cmb_qual_display.get() == "Porcentaje":
                                    freq_dict[cat_] = f"{pct_:.1f}%"
                                else:
                                    freq_dict[cat_] = f"{count_} ({pct_:.1f}%)"
                            desc = "; ".join([f"{k}: {v}" for k, v in freq_dict.items()])

                        big_list.append({
                            "Categoria": "Global",
                            "Variable": var,
                            "SubEstadistico": "Descriptivos",
                            "Valor": desc
                        })

        df_temp = pd.DataFrame(big_list)
        if df_temp.empty:
            messagebox.showwarning("Aviso", "No se generaron filas en la tabla compuesta.")
            return

        if orientation == "Categorías en filas":
            self.df_final = pd.pivot_table(
                df_temp,
                index=["Categoria", "SubEstadistico"],
                columns="Variable",
                values="Valor",
                aggfunc=lambda x: x,
                sort=False
            )
        else:
            self.df_final = pd.pivot_table(
                df_temp,
                index=["Variable", "SubEstadistico"],
                columns="Categoria",
                values="Valor",
                aggfunc=lambda x: x,
                sort=False
            )

        self.df_final = self.df_final.fillna("")

        preview_win = tk.Toplevel(self)
        preview_win.title("Vista Previa de la Tabla Compuesta")

        txt_preview = tk.Text(preview_win, wrap="none", width=120, height=30)
        vsb = ttk.Scrollbar(preview_win, orient="vertical", command=txt_preview.yview)
        hsb = ttk.Scrollbar(preview_win, orient="horizontal", command=txt_preview.xview)
        txt_preview.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

        txt_preview.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")

        preview_win.rowconfigure(0, weight=1)
        preview_win.columnconfigure(0, weight=1)

        table_str = self.df_final.to_string()
        txt_preview.insert("1.0", table_str)

        def export_to_excel():
            dest = filedialog.asksaveasfilename(
                title="Guardar Tabla en Excel",
                defaultextension=".xlsx",
                filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")]
            )
            if not dest:
                return
            try:
                with pd.ExcelWriter(dest, engine="openpyxl") as writer:
                    self.df_final.to_excel(writer, index=True, sheet_name="TablaCompuesta")
                messagebox.showinfo("Éxito", f"Tabla guardada como: {dest}")
            except Exception as e:
                messagebox.showerror("Error", f"No se pudo guardar la tabla en Excel:\n{e}")

        btn_export = ttk.Button(preview_win, text="Exportar a Excel", command=export_to_excel)
        btn_export.grid(row=2, column=0, pady=5, sticky="w")

    def save_graph_directly(self):
        # """Método opcional (no se usa aquí).""" # Docstring can be kept or removed.
        messagebox.showinfo("Función No Implementada", 
                            "La funcionalidad para 'Guardar Gráfica Directamente' no está implementada todavía en esta pestaña.")

# Tab para scripts externos (no modificado)
class ExternalScriptsTab(ttk.Frame):
    def __init__(self, master): # Removed log_callback as it's not used here
        super().__init__(master)
        lbl = ttk.Label(self, text="Aquí se cargarían scripts externos.")
        lbl.pack(pady=20)

__all__ = ["TablasCat", "ExternalScriptsTab"]

# Se eliminan get_filtered_data y apply_filter_criteria

if __name__ == "__main__":
    root = tk.Tk()
    root.title("Aplicación de Ejemplo")
    notebook = ttk.Notebook(root)
    notebook.pack(fill=tk.BOTH, expand=True)

    # Example of passing a simple print-based logger to TablasCat
    def main_app_logger(message, level="INFO"):
        print(f"[{level}] MainApp->TablasCat: {message}")

    app = TablasCat(notebook, log_callback=main_app_logger) # Pass the logger
    notebook.add(app, text="Filtros/Estadísticas")

    ext_tab = ExternalScriptsTab(notebook)
    notebook.add(ext_tab, text="Scripts Externos")

    root.mainloop()
