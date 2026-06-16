# -*- coding: utf-8 -*-
"""
MATLAB_data_editor.py
Editor de Datos Avanzado para Mathabs

Funcionalidades:
- Ver y editar tabla completa
- Agregar/eliminar filas y columnas
- Ordenar y filtrar datos
- Crear columnas calculadas
- Rellenar valores faltantes
- Cambiar tipos de datos
- Deshacer/Rehacer
- Exportar datos
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog, simpledialog
import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any
import copy


class DataEditorTab:
    """
    Pestaña de Editor de Datos Avanzado.
    Permite visualizar, editar y transformar el DataFrame de trabajo.
    """
    
    def __init__(self, parent_frame, app_instance):
        self.parent = parent_frame
        self.app = app_instance
        self.df: Optional[pd.DataFrame] = None
        self.df_original: Optional[pd.DataFrame] = None  # Copia original para comparar
        
        # Sistema de Undo/Redo
        self.undo_stack: List[pd.DataFrame] = []
        self.redo_stack: List[pd.DataFrame] = []
        self.max_undo = 50
        
        # Estado de ordenamiento
        self.sort_column = None
        self.sort_ascending = True
        
        # Variables de búsqueda/filtro
        self.search_var = tk.StringVar()
        self.filter_column_var = tk.StringVar()
        self.filter_value_var = tk.StringVar()
        self.show_filtered_only = tk.BooleanVar(value=False)
        
        # Celda seleccionada
        self.selected_item = None
        self.selected_column = None
        
        self._build_ui()
    
    def _build_ui(self):
        """Construye la interfaz del editor de datos."""
        # Frame principal con PanedWindow
        self.main_paned = ttk.PanedWindow(self.parent, orient=tk.HORIZONTAL)
        self.main_paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Panel izquierdo - Herramientas
        self.tools_frame = ttk.Frame(self.main_paned, width=280)
        self.main_paned.add(self.tools_frame, weight=0)
        
        # Panel derecho - Tabla
        self.table_frame = ttk.Frame(self.main_paned)
        self.main_paned.add(self.table_frame, weight=1)
        
        self._build_tools_panel()
        self._build_table_panel()
    
    def _build_tools_panel(self):
        """Construye el panel de herramientas."""
        # Scrollable frame para herramientas
        canvas = tk.Canvas(self.tools_frame, width=260)
        scrollbar = ttk.Scrollbar(self.tools_frame, orient="vertical", command=canvas.yview)
        self.tools_inner = ttk.Frame(canvas)
        
        self.tools_inner.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=self.tools_inner, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # === Sección: Info del Dataset ===
        frm_info = ttk.LabelFrame(self.tools_inner, text="📊 Información", padding=5)
        frm_info.pack(fill=tk.X, padx=5, pady=5)
        
        self.lbl_info = ttk.Label(frm_info, text="Sin datos cargados", wraplength=240)
        self.lbl_info.pack(fill=tk.X)
        
        ttk.Button(frm_info, text="🔄 Actualizar desde Archivo de Trabajo", 
                   command=self._load_from_workfile).pack(fill=tk.X, pady=2)
        
        # === Sección: Deshacer/Rehacer ===
        frm_undo = ttk.LabelFrame(self.tools_inner, text="↩️ Deshacer / Rehacer", padding=5)
        frm_undo.pack(fill=tk.X, padx=5, pady=5)
        
        undo_row = ttk.Frame(frm_undo)
        undo_row.pack(fill=tk.X)
        
        self.btn_undo = ttk.Button(undo_row, text="↩ Deshacer", command=self._undo, state="disabled")
        self.btn_undo.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)
        
        self.btn_redo = ttk.Button(undo_row, text="↪ Rehacer", command=self._redo, state="disabled")
        self.btn_redo.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)
        
        self.lbl_undo_count = ttk.Label(frm_undo, text="Historial: 0 | 0")
        self.lbl_undo_count.pack()
        
        ttk.Button(frm_undo, text="🔙 Restaurar Original", 
                   command=self._restore_original).pack(fill=tk.X, pady=2)
        
        # === Sección: Búsqueda ===
        frm_search = ttk.LabelFrame(self.tools_inner, text="🔍 Buscar", padding=5)
        frm_search.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Entry(frm_search, textvariable=self.search_var).pack(fill=tk.X, pady=2)
        
        search_btns = ttk.Frame(frm_search)
        search_btns.pack(fill=tk.X)
        ttk.Button(search_btns, text="Buscar", command=self._search_next).pack(side=tk.LEFT, expand=True, fill=tk.X)
        ttk.Button(search_btns, text="Resaltar Todo", command=self._highlight_all).pack(side=tk.LEFT, expand=True, fill=tk.X)
        
        # === Sección: Filtrar ===
        frm_filter = ttk.LabelFrame(self.tools_inner, text="🔎 Filtrar Vista", padding=5)
        frm_filter.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(frm_filter, text="Columna:").pack(anchor="w")
        self.cmb_filter_col = ttk.Combobox(frm_filter, textvariable=self.filter_column_var, state="readonly")
        self.cmb_filter_col.pack(fill=tk.X, pady=2)
        
        ttk.Label(frm_filter, text="Contiene:").pack(anchor="w")
        ttk.Entry(frm_filter, textvariable=self.filter_value_var).pack(fill=tk.X, pady=2)
        
        ttk.Checkbutton(frm_filter, text="Mostrar solo filtrados", 
                        variable=self.show_filtered_only, 
                        command=self._apply_view_filter).pack(anchor="w")
        
        filter_btns = ttk.Frame(frm_filter)
        filter_btns.pack(fill=tk.X)
        ttk.Button(filter_btns, text="Aplicar Filtro", command=self._apply_view_filter).pack(side=tk.LEFT, expand=True, fill=tk.X)
        ttk.Button(filter_btns, text="Limpiar", command=self._clear_filter).pack(side=tk.LEFT, expand=True, fill=tk.X)
        
        # === Sección: Editar Filas ===
        frm_rows = ttk.LabelFrame(self.tools_inner, text="📝 Filas", padding=5)
        frm_rows.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(frm_rows, text="➕ Agregar Fila", command=self._add_row).pack(fill=tk.X, pady=1)
        ttk.Button(frm_rows, text="➖ Eliminar Fila(s) Seleccionada(s)", command=self._delete_rows).pack(fill=tk.X, pady=1)
        ttk.Button(frm_rows, text="📋 Duplicar Fila", command=self._duplicate_row).pack(fill=tk.X, pady=1)
        
        # === Sección: Editar Columnas ===
        frm_cols = ttk.LabelFrame(self.tools_inner, text="📊 Columnas", padding=5)
        frm_cols.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(frm_cols, text="➕ Agregar Columna", command=self._add_column).pack(fill=tk.X, pady=1)
        ttk.Button(frm_cols, text="➖ Eliminar Columna", command=self._delete_column).pack(fill=tk.X, pady=1)
        ttk.Button(frm_cols, text="✏️ Renombrar Columna", command=self._rename_column).pack(fill=tk.X, pady=1)
        ttk.Button(frm_cols, text="🔄 Cambiar Tipo de Dato", command=self._change_dtype).pack(fill=tk.X, pady=1)
        
        # === Sección: Columnas Calculadas ===
        frm_calc = ttk.LabelFrame(self.tools_inner, text="🧮 Columna Calculada", padding=5)
        frm_calc.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(frm_calc, text="Nombre nueva columna:").pack(anchor="w")
        self.entry_calc_name = ttk.Entry(frm_calc)
        self.entry_calc_name.pack(fill=tk.X, pady=2)
        
        ttk.Label(frm_calc, text="Fórmula (ej: Col1 + Col2 * 2):").pack(anchor="w")
        self.entry_calc_formula = ttk.Entry(frm_calc)
        self.entry_calc_formula.pack(fill=tk.X, pady=2)
        
        ttk.Button(frm_calc, text="➕ Crear Columna Calculada", 
                   command=self._create_calculated_column).pack(fill=tk.X, pady=2)
        
        ttk.Label(frm_calc, text="Funciones: np.log, np.exp, np.sqrt,\nnp.abs, np.mean, np.power", 
                  font=("Consolas", 8), foreground="gray").pack(anchor="w")
        
        # === Sección: Valores Faltantes ===
        frm_missing = ttk.LabelFrame(self.tools_inner, text="❓ Valores Faltantes", padding=5)
        frm_missing.pack(fill=tk.X, padx=5, pady=5)
        
        self.lbl_missing = ttk.Label(frm_missing, text="NaN: calculando...")
        self.lbl_missing.pack(anchor="w")
        
        ttk.Label(frm_missing, text="Columna:").pack(anchor="w")
        self.cmb_missing_col = ttk.Combobox(frm_missing, state="readonly")
        self.cmb_missing_col.pack(fill=tk.X, pady=2)
        
        ttk.Label(frm_missing, text="Método:").pack(anchor="w")
        self.cmb_fill_method = ttk.Combobox(frm_missing, state="readonly",
                                             values=["Media", "Mediana", "Moda", "Valor específico", 
                                                    "Interpolación lineal", "Forward fill", "Backward fill", "Eliminar filas"])
        self.cmb_fill_method.set("Media")
        self.cmb_fill_method.pack(fill=tk.X, pady=2)
        
        self.entry_fill_value = ttk.Entry(frm_missing)
        self.entry_fill_value.pack(fill=tk.X, pady=2)
        self.entry_fill_value.insert(0, "0")
        
        ttk.Button(frm_missing, text="🔧 Aplicar a Columna", 
                   command=self._fill_missing).pack(fill=tk.X, pady=1)
        ttk.Button(frm_missing, text="🔧 Aplicar a Todas (numéricas)", 
                   command=self._fill_all_missing).pack(fill=tk.X, pady=1)
        
        # === Sección: Exportar ===
        frm_export = ttk.LabelFrame(self.tools_inner, text="💾 Exportar", padding=5)
        frm_export.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(frm_export, text="📄 Exportar a CSV", command=self._export_csv).pack(fill=tk.X, pady=1)
        ttk.Button(frm_export, text="📊 Exportar a Excel", command=self._export_excel).pack(fill=tk.X, pady=1)
        ttk.Button(frm_export, text="✅ Aplicar Cambios al Archivo de Trabajo", 
                   command=self._apply_to_workfile).pack(fill=tk.X, pady=2)
    
    def _build_table_panel(self):
        """Construye el panel de la tabla."""
        # Toolbar superior
        toolbar = ttk.Frame(self.table_frame)
        toolbar.pack(fill=tk.X, pady=2)
        
        ttk.Label(toolbar, text="Filas visibles:").pack(side=tk.LEFT, padx=5)
        self.lbl_visible_rows = ttk.Label(toolbar, text="0 / 0")
        self.lbl_visible_rows.pack(side=tk.LEFT)
        
        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=10)
        
        ttk.Label(toolbar, text="Ir a fila:").pack(side=tk.LEFT, padx=5)
        self.entry_goto = ttk.Entry(toolbar, width=8)
        self.entry_goto.pack(side=tk.LEFT)
        ttk.Button(toolbar, text="Ir", command=self._goto_row, width=4).pack(side=tk.LEFT, padx=2)
        
        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=10)
        
        self.lbl_selection = ttk.Label(toolbar, text="Selección: ninguna")
        self.lbl_selection.pack(side=tk.LEFT, padx=5)
        
        # Frame para tabla con scrollbars
        table_container = ttk.Frame(self.table_frame)
        table_container.pack(fill=tk.BOTH, expand=True)
        
        # Scrollbars
        self.vsb = ttk.Scrollbar(table_container, orient="vertical")
        self.hsb = ttk.Scrollbar(table_container, orient="horizontal")
        
        # Treeview
        self.tree = ttk.Treeview(
            table_container,
            yscrollcommand=self.vsb.set,
            xscrollcommand=self.hsb.set,
            selectmode="extended"
        )
        
        self.vsb.config(command=self.tree.yview)
        self.hsb.config(command=self.tree.xview)
        
        # Grid layout
        self.tree.grid(row=0, column=0, sticky="nsew")
        self.vsb.grid(row=0, column=1, sticky="ns")
        self.hsb.grid(row=1, column=0, sticky="ew")
        
        table_container.grid_rowconfigure(0, weight=1)
        table_container.grid_columnconfigure(0, weight=1)
        
        # Bindings
        self.tree.bind("<Double-1>", self._on_double_click)
        self.tree.bind("<<TreeviewSelect>>", self._on_select)
        self.tree.bind("<Button-1>", self._on_header_click)
        self.tree.bind("<Delete>", lambda e: self._delete_rows())
        
        # Tags para resaltar
        self.tree.tag_configure("highlight", background="#FFFF00")
        self.tree.tag_configure("edited", background="#90EE90")
        self.tree.tag_configure("missing", background="#FFB6C1")
        
        # Barra de estado
        self.status_bar = ttk.Label(self.table_frame, text="Listo", relief=tk.SUNKEN, anchor="w")
        self.status_bar.pack(fill=tk.X, side=tk.BOTTOM)
    
    def _load_from_workfile(self):
        """Carga datos desde el Archivo de Trabajo."""
        if hasattr(self.app, 'shared_filtered_df') and self.app.shared_filtered_df is not None:
            self._save_undo_state()
            self.df = self.app.shared_filtered_df.copy()
            self.df_original = self.app.shared_filtered_df.copy()
            self._refresh_table()
            self._update_info()
            self._update_combos()
            self.status_bar.config(text=f"Datos cargados: {len(self.df)} filas, {len(self.df.columns)} columnas")
        elif hasattr(self.app, 'shared_dataset') and self.app.shared_dataset is not None:
            self._save_undo_state()
            self.df = self.app.shared_dataset.copy()
            self.df_original = self.app.shared_dataset.copy()
            self._refresh_table()
            self._update_info()
            self._update_combos()
            self.status_bar.config(text=f"Datos cargados: {len(self.df)} filas, {len(self.df.columns)} columnas")
        else:
            messagebox.showwarning("Sin datos", "No hay datos en el Archivo de Trabajo.\nCargue un archivo primero.")
    
    def receive_shared_dataset(self, dataset: pd.DataFrame, filtered_data: pd.DataFrame = None, 
                                applied_filters: list = None):
        """Recibe dataset compartido desde la aplicación principal."""
        if filtered_data is not None:
            self.df = filtered_data.copy()
            self.df_original = filtered_data.copy()
        elif dataset is not None:
            self.df = dataset.copy()
            self.df_original = dataset.copy()
        
        self.undo_stack.clear()
        self.redo_stack.clear()
        
        self._refresh_table()
        self._update_info()
        self._update_combos()
        self._update_undo_buttons()
    
    def _refresh_table(self):
        """Refresca la tabla con los datos actuales."""
        if self.df is None:
            return
        
        # Limpiar tabla
        self.tree.delete(*self.tree.get_children())
        
        # Configurar columnas
        columns = ["#"] + list(self.df.columns)
        self.tree["columns"] = columns
        self.tree["show"] = "headings"
        
        for col in columns:
            self.tree.heading(col, text=col, command=lambda c=col: self._sort_by_column(c))
            # Ajustar ancho
            if col == "#":
                self.tree.column(col, width=50, minwidth=50, stretch=False)
            else:
                self.tree.column(col, width=100, minwidth=50)
        
        # Insertar filas
        df_display = self.df if not self.show_filtered_only.get() else self._get_filtered_df()
        
        for idx, row in df_display.iterrows():
            values = [idx] + [self._format_cell(v) for v in row.values]
            tags = ()
            
            # Marcar filas con valores faltantes
            if row.isna().any():
                tags = ("missing",)
            
            self.tree.insert("", tk.END, iid=str(idx), values=values, tags=tags)
        
        self.lbl_visible_rows.config(text=f"{len(df_display)} / {len(self.df)}")
    
    def _format_cell(self, value):
        """Formatea un valor para mostrar en la tabla."""
        if pd.isna(value):
            return "NaN"
        elif isinstance(value, float):
            if value == int(value):
                return str(int(value))
            return f"{value:.4g}"
        return str(value)
    
    def _update_info(self):
        """Actualiza la información del dataset."""
        if self.df is None:
            self.lbl_info.config(text="Sin datos cargados")
            self.lbl_missing.config(text="NaN: -")
            return
        
        info_text = f"Filas: {len(self.df)}\nColumnas: {len(self.df.columns)}\n"
        info_text += f"Memoria: {self.df.memory_usage(deep=True).sum() / 1024:.1f} KB"
        self.lbl_info.config(text=info_text)
        
        # Valores faltantes
        total_nan = self.df.isna().sum().sum()
        self.lbl_missing.config(text=f"NaN total: {total_nan}")
    
    def _update_combos(self):
        """Actualiza los combobox con las columnas."""
        if self.df is None:
            return
        
        cols = list(self.df.columns)
        self.cmb_filter_col["values"] = cols
        self.cmb_missing_col["values"] = cols
        
        if cols:
            self.cmb_filter_col.set(cols[0])
            self.cmb_missing_col.set(cols[0])
    
    def _save_undo_state(self):
        """Guarda el estado actual para deshacer."""
        if self.df is not None:
            self.undo_stack.append(self.df.copy())
            if len(self.undo_stack) > self.max_undo:
                self.undo_stack.pop(0)
            self.redo_stack.clear()
            self._update_undo_buttons()
    
    def _undo(self):
        """Deshace el último cambio."""
        if self.undo_stack:
            self.redo_stack.append(self.df.copy())
            self.df = self.undo_stack.pop()
            self._refresh_table()
            self._update_info()
            self._update_undo_buttons()
            self.status_bar.config(text="Deshacer aplicado")
    
    def _redo(self):
        """Rehace el último cambio deshecho."""
        if self.redo_stack:
            self.undo_stack.append(self.df.copy())
            self.df = self.redo_stack.pop()
            self._refresh_table()
            self._update_info()
            self._update_undo_buttons()
            self.status_bar.config(text="Rehacer aplicado")
    
    def _update_undo_buttons(self):
        """Actualiza el estado de los botones deshacer/rehacer."""
        self.btn_undo.config(state="normal" if self.undo_stack else "disabled")
        self.btn_redo.config(state="normal" if self.redo_stack else "disabled")
        self.lbl_undo_count.config(text=f"Historial: {len(self.undo_stack)} | {len(self.redo_stack)}")
    
    def _restore_original(self):
        """Restaura los datos originales."""
        if self.df_original is not None:
            if messagebox.askyesno("Confirmar", "¿Restaurar datos originales?\nSe perderán todos los cambios."):
                self._save_undo_state()
                self.df = self.df_original.copy()
                self._refresh_table()
                self._update_info()
                self.status_bar.config(text="Datos originales restaurados")
    
    # === Búsqueda y Filtro ===
    
    def _search_next(self):
        """Busca el siguiente texto en la tabla."""
        search_text = self.search_var.get().lower()
        if not search_text or self.df is None:
            return
        
        # Obtener selección actual
        selection = self.tree.selection()
        start_idx = 0
        if selection:
            try:
                start_idx = self.tree.index(selection[0]) + 1
            except:
                pass
        
        # Buscar
        items = self.tree.get_children()
        for i in range(start_idx, len(items)):
            item = items[i]
            values = self.tree.item(item, "values")
            for v in values:
                if search_text in str(v).lower():
                    self.tree.selection_set(item)
                    self.tree.see(item)
                    self.status_bar.config(text=f"Encontrado en fila {i+1}")
                    return
        
        # Buscar desde el inicio si no se encontró
        for i in range(0, start_idx):
            item = items[i]
            values = self.tree.item(item, "values")
            for v in values:
                if search_text in str(v).lower():
                    self.tree.selection_set(item)
                    self.tree.see(item)
                    self.status_bar.config(text=f"Encontrado en fila {i+1} (desde inicio)")
                    return
        
        self.status_bar.config(text="No encontrado")
    
    def _highlight_all(self):
        """Resalta todas las coincidencias."""
        search_text = self.search_var.get().lower()
        if not search_text or self.df is None:
            return
        
        # Limpiar resaltados anteriores
        for item in self.tree.get_children():
            current_tags = list(self.tree.item(item, "tags"))
            if "highlight" in current_tags:
                current_tags.remove("highlight")
                self.tree.item(item, tags=current_tags)
        
        # Resaltar coincidencias
        count = 0
        for item in self.tree.get_children():
            values = self.tree.item(item, "values")
            for v in values:
                if search_text in str(v).lower():
                    current_tags = list(self.tree.item(item, "tags"))
                    if "highlight" not in current_tags:
                        current_tags.append("highlight")
                    self.tree.item(item, tags=current_tags)
                    count += 1
                    break
        
        self.status_bar.config(text=f"{count} filas resaltadas")
    
    def _get_filtered_df(self):
        """Obtiene el DataFrame filtrado por vista."""
        if self.df is None:
            return None
        
        col = self.filter_column_var.get()
        val = self.filter_value_var.get().lower()
        
        if not col or not val or col not in self.df.columns:
            return self.df
        
        mask = self.df[col].astype(str).str.lower().str.contains(val, na=False)
        return self.df[mask]
    
    def _apply_view_filter(self):
        """Aplica el filtro de vista."""
        self._refresh_table()
    
    def _clear_filter(self):
        """Limpia el filtro de vista."""
        self.filter_column_var.set("")
        self.filter_value_var.set("")
        self.show_filtered_only.set(False)
        self._refresh_table()
    
    # === Edición de Celdas ===
    
    def _on_double_click(self, event):
        """Maneja doble click para editar celda."""
        region = self.tree.identify("region", event.x, event.y)
        if region != "cell":
            return
        
        item = self.tree.identify_row(event.y)
        column = self.tree.identify_column(event.x)
        
        if not item or not column:
            return
        
        # Obtener índice de columna (ignorar #0 y columna de índice)
        col_idx = int(column.replace("#", "")) - 1
        if col_idx < 0:  # Columna de índice
            return
        
        # Obtener valor actual
        col_name = self.df.columns[col_idx - 1] if col_idx > 0 else None
        if col_name is None:
            return
        
        row_idx = int(item)
        current_value = self.df.loc[row_idx, col_name]
        
        # Crear entry para edición
        bbox = self.tree.bbox(item, column)
        if not bbox:
            return
        
        x, y, width, height = bbox
        
        edit_entry = ttk.Entry(self.tree)
        edit_entry.place(x=x, y=y, width=width, height=height)
        edit_entry.insert(0, "" if pd.isna(current_value) else str(current_value))
        edit_entry.select_range(0, tk.END)
        edit_entry.focus()
        
        def save_edit(event=None):
            new_value = edit_entry.get()
            edit_entry.destroy()
            
            self._save_undo_state()
            
            # Convertir al tipo de dato de la columna
            try:
                dtype = self.df[col_name].dtype
                if new_value == "" or new_value.lower() == "nan":
                    self.df.loc[row_idx, col_name] = np.nan
                elif pd.api.types.is_numeric_dtype(dtype):
                    self.df.loc[row_idx, col_name] = float(new_value)
                else:
                    self.df.loc[row_idx, col_name] = new_value
                
                self._refresh_table()
                self._update_info()
                self.status_bar.config(text=f"Celda editada: [{row_idx}, {col_name}]")
            except Exception as e:
                messagebox.showerror("Error", f"No se pudo guardar el valor:\n{e}")
        
        def cancel_edit(event=None):
            edit_entry.destroy()
        
        edit_entry.bind("<Return>", save_edit)
        edit_entry.bind("<Escape>", cancel_edit)
        edit_entry.bind("<FocusOut>", save_edit)
    
    def _on_select(self, event):
        """Maneja selección de filas."""
        selection = self.tree.selection()
        if selection:
            self.lbl_selection.config(text=f"Selección: {len(selection)} fila(s)")
        else:
            self.lbl_selection.config(text="Selección: ninguna")
    
    def _on_header_click(self, event):
        """Maneja click en header para ordenar."""
        region = self.tree.identify("region", event.x, event.y)
        if region == "heading":
            column = self.tree.identify_column(event.x)
            col_idx = int(column.replace("#", "")) - 1
            if col_idx >= 0:
                col_name = self.tree["columns"][col_idx]
                self._sort_by_column(col_name)
    
    def _sort_by_column(self, col_name):
        """Ordena la tabla por columna."""
        if self.df is None or col_name == "#":
            return
        
        if col_name not in self.df.columns:
            return
        
        # Toggle ascending/descending
        if self.sort_column == col_name:
            self.sort_ascending = not self.sort_ascending
        else:
            self.sort_column = col_name
            self.sort_ascending = True
        
        self._save_undo_state()
        
        try:
            self.df = self.df.sort_values(by=col_name, ascending=self.sort_ascending, na_position='last')
            self.df = self.df.reset_index(drop=True)
            self._refresh_table()
            
            order = "↑" if self.sort_ascending else "↓"
            self.status_bar.config(text=f"Ordenado por {col_name} {order}")
        except Exception as e:
            self.status_bar.config(text=f"Error al ordenar: {e}")
    
    def _goto_row(self):
        """Va a una fila específica."""
        try:
            row_num = int(self.entry_goto.get())
            items = self.tree.get_children()
            if 0 <= row_num < len(items):
                item = items[row_num]
                self.tree.selection_set(item)
                self.tree.see(item)
                self.status_bar.config(text=f"Fila {row_num}")
            else:
                self.status_bar.config(text=f"Fila fuera de rango (0-{len(items)-1})")
        except ValueError:
            self.status_bar.config(text="Ingrese un número válido")
    
    # === Edición de Filas ===
    
    def _add_row(self):
        """Agrega una nueva fila vacía."""
        if self.df is None:
            return
        
        self._save_undo_state()
        
        new_row = pd.DataFrame([[np.nan] * len(self.df.columns)], columns=self.df.columns)
        self.df = pd.concat([self.df, new_row], ignore_index=True)
        
        self._refresh_table()
        self._update_info()
        
        # Seleccionar la nueva fila
        items = self.tree.get_children()
        if items:
            self.tree.selection_set(items[-1])
            self.tree.see(items[-1])
        
        self.status_bar.config(text="Nueva fila agregada")
    
    def _delete_rows(self):
        """Elimina las filas seleccionadas."""
        if self.df is None:
            return
        
        selection = self.tree.selection()
        if not selection:
            messagebox.showinfo("Info", "Seleccione filas para eliminar")
            return
        
        if not messagebox.askyesno("Confirmar", f"¿Eliminar {len(selection)} fila(s)?"):
            return
        
        self._save_undo_state()
        
        indices_to_drop = [int(item) for item in selection]
        self.df = self.df.drop(indices_to_drop).reset_index(drop=True)
        
        self._refresh_table()
        self._update_info()
        self.status_bar.config(text=f"{len(selection)} fila(s) eliminada(s)")
    
    def _duplicate_row(self):
        """Duplica la fila seleccionada."""
        if self.df is None:
            return
        
        selection = self.tree.selection()
        if not selection or len(selection) != 1:
            messagebox.showinfo("Info", "Seleccione exactamente una fila para duplicar")
            return
        
        self._save_undo_state()
        
        row_idx = int(selection[0])
        new_row = self.df.loc[[row_idx]].copy()
        self.df = pd.concat([self.df, new_row], ignore_index=True)
        
        self._refresh_table()
        self._update_info()
        self.status_bar.config(text="Fila duplicada")
    
    # === Edición de Columnas ===
    
    def _add_column(self):
        """Agrega una nueva columna."""
        if self.df is None:
            return
        
        col_name = simpledialog.askstring("Nueva Columna", "Nombre de la nueva columna:")
        if not col_name:
            return
        
        if col_name in self.df.columns:
            messagebox.showerror("Error", f"La columna '{col_name}' ya existe")
            return
        
        self._save_undo_state()
        self.df[col_name] = np.nan
        
        self._refresh_table()
        self._update_info()
        self._update_combos()
        self.status_bar.config(text=f"Columna '{col_name}' agregada")
    
    def _delete_column(self):
        """Elimina una columna."""
        if self.df is None:
            return
        
        col_name = simpledialog.askstring("Eliminar Columna", 
                                          f"Columnas: {', '.join(self.df.columns)}\n\nNombre de la columna a eliminar:")
        if not col_name:
            return
        
        if col_name not in self.df.columns:
            messagebox.showerror("Error", f"La columna '{col_name}' no existe")
            return
        
        if not messagebox.askyesno("Confirmar", f"¿Eliminar columna '{col_name}'?"):
            return
        
        self._save_undo_state()
        self.df = self.df.drop(columns=[col_name])
        
        self._refresh_table()
        self._update_info()
        self._update_combos()
        self.status_bar.config(text=f"Columna '{col_name}' eliminada")
    
    def _rename_column(self):
        """Renombra una columna."""
        if self.df is None:
            return
        
        # Diálogo para seleccionar columna
        dialog = tk.Toplevel(self.parent)
        dialog.title("Renombrar Columna")
        dialog.geometry("350x150")
        dialog.transient(self.parent)
        dialog.grab_set()
        
        ttk.Label(dialog, text="Columna actual:").pack(pady=5)
        cmb_col = ttk.Combobox(dialog, values=list(self.df.columns), state="readonly")
        cmb_col.pack(fill=tk.X, padx=20)
        if self.df.columns.any():
            cmb_col.set(self.df.columns[0])
        
        ttk.Label(dialog, text="Nuevo nombre:").pack(pady=5)
        entry_new = ttk.Entry(dialog)
        entry_new.pack(fill=tk.X, padx=20)
        
        def do_rename():
            old_name = cmb_col.get()
            new_name = entry_new.get().strip()
            
            if not new_name:
                messagebox.showerror("Error", "Ingrese un nombre válido")
                return
            
            if new_name in self.df.columns and new_name != old_name:
                messagebox.showerror("Error", f"La columna '{new_name}' ya existe")
                return
            
            self._save_undo_state()
            self.df = self.df.rename(columns={old_name: new_name})
            
            self._refresh_table()
            self._update_combos()
            self.status_bar.config(text=f"Columna renombrada: {old_name} → {new_name}")
            dialog.destroy()
        
        ttk.Button(dialog, text="Renombrar", command=do_rename).pack(pady=10)
    
    def _change_dtype(self):
        """Cambia el tipo de dato de una columna."""
        if self.df is None:
            return
        
        dialog = tk.Toplevel(self.parent)
        dialog.title("Cambiar Tipo de Dato")
        dialog.geometry("400x200")
        dialog.transient(self.parent)
        dialog.grab_set()
        
        ttk.Label(dialog, text="Columna:").pack(pady=5)
        cmb_col = ttk.Combobox(dialog, values=list(self.df.columns), state="readonly")
        cmb_col.pack(fill=tk.X, padx=20)
        if self.df.columns.any():
            cmb_col.set(self.df.columns[0])
        
        lbl_current = ttk.Label(dialog, text="Tipo actual: -")
        lbl_current.pack(pady=2)
        
        def update_current_type(event=None):
            col = cmb_col.get()
            if col in self.df.columns:
                lbl_current.config(text=f"Tipo actual: {self.df[col].dtype}")
        
        cmb_col.bind("<<ComboboxSelected>>", update_current_type)
        update_current_type()
        
        ttk.Label(dialog, text="Nuevo tipo:").pack(pady=5)
        cmb_dtype = ttk.Combobox(dialog, values=["float64", "int64", "str", "category", "datetime64[ns]", "bool"], 
                                  state="readonly")
        cmb_dtype.set("float64")
        cmb_dtype.pack(fill=tk.X, padx=20)
        
        def do_change():
            col = cmb_col.get()
            new_dtype = cmb_dtype.get()
            
            self._save_undo_state()
            
            try:
                if new_dtype == "datetime64[ns]":
                    self.df[col] = pd.to_datetime(self.df[col], errors='coerce')
                elif new_dtype == "str":
                    self.df[col] = self.df[col].astype(str)
                elif new_dtype == "category":
                    self.df[col] = self.df[col].astype("category")
                elif new_dtype == "bool":
                    self.df[col] = self.df[col].astype(bool)
                else:
                    self.df[col] = pd.to_numeric(self.df[col], errors='coerce').astype(new_dtype)
                
                self._refresh_table()
                self.status_bar.config(text=f"Tipo de {col} cambiado a {new_dtype}")
                dialog.destroy()
            except Exception as e:
                messagebox.showerror("Error", f"No se pudo cambiar el tipo:\n{e}")
        
        ttk.Button(dialog, text="Cambiar Tipo", command=do_change).pack(pady=10)
    
    # === Columnas Calculadas ===
    
    def _create_calculated_column(self):
        """Crea una columna calculada."""
        if self.df is None:
            return
        
        col_name = self.entry_calc_name.get().strip()
        formula = self.entry_calc_formula.get().strip()
        
        if not col_name:
            messagebox.showerror("Error", "Ingrese un nombre para la columna")
            return
        
        if not formula:
            messagebox.showerror("Error", "Ingrese una fórmula")
            return
        
        if col_name in self.df.columns:
            if not messagebox.askyesno("Confirmar", f"La columna '{col_name}' ya existe. ¿Sobrescribir?"):
                return
        
        self._save_undo_state()
        
        try:
            # Crear un diccionario con las columnas como variables
            local_vars = {col: self.df[col] for col in self.df.columns}
            local_vars['np'] = np
            local_vars['pd'] = pd
            
            # Evaluar la fórmula
            result = eval(formula, {"__builtins__": {}}, local_vars)
            self.df[col_name] = result
            
            self._refresh_table()
            self._update_info()
            self._update_combos()
            self.status_bar.config(text=f"Columna calculada '{col_name}' creada")
            
            # Limpiar entradas
            self.entry_calc_name.delete(0, tk.END)
            self.entry_calc_formula.delete(0, tk.END)
            
        except Exception as e:
            messagebox.showerror("Error en Fórmula", f"No se pudo evaluar la fórmula:\n{e}\n\n"
                                 f"Ejemplo: Col1 + Col2 * 2\n"
                                 f"Funciones: np.log(Col1), np.sqrt(Col1)")
    
    # === Valores Faltantes ===
    
    def _fill_missing(self):
        """Rellena valores faltantes en una columna."""
        if self.df is None:
            return
        
        col = self.cmb_missing_col.get()
        method = self.cmb_fill_method.get()
        
        if not col or col not in self.df.columns:
            messagebox.showerror("Error", "Seleccione una columna válida")
            return
        
        self._save_undo_state()
        
        try:
            if method == "Media":
                self.df[col] = self.df[col].fillna(self.df[col].mean())
            elif method == "Mediana":
                self.df[col] = self.df[col].fillna(self.df[col].median())
            elif method == "Moda":
                mode_val = self.df[col].mode()
                if len(mode_val) > 0:
                    self.df[col] = self.df[col].fillna(mode_val.iloc[0])
            elif method == "Valor específico":
                val = self.entry_fill_value.get()
                try:
                    val = float(val)
                except:
                    pass
                self.df[col] = self.df[col].fillna(val)
            elif method == "Interpolación lineal":
                self.df[col] = self.df[col].interpolate(method='linear')
            elif method == "Forward fill":
                self.df[col] = self.df[col].ffill()
            elif method == "Backward fill":
                self.df[col] = self.df[col].bfill()
            elif method == "Eliminar filas":
                self.df = self.df.dropna(subset=[col]).reset_index(drop=True)
            
            self._refresh_table()
            self._update_info()
            self.status_bar.config(text=f"Valores faltantes en '{col}' rellenados con {method}")
            
        except Exception as e:
            messagebox.showerror("Error", f"No se pudieron rellenar los valores:\n{e}")
    
    def _fill_all_missing(self):
        """Rellena valores faltantes en todas las columnas numéricas."""
        if self.df is None:
            return
        
        method = self.cmb_fill_method.get()
        
        if not messagebox.askyesno("Confirmar", f"¿Aplicar '{method}' a todas las columnas numéricas?"):
            return
        
        self._save_undo_state()
        
        try:
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            
            for col in numeric_cols:
                if method == "Media":
                    self.df[col] = self.df[col].fillna(self.df[col].mean())
                elif method == "Mediana":
                    self.df[col] = self.df[col].fillna(self.df[col].median())
                elif method == "Interpolación lineal":
                    self.df[col] = self.df[col].interpolate(method='linear')
                elif method == "Forward fill":
                    self.df[col] = self.df[col].ffill()
                elif method == "Backward fill":
                    self.df[col] = self.df[col].bfill()
            
            self._refresh_table()
            self._update_info()
            self.status_bar.config(text=f"Valores faltantes rellenados en {len(numeric_cols)} columnas")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error al rellenar valores:\n{e}")
    
    # === Exportar ===
    
    def _export_csv(self):
        """Exporta los datos a CSV."""
        if self.df is None:
            return
        
        filepath = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv"), ("Todos", "*.*")],
            title="Exportar a CSV"
        )
        
        if filepath:
            try:
                self.df.to_csv(filepath, index=False, encoding='utf-8-sig')
                self.status_bar.config(text=f"Exportado: {filepath}")
                messagebox.showinfo("Éxito", f"Datos exportados a:\n{filepath}")
            except Exception as e:
                messagebox.showerror("Error", f"No se pudo exportar:\n{e}")
    
    def _export_excel(self):
        """Exporta los datos a Excel."""
        if self.df is None:
            return
        
        filepath = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            filetypes=[("Excel", "*.xlsx"), ("Todos", "*.*")],
            title="Exportar a Excel"
        )
        
        if filepath:
            try:
                self.df.to_excel(filepath, index=False)
                self.status_bar.config(text=f"Exportado: {filepath}")
                messagebox.showinfo("Éxito", f"Datos exportados a:\n{filepath}")
            except Exception as e:
                messagebox.showerror("Error", f"No se pudo exportar:\n{e}")
    
    def _apply_to_workfile(self):
        """Aplica los cambios al Archivo de Trabajo."""
        if self.df is None:
            return
        
        if not messagebox.askyesno("Confirmar", 
                                   "¿Aplicar los cambios al Archivo de Trabajo?\n\n"
                                   "Esto actualizará los datos en todas las pestañas."):
            return
        
        try:
            # Actualizar el dataset compartido
            if hasattr(self.app, 'shared_filtered_df'):
                self.app.shared_filtered_df = self.df.copy()
            
            if hasattr(self.app, 'shared_dataset'):
                # Actualizar solo las filas que existen en ambos
                # o reemplazar completamente si el usuario modificó estructura
                self.app.shared_dataset = self.df.copy()
            
            # Notificar a otras pestañas si tienen el método
            if hasattr(self.app, 'propagate_dataset_to_tabs'):
                self.app.propagate_dataset_to_tabs()
            
            self.df_original = self.df.copy()  # Actualizar original
            self.status_bar.config(text="Cambios aplicados al Archivo de Trabajo")
            messagebox.showinfo("Éxito", "Los cambios se han aplicado al Archivo de Trabajo.")
            
        except Exception as e:
            messagebox.showerror("Error", f"No se pudieron aplicar los cambios:\n{e}")


def create_data_editor_tab(notebook, app_instance) -> DataEditorTab:
    """
    Crea y retorna una instancia de DataEditorTab.
    
    Args:
        notebook: El widget ttk.Notebook donde agregar la pestaña
        app_instance: Instancia de la aplicación principal
    
    Returns:
        DataEditorTab: Instancia del editor de datos
    """
    frame = ttk.Frame(notebook)
    notebook.add(frame, text="📝 Editor de Datos")
    
    editor = DataEditorTab(frame, app_instance)
    return editor
