#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tkinter as tk
from tkinter import ttk, simpledialog, messagebox, filedialog
import numpy as np
import pandas as pd
import re
import traceback
import os
import json
import copy
from datetime import datetime

try:
    from MATLAB_filter_component import FilterComponent
    FILTER_COMPONENT_AVAILABLE = True
except ImportError:
    FilterComponent = None
    FILTER_COMPONENT_AVAILABLE = False

RECENT_WORKFILES_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "recent_workfiles.json")
RECENT_WORKFILES_LIMIT = 8

class AddVariablesTab(ttk.Frame):
    def __init__(self, parent_notebook, main_app_instance):
        super().__init__(parent_notebook)
        self.main_app = main_app_instance
        self.data = None
        self.filtered_data = None
        self.file_path = None
        self.filter_component = None
        self.filter_status_var = tk.StringVar(value="Sin datos cargados.")
        self.original_columns = [] # Store original order
        self.selected_columns = [] # Store user selected columns for analysis
        self.source_path = None
        self.source_sheet = None
        self.analysis_presets = {}
        self._recent_entries = self._load_recent_entries()
        self._recent_label_map = {}
        self.recent_combo = None
        self.recent_combo_var = tk.StringVar(value="")

        # --- Setup Scrollable Canvas Structure ---
        self.canvas = tk.Canvas(self)
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        
        # Configure canvas
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        
        # Pack external containers
        self.scrollbar.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)
        
        # Create scrollable frame inside canvas
        main_frame = ttk.Frame(self.canvas, padding="10")
        self.canvas_window_id = self.canvas.create_window((0, 0), window=main_frame, anchor="nw")
        
        # Configure scrollbar events
        main_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        self.canvas.bind(
            "<Configure>",
            self._on_canvas_configure
        )
        
        # Also bind mousewheel for convenience
        self.bind_all("<MouseWheel>", self._on_mousewheel)

        # --- Content of AddVariablesTab (now parented to main_frame) ---

        # --- Top Controls Frame (Load, Info, Create, Sort) ---
        top_controls_frame = ttk.Frame(main_frame)
        top_controls_frame.pack(fill=tk.X, padx=10, pady=10)

        # Row 1: Load Data & Info
        row1_frame = ttk.Frame(top_controls_frame)
        row1_frame.pack(fill=tk.X, pady=(0, 5))
        
        btn_load_data = ttk.Button(row1_frame, text="Cargar Archivo", command=self.load_data)
        btn_load_data.pack(side=tk.LEFT, padx=(0, 10))

        ttk.Label(row1_frame, text="Recientes:").pack(side=tk.LEFT, padx=(0, 5))
        self.recent_combo = ttk.Combobox(row1_frame, textvariable=self.recent_combo_var, state="readonly", width=35)
        self.recent_combo.pack(side=tk.LEFT, padx=(0, 10))
        self.recent_combo.bind("<<ComboboxSelected>>", self._handle_recent_selection)

        self.dataset_info_var = tk.StringVar(value="Archivo actual: Sin archivo cargado.")
        self.dataset_info_label = ttk.Label(row1_frame, textvariable=self.dataset_info_var, anchor="w")
        self.dataset_info_label.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Row 2: Create Variable & Sort
        row2_frame = ttk.Frame(top_controls_frame)
        row2_frame.pack(fill=tk.X)

        btn_create_var = ttk.Button(row2_frame, text="Crear Nueva Variable", command=self.create_variable_by_formula)
        btn_create_var.pack(side=tk.LEFT, padx=(0, 10))

        self.sort_vars = tk.BooleanVar(value=False)
        self.chk_sort = ttk.Checkbutton(row2_frame, text="Ordenar variables alfabéticamente", 
                                        variable=self.sort_vars, command=self._on_sort_toggled)
        self.chk_sort.pack(side=tk.LEFT)

        # Button to Select Variables (Popup)
        btn_select_vars = ttk.Button(row2_frame, text="Seleccionar Variables...", command=self._open_variable_selection_dialog)
        btn_select_vars.pack(side=tk.LEFT, padx=(10, 0))

        # Shared filters section
        filters_frame = ttk.LabelFrame(main_frame, text="Filtros Compartidos")
        filters_frame.pack(fill=tk.X, padx=10, pady=5)

        if FILTER_COMPONENT_AVAILABLE and FilterComponent is not None:
            self.filter_component = FilterComponent(filters_frame, log_callback=self._log)
            self.filter_component.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

            filter_buttons_frame = ttk.Frame(filters_frame)
            filter_buttons_frame.pack(fill=tk.X, padx=5, pady=5)

            self.btn_apply_filters = ttk.Button(filter_buttons_frame, text="Aplicar Filtros", command=self.apply_filters_to_dataset, state=tk.DISABLED)
            self.btn_apply_filters.pack(side=tk.LEFT, padx=2)

            self.btn_clear_filters = ttk.Button(filter_buttons_frame, text="Limpiar Filtros", command=self.clear_filters, state=tk.DISABLED)
            self.btn_clear_filters.pack(side=tk.LEFT, padx=2)
        else:
            ttk.Label(filters_frame, text="El componente de filtros no está disponible.").pack(padx=10, pady=10, anchor="w")
            self.btn_apply_filters = None
            self.btn_clear_filters = None

        self.filter_status_label = ttk.Label(filters_frame, textvariable=self.filter_status_var, wraplength=800, justify=tk.LEFT)
        self.filter_status_label.pack(fill=tk.X, padx=5, pady=(0, 5))

        self._update_filter_buttons_state()
        self._propagate_shared_dataset(None, [])
        self._update_dataset_info_display()
        self._refresh_recent_combo()



        # --- Variable Operations Frame (Rename & Recode Side-by-Side or Compact) ---
        ops_container = ttk.Frame(main_frame)
        ops_container.pack(fill=tk.X, padx=10, pady=10)

        # Rename Frame
        frame_rename = ttk.LabelFrame(ops_container, text="Renombrar Variable")
        frame_rename.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

        ttk.Label(frame_rename, text="Variable:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.combo_rename_var = ttk.Combobox(frame_rename, state="readonly", width=25)
        self.combo_rename_var.grid(row=0, column=1, padx=5, pady=5, sticky="w")

        ttk.Label(frame_rename, text="Nuevo Nombre:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.entry_new_name = ttk.Entry(frame_rename, width=25)
        self.entry_new_name.grid(row=1, column=1, padx=5, pady=5, sticky="w")

        btn_rename = ttk.Button(frame_rename, text="Renombrar", command=self.rename_variable)
        btn_rename.grid(row=2, column=1, padx=5, pady=5, sticky="e")

        # Recode Frame
        frame_recode = ttk.LabelFrame(ops_container, text="Recodificar Variable")
        frame_recode.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(5, 0))

        ttk.Label(frame_recode, text="Variable:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.combo_recode_var = ttk.Combobox(frame_recode, state="readonly", width=25)
        self.combo_recode_var.grid(row=0, column=1, padx=5, pady=5, sticky="w")

        ttk.Label(frame_recode, text="Mapa (F:1, M:0):").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.entry_recode_map = ttk.Entry(frame_recode, width=25)
        self.entry_recode_map.grid(row=1, column=1, padx=5, pady=5, sticky="w")

        btn_recode = ttk.Button(frame_recode, text="Recodificar", command=self.recode_variable)
        btn_recode.grid(row=2, column=1, padx=5, pady=5, sticky="e")

        # Frame for saving data
        frame_save = ttk.LabelFrame(main_frame, text="Guardar Datos Modificados")
        frame_save.pack(fill=tk.X, padx=10, pady=10, ipady=5)

        btn_save_as = ttk.Button(frame_save, text="Guardar como...", command=self.save_data_as)
        btn_save_as.pack(side=tk.LEFT, padx=10, pady=10)

        btn_overwrite = ttk.Button(frame_save, text="Sustituir base actual", command=self.overwrite_data)
        btn_overwrite.pack(side=tk.LEFT, padx=10, pady=10)

    def _on_canvas_configure(self, event):
        """Ensure the inner frame fills the canvas width."""
        self.canvas.itemconfig(self.canvas_window_id, width=event.width)

    def _on_mousewheel(self, event):
        """Scroll with mouse wheel."""
        if self.canvas.winfo_exists():
            self.canvas.yview_scroll(int(-1*(event.delta/120)), "units")

    def load_data(self):
        filepath = filedialog.askopenfilename(filetypes=[("Excel files", "*.xlsx *.xls"), ("CSV files", "*.csv")])
        if not filepath:
            return
        try:
            read_result = self._read_dataset_from_path(filepath, prompt_sheet=True)
            if read_result is None:
                return

            df, display_path, source_path, sheet_name = read_result
            self._finalize_dataset_load(df, display_path, source_path, sheet_name)
            messagebox.showinfo("Éxito", "Datos cargados correctamente.", parent=self)
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo cargar el archivo: {e}", parent=self)
            self.data = None
            self.file_path = None
            self.source_path = None
            self.source_sheet = None
            self.filtered_data = None
            self.analysis_presets = {}
            self._after_dataset_changed(reapply_filters=False)

    def _ask_sheet_name(self, sheet_names):
        """Muestra un diálogo para seleccionar la hoja de Excel."""
        dialog = tk.Toplevel(self)
        dialog.title("Seleccionar Hoja")
        dialog.geometry("300x150")
        dialog.transient(self)
        dialog.grab_set()
        
        # Centrar
        x = self.winfo_x() + 50
        y = self.winfo_y() + 50
        dialog.geometry(f"+{x}+{y}")
        
        ttk.Label(dialog, text="El archivo tiene múltiples hojas.\nSeleccione una:").pack(pady=10)
        
        cmb_sheets = ttk.Combobox(dialog, values=sheet_names, state="readonly")
        cmb_sheets.pack(pady=5, padx=10, fill="x")
        cmb_sheets.set(sheet_names[0])
        
        selected_sheet = tk.StringVar()
        
        def on_ok():
            selected_sheet.set(cmb_sheets.get())
            dialog.destroy()
            
        def on_cancel():
            selected_sheet.set("")
            dialog.destroy()
            
        btn_frame = ttk.Frame(dialog)
        btn_frame.pack(pady=10)
        
        ttk.Button(btn_frame, text="Aceptar", command=on_ok).pack(side="left", padx=5)
        ttk.Button(btn_frame, text="Cancelar", command=on_cancel).pack(side="left", padx=5)
        
        self.wait_window(dialog)
        return selected_sheet.get() if selected_sheet.get() else None

    def recode_variable(self):
        if self.data is None:
            messagebox.showerror("Error", "No hay datos cargados.", parent=self)
            return

        var_to_recode = self.combo_recode_var.get()
        recode_map_str = self.entry_recode_map.get().strip()

        if not var_to_recode:
            messagebox.showerror("Error", "Seleccione una variable para recodificar.", parent=self)
            return

        if not recode_map_str:
            messagebox.showerror("Error", "Ingrese el mapeo de valores.", parent=self)
            return

        try:
            # Parse the mapping string: "key1:val1,key2:val2"
            recode_map = {}
            for pair in recode_map_str.split(','):
                if ':' not in pair:
                    continue
                key, value = pair.split(':', 1)
                key = key.strip()
                value = value.strip()
                # Try to convert key and value to numeric if possible
                try:
                    key = pd.to_numeric(key)
                except ValueError:
                    pass
                try:
                    value = pd.to_numeric(value)
                except ValueError:
                    pass
                recode_map[key] = value

            if not recode_map:
                messagebox.showerror("Error", "El formato del mapeo es incorrecto. Use: clave1:valor1,clave2:valor2", parent=self)
                return

            # Apply the recoding
            self.data[var_to_recode] = self.data[var_to_recode].replace(recode_map)
            
            # Update variable selectors and broadcast change
            self._after_dataset_changed(reapply_filters=True)
            self.entry_recode_map.delete(0, tk.END)

        except Exception as e:
            messagebox.showerror("Error", f"Ocurrió un error al recodificar la variable:\n{e}", parent=self)

    def rename_variable(self):
        if self.data is None:
            messagebox.showerror("Error", "No hay datos cargados.", parent=self)
            return

        old_name = self.combo_rename_var.get()
        new_name = self.entry_new_name.get().strip()

        if not old_name:
            messagebox.showerror("Error", "Seleccione una variable para renombrar.", parent=self)
            return

        if not new_name:
            messagebox.showerror("Error", "Ingrese un nuevo nombre para la variable.", parent=self)
            return

        if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", new_name):
            messagebox.showerror("Nombre Inválido", "El nuevo nombre de la variable debe ser un identificador Python válido.", parent=self)
            return

        if new_name in self.data.columns:
            messagebox.showerror("Error", f"La variable '{new_name}' ya existe.", parent=self)
            return

        try:
            self.data.rename(columns={old_name: new_name}, inplace=True)
            messagebox.showinfo("Éxito", f"La variable '{old_name}' ha sido renombrada a '{new_name}'.", parent=self)

            # Update original_columns list
            if hasattr(self, 'original_columns') and self.original_columns:
                self.original_columns = [new_name if x == old_name else x for x in self.original_columns]
            
            if hasattr(self, 'selected_columns') and self.selected_columns:
                self.selected_columns = [new_name if x == old_name else x for x in self.selected_columns]

            # Update variable selectors and broadcast change
            self._enforce_sort()
            self._after_dataset_changed(reapply_filters=True)
            self.entry_new_name.delete(0, tk.END)

        except Exception as e:
            messagebox.showerror("Error", f"Ocurrió un error al renombrar la variable:\n{e}", parent=self)

    def update_variable_lists(self):
        if self.data is not None:
            cols = list(self.data.columns)
            
            # Filter dropdowns to only show selected variables
            if self.selected_columns:
                display_cols = [c for c in cols if c in self.selected_columns]
            else:
                display_cols = [] # Or cols if we wanted fallback, but strict is requested

            self.combo_rename_var['values'] = display_cols
            self.combo_recode_var['values'] = display_cols
            
            if display_cols:
                current_rename = self.combo_rename_var.get()
                current_recode = self.combo_recode_var.get()
                
                if current_rename not in display_cols:
                    self.combo_rename_var.set(display_cols[0])
                if current_recode not in display_cols:
                    self.combo_recode_var.set(display_cols[0])
            else:
                self.combo_rename_var.set('')
                self.combo_recode_var.set('')

            
            # Removing aggressive auto-filtering of selected_columns. 
            # Propagation logic handles validity check. 
            pass 


        else:
            self.combo_rename_var['values'] = []
            self.combo_rename_var.set('')
            self.combo_recode_var['values'] = []
            self.combo_recode_var.set('')
            self.selected_columns = []

        self._update_filter_buttons_state()

    def _open_variable_selection_dialog(self):
        if self.data is None:
             messagebox.showinfo("Info", "Cargue datos primero.", parent=self)
             return

        dlg = tk.Toplevel(self)
        dlg.title("Seleccionar Variables")
        dlg.geometry("450x600")
        dlg.transient(self)
        dlg.grab_set()

        # --- Search Bar ---
        frame_search = ttk.Frame(dlg, padding="5")
        frame_search.pack(fill=tk.X)
        ttk.Label(frame_search, text="Buscar:").pack(side=tk.LEFT, padx=5)
        
        search_var = tk.StringVar()
        entry_search = ttk.Entry(frame_search, textvariable=search_var)
        entry_search.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # --- List Area ---
        frame_list_container = ttk.Frame(dlg)
        frame_list_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        canvas = tk.Canvas(frame_list_container, bg="white")
        scrollbar = ttk.Scrollbar(frame_list_container, orient="vertical", command=canvas.yview)
        
        # Create a frame inside the canvas with a specific style or bg if needed
        # Using a style or just frame is fine. 
        scroll_frame = ttk.Frame(canvas)

        scroll_frame_id = canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
        
        def _configure_scroll_region(event):
            canvas.configure(scrollregion=canvas.bbox("all"))
        
        def _configure_window_width(event):
            canvas.itemconfig(scroll_frame_id, width=event.width)

        scroll_frame.bind("<Configure>", _configure_scroll_region)
        canvas.bind("<Configure>", _configure_window_width)
        
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Mousewheel binding
        def _on_dlg_mousewheel(event):
             if canvas.winfo_exists():
                canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
        # Bind mousewheel to canvas and its children if possible, or just bind_all and check focus?
        # bind_all is easiest for a modal dialog
        dlg.bind_all("<MouseWheel>", _on_dlg_mousewheel)

        # State management
        # We need a master dictionary of {col_name: BooleanVar} to track state across searches
        self._dlg_check_vars = {} 
        all_cols = list(self.data.columns)
        
        # Initialize vars based on current selection
        for col in all_cols:
            is_selected = col in self.selected_columns
            self._dlg_check_vars[col] = tk.BooleanVar(value=is_selected)

        def _populate_list(filter_text=""):
            # Clear current widgets in scroll_frame
            for widget in scroll_frame.winfo_children():
                widget.destroy()
            
            filter_text = filter_text.lower()
            
            # Determine order
            if self.sort_vars.get():
                 display_cols = sorted(all_cols, key=str.lower)
            else:
                 display_cols = all_cols
            
            # Create checkboxes
            row_idx = 0
            for col in display_cols:
                if filter_text and filter_text not in col.lower():
                    continue
                
                var = self._dlg_check_vars[col]
                cb = ttk.Checkbutton(scroll_frame, text=col, variable=var)
                cb.pack(anchor="w", padx=5, pady=2, fill=tk.X)
                row_idx += 1
            
            if row_idx == 0:
                ttk.Label(scroll_frame, text="(No hay coincidencias)", foreground="gray").pack(padx=5, pady=5)

        # Search trace
        def _on_search_change(*args):
            _populate_list(search_var.get())
        
        search_var.trace_add("write", _on_search_change)

        # Initial populate
        _populate_list()

        # --- Footer Actions ---
        btn_frame = ttk.Frame(dlg, padding="10")
        btn_frame.pack(side=tk.BOTTOM, fill=tk.X)

        def select_all_visible():
            filter_txt = search_var.get().lower()
            for col, var in self._dlg_check_vars.items():
                if not filter_txt or filter_txt in col.lower():
                    var.set(True)

        def deselect_all_visible():
            filter_txt = search_var.get().lower()
            for col, var in self._dlg_check_vars.items():
                if not filter_txt or filter_txt in col.lower():
                    var.set(False)

        def apply_selection():
             # Build list from vars
             new_selection = [col for col, var in self._dlg_check_vars.items() if var.get()]
             
             if not new_selection:
                 if not messagebox.askyesno("Confirmar", "No ha seleccionado ninguna variable.\n¿Desea continuar con cero variables (ocultará todo)?", parent=dlg):
                     return

             self.selected_columns = new_selection
             
             dlg.unbind_all("<MouseWheel>") # Clean up
             self._after_dataset_changed(reapply_filters=False)
             dlg.destroy()

        # Left side: Select/Deselect
        frame_actions_left = ttk.Frame(btn_frame)
        frame_actions_left.pack(side=tk.LEFT)
        
        ttk.Button(frame_actions_left, text="Marcar Todos", command=select_all_visible).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(frame_actions_left, text="Desmarcar Todos", command=deselect_all_visible).pack(side=tk.LEFT)

        # Right side: Apply
        try:
            current_themes = ttk.Style().theme_names()
            btn_style = "Accent.TButton" if "Accent.TButton" in current_themes else None
        except:
            btn_style = None
            
        ttk.Button(btn_frame, text="Aceptar", command=apply_selection, style=btn_style).pack(side=tk.RIGHT)

        ttk.Button(btn_frame, text="Cancelar", command=dlg.destroy).pack(side=tk.RIGHT, padx=10)


    def _propagate_shared_dataset(self, filtered_df=None, summary=None):
        """Informa al contenedor sobre cambios en el dataset compartido y sus filtros."""
        if not self.main_app or not hasattr(self.main_app, 'update_shared_dataset'):
            return

        metadata = {}
        if self.source_path:
            metadata['source_path'] = self.source_path
        elif self.file_path:
            metadata['source_path'] = self.file_path
        if self.source_sheet:
            metadata['sheet_name'] = self.source_sheet
        metadata['display_name'] = self._build_display_name(metadata.get('source_path'), metadata.get('sheet_name'))
        if isinstance(self.data, pd.DataFrame):
            metadata['row_count'] = len(self.data)
            metadata['column_count'] = len(self.data.columns)
        metadata['analysis_presets'] = self._copy_analysis_presets()

        if summary is None:
            summary = self.get_filter_summary()
        metadata['filter_summary'] = list(summary or [])

        # --- APPLY COLUMN SELECTION ---
        # Guard: If no data loaded yet, propagate None
        if self.data is None:
            self.main_app.update_shared_dataset(
                None,
                filtered_dataset=None,
                filter_summary=summary or [],
                source_widget=self,
                metadata=metadata
            )
            return
        
        # Calculate intersection
        valid_cols = [c for c in self.selected_columns if c in self.data.columns]
        
        print(f"[DEBUG] Propagate: Selected={len(self.selected_columns)}, Valid in Data={len(valid_cols)}, Total Data={len(self.data.columns)}")

        # Prepare DataFrame to send
        if valid_cols:
             try:
                 df_to_send = self.data[valid_cols]
                 if filtered_df is not None:
                     filtered_to_send = filtered_df[valid_cols]
                 else:
                     filtered_to_send = None
                 
                 metadata['visible_columns'] = len(valid_cols)
             except Exception as e:
                 print(f"[ERROR] Slicing failed: {e}")
                 df_to_send = pd.DataFrame() # Send empty if slice fails
                 filtered_to_send = None
        else:
             # Fallback: If no columns selected or valid, send FULL dataset for backward compatibility
             # This ensures charts still work if user hasn't configured selection yet
             df_to_send = self.data
             filtered_to_send = filtered_df
             metadata['visible_columns'] = len(self.data.columns)
             print(f"[DEBUG] No valid selection, falling back to full dataset: {len(self.data.columns)} cols")
        
        # Diagnostic check for the "0 active variables" issue
        if self.selected_columns and not valid_cols and not self.data.empty:
             print(f"[WARNING] Selection Mismatch! Selected: {self.selected_columns[:3]}... Data Cols: {list(self.data.columns)[:3]}...")
             # Optionally show concise warning to user if this happens interactively
             # messagebox.showwarning("Aviso de Selección", "Se seleccionaron variables pero no coinciden con los datos actuales.\nVerifique nombres de columna.")

        metadata['selected_columns'] = list(valid_cols) if valid_cols else []

        self._update_dataset_info_display(summary=summary, metadata=metadata, actual_valid_count=len(valid_cols))

        self.main_app.update_shared_dataset(
            df_to_send,
            filtered_dataset=filtered_to_send,
            filter_summary=summary,
            source_widget=self,
            metadata=metadata
        )

    # ... (Rest of existing methods like _enforce_sort, save_data_as can remain roughly same) ...
    
    def _on_sort_toggled(self):
        self._enforce_sort()
        self._after_dataset_changed(reapply_filters=False)

    def _enforce_sort(self):
        """Reorders the DataFrame columns based on the sort checkbox."""
        if self.data is None:
            return

        if self.sort_vars.get():
            sorted_cols = sorted(self.data.columns, key=str.lower)
            self.data = self.data[sorted_cols]
        elif hasattr(self, 'original_columns') and self.original_columns:
            # Try to restore original order
            current_cols = set(self.data.columns)
            restored = [c for c in self.original_columns if c in current_cols]
            others = [c for c in self.data.columns if c not in restored] 
            final_order = restored + others
            self.data = self.data[final_order]

    def save_data_as(self):
        if self.data is None:
            messagebox.showerror("Error", "No hay datos para guardar.", parent=self)
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            filetypes=[("Excel files", "*.xlsx",), ("CSV files", "*.csv"), ("All files", "*.*")]
        )

        if not file_path:
            return

        try:
            df_to_save = self._build_export_dataframe()

            if file_path.lower().endswith('.csv'):
                df_to_save.to_csv(file_path, index=False)
            else:
                df_to_save.to_excel(file_path, index=False)
            messagebox.showinfo("Éxito", f"Datos guardados en {file_path}", parent=self)
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo guardar el archivo:\n{e}", parent=self)

    def overwrite_data(self):
        if self.data is None:
            messagebox.showerror("Error", "No hay datos para guardar.", parent=self)
            return
            
        original_path = self.source_path
        if not original_path:
            messagebox.showerror("Error", "No se conoce la ruta del archivo original. Use 'Guardar como...'", parent=self)
            return

        display_ref = self.file_path or original_path
        if not messagebox.askyesno("Confirmar", f"¿Está seguro de que desea sobrescribir el archivo original?\n{display_ref}"):
            return

        try:
            df_to_save = self._build_export_dataframe()

            if original_path.lower().endswith('.csv'):
                df_to_save.to_csv(original_path, index=False)
            else:
                df_to_save.to_excel(original_path, index=False)
            messagebox.showinfo("Éxito", "El archivo original ha sido sobrescrito.", parent=self)
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo sobrescribir el archivo:\n{e}", parent=self)

    def _build_export_dataframe(self):
        """Devuelve el dataframe listo para exportar aplicando filtros y selección de columnas."""
        if self.data is None:
            return pd.DataFrame()

        # Prioridad: datos filtrados si existen
        base_df = self.filtered_data if isinstance(self.filtered_data, pd.DataFrame) else self.data

        # Aplicar selección de columnas explícita
        if self.selected_columns:
            cols = [c for c in self.selected_columns if c in base_df.columns]
            if cols:
                return base_df[cols].copy()

        return base_df.copy()

    def apply_filters_to_dataset(self):
        if not self.filter_component:
            return
        if self.data is None:
            messagebox.showwarning("Sin Datos", "Cargue datos antes de aplicar filtros.", parent=self)
            return

        filtered_df = self.filter_component.apply_filters()
        if filtered_df is None:
            return

        summary = self.filter_component.get_last_filters_summary()
        if summary:
            self.filtered_data = filtered_df
        else:
            self.filtered_data = None

        self._update_filter_status(summary)
        self._propagate_shared_dataset(self.filtered_data if self.filtered_data is not None else None, summary)
        self._update_filter_buttons_state()
        self._record_recent_workfile()

    def clear_filters(self):
        if not self.filter_component:
            return
        self.filter_component.clear_all_filters()
        self.filtered_data = None
        self._update_filter_status([])
        self._propagate_shared_dataset(None, [])
        self._update_filter_buttons_state()
        self._record_recent_workfile()

    def _after_dataset_changed(self, reapply_filters=False):
        self.update_variable_lists()

        if self.filter_component:
            df_for_filters = self.data if isinstance(self.data, pd.DataFrame) else pd.DataFrame()
            self.filter_component.set_dataframe(df_for_filters)

        summary = []
        filtered_df = None

        if self.data is None:
            self.filtered_data = None
        elif reapply_filters and self.filter_component and self.filter_component.filter_conditions:
            filtered_result = self.filter_component.apply_filters()
            if filtered_result is not None:
                summary = self.filter_component.get_last_filters_summary()
                if summary:
                    self.filtered_data = filtered_result
                    filtered_df = filtered_result
                else:
                    self.filtered_data = None
            else:
                self.filtered_data = None
        else:
            if self.filter_component:
                summary = self.filter_component.get_last_filters_summary()
                if summary and self.filtered_data is not None:
                    filtered_df = self.filtered_data
                else:
                    summary = []
                    self.filtered_data = None

        self._update_filter_status(summary)
        self._update_filter_buttons_state()
        self._propagate_shared_dataset(filtered_df, summary)
        self._record_recent_workfile()

    def _update_filter_buttons_state(self):
        enable = tk.NORMAL if self.data is not None and self.filter_component else tk.DISABLED
        if self.btn_apply_filters:
            self.btn_apply_filters.config(state=enable)

        if self.btn_clear_filters:
            has_filters_defined = False
            if self.filter_component:
                has_filters_defined = bool(self.filter_component.filter_conditions) or bool(self.filter_component.get_last_filters_summary())
            clear_state = tk.NORMAL if enable == tk.NORMAL and has_filters_defined else tk.DISABLED
            self.btn_clear_filters.config(state=clear_state)

    def _update_filter_status(self, summary):
        if self.data is None:
            self.filter_status_var.set("Sin datos cargados.")
            return

        total_rows = len(self.data)
        if summary and self.filtered_data is not None:
            filtered_rows = len(self.filtered_data)
            details = '; '.join(summary)
            self.filter_status_var.set(
                f"Filtros activos ({len(summary)}). Registros: {filtered_rows}/{total_rows}. Detalle: {details}"
            )
        else:
            self.filter_status_var.set(f"Sin filtros aplicados. Registros disponibles: {total_rows}.")

    def _log(self, message, level="INFO"):
        print(f"[AddVariables::{level}] {message}")

    def get_filter_summary(self):
        if self.filter_component:
            return self.filter_component.get_last_filters_summary()
        return []

    def get_current_dataset(self):
        return self.filtered_data if self.filtered_data is not None else self.data
    def create_variable_by_formula(self):
        if self.data is None:
            messagebox.showerror("Error", "No hay datos cargados para crear una variable.", parent=self)
            return

        self._open_formula_builder()

    def _open_formula_builder(self):
        """Abre un diálogo avanzado para construir fórmulas con botones y lista de variables."""
        dlg = tk.Toplevel(self)
        dlg.title("Constructor de Fórmulas")
        dlg.geometry("900x600")
        dlg.transient(self)
        dlg.grab_set()

        # --- Layout Principal ---
        # Top: Nombre de variable
        # Left: Lista de variables
        # Center: Editor de fórmula
        # Right: Botonera de operaciones
        # Bottom: Ayuda y Botones de Acción

        # 1. Nombre de la Variable
        frame_name = ttk.Frame(dlg, padding="10")
        frame_name.pack(fill=tk.X)
        
        ttk.Label(frame_name, text="Nombre de la Nueva Variable:").pack(side=tk.LEFT)
        var_name_var = tk.StringVar()
        entry_name = ttk.Entry(frame_name, textvariable=var_name_var, width=30)
        entry_name.pack(side=tk.LEFT, padx=10)
        entry_name.focus_set()

        # 2. Área Central (Split Panes o Grilla)
        frame_center = ttk.Frame(dlg, padding="10")
        frame_center.pack(fill=tk.BOTH, expand=True)
        
        frame_center.columnconfigure(1, weight=1) # El editor se expande
        frame_center.rowconfigure(0, weight=1)

        # 2A. Lista de Variables (Izquierda)
        frame_vars = ttk.LabelFrame(frame_center, text="Variables Disponibles")
        frame_vars.grid(row=0, column=0, sticky="ns", padx=(0, 10))
        
        list_vars = tk.Listbox(frame_vars, width=25)
        scroll_vars = ttk.Scrollbar(frame_vars, orient="vertical", command=list_vars.yview)
        list_vars.configure(yscrollcommand=scroll_vars.set)
        
        list_vars.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll_vars.pack(side=tk.RIGHT, fill=tk.Y)

        sorted_cols = sorted(self.data.columns, key=str.lower)
        for col in sorted_cols:
            list_vars.insert(tk.END, col)

        # 2B. Editor de Fórmula (Centro)
        frame_editor = ttk.LabelFrame(frame_center, text="Definición de Fórmula")
        frame_editor.grid(row=0, column=1, sticky="nsew", padx=(0, 10))
        
        text_formula = tk.Text(frame_editor, width=40, height=10, wrap=tk.WORD, font=("Consolas", 10))
        scroll_formula = ttk.Scrollbar(frame_editor, orient="vertical", command=text_formula.yview)
        text_formula.configure(yscrollcommand=scroll_formula.set)
        
        text_formula.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll_formula.pack(side=tk.RIGHT, fill=tk.Y)

        # 2C. Botonera (Derecha)
        frame_keypad = ttk.LabelFrame(frame_center, text="Operaciones")
        frame_keypad.grid(row=0, column=2, sticky="ns")

        # Helpers para insertar texto
        def insert_text(text):
            text_formula.insert(tk.INSERT, text)
            text_formula.focus()

        def insert_var(event=None):
            selection = list_vars.curselection()
            if selection:
                var = list_vars.get(selection[0])
                # Si tiene espacios o caracteres especiales, usar backticks (estilo pandas eval)
                if not re.match(r"^[a-zA-Z0-9_]+$", var):
                   insert_text(f"`{var}`")
                else:
                   insert_text(var)

        list_vars.bind("<Double-1>", insert_var)

        # Botones
        operations = [
            ('+', '+'), ('-', '-'), ('*', '*'), ('/', '/'),
            ('(', '('), (')', ')'), ('^', '**'), ('=', '=='),
            ('>', '>'), ('<', '<'), ('>=', '>='), ('<=', '<='),
            ('AND', '&'), ('OR', '|'), ('NOT', '~'), ('Mod', '%'),
        ]
        
        functions = [
            ('Log (e)', 'np.log('), ('Exp', 'np.exp('), 
            ('Raiz', 'np.sqrt('), ('Abs', 'np.abs('),
            ('Mean', 'np.mean('), ('Max', 'np.max('),
            ('Min', 'np.min('), ('If/Else', 'np.where(condition, true, false)')
        ]

        # Grid de Ops
        r_kp = 0
        c_kp = 0
        for label, code in operations:
            btn = ttk.Button(frame_keypad, text=label, width=5, command=lambda c=code: insert_text(c))
            btn.grid(row=r_kp, column=c_kp, padx=2, pady=2)
            c_kp += 1
            if c_kp > 3:
                c_kp = 0
                r_kp += 1

        ttk.Separator(frame_keypad, orient=tk.HORIZONTAL).grid(row=r_kp, column=0, columnspan=4, sticky="ew", pady=5)
        r_kp += 1
        
        # Lista de Funciones
        c_kp = 0
        for label, code in functions:
            btn = ttk.Button(frame_keypad, text=label, width=8, command=lambda c=code: insert_text(c))
            btn.grid(row=r_kp, column=c_kp, columnspan=2, padx=2, pady=2, sticky="ew")
            c_kp += 2
            if c_kp > 3:
                c_kp = 0
                r_kp += 1

        # 3. Bottom - Status y Ejecución
        frame_bottom = ttk.Frame(dlg, padding="10")
        frame_bottom.pack(fill=tk.X, side=tk.BOTTOM)

        lbl_help = ttk.Label(frame_bottom, text="Ayuda: Doble click en variable para insertar. Sintaxis compatible con Pandas eval/NumPy.", foreground="gray")
        lbl_help.pack(side=tk.TOP, anchor="w", pady=(0, 10))

        def verify_formula():
            formula = text_formula.get("1.0", tk.END).strip()
            if not formula:
                return
            try:
                env = {'np': np, 'math': __import__('math'), 'pd': pd}
                # Probar con las primeras 5 filas para rapidez
                sample_data = self.data.head(5).copy()
                sample_data.eval(formula, engine='python', local_dict=env)
                messagebox.showinfo("Verificación", "La fórmula parece válida (probada en 5 filas).", parent=dlg)
            except Exception as e:
                messagebox.showerror("Error de Sintaxis", f"Error: {e}", parent=dlg)

        def apply_creation():
            name = var_name_var.get().strip()
            formula = text_formula.get("1.0", tk.END).strip()

            if not name:
                messagebox.showerror("Falta Nombre", "Ingrese un nombre para la variable.", parent=dlg)
                return
            if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", name):
                messagebox.showerror("Nombre Inválido", "El nombre debe ser un identificador válido (a-z, 0-9, _).", parent=dlg)
                return
            if name in self.data.columns:
                messagebox.showerror("Duplicado", f"La variable '{name}' ya existe.", parent=dlg)
                return
            if not formula:
                messagebox.showerror("Falta Fórmula", "Ingrese una fórmula.", parent=dlg)
                return

            try:
                env = {'np': np, 'math': __import__('math'), 'pd': pd}
                
                # Evaluar en todo el dataset
                result_series = self.data.eval(formula, engine='python', local_dict=env)
                
                # Verificaciones simples
                if not hasattr(result_series, 'values'):
                     # Intento de scalar broadcast?
                     result_series = pd.Series(result_series, index=self.data.index)

                if len(result_series) != len(self.data):
                     raise ValueError(f"Longitud incorrecta: {len(result_series)} vs {len(self.data)}")

                # Guardar
                self.data[name] = result_series.values
                if hasattr(self, 'original_columns'):
                    self.original_columns.append(name)
                
                # Auto-add to selection if selection is active (not empty)
                # If empty, it means "All selected" usually, but we have logic that empty = empty.
                # But wait, if empty, our logic propagates NOTHING? 
                # Actually, our propagation logic says "if not selected_cols and ...: selected_cols = list(all)"
                # So if it was truly empty (user deselect all), adding one makes it length 1.
                # If it was "implicitly all" (empty list handled as all in logic?), adding one might make it explicit?
                # Let's just append it if the user has a specific selection list going on.
                # Safest is to append.
                if hasattr(self, 'selected_columns'):
                    if name not in self.selected_columns:
                        self.selected_columns.append(name)
                
                self._enforce_sort()
                self._after_dataset_changed(reapply_filters=True)
                
                messagebox.showinfo("Éxito", f"Variable '{name}' creada.\n\nRecuerde guardar el archivo (Botón 'Guardar como...' o 'Sustituir') para conservar los cambios.", parent=dlg)
                dlg.destroy()

            except Exception as e:
                messagebox.showerror("Error al Crear", f"Error en la fórmula:\n{e}", parent=dlg)

        btn_verify = ttk.Button(frame_bottom, text="Verificar Sintaxis", command=verify_formula)
        btn_verify.pack(side=tk.LEFT, padx=5)

        btn_cancel = ttk.Button(frame_bottom, text="Cancelar", command=dlg.destroy)
        btn_cancel.pack(side=tk.RIGHT, padx=5)

        btn_ok = ttk.Button(frame_bottom, text="Crear Variable", command=apply_creation)
        btn_ok.pack(side=tk.RIGHT, padx=5)



    def _update_dataset_info_display(self, summary=None, metadata=None, actual_valid_count=None):
        if not hasattr(self, "dataset_info_var"):
            return
            
        if actual_valid_count is None:
            # Try to infer
            actual_valid_count = len([c for c in self.selected_columns if c in self.data.columns]) if self.data is not None else 0

        if summary is None:
            summary = self.get_filter_summary()

        if self.data is None or not isinstance(self.data, pd.DataFrame):
            self.dataset_info_var.set("Archivo actual: Sin archivo cargado.")
            return

        metadata_path = metadata.get('source_path') if metadata else None
        metadata_sheet = metadata.get('sheet_name') if metadata else None
        base_name = self._build_display_name(self.source_path or metadata_path or self.file_path, self.source_sheet or metadata_sheet)
        total_rows = len(self.data)

        if self.filtered_data is not None and isinstance(self.filtered_data, pd.DataFrame):
            visible_rows = len(self.filtered_data)
            info = f"Archivo actual: {base_name} • Filas: {visible_rows}/{total_rows} • Variables Activas: {actual_valid_count}/{len(self.data.columns)}"
        else:
            info = f"Archivo actual: {base_name} • Filas: {total_rows} • Variables Activas: {actual_valid_count}/{len(self.data.columns)}"
        
        self.dataset_info_var.set(info)
        if summary:
            info += f" • Filtros activos: {len(summary)}"

        self.dataset_info_var.set(info)

    def _read_dataset_from_path(self, filepath, prompt_sheet=False, preferred_sheet=None):
        if not filepath:
            return None

        if filepath.lower().endswith('.csv'):
            df = pd.read_csv(filepath)
            return df, filepath, filepath, None

        excel_file = pd.ExcelFile(filepath)
        sheet_names = excel_file.sheet_names
        sheet_name = preferred_sheet if preferred_sheet in sheet_names else None

        if prompt_sheet and len(sheet_names) > 1 and sheet_name is None:
            sheet_name = self._ask_sheet_name(sheet_names)
            if sheet_name is None:
                return None

        if sheet_name is None:
            sheet_name = sheet_names[0]

        df = excel_file.parse(sheet_name)
        display_path = f"{filepath} [{sheet_name}]"
        return df, display_path, filepath, sheet_name

    def _finalize_dataset_load(self, dataframe, display_path, source_path, sheet_name, recent_entry=None):
        if dataframe is None:
            return

        self.data = dataframe
        self.file_path = display_path
        self.source_path = source_path
        self.source_sheet = sheet_name
        self.original_columns = list(dataframe.columns)

        saved_selection = None
        saved_filters = None
        saved_presets = None
        if recent_entry:
            saved_selection = recent_entry.get('selected_columns')
            saved_filters = recent_entry.get('filters')
            saved_presets = recent_entry.get('analysis_presets')

        if saved_selection:
            selection = [col for col in saved_selection if col in dataframe.columns]
            self.selected_columns = selection if selection else list(dataframe.columns)
        else:
            self.selected_columns = list(dataframe.columns)

        self.analysis_presets = copy.deepcopy(saved_presets) if isinstance(saved_presets, dict) else {}
        self.filtered_data = None

        if self.filter_component:
            self.filter_component.set_dataframe(dataframe)
            if saved_filters:
                self.filter_component.load_conditions(saved_filters)
            else:
                self.filter_component.clear_all_filters()

        self._enforce_sort()
        reapply_filters = bool(saved_filters and self.filter_component)
        self._after_dataset_changed(reapply_filters=reapply_filters)

    def get_analysis_presets(self):
        return self._copy_analysis_presets()

    def get_analysis_preset(self, preset_key, default=None):
        if preset_key is None:
            return copy.deepcopy(default)
        return copy.deepcopy(self.analysis_presets.get(preset_key, default))

    def update_analysis_preset(self, preset_key, preset_payload):
        if preset_key is None:
            return

        if preset_payload is None:
            if preset_key in self.analysis_presets:
                del self.analysis_presets[preset_key]
        else:
            self.analysis_presets[preset_key] = copy.deepcopy(preset_payload)

        if self.data is None:
            return

        self._notify_metadata_change()
        self._record_recent_workfile()

    def replace_analysis_presets(self, presets_mapping):
        cleaned = presets_mapping or {}
        self.analysis_presets = copy.deepcopy(cleaned)
        if self.data is None:
            return

        self._notify_metadata_change()
        self._record_recent_workfile()

    def _copy_analysis_presets(self):
        return copy.deepcopy(self.analysis_presets or {})

    def _notify_metadata_change(self):
        filtered_df = self.filtered_data if isinstance(self.filtered_data, pd.DataFrame) else None
        summary = self.get_filter_summary()
        self._propagate_shared_dataset(filtered_df, summary)

    def _record_recent_workfile(self):
        if not self.source_path or self.data is None:
            return

        try:
            filters_state = self.filter_component.export_conditions() if self.filter_component else []
        except Exception as exc:
            self._log(f"No se pudo serializar filtros: {exc}", level="WARN")
            filters_state = []

        summary = self.get_filter_summary() or []
        selected_cols = [col for col in self.selected_columns if self.data is not None and col in self.data.columns]
        entry = {
            "path": self.source_path,
            "sheet_name": self.source_sheet,
            "display_name": self._build_display_name(self.source_path, self.source_sheet),
            "loaded_at": datetime.utcnow().isoformat(),
            "selected_columns": selected_cols,
            "filters": self._json_safe(filters_state),
            "filter_summary": self._json_safe(summary),
            "analysis_presets": self._json_safe(self.analysis_presets or {}),
            "row_count": len(self.data) if isinstance(self.data, pd.DataFrame) else 0,
            "column_count": len(self.data.columns) if isinstance(self.data, pd.DataFrame) else 0
        }

        self._recent_entries = [e for e in self._recent_entries if not self._same_recent_entry(e, entry['path'], entry['sheet_name'])]
        self._recent_entries.insert(0, entry)
        self._recent_entries = self._recent_entries[:RECENT_WORKFILES_LIMIT]
        self._save_recent_entries(self._recent_entries)
        self._refresh_recent_combo()

    def _handle_recent_selection(self, _event=None):
        label = self.recent_combo_var.get()
        if not label:
            return

        entry = self._recent_label_map.get(label)
        self.recent_combo_var.set("")
        if self.recent_combo:
            self.recent_combo.set("")
        if not entry:
            return

        self._load_recent_entry(entry)

    def _load_recent_entry(self, entry):
        path = entry.get('path')
        sheet_name = entry.get('sheet_name')
        if not path or not os.path.exists(path):
            messagebox.showwarning("Archivo no disponible", "No se encontró el archivo asociado. Se eliminará de la lista de recientes.", parent=self)
            self._remove_recent_entry(entry)
            return

        try:
            read_result = self._read_dataset_from_path(path, prompt_sheet=False, preferred_sheet=sheet_name)
            if read_result is None:
                return
            df, display_path, source_path, resolved_sheet = read_result
            self._finalize_dataset_load(df, display_path, source_path, resolved_sheet, recent_entry=entry)
            friendly_name = entry.get('display_name') or self._build_display_name(path, resolved_sheet)
            messagebox.showinfo("Reciente", f"Se restauró '{friendly_name}'.", parent=self)
        except Exception as exc:
            messagebox.showerror("Error", f"No se pudo cargar el archivo reciente:\n{exc}", parent=self)
            self._remove_recent_entry(entry)

    def _refresh_recent_combo(self):
        if not self.recent_combo:
            return

        labels = []
        self._recent_label_map = {}
        for entry in self._recent_entries:
            label = self._format_recent_label(entry)
            labels.append(label)
            self._recent_label_map[label] = entry

        self.recent_combo['values'] = labels
        if labels:
            self.recent_combo.configure(state="readonly")
            self.recent_combo_var.set("")
            self.recent_combo.set("")
        else:
            self.recent_combo.configure(state="disabled")
            self.recent_combo.set("Sin historial")
            self.recent_combo_var.set("Sin historial")

    def _format_recent_label(self, entry):
        display_name = entry.get('display_name') or self._build_display_name(entry.get('path'), entry.get('sheet_name'))
        timestamp = entry.get('loaded_at')
        if timestamp:
            friendly = timestamp.replace('T', ' ').split('.')[0]
            return f"{display_name} — {friendly}"
        return display_name

    def _remove_recent_entry(self, entry):
        path = entry.get('path')
        sheet_name = entry.get('sheet_name')
        self._recent_entries = [e for e in self._recent_entries if not self._same_recent_entry(e, path, sheet_name)]
        self._save_recent_entries(self._recent_entries)
        self._refresh_recent_combo()

    def _load_recent_entries(self):
        if not os.path.exists(RECENT_WORKFILES_PATH):
            return []
        try:
            with open(RECENT_WORKFILES_PATH, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data if isinstance(data, list) else []
        except Exception as exc:
            self._log(f"No se pudieron leer archivos recientes: {exc}", level="WARN")
            return []

    def _save_recent_entries(self, entries):
        try:
            os.makedirs(os.path.dirname(RECENT_WORKFILES_PATH), exist_ok=True)
            with open(RECENT_WORKFILES_PATH, 'w', encoding='utf-8') as f:
                json.dump(entries, f, ensure_ascii=False, indent=2)
        except Exception as exc:
            self._log(f"No se pudieron guardar archivos recientes: {exc}", level="WARN")

    def _same_recent_entry(self, entry, path, sheet_name):
        return entry.get('path') == path and (entry.get('sheet_name') or None) == (sheet_name or None)

    def _json_safe(self, value):
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, (datetime, pd.Timestamp)):
            return value.isoformat()
        if isinstance(value, dict):
            return {str(k): self._json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [self._json_safe(v) for v in value]
        return str(value)

    def _build_display_name(self, path, sheet_name):
        base = os.path.basename(path) if path else "Datos en memoria"
        if sheet_name:
            if f"[{sheet_name}]" not in base:
                return f"{base} [{sheet_name}]"
        return base
