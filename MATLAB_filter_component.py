#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np

class FilterComponent(ttk.Frame):
    """
    Componente reutilizable para aplicar filtros multifuncionales a un DataFrame.
    Permite añadir/eliminar condiciones de filtro dinámicamente.
    Muestra controles adecuados (checkboxes, rangos) según el tipo de columna.
    """
    def __init__(self, master, max_unique_cat=50, log_callback=None, *args, **kwargs):
        """
        Inicializa el componente de filtro.
        :param master: Widget padre.
        :param max_unique_cat: Umbral de valores únicos para tratar una columna como categórica.
        :param log_callback: Función para registrar mensajes.
        """
        # Extraer log_callback de kwargs si se pasó así, o tomar el parámetro nombrado
        self.log_func = kwargs.pop('log_callback', log_callback if log_callback else print)
        
        super().__init__(master, *args, **kwargs)
        self.df_original = None
        self.column_list = []
        self.filter_conditions = [] # Lista para almacenar widgets de cada condición
        self.max_unique_cat = max_unique_cat

        # --- UI Principal ---
        self.main_frame = ttk.Frame(self)
        self.main_frame.pack(fill="both", expand=True)

        # Botón para añadir nueva condición
        self.add_button = ttk.Button(self.main_frame, text="+ Añadir Filtro", command=self._add_filter_condition_row)
        self.add_button.pack(pady=5, anchor="w", padx=5)

        # Frame contenedor para las filas de filtros (con posible scroll futuro)
        self.conditions_frame = ttk.Frame(self.main_frame)
        self.conditions_frame.pack(fill="both", expand=True, padx=5, pady=5)

        # Inicialmente añadir una fila vacía si se desea
        # self._add_filter_condition_row()

    def set_dataframe(self, df):
        """
        Establece o actualiza el DataFrame base para filtrar.
        Actualiza la lista de columnas disponibles.
        :param df: pandas.DataFrame
        """
        if df is None or not isinstance(df, pd.DataFrame):
            self.df_original = None
            self.column_list = []
            self.log("DataFrame no válido o nulo establecido.", "WARN")
        else:
            self.df_original = df.copy() # Trabajar con una copia
            self.column_list = [""] + sorted(self.df_original.columns.tolist()) # Añadir opción vacía
            self.log(f"DataFrame establecido ({self.df_original.shape}). Columnas: {len(self.column_list)-1}", "INFO")

        # Actualizar comboboxes existentes y limpiar valores
        self._update_all_column_comboboxes()
        # Podríamos optar por limpiar todos los filtros al cargar nuevo DF
        # self._clear_all_conditions()
        # self._add_filter_condition_row() # Añadir una fila inicial

    def _add_filter_condition_row(self):
        """Añade una nueva fila de widgets para definir una condición de filtro."""
        condition_row_frame = ttk.Frame(self.conditions_frame)
        condition_row_frame.pack(fill="x", pady=2)

        # Combobox para seleccionar columna
        col_combo = ttk.Combobox(condition_row_frame, values=self.column_list, state="readonly", width=20)
        col_combo.pack(side="left", padx=2)
        col_combo.set("") # Iniciar vacío

        # Frame para los controles específicos del tipo de columna (se llenará dinámicamente)
        controls_frame = ttk.Frame(condition_row_frame)
        controls_frame.pack(side="left", padx=2, fill="x", expand=True)

        # Botón para eliminar esta fila
        remove_button = ttk.Button(condition_row_frame, text="-", width=3,
                                   command=lambda frame=condition_row_frame: self._remove_filter_condition_row(frame))
        remove_button.pack(side="right", padx=2)

        # Guardar referencia a los widgets de esta fila
        condition_widgets = {
            "frame": condition_row_frame,
            "col_combo": col_combo,
            "controls_frame": controls_frame,
            "specific_controls": None # Se llenará al seleccionar columna
        }
        self.filter_conditions.append(condition_widgets)

        # Asociar evento al combobox de columna
        col_combo.bind("<<ComboboxSelected>>", lambda event, cw=condition_widgets: self._on_column_selected(event, cw))

    def _remove_filter_condition_row(self, frame_to_remove):
        """Elimina una fila de condición de filtro."""
        widgets_to_remove = None
        for widgets in self.filter_conditions:
            if widgets["frame"] == frame_to_remove:
                widgets_to_remove = widgets
                break

        if widgets_to_remove:
            widgets_to_remove["frame"].destroy()
            self.filter_conditions.remove(widgets_to_remove)
            self.log("Fila de filtro eliminada.", "DEBUG")

    def _on_column_selected(self, event, condition_widgets):
        """Se ejecuta cuando se selecciona una columna en un Combobox."""
        selected_col = condition_widgets["col_combo"].get()
        controls_frame = condition_widgets["controls_frame"]

        # Limpiar controles anteriores
        for widget in controls_frame.winfo_children():
            widget.destroy()
        condition_widgets["specific_controls"] = None

        if not selected_col or self.df_original is None:
            return

        col_data = self.df_original[selected_col].dropna()
        col_dtype = self.df_original[selected_col].dtype

        # Determinar si es categórica, numérica o fecha
        is_categorical = False
        if pd.api.types.is_datetime64_any_dtype(col_dtype):
            self._create_date_controls(controls_frame, condition_widgets, selected_col)
        elif pd.api.types.is_object_dtype(col_dtype) or pd.api.types.is_categorical_dtype(col_dtype):
            # Si es object o category, verificar si tratar como categórica (pocos únicos) o texto libre
            unique_count = col_data.nunique()
            if unique_count <= self.max_unique_cat:
                is_categorical = True
            else:
                # Tratar como texto libre si hay muchos valores únicos
                self._create_text_controls(controls_frame, condition_widgets, selected_col)
                is_categorical = False # Asegurar que no caiga en el bloque categórico
        elif pd.api.types.is_numeric_dtype(col_dtype):
            unique_count = col_data.nunique()
            # No tratar booleanos como categóricos aquí si tienen pocos valores, ni numéricos que ya se decidieron como no categóricos
            if unique_count <= self.max_unique_cat and not pd.api.types.is_bool_dtype(col_dtype):
                is_categorical = True # Tratar como categórica si pocos valores únicos
            else:
                is_categorical = False # Definitivamente numérica
        
        if is_categorical: # Solo si se determinó explícitamente como categórica
            self._create_categorical_controls(controls_frame, condition_widgets, selected_col, col_data)
        elif pd.api.types.is_numeric_dtype(col_dtype) and \
             not pd.api.types.is_datetime64_any_dtype(col_dtype) and \
             not is_categorical: # Asegurar que no sea fecha y no se haya marcado como categórica
             self._create_numeric_controls(controls_frame, condition_widgets, selected_col)
        # else: Añadir manejo para otros tipos (booleano explícito si se desea)

    def _create_categorical_controls(self, parent_frame, condition_widgets, col_name, col_data):
        """Crea controles para filtrar una columna categórica (Listbox con búsqueda)."""
        all_unique_vals = sorted([str(v) for v in col_data.unique()]) # Guardar todos los valores originales
        if not all_unique_vals: return

        # Frame contenedor para búsqueda y listbox
        outer_frame = ttk.Frame(parent_frame)
        outer_frame.pack(side="left", fill="x", expand=True, padx=2)

        search_entry = ttk.Entry(outer_frame, width=20)
        search_entry.pack(fill="x", padx=2, pady=(0,2))
        
        listbox_frame = ttk.Frame(outer_frame) # Nuevo frame para listbox y scrollbar
        listbox_frame.pack(fill="both", expand=True)

        listbox = tk.Listbox(listbox_frame, selectmode=tk.MULTIPLE, height=min(5, len(all_unique_vals)), exportselection=False)
        scrollbar = ttk.Scrollbar(listbox_frame, orient="vertical", command=listbox.yview)
        listbox.configure(yscrollcommand=scrollbar.set)

        scrollbar.pack(side="right", fill="y")
        listbox.pack(side="left", fill="both", expand=True)

        def populate_listbox(values_to_show):
            listbox.delete(0, tk.END)
            for val in values_to_show:
                listbox.insert(tk.END, val)
            # Ajustar altura dinámicamente si se desea, o mantenerla fija
            listbox.config(height=min(5, len(values_to_show) if values_to_show else 1))


        populate_listbox(all_unique_vals) # Poblar inicialmente con todos

        def on_search_change(event):
            search_term = search_entry.get().lower()
            if not search_term:
                filtered_vals = all_unique_vals
            else:
                filtered_vals = [val for val in all_unique_vals if search_term in val.lower()]
            populate_listbox(filtered_vals)

        search_entry.bind("<KeyRelease>", on_search_change)
        # También podría ser útil un pequeño botón "Limpiar búsqueda"
        # search_entry.bind("<FocusOut>", lambda e: on_search_change(None) if not search_entry.get() else None)


        condition_widgets["specific_controls"] = {"type": "categorical", "listbox": listbox, "search_entry": search_entry, "_all_unique_vals": all_unique_vals}
        self.log(f"Controles categóricos con búsqueda creados para '{col_name}'.", "DEBUG")

    def _create_numeric_controls(self, parent_frame, condition_widgets, col_name):
        """Crea controles para filtrar una columna numérica (rango Desde/Hasta)."""
        ttk.Label(parent_frame, text="Desde:").pack(side="left", padx=2)
        entry_from = ttk.Entry(parent_frame, width=10)
        entry_from.pack(side="left", padx=2)

        ttk.Label(parent_frame, text="Hasta:").pack(side="left", padx=2)
        entry_to = ttk.Entry(parent_frame, width=10)
        entry_to.pack(side="left", padx=2)

        condition_widgets["specific_controls"] = {"type": "numeric", "from": entry_from, "to": entry_to}
        self.log(f"Controles numéricos creados para '{col_name}'.", "DEBUG")

    def _create_date_controls(self, parent_frame, condition_widgets, col_name):
        """Crea controles para filtrar una columna de fecha (rango Desde/Hasta)."""
        ttk.Label(parent_frame, text="Fecha Desde (YYYY-MM-DD):").pack(side="left", padx=2)
        entry_from = ttk.Entry(parent_frame, width=15)
        entry_from.pack(side="left", padx=2)

        ttk.Label(parent_frame, text="Fecha Hasta (YYYY-MM-DD):").pack(side="left", padx=2)
        entry_to = ttk.Entry(parent_frame, width=15)
        entry_to.pack(side="left", padx=2)

        condition_widgets["specific_controls"] = {"type": "date", "from": entry_from, "to": entry_to}
        self.log(f"Controles de fecha creados para '{col_name}'.", "DEBUG")

    def _create_text_controls(self, parent_frame, condition_widgets, col_name):
        """Crea controles para filtrar una columna de texto con opciones avanzadas."""
        operations = ["contiene", "no contiene", "empieza con", "termina con", "es exactamente", "es vacío", "no es vacío", "regex"]
        
        ttk.Label(parent_frame, text="Operación:").pack(side="left", padx=2)
        op_combo = ttk.Combobox(parent_frame, values=operations, state="readonly", width=15)
        op_combo.pack(side="left", padx=2)
        op_combo.set("contiene")

        ttk.Label(parent_frame, text="Valor:").pack(side="left", padx=2)
        entry_value = ttk.Entry(parent_frame, width=20)
        entry_value.pack(side="left", padx=2, fill="x", expand=True)
        
        # Deshabilitar entry_value si la operación es "es vacío" o "no es vacío"
        def on_op_selected(event):
            selected_op = op_combo.get()
            if selected_op in ["es vacío", "no es vacío"]:
                entry_value.delete(0, tk.END)
                entry_value.config(state="disabled")
            else:
                entry_value.config(state="normal")
        op_combo.bind("<<ComboboxSelected>>", on_op_selected)

        condition_widgets["specific_controls"] = {
            "type": "text",
            "op_combo": op_combo,
            "value_entry": entry_value
        }
        self.log(f"Controles de texto creados para '{col_name}'.", "DEBUG")

    def _update_all_column_comboboxes(self):
        """Actualiza las opciones en todos los comboboxes de columna."""
        for widgets in self.filter_conditions:
            current_val = widgets["col_combo"].get()
            widgets["col_combo"]["values"] = self.column_list
            if current_val in self.column_list:
                widgets["col_combo"].set(current_val)
            else:
                widgets["col_combo"].set("")
                # Limpiar controles específicos si la columna ya no existe
                for widget in widgets["controls_frame"].winfo_children():
                    widget.destroy()
                widgets["specific_controls"] = None

    def _clear_all_conditions(self):
        """Elimina todas las filas de condiciones de filtro."""
        # Iterar en reversa para evitar problemas al eliminar
        for widgets in reversed(self.filter_conditions):
            widgets["frame"].destroy()
        self.filter_conditions = []
        self.log("Todas las condiciones de filtro eliminadas.", "DEBUG")


    def apply_filters(self):
        """
        Construye y aplica los filtros definidos en la UI al DataFrame original.
        :return: pandas.DataFrame filtrado o None si hay error o no hay datos.
        """
        if self.df_original is None:
            self.log("No hay DataFrame original para filtrar.", "WARN")
            return None

        df_filtered = self.df_original.copy()
        active_filters_summary = []

        for widgets in self.filter_conditions:
            col_name = widgets["col_combo"].get()
            controls = widgets["specific_controls"]

            if not col_name or not controls:
                continue # Saltar filas sin columna seleccionada o sin controles

            control_type = controls.get("type")

            try:
                if control_type == "categorical":
                    listbox = controls["listbox"]
                    selected_indices = listbox.curselection()
                    if not selected_indices: continue # Saltar si no hay selección

                    selected_values = [listbox.get(i) for i in selected_indices]
                    # Filtrar: asegurar que la columna se trate como string para la comparación
                    df_filtered = df_filtered[df_filtered[col_name].astype(str).isin(selected_values)]
                    active_filters_summary.append(f"{col_name} IN ({', '.join(selected_values)})")

                elif control_type == "numeric":
                    val_from_str = controls["from"].get().strip()
                    val_to_str = controls["to"].get().strip()

                    # Intentar convertir a numérico, ignorar si está vacío
                    val_from = pd.to_numeric(val_from_str, errors='coerce')
                    val_to = pd.to_numeric(val_to_str, errors='coerce')

                    # Aplicar filtro solo si al menos uno es un número válido
                    conditions = []
                    summary_parts = []
                    if pd.notna(val_from):
                        conditions.append(df_filtered[col_name] >= val_from)
                        summary_parts.append(f">= {val_from}")
                    if pd.notna(val_to):
                        conditions.append(df_filtered[col_name] <= val_to)
                        summary_parts.append(f"<= {val_to}")

                    if conditions:
                        combined_condition = conditions[0]
                        for cond in conditions[1:]:
                            combined_condition &= cond
                        df_filtered = df_filtered[combined_condition]
                        active_filters_summary.append(f"{col_name} ({' y '.join(summary_parts)})")

                elif control_type == "date":
                    val_from_str = controls["from"].get().strip()
                    val_to_str = controls["to"].get().strip()

                    # Convertir la columna del DataFrame a datetime ANTES de filtrar
                    # Esto es crucial para que la comparación de fechas funcione correctamente.
                    # Se hace una copia para no modificar el df_original subyacente en df_filtered[col_name]
                    original_column_as_datetime = pd.to_datetime(df_filtered[col_name], errors='coerce')

                    val_from = pd.to_datetime(val_from_str, errors='coerce') if val_from_str else pd.NaT
                    val_to = pd.to_datetime(val_to_str, errors='coerce') if val_to_str else pd.NaT
                    
                    conditions = []
                    summary_parts = []

                    if pd.notna(val_from):
                        conditions.append(original_column_as_datetime >= val_from)
                        summary_parts.append(f">= {val_from_str}")
                    if pd.notna(val_to):
                        # Para que el rango "hasta" sea inclusivo, se puede ajustar al final del día si no se proporciona hora
                        # O simplemente usar <= que para fechas sin hora funciona bien.
                        conditions.append(original_column_as_datetime <= val_to)
                        summary_parts.append(f"<= {val_to_str}")
                    
                    if conditions:
                        combined_condition = conditions[0]
                        for cond in conditions[1:]:
                            combined_condition &= cond
                        df_filtered = df_filtered[combined_condition]
                        active_filters_summary.append(f"{col_name} ({' y '.join(summary_parts)})")

                elif control_type == "text":
                    operator = controls["op_combo"].get()
                    value_str = controls["value_entry"].get() # No strip() aquí, podría ser intencional

                    # Asegurar que la columna se trate como string para operaciones de string
                    # Usar .astype(str) maneja NaNs convirtiéndolos a 'nan' string.
                    col_as_str = df_filtered[col_name].astype(str)

                    condition = None
                    summary_op = operator
                    summary_val = value_str

                    if operator == "contiene":
                        condition = col_as_str.str.contains(value_str, case=False, na=False, regex=False)
                    elif operator == "no contiene":
                        condition = ~col_as_str.str.contains(value_str, case=False, na=False, regex=False)
                    elif operator == "empieza con":
                        condition = col_as_str.str.startswith(value_str, na=False)
                    elif operator == "termina con":
                        condition = col_as_str.str.endswith(value_str, na=False)
                    elif operator == "es exactamente":
                        condition = (col_as_str == value_str)
                    elif operator == "es vacío":
                        # Considera NaN, None, y strings vacíos como "vacío"
                        condition = df_filtered[col_name].isna() | (col_as_str == '')
                        summary_val = "" # No hay valor para el sumario
                    elif operator == "no es vacío":
                        condition = df_filtered[col_name].notna() & (col_as_str != '')
                        summary_val = "" # No hay valor para el sumario
                    elif operator == "regex":
                        try:
                            condition = col_as_str.str.contains(value_str, case=True, na=False, regex=True) # Case sensitive para regex por defecto
                        except Exception as regex_err:
                            self.log(f"Error de Regex en '{col_name}' con patrón '{value_str}': {regex_err}", "ERROR")
                            messagebox.showerror("Error de Regex", f"Patrón de Regex inválido para '{col_name}':\n{value_str}\n\n{regex_err}")
                            return None # Detener si el regex es inválido
                    
                    if condition is not None:
                        df_filtered = df_filtered[condition]
                        active_filters_summary.append(f"{col_name} {summary_op} '{summary_val}'")
                
                # Añadir lógica para otros tipos de control aquí

            except Exception as e:
                self.log(f"Error aplicando filtro para '{col_name}': {e}", "ERROR")
                messagebox.showerror("Error de Filtro", f"Error al aplicar filtro en columna '{col_name}':\n{e}")
                return None # Detener si un filtro falla

            if df_filtered.empty:
                self.log("DataFrame vacío después de aplicar filtros.", "WARN")
                break # No tiene sentido seguir filtrando

        if active_filters_summary:
            self.log(f"Filtros aplicados: {'; '.join(active_filters_summary)}", "INFO")
        else:
            self.log("No se aplicaron filtros activos.", "INFO")

        return df_filtered

    def log(self, message, level="INFO"):
        """Placeholder para logging. Integrar con sistema de log principal si existe."""
        print(f"[{level}] FilterComponent: {message}")

# --- Ejemplo de uso (si se ejecuta este archivo directamente) ---
if __name__ == '__main__':
    root = tk.Tk()
    root.title("Test FilterComponent")
    root.geometry("600x400")

    # Crear datos de ejemplo
    data = {
        'ID': range(1, 101),
        'Categoria': np.random.choice(['A', 'B', 'C', 'D'], 100),
        'ValorNum': np.random.rand(100) * 100,
        'Grupo': np.random.choice([10, 20, 30], 100),
        'Fecha': pd.to_datetime(np.random.randint(1640995200, 1704067200, size=100), unit='s') # Fechas aleatorias
    }
    sample_df = pd.DataFrame(data)
    sample_df.loc[::10, 'ValorNum'] = np.nan # Introducir algunos NaNs

    # Frame principal para la demo
    main_frame = ttk.Frame(root, padding="10")
    main_frame.pack(fill="both", expand=True)

    # Instanciar el componente de filtro
    filter_comp = FilterComponent(main_frame, max_unique_cat=10)
    filter_comp.pack(fill="x", pady=(0, 10))

    # Botón para cargar datos en el componente
    def load_sample_data():
        filter_comp.set_dataframe(sample_df)

    load_button = ttk.Button(main_frame, text="Cargar Datos de Ejemplo", command=load_sample_data)
    load_button.pack(pady=5)

    # Área de texto para mostrar resultados
    results_text = tk.Text(main_frame, height=10, wrap="none")
    results_text.pack(fill="both", expand=True, pady=(5, 0))

    # Botón para aplicar filtros y mostrar resultado
    def show_filtered_data():
        filtered_df = filter_comp.apply_filters()
        results_text.delete("1.0", tk.END)
        if filtered_df is not None:
            results_text.insert(tk.END, f"DataFrame Filtrado ({filtered_df.shape}):\n\n")
            results_text.insert(tk.END, filtered_df.to_string())
        else:
            results_text.insert(tk.END, "Error al filtrar o DataFrame original no cargado.")

    apply_button = ttk.Button(main_frame, text="Aplicar Filtros y Mostrar", command=show_filtered_data)
    apply_button.pack(pady=5)


    root.mainloop()