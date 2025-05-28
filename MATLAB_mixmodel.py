#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interfaz para Modelado Mixto, adaptado desde un script original en PySimpleGUI.
Esta versión utiliza tkinter para integrarse como una pestaña en la aplicación principal.
Se conservan todas las funciones y la lógica original, con mejoras para permitir elegir
diferentes "tipos" de modelo mixto y se incorpora un scrollbar vertical para ver todo el contenido.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
import os
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import traceback

# FilterComponent ha sido eliminado.
FilterComponent = None # Mantener para evitar errores si alguna lógica residual lo verifica.

# -----------------------
# Funciones de utilidad
# -----------------------

def leer_archivo(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    try:
        if ext == ".csv":
            df = pd.read_csv(file_path, sep=",")
        elif ext in [".xls", ".xlsx"]:
            df = pd.read_excel(file_path)
        else:
            messagebox.showerror("Error", "Tipo de archivo no soportado. Usa CSV o Excel.")
            return None
        return df
    except Exception as e:
        messagebox.showerror("Error", f"Error al leer el archivo:\n{e}")
        return None

def recodificar_patron(df, group_var, hora_var, threshold=3):
    resumen = df.groupby(group_var)[hora_var].max().reset_index()
    resumen.rename(columns={hora_var: "TiempoTotal"}, inplace=True)
    resumen["mas3menos"] = np.where(resumen["TiempoTotal"] < threshold, 1, 2)
    df = df.merge(resumen[[group_var, "mas3menos"]], on=group_var, how="left")
    return df

def crear_variables_combinadas(df, formulas_text):
    if formulas_text.strip() == "":
        return df
    for linea in formulas_text.splitlines():
        if "=" in linea:
            nuevo, expresion = linea.split("=", 1)
            nuevo = nuevo.strip()
            expresion = expresion.strip()
            try:
                df[nuevo] = df.eval(expresion)
            except Exception as e:
                messagebox.showerror("Error", f"Error creando {nuevo} con '{expresion}': {e}")
    return df

def construir_formula(dep_var, time_var, incluir_mas3):
    if incluir_mas3:
        formula = f"{dep_var} ~ {time_var} + mas3menos + {time_var}:mas3menos"
    else:
        formula = f"{dep_var} ~ {time_var}"
    return formula

def ajustar_modelo(formula, data, group_var, re_formula="1", vc_formula=None):
    try:
        md = smf.mixedlm(formula, data, groups=data[group_var], re_formula=re_formula, vc_formula=vc_formula)
        mdf = md.fit(reml=True)
        return mdf
    except Exception as e:
        messagebox.showerror("Error", f"Error al ajustar el modelo:\n{e}")
        traceback.print_exc()
        return None

# -----------------------
# Clase ScrollableFrame para envolver el contenido
# -----------------------

class ScrollableFrame(ttk.Frame):
    def __init__(self, container, *args, **kwargs):
        super().__init__(container, *args, **kwargs)
        canvas = tk.Canvas(self)
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        self.scrollable_frame = ttk.Frame(canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

# -----------------------
# Clase MixModelTab: la interfaz convertida a Tkinter con scrollbar
# -----------------------

class MixModelTab(ttk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.df = None
        self.fixed_list = []
        self.random_list = []

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
        
        self.create_widgets()

    def create_widgets(self):
        # Se crea un contenedor ScrollableFrame para incluir todo el contenido con scrollbar vertical
        scroll_frame = ScrollableFrame(self)
        scroll_frame.pack(fill="both", expand=True)
        container = scroll_frame.scrollable_frame

        # Configurar grid del contenedor
        container.grid_columnconfigure(0, weight=1)
        title = ttk.Label(container, text="Aplicación de Modelado Mixto", font=("Helvetica", 16))
        title.grid(row=0, column=0, columnspan=4, pady=10)

        ttk.Label(container, text="Selecciona el archivo de datos (CSV o Excel):").grid(row=1, column=0, sticky="w", padx=5)
        self.file_entry = ttk.Entry(container, width=50)
        self.file_entry.grid(row=1, column=1, padx=5, pady=5)
        btn_browse = ttk.Button(container, text="Examinar", command=self.browse_file)
        btn_browse.grid(row=1, column=2, padx=5, pady=5)

        # --- Filtros Generales (Implementación directa) ---
        frm_filters_general = ttk.LabelFrame(container, text="Filtros Generales (Opcional)")
        frm_filters_general.grid(row=2, column=0, columnspan=4, padx=5, pady=5, sticky="ew") # Ajustar fila

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

        # --- Selección de Variables (Ajustar número de fila) ---
        ttk.Label(container, text="Variable dependiente (ej. SARA):").grid(row=3, column=0, sticky="w", padx=5)
        self.depvar_cb = ttk.Combobox(container, values=[], state="readonly", width=20)
        self.depvar_cb.grid(row=3, column=1, padx=5, pady=5) # Ajustar fila

        ttk.Label(container, text="Variable de grupo (ID/Sujeto):").grid(row=4, column=0, sticky="w", padx=5) # Ajustar fila
        self.groupvar_cb = ttk.Combobox(container, values=[], state="readonly", width=20)
        self.groupvar_cb.grid(row=4, column=1, padx=5, pady=5) # Ajustar fila

        ttk.Label(container, text="Variable de tiempo (ej. Hora o años de seguimiento):").grid(row=5, column=0, sticky="w", padx=5) # Ajustar fila
        self.horavar_cb = ttk.Combobox(container, values=[], state="readonly", width=20)
        self.horavar_cb.grid(row=5, column=1, padx=5, pady=5) # Ajustar fila

        self.centrar_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(container, text="Centrar la variable de tiempo", variable=self.centrar_var).grid(row=5, column=2, padx=5, pady=5) # Ajustar fila

        ttk.Label(container, text="Umbral para recodificar 'mas3menos' (en años):").grid(row=6, column=0, sticky="w", padx=5) # Ajustar fila
        self.threshold_entry = ttk.Entry(container, width=5)
        self.threshold_entry.insert(0, "3")
        self.threshold_entry.grid(row=6, column=1, padx=5, pady=5, sticky="w") # Ajustar fila

        self.incluir_mas3 = tk.BooleanVar(value=True)
        ttk.Checkbutton(container, text="Incluir 'mas3menos' y su interacción con el tiempo", variable=self.incluir_mas3).grid(row=7, column=0, columnspan=2, padx=5, pady=5) # Ajustar fila

        self.incluir_eval = tk.BooleanVar(value=True)
        ttk.Checkbutton(container, text="Incluir 'Evaluador' como efecto aleatorio adicional", variable=self.incluir_eval).grid(row=8, column=0, columnspan=2, padx=5, pady=5) # Ajustar fila

        ttk.Label(container, text="Efectos aleatorios:").grid(row=9, column=0, sticky="w", padx=5) # Ajustar fila
        self.re_option = tk.StringVar(value="RE1")
        ttk.Radiobutton(container, text="Solo intercepto aleatorio", variable=self.re_option, value="RE1").grid(row=9, column=1, sticky="w", padx=5) # Ajustar fila
        ttk.Radiobutton(container, text="Intercepto + pendiente", variable=self.re_option, value="RE2").grid(row=9, column=2, sticky="w", padx=5) # Ajustar fila

        # Nuevo: Combobox para seleccionar el "Tipo de Modelo Mixto"
        ttk.Label(container, text="Tipo de Modelo Mixto:").grid(row=10, column=0, sticky="w", padx=5, pady=5) # Ajustar fila
        self.tipo_modelo_cb = ttk.Combobox(container, values=["Automático", "Personalizado"], state="readonly", width=20)
        self.tipo_modelo_cb.grid(row=10, column=1, padx=5, pady=5) # Ajustar fila
        self.tipo_modelo_cb.set("Personalizado")
        # Si se elige "Automático", se podrían activar configuraciones adicionales (esto es solo un ejemplo).

        # Sección Variables adicionales
        frame_add = ttk.LabelFrame(container, text="Variables adicionales")
        frame_add.grid(row=11, column=0, columnspan=4, padx=5, pady=5, sticky="ew") # Ajustar fila
        ttk.Label(frame_add, text="Agregar variable fija:").grid(row=0, column=0, sticky="w", padx=5, pady=2)
        self.fixed_cb = ttk.Combobox(frame_add, values=[], state="readonly", width=20)
        self.fixed_cb.grid(row=0, column=1, padx=5, pady=2)
        btn_fixed = ttk.Button(frame_add, text="Agregar fija", command=self.agregar_fija)
        btn_fixed.grid(row=0, column=2, padx=5, pady=2)
        ttk.Label(frame_add, text="Fijas seleccionadas:").grid(row=1, column=0, sticky="w", padx=5, pady=2)
        self.fixed_listbox = tk.Listbox(frame_add, height=3)
        self.fixed_listbox.grid(row=1, column=1, padx=5, pady=2)
        ttk.Label(frame_add, text="Agregar variable aleatoria:").grid(row=2, column=0, sticky="w", padx=5, pady=2)
        self.random_cb = ttk.Combobox(frame_add, values=[], state="readonly", width=20)
        self.random_cb.grid(row=2, column=1, padx=5, pady=2)
        btn_random = ttk.Button(frame_add, text="Agregar aleatoria", command=self.agregar_aleatoria)
        btn_random.grid(row=2, column=2, padx=5, pady=2)
        ttk.Label(frame_add, text="Aleatorias seleccionadas:").grid(row=3, column=0, sticky="w", padx=5, pady=2)
        self.random_listbox = tk.Listbox(frame_add, height=2)
        self.random_listbox.grid(row=3, column=1, padx=5, pady=2)

        # Sección Análisis Multivariado
        frame_multi = ttk.LabelFrame(container, text="Análisis Multivariado")
        frame_multi.grid(row=12, column=0, columnspan=4, padx=5, pady=5, sticky="ew") # Ajustar fila
        ttk.Label(frame_multi, text="Selecciona variables predictoras (múltiple):").grid(row=0, column=0, sticky="w", padx=5, pady=2)
        self.multivar_listbox = tk.Listbox(frame_multi, selectmode=tk.MULTIPLE, height=4)
        self.multivar_listbox.grid(row=1, column=0, padx=5, pady=2, sticky="ew")

        # Variables combinadas
        ttk.Label(container, text="Si deseas crear variables combinadas, ingresa fórmulas (ej.: nuevo = Var1+Var2):").grid(row=13, column=0, sticky="w", padx=5, pady=5) # Ajustar fila
        self.combined_ml = tk.Text(container, width=60, height=3)
        self.combined_ml.grid(row=14, column=0, columnspan=4, padx=5, pady=5) # Ajustar fila

        btn_ejecutar = ttk.Button(container, text="Ejecutar Modelos", command=self.ejecutar_modelos)
        btn_ejecutar.grid(row=15, column=0, columnspan=4, pady=10) # Ajustar fila

        ttk.Label(container, text="Salida de resultados:").grid(row=16, column=0, sticky="w", padx=5, pady=5) # Ajustar fila
        self.output_ml = tk.Text(container, width=110, height=20, wrap="none")
        self.output_ml.grid(row=17, column=0, columnspan=4, padx=5, pady=5) # Ajustar fila

        self.populate_comboboxes()

    def populate_comboboxes(self):
        # Inicialmente sin datos, se actualizarán al cargar archivo
        columnas = []
        self.fixed_cb['values'] = columnas
        self.random_cb['values'] = columnas
        self.depvar_cb['values'] = columnas
        self.groupvar_cb['values'] = columnas
        self.horavar_cb['values'] = columnas
        self.multivar_listbox.delete(0, tk.END)
        for col in columnas:
            self.multivar_listbox.insert(tk.END, col)

    def browse_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV, Excel Files", "*.csv;*.xls;*.xlsx")])
        if file_path:
            self.file_entry.delete(0, tk.END)
            self.file_entry.insert(0, file_path)
            self.df = leer_archivo(file_path)
            if self.df is not None:
                columnas = list(self.df.columns)
                self.depvar_cb['values'] = columnas
                self.groupvar_cb['values'] = columnas
                self.horavar_cb['values'] = columnas
                self.fixed_cb['values'] = columnas
                self.random_cb['values'] = columnas
                self.multivar_listbox.delete(0, tk.END)
                for col in columnas:
                    self.multivar_listbox.insert(tk.END, col)
                # FilterComponent removido.
                # Actualizar combos de filtros generales
                filter_cols_options = [''] + columnas
                if hasattr(self, 'filter_col_1_combo'): # Verificar si los widgets ya existen
                    self.filter_col_1_combo['values'] = filter_cols_options
                    if not self.filter_col_1_var.get() and columnas: self.filter_col_1_var.set('')
                if hasattr(self, 'filter_col_2_combo'):
                    self.filter_col_2_combo['values'] = filter_cols_options
                    if not self.filter_col_2_var.get() and columnas: self.filter_col_2_var.set('')
                messagebox.showinfo("Información", "Archivo cargado correctamente.")

    def agregar_fija(self):
        var = self.fixed_cb.get()
        if var and var not in self.fixed_list:
            self.fixed_list.append(var)
            self.fixed_listbox.insert(tk.END, var)
            self.output_ml.insert(tk.END, f"Variable fija agregada: {var}\n")

    def agregar_aleatoria(self):
        var = self.random_cb.get()
        if var and var not in self.random_list:
            self.random_list.append(var)
            self.random_listbox.insert(tk.END, var)
            self.output_ml.insert(tk.END, f"Variable aleatoria agregada: {var}\n")

    def ejecutar_modelos(self):
        # Aplicar filtros primero
        df_original_cargado = self.df
        if df_original_cargado is None:
            messagebox.showerror("Error", "Primero carga un archivo de datos.")
            return

        # FilterComponent ha sido removido. Se trabaja directamente con una copia de df_original_cargado.
        df_para_analisis_inicial = df_original_cargado.copy()
        self.output_ml.insert(tk.END, "Usando datos originales para análisis (antes de filtros generales).\n")

        # Aplicar filtros generales definidos en la UI
        df_para_analisis = self._apply_general_filters(df_para_analisis_inicial)

        if df_para_analisis is None or df_para_analisis.empty:
            messagebox.showerror("Error", "No quedan datos después de aplicar los filtros generales.")
            return

        # Continuar con el DataFrame filtrado (df_para_analisis)
        dep_var = self.depvar_cb.get().strip()
        group_var = self.groupvar_cb.get().strip()
        hora_var = self.horavar_cb.get().strip()
        try:
            threshold = float(self.threshold_entry.get().strip())
        except:
            messagebox.showerror("Error", "El umbral debe ser numérico.")
            return

        centrar = self.centrar_var.get()
        incluir_mas3 = self.incluir_mas3.get()
        incluir_eval = self.incluir_eval.get()

        if not dep_var or not group_var or not hora_var:
            messagebox.showerror("Error", "Debes seleccionar las variables dependiente, de grupo y de tiempo.")
            return

        # Crear variables combinadas si se ingresaron
        formulas_text = self.combined_ml.get("1.0", tk.END)
        if formulas_text.strip():
            # Aplicar a df_para_analisis
            df_para_analisis = crear_variables_combinadas(df_para_analisis, formulas_text)
            self.output_ml.insert(tk.END, "Variables combinadas creadas.\n")

        # Recodificar 'mas3menos' en df_para_analisis
        df_para_analisis = recodificar_patron(df_para_analisis, group_var, hora_var, threshold=threshold)
        self.output_ml.insert(tk.END, f"Variable 'mas3menos' recodificada (umbral: {threshold}).\n")

        # Centrar la variable de tiempo en df_para_analisis
        time_var = hora_var
        if centrar:
            # Calcular media sobre los datos filtrados
            mean_time = df_para_analisis[hora_var].mean()
            df_para_analisis["HoraC"] = df_para_analisis[hora_var] - mean_time
            time_var = "HoraC"
            self.output_ml.insert(tk.END, f"Variable de tiempo centrada en 'HoraC' (media={mean_time:.2f}).\n")

        formula_prog = construir_formula(dep_var, time_var, incluir_mas3)
        self.output_ml.insert(tk.END, "\n--- MODELO DE PROGRESIÓN ---\n")
        self.output_ml.insert(tk.END, f"Fórmula de efectos fijos:\n{formula_prog}\n")

        # Efectos aleatorios según radiobutton y tipo de modelo mixto seleccionado
        if self.re_option.get() == "RE2":
            re_formula = f"1 + {time_var}"
        else:
            re_formula = "1"
        self.output_ml.insert(tk.END, f"Efectos aleatorios: {re_formula}\n")

        # Mostrar el tipo de modelo mixto seleccionado (Ejemplo: Automático o Personalizado)
        tipo_modelo = self.tipo_modelo_cb.get()
        self.output_ml.insert(tk.END, f"Tipo de Modelo Mixto seleccionado: {tipo_modelo}\n")
        # Si se desea, se pueden agregar ajustes o configuraciones automáticas para ciertos tipos.
        if tipo_modelo == "Automático":
            self.output_ml.insert(tk.END, "Se utilizará un ajuste automático para el modelo mixto (configuración predefinida).\n")
            # Por ejemplo, se puede definir un re_formula o vc_formula distinto.
            # Aquí se deja como ejemplo; se puede personalizar según la necesidad.
            # re_formula = "1 + " + time_var
        # Si se elige "Personalizado" se usan los valores establecidos en los controles.

        # Si se incluye Evaluador como efecto aleatorio adicional
        vc_formula = {}
        if incluir_eval:
            # Verificar en df_para_analisis
            if "Evaluador" in df_para_analisis.columns:
                vc_formula = {"Evaluador": "0 + C(Evaluador)"}
                self.output_ml.insert(tk.END, "Se incluye 'Evaluador' como efecto aleatorio adicional.\n")
            else:
                self.output_ml.insert(tk.END, "La columna 'Evaluador' no se encontró en los datos filtrados.\n")

        # Ajustar el modelo de progresión usando df_para_analisis
        try:
            modelo = ajustar_modelo(formula_prog, df_para_analisis, group_var, re_formula, vc_formula)
            if modelo is not None:
                self.output_ml.insert(tk.END, "\nResumen del Modelo de Progresión:\n")
                self.output_ml.insert(tk.END, str(modelo.summary()) + "\n")
        except Exception as e:
            messagebox.showerror("Error", f"Error al ajustar el modelo de progresión:\n{e}")
            traceback.print_exc()

        # MODELO MULTIVARIADO
        multivar_indices = self.multivar_listbox.curselection()
        if multivar_indices:
            multivar_selected = [self.multivar_listbox.get(i) for i in multivar_indices]
            interacciones = [f"{pred}:{time_var}" for pred in multivar_selected]
            formula_multivar = formula_prog + " + " + " + ".join(multivar_selected + interacciones)
            self.output_ml.insert(tk.END, "\n--- MODELO MULTIVARIADO ---\n")
            self.output_ml.insert(tk.END, f"Fórmula completa:\n{formula_multivar}\n")
            try:
                # Ajustar usando df_para_analisis
                modelo_mv = ajustar_modelo(formula_multivar, df_para_analisis, group_var, re_formula, vc_formula)
                if modelo_mv is not None:
                    self.output_ml.insert(tk.END, "\nResumen del Modelo Multivariado:\n")
                    self.output_ml.insert(tk.END, str(modelo_mv.summary()) + "\n")
            except Exception as e:
                messagebox.showerror("Error", f"Error al ajustar el modelo multivariado:\n{e}")
                traceback.print_exc()
        else:
            self.output_ml.insert(tk.END, "\nNo se seleccionaron variables predictoras para el modelo multivariado.\n")

if __name__ == "__main__":
    # Si se ejecuta mixmodel.py de forma independiente, se mostrará una ventana.
    root = tk.Tk()
    root.title("Aplicación Modelos Mixtos")
    root.geometry("900x700")
    app = MixModelTab(root)
    app.pack(fill="both", expand=True)
    root.mainloop()
