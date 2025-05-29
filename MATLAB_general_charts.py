#!/usr/bin/env python
# -*- coding: utf-8 -*-
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog, StringVar
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns
import os # <--- AÑADIDO
import csv # <--- AÑADIDO
# Otros imports necesarios para gráficos específicos se añadirán después

class GeneralChartsApp(ttk.Frame):
    def __init__(self, parent, main_app_instance=None): # main_app_instance para acceder a datos/logs globales si es necesario
        super().__init__(parent)
        self.parent_for_dialogs = parent
        self.main_app = main_app_instance # Referencia a la aplicación principal
        self.data = None # DataFrame actual
        self.chart_descriptions = self.load_chart_descriptions() # Cargar descripciones

        # --- UI Principal de la Pestaña ---
        main_frame = ttk.Frame(self, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Panel de Controles (Izquierda)
        controls_panel = ttk.LabelFrame(main_frame, text="Controles de Gráfico", padding="10")
        controls_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10), pady=5)

        # Panel de Visualización (Derecha)
        display_panel = ttk.Frame(main_frame)
        display_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, pady=5)

        # --- Controles ---
        ttk.Label(controls_panel, text="Tipo de Gráfico:").pack(pady=(0,2), anchor="w")
        self.chart_type_var = StringVar()
        self.chart_types = [
            "Diagrama de Dispersión", "Trimap (Mosaico Jerárquico)", "Diagramas de Barras",
            "Histogramas", "Gráficos Q-Q", "Diagramas de Caja",
            "Gráfico de Densidad Suave", "Gráfico de Líneas", "Curvas de Kaplan-Meier",
            "Mapas de Calor", "Polígonos de Frecuencia", "Gráfico de Pastel",
            "Diagrama de Flujo" # Considerar cómo implementar
        ]
        self.chart_type_combo = ttk.Combobox(controls_panel, textvariable=self.chart_type_var, values=self.chart_types, state="readonly", width=30)
        self.chart_type_combo.pack(pady=(0,10), fill="x")
        self.chart_type_combo.bind("<<ComboboxSelected>>", self._update_parameter_controls)

        self.parameter_controls_frame = ttk.Frame(controls_panel)
        self.parameter_controls_frame.pack(fill="x", expand=True, pady=(0,10))

        ttk.Button(controls_panel, text="Generar Gráfico", command=self._generate_chart).pack(pady=10, fill="x")
        ttk.Button(controls_panel, text="Cargar Datos (Excel/CSV)", command=self.cargar_datos_para_graficos).pack(pady=5, fill="x")
        self.lbl_data_status = ttk.Label(controls_panel, text="Ningún archivo cargado.")
        self.lbl_data_status.pack(pady=(5,0), anchor="w")


        # --- Área de Visualización del Gráfico ---
        self.chart_display_frame = ttk.LabelFrame(display_panel, text="Visualización del Gráfico", padding="5")
        self.chart_display_frame.pack(fill=tk.BOTH, expand=True)
        # El canvas para Matplotlib/Seaborn se creará aquí dinámicamente
        # Para Plotly, podríamos usar un WebView o exportar y abrir.

        # --- Log ---
        log_frame = ttk.LabelFrame(display_panel, text="Información del Gráfico y Log", height=150)
        log_frame.pack(fill=tk.X, pady=(10,0))
        log_frame.pack_propagate(False) # Evitar que se encoja

        self.log_text_widget = tk.Text(log_frame, height=8, wrap=tk.WORD, state=tk.DISABLED, font=("Courier New", 9))
        log_scroll_y = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.log_text_widget.yview)
        self.log_text_widget.config(yscrollcommand=log_scroll_y.set)
        log_scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=2, pady=2)
        self._configure_log_tags()

        self.log("Pestaña 'Gráficas' inicializada. Seleccione un tipo de gráfico y cargue datos.", "INFO")

    def _configure_log_tags(self):
        self.log_text_widget.tag_config("INFO", foreground="black")
        self.log_text_widget.tag_config("DEBUG", foreground="gray")
        self.log_text_widget.tag_config("WARN", foreground="orange")
        self.log_text_widget.tag_config("ERROR", foreground="red", font=("Courier New", 9, "bold"))
        self.log_text_widget.tag_config("SUCCESS", foreground="green")
        self.log_text_widget.tag_config("DESC", foreground="navy", font=("Courier New", 9, "bold"))
        self.log_text_widget.tag_config("PARAMS", foreground="purple")
        self.log_text_widget.tag_config("RECOM", foreground="darkgreen")

    def log(self, message, level="INFO"):
        try:
            timestamp = pd.Timestamp.now().strftime('%H:%M:%S.%f')[:-3]
            self.log_text_widget.config(state=tk.NORMAL)
            self.log_text_widget.insert(tk.END, f"[{timestamp}] [{level.upper()}] {message}\n", level.upper())
            self.log_text_widget.config(state=tk.DISABLED)
            self.log_text_widget.see(tk.END)
        except Exception as e:
            print(f"Error en logger de GeneralChartsApp: {e}")

    def load_chart_descriptions(self):
        # Aquí cargarías las descripciones, parámetros y recomendaciones desde un archivo o diccionario
        # Por ahora, un placeholder
        return {
            "Diagrama de Dispersión": {
                "descripcion": "Representa puntos en un plano cartesiano (x, y). Útil para mostrar relaciones o correlaciones entre dos variables.",
                "parametros_clave": "Variables X, Y. Opcional: color, tamaño, hover_name (Plotly).",
                "recomendaciones": "Ideal para visualizar la relación entre dos variables continuas. Considerar la sobreimpresión de puntos si hay muchos datos."
            },
            "Gráfico de Pastel": {
                "descripcion": "Un círculo dividido en secciones, donde cada sección representa una proporción del total.",
                "parametros_clave": "Columna de valores, columna de etiquetas.",
                "recomendaciones": "Generalmente desaconsejado para comparaciones precisas. Usar con pocas categorías. Evitar versiones 3D."
            },
            "Histogramas": {
                "descripcion": "Muestra la distribución de una variable numérica dividiendo los datos en 'bins' (intervalos) y contando las observaciones en cada bin.",
                "parametros_clave": "Variable (numérica), Número de bins (opcional), Mostrar KDE (opcional).",
                "recomendaciones": "Útil para entender la forma de la distribución de los datos (simetría, picos, etc.). Experimentar con el número de bins."
            }
            # ... Añadir descripciones para todos los demás gráficos
        }

    def cargar_datos_para_graficos(self):
        filepath = filedialog.askopenfilename(
            title="Seleccionar archivo de datos para gráficos",
            filetypes=(("Archivos Excel", "*.xlsx *.xls"),
                       ("Archivos CSV", "*.csv"),
                       ("Todos los archivos", "*.*")),
            parent=self.parent_for_dialogs
        )
        if not filepath:
            self.log("Carga de archivo cancelada.", "INFO")
            return

        try:
            filename = os.path.basename(filepath) # 'os' ya está importado
            if filepath.endswith(('.xlsx', '.xls')):
                self.data = pd.read_excel(filepath, engine='openpyxl')
            elif filepath.endswith('.csv'):
                try:
                    sniffer = csv.Sniffer() # 'csv' ya está importado
                    with open(filepath, 'r', encoding='utf-8-sig') as f:
                        dialect = sniffer.sniff(f.read(1024))
                    self.data = pd.read_csv(filepath, sep=dialect.delimiter)
                    self.log(f"Archivo CSV '{filename}' cargado con separador '{dialect.delimiter}' detectado.", "INFO")
                except Exception: 
                    try: self.data = pd.read_csv(filepath, sep=',')
                    except pd.errors.ParserError: self.data = pd.read_csv(filepath, sep=';')
            else:
                messagebox.showerror("Error de Archivo", f"Tipo de archivo no soportado: {filename}", parent=self.parent_for_dialogs)
                self.log(f"Tipo de archivo no soportado: {filename}", "ERROR")
                return
            
            self.lbl_data_status.config(text=f"{filename} ({self.data.shape[0]}x{self.data.shape[1]})")
            self.log(f"Datos cargados desde '{filename}'. Dimensiones: {self.data.shape}", "SUCCESS")
            self._update_parameter_controls() 

        except Exception as e:
            messagebox.showerror("Error de Lectura", f"No se pudo leer el archivo:\n{e}", parent=self.parent_for_dialogs)
            self.log(f"Error leyendo archivo '{filepath}': {e}", "ERROR")
            self.data = None
            self.lbl_data_status.config(text="Error al cargar.")


    def _update_parameter_controls(self, event=None):
        for widget in self.parameter_controls_frame.winfo_children():
            widget.destroy()

        chart_type = self.chart_type_var.get()
        if not chart_type:
            return

        self.log(f"Configurando parámetros para: {chart_type}", "DEBUG")
        
        description_data = self.chart_descriptions.get(chart_type, {})
        if description_data:
            self.log(f"Descripción ({chart_type}): {description_data.get('descripcion', 'N/A')}", "DESC")
            self.log(f"Recomendaciones: {description_data.get('recomendaciones', 'N/A')}", "RECOM")


        if self.data is None:
            ttk.Label(self.parameter_controls_frame, text="Cargue datos primero.").pack()
            return

        columnas = list(self.data.columns)
        
        if chart_type == "Diagrama de Dispersión":
            ttk.Label(self.parameter_controls_frame, text="Variable X:").pack(anchor="w")
            self.param_x_var = StringVar()
            ttk.Combobox(self.parameter_controls_frame, textvariable=self.param_x_var, values=columnas, state="readonly").pack(fill="x", pady=(0,5))
            
            ttk.Label(self.parameter_controls_frame, text="Variable Y:").pack(anchor="w")
            self.param_y_var = StringVar()
            ttk.Combobox(self.parameter_controls_frame, textvariable=self.param_y_var, values=columnas, state="readonly").pack(fill="x", pady=(0,5))

            ttk.Label(self.parameter_controls_frame, text="Color (Opcional):").pack(anchor="w")
            self.param_color_var = StringVar()
            ttk.Combobox(self.parameter_controls_frame, textvariable=self.param_color_var, values=[""] + columnas, state="readonly").pack(fill="x", pady=(0,5))
            
            ttk.Label(self.parameter_controls_frame, text="Tamaño (Opcional, Numérica):").pack(anchor="w")
            self.param_size_var = StringVar()
            ttk.Combobox(self.parameter_controls_frame, textvariable=self.param_size_var, values=[""] + columnas, state="readonly").pack(fill="x", pady=(0,5))
            
            ttk.Label(self.parameter_controls_frame, text="Hover Name (Opcional, Plotly):").pack(anchor="w")
            self.param_hover_name_var = StringVar()
            ttk.Combobox(self.parameter_controls_frame, textvariable=self.param_hover_name_var, values=[""] + columnas, state="readonly").pack(fill="x", pady=(0,5))

        elif chart_type == "Gráfico de Pastel":
            ttk.Label(self.parameter_controls_frame, text="Columna de Valores:").pack(anchor="w")
            self.param_pie_values_var = StringVar()
            ttk.Combobox(self.parameter_controls_frame, textvariable=self.param_pie_values_var, values=columnas, state="readonly").pack(fill="x", pady=(0,5))

            ttk.Label(self.parameter_controls_frame, text="Columna de Etiquetas:").pack(anchor="w")
            self.param_pie_labels_var = StringVar()
            ttk.Combobox(self.parameter_controls_frame, textvariable=self.param_pie_labels_var, values=columnas, state="readonly").pack(fill="x", pady=(0,5))
        
        else:
            ttk.Label(self.parameter_controls_frame, text=f"Controles para '{chart_type}' no implementados.").pack()


    def _clear_chart_display(self):
        for widget in self.chart_display_frame.winfo_children():
            widget.destroy()

    def _generate_chart(self):
        self._clear_chart_display()
        chart_type = self.chart_type_var.get()
        
        if self.data is None:
            messagebox.showwarning("Sin Datos", "Por favor, cargue un archivo de datos primero.", parent=self.parent_for_dialogs)
            return
        if not chart_type:
            messagebox.showwarning("Sin Selección", "Por favor, seleccione un tipo de gráfico.", parent=self.parent_for_dialogs)
            return

        self.log(f"Generando gráfico: {chart_type}", "INFO")
        params_used_log = [f"Tipo de Gráfico: {chart_type}"]

        try:
            if chart_type == "Diagrama de Dispersión":
                x_col = self.param_x_var.get()
                y_col = self.param_y_var.get()
                color_col = self.param_color_var.get() if hasattr(self, 'param_color_var') and self.param_color_var.get() else None
                size_col = self.param_size_var.get() if hasattr(self, 'param_size_var') and self.param_size_var.get() else None
                hover_name_col = self.param_hover_name_var.get() if hasattr(self, 'param_hover_name_var') and self.param_hover_name_var.get() else None

                if not x_col or not y_col:
                    messagebox.showerror("Error", "Debe seleccionar variables para X e Y.", parent=self.parent_for_dialogs)
                    return

                params_used_log.append(f"  Variable X: {x_col}")
                params_used_log.append(f"  Variable Y: {y_col}")
                if color_col: params_used_log.append(f"  Color por: {color_col}")
                if size_col: params_used_log.append(f"  Tamaño por: {size_col}")
                if hover_name_col: params_used_log.append(f"  Hover Name: {hover_name_col}")
                
                plt_fig, ax = plt.subplots()
                if size_col and color_col:
                    sns.scatterplot(data=self.data, x=x_col, y=y_col, hue=color_col, size=size_col, ax=ax, legend="auto")
                elif color_col:
                    sns.scatterplot(data=self.data, x=x_col, y=y_col, hue=color_col, ax=ax, legend="auto")
                elif size_col:
                    sns.scatterplot(data=self.data, x=x_col, y=y_col, size=size_col, ax=ax, legend="auto")
                else:
                    sns.scatterplot(data=self.data, x=x_col, y=y_col, ax=ax)
                
                ax.set_title(f"Diagrama de Dispersión: {y_col} vs {x_col}")
                ax.set_xlabel(x_col)
                ax.set_ylabel(y_col)
                
                canvas = FigureCanvasTkAgg(plt_fig, master=self.chart_display_frame)
                canvas.draw()
                canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
                self.log("Diagrama de dispersión generado con Seaborn/Matplotlib.", "SUCCESS")

            elif chart_type == "Gráfico de Pastel":
                values_col = self.param_pie_values_var.get()
                labels_col = self.param_pie_labels_var.get()

                if not values_col or not labels_col:
                    messagebox.showerror("Error", "Debe seleccionar columna de valores y etiquetas.", parent=self.parent_for_dialogs)
                    return
                
                params_used_log.append(f"  Columna Valores: {values_col}")
                params_used_log.append(f"  Columna Etiquetas: {labels_col}")
                
                if self.data[labels_col].nunique() > 15:
                    self.log("Advertencia: Demasiadas categorías para un gráfico de pastel efectivo.", "WARN")
                
                pie_data = self.data.set_index(labels_col)[values_col]

                plt_fig, ax = plt.subplots()
                ax.pie(pie_data, labels=pie_data.index, autopct='%1.1f%%', startangle=90)
                ax.axis('equal') 
                ax.set_title(f"Gráfico de Pastel por {labels_col}")
                
                canvas = FigureCanvasTkAgg(plt_fig, master=self.chart_display_frame)
                canvas.draw()
                canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
                self.log("Gráfico de pastel generado.", "SUCCESS")

            else:
                messagebox.showinfo("Pendiente", f"La generación de '{chart_type}' aún no está implementada.", parent=self.parent_for_dialogs)
                self.log(f"Generación de '{chart_type}' pendiente.", "WARN")
                return

            self.log("Parámetros Utilizados:", "PARAMS")
            for param_log in params_used_log:
                self.log(f"  {param_log}", "PARAMS")

        except Exception as e:
            self.log(f"Error generando gráfico '{chart_type}': {e}", "ERROR")
            traceback.print_exc()
            messagebox.showerror("Error de Graficación", f"No se pudo generar el gráfico:\n{e}", parent=self.parent_for_dialogs)

if __name__ == '__main__':
    root = tk.Tk()
    root.title("Prueba Pestaña Gráficas Generales")
    root.geometry("1000x700")
    
    app_tab = GeneralChartsApp(root)
    app_tab.pack(fill="both", expand=True)
    root.mainloop()