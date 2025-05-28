#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import io
import os
import traceback

# FilterComponent ha sido eliminado.
FilterComponent = None # Mantener para evitar errores si alguna lógica residual lo verifica.

try:
    from factor_analyzer.factor_analyzer import calculate_kmo, calculate_bartlett_sphericity
    FACTOR_ANALYZER_AVAILABLE = True
except ModuleNotFoundError:
    FACTOR_ANALYZER_AVAILABLE = False

class PrincompTab(ttk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.df_original = None # DataFrame cargado originalmente
        self.df_for_pca = None  # DataFrame después de filtros, listo para PCA
        self.image_references = [] 

        # Variables de control de Tkinter
        self.estandarizar_var = tk.BooleanVar(value=True)
        self.filepath_var = tk.StringVar()
        self.n_components_var = tk.StringVar()

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
        
        # Se eliminan las variables de filtro antiguas
        # self.filter_var1_name = tk.StringVar()
        # self.filter_var1_value = tk.StringVar()
        # self.filter_var2_name = tk.StringVar()
        # self.filter_var2_value = tk.StringVar()
        self.exclude_nans_in_filter_var = tk.BooleanVar(value=True) # Se mantiene para limpieza post-filtro general

        self._build_ui_with_scroll()

    def _build_ui_with_scroll(self):
        # Canvas principal para scroll
        canvas = tk.Canvas(self)
        canvas.pack(side="left", fill="both", expand=True)

        # Scrollbar vertical
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        scrollbar.pack(side="right", fill="y")
        canvas.configure(yscrollcommand=scrollbar.set)

        # Frame contenido dentro del canvas
        self.scrollable_frame = ttk.Frame(canvas)
        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        self._build_main_content(self.scrollable_frame)


    def _build_main_content(self, parent_frame):
        # Frame principal para los controles
        main_controls_frame = ttk.LabelFrame(parent_frame, text="Análisis de Componentes Principales (PCA)")
        main_controls_frame.pack(padx=10, pady=10, fill="x")

        # --- Carga de Archivo ---
        file_frame = ttk.LabelFrame(main_controls_frame, text="1. Carga de Datos")
        file_frame.pack(fill="x", pady=5, padx=5)
        
        path_frame = ttk.Frame(file_frame)
        path_frame.pack(fill="x", pady=2)
        ttk.Label(path_frame, text="Archivo:").pack(side="left", padx=5)
        ttk.Entry(path_frame, textvariable=self.filepath_var, width=70, state="readonly").pack(side="left", padx=5, expand=True, fill="x")
        ttk.Button(path_frame, text="Buscar Archivo", command=self._load_file).pack(side="left", padx=5)

        # --- Filtros Generales (Implementación directa) ---
        frm_filters_general = ttk.LabelFrame(main_controls_frame, text="Filtros Generales (Opcional)")
        frm_filters_general.pack(fill="x", pady=5, padx=5)

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
        
        # Opción de limpieza de NaNs (se aplica después de los filtros generales)
        # Mover esta opción a un frame diferente o directamente si es necesario.
        # Por ahora, la mantenemos aquí para que la UI no cambie drásticamente,
        # aunque su funcionalidad ahora dependerá de cómo se manejen los datos sin FilterComponent.
        # Considerar crear un nuevo LabelFrame para esta opción si es necesario.
        temp_filter_options_frame = ttk.LabelFrame(main_controls_frame, text="2. Opciones de Limpieza de Datos")
        temp_filter_options_frame.pack(fill="x", pady=5, padx=5)
        ttk.Checkbutton(temp_filter_options_frame, text="Eliminar filas con NaNs en variables de PCA", variable=self.exclude_nans_in_filter_var).pack(anchor="w", padx=5, pady=2)


        # --- Selección de Variables para PCA ---
        vars_frame = ttk.LabelFrame(main_controls_frame, text="3. Variables para PCA y Opciones")
        vars_frame.pack(fill="x", pady=5, padx=5)
        
        ttk.Label(vars_frame, text="Variables (una por línea; formato: nombre_original o nombre_original:NuevoNombre):").pack(anchor="w", padx=5)
        
        vars_text_frame = ttk.Frame(vars_frame)
        vars_text_frame.pack(fill="both", expand=True, pady=2, padx=5)
        self.pca_vars_text = tk.Text(vars_text_frame, height=8, width=60, wrap="word")
        vars_text_scrollbar = ttk.Scrollbar(vars_text_frame, orient="vertical", command=self.pca_vars_text.yview)
        self.pca_vars_text.configure(yscrollcommand=vars_text_scrollbar.set)
        vars_text_scrollbar.pack(side="right", fill="y")
        self.pca_vars_text.pack(side="left", fill="both", expand=True)
        
        # --- Opciones de PCA ---
        options_frame = ttk.Frame(vars_frame)
        options_frame.pack(fill="x", pady=5, padx=5)
        
        ttk.Checkbutton(options_frame, text="Estandarizar variables", variable=self.estandarizar_var).pack(side="left", padx=5)
        ttk.Label(options_frame, text="Núm. Componentes (opcional):").pack(side="left", padx=15)
        ttk.Entry(options_frame, textvariable=self.n_components_var, width=5).pack(side="left", padx=5)

        # --- Botones de Acción ---
        actions_frame = ttk.LabelFrame(main_controls_frame, text="4. Ejecutar Análisis")
        actions_frame.pack(fill="x", pady=10, padx=5)
        ttk.Button(actions_frame, text="Calcular KMO y Bartlett", command=self._calculate_kmo_bartlett).pack(side="left", padx=5)
        ttk.Button(actions_frame, text="Ejecutar PCA", command=self._run_pca).pack(side="left", padx=5)

        # --- Resultados y Gráficos (Notebook) ---
        results_notebook = ttk.Notebook(parent_frame) # Empaquetar en el parent_frame (scrollable_frame)
        results_notebook.pack(padx=10, pady=10, fill="both", expand=True)

        tables_frame = ttk.Frame(results_notebook)
        results_notebook.add(tables_frame, text="Tablas y Resultados")
        self.results_text = tk.Text(tables_frame, wrap="none", height=20) # wrap="none" para scroll horizontal
        results_v_scrollbar = ttk.Scrollbar(tables_frame, orient="vertical", command=self.results_text.yview)
        results_h_scrollbar = ttk.Scrollbar(tables_frame, orient="horizontal", command=self.results_text.xview)
        self.results_text.configure(yscrollcommand=results_v_scrollbar.set, xscrollcommand=results_h_scrollbar.set)
        results_v_scrollbar.pack(side="right", fill="y")
        results_h_scrollbar.pack(side="bottom", fill="x")
        self.results_text.pack(fill="both", expand=True, padx=5, pady=5)

        plots_main_frame = ttk.Frame(results_notebook)
        results_notebook.add(plots_main_frame, text="Gráficos")
        
        self.plots_notebook = ttk.Notebook(plots_main_frame)
        self.plots_notebook.pack(fill="both", expand=True, padx=5, pady=5)

        self.scree_frame = ttk.Frame(self.plots_notebook)
        self.plots_notebook.add(self.scree_frame, text="Scree Plot")
        self.scree_label_img = ttk.Label(self.scree_frame)
        self.scree_label_img.pack(padx=5, pady=5)

        self.biplot_frame = ttk.Frame(self.plots_notebook)
        self.plots_notebook.add(self.biplot_frame, text="Biplot")
        self.biplot_label_img = ttk.Label(self.biplot_frame)
        self.biplot_label_img.pack(padx=5, pady=5)

        self.heatmap_frame = ttk.Frame(self.plots_notebook)
        self.plots_notebook.add(self.heatmap_frame, text="Heatmap Cargas")
        self.heatmap_label_img = ttk.Label(self.heatmap_frame)
        self.heatmap_label_img.pack(padx=5, pady=5)

    def _load_file(self):
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
            if self.df_original is not None:
                all_cols = self.df_original.columns.tolist()
                numeric_cols = self.df_original.select_dtypes(include=np.number).columns.tolist()
                
                # FilterComponent removido.
                # Actualizar combos de filtros generales
                filter_cols_options = [''] + all_cols # Usar all_cols que ya está definido
                if hasattr(self, 'filter_col_1_combo'): # Verificar si los widgets ya existen
                    self.filter_col_1_combo['values'] = filter_cols_options
                    if not self.filter_col_1_var.get() and all_cols: self.filter_col_1_var.set('')
                if hasattr(self, 'filter_col_2_combo'):
                    self.filter_col_2_combo['values'] = filter_cols_options
                    if not self.filter_col_2_var.get() and all_cols: self.filter_col_2_var.set('')
                
                # Limpiar y rellenar lista de variables para PCA
                self.pca_vars_text.delete("1.0", tk.END)
                for col in numeric_cols:
                    self.pca_vars_text.insert(tk.END, col + "\n")
                
                messagebox.showinfo("Archivo Cargado", f"Archivo '{os.path.basename(filepath)}' cargado. {len(numeric_cols)} variables numéricas encontradas y listadas para PCA.")
        except Exception as e:
            messagebox.showerror("Error al Leer Archivo", f"No se pudo leer el archivo:\n{e}")
            self.df_original = None; self.filepath_var.set(""); self.pca_vars_text.delete("1.0", tk.END)
            # FilterComponent removido.
            # Limpiar combos de filtros generales en caso de error
            if hasattr(self, 'filter_col_1_combo'): self.filter_col_1_combo['values'] = ['']
            if hasattr(self, 'filter_col_2_combo'): self.filter_col_2_combo['values'] = ['']

    # Se elimina _parse_filter_values

    def _get_filtered_df(self):
        """Obtiene los datos filtrados usando el FilterComponent y aplica limpieza de NaNs si está seleccionada."""
        if self.df_original is None:
            self.log("No hay datos originales cargados.", "WARN")
            return None

        if self.df_original is None:
            self.log("No hay datos originales cargados en _get_filtered_df.", "WARN")
            return None

        # FilterComponent ha sido removido. Se trabaja directamente con una copia de self.df_original.
        df_initial = self.df_original.copy()
        self.log("Usando datos originales (antes de filtros generales) en _get_filtered_df.", "INFO")

        # Aplicar filtros generales definidos en la UI
        df_filtered = self._apply_general_filters(df_initial)

        if df_filtered is None or df_filtered.empty:
            self.log("DataFrame vacío después de aplicar filtros generales en _get_filtered_df.", "WARN")
            return df_filtered # Devolver df_filtered que podría ser None o vacío

        # La opción exclude_nans_in_filter_var ahora se aplica DESPUÉS de los filtros generales,
        # y específicamente a las columnas que se usarán en PCA (ver _run_pca).
        # Aquí solo devolvemos el resultado del FilterComponent.
        # La limpieza de NaNs específica para PCA se hará en _run_pca sobre las columnas seleccionadas.
        return df_filtered

    def _parse_pca_variables(self):
        text_content = self.pca_vars_text.get("1.0", tk.END).strip()
        if not text_content:
            messagebox.showerror("Error", "No se especificaron variables para PCA.")
            return None, None

        lines = [line.strip() for line in text_content.split("\n") if line.strip()]
        original_vars = []
        display_names = []
        
        df_to_check = self._get_filtered_df()
        if df_to_check is None: 
            messagebox.showerror("Error", "No se pudo obtener el DataFrame (posiblemente error en filtros o no hay datos cargados).")
            return None, None
        
        available_cols = df_to_check.columns.tolist()

        for line in lines:
            if ":" in line:
                parts = line.split(":", 1)
                original = parts[0].strip()
                display = parts[1].strip()
            else:
                original = line.strip()
                display = original
            
            if original not in available_cols:
                messagebox.showwarning("Variable no Encontrada", f"La variable original '{original}' no se encuentra en el DataFrame filtrado. Será omitida.")
                continue
            if not pd.api.types.is_numeric_dtype(df_to_check[original]):
                 messagebox.showwarning("Variable no Numérica", f"La variable '{original}' no es numérica en el DataFrame filtrado y será omitida del PCA.")
                 continue

            original_vars.append(original)
            display_names.append(display)
        
        if not original_vars:
            messagebox.showerror("Error", "No hay variables válidas seleccionadas o disponibles para PCA después de la validación.")
            return None, None
            
        return original_vars, display_names

    def _clear_previous_results(self):
        self.results_text.delete("1.0", tk.END)
        self.scree_label_img.configure(image=None)
        self.biplot_label_img.configure(image=None)
        self.heatmap_label_img.configure(image=None)
        self.image_references = [] 

    def _calculate_kmo_bartlett(self):
        if not FACTOR_ANALYZER_AVAILABLE:
            messagebox.showerror("Error de Módulo", "El módulo 'factor_analyzer' no está instalado.\nPor favor, instálalo ejecutando: pip install factor_analyzer")
            return

        original_vars, _ = self._parse_pca_variables()
        if not original_vars: return

        df_current = self._get_filtered_df()
        if df_current is None or df_current.empty:
            messagebox.showerror("Error", "No hay datos disponibles después de aplicar filtros.")
            return
        
        X = df_current[original_vars].apply(pd.to_numeric, errors='coerce').dropna()
        if X.empty or X.shape[1] < 2: 
             messagebox.showerror("Error de Datos", "Se requieren al menos dos variables numéricas válidas sin NaNs para KMO y Bartlett.")
             return

        try:
            kmo_all, kmo_model = calculate_kmo(X)
            chi_square_value, p_value = calculate_bartlett_sphericity(X)
            result_text = (f"Prueba de KMO y Bartlett (sobre {len(X)} observaciones):\n" + "-"*30 +
                           f"\nKMO (global): {kmo_model:.3f}\nChi-cuadrado Bartlett: {chi_square_value:.3f}\np-valor Bartlett: {p_value:.3e}\n\n"
                           "Interpretación KMO:\n  0.9+: Sobresaliente | 0.8-0.9: Meritorio | 0.7-0.8: Mediano\n  0.6-0.7: Mediocre | 0.5-0.6: Mínimo | <0.5: Inaceptable\n"
                           "Bartlett: Si p-valor < 0.05, adecuado para PCA.")
            messagebox.showinfo("Resultados KMO y Bartlett", result_text)
        except Exception as e:
            messagebox.showerror("Error en Cálculo", f"Error al calcular KMO y Bartlett:\n{e}")

    def _run_pca(self):
        self._clear_previous_results()
        
        original_vars, display_vars = self._parse_pca_variables()
        if not original_vars: return

        df_current = self._get_filtered_df()
        if df_current is None or df_current.empty:
            messagebox.showerror("Error", "No hay datos disponibles después de aplicar filtros para PCA.")
            return
        
        X_selected = df_current[original_vars].apply(pd.to_numeric, errors='coerce')
        X_clean = X_selected.dropna()

        if X_clean.empty or X_clean.shape[0] < X_clean.shape[1] or X_clean.shape[1] == 0:
            messagebox.showerror("Error de Datos", "No hay suficientes datos válidos (filas sin NaNs y columnas numéricas) o hay menos observaciones que variables para realizar el PCA.")
            return

        estandarizar = self.estandarizar_var.get()
        n_comp_str = self.n_components_var.get()
        n_comp = None
        if n_comp_str:
            try:
                n_comp = int(n_comp_str)
                if n_comp <= 0 or n_comp > X_clean.shape[1]:
                    messagebox.showwarning("Advertencia", f"Núm. componentes inválido. Usando {X_clean.shape[1]} componentes.")
                    n_comp = X_clean.shape[1]
            except ValueError:
                messagebox.showwarning("Advertencia", "Núm. componentes no es entero. Usando todos.")
                n_comp = X_clean.shape[1]
        else:
            n_comp = X_clean.shape[1]

        try:
            X_data = X_clean.values
            if estandarizar:
                scaler = StandardScaler(); X_scaled = scaler.fit_transform(X_data)
            else: X_scaled = X_data

            pca = PCA(n_components=n_comp); pca.fit(X_scaled)
            scores = pca.transform(X_scaled)
            eigenvalues = pca.explained_variance_
            loadings = pca.components_.T * np.sqrt(eigenvalues) 
            comunalidades = np.sum(loadings**2, axis=1)
            varianza_explicada = pca.explained_variance_ratio_

            cum_var = np.cumsum(varianza_explicada) * 100
            tabla_var_str = "Componente\tAutovalor\t% Varianza\t% Acumulado\n"
            for i, (ev, var, ac) in enumerate(zip(eigenvalues, varianza_explicada, cum_var), start=1):
                tabla_var_str += f"CP{i}\t\t{ev:.3f}\t\t{var*100:.2f}%\t\t{ac:.2f}%\n"
            
            df_load = pd.DataFrame(loadings, index=display_vars, columns=[f"CP{i}" for i in range(1, loadings.shape[1] + 1)])
            df_comm = pd.DataFrame(comunalidades, index=display_vars, columns=["Comunalidad"])
            
            resultados_texto = "Tabla de Varianza Explicada:\n" + tabla_var_str + "\n\n"
            resultados_texto += "Matriz de Cargas Factoriales:\n" + df_load.to_string(float_format="%.3f") + "\n\n"
            resultados_texto += "Comunalidades:\n" + df_comm.to_string(float_format="%.3f") + "\n"
            self.results_text.insert(tk.END, resultados_texto)

            plt.figure(figsize=(6, 4))
            plt.plot(range(1, len(eigenvalues) + 1), eigenvalues, marker="o", linestyle="-")
            plt.title("Scree Plot"); plt.xlabel("Componente"); plt.ylabel("Autovalor")
            plt.xticks(range(1, len(eigenvalues) + 1)); plt.grid(True)
            scree_path = self._save_plot_to_temp("scree_plot.png"); plt.close()
            if scree_path: img_scree = tk.PhotoImage(file=scree_path); self.scree_label_img.configure(image=img_scree); self.image_references.append(img_scree)

            if scores.shape[1] >= 2:
                plt.figure(figsize=(6.5, 6))
                plt.scatter(scores[:,0], scores[:,1], alpha=0.5, s=30)
                for i, var_name in enumerate(display_vars):
                    arrow_x, arrow_y = loadings[i,0], loadings[i,1]
                    plt.arrow(0, 0, arrow_x, arrow_y, color="r", width=0.005, head_width=0.05, alpha=0.7)
                    plt.text(arrow_x * 1.15, arrow_y * 1.15, var_name, color="r", ha='center', va='center')
                plt.xlabel(f"CP1 ({varianza_explicada[0]*100:.1f}%)"); plt.ylabel(f"CP2 ({varianza_explicada[1]*100:.1f}%)")
                plt.title("Biplot (CP1 vs CP2)"); plt.axhline(0, color='black', lw=0.5); plt.axvline(0, color='black', lw=0.5); plt.grid(True, ls='--', alpha=0.7)
                biplot_path = self._save_plot_to_temp("biplot.png"); plt.close()
                if biplot_path: img_biplot = tk.PhotoImage(file=biplot_path); self.biplot_label_img.configure(image=img_biplot); self.image_references.append(img_biplot)
            else: self.biplot_label_img.configure(text="No hay suficientes componentes para Biplot.")

            plt.figure(figsize=(max(8, loadings.shape[1]*1.2), max(6, len(display_vars)*0.5)))
            sns.heatmap(df_load, annot=True, cmap="coolwarm", center=0, fmt=".2f", linewidths=.5)
            plt.title("Heatmap de Cargas Factoriales"); plt.xticks(rotation=45, ha="right"); plt.yticks(rotation=0)
            heatmap_path = self._save_plot_to_temp("heatmap.png"); plt.close()
            if heatmap_path: img_heatmap = tk.PhotoImage(file=heatmap_path); self.heatmap_label_img.configure(image=img_heatmap); self.image_references.append(img_heatmap)
            
            messagebox.showinfo("PCA Completado", "Análisis de Componentes Principales finalizado.")
        except Exception as e:
            messagebox.showerror("Error en PCA", f"Ocurrió un error durante el PCA:\n{e}")
            traceback.print_exc()

    def _save_plot_to_temp(self, filename):
        try:
            temp_dir = "temp_plots_pca" # Directorio específico para esta pestaña
            if not os.path.exists(temp_dir): os.makedirs(temp_dir, exist_ok=True)
            filepath = os.path.join(temp_dir, filename)
            plt.tight_layout(); plt.savefig(filepath, bbox_inches="tight", dpi=100)
            return filepath
        except Exception as e:
            print(f"Error guardando gráfico temporal {filename}: {e}"); traceback.print_exc()
            return None

    def log(self, message, level="INFO"):
        """Placeholder para logging."""
        # En una aplicación real, esto se integraría con un sistema de log más robusto
        print(f"[{level}] PrincompTab: {message}")

if __name__ == '__main__':
    root = tk.Tk()
    root.title("Test Pestaña PCA")
    # Para probar el scroll, necesitamos un frame padre que no se expanda infinitamente
    main_frame_for_test = ttk.Frame(root)
    main_frame_for_test.pack(expand=True, fill='both')
    pca_tab = PrincompTab(main_frame_for_test)
    pca_tab.pack(expand=True, fill='both')
    root.mainloop()