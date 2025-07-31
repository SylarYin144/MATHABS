#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import tkinter as tk
from tkinter import ttk
import json
import matplotlib.pyplot as plt

# Asegurarse de que el directorio actual esté en el PYTHONPATH
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Importar las pestañas existentes y las nuevas
try:
    from MATLAB_data_filter import DataFilterTab
    from MATLAB_regresiones import RegresionesTab
    from matlab_survival_analysis import SurvivalAnalysisTab
    from matlab_tablasCat import TablasCat
    from MATLAB_graficaqq import GraficaQQ
    from MATLAB_map import MapTab
    from MATLAB_cox import CoxModelingApp
    from MATLAB_mixmodel import MixModelTab
    from MATLAB_princomp import PrincompTab
    from MATLAB_logistic_regression import LogisticRegressionTab
    from MATLAB_graficas_test import GraficasTestTab
    from MATLAB_sample_size_calculator import SampleSizeCalculatorTab
    from scientific_calculator import ScientificCalculatorTab
    from graphing_calculator import GraphingCalculatorTab # Nueva importación
    from MATLAB_appearance import AppearanceTab
except ImportError as e:
    print("Error al importar uno o más módulos:", e)
    sys.exit(1)

class MainApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Proyecto FEP v2.01.02")
        self.geometry("1200x800")

        self.style = ttk.Style(self)
        available_themes = self.style.theme_names()
        if 'clam' in available_themes:
            self.style.theme_use('clam')

        self.load_config()

        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True, padx=10, pady=10)

        self.graficas_test_tab = GraficasTestTab(self.notebook, main_app_instance=self)
        self.notebook.add(self.graficas_test_tab, text="Gráficas Test")

        self.data_filter_tab = DataFilterTab(self.notebook)
        self.notebook.add(self.data_filter_tab, text="Filtro de Datos")

        self.regresiones_tab = RegresionesTab(self.notebook)
        self.notebook.add(self.regresiones_tab, text="Regresiones")

        self.survival_tab = SurvivalAnalysisTab(self.notebook)
        self.notebook.add(self.survival_tab, text="Supervivencia")

        self.tablas_cat_tab = TablasCat(self.notebook)
        self.notebook.add(self.tablas_cat_tab, text="Tablas")

        self.grafica_qq_tab = GraficaQQ(self.notebook)
        self.notebook.add(self.grafica_qq_tab, text="QQ")

        self.map_tab = MapTab(self.notebook)
        self.notebook.add(self.map_tab, text="Mapa")

        self.cox_tab = CoxModelingApp(self.notebook)
        self.notebook.add(self.cox_tab, text="Cox")

        self.mix_model_tab = MixModelTab(self.notebook)
        self.notebook.add(self.mix_model_tab, text="Mixtos")

        self.princomp_tab = PrincompTab(self.notebook)
        self.notebook.add(self.princomp_tab, text="PCA")

        self.logistic_tab = LogisticRegressionTab(self.notebook)
        self.notebook.add(self.logistic_tab, text="Logística")

        # Pestaña : Calculadora Científica
        self.calculator_tab = ScientificCalculatorTab(self.notebook)
        self.notebook.add(self.calculator_tab, text="Calculadora Científica")

        # Pestaña : Calculadora Gráfica
        self.graphing_calculator_tab = GraphingCalculatorTab(self.notebook)
        self.notebook.add(self.graphing_calculator_tab, text="Calculadora Gráfica")

        # Pestaña : Calculo de Muestra
        self.sample_size_calculator_tab = SampleSizeCalculatorTab(self.notebook, main_app_instance=self)
        self.notebook.add(self.sample_size_calculator_tab, text="Cálculo de Muestra")

        # Pestaña de Apariencia
        self.appearance_tab = AppearanceTab(self.notebook, self)
        self.notebook.add(self.appearance_tab, text="Apariencia")

        self.about_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.about_tab, text="About")
        about_label = ttk.Label(self.about_tab, text="Desarrollado por: César Misael Cerecedo Zapata\nVersión: 2.01.02", justify=tk.LEFT, padding=(10, 10))
        about_label.pack(anchor="nw", padx=10, pady=10)

    def update_global_styles(self, styles):
        """Aplica la configuración de estilos detallada a toda la aplicación."""
        font_family = styles.get('global_font_family', 'Arial')

        # --- Estilos de Pestañas (Tabs) ---
        tab_style = styles.get('tab', {})
        self.style.configure('TNotebook.Tab',
                             font=(font_family, tab_style.get('font_size', 10), tab_style.get('font_weight', 'normal')),
                             foreground=tab_style.get('fg_color', 'black'),
                             background=tab_style.get('bg_color', '#d9d9d9'),
                             padding=[5, 2])

        selected_tab_style = styles.get('selected_tab', {})
        self.style.map('TNotebook.Tab',
                       foreground=[('selected', selected_tab_style.get('fg_color', 'black'))],
                       background=[('selected', selected_tab_style.get('bg_color', '#d9d9d9'))])

        # --- Estilo de Botones ---
        button_style = styles.get('button', {})
        self.style.configure('TButton',
                             font=(font_family, button_style.get('font_size', 10), button_style.get('font_weight', 'normal')),
                             foreground=button_style.get('fg_color', 'black'),
                             background=button_style.get('bg_color', '#d9d9d9'))

        # --- Estilo de Campos de Texto (Entry) ---
        entry_style = styles.get('entry', {})
        self.style.configure('TEntry',
                             font=(font_family, entry_style.get('font_size', 10), entry_style.get('font_weight', 'normal')),
                             foreground=entry_style.get('fg_color', 'black'))
        # Note: 'background' for TEntry is handled by 'fieldbackground'
        self.style.configure('TEntry', fieldbackground=entry_style.get('bg_color', 'white'))


        # --- Otros estilos (pueden ser configurados también si se desea) ---
        self.style.configure('TLabelframe.Label', font=(font_family, 10, 'bold'))
        self.style.configure('Treeview', font=(font_family, 10))
        self.style.configure('Treeview.Heading', font=(font_family, 10, 'bold'))

        # Aplicar a Matplotlib
        try:
            plt.rcParams['font.family'] = font_family
        except Exception as e:
            print(f"No se pudo aplicar la fuente '{font_family}' a matplotlib: {e}")

        self.title(f"Proyecto FEP v2.01.02 - {font_family}")
        self.save_config(styles)

    def save_config(self, styles):
        """Guarda la configuración de apariencia detallada en un archivo JSON."""
        try:
            with open("config.json", "w") as f:
                json.dump(styles, f, indent=4)
        except Exception as e:
            print(f"Error guardando configuración: {e}")

    def load_config(self):
        """Carga la configuración de apariencia detallada desde un archivo JSON."""
        try:
            with open("config.json", "r") as f:
                styles = json.load(f)
                self.update_global_styles(styles)
                # Cargar estilos en la pestaña de apariencia
                if hasattr(self, 'appearance_tab') and hasattr(self.appearance_tab, 'load_styles'):
                    self.appearance_tab.load_styles(styles)
        except (FileNotFoundError, json.JSONDecodeError):
            # Si no hay archivo o está corrupto, no hacer nada (se usarán los defaults)
            pass
        except Exception as e:
            print(f"Error cargando configuración: {e}")


if __name__ == "__main__":
    app = MainApp()
    app.mainloop()
