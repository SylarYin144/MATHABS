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
    from MATLAB_general_charts import GeneralChartsApp
    from MATLAB_combined_analysis import CombinedAnalysisTab
    from MATLAB_sample_size_calculator import SampleSizeCalculatorTab
    from scientific_calculator import ScientificCalculatorTab # Nueva importación
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

        self.charts_tab = GeneralChartsApp(self.notebook, main_app_instance=self)
        self.notebook.add(self.charts_tab, text="Gráficas")

        self.combined_analysis_tab = CombinedAnalysisTab(self.notebook, main_app_instance=self)
        self.notebook.add(self.combined_analysis_tab, text="Análisis y Gráficos")

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

    def update_global_styles(self, font_family, font_size, font_color):
        """Aplica los estilos de fuente y color a todos los widgets ttk y a matplotlib."""
        # Aplicar a widgets ttk
        self.style.configure('.', font=(font_family, font_size), foreground=font_color)
        self.style.configure('TNotebook.Tab', font=(font_family, font_size + 1, 'bold'), padding=[5, 2])
        self.style.configure('TLabelframe.Label', font=(font_family, font_size, 'bold'), foreground=font_color)

        # Aplicar solo la familia de fuente a Matplotlib
        try:
            plt.rcParams['font.family'] = font_family
        except Exception as e:
            print(f"Error al aplicar la fuente '{font_family}' a matplotlib: {e}")

        # Estilo para Treeview
        self.style.configure('Treeview', font=(font_family, font_size))
        self.style.configure('Treeview.Heading', font=(font_family, font_size, 'bold'))

        # Actualizar fuentes en widgets no-ttk
        if hasattr(self, 'cox_tab') and hasattr(self.cox_tab, 'update_font_styles'):
            self.cox_tab.update_font_styles(font_family, font_size)
        if hasattr(self, 'regresiones_tab') and hasattr(self.regresiones_tab, 'update_font_styles'):
            self.regresiones_tab.update_font_styles(font_family, font_size)

        self.title(f"Proyecto FEP v2.01.02 - {font_family}")
        self.save_config(font_family, font_size, font_color)

    def save_config(self, font_family, font_size, font_color):
        """Guarda la configuración de apariencia en un archivo JSON."""
        config = {
            "font_family": font_family,
            "font_size": font_size,
            "font_color": font_color
        }
        try:
            with open("config.json", "w") as f:
                json.dump(config, f)
        except Exception as e:
            print(f"Error guardando configuración: {e}")

    def load_config(self):
        """Carga la configuración de apariencia desde un archivo JSON."""
        try:
            with open("config.json", "r") as f:
                config = json.load(f)
                font_family = config.get("font_family", "Palatino Linotype")
                font_size = config.get("font_size", 10)
                font_color = config.get("font_color", "black")
                self.update_global_styles(font_family, font_size, font_color)
        except FileNotFoundError:
            # Si no hay archivo de config, usa los valores por defecto.
            pass
        except Exception as e:
            print(f"Error cargando configuración: {e}")


if __name__ == "__main__":
    app = MainApp()
    app.mainloop()
