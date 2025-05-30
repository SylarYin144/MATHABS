#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import tkinter as tk
from tkinter import ttk

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
except ImportError as e:
    print("Error al importar uno o más módulos:", e)
    sys.exit(1)

class MainApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Proyecto FEP v2.01.02")
        self.geometry("1200x800")

        style = ttk.Style(self)
        available_themes = style.theme_names()
        if 'clam' in available_themes:
            style.theme_use('clam')

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

        # Pestaña : Calculo de Muestra
        self.sample_size_calculator_tab = SampleSizeCalculatorTab(self.notebook, main_app_instance=self)
        self.notebook.add(self.sample_size_calculator_tab, text="Cálculo de Muestra")

        self.about_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.about_tab, text="About")
        about_label = ttk.Label(self.about_tab, text="Desarrollado por: César Misael Cerecedo Zapata\nVersión: 2.01.02", justify=tk.LEFT, padding=(10, 10))
        about_label.pack(anchor="nw", padx=10, pady=10)


if __name__ == "__main__":
    app = MainApp()
    app.mainloop()
