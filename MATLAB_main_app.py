import matplotlib
matplotlib.use("TkAgg")

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import ctypes
import traceback
import tkinter as tk
from tkinter import ttk, messagebox
import json
import copy
import pandas as pd
from collections import OrderedDict, defaultdict
import matplotlib.pyplot as plt

try:
    from PIL import Image, ImageTk
except Exception:
    Image = None
    ImageTk = None

APP_DISPLAY_NAME = "Mathabs 1.00"
APP_WINDOWS_APP_ID = "Mathabs.MainApp"
APP_ICON_TIF_NAME = "ICONO.tif"
APP_ICON_ICO_NAME = "ICONO.ico"

DEFAULT_APPEARANCE_SETTINGS = {
    "global_font_family": "Arial",
    "tab": {
        "font_size": 10,
        "font_weight": "normal",
        "fg_color": "#000000",
        "bg_color": "#d9d9d9"
    },
    "selected_tab": {
        "fg_color": "#000000",
        "bg_color": "#f5f5f5"
    },
    "button": {
        "font_size": 10,
        "font_weight": "normal",
        "fg_color": "#000000",
        "bg_color": "#d9d9d9"
    },
    "entry": {
        "font_size": 10,
        "font_weight": "normal",
        "fg_color": "#000000",
        "bg_color": "#ffffff"
    }
}

# Asegurarse de que el directorio actual esté en el PYTHONPATH
# Cuando se ejecuta como .exe empaquetado con PyInstaller, los archivos
# bundled (ícono, etc.) están en sys._MEIPASS, mientras que archivos de
# usuario (config.json, recent_workfiles.json) se buscan junto al .exe.
if getattr(sys, 'frozen', False):
    # Ejecutándose como .exe empaquetado
    _bundle_dir = sys._MEIPASS            # archivos bundled (iconos, etc.)
    _app_dir = os.path.dirname(sys.executable)  # junto al .exe (config, etc.)
    current_dir = _bundle_dir
    if _app_dir not in sys.path:
        sys.path.insert(0, _app_dir)
    if _bundle_dir not in sys.path:
        sys.path.insert(0, _bundle_dir)
    # Establecer CWD al directorio del .exe para que config.json se guarde ahí
    os.chdir(_app_dir)
else:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.append(current_dir)

# Importar las pestañas existentes y las nuevas
try:

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
    from scientific_calculator import ScientificCalculatorTab
    from graphing_calculator import GraphingCalculatorTab # Nueva importación
    from MATLAB_appearance import AppearanceTab
    from MATLAB_linear_regression import LinearRegressionApp
    from MATLAB_add_variables import AddVariablesTab
    from MATLAB_aft import AFTsTab
    from MATLAB_rsf import RSFTab
    from MATLAB_rf import RFTab
    from workflow_tab import WorkflowTab
    from MATLAB_data_editor import DataEditorTab, create_data_editor_tab
except ImportError as e:
    print("Error al importar uno o más módulos:", e)
    sys.exit(1)

class MainApp(tk.Tk):
    def __init__(self):
        self._set_windows_app_user_model_id()
        super().__init__()
        self.title(APP_DISPLAY_NAME)
        self.geometry("1200x800")
        self._window_icon_image = None
        self._window_hicon = None
        self._configure_window_identity()

        self.style = ttk.Style(self)
        available_themes = self.style.theme_names()
        if 'vista' in available_themes:
            self.style.theme_use('vista')
        elif 'xpnative' in available_themes:
            self.style.theme_use('xpnative')
        elif 'clam' in available_themes:
            self.style.theme_use('clam')

        self.loaded_styles = None
        self._close_images = {}
        self._closable_tabs = set()
        self._closable_style_ready = False
        self._init_closable_tab_style()
        self.modules_catalog = self._define_modules()
        self.module_instance_counters = defaultdict(int)
        self.tab_metadata = {}
        self.appearance_tabs = []
        self.embedded_appearance_tab = None
        self.shared_dataset = None
        self.shared_filtered_dataset = None
        self.shared_filter_summary = []
        self.shared_metadata = {}
        self.shared_source_widget = None
        self.data_filter_tab = None
        self._table_layouts = {}
        self._table_layout_sources = {}
        self._module_states = {}

        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True, padx=10, pady=10)
        self.notebook.bind("<ButtonPress-1>", self._handle_tab_click, add="+")

        self.launcher_container = ttk.Frame(self.notebook)
        self.launcher_canvas = tk.Canvas(self.launcher_container, borderwidth=0, highlightthickness=0)
        self.launcher_scrollbar = ttk.Scrollbar(self.launcher_container, orient="vertical", command=self.launcher_canvas.yview)
        self.launcher_inner_frame = ttk.Frame(self.launcher_canvas, padding=20)

        self.launcher_inner_frame.bind(
            "<Configure>",
            lambda event: self.launcher_canvas.configure(scrollregion=self.launcher_canvas.bbox("all"))
        )

        self.launcher_canvas.create_window((0, 0), window=self.launcher_inner_frame, anchor="nw")
        self.launcher_canvas.configure(yscrollcommand=self.launcher_scrollbar.set)

        self.launcher_canvas.pack(side="left", fill="both", expand=True)
        self.launcher_scrollbar.pack(side="right", fill="y")

        self.launcher_frame = self.launcher_inner_frame
        self.notebook.add(self.launcher_container, text="Inicio")
        self._closable_tabs.discard(self.notebook.tabs()[0])
        self._bind_launcher_mousewheel()
        self._build_launcher_ui()

        self.tab_context_menu = tk.Menu(self, tearoff=0)
        self.tab_context_menu.add_command(label="Cerrar pestaña", command=self._close_context_tab)
        self.notebook.bind("<Button-3>", self._show_tab_context_menu)
        self.bind_all("<Control-w>", self._ctrl_w_close_current_tab)
        self.protocol("WM_DELETE_WINDOW", self._on_app_close)

        self.options_tab = ttk.Frame(self.notebook, padding=20)
        self.notebook.add(self.options_tab, text="Opciones")
        if len(self.notebook.tabs()) >= 2:
            self._closable_tabs.discard(self.notebook.tabs()[1])
        self._build_options_tab()

        self.load_config()

        workfile_widget = self.open_module_tab("add_variables")
        if workfile_widget is not None:
            self.notebook.select(self.launcher_container)

    @classmethod
    def _resolve_app_icon_paths(cls):
        tif_path = os.path.join(current_dir, APP_ICON_TIF_NAME)
        ico_path = os.path.join(current_dir, APP_ICON_ICO_NAME)
        return tif_path, ico_path

    @staticmethod
    def _set_windows_app_user_model_id():
        if sys.platform.startswith("win"):
            try:
                ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(APP_WINDOWS_APP_ID)
            except Exception:
                pass

    def _apply_windows_hicon(self, ico_path):
        if not sys.platform.startswith("win") or not os.path.exists(ico_path):
            return

        try:
            self.update_idletasks()
            image_flags = 0x00000010 | 0x00000040  # LR_LOADFROMFILE | LR_DEFAULTSIZE
            image_type_icon = 1  # IMAGE_ICON
            hicon = ctypes.windll.user32.LoadImageW(None, ico_path, image_type_icon, 0, 0, image_flags)
            if hicon:
                wm_seticon = 0x0080
                icon_small = 0
                icon_big = 1
                ctypes.windll.user32.SendMessageW(self.winfo_id(), wm_seticon, icon_small, hicon)
                ctypes.windll.user32.SendMessageW(self.winfo_id(), wm_seticon, icon_big, hicon)
                self._window_hicon = hicon
        except Exception:
            pass

    def _configure_window_identity(self):
        self._set_windows_app_user_model_id()

        tif_path, ico_path = self._resolve_app_icon_paths()

        if Image is not None and os.path.exists(tif_path):
            try:
                with Image.open(tif_path) as icon_image:
                    rgba_icon = icon_image.convert("RGBA")
                    try:
                        rgba_icon.save(ico_path, format="ICO", sizes=[(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
                    except Exception:
                        pass

                    try:
                        self._window_icon_image = ImageTk.PhotoImage(rgba_icon)
                        self.iconphoto(True, self._window_icon_image)
                    except Exception:
                        self._window_icon_image = None
            except Exception as exc:
                print(f"No se pudo cargar el icono TIFF de la app: {exc}")

        if os.path.exists(ico_path):
            try:
                self.iconbitmap(default=ico_path)
            except Exception:
                pass
            self._apply_windows_hicon(ico_path)

    def apply_saved_table_layout(self, layout_key, col_config):
        if not layout_key or not isinstance(col_config, dict):
            return col_config

        saved_layouts = self.__dict__.get("_table_layouts", {}) or {}
        saved_config = saved_layouts.get(layout_key, {}) if isinstance(saved_layouts, dict) else {}
        if not isinstance(saved_config, dict):
            return col_config

        for col_id, cfg in col_config.items():
            saved_entry = saved_config.get(col_id)
            if not isinstance(saved_entry, dict):
                continue
            if "visible" in saved_entry:
                cfg["visible"] = bool(saved_entry.get("visible"))
            saved_width = saved_entry.get("width")
            if isinstance(saved_width, (int, float)) and saved_width > 0:
                cfg["width"] = int(saved_width)

        return col_config

    def register_table_layout_source(self, layout_key, col_config, treeview):
        if not layout_key or not isinstance(col_config, dict) or treeview is None:
            return
        table_sources = self.__dict__.setdefault("_table_layout_sources", {})
        table_sources[layout_key] = {"config": col_config, "treeview": treeview}

    def persist_table_layout(self, layout_key, col_config, treeview=None):
        if not layout_key or not isinstance(col_config, dict):
            return

        normalized_layout = {}
        for col_id, cfg in col_config.items():
            is_visible = bool(cfg.get("visible", True))
            width_value = cfg.get("width", 80)
            if treeview is not None and is_visible:
                try:
                    current_width = int(treeview.column(col_id, option="width"))
                    if current_width > 0:
                        width_value = current_width
                except Exception:
                    pass
            normalized_layout[col_id] = {
                "visible": is_visible,
                "width": max(int(width_value) if isinstance(width_value, (int, float)) else 80, 24),
                "heading": cfg.get("heading", str(col_id)),
            }

        table_layouts = self.__dict__.setdefault("_table_layouts", {})
        table_layouts[layout_key] = normalized_layout

        styles_to_save = copy.deepcopy(self.__dict__.get("loaded_styles") or DEFAULT_APPEARANCE_SETTINGS)
        self.save_config(styles_to_save)

    def persist_registered_table_layouts(self):
        table_sources = self.__dict__.get("_table_layout_sources", {}) or {}
        for layout_key, payload in table_sources.items():
            if not isinstance(payload, dict):
                continue
            self.persist_table_layout(layout_key, payload.get("config"), payload.get("treeview"))

    def _on_app_close(self):
        if self._closable_tabs:
            from tkinter import messagebox
            if not messagebox.askyesno(
                "Confirmar salida",
                f"Hay {len(self._closable_tabs)} pestaña(s) abierta(s).\n¿Deseas salir de verdad?",
                default="no",
            ):
                return
        try:
            self.persist_registered_table_layouts()
        except Exception as exc:
            print(f"Error guardando configuración de tablas al cerrar: {exc}")
        try:
            self._collect_open_module_states()
            self.save_config(copy.deepcopy(self.loaded_styles or DEFAULT_APPEARANCE_SETTINGS))
        except Exception as exc:
            print(f"Error guardando estado de módulos al cerrar: {exc}")
        self.destroy()

    def _define_modules(self):
        catalog = OrderedDict([
            ("add_variables", {
                "label": "Archivo de Trabajo",
                "tab_title": "Archivo de Trabajo",
                "factory": AddVariablesTab,
                "category": "Datos y Transformaciones",
                "pass_main_app": True,
                "attr_name": "workfile_tab",
                "singleton": True,
                "register_as_data_source": True,
                "closable": False,
                "show_in_launcher": False
            }),
            ("data_editor", {
                "label": "Editor de Datos",
                "tab_title": "📝 Editor de Datos",
                "factory": DataEditorTab,
                "category": "Datos y Transformaciones",
                "pass_main_app": True,
                "attr_name": "data_editor_tab"
            }),
            ("cox", {
                "label": "Modelo Cox",
                "tab_title": "Cox",
                "factory": CoxModelingApp,
                "category": "Modelado",
                "pass_main_app": True,
                "attr_name": "cox_tab"
            }),
            ("regresiones", {
                "label": "Regresiones",
                "tab_title": "Regresiones",
                "factory": RegresionesTab,
                "category": "Modelado",
                "pass_main_app": True,
                "attr_name": "regresiones_tab"
            }),
            ("linear_regression", {
                "label": "Regresión Lineal",
                "tab_title": "Regresión Lineal",
                "factory": LinearRegressionApp,
                "category": "Modelado",
                "pass_main_app": True,
                "attr_name": "linear_regression_tab"
            }),
            ("logistic", {
                "label": "Regresión Logística",
                "tab_title": "Logística",
                "factory": LogisticRegressionTab,
                "category": "Modelado",
                "pass_main_app": True,
                "attr_name": "logistic_tab"
            }),
            ("survival", {
                "label": "Supervivencia",
                "tab_title": "Supervivencia",
                "factory": SurvivalAnalysisTab,
                "category": "Modelado",
                "pass_main_app": True,
                "attr_name": "survival_tab"
            }),
            ("aft", {
                "label": "Modelos AFT",
                "tab_title": "AFTs",
                "factory": AFTsTab,
                "category": "Modelado",
                "pass_main_app": True,
                "attr_name": "aft_tab"
            }),
            ("rsf", {
                "label": "Bosque RSF",
                "tab_title": "RSF",
                "factory": RSFTab,
                "category": "Modelado",
                "pass_main_app": True,
                "attr_name": "rsf_tab"
            }),
            ("rf", {
                "label": "Random Forest (RFC/RFR)",
                "tab_title": "RF",
                "factory": RFTab,
                "category": "Modelado",
                "pass_main_app": True,
                "main_app_kwarg": "main_app",
                "attr_name": "rf_tab"
            }),
            ("mix_model", {
                "label": "Modelos Mixtos",
                "tab_title": "Mixtos",
                "factory": MixModelTab,
                "category": "Modelado",
                "pass_main_app": True,
                "attr_name": "mix_model_tab"
            }),
            ("princomp", {
                "label": "PCA",
                "tab_title": "PCA",
                "factory": PrincompTab,
                "category": "Modelado",
                "pass_main_app": True,
                "attr_name": "princomp_tab"
            }),
            ("combined_analysis", {
                "label": "Análisis y Gráficos",
                "tab_title": "Análisis y Gráficos",
                "factory": CombinedAnalysisTab,
                "category": "Modelado",
                "pass_main_app": True,
                "attr_name": "combined_analysis_tab"
            }),
            ("general_charts", {
                "label": "Gráficas Generales",
                "tab_title": "Gráficas",
                "factory": GeneralChartsApp,
                "category": "Visualización",
                "pass_main_app": True,
                "attr_name": "charts_tab"
            }),
            ("tablas", {
                "label": "Tablas",
                "tab_title": "Tablas",
                "factory": TablasCat,
                "category": "Visualización",
                "pass_main_app": True,
                "attr_name": "tablas_cat_tab"
            }),
            ("grafica_qq", {
                "label": "Comparación de Medias",
                "tab_title": "Comparación Medias",
                "factory": GraficaQQ,
                "category": "Visualización",
                "attr_name": "grafica_qq_tab"
            }),
            ("map", {
                "label": "Mapa",
                "tab_title": "Mapa",
                "factory": MapTab,
                "category": "Visualización",
                "attr_name": "map_tab"
            }),
            ("scientific_calc", {
                "label": "Calculadora Científica",
                "tab_title": "Calculadora Científica",
                "factory": ScientificCalculatorTab,
                "category": "Herramientas",
                "attr_name": "calculator_tab"
            }),
            ("graphing_calc", {
                "label": "Calculadora Gráfica",
                "tab_title": "Calculadora Gráfica",
                "factory": GraphingCalculatorTab,
                "category": "Herramientas",
                "attr_name": "graphing_calculator_tab"
            }),
            ("color_tools", {
                "label": "Herramientas de Color",
                "tab_title": "Colores",
                "factory": __import__('color_tools_tab').ColorToolsTab,
                "category": "Herramientas",
                "closable": True,
                "pass_main_app": True,
                "attr_name": "color_tools_tab"
            }),
            ("sample_size", {
                "label": "Cálculo de Muestra",
                "tab_title": "Cálculo de Muestra",
                "factory": SampleSizeCalculatorTab,
                "category": "Herramientas",
                "pass_main_app": True,
                "attr_name": "sample_size_calculator_tab"
            }),
            ("appearance", {
                "label": "Apariencia",
                "tab_title": "Apariencia",
                "factory": AppearanceTab,
                "category": "Herramientas",
                "pass_main_app": True,
                "main_app_kwarg": "app_instance",
                "attr_name": "appearance_tab",
                "show_in_launcher": False
            }),
                ("workflow", {
                    "label": "Flujo de Trabajo",
                    "tab_title": "Flujo de Trabajo",
                    "factory": WorkflowTab,
                    "category": "Herramientas",
                    "pass_main_app": True,
                    "attr_name": "workflow_tab",
                    "singleton": True,
                    "description": "Resumen lógico de pasos, dependencias y pestañas clave para el análisis."
                }),
        ])
        return catalog

    def _build_launcher_ui(self):
        for child in self.launcher_frame.winfo_children():
            child.destroy()

        ttk.Label(
            self.launcher_frame,
            text="Selecciona un módulo para abrirlo en una nueva pestaña (puedes abrir múltiples instancias):",
            wraplength=800,
            justify=tk.LEFT
        ).pack(anchor="nw", pady=(0, 15))

        categorized = OrderedDict()
        for key, info in self.modules_catalog.items():
            if not info.get("show_in_launcher", True):
                continue
            category = info.get("category", "Módulos")
            categorized.setdefault(category, []).append((key, info))

        for category, entries in categorized.items():
            group_frame = ttk.LabelFrame(self.launcher_frame, text=category, padding=10)
            group_frame.pack(fill="both", expand=True, pady=8)

            buttons_frame = ttk.Frame(group_frame)
            buttons_frame.pack(fill="both", expand=True)

            max_columns = min(3, max(1, len(entries)))
            for index, (module_key, info) in enumerate(entries):
                btn = ttk.Button(
                    buttons_frame,
                    text=info["label"],
                    command=lambda k=module_key: self.open_module_tab(k)
                )
                row = index // max_columns
                col = index % max_columns
                btn.grid(row=row, column=col, padx=6, pady=6, sticky="nsew")

            for col_index in range(max_columns):
                buttons_frame.columnconfigure(col_index, weight=1)

    def _bind_launcher_mousewheel(self):
        def _on_mousewheel(event):
            if event.delta:
                self.launcher_canvas.yview_scroll(int(-event.delta / 120), "units")
            elif event.num in (4, 5):
                direction = -1 if event.num == 4 else 1
                self.launcher_canvas.yview_scroll(direction, "units")
            return "break"

        def _bind_to_mousewheel(_event):
            self.launcher_canvas.bind_all("<MouseWheel>", _on_mousewheel)
            self.launcher_canvas.bind_all("<Button-4>", _on_mousewheel)
            self.launcher_canvas.bind_all("<Button-5>", _on_mousewheel)

        def _unbind_from_mousewheel(_event):
            self.launcher_canvas.unbind_all("<MouseWheel>")
            self.launcher_canvas.unbind_all("<Button-4>")
            self.launcher_canvas.unbind_all("<Button-5>")

        self.launcher_canvas.bind("<Enter>", _bind_to_mousewheel)
        self.launcher_canvas.bind("<Leave>", _unbind_from_mousewheel)
        self.launcher_inner_frame.bind("<Enter>", _bind_to_mousewheel)
        self.launcher_inner_frame.bind("<Leave>", _unbind_from_mousewheel)

    def _build_options_tab(self):
        for child in self.options_tab.winfo_children():
            child.destroy()

        header = ttk.Label(
            self.options_tab,
            text="Desarrollado por: César Misael Cerecedo Zapata\nVersión: 2.01.02",
            justify=tk.LEFT,
            padding=(10, 10)
        )
        header.pack(anchor="nw", padx=10, pady=(10, 5))

        defaults_frame = ttk.LabelFrame(self.options_tab, text="Configuraciones por defecto", padding=10)
        defaults_frame.pack(fill="x", padx=10, pady=(0, 10))

        defaults_text_lines = [
            f"Fuente global: {DEFAULT_APPEARANCE_SETTINGS['global_font_family']}",
            f"Pestañas: tamaño {DEFAULT_APPEARANCE_SETTINGS['tab']['font_size']} | color texto {DEFAULT_APPEARANCE_SETTINGS['tab']['fg_color']} | fondo {DEFAULT_APPEARANCE_SETTINGS['tab']['bg_color']}",
            f"Pestaña seleccionada: color texto {DEFAULT_APPEARANCE_SETTINGS['selected_tab']['fg_color']} | fondo {DEFAULT_APPEARANCE_SETTINGS['selected_tab']['bg_color']}",
            f"Botones: tamaño {DEFAULT_APPEARANCE_SETTINGS['button']['font_size']} | color texto {DEFAULT_APPEARANCE_SETTINGS['button']['fg_color']}",
            f"Entradas: tamaño {DEFAULT_APPEARANCE_SETTINGS['entry']['font_size']} | color texto {DEFAULT_APPEARANCE_SETTINGS['entry']['fg_color']}"
        ]

        for line in defaults_text_lines:
            ttk.Label(defaults_frame, text=line, anchor="w").pack(fill="x", pady=1)

        ttk.Label(
            defaults_frame,
            text="Estos valores se aplican automáticamente al iniciar la aplicación y puedes ajustarlos en el panel de Apariencia inferior.",
            anchor="w",
            wraplength=600
        ).pack(fill="x", pady=(6, 0))

        appearance_frame = ttk.LabelFrame(self.options_tab, text="Personalizar apariencia", padding=10)
        appearance_frame.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        if self.embedded_appearance_tab and self.embedded_appearance_tab in self.appearance_tabs:
            self.appearance_tabs.remove(self.embedded_appearance_tab)

        self.embedded_appearance_tab = AppearanceTab(appearance_frame, app_instance=self)
        self.embedded_appearance_tab.pack(fill="both", expand=True)
        self.appearance_tabs.append(self.embedded_appearance_tab)
        if self.loaded_styles and hasattr(self.embedded_appearance_tab, "load_styles"):
            try:
                self.embedded_appearance_tab.load_styles(self.loaded_styles)
            except Exception:
                pass

        credits_label = ttk.Label(
            self.options_tab,
            text="Mathabs 1.00 - Plataforma de análisis estadístico para supervivencia y modelos clínicos.",
            justify=tk.LEFT,
            wraplength=700,
            padding=(10, 5)
        )
        credits_label.pack(anchor="nw", padx=10, pady=(0, 10))

    def _init_closable_tab_style(self):
        if self._closable_style_ready:
            return

        normal_image = self._create_close_image("#505050")
        active_image = self._create_close_image("#c62828")
        self._close_images["normal"] = normal_image
        self._close_images["active"] = active_image

        try:
            self.style.element_create(
                "ClosableNotebook.close",
                "image",
                normal_image,
                ("active", active_image),
                ("pressed", active_image)
            )
        except tk.TclError:
            pass

        self.style.layout(
            "ClosableNotebook.Tab",
            [
                ("Notebook.tab", {
                    "sticky": "nswe",
                    "children": [
                        ("Notebook.padding", {
                            "side": "top",
                            "sticky": "nswe",
                            "children": [
                                ("Notebook.focus", {
                                    "side": "top",
                                    "sticky": "nswe",
                                    "children": [
                                        ("Notebook.label", {"side": "left", "sticky": ""}),
                                        ("ClosableNotebook.close", {"side": "left", "sticky": ""})
                                    ]
                                })
                            ]
                        })
                    ]
                })
            ]
        )

        self.style.configure("ClosableNotebook.Tab", padding=(10, 4, 18, 4))
        self._closable_style_ready = True

    def _create_close_image(self, color):
        size = 14
        image = tk.PhotoImage(master=self, width=size, height=size)
        raw_background = self.style.lookup("TNotebook", "background") or self.cget("background") or "#f0f0f0"
        background = self._normalize_color(raw_background)

        for x in range(size):
            for y in range(size):
                image.put(background, (x, y))

        thickness = 2
        for offset in range(size):
            for delta in range(-thickness, thickness + 1):
                y1 = offset + delta
                y2 = (size - 1 - offset) + delta
                if 0 <= offset < size and 0 <= y1 < size:
                    image.put(color, (offset, y1))
                if 0 <= offset < size and 0 <= y2 < size:
                    image.put(color, (offset, y2))

        return image

    def _normalize_color(self, color_value):
        if not color_value:
            return "#f0f0f0"
        if isinstance(color_value, str) and color_value.startswith("#"):
            if len(color_value) == 4:
                try:
                    r = color_value[1]
                    g = color_value[2]
                    b = color_value[3]
                    return f"#{r}{r}{g}{g}{b}{b}"
                except Exception:
                    return "#f0f0f0"
            return color_value

        try:
            r, g, b = self.winfo_rgb(color_value)
            return f"#{r // 256:02x}{g // 256:02x}{b // 256:02x}"
        except Exception:
            return "#f0f0f0"

    def _handle_tab_click(self, event):
        element = self.notebook.identify(event.x, event.y)
        if element != "close":
            return

        try:
            tab_index = self.notebook.index(f"@{event.x},{event.y}")
        except tk.TclError:
            return "break"

        tabs = self.notebook.tabs()
        if tab_index < 0 or tab_index >= len(tabs):
            return "break"

        tab_id = tabs[tab_index]
        if tab_id not in self._closable_tabs:
            return "break"

        self._close_tab(tab_id)
        return "break"

    def _show_tab_context_menu(self, event):
        try:
            tab_index = self.notebook.index(f"@{event.x},{event.y}")
        except tk.TclError:
            return

        tab_id = self.notebook.tabs()[tab_index]
        tab_widget = self.nametowidget(tab_id)
        if tab_widget in (self.launcher_container, self.options_tab):
            return

        metadata = self.tab_metadata.get(tab_id, {})
        if not metadata.get("closable", True):
            return

        self._context_tab_id = tab_id
        self.tab_context_menu.tk_popup(event.x_root, event.y_root)
        self.tab_context_menu.grab_release()

    def _close_context_tab(self):
        tab_id = getattr(self, "_context_tab_id", None)
        if tab_id:
            self._close_tab(tab_id)
            self._context_tab_id = None

    def _ctrl_w_close_current_tab(self, event):
        current_tab = self.notebook.select()
        if current_tab:
            widget = self.nametowidget(current_tab)
            if widget not in (self.launcher_container, self.options_tab):
                metadata = self.tab_metadata.get(current_tab, {})
                if metadata.get("closable", True):
                    self._close_tab(current_tab)
                return "break"

    def _close_tab(self, tab_id):
        if tab_id not in self.notebook.tabs():
            return

        metadata = self.tab_metadata.get(tab_id, {})
        if not metadata.get("closable", True):
            return

        tab_widget = self.nametowidget(tab_id)
        if tab_widget in (self.launcher_container, self.options_tab):
            return

        metadata = self.tab_metadata.pop(tab_id, metadata)
        module_key = metadata.get("module_key")
        attr_name = metadata.get("attr_name")
        self._closable_tabs.discard(tab_id)

        if module_key and hasattr(tab_widget, "get_persistent_state"):
            try:
                module_state = tab_widget.get_persistent_state()
                if module_state is not None:
                    self._module_states[module_key] = copy.deepcopy(module_state)
            except Exception as exc:
                print(f"Error guardando estado de módulo '{module_key}' al cerrar pestaña: {exc}")

        if module_key == "appearance":
            self.appearance_tabs = [tab for tab in self.appearance_tabs if tab is not tab_widget]

        if attr_name and getattr(self, attr_name, None) is tab_widget:
            delattr(self, attr_name)

        was_data_source = bool(metadata.get("register_as_data_source")) and self.shared_source_widget is tab_widget
        if was_data_source:
            self.data_filter_tab = None
            self.shared_source_widget = None

        self.notebook.forget(tab_id)
        try:
            tab_widget.destroy()
        except Exception:
            pass

        if was_data_source:
            self.update_shared_dataset(None)

    def open_module_tab(self, module_key):
        info = self.modules_catalog.get(module_key)
        if not info:
            return None

        attr_name = info.get("attr_name")
        if info.get("singleton") and attr_name:
            existing_widget = getattr(self, attr_name, None)
            try:
                exists = bool(existing_widget and existing_widget.winfo_exists())
            except Exception:
                exists = False
            if exists:
                for tab_id in self.notebook.tabs():
                    if self.nametowidget(tab_id) is existing_widget:
                        self.notebook.select(tab_id)
                        return existing_widget
            else:
                if existing_widget is not None:
                    setattr(self, attr_name, None)

        kwargs = {}
        if info.get("pass_main_app"):
            kwarg_name = info.get("main_app_kwarg", "main_app_instance")
            kwargs[kwarg_name] = self
        if info.get("extra_kwargs"):
            kwargs.update(info["extra_kwargs"])

        try:
            module_widget = info["factory"](self.notebook, **kwargs)
        except Exception as exc:
            error_text = traceback.format_exc(limit=3)
            print(f"Error abriendo módulo '{module_key}': {exc}\n{error_text}")
            try:
                messagebox.showerror(
                    "Error al abrir módulo",
                    f"No se pudo abrir '{info.get('label', module_key)}'.\n\nDetalle: {exc}",
                )
            except Exception:
                pass
            return None

        self.module_instance_counters[module_key] += 1
        instance_index = self.module_instance_counters[module_key]
        title_base = info.get("tab_title", info.get("label", module_key))
        tab_title = title_base if instance_index == 1 else f"{title_base} #{instance_index}"

        self.notebook.add(module_widget, text=tab_title)
        tab_id = self.notebook.tabs()[-1]

        is_closable = info.get("closable", True)
        if is_closable:
            if self._closable_style_ready:
                try:
                    self.notebook.tab(tab_id, style="ClosableNotebook.Tab")
                except tk.TclError:
                    # Tk builds without tab style support raise TclError; fallback to default visuals.
                    pass
            self._closable_tabs.add(tab_id)
        else:
            try:
                self.notebook.tab(tab_id, style="TNotebook.Tab")
            except tk.TclError:
                # Some Tk versions do not accept per-tab style configuration.
                pass
            self._closable_tabs.discard(tab_id)

        metadata = {
            "module_key": module_key,
            "widget": module_widget,
            "title_base": title_base,
            "instance_index": instance_index,
            "attr_name": attr_name,
            "register_as_data_source": info.get("register_as_data_source", False),
            "closable": is_closable
        }
        self.tab_metadata[tab_id] = metadata

        if attr_name:
            setattr(self, attr_name, module_widget)

        if info.get("register_as_data_source"):
            self.data_filter_tab = module_widget
            self.shared_source_widget = module_widget

        if module_key == "appearance":
            self.appearance_tabs.append(module_widget)
            if self.loaded_styles and hasattr(module_widget, "load_styles"):
                module_widget.load_styles(self.loaded_styles)

        if self.loaded_styles and hasattr(module_widget, "apply_styles"):
            module_widget.apply_styles(self.loaded_styles)

        if self.shared_dataset is not None and module_widget is not self.shared_source_widget:
            if hasattr(module_widget, "receive_shared_dataset"):
                try:
                    module_widget.receive_shared_dataset(
                        dataset=self.shared_dataset,
                        filtered_dataset=self.shared_filtered_dataset,
                        filter_summary=self.shared_filter_summary,
                        metadata=self.shared_metadata,
                        source_widget=self.shared_source_widget
                    )
                except Exception as exc:
                    print(f"Error pre-cargando dataset compartido en {type(module_widget).__name__}: {exc}")

        self._apply_saved_module_state(module_key, module_widget)

        self.notebook.select(module_widget)
        return module_widget

    def _collect_open_module_states(self):
        if not hasattr(self, "notebook"):
            return

        for tab_id in self.notebook.tabs():
            try:
                tab_widget = self.nametowidget(tab_id)
            except Exception:
                continue

            if tab_widget in (self.launcher_container, self.options_tab):
                continue

            metadata = self.tab_metadata.get(tab_id, {}) if isinstance(self.tab_metadata, dict) else {}
            module_key = metadata.get("module_key")
            if not module_key:
                continue

            if hasattr(tab_widget, "get_persistent_state"):
                try:
                    module_state = tab_widget.get_persistent_state()
                    if module_state is not None:
                        self._module_states[module_key] = copy.deepcopy(module_state)
                except Exception as exc:
                    print(f"Error recolectando estado del módulo '{module_key}': {exc}")

    def _apply_saved_module_state(self, module_key, module_widget):
        if not module_key or module_widget is None:
            return
        if not hasattr(module_widget, "load_persistent_state"):
            return

        state = None
        try:
            state = copy.deepcopy((self._module_states or {}).get(module_key))
        except Exception:
            state = None
        if state is None:
            return

        try:
            module_widget.load_persistent_state(state)
        except Exception as exc:
            print(f"Error aplicando estado persistente al módulo '{module_key}': {exc}")

    def update_all_variable_lists(self):
        """Iterates through open tabs and triggers their variable refresh routines when available."""
        candidate_methods = [
            'update_variable_lists',
            'update_variable_selectors',
            '_update_variable_selectors',
            'actualizar_controles_preproc',
            'actualizar_listas_variables'
        ]

        for tab_id in self.notebook.tabs():
            tab_widget = self.nametowidget(tab_id)
            if tab_widget in (self.launcher_container, self.options_tab):
                continue

            for method_name in candidate_methods:
                if hasattr(tab_widget, method_name):
                    method = getattr(tab_widget, method_name)
                    if callable(method):
                        try:
                            method()
                        except Exception as exc:
                            print(f"Error updating variables in tab {type(tab_widget).__name__}: {exc}")
                    break

    def update_shared_dataset(self, dataset, *, filtered_dataset=None, filter_summary=None, source_widget=None, metadata=None):
        """Actualiza el dataset compartido y notifica a las pestañas abiertas."""
        self.shared_dataset = dataset
        if dataset is None:
            self.shared_filtered_dataset = None
            self.shared_filter_summary = []
            self.shared_metadata = {}
        else:
            if filtered_dataset is not None and isinstance(filtered_dataset, pd.DataFrame):
                self.shared_filtered_dataset = filtered_dataset
            else:
                self.shared_filtered_dataset = None
            self.shared_filter_summary = list(filter_summary or [])
            self.shared_metadata = dict(metadata or {})

        if source_widget is not None:
            self.shared_source_widget = source_widget
            try:
                if self.data_filter_tab is None and source_widget.winfo_exists():
                    self.data_filter_tab = source_widget
            except Exception:
                pass

        for tab_id in self.notebook.tabs():
            tab_widget = self.nametowidget(tab_id)
            if tab_widget in (self.launcher_container, self.options_tab, source_widget):
                continue

            if hasattr(tab_widget, "receive_shared_dataset"):
                try:
                    tab_widget.receive_shared_dataset(
                        dataset=self.shared_dataset,
                        filtered_dataset=self.shared_filtered_dataset,
                        filter_summary=self.shared_filter_summary,
                        metadata=self.shared_metadata,
                        source_widget=self.shared_source_widget
                    )
                except Exception as exc:
                    print(f"Error delivering shared dataset to {type(tab_widget).__name__}: {exc}")

        self.update_all_variable_lists()

    def get_shared_dataset(self):
        return self.shared_dataset

    def get_active_dataset(self):
        return self.shared_filtered_dataset if self.shared_filtered_dataset is not None else self.shared_dataset

    def get_filter_summary(self):
        return list(self.shared_filter_summary)

    def get_shared_metadata(self):
        return dict(self.shared_metadata)

    def get_analysis_presets(self):
        if self.data_filter_tab and hasattr(self.data_filter_tab, 'get_analysis_presets'):
            try:
                return self.data_filter_tab.get_analysis_presets()
            except Exception as exc:
                print(f"Error obteniendo presets del origen: {exc}")
        return copy.deepcopy(self.shared_metadata.get('analysis_presets') or {})

    def get_analysis_preset(self, preset_key, default=None):
        presets = self.get_analysis_presets()
        if preset_key is None:
            return copy.deepcopy(default)
        return copy.deepcopy(presets.get(preset_key, default))

    def save_analysis_preset(self, preset_key, payload):
        if preset_key is None:
            return

        source = self.data_filter_tab
        if source and hasattr(source, 'update_analysis_preset'):
            source.update_analysis_preset(preset_key, payload)
            return

        presets = self.get_analysis_presets()
        if payload is None:
            presets.pop(preset_key, None)
        else:
            presets[preset_key] = copy.deepcopy(payload)
        self.shared_metadata['analysis_presets'] = presets

    def replace_analysis_presets(self, presets_mapping):
        source = self.data_filter_tab
        if source and hasattr(source, 'replace_analysis_presets'):
            source.replace_analysis_presets(presets_mapping)
            return
        self.shared_metadata['analysis_presets'] = copy.deepcopy(presets_mapping or {})

    def update_global_styles(self, styles):
        """Aplica la configuración de estilos detallada a toda la aplicación."""
        self.loaded_styles = styles.copy()
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
        # Fix: Configuring TEntry fieldbackground breaks Combobox dropdown arrow native rendering in Windows.
        # self.style.configure('TEntry', fieldbackground=entry_style.get('bg_color', 'white'))


        # --- Otros estilos (pueden ser configurados también si se desea) ---
        self.style.configure('TLabelframe.Label', font=(font_family, 10, 'bold'))
        self.style.configure('Treeview', font=(font_family, 10))
        self.style.configure('Treeview.Heading', font=(font_family, 10, 'bold'))

        # Aplicar a Matplotlib
        try:
            plt.rcParams['font.family'] = font_family
        except Exception as e:
            print(f"No se pudo aplicar la fuente '{font_family}' a matplotlib: {e}")

        self.loaded_styles = copy.deepcopy(styles)
        self.title(APP_DISPLAY_NAME)

        # Aplicar estilos a pestañas que lo soporten
        for tab_id in self.notebook.tabs():
            try:
                tab_widget = self.nametowidget(tab_id)
                if hasattr(tab_widget, 'apply_styles'):
                    tab_widget.apply_styles(styles)
            except Exception as e:
                print(f"Error aplicando estilos a la pestaña {tab_id}: {e}")

        self.save_config(styles)

    def save_config(self, styles):
        """Guarda la configuración de apariencia y la disposición de tablas en un archivo JSON."""
        try:
            self._collect_open_module_states()
            appearance_settings = copy.deepcopy(styles or DEFAULT_APPEARANCE_SETTINGS)
            payload = {
                "appearance": appearance_settings,
                "table_layouts": copy.deepcopy(self.__dict__.get("_table_layouts", {}) or {}),
                "module_states": copy.deepcopy(self.__dict__.get("_module_states", {}) or {}),
            }
            with open("config.json", "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=4, ensure_ascii=False)
        except Exception as e:
            print(f"Error guardando configuración: {e}")

    def load_config(self):
        """Carga la configuración de apariencia y las tablas desde un archivo JSON."""
        try:
            with open("config.json", "r", encoding="utf-8") as f:
                raw_config = json.load(f)

            if isinstance(raw_config, dict) and "appearance" in raw_config:
                styles = raw_config.get("appearance", DEFAULT_APPEARANCE_SETTINGS)
                self._table_layouts = copy.deepcopy(raw_config.get("table_layouts", {}) or {})
                self._module_states = copy.deepcopy(raw_config.get("module_states", {}) or {})
            else:
                styles = raw_config if isinstance(raw_config, dict) else copy.deepcopy(DEFAULT_APPEARANCE_SETTINGS)
                self._table_layouts = {}
                self._module_states = {}

            self.update_global_styles(styles)
            for appearance_tab in list(self.appearance_tabs):
                if hasattr(appearance_tab, 'load_styles'):
                    appearance_tab.load_styles(styles)
        except (FileNotFoundError, json.JSONDecodeError):
            # Si no hay archivo o está corrupto, no hacer nada (se usarán los defaults)
            self._table_layouts = {}
            self._module_states = {}
        except Exception as e:
            print(f"Error cargando configuración: {e}")


if __name__ == "__main__":
    # Ocultar la ventana de consola de Python en Windows para que solo
    # se vea la ventana de la aplicación.
    if sys.platform.startswith("win"):
        try:
            hwnd = ctypes.windll.kernel32.GetConsoleWindow()
            if hwnd:
                ctypes.windll.user32.ShowWindow(hwnd, 0)  # SW_HIDE = 0
        except Exception:
            pass
    app = MainApp()
    app.mainloop()
