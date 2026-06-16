#!/usr/bin/env python
# -*- coding: utf-8 -*-
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog, StringVar
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns
import squarify
import scipy.stats as stats
import numpy as np
from statistics import NormalDist
import os # <--- AÑADIDO
import json
import traceback
import csv # <--- AÑADIDO
import matplotlib.font_manager as fm
from matplotlib.collections import PolyCollection
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
from pandas.api.types import is_categorical_dtype
from MATLAB_filter_component import FilterComponent
from chart_utils import (
    apply_axis_limits as shared_apply_axis_limits,
    apply_label_mapping_to_dataframe as shared_apply_label_mapping_to_dataframe,
    build_category_palette as shared_build_category_palette,
    color_to_hex as shared_color_to_hex,
    configure_plot_style as shared_configure_plot_style,
    parse_label_mapping as shared_parse_label_mapping,
)
# Imports para gráficos específicos
from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib.sankey import Sankey
from matplotlib_venn import venn2, venn3

USER_PALETTES_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "user_palettes.json")

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

def _is_number_and_between(v, lo, hi):
    try:
        vv = float(v)
        return vv >= lo and vv <= hi
    except Exception:
        return False

class GeneralChartsApp(ttk.Frame):
    def __init__(self, parent, main_app_instance=None): # main_app_instance para acceder a datos/logs globales si es necesario
        super().__init__(parent)
        self.parent_for_dialogs = parent
        self.main_app = main_app_instance # Referencia a la aplicación principal
        self.data = None # DataFrame actual
        self.shared_dataset_metadata = {}
        self.current_shared_filter_summary = []
        # Load chart descriptions if the method exists (defensive: avoid AttributeError during import)
        try:
            if hasattr(self, 'load_chart_descriptions'):
                self.chart_descriptions = self.load_chart_descriptions()
            else:
                self.chart_descriptions = {}
        except Exception:
            self.chart_descriptions = {}
        self.color_options = ["blue", "green", "red", "skyblue", "orange", "purple",
                               "black", "gray", "brown", "pink", "cyan", "magenta",
                               "teal", "olive", "navy", "maroon", "lime", "gold"]
        self.param_dist_group_color_map = {}
        self.param_dist_category_color_map = {}
        self._recode_orders = {}
        # referencia al último figure generado
        self.last_fig = None

        # --- UI Principal de la Pestaña ---
        main_paned_window = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        main_paned_window.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Panel de Controles (Izquierda) con Scroll
        left_panel_container = ttk.Frame(main_paned_window)
        main_paned_window.add(left_panel_container, weight=1)

        canvas = tk.Canvas(left_panel_container)
        scrollbar = ttk.Scrollbar(left_panel_container, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(
                scrollregion=canvas.bbox("all")
            )
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        controls_panel = ttk.LabelFrame(scrollable_frame, text="Controles de Gráfico", padding="10")
        controls_panel.pack(fill=tk.BOTH, expand=True)

        # Panel de Visualización (Derecha)
        display_panel = ttk.Frame(main_paned_window)
        main_paned_window.add(display_panel, weight=3)

        # --- Controles ---
        self.lbl_data_status = ttk.Label(controls_panel, text="Ningún archivo cargado.")
        ttk.Button(controls_panel, text="Guardar PNG", command=lambda: self._safe_call('_save_current_chart', 'png')).pack(pady=5, fill="x")
        ttk.Button(controls_panel, text="Guardar SVG", command=lambda: self._safe_call('_save_current_chart', 'svg')).pack(pady=5, fill="x")

        self.chart_family_var = StringVar(value="Todos")
        self.chart_type_var = StringVar()
        self.chart_selector_catalog = self._build_chart_selector_catalog()
        self.chart_family_options = ["Todos"] + list(self.chart_selector_catalog.keys())
        self.chart_types = self._flatten_chart_selector_catalog(self.chart_selector_catalog)

        ttk.Label(controls_panel, text="Familia de gráficas:").pack(anchor="w", pady=(2, 0))
        self.chart_family_combo = ttk.Combobox(
            controls_panel,
            textvariable=self.chart_family_var,
            values=self.chart_family_options,
            state="readonly",
            width=30,
        )
        self.chart_family_combo.pack(pady=(0, 6), fill="x")
        self.chart_family_combo.bind("<<ComboboxSelected>>", lambda e: self._safe_call('_on_chart_family_selected', e))

        ttk.Label(controls_panel, text="Tipo de gráfica:").pack(anchor="w")
        self.chart_type_combo = ttk.Combobox(
            controls_panel,
            textvariable=self.chart_type_var,
            values=self.chart_types,
            state="readonly",
            width=30,
        )
        self.chart_type_combo.pack(pady=(0, 4), fill="x")
        self.chart_type_combo.bind("<<ComboboxSelected>>", lambda e: self._safe_call('_on_chart_type_selected', e))

        self.chart_subtype_hint_var = StringVar(value="Selecciona un tipo para ver sus subtipos disponibles.")
        ttk.Label(
            controls_panel,
            textvariable=self.chart_subtype_hint_var,
            foreground="#666666",
            wraplength=290,
            justify="left",
        ).pack(anchor="w", pady=(0, 10))

        self._refresh_chart_type_options()

        # Panel de Filtros
        filters_panel = ttk.LabelFrame(controls_panel, text="Filtros", padding="10")
        filters_panel.pack(fill="x", expand=True, pady=(10,0))
        self.filter_component = FilterComponent(filters_panel)
        self.filter_component.pack(fill="x", expand=True)

        self.parameter_controls_frame = ttk.Frame(controls_panel)
        self.parameter_controls_frame.pack(fill="x", expand=True, pady=(0,10))

        ttk.Button(controls_panel, text="Generar Gráfico", command=lambda: self._safe_call('_generate_chart')).pack(pady=10, fill="x")
        ttk.Button(controls_panel, text="Cargar Datos (Excel/CSV)", command=lambda: self._safe_call('cargar_datos_para_graficos')).pack(pady=5, fill="x")
    # example data button removed per user's request
        self.lbl_data_status = ttk.Label(controls_panel, text="Ningún archivo cargado.")
        self.lbl_data_status.pack(pady=(5,0), anchor="w")


        # --- Área de Visualización del Gráfico ---
        self.chart_display_frame = ttk.LabelFrame(display_panel, text="Visualización del Gráfico", padding="5")
        self.chart_display_frame.pack(fill=tk.BOTH, expand=True)

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

    def _build_chart_selector_catalog(self):
        return {
            "Comparación y ranking": [
                "Gráfico de Barras",
                "Gráfico Lollipop",
                "Gráfico de Bala",
                "Gráfico de Pirámide",
                "Forest Plot (Comparaciones)",
                "Gráfico de Cascada",
                "Gráfico de Embudo",
            ],
            "Distribución y frecuencia": [
                "Histograma",
                "Gráfico de Densidad",
                "Gráfico de Distribución",
                "Polígonos de Frecuencia",
                "Diagrama de Tallo y Hojas",
                "Mapa de Calor",
                "Correlograma",
            ],
            "Relación y tendencia": [
                "Diagrama de Dispersión",
                "Gráfico de Líneas / Área",
                "Gráfico de Coordenadas Paralelas",
                "Gráfico Radial",
            ],
            "Composición y jerarquía": [
                "Gráfico Circular / Anillo",
                "Mapa de Árbol",
                "Diagrama Sunburst",
                "Diagrama de Marimekko",
            ],
            "Redes, flujo y tiempo": [
                "Dendrograma",
                "Diagrama de Sankey",
                "Gráfico de Flujo",
                "Diagrama de Gantt",
                "Gráfico de Velas",
                "Línea de Tiempo",
            ],
            "Mapas geográficos": [
                "Mapa Coroplético",
                "Mapa de Burbujas",
                "Mapa de Puntos",
            ],
            "Conjuntos y especiales": [
                "Diagrama de Venn",
            ],
        }

    def _flatten_chart_selector_catalog(self, catalog=None, family_name=None):
        catalog = catalog or self._build_chart_selector_catalog()
        selected_family = (family_name or "").strip()
        if selected_family and selected_family != "Todos":
            return list(catalog.get(selected_family, []))

        ordered_types = []
        seen = set()
        for family_items in catalog.values():
            for chart_name in family_items:
                if chart_name in seen:
                    continue
                ordered_types.append(chart_name)
                seen.add(chart_name)
        return ordered_types

    def _get_chart_subtype_hint(self, chart_type):
        subtype_hints = {
            "Gráfico de Barras": "Subtipos: simple, agrupado, apilado, 100%, divergente y con significancia.",
            "Gráfico de Distribución": "Subtipos: Strip/Jitter, Swarm, Caja, Violín, Raincloud, Medio-Violín y Resumen con IC.",
            "Gráfico de Líneas / Área": "Subtipos: línea simple, múltiples series y área rellena.",
            "Gráfico Circular / Anillo": "Subtipos: circular y anillo.",
            "Diagrama de Dispersión": "Subtipos: dispersión simple, con tamaño y con color por grupo.",
            "Histograma": "Subtipos: vertical y horizontal.",
            "Gráfico de Densidad": "Subtipos: KDE simple o por grupos.",
            "Mapa de Calor": "Subtipos: calor de valores o matriz resumida.",
            "Correlograma": "Subtipos: correlación por color.",
            "Forest Plot (Comparaciones)": "Subtipos: paramétrico, no paramétrico y automático.",
            "Polígonos de Frecuencia": "Subtipos: simple o comparado por grupo.",
        }
        return subtype_hints.get(
            chart_type,
            "Subtipos: ajusta las opciones del panel para refinar esta gráfica."
        )

    def _refresh_chart_type_options(self, preferred_chart_type=None):
        catalog = getattr(self, 'chart_selector_catalog', None) or self._build_chart_selector_catalog()
        selected_family = self.chart_family_var.get().strip() if hasattr(self, 'chart_family_var') else 'Todos'
        available_types = self._flatten_chart_selector_catalog(catalog, selected_family)

        if hasattr(self, 'chart_type_combo'):
            self.chart_type_combo.configure(values=available_types)

        current_type = str(preferred_chart_type or self.chart_type_var.get() or '').strip()
        if current_type not in available_types:
            current_type = available_types[0] if available_types else ''

        if hasattr(self, 'chart_type_var'):
            self.chart_type_var.set(current_type)
        if hasattr(self, 'chart_subtype_hint_var'):
            self.chart_subtype_hint_var.set(
                self._get_chart_subtype_hint(current_type) if current_type else "Selecciona un tipo para ver sus subtipos disponibles."
            )

    def _on_chart_family_selected(self, event=None):
        self._refresh_chart_type_options()
        self._update_parameter_controls(event=event)

    def _on_chart_type_selected(self, event=None):
        if hasattr(self, 'chart_subtype_hint_var'):
            self.chart_subtype_hint_var.set(self._get_chart_subtype_hint(self.chart_type_var.get().strip()))
        self._update_parameter_controls(event=event)

    def _configure_log_tags(self):
        self.log_text_widget.tag_config("INFO", foreground="black")
        self.log_text_widget.tag_config("DEBUG", foreground="gray")
        self.log_text_widget.tag_config("WARN", foreground="orange")
        self.log_text_widget.tag_config("ERROR", foreground="red", font=("Courier New", 9, "bold"))
        self.log_text_widget.tag_config("SUCCESS", foreground="green")
        self.log_text_widget.tag_config("DESC", foreground="navy", font=("Courier New", 9, "bold"))
        self.log_text_widget.tag_config("PARAMS", foreground="purple")
        self.log_text_widget.tag_config("RECOM", foreground="darkgreen")

    def _load_user_palettes(self):
        """Load palettes saved via the Color Tools tab so they can be reused in charts."""
        if not os.path.exists(USER_PALETTES_PATH):
            return {}
        try:
            with open(USER_PALETTES_PATH, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _parse_custom_colors(self, raw_text):
        """Convert comma/semicolon/line separated colors into a list accepted by matplotlib."""
        if not raw_text:
            return []
        cleaned = raw_text.replace(';', ',').replace('\n', ',')
        tokens = [t.strip() for t in cleaned.split(',') if t.strip()]
        palette = []
        for tok in tokens:
            try:
                palette.append(self._color_to_hex(tok))
            except Exception:
                palette.append(str(tok))
        return palette

    def _p_to_stars(self, p_value):
        """Convierte un valor p en asteriscos de significancia."""
        try:
            p = float(p_value)
        except (ValueError, TypeError):
            return "ns"
        if p <= 0.0001:
            return "****"
        elif p <= 0.001:
            return "***"
        elif p <= 0.01:
            return "**"
        elif p <= 0.05:
            return "*"
        else:
            return "ns"

    def _draw_significance_bracket(self, ax, x1, x2, y, h, text, orientation='Vertical', 
                                    color='black', linewidth=1.0, fontsize=10, fontfamily=None):
        """
        Dibuja un bracket de significancia entre dos posiciones x1 y x2.
        y: altura base del bracket
        h: altura del bracket (cuánto sube)
        text: texto a mostrar (asteriscos o valor p)
        """
        text_kwargs = {'ha': 'center', 'va': 'bottom', 'color': color, 'fontsize': fontsize}
        if fontfamily and fontfamily != 'Default':
            text_kwargs['fontfamily'] = fontfamily
            
        if orientation == 'Horizontal':
            # Para barras horizontales, intercambiamos x e y
            ax.plot([y, y + h, y + h, y], [x1, x1, x2, x2], lw=linewidth, c=color)
            text_kwargs['ha'] = 'left'
            text_kwargs['va'] = 'center'
            ax.text(y + h, (x1 + x2) / 2, text, **text_kwargs)
        else:
            # Barras verticales (normal)
            ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=linewidth, c=color)
            ax.text((x1 + x2) / 2, y + h, text, **text_kwargs)

    def _draw_significance_annotations(self, ax, comparisons, patches, category_order, hue_order, 
                                        orientation='Vertical', settings=None):
        """
        Dibuja anotaciones de significancia estadística entre barras.
        comparisons: lista de diccionarios con {cat1, hue1, cat2, hue2, p_value, show_as}
        """
        if not comparisons or not patches:
            return
        
        settings = settings or {}
        bracket_color = settings.get('color', 'black')
        bracket_lw = float(settings.get('linewidth', 1.0))
        font_size = float(settings.get('fontsize', 9))
        bracket_height = float(settings.get('height', 0.03))  # como fracción del rango
        show_ns = settings.get('show_ns', False)
        vertical_offset = float(settings.get('vertical_offset', 0))  # offset adicional para subir/bajar
        fontfamily = settings.get('fontfamily', None)
        spacing_mult = float(settings.get('spacing', 1.8))  # multiplicador de separación entre brackets
        
        # Filtrar patches válidos
        valid_patches = [p for p in patches if isinstance(p, Rectangle) and (p.get_width() != 0 or p.get_height() != 0)]
        
        # Crear mapeo de (categoría, hue) -> índice de patch
        cat_list = [str(c) for c in category_order] if category_order else []
        hue_list = [str(h) for h in hue_order] if hue_order else [None]
        
        # Construir el orden de combos como lo hace seaborn (hue-major)
        combo_to_idx = {}
        idx = 0
        for hv in hue_list:
            for cat in cat_list:
                key = (cat, str(hv) if hv else None)
                combo_to_idx[key] = idx
                idx += 1
        
        # Obtener rango para calcular alturas
        if orientation == 'Horizontal':
            data_min, data_max = ax.get_xlim()
        else:
            data_min, data_max = ax.get_ylim()
        data_range = data_max - data_min
        
        # Altura base para los brackets (empezar arriba de las barras)
        if orientation == 'Horizontal':
            base_y = max(p.get_x() + p.get_width() for p in valid_patches) + data_range * 0.02
        else:
            base_y = max(p.get_y() + p.get_height() for p in valid_patches) + data_range * 0.02
        
        # Aplicar offset vertical adicional
        base_y += data_range * vertical_offset
        
        current_y = base_y
        h_step = data_range * bracket_height
        
        for comp in comparisons:
            cat1 = str(comp.get('cat1', ''))
            hue1 = str(comp.get('hue1', '')) if comp.get('hue1') else None
            cat2 = str(comp.get('cat2', ''))
            hue2 = str(comp.get('hue2', '')) if comp.get('hue2') else None
            p_value = comp.get('p_value', 1.0)
            show_as = comp.get('show_as', 'stars')  # 'stars' o 'p_value'
            
            # Obtener texto a mostrar
            if show_as == 'p_value':
                try:
                    p_float = float(p_value)
                    if p_float < 0.0001:
                        text = "p<0.0001"
                    else:
                        # Formatear sin ceros trailing innecesarios
                        text = f"p={p_float:.4f}".rstrip('0').rstrip('.')
                        # Asegurar al menos un decimal si es entero
                        if 'p=' in text and '.' not in text.split('=')[1]:
                            text = f"p={p_float:.1f}"
                except:
                    text = str(p_value)
            else:
                text = self._p_to_stars(p_value)
            
            # No dibujar si es ns y no queremos mostrar ns
            if text == "ns" and not show_ns:
                continue
            
            # Encontrar índices de las barras
            key1 = (cat1, hue1)
            key2 = (cat2, hue2)
            
            idx1 = combo_to_idx.get(key1)
            idx2 = combo_to_idx.get(key2)
            
            if idx1 is None or idx2 is None:
                self.log(f"Significancia: no se encontró combinación {key1} o {key2}", "WARN")
                continue
            
            if idx1 >= len(valid_patches) or idx2 >= len(valid_patches):
                continue
            
            patch1 = valid_patches[idx1]
            patch2 = valid_patches[idx2]
            
            # Obtener posiciones x de las barras
            if orientation == 'Horizontal':
                x1 = patch1.get_y() + patch1.get_height() / 2
                x2 = patch2.get_y() + patch2.get_height() / 2
            else:
                x1 = patch1.get_x() + patch1.get_width() / 2
                x2 = patch2.get_x() + patch2.get_width() / 2
            
            # Dibujar bracket
            self._draw_significance_bracket(
                ax, x1, x2, current_y, h_step, text,
                orientation=orientation,
                color=bracket_color,
                linewidth=bracket_lw,
                fontsize=font_size,
                fontfamily=fontfamily
            )
            
            # Aumentar altura para el siguiente bracket (usando separación configurable)
            current_y += h_step * spacing_mult

    def _parse_axis_breaks(self, raw_text):
        """
        Parsea el texto de cortes de eje. Formato: desde1-hasta1, desde2-hasta2, ...
        Retorna lista de tuplas [(desde, hasta), ...]
        """
        if not raw_text or not raw_text.strip():
            return []
        
        breaks = []
        parts = raw_text.replace(';', ',').split(',')
        
        for part in parts:
            part = part.strip()
            if '-' in part:
                try:
                    # Manejar números negativos
                    if part.startswith('-'):
                        # Número negativo al inicio
                        rest = part[1:]
                        if '-' in rest:
                            idx = rest.index('-')
                            desde = -float(rest[:idx])
                            hasta = float(rest[idx+1:])
                        else:
                            continue
                    else:
                        desde_str, hasta_str = part.split('-', 1)
                        desde = float(desde_str.strip())
                        hasta = float(hasta_str.strip())
                    
                    if desde < hasta:
                        breaks.append((desde, hasta))
                except (ValueError, IndexError):
                    continue
        
        return sorted(breaks, key=lambda x: x[0])

    def _draw_axis_break_marks(self, ax, breaks, axis='y', break_size=0.015, line_width=1.0):
        """
        Dibuja las marcas de corte (líneas diagonales compactas //) en el eje especificado.
        breaks: lista de tuplas (desde, hasta) indicando los rangos cortados
        break_size: tamaño relativo de las marcas (multiplica el tamaño base)
        line_width: grosor de las líneas
        """
        if not breaks:
            return
        
        # Obtener límites actuales
        if axis == 'y':
            lo, hi = ax.get_ylim()
            data_range = hi - lo
        else:
            lo, hi = ax.get_xlim()
            data_range = hi - lo
        
        # Tamaño base de las marcas (más compacto)
        d = data_range * 0.008 * break_size / 0.015  # altura de cada línea diagonal
        
        for (desde, hasta) in breaks:
            if axis == 'y':
                xmin, xmax = ax.get_xlim()
                x_range = xmax - xmin
                
                # Ancho de la marca (qué tan horizontal es)
                mark_w = x_range * 0.015
                # Separación entre las dos líneas paralelas (muy pequeña)
                gap = d * 0.5
                
                # Dibujar en ambas posiciones del corte
                for y_pos in [desde, hasta]:
                    # === Lado izquierdo del gráfico ===
                    # Primera línea diagonal /
                    x_pts = [xmin - mark_w, xmin + mark_w]
                    y_pts = [y_pos - d, y_pos + d]
                    # Fondo blanco para tapar el eje
                    ax.plot(x_pts, y_pts, color='white', lw=line_width*3, clip_on=False, zorder=100, solid_capstyle='butt')
                    ax.plot(x_pts, y_pts, color='black', lw=line_width, clip_on=False, zorder=101, solid_capstyle='butt')
                    
                    # Segunda línea diagonal / (paralela, muy cerca)
                    y_pts2 = [y_pos - d + gap, y_pos + d + gap]
                    ax.plot(x_pts, y_pts2, color='white', lw=line_width*3, clip_on=False, zorder=100, solid_capstyle='butt')
                    ax.plot(x_pts, y_pts2, color='black', lw=line_width, clip_on=False, zorder=101, solid_capstyle='butt')
                    
                    # === Lado derecho del gráfico ===
                    x_pts_r = [xmax - mark_w, xmax + mark_w]
                    ax.plot(x_pts_r, y_pts, color='white', lw=line_width*3, clip_on=False, zorder=100, solid_capstyle='butt')
                    ax.plot(x_pts_r, y_pts, color='black', lw=line_width, clip_on=False, zorder=101, solid_capstyle='butt')
                    ax.plot(x_pts_r, y_pts2, color='white', lw=line_width*3, clip_on=False, zorder=100, solid_capstyle='butt')
                    ax.plot(x_pts_r, y_pts2, color='black', lw=line_width, clip_on=False, zorder=101, solid_capstyle='butt')
                    
            else:
                # Marcas diagonales en el eje X
                ymin, ymax = ax.get_ylim()
                y_range = ymax - ymin
                mark_h = y_range * 0.015
                gap = d * 0.5
                
                for x_pos in [desde, hasta]:
                    # Parte inferior del gráfico
                    y_pts = [ymin - mark_h, ymin + mark_h]
                    x_pts = [x_pos - d, x_pos + d]
                    ax.plot(x_pts, y_pts, color='white', lw=line_width*3, clip_on=False, zorder=100, solid_capstyle='butt')
                    ax.plot(x_pts, y_pts, color='black', lw=line_width, clip_on=False, zorder=101, solid_capstyle='butt')
                    
                    x_pts2 = [x_pos - d + gap, x_pos + d + gap]
                    ax.plot(x_pts2, y_pts, color='white', lw=line_width*3, clip_on=False, zorder=100, solid_capstyle='butt')
                    ax.plot(x_pts2, y_pts, color='black', lw=line_width, clip_on=False, zorder=101, solid_capstyle='butt')

    def _apply_axis_breaks(self, fig, ax, y_breaks=None, x_breaks=None, data_elements=None, 
                            break_size=0.015, line_width=1.0, category_labels=None):
        """
        Aplica un corte de eje REAL creando dos subplots.
        
        Uso: Si escribes "200-1400" en cortes Y:
        - Parte inferior mostrará 0 a 200
        - Parte superior mostrará 1400 al máximo
        - Las marcas // aparecen entre ambas partes
        
        break_size: tamaño relativo de las marcas
        line_width: grosor de las líneas
        category_labels: lista de etiquetas de categorías para el eje X (opcional)
        """
        if not y_breaks and not x_breaks:
            return ax
        
        # Solo implementamos corte en Y por ahora (el más común)
        if y_breaks:
            desde, hasta = y_breaks[0]
            
            # Obtener límites actuales y propiedades del gráfico original
            y_lo_orig, y_hi_orig = ax.get_ylim()
            x_lo_orig, x_hi_orig = ax.get_xlim()
            
            # El corte solo tiene sentido si desde < hasta y ambos están dentro del rango
            if desde >= hasta:
                self.log(f"Corte inválido: desde ({desde}) debe ser menor que hasta ({hasta})", "WARN")
                return ax
            
            # Determinar los rangos de cada parte
            # Parte inferior: y_lo_orig hasta 'desde' (con margen)
            # Parte superior: 'hasta' hasta y_hi_orig (con margen)
            bottom_range = desde - y_lo_orig
            top_range = y_hi_orig - hasta
            
            if bottom_range <= 0 or top_range <= 0:
                self.log("El corte debe estar dentro del rango de datos", "WARN")
                return ax
            
            # Calcular proporciones para los subplots
            total_visible = bottom_range + top_range
            bottom_ratio = bottom_range / total_visible
            top_ratio = top_range / total_visible
            
            # Mínimo 20% para cada parte para que se vea bien
            bottom_ratio = max(0.2, min(0.8, bottom_ratio))
            top_ratio = 1 - bottom_ratio
            
            # Guardar propiedades del gráfico original
            title = ax.get_title()
            xlabel = ax.get_xlabel()
            ylabel = ax.get_ylabel()
            
            # Guardar ticks del eje Y (para respetar ticks personalizados)
            ytick_positions = list(ax.get_yticks())
            self.log(f"Capturados yticks originales: {ytick_positions}", "DEBUG")
            
            # Guardar ticks y etiquetas del eje X (importante para variables categóricas)
            xtick_positions = []
            xtick_labels = []
            xtick_rotation = 0
            
            # Si tenemos category_labels pasadas, usarlas directamente (más confiable)
            if category_labels:
                xtick_labels = [str(c) for c in category_labels]
                xtick_positions = list(range(len(xtick_labels)))
                self.log(f"Usando category_labels pasadas: {xtick_labels}", "DEBUG")
            else:
                # Intentar capturar del gráfico existente
                try:
                    # Método 1: Forzar renderizado y obtener etiquetas
                    fig.canvas.draw_idle()
                    
                    # Obtener los objetos de etiquetas
                    ticklabels_obj = ax.get_xticklabels()
                    if ticklabels_obj:
                        xtick_rotation = ticklabels_obj[0].get_rotation() if ticklabels_obj else 0
                        xtick_labels = [t.get_text() for t in ticklabels_obj]
                        xtick_positions = list(ax.get_xticks())
                    
                    # Si las etiquetas están vacías después del draw
                    if not any(xtick_labels) or all(lbl == '' for lbl in xtick_labels):
                        fig.canvas.draw()
                        ticklabels_obj = ax.get_xticklabels()
                        xtick_labels = [t.get_text() for t in ticklabels_obj]
                        xtick_positions = list(ax.get_xticks())
                        
                    self.log(f"Capturadas {len(xtick_labels)} etiquetas X: {xtick_labels[:5]}...", "DEBUG")
                except Exception as e:
                    self.log(f"Error capturando etiquetas X: {e}", "WARN")
            
            # Obtener los elementos dibujados (patches = barras)
            patches_data = []
            for p in ax.patches:
                if hasattr(p, 'get_x') and hasattr(p, 'get_height'):
                    patches_data.append({
                        'x': p.get_x(),
                        'y': p.get_y(),
                        'width': p.get_width(),
                        'height': p.get_height(),
                        'facecolor': p.get_facecolor(),
                        'edgecolor': p.get_edgecolor(),
                        'linewidth': p.get_linewidth()
                    })
            
            # Obtener líneas (para error bars) - incluir más propiedades
            lines_data = []
            for line in ax.lines:
                lines_data.append({
                    'xdata': line.get_xdata().copy(),
                    'ydata': line.get_ydata().copy(),
                    'color': line.get_color(),
                    'linewidth': line.get_linewidth(),
                    'linestyle': line.get_linestyle(),
                    'marker': line.get_marker(),
                    'markersize': line.get_markersize(),
                    'markerfacecolor': line.get_markerfacecolor(),
                    'markeredgecolor': line.get_markeredgecolor(),
                    'zorder': line.get_zorder()
                })
            
            # Obtener colecciones (LineCollections para marcadores centrales de IC, etc.)
            from matplotlib.collections import LineCollection
            collections_data = []
            for coll in ax.collections:
                if isinstance(coll, LineCollection):
                    collections_data.append({
                        'segments': coll.get_segments(),
                        'colors': coll.get_colors(),
                        'linewidths': coll.get_linewidths(),
                        'linestyles': coll.get_linestyles(),
                        'zorder': coll.get_zorder()
                    })
            
            # Obtener textos (etiquetas n= y Total N)
            texts_data = []
            texts_relative = []  # Textos con coordenadas relativas (como Total N)
            for t in ax.texts:
                pos = t.get_position()
                text_content = t.get_text()
                # Detectar si es un texto con posición relativa (como Total N)
                # Los textos relativos tienen coordenadas entre 0 y 1
                is_relative = (0 <= pos[0] <= 1 and 0 <= pos[1] <= 1 and 
                              ('Total' in text_content or 'N =' in text_content or 'N=' in text_content))
                
                text_info = {
                    'x': pos[0],
                    'y': pos[1],
                    'text': text_content,
                    'ha': t.get_ha(),
                    'va': t.get_va(),
                    'fontsize': t.get_fontsize(),
                    'color': t.get_color()
                }
                
                if is_relative:
                    texts_relative.append(text_info)
                else:
                    texts_data.append(text_info)
            
            # Limpiar la figura y crear dos subplots
            fig.clear()
            
            # Crear gridspec con espacio para las marcas de corte
            import matplotlib.gridspec as gridspec
            gs = gridspec.GridSpec(2, 1, height_ratios=[top_ratio, bottom_ratio], hspace=0.05)
            
            ax_top = fig.add_subplot(gs[0])
            ax_bottom = fig.add_subplot(gs[1])  # Sin sharex para poder controlar etiquetas independientemente
            
            # Configurar límites
            margin_bottom = bottom_range * 0.05
            margin_top = top_range * 0.05
            
            ax_bottom.set_ylim(y_lo_orig - margin_bottom, desde + margin_bottom)
            ax_top.set_ylim(hasta - margin_top, y_hi_orig + margin_top)
            ax_bottom.set_xlim(x_lo_orig, x_hi_orig)
            ax_top.set_xlim(x_lo_orig, x_hi_orig)
            
            # Aplicar yticks personalizados a cada subplot (filtrar por rango)
            if ytick_positions:
                # Ticks para el subplot inferior (desde 0 hasta 'desde')
                yticks_bottom = [t for t in ytick_positions if y_lo_orig - margin_bottom <= t <= desde + margin_bottom]
                if yticks_bottom:
                    ax_bottom.set_yticks(yticks_bottom)
                    self.log(f"Aplicados yticks inferior: {yticks_bottom}", "DEBUG")
                
                # Ticks para el subplot superior (desde 'hasta' hasta el máximo)
                yticks_top = [t for t in ytick_positions if hasta - margin_top <= t <= y_hi_orig + margin_top]
                if yticks_top:
                    ax_top.set_yticks(yticks_top)
                    self.log(f"Aplicados yticks superior: {yticks_top}", "DEBUG")
            
            # Redibujar barras en ambos ejes
            from matplotlib.patches import Rectangle
            for pd in patches_data:
                rect_bottom = Rectangle((pd['x'], pd['y']), pd['width'], pd['height'],
                                        facecolor=pd['facecolor'], edgecolor=pd['edgecolor'],
                                        linewidth=pd['linewidth'])
                rect_top = Rectangle((pd['x'], pd['y']), pd['width'], pd['height'],
                                     facecolor=pd['facecolor'], edgecolor=pd['edgecolor'],
                                     linewidth=pd['linewidth'])
                ax_bottom.add_patch(rect_bottom)
                ax_top.add_patch(rect_top)
            
            # Redibujar líneas (error bars) con todas sus propiedades
            for ld in lines_data:
                ax_bottom.plot(ld['xdata'], ld['ydata'], color=ld['color'], 
                              linewidth=ld['linewidth'], linestyle=ld['linestyle'],
                              marker=ld['marker'], markersize=ld['markersize'],
                              markerfacecolor=ld['markerfacecolor'], 
                              markeredgecolor=ld['markeredgecolor'],
                              zorder=ld['zorder'])
                ax_top.plot(ld['xdata'], ld['ydata'], color=ld['color'], 
                           linewidth=ld['linewidth'], linestyle=ld['linestyle'],
                           marker=ld['marker'], markersize=ld['markersize'],
                           markerfacecolor=ld['markerfacecolor'],
                           markeredgecolor=ld['markeredgecolor'],
                           zorder=ld['zorder'])
            
            # Redibujar colecciones (LineCollections para marcadores centrales de IC)
            for cd in collections_data:
                lc_bottom = LineCollection(cd['segments'], colors=cd['colors'],
                                          linewidths=cd['linewidths'], 
                                          linestyles=cd['linestyles'],
                                          zorder=cd['zorder'])
                lc_top = LineCollection(cd['segments'], colors=cd['colors'],
                                       linewidths=cd['linewidths'],
                                       linestyles=cd['linestyles'],
                                       zorder=cd['zorder'])
                ax_bottom.add_collection(lc_bottom)
                ax_top.add_collection(lc_top)
            
            # Redibujar textos en el eje apropiado
            for td in texts_data:
                if td['y'] <= desde:
                    ax_bottom.text(td['x'], td['y'], td['text'], ha=td['ha'], va=td['va'],
                                  fontsize=td['fontsize'], color=td['color'])
                else:
                    ax_top.text(td['x'], td['y'], td['text'], ha=td['ha'], va=td['va'],
                               fontsize=td['fontsize'], color=td['color'])
            
            # Redibujar textos relativos (como Total N) en el eje superior
            for td in texts_relative:
                ax_top.text(td['x'], td['y'], td['text'], ha=td['ha'], va=td['va'],
                           fontsize=td['fontsize'], color=td['color'], transform=ax_top.transAxes)
            
            # Ocultar spines entre los dos gráficos
            ax_top.spines['bottom'].set_visible(False)
            ax_bottom.spines['top'].set_visible(False)
            ax_top.tick_params(bottom=False, labelbottom=False)
            ax_bottom.xaxis.tick_bottom()
            
            # Restaurar etiquetas del eje X (nombres de categorías)
            # PRIORIDAD: usar category_labels si se pasaron directamente
            if category_labels and len(category_labels) > 0:
                # Usar las categorías pasadas directamente (más confiable)
                indices = list(range(len(category_labels)))
                ax_bottom.set_xticks(indices)
                ax_bottom.set_xticklabels([str(c) for c in category_labels], rotation=xtick_rotation)
                self.log(f"Etiquetas X de category_labels: {category_labels}", "DEBUG")
            elif xtick_labels and any(lbl.strip() for lbl in xtick_labels if lbl):
                # Fallback: usar las etiquetas capturadas del gráfico original
                valid_pairs = [(pos, lbl) for pos, lbl in zip(xtick_positions, xtick_labels) 
                              if lbl and lbl.strip()]
                if valid_pairs:
                    positions, labels = zip(*valid_pairs)
                    ax_bottom.set_xticks(list(positions))
                    ax_bottom.set_xticklabels(list(labels), rotation=xtick_rotation)
                    self.log(f"Restauradas {len(labels)} etiquetas X capturadas", "DEBUG")
            else:
                self.log("No se encontraron etiquetas X para restaurar", "WARN")
            
            # Dibujar las marcas de corte // (estilo estándar matplotlib)
            # Usamos el método probado de broken axis
            d = 0.015  # tamaño de las diagonales
            kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
                         linestyle='none', color='k', mec='k', mew=line_width, clip_on=False)
            
            # Marcas en el subplot superior (parte inferior)
            ax_top.plot([0, 1], [0, 0], transform=ax_top.transAxes, **kwargs)
            
            # Marcas en el subplot inferior (parte superior)
            ax_bottom.plot([0, 1], [1, 1], transform=ax_bottom.transAxes, **kwargs)
            
            # Restaurar etiquetas
            ax_top.set_title(title)
            ax_bottom.set_xlabel(xlabel)
            
            # Etiqueta Y centrada entre ambos ejes
            fig.text(0.02, 0.5, ylabel, va='center', rotation='vertical')
            
            return ax_top  # Retornar el eje superior como principal
        
        return ax

    def _create_broken_axis_chart(self, fig, original_ax, y_breaks=None, x_breaks=None, 
                                   chart_func=None, chart_kwargs=None):
        """
        Crea un gráfico con eje cortado usando múltiples subplots.
        Retorna el axes principal para continuar añadiendo elementos.
        """
        if not y_breaks or not chart_func:
            return original_ax
        
        # Para simplificar, manejamos un solo corte en Y
        if len(y_breaks) > 1:
            self.log("Solo se soporta un corte de eje Y por ahora. Usando el primero.", "WARN")
        
        break_from, break_to = y_breaks[0]
        
        # Obtener límites de datos
        # Necesitamos saber el rango total de los datos
        y_data_min = 0
        y_data_max = break_to * 1.5  # estimación inicial
        
        # Eliminar el axes original
        original_ax.remove()
        
        # Calcular proporciones para los subplots
        # El subplot inferior va de y_data_min a break_from
        # El subplot superior va de break_to a y_data_max
        lower_range = break_from - y_data_min
        upper_range = y_data_max - break_to
        total_range = lower_range + upper_range
        
        lower_ratio = max(0.2, lower_range / total_range) if total_range > 0 else 0.5
        upper_ratio = 1 - lower_ratio
        
        # Crear dos subplots
        gs = fig.add_gridspec(2, 1, height_ratios=[upper_ratio, lower_ratio], hspace=0.05)
        ax_upper = fig.add_subplot(gs[0])
        ax_lower = fig.add_subplot(gs[1], sharex=ax_upper)
        
        # Dibujar el gráfico en ambos axes
        if chart_kwargs:
            chart_kwargs_upper = chart_kwargs.copy()
            chart_kwargs_upper['ax'] = ax_upper
            chart_kwargs_lower = chart_kwargs.copy()
            chart_kwargs_lower['ax'] = ax_lower
            
            try:
                chart_func(**chart_kwargs_upper)
                chart_func(**chart_kwargs_lower)
            except Exception as e:
                self.log(f"Error creando gráfico con eje cortado: {e}", "ERROR")
                return original_ax
        
        # Establecer límites
        ax_upper.set_ylim(break_to, y_data_max)
        ax_lower.set_ylim(y_data_min, break_from)
        
        # Ocultar spines entre los plots
        ax_upper.spines['bottom'].set_visible(False)
        ax_lower.spines['top'].set_visible(False)
        ax_upper.tick_params(bottom=False, labelbottom=False)
        ax_lower.xaxis.tick_bottom()
        
        # Dibujar las marcas de corte diagonales
        d = 0.015  # tamaño de las marcas
        kwargs = dict(transform=ax_upper.transAxes, color='k', clip_on=False, lw=1)
        ax_upper.plot((-d, +d), (-d, +d), **kwargs)
        ax_upper.plot((1 - d, 1 + d), (-d, +d), **kwargs)
        
        kwargs.update(transform=ax_lower.transAxes)
        ax_lower.plot((-d, +d), (1 - d, 1 + d), **kwargs)
        ax_lower.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)
        
        return ax_lower  # Retornar el inferior para etiquetas de eje X

    def _parse_limits_and_ticks(self, raw_text):
        """Return (limits, ticks) parsed from a comma list. Accepts min,max or explicit ticks."""
        if not raw_text:
            return None, None
        parts = [p.strip() for p in raw_text.split(',') if p.strip()]
        try:
            nums = [float(p) for p in parts]
        except Exception:
            return None, None

        if not nums:
            return None, None
        if len(nums) == 1:
            return nums[0], None
        if len(nums) == 2:
            return (nums[0], nums[1]), None

        limits = (min(nums), max(nums))
        ticks = nums
        return limits, ticks

    def _annotate_bar_counts(self, ax, data, x_col, y_col, hue_col, category_order, hue_order, orientation,
                              show_counts=False, show_total=False, label_size="9", label_color="#000000", label_position="arriba de la barra",
                              total_position="sup derecha", fontfamily=None):
        """Añade etiquetas n por barra y un total opcional."""
        if not show_counts and not show_total:
            return

        try:
            size_val = float(label_size)
        except Exception:
            size_val = 9.0
        color_val = label_color or "#000000"
        
        # Preparar kwargs de fuente
        font_kwargs = {}
        if fontfamily and fontfamily != 'Default':
            font_kwargs['fontfamily'] = fontfamily

        # Normalizar tipos a string para que coincidan con category_order/hue_order
        data_counts = data.copy()
        try:
            data_counts[x_col] = data_counts[x_col].astype(str)
        except Exception:
            data_counts[x_col] = data_counts[x_col].map(lambda v: str(v))
        if hue_col:
            try:
                data_counts[hue_col] = data_counts[hue_col].astype(str)
            except Exception:
                data_counts[hue_col] = data_counts[hue_col].map(lambda v: str(v))

        group_cols = [x_col]
        if hue_col:
            group_cols.append(hue_col)
        counts = data_counts.groupby(group_cols, dropna=False).size()

        cat_order = category_order or list(dict.fromkeys(data[x_col].astype(str).tolist()))
        hue_order_resolved = hue_order if hue_col else None
        combos = []
        if hue_col:
            hue_order_resolved = hue_order_resolved or list(dict.fromkeys(data[hue_col].astype(str).tolist()))
            for cat in cat_order:
                for hv in hue_order_resolved:
                    combos.append((cat, hv))
        else:
            combos = [(cat, None) for cat in cat_order]

        patches = [p for p in ax.patches if isinstance(p, (Rectangle,)) and (p.get_width() != 0 or p.get_height() != 0)]
        combos = combos[:len(patches)]

        y_lo, y_hi = ax.get_ylim()
        x_lo, x_hi = ax.get_xlim()
        y_pad = (y_hi - y_lo) * 0.01 if y_hi != y_lo else 0.5
        x_pad = (x_hi - x_lo) * 0.01 if x_hi != x_lo else 0.5

        for patch, combo in zip(patches, combos):
            cat, hv = combo
            if hue_col:
                key = (cat, hv)
            else:
                key = cat
            try:
                n_val = counts.get(key, 0)
            except Exception:
                n_val = 0

            pos = (label_position or "arriba de la barra").strip().lower()

            if orientation == "Horizontal":
                value_pos = patch.get_width()
                y_pos = patch.get_y() + patch.get_height() / 2
                # Obtener el límite inferior real del eje X
                x_axis_min = ax.get_xlim()[0]
                # La base visible de la barra es el máximo entre la base real y el límite del eje
                bar_base_x = max(patch.get_x(), x_axis_min)
                
                if pos == "pie de la barra":
                    x_text = bar_base_x - x_pad
                    # Si queda fuera del eje, ponerla justo en el borde
                    if x_text < x_axis_min:
                        x_text = x_axis_min + x_pad
                        ha = "left"
                    else:
                        ha = "right"
                    ax.text(x_text, y_pos, f"n={n_val}", va="center", ha=ha, fontsize=size_val, color=color_val, **font_kwargs)
                elif pos == "dentro al pie":
                    x_text = bar_base_x + x_pad
                    ax.text(x_text, y_pos, f"n={n_val}", va="center", ha="left", fontsize=size_val, color=color_val, **font_kwargs)
                elif pos == "arriba del gráfico":
                    x_hi = ax.get_xlim()[1]
                    x_text = x_hi - x_pad
                    ax.text(x_text, y_pos, f"n={n_val}", va="center", ha="right", fontsize=size_val, color=color_val, **font_kwargs)
                else:  # arriba de la barra (default)
                    x_text = value_pos + x_pad
                    ax.text(x_text, y_pos, f"n={n_val}", va="center", ha="left", fontsize=size_val, color=color_val, **font_kwargs)
            else:
                x_pos = patch.get_x() + patch.get_width() / 2
                value_pos = patch.get_y() + patch.get_height()
                # Obtener el límite inferior real del eje Y
                y_axis_min = ax.get_ylim()[0]
                # La base visible de la barra es el máximo entre la base real y el límite del eje
                bar_base_y = max(patch.get_y(), y_axis_min)
                
                if pos == "pie de la barra":
                    y_text = bar_base_y - y_pad
                    # Si queda fuera del eje, ponerla justo en el borde inferior
                    if y_text < y_axis_min:
                        y_text = y_axis_min + y_pad
                        va = "bottom"
                    else:
                        va = "top"
                elif pos == "dentro al pie":
                    y_text = bar_base_y + y_pad
                    va = "bottom"
                elif pos == "arriba del gráfico":
                    y_hi = ax.get_ylim()[1]
                    y_text = y_hi - y_pad
                    va = "top"
                else:  # arriba de la barra (default)
                    y_text = value_pos + y_pad
                    if ax.get_yscale() == 'log':
                        y_text = value_pos * 1.02 if value_pos > 0 else value_pos + y_pad
                    va = "bottom"
                ax.text(x_pos, y_text, f"n={n_val}", va=va, ha="center", fontsize=size_val, color=color_val, **font_kwargs)

        if show_total:
            total_n = int(len(data))
            pos_map = {
                "sup derecha": (0.99, 0.99, 'right', 'top'),
                "sup izquierda": (0.01, 0.99, 'left', 'top'),
                "inf derecha": (0.99, 0.01, 'right', 'bottom'),
                "inf izquierda": (0.01, 0.01, 'left', 'bottom'),
            }
            x_t, y_t, ha_t, va_t = pos_map.get(total_position, (0.99, 0.99, 'right', 'top'))
            ax.text(x_t, y_t, f"Total N = {total_n}", transform=ax.transAxes, ha=ha_t, va=va_t,
                    fontsize=size_val, color=color_val, **font_kwargs)

    def log(self, message, level="INFO"):
        try:
            timestamp = pd.Timestamp.now().strftime('%H:%M:%S.%f')[:-3]
            self.log_text_widget.config(state=tk.NORMAL)
            self.log_text_widget.insert(tk.END, f"[{timestamp}] [{level.upper()}] {message}\n", level.upper())
            self.log_text_widget.config(state=tk.DISABLED)
            self.log_text_widget.see(tk.END)
        except Exception as e:
            print(f"Error en logger de GeneralChartsApp: {e}")

    def _safe_call(self, name, *args, **kwargs):
        """Call a method by name safely from Tk callbacks. Logs and shows an error if missing or if the call fails."""
        func = getattr(self, name, None)

        if not callable(func):
            msg = f"Función no encontrada: {name}"
            self.log(msg, "ERROR")
            try:
                messagebox.showerror("Error Interno", msg, parent=self.parent_for_dialogs)
            except Exception:
                pass
            return None
        try:
            return func(*args, **kwargs)
        except Exception as e:
            self.log(f"Error ejecutando {name}: {e}", "ERROR")
            import traceback
            self.log(traceback.format_exc(), "DEBUG")
            try:
                messagebox.showerror("Error", f"Error ejecutando {name}: {e}", parent=self.parent_for_dialogs)
            except Exception:
                pass
            return None

    def _create_variable_selector(self, parent_frame, label_text, columns_list, default_value="", allow_recode=False):
        label_widget = ttk.Label(parent_frame, text=label_text)
        label_widget.pack(anchor="w")
        label_pack_kwargs = {'anchor': 'w'}
        
        var_frame = ttk.Frame(parent_frame)
        var_frame.pack(fill="x", pady=(0,5))
        frame_pack_kwargs = {'fill': "x", 'pady': (0,5)}

        # Variable to hold the selected column name
        selected_col_var = StringVar(value=default_value)
        selector_combo = ttk.Combobox(var_frame, textvariable=selected_col_var, values=columns_list, state="readonly")
        selector_combo.pack(side=tk.LEFT, expand=True, fill="x")

        # Variable to hold the display name for this specific parameter
        display_name_var = StringVar(value=default_value) # Default to selected column name
        display_entry = ttk.Entry(var_frame, textvariable=display_name_var, width=15)
        display_entry.pack(side=tk.RIGHT)

        recode_var = None
        if allow_recode:
            recode_frame = ttk.Frame(parent_frame)
            recode_frame.pack(fill="x", pady=(0, 5))
            ttk.Label(recode_frame, text="Decodificación y Orden (e.g., 1:Leve, 2:Moderado):").pack(anchor="w")
            recode_var = StringVar()
            recode_entry = ttk.Entry(recode_frame, textvariable=recode_var)
            recode_entry.pack(fill="x")
            # Button to preview recoding
            ttk.Button(recode_frame, text="Probar recodificación", command=lambda sv=selected_col_var, rv=recode_var: self._test_recode(sv, rv)).pack(anchor="e", pady=(3,0))
            setattr(recode_var, "_widget_frame", recode_frame)
            recode_pack_kwargs = {'fill': "x", 'pady': (0,5)}
        else:
            recode_frame = None
            recode_pack_kwargs = None
        
        setattr(selected_col_var, "_label_widget", label_widget)
        setattr(selected_col_var, "_widget_frame", var_frame)
        setattr(selected_col_var, "_selector_widget", selector_combo)
        setattr(selected_col_var, "_display_entry", display_entry)
        setattr(selected_col_var, "_recode_var", recode_var)
        setattr(selected_col_var, "_label_pack_kwargs", label_pack_kwargs)
        setattr(selected_col_var, "_frame_pack_kwargs", frame_pack_kwargs)
        setattr(selected_col_var, "_recode_pack_kwargs", recode_pack_kwargs)
        if recode_var is not None:
            setattr(recode_var, "_label_widget", None)
            setattr(recode_var, "_entry_widget", recode_entry)
            setattr(recode_var, "_parent_selector", selected_col_var)
        setattr(selected_col_var, "_recode_frame", recode_frame)

        return selected_col_var, display_name_var, recode_var

    def _apply_recode(self, df, column, recode_str, remember_order=True):
        if not recode_str or not column:
            if remember_order and column:
                self._recode_orders.pop(column, None)
            return df

        try:
            self.log(f"Iniciando decodificación para la columna '{column}'.", "DEBUG")
            self.log(f"Cadena de decodificación recibida: '{recode_str}'", "DEBUG")

            df = df.copy()

            # Reusar el helper compartido para el caso simple clave:etiqueta y
            # conservar la lógica avanzada local para rangos o múltiples tokens.
            basic_order, basic_mapping = shared_parse_label_mapping(recode_str)

            # Parse recode_str into ordered rules supporting single keys, multiple keys
            # separated by '|' or '/', and numeric ranges like '1-3'. Each rule is
            # (matcher_func, label). We'll apply rules in order; first match wins.
            pairs = [p.strip() for p in recode_str.split(',') if p.strip()]
            rules = []
            order = list(basic_order) if basic_order else []
            use_shared_direct_mapping = bool(basic_mapping)
            for pair in pairs:
                if ':' in pair:
                    left, label = pair.split(':', 1)
                    left = left.strip()
                    label = label.strip()
                else:
                    left = pair.strip()
                    label = left
                if not left:
                    continue
                if not label:
                    label = left
                order.append(label)

                # support multiple tokens separated by '|' or '/'
                tokens = []
                if '|' in left:
                    tokens = [t.strip() for t in left.split('|') if t.strip()]
                    use_shared_direct_mapping = False
                elif '/' in left:
                    tokens = [t.strip() for t in left.split('/') if t.strip()]
                    use_shared_direct_mapping = False
                else:
                    tokens = [left]

                # build matchers from tokens
                for tok in tokens:
                    # range like 1-3
                    if '-' in tok:
                        use_shared_direct_mapping = False
                        parts = tok.split('-', 1)
                        try:
                            lo = float(parts[0].strip())
                            hi = float(parts[1].strip())
                            def make_range(lo, hi):
                                return lambda v: (not pd.isna(v)) and _is_number_and_between(v, lo, hi)
                            rules.append((make_range(lo, hi), label, f"{lo}-{hi}"))
                        except Exception:
                            # fallback to literal
                            rules.append((lambda v, t=tok: str(v).strip() == t, label, tok))
                    else:
                        # single token matcher: match numeric or string equivalently
                        def make_tok_match(t):
                            def match(v):
                                if pd.isna(v):
                                    return False

                                def _annotate_bar_counts(self, ax, data, x_col, y_col, hue_col, category_order, hue_order, orientation,
                                                          show_counts=False, show_total=False, label_size="9", label_color="#000000"):
                                    """Añade etiquetas de n por barra y un total opcional."""
                                    if not show_counts and not show_total:
                                        return

                                    try:
                                        size_val = float(label_size)
                                    except Exception:
                                        size_val = 9.0
                                    color_val = label_color or "#000000"

                                    # Calcular conteos por grupo
                                    group_cols = [x_col]
                                    if hue_col:
                                        group_cols.append(hue_col)
                                    counts = data.groupby(group_cols, dropna=False).size()

                                    # Construir el orden esperado de combinaciones
                                    cat_order = category_order or list(dict.fromkeys(data[x_col].astype(str).tolist()))
                                    hue_order_resolved = hue_order if hue_col else None
                                    combos = []
                                    if hue_col:
                                        hue_order_resolved = hue_order_resolved or list(dict.fromkeys(data[hue_col].astype(str).tolist()))
                                        for cat in cat_order:
                                            for hv in hue_order_resolved:
                                                combos.append((cat, hv))
                                    else:
                                        combos = [(cat, None) for cat in cat_order]

                                    patches = [p for p in ax.patches if isinstance(p, (Rectangle,)) and (p.get_width() != 0 or p.get_height() != 0)]
                                    combos = combos[:len(patches)]  # evitar desbordes si seaborn agrega extras

                                    # Pequeño padding para separar texto de la barra
                                    y_lo, y_hi = ax.get_ylim()
                                    x_lo, x_hi = ax.get_xlim()
                                    y_pad = (y_hi - y_lo) * 0.01 if y_hi != y_lo else 0.5
                                    x_pad = (x_hi - x_lo) * 0.01 if x_hi != x_lo else 0.5

                                    for patch, combo in zip(patches, combos):
                                        cat, hv = combo
                                        key = (cat,) if hv is None else (cat, hv)
                                        n_val = counts.get(key, 0)

                                        if orientation == "Horizontal":
                                            value_pos = patch.get_width()
                                            y_pos = patch.get_y() + patch.get_height() / 2
                                            x_text = value_pos + x_pad
                                            ax.text(x_text, y_pos, f"n={n_val}", va="center", ha="left", fontsize=size_val, color=color_val)
                                        else:
                                            x_pos = patch.get_x() + patch.get_width() / 2
                                            value_pos = patch.get_height()
                                            y_text = value_pos + y_pad
                                            # Para escala log, multiplicar un factor para evitar superposición
                                            if ax.get_yscale() == 'log':
                                                y_text = value_pos * 1.02 if value_pos > 0 else value_pos + y_pad
                                            ax.text(x_pos, y_text, f"n={n_val}", va="bottom", ha="center", fontsize=size_val, color=color_val)

                                    if show_total:
                                        total_n = int(len(data))
                                        if orientation == "Horizontal":
                                            ax.text(0.99, 0.01, f"Total N = {total_n}", transform=ax.transAxes, ha="right", va="bottom",
                                                    fontsize=size_val, color=color_val)
                                        else:
                                            ax.text(0.99, 0.99, f"Total N = {total_n}", transform=ax.transAxes, ha="right", va="top",
                                                    fontsize=size_val, color=color_val)

                                def _parse_limits_and_ticks(self, raw_text):
                                    """Devuelve (limits, ticks) donde limits es una tupla o None y ticks una lista o None.
                                    Permite: "min,max" o "min,max,t1,t2,..." o solo "t1,t2,..." (se usan min/max de la lista)."""
                                    if not raw_text:
                                        return None, None
                                    parts = [p.strip() for p in raw_text.split(',') if p.strip()]
                                    nums = [float(p) for p in parts]
                                    if not nums:
                                        return None, None

                                    if len(nums) == 1:
                                        # solo un número -> centrar en ese valor
                                        return nums[0], None

                                    if len(nums) == 2:
                                        # clásico min,max
                                        return (nums[0], nums[1]), None

                                    # tres o más: tratar como lista de ticks; límites = min/max
                                    limits = (min(nums), max(nums))
                                    ticks = nums
                                    return limits, ticks

                                def _parse_custom_colors(self, raw_text):
                                    if not raw_text:
                                        return []
                                    tokens = [t.strip() for t in raw_text.split(',') if t.strip()]
                                    palette = []
                                    for tok in tokens:
                                        try:
                                            palette.append(self._color_to_hex(tok))
                                        except Exception:
                                            try:
                                                palette.append(str(tok))
                                            except Exception:
                                                pass
                                    return palette
                                # try numeric compare
                                try:
                                    tnum = float(t)
                                    try:
                                        vnum = float(v)
                                        return vnum == tnum
                                    except Exception:
                                        pass
                                except Exception:
                                    pass
                                # otherwise string compare
                                try:
                                    return str(v).strip() == t
                                except Exception:
                                    return False
                            return match
                        rules.append((make_tok_match(tok), label, tok))

            if not rules:
                self.log("El mapa de decodificación está vacío o mal formado. No se aplicarán cambios.", "DEBUG")
                return df

            if order:
                seen_labels = []
                deduped = []
                for lbl in order:
                    lbl_str = str(lbl)
                    if lbl_str in seen_labels:
                        continue
                    seen_labels.append(lbl_str)
                    deduped.append(lbl_str)
                order = deduped
                if remember_order:
                    self._recode_orders[column] = list(order)
            elif remember_order:
                self._recode_orders.pop(column, None)

            self.log(f"Reglas de decodificación creadas: {[r[2] for r in rules]}", "DEBUG")
            col_series = df[column]
            original_unique = col_series.unique()
            self.log(f"Valores únicos en '{column}' ANTES de decodificar: {original_unique}", "DEBUG")

            if use_shared_direct_mapping and basic_mapping:
                mapped_df = shared_apply_label_mapping_to_dataframe(df[[column]].copy(), column, basic_mapping)
                mapped_series = mapped_df[column]
                if order:
                    try:
                        mapped_series = pd.Series(
                            pd.Categorical(mapped_series, categories=order, ordered=True),
                            index=mapped_series.index
                        )
                    except Exception:
                        pass
                df[column] = mapped_series
                self.log(f"Columna '{column}' recodificada usando helper compartido.", "INFO")
                return df

            # Apply rules in order to produce mapped values (without altering original df until success)
            def apply_rules_to_value(v):
                if pd.isna(v):
                    return np.nan

                for matcher, label, _desc in rules:
                    try:
                        if matcher(v):
                            return label
                    except Exception:
                        continue

                # If no rule matched, mark as NaN so it can ser as filtro
                return np.nan

            try:
                mapped_series = col_series.map(lambda v: apply_rules_to_value(v))
            except Exception:
                mapped_series = col_series.astype(object).map(lambda v: apply_rules_to_value(v))

            if order:
                try:
                    mapped_series = pd.Series(
                        pd.Categorical(mapped_series, categories=order, ordered=True),
                        index=mapped_series.index
                    )
                except Exception:
                    pass

            df = df.copy()
            df[column] = mapped_series

            self.log(f"Columna '{column}' dtype después de categorizar: {df[column].dtype}", "DEBUG")

            final_unique = df[column].unique()
            self.log(f"Valores únicos en '{column}' DESPUÉS de categorizar: {final_unique}", "DEBUG")

            # Warn about NaNs which may indicate unmatched keys
            if df[column].isnull().any():
                self.log(f"¡Atención! Se encontraron valores nulos en '{column}' después de la decodificación. "
                         f"Esto puede ocurrir si algunos valores originales no estaban en las reglas de decodificación: {recode_str}", "WARN")

            self.log(f"Columna '{column}' recodificada y ordenada.", "INFO")
            return df

        except Exception as e:
            self.log(f"Error al decodificar la columna '{column}': {e}", "ERROR")
            import traceback
            self.log(traceback.format_exc(), "DEBUG")
            return df

    def _test_recode(self, selected_col_var, recode_var):
        """Show a preview dialog of how recoding will map the unique values of the column."""
        try:
            if self.data is None:
                messagebox.showinfo("Sin Datos", "Cargue datos antes de probar la recodificación.", parent=self.parent_for_dialogs)
                return
            col_name = selected_col_var.get()
            if not col_name:
                messagebox.showinfo("Sin Columna", "Seleccione primero la columna a recodificar.", parent=self.parent_for_dialogs)
                return
            recode_str = recode_var.get()
            if not recode_str:
                messagebox.showinfo("Sin Regla", "Ingrese una cadena de recodificación antes de probar.", parent=self.parent_for_dialogs)
                return

            unique_vals = list(self.data[col_name].dropna().unique())

            # Reuse _apply_recode on a small DataFrame copy to compute mapped values safely
            sample_df = pd.DataFrame({col_name: unique_vals})
            preview_df = self._apply_recode(sample_df, col_name, recode_str, remember_order=False)

            # Build dialog showing original -> mapped
            dlg = tk.Toplevel(self)
            dlg.title(f"Previsualizar recodificación: {col_name}")
            dlg.transient(self.parent_for_dialogs)
            dlg.grab_set()

            frame = ttk.Frame(dlg, padding=8)
            frame.pack(fill='both', expand=True)
            ttk.Label(frame, text=f"Columna: {col_name}").pack(anchor='w')

            tree = ttk.Treeview(frame, columns=('original', 'mapped'), show='headings', height=12)
            tree.heading('original', text='Original')
            tree.heading('mapped', text='Mapeado')
            tree.column('original', width=200)
            tree.column('mapped', width=200)
            tree.pack(fill='both', expand=True, pady=(4,8))

            for orig, mapped in zip(unique_vals, preview_df[col_name].tolist()):
                tree.insert('', 'end', values=(str(orig), str(mapped)))

            btn_frame = ttk.Frame(frame)
            btn_frame.pack(fill='x')
            ttk.Button(btn_frame, text='Cerrar', command=lambda: (dlg.grab_release(), dlg.destroy())).pack(side='right')

        except Exception as e:
            self.log(f"Error en previsualizar recodificación: {e}", 'ERROR')
            messagebox.showerror("Error", f"No se pudo previsualizar la recodificación: {e}", parent=self.parent_for_dialogs)


    def _open_group_color_picker(self):
        """Open a small dialog to pick colors per level for the selected hue variable."""
        try:
            hue_var = getattr(self, 'param_dist_hue_var', None)
            hue_name = hue_var.get() if hue_var else ''
            target_name = None
            target_attr = 'param_dist_group_color_map'

            if hue_name:
                target_name = hue_name
                target_attr = 'param_dist_group_color_map'
            else:
                x_var = getattr(self, 'param_dist_x_var', None)
                x_name = x_var.get() if x_var else ''
                if not x_name:
                    messagebox.showinfo("Sin Variable", "Seleccione primero una variable en 'Agrupar por Color' o en el eje X.", parent=self.parent_for_dialogs)
                    return
                target_name = x_name
                target_attr = 'param_dist_category_color_map'

            if self.data is None:
                messagebox.showinfo("Sin Datos", "Cargue datos antes de asignar colores por nivel.", parent=self.parent_for_dialogs)
                return

            levels = list(self.data[target_name].dropna().unique())
            if not levels:
                messagebox.showinfo("Sin Niveles", f"La columna '{target_name}' no tiene niveles válidos.", parent=self.parent_for_dialogs)
                return

            dlg = tk.Toplevel(self)
            dlg.title(f"Colores para niveles de {target_name}")
            dlg.transient(self)
            dlg.grab_set()

            vars_map = {}
            # Suggest palette colors using seaborn if available
            try:
                palette_name = getattr(self, 'param_dist_group_palette_var', 'deep').get() if hasattr(self, 'param_dist_group_palette_var') else 'deep'
                suggested = sns.color_palette(palette_name, n_colors=len(levels))
                suggested_hex = [('#%02x%02x%02x' % tuple(int(255*c) for c in ctuple)) if isinstance(ctuple, tuple) else str(ctuple) for ctuple in [(int(255*c[0]), int(255*c[1]), int(255*c[2])) for c in suggested]]
            except Exception:
                suggested_hex = [self.color_options[i % len(self.color_options)] for i in range(len(levels))]

            for i, lvl in enumerate(levels):
                row = ttk.Frame(dlg)
                row.pack(fill='x', padx=6, pady=3)
                ttk.Label(row, text=str(lvl)).pack(side='left')
                var = StringVar(value=suggested_hex[i] if i < len(suggested_hex) else self.color_options[0])
                vars_map[lvl] = var
                ttk.Combobox(row, textvariable=var, values=self.color_options, state='readonly', width=12).pack(side='right')

            def on_save():
                mapping = {str(k): v.get() for k, v in vars_map.items()}
                setattr(self, target_attr, mapping)
                dlg.grab_release()
                dlg.destroy()
                self.log(f"Asignados colores por nivel para {target_name}: {mapping}", 'INFO')

            btn_frame = ttk.Frame(dlg)
            btn_frame.pack(fill='x', padx=6, pady=6)
            ttk.Button(btn_frame, text='Guardar', command=on_save).pack(side='right', padx=4)
            ttk.Button(btn_frame, text='Cancelar', command=lambda: (dlg.grab_release(), dlg.destroy())).pack(side='right')

        except Exception as e:
            self.log(f"Error abriendo selector de colores por nivel: {e}", 'ERROR')
            messagebox.showerror("Error", f"No se pudo abrir el selector: {e}", parent=self.parent_for_dialogs)

    def _open_significance_editor(self):
        """Abre un editor visual para definir comparaciones de significancia."""
        try:
            # Obtener variables actuales
            x_var = getattr(self, 'param_bar_x_var', None)
            hue_var = getattr(self, 'param_bar_hue_var', None)
            x_col = x_var.get().strip() if x_var else ''
            hue_col = hue_var.get().strip() if hue_var else ''
            
            if not x_col:
                messagebox.showinfo("Sin Variable", "Seleccione primero la variable de categorías (eje X).", parent=self.parent_for_dialogs)
                return
            
            if self.data is None:
                messagebox.showinfo("Sin Datos", "Cargue datos antes de definir comparaciones.", parent=self.parent_for_dialogs)
                return
            
            # Obtener categorías y niveles de hue
            categories = [str(c) for c in self.data[x_col].dropna().unique()]
            hue_levels = [str(h) for h in self.data[hue_col].dropna().unique()] if hue_col else ['']
            
            if not categories:
                messagebox.showinfo("Sin Categorías", f"La columna '{x_col}' no tiene valores.", parent=self.parent_for_dialogs)
                return
            
            # Crear diálogo
            dlg = tk.Toplevel(self)
            dlg.title("Editor de Comparaciones de Significancia")
            dlg.transient(self)
            dlg.grab_set()
            dlg.geometry("600x500")
            
            # Frame de instrucciones
            instr_frame = ttk.Frame(dlg)
            instr_frame.pack(fill="x", padx=10, pady=5)
            ttk.Label(instr_frame, text="Defina las comparaciones entre barras. Seleccione las categorías/grupos y el valor p.",
                     wraplength=550).pack(anchor="w")
            
            # Frame para las comparaciones
            comp_canvas = tk.Canvas(dlg)
            scrollbar = ttk.Scrollbar(dlg, orient="vertical", command=comp_canvas.yview)
            comp_frame = ttk.Frame(comp_canvas)
            
            comp_canvas.configure(yscrollcommand=scrollbar.set)
            scrollbar.pack(side="right", fill="y")
            comp_canvas.pack(side="left", fill="both", expand=True, padx=10)
            canvas_frame = comp_canvas.create_window((0,0), window=comp_frame, anchor="nw")
            
            def on_frame_configure(event):
                comp_canvas.configure(scrollregion=comp_canvas.bbox("all"))
            comp_frame.bind("<Configure>", on_frame_configure)
            
            # Lista de comparaciones (variables)
            self._sig_editor_rows = []
            
            def add_comparison_row(cat1='', hue1='', cat2='', hue2='', p_val=''):
                row_frame = ttk.Frame(comp_frame)
                row_frame.pack(fill="x", pady=2)
                
                row_data = {}
                
                # Categoría 1
                ttk.Label(row_frame, text="Cat1:").pack(side="left")
                cat1_var = StringVar(value=cat1)
                ttk.Combobox(row_frame, textvariable=cat1_var, values=categories, width=12, state="readonly").pack(side="left", padx=2)
                row_data['cat1'] = cat1_var
                
                # Hue 1
                if hue_col:
                    ttk.Label(row_frame, text="Hue1:").pack(side="left")
                    hue1_var = StringVar(value=hue1)
                    ttk.Combobox(row_frame, textvariable=hue1_var, values=hue_levels, width=10, state="readonly").pack(side="left", padx=2)
                    row_data['hue1'] = hue1_var
                else:
                    row_data['hue1'] = StringVar(value='')
                
                ttk.Label(row_frame, text="vs").pack(side="left", padx=4)
                
                # Categoría 2
                ttk.Label(row_frame, text="Cat2:").pack(side="left")
                cat2_var = StringVar(value=cat2)
                ttk.Combobox(row_frame, textvariable=cat2_var, values=categories, width=12, state="readonly").pack(side="left", padx=2)
                row_data['cat2'] = cat2_var
                
                # Hue 2
                if hue_col:
                    ttk.Label(row_frame, text="Hue2:").pack(side="left")
                    hue2_var = StringVar(value=hue2)
                    ttk.Combobox(row_frame, textvariable=hue2_var, values=hue_levels, width=10, state="readonly").pack(side="left", padx=2)
                    row_data['hue2'] = hue2_var
                else:
                    row_data['hue2'] = StringVar(value='')
                
                # Valor p
                ttk.Label(row_frame, text="p=").pack(side="left", padx=(8,0))
                p_var = StringVar(value=p_val)
                ttk.Entry(row_frame, textvariable=p_var, width=8).pack(side="left", padx=2)
                row_data['p_value'] = p_var
                
                # Botón eliminar
                def remove_row():
                    row_frame.destroy()
                    if row_data in self._sig_editor_rows:
                        self._sig_editor_rows.remove(row_data)
                
                ttk.Button(row_frame, text="✕", width=2, command=remove_row).pack(side="left", padx=4)
                
                self._sig_editor_rows.append(row_data)
            
            # Parsear comparaciones existentes
            try:
                existing_text = self.param_bar_sig_comparisons_text.get("1.0", tk.END).strip()
                for line in existing_text.split('\n'):
                    if not line.strip():
                        continue
                    parts = [p.strip() for p in line.split(',')]
                    if len(parts) >= 3:
                        if hue_col and len(parts) >= 5:
                            add_comparison_row(parts[0], parts[1], parts[2], parts[3], parts[4] if len(parts) > 4 else '')
                        elif not hue_col and len(parts) >= 3:
                            add_comparison_row(parts[0], '', parts[1], '', parts[2] if len(parts) > 2 else '')
            except:
                pass
            
            # Si no hay comparaciones, agregar una vacía
            if not self._sig_editor_rows:
                add_comparison_row()
            
            # Botones
            btn_frame = ttk.Frame(dlg)
            btn_frame.pack(fill="x", padx=10, pady=10)
            
            ttk.Button(btn_frame, text="+ Agregar comparación", command=lambda: add_comparison_row()).pack(side="left")
            
            def on_save():
                lines = []
                for row in self._sig_editor_rows:
                    cat1 = row['cat1'].get()
                    hue1 = row['hue1'].get()
                    cat2 = row['cat2'].get()
                    hue2 = row['hue2'].get()
                    p_val = row['p_value'].get()
                    
                    if not cat1 or not cat2:
                        continue
                    
                    if hue_col:
                        lines.append(f"{cat1},{hue1},{cat2},{hue2},{p_val}")
                    else:
                        lines.append(f"{cat1},{cat2},{p_val}")
                
                # Actualizar el texto
                self.param_bar_sig_comparisons_text.delete("1.0", tk.END)
                self.param_bar_sig_comparisons_text.insert("1.0", '\n'.join(lines))
                
                dlg.grab_release()
                dlg.destroy()
                self.log(f"Guardadas {len(lines)} comparaciones de significancia", "INFO")
            
            ttk.Button(btn_frame, text="Guardar", command=on_save).pack(side="right", padx=4)
            ttk.Button(btn_frame, text="Cancelar", command=lambda: (dlg.grab_release(), dlg.destroy())).pack(side="right")
            
        except Exception as e:
            self.log(f"Error abriendo editor de significancia: {e}", "ERROR")
            messagebox.showerror("Error", f"No se pudo abrir el editor: {e}", parent=self.parent_for_dialogs)

    def _parse_significance_comparisons(self, text, has_hue=False):
        """Parsea el texto de comparaciones y devuelve lista de diccionarios."""
        comparisons = []
        if not text:
            return comparisons
        
        for line in text.strip().split('\n'):
            line = line.strip()
            if not line:
                continue
            
            parts = [p.strip() for p in line.split(',')]
            
            try:
                if has_hue:
                    # Formato: Cat1,Hue1,Cat2,Hue2,p
                    if len(parts) >= 5:
                        comparisons.append({
                            'cat1': parts[0],
                            'hue1': parts[1] if parts[1] else None,
                            'cat2': parts[2],
                            'hue2': parts[3] if parts[3] else None,
                            'p_value': float(parts[4]) if parts[4] else 1.0
                        })
                else:
                    # Formato: Cat1,Cat2,p
                    if len(parts) >= 3:
                        comparisons.append({
                            'cat1': parts[0],
                            'hue1': None,
                            'cat2': parts[1],
                            'hue2': None,
                            'p_value': float(parts[2]) if parts[2] else 1.0
                        })
            except (ValueError, IndexError) as e:
                self.log(f"Error parseando línea de significancia '{line}': {e}", "WARN")
                continue
        
        return comparisons


    def load_chart_descriptions(self):
        # Aquí cargarías las descripciones, parámetros y recomendaciones desde un archivo o diccionario
        # Por ahora, un placeholder
        return {
            "Diagrama de Dispersión": {
                "descripcion": "Representa puntos en un plano cartesiano (x, y). Útil para mostrar relaciones o correlaciones entre dos variables. Para crear un gráfico de burbujas, seleccione una variable en el parámetro 'Tamaño'.",
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

    def _describe_shared_source(self, metadata):
        if not isinstance(metadata, dict):
            return "Archivo compartido"
        source_path = metadata.get("source_path")
        if not source_path:
            return metadata.get("label") or "Archivo compartido"
        try:
            return os.path.basename(source_path)
        except Exception:
            return source_path

    def cargar_datos_para_graficos(self, filepath=None):
        # Allow passing a filepath programmatically for testing; otherwise open file dialog
        self.log(f"Invocando cargar_datos_para_graficos (filepath param: {bool(filepath)})", "DEBUG")
        if not filepath:
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
            
            # Try to load the new data into a temporary variable first
            temp_data = None
            if filepath.endswith(('.xlsx', '.xls')):
                # Excel reading may require openpyxl. Check and provide a clear message if missing.
                try:
                    import openpyxl  # noqa: F401
                except Exception:
                    msg = ("Para leer archivos .xlsx se requiere el paquete 'openpyxl'.\n"
                           "Instálalo en tu entorno virtual: pip install openpyxl")
                    messagebox.showerror("Dependencia faltante", msg, parent=self.parent_for_dialogs)
                    self.log(f"Falta dependencia openpyxl para leer '{filename}'.", "ERROR")
                    return
                try:
                    temp_data = pd.read_excel(filepath, engine='openpyxl')
                except Exception as e:
                    self.log(f"Error al leer Excel '{filename}': {e}", "ERROR")
                    messagebox.showerror("Error leyendo Excel", f"No se pudo leer el archivo Excel:\n{e}", parent=self.parent_for_dialogs)
                    return
            elif filepath.endswith('.csv'):
                try:
                    sniffer = csv.Sniffer() # 'csv' ya está importado
                    with open(filepath, 'r', encoding='utf-8-sig') as f:
                        dialect = sniffer.sniff(f.read(1024))
                    temp_data = pd.read_csv(filepath, sep=dialect.delimiter)
                    self.log(f"Archivo CSV '{filename}' cargado con separador '{dialect.delimiter}' detectado.", "INFO")
                except Exception: 
                    # Try common encodings and separators
                    read_attempts = [
                        {'sep': ',', 'encoding': 'utf-8-sig'},
                        {'sep': ',', 'encoding': 'utf-8'},
                        {'sep': ';', 'encoding': 'utf-8-sig'},
                        {'sep': ';', 'encoding': 'latin1'},
                    ]
                    last_exc = None
                    for opts in read_attempts:
                        try:
                            temp_data = pd.read_csv(filepath, sep=opts['sep'], encoding=opts['encoding'])
                            self.log(f"CSV '{filename}' leído con sep='{opts['sep']}' encoding='{opts['encoding']}'", "INFO")
                            break
                        except Exception as e:
                            last_exc = e
                            continue
                    if temp_data is None:
                        self.log(f"Error al leer CSV '{filename}': {last_exc}", "ERROR")
                        messagebox.showerror("Error leyendo CSV", f"No se pudo leer el CSV:\n{last_exc}", parent=self.parent_for_dialogs)
                        return
            else:
                messagebox.showerror("Error de Archivo", f"Tipo de archivo no soportado: {filename}", parent=self.parent_for_dialogs)
                self.log(f"Tipo de archivo no soportado: {filename}", "ERROR")
                return
            
            # Only update self.data after successful load
            if temp_data is not None:
                # Normalize column names (strip BOMs/spaces)
                temp_data.columns = [str(c).strip() for c in temp_data.columns]
                self.data = temp_data
                self.lbl_data_status.config(text=f"{filename} ({self.data.shape[0]}x{self.data.shape[1]})")
                self.log(f"Datos cargados desde '{filename}'. Dimensiones: {self.data.shape}", "SUCCESS")
                
                self.log("Actualizando componente de filtros...", "DEBUG")
                try:
                    self.filter_component.set_dataframe(self.data)
                    self.log("Componente de filtros actualizado.", "DEBUG")
                except Exception as e:
                    self.log(f"Fallo al actualizar el componente de filtros: {e}", "ERROR")
                    self.log(traceback.format_exc(), "DEBUG")

                # Refresh parameter controls
                self.log("Actualizando controles de parámetros de gráfico...", "DEBUG")
                try:
                    self._update_parameter_controls()
                    self.log("Controles de parámetros de gráfico actualizados.", "DEBUG")
                except Exception as e:
                    self.log(f"Fallo al actualizar controles de parámetros: {e}", "ERROR")
                    self.log(traceback.format_exc(), "DEBUG")
            else:
                raise ValueError("No se pudieron cargar los datos correctamente.")

        except Exception as e:
            messagebox.showerror("Error de Lectura", f"No se pudo leer el archivo:\n{e}", parent=self.parent_for_dialogs)
            self.log(f"Error leyendo archivo '{filepath}': {e}", "ERROR")
            # Don't clear self.data on error - keep existing data
            try:
                self.lbl_data_status.config(text="Error al cargar, datos anteriores conservados.")
            except Exception:
                pass

    # cargar_ejemplo removed per user request

    def receive_shared_dataset(self, *, dataset, filtered_dataset=None, filter_summary=None, metadata=None, source_widget=None):
        if source_widget is self:
            return

        self.shared_dataset_metadata = dict(metadata or {})
        self.current_shared_filter_summary = list(filter_summary or [])

        if dataset is None:
            self.data = None
            try:
                self.filter_component.set_dataframe(pd.DataFrame())
            except Exception:
                pass
            try:
                self.lbl_data_status.config(text="Sin dataset compartido.")
            except Exception:
                pass
            self._clear_chart_display()
            for widget in self.parameter_controls_frame.winfo_children():
                widget.destroy()
            self.log("Dataset compartido eliminado; se limpiaron controles y gráfico.", "INFO")
            return

        if isinstance(dataset, pd.DataFrame):
            try:
                base_df = dataset.copy(deep=True)
            except Exception:
                base_df = dataset
        else:
            self.log("Dataset compartido recibido no es un DataFrame. Acción cancelada.", "ERROR")
            return

        active_df = base_df
        if isinstance(filtered_dataset, pd.DataFrame):
            try:
                active_df = filtered_dataset.copy(deep=True)
            except Exception:
                active_df = filtered_dataset

        self.data = active_df

        try:
            self.filter_component.set_dataframe(active_df)
        except Exception as exc:
            self.log(f"No se pudo actualizar el componente de filtros con el dataset compartido: {exc}", "ERROR")

        try:
            rows, cols = active_df.shape
            source_desc = self._describe_shared_source(self.shared_dataset_metadata)
            filters_applied = len(self.current_shared_filter_summary)
            suffix = f" | Filtros: {filters_applied}" if filters_applied else ""
            self.lbl_data_status.config(text=f"Compartido: {source_desc} ({rows}x{cols}){suffix}")
        except Exception:
            pass

        self.log("Dataset compartido recibido en GeneralChartsApp.", "INFO")
        if self.current_shared_filter_summary:
            for summary in self.current_shared_filter_summary:
                self.log(f"  - {summary}", "DEBUG")

        try:
            self._update_parameter_controls()
        except Exception as exc:
            self.log(f"Error al refrescar controles tras recibir dataset compartido: {exc}", "ERROR")

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
        numeric_columns = [col for col in columnas if pd.api.types.is_numeric_dtype(self.data[col])] if self.data is not None else []

        # Reset bar-specific widget references before (re)building the controls
        self.bar_orientation_label = None
        self.bar_orientation_combo = None
        self.bar_mode_frame = None
        self.bar_color_frame = None
        self.bar_stacked_options_frame = None
        self.bar_segment_frame = None
        self.bar_segment_color_frame = None
        
        if chart_type == "Diagrama de Dispersión":
            self.param_x_var, self.param_x_display_name_var, self.param_x_recode_var = self._create_variable_selector(
                self.parameter_controls_frame, "Variable X:", columnas, allow_recode=True
            )
            self.param_y_var, self.param_y_display_name_var, self.param_y_recode_var = self._create_variable_selector(
                self.parameter_controls_frame, "Variable Y:", columnas, allow_recode=True
            )
            self.param_color_var, self.param_color_display_name_var, self.param_color_recode_var = self._create_variable_selector(
                self.parameter_controls_frame, "Color (Opcional):", [""] + columnas, allow_recode=True
            )
            self.param_size_var, self.param_size_display_name_var, _ = self._create_variable_selector(
                self.parameter_controls_frame, "Tamaño (Opcional, Numérica):", [""] + [col for col in columnas if pd.api.types.is_numeric_dtype(self.data[col])] if self.data is not None else []
            )
            self.param_scatter_style_var, _, _ = self._create_variable_selector(
                self.parameter_controls_frame, "Forma de Marcador (Opcional):", [""] + columnas
            )

            self.param_scatter_fit_reg_var = tk.BooleanVar(value=False)
            ttk.Checkbutton(self.parameter_controls_frame, text="Añadir línea de regresión", variable=self.param_scatter_fit_reg_var).pack(anchor="w", pady=(5,0))
            self.param_scatter_show_corr_var = tk.BooleanVar(value=False)
            ttk.Checkbutton(self.parameter_controls_frame, text="Mostrar correlación (r, p)", variable=self.param_scatter_show_corr_var).pack(anchor="w")
            self.param_scatter_marginal_var = StringVar(value="Ninguno")
            ttk.Label(self.parameter_controls_frame, text="Distribuciones Marginales:").pack(anchor="w")
            ttk.Combobox(self.parameter_controls_frame, textvariable=self.param_scatter_marginal_var,
                         values=["Ninguno", "Histograma", "KDE", "Rug"], state="readonly").pack(fill="x", pady=(0,5))
            self.param_scatter_alpha_var = StringVar(value="0.7")
            ttk.Label(self.parameter_controls_frame, text="Transparencia (alpha):").pack(anchor="w")
            ttk.Entry(self.parameter_controls_frame, textvariable=self.param_scatter_alpha_var).pack(fill="x", pady=(0,5))

        elif chart_type == "Histograma":
            self.param_hist_var, _, _ = self._create_variable_selector(
                self.parameter_controls_frame, "Variable Numérica:", numeric_columns
            )
            self.param_hist_hue_var, _, self.param_hist_hue_recode_var = self._create_variable_selector(
                self.parameter_controls_frame, "Agrupar por Color (Opcional):", [""] + columnas, allow_recode=True
            )
            
            ttk.Label(self.parameter_controls_frame, text="Número de Bins (Opcional):").pack(anchor="w")
            self.param_hist_bins_var = StringVar(value="auto")
            ttk.Entry(self.parameter_controls_frame, textvariable=self.param_hist_bins_var).pack(fill="x", pady=(0,5))

            self.param_hist_kde_var = tk.BooleanVar(value=False)
            ttk.Checkbutton(self.parameter_controls_frame, text="Mostrar Curva de Densidad (KDE)", variable=self.param_hist_kde_var).pack(anchor="w", pady=(0,5))
            
            ttk.Label(self.parameter_controls_frame, text="Orientación:").pack(anchor="w")
            self.param_hist_orientation_var = StringVar(value="Vertical")
            ttk.Combobox(self.parameter_controls_frame, textvariable=self.param_hist_orientation_var, values=["Vertical", "Horizontal"], state="readonly").pack(fill="x", pady=(0,5))

            ttk.Label(self.parameter_controls_frame, text="Tipo de Estadístico:").pack(anchor="w")
            self.param_hist_stat_var = StringVar(value="count")
            ttk.Combobox(self.parameter_controls_frame, textvariable=self.param_hist_stat_var,
                         values=["count", "frequency", "probability", "percent", "density"], state="readonly").pack(fill="x", pady=(0,5))

            self.param_hist_cumulative_var = tk.BooleanVar(value=False)
            ttk.Checkbutton(self.parameter_controls_frame, text="Histograma Acumulado", variable=self.param_hist_cumulative_var).pack(anchor="w")

            ttk.Label(self.parameter_controls_frame, text="Modo Múltiple (con Hue):").pack(anchor="w")
            self.param_hist_multiple_var = StringVar(value="layer")
            ttk.Combobox(self.parameter_controls_frame, textvariable=self.param_hist_multiple_var,
                         values=["layer", "dodge", "stack", "fill"], state="readonly").pack(fill="x", pady=(0,5))

            self.param_hist_show_stats_var = tk.BooleanVar(value=False)
            ttk.Checkbutton(self.parameter_controls_frame, text="Mostrar estadísticas (media, mediana, std)", variable=self.param_hist_show_stats_var).pack(anchor="w")

            self.param_hist_show_rug_var = tk.BooleanVar(value=False)
            ttk.Checkbutton(self.parameter_controls_frame, text="Mostrar Rug Plot", variable=self.param_hist_show_rug_var).pack(anchor="w")

        elif chart_type == "Gráfico de Densidad":
            self.param_density_var, _, _ = self._create_variable_selector(
                self.parameter_controls_frame, "Variable Numérica:", numeric_columns
            )
            self.param_density_hue_var, _, self.param_density_hue_recode_var = self._create_variable_selector(
                self.parameter_controls_frame, "Agrupar por Color (Opcional):", [""] + columnas, allow_recode=True
            )
            self.param_density_fill_var = tk.BooleanVar(value=True)
            ttk.Checkbutton(self.parameter_controls_frame, text="Rellenar bajo la curva", variable=self.param_density_fill_var).pack(anchor="w")
            self.param_density_rug_var = tk.BooleanVar(value=False)
            ttk.Checkbutton(self.parameter_controls_frame, text="Mostrar Rug Plot", variable=self.param_density_rug_var).pack(anchor="w")
            self.param_density_cumulative_var = tk.BooleanVar(value=False)
            ttk.Checkbutton(self.parameter_controls_frame, text="Densidad Acumulada", variable=self.param_density_cumulative_var).pack(anchor="w")
            ttk.Label(self.parameter_controls_frame, text="Ancho de banda (bw_adjust):").pack(anchor="w")
            self.param_density_bw_var = StringVar(value="1.0")
            ttk.Entry(self.parameter_controls_frame, textvariable=self.param_density_bw_var).pack(fill="x", pady=(0,5))
            ttk.Label(self.parameter_controls_frame, text="Modo Múltiple (con Hue):").pack(anchor="w")
            self.param_density_multiple_var = StringVar(value="layer")
            ttk.Combobox(self.parameter_controls_frame, textvariable=self.param_density_multiple_var,
                         values=["layer", "stack", "fill"], state="readonly").pack(fill="x", pady=(0,5))
            self.param_density_show_stats_var = tk.BooleanVar(value=False)
            ttk.Checkbutton(self.parameter_controls_frame, text="Mostrar estadísticas (media, mediana)", variable=self.param_density_show_stats_var).pack(anchor="w")

        elif chart_type == "Gráfico Circular / Anillo":
            self.param_pie_label_var, _, self.param_pie_label_recode_var = self._create_variable_selector(
                self.parameter_controls_frame, "Variable de Etiquetas (Categorías):", columnas, allow_recode=True
            )
            self.param_pie_value_var, _, _ = self._create_variable_selector(
                self.parameter_controls_frame, "Variable de Valores (Opcional, Numérica):", [""] + numeric_columns
            )

            pie_frame = ttk.LabelFrame(self.parameter_controls_frame, text="Configuración", padding="5")
            pie_frame.pack(fill="x", expand=True, pady=(5, 0))

            ttk.Label(pie_frame, text="Hueco (0 = pastel, 0.6 = anillo)").pack(anchor="w")
            self.param_pie_hole_var = StringVar(value="0.0")
            ttk.Entry(pie_frame, textvariable=self.param_pie_hole_var).pack(fill="x", pady=(0, 5))

            ttk.Label(pie_frame, text="Máx. categorías visibles (0 = todas)").pack(anchor="w")
            self.param_pie_max_categories_var = StringVar(value="8")
            ttk.Entry(pie_frame, textvariable=self.param_pie_max_categories_var).pack(fill="x", pady=(0, 5))

            ttk.Label(pie_frame, text="Etiqueta para 'Otros'").pack(anchor="w")
            self.param_pie_other_label_var = StringVar(value="Otros")
            ttk.Entry(pie_frame, textvariable=self.param_pie_other_label_var).pack(fill="x", pady=(0, 5))

            ttk.Label(pie_frame, text="Orden de categorías").pack(anchor="w")
            self.param_pie_sort_mode_var = StringVar(value="Descendente")
            ttk.Combobox(
                pie_frame,
                textvariable=self.param_pie_sort_mode_var,
                values=["Descendente", "Ascendente", "Original"],
                state="readonly"
            ).pack(fill="x", pady=(0, 5))

            self.param_pie_exclude_blank_var = tk.BooleanVar(value=True)
            ttk.Checkbutton(pie_frame, text="Excluir blancos / NaN", variable=self.param_pie_exclude_blank_var).pack(anchor="w")
            self.param_pie_show_percent_var = tk.BooleanVar(value=True)
            ttk.Checkbutton(pie_frame, text="Mostrar porcentaje", variable=self.param_pie_show_percent_var).pack(anchor="w")
            self.param_pie_show_value_var = tk.BooleanVar(value=False)
            ttk.Checkbutton(pie_frame, text="Mostrar valor absoluto", variable=self.param_pie_show_value_var).pack(anchor="w")

        elif chart_type == "Gráfico de Barras":
            self.param_bar_x_var, _, self.param_bar_x_recode_var = self._create_variable_selector(
                self.parameter_controls_frame, "Variable de Eje (Categorías):", columnas, allow_recode=True
            )
            self.param_bar_y_var, _, _ = self._create_variable_selector(
                self.parameter_controls_frame, "Variable de Valor (Numérica, Opcional):", [""] + numeric_columns
            )
            self.param_bar_hue_var, _, self.param_bar_hue_recode_var = self._create_variable_selector(
                self.parameter_controls_frame, "Agrupar por Color (Opcional):", [""] + columnas, allow_recode=True
            )

            self.bar_orientation_label = ttk.Label(self.parameter_controls_frame, text="Orientación:")
            self.bar_orientation_label.pack(anchor="w")
            setattr(self.bar_orientation_label, '_default_pack_kwargs', {'anchor': 'w'})
            self.param_bar_orientation_var = StringVar(value="Vertical")
            self.bar_orientation_combo = ttk.Combobox(self.parameter_controls_frame, textvariable=self.param_bar_orientation_var, values=["Vertical", "Horizontal"], state="readonly")
            self.bar_orientation_combo.pack(fill="x", pady=(0,5))
            setattr(self.bar_orientation_combo, '_default_pack_kwargs', {'fill': "x", 'pady': (0,5)})

            mode_frame = ttk.LabelFrame(self.parameter_controls_frame, text="Modo de Barras", padding="5")
            mode_frame.pack(fill="x", expand=True, pady=(0,5))
            self.bar_mode_frame = mode_frame
            ttk.Label(mode_frame, text="Seleccione el tipo de barras a generar:").pack(anchor="w")
            mode_options = ["Simple", "Agrupado", "Apilado", "Segmentado"]
            previous_mode_var = getattr(self, 'param_bar_mode_var', None)
            previous_mode_value = previous_mode_var.get() if isinstance(previous_mode_var, tk.Variable) else "Simple"
            if previous_mode_value not in mode_options:
                previous_mode_value = "Simple"
            self.param_bar_mode_var = StringVar(value=previous_mode_value)
            mode_combo = ttk.Combobox(mode_frame, textvariable=self.param_bar_mode_var, values=mode_options, state="readonly")
            mode_combo.pack(fill="x", pady=(0,4))

            self.bar_color_frame = ttk.LabelFrame(self.parameter_controls_frame, text="Colores de Barras", padding="5")
            self.bar_color_frame.pack(fill="x", expand=True, pady=(0,5))
            setattr(self.bar_color_frame, '_default_pack_kwargs', {'fill': "x", 'expand': True, 'pady': (0,5)})

            ttk.Label(self.bar_color_frame, text="Modo de Color:").pack(anchor="w")
            self.param_bar_color_mode_var = StringVar(value="auto")
            color_modes = ["auto", "un color", "paleta"]
            ttk.Combobox(self.bar_color_frame, textvariable=self.param_bar_color_mode_var, values=color_modes, state="readonly").pack(fill="x", pady=(0,4))

            ttk.Label(self.bar_color_frame, text="Color único (#hex o nombre):").pack(anchor="w")
            self.param_bar_single_color_var = StringVar(value="#4C72B0")
            ttk.Entry(self.bar_color_frame, textvariable=self.param_bar_single_color_var).pack(fill="x", pady=(0,4))

            ttk.Label(self.bar_color_frame, text="Paleta (cuando aplique):").pack(anchor="w")
            base_bar_palettes = ['default', 'deep', 'muted', 'pastel', 'bright', 'dark', 'colorblind', 'viridis', 'plasma', 'inferno', 'magma', 'cividis']
            user_bar_palettes = [f"usuario: {name}" for name in sorted(self._load_user_palettes().keys())]
            bar_palette_choices = base_bar_palettes + user_bar_palettes
            self.param_bar_palette_choice_var = StringVar(value='deep')
            ttk.Combobox(self.bar_color_frame, textvariable=self.param_bar_palette_choice_var, values=bar_palette_choices, state="readonly").pack(fill="x", pady=(0,4))

            ttk.Label(self.bar_color_frame, text="Colores personalizados (coma separada):").pack(anchor="w")
            self.param_bar_custom_colors_var = StringVar()
            ttk.Entry(self.bar_color_frame, textvariable=self.param_bar_custom_colors_var).pack(fill="x", pady=(0,4))

            ttk.Label(self.bar_color_frame, text="Ancho de barra (0.1-0.9):").pack(anchor="w")
            previous_bar_width = getattr(self, 'param_bar_width_var', None)
            width_default = previous_bar_width.get() if isinstance(previous_bar_width, tk.Variable) else '0.8'
            self.param_bar_width_var = StringVar(value=width_default)
            ttk.Entry(self.bar_color_frame, textvariable=self.param_bar_width_var).pack(fill="x", pady=(0,4))

            # Etiquetas de conteo / total
            labels_frame = ttk.LabelFrame(self.parameter_controls_frame, text="Etiquetas de Conteo", padding="5")
            labels_frame.pack(fill="x", expand=True, pady=(0,5))
            setattr(labels_frame, '_default_pack_kwargs', {'fill': "x", 'expand': True, 'pady': (0,5)})

            self.param_bar_show_counts_var = tk.BooleanVar(value=False)
            ttk.Checkbutton(labels_frame, text="Mostrar n en cada barra", variable=self.param_bar_show_counts_var).pack(anchor="w")
            self.param_bar_show_total_var = tk.BooleanVar(value=False)
            ttk.Checkbutton(labels_frame, text="Mostrar total (N)", variable=self.param_bar_show_total_var).pack(anchor="w")

            total_pos_row = ttk.Frame(labels_frame)
            total_pos_row.pack(fill="x", pady=(4,0))
            ttk.Label(total_pos_row, text="Posición del total N:").pack(side="left")
            total_positions = [
                "sup derecha",
                "sup izquierda",
                "inf derecha",
                "inf izquierda"
            ]
            self.param_bar_total_position_var = StringVar(value="sup derecha")
            ttk.Combobox(total_pos_row, textvariable=self.param_bar_total_position_var, values=total_positions, state="readonly", width=14).pack(side="left", padx=(4,0))

            size_color_row = ttk.Frame(labels_frame)
            size_color_row.pack(fill="x", pady=(4,0))
            ttk.Label(size_color_row, text="Tamaño texto:").pack(side="left")
            self.param_bar_label_size_var = StringVar(value="9")
            ttk.Entry(size_color_row, textvariable=self.param_bar_label_size_var, width=6).pack(side="left", padx=(4,10))
            ttk.Label(size_color_row, text="Color (#hex o nombre):").pack(side="left")
            self.param_bar_label_color_var = StringVar(value="#000000")
            ttk.Entry(size_color_row, textvariable=self.param_bar_label_color_var, width=12).pack(side="left", padx=(4,0))

            position_row = ttk.Frame(labels_frame)
            position_row.pack(fill="x", pady=(4,0))
            ttk.Label(position_row, text="Posición de n:").pack(side="left")
            label_positions = [
                "arriba de la barra",
                "pie de la barra",
                "dentro al pie",
                "arriba del gráfico"
            ]
            self.param_bar_label_position_var = StringVar(value="arriba de la barra")
            ttk.Combobox(position_row, textvariable=self.param_bar_label_position_var, values=label_positions, state="readonly", width=20).pack(side="left", padx=(4,0))

            self.bar_stacked_options_frame = ttk.LabelFrame(self.parameter_controls_frame, text="Opciones de Apilado", padding="5")
            setattr(self.bar_stacked_options_frame, '_default_pack_kwargs', {'fill': "x", 'expand': True, 'pady': (0,5)})
            self.param_stacked_100_var = tk.BooleanVar(value=False)
            ttk.Checkbutton(self.bar_stacked_options_frame, text="Normalizar a 100%", variable=self.param_stacked_100_var).pack(anchor="w")

            self.bar_segment_frame = ttk.LabelFrame(self.parameter_controls_frame, text="Segmentos Numéricos", padding="5")
            setattr(self.bar_segment_frame, '_default_pack_kwargs', {'fill': "x", 'expand': True, 'pady': (0,5)})
            ttk.Label(
                self.bar_segment_frame,
                text="En modo Segmentado cada barra suma varias columnas numéricas; los campos de Valor y Color se desactivan automáticamente."
            ).pack(anchor="w", pady=(0,4))
            self.param_bar_segment_vars = []
            max_segments = 4
            for idx in range(max_segments):
                var, display_var, _ = self._create_variable_selector(
                    self.bar_segment_frame,
                    f"Segmento {idx + 1}:",
                    [""] + numeric_columns
                )
                self.param_bar_segment_vars.append((var, display_var))

            self.param_bar_segment_normalize_var = tk.BooleanVar(value=False)
            ttk.Checkbutton(self.bar_segment_frame, text="Normalizar cada barra a 100%", variable=self.param_bar_segment_normalize_var).pack(anchor="w", pady=(4,0))

            self.bar_segment_color_frame = ttk.LabelFrame(self.parameter_controls_frame, text="Colores de Segmentos", padding="5")
            setattr(self.bar_segment_color_frame, '_default_pack_kwargs', {'fill': "x", 'expand': True, 'pady': (0,5)})
            base_segment_palettes = ['deep', 'muted', 'pastel', 'bright', 'dark', 'colorblind', 'viridis', 'plasma', 'inferno', 'magma', 'cividis']
            user_segment_palettes = [f"usuario: {name}" for name in sorted(self._load_user_palettes().keys())]
            segment_palette_choices = base_segment_palettes + user_segment_palettes
            self.param_bar_segment_palette_var = StringVar(value='deep')
            ttk.Label(self.bar_segment_color_frame, text="Paleta de Segmentos:").pack(anchor="w")
            ttk.Combobox(self.bar_segment_color_frame, textvariable=self.param_bar_segment_palette_var, values=segment_palette_choices, state="readonly").pack(fill="x", pady=(0,4))
            ttk.Label(self.bar_segment_color_frame, text="Colores personalizados (coma separada):").pack(anchor="w")
            self.param_bar_segment_custom_colors_var = StringVar()
            ttk.Entry(self.bar_segment_color_frame, textvariable=self.param_bar_segment_custom_colors_var).pack(fill="x", pady=(0,4))

            # --- Panel de Marcas de Significancia ---
            self.bar_significance_frame = ttk.LabelFrame(self.parameter_controls_frame, text="Marcas de Significancia", padding="5")
            setattr(self.bar_significance_frame, '_default_pack_kwargs', {'fill': "x", 'expand': True, 'pady': (0,5)})
            self.bar_significance_frame.pack(fill="x", expand=True, pady=(0,5))
            
            self.param_bar_show_sig_var = tk.BooleanVar(value=False)
            ttk.Checkbutton(self.bar_significance_frame, text="Mostrar marcas de significancia", 
                           variable=self.param_bar_show_sig_var).pack(anchor="w")
            
            sig_options_row = ttk.Frame(self.bar_significance_frame)
            sig_options_row.pack(fill="x", pady=(4,0))
            ttk.Label(sig_options_row, text="Mostrar como:").pack(side="left")
            self.param_bar_sig_display_var = StringVar(value="estrellas")
            ttk.Combobox(sig_options_row, textvariable=self.param_bar_sig_display_var, 
                        values=["estrellas", "valor p"], state="readonly", width=12).pack(side="left", padx=(4,0))
            
            self.param_bar_sig_show_ns_var = tk.BooleanVar(value=False)
            ttk.Checkbutton(sig_options_row, text="Mostrar ns", variable=self.param_bar_sig_show_ns_var).pack(side="left", padx=(10,0))
            
            sig_style_row = ttk.Frame(self.bar_significance_frame)
            sig_style_row.pack(fill="x", pady=(4,0))
            ttk.Label(sig_style_row, text="Color:").pack(side="left")
            self.param_bar_sig_color_var = StringVar(value="black")
            ttk.Entry(sig_style_row, textvariable=self.param_bar_sig_color_var, width=10).pack(side="left", padx=(4,0))
            ttk.Label(sig_style_row, text="Tamaño:").pack(side="left", padx=(8,0))
            self.param_bar_sig_fontsize_var = StringVar(value="10")
            ttk.Entry(sig_style_row, textvariable=self.param_bar_sig_fontsize_var, width=5).pack(side="left", padx=(4,0))
            
            sig_style_row2 = ttk.Frame(self.bar_significance_frame)
            sig_style_row2.pack(fill="x", pady=(4,0))
            ttk.Label(sig_style_row2, text="Grosor línea:").pack(side="left")
            self.param_bar_sig_linewidth_var = StringVar(value="1.0")
            ttk.Combobox(sig_style_row2, textvariable=self.param_bar_sig_linewidth_var, 
                        values=["0.5", "1.0", "1.5", "2.0"], width=5).pack(side="left", padx=(4,0))
            ttk.Label(sig_style_row2, text="Offset vertical:").pack(side="left", padx=(10,0))
            self.param_bar_sig_voffset_var = StringVar(value="0")
            ttk.Combobox(sig_style_row2, textvariable=self.param_bar_sig_voffset_var, 
                        values=["-0.1", "-0.05", "0", "0.05", "0.1", "0.15", "0.2"], width=6).pack(side="left", padx=(4,0))
            
            # Nueva fila para altura y separación de brackets
            sig_style_row3 = ttk.Frame(self.bar_significance_frame)
            sig_style_row3.pack(fill="x", pady=(4,0))
            ttk.Label(sig_style_row3, text="Altura bracket:").pack(side="left")
            self.param_bar_sig_height_var = StringVar(value="3")
            ttk.Combobox(sig_style_row3, textvariable=self.param_bar_sig_height_var, 
                        values=["1", "2", "3", "4", "5", "6", "8", "10"], width=5).pack(side="left", padx=(4,0))
            ttk.Label(sig_style_row3, text="Separación:").pack(side="left", padx=(10,0))
            self.param_bar_sig_spacing_var = StringVar(value="1.8")
            ttk.Combobox(sig_style_row3, textvariable=self.param_bar_sig_spacing_var, 
                        values=["1.2", "1.5", "1.8", "2.0", "2.5", "3.0", "4.0"], width=5).pack(side="left", padx=(4,0))
            
            ttk.Label(self.bar_significance_frame, text="Comparaciones (una por línea: Cat1,Hue1,Cat2,Hue2,p):").pack(anchor="w", pady=(6,0))
            ttk.Label(self.bar_significance_frame, text="Ejemplo: Materna,Masculino,Paterna,Masculino,0.023", 
                     font=('TkDefaultFont', 8, 'italic')).pack(anchor="w")
            
            self.param_bar_sig_comparisons_text = tk.Text(self.bar_significance_frame, height=4, width=40)
            self.param_bar_sig_comparisons_text.pack(fill="x", pady=(2,4))
            
            # Botón para abrir editor visual de comparaciones
            ttk.Button(self.bar_significance_frame, text="Editor visual de comparaciones...", 
                      command=self._open_significance_editor).pack(fill="x", pady=(2,0))

            mode_combo.bind("<<ComboboxSelected>>", lambda _evt: self._refresh_bar_mode_controls())
            try:
                self.param_bar_mode_var.trace_add('write', lambda *_: self._refresh_bar_mode_controls())
            except Exception:
                pass
            self._refresh_bar_mode_controls()

        elif chart_type == "Gráfico de Pirámide":
            self.param_pyr_age_var, _, self.param_pyr_age_recode_var = self._create_variable_selector(
                self.parameter_controls_frame, "Variable de Eje Y (Categorías de Edad):", columnas, allow_recode=True
            )
            self.param_pyr_male_var, _, _ = self._create_variable_selector(
                self.parameter_controls_frame, "Variable de Barra Izquierda (Numérica):", numeric_columns
            )
            self.param_pyr_female_var, _, _ = self._create_variable_selector(
                self.parameter_controls_frame, "Variable de Barra Derecha (Numérica):", numeric_columns
            )

        elif chart_type == "Gráfico Radial":
            ttk.Label(self.parameter_controls_frame, text="Variables Numéricas (3 o más):").pack(anchor="w")
            self.param_radar_vars = []
            for i in range(5):
                var, _, _ = self._create_variable_selector(self.parameter_controls_frame, f"Variable {i+1}:", [""] + numeric_columns)
                self.param_radar_vars.append(var)

        elif chart_type == "Gráfico de Bala":
            self.param_bullet_value_var, _, _ = self._create_variable_selector(self.parameter_controls_frame, "Variable de Valor (Numérica):", numeric_columns)
            self.param_bullet_target_var, _, _ = self._create_variable_selector(self.parameter_controls_frame, "Variable de Objetivo (Numérica):", numeric_columns)
            ttk.Label(self.parameter_controls_frame, text="Rangos (e.g., 20,50,100):").pack(anchor="w")
            self.param_bullet_ranges_var = StringVar()
            ttk.Entry(self.parameter_controls_frame, textvariable=self.param_bullet_ranges_var).pack(fill="x", pady=(0,5))

        elif chart_type == "Gráfico Lollipop":
            self.param_lollipop_x_var, _, self.param_lollipop_x_recode_var = self._create_variable_selector(
                self.parameter_controls_frame, "Variable Categórica:", columnas, allow_recode=True
            )
            self.param_lollipop_y_var, _, _ = self._create_variable_selector(
                self.parameter_controls_frame, "Variable Numérica:", numeric_columns
            )
            ttk.Label(self.parameter_controls_frame, text="Orientación:").pack(anchor="w")
            self.param_lollipop_orientation_var = StringVar(value="Vertical")
            ttk.Combobox(self.parameter_controls_frame, textvariable=self.param_lollipop_orientation_var, values=["Vertical", "Horizontal"], state="readonly").pack(fill="x", pady=(0,5))

        elif chart_type == "Forest Plot (Comparaciones)":
            self.param_forest_y_var, _, self.param_forest_y_recode_var = self._create_variable_selector(
                self.parameter_controls_frame, "Variable Categórica (Filas/Grupos):", columnas, allow_recode=True
            )
            self.param_forest_value_var, _, _ = self._create_variable_selector(
                self.parameter_controls_frame, "Variable Numérica (Prevalencia/Media):", numeric_columns
            )
            self.param_forest_hue_var, _, self.param_forest_hue_recode_var = self._create_variable_selector(
                self.parameter_controls_frame, "Variable de Comparación (Opcional):", [""] + columnas, allow_recode=True
            )
            ttk.Label(self.parameter_controls_frame, text="Prueba de Normalidad:").pack(anchor="w")
            self.param_forest_norm_mode_var = StringVar(value="Automático")
            ttk.Combobox(self.parameter_controls_frame, textvariable=self.param_forest_norm_mode_var, values=["Automático", "Shapiro-Wilk", "Kolmogorov-Smirnov"], state="readonly").pack(fill="x", pady=(0,5))
            ttk.Label(self.parameter_controls_frame, text="Prueba de Diferencias:").pack(anchor="w")
            self.param_forest_comp_mode_var = StringVar(value="Automático")
            ttk.Combobox(self.parameter_controls_frame, textvariable=self.param_forest_comp_mode_var, values=["Automático", "Paramétrico", "No paramétrico"], state="readonly").pack(fill="x", pady=(0,5))
            self.param_forest_show_ci_var = tk.BooleanVar(value=True)
            ttk.Checkbutton(self.parameter_controls_frame, text="Mostrar intervalos de confianza (95%)", variable=self.param_forest_show_ci_var).pack(anchor="w")
            self.param_forest_show_sig_var = tk.BooleanVar(value=True)
            ttk.Checkbutton(self.parameter_controls_frame, text="Marcar significancia (* p<0.05, ** p<0.01, *** p<0.001)", variable=self.param_forest_show_sig_var).pack(anchor="w")
        
        elif chart_type == "Mapa de Árbol":
            self.param_treemap_values_var, _, _ = self._create_variable_selector(
                self.parameter_controls_frame, "Columna de Valores (Tamaño):", numeric_columns
            )
            self.param_treemap_names_var, _, self.param_treemap_names_recode_var = self._create_variable_selector(
                self.parameter_controls_frame, "Columna de Nombres (Etiquetas):", columnas, allow_recode=True
            )

        elif chart_type == "Polígonos de Frecuencia":
            self.param_poly_var, _, _ = self._create_variable_selector(self.parameter_controls_frame, "Variable Numérica:", numeric_columns)
            self.param_poly_hue_var, _, self.param_poly_hue_recode_var = self._create_variable_selector(self.parameter_controls_frame, "Variable Categórica (Comparación):", [""] + columnas, allow_recode=True)

        elif chart_type == "Gráfico de Líneas / Área":
            self.param_line_x_var, _, self.param_line_x_recode_var = self._create_variable_selector(
                self.parameter_controls_frame, "Variable X (Eje X):", columnas, allow_recode=True
            )
            self.param_line_y_var, _, self.param_line_y_recode_var = self._create_variable_selector(
                self.parameter_controls_frame, "Variable Y (Eje Y):", columnas, allow_recode=True
            )
            self.param_line_hue_var, _, self.param_line_hue_recode_var = self._create_variable_selector(
                self.parameter_controls_frame, "Agrupación (Color):", [""] + columnas, allow_recode=True
            )
            self.param_area_fill_var = tk.BooleanVar(value=False)
            ttk.Checkbutton(self.parameter_controls_frame, text="Rellenar como Gráfico de Área", variable=self.param_area_fill_var).pack(anchor="w")

            ttk.Label(self.parameter_controls_frame, text="Estilo de Línea:").pack(anchor="w")
            self.param_line_style_var = StringVar(value="Sólida")
            ttk.Combobox(self.parameter_controls_frame, textvariable=self.param_line_style_var,
                         values=["Sólida", "Punteada", "Rayada", "Punto-Raya"], state="readonly").pack(fill="x", pady=(0,5))

            ttk.Label(self.parameter_controls_frame, text="Grosor de Línea:").pack(anchor="w")
            self.param_line_width_var = StringVar(value="2.0")
            ttk.Entry(self.parameter_controls_frame, textvariable=self.param_line_width_var).pack(fill="x", pady=(0,5))

            self.param_line_markers_var = tk.BooleanVar(value=False)
            ttk.Checkbutton(self.parameter_controls_frame, text="Mostrar marcadores en cada punto", variable=self.param_line_markers_var).pack(anchor="w")

            self.param_line_ci_var = tk.BooleanVar(value=False)
            ttk.Checkbutton(self.parameter_controls_frame, text="Mostrar banda de confianza (95%)", variable=self.param_line_ci_var).pack(anchor="w")

            ttk.Label(self.parameter_controls_frame, text="Agregación (para X repetidos):").pack(anchor="w")
            self.param_line_estimator_var = StringVar(value="mean")
            ttk.Combobox(self.parameter_controls_frame, textvariable=self.param_line_estimator_var,
                         values=["mean", "median", "sum", "min", "max", "Ninguno"], state="readonly").pack(fill="x", pady=(0,5))

        elif chart_type == "Gráfico de Distribución":
            self.param_dist_y_var, _, _ = self._create_variable_selector(self.parameter_controls_frame, "Variable Numérica (Eje Y):", numeric_columns)
            self.param_dist_x_var, _, self.param_dist_x_recode_var = self._create_variable_selector(self.parameter_controls_frame, "Variable Categórica (Eje X, Opcional):", [""] + columnas, allow_recode=True)
            self.param_dist_hue_var, _, self.param_dist_hue_recode_var = self._create_variable_selector(self.parameter_controls_frame, "Agrupar por Color (Opcional):", [""] + columnas, allow_recode=True)
            ttk.Label(self.parameter_controls_frame, text="Agrupar por Tamaño (Opcional):").pack(anchor="w")
            self.param_dist_size_var = StringVar(value="")
            ttk.Combobox(self.parameter_controls_frame, textvariable=self.param_dist_size_var, values=[""] + columnas, state="readonly").pack(fill="x", pady=(0,5))
            ttk.Label(self.parameter_controls_frame, text="Agrupar por Forma (Opcional):").pack(anchor="w")
            self.param_dist_style_var = StringVar(value="")
            ttk.Combobox(self.parameter_controls_frame, textvariable=self.param_dist_style_var, values=[""] + columnas, state="readonly").pack(fill="x", pady=(0,8))

            layers_frame = ttk.LabelFrame(self.parameter_controls_frame, text="Capas de Visualización", padding="5")
            layers_frame.pack(fill="x", expand=True, pady=(10, 0))

            ttk.Label(layers_frame, text="Tipo de Puntos:").pack(anchor="w")
            self.param_dist_point_type_var = StringVar(value='Strip (Jitter)')
            ttk.Combobox(layers_frame, textvariable=self.param_dist_point_type_var, values=['Ninguno', 'Strip (Jitter)', 'Swarm', 'Center', 'Hex', 'Square'], state="readonly").pack(fill="x", pady=(0,5))
            ttk.Label(layers_frame, text="Método de Acomodo:").pack(anchor="w")
            self.param_dist_layout_method_var = StringVar(value='Auto')
            ttk.Combobox(layers_frame, textvariable=self.param_dist_layout_method_var, values=['Auto', 'Center', 'Hex', 'Square'], state="readonly").pack(fill="x", pady=(0,5))
            ttk.Label(layers_frame, text="Ancho de Jitter Aleatorio:").pack(anchor="w")
            self.param_dist_jitter_width_var = StringVar(value='0.20')
            ttk.Entry(layers_frame, textvariable=self.param_dist_jitter_width_var).pack(fill="x", pady=(0,5))
            # Option to connect category points with a line (pointplot join)
            self.param_dist_connect_points_var = tk.BooleanVar(value=False)
            ttk.Checkbutton(layers_frame, text="Conectar puntos con línea (unir medias)", variable=self.param_dist_connect_points_var).pack(anchor="w")

            self.param_dist_show_box_var = tk.BooleanVar(value=False)
            ttk.Checkbutton(layers_frame, text="Mostrar Diagrama de Caja", variable=self.param_dist_show_box_var).pack(anchor="w")

            self.param_dist_show_violin_var = tk.BooleanVar(value=False)
            ttk.Checkbutton(layers_frame, text="Mostrar Gráfico de Violín", variable=self.param_dist_show_violin_var).pack(anchor="w")

            self.param_dist_raincloud_var = tk.BooleanVar(value=False)
            ttk.Checkbutton(layers_frame, text="Estilo Raincloud (medio violín)", variable=self.param_dist_raincloud_var).pack(anchor="w")
            # Mostrar histograma cuando no hay X seleccionado
            self.param_dist_show_hist_var = tk.BooleanVar(value=False)
            ttk.Checkbutton(layers_frame, text="Mostrar Histograma (cuando no hay X)", variable=self.param_dist_show_hist_var).pack(anchor="w")
            # Mostrar resumen (punto + IC) por categoría
            self.param_dist_show_summary_var = tk.BooleanVar(value=False)
            ttk.Checkbutton(layers_frame, text="Mostrar Resumen (punto + IC)", variable=self.param_dist_show_summary_var).pack(anchor="w")
            # Medio-violín (semi-violin) por categoría (cuando está activado, reemplaza violin completeness)
            self.param_dist_half_violin_var = tk.BooleanVar(value=False)
            ttk.Checkbutton(layers_frame, text="Medio-Violín (semi-violin)", variable=self.param_dist_half_violin_var).pack(anchor="w")
            # Medio-violín con caja y puntos (overlay)
            self.param_dist_half_violin_boxpoints_var = tk.BooleanVar(value=False)
            ttk.Checkbutton(layers_frame, text="Medio-Violín + Caja y Puntos", variable=self.param_dist_half_violin_boxpoints_var).pack(anchor="w")
            # Colapsar categorías en un solo tick y seleccionar paleta de grupo
            self.param_dist_collapse_var = tk.BooleanVar(value=False)
            ttk.Checkbutton(layers_frame, text="Colapsar categorías en un solo tick (usar hue)", variable=self.param_dist_collapse_var).pack(anchor="w")
            ttk.Label(layers_frame, text="Paleta para Grupos:").pack(anchor="w")
            self.param_dist_group_palette_var = StringVar(value='deep')
            group_palettes = ['deep', 'muted', 'pastel', 'bright', 'dark', 'colorblind', 'viridis', 'plasma']
            ttk.Combobox(layers_frame, textvariable=self.param_dist_group_palette_var, values=group_palettes, state="readonly").pack(fill="x", pady=(0,5))
            ttk.Button(layers_frame, text="Asignar colores por nivel", command=lambda: self._open_group_color_picker()).pack(fill="x", pady=(3,5))

        elif chart_type == "Diagrama de Tallo y Hojas":
            self.param_stem_var, _, _ = self._create_variable_selector(self.parameter_controls_frame, "Variable Numérica:", numeric_columns)

        elif chart_type == "Mapa de Calor" or chart_type == "Correlograma":
            ttk.Label(self.parameter_controls_frame, text="Seleccione variables numéricas:").pack(anchor="w")
            self.param_multi_select_vars = []
            for i in range(10):
                var, _, _ = self._create_variable_selector(self.parameter_controls_frame, f"Variable {i+1}:", [""] + numeric_columns)
                self.param_multi_select_vars.append(var)

        elif chart_type == "Gráfico de Coordenadas Paralelas":
            self.param_pc_class_var, _, self.param_pc_class_recode_var = self._create_variable_selector(
                self.parameter_controls_frame, "Variable de Clase (Color):", columnas, allow_recode=True
            )
            ttk.Label(self.parameter_controls_frame, text="Variables Numéricas a Graficar:").pack(anchor="w")
            self.param_pc_vars = []
            for i in range(10):
                var, _, _ = self._create_variable_selector(self.parameter_controls_frame, f"Variable {i+1}:", [""] + numeric_columns)
                self.param_pc_vars.append(var)

        elif chart_type == "Gráfico de Cascada":
            self.param_waterfall_label_var, _, self.param_waterfall_label_recode_var = self._create_variable_selector(
                self.parameter_controls_frame, "Variable de Categorías (Etiquetas):", columnas, allow_recode=True
            )
            self.param_waterfall_value_var, _, _ = self._create_variable_selector(
                self.parameter_controls_frame, "Variable de Valores (Numérica):", numeric_columns
            )

        elif chart_type == "Gráfico de Embudo":
            self.param_funnel_stage_var, _, self.param_funnel_stage_recode_var = self._create_variable_selector(
                self.parameter_controls_frame, "Variable de Etapas (Categórica):", columnas, allow_recode=True
            )
            self.param_funnel_value_var, _, _ = self._create_variable_selector(
                self.parameter_controls_frame, "Variable de Valores (Numérica):", numeric_columns
            )

        elif chart_type == "Diagrama Sunburst":
            self.param_sunburst_level1_var, _, self.param_sunburst_level1_recode_var = self._create_variable_selector(
                self.parameter_controls_frame, "Nivel 1 (Anillo interior):", columnas, allow_recode=True
            )
            self.param_sunburst_level2_var, _, self.param_sunburst_level2_recode_var = self._create_variable_selector(
                self.parameter_controls_frame, "Nivel 2 (Anillo exterior, opcional):", [""] + columnas, allow_recode=True
            )
            self.param_sunburst_value_var, _, _ = self._create_variable_selector(
                self.parameter_controls_frame, "Variable de Tamaño (Numérica, opcional):", [""] + numeric_columns
            )

        elif chart_type == "Diagrama de Marimekko":
            self.param_marimekko_x_var, _, self.param_marimekko_x_recode_var = self._create_variable_selector(
                self.parameter_controls_frame, "Variable X (Categorías de Ancho):", columnas, allow_recode=True
            )
            self.param_marimekko_stack_var, _, self.param_marimekko_stack_recode_var = self._create_variable_selector(
                self.parameter_controls_frame, "Variable de Apilado (Categorías):", columnas, allow_recode=True
            )
            self.param_marimekko_value_var, _, _ = self._create_variable_selector(
                self.parameter_controls_frame, "Variable de Valores (Numérica):", numeric_columns
            )

        elif chart_type == "Dendrograma":
            ttk.Label(self.parameter_controls_frame, text="Variables Numéricas para Agrupamiento:").pack(anchor="w")
            self.param_dendro_vars = []
            for i in range(10):
                var, _, _ = self._create_variable_selector(self.parameter_controls_frame, f"Variable {i+1}:", [""] + numeric_columns)
                self.param_dendro_vars.append(var)
            ttk.Label(self.parameter_controls_frame, text="Método de Enlace:").pack(anchor="w")
            self.param_dendro_method_var = StringVar(value="ward")
            ttk.Combobox(self.parameter_controls_frame, textvariable=self.param_dendro_method_var,
                         values=["ward", "complete", "average", "single"], state="readonly").pack(fill="x", pady=(0,5))
            self.param_dendro_label_var, _, self.param_dendro_label_recode_var = self._create_variable_selector(
                self.parameter_controls_frame, "Etiquetas de Observaciones (Opcional):", [""] + columnas, allow_recode=True
            )

        elif chart_type == "Diagrama de Sankey":
            self.param_sankey_source_var, _, self.param_sankey_source_recode_var = self._create_variable_selector(
                self.parameter_controls_frame, "Variable de Origen:", columnas, allow_recode=True
            )
            self.param_sankey_target_var, _, self.param_sankey_target_recode_var = self._create_variable_selector(
                self.parameter_controls_frame, "Variable de Destino:", columnas, allow_recode=True
            )
            self.param_sankey_value_var, _, _ = self._create_variable_selector(
                self.parameter_controls_frame, "Variable de Flujo (Numérica):", numeric_columns
            )

        elif chart_type == "Gráfico de Flujo":
            self.param_stream_x_var, _, self.param_stream_x_recode_var = self._create_variable_selector(
                self.parameter_controls_frame, "Variable de Eje X (Tiempo/Orden):", columnas, allow_recode=True
            )
            ttk.Label(self.parameter_controls_frame, text="Variables Numéricas (capas):").pack(anchor="w")
            self.param_stream_y_vars = []
            for i in range(8):
                var, _, _ = self._create_variable_selector(self.parameter_controls_frame, f"Capa {i+1}:", [""] + numeric_columns)
                self.param_stream_y_vars.append(var)

        elif chart_type == "Diagrama de Gantt":
            self.param_gantt_task_var, _, self.param_gantt_task_recode_var = self._create_variable_selector(
                self.parameter_controls_frame, "Variable de Tarea:", columnas, allow_recode=True
            )
            self.param_gantt_start_var, _, _ = self._create_variable_selector(
                self.parameter_controls_frame, "Variable de Inicio:", columnas
            )
            self.param_gantt_end_var, _, _ = self._create_variable_selector(
                self.parameter_controls_frame, "Variable de Fin:", columnas
            )
            self.param_gantt_group_var, _, self.param_gantt_group_recode_var = self._create_variable_selector(
                self.parameter_controls_frame, "Variable de Grupo (Opcional, Color):", [""] + columnas, allow_recode=True
            )

        elif chart_type == "Gráfico de Velas":
            self.param_candle_date_var, _, _ = self._create_variable_selector(
                self.parameter_controls_frame, "Variable de Fecha:", columnas
            )
            self.param_candle_open_var, _, _ = self._create_variable_selector(
                self.parameter_controls_frame, "Apertura (Open):", numeric_columns
            )
            self.param_candle_high_var, _, _ = self._create_variable_selector(
                self.parameter_controls_frame, "Máximo (High):", numeric_columns
            )
            self.param_candle_low_var, _, _ = self._create_variable_selector(
                self.parameter_controls_frame, "Mínimo (Low):", numeric_columns
            )
            self.param_candle_close_var, _, _ = self._create_variable_selector(
                self.parameter_controls_frame, "Cierre (Close):", numeric_columns
            )

        elif chart_type == "Línea de Tiempo":
            self.param_timeline_date_var, _, _ = self._create_variable_selector(
                self.parameter_controls_frame, "Variable de Fecha:", columnas
            )
            self.param_timeline_event_var, _, self.param_timeline_event_recode_var = self._create_variable_selector(
                self.parameter_controls_frame, "Variable de Evento (Etiqueta):", columnas, allow_recode=True
            )
            self.param_timeline_group_var, _, self.param_timeline_group_recode_var = self._create_variable_selector(
                self.parameter_controls_frame, "Variable de Grupo (Opcional, Color):", [""] + columnas, allow_recode=True
            )

        elif chart_type == "Mapa Coroplético":
            self.param_choro_region_var, _, self.param_choro_region_recode_var = self._create_variable_selector(
                self.parameter_controls_frame, "Variable de Región (Nombre/ISO):", columnas, allow_recode=True
            )
            self.param_choro_value_var, _, _ = self._create_variable_selector(
                self.parameter_controls_frame, "Variable de Valor (Numérica):", numeric_columns
            )
            ttk.Label(self.parameter_controls_frame, text="Tipo de Región:").pack(anchor="w")
            self.param_choro_region_type_var = StringVar(value="País")
            ttk.Combobox(self.parameter_controls_frame, textvariable=self.param_choro_region_type_var,
                         values=["País", "Estado/Provincia"], state="readonly").pack(fill="x", pady=(0,5))

        elif chart_type == "Mapa de Burbujas":
            self.param_bubble_map_lat_var, _, _ = self._create_variable_selector(
                self.parameter_controls_frame, "Variable de Latitud:", numeric_columns
            )
            self.param_bubble_map_lon_var, _, _ = self._create_variable_selector(
                self.parameter_controls_frame, "Variable de Longitud:", numeric_columns
            )
            self.param_bubble_map_size_var, _, _ = self._create_variable_selector(
                self.parameter_controls_frame, "Variable de Tamaño (Numérica):", numeric_columns
            )
            self.param_bubble_map_label_var, _, _ = self._create_variable_selector(
                self.parameter_controls_frame, "Variable de Etiqueta (Opcional):", [""] + columnas
            )

        elif chart_type == "Mapa de Puntos":
            self.param_dotmap_lat_var, _, _ = self._create_variable_selector(
                self.parameter_controls_frame, "Variable de Latitud:", numeric_columns
            )
            self.param_dotmap_lon_var, _, _ = self._create_variable_selector(
                self.parameter_controls_frame, "Variable de Longitud:", numeric_columns
            )
            self.param_dotmap_color_var, _, self.param_dotmap_color_recode_var = self._create_variable_selector(
                self.parameter_controls_frame, "Variable de Color (Opcional):", [""] + columnas, allow_recode=True
            )

        elif chart_type == "Diagrama de Venn":
            self.param_venn_group_var, _, self.param_venn_group_recode_var = self._create_variable_selector(
                self.parameter_controls_frame, "Variable de Grupo (2 o 3 grupos):", columnas, allow_recode=True
            )
            self.param_venn_element_var, _, _ = self._create_variable_selector(
                self.parameter_controls_frame, "Variable de Elemento (Identificador):", columnas
            )

        else:
            ttk.Label(self.parameter_controls_frame, text=f"Controles para '{chart_type}' no implementados.").pack()

        # Restaurar paleta global cuando no es Barras
        if chart_type != "Gráfico de Barras":
            try:
                if hasattr(self, 'palette_global_label') and not self.palette_global_label.winfo_manager():
                    self.palette_global_label.pack(anchor="w")
                if hasattr(self, 'palette_global_combo') and not self.palette_global_combo.winfo_manager():
                    self.palette_global_combo.pack(fill="x", pady=(0,5))
                if hasattr(self, 'palette_code_label') and not self.palette_code_label.winfo_manager():
                    self.palette_code_label.pack(anchor="w")
                if hasattr(self, 'palette_code_entry') and not self.palette_code_entry.winfo_manager():
                    self.palette_code_entry.pack(fill="x", pady=(0,5))
                if hasattr(self, 'violin_bw_label') and not self.violin_bw_label.winfo_manager():
                    self.violin_bw_label.pack(anchor="w")
                if hasattr(self, 'violin_bw_entry') and not self.violin_bw_entry.winfo_manager():
                    self.violin_bw_entry.pack(fill="x", pady=(0,5))
                if hasattr(self, 'violin_alpha_label') and not self.violin_alpha_label.winfo_manager():
                    self.violin_alpha_label.pack(anchor="w")
                if hasattr(self, 'violin_alpha_entry') and not self.violin_alpha_entry.winfo_manager():
                    self.violin_alpha_entry.pack(fill="x", pady=(0,5))
            except Exception:
                pass


        # --- Generic Customization Options ---
        customization_frame = ttk.LabelFrame(self.parameter_controls_frame, text="Personalización del Gráfico", padding="10")
        customization_frame.pack(fill="x", expand=True, pady=(15, 0), anchor="n")

        # Title
        ttk.Label(customization_frame, text="Título del Gráfico:").pack(anchor="w")
        self.param_title_var = StringVar()
        ttk.Entry(customization_frame, textvariable=self.param_title_var).pack(fill="x", pady=(0,5))

        # Title color and size
        ttk.Label(customization_frame, text="Color del Título:").pack(anchor="w")
        self.param_title_color_var = StringVar(value='black')
        ttk.Combobox(customization_frame, textvariable=self.param_title_color_var, values=self.color_options, state="readonly").pack(fill="x", pady=(0,5))
        ttk.Label(customization_frame, text="Tamaño del Título:").pack(anchor="w")
        self.param_title_size_var = StringVar(value='12')
        ttk.Entry(customization_frame, textvariable=self.param_title_size_var).pack(fill="x", pady=(0,5))

        # X Label
        ttk.Label(customization_frame, text="Etiqueta Eje X:").pack(anchor="w")
        self.param_xlabel_var = StringVar()
        ttk.Entry(customization_frame, textvariable=self.param_xlabel_var).pack(fill="x", pady=(0,5))

        # Y Label
        ttk.Label(customization_frame, text="Etiqueta Eje Y:").pack(anchor="w")
        self.param_ylabel_var = StringVar()
        ttk.Entry(customization_frame, textvariable=self.param_ylabel_var).pack(fill="x", pady=(0,5))

        # Point Size
        self.point_size_label_pack = {'anchor': 'w'}
        self.point_size_entry_pack = {'fill': "x", 'pady': (0,5)}
        self.point_color_label_pack = {'anchor': 'w'}
        self.point_color_combo_pack = {'fill': "x", 'pady': (0,5)}

        self.point_size_label = ttk.Label(customization_frame, text="Tamaño de Puntos:")
        self.point_size_label.pack(**self.point_size_label_pack)
        self.param_point_size_var = StringVar(value="5")
        self.point_size_entry = ttk.Entry(customization_frame, textvariable=self.param_point_size_var)
        self.point_size_entry.pack(**self.point_size_entry_pack)

        # Point Color
        self.point_color_label = ttk.Label(customization_frame, text="Color de Puntos:")
        self.point_color_label.pack(**self.point_color_label_pack)
        self.param_point_color_var = StringVar(value="auto")
        point_color_choices = ['auto'] + self.color_options
        self.point_color_combo = ttk.Combobox(customization_frame, textvariable=self.param_point_color_var, values=point_color_choices, state="readonly")
        self.point_color_combo.pack(**self.point_color_combo_pack)

        # Grid
        self.param_grid_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(customization_frame, text="Mostrar Cuadrícula", variable=self.param_grid_var).pack(anchor="w")

        # Font family selector (per-chart override)
        try:
            available_fonts = sorted({f.name for f in fm.fontManager.ttflist})
            font_choices = ['Default'] + available_fonts[:60]
        except Exception:
            font_choices = ['Default', 'Arial', 'DejaVu Sans', 'Times New Roman', 'Courier New']
        
        # Ensure "Palatino Linotype" is always an option
        if "Palatino Linotype" not in font_choices:
            font_choices.insert(1, "Palatino Linotype") # Insert after 'Default'
        
        ttk.Label(customization_frame, text="Fuente (Family):").pack(anchor="w")
        self.param_font_family_var = StringVar(value='Default')
        ttk.Combobox(customization_frame, textvariable=self.param_font_family_var, values=font_choices, state="readonly").pack(fill="x", pady=(0,5))

        # Font family, size, and color are now part of the global appearance tab

        # Limits and Ticks
        ttk.Label(customization_frame, text="X lim (min,max):").pack(anchor="w")
        self.param_xlim_var = StringVar()
        ttk.Entry(customization_frame, textvariable=self.param_xlim_var).pack(fill="x", pady=(0,5))
        ttk.Label(customization_frame, text="Y lim (min,max):").pack(anchor="w")
        self.param_ylim_var = StringVar()
        ttk.Entry(customization_frame, textvariable=self.param_ylim_var).pack(fill="x", pady=(0,5))

        # --- Cortes de Eje (Axis Breaks) ---
        breaks_frame = ttk.LabelFrame(customization_frame, text="Cortes de Eje (Axis Breaks)", padding="5")
        breaks_frame.pack(fill="x", expand=True, pady=(5, 0))
        
        ttk.Label(breaks_frame, text="Cortes eje Y (desde-hasta, desde2-hasta2):").pack(anchor="w")
        ttk.Label(breaks_frame, text="Ej: 100-900 (corta de 100 a 900)", font=('TkDefaultFont', 8, 'italic')).pack(anchor="w")
        self.param_y_breaks_var = StringVar()
        ttk.Entry(breaks_frame, textvariable=self.param_y_breaks_var).pack(fill="x", pady=(0,4))
        
        ttk.Label(breaks_frame, text="Cortes eje X (desde-hasta):").pack(anchor="w")
        self.param_x_breaks_var = StringVar()
        ttk.Entry(breaks_frame, textvariable=self.param_x_breaks_var).pack(fill="x", pady=(0,4))
        
        # Controles de estilo para las marcas de corte
        break_style_row = ttk.Frame(breaks_frame)
        break_style_row.pack(fill="x", pady=(4,0))
        
        ttk.Label(break_style_row, text="Tamaño marca:").pack(side="left", padx=(0,2))
        self.param_break_size_var = StringVar(value="1.5")
        ttk.Combobox(break_style_row, textvariable=self.param_break_size_var, 
                     values=["0.5", "1.0", "1.5", "2.0", "2.5", "3.0"], width=5).pack(side="left", padx=(0,10))
        
        ttk.Label(break_style_row, text="Grosor línea:").pack(side="left", padx=(0,2))
        self.param_break_lw_var = StringVar(value="1.0")
        ttk.Combobox(break_style_row, textvariable=self.param_break_lw_var, 
                     values=["0.5", "1.0", "1.5", "2.0", "2.5"], width=5).pack(side="left")

        # --- Advanced Customization Options ---
        adv_custom_frame = ttk.LabelFrame(customization_frame, text="Opciones Avanzadas de Ejes y Tamaño", padding="5")
        adv_custom_frame.pack(fill="x", expand=True, pady=(10, 0))

        ttk.Label(adv_custom_frame, text="Rotación Etiquetas X:").grid(row=0, column=0, padx=5, pady=2, sticky="w")
        self.param_tick_rotation_var = StringVar(value="0")
        ttk.Combobox(adv_custom_frame, textvariable=self.param_tick_rotation_var, values=["0", "30", "45", "90"], state="readonly", width=8).grid(row=0, column=1, padx=5, pady=2, sticky="w")

        ttk.Label(adv_custom_frame, text="Rotación Etiquetas Y:").grid(row=0, column=2, padx=5, pady=2, sticky="w")
        self.param_y_tick_rotation_var = StringVar(value="0")
        ttk.Combobox(adv_custom_frame, textvariable=self.param_y_tick_rotation_var, values=["0", "30", "45", "90"], state="readonly", width=8).grid(row=0, column=3, padx=5, pady=2, sticky="w")

        ttk.Label(adv_custom_frame, text="Escala Eje X:").grid(row=1, column=0, padx=5, pady=2, sticky="w")
        self.param_xscale_var = StringVar(value="linear")
        ttk.Combobox(adv_custom_frame, textvariable=self.param_xscale_var, values=["linear", "log"], state="readonly", width=8).grid(row=1, column=1, padx=5, pady=2, sticky="w")

        ttk.Label(adv_custom_frame, text="Escala Eje Y:").grid(row=1, column=2, padx=5, pady=2, sticky="w")
        self.param_yscale_var = StringVar(value="linear")
        ttk.Combobox(adv_custom_frame, textvariable=self.param_yscale_var, values=["linear", "log"], state="readonly", width=8).grid(row=1, column=3, padx=5, pady=2, sticky="w")

        ttk.Label(adv_custom_frame, text="Ancho Gráfica (pulgadas):").grid(row=2, column=0, padx=5, pady=2, sticky="w")
        self.param_fig_width_var = StringVar(value="8")
        ttk.Entry(adv_custom_frame, textvariable=self.param_fig_width_var, width=10).grid(row=2, column=1, padx=5, pady=2, sticky="w")

        ttk.Label(adv_custom_frame, text="Alto Gráfica (pulgadas):").grid(row=2, column=2, padx=5, pady=2, sticky="w")
        self.param_fig_height_var = StringVar(value="5")
        ttk.Entry(adv_custom_frame, textvariable=self.param_fig_height_var, width=10).grid(row=2, column=3, padx=5, pady=2, sticky="w")

        ttk.Label(adv_custom_frame, text="Formato etiquetas X:").grid(row=3, column=0, padx=5, pady=2, sticky="w")
        self.param_x_format_var = StringVar(value="auto")
        ttk.Combobox(adv_custom_frame, textvariable=self.param_x_format_var, values=["auto", "normal", "científica"], state="readonly", width=10).grid(row=3, column=1, padx=5, pady=2, sticky="w")

        ttk.Label(adv_custom_frame, text="Formato etiquetas Y:").grid(row=3, column=2, padx=5, pady=2, sticky="w")
        self.param_y_format_var = StringVar(value="auto")
        ttk.Combobox(adv_custom_frame, textvariable=self.param_y_format_var, values=["auto", "normal", "científica"], state="readonly", width=10).grid(row=3, column=3, padx=5, pady=2, sticky="w")


        # Seaborn Theme
        ttk.Label(customization_frame, text="Tema de Seaborn:").pack(anchor="w")
        self.param_theme_var = StringVar(value='whitegrid')
        themes = ['whitegrid', 'darkgrid', 'grid', 'white', 'dark', 'ticks']
        ttk.Combobox(customization_frame, textvariable=self.param_theme_var, values=themes, state="readonly").pack(fill="x", pady=(0,5))

        # Color Palette
        self.palette_global_label = ttk.Label(customization_frame, text="Paleta de Colores:")
        self.param_palette_var = StringVar(value='default')
        palettes = ['default', 'viridis', 'plasma', 'inferno', 'magma', 'cividis', 'deep', 'muted', 'pastel', 'bright', 'dark', 'colorblind']
        self.palette_global_combo = ttk.Combobox(customization_frame, textvariable=self.param_palette_var, values=palettes, state="readonly")
        self.palette_code_label = ttk.Label(customization_frame, text="Código de paleta (hex separados por coma):")
        self.param_palette_code_var = StringVar()
        self.palette_code_entry = ttk.Entry(customization_frame, textvariable=self.param_palette_code_var)
        if chart_type != "Gráfico de Barras":
            self.palette_global_label.pack(anchor="w")
            self.palette_global_combo.pack(fill="x", pady=(0,5))
            self.palette_code_label.pack(anchor="w")
            self.palette_code_entry.pack(fill="x", pady=(0,5))
        # Violin fine controls
        self.violin_bw_label = ttk.Label(customization_frame, text="bw (suavizado KDE) para Violín:")
        previous_bw = getattr(self, 'param_violin_bw_var', None)
        bw_default = previous_bw.get() if isinstance(previous_bw, tk.Variable) else '0.2'
        self.param_violin_bw_var = StringVar(value=bw_default)
        self.violin_bw_entry = ttk.Entry(customization_frame, textvariable=self.param_violin_bw_var)
        if chart_type == "Gráfico de Distribución":
            self.violin_bw_label.pack(anchor="w")
            self.violin_bw_entry.pack(fill="x", pady=(0,5))

        previous_width = getattr(self, 'param_violin_width_var', None)
        width_default = previous_width.get() if isinstance(previous_width, tk.Variable) else '0.8'
        self.param_violin_width_var = StringVar(value=width_default)
        if chart_type == "Gráfico de Distribución":
            ttk.Label(customization_frame, text="Ancho (width) del Violín:").pack(anchor="w")
            ttk.Entry(customization_frame, textvariable=self.param_violin_width_var).pack(fill="x", pady=(0,5))

        self.violin_alpha_label = ttk.Label(customization_frame, text="Alpha (transparencia) del Violín:")
        previous_alpha = getattr(self, 'param_violin_alpha_var', None)
        alpha_default = previous_alpha.get() if isinstance(previous_alpha, tk.Variable) else '0.6'
        self.param_violin_alpha_var = StringVar(value=alpha_default)
        self.violin_alpha_entry = ttk.Entry(customization_frame, textvariable=self.param_violin_alpha_var)
        if chart_type == "Gráfico de Distribución":
            self.violin_alpha_label.pack(anchor="w")
            self.violin_alpha_entry.pack(fill="x", pady=(0,5))

        # Legend controls
        self.param_show_legend_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(customization_frame, text="Mostrar Leyenda", variable=self.param_show_legend_var).pack(anchor="w")
        ttk.Label(customization_frame, text="Posición Leyenda:").pack(anchor="w")
        self.param_legend_loc_var = StringVar(value='best')
        legend_positions = ['best','upper right','upper left','lower left','lower right','right','center left','center right','lower center','upper center','center']
        ttk.Combobox(customization_frame, textvariable=self.param_legend_loc_var, values=legend_positions, state="readonly").pack(fill="x", pady=(0,5))
        ttk.Label(customization_frame, text="Tamaño Fuente Leyenda:").pack(anchor="w")
        self.param_legend_size_var = StringVar(value='10')
        ttk.Entry(customization_frame, textvariable=self.param_legend_size_var).pack(fill="x", pady=(0,5))
        
        # Bootstrap and CI
        ci_frame = ttk.LabelFrame(customization_frame, text="Intervalos de Confianza", padding="5")
        ci_frame.pack(fill="x", expand=True, pady=(10, 0))
        ci_frame.columnconfigure(1, weight=1)
        ci_frame.columnconfigure(3, weight=1)

        self.param_ci_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(ci_frame, text="Calcular IC", variable=self.param_ci_var).grid(row=0, column=0, sticky="w")

        self.param_ci_bootstrap_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(ci_frame, text="Usar Bootstrap", variable=self.param_ci_bootstrap_var).grid(row=0, column=1, columnspan=2, sticky="w")

        ttk.Label(ci_frame, text="Iteraciones:").grid(row=1, column=0, sticky="w")
        self.param_n_boot_var = StringVar(value="1000")
        ttk.Entry(ci_frame, textvariable=self.param_n_boot_var, width=8).grid(row=1, column=1, sticky="w")

        ttk.Label(ci_frame, text="Nivel (%):").grid(row=2, column=0, sticky="w")
        self.param_ci_level_var = StringVar(value="95")
        ttk.Entry(ci_frame, textvariable=self.param_ci_level_var, width=6).grid(row=2, column=1, sticky="w")

        ttk.Label(ci_frame, text="Semilla (opcional):").grid(row=2, column=2, sticky="e")
        self.param_ci_seed_var = StringVar()
        ttk.Entry(ci_frame, textvariable=self.param_ci_seed_var, width=10).grid(row=2, column=3, sticky="w")

        ttk.Label(ci_frame, text="Visualización:").grid(row=3, column=0, sticky="w")
        self.param_ci_style_var = StringVar(value="Barras de error")
        ttk.Combobox(ci_frame, textvariable=self.param_ci_style_var,
                     values=["Barras de error", "Banda"], state="readonly")\
            .grid(row=3, column=1, sticky="we", pady=(2,0))

        ttk.Label(ci_frame, text="Color IC:").grid(row=3, column=2, sticky="e")
        self.param_ci_color_var = StringVar(value="#333333")
        ttk.Entry(ci_frame, textvariable=self.param_ci_color_var, width=10).grid(row=3, column=3, sticky="w")

        ttk.Label(ci_frame, text="Alpha banda:").grid(row=4, column=0, sticky="w")
        self.param_ci_alpha_var = StringVar(value="0.25")
        ttk.Entry(ci_frame, textvariable=self.param_ci_alpha_var, width=6).grid(row=4, column=1, sticky="w")

        ttk.Label(ci_frame, text="Tamaño punta:").grid(row=4, column=2, sticky="e")
        self.param_ci_capsize_var = StringVar(value="4")
        ttk.Entry(ci_frame, textvariable=self.param_ci_capsize_var, width=6).grid(row=4, column=3, sticky="w")

        ttk.Label(ci_frame, text="Grosor línea IC:").grid(row=5, column=0, sticky="w")
        self.param_ci_linewidth_var = StringVar(value="1.2")
        ttk.Entry(ci_frame, textvariable=self.param_ci_linewidth_var, width=6).grid(row=5, column=1, sticky="w")

        ttk.Label(ci_frame, text="Marcador centro:").grid(row=5, column=2, sticky="e")
        self.param_ci_marker_var = StringVar(value="ninguno")
        ttk.Combobox(ci_frame, textvariable=self.param_ci_marker_var,
                     values=["ninguno", "circulo", "cuadro", "triangulo"], state="readonly")\
            .grid(row=5, column=3, sticky="we")

        ttk.Label(ci_frame, text="Tamaño marcador:").grid(row=6, column=0, sticky="w")
        self.param_ci_marker_size_var = StringVar(value="6")
        ttk.Entry(ci_frame, textvariable=self.param_ci_marker_size_var, width=6).grid(row=6, column=1, sticky="w")

        self._set_point_control_visibility(chart_type)

    def _refresh_bar_mode_controls(self):
        mode_var = getattr(self, 'param_bar_mode_var', None)
        if mode_var is None:
            return

        mode_value = mode_var.get() if callable(getattr(mode_var, 'get', None)) else 'Simple'
        mode = (mode_value or 'Simple').strip().lower()

        show_y_selector = True
        show_hue_selector = True
        show_bar_colors = mode in ('simple', 'agrupado', 'apilado')
        show_stacked_options = mode == 'apilado'
        show_segment_controls = mode == 'segmentado'
        disable_y_selector = False
        disable_hue_selector = False

        anchor_order_selectors = [getattr(self, 'bar_orientation_label', None)]

        def get_default_kwargs(obj, attr_name, fallback):
            return getattr(obj, attr_name, fallback) if obj is not None else fallback

        def pack_with_anchor(widget, pack_kwargs, anchor_candidates):
            if widget is None or widget.winfo_manager() == 'pack':
                return
            kwargs = dict(pack_kwargs or {})
            for anchor in anchor_candidates or []:
                if anchor is not None and anchor.winfo_manager() == 'pack':
                    kwargs['before'] = anchor
                    break
            widget.pack(**kwargs)

        def toggle_selector(var_obj, should_show):
            if var_obj is None:
                return
            label = getattr(var_obj, '_label_widget', None)
            frame = getattr(var_obj, '_widget_frame', None)
            recode_frame = getattr(var_obj, '_recode_frame', None)
            label_kwargs = getattr(var_obj, '_label_pack_kwargs', {'anchor': 'w'})
            frame_kwargs = getattr(var_obj, '_frame_pack_kwargs', {'fill': "x", 'pady': (0,5)})
            recode_kwargs = getattr(var_obj, '_recode_pack_kwargs', {'fill': "x", 'pady': (0,5)})

            if should_show:
                if label is not None and label.winfo_manager() != 'pack':
                    pack_with_anchor(label, label_kwargs, anchor_order_selectors)
                if frame is not None and frame.winfo_manager() != 'pack':
                    pack_with_anchor(frame, frame_kwargs, anchor_order_selectors)
                if recode_frame is not None and recode_frame.winfo_manager() != 'pack':
                    pack_with_anchor(recode_frame, recode_kwargs, anchor_order_selectors)
            else:
                if label is not None and label.winfo_manager() == 'pack':
                    label.pack_forget()
                if frame is not None and frame.winfo_manager() == 'pack':
                    frame.pack_forget()
                if recode_frame is not None and recode_frame.winfo_manager() == 'pack':
                    recode_frame.pack_forget()

        def toggle_frame(frame_obj, should_show, anchor_candidates=None):
            if frame_obj is None:
                return
            pack_kwargs = get_default_kwargs(frame_obj, '_default_pack_kwargs', {'fill': "x", 'expand': True, 'pady': (0,5)})
            if should_show:
                pack_with_anchor(frame_obj, pack_kwargs, anchor_candidates)
            else:
                if frame_obj.winfo_manager() == 'pack':
                    frame_obj.pack_forget()

        toggle_selector(getattr(self, 'param_bar_y_var', None), show_y_selector)
        toggle_selector(getattr(self, 'param_bar_hue_var', None), show_hue_selector)

        color_anchor = [getattr(self, 'bar_stacked_options_frame', None), getattr(self, 'bar_segment_frame', None), getattr(self, 'bar_segment_color_frame', None)]
        stacked_anchor = [getattr(self, 'bar_segment_frame', None), getattr(self, 'bar_segment_color_frame', None)]
        segment_anchor = [getattr(self, 'bar_segment_color_frame', None)]

        toggle_frame(getattr(self, 'bar_color_frame', None), show_bar_colors, color_anchor)
        toggle_frame(getattr(self, 'bar_stacked_options_frame', None), show_stacked_options, stacked_anchor)
        toggle_frame(getattr(self, 'bar_segment_frame', None), show_segment_controls, segment_anchor)
        toggle_frame(getattr(self, 'bar_segment_color_frame', None), show_segment_controls, None)

        self._set_selector_state(getattr(self, 'param_bar_y_var', None), not disable_y_selector)
        self._set_selector_state(getattr(self, 'param_bar_hue_var', None), not disable_hue_selector)

    def _set_point_control_visibility(self, chart_type):
        point_widgets = [
            (getattr(self, 'point_size_label', None), getattr(self, 'point_size_label_pack', {'anchor': 'w'})),
            (getattr(self, 'point_size_entry', None), getattr(self, 'point_size_entry_pack', {'fill': "x", 'pady': (0,5)})),
            (getattr(self, 'point_color_label', None), getattr(self, 'point_color_label_pack', {'anchor': 'w'})),
            (getattr(self, 'point_color_combo', None), getattr(self, 'point_color_combo_pack', {'fill': "x", 'pady': (0,5)}))
        ]

        charts_with_points = {
            "Diagrama de Dispersión",
            "Gráfico de Distribución",
            "Gráfico de Líneas / Área",
            "Gráfico Lollipop",
            "Forest Plot (Comparaciones)"
        }
        show_points = chart_type in charts_with_points

        for widget, pack_kwargs in point_widgets:
            if widget is None:
                continue
            if show_points:
                if widget.winfo_manager() != 'pack':
                    widget.pack(**(pack_kwargs or {}))
            else:
                if widget.winfo_manager() == 'pack':
                    widget.pack_forget()

    def _set_selector_state(self, var_obj, enabled):
        if var_obj is None:
            return
        combo = getattr(var_obj, '_selector_widget', None)
        entry = getattr(var_obj, '_display_entry', None)
        recode_var = getattr(var_obj, '_recode_var', None)

        combo_state = 'readonly' if enabled else 'disabled'
        entry_state = 'normal' if enabled else 'disabled'

        if combo is not None and combo.cget('state') != combo_state:
            combo.configure(state=combo_state)
        if entry is not None and entry.cget('state') != entry_state:
            entry.configure(state=entry_state)

        if recode_var is not None:
            recode_entry = getattr(recode_var, '_entry_widget', None)
            if recode_entry is not None and recode_entry.cget('state') != entry_state:
                recode_entry.configure(state=entry_state)

    def _get_ci_settings(self):
        ci_enabled = bool(getattr(self, 'param_ci_var', tk.BooleanVar(value=False)).get())
        use_bootstrap = bool(getattr(self, 'param_ci_bootstrap_var', tk.BooleanVar(value=True)).get())
        settings = {
            'enabled': ci_enabled,
            'bootstrap': use_bootstrap,
            'n_boot': 1000,
            'level': 0.95,
            'seed': None,
            'style': 'error',
            'style_name': 'Barras de error',
            'color': '#333333',
            'alpha': 0.25,
            'capsize': 4.0,
            'linewidth': 1.2,
            'marker': None,
            'marker_size': 6.0
        }
        if not ci_enabled:
            return settings

        def _safe_float(var_name, default):
            raw = getattr(self, var_name, StringVar(value=str(default))).get()
            try:
                return float(raw)
            except Exception:
                self.log(f"Valor inválido para {var_name}: '{raw}', se usará {default}", "WARN")
                return default

        n_boot_raw = getattr(self, 'param_n_boot_var', StringVar(value='1000')).get()
        try:
            n_boot = int(float(n_boot_raw))
        except Exception:
            self.log(f"Iteraciones bootstrap inválidas '{n_boot_raw}', se usará 1000.", "WARN")
            n_boot = 1000
        n_boot = max(100, min(20000, n_boot))

        level_raw = getattr(self, 'param_ci_level_var', StringVar(value='95')).get()
        try:
            level = float(level_raw)
        except Exception:
            self.log(f"Nivel de confianza inválido '{level_raw}', se usará 95%.", "WARN")
            level = 95.0
        level = level / 100.0 if level > 1 else level
        level = min(max(level, 0.5), 0.999)

        seed_raw = getattr(self, 'param_ci_seed_var', StringVar()).get()
        seed_value = None
        if seed_raw.strip():
            try:
                seed_value = int(seed_raw.strip())
            except Exception:
                self.log(f"Semilla inválida '{seed_raw}', se ignorará.", "WARN")

        style_raw = getattr(self, 'param_ci_style_var', StringVar(value='Barras de error')).get() or 'Barras de error'
        style_lower = style_raw.strip().lower()
        if 'band' in style_lower or 'banda' in style_lower:
            style = 'band'
            style_name = 'Banda'
        else:
            style = 'error'
            style_name = 'Barras de error'

        color_raw = getattr(self, 'param_ci_color_var', StringVar(value='#333333')).get() or '#333333'
        color_hex = self._color_to_hex(color_raw)

        alpha_value = _safe_float('param_ci_alpha_var', 0.25)
        alpha_value = min(max(alpha_value, 0.0), 1.0)

        capsize_value = _safe_float('param_ci_capsize_var', 4.0)
        capsize_value = max(0.0, min(20.0, capsize_value))

        linewidth_value = _safe_float('param_ci_linewidth_var', 1.2)
        linewidth_value = max(0.2, min(10.0, linewidth_value))

        marker_raw = getattr(self, 'param_ci_marker_var', StringVar(value='ninguno')).get() or 'ninguno'
        marker_key = marker_raw.strip().lower()
        marker_map = {
            'ninguno': None,
            'circulo': 'o',
            'cuadro': 's',
            'triangulo': '^'
        }
        marker_symbol = marker_map.get(marker_key, None)

        marker_size_value = _safe_float('param_ci_marker_size_var', 6.0)
        marker_size_value = max(1.0, min(20.0, marker_size_value))

        settings.update({
            'n_boot': n_boot,
            'level': level,
            'seed': seed_value,
            'style': style,
            'style_name': style_name,
            'color': color_hex,
            'alpha': alpha_value,
            'capsize': capsize_value,
            'linewidth': linewidth_value,
            'marker': marker_symbol,
            'marker_size': marker_size_value
        })
        return settings

    def _bootstrap_group_means(self, df, group_cols, value_col, settings):
        if df is None or df.empty:
            return {}

        try:
            rng = np.random.default_rng(settings['seed']) if settings.get('seed') is not None else np.random.default_rng()
        except Exception:
            rng = np.random.default_rng()

        lower_q = (1.0 - settings['level']) / 2.0
        upper_q = 1.0 - lower_q
        n_boot = int(settings.get('n_boot', 1000))

        results = {}
        grouped = df.groupby(group_cols, sort=False, dropna=False)[value_col]
        for key, series in grouped:
            values = pd.to_numeric(series, errors='coerce').to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            if values.size < 2:
                continue
            boots = np.empty(n_boot, dtype=float)
            for idx in range(n_boot):
                sample_idx = rng.integers(0, values.size, size=values.size)
                boots[idx] = values[sample_idx].mean()
            center = float(values.mean())
            lower = float(np.quantile(boots, lower_q))
            upper = float(np.quantile(boots, upper_q))
            # Convertir la clave a tupla de strings para consistency
            tuple_key = key if isinstance(key, tuple) else (key,)
            tuple_key = tuple(str(k) if k is not None else None for k in tuple_key)
            results[tuple_key] = (center, lower, upper)
        return results

    def _analytic_group_means(self, df, group_cols, value_col, settings):
        if df is None or df.empty:
            return {}

        try:
            level = float(settings.get('level', 0.95))
            tail = (1.0 - level) / 2.0
            z_value = NormalDist().inv_cdf(1.0 - tail)
        except Exception:
            z_value = 1.96

        results = {}
        grouped = df.groupby(group_cols, sort=False, dropna=False)[value_col]
        for key, series in grouped:
            values = pd.to_numeric(series, errors='coerce').to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            n = values.size
            if n < 2:
                continue
            center = float(values.mean())
            std = float(values.std(ddof=1)) if n > 1 else 0.0
            stderr = std / np.sqrt(n) if n > 0 else 0.0
            margin = z_value * stderr
            lower = center - margin
            upper = center + margin
            # Convertir la clave a tupla de strings para consistency
            tuple_key = key if isinstance(key, tuple) else (key,)
            tuple_key = tuple(str(k) if k is not None else None for k in tuple_key)
            results[tuple_key] = (center, lower, upper)
        return results

    def _draw_bar_ci_elements(self, ax, ordered_items, orientation, settings):
        if not ordered_items:
            return
        patches = [patch for patch in getattr(ax, 'patches', []) if isinstance(patch, Rectangle)]
        if not patches:
            return
        color = settings.get('color', '#333333')
        capsize = settings.get('capsize', 4.0)
        alpha = settings.get('alpha', 0.25)
        style = settings.get('style', 'error')
        linewidth = settings.get('linewidth', 1.2)
        marker = settings.get('marker')
        marker_size = settings.get('marker_size', 6.0)
        if len(patches) != len(ordered_items):
            self.log(f"Advertencia: número de barras ({len(patches)}) y de IC ({len(ordered_items)}) no coincide; se dibujarán las coincidencias mínimas.", "WARN")
        for patch, item in zip(patches, ordered_items):
            key, stats = item
            if not stats:
                continue
            center, lower, upper = stats
            if upper <= lower:
                continue
            if orientation == 'Horizontal':
                y_center = patch.get_y() + patch.get_height() / 2.0
                bar_value = patch.get_width()
                # Usar center del IC calculado como punto de anclaje para simetría correcta
                anchor = center if np.isfinite(center) else bar_value
                if style == 'band':
                    band_left = lower
                    band_width = upper - lower
                    rect = Rectangle((band_left, patch.get_y()), band_width, patch.get_height(),
                                     color=color, alpha=alpha, zorder=patch.get_zorder() + 0.2)
                    ax.add_patch(rect)
                    continue
                err_low = max(0.0, anchor - lower)
                err_high = max(0.0, upper - anchor)
                ax.errorbar(anchor, y_center, xerr=[[err_low], [err_high]],
                            color=color, capsize=capsize, linewidth=linewidth,
                            zorder=patch.get_zorder() + 0.3)
                if marker:
                    ax.scatter(anchor, y_center, color=color, marker=marker,
                               s=marker_size ** 2, zorder=patch.get_zorder() + 0.4)
            else:
                x_center = patch.get_x() + patch.get_width() / 2.0
                bar_value = patch.get_height()
                # Usar center del IC calculado como punto de anclaje para simetría correcta
                anchor = center if np.isfinite(center) else bar_value
                if style == 'band':
                    band_bottom = lower
                    band_height = upper - lower
                    rect = Rectangle((patch.get_x(), band_bottom), patch.get_width(), band_height,
                                     color=color, alpha=alpha, zorder=patch.get_zorder() + 0.2)
                    ax.add_patch(rect)
                    continue
                err_low = max(0.0, anchor - lower)
                err_high = max(0.0, upper - anchor)
                ax.errorbar(x_center, anchor, yerr=[[err_low], [err_high]],
                            color=color, capsize=capsize, linewidth=linewidth,
                            zorder=patch.get_zorder() + 0.3)
                if marker:
                    ax.scatter(x_center, anchor, color=color, marker=marker,
                               s=marker_size ** 2, zorder=patch.get_zorder() + 0.4)

    def _apply_bar_confidence_intervals(self, ax, data, x_col, y_col, hue_col, category_order,
                                        hue_order, orientation, settings, patches=None):
        if not settings.get('enabled'):
            return
        if not y_col or y_col not in data.columns:
            self.log("Intervalos de confianza requieren una variable de valor numérica.", "WARN")
            return
        if orientation is None:
            orientation = 'Vertical'

        relevant_cols = [x_col]
        if hue_col:
            relevant_cols.append(hue_col)
        relevant_cols.append(y_col)

        subset = data[relevant_cols].copy()
        subset[y_col] = pd.to_numeric(subset[y_col], errors='coerce')
        subset = subset.replace([np.inf, -np.inf], np.nan).dropna(subset=[x_col, y_col])
        if subset.empty:
            self.log("No hay datos válidos para calcular IC en las barras.", "WARN")
            return

        group_cols = [x_col] + ([hue_col] if hue_col else [])
        use_bootstrap = settings.get('bootstrap', True)
        if use_bootstrap:
            results = self._bootstrap_group_means(subset, group_cols, y_col, settings)
        else:
            results = self._analytic_group_means(subset, group_cols, y_col, settings)
        if not results:
            self.log("No se pudieron calcular IC (muestras insuficientes).", "WARN")
            return

        def idx_for(value, order_list):
            if not order_list:
                return 0
            value_str = str(value)
            return order_list.index(value_str) if value_str in order_list else len(order_list)

        # Construir la misma secuencia de combinaciones que usa seaborn al dibujar las barras
        # Asegurar que todos los valores sean strings para coincidir con las claves de IC
        cat_list = category_order if category_order else list(dict.fromkeys(data[x_col].astype(str).tolist()))
        cat_list = [str(cat) for cat in cat_list]

        combos = []
        if hue_col:
            raw_hue_list = hue_order if hue_order else list(dict.fromkeys(data[hue_col].astype(str).tolist()))
            hue_list = [str(h) if h is not None else None for h in raw_hue_list]
            # Seaborn dibuja todas las categorías de un hue antes de pasar al siguiente, replicamos ese orden
            for hv in hue_list:
                for cat in cat_list:
                    combos.append((str(cat), str(hv) if hv is not None else None))
        else:
            combos = [(str(cat), None) for cat in cat_list]

        if patches is None:
            patches = [p for p in ax.patches if isinstance(p, Rectangle)]
        else:
            patches = [p for p in patches if isinstance(p, Rectangle)]

        patches = [p for p in patches if (p.get_width() != 0 or p.get_height() != 0)]
        combos = combos[:len(patches)]  # igualar a cantidad de patches dibujados

        ordered_stats = []
        for combo in combos:
            # Construir la clave para buscar en results, asegurando que sea una tupla de strings
            if hue_col:
                key = (str(combo[0]), str(combo[1]) if combo[1] is not None else None)
            else:
                key = (str(combo[0]),)
            stats = results.get(key)
            if stats:
                ordered_stats.append((key, stats))
            else:
                ordered_stats.append((key, None))

        # Si falta alguna estadística, avisar pero seguir con las presentes
        missing = sum(1 for _, s in ordered_stats if s is None)
        if missing:
            self.log(f"IC: {missing} combinaciones sin datos, se omiten en el dibujo.", "WARN")

        self._draw_bar_ci_elements(ax, ordered_stats, orientation, settings)

    def _clear_chart_display(self):
        for widget in self.chart_display_frame.winfo_children():
            widget.destroy()
        self.last_fig = None

    def _save_current_chart(self, fmt='png'):
        if self.last_fig is None:
            messagebox.showinfo("Sin gráfico", "Genere un gráfico antes de guardar.", parent=self.parent_for_dialogs)
            return
        filetypes = [(fmt.upper(), f"*.{fmt}"), ("Todos", "*.*")]
        path = filedialog.asksaveasfilename(defaultextension=f".{fmt}", filetypes=filetypes)
        if not path:
            return
        try:
            self.last_fig.savefig(path, format=fmt, dpi=300, bbox_inches='tight')
            self.log(f"Gráfico guardado en {path}", "SUCCESS")
        except Exception as exc:
            self.log(f"No se pudo guardar el gráfico: {exc}", "ERROR")
            messagebox.showerror("Error", f"No se pudo guardar el gráfico:\n{exc}", parent=self.parent_for_dialogs)

    def _is_categorical(self, obj):
        try:
            checker = getattr(pd.api.types, "is_categorical_dtype", None)
            if callable(checker):
                return bool(checker(obj))
        except Exception:
            pass
        try:
            dtype = getattr(obj, 'dtype', obj)
            return getattr(dtype, 'name', '').startswith('category')
        except Exception:
            return False

    def _set_categorical_tick_labels(self, ax, data, x_col=None, y_col=None):
        if x_col and x_col in data.columns:
            series_x = data[x_col]
            if self._is_categorical(series_x) or pd.api.types.is_object_dtype(series_x):
                if self._is_categorical(series_x):
                    categories = [str(cat) for cat in series_x.cat.categories]
                else:
                    series_no_na = series_x.dropna()
                    categories = [str(val) for val in series_no_na.unique()]
                if categories:
                    self.log(f"Setting x-tick labels para '{x_col}': {categories}", "DEBUG")
                    ax.set_xticks(range(len(categories)))
                    ax.set_xticklabels(categories)

        if y_col and y_col in data.columns:
            series_y = data[y_col]
            if self._is_categorical(series_y) or pd.api.types.is_object_dtype(series_y):
                if self._is_categorical(series_y):
                    categories = [str(cat) for cat in series_y.cat.categories]
                else:
                    series_no_na = series_y.dropna()
                    categories = [str(val) for val in series_no_na.unique()]
                if categories:
                    self.log(f"Setting y-tick labels para '{y_col}': {categories}", "DEBUG")
                    ax.set_yticks(range(len(categories)))
                    ax.set_yticklabels(categories)

    def _call_seaborn_plot(self, plot_func, plot_kwargs):
        attempt_kwargs = dict(plot_kwargs)
        while True:
            try:
                return plot_func(**attempt_kwargs)
            except TypeError as exc:
                msg = str(exc).lower()
                removed = False
                for key in ('errorbar', 'ci'):
                    if key in attempt_kwargs and key in msg:
                        attempt_kwargs.pop(key, None)
                        removed = True
                        break
                if not removed:
                    raise

    def _apply_violin_alpha(self, ax, alpha):
        try:
            if alpha is None:
                return
            for coll in getattr(ax, 'collections', []):
                try:
                    if isinstance(coll, PolyCollection):
                        coll.set_alpha(alpha)
                except Exception:
                    continue
        except Exception:
            pass

    def _overlay_boxplot(self, ax, data, x_col, y_col, order, hue_col, palette_arg, width, edge_color, category_palette=None):
        try:
            box_kwargs = {
                'data': data,
                'x': x_col,
                'y': y_col,
                'order': order,
                'ax': ax,
                'width': width,
                'showfliers': False,
                'whis': 1.5
            }
            if hue_col:
                box_kwargs['hue'] = hue_col
                box_kwargs['dodge'] = True
                if palette_arg is not None:
                    box_kwargs['palette'] = palette_arg
                sns.boxplot(
                    **box_kwargs,
                    boxprops={'facecolor': 'none', 'linewidth': 1.1},
                    medianprops={'linewidth': 1.1},
                    whiskerprops={'linewidth': 1.0},
                    capprops={'linewidth': 1.0}
                )
            else:
                if category_palette:
                    box_kwargs['palette'] = {str(k): v for k, v in category_palette.items()}
                    draw_color = None
                else:
                    draw_color = edge_color if edge_color else '#333333'
                    box_kwargs['color'] = draw_color
                sns.boxplot(
                    **box_kwargs,
                    boxprops={'facecolor': 'none', 'edgecolor': draw_color if draw_color else 'black', 'linewidth': 1.2},
                    medianprops={'color': draw_color if draw_color else 'black', 'linewidth': 1.2},
                    whiskerprops={'color': draw_color if draw_color else 'black', 'linewidth': 1.0},
                    capprops={'color': draw_color if draw_color else 'black', 'linewidth': 1.0}
                )
        except Exception:
            pass

    def _color_to_hex(self, color):
        return shared_color_to_hex(color)

    def _build_category_palette(self, categories, palette_name, explicit_map, fallback_color):
        return shared_build_category_palette(
            categories=categories,
            palette_name=palette_name if palette_name else 'deep',
            explicit_map=explicit_map or {},
            fallback_color=fallback_color,
        )

    def _build_size_mapper(self, series, base_size):
        base_area = (base_size ** 2)
        if series is None:
            return lambda values: np.full(len(values), base_area)
        try:
            numeric_series = pd.to_numeric(series, errors='coerce')
            if numeric_series.notna().any():
                valid = numeric_series.dropna()
                min_v = valid.min()
                max_v = valid.max()
                if not np.isfinite(min_v):
                    min_v = 0.0
                if not np.isfinite(max_v):
                    max_v = min_v + 1.0
                span = max(max_v - min_v, 1e-9)

                def mapper(values):
                    vals = pd.to_numeric(values, errors='coerce').fillna(min_v)
                    norm = (vals - min_v) / span
                    norm = norm.clip(lower=0.0, upper=1.0)
                    factors = 0.6 + norm * 1.2  # scale between 0.6 and 1.8
                    return (base_size * factors.to_numpy()) ** 2

                return mapper
        except Exception:
            pass

        categories = [str(val) for val in pd.unique(series.dropna())]
        if not categories:
            return lambda values: np.full(len(values), base_area)
        factors = np.linspace(0.7, 1.6, num=len(categories)) if len(categories) > 1 else np.array([1.0])
        mapping = {cat: factors[idx] for idx, cat in enumerate(categories)}

        def mapper(values):
            vals = values.astype(str)
            computed = np.array([mapping.get(val, 1.0) for val in vals])
            return (base_size * computed) ** 2

        return mapper

    def _build_marker_mapper(self, series):
        if series is None:
            return lambda values: np.array(['o'] * len(values))
        markers_cycle = ['o', 'X', 's', '^', 'D', 'P', 'x', '+', '*', 'v', '>', '<', 'h', 'H', 'p']
        categories = [str(val) for val in pd.unique(series.dropna())]
        if not categories:
            return lambda values: np.array(['o'] * len(values))
        mapping = {cat: markers_cycle[idx % len(markers_cycle)] for idx, cat in enumerate(categories)}

        def mapper(values):
            vals = values.astype(str)
            return np.array([mapping.get(val, 'o') for val in vals])

        return mapper

    def _estimate_marker_radii(self, ax, point_size):
        try:
            fig = ax.figure
            if fig is None:
                raise ValueError("Missing figure context")
            bbox = ax.get_window_extent()
            if bbox.width <= 0 or bbox.height <= 0:
                raise ValueError("Invalid axis bounding box")
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            x_range = abs(xlim[1] - xlim[0]) if np.all(np.isfinite(xlim)) else 1.0
            y_range = abs(ylim[1] - ylim[0]) if np.all(np.isfinite(ylim)) else 1.0
            pixels_per_point = fig.dpi / 72.0
            diameter_pixels = max(point_size, 1.0) * pixels_per_point
            radius_x = (diameter_pixels / max(bbox.width, 1.0)) * x_range * 0.5
            radius_y = (diameter_pixels / max(bbox.height, 1.0)) * y_range * 0.5
            if not np.isfinite(radius_x) or radius_x <= 0:
                radius_x = max(x_range * 0.015, 0.05)
            if not np.isfinite(radius_y) or radius_y <= 0:
                radius_y = max(y_range * 0.015, 0.05)
            return radius_x, radius_y
        except Exception:
            base = max(point_size * 0.01, 0.05)
            return base, base

    def _generate_symmetric_candidates(self, spacing, limit=200):
        yield 0.0
        step = max(spacing, 1e-4)
        for k in range(1, limit):
            yield step * k
            yield -step * k

    def _compute_center_offsets(self, y_values, spacing_x, radius_y):
        n = len(y_values)
        offsets = np.zeros(n, dtype=float)
        if n == 0:
            return offsets
        order = np.argsort(y_values)
        placed = []
        horizontal_step = max(spacing_x, 1e-4)
        vertical_scale = max(radius_y, 1e-4) * 1.05
        for idx in order:
            y_val = y_values[idx]
            assigned = False
            for candidate in self._generate_symmetric_candidates(horizontal_step):
                ok = True
                for px, py in placed:
                    dx = (candidate - px) / (horizontal_step * 1.05)
                    dy = (y_val - py) / vertical_scale
                    if dx * dx + dy * dy < 1.0:
                        ok = False
                        break
                if ok:
                    offsets[idx] = candidate
                    placed.append((candidate, y_val))
                    assigned = True
                    break
            if not assigned:
                offsets[idx] = 0.0
        return offsets

    def _compute_grid_offsets(self, y_values, spacing_x, row_threshold, hex_mode=False):
        n = len(y_values)
        offsets = np.zeros(n, dtype=float)
        if n == 0:
            return offsets
        order = np.argsort(y_values)
        rows = []
        threshold = max(row_threshold, 1e-4)
        for idx in order:
            y_val = y_values[idx]
            allocated = False
            for row in rows:
                if abs(y_val - row['anchor']) <= threshold:
                    row['indices'].append(idx)
                    allocated = True
                    break
            if not allocated:
                rows.append({'anchor': y_val, 'indices': [idx]})
        step = max(spacing_x, 0.04)
        for row_idx, row in enumerate(rows):
            members = row['indices']
            count = len(members)
            if count == 1:
                offsets[members[0]] = 0.0
                continue
            members_sorted = sorted(members, key=lambda i: y_values[i])
            if count % 2:
                center = count // 2
                row_offsets = [(i - center) * step for i in range(count)]
            else:
                center = count / 2 - 0.5
                row_offsets = [(i - center) * step for i in range(count)]
            if hex_mode and (row_idx % 2 == 1) and count > 1:
                row_offsets = [off + step / 2.0 for off in row_offsets]
            for member_idx, offset in zip(members_sorted, row_offsets):
                offsets[member_idx] = offset
        return offsets

    def _compute_point_offsets(self, y_values, method, spacing_x, row_threshold, radius_x, radius_y):
        values = np.asarray(y_values, dtype=float)
        if values.size == 0:
            return np.array([], dtype=float)
        if method == 'center':
            return self._compute_center_offsets(values, max(spacing_x, radius_x * 1.2), radius_y)
        if method == 'hex':
            return self._compute_grid_offsets(values, max(spacing_x, radius_x * 1.6), row_threshold, hex_mode=True)
        if method == 'square':
            return self._compute_grid_offsets(values, max(spacing_x, radius_x * 1.6), row_threshold, hex_mode=False)
        return np.zeros_like(values, dtype=float)

    def _resolve_point_color(self, hue_value, hue_index, palette_dict, palette_list, fallback):
        if palette_dict:
            if hue_value in palette_dict and palette_dict[hue_value]:
                return palette_dict[hue_value]
            hue_key = str(hue_value)
            if hue_key in palette_dict and palette_dict[hue_key]:
                return palette_dict[hue_key]
        if palette_list:
            try:
                return palette_list[hue_index % len(palette_list)]
            except Exception:
                pass
        return fallback if fallback else 'C0'

    def _plot_custom_point_layout(self, ax, data, x_col, y_col, order, hue_col, palette_arg, category_palette, default_color, point_size, method, size_col=None, size_mapper=None, style_col=None, style_mapper=None):
        method = (method or '').lower()
        if method not in ('center', 'hex', 'square'):
            return
        if data is None or data.empty or x_col not in data.columns or y_col not in data.columns:
            return
        if order:
            categories = [cat for cat in order if cat is not None and not (isinstance(cat, float) and np.isnan(cat))]
        else:
            categories = [cat for cat in pd.unique(data[x_col]) if pd.notna(cat)]
        if not categories:
            return

        category_palette = category_palette or {}
        category_palette_norm = {str(cat): self._color_to_hex(col) for cat, col in category_palette.items()}
        try:
            tick_positions = ax.get_xticks()
            tick_labels = [tick.get_text() for tick in ax.get_xticklabels()]
            label_map = {label: pos for label, pos in zip(tick_labels, tick_positions)}
        except Exception:
            label_map = {}
        enumerated_map = {str(cat): idx for idx, cat in enumerate(categories)}
        radius_x, radius_y = self._estimate_marker_radii(ax, point_size)
        spacing_x = max(radius_x * 1.6, 0.05)
        row_threshold = max(radius_y * 1.4, 0.05)
        base_area = point_size ** 2

        hue_levels = []
        if hue_col and hue_col in data.columns:
            hue_series = data[hue_col]
            hue_dtype = getattr(hue_series, 'dtype', None)
            if self._is_categorical(hue_dtype):
                hue_levels = [lvl for lvl in hue_series.cat.categories if pd.notna(lvl)]
            else:
                hue_levels = [lvl for lvl in pd.unique(hue_series.dropna())]

        palette_dict = None
        palette_list = None
        if hue_levels:
            if isinstance(palette_arg, dict):
                palette_dict = {str(k): self._color_to_hex(v) for k, v in palette_arg.items()}
            elif isinstance(palette_arg, (list, tuple)):
                palette_list = [self._color_to_hex(col) for col in palette_arg]
            elif isinstance(palette_arg, str):
                try:
                    palette_list = list(sns.color_palette(palette_arg, n_colors=len(hue_levels)))
                    palette_list = [self._color_to_hex(col) for col in palette_list]
                except Exception:
                    palette_list = None
            if palette_list is None and palette_dict is None:
                try:
                    palette_list = [self._color_to_hex(col) for col in sns.color_palette('deep', n_colors=len(hue_levels))]
                except Exception:
                    palette_list = None

        if hue_levels:
            if len(hue_levels) > 1:
                dodge_total = max(spacing_x * (len(hue_levels) - 1) * 1.3, spacing_x * 2.5)
                hue_offsets = np.linspace(-dodge_total / 2.0, dodge_total / 2.0, len(hue_levels))
            else:
                hue_offsets = np.array([0.0])
        else:
            hue_offsets = None

        def compute_sizes(series, length):
            if size_mapper:
                try:
                    return size_mapper(series if series is not None else pd.Series([np.nan] * length))
                except Exception:
                    pass
            return np.full(length, base_area)

        def compute_markers(series, length):
            if style_mapper:
                try:
                    marker_arr = style_mapper(series if series is not None else pd.Series([np.nan] * length))
                    return np.array(marker_arr)
                except Exception:
                    pass
            return np.array(['o'] * length)

        for cat in categories:
            subset = data[data[x_col] == cat]
            if subset.empty:
                continue
            numeric_series = pd.to_numeric(subset[y_col], errors='coerce')
            valid_mask = numeric_series.notna()
            if not valid_mask.any():
                continue
            subset_valid = subset.loc[valid_mask]
            y_vals_full = numeric_series[valid_mask].to_numpy()
            base_label = str(cat)
            base_x = label_map.get(base_label)
            if base_x is None:
                base_x = enumerated_map.get(base_label)
            if base_x is None:
                new_position = len(enumerated_map)
                enumerated_map[base_label] = new_position
                base_x = new_position

            base_color = category_palette_norm.get(base_label, default_color if default_color else 'C0')

            if hue_levels:
                for hue_idx, hue_val in enumerate(hue_levels):
                    hue_subset = subset_valid[subset_valid[hue_col] == hue_val]
                    if hue_subset.empty:
                        continue
                    y_numeric = pd.to_numeric(hue_subset[y_col], errors='coerce')
                    valid_local = y_numeric.notna()
                    if not valid_local.any():
                        continue
                    hue_valid = hue_subset.loc[valid_local]
                    y_vals = y_numeric[valid_local].to_numpy()
                    offsets = self._compute_point_offsets(y_vals, method, spacing_x, row_threshold, radius_x, radius_y)
                    size_series = hue_valid[size_col] if size_col and size_col in hue_valid.columns else None
                    sizes = compute_sizes(size_series, len(hue_valid))
                    style_series = hue_valid[style_col] if style_col and style_col in hue_valid.columns else None
                    markers = compute_markers(style_series, len(hue_valid))
                    color = self._resolve_point_color(hue_val, hue_idx, palette_dict, palette_list, base_color)
                    base_offset = hue_offsets[hue_idx] if hue_offsets is not None else 0.0
                    for marker in np.unique(markers):
                        idx_mask = markers == marker
                        if not np.any(idx_mask):
                            continue
                        ax.scatter(base_x + base_offset + offsets[idx_mask], y_vals[idx_mask], s=sizes[idx_mask], marker=marker,
                                   color=color, alpha=0.85, edgecolors='none', linewidths=0, zorder=4)
            else:
                y_vals = y_vals_full
                offsets = self._compute_point_offsets(y_vals, method, spacing_x, row_threshold, radius_x, radius_y)
                size_series = subset_valid[size_col] if size_col and size_col in subset_valid.columns else None
                sizes = compute_sizes(size_series, len(subset_valid))
                style_series = subset_valid[style_col] if style_col and style_col in subset_valid.columns else None
                markers = compute_markers(style_series, len(subset_valid))
                for marker in np.unique(markers):
                    idx_mask = markers == marker
                    if not np.any(idx_mask):
                        continue
                    ax.scatter(base_x + offsets[idx_mask], y_vals[idx_mask], s=sizes[idx_mask], marker=marker,
                               color=base_color, alpha=0.85, edgecolors='none', linewidths=0, zorder=4)

    def _generate_chart(self):
        self._clear_chart_display()
        chart_type = self.chart_type_var.get()

        if self.data is None:
            messagebox.showwarning("Sin Datos", "Por favor, cargue un archivo de datos primero.", parent=self.parent_for_dialogs)
            return
        if not chart_type:
            messagebox.showwarning("Sin Selección", "Por favor, seleccione un tipo de gráfico.", parent=self.parent_for_dialogs)
            return

        filtered_data = self.filter_component.apply_filters()

        if filtered_data is not None:
            recoded_columns = []
            for param_name in dir(self):
                if not param_name.endswith('_recode_var'):
                    continue
                recode_var = getattr(self, param_name)
                if not (recode_var and isinstance(recode_var, tk.StringVar)):
                    continue
                recode_value = recode_var.get()
                col_var_name = param_name.replace('_recode_var', '_var')
                if not hasattr(self, col_var_name):
                    continue
                col_var = getattr(self, col_var_name)
                column_name = col_var.get()
                if not column_name:
                    continue
                if recode_value:
                    filtered_data = self._apply_recode(filtered_data, column_name, recode_value)
                    recoded_columns.append(column_name)
                else:
                    self._recode_orders.pop(column_name, None)
            if recoded_columns:
                try:
                    filtered_data = filtered_data.dropna(subset=recoded_columns)
                except Exception:
                    pass
        if filtered_data is None:
            self.log("Error al aplicar filtros.", "ERROR")
            return
        
        if filtered_data.empty:
            self.log("No quedan datos después de aplicar los filtros.", "WARN")
            messagebox.showinfo("Datos Vacíos", "No quedan datos después de aplicar los filtros.", parent=self.parent_for_dialogs)
            return

        axis_x_col = None
        axis_y_col = None
        tick_data_x = None
        tick_data_y = None

        def mark_axis(axis_name, column_name, data_frame=None, *, force=False):
            nonlocal axis_x_col, axis_y_col, tick_data_x, tick_data_y
            if not column_name:
                return
            df_ref = data_frame if data_frame is not None else filtered_data
            if df_ref is None or column_name not in df_ref.columns:
                return
            try:
                series_ref = df_ref[column_name]
            except Exception:
                return

            try:
                is_cat = self._is_categorical(series_ref)
            except Exception:
                is_cat = False

            try:
                is_obj = pd.api.types.is_object_dtype(series_ref)
            except Exception:
                is_obj = False

            if not (is_cat or is_obj):
                return

            if axis_name == 'x':
                if force or axis_x_col is None:
                    axis_x_col = column_name
                    tick_data_x = df_ref
            elif axis_name == 'y':
                if force or axis_y_col is None:
                    axis_y_col = column_name
                    tick_data_y = df_ref

        self.log(f"Generando gráfico: {chart_type}", "INFO")
        params_used_log = [f"Tipo de Gráfico: {chart_type}"]

        try:
            # --- Get Customization Options ---
            title = self.param_title_var.get()
            xlabel = self.param_xlabel_var.get()
            ylabel = self.param_ylabel_var.get()
            grid = self.param_grid_var.get()
            xlim_raw = self.param_xlim_var.get()
            ylim_raw = self.param_ylim_var.get()
            theme = self.param_theme_var.get()
            palette_raw = self.param_palette_var.get()
            point_size_str = self.param_point_size_var.get()
            try:
                point_size = float(point_size_str) if point_size_str else 5.0
            except Exception:
                self.log(f"Tamaño de puntos inválido '{point_size_str}', se usará 5.", "WARN")
                point_size = 5.0
            if point_size <= 0:
                self.log(f"Tamaño de puntos no positivo ({point_size}); se ajusta a 0.1.", "WARN")
                point_size = 0.1

            point_color_raw = getattr(self, 'param_point_color_var', StringVar(value='auto')).get()
            point_color = point_color_raw.strip() if point_color_raw else ''
            point_color = None if not point_color or point_color.lower() == 'auto' else point_color
            params_used_log.append(f"  Tamaño puntos: {point_size}")
            if point_color:
                params_used_log.append(f"  Color puntos: {point_color}")

            # Violin / Legend fine controls
            try:
                violin_bw = float(getattr(self, 'param_violin_bw_var', StringVar(value='0.2')).get())
            except Exception:
                violin_bw = 0.2
            try:
                violin_width = float(getattr(self, 'param_violin_width_var', StringVar(value='0.8')).get())
            except Exception:
                violin_width = 0.8
            box_overlay_width = max(0.05, min(violin_width * 0.65, 0.95))
            try:
                violin_alpha = float(getattr(self, 'param_violin_alpha_var', StringVar(value='0.6')).get())
            except Exception:
                violin_alpha = 0.6

            show_legend = getattr(self, 'param_show_legend_var', tk.BooleanVar(value=True)).get()
            legend_loc = getattr(self, 'param_legend_loc_var', StringVar(value='best')).get()
            try:
                legend_size = int(getattr(self, 'param_legend_size_var', StringVar(value='10')).get())
            except Exception:
                legend_size = 10

            # New advanced options
            tick_rotation = int(self.param_tick_rotation_var.get())
            try:
                y_tick_rotation = int(self.param_y_tick_rotation_var.get())
            except Exception:
                y_tick_rotation = 0
            x_scale = self.param_xscale_var.get()
            y_scale = self.param_yscale_var.get()
            fig_width = float(self.param_fig_width_var.get())
            fig_height = float(self.param_fig_height_var.get())

            palette_code_raw = getattr(self, 'param_palette_code_var', StringVar(value='')).get() if hasattr(self, 'param_palette_code_var') else ''
            palette_code_raw = (palette_code_raw or '').strip()
            user_palettes_map = self._load_user_palettes()
            palette = None
            if palette_code_raw:
                try:
                    palette = self._parse_custom_colors(palette_code_raw)
                except Exception as exc:
                    self.log(f"No se pudo leer el código de paleta: {exc}", "WARN")
            elif palette_raw and palette_raw.lower().startswith('usuario:'):
                key = palette_raw.split(':', 1)[1].strip()
                palette = user_palettes_map.get(key)
            elif palette_raw != 'default':
                palette = palette_raw
            # Seaborn set_theme accepts style as a positional string in many versions; pass as positional to avoid typing issues

            try:
                sns.set_theme(style=theme, palette=palette)
            except Exception:
                # Fallback: pass style as positional arg
                sns.set_theme(theme, palette=palette)
            
            # Aplicar fuente DESPUÉS de set_theme para que no se sobrescriba
            try:
                font_family_sel = getattr(self, 'param_font_family_var', StringVar(value='Default')).get()
                if font_family_sel and font_family_sel != 'Default':
                    plt.rcParams['font.family'] = font_family_sel
                    plt.rcParams['font.sans-serif'] = [font_family_sel]
                    plt.rcParams['axes.titleweight'] = 'normal'
            except Exception:
                font_family_sel = 'Default'
            
            params_used_log.append(f"  Tema: {theme}")
            if palette:
                params_used_log.append(f"  Paleta: {palette}")
            if palette_code_raw:
                params_used_log.append(f"  Paleta (código): {palette_code_raw}")

            ci_settings = self._get_ci_settings()
            if ci_settings.get('enabled'):
                method_label = "Bootstrap" if ci_settings.get('bootstrap', True) else "Paramétrico"
                iter_text = f" | Iteraciones: {ci_settings['n_boot']}" if ci_settings.get('bootstrap', True) else ""
                params_used_log.append(
                    f"  IC {method_label}: {ci_settings['level'] * 100:.1f}%{iter_text} | Estilo: {ci_settings['style_name']}"
                )
            
            plt_fig, ax = plt.subplots(figsize=(fig_width, fig_height))

            # --- Main Plotting Logic (to be refactored) ---
            # NOTE: This section will be updated next to handle the new consolidated chart types.
            if chart_type == "Diagrama de Dispersión":
                x_col = self.param_x_var.get()
                y_col = self.param_y_var.get()
                
                if not x_col or not y_col:
                    messagebox.showerror("Error", "Debe seleccionar las variables X e Y.", parent=self.parent_for_dialogs)
                    plt.close(plt_fig)
                    return

                params_used_log.extend([f"  X: {x_col}", f"  Y: {y_col}"])

                color_col = self.param_color_var.get()
                size_col = self.param_size_var.get()
                style_col_scatter = getattr(self, 'param_scatter_style_var', None)
                style_col_scatter = style_col_scatter.get() if style_col_scatter else None
                fit_reg = self.param_scatter_fit_reg_var.get()
                show_corr = getattr(self, 'param_scatter_show_corr_var', tk.BooleanVar(value=False)).get()
                marginal = getattr(self, 'param_scatter_marginal_var', StringVar(value='Ninguno')).get()
                try:
                    scatter_alpha = float(getattr(self, 'param_scatter_alpha_var', StringVar(value='0.7')).get())
                except Exception:
                    scatter_alpha = 0.7
                scatter_alpha = max(0.05, min(scatter_alpha, 1.0))

                # If marginal distributions requested, use JointGrid
                if marginal and marginal != "Ninguno":
                    params_used_log.append(f"  Marginales: {marginal}")
                    plt.close(plt_fig)
                    marginal_kind = {'Histograma': 'hist', 'KDE': 'kde', 'Rug': 'rug'}.get(marginal, 'hist')
                    jg_kwargs = {
                        'data': filtered_data, 'x': x_col, 'y': y_col,
                        'height': min(fig_width, fig_height),
                        'marginal_kws': {'fill': True}
                    }
                    if color_col:
                        jg_kwargs['hue'] = color_col
                    try:
                        g = sns.jointplot(**jg_kwargs, kind='scatter', marginal_kind=marginal_kind,
                                          joint_kws={'alpha': scatter_alpha, 's': point_size * 10})
                    except Exception:
                        g = sns.jointplot(**jg_kwargs, kind='scatter',
                                          joint_kws={'alpha': scatter_alpha, 's': point_size * 10})
                    plt_fig = g.figure
                    ax = g.ax_joint
                    if fit_reg:
                        try:
                            x_data = pd.to_numeric(filtered_data[x_col], errors='coerce')
                            y_data = pd.to_numeric(filtered_data[y_col], errors='coerce')
                            mask = x_data.notna() & y_data.notna()
                            if mask.sum() > 2:
                                z = np.polyfit(x_data[mask], y_data[mask], 1)
                                p = np.poly1d(z)
                                x_range = np.linspace(x_data[mask].min(), x_data[mask].max(), 100)
                                ax.plot(x_range, p(x_range), 'r--', linewidth=1.5, alpha=0.8, label=f'y = {z[0]:.3f}x + {z[1]:.3f}')
                                ax.legend(loc='best', fontsize=8)
                        except Exception:
                            pass
                else:
                    # Standard scatter plot
                    plot_kwargs = {
                        'data': filtered_data, 'x': x_col, 'y': y_col, 'ax': ax,
                        'alpha': scatter_alpha
                    }
                    if color_col:
                        plot_kwargs['hue'] = color_col
                        params_used_log.append(f"  Color: {color_col}")
                    if size_col:
                        plot_kwargs['size'] = size_col
                        params_used_log.append(f"  Tamaño: {size_col}")
                    if style_col_scatter:
                        plot_kwargs['style'] = style_col_scatter
                        params_used_log.append(f"  Forma: {style_col_scatter}")

                    if fit_reg:
                        params_used_log.append("  Línea de Regresión: Sí")
                        reg_kwargs = {'data': filtered_data, 'x': x_col, 'y': y_col, 'ax': ax,
                                      'scatter_kws': {'alpha': scatter_alpha}}
                        if color_col:
                            reg_kwargs['scatter_kws']['hue'] = color_col
                        sns.regplot(**reg_kwargs)
                    else:
                        sns.scatterplot(**plot_kwargs)

                # Correlation annotation
                if show_corr:
                    try:
                        x_num = pd.to_numeric(filtered_data[x_col], errors='coerce')
                        y_num = pd.to_numeric(filtered_data[y_col], errors='coerce')
                        mask = x_num.notna() & y_num.notna()
                        if mask.sum() > 2:
                            r, p_val = stats.pearsonr(x_num[mask], y_num[mask])
                            rho, p_rho = stats.spearmanr(x_num[mask], y_num[mask])
                            n = mask.sum()
                            corr_text = f"Pearson r = {r:.3f} (p = {p_val:.2e})\nSpearman ρ = {rho:.3f} (p = {p_rho:.2e})\nn = {n}"
                            params_used_log.append(f"  Pearson r: {r:.3f}, p: {p_val:.2e}")
                            params_used_log.append(f"  Spearman ρ: {rho:.3f}, p: {p_rho:.2e}")
                            ax.annotate(corr_text, xy=(0.02, 0.98), xycoords='axes fraction',
                                        fontsize=8, va='top', ha='left',
                                        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8, edgecolor='gray'))
                    except Exception:
                        pass

                mark_axis('x', x_col)
                mark_axis('y', y_col)

            elif chart_type == "Histograma":
                x_col = self.param_hist_var.get()
                if not x_col:
                    messagebox.showerror("Error", "Debe seleccionar una variable numérica.", parent=self.parent_for_dialogs)
                    plt.close(plt_fig)
                    return

                params_used_log.append(f"  Variable: {x_col}")
                
                hue_col = self.param_hist_hue_var.get()
                bins_str = self.param_hist_bins_var.get()
                show_kde = self.param_hist_kde_var.get()
                orientation = self.param_hist_orientation_var.get()
                hist_stat = getattr(self, 'param_hist_stat_var', StringVar(value='count')).get()
                hist_cumulative = getattr(self, 'param_hist_cumulative_var', tk.BooleanVar(value=False)).get()
                hist_multiple = getattr(self, 'param_hist_multiple_var', StringVar(value='layer')).get()
                hist_show_stats = getattr(self, 'param_hist_show_stats_var', tk.BooleanVar(value=False)).get()
                hist_show_rug = getattr(self, 'param_hist_show_rug_var', tk.BooleanVar(value=False)).get()

                plot_kwargs = {
                    'data': filtered_data,
                    'kde': show_kde,
                    'ax': ax,
                    'stat': hist_stat,
                }
                if hist_cumulative:
                    plot_kwargs['cumulative'] = True
                    params_used_log.append("  Acumulado: Sí")
                params_used_log.append(f"  Estadístico: {hist_stat}")

                if orientation == "Horizontal":
                    plot_kwargs['y'] = x_col
                else:
                    plot_kwargs['x'] = x_col

                if hue_col:
                    plot_kwargs['hue'] = hue_col
                    plot_kwargs['multiple'] = hist_multiple
                    params_used_log.extend([f"  Hue: {hue_col}", f"  Modo múltiple: {hist_multiple}"])

                if bins_str and bins_str.lower() != 'auto':
                    try:
                        plot_kwargs['bins'] = int(bins_str)
                        params_used_log.append(f"  Bins: {bins_str}")
                    except ValueError:
                        self.log(f"Número de bins inválido: '{bins_str}'. Usando 'auto'.", "WARN")
                
                sns.histplot(**plot_kwargs)

                # Rug plot
                if hist_show_rug:
                    rug_kwargs = {'data': filtered_data, 'ax': ax, 'height': 0.05}
                    if orientation == "Horizontal":
                        rug_kwargs['y'] = x_col
                    else:
                        rug_kwargs['x'] = x_col
                    if hue_col:
                        rug_kwargs['hue'] = hue_col
                    try:
                        sns.rugplot(**rug_kwargs)
                    except Exception:
                        pass

                # Stats overlay
                if hist_show_stats:
                    try:
                        num_vals = pd.to_numeric(filtered_data[x_col], errors='coerce').dropna()
                        if len(num_vals) > 0:
                            mean_val = num_vals.mean()
                            median_val = num_vals.median()
                            std_val = num_vals.std()
                            stats_text = f"n={len(num_vals)}\nMedia={mean_val:.3f}\nMediana={median_val:.3f}\nDE={std_val:.3f}"
                            params_used_log.extend([f"  Media: {mean_val:.3f}", f"  Mediana: {median_val:.3f}", f"  DE: {std_val:.3f}"])
                            if orientation != "Horizontal":
                                ax.axvline(mean_val, color='red', linestyle='--', linewidth=1.2, alpha=0.8, label=f'Media ({mean_val:.2f})')
                                ax.axvline(median_val, color='green', linestyle=':', linewidth=1.2, alpha=0.8, label=f'Mediana ({median_val:.2f})')
                            else:
                                ax.axhline(mean_val, color='red', linestyle='--', linewidth=1.2, alpha=0.8, label=f'Media ({mean_val:.2f})')
                                ax.axhline(median_val, color='green', linestyle=':', linewidth=1.2, alpha=0.8, label=f'Mediana ({median_val:.2f})')
                            ax.annotate(stats_text, xy=(0.98, 0.98), xycoords='axes fraction',
                                        fontsize=8, va='top', ha='right',
                                        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8, edgecolor='gray'))
                            ax.legend(loc='best', fontsize=8)
                    except Exception:
                        pass

            elif chart_type == "Gráfico Circular / Anillo":
                label_var = getattr(self, 'param_pie_label_var', None)
                label_col = label_var.get().strip() if label_var else ""
                value_var = getattr(self, 'param_pie_value_var', None)
                value_col = value_var.get().strip() if value_var else ""

                if not label_col:
                    messagebox.showerror("Error", "Seleccione la variable de etiquetas para el gráfico circular.", parent=self.parent_for_dialogs)
                    plt.close(plt_fig)
                    return
                if label_col not in filtered_data.columns:
                    messagebox.showerror("Error", f"La columna '{label_col}' no existe en los datos filtrados.", parent=self.parent_for_dialogs)
                    plt.close(plt_fig)
                    return
                if value_col and value_col not in filtered_data.columns:
                    messagebox.showerror("Error", f"La columna '{value_col}' no existe en los datos filtrados.", parent=self.parent_for_dialogs)
                    plt.close(plt_fig)
                    return

                exclude_blank = getattr(self, 'param_pie_exclude_blank_var', tk.BooleanVar(value=True)).get()
                show_percentage = getattr(self, 'param_pie_show_percent_var', tk.BooleanVar(value=True)).get()
                show_value = getattr(self, 'param_pie_show_value_var', tk.BooleanVar(value=False)).get()
                sort_mode_raw = getattr(self, 'param_pie_sort_mode_var', StringVar(value='Descendente')).get() or 'Descendente'
                sort_mode = sort_mode_raw.strip().lower()
                hole_str = getattr(self, 'param_pie_hole_var', StringVar(value='0.0')).get()
                max_cat_str = getattr(self, 'param_pie_max_categories_var', StringVar(value='0')).get()
                other_label_var = getattr(self, 'param_pie_other_label_var', StringVar(value='Otros'))
                other_label = other_label_var.get().strip() if other_label_var else "Otros"
                if not other_label:
                    other_label = "Otros"

                try:
                    donut_ratio = float(hole_str)
                except Exception:
                    donut_ratio = 0.0
                donut_ratio = min(max(donut_ratio, 0.0), 0.95)

                try:
                    max_categories = int(max_cat_str)
                except Exception:
                    max_categories = 0
                if max_categories < 0:
                    max_categories = 0

                blank_placeholder = "Sin dato"
                label_series = filtered_data[label_col].copy()
                label_series = label_series.fillna('')
                label_series = label_series.astype(str).str.strip()
                pie_df = pd.DataFrame({'_pie_label': label_series})

                if exclude_blank:
                    pie_df = pie_df[pie_df['_pie_label'] != ""]
                else:
                    pie_df.loc[pie_df['_pie_label'] == "", '_pie_label'] = blank_placeholder
                    pie_df['_pie_label'] = pie_df['_pie_label'].fillna(blank_placeholder)

                if value_col:
                    numeric_series = pd.to_numeric(filtered_data[value_col], errors='coerce')
                    pie_df['_pie_value'] = numeric_series
                    pie_df = pie_df.dropna(subset=['_pie_value'])
                
                if pie_df.empty:
                    plt.close(plt_fig)
                    messagebox.showwarning("Sin Datos", "No hay valores válidos para construir el gráfico circular.", parent=self.parent_for_dialogs)
                    return

                if value_col:
                    aggregation = pie_df.groupby('_pie_label', dropna=False)['_pie_value'].sum()
                else:
                    aggregation = pie_df['_pie_label'].value_counts()

                aggregation = aggregation.astype(float)

                if sort_mode == 'ascendente':
                    aggregation = aggregation.sort_values(ascending=True)
                elif sort_mode == 'original':
                    order = pd.Index(pie_df['_pie_label'].drop_duplicates())
                    aggregation = aggregation.reindex(order, fill_value=0)
                else:
                    aggregation = aggregation.sort_values(ascending=False)

                aggregation = aggregation[aggregation > 0]
                if aggregation.empty or aggregation.sum() <= 0:
                    plt.close(plt_fig)
                    messagebox.showwarning("Sin Datos", "Los valores resultantes son cero, no se puede construir el gráfico circular.", parent=self.parent_for_dialogs)
                    return

                if max_categories and max_categories < len(aggregation):
                    keep_count = max(1, max_categories - 1)
                    primary = aggregation.iloc[:keep_count].copy()
                    remainder = aggregation.iloc[keep_count:].sum()
                    if remainder > 0:
                        primary[other_label] = primary.get(other_label, 0.0) + remainder
                    aggregation = primary

                labels = [lbl if lbl else blank_placeholder for lbl in aggregation.index.tolist()]
                values = aggregation.to_numpy(dtype=float)
                total_value = float(values.sum())

                def format_number(value):
                    if abs(value - round(value)) < 1e-6:
                        return str(int(round(value)))
                    return f"{value:.2f}".rstrip('0').rstrip('.')

                autopct = None
                if show_percentage or show_value:
                    def autopct_func(pct):
                        if pct <= 0:
                            return ''
                        parts = []
                        if show_percentage:
                            parts.append(f"{pct:.1f}%")
                        if show_value:
                            absolute = pct * total_value / 100.0
                            parts.append(format_number(absolute))
                        return '\n'.join(parts)

                    autopct = autopct_func

                def resolve_pie_colors(count):
                    if count <= 0:
                        return []
                    palette_name = palette if palette else None
                    try:
                        palette_colors = sns.color_palette(palette_name if palette_name else 'deep', n_colors=count)
                        return [self._color_to_hex(col) for col in palette_colors]
                    except Exception:
                        return [self.color_options[idx % len(self.color_options)] for idx in range(count)]

                wedge_width = 1.0 if donut_ratio == 0 else max(0.05, 1.0 - donut_ratio)
                pie_colors = resolve_pie_colors(len(values))
                
                # ax.pie returns different number of values depending on autopct
                if autopct:
                    wedges, texts, autotexts = ax.pie(
                        values,
                        labels=labels,
                        autopct=autopct,
                        startangle=90,
                        counterclock=False,
                        colors=pie_colors,
                        wedgeprops={'width': wedge_width, 'edgecolor': 'white'}
                    )
                else:
                    wedges, texts = ax.pie(
                        values,
                        labels=labels,
                        startangle=90,
                        counterclock=False,
                        colors=pie_colors,
                        wedgeprops={'width': wedge_width, 'edgecolor': 'white'}
                    )
                    autotexts = []
                
                ax.axis('equal')
                if not grid:
                    ax.grid(False)

                if show_legend:
                    ax.legend(wedges, labels, loc=legend_loc, fontsize=legend_size)
                else:
                    try:
                        legend = ax.get_legend()
                        if legend:
                            legend.remove()
                    except Exception:
                        pass

                params_used_log.append(f"  Etiquetas: {label_col}")
                if value_col:
                    params_used_log.append(f"  Valores: {value_col}")
                params_used_log.append(f"  Hueco: {donut_ratio:.2f}")
                params_used_log.append(f"  Orden: {sort_mode_raw}")
                params_used_log.append(f"  Máx. categorías: {'Todas' if max_categories == 0 else max_categories}")
                params_used_log.append(f"  Excluir blancos: {'Sí' if exclude_blank else 'No'}")
                params_used_log.append(f"  Mostrar %: {'Sí' if show_percentage else 'No'}")
                params_used_log.append(f"  Mostrar valor: {'Sí' if show_value else 'No'}")

            elif chart_type == "Gráfico de Barras":
                mode_raw = getattr(self, 'param_bar_mode_var', StringVar(value='Simple')).get()
                mode = (mode_raw or 'Simple').strip().lower()
                if mode not in ('simple', 'agrupado', 'apilado', 'segmentado'):
                    mode = 'simple'
                params_used_log.append(f"  Modo barras: {mode_raw}")

                x_col = self.param_bar_x_var.get().strip() if self.param_bar_x_var.get() else ""
                y_raw = self.param_bar_y_var.get().strip() if self.param_bar_y_var.get() else ""
                hue_raw = self.param_bar_hue_var.get().strip() if self.param_bar_hue_var.get() else ""
                orientation = self.param_bar_orientation_var.get()

                if not x_col:
                    messagebox.showerror("Error", "Debe seleccionar la variable de eje.", parent=self.parent_for_dialogs)
                    plt.close(plt_fig)
                    return

                y_col = y_raw if y_raw else None
                hue_col = hue_raw if hue_raw else None

                try:
                    bar_width_value = float(getattr(self, 'param_bar_width_var', StringVar(value='0.8')).get())
                except Exception:
                    bar_width_value = 0.8
                bar_width_value = max(0.1, min(0.9, bar_width_value))

                params_used_log.append(f"  Categorías: {x_col}")
                if y_col and mode != 'segmentado':
                    params_used_log.append(f"  Valores: {y_col}")
                if hue_col and mode != 'segmentado':
                    params_used_log.append(f"  Hue: {hue_col}")
                params_used_log.append(f"  Orientación: {orientation}")

                if orientation == "Horizontal":
                    mark_axis('y', x_col)
                else:
                    mark_axis('x', x_col)

                plot_data = filtered_data.copy()
                if y_col:
                    plot_data = plot_data.copy()
                    plot_data[y_col] = pd.to_numeric(plot_data[y_col], errors='coerce')

                category_order = None
                hue_order = None

                try:
                    series_x = filtered_data[x_col]
                    if pd.api.types.is_categorical_dtype(series_x):
                        category_order = [str(cat) for cat in series_x.cat.categories]
                    else:
                        category_order = list(dict.fromkeys(series_x.dropna().tolist()))
                except Exception:
                    category_order = None

                if category_order is not None:
                    category_order = [str(val) for val in category_order]
                    try:
                        plot_data[x_col] = pd.Categorical(plot_data[x_col], categories=category_order, ordered=True)
                    except Exception:
                        plot_data[x_col] = plot_data[x_col].astype(str)
                else:
                    plot_data[x_col] = plot_data[x_col].astype(str)
                    category_order = list(dict.fromkeys(plot_data[x_col].tolist()))

                custom_order = self._recode_orders.get(x_col)
                if custom_order:
                    category_order = [str(val) for val in custom_order]
                    try:
                        plot_data[x_col] = pd.Categorical(plot_data[x_col], categories=category_order, ordered=True)
                    except Exception:
                        pass

                hue_levels = None
                if hue_col and mode != 'segmentado':
                    try:
                        series_h = filtered_data[hue_col]
                        if pd.api.types.is_categorical_dtype(series_h):
                            hue_order = [str(cat) for cat in series_h.cat.categories]
                        else:
                            hue_order = list(dict.fromkeys(series_h.dropna().tolist()))
                    except Exception:
                        hue_order = None

                    if hue_order is not None:
                        hue_order = [str(val) for val in hue_order]
                        try:
                            plot_data[hue_col] = pd.Categorical(plot_data[hue_col], categories=hue_order, ordered=True)
                        except Exception:
                            plot_data[hue_col] = plot_data[hue_col].astype(str)
                    else:
                        plot_data[hue_col] = plot_data[hue_col].astype(str)
                        hue_order = list(dict.fromkeys(plot_data[hue_col].tolist()))

                    custom_hue_order = self._recode_orders.get(hue_col)
                    if custom_hue_order:
                        hue_order = [str(val) for val in custom_hue_order]
                        try:
                            plot_data[hue_col] = pd.Categorical(plot_data[hue_col], categories=hue_order, ordered=True)
                        except Exception:
                            pass

                    hue_levels = hue_order

                bar_color_mode = getattr(self, 'param_bar_color_mode_var', StringVar(value='auto')).get() if hasattr(self, 'param_bar_color_mode_var') else 'auto'
                bar_color_mode = (bar_color_mode or 'auto').strip().lower()
                base_color_raw = getattr(self, 'param_bar_single_color_var', StringVar(value='#4C72B0')).get() if hasattr(self, 'param_bar_single_color_var') else '#4C72B0'
                base_color_raw = base_color_raw.strip() if base_color_raw else ''
                palette_choice_raw = getattr(self, 'param_bar_palette_choice_var', StringVar(value='deep')).get() if hasattr(self, 'param_bar_palette_choice_var') else 'deep'
                palette_choice_key = (palette_choice_raw or '').strip()
                user_palettes_map = {f"usuario: {name}": colors for name, colors in self._load_user_palettes().items()}
                user_palette_selected = user_palettes_map.get(palette_choice_key)
                palette_choice = palette_choice_key.lower()
                if palette_choice == 'default':
                    palette_choice = ''

                # Colores personalizados por coma (aplica cuando no hay hue y modo no segmentado)
                custom_colors_raw = getattr(self, 'param_bar_custom_colors_var', StringVar(value='')).get()
                custom_colors = self._parse_custom_colors(custom_colors_raw)

                base_color = base_color_raw if base_color_raw else None

                if bar_color_mode and bar_color_mode != 'auto':
                    params_used_log.append(f"  Modo color barras: {bar_color_mode}")
                if bar_color_mode == 'un color' and base_color:
                    params_used_log.append(f"    Color único: {base_color}")
                if bar_color_mode == 'paleta' and palette_choice_key:
                    params_used_log.append(f"    Paleta: {palette_choice_key}")

                try:
                    if mode in ('simple', 'agrupado'):
                        if mode == 'agrupado' and not hue_col:
                            self.log("Se seleccionó modo 'Agrupado' sin variable de color; se graficará como barras simples.", "WARN")

                        palette_spec = None
                        color_spec = None
                        if bar_color_mode == 'un color' and base_color:
                            single_color = self._color_to_hex(base_color)
                            if hue_col and hue_levels:
                                palette_spec = {str(level): single_color for level in hue_levels}
                            else:
                                color_spec = single_color
                        elif bar_color_mode == 'paleta':
                            if user_palette_selected:
                                palette_spec = user_palette_selected
                            elif hue_col:
                                palette_spec = palette_choice if palette_choice else None
                            else:
                                palette_spec = palette_choice if palette_choice else None
                        if isinstance(palette_spec, dict):
                            normalized_palette = {}
                            for key, value in palette_spec.items():
                                color_hex = self._color_to_hex(value)
                                normalized_palette[str(key)] = color_hex
                            palette_spec = normalized_palette

                        if y_col:
                            plot_kwargs = {
                                'data': plot_data,
                                'hue': hue_col if hue_col else None,
                                'ax': ax,
                                'width': bar_width_value,
                                'ci': None
                            }
                            if hue_order:
                                plot_kwargs['hue_order'] = hue_order
                            if custom_colors and not hue_col:
                                plot_kwargs['palette'] = custom_colors
                            elif palette_spec is not None:
                                plot_kwargs['palette'] = palette_spec
                            elif color_spec is not None:
                                plot_kwargs['color'] = color_spec

                            if orientation == "Horizontal":
                                plot_kwargs.update({'x': y_col, 'y': x_col})
                                if category_order:
                                    plot_kwargs['order'] = category_order
                            else:
                                plot_kwargs.update({'x': x_col, 'y': y_col})
                                if category_order:
                                    plot_kwargs['order'] = category_order

                            self._call_seaborn_plot(sns.barplot, plot_kwargs)

                            if ci_settings.get('enabled'):
                                try:
                                    self._apply_bar_confidence_intervals(
                                        ax=ax,
                                        data=plot_data,
                                        x_col=x_col if orientation != "Horizontal" else x_col,
                                        y_col=y_col,
                                        hue_col=hue_col,
                                        category_order=category_order,
                                        hue_order=hue_order,
                                        orientation=orientation,
                                        settings=ci_settings,
                                        patches=ax.patches
                                    )
                                except Exception as exc:
                                    self.log(f"No se pudieron calcular/plotear IC en barras: {exc}", "WARN")
                        else:
                            plot_kwargs = {
                                'data': plot_data,
                                'hue': hue_col if hue_col else None,
                                'ax': ax,
                                'width': bar_width_value,
                                'ci': None
                            }
                            if hue_order:
                                plot_kwargs['hue_order'] = hue_order
                            if custom_colors and not hue_col:
                                plot_kwargs['palette'] = custom_colors
                            elif palette_spec is not None:
                                plot_kwargs['palette'] = palette_spec
                            elif color_spec is not None:
                                plot_kwargs['color'] = color_spec

                            if orientation == "Horizontal":
                                plot_kwargs.update({'y': x_col})
                                if category_order:
                                    plot_kwargs['order'] = category_order
                            else:
                                plot_kwargs.update({'x': x_col})
                                if category_order:
                                    plot_kwargs['order'] = category_order

                            self._call_seaborn_plot(sns.countplot, plot_kwargs)

                        if orientation == "Horizontal":
                            mark_axis('y', x_col, plot_data, force=True)
                        else:
                            mark_axis('x', x_col, plot_data, force=True)

                        if ci_settings.get('enabled') and not y_col:
                            self.log("IC bootstrap solo disponible cuando se proporciona una variable de valor.", "WARN")

                        # Etiquetas de conteo / total
                        try:
                            self._annotate_bar_counts(
                                ax=ax,
                                data=plot_data,
                                x_col=x_col,
                                y_col=y_col,
                                hue_col=hue_col,
                                category_order=category_order,
                                hue_order=hue_order,
                                orientation=orientation,
                                show_counts=self.param_bar_show_counts_var.get() if hasattr(self, 'param_bar_show_counts_var') else False,
                                show_total=self.param_bar_show_total_var.get() if hasattr(self, 'param_bar_show_total_var') else False,
                                label_size=self.param_bar_label_size_var.get() if hasattr(self, 'param_bar_label_size_var') else "9",
                                label_color=self.param_bar_label_color_var.get() if hasattr(self, 'param_bar_label_color_var') else "#000000",
                                label_position=self.param_bar_label_position_var.get() if hasattr(self, 'param_bar_label_position_var') else "arriba de la barra",
                                total_position=self.param_bar_total_position_var.get() if hasattr(self, 'param_bar_total_position_var') else "sup derecha",
                                fontfamily=font_family_sel if font_family_sel and font_family_sel != 'Default' else None
                            )
                        except Exception as exc:
                            self.log(f"No se pudieron dibujar etiquetas de conteo: {exc}", "WARN")

                        # Marcas de significancia
                        try:
                            show_sig = getattr(self, 'param_bar_show_sig_var', tk.BooleanVar(value=False)).get()
                            if show_sig and hasattr(self, 'param_bar_sig_comparisons_text'):
                                sig_text = self.param_bar_sig_comparisons_text.get("1.0", tk.END).strip()
                                if sig_text:
                                    sig_display = getattr(self, 'param_bar_sig_display_var', StringVar(value='estrellas')).get()
                                    sig_comparisons = self._parse_significance_comparisons(sig_text, has_hue=bool(hue_col))
                                    
                                    # Agregar show_as a cada comparación
                                    for comp in sig_comparisons:
                                        comp['show_as'] = 'stars' if sig_display == 'estrellas' else 'p_value'
                                    
                                    sig_settings = {
                                        'color': getattr(self, 'param_bar_sig_color_var', StringVar(value='black')).get(),
                                        'fontsize': getattr(self, 'param_bar_sig_fontsize_var', StringVar(value='10')).get(),
                                        'show_ns': getattr(self, 'param_bar_sig_show_ns_var', tk.BooleanVar(value=False)).get(),
                                        'height': float(getattr(self, 'param_bar_sig_height_var', StringVar(value='3')).get()) * 0.01,
                                        'linewidth': float(getattr(self, 'param_bar_sig_linewidth_var', StringVar(value='1.0')).get()),
                                        'vertical_offset': float(getattr(self, 'param_bar_sig_voffset_var', StringVar(value='0')).get()),
                                        'spacing': float(getattr(self, 'param_bar_sig_spacing_var', StringVar(value='1.8')).get()),
                                        'fontfamily': font_family_sel if font_family_sel and font_family_sel != 'Default' else None
                                    }
                                    
                                    valid_patches = [p for p in ax.patches if isinstance(p, Rectangle) and (p.get_width() != 0 or p.get_height() != 0)]
                                    
                                    self._draw_significance_annotations(
                                        ax=ax,
                                        comparisons=sig_comparisons,
                                        patches=valid_patches,
                                        category_order=category_order,
                                        hue_order=hue_order,
                                        orientation=orientation,
                                        settings=sig_settings
                                    )
                                    self.log(f"Dibujadas {len(sig_comparisons)} marcas de significancia", "INFO")
                        except Exception as exc:
                            self.log(f"No se pudieron dibujar marcas de significancia: {exc}", "WARN")

                    elif mode == 'apilado':
                        if ci_settings.get('enabled'):
                            self.log("IC bootstrap no se aplica en modo Apilado actualmente.", "WARN")
                        if not hue_col:
                            plt.close(plt_fig)
                            messagebox.showerror("Error", "Seleccione una variable en 'Agrupar por Color' para apilar las barras.", parent=self.parent_for_dialogs)
                            return

                        normalize_100 = getattr(self, 'param_stacked_100_var', tk.BooleanVar(value=False)).get()
                        params_used_log.append(f"  Normalizar a 100%: {'Sí' if normalize_100 else 'No'}")

                        value_field = y_col if y_col else '_count'
                        if y_col:
                            value_data = filtered_data[[x_col, hue_col, y_col]].dropna(subset=[x_col, hue_col])
                            value_data[y_col] = pd.to_numeric(value_data[y_col], errors='coerce').fillna(0.0)
                            grouped = value_data.groupby([x_col, hue_col], dropna=False)[y_col].sum().reset_index(name=value_field)
                        else:
                            value_data = filtered_data[[x_col, hue_col]].dropna(subset=[x_col, hue_col])
                            grouped = value_data.groupby([x_col, hue_col], dropna=False).size().reset_index(name=value_field)

                        grouped[x_col] = grouped[x_col].astype(str)
                        grouped[hue_col] = grouped[hue_col].astype(str)

                        if category_order:
                            grouped['_cat_order'] = grouped[x_col].apply(lambda v: category_order.index(v) if v in category_order else len(category_order))
                        else:
                            grouped['_cat_order'] = grouped[x_col].rank(method='dense')
                        if hue_levels:
                            grouped['_hue_order'] = grouped[hue_col].apply(lambda v: hue_levels.index(v) if v in hue_levels else len(hue_levels))
                        else:
                            grouped['_hue_order'] = grouped[hue_col].rank(method='dense')

                        grouped = grouped.sort_values(['_cat_order', '_hue_order']).drop(columns=['_cat_order', '_hue_order'])

                        pivot = grouped.pivot(index=x_col, columns=hue_col, values=value_field).fillna(0.0)
                        if category_order:
                            pivot = pivot.reindex(category_order, axis=0)
                        if hue_levels:
                            pivot = pivot.reindex(hue_levels, axis=1)

                        pivot = pivot.fillna(0.0)
                        categories = list(pivot.index)
                        stack_levels = list(pivot.columns)
                        if not stack_levels:
                            plt.close(plt_fig)
                            messagebox.showwarning("Datos Vacíos", "No hay categorías disponibles para la variable de color seleccionada.", parent=self.parent_for_dialogs)
                            return
                        values_matrix = pivot.to_numpy(dtype=float)

                        if normalize_100:
                            totals = values_matrix.sum(axis=1)
                            totals[totals == 0] = np.nan
                            values_matrix = np.divide(values_matrix, totals[:, None]) * 100.0
                            values_matrix = np.nan_to_num(values_matrix, nan=0.0)

                        def resolve_stack_colors(levels):
                            if not levels:
                                return []
                            colors_local = []
                            if bar_color_mode == 'un color' and base_color:
                                single_hex = self._color_to_hex(base_color)
                                colors_local = [single_hex] * len(levels)
                            elif bar_color_mode == 'paleta':
                                if user_palette_selected:
                                    colors_local = [self._color_to_hex(col) for col in user_palette_selected][:len(levels)]
                                elif palette_choice:
                                    try:
                                        palette_generated = sns.color_palette(palette_choice, n_colors=len(levels))
                                        colors_local = [self._color_to_hex(col) for col in palette_generated]
                                    except Exception:
                                        colors_local = []
                            if not colors_local:
                                try:
                                    base_palette = palette_choice if palette_choice else 'deep'
                                    palette_generated = sns.color_palette(base_palette, n_colors=len(levels))
                                    colors_local = [self._color_to_hex(col) for col in palette_generated]
                                except Exception:
                                    colors_local = [self.color_options[idx % len(self.color_options)] for idx in range(len(levels))]
                            if len(colors_local) < len(levels):
                                fallback_colors = [self.color_options[idx % len(self.color_options)] for idx in range(len(levels))]
                                while len(colors_local) < len(levels):
                                    colors_local.append(fallback_colors[len(colors_local) % len(fallback_colors)])
                            return colors_local

                        stack_colors = resolve_stack_colors(stack_levels)

                        if orientation == "Horizontal":
                            indices = np.arange(len(categories))
                            cumulative = np.zeros(len(categories))
                            for idx_level, (level, color_hex) in enumerate(zip(stack_levels, stack_colors)):
                                segment_values = values_matrix[:, idx_level]
                                ax.barh(indices, segment_values, left=cumulative, color=color_hex,
                                         label=str(level), height=bar_width_value)
                                cumulative += segment_values
                            ax.set_yticks(indices)
                            ax.set_yticklabels(categories)
                            category_df = pd.DataFrame({x_col: categories})
                            mark_axis('y', x_col, category_df, force=True)
                        else:
                            indices = np.arange(len(categories))
                            cumulative = np.zeros(len(categories))
                            for idx_level, (level, color_hex) in enumerate(zip(stack_levels, stack_colors)):
                                segment_values = values_matrix[:, idx_level]
                                ax.bar(indices, segment_values, bottom=cumulative, color=color_hex,
                                       label=str(level), width=bar_width_value)
                                cumulative += segment_values
                            ax.set_xticks(indices)
                            ax.set_xticklabels(categories)
                            category_df = pd.DataFrame({x_col: categories})
                            mark_axis('x', x_col, category_df, force=True)

                        if normalize_100 and not ylabel:
                            ax.set_ylabel('Porcentaje (%)')

                        if show_legend:
                            ax.legend(loc=legend_loc)
                        else:
                            legend = ax.get_legend()
                            if legend is not None:
                                legend.remove()

                    elif mode == 'segmentado':
                        if ci_settings.get('enabled'):
                            self.log("IC bootstrap no se aplica en modo Segmentado actualmente.", "WARN")
                        normalize_100 = getattr(self, 'param_bar_segment_normalize_var', tk.BooleanVar(value=False)).get()
                        params_used_log.append(f"  Normalizar segmentos a 100%: {'Sí' if normalize_100 else 'No'}")

                        segment_pairs = getattr(self, 'param_bar_segment_vars', []) or []
                        segment_cols = []
                        segment_labels = []
                        for var_pair in segment_pairs:
                            if not isinstance(var_pair, tuple) or len(var_pair) < 2:
                                continue
                            var_obj, label_obj = var_pair
                            col_name = var_obj.get().strip() if var_obj and var_obj.get() else ''
                            if not col_name:
                                continue
                            if col_name not in filtered_data.columns:
                                self.log(f"Columna de segmento '{col_name}' no encontrada en los datos filtrados.", "WARN")
                                continue
                            if not pd.api.types.is_numeric_dtype(filtered_data[col_name]):
                                self.log(f"Columna de segmento '{col_name}' no es numérica; se omite.", "WARN")
                                continue
                            display_label = label_obj.get().strip() if label_obj and label_obj.get() else col_name
                            segment_cols.append(col_name)
                            segment_labels.append(display_label)

                        if len(segment_cols) < 2:
                            plt.close(plt_fig)
                            messagebox.showerror("Error", "Seleccione al menos dos columnas numéricas para formar los segmentos de cada barra.", parent=self.parent_for_dialogs)
                            return

                        params_used_log.append(f"  Segmentos: {', '.join(segment_labels)}")

                        data_columns = [x_col] + segment_cols
                        plot_source = filtered_data[data_columns].copy()
                        plot_source = plot_source.dropna(subset=[x_col])
                        plot_source[segment_cols] = plot_source[segment_cols].apply(pd.to_numeric, errors='coerce').fillna(0.0)
                        if plot_source.empty:
                            plt.close(plt_fig)
                            messagebox.showwarning("Datos Vacíos", "No hay datos disponibles para las columnas seleccionadas.", parent=self.parent_for_dialogs)
                            return

                        try:
                            grouped = plot_source.groupby(x_col, dropna=False)[segment_cols].sum().reset_index()
                        except Exception as exc:
                            plt.close(plt_fig)
                            self.log(f"Error al agrupar datos para barras segmentadas: {exc}", "ERROR")
                            messagebox.showerror("Error", f"No se pudo preparar la información para el gráfico:\n{exc}", parent=self.parent_for_dialogs)
                            return

                        if grouped.empty:
                            plt.close(plt_fig)
                            messagebox.showwarning("Datos Vacíos", "La agrupación resultó vacía para las columnas seleccionadas.", parent=self.parent_for_dialogs)
                            return

                        grouped[x_col] = grouped[x_col].astype(str)
                        if category_order:
                            grouped['_order_index'] = grouped[x_col].apply(lambda val: category_order.index(val) if val in category_order else len(category_order))
                            grouped = grouped.sort_values('_order_index').drop(columns=['_order_index'])

                        categories = grouped[x_col].tolist()
                        values_matrix = grouped[segment_cols].to_numpy(dtype=float)

                        if normalize_100:
                            totals = values_matrix.sum(axis=1)
                            totals[totals == 0] = np.nan
                            values_matrix = np.divide(values_matrix, totals[:, None]) * 100.0
                            values_matrix = np.nan_to_num(values_matrix, nan=0.0)

                        palette_choice_segments_raw = getattr(self, 'param_bar_segment_palette_var', StringVar(value='deep')).get()
                        palette_choice_segments = (palette_choice_segments_raw or '').strip()
                        user_palette_segments_map = {f"usuario: {name}": colors for name, colors in self._load_user_palettes().items()}
                        user_palette_segments = user_palette_segments_map.get(palette_choice_segments)
                        custom_colors_raw = getattr(self, 'param_bar_segment_custom_colors_var', StringVar(value='')).get()
                        custom_tokens = [tok.strip() for tok in custom_colors_raw.split(',') if tok.strip()]
                        colors = [self._color_to_hex(token) for token in custom_tokens]

                        if len(colors) < len(segment_cols):
                            if user_palette_segments:
                                palette_generated = [self._color_to_hex(col) for col in user_palette_segments]
                            else:
                                palette_name = palette_choice_segments if palette_choice_segments else 'deep'
                                try:
                                    palette_generated = sns.color_palette(palette_name, n_colors=len(segment_cols))
                                    palette_generated = [self._color_to_hex(col) for col in palette_generated]
                                except Exception:
                                    palette_generated = [self.color_options[idx % len(self.color_options)] for idx in range(len(segment_cols))]
                            while len(colors) < len(segment_cols):
                                colors.append(palette_generated[len(colors) % len(palette_generated)])

                        params_used_log.append(f"  Paleta segmentos: {palette_choice_segments_raw if palette_choice_segments_raw else 'auto'}")
                        if custom_tokens:
                            params_used_log.append(f"  Colores personalizados: {', '.join(custom_tokens)}")

                        indices = np.arange(len(categories))
                        cumulative = np.zeros(len(categories))

                        if orientation == "Horizontal":
                            for seg_idx, (label, color_hex) in enumerate(zip(segment_labels, colors)):
                                segment_values = values_matrix[:, seg_idx]
                                ax.barh(indices, segment_values, left=cumulative, color=color_hex,
                                         label=label, height=bar_width_value)
                                cumulative += segment_values
                            ax.set_yticks(indices)
                            ax.set_yticklabels(categories)
                            category_df = pd.DataFrame({x_col: categories})
                            mark_axis('y', x_col, category_df, force=True)
                        else:
                            for seg_idx, (label, color_hex) in enumerate(zip(segment_labels, colors)):
                                segment_values = values_matrix[:, seg_idx]
                                ax.bar(indices, segment_values, bottom=cumulative, color=color_hex,
                                       label=label, width=bar_width_value)
                                cumulative += segment_values
                            ax.set_xticks(indices)
                            ax.set_xticklabels(categories)
                            category_df = pd.DataFrame({x_col: categories})
                            mark_axis('x', x_col, category_df, force=True)

                        if normalize_100 and not ylabel:
                            ax.set_ylabel('Porcentaje (%)')

                        if show_legend:
                            ax.legend(loc=legend_loc)
                        else:
                            legend = ax.get_legend()
                            if legend is not None:
                                legend.remove()

                except Exception as exc:
                    plt.close(plt_fig)
                    self.log(f"Error generando gráfico de barras: {exc}", "ERROR")
                    messagebox.showerror("Error", f"No se pudo generar el gráfico de barras:\n{exc}", parent=self.parent_for_dialogs)
                    return

            elif chart_type == "Gráfico de Líneas / Área":
                x_col = self.param_line_x_var.get()
                y_col = self.param_line_y_var.get()

                if not x_col or not y_col:
                    messagebox.showerror("Error", "Debe seleccionar las variables X e Y.", parent=self.parent_for_dialogs)
                    plt.close(plt_fig)
                    return

                params_used_log.extend([f"  X: {x_col}", f"  Y: {y_col}"])

                hue_col = self.param_line_hue_var.get()
                fill_area = self.param_area_fill_var.get()
                line_style_map = {'Sólida': '-', 'Punteada': ':', 'Rayada': '--', 'Punto-Raya': '-.'}
                line_style_raw = getattr(self, 'param_line_style_var', StringVar(value='Sólida')).get()
                line_style = line_style_map.get(line_style_raw, '-')
                try:
                    line_width = float(getattr(self, 'param_line_width_var', StringVar(value='2.0')).get())
                except Exception:
                    line_width = 2.0
                show_markers = getattr(self, 'param_line_markers_var', tk.BooleanVar(value=False)).get()
                show_ci_line = getattr(self, 'param_line_ci_var', tk.BooleanVar(value=False)).get()
                estimator_raw = getattr(self, 'param_line_estimator_var', StringVar(value='mean')).get()

                params_used_log.append(f"  Estilo: {line_style_raw}")
                if line_width != 2.0:
                    params_used_log.append(f"  Grosor: {line_width}")

                estimator_func = None
                errorbar_arg = None
                if estimator_raw == "Ninguno":
                    estimator_func = None
                elif estimator_raw == "median":
                    estimator_func = "median"
                elif estimator_raw == "sum":
                    estimator_func = "sum"
                elif estimator_raw == "min":
                    estimator_func = "min"
                elif estimator_raw == "max":
                    estimator_func = "max"
                else:
                    estimator_func = "mean"

                if show_ci_line and estimator_func:
                    errorbar_arg = ('ci', 95)
                    params_used_log.append("  Banda de confianza: 95%")
                elif not show_ci_line:
                    errorbar_arg = None

                plot_kwargs = {
                    'data': filtered_data,
                    'x': x_col,
                    'y': y_col,
                    'ax': ax,
                    'linestyle': line_style,
                    'linewidth': line_width,
                }
                if estimator_func:
                    plot_kwargs['estimator'] = estimator_func
                    params_used_log.append(f"  Agregación: {estimator_raw}")
                else:
                    plot_kwargs['estimator'] = None
                    plot_kwargs['units'] = None

                if errorbar_arg:
                    plot_kwargs['errorbar'] = errorbar_arg
                else:
                    plot_kwargs['errorbar'] = None

                if show_markers:
                    plot_kwargs['marker'] = 'o'
                    plot_kwargs['markersize'] = point_size
                    params_used_log.append("  Marcadores: Sí")

                if hue_col:
                    plot_kwargs['hue'] = hue_col
                    params_used_log.append(f"  Hue: {hue_col}")

                try:
                    sns.lineplot(**plot_kwargs)
                except TypeError:
                    # Fallback for older seaborn versions without errorbar param
                    plot_kwargs.pop('errorbar', None)
                    sns.lineplot(**plot_kwargs)

                if fill_area:
                    params_used_log.append("  Rellenar Área: Sí")
                    lines = ax.get_lines()
                    for line in lines:
                        xd = line.get_xdata()
                        yd = line.get_ydata()
                        if len(xd) > 1:
                            ax.fill_between(xd, yd, alpha=0.2, color=line.get_color())

                mark_axis('x', x_col)
                mark_axis('y', y_col)
            
            elif chart_type == "Gráfico de Densidad":
                x_col = self.param_density_var.get()
                if not x_col:
                    messagebox.showerror("Error", "Debe seleccionar una variable numérica.", parent=self.parent_for_dialogs)
                    plt.close(plt_fig)
                    return

                params_used_log.append(f"  Variable: {x_col}")

                hue_col = self.param_density_hue_var.get()
                density_fill = getattr(self, 'param_density_fill_var', tk.BooleanVar(value=True)).get()
                density_rug = getattr(self, 'param_density_rug_var', tk.BooleanVar(value=False)).get()
                density_cumulative = getattr(self, 'param_density_cumulative_var', tk.BooleanVar(value=False)).get()
                density_show_stats = getattr(self, 'param_density_show_stats_var', tk.BooleanVar(value=False)).get()
                try:
                    density_bw = float(getattr(self, 'param_density_bw_var', StringVar(value='1.0')).get())
                except Exception:
                    density_bw = 1.0
                density_multiple = getattr(self, 'param_density_multiple_var', StringVar(value='layer')).get()

                plot_kwargs = {
                    'data': filtered_data,
                    'x': x_col,
                    'ax': ax,
                    'fill': density_fill,
                    'bw_adjust': density_bw,
                    'cumulative': density_cumulative,
                }
                params_used_log.append(f"  Relleno: {'Sí' if density_fill else 'No'}")
                if density_cumulative:
                    params_used_log.append("  Acumulada: Sí")
                if density_bw != 1.0:
                    params_used_log.append(f"  bw_adjust: {density_bw}")

                if hue_col:
                    plot_kwargs['hue'] = hue_col
                    plot_kwargs['multiple'] = density_multiple
                    params_used_log.extend([f"  Hue: {hue_col}", f"  Modo múltiple: {density_multiple}"])
                
                sns.kdeplot(**plot_kwargs)

                # Rug plot
                if density_rug:
                    rug_kw = {'data': filtered_data, 'x': x_col, 'ax': ax, 'height': 0.05}
                    if hue_col:
                        rug_kw['hue'] = hue_col
                    try:
                        sns.rugplot(**rug_kw)
                    except Exception:
                        pass

                # Stats overlay
                if density_show_stats:
                    try:
                        num_vals = pd.to_numeric(filtered_data[x_col], errors='coerce').dropna()
                        if len(num_vals) > 0:
                            mean_val = num_vals.mean()
                            median_val = num_vals.median()
                            std_val = num_vals.std()
                            params_used_log.extend([f"  Media: {mean_val:.3f}", f"  Mediana: {median_val:.3f}", f"  DE: {std_val:.3f}"])
                            ax.axvline(mean_val, color='red', linestyle='--', linewidth=1.2, alpha=0.8, label=f'Media ({mean_val:.2f})')
                            ax.axvline(median_val, color='green', linestyle=':', linewidth=1.2, alpha=0.8, label=f'Mediana ({median_val:.2f})')
                            ax.annotate(f"n={len(num_vals)}\nMedia={mean_val:.3f}\nMediana={median_val:.3f}\nDE={std_val:.3f}",
                                        xy=(0.98, 0.98), xycoords='axes fraction',
                                        fontsize=8, va='top', ha='right',
                                        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8, edgecolor='gray'))
                            ax.legend(loc='best', fontsize=8)
                    except Exception:
                        pass

            # This is a placeholder for the new logic. It will be fully implemented in the next step.
            elif chart_type == "Gráfico de Distribución":
                # Parameters
                y_var = getattr(self, 'param_dist_y_var', None)
                x_var = getattr(self, 'param_dist_x_var', None)
                hue_var = getattr(self, 'param_dist_hue_var', None)
                point_type_var = getattr(self, 'param_dist_point_type_var', None)
                size_var = getattr(self, 'param_dist_size_var', None)
                style_var = getattr(self, 'param_dist_style_var', None)
                show_box = getattr(self, 'param_dist_show_box_var', tk.BooleanVar(value=False)).get()
                show_violin = getattr(self, 'param_dist_show_violin_var', tk.BooleanVar(value=False)).get()
                raincloud = getattr(self, 'param_dist_raincloud_var', tk.BooleanVar(value=False)).get()

                y_col = y_var.get() if y_var is not None else None
                x_col = x_var.get() if x_var is not None else None
                hue_col = hue_var.get() if hue_var is not None else None
                point_type = point_type_var.get() if point_type_var is not None else 'Ninguno'
                size_col = size_var.get().strip() if size_var is not None and size_var.get() else ''
                style_col = style_var.get().strip() if style_var is not None and style_var.get() else ''
                size_col = size_col if size_col else None
                style_col = style_col if style_col else None
                layout_method_raw = getattr(self, 'param_dist_layout_method_var', StringVar(value='Auto')).get()
                layout_method_raw = layout_method_raw.strip() if layout_method_raw else 'Auto'
                jitter_width_raw = getattr(self, 'param_dist_jitter_width_var', StringVar(value='0.20')).get()
                try:
                    jitter_width = float(jitter_width_raw)
                except Exception:
                    jitter_width = 0.2
                if jitter_width < 0:
                    jitter_width = 0.0
                jitter_effective = jitter_width

                if raincloud and not show_violin:
                    show_violin = True
                    params_used_log.append("  Raincloud requiere violín: activado automáticamente")

                params_used_log.append(f"  Y: {y_col}")
                if x_col: params_used_log.append(f"  X (categoría): {x_col}")
                if hue_col: params_used_log.append(f"  Hue: {hue_col}")
                params_used_log.append(f"  Mostrar caja: {show_box}")
                params_used_log.append(f"  Mostrar violín: {show_violin}")
                params_used_log.append(f"  Raincloud: {raincloud}")
                params_used_log.append(f"  Tipo puntos: {point_type}")
                if size_col:
                    params_used_log.append(f"  Variable tamaño: {size_col}")
                if style_col:
                    params_used_log.append(f"  Variable forma: {style_col}")
                params_used_log.append(f"  Método acomodo: {layout_method_raw}")
                params_used_log.append(f"  Jitter aleatorio: {jitter_width}")

                if not y_col or y_col not in filtered_data.columns:
                    messagebox.showerror("Error", "Seleccione una variable numérica para el eje Y.", parent=self.parent_for_dialogs)
                    return

                if x_col:
                    mark_axis('x', x_col, filtered_data)

                # Clean data subset
                columns_needed = [y_col]
                if x_col:
                    columns_needed.append(x_col)
                if hue_col:
                    columns_needed.append(hue_col)
                if size_col:
                    columns_needed.append(size_col)
                if style_col:
                    columns_needed.append(style_col)

                drop_subset = [y_col]
                if x_col:
                    drop_subset.append(x_col)
                if hue_col:
                    drop_subset.append(hue_col)

                plot_df = filtered_data[columns_needed].dropna(subset=drop_subset)

                # Determine order for X axis labels and ensure the plotting column is string-based
                x_order = None
                x_col_original = x_col  # Save original x_col before any modifications
                if x_col:
                    plot_df = plot_df.copy()
                    if self._is_categorical(plot_df[x_col]):
                        categories = list(plot_df[x_col].cat.categories)
                        x_order = [str(cat) for cat in categories]
                        plot_df[x_col] = plot_df[x_col].astype(str)
                        self.log(f"Variable categórica '{x_col}' detectada. Orden: {x_order}", "DEBUG")
                    else:
                        # Preserve the order of appearance to keep intuitive ordering
                        unique_vals = list(dict.fromkeys(plot_df[x_col].tolist()))
                        x_order = [str(v) for v in unique_vals]
                        plot_df[x_col] = plot_df[x_col].astype(str)
                        self.log(f"Variable '{x_col}' convertida a string. Orden: {x_order}", "DEBUG")

                    if x_order:
                        try:
                            plot_df[x_col] = pd.Categorical(plot_df[x_col], categories=x_order, ordered=True)
                        except Exception:
                            # Fallback: leave as string if categorical creation fails
                            self.log("No se pudo convertir la columna a Categorical para mantener el orden.", "DEBUG")

                # Handle collapse-to-single-tick option and group palette
                collapse = getattr(self, 'param_dist_collapse_var', tk.BooleanVar(value=False)).get()
                group_palette_name = getattr(self, 'param_dist_group_palette_var', StringVar(value='deep')).get()
                plot_palette_local = None if not group_palette_name or group_palette_name == 'default' else group_palette_name
                if collapse:
                    # create a temporary single x column so seaborn will use hue to separate groups at same tick
                    plot_df = plot_df.copy()
                    plot_df['__single_x__'] = pd.Categorical(['Todas'] * len(plot_df), categories=['Todas'], ordered=True)
                    x_col = '__single_x__'
                    x_order = ['Todas']

                if x_col and x_col in plot_df.columns:
                    force_override = (x_col_original is None) or (x_col != x_col_original)
                    mark_axis('x', x_col, plot_df, force=force_override)

                # Determine palette argument: prefer explicit mapping if user assigned colors per level
                palette_arg = None
                try:
                    mapping = getattr(self, 'param_dist_group_color_map', None)
                    if mapping and hue_col:
                        # mapping keys are strings; ensure keys exist for levels
                        palette_arg = mapping
                    else:
                        palette_arg = plot_palette_local if plot_palette_local else (palette if 'palette' in locals() else None)
                except Exception:
                    palette_arg = plot_palette_local if plot_palette_local else None

                if isinstance(palette_arg, dict):
                    normalized = {}
                    for key, value in palette_arg.items():
                        color_hex = self._color_to_hex(value)
                        normalized[key] = color_hex
                        normalized[str(key)] = color_hex
                    palette_arg = normalized

                category_color_map_attr = getattr(self, 'param_dist_category_color_map', {})
                category_list = []
                if x_col:
                    if x_order:
                        category_list = [str(cat) for cat in x_order]
                    else:
                        category_list = [str(cat) for cat in pd.unique(plot_df[x_col]) if pd.notna(cat)]
                category_palette = self._build_category_palette(category_list, group_palette_name, category_color_map_attr, point_color)

                size_mapper = self._build_size_mapper(plot_df[size_col], point_size) if size_col and size_col in plot_df.columns else None
                style_mapper = self._build_marker_mapper(plot_df[style_col]) if style_col and style_col in plot_df.columns else None

                custom_method_lookup = {'center': 'center', 'hex': 'hex', 'square': 'square'}
                layout_method_option = (layout_method_raw or 'Auto').strip().lower()
                layout_method_resolved = custom_method_lookup.get(layout_method_option)
                point_type_key = (point_type or '').lower()
                point_layout_method = None
                for key, value in custom_method_lookup.items():
                    if key in point_type_key:
                        point_layout_method = value
                        break
                if not point_layout_method:
                    point_layout_method = layout_method_resolved
                if point_layout_method not in custom_method_lookup.values():
                    point_layout_method = None
                fallback_layout_method = point_layout_method or layout_method_resolved
                if fallback_layout_method not in custom_method_lookup.values():
                    fallback_layout_method = None
                if fallback_layout_method is None and (size_col or style_col):
                    fallback_layout_method = 'hex'
                custom_layout_failure_logged = False

                # Centralize point overlay rendering so layout/jitter rules stay consistent
                def render_points(order, mode='auto', enforce_single_color=False, allow_dodge=True, legend_mode=None):
                    nonlocal custom_layout_failure_logged
                    order_to_use = order if order is not None else x_order
                    jitter_value = max(jitter_effective, 0.0)

                    if fallback_layout_method:
                        custom_palette = {} if enforce_single_color else (category_palette or {})
                        try:
                            self._plot_custom_point_layout(
                                ax=ax,
                                data=plot_df,
                                x_col=x_col,
                                y_col=y_col,
                                order=order_to_use,
                                hue_col=hue_col,
                                palette_arg=palette_arg,
                                category_palette=custom_palette,
                                default_color=custom_point_color,
                                point_size=point_size,
                                method=fallback_layout_method,
                                size_col=size_col,
                                size_mapper=size_mapper,
                                style_col=style_col,
                                style_mapper=style_mapper
                            )
                            return
                        except Exception:
                            if not custom_layout_failure_logged:
                                self.log("No se pudo aplicar el acomodo personalizado; se usa dispersión estándar.", "WARN")
                                custom_layout_failure_logged = True

                    mode_key = (mode or 'auto').lower()
                    if mode_key == 'auto':
                        if 'swarm' in point_type_key:
                            mode_key = 'swarm'
                        elif point_type_key in ('center', 'hex', 'square'):
                            mode_key = 'swarm'
                        else:
                            mode_key = 'strip'

                    palette_for_points = None
                    color_override = None
                    if hue_col and palette_arg is not None and not enforce_single_color:
                        palette_for_points = palette_arg
                    elif not hue_col and not enforce_single_color and category_palette:
                        palette_for_points = {str(k): v for k, v in category_palette.items()}
                    else:
                        color_override = custom_point_color

                    common_kwargs = {
                        'data': plot_df,
                        'x': x_col,
                        'y': y_col,
                        'order': order_to_use,
                        'ax': ax,
                        'size': point_size
                    }
                    if hue_col:
                        common_kwargs['hue'] = hue_col
                        if allow_dodge:
                            common_kwargs['dodge'] = True
                    if palette_for_points is not None:
                        common_kwargs['palette'] = palette_for_points
                    if color_override is not None:
                        common_kwargs['color'] = color_override
                    if legend_mode is not None:
                        common_kwargs['legend'] = legend_mode

                    try:
                        if mode_key == 'swarm':
                            swarm_kwargs = common_kwargs.copy()
                            swarm_kwargs['alpha'] = 0.85
                            sns.swarmplot(**swarm_kwargs)
                        else:
                            strip_kwargs = common_kwargs.copy()
                            strip_kwargs['alpha'] = 0.7
                            strip_kwargs['jitter'] = jitter_value
                            sns.stripplot(**strip_kwargs)
                    except Exception:
                        if mode_key != 'strip':
                            try:
                                strip_kwargs = common_kwargs.copy()
                                strip_kwargs['alpha'] = 0.7
                                strip_kwargs['jitter'] = jitter_value
                                sns.stripplot(**strip_kwargs)
                            except Exception:
                                pass

                # If categorical X is provided, draw grouped violins/boxes/points
                if x_col:
                    custom_point_color = point_color if point_color else 'k'
                    violin_color = point_color if point_color else None
                    connect_points = getattr(self, 'param_dist_connect_points_var', tk.BooleanVar(value=False)).get()
                    violin_palette_kwargs = {}
                    if hue_col:
                        if palette_arg:
                            violin_palette_kwargs['palette'] = palette_arg
                    else:
                        if category_palette:
                            violin_palette_kwargs['palette'] = {str(k): v for k, v in category_palette.items()}
                        elif violin_color:
                            violin_palette_kwargs['color'] = violin_color

                    # Use violinplot or boxplot as base
                    half_violin = getattr(self, 'param_dist_half_violin_var', tk.BooleanVar(value=False)).get()
                    if show_violin:
                        # If half-violin requested and there's no hue, render custom half-violins
                        if half_violin and not hue_col:
                            try:
                                # Use x_order if available, otherwise get unique values
                                if x_order:
                                    cats = x_order
                                else:
                                    cats = list(plot_df[x_col].astype(str).unique())
                                # numeric positions for categories
                                positions = np.arange(len(cats))
                                all_min = float(plot_df[y_col].min())
                                all_max = float(plot_df[y_col].max())
                                if all_min == all_max:
                                    all_min -= 0.5
                                    all_max += 0.5
                                y_grid = np.linspace(all_min, all_max, 300)
                                max_width = max(0.05, violin_width * 0.5)
                                kde_list = []
                                for i, cat in enumerate(cats):
                                    vals = plot_df.loc[plot_df[x_col].astype(str) == cat, y_col].dropna().values
                                    if len(vals) < 2:
                                        kde_list.append(None)
                                        continue
                                    try:
                                        kde = stats.gaussian_kde(vals)
                                        kde_vals = kde(y_grid)
                                    except Exception:
                                        kde_list.append(None)
                                        continue
                                    # normalize so the widest violin has width max_width
                                    kde_list.append(kde_vals)

                                # compute global max for normalization
                                max_kde = 0.0
                                for vals in kde_list:
                                    if vals is not None:
                                        max_kde = max(max_kde, np.max(vals))
                                if max_kde <= 0:
                                    max_kde = 1.0

                                for i, cat in enumerate(cats):
                                    kde_vals = kde_list[i]
                                    x0 = positions[i]
                                    if kde_vals is None:
                                        # draw a small vertical line to indicate presence
                                        subvals = plot_df.loc[plot_df[x_col].astype(str) == cat, y_col].dropna().values
                                        if len(subvals) == 0:
                                            continue
                                        ax.vlines(x0 - 0.05, np.min(subvals), np.max(subvals), color='gray', alpha=violin_alpha, linewidth=4)
                                        continue
                                    kde_norm = (kde_vals / max_kde) * max_width
                                    # draw half-violin to the left of x0
                                    facecolor = category_palette.get(str(cat), violin_color if violin_color else 'C0')
                                    ax.fill_betweenx(y_grid, x0, x0 - kde_norm, facecolor=facecolor, alpha=violin_alpha, linewidth=0)

                                # ticks and limits
                                ax.set_xticks(positions)
                                ax.set_xticklabels(cats)
                                ax.set_xlim(-0.5, len(cats) - 0.5)

                                # If requested, overlay boxplots and points
                                half_boxpoints = getattr(self, 'param_dist_half_violin_boxpoints_var', tk.BooleanVar(value=False)).get()
                                if half_boxpoints:
                                    # prepare data for matplotlib boxplot (list per category)
                                    bp_data = [plot_df.loc[plot_df[x_col].astype(str) == cat, y_col].dropna().values for cat in cats]
                                    try:
                                        box_shift = max_width * 0.95
                                        box_positions = positions + box_shift
                                        bp = ax.boxplot(bp_data, positions=box_positions, widths=box_overlay_width, patch_artist=True, manage_ticks=False, showfliers=False)
                                        box_colors = [category_palette.get(str(cat), '#FFFFFF') for cat in cats]
                                        for patch, color in zip(bp.get('boxes', []), box_colors):
                                            patch.set_facecolor(color)
                                            patch.set_edgecolor('black')
                                            patch.set_alpha(0.85)
                                        for median in bp.get('medians', []):
                                            median.set_color('black')
                                        for whisker in bp.get('whiskers', []):
                                            whisker.set_color('black')
                                        for cap in bp.get('caps', []):
                                            cap.set_color('black')
                                    except Exception:
                                        pass

                                    # overlay points (swarm preferred)
                                    render_points(order=cats, mode='swarm', allow_dodge=False, legend_mode=False)

                            except Exception:
                                sns.violinplot(data=plot_df, x=x_col, y=y_col, hue=hue_col if hue_col else None, order=x_order, ax=ax, cut=0, scale='width', bw=violin_bw, width=violin_width, **violin_palette_kwargs)
                            self._apply_violin_alpha(ax, violin_alpha)
                            if show_box:
                                self._overlay_boxplot(
                                    ax=ax,
                                    data=plot_df,
                                    x_col=x_col,
                                    y_col=y_col,
                                    order=x_order if x_order else None,
                                    hue_col=hue_col,
                                    palette_arg=palette_arg,
                                    width=box_overlay_width,
                                    edge_color=violin_color,
                                    category_palette=category_palette
                                )
                        else:
                            # If raincloud requested, try to render a compact violin + points overlay
                            plot_palette = palette if palette else None
                            if raincloud:
                                # If hue is present and has exactly 2 levels, we can use split=True
                                if hue_col:
                                    try:
                                        unique_hues = plot_df[hue_col].dropna().unique()
                                        if len(unique_hues) == 2:
                                            sns.violinplot(data=plot_df, x=x_col, y=y_col, hue=hue_col, order=x_order, split=True, inner=None, ax=ax, cut=0, scale='width', bw=violin_bw, width=violin_width, **violin_palette_kwargs)
                                        else:
                                            sns.violinplot(data=plot_df, x=x_col, y=y_col, hue=hue_col, order=x_order, inner=None, ax=ax, cut=0, scale='width', bw=violin_bw, width=violin_width, **violin_palette_kwargs)
                                    except Exception:
                                        sns.violinplot(data=plot_df, x=x_col, y=y_col, hue=hue_col if hue_col else None, order=x_order, inner=None, ax=ax, cut=0, scale='width', bw=violin_bw, width=violin_width, **violin_palette_kwargs)
                                else:
                                    sns.violinplot(data=plot_df, x=x_col, y=y_col, order=x_order, inner=None, ax=ax, cut=0, scale='width', bw=violin_bw, width=violin_width, **violin_palette_kwargs)
                                # overlay scattered points to emulate raincloud (dodge only meaningful with hue)
                                    render_points(order=x_order, mode='strip', enforce_single_color=not bool(hue_col), allow_dodge=bool(hue_col), legend_mode=False)
                            else:
                                sns.violinplot(data=plot_df, x=x_col, y=y_col, hue=hue_col if hue_col else None, order=x_order, ax=ax, cut=0, scale='width', bw=violin_bw, width=violin_width, **violin_palette_kwargs)
                            self._apply_violin_alpha(ax, violin_alpha)
                            if show_box:
                                self._overlay_boxplot(
                                    ax=ax,
                                    data=plot_df,
                                    x_col=x_col,
                                    y_col=y_col,
                                    order=x_order if x_order else None,
                                    hue_col=hue_col,
                                    palette_arg=palette_arg,
                                    width=box_overlay_width,
                                    edge_color=violin_color,
                                    category_palette=category_palette
                                )
                            # Custom legend assembly consolidating color/size/style once each
                            try:
                                existing_legend = ax.get_legend()
                                if existing_legend is not None:
                                    existing_legend.remove()
                            except Exception:
                                pass

                            if show_legend:
                                point_color_hex = self._color_to_hex(custom_point_color)
                                combined_handles = []
                                combined_labels = []

                                def add_section(section_title, entries):
                                    if not section_title or not entries:
                                        return
                                    combined_handles.append(Line2D([0], [0], linestyle='None', marker=None, alpha=0))
                                    combined_labels.append(f"{section_title}:")
                                    for entry_handle, entry_label in entries:
                                        combined_handles.append(entry_handle)
                                        combined_labels.append(f"  {entry_label}")

                                # Color legend (hue or category palette)
                                color_entries = []
                                color_title = None
                                if hue_col and hue_col in plot_df.columns:
                                    hue_series = plot_df[hue_col]
                                    hue_dtype = getattr(hue_series, 'dtype', None)
                                    if self._is_categorical(hue_dtype):
                                        hue_levels = [lvl for lvl in hue_series.cat.categories if pd.notna(lvl)]
                                    else:
                                        hue_levels = [lvl for lvl in pd.unique(hue_series.dropna())]
                                    if hue_levels:
                                        color_title = hue_col
                                        palette_colors = {}
                                        if isinstance(palette_arg, dict):
                                            palette_colors = {str(k): self._color_to_hex(v) for k, v in palette_arg.items() if v}
                                        else:
                                            palette_list = None
                                            if isinstance(palette_arg, (list, tuple)):
                                                palette_list = [self._color_to_hex(col) for col in palette_arg]
                                            elif isinstance(palette_arg, str):
                                                try:
                                                    palette_list = [self._color_to_hex(col) for col in sns.color_palette(palette_arg, n_colors=len(hue_levels))]
                                                except Exception:
                                                    palette_list = None
                                            if palette_list is None:
                                                try:
                                                    palette_list = [self._color_to_hex(col) for col in sns.color_palette(n_colors=len(hue_levels))]
                                                except Exception:
                                                    palette_list = None
                                            if palette_list:
                                                palette_colors = {str(hue_levels[idx]): palette_list[idx % len(palette_list)] for idx in range(len(hue_levels))}
                                        for idx, level in enumerate(hue_levels):
                                            key = str(level)
                                            color_hex = palette_colors.get(key) if palette_colors else None
                                            if not color_hex and palette_colors:
                                                color_hex = palette_colors.get(level)
                                            if not color_hex:
                                                color_hex = point_color_hex
                                            handle = Line2D([0], [0], marker='o', linestyle='None', markersize=max(point_size, 6), markerfacecolor=color_hex, markeredgecolor=color_hex)
                                            color_entries.append((handle, str(level)))
                                elif category_palette:
                                    mapped_palette = {str(k): self._color_to_hex(v) for k, v in category_palette.items() if v}
                                    categories_for_legend = x_order if x_order else [str(cat) for cat in mapped_palette.keys()]
                                    categories_for_legend = [str(cat) for cat in categories_for_legend if cat is not None]
                                    if categories_for_legend:
                                        color_title = x_col_original if x_col_original else (x_col if x_col else 'Color')
                                        for cat in categories_for_legend:
                                            color_hex = mapped_palette.get(cat)
                                            if not color_hex:
                                                continue
                                            handle = Line2D([0], [0], marker='o', linestyle='None', markersize=max(point_size, 6), markerfacecolor=color_hex, markeredgecolor=color_hex)
                                            color_entries.append((handle, cat))
                                if color_entries and not color_title:
                                    color_title = 'Color'

                                # Size legend
                                size_entries = []
                                size_title = size_col if size_col else None
                                if size_col and size_mapper and size_col in plot_df.columns:
                                    size_series = plot_df[size_col]
                                    if not size_series.empty:
                                        numeric_values = pd.to_numeric(size_series, errors='coerce').dropna()
                                        legend_candidates = []
                                        if not numeric_values.empty:
                                            quantiles = np.quantile(numeric_values, [0.0, 0.5, 1.0])
                                            quantiles = np.unique(np.round(quantiles, decimals=6))
                                            if quantiles.size == 1:
                                                quantiles = np.unique(np.round(numeric_values.to_numpy(), decimals=6))
                                            quantiles = sorted(float(val) for val in quantiles)
                                            if len(quantiles) > 4:
                                                quantiles = [quantiles[0], float(np.median(quantiles)), quantiles[-1]]
                                            for val in quantiles:
                                                try:
                                                    area = float(size_mapper(pd.Series([val]))[0])
                                                except Exception:
                                                    continue
                                                legend_candidates.append((f"{val:g}", area))
                                        else:
                                            cat_values = [val for val in pd.unique(size_series.dropna())]
                                            limit = min(len(cat_values), 6)
                                            for val in cat_values[:limit]:
                                                try:
                                                    area = float(size_mapper(pd.Series([val]))[0])
                                                except Exception:
                                                    continue
                                                legend_candidates.append((str(val), area))
                                        if legend_candidates:
                                            legend_candidates.sort(key=lambda item: item[1])
                                            area_seen = set()
                                            for label_text, area in legend_candidates:
                                                if area <= 0 or label_text in area_seen:
                                                    continue
                                                area_seen.add(label_text)
                                                markersize = max(np.sqrt(abs(area)), 4.0)
                                                handle = Line2D([0], [0], marker='o', linestyle='None', markersize=markersize, markerfacecolor='none', markeredgecolor='#333333', color='#333333')
                                                size_entries.append((handle, label_text))
                                                if len(size_entries) >= 5:
                                                    break
                                        if not size_title:
                                            size_title = 'Tamaño'

                                # Style legend
                                style_entries = []
                                style_title = style_col if style_col else None
                                if style_col and style_mapper and style_col in plot_df.columns:
                                    style_series = plot_df[style_col].dropna()
                                    if not style_series.empty:
                                        style_levels = [val for val in pd.unique(style_series)]
                                        limit = min(len(style_levels), 8)
                                        marker_size = max(point_size, 8)
                                        for val in style_levels[:limit]:
                                            try:
                                                marker_symbol = style_mapper(pd.Series([val]))[0]
                                            except Exception:
                                                marker_symbol = 'o'
                                            handle = Line2D([0], [0], marker=marker_symbol, linestyle='None', markersize=marker_size, markerfacecolor='none', markeredgecolor=point_color_hex, color=point_color_hex, markeredgewidth=1.4)
                                            style_entries.append((handle, str(val)))
                                        if not style_title:
                                            style_title = 'Forma'

                                add_section(color_title, color_entries)
                                add_section(size_title, size_entries)
                                add_section(style_title, style_entries)

                                if combined_handles:
                                    legend = ax.legend(combined_handles, combined_labels, loc=legend_loc, frameon=True, handlelength=2)
                                    try:
                                        legend._legend_box.align = "left"
                                    except Exception:
                                        pass
                                    try:
                                        for text in legend.get_texts():
                                            text.set_fontsize(legend_size)
                                    except Exception:
                                        pass
                                else:
                                    try:
                                        legend = ax.get_legend()
                                        if legend is not None:
                                            legend.remove()
                                    except Exception:
                                        pass
                            else:
                                try:
                                    legend = ax.get_legend()
                                    if legend is not None:
                                        legend.remove()
                                except Exception:
                                    pass
                    elif show_box:
                        sns.boxplot(data=plot_df, x=x_col, y=y_col, hue=hue_col if hue_col else None, order=x_order, ax=ax)
                    else:
                        # Dot/point summary (use pointplot of means/medians)
                        show_summary = getattr(self, 'param_dist_show_summary_var', tk.BooleanVar(value=False)).get()
                        join_flag = True if connect_points else False
                        if show_summary:
                            try:
                                sns.pointplot(data=plot_df, x=x_col, y=y_col, hue=hue_col if hue_col else None, order=x_order, join=join_flag, ax=ax, ci=95)
                            except Exception:
                                try:
                                    sns.pointplot(data=plot_df, x=x_col, y=y_col, hue=hue_col if hue_col else None, order=x_order, join=join_flag, ax=ax, ci='sd')
                                except Exception:
                                    pass

                    # Overlay points if requested
                    if point_type and point_type != 'Ninguno':
                        point_mode = 'strip'
                        if point_type_key == 'swarm' or point_type in ('Swarm',):
                            point_mode = 'swarm'
                        elif point_type_key in ('center', 'hex', 'square') or point_type in ('Center', 'Hex', 'Square'):
                            point_mode = 'swarm'
                        render_points(order=x_order, mode=point_mode, allow_dodge=bool(hue_col), legend_mode=False)

                else:
                    # Single numeric distribution: histogram + optional KDE, box/violin as inset
                    # Show histogram only if user enabled the checkbox; otherwise show KDE only
                    show_hist = getattr(self, 'param_dist_show_hist_var', tk.BooleanVar(value=False)).get()
                    if show_hist:
                        sns.histplot(plot_df[y_col], kde=True, ax=ax)
                    else:
                        try:
                            sns.kdeplot(plot_df[y_col], ax=ax, fill=False)
                        except Exception:
                            # Fallback: histogram if KDE fails
                            sns.histplot(plot_df[y_col], kde=True, ax=ax)

                    if show_violin or show_box:
                        # Add a small inset axis for box/violin
                        try:
                            from mpl_toolkits.axes_grid1.inset_locator import inset_axes
                            iax = inset_axes(ax, width="30%", height="30%", loc='upper right')
                            if show_violin:
                                sns.violinplot(y=plot_df[y_col], ax=iax, orient='v')
                            else:
                                sns.boxplot(y=plot_df[y_col], ax=iax, orient='v')
                            iax.set_xlabel('')
                            iax.set_ylabel('')
                            iax.set_xticks([])
                        except Exception:
                            # Fallback: draw boxplot on main ax at top
                            sns.boxplot(y=plot_df[y_col], ax=ax, width=0.2)

                    # Overlay points: for single-variable distribution prefer swarm (jitter)
                    try:
                        # Use swarmplot along the x-axis (orient horizontal)
                        single_point_color = point_color if point_color else 'k'
                        sns.swarmplot(x=plot_df[y_col], ax=ax, color=single_point_color, size=point_size)
                    except Exception:
                        # Fallback to stripplot (jitter)
                        single_point_color = point_color if point_color else 'k'
                        sns.stripplot(x=plot_df[y_col], ax=ax, jitter=0.25, color=single_point_color, size=point_size)

            elif chart_type == "Forest Plot (Comparaciones)":
                y_var = getattr(self, 'param_forest_y_var', None)
                value_var = getattr(self, 'param_forest_value_var', None)
                hue_var = getattr(self, 'param_forest_hue_var', None)
                norm_mode = getattr(self, 'param_forest_norm_mode_var', StringVar(value='Automático')).get()
                comp_mode = getattr(self, 'param_forest_comp_mode_var', StringVar(value='Automático')).get()
                show_ci = getattr(self, 'param_forest_show_ci_var', tk.BooleanVar(value=True)).get()
                show_sig = getattr(self, 'param_forest_show_sig_var', tk.BooleanVar(value=True)).get()

                y_col = y_var.get() if y_var else None
                value_col = value_var.get() if value_var else None
                hue_col = hue_var.get() if hue_var else None

                if not y_col or not value_col:
                    messagebox.showerror("Error", "Seleccione la variable categórica y la variable numérica.", parent=self.parent_for_dialogs)
                    plt.close(plt_fig)
                    return

                params_used_log.extend([f"  Filas: {y_col}", f"  Valor: {value_col}"])
                if hue_col:
                    params_used_log.append(f"  Comparación: {hue_col}")
                params_used_log.extend([f"  Normalidad: {norm_mode}", f"  Diferencias: {comp_mode}"])

                plot_df = filtered_data[[y_col, value_col] + ([hue_col] if hue_col else [])].dropna()
                plot_df[value_col] = pd.to_numeric(plot_df[value_col], errors='coerce')
                plot_df = plot_df.dropna(subset=[value_col])

                if plot_df.empty:
                    messagebox.showwarning("Sin Datos", "No hay datos válidos para graficar.", parent=self.parent_for_dialogs)
                    plt.close(plt_fig)
                    return

                # Compute means, CIs, and significance for each row category
                categories = sorted(plot_df[y_col].unique())
                forest_data = []

                for cat in categories:
                    cat_df = plot_df[plot_df[y_col] == cat]
                    if hue_col:
                        groups = sorted(cat_df[hue_col].unique())
                        if len(groups) < 2:
                            continue
                        group_vals = {g: cat_df[cat_df[hue_col] == g][value_col].dropna().values for g in groups}
                        # Test normality per group
                        normal_dict = {}
                        for g, vals in group_vals.items():
                            if len(vals) < 3:
                                normal_dict[g] = False
                                continue
                            n = len(vals)
                            if norm_mode == 'Automático':
                                test_name = 'Shapiro-Wilk' if n < 50 else 'Kolmogorov-Smirnov'
                            else:
                                test_name = norm_mode
                            try:
                                if 'Shapiro' in test_name:
                                    stat, p_val = shapiro(vals)
                                else:
                                    from scipy.stats import kstest
                                    vals_std = (vals - np.mean(vals)) / (np.std(vals) + 1e-9)
                                    stat, p_val = kstest(vals_std, 'norm')
                                normal_dict[g] = (p_val > 0.05)
                            except Exception:
                                normal_dict[g] = False
                        all_normal = all(normal_dict.values())
                        # Choose test
                        if comp_mode == 'Paramétrico' or (comp_mode == 'Automático' and all_normal):
                            test_name = 't-test' if len(groups) == 2 else 'ANOVA'
                            if len(groups) == 2:
                                from scipy.stats import ttest_ind
                                stat_val, p_val = ttest_ind(group_vals[groups[0]], group_vals[groups[1]], nan_policy='omit')
                            else:
                                from scipy.stats import f_oneway
                                stat_val, p_val = f_oneway(*[group_vals[g] for g in groups])
                        else:
                            test_name = 'Mann-Whitney' if len(groups) == 2 else 'Kruskal-Wallis'
                            if len(groups) == 2:
                                from scipy.stats import mannwhitneyu
                                stat_val, p_val = mannwhitneyu(group_vals[groups[0]], group_vals[groups[1]], alternative='two-sided')
                            else:
                                from scipy.stats import kruskal
                                stat_val, p_val = kruskal(*[group_vals[g] for g in groups])

                        sig_marker = ''
                        if show_sig:
                            if p_val < 0.001:
                                sig_marker = '***'
                            elif p_val < 0.01:
                                sig_marker = '**'
                            elif p_val < 0.05:
                                sig_marker = '*'

                        for g in groups:
                            vals = group_vals[g]
                            mean = np.mean(vals)
                            sem = stats.sem(vals)
                            ci_low = mean - 1.96 * sem
                            ci_high = mean + 1.96 * sem
                            forest_data.append({
                                'category': str(cat),
                                'group': str(g),
                                'mean': mean,
                                'ci_low': ci_low,
                                'ci_high': ci_high,
                                'sig': sig_marker,
                                'p_value': p_val,
                                'effect': mean
                            })
                    else:
                        vals = cat_df[value_col].values
                        mean = np.mean(vals)
                        sem = stats.sem(vals)
                        ci_low = mean - 1.96 * sem
                        ci_high = mean + 1.96 * sem
                        forest_data.append({
                            'category': str(cat),
                            'group': None,
                            'mean': mean,
                            'ci_low': ci_low,
                            'ci_high': ci_high,
                            'sig': '',
                            'p_value': 1.0,
                            'effect': mean
                        })

                if not forest_data:
                    messagebox.showwarning("Sin Datos", "No se pudieron calcular datos para el forest plot.", parent=self.parent_for_dialogs)
                    plt.close(plt_fig)
                    return

                # Corrección múltiple Holm sobre todos los p
                p_vals = [row['p_value'] for row in forest_data if row['p_value'] is not None]
                if p_vals:
                    try:
                        from MATLAB_graficaqq import holm_correction
                        p_adj_all = holm_correction(p_vals)
                        idx = 0
                        for row in forest_data:
                            if row['p_value'] is not None:
                                row['p_adj'] = p_adj_all[idx]
                                idx += 1
                            else:
                                row['p_adj'] = None
                    except Exception:
                        for row in forest_data:
                            row['p_adj'] = row.get('p_value', None)

                # Plot
                df_forest = pd.DataFrame(forest_data)
                unique_cats = df_forest['category'].unique()
                y_positions = {}
                current_y = 0
                for cat in unique_cats:
                    cat_rows = df_forest[df_forest['category'] == cat]
                    n_rows = len(cat_rows)
                    positions = np.arange(current_y, current_y + n_rows)
                    for i, (idx, row) in enumerate(cat_rows.iterrows()):
                        y_positions[idx] = positions[i]
                    current_y += n_rows + 0.5

                # Color palette for groups
                if hue_col:
                    unique_groups = sorted(df_forest['group'].dropna().unique())
                    try:
                        group_colors_palette = sns.color_palette('deep', n_colors=len(unique_groups))
                        group_color_map = {g: group_colors_palette[i] for i, g in enumerate(unique_groups)}
                    except Exception:
                        group_color_map = {g: self.color_options[i % len(self.color_options)] for i, g in enumerate(unique_groups)}
                else:
                    group_color_map = {}

                for idx, row in df_forest.iterrows():
                    y_pos = y_positions[idx]
                    mean_val = row['mean']
                    ci_low_val = row['ci_low']
                    ci_high_val = row['ci_high']
                    group_label = row['group']
                    color = group_color_map.get(group_label, 'C0') if group_label else 'C0'
                    # Plot CI as error bar
                    if show_ci:
                        ax.plot([ci_low_val, ci_high_val], [y_pos, y_pos], color=color, linewidth=2, alpha=0.6)
                    # Plot mean as marker
                    ax.scatter(mean_val, y_pos, color=color, s=80, zorder=5, edgecolors='black', linewidths=1)
                    # Annotate significance
                    text_x = ci_high_val if show_ci else mean_val
                    if show_sig and row['sig']:
                        ax.text(text_x + 0.02 * (ax.get_xlim()[1] - ax.get_xlim()[0]), y_pos, row['sig'], ha='left', va='center', fontsize=10, fontweight='bold')
                    # annotate p and effect size
                        p_show = row['p_adj'] if row.get('p_adj') is not None else row['p_value']
                        ax.text(ax.get_xlim()[0] + 0.01 * (ax.get_xlim()[1] - ax.get_xlim()[0]), y_pos,
                            f"p={p_show:.3g}, efecto={row['effect']:.3f}",
                            ha='left', va='center', fontsize=8, color='gray')

                # Set y-tick labels
                y_labels = []
                y_ticks = []
                for cat in unique_cats:
                    cat_rows = df_forest[df_forest['category'] == cat]
                    for idx, row_data in cat_rows.iterrows():
                        y_ticks.append(y_positions[idx])
                        if row_data['group']:
                            y_labels.append(f"{cat} - {row_data['group']}")
                        else:
                            y_labels.append(str(cat))

                ax.set_yticks(y_ticks)
                ax.set_yticklabels(y_labels)
                ax.set_xlabel(value_col)
                ax.set_ylabel(y_col)
                ax.axvline(0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
                ax.invert_yaxis()

                # Legend for groups
                if hue_col and group_color_map:
                    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=group_color_map[g], markersize=8, label=g) for g in unique_groups]
                    ax.legend(handles=handles, title=hue_col, loc='best')

            elif chart_type == "Gráfico Lollipop":
                x_var = getattr(self, 'param_lollipop_x_var', None)
                y_var = getattr(self, 'param_lollipop_y_var', None)
                orient_var = getattr(self, 'param_lollipop_orientation_var', None)
                x_col = x_var.get() if x_var else None
                y_col_l = y_var.get() if y_var else None
                orientation = orient_var.get() if orient_var else "Vertical"

                if not x_col or not y_col_l:
                    messagebox.showerror("Error", "Seleccione la variable categórica y la variable numérica.", parent=self.parent_for_dialogs)
                    plt.close(plt_fig)
                    return

                params_used_log.extend([f"  Categoría: {x_col}", f"  Valor: {y_col_l}", f"  Orientación: {orientation}"])
                plot_df = filtered_data[[x_col, y_col_l]].dropna()
                plot_df[y_col_l] = pd.to_numeric(plot_df[y_col_l], errors='coerce')
                plot_df = plot_df.dropna(subset=[y_col_l])
                agg_df = plot_df.groupby(x_col, sort=False)[y_col_l].mean().reset_index()
                custom_order = self._recode_orders.get(x_col)
                if custom_order:
                    agg_df[x_col] = pd.Categorical(agg_df[x_col], categories=custom_order, ordered=True)
                    agg_df = agg_df.sort_values(x_col)

                colors = sns.color_palette(palette, n_colors=len(agg_df))
                if orientation == "Horizontal":
                    ax.hlines(y=range(len(agg_df)), xmin=0, xmax=agg_df[y_col_l].values, colors=colors, linewidth=2)
                    ax.scatter(agg_df[y_col_l].values, range(len(agg_df)), color=colors, s=point_size * 20, zorder=5, edgecolors='black', linewidths=0.5)
                    ax.set_yticks(range(len(agg_df)))
                    ax.set_yticklabels(agg_df[x_col].astype(str))
                    ax.set_xlabel(y_col_l)
                    ax.set_ylabel(x_col)
                    mark_axis('y', x_col, agg_df)
                else:
                    ax.vlines(x=range(len(agg_df)), ymin=0, ymax=agg_df[y_col_l].values, colors=colors, linewidth=2)
                    ax.scatter(range(len(agg_df)), agg_df[y_col_l].values, color=colors, s=point_size * 20, zorder=5, edgecolors='black', linewidths=0.5)
                    ax.set_xticks(range(len(agg_df)))
                    ax.set_xticklabels(agg_df[x_col].astype(str), rotation=tick_rotation if tick_rotation else 45, ha='right')
                    ax.set_xlabel(x_col)
                    ax.set_ylabel(y_col_l)
                    mark_axis('x', x_col, agg_df)

            elif chart_type == "Gráfico de Pirámide":
                age_var = getattr(self, 'param_pyr_age_var', None)
                male_var = getattr(self, 'param_pyr_male_var', None)
                female_var = getattr(self, 'param_pyr_female_var', None)
                age_col = age_var.get() if age_var else None
                male_col = male_var.get() if male_var else None
                female_col = female_var.get() if female_var else None

                if not age_col or not male_col or not female_col:
                    messagebox.showerror("Error", "Seleccione las 3 variables requeridas (Edad, Izquierda, Derecha).", parent=self.parent_for_dialogs)
                    plt.close(plt_fig)
                    return

                params_used_log.extend([f"  Eje Y: {age_col}", f"  Izquierda: {male_col}", f"  Derecha: {female_col}"])
                plot_df = filtered_data[[age_col, male_col, female_col]].dropna()
                plot_df[male_col] = pd.to_numeric(plot_df[male_col], errors='coerce')
                plot_df[female_col] = pd.to_numeric(plot_df[female_col], errors='coerce')
                plot_df = plot_df.dropna()
                agg_df = plot_df.groupby(age_col, sort=False)[[male_col, female_col]].sum().reset_index()
                custom_order = self._recode_orders.get(age_col)
                if custom_order:
                    agg_df[age_col] = pd.Categorical(agg_df[age_col], categories=custom_order, ordered=True)
                    agg_df = agg_df.sort_values(age_col)

                y_pos = range(len(agg_df))
                ax.barh(y_pos, -agg_df[male_col].values, color=sns.color_palette(palette, 2)[0] if palette else 'steelblue', label=male_col, edgecolor='white', linewidth=0.5)
                ax.barh(y_pos, agg_df[female_col].values, color=sns.color_palette(palette, 2)[1] if palette else 'salmon', label=female_col, edgecolor='white', linewidth=0.5)
                ax.set_yticks(list(y_pos))
                ax.set_yticklabels(agg_df[age_col].astype(str))
                max_val = max(agg_df[male_col].max(), agg_df[female_col].max()) * 1.1
                ax.set_xlim(-max_val, max_val)
                ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{abs(x):.0f}'))
                ax.axvline(0, color='black', linewidth=0.8)
                ax.legend(loc='best')
                mark_axis('y', age_col, agg_df)

            elif chart_type == "Gráfico Radial":
                radar_vars_list = getattr(self, 'param_radar_vars', [])
                selected_vars = [v.get() for v in radar_vars_list if v.get()]
                if len(selected_vars) < 3:
                    messagebox.showerror("Error", "Seleccione al menos 3 variables numéricas.", parent=self.parent_for_dialogs)
                    plt.close(plt_fig)
                    return

                params_used_log.append(f"  Variables: {', '.join(selected_vars)}")
                plot_df = filtered_data[selected_vars].dropna()
                for c in selected_vars:
                    plot_df[c] = pd.to_numeric(plot_df[c], errors='coerce')
                plot_df = plot_df.dropna()
                if plot_df.empty:
                    messagebox.showwarning("Sin Datos", "No hay datos válidos.", parent=self.parent_for_dialogs)
                    plt.close(plt_fig)
                    return

                means = plot_df[selected_vars].mean().values
                # Normalize to 0-1 range for radar
                mins = plot_df[selected_vars].min().values
                maxs = plot_df[selected_vars].max().values
                ranges = maxs - mins
                ranges[ranges == 0] = 1
                normalized = (means - mins) / ranges

                N = len(selected_vars)
                angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
                values = normalized.tolist()
                values += values[:1]
                angles += angles[:1]

                plt.close(plt_fig)
                plt_fig, ax = plt.subplots(figsize=(fig_width, fig_height), subplot_kw=dict(polar=True))
                ax.fill(angles, values, alpha=0.25, color='steelblue')
                ax.plot(angles, values, 'o-', linewidth=2, color='steelblue')
                ax.set_xticks(angles[:-1])
                ax.set_xticklabels(selected_vars, size=9)
                ax.set_ylim(0, 1)
                ax.set_title(title if title else chart_type, pad=20)

            elif chart_type == "Gráfico de Bala":
                val_var = getattr(self, 'param_bullet_value_var', None)
                tgt_var = getattr(self, 'param_bullet_target_var', None)
                rng_var = getattr(self, 'param_bullet_ranges_var', None)
                val_col = val_var.get() if val_var else None
                tgt_col = tgt_var.get() if tgt_var else None
                ranges_str = rng_var.get() if rng_var else ""

                if not val_col or not tgt_col:
                    messagebox.showerror("Error", "Seleccione las variables de Valor y Objetivo.", parent=self.parent_for_dialogs)
                    plt.close(plt_fig)
                    return

                params_used_log.extend([f"  Valor: {val_col}", f"  Objetivo: {tgt_col}"])
                plot_df = filtered_data[[val_col, tgt_col]].dropna()
                plot_df[val_col] = pd.to_numeric(plot_df[val_col], errors='coerce')
                plot_df[tgt_col] = pd.to_numeric(plot_df[tgt_col], errors='coerce')
                plot_df = plot_df.dropna()
                actual = plot_df[val_col].mean()
                target = plot_df[tgt_col].mean()

                if ranges_str.strip():
                    try:
                        range_vals = sorted([float(x.strip()) for x in ranges_str.split(',')])
                    except ValueError:
                        range_vals = [target * 0.5, target * 0.75, target * 1.2]
                else:
                    range_vals = [target * 0.5, target * 0.75, target * 1.2]
                params_used_log.append(f"  Rangos: {range_vals}")

                grays = ['#ddd', '#bbb', '#999']
                prev = 0
                for i, r in enumerate(range_vals):
                    ax.barh(0, r - prev, left=prev, height=0.6, color=grays[i % len(grays)], edgecolor='none')
                    prev = r
                ax.barh(0, actual, height=0.3, color='steelblue', edgecolor='black', linewidth=0.5, zorder=3)
                ax.axvline(target, color='red', linewidth=2.5, zorder=4)
                ax.set_yticks([])
                ax.set_xlabel("Valor")
                ax.text(target, 0.45, f'Objetivo: {target:.1f}', ha='center', va='bottom', fontsize=9, color='red', fontweight='bold')
                ax.text(actual, -0.45, f'Actual: {actual:.1f}', ha='center', va='top', fontsize=9, color='steelblue', fontweight='bold')

            elif chart_type == "Mapa de Árbol":
                val_var = getattr(self, 'param_treemap_values_var', None)
                name_var = getattr(self, 'param_treemap_names_var', None)
                val_col = val_var.get() if val_var else None
                name_col = name_var.get() if name_var else None

                if not val_col or not name_col:
                    messagebox.showerror("Error", "Seleccione las columnas de Valores y Nombres.", parent=self.parent_for_dialogs)
                    plt.close(plt_fig)
                    return

                params_used_log.extend([f"  Valores: {val_col}", f"  Nombres: {name_col}"])
                plot_df = filtered_data[[name_col, val_col]].dropna()
                plot_df[val_col] = pd.to_numeric(plot_df[val_col], errors='coerce')
                plot_df = plot_df.dropna(subset=[val_col])
                agg_df = plot_df.groupby(name_col, sort=False)[val_col].sum().reset_index()
                agg_df = agg_df[agg_df[val_col] > 0].sort_values(val_col, ascending=False)

                if agg_df.empty:
                    messagebox.showwarning("Sin Datos", "No hay datos positivos para el mapa de árbol.", parent=self.parent_for_dialogs)
                    plt.close(plt_fig)
                    return

                colors = sns.color_palette(palette, n_colors=len(agg_df))
                labels = [f"{n}\n{v:,.0f}" for n, v in zip(agg_df[name_col], agg_df[val_col])]
                squarify.plot(sizes=agg_df[val_col].values, label=labels, color=colors, alpha=0.85, ax=ax, edgecolor='white', linewidth=2, text_kwargs={'fontsize': 9})
                ax.axis('off')

            elif chart_type == "Polígonos de Frecuencia":
                poly_var = getattr(self, 'param_poly_var', None)
                hue_var = getattr(self, 'param_poly_hue_var', None)
                poly_col = poly_var.get() if poly_var else None
                hue_col = hue_var.get() if hue_var else None

                if not poly_col:
                    messagebox.showerror("Error", "Seleccione una variable numérica.", parent=self.parent_for_dialogs)
                    plt.close(plt_fig)
                    return

                params_used_log.append(f"  Variable: {poly_col}")
                plot_df = filtered_data[[poly_col] + ([hue_col] if hue_col else [])].dropna()
                plot_df[poly_col] = pd.to_numeric(plot_df[poly_col], errors='coerce')
                plot_df = plot_df.dropna(subset=[poly_col])

                if hue_col:
                    params_used_log.append(f"  Agrupación: {hue_col}")
                    groups = sorted(plot_df[hue_col].unique())
                    colors = sns.color_palette(palette, n_colors=len(groups))
                    for i, g in enumerate(groups):
                        vals = plot_df[plot_df[hue_col] == g][poly_col].values
                        counts, bin_edges = np.histogram(vals, bins='auto')
                        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                        ax.plot(bin_centers, counts, 'o-', label=str(g), color=colors[i], linewidth=1.5, markersize=4)
                    ax.legend(title=hue_col, loc='best')
                else:
                    vals = plot_df[poly_col].values
                    counts, bin_edges = np.histogram(vals, bins='auto')
                    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                    ax.plot(bin_centers, counts, 'o-', color='steelblue', linewidth=1.5, markersize=4)
                    ax.fill_between(bin_centers, counts, alpha=0.15, color='steelblue')

                ax.set_xlabel(poly_col)
                ax.set_ylabel("Frecuencia")

            elif chart_type == "Diagrama de Tallo y Hojas":
                stem_var = getattr(self, 'param_stem_var', None)
                stem_col = stem_var.get() if stem_var else None
                if not stem_col:
                    messagebox.showerror("Error", "Seleccione una variable numérica.", parent=self.parent_for_dialogs)
                    plt.close(plt_fig)
                    return

                params_used_log.append(f"  Variable: {stem_col}")
                vals = pd.to_numeric(filtered_data[stem_col], errors='coerce').dropna().values
                if len(vals) == 0:
                    messagebox.showwarning("Sin Datos", "No hay datos numéricos válidos.", parent=self.parent_for_dialogs)
                    plt.close(plt_fig)
                    return

                sorted_vals = np.sort(vals)
                stem_dict = {}
                for v in sorted_vals:
                    stem_key = int(v // 10)
                    leaf = int(abs(v) % 10)
                    stem_dict.setdefault(stem_key, []).append(leaf)

                all_stems = sorted(stem_dict.keys())
                lines = []
                for s in all_stems:
                    leaves = ' '.join(str(l) for l in stem_dict[s])
                    lines.append(f"{s:>4} | {leaves}")

                display_text = "\n".join(lines)
                ax.text(0.05, 0.95, display_text, transform=ax.transAxes, fontsize=10, verticalalignment='top', fontfamily='monospace',
                        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
                ax.set_title(title if title else f"Tallo y Hojas — {stem_col}")
                ax.axis('off')

            elif chart_type == "Mapa de Calor":
                multi_vars = getattr(self, 'param_multi_select_vars', [])
                selected = [v.get() for v in multi_vars if v.get()]
                if len(selected) < 2:
                    messagebox.showerror("Error", "Seleccione al menos 2 variables numéricas.", parent=self.parent_for_dialogs)
                    plt.close(plt_fig)
                    return

                params_used_log.append(f"  Variables: {', '.join(selected)}")
                plot_df = filtered_data[selected].apply(pd.to_numeric, errors='coerce').dropna()
                if plot_df.empty:
                    messagebox.showwarning("Sin Datos", "No hay datos válidos.", parent=self.parent_for_dialogs)
                    plt.close(plt_fig)
                    return

                sns.heatmap(plot_df.corr(), annot=True, fmt=".2f", cmap=palette if palette and isinstance(palette, str) else 'coolwarm',
                            ax=ax, linewidths=0.5, square=True, vmin=-1, vmax=1)

            elif chart_type == "Correlograma":
                multi_vars = getattr(self, 'param_multi_select_vars', [])
                selected = [v.get() for v in multi_vars if v.get()]
                if len(selected) < 2:
                    messagebox.showerror("Error", "Seleccione al menos 2 variables numéricas.", parent=self.parent_for_dialogs)
                    plt.close(plt_fig)
                    return

                params_used_log.append(f"  Variables: {', '.join(selected)}")
                plot_df = filtered_data[selected].apply(pd.to_numeric, errors='coerce').dropna()
                if plot_df.empty:
                    messagebox.showwarning("Sin Datos", "No hay datos válidos.", parent=self.parent_for_dialogs)
                    plt.close(plt_fig)
                    return

                corr = plot_df.corr()
                mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
                sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
                            cmap=palette if palette and isinstance(palette, str) else 'RdBu_r',
                            ax=ax, linewidths=0.5, square=True, vmin=-1, vmax=1)

            elif chart_type == "Gráfico de Coordenadas Paralelas":
                class_var = getattr(self, 'param_pc_class_var', None)
                pc_vars_list = getattr(self, 'param_pc_vars', [])
                class_col = class_var.get() if class_var else None
                selected_cols = [v.get() for v in pc_vars_list if v.get()]

                if not class_col or len(selected_cols) < 2:
                    messagebox.showerror("Error", "Seleccione la variable de clase y al menos 2 variables numéricas.", parent=self.parent_for_dialogs)
                    plt.close(plt_fig)
                    return

                params_used_log.extend([f"  Clase: {class_col}", f"  Variables: {', '.join(selected_cols)}"])
                all_cols = [class_col] + selected_cols
                plot_df = filtered_data[all_cols].dropna()
                for c in selected_cols:
                    plot_df[c] = pd.to_numeric(plot_df[c], errors='coerce')
                plot_df = plot_df.dropna()

                if plot_df.empty:
                    messagebox.showwarning("Sin Datos", "No hay datos válidos.", parent=self.parent_for_dialogs)
                    plt.close(plt_fig)
                    return

                # Normalize each numeric column to 0-1
                norm_df = plot_df.copy()
                for c in selected_cols:
                    cmin, cmax = norm_df[c].min(), norm_df[c].max()
                    if cmax > cmin:
                        norm_df[c] = (norm_df[c] - cmin) / (cmax - cmin)
                    else:
                        norm_df[c] = 0.5

                classes = sorted(norm_df[class_col].unique())
                colors = sns.color_palette(palette, n_colors=len(classes))
                color_map = {cls: colors[i] for i, cls in enumerate(classes)}

                x_coords = list(range(len(selected_cols)))
                for _, row in norm_df.iterrows():
                    vals = [row[c] for c in selected_cols]
                    ax.plot(x_coords, vals, color=color_map.get(row[class_col], 'gray'), alpha=0.4, linewidth=0.8)

                ax.set_xticks(x_coords)
                ax.set_xticklabels(selected_cols, rotation=tick_rotation if tick_rotation else 30, ha='right')
                ax.set_ylabel("Valor normalizado (0-1)")
                handles = [plt.Line2D([0], [0], color=color_map[cls], linewidth=2, label=str(cls)) for cls in classes]
                ax.legend(handles=handles, title=class_col, loc='best')
                mark_axis('x', class_col, norm_df)

            elif chart_type == "Gráfico de Cascada":
                lbl_var = getattr(self, 'param_waterfall_label_var', None)
                val_var = getattr(self, 'param_waterfall_value_var', None)
                lbl_col = lbl_var.get() if lbl_var else None
                val_col = val_var.get() if val_var else None
                if not lbl_col or not val_col:
                    messagebox.showerror("Error", "Seleccione las variables de Categorías y Valores.", parent=self.parent_for_dialogs)
                    plt.close(plt_fig); return
                params_used_log.extend([f"  Categorías: {lbl_col}", f"  Valores: {val_col}"])
                plot_df = filtered_data[[lbl_col, val_col]].dropna()
                plot_df[val_col] = pd.to_numeric(plot_df[val_col], errors='coerce')
                plot_df = plot_df.dropna(subset=[val_col])
                agg_df = plot_df.groupby(lbl_col, sort=False)[val_col].sum().reset_index()
                custom_order = self._recode_orders.get(lbl_col)
                if custom_order:
                    agg_df[lbl_col] = pd.Categorical(agg_df[lbl_col], categories=custom_order, ordered=True)
                    agg_df = agg_df.sort_values(lbl_col)
                values = agg_df[val_col].values
                labels = agg_df[lbl_col].astype(str).values
                cumulative = np.cumsum(values)
                bottoms = np.concatenate([[0], cumulative[:-1]])
                colors_wf = ['#2ecc71' if v >= 0 else '#e74c3c' for v in values]
                ax.bar(range(len(values)), values, bottom=bottoms, color=colors_wf, edgecolor='white', linewidth=0.8, width=0.7)
                # Total bar
                total = cumulative[-1]
                ax.bar(len(values), total, color='#3498db', edgecolor='white', linewidth=0.8, width=0.7)
                all_labels = list(labels) + ['Total']
                ax.set_xticks(range(len(all_labels)))
                ax.set_xticklabels(all_labels, rotation=tick_rotation if tick_rotation else 45, ha='right')
                ax.set_ylabel(val_col)
                # Connector lines
                for i in range(len(values)):
                    ax.plot([i + 0.35, i + 0.65], [cumulative[i], cumulative[i]], color='gray', linewidth=0.8)
                mark_axis('x', lbl_col, agg_df)

            elif chart_type == "Gráfico de Embudo":
                stage_var = getattr(self, 'param_funnel_stage_var', None)
                val_var = getattr(self, 'param_funnel_value_var', None)
                stage_col = stage_var.get() if stage_var else None
                val_col = val_var.get() if val_var else None
                if not stage_col or not val_col:
                    messagebox.showerror("Error", "Seleccione las variables de Etapas y Valores.", parent=self.parent_for_dialogs)
                    plt.close(plt_fig); return
                params_used_log.extend([f"  Etapas: {stage_col}", f"  Valores: {val_col}"])
                plot_df = filtered_data[[stage_col, val_col]].dropna()
                plot_df[val_col] = pd.to_numeric(plot_df[val_col], errors='coerce')
                plot_df = plot_df.dropna(subset=[val_col])
                agg_df = plot_df.groupby(stage_col, sort=False)[val_col].sum().reset_index()
                custom_order = self._recode_orders.get(stage_col)
                if custom_order:
                    agg_df[stage_col] = pd.Categorical(agg_df[stage_col], categories=custom_order, ordered=True)
                    agg_df = agg_df.sort_values(stage_col)
                values = agg_df[val_col].values
                labels = agg_df[stage_col].astype(str).values
                max_val = values.max() if len(values) > 0 else 1
                colors_fn = sns.color_palette(palette, n_colors=len(values))
                for i, (lbl, val) in enumerate(zip(labels, values)):
                    width = val / max_val
                    left = (1 - width) / 2
                    ax.barh(len(values) - 1 - i, width, left=left, height=0.8, color=colors_fn[i], edgecolor='white', linewidth=1)
                    ax.text(0.5, len(values) - 1 - i, f"{lbl}\n{val:,.0f}", ha='center', va='center', fontsize=9, fontweight='bold')
                ax.set_yticks([])
                ax.set_xticks([])
                ax.set_xlim(0, 1)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['bottom'].set_visible(False)
                ax.spines['left'].set_visible(False)

            elif chart_type == "Diagrama Sunburst":
                l1_var = getattr(self, 'param_sunburst_level1_var', None)
                l2_var = getattr(self, 'param_sunburst_level2_var', None)
                sv_var = getattr(self, 'param_sunburst_value_var', None)
                l1_col = l1_var.get() if l1_var else None
                l2_col = l2_var.get() if l2_var else None
                sv_col = sv_var.get() if sv_var else None
                if not l1_col:
                    messagebox.showerror("Error", "Seleccione al menos el Nivel 1.", parent=self.parent_for_dialogs)
                    plt.close(plt_fig); return
                params_used_log.append(f"  Nivel 1: {l1_col}")
                cols_needed = [l1_col]
                if l2_col: cols_needed.append(l2_col)
                if sv_col: cols_needed.append(sv_col)
                plot_df = filtered_data[cols_needed].dropna()
                if sv_col:
                    plot_df[sv_col] = pd.to_numeric(plot_df[sv_col], errors='coerce')
                    plot_df = plot_df.dropna(subset=[sv_col])

                # Inner ring: level 1
                if sv_col:
                    inner = plot_df.groupby(l1_col)[sv_col].sum()
                else:
                    inner = plot_df[l1_col].value_counts()
                inner = inner[inner > 0]
                inner_colors = sns.color_palette(palette, n_colors=len(inner))

                plt.close(plt_fig)
                plt_fig, ax = plt.subplots(figsize=(fig_width, fig_height))
                if l2_col:
                    params_used_log.append(f"  Nivel 2: {l2_col}")
                    # Outer ring: level 2 within level 1
                    outer_sizes = []
                    outer_colors = []
                    outer_labels = []
                    for i, cat1 in enumerate(inner.index):
                        sub = plot_df[plot_df[l1_col] == cat1]
                        if sv_col:
                            outer = sub.groupby(l2_col)[sv_col].sum()
                        else:
                            outer = sub[l2_col].value_counts()
                        outer = outer[outer > 0]
                        base_color = inner_colors[i]
                        n_sub = len(outer)
                        for j, (cat2, val) in enumerate(outer.items()):
                            outer_sizes.append(val)
                            factor = 0.6 + 0.4 * (j / max(n_sub - 1, 1))
                            outer_colors.append(tuple(c * factor for c in base_color[:3]))
                            outer_labels.append(str(cat2))
                    ax.pie(outer_sizes, labels=outer_labels, colors=outer_colors, radius=1.0,
                           wedgeprops=dict(width=0.35, edgecolor='white'), labeldistance=1.05, textprops={'fontsize': 7})
                    ax.pie(inner.values, labels=inner.index.astype(str), colors=inner_colors, radius=0.65,
                           wedgeprops=dict(width=0.35, edgecolor='white'), labeldistance=0.7, textprops={'fontsize': 8, 'fontweight': 'bold'})
                else:
                    ax.pie(inner.values, labels=inner.index.astype(str), colors=inner_colors, radius=1.0,
                           wedgeprops=dict(width=0.5, edgecolor='white'), labeldistance=1.05, textprops={'fontsize': 9})
                ax.set_title(title if title else chart_type)

            elif chart_type == "Diagrama de Marimekko":
                x_var = getattr(self, 'param_marimekko_x_var', None)
                s_var = getattr(self, 'param_marimekko_stack_var', None)
                v_var = getattr(self, 'param_marimekko_value_var', None)
                x_col = x_var.get() if x_var else None
                s_col = s_var.get() if s_var else None
                v_col = v_var.get() if v_var else None
                if not x_col or not s_col or not v_col:
                    messagebox.showerror("Error", "Seleccione las 3 variables requeridas.", parent=self.parent_for_dialogs)
                    plt.close(plt_fig); return
                params_used_log.extend([f"  X: {x_col}", f"  Apilado: {s_col}", f"  Valor: {v_col}"])
                plot_df = filtered_data[[x_col, s_col, v_col]].dropna()
                plot_df[v_col] = pd.to_numeric(plot_df[v_col], errors='coerce')
                plot_df = plot_df.dropna(subset=[v_col])
                pivot = plot_df.groupby([x_col, s_col])[v_col].sum().unstack(fill_value=0)
                col_totals = pivot.sum(axis=0)
                widths = (col_totals / col_totals.sum()).values
                # Normalize each column to proportions
                pivot_norm = pivot.div(pivot.sum(axis=0), axis=1)
                stack_cats = pivot_norm.index.tolist()
                x_cats = pivot_norm.columns.tolist()
                colors_m = sns.color_palette(palette, n_colors=len(stack_cats))
                x_starts = np.concatenate([[0], np.cumsum(widths[:-1])])
                for i, scat in enumerate(stack_cats):
                    bottoms_m = np.zeros(len(x_cats))
                    for j in range(i):
                        bottoms_m += pivot_norm.iloc[j].values
                    heights = pivot_norm.iloc[i].values
                    for k in range(len(x_cats)):
                        ax.bar(x_starts[k], heights[k], width=widths[k], bottom=bottoms_m[k],
                               color=colors_m[i], edgecolor='white', linewidth=0.5, align='edge')
                ax.set_xticks(x_starts + widths / 2)
                ax.set_xticklabels(x_cats, rotation=tick_rotation if tick_rotation else 30, ha='right')
                ax.set_ylabel("Proporción")
                ax.set_xlim(0, sum(widths))
                ax.set_ylim(0, 1)
                handles_m = [plt.Rectangle((0,0),1,1, color=colors_m[i]) for i in range(len(stack_cats))]
                ax.legend(handles_m, stack_cats, title=s_col, loc='best')

            elif chart_type == "Dendrograma":
                dendro_vars_list = getattr(self, 'param_dendro_vars', [])
                selected = [v.get() for v in dendro_vars_list if v.get()]
                method_var = getattr(self, 'param_dendro_method_var', None)
                label_var = getattr(self, 'param_dendro_label_var', None)
                method = method_var.get() if method_var else 'ward'
                label_col = label_var.get() if label_var else None
                if len(selected) < 2:
                    messagebox.showerror("Error", "Seleccione al menos 2 variables numéricas.", parent=self.parent_for_dialogs)
                    plt.close(plt_fig); return
                params_used_log.extend([f"  Variables: {', '.join(selected)}", f"  Método: {method}"])
                plot_df = filtered_data[selected + ([label_col] if label_col else [])].dropna()
                for c in selected:
                    plot_df[c] = pd.to_numeric(plot_df[c], errors='coerce')
                plot_df = plot_df.dropna(subset=selected)
                if plot_df.empty or len(plot_df) < 2:
                    messagebox.showwarning("Sin Datos", "Se necesitan al menos 2 observaciones.", parent=self.parent_for_dialogs)
                    plt.close(plt_fig); return
                X_dendro = plot_df[selected].values
                Z = linkage(X_dendro, method=method)
                labels_d = plot_df[label_col].astype(str).values if label_col else None
                dendrogram(Z, labels=labels_d, ax=ax, leaf_rotation=tick_rotation if tick_rotation else 90, leaf_font_size=8)
                ax.set_ylabel("Distancia")

            elif chart_type == "Diagrama de Sankey":
                src_var = getattr(self, 'param_sankey_source_var', None)
                tgt_var = getattr(self, 'param_sankey_target_var', None)
                val_var = getattr(self, 'param_sankey_value_var', None)
                src_col = src_var.get() if src_var else None
                tgt_col = tgt_var.get() if tgt_var else None
                val_col = val_var.get() if val_var else None
                if not src_col or not tgt_col or not val_col:
                    messagebox.showerror("Error", "Seleccione Origen, Destino y Flujo.", parent=self.parent_for_dialogs)
                    plt.close(plt_fig); return
                params_used_log.extend([f"  Origen: {src_col}", f"  Destino: {tgt_col}", f"  Flujo: {val_col}"])
                plot_df = filtered_data[[src_col, tgt_col, val_col]].dropna()
                plot_df[val_col] = pd.to_numeric(plot_df[val_col], errors='coerce')
                plot_df = plot_df.dropna(subset=[val_col])
                agg_df = plot_df.groupby([src_col, tgt_col])[val_col].sum().reset_index()
                if agg_df.empty:
                    messagebox.showwarning("Sin Datos", "No hay flujos para graficar.", parent=self.parent_for_dialogs)
                    plt.close(plt_fig); return
                # Build node index
                all_nodes = list(pd.unique(agg_df[[src_col, tgt_col]].values.ravel()))
                node_idx = {n: i for i, n in enumerate(all_nodes)}
                sources = agg_df[src_col].map(node_idx).values
                targets = agg_df[tgt_col].map(node_idx).values
                values_sk = agg_df[val_col].values
                # Use manual Sankey via plotly-style rendering with matplotlib alluvial
                n_nodes = len(all_nodes)
                colors_sk = sns.color_palette(palette, n_colors=n_nodes)
                # Position nodes in 2 columns
                left_nodes = sorted(set(sources))
                right_nodes = sorted(set(targets))
                y_left = {n: i for i, n in enumerate(left_nodes)}
                y_right = {n: i for i, n in enumerate(right_nodes)}
                max_y = max(len(left_nodes), len(right_nodes))
                for n in left_nodes:
                    ax.barh(y_left[n], 0.1, left=0, height=0.6, color=colors_sk[n], edgecolor='white')
                    ax.text(-0.05, y_left[n], all_nodes[n], ha='right', va='center', fontsize=8)
                for n in right_nodes:
                    ax.barh(y_right[n], 0.1, left=0.9, height=0.6, color=colors_sk[n], edgecolor='white')
                    ax.text(1.05, y_right[n], all_nodes[n], ha='left', va='center', fontsize=8)
                # Draw flows as curved bands
                max_flow = values_sk.max() if len(values_sk) > 0 else 1
                for s, t, v in zip(sources, targets, values_sk):
                    lw = max(1, (v / max_flow) * 10)
                    x_pts = np.linspace(0.1, 0.9, 50)
                    y_pts = np.linspace(y_left[s], y_right[t], 50)
                    # Cubic bezier-like curve
                    t_param = np.linspace(0, 1, 50)
                    y_curve = y_left[s] + (y_right[t] - y_left[s]) * (3 * t_param**2 - 2 * t_param**3)
                    ax.plot(x_pts, y_curve, color=colors_sk[s], alpha=0.4, linewidth=lw)
                ax.set_xlim(-0.3, 1.3)
                ax.set_ylim(-0.5, max_y)
                ax.axis('off')

            elif chart_type == "Gráfico de Flujo":
                x_var = getattr(self, 'param_stream_x_var', None)
                y_vars_list = getattr(self, 'param_stream_y_vars', [])
                x_col = x_var.get() if x_var else None
                y_cols = [v.get() for v in y_vars_list if v.get()]
                if not x_col or len(y_cols) < 1:
                    messagebox.showerror("Error", "Seleccione la variable X y al menos 1 capa.", parent=self.parent_for_dialogs)
                    plt.close(plt_fig); return
                params_used_log.extend([f"  X: {x_col}", f"  Capas: {', '.join(y_cols)}"])
                plot_df = filtered_data[[x_col] + y_cols].dropna()
                for c in y_cols:
                    plot_df[c] = pd.to_numeric(plot_df[c], errors='coerce')
                plot_df = plot_df.dropna().sort_values(x_col)
                if plot_df.empty:
                    messagebox.showwarning("Sin Datos", "No hay datos válidos.", parent=self.parent_for_dialogs)
                    plt.close(plt_fig); return
                x_vals = range(len(plot_df))
                y_matrix = [plot_df[c].values for c in y_cols]
                colors_sf = sns.color_palette(palette, n_colors=len(y_cols))
                ax.stackplot(x_vals, *y_matrix, labels=y_cols, colors=colors_sf, alpha=0.8)
                # Set x ticks to original x values
                tick_step = max(1, len(plot_df) // 10)
                ax.set_xticks(list(x_vals)[::tick_step])
                ax.set_xticklabels(plot_df[x_col].astype(str).values[::tick_step],
                                   rotation=tick_rotation if tick_rotation else 45, ha='right')
                ax.set_xlabel(x_col)
                ax.legend(loc='best')

            elif chart_type == "Diagrama de Gantt":
                task_var = getattr(self, 'param_gantt_task_var', None)
                start_var = getattr(self, 'param_gantt_start_var', None)
                end_var = getattr(self, 'param_gantt_end_var', None)
                group_var = getattr(self, 'param_gantt_group_var', None)
                task_col = task_var.get() if task_var else None
                start_col = start_var.get() if start_var else None
                end_col = end_var.get() if end_var else None
                group_col = group_var.get() if group_var else None
                if not task_col or not start_col or not end_col:
                    messagebox.showerror("Error", "Seleccione Tarea, Inicio y Fin.", parent=self.parent_for_dialogs)
                    plt.close(plt_fig); return
                params_used_log.extend([f"  Tarea: {task_col}", f"  Inicio: {start_col}", f"  Fin: {end_col}"])
                cols_g = [task_col, start_col, end_col] + ([group_col] if group_col else [])
                plot_df = filtered_data[cols_g].dropna()
                # Try to parse as dates
                for dc in [start_col, end_col]:
                    try:
                        plot_df[dc] = pd.to_datetime(plot_df[dc])
                    except Exception:
                        plot_df[dc] = pd.to_numeric(plot_df[dc], errors='coerce')
                plot_df = plot_df.dropna(subset=[start_col, end_col])
                if plot_df.empty:
                    messagebox.showwarning("Sin Datos", "No hay datos válidos.", parent=self.parent_for_dialogs)
                    plt.close(plt_fig); return
                tasks = plot_df[task_col].astype(str).values
                starts = plot_df[start_col].values
                ends = plot_df[end_col].values
                durations = ends - starts
                if group_col:
                    groups = plot_df[group_col].astype(str).values
                    unique_groups = sorted(set(groups))
                    colors_g = sns.color_palette(palette, n_colors=len(unique_groups))
                    color_map_g = {g: colors_g[i] for i, g in enumerate(unique_groups)}
                    bar_colors = [color_map_g[g] for g in groups]
                else:
                    bar_colors = sns.color_palette(palette, n_colors=len(tasks))
                for i in range(len(tasks)):
                    ax.barh(i, durations[i], left=starts[i], height=0.6, color=bar_colors[i], edgecolor='white', linewidth=0.5)
                ax.set_yticks(range(len(tasks)))
                ax.set_yticklabels(tasks)
                ax.invert_yaxis()
                ax.set_xlabel("Tiempo")
                if group_col:
                    handles_g = [plt.Rectangle((0,0),1,1, color=color_map_g[g]) for g in unique_groups]
                    ax.legend(handles_g, unique_groups, title=group_col, loc='best')
                # Format x-axis for dates if applicable
                if pd.api.types.is_datetime64_any_dtype(plot_df[start_col]):
                    plt_fig.autofmt_xdate()

            elif chart_type == "Gráfico de Velas":
                date_var = getattr(self, 'param_candle_date_var', None)
                open_var = getattr(self, 'param_candle_open_var', None)
                high_var = getattr(self, 'param_candle_high_var', None)
                low_var = getattr(self, 'param_candle_low_var', None)
                close_var = getattr(self, 'param_candle_close_var', None)
                date_col = date_var.get() if date_var else None
                open_col = open_var.get() if open_var else None
                high_col = high_var.get() if high_var else None
                low_col = low_var.get() if low_var else None
                close_col = close_var.get() if close_var else None
                if not all([date_col, open_col, high_col, low_col, close_col]):
                    messagebox.showerror("Error", "Seleccione las 5 variables OHLC + Fecha.", parent=self.parent_for_dialogs)
                    plt.close(plt_fig); return
                params_used_log.extend([f"  Fecha: {date_col}", f"  OHLC: {open_col}, {high_col}, {low_col}, {close_col}"])
                ohlc_cols = [open_col, high_col, low_col, close_col]
                plot_df = filtered_data[[date_col] + ohlc_cols].dropna()
                for c in ohlc_cols:
                    plot_df[c] = pd.to_numeric(plot_df[c], errors='coerce')
                plot_df = plot_df.dropna()
                try:
                    plot_df[date_col] = pd.to_datetime(plot_df[date_col])
                    plot_df = plot_df.sort_values(date_col)
                except Exception:
                    pass
                if plot_df.empty:
                    messagebox.showwarning("Sin Datos", "No hay datos OHLC válidos.", parent=self.parent_for_dialogs)
                    plt.close(plt_fig); return
                for i, (_, row) in enumerate(plot_df.iterrows()):
                    o, h, l, c = row[open_col], row[high_col], row[low_col], row[close_col]
                    color = '#2ecc71' if c >= o else '#e74c3c'
                    # High-low line (wick)
                    ax.plot([i, i], [l, h], color=color, linewidth=0.8)
                    # Body
                    body_bottom = min(o, c)
                    body_height = abs(c - o)
                    ax.add_patch(Rectangle((i - 0.3, body_bottom), 0.6, body_height if body_height > 0 else 0.01,
                                           facecolor=color, edgecolor=color, linewidth=0.5))
                tick_step = max(1, len(plot_df) // 10)
                ax.set_xticks(list(range(len(plot_df)))[::tick_step])
                date_labels = plot_df[date_col].astype(str).values[::tick_step]
                ax.set_xticklabels(date_labels, rotation=tick_rotation if tick_rotation else 45, ha='right')
                ax.set_ylabel("Precio")
                ax.set_xlabel(date_col)

            elif chart_type == "Línea de Tiempo":
                date_var = getattr(self, 'param_timeline_date_var', None)
                event_var = getattr(self, 'param_timeline_event_var', None)
                group_var = getattr(self, 'param_timeline_group_var', None)
                date_col = date_var.get() if date_var else None
                event_col = event_var.get() if event_var else None
                group_col = group_var.get() if group_var else None
                if not date_col or not event_col:
                    messagebox.showerror("Error", "Seleccione Fecha y Evento.", parent=self.parent_for_dialogs)
                    plt.close(plt_fig); return
                params_used_log.extend([f"  Fecha: {date_col}", f"  Evento: {event_col}"])
                cols_tl = [date_col, event_col] + ([group_col] if group_col else [])
                plot_df = filtered_data[cols_tl].dropna()
                try:
                    plot_df[date_col] = pd.to_datetime(plot_df[date_col])
                except Exception:
                    plot_df[date_col] = pd.to_numeric(plot_df[date_col], errors='coerce')
                plot_df = plot_df.dropna(subset=[date_col]).sort_values(date_col)
                if plot_df.empty:
                    messagebox.showwarning("Sin Datos", "No hay datos válidos.", parent=self.parent_for_dialogs)
                    plt.close(plt_fig); return
                dates = plot_df[date_col].values
                events = plot_df[event_col].astype(str).values
                if group_col:
                    groups = plot_df[group_col].astype(str).values
                    unique_groups = sorted(set(groups))
                    colors_tl = sns.color_palette(palette, n_colors=len(unique_groups))
                    color_map_tl = {g: colors_tl[i] for i, g in enumerate(unique_groups)}
                else:
                    color_map_tl = {}
                # Alternating heights for readability
                levels = []
                for i in range(len(events)):
                    levels.append((-1)**i * (1 + i % 3))
                ax.axhline(0, color='gray', linewidth=0.8)
                for i in range(len(events)):
                    color = color_map_tl.get(groups[i], 'steelblue') if group_col else 'steelblue'
                    ax.plot([dates[i], dates[i]], [0, levels[i]], color=color, linewidth=1, alpha=0.7)
                    ax.scatter(dates[i], levels[i], color=color, s=40, zorder=5, edgecolors='white')
                    ax.text(dates[i], levels[i] + 0.2 * np.sign(levels[i]), events[i],
                            ha='center', va='bottom' if levels[i] > 0 else 'top', fontsize=7, rotation=15)
                ax.set_xlabel(date_col)
                ax.set_yticks([])
                if group_col:
                    handles_tl = [plt.Line2D([0],[0], marker='o', color='w', markerfacecolor=color_map_tl[g], markersize=8, label=g) for g in unique_groups]
                    ax.legend(handles=handles_tl, title=group_col, loc='best')
                if pd.api.types.is_datetime64_any_dtype(plot_df[date_col]):
                    plt_fig.autofmt_xdate()

            elif chart_type == "Mapa Coroplético":
                region_var = getattr(self, 'param_choro_region_var', None)
                val_var = getattr(self, 'param_choro_value_var', None)
                rtype_var = getattr(self, 'param_choro_region_type_var', None)
                region_col = region_var.get() if region_var else None
                val_col = val_var.get() if val_var else None
                rtype = rtype_var.get() if rtype_var else "País"
                if not region_col or not val_col:
                    messagebox.showerror("Error", "Seleccione Región y Variable de Valor.", parent=self.parent_for_dialogs)
                    plt.close(plt_fig); return
                params_used_log.extend([f"  Región: {region_col}", f"  Valor: {val_col}", f"  Tipo: {rtype}"])
                plot_df = filtered_data[[region_col, val_col]].dropna()
                plot_df[val_col] = pd.to_numeric(plot_df[val_col], errors='coerce')
                plot_df = plot_df.dropna(subset=[val_col])
                agg_df = plot_df.groupby(region_col)[val_col].mean().reset_index()
                try:
                    import geopandas as gpd
                    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres') if hasattr(gpd.datasets, 'get_path') else gpd.datasets.get_path('naturalearth_lowres'))
                    name_col_geo = 'name' if 'name' in world.columns else world.columns[0]
                    merged = world.merge(agg_df, left_on=name_col_geo, right_on=region_col, how='left')
                    merged.plot(column=val_col, ax=ax, legend=True,
                                cmap=palette if palette and isinstance(palette, str) else 'YlOrRd',
                                missing_kwds={"color": "lightgrey"}, edgecolor='white', linewidth=0.3)
                    ax.set_title(title if title else f"{val_col} por {region_col}")
                    ax.axis('off')
                except Exception as e:
                    ax.text(0.5, 0.5, f"Error al generar mapa: {e}", ha='center', va='center', fontsize=9, wrap=True)

            elif chart_type == "Mapa de Burbujas":
                lat_var = getattr(self, 'param_bubble_map_lat_var', None)
                lon_var = getattr(self, 'param_bubble_map_lon_var', None)
                size_var = getattr(self, 'param_bubble_map_size_var', None)
                label_var = getattr(self, 'param_bubble_map_label_var', None)
                lat_col = lat_var.get() if lat_var else None
                lon_col = lon_var.get() if lon_var else None
                size_col = size_var.get() if size_var else None
                label_col = label_var.get() if label_var else None
                if not lat_col or not lon_col or not size_col:
                    messagebox.showerror("Error", "Seleccione Latitud, Longitud y Tamaño.", parent=self.parent_for_dialogs)
                    plt.close(plt_fig); return
                params_used_log.extend([f"  Lat: {lat_col}", f"  Lon: {lon_col}", f"  Tamaño: {size_col}"])
                cols_bm = [lat_col, lon_col, size_col] + ([label_col] if label_col else [])
                plot_df = filtered_data[cols_bm].dropna()
                for c in [lat_col, lon_col, size_col]:
                    plot_df[c] = pd.to_numeric(plot_df[c], errors='coerce')
                plot_df = plot_df.dropna(subset=[lat_col, lon_col, size_col])
                if plot_df.empty:
                    messagebox.showwarning("Sin Datos", "No hay datos válidos.", parent=self.parent_for_dialogs)
                    plt.close(plt_fig); return
                try:
                    import geopandas as gpd
                    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres') if hasattr(gpd.datasets, 'get_path') else gpd.datasets.get_path('naturalearth_lowres'))
                    world.plot(ax=ax, color='lightgrey', edgecolor='white', linewidth=0.3)
                except Exception:
                    pass
                sizes = plot_df[size_col].values
                sizes_norm = (sizes - sizes.min()) / (sizes.max() - sizes.min() + 1e-9) * 500 + 20
                ax.scatter(plot_df[lon_col], plot_df[lat_col], s=sizes_norm, alpha=0.6, c='steelblue', edgecolors='black', linewidths=0.5, zorder=5)
                if label_col:
                    for _, row in plot_df.iterrows():
                        ax.text(row[lon_col], row[lat_col], str(row[label_col]), fontsize=6, ha='center', va='bottom')
                ax.set_xlabel("Longitud")
                ax.set_ylabel("Latitud")

            elif chart_type == "Mapa de Puntos":
                lat_var = getattr(self, 'param_dotmap_lat_var', None)
                lon_var = getattr(self, 'param_dotmap_lon_var', None)
                color_var = getattr(self, 'param_dotmap_color_var', None)
                lat_col = lat_var.get() if lat_var else None
                lon_col = lon_var.get() if lon_var else None
                color_col = color_var.get() if color_var else None
                if not lat_col or not lon_col:
                    messagebox.showerror("Error", "Seleccione Latitud y Longitud.", parent=self.parent_for_dialogs)
                    plt.close(plt_fig); return
                params_used_log.extend([f"  Lat: {lat_col}", f"  Lon: {lon_col}"])
                cols_dm = [lat_col, lon_col] + ([color_col] if color_col else [])
                plot_df = filtered_data[cols_dm].dropna()
                for c in [lat_col, lon_col]:
                    plot_df[c] = pd.to_numeric(plot_df[c], errors='coerce')
                plot_df = plot_df.dropna(subset=[lat_col, lon_col])
                if plot_df.empty:
                    messagebox.showwarning("Sin Datos", "No hay datos válidos.", parent=self.parent_for_dialogs)
                    plt.close(plt_fig); return
                try:
                    import geopandas as gpd
                    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres') if hasattr(gpd.datasets, 'get_path') else gpd.datasets.get_path('naturalearth_lowres'))
                    world.plot(ax=ax, color='lightgrey', edgecolor='white', linewidth=0.3)
                except Exception:
                    pass
                if color_col:
                    params_used_log.append(f"  Color: {color_col}")
                    groups = sorted(plot_df[color_col].unique())
                    colors_dm = sns.color_palette(palette, n_colors=len(groups))
                    for i, g in enumerate(groups):
                        sub = plot_df[plot_df[color_col] == g]
                        ax.scatter(sub[lon_col], sub[lat_col], s=point_size * 5, c=[colors_dm[i]], label=str(g), alpha=0.7, edgecolors='black', linewidths=0.3, zorder=5)
                    ax.legend(title=color_col, loc='best')
                else:
                    ax.scatter(plot_df[lon_col], plot_df[lat_col], s=point_size * 5, c='steelblue', alpha=0.7, edgecolors='black', linewidths=0.3, zorder=5)
                ax.set_xlabel("Longitud")
                ax.set_ylabel("Latitud")

            elif chart_type == "Diagrama de Venn":
                group_var = getattr(self, 'param_venn_group_var', None)
                elem_var = getattr(self, 'param_venn_element_var', None)
                group_col = group_var.get() if group_var else None
                elem_col = elem_var.get() if elem_var else None
                if not group_col or not elem_col:
                    messagebox.showerror("Error", "Seleccione Variable de Grupo y Variable de Elemento.", parent=self.parent_for_dialogs)
                    plt.close(plt_fig); return
                params_used_log.extend([f"  Grupo: {group_col}", f"  Elemento: {elem_col}"])
                plot_df = filtered_data[[group_col, elem_col]].dropna()
                groups = sorted(plot_df[group_col].unique())
                if len(groups) < 2 or len(groups) > 3:
                    messagebox.showerror("Error", f"Se necesitan 2 o 3 grupos, se encontraron {len(groups)}.", parent=self.parent_for_dialogs)
                    plt.close(plt_fig); return
                sets = []
                for g in groups:
                    elements = set(plot_df[plot_df[group_col] == g][elem_col].astype(str))
                    sets.append(elements)
                plt.close(plt_fig)
                plt_fig, ax = plt.subplots(figsize=(fig_width, fig_height))
                if len(groups) == 2:
                    venn2(sets, set_labels=groups, ax=ax)
                else:
                    venn3(sets, set_labels=groups, ax=ax)
                ax.set_title(title if title else f"Diagrama de Venn — {group_col}")

            else:
                ax.text(0.5, 0.5, f"Gráfico '{chart_type}' aún no implementado.", ha='center', va='center')

            # --- Apply Specific Customizations to Axes ---
            try:
                title_color = getattr(self, 'param_title_color_var', StringVar(value='black')).get()
            except Exception:
                title_color = 'black'
            try:
                title_size = int(getattr(self, 'param_title_size_var', StringVar(value='12')).get())
            except Exception:
                title_size = None
            shared_configure_plot_style(
                ax,
                title=title if title else chart_type,
                xlabel=xlabel,
                ylabel=ylabel,
                font_size=title_size if title_size else 10,
                font_color=title_color,
                grid=grid,
            )

            if title_size:
                title_kwargs = {'color': title_color, 'fontsize': title_size}
                if font_family_sel and font_family_sel != 'Default':
                    title_kwargs['fontfamily'] = font_family_sel
                ax.set_title(title if title else chart_type, **title_kwargs)
            else:
                title_kwargs = {'color': title_color}
                if font_family_sel and font_family_sel != 'Default':
                    title_kwargs['fontfamily'] = font_family_sel
                ax.set_title(title if title else chart_type, **title_kwargs)
            
            # Aplicar fuente a etiquetas de ejes
            xlabel_kwargs = {}
            ylabel_kwargs = {}
            if font_family_sel and font_family_sel != 'Default':
                xlabel_kwargs['fontfamily'] = font_family_sel
                ylabel_kwargs['fontfamily'] = font_family_sel
            ax.set_xlabel(xlabel, **xlabel_kwargs)
            ax.set_ylabel(ylabel, **ylabel_kwargs)
            
            # Aplicar fuente a los ticks
            if font_family_sel and font_family_sel != 'Default':
                for label in ax.get_xticklabels():
                    label.set_fontfamily(font_family_sel)
                for label in ax.get_yticklabels():
                    label.set_fontfamily(font_family_sel)
            if grid:
                ax.grid(True, linestyle='--', alpha=0.7)
            else:
                ax.grid(False)

            # Apply new advanced options
            ax.set_xscale(x_scale)
            ax.set_yscale(y_scale)
            x_fmt = getattr(self, 'param_x_format_var', StringVar(value='auto')).get() if hasattr(self, 'param_x_format_var') else 'auto'
            y_fmt = getattr(self, 'param_y_format_var', StringVar(value='auto')).get() if hasattr(self, 'param_y_format_var') else 'auto'
            use_x_sci = (x_fmt == 'científica')
            use_x_norm = (x_fmt == 'normal')
            use_y_sci = (y_fmt == 'científica')
            use_y_norm = (y_fmt == 'normal')
            plt.setp(ax.get_xticklabels(), rotation=tick_rotation)
            plt.setp(ax.get_yticklabels(), rotation=y_tick_rotation)

            # Límites y ticks personalizados (permite min,max o lista de ticks)
            try:
                x_limits, x_ticks = self._parse_limits_and_ticks(xlim_raw)
                if x_limits is not None:
                    shared_apply_axis_limits(ax, x_limits[0], x_limits[1], None, None)
                if x_ticks is not None and len(x_ticks) > 0:
                    x_ticks = [t for t in x_ticks if (t > 0 if x_scale == 'log' else True)]
                    if x_ticks:
                        locator = mticker.FixedLocator(x_ticks)
                        if x_scale == 'log':
                            if use_x_norm:
                                formatter = mticker.FuncFormatter(lambda v, pos: f"{v:g}")
                            else:  # auto or científica -> científica por legibilidad en log
                                formatter = mticker.LogFormatterMathtext(base=10)
                            ax.xaxis.set_minor_locator(mticker.NullLocator())
                            ax.xaxis.set_minor_formatter(mticker.NullFormatter())
                        else:
                            formatter = mticker.ScalarFormatter(useMathText=True)
                            if use_x_sci:
                                formatter.set_powerlimits((0, 0))
                                formatter.set_scientific(True)
                            else:
                                formatter.set_scientific(False)
                            formatter.set_useOffset(False)
                        ax.xaxis.set_major_locator(locator)
                        ax.xaxis.set_major_formatter(formatter)
                else:
                    if x_scale == 'log':
                        ax.xaxis.set_minor_locator(mticker.NullLocator())
                        ax.xaxis.set_minor_formatter(mticker.NullFormatter())
                        ax.xaxis.set_major_locator(mticker.LogLocator(base=10))
                        if use_x_norm:
                            ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, pos: f"{v:g}"))
                        else:
                            ax.xaxis.set_major_formatter(mticker.LogFormatterMathtext(base=10))
                    elif use_x_sci:
                        formatter = mticker.ScalarFormatter(useMathText=True)
                        formatter.set_powerlimits((0, 0))
                        formatter.set_scientific(True)
                        formatter.set_useOffset(False)
                        ax.xaxis.set_major_formatter(formatter)
                    elif use_x_norm:
                        formatter = mticker.ScalarFormatter(useMathText=True)
                        formatter.set_scientific(False)
                        formatter.set_useOffset(False)
                        ax.xaxis.set_major_formatter(formatter)
            except Exception as exc:
                self.log(f"Formato de xlim/ticks inválido: {exc}", "WARN")

            try:
                y_limits, y_ticks = self._parse_limits_and_ticks(ylim_raw)
                if y_limits is not None:
                    shared_apply_axis_limits(ax, None, None, y_limits[0], y_limits[1])
                if y_ticks is not None and len(y_ticks) > 0:
                    y_ticks = [t for t in y_ticks if (t > 0 if y_scale == 'log' else True)]
                    if y_ticks:
                        locator = mticker.FixedLocator(y_ticks)
                        if y_scale == 'log':
                            if use_y_norm:
                                formatter = mticker.FuncFormatter(lambda v, pos: f"{v:g}")
                            else:
                                formatter = mticker.LogFormatterMathtext(base=10)
                            ax.yaxis.set_minor_locator(mticker.NullLocator())
                            ax.yaxis.set_minor_formatter(mticker.NullFormatter())
                        else:
                            formatter = mticker.ScalarFormatter(useMathText=True)
                            if use_y_sci:
                                formatter.set_powerlimits((0, 0))
                                formatter.set_scientific(True)
                            else:
                                formatter.set_scientific(False)
                            formatter.set_useOffset(False)
                        ax.yaxis.set_major_locator(locator)
                        ax.yaxis.set_major_formatter(formatter)
                else:
                    if y_scale == 'log':
                        ax.yaxis.set_minor_locator(mticker.NullLocator())
                        ax.yaxis.set_minor_formatter(mticker.NullFormatter())
                        ax.yaxis.set_major_locator(mticker.LogLocator(base=10))
                        if use_y_norm:
                            ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, pos: f"{v:g}"))
                        else:
                            ax.yaxis.set_major_formatter(mticker.LogFormatterMathtext(base=10))
                    elif use_y_sci:
                        formatter = mticker.ScalarFormatter(useMathText=True)
                        formatter.set_powerlimits((0, 0))
                        formatter.set_scientific(True)
                        formatter.set_useOffset(False)
                        ax.yaxis.set_major_formatter(formatter)
                    elif use_y_norm:
                        formatter = mticker.ScalarFormatter(useMathText=True)
                        formatter.set_scientific(False)
                        formatter.set_useOffset(False)
                        ax.yaxis.set_major_formatter(formatter)
            except Exception as exc:
                self.log(f"Formato de ylim/ticks inválido: {exc}", "WARN")

            # --- Aplicar cortes de eje (Axis Breaks) ---
            try:
                y_breaks_raw = getattr(self, 'param_y_breaks_var', StringVar(value='')).get()
                x_breaks_raw = getattr(self, 'param_x_breaks_var', StringVar(value='')).get()
                
                y_breaks = self._parse_axis_breaks(y_breaks_raw) if y_breaks_raw else []
                x_breaks = self._parse_axis_breaks(x_breaks_raw) if x_breaks_raw else []
                
                if y_breaks or x_breaks:
                    # Obtener parámetros de estilo
                    break_size_val = float(getattr(self, 'param_break_size_var', StringVar(value='1.5')).get()) * 0.01
                    break_lw_val = float(getattr(self, 'param_break_lw_var', StringVar(value='1.0')).get())
                    
                    # Obtener las etiquetas de categoría del gráfico de barras
                    # category_order puede no existir si no es gráfico de barras
                    try:
                        cat_labels = category_order if 'category_order' in dir() and category_order else None
                    except NameError:
                        cat_labels = None
                    
                    self._apply_axis_breaks(plt_fig, ax, y_breaks=y_breaks, x_breaks=x_breaks,
                                           break_size=break_size_val, line_width=break_lw_val,
                                           category_labels=cat_labels)
                    if y_breaks:
                        self.log(f"Aplicados {len(y_breaks)} corte(s) en eje Y: {y_breaks}", "INFO")
                    if x_breaks:
                        self.log(f"Aplicados {len(x_breaks)} corte(s) en eje X: {x_breaks}", "INFO")
            except Exception as exc:
                self.log(f"Error aplicando cortes de eje: {exc}", "WARN")

            # Ajustar leyenda según preferencia del usuario
            try:
                legend = ax.get_legend()
                if not show_legend:
                    if legend:
                        legend.remove()
                else:
                    handles, labels = ax.get_legend_handles_labels()
                    if handles and labels:
                        if legend:
                            legend.remove()
                        legend_kwargs = {'loc': legend_loc or 'best', 'fontsize': legend_size}
                        if font_family_sel and font_family_sel != 'Default':
                            legend_kwargs['prop'] = {'family': font_family_sel, 'size': legend_size}
                        legend = ax.legend(handles, labels, **legend_kwargs)
                    elif legend:
                        try:
                            legend.set_loc(legend_loc or 'best')
                            for text in legend.get_texts():
                                text.set_fontsize(legend_size)
                                if font_family_sel and font_family_sel != 'Default':
                                    text.set_fontfamily(font_family_sel)
                        except Exception:
                            pass
            except Exception:
                pass

            # --- Set categorical tick labels for distribution plots (after axis scaling) ---
            if chart_type == "Gráfico de Distribución":
                if 'x_order' in locals() and x_order:
                    try:
                        locator = mticker.FixedLocator(range(len(x_order)))
                        formatter = mticker.FixedFormatter(x_order)
                        ax.xaxis.set_major_locator(locator)
                        ax.xaxis.set_major_formatter(formatter)
                        self.log(f"Locator y formatter personalizados aplicados al eje X con orden: {x_order}", "DEBUG")
                        tick_texts = [tick.get_text() for tick in ax.get_xticklabels()]
                        self.log(f"Ticklabels actuales tras aplicar formatter: {tick_texts}", "DEBUG")
                        self.log(f"Labels del eje X establecidos: {x_order}", "INFO")
                    except Exception as e:
                        self.log(f"Error al establecer labels del eje X: {e}", "WARN")
                        self.log(traceback.format_exc(), "DEBUG")

            # Reapply categorical tick overrides after scale/lim adjustments (skip forest plot because it uses custom ticks)
            if chart_type != "Forest Plot (Comparaciones)":
                if axis_x_col:
                    data_for_x = tick_data_x if tick_data_x is not None else filtered_data
                    self._set_categorical_tick_labels(ax, data_for_x, x_col=axis_x_col)
                if axis_y_col:
                    data_for_y = tick_data_y if tick_data_y is not None else filtered_data
                    self._set_categorical_tick_labels(ax, data_for_y, y_col=axis_y_col)

            plt.tight_layout()
            self.last_fig = plt_fig
            canvas = FigureCanvasTkAgg(plt_fig, master=self.chart_display_frame)
            canvas.draw()
            toolbar = NavigationToolbar2Tk(canvas, self.chart_display_frame)
            toolbar.update()
            canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

            self.log(f"'{chart_type}' generado con éxito.", "SUCCESS")
            self.log("Parámetros usados:\n" + "\n".join(params_used_log), "PARAMS")

        except Exception as e:
            self.log(f"Error generando gráfico '{chart_type}': {e}", "ERROR")
            messagebox.showerror("Error de Graficación", f"No se pudo generar el gráfico:\n{e}", parent=self.parent_for_dialogs)

if __name__ == '__main__':
    root = tk.Tk()
    root.title("Prueba Pestaña Gráficas Generales")
    root.geometry("1000x700")
    
    app_tab = GeneralChartsApp(root)
    app_tab.pack(fill="both", expand=True)
    root.mainloop()