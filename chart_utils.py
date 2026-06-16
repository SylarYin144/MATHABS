#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""chart_utils.py - Utilidades compartidas para graficas en MATHABS."""

import pandas as pd
import numpy as np
import matplotlib.colors as mcolors


def color_to_hex(color):
    """Convierte cualquier color valido de matplotlib a formato hexadecimal."""
    if not color:
        return "#000000"
    try:
        rgba = mcolors.to_rgba(color)
        return mcolors.to_hex(rgba)
    except Exception:
        return str(color)


def apply_axis_limits(ax, xmin=None, xmax=None, ymin=None, ymax=None):
    """Aplica limites de ejes X e Y al axes dado. Los valores None se ignoran."""
    if xmin is not None or xmax is not None:
        cur_xmin, cur_xmax = ax.get_xlim()
        ax.set_xlim(xmin if xmin is not None else cur_xmin,
                    xmax if xmax is not None else cur_xmax)
    if ymin is not None or ymax is not None:
        cur_ymin, cur_ymax = ax.get_ylim()
        ax.set_ylim(ymin if ymin is not None else cur_ymin,
                    ymax if ymax is not None else cur_ymax)


def parse_label_mapping(recode_str):
    """Parsea cadena tipo "1:Leve, 2:Moderado" -> (order_list, mapping_dict)."""
    order = []
    mapping = {}
    if not recode_str:
        return order, mapping
    for pair in recode_str.split(","):
        pair = pair.strip()
        if ":" not in pair:
            continue
        left, _, right = pair.partition(":")
        key = left.strip()
        label = right.strip()
        if not key or not label:
            continue
        if label not in order:
            order.append(label)
        mapping[key] = label
    return order, mapping


def apply_label_mapping_to_dataframe(df, column, mapping):
    """Aplica un mapping {clave -> etiqueta} a la columna indicada del DataFrame."""
    if not mapping or column not in df.columns:
        return df
    df = df.copy()
    str_mapping = {str(k): v for k, v in mapping.items()}
    df[column] = df[column].map(lambda v: str_mapping.get(str(v), v))
    return df


def build_category_palette(categories, palette_name="deep", explicit_map=None,
                            fallback_color="#888888"):
    """Construye {categoria: color_hex} para una lista de categorias."""
    explicit_map = explicit_map or {}
    result = {}
    try:
        import seaborn as sns
        palette_colors = sns.color_palette(palette_name, n_colors=max(len(categories), 1))
        palette_hex = [mcolors.to_hex(c) for c in palette_colors]
    except Exception:
        palette_hex = [
            "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
            "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
        ]
    idx = 0
    for cat in categories:
        cat_str = str(cat)
        if cat_str in explicit_map and explicit_map[cat_str]:
            try:
                result[cat_str] = color_to_hex(explicit_map[cat_str])
            except Exception:
                result[cat_str] = str(explicit_map[cat_str])
        else:
            result[cat_str] = palette_hex[idx % len(palette_hex)] if palette_hex else fallback_color
            idx += 1
    return result


def configure_plot_style(ax, title=None, xlabel=None, ylabel=None,
                          font_size=10, font_color=None, grid=False,
                          fontfamily=None):
    """Aplica estilo general: titulo, etiquetas, cuadricula y fuente."""
    extra = {}
    if fontfamily:
        extra["fontfamily"] = fontfamily
    if title:
        color_kw = {"color": font_color} if font_color else {}
        ax.set_title(title, fontsize=font_size, **color_kw, **extra)
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=max(font_size - 1, 8), **extra)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=max(font_size - 1, 8), **extra)
    ax.grid(True, linestyle="--", alpha=0.5) if grid else ax.grid(False)
