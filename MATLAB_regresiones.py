#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import io, base64
import textwrap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk # Importar FigureCanvasTkAgg
from matplotlib.ticker import LogFormatterSciNotation, LogFormatterExponent, ScalarFormatter # Para formato de ejes log
from matplotlib.lines import Line2D # Para leyendas personalizadas
import warnings
import scipy.stats as stats
from scipy.optimize import curve_fit, fsolve
import statsmodels.api as sm
from statsmodels.nonparametric.smoothers_lowess import lowess

# -------------------------------
# IMPORTS PARA FLASK (versión web)
# -------------------------------
from flask import Flask, request, render_template_string, redirect, url_for, session, flash
from werkzeug.utils import secure_filename

# -------------------------------
# IMPORTS PARA TKINTER (versión desktop)
# -------------------------------
import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext, messagebox # Añadir messagebox
import shutil
import traceback # Añadido para logging
from patsy import dmatrix

# Importar el componente de filtro
try:
    from MATLAB_filter_component import FilterComponent
except ImportError:
    messagebox.showerror("Error de Importación", "No se pudo importar FilterComponent.")
    FilterComponent = None

# ==============================
# Código de la aplicación WEB (Flask)
# ==============================
def run_flask_app():
    app = Flask(__name__)
    app.secret_key = 'tu_clave_secreta'  # Cambia esto por una clave segura

    # Configuración de carpeta de subida
    UPLOAD_FOLDER = 'uploads'
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

    @app.route('/', methods=['GET', 'POST'])
    def upload_file():
        if request.method == 'POST':
            if 'excel_file' not in request.files:
                flash("No se encontró el archivo")
                return redirect(request.url)
            file = request.files['excel_file']
            if file.filename == '':
                flash("No se seleccionó ningún archivo")
                return redirect(request.url)
            if file:
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                session['filepath'] = filepath
                return redirect(url_for('configure'))
        return render_template_string('''
        <!doctype html>
        <html>
          <head>
            <title>Cargar Archivo Excel</title>
          </head>
          <body>
            <h1>Cargar Archivo Excel</h1>
            <form method="post" enctype="multipart/form-data">
              <input type="file" name="excel_file">
              <input type="submit" value="Subir">
            </form>
          </body>
        </html>
        ''')

    @app.route('/configure', methods=['GET'])
    def configure():
        filepath = session.get('filepath', None)
        if not filepath or not os.path.exists(filepath):
            return redirect(url_for('upload_file'))
        try:
            df = pd.read_excel(filepath)
        except Exception as e:
            return "Error al leer el archivo: " + str(e)
        summary_text = summarize_data(df)
        columns = list(df.columns)
        return render_template_string('''
        <!doctype html>
        <html>
          <head>
            <title>Configuración de Filtros</title>
            <style>
              .container { display: flex; }
              .left { flex: 1; padding: 10px; }
              .right { flex: 2; padding: 10px; border-left: 1px solid #ccc; }
            </style>
          </head>
          <body>
            <h1>Resumen del Archivo Subido</h1>
            <pre>{{ summary_text }}</pre>
            <hr>
            <h2>Configura los Filtros y Parámetros de Gráfica</h2>
            <form method="post" action="{{ url_for('process') }}">
              <label>Variable de Caso:</label>
              <select name="var">
                {% for col in columns %}
                  <option value="{{ col }}">{{ col }}</option>
                {% endfor %}
              </select><br><br>
              
              <label>Filtro (coma para múltiples o guion para rango):</label>
              <input type="text" name="filter"><br><br>
              
              <label>Tipo de Variable:</label>
              <select name="tipo">
                <option value="Cuantitativa">Cuantitativa</option>
                <option value="Cualitativa">Cualitativa</option>
              </select><br><br>
              
              <label>Etiquetas Personalizadas:</label>
              <input type="text" name="etiquetas" placeholder="Ej: 1:Bajo,2:Medio,3:Alto"><br>
              <small>(Formato: valor:etiqueta, separados por coma)</small><br><br>
              
              <label>Excluir casillas en blanco en estadísticas:</label>
              <input type="checkbox" name="exclude_blank"><br><br>
              
              <h3>Filtros Adicionales (Opcionales)</h3>
              <label>Variable 2:</label>
              <select name="var2">
                <option value="">--Ninguno--</option>
                {% for col in columns %}
                  <option value="{{ col }}">{{ col }}</option>
                {% endfor %}
              </select>
              <label>Filtro 2:</label>
              <input type="text" name="filter2"><br><br>
              
              <label>Variable 3:</label>
              <select name="var3">
                <option value="">--Ninguno--</option>
                {% for col in columns %}
                  <option value="{{ col }}">{{ col }}</option>
                {% endfor %}
              </select>
              <label>Filtro 3:</label>
              <input type="text" name="filter3"><br><br>
              
              <label>Variable 4:</label>
              <select name="var4">
                <option value="">--Ninguno--</option>
                {% for col in columns %}
                  <option value="{{ col }}">{{ col }}</option>
                {% endfor %}
              </select>
              <label>Filtro 4:</label>
              <input type="text" name="filter4"><br><br>
              
              <h3>Parámetros de Gráfica</h3>
              <label>DPI:</label>
              <input type="text" name="dpi" value="100"><br><br>
              <label>Ancho (px):</label>
              <input type="text" name="width" value="1000"><br><br>
              <label>Alto (px):</label>
              <input type="text" name="height" value="600"><br><br>
              <label>Color gráfico:</label>
              <input type="text" name="color" value="skyblue"><br><br>
              <label>Color borde:</label>
              <input type="text" name="edge_color" value="black"><br><br>
              <label>Color mediana:</label>
              <input type="text" name="mediana_color" value="red"><br><br>

              <label>Número de barras (histograma):</label>
              <input type="number" name="hist_bins" value="20" min="1"><br><br>

              <!-- CONTROLES ADICIONALES DE ESTILO -->
              <label>Título Gráfica:</label>
              <input type="text" name="title" placeholder="Mi título"><br><br>
              <label>Tamaño Título:</label>
              <input type="text" name="title_size" value="14"><br><br>
              <label>Color Título:</label>
              <input type="text" name="title_color" value="black"><br><br>
              
              <label>Etiqueta Eje X:</label>
              <input type="text" name="xlabel" placeholder="Eje X"><br><br>
              <label>Tamaño X:</label>
              <input type="text" name="xlabel_size" value="10"><br><br>
              <label>Color X:</label>
              <input type="text" name="xlabel_color" value="black"><br><br>

              <label>Etiqueta Eje Y:</label>
              <input type="text" name="ylabel" placeholder="Eje Y"><br><br>
              <label>Tamaño Y:</label>
              <input type="text" name="ylabel_size" value="10"><br><br>
              <label>Color Y:</label>
              <input type="text" name="ylabel_color" value="black"><br><br>
              
              <label>X limits (min,max):</label>
              <input type="text" name="xlim" placeholder="0,100"><br><br>
              <label>Y limits (min,max):</label>
              <input type="text" name="ylim" placeholder="0,1"><br><br>
              <label>X ticks (coma sep):</label>
              <input type="text" name="xticks" placeholder="0,20,40"><br><br>
              <label>Y ticks (coma sep):</label>
              <input type="text" name="yticks" placeholder="0,0.5,1"><br><br>

              <label>Mostrar cuadrícula:</label>
              <input type="checkbox" name="grid" checked><br><br>
              <label>Mostrar info N y filtros:</label>
              <input type="checkbox" name="show_info" checked><br><br>
              
              <label>Anotar Pearson/Spearman global:</label>
              <input type="checkbox" name="plot_corr"><br><br>
              
              <input type="submit" value="Aplicar Filtros/Estadísticas">
            </form>
            <br>
            <a href="{{ url_for('upload_file') }}">Cargar otro archivo</a>
          </body>
        </html>
        ''', summary_text=summary_text, columns=columns)

    def summarize_data(df):
        summary_lines = []
        n_vars = df.shape[1]
        n_rows = df.shape[0]
        summary_lines.append("Resumen del archivo subido:")
        summary_lines.append(f"  Número de variables (columnas): {n_vars}")
        summary_lines.append(f"  Número de registros (filas): {n_rows}")
        summary_lines.append("")
        for col in df.columns:
            series = df[col]
            valid_count = series.count()
            summary_lines.append(f"Variable: {col}")
            summary_lines.append(f"  Tipo: {series.dtype}")
            summary_lines.append(f"  Valores válidos: {valid_count}")
            if pd.api.types.is_numeric_dtype(series):
                try:
                    mean_val = series.mean()
                    mode_val = series.mode().iloc[0] if not series.mode().empty else "N/A"
                    summary_lines.append(f"  Promedio: {mean_val:.2f}")
                    summary_lines.append(f"  Moda: {mode_val}")
                except Exception:
                    summary_lines.append("  Error al calcular estadísticas numéricas.")
            else:
                freq = series.value_counts(dropna=True)
                summary_lines.append("  Frecuencia de etiquetas (top 10):")
                for label, count in freq.head(10).items():
                    perc = (count / valid_count * 100) if valid_count > 0 else 0
                    summary_lines.append(f"    {label}: {count} ({perc:.2f}%)")
            summary_lines.append("")
        return "\n".join(summary_lines)

    @app.route('/process', methods=['POST'])
    def process():
        filepath = session.get('filepath', None)
        if not filepath or not os.path.exists(filepath):
            return redirect(url_for('upload_file'))
        try:
            df = pd.read_excel(filepath)
        except Exception as e:
            return "Error al leer el archivo: " + str(e)

        # Lectura de formularios
        var = request.form.get("var")
        filtro = request.form.get("filter", "").strip()
        tipo = request.form.get("tipo")
        etiquetas = request.form.get("etiquetas", "").strip()
        exclude_blank = True if request.form.get("exclude_blank") == "on" else False

        var2 = request.form.get("var2", "").strip()
        filter2 = request.form.get("filter2", "").strip()
        var3 = request.form.get("var3", "").strip()
        filter3 = request.form.get("filter3", "").strip()
        var4 = request.form.get("var4", "").strip()
        filter4 = request.form.get("filter4", "").strip()

        try:
            dpi = int(request.form.get("dpi", "100"))
            width = int(request.form.get("width", "1000"))
            height = int(request.form.get("height", "600"))
        except Exception as e:
            return "Error en los parámetros de gráfica: " + str(e)

        color = request.form.get("color", "skyblue")
        edge_color = request.form.get("edge_color", "black")
        mediana_color = request.form.get("mediana_color", "red")

        hist_bins_raw = request.form.get("hist_bins", "").strip()
        try:
            hist_bins = int(hist_bins_raw) if hist_bins_raw else 20
        except (TypeError, ValueError):
            hist_bins = 20
        if hist_bins < 1:
            hist_bins = 1

        # Nuevos controles de estilo
        title = request.form.get("title", "")
        title_size = int(request.form.get("title_size", "14"))
        title_color = request.form.get("title_color", "black")

        xlabel = request.form.get("xlabel", "")
        xlabel_size = int(request.form.get("xlabel_size", "10"))
        xlabel_color = request.form.get("xlabel_color", "black")

        ylabel = request.form.get("ylabel", "")
        ylabel_size = int(request.form.get("ylabel_size", "10"))
        ylabel_color = request.form.get("ylabel_color", "black")

        xlim_raw = request.form.get("xlim", "").strip()
        ylim_raw = request.form.get("ylim", "").strip()
        xticks_raw = request.form.get("xticks", "").strip()
        yticks_raw = request.form.get("yticks", "").strip()

        grid_on = True if request.form.get("grid") == "on" else False
        show_info = True if request.form.get("show_info") == "on" else False
        plot_corr = True if request.form.get("plot_corr") == "on" else False

        def resolve_plot_text(user_value, default_value):
            if user_value is None:
                return default_value
            if user_value == "":
                return default_value
            stripped = user_value.strip()
            if stripped == "":
                return ""
            return stripped

        # Función de filtrado idéntica a la de escritorio
        def apply_filter_criteria(df, var, filtro):
            if not filtro:
                return df
            if "-" in filtro:
                parts = filtro.split("-")
                if len(parts) == 2:
                    try:
                        lower = float(parts[0].strip())
                        upper = float(parts[1].strip())
                        return df[(df[var] >= lower) & (df[var] <= upper)]
                    except Exception:
                        return df
                else:
                    return df
            elif "," in filtro:
                parts = [p.strip() for p in filtro.split(",")]
                vals = []
                for p in parts:
                    try:
                        vals.append(float(p))
                    except:
                        vals.append(p)
                return df[df[var].isin(vals)]
            else:
                try:
                    val = float(filtro)
                except:
                    val = filtro
                return df[df[var] == val]

        # Aplicar filtros
        df_filtered = df.copy()
        df_filtered = apply_filter_criteria(df_filtered, var, filtro)
        if var2 and filter2:
            df_filtered = apply_filter_criteria(df_filtered, var2, filter2)
        if var3 and filter3:
            df_filtered = apply_filter_criteria(df_filtered, var3, filter3)
        if var4 and filter4:
            df_filtered = apply_filter_criteria(df_filtered, var4, filter4)

        # Conteo de observaciones
        n_obs = df_filtered[var].count() if exclude_blank else len(df_filtered[var])

        result_text = ""
        graph_img = None

        # Estadísticas y gráfico
        if tipo == "Cuantitativa":
            resumen = df_filtered[var].describe().to_string()
            mediana = df_filtered[var].median()
            freq_series = df_filtered[var].value_counts(dropna=exclude_blank)
            freq_df = pd.DataFrame({
                'Count': freq_series,
                'Percentage': (freq_series / n_obs * 100).round(2)
            })
            if not exclude_blank:
                n_blank = len(df_filtered[var]) - df_filtered[var].count()
                result_text = (f"Resumen de {var}:\n{resumen}\n\nMediana: {mediana}\n\n"
                               f"Frecuencia de valores:\n{freq_df.to_string()}\n\n"
                               f"Casillas en blanco: {n_blank}")
            else:
                result_text = f"Resumen de {var}:\n{resumen}\n\nMediana: {mediana}\n\nFrecuencia de valores:\n{freq_df.to_string()}"

            # Generar histograma
            plt.figure(figsize=(width/dpi, height/dpi), dpi=dpi)
            plt.hist(df_filtered[var].dropna(), bins=hist_bins, color=color, edgecolor=edge_color)
            resolved_title = resolve_plot_text(title, f"Histograma de {var}")
            plt.title(resolved_title, fontsize=title_size, color=title_color)
            resolved_xlabel = resolve_plot_text(xlabel, var)
            plt.xlabel(resolved_xlabel, fontsize=xlabel_size, color=xlabel_color)
            resolved_ylabel = resolve_plot_text(ylabel, "Frecuencia")
            plt.ylabel(resolved_ylabel, fontsize=ylabel_size, color=ylabel_color)
            plt.axvline(mediana, color=mediana_color, linestyle="dashed", linewidth=2, label=f"Mediana: {mediana}")
            plt.legend()

            # Límites y ticks
            if xlim_raw:
                try:
                    lo, hi = [float(v) for v in xlim_raw.split(",")]
                    plt.xlim(lo, hi)
                except:
                    pass
            if ylim_raw:
                try:
                    lo, hi = [float(v) for v in ylim_raw.split(",")]
                    plt.ylim(lo, hi)
                except:
                    pass
            if xticks_raw:
                try:
                    ticks = [float(v) for v in xticks_raw.split(",")]
                    plt.xticks(ticks)
                except:
                    pass
            if yticks_raw:
                try:
                    ticks = [float(v) for v in yticks_raw.split(",")]
                    plt.yticks(ticks)
                except:
                    pass

            # Cuadrícula
            plt.grid(grid_on)

            # Anotaciones de Pearson/Spearman global
            if plot_corr and n_obs > 1:
                try:
                    r_p, _ = stats.pearsonr(df_filtered[var].dropna(), df_filtered[var].dropna())
                    r_s, _ = stats.spearmanr(df_filtered[var].dropna(), df_filtered[var].dropna())
                    plt.annotate(f"Pearson: {r_p:.3f}\nSpearman: {r_s:.3f}",
                                 xy=(0.05, 0.05), xycoords="axes fraction",
                                 fontsize=10, ha="left", va="bottom",
                                 bbox=dict(boxstyle="round", facecolor="white", alpha=0.5))
                except:
                    pass

            # Mostrar info de n y filtros
            if show_info:
                info_text = f"n = {n_obs}\nFiltros: {var}={filtro}"
                if var2 and filter2: info_text += f", {var2}={filter2}"
                if var3 and filter3: info_text += f", {var3}={filter3}"
                if var4 and filter4: info_text += f", {var4}={filter4}"
                plt.annotate(info_text, xy=(0.95, 0.95), xycoords="axes fraction",
                             fontsize=8, ha="right", va="top",
                             bbox=dict(boxstyle="round", facecolor="white", alpha=0.5))

            plt.tight_layout()
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            graph_img = base64.b64encode(buf.getvalue()).decode('utf-8')
            plt.close()

        elif tipo == "Cualitativa":
            freq_series = df_filtered[var].value_counts(dropna=exclude_blank)
            if not exclude_blank:
                freq_series.index = [("nan" if pd.isna(x) else x) for x in freq_series.index]
            freq_df = pd.DataFrame({
                'Count': freq_series,
                'Percentage': (freq_series / n_obs * 100).round(2)
            })
            if etiquetas:
                mapping = {}
                for pair in etiquetas.split(","):
                    if ":" in pair:
                        original, label = pair.split(":", 1)
                        mapping[original.strip()] = label.strip()
                new_index = [mapping.get(str(val), str(val)) for val in freq_df.index]
                freq_df.index = new_index
            resumen = df_filtered[var].describe().to_string()
            if not exclude_blank:
                n_blank = len(df_filtered[var]) - df_filtered[var].count()
                result_text = (f"Resumen de {var}:\n{resumen}\n\n"
                               f"Frecuencia de valores:\n{freq_df.to_string()}\n\n"
                               f"Casillas en blanco: {n_blank}")
            else:
                result_text = f"Resumen de {var}:\n{resumen}\n\nFrecuencia de valores:\n{freq_df.to_string()}"

            # Gráfico de barras
            plt.figure(figsize=(width/dpi, height/dpi), dpi=dpi)
            plt.bar(freq_df.index, freq_df['Count'].values, color=color, edgecolor=edge_color)
            resolved_bar_title = resolve_plot_text(title, f"Gráfico de barras de frecuencias para {var}")
            plt.title(resolved_bar_title, fontsize=title_size, color=title_color)
            resolved_bar_xlabel = resolve_plot_text(xlabel, var)
            plt.xlabel(resolved_bar_xlabel, fontsize=xlabel_size, color=xlabel_color)
            resolved_bar_ylabel = resolve_plot_text(ylabel, "Frecuencia")
            plt.ylabel(resolved_bar_ylabel, fontsize=ylabel_size, color=ylabel_color)
            plt.xticks(rotation=45, ha="right")

            # Límites y ticks
            if ylim_raw:
                try:
                    lo, hi = [float(v) for v in ylim_raw.split(",")]
                    plt.ylim(lo, hi)
                except:
                    pass
            if yticks_raw:
                try:
                    ticks = [float(v) for v in yticks_raw.split(",")]
                    plt.yticks(ticks)
                except:
                    pass

            # Cuadrícula
            plt.grid(grid_on)

            # Pearson/Spearman global
            if plot_corr and n_obs > 1:
                try:
                    # Para cualitativa no tiene sentido global, omitido
                    pass
                except:
                    pass

            # Info de n y filtros
            if show_info:
                info_text = f"n = {n_obs}\nFiltros: {var}={filtro}"
                if var2 and filter2: info_text += f", {var2}={filter2}"
                if var3 and filter3: info_text += f", {var3}={filter3}"
                if var4 and filter4: info_text += f", {var4}={filter4}"
                plt.annotate(info_text, xy=(0.95, 0.95), xycoords="axes fraction",
                             fontsize=8, ha="right", va="top",
                             bbox=dict(boxstyle="round", facecolor="white", alpha=0.5))

            plt.tight_layout()
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            graph_img = base64.b64encode(buf.getvalue()).decode('utf-8')
            plt.close()

        return render_template_string('''
        <!doctype html>
        <html>
          <head>
            <title>Resultados</title>
            <style>
              .container { display: flex; }
              .left { flex: 1; padding: 10px; }
              .right { flex: 1; padding: 10px; }
            </style>
          </head>
          <body>
            <h1>Resultados de Filtros/Estadísticas</h1>
            <div class="container">
              <div class="left">
                <h2>Resumen</h2>
                <pre>{{ result_text }}</pre>
              </div>
              <div class="right">
                <h2>Gráfica</h2>
                {% if graph_img %}
                  <img src="data:image/png;base64,{{ graph_img }}" alt="Graph">
                {% else %}
                  <p>No se generó gráfica.</p>
                {% endif %}
              </div>
            </div>
            <br>
            <a href="{{ url_for('configure') }}">Volver a Configuración</a> | 
            <a href="{{ url_for('upload_file') }}">Cargar otro archivo</a>
          </body>
        </html>
        ''', result_text=result_text, graph_img=graph_img)

    app.run(debug=True)


# ==============================
# Código de la aplicación de ESCRITORIO (Tkinter – RegresionesTab)
# ==============================
def fmt_p(p):
    try:
        return f"{p:.3e}" if not np.isnan(p) else "0.000e+00"
    except Exception:
        return "N/A"

def safe_pearson(y, y_pred):
    if len(y) < 2 or len(y_pred) < 2 or np.std(y) < 1e-8 or np.std(y_pred) < 1e-8: #Añadido chequeo de longitud
        return np.nan, np.nan
    return stats.pearsonr(y, y_pred)

def safe_spearman(y, y_pred):
    if len(y) < 2 or len(y_pred) < 2 or np.std(y) < 1e-8 or np.std(y_pred) < 1e-8: #Añadido chequeo de longitud
        return np.nan, np.nan
    return stats.spearmanr(y, y_pred)

def exp_model1(x, a, b):
    return a + np.power(b, x)

def exp_model2(x, a, b):
    return a + np.power(x, b)

def exp_model3(x, A, B):
    return A * np.power(B, x)

def exp_decay(x, A, B):
    return A * np.exp(-B * x)

def sigmoid(x, L, k, x0, off):
    return L / (1 + np.exp(-k*(x - x0))) + off

def find_x0(model_func, params, x_guess):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            sol = fsolve(lambda x: model_func(x, *params), x_guess, maxfev=10000)
            return sol[0]
        except Exception:
            return None

def compute_acme(func, x_range):
    y_vals = func(x_range)
    idx = np.argmax(y_vals) if (np.max(y_vals) - np.min(y_vals)) >= 0 else np.argmin(y_vals)
    return x_range[idx], y_vals[idx]

class ScrollableFrame(ttk.Frame):
    def __init__(self, container, *args, **kwargs):
        super().__init__(container, *args, **kwargs)
        self.canvas = tk.Canvas(self)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.v_scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.v_scrollbar.grid(row=0, column=1, sticky="ns")
        self.h_scrollbar = ttk.Scrollbar(self, orient="horizontal", command=self.canvas.xview)
        self.h_scrollbar.grid(row=1, column=0, sticky="ew")
        self.canvas.configure(yscrollcommand=self.v_scrollbar.set, xscrollcommand=self.h_scrollbar.set)
        self.scrollable_frame = ttk.Frame(self.canvas)
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.scrollable_frame.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)

class RegresionesTab(ttk.Frame):
    def __init__(self, master, main_app_instance=None):
        super().__init__(master)
        self.main_app = main_app_instance
        self.data = None # DataFrame original cargado
        self.filtered_data = None # DataFrame después de aplicar filtros
        self.shared_filter_summary = []  # Resumen de filtros del Archivo de Trabajo
        self.graph_path = "temp_graph.png" # Considerar un subdirectorio temporal
        self.results_text_content = "" # Para almacenar el texto de resultados
        self.plot_label_map = {} # Para mapear nombres de variables a etiquetas de gráfico
        self.default_colors = ["#0072B2", "#D55E00", "#009E73", "#F0E442", "#56B4E9", "#CC79A7", "#999999", "#E69F00"]
        self.color_options = ["blue", "green", "red", "skyblue", "orange", "purple",
                               "black", "gray", "brown", "pink", "cyan", "magenta",
                               "teal", "olive", "navy", "maroon", "lime", "gold"]

        self.model_styles = {
            "Lineal": {"linestyle": "-", "marker": "o"},
            "Cuadrático": {"linestyle": "--", "marker": "s"},
            "Cúbico": {"linestyle": ":", "marker": "d"},
            "Potencia": {"linestyle": "-.", "marker": "^"},
            "Logarítmico": {"linestyle": (0, (3, 1, 1, 1)), "marker": "v"},
            "LOESS": {"linestyle": (0, (5, 5)), "marker": "x"},
            "Exp (a+b^x)": {"linestyle": (0, (1, 1)), "marker": "p"},
            "Exp (a+x^b)": {"linestyle": (0, (3, 5, 1, 5)), "marker": "h"},
            "Exp (A*B^x)": {"linestyle": (0, (5, 2, 1, 2)), "marker": "*"},
            "Sigmoide": {"linestyle": (0, (1, 10)), "marker": "+"},
            "Exp Decreciente": {"linestyle": (0, (2, 2)), "marker": "D"},
        }

        self.paned = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        self.paned.pack(fill="both", expand=True)
        
        # Frame izquierdo con scroll
        left_scroll_container = ScrollableFrame(self.paned)
        self.paned.add(left_scroll_container, weight=1)
        container = left_scroll_container.scrollable_frame # Este es el frame donde van los widgets

        # Frame derecho para resultados
        right_frame = ttk.Frame(self.paned)
        self.paned.add(right_frame, weight=1)

        # Notebook en el panel derecho para resultados y gráfica
        self.results_notebook = ttk.Notebook(right_frame)
        self.results_notebook.pack(fill="both", expand=True, padx=10, pady=10)

        # Pestaña de Resumen
        frm_results = ttk.Frame(self.results_notebook)
        self.results_notebook.add(frm_results, text="Resumen de Resultados")
        self.txt_results = scrolledtext.ScrolledText(frm_results, wrap="none", height=15)
        self.txt_results.pack(fill="both", expand=True)
        self.txt_results.config(state="disabled")

        # Pestaña de Gráfica
        self.graph_frame = ttk.Frame(self.results_notebook)
        self.results_notebook.add(self.graph_frame, text="Gráfica")

        # Canvas para la gráfica
        self.fig = plt.figure(figsize=(5, 4), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.graph_frame)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Toolbar para la gráfica
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.graph_frame)
        self.toolbar.update()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # --- Contenido del frame izquierdo (container) ---
        frm_top = ttk.Frame(container)
        frm_top.pack(fill="x", padx=10, pady=5)
        lbl_title = ttk.Label(frm_top, text="Análisis de Regresiones y Dispersión", font=("Helvetica", 14))
        lbl_title.pack(side="left")
        self.msg_label = ttk.Label(frm_top, text="", foreground="blue", wraplength=300)
        self.msg_label.pack(side="right", anchor="ne", padx=10)

        btn_load = ttk.Button(container, text="Cargar Datos (Excel)", command=self.load_data)
        btn_load.pack(pady=5, padx=10, fill="x")
        self.lbl_file = ttk.Label(container, text="Ningún archivo cargado.")
        self.lbl_file.pack(pady=5, padx=10, anchor="w")

        # --- Filtros (Nuevo Componente) ---
        frm_filters_outer = ttk.LabelFrame(container, text="Filtros de Datos (Opcional)")
        frm_filters_outer.pack(fill="x", padx=10, pady=5)
        if FilterComponent:
            self.filter_component = FilterComponent(frm_filters_outer, max_unique_cat=50, log_callback=self.log_message)
            self.filter_component.pack(fill="both", expand=True) # Usar fill="both" y expand=True
        else:
            ttk.Label(frm_filters_outer, text="Error: Componente de filtro no cargado.").pack()
        
        frm_filters_outer.columnconfigure(0, weight=1) # Asegurar que el frame de filtros se expanda
        frm_filters_outer.rowconfigure(0, weight=1) # Asegurar que el frame de filtros se expanda


        # --- Selección de Variables para Regresión ---
        frm_vars = ttk.LabelFrame(container, text="Selección de Variables")
        frm_vars.pack(fill="x", padx=10, pady=5)
        
        dep_var_frame = ttk.Frame(frm_vars)
        dep_var_frame.pack(fill="x", pady=2)
        ttk.Label(dep_var_frame, text="Variables Dependientes (selección múltiple):").pack(anchor="nw", padx=5)
        self.listbox_dep_vars_spec = tk.Listbox(dep_var_frame, selectmode=tk.MULTIPLE, height=4, exportselection=False)
        dep_vars_v_scrollbar = ttk.Scrollbar(dep_var_frame, orient="vertical", command=self.listbox_dep_vars_spec.yview)
        self.listbox_dep_vars_spec.configure(yscrollcommand=dep_vars_v_scrollbar.set)
        dep_vars_v_scrollbar.pack(side="right", fill="y")
        self.listbox_dep_vars_spec.pack(side="left", fill="both", expand=True, padx=5, pady=(0,5))
        self.listbox_dep_vars_spec.bind('<<ListboxSelect>>', self._update_indep_vars_listbox)

        indep_vars_frame = ttk.Frame(frm_vars)
        indep_vars_frame.pack(fill="both", expand=True, pady=2)
        # Store original label text for reuse
        indep_vars_label_text = "Variables Independientes (seleccione de la lista):"
        ttk.Label(indep_vars_frame, text=indep_vars_label_text).pack(anchor="nw", padx=5)

        self.listbox_indep_vars_spec = tk.Listbox(indep_vars_frame, selectmode=tk.MULTIPLE, height=6, exportselection=False)
        indep_vars_v_scrollbar = ttk.Scrollbar(indep_vars_frame, orient="vertical", command=self.listbox_indep_vars_spec.yview)
        indep_vars_h_scrollbar = ttk.Scrollbar(indep_vars_frame, orient="horizontal", command=self.listbox_indep_vars_spec.xview)
        self.listbox_indep_vars_spec.configure(yscrollcommand=indep_vars_v_scrollbar.set, xscrollcommand=indep_vars_h_scrollbar.set)

        indep_vars_v_scrollbar.pack(side="right", fill="y")
        indep_vars_h_scrollbar.pack(side="bottom", fill="x")
        self.listbox_indep_vars_spec.pack(side="left", fill="both", expand=True, padx=5, pady=(0,5))
        
        rename_vars_frame = ttk.Frame(frm_vars)
        rename_vars_frame.pack(fill="x", pady=5)
        ttk.Label(rename_vars_frame, text="Renombrar variable (opcional):").pack(side="left", padx=5)
        self.rename_var_entry = ttk.Entry(rename_vars_frame, width=20)
        self.rename_var_entry.pack(side="left", padx=5)
        ttk.Button(rename_vars_frame, text="Renombrar", command=self.rename_variable).pack(side="left", padx=5)

        # --- Comparación por Grupos ---
        frm_groups = ttk.LabelFrame(container, text="Comparación entre Grupos (Opcional)")
        frm_groups.pack(fill="x", padx=10, pady=5)
        
        group_row1 = ttk.Frame(frm_groups)
        group_row1.pack(fill="x", pady=2)
        
        self.var_compare_groups = tk.BooleanVar(value=False)
        ttk.Checkbutton(group_row1, text="Comparar por grupos", variable=self.var_compare_groups, 
                       command=self._toggle_group_comparison).pack(side="left", padx=5)
        
        ttk.Label(group_row1, text="Variable de agrupación:").pack(side="left", padx=(15, 5))
        self.cmb_group_var = ttk.Combobox(group_row1, values=[], state="disabled", width=20)
        self.cmb_group_var.pack(side="left", padx=5)
        
        group_row2 = ttk.Frame(frm_groups)
        group_row2.pack(fill="x", pady=2)
        
        ttk.Label(group_row2, text="Grupos a incluir (dejar vacío = todos):").pack(side="left", padx=5)
        self.entry_group_filter = ttk.Entry(group_row2, width=30, state="disabled")
        self.entry_group_filter.pack(side="left", padx=5)
        ttk.Label(group_row2, text="Ej: Masculino,Femenino", foreground="gray").pack(side="left", padx=5)
        
        group_row3 = ttk.Frame(frm_groups)
        group_row3.pack(fill="x", pady=2)
        
        self.var_separate_plots = tk.BooleanVar(value=False)
        self.chk_separate_plots = ttk.Checkbutton(group_row3, text="Gráficos separados por grupo (facetas)", 
                                                   variable=self.var_separate_plots, state="disabled")
        self.chk_separate_plots.pack(side="left", padx=5)
        
        self.var_show_group_stats = tk.BooleanVar(value=True)
        self.chk_group_stats = ttk.Checkbutton(group_row3, text="Mostrar estadísticas por grupo", 
                                                variable=self.var_show_group_stats, state="disabled")
        self.chk_group_stats.pack(side="left", padx=15)
        
        # Análisis de interacción (comparación formal de pendientes)
        group_row4 = ttk.Frame(frm_groups)
        group_row4.pack(fill="x", pady=2)
        
        self.var_interaction_analysis = tk.BooleanVar(value=True)
        self.chk_interaction = ttk.Checkbutton(group_row4, 
            text="Análisis de interacción (comparar pendientes)", 
            variable=self.var_interaction_analysis, state="disabled")
        self.chk_interaction.pack(side="left", padx=5)
        
        self.var_ancova = tk.BooleanVar(value=True)
        self.chk_ancova = ttk.Checkbutton(group_row4, 
            text="ANCOVA (medias ajustadas)", 
            variable=self.var_ancova, state="disabled")
        self.chk_ancova.pack(side="left", padx=10)
        
        self.var_chow_test = tk.BooleanVar(value=True)
        self.chk_chow = ttk.Checkbutton(group_row4, 
            text="Prueba de Chow (ruptura estructural)", 
            variable=self.var_chow_test, state="disabled")
        self.chk_chow.pack(side="left", padx=10)
        
        # Fila adicional para comparación NLS
        group_row4b = ttk.Frame(frm_groups)
        group_row4b.pack(fill="x", pady=2)
        
        self.var_nls_comparison = tk.BooleanVar(value=True)
        self.chk_nls = ttk.Checkbutton(group_row4b, 
            text="Comparación NLS (AIC/BIC modelos no lineales)", 
            variable=self.var_nls_comparison, state="disabled")
        self.chk_nls.pack(side="left", padx=5)
        
        ttk.Label(group_row4b, text="Usa AIC y Likelihood Ratio Test para comparar curvas no lineales",
                  foreground="gray", font=("Consolas", 8)).pack(side="left", padx=10)
        
        # Tooltip/ayuda para los análisis
        group_row5 = ttk.Frame(frm_groups)
        group_row5.pack(fill="x", pady=2)
        ttk.Label(group_row5, text="Interacción: Y = β₀ + β₁X + β₂G + β₃(X·G)  |  ANCOVA: Y = β₀ + β₁X + β₂G (pendientes paralelas)", 
                  foreground="gray", font=("Consolas", 8)).pack(side="left", padx=5)

        # --- Parámetros Gráficos ---
        frm_params = ttk.LabelFrame(container, text="Parámetros Gráficos")
        frm_params.pack(fill="x", padx=10, pady=5)
        param_grid_frame = ttk.Frame(frm_params) 
        param_grid_frame.pack(fill="x", expand=True)

        ttk.Label(param_grid_frame, text="DPI:").grid(row=0, column=0, padx=5, pady=2, sticky="w")
        self.entry_dpi = ttk.Entry(param_grid_frame, width=7)
        self.entry_dpi.grid(row=0, column=1, padx=5, pady=2, sticky="w")
        self.entry_dpi.insert(0, "100")
        ttk.Label(param_grid_frame, text="Ancho (px):").grid(row=0, column=2, padx=5, pady=2, sticky="w")
        self.entry_width = ttk.Entry(param_grid_frame, width=7)
        self.entry_width.grid(row=0, column=3, padx=5, pady=2, sticky="w")
        self.entry_width.insert(0, "800")
        ttk.Label(param_grid_frame, text="Alto (px):").grid(row=0, column=4, padx=5, pady=2, sticky="w")
        self.entry_height = ttk.Entry(param_grid_frame, width=7)
        self.entry_height.grid(row=0, column=5, padx=5, pady=2, sticky="w")
        self.entry_height.insert(0, "600")

        ttk.Label(param_grid_frame, text="Color Puntos:").grid(row=1, column=0, padx=5, pady=2, sticky="w")
        self.cmb_pt_color = ttk.Combobox(param_grid_frame, values=self.color_options, state="readonly", width=10)
        self.cmb_pt_color.grid(row=1, column=1, padx=5, pady=2, sticky="w")
        self.cmb_pt_color.set("blue")

        ttk.Label(param_grid_frame, text="Tamaño Puntos:").grid(row=1, column=2, padx=5, pady=2, sticky="w")
        self.entry_pt_size = ttk.Entry(param_grid_frame, width=7)
        self.entry_pt_size.grid(row=1, column=3, padx=5, pady=2, sticky="w")
        self.entry_pt_size.insert(0, "20") # Valor por defecto para el tamaño de puntos

        ttk.Label(param_grid_frame, text="Tamaño Texto Ejes:").grid(row=1, column=4, padx=5, pady=2, sticky="w")
        self.entry_text_size = ttk.Entry(param_grid_frame, width=7)
        self.entry_text_size.grid(row=1, column=5, padx=5, pady=2, sticky="w")
        self.entry_text_size.insert(0, "10")

        ttk.Label(param_grid_frame, text="Título Gráfica:").grid(row=2, column=0, sticky="w", padx=5, pady=2)
        self.entry_title = ttk.Entry(param_grid_frame, width=20)
        self.entry_title.grid(row=2, column=1, columnspan=2, sticky="we", padx=5)
        ttk.Label(param_grid_frame, text="Tamaño Título:").grid(row=2, column=3, sticky="w", padx=5) 
        self.entry_title_size = ttk.Entry(param_grid_frame, width=5)
        self.entry_title_size.insert(0, "14")
        self.entry_title_size.grid(row=2, column=4, sticky="w", padx=5)

        ttk.Label(param_grid_frame, text="Fuente:").grid(row=8, column=0, sticky="w", padx=5, pady=2)
        self.font_family_var = tk.StringVar(value="sans-serif")
        font_families = ["serif", "sans-serif", "monospace", "Arial", "Times New Roman", "Courier New", "Palatino Linotype"]
        self.font_family_combo = ttk.Combobox(param_grid_frame, textvariable=self.font_family_var, values=font_families, state="readonly", width=15)
        self.font_family_combo.grid(row=8, column=1, columnspan=2, sticky="we", padx=5)

        ttk.Label(param_grid_frame, text="Etiqueta Eje X:").grid(row=3, column=0, sticky="w", padx=5, pady=2)
        self.entry_xlabel = ttk.Entry(param_grid_frame, width=20)
        self.entry_xlabel.grid(row=3, column=1, columnspan=2, sticky="we", padx=5)
        ttk.Label(param_grid_frame, text="Etiqueta Eje Y:").grid(row=4, column=0, sticky="w", padx=5, pady=2)
        self.entry_ylabel = ttk.Entry(param_grid_frame, width=20)
        self.entry_ylabel.grid(row=4, column=1, columnspan=2, sticky="we", padx=5)

        ttk.Label(param_grid_frame, text="X lim (min,max):").grid(row=5, column=0, sticky="w", padx=5, pady=2)
        self.entry_xlim = ttk.Entry(param_grid_frame, width=10)
        self.entry_xlim.grid(row=5, column=1, sticky="w", padx=5)
        ttk.Label(param_grid_frame, text="Y lim (min,max):").grid(row=5, column=2, sticky="w", padx=5)
        self.entry_ylim = ttk.Entry(param_grid_frame, width=10)
        self.entry_ylim.grid(row=5, column=3, sticky="w", padx=5)
        ttk.Label(param_grid_frame, text="X ticks (a,b,c):").grid(row=6, column=0, sticky="w", padx=5, pady=2)
        self.entry_xticks = ttk.Entry(param_grid_frame, width=10)
        self.entry_xticks.grid(row=6, column=1, sticky="w", padx=5)
        ttk.Label(param_grid_frame, text="Y ticks (a,b,c):").grid(row=6, column=2, sticky="w", padx=5)
        self.entry_yticks = ttk.Entry(param_grid_frame, width=10)
        self.entry_yticks.grid(row=6, column=3, sticky="w", padx=5)

        self.var_grid = tk.BooleanVar(value=True)
        ttk.Checkbutton(param_grid_frame, text="Cuadrícula", variable=self.var_grid).grid(row=7, column=0, sticky="w", padx=5, pady=2)
        self.var_show_info = tk.BooleanVar(value=True)
        ttk.Checkbutton(param_grid_frame, text="Info N/Filtros", variable=self.var_show_info).grid(row=7, column=1, sticky="w", padx=5)
        self.var_plot_corr = tk.BooleanVar(value=False)
        ttk.Checkbutton(param_grid_frame, text="Anotar Correl.", variable=self.var_plot_corr).grid(row=7, column=2, sticky="w", padx=5)

        self.var_hide_points_labels = tk.BooleanVar(value=False)
        ttk.Checkbutton(param_grid_frame, text="Ocultar etiquetas de puntos", variable=self.var_hide_points_labels).grid(row=7, column=3, sticky="w", padx=5)

        ttk.Label(param_grid_frame, text="Decimales:").grid(row=9, column=0, sticky="w", padx=5, pady=2)
        self.decimals_var = tk.IntVar(value=2)
        ttk.Spinbox(param_grid_frame, from_=0, to=10, textvariable=self.decimals_var, width=5).grid(row=9, column=1, sticky="w", padx=5)

        self.sci_notation_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(param_grid_frame, text="Notación científica", variable=self.sci_notation_var).grid(row=9, column=2, sticky="w", padx=5)

        self.sci_notation_conditional_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(param_grid_frame, text="Notación científica (solo si aplica)", variable=self.sci_notation_conditional_var).grid(row=9, column=3, sticky="w", padx=5)

        self.var_table_only = tk.BooleanVar(value=False)
        ttk.Checkbutton(param_grid_frame, text="Mostrar solo tabla de correlaciones", variable=self.var_table_only).grid(row=10, column=0, columnspan=2, sticky="w", padx=5, pady=2)

        self.var_show_formula = tk.BooleanVar(value=True)
        ttk.Checkbutton(param_grid_frame, text="Mostrar fórmula", variable=self.var_show_formula).grid(row=10, column=2, sticky="w", padx=5)

        self.var_show_r2 = tk.BooleanVar(value=True)
        ttk.Checkbutton(param_grid_frame, text="Mostrar R²", variable=self.var_show_r2).grid(row=10, column=3, sticky="w", padx=5)

        # New: Custom Labels for Dependent Variable
        ttk.Label(param_grid_frame, text="Etiquetas Personalizadas (Var. Dep.):").grid(row=11, column=0, sticky="w", padx=5, pady=2)
        self.entry_dep_var_labels = ttk.Entry(param_grid_frame, width=30)
        self.entry_dep_var_labels.grid(row=11, column=1, columnspan=3, sticky="we", padx=5)
        ttk.Label(param_grid_frame, text="Ej: 1:Bajo,2:Medio").grid(row=11, column=4, columnspan=2, sticky="w", padx=5)


        # --- Modelos de Regresión ---
        frm_models = ttk.LabelFrame(container, text="Modelos de Regresión a Aplicar")
        frm_models.pack(fill="x", padx=10, pady=5)
        self.var_linear = tk.BooleanVar(value=True); ttk.Checkbutton(frm_models, text="Lineal", variable=self.var_linear).grid(row=0, column=0, padx=2, pady=2, sticky="w")
        self.var_quadratic = tk.BooleanVar(value=False); ttk.Checkbutton(frm_models, text="Cuadrática", variable=self.var_quadratic).grid(row=0, column=1, padx=2, pady=2, sticky="w")
        self.var_cubic = tk.BooleanVar(value=False); ttk.Checkbutton(frm_models, text="Cúbica", variable=self.var_cubic).grid(row=0, column=2, padx=2, pady=2, sticky="w")
        self.var_power = tk.BooleanVar(value=False); ttk.Checkbutton(frm_models, text="Potencia", variable=self.var_power).grid(row=0, column=3, padx=2, pady=2, sticky="w")
        self.var_log = tk.BooleanVar(value=False); ttk.Checkbutton(frm_models, text="Logarítmica", variable=self.var_log).grid(row=0, column=4, padx=2, pady=2, sticky="w")
        self.var_loess = tk.BooleanVar(value=False); ttk.Checkbutton(frm_models, text="LOESS", variable=self.var_loess).grid(row=0, column=5, padx=2, pady=2, sticky="w")
        self.var_inverse = tk.BooleanVar(value=False); ttk.Checkbutton(frm_models, text="Inversa (1/X)", variable=self.var_inverse).grid(row=1, column=0, padx=2, pady=2, sticky="w")
        self.var_rcs = tk.BooleanVar(value=False); ttk.Checkbutton(frm_models, text="Splines (RCS)", variable=self.var_rcs).grid(row=1, column=1, padx=2, pady=2, sticky="w")
        
        # --- Transformaciones de Variables ---
        frm_transform = ttk.LabelFrame(container, text="Transformaciones de Variables")
        frm_transform.pack(fill="x", padx=10, pady=5)
        
        self.var_ln_x = tk.BooleanVar(value=False)
        ttk.Checkbutton(frm_transform, text="ln(X) - Transformar Variable Independiente", 
                        variable=self.var_ln_x).grid(row=0, column=0, padx=5, pady=2, sticky="w")
        
        self.var_ln_y = tk.BooleanVar(value=False)
        ttk.Checkbutton(frm_transform, text="ln(Y) - Transformar Variable Dependiente", 
                        variable=self.var_ln_y).grid(row=0, column=1, padx=5, pady=2, sticky="w")
        
        ttk.Label(frm_transform, text="Nota: Los valores ≤ 0 serán excluidos al aplicar ln", 
                  foreground="gray").grid(row=1, column=0, columnspan=2, padx=5, pady=2, sticky="w")
        
        frm_otros = ttk.LabelFrame(container, text="Otros Modelos (Solo Resumen)")
        frm_otros.pack(fill="x", padx=10, pady=5)
        self.var_exp1 = tk.BooleanVar(value=False); ttk.Checkbutton(frm_otros, text="Exp (a+b^x)", variable=self.var_exp1).grid(row=0, column=0, padx=2, pady=2, sticky="w")
        self.var_exp2 = tk.BooleanVar(value=False); ttk.Checkbutton(frm_otros, text="Exp (a+x^b)", variable=self.var_exp2).grid(row=0, column=1, padx=2, pady=2, sticky="w")
        self.var_exp3 = tk.BooleanVar(value=False); ttk.Checkbutton(frm_otros, text="Exp (A*B^x)", variable=self.var_exp3).grid(row=0, column=2, padx=2, pady=2, sticky="w")
        self.var_sigmoid = tk.BooleanVar(value=False); ttk.Checkbutton(frm_otros, text="Sigmoide", variable=self.var_sigmoid).grid(row=0, column=3, padx=2, pady=2, sticky="w")
        self.var_exp_decay = tk.BooleanVar(value=False); ttk.Checkbutton(frm_otros, text="Exp Decreciente", variable=self.var_exp_decay).grid(row=0, column=4, padx=2, pady=2, sticky="w")

        frm_line_options = ttk.LabelFrame(container, text="Opciones de Línea de Regresión")
        frm_line_options.pack(fill="x", padx=10, pady=5)

        ttk.Label(frm_line_options, text="Color Líneas:").grid(row=0, column=0, padx=5, pady=2, sticky="w")
        self.cmb_line_color = ttk.Combobox(frm_line_options, values=self.color_options, state="readonly", width=10)
        self.cmb_line_color.grid(row=0, column=1, padx=5, pady=2, sticky="w")
        self.cmb_line_color.set("red")

        ttk.Label(frm_line_options, text="Grosor Líneas:").grid(row=0, column=2, padx=5, pady=2, sticky="w")
        self.entry_line_width = ttk.Entry(frm_line_options, width=7)
        self.entry_line_width.grid(row=0, column=3, padx=5, pady=2, sticky="w")
        self.entry_line_width.insert(0, "2")


        # --- Botones de Acción Finales ---
        frm_buttons_bottom = ttk.Frame(container)
        frm_buttons_bottom.pack(pady=10, fill="x", padx=10)
        btn_plot = ttk.Button(frm_buttons_bottom, text="Generar Dispersión y Regresión", command=self.plot_regression)
        btn_plot.pack(side="left", padx=5, expand=True, fill="x")
        btn_corr_table = ttk.Button(frm_buttons_bottom, text="Mostrar Tabla de Correlaciones", command=self.calculate_and_show_correlations)
        btn_corr_table.pack(side="left", padx=5, expand=True, fill="x")
        btn_save = ttk.Button(frm_buttons_bottom, text="Guardar Gráfica", command=self.save_graph_directly)
        btn_save.pack(side="left", padx=5, expand=True, fill="x")

    def receive_shared_dataset(self, *, dataset, filtered_dataset=None, filter_summary=None, metadata=None, source_widget=None):
        """Recibe el dataset compartido de la aplicación principal."""
        if source_widget is self:
            return

        if dataset is None:
            self.data = None
            self.filtered_data = None
            self.shared_filter_summary = []
            self.lbl_file.config(text="Ningún archivo cargado.")
            self.listbox_dep_vars_spec.delete(0, tk.END)
            self.listbox_indep_vars_spec.delete(0, tk.END)
            self.cmb_group_var['values'] = ['']
            self.cmb_group_var.set('')
            if hasattr(self, 'filter_component') and self.filter_component:
                try:
                    self.filter_component.set_dataframe(None)
                except Exception:
                    pass
            self.log_message("Dataset compartido limpiado en Regresiones.")
            return

        # Si recibimos un dataset, lo usamos
        if dataset is not None:
            self.data = dataset
            
            # Usar el dataset filtrado si está disponible
            if filtered_dataset is not None and isinstance(filtered_dataset, pd.DataFrame):
                self.filtered_data = filtered_dataset
                self.shared_filter_summary = list(filter_summary or [])
            else:
                self.filtered_data = dataset.copy()
                self.shared_filter_summary = []
            
            # Actualizar etiqueta de archivo
            source_name = "Dataset Compartido"
            if metadata and 'source_path' in metadata:
                source_name = os.path.basename(metadata['source_path'])
            
            # Intentar obtener dimensiones - mostrar del filtrado
            try:
                rows_orig, cols = self.data.shape
                rows_filt = self.filtered_data.shape[0] if self.filtered_data is not None else rows_orig
                
                if self.shared_filter_summary:
                    filter_info = f" | {len(self.shared_filter_summary)} filtro(s): {rows_filt} filas"
                else:
                    filter_info = ""
                
                self.lbl_file.config(text=f"{source_name} ({rows_orig}x{cols}){filter_info}")
                print(f"[RegresionesTab] Dataset recibido: {rows_filt} filas (de {rows_orig} original){filter_info}")
            except Exception as e:
                self.lbl_file.config(text=f"{source_name} [Compartido]")
                print(f"[RegresionesTab] Dataset recibido: {source_name}")

            # Actualizar selectores (esto respetará el orden de columnas del dataset recibido)
            self._update_variable_selectors()
            
            # Actualizar componente de filtros si existe
            if hasattr(self, 'filter_component') and self.filter_component:
                self.filter_component.set_dataframe(self.data)
            
            # Limpiar resultados anteriores ya que cambiaron los datos
            self.txt_results.config(state="normal")
            self.txt_results.delete("1.0", tk.END)
            self.txt_results.config(state="disabled")
            self.fig.clear()
            self.canvas.draw()
            self.log_message("Dataset compartido listo en Regresiones. Selecciones por defecto actualizadas.")

    def _update_variable_selectors(self):
        """Actualiza todos los selectores de variables después de un cambio en las columnas del DataFrame."""
        if self.data is None:
            all_cols = []
            num_cols = []
        else:
            all_cols = list(self.data.columns)
            num_cols = list(self.data.select_dtypes(include=np.number).columns)

        # Guardar selecciones actuales de variables dependientes
        current_dep_indices = self.listbox_dep_vars_spec.curselection()
        current_dep_vars = {self.listbox_dep_vars_spec.get(i) for i in current_dep_indices}

        current_indep_indices = self.listbox_indep_vars_spec.curselection()
        current_indep_vars = {self.listbox_indep_vars_spec.get(i) for i in current_indep_indices}

        # Actualizar Listbox de variables dependientes
        self.listbox_dep_vars_spec.delete(0, tk.END)
        for idx, col in enumerate(num_cols):
            self.listbox_dep_vars_spec.insert(tk.END, col)
            if col in current_dep_vars:
                self.listbox_dep_vars_spec.selection_set(idx)
        if not current_dep_vars and num_cols:
            self.listbox_dep_vars_spec.selection_set(0)
        
        # Actualizar Listbox de variables independientes
        self._update_indep_vars_listbox(current_indep_vars=current_indep_vars)

        # Actualizar combo de variable de agrupación (todas las columnas, no solo numéricas)
        current_group = self.cmb_group_var.get()
        self.cmb_group_var['values'] = [''] + all_cols
        if current_group in all_cols:
            self.cmb_group_var.set(current_group)
        else:
            self.cmb_group_var.set('')

        # Actualizar componente de filtro si existe
        if hasattr(self, 'filter_component') and self.filter_component:
            self.filter_component.set_dataframe(self.data)

    def _toggle_group_comparison(self):
        """Activa o desactiva los controles de comparación por grupos."""
        enabled = self.var_compare_groups.get()
        state = "readonly" if enabled else "disabled"
        entry_state = "normal" if enabled else "disabled"
        
        self.cmb_group_var.config(state=state)
        self.entry_group_filter.config(state=entry_state)
        self.chk_separate_plots.config(state=entry_state)
        self.chk_group_stats.config(state=entry_state)
        self.chk_interaction.config(state=entry_state)
        self.chk_ancova.config(state=entry_state)
        self.chk_chow.config(state=entry_state)
        self.chk_nls.config(state=entry_state)

    def _run_interaction_analysis(self, df, dep_var, indep_var, group_var, groups, format_number, apply_ln_x=False, apply_ln_y=False):
        """
        Ejecuta el análisis de regresión con término de interacción.
        
        Modelo: Y = β₀ + β₁(X) + β₂(G) + β₃(X·G) + ε
        
        Donde:
        - β₀: Intercepto para el grupo de referencia
        - β₁: Pendiente para el grupo de referencia  
        - β₂: Diferencia en intercepto entre grupos
        - β₃: Diferencia en pendientes (término de interacción)
        
        Si β₃ es significativo (p < 0.05), las pendientes son diferentes entre grupos.
        """
        # Nombres de variables para mostrar
        dep_display = f"ln({dep_var})" if apply_ln_y else dep_var
        indep_display = f"ln({indep_var})" if apply_ln_x else indep_var
        
        results_text = []
        results_text.append("\n" + "=" * 70)
        results_text.append("ANÁLISIS DE INTERACCIÓN (Comparación Formal de Pendientes)")
        results_text.append("=" * 70)
        results_text.append(f"\nModelo: {dep_display} = β₀ + β₁({indep_display}) + β₂({group_var}) + β₃({indep_display}·{group_var})")
        if apply_ln_x or apply_ln_y:
            results_text.append(f"[Transformaciones aplicadas: {'ln(X) ' if apply_ln_x else ''}{'ln(Y)' if apply_ln_y else ''}]")
        results_text.append("")
        
        try:
            # Preparar datos
            temp_df = df[[dep_var, indep_var, group_var]].dropna().copy()
            
            # Aplicar transformaciones ln
            if apply_ln_x:
                temp_df = temp_df[temp_df[indep_var] > 0]
                temp_df[indep_var] = np.log(temp_df[indep_var])
            if apply_ln_y:
                temp_df = temp_df[temp_df[dep_var] > 0]
                temp_df[dep_var] = np.log(temp_df[dep_var])
            
            if len(groups) != 2:
                results_text.append(f"⚠ Se requieren exactamente 2 grupos para este análisis.")
                results_text.append(f"  Grupos recibidos: {groups}")
                return "\n".join(results_text), None
            
            # Filtrar solo los grupos seleccionados
            temp_df = temp_df[temp_df[group_var].astype(str).isin([str(g) for g in groups])]
            
            results_text.append(f"Grupos comparados: {groups[0]} vs {groups[1]}")
            
            if temp_df.shape[0] < 10:
                results_text.append(f"⚠ Datos insuficientes para el análisis (n={temp_df.shape[0]})")
                return "\n".join(results_text)
            
            # Crear variable dummy para el grupo (0 = grupo referencia, 1 = otro grupo)
            ref_group = str(groups[0])
            other_group = str(groups[1]) if len(groups) > 1 else None
            
            temp_df['G'] = (temp_df[group_var].astype(str) == str(other_group)).astype(int)
            
            # Crear término de interacción
            X = temp_df[indep_var].values
            Y = temp_df[dep_var].values
            G = temp_df['G'].values
            XG = X * G  # Término de interacción
            
            # Construir matriz de diseño: [1, X, G, X*G]
            design_matrix = sm.add_constant(np.column_stack([X, G, XG]))
            
            # Ajustar modelo
            model = sm.OLS(Y, design_matrix).fit()
            
            # Extraer coeficientes
            beta0, beta1, beta2, beta3 = model.params
            pvals = model.pvalues
            conf_int = model.conf_int()
            
            results_text.append(f"Grupo de referencia: {ref_group}")
            results_text.append(f"Grupo de comparación: {other_group}")
            results_text.append(f"n total: {len(temp_df)}")
            results_text.append("")
            
            # Resultados del modelo
            results_text.append("COEFICIENTES DEL MODELO:")
            results_text.append("-" * 50)
            results_text.append(f"{'Término':<25} {'Coef':>10} {'IC 95%':>20} {'p-valor':>12}")
            results_text.append("-" * 50)
            
            terms = [
                ("β₀ (Intercepto)", beta0, conf_int[0], pvals[0]),
                (f"β₁ ({indep_var})", beta1, conf_int[1], pvals[1]),
                (f"β₂ ({group_var})", beta2, conf_int[2], pvals[2]),
                (f"β₃ (Interacción)", beta3, conf_int[3], pvals[3])
            ]
            
            for name, coef, ci, pval in terms:
                sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
                ci_str = f"[{format_number(ci[0])}, {format_number(ci[1])}]"
                results_text.append(f"{name:<25} {format_number(coef):>10} {ci_str:>20} {format_number(pval):>10} {sig}")
            
            results_text.append("-" * 50)
            results_text.append(f"R² = {format_number(model.rsquared)}  |  R² ajustado = {format_number(model.rsquared_adj)}")
            results_text.append(f"F = {format_number(model.fvalue)}  |  p(F) = {format_number(model.f_pvalue)}")
            results_text.append("")
            
            # Interpretación clínica
            results_text.append("INTERPRETACIÓN:")
            results_text.append("-" * 50)
            
            # Pendientes por grupo
            slope_ref = beta1
            slope_other = beta1 + beta3
            
            results_text.append(f"• Pendiente en {ref_group}: {format_number(slope_ref)}")
            results_text.append(f"• Pendiente en {other_group}: {format_number(slope_other)}")
            results_text.append("")
            
            # Interpretación del término de interacción
            if pvals[3] < 0.05:
                results_text.append(f"✓ El término de INTERACCIÓN ES SIGNIFICATIVO (p = {format_number(pvals[3])})")
                results_text.append(f"  → Las pendientes son DIFERENTES entre los grupos.")
                results_text.append(f"  → Por cada unidad de {indep_var}:")
                results_text.append(f"     - En {ref_group}: {dep_var} cambia {format_number(slope_ref)} unidades")
                results_text.append(f"     - En {other_group}: {dep_var} cambia {format_number(slope_other)} unidades")
                diff_effect = abs(beta3)
                if beta3 < 0:
                    results_text.append(f"  → El efecto de {indep_var} es {format_number(diff_effect)} unidades MÁS AGRESIVO en {other_group}")
                else:
                    results_text.append(f"  → El efecto de {indep_var} es {format_number(diff_effect)} unidades MÁS AGRESIVO en {ref_group}")
            else:
                results_text.append(f"✗ El término de interacción NO es significativo (p = {format_number(pvals[3])})")
                results_text.append(f"  → Las pendientes son PARALELAS (no hay diferencia significativa).")
                results_text.append(f"  → El efecto de {indep_var} sobre {dep_var} es similar en ambos grupos.")
            
            results_text.append("")
            
            # Interpretación del efecto del grupo (β₂)
            if pvals[2] < 0.05:
                results_text.append(f"✓ El efecto del GRUPO ES SIGNIFICATIVO (p = {format_number(pvals[2])})")
                if beta2 < 0:
                    results_text.append(f"  → {other_group} tiene valores de {dep_var} {format_number(abs(beta2))} unidades MENORES que {ref_group}")
                else:
                    results_text.append(f"  → {other_group} tiene valores de {dep_var} {format_number(abs(beta2))} unidades MAYORES que {ref_group}")
            else:
                results_text.append(f"✗ El efecto del grupo NO es significativo (p = {format_number(pvals[2])})")
                results_text.append(f"  → No hay diferencia vertical significativa entre las líneas.")
            
            results_text.append("")
            results_text.append("=" * 70)
            
            return "\n".join(results_text), {
                'model': model,
                'ref_group': ref_group,
                'other_group': other_group,
                'slope_ref': slope_ref,
                'slope_other': slope_other,
                'beta3_pval': pvals[3]
            }
            
        except Exception as e:
            results_text.append(f"Error en análisis de interacción: {e}")
            import traceback
            results_text.append(traceback.format_exc())
            return "\n".join(results_text), None

    def _run_pairwise_interaction(self, df, dep_var, indep_var, group_var, groups, format_number, apply_ln_x=False, apply_ln_y=False):
        """
        Ejecuta análisis de interacción para todas las combinaciones de pares de grupos.
        """
        from itertools import combinations
        
        results_text = []
        results_text.append("\n" + "=" * 70)
        results_text.append("ANÁLISIS DE INTERACCIÓN - COMPARACIONES PAIRWISE")
        results_text.append(f"({len(groups)} grupos: {len(list(combinations(groups, 2)))} comparaciones)")
        results_text.append("=" * 70)
        
        # Tabla resumen
        summary_data = []
        
        for g1, g2 in combinations(groups, 2):
            pair_groups = [g1, g2]
            text, data = self._run_interaction_analysis(
                df, dep_var, indep_var, group_var, pair_groups, format_number,
                apply_ln_x=apply_ln_x, apply_ln_y=apply_ln_y
            )
            
            if data and 'beta3_pval' in data:
                sig = "***" if data['beta3_pval'] < 0.001 else "**" if data['beta3_pval'] < 0.01 else "*" if data['beta3_pval'] < 0.05 else "ns"
                summary_data.append({
                    'comparacion': f"{g1} vs {g2}",
                    'pendiente_1': data.get('slope_ref', 'NA'),
                    'pendiente_2': data.get('slope_other', 'NA'),
                    'p_interaccion': data['beta3_pval'],
                    'sig': sig
                })
        
        # Mostrar tabla resumen
        if summary_data:
            results_text.append("\nRESUMEN DE COMPARACIONES:")
            results_text.append("-" * 80)
            results_text.append(f"{'Comparación':<25} {'Pend.G1':>12} {'Pend.G2':>12} {'p(Interacc)':>15} {'Sig':>6}")
            results_text.append("-" * 80)
            
            for row in summary_data:
                p1 = format_number(row['pendiente_1']) if isinstance(row['pendiente_1'], (int, float)) else row['pendiente_1']
                p2 = format_number(row['pendiente_2']) if isinstance(row['pendiente_2'], (int, float)) else row['pendiente_2']
                results_text.append(f"{row['comparacion']:<25} {p1:>12} {p2:>12} {format_number(row['p_interaccion']):>15} {row['sig']:>6}")
            
            results_text.append("-" * 80)
            results_text.append("Sig: *** p<0.001, ** p<0.01, * p<0.05, ns = no significativo")
            
            # Contar significativos
            n_sig = sum(1 for r in summary_data if r['p_interaccion'] < 0.05)
            results_text.append(f"\n→ {n_sig} de {len(summary_data)} comparaciones tienen pendientes significativamente diferentes (p<0.05)")
        
        return "\n".join(results_text)

    def _run_pairwise_ancova(self, df, dep_var, indep_var, group_var, groups, format_number, apply_ln_x=False, apply_ln_y=False):
        """
        Ejecuta ANCOVA para todas las combinaciones de pares de grupos.
        """
        from itertools import combinations
        
        results_text = []
        results_text.append("\n" + "=" * 70)
        results_text.append("ANCOVA - COMPARACIONES PAIRWISE")
        results_text.append(f"({len(groups)} grupos: {len(list(combinations(groups, 2)))} comparaciones)")
        results_text.append("=" * 70)
        
        summary_data = []
        
        for g1, g2 in combinations(groups, 2):
            pair_groups = [g1, g2]
            text, data = self._run_ancova(
                df, dep_var, indep_var, group_var, pair_groups, format_number,
                apply_ln_x=apply_ln_x, apply_ln_y=apply_ln_y
            )
            
            if data and 'group_pval' in data:
                sig = "***" if data['group_pval'] < 0.001 else "**" if data['group_pval'] < 0.01 else "*" if data['group_pval'] < 0.05 else "ns"
                summary_data.append({
                    'comparacion': f"{g1} vs {g2}",
                    'media_aj_1': data.get('adj_mean_ref', 'NA'),
                    'media_aj_2': data.get('adj_mean_other', 'NA'),
                    'diferencia': data.get('diff_adj_means', 'NA'),
                    'p_grupo': data['group_pval'],
                    'sig': sig
                })
        
        if summary_data:
            results_text.append("\nRESUMEN DE COMPARACIONES:")
            results_text.append("-" * 90)
            results_text.append(f"{'Comparación':<25} {'Media Aj.G1':>12} {'Media Aj.G2':>12} {'Diferencia':>12} {'p(Grupo)':>12} {'Sig':>6}")
            results_text.append("-" * 90)
            
            for row in summary_data:
                m1 = format_number(row['media_aj_1']) if isinstance(row['media_aj_1'], (int, float)) else row['media_aj_1']
                m2 = format_number(row['media_aj_2']) if isinstance(row['media_aj_2'], (int, float)) else row['media_aj_2']
                diff = format_number(row['diferencia']) if isinstance(row['diferencia'], (int, float)) else row['diferencia']
                results_text.append(f"{row['comparacion']:<25} {m1:>12} {m2:>12} {diff:>12} {format_number(row['p_grupo']):>12} {row['sig']:>6}")
            
            results_text.append("-" * 90)
            results_text.append("Sig: *** p<0.001, ** p<0.01, * p<0.05, ns = no significativo")
            
            n_sig = sum(1 for r in summary_data if r['p_grupo'] < 0.05)
            results_text.append(f"\n→ {n_sig} de {len(summary_data)} comparaciones tienen medias ajustadas significativamente diferentes (p<0.05)")
        
        return "\n".join(results_text)

    def _run_pairwise_chow(self, df, dep_var, indep_var, group_var, groups, format_number, apply_ln_x=False, apply_ln_y=False):
        """
        Ejecuta prueba de Chow para todas las combinaciones de pares de grupos.
        """
        from itertools import combinations
        
        results_text = []
        results_text.append("\n" + "=" * 70)
        results_text.append("PRUEBA DE CHOW - COMPARACIONES PAIRWISE")
        results_text.append(f"({len(groups)} grupos: {len(list(combinations(groups, 2)))} comparaciones)")
        results_text.append("=" * 70)
        
        summary_data = []
        
        for g1, g2 in combinations(groups, 2):
            pair_groups = [g1, g2]
            text, data = self._run_chow_test(
                df, dep_var, indep_var, group_var, pair_groups, format_number,
                apply_ln_x=apply_ln_x, apply_ln_y=apply_ln_y
            )
            
            if data and 'p_value' in data:
                sig = "***" if data['p_value'] < 0.001 else "**" if data['p_value'] < 0.01 else "*" if data['p_value'] < 0.05 else "ns"
                summary_data.append({
                    'comparacion': f"{g1} vs {g2}",
                    'n1': data.get('n1', 'NA'),
                    'n2': data.get('n2', 'NA'),
                    'F_stat': data.get('F_stat', 'NA'),
                    'p_value': data['p_value'],
                    'sig': sig
                })
        
        if summary_data:
            results_text.append("\nRESUMEN DE COMPARACIONES:")
            results_text.append("-" * 80)
            results_text.append(f"{'Comparación':<25} {'n(G1)':>8} {'n(G2)':>8} {'F':>12} {'p-valor':>12} {'Sig':>6}")
            results_text.append("-" * 80)
            
            for row in summary_data:
                f_stat = format_number(row['F_stat']) if isinstance(row['F_stat'], (int, float)) else row['F_stat']
                results_text.append(f"{row['comparacion']:<25} {row['n1']:>8} {row['n2']:>8} {f_stat:>12} {format_number(row['p_value']):>12} {row['sig']:>6}")
            
            results_text.append("-" * 80)
            results_text.append("Sig: *** p<0.001, ** p<0.01, * p<0.05, ns = no significativo")
            
            n_sig = sum(1 for r in summary_data if r['p_value'] < 0.05)
            results_text.append(f"\n→ {n_sig} de {len(summary_data)} comparaciones requieren modelos separados (ruptura estructural, p<0.05)")
        
        return "\n".join(results_text)

    def _run_pairwise_nls(self, df, dep_var, indep_var, group_var, groups, format_number, apply_ln_x=False, apply_ln_y=False):
        """
        Ejecuta comparación NLS para todas las combinaciones de pares de grupos.
        """
        from itertools import combinations
        
        results_text = []
        results_text.append("\n" + "=" * 70)
        results_text.append("COMPARACIÓN NLS (AIC/BIC) - COMPARACIONES PAIRWISE")
        results_text.append(f"({len(groups)} grupos: {len(list(combinations(groups, 2)))} comparaciones)")
        results_text.append("=" * 70)
        
        summary_data = []
        
        for g1, g2 in combinations(groups, 2):
            pair_groups = [g1, g2]
            text, data = self._run_nls_comparison(
                df, dep_var, indep_var, group_var, pair_groups, format_number,
                apply_ln_x=apply_ln_x, apply_ln_y=apply_ln_y
            )
            
            if data and 'best_model' in data:
                best = data['best_model']
                summary_data.append({
                    'comparacion': f"{g1} vs {g2}",
                    'mejor_modelo': best.get('model_name', 'NA'),
                    'delta_AIC': best.get('delta_aic', 'NA'),
                    'LRT_pval': best.get('p_value_lr', 'NA')
                })
        
        if summary_data:
            results_text.append("\nRESUMEN DE COMPARACIONES (mejor modelo por par):")
            results_text.append("-" * 80)
            results_text.append(f"{'Comparación':<25} {'Mejor Modelo':>20} {'ΔAIC':>12} {'p(LRT)':>12}")
            results_text.append("-" * 80)
            
            for row in summary_data:
                delta = format_number(row['delta_AIC']) if isinstance(row['delta_AIC'], (int, float)) else str(row['delta_AIC'])
                lrt = format_number(row['LRT_pval']) if isinstance(row['LRT_pval'], (int, float)) else str(row['LRT_pval'])
                sig = "*" if isinstance(row['LRT_pval'], (int, float)) and row['LRT_pval'] < 0.05 else ""
                results_text.append(f"{row['comparacion']:<25} {str(row['mejor_modelo']):>20} {delta:>12} {lrt:>10}{sig}")
            
            results_text.append("-" * 80)
            results_text.append("ΔAIC > 0: Modelos separados son mejores | * p(LRT) < 0.05: Diferencia significativa")
            
            # Contar pares que requieren modelos separados
            n_sep = sum(1 for r in summary_data if isinstance(r['LRT_pval'], (int, float)) and r['LRT_pval'] < 0.05)
            results_text.append(f"\n→ {n_sep} de {len(summary_data)} pares requieren modelos separados (p<0.05)")
        
        return "\n".join(results_text)

    def _run_ancova(self, df, dep_var, indep_var, group_var, groups, format_number, apply_ln_x=False, apply_ln_y=False):
        """
        Ejecuta ANCOVA (Análisis de Covarianza).
        
        Modelo de efectos principales (asume pendientes paralelas):
        Y = β₀ + β₁(X) + β₂(G) + ε
        
        Compara las medias ajustadas de Y entre grupos, controlando por la covariable X.
        REQUISITO: Las pendientes deben ser paralelas (verificar con análisis de interacción).
        """
        # Nombres de variables para mostrar
        dep_display = f"ln({dep_var})" if apply_ln_y else dep_var
        indep_display = f"ln({indep_var})" if apply_ln_x else indep_var
        
        results_text = []
        results_text.append("\n" + "=" * 70)
        results_text.append("ANCOVA (Análisis de Covarianza)")
        results_text.append("Comparación de medias ajustadas controlando por covariable")
        results_text.append("=" * 70)
        results_text.append(f"\nModelo: {dep_display} = β₀ + β₁({indep_display}) + β₂({group_var})")
        results_text.append("(Asume pendientes PARALELAS entre grupos)")
        if apply_ln_x or apply_ln_y:
            results_text.append(f"[Transformaciones aplicadas: {'ln(X) ' if apply_ln_x else ''}{'ln(Y)' if apply_ln_y else ''}]")
        results_text.append("")
        
        try:
            # Preparar datos
            temp_df = df[[dep_var, indep_var, group_var]].dropna().copy()
            
            # Aplicar transformaciones ln
            if apply_ln_x:
                temp_df = temp_df[temp_df[indep_var] > 0]
                temp_df[indep_var] = np.log(temp_df[indep_var])
            if apply_ln_y:
                temp_df = temp_df[temp_df[dep_var] > 0]
                temp_df[dep_var] = np.log(temp_df[dep_var])
            
            if len(groups) != 2:
                results_text.append(f"⚠ Se requieren exactamente 2 grupos para este análisis.")
                results_text.append(f"  Grupos recibidos: {groups}")
                return "\n".join(results_text), None
            
            # Filtrar solo los grupos seleccionados
            temp_df = temp_df[temp_df[group_var].astype(str).isin([str(g) for g in groups])]
            
            results_text.append(f"Grupos comparados: {groups[0]} vs {groups[1]}")
            
            if temp_df.shape[0] < 10:
                results_text.append(f"⚠ Datos insuficientes para el análisis (n={temp_df.shape[0]})")
                return "\n".join(results_text), None
            
            # Crear variable dummy para el grupo
            ref_group = str(groups[0])
            other_group = str(groups[1]) if len(groups) > 1 else None
            
            temp_df['G'] = (temp_df[group_var].astype(str) == str(other_group)).astype(int)
            
            X = temp_df[indep_var].values
            Y = temp_df[dep_var].values
            G = temp_df['G'].values
            
            # Modelo ANCOVA (sin interacción): [1, X, G]
            design_matrix = sm.add_constant(np.column_stack([X, G]))
            model = sm.OLS(Y, design_matrix).fit()
            
            beta0, beta1, beta2 = model.params
            pvals = model.pvalues
            conf_int = model.conf_int()
            
            # Calcular medias ajustadas
            X_mean = X.mean()
            adjusted_mean_ref = beta0 + beta1 * X_mean
            adjusted_mean_other = beta0 + beta1 * X_mean + beta2
            
            # Medias crudas para comparación
            raw_mean_ref = temp_df[temp_df['G'] == 0][dep_var].mean()
            raw_mean_other = temp_df[temp_df['G'] == 1][dep_var].mean()
            
            n_ref = (temp_df['G'] == 0).sum()
            n_other = (temp_df['G'] == 1).sum()
            
            results_text.append(f"Grupo de referencia: {ref_group} (n={n_ref})")
            results_text.append(f"Grupo de comparación: {other_group} (n={n_other})")
            results_text.append(f"Covariable ({indep_var}) media: {format_number(X_mean)}")
            results_text.append("")
            
            # Resultados del modelo
            results_text.append("COEFICIENTES DEL MODELO ANCOVA:")
            results_text.append("-" * 50)
            results_text.append(f"{'Término':<25} {'Coef':>10} {'IC 95%':>20} {'p-valor':>12}")
            results_text.append("-" * 50)
            
            terms = [
                ("β₀ (Intercepto)", beta0, conf_int[0], pvals[0]),
                (f"β₁ ({indep_var})", beta1, conf_int[1], pvals[1]),
                (f"β₂ ({group_var})", beta2, conf_int[2], pvals[2])
            ]
            
            for name, coef, ci, pval in terms:
                sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
                ci_str = f"[{format_number(ci[0])}, {format_number(ci[1])}]"
                results_text.append(f"{name:<25} {format_number(coef):>10} {ci_str:>20} {format_number(pval):>10} {sig}")
            
            results_text.append("-" * 50)
            results_text.append(f"R² = {format_number(model.rsquared)}  |  R² ajustado = {format_number(model.rsquared_adj)}")
            results_text.append(f"F = {format_number(model.fvalue)}  |  p(F) = {format_number(model.f_pvalue)}")
            results_text.append("")
            
            # Medias ajustadas vs crudas
            results_text.append("COMPARACIÓN DE MEDIAS:")
            results_text.append("-" * 50)
            results_text.append(f"{'Grupo':<20} {'Media Cruda':>15} {'Media Ajustada':>15}")
            results_text.append("-" * 50)
            results_text.append(f"{ref_group:<20} {format_number(raw_mean_ref):>15} {format_number(adjusted_mean_ref):>15}")
            results_text.append(f"{other_group:<20} {format_number(raw_mean_other):>15} {format_number(adjusted_mean_other):>15}")
            results_text.append("-" * 50)
            results_text.append(f"{'Diferencia':<20} {format_number(raw_mean_other - raw_mean_ref):>15} {format_number(beta2):>15}")
            results_text.append("")
            
            # Interpretación
            results_text.append("INTERPRETACIÓN:")
            results_text.append("-" * 50)
            results_text.append(f"• Pendiente común: {format_number(beta1)} (asumiendo líneas paralelas)")
            results_text.append(f"• Por cada unidad de {indep_var}, {dep_var} cambia {format_number(beta1)} unidades")
            results_text.append("")
            
            if pvals[2] < 0.05:
                results_text.append(f"✓ La diferencia entre grupos ES SIGNIFICATIVA (p = {format_number(pvals[2])})")
                results_text.append(f"  → Controlando por {indep_var}:")
                if beta2 < 0:
                    results_text.append(f"     {other_group} tiene valores de {dep_var} {format_number(abs(beta2))} unidades MENORES")
                else:
                    results_text.append(f"     {other_group} tiene valores de {dep_var} {format_number(abs(beta2))} unidades MAYORES")
                results_text.append(f"  → Si todos los sujetos tuvieran el mismo nivel de {indep_var},")
                results_text.append(f"     la diferencia en {dep_var} entre grupos sería {format_number(abs(beta2))} unidades.")
            else:
                results_text.append(f"✗ La diferencia entre grupos NO es significativa (p = {format_number(pvals[2])})")
                results_text.append(f"  → Después de controlar por {indep_var}, no hay diferencia significativa")
                results_text.append(f"     en {dep_var} entre los grupos.")
            
            results_text.append("")
            results_text.append("NOTA: Este análisis asume que las pendientes son paralelas.")
            results_text.append("      Verifique con el análisis de interacción que β₃ ≈ 0.")
            results_text.append("=" * 70)
            
            return "\n".join(results_text), {
                'model': model,
                'adj_mean_ref': adjusted_mean_ref,
                'adj_mean_other': adjusted_mean_other,
                'diff_adj_means': beta2,
                'group_pval': pvals[2]
            }
            
        except Exception as e:
            results_text.append(f"Error en ANCOVA: {e}")
            import traceback
            results_text.append(traceback.format_exc())
            return "\n".join(results_text), None

    def _run_chow_test(self, df, dep_var, indep_var, group_var, groups, format_number, apply_ln_x=False, apply_ln_y=False):
        """
        Ejecuta la Prueba de Chow para detectar ruptura estructural.
        
        Compara la bondad de ajuste de:
        - Un modelo combinado (pooled) con todos los datos
        - Dos modelos separados, uno por cada grupo
        
        Estadístico F:
        F = [(RSS_pool - (RSS_1 + RSS_2)) / k] / [(RSS_1 + RSS_2) / (N₁ + N₂ - 2k)]
        
        Donde k = número de parámetros (2 para regresión lineal simple)
        """
        # Nombres de variables para mostrar
        dep_display = f"ln({dep_var})" if apply_ln_y else dep_var
        indep_display = f"ln({indep_var})" if apply_ln_x else indep_var
        
        results_text = []
        results_text.append("\n" + "=" * 70)
        results_text.append("PRUEBA DE CHOW (Ruptura Estructural)")
        results_text.append("¿Son necesarios modelos separados para cada grupo?")
        results_text.append("=" * 70)
        if apply_ln_x or apply_ln_y:
            results_text.append(f"[Transformaciones aplicadas: {'ln(X) ' if apply_ln_x else ''}{'ln(Y)' if apply_ln_y else ''}]")
        results_text.append("")
        
        try:
            # Preparar datos
            temp_df = df[[dep_var, indep_var, group_var]].dropna().copy()
            
            # Aplicar transformaciones ln
            if apply_ln_x:
                temp_df = temp_df[temp_df[indep_var] > 0]
                temp_df[indep_var] = np.log(temp_df[indep_var])
            if apply_ln_y:
                temp_df = temp_df[temp_df[dep_var] > 0]
                temp_df[dep_var] = np.log(temp_df[dep_var])
            
            if len(groups) != 2:
                results_text.append(f"⚠ Se requieren exactamente 2 grupos para este análisis.")
                results_text.append(f"  Grupos recibidos: {groups}")
                return "\n".join(results_text), None
            
            # Filtrar solo los grupos seleccionados
            temp_df = temp_df[temp_df[group_var].astype(str).isin([str(g) for g in groups])]
            
            results_text.append(f"Grupos comparados: {groups[0]} vs {groups[1]}")
            
            group1_name = str(groups[0])
            group2_name = str(groups[1]) if len(groups) > 1 else None
            
            df1 = temp_df[temp_df[group_var].astype(str) == group1_name]
            df2 = temp_df[temp_df[group_var].astype(str) == group2_name]
            
            n1 = len(df1)
            n2 = len(df2)
            k = 2  # Número de parámetros (intercepto + pendiente)
            
            if n1 < k + 1 or n2 < k + 1:
                results_text.append(f"⚠ Datos insuficientes (n1={n1}, n2={n2}, k={k})")
                return "\n".join(results_text), None
            
            results_text.append(f"Grupo 1: {group1_name} (n={n1})")
            results_text.append(f"Grupo 2: {group2_name} (n={n2})")
            results_text.append(f"Total: n={n1 + n2}")
            results_text.append(f"Parámetros por modelo: k={k}")
            results_text.append("")
            
            # Modelo combinado (pooled)
            X_pool = sm.add_constant(temp_df[indep_var].values)
            Y_pool = temp_df[dep_var].values
            model_pool = sm.OLS(Y_pool, X_pool).fit()
            RSS_pool = model_pool.ssr  # Sum of Squared Residuals
            
            # Modelo para grupo 1
            X1 = sm.add_constant(df1[indep_var].values)
            Y1 = df1[dep_var].values
            model1 = sm.OLS(Y1, X1).fit()
            RSS1 = model1.ssr
            
            # Modelo para grupo 2
            X2 = sm.add_constant(df2[indep_var].values)
            Y2 = df2[dep_var].values
            model2 = sm.OLS(Y2, X2).fit()
            RSS2 = model2.ssr
            
            # Calcular estadístico F de Chow
            numerator = (RSS_pool - (RSS1 + RSS2)) / k
            denominator = (RSS1 + RSS2) / (n1 + n2 - 2 * k)
            F_chow = numerator / denominator
            
            # Calcular p-valor
            df_num = k
            df_denom = n1 + n2 - 2 * k
            p_value = 1 - stats.f.cdf(F_chow, df_num, df_denom)
            
            # Resultados de los modelos
            results_text.append("MODELOS DE REGRESIÓN:")
            results_text.append("-" * 60)
            
            # Modelo pooled
            results_text.append(f"\n1. MODELO COMBINADO (todos los datos):")
            results_text.append(f"   {dep_var} = {format_number(model_pool.params[0])} + {format_number(model_pool.params[1])}·{indep_var}")
            results_text.append(f"   R² = {format_number(model_pool.rsquared)}")
            results_text.append(f"   RSS_pool = {format_number(RSS_pool)}")
            
            # Modelo grupo 1
            results_text.append(f"\n2. MODELO {group1_name}:")
            results_text.append(f"   {dep_var} = {format_number(model1.params[0])} + {format_number(model1.params[1])}·{indep_var}")
            results_text.append(f"   R² = {format_number(model1.rsquared)}")
            results_text.append(f"   RSS_1 = {format_number(RSS1)}")
            
            # Modelo grupo 2
            results_text.append(f"\n3. MODELO {group2_name}:")
            results_text.append(f"   {dep_var} = {format_number(model2.params[0])} + {format_number(model2.params[1])}·{indep_var}")
            results_text.append(f"   R² = {format_number(model2.rsquared)}")
            results_text.append(f"   RSS_2 = {format_number(RSS2)}")
            
            results_text.append("")
            results_text.append("-" * 60)
            results_text.append("PRUEBA DE CHOW:")
            results_text.append("-" * 60)
            results_text.append(f"RSS_pool = {format_number(RSS_pool)}")
            results_text.append(f"RSS_1 + RSS_2 = {format_number(RSS1 + RSS2)}")
            results_text.append(f"Reducción en RSS = {format_number(RSS_pool - (RSS1 + RSS2))}")
            results_text.append("")
            results_text.append(f"F = [(RSS_pool - (RSS_1+RSS_2))/k] / [(RSS_1+RSS_2)/(N-2k)]")
            results_text.append(f"F = [{format_number(RSS_pool - (RSS1 + RSS2))}/{k}] / [{format_number(RSS1 + RSS2)}/{df_denom}]")
            results_text.append(f"F = {format_number(numerator)} / {format_number(denominator)}")
            results_text.append(f"F = {format_number(F_chow)}")
            results_text.append(f"Grados de libertad: ({df_num}, {df_denom})")
            results_text.append(f"p-valor = {format_number(p_value)}")
            
            sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
            results_text.append(f"Significancia: {sig}" if sig else "No significativo")
            results_text.append("")
            
            # Interpretación
            results_text.append("INTERPRETACIÓN:")
            results_text.append("-" * 60)
            
            if p_value < 0.05:
                results_text.append(f"✓ HAY RUPTURA ESTRUCTURAL (p = {format_number(p_value)})")
                results_text.append(f"  → Los datos de {group1_name} y {group2_name} NO pueden explicarse")
                results_text.append(f"     con la misma ecuación de regresión.")
                results_text.append(f"  → Se recomienda usar MODELOS SEPARADOS para cada grupo.")
                results_text.append(f"  → La diferencia puede ser en:")
                results_text.append(f"     • La pendiente (tasa de cambio)")
                results_text.append(f"     • El intercepto (nivel base)")
                results_text.append(f"     • Ambos")
                results_text.append("")
                results_text.append("  Use el análisis de INTERACCIÓN para determinar")
                results_text.append("  si la diferencia está en las pendientes.")
            else:
                results_text.append(f"✗ NO hay ruptura estructural (p = {format_number(p_value)})")
                results_text.append(f"  → Un modelo combinado es suficiente para ambos grupos.")
                results_text.append(f"  → La relación entre {indep_var} y {dep_var} es similar")
                results_text.append(f"     en {group1_name} y {group2_name}.")
            
            results_text.append("")
            results_text.append("=" * 70)
            
            return "\n".join(results_text), {
                'F_stat': F_chow,
                'p_value': p_value,
                'n1': n1,
                'n2': n2,
                'RSS_pool': RSS_pool,
                'RSS1': RSS1,
                'RSS2': RSS2,
                'model_pool': model_pool,
                'model1': model1,
                'model2': model2
            }
            
        except Exception as e:
            results_text.append(f"Error en Prueba de Chow: {e}")
            import traceback
            results_text.append(traceback.format_exc())
            return "\n".join(results_text), None

    def _run_nls_comparison(self, df, dep_var, indep_var, group_var, groups, format_number, apply_ln_x=False, apply_ln_y=False):
        """
        Comparación de modelos de Regresión No Lineal (NLS).
        
        Compara múltiples modelos no lineales entre grupos usando:
        - AIC (Akaike Information Criterion)
        - BIC (Bayesian Information Criterion)
        - Likelihood Ratio Test aproximado
        
        Modelos soportados:
        - Potencia: Y = a · X^b
        - Exponencial: Y = a · e^(b·X)
        - Logarítmico: Y = a + b·ln(X)
        - Sigmoide: Y = L / (1 + e^(-k·(X-x0))) + c
        """
        from scipy.optimize import curve_fit
        
        # Nombres de variables para mostrar
        dep_display = f"ln({dep_var})" if apply_ln_y else dep_var
        indep_display = f"ln({indep_var})" if apply_ln_x else indep_var
        
        results_text = []
        results_text.append("\n" + "=" * 70)
        results_text.append("COMPARACIÓN DE MODELOS NO LINEALES (NLS)")
        results_text.append("AIC, BIC y Likelihood Ratio Test")
        results_text.append("=" * 70)
        if apply_ln_x or apply_ln_y:
            results_text.append(f"[Transformaciones aplicadas: {'ln(X) ' if apply_ln_x else ''}{'ln(Y)' if apply_ln_y else ''}]")
        results_text.append("")
        
        # Definir modelos no lineales
        def power_model(x, a, b):
            """Y = a · X^b"""
            return a * np.power(x, b)
        
        def exp_model(x, a, b):
            """Y = a · e^(b·X)"""
            return a * np.exp(b * x)
        
        def log_model(x, a, b):
            """Y = a + b·ln(X)"""
            return a + b * np.log(x)
        
        def sigmoid_model(x, L, k, x0, c):
            """Y = L / (1 + e^(-k·(X-x0))) + c"""
            return L / (1 + np.exp(-k * (x - x0))) + c
        
        def exp_decay_model(x, a, b):
            """Y = a · e^(-b·X)"""
            return a * np.exp(-b * x)
        
        # Lista de modelos a probar (ajustar según transformaciones)
        # Si ya transformamos ln, algunos modelos cambian su interpretación
        models_config = [
            ("Potencia (Y=a·X^b)", power_model, [100.0, -1.0], 2, True, False),  # (nombre, func, p0, k, need_pos_x, need_pos_y)
            ("Exponencial (Y=a·e^(bX))", exp_model, [50.0, -0.05], 2, False, True),
            ("Logarítmico (Y=a+b·ln(X))", log_model, [50.0, -10.0], 2, True, False),
            ("Exp Decreciente (Y=a·e^(-bX))", exp_decay_model, [100.0, 0.05], 2, False, True),
        ]
        
        try:
            # Preparar datos
            temp_df = df[[dep_var, indep_var, group_var]].dropna().copy()
            
            # Aplicar transformaciones ln ANTES de ajustar modelos
            if apply_ln_x:
                temp_df = temp_df[temp_df[indep_var] > 0]
                temp_df[indep_var] = np.log(temp_df[indep_var])
            if apply_ln_y:
                temp_df = temp_df[temp_df[dep_var] > 0]
                temp_df[dep_var] = np.log(temp_df[dep_var])
            
            if len(groups) != 2:
                results_text.append(f"⚠ Se requieren exactamente 2 grupos para este análisis.")
                results_text.append(f"  Grupos recibidos: {groups}")
                return "\n".join(results_text), None
            
            # Filtrar solo los grupos seleccionados
            temp_df = temp_df[temp_df[group_var].astype(str).isin([str(g) for g in groups])]
            
            results_text.append(f"Grupos comparados: {groups[0]} vs {groups[1]}")
            
            group1_name = str(groups[0])
            group2_name = str(groups[1]) if len(groups) > 1 else None
            
            df1 = temp_df[temp_df[group_var].astype(str) == group1_name]
            df2 = temp_df[temp_df[group_var].astype(str) == group2_name]
            
            n1_orig = len(df1)
            n2_orig = len(df2)
            
            results_text.append(f"Grupo 1: {group1_name} (n={n1_orig})")
            results_text.append(f"Grupo 2: {group2_name} (n={n2_orig})")
            results_text.append("")
            
            # Función para calcular AIC y BIC
            def calc_aic_bic(y_true, y_pred, k, n):
                """Calcula AIC y BIC dado los residuos"""
                rss = np.sum((y_true - y_pred) ** 2)
                if rss <= 0 or n <= k:
                    return np.inf, np.inf, -np.inf, rss
                sigma2 = rss / n
                ll = -n/2 * np.log(2*np.pi) - n/2 * np.log(sigma2) - rss/(2*sigma2)
                aic = 2*k - 2*ll
                bic = k*np.log(n) - 2*ll
                return aic, bic, ll, rss
            
            # Tabla de resultados
            all_model_results = []
            
            for model_name, model_func, p0, k, need_pos_x, need_pos_y in models_config:
                try:
                    # Filtrar datos según requerimientos del modelo
                    df1_filt = df1.copy()
                    df2_filt = df2.copy()
                    temp_df_filt = temp_df.copy()
                    
                    if need_pos_x:
                        df1_filt = df1_filt[df1_filt[indep_var] > 0]
                        df2_filt = df2_filt[df2_filt[indep_var] > 0]
                        temp_df_filt = temp_df_filt[temp_df_filt[indep_var] > 0]
                    if need_pos_y:
                        df1_filt = df1_filt[df1_filt[dep_var] > 0]
                        df2_filt = df2_filt[df2_filt[dep_var] > 0]
                        temp_df_filt = temp_df_filt[temp_df_filt[dep_var] > 0]
                    
                    x1, y1 = df1_filt[indep_var].values, df1_filt[dep_var].values
                    x2, y2 = df2_filt[indep_var].values, df2_filt[dep_var].values
                    x_pool = temp_df_filt[indep_var].values
                    y_pool = temp_df_filt[dep_var].values
                    
                    n1, n2 = len(x1), len(x2)
                    n_pool = len(x_pool)
                    
                    if n1 < k + 2 or n2 < k + 2:
                        continue
                    
                    # Ajustar modelo combinado
                    try:
                        popt_pool, _ = curve_fit(model_func, x_pool, y_pool, p0=p0, maxfev=10000)
                        y_pred_pool = model_func(x_pool, *popt_pool)
                        aic_pool, bic_pool, ll_pool, rss_pool = calc_aic_bic(y_pool, y_pred_pool, k, n_pool)
                    except:
                        continue
                    
                    # Ajustar modelo grupo 1
                    try:
                        popt1, _ = curve_fit(model_func, x1, y1, p0=p0, maxfev=10000)
                        y_pred1 = model_func(x1, *popt1)
                        aic1, bic1, ll1, rss1 = calc_aic_bic(y1, y_pred1, k, n1)
                        ss_tot1 = np.sum((y1 - np.mean(y1)) ** 2)
                        r2_1 = 1 - rss1/ss_tot1 if ss_tot1 > 0 else 0
                    except:
                        continue
                    
                    # Ajustar modelo grupo 2
                    try:
                        popt2, _ = curve_fit(model_func, x2, y2, p0=p0, maxfev=10000)
                        y_pred2 = model_func(x2, *popt2)
                        aic2, bic2, ll2, rss2 = calc_aic_bic(y2, y_pred2, k, n2)
                        ss_tot2 = np.sum((y2 - np.mean(y2)) ** 2)
                        r2_2 = 1 - rss2/ss_tot2 if ss_tot2 > 0 else 0
                    except:
                        continue
                    
                    # Calcular métricas combinadas
                    ll_sep = ll1 + ll2
                    aic_sep = aic1 + aic2
                    bic_sep = bic1 + bic2
                    
                    # Likelihood Ratio Test
                    lr_stat = 2 * (ll_sep - ll_pool)
                    p_value_lr = 1 - stats.chi2.cdf(lr_stat, k) if lr_stat > 0 else 1.0
                    
                    delta_aic = aic_pool - aic_sep
                    delta_bic = bic_pool - bic_sep
                    
                    all_model_results.append({
                        'model_name': model_name,
                        'popt_pool': popt_pool,
                        'popt1': popt1, 'popt2': popt2,
                        'r2_1': r2_1, 'r2_2': r2_2,
                        'aic_pool': aic_pool, 'aic_sep': aic_sep,
                        'bic_pool': bic_pool, 'bic_sep': bic_sep,
                        'delta_aic': delta_aic, 'delta_bic': delta_bic,
                        'lr_stat': lr_stat, 'p_value_lr': p_value_lr,
                        'n1': n1, 'n2': n2, 'k': k
                    })
                    
                except Exception as e:
                    self.log_message(f"Error en modelo {model_name}: {e}", "WARN")
                    continue
            
            if not all_model_results:
                results_text.append("⚠ No se pudo ajustar ningún modelo no lineal.")
                return "\n".join(results_text), None
            
            # Mostrar tabla resumen de todos los modelos
            results_text.append("=" * 70)
            results_text.append("RESUMEN: COMPARACIÓN DE MODELOS POR GRUPO")
            results_text.append("=" * 70)
            results_text.append("")
            results_text.append(f"{'Modelo':<30} {'AIC comb':>10} {'AIC sep':>10} {'ΔAIC':>8} {'p(LRT)':>10} {'Mejor':>8}")
            results_text.append("-" * 76)
            
            for res in all_model_results:
                mejor = "Separados" if res['delta_aic'] > 2 else "Combinado"
                sig = "*" if res['p_value_lr'] < 0.05 else ""
                results_text.append(
                    f"{res['model_name']:<30} {res['aic_pool']:>10.1f} {res['aic_sep']:>10.1f} "
                    f"{res['delta_aic']:>+8.1f} {res['p_value_lr']:>9.4f}{sig} {mejor:>8}"
                )
            
            results_text.append("-" * 76)
            results_text.append("ΔAIC > 0: Modelos separados son mejores | p(LRT) < 0.05*: Diferencia significativa")
            results_text.append("")
            
            # Detalles por modelo
            results_text.append("=" * 70)
            results_text.append("DETALLE DE PARÁMETROS POR MODELO")
            results_text.append("=" * 70)
            
            for res in all_model_results:
                results_text.append(f"\n--- {res['model_name']} ---")
                results_text.append(f"  Combinado: parámetros = {[format_number(p) for p in res['popt_pool']]}")
                results_text.append(f"  {group1_name}: parámetros = {[format_number(p) for p in res['popt1']]}, R² = {format_number(res['r2_1'])}")
                results_text.append(f"  {group2_name}: parámetros = {[format_number(p) for p in res['popt2']]}, R² = {format_number(res['r2_2'])}")
                
                # Interpretación específica
                if res['p_value_lr'] < 0.05:
                    results_text.append(f"  ✓ Diferencia SIGNIFICATIVA (p = {format_number(res['p_value_lr'])})")
                else:
                    results_text.append(f"  ✗ Sin diferencia significativa (p = {format_number(res['p_value_lr'])})")
            
            # Mejor modelo global (menor AIC combinado)
            results_text.append("")
            results_text.append("=" * 70)
            results_text.append("RECOMENDACIÓN")
            results_text.append("=" * 70)
            
            best_model = min(all_model_results, key=lambda x: x['aic_pool'])
            results_text.append(f"\nMejor modelo (menor AIC combinado): {best_model['model_name']}")
            results_text.append(f"  AIC = {format_number(best_model['aic_pool'])}")
            
            # Verificar si hay diferencia significativa en ese modelo
            if best_model['p_value_lr'] < 0.05:
                results_text.append(f"\n  → Se recomienda usar MODELOS SEPARADOS por grupo")
                results_text.append(f"     (LRT p = {format_number(best_model['p_value_lr'])})")
            else:
                results_text.append(f"\n  → Un modelo COMBINADO puede ser suficiente")
                results_text.append(f"     (LRT p = {format_number(best_model['p_value_lr'])})")
            
            results_text.append("")
            results_text.append("=" * 70)
            
            return "\n".join(results_text), {
                'all_results': all_model_results,
                'best_model': best_model
            }
            
        except Exception as e:
            results_text.append(f"Error en comparación NLS: {e}")
            import traceback
            results_text.append(traceback.format_exc())
            return "\n".join(results_text), None

    def rename_variable(self):
        selected_indices = self.listbox_indep_vars_spec.curselection()
        if not selected_indices:
            messagebox.showwarning("Sin Selección", "Seleccione una variable de la lista para renombrar.")
            return

        if len(selected_indices) > 1:
            messagebox.showwarning("Múltiples Selecciones", "Por favor, seleccione solo una variable para renombrar a la vez.")
            return

        original_name = self.listbox_indep_vars_spec.get(selected_indices[0])
        new_name = self.rename_var_entry.get().strip()

        if not new_name:
            messagebox.showwarning("Nombre Vacío", "Por favor, ingrese un nuevo nombre para la variable.")
            return

        if new_name in self.data.columns and new_name != original_name:
            messagebox.showerror("Nombre Duplicado", f"El nombre '{new_name}' ya existe en el conjunto de datos.")
            return

        # Renombrar en el DataFrame
        self.data.rename(columns={original_name: new_name}, inplace=True)
        self.log_message(f"Variable '{original_name}' renombrada a '{new_name}'.")

        # Actualizar todas las listas de variables en la UI
        self._update_variable_selectors()
        self.rename_var_entry.delete(0, tk.END)

    def open_results_editor(self):
        if not self.results_text_content:
            messagebox.showinfo("Sin Resultados", "Primero genere un análisis para ver y editar los resultados.")
            return

        editor_window = tk.Toplevel(self)
        editor_window.title("Editar Texto de Resultados")
        editor_window.geometry("600x500")

        text_widget = scrolledtext.ScrolledText(editor_window, wrap="word", font=("Courier New", 10))
        text_widget.pack(fill="both", expand=True, padx=10, pady=10)
        text_widget.insert("1.0", self.results_text_content)

        def apply_and_close():
            self.results_text_content = text_widget.get("1.0", tk.END)
            self.show_results_tab()
            editor_window.destroy()

        btn_apply = ttk.Button(editor_window, text="Aplicar y Cerrar", command=apply_and_close)
        btn_apply.pack(pady=10)

    def update_font_styles(self, font_family, font_size):
        """Actualiza la fuente en los widgets de texto de esta pestaña."""
        if hasattr(self, 'txt_results'):
            self.txt_results.config(font=(font_family, font_size))

    def log_message(self, msg, level="INFO"): # Añadido nivel por defecto
        # Actualizar la etiqueta en la GUI
        if hasattr(self, 'msg_label') and self.msg_label:
            self.msg_label.config(text=str(msg))

        # Imprimir también en la consola para depuración más persistente
        print(f"[RegresionesTab - {level.upper()}] {msg}")
        sys.stdout.flush() # Asegurar que se imprima inmediatamente

    def _apply_custom_labels_to_series(self, series, labels_string):
        if not labels_string:
            return series
        
        mapping = {}
        for pair in labels_string.split(","):
            if ":" in pair:
                original, label = pair.split(":", 1)
                try:
                    # Try to convert original to numeric if the series is numeric
                    if pd.api.types.is_numeric_dtype(series):
                        mapping[float(original.strip())] = label.strip()
                    else:
                        mapping[original.strip()] = label.strip()
                except ValueError:
                    mapping[original.strip()] = label.strip()
        
        # Apply mapping. Use .get() with default to keep original if no mapping found
        return series.apply(lambda x: mapping.get(x, x))

    def load_data(self):
        file_path = filedialog.askopenfilename(title="Selecciona archivo Excel",
                                               filetypes=[("Excel files", "*.xlsx *.xls"), ("All files", "*.*")])
        if not file_path:
            self.log_message("Carga cancelada")
            return
        try:
            _, file_extension = os.path.splitext(file_path)
            file_extension = file_extension.lower()

            if file_extension == '.csv':
                df = pd.read_csv(file_path)
            elif file_extension in ['.xls', '.xlsx']:
                df = pd.read_excel(file_path)
            else:
                error_message = f"Tipo de archivo no soportado: {file_extension}. Por favor, seleccione un archivo CSV o Excel."
                self.log_message(error_message)
                messagebox.showerror("Tipo de Archivo No Soportado", error_message)
                return
            
            if df.columns[0].lower().startswith("unnamed: 0") or df.columns[0].lower() == "":
                df = df.iloc[:, 1:]
            
            if "edad de inicio" in df.columns: 
                orig_rows = df.shape[0]
                df = df[df["edad de inicio"] >= 0]
                self.log_message(f"Filtrado 'edad de inicio': {orig_rows-df.shape[0]} descartados de {orig_rows}")
            
            self.data = df
            self.lbl_file.config(text=f"Archivo: {os.path.basename(file_path)} ({df.shape[0]} filas, {df.shape[1]} columnas)")
            
            all_cols = list(df.columns)
            num_cols = list(df.select_dtypes(include=[np.number]).columns)

            # Actualizar componente de filtro
            if hasattr(self, 'filter_component') and self.filter_component:
                try:
                    self.filter_component.set_dataframe(self.data)
                except TypeError as te_filter_comp:
                    self.log_message(f"TypeError específico al llamar a FilterComponent.set_dataframe: {te_filter_comp}")
                    messagebox.showerror("Error de Tipo en Componente Filtro", 
                                         f"Se produjo un error de tipo (argumentos) al configurar el componente de filtro.\n"
                                         f"Detalle: {te_filter_comp}\n\n"
                                         "Esto puede indicar una incompatibilidad o un problema interno en FilterComponent al procesar los datos cargados.")
                    # Decide if you want to re-raise, or if other cleanup is needed.
                    # For now, logging and showing the error is the primary goal for diagnostics.
                    # If this error occurs, subsequent operations relying on filter_component might be affected.
                except Exception as e_filter_comp_other:
                    # Catch other potential errors from set_dataframe too
                    self.log_message(f"Otra excepción al llamar a FilterComponent.set_dataframe: {e_filter_comp_other}")
                    messagebox.showerror("Error en Componente Filtro",
                                         f"Se produjo una excepción general al configurar el componente de filtro.\n"
                                         f"Detalle: {e_filter_comp_other}")

            # Actualizar selectores de variables de regresión
            self.listbox_dep_vars_spec.delete(0, tk.END)
            for col in num_cols:
                self.listbox_dep_vars_spec.insert(tk.END, col)
            if num_cols:
                self.listbox_dep_vars_spec.selection_set(0) # Select the first item by default

            self.listbox_indep_vars_spec.delete(0, tk.END)
            # Call _update_indep_vars_listbox to populate it based on initial dep var selection
            self._update_indep_vars_listbox()
            
            self.log_message("Datos cargados. Configure variables y filtros.")
        except Exception as e:
            self.log_message(f"Error al cargar datos: {e}")
            traceback.print_exc()
            # Add the messagebox here
            messagebox.showerror("Error al Cargar Archivo", 
                                 f"Ocurrió un error detallado al intentar cargar el archivo:\n\n{e}\n\nConsulte la consola para ver el traceback completo si es necesario.")
            self.data = None
            self.lbl_file.config(text="Ningún archivo cargado.")
            self.listbox_dep_vars_spec.delete(0, tk.END) # Clear the listbox
            self.listbox_indep_vars_spec.delete(0, tk.END)
            # Limpiar componente de filtro en caso de error
            if hasattr(self, 'filter_component') and self.filter_component:
                self.filter_component.set_dataframe(None)

    def _update_indep_vars_listbox(self, event=None, current_indep_vars=None):
        if not hasattr(self, 'data') or self.data is None:
            return

        # Get all currently selected dependent variables
        selected_dep_indices = self.listbox_dep_vars_spec.curselection()
        selected_dep_vars = {self.listbox_dep_vars_spec.get(i) for i in selected_dep_indices}

        # If current_indep_vars is not provided (e.g., called from event), get current selection
        if current_indep_vars is None:
            current_indep_selection_indices = self.listbox_indep_vars_spec.curselection()
            current_indep_vars = {self.listbox_indep_vars_spec.get(i) for i in current_indep_selection_indices}

        self.listbox_indep_vars_spec.delete(0, tk.END)

        numeric_cols = list(self.data.select_dtypes(include=[np.number]).columns)
        # Independent variables should not include any selected dependent variables
        available_indep_vars = [col for col in numeric_cols if col not in selected_dep_vars]

        for idx, var_name in enumerate(available_indep_vars):
            self.listbox_indep_vars_spec.insert(tk.END, var_name)
            if var_name in current_indep_vars:
                self.listbox_indep_vars_spec.selection_set(idx)
        if not current_indep_vars and available_indep_vars:
            self.listbox_indep_vars_spec.selection_set(0)

    # Se eliminan apply_filter_criteria y apply_filter_qual

    def _get_filtered_data_for_regression(self):
        """Obtiene los datos filtrados. Usa primero los filtros del Archivo de Trabajo,
        luego aplica filtros adicionales del FilterComponent si existe."""
        if self.data is None:
            self.log_message("Error: No hay datos cargados.")
            return None

        # Paso 1: Usar datos filtrados del Archivo de Trabajo si existen
        if hasattr(self, 'filtered_data') and self.filtered_data is not None:
            df_filtered = self.filtered_data.copy()
            if hasattr(self, 'shared_filter_summary') and self.shared_filter_summary:
                self.log_message(f"Usando datos del Archivo de Trabajo: {df_filtered.shape[0]} filas (filtros: {len(self.shared_filter_summary)})")
            else:
                self.log_message(f"Usando datos del Archivo de Trabajo: {df_filtered.shape[0]} filas.")
        else:
            df_filtered = self.data.copy()
            self.log_message("Usando datos originales (sin filtros del Archivo de Trabajo).")

        # Paso 2: Aplicar filtros adicionales del FilterComponent si existe
        if hasattr(self, 'filter_component') and self.filter_component:
            # El FilterComponent debe trabajar sobre los datos ya filtrados
            self.filter_component.set_dataframe(df_filtered)
            df_filtered = self.filter_component.apply_filters()
            if df_filtered is None:
                self.log_message("Error al aplicar filtros adicionales.")
                return None
            self.log_message(f"Después de filtros adicionales: {df_filtered.shape[0]} filas.")

        if df_filtered.empty:
            self.log_message("No hay datos después de aplicar filtros.")
            return pd.DataFrame()

        return df_filtered

    def _parse_variable_specifications(self, spec_string, df_for_check, is_single_var=False):
        if not spec_string:
            self.log_message("Advertencia: No se especificaron variables.")
            return []
        
        lines = [spec_string.strip()] if is_single_var else [line.strip() for line in spec_string.split("\n") if line.strip()]
        parsed_vars = []
        
        if not lines:
             self.log_message(f"Advertencia: No se especificaron variables {'dependientes' if is_single_var else 'independientes'}.")
             return []

        for line_idx, line in enumerate(lines):
            original_name, display_name = line, line
            if ":" in line:
                parts = line.split(":", 1)
                original_name = parts[0].strip()
                display_name = parts[1].strip()
                if not display_name: display_name = original_name 
            
            if original_name not in df_for_check.columns:
                self.log_message(f"Advertencia: Variable '{original_name}' (línea {line_idx+1}) no encontrada en datos filtrados. Omitida.")
                continue
            
            if not pd.api.types.is_numeric_dtype(df_for_check[original_name]):
                self.log_message(f"Advertencia: Variable '{original_name}' (línea {line_idx+1}) no es numérica en datos filtrados. Omitida para regresión.")
                continue
            
            parsed_vars.append((original_name, display_name))
        
        if not parsed_vars:
             self.log_message("Advertencia: Ninguna variable válida para regresión fue procesada.")
        return parsed_vars

    @staticmethod
    def get_p_values_from_curve_fit(popt, pcov, n):
        p = len(popt)
        dof = max(0, n - p)
        
        if np.isinf(pcov).any():
            return [np.nan] * p

        perr = np.sqrt(np.diag(pcov))
        t_stats = popt / perr
        p_values = [2 * stats.t.sf(np.abs(t), dof) for t in t_stats]
        return p_values

    def calculate_and_show_correlations(self):
        self.log_message("Iniciando calculate_and_show_correlations...", "DEBUG")
        if self.data is None:
            self.log_message("calculate_and_show_correlations: No hay datos cargados. Retornando.", "WARN")
            messagebox.showwarning("Sin datos", "No hay datos cargados en la pestaña Regresiones.", parent=self)
            return

        df_f = self._get_filtered_data_for_regression()
        if df_f is None:
            self.log_message("calculate_and_show_correlations: _get_filtered_data_for_regression devolvió None. Retornando.", "ERROR")
            messagebox.showerror("Error de filtros", "No se pudieron obtener datos filtrados para Regresiones.", parent=self)
            return
        if df_f.empty:
            self.log_message("calculate_and_show_correlations: No hay datos después de aplicar filtros. Retornando.", "WARN")
            messagebox.showwarning("Sin datos", "No hay datos disponibles después de aplicar filtros en Regresiones.", parent=self)
            return

        selected_dep_indices = self.listbox_dep_vars_spec.curselection()
        selected_dep_vars_names = [self.listbox_dep_vars_spec.get(i) for i in selected_dep_indices]
        if not selected_dep_vars_names:
            messagebox.showwarning("Advertencia", "Por favor, seleccione al menos una variable dependiente.", parent=self)
            return

        selected_indep_indices = self.listbox_indep_vars_spec.curselection()
        selected_indep_vars_names = [self.listbox_indep_vars_spec.get(i) for i in selected_indep_indices]
        if not selected_indep_vars_names:
            messagebox.showwarning("Advertencia", "Por favor, seleccione al menos una variable independiente.", parent=self)
            return

        all_results = []

        for dep_var in selected_dep_vars_names:
            for indep_var in selected_indep_vars_names:
                if dep_var == indep_var:
                    continue

                temp_df = df_f[[dep_var, indep_var]].dropna()
                if temp_df.shape[0] < 2:
                    continue

                x = temp_df[indep_var].values
                y = temp_df[dep_var].values

                # Pearson
                pearson_r, pearson_p = safe_pearson(x, y)
                all_results.append({
                    "Variable Dependiente": dep_var,
                    "Variable Independiente": indep_var,
                    "Modelo": "Pearson",
                    "r": pearson_r,
                    "R²": pearson_r**2 if not np.isnan(pearson_r) else np.nan,
                    "P-valor": pearson_p
                })

                # Spearman
                spearman_r, spearman_p = safe_spearman(x, y)
                all_results.append({
                    "Variable Dependiente": dep_var,
                    "Variable Independiente": indep_var,
                    "Modelo": "Spearman",
                    "r": spearman_r,
                    "R²": np.nan, # R² is not typically reported for Spearman
                    "P-valor": spearman_p
                })

                if self.var_linear.get():
                    try:
                        X_lin = sm.add_constant(x)
                        mod = sm.OLS(y, X_lin).fit()
                        yhat = mod.predict(X_lin)
                        r, p_corr = safe_pearson(y, yhat)
                        all_results.append({
                            "Variable Dependiente": dep_var,
                            "Variable Independiente": indep_var,
                            "Modelo": "Lineal",
                            "r": r,
                            "R²": mod.rsquared,
                            "P-valor": mod.f_pvalue
                        })
                    except Exception as e: self.log_message(f"Error en modelo Lineal: {e}", "ERROR")
                
                if self.var_quadratic.get() and len(x) >= 3:
                    try:
                        X_quad = sm.add_constant(np.column_stack((x, x**2)))
                        mod_quad = sm.OLS(y, X_quad).fit()
                        yhat_quad = mod_quad.predict(X_quad)
                        p_quad, _ = safe_pearson(y, yhat_quad)
                        all_results.append({
                            "Variable Dependiente": dep_var,
                            "Variable Independiente": indep_var,
                            "Modelo": "Cuadrático",
                            "r": p_quad,
                            "R²": mod_quad.rsquared,
                            "P-valor": mod_quad.f_pvalue
                        })
                    except Exception as e: self.log_message(f"Error en modelo Cuadrático: {e}", "ERROR")

                if self.var_cubic.get() and len(x) >= 4:
                    try:
                        X_cubic = sm.add_constant(np.column_stack((x, x**2, x**3)))
                        mod_cubic = sm.OLS(y, X_cubic).fit()
                        yhat_cubic = mod_cubic.predict(X_cubic)
                        p_cubic, _ = safe_pearson(y, yhat_cubic)
                        all_results.append({
                            "Variable Dependiente": dep_var,
                            "Variable Independiente": indep_var,
                            "Modelo": "Cúbico",
                            "r": p_cubic,
                            "R²": mod_cubic.rsquared,
                            "P-valor": mod_cubic.f_pvalue
                        })
                    except Exception as e: self.log_message(f"Error en modelo Cúbico: {e}", "ERROR")

                if self.var_power.get():
                    mask_p = (x > 0) & (y > 0)
                    if mask_p.sum() > 2:
                        xp, yp = x[mask_p], y[mask_p]
                        try:
                            sl,it,_,p_val_b,_ = stats.linregress(np.log(xp),np.log(yp)); a,b=np.exp(it),sl; yhat=a*(xp**b)
                            p,pp=safe_pearson(yp,yhat); s,ps=safe_spearman(yp,yhat); r2=p**2 if not np.isnan(p) else np.nan
                            all_results.append({
                                "Variable Dependiente": dep_var,
                                "Variable Independiente": indep_var,
                                "Modelo": "Potencia",
                                "r": p,
                                "R²": r2,
                                "P-valor": pp
                            })
                        except Exception as e: self.log_message(f"Error en modelo Potencia: {e}", "ERROR")

                if self.var_log.get():
                    mask_l = x > 0
                    if mask_l.sum() > 2:
                        xp, yp = x[mask_l], y[mask_l]
                        try:
                            popt, pcov = curve_fit(lambda z,a,b:a+b*np.log(z),xp,yp,maxfev=10000); yhat=popt[0]+popt[1]*np.log(xp)
                            p,pp=safe_pearson(yp,yhat); s,ps=safe_spearman(yp,yhat); r2=p**2 if not np.isnan(p) else np.nan
                            all_results.append({
                                "Variable Dependiente": dep_var,
                                "Variable Independiente": indep_var,
                                "Modelo": "Logarítmico",
                                "r": p,
                                "R²": r2,
                                "P-valor": pp
                            })
                        except Exception as e: self.log_message(f"Error en modelo Logarítmico: {e}", "ERROR")

                if self.var_inverse.get():
                    mask_i = x != 0
                    if mask_i.sum() > 2:
                        xi, yi = x[mask_i], y[mask_i]
                        try:
                            X_inv = sm.add_constant(1 / xi)
                            mod_inv = sm.OLS(yi, X_inv).fit()
                            yhat_inv = mod_inv.predict(X_inv)
                            p_inv, pp_inv = safe_pearson(yi, yhat_inv)
                            all_results.append({
                                "Variable Dependiente": dep_var,
                                "Variable Independiente": indep_var,
                                "Modelo": "Inverso",
                                "r": p_inv,
                                "R²": mod_inv.rsquared,
                                "P-valor": mod_inv.f_pvalue
                            })
                        except Exception as e: self.log_message(f"Error en modelo Inverso: {e}", "ERROR")

                if self.var_rcs.get():
                    try:
                        X_rcs = dmatrix(f"cr(x, df=4)", {"x": x}, return_type='dataframe')
                        mod_rcs = sm.OLS(y, X_rcs).fit()
                        yhat_rcs = mod_rcs.predict(X_rcs)
                        p_rcs, _ = safe_pearson(y, yhat_rcs)
                        all_results.append({
                            "Variable Dependiente": dep_var,
                            "Variable Independiente": indep_var,
                            "Modelo": "Spline (RCS, df=4)",
                            "r": p_rcs,
                            "R²": mod_rcs.rsquared_adj,
                            "P-valor": mod_rcs.f_pvalue
                        })
                    except Exception as e: self.log_message(f"Error en modelo Spline (RCS): {e}", "ERROR")

                other_models_to_fit = []
                if self.var_exp1.get(): other_models_to_fit.append(("Exp (a+b^x)", exp_model1))
                if self.var_exp2.get(): other_models_to_fit.append(("Exp (a+x^b)", exp_model2))
                if self.var_exp3.get(): other_models_to_fit.append(("Exp (A*B^x)", exp_model3))
                if self.var_sigmoid.get(): other_models_to_fit.append(("Sigmoide", sigmoid))
                if self.var_exp_decay.get(): other_models_to_fit.append(("Exp Decreciente", exp_decay))

                for model_name, model_func in other_models_to_fit:
                    try:
                        p0_other = None
                        if model_name == "Sigmoide" and len(y)>1 and len(x)>0 : p0_other = [max(y)-min(y), 1.0, np.median(x), min(y)]
                        elif model_name == "Sigmoide": p0_other = [1,1,0,0]
                        
                        popt_other, pcov_other = curve_fit(model_func, x, y, p0=p0_other, maxfev=10000)
                        yhat_other = model_func(x, *popt_other)
                        pear_other, p_pear_other = safe_pearson(y, yhat_other)
                        r2_other = pear_other**2 if not np.isnan(pear_other) else np.nan
                        all_results.append({
                            "Variable Dependiente": dep_var,
                            "Variable Independiente": indep_var,
                            "Modelo": model_name,
                            "r": pear_other,
                            "R²": r2_other,
                            "P-valor": p_pear_other
                        })
                    except RuntimeError: self.log_message(f"No se pudo ajustar {model_name} para {indep_var} (RuntimeError).", "WARN")
                    except Exception as e_other_model: self.log_message(f"Error en {model_name} para {indep_var}: {e_other_model}", "ERROR")

                if self.var_cubic.get() and len(x) >= 4:
                    try:
                        X_cubic = sm.add_constant(np.column_stack((x, x**2, x**3)))
                        mod_cubic = sm.OLS(y, X_cubic).fit()
                        yhat_cubic = mod_cubic.predict(X_cubic)
                        p_cubic, _ = safe_pearson(y, yhat_cubic)
                        all_results.append({
                            "Variable Dependiente": dep_var,
                            "Variable Independiente": indep_var,
                            "Modelo": "Cúbico",
                            "r": p_cubic,
                            "R²": mod_cubic.rsquared,
                            "P-valor": mod_cubic.f_pvalue
                        })
                    except Exception as e: self.log_message(f"Error en modelo Cúbico: {e}", "ERROR")

        if not all_results:
            self.results_text_content = "No se pudieron calcular correlaciones para las variables seleccionadas."
            self.show_results_tab()
            return

        df_results = pd.DataFrame(all_results)
        df_results['is_significant'] = df_results['P-valor'] <= 0.05
        df_results['abs_r'] = df_results['r'].abs()
        df_results.sort_values(by=['is_significant', 'abs_r'], ascending=[False, False], inplace=True)
        df_results.drop(columns=['is_significant', 'abs_r'], inplace=True)
        
        pd.options.display.float_format = '{:,.3f}'.format

        self.results_text_content = df_results.to_string(index=False)
        self.show_results_tab()
        self.log_message("calculate_and_show_correlations: Tabla de correlaciones generada.", "INFO")

    def plot_regression(self):
        if self.var_table_only.get():
            self.calculate_and_show_correlations()
            return
        self.log_message("Iniciando plot_regression...", "DEBUG")
        if self.data is None:
            self.log_message("plot_regression: No hay datos cargados. Retornando.", "WARN")
            messagebox.showwarning("Sin datos", "No hay datos cargados en la pestaña Regresiones.", parent=self)
            return

        df_f = self._get_filtered_data_for_regression()
        if df_f is None:
            self.log_message("plot_regression: _get_filtered_data_for_regression devolvió None. Retornando.", "ERROR")
            messagebox.showerror("Error de filtros", "No se pudieron obtener datos filtrados para generar la regresión.", parent=self)
            return
        if df_f.empty:
            self.log_message("plot_regression: No hay datos después de aplicar filtros. Retornando.", "WARN")
            messagebox.showwarning("Sin datos", "No hay datos disponibles después de aplicar filtros en Regresiones.", parent=self)
            return

        self.log_message(f"plot_regression: Datos filtrados obtenidos con {df_f.shape[0]} filas.", "DEBUG")

        selected_dep_indices = self.listbox_dep_vars_spec.curselection()
        selected_dep_vars_names = [self.listbox_dep_vars_spec.get(i) for i in selected_dep_indices]
        if not selected_dep_vars_names:
            messagebox.showwarning("Advertencia", "Por favor, seleccione al menos una variable dependiente.", parent=self)
            self.log_message("plot_regression: No hay variables dependientes seleccionadas. Retornando.", "WARN")
            return

        selected_indep_indices = self.listbox_indep_vars_spec.curselection()
        selected_indep_vars_names = [self.listbox_indep_vars_spec.get(i) for i in selected_indep_indices]
        if not selected_indep_vars_names:
            messagebox.showwarning("Advertencia", "Por favor, seleccione al menos una variable independiente.", parent=self)
            self.log_message("plot_regression: No hay variables independientes seleccionadas. Retornando.", "WARN")
            return

        selected_model_flags = [
            self.var_linear.get(),
            self.var_quadratic.get(),
            self.var_cubic.get(),
            self.var_power.get(),
            self.var_log.get(),
            self.var_loess.get(),
            self.var_inverse.get(),
            self.var_rcs.get(),
            self.var_exp1.get(),
            self.var_exp2.get(),
            self.var_exp3.get(),
            self.var_sigmoid.get(),
            self.var_exp_decay.get(),
        ]
        if not any(selected_model_flags):
            self.var_linear.set(True)
            self.log_message("No había modelos seleccionados; se activó 'Lineal' automáticamente.", "WARN")
            messagebox.showinfo(
                "Modelo automático",
                "No había modelos seleccionados. Se activó automáticamente 'Lineal' para ejecutar la regresión.",
                parent=self,
            )

        self.log_message(f"plot_regression: Variables dependientes seleccionadas: {selected_dep_vars_names}", "DEBUG")
        self.log_message(f"plot_regression: Variables independientes seleccionadas: {selected_indep_vars_names}", "DEBUG")

        try:
            dpi = int(self.entry_dpi.get()); w_px = int(self.entry_width.get()); h_px = int(self.entry_height.get())
            w_in, h_in = w_px/dpi, h_px/dpi; pt_size = float(self.entry_pt_size.get()); txt_size = int(self.entry_text_size.get())
            title_sz = int(self.entry_title_size.get())
            self.log_message(f"plot_regression: Parámetros gráficos leídos: DPI={dpi}, W={w_px}, H={h_px}, PtSize={pt_size}, TxtSize={txt_size}", "DEBUG")
        except ValueError as e_params:
            self.log_message(f"plot_regression: Error en parámetros DPI/tamaño: {e_params}. Retornando.", "ERROR")
            messagebox.showerror("Error de Parámetros", f"Error en los valores de DPI o tamaño de gráfico:\n{e_params}", parent=self)
            return

        font_family = self.font_family_var.get()
        font_size = int(self.entry_text_size.get())
        decimals = self.decimals_var.get()
        if decimals is None:
            decimals = 2
        decimals = max(0, int(decimals))
        use_sci_notation = self.sci_notation_var.get()
        use_sci_notation_conditional = self.sci_notation_conditional_var.get()

        def resolve_entry_text(entry_widget, default_value):
            raw_value = entry_widget.get()
            if raw_value == "":
                return default_value
            stripped = raw_value.strip()
            if stripped == "":
                return ""
            return stripped

        def format_number(num):
            try:
                value = float(num)
            except (TypeError, ValueError):
                return str(num)

            lower_threshold = 10 ** (-(decimals + 1)) if decimals is not None else 0.001
            upper_threshold = 10 ** (decimals + 1) if decimals is not None else 1000

            if use_sci_notation:
                return f"{value:.{decimals}e}"
            if use_sci_notation_conditional and value != 0:
                abs_value = abs(value)
                if abs_value >= upper_threshold or abs_value < lower_threshold:
                    return f"{value:.{decimals}e}"
            return f"{value:.{decimals}f}"

        show_formula_flag = self.var_show_formula.get()
        show_r2_flag = self.var_show_r2.get()
        show_grid = self.var_grid.get()
        show_info = self.var_show_info.get()
        hide_point_legend = self.var_hide_points_labels.get()

        def build_poly_formula(dep_name, coeffs, indep_name):
            terms = []
            for power, coef in enumerate(coeffs):
                if coef is None:
                    continue
                try:
                    if np.isnan(coef):
                        continue
                except TypeError:
                    pass
                if power == 0:
                    terms.append(format_number(coef))
                else:
                    sign = "-" if coef < 0 else "+"
                    coef_abs = format_number(abs(coef))
                    var_part = indep_name if power == 1 else f"{indep_name}^{power}"
                    terms.append(f"{sign} {coef_abs}·{var_part}")
            if not terms:
                return f"{dep_name} = 0"
            first = terms[0]
            rest = " ".join(terms[1:]) if len(terms) > 1 else ""
            body = f"{first} {rest}".strip()
            return f"{dep_name} = {body}"

        def build_label(base, formula=None, r2_value=None):
            detail_parts = []
            if show_formula_flag and formula:
                detail_parts.append(formula)
            if show_r2_flag and r2_value is not None:
                try:
                    if not np.isnan(r2_value):
                        detail_parts.append(f"R²={format_number(r2_value)}")
                except TypeError:
                    detail_parts.append(f"R²={format_number(r2_value)}")
            if detail_parts:
                return f"{base} ({'; '.join(detail_parts)})"
            return base

        def normalize_p_values(p_values):
            if p_values is None:
                return []
            if isinstance(p_values, pd.Series):
                return list(p_values.items())
            if isinstance(p_values, dict):
                return list(p_values.items())
            try:
                return [(f"Coef {idx}", val) for idx, val in enumerate(p_values)]
            except TypeError:
                return [("Coef", p_values)]

        def parse_range_string(text_value):
            raw = (text_value or "").strip()
            if not raw:
                return None
            try:
                parts = [float(part.strip()) for part in raw.split(',') if part.strip()]
                if len(parts) == 2:
                    lo, hi = parts
                    if lo == hi:
                        return (lo, hi)
                    return (min(lo, hi), max(lo, hi))
            except ValueError:
                self.log_message(f"Rango inválido ingresado: '{raw}'", "WARN")
            return None

        def parse_ticks_string(text_value):
            raw = (text_value or "").strip()
            if not raw:
                return []
            ticks = []
            for piece in raw.split(','):
                piece = piece.strip()
                if not piece:
                    continue
                try:
                    ticks.append(float(piece))
                except ValueError:
                    self.log_message(f"Tick inválido ignorado: '{piece}'", "WARN")
            return ticks

        def configure_axis_format(axis):
            formatter = ScalarFormatter(useMathText=True)
            formatter.set_useOffset(False)
            if use_sci_notation:
                formatter.set_scientific(True)
                formatter.set_powerlimits((0, 0))
            elif use_sci_notation_conditional:
                formatter.set_scientific(True)
                power_limit = decimals + 1 if decimals is not None else 3
                formatter.set_powerlimits((-(power_limit), power_limit))
            else:
                formatter.set_scientific(False)
            axis.set_major_formatter(formatter)

        x_limits = parse_range_string(self.entry_xlim.get()) if hasattr(self, 'entry_xlim') else None
        y_limits = parse_range_string(self.entry_ylim.get()) if hasattr(self, 'entry_ylim') else None
        x_ticks_manual = parse_ticks_string(self.entry_xticks.get()) if hasattr(self, 'entry_xticks') else []
        y_ticks_manual = parse_ticks_string(self.entry_yticks.get()) if hasattr(self, 'entry_yticks') else []

        plt.rcParams.update({
            'font.family': font_family,
            'font.size': font_size,
            'axes.titlesize': title_sz,
            'axes.labelsize': font_size,
            'xtick.labelsize': font_size,
            'ytick.labelsize': font_size,
            'legend.fontsize': font_size
        })

        all_results_summary = []

        self.fig.clear()

        total_filtered_rows = len(df_f)
        filter_summary = ""
        if show_info and hasattr(self, 'filter_component') and self.filter_component:
            try:
                filter_summary = self.filter_component.get_active_filters_description()
            except Exception as exc:
                self.log_message(f"No se pudo obtener el resumen de filtros: {exc}", "WARN")
                filter_summary = ""

        # --- Manejo de comparación por grupos ---
        compare_groups = self.var_compare_groups.get()
        group_var = self.cmb_group_var.get().strip() if compare_groups else ""
        groups_to_plot = []
        group_colors = {}
        
        if compare_groups and group_var and group_var in df_f.columns:
            # Obtener grupos a incluir
            group_filter_str = self.entry_group_filter.get().strip()
            if group_filter_str:
                groups_to_plot = [g.strip() for g in group_filter_str.split(',') if g.strip()]
            else:
                groups_to_plot = df_f[group_var].dropna().unique().tolist()
            
            # Asignar colores a cada grupo
            for idx, grp in enumerate(groups_to_plot):
                group_colors[grp] = self.default_colors[idx % len(self.default_colors)]
            
            self.log_message(f"Comparación por grupos activada. Variable: {group_var}, Grupos: {groups_to_plot}", "DEBUG")
        else:
            groups_to_plot = [None]  # Sin agrupación
            group_colors[None] = self.cmb_pt_color.get()

        for dep_original in selected_dep_vars_names:
            dep_display = dep_original
            self.log_message(f"plot_regression: Procesando VD: '{dep_original}'.", "DEBUG")

            if dep_original not in df_f.columns or not pd.api.types.is_numeric_dtype(df_f[dep_original]):
                self.log_message(f"plot_regression: VD '{dep_original}' no es válida. Saltando.", "ERROR")
                continue

            num_indep = len(selected_indep_vars_names)
            if num_indep == 0:
                continue

            self.fig.set_size_inches(w_in, h_in * num_indep)
            axes = self.fig.subplots(nrows=num_indep, ncols=1, squeeze=False)
            
            summary_for_dep = [f"Resumen de Modelos para VD: {dep_display}\n" + ("-" * 70) + "\n"]

            for i, indep_original in enumerate(selected_indep_vars_names):
                ax = axes[i, 0]
                indep_display = indep_original
                self.log_message(f"plot_regression: Procesando VI #{i+1}: '{indep_original}' para VD '{dep_original}'.", "DEBUG")

                if indep_original not in df_f.columns or not pd.api.types.is_numeric_dtype(df_f[indep_original]):
                    self.log_message(f"plot_regression: VI '{indep_original}' no es válida. Saltando.", "WARN")
                    ax.text(0.5, 0.5, f"Variable '{indep_original}' no válida.", ha='center', va='center')
                    continue

                # --- Iterar por grupos si está activada la comparación ---
                try:
                    line_width = float(self.entry_line_width.get())
                except (TypeError, ValueError):
                    line_width = 2.0
                    self.log_message("Grosor de línea inválido; se usará 2.0.", "WARN")
                results_list = []
                general_results_list = []  # Para el modelo GENERAL (todos los datos)
                
                # Verificar transformaciones ln
                apply_ln_x = self.var_ln_x.get()
                apply_ln_y = self.var_ln_y.get()
                
                # Sufijos para etiquetas si hay transformación
                x_label_suffix = " [ln]" if apply_ln_x else ""
                y_label_suffix = " [ln]" if apply_ln_y else ""
                indep_display_transformed = indep_display + x_label_suffix
                dep_display_transformed = dep_display + y_label_suffix
                
                # --- MODELO GENERAL (todos los datos combinados) ---
                if compare_groups and group_var and len(groups_to_plot) >= 2:
                    # Calcular regresión con TODOS los datos primero
                    temp_df_all = df_f[[dep_original, indep_original]].dropna()
                    if temp_df_all.shape[0] >= 2:
                        x_all = temp_df_all[indep_original].values.astype(float)
                        y_all = temp_df_all[dep_original].values.astype(float)
                        
                        # Aplicar transformaciones ln si están habilitadas
                        if apply_ln_x:
                            mask_x = x_all > 0
                            x_all = x_all[mask_x]
                            y_all = y_all[mask_x]
                            x_all = np.log(x_all)
                        if apply_ln_y:
                            mask_y = y_all > 0
                            x_all = x_all[mask_y]
                            y_all = y_all[mask_y]
                            y_all = np.log(y_all)
                        
                        n_all = len(x_all)
                        if n_all >= 2:
                            x_sorted_all = np.sort(x_all)
                            
                            # Modelo Lineal GENERAL
                            if self.var_linear.get():
                                try:
                                    X_lin_all = sm.add_constant(x_all)
                                    mod_all = sm.OLS(y_all, X_lin_all).fit()
                                    const_all, slope_all = mod_all.params
                                    formula_all = build_poly_formula(dep_display_transformed, [const_all, slope_all], indep_display_transformed)
                                    r2_all = mod_all.rsquared
                                    general_results_list.append({
                                        "model": "Lineal",
                                        "group": "GENERAL",
                                        "var": indep_display_transformed,
                                        "dep_var": dep_display_transformed,
                                        "r2": r2_all,
                                        "formula": formula_all,
                                        "p_general": mod_all.f_pvalue,
                                        "p_values": normalize_p_values(mod_all.pvalues),
                                        "n": n_all
                                    })
                                except Exception as e: 
                                    self.log_message(f"Error en modelo Lineal GENERAL: {e}", "ERROR")
                            
                            # Modelo Potencia GENERAL
                            if self.var_power.get():
                                mask_p = (x_all > 0) & (y_all > 0) if not apply_ln_x and not apply_ln_y else np.ones(len(x_all), dtype=bool)
                                if mask_p.sum() > 2:
                                    xp, yp = x_all[mask_p], y_all[mask_p]
                                    try:
                                        sl, it, _, p_val_b, _ = stats.linregress(np.log(xp) if not apply_ln_x else xp, 
                                                                                   np.log(yp) if not apply_ln_y else yp)
                                        a, b = np.exp(it), sl
                                        formula_pow = f"{dep_display_transformed} = {format_number(a)}·{indep_display_transformed}^{format_number(b)}"
                                        yhat_pow = a * (xp ** b) if not apply_ln_x else a * np.exp(xp * b)
                                        r2_pow = stats.pearsonr(yp, yhat_pow)[0] ** 2
                                        general_results_list.append({
                                            "model": "Potencia",
                                            "group": "GENERAL",
                                            "var": indep_display_transformed,
                                            "dep_var": dep_display_transformed,
                                            "r2": r2_pow,
                                            "formula": formula_pow,
                                            "p_values": [("Intercepto", 0), ("Exponente", p_val_b)],
                                            "n": n_all
                                        })
                                    except Exception as e:
                                        self.log_message(f"Error en modelo Potencia GENERAL: {e}", "ERROR")
                            
                            # Modelo Cuadrático GENERAL
                            if self.var_quadratic.get() and n_all >= 3:
                                try:
                                    X_quad_all = sm.add_constant(np.column_stack((x_all, x_all**2)))
                                    mod_quad_all = sm.OLS(y_all, X_quad_all).fit()
                                    params_quad = mod_quad_all.params
                                    formula_quad = build_poly_formula(dep_display_transformed, params_quad, indep_display_transformed)
                                    r2_quad = mod_quad_all.rsquared
                                    general_results_list.append({
                                        "model": "Cuadrático",
                                        "group": "GENERAL",
                                        "var": indep_display_transformed,
                                        "dep_var": dep_display_transformed,
                                        "r2": r2_quad,
                                        "formula": formula_quad,
                                        "p_general": mod_quad_all.f_pvalue,
                                        "p_values": normalize_p_values(mod_quad_all.pvalues),
                                        "n": n_all
                                    })
                                except Exception as e:
                                    self.log_message(f"Error en modelo Cuadrático GENERAL: {e}", "ERROR")
                            
                            # Modelo Logarítmico GENERAL
                            if self.var_log.get():
                                mask_log = x_all > 0 if not apply_ln_x else np.ones(len(x_all), dtype=bool)
                                if mask_log.sum() > 2:
                                    xl, yl = x_all[mask_log], y_all[mask_log]
                                    try:
                                        log_x = np.log(xl) if not apply_ln_x else xl
                                        X_log_all = sm.add_constant(log_x)
                                        mod_log_all = sm.OLS(yl, X_log_all).fit()
                                        const_log, slope_log = mod_log_all.params
                                        formula_log = f"{dep_display_transformed} = {format_number(const_log)} + {format_number(slope_log)}·ln({indep_display})"
                                        r2_log = mod_log_all.rsquared
                                        general_results_list.append({
                                            "model": "Logarítmico",
                                            "group": "GENERAL",
                                            "var": indep_display_transformed,
                                            "dep_var": dep_display_transformed,
                                            "r2": r2_log,
                                            "formula": formula_log,
                                            "p_general": mod_log_all.f_pvalue,
                                            "p_values": normalize_p_values(mod_log_all.pvalues),
                                            "n": n_all
                                        })
                                    except Exception as e:
                                        self.log_message(f"Error en modelo Logarítmico GENERAL: {e}", "ERROR")
                
                for group_val in groups_to_plot:
                    # Filtrar por grupo si aplica
                    if group_val is not None and group_var:
                        group_df = df_f[df_f[group_var].astype(str) == str(group_val)]
                        group_label_prefix = f"[{group_val}] "
                        point_color = group_colors.get(group_val, self.cmb_pt_color.get())
                        line_color = point_color  # Mismo color para puntos y línea del grupo
                    else:
                        group_df = df_f
                        group_label_prefix = ""
                        point_color = self.cmb_pt_color.get()
                        line_color = self.cmb_line_color.get()
                    
                    temp_df = group_df[[dep_original, indep_original]].dropna()
                    if temp_df.shape[0] < 2:
                        if group_val is not None:
                            self.log_message(f"plot_regression: No hay suficientes datos para grupo '{group_val}'. Saltando.", "WARN")
                        continue

                    x = temp_df[indep_original].values.astype(float)
                    y = temp_df[dep_original].values.astype(float)
                    
                    # Guardar originales para la tabla de frecuencia
                    x_original = x.copy()
                    y_original = y.copy()
                    
                    # Aplicar transformaciones ln si están habilitadas
                    if apply_ln_x:
                        mask_x = x > 0
                        x = x[mask_x]
                        y = y[mask_x]
                        x_original = x_original[mask_x]
                        y_original = y_original[mask_x]
                        x = np.log(x)
                    if apply_ln_y:
                        mask_y = y > 0
                        x = x[mask_y]
                        y = y[mask_y]
                        x_original = x_original[mask_y]
                        y_original = y_original[mask_y]
                        y = np.log(y)
                    
                    if len(x) < 2:
                        if group_val is not None:
                            self.log_message(f"plot_regression: No hay suficientes datos para grupo '{group_val}' después de transformación ln.", "WARN")
                        continue
                    
                    # Apply custom labels to the dependent variable for display in summary
                    custom_labels_str = self.entry_dep_var_labels.get().strip()
                    y_for_summary = pd.Series(y_original)  # Usar valores originales para frecuencia
                    if custom_labels_str:
                        y_for_summary = self._apply_custom_labels_to_series(y_for_summary, custom_labels_str)

                    # Mostrar encabezado de grupo si hay comparación de grupos
                    if group_val is not None and self.var_show_group_stats.get():
                        summary_for_dep.append(f"\n--- Grupo: {group_val} (n={len(x)}) ---\n")

                    scatter_label = f"{group_label_prefix}{indep_display}" if not hide_point_legend else "_nolegend_"
                    ax.scatter(x, y, color=point_color, s=pt_size, alpha=0.6, label=scatter_label)

                    sort_idx = np.argsort(x)
                    x_sorted = x[sort_idx]

                    if self.var_linear.get():
                        try:
                            X_lin = sm.add_constant(x)
                            mod = sm.OLS(y, X_lin).fit()
                            const_lin, slope_lin = mod.params
                            formula_lin = build_poly_formula(dep_display, [const_lin, slope_lin], indep_display)
                            yhat = mod.predict(X_lin)
                            p_lin, _ = safe_pearson(y, yhat)
                            r2_lin = mod.rsquared
                            legend_label = build_label(f"{group_label_prefix}Lineal", formula_lin, r2_lin)
                            ax.plot(x_sorted, mod.predict(sm.add_constant(x_sorted)), linestyle=self.model_styles["Lineal"]["linestyle"], color=line_color, lw=line_width, label=legend_label)
                            results_list.append({
                                "model": "Lineal",
                                "group": group_val,
                                "var": indep_display,
                                "dep_var": dep_display,
                                "r": p_lin,
                                "r2": r2_lin,
                                "formula": formula_lin,
                                "p_general": mod.f_pvalue,
                                "p_values": normalize_p_values(mod.pvalues),
                                "n": len(x)
                            })
                        except Exception as e: self.log_message(f"Error en modelo Lineal ({indep_display}, grupo={group_val}): {e}", "ERROR")

                    if self.var_quadratic.get() and len(x) >= 3:
                        try:
                            X_quad = sm.add_constant(np.column_stack((x, x**2)))
                            mod_quad = sm.OLS(y, X_quad).fit()
                            const_quad, coef1_quad, coef2_quad = mod_quad.params
                            formula_quad = build_poly_formula(dep_display, [const_quad, coef1_quad, coef2_quad], indep_display)
                            yhat_quad = mod_quad.predict(X_quad)
                            p_quad, _ = safe_pearson(y, yhat_quad)
                            r2_quad = mod_quad.rsquared
                            legend_label = build_label(f"{group_label_prefix}Cuadrático", formula_quad, r2_quad)
                            ax.plot(x_sorted, np.polyval([coef2_quad, coef1_quad, const_quad], x_sorted), linestyle=self.model_styles["Cuadrático"]["linestyle"], color=line_color, lw=line_width, label=legend_label)
                            results_list.append({
                                "model": "Cuadrático",
                                "group": group_val,
                                "var": indep_display,
                                "dep_var": dep_display,
                                "r": p_quad,
                                "r2": r2_quad,
                                "formula": formula_quad,
                                "p_general": mod_quad.f_pvalue,
                                "p_values": normalize_p_values(mod_quad.pvalues),
                                "n": len(x)
                            })
                        except Exception as e: self.log_message(f"Error Cuad ({indep_display}, grupo={group_val}): {e}", "ERROR")

                    if self.var_cubic.get() and len(x) >= 4:
                        try:
                            X_cubic = sm.add_constant(np.column_stack((x, x**2, x**3)))
                            mod_cubic = sm.OLS(y, X_cubic).fit()
                            params_cubic = mod_cubic.params
                            formula_cubic = build_poly_formula(dep_display, params_cubic, indep_display)
                            yhat_cubic = mod_cubic.predict(X_cubic)
                            p_cubic, _ = safe_pearson(y, yhat_cubic)
                            r2_cubic = mod_cubic.rsquared
                            legend_label = build_label(f"{group_label_prefix}Cúbico", formula_cubic, r2_cubic)
                            ax.plot(x_sorted, np.polyval(params_cubic[::-1], x_sorted), linestyle=self.model_styles["Cúbico"]["linestyle"], color=line_color, lw=line_width, label=legend_label)
                            results_list.append({
                                "model": "Cúbico",
                                "group": group_val,
                                "var": indep_display,
                                "dep_var": dep_display,
                                "r": p_cubic,
                                "r2": r2_cubic,
                                "formula": formula_cubic,
                                "p_general": mod_cubic.f_pvalue,
                                "p_values": normalize_p_values(mod_cubic.pvalues)
                            })
                        except Exception as e: self.log_message(f"Error Cúbico ({indep_display}): {e}", "ERROR")

                if self.var_power.get():
                    mask_p = (x > 0) & (y > 0)
                    if mask_p.sum() > 2:
                        xp, yp = x[mask_p], y[mask_p]
                        try:
                            sl,it,_,p_val_b,_ = stats.linregress(np.log(xp),np.log(yp)); a,b=np.exp(it),sl; yhat=a*(xp**b)
                            p,pp=safe_pearson(yp,yhat); s,ps=safe_spearman(yp,yhat); r2=p**2 if not np.isnan(p) else np.nan
                            formula_power = f"{dep_display} = {format_number(a)}·{indep_display}^{format_number(b)}"
                            legend_label = build_label("Potencia", formula_power, r2)
                            ax.plot(np.sort(xp), a*np.power(np.sort(xp),b), linestyle=self.model_styles["Potencia"]["linestyle"], color=line_color, label=legend_label)
                            results_list.append({
                                "model": "Potencia",
                                "var": indep_display,
                                "dep_var": dep_display,
                                "r": p,
                                "r2": r2,
                                "formula": formula_power,
                                "p_values": [("Intercepto", np.nan), ("Exponente", p_val_b)]
                            })
                        except Exception as e: self.log_message(f"Error Potencia ({indep_display}): {e}", "ERROR")

                if self.var_log.get():
                    mask_l = x > 0
                    if mask_l.sum() > 2:
                        xp, yp = x[mask_l], y[mask_l]
                        try:
                            popt, pcov = curve_fit(lambda z,a,b:a+b*np.log(z),xp,yp,maxfev=10000); yhat=popt[0]+popt[1]*np.log(xp)
                            p,pp=safe_pearson(yp,yhat); s,ps=safe_spearman(yp,yhat); r2=p**2 if not np.isnan(p) else np.nan
                            p_values = self.get_p_values_from_curve_fit(popt, pcov, len(yp))
                            formula_log = f"{dep_display} = {format_number(popt[0])} + {format_number(popt[1])}·ln({indep_display})"
                            legend_label = build_label("Logarítmico", formula_log, r2)
                            ax.plot(np.sort(xp), popt[0]+popt[1]*np.log(np.sort(xp)), linestyle=self.model_styles["Logarítmico"]["linestyle"], color=line_color, label=legend_label)
                            p_values_named = [(f"Parámetro {idx}", val) for idx, val in enumerate(p_values)] if p_values is not None else []
                            results_list.append({
                                "model": "Logarítmico",
                                "var": indep_display,
                                "dep_var": dep_display,
                                "r": p,
                                "r2": r2,
                                "formula": formula_log,
                                "p_values": p_values_named
                            })
                        except Exception as e: self.log_message(f"Error Log ({indep_display}): {e}", "ERROR")

                if self.var_loess.get() and len(x) > 5:
                    try:
                        lo=lowess(y,x,frac=0.3); xs,ys=lo[:,0],lo[:,1]; acme_x,acme_y=compute_acme(lambda z:np.interp(z,xs,ys),np.linspace(xs.min(),xs.max(),200))
                        formula_loess = f"LOESS({indep_display})"
                        legend_label = build_label("LOESS", formula_loess, None)
                        results_list.append({"model":"LOESS", "var":indep_display, "dep_var":dep_display, "r":np.nan, "r2":np.nan, "formula":f"LOESS para {indep_display}: acme en x={format_number(acme_x)}, y={format_number(acme_y)}"})
                        ax.plot(xs,ys, linestyle=self.model_styles["LOESS"]["linestyle"], color=line_color, label=legend_label)
                    except Exception as e: self.log_message(f"Error LOESS ({indep_display}): {e}", "ERROR")

                if self.var_inverse.get():
                    mask_i = x != 0
                    if mask_i.sum() > 2:
                        xi, yi = x[mask_i], y[mask_i]
                        try:
                            X_inv = sm.add_constant(1 / xi)
                            mod_inv = sm.OLS(yi, X_inv).fit()
                            a_inv, b_inv = mod_inv.params
                            yhat_inv = mod_inv.predict(X_inv)
                            p_inv, pp_inv = safe_pearson(yi, yhat_inv)
                            s_inv, ps_inv = safe_spearman(yi, yhat_inv)
                            r2_inv = mod_inv.rsquared
                            formula_inv = f"{dep_display} = {format_number(a_inv)} + {format_number(b_inv)}/{indep_display}"
                            legend_label = build_label("Inverso", formula_inv, r2_inv)
                            results_list.append({
                                "model": "Inverso",
                                "var": indep_display,
                                "dep_var": dep_display,
                                "r": p_inv,
                                "r2": r2_inv,
                                "formula": formula_inv,
                                "p_general": mod_inv.f_pvalue,
                                "p_values": normalize_p_values(mod_inv.pvalues)
                            })
                            x_sorted_inv = np.sort(xi)
                            ax.plot(x_sorted_inv, a_inv + b_inv / x_sorted_inv, linestyle="-", color=line_color, label=legend_label)
                        except Exception as e_inv: self.log_message(f"Error Inverso ({indep_display}): {e_inv}", "ERROR")

                if self.var_rcs.get():
                    try:
                        X_rcs = dmatrix(f"cr(x, df=4)", {"x": x}, return_type='dataframe')
                        mod_rcs = sm.OLS(y, X_rcs).fit()
                        yhat_rcs = mod_rcs.predict(dmatrix(f"cr(x_sorted, df=4)", {"x_sorted": x_sorted}, return_type='dataframe'))
                        p_rcs, pp_rcs = safe_pearson(y, mod_rcs.predict(X_rcs))
                        s_rcs, ps_rcs = safe_spearman(y, mod_rcs.predict(X_rcs))
                        r2_rcs = mod_rcs.rsquared_adj
                        formula_rcs = "Spline Cúbico Restringido (df=4)"
                        legend_label = build_label("Spline RCS", formula_rcs, r2_rcs)
                        results_list.append({
                            "model": "Spline (RCS, df=4)",
                            "var": indep_display,
                            "dep_var": dep_display,
                            "r": p_rcs,
                            "r2": r2_rcs,
                            "formula": formula_rcs,
                            "p_general": mod_rcs.f_pvalue,
                            "p_values": normalize_p_values(mod_rcs.pvalues)
                        })
                        ax.plot(x_sorted, yhat_rcs, linestyle="--", color=line_color, label=legend_label)
                    except Exception as e_rcs: self.log_message(f"Error en modelo Spline (RCS) ({indep_display}): {e_rcs}", "ERROR")

                other_models_to_fit = []
                if self.var_exp1.get(): other_models_to_fit.append(("Exp (a+b^x)", exp_model1))
                if self.var_exp2.get(): other_models_to_fit.append(("Exp (a+x^b)", exp_model2))
                if self.var_exp3.get(): other_models_to_fit.append(("Exp (A*B^x)", exp_model3))
                if self.var_sigmoid.get(): other_models_to_fit.append(("Sigmoide", sigmoid))
                if self.var_exp_decay.get(): other_models_to_fit.append(("Exp Decreciente", exp_decay))

                for model_name, model_func in other_models_to_fit:
                    try:
                        p0_other = None
                        if model_name == "Sigmoide" and len(y)>1 and len(x)>0 : p0_other = [max(y)-min(y), 1.0, np.median(x), min(y)]
                        elif model_name == "Sigmoide": p0_other = [1,1,0,0]
                        
                        popt_other, pcov_other = curve_fit(model_func, x, y, p0=p0_other, maxfev=10000)
                        yhat_other = model_func(x, *popt_other)
                        pear_other, p_pear_other = safe_pearson(y, yhat_other)
                        spear_other, p_spear_other = safe_spearman(y, yhat_other)
                        r2_other = pear_other**2 if not np.isnan(pear_other) else np.nan
                        p_values_other = self.get_p_values_from_curve_fit(popt_other, pcov_other, len(y))
                        if model_name == "Exp (a+b^x)":
                            formula_str_other = f"{dep_display} = {format_number(popt_other[0])} + {format_number(popt_other[1])}^{indep_display}"
                        elif model_name == "Exp (a+x^b)":
                            formula_str_other = f"{dep_display} = {format_number(popt_other[0])} + {indep_display}^{format_number(popt_other[1])}"
                        elif model_name == "Exp (A*B^x)":
                            formula_str_other = f"{dep_display} = {format_number(popt_other[0])}·{format_number(popt_other[1])}^{indep_display}"
                        elif model_name == "Sigmoide":
                            formula_str_other = (f"{dep_display} = {format_number(popt_other[0])} / (1 + exp(-{format_number(popt_other[1])}·({indep_display}-{format_number(popt_other[2])})))"
                                                f" + {format_number(popt_other[3])}")
                        elif model_name == "Exp Decreciente":
                            formula_str_other = f"{dep_display} = {format_number(popt_other[0])}·exp(-{format_number(popt_other[1])}·{indep_display})"
                        else:
                            formula_str_other = f"{model_name}"

                        legend_label = build_label(model_name, formula_str_other, r2_other)
                        p_values_named = [(f"Parámetro {idx}", val) for idx, val in enumerate(p_values_other)] if p_values_other is not None else []
                        results_list.append({
                            "model": model_name,
                            "var": indep_display,
                            "dep_var": dep_display,
                            "r": pear_other,
                            "r2": r2_other,
                            "formula": formula_str_other,
                            "p_values": p_values_named
                        })
                        ax.plot(x_sorted, model_func(x_sorted, *popt_other), linestyle=self.model_styles[model_name]["linestyle"], color=line_color, label=legend_label)
                    except RuntimeError: self.log_message(f"No se pudo ajustar {model_name} para {indep_display} (RuntimeError).", "WARN")
                    except Exception as e_other_model: self.log_message(f"Error en {model_name} para {indep_display}: {e_other_model}", "ERROR")

                resolved_xlabel = resolve_entry_text(self.entry_xlabel, indep_display)
                ax.set_xlabel(resolved_xlabel, fontsize=txt_size)
                resolved_ylabel = resolve_entry_text(self.entry_ylabel, dep_display)
                ax.set_ylabel(resolved_ylabel, fontsize=txt_size)
                resolved_title = resolve_entry_text(self.entry_title, f"Regresión de {dep_display} vs {indep_display}")
                ax.set_title(resolved_title, fontsize=title_sz)

                if self.var_plot_corr.get():
                    try:
                        r_p, _ = safe_pearson(x, y)
                        r_s, _ = safe_spearman(x, y)
                        corr_text = f"Pearson: {r_p:.3f}\nSpearman: {r_s:.3f}"
                        ax.annotate(corr_text, xy=(0.05, 0.95), xycoords='axes fraction',
                                    fontsize=max(6, txt_size-2), ha='left', va='top',
                                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.7))
                    except Exception as e: self.log_message(f"Error anotando correlaciones: {e}", "WARN")

                if x_limits:
                    ax.set_xlim(x_limits)
                if y_limits:
                    ax.set_ylim(y_limits)
                if x_ticks_manual:
                    ax.set_xticks(x_ticks_manual)
                if y_ticks_manual:
                    ax.set_yticks(y_ticks_manual)

                configure_axis_format(ax.xaxis)
                configure_axis_format(ax.yaxis)

                if show_grid:
                    ax.grid(True, linestyle='--', alpha=0.7)
                else:
                    ax.grid(False)

                if show_info:
                    info_lines = []
                    if total_filtered_rows is not None:
                        info_lines.append(f"n filtrado: {total_filtered_rows}")
                    # Mostrar n por grupo si hay comparación de grupos
                    if compare_groups and group_var and len(groups_to_plot) >= 2 and group_var in df_f.columns:
                        for grp in groups_to_plot:
                            grp_df = df_f[df_f[group_var].astype(str) == str(grp)]
                            info_lines.append(f"n {grp}: {len(grp_df)}")
                    else:
                        info_lines.append(f"n modelo: {len(temp_df)}")
                    if filter_summary:
                        wrapped_filters = textwrap.wrap(filter_summary, width=45)
                        if wrapped_filters:
                            info_lines.append("Filtros:")
                            info_lines.extend(wrapped_filters)
                    info_text = "\n".join(info_lines)
                    ax.annotate(info_text, xy=(0.95, 0.95), xycoords='axes fraction', fontsize=max(6, txt_size-2),
                                ha='right', va='top', bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#666666", alpha=0.6))

                handles, labels = ax.get_legend_handles_labels()
                legend_pairs = [(h, l) for h, l in zip(handles, labels) if l and l != '_nolegend_']
                if legend_pairs:
                    handles, labels = zip(*legend_pairs)
                    ax.legend(handles, labels, fontsize=max(6, txt_size-2), loc='best')
                else:
                    existing_legend = ax.get_legend()
                    if existing_legend is not None:
                        existing_legend.remove()

                results_list.sort(key=lambda x: x.get("r2", -1), reverse=True)
                
                # Separar resultados por grupo si aplica
                if compare_groups and group_var and len(groups_to_plot) >= 2:
                    # --- PRIMERO: Mostrar modelo GENERAL (todos los datos) ---
                    if general_results_list:
                        n_total = general_results_list[0].get('n', '?') if general_results_list else '?'
                        summary_for_dep.append(f"\n{'='*70}\n")
                        summary_for_dep.append(f"MODELO GENERAL (todos los datos combinados, n={n_total})\n")
                        summary_for_dep.append(f"{'='*70}\n")
                        
                        for r_item in general_results_list:
                            summary_for_dep.append(f"\nModelo: {r_item['model']} | VI: {r_item['var']}\n")
                            if show_formula_flag and r_item.get('formula'):
                                summary_for_dep.append(f"  Fórmula: {r_item['formula']}\n")
                            if show_r2_flag and ('r2' in r_item) and r_item['r2'] is not None:
                                try:
                                    summary_for_dep.append(f"  R² = {format_number(r_item['r2'])}\n")
                                except Exception:
                                    summary_for_dep.append(f"  R² = {r_item['r2']}\n")
                            if 'p_general' in r_item and r_item['p_general'] is not None:
                                summary_for_dep.append(f"  P-valor (F) = {fmt_p(r_item['p_general'])}\n")
                            if 'p_values' in r_item:
                                summary_for_dep.append("  P-valores de coeficientes:\n")
                                p_iterable = r_item['p_values']
                                if isinstance(p_iterable, list):
                                    for entry in p_iterable:
                                        if isinstance(entry, tuple) and len(entry) == 2:
                                            label, p_val = entry
                                            summary_for_dep.append(f"    {label}: {fmt_p(p_val)}\n")
                                        else:
                                            summary_for_dep.append(f"    Coef: {fmt_p(entry)}\n")
                                elif isinstance(p_iterable, (pd.Series, dict)):
                                    for label, p_val in (p_iterable.items() if hasattr(p_iterable, 'items') else p_iterable):
                                        summary_for_dep.append(f"    {label}: {fmt_p(p_val)}\n")
                        summary_for_dep.append(f"{'-' * 70}\n")
                    
                    # --- SEGUNDO: Agrupar y mostrar resultados por grupo ---
                    results_by_group = {}
                    for r_item in results_list:
                        grp = r_item.get('group', 'General')
                        if grp not in results_by_group:
                            results_by_group[grp] = []
                        results_by_group[grp].append(r_item)
                    
                    # Mostrar resultados por grupo
                    for grp in groups_to_plot:
                        if grp in results_by_group:
                            grp_results = results_by_group[grp]
                            # Obtener n del grupo
                            n_grp = grp_results[0].get('n', '?') if grp_results else '?'
                            summary_for_dep.append(f"\n{'='*70}\n")
                            summary_for_dep.append(f"REGRESIONES PARA GRUPO: {grp} (n={n_grp})\n")
                            summary_for_dep.append(f"{'='*70}\n")
                            
                            for r_item in grp_results:
                                summary_for_dep.append(f"\nModelo: {r_item['model']} | VI: {r_item['var']}\n")
                                if show_formula_flag and r_item.get('formula'):
                                    summary_for_dep.append(f"  Fórmula: {r_item['formula']}\n")
                                if show_r2_flag and ('r2' in r_item) and r_item['r2'] is not None:
                                    try:
                                        summary_for_dep.append(f"  R² = {format_number(r_item['r2'])}\n")
                                    except Exception:
                                        summary_for_dep.append(f"  R² = {r_item['r2']}\n")
                                if 'p_general' in r_item and r_item['p_general'] is not None:
                                    summary_for_dep.append(f"  P-valor (F) = {fmt_p(r_item['p_general'])}\n")
                                if 'p_values' in r_item:
                                    summary_for_dep.append("  P-valores de coeficientes:\n")
                                    p_iterable = r_item['p_values']
                                    if isinstance(p_iterable, list):
                                        for entry in p_iterable:
                                            if isinstance(entry, tuple) and len(entry) == 2:
                                                label, p_val = entry
                                                summary_for_dep.append(f"    {label}: {fmt_p(p_val)}\n")
                                            else:
                                                summary_for_dep.append(f"    Coef: {fmt_p(entry)}\n")
                                    elif isinstance(p_iterable, pd.Series):
                                        for label, p_val in p_iterable.items():
                                            summary_for_dep.append(f"    {label}: {fmt_p(p_val)}\n")
                                    elif isinstance(p_iterable, dict):
                                        for label, p_val in p_iterable.items():
                                            summary_for_dep.append(f"    {label}: {fmt_p(p_val)}\n")
                                    else:
                                        summary_for_dep.append(f"    Coef: {fmt_p(p_iterable)}\n")
                            summary_for_dep.append(f"{'-' * 70}\n")
                else:
                    # Sin grupos - mostrar todos los resultados
                    for r_item in results_list:
                        summary_for_dep.append(f"Modelo: {r_item['model']} | VI: {r_item['var']}\n")
                        if show_formula_flag and r_item.get('formula'):
                            summary_for_dep.append(f"  Fórmula: {r_item['formula']}\n")
                        if show_r2_flag and ('r2' in r_item) and r_item['r2'] is not None:
                            try:
                                summary_for_dep.append(f"  R² = {format_number(r_item['r2'])}\n")
                            except Exception:
                                summary_for_dep.append(f"  R² = {r_item['r2']}\n")
                        if 'p_general' in r_item and r_item['p_general'] is not None:
                            summary_for_dep.append(f"  P-valor (general) = {fmt_p(r_item['p_general'])}\n")
                        if 'p_values' in r_item:
                            summary_for_dep.append("  P-valores de coeficientes:\n")
                            p_iterable = r_item['p_values']
                            if isinstance(p_iterable, list):
                                for entry in p_iterable:
                                    if isinstance(entry, tuple) and len(entry) == 2:
                                        label, p_val = entry
                                        summary_for_dep.append(f"    {label}: {fmt_p(p_val)}\n")
                                    else:
                                        summary_for_dep.append(f"    Coef: {fmt_p(entry)}\n")
                            elif isinstance(p_iterable, pd.Series):
                                for label, p_val in p_iterable.items():
                                    summary_for_dep.append(f"    {label}: {fmt_p(p_val)}\n")
                            elif isinstance(p_iterable, dict):
                                for label, p_val in p_iterable.items():
                                    summary_for_dep.append(f"    {label}: {fmt_p(p_val)}\n")
                            else:
                                summary_for_dep.append(f"    Coef: {fmt_p(p_iterable)}\n")
                        summary_for_dep.append(f"{'-' * 70}\n")

                # --- Análisis de Interacción (comparación formal de pendientes) ---
                if compare_groups and group_var and len(groups_to_plot) >= 2 and self.var_interaction_analysis.get():
                    try:
                        apply_ln_x = self.var_ln_x.get()
                        apply_ln_y = self.var_ln_y.get()
                        # Comparaciones pairwise de todos los grupos
                        interaction_text = self._run_pairwise_interaction(
                            df_f, dep_original, indep_original, group_var, groups_to_plot, format_number,
                            apply_ln_x=apply_ln_x, apply_ln_y=apply_ln_y)
                        if interaction_text:
                            summary_for_dep.append("\n" + "=" * 70 + "\n")
                            summary_for_dep.append(interaction_text)
                            summary_for_dep.append("=" * 70 + "\n")
                    except Exception as e_int:
                        self.log_message(f"Error en análisis de interacción: {e_int}", "ERROR")
                        summary_for_dep.append(f"\n[Error en análisis de interacción: {e_int}]\n")

                # --- ANCOVA (Análisis de Covarianza) ---
                if compare_groups and group_var and len(groups_to_plot) >= 2 and self.var_ancova.get():
                    try:
                        apply_ln_x = self.var_ln_x.get()
                        apply_ln_y = self.var_ln_y.get()
                        # Comparaciones pairwise de todos los grupos
                        ancova_text = self._run_pairwise_ancova(
                            df_f, dep_original, indep_original, group_var, groups_to_plot, format_number,
                            apply_ln_x=apply_ln_x, apply_ln_y=apply_ln_y)
                        if ancova_text:
                            summary_for_dep.append("\n")
                            summary_for_dep.append(ancova_text)
                            summary_for_dep.append("\n")
                    except Exception as e_ancova:
                        self.log_message(f"Error en ANCOVA: {e_ancova}", "ERROR")
                        summary_for_dep.append(f"\n[Error en ANCOVA: {e_ancova}]\n")

                # --- Prueba de Chow (Ruptura Estructural) ---
                if compare_groups and group_var and len(groups_to_plot) >= 2 and self.var_chow_test.get():
                    try:
                        apply_ln_x = self.var_ln_x.get()
                        apply_ln_y = self.var_ln_y.get()
                        # Comparaciones pairwise de todos los grupos
                        chow_text = self._run_pairwise_chow(
                            df_f, dep_original, indep_original, group_var, groups_to_plot, format_number,
                            apply_ln_x=apply_ln_x, apply_ln_y=apply_ln_y)
                        if chow_text:
                            summary_for_dep.append("\n")
                            summary_for_dep.append(chow_text)
                            summary_for_dep.append("\n")
                    except Exception as e_chow:
                        self.log_message(f"Error en Prueba de Chow: {e_chow}", "ERROR")
                        summary_for_dep.append(f"\n[Error en Prueba de Chow: {e_chow}]\n")

                # --- Comparación NLS (Modelos No Lineales: AIC, BIC, LRT) ---
                if compare_groups and group_var and len(groups_to_plot) >= 2 and self.var_nls_comparison.get():
                    try:
                        apply_ln_x = self.var_ln_x.get()
                        apply_ln_y = self.var_ln_y.get()
                        # Comparaciones pairwise de todos los grupos
                        nls_text = self._run_pairwise_nls(
                            df_f, dep_original, indep_original, group_var, groups_to_plot, format_number,
                            apply_ln_x=apply_ln_x, apply_ln_y=apply_ln_y)
                        if nls_text:
                            summary_for_dep.append("\n")
                            summary_for_dep.append(nls_text)
                            summary_for_dep.append("\n")
                    except Exception as e_nls:
                        self.log_message(f"Error en comparación NLS: {e_nls}", "ERROR")
                        summary_for_dep.append(f"\n[Error en comparación NLS: {e_nls}]\n")

            all_results_summary.extend(summary_for_dep)
            self.fig.tight_layout()
            self.canvas.draw()
            self.log_message(f"plot_regression: Gráfico para '{dep_display}' dibujado en el canvas.", "INFO")

            # Switch to the graph tab
            self.results_notebook.select(self.graph_frame)

        self.results_text_content = "".join(all_results_summary)
        self.show_results_tab()
        self.log_message("plot_regression: Todas las gráficas y resúmenes generados.", "INFO")
    def show_results_tab(self):
        self.log_message("show_results_tab: Actualizando widget de texto con resultados.", "DEBUG")
        self.txt_results.config(state="normal")
        self.txt_results.delete("1.0", tk.END)
        self.txt_results.insert("1.0", self.results_text_content)
        self.txt_results.config(state="disabled")




    def save_graph_directly(self):
        self.log_message("Iniciando save_graph_directly...", "DEBUG")
        
        if not self.fig.axes:
            self.log_message("save_graph_directly: No hay gráfica para guardar (figura vacía). Retornando.", "WARN")
            messagebox.showwarning("Sin Gráfica", "No hay gráfica generada para guardar. Genere una primero.", parent=self)
            return

        dest = filedialog.asksaveasfilename(initialfile="regresion_plot.png",
                                            defaultextension=".png",
                                            filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg;*.jpeg"), ("All files", "*.*")])
        if dest:
            try:
                self.fig.savefig(dest)
                self.log_message(f"Gráfica guardada en {dest}")
            except Exception as e:
                self.log_message(f"Error al guardar gráfica: {e}")

def run_tkinter_app():
    root = tk.Tk()
    root.title("Regresiones y Dispersión")
    root.geometry("1200x800") # Set initial size, but the window will be resizable
    root.rowconfigure(0, weight=1)
    root.columnconfigure(0, weight=1)
    nb = ttk.Notebook(root)
    nb.grid(row=0, column=0, sticky="nsew")
    tab = RegresionesTab(nb)
    nb.add(tab, text="Regresiones")
    root.mainloop()

# ==============================
# Modo de ejecución: Elegir entre WEB o DESKTOP
# ==============================
if __name__ == "__main__":
    # Para simplificar, ejecutar directamente la versión Tkinter
    run_tkinter_app()
    # mode = input("Seleccione el modo de ejecución (web/desktop): ").strip().lower()
    # if mode in ["web", "w"]:
    #     run_flask_app()
    # elif mode in ["desktop", "tk", "t"]:
    #     run_tkinter_app()
    # else:
    #     print("Modo no reconocido. Escriba 'web' o 'desktop'.")
