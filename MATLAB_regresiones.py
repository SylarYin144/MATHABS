#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import io, base64
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg # Importar FigureCanvasTkAgg
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

        # Nuevos controles de estilo
        title = request.form.get("title", "").strip()
        title_size = int(request.form.get("title_size", "14"))
        title_color = request.form.get("title_color", "black")

        xlabel = request.form.get("xlabel", "").strip()
        xlabel_size = int(request.form.get("xlabel_size", "10"))
        xlabel_color = request.form.get("xlabel_color", "black")

        ylabel = request.form.get("ylabel", "").strip()
        ylabel_size = int(request.form.get("ylabel_size", "10"))
        ylabel_color = request.form.get("ylabel_color", "black")

        xlim_raw = request.form.get("xlim", "").strip()
        ylim_raw = request.form.get("ylim", "").strip()
        xticks_raw = request.form.get("xticks", "").strip()
        yticks_raw = request.form.get("yticks", "").strip()

        grid_on = True if request.form.get("grid") == "on" else False
        show_info = True if request.form.get("show_info") == "on" else False
        plot_corr = True if request.form.get("plot_corr") == "on" else False

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
            plt.hist(df_filtered[var].dropna(), bins=20, color=color, edgecolor=edge_color)
            plt.title(title or f"Histograma de {var}", fontsize=title_size, color=title_color)
            plt.xlabel(xlabel or var, fontsize=xlabel_size, color=xlabel_color)
            plt.ylabel(ylabel or "Frecuencia", fontsize=ylabel_size, color=ylabel_color)
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
            plt.title(title or f"Gráfico de barras de frecuencias para {var}", fontsize=title_size, color=title_color)
            plt.xlabel(xlabel or var, fontsize=xlabel_size, color=xlabel_color)
            plt.ylabel(ylabel or "Frecuencia", fontsize=ylabel_size, color=ylabel_color)
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
    def __init__(self, master):
        super().__init__(master)
        self.data = None # DataFrame original cargado
        self.filtered_data = None # DataFrame después de aplicar filtros
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
        frm_results = ttk.LabelFrame(right_frame, text="Resumen de Resultados")
        frm_results.pack(fill="both", expand=True, padx=10, pady=10)
        self.txt_results = scrolledtext.ScrolledText(frm_results, wrap="none", height=15) # wrap="none" para scroll H
        self.txt_results.pack(fill="both", expand=True)
        self.txt_results.config(state="disabled") # Iniciar deshabilitado

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
        frm_vars = ttk.LabelFrame(container, text="Selección de Variables (Formato: nombre_original o nombre_original:NuevoNombre)")
        frm_vars.pack(fill="x", padx=10, pady=5)
        
        dep_var_frame = ttk.Frame(frm_vars)
        dep_var_frame.pack(fill="x", pady=2)
        ttk.Label(dep_var_frame, text="Variable Dependiente:").pack(side="left", padx=5)
        self.combo_dep_var_spec = ttk.Combobox(dep_var_frame, width=38, state="readonly")
        self.combo_dep_var_spec.pack(side="left", padx=5, expand=True, fill="x")
        self.combo_dep_var_spec.bind("<<ComboboxSelected>>", self._update_indep_vars_listbox)

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
        self.entry_pt_size.insert(0, "50")
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


        # --- Modelos de Regresión ---
        frm_models = ttk.LabelFrame(container, text="Modelos de Regresión a Aplicar")
        frm_models.pack(fill="x", padx=10, pady=5)
        self.var_linear = tk.BooleanVar(value=False); ttk.Checkbutton(frm_models, text="Lineal", variable=self.var_linear).grid(row=0, column=0, padx=2, pady=2, sticky="w")
        self.var_quadratic = tk.BooleanVar(value=False); ttk.Checkbutton(frm_models, text="Cuadrática", variable=self.var_quadratic).grid(row=0, column=1, padx=2, pady=2, sticky="w")
        self.var_cubic = tk.BooleanVar(value=False); ttk.Checkbutton(frm_models, text="Cúbica", variable=self.var_cubic).grid(row=0, column=2, padx=2, pady=2, sticky="w")
        self.var_power = tk.BooleanVar(value=False); ttk.Checkbutton(frm_models, text="Potencia", variable=self.var_power).grid(row=0, column=3, padx=2, pady=2, sticky="w")
        self.var_log = tk.BooleanVar(value=False); ttk.Checkbutton(frm_models, text="Logarítmica", variable=self.var_log).grid(row=0, column=4, padx=2, pady=2, sticky="w")
        self.var_loess = tk.BooleanVar(value=False); ttk.Checkbutton(frm_models, text="LOESS", variable=self.var_loess).grid(row=0, column=5, padx=2, pady=2, sticky="w")
        
        frm_otros = ttk.LabelFrame(container, text="Otros Modelos (Solo Resumen)")
        frm_otros.pack(fill="x", padx=10, pady=5)
        self.var_exp1 = tk.BooleanVar(value=False); ttk.Checkbutton(frm_otros, text="Exp (a+b^x)", variable=self.var_exp1).grid(row=0, column=0, padx=2, pady=2, sticky="w")
        self.var_exp2 = tk.BooleanVar(value=False); ttk.Checkbutton(frm_otros, text="Exp (a+x^b)", variable=self.var_exp2).grid(row=0, column=1, padx=2, pady=2, sticky="w")
        self.var_exp3 = tk.BooleanVar(value=False); ttk.Checkbutton(frm_otros, text="Exp (A*B^x)", variable=self.var_exp3).grid(row=0, column=2, padx=2, pady=2, sticky="w")
        self.var_sigmoid = tk.BooleanVar(value=False); ttk.Checkbutton(frm_otros, text="Sigmoide", variable=self.var_sigmoid).grid(row=0, column=3, padx=2, pady=2, sticky="w")
        self.var_exp_decay = tk.BooleanVar(value=False); ttk.Checkbutton(frm_otros, text="Exp Decreciente", variable=self.var_exp_decay).grid(row=0, column=4, padx=2, pady=2, sticky="w")


        # --- Botones de Acción Finales ---
        frm_buttons_bottom = ttk.Frame(container)
        frm_buttons_bottom.pack(pady=10, fill="x", padx=10)
        btn_plot = ttk.Button(frm_buttons_bottom, text="Generar Dispersión y Regresión", command=self.plot_regression)
        btn_plot.pack(side="left", padx=5, expand=True, fill="x")
        btn_view = ttk.Button(frm_buttons_bottom, text="Ver Gráfica en Popup", command=self.view_graph_popup)
        btn_view.pack(side="left", padx=5, expand=True, fill="x")
        btn_save = ttk.Button(frm_buttons_bottom, text="Guardar Gráfica", command=self.save_graph_directly)
        btn_save.pack(side="left", padx=5, expand=True, fill="x")

    def log_message(self, msg, level=None):
        self.msg_label.config(text=msg)

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
            self.combo_dep_var_spec['values'] = num_cols
            if num_cols:
                self.combo_dep_var_spec.set(num_cols[0])
            else:
                self.combo_dep_var_spec.set("")
            # self.entry_dep_var_spec.delete(0, tk.END) # Replaced by Combobox logic
            # if num_cols: # Replaced by Combobox logic
            # self.entry_dep_var_spec.insert(0, num_cols[0]) # Replaced by Combobox logic
            
            # self.text_indep_vars_spec.delete("1.0", tk.END) # Replaced by Listbox logic
            # if num_cols: # Replaced by Listbox logic
                # for col in num_cols: # Replaced by Listbox logic
                    # self.text_indep_vars_spec.insert(tk.END, col + "\n") # Replaced by Listbox logic

            self.listbox_indep_vars_spec.delete(0, tk.END)
            selected_dep_var = self.combo_dep_var_spec.get()
            # Populate independent variables listbox, excluding the selected dependent variable
            # Using all_cols from earlier in the load_data method
            available_indep_vars = [col for col in all_cols if col != selected_dep_var]
            for var_name in available_indep_vars:
                self.listbox_indep_vars_spec.insert(tk.END, var_name)
            
            self._update_indep_vars_listbox() # Ensure list is correctly populated initially
            self.log_message("Datos cargados. Configure variables y filtros.")
        except Exception as e:
            self.log_message(f"Error al cargar datos: {e}")
            traceback.print_exc()
            # Add the messagebox here
            messagebox.showerror("Error al Cargar Archivo", 
                                 f"Ocurrió un error detallado al intentar cargar el archivo:\n\n{e}\n\nConsulte la consola para ver el traceback completo si es necesario.")
            self.data = None
            self.lbl_file.config(text="Ningún archivo cargado.")
            self.combo_dep_var_spec['values'] = []
            self.combo_dep_var_spec.set("")
            # self.entry_dep_var_spec.delete(0, tk.END) # Replaced by Combobox logic
            self.listbox_indep_vars_spec.delete(0, tk.END)
            # Limpiar componente de filtro en caso de error
            if hasattr(self, 'filter_component') and self.filter_component:
                self.filter_component.set_dataframe(None)

    def _update_indep_vars_listbox(self, event=None):
        if not hasattr(self, 'data') or self.data is None:
            return

        selected_dep_var = self.combo_dep_var_spec.get()
        current_indep_selection_indices = self.listbox_indep_vars_spec.curselection()
        current_indep_selected_values = [self.listbox_indep_vars_spec.get(i) for i in current_indep_selection_indices]

        self.listbox_indep_vars_spec.delete(0, tk.END)

        all_cols = list(self.data.columns)
        available_indep_vars = [col for col in all_cols if col != selected_dep_var]

        for var_name in available_indep_vars:
            self.listbox_indep_vars_spec.insert(tk.END, var_name)
            if var_name in current_indep_selected_values:
                # Try to reselect previously selected items if they are still valid
                try:
                    idx = available_indep_vars.index(var_name) # Get new index in the (potentially) changed list
                    self.listbox_indep_vars_spec.selection_set(idx)
                except ValueError: # Should not happen if var_name is in available_indep_vars
                    pass

    # Se eliminan apply_filter_criteria y apply_filter_qual

    def _get_filtered_data_for_regression(self):
        """Obtiene los datos filtrados usando el FilterComponent."""
        if self.data is None:
            self.log_message("Error: No hay datos cargados.")
            return None

        df_filtered = None
        if hasattr(self, 'filter_component') and self.filter_component:
            df_filtered = self.filter_component.apply_filters()
            if df_filtered is None:
                self.log_message("Error al aplicar filtros.")
                return None # El componente ya mostró el error
            self.log_message(f"Datos filtrados: {df_filtered.shape[0]} filas.")
        else:
            df_filtered = self.data.copy() # Usar original si no hay filtro
            self.log_message("Usando datos originales (sin filtros).")

        if df_filtered.empty:
            self.log_message("No hay datos después de aplicar filtros.")
            return pd.DataFrame() # Devolver DF vacío en lugar de None

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

    def plot_regression(self):
        if self.data is None:
            self.log_message("No hay datos cargados")
            return

        df_f = self._get_filtered_data_for_regression()
        if df_f is None or df_f.empty:
            self.log_message("No hay datos después de aplicar filtros o error en filtros para graficar.")
            return

        dep_var_selected = self.combo_dep_var_spec.get()
        if not dep_var_selected:
            self.log_message("Variable dependiente no seleccionada para graficar.")
            return
        dep_original, dep_display = dep_var_selected, dep_var_selected

        if dep_original not in df_f.columns:
            self.log_message(f"Error: Variable dependiente '{dep_original}' no encontrada en los datos filtrados.")
            return
        if not pd.api.types.is_numeric_dtype(df_f[dep_original]):
            self.log_message(f"Error: Variable dependiente '{dep_original}' no es numérica en los datos filtrados.")
            return

        selected_indices = self.listbox_indep_vars_spec.curselection()
        selected_indep_vars_names = [self.listbox_indep_vars_spec.get(i) for i in selected_indices]

        # The _parse_variable_specifications was also checking for numeric types and existence.
        # We need to replicate that an d build parsed_indep_specs structure.
        parsed_indep_specs = []
        if not selected_indep_vars_names:
            self.log_message("No se seleccionaron variables independientes.")
            # Depending on desired behavior, either return or allow plotting with no indep vars (just scatter of dep var if that makes sense)
            # For now, let's assume at least one independent variable is desired for regression lines.
            # If only a scatter plot of dependent vs independent is desired, this logic might change.
        else:
            for var_name in selected_indep_vars_names:
                if var_name not in df_f.columns:
                    self.log_message(f"Advertencia: Variable independiente '{var_name}' no encontrada en datos filtrados. Omitida.")
                    continue
                if not pd.api.types.is_numeric_dtype(df_f[var_name]):
                    self.log_message(f"Advertencia: Variable independiente '{var_name}' no es numérica. Omitida para regresión.")
                    continue
                parsed_indep_specs.append((var_name, var_name)) # Using var_name for both original and display name

        if not parsed_indep_specs:
            self.log_message("Variables independientes no válidas o no especificadas para graficar.")
            return
        
        try:
            dpi = int(self.entry_dpi.get()); w_px = int(self.entry_width.get()); h_px = int(self.entry_height.get())
            w_in, h_in = w_px/dpi, h_px/dpi; pt_size = float(self.entry_pt_size.get()); txt_size = int(self.entry_text_size.get())
        except ValueError: self.log_message("Error en parámetros DPI/tamaño."); return

        title_text = self.entry_title.get().strip() or f"Regresión de {dep_display} sobre Variables Seleccionadas"
        xlabel_text_base = self.entry_xlabel.get().strip() 
        ylabel_text = self.entry_ylabel.get().strip() or dep_display
        title_sz = int(self.entry_title_size.get())
        # title_col = self.entry_title_color.get().strip() # No existe este widget en la UI actual
        # xlabel_sz = int(self.entry_xlabel_size.get()) # No existe
        # xlabel_col = self.entry_xlabel_color.get().strip() # No existe
        # ylabel_sz = int(self.entry_ylabel_size.get()) # No existe
        # ylabel_col = self.entry_ylabel_color.get().strip() # No existe

        xlim_raw    = self.entry_xlim.get().strip()
        ylim_raw    = self.entry_ylim.get().strip()
        xticks_raw  = self.entry_xticks.get().strip()
        yticks_raw  = self.entry_yticks.get().strip()
        grid_on     = self.var_grid.get()
        show_info   = self.var_show_info.get()
        plot_corr   = self.var_plot_corr.get()
        
        fig, ax = plt.subplots(figsize=(w_in, h_in), dpi=dpi)
        results_list = []
        overall_scatter_x = []
        overall_scatter_y = []

        for idx, (indep_original, indep_display) in enumerate(parsed_indep_specs):
            if not pd.api.types.is_numeric_dtype(df_f[indep_original]):
                self.log_message(f"Advertencia: Variable '{indep_display}' no es numérica. Saltando.")
                continue

            temp_df = df_f[[dep_original, indep_original]].dropna()
            if temp_df.shape[0] < 2: continue

            x = temp_df[indep_original].values
            y = temp_df[dep_original].values
            
            current_color = self.default_colors[idx % len(self.default_colors)]
            ax.scatter(x, y, color=current_color, s=pt_size, alpha=0.6, label=f"{indep_display} (datos)")
            overall_scatter_x.extend(x)
            overall_scatter_y.extend(y)

            sort_idx = np.argsort(x)
            x_sorted = x[sort_idx]
            
            if self.var_linear.get():
                X_lin = sm.add_constant(x); mod = sm.OLS(y, X_lin).fit(); a,b = mod.params; yhat = mod.predict(X_lin)
                p,pp = safe_pearson(y,yhat); s,ps = safe_spearman(y,yhat); r2=mod.rsquared if not np.isnan(p) else np.nan
                results_list.append({"model":"Lineal", "var":indep_display, "dep_var":dep_display, "r":p, "r2":r2, "formula":f"{dep_display}={a:.2f}+{b:.2f}*{indep_display}\nP:{p:.3f}(p={fmt_p(pp)}) S:{s:.3f}(p={fmt_p(ps)})"})
                ax.plot(x_sorted, mod.predict(sm.add_constant(x_sorted)), linestyle=self.model_styles["Lineal"]["linestyle"], color=current_color, label=f"{indep_display} Lin (R²={r2:.3f})")
            if self.var_quadratic.get() and len(x) >=3:
                try:
                    c = np.polyfit(x,y,2); yhat=np.polyval(c,x); p,pp=safe_pearson(y,yhat); s,ps=safe_spearman(y,yhat); r2=p**2 if not np.isnan(p) else np.nan
                    results_list.append({"model":"Cuadrático", "var":indep_display, "dep_var":dep_display, "r":p, "r2":r2, "formula":f"{dep_display}={c[0]:.2f}*{indep_display}²+{c[1]:.2f}*{indep_display}+{c[2]:.2f}\nP:{p:.3f}(p={fmt_p(pp)}) S:{s:.3f}(p={fmt_p(ps)})"})
                    ax.plot(x_sorted, np.polyval(c,x_sorted), linestyle=self.model_styles["Cuadrático"]["linestyle"], color=current_color, label=f"{indep_display} Cuad (R²={r2:.3f})")
                except Exception as e: self.log_message(f"Error Cuad ({indep_display}): {e}")
            if self.var_cubic.get() and len(x) >= 4:
                try:
                    c = np.polyfit(x,y,3); yhat=np.polyval(c,x); p,pp=safe_pearson(y,yhat); s,ps=safe_spearman(y,yhat); r2=p**2 if not np.isnan(p) else np.nan
                    results_list.append({"model":"Cúbico", "var":indep_display, "dep_var":dep_display, "r":p, "r2":r2, "formula":f"{dep_display}={c[0]:.2f}*{indep_display}³+{c[1]:.2f}*{indep_display}²+{c[2]:.2f}*{indep_display}+{c[3]:.2f}\nP:{p:.3f}(p={fmt_p(pp)}) S:{s:.3f}(p={fmt_p(ps)})"})
                    ax.plot(x_sorted, np.polyval(c,x_sorted), linestyle=self.model_styles["Cúbico"]["linestyle"], color=current_color, label=f"{indep_display} Cúb (R²={r2:.3f})")
                except Exception as e: self.log_message(f"Error Cúbico ({indep_display}): {e}")
            if self.var_power.get():
                mask_p = (x > 0) & (y > 0)
                if mask_p.sum() > 2:
                    xp, yp = x[mask_p], y[mask_p]
                    try:
                        sl,it,_,_,_ = stats.linregress(np.log(xp),np.log(yp)); a,b=np.exp(it),sl; yhat=a*(xp**b)
                        p,pp=safe_pearson(yp,yhat); s,ps=safe_spearman(yp,yhat); r2=p**2 if not np.isnan(p) else np.nan
                        results_list.append({"model":"Potencia", "var":indep_display, "dep_var":dep_display, "r":p, "r2":r2, "formula":f"{dep_display}={a:.2f}*{indep_display}^{b:.2f}\nP:{p:.3f}(p={fmt_p(pp)}) S:{s:.3f}(p={fmt_p(ps)})"})
                        ax.plot(np.sort(xp), a*np.power(np.sort(xp),b), linestyle=self.model_styles["Potencia"]["linestyle"], color=current_color, label=f"{indep_display} Pot (R²={r2:.3f})")
                    except Exception as e: self.log_message(f"Error Potencia ({indep_display}): {e}")
            if self.var_log.get():
                mask_l = x > 0
                if mask_l.sum() > 2:
                    xp, yp = x[mask_l], y[mask_l]
                    try:
                        pop, _ = curve_fit(lambda z,a,b:a+b*np.log(z),xp,yp,maxfev=10000); yhat=pop[0]+pop[1]*np.log(xp)
                        p,pp=safe_pearson(yp,yhat); s,ps=safe_spearman(yp,yhat); r2=p**2 if not np.isnan(p) else np.nan
                        results_list.append({"model":"Logarítmico", "var":indep_display, "dep_var":dep_display, "r":p, "r2":r2, "formula":f"{dep_display}={pop[0]:.2f}+{pop[1]:.2f}*ln({indep_display})\nP:{p:.3f}(p={fmt_p(pp)}) S:{s:.3f}(p={fmt_p(ps)})"})
                        ax.plot(np.sort(xp), pop[0]+pop[1]*np.log(np.sort(xp)), linestyle=self.model_styles["Logarítmico"]["linestyle"], color=current_color, label=f"{indep_display} Log (R²={r2:.3f})")
                    except Exception as e: self.log_message(f"Error Log ({indep_display}): {e}")
            if self.var_loess.get() and len(x) > 5:
                try:
                    lo=lowess(y,x,frac=0.3); xs,ys=lo[:,0],lo[:,1]; acme_x,acme_y=compute_acme(lambda z:np.interp(z,xs,ys),np.linspace(xs.min(),xs.max(),200))
                    results_list.append({"model":"LOESS", "var":indep_display, "dep_var":dep_display, "r":np.nan, "r2":np.nan, "formula":f"LOESS para {indep_display}: acme en x={acme_x:.2f}, y={acme_y:.2f}"})
                    ax.plot(xs,ys, linestyle=self.model_styles["LOESS"]["linestyle"], color=current_color, label=f"{indep_display} LOESS")
                except Exception as e: self.log_message(f"Error LOESS ({indep_display}): {e}")
            
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
                    
                    popt_other, _ = curve_fit(model_func, x, y, p0=p0_other, maxfev=10000)
                    yhat_other = model_func(x, *popt_other)
                    pear_other, p_pear_other = safe_pearson(y, yhat_other)
                    spear_other, p_spear_other = safe_spearman(y, yhat_other)
                    r2_other = pear_other**2 if not np.isnan(pear_other) else np.nan
                    
                    formula_str_other = f"{model_name} ({dep_display} vs {indep_display}): "
                    if model_name == "Exp (a+b^x)": formula_str_other += f"{dep_display}={popt_other[0]:.2f}+{popt_other[1]:.2f}^{indep_display}"
                    elif model_name == "Exp (a+x^b)": formula_str_other += f"{dep_display}={popt_other[0]:.2f}+{indep_display}^{popt_other[1]:.2f}"
                    elif model_name == "Exp (A*B^x)": formula_str_other += f"{dep_display}={popt_other[0]:.2f}*{popt_other[1]:.2f}^{indep_display}"
                    elif model_name == "Sigmoide": formula_str_other += f"L={popt_other[0]:.2f},k={popt_other[1]:.2f},x0={popt_other[2]:.2f},off={popt_other[3]:.2f}"
                    elif model_name == "Exp Decreciente": formula_str_other += f"A={popt_other[0]:.2f},B={popt_other[1]:.2f}"
                    
                    results_list.append({"model":model_name,"var":indep_display, "dep_var":dep_display, "r":pear_other,"r2":r2_other, "formula":formula_str_other + f"\nP:{pear_other:.3f}(p={fmt_p(p_pear_other)}) S:{spear_other:.3f}(p={fmt_p(p_spear_other)})"})
                    ax.plot(x_sorted, model_func(x_sorted, *popt_other), linestyle=self.model_styles[model_name]["linestyle"], color=current_color, label=f"{indep_display} {model_name.split(' ')[0]} (R²={r2_other:.3f})")
                except RuntimeError: self.log_message(f"No se pudo ajustar {model_name} para {indep_display}.")
                except Exception as e_other_model: self.log_message(f"Error en {model_name} para {indep_display}: {e_other_model}")
        
        ax.set_xlabel(xlabel_text or (parsed_indep_specs[0][1] if len(parsed_indep_specs)==1 else "Variables Independientes"), fontsize=txt_size) # Usar txt_size para ejes
        ax.set_ylabel(ylabel_text or dep_display, fontsize=txt_size)
        ax.set_title(title_text or f"Regresión de {dep_display}", fontsize=title_sz) # Usar title_sz para título

        if grid_on: ax.grid(True, linestyle='--', alpha=0.7)
        else: ax.grid(False)

        if xlim_raw:
            try: 
                lo, hi = map(float, xlim_raw.split(','))
                ax.set_xlim(lo, hi) 
            except Exception as e: 
                self.log_message(f"Formato Xlim inválido (use min,max): {e}")
        if ylim_raw:
            try: 
                lo, hi = map(float, ylim_raw.split(','))
                ax.set_ylim(lo, hi)
            except Exception as e: 
                self.log_message(f"Formato Ylim inválido (use min,max): {e}")
        if xticks_raw:
            try: 
                ax.set_xticks(list(map(float, xticks_raw.split(','))))
            except Exception as e: 
                self.log_message(f"Formato Xticks inválido (use a,b,c): {e}")
        if yticks_raw:
            try: 
                ax.set_yticks(list(map(float, yticks_raw.split(','))))
            except Exception as e: 
                self.log_message(f"Formato Yticks inválido (use a,b,c): {e}")
        
        if plot_corr and len(overall_scatter_x) > 2 and len(overall_scatter_y) > 2:
            try:
                rp_overall, pp_overall = safe_pearson(overall_scatter_x, overall_scatter_y)
                rs_overall, ps_overall = safe_spearman(overall_scatter_x, overall_scatter_y)
                corr_text = f"Global (todas indep. vs dep):\nPearson: {rp_overall:.3f} (p={fmt_p(pp_overall)})\nSpearman: {rs_overall:.3f} (p={fmt_p(ps_overall)})"
                ax.annotate(corr_text, xy=(0.02, 0.02), xycoords="axes fraction", fontsize=max(6,txt_size-2), ha="left", va="bottom", bbox=dict(boxstyle="round,pad=0.3", fc="wheat", alpha=0.5))
            except Exception as e_corr: self.log_message(f"Error calculando correlación global: {e_corr}")

        if show_info:
            n_tot_plot = len(overall_scatter_x) 
            filtros_usados = []
            for cmb_w, entry_w in [(self.cmb_filter1, self.entry_filter1), (self.cmb_filter2, self.entry_filter2), 
                                   (self.cmb_filter_qual1, self.entry_filter_qual1), (self.cmb_filter_qual2, self.entry_filter_qual2)]:
                if cmb_w.get() and entry_w.get(): filtros_usados.append(f"{cmb_w.get()}: {entry_w.get()}")
            info_str = f"n (puntos graficados) = {n_tot_plot}"
            if filtros_usados: info_str += "\nFiltros: " + "; ".join(filtros_usados)
            ax.annotate(info_str, xy=(0.98, 0.98), xycoords="axes fraction", fontsize=max(6,txt_size-2), ha="right", va="top", bbox=dict(boxstyle="round,pad=0.3", fc="aliceblue", alpha=0.7))

        handles, labels = ax.get_legend_handles_labels()
        if handles: 
            ax.legend(fontsize=max(6, txt_size-2), loc='best')
        plt.tight_layout()
        fig.savefig(self.graph_path); plt.close(fig)

        results_list.sort(key=lambda x: x.get("r2", -1), reverse=True)
        summary = f"Resumen de Modelos para VD: {dep_display}\n" + ("-"*70) + "\n"
        for r_item in results_list: # Renombrado r a r_item para evitar conflicto con r de pearson
            summary += (f"Modelo: {r_item['model']} | VI: {r_item['var']}\n"
                        f"{r_item['formula']}\n"
                        f"  R² = {r_item.get('r2', np.nan):.3f}\n" + ("-"*70) + "\n")
        self.results_text_content = summary
        self.show_results_tab()
        self.log_message("Gráfica y resumen generados.")


    def show_results_tab(self):
        self.txt_results.config(state="normal")
        self.txt_results.delete("1.0", tk.END)
        self.txt_results.insert("1.0", self.results_text_content)
        self.txt_results.config(state="disabled")

    def view_graph_popup(self):
        if not os.path.exists(self.graph_path):
            self.log_message("No hay gráfica generada")
            return
        popup = tk.Toplevel(self)
        popup.title("Vista Ampliada de la Gráfica")
        try:
            img = tk.PhotoImage(file=self.graph_path)
            lbl = tk.Label(popup, image=img)
            lbl.image = img
            lbl.pack(fill="both", expand=True)
            popup.geometry("900x700") 
        except Exception as e:
            self.log_message(f"Error al abrir gráfica: {e}")
            if popup: popup.destroy()


    def save_graph_directly(self):
        if not os.path.exists(self.graph_path):
            self.log_message("No hay gráfica para guardar")
            return
        dest = filedialog.asksaveasfilename(initialfile="regresion_plot.png",
                                            defaultextension=".png",
                                            filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg;*.jpeg"), ("All files", "*.*")])
        if dest:
            try:
                shutil.copy2(self.graph_path, dest)
                self.log_message(f"Gráfica guardada en {dest}")
            except Exception as e:
                self.log_message(f"Error al guardar gráfica: {e}")

def run_tkinter_app():
    root = tk.Tk()
    root.title("Regresiones y Dispersión")
    root.geometry("1200x800")
    nb = ttk.Notebook(root)
    nb.pack(fill="both", expand=True)
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
