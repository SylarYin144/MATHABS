import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib.ticker as mticker
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backends._backend_tk import NavigationToolbar2Tk
from matplotlib.figure import Figure


SAFE_FUNCTIONS = {
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "asin": np.arcsin,
    "acos": np.arccos,
    "atan": np.arctan,
    "sinh": np.sinh,
    "cosh": np.cosh,
    "tanh": np.tanh,
    "sqrt": np.sqrt,
    "exp": np.exp,
    "log": np.log,
    "log10": np.log10,
    "log2": np.log2,
    "abs": np.abs,
    "floor": np.floor,
    "ceil": np.ceil,
    "sign": np.sign,
    "pow": np.power,
    "deg2rad": np.deg2rad,
    "rad2deg": np.rad2deg,
    "pi": np.pi,
    "e": np.e,
    "ln": np.log
}

COLOR_OPTIONS = [
    "tab:blue",
    "tab:orange",
    "tab:green",
    "tab:red",
    "tab:purple",
    "tab:brown",
    "tab:pink",
    "tab:gray",
    "tab:olive",
    "tab:cyan",
    "black"
]

FONT_OPTIONS = [
    "Arial",
    "Calibri",
    "Cambria",
    "Consolas",
    "Courier New",
    "Helvetica",
    "Segoe UI",
    "Tahoma",
    "Times New Roman",
    "Verdana"
]

SYMBOL_BUTTONS = [
    ("(", "("),
    (")", ")"),
    ("+", "+"),
    ("-", "-"),
    ("*", "*"),
    ("/", "/"),
    ("^", "**"),
    ("abs", "abs("),
    ("sqrt", "sqrt("),
    ("sin", "sin("),
    ("cos", "cos("),
    ("tan", "tan("),
    ("exp", "exp("),
    ("log", "log("),
    ("log10", "log10("),
    ("sinh", "sinh("),
    ("cosh", "cosh("),
    ("tanh", "tanh("),
    ("pi", "pi"),
    ("e", "e")
]

POINT_COLOR = "#d32f2f"


class GraphingCalculatorTab(ttk.Frame):
    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)

        self.sample_points = 1000
        self.current_function = None
        self.x_data = None
        self.y_data = None
        self.point_artist = None
        self.tangent_line_artist = None
        self.point_annotation = None
        self.canvas_click_cid = None

        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)

        # Variables de configuración
        self.x_min_var = tk.StringVar(value="-10")
        self.x_max_var = tk.StringVar(value="10")
        self.y_min_var = tk.StringVar(value="-10")
        self.y_max_var = tk.StringVar(value="10")
        self.color_var = tk.StringVar(value=COLOR_OPTIONS[0])
        self.linewidth_var = tk.StringVar(value="2")
        self.grid_var = tk.BooleanVar(value=True)
        self.font_var = tk.StringVar(value=FONT_OPTIONS[0])
        self.x_scale_var = tk.StringVar(value="linear")
        self.y_scale_var = tk.StringVar(value="linear")
        self.x_tick_var = tk.StringVar(value="")
        self.y_tick_var = tk.StringVar(value="")
        self.tick_size_var = tk.StringVar(value="11")
        self.status_var = tk.StringVar(value="Introduce una función para graficar.")

        self._build_ui()

    def _build_ui(self):
        control_frame = ttk.Frame(self)
        control_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=(10, 5))
        control_frame.columnconfigure(1, weight=1)

        # Fila de función
        ttk.Label(control_frame, text="f(x) =", font=("Arial", 12)).grid(row=0, column=0, sticky="w", padx=(0, 6))
        self.function_entry = ttk.Entry(control_frame, font=("Arial", 13))
        self.function_entry.grid(row=0, column=1, sticky="ew", pady=4)
        self.function_entry.bind("<Return>", lambda _event: self.plot_function())

        buttons_frame = ttk.Frame(control_frame)
        buttons_frame.grid(row=0, column=2, sticky="ew", padx=(8, 0))

        plot_button = ttk.Button(buttons_frame, text="Graficar", command=self.plot_function, style="Accent.TButton")
        plot_button.grid(row=0, column=0, padx=(0, 4))

        clear_button = ttk.Button(buttons_frame, text="Limpiar punto", command=self._clear_selected_point)
        clear_button.grid(row=0, column=1)

        reset_button = ttk.Button(buttons_frame, text="Restablecer", command=self._reset_settings)
        reset_button.grid(row=0, column=2, padx=(4, 0))

        style = ttk.Style()
        style.configure("Accent.TButton", font=("Arial", 11, "bold"))

        # Botones de símbolos
        symbols_frame = ttk.LabelFrame(control_frame, text="Símbolos rápidos")
        symbols_frame.grid(row=1, column=0, columnspan=3, sticky="ew", pady=6)
        symbols_columns = 10
        for index, (label, insertion) in enumerate(SYMBOL_BUTTONS):
            btn = ttk.Button(symbols_frame, text=label, width=6, command=lambda text=insertion: self._insert_symbol(text))
            row = index // symbols_columns
            col = index % symbols_columns
            btn.grid(row=row, column=col, padx=2, pady=2, sticky="ew")
        for col_index in range(symbols_columns):
            symbols_frame.columnconfigure(col_index, weight=1)

        # Configuración de ejes
        axis_frame = ttk.LabelFrame(control_frame, text="Rangos de ejes")
        axis_frame.grid(row=2, column=0, columnspan=3, sticky="ew", pady=6)
        for idx in range(4):
            axis_frame.columnconfigure(idx, weight=1)

        ttk.Label(axis_frame, text="X min:").grid(row=0, column=0, sticky="w", padx=(0, 4))
        ttk.Entry(axis_frame, textvariable=self.x_min_var, width=8).grid(row=0, column=1, sticky="ew", padx=(0, 8))
        ttk.Label(axis_frame, text="X max:").grid(row=0, column=2, sticky="w", padx=(0, 4))
        ttk.Entry(axis_frame, textvariable=self.x_max_var, width=8).grid(row=0, column=3, sticky="ew")

        ttk.Label(axis_frame, text="Y min:").grid(row=1, column=0, sticky="w", padx=(0, 4))
        ttk.Entry(axis_frame, textvariable=self.y_min_var, width=8).grid(row=1, column=1, sticky="ew", padx=(0, 8))
        ttk.Label(axis_frame, text="Y max:").grid(row=1, column=2, sticky="w", padx=(0, 4))
        ttk.Entry(axis_frame, textvariable=self.y_max_var, width=8).grid(row=1, column=3, sticky="ew")

        # Opciones de formato
        format_frame = ttk.LabelFrame(control_frame, text="Formato")
        format_frame.grid(row=3, column=0, columnspan=3, sticky="ew", pady=6)
        format_frame.columnconfigure(3, weight=1)

        ttk.Label(format_frame, text="Color:").grid(row=0, column=0, sticky="w")
        ttk.Combobox(format_frame, textvariable=self.color_var, values=COLOR_OPTIONS, width=10, state="readonly").grid(row=0, column=1, padx=4, pady=2)

        ttk.Label(format_frame, text="Grosor:").grid(row=0, column=2, sticky="w")
        linewidth_spin = ttk.Spinbox(format_frame, from_=0.5, to=10.0, increment=0.5, textvariable=self.linewidth_var, width=6)
        linewidth_spin.grid(row=0, column=3, padx=4, pady=2, sticky="w")

        grid_check = ttk.Checkbutton(format_frame, text="Mostrar cuadrícula", variable=self.grid_var)
        grid_check.grid(row=1, column=0, columnspan=2, sticky="w", pady=2)

        ttk.Label(format_frame, text="Fuente:").grid(row=1, column=2, sticky="w")
        ttk.Combobox(format_frame, textvariable=self.font_var, values=FONT_OPTIONS, width=15, state="readonly").grid(row=1, column=3, padx=4, pady=2, sticky="w")

        ttk.Label(format_frame, text="Tamaño ticks:").grid(row=2, column=0, sticky="w")
        ttk.Spinbox(format_frame, from_=6, to=24, increment=1, textvariable=self.tick_size_var, width=6).grid(row=2, column=1, padx=4, pady=2, sticky="w")

        # Escalas y ticks
        scale_frame = ttk.LabelFrame(control_frame, text="Escalas y pasos")
        scale_frame.grid(row=4, column=0, columnspan=3, sticky="ew", pady=(6, 0))

        ttk.Label(scale_frame, text="Escala X:").grid(row=0, column=0, sticky="w")
        ttk.Combobox(scale_frame, textvariable=self.x_scale_var, values=["linear", "log"], width=8, state="readonly").grid(row=0, column=1, padx=(0, 12))

        ttk.Label(scale_frame, text="Escala Y:").grid(row=0, column=2, sticky="w")
        ttk.Combobox(scale_frame, textvariable=self.y_scale_var, values=["linear", "log"], width=8, state="readonly").grid(row=0, column=3)

        ttk.Label(scale_frame, text="Paso ticks X:").grid(row=1, column=0, sticky="w", pady=4)
        ttk.Entry(scale_frame, textvariable=self.x_tick_var, width=10).grid(row=1, column=1, sticky="w")

        ttk.Label(scale_frame, text="Paso ticks Y:").grid(row=1, column=2, sticky="w", pady=4)
        ttk.Entry(scale_frame, textvariable=self.y_tick_var, width=10).grid(row=1, column=3, sticky="w")

        # Figura de Matplotlib
        self.fig = Figure(figsize=(6, 5), dpi=100)
        self.ax = self.fig.add_subplot(111)

        self.canvas = FigureCanvasTkAgg(self.fig, self)
        self.canvas.get_tk_widget().grid(row=1, column=0, sticky="nsew", padx=10, pady=5)

        toolbar_frame = ttk.Frame(self)
        toolbar_frame.grid(row=2, column=0, sticky="ew", padx=10)
        toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        toolbar.update()

        status_label = ttk.Label(self, textvariable=self.status_var, anchor="w")
        status_label.grid(row=3, column=0, sticky="ew", padx=10, pady=(0, 10))

        if self.canvas_click_cid is None:
            self.canvas_click_cid = self.canvas.mpl_connect('button_press_event', self._on_canvas_click)

    def _insert_symbol(self, text):
        current = self.function_entry.get()
        index = self.function_entry.index(tk.INSERT)
        new_text = current[:index] + text + current[index:]
        self.function_entry.delete(0, tk.END)
        self.function_entry.insert(0, new_text)
        self.function_entry.icursor(index + len(text))
        self.function_entry.focus_set()

    def _reset_settings(self):
        self.x_min_var.set("-10")
        self.x_max_var.set("10")
        self.y_min_var.set("-10")
        self.y_max_var.set("10")
        self.color_var.set(COLOR_OPTIONS[0])
        self.linewidth_var.set("2")
        self.grid_var.set(True)
        self.font_var.set(FONT_OPTIONS[0])
        self.x_scale_var.set("linear")
        self.y_scale_var.set("linear")
        self.x_tick_var.set("")
        self.y_tick_var.set("")
        self.tick_size_var.set("11")
        self.status_var.set("Valores restablecidos. Vuelve a graficar para aplicar cambios.")

    def _parse_float(self, value, fallback):
        if value is None:
            return fallback
        text = str(value).strip()
        if not text:
            return fallback
        return float(text)

    def plot_function(self):
        function_str = self.function_entry.get().strip()
        if not function_str:
            self.status_var.set("Ingresa una función antes de graficar.")
            return

        normalized_function = function_str.replace('^', '**')

        try:
            x_min = self._parse_float(self.x_min_var.get(), -10)
            x_max = self._parse_float(self.x_max_var.get(), 10)
            y_min_text = self.y_min_var.get().strip()
            y_max_text = self.y_max_var.get().strip()
            y_min = float(y_min_text) if y_min_text else None
            y_max = float(y_max_text) if y_max_text else None
            line_width = float(self.linewidth_var.get())
            tick_size = float(self.tick_size_var.get())
        except ValueError:
            messagebox.showerror("Valores inválidos", "Revisa que los límites y grosores sean numéricos.", parent=self)
            return

        if x_min >= x_max:
            messagebox.showerror("Rangos inválidos", "X min debe ser menor que X max.", parent=self)
            return

        x_scale = self.x_scale_var.get()
        y_scale = self.y_scale_var.get()

        if x_scale == "log" and (x_min <= 0 or x_max <= 0):
            messagebox.showerror("Escala logarítmica", "Para usar escala logarítmica en X, los límites deben ser positivos.", parent=self)
            return

        if y_scale == "log" and y_min is not None and y_min <= 0:
            messagebox.showerror("Escala logarítmica", "Para usar escala logarítmica en Y, el límite inferior debe ser positivo.", parent=self)
            return

        self.ax.clear()

        if x_scale == "log":
            x_values = np.logspace(np.log10(x_min), np.log10(x_max), self.sample_points)
        else:
            x_values = np.linspace(x_min, x_max, self.sample_points)

        try:
            y_values = self._evaluate_expression(normalized_function, x_values)
        except Exception as exc:
            self.status_var.set(f"No se pudo evaluar la función: {exc}")
            messagebox.showerror("Error al evaluar", str(exc), parent=self)
            self.canvas.draw_idle()
            return

        if np.iscomplexobj(y_values):
            self.status_var.set("La función generó valores complejos; no se puede graficar.")
            messagebox.showerror("Valores complejos", "La función produjo valores complejos.", parent=self)
            self.canvas.draw_idle()
            return

        y_values = np.asarray(y_values, dtype=float)

        if y_scale == "log" and np.any(y_values <= 0):
            self.status_var.set("La función tiene valores no positivos; no se puede usar escala logarítmica en Y.")
            messagebox.showerror("Escala logarítmica", "La función produce valores no positivos, no compatibles con escala logarítmica en Y.", parent=self)
            return

        self.ax.plot(x_values, y_values, color=self.color_var.get(), linewidth=line_width)
        self.ax.grid(self.grid_var.get(), linestyle='--', alpha=0.35)

        self.ax.set_xscale(x_scale)
        self.ax.set_yscale(y_scale)

        self.ax.set_xlabel("x", fontname=self.font_var.get())
        self.ax.set_ylabel("f(x)", fontname=self.font_var.get())
        self.ax.set_title(f"f(x) = {function_str}", fontname=self.font_var.get())

        if y_min is not None and y_max is not None and y_min < y_max:
            self.ax.set_ylim(y_min, y_max)
        elif y_scale == "linear":
            self.ax.set_ylim(np.nanmin(y_values), np.nanmax(y_values))

        self.ax.set_xlim(x_min, x_max)

        self.ax.tick_params(labelsize=tick_size)
        for label in list(self.ax.get_xticklabels()) + list(self.ax.get_yticklabels()):
            label.set_fontname(self.font_var.get())

        self._apply_tick_settings()

        self.current_function = normalized_function
        self.x_data = x_values
        self.y_data = y_values

        self._clear_selected_point(draw_canvas=False)

        self.status_var.set("Gráfica actualizada. Haz clic sobre la curva para marcar un punto.")
        self.canvas.draw_idle()

    def _apply_tick_settings(self):
        x_tick_step = self.x_tick_var.get().strip()
        y_tick_step = self.y_tick_var.get().strip()

        if self.x_scale_var.get() == "linear":
            if x_tick_step:
                try:
                    locator = mticker.MultipleLocator(float(x_tick_step))
                    self.ax.xaxis.set_major_locator(locator)
                except ValueError:
                    self.status_var.set("No se pudo aplicar el paso de ticks en X; revisa el valor.")
            else:
                self.ax.xaxis.set_major_locator(mticker.AutoLocator())
        else:
            self.ax.xaxis.set_major_locator(mticker.AutoLocator())

        if self.y_scale_var.get() == "linear":
            if y_tick_step:
                try:
                    locator = mticker.MultipleLocator(float(y_tick_step))
                    self.ax.yaxis.set_major_locator(locator)
                except ValueError:
                    self.status_var.set("No se pudo aplicar el paso de ticks en Y; revisa el valor.")
            else:
                self.ax.yaxis.set_major_locator(mticker.AutoLocator())
        else:
            self.ax.yaxis.set_major_locator(mticker.AutoLocator())

    def _evaluate_expression(self, expression, x_values):
        scope = dict(SAFE_FUNCTIONS)
        scope["x"] = x_values
        scope["np"] = np
        return eval(expression, {"__builtins__": {}}, scope)

    def _on_canvas_click(self, event):
        if event.inaxes != self.ax or not self.current_function:
            return

        if self.point_artist is not None:
            contains, _ = self.point_artist.contains(event)
            if contains:
                self._clear_selected_point()
                self.status_var.set("Marcador eliminado.")
                self.canvas.draw_idle()
                return

        x_coord = event.xdata
        if x_coord is None:
            return

        try:
            y_coord = float(self._evaluate_expression(self.current_function, x_coord))
        except Exception as exc:
            self.status_var.set(f"No se pudo evaluar en x = {x_coord:.4f}: {exc}")
            return

        if not np.isfinite(y_coord):
            self.status_var.set("El valor de la función no es finito en ese punto.")
            return

        derivative = self._compute_derivative(x_coord)
        if derivative is None or not np.isfinite(derivative):
            self.status_var.set("No se pudo calcular la derivada en ese punto.")
            return

        self._draw_point_and_tangent(x_coord, y_coord, derivative)
        self.status_var.set(f"x = {x_coord:.4f} | f(x) = {y_coord:.4f} | f'(x) = {derivative:.4f}")
        self.canvas.draw_idle()

    def _compute_derivative(self, x0):
        try:
            x_limits = self.ax.get_xlim()
            domain_span = max(abs(x_limits[1] - x_limits[0]), 1e-5)
        except Exception:
            domain_span = 1.0

        if self.x_scale_var.get() == "log":
            h = max(x0 * 0.01, 1e-5)
            lower_bound = max(self.ax.get_xlim()[0], 1e-8)
        else:
            h = max(domain_span / 1000.0, 1e-5)
            lower_bound = self.ax.get_xlim()[0]

        upper_bound = self.ax.get_xlim()[1]
        x_forward = min(upper_bound, x0 + h)
        x_backward = max(lower_bound, x0 - h)

        if x_forward <= x_backward:
            return None

        try:
            y_forward = float(self._evaluate_expression(self.current_function, x_forward))
            y_backward = float(self._evaluate_expression(self.current_function, x_backward))
        except Exception:
            return None

        if not (np.isfinite(y_forward) and np.isfinite(y_backward)):
            return None

        return (y_forward - y_backward) / (x_forward - x_backward)

    def _draw_point_and_tangent(self, x_coord, y_coord, derivative):
        self._clear_selected_point(draw_canvas=False)

        self.point_artist = self.ax.scatter([x_coord], [y_coord], color=POINT_COLOR, s=60, zorder=5)

        x_lower, x_upper = self.ax.get_xlim()
        if self.x_scale_var.get() == "log":
            x_lower = max(x_lower, 1e-8)
            x_line = np.logspace(np.log10(x_lower), np.log10(x_upper), 200)
        else:
            x_line = np.linspace(x_lower, x_upper, 200)

        y_line = y_coord + derivative * (x_line - x_coord)

        if self.y_scale_var.get() == "log":
            mask = y_line > 0
            x_line = x_line[mask]
            y_line = y_line[mask]
            if x_line.size == 0:
                self.status_var.set("La recta tangente excede los límites permitidos para la escala logarítmica.")
                return

        (self.tangent_line_artist,) = self.ax.plot(x_line, y_line, color=POINT_COLOR, linestyle='--', linewidth=1.6, alpha=0.8)

        self.point_annotation = self.ax.annotate(
            f"f(x)={y_coord:.3f}",
            xy=(x_coord, y_coord),
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=9,
            fontname=self.font_var.get(),
            color=POINT_COLOR,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=POINT_COLOR, alpha=0.8)
        )

    def _clear_selected_point(self, draw_canvas=True):
        if self.point_annotation is not None:
            self.point_annotation.remove()
            self.point_annotation = None

        if self.point_artist is not None:
            self.point_artist.remove()
            self.point_artist = None

        if self.tangent_line_artist is not None:
            self.tangent_line_artist.remove()
            self.tangent_line_artist = None

        if draw_canvas:
            self.canvas.draw_idle()


if __name__ == '__main__':
    root = tk.Tk()
    root.title("Calculadora Gráfica")

    style = ttk.Style(root)
    available_themes = style.theme_names()
    if 'clam' in available_themes:
        style.theme_use('clam')

    notebook = ttk.Notebook(root)

    graph_tab = GraphingCalculatorTab(notebook)
    notebook.add(graph_tab, text='Calculadora Gráfica')

    notebook.pack(expand=True, fill='both', padx=10, pady=10)

    root.geometry("800x700")
    root.mainloop()
