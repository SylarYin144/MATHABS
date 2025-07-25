import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class GraphingCalculatorTab(ttk.Frame):
    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)

        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)

        # Frame for input and controls
        control_frame = ttk.Frame(self)
        control_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        control_frame.grid_columnconfigure(1, weight=1)

        ttk.Label(control_frame, text="f(x) =", font=('Arial', 12)).grid(row=0, column=0, sticky="w", pady=5)
        self.function_entry = ttk.Entry(control_frame, font=('Arial', 14))
        self.function_entry.grid(row=0, column=1, sticky="ew", pady=5)

        plot_button = ttk.Button(control_frame, text="Plot", command=self.plot_function, style='Accent.TButton')
        plot_button.grid(row=0, column=2, padx=10, pady=5)

        # Estilo para el botón de Plot
        style = ttk.Style()
        style.configure('Accent.TButton', font=('Arial', 12, 'bold'))

        # Matplotlib Figure and Toolbar
        self.fig = plt.Figure(figsize=(5, 4), dpi=100)
        self.ax = self.fig.add_subplot(111)

        self.canvas = FigureCanvasTkAgg(self.fig, self)
        self.canvas.get_tk_widget().grid(row=1, column=0, columnspan=3, sticky="nsew")

        # Toolbar for navigation
        from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
        toolbar_frame = ttk.Frame(self)
        toolbar_frame.grid(row=2, column=0, columnspan=3, sticky="ew")
        toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        toolbar.update()

    def plot_function(self):
        function_str = self.function_entry.get()
        if not function_str:
            return

        self.ax.clear()
        x = np.linspace(-10, 10, 400)
        try:
            # Basic security: only allow safe functions
            safe_dict = {
                "x": x,
                "sin": np.sin,
                "cos": np.cos,
                "tan": np.tan,
                "sqrt": np.sqrt,
                "exp": np.exp,
                "log": np.log,
                "log10": np.log10,
                "pi": np.pi,
                "e": np.e
            }
            y = eval(function_str, {"__builtins__": {}}, safe_dict)
            self.ax.plot(x, y)
            self.ax.grid(True)
            self.ax.set_title(f"Graph of f(x) = {function_str}")
            self.ax.set_xlabel("x")
            self.ax.set_ylabel("f(x)")
        except Exception as e:
            self.ax.text(0.5, 0.5, f"Error: {e}", ha='center', va='center')

        self.canvas.draw()

if __name__ == '__main__':
    root = tk.Tk()
    root.title("Graphing Calculator Test")

    style = ttk.Style(root)
    available_themes = style.theme_names()
    if 'clam' in available_themes:
        style.theme_use('clam')

    notebook = ttk.Notebook(root)

    graph_tab = GraphingCalculatorTab(notebook)
    notebook.add(graph_tab, text='Graphing Calculator')

    notebook.pack(expand=True, fill='both', padx=10, pady=10)

    root.geometry("600x600")
    root.mainloop()
