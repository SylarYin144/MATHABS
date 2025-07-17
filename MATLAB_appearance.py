#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tkinter as tk
from tkinter import ttk, colorchooser
import json

class AppearanceTab(ttk.Frame):
    def __init__(self, master, app_instance, *args, **kwargs):
        super().__init__(master, *args, **kwargs)
        self.app_instance = app_instance

        # --- Frame Principal ---
        main_frame = ttk.Frame(self, padding="20")
        main_frame.pack(fill="both", expand=True)

        # --- Controles de Tipografía ---
        font_frame = ttk.LabelFrame(main_frame, text="Configuración de Fuente Global", padding="15")
        font_frame.pack(fill="x", pady=10)

        # Selección de Familia de Fuente
        ttk.Label(font_frame, text="Familia de Fuente:").grid(row=0, column=0, padx=5, pady=5, sticky="w")

        # Fuentes comunes y seguras
        self.font_families = ["Palatino Linotype", "Arial", "Helvetica", "Times New Roman", "Courier New", "Verdana", "Tahoma"]
        self.font_var = tk.StringVar()
        self.font_combo = ttk.Combobox(font_frame, textvariable=self.font_var, values=self.font_families, state="readonly", width=30)
        self.font_combo.grid(row=0, column=1, padx=5, pady=5)
        self.font_combo.set("Palatino Linotype") # Default

        # Selección de Tamaño de Fuente
        ttk.Label(font_frame, text="Tamaño de Fuente:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.size_var = tk.IntVar(value=10)
        self.size_spinbox = ttk.Spinbox(font_frame, from_=8, to=20, textvariable=self.size_var, width=5)
        self.size_spinbox.grid(row=1, column=1, padx=5, pady=5, sticky="w")

        # Selección de Color de Fuente
        ttk.Label(font_frame, text="Color de Fuente:").grid(row=2, column=0, padx=5, pady=5, sticky="w")
        self.color_var = tk.StringVar(value="black")
        self.color_button = ttk.Button(font_frame, text="Seleccionar Color", command=self._choose_color)
        self.color_button.grid(row=2, column=1, padx=5, pady=5, sticky="w")
        self.color_preview = tk.Label(font_frame, text="■", fg=self.color_var.get(), font=("Arial", 16))
        self.color_preview.grid(row=2, column=2, padx=5)

        # Botón para aplicar los cambios
        apply_button = ttk.Button(main_frame, text="Aplicar Cambios de Apariencia", command=self._apply_styles)
        apply_button.pack(pady=20)

    def _choose_color(self):
        color_code = colorchooser.askcolor(title="Seleccionar Color de Fuente")
        if color_code and color_code[1]:
            self.color_var.set(color_code[1])
            self.color_preview.config(fg=self.color_var.get())

    def _apply_styles(self):
        # Esta función llamará a la lógica principal en MainApp
        font_family = self.font_var.get()
        font_size = self.size_var.get()
        font_color = self.color_var.get()

        self.app_instance.update_global_styles(font_family, font_size, font_color)
        messagebox.showinfo("Estilos Aplicados", "La nueva apariencia ha sido aplicada a toda la aplicación.", parent=self)
