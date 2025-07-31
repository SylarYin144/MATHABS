#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tkinter as tk
from tkinter import ttk, colorchooser, messagebox
import json

class AppearanceTab(ttk.Frame):
    def __init__(self, master, app_instance, *args, **kwargs):
        super().__init__(master, *args, **kwargs)
        self.app_instance = app_instance
        self.vars = {}

        # --- Frame Principal con Scroll ---
        canvas = tk.Canvas(self)
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
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

        # --- Controles de Apariencia ---
        self._create_global_settings(scrollable_frame)
        self._create_widget_settings(scrollable_frame, "Pestañas (Tabs)", "tab")
        self._create_widget_settings(scrollable_frame, "Pestaña Seleccionada", "selected_tab")
        self._create_widget_settings(scrollable_frame, "Botones", "button")
        self._create_widget_settings(scrollable_frame, "Campos de Texto", "entry")

        # Botón para aplicar los cambios
        apply_button = ttk.Button(scrollable_frame, text="Aplicar Cambios de Apariencia", command=self._apply_styles)
        apply_button.pack(pady=20, padx=20)

    def _create_global_settings(self, parent):
        """Crea los controles para la configuración de fuente global."""
        frame = ttk.LabelFrame(parent, text="Configuración Global", padding="15")
        frame.pack(fill="x", pady=10, padx=10)

        ttk.Label(frame, text="Familia de Fuente General:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.vars['global_font_family'] = tk.StringVar(value="Palatino Linotype")
        font_families = sorted([
            "Palatino Linotype", "Georgia", "Garamond", "Times New Roman", "Arial",
            "Helvetica", "Calibri", "Verdana", "Tahoma", "Trebuchet MS", "Courier New", "Consolas"
        ])
        ttk.Combobox(frame, textvariable=self.vars['global_font_family'], values=font_families, state="readonly", width=30).grid(row=0, column=1, padx=5, pady=5)

    def _create_widget_settings(self, parent, title, key):
        """Helper para crear un conjunto de controles de apariencia para un widget."""
        frame = ttk.LabelFrame(parent, text=title, padding="15")
        frame.pack(fill="x", pady=10, padx=10)

        # Font settings
        ttk.Label(frame, text="Tamaño de Fuente:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.vars[f'{key}_font_size'] = tk.IntVar(value=10)
        ttk.Spinbox(frame, from_=8, to=24, textvariable=self.vars[f'{key}_font_size'], width=5).grid(row=0, column=1, padx=5, pady=5, sticky="w")

        ttk.Label(frame, text="Estilo de Fuente:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.vars[f'{key}_font_weight'] = tk.StringVar(value="normal")
        ttk.Combobox(frame, textvariable=self.vars[f'{key}_font_weight'], values=["normal", "bold"], state="readonly", width=10).grid(row=1, column=1, padx=5, pady=5, sticky="w")

        # Color settings
        self.vars[f'{key}_fg_color'] = tk.StringVar(value="black")
        ttk.Label(frame, text="Color de Fuente:").grid(row=2, column=0, padx=5, pady=5, sticky="w")
        ttk.Button(frame, text="Seleccionar", command=lambda k=key: self._choose_color(k, 'fg_color')).grid(row=2, column=1, padx=5, pady=5, sticky="w")
        self.vars[f'{key}_fg_preview'] = tk.Label(frame, text="■", fg=self.vars[f'{key}_fg_color'].get(), font=("Arial", 16))
        self.vars[f'{key}_fg_preview'].grid(row=2, column=2, padx=5)

        if key != 'entry': # Entries usually dont have a background color setting from the user
            self.vars[f'{key}_bg_color'] = tk.StringVar(value="#d9d9d9")
            ttk.Label(frame, text="Color de Fondo:").grid(row=3, column=0, padx=5, pady=5, sticky="w")
            ttk.Button(frame, text="Seleccionar", command=lambda k=key: self._choose_color(k, 'bg_color')).grid(row=3, column=1, padx=5, pady=5, sticky="w")
            self.vars[f'{key}_bg_preview'] = tk.Label(frame, text="■", fg=self.vars[f'{key}_bg_color'].get(), font=("Arial", 16))
            self.vars[f'{key}_bg_preview'].grid(row=3, column=2, padx=5)


    def _choose_color(self, key, type):
        """Abre el selector de color y actualiza la variable y la vista previa."""
        var_name = f'{key}_{type}_color'
        preview_name = f'{key}_{type}_preview'

        color_code = colorchooser.askcolor(title=f"Seleccionar Color de {type.replace('_', ' ').title()}")
        if color_code and color_code[1]:
            self.vars[var_name].set(color_code[1])
            self.vars[preview_name].config(fg=color_code[1])

    def _apply_styles(self):
        """Recopila todas las configuraciones y las envía a la aplicación principal."""
        styles = {
            'global_font_family': self.vars['global_font_family'].get()
        }

        for key in ["tab", "selected_tab", "button", "entry"]:
            styles[key] = {
                'font_size': self.vars[f'{key}_font_size'].get(),
                'font_weight': self.vars[f'{key}_font_weight'].get(),
                'fg_color': self.vars[f'{key}_fg_color'].get(),
                'bg_color': self.vars.get(f'{key}_bg_color', tk.StringVar(value=None)).get()
            }

        self.app_instance.update_global_styles(styles)
        messagebox.showinfo("Estilos Aplicados", "La nueva apariencia ha sido aplicada a toda la aplicación.", parent=self)

    def load_styles(self, styles):
        """Carga los estilos desde un diccionario (p. ej., desde un archivo de configuración)."""
        self.vars['global_font_family'].set(styles.get('global_font_family', 'Palatino Linotype'))

        for key in ["tab", "selected_tab", "button", "entry"]:
            widget_styles = styles.get(key, {})
            self.vars[f'{key}_font_size'].set(widget_styles.get('font_size', 10))
            self.vars[f'{key}_font_weight'].set(widget_styles.get('font_weight', 'normal'))

            fg_color = widget_styles.get('fg_color', 'black')
            self.vars[f'{key}_fg_color'].set(fg_color)
            self.vars[f'{key}_fg_preview'].config(fg=fg_color)

            if f'{key}_bg_color' in self.vars:
                bg_color = widget_styles.get('bg_color', '#d9d9d9')
                self.vars[f'{key}_bg_color'].set(bg_color)
                self.vars[f'{key}_bg_preview'].config(fg=bg_color)
