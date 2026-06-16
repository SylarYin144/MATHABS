#!/usr/bin/env python
# -*- coding: utf-8 -*-
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
from tkinter import colorchooser
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import json
import os

USER_PALETTES_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "user_palettes.json")

class ColorToolsTab(ttk.Frame):
    def __init__(self, parent, main_app_instance=None):
        super().__init__(parent, padding=10)
        self.main_app = main_app_instance
        self.selected_colors = []
        self.user_palettes = self._load_user_palettes()

        info = ttk.LabelFrame(self, text="Selector de Colores", padding=8)
        info.pack(fill=tk.BOTH, expand=True)

        top_row = ttk.Frame(info)
        top_row.pack(fill=tk.X, pady=(0,6))
        ttk.Label(top_row, text="Código Hex seleccionado:").pack(side=tk.LEFT)
        self.hex_var = tk.StringVar(value="#4C72B0")
        ttk.Entry(top_row, textvariable=self.hex_var, width=12).pack(side=tk.LEFT, padx=6)
        ttk.Button(top_row, text="Elegir color...", command=self.pick_color).pack(side=tk.LEFT)
        ttk.Button(top_row, text="Agregar a lista", command=self.add_color).pack(side=tk.LEFT, padx=4)

        palette_frame = ttk.LabelFrame(info, text="Paletas", padding=6)
        palette_frame.pack(fill=tk.X, pady=(0,6))
        palettes = [
            "tab10", "tab20", "tab20b", "tab20c",
            "Set1", "Set2", "Set3", "Pastel1", "Pastel2", "Accent", "Dark2", "Paired",
            "viridis", "plasma", "magma", "inferno", "cividis", "turbo",
            "Blues", "Greens", "Reds", "Purples", "OrRd", "PuBuGn", "RdYlGn", "Spectral", "coolwarm", "bwr", "cubehelix"
        ]
        self.palette_var = tk.StringVar(value=palettes[0])
        ttk.Label(palette_frame, text="Paleta:").pack(side=tk.LEFT)
        ttk.Combobox(palette_frame, textvariable=self.palette_var, values=palettes, state="readonly", width=14).pack(side=tk.LEFT, padx=4)
        ttk.Button(palette_frame, text="Cargar 8 colores", command=self.load_palette).pack(side=tk.LEFT, padx=4)

        saved_frame = ttk.Frame(palette_frame)
        saved_frame.pack(side=tk.LEFT, padx=8)
        ttk.Label(saved_frame, text="Paletas guardadas:").pack(anchor="w")
        self.saved_palette_var = tk.StringVar()
        self.saved_palette_combo = ttk.Combobox(saved_frame, textvariable=self.saved_palette_var, values=sorted(self.user_palettes.keys()), state="readonly", width=18)
        self.saved_palette_combo.pack(fill=tk.X)
        ttk.Button(saved_frame, text="Cargar guardada", command=self.load_saved_palette).pack(fill=tk.X, pady=(2,0))
        ttk.Button(saved_frame, text="Guardar actual...", command=self.save_current_palette).pack(fill=tk.X, pady=(2,0))
        ttk.Button(saved_frame, text="Borrar guardada", command=self.delete_saved_palette).pack(fill=tk.X, pady=(2,0))

        list_frame = ttk.LabelFrame(info, text="Colores recopilados", padding=6)
        list_frame.pack(fill=tk.BOTH, expand=True)
        self.colors_list = tk.Listbox(list_frame, height=8)
        self.colors_list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.colors_list.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.colors_list.configure(yscrollcommand=scrollbar.set)

        preview_frame = ttk.Frame(info)
        preview_frame.pack(fill=tk.X, pady=(6,0))
        ttk.Label(preview_frame, text="Formato para pegar en Gráficos:").pack(anchor="w")
        self.colors_text_var = tk.StringVar(value="")
        ttk.Entry(preview_frame, textvariable=self.colors_text_var, state="readonly").pack(fill=tk.X)

        buttons = ttk.Frame(info)
        buttons.pack(fill=tk.X, pady=(6,0))
        ttk.Button(buttons, text="Copiar coma-separado", command=self.copy_colors).pack(side=tk.LEFT)
        ttk.Button(buttons, text="Limpiar", command=self.clear_colors).pack(side=tk.LEFT, padx=4)

    def pick_color(self):
        try:
            rgb, hx = colorchooser.askcolor(parent=self)
            if hx:
                self.hex_var.set(hx)
        except Exception as exc:
            messagebox.showwarning("Color", f"No se pudo abrir el selector: {exc}", parent=self)

    def add_color(self):
        hx = self.hex_var.get().strip()
        if not hx:
            return
        self.selected_colors.append(hx)
        self.colors_list.insert(tk.END, hx)
        self._update_colors_text()

    def load_palette(self):
        name = self.palette_var.get() or "tab10"
        try:
            cmap = cm.get_cmap(name)
            colors = [mcolors.to_hex(cmap(i/ max(1,7))) for i in range(8)]
        except Exception:
            colors = []
        for c in colors:
            self.selected_colors.append(c)
            self.colors_list.insert(tk.END, c)
        self._update_colors_text()

    def load_saved_palette(self):
        name = self.saved_palette_var.get()
        if not name:
            return
        colors = self.user_palettes.get(name, [])
        if not colors:
            return
        self.selected_colors.extend(colors)
        for c in colors:
            self.colors_list.insert(tk.END, c)
        self._update_colors_text()

    def save_current_palette(self):
        if not self.selected_colors:
            messagebox.showinfo("Paleta", "No hay colores para guardar.", parent=self)
            return
        name = simpledialog.askstring("Guardar paleta", "Nombre de la paleta:", parent=self)
        if not name:
            return
        self.user_palettes[name] = list(self.selected_colors)
        self._save_user_palettes()
        self.saved_palette_combo['values'] = sorted(self.user_palettes.keys())
        self.saved_palette_var.set(name)

    def delete_saved_palette(self):
        name = self.saved_palette_var.get()
        if not name or name not in self.user_palettes:
            return
        del self.user_palettes[name]
        self._save_user_palettes()
        self.saved_palette_combo['values'] = sorted(self.user_palettes.keys())
        self.saved_palette_var.set("")

    def clear_colors(self):
        self.selected_colors.clear()
        self.colors_list.delete(0, tk.END)
        self._update_colors_text()

    def copy_colors(self):
        if not self.selected_colors:
            return
        s = self.colors_text_var.get() or ",".join(self.selected_colors)
        try:
            self.clipboard_clear()
            self.clipboard_append(s)
            messagebox.showinfo("Colores", "Copiado al portapapeles.", parent=self)
        except Exception:
            messagebox.showinfo("Colores", s, parent=self)

    def get_colors_text(self):
        return ",".join(self.selected_colors)

    def _update_colors_text(self):
        self.colors_text_var.set(",".join(self.selected_colors))

    def _load_user_palettes(self):
        if not os.path.exists(USER_PALETTES_PATH):
            return {}
        try:
            with open(USER_PALETTES_PATH, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _save_user_palettes(self):
        try:
            with open(USER_PALETTES_PATH, 'w', encoding='utf-8') as f:
                json.dump(self.user_palettes, f, ensure_ascii=False, indent=2)
        except Exception as exc:
            messagebox.showwarning("Paletas", f"No se pudo guardar la paleta: {exc}", parent=self)
