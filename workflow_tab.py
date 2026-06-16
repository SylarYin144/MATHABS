import tkinter as tk
from tkinter import ttk
from typing import Optional, Sequence


class WorkflowTab(ttk.Frame):
    """Pestaña informativa que resume el flujo de trabajo sugerido."""

    def __init__(self, parent, main_app_instance=None):
        super().__init__(parent)
        self.main_app = main_app_instance
        self.current_dataset = None
        self.current_filtered = None
        self.current_filter_summary: Sequence[str] = []
        self._build_ui()
        self._render_dataset_summary()

    def _build_ui(self):
        header = ttk.Label(
            self,
            text=(
                "Flujo lógico para pasar de datos crudos a resultados accionables. "
                "Cada etapa referencia las pestañas ya disponibles en Mathabs."
            ),
            wraplength=900,
            justify=tk.LEFT
        )
        header.pack(fill=tk.X, padx=10, pady=10)

        summary_frame = ttk.LabelFrame(self, text="Estado actual de los datos compartidos")
        summary_frame.pack(fill=tk.X, padx=10, pady=(0, 10))

        info_container = ttk.Frame(summary_frame)
        info_container.pack(fill=tk.X, padx=5, pady=5)

        self.dataset_status_var = tk.StringVar(value="No hay dataset compartido aún.")
        status_label = ttk.Label(info_container, textvariable=self.dataset_status_var, justify=tk.LEFT)
        status_label.pack(side=tk.LEFT, fill=tk.X, expand=True)

        ttk.Button(summary_frame, text="Actualizar", command=self._refresh_dataset_summary).pack(padx=5, pady=5, anchor="e")

        steps_frame = ttk.LabelFrame(self, text="Ruta recomendada")
        steps_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        columns = ("objetivo", "acciones", "pestanas")
        self.tree = ttk.Treeview(steps_frame, columns=columns, show="headings", height=8)
        self.tree.heading("objetivo", text="Objetivo")
        self.tree.heading("acciones", text="Acciones clave")
        self.tree.heading("pestanas", text="Pestañas / Herramientas")
        self.tree.column("objetivo", width=220, anchor=tk.W)
        self.tree.column("acciones", width=420, anchor=tk.W)
        self.tree.column("pestanas", width=260, anchor=tk.W)
        self.tree.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        for step in self._workflow_steps():
            self.tree.insert(
                "",
                tk.END,
                values=(step["objetivo"], step["acciones"], step["pestanas"])
            )

        detail_frame = ttk.LabelFrame(self, text="Detalles operativos")
        detail_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        self.detail_text = tk.Text(detail_frame, height=12, wrap=tk.WORD)
        self.detail_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.detail_text.insert(tk.END, self._build_detail_text())
        self.detail_text.config(state=tk.DISABLED)

    def _workflow_steps(self):
        return [
            {
                "objetivo": "1. Consolidar datos",
                "acciones": "Definir archivo maestro, documentar variables críticas, aplicar formatos.",
                "pestanas": "Archivo de Trabajo, Apariencia"
            },
            {
                "objetivo": "2. Filtrar y preparar",
                "acciones": "Aplicar filtros globales, crear vistas derivadas, compartir dataset limpio.",
                "pestanas": "Archivo de Trabajo, Filtros avanzados, Compartir dataset"
            },
            {
                "objetivo": "3. Exploración visual",
                "acciones": "Identificar patrones iniciales, validar supuestos, guardar gráficas clave.",
                "pestanas": "Gráficas Generales, Calculadora Gráfica"
            },
            {
                "objetivo": "4. Comparación de grupos",
                "acciones": "Verificar normalidad, pruebas globales y pareadas, exportar resúmenes.",
                "pestanas": "Comparación de Medias, Tablas"
            },
            {
                "objetivo": "5. Modelado estadístico",
                "acciones": "Seleccionar familia de modelos, ajustar parámetros, validar supuestos.",
                "pestanas": "Regresiones, Cox, AFT, Mezclas, Logistic/Linear"
            },
            {
                "objetivo": "6. Resultados y comunicación",
                "acciones": "Construir reportes, guardar figuras finales, documentar decisiones.",
                "pestanas": "Análisis y Gráficos, Apariencia, Exportaciones"
            },
        ]

    def _build_detail_text(self) -> str:
        return (
            "Preparación (Archivo de Trabajo): \n"
            "  • Importa archivos, armoniza nombres y usa 'Guardar dataset compartido'.\n"
            "Filtrado y documentación: \n"
            "  • Usa los filtros globales y guarda resúmenes en la bitácora.\n"
            "Exploración visual: \n"
            "  • En 'Gráficas Generales' prioriza histogramas, forest plots y mapas de calor.\n"
            "Comparaciones: \n"
            "  • 'Comparación de Medias' ofrece modo resumen/detallado y pop-up para gráficas.\n"
            "Modelado: \n"
            "  • Escoge el tab adecuado (Regresión lineal, logística, Cox, AFT o MixModel) según la métrica.\n"
            "Resultados finales: \n"
            "  • Usa 'Análisis y Gráficos' para integrar tablas, gráficos y comentarios antes de exportar."
        )

    def _refresh_dataset_summary(self):
        if not self.main_app:
            self.dataset_status_var.set("La aplicación principal no está disponible.")
            return
        dataset = getattr(self.main_app, "shared_dataset", None)
        filtered = getattr(self.main_app, "shared_filtered_dataset", None)
        summary = getattr(self.main_app, "shared_filter_summary", [])
        self._update_summary(dataset, filtered, summary)

    def _update_summary(self, dataset: Optional[object], filtered: Optional[object], summary):
        if dataset is None:
            self.dataset_status_var.set("No hay dataset compartido. Usa 'Archivo de Trabajo' para compartir uno.")
            return
        rows, cols = dataset.shape if hasattr(dataset, "shape") else ("?", "?")
        text = f"Dataset compartido: {rows} filas x {cols} columnas"
        if filtered is not None and hasattr(filtered, "shape"):
            f_rows, f_cols = filtered.shape
            text += f" | Vista filtrada: {f_rows} filas x {f_cols} columnas"
        if summary:
            text += "\nFiltros aplicados: " + "; ".join(str(item) for item in summary[:4])
            if len(summary) > 4:
                text += " ..."
        self.dataset_status_var.set(text)

    def _render_dataset_summary(self):
        self._update_summary(self.current_dataset, self.current_filtered, self.current_filter_summary)

    # Método compatible con la infraestructura de datasets compartidos del MainApp
    def receive_shared_dataset(self, dataset=None, filtered_dataset=None, filter_summary=None, metadata=None, source_widget=None):
        self.current_dataset = dataset
        self.current_filtered = filtered_dataset
        self.current_filter_summary = filter_summary or []
        self._render_dataset_summary()
