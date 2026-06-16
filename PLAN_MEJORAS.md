# Plan de mejoras para Mathabs

Este documento resume los hallazgos más relevantes tras revisar los módulos principales recientemente editados (`MATLAB_general_charts.py`, `MATLAB_data_filter.py`, `MATLAB_main_app.py`, `MATLAB_graficaqq.py`, `workflow_tab.py`) y el registro de pendientes existente en `TODO.md`.

## 1. Cobertura funcional incompleta
- **`MATLAB_general_charts.py` (líneas ~90-110 y ~700-1400):** el tipo "Gráfico Circular / Anillo" existe en la lista de opciones, pero no hay rama correspondiente dentro de `_generate_chart`. Esto genera una opción engañosa para el usuario y explica por qué las "tortas" solo funcionan en el módulo antiguo (`MATLAB_data_filter.py`).
  - *Acción:* implementar la rama que construya pie/donut charts reutilizando la lógica ya probada en `_plot_pie_chart` de `MATLAB_data_filter.py` (idealmente extraída a utilería compartida) y respetar el selector de recodificación/orden existente en controles generales.

## 2. Función monolítica difícil de mantener
- **`_generate_chart` en `MATLAB_general_charts.py` (>1.600 líneas):** concentra toda la lógica de graficación con condicionales enormes y parámetros repetidos. Esto ralentiza cualquier fix (p.ej. el bug de etiquetas) y dificulta las pruebas unitarias.
  - *Acción:* extraer cada tipo de gráfico a métodos privados (`_render_histograma`, `_render_barras`, etc.) y encapsular estructuras auxiliares (paletas, layouts, anotaciones). Complementar con un registro tipo diccionario `{chart_type: handler}` para reducir la cascada de `elif`.

## 3. Inconsistencia en el flujo de datasets compartidos
- **`WorkflowTab` vs. `MainApp`:** la nueva pestaña ofrece un resumen del dataset, pero solo se actualiza si el usuario presiona "Actualizar". No se suscribe automáticamente a `receive_shared_dataset`, por lo que la guía queda desalineada con el resto de tabs.
  - *Acción:* registrar `WorkflowTab` en la lista de widgets que reciben `shared_dataset` (ver cómo `GeneralChartsApp` lo hace) y disparar `_update_summary` en cada broadcast. También aprovechar para mostrar alertas (colores) cuando no hay dataset compartido.

## 4. Controles de filtros en `GraficaQQ`
- **`MATLAB_graficaqq.py`:** el `FilterComponent` fue eliminado, pero persisten métodos (`_apply_general_filters`, atributos `filter_col_*`) que llaman a `self.log` (no definido) y a widgets que ya no se generan. Esto implica rutas muertas y posibles excepciones si se activan desde otro módulo.
  - *Acción:* eliminar las referencias restantes al componente antiguo o reemplazarlas con un filtro inline coherente. Si se necesita logging, integrar con el patrón usado en `GeneralChartsApp`.

## 5. Reutilización limitada entre módulos de gráficos
- Existen dos implementaciones distintas para tareas similares (recodificación, selección de paletas, guardado de PNG) en `MATLAB_general_charts.py` y `MATLAB_data_filter.py`. Los bugs corregidos en uno no llegan al otro (p.ej. limpieza de etiquetas en pie charts que acabamos de ajustar).
  - *Acción:* mover utilidades comunes (parsing de etiquetas, generación de paletas, guardado con `bbox_inches`) a un módulo `chart_utils.py` y consumirlo desde ambos tabs. Esto reducirá duplicidad y permitirá probar funciones aisladas.

## 6. Experiencia de usuario en la pestaña "Flujo de Trabajo"
- Actualmente la tabla es estática: no hay enlaces directos que abran las pestañas sugeridas ni indicadores visuales del estado real (por ejemplo, si ya se compartió dataset). Además, el texto de detalle no se puede copiar fácilmente porque el `Text` está deshabilitado, ni se ofrecen acciones rápidas.
  - *Acción:* agregar botones contextualizados ("Abrir pestaña"), un resumen dinámico con íconos y permitir copiar el detalle (toggle `state`). Considerar informar si faltan pasos críticos (sin filtros aplicados, sin gráficos guardados, etc.).

## 7. Calidad y pruebas
- No existen pruebas automatizadas para las transformaciones de datos (recodificación, normalización de pies, filtros). La ausencia de tests impide detectar regresiones (como la conversión de `NaN` en la gráfica circular).
  - *Acción:* crear un paquete `tests/` con casos mínimos usando `pytest`, empezando por utilidades puras (`_apply_recode`, `_plot_pie_chart` helpers). Integrar en CI si es posible.

## 8. Pending items en `TODO.md`
- Las tres tareas listadas (ajustes en `MATLAB_cox.py`, asignaciones a `self.result[...]`, defensas para `selected_model_in_treeview`) siguen abiertas. Deberían etiquetarse con responsable y prioridad, o migrarse a un sistema de issues para no perderlas.
  - *Acción:* revisar `MATLAB_cox.py` para confirmar si los cambios ya se aplicaron y cerrar el pendiente; si no, documentar el impacto (warnings tipo `MatplotlibDeprecationWarning`).

---
**Siguiente paso sugerido:** priorizar la refactorización del módulo de gráficas generales (puntos 1 y 2). Con ese trabajo en marcha se desbloquea la reutilización de pie charts y la incorporación de pruebas unitarias descritas en los apartados 5 y 7.
