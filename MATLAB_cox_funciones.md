# Análisis Detallado de Funciones en `MATABS/MATLAB_cox.py`

Este documento detalla las funciones programadas en el archivo `MATABS/MATLAB_cox.py`, que implementa una aplicación de escritorio para el modelado de supervivencia de Cox con una interfaz gráfica de usuario (GUI) utilizando Tkinter y librerías como `lifelines`, `pandas` y `matplotlib`.

La aplicación está estructurada en varias pestañas (`ttk.Notebook`) que guían al usuario a través del flujo de trabajo:

*   **Pestaña 1: Carga y Preprocesamiento de Datos**: Permite cargar archivos de datos (CSV, Excel), aplicar filtros avanzados (si el componente `MATLAB_filter_component` está disponible), definir las columnas de tiempo y evento, y configurar las covariables (tipo, categoría de referencia, splines). También incluye funciones para transformaciones logarítmicas y creación de nuevas variables por fórmula.
*   **Pestaña 2: Modelado Cox**: Aquí se configuran y ejecutan los modelos de Cox. Permite elegir entre modelado univariado o multivariado, seleccionar métodos de selección de variables (Backward, Forward, Stepwise), configurar la regularización (L1, L2, ElasticNet) y el manejo de empates. Una vez ejecutado el modelado, los modelos generados se listan en una `Treeview`, desde donde se pueden seleccionar para ver resultados detallados y generar gráficos.
*   **Pestaña 3: Resultados y Visualización**: Esta pestaña sirve como un área para configurar opciones generales de gráficos y, potencialmente, para mostrar resultados o comparaciones de modelos (aunque la mayoría de las visualizaciones se abren en ventanas separadas).
*   **Pestaña "Log"**: Muestra un registro detallado de las acciones de la aplicación, advertencias y errores, lo cual es crucial para la depuración y el seguimiento del proceso.

A continuación, se detallan las funciones más importantes de la clase principal `CoxModelingApp` y algunas funciones auxiliares clave:

---

## Funciones Auxiliares Globales

1.  **`format_p_value(p_val, threshold=0.0001)`**
    *   **Propósito**: Formatea un valor p para su visualización, usando notación científica si es menor que un umbral dado.
    *   **Entradas**: `p_val` (valor p numérico), `threshold` (umbral para notación científica, por defecto 0.0001).
    *   **Salidas**: Una cadena de texto formateada del valor p.
    *   **Interacciones**: Utilizada en `compute_model_metrics` y `_update_models_treeview` para presentar los p-valores de manera legible.

2.  **`compute_model_metrics(model, X_design, y_data, time_col, event_col, ...)`**
    *   **Propósito**: Calcula y organiza diversas métricas de evaluación para un modelo de Cox ajustado, como Log-Likelihood, AIC, BIC, C-Index (entrenamiento y CV), p-valores de Wald y resultados del test de Schoenfeld.
    *   **Entradas**: El objeto `CoxPHFitter` ajustado, la matriz de diseño `X_design`, los datos de supervivencia `y_data`, nombres de columnas de tiempo y evento, y resultados opcionales de C-Index CV y Schoenfeld.
    *   **Salidas**: Un diccionario con todas las métricas calculadas.
    *   **Interacciones**: Es llamada por `_run_model_and_get_metrics` después de ajustar el modelo para recopilar todas las métricas relevantes.

3.  **`apply_plot_options(ax, options_dict, log_func=print)`**
    *   **Propósito**: Aplica un conjunto de opciones de configuración (título, etiquetas de ejes, límites, escalas, ancho de línea, tamaño de marcador, rejilla) a un objeto `matplotlib.axes.Axes`.
    *   **Entradas**: Un objeto `Axes` de Matplotlib, un diccionario de opciones de gráfico y una función de logging.
    *   **Salidas**: Modifica el objeto `Axes` in-place.
    *   **Interacciones**: Utilizada por todas las funciones de visualización de gráficos (`show_baseline_survival`, `generar_forest_plot`, etc.) para estandarizar la apariencia de los plots.

---

## Clase `CoxModelingApp` (Métodos Clave)

### Métodos de Carga y Preprocesamiento (Pestaña 1)

1.  **`cargar_archivo()`**
    *   **Propósito**: Permite al usuario seleccionar y cargar un archivo de datos (CSV o Excel) en la aplicación.
    *   **Entradas**: Interacción del usuario a través de un diálogo de selección de archivo.
    *   **Proceso**: Abre un `filedialog`, lee el archivo con `pandas` (detectando el tipo por extensión), almacena los datos en `self.raw_data` y `self.data` (una copia para trabajar), y actualiza la UI.
    *   **Salidas**: Carga el DataFrame en la aplicación y actualiza los controles de preprocesamiento.
    *   **Interacciones**: Llama a `actualizar_controles_preproc` para poblar los comboboxes y listboxes con las columnas del archivo cargado.

2.  **`apply_covariate_config_to_selected()`**
    *   **Propósito**: Aplica la configuración de tipo (cuantitativa/cualitativa), spline o categoría de referencia a las covariables seleccionadas en la interfaz de usuario.
    *   **Entradas**: Selección de covariables en la `listbox` y valores de los controles de configuración (radio buttons, comboboxes, spinbox).
    *   **Proceso**: Itera sobre las covariables seleccionadas, actualiza los diccionarios `self.covariables_type_config`, `self.ref_categories_config` y `self.spline_config_details` con la configuración elegida.
    *   **Salidas**: Almacena la configuración de las covariables para su uso posterior en la construcción de la matriz de diseño.
    *   **Interacciones**: Es crucial porque esta configuración es utilizada por `build_design_matrix` para generar la fórmula Patsy y la matriz de diseño.

### Métodos de Modelado (Pestaña 2)

1.  **`_execute_cox_modeling_orchestrator()`**
    *   **Propósito**: Es la función principal que orquesta todo el proceso de modelado de Cox, desde la preparación de datos hasta el ajuste del modelo y la recopilación de resultados.
    *   **Entradas**: Configuración de la UI para el modelado (tipo de modelado, selección de variables, penalización, etc.).
    *   **Proceso**:
        1.  Llama a `_preparar_datos_para_modelado` para obtener el DataFrame listo para `lifelines` y la matriz de diseño inicial.
        2.  Determina los parámetros de penalización (`penalizer_value`, `l1_ratio`).
        3.  Si el modelado es "Univariado", itera sobre cada covariable seleccionada, llama a `build_design_matrix` para esa covariable y luego a `_run_model_and_get_metrics` para ajustar un modelo individual.
        4.  Si el modelado es "Multivariado", aplica la `_perform_variable_selection` si se ha elegido un método, y luego llama a `_run_model_and_get_metrics` para ajustar el modelo multivariado final.
        5.  Almacena los resultados de cada modelo en `self.generated_models_data`.
        6.  Llama a `_update_models_treeview` para mostrar los modelos generados en la UI.
    *   **Salidas**: Ajusta uno o varios modelos de Cox y los registra en la aplicación.
    *   **Interacciones**: Es el punto de entrada para el análisis, coordinando las llamadas a `_preparar_datos_para_modelado`, `build_design_matrix`, `_perform_variable_selection`, `_run_model_and_get_metrics` y `_update_models_treeview`.

2.  **`_preparar_datos_para_modelado()`**
    *   **Propósito**: Prepara el DataFrame de datos para ser utilizado por la librería `lifelines`. Esto incluye seleccionar las columnas de tiempo y evento, renombrarlas si es necesario, manejar valores `NaN` en estas columnas y construir la matriz de diseño inicial de covariables usando `patsy`.
    *   **Entradas**: El DataFrame `self.data` y las selecciones de columnas de tiempo, evento y covariables de la UI.
    *   **Proceso**:
        1.  Valida la selección de columnas de tiempo y evento.
        2.  Crea un sub-DataFrame con las columnas relevantes.
        3.  Renombra las columnas de tiempo y evento si se especificó.
        4.  Elimina filas con `NaN` en las columnas de tiempo o evento.
        5.  Llama a `build_design_matrix` para crear la matriz de diseño `X` y la fórmula Patsy.
    *   **Salidas**: Una tupla que contiene el DataFrame filtrado, la matriz de diseño `X`, los datos de supervivencia `y`, la fórmula Patsy, los nombres de los términos procesados y los nombres finales de las columnas de tiempo y evento.
    *   **Interacciones**: Es la primera función llamada por el orquestador de modelado. Depende de `build_design_matrix`.

3.  **`build_design_matrix(df_input, selected_covariates_original_names, time_col_name, event_col_name)`**
    *   **Propósito**: Construye la matriz de diseño (`X`) y la fórmula Patsy (`formula_patsy`) que `lifelines` espera como entrada para el ajuste del modelo. Maneja la transformación de variables según la configuración del usuario (cuantitativas, cualitativas con categoría de referencia, splines).
    *   **Entradas**: Un DataFrame de entrada, una lista de nombres de covariables originales seleccionadas, y los nombres finales de las columnas de tiempo y evento.
    *   **Proceso**:
        1.  Verifica la disponibilidad de `patsy`.
        2.  Si no hay covariables, crea una matriz de diseño vacía (para un modelo nulo).
        3.  Para cada covariable seleccionada, construye la sintaxis Patsy adecuada basándose en `self.covariables_type_config`, `self.ref_categories_config` y `self.spline_config_details`. Esto incluye el uso de `C()` para categóricas (con `Treatment()` para la categoría de referencia) y `cr()` o `bs()` para splines.
        4.  Combina las partes de la fórmula en una única cadena Patsy.
        5.  Usa `patsy.dmatrix` para generar la matriz de diseño `X`. `dmatrix` también se encarga de eliminar filas con `NaN` en las covariables utilizadas.
        6.  Filtra el DataFrame de entrada original para que coincida con el índice de la matriz de diseño resultante.
    *   **Salidas**: Una tupla con el DataFrame filtrado por Patsy, la matriz de diseño `X`, la fórmula Patsy generada y los nombres de los términos finales de Patsy.
    *   **Interacciones**: Es una función central en la preparación de datos, llamada por `_preparar_datos_para_modelado` y también directamente en el bucle de modelado univariado.

4.  **`_perform_variable_selection(df_aligned_original, X_design_initial, time_col, event_col, formula_initial, terms_initial)`**
    *   **Propósito**: Implementa la lógica de selección de variables (Backward, Forward, Stepwise) utilizando los métodos de `lifelines.CoxPHFitter`.
    *   **Entradas**: El DataFrame alineado, la matriz de diseño inicial completa, nombres de columnas de tiempo/evento, la fórmula Patsy inicial y los términos iniciales.
    *   **Proceso**:
        1.  Obtiene el método de selección y los umbrales de p-valor de la UI.
        2.  Crea un DataFrame combinado con `X_design_initial` y las columnas de tiempo/evento.
        3.  Utiliza los métodos `forward_select`, `backward_select` o `stepwise_select` de una instancia temporal de `CoxPHFitter` para determinar el subconjunto óptimo de covariables.
        4.  Filtra la matriz de diseño inicial para incluir solo los términos seleccionados.
    *   **Salidas**: El DataFrame alineado (sin cambios de filas), la matriz de diseño `X` con las covariables seleccionadas, la fórmula Patsy actualizada y los nombres de los términos seleccionados.
    *   **Interacciones**: Es llamada por `_execute_cox_modeling_orchestrator` cuando el usuario ha elegido un método de selección de variables.

5.  **`_run_model_and_get_metrics(df_lifelines, X_design, y_survival, time_col, event_col, formula_patsy, model_name, covariates_display_terms, penalizer_val=0.0, l1_ratio_val=0.0)`**
    *   **Propósito**: Ajusta un modelo `CoxPHFitter` con los datos y configuraciones proporcionadas, y luego calcula y recopila todas las métricas de evaluación relevantes.
    *   **Entradas**: El DataFrame listo para `lifelines`, la matriz de diseño `X`, los datos de supervivencia `y`, nombres de columnas de tiempo/evento, la fórmula Patsy, un nombre para el modelo y parámetros de penalización.
    *   **Proceso**:
        1.  Ajusta un modelo nulo (sin covariables) para obtener el Log-Likelihood nulo (necesario para el test LR global).
        2.  Ajusta el modelo principal `CoxPHFitter` utilizando `X_design`, duraciones y eventos observados, aplicando la penalización y el método de manejo de empates configurados.
        3.  Si el modelo tiene covariables, realiza el test de residuos de Schoenfeld (`lifelines.statistics.proportional_hazard_test`) para evaluar el supuesto de riesgos proporcionales.
        4.  Si se solicitó, calcula el C-Index por validación cruzada (`KFold`).
        5.  Llama a la función auxiliar `compute_model_metrics` para consolidar todas las métricas.
    *   **Salidas**: Un diccionario `model_data` que contiene el objeto `CoxPHFitter` ajustado, los datos utilizados para el ajuste, los resultados de Schoenfeld, el Log-Likelihood nulo, los C-Indices CV y el diccionario de métricas.
    *   **Interacciones**: Es el componente que realmente ajusta el modelo y es llamado por `_execute_cox_modeling_orchestrator` para cada modelo a generar.

### Métodos de Visualización y Reportes (Pestaña 3 y Acciones de Pestaña 2)

1.  **`generar_forest_plot()`**
    *   **Propósito**: Genera un Forest Plot que visualiza los Hazard Ratios (HRs) y sus intervalos de confianza del 95% para cada covariable en el modelo seleccionado.
    *   **Entradas**: El modelo seleccionado (`self.selected_model_in_treeview`).
    *   **Proceso**:
        1.  Obtiene el `summary_df` del modelo (que contiene los HRs y CIs).
        2.  Ordena el DataFrame para el plot según la opción de ordenamiento seleccionada en la UI (`sort_order`).
        3.  Crea una figura y ejes de Matplotlib.
        4.  Plotea los HRs como puntos y los intervalos de confianza como barras de error.
        5.  Añade una línea vertical en HR=1 (no efecto).
        6.  Aplica las opciones de gráfico globales (`self.current_plot_options`).
        7.  Muestra el gráfico en una nueva ventana usando `_create_plot_window`.
    *   **Salidas**: Una ventana emergente con el Forest Plot.
    *   **Interacciones**: Depende de `_check_model_selected_and_valid` y `_create_plot_window`.

2.  **`realizar_prediccion()` y `_perform_prediction_and_plot(...)`**
    *   **Propósito**: Permite al usuario ingresar valores para las covariables de un modelo seleccionado y luego predice y plotea la función de supervivencia o riesgo acumulado para ese perfil de paciente.
    *   **Entradas**: `realizar_prediccion` abre un diálogo que solicita valores para las covariables y tiempos de predicción. `_perform_prediction_and_plot` recibe estos valores.
    *   **Proceso**:
        1.  `realizar_prediccion` identifica las variables originales del modelo y crea campos de entrada en un `Toplevel` para que el usuario ingrese valores.
        2.  `_perform_prediction_and_plot` valida los tiempos de predicción y los valores de entrada.
        3.  Crea un `DataFrame` de una fila con los valores de entrada.
        4.  Utiliza `patsy.dmatrix` con la `design_info_` del modelo original para transformar los valores de entrada en la matriz de diseño (`X_pred_patsy`) esperada por el modelo.
        5.  Llama a `cph_model.predict_survival_function` o `cph_model.predict_cumulative_hazard` para obtener la curva de predicción.
        6.  Plotea la curva y marca los puntos de los tiempos específicos solicitados.
        7.  Muestra los resultados numéricos en un `messagebox`.
    *   **Salidas**: Una ventana emergente con el gráfico de la curva de supervivencia/riesgo predicha y un `messagebox` con los valores numéricos en los tiempos especificados.
    *   **Interacciones**: Depende de `_check_model_selected_and_valid`, `dmatrix` de `patsy` y `_create_plot_window`.

3.  **`show_methodological_report()`**
    *   **Propósito**: Genera y muestra un reporte metodológico detallado para el modelo seleccionado, incluyendo la configuración del ajuste, coeficientes, métricas y resultados del test de Schoenfeld.
    *   **Entradas**: El modelo seleccionado (`self.selected_model_in_treeview`).
    *   **Proceso**: Reutiliza `_generate_text_summary_for_model` para obtener el resumen técnico y añade secciones introductorias y de contexto metodológico.
    *   **Salidas**: Una nueva ventana (`ModelSummaryWindow`) que muestra el reporte completo.
    *   **Interacciones**: Depende de `_check_model_selected_and_valid` y `_generate_text_summary_for_model`.