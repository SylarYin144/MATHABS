# Plan Detallado para el Módulo de Análisis de Supervivencia en Python

**Objetivo General:** Implementar un módulo robusto de análisis de supervivencia en Python, replicando las funcionalidades de carga de datos, preprocesamiento, ajuste de modelos, extracción de métricas y generación de datos para gráficos, según la especificación detallada.

**Archivo Objetivo Principal:** [`MATABS/MATLAB_cox2.py`](MATABS/MATLAB_cox2.py)

**Consideraciones Adicionales del Usuario:**
*   Los filtros avanzados se tomarán de `D:\APPS\MATABS\MATLAB_filter_component.py`. Esto ya está contemplado en la importación y uso de `FilterComponent` y la función `apply_advanced_filters` en `MATLAB_cox2.py`.
*   Este módulo es parte de `main_app.py` y no una pestaña independiente. La implementación actual en `MATLAB_cox2.py` incluye un bloque `if __name__ == "__main__":` para demostración y pruebas. Para la integración final en `main_app.py`, este bloque se adaptará o eliminará, y las funciones principales (`run_survival_analysis`, `load_data`, `preprocess_data`, etc.) serán llamadas directamente desde la lógica de la aplicación principal.

---

### Fases del Plan:

El plan se estructura siguiendo las secciones de la especificación proporcionada, destacando las funcionalidades ya implementadas y las que requieren ajustes o confirmación.

#### **Fase 1: Configuración del Proyecto y Carga de Datos**

*   **Objetivo:** Establecer el entorno Python y cargar los datos iniciales.
*   **Pasos:**
    1.  **Verificar Importaciones de Librerías:** Asegurar que todas las librerías necesarias (`pandas`, `numpy`, `patsy`, `lifelines`, `sklearn.impute.SimpleImputer` - opcional) estén correctamente importadas al inicio del script.
        *   **Estado Actual:** Las importaciones principales ya están presentes en [`MATABS/MATLAB_cox2.py`](MATABS/MATLAB_cox2.py).
    2.  **Implementar `load_data(file_path)`:** Función para cargar datos desde archivos CSV o Excel, manejando la detección del tipo de archivo, encabezados, celdas vacías (NaN) y filtrado de filas completamente nulas.
        *   **Estado Actual:** Esta función ya está implementada en [`MATABS/MATLAB_cox2.py`](MATABS/MATLAB_cox2.py).
    3.  **Implementar `apply_advanced_filters(df, filters_config)`:** Función para filtrar el DataFrame basado en reglas definidas por el usuario para columnas específicas (valores únicos, múltiples o rangos numéricos).
        *   **Estado Actual:** Esta función ya está implementada en [`MATABS/MATLAB_cox2.py`](MATABS/MATLAB_cox2.py) y utiliza la lógica de filtrado según la especificación.

#### **Fase 2: Preprocesamiento de Datos**

*   **Objetivo:** Preparar el DataFrame para el modelado de supervivencia, incluyendo la definición de variables de tiempo y evento, y el manejo de covariables.
*   **Pasos:**
    1.  **Implementar `preprocess_data(...)`:** Esta función central manejará:
        *   **Definición y Renombrado de Variables de Supervivencia:** Identificar y preparar las columnas de tiempo y evento, incluyendo el renombrado opcional y la conversión a tipos numéricos (evento a 0/1).
        *   **Preprocesamiento y Manejo de Covariables:** Iterar a través de las covariables seleccionadas y sus configuraciones:
            *   **Variables Cualitativas:** Realizar la creación de variables dummy, preferiblemente usando la interfaz de fórmulas de `patsy` para manejar categorías de referencia.
            *   **Variables Cuantitativas con Splines:** Generar términos de spline utilizando `patsy` (B-splines o splines naturales).
            *   **Variables Cuantitativas (sin Splines):** Asegurar que sean numéricas y se incluyan tal cual.
            *   **Covariables Dependientes del Tiempo (TDCs):** Notificar su presencia, aunque el manejo explícito de TDCs ocurre en la etapa de ajuste del modelo.
        *   **Manejo de Datos Faltantes:** Aplicar la estrategia de manejo de datos faltantes (`ListwiseDeletion`, `MeanImputation`, `MedianImputation`) para las columnas esenciales (tiempo, evento, pesos y covariables).
        *   **Estado Actual:** Esta función ya está implementada en [`MATABS/MATLAB_cox2.py`](MATABS/MATLAB_cox2.py), cubriendo todos los puntos de la especificación.

#### **Fase 3: Ajuste del Modelo**

*   **Objetivo:** Instanciar y ajustar los modelos de supervivencia (CoxPH o paramétricos) a los datos preprocesados.
*   **Pasos:**
    1.  **Implementar `fit_survival_model(...)`:**
        *   **Selección del Tipo de Modelo:** Basado en `model_config_params['parametricModelType']` (CoxPH, Weibull, Exponential, LogNormal, LogLogistic, Gompertz).
        *   **Instanciación del Fitter:** Crear la instancia del fitter (`CoxPHFitter`, `WeibullAFTFitter`, etc.) con los parámetros de penalización, `l1_ratio` y método de manejo de empates.
        *   **Construcción de la Fórmula:** Generar la cadena de fórmula para `lifelines` a partir de los términos de covariables procesados.
        *   **Ajuste del Modelo:** Llamar a `fitter.fit()` con el DataFrame procesado, columnas de duración/evento, fórmula, estratos y pesos.
        *   **Estado Actual:** Esta función ya está implementada en [`MATABS/MATLAB_cox2.py`](MATABS/MATLAB_cox2.py).

#### **Fase 4: Extracción de Métricas y Verificación de Supuestos**

*   **Objetivo:** Obtener y almacenar las métricas clave de rendimiento del modelo y realizar verificaciones de supuestos.
*   **Pasos:**
    1.  **Implementar `extract_model_metrics(...)`:**
        *   **Tabla Resumen del Modelo:** Extraer `fitted_model.summary` (coeficientes, HRs/TRs, SEs, CIs, p-valores) y formatear los p-valores.
        *   **Índice de Concordancia (C-Index):** Obtener `fitted_model.concordance_index_`.
        *   **Log-Likelihood y AIC:** Acceder a `fitted_model.log_likelihood_` y `fitted_model.AIC_`.
        *   **Significancia Global del Modelo:** Obtener el p-valor de la prueba de razón de verosimilitud (LRT) o similar.
        *   **Conteo de Observaciones:** Calcular el total de observaciones, número de eventos y censuras.
        *   **Estado Actual:** Esta función ya está implementada en [`MATABS/MATLAB_cox2.py`](MATABS/MATLAB_cox2.py).
    2.  **Implementar `check_ph_assumption(...)`:**
        *   **Verificación de Supuesto PH (solo para CoxPHFitter):** Utilizar `lifelines.statistics.proportional_hazard_test` para obtener los p-valores individuales de los residuos de Schoenfeld y, si es posible, un p-valor global.
        *   **Estado Actual:** Esta función ya está implementada en [`MATABS/MATLAB_cox2.py`](MATABS/MATLAB_cox2.py).
    3.  **Implementar `perform_cindex_cross_validation(...)`:**
        *   **Validación Cruzada para C-Index:** Utilizar `lifelines.utils.k_fold_cross_validation` para estimar el C-Index fuera de la muestra.
        *   **Estado Actual:** Esta función ya está implementada en [`MATABS/MATLAB_cox2.py`](MATABS/MATLAB_cox2.py).

#### **Fase 5: Generación de Datos para Gráficos**

*   **Objetivo:** Generar los datos estructurados necesarios para la visualización de varios gráficos de diagnóstico y resultados.
*   **Pasos:**
    1.  **Implementar `generate_plot_data(...)`:**
        *   **Supervivencia/Riesgo Base:** Extraer `baseline_survival_` y `baseline_hazard_` del modelo ajustado.
        *   **Residuos de Schoenfeld:** Calcular `fitted_model.compute_residuals(kind='schoenfeld')` y estructurar los datos.
        *   **Gráfico de Bosque (Forest Plot):** Extraer coeficientes, HRs y CIs de `fitted_model.summary`.
        *   **Gráfico Log-Menos-Log:** Generar datos para la verificación visual del supuesto PH, utilizando `KaplanMeierFitter` por grupos si hay estratificación.
        *   **Residuos de Martingala/DFBETA:** Calcular `fitted_model.compute_residuals(kind='martingale')` y `(kind='dfbeta')` y estructurar los datos.
        *   **Gráfico de Calibración:** (Nota: La especificación indica que la generación de datos es más compleja y se omite la implementación detallada aquí, ya que `lifelines` proporciona funciones de trazado directo).
        *   **Estado Actual:** Esta función ya está implementada en [`MATABS/MATLAB_cox2.py`](MATABS/MATLAB_cox2.py), generando los datos para los gráficos principales.

#### **Fase 6: Integración y Ejecución**

*   **Objetivo:** Orquestar todas las fases del análisis y proporcionar un punto de entrada ejecutable.
*   **Pasos:**
    1.  **Función `run_survival_analysis(...)`:** Actuará como la función principal que llama a todas las funciones de las fases anteriores en el orden correcto, manejando el flujo de datos y los posibles errores.
        *   **Estado Actual:** Esta función ya está implementada en [`MATABS/MATLAB_cox2.py`](MATABS/MATLAB_cox2.py).
    2.  **Bloque de Ejecución Principal (`if __name__ == "__main__":`)**:
        *   Definir variables de configuración globales de ejemplo (columnas, configuraciones de covariables, estrategia de datos faltantes, parámetros del modelo).
        *   Llamar a `run_survival_analysis` con los parámetros de ejemplo.
        *   Imprimir un resumen de los resultados clave.
        *   **Visualización de Gráficos de Ejemplo:** Incluir código para mostrar los gráficos generados (Supervivencia Base, Riesgo Base, Forest Plot, Log-Minus-Log) utilizando `matplotlib.pyplot.show()`.
        *   **Estado Actual:** Este bloque ya está implementado en [`MATABS/MATLAB_cox2.py`](MATABS/MATLAB_cox2.py), incluyendo la visualización de gráficos de ejemplo.
    3.  **Archivo de Datos de Ejemplo:** Se recomienda crear un archivo `survival_data.csv` en el directorio `d:/APPS` con los datos de ejemplo proporcionados en la especificación para facilitar la ejecución y prueba del script.
        *   **Estado Actual:** Se ha sugerido en los comentarios del código.

---

### Diagrama de Flujo del Proceso de Análisis de Supervivencia

```mermaid
graph TD
    A[Inicio] --> B(Cargar Datos: load_data);
    B --> C{Aplicar Filtros Avanzados?};
    C -- Sí --> D(Filtrar Datos: apply_advanced_filters);
    C -- No --> E(Preprocesar Datos: preprocess_data);
    D --> E;
    E --> F(Ajustar Modelo de Supervivencia: fit_survival_model);
    F --> G(Extraer Métricas: extract_model_metrics);
    G --> H{Modelo CoxPH?};
    H -- Sí --> I(Verificar Supuesto PH: check_ph_assumption);
    H -- No --> J(Realizar Validación Cruzada C-Index);
    I --> J;
    J --> K(Generar Datos para Gráficos: generate_plot_data);
    K --> L(Ensamblar Resultados y Resumen);
    L --> M[Fin];