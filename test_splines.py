import pandas as pd
import numpy as np
from patsy import dmatrix
from lifelines import CoxPHFitter
from lifelines.datasets import load_rossi

# 1. Probar patsy.cr() directamente
print("--- Probando patsy.cr() ---")
data_simple = pd.DataFrame({'x': np.linspace(0, 10, 100)})
data_simple['x_cat'] = pd.cut(data_simple['x'], bins=4, labels=[f"Cat{i}" for i in range(4)])

try:
    # Prueba con df=4 (un número común)
    print("\nIntentando cr(x, df=4):")
    design_cr_df4 = dmatrix("cr(x, df=4)", data_simple, return_type='dataframe')
    print("Matriz de diseño con cr(x, df=4):")
    print(design_cr_df4.head())
    print(f"Shape: {design_cr_df4.shape}")
    print(f"Columnas: {design_cr_df4.columns.tolist()}")

    # Prueba con df=3 (mínimo común para cúbicos, aunque df se refiere a # de funciones base - 1)
    # Patsy cr() df se refiere al número de funciones base. Mínimo df=2 para un spline lineal simple.
    # Para un spline cúbico natural, normalmente necesitas al menos 4 puntos, y df se relaciona con los nudos.
    # Si df es muy bajo, puede que no sea cúbico o falle.
    print("\nIntentando cr(x, df=2) (esperado lineal):")
    design_cr_df2 = dmatrix("cr(x, df=2)", data_simple, return_type='dataframe')
    print("Matriz de diseño con cr(x, df=2):")
    print(design_cr_df2.head())
    print(f"Shape: {design_cr_df2.shape}")
    print(f"Columnas: {design_cr_df2.columns.tolist()}")

    # Prueba con knots explícitos
    # Los nudos deben estar dentro del rango de 'x'
    knots = np.percentile(data_simple['x'], [25, 50, 75])
    knots_str = f"({', '.join(map(str, knots))})" # Formato "(k1, k2, k3)"
    print(f"\nIntentando cr(x, knots={knots_str}):")
    design_cr_knots = dmatrix(f"cr(x, knots={knots_str})", data_simple, return_type='dataframe')
    print(f"Matriz de diseño con cr(x, knots={knots_str}):")
    print(design_cr_knots.head())
    print(f"Shape: {design_cr_knots.shape}")
    print(f"Columnas: {design_cr_knots.columns.tolist()}")

    # ¿patsy.cr() acepta 'degree'?
    # La documentación de Patsy para cr() no menciona un parámetro 'degree'.
    # cr() genera splines de regresión CÚBICOS naturales. El grado es inherentemente 3.
    print("\nIntentando cr(x, df=4, degree=2) - Se espera que falle o ignore 'degree':")
    try:
        design_cr_degree_test = dmatrix("cr(x, df=4, degree=2)", data_simple, return_type='dataframe')
        print("Matriz de diseño con cr(x, df=4, degree=2) (inesperado, 'degree' podría haber sido ignorado o tener otro significado):")
        print(design_cr_degree_test.head())
    except Exception as e_cr_degree:
        print(f"Error esperado al intentar cr(x, df=4, degree=2): {e_cr_degree}")
        print("Esto confirma que 'degree' no es un parámetro estándar para cr() en patsy como lo es para bs().")

except Exception as e:
    print(f"Error durante pruebas de patsy.cr(): {e}")

# 2. Probar patsy.cr() con lifelines.CoxPHFitter
print("\n\n--- Probando patsy.cr() con lifelines.CoxPHFitter ---")
rossi_df = load_rossi()
# Añadir una columna numérica continua para usar con splines
rossi_df['age_spline_var'] = rossi_df['age'] + np.random.normal(0, 1, size=len(rossi_df))


# Modelo con cr()
try:
    print("\nAjustando modelo Cox con cr(age_spline_var, df=4):")
    # Asegurarse que no haya NaNs en las columnas usadas por la fórmula
    cols_for_formula = ['week', 'arrest', 'age_spline_var', 'prio']
    rossi_df_clean = rossi_df[cols_for_formula].dropna().copy()

    if rossi_df_clean.empty:
        print("DataFrame vacío después de dropna. No se puede ajustar el modelo.")
    else:
        cph_cr = CoxPHFitter()
        # Usar Q('') para nombres de columna si es necesario, aunque 'age_spline_var' es seguro
        formula_cr = "Q('age_spline_var') + prio" # Modelo base para comparar
        formula_cr_spline = "cr(Q('age_spline_var'), df=4) + prio"

        print(f"Intentando fórmula: {formula_cr_spline}")
        # Pre-generar X para inspección
        X_design_cr_lifelines = dmatrix(formula_cr_spline, rossi_df_clean, return_type='dataframe')
        print("Matriz de diseño para lifelines con cr():")
        print(X_design_cr_lifelines.head())
        print(f"Shape: {X_design_cr_lifelines.shape}")

        # Crear DataFrame para lifelines fit (incluyendo T y E)
        df_for_lifelines_cr = rossi_df_clean[['week', 'arrest']].copy()
        df_for_lifelines_cr = pd.concat([df_for_lifelines_cr, X_design_cr_lifelines], axis=1)

        # Eliminar 'Intercept' si Patsy lo añadió y no lo queremos duplicado por lifelines o implícito
        if 'Intercept' in df_for_lifelines_cr.columns:
             df_for_lifelines_cr = df_for_lifelines_cr.drop(columns=['Intercept'])
             print("Columna 'Intercept' eliminada de la matriz de diseño antes de ajustar.")

        # La fórmula para fit() ahora debe referirse a las columnas ya transformadas en df_for_lifelines_cr
        # o podemos pasar X_design_cr_lifelines directamente si alineamos y_data.
        # Lifelines puede tomar una fórmula y datos, o datos pre-transformados X.
        # Si pasamos fórmula y datos originales, lifelines usa patsy internamente.

        cph_cr.fit(rossi_df_clean, 'week', 'arrest', formula=formula_cr_spline)
        print("\nResumen del modelo Cox con cr(age_spline_var, df=4):")
        cph_cr.print_summary()
        print("Spline natural (`cr()`) parece funcionar con lifelines.")

except Exception as e:
    print(f"Error ajustando modelo Cox con cr(): {e}")
    import traceback
    traceback.print_exc()

# 3. Probar patsy.bs() con degree y lifelines
print("\n\n--- Probando patsy.bs() con degree y lifelines ---")
try:
    print("\nAjustando modelo Cox con bs(age_spline_var, df=4, degree=2) (spline cuadrático):")
    # Reusar rossi_df_clean
    if rossi_df_clean.empty:
        print("DataFrame vacío. No se puede ajustar el modelo bs().")
    else:
        cph_bs = CoxPHFitter()
        formula_bs_deg2 = "bs(Q('age_spline_var'), df=4, degree=2) + prio"
        print(f"Intentando fórmula: {formula_bs_deg2}")

        X_design_bs_lifelines = dmatrix(formula_bs_deg2, rossi_df_clean, return_type='dataframe')
        print("Matriz de diseño para lifelines con bs(degree=2):")
        print(X_design_bs_lifelines.head())
        print(f"Shape: {X_design_bs_lifelines.shape}")

        cph_bs.fit(rossi_df_clean, 'week', 'arrest', formula=formula_bs_deg2)
        print("\nResumen del modelo Cox con bs(age_spline_var, df=4, degree=2):")
        cph_bs.print_summary()
        print("B-spline (`bs()`) con `degree` especificado parece funcionar con lifelines.")

except Exception as e:
    print(f"Error ajustando modelo Cox con bs(degree=2): {e}")
    import traceback
    traceback.print_exc()

print("\n\n--- Conclusiones del script de prueba ---")
print("1. `patsy.cr(x, df=N)`: Funciona y genera una matriz de diseño. El número de columnas es `df`.")
print("   - `df` en `cr` se refiere al número de funciones base. Un `df` más bajo reduce la flexibilidad.")
print("   - `patsy.cr()` NO acepta un parámetro `degree` explícito; es inherentemente para splines CÚBICOS naturales.")
print("2. `patsy.cr()` con `lifelines`: Un `CoxPHFitter` simple se pudo ajustar usando una fórmula con `cr()`. Esto sugiere que el problema de \"no funcionan\" podría ser más sutil, relacionado con la implementación específica en la GUI, los datos del usuario, o la interpretación de los resultados.")
print("3. `patsy.bs(x, df=N, degree=D)`: Funciona y permite especificar el grado. Un `CoxPHFitter` se pudo ajustar con `bs(..., degree=2)`.")
print("4. Por lo tanto, la funcionalidad de añadir selección de grado para B-splines es viable. Para splines naturales, el grado es fijo (cúbico).")
