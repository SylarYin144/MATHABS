#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
try:
    from statsmodels.stats.power import TTestIndPower, NormalIndPower
    from statsmodels.stats.proportion import effectsize_proportions, samplesize_confint_proportion
    from scipy.stats import norm # Already should be here for precision calculations
except ImportError as e:
    # It's crucial that statsmodels is installed. If not, many functions will fail.
    messagebox.showerror("Error de Importación Crítico",
                         f"No se pudieron importar componentes de 'statsmodels' o 'scipy': {e}. "
                         "Estas librerías son esenciales. Por favor, asegúrese de que estén instaladas.")
    # Optionally, disable the calculate button or the entire tab here.
    # For now, we'll let it proceed, but calculations will likely fail if imports are missing.
    TTestIndPower = None
    NormalIndPower = None
    effectsize_proportions = None
    samplesize_confint_proportion = None # Added this
    norm = None # Though norm is usually available with scipy, which is a core dep.

class SampleSizeCalculatorTab(ttk.Frame):
    def __init__(self, notebook, main_app_instance=None):
        super().__init__(notebook)
        self.main_app = main_app_instance

        main_frame = ttk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # --- Study Design Selection ---
        study_design_frame = ttk.LabelFrame(main_frame, text="Diseño del Estudio")
        study_design_frame.pack(fill=tk.X, padx=5, pady=5)
        self.study_design_var = tk.StringVar()
        study_designs = [
            "Estudios descriptivos (encuestas, prevalencia)", # Index 0
            "Ensayos clínicos (comparación entre dos o más grupos)", # Index 1
            "Estudios correlacionales",
            "Estudios explicativos (experimentales, cuasiexperimentales)",
            "Estudios de laboratorio (comparación de métodos, variación lote a lote)",
            "Estudios diagnósticos y pronósticos (análisis ROC)"
        ]
        ttk.Label(study_design_frame, text="Tipo de Diseño:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.study_design_combo = ttk.Combobox(study_design_frame, textvariable=self.study_design_var, values=study_designs, width=60, state="readonly")
        self.study_design_combo.grid(row=0, column=1, padx=5, pady=5, sticky=tk.EW)
        self.study_design_combo.current(1) # Default to clinical trials
        study_design_frame.grid_columnconfigure(1, weight=1)
        self.study_design_combo.bind("<<ComboboxSelected>>", self.on_study_design_change)


        # --- Variable Type Selection ---
        variable_type_frame = ttk.LabelFrame(main_frame, text="Tipo de Variable Principal")
        variable_type_frame.pack(fill=tk.X, padx=5, pady=5, anchor="n")
        self.variable_type_var = tk.StringVar()
        variable_types = [
            "Datos cuantitativos (medias, desviaciones estándar)",
            "Datos categóricos (proporciones)",
            "Variables binarias (Odds Ratio, Riesgo Relativo)"
        ]
        ttk.Label(variable_type_frame, text="Tipo de Variable:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.variable_type_combo = ttk.Combobox(variable_type_frame, textvariable=self.variable_type_var, values=variable_types, width=60, state="readonly")
        self.variable_type_combo.grid(row=0, column=1, padx=5, pady=5, sticky=tk.EW)
        self.variable_type_combo.current(1) # Default to proportions
        variable_type_frame.grid_columnconfigure(1, weight=1)
        self.variable_type_combo.bind("<<ComboboxSelected>>", self.update_effect_size_options)

        # --- Parameters Frame (dynamic content based on study type) ---
        self.parameters_frame = ttk.Frame(main_frame)
        self.parameters_frame.pack(fill=tk.X, padx=0, pady=0) # No internal padding for this frame itself

        # --- Power Analysis Parameters (becomes part of dynamic parameters_frame) ---
        self.power_params_frame = ttk.LabelFrame(self.parameters_frame, text="Parámetros de Potencia y Efecto")
        # Packed later by on_study_design_change

        ttk.Label(self.power_params_frame, text="Potencia deseada (1-β):").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.power_var = tk.StringVar(value="0.80")
        self.power_entry = ttk.Entry(self.power_params_frame, textvariable=self.power_var, width=12)
        self.power_entry.grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)

        ttk.Label(self.power_params_frame, text="Nivel de significancia (α):").grid(row=0, column=2, padx=5, pady=5, sticky=tk.W)
        self.alpha_var = tk.StringVar(value="0.05")
        self.alpha_entry = ttk.Entry(self.power_params_frame, textvariable=self.alpha_var, width=12)
        self.alpha_entry.grid(row=0, column=3, padx=5, pady=5, sticky=tk.W)

        self.effect_size_input_frame = ttk.Frame(self.power_params_frame)
        self.effect_size_input_frame.grid(row=1, column=0, columnspan=4, sticky=tk.EW, pady=5)

        ttk.Label(self.effect_size_input_frame, text="Tipo de Tamaño del Efecto:").grid(row=0, column=0, padx=5, pady=2, sticky=tk.W)
        self.effect_size_type_var = tk.StringVar()
        self.effect_size_type_combo = ttk.Combobox(self.effect_size_input_frame, textvariable=self.effect_size_type_var, width=25, state="readonly")
        self.effect_size_type_combo.grid(row=0, column=1, padx=5, pady=2, sticky=tk.W)
        self.effect_size_type_combo.bind("<<ComboboxSelected>>", self.update_specific_effect_inputs)

        self.specific_effect_inputs_frame = ttk.Frame(self.effect_size_input_frame)
        self.specific_effect_inputs_frame.grid(row=0, column=2, padx=5, pady=0, sticky=tk.W)

        # --- Precision Parameters (becomes part of dynamic parameters_frame) ---
        self.precision_params_frame = ttk.LabelFrame(self.parameters_frame, text="Parámetros de Precisión (Estudios Descriptivos)")
        # Packed later by on_study_design_change

        ttk.Label(self.precision_params_frame, text="Margen de Error Deseado (±):").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.margin_error_var = tk.StringVar(value="0.05") # Example: 5% for proportion, or units for mean
        self.margin_error_entry = ttk.Entry(self.precision_params_frame, textvariable=self.margin_error_var, width=12)
        self.margin_error_entry.grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)

        ttk.Label(self.precision_params_frame, text="Nivel de Confianza (1-α):").grid(row=0, column=2, padx=5, pady=5, sticky=tk.W)
        self.confidence_level_var = tk.StringVar(value="0.95") # Example: 95%
        self.confidence_level_entry = ttk.Entry(self.precision_params_frame, textvariable=self.confidence_level_var, width=12)
        self.confidence_level_entry.grid(row=0, column=3, padx=5, pady=5, sticky=tk.W)

        # Additional inputs for precision if needed (e.g., estimated proportion/mean, population size)
        ttk.Label(self.precision_params_frame, text="Proporción Estimada (P):").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        self.estimated_p_var = tk.StringVar(value="0.5") # For proportion precision
        self.estimated_p_entry = ttk.Entry(self.precision_params_frame, textvariable=self.estimated_p_var, width=12)
        self.estimated_p_entry.grid(row=1, column=1, padx=5, pady=5, sticky=tk.W)

        ttk.Label(self.precision_params_frame, text="DE Estimada (σ):").grid(row=1, column=2, padx=5, pady=5, sticky=tk.W)
        self.estimated_sd_var = tk.StringVar(value="1.0") # For mean precision
        self.estimated_sd_entry = ttk.Entry(self.precision_params_frame, textvariable=self.estimated_sd_var, width=12)
        self.estimated_sd_entry.grid(row=1, column=3, padx=5, pady=5, sticky=tk.W)


        # --- Calculation Button and Results ---
        results_frame = ttk.Frame(main_frame) # Placed back in main_frame
        results_frame.pack(fill=tk.X, padx=5, pady=10, anchor="n")
        self.calculate_button = ttk.Button(results_frame, text="Calcular Tamaño de Muestra", command=self.calculate_sample_size)
        self.calculate_button.pack(pady=5)

        self.result_label_text_var = tk.StringVar(value="Tamaño de Muestra Calculado:")
        self.result_label = ttk.Label(results_frame, textvariable=self.result_label_text_var)
        self.result_label.pack(side=tk.LEFT, padx=5)

        self.sample_size_result_var = tk.StringVar(value="---")
        self.sample_size_result_label = ttk.Label(results_frame, textvariable=self.sample_size_result_var, font=("TkDefaultFont", 10, "bold"))
        self.sample_size_result_label.pack(side=tk.LEFT, padx=5)

        self.on_study_design_change() # Initial call to set visibility
        self.update_effect_size_options()

    def on_study_design_change(self, event=None):
        study_type = self.study_design_var.get()
        if "Estudios descriptivos" in study_type:
            self.power_params_frame.pack_forget()
            self.precision_params_frame.pack(fill=tk.X, padx=5, pady=5, anchor="n")
            self.result_label_text_var.set("Tamaño de Muestra (Precisión):")
        else: # For comparative studies, etc.
            self.precision_params_frame.pack_forget()
            self.power_params_frame.pack(fill=tk.X, padx=5, pady=5, anchor="n")
            self.result_label_text_var.set("Tamaño de Muestra (por grupo):")
        self.update_effect_size_options() # Update effect size options as they might depend on study type implicitly


    def update_effect_size_options(self, event=None):
        for widget in self.specific_effect_inputs_frame.winfo_children():
            widget.destroy()
        self.effect_size_type_var.set('')

        var_type = self.variable_type_var.get()
        study_type = self.study_design_var.get() # Get current study type
        effect_size_options = []

        # Effect size options are typically for comparative studies, not descriptive precision-based ones
        if "Estudios descriptivos" not in study_type:
            if "proporciones" in var_type or "binarias" in var_type:
                effect_size_options = ["Diferencia de Proporciones (P1, P2)", "Odds Ratio (OR)", "Riesgo Relativo (RR)"]
            elif "cuantitativos" in var_type:
                effect_size_options = ["d de Cohen (Diferencia de Medias)", "Diferencia de Medias (Absoluta)"]

        self.effect_size_type_combo['values'] = effect_size_options
        if effect_size_options:
            self.effect_size_type_combo.current(0)
            self.effect_size_type_combo.config(state="readonly")
        else:
            self.effect_size_type_combo.config(state="disabled") # Disable if no options

        self.update_specific_effect_inputs()


    def update_specific_effect_inputs(self, event=None):
        for widget in self.specific_effect_inputs_frame.winfo_children():
            widget.destroy()

        effect_type = self.effect_size_type_var.get()
        # Only populate if effect_type is meaningful (i.e., not a descriptive study focused on precision)
        if not self.effect_size_type_combo.cget('values') or self.effect_size_type_combo.cget('state') == 'disabled':
            return

        if effect_type == "Diferencia de Proporciones (P1, P2)":
            ttk.Label(self.specific_effect_inputs_frame, text="P1:").grid(row=0, column=0, padx=2, pady=2, sticky=tk.W)
            self.p1_var = tk.StringVar(value="0.50")
            self.p1_entry = ttk.Entry(self.specific_effect_inputs_frame, textvariable=self.p1_var, width=8)
            self.p1_entry.grid(row=0, column=1, padx=2, pady=2, sticky=tk.W)

            ttk.Label(self.specific_effect_inputs_frame, text="P2:").grid(row=0, column=2, padx=2, pady=2, sticky=tk.W)
            self.p2_var = tk.StringVar(value="0.60")
            self.p2_entry = ttk.Entry(self.specific_effect_inputs_frame, textvariable=self.p2_var, width=8)
            self.p2_entry.grid(row=0, column=3, padx=2, pady=2, sticky=tk.W)

        elif effect_type == "d de Cohen (Diferencia de Medias)":
            ttk.Label(self.specific_effect_inputs_frame, text="d de Cohen:").grid(row=0, column=0, padx=2, pady=2, sticky=tk.W)
            self.cohen_d_var = tk.StringVar(value="0.5")
            self.cohen_d_entry = ttk.Entry(self.specific_effect_inputs_frame, textvariable=self.cohen_d_var, width=10)
            self.cohen_d_entry.grid(row=0, column=1, padx=2, pady=2, sticky=tk.W)

            ttk.Label(self.specific_effect_inputs_frame, text="DE (opcional):").grid(row=0, column=2, padx=2, pady=2, sticky=tk.W)
            self.sd_var = tk.StringVar(value="1.0")
            self.sd_entry = ttk.Entry(self.specific_effect_inputs_frame, textvariable=self.sd_var, width=10)
            self.sd_entry.grid(row=0, column=3, padx=2, pady=2, sticky=tk.W)

        elif effect_type == "Diferencia de Medias (Absoluta)":
            ttk.Label(self.specific_effect_inputs_frame, text="Media Grp 1:").grid(row=0, column=0, padx=2, pady=2, sticky=tk.W)
            self.mean1_var = tk.StringVar(value="10")
            self.mean1_entry = ttk.Entry(self.specific_effect_inputs_frame, textvariable=self.mean1_var, width=8)
            self.mean1_entry.grid(row=0, column=1, padx=2, pady=2, sticky=tk.W)

            ttk.Label(self.specific_effect_inputs_frame, text="Media Grp 2:").grid(row=0, column=2, padx=2, pady=2, sticky=tk.W)
            self.mean2_var = tk.StringVar(value="12")
            self.mean2_entry = ttk.Entry(self.specific_effect_inputs_frame, textvariable=self.mean2_var, width=8)
            self.mean2_entry.grid(row=0, column=3, padx=2, pady=2, sticky=tk.W)

            ttk.Label(self.specific_effect_inputs_frame, text="DE Común:").grid(row=0, column=4, padx=2, pady=2, sticky=tk.W)
            self.common_sd_var = tk.StringVar(value="3")
            self.common_sd_entry = ttk.Entry(self.specific_effect_inputs_frame, textvariable=self.common_sd_var, width=8)
            self.common_sd_entry.grid(row=0, column=5, padx=2, pady=2, sticky=tk.W)

        elif effect_type == "Odds Ratio (OR)":
            ttk.Label(self.specific_effect_inputs_frame, text="OR:").grid(row=0, column=0, padx=2, pady=2, sticky=tk.W)
            self.or_var = tk.StringVar(value="1.5")
            self.or_entry = ttk.Entry(self.specific_effect_inputs_frame, textvariable=self.or_var, width=10)
            self.or_entry.grid(row=0, column=1, padx=2, pady=2, sticky=tk.W)
            ttk.Label(self.specific_effect_inputs_frame, text="P0 (ref):").grid(row=0, column=2, padx=2, pady=2, sticky=tk.W)
            self.p0_or_var = tk.StringVar(value="0.2")
            self.p0_or_entry = ttk.Entry(self.specific_effect_inputs_frame, textvariable=self.p0_or_var, width=10)
            self.p0_or_entry.grid(row=0, column=3, padx=2, pady=2, sticky=tk.W)

        elif effect_type == "Riesgo Relativo (RR)":
            ttk.Label(self.specific_effect_inputs_frame, text="RR:").grid(row=0, column=0, padx=2, pady=2, sticky=tk.W)
            self.rr_var = tk.StringVar(value="1.2")
            self.rr_entry = ttk.Entry(self.specific_effect_inputs_frame, textvariable=self.rr_var, width=10)
            self.rr_entry.grid(row=0, column=1, padx=2, pady=2, sticky=tk.W)
            ttk.Label(self.specific_effect_inputs_frame, text="P0 (ref):").grid(row=0, column=2, padx=2, pady=2, sticky=tk.W)
            self.p0_rr_var = tk.StringVar(value="0.2")
            self.p0_rr_entry = ttk.Entry(self.specific_effect_inputs_frame, textvariable=self.p0_rr_var, width=10)
            self.p0_rr_entry.grid(row=0, column=3, padx=2, pady=2, sticky=tk.W)

    def calculate_sample_size(self):
        # Check for critical imports for calculation
        # Updated check to include samplesize_confint_proportion
        if TTestIndPower is None or NormalIndPower is None or effectsize_proportions is None or norm is None or samplesize_confint_proportion is None:
            messagebox.showerror("Error de Librería",
                                 "Faltan componentes esenciales de 'statsmodels' o 'scipy' que no se pudieron importar al inicio. "
                                 "El cálculo no puede continuar. Verifique la instalación de estas librerías.")
            return

        study_type = self.study_design_var.get()
        var_type = self.variable_type_var.get()
        calculated_sample_size = None

        try:
            if "Estudios descriptivos" in study_type:
                # Precision-based calculation
                margin_of_error = float(self.margin_error_var.get())
                confidence_level = float(self.confidence_level_var.get())
                alpha_precision = 1 - confidence_level # alpha for precision is 1 - confidence

                if "proporciones" in var_type:
                    estimated_p = float(self.estimated_p_var.get())
                    if not (0 <= estimated_p <= 1):
                        messagebox.showerror("Error de Entrada", "La proporción estimada debe estar entre 0 y 1.")
                        return
                    if not (0 < margin_of_error < 1):
                        messagebox.showerror("Error de Entrada", "El margen de error para proporciones debe estar entre 0 y 1.")
                        return

                    calculated_sample_size = samplesize_confint_proportion(
                        proportion=estimated_p,
                        half_length=margin_of_error,
                        alpha=alpha_precision,
                        method='normal'
                    )

                elif "cuantitativos" in var_type:
                    estimated_sd = float(self.estimated_sd_var.get())
                    if estimated_sd <= 0:
                        messagebox.showerror("Error de Entrada", "La DE estimada debe ser positiva.")
                        return
                    if margin_of_error <= 0:
                        messagebox.showerror("Error de Entrada", "El margen de error para medias debe ser positivo.")
                        return
                    # Formula: n = (Z_alpha/2 * sigma / E)^2
                    # from scipy.stats import norm # This line is removed, norm is imported globally
                    z_score = norm.ppf(1 - (alpha_precision / 2))
                    calculated_sample_size = (z_score * estimated_sd / margin_of_error)**2
                else:
                    messagebox.showinfo("Información", "Cálculo de precisión para este tipo de variable no implementado.")
                    self.sample_size_result_var.set("---")
                    return

            else: # Power-based calculation for comparative studies
                power = float(self.power_var.get())
                alpha = float(self.alpha_var.get())
                effect_type_selected = self.effect_size_type_var.get()

                if "proporciones" in var_type or "binarias" in var_type:
                    if effect_type_selected == "Diferencia de Proporciones (P1, P2)":
                        p1 = float(self.p1_var.get())
                        p2 = float(self.p2_var.get())
                        if not (0 <= p1 <= 1 and 0 <= p2 <= 1):
                            messagebox.showerror("Error de Entrada", "Las proporciones P1 y P2 deben estar entre 0 y 1.")
                            return
                        if p1 == p2:
                             messagebox.showerror("Error de Entrada", "P1 y P2 no pueden ser iguales para este cálculo.")
                             return

                        es = effectsize_proportions(p1, p2, method='normal')
                        power_analysis = NormalIndPower()
                        calculated_sample_size = power_analysis.solve_power(
                            effect_size=es,
                            alpha=alpha,
                            power=power,
                            ratio=1.0,
                            alternative='two-sided',
                            nobs=None
                        )
                    elif effect_type_selected in ["Odds Ratio (OR)", "Riesgo Relativo (RR)"]:
                         messagebox.showinfo("Información", f"Cálculo para {effect_type_selected} aún no implementado con 'statsmodels' en esta interfaz.")
                         self.sample_size_result_var.set("---")
                         return
                    else:
                        messagebox.showinfo("Información", "Seleccione un tipo de tamaño del efecto válido para variables categóricas/binarias.")
                        self.sample_size_result_var.set("---")
                        return

                elif "cuantitativos" in var_type:
                    if effect_type_selected == "d de Cohen (Diferencia de Medias)":
                        cohen_d = float(self.cohen_d_var.get())
                        power_analysis = TTestIndPower()
                        calculated_sample_size = power_analysis.solve_power(
                            effect_size=cohen_d,
                            alpha=alpha,
                            power=power,
                            ratio=1.0,
                            alternative='two-sided',
                            nobs=None
                        )
                    elif effect_type_selected == "Diferencia de Medias (Absoluta)":
                        mean1 = float(self.mean1_var.get())
                        mean2 = float(self.mean2_var.get())
                        common_sd = float(self.common_sd_var.get())
                        if common_sd <= 0:
                             messagebox.showerror("Error de Entrada", "La DE Común debe ser positiva.")
                             return

                        cohen_d_calculated = abs(mean1 - mean2) / common_sd
                        power_analysis = TTestIndPower()
                        calculated_sample_size = power_analysis.solve_power(
                            effect_size=cohen_d_calculated,
                            alpha=alpha,
                            power=power,
                            ratio=1.0,
                            alternative='two-sided',
                            nobs=None
                        )
                    else:
                        messagebox.showinfo("Información", "Seleccione un tipo de tamaño del efecto válido para variables cuantitativas.")
                        self.sample_size_result_var.set("---")
                        return
                else:
                    messagebox.showinfo("Información", "Tipo de variable no soportado para cálculo de potencia con los parámetros actuales.")
                    self.sample_size_result_var.set("---")
                    return

            if calculated_sample_size is not None:
                self.sample_size_result_var.set(f"{calculated_sample_size:.0f}")
            else:
                self.sample_size_result_var.set("Error calc.")

        except ValueError:
            messagebox.showerror("Error de Entrada", "Por favor, ingrese valores numéricos válidos.")
            self.sample_size_result_var.set("Error")
        except ImportError: # Specifically for scipy.stats if it somehow wasn't there
            messagebox.showerror("Error de Importación", "Se requiere 'scipy' para este cálculo. Asegúrese de que esté instalado.")
            self.sample_size_result_var.set("Error")
        except Exception as e:
            messagebox.showerror("Error de Cálculo", f"Ocurrió un error: {e}")
            self.sample_size_result_var.set("Error")


if __name__ == '__main__':
    root = tk.Tk()
    root.title("Test Sample Size Calculator Tab")
    notebook = ttk.Notebook(root)
    app_tab = SampleSizeCalculatorTab(notebook, main_app_instance=None)
    notebook.add(app_tab, text="Sample Size Calculator")
    notebook.pack(expand=True, fill='both', padx=10, pady=10)
    root.geometry("750x650") # Adjusted size
    root.mainloop()
