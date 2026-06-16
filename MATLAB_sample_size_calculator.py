#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import math
try:
    from statsmodels.stats.power import TTestIndPower, NormalIndPower, FTestAnovaPower, TTestPower
    from statsmodels.stats.proportion import proportion_effectsize, samplesize_confint_proportion
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
    FTestAnovaPower = None
    TTestPower = None
    proportion_effectsize = None # Changed from effectsize_proportions
    samplesize_confint_proportion = None
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

        ttk.Label(self.power_params_frame, text="Hipótesis:").grid(row=0, column=4, padx=5, pady=5, sticky=tk.W)
        self.hypothesis_tail_var = tk.StringVar(value="Bilateral (dos colas)")
        self.hypothesis_tail_combo = ttk.Combobox(
            self.power_params_frame,
            textvariable=self.hypothesis_tail_var,
            values=["Bilateral (dos colas)", "Unilateral (una cola)"],
            width=18,
            state="readonly"
        )
        self.hypothesis_tail_combo.grid(row=0, column=5, padx=5, pady=5, sticky=tk.W)

        self.effect_size_input_frame = ttk.Frame(self.power_params_frame)
        self.effect_size_input_frame.grid(row=1, column=0, columnspan=4, sticky=tk.EW, pady=5)

        ttk.Label(self.effect_size_input_frame, text="Tipo de Tamaño del Efecto:").grid(row=0, column=0, padx=5, pady=2, sticky=tk.W)
        self.effect_size_type_var = tk.StringVar()
        self.effect_size_type_combo = ttk.Combobox(self.effect_size_input_frame, textvariable=self.effect_size_type_var, width=25, state="readonly")
        self.effect_size_type_combo.grid(row=0, column=1, padx=5, pady=2, sticky=tk.W)
        self.effect_size_type_combo.bind("<<ComboboxSelected>>", self.update_specific_effect_inputs)

        self.anova_effect_size_var = tk.StringVar(value="0.25")
        self.anova_effect_size_choice_var = tk.StringVar(value="Mediano (0.25)")
        self.anova_groups_var = tk.StringVar(value="2")
        self.response_rate_var = tk.StringVar(value="0.85")
        self.eligibility_rate_var = tk.StringVar(value="0.95")
        self.attrition_rate_var = tk.StringVar(value="0.15")
        self.paired_delta_var = tk.StringVar(value="2.0")
        self.paired_sd_diff_var = tk.StringVar(value="3.0")
        self.change_control_var = tk.StringVar(value="0.0")
        self.change_treatment_var = tk.StringVar(value="2.0")
        self.change_sd_var = tk.StringVar(value="3.0")
        self.equivalence_margin_var = tk.StringVar(value="2.0")
        self.equivalence_expected_diff_var = tk.StringVar(value="0.0")
        self.equivalence_sd_var = tk.StringVar(value="3.0")

        self.specific_effect_inputs_frame = ttk.Frame(self.effect_size_input_frame)
        self.specific_effect_inputs_frame.grid(row=0, column=2, padx=5, pady=0, sticky=tk.W)

        ttk.Label(self.power_params_frame, text="Pérdidas esperadas (R):").grid(row=2, column=0, padx=5, pady=5, sticky=tk.W)
        self.attrition_rate_entry = ttk.Entry(self.power_params_frame, textvariable=self.attrition_rate_var, width=12)
        self.attrition_rate_entry.grid(row=2, column=1, padx=5, pady=5, sticky=tk.W)
        ttk.Label(self.power_params_frame, text="(0 a 1, ej. 0.15)").grid(row=2, column=2, padx=5, pady=5, sticky=tk.W)

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
        self.estimated_p_label = ttk.Label(self.precision_params_frame, text="Proporción Estimada (P):")
        self.estimated_p_label.grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        self.estimated_p_var = tk.StringVar(value="0.5") # For proportion precision
        self.estimated_p_entry = ttk.Entry(self.precision_params_frame, textvariable=self.estimated_p_var, width=12)
        self.estimated_p_entry.grid(row=1, column=1, padx=5, pady=5, sticky=tk.W)

        self.estimated_sd_label = ttk.Label(self.precision_params_frame, text="DE Estimada (σ):")
        self.estimated_sd_label.grid(row=1, column=2, padx=5, pady=5, sticky=tk.W)
        self.estimated_sd_var = tk.StringVar(value="1.0") # For mean precision
        self.estimated_sd_entry = ttk.Entry(self.precision_params_frame, textvariable=self.estimated_sd_var, width=12)
        self.estimated_sd_entry.grid(row=1, column=3, padx=5, pady=5, sticky=tk.W)

        ttk.Label(self.precision_params_frame, text="Tamaño población accesible (N):").grid(row=2, column=0, padx=5, pady=5, sticky=tk.W)
        self.population_size_var = tk.StringVar(value="")
        self.population_size_entry = ttk.Entry(self.precision_params_frame, textvariable=self.population_size_var, width=12)
        self.population_size_entry.grid(row=2, column=1, padx=5, pady=5, sticky=tk.W)
        self.force_fpc_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            self.precision_params_frame,
            text="Forzar corrección FPC (aunque n0/N <= 5%)",
            variable=self.force_fpc_var
        ).grid(row=2, column=2, columnspan=2, padx=5, pady=5, sticky=tk.W)

        prevalence_helper_frame = ttk.LabelFrame(self.precision_params_frame, text="Conversión de Prevalencia a N", padding=5)
        prevalence_helper_frame.grid(row=3, column=0, columnspan=4, padx=5, pady=5, sticky=tk.EW)
        prevalence_helper_frame.columnconfigure(5, weight=1)

        ttk.Label(prevalence_helper_frame, text="Casos (numerador):").grid(row=0, column=0, padx=2, pady=2, sticky=tk.W)
        self.prevalence_cases_var = tk.StringVar(value="4")
        ttk.Entry(prevalence_helper_frame, textvariable=self.prevalence_cases_var, width=8).grid(row=0, column=1, padx=2, pady=2, sticky=tk.W)

        ttk.Label(prevalence_helper_frame, text="por cada:").grid(row=0, column=2, padx=2, pady=2, sticky=tk.W)
        self.prevalence_denominator_var = tk.StringVar(value="100000")
        ttk.Entry(prevalence_helper_frame, textvariable=self.prevalence_denominator_var, width=10).grid(row=0, column=3, padx=2, pady=2, sticky=tk.W)

        ttk.Label(prevalence_helper_frame, text="Población región:").grid(row=0, column=4, padx=2, pady=2, sticky=tk.W)
        self.region_population_var = tk.StringVar(value="")
        ttk.Entry(prevalence_helper_frame, textvariable=self.region_population_var, width=12).grid(row=0, column=5, padx=2, pady=2, sticky=tk.W)

        ttk.Button(
            prevalence_helper_frame,
            text="Calcular N",
            command=self._calculate_population_from_prevalence
        ).grid(row=0, column=6, padx=5, pady=2, sticky=tk.W)


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

    def _refresh_precision_inputs_visibility(self):
        """Show DE field only when the descriptive calc needs σ (variables cuantitativas)."""
        var_type = (self.variable_type_var.get() or "").lower()
        needs_sd = "cuantitativos" in var_type
        needs_p = "proporciones" in var_type or "categóricos" in var_type or "binarias" in var_type
        sd_widgets = [self.estimated_sd_label, self.estimated_sd_entry]
        p_widgets = [self.estimated_p_label, self.estimated_p_entry]
        for widget in sd_widgets:
            if needs_sd:
                widget.grid()
            else:
                widget.grid_remove()
        for widget in p_widgets:
            if needs_p:
                widget.grid()
            else:
                widget.grid_remove()

    def _calculate_population_from_prevalence(self):
        """Estimate N from prevalence inputs (cases per denominator) and region population."""
        try:
            cases = float(self.prevalence_cases_var.get())
            denominator = float(self.prevalence_denominator_var.get())
            region_population = float(self.region_population_var.get())
        except ValueError:
            messagebox.showerror("Error de Entrada", "Ingrese valores numéricos en los campos de prevalencia y población.")
            return

        if denominator <= 0 or region_population <= 0 or cases < 0:
            messagebox.showerror("Error de Entrada", "Asegúrese de que casos ≥ 0 y que el denominador y la población regional sean mayores que 0.")
            return

        prevalence_rate = cases / denominator
        estimated_population = region_population * prevalence_rate
        if estimated_population <= 0:
            messagebox.showerror("Resultado Inválido", "La combinación ingresada produce un N accesible no positivo.")
            return

        self.population_size_var.set(f"{estimated_population:.2f}")
        messagebox.showinfo(
            "Prevalencia convertida",
            f"Se estimó N ≈ {estimated_population:.2f}. Este valor se colocó en el campo de población accesible para aplicar la FPC."
        )

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
                effect_size_options = [
                    "d de Cohen (Diferencia de Medias)",
                    "Diferencia de Medias (Absoluta)",
                    "Cambio pre-post (grupo único)",
                    "Cambio pre-post entre grupos",
                    "ANOVA (Cohen f)",
                    "Equivalencia (dos medias)"
                ]

        self.effect_size_type_combo['values'] = effect_size_options
        if effect_size_options:
            self.effect_size_type_combo.current(0)
            self.effect_size_type_combo.config(state="readonly")
        else:
            self.effect_size_type_combo.config(state="disabled") # Disable if no options

        self.update_specific_effect_inputs()
        self._refresh_precision_inputs_visibility()


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

        elif effect_type == "Cambio pre-post (grupo único)":
            ttk.Label(self.specific_effect_inputs_frame, text="Δ (post - pre):").grid(row=0, column=0, padx=2, pady=2, sticky=tk.W)
            self.paired_delta_entry = ttk.Entry(self.specific_effect_inputs_frame, textvariable=self.paired_delta_var, width=10)
            self.paired_delta_entry.grid(row=0, column=1, padx=2, pady=2, sticky=tk.W)

            ttk.Label(self.specific_effect_inputs_frame, text="DE de diferencias (σΔ):").grid(row=0, column=2, padx=2, pady=2, sticky=tk.W)
            self.paired_sd_diff_entry = ttk.Entry(self.specific_effect_inputs_frame, textvariable=self.paired_sd_diff_var, width=10)
            self.paired_sd_diff_entry.grid(row=0, column=3, padx=2, pady=2, sticky=tk.W)
            ttk.Label(self.specific_effect_inputs_frame, text="(Se usa d = Δ/σΔ)").grid(row=1, column=0, columnspan=4, padx=2, pady=2, sticky=tk.W)

        elif effect_type == "Cambio pre-post entre grupos":
            ttk.Label(self.specific_effect_inputs_frame, text="Cambio medio control (Mdc):").grid(row=0, column=0, padx=2, pady=2, sticky=tk.W)
            self.change_control_entry = ttk.Entry(self.specific_effect_inputs_frame, textvariable=self.change_control_var, width=10)
            self.change_control_entry.grid(row=0, column=1, padx=2, pady=2, sticky=tk.W)

            ttk.Label(self.specific_effect_inputs_frame, text="Cambio medio tratamiento (Mde):").grid(row=0, column=2, padx=2, pady=2, sticky=tk.W)
            self.change_treatment_entry = ttk.Entry(self.specific_effect_inputs_frame, textvariable=self.change_treatment_var, width=10)
            self.change_treatment_entry.grid(row=0, column=3, padx=2, pady=2, sticky=tk.W)

            ttk.Label(self.specific_effect_inputs_frame, text="DE de los cambios (σ):").grid(row=1, column=0, padx=2, pady=2, sticky=tk.W)
            self.change_sd_entry = ttk.Entry(self.specific_effect_inputs_frame, textvariable=self.change_sd_var, width=10)
            self.change_sd_entry.grid(row=1, column=1, padx=2, pady=2, sticky=tk.W)
            ttk.Label(self.specific_effect_inputs_frame, text="(d = (Mde - Mdc)/σ)").grid(row=1, column=2, columnspan=2, padx=2, pady=2, sticky=tk.W)

        elif effect_type == "Equivalencia (dos medias)":
            ttk.Label(self.specific_effect_inputs_frame, text="Margen equivalencia (Δeq):").grid(row=0, column=0, padx=2, pady=2, sticky=tk.W)
            ttk.Entry(self.specific_effect_inputs_frame, textvariable=self.equivalence_margin_var, width=10).grid(row=0, column=1, padx=2, pady=2, sticky=tk.W)

            ttk.Label(self.specific_effect_inputs_frame, text="Diferencia esperada (Δtrue):").grid(row=0, column=2, padx=2, pady=2, sticky=tk.W)
            ttk.Entry(self.specific_effect_inputs_frame, textvariable=self.equivalence_expected_diff_var, width=10).grid(row=0, column=3, padx=2, pady=2, sticky=tk.W)

            ttk.Label(self.specific_effect_inputs_frame, text="DE común (σ):").grid(row=1, column=0, padx=2, pady=2, sticky=tk.W)
            ttk.Entry(self.specific_effect_inputs_frame, textvariable=self.equivalence_sd_var, width=10).grid(row=1, column=1, padx=2, pady=2, sticky=tk.W)
            ttk.Label(self.specific_effect_inputs_frame, text="(usa fórmula TOST n = 2 (Zα + Zβ)^2 σ^2 / (Δeq - |Δtrue|)^2 )").grid(row=2, column=0, columnspan=4, padx=2, pady=2, sticky=tk.W)

        elif effect_type == "ANOVA (Cohen f)":
            ttk.Label(self.specific_effect_inputs_frame, text="Cohen f:").grid(row=0, column=0, padx=2, pady=2, sticky=tk.W)
            self.anova_effect_size_entry = ttk.Entry(self.specific_effect_inputs_frame, textvariable=self.anova_effect_size_var, width=10)
            self.anova_effect_size_entry.grid(row=0, column=1, padx=2, pady=2, sticky=tk.W)

            ttk.Label(self.specific_effect_inputs_frame, text="Referencia:").grid(row=0, column=2, padx=2, pady=2, sticky=tk.W)
            self.anova_effect_size_combo = ttk.Combobox(
                self.specific_effect_inputs_frame,
                textvariable=self.anova_effect_size_choice_var,
                values=["Pequeño (0.10)", "Mediano (0.25)", "Grande (0.40)"],
                width=15,
                state="readonly"
            )
            self.anova_effect_size_combo.grid(row=0, column=3, padx=2, pady=2, sticky=tk.W)
            self.anova_effect_size_combo.bind("<<ComboboxSelected>>", self._on_anova_effect_size_choice)

            ttk.Label(self.specific_effect_inputs_frame, text="Número de grupos (k):").grid(row=1, column=0, padx=2, pady=2, sticky=tk.W)
            self.anova_groups_entry = ttk.Entry(self.specific_effect_inputs_frame, textvariable=self.anova_groups_var, width=10)
            self.anova_groups_entry.grid(row=1, column=1, padx=2, pady=2, sticky=tk.W)

            ttk.Label(self.specific_effect_inputs_frame, text="Proporción respondientes (R):").grid(row=2, column=0, padx=2, pady=2, sticky=tk.W)
            self.response_rate_entry = ttk.Entry(self.specific_effect_inputs_frame, textvariable=self.response_rate_var, width=10)
            self.response_rate_entry.grid(row=2, column=1, padx=2, pady=2, sticky=tk.W)

            ttk.Label(self.specific_effect_inputs_frame, text="Proporción elegibles (E):").grid(row=2, column=2, padx=2, pady=2, sticky=tk.W)
            self.eligibility_rate_entry = ttk.Entry(self.specific_effect_inputs_frame, textvariable=self.eligibility_rate_var, width=10)
            self.eligibility_rate_entry.grid(row=2, column=3, padx=2, pady=2, sticky=tk.W)

    def _on_anova_effect_size_choice(self, event=None):
        selected = self.anova_effect_size_choice_var.get()
        if not selected:
            return
        try:
            if "(" in selected:
                value = selected.split("(")[-1].rstrip(")")
            else:
                value = selected
            self.anova_effect_size_var.set(value.strip())
        except Exception:
            pass

    def _parse_attrition_rate(self):
        try:
            rate = float(self.attrition_rate_var.get())
        except ValueError:
            raise ValueError("La proporción de pérdidas esperadas (R) debe ser numérica.")
        if not (0 <= rate < 1):
            raise ValueError("La proporción de pérdidas esperadas (R) debe estar entre 0 y 1.")
        return rate

    def _compute_power_numbers(self, per_group_estimate, group_count):
        if per_group_estimate is None or per_group_estimate <= 0:
            raise ValueError("El tamaño muestral calculado debe ser mayor que 0.")
        attr_rate = self._parse_attrition_rate()
        base_per_group = max(1, math.ceil(per_group_estimate))
        base_total = base_per_group * max(1, int(group_count))
        if attr_rate == 0:
            adjusted_total = base_total
        else:
            adjusted_total = math.ceil(base_total / (1 - attr_rate))
        adjusted_per_group = math.ceil(adjusted_total / max(1, int(group_count)))
        return {
            "attrition_rate": attr_rate,
            "base_per_group": base_per_group,
            "base_total": base_total,
            "adjusted_total": adjusted_total,
            "adjusted_per_group": adjusted_per_group,
            "group_count": max(1, int(group_count)),
        }

    def _present_power_result(self, *, effect_label=None, effect_value=None, numbers=None, extra_lines=None, header_note=None, formula_note=None):
        if numbers is None:
            return
        lines = []
        if header_note:
            lines.append(header_note)
        lines.append(self._tail_description())
        if effect_label is not None and effect_value is not None:
            lines.append(f"{effect_label}: {effect_value:.3f}")
        lines.append(
            f"n_calculado total: {numbers['base_total']} (≈ {numbers['base_per_group']} por grupo)"
        )
        attr_rate = numbers['attrition_rate']
        lines.append(
            f"n_ajustado total: {numbers['adjusted_total']} (≈ {numbers['adjusted_per_group']} por grupo)"
            + (f" aplicando pérdidas R={attr_rate:.0%}" if attr_rate > 0 else "")
        )
        if extra_lines:
            lines.extend(extra_lines)
        if formula_note:
            lines.append(f"Fórmula: {formula_note}")
        self.result_label_text_var.set("Tamaño de Muestra (potencia):")
        self.sample_size_result_var.set("\n".join(lines))

    def _is_one_sided(self):
        choice = (self.hypothesis_tail_var.get() or "").lower()
        return "unilateral" in choice

    def _tail_alternative(self, effect_direction=1.0):
        if not self._is_one_sided():
            return "two-sided"
        return "larger" if effect_direction >= 0 else "smaller"

    def _tail_description(self):
        return "Hipótesis: unilateral (una cola)" if self._is_one_sided() else "Hipótesis: bilateral (dos colas)"

    def _critical_z(self, alpha):
        if alpha <= 0 or alpha >= 1:
            raise ValueError("El alpha debe estar entre 0 y 1.")
        if self._is_one_sided():
            return norm.ppf(1 - alpha)
        return norm.ppf(1 - alpha / 2)

    def calculate_sample_size(self):
        # Check for critical imports for calculation
        # Updated check to include samplesize_confint_proportion and use proportion_effectsize
        if (
            TTestIndPower is None
            or TTestPower is None
            or NormalIndPower is None
            or proportion_effectsize is None
            or norm is None
            or samplesize_confint_proportion is None
        ):
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

                population_size = None
                population_raw = self.population_size_var.get().strip()
                if population_raw:
                    try:
                        population_size = float(population_raw)
                    except ValueError:
                        messagebox.showerror("Error de Entrada", "El tamaño de población accesible (N) debe ser numérico.")
                        return
                    if population_size <= 0:
                        messagebox.showerror("Error de Entrada", "El tamaño de población accesible (N) debe ser mayor que 0.")
                        return

                base_formula_desc = None
                proportion_scale_info = None

                if "proporciones" in var_type or "binarias" in var_type:
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
                    base_formula_desc = "N = Z^2 * p * (1 - p) / e^2"
                    proportion_scale_info = {
                        "p": estimated_p,
                        "margin": margin_of_error,
                    }

                elif "cuantitativos" in var_type:
                    estimated_sd = float(self.estimated_sd_var.get())
                    if estimated_sd <= 0:
                        messagebox.showerror("Error de Entrada", "La DE estimada debe ser positiva.")
                        return
                    if margin_of_error <= 0:
                        messagebox.showerror("Error de Entrada", "El margen de error para medias debe ser positivo.")
                        return
                    z_score = norm.ppf(1 - (alpha_precision / 2))
                    calculated_sample_size = (z_score * estimated_sd / margin_of_error)**2
                    base_formula_desc = "N = (Z * σ / e)^2"
                else:
                    messagebox.showinfo("Información", "Cálculo de precisión para este tipo de variable no implementado.")
                    self.sample_size_result_var.set("---")
                    return

                if calculated_sample_size is None or calculated_sample_size <= 0:
                    self.sample_size_result_var.set("Error calc.")
                    return

                raw_n0 = calculated_sample_size
                base_n = math.ceil(raw_n0)
                result_lines = [
                    f"n0 calculado (valor decimal): {raw_n0:.2f}",
                    f"n0 (población infinita, redondeado): {base_n}"
                ]
                final_n = base_n
                fpc_applied = False
                if population_size:
                    sampling_fraction = calculated_sample_size / population_size
                    fpc_required = sampling_fraction > 0.05 or self.force_fpc_var.get()
                    result_lines.append(
                        f"N accesible: {population_size:.0f}" if population_size.is_integer() else f"N accesible: {population_size}"
                    )
                    result_lines.append(f"Fracción n0/N: {sampling_fraction:.2%}")
                    if fpc_required:
                        fpc_n = calculated_sample_size / (1 + ((calculated_sample_size - 1) / population_size))
                        final_n = max(1, math.ceil(fpc_n))
                        result_lines.append(f"n ajustado (FPC, decimal): {fpc_n:.2f}")
                        result_lines.append(f"n ajustado (FPC, redondeado): {final_n}")
                        fpc_applied = True
                    else:
                        result_lines.append("n ajustado (FPC): no aplicado (n0/N <= 5%)")
                result_lines.append(f"n recomendado: {final_n}")
                if base_formula_desc:
                    result_lines.append(f"Fórmula base: {base_formula_desc}")
                if proportion_scale_info is not None:
                    result_lines.append(
                        f"P estimada: {proportion_scale_info['p']:.6f} ({proportion_scale_info['p']*100:.3f}%)"
                    )
                    result_lines.append(
                        f"Margen especificado: ±{proportion_scale_info['margin']:.6f} (±{proportion_scale_info['margin']*100:.3f} puntos porcentuales)"
                    )
                    if proportion_scale_info['p'] > 0:
                        relative_margin = proportion_scale_info['margin'] / proportion_scale_info['p']
                        result_lines.append(
                            f"Margen relativo: ±{relative_margin*100:.2f}% del valor esperado de P"
                        )
                    else:
                        result_lines.append("Margen relativo: no definido (P=0)")
                if population_size:
                    if fpc_applied:
                        result_lines.append("FPC: n = N * n0 / (n0 + N - 1)")
                    else:
                        result_lines.append("FPC: no aplicado (criterio 5% no superado)")
                self.sample_size_result_var.set("\n".join(result_lines))
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

                        es = proportion_effectsize(p1, p2, method='normal') # Changed here
                        alternative = self._tail_alternative(es)
                        power_analysis = NormalIndPower()
                        per_group = power_analysis.solve_power(
                            effect_size=es,
                            alpha=alpha,
                            power=power,
                            ratio=1.0,
                            alternative=alternative,
                            nobs=None
                        )
                        numbers = self._compute_power_numbers(per_group, 2)
                        self._present_power_result(
                            effect_label="h de Cohen",
                            effect_value=es,
                            numbers=numbers,
                            extra_lines=[f"Basado en proporciones P1={p1:.2f}, P2={p2:.2f}."],
                            formula_note="NormalIndPower (dos proporciones, cola según selección)"
                        )
                        return
                    elif effect_type_selected in ["Odds Ratio (OR)", "Riesgo Relativo (RR)"]:
                        baseline_var = self.p0_or_var if effect_type_selected == "Odds Ratio (OR)" else self.p0_rr_var
                        try:
                            baseline = float(baseline_var.get())
                        except (AttributeError, ValueError):
                            messagebox.showerror("Error de Entrada", "P0 debe ser numérico.")
                            return
                        if not (0 < baseline < 1):
                            messagebox.showerror("Error de Entrada", "P0 debe estar entre 0 y 1.")
                            return
                        if effect_type_selected == "Odds Ratio (OR)":
                            odds_ratio = float(self.or_var.get())
                            if odds_ratio <= 0:
                                messagebox.showerror("Error de Entrada", "El OR debe ser mayor que 0.")
                                return
                            numerator = odds_ratio * baseline
                            p1 = numerator / (1 - baseline + numerator)
                            descriptor = f"OR={odds_ratio:.3f}"
                        else:
                            risk_ratio = float(self.rr_var.get())
                            if risk_ratio <= 0:
                                messagebox.showerror("Error de Entrada", "El RR debe ser mayor que 0.")
                                return
                            p1 = risk_ratio * baseline
                            if not (0 < p1 < 1):
                                messagebox.showerror("Error de Entrada", "El RR produce una proporción >1. Ajuste RR o P0.")
                                return
                            descriptor = f"RR={risk_ratio:.3f}"
                        if not (0 < p1 < 1):
                            messagebox.showerror("Error de Entrada", "Las proporciones derivadas deben estar entre 0 y 1.")
                            return
                        effect_size = proportion_effectsize(baseline, p1, method='normal')
                        alternative = self._tail_alternative(effect_size)
                        power_analysis = NormalIndPower()
                        per_group = power_analysis.solve_power(
                            effect_size=effect_size,
                            alpha=alpha,
                            power=power,
                            ratio=1.0,
                            alternative=alternative,
                            nobs=None
                        )
                        numbers = self._compute_power_numbers(per_group, 2)
                        self._present_power_result(
                            effect_label="h de Cohen",
                            effect_value=effect_size,
                            numbers=numbers,
                            extra_lines=[f"P0={baseline:.3f}, P1 derivada={p1:.3f} ({descriptor})"],
                            formula_note="NormalIndPower usando OR/RR convertidos a proporciones"
                        )
                        return
                    else:
                        messagebox.showinfo("Información", "Seleccione un tipo de tamaño del efecto válido para variables categóricas/binarias.")
                        self.sample_size_result_var.set("---")
                        return

                elif "cuantitativos" in var_type:
                    if effect_type_selected == "d de Cohen (Diferencia de Medias)":
                        cohen_d = float(self.cohen_d_var.get())
                        alternative = self._tail_alternative(cohen_d)
                        power_analysis = TTestIndPower()
                        per_group = power_analysis.solve_power(
                            effect_size=cohen_d,
                            alpha=alpha,
                            power=power,
                            ratio=1.0,
                            alternative=alternative,
                            nobs=None
                        )
                        numbers = self._compute_power_numbers(per_group, 2)
                        self._present_power_result(
                            effect_label="d de Cohen",
                            effect_value=cohen_d,
                            numbers=numbers,
                            formula_note="TTestIndPower (t independiente, cola según selección)"
                        )
                        return
                    elif effect_type_selected == "Diferencia de Medias (Absoluta)":
                        mean1 = float(self.mean1_var.get())
                        mean2 = float(self.mean2_var.get())
                        common_sd = float(self.common_sd_var.get())
                        if common_sd <= 0:
                             messagebox.showerror("Error de Entrada", "La DE Común debe ser positiva.")
                             return

                        delta = mean2 - mean1
                        cohen_d_calculated = delta / common_sd
                        alternative = self._tail_alternative(cohen_d_calculated)
                        power_analysis = TTestIndPower()
                        per_group = power_analysis.solve_power(
                            effect_size=cohen_d_calculated,
                            alpha=alpha,
                            power=power,
                            ratio=1.0,
                            alternative=alternative,
                            nobs1=None
                        )
                        numbers = self._compute_power_numbers(per_group, 2)
                        self._present_power_result(
                            effect_label="d calculado",
                            effect_value=abs(cohen_d_calculated),
                            numbers=numbers,
                            extra_lines=[f"Δ de medias = {delta:.3f}"],
                            formula_note="TTestIndPower (t independiente, Δ/σ)"
                        )
                        return
                    elif effect_type_selected == "Cambio pre-post (grupo único)":
                        delta = float(self.paired_delta_var.get())
                        sd_diff = float(self.paired_sd_diff_var.get())
                        if sd_diff <= 0:
                            messagebox.showerror("Error de Entrada", "La DE de las diferencias debe ser positiva.")
                            return
                        effect = delta / sd_diff
                        alternative = self._tail_alternative(effect)
                        power_analysis = TTestPower()
                        n_total = power_analysis.solve_power(
                            effect_size=effect,
                            alpha=alpha,
                            power=power,
                            alternative=alternative,
                            nobs=None
                        )
                        numbers = self._compute_power_numbers(n_total, 1)
                        self._present_power_result(
                            effect_label="d pareado (Δ/σΔ)",
                            effect_value=abs(effect),
                            numbers=numbers,
                            header_note="Cambio pre-post dentro de un solo grupo",
                            extra_lines=[f"Δ observado = {delta:.3f}"],
                            formula_note="TTestPower (t pareado, cola según selección)"
                        )
                        return
                    elif effect_type_selected == "Cambio pre-post entre grupos":
                        change_control = float(self.change_control_var.get())
                        change_treatment = float(self.change_treatment_var.get())
                        sd_change = float(self.change_sd_var.get())
                        if sd_change <= 0:
                            messagebox.showerror("Error de Entrada", "La DE de los cambios debe ser positiva.")
                            return
                        delta_changes = change_treatment - change_control
                        effect = delta_changes / sd_change
                        alternative = self._tail_alternative(effect)
                        power_analysis = TTestIndPower()
                        per_group = power_analysis.solve_power(
                            effect_size=effect,
                            alpha=alpha,
                            power=power,
                            ratio=1.0,
                            alternative=alternative,
                            nobs=None
                        )
                        numbers = self._compute_power_numbers(per_group, 2)
                        self._present_power_result(
                            effect_label="d entre cambios",
                            effect_value=abs(effect),
                            numbers=numbers,
                            header_note="Comparación entre grupos del cambio pre-post",
                            extra_lines=[f"Δ tratamiento-control = {delta_changes:.3f}"],
                            formula_note="TTestIndPower (t independiente aplicado al cambio)"
                        )
                        return
                    elif effect_type_selected == "Equivalencia (dos medias)":
                        margin = float(self.equivalence_margin_var.get())
                        delta_true = float(self.equivalence_expected_diff_var.get())
                        sigma = float(self.equivalence_sd_var.get())
                        if margin <= 0:
                            messagebox.showerror("Error de Entrada", "El margen de equivalencia debe ser mayor que 0.")
                            return
                        if sigma <= 0:
                            messagebox.showerror("Error de Entrada", "La DE común debe ser positiva.")
                            return
                        if abs(delta_true) >= margin:
                            messagebox.showerror("Error de Entrada", "La diferencia esperada debe ser menor que el margen de equivalencia.")
                            return
                        z_alpha = self._critical_z(alpha)
                        z_beta = norm.ppf(power)
                        gap = margin - abs(delta_true)
                        numerator = (z_alpha + z_beta) * sigma
                        per_group = 2 * (numerator / gap) ** 2
                        numbers = self._compute_power_numbers(per_group, 2)
                        tail_note = "TOST (dos límites)" if not self._is_one_sided() else "No-inferioridad (una cola)"
                        self._present_power_result(
                            effect_label="Margen equivalencia",
                            effect_value=margin,
                            numbers=numbers,
                            extra_lines=[
                                f"Δtrue asumido = {delta_true:.3f}",
                                f"σ común = {sigma:.3f}",
                                tail_note
                            ],
                            formula_note="n = 2 * (Zcrit + Z(1-β))^2 * σ^2 / (Δeq - |Δtrue|)^2"
                        )
                        return
                    elif effect_type_selected == "ANOVA (Cohen f)":
                        if FTestAnovaPower is None:
                            messagebox.showerror(
                                "Error de Librería",
                                "'statsmodels' no dispone de FTestAnovaPower en este entorno. Instale/actualice statsmodels para continuar."
                            )
                            self.sample_size_result_var.set("Error")
                            return

                        effect_size = float(self.anova_effect_size_var.get())
                        k_groups = int(self.anova_groups_var.get())
                        if effect_size <= 0:
                            messagebox.showerror("Error de Entrada", "Cohen f debe ser mayor que 0.")
                            return
                        if k_groups < 2:
                            messagebox.showerror("Error de Entrada", "El número de grupos debe ser al menos 2.")
                            return

                        response_rate = float(self.response_rate_var.get())
                        eligibility_rate = float(self.eligibility_rate_var.get())
                        if not (0 < response_rate <= 1) or not (0 < eligibility_rate <= 1):
                            messagebox.showerror("Error de Entrada", "R y E deben estar entre 0 y 1.")
                            return

                        power_analysis = FTestAnovaPower()
                        n_per_group = power_analysis.solve_power(
                            effect_size=effect_size,
                            alpha=alpha,
                            power=power,
                            k_groups=k_groups,
                            nobs=None
                        )
                        total_n = n_per_group * k_groups
                        n_total = max(1, math.ceil(total_n))
                        adjusted_total = math.ceil(n_total / (response_rate * eligibility_rate))
                        per_group_adjusted = math.ceil(adjusted_total / k_groups)

                        lines = [
                            f"f usado: {effect_size:.3f}",
                            f"n_calculado (total): {n_total}",
                            f"n_ajustado (total): {adjusted_total}",
                            f"Distribuya ≈ {per_group_adjusted} sujetos por cada uno de los {k_groups} grupos.",
                            "Fórmula: FTestAnovaPower (ANOVA con f de Cohen)",
                            self._tail_description() + " (F siempre en cola superior)"
                        ]
                        self.result_label_text_var.set("Tamaño de Muestra (ANOVA):")
                        self.sample_size_result_var.set("\n".join(lines))
                        return
                    else:
                        messagebox.showinfo("Información", "Seleccione un tipo de tamaño del efecto válido para variables cuantitativas.")
                        self.sample_size_result_var.set("---")
                        return
                else:
                    messagebox.showinfo("Información", "Tipo de variable no soportado para cálculo de potencia con los parámetros actuales.")
                    self.sample_size_result_var.set("---")
                    return

            if calculated_sample_size is not None:
                self.sample_size_result_var.set(f"{math.ceil(calculated_sample_size)}")
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
