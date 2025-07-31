import tkinter as tk
from tkinter import ttk, font as tkfont, simpledialog
import math

class ScientificCalculatorTab(ttk.Frame):
    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)

        # --- Layout Principal (Calculadora a la izquierda, Memoria a la derecha) ---
        self.grid_columnconfigure(0, weight=3) # Frame de la calculadora
        self.grid_columnconfigure(1, weight=1) # Frame de la memoria
        self.grid_rowconfigure(0, weight=1)

        calc_frame = ttk.Frame(self)
        calc_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 5))

        mem_frame = ttk.Frame(self)
        mem_frame.grid(row=0, column=1, sticky="nsew", padx=(5, 0))

        # --- UI de la Calculadora (en calc_frame) ---
        for i in range(6):
            calc_frame.grid_columnconfigure(i, weight=1)
        for i in range(8):
            calc_frame.grid_rowconfigure(i, weight=1)

        self.display_var = tk.StringVar()
        self.display = ttk.Entry(calc_frame, textvariable=self.display_var, font=('Arial', 24), state='readonly', justify='right')
        self.display.grid(row=0, column=0, columnspan=6, sticky="nsew", padx=5, pady=5)

        self.style = ttk.Style()
        self.style.configure('TButton', font=('Arial', 14), padding=10)
        self.style.configure('Sci.TButton', font=('Arial', 12))

        self.color_map = {
            'num': '#E6F0FF', 'op_binary': '#FFE4C4', 'sci_unary': '#E0FBE2',
            'const': '#D1F2EB', 'equals': '#D4E4FF', 'clear_all': '#FADADD',
            'clear_entry': '#FADADD', 'func': '#F0E6FF'
        }

        buttons = [
            ('sin', 1, 0), ('cos', 1, 1), ('tan', 1, 2), ('log₁₀', 1, 3), ('ln', 1, 4), ('x!', 1, 5, 'sci_unary'),
            ('sin⁻¹', 2, 0), ('cos⁻¹', 2, 1), ('tan⁻¹', 2, 2), ('√', 2, 3), ('10ˣ', 2, 4), ('1/x', 2, 5, 'sci_unary'),
            ('π', 3, 0, 'const'), ('e', 3, 1, 'const'), ('xʸ', 3, 2, 'op_binary'), ('DEL', 3, 3, 'func'), ('C', 3, 4, 'clear_all'), ('CE', 3, 5, 'clear_entry'),
            ('7', 4, 0), ('8', 4, 1), ('9', 4, 2), ('/', 4, 3, 'op_binary'), ('*', 4, 4, 'op_binary'), ('-', 4, 5, 'op_binary'),
            ('4', 5, 0), ('5', 5, 1), ('6', 5, 2), ('+', 5, 3, 'op_binary'),
            ('1', 6, 0), ('2', 6, 1), ('3', 6, 2),
            ('0', 7, 0, 'num', 1, 2), ('±', 7, 2, 'op_unary'), ('.', 7, 3),
            ('=', 5, 4, 'equals', 3, 2)
        ]

        self.buttons = []
        for i, btn_info in enumerate(buttons):
            text, r, c = btn_info[0], btn_info[1], btn_info[2]
            btype = btn_info[3] if len(btn_info) > 3 else ('num' if text.isdigit() or text == '.' else 'sci_unary')
            rs = btn_info[4] if len(btn_info) > 4 else 1
            cs = btn_info[5] if len(btn_info) > 5 else 1

            base_style = 'TButton' if btype in ['num', 'op_binary', 'equals', 'clear_all', 'clear_entry', 'func', 'op_unary'] else 'Sci.TButton'
            unique_style = f'B{i}.{base_style}'
            bg_color = self.color_map.get(btype, '#FFFFFF')
            self.style.configure(unique_style, background=bg_color)

            button = ttk.Button(calc_frame, text=text, style=unique_style, command=lambda t=text, type=btype: self.on_button_click(t, type))
            button.grid(row=r, column=c, rowspan=rs, columnspan=cs, sticky="nsew", padx=2, pady=2)
            self.buttons.append(button)

        self.bind("<Configure>", self._adjust_font_size)
        self.after(10, self._adjust_font_size)
        self.bind_all("<Control-v>", self._paste_from_clipboard)

        # --- UI de la Memoria (en mem_frame) ---
        mem_frame.grid_rowconfigure(1, weight=1)
        mem_frame.grid_columnconfigure(0, weight=1)

        mem_label = ttk.Label(mem_frame, text="Memorias", font=('Arial', 14, 'bold'))
        mem_label.grid(row=0, column=0, pady=(5,10), sticky='w')

        tree_cols = ('name', 'value')
        self.memory_tree = ttk.Treeview(mem_frame, columns=tree_cols, show='headings')
        self.memory_tree.heading('name', text='Nombre')
        self.memory_tree.heading('value', text='Valor')
        self.memory_tree.column('name', width=80, anchor='w', stretch=tk.NO)
        self.memory_tree.column('value', anchor='w')
        self.memory_tree.grid(row=1, column=0, sticky='nsew')

        mem_btn_frame = ttk.Frame(mem_frame)
        mem_btn_frame.grid(row=2, column=0, sticky='ew', pady=5)
        mem_btn_frame.grid_columnconfigure((0, 1, 2), weight=1)

        self.btn_m_store = ttk.Button(mem_btn_frame, text="Guardar", command=self._memory_store)
        self.btn_m_recall = ttk.Button(mem_btn_frame, text="Recuperar", command=self._memory_recall)
        self.btn_m_clear = ttk.Button(mem_btn_frame, text="Limpiar", command=self._memory_clear)

        self.btn_m_store.grid(row=0, column=0, sticky='ew', padx=2)
        self.btn_m_recall.grid(row=0, column=1, sticky='ew', padx=2)
        self.btn_m_clear.grid(row=0, column=2, sticky='ew', padx=2)

        # --- Variables de Estado ---
        self.first_operand = None
        self.operator = None
        self.clear_display_on_next_input = False
        self.memories = {}

    def _adjust_font_size(self, event=None):
        max_font_size = 18
        min_font_size = 8
        padding = 10  # px

        for button in self.buttons:
            text = button.cget("text")
            if not text:
                continue

            button_width = button.winfo_width()
            if button_width <= padding:
                continue

            font_size = max_font_size

            temp_font = tkfont.Font(family="Arial", size=font_size)
            text_width = temp_font.measure(text)

            while text_width > button_width - padding and font_size > min_font_size:
                font_size -= 1
                temp_font.config(size=font_size)
                text_width = temp_font.measure(text)

            style_name = button.cget("style")
            self.style.configure(style_name, font=("Arial", font_size))

    def _paste_from_clipboard(self, event=None):
        try:
            clipboard_content = self.clipboard_get()
            # Validar que el contenido sea un número (entero o flotante)
            float(clipboard_content)

            # Si es un número válido, lo ponemos en la pantalla, reseteando el estado.
            self.display_var.set(clipboard_content)
            self.clear_display_on_next_input = False
            self.first_operand = None
            self.operator = None
        except (tk.TclError, ValueError):
            # TclError: el portapapeles está vacío o no contiene texto.
            # ValueError: el contenido no es un número válido.
            # En ambos casos, la operación de pegado se ignora silenciosamente.
            pass

    def _display_error(self, message="Error"):
        self.display_var.set(message)
        self.clear_display_on_next_input = True
        self.first_operand = None
        self.operator = None

    def on_button_click(self, char, button_type):
        current_text = self.display_var.get()
        if current_text == "Error" or current_text == "Infinity": # Reset on new input if error was shown
            current_text = ""
            self.display_var.set("")


        elif button_type == 'num':
            if self.clear_display_on_next_input:
                current_text = ""
                self.clear_display_on_next_input = False

            if char == '.' and '.' in current_text: return # Avoid multiple dots

            self.display_var.set(current_text + char)

        elif button_type == 'const':
            if self.clear_display_on_next_input: # If a result was just shown, replace it
                current_text = ""
            # If not, and there's already a number, user might want to multiply (advanced)
            # For now, just append or replace if display was to be cleared.
            if self.clear_display_on_next_input:
                self.clear_display_on_next_input = False

            if char == 'π':
                self.display_var.set(current_text + str(math.pi))
            elif char == 'e':
                self.display_var.set(current_text + str(math.e))

        elif button_type == 'clear_all': # C
            self.display_var.set("")
            self.first_operand = None
            self.operator = None
            self.clear_display_on_next_input = False

        elif button_type == 'clear_entry': # CE
            self.display_var.set("")
            # Don't clear operator or first_operand, CE means clear current input field

        elif button_type == 'func' and char == 'DEL':
            if self.clear_display_on_next_input: # If a result was just shown, DEL clears it
                self.display_var.set("")
                self.clear_display_on_next_input = False
            else:
                self.display_var.set(current_text[:-1])

        elif button_type == 'op_binary': # +, -, *, /, xʸ
            if current_text and current_text != "-": # Ensure current_text is a valid number
                # If there's already an operator and first_operand, calculate intermediate result
                if self.first_operand is not None and self.operator is not None and not self.clear_display_on_next_input:
                    self.on_button_click('=', 'equals') # Calculate previous
                    current_text = self.display_var.get() # Get result of previous calculation
                    if current_text == "Error" or current_text == "Infinity": return

                try:
                    self.first_operand = float(current_text)
                    self.operator = char
                    self.clear_display_on_next_input = True
                except ValueError:
                    if current_text: # Avoid error if current_text is empty after a CE for example
                        self._display_error()
            # Allow changing operator if no new number has been input yet
            elif self.first_operand is not None:
                self.operator = char # Update operator
                self.clear_display_on_next_input = True # Expect new number


        elif button_type == 'op_unary' and char == '±':
            if current_text and current_text != "0" and current_text != "Error" and current_text != "Infinity":
                if self.clear_display_on_next_input: # If result is shown, negate the result
                    self.first_operand = None # No longer a pending operation with this result
                    self.operator = None
                    self.clear_display_on_next_input = False

                if current_text.startswith('-'):
                    self.display_var.set(current_text[1:])
                else:
                    self.display_var.set('-' + current_text)

        elif button_type == 'sci_unary': # sin, cos, tan, log₁₀, ln, √
            if current_text and current_text != "-":
                try:
                    value = float(current_text)
                    result = 0
                    if char == 'sin':
                        result = math.sin(math.radians(value)) # Assuming degrees input
                    elif char == 'cos':
                        result = math.cos(math.radians(value)) # Assuming degrees input
                    elif char == 'tan':
                        # Avoid tan(90), tan(270), etc.
                        if (value % 180) == 90:
                             self._display_error("Infinity")
                             return
                        result = math.tan(math.radians(value)) # Assuming degrees input
                    elif char == 'log₁₀':
                        if value <= 0:
                            self._display_error("Error: log(≤0)")
                            return
                        result = math.log10(value)
                    elif char == 'ln':
                        if value <= 0:
                            self._display_error("Error: ln(≤0)")
                            return
                        result = math.log(value)
                    elif char == '√':
                        if value < 0:
                            self._display_error("Error: √(<0)")
                            return
                        result = math.sqrt(value)
                    elif char == 'x!':
                        if value < 0 or value != int(value):
                            self._display_error("Error: Factorial(int≥0)")
                            return
                        if value > 20: # Limitar factorial para evitar overflow
                            self._display_error("Error: Factorial(>20)")
                            return
                        result = math.factorial(int(value))
                    elif char == '1/x':
                        if value == 0:
                            self._display_error("Infinity")
                            return
                        result = 1 / value
                    elif char == '10ˣ':
                        result = math.pow(10, value)
                    elif char == 'sin⁻¹':
                        if not -1 <= value <= 1:
                            self._display_error("Error: asin(rango)")
                            return
                        result = math.degrees(math.asin(value))
                    elif char == 'cos⁻¹':
                        if not -1 <= value <= 1:
                            self._display_error("Error: acos(rango)")
                            return
                        result = math.degrees(math.acos(value))
                    elif char == 'tan⁻¹':
                        result = math.degrees(math.atan(value))

                    # Round to a reasonable number of decimal places
                    if abs(result) < 1e-10 and abs(result) != 0: # Handle very small numbers as 0 or sci notation
                        result_str = f"{result:.10e}"
                    elif abs(result) > 1e15:
                         result_str = f"{result:.10e}"
                    else:
                        result_str = str(round(result, 10))

                    self.display_var.set(result_str)
                    self.clear_display_on_next_input = True
                    self.first_operand = None
                    self.operator = None

                except ValueError:
                    self._display_error("Error: Invalid input")
                except Exception as e:
                    self._display_error(f"Error: {str(e)[:10]}")


        elif button_type == 'equals': # =
            if self.first_operand is not None and self.operator is not None and current_text and current_text != "-":
                try:
                    second_operand = float(current_text)
                    result = 0
                    if self.operator == '+':
                        result = self.first_operand + second_operand
                    elif self.operator == '-':
                        result = self.first_operand - second_operand
                    elif self.operator == '*':
                        result = self.first_operand * second_operand
                    elif self.operator == '/':
                        if second_operand == 0:
                            self._display_error("Infinity")
                            return
                        result = self.first_operand / second_operand
                    elif self.operator == 'xʸ':
                        # Handle large exponents carefully, may lead to OverflowError
                        if self.first_operand == 0 and second_operand < 0:
                            self._display_error("Error: 0^negative")
                            return
                        try:
                            result = self.first_operand ** second_operand
                            if abs(result) > 1e15 : # Check for potential overflow not caught
                                raise OverflowError("Result too large")
                        except OverflowError:
                            self._display_error("Overflow")
                            return


                    if abs(result) < 1e-10 and abs(result) != 0:
                        result_str = f"{result:.10e}"
                    elif abs(result) > 1e15: # Numbers too large for standard float display
                         result_str = f"{result:.10e}"
                    else:
                        result_str = str(round(result, 10))

                    self.display_var.set(result_str)
                    self.first_operand = None
                    self.operator = None
                    self.clear_display_on_next_input = True

                except ValueError:
                    self._display_error("Error: Invalid input")
                except ZeroDivisionError: # Should be caught by the check above, but as a fallback
                    self._display_error("Infinity")
                except Exception as e: # Catch any other math errors
                    self._display_error(f"Error: {str(e)[:10]}")
            # If only a number is present and = is pressed, or op is missing, do nothing.
            # Allow re-pressing equals if an operation was just completed.
            elif current_text and self.first_operand is None and self.operator is None and self.clear_display_on_next_input:
                # This means a result is already on display. Pressing = again does nothing to it.
                pass


    # --- Métodos de Memoria ---

    def _update_memory_tree(self):
        """Borra y vuelve a poblar el Treeview con el diccionario de memorias."""
        for i in self.memory_tree.get_children():
            self.memory_tree.delete(i)
        for name, value in self.memories.items():
            self.memory_tree.insert('', 'end', iid=name, values=(name, value))

    def _memory_store(self):
        """Guarda el valor actual de la pantalla en una memoria con nombre."""
        current_value_str = self.display_var.get()
        try:
            value_to_store = float(current_value_str)
        except ValueError:
            return

        name = simpledialog.askstring("Guardar Memoria", "Introduce un nombre para el valor:", parent=self)
        if name:
            self.memories[name] = value_to_store
            self._update_memory_tree()

    def _memory_recall(self):
        """Recupera un valor de la memoria y lo pone en pantalla."""
        selected_item = self.memory_tree.focus()
        if not selected_item:
            return

        recalled_value = self.memories.get(selected_item)
        if recalled_value is not None:
            self.display_var.set(str(recalled_value))
            self.clear_display_on_next_input = False
            self.first_operand = None
            self.operator = None

    def _memory_clear(self):
        """Limpia la memoria seleccionada."""
        selected_item = self.memory_tree.focus()
        if not selected_item:
            return

        if selected_item in self.memories:
            del self.memories[selected_item]
            self._update_memory_tree()


if __name__ == '__main__':
    root = tk.Tk()
    root.title("Scientific Calculator Test")

    style = ttk.Style(root)
    available_themes = style.theme_names()
    if 'clam' in available_themes:
        style.theme_use('clam')

    notebook = ttk.Notebook(root)

    calculator_tab = ScientificCalculatorTab(notebook)
    notebook.add(calculator_tab, text='Scientific Calculator')

    notebook.pack(expand=True, fill='both', padx=10, pady=10)

    root.geometry("400x550")
    root.mainloop()
