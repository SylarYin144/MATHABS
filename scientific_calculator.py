import tkinter as tk
from tkinter import ttk, font as tkfont
import math

class ScientificCalculatorTab(ttk.Frame):
    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)

        # Ampliamos la parrilla a 6 columnas para más funciones
        for i in range(6):
            self.grid_columnconfigure(i, weight=1)
        # El número de filas se ajusta más abajo

        self.display_var = tk.StringVar()
        self.display = ttk.Entry(self, textvariable=self.display_var, font=('Arial', 24), state='readonly', justify='right')
        self.display.grid(row=0, column=0, columnspan=6, sticky="nsew", padx=10, pady=10)

        # Estilo de los botones
        self.style = ttk.Style()
        self.style.configure('TButton', font=('Arial', 14), padding=10)
        self.style.configure('Sci.TButton', font=('Arial', 12))

        buttons = [
            # Fila 1
            ('sin', 1, 0, 1, 1, 'sci_unary'), ('cos', 1, 1, 1, 1, 'sci_unary'), ('tan', 1, 2, 1, 1, 'sci_unary'), ('log₁₀', 1, 3, 1, 1, 'sci_unary'), ('ln', 1, 4, 1, 1, 'sci_unary'), ('x!', 1, 5, 1, 1, 'sci_unary'),
            # Fila 2
            ('sin⁻¹', 2, 0, 1, 1, 'sci_unary'), ('cos⁻¹', 2, 1, 1, 1, 'sci_unary'), ('tan⁻¹', 2, 2, 1, 1, 'sci_unary'), ('√', 2, 3, 1, 1, 'sci_unary'), ('10ˣ', 2, 4, 1, 1, 'sci_unary'), ('1/x', 2, 5, 1, 1, 'sci_unary'),
            # Fila 3
            ('π', 3, 0, 1, 1, 'const'), ('e', 3, 1, 1, 1, 'const'), ('xʸ', 3, 2, 1, 1, 'op_binary'), ('DEL', 3, 3, 1, 1, 'func'), ('C', 3, 4, 1, 1, 'clear_all'), ('CE', 3, 5, 1, 1, 'clear_entry'),
            # Fila 4
            ('7', 4, 0, 1, 1, 'num'), ('8', 4, 1, 1, 1, 'num'), ('9', 4, 2, 1, 1, 'num'), ('/', 4, 3, 1, 1, 'op_binary'), ('*', 4, 4, 1, 1, 'op_binary'), ('-', 4, 5, 1, 1, 'op_binary'),
            # Fila 5
            ('4', 5, 0, 1, 1, 'num'), ('5', 5, 1, 1, 1, 'num'), ('6', 5, 2, 1, 1, 'num'), ('+', 5, 3, 1, 1, 'op_binary'),
            # Fila 6
            ('1', 6, 0, 1, 1, 'num'), ('2', 6, 1, 1, 1, 'num'), ('3', 6, 2, 1, 1, 'num'),
            # Fila 7
            ('0', 7, 0, 1, 2, 'num'), ('±', 7, 2, 1, 1, 'op_unary'), ('.', 7, 3, 1, 1, 'num'),
            # Botones que ocupan varias celdas
            ('=', 5, 4, 3, 2, 'equals')
        ]

        # Aumentar el número de filas para acomodar el nuevo layout
        for i in range(8):
            self.grid_rowconfigure(i, weight=1)

        self.buttons = []
        for i, (text, r, c, cs, rs, btype) in enumerate(buttons):
            base_style = 'TButton' if btype in ['num', 'op_binary', 'equals', 'clear_all', 'clear_entry', 'func'] else 'Sci.TButton'
            unique_style = f'B{i}.{base_style}'
            button = ttk.Button(self, text=text, style=unique_style, command=lambda t=text, type=btype: self.on_button_click(t, type))
            button.grid(row=r, column=c, columnspan=cs, rowspan=rs, sticky="nsew", padx=5, pady=5)
            self.buttons.append(button)

        self.bind("<Configure>", self._adjust_font_size)
        self.after(10, self._adjust_font_size)

        self.first_operand = None
        self.operator = None
        self.clear_display_on_next_input = False

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
