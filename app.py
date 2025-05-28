import tkinter as tk
from tkinter import ttk
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
import main
import inspect

matplotlib.use('TkAgg')
plt.rcParams.update({
    "text.usetex": False,  # включи True, если есть LaTeX + dvipng/ghostscript
    "mathtext.fontset": "stix",  # использовать mathtext (встроенный)
    "font.family": "STIXGeneral"
})

DATA_DIR = "OUTPUT"

functions = {
    name: func for name, func in inspect.getmembers(main, inspect.isfunction)
    if func.__module__ == 'main'
}

cores = {'PT1-1': 'ptc3', 'PT1-2': 'bay', 'PT1-3': 'nez'}


class TaskApp:
    def __init__(self, root):
        self.root = root
        root.title("Генератор заданий")

        self.theme_var = tk.StringVar()
        self.task_var = tk.StringVar()

        ttk.Label(root, text="Выберите тему:").grid(row=0, column=0, sticky='w')
        self.theme_menu = ttk.Combobox(root, textvariable=self.theme_var, state='readonly')
        self.theme_menu.grid(row=0, column=1, padx=5, pady=5)
        self.theme_menu.bind("<<ComboboxSelected>>", self.update_tasks)

        ttk.Label(root, text="Выберите задание:").grid(row=1, column=0, sticky='w')
        self.task_menu = ttk.Combobox(root, textvariable=self.task_var, state='readonly')
        self.task_menu.grid(row=1, column=1, padx=5, pady=5)

        self.generate_button = ttk.Button(root, text="Составить задание", command=self.display_task)
        self.generate_button.grid(row=2, column=0, columnspan=2, pady=10)

        ttk.Label(root, text="Формулировка задания:").grid(row=3, column=0, sticky='nw')
        self.task_canvas = tk.Canvas(root, width=700, height=200)
        self.task_canvas.grid(row=4, column=0, columnspan=2)

        ttk.Label(root, text="Решение:").grid(row=5, column=0, sticky='nw')
        self.solution_canvas = tk.Canvas(root, width=700, height=300)
        self.solution_canvas.grid(row=6, column=0, columnspan=2)

        self.task_fig_canvas = None
        self.solution_fig_canvas = None

        self.refresh_themes()

    def refresh_themes(self):
        themes = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]
        self.theme_menu['values'] = themes
        if themes:
            self.theme_var.set(themes[0])
            self.update_tasks()

    def update_tasks(self, event=None):
        theme = self.theme_var.get()
        prefix = cores.get(theme, "")
        task_names = [s for s in functions.keys() if s.startswith(prefix)]
        self.task_menu['values'] = task_names
        if task_names:
            self.task_var.set(task_names[0])

    def render_latex(self, canvas_widget, fig_canvas_ref, text, height):
        fig = plt.figure(figsize=(8, height), dpi=150)
        fig.text(0.1, 0.9, text, ha='left', va='top', wrap=True)

        if fig_canvas_ref and fig_canvas_ref.get_tk_widget():
            fig_canvas_ref.get_tk_widget().destroy()

        fig_canvas = FigureCanvasTkAgg(fig, master=canvas_widget)
        fig_canvas.draw()
        fig_canvas.get_tk_widget().pack()
        return fig_canvas

    def display_task(self):
        theme = self.theme_var.get()
        task_name = self.task_var.get()
        if not (theme and task_name):
            return

        functions[task_name]()

        if task_name.startswith('ptc'):
            task_name = task_name[:4] + '-' + task_name[4:]

        task_path = os.path.join(DATA_DIR, theme, f"{task_name}-data.txt")
        solution_path = os.path.join(DATA_DIR, theme, f"{task_name}-answer.txt")

        task_text = "[Задание не найдено]"
        if os.path.exists(task_path):
            with open(task_path, encoding='Windows-1251') as f:
                task_text = f.read()

        solution_text = "[Решение не найдено]"
        if os.path.exists(solution_path):
            with open(solution_path, encoding='utf-8') as f:
                solution_text = f.read()

        # Рендеринг
        self.task_fig_canvas = self.render_latex(self.task_canvas, self.task_fig_canvas, task_text, height=2)
        self.solution_fig_canvas = self.render_latex(self.solution_canvas, self.solution_fig_canvas, solution_text, height=10)


if __name__ == "__main__":
    root = tk.Tk()
    app = TaskApp(root)

    def on_close():
        plt.close('all')  # закрыть все matplotlib-фигуры
        root.destroy()  # завершить главный цикл Tkinter

    root.protocol("WM_DELETE_WINDOW", on_close)  # обработчик закрытия
    root.mainloop()