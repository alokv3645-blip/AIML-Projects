"""
AI/ML Projects Studio - a desktop GUI for both projects.
    python app.py

Tab 1: Student Performance Prediction System
Tab 2: House Price Prediction System
Each tab lets you load a CSV, run the full ML pipeline, browse the graphs,
read the step-by-step log, and make live predictions.
"""
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

from data_generator import LOCATIONS, ensure_datasets
from house_price import FURNISHING, HousePriceSystem
from main import save
from student_performance import StudentPerformanceSystem

PASS_C, FAIL_C, BLUE = "#2E9E5B", "#D9534F", "#1F4E79"


class Form(ttk.LabelFrame):
    """Builds input widgets from a field spec: (key, label, 'num'|'choice', args)."""

    def __init__(self, master, title, fields):
        super().__init__(master, text=title, padding=8)
        self.vars = {}
        for r, (key, label, kind, arg) in enumerate(fields):
            ttk.Label(self, text=label).grid(row=r, column=0, sticky="w", pady=2)
            if kind == "num":
                lo, hi, default, step = arg
                var = tk.DoubleVar(value=default)
                widget = ttk.Spinbox(self, from_=lo, to=hi, increment=step, textvariable=var, width=12)
            else:
                options, default = arg
                var = tk.StringVar(value=default)
                widget = ttk.Combobox(self, values=options, textvariable=var, state="readonly", width=12)
            widget.grid(row=r, column=1, padx=(8, 0), pady=2, sticky="e")
            self.vars[key] = (var, kind, label)

    def values(self):
        out = {}
        for key, (var, kind, label) in self.vars.items():
            try:
                out[key] = float(var.get()) if kind == "num" else var.get()
            except tk.TclError:
                raise ValueError(f"Please enter a valid number for '{label}'.")
        return out


class ProjectTab(ttk.Frame):
    def __init__(self, master, title, system_cls, default_csv, fields, describe, out_prefix):
        super().__init__(master, padding=8)
        self.system_cls, self.csv_path, self.describe, self.out_prefix = system_cls, default_csv, describe, out_prefix
        self.system = None

        # ---- left control panel
        left = ttk.Frame(self)
        left.pack(side="left", fill="y", padx=(0, 10))
        ttk.Label(left, text=title, font=("Segoe UI", 13, "bold"), foreground=BLUE).pack(anchor="w", pady=(0, 6))

        box = ttk.LabelFrame(left, text="1. Dataset", padding=8)
        box.pack(fill="x")
        self.path_var = tk.StringVar(value=str(default_csv.name))
        ttk.Label(box, textvariable=self.path_var, wraplength=250).pack(anchor="w")
        ttk.Button(box, text="Browse CSV...", command=self.browse).pack(fill="x", pady=(6, 0))

        box = ttk.LabelFrame(left, text="2. Train & Evaluate", padding=8)
        box.pack(fill="x", pady=8)
        ttk.Button(box, text="Run Full Pipeline", command=self.run).pack(fill="x")
        self.export_btn = ttk.Button(box, text="Export Graphs + Report", command=self.export, state="disabled")
        self.export_btn.pack(fill="x", pady=(6, 0))
        self.metrics_var = tk.StringVar(value="Not trained yet")
        ttk.Label(box, textvariable=self.metrics_var, font=("Segoe UI", 10, "bold"),
                  foreground=PASS_C, justify="left").pack(anchor="w", pady=(8, 0))

        self.form = Form(left, "3. Predict", fields)
        self.form.pack(fill="x")
        self.predict_btn = ttk.Button(self.form, text="Predict", command=self.predict, state="disabled")
        self.predict_btn.grid(row=len(fields), column=0, columnspan=2, sticky="ew", pady=(8, 0))
        self.result_var = tk.StringVar(value="Train the models first")
        self.result_lbl = ttk.Label(left, textvariable=self.result_var, font=("Segoe UI", 14, "bold"),
                                    wraplength=270, justify="left")
        self.result_lbl.pack(anchor="w", pady=(10, 2))
        self.detail_var = tk.StringVar()
        ttk.Label(left, textvariable=self.detail_var, font=("Segoe UI", 10), justify="left").pack(anchor="w")

        # ---- right: log + graphs
        self.views = ttk.Notebook(self)
        self.views.pack(side="left", fill="both", expand=True)
        self.log_box = scrolledtext.ScrolledText(self.views, font=("Consolas", 10), wrap="none")
        self.views.add(self.log_box, text="Log / Report")
        self.log_box.insert("end", "Click 'Run Full Pipeline' to load, clean, analyse, train and evaluate.\n")
        self.fig_tabs = []

    # ---- helpers
    def log(self, msg):
        self.log_box.insert("end", str(msg) + "\n")
        self.log_box.see("end")
        self.update_idletasks()

    def browse(self):
        p = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if p:
            self.csv_path = p
            self.path_var.set(p)

    def run(self):
        for t in self.fig_tabs:
            self.views.forget(t)
        self.fig_tabs.clear()
        self.log_box.delete("1.0", "end")
        self.metrics_var.set("Training...")
        self.config(cursor="watch")
        self.update()
        try:
            self.system = self.system_cls(log=self.log)
            res = self.system.run_all(self.csv_path)
        except Exception as e:
            self.metrics_var.set("Failed")
            messagebox.showerror("Pipeline error", str(e))
            return
        finally:
            self.config(cursor="")
        for name, fig in res["figures"].items():
            frame = ttk.Frame(self.views)
            canvas = FigureCanvasTkAgg(fig, master=frame)
            NavigationToolbar2Tk(canvas, frame)
            canvas.get_tk_widget().pack(fill="both", expand=True)
            canvas.draw()
            self.views.add(frame, text=name)
            self.fig_tabs.append(frame)
        self.metrics_var.set(res["summary"])
        self.predict_btn.state(["!disabled"])
        self.export_btn.state(["!disabled"])
        self.result_var.set("Fill the form and press Predict")
        self.views.select(self.fig_tabs[-1] if self.fig_tabs else 0)

    def predict(self):
        try:
            head, detail, ok = self.describe(self.system.predict(self.form.values()))
        except ValueError as e:
            messagebox.showwarning("Invalid input", str(e))
            return
        self.result_var.set(head)
        self.detail_var.set(detail)
        self.result_lbl.configure(foreground=PASS_C if ok else FAIL_C)

    def export(self):
        lines = self.log_box.get("1.0", "end").splitlines()
        save(self.out_prefix, self.system, lines)
        messagebox.showinfo("Exported", "Graphs (PNG) and the report were saved in the 'outputs' folder.")


def describe_student(r):
    ok = r["result"] == "PASS"
    return (f"{'✔' if ok else '✘'} {r['result']}",
            f"Estimated marks: {r['marks']:.1f} / 100\nPass probability: {r['pass_probability'] * 100:.0f}%", ok)


def describe_house(r):
    others = "\n".join(f"  {k}: Rs. {v:.1f} L" for k, v in r["by_model"].items())
    return r["formatted"], f"Predicted price (best model)\n\nAll models:\n{others}", True


STUDENT_FIELDS = [
    ("study_hours", "Study hours / day", "num", (0, 16, 4.5, 0.5)),
    ("attendance", "Attendance %", "num", (0, 100, 80, 1)),
    ("previous_score", "Previous score %", "num", (0, 100, 60, 1)),
    ("sleep_hours", "Sleep hours", "num", (2, 12, 7, 0.5)),
    ("assignments_completed", "Assignments (0-10)", "num", (0, 10, 6, 1)),
    ("parental_support", "Parental support", "choice", (["Low", "Medium", "High"], "Medium")),
    ("extra_classes", "Extra classes", "choice", (["No", "Yes"], "No")),
]
HOUSE_FIELDS = [
    ("area_sqft", "Area (sq ft)", "num", (300, 6000, 1400, 50)),
    ("bedrooms", "Bedrooms", "num", (1, 8, 3, 1)),
    ("bathrooms", "Bathrooms", "num", (1, 8, 2, 1)),
    ("age_years", "Age (years)", "num", (0, 60, 5, 1)),
    ("parking", "Parking slots", "num", (0, 5, 1, 1)),
    ("distance_to_center_km", "Distance to centre (km)", "num", (0, 60, 8, 0.5)),
    ("location", "Location", "choice", (LOCATIONS, "Suburb")),
    ("furnishing", "Furnishing", "choice", (list(FURNISHING), "Semi-Furnished")),
]


def main():
    student_csv, house_csv = ensure_datasets()
    root = tk.Tk()
    root.title("AI / ML Projects Studio")
    root.geometry("1280x800")
    ttk.Style().theme_use("clam")
    tabs = ttk.Notebook(root)
    tabs.pack(fill="both", expand=True, padx=6, pady=6)
    tabs.add(ProjectTab(tabs, "Student Performance Prediction", StudentPerformanceSystem, student_csv,
                        STUDENT_FIELDS, describe_student, "student"), text="  Project 1: Student Performance  ")
    tabs.add(ProjectTab(tabs, "House Price Prediction", HousePriceSystem, house_csv,
                        HOUSE_FIELDS, describe_house, "house"), text="  Project 2: House Price  ")
    root.mainloop()


if __name__ == "__main__":
    main()
