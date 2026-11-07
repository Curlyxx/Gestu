"""Panel de control (Tkinter + Matplotlib). Cerrar la ventana NO detiene el demonio."""
import queue
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import messagebox, ttk

import numpy as np
import yaml
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from . import osinfo, status
from .actions import resolve_backend
from .config import load_settings, update_settings
from .paths import ACTIONS_FILE, HISTORY_FILE, ROOT, SCRIPTS, TEST_RESULTS_FILE, ensure_dirs

GREEN, RED, AMBER = "#1e8e3e", "#c5221f", "#b06000"


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Gesture Control")
        self.geometry("1020x780")
        self.minsize(900, 680)
        self.settings = load_settings()
        ensure_dirs(self.settings["gestures"])
        self.info = osinfo.detect()
        self.backend = resolve_backend(self.settings, self.info)
        self.cam = None
        self.proc = None
        self.q = queue.Queue()
        self.task_buttons = []
        self.dry = tk.BooleanVar(value=True)
        self.preview = tk.BooleanVar(value=False)
        self._build()
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.check_camera()
        self.refresh()
        self.after(100, self.pump)
        self.after(1000, self.tick)

    # ------------------------------------------------------------ UI
    def _build(self):
        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True)
        home, plots, cfg, acts = (ttk.Frame(nb, padding=8) for _ in range(4))
        nb.add(home, text="Inicio")
        nb.add(plots, text="Gráficas")
        nb.add(cfg, text="Configuración")
        nb.add(acts, text="Acciones")
        self._build_home(home)
        self._build_plots(plots)
        self._build_cfg(cfg)
        self._build_actions(acts)

    def _build_home(self, f):
        self.banner = tk.Label(f, text="…", font=("TkDefaultFont", 18, "bold"), fg="white", pady=8)
        self.banner.pack(fill="x")
        ttk.Label(f, text=osinfo.describe(self.info, self.backend)).pack(anchor="w", pady=(6, 4))

        self.checks = ttk.Treeview(f, columns=("ok", "item", "detail"), show="headings", height=6)
        for c, t, w in (("ok", "", 40), ("item", "Elemento", 170), ("detail", "Detalle", 700)):
            self.checks.heading(c, text=t)
            self.checks.column(c, width=w, anchor="w" if c != "ok" else "center")
        self.checks.pack(fill="x")

        self.table = ttk.Treeview(
            f, columns=("g", "clips", "train", "val", "test", "vid", "st"), show="headings", height=6)
        for c, t, w in (("g", "Gesto", 160), ("clips", "Clips", 70), ("train", "Train", 70),
                        ("val", "Val", 70), ("test", "Test", 70), ("vid", "Videos pend.", 100),
                        ("st", "Estado", 380)):
            self.table.heading(c, text=t)
            self.table.column(c, width=w, anchor="w" if c in ("g", "st") else "center")
        self.table.pack(fill="x", pady=(8, 0))
        for t in (self.checks, self.table):
            t.tag_configure("ok", foreground=GREEN)
            t.tag_configure("bad", foreground=RED)
            t.tag_configure("warn", foreground=AMBER)

        row = ttk.LabelFrame(f, text="Datos y modelo", padding=6)
        row.pack(fill="x", pady=(8, 0))
        for txt, cmd in (("1 · Grabar datos", lambda: self.task("record.py", ask_stop=True)),
                         ("2 · Importar videos", lambda: self.task("import_videos.py")),
                         ("3 · Construir dataset", lambda: self.task("build_dataset.py")),
                         ("4 · Entrenar", lambda: self.task("train.py")),
                         ("5 · Testear", lambda: self.task("test.py")),
                         ("Pipeline 2→5", lambda: self.task("pipeline.py"))):
            b = ttk.Button(row, text=txt, command=cmd)
            b.pack(side="left", padx=3)
            self.task_buttons.append(b)
        ttk.Button(row, text="↻ Refrescar", command=self.manual_refresh).pack(side="right")

        d = ttk.LabelFrame(f, text="Demonio (fondo, como un live wallpaper)", padding=6)
        d.pack(fill="x", pady=(8, 0))
        self.btn_start = ttk.Button(d, text="▶ Iniciar demonio", command=self.start)
        self.btn_stop = ttk.Button(d, text="■ Detener demonio", command=self.stop)
        self.btn_start.pack(side="left", padx=3)
        self.btn_stop.pack(side="left", padx=3)
        ttk.Checkbutton(d, text="Dry-run (no ejecuta acciones)", variable=self.dry,
                        command=self.on_dry).pack(side="left", padx=12)
        ttk.Checkbutton(d, text="Ventana de cámara", variable=self.preview).pack(side="left")
        self.daemon_lbl = ttk.Label(f, text="", font=("TkDefaultFont", 10, "bold"))
        self.daemon_lbl.pack(anchor="w", pady=(6, 0))

        lf = ttk.Frame(f)
        lf.pack(fill="both", expand=True, pady=(6, 0))
        self.log = tk.Text(lf, height=8, wrap="word", state="disabled", font=("TkFixedFont", 9))
        sb = ttk.Scrollbar(lf, command=self.log.yview)
        self.log.configure(yscrollcommand=sb.set)
        self.log.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")

    def _build_plots(self, f):
        self.fig = Figure(figsize=(10, 4), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=f)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def _build_cfg(self, f):
        s = self.settings
        spec = [
            ("Cámara (índice)", ("camera", "index"), int), ("Espejo (true/false)", ("camera", "mirror"), bool),
            ("Backend (auto/pynput/ydotool)", ("system", "backend"), str),
            ("Umbral de confianza", ("inference", "threshold"), float),
            ("Umbral cerrar ventana", ("inference", "destructive_threshold"), float),
            ("Votos consecutivos", ("inference", "votes"), int),
            ("Cooldown (s)", ("inference", "cooldown"), float),
            ("Mantener Alt (s)", ("inference", "alt_hold_seconds"), float),
            ("Clips mínimos por gesto", ("dataset", "min_clips_per_gesture"), int),
            ("Épocas", ("training", "epochs"), int),
            ("Duración del clip (s) *", ("sequence", "clip_seconds"), float),
        ]
        self.cfg_vars = []
        for i, (label, path, typ) in enumerate(spec):
            ttk.Label(f, text=label).grid(row=i, column=0, sticky="w", pady=2, padx=(0, 12))
            v = tk.StringVar(value=str(s[path[0]][path[1]]))
            ttk.Entry(f, textvariable=v, width=18).grid(row=i, column=1, sticky="w")
            self.cfg_vars.append((path, typ, v))
        ttk.Button(f, text="Guardar", command=self.save_cfg).grid(row=len(spec), column=0, pady=12, sticky="w")
        ttk.Label(f, foreground=AMBER, wraplength=700, justify="left",
                  text="* Cambiar la duración del clip exige volver a grabar/importar y reentrenar. "
                       "Los cambios aplican al reiniciar el demonio.").grid(
            row=len(spec) + 1, column=0, columnspan=3, sticky="w")

    def _build_actions(self, f):
        ttk.Label(f, text="config/actions.yaml: gesto → acción y comandos por sistema").pack(anchor="w")
        self.actions_txt = tk.Text(f, wrap="none", font=("TkFixedFont", 10))
        self.actions_txt.pack(fill="both", expand=True, pady=6)
        self.actions_txt.insert("1.0", ACTIONS_FILE.read_text(encoding="utf-8"))
        ttk.Button(f, text="Guardar acciones", command=self.save_actions).pack(anchor="w")

    # ------------------------------------------------------------ estado
    def check_camera(self):
        self.cam = (False, "verificando…")
        threading.Thread(target=lambda: setattr(self, "cam", status.camera_check(self.settings)),
                         daemon=True).start()

    def manual_refresh(self):
        self.settings = load_settings()
        self.check_camera()
        self.after(1200, self.refresh)
        self.refresh()

    def refresh(self):
        checks, ready = status.readiness(self.settings, self.cam)
        self.checks.delete(*self.checks.get_children())
        for name, ok, detail, req in checks:
            tag = "ok" if ok else ("bad" if req else "warn")
            self.checks.insert("", "end", values=("✔" if ok else "✖", name, detail), tags=(tag,))
        minc = self.settings["dataset"]["min_clips_per_gesture"]
        self.table.delete(*self.table.get_children())
        for r in status.dataset_status(self.settings):
            problems = []
            if r["clips"] < minc:
                problems.append(f"faltan {minc - r['clips']} clips")
            if r["unsplit"]:
                problems.append("reconstruir dataset")
            elif r["clips"] and min(r["train"], r["val"], r["test"]) < 1:
                problems.append("falta train/val/test")
            self.table.insert("", "end", tags=("bad" if problems else "ok",), values=(
                r["gesture"], r["clips"], r["train"], r["val"], r["test"], r["pending_videos"],
                "✖ " + ", ".join(problems) if problems else "✔ listo"))
        if ready:
            self.banner.config(text="LISTO", bg=GREEN)
        else:
            falta = [n for n, ok, _, req in checks if req and not ok]
            self.banner.config(text="FALTA: " + ", ".join(falta), bg=RED)
        self.update_daemon()
        self.draw_plots()

    def update_daemon(self):
        st = status.read_daemon_status()
        running = st is not None
        self.btn_start.state(["disabled"] if running else ["!disabled"])
        self.btn_stop.state(["!disabled"] if running else ["disabled"])
        if not running:
            self.daemon_lbl.config(text="Demonio: detenido", foreground="gray")
            return
        txt = f"Demonio: {st.get('state')} (pid {st.get('pid')})"
        if st.get("state") == "running":
            txt += (f" · {'DRY-RUN' if st.get('dry_run') else 'ACCIONES REALES'} · {st.get('fps', 0)} fps"
                    f" · mano: {'sí' if st.get('hand') else 'no'}")
            if st.get("last_gesture"):
                txt += f" · último: {st['last_gesture']} ({st['last_conf']}) → {st['last_action']}"
        self.daemon_lbl.config(text=txt, foreground=GREEN)

    def tick(self):
        self.update_daemon()
        if not hasattr(self, "_n"):
            self._n = 0
        self._n += 1
        if self._n % 3 == 0 and not self.proc:
            self.refresh()
        self.after(1000, self.tick)

    def draw_plots(self):
        hist, res = status.read_json(HISTORY_FILE), status.read_json(TEST_RESULTS_FILE)
        fig = self.fig
        fig.clear()
        ax1, ax2, ax3 = fig.subplots(1, 3)
        for ax, keys, title in ((ax1, ("loss", "val_loss"), "Pérdida"),
                                (ax2, ("accuracy", "val_accuracy"), "Exactitud")):
            if hist:
                for k, lab in zip(keys, ("train", "val")):
                    ax.plot(hist.get(k, []), label=lab)
                ax.legend()
                ax.set_xlabel("época")
            else:
                ax.text(0.5, 0.5, "sin entrenamiento", ha="center", va="center")
                ax.set_axis_off()
            ax.set_title(title)
        if res:
            cm, labs = np.array(res["confusion"]), res["labels"]
            ax3.imshow(cm, cmap="Blues")
            ax3.set_xticks(range(len(labs)), labs, rotation=60, ha="right", fontsize=7)
            ax3.set_yticks(range(len(labs)), labs, fontsize=7)
            for i in range(len(labs)):
                for j in range(len(labs)):
                    ax3.text(j, i, cm[i, j], ha="center", va="center", fontsize=8,
                             color="white" if cm[i, j] > cm.max() / 2 else "black")
            ax3.set_title(f"Test · {res['accuracy']:.1%}")
        else:
            ax3.text(0.5, 0.5, "sin test", ha="center", va="center")
            ax3.set_axis_off()
        fig.tight_layout()
        self.canvas.draw_idle()

    # ------------------------------------------------------------ tareas
    def logw(self, text):
        self.log.config(state="normal")
        self.log.insert("end", text)
        self.log.see("end")
        self.log.config(state="disabled")

    def set_busy(self, busy):
        for b in self.task_buttons:
            b.state(["disabled"] if busy else ["!disabled"])

    def task(self, script, ask_stop=False):
        if self.proc and self.proc.poll() is None:
            messagebox.showinfo("Ocupado", "Ya hay una tarea en ejecución.")
            return
        if ask_stop and status.daemon_pid():
            if not messagebox.askyesno("Cámara en uso",
                                       "El demonio está usando la cámara. ¿Detenerlo para grabar?"):
                return
            status.stop_daemon()
        cmd = [sys.executable, "-u", str(SCRIPTS / script)]
        self.logw(f"\n$ python scripts/{script}\n")
        self.proc = subprocess.Popen(cmd, cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     text=True, bufsize=1, encoding="utf-8", errors="replace")
        self.set_busy(True)
        threading.Thread(target=self._reader, args=(self.proc,), daemon=True).start()

    def _reader(self, proc):
        for line in proc.stdout:
            self.q.put(("line", line))
        self.q.put(("done", proc.wait()))

    def pump(self):
        try:
            while True:
                kind, val = self.q.get_nowait()
                if kind == "line":
                    self.logw(val)
                else:
                    self.logw(f"[terminó con código {val}]\n")
                    self.proc = None
                    self.set_busy(False)
                    self.refresh()
        except queue.Empty:
            pass
        self.after(100, self.pump)

    # ------------------------------------------------------------ demonio
    def start(self):
        checks, _ = status.readiness(self.settings, self.cam)
        if not next(ok for n, ok, _, _ in checks if n == "Modelo"):
            messagebox.showerror("Falta el modelo", "Entrena el modelo antes de iniciar el demonio.")
            return
        if self.proc and self.proc.poll() is None:
            messagebox.showinfo("Cámara en uso", "Espera a que termine la tarea actual.")
            return
        pid = status.start_daemon(self.dry.get(), self.preview.get())
        self.logw(f"Demonio lanzado (pid {pid}). Puedes cerrar esta ventana: sigue en segundo plano.\n")
        self.after(1500, self.update_daemon)

    def stop(self):
        status.stop_daemon()
        self.logw("Demonio detenido.\n")
        self.update_daemon()

    def on_dry(self):
        if status.daemon_pid():
            status.write_command(dry_run=self.dry.get())

    # ------------------------------------------------------------ config
    def save_cfg(self):
        changes = {}
        try:
            for path, typ, var in self.cfg_vars:
                raw = var.get().strip()
                changes[path] = (raw.lower() in ("1", "true", "sí", "si", "yes")) if typ is bool else typ(raw)
        except ValueError as e:
            messagebox.showerror("Valor inválido", str(e))
            return
        update_settings(changes)
        s = load_settings()
        self.settings = s
        self.backend = resolve_backend(s, self.info)
        self.refresh()
        messagebox.showinfo("Guardado", "Configuración guardada. Reinicia el demonio para aplicarla.")

    def save_actions(self):
        text = self.actions_txt.get("1.0", "end")
        try:
            yaml.safe_load(text)
        except yaml.YAMLError as e:
            messagebox.showerror("YAML inválido", str(e))
            return
        ACTIONS_FILE.write_text(text, encoding="utf-8")
        messagebox.showinfo("Guardado", "Acciones guardadas. Reinicia el demonio para aplicarlas.")

    def on_close(self):
        if self.proc and self.proc.poll() is None:
            if not messagebox.askyesno("Tarea en curso", "Hay una tarea ejecutándose. ¿Cerrar y cancelarla?"):
                return
            self.proc.terminate()
        if status.daemon_pid():
            messagebox.showinfo("Demonio activo", "El demonio sigue corriendo en segundo plano.\n"
                                "Vuelve a abrir el panel para detenerlo.")
        self.destroy()


def main():
    App().mainloop()
