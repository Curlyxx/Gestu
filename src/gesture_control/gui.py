"""Panel de control (Tkinter + Matplotlib), tema rojo y negro.

Cerrar la ventana NO detiene el demonio.
Logo opcional: coloca una imagen en  assets/logo.png  (PNG o JPG, cuadrada, >= 300 px).
Si no existe, se dibuja el esqueleto de una mano.
"""
import queue
import subprocess
import sys
import threading
import tkinter as tk
import tkinter.font as tkfont
from tkinter import messagebox, ttk

import numpy as np
import yaml
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter

from . import osinfo, status
from .actions import resolve_backend
from .config import load_actions, load_settings, update_settings
from .hands import CONNECTIONS
from .paths import ACTIONS_FILE, HISTORY_FILE, ROOT, SCRIPTS, TEST_RESULTS_FILE, ensure_dirs

# ------------------------------------------------------------------ tema
BG, SURFACE, SURFACE2, BORDER = "#0b0b0d", "#141418", "#1c1c22", "#2b2b34"
RED, RED_HI, RED_LO = "#d90429", "#ff2e4d", "#7a0f1f"
TEXT, MUTED = "#f1f1f3", "#9b9ba8"
OK, WARN, BAD = "#3ddc84", "#ffb020", "#ff5a6a"
OK_BG, WARN_BG, BAD_BG = "#10291b", "#2e220a", "#2e1016"
PLOT_BG = "#0f0f13"

LOGO = ROOT / "assets" / "logo.png"
FAMILY = "TkDefaultFont"
MONO = "TkFixedFont"

GESTURE_INFO = {
    "swipe_derecha": "Swipe a la derecha",
    "swipe_izquierda": "Swipe a la izquierda",
    "swipe_arriba": "Swipe hacia arriba",
    "swipe_abajo": "Swipe hacia abajo",
    "agarrar": "Palma abierta, luego puño",
    "ninguno": "Movimiento normal",
}
ACTION_INFO = {
    "next_window": "Ventana siguiente", "prev_window": "Ventana anterior", "overview": "Vista general",
    "minimize": "Minimizar ventana", "close_window": "Cerrar ventana",
}

CFG_GROUPS = [
    ("Cámara", [
        ("Índice de cámara", "camera", "index", int, "0 es la cámara por defecto. Cámbialo si tienes varias."),
        ("Modo espejo", "camera", "mirror", bool,
         "Mover la mano a tu derecha cuenta como «derecha». Si lo cambias, hay que volver a grabar."),
    ]),
    ("Reconocimiento", [
        ("Confianza mínima", "inference", "threshold", float, "De 0 a 1. Súbela si hay falsos positivos."),
        ("Confianza para cerrar", "inference", "destructive_threshold", float,
         "Más alta que la anterior: cerrar una ventana no tiene vuelta atrás."),
        ("Votos seguidos", "inference", "votes", int, "Predicciones iguales consecutivas antes de actuar."),
        ("Pausa entre acciones (s)", "inference", "cooldown", float, "Evita que un mismo gesto dispare dos veces."),
        ("Mantener Alt (s)", "inference", "alt_hold_seconds", float, "Tiempo para encadenar swipes de Alt+Tab."),
    ]),
    ("Datos y entrenamiento", [
        ("Clips mínimos por gesto", "dataset", "min_clips_per_gesture", int, "Lo que el panel exige para dar «listo»."),
        ("Épocas máximas", "training", "epochs", int, "El entrenamiento se detiene antes si deja de mejorar."),
        ("Duración del clip (s)", "sequence", "clip_seconds", float, "Cambiarla exige volver a grabar y reentrenar."),
    ]),
    ("Sistema", [
        ("Backend de teclas", "system", "backend", str,
         "auto elige solo: pynput (Windows / X11) o ydotool (Wayland)."),
    ]),
]

HAND = {0: (.5, .95), 1: (.38, .82), 2: (.28, .68), 3: (.2, .55), 4: (.14, .43), 5: (.38, .55), 6: (.36, .38),
        7: (.35, .26), 8: (.34, .15), 9: (.5, .52), 10: (.5, .33), 11: (.5, .2), 12: (.5, .08), 13: (.62, .55),
        14: (.64, .38), 15: (.65, .27), 16: (.66, .17), 17: (.73, .62), 18: (.77, .48), 19: (.79, .39),
        20: (.81, .3)}


def font(size=10, weight="normal"):
    return (FAMILY, size, weight)


def draw_hand(cv, w, h, lw=3, r=3):
    """Esqueleto de mano (los mismos 21 puntos que detecta MediaPipe)."""
    pad = 0.1

    def pt(i):
        x, y = HAND[i]
        return w * (pad + x * (1 - 2 * pad)), h * (pad + y * (1 - 2 * pad))

    cv.create_oval(w * .04, h * .04, w * .96, h * .96, outline=RED_LO, width=1)
    for a, b in CONNECTIONS:
        cv.create_line(*pt(a), *pt(b), fill=RED_HI, width=lw, capstyle="round")
    for i in HAND:
        x, y = pt(i)
        cv.create_oval(x - r, y - r, x + r, y + r, fill=TEXT, outline=RED)


def draw_status(cv, kind):
    """Icono de estado dibujado (/ / •) sin depender de que la fuente tenga esos símbolos."""
    cv.delete("all")
    col = {"ok": OK, "bad": BAD, "warn": WARN}[kind]
    if kind == "ok":
        cv.create_line(6, 13, 11, 18, 19, 7, fill=col, width=3, capstyle="round", joinstyle="round")
    elif kind == "bad":
        cv.create_line(7, 7, 17, 17, fill=col, width=3, capstyle="round")
        cv.create_line(17, 7, 7, 17, fill=col, width=3, capstyle="round")
    else:
        cv.create_oval(8, 8, 16, 16, fill=col, outline=col)


def draw_gesture_glyph(cv, gesture, active):
    """Flecha / símbolo del gesto, dibujado con líneas."""
    cx, cy = 29, 25
    col = "white" if active else MUTED
    d = {"swipe_derecha": (1, 0), "swipe_izquierda": (-1, 0), "swipe_arriba": (0, -1), "swipe_abajo": (0, 1)}
    if gesture in d:
        dx, dy = d[gesture]
        cv.create_line(cx - dx * 14, cy - dy * 14, cx + dx * 14, cy + dy * 14, fill=col, width=3,
                       arrow="last", arrowshape=(11, 13, 5), capstyle="round")
    elif gesture == "agarrar":
        cv.create_oval(cx - 15, cy - 15, cx + 15, cy + 15, outline=col, width=2)
        cv.create_oval(cx - 6, cy - 6, cx + 6, cy + 6, fill=col, outline=col)
    else:
        cv.create_oval(cx - 8, cy - 8, cx + 8, cy + 8, outline=col, width=2)


def apply_theme(root):
    st = ttk.Style(root)
    st.theme_use("clam")
    st.configure(".", background=BG, foreground=TEXT, fieldbackground=SURFACE2, bordercolor=BORDER,
                 lightcolor=BORDER, darkcolor=BORDER, troughcolor=SURFACE, focuscolor=RED, font=font(10))
    st.configure("TNotebook", background=BG, bordercolor=BG, lightcolor=BG, darkcolor=BG, tabmargins=(20, 8, 20, 0))
    st.configure("TNotebook.Tab", background=SURFACE, foreground=MUTED, padding=(22, 10), borderwidth=0,
                 bordercolor=BG, lightcolor=SURFACE, darkcolor=SURFACE, font=font(10, "bold"))
    st.map("TNotebook.Tab", padding=[("selected", (22, 10)), ("!selected", (22, 10))],
           background=[("selected", RED), ("active", SURFACE2)],
           foreground=[("selected", "white"), ("active", TEXT)],
           lightcolor=[("selected", RED)], darkcolor=[("selected", RED)])
    for name, bg, fg, hover, bd in (("Accent", RED, "white", RED_HI, RED), ("Ghost", "#262630", TEXT, "#34343f", BORDER),
                                    ("Danger", SURFACE2, RED_HI, RED_LO, RED)):
        st.configure(f"{name}.TButton", background=bg, foreground=fg, bordercolor=bd, lightcolor=bg, darkcolor=bg,
                     focuscolor=bg, borderwidth=1, padding=(16, 8), font=font(10, "bold"), relief="flat")
        st.map(f"{name}.TButton",
               background=[("disabled", SURFACE2), ("pressed", RED_LO), ("active", hover)],
               foreground=[("disabled", MUTED), ("active", "white")],
               bordercolor=[("disabled", BORDER)],
               lightcolor=[("disabled", SURFACE2), ("active", hover)], darkcolor=[("disabled", SURFACE2), ("active", hover)])
    st.configure("TCheckbutton", background=SURFACE, foreground=TEXT, focuscolor=SURFACE, font=font(10))
    st.map("TCheckbutton", background=[("active", SURFACE)],
           indicatorbackground=[("selected", RED), ("!selected", SURFACE2)],
           indicatorforeground=[("selected", "white")],
           indicatorcolor=[("selected", RED), ("!selected", SURFACE2)],
           upperbordercolor=[("selected", RED_HI)], lowerbordercolor=[("selected", RED_HI)])
    st.configure("TEntry", fieldbackground=SURFACE2, foreground=TEXT, insertcolor=TEXT, bordercolor=BORDER,
                 lightcolor=BORDER, darkcolor=BORDER, padding=6)
    st.map("TEntry", bordercolor=[("focus", RED)], lightcolor=[("focus", RED)], darkcolor=[("focus", RED)])
    st.configure("TCombobox", fieldbackground=SURFACE2, background=SURFACE2, foreground=TEXT, arrowcolor=TEXT,
                 bordercolor=BORDER, lightcolor=BORDER, darkcolor=BORDER, padding=6)
    st.map("TCombobox", fieldbackground=[("readonly", SURFACE2)], foreground=[("readonly", TEXT)],
           bordercolor=[("focus", RED)], selectbackground=[("readonly", SURFACE2)],
           selectforeground=[("readonly", TEXT)])
    root.option_add("*TCombobox*Listbox.background", SURFACE2)
    root.option_add("*TCombobox*Listbox.foreground", TEXT)
    root.option_add("*TCombobox*Listbox.selectBackground", RED)
    st.configure("Vertical.TScrollbar", background=SURFACE2, troughcolor=BG, bordercolor=BG, arrowcolor=MUTED,
                 lightcolor=SURFACE2, darkcolor=SURFACE2, gripcount=0, arrowsize=14, relief="flat")
    st.map("Vertical.TScrollbar", background=[("pressed", RED), ("active", RED_LO)],
           lightcolor=[("pressed", RED), ("active", RED_LO)], darkcolor=[("pressed", RED), ("active", RED_LO)])
    for name, color in (("Red", RED), ("Ok", "#2fd17a")):
        st.configure(f"{name}.Horizontal.TProgressbar", troughcolor=SURFACE2, background=color, bordercolor=SURFACE2,
                     lightcolor=color, darkcolor=color, thickness=8)


# ------------------------------------------------------------------ widgets
class WrapLabel(tk.Label):
    """Label que ajusta el salto de línea al ancho disponible (usar con fill='x' / sticky='ew')."""

    def __init__(self, parent, text="", bg=SURFACE, fg=TEXT, size=10, weight="normal", **kw):
        super().__init__(parent, text=text, bg=bg, fg=fg, font=font(size, weight), justify="left", anchor="w",
                         wraplength=300, **kw)
        self._last_w = 0
        self.bind("<Configure>", self._rewrap)

    def _rewrap(self, e):
        if abs(e.width - self._last_w) > 3:
            self._last_w = e.width
            self.configure(wraplength=max(60, e.width - 2 * int(self.cget("padx"))))


def pill(parent, text="", bg=SURFACE2, fg=MUTED, size=9):
    return tk.Label(parent, text=text, bg=bg, fg=fg, font=font(size, "bold"), padx=10, pady=4)


class Card(tk.Frame):
    """Tarjeta con borde fino y título con acento rojo. El contenido va en .body"""

    def __init__(self, parent, title=None, subtitle=None, bg=SURFACE, pad=16):
        super().__init__(parent, bg=BORDER)
        self.bg = bg
        inner = tk.Frame(self, bg=bg)
        inner.pack(fill="both", expand=True, padx=1, pady=1)
        self.body = tk.Frame(inner, bg=bg)
        self.body.pack(fill="both", expand=True, padx=pad, pady=pad)
        if title:
            head = tk.Frame(self.body, bg=bg)
            head.pack(fill="x", pady=(0, 12))
            tk.Frame(head, bg=RED, width=4, height=18).pack(side="left", padx=(0, 10))
            tk.Label(head, text=title, bg=bg, fg=TEXT, font=font(12, "bold")).pack(side="left")
            if subtitle:
                tk.Label(head, text=subtitle, bg=bg, fg=MUTED, font=font(9)).pack(side="left", padx=(12, 0))
            self.head = head


class ScrollFrame(tk.Frame):
    """Marco con scroll vertical. Poner el contenido en .body"""

    def __init__(self, parent, bg=BG):
        super().__init__(parent, bg=bg)
        self.canvas = tk.Canvas(self, bg=bg, highlightthickness=0, bd=0, yscrollincrement=30)
        self.vsb = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.vsb.set)
        self.vsb.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)
        self.body = tk.Frame(self.canvas, bg=bg)
        self._win = self.canvas.create_window((0, 0), window=self.body, anchor="nw")
        self.body.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.bind("<Configure>", lambda e: self.canvas.itemconfigure(self._win, width=e.width))

    def scroll(self, direction):
        if self.body.winfo_reqheight() > self.canvas.winfo_height():
            self.canvas.yview_scroll(direction * 2, "units")


class FigureCard(Card):
    """Tarjeta con una figura de proporción fija (nunca se estira a lo alto) y texto explicativo."""

    def __init__(self, parent, title, subtitle, explain, aspect=0.6, side=False):
        super().__init__(parent, title, subtitle)
        self.aspect = aspect
        wrap = tk.Frame(self.body, bg=SURFACE)
        wrap.pack(fill="x")
        holder = tk.Frame(wrap, bg=SURFACE)
        self.fig = Figure(figsize=(5, 5 * aspect), dpi=100, facecolor=SURFACE, layout="constrained")
        self.canvas = FigureCanvasTkAgg(self.fig, master=holder)
        self.widget = self.canvas.get_tk_widget()
        self.widget.configure(bg=SURFACE, highlightthickness=0, height=int(480 * aspect))
        self.widget.pack(fill="x")
        holder.bind("<Configure>", self._resize)
        if side:
            wrap.columnconfigure(0, weight=3, uniform="f")
            wrap.columnconfigure(1, weight=2, uniform="f")
            holder.grid(row=0, column=0, sticky="nsew")
            self.side = tk.Frame(wrap, bg=SURFACE)
            self.side.grid(row=0, column=1, sticky="nsew", padx=(20, 0))
            text_parent = self.side
        else:
            holder.pack(fill="x")
            text_parent = self.body
        tk.Label(text_parent, text="CÓMO LEERLA", bg=SURFACE, fg=RED_HI, font=font(8, "bold"),
                 anchor="w").pack(fill="x", pady=(10, 2))
        WrapLabel(text_parent, explain, fg=MUTED, size=10).pack(fill="x")
        self.extra = tk.Frame(text_parent, bg=SURFACE)
        self.extra.pack(fill="x")
        tk.Label(text_parent, text="LECTURA", bg=SURFACE, fg=RED_HI, font=font(8, "bold"),
                 anchor="w").pack(fill="x", pady=(12, 2))
        self.reading = WrapLabel(text_parent, "—", fg=TEXT, size=10)
        self.reading.pack(fill="x")

    def _resize(self, e):
        h = max(140, int(e.width * self.aspect))
        if abs(int(self.widget.cget("height")) - h) > 2:
            self.widget.configure(height=h)


def stat_tile(parent, caption):
    box = tk.Frame(parent, bg=SURFACE2, highlightbackground=BORDER, highlightthickness=1)
    value = tk.Label(box, text="—", bg=SURFACE2, fg=TEXT, font=font(22, "bold"))
    value.pack(anchor="w", padx=14, pady=(10, 0))
    tk.Label(box, text=caption, bg=SURFACE2, fg=MUTED, font=font(9)).pack(anchor="w", padx=14, pady=(0, 10))
    return box, value


# ------------------------------------------------------------------ gráficas
CM_GOOD = LinearSegmentedColormap.from_list("good", ["#16161b", "#14532d", "#2fd17a"])
CM_BAD = LinearSegmentedColormap.from_list("bad", ["#16161b", "#7a0f1f", "#ff2e4d"])
pct = FuncFormatter(lambda v, _: f"{v:.0%}")


def style_axes(ax):
    ax.set_facecolor(PLOT_BG)
    for sp in ax.spines.values():
        sp.set_color(BORDER)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(True, color="#23232b", lw=0.8)
    ax.set_axisbelow(True)


def axis_labels(ax, x=None, y=None):
    if x:
        ax.set_xlabel(x, color=MUTED, fontsize=8)
    if y:
        ax.set_ylabel(y, color=MUTED, fontsize=8)


def empty_plot(fig, msg):
    fig.clear()
    fig.patch.set_facecolor(SURFACE)
    fig.text(0.5, 0.5, msg, ha="center", va="center", color=MUTED, fontsize=10)


def short(label):
    return label.replace("swipe_", "swipe ")


# ------------------------------------------------------------------ app
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        global FAMILY, MONO
        FAMILY = tkfont.nametofont("TkDefaultFont").actual("family")
        MONO = tkfont.nametofont("TkFixedFont").actual("family")
        self.title("Gesture Control")
        self.geometry("1140x840")
        self.minsize(980, 680)
        self.configure(bg=BG)
        apply_theme(self)
        self.settings = load_settings()
        ensure_dirs(self.settings["gestures"])
        self.info = osinfo.detect()
        self.backend = resolve_backend(self.settings, self.info)
        self.cam = None
        self.proc = None
        self.q = queue.Queue()
        self.task_buttons = []
        self.check_rows = {}
        self.table_rows = {}
        self._table_gestures = None
        self._plot_sig = None
        self._model_ok = False
        self._n = 0
        self.dry = tk.BooleanVar(value=True)
        self.preview = tk.BooleanVar(value=False)
        self._build()
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        for seq in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            self.bind_all(seq, self._wheel)
        self.check_camera()
        self.refresh()
        self.after(100, self.pump)
        self.after(1000, self.tick)

    # ------------------------------------------------------------ estructura
    def _build(self):
        self._build_header()
        self._build_footer()
        self.nb = ttk.Notebook(self)
        self.nb.pack(fill="both", expand=True)
        self.tabs = {}
        for key, label, builder in (("welcome", "Bienvenida", self._build_welcome), ("panel", "Panel", self._build_panel),
                                    ("plots", "Gráficas", self._build_plots),
                                    ("cfg", "Configuración", self._build_cfg), ("acts", "Acciones", self._build_actions)):
            sf = ScrollFrame(self.nb)
            self.nb.add(sf, text=label)
            self.tabs[key] = sf
            wrap = tk.Frame(sf.body, bg=BG)
            wrap.pack(fill="both", expand=True, padx=24, pady=20)
            builder(wrap)
        self.nb.bind("<<NotebookTabChanged>>", self._on_tab)

    def goto(self, key):
        self.nb.select(self.tabs[key])

    def _on_tab(self, _e=None):
        if self.nb.select() == str(self.tabs["plots"]):
            self.draw_plots()

    def _build_header(self):
        bar = tk.Frame(self, bg=BG)
        bar.pack(fill="x", padx=22, pady=(14, 12))
        cv = tk.Canvas(bar, width=48, height=48, bg=BG, highlightthickness=0)
        cv.pack(side="left")
        draw_hand(cv, 48, 48, lw=2, r=2)
        tit = tk.Frame(bar, bg=BG)
        tit.pack(side="left", padx=14)
        row = tk.Frame(tit, bg=BG)
        row.pack(anchor="w")
        tk.Label(row, text="GESTURE ", bg=BG, fg=TEXT, font=font(17, "bold")).pack(side="left")
        tk.Label(row, text="CONTROL", bg=BG, fg=RED_HI, font=font(17, "bold")).pack(side="left")
        tk.Label(tit, text="Control de ventanas con gestos de una mano", bg=BG, fg=MUTED, font=font(9)).pack(anchor="w")
        right = tk.Frame(bar, bg=BG)
        right.pack(side="right")
        self.pill_ready = pill(right, "…", size=10)
        self.pill_ready.pack(side="left", padx=4)
        self.pill_daemon = pill(right, "Demonio detenido", size=10)
        self.pill_daemon.pack(side="left", padx=4)
        tk.Frame(self, bg=RED, height=2).pack(fill="x")

    def _build_footer(self):
        foot = tk.Frame(self, bg=SURFACE)
        foot.pack(side="bottom", fill="x")
        text = osinfo.describe(self.info, self.backend).replace("\n", "   ")
        tk.Label(foot, text="Cerrar esta ventana no detiene el demonio   ·   " + text, bg=SURFACE,
                 fg=WARN if "Aviso" in text else MUTED, font=font(8), anchor="w").pack(fill="x", padx=22, pady=6)

    def _wheel(self, e):
        w = e.widget
        if isinstance(w, str):
            try:
                w = self.nametowidget(w)
            except KeyError:
                return
        if w.winfo_class() in ("Text", "Listbox", "TCombobox"):
            return
        while w is not None and not isinstance(w, ScrollFrame):
            w = w.master
        if w is None:
            return
        if e.num == 4:
            d = -1
        elif e.num == 5:
            d = 1
        else:
            d = -1 if e.delta > 0 else 1
        w.scroll(d)

    # ------------------------------------------------------------ bienvenida
    def _logo(self, parent):
        if LOGO.exists():
            try:
                from PIL import Image, ImageTk
                im = Image.open(LOGO).convert("RGBA")
                im.thumbnail((230, 230), getattr(Image, "Resampling", Image).LANCZOS)
                self._logo_img = ImageTk.PhotoImage(im)
                tk.Label(parent, image=self._logo_img, bg=SURFACE).pack()
                return
            except Exception:  # noqa: BLE001
                pass
        cv = tk.Canvas(parent, width=200, height=220, bg=SURFACE, highlightthickness=0)
        cv.pack()
        draw_hand(cv, 200, 220, lw=4, r=4)

    def _build_welcome(self, wrap):
        hero = Card(wrap, pad=26)
        hero.pack(fill="x")
        left = tk.Frame(hero.body, bg=SURFACE)
        left.pack(side="left", padx=(0, 30))
        self._logo(left)
        right = tk.Frame(hero.body, bg=SURFACE)
        right.pack(side="left", fill="both", expand=True)
        tk.Label(right, text="Controla tus ventanas\ncon un gesto de la mano", bg=SURFACE, fg=TEXT,
                 font=font(26, "bold"), justify="left", anchor="w").pack(anchor="w")
        WrapLabel(right, "Mueve la mano frente a la cámara: cambia de ventana, minimiza o cierra sin tocar el "
                         "teclado. Funciona en segundo plano, como un fondo animado: este panel solo lo abres "
                         "para configurarlo.", fg=MUTED, size=11).pack(fill="x", pady=(12, 18))
        btns = tk.Frame(right, bg=SURFACE)
        btns.pack(anchor="w")
        ttk.Button(btns, text="Ir al panel", style="Accent.TButton", command=lambda: self.goto("panel")).pack(side="left")
        ttk.Button(btns, text="Ver gráficas", style="Ghost.TButton", command=lambda: self.goto("plots")).pack(side="left", padx=10)
        self.welcome_state = tk.Label(right, text="", bg=SURFACE, fg=MUTED, font=font(10), anchor="w")
        self.welcome_state.pack(anchor="w", pady=(16, 0))

        flow = Card(wrap, "Cómo funciona", "de la cámara a la acción")
        flow.pack(fill="x", pady=(16, 0))
        steps = [("1", "Cámara", "Tu webcam, siempre abierta en segundo plano."),
                 ("2", "MediaPipe", "Detecta 21 puntos de una mano."),
                 ("3", "Red GRU", "TensorFlow analiza ~1.5 s de movimiento."),
                 ("4", "Gesto", "Solo cuenta si la confianza es alta."),
                 ("5", "Acción", "Alt+Tab, minimizar, cerrar…")]
        g = tk.Frame(flow.body, bg=SURFACE)
        g.pack(fill="x")
        for i, (num, title, desc) in enumerate(steps):
            col = i * 2
            g.columnconfigure(col, weight=1, uniform="s")
            tile = tk.Frame(g, bg=SURFACE2, highlightbackground=BORDER, highlightthickness=1)
            tile.grid(row=0, column=col, sticky="nsew")
            top = tk.Frame(tile, bg=SURFACE2)
            top.pack(fill="x", padx=12, pady=(12, 4))
            tk.Label(top, text=num, bg=RED, fg="white", font=font(9, "bold"), width=2).pack(side="left")
            tk.Label(top, text=title, bg=SURFACE2, fg=TEXT, font=font(11, "bold")).pack(side="left", padx=8)
            WrapLabel(tile, desc, bg=SURFACE2, fg=MUTED, size=9).pack(fill="x", padx=12, pady=(0, 12))
            if i < len(steps) - 1:
                tk.Label(g, text="›", bg=SURFACE, fg=RED_HI, font=font(20, "bold")).grid(row=0, column=col + 1, padx=6)

        info = tk.Frame(wrap, bg=BG)
        info.pack(fill="x", pady=(16, 0))
        for c in range(3):
            info.columnconfigure(c, weight=1, uniform="i")
        c1 = Card(info, "El modelo")
        c1.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        tk.Label(c1.body, text="GRU", bg=SURFACE, fg=RED_HI, font=font(30, "bold")).pack(anchor="w")
        tk.Label(c1.body, text="Gated Recurrent Unit", bg=SURFACE, fg=TEXT, font=font(10, "bold")).pack(anchor="w")
        WrapLabel(c1.body, "Sí, es una red recurrente (RNN): lee el movimiento cuadro a cuadro y recuerda lo "
                           "anterior. Dos capas Conv1D detectan patrones cortos y la GRU entiende el orden. "
                           "TensorFlow / Keras, ~64 mil parámetros: corre en CPU.", fg=MUTED, size=10).pack(fill="x", pady=(8, 0))
        c2 = Card(info, "Qué ve")
        c2.grid(row=0, column=1, sticky="nsew", padx=8)
        tk.Label(c2.body, text="21", bg=SURFACE, fg=RED_HI, font=font(30, "bold")).pack(anchor="w")
        tk.Label(c2.body, text="puntos de una mano", bg=SURFACE, fg=TEXT, font=font(10, "bold")).pack(anchor="w")
        WrapLabel(c2.body, "Solo coordenadas, no la imagen. El demonio no guarda fotos ni video: procesa "
                           "todo en memoria y lo descarta.", fg=MUTED, size=10).pack(fill="x", pady=(8, 0))
        c3 = Card(info, "Cuándo actúa")
        c3.grid(row=0, column=2, sticky="nsew", padx=(8, 0))
        self.w_conf = tk.Label(c3.body, text="85%", bg=SURFACE, fg=RED_HI, font=font(30, "bold"))
        self.w_conf.pack(anchor="w")
        tk.Label(c3.body, text="confianza mínima", bg=SURFACE, fg=TEXT, font=font(10, "bold")).pack(anchor="w")
        self.w_conf_txt = WrapLabel(c3.body, "", fg=MUTED, size=10)
        self.w_conf_txt.pack(fill="x", pady=(8, 0))

        self.chips_card = Card(wrap, "Gestos incluidos", "cada gesto con su acción (editable en Acciones)")
        self.chips_card.pack(fill="x", pady=(16, 0))
        self.build_chips()

    def build_chips(self):
        body = self.chips_card.body
        for w in list(body.winfo_children())[1:]:
            w.destroy()
        try:
            mapping = (load_actions().get("gestures") or {})
        except Exception:  # noqa: BLE001
            mapping = {}
        grid = tk.Frame(body, bg=SURFACE)
        grid.pack(fill="x")
        for c in range(3):
            grid.columnconfigure(c, weight=1, uniform="c")
        for i, g in enumerate(self.settings["gestures"]):
            name = GESTURE_INFO.get(g, g)
            act = mapping.get(g)
            text = ACTION_INFO.get(act, act) if act else "No hace nada"
            chip = tk.Frame(grid, bg=SURFACE2, highlightbackground=BORDER, highlightthickness=1)
            chip.grid(row=i // 3, column=i % 3, sticky="nsew", padx=4, pady=4)
            cv = tk.Canvas(chip, width=58, height=50, bg=RED if act else SURFACE, highlightthickness=0)
            cv.pack(side="left", fill="y")
            draw_gesture_glyph(cv, g, bool(act))
            col = tk.Frame(chip, bg=SURFACE2)
            col.pack(side="left", fill="x", expand=True, padx=10, pady=8)
            tk.Label(col, text=name, bg=SURFACE2, fg=TEXT, font=font(10, "bold"), anchor="w").pack(fill="x")
            tk.Label(col, text=text, bg=SURFACE2, fg=MUTED if act else MUTED, font=font(9), anchor="w").pack(fill="x")

    def update_welcome(self):
        inf = self.settings["inference"]
        self.w_conf.config(text=f"{inf['threshold']:.0%}")
        self.w_conf_txt.config(text=f"{inf['destructive_threshold']:.0%} para cerrar una ventana, {inf['votes']} "
                                    f"detecciones seguidas y {inf['cooldown']:g} s de pausa entre acciones.")
        running = status.daemon_pid() is not None
        self.welcome_state.config(
            text=f"{'Demonio activo' if running else 'Demonio detenido'}   ·   "
                 f"Modelo {'entrenado' if self._model_ok else 'sin entrenar'}",
            fg=OK if running else MUTED)

    # ------------------------------------------------------------ panel
    def _build_panel(self, wrap):
        top = tk.Frame(wrap, bg=BG)
        top.pack(fill="x")
        top.columnconfigure(0, weight=1, uniform="p")
        top.columnconfigure(1, weight=1, uniform="p")

        est = Card(top, "Estado", "qué falta para estar listo")
        est.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        self.check_frame = tk.Frame(est.body, bg=SURFACE)
        self.check_frame.pack(fill="x")
        self.check_frame.columnconfigure(2, weight=1)
        ttk.Button(est.body, text="Refrescar", style="Ghost.TButton", command=self.manual_refresh).pack(anchor="w", pady=(12, 0))

        d = Card(top, "Demonio", "se queda en segundo plano")
        d.grid(row=0, column=1, sticky="nsew", padx=(8, 0))
        self.d_state = tk.Label(d.body, text="DETENIDO", bg=SURFACE, fg=MUTED, font=font(20, "bold"), anchor="w")
        self.d_state.pack(fill="x")
        self.d_sub = WrapLabel(d.body, "", fg=MUTED, size=9)
        self.d_sub.pack(fill="x", pady=(0, 10))
        pills = tk.Frame(d.body, bg=SURFACE)
        pills.pack(fill="x")
        self.p_mode, self.p_fps, self.p_hand = pill(pills, "—"), pill(pills, "—"), pill(pills, "—")
        for p in (self.p_mode, self.p_fps, self.p_hand):
            p.pack(side="left", padx=(0, 6))
        tk.Label(d.body, text="ÚLTIMO GESTO", bg=SURFACE, fg=RED_HI, font=font(8, "bold"), anchor="w").pack(fill="x", pady=(14, 2))
        self.d_last = WrapLabel(d.body, "—", fg=TEXT, size=10)
        self.d_last.pack(fill="x")
        row = tk.Frame(d.body, bg=SURFACE)
        row.pack(fill="x", pady=(14, 0))
        self.btn_start = ttk.Button(row, text="Iniciar", style="Accent.TButton", command=self.start)
        self.btn_stop = ttk.Button(row, text="Detener", style="Danger.TButton", command=self.stop)
        self.btn_start.pack(side="left")
        self.btn_stop.pack(side="left", padx=8)
        ttk.Checkbutton(d.body, text="Dry-run: detecta y muestra la acción, pero no la ejecuta", variable=self.dry,
                        command=self.on_dry).pack(anchor="w", pady=(12, 2))
        ttk.Checkbutton(d.body, text="Mostrar ventana de la cámara", variable=self.preview).pack(anchor="w")

        data = Card(wrap, "Gestos y datos", "cuántos clips hay y si alcanzan")
        data.pack(fill="x", pady=(16, 0))
        self.table_frame = tk.Frame(data.body, bg=SURFACE)
        self.table_frame.pack(fill="x")
        self.table_note = WrapLabel(data.body, "", fg=MUTED, size=9)
        self.table_note.pack(fill="x", pady=(10, 0))

        flow = Card(wrap, "Flujo de trabajo", "sigue el orden: 1 a 5")
        flow.pack(fill="x", pady=(16, 0))
        grid = tk.Frame(flow.body, bg=SURFACE)
        grid.pack(fill="x")
        steps = [("1", "Grabar", "Graba clips con la cámara. Teclas 1-6 y ESPACIO.", "record.py", True),
                 ("2", "Importar videos", "Convierte los videos de data/raw_videos en clips.", "import_videos.py", False),
                 ("3", "Construir dataset", "Reparte los clips en train, val y test.", "build_dataset.py", False),
                 ("4", "Entrenar", "Entrena la red GRU con los datos de train.", "train.py", False),
                 ("5", "Testear", "Mide la exactitud con clips que nunca vio.", "test.py", False),
                 ("2-5", "Pipeline 2 a 5", "Hace los pasos 2 a 5 de una vez.", "pipeline.py", False)]
        for c in range(3):
            grid.columnconfigure(c, weight=1, uniform="t")
        for i, (num, title, desc, script, ask) in enumerate(steps):
            tile = tk.Frame(grid, bg=SURFACE2, highlightbackground=BORDER, highlightthickness=1)
            tile.grid(row=i // 3, column=i % 3, sticky="nsew", padx=4, pady=4)
            head = tk.Frame(tile, bg=SURFACE2)
            head.pack(fill="x", padx=14, pady=(12, 4))
            tk.Label(head, text=num, bg=RED, fg="white", font=font(10, "bold"), width=3).pack(side="left")
            tk.Label(head, text=title, bg=SURFACE2, fg=TEXT, font=font(11, "bold")).pack(side="left", padx=10)
            WrapLabel(tile, desc, bg=SURFACE2, fg=MUTED, size=9).pack(fill="x", padx=14)
            b = ttk.Button(tile, text="Ejecutar", style="Accent.TButton" if script == "pipeline.py" else "Ghost.TButton",
                           command=lambda s=script, a=ask: self.task(s, ask_stop=a))
            b.pack(anchor="w", padx=14, pady=(10, 14))
            self.task_buttons.append(b)

        logc = Card(wrap, "Registro", "salida de las tareas")
        logc.pack(fill="x", pady=(16, 0))
        bar = tk.Frame(logc.body, bg=SURFACE)
        bar.pack(fill="x", pady=(0, 8))
        self.task_lbl = tk.Label(bar, text="Sin tareas en ejecución", bg=SURFACE, fg=MUTED, font=font(9), anchor="w")
        self.task_lbl.pack(side="left")
        self.task_pb = ttk.Progressbar(bar, mode="indeterminate", style="Red.Horizontal.TProgressbar", length=180)
        lf = tk.Frame(logc.body, bg=SURFACE)
        lf.pack(fill="x")
        self.log = tk.Text(lf, height=11, wrap="word", state="disabled", bg=PLOT_BG, fg=TEXT, insertbackground=TEXT,
                           relief="flat", font=(MONO, 9), padx=10, pady=8, highlightthickness=1,
                           highlightbackground=BORDER, selectbackground=RED_LO)
        sb = ttk.Scrollbar(lf, command=self.log.yview)
        self.log.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self.log.pack(side="left", fill="x", expand=True)
        self.log.tag_configure("cmd", foreground=RED_HI)
        self.log.tag_configure("ok", foreground=OK)
        self.log.tag_configure("err", foreground=BAD)

    def _check_row(self, name):
        if name not in self.check_rows:
            r = len(self.check_rows)
            icon = tk.Canvas(self.check_frame, width=24, height=24, bg=SURFACE, highlightthickness=0)
            icon.grid(row=r, column=0, sticky="n", padx=(0, 6), pady=5)
            tk.Label(self.check_frame, text=name, bg=SURFACE, fg=TEXT, font=font(10, "bold"), anchor="w",
                     width=18).grid(row=r, column=1, sticky="nw", pady=6)
            det = WrapLabel(self.check_frame, "", fg=MUTED, size=9)
            det.grid(row=r, column=2, sticky="ew", pady=6)
            self.check_rows[name] = (icon, det)
        return self.check_rows[name]

    def _build_table(self):
        t = self.table_frame
        for w in t.winfo_children():
            w.destroy()
        heads = ("Gesto", "Clips / mínimo", "Train", "Val", "Test", "Videos", "Estado")
        for c, h in enumerate(heads):
            tk.Label(t, text=h.upper(), bg=SURFACE, fg=MUTED, font=font(8, "bold"),
                     anchor="w" if c in (0, 1, 6) else "center").grid(row=0, column=c, sticky="ew", padx=6, pady=(0, 8))
        t.columnconfigure(1, weight=2)
        t.columnconfigure(6, weight=2)
        self.table_rows = {}
        for r, g in enumerate(self.settings["gestures"], start=1):
            name = tk.Label(t, text=g, bg=SURFACE, fg=TEXT, font=font(10, "bold"), anchor="w")
            name.grid(row=r, column=0, sticky="ew", padx=6, pady=5)
            cell = tk.Frame(t, bg=SURFACE)
            cell.grid(row=r, column=1, sticky="ew", padx=6)
            pb = ttk.Progressbar(cell, style="Red.Horizontal.TProgressbar", maximum=100)
            pb.pack(side="left", fill="x", expand=True)
            cnt = tk.Label(cell, text="", bg=SURFACE, fg=TEXT, font=font(9), width=8, anchor="e")
            cnt.pack(side="left")
            nums = []
            for c in (2, 3, 4, 5):
                lab = tk.Label(t, text="", bg=SURFACE, fg=TEXT, font=font(10), width=6)
                lab.grid(row=r, column=c, sticky="ew", padx=2)
                nums.append(lab)
            st = tk.Label(t, text="", bg=SURFACE, fg=MUTED, font=font(9, "bold"), anchor="w")
            st.grid(row=r, column=6, sticky="ew", padx=6)
            self.table_rows[g] = (pb, cnt, nums, st)
        self._table_gestures = list(self.settings["gestures"])

    # ------------------------------------------------------------ gráficas
    def _build_plots(self, wrap):
        head = tk.Frame(wrap, bg=BG)
        head.pack(fill="x")
        tk.Label(head, text="Resultados del modelo", bg=BG, fg=TEXT, font=font(16, "bold")).pack(side="left")
        ttk.Button(head, text="Actualizar", style="Ghost.TButton",
                   command=lambda: self.draw_plots(force=True)).pack(side="right")

        tiles = tk.Frame(wrap, bg=BG)
        tiles.pack(fill="x", pady=(14, 0))
        self.tile_vals = []
        for c, cap in enumerate(("Exactitud en test", "Clips de test", "Mejor validación", "Épocas entrenadas")):
            tiles.columnconfigure(c, weight=1, uniform="s")
            box, val = stat_tile(tiles, cap)
            box.grid(row=0, column=c, sticky="nsew", padx=(0 if c == 0 else 6, 0 if c == 3 else 6))
            self.tile_vals.append(val)
        self.verdict = tk.Label(wrap, text="", bg=SURFACE, fg=TEXT, font=font(11, "bold"), anchor="w", padx=16, pady=12,
                                justify="left", highlightbackground=BORDER, highlightthickness=1)
        self.verdict.pack(fill="x", pady=(12, 0))

        row = tk.Frame(wrap, bg=BG)
        row.pack(fill="x", pady=(16, 0))
        row.columnconfigure(0, weight=1, uniform="g")
        row.columnconfigure(1, weight=1, uniform="g")
        self.fc_loss = FigureCard(row, "Error durante el entrenamiento", "pérdida (loss)",
                                  "Mide qué tan equivocado está el modelo: más bajo es mejor. Si las dos líneas bajan "
                                  "y se juntan, aprende bien. Si la roja sube mientras la gris baja, está memorizando.")
        self.fc_loss.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        self.fc_acc = FigureCard(row, "Aciertos durante el entrenamiento", "exactitud (accuracy)",
                                 "Porcentaje de clips bien clasificados. «Validación» son clips que el modelo no usó "
                                 "para aprender: es la medida honesta. La meta razonable es 90 % o más.")
        self.fc_acc.grid(row=0, column=1, sticky="nsew", padx=(8, 0))

        self.fc_cm = FigureCard(
            wrap, "Matriz de confusión", "qué gesto hiciste vs. qué entendió el modelo",
            "Cada fila es el gesto que hiciste y cada columna lo que el modelo entendió. Una diagonal verde "
            "significa que acertó. Cualquier cuadro rojo fuera de la diagonal es una confusión.",
            aspect=0.82, side=True)
        self.fc_cm.pack(fill="x", pady=(16, 0))
        for color, text in ((OK, "Verde: acertó"), (RED_HI, "Rojo: se confundió"), (MUTED, "Vacío: sin casos")):
            lg = tk.Frame(self.fc_cm.extra, bg=SURFACE)
            lg.pack(anchor="w", pady=2)
            tk.Frame(lg, bg=color, width=12, height=12).pack(side="left", padx=(0, 8))
            tk.Label(lg, text=text, bg=SURFACE, fg=TEXT, font=font(9)).pack(side="left")

        self.fc_pc = FigureCard(
            wrap, "Detección y precisión por gesto", "cuánto confiar en cada uno",
            "Detección: de las veces que hiciste el gesto, cuántas lo reconoció. Precisión: cuando el modelo dice "
            "«este gesto», cuántas veces tiene razón; una precisión baja significa acciones que se disparan sin querer.",
            aspect=0.42)
        self.fc_pc.pack(fill="x", pady=(16, 0))

    def draw_plots(self, force=False):
        sig = tuple(p.stat().st_mtime if p.exists() else 0 for p in (HISTORY_FILE, TEST_RESULTS_FILE))
        if sig == self._plot_sig and not force:
            return
        self._plot_sig = sig
        hist, res = status.read_json(HISTORY_FILE), status.read_json(TEST_RESULTS_FILE)
        self._draw_curve(self.fc_loss, hist, ("loss", "val_loss"), "Error (loss)", False)
        self._draw_curve(self.fc_acc, hist, ("accuracy", "val_accuracy"), "Exactitud", True)
        self._draw_cm(res)
        self._draw_per_class(res)
        self._update_stats(hist, res)
        for fc in (self.fc_loss, self.fc_acc, self.fc_cm, self.fc_pc):
            fc.canvas.draw_idle()

    def _draw_curve(self, fc, hist, keys, ylabel, is_acc):
        fig = fc.fig
        if not hist or not hist.get(keys[0]):
            empty_plot(fig, "Sin entrenamiento todavía\nPanel, paso 4: Entrenar")
            fc.reading.config(text="Aún no hay un entrenamiento guardado.")
            return
        fig.clear()
        ax = fig.add_subplot(111)
        style_axes(ax)
        tr, va = np.array(hist[keys[0]]), np.array(hist[keys[1]])
        xs = np.arange(1, len(tr) + 1)
        ax.plot(xs, tr, color="#c9c9d3", lw=1.8, label="Entrenamiento")
        ax.plot(xs, va, color=RED_HI, lw=2.6, label="Validación")
        b = int(np.argmax(va) if is_acc else np.argmin(va))
        ax.scatter([xs[b]], [va[b]], s=70, color=RED_HI, edgecolor="white", zorder=5)
        ax.annotate(f"mejor: época {xs[b]}", (xs[b], va[b]), xytext=(-90, 40 if not is_acc else -40),
                    textcoords="offset points", color=TEXT, fontsize=8,
                    arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
        if is_acc:
            ax.axhline(0.9, color=MUTED, lw=1, ls="--")
            ax.text(xs[0], 0.9, "meta 90 %", color=MUTED, fontsize=8, va="bottom", ha="left")
            ax.yaxis.set_major_formatter(pct)
            ax.set_ylim(max(0, min(tr.min(), va.min()) - 0.05), 1.03)
        axis_labels(ax, "Época (una vuelta completa a los datos)", ylabel)
        ax.legend(frameon=False, labelcolor=TEXT, fontsize=8, loc="center right")
        if is_acc:
            ok = va[b] >= 0.9
            fc.reading.config(text=f"Mejor validación: {va[b]:.1%} (época {xs[b]}). "
                                   + ("Supera la meta del 90 %." if ok else
                                      "Aún por debajo del 90 %: más datos o más variedad ayudan."),
                              fg=OK if ok else WARN)
        else:
            acc_t = np.array(hist.get("accuracy", [0] * len(tr)))[b]
            acc_v = np.array(hist.get("val_accuracy", [0] * len(tr)))[b]
            gap = acc_t - acc_v
            if gap > 0.10:
                fc.reading.config(text=f"Entrenó {len(tr)} épocas. Hay {gap:.0%} de diferencia entre entrenamiento y "
                                       "validación: tiende a memorizar. Graba más variedad (sesiones, luz, distancia).", fg=WARN)
            else:
                fc.reading.config(text=f"Entrenó {len(tr)} épocas. Entrenamiento y validación van juntos: está "
                                       "aprendiendo, no memorizando.", fg=OK)

    def _draw_cm(self, res):
        fc = self.fc_cm
        fig = fc.fig
        if not res:
            empty_plot(fig, "Sin test todavía\nPanel, paso 5: Testear")
            fc.reading.config(text="Aún no hay resultados de test.", fg=TEXT)
            return
        labels, cm = res["labels"], np.array(res["confusion"])
        n = len(labels)
        rows = cm.sum(axis=1, keepdims=True)
        frac = np.divide(cm, rows, out=np.zeros(cm.shape, float), where=rows > 0)
        diag = np.eye(n, dtype=bool)
        fig.clear()
        ax = fig.add_subplot(111)
        ax.set_facecolor(PLOT_BG)
        good = np.ma.masked_where(~diag, frac)
        bad_vals = np.where(cm > 0, 0.35 + 0.65 * np.minimum(frac * 2.5, 1), 0.0)
        bad = np.ma.masked_where(diag, bad_vals)
        ax.imshow(good, cmap=CM_GOOD, vmin=0, vmax=1, aspect="auto")
        ax.imshow(bad, cmap=CM_BAD, vmin=0, vmax=1, aspect="auto")
        for i in range(n):
            for j in range(n):
                if i == j:
                    ax.text(j, i, f"{cm[i, j]}\n{frac[i, j]:.0%}", ha="center", va="center", color="white",
                            fontsize=9, fontweight="bold")
                elif cm[i, j] > 0:
                    ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="white", fontsize=10, fontweight="bold")
        ax.set_xticks(range(n), [short(l) for l in labels], rotation=30, ha="right", fontsize=8, color=MUTED)
        ax.set_yticks(range(n), [short(l) for l in labels], fontsize=8, color=MUTED)
        ax.set_xticks(np.arange(-.5, n, 1), minor=True)
        ax.set_yticks(np.arange(-.5, n, 1), minor=True)
        ax.grid(which="minor", color=SURFACE, lw=3)
        ax.grid(which="major", visible=False)
        ax.tick_params(which="both", length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
        axis_labels(ax, "Lo que entendió el modelo", "Lo que hiciste (real)")
        ax.xaxis.set_label_position("top")
        ax.xaxis.tick_top()
        plt_off = cm.copy()
        np.fill_diagonal(plt_off, 0)
        if plt_off.max() == 0:
            fc.reading.config(text="Sin confusiones en test. Ojo: con pocos clips de test esto puede ser engañoso.", fg=OK)
            return
        i, j = np.unravel_index(plt_off.argmax(), plt_off.shape)
        txt = f"Confusión más frecuente: «{labels[i]}» entendido como «{labels[j]}» ({plt_off[i, j]} de {rows[i, 0]} clips). "
        if labels[i] == "ninguno":
            txt += "Esto provoca acciones accidentales: graba más movimientos normales de la mano."
        elif labels[j] == "ninguno":
            txt += "El gesto se pierde, pero no se dispara nada por error (es el fallo seguro). Graba más clips de ese gesto."
        else:
            txt += "Dos gestos se parecen: graba más variaciones de cada uno."
        fc.reading.config(text=txt, fg=WARN)

    def _draw_per_class(self, res):
        fc = self.fc_pc
        fig = fc.fig
        if not res:
            empty_plot(fig, "Sin test todavía\nPanel, paso 5: Testear")
            fc.reading.config(text="Aún no hay resultados de test.", fg=TEXT)
            return
        labels = res["labels"]
        rec = np.array([res["per_class"][l]["recall"] for l in labels])
        prec = np.array([res["per_class"][l]["precision"] for l in labels])
        fig.clear()
        ax = fig.add_subplot(111)
        style_axes(ax)
        ax.grid(axis="y", visible=False)
        y = np.arange(len(labels))
        ax.barh(y - 0.2, rec, height=0.36, color=RED_HI, label="Detección")
        ax.barh(y + 0.2, prec, height=0.36, color="#c9c9d3", label="Precisión")
        for yy, v in zip(y - 0.2, rec):
            ax.text(min(v, 1) + 0.01, yy, f"{v:.0%}", va="center", color=TEXT, fontsize=8)
        for yy, v in zip(y + 0.2, prec):
            ax.text(min(v, 1) + 0.01, yy, f"{v:.0%}", va="center", color=MUTED, fontsize=8)
        ax.set_yticks(y, [short(l) for l in labels])
        ax.invert_yaxis()
        ax.set_xlim(0, 1.12)
        ax.set_xticks([0, .25, .5, .75, 1])
        ax.xaxis.set_major_formatter(pct)
        ax.legend(frameon=False, labelcolor=TEXT, fontsize=8, loc="lower right")
        wi, pi = int(rec.argmin()), int(prec.argmin())
        fc.reading.config(text=f"Menor detección: «{labels[wi]}» ({rec[wi]:.0%}): a veces no lo reconoce. "
                               f"Menor precisión: «{labels[pi]}» ({prec[pi]:.0%}): "
                               + ("a veces se dispara sin que lo hagas." if prec[pi] < 0.9 else "va bien."),
                          fg=WARN if min(rec[wi], prec[pi]) < 0.85 else OK)

    def _update_stats(self, hist, res):
        v = self.tile_vals
        if res:
            acc = res["accuracy"]
            v[0].config(text=f"{acc:.1%}", fg=OK if acc >= .9 else (WARN if acc >= .75 else BAD))
            v[1].config(text=str(res["n_test"]))
        else:
            for t in v[:2]:
                t.config(text="—", fg=TEXT)
        if hist and hist.get("val_accuracy"):
            v[2].config(text=f"{max(hist['val_accuracy']):.1%}", fg=TEXT)
            v[3].config(text=str(len(hist["val_accuracy"])), fg=TEXT)
        else:
            for t in v[2:]:
                t.config(text="—", fg=TEXT)
        if not res:
            self.verdict.config(text="Entrena y testea el modelo para ver aquí un resumen.", fg=MUTED)
        else:
            acc = res["accuracy"]
            few = "  (pocos clips de test: el porcentaje puede variar mucho)" if res["n_test"] < 60 else ""
            if acc >= .9:
                self.verdict.config(text=f"Buen resultado. Pruébalo en dry-run antes de usarlo en serio.{few}", fg=OK)
            elif acc >= .75:
                self.verdict.config(text=f"Aceptable. Graba más clips en los gestos que se confunden.{few}", fg=WARN)
            else:
                self.verdict.config(text=f"Insuficiente. Faltan datos o más variedad de sesiones.{few}", fg=BAD)

    # ------------------------------------------------------------ configuración
    def _build_cfg(self, wrap):
        tk.Label(wrap, text="Configuración", bg=BG, fg=TEXT, font=font(16, "bold")).pack(anchor="w")
        WrapLabel(wrap, "Se guarda en config/settings.yaml conservando los comentarios. Reinicia el demonio para "
                        "aplicar los cambios.", bg=BG, fg=MUTED, size=10).pack(fill="x", pady=(2, 0))
        self.cfg_vars = {}
        for title, fields in CFG_GROUPS:
            card = Card(wrap, title)
            card.pack(fill="x", pady=(14, 0))
            for label, sec, key, typ, desc in fields:
                row = tk.Frame(card.body, bg=SURFACE)
                row.pack(fill="x", pady=6)
                left = tk.Frame(row, bg=SURFACE)
                left.pack(side="left", fill="x", expand=True)
                tk.Label(left, text=label, bg=SURFACE, fg=TEXT, font=font(10, "bold"), anchor="w").pack(fill="x")
                if desc:
                    WrapLabel(left, desc, fg=MUTED, size=9).pack(fill="x")
                if typ is bool:
                    var = tk.BooleanVar()
                    ttk.Checkbutton(row, text="Activado", variable=var).pack(side="right", padx=(16, 0))
                elif key == "backend":
                    var = tk.StringVar()
                    ttk.Combobox(row, textvariable=var, values=("auto", "pynput", "ydotool"), state="readonly",
                                 width=12).pack(side="right", padx=(16, 0))
                else:
                    var = tk.StringVar()
                    ttk.Entry(row, textvariable=var, width=12).pack(side="right", padx=(16, 0))
                self.cfg_vars[(sec, key)] = (typ, var)
        bar = tk.Frame(wrap, bg=BG)
        bar.pack(fill="x", pady=(16, 0))
        ttk.Button(bar, text="Guardar cambios", style="Accent.TButton", command=self.save_cfg).pack(side="left")
        ttk.Button(bar, text="Descartar", style="Ghost.TButton", command=self.load_cfg_values).pack(side="left", padx=8)
        self.load_cfg_values()

    def load_cfg_values(self):
        for (sec, key), (typ, var) in self.cfg_vars.items():
            val = self.settings[sec][key]
            var.set(bool(val) if typ is bool else str(val))

    def save_cfg(self):
        changes = {}
        try:
            for (sec, key), (typ, var) in self.cfg_vars.items():
                v = var.get()
                changes[(sec, key)] = bool(v) if typ is bool else (int(float(v)) if typ is int else typ(str(v).strip()))
        except ValueError as e:
            messagebox.showerror("Valor inválido", str(e))
            return
        update_settings(changes)
        self.settings = load_settings()
        self.backend = resolve_backend(self.settings, self.info)
        self.refresh()
        messagebox.showinfo("Guardado", "Configuración guardada. Reinicia el demonio para aplicarla.")

    # ------------------------------------------------------------ acciones
    def _build_actions(self, wrap):
        tk.Label(wrap, text="Acciones", bg=BG, fg=TEXT, font=font(16, "bold")).pack(anchor="w")
        WrapLabel(wrap, "Qué hace cada gesto en tu sistema. Es el archivo config/actions.yaml.", bg=BG, fg=MUTED,
                  size=10).pack(fill="x", pady=(2, 0))
        ref = Card(wrap, "Referencia")
        ref.pack(fill="x", pady=(14, 0))
        for code, meaning in (('keys: "alt+f4"', "combinación de teclas"),
                              ("alt_tab: forward", "Alt+Tab encadenable (forward / backward)"),
                              ('cmd: "xdotool …"', "comando externo"),
                              ("windows · linux · linux_wayland · all", "sistema al que aplica cada definición")):
            r = tk.Frame(ref.body, bg=SURFACE)
            r.pack(fill="x", pady=2)
            tk.Label(r, text=code, bg=SURFACE, fg=RED_HI, font=(MONO, 10), width=38, anchor="w").pack(side="left")
            tk.Label(r, text=meaning, bg=SURFACE, fg=MUTED, font=font(10), anchor="w").pack(side="left")
        ed = Card(wrap, "Editor", "actions.yaml")
        ed.pack(fill="x", pady=(14, 0))
        lf = tk.Frame(ed.body, bg=SURFACE)
        lf.pack(fill="x")
        self.actions_txt = tk.Text(lf, height=24, wrap="none", bg=PLOT_BG, fg=TEXT, insertbackground=RED_HI,
                                   relief="flat", font=(MONO, 10), padx=12, pady=10, highlightthickness=1,
                                   highlightbackground=BORDER, selectbackground=RED_LO, undo=True)
        sb = ttk.Scrollbar(lf, command=self.actions_txt.yview)
        self.actions_txt.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self.actions_txt.pack(side="left", fill="x", expand=True)
        self.load_actions_text()
        bar = tk.Frame(ed.body, bg=SURFACE)
        bar.pack(fill="x", pady=(12, 0))
        ttk.Button(bar, text="Guardar acciones", style="Accent.TButton", command=self.save_actions).pack(side="left")
        ttk.Button(bar, text="Recargar", style="Ghost.TButton", command=self.load_actions_text).pack(side="left", padx=8)

    def load_actions_text(self):
        self.actions_txt.delete("1.0", "end")
        self.actions_txt.insert("1.0", ACTIONS_FILE.read_text(encoding="utf-8"))

    def save_actions(self):
        text = self.actions_txt.get("1.0", "end")
        try:
            yaml.safe_load(text)
        except yaml.YAMLError as e:
            messagebox.showerror("YAML inválido", str(e))
            return
        ACTIONS_FILE.write_text(text, encoding="utf-8")
        self.build_chips()
        messagebox.showinfo("Guardado", "Acciones guardadas. Reinicia el demonio para aplicarlas.")

    # ------------------------------------------------------------ estado
    def check_camera(self):
        self.cam = (False, "verificando…")
        threading.Thread(target=lambda: setattr(self, "cam", status.camera_check(self.settings)), daemon=True).start()

    def manual_refresh(self):
        self.settings = load_settings()
        self.check_camera()
        self.after(1500, self.refresh)
        self.refresh()

    def refresh(self):
        checks, ready = status.readiness(self.settings, self.cam)
        for name, ok, detail, req in checks:
            icon, det = self._check_row(name)
            draw_status(icon, "ok" if ok else ("bad" if req else "warn"))
            det.config(text=detail, fg=MUTED if ok else TEXT)
        missing = [n for n, ok, _, req in checks if req and not ok]
        if ready:
            self.pill_ready.config(text="LISTO PARA USAR", bg=OK_BG, fg=OK)
        else:
            self.pill_ready.config(text="FALTA 1 PASO" if len(missing) == 1 else f"FALTAN {len(missing)} PASOS", bg=BAD_BG, fg=BAD)
        self._model_ok = next(ok for n, ok, _, _ in checks if n == "Modelo")

        if self._table_gestures != self.settings["gestures"]:
            self._build_table()
        minc = self.settings["dataset"]["min_clips_per_gesture"]
        for r in status.dataset_status(self.settings):
            pb, cnt, nums, st = self.table_rows[r["gesture"]]
            enough = r["clips"] >= minc
            pb.configure(value=min(100, r["clips"] / max(minc, 1) * 100), style="Ok.Horizontal.TProgressbar" if enough else "Red.Horizontal.TProgressbar")
            cnt.config(text=f"{r['clips']} / {minc}", fg=OK if enough else TEXT)
            for lab, k in zip(nums, ("train", "val", "test", "pending_videos")):
                lab.config(text=str(r[k]), fg=MUTED if r[k] == 0 else TEXT)
            problems = []
            if not enough:
                problems.append(f"faltan {minc - r['clips']} clips")
            if r["unsplit"]:
                problems.append("reconstruir dataset")
            elif r["clips"] and min(r["train"], r["val"], r["test"]) < 1:
                problems.append("falta train/val/test")
            st.config(text=("" + ", ".join(problems)) if problems else "listo", fg=BAD if problems else OK)
        self.table_note.config(text=f"Mínimo por gesto: {minc} clips (se cambia en Configuración). Cada gesto necesita "
                                    "al menos un clip en train, val y test; si no, pulsa «3 · Construir dataset».")
        self.update_daemon()
        self.update_welcome()
        self.draw_plots()

    def update_daemon(self):
        st = status.read_daemon_status()
        running = st is not None
        self.btn_start.state(["disabled"] if running else ["!disabled"])
        self.btn_stop.state(["!disabled"] if running else ["disabled"])
        if not running:
            self.pill_daemon.config(text="Demonio detenido", bg=SURFACE2, fg=MUTED)
            self.d_state.config(text="DETENIDO", fg=MUTED)
            self.d_sub.config(text="No hay ningún proceso en segundo plano.", fg=MUTED)
            for p in (self.p_mode, self.p_fps, self.p_hand):
                p.config(text="—", bg=SURFACE2, fg=MUTED)
            self.d_last.config(text="—")
            return
        state = st.get("state")
        if state == "running":
            self.pill_daemon.config(text="Demonio activo", bg=OK_BG, fg=OK)
            self.d_state.config(text="ACTIVO", fg=OK)
            self.d_sub.config(text=f"pid {st.get('pid')} · teclas: {st.get('backend')}", fg=MUTED)
            if st.get("dry_run"):
                self.p_mode.config(text="DRY-RUN", bg=WARN_BG, fg=WARN)
            else:
                self.p_mode.config(text="ACCIONES REALES", bg=RED_LO, fg="#ffd7dc")
            self.p_fps.config(text=f"{st.get('fps', 0)} fps", bg=SURFACE2, fg=TEXT)
            self.p_hand.config(text="MANO ✓" if st.get("hand") else "sin mano", bg=OK_BG if st.get("hand") else SURFACE2,
                               fg=OK if st.get("hand") else MUTED)
            if st.get("last_gesture"):
                self.d_last.config(text=f"{st['last_gesture']} · {st['last_conf']}  ·  {st['last_action']}  "
                                        f"({st['last_time']})", fg=TEXT)
            else:
                self.d_last.config(text="Esperando un gesto…", fg=MUTED)
        elif state == "error":
            self.pill_daemon.config(text="Demonio con error", bg=BAD_BG, fg=BAD)
            self.d_state.config(text="ERROR", fg=BAD)
            self.d_sub.config(text=str(st.get("message", "")), fg=BAD)
        else:
            self.pill_daemon.config(text="Demonio iniciando…", bg=WARN_BG, fg=WARN)
            self.d_state.config(text="INICIANDO", fg=WARN)
            self.d_sub.config(text=str(st.get("message", "cargando modelo")), fg=MUTED)

    def tick(self):
        self.update_daemon()
        self._n += 1
        if self._n % 3 == 0 and not self.proc:
            self.refresh()
        self.after(1000, self.tick)

    # ------------------------------------------------------------ tareas
    def logw(self, text):
        s = text.lstrip()
        tag = ()
        if s.startswith("$"):
            tag = ("cmd",)
        elif s.startswith("[terminó con código 0"):
            tag = ("ok",)
        elif s.startswith("[terminó") or any(k in text for k in ("ERROR", "Error", "Traceback")):
            tag = ("err",)
        self.log.config(state="normal")
        self.log.insert("end", text, tag)
        self.log.see("end")
        self.log.config(state="disabled")

    def set_busy(self, busy, name=""):
        for b in self.task_buttons:
            b.state(["disabled"] if busy else ["!disabled"])
        if busy:
            self.task_lbl.config(text=f"Ejecutando {name}…", fg=TEXT)
            self.task_pb.pack(side="right")
            self.task_pb.start(12)
        else:
            self.task_pb.stop()
            self.task_pb.pack_forget()

    def task(self, script, ask_stop=False):
        if self.proc and self.proc.poll() is None:
            messagebox.showinfo("Ocupado", "Ya hay una tarea en ejecución.")
            return
        if ask_stop and status.daemon_pid():
            if not messagebox.askyesno("Cámara en uso", "El demonio está usando la cámara. ¿Detenerlo para grabar?"):
                return
            status.stop_daemon()
        cmd = [sys.executable, "-u", str(SCRIPTS / script)]
        self.logw(f"\n$ python scripts/{script}\n")
        self.proc = subprocess.Popen(cmd, cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     text=True, bufsize=1, encoding="utf-8", errors="replace")
        self._task_name = script
        self.set_busy(True, script)
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
                    self.task_lbl.config(text=f"Última tarea: {self._task_name} ({'ok' if val == 0 else f'código {val}'})",
                                         fg=OK if val == 0 else BAD)
                    self.refresh()
        except queue.Empty:
            pass
        self.after(100, self.pump)

    # ------------------------------------------------------------ demonio
    def start(self):
        if not self._model_ok:
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