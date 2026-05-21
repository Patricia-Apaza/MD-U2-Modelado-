# =============================================================================
#  MINERÍA DE DATOS — EXAMEN UNIDAD 2
#  Interfaz gráfica con tkinter
#  Ing.: Milton Edward Humpiri Flores — E.P. Ingeniería de Sistemas UPeU
# =============================================================================
#  INSTALAR dependencias (ejecutar UNA vez en PowerShell):
#  pip install pandas numpy matplotlib seaborn scikit-learn scipy
# =============================================================================

import warnings, threading
warnings.filterwarnings("ignore")

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import os, sys

# ── Verificar librerías ───────────────────────────────────────────────────
MISSING = []
try: import pandas as pd
except: MISSING.append("pandas")
try: import numpy as np
except: MISSING.append("numpy")
try: import matplotlib
except: MISSING.append("matplotlib")
try: import seaborn
except: MISSING.append("seaborn")
try: import sklearn
except: MISSING.append("scikit-learn")
try: import scipy
except: MISSING.append("scipy")

if MISSING:
    root = tk.Tk(); root.withdraw()
    messagebox.showerror(
        "Librerías faltantes",
        f"Instala estas librerías antes de continuar:\n\n"
        f"pip install {' '.join(MISSING)}\n\n"
        f"Abre PowerShell y ejecuta ese comando."
    )
    sys.exit(1)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns

from sklearn.model_selection   import train_test_split
from sklearn.preprocessing     import MinMaxScaler, LabelEncoder
from sklearn.dummy             import DummyClassifier
from sklearn.linear_model      import LogisticRegression
from sklearn.tree              import DecisionTreeClassifier, plot_tree
from sklearn.ensemble          import RandomForestClassifier
from sklearn.neighbors         import KNeighborsClassifier
from sklearn.naive_bayes       import BernoulliNB
from sklearn.cluster           import KMeans, AgglomerativeClustering
from sklearn.metrics           import (
    silhouette_score, accuracy_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix,
    precision_score, recall_score, classification_report
)
from scipy.cluster.hierarchy   import dendrogram, linkage

# ═════════════════════════════════════════════════════════════════════════════
#  COLORES Y ESTILOS
# ═════════════════════════════════════════════════════════════════════════════
C_BG       = "#F8F9FA"
C_WHITE    = "#FFFFFF"
C_PRIMARY  = "#2563EB"
C_PRIMARY2 = "#1D4ED8"
C_SUCCESS  = "#16A34A"
C_WARN     = "#D97706"
C_DANGER   = "#DC2626"
C_MUTED    = "#6B7280"
C_BORDER   = "#E5E7EB"
C_DARK     = "#111827"
C_LIGHT    = "#F3F4F6"
C_HEADER   = "#1E3A5F"

PALETA = ["#2563EB","#16A34A","#D97706","#7C3AED","#DB2777","#0891B2"]

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor":   "#F9FAFB",
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.grid":        True,
    "grid.alpha":       0.3,
    "grid.linestyle":   "--",
    "font.family":      "sans-serif",
    "font.size":        10,
})

# ═════════════════════════════════════════════════════════════════════════════
#  ESTADO GLOBAL
# ═════════════════════════════════════════════════════════════════════════════
state = {
    "df": None, "X": None, "y": None,
    "X_train": None, "X_val": None, "X_test": None,
    "y_train": None, "y_val": None, "y_test": None,
    "X_train_sc": None, "X_val_sc": None, "X_test_sc": None,
    "resultados": None, "df_res": None,
    "labels_km": None, "labels_agg": None,
    "sil_km": 0, "sil_agg": 0,
    "mejor_modelo": None,
    "num_cols": None,
    "X_clust": None,
}

# ═════════════════════════════════════════════════════════════════════════════
#  VENTANA PRINCIPAL
# ═════════════════════════════════════════════════════════════════════════════
root = tk.Tk()
root.title("Minería de Datos — Examen Unidad 2  |  UPeU")
root.geometry("1200x800")
root.configure(bg=C_BG)
root.minsize(1000, 680)

# ── Fuentes ───────────────────────────────────────────────────────────────
F_TITLE  = ("Segoe UI", 15, "bold")
F_HEAD   = ("Segoe UI", 11, "bold")
F_BODY   = ("Segoe UI", 10)
F_SMALL  = ("Segoe UI",  9)
F_MONO   = ("Consolas",  9)
F_BIG    = ("Segoe UI", 22, "bold")

# ═════════════════════════════════════════════════════════════════════════════
#  HEADER
# ═════════════════════════════════════════════════════════════════════════════
hdr = tk.Frame(root, bg=C_HEADER, height=62)
hdr.pack(fill="x"); hdr.pack_propagate(False)

tk.Label(hdr, text="⛏  Minería de Datos — Examen Unidad 2",
         font=("Segoe UI", 14, "bold"), bg=C_HEADER, fg="white").pack(side="left", padx=20, pady=14)
tk.Label(hdr, text="E.P. Ingeniería de Sistemas  |  UPeU",
         font=F_SMALL, bg=C_HEADER, fg="#93C5FD").pack(side="right", padx=20)

# ═════════════════════════════════════════════════════════════════════════════
#  CUERPO: sidebar + contenido
# ═════════════════════════════════════════════════════════════════════════════
body = tk.Frame(root, bg=C_BG)
body.pack(fill="both", expand=True, padx=0, pady=0)

# ── Sidebar ───────────────────────────────────────────────────────────────
sidebar = tk.Frame(body, bg=C_WHITE, width=210,
                   relief="flat", bd=0,
                   highlightthickness=1, highlightbackground=C_BORDER)
sidebar.pack(side="left", fill="y")
sidebar.pack_propagate(False)

# ── Área de contenido ─────────────────────────────────────────────────────
content = tk.Frame(body, bg=C_BG)
content.pack(side="left", fill="both", expand=True)

# ═════════════════════════════════════════════════════════════════════════════
#  PANELS DICT
# ═════════════════════════════════════════════════════════════════════════════
panels = {}

def show_panel(name):
    for p in panels.values():
        p.pack_forget()
    panels[name].pack(fill="both", expand=True)
    for btn in nav_buttons:
        btn.configure(bg=C_WHITE, fg=C_DARK, relief="flat")
    nav_buttons[list(panels.keys()).index(name)].configure(
        bg=C_PRIMARY, fg="white")

# ═════════════════════════════════════════════════════════════════════════════
#  HELPER: frame con scroll
# ═════════════════════════════════════════════════════════════════════════════
def scrollable(parent):
    outer = tk.Frame(parent, bg=C_BG)
    outer.pack(fill="both", expand=True)
    canvas = tk.Canvas(outer, bg=C_BG, bd=0, highlightthickness=0)
    sb = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
    inner = tk.Frame(canvas, bg=C_BG)
    inner.bind("<Configure>",
               lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
    canvas.create_window((0, 0), window=inner, anchor="nw")
    canvas.configure(yscrollcommand=sb.set)
    canvas.pack(side="left", fill="both", expand=True)
    sb.pack(side="right", fill="y")
    canvas.bind_all("<MouseWheel>",
                    lambda e: canvas.yview_scroll(-1*(e.delta//120), "units"))
    return inner

def card(parent, title=None, pad=14):
    f = tk.Frame(parent, bg=C_WHITE,
                 highlightthickness=1, highlightbackground=C_BORDER)
    f.pack(fill="x", padx=16, pady=8)
    if title:
        tk.Label(f, text=title, font=F_HEAD, bg=C_WHITE,
                 fg=C_HEADER).pack(anchor="w", padx=pad, pady=(pad,4))
        sep = tk.Frame(f, bg=C_BORDER, height=1)
        sep.pack(fill="x", padx=pad)
    inner = tk.Frame(f, bg=C_WHITE)
    inner.pack(fill="x", padx=pad, pady=(8, pad))
    return inner

def metric_box(parent, label, value, color=C_PRIMARY):
    f = tk.Frame(parent, bg=color, width=130, height=72)
    f.pack(side="left", padx=5, pady=4)
    f.pack_propagate(False)
    tk.Label(f, text=str(value), font=("Segoe UI", 18, "bold"),
             bg=color, fg="white").pack(pady=(10, 0))
    tk.Label(f, text=label, font=("Segoe UI", 8),
             bg=color, fg="white").pack()

def text_box(parent, height=8):
    t = scrolledtext.ScrolledText(parent, font=F_MONO, height=height,
                                  bg="#F1F5F9", fg=C_DARK,
                                  relief="flat", bd=0, wrap="word",
                                  highlightthickness=1,
                                  highlightbackground=C_BORDER)
    t.pack(fill="both", expand=True, pady=4)
    return t

def write(t, txt):
    t.configure(state="normal")
    t.delete("1.0", "end")
    t.insert("end", txt)
    t.configure(state="disabled")

# ═════════════════════════════════════════════════════════════════════════════
#  PANEL 0 — DATOS
# ═════════════════════════════════════════════════════════════════════════════
p0 = tk.Frame(content, bg=C_BG)
panels["Datos"] = p0
sc0 = scrollable(p0)

# Zona de carga
c0 = card(sc0, "📂  Cargar dataset")

# Recuadro drag-and-drop visual
drop_frame = tk.Frame(c0, bg="#EFF6FF",
                      highlightthickness=2, highlightbackground=C_PRIMARY)
drop_frame.pack(fill="x", pady=(0,10))

tk.Label(drop_frame, text="⬆", font=("Segoe UI", 28),
         bg="#EFF6FF", fg=C_PRIMARY).pack(pady=(18,4))
lbl_file = tk.Label(drop_frame,
                    text="Haga clic en el botón para seleccionar su archivo CSV",
                    font=F_BODY, bg="#EFF6FF", fg=C_MUTED)
lbl_file.pack(pady=(0,4))
tk.Label(drop_frame, text="Formato: .csv separado por comas",
         font=F_SMALL, bg="#EFF6FF", fg="#93C5FD").pack(pady=(0,16))

def browse_file():
    path = filedialog.askopenfilename(
        title="Seleccionar dataset CSV",
        filetypes=[("CSV files","*.csv"),("All files","*.*")])
    if path:
        var_path.set(path)
        lbl_file.configure(
            text=f"✓  {os.path.basename(path)}",
            fg=C_SUCCESS, font=(F_BODY[0], F_BODY[1], "bold"))
        drop_frame.configure(highlightbackground=C_SUCCESS, bg="#F0FDF4")

var_path = tk.StringVar()

btn_browse = tk.Button(c0, text="  📁  Seleccionar archivo CSV  ",
                       font=F_HEAD, bg=C_PRIMARY, fg="white",
                       activebackground=C_PRIMARY2, relief="flat",
                       cursor="hand2", command=browse_file, padx=10, pady=8)
btn_browse.pack(pady=4)

# Configuración
c0b = card(sc0, "⚙️  Configuración del análisis")

row1 = tk.Frame(c0b, bg=C_WHITE)
row1.pack(fill="x", pady=4)

tk.Label(row1, text="Columna objetivo:", font=F_BODY,
         bg=C_WHITE, fg=C_DARK, width=20, anchor="w").pack(side="left")
var_target = tk.StringVar(value="")
entry_target = tk.Entry(row1, textvariable=var_target, font=F_BODY,
                        width=22, relief="solid", bd=1)
entry_target.pack(side="left", padx=6)
tk.Label(row1, text="(dejar vacío = última columna)",
         font=F_SMALL, bg=C_WHITE, fg=C_MUTED).pack(side="left")

row2 = tk.Frame(c0b, bg=C_WHITE)
row2.pack(fill="x", pady=4)
tk.Label(row2, text="Número de clústeres K:", font=F_BODY,
         bg=C_WHITE, fg=C_DARK, width=20, anchor="w").pack(side="left")
var_k = tk.IntVar(value=3)
spin_k = tk.Spinbox(row2, from_=2, to=10, textvariable=var_k,
                    font=F_BODY, width=5, relief="solid", bd=1)
spin_k.pack(side="left", padx=6)

# Botón ejecutar
btn_run = tk.Button(c0b,
                    text="  ▶  EJECUTAR ANÁLISIS COMPLETO  ",
                    font=("Segoe UI", 11, "bold"),
                    bg=C_SUCCESS, fg="white",
                    activebackground="#15803D",
                    relief="flat", cursor="hand2",
                    padx=14, pady=10)
btn_run.pack(pady=12)

# Log
c0c = card(sc0, "📋  Log de ejecución")
log_box = scrolledtext.ScrolledText(c0c, font=F_MONO, height=10,
                                    bg="#0F172A", fg="#86EFAC",
                                    relief="flat", bd=0, state="disabled",
                                    highlightthickness=1,
                                    highlightbackground=C_BORDER)
log_box.pack(fill="both")

# Preview dataset
c0d = card(sc0, "👁  Vista previa del dataset")
preview_box = scrolledtext.ScrolledText(c0d, font=F_MONO, height=8,
                                        bg="#F8FAFC", fg=C_DARK,
                                        relief="flat", bd=0, state="disabled",
                                        highlightthickness=1,
                                        highlightbackground=C_BORDER)
preview_box.pack(fill="both")

def log(msg, color=None):
    log_box.configure(state="normal")
    log_box.insert("end", msg + "\n")
    log_box.see("end")
    log_box.configure(state="disabled")
    root.update_idletasks()

# ═════════════════════════════════════════════════════════════════════════════
#  PANEL 1 — PARTICIÓN
# ═════════════════════════════════════════════════════════════════════════════
p1 = tk.Frame(content, bg=C_BG)
panels["Partición"] = p1
sc1 = scrollable(p1)

c1a = card(sc1, "📊  Métricas de partición")
metrics_part_frame = tk.Frame(c1a, bg=C_WHITE)
metrics_part_frame.pack(fill="x")

c1b = card(sc1, "📈  Gráfico de distribución")
fig_part_holder = tk.Frame(c1b, bg=C_WHITE)
fig_part_holder.pack(fill="both", expand=True)

c1c = card(sc1, "💬  Explicación — Data Leakage y Baseline")
part_text = text_box(c1c, height=14)

# ═════════════════════════════════════════════════════════════════════════════
#  PANEL 2 — CLUSTERING
# ═════════════════════════════════════════════════════════════════════════════
p2 = tk.Frame(content, bg=C_BG)
panels["Clustering"] = p2
sc2 = scrollable(p2)

c2a = card(sc2, "🔵  Métricas de clustering")
metrics_clust_frame = tk.Frame(c2a, bg=C_WHITE)
metrics_clust_frame.pack(fill="x")

c2b = card(sc2, "📈  Gráficos — K-means, Silhouette y Dendrograma")
fig_clust_holder = tk.Frame(c2b, bg=C_WHITE)
fig_clust_holder.pack(fill="both")

c2c = card(sc2, "🗂  Perfiles de clústeres")
clust_text = text_box(c2c, height=12)

c2d = card(sc2, "💬  Interpretación y perfiles de clientes")
clust_interp = text_box(c2d, height=8)

# ═════════════════════════════════════════════════════════════════════════════
#  PANEL 3 — CLASIFICACIÓN
# ═════════════════════════════════════════════════════════════════════════════
p3 = tk.Frame(content, bg=C_BG)
panels["Clasificación"] = p3
sc3 = scrollable(p3)

c3a = card(sc3, "🏆  Mejor modelo")
metrics_clas_frame = tk.Frame(c3a, bg=C_WHITE)
metrics_clas_frame.pack(fill="x")

c3b = card(sc3, "📈  Comparativa de modelos e importancia de variables")
fig_clas_holder = tk.Frame(c3b, bg=C_WHITE)
fig_clas_holder.pack(fill="both")

c3c = card(sc3, "💬  Explicación — Árbol de Decisión y Random Forest")
clas_text = text_box(c3c, height=14)

# ═════════════════════════════════════════════════════════════════════════════
#  PANEL 4 — EVALUACIÓN
# ═════════════════════════════════════════════════════════════════════════════
p4 = tk.Frame(content, bg=C_BG)
panels["Evaluación"] = p4
sc4 = scrollable(p4)

c4a = card(sc4, "📈  Curva ROC")
fig_roc_holder = tk.Frame(c4a, bg=C_WHITE)
fig_roc_holder.pack(fill="both")

c4b = card(sc4, "📋  Tabla comparativa de corridas")
tabla_frame = tk.Frame(c4b, bg=C_WHITE)
tabla_frame.pack(fill="x")

c4c = card(sc4, "💬  Comunicación de resultados — Resumen ejecutivo")
eval_text = text_box(c4c, height=12)

# ═════════════════════════════════════════════════════════════════════════════
#  PANEL 5 — MATRIZ DE CONFUSIÓN
# ═════════════════════════════════════════════════════════════════════════════
p5 = tk.Frame(content, bg=C_BG)
panels["Matriz"] = p5
sc5 = scrollable(p5)

c5a = card(sc5, "🔲  Seleccionar modelo")
var_model_sel = tk.StringVar()
combo_model = ttk.Combobox(c5a, textvariable=var_model_sel,
                           font=F_BODY, state="readonly", width=30)
combo_model.pack(anchor="w", pady=4)

c5b = card(sc5, "🗺  Matriz de confusión")
fig_cm_holder = tk.Frame(c5b, bg=C_WHITE)
fig_cm_holder.pack(fill="both")

c5c = card(sc5, "📐  Métricas derivadas")
cm_metrics_frame = tk.Frame(c5c, bg=C_WHITE)
cm_metrics_frame.pack(fill="x")

c5d = card(sc5, "💬  Interpretación de la matriz")
matrix_text = text_box(c5d, height=10)

# ═════════════════════════════════════════════════════════════════════════════
#  SIDEBAR NAVIGATION
# ═════════════════════════════════════════════════════════════════════════════
tk.Label(sidebar, text="NAVEGACIÓN", font=("Segoe UI", 8, "bold"),
         bg=C_WHITE, fg=C_MUTED).pack(anchor="w", padx=16, pady=(16,6))

nav_icons  = ["📂","📊","🔵","🌲","📈","🔲"]
nav_labels = ["Datos","Partición","Clustering","Clasificación","Evaluación","Matriz"]
nav_buttons = []

for icon, lbl in zip(nav_icons, nav_labels):
    b = tk.Button(sidebar, text=f"  {icon}  {lbl}",
                  font=F_BODY, bg=C_WHITE, fg=C_DARK,
                  activebackground=C_PRIMARY, activeforeground="white",
                  relief="flat", anchor="w", cursor="hand2",
                  command=lambda n=lbl: show_panel(n))
    b.pack(fill="x", padx=8, pady=2, ipady=6)
    nav_buttons.append(b)

tk.Frame(sidebar, bg=C_BORDER, height=1).pack(fill="x", padx=16, pady=16)
tk.Label(sidebar, text="Sé Íntegro\nSé Misionero\nSé Innovador",
         font=("Segoe UI", 8, "italic"), bg=C_WHITE,
         fg=C_MUTED, justify="center").pack(padx=8)

# ═════════════════════════════════════════════════════════════════════════════
#  HELPERS PARA INCRUSTAR FIGURAS
# ═════════════════════════════════════════════════════════════════════════════
def embed_fig(holder, fig):
    for w in holder.winfo_children():
        w.destroy()
    canvas = FigureCanvasTkAgg(fig, master=holder)
    canvas.draw()
    canvas.get_tk_widget().pack(fill="both", expand=True)

def build_table(parent, df_table, best_row=None):
    for w in parent.winfo_children():
        w.destroy()
    cols = list(df_table.columns)
    idx  = list(df_table.index)
    style_h = {"bg": C_HEADER, "fg": "white", "font": F_HEAD,
                "relief": "flat", "padx": 8, "pady": 6}
    style_n = {"bg": C_WHITE,  "fg": C_DARK,  "font": F_BODY,
                "relief": "flat", "padx": 8, "pady": 5}
    style_b = {"bg": "#DCFCE7","fg": "#15803D","font": F_BODY,
                "relief": "flat", "padx": 8, "pady": 5}

    tk.Label(parent, text="Modelo", **style_h).grid(row=0, column=0, sticky="nsew")
    for j, col in enumerate(cols):
        tk.Label(parent, text=col, **style_h).grid(row=0,column=j+1,sticky="nsew")
    tk.Frame(parent, bg=C_BORDER, height=1).grid(
        row=1, columnspan=len(cols)+1, sticky="ew")

    for i, row_idx in enumerate(idx):
        st = style_b if row_idx == best_row else style_n
        bg = st["bg"]
        tk.Label(parent, text=row_idx, **st).grid(
            row=i+2, column=0, sticky="nsew")
        for j, col in enumerate(cols):
            val = df_table.loc[row_idx, col]
            txt = f"{val:.4f}" if isinstance(val, float) else str(val)
            tk.Label(parent, text=txt, bg=bg, fg=st["fg"],
                     font=F_BODY, relief="flat", padx=8, pady=5
                     ).grid(row=i+2, column=j+1, sticky="nsew")
        tk.Frame(parent, bg=C_BORDER, height=1).grid(
            row=100+i, columnspan=len(cols)+1, sticky="ew")

# ═════════════════════════════════════════════════════════════════════════════
#  LÓGICA DE ANÁLISIS
# ═════════════════════════════════════════════════════════════════════════════
def run_analysis():
    path   = var_path.get()
    target = var_target.get().strip()
    K      = var_k.get()

    if not path:
        messagebox.showwarning("Sin archivo", "Seleccione un archivo CSV primero.")
        return

    btn_run.configure(state="disabled", text="  ⏳  Ejecutando...")
    log("=" * 55)
    log("▶ Iniciando análisis completo...")

    # ── Cargar datos ─────────────────────────────────────────────────────
    try:
        df = pd.read_csv(path)
        state["df"] = df
    except Exception as e:
        messagebox.showerror("Error al cargar", str(e))
        btn_run.configure(state="normal", text="  ▶  EJECUTAR ANÁLISIS COMPLETO  ")
        return

    log(f"✓ Dataset cargado: {df.shape[0]} filas × {df.shape[1]} columnas")

    if not target:
        target = df.columns[-1]
        log(f"  Columna objetivo detectada: '{target}'")

    # Preview
    write(preview_box, df.head(10).to_string())

    # ── Preparar features ─────────────────────────────────────────────────
    df_m = df.copy()
    cat_c = [c for c in df_m.select_dtypes(include=["object","category"]).columns
             if c != target]
    le = LabelEncoder()
    for col in cat_c:
        df_m[col] = le.fit_transform(df_m[col].astype(str))
    if df_m[target].dtype == object:
        df_m[target] = le.fit_transform(df_m[target].astype(str))

    X = df_m.drop(columns=[target])
    y = df_m[target]
    state["X"] = X; state["y"] = y

    # ── Partición 70/15/15 ───────────────────────────────────────────────
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=0.15/0.85, random_state=42, stratify=y_tmp)

    scaler = MinMaxScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_val_sc   = scaler.transform(X_val)
    X_test_sc  = scaler.transform(X_test)

    state.update({"X_train":X_train,"X_val":X_val,"X_test":X_test,
                  "y_train":y_train,"y_val":y_val,"y_test":y_test,
                  "X_train_sc":X_train_sc,"X_val_sc":X_val_sc,
                  "X_test_sc":X_test_sc})

    log(f"✓ Partición: {len(X_train)} train / {len(X_val)} val / {len(X_test)} test")

    # ── Panel Partición ───────────────────────────────────────────────────
    for w in metrics_part_frame.winfo_children(): w.destroy()
    n = len(X)
    for lbl, val, col in [
        ("Total",          str(n),          C_HEADER),
        ("Entrenamiento",  f"{len(X_train)}\n(70%)", C_PRIMARY),
        ("Validación",     f"{len(X_val)}\n(15%)",   C_WARN),
        ("Prueba",         f"{len(X_test)}\n(15%)",  C_SUCCESS),
        ("Columnas",       str(X.shape[1]), C_MUTED),
    ]:
        metric_box(metrics_part_frame, lbl, val, col)

    fig_p, axes_p = plt.subplots(1, 2, figsize=(10, 3.5))
    sizes  = [len(X_train), len(X_val), len(X_test)]
    labels_p = [f"Entrenamiento\n{len(X_train)} ({len(X_train)/n*100:.0f}%)",
                f"Validación\n{len(X_val)} ({len(X_val)/n*100:.0f}%)",
                f"Prueba\n{len(X_test)} ({len(X_test)/n*100:.0f}%)"]
    axes_p[0].pie(sizes, labels=labels_p,
                  colors=[PALETA[0],PALETA[2],PALETA[1]],
                  startangle=90, wedgeprops=dict(width=0.5))
    axes_p[0].set_title("Partición del dataset", fontweight="bold")

    cc = y_train.value_counts().sort_index()
    axes_p[1].bar([str(c) for c in cc.index], cc.values,
                  color=PALETA[:len(cc)], width=0.5, edgecolor="white")
    axes_p[1].set_title("Distribución de clases (entrenamiento)", fontweight="bold")
    axes_p[1].set_xlabel("Clase"); axes_p[1].set_ylabel("Cantidad")
    for i,v in enumerate(cc.values):
        axes_p[1].text(i, v+2, str(v), ha="center", fontsize=9)
    fig_p.tight_layout()
    embed_fig(fig_part_holder, fig_p)

    maj = y.value_counts().max()
    acc_base = maj/n
    write(part_text, f"""PARTICIÓN DE DATOS — 70 / 15 / 15
{'='*55}

El dataset se divide ESTRATIFICADAMENTE en tres subconjuntos:

  • Entrenamiento (70% = {len(X_train)} registros)
    El modelo aprende los patrones de los datos.

  • Validación (15% = {len(X_val)} registros)
    Se usa para ajustar hiperparámetros y detectar
    sobreajuste (overfitting) durante el desarrollo.

  • Prueba (15% = {len(X_test)} registros)
    Evaluación FINAL e imparcial del modelo.
    Solo se usa una vez, al final.

{'='*55}
DATA LEAKAGE — ¿Qué es y cómo se evita?
{'='*55}

El Data Leakage ocurre cuando información de validación/prueba
se filtra al entrenamiento, haciendo que el modelo parezca
mejor de lo que realmente es en producción.

  ✗ INCORRECTO:
    scaler.fit_transform(X_completo) → luego dividir
    (el scaler ya vio los datos de prueba)

  ✓ CORRECTO (aplicado aquí):
    scaler.fit(X_train)        ← aprende min/max solo de train
    scaler.transform(X_val)    ← aplica la misma escala
    scaler.transform(X_test)   ← aplica la misma escala

  También aplica a: imputación de nulos, encoding,
  selección de features, PCA, etc.

{'='*55}
MODELO BASELINE
{'='*55}

Baseline elegido: DummyClassifier (clase mayoritaria)
  → Accuracy baseline: {acc_base:.4f} ({acc_base*100:.1f}%)

Todo modelo más complejo DEBE superar este valor.
Se usa Regresión Logística como baseline más sofisticado.
""")
    log("✓ Partición completada")

    # ── Clustering ────────────────────────────────────────────────────────
    log("→ Ejecutando clustering...")
    num_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                if c != target]
    if not num_cols:
        num_cols = list(X.columns)
    state["num_cols"] = num_cols

    Xc_raw = df[num_cols].fillna(df[num_cols].median())
    sc_c   = MinMaxScaler()
    Xc     = sc_c.fit_transform(Xc_raw)
    state["X_clust"] = Xc

    K_range = range(2, min(8, len(Xc)//10+2))
    sil_list, inert_list = [], []
    for k in K_range:
        km  = KMeans(n_clusters=k, random_state=42, n_init=10)
        lb  = km.fit_predict(Xc)
        sil_list.append(silhouette_score(Xc, lb) if len(set(lb))>1 else 0)
        inert_list.append(km.inertia_)

    km_fin = KMeans(n_clusters=K, random_state=42, n_init=10)
    lbl_km = km_fin.fit_predict(Xc)
    sil_km = silhouette_score(Xc, lbl_km) if len(set(lbl_km))>1 else 0

    agg = AgglomerativeClustering(n_clusters=K, linkage="ward")
    lbl_agg = agg.fit_predict(Xc)
    sil_agg = silhouette_score(Xc, lbl_agg) if len(set(lbl_agg))>1 else 0

    state["labels_km"] = lbl_km
    state["labels_agg"] = lbl_agg
    state["sil_km"] = sil_km
    state["sil_agg"] = sil_agg

    df["cluster_kmeans"] = lbl_km
    cnt_km = pd.Series(lbl_km).value_counts().sort_index()

    for w in metrics_clust_frame.winfo_children(): w.destroy()
    for lbl2, val2, col2 in [
        ("K clústeres",     str(K),           C_PRIMARY),
        ("Silhouette KM",   f"{sil_km:.3f}",  C_SUCCESS if sil_km>0.3 else C_WARN),
        ("Silhouette Jer.", f"{sil_agg:.3f}", C_SUCCESS if sil_agg>0.3 else C_WARN),
        ("Clúster mayor",   str(cnt_km.max()), C_MUTED),
        ("Clúster menor",   str(cnt_km.min()), C_MUTED),
    ]:
        metric_box(metrics_clust_frame, lbl2, val2, col2)

    # Figuras clustering
    fig_cl, axes_cl = plt.subplots(2, 2, figsize=(12, 8))
    fig_cl.suptitle("Clustering — K-means y Jerárquico", fontweight="bold")

    axes_cl[0,0].plot(list(K_range), sil_list, marker="o",
                      color=PALETA[3], lw=2, ms=7)
    axes_cl[0,0].axvline(x=K, color=PALETA[2], ls="--", label=f"K={K}")
    axes_cl[0,0].set_title("Silhouette Score vs K"); axes_cl[0,0].legend()
    axes_cl[0,0].set_xlabel("K"); axes_cl[0,0].set_ylabel("Silhouette")

    axes_cl[0,1].plot(list(K_range), inert_list, marker="s",
                      color=PALETA[0], lw=2, ms=7)
    axes_cl[0,1].axvline(x=K, color=PALETA[2], ls="--", label=f"K={K}")
    axes_cl[0,1].set_title("Método del Codo (Inercia)"); axes_cl[0,1].legend()
    axes_cl[0,1].set_xlabel("K"); axes_cl[0,1].set_ylabel("Inercia")

    axes_cl[1,0].bar([f"C{i+1}" for i in cnt_km.index], cnt_km.values,
                     color=PALETA[:K], edgecolor="white", width=0.6)
    axes_cl[1,0].set_title(f"K-means — Distribución (K={K})")
    axes_cl[1,0].set_ylabel("Registros")
    for i,v in enumerate(cnt_km.values):
        axes_cl[1,0].text(i, v+2, str(v), ha="center", fontsize=9)

    samp = np.random.choice(len(Xc), min(150, len(Xc)), replace=False)
    Z = linkage(Xc[samp], method="ward")
    dendrogram(Z, ax=axes_cl[1,1], no_labels=True,
               above_threshold_color="#B4B2A9", color_threshold=0)
    axes_cl[1,1].set_title("Dendrograma — Jerárquico (muestra)")
    axes_cl[1,1].set_xlabel("Muestras"); axes_cl[1,1].set_ylabel("Distancia Ward")
    thresh = np.sort(Z[:,2])[-(K-1)] if K>1 else Z[:,2].max()
    axes_cl[1,1].axhline(y=thresh, color=PALETA[2], ls="--", label=f"Corte K={K}")
    axes_cl[1,1].legend()

    fig_cl.tight_layout()
    embed_fig(fig_clust_holder, fig_cl)

    # Perfil textual
    perfil = df.groupby("cluster_kmeans")[num_cols].mean().round(2)
    perfil.index = [f"Clúster {i+1}" for i in perfil.index]
    write(clust_text, "MEDIAS POR CLÚSTER\n" + "="*50 + "\n" + perfil.to_string())

    best_k_sil = list(K_range)[sil_list.index(max(sil_list))] if sil_list else K
    q = "buena ✓" if sil_km>0.5 else "moderada" if sil_km>0.25 else "débil ⚠"

    # Generar perfiles dinámicos basados en los datos reales
    col_ppal = num_cols[0] if num_cols else "variable principal"
    perfiles_txt = ""
    for i in perfil.index:
        fila = perfil.loc[i]
        top_col  = fila.idxmax()
        low_col  = fila.idxmin()
        top_val  = fila[top_col]
        low_val  = fila[low_col]
        n_reg    = cnt_km.get(int(i.split()[-1])-1, "?")
        perfiles_txt += f"\n  {i} ({n_reg} registros)\n"
        perfiles_txt += f"    Valor más alto  → {top_col}: {top_val:.2f}\n"
        perfiles_txt += f"    Valor más bajo  → {low_col}: {low_val:.2f}\n"

    write(clust_interp, f"""INTERPRETACIÓN DE CLÚSTERES
{'='*55}

K-means (K={K}):
  Silhouette = {sil_km:.4f} → separación {q}
  K óptimo sugerido por silhouette: K={best_k_sil}

Clustering Jerárquico (Ward, K={K}):
  Silhouette = {sil_agg:.4f}

PERFILES REALES DE CADA CLÚSTER
{'='*55}
(Basados en los valores medios del dataset cargado)
{perfiles_txt}
DIFERENCIA K-means vs Jerárquico
{'='*55}
  K-means: rápido, requiere K previo, asume clústeres esféricos.
  Jerárquico: no requiere K previo, usa dendrograma para
  visualizar agrupaciones; más lento en datasets grandes.
""")
    log(f"✓ Clustering: sil_km={sil_km:.3f}, sil_agg={sil_agg:.3f}")

    # ── Clasificación ─────────────────────────────────────────────────────
    log("→ Entrenando modelos de clasificación...")
    modelos = {
        "Baseline":          DummyClassifier(strategy="most_frequent", random_state=42),
        "Reg. Logística":    LogisticRegression(solver="lbfgs", max_iter=500, random_state=42),
        "KNN":               KNeighborsClassifier(n_neighbors=10),
        "Naive Bayes":       BernoulliNB(),
        "Árbol Decisión":    DecisionTreeClassifier(max_depth=5, random_state=42),
        "Random Forest":     RandomForestClassifier(n_estimators=100, random_state=42),
    }
    # Detectar si es binario o multiclase
    n_clases = len(y.unique())
    es_multi  = n_clases > 2
    avg       = "weighted" if es_multi else "binary"
    log(f"  Clases detectadas: {n_clases} → modo {'multiclase' if es_multi else 'binario'}")

    resultados = {}
    for nombre, modelo in modelos.items():
        try:
            modelo.fit(X_train_sc, y_train)
            yhat    = modelo.predict(X_test_sc)
            yp_full = modelo.predict_proba(X_test_sc)   # shape (n, n_clases)

            # AUC: para multiclase usar ovr+weighted; para binario usar col[:,1]
            if es_multi:
                auc_val = roc_auc_score(y_test, yp_full,
                                        multi_class="ovr", average="weighted", labels=sorted(y_test.unique()))
                yp_plot = yp_full   # guardar matriz completa para ROC
            else:
                auc_val = roc_auc_score(y_test, yp_full[:, 1])
                yp_plot = yp_full[:, 1]

            resultados[nombre] = {
                "modelo":   modelo,
                "y_pred":   yhat,
                "y_proba":  yp_plot,
                "es_multi": es_multi,
                "accuracy": accuracy_score(y_test, yhat),
                "f1":       f1_score(y_test, yhat, average=avg, zero_division=0),
                "auc":      auc_val,
                "precision":precision_score(y_test, yhat, average=avg, zero_division=0),
                "recall":   recall_score(y_test, yhat, average=avg, zero_division=0),
            }
            log(f"  ✓ {nombre}  AUC={auc_val:.3f}")
        except Exception as ex:
            log(f"  ⚠ {nombre}: {ex}")

    state["resultados"] = resultados

    df_res = pd.DataFrame({
        n: {"Accuracy": round(v["accuracy"],4),
            "F1-Score": round(v["f1"],4),
            "AUC":      round(v["auc"],4),
            "Precisión":round(v["precision"],4),
            "Recall":   round(v["recall"],4)}
        for n,v in resultados.items()
    }).T.sort_values("AUC", ascending=False)
    state["df_res"] = df_res

    mejor = df_res.index[0]
    state["mejor_modelo"] = mejor

    for w in metrics_clas_frame.winfo_children(): w.destroy()
    bv = resultados[mejor]
    for lbl3, val3, col3 in [
        ("Mejor modelo",  mejor.split()[0], C_HEADER),
        ("Accuracy",  f"{bv['accuracy']:.3f}",  C_PRIMARY),
        ("F1-Score",  f"{bv['f1']:.3f}",        C_SUCCESS),
        ("AUC",       f"{bv['auc']:.3f}",        C_WARN),
        ("Recall",    f"{bv['recall']:.3f}",     PALETA[3]),
    ]:
        metric_box(metrics_clas_frame, lbl3, val3, col3)

    # Figuras clasificación
    fig_c2, axes_c2 = plt.subplots(1, 2, figsize=(12, 4))
    names_c = list(df_res.index)
    x = np.arange(len(names_c)); w3 = 0.25
    for j, (met, col4) in enumerate([("Accuracy",PALETA[0]),("F1-Score",PALETA[1]),("AUC",PALETA[3])]):
        axes_c2[0].bar(x+j*w3, df_res[met].values, w3,
                       label=met, color=col4, edgecolor="white")
    axes_c2[0].set_xticks(x+w3)
    axes_c2[0].set_xticklabels([n.split()[0] for n in names_c],
                                rotation=15, ha="right", fontsize=8)
    axes_c2[0].set_title("Comparativa de métricas", fontweight="bold")
    axes_c2[0].set_ylim(0, 1.15); axes_c2[0].legend(fontsize=8)

    if "Random Forest" in resultados:
        rf_m = resultados["Random Forest"]["modelo"]
        imp  = pd.Series(rf_m.feature_importances_, index=X.columns
                         ).sort_values(ascending=False).head(8)
        imp.plot(kind="barh", ax=axes_c2[1], color=PALETA[3], edgecolor="white")
        axes_c2[1].invert_yaxis()
        axes_c2[1].set_title("Importancia de variables (RF)", fontweight="bold")

    fig_c2.tight_layout()
    embed_fig(fig_clas_holder, fig_c2)

    write(clas_text, f"""ÁRBOL DE DECISIÓN
{'='*55}
Divide el espacio de features mediante reglas SI/ENTONCES.

  Ejemplo:
    SI duracion_llamada > 300 seg
      Y num_productos > 2  → RESPONDE (clase 1)
    SI NO                  → NO RESPONDE (clase 0)

  Ventajas:
    • Interpretable como reglas de negocio
    • No requiere normalización previa
    • Sirve de base para modelos ensamble

  Parámetro max_depth=5: limita la profundidad para evitar
  sobreajuste y mejorar generalización.

RANDOM FOREST
{'='*55}
Construye {100} árboles sobre subconjuntos aleatorios (Bootstrap)
y promedia sus predicciones (Bagging).

  Ventajas:
    • Reduce varianza → menos sobreajuste que un solo árbol
    • Estable ante datos ruidosos o con outliers
    • Genera ranking de importancia de variables

  n_estimators=100: 100 árboles en paralelo.
  La predicción final es la votación de todos los árboles.

TABLA COMPARATIVA
{'='*55}
{df_res.to_string()}

→ Mejor modelo: {mejor}
""")
    log(f"✓ Clasificación: mejor={mejor} AUC={bv['auc']:.3f}")

    # ── Evaluación / ROC ──────────────────────────────────────────────────
    log("→ Generando curva ROC y tabla...")
    from sklearn.preprocessing import label_binarize
    clases_unicas = sorted(y.unique())

    fig_roc, ax_roc = plt.subplots(figsize=(8, 5))
    for i, (nm, v) in enumerate(resultados.items()):
        try:
            if v["es_multi"]:
                # Curva ROC macro para multiclase: promedio de cada clase vs resto
                y_bin = label_binarize(y_test, classes=clases_unicas)
                yp_m  = v["y_proba"]  # (n, n_clases)
                tpr_all, fpr_all = [], []
                for ci in range(len(clases_unicas)):
                    fpr_c, tpr_c, _ = roc_curve(y_bin[:, ci], yp_m[:, ci])
                    tpr_all.append(np.interp(np.linspace(0,1,100), fpr_c, tpr_c))
                mean_tpr = np.mean(tpr_all, axis=0)
                mean_fpr = np.linspace(0, 1, 100)
                ax_roc.plot(mean_fpr, mean_tpr,
                            label=f"{nm} (AUC={v['auc']:.3f})",
                            color=PALETA[i%len(PALETA)], lw=2)
            else:
                fpr, tpr, _ = roc_curve(y_test, v["y_proba"])
                ax_roc.plot(fpr, tpr,
                            label=f"{nm} (AUC={v['auc']:.3f})",
                            color=PALETA[i%len(PALETA)], lw=2)
        except Exception as ex:
            log(f"  ⚠ ROC {nm}: {ex}")

    ax_roc.plot([0,1],[0,1],"--", color="#B4B2A9", lw=1.5, label="Aleatorio (0.50)")
    ax_roc.set_xlabel("Tasa de Falsos Positivos (FPR)")
    ax_roc.set_ylabel("Tasa de Verdaderos Positivos (TPR)")
    titulo_roc = "Curva ROC macro-promedio (multiclase)" if es_multi else "Curva ROC"
    ax_roc.set_title(titulo_roc, fontweight="bold")
    ax_roc.legend(fontsize=8, loc="lower right")
    fig_roc.tight_layout()
    embed_fig(fig_roc_holder, fig_roc)

    build_table(tabla_frame, df_res, best_row=mejor)

    bv2 = resultados[mejor]
    write(eval_text, f"""RESUMEN EJECUTIVO — COMUNICACIÓN GERENCIAL
{'='*55}

Se evaluaron {len(modelos)} modelos de inteligencia artificial para
predecir qué clientes responderán a la campaña.

Modelo recomendado: {mejor}

  • Identifica correctamente al {bv2['recall']*100:.0f}% de clientes que
    SÍ responderán (Recall = {bv2['recall']:.3f}).

  • Cuando predice "cliente positivo", acierta en el
    {bv2['precision']*100:.0f}% de los casos (Precisión = {bv2['precision']:.3f}).

  • Clasifica correctamente el {bv2['accuracy']*100:.0f}% de los casos
    (Accuracy = {bv2['accuracy']:.3f}).

  • AUC = {bv2['auc']:.3f}: capacidad discriminativa del modelo.
    (0.5 = aleatorio, 1.0 = perfecto)

IMPACTO EN EL NEGOCIO:
  Usar este modelo permite enfocar el presupuesto de marketing
  en clientes con mayor probabilidad de conversión, reduciendo
  costos y mejorando el retorno de inversión (ROI).

RECOMENDACIÓN:
  Desplegar "{mejor}" en producción y monitorear mensualmente
  para detectar degradación del modelo (concept drift).
""")
    log("✓ Evaluación completada")

    # ── Matriz de confusión ───────────────────────────────────────────────
    combo_model["values"] = list(resultados.keys())
    combo_model.set(mejor)
    render_matrix(mejor)

    log("=" * 55)
    log("✅ ANÁLISIS COMPLETO FINALIZADO")
    btn_run.configure(state="normal", text="  ▶  EJECUTAR ANÁLISIS COMPLETO  ")
    show_panel("Partición")

def render_matrix(model_name=None):
    if model_name is None:
        model_name = var_model_sel.get()
    if not model_name or state["resultados"] is None:
        return
    v = state["resultados"].get(model_name)
    if v is None: return

    cm        = confusion_matrix(state["y_test"], v["y_pred"])
    es_multi  = v.get("es_multi", cm.shape[0] > 2)
    clases_str= [str(c) for c in sorted(state["y"].unique())]
    n_cls     = cm.shape[0]

    ann_size = max(7, 14 - n_cls)
    fig_w    = max(10, n_cls * 1.5 + 4)
    fig_h    = max(4,  n_cls * 0.9 + 2)
    fig_cm, axes_cm = plt.subplots(1, 2, figsize=(fig_w, fig_h))
    fig_cm.suptitle(f"Matriz de Confusión — {model_name}", fontweight="bold")

    sns.heatmap(cm, annot=True, fmt="d", ax=axes_cm[0],
                cmap="Blues", linewidths=0.5,
                xticklabels=clases_str,
                yticklabels=clases_str,
                annot_kws={"size": ann_size, "weight": "bold"})
    axes_cm[0].set_xlabel("Predicción")
    axes_cm[0].set_ylabel("Valor Real")
    axes_cm[0].set_xticklabels(axes_cm[0].get_xticklabels(),
                                rotation=30, ha="right", fontsize=8)
    axes_cm[0].set_yticklabels(axes_cm[0].get_yticklabels(),
                                rotation=0, fontsize=8)

    metr_names = ["Accuracy","Precisión (w)","Recall (w)","F1-Score (w)","AUC (OVR)"]
    metr_vals  = [v["accuracy"],v["precision"],v["recall"],v["f1"],v["auc"]]
    colors_bar = [PALETA[0],PALETA[1],PALETA[2],PALETA[3],PALETA[4]]
    axes_cm[1].barh(metr_names, metr_vals, color=colors_bar, edgecolor="white")
    axes_cm[1].set_xlim(0, 1.2)
    axes_cm[1].set_title("Métricas del modelo")
    for i, val5 in enumerate(metr_vals):
        axes_cm[1].text(val5+0.02, i, f"{val5:.3f}", va="center", fontsize=10)

    fig_cm.tight_layout()
    embed_fig(fig_cm_holder, fig_cm)

    # ── Métricas en el panel lateral ─────────────────────────────────────
    for w in cm_metrics_frame.winfo_children(): w.destroy()
    nota_multi = " (prom. weighted)" if es_multi else ""

    if not es_multi and n_cls == 2:
        VN2,FP2,FN2,VP2 = cm.ravel()
        total = cm.sum()
        acc   = (VP2+VN2)/total
        prec  = VP2/(VP2+FP2) if VP2+FP2 else 0
        rec   = VP2/(VP2+FN2) if VP2+FN2 else 0
        spec  = VN2/(VN2+FP2) if VN2+FP2 else 0
        f1v   = 2*prec*rec/(prec+rec) if prec+rec else 0
        fnr   = FN2/(FN2+VP2) if FN2+VP2 else 0
        npv   = VN2/(VN2+FN2) if VN2+FN2 else 0

        data_cm = [
            ("VP (Verdaderos Positivos)", str(VP2), C_SUCCESS),
            ("VN (Verdaderos Negativos)", str(VN2), C_SUCCESS),
            ("FP (Falsos Positivos)",     str(FP2), C_DANGER),
            ("FN (Falsos Negativos)",     str(FN2), C_WARN),
        ]
        for nm2, vl2, cl2 in data_cm:
            f_row = tk.Frame(cm_metrics_frame, bg=C_WHITE)
            f_row.pack(fill="x", pady=1)
            tk.Label(f_row, text=nm2, font=F_BODY, bg=C_WHITE,
                     fg=C_DARK, width=28, anchor="w").pack(side="left")
            tk.Label(f_row, text=vl2, font=("Segoe UI",10,"bold"),
                     bg=C_WHITE, fg=cl2).pack(side="left")

        write(matrix_text, f"""INTERPRETACIÓN DE LA MATRIZ DE CONFUSIÓN
{"="*55}
Modelo: {model_name}

┌─────────────────────┬───────────────┬───────────────┐
│                     │ Pred. POSITIVO│ Pred. NEGATIVO│
├─────────────────────┼───────────────┼───────────────┤
│ Real POSITIVO       │  VP = {VP2:>6}   │  FN = {FN2:>6}   │
├─────────────────────┼───────────────┼───────────────┤
│ Real NEGATIVO       │  FP = {FP2:>6}   │  VN = {VN2:>6}   │
└─────────────────────┴───────────────┴───────────────┘

MÉTRICAS DERIVADAS
{"="*55}
Exactitud  (Accuracy) = (VP+VN)/Total     = {acc:.4f}  ({acc*100:.1f}%)
Precisión             = VP/(VP+FP)        = {prec:.4f}  ({prec*100:.1f}%)
Sensibilidad (Recall) = VP/(VP+FN)        = {rec:.4f}  ({rec*100:.1f}%)
Especificidad         = VN/(VN+FP)        = {spec:.4f}  ({spec*100:.1f}%)
F1-Score              = 2·P·R/(P+R)       = {f1v:.4f}  ({f1v*100:.1f}%)
AUC-ROC               = Área bajo ROC     = {v["auc"]:.4f}
Tasa Falsos Negativos = FN/(FN+VP)        = {fnr:.4f}  ({fnr*100:.1f}%)
Valor Pred. Positivo  = VP/(FP+VP)        = {prec:.4f}  ({prec*100:.1f}%)
Valor Pred. Negativo  = VN/(VN+FN)        = {npv:.4f}  ({npv*100:.1f}%)

INTERPRETACIÓN
{"="*55}
• VP={VP2}: clientes que SÍ responden, identificados correctamente.
• VN={VN2}: clientes que NO responden, descartados correctamente.
• FP={FP2}: Error tipo I — predijo positivo cuando era negativo.
• FN={FN2}: Error tipo II — perdió casos positivos reales.

⚠ Los Falsos Negativos (FN) son más costosos en marketing:
  representan clientes interesados que no se contactaron.
""")
    else:
        # Multiclase — mostrar métricas globales weighted
        acc2  = v["accuracy"]
        f1w   = v["f1"]
        aucw  = v["auc"]
        precw = v["precision"]
        recw  = v["recall"]
        correct = np.diag(cm).sum()
        total2  = cm.sum()

        metricas_show = [
            ("Accuracy (global)",    f"{acc2:.4f}  ({acc2*100:.1f}%)",    C_PRIMARY),
            ("Precisión (weighted)", f"{precw:.4f} ({precw*100:.1f}%)",   C_SUCCESS),
            ("Recall (weighted)",    f"{recw:.4f}  ({recw*100:.1f}%)",    C_WARN),
            ("F1-Score (weighted)",  f"{f1w:.4f}  ({f1w*100:.1f}%)",      PALETA[3]),
            ("AUC OVR (weighted)",   f"{aucw:.4f}",                        C_DANGER),
            ("Total correcto",       f"{correct} / {total2}",              C_SUCCESS),
        ]
        for nm3, vl3, cl3 in metricas_show:
            f_row = tk.Frame(cm_metrics_frame, bg=C_WHITE)
            f_row.pack(fill="x", pady=2)
            tk.Label(f_row, text=nm3, font=F_BODY, bg=C_WHITE,
                     fg=C_DARK, width=26, anchor="w").pack(side="left")
            tk.Label(f_row, text=vl3, font=("Segoe UI",10,"bold"),
                     bg=C_WHITE, fg=cl3).pack(side="left")

        # Precisión por clase
        per_class = ""
        for ci, cls in enumerate(clases_str):
            tp_c = cm[ci, ci]
            fp_c = cm[:, ci].sum() - tp_c
            fn_c = cm[ci, :].sum() - tp_c
            pr_c = tp_c/(tp_c+fp_c) if tp_c+fp_c else 0
            re_c = tp_c/(tp_c+fn_c) if tp_c+fn_c else 0
            f1_c = 2*pr_c*re_c/(pr_c+re_c) if pr_c+re_c else 0
            per_class += f"  {cls:<18} Prec={pr_c:.3f}  Rec={re_c:.3f}  F1={f1_c:.3f}\n"

        write(matrix_text, f"""INTERPRETACIÓN DE LA MATRIZ DE CONFUSIÓN (MULTICLASE)
{"="*55}
Modelo: {model_name}
Clases: {", ".join(clases_str)}

La diagonal principal (↘) representa las predicciones CORRECTAS.
Los valores fuera de la diagonal son errores (confusión entre clases).

MÉTRICAS GLOBALES (promedio weighted)
{"="*55}
Accuracy           = {acc2:.4f}  ({acc2*100:.1f}%)  → corrección global
Precisión weighted = {precw:.4f}  ({precw*100:.1f}%)  → VP/(VP+FP) por clase
Recall weighted    = {recw:.4f}  ({recw*100:.1f}%)  → VP/(VP+FN) por clase
F1-Score weighted  = {f1w:.4f}  ({f1w*100:.1f}%)  → media armónica P/R
AUC OVR weighted   = {aucw:.4f}            → discriminación multiclase

MÉTRICAS POR CLASE
{"="*55}
  Clase              Precisión    Recall       F1-Score
{per_class}
INTERPRETACIÓN
{"="*55}
• Las métricas "weighted" ponderan por el soporte (tamaño) de cada clase.
• Una clase con F1 bajo indica confusión frecuente con otra clase similar.
• Revisar las filas/columnas con más errores en el heatmap para identificar
  qué clases el modelo confunde con mayor frecuencia.
• Para campañas: priorizar las clases con mayor Recall para no perder
  los casos más importantes del negocio.
""")


combo_model.bind("<<ComboboxSelected>>", lambda e: render_matrix())
btn_run.configure(command=run_analysis)

# ═════════════════════════════════════════════════════════════════════════════
#  ARRANCAR
# ═════════════════════════════════════════════════════════════════════════════
show_panel("Datos")
root.mainloop()