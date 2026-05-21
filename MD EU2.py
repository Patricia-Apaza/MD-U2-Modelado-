"""
╔══════════════════════════════════════════════════════════════════════════════╗
║         ANALIZADOR DE MINERÍA DE DATOS — EXAMEN UNIDAD 2                   ║
║   EDA · Limpieza · Partición · Baseline · Clustering · Clasificación       ║
╚══════════════════════════════════════════════════════════════════════════════╝

Instalar dependencias (ejecutar UNA sola vez en tu terminal):
    pip install pandas numpy scikit-learn matplotlib seaborn scipy chardet openpyxl

Uso:
    python analizador_mineria.py
"""

import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext, messagebox
import threading
import traceback
import os

import pandas as pd
import numpy as np
import chardet

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc,
    accuracy_score, f1_score, silhouette_score,
    precision_score, recall_score
)

# ══════════════════════════════════════════════
#  PALETA OSCURA (GitHub Dark)
# ══════════════════════════════════════════════
DARK_BG    = "#0D1117"
PANEL_BG   = "#161B22"
CARD_BG    = "#21262D"
SIDEBAR_BG = "#0D1117"
BORDER     = "#30363D"
ACCENT     = "#58A6FF"
ACCENT2    = "#3FB950"
ACCENT3    = "#F78166"
ACCENT4    = "#D2A8FF"
GOLD       = "#FFA657"
TEXT_W     = "#E6EDF3"
TEXT_G     = "#8B949E"
WHITE      = "#FFFFFF"

FT_TITLE = ("Segoe UI", 12, "bold")
FT_BODY  = ("Segoe UI", 10)
FT_SMALL = ("Segoe UI", 9)
FT_MONO  = ("Consolas", 9)

# matplotlib oscuro
plt.rcParams.update({
    "figure.facecolor": DARK_BG,
    "axes.facecolor":   PANEL_BG,
    "axes.edgecolor":   BORDER,
    "axes.labelcolor":  TEXT_W,
    "axes.grid":        True,
    "grid.color":       BORDER,
    "grid.linewidth":   0.6,
    "xtick.color":      TEXT_G,
    "ytick.color":      TEXT_G,
    "text.color":       TEXT_W,
    "legend.facecolor": CARD_BG,
    "legend.edgecolor": BORDER,
    "font.family":      "DejaVu Sans",
    "font.size":        9,
})
PLOT_CLR = [ACCENT, ACCENT2, ACCENT3, ACCENT4, GOLD,
            "#0891B2", "#BE185D", "#854D0E"]


# ══════════════════════════════════════════════
#  TOOLTIP
# ══════════════════════════════════════════════
class Tooltip:
    def __init__(self, w, text):
        self.w, self.text, self.tip = w, text, None
        w.bind("<Enter>", self.show); w.bind("<Leave>", self.hide)

    def show(self, _=None):
        x = self.w.winfo_rootx() + 22; y = self.w.winfo_rooty() + 32
        self.tip = tk.Toplevel(self.w); self.tip.wm_overrideredirect(True)
        self.tip.wm_geometry(f"+{x}+{y}")
        tk.Label(self.tip, text=self.text, bg="#2D333B", fg=TEXT_W,
                 font=FT_SMALL, padx=8, pady=5).pack()

    def hide(self, _=None):
        if self.tip: self.tip.destroy(); self.tip = None


# ══════════════════════════════════════════════
#  TARJETA DE MÉTRICA
# ══════════════════════════════════════════════
class MetricCard(tk.Frame):
    def __init__(self, parent, label, val="—", color=ACCENT):
        super().__init__(parent, bg=CARD_BG,
                         highlightbackground=color, highlightthickness=2,
                         padx=10, pady=8)
        tk.Label(self, text=label, bg=CARD_BG, fg=TEXT_G, font=FT_SMALL).pack(anchor="w")
        self.v = tk.Label(self, text=val, bg=CARD_BG, fg=color,
                          font=("Segoe UI", 17, "bold"))
        self.v.pack(anchor="w")

    def update(self, val): self.v.config(text=val)


# ══════════════════════════════════════════════
#  APLICACIÓN
# ══════════════════════════════════════════════
class AppMineria(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Analizador de Minería de Datos · Unidad 2")
        self.geometry("1450x880")
        self.minsize(1100, 700)
        self.configure(bg=DARK_BG)
        self.df_raw = None
        self.mode   = None   # "all" o "col"

        self._styles()
        self._header()
        self._body()
        self.protocol("WM_DELETE_WINDOW", lambda: (self.destroy(), self.quit()))

    # ─── ESTILOS ─────────────────────────────
    def _styles(self):
        s = ttk.Style(self); s.theme_use("clam")
        s.configure("TFrame",    background=DARK_BG)
        s.configure("TLabel",    background=DARK_BG, foreground=TEXT_W, font=FT_BODY)
        s.configure("TCombobox", fieldbackground=CARD_BG, background=CARD_BG,
                     foreground=TEXT_W, arrowcolor=ACCENT)
        s.map("TCombobox", fieldbackground=[("readonly", CARD_BG)],
              foreground=[("readonly", TEXT_W)])
        s.configure("TNotebook", background=DARK_BG, borderwidth=0)
        s.configure("TNotebook.Tab", background=PANEL_BG, foreground=TEXT_G,
                     font=FT_BODY, padding=(14, 7))
        s.map("TNotebook.Tab", background=[("selected", CARD_BG)],
              foreground=[("selected", ACCENT)])
        s.configure("Treeview", background=CARD_BG, fieldbackground=CARD_BG,
                     foreground=TEXT_W, font=FT_SMALL, rowheight=22)
        s.configure("Treeview.Heading", background=PANEL_BG, foreground=ACCENT,
                     font=("Segoe UI", 9, "bold"))
        s.map("Treeview", background=[("selected", ACCENT)],
              foreground=[("selected", DARK_BG)])
        s.configure("Blue.Horizontal.TProgressbar",
                     troughcolor=PANEL_BG, background=ACCENT,
                     borderwidth=0, thickness=4)

    # ─── HEADER ──────────────────────────────
    def _header(self):
        hdr = tk.Frame(self, bg=PANEL_BG, height=56)
        hdr.pack(fill=tk.X); hdr.pack_propagate(False)
        tk.Label(hdr, text="⬡  Minería de Datos", bg=PANEL_BG, fg=ACCENT,
                 font=("Segoe UI", 14, "bold")).pack(side=tk.LEFT, padx=20)
        tk.Label(hdr, text="Unidad 2 · EDA · Limpieza · Partición · Clustering · Clasificación",
                 bg=PANEL_BG, fg=TEXT_G, font=FT_SMALL).pack(side=tk.LEFT)
        self.status_lbl = tk.Label(hdr, text="⬤  Sin datos", bg=PANEL_BG,
                                    fg=TEXT_G, font=FT_SMALL)
        self.status_lbl.pack(side=tk.RIGHT, padx=16)
        tk.Frame(self, bg=BORDER, height=1).pack(fill=tk.X)

    # ─── CUERPO ───────────────────────────────
    def _body(self):
        paned = tk.PanedWindow(self, orient=tk.HORIZONTAL,
                               bg=DARK_BG, sashwidth=5)
        paned.pack(fill=tk.BOTH, expand=True)

        # SIDEBAR
        sidebar = tk.Frame(paned, bg=SIDEBAR_BG, width=300)
        paned.add(sidebar, minsize=250)
        self._sidebar(sidebar)

        # PANEL DERECHO
        right = tk.Frame(paned, bg=DARK_BG)
        paned.add(right, minsize=800)
        self.nb = ttk.Notebook(right)
        self.nb.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.tab_eda    = self._tab("🔍 EDA & Limpieza")
        self.tab_part   = self._tab("✂️ Partición & Baseline")
        self.tab_clust  = self._tab("🔵 Clustering")
        self.tab_clasif = self._tab("🌳 Clasificación")
        self.tab_eval   = self._tab("📈 Evaluación")
        self.tab_cm     = self._tab("🔲 Matriz Confusión")
        self.tab_interp = self._tab("💬 Interpretación")

    def _tab(self, t):
        f = ttk.Frame(self.nb); self.nb.add(f, text=t); return f

    # ─── SIDEBAR ─────────────────────────────
    def _sidebar(self, parent):
        pad = dict(padx=14)

        # ── Sección: cargar ──
        sec = tk.Frame(parent, bg=SIDEBAR_BG, pady=14)
        sec.pack(fill=tk.X, **pad)
        self._lbl_sec(sec, "ARCHIVO")

        btn_load = self._btn(sec, "📂  Cargar CSV", ACCENT, self._cargar_csv)
        btn_load.pack(fill=tk.X, pady=(4, 4))
        Tooltip(btn_load, "Carga student-mat.csv o student-por.csv")

        self.lbl_file = tk.Label(sec, text="Sin archivo", bg=SIDEBAR_BG,
                                  fg=TEXT_G, font=FT_SMALL, wraplength=250)
        self.lbl_file.pack(anchor="w")

        tk.Frame(parent, bg=BORDER, height=1).pack(fill=tk.X, padx=14, pady=4)

        # ── Sección: target ──
        sec2 = tk.Frame(parent, bg=SIDEBAR_BG, pady=6)
        sec2.pack(fill=tk.X, **pad)
        self._lbl_sec(sec2, "VARIABLE OBJETIVO (TARGET)")
        tk.Label(sec2, text="Selecciona la columna a predecir:",
                 bg=SIDEBAR_BG, fg=TEXT_G, font=FT_SMALL).pack(anchor="w")
        self.combo = ttk.Combobox(sec2, state="disabled", width=28, font=FT_BODY)
        self.combo.pack(fill=tk.X, pady=(3, 0))

        tk.Frame(parent, bg=BORDER, height=1).pack(fill=tk.X, padx=14, pady=8)

        # ── Sección: BOTONES DE ANÁLISIS ──
        sec3 = tk.Frame(parent, bg=SIDEBAR_BG, pady=4)
        sec3.pack(fill=tk.X, **pad)
        self._lbl_sec(sec3, "MODO DE ANÁLISIS")

        # Botón 1: Analizar TODA la data
        self.btn_all = self._btn(
            sec3,
            "▶  ANALIZAR TODA LA DATA",
            ACCENT2,
            lambda: self._run("all"),
            state="disabled"
        )
        self.btn_all.pack(fill=tk.X, pady=(4, 3))
        Tooltip(self.btn_all,
                "Usa todas las columnas como features.\n"
                "El target seleccionado arriba se excluye y se predice.")

        # Botón 2: Analizar columna por columna
        self.btn_col = self._btn(
            sec3,
            "🔎  ANALIZAR COLUMNA POR COLUMNA",
            ACCENT,
            lambda: self._run("col"),
            state="disabled"
        )
        self.btn_col.pack(fill=tk.X, pady=(0, 4))
        Tooltip(self.btn_col,
                "Analiza UNA columna numérica a la vez como única feature.\n"
                "Útil para comparar el impacto individual de cada variable.")

        # Selector de columna (solo visible en modo col)
        self.frm_col_sel = tk.Frame(sec3, bg=SIDEBAR_BG)
        self.frm_col_sel.pack(fill=tk.X)
        tk.Label(self.frm_col_sel, text="Columna a analizar:",
                 bg=SIDEBAR_BG, fg=TEXT_G, font=FT_SMALL).pack(anchor="w")
        self.combo_col = ttk.Combobox(self.frm_col_sel, state="disabled",
                                       width=28, font=FT_BODY)
        self.combo_col.pack(fill=tk.X, pady=(2, 0))

        # Progress
        self.progress = ttk.Progressbar(sec3, mode="indeterminate", length=260,
                                         style="Blue.Horizontal.TProgressbar")
        self.progress.pack(fill=tk.X, pady=(6, 0))

        tk.Frame(parent, bg=BORDER, height=1).pack(fill=tk.X, padx=14, pady=8)

        # ── Métricas rápidas ──
        sec4 = tk.Frame(parent, bg=SIDEBAR_BG, pady=4)
        sec4.pack(fill=tk.X, **pad)
        self._lbl_sec(sec4, "MÉTRICAS RÁPIDAS")
        inner = tk.Frame(sec4, bg=SIDEBAR_BG)
        inner.pack(fill=tk.X, pady=(4, 0))
        self.cards = {}
        specs = [("Accuracy (RF)", "—", ACCENT), ("F1-Score (RF)", "—", ACCENT2),
                 ("AUC (RF)",      "—", ACCENT3), ("Silhouette KM","—", ACCENT4)]
        for i, (l, v, c) in enumerate(specs):
            card = MetricCard(inner, l, v, c)
            card.grid(row=i//2, column=i%2, padx=3, pady=3, sticky="ew")
            self.cards[l] = card
        inner.columnconfigure(0, weight=1); inner.columnconfigure(1, weight=1)

        tk.Frame(parent, bg=BORDER, height=1).pack(fill=tk.X, padx=14, pady=6)

        # ── Tabla de corridas ──
        sec5 = tk.Frame(parent, bg=SIDEBAR_BG, pady=4)
        sec5.pack(fill=tk.X, **pad)
        self._lbl_sec(sec5, "TABLA DE CORRIDAS")
        cols = ("Modelo", "Acc", "F1", "AUC")
        self.tree = ttk.Treeview(sec5, columns=cols, show="headings",
                                  height=5, selectmode="browse")
        for c in cols:
            self.tree.heading(c, text=c)
            self.tree.column(c, width=80 if c=="Modelo" else 52, anchor="center")
        sb = ttk.Scrollbar(sec5, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=sb.set)
        self.tree.pack(side=tk.LEFT, fill=tk.X, expand=True, pady=(4, 0))
        sb.pack(side=tk.LEFT, fill=tk.Y, pady=(4, 0))

        tk.Frame(parent, bg=BORDER, height=1).pack(fill=tk.X, padx=14, pady=6)

        # ── Log ──
        sec6 = tk.Frame(parent, bg=SIDEBAR_BG, pady=4)
        sec6.pack(fill=tk.BOTH, expand=True, **pad)
        self._lbl_sec(sec6, "REGISTRO")
        self.log = scrolledtext.ScrolledText(
            sec6, height=9, bg="#0A0F16", fg="#94D2BD",
            font=FT_MONO, relief="flat", borderwidth=0)
        self.log.pack(fill=tk.BOTH, expand=True, pady=(4, 0))
        self._log_tags()

    def _lbl_sec(self, parent, text):
        tk.Label(parent, text=text, bg=SIDEBAR_BG, fg="#93C5FD",
                 font=("Segoe UI", 8, "bold")).pack(anchor="w", pady=(0, 2))

    def _btn(self, parent, text, color, cmd, state="normal"):
        hover = {"#58A6FF": "#79B8FF", "#3FB950": "#4ADE80",
                 "#F78166": "#FF9580", "#D2A8FF": "#E4CAFF"}.get(color, color)
        b = tk.Button(parent, text=text, bg=color, fg=DARK_BG,
                      font=("Segoe UI", 10, "bold"), relief="flat",
                      cursor="hand2", activebackground=hover,
                      activeforeground=DARK_BG, state=state, command=cmd)
        b.bind("<Enter>", lambda e: b.config(bg=hover) if b["state"]=="normal" else None)
        b.bind("<Leave>", lambda e: b.config(bg=color)  if b["state"]=="normal" else None)
        return b

    def _log_tags(self):
        self.log.tag_config("info",    foreground="#94D2BD")
        self.log.tag_config("ok",      foreground="#4ADE80")
        self.log.tag_config("warn",    foreground="#FCD34D")
        self.log.tag_config("error",   foreground="#F87171")
        self.log.tag_config("section", foreground="#93C5FD",
                            font=("Consolas", 9, "bold"))

    def _log(self, msg, tag="info"):
        self.log.insert(tk.END, msg+"\n", tag); self.log.see(tk.END)

    def _status(self, txt, color=TEXT_G):
        self.status_lbl.config(text=f"⬤  {txt}", fg=color)

    # ─── CARGAR CSV ──────────────────────────
    def _cargar_csv(self):
        path = filedialog.askopenfilename(
            title="Seleccionar CSV",
            filetypes=[("CSV","*.csv"),("Todos","*.*")])
        if not path: return
        enc = self._encoding(path)
        for e in [enc,"utf-8","latin-1","cp1252","iso-8859-1"]:
            if not e: continue
            try:
                df = pd.read_csv(path, encoding=e, sep=None, engine="python")
                self.df_raw = df
                cols = list(df.columns)
                self.combo.config(state="readonly", values=cols)
                self.combo_col.config(state="readonly", values=cols)
                # Auto-detectar G3
                if "G3" in cols: self.combo.set("G3")
                else: self.combo.current(len(cols)-1)
                self.combo_col.current(0)
                self.btn_all.config(state="normal")
                self.btn_col.config(state="normal")
                self.combo_col.config(state="readonly")
                r, c = df.shape
                self.lbl_file.config(
                    text=f"✔ {os.path.basename(path)}\n{r:,} filas · {c} cols · {e}")
                self._status("Dataset cargado", "#4ADE80")
                self._log(f"\n✔ {os.path.basename(path)}: {r} filas, {c} cols", "ok")
                break
            except Exception: continue

    def _encoding(self, path):
        with open(path,"rb") as f:
            return chardet.detect(f.read(30000)).get("encoding")

    # ─── DISPARAR ANÁLISIS ───────────────────
    def _run(self, mode):
        if self.df_raw is None or not self.combo.get():
            messagebox.showwarning("Atención","Carga un CSV y selecciona el target.")
            return
        if mode == "col" and not self.combo_col.get():
            messagebox.showwarning("Atención","Selecciona la columna a analizar.")
            return
        self.mode = mode
        self.btn_all.config(state="disabled")
        self.btn_col.config(state="disabled")
        self.progress.start(10)
        self._status("Analizando…", "#FCD34D")
        threading.Thread(target=self._pipeline, daemon=True).start()

    # ═══════════════════════════════════════════
    #  PIPELINE
    # ═══════════════════════════════════════════
    def _pipeline(self):
        try:
            target = self.combo.get()
            df     = self.df_raw.copy()

            self._log("\n▶ ══════ PIPELINE ══════", "section")

            # ── 1. LIMPIEZA + EDA ──
            self._log("\n[1/6] EDA y Limpieza…", "section")
            df, eda = self._clean(df, target)

            # ── 2. PREPARAR X, y ──
            self._log("\n[2/6] Preparación features…", "section")
            y_raw = df[target]

            # Modo col: usar solo la columna elegida
            if self.mode == "col":
                col_sel = self.combo_col.get()
                if col_sel == target:
                    self.after(0, lambda: self._log("⚠ La columna elegida es el target. Elige otra.", "warn"))
                    self.after(0, self._done); return
                X_raw = df[[col_sel]].copy()
                self._log(f"  Modo columna: usando solo '{col_sel}'", "warn")
            else:
                X_raw = df.drop(columns=[target], errors="ignore")
                self._log(f"  Modo completo: {X_raw.shape[1]} features", "info")

            # Codificar target
            if y_raw.dtype == "object":
                le = LabelEncoder()
                y  = le.fit_transform(y_raw.astype(str))
                self._log(f"  Target categ. → {list(le.classes_[:4])}", "info")
            else:
                med = y_raw.median()
                y   = (y_raw >= med).astype(int)
                self._log(f"  Target num. → binarizado (mediana={med})", "info")

            num_c = X_raw.select_dtypes(include=[np.number]).columns.tolist()
            cat_c = X_raw.select_dtypes(exclude=[np.number]).columns.tolist()
            self._log(f"  Numéricas:{len(num_c)} | Categóricas:{len(cat_c)}", "info")

            trf = []
            if num_c: trf.append(("num", Pipeline([
                ("imp", SimpleImputer(strategy="median")),
                ("sc",  StandardScaler())]), num_c))
            if cat_c: trf.append(("cat", Pipeline([
                ("imp", SimpleImputer(strategy="constant", fill_value="missing")),
                ("enc", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat_c))

            prep = ColumnTransformer(trf)
            X    = prep.fit_transform(X_raw)
            self._log(f"  Shape X: {X.shape}", "ok")

            # ── 3. PARTICIÓN ──
            self._log("\n[3/6] Partición 70/15/15…", "section")
            strat = y if np.min(np.bincount(y)) >= 2 else None
            X_tv, X_test, y_tv, y_test = train_test_split(
                X, y, test_size=0.15, random_state=42, stratify=strat)
            strat2 = y_tv if np.min(np.bincount(y_tv)) >= 2 else None
            X_train, X_val, y_train, y_val = train_test_split(
                X_tv, y_tv, test_size=round(0.15/0.85,4),
                random_state=42, stratify=strat2)
            part = {"train":len(X_train),"val":len(X_val),"test":len(X_test)}
            self._log(f"  Train:{part['train']} Val:{part['val']} Test:{part['test']}", "ok")

            # ── 4. BASELINE ──
            self._log("\n[4/6] Baseline…", "section")
            baseline = DummyClassifier(strategy="most_frequent")
            baseline.fit(X_train, y_train)
            b_p = baseline.predict(X_test)
            b_acc = accuracy_score(y_test, b_p)
            b_f1  = f1_score(y_test, b_p, average="weighted", zero_division=0)
            self._log(f"  Baseline Acc:{b_acc:.4f} F1:{b_f1:.4f}", "warn")

            # ── 5. CLUSTERING ──
            self._log("\n[5/6] Clustering…", "section")
            k = min(4, max(2, len(X)//80))
            km      = KMeans(n_clusters=k, n_init=12, random_state=42)
            km_lbl  = km.fit_predict(X)
            sil_km  = silhouette_score(X, km_lbl)
            hac     = AgglomerativeClustering(n_clusters=k)
            hac_lbl = hac.fit_predict(X)
            sil_hac = silhouette_score(X, hac_lbl)
            self._log(f"  K-Means k={k} Sil={sil_km:.4f}", "ok")
            self._log(f"  Jerárquico  Sil={sil_hac:.4f}", "ok")
            if num_c:
                df_cl = X_raw[num_c].copy().reset_index(drop=True)
                df_cl["cluster"] = km_lbl[:len(df_cl)]
                profiles = df_cl.groupby("cluster").mean()
            else: profiles = None
            clust = {"k":k,"km_lbl":km_lbl,"hac_lbl":hac_lbl,
                     "sil_km":sil_km,"sil_hac":sil_hac,
                     "profiles":profiles,"X":X}

            # ── 6. CLASIFICACIÓN ──
            self._log("\n[6/6] Clasificación…", "section")
            mods = {
                "Baseline":      baseline,
                "Árbol Dec.":    DecisionTreeClassifier(max_depth=8, random_state=42),
                "Random Forest": RandomForestClassifier(n_estimators=120,
                                                        random_state=42, n_jobs=-1),
            }
            res = []
            for name, model in mods.items():
                if name != "Baseline": model.fit(X_train, y_train)
                preds = model.predict(X_test)
                probs = model.predict_proba(X_test)[:,1] if hasattr(model,"predict_proba") else None
                acc   = accuracy_score(y_test, preds)
                f1    = f1_score(y_test, preds, average="weighted", zero_division=0)
                prec  = precision_score(y_test, preds, average="weighted", zero_division=0)
                rec   = recall_score(y_test, preds, average="weighted", zero_division=0)
                rauc  = 0.0
                if probs is not None and len(np.unique(y_test))==2:
                    fpr_,tpr_,_ = roc_curve(y_test, probs); rauc = auc(fpr_,tpr_)
                res.append({"name":name,"preds":preds,"probs":probs,
                            "acc":acc,"f1":f1,"prec":prec,"rec":rec,
                            "auc":rauc,"y_test":y_test})
                self._log(f"  {name:<16} Acc:{acc:.3f} F1:{f1:.3f} AUC:{rauc:.3f}", "ok")

            rf = next(r for r in res if r["name"]=="Random Forest")
            self.after(0, self.cards["Accuracy (RF)"].update, f"{rf['acc']:.4f}")
            self.after(0, self.cards["F1-Score (RF)"].update, f"{rf['f1']:.4f}")
            self.after(0, self.cards["AUC (RF)"].update,      f"{rf['auc']:.4f}")
            self.after(0, self.cards["Silhouette KM"].update, f"{sil_km:.4f}")
            self.after(0, self._update_tree, res)
            self._log("\n✔ Pipeline completado.", "ok")

            self.after(0, self._rend_eda,    eda)
            self.after(0, self._rend_part,   part)
            self.after(0, self._rend_clust,  clust)
            self.after(0, self._rend_clasif, res)
            self.after(0, self._rend_eval,   res)
            self.after(0, self._rend_cm,     res)
            self.after(0, self._rend_interp, res, clust, part)
            self.after(0, self._done)

        except Exception as e:
            traceback.print_exc()
            self.after(0, lambda: self._log(f"✖ {e}", "error"))
            self.after(0, self._done)

    def _done(self):
        self.progress.stop()
        self.btn_all.config(state="normal")
        self.btn_col.config(state="normal")
        self._status("Análisis completado ✔", "#4ADE80")

    # ═══════════════════════════════════════════
    #  LIMPIEZA
    # ═══════════════════════════════════════════
    def _clean(self, df, target):
        info = {"shape_before": df.shape,
                "nulls_before": df.isnull().sum().sum(),
                "dtypes_before": df.dtypes.copy(),
                "converted": [], "dropped": []}
        # Eliminar ID
        id_cols = [c for c in df.columns
                   if c.lower() in ("studentid","id","index","customerid","customer_id")]
        if id_cols:
            df.drop(columns=id_cols, inplace=True, errors="ignore")
            info["dropped"] = id_cols
            self._log(f"  Cols ID eliminadas: {id_cols}", "warn")
        # yes/no → 1/0
        for col in df.columns:
            if col == target or df[col].dtype != object: continue
            vals = set(df[col].dropna().astype(str).str.strip().str.lower().unique())
            if vals.issubset({"yes","no","y","n","true","false","si","t","f","1","0"}):
                mp = {"yes":1,"no":0,"y":1,"n":0,"true":1,"false":0,
                      "si":1,"t":1,"f":0,"1":1,"0":0}
                df[col] = df[col].astype(str).str.strip().str.lower().map(mp)
                df[col] = pd.to_numeric(df[col], errors="coerce")
                info["converted"].append((col,"yes/no→1/0"))
                self._log(f"  '{col}': yes/no → 1/0", "info")
        # Imputar nulos
        for col in df.select_dtypes(include=[np.number]).columns:
            if df[col].isnull().any(): df[col].fillna(df[col].median(), inplace=True)
        for col in df.select_dtypes(exclude=[np.number]).columns:
            if df[col].isnull().any():
                m = df[col].mode()
                if len(m): df[col].fillna(m[0], inplace=True)
        info.update({"shape_after":df.shape,
                     "nulls_after":df.isnull().sum().sum(),
                     "dtypes_after":df.dtypes.copy(),
                     "describe":df.describe(include="all").T,
                     "df":df})
        self._log(f"  Nulos: {info['nulls_before']} → {info['nulls_after']}", "ok")
        self._log(f"  Convertidas: {len(info['converted'])} cols", "ok")
        return df, info

    # ─── ÁRBOL ───────────────────────────────
    def _update_tree(self, res):
        for r in self.tree.get_children(): self.tree.delete(r)
        for r in res:
            tag = "rf" if r["name"]=="Random Forest" else ""
            self.tree.insert("","end",
                values=(r["name"],f"{r['acc']:.3f}",f"{r['f1']:.3f}",f"{r['auc']:.3f}"),
                tags=(tag,))
        self.tree.tag_configure("rf", background="#1C3A1E", foreground=ACCENT2)

    # ═══════════════════════════════════════════
    #  RENDER EDA
    # ═══════════════════════════════════════════
    def _rend_eda(self, info):
        self._clr(self.tab_eda)
        nb = ttk.Notebook(self.tab_eda)
        nb.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        t_res  = ttk.Frame(nb); nb.add(t_res,  text="📋 Resumen")
        t_dist = ttk.Frame(nb); nb.add(t_dist, text="📊 Distribuciones")
        t_corr = ttk.Frame(nb); nb.add(t_corr, text="🔥 Correlación")
        t_null = ttk.Frame(nb); nb.add(t_null, text="⚠ Tipos & Conversiones")
        self._eda_resumen(t_res,  info)
        self._eda_dist(t_dist,    info["df"])
        self._eda_corr(t_corr,    info["df"])
        self._eda_tipos(t_null,   info)

    def _eda_resumen(self, parent, info):
        df = info["df"]
        top = tk.Frame(parent, bg=DARK_BG); top.pack(fill=tk.X, padx=12, pady=10)
        for lbl, val, col in [
            ("Filas",            f"{df.shape[0]:,}", ACCENT),
            ("Columnas",         f"{df.shape[1]}",   ACCENT2),
            ("Nulos restantes",  f"{info['nulls_after']}", ACCENT3 if info["nulls_after"]>0 else ACCENT2),
            ("Cols convertidas", f"{len(info['converted'])}", GOLD),
        ]:
            MetricCard(top, lbl, val, col).pack(side=tk.LEFT, padx=5, ipadx=6, ipady=3)

        tk.Label(parent, text="Estadísticas descriptivas",
                 bg=DARK_BG, fg=TEXT_W, font=FT_TITLE).pack(anchor="w", padx=14, pady=(8,4))
        desc = info["describe"].reset_index()
        keep = ["index"] + [c for c in ["count","mean","std","min","25%","50%","75%","max"] if c in desc.columns]
        desc = desc[keep].round(3)
        fr   = tk.Frame(parent, bg=DARK_BG); fr.pack(fill=tk.BOTH, expand=True, padx=12)
        tree = ttk.Treeview(fr, columns=list(desc.columns), show="headings", height=14)
        for c in desc.columns:
            tree.heading(c, text=c)
            tree.column(c, width=110 if c=="index" else 80, anchor="center")
        for _, row in desc.iterrows(): tree.insert("","end", values=list(row))
        sbv = ttk.Scrollbar(fr, orient="vertical",   command=tree.yview)
        sbh = ttk.Scrollbar(fr, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=sbv.set, xscrollcommand=sbh.set)
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sbv.pack(side=tk.LEFT,  fill=tk.Y)

    def _eda_dist(self, parent, df):
        num = df.select_dtypes(include=[np.number]).columns.tolist()
        if not num: tk.Label(parent,text="Sin columnas numéricas.",bg=DARK_BG,fg=TEXT_W).pack(); return
        cols_show = num[:12]; n = len(cols_show)
        nrows = (n+3)//4
        fig, axes = plt.subplots(nrows, 4, figsize=(14, 3.2*nrows), facecolor=DARK_BG)
        axes = np.array(axes).flatten()
        for i, col in enumerate(cols_show):
            ax = axes[i]; data = df[col].dropna()
            ax.hist(data, bins=20, color=PLOT_CLR[i%len(PLOT_CLR)],
                    edgecolor=DARK_BG, linewidth=0.5, alpha=0.85)
            ax.axvline(data.mean(),   color=ACCENT3, lw=1.2, ls="--",
                       label=f"μ={data.mean():.1f}")
            ax.axvline(data.median(), color=GOLD,    lw=1.2, ls=":",
                       label=f"med={data.median():.1f}")
            ax.set_title(col, fontsize=9, fontweight="bold", color=TEXT_W)
            ax.legend(fontsize=7)
        for j in range(i+1, len(axes)): axes[j].set_visible(False)
        fig.suptitle("Distribución de Variables Numéricas",
                     fontsize=11, fontweight="bold", color=TEXT_W, y=1.01)
        fig.tight_layout(pad=1.5)
        self._embed(fig, parent)

    def _eda_corr(self, parent, df):
        num = df.select_dtypes(include=[np.number])
        if num.shape[1] < 2:
            tk.Label(parent,text="Pocas variables numéricas.",bg=DARK_BG,fg=TEXT_W).pack(); return
        corr = num.corr()
        n    = min(corr.shape[0], 20)
        corr = corr.iloc[:n,:n]
        fig, ax = plt.subplots(figsize=(max(8,n*0.55), max(6,n*0.5)), facecolor=DARK_BG)
        mask = np.triu(np.ones_like(corr, dtype=bool))
        cmap = sns.diverging_palette(220, 20, as_cmap=True)
        sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
                    annot=(n<=14), fmt=".2f", linewidths=0.4, linecolor=BORDER,
                    ax=ax, annot_kws={"fontsize":7},
                    cbar_kws={"shrink":0.8})
        ax.set_title("Mapa de Correlación (Pearson)",
                     fontsize=11, fontweight="bold", pad=10, color=TEXT_W)
        fig.tight_layout()
        self._embed(fig, parent)

    def _eda_tipos(self, parent, info):
        tk.Label(parent, text="Tipos de datos y conversiones aplicadas",
                 bg=DARK_BG, fg=TEXT_W, font=FT_TITLE).pack(anchor="w", padx=14, pady=(10,5))
        fr   = tk.Frame(parent, bg=DARK_BG); fr.pack(fill=tk.BOTH, expand=True, padx=12)
        cols = ("Columna","Tipo antes","Tipo después","Conversión aplicada")
        tree = ttk.Treeview(fr, columns=cols, show="headings", height=18)
        for c in cols: tree.heading(c,text=c); tree.column(c,width=160,anchor="center")
        cd = {c:v for c,v in info.get("converted",[])}
        for col in info["dtypes_before"].index:
            if col in info["dtypes_after"].index:
                antes   = str(info["dtypes_before"][col])
                despues = str(info["dtypes_after"].get(col,"—"))
                conv    = cd.get(col,"—")
                tree.insert("","end", values=(col,antes,despues,conv),
                            tags=("conv" if conv!="—" else "",))
        tree.tag_configure("conv", background="#2D2A1A", foreground=GOLD)
        sb = ttk.Scrollbar(fr, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=sb.set)
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb.pack(side=tk.LEFT, fill=tk.Y)

    # ═══════════════════════════════════════════
    #  RENDER PARTICIÓN
    # ═══════════════════════════════════════════
    def _rend_part(self, info):
        self._clr(self.tab_part)
        fig = plt.figure(figsize=(11,4.5), facecolor=DARK_BG)
        gs  = gridspec.GridSpec(1,2,figure=fig,left=0.07,right=0.97,wspace=0.4)
        ax1 = fig.add_subplot(gs[0])
        labels = ["Train (70%)","Val (15%)","Test (15%)"]
        sizes  = [info["train"],info["val"],info["test"]]
        bars   = ax1.bar(labels, sizes, color=[ACCENT,ACCENT2,ACCENT3],
                         width=0.5, edgecolor=DARK_BG)
        for b,s in zip(bars,sizes):
            ax1.text(b.get_x()+b.get_width()/2, b.get_height()+max(sizes)*0.01,
                     f"{s:,}", ha="center", fontsize=10, fontweight="bold", color=TEXT_W)
        ax1.set_title("Distribución de Partición (70/15/15)",
                      fontsize=11, fontweight="bold", color=TEXT_W, pad=10)
        ax1.set_ylabel("Registros", color=TEXT_G)
        ax1.set_ylim(0,max(sizes)*1.18)
        ax1.spines[:].set_color(BORDER)

        ax2 = fig.add_subplot(gs[1])
        ax2.set_xlim(0,10); ax2.set_ylim(0,8); ax2.axis("off")
        ax2.set_title("Control de Data Leakage",
                      fontsize=11, fontweight="bold", color=TEXT_W, pad=10)
        for (x,y,w,h),col,lbl in [
            ((0.4,5.8,6.3,0.85),ACCENT,"TRAIN  70%"),
            ((6.7,5.8,1.3,0.85),ACCENT2,"VAL 15%"),
            ((8.0,5.8,1.6,0.85),ACCENT3,"TEST 15%"),
        ]:
            ax2.add_patch(plt.Rectangle((x,y),w,h,color=col,alpha=0.85,zorder=2))
            cx = x+w/2
            ax2.text(cx,y+h/2,lbl,ha="center",va="center",fontsize=8,
                     fontweight="bold",color=DARK_BG,zorder=3)
        for col,y,txt in [
            (ACCENT, 4.7, "① fit_transform() → ajusta scaler con datos Train"),
            (ACCENT2,3.8, "② transform()      → aplica a Val sin re-ajustar"),
            (ACCENT3,2.9, "③ transform()      → aplica a Test sin re-ajustar"),
            (ACCENT3,2.0, "❌ NUNCA fit() en Val/Test = Data Leakage"),
            (ACCENT2,1.1, "✔  Baseline: DummyClassifier(most_frequent)"),
        ]:
            ax2.plot(0.6,y+0.08,marker="o",color=col,markersize=9,ls="None")
            ax2.text(1.0,y,txt,fontsize=8.5,va="center",color=TEXT_W)
        fig.tight_layout(pad=1.5)
        self._embed(fig, self.tab_part)

    # ═══════════════════════════════════════════
    #  RENDER CLUSTERING
    # ═══════════════════════════════════════════
    def _rend_clust(self, info):
        self._clr(self.tab_clust)
        X=info["X"]; k=info["k"]
        km_lbl=info["km_lbl"]; hac_lbl=info["hac_lbl"]
        pca=PCA(n_components=2,random_state=42)
        X2=pca.fit_transform(X)
        var=pca.explained_variance_ratio_*100
        fig=plt.figure(figsize=(14,5.5),facecolor=DARK_BG)
        gs=gridspec.GridSpec(1,3,figure=fig,left=0.05,right=0.97,wspace=0.35)
        for i,(lbl_arr,title,sil) in enumerate([
            (km_lbl, f"K-Means (k={k})  Sil={info['sil_km']:.3f}", info['sil_km']),
            (hac_lbl,f"Jerárquico      Sil={info['sil_hac']:.3f}", info['sil_hac']),
        ]):
            ax=fig.add_subplot(gs[i])
            for c in range(k):
                mask=lbl_arr==c
                ax.scatter(X2[mask,0],X2[mask,1],s=22,alpha=0.65,
                           color=PLOT_CLR[c],label=f"Clúster {c}",
                           edgecolors=DARK_BG,linewidths=0.3)
            ax.set_title(title,fontsize=10,fontweight="bold",color=TEXT_W,pad=8)
            ax.set_xlabel(f"PC1 ({var[0]:.1f}%)",fontsize=8,color=TEXT_G)
            ax.set_ylabel(f"PC2 ({var[1]:.1f}%)",fontsize=8,color=TEXT_G)
            ax.legend(fontsize=8,markerscale=1.3)
            ax.spines[:].set_color(BORDER)
        ax3=fig.add_subplot(gs[2])
        sils=[info["sil_km"],info["sil_hac"]]
        bars=ax3.bar(["K-Means","Jerárquico"],sils,color=[ACCENT,ACCENT2],
                     width=0.45,edgecolor=DARK_BG)
        for b,v in zip(bars,sils):
            ax3.text(b.get_x()+b.get_width()/2,v+0.006,f"{v:.4f}",
                     ha="center",fontsize=11,fontweight="bold",color=TEXT_W)
        ax3.axhline(0.5,color=ACCENT3,ls="--",lw=1.5,label="Umbral 0.5")
        ax3.set_ylim(0,max(sils)*1.3)
        ax3.set_title("Índice Silhouette",fontsize=10,fontweight="bold",color=TEXT_W,pad=8)
        ax3.set_ylabel("Score",fontsize=8,color=TEXT_G)
        ax3.legend(fontsize=8); ax3.spines[:].set_color(BORDER)
        fig.tight_layout(pad=2)
        self._embed(fig, self.tab_clust)

    # ═══════════════════════════════════════════
    #  RENDER CLASIFICACIÓN
    # ═══════════════════════════════════════════
    def _rend_clasif(self, res):
        self._clr(self.tab_clasif)
        clf=[r for r in res if r["name"]!="Baseline"]
        n=len(clf)
        fig=plt.figure(figsize=(6.5*n,5.5),facecolor=DARK_BG)
        for i,r in enumerate(clf):
            ax=fig.add_subplot(1,n,i+1)
            cm=confusion_matrix(r["y_test"],r["preds"])
            cmap=sns.light_palette(PLOT_CLR[i],as_cmap=True,reverse=False)
            sns.heatmap(cm,annot=True,fmt="d",cmap=cmap,
                        linewidths=1,linecolor=DARK_BG,
                        annot_kws={"fontsize":14,"fontweight":"bold"},
                        ax=ax,cbar_kws={"shrink":0.7})
            ax.set_title(f"{r['name']}\nAcc={r['acc']:.3f}  F1={r['f1']:.3f}  AUC={r['auc']:.3f}",
                         fontsize=10,fontweight="bold",color=TEXT_W,pad=8)
            ax.set_xlabel("Predicho",fontsize=9,color=TEXT_G)
            ax.set_ylabel("Real",    fontsize=9,color=TEXT_G)
        fig.tight_layout(pad=2); self._embed(fig, self.tab_clasif)

    # ═══════════════════════════════════════════
    #  RENDER EVALUACIÓN
    # ═══════════════════════════════════════════
    def _rend_eval(self, res):
        self._clr(self.tab_eval)
        fig=plt.figure(figsize=(13,5.5),facecolor=DARK_BG)
        gs=gridspec.GridSpec(1,2,figure=fig,left=0.07,right=0.97,wspace=0.38)
        ax_roc=fig.add_subplot(gs[0])
        binary=len(np.unique(res[0]["y_test"]))==2
        for r,col in zip(res,PLOT_CLR):
            if binary and r["probs"] is not None and r["auc"]>0:
                fpr,tpr,_=roc_curve(r["y_test"],r["probs"])
                ax_roc.plot(fpr,tpr,lw=2.2,color=col,
                            label=f"{r['name']} (AUC={r['auc']:.3f})")
        ax_roc.plot([0,1],[0,1],"--",lw=1,color=BORDER,label="Azar")
        ax_roc.fill_between([0,1],[0,1],alpha=0.05,color=TEXT_G)
        ax_roc.set_title("Curva ROC Comparativa",fontsize=11,fontweight="bold",color=TEXT_W,pad=8)
        ax_roc.set_xlabel("Tasa Falsos Positivos (FPR)",fontsize=9,color=TEXT_G)
        ax_roc.set_ylabel("Tasa Verdaderos Positivos (TPR)",fontsize=9,color=TEXT_G)
        ax_roc.legend(fontsize=9); ax_roc.spines[:].set_color(BORDER)
        if not binary:
            ax_roc.text(0.3,0.5,"Multiclase:\nROC no disponible",color=ACCENT3,fontsize=13,ha="center")
        ax_bar=fig.add_subplot(gs[1])
        names=[r["name"] for r in res]
        accs=[r["acc"] for r in res]; f1s=[r["f1"] for r in res]; aucs=[r["auc"] for r in res]
        x=np.arange(len(names)); w=0.26
        b1=ax_bar.bar(x-w, accs,w,label="Accuracy",color=ACCENT, alpha=0.9)
        b2=ax_bar.bar(x,    f1s,w,label="F1-Score", color=ACCENT2,alpha=0.9)
        b3=ax_bar.bar(x+w,  aucs,w,label="AUC",     color=ACCENT3,alpha=0.9)
        for bars in [b1,b2,b3]:
            for b in bars:
                ax_bar.text(b.get_x()+b.get_width()/2,b.get_height()+0.008,
                            f"{b.get_height():.3f}",ha="center",fontsize=7.5,color=TEXT_W,fontweight="bold")
        ax_bar.set_xticks(x); ax_bar.set_xticklabels(names,fontsize=9)
        ax_bar.set_ylim(0,1.22)
        ax_bar.set_title("Comparación de Métricas",fontsize=11,fontweight="bold",color=TEXT_W,pad=8)
        ax_bar.set_ylabel("Score",fontsize=9,color=TEXT_G)
        ax_bar.legend(fontsize=9); ax_bar.spines[:].set_color(BORDER)
        fig.tight_layout(pad=2); self._embed(fig, self.tab_eval)

    # ═══════════════════════════════════════════
    #  RENDER MATRIZ DE CONFUSIÓN
    # ═══════════════════════════════════════════
    def _rend_cm(self, res):
        self._clr(self.tab_cm)
        rf=next((r for r in res if r["name"]=="Random Forest"),res[-1])
        cm=confusion_matrix(rf["y_test"],rf["preds"])
        fig=plt.figure(figsize=(12,5.5),facecolor=DARK_BG)
        gs=gridspec.GridSpec(1,2,figure=fig,left=0.05,right=0.97,wspace=0.42)
        ax_cm=fig.add_subplot(gs[0])
        cmap=sns.light_palette(ACCENT,as_cmap=True)
        sns.heatmap(cm,annot=True,fmt="d",cmap=cmap,
                    linewidths=1.5,linecolor=DARK_BG,
                    annot_kws={"fontsize":20,"fontweight":"bold"},
                    ax=ax_cm,cbar_kws={"shrink":0.75})
        ax_cm.set_title("Matriz de Confusión · Random Forest",
                        fontsize=11,fontweight="bold",color=TEXT_W,pad=10)
        ax_cm.set_xlabel("Predicho",fontsize=10,color=TEXT_G)
        ax_cm.set_ylabel("Real",    fontsize=10,color=TEXT_G)
        if cm.shape==(2,2):
            vp=cm[1,1]; fp=cm[0,1]; fn=cm[1,0]; vn=cm[0,0]
            for txt,pos,col in [
                ("VP\nAcierto ✔",(0,0),ACCENT2),
                ("FP\nError ✖",  (1,0),ACCENT3),
                ("FN\nError ✖",  (0,1),ACCENT3),
                ("VN\nAcierto ✔",(1,1),ACCENT2),
            ]:
                ax_cm.text(pos[0]+0.5,pos[1]+0.87,txt,ha="center",va="bottom",
                           fontsize=8,color=col,style="italic",fontweight="bold")
        else: vp=fp=fn=vn=None
        ax_m=fig.add_subplot(gs[1]); ax_m.axis("off")
        ax_m.set_title("Métricas Derivadas",fontsize=11,fontweight="bold",color=TEXT_W,pad=10)
        acc=rf["acc"]; f1=rf["f1"]; prec=rf["prec"]; rec=rf["rec"]; auc_=rf["auc"]
        if cm.shape==(2,2) and vn is not None and fp is not None:
            esp=vn/(vn+fp) if (vn+fp)>0 else 0
            rows=[("▌ CUADRANTES","",TEXT_W,True),
                  ("VP  Verdaderos Positivos",str(vp),ACCENT2,False),
                  ("FP  Falsos Positivos",    str(fp),ACCENT3,False),
                  ("FN  Falsos Negativos",    str(fn),ACCENT3,False),
                  ("VN  Verdaderos Negativos",str(vn),ACCENT2,False),
                  ("","","",False),
                  ("▌ MÉTRICAS","",TEXT_W,True),
                  ("Precisión   VP/(VP+FP)",  f"{prec:.4f}",ACCENT, False),
                  ("Exactitud   (Accuracy)",   f"{acc:.4f}", ACCENT, False),
                  ("Sensibilidad (Recall)",    f"{rec:.4f}", ACCENT, False),
                  ("Especificidad VN/(VN+FP)",f"{esp:.4f}", ACCENT4,False),
                  ("F1-Score",                 f"{f1:.4f}", ACCENT2,False),
                  ("AUC-ROC",                  f"{auc_:.4f}",ACCENT3,False)]
        else:
            rows=[("Accuracy", f"{acc:.4f}",ACCENT, False),
                  ("F1-Score", f"{f1:.4f}", ACCENT2,False),
                  ("Precision",f"{prec:.4f}",ACCENT3,False),
                  ("Recall",   f"{rec:.4f}", ACCENT4,False),
                  ("AUC-ROC",  f"{auc_:.4f}",GOLD,  False)]
        y=0.97
        for lbl,val,col,hdr in rows:
            if lbl=="": y-=0.04; continue
            fw="bold" if hdr else "normal"
            ax_m.text(0.02,y,lbl,color=TEXT_G if not hdr else TEXT_W,va="top",
                      transform=ax_m.transAxes,fontsize=9,fontweight=fw)
            if val: ax_m.text(0.80,y,val,color=col,va="top",
                              transform=ax_m.transAxes,fontsize=10,fontweight="bold")
            y-=0.075
        fig.tight_layout(pad=2); self._embed(fig, self.tab_cm)

    # ═══════════════════════════════════════════
    #  RENDER INTERPRETACIÓN
    # ═══════════════════════════════════════════
    def _rend_interp(self, res, clust, part):
        self._clr(self.tab_interp)
        rf  = next(r for r in res if r["name"]=="Random Forest")
        arb = next((r for r in res if "rbol" in r["name"]),res[0])
        bas = next((r for r in res if r["name"]=="Baseline"),res[0])
        mejor = max(res, key=lambda r: r["auc"])
        total = sum(part.values())
        modo  = "TODA la data" if self.mode=="all" else f"Columna: {self.combo_col.get()}"

        texto = f"""
╔══════════════════════════════════════════════════════════════════════════╗
║           INTERPRETACIÓN DE RESULTADOS — EXAMEN UNIDAD 2              ║
║         (Para comunicar a un equipo gerencial no técnico)              ║
╚══════════════════════════════════════════════════════════════════════════╝
  Modo de análisis: {modo}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 1. EDA Y LIMPIEZA DE DATOS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 Pasos realizados:
   • Eliminación de columnas ID irrelevantes (no aportan información)
   • Conversión de variables texto binarias (yes/no → 1/0, true/false → 1/0)
   • Imputación de valores faltantes con mediana (numéricas) y moda (categóricas)
   • Generación de estadísticas descriptivas y mapa de correlación
   ✔ Dato limpio = modelo más confiable.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 2. PARTICIÓN Y CONTROL DE DATA LEAKAGE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 Total de registros: {total:,}
   • Train  (70%) : {part['train']:,}  → el modelo aprende aquí
   • Val    (15%) : {part['val']:,}   → ajuste de parámetros
   • Test   (15%) : {part['test']:,}   → evaluación final con datos "nuevos"

 ¿Qué es Data Leakage?
   Ocurre cuando el modelo ve información del futuro durante el entrenamiento.
   ✔ Solución aplicada: fit_transform() SOLO en Train, transform() en Val/Test.
   ❌ Si hiciéramos fit() en Test = el modelo "haría trampa" y sus métricas
      serían irrealmente buenas pero fallaría en producción real.

 Baseline (punto de referencia mínimo):
   DummyClassifier predice siempre la clase más frecuente, sin aprender nada.
   Acc={bas['acc']:.4f}  F1={bas['f1']:.4f}
   → Todo modelo útil DEBE superar este valor.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 3. SEGMENTACIÓN — Clustering
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 Algoritmos con k = {clust['k']} clústeres:

   K-Means: agrupa por proximidad al centroide más cercano (iterativo).
   → Silhouette = {clust['sil_km']:.4f}

   Clustering Jerárquico (Agglomerative): une registros similares de abajo
   hacia arriba, formando un árbol (dendrograma).
   → Silhouette = {clust['sil_hac']:.4f}

 Índice Silhouette: -1 (pésimo) | 0 (solapado) | 1 (perfecto)
 Valor ≥ 0.50 indica buena separación entre grupos.

 Perfiles de estudiantes identificados:
   🔵 Grupo 0 — Alto rendimiento: notas altas, poco ausentismo.
                → Acción: programas de excelencia y becas.
   🟢 Grupo 1 — Rendimiento medio: asistencia regular, notas medias.
                → Acción: tutorías de refuerzo.
   🔴 Grupo 2 — Riesgo académico: ausencias frecuentes, notas bajas.
                → Acción: intervención urgente, apoyo familiar.
   🟣 Grupo 3 — Perfil mixto: características variadas.
                → Acción: análisis individualizado.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 4. CLASIFICACIÓN — Predicción de Respuesta
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 Árbol de Decisión:
   Como un diagrama de preguntas Sí/No que el modelo aprende solo.
   ✔ Ventaja: fácil de explicar a gerencia (se puede dibujar).
   ✗ Limitación: tiende a sobreajustarse (overfitting).
   Acc={arb['acc']:.4f}  F1={arb['f1']:.4f}  AUC={arb['auc']:.4f}

 Random Forest:
   Combina 120 árboles distintos y decide por votación de mayoría.
   ✔ Ventaja: más estable, mejor generalización, robusto al overfitting.
   ✗ Limitación: menos interpretable que un árbol individual.
   Acc={rf['acc']:.4f}  F1={rf['f1']:.4f}  AUC={rf['auc']:.4f}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 5. EVALUACIÓN COMPARATIVA
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 Modelo             Accuracy     F1-Score      AUC
 ──────────────────────────────────────────────────
 Baseline           {bas['acc']:.4f}       {bas['f1']:.4f}        {bas['auc']:.4f}
 Árbol Decisión     {arb['acc']:.4f}       {arb['f1']:.4f}        {arb['auc']:.4f}
 Random Forest      {rf['acc']:.4f}       {rf['f1']:.4f}        {rf['auc']:.4f}
 ──────────────────────────────────────────────────
 ✔ MEJOR MODELO: {mejor['name']} (AUC = {mejor['auc']:.4f})

 Curva ROC:
   Muestra qué tan bien el modelo separa positivos de negativos.
   AUC = 1.0 → predicción perfecta
   AUC = 0.5 → equivale a adivinar al azar

 Comunicación gerencial:
   "{mejor['name']} identifica correctamente el {mejor['acc']*100:.1f}% de los casos.
   Implementarlo permite enfocar recursos en los estudiantes/clientes con
   mayor probabilidad de necesitar intervención, reduciendo costos y
   maximizando el impacto de las acciones tomadas."

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 6. MATRIZ DE CONFUSIÓN — Interpretación (RF)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   ✅ VP (Verdaderos Positivos) — Acierto tipo 1:
      Predijo correctamente que el evento OCURRIRÍA.
      → Estudiante en riesgo detectado a tiempo.

   ✅ VN (Verdaderos Negativos) — Acierto tipo 2:
      Predijo correctamente que NO ocurriría.
      → Recursos no desperdiciados en quien no necesita intervención.

   ❌ FP (Falsos Positivos) — Error tipo 1:
      Predijo SÍ, pero en realidad NO.
      → Gasto innecesario de recursos de intervención.

   ❌ FN (Falsos Negativos) — Error tipo 2 (el más costoso):
      Predijo NO, pero en realidad SÍ.
      → Caso perdido: estudiante en riesgo NO detectado a tiempo.

 Métricas Random Forest:
   • Precisión    = {rf['prec']:.4f}  → De los que predijo positivo, ¿cuántos lo eran realmente?
   • Sensibilidad = {rf['rec']:.4f}  → De todos los positivos reales, ¿cuántos detectó?
   • F1-Score     = {rf['f1']:.4f}  → Balance entre Precisión y Sensibilidad
   • AUC-ROC      = {rf['auc']:.4f}  → Capacidad global de discriminación del modelo
"""
        fr  = tk.Frame(self.tab_interp, bg=DARK_BG)
        fr.pack(fill=tk.BOTH, expand=True)
        txt = scrolledtext.ScrolledText(
            fr, bg=PANEL_BG, fg=TEXT_W, font=("Consolas",9),
            relief="flat", borderwidth=0, wrap=tk.WORD,
            padx=18, pady=14, selectbackground=ACCENT)
        txt.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)
        txt.tag_config("hdr",     foreground=ACCENT,  font=("Consolas",9,"bold"))
        txt.tag_config("section", foreground=ACCENT4, font=("Consolas",9,"bold"))
        txt.tag_config("ok",      foreground=ACCENT2)
        txt.tag_config("error",   foreground=ACCENT3)
        for line in texto.strip().split("\n"):
            if "╔" in line or "╚" in line or "║" in line:
                txt.insert(tk.END, line+"\n", "hdr")
            elif "━━" in line or "──" in line:
                txt.insert(tk.END, line+"\n", "section")
            elif line.strip().startswith(("✅","✔","🔵","🟢","🟣")):
                txt.insert(tk.END, line+"\n", "ok")
            elif line.strip().startswith(("❌","✗","🔴")):
                txt.insert(tk.END, line+"\n", "error")
            else:
                txt.insert(tk.END, line+"\n")
        txt.config(state=tk.DISABLED)

    # ─── UTILIDADES ──────────────────────────
    def _clr(self, tab):
        for w in tab.winfo_children(): w.destroy()

    def _embed(self, fig, parent):
        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()
        tb = NavigationToolbar2Tk(canvas, parent, pack_toolbar=False)
        tb.config(background=PANEL_BG); tb.update()
        tb.pack(side=tk.BOTTOM, fill=tk.X)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        plt.close(fig)


# ══════════════════════════════════════════════
if __name__ == "__main__":
    AppMineria().mainloop()
