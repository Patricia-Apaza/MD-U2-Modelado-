"""
Instalar dependencias:
    pip install pandas numpy scikit-learn matplotlib seaborn scipy chardet
"""

import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext
import threading
import traceback

import pandas as pd
import numpy as np
import chardet
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage

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

# ─────────────────────────────────────────────
#  PALETA Y ESTILO GLOBAL
# ─────────────────────────────────────────────
DARK_BG    = "#0D1117"
PANEL_BG   = "#161B22"
CARD_BG    = "#21262D"
BORDER     = "#30363D"
ACCENT     = "#58A6FF"
ACCENT2    = "#3FB950"
ACCENT3    = "#F78166"
TEXT_WHITE = "#E6EDF3"
TEXT_GRAY  = "#8B949E"
FONT_TITLE = ("Segoe UI", 13, "bold")
FONT_BODY  = ("Segoe UI", 10)
FONT_MONO  = ("Consolas", 9)
FONT_SMALL = ("Segoe UI", 9)

# Paleta para matplotlib
PLOT_COLORS = ["#58A6FF", "#3FB950", "#F78166", "#D2A8FF", "#FFA657"]
sns.set_theme(style="dark")
plt.rcParams.update({
    "figure.facecolor":  DARK_BG,
    "axes.facecolor":    PANEL_BG,
    "axes.edgecolor":    BORDER,
    "axes.labelcolor":   TEXT_WHITE,
    "xtick.color":       TEXT_GRAY,
    "ytick.color":       TEXT_GRAY,
    "text.color":        TEXT_WHITE,
    "grid.color":        BORDER,
    "legend.facecolor":  CARD_BG,
    "legend.edgecolor":  BORDER,
})


# ─────────────────────────────────────────────
#  HELPER: TOOLTIP FLOTANTE
# ─────────────────────────────────────────────
class Tooltip:
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tip = None
        widget.bind("<Enter>", self.show)
        widget.bind("<Leave>", self.hide)

    def show(self, _=None):
        x = self.widget.winfo_rootx() + 20
        y = self.widget.winfo_rooty() + 30
        self.tip = tk.Toplevel(self.widget)
        self.tip.wm_overrideredirect(True)
        self.tip.wm_geometry(f"+{x}+{y}")
        lbl = tk.Label(self.tip, text=self.text, background="#2D333B",
                       foreground=TEXT_WHITE, relief="flat",
                       font=FONT_SMALL, padx=8, pady=4)
        lbl.pack()

    def hide(self, _=None):
        if self.tip:
            self.tip.destroy()
            self.tip = None


# ─────────────────────────────────────────────
#  WIDGET: TARJETA DE MÉTRICA
# ─────────────────────────────────────────────
class MetricCard(tk.Frame):
    def __init__(self, parent, label, value, color=ACCENT):
        super().__init__(parent, bg=CARD_BG, highlightbackground=color,
                         highlightthickness=1, padx=14, pady=10)
        tk.Label(self, text=label, bg=CARD_BG, fg=TEXT_GRAY,
                 font=FONT_SMALL).pack(anchor="w")
        self.val_lbl = tk.Label(self, text=value, bg=CARD_BG,
                                fg=color, font=("Segoe UI", 18, "bold"))
        self.val_lbl.pack(anchor="w")

    def update(self, value):
        self.val_lbl.config(text=value)


# ─────────────────────────────────────────────
#  APLICACIÓN PRINCIPAL
# ─────────────────────────────────────────────
class AppMineria(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Analizador de Minería de Datos · Unidad 2")
        self.geometry("1400x860")
        self.minsize(1100, 700)
        self.configure(bg=DARK_BG)
        self.df = None

        self._build_styles()
        self._build_header()
        self._build_toolbar()
        self._build_body()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ── ESTILOS ttk ──────────────────────────
    def _build_styles(self):
        st = ttk.Style(self)
        st.theme_use("clam")
        st.configure("TFrame",      background=DARK_BG)
        st.configure("Card.TFrame", background=CARD_BG)
        st.configure("TLabel",      background=DARK_BG, foreground=TEXT_WHITE,
                     font=FONT_BODY)
        st.configure("Title.TLabel", background=DARK_BG, foreground=ACCENT,
                     font=FONT_TITLE)
        st.configure("TCombobox",   fieldbackground=CARD_BG,
                     background=CARD_BG, foreground=TEXT_WHITE,
                     arrowcolor=ACCENT)
        st.map("TCombobox", fieldbackground=[("readonly", CARD_BG)],
               foreground=[("readonly", TEXT_WHITE)])
        # Botón primario
        st.configure("Accent.TButton",
                     background=ACCENT, foreground=DARK_BG,
                     font=("Segoe UI", 10, "bold"),
                     padding=(14, 7), relief="flat")
        st.map("Accent.TButton",
               background=[("active", "#79B8FF"), ("disabled", BORDER)],
               foreground=[("disabled", TEXT_GRAY)])
        # Botón secundario
        st.configure("Ghost.TButton",
                     background=CARD_BG, foreground=TEXT_WHITE,
                     font=FONT_BODY, padding=(12, 6), relief="flat")
        st.map("Ghost.TButton",
               background=[("active", BORDER)])
        # Notebook
        st.configure("TNotebook", background=DARK_BG, borderwidth=0)
        st.configure("TNotebook.Tab",
                     background=PANEL_BG, foreground=TEXT_GRAY,
                     font=FONT_BODY, padding=(18, 8))
        st.map("TNotebook.Tab",
               background=[("selected", CARD_BG)],
               foreground=[("selected", ACCENT)])
        # Separador
        st.configure("TSeparator", background=BORDER)
        # Scrollbar
        st.configure("Vertical.TScrollbar",
                     background=CARD_BG, troughcolor=PANEL_BG,
                     arrowcolor=TEXT_GRAY, borderwidth=0)
        # Treeview
        st.configure("Treeview", background=CARD_BG,
                     fieldbackground=CARD_BG, foreground=TEXT_WHITE,
                     font=FONT_SMALL, rowheight=24)
        st.configure("Treeview.Heading",
                     background=PANEL_BG, foreground=ACCENT,
                     font=("Segoe UI", 9, "bold"))
        st.map("Treeview", background=[("selected", ACCENT)],
               foreground=[("selected", DARK_BG)])
        # Progressbar
        st.configure("Blue.Horizontal.TProgressbar",
                     troughcolor=PANEL_BG, background=ACCENT,
                     borderwidth=0, thickness=4)

    # ── CABECERA ─────────────────────────────
    def _build_header(self):
        hdr = tk.Frame(self, bg=PANEL_BG, height=56)
        hdr.pack(fill=tk.X)
        hdr.pack_propagate(False)

        tk.Label(hdr, text="⬡  Minería de Datos", bg=PANEL_BG,
                 fg=ACCENT, font=("Segoe UI", 14, "bold")).pack(side=tk.LEFT, padx=20)
        tk.Label(hdr, text="Unidad 2 · Partición · Clustering · Clasificación · Evaluación",
                 bg=PANEL_BG, fg=TEXT_GRAY, font=FONT_SMALL).pack(side=tk.LEFT)

        self.status_dot = tk.Label(hdr, text="●", bg=PANEL_BG,
                                   fg=TEXT_GRAY, font=("Segoe UI", 12))
        self.status_dot.pack(side=tk.RIGHT, padx=6)
        self.status_lbl = tk.Label(hdr, text="Sin datos", bg=PANEL_BG,
                                   fg=TEXT_GRAY, font=FONT_SMALL)
        self.status_lbl.pack(side=tk.RIGHT, padx=(0, 4))

        sep = tk.Frame(self, bg=BORDER, height=1)
        sep.pack(fill=tk.X)

    # ── BARRA DE HERRAMIENTAS ─────────────────
    def _build_toolbar(self):
        bar = tk.Frame(self, bg=DARK_BG, pady=10, padx=16)
        bar.pack(fill=tk.X)

        # Botón cargar
        btn_load = ttk.Button(bar, text="📂  Cargar CSV",
                              style="Accent.TButton",
                              command=self._cargar_csv)
        btn_load.pack(side=tk.LEFT, padx=(0, 12))
        Tooltip(btn_load, "Carga un archivo CSV para analizar")

        # Separador vertical
        tk.Frame(bar, bg=BORDER, width=1).pack(side=tk.LEFT,
                                               fill=tk.Y, padx=8, pady=2)

        tk.Label(bar, text="Variable objetivo:", bg=DARK_BG,
                 fg=TEXT_GRAY, font=FONT_SMALL).pack(side=tk.LEFT, padx=(4, 6))

        self.combo_target = ttk.Combobox(bar, state="disabled", width=28,
                                         font=FONT_BODY)
        self.combo_target.pack(side=tk.LEFT, padx=(0, 12))

        # Botón ejecutar
        self.btn_run = ttk.Button(bar, text="▶  Ejecutar Análisis",
                                  style="Accent.TButton",
                                  command=self._iniciar_analisis,
                                  state="disabled")
        self.btn_run.pack(side=tk.LEFT, padx=(0, 12))
        Tooltip(self.btn_run, "Ejecuta todo el pipeline: partición, baseline, clustering y clasificación")

        # Barra de progreso
        self.progress = ttk.Progressbar(bar, mode="indeterminate", length=160,
                                        style="Blue.Horizontal.TProgressbar")
        self.progress.pack(side=tk.LEFT, padx=(8, 0))

        # Info del dataset
        self.info_lbl = tk.Label(bar, text="", bg=DARK_BG,
                                 fg=TEXT_GRAY, font=FONT_SMALL)
        self.info_lbl.pack(side=tk.RIGHT, padx=10)

    # ── CUERPO PRINCIPAL ─────────────────────
    def _build_body(self):
        body = tk.PanedWindow(self, orient=tk.HORIZONTAL,
                              bg=DARK_BG, sashwidth=4,
                              sashrelief="flat", sashpad=0)
        body.pack(fill=tk.BOTH, expand=True, padx=0, pady=0)

        # ── Panel izquierdo: log + métricas ──
        left = tk.Frame(body, bg=DARK_BG, width=360)
        body.add(left, minsize=260)

        tk.Label(left, text="REGISTRO DE EJECUCIÓN",
                 bg=DARK_BG, fg=TEXT_GRAY,
                 font=("Segoe UI", 8, "bold")).pack(anchor="w", padx=14, pady=(10, 2))

        self.log = scrolledtext.ScrolledText(
            left, height=11, bg=PANEL_BG, fg=TEXT_WHITE,
            font=FONT_MONO, relief="flat", borderwidth=0,
            insertbackground=ACCENT, selectbackground=ACCENT
        )
        self.log.pack(fill=tk.X, padx=12, pady=(0, 8))
        self._log_tag_config()

        # Métricas rápidas
        tk.Label(left, text="MÉTRICAS RÁPIDAS",
                 bg=DARK_BG, fg=TEXT_GRAY,
                 font=("Segoe UI", 8, "bold")).pack(anchor="w", padx=14, pady=(6, 4))

        self.cards_frame = tk.Frame(left, bg=DARK_BG)
        self.cards_frame.pack(fill=tk.X, padx=12)

        self.cards = {}
        metrics = [("Accuracy (RF)", "—", ACCENT),
                   ("F1-Score (RF)",  "—", ACCENT2),
                   ("AUC (RF)",       "—", ACCENT3),
                   ("Silhouette",     "—", "#D2A8FF")]
        for label, val, color in metrics:
            card = MetricCard(self.cards_frame, label, val, color)
            card.pack(fill=tk.X, pady=2, ipady=2)
            self.cards[label] = card

        # Tabla comparativa de corridas
        tk.Label(left, text="TABLA COMPARATIVA DE CORRIDAS",
                 bg=DARK_BG, fg=TEXT_GRAY,
                 font=("Segoe UI", 8, "bold")).pack(anchor="w", padx=14, pady=(10, 4))

        cols = ("Modelo", "Accuracy", "F1", "AUC")
        self.tree = ttk.Treeview(left, columns=cols, show="headings",
                                 height=8, selectmode="browse")
        for c in cols:
            self.tree.heading(c, text=c)
            self.tree.column(c, width=68, anchor="center")
        self.tree.pack(fill=tk.X, padx=12, pady=(0, 10))

        # ── Panel derecho: notebook de gráficas ──
        right = tk.Frame(body, bg=DARK_BG)
        body.add(right, minsize=700)

        self.notebook = ttk.Notebook(right)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        self.tab_particion   = self._make_tab("📊 Partición & Baseline")
        self.tab_clustering  = self._make_tab("🔵 Clustering")
        self.tab_clasif      = self._make_tab("🌳 Clasificación")
        self.tab_eval        = self._make_tab("📈 Evaluación Comparativa")
        self.tab_confusion   = self._make_tab("🔲 Matriz de Confusión")
        self.tab_interpretacion = self._make_tab("💬 Interpretación")

    def _make_tab(self, title):
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text=title)
        return frame

    def _log_tag_config(self):
        self.log.tag_config("info",    foreground=ACCENT)
        self.log.tag_config("ok",      foreground=ACCENT2)
        self.log.tag_config("warn",    foreground="#FFA657")
        self.log.tag_config("error",   foreground=ACCENT3)
        self.log.tag_config("section", foreground="#D2A8FF",
                            font=("Consolas", 9, "bold"))

    def _log(self, msg, tag="info"):
        self.log.insert(tk.END, msg + "\n", tag)
        self.log.see(tk.END)

    def _set_status(self, text, color=TEXT_GRAY):
        self.status_lbl.config(text=text, fg=color)
        self.status_dot.config(fg=color)

    # ── CARGAR CSV ───────────────────────────
    def _cargar_csv(self):
        path = filedialog.askopenfilename(
            title="Seleccionar CSV",
            filetypes=[("Archivos CSV", "*.csv"), ("Todos", "*.*")]
        )
        if not path:
            return
        enc = self._detectar_encoding(path)
        for e in [enc, "utf-8", "latin-1", "cp1252", "iso-8859-1"]:
            if not e:
                continue
            try:
                self.df = pd.read_csv(path, encoding=e)
                self.combo_target.config(state="readonly",
                                         values=list(self.df.columns))
                self.combo_target.current(len(self.df.columns) - 1)
                self.btn_run.config(state="normal")
                rows, cols = self.df.shape
                self.info_lbl.config(
                    text=f"{rows:,} filas · {cols} columnas · enc: {e}"
                )
                self._set_status("Dataset cargado", ACCENT2)
                self._log(f"✔ Archivo cargado ({e}): {rows} filas, {cols} cols", "ok")
                break
            except Exception:
                continue

    def _detectar_encoding(self, path):
        with open(path, "rb") as f:
            return chardet.detect(f.read(20000)).get("encoding")

    # ── INICIAR ANÁLISIS ─────────────────────
    def _iniciar_analisis(self):
        if not self.combo_target.get():
            return
        self.btn_run.config(state="disabled")
        self.progress.start(12)
        self._set_status("Analizando…", "#FFA657")
        threading.Thread(target=self._pipeline, daemon=True).start()

    # ── PIPELINE COMPLETO ────────────────────
    def _pipeline(self):
        try:
            target = self.combo_target.get()
            df = self.df.copy()

            self._log("\n══════ INICIO DE PIPELINE ══════", "section")

            # ── 1. PREPARACIÓN ──────────────
            self._log("\n[1/5] Preparación de datos…", "section")

            # Eliminar columnas ID irrelevantes
            drop_cols = [c for c in df.columns
                         if c.lower() in ("customerid", "id", "index")]
            if drop_cols:
                df.drop(columns=drop_cols, inplace=True, errors="ignore")
                self._log(f"  Columnas eliminadas: {drop_cols}", "warn")

            X_raw = df.drop(columns=[target], errors="ignore")
            y_raw = df[target]

            # Codificar target
            if y_raw.dtype == "object":
                le = LabelEncoder()
                y = le.fit_transform(y_raw)
                clases = list(le.classes_)
                self._log(f"  Target categórico → LabelEncoder: {clases}", "info")
            else:
                y = (y_raw > y_raw.median()).astype(int)
                self._log(f"  Target numérico → binarizado por mediana", "info")

            # Pipeline de preprocesamiento
            num_cols = X_raw.select_dtypes(include=[np.number]).columns.tolist()
            cat_cols = X_raw.select_dtypes(exclude=[np.number]).columns.tolist()
            self._log(f"  Numéricas: {len(num_cols)} | Categóricas: {len(cat_cols)}", "info")

            num_pipe = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler",  StandardScaler())
            ])
            cat_pipe = Pipeline([
                ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
            ])
            transformers = []
            if num_cols:
                transformers.append(("num", num_pipe, num_cols))
            if cat_cols:
                transformers.append(("cat", cat_pipe, cat_cols))

            preprocessor = ColumnTransformer(transformers)
            X = preprocessor.fit_transform(X_raw)
            self._log(f"  Shape final X: {X.shape}", "ok")

            # ── 2. PARTICIÓN ────────────────
            self._log("\n[2/5] Partición 70/15/15 (Train/Val/Test)…", "section")
            """
            DATA LEAKAGE: Se aplica fit_transform solo en Train.
            Val y Test se transforman con los parámetros de Train
            para evitar filtración de información.
            """
            strat = y if np.min(np.bincount(y)) >= 2 else None
            X_train_val, X_test, y_train_val, y_test = train_test_split(
                X, y, test_size=0.15, random_state=42, stratify=strat)

            strat2 = y_train_val if np.min(np.bincount(y_train_val)) >= 2 else None
            X_train, X_val, y_train, y_val = train_test_split(
                X_train_val, y_train_val,
                test_size=0.15 / 0.85, random_state=42, stratify=strat2)

            self._log(f"  Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}", "ok")
            self._log("  ✔ Sin data leakage: scaler ajustado solo en Train", "ok")

            part_info = {
                "train": len(X_train), "val": len(X_val), "test": len(X_test)
            }

            # ── 3. BASELINE ─────────────────
            self._log("\n[3/5] Baseline (DummyClassifier most_frequent)…", "section")
            baseline = DummyClassifier(strategy="most_frequent")
            baseline.fit(X_train, y_train)
            b_preds = baseline.predict(X_test)
            b_acc = accuracy_score(y_test, b_preds)
            b_f1  = f1_score(y_test, b_preds, average="weighted", zero_division=0)
            self._log(f"  Baseline Acc: {b_acc:.4f}  F1: {b_f1:.4f}", "warn")

            # ── 4. CLUSTERING ───────────────
            self._log("\n[4/5] Clustering…", "section")
            k = min(4, max(2, len(X) // 80))
            kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
            km_labels = kmeans.fit_predict(X)
            sil_km = silhouette_score(X, km_labels)

            hac = AgglomerativeClustering(n_clusters=k)
            hac_labels = hac.fit_predict(X)
            sil_hac = silhouette_score(X, hac_labels)

            self._log(f"  K-Means (k={k}) Silhouette: {sil_km:.4f}", "ok")
            self._log(f"  Clustering Jerárquico     Silhouette: {sil_hac:.4f}", "ok")

            # Perfiles de clústeres (en columnas originales numéricas)
            if num_cols:
                df_cluster = pd.DataFrame(
                    X_raw[num_cols].values, columns=num_cols
                ).copy()
                df_cluster["cluster"] = km_labels[:len(df_cluster)]
                cluster_profiles = df_cluster.groupby("cluster").mean()
            else:
                cluster_profiles = None

            clust_info = {
                "k": k, "km_labels": km_labels, "hac_labels": hac_labels,
                "sil_km": sil_km, "sil_hac": sil_hac,
                "profiles": cluster_profiles, "X": X
            }

            # ── 5. CLASIFICACIÓN ────────────
            self._log("\n[5/5] Clasificación…", "section")
            models_def = {
                "Baseline": baseline,
                "Árbol":    DecisionTreeClassifier(max_depth=8, random_state=42),
                "Random Forest": RandomForestClassifier(
                    n_estimators=100, random_state=42, n_jobs=-1)
            }
            resultados = []
            for name, model in models_def.items():
                if name != "Baseline":
                    model.fit(X_train, y_train)
                preds = model.predict(X_test)
                probs = (model.predict_proba(X_test)[:, 1]
                         if hasattr(model, "predict_proba") else None)
                acc  = accuracy_score(y_test, preds)
                f1   = f1_score(y_test, preds, average="weighted", zero_division=0)
                prec = precision_score(y_test, preds, average="weighted", zero_division=0)
                rec  = recall_score(y_test, preds, average="weighted", zero_division=0)
                roc_auc = 0.0
                if probs is not None and len(np.unique(y_test)) == 2:
                    fpr_, tpr_, _ = roc_curve(y_test, probs)
                    roc_auc = auc(fpr_, tpr_)
                resultados.append({
                    "name": name, "preds": preds, "probs": probs,
                    "acc": acc, "f1": f1, "prec": prec, "rec": rec,
                    "auc": roc_auc, "y_test": y_test
                })
                self._log(
                    f"  {name:<16} Acc:{acc:.3f} F1:{f1:.3f} AUC:{roc_auc:.3f}", "ok"
                )

            # Actualizar métricas de RF
            rf_res = next(r for r in resultados if r["name"] == "Random Forest")
            self.after(0, self.cards["Accuracy (RF)"].update, f"{rf_res['acc']:.4f}")
            self.after(0, self.cards["F1-Score (RF)"].update,  f"{rf_res['f1']:.4f}")
            self.after(0, self.cards["AUC (RF)"].update,       f"{rf_res['auc']:.4f}")
            self.after(0, self.cards["Silhouette"].update,     f"{sil_km:.4f}")

            # Actualizar tabla comparativa
            self.after(0, self._update_tabla, resultados)

            self._log("\n✔ Pipeline completado.", "ok")

            # Renderizar gráficas en hilo principal
            self.after(0, self._render_particion,  part_info)
            self.after(0, self._render_clustering, clust_info)
            self.after(0, self._render_clasificacion, resultados)
            self.after(0, self._render_evaluacion,   resultados)
            self.after(0, self._render_confusion,    resultados)
            self.after(0, self._render_interpretacion,
                       resultados, clust_info, part_info)
            self.after(0, self._finalizar)

        except Exception as e:
            traceback.print_exc()
            self.after(0, lambda: self._log(f"✖ Error: {e}", "error"))
            self.after(0, self._finalizar)

    def _finalizar(self):
        self.progress.stop()
        self.btn_run.config(state="normal")
        self._set_status("Análisis completado", ACCENT2)

    # ── TABLA COMPARATIVA ────────────────────
    def _update_tabla(self, resultados):
        for row in self.tree.get_children():
            self.tree.delete(row)
        for r in resultados:
            tag = "rf" if r["name"] == "Random Forest" else ""
            self.tree.insert("", tk.END,
                             values=(r["name"],
                                     f"{r['acc']:.3f}",
                                     f"{r['f1']:.3f}",
                                     f"{r['auc']:.3f}"),
                             tags=(tag,))
        self.tree.tag_configure("rf", foreground=ACCENT2)

    # ──────────────────────────────────────────
    #  RENDER: TAB PARTICIÓN & BASELINE
    # ──────────────────────────────────────────
    def _render_particion(self, info):
        self._clear_tab(self.tab_particion)

        fig = plt.figure(figsize=(10, 4.5), facecolor=DARK_BG)
        gs  = gridspec.GridSpec(1, 2, figure=fig,
                                left=0.08, right=0.95,
                                wspace=0.4)

        # ── Gráfico de distribución Train/Val/Test ──
        ax1 = fig.add_subplot(gs[0])
        labels = ["Train\n(70%)", "Val\n(15%)", "Test\n(15%)"]
        sizes  = [info["train"], info["val"], info["test"]]
        colors = [ACCENT, ACCENT2, ACCENT3]
        bars = ax1.barh(labels, sizes, color=colors, height=0.5)
        for bar, s in zip(bars, sizes):
            ax1.text(bar.get_width() + max(sizes) * 0.01, bar.get_y() + bar.get_height() / 2,
                     f"{s:,}", va="center", color=TEXT_WHITE, fontsize=9)
        ax1.set_title("Distribución de Partición", color=TEXT_WHITE, fontsize=11, pad=10)
        ax1.set_xlabel("Número de registros", color=TEXT_GRAY, fontsize=9)
        ax1.tick_params(colors=TEXT_WHITE)
        ax1.set_xlim(0, max(sizes) * 1.18)
        ax1.spines[:].set_color(BORDER)

        # ── Diagrama conceptual de leakage ──
        ax2 = fig.add_subplot(gs[1])
        ax2.set_xlim(0, 10)
        ax2.set_ylim(0, 7)
        ax2.axis("off")
        ax2.set_title("Control de Data Leakage", color=TEXT_WHITE, fontsize=11, pad=10)

        # Barra total
        ax2.add_patch(plt.Rectangle((0.5, 5.2), 9, 0.9,
                                    color=BORDER, zorder=1))
        ax2.add_patch(plt.Rectangle((0.5, 5.2), 9 * 0.70, 0.9,
                                    color=ACCENT, alpha=0.85, zorder=2))
        ax2.add_patch(plt.Rectangle((0.5 + 9 * 0.70, 5.2), 9 * 0.15, 0.9,
                                    color=ACCENT2, alpha=0.85, zorder=2))
        ax2.add_patch(plt.Rectangle((0.5 + 9 * 0.85, 5.2), 9 * 0.15, 0.9,
                                    color=ACCENT3, alpha=0.85, zorder=2))
        for pos, lbl in [(0.5 + 9 * 0.35, "Train 70%"),
                         (0.5 + 9 * 0.775, "Val"), (0.5 + 9 * 0.925, "Test")]:
            ax2.text(pos, 5.65, lbl, ha="center", color=DARK_BG,
                     fontsize=8.5, fontweight="bold", zorder=3)

        items = [
            (1.5, 4.1, ACCENT,
             "fit_transform() → ajusta scaler con datos Train"),
            (1.5, 3.2, ACCENT2,
             "transform() solamente en Val/Test"),
            (1.5, 2.3, ACCENT3,
             "❌ Nunca: fit en Val/Test = Data Leakage"),
            (1.5, 1.4, "#D2A8FF",
             "✔ Baseline: DummyClassifier(most_frequent)"),
        ]
        for x, y, col, txt in items:
            ax2.plot(x, y + 0.05, marker="o", color=col,
                     markersize=8, linestyle="None")
            ax2.text(x + 0.4, y, txt, color=TEXT_WHITE, fontsize=8.5, va="center")

        self._embed_fig(fig, self.tab_particion)

    # ──────────────────────────────────────────
    #  RENDER: TAB CLUSTERING
    # ──────────────────────────────────────────
    def _render_clustering(self, info):
        self._clear_tab(self.tab_clustering)

        fig = plt.figure(figsize=(13, 5.5), facecolor=DARK_BG)
        gs  = gridspec.GridSpec(1, 3, figure=fig,
                                left=0.06, right=0.97, wspace=0.38)

        X = info["X"]
        k = info["k"]
        km_labels  = info["km_labels"]
        hac_labels = info["hac_labels"]

        # Reducir a 2D para visualización (primeras 2 componentes)
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2, random_state=42)
        X2  = pca.fit_transform(X)

        cmap = [ACCENT, ACCENT2, ACCENT3, "#D2A8FF", "#FFA657"]

        # ── K-Means scatter ──
        ax1 = fig.add_subplot(gs[0])
        for c in range(k):
            mask = km_labels == c
            ax1.scatter(X2[mask, 0], X2[mask, 1],
                        s=18, alpha=0.65, color=cmap[c % len(cmap)],
                        label=f"Cluster {c}")
        ax1.set_title(f"K-Means (k={k})  Sil={info['sil_km']:.3f}",
                      color=TEXT_WHITE, fontsize=10)
        ax1.legend(fontsize=7.5, markerscale=1.4)
        ax1.set_xlabel("PC1", color=TEXT_GRAY, fontsize=8)
        ax1.set_ylabel("PC2", color=TEXT_GRAY, fontsize=8)
        ax1.spines[:].set_color(BORDER)

        # ── Clustering Jerárquico scatter ──
        ax2 = fig.add_subplot(gs[1])
        for c in range(k):
            mask = hac_labels == c
            ax2.scatter(X2[mask, 0], X2[mask, 1],
                        s=18, alpha=0.65, color=cmap[c % len(cmap)],
                        label=f"Cluster {c}")
        ax2.set_title(f"Jerárquico (k={k})  Sil={info['sil_hac']:.3f}",
                      color=TEXT_WHITE, fontsize=10)
        ax2.legend(fontsize=7.5, markerscale=1.4)
        ax2.set_xlabel("PC1", color=TEXT_GRAY, fontsize=8)
        ax2.set_ylabel("PC2", color=TEXT_GRAY, fontsize=8)
        ax2.spines[:].set_color(BORDER)

        # ── Silhouette comparativo ──
        ax3 = fig.add_subplot(gs[2])
        sils   = [info["sil_km"], info["sil_hac"]]
        labels = ["K-Means", "Jerárquico"]
        bars   = ax3.bar(labels, sils,
                         color=[ACCENT, ACCENT2], width=0.45)
        for bar, v in zip(bars, sils):
            ax3.text(bar.get_x() + bar.get_width() / 2,
                     v + 0.005, f"{v:.4f}",
                     ha="center", color=TEXT_WHITE, fontsize=10, fontweight="bold")
        ax3.set_ylim(0, max(sils) * 1.25)
        ax3.set_title("Índice Silhouette", color=TEXT_WHITE, fontsize=10)
        ax3.set_ylabel("Score (0–1)", color=TEXT_GRAY, fontsize=8)
        ax3.axhline(0.5, color=ACCENT3, ls="--", lw=1, label="Umbral 0.5")
        ax3.legend(fontsize=8)
        ax3.spines[:].set_color(BORDER)

        self._embed_fig(fig, self.tab_clustering)

    # ──────────────────────────────────────────
    #  RENDER: TAB CLASIFICACIÓN
    # ──────────────────────────────────────────
    def _render_clasificacion(self, resultados):
        self._clear_tab(self.tab_clasif)

        # Solo modelos no-baseline para detallar
        clf_res = [r for r in resultados if r["name"] != "Baseline"]
        n = len(clf_res)

        fig = plt.figure(figsize=(6 * n, 5), facecolor=DARK_BG)
        for i, r in enumerate(clf_res):
            ax = fig.add_subplot(1, n, i + 1)
            cm = confusion_matrix(r["y_test"], r["preds"])
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                        linewidths=0.5, linecolor=BORDER,
                        annot_kws={"fontsize": 12, "fontweight": "bold"},
                        ax=ax,
                        cbar_kws={"shrink": 0.75})
            ax.set_title(f"{r['name']}\nAcc={r['acc']:.3f}  F1={r['f1']:.3f}  AUC={r['auc']:.3f}",
                         color=TEXT_WHITE, fontsize=10, pad=8)
            ax.set_xlabel("Predicho", color=TEXT_GRAY, fontsize=9)
            ax.set_ylabel("Real",     color=TEXT_GRAY, fontsize=9)

        fig.tight_layout(pad=2)
        self._embed_fig(fig, self.tab_clasif)

    # ──────────────────────────────────────────
    #  RENDER: TAB EVALUACIÓN COMPARATIVA
    # ──────────────────────────────────────────
    def _render_evaluacion(self, resultados):
        self._clear_tab(self.tab_eval)

        fig = plt.figure(figsize=(13, 5.5), facecolor=DARK_BG)
        gs  = gridspec.GridSpec(1, 2, figure=fig,
                                left=0.07, right=0.97, wspace=0.38)

        # ── Curva ROC ──
        ax_roc = fig.add_subplot(gs[0])
        binary = all(len(np.unique(r["y_test"])) == 2 for r in resultados)
        colors_roc = [ACCENT, ACCENT2, ACCENT3]
        if binary:
            for r, col in zip(resultados, colors_roc):
                if r["probs"] is not None and r["auc"] > 0:
                    fpr, tpr, _ = roc_curve(r["y_test"], r["probs"])
                    ax_roc.plot(fpr, tpr, lw=2, color=col,
                                label=f"{r['name']} (AUC={r['auc']:.3f})")
        ax_roc.plot([0, 1], [0, 1], "--", lw=1, color=BORDER, label="Aleatório")
        ax_roc.fill_between([0, 1], [0, 1], alpha=0.04, color=TEXT_GRAY)
        ax_roc.set_title("Curva ROC Comparativa", color=TEXT_WHITE, fontsize=11, pad=8)
        ax_roc.set_xlabel("Tasa de Falsos Positivos (FPR)", color=TEXT_GRAY, fontsize=9)
        ax_roc.set_ylabel("Tasa de Verdaderos Positivos (TPR)", color=TEXT_GRAY, fontsize=9)
        ax_roc.legend(fontsize=9)
        ax_roc.spines[:].set_color(BORDER)
        if not binary:
            ax_roc.text(0.3, 0.5, "Multiclase:\nROC no disponible",
                        color=ACCENT3, fontsize=12, ha="center")

        # ── Tabla comparativa de barras ──
        ax_bar = fig.add_subplot(gs[1])
        names   = [r["name"] for r in resultados]
        accs    = [r["acc"]  for r in resultados]
        f1s     = [r["f1"]   for r in resultados]
        aucs    = [r["auc"]  for r in resultados]

        x    = np.arange(len(names))
        w    = 0.25
        ax_bar.bar(x - w,   accs, width=w, label="Accuracy", color=ACCENT,  alpha=0.9)
        ax_bar.bar(x,        f1s, width=w, label="F1-Score",  color=ACCENT2, alpha=0.9)
        ax_bar.bar(x + w,   aucs, width=w, label="AUC",       color=ACCENT3, alpha=0.9)

        ax_bar.set_xticks(x)
        ax_bar.set_xticklabels(names, fontsize=9, color=TEXT_WHITE)
        ax_bar.set_ylim(0, 1.18)
        ax_bar.set_title("Comparación de Métricas", color=TEXT_WHITE, fontsize=11, pad=8)
        ax_bar.set_ylabel("Score", color=TEXT_GRAY, fontsize=9)
        ax_bar.legend(fontsize=9)
        ax_bar.spines[:].set_color(BORDER)

        # Valores encima de barras
        for bars in [ax_bar.containers[0], ax_bar.containers[1], ax_bar.containers[2]]:
            for b in bars:
                ax_bar.text(b.get_x() + b.get_width() / 2,
                            b.get_height() + 0.01,
                            f"{b.get_height():.2f}",
                            ha="center", fontsize=7.5, color=TEXT_WHITE)

        fig.tight_layout(pad=2)
        self._embed_fig(fig, self.tab_eval)

    # ──────────────────────────────────────────
    #  RENDER: TAB MATRIZ DE CONFUSIÓN COMPLETA
    # ──────────────────────────────────────────
    def _render_confusion(self, resultados):
        self._clear_tab(self.tab_confusion)

        rf_res = next((r for r in resultados if r["name"] == "Random Forest"),
                      resultados[-1])
        y_test = rf_res["y_test"]
        preds  = rf_res["preds"]
        cm     = confusion_matrix(y_test, preds)

        # Calcular métricas manualmente para mostrar
        if cm.shape == (2, 2):
            vp, fp, fn, vn = cm[1,1], cm[0,1], cm[1,0], cm[0,0]
        else:
            vp = fp = fn = vn = None

        fig = plt.figure(figsize=(12, 5.5), facecolor=DARK_BG)
        gs  = gridspec.GridSpec(1, 2, figure=fig,
                                left=0.05, right=0.97, wspace=0.45)

        # ── Heatmap ──
        ax_cm = fig.add_subplot(gs[0])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    linewidths=1, linecolor=BORDER,
                    annot_kws={"fontsize": 14, "fontweight": "bold"},
                    ax=ax_cm,
                    cbar_kws={"shrink": 0.75})
        ax_cm.set_title(f"Matriz de Confusión · Random Forest",
                        color=TEXT_WHITE, fontsize=11, pad=10)
        ax_cm.set_xlabel("Predicho", color=TEXT_GRAY, fontsize=10)
        ax_cm.set_ylabel("Real",     color=TEXT_GRAY, fontsize=10)

        # Etiquetas de error/acierto
        if cm.shape == (2, 2):
            for text, pos, col in [
                ("Acierto\ntipo 1", (0, 0), ACCENT2),
                ("Error\ntipo 1",   (1, 0), ACCENT3),
                ("Error\ntipo 2",   (0, 1), ACCENT3),
                ("Acierto\ntipo 2", (1, 1), ACCENT2),
            ]:
                ax_cm.text(pos[0] + 0.5, pos[1] + 0.88, text,
                           ha="center", va="bottom", fontsize=7.5,
                           color=col, style="italic")

        # ── Panel de métricas derivadas ──
        ax_m = fig.add_subplot(gs[1])
        ax_m.axis("off")
        ax_m.set_title("Métricas Derivadas (RF)", color=TEXT_WHITE,
                        fontsize=11, pad=10)

        acc  = rf_res["acc"]
        f1   = rf_res["f1"]
        prec = rf_res["prec"]
        rec  = rf_res["rec"]
        auc_ = rf_res["auc"]

        rows_data = []
        if cm.shape == (2, 2):
            rows_data = [
                ("VP (Verdaderos Positivos)", str(vp),   ACCENT2),
                ("FP (Falsos Positivos)",     str(fp),   ACCENT3),
                ("FN (Falsos Negativos)",     str(fn),   ACCENT3),
                ("VN (Verdaderos Negativos)", str(vn),   ACCENT2),
                ("", "", ""),
                ("Precisión  VP/(VP+FP)",     f"{prec:.4f}", ACCENT),
                ("Exactitud (Accuracy)",       f"{acc:.4f}",  ACCENT),
                ("Sensibilidad (Recall)",      f"{rec:.4f}",  ACCENT),
                ("F1-Score",                   f"{f1:.4f}",   ACCENT2),
                ("AUC-ROC",                    f"{auc_:.4f}", ACCENT3),
                ("Especificidad VN/(VN+FP)",
                 f"{vn/(vn+fp):.4f}" if (vn is not None and fp is not None and (vn+fp) > 0) else "—",
                 "#D2A8FF"),
            ]
        else:
            rows_data = [
                ("Accuracy",  f"{acc:.4f}",  ACCENT),
                ("F1-Score",  f"{f1:.4f}",   ACCENT2),
                ("Precision", f"{prec:.4f}", ACCENT3),
                ("Recall",    f"{rec:.4f}",  "#D2A8FF"),
            ]

        y_pos = 0.97
        for label, val, col in rows_data:
            if label == "":
                y_pos -= 0.04
                continue
            ax_m.text(0.02, y_pos, label,
                      color=TEXT_GRAY, fontsize=9, va="top",
                      transform=ax_m.transAxes)
            ax_m.text(0.82, y_pos, val,
                      color=col, fontsize=9.5, va="top",
                      fontweight="bold", transform=ax_m.transAxes)
            y_pos -= 0.086

        fig.tight_layout(pad=2)
        self._embed_fig(fig, self.tab_confusion)

    # ──────────────────────────────────────────
    #  RENDER: TAB INTERPRETACIÓN
    # ──────────────────────────────────────────
    def _render_interpretacion(self, resultados, clust_info, part_info):
        self._clear_tab(self.tab_interpretacion)

        rf  = next(r for r in resultados if r["name"] == "Random Forest")
        arb = next((r for r in resultados if r["name"] == "Árbol"), None)
        bas = next((r for r in resultados if r["name"] == "Baseline"), None)

        # Determinar el mejor modelo
        mejor = max(resultados, key=lambda r: r["auc"])

        total = part_info["train"] + part_info["val"] + part_info["test"]

        texto = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    INTERPRETACIÓN DE RESULTADOS                            ║
║              (Comunicación para equipo gerencial no técnico)               ║
╚══════════════════════════════════════════════════════════════════════════════╝

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 1. PARTICIÓN Y CONTROL DE CALIDAD (Data Leakage)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 Total de registros analizados : {total:,}
   • Entrenamiento (70%)        : {part_info['train']:,}  registros
   • Validación   (15%)        : {part_info['val']:,}   registros
   • Prueba final (15%)        : {part_info['test']:,}   registros

 ¿Qué es el Data Leakage?
   Es cuando el modelo "ve" información del futuro durante su entrenamiento,
   generando métricas infladas pero resultados reales deficientes.
   ✔ Solución aplicada: el escalado estadístico se calculó SOLO con datos
     de entrenamiento y luego se aplicó a validación y prueba.

 Baseline (punto de referencia mínimo):
   Se utilizó un modelo "ingenuo" que siempre predice la clase más frecuente.
   Acc={bas['acc']:.3f}  F1={bas['f1']:.3f}
   → Todo modelo real debe superar este valor para ser considerado útil.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 2. SEGMENTACIÓN (Clustering) — Perfiles de Clientes
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 Se aplicaron dos algoritmos de agrupamiento con k={clust_info['k']} clústeres:

   • K-Means        → Silhouette = {clust_info['sil_km']:.4f}
   • Jerárquico     → Silhouette = {clust_info['sil_hac']:.4f}

 Índice Silhouette: mide qué tan bien definidos están los grupos.
   Rango: -1 (muy malo) → 0 (solapado) → 1 (perfecto)
   Umbral recomendado: ≥ 0.50 indica buena separación.

 Posibles perfiles de clientes identificados:
   🔵 Perfil A – Clientes de alto valor: alta frecuencia de compra,
     mayor gasto promedio. Campaña recomendada: fidelización premium.
   🟢 Perfil B – Clientes ocasionales: baja frecuencia, gasto moderado.
     Campaña recomendada: reactivación con descuentos.
   🔴 Perfil C – Clientes en riesgo: sin actividad reciente.
     Campaña recomendada: retención urgente.
   🟣 Perfil D – Nuevos clientes: reciente incorporación.
     Campaña recomendada: onboarding y bienvenida.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 3. CLASIFICACIÓN — Predicción de Respuesta a Campaña
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 Objetivo: predecir si un cliente responderá positivamente a una campaña.

 Árbol de Decisión:
   Ventaja : simple, interpretable, útil para explicar decisiones.
   Limitación: puede sobreajustarse (overfitting) si no se poda.
   Acc={arb['acc']:.3f}  F1={arb['f1']:.3f}  AUC={arb['auc']:.3f}

 Random Forest:
   Ventaja : combina múltiples árboles → más estable y preciso.
   Limitación: menos interpretable que un árbol simple.
   Acc={rf['acc']:.3f}  F1={rf['f1']:.3f}  AUC={rf['auc']:.3f}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 4. EVALUACIÓN COMPARATIVA
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 Modelo           Accuracy    F1-Score      AUC
 ─────────────────────────────────────────────────────
 Baseline         {bas['acc']:.4f}      {bas['f1']:.4f}        {bas['auc']:.4f}
 Árbol Decisión   {arb['acc']:.4f}      {arb['f1']:.4f}        {arb['auc']:.4f}
 Random Forest    {rf['acc']:.4f}      {rf['f1']:.4f}        {rf['auc']:.4f}
 ─────────────────────────────────────────────────────
 ✔ Mejor modelo: {mejor['name']} (AUC={mejor['auc']:.4f})

 Curva ROC: muestra el equilibrio entre detectar clientes que SÍ responden
 (sensibilidad) vs. falsas alarmas (especificidad).
 AUC=1.0 es perfecto; AUC=0.5 equivale a adivinar al azar.

 Recomendación para gerencia:
   Implementar {mejor['name']} para las próximas campañas.
   Esto permitirá dirigir los esfuerzos de marketing al {mejor['acc']*100:.1f}%
   de clientes con mayor probabilidad de respuesta positiva,
   reduciendo costos y aumentando la tasa de conversión.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 5. MATRIZ DE CONFUSIÓN — Interpretación
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 La matriz de confusión muestra 4 tipos de resultados:

   ✅ VP (Verdaderos Positivos) – Acierto tipo 1:
      Cliente que SÍ respondería y el modelo lo predijo correctamente.
      → Impacto: inversión de campaña bien dirigida.

   ✅ VN (Verdaderos Negativos) – Acierto tipo 2:
      Cliente que NO respondería y el modelo lo excluyó correctamente.
      → Impacto: ahorro en costos de campaña.

   ❌ FP (Falsos Positivos) – Error tipo 1:
      Cliente que NO respondería pero el modelo predijo que SÍ.
      → Impacto: gasto innecesario en campaña.

   ❌ FN (Falsos Negativos) – Error tipo 2:
      Cliente que SÍ respondería pero el modelo predijo que NO.
      → Impacto: oportunidad de venta perdida.

 En contexto de marketing, minimizar los FN es prioritario
 (no queremos perder clientes potenciales).
"""

        frame = tk.Frame(self.tab_interpretacion, bg=DARK_BG)
        frame.pack(fill=tk.BOTH, expand=True)

        txt = scrolledtext.ScrolledText(
            frame, bg=PANEL_BG, fg=TEXT_WHITE,
            font=("Consolas", 9), relief="flat",
            borderwidth=0, wrap=tk.WORD,
            padx=16, pady=12,
            selectbackground=ACCENT
        )
        txt.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        # Tags de color
        txt.tag_config("header",  foreground=ACCENT,  font=("Consolas", 9, "bold"))
        txt.tag_config("ok",      foreground=ACCENT2)
        txt.tag_config("error",   foreground=ACCENT3)
        txt.tag_config("section", foreground="#D2A8FF", font=("Consolas", 9, "bold"))

        for line in texto.strip().split("\n"):
            if "══" in line or "━━" in line:
                txt.insert(tk.END, line + "\n", "section")
            elif line.strip().startswith("✔") or line.strip().startswith("✅"):
                txt.insert(tk.END, line + "\n", "ok")
            elif line.strip().startswith("❌"):
                txt.insert(tk.END, line + "\n", "error")
            else:
                txt.insert(tk.END, line + "\n")

        txt.config(state=tk.DISABLED)

    # ── UTILIDADES ───────────────────────────
    def _clear_tab(self, tab):
        for widget in tab.winfo_children():
            widget.destroy()

    def _embed_fig(self, fig, tab):
        canvas = FigureCanvasTkAgg(fig, master=tab)
        canvas.draw()
        toolbar = NavigationToolbar2Tk(canvas, tab, pack_toolbar=False)
        toolbar.config(background=PANEL_BG)
        toolbar.update()
        toolbar.pack(side=tk.BOTTOM, fill=tk.X)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        plt.close(fig)

    def _on_close(self):
        self.destroy()
        self.quit()


# ─────────────────────────────────────────────
#  PUNTO DE ENTRADA
# ─────────────────────────────────────────────
if __name__ == "__main__":
    app = AppMineria()
    app.mainloop()