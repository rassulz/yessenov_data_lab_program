"""build_figs.py — generate the 3 narrative figures (EN + RU) for the report.

Run with the Anaconda interpreter (has matplotlib):
    C:/ProgramData/anaconda3/python.exe build_figs.py
Writes fig1_<lang>.png, fig2_<lang>.png, fig3_<lang>.png into this folder.
All numbers are the project's real OOF / public-LB results (see results_log.csv).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from pathlib import Path

HERE = Path(__file__).resolve().parent
plt.rcParams["font.family"] = "DejaVu Sans"      # bundled with matplotlib, full Cyrillic
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3
plt.rcParams["figure.dpi"] = 150

INK = "#1b2a4a"; WALL = "#c0392b"; WIN = "#1e8449"; GOLD = "#d4af37"
TREE = "#e07b39"; KERNEL = "#2e86c1"; NEURAL = "#8e44ad"; GEN = "#16a085"; ANCH = "#7f8c8d"

# ---- real results (OOF macro-F1, error-correlation with CatBoost) ----
SINGLE = [
    # name, oof, errcorr, family, color
    ("CatBoost", 0.82898, 1.000, "tree", ANCH),
    ("GradBoost", 0.81284, 0.781, "tree", TREE),
    ("ExtraTrees", 0.79462, 0.722, "tree", TREE),
    ("RandomForest", 0.79203, 0.737, "tree", TREE),
    ("SVM-RBF", 0.83331, 0.759, "kernel", KERNEL),
    ("MLP", 0.83677, 0.680, "neural", NEURAL),
    ("Nystroem", 0.82304, 0.762, "kernel", KERNEL),
    ("QDA", 0.80836, 0.605, "generative", GEN),
    ("GMM(2)", 0.78867, 0.520, "generative", GEN),
    ("GMM(3)", 0.78170, 0.495, "generative", GEN),
    ("LDA", 0.73836, 0.462, "generative", GEN),
    ("GaussianNB", 0.62449, 0.364, "generative", GEN),
]

TXT = {
    "en": {
        "f1title": "The journey of a macro-F1 score",
        "f1y": "OOF macro-F1 (local cross-validation)",
        "wall": "THE WALL  ~0.829",
        "steps": ["Baseline\nCatBoost", "Tuned\nCatBoost", "Tree\nensembles", "MLP\n(set aside)",
                  "SVM-RBF\nbag", "CatBoost+SVM\n(sub06)", "Multi-family\nceiling"],
        "ann_ord": "Ordinary world:\na clean CatBoost", "ann_wall": "hit the wall",
        "ann_back": "tried harder with\nmore trees -> backwards",
        "ann_cross": "crossed into\nnon-tree land",
        "ann_win": "TRIUMPH\npublic LB 0.84157",
        "ann_ceil": "the true\nceiling",
        "pub": "public LB",
        "f2title": "A good ally must be STRONG and DIFFERENT",
        "f2x": "Strength  ->  OOF macro-F1", "f2y": "Redundancy  ->  error-correlation with CatBoost",
        "f2zone": "the useful zone\n(strong AND decorrelated)",
        "f2strong": "must be at least as strong as CatBoost",
        "f2weak": "diverse but too weak\n(generative models)",
        "f3title": "The leaderboard, where it actually counts (public 30%)",
        "f3y": "Public leaderboard macro-F1",
        "f3wall": "best single CatBoost",
        "legend_tree": "tree", "legend_kernel": "kernel (SVM)", "legend_neural": "neural",
        "legend_gen": "generative/linear",
    },
    "ru": {
        "f1title": "Путешествие одной macro-F1",
        "f1y": "OOF macro-F1 (локальная кросс-валидация)",
        "wall": "СТЕНА  ~0.829",
        "steps": ["Базовый\nCatBoost", "Настроенный\nCatBoost", "Ансамбли\nдеревьев", "MLP\n(отложен)",
                  "SVM-RBF\nбэг", "CatBoost+SVM\n(sub06)", "Потолок\n(мульти-семья)"],
        "ann_ord": "Обычный мир:\nчистый CatBoost", "ann_wall": "упёрся в стену",
        "ann_back": "старался сильнее с деревьями\n-> стало хуже",
        "ann_cross": "шагнул в землю\nне-деревьев",
        "ann_win": "ТРИУМФ\npublic LB 0.84157",
        "ann_ceil": "истинный\nпотолок",
        "pub": "public LB",
        "f2title": "Хороший союзник должен быть СИЛЬНЫМ и ДРУГИМ",
        "f2x": "Сила  ->  OOF macro-F1", "f2y": "Похожесть  ->  корреляция ошибок с CatBoost",
        "f2zone": "полезная зона\n(сильный И декоррелированный)",
        "f2strong": "должен быть не слабее CatBoost",
        "f2weak": "разнообразны, но слабы\n(генеративные модели)",
        "f3title": "Лидерборд, где это реально считается (public 30%)",
        "f3y": "Public leaderboard macro-F1",
        "f3wall": "лучший одиночный CatBoost",
        "legend_tree": "дерево", "legend_kernel": "ядро (SVM)", "legend_neural": "нейросеть",
        "legend_gen": "генеративные/линейные",
    },
}


def fig1(lang):
    t = TXT[lang]
    xs = list(range(7))
    oof = [0.82494, 0.82898, 0.82324, 0.83677, 0.83331, 0.83269, 0.83374]
    fig, ax = plt.subplots(figsize=(10.5, 5.4))
    ax.axhline(0.82898, ls="--", lw=1.6, color=WALL, alpha=0.8)
    ax.text(6.05, 0.8292, t["wall"], color=WALL, fontsize=10, va="center", fontweight="bold")
    ax.plot(xs, oof, "-o", lw=2.4, color=INK, ms=8, zorder=3)
    # color the breakthrough segment
    ax.plot(xs[3:], oof[3:], "-o", lw=2.6, color=WIN, ms=8, zorder=4)
    ax.scatter([3], [0.83677], s=170, facecolors="none", edgecolors=NEURAL, lw=2.2, zorder=5)
    # public LB markers
    ax.scatter([0, 1], [0.82809, 0.83026], marker="s", s=55, color=ANCH, zorder=4)
    ax.scatter([5], [0.84157], marker="*", s=420, color=GOLD, edgecolor=INK, lw=1.1, zorder=6)
    ax.annotate(t["pub"] + " 0.842", (5, 0.84157), (4.0, 0.8452),
                fontsize=10, color=INK, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=INK))
    ax.annotate(t["ann_ord"], (0, 0.82494), (-0.35, 0.8205), fontsize=8.5, color=INK)
    ax.annotate(t["ann_back"], (2, 0.82324), (1.35, 0.8175), fontsize=8.5, color=WALL,
                arrowprops=dict(arrowstyle="->", color=WALL))
    ax.annotate(t["ann_cross"], (3, 0.83677), (2.55, 0.8395), fontsize=8.5, color=WIN)
    ax.annotate(t["ann_ceil"], (6, 0.83374), (5.55, 0.8358), fontsize=8.5, color=INK)
    ax.set_xticks(xs); ax.set_xticklabels(t["steps"], fontsize=9)
    ax.set_ylabel(t["f1y"], fontsize=10.5)
    ax.set_title(t["f1title"], fontsize=14, fontweight="bold", color=INK, pad=12)
    ax.set_ylim(0.812, 0.848)
    fig.tight_layout(); fig.savefig(HERE / f"fig1_{lang}.png", bbox_inches="tight"); plt.close(fig)


def fig2(lang):
    t = TXT[lang]
    # explicit label offsets (dx, dy, ha) to avoid collisions
    LBL = {
        "CatBoost": (0, 0.028, "center"), "GradBoost": (0, 0.028, "center"),
        "ExtraTrees": (-0.004, -0.034, "center"), "RandomForest": (-0.004, 0.026, "center"),
        "SVM-RBF": (0.001, -0.040, "center"), "MLP": (0.0, 0.030, "center"),
        "Nystroem": (-0.006, 0.028, "center"), "QDA": (0, 0.028, "center"),
        "GMM(2)": (0, 0.028, "center"), "GMM(3)": (0.004, -0.034, "center"),
        "LDA": (0, 0.028, "center"), "GaussianNB": (0, 0.030, "center"),
    }
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    ax.axvspan(0.82898, 0.852, ymin=0, ymax=0.62, color=WIN, alpha=0.08, zorder=0)
    ax.axvline(0.82898, ls="--", lw=1.5, color=ANCH, alpha=0.8)
    ax.text(0.8268, 0.90, t["f2strong"], rotation=90, fontsize=8.5, color="#555",
            va="center", ha="center")
    seen = set()
    for name, oof, ec, fam, col in SINGLE:
        lbl = {"tree": t["legend_tree"], "kernel": t["legend_kernel"],
               "neural": t["legend_neural"], "generative": t["legend_gen"]}[fam]
        ax.scatter(oof, ec, s=140, color=col, edgecolor="white", lw=1.2, zorder=3,
                   label=lbl if fam not in seen else None)
        seen.add(fam)
        dx, dy, ha = LBL[name]
        ax.annotate(name, (oof + dx, ec + dy), fontsize=8.7, ha=ha, color=INK)
    for name, oof, ec in [("SVM-RBF", 0.83331, 0.759), ("MLP", 0.83677, 0.680)]:
        ax.scatter(oof, ec, s=340, marker="*", facecolors="none", edgecolors=GOLD, lw=2.3, zorder=4)
    ax.annotate(t["f2zone"], xy=(0.8315, 0.45), xytext=(0.745, 0.36), fontsize=10, color=WIN,
                fontweight="bold", ha="center", va="center",
                arrowprops=dict(arrowstyle="->", color=WIN, lw=1.6))
    ax.annotate(t["f2weak"], (0.783, 0.515), (0.638, 0.45), fontsize=9, color=GEN,
                arrowprops=dict(arrowstyle="->", color=GEN))
    ax.set_xlabel(t["f2x"], fontsize=10.5); ax.set_ylabel(t["f2y"], fontsize=10.5)
    ax.set_title(t["f2title"], fontsize=14, fontweight="bold", color=INK, pad=12)
    ax.set_xlim(0.60, 0.852); ax.set_ylim(0.32, 1.05)
    ax.legend(loc="lower left", fontsize=9, framealpha=0.9)
    fig.tight_layout(); fig.savefig(HERE / f"fig2_{lang}.png", bbox_inches="tight"); plt.close(fig)


def fig3(lang):
    t = TXT[lang]
    subs = ["sub00\nCatBoost\nbaseline", "sub02\nSoftVote\n(trees)", "sub03\nseed-bag",
            "sub01\nCatBoost\ntuned", "sub06\nCatBoost+SVM"]
    pub = [0.82809, 0.82478, 0.82820, 0.83026, 0.84157]
    cols = [ANCH, TREE, ANCH, KERNEL, GOLD]
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    bars = ax.bar(range(len(subs)), pub, color=cols, edgecolor=INK, lw=0.8, width=0.62, zorder=3)
    bars[-1].set_edgecolor(INK); bars[-1].set_linewidth(2.0)
    ax.axhline(0.83026, ls="--", lw=1.4, color=ANCH, alpha=0.8)
    ax.text(0.0, 0.8308, t["f3wall"], fontsize=9, color="#555")
    for i, v in enumerate(pub):
        ax.text(i, v + 0.0009, f"{v:.5f}", ha="center", fontsize=9.5,
                fontweight="bold" if i == len(pub) - 1 else "normal", color=INK)
    ax.set_xticks(range(len(subs))); ax.set_xticklabels(subs, fontsize=9)
    ax.set_ylabel(t["f3y"], fontsize=10.5)
    ax.set_title(t["f3title"], fontsize=14, fontweight="bold", color=INK, pad=12)
    ax.set_ylim(0.818, 0.846)
    fig.tight_layout(); fig.savefig(HERE / f"fig3_{lang}.png", bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    for lang in ("en", "ru"):
        fig1(lang); fig2(lang); fig3(lang)
        print("wrote figures for", lang)
    print("done")
