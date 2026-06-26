"""
build_notebooks.py — regenerate the 6 pipeline notebooks from one source of truth.

We keep ALL analysis logic inside the notebooks (course style), but to guarantee
that every notebook uses the *exact same* metric, cross-validation folds, OOF
harness and feature-engineering code, those shared snippets are defined here as
string constants and injected into each notebook. That makes the notebooks
self-contained (open and run top-to-bottom) while staying drift-free.

Run with the Anaconda interpreter:
    C:/ProgramData/anaconda3/python.exe build_notebooks.py
Then execute them in order (run_all.py does this headlessly).
"""
import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

KERNEL = {"name": "anaconda3", "display_name": "Python (anaconda3)", "language": "python"}


def make_nb(cells):
    nb = new_notebook()
    nb.metadata["kernelspec"] = KERNEL
    nb.metadata["language_info"] = {"name": "python"}
    nb.cells = cells
    return nb


def md(text):
    return new_markdown_cell(text)


def code(src):
    return new_code_cell(src)


# ===========================================================================
# Shared source snippets (single source of truth, injected into notebooks)
# ===========================================================================

TOOLBOX = r'''# --- Shared toolbox (identical across notebooks; see build_notebooks.py) ---
import warnings, json
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, classification_report, confusion_matrix

SEED = 42
N_FOLDS = 5
CLASS_NAMES = {0: "Wake", 1: "Light", 2: "Deep", 3: "REM"}
CLASSES = np.array([0, 1, 2, 3])
EOG = "eog_burst_index"            # the only column with missing values (~50%)

RAW_FEATURES = [
    "eeg_delta_power", "eeg_theta_power", "eeg_alpha_power", "eeg_sigma_power",
    "eeg_beta_power", "eeg_gamma_power", "eeg_slow_osc_power", "eeg_spectral_entropy",
    "eeg_spindle_density", "eeg_kcomplex_rate", "emg_chin_tone", "emg_tone_variance",
    "eog_movement_density", "eog_amplitude", "heart_rate_mean", "heart_rate_variability",
    "respiration_rate", "respiration_variability", "spo2_mean", "body_movement_index",
    EOG,
]
POWER = ["eeg_delta_power", "eeg_theta_power", "eeg_alpha_power", "eeg_sigma_power",
         "eeg_beta_power", "eeg_gamma_power", "eeg_slow_osc_power"]

HERE = Path.cwd()
ART = HERE / "artifacts"; ART.mkdir(exist_ok=True)
SUB = HERE / "submissions"; SUB.mkdir(exist_ok=True)


def load_data():
    """Return (train_df, test_df). Features kept as-is (NaN preserved)."""
    tr = pd.read_csv(HERE / "train.csv")
    te = pd.read_csv(HERE / "test.csv")
    return tr, te


def macro_f1(y_true, y_pred):
    """Competition metric: macro-averaged F1 over the 4 classes."""
    return f1_score(y_true, y_pred, average="macro")


def per_class_f1(y_true, y_pred):
    f = f1_score(y_true, y_pred, average=None, labels=CLASSES)
    return {CLASS_NAMES[c]: round(float(f[i]), 4) for i, c in enumerate(CLASSES)}


def softplus(x):
    """Numerically stable log(1+exp(x)); strictly positive and monotonic.
    Used to turn z-scored band powers (~50% negative) into positive magnitudes
    so band ratios are well-defined instead of dividing by near-zero."""
    x = np.asarray(x, dtype=float)
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


def _aligned_proba(model, X):
    """predict_proba with columns aligned to CLASSES = [0,1,2,3]."""
    p = model.predict_proba(X)
    cls = list(np.asarray(model.classes_))
    idx = [cls.index(c) for c in CLASSES]
    return p[:, idx]


def run_oof(make_model, X, y, X_test, folds, needs_impute=False, use_eval_set=False):
    """Out-of-fold training under fixed folds.

    Returns (oof, test_p, oof_macro, fold_scores):
      oof     : (n_train, 4) out-of-fold probabilities (each row predicted once)
      test_p  : (n_test, 4) test probabilities, averaged over the 5 fold-models
      oof_macro: global macro-F1 over the assembled OOF matrix (primary metric)

    Two model families, identical folds:
      - CatBoost (needs_impute=False): NaN passed through natively.
      - sklearn trees (needs_impute=True): add EOG-missing flag + fill EOG NaN
        with the TRAIN-FOLD median (fit on train fold only -> no leakage)."""
    n = len(y)
    oof = np.zeros((n, len(CLASSES)))
    test_p = np.zeros((len(X_test), len(CLASSES)))
    fold_scores = []
    for tr_idx, va_idx in folds:
        Xtr, Xva, Xte = X.iloc[tr_idx].copy(), X.iloc[va_idx].copy(), X_test.copy()
        ytr, yva = y[tr_idx], y[va_idx]
        if needs_impute:
            med = Xtr[EOG].median()
            for d in (Xtr, Xva, Xte):
                if EOG + "_missing" not in d.columns:
                    d[EOG + "_missing"] = d[EOG].isna().astype("int8")
                d[EOG] = d[EOG].fillna(med)
            assert not Xtr.isna().any().any(), "NaN remained after impute"
        model = make_model()
        if use_eval_set:
            model.fit(Xtr, ytr, eval_set=(Xva, yva))
        else:
            model.fit(Xtr, ytr)
        oof[va_idx] = _aligned_proba(model, Xva)
        test_p += _aligned_proba(model, Xte) / len(folds)
        fold_scores.append(macro_f1(yva, oof[va_idx].argmax(1)))
    oof_macro = macro_f1(y, oof.argmax(1))
    return oof, test_p, oof_macro, fold_scores


def load_folds():
    """Load the fixed StratifiedKFold split saved by 02_baseline."""
    d = np.load(ART / "folds.npz", allow_pickle=True)
    return [(d[f"tr{i}"], d[f"va{i}"]) for i in range(N_FOLDS)]
'''

ADD_FEATURES = r'''# --- z-score-safe feature engineering (single source of truth) ---
EPS = 1e-3

def add_features(df, groups):
    """Return RAW_FEATURES plus the requested engineered feature groups.

    Features are z-scored, so ~50% of every band-power value is <= 0. Raw ratios
    (delta/theta, ...) would divide by near-zero and flip sign -> meaningless.
    We therefore use (a) differences/sums (well-defined for z-scores) and
    (b) softplus-transformed ratios (positive magnitudes). The EOG channel is
    ~50% missing; we never derive features from it (only a missing-indicator)."""
    X = df[RAW_FEATURES].copy()
    miss = df[EOG].isna().astype("int8")          # informative channel on/off

    if "missing" in groups:
        X[EOG + "_missing"] = miss

    if "contrast" in groups:
        d, th, al, sig, be, ga, so = (df["eeg_delta_power"], df["eeg_theta_power"],
            df["eeg_alpha_power"], df["eeg_sigma_power"], df["eeg_beta_power"],
            df["eeg_gamma_power"], df["eeg_slow_osc_power"])
        X["delta_minus_theta"] = d - th
        X["theta_minus_alpha"] = th - al
        X["slowosc_minus_delta"] = so - d
        X["beta_minus_delta"] = be - d
        X["dt_minus_ab"] = (d + th) - (al + be)
        X["eeg_total"] = df[POWER].sum(axis=1)

    if "ratio" in groups:
        sp = {c: softplus(df[c]) for c in POWER}
        X["r_delta_theta"] = sp["eeg_delta_power"] / (sp["eeg_theta_power"] + EPS)
        X["r_theta_alpha"] = sp["eeg_theta_power"] / (sp["eeg_alpha_power"] + EPS)
        X["r_slowosc_delta"] = sp["eeg_slow_osc_power"] / (sp["eeg_delta_power"] + EPS)
        X["r_beta_delta"] = sp["eeg_beta_power"] / (sp["eeg_delta_power"] + EPS)
        X["r_dt_ab"] = (sp["eeg_delta_power"] + sp["eeg_theta_power"]) / (
            sp["eeg_alpha_power"] + sp["eeg_beta_power"] + EPS)

    if "relpower" in groups:
        sp = {c: softplus(df[c]) for c in POWER}
        tot = sum(sp.values()) + EPS
        for c in POWER:
            X["frac_" + c.replace("eeg_", "").replace("_power", "")] = sp[c] / tot

    if "autonomic" in groups:
        X["hr_over_resp"] = df["heart_rate_mean"] / (softplus(df["respiration_rate"]) + EPS)
        X["hrv_x_respvar"] = df["heart_rate_variability"] * df["respiration_variability"]
        X["move_x_emg"] = df["body_movement_index"] * df["emg_chin_tone"]

    if "eog" in groups:
        X["eog_amp_x_missing"] = df["eog_amplitude"] * miss
        X["eog_move_x_missing"] = df["eog_movement_density"] * miss

    return X
'''

LOG_HELPER = r'''def log_result(step, model, feature_set, oof_macro, pcf, notes=""):
    """Write one row to results_log.csv. Idempotent per (step, model): re-running
    a notebook replaces its own row rather than duplicating it."""
    import csv
    path = HERE / "results_log.csv"
    header = ["step", "model", "feature_set", "oof_macro_f1",
              "f1_Wake", "f1_Light", "f1_Deep", "f1_REM", "notes"]
    rows = []
    if path.exists():
        with open(path, newline="") as fh:
            data = list(csv.reader(fh))
        if data and data[0] == header:
            rows = [r for r in data[1:] if not (len(r) >= 2 and r[0] == step and r[1] == model)]
    row = [step, model, feature_set, round(float(oof_macro), 5),
           pcf["Wake"], pcf["Light"], pcf["Deep"], pcf["REM"], notes]
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header); w.writerows(rows); w.writerow(row)
    print("logged:", step, model, "OOF macro-F1 =", round(float(oof_macro), 5))
'''

# ===========================================================================
# 01_eda.ipynb
# ===========================================================================
nb01 = make_nb([
    md("# 01 — Exploratory Data Analysis\n"
       "**Task:** predict `sleep_stage` (0=Wake, 1=Light, 2=Deep, 3=REM) per 30-s epoch. "
       "**Metric:** macro-F1 (every class equal — the weakest class drags the score).\n\n"
       "This notebook only explores; it writes no model artifacts."),
    code(TOOLBOX),
    code('%matplotlib inline\n'
         'import matplotlib.pyplot as plt\n'
         'import seaborn as sns\n'
         'sns.set_theme(style="whitegrid")\n'
         'train, test = load_data()\n'
         'print("train", train.shape, "| test", test.shape)\n'
         'train.head()'),
    md("## Class balance\nClasses are nearly balanced, so a stratified split is enough. "
       "But macro-F1 still rewards fixing the *weakest* class, not overall accuracy."),
    code('counts = train["sleep_stage"].value_counts().sort_index()\n'
         'print({CLASS_NAMES[k]: int(v) for k, v in counts.items()})\n'
         'ax = counts.rename(CLASS_NAMES).plot.bar(rot=0, title="Class balance (train)")\n'
         'ax.set_ylabel("epochs"); plt.show()'),
    md("## Missingness\n`eog_burst_index` is the only column with missing values "
       "(~50% in both train and test). The EOG burst channel was simply off for some "
       "recordings, so *missingness is informative* — we add an `eog_burst_missing` flag "
       "and (for tree models that can't take NaN) impute with the train-fold median."),
    code('na = train.isna().sum(); print("train NaN:", na[na > 0].to_dict())\n'
         'na_t = test.isna().sum(); print("test  NaN:", na_t[na_t > 0].to_dict())\n'
         '# Does the channel being on/off relate to the stage?\n'
         'm = train[EOG].isna().astype(int)\n'
         'rate = train.groupby(m)["sleep_stage"].value_counts(normalize=True).unstack().round(3)\n'
         'rate.index = ["channel ON", "channel OFF"]\n'
         'print("\\nclass distribution by EOG-channel availability:"); print(rate)'),
    md("## Feature distributions by stage\nBoxplots show which features separate stages. "
       "Expect high delta/slow-osc in Deep, theta + eye movement in REM, high EMG tone and "
       "movement in Wake."),
    code('feats = [c for c in RAW_FEATURES if c != EOG]\n'
         'fig, axes = plt.subplots(5, 4, figsize=(18, 18))\n'
         'for ax, f in zip(axes.ravel(), feats):\n'
         '    sns.boxplot(data=train, x="sleep_stage", y=f, ax=ax, showfliers=False)\n'
         '    ax.set_title(f, fontsize=9); ax.set_xlabel("")\n'
         'for ax in axes.ravel()[len(feats):]:\n'
         '    ax.axis("off")\n'
         'plt.tight_layout(); plt.show()'),
    md("## Correlation heatmap\nHighly correlated band powers motivate the engineered "
       "contrasts/ratios in `03_features` (they capture *relative* power, which trees split on)."),
    code('corr = train[feats].corr()\n'
         'plt.figure(figsize=(12, 10))\n'
         'sns.heatmap(corr, cmap="coolwarm", center=0, square=True,\n'
         '            cbar_kws={"shrink": .6}, xticklabels=True, yticklabels=True)\n'
         'plt.title("Feature correlation"); plt.tight_layout(); plt.show()'),
    md("## Quick sanity model (OOB)\nA bagged tree with out-of-bag scoring gives a cheap "
       "first read on signal strength before we build the proper CV pipeline."),
    code('from sklearn.ensemble import BaggingClassifier\n'
         'from sklearn.tree import DecisionTreeClassifier\n'
         'Xq = train[feats].copy()          # drop EOG (NaN) for this quick check\n'
         'yq = train["sleep_stage"].values\n'
         'bag = BaggingClassifier(DecisionTreeClassifier(max_depth=None),\n'
         '        n_estimators=200, oob_score=True, n_jobs=-1, random_state=SEED)\n'
         'bag.fit(Xq, yq)\n'
         'print("Bagging OOB accuracy:", round(bag.oob_score_, 4))\n'
         'print("(accuracy only — macro-F1 via proper CV starts in 02_baseline)")'),
    md("### Takeaways\n"
       "- 4 near-balanced classes → StratifiedKFold(5); optimize the weakest class.\n"
       "- Only `eog_burst_index` is missing (~50%, informative) → flag + train-fold-median impute / native NaN.\n"
       "- Band powers are z-scored and correlated → build z-score-safe contrasts/ratios in `03`.\n"
       "- No id/recording grouping → plain stratified CV is valid (no leakage)."),
])

# ===========================================================================
# 02_baseline.ipynb
# ===========================================================================
nb02 = make_nb([
    md("# 02 — Baseline: CatBoost 5-fold OOF\n"
       "Establish the fixed CV folds and the number to beat. CatBoost is the primary model: "
       "it handles the 50% NaN in `eog_burst_index` natively and optimizes macro-F1 directly "
       "(`eval_metric='TotalF1:average=Macro'`). We also run a RandomForest as a cross-check."),
    code(TOOLBOX),
    code(LOG_HELPER),
    md("## Fix and save the folds\n`StratifiedKFold(5, shuffle=True, random_state=42)`. "
       "Saved to `artifacts/folds.npz` so **every** later notebook uses byte-identical folds "
       "— the precondition for valid OOF blending."),
    code('from sklearn.model_selection import StratifiedKFold\n'
         '(HERE / "results_log.csv").unlink(missing_ok=True)   # fresh evidence log per full run\n'
         'train, test = load_data()\n'
         'y = train["sleep_stage"].values.astype(int)\n'
         'test_ids = test["id"].values\n'
         'skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)\n'
         'folds = [(tr, va) for tr, va in skf.split(np.zeros(len(y)), y)]\n'
         'np.savez(ART / "folds.npz", **{f"tr{i}": tr for i, (tr, _) in enumerate(folds)},\n'
         '         **{f"va{i}": va for i, (_, va) in enumerate(folds)})\n'
         'np.save(ART / "y_train.npy", y)\n'
         'np.save(ART / "test_ids.npy", test_ids)\n'
         'print("folds saved; sizes:", [len(va) for _, va in folds])'),
    md("## CatBoost baseline (raw 21 features + missing flag)\n"
       "CatBoost keeps the NaN; we add an explicit `eog_burst_missing` flag on top "
       "(cheap, near-certain help)."),
    code('from catboost import CatBoostClassifier\n'
         '\n'
         'def build_base(df):\n'
         '    X = df[RAW_FEATURES].copy()\n'
         '    X[EOG + "_missing"] = df[EOG].isna().astype("int8")\n'
         '    return X\n'
         '\n'
         'Xtr_base, Xte_base = build_base(train), build_base(test)\n'
         '\n'
         'def make_catboost():\n'
         '    return CatBoostClassifier(\n'
         '        loss_function="MultiClass", eval_metric="TotalF1:average=Macro",\n'
         '        iterations=3000, learning_rate=0.03, depth=6, l2_leaf_reg=3.0,\n'
         '        random_seed=SEED, od_type="Iter", od_wait=150, use_best_model=True,\n'
         '        thread_count=-1, allow_writing_files=False, verbose=False)\n'
         '\n'
         'cat_oof, cat_test, cat_macro, cat_folds = run_oof(\n'
         '    make_catboost, Xtr_base, y, Xte_base, folds, needs_impute=False, use_eval_set=True)\n'
         'print("CatBoost baseline OOF macro-F1:", round(cat_macro, 5))\n'
         'print("per-fold:", [round(s, 4) for s in cat_folds],\n'
         '      "| mean %.4f +/- %.4f" % (np.mean(cat_folds), np.std(cat_folds)))\n'
         'print("per-class F1:", per_class_f1(y, cat_oof.argmax(1)))'),
    code('print(confusion_matrix(y, cat_oof.argmax(1)))\n'
         'print(classification_report(y, cat_oof.argmax(1),\n'
         '      target_names=[CLASS_NAMES[c] for c in CLASSES]))\n'
         'np.save(ART / "cat_base_oof.npy", cat_oof)\n'
         'np.save(ART / "cat_base_test.npy", cat_test)\n'
         'log_result("02_baseline", "catboost", "raw21+missing", cat_macro,\n'
         '           per_class_f1(y, cat_oof.argmax(1)), "primary baseline")'),
    md("## RandomForest cross-check\nSame folds, but sklearn trees can't take NaN → "
       "impute EOG with the train-fold median + keep the missing flag (handled in `run_oof`)."),
    code('from sklearn.ensemble import RandomForestClassifier\n'
         'def make_rf():\n'
         '    return RandomForestClassifier(n_estimators=600, max_features="sqrt",\n'
         '        min_samples_leaf=2, class_weight="balanced", n_jobs=-1, random_state=SEED)\n'
         'rf_oof, rf_test, rf_macro, rf_folds = run_oof(\n'
         '    make_rf, Xtr_base, y, Xte_base, folds, needs_impute=True)\n'
         'print("RandomForest OOF macro-F1:", round(rf_macro, 5))\n'
         'print("per-class F1:", per_class_f1(y, rf_oof.argmax(1)))\n'
         'log_result("02_baseline", "randomforest", "raw21+missing", rf_macro,\n'
         '           per_class_f1(y, rf_oof.argmax(1)), "cross-check")'),
    md("## First end-to-end submission (`sub00`)\nFold-bagged CatBoost test probabilities, "
       "plain argmax. Proves the submission path works before we optimize."),
    code('pred = cat_test.argmax(1)\n'
         'sub = pd.DataFrame({"id": test_ids, "sleep_stage": pred.astype(int)})\n'
         'assert sub.shape == (5000, 2) and sub["sleep_stage"].isin(CLASSES).all()\n'
         'sub.to_csv(SUB / "sub00_catboost_baseline.csv", index=False)\n'
         'print("wrote", SUB / "sub00_catboost_baseline.csv")\n'
         'print(sub["sleep_stage"].value_counts().sort_index().to_dict())'),
])

# ===========================================================================
# 03_features.ipynb
# ===========================================================================
nb03 = make_nb([
    md("# 03 — Feature engineering (z-score-safe) + greedy selection\n"
       "Band powers are z-scored, so ~50% of values are <= 0. Raw ratios would divide by "
       "near-zero and flip sign. We use **differences/sums** and **softplus ratios** instead. "
       "We add one feature group at a time and keep it only if it improves OOF macro-F1 by a "
       "margin larger than the per-fold noise (>0.002)."),
    code(TOOLBOX),
    code(ADD_FEATURES),
    code('from catboost import CatBoostClassifier\n'
         'train, test = load_data()\n'
         'y = np.load(ART / "y_train.npy")\n'
         'folds = load_folds()\n'
         '\n'
         '# Use the SAME full-power CatBoost as the final model so feature decisions\n'
         '# match what we actually deploy (a cheap proxy misjudged the missing flag).\n'
         '# Early stopping keeps it bounded.\n'
         'def make_cat():\n'
         '    return CatBoostClassifier(loss_function="MultiClass",\n'
         '        eval_metric="TotalF1:average=Macro", iterations=3000, learning_rate=0.03,\n'
         '        depth=6, l2_leaf_reg=3.0, random_seed=SEED, od_type="Iter", od_wait=150,\n'
         '        use_best_model=True, allow_writing_files=False, thread_count=-1, verbose=False)\n'
         '\n'
         'def score_groups(groups):\n'
         '    Xtr = add_features(train, groups); Xte = add_features(test, groups)\n'
         '    _, _, m, _ = run_oof(make_cat, Xtr, y, Xte, folds,\n'
         '                         needs_impute=False, use_eval_set=True)\n'
         '    return m'),
    md("## Greedy forward selection\nThe EOG missing-indicator is a standard, near-free "
       "treatment for *informative* missingness (and the baseline already showed it helps), "
       "so we always keep it. We greedily test only the heavier engineered groups, keeping "
       "one only if it beats the per-fold noise (>0.002)."),
    code('BASE_GROUPS = ["missing"]                       # always kept (informative missingness)\n'
         'CAND_GROUPS = ["contrast", "ratio", "relpower", "autonomic", "eog"]\n'
         'MARGIN = 0.002\n'
         'kept = list(BASE_GROUPS)\n'
         'best = score_groups(kept)\n'
         'print(f"base (raw 21 + missing flag): OOF macro-F1 = {best:.5f}")\n'
         'trials = [("raw21+missing", best, "base")]\n'
         'for g in CAND_GROUPS:\n'
         '    cand = score_groups(kept + [g])\n'
         '    keep = cand > best + MARGIN\n'
         '    verdict = "KEEP" if keep else "drop"\n'
         '    trials.append((g, cand, verdict))\n'
         '    print(f"  + {g:<10} -> {cand:.5f}  ({cand-best:+.5f})  {verdict}")\n'
         '    if keep:\n'
         '        kept.append(g); best = cand\n'
         'print("\\nkept groups:", kept, "| best OOF macro-F1:", round(best, 5))'),
    md("## Save the chosen feature set (`features_v1`)\nEngineered train+test matrices "
       "(NaN preserved in `eog_burst_index`) + the column list, so downstream notebooks "
       "reload exactly these features."),
    code('Xtr_v1 = add_features(train, kept)\n'
         'Xte_v1 = add_features(test, kept)\n'
         'feature_cols = list(Xtr_v1.columns)\n'
         'np.save(ART / "features_v1_train.npy", Xtr_v1.values.astype("float64"))\n'
         'np.save(ART / "features_v1_test.npy", Xte_v1.values.astype("float64"))\n'
         'json.dump({"groups": kept, "columns": feature_cols},\n'
         '          open(ART / "feature_cols.json", "w"), indent=2)\n'
         'print(f"features_v1: {len(feature_cols)} columns")\n'
         'print(feature_cols)\n'
         '# sanity: no inf, and NaN only in the EOG column\n'
         'assert not np.isinf(Xtr_v1.select_dtypes("number").values).any()\n'
         'na_cols = Xtr_v1.columns[Xtr_v1.isna().any()].tolist()\n'
         'print("columns with NaN (expected just EOG):", na_cols)'),
])

# ===========================================================================
# 04_models.ipynb
# ===========================================================================
nb04 = make_nb([
    md("# 04 — Model comparison on `features_v1`\n"
       "All four models run through the **same** OOF harness and the **same** folds, so their "
       "OOF/test probability matrices are row-aligned and directly blendable. We report OOF "
       "macro-F1 + per-class F1 for each and inspect feature importances."),
    code(TOOLBOX),
    code(LOG_HELPER),
    code('# Load the engineered feature matrices as DataFrames (NaN preserved)\n'
         'cols = json.load(open(ART / "feature_cols.json"))["columns"]\n'
         'Xtr = pd.DataFrame(np.load(ART / "features_v1_train.npy"), columns=cols)\n'
         'Xte = pd.DataFrame(np.load(ART / "features_v1_test.npy"), columns=cols)\n'
         'y = np.load(ART / "y_train.npy")\n'
         'folds = load_folds()\n'
         'print("features_v1:", Xtr.shape[1], "columns")'),
    code('from catboost import CatBoostClassifier\n'
         'from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,\n'
         '                              GradientBoostingClassifier)\n'
         '\n'
         'def make_catboost():\n'
         '    return CatBoostClassifier(loss_function="MultiClass",\n'
         '        eval_metric="TotalF1:average=Macro", iterations=3000, learning_rate=0.03,\n'
         '        depth=6, l2_leaf_reg=3.0, random_seed=SEED, od_type="Iter", od_wait=150,\n'
         '        use_best_model=True, thread_count=-1, allow_writing_files=False, verbose=False)\n'
         'def make_rf():\n'
         '    return RandomForestClassifier(n_estimators=600, max_features="sqrt",\n'
         '        min_samples_leaf=2, class_weight="balanced", n_jobs=-1, random_state=SEED)\n'
         'def make_et():\n'
         '    return ExtraTreesClassifier(n_estimators=800, max_features="sqrt",\n'
         '        min_samples_leaf=1, class_weight="balanced", n_jobs=-1, random_state=SEED)\n'
         'def make_gb():\n'
         '    return GradientBoostingClassifier(n_estimators=400, learning_rate=0.05,\n'
         '        max_depth=3, subsample=0.9, random_state=SEED)\n'
         '\n'
         'specs = [("catboost", make_catboost, False, True),\n'
         '         ("rf", make_rf, True, False),\n'
         '         ("et", make_et, True, False),\n'
         '         ("gb", make_gb, True, False)]\n'
         'results = {}\n'
         'for name, mk, imp, ev in specs:\n'
         '    oof, tst, m, fs = run_oof(mk, Xtr, y, Xte, folds, needs_impute=imp, use_eval_set=ev)\n'
         '    np.save(ART / f"{name}_oof.npy", oof)\n'
         '    np.save(ART / f"{name}_test.npy", tst)\n'
         '    results[name] = m\n'
         '    print(f"{name:<10} OOF macro-F1 = {m:.5f} | per-class {per_class_f1(y, oof.argmax(1))}")\n'
         '    log_result("04_models", name, "features_v1", m, per_class_f1(y, oof.argmax(1)))'),
    md("## Feature importances\nConfirms the engineered features are actually used "
       "(evidence for the defense)."),
    code('# Feature importance: a CatBoost fit on full data WITHOUT early stopping\n'
         '# (use_best_model needs an eval_set; here we use fixed iterations instead).\n'
         'cb_imp = CatBoostClassifier(loss_function="MultiClass", iterations=800,\n'
         '    learning_rate=0.03, depth=6, l2_leaf_reg=3.0, random_seed=SEED,\n'
         '    allow_writing_files=False, thread_count=-1, verbose=False)\n'
         'cb_imp.fit(Xtr, y)\n'
         'imp = cb_imp.get_feature_importance(prettified=True)\n'
         'print("Top 15 CatBoost features:"); print(imp.head(15).to_string(index=False))'),
    md("Summary of OOF macro-F1 by model:"),
    code('print({k: round(v, 5) for k, v in sorted(results.items(), key=lambda x: -x[1])})'),
])

# ===========================================================================
# 05_tune_blend.ipynb
# ===========================================================================
nb05 = make_nb([
    md("# 05 — Tune CatBoost, blend, and search per-class thresholds\n"
       "1. Bounded staged tuning of CatBoost (depth, learning rate, l2, class weights).\n"
       "2. Convex soft-vote blend of the model OOF matrices (weights searched on OOF).\n"
       "3. Coordinate-ascent per-class probability multipliers on the **blended** OOF.\n"
       "Everything is decided on OOF (never the public LB)."),
    code(TOOLBOX),
    code('from catboost import CatBoostClassifier\n'
         'cols = json.load(open(ART / "feature_cols.json"))["columns"]\n'
         'Xtr = pd.DataFrame(np.load(ART / "features_v1_train.npy"), columns=cols)\n'
         'Xte = pd.DataFrame(np.load(ART / "features_v1_test.npy"), columns=cols)\n'
         'y = np.load(ART / "y_train.npy")\n'
         'folds = load_folds()'),
    md("## 1. Bounded staged CatBoost tuning\nWe sweep one hyperparameter at a time around a "
       "sensible default (≈13 fits of 5 folds, not a full grid). Iterations are capped with "
       "early stopping, never hand-tuned. All seeds fixed."),
    code('def make_cat(depth=6, lr=0.03, l2=3.0, balanced=False):\n'
         '    kw = dict(loss_function="MultiClass", eval_metric="TotalF1:average=Macro",\n'
         '        iterations=3000, learning_rate=lr, depth=depth, l2_leaf_reg=l2,\n'
         '        random_seed=SEED, od_type="Iter", od_wait=150, use_best_model=True,\n'
         '        thread_count=-1, allow_writing_files=False, verbose=False)\n'
         '    if balanced:\n'
         '        kw["auto_class_weights"] = "Balanced"\n'
         '    return CatBoostClassifier(**kw)\n'
         '\n'
         'def cat_score(**kw):\n'
         '    _, _, m, _ = run_oof(lambda: make_cat(**kw), Xtr, y, Xte, folds,\n'
         '                         needs_impute=False, use_eval_set=True)\n'
         '    return m\n'
         '\n'
         'best = {"depth": 6, "lr": 0.03, "l2": 3.0, "balanced": False}\n'
         'best_score = cat_score(**best)\n'
         'print(f"start depth=6 lr=0.03 l2=3 -> {best_score:.5f}")\n'
         'for depth in [5, 7, 8]:\n'
         '    s = cat_score(**{**best, "depth": depth})\n'
         '    print(f"  depth={depth}: {s:.5f}")\n'
         '    if s > best_score: best_score, best["depth"] = s, depth\n'
         'for lr in [0.02, 0.05]:\n'
         '    s = cat_score(**{**best, "lr": lr})\n'
         '    print(f"  lr={lr}: {s:.5f}")\n'
         '    if s > best_score: best_score, best["lr"] = s, lr\n'
         'for l2 in [1.0, 5.0, 9.0]:\n'
         '    s = cat_score(**{**best, "l2": l2})\n'
         '    print(f"  l2={l2}: {s:.5f}")\n'
         '    if s > best_score: best_score, best["l2"] = s, l2\n'
         'for bal in [True]:\n'
         '    s = cat_score(**{**best, "balanced": bal})\n'
         '    print(f"  balanced={bal}: {s:.5f}")\n'
         '    if s > best_score: best_score, best["balanced"] = s, bal\n'
         'print("\\nbest config:", best, "-> OOF macro-F1", round(best_score, 5))'),
    code('# Refit the tuned CatBoost across folds; save its OOF + test matrices\n'
         'cat_oof, cat_test, cat_macro, _ = run_oof(lambda: make_cat(**best), Xtr, y, Xte,\n'
         '    folds, needs_impute=False, use_eval_set=True)\n'
         'np.save(ART / "catboost_tuned_oof.npy", cat_oof)\n'
         'np.save(ART / "catboost_tuned_test.npy", cat_test)\n'
         'json.dump(best, open(ART / "catboost_best_params.json", "w"), indent=2)\n'
         'print("tuned CatBoost OOF macro-F1:", round(cat_macro, 5))'),
    md("## 2. Convex blend (weights searched on OOF)\nBlend CatBoost(tuned) + RF + ET + GB. "
       "We search nonnegative weights summing to 1 on a 0.1 grid; weak members get ~0 weight."),
    code('members = ["catboost_tuned", "rf", "et", "gb"]\n'
         'oofs = {m: (cat_oof if m == "catboost_tuned" else np.load(ART / f"{m}_oof.npy")) for m in members}\n'
         'tests = {m: (cat_test if m == "catboost_tuned" else np.load(ART / f"{m}_test.npy")) for m in members}\n'
         '\n'
         'def compositions(total, parts):\n'
         '    if parts == 1:\n'
         '        yield (total,); return\n'
         '    for i in range(total + 1):\n'
         '        for rest in compositions(total - i, parts - 1):\n'
         '            yield (i,) + rest\n'
         '\n'
         'oof_list = [oofs[m] for m in members]\n'
         'STEPS = 10\n'
         'best_w, best_blend = None, -1\n'
         'for comp in compositions(STEPS, len(members)):\n'
         '    w = np.array(comp) / STEPS\n'
         '    blended = sum(wi * o for wi, o in zip(w, oof_list))\n'
         '    s = macro_f1(y, blended.argmax(1))\n'
         '    if s > best_blend:\n'
         '        best_blend, best_w = s, w\n'
         'weights = {m: round(float(w), 2) for m, w in zip(members, best_w)}\n'
         'print("best blend weights:", weights, "-> OOF macro-F1", round(best_blend, 5))\n'
         '\n'
         'blend_oof = sum(best_w[i] * oofs[m] for i, m in enumerate(members))\n'
         'blend_test = sum(best_w[i] * tests[m] for i, m in enumerate(members))\n'
         'np.save(ART / "blend_oof.npy", blend_oof)\n'
         'np.save(ART / "blend_test.npy", blend_test)\n'
         'json.dump({"members": members, "weights": weights, "oof_macro": round(float(best_blend), 5)},\n'
         '          open(ART / "blend.json", "w"), indent=2)'),
    md("## 3. Per-class threshold (probability multiplier) search\nCoordinate ascent on the "
       "blended OOF. Multiplying class probabilities then argmax shifts per-class priors to "
       "lift the weakest class — exactly what macro-F1 rewards. Searched on OOF only."),
    code('def search_thresholds(oof, y, grid, passes=12):\n'
         '    m = np.ones(oof.shape[1])\n'
         '    best = macro_f1(y, oof.argmax(1))\n'
         '    for _ in range(passes):\n'
         '        improved = False\n'
         '        for c in range(len(m)):\n'
         '            keep_g, keep_s = m[c], best\n'
         '            for g in grid:\n'
         '                m[c] = g; s = macro_f1(y, (oof * m).argmax(1))\n'
         '                if s > keep_s + 1e-12: keep_s, keep_g = s, g\n'
         '            m[c] = keep_g\n'
         '            if keep_s > best + 1e-12: best, improved = keep_s, True\n'
         '        if not improved: break\n'
         '    return m, best\n'
         '\n'
         '# single thorough coordinate-ascent pass (fine 0.02 grid over a wide range)\n'
         'mult_final, final = search_thresholds(blend_oof, y, np.round(np.arange(0.4, 1.61, 0.02), 3))\n'
         'print("no-threshold blended OOF macro-F1:", round(macro_f1(y, blend_oof.argmax(1)), 5))\n'
         'print("multipliers:", np.round(mult_final, 3), "-> OOF macro-F1", round(final, 5))\n'
         'print("per-class F1 before:", per_class_f1(y, blend_oof.argmax(1)))\n'
         'print("per-class F1 after :", per_class_f1(y, (blend_oof * mult_final).argmax(1)))\n'
         'json.dump({"multipliers": [round(float(x), 3) for x in mult_final],\n'
         '           "oof_macro": round(float(final), 5)},\n'
         '          open(ART / "thresholds.json", "w"), indent=2)'),
    code('print("confusion BEFORE thresholds:"); print(confusion_matrix(y, blend_oof.argmax(1)))\n'
         'print("confusion AFTER thresholds:");  print(confusion_matrix(y, (blend_oof * mult_final).argmax(1)))'),
])

# ===========================================================================
# 06_submit.ipynb
# ===========================================================================
nb06 = make_nb([
    md("# 06 — Final submissions\n"
       "Two submissions, both selected on OOF macro-F1 (never the public LB). Each file is named "
       "after the model/algorithm that produced it:\n"
       "- **sub01 (primary)** — the best-OOF configuration. The blend search selected CatBoost "
       "alone and the threshold search returned no change, so this is the tuned **CatBoost** "
       "(gradient-boosted decision trees). OOF macro-F1 ≈ 0.829.\n"
       "- **sub02 (diverse safety)** — an equal-weight **soft-voting ensemble** of all four "
       "learners (CatBoost + RandomForest + ExtraTrees + GradientBoosting). Slightly lower OOF, "
       "but a genuinely different prediction — a hedge for the private 70% split."),
    code(TOOLBOX),
    code(LOG_HELPER),
    code('y = np.load(ART / "y_train.npy")\n'
         'test_ids = np.load(ART / "test_ids.npy")\n'
         'blend_oof = np.load(ART / "blend_oof.npy")\n'
         'blend_test = np.load(ART / "blend_test.npy")\n'
         'mult = np.array(json.load(open(ART / "thresholds.json"))["multipliers"])\n'
         'blend_info = json.load(open(ART / "blend.json"))\n'
         'MODELS = ["catboost_tuned", "rf", "et", "gb"]\n'
         'oofs = {m: np.load(ART / f"{m}_oof.npy") for m in MODELS}\n'
         'tests = {m: np.load(ART / f"{m}_test.npy") for m in MODELS}\n'
         'print("blend weights:", blend_info["weights"], "| thresholds:", mult)'),
    md("## sub01 — best-OOF model (named for the algorithm actually selected)\n"
       "The filename is built from what the pipeline actually chose, so it never lies: the blend "
       "members with non-zero weight, plus `_ThresholdTuned` only if the threshold search changed "
       "anything."),
    code('ALGO = {"catboost_tuned": "CatBoost", "rf": "RandomForest",\n'
         '        "et": "ExtraTrees", "gb": "GradientBoosting"}\n'
         'used = [m for m in blend_info["members"] if blend_info["weights"][m] > 0]\n'
         'if len(used) == 1:\n'
         '    tag = ALGO[used[0]] + ("_GBDT" if used[0] == "catboost_tuned" else "")\n'
         'else:\n'
         '    tag = "SoftVotingEnsemble-" + "-".join(ALGO[m] for m in used)\n'
         'tag += "_ThresholdTuned" if not np.allclose(mult, 1.0) else ""\n'
         'name1 = f"sub01_{tag}.csv"\n'
         'pred1 = (blend_test * mult).argmax(1).astype(int)\n'
         'sub1 = pd.DataFrame({"id": test_ids, "sleep_stage": pred1})\n'
         'assert sub1.shape == (5000, 2)\n'
         'assert sub1["id"].tolist() == list(range(9000, 14000))\n'
         'assert sub1["sleep_stage"].isin(CLASSES).all()\n'
         'sub1.to_csv(SUB / name1, index=False)\n'
         'oof1 = macro_f1(y, (blend_oof * mult).argmax(1))\n'
         'print(f"sub01 -> {name1} | OOF macro-F1:", round(oof1, 5))\n'
         'print("class counts:", sub1["sleep_stage"].value_counts().sort_index().to_dict())\n'
         'log_result("06_submit", tag, "+".join(used), oof1,\n'
         '           per_class_f1(y, (blend_oof * mult).argmax(1)), "sub01 (primary): " + name1)'),
    md("## sub02 — soft-voting ensemble (diverse safety net)\nEqual-weight average of the four "
       "models' probabilities — a soft `VotingClassifier`. Genuinely different from CatBoost alone."),
    code('ens_oof = sum(oofs[m] for m in MODELS) / len(MODELS)\n'
         'ens_test = sum(tests[m] for m in MODELS) / len(MODELS)\n'
         'name2 = "sub02_SoftVotingEnsemble_CatBoost-RandomForest-ExtraTrees-GradientBoosting.csv"\n'
         'pred2 = ens_test.argmax(1).astype(int)\n'
         'sub2 = pd.DataFrame({"id": test_ids, "sleep_stage": pred2})\n'
         'assert sub2.shape == (5000, 2) and sub2["sleep_stage"].isin(CLASSES).all()\n'
         'sub2.to_csv(SUB / name2, index=False)\n'
         'oof2 = macro_f1(y, ens_oof.argmax(1))\n'
         'print(f"sub02 -> {name2} | OOF macro-F1:", round(oof2, 5))\n'
         'print("class counts:", sub2["sleep_stage"].value_counts().sort_index().to_dict())\n'
         'print("sub01 vs sub02 differ on",\n'
         '      int((sub1["sleep_stage"] != sub2["sleep_stage"]).sum()), "of 5000 rows")\n'
         'log_result("06_submit", "SoftVotingEnsemble", "+".join(MODELS), oof2,\n'
         '           per_class_f1(y, ens_oof.argmax(1)), "sub02 (diverse safety): " + name2)'),
    md("## Done\nUpload limit is 20/day. **Primary = `sub01_CatBoost_GBDT.csv`** (best OOF ≈ 0.829); "
       "**safety = `sub02_SoftVotingEnsemble_*.csv`** (diverse hedge). Both chosen by OOF macro-F1, "
       "not the public LB. See `results_log.csv` for the full evidence table."),
])

# ===========================================================================
# 07_robustness.ipynb  (private-LB hardening)
# ===========================================================================
# Two robustness steps the OOF≈public-LB result motivates (gains are tiny near
# this synthetic ceiling, so the real win is a TRUSTWORTHY metric + a
# LOWER-VARIANCE final model):
#   1. Repeated CV gate: RepeatedStratifiedKFold(5x5) -> mean +/- std of OOF
#      macro-F1. The std is the noise floor; only accept a future change whose
#      OOF gain clearly exceeds it. Pure risk-reducer (cannot leak/overfit).
#   2. Seed-bagged CatBoost: average predict_proba over K=9 seeds of the exact
#      deployed config. Diagnostic first (per-seed OOF spread); then the bag.
#      Variance reduction (textbook bagging) -> private-LB stability. Written as
#      sub03; the single-seed CatBoost stays the documented reference (sub01).
nb07 = make_nb([
    md("# 07 — Robustness: repeated-CV gate + seed-bagged CatBoost\n"
       "Public LB (0.83026) ≈ OOF (0.82898): the model is **not** overfit, but headroom is tiny "
       "and the error structure shows the weak Deep class is diffuse, confident overlap (partly "
       "irreducible synthetic noise). So the highest-value moves left are not a cleverer model but "
       "**(1)** a de-noised metric we can trust for accept/reject decisions, and **(2)** a "
       "lower-variance final model for the private 70% split.\n\n"
       "- **Repeated CV gate** — `RepeatedStratifiedKFold(5×5)` → mean ± std OOF macro-F1. The std "
       "is the noise floor: only keep a future change if its OOF gain clearly beats it.\n"
       "- **Seed-bagged CatBoost** — average `predict_proba` over K=9 seeds of the *exact* deployed "
       "config (textbook bagging from the course). Pure variance reduction; cannot widen the CV–LB gap."),
    code(TOOLBOX),
    code(LOG_HELPER),
    code('from catboost import CatBoostClassifier\n'
         'cols = json.load(open(ART / "feature_cols.json"))["columns"]\n'
         'Xtr = pd.DataFrame(np.load(ART / "features_v1_train.npy"), columns=cols)\n'
         'Xte = pd.DataFrame(np.load(ART / "features_v1_test.npy"), columns=cols)\n'
         'y = np.load(ART / "y_train.npy")\n'
         'test_ids = np.load(ART / "test_ids.npy")\n'
         'folds = load_folds()\n'
         '# Use the EXACT config sub01 deployed (tuning kept the defaults).\n'
         'bp = json.load(open(ART / "catboost_best_params.json"))\n'
         'print("deployed CatBoost config:", bp, "| features_v1:", Xtr.shape[1], "cols")\n'
         '\n'
         'def make_cat(seed=SEED):\n'
         '    kw = dict(loss_function="MultiClass", eval_metric="TotalF1:average=Macro",\n'
         '        iterations=3000, learning_rate=bp["lr"], depth=bp["depth"], l2_leaf_reg=bp["l2"],\n'
         '        random_seed=seed, od_type="Iter", od_wait=150, use_best_model=True,\n'
         '        thread_count=-1, allow_writing_files=False, verbose=False)\n'
         '    if bp.get("balanced"):\n'
         '        kw["auto_class_weights"] = "Balanced"\n'
         '    return CatBoostClassifier(**kw)'),
    md("## 1. Repeated-CV gate (`RepeatedStratifiedKFold(5×5)`)\n"
       "Single 5-fold OOF noise is on the same order as the gains we chase (thousandths), so "
       "selecting on one split risks picking a lucky-split model that regresses on the private LB. "
       "We run 5 independent 5-fold splits (25 fits) and report the mean and **std** of the global "
       "OOF macro-F1. That std is the accept/reject threshold for every later experiment."),
    code('from sklearn.model_selection import RepeatedStratifiedKFold\n'
         'N_REPEATS = 5\n'
         'rskf = RepeatedStratifiedKFold(n_splits=N_FOLDS, n_repeats=N_REPEATS, random_state=SEED)\n'
         'splits = list(rskf.split(np.zeros(len(y)), y))\n'
         'repeat_macro, repeat_pcf = [], []\n'
         'for r in range(N_REPEATS):\n'
         '    oof = np.zeros((len(y), len(CLASSES)))\n'
         '    for tr_idx, va_idx in splits[r*N_FOLDS:(r+1)*N_FOLDS]:\n'
         '        m = make_cat()\n'
         '        m.fit(Xtr.iloc[tr_idx], y[tr_idx], eval_set=(Xtr.iloc[va_idx], y[va_idx]))\n'
         '        oof[va_idx] = _aligned_proba(m, Xtr.iloc[va_idx])\n'
         '    s = macro_f1(y, oof.argmax(1))\n'
         '    repeat_macro.append(s); repeat_pcf.append(per_class_f1(y, oof.argmax(1)))\n'
         '    print(f"  repeat {r+1}: OOF macro-F1 = {s:.5f}")\n'
         'gate_mean, gate_std = float(np.mean(repeat_macro)), float(np.std(repeat_macro))\n'
         'pc_mean = {c: round(float(np.mean([d[c] for d in repeat_pcf])), 4)\n'
         '           for c in ["Wake", "Light", "Deep", "REM"]}\n'
         'print(f"\\nGATE: OOF macro-F1 = {gate_mean:.5f} +/- {gate_std:.5f}  "\n'
         '      f"(over {N_REPEATS} repeats x {N_FOLDS} folds)")\n'
         'print(f"=> accept a future change only if its OOF gain clearly exceeds ~{gate_std:.4f} (1 sigma).")\n'
         'print("mean per-class F1:", pc_mean)\n'
         'json.dump({"repeat_macro": [round(x, 5) for x in repeat_macro], "mean": round(gate_mean, 5),\n'
         '           "std": round(gate_std, 5), "n_repeats": N_REPEATS, "n_folds": N_FOLDS},\n'
         '          open(ART / "repeated_cv_gate.json", "w"), indent=2)\n'
         'log_result("07_robustness", "catboost_repeatedCV", "features_v1", gate_mean, pc_mean,\n'
         '           f"GATE mean+/-std over {N_REPEATS}x{N_FOLDS}; std={gate_std:.5f}")'),
    md("## 2. Seed-bagged CatBoost\n"
       "**Diagnostic first:** train K=9 seeds and look at the *single-model* per-seed OOF spread. "
       "If the spread is tiny the averaging ceiling is tiny too — we then bag for **stability**, not "
       "for score. The bag averages `predict_proba` across all seeds (OOF) and across folds×seeds "
       "(test). All seeds are fixed and logged, so the result is fully reproducible."),
    code('SEEDS = [42, 7, 13, 101, 202, 303, 404, 505, 2024]   # K=9, fixed & logged\n'
         'K = len(SEEDS)\n'
         'seed_oofs = {s: np.zeros((len(y), len(CLASSES))) for s in SEEDS}\n'
         'sb_test = np.zeros((len(Xte), len(CLASSES)))\n'
         'for tr_idx, va_idx in folds:\n'
         '    Xt, Xv = Xtr.iloc[tr_idx], Xtr.iloc[va_idx]\n'
         '    yt, yv = y[tr_idx], y[va_idx]\n'
         '    for s in SEEDS:\n'
         '        m = make_cat(seed=s)\n'
         '        m.fit(Xt, yt, eval_set=(Xv, yv))\n'
         '        seed_oofs[s][va_idx] = _aligned_proba(m, Xv)\n'
         '        sb_test += _aligned_proba(m, Xte) / (len(folds) * K)\n'
         '\n'
         '# diagnostic: how much does the seed alone move the OOF score?\n'
         'per_seed = {s: macro_f1(y, seed_oofs[s].argmax(1)) for s in SEEDS}\n'
         'vals = np.array(list(per_seed.values()))\n'
         'print("per-seed single-model OOF macro-F1:")\n'
         'for s in SEEDS:\n'
         '    print(f"   seed {s:>4}: {per_seed[s]:.5f}")\n'
         'print(f"spread -> min {vals.min():.5f}  max {vals.max():.5f}  std {vals.std():.5f}")\n'
         '\n'
         '# the bag\n'
         'sb_oof = sum(seed_oofs.values()) / K\n'
         'sb_macro = macro_f1(y, sb_oof.argmax(1))\n'
         'single = per_seed[SEED]\n'
         'print(f"\\nsingle-seed (42): {single:.5f}")\n'
         'print(f"seed-bag (K={K}):  {sb_macro:.5f}   ({sb_macro - single:+.5f} vs single seed)")\n'
         'print("seed-bag per-class F1:", per_class_f1(y, sb_oof.argmax(1)))\n'
         'np.save(ART / "seedbag_oof.npy", sb_oof)\n'
         'np.save(ART / "seedbag_test.npy", sb_test)\n'
         'json.dump({"seeds": SEEDS, "per_seed_macro": {str(s): round(per_seed[s], 5) for s in SEEDS},\n'
         '           "seed_std": round(float(vals.std()), 5), "single_seed42": round(single, 5),\n'
         '           "bag_macro": round(float(sb_macro), 5)},\n'
         '          open(ART / "seedbag.json", "w"), indent=2)\n'
         'log_result("07_robustness", f"catboost_seedbag_K{K}", "features_v1", sb_macro,\n'
         '           per_class_f1(y, sb_oof.argmax(1)),\n'
         '           f"K={K} seeds; single={single:.5f}; seed_std={vals.std():.5f}")'),
    md("## Seed-bag submission (`sub03`)\nShipped as a stability-hardened candidate **regardless** of "
       "OOF lift — variance reduction is private-LB insurance, not a score chase. The deterministic "
       "single-seed CatBoost (`sub01`) remains the documented reference."),
    code('name3 = f"sub03_CatBoostSeedBagK{K}_GBDT.csv"\n'
         'pred3 = sb_test.argmax(1).astype(int)\n'
         'sub3 = pd.DataFrame({"id": test_ids, "sleep_stage": pred3})\n'
         'assert sub3.shape == (5000, 2)\n'
         'assert sub3["id"].tolist() == list(range(9000, 14000))\n'
         'assert sub3["sleep_stage"].isin(CLASSES).all()\n'
         'sub3.to_csv(SUB / name3, index=False)\n'
         'print("wrote", SUB / name3, "| OOF macro-F1:", round(sb_macro, 5))\n'
         'print("class counts:", sub3["sleep_stage"].value_counts().sort_index().to_dict())\n'
         '# divergence from the single-CatBoost primary (sub01)\n'
         'single_test = np.load(ART / "catboost_tuned_test.npy")\n'
         'diff = int((sb_test.argmax(1) != single_test.argmax(1)).sum())\n'
         'print(f"seed-bag vs single-CatBoost test predictions differ on {diff} of 5000 rows")'),
    md("### Takeaways\n"
       "- The repeated-CV **gate** (mean ± std) is now the accept/reject rule for every future "
       "experiment — no more chasing sub-noise gains on a single split.\n"
       "- The **seed-bag** trades a few minutes of compute for lower private-LB variance with no "
       "overfit risk; if the per-seed spread is < ~0.002 the OOF number barely moves, which is the "
       "expected (and honest) outcome near this ceiling.\n"
       "- Submissions to keep as the private-LB pair: `sub01` (deterministic single CatBoost, the "
       "validated reference) and `sub03` (seed-bagged, variance-reduced)."),
])

# ===========================================================================
# 08_catboost_ensemble.ipynb  (CatBoost-only diversity ensemble)
# ===========================================================================
# The earlier soft-vote of CatBoost+RF+ET+GB lost because RF/ET/GB are weak
# (0.79-0.81). Here every member is a STRONG CatBoost (~0.82-0.83) made diverse
# structurally: depth in {5,6,7,8}, one MultiClassOneVsAll head, one rsm=0.7
# column-subsampled model. Combine by SIMPLE equal-weight probability averaging
# (no OOF weight search -- it collapsed to CatBoost=1.0 before and overfits the
# 0.0017 noise floor). Judge the result against the repeated-CV gate from 07.
nb08 = make_nb([
    md("# 08 — CatBoost-only diversity ensemble\n"
       "The 04/06 soft-vote failed because RF/ET/GB are individually weak (0.79–0.81) and dragged "
       "the average down. A useful ensemble needs members that are **both strong and decorrelated**. "
       "Here every member is a strong CatBoost (~0.82–0.83), made diverse *structurally*:\n"
       "- depth ∈ {5, 6, 7, 8} (different bias/variance trade-offs),\n"
       "- one `MultiClassOneVsAll` head (decouples the per-class gradient from the softmax),\n"
       "- one `rsm=0.7` column-subsampled model (random-subspace decorrelation).\n\n"
       "Combine by **equal-weight** probability averaging — *no* OOF weight search (it collapsed to "
       "CatBoost=1.0 before and would just overfit the ±0.0017 noise floor measured in 07). We judge "
       "the ensemble strictly against the repeated-CV **gate**: it is a real gain only if it beats the "
       "single deployed model on the same folds by more than ~1σ."),
    code(TOOLBOX),
    code(LOG_HELPER),
    code('from catboost import CatBoostClassifier\n'
         'import itertools\n'
         'cols = json.load(open(ART / "feature_cols.json"))["columns"]\n'
         'Xtr = pd.DataFrame(np.load(ART / "features_v1_train.npy"), columns=cols)\n'
         'Xte = pd.DataFrame(np.load(ART / "features_v1_test.npy"), columns=cols)\n'
         'y = np.load(ART / "y_train.npy")\n'
         'test_ids = np.load(ART / "test_ids.npy")\n'
         'folds = load_folds()\n'
         'gate = json.load(open(ART / "repeated_cv_gate.json"))\n'
         'print("features_v1:", Xtr.shape[1], "cols | gate:", gate["mean"], "+/-", gate["std"])\n'
         '\n'
         'def norm_rows(p):\n'
         '    """Row-normalize to sum 1 (MultiClassOneVsAll probs need it; no-op for softmax)."""\n'
         '    s = p.sum(1, keepdims=True)\n'
         '    return p / np.where(s == 0, 1.0, s)\n'
         '\n'
         'def make_member(depth=6, loss="MultiClass", rsm=None, seed=SEED):\n'
         '    kw = dict(loss_function=loss, eval_metric="TotalF1:average=Macro",\n'
         '        iterations=3000, learning_rate=0.03, depth=depth, l2_leaf_reg=3.0,\n'
         '        random_seed=seed, od_type="Iter", od_wait=150, use_best_model=True,\n'
         '        thread_count=-1, allow_writing_files=False, verbose=False)\n'
         '    if rsm is not None:\n'
         '        kw["rsm"] = rsm\n'
         '    return CatBoostClassifier(**kw)'),
    md("## Train the diverse members (same folds, OOF harness)\nEach member runs through the shared "
       "`run_oof` on the saved folds, so all OOF/test matrices are row-aligned and averageable. The "
       "`cat_d6` member is the exact deployed config — our anchor for the gate comparison."),
    code('MEMBERS = [\n'
         '    ("cat_d6",  dict(depth=6, loss="MultiClass")),                 # anchor (= sub01 config)\n'
         '    ("cat_d5",  dict(depth=5, loss="MultiClass")),\n'
         '    ("cat_d7",  dict(depth=7, loss="MultiClass")),\n'
         '    ("cat_d8",  dict(depth=8, loss="MultiClass")),\n'
         '    ("cat_ova", dict(depth=6, loss="MultiClassOneVsAll")),         # one-vs-all head\n'
         '    ("cat_rsm", dict(depth=6, loss="MultiClass", rsm=0.7, seed=123)),  # column subsampling\n'
         ']\n'
         'm_oof, m_test, m_macro = {}, {}, {}\n'
         'for name, cfg in MEMBERS:\n'
         '    oof, tst, mac, _ = run_oof(lambda c=cfg: make_member(**c), Xtr, y, Xte, folds,\n'
         '                               needs_impute=False, use_eval_set=True)\n'
         '    m_oof[name], m_test[name] = norm_rows(oof), norm_rows(tst)\n'
         '    m_macro[name] = mac\n'
         '    print(f"  {name:<8} {str(cfg):<55} OOF macro-F1 = {mac:.5f}")'),
    md("## Equal-weight ensemble + diversity check\nSimple mean of the member probabilities. We also "
       "report mean pairwise prediction **disagreement** — the ensemble can only help to the extent "
       "members actually disagree."),
    code('names = [n for n, _ in MEMBERS]\n'
         'ens_oof = sum(m_oof[n] for n in names) / len(names)\n'
         'ens_test = sum(m_test[n] for n in names) / len(names)\n'
         'ens_macro = macro_f1(y, ens_oof.argmax(1))\n'
         '\n'
         'preds = {n: m_oof[n].argmax(1) for n in names}\n'
         'dis = [float((preds[a] != preds[b]).mean()) for a, b in itertools.combinations(names, 2)]\n'
         'anchor = m_macro["cat_d6"]                     # single deployed model, same folds\n'
         'print("member OOF macro-F1:", {n: round(m_macro[n], 5) for n in names})\n'
         'print("mean pairwise disagreement:", round(float(np.mean(dis)), 4))\n'
         'print(f"\\nENSEMBLE OOF macro-F1 = {ens_macro:.5f}")\n'
         'print(f"anchor (single cat_d6) = {anchor:.5f}   delta = {ens_macro - anchor:+.5f}")\n'
         'print(f"gate noise floor (1 sigma) = {gate[\'std\']:.5f}")\n'
         'real = ens_macro > anchor + gate["std"]\n'
         'print("=> real gain beyond noise?", bool(real),\n'
         '      "(need >", round(anchor + gate["std"], 5), ")")\n'
         'print("ensemble per-class F1:", per_class_f1(y, ens_oof.argmax(1)))\n'
         'np.save(ART / "cat_ensemble_oof.npy", ens_oof)\n'
         'np.save(ART / "cat_ensemble_test.npy", ens_test)\n'
         'json.dump({"members": names, "member_macro": {n: round(m_macro[n], 5) for n in names},\n'
         '           "ensemble_macro": round(float(ens_macro), 5), "anchor": round(float(anchor), 5),\n'
         '           "mean_disagreement": round(float(np.mean(dis)), 4),\n'
         '           "beats_gate": bool(real)}, open(ART / "cat_ensemble.json", "w"), indent=2)\n'
         'log_result("08_ensemble", "catboost_diversity_ensemble", "features_v1", ens_macro,\n'
         '           per_class_f1(y, ens_oof.argmax(1)),\n'
         '           f"equal-wt {len(names)} CatBoosts; vs anchor {anchor:.5f} ({ens_macro-anchor:+.5f}); '
         'disag={np.mean(dis):.3f}")'),
    md("## Submission (`sub04`) — diverse CatBoost-family hedge\nWritten as a candidate private-LB "
       "safety submission (single algorithm family, clean to defend). Whether it becomes a *final* "
       "pick depends on the gate verdict above — if it does not beat the anchor by >1σ it is a "
       "diversity hedge, not an improvement."),
    code('name4 = "sub04_CatBoostDiversityEnsemble_GBDT.csv"\n'
         'pred4 = ens_test.argmax(1).astype(int)\n'
         'sub4 = pd.DataFrame({"id": test_ids, "sleep_stage": pred4})\n'
         'assert sub4.shape == (5000, 2)\n'
         'assert sub4["id"].tolist() == list(range(9000, 14000))\n'
         'assert sub4["sleep_stage"].isin(CLASSES).all()\n'
         'sub4.to_csv(SUB / name4, index=False)\n'
         'print("wrote", SUB / name4, "| OOF macro-F1:", round(ens_macro, 5))\n'
         'print("class counts:", sub4["sleep_stage"].value_counts().sort_index().to_dict())\n'
         'single_test = np.load(ART / "catboost_tuned_test.npy")\n'
         'diff = int((ens_test.argmax(1) != single_test.argmax(1)).sum())\n'
         'print(f"ensemble vs single-CatBoost (sub01) test preds differ on {diff} of 5000 rows")'),
    md("### Takeaways\n"
       "- Members are all CatBoost on the same features, so they stay fairly correlated; equal-weight "
       "averaging smooths variance but, near this synthetic ceiling, is expected to land **within the "
       "±0.0017 gate noise** of the single model — an honest result, not a failure.\n"
       "- Kept as a single-algorithm-family **diversity hedge** (`sub04`). The single CatBoost (`sub01`) "
       "and the seed-bag (`sub03`) remain the primary private-LB pair unless `sub04` beats the gate."),
])

# ===========================================================================
# 09_mlp_ensemble.ipynb  (cross-FAMILY ensemble: CatBoost + MLP neural net)
# ===========================================================================
# Everything before this notebook used only tree models (CatBoost/RF/ET/GB),
# which share an axis-aligned inductive bias -- that is exactly why their
# soft-vote (sub02) and the CatBoost-diversity ensemble (sub04) could not clear
# the ~0.829 wall. A non-tree learner breaks that bias. A standardized MLP
# (neural net) individually rivals/beats CatBoost AND makes *different* errors
# (error-correlation ~0.65 vs ~0.9 tree-vs-tree). The honest payoff is the
# CROSS-FAMILY blend. Weights are FIXED 0.5/0.5 a-priori: an OOF weight search
# inflated the score by ~0.006 of noise that did not transfer (verified by an
# adversarial paired re-test), so we do not tune weights. The headline number is
# a PAIRED RepeatedStratifiedKFold(5x5) comparison vs single CatBoost.
NB09_LOAD = r'''from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RepeatedStratifiedKFold
from catboost import CatBoostClassifier

cols = json.load(open(ART / "feature_cols.json"))["columns"]
Xtr = pd.DataFrame(np.load(ART / "features_v1_train.npy"), columns=cols)
Xte = pd.DataFrame(np.load(ART / "features_v1_test.npy"), columns=cols)
y = np.load(ART / "y_train.npy")
test_ids = np.load(ART / "test_ids.npy")
folds = load_folds()
gate = json.load(open(ART / "repeated_cv_gate.json"))
bp = json.load(open(ART / "catboost_best_params.json"))

# Anchor = the deployed single CatBoost (sub01): reuse its saved OOF/test probs.
cat_oof = np.load(ART / "catboost_tuned_oof.npy")
cat_test = np.load(ART / "catboost_tuned_test.npy")
anchor = macro_f1(y, cat_oof.argmax(1))
print("features_v1:", Xtr.shape[1], "cols | anchor CatBoost OOF macro-F1:", round(anchor, 5))
print("gate:", gate["mean"], "+/-", gate["std"], "| deployed CatBoost config:", bp)'''

NB09_MLP = r'''# MLP under the SAME folds. Non-tree models need scaling and cannot take NaN, so
# per fold we fit a StandardScaler and the EOG median on the TRAIN fold only
# (no leakage), then bag over a few seeds (an MLP is higher-variance than CatBoost).
MLP_SEEDS = [42, 7, 13]

def make_mlp(seed):
    return MLPClassifier(hidden_layer_sizes=(256, 128), alpha=1e-3,
        activation="relu", solver="adam", batch_size=256, max_iter=400,
        early_stopping=True, n_iter_no_change=20, validation_fraction=0.1,
        random_state=seed)

def mlp_oof(Xdf, ydat, Xtest_df, the_folds, seeds):
    """Seed-bagged MLP OOF + test probabilities, scaler+impute fit per train fold."""
    ncl = len(CLASSES)
    oof = np.zeros((len(ydat), ncl)); test_p = np.zeros((len(Xtest_df), ncl))
    for tr_idx, va_idx in the_folds:
        Xt, Xv, Xe = Xdf.iloc[tr_idx].copy(), Xdf.iloc[va_idx].copy(), Xtest_df.copy()
        med = Xt[EOG].median()
        for d in (Xt, Xv, Xe):
            d[EOG] = d[EOG].fillna(med)
        sc = StandardScaler().fit(Xt.values)
        Xts, Xvs, Xes = sc.transform(Xt.values), sc.transform(Xv.values), sc.transform(Xe.values)
        for s in seeds:
            m = make_mlp(s); m.fit(Xts, ydat[tr_idx])
            oof[va_idx] += _aligned_proba(m, Xvs) / len(seeds)
            test_p += _aligned_proba(m, Xes) / (len(the_folds) * len(seeds))
    return oof, test_p

mlp_oof_p, mlp_test_p = mlp_oof(Xtr, y, Xte, folds, MLP_SEEDS)
mlp_macro = macro_f1(y, mlp_oof_p.argmax(1))
cat_wrong = (cat_oof.argmax(1) != y).astype(int)
mlp_wrong = (mlp_oof_p.argmax(1) != y).astype(int)
errcorr = float(np.corrcoef(cat_wrong, mlp_wrong)[0, 1])
print(f"MLP seed-bag (K={len(MLP_SEEDS)}) OOF macro-F1 = {mlp_macro:.5f}")
print("MLP per-class F1:", per_class_f1(y, mlp_oof_p.argmax(1)))
print(f"error-correlation MLP vs CatBoost = {errcorr:.3f}  (lower = more diverse; tree-vs-tree ~0.9)")
np.save(ART / "mlp_oof.npy", mlp_oof_p); np.save(ART / "mlp_test.npy", mlp_test_p)
log_result("09_mlp_ensemble", "mlp_seedbag", "features_v1", mlp_macro,
           per_class_f1(y, mlp_oof_p.argmax(1)),
           f"MLP(256,128) a=1e-3 K={len(MLP_SEEDS)}; errcorr_vs_cat={errcorr:.3f}")'''

NB09_ENS = r'''# Fixed equal weights chosen A-PRIORI -- NO OOF weight search. The audit showed
# tuning weights on the full OOF inflated the score by ~0.006 of noise that did
# not transfer out-of-fold, so equal weights are the honest, defensible choice.
W_CAT, W_MLP = 0.5, 0.5
ens_oof = W_CAT * cat_oof + W_MLP * mlp_oof_p
ens_test = W_CAT * cat_test + W_MLP * mlp_test_p
ens_macro = macro_f1(y, ens_oof.argmax(1))
beats = ens_macro > anchor + gate["std"]
print(f"anchor single CatBoost     = {anchor:.5f}")
print(f"MLP seed-bag               = {mlp_macro:.5f}")
print(f"ENSEMBLE 0.5*Cat + 0.5*MLP = {ens_macro:.5f}   (delta vs anchor {ens_macro - anchor:+.5f})")
print(f"gate 1-sigma = {gate['std']:.5f} -> clears anchor+1sigma on the seed-42 split?", bool(beats))
print("ensemble per-class F1:", per_class_f1(y, ens_oof.argmax(1)))
print("(on the seed-42 split CatBoost is itself ~+0.003 lucky, so this single-split delta")
print(" UNDERSTATES the true gain; the paired repeated-CV test below is the honest headline.)")
np.save(ART / "mlp_ensemble_oof.npy", ens_oof); np.save(ART / "mlp_ensemble_test.npy", ens_test)
json.dump({"weights": {"catboost": W_CAT, "mlp": W_MLP}, "mlp_seeds": MLP_SEEDS,
           "mlp_macro": round(float(mlp_macro), 5), "anchor": round(float(anchor), 5),
           "ensemble_macro": round(float(ens_macro), 5), "errcorr_vs_cat": round(errcorr, 3),
           "beats_gate_seed42": bool(beats)}, open(ART / "mlp_ensemble.json", "w"), indent=2)
log_result("09_mlp_ensemble", "catboost_plus_mlp", "features_v1", ens_macro,
           per_class_f1(y, ens_oof.argmax(1)),
           f"fixed 0.5/0.5 Cat+MLP; vs anchor {anchor:.5f} ({ens_macro - anchor:+.5f}); errcorr={errcorr:.3f}")'''

NB09_PAIRED = r'''# Honest headline: a PAIRED comparison on fresh folds. On any single split
# CatBoost's fold luck can hide the gain, so we average over
# RepeatedStratifiedKFold(5x5, seed=2026) and compare the ensemble against single
# CatBoost on IDENTICAL folds. Both arms are leak-free: CatBoost uses FIXED
# iterations (no early-stopping peek at the valid fold); the MLP fits its
# scaler+impute strictly inside the train fold. Weights stay fixed 0.5/0.5.
N_REP = 5
PAIR_SEEDS = [42, 7]                         # 2-seed MLP bag inside the loop (cost control)
CAT_ITERS = 900                              # ~ deployed best_iter (~921); fixed => leak-free
rskf = RepeatedStratifiedKFold(n_splits=N_FOLDS, n_repeats=N_REP, random_state=2026)
splits = list(rskf.split(np.zeros(len(y)), y))

def make_cat_fixed(seed=SEED):
    return CatBoostClassifier(loss_function="MultiClass", eval_metric="TotalF1:average=Macro",
        iterations=CAT_ITERS, learning_rate=bp["lr"], depth=bp["depth"], l2_leaf_reg=bp["l2"],
        random_seed=seed, allow_writing_files=False, thread_count=-1, verbose=False)

cat_sc, ens_sc = [], []
for r in range(N_REP):
    cat_o = np.zeros((len(y), len(CLASSES)))
    mlp_o = np.zeros((len(y), len(CLASSES)))
    for tr_idx, va_idx in splits[r * N_FOLDS:(r + 1) * N_FOLDS]:
        cb = make_cat_fixed(); cb.fit(Xtr.iloc[tr_idx], y[tr_idx])
        cat_o[va_idx] = _aligned_proba(cb, Xtr.iloc[va_idx])
        Xt, Xv = Xtr.iloc[tr_idx].copy(), Xtr.iloc[va_idx].copy()
        med = Xt[EOG].median(); Xt[EOG] = Xt[EOG].fillna(med); Xv[EOG] = Xv[EOG].fillna(med)
        sc = StandardScaler().fit(Xt.values)
        Xts, Xvs = sc.transform(Xt.values), sc.transform(Xv.values)
        for s in PAIR_SEEDS:
            m = make_mlp(s); m.fit(Xts, y[tr_idx])
            mlp_o[va_idx] += _aligned_proba(m, Xvs) / len(PAIR_SEEDS)
    cs = macro_f1(y, cat_o.argmax(1))
    es = macro_f1(y, (0.5 * cat_o + 0.5 * mlp_o).argmax(1))
    cat_sc.append(cs); ens_sc.append(es)
    print(f"  repeat {r + 1}: CatBoost {cs:.5f} | Ensemble {es:.5f} | delta {es - cs:+.5f}")

cat_sc, ens_sc = np.array(cat_sc), np.array(ens_sc)
d = ens_sc - cat_sc
print(f"\nPAIRED repeated CV ({N_REP}x{N_FOLDS}, seed=2026):")
print(f"  single CatBoost : {cat_sc.mean():.5f} +/- {cat_sc.std():.5f}")
print(f"  Cat+MLP ensemble: {ens_sc.mean():.5f} +/- {ens_sc.std():.5f}")
print(f"  paired delta    : {d.mean():+.5f} +/- {d.std():.5f}  (positive in {int((d > 0).sum())}/{N_REP} repeats)")
print(f"  => honest gain beyond the {gate['std']:.5f} noise floor?", bool(d.mean() > gate["std"]))
json.dump({"n_repeats": N_REP, "n_folds": N_FOLDS, "cat_iters": CAT_ITERS,
           "pair_mlp_seeds": PAIR_SEEDS, "cat_mean": round(float(cat_sc.mean()), 5),
           "cat_std": round(float(cat_sc.std()), 5), "ens_mean": round(float(ens_sc.mean()), 5),
           "ens_std": round(float(ens_sc.std()), 5), "delta_mean": round(float(d.mean()), 5),
           "delta_std": round(float(d.std()), 5), "pos_repeats": int((d > 0).sum())},
          open(ART / "mlp_ensemble_paired_cv.json", "w"), indent=2)
log_result("09_mlp_ensemble", "catboost_plus_mlp_pairedCV", "features_v1", float(ens_sc.mean()),
           per_class_f1(y, ens_oof.argmax(1)),
           f"paired {N_REP}x{N_FOLDS}: ens {ens_sc.mean():.5f} vs cat {cat_sc.mean():.5f} (delta {d.mean():+.5f})")'''

NB09_SUB = r'''name5 = "sub05_CatBoostPlusMLP_Ensemble.csv"
pred5 = ens_test.argmax(1).astype(int)
sub5 = pd.DataFrame({"id": test_ids, "sleep_stage": pred5})
assert sub5.shape == (5000, 2)
assert sub5["id"].tolist() == list(range(9000, 14000))
assert sub5["sleep_stage"].isin(CLASSES).all()
sub5.to_csv(SUB / name5, index=False)
print("wrote", SUB / name5, "| ensemble OOF macro-F1:", round(ens_macro, 5))
print("class counts:", sub5["sleep_stage"].value_counts().sort_index().to_dict())
single_test = np.load(ART / "catboost_tuned_test.npy")
diff = int((ens_test.argmax(1) != single_test.argmax(1)).sum())
print(f"sub05 vs single-CatBoost (sub01) test preds differ on {diff} of 5000 rows")'''

nb09 = make_nb([
    md("# 09 — Cross-family ensemble: CatBoost + MLP (neural net)\n"
       "Every earlier model is a **tree** (CatBoost, RF, ExtraTrees, GB). They share an "
       "axis-aligned inductive bias, so their ensembles (`sub02` soft-vote, `sub04` CatBoost-"
       "diversity) stayed stuck at the ~0.829 wall — the members make the *same* mistakes. "
       "A neural net breaks that bias: a standardized **MLP** rivals/beats CatBoost on its own "
       "and, crucially, makes **different** errors. The real win is the **cross-family blend**.\n\n"
       "Honesty guards (all motivated by an adversarial audit of this idea):\n"
       "- **Fixed 0.5/0.5 weights**, chosen a-priori. Searching weights on the OOF inflated the "
       "score by ~0.006 of noise that did *not* transfer out-of-fold — so we don't tune weights.\n"
       "- **Paired repeated-CV headline.** On the single seed-42 split CatBoost is ~+0.003 lucky, "
       "which hides the gain; the honest number is a paired `RepeatedStratifiedKFold(5×5)` vs "
       "single CatBoost on identical, leak-free folds."),
    code(TOOLBOX),
    code(LOG_HELPER),
    code(NB09_LOAD),
    md("## MLP under the shared folds (scaled, imputed, seed-bagged)\n"
       "`MLP(256,128)`, ReLU, `alpha=1e-3`, early stopping. Per fold: `StandardScaler` + EOG "
       "median fit on the **train fold only**; bagged over 3 seeds to tame MLP variance. We also "
       "report the **error-correlation** with CatBoost — the lower it is, the more an ensemble can help."),
    code(NB09_MLP),
    md("## Fixed-weight cross-family ensemble (no OOF weight search)\n"
       "Equal-weight average of the CatBoost and MLP probabilities. Reported on the seed-42 split "
       "for continuity with the other notebooks, but read the paired test below for the honest gain."),
    code(NB09_ENS),
    md("## Honest headline — paired `RepeatedStratifiedKFold(5×5)` vs single CatBoost\n"
       "Both arms run on identical fresh folds (seed 2026) and are strictly leak-free (CatBoost at "
       "fixed iterations, MLP scaler/impute inside the train fold). This is the number to trust: it "
       "averages out the single-split fold luck that makes the seed-42 delta look smaller than it is."),
    code(NB09_PAIRED),
    md("## Submission (`sub05`) — CatBoost + MLP cross-family ensemble\n"
       "The first submission built from **two model families**. Recommended as the new primary "
       "private-LB pick (with `sub01` single CatBoost kept as the deterministic reference hedge)."),
    code(NB09_SUB),
    md("### Takeaways\n"
       "- The ~0.829 wall was **not** an irreducible data limit — it was a *single-family* limit. "
       "Adding a non-tree learner (MLP) clears it.\n"
       "- The MLP is both competitive on its own and **decorrelated** from CatBoost (error-corr "
       "~0.65 vs ~0.9 tree-vs-tree), which is why the cross-family blend works where the all-tree "
       "ensembles failed.\n"
       "- Honest, paired-CV gain over single CatBoost is **~+0.005 macro-F1** (well past the "
       "~0.0017 noise floor), and it lifts the weak **Deep** class. Weights are fixed a-priori, so "
       "there is no in-sample weight overfitting.\n"
       "- Remaining ceiling: Deep is symmetrically confused with Light/REM in feature space — "
       "lifting it further needs genuinely new signal, not more recombinations of `features_v1`."),
])

# ===========================================================================
# 10_svm_ensemble.ipynb  (cross-FAMILY ensemble: CatBoost + SVM-RBF, grid + bag)
# ===========================================================================
# Same cross-family idea as 09 but with an SVM-RBF instead of an MLP (the chosen
# non-tree learner). Pipeline: (1) GRID-CV over (C, gamma) on the shared folds,
# scored fast with predict-only OOF; (2) BAG the top-K configs (probability=True)
# into one SVM-RBF probability matrix; (3) FIXED 0.5/0.5 blend with the deployed
# CatBoost; (4) honest PAIRED RepeatedStratifiedKFold vs single CatBoost. SVM-RBF
# is more correlated with CatBoost (~0.79) than the MLP was (~0.65), so expect a
# smaller-but-real cross-family gain. Weights fixed a-priori (no OOF weight search).
NB10_LOAD = r'''from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RepeatedStratifiedKFold
from catboost import CatBoostClassifier

cols = json.load(open(ART / "feature_cols.json"))["columns"]
Xtr = pd.DataFrame(np.load(ART / "features_v1_train.npy"), columns=cols)
Xte = pd.DataFrame(np.load(ART / "features_v1_test.npy"), columns=cols)
y = np.load(ART / "y_train.npy")
test_ids = np.load(ART / "test_ids.npy")
folds = load_folds()
gate = json.load(open(ART / "repeated_cv_gate.json"))
bp = json.load(open(ART / "catboost_best_params.json"))

# Anchor = the deployed single CatBoost (sub01): reuse its saved OOF/test probs.
cat_oof = np.load(ART / "catboost_tuned_oof.npy")
cat_test = np.load(ART / "catboost_tuned_test.npy")
anchor = macro_f1(y, cat_oof.argmax(1))
print("features_v1:", Xtr.shape[1], "cols | anchor CatBoost OOF macro-F1:", round(anchor, 5))
print("gate:", gate["mean"], "+/-", gate["std"], "| deployed CatBoost config:", bp)


def scaled_fold(Xdf, tr_idx, va_idx):
    """Train-fold-only EOG median impute + StandardScaler; return scaled arrays."""
    Xt, Xv = Xdf.iloc[tr_idx].copy(), Xdf.iloc[va_idx].copy()
    med = Xt[EOG].median(); Xt[EOG] = Xt[EOG].fillna(med); Xv[EOG] = Xv[EOG].fillna(med)
    sc = StandardScaler().fit(Xt.values)
    return sc.transform(Xt.values), sc.transform(Xv.values), sc, med'''

NB10_GRID = r'''# (1) GRID-CV over (C, gamma). Scored with predict-only OOF (no Platt calibration)
# so the search is cheap; the top configs are refit WITH probability for blending.
C_GRID = [0.5, 1, 2, 3, 5]
GAMMA_GRID = ["scale", 0.02, 0.08]

def svm_oof_predict(Xdf, ydat, the_folds, C, gamma):
    pred = np.zeros(len(ydat), dtype=int)
    for tr_idx, va_idx in the_folds:
        Xts, Xvs, _, _ = scaled_fold(Xdf, tr_idx, va_idx)
        m = SVC(C=C, gamma=gamma, kernel="rbf", random_state=SEED)
        m.fit(Xts, ydat[tr_idx])
        pred[va_idx] = m.predict(Xvs)
    return macro_f1(ydat, pred)

grid = []
for C in C_GRID:
    for g in GAMMA_GRID:
        s = svm_oof_predict(Xtr, y, folds, C, g)
        grid.append({"C": C, "gamma": g, "oof": round(float(s), 5)})
        print(f"  C={C:<3} gamma={str(g):<6} -> OOF macro-F1 = {s:.5f}")
grid.sort(key=lambda d: -d["oof"])
TOPK = 3
top = grid[:TOPK]
print("\nbest single SVM-RBF:", grid[0], "| anchor CatBoost:", round(anchor, 5))
print(f"top-{TOPK} configs to bag:", [(d["C"], d["gamma"], d["oof"]) for d in top])
json.dump({"grid": grid, "topk": top}, open(ART / "svm_grid.json", "w"), indent=2)'''

NB10_BAG = r'''# (2) BAG the top-K configs with probability=True (Platt) into one SVM probability
# matrix. probability=True is slow (internal CV), so we only pay it for the few
# kept configs, not the whole grid.
def svm_proba_bag(Xdf, ydat, Xtest_df, the_folds, configs):
    ncl = len(CLASSES)
    oof = np.zeros((len(ydat), ncl)); test_p = np.zeros((len(Xtest_df), ncl))
    for tr_idx, va_idx in the_folds:
        Xt, Xv, Xe = Xdf.iloc[tr_idx].copy(), Xdf.iloc[va_idx].copy(), Xtest_df.copy()
        med = Xt[EOG].median()
        for d in (Xt, Xv, Xe):
            d[EOG] = d[EOG].fillna(med)
        sc = StandardScaler().fit(Xt.values)
        Xts, Xvs, Xes = sc.transform(Xt.values), sc.transform(Xv.values), sc.transform(Xe.values)
        for cfg in configs:
            m = SVC(C=cfg["C"], gamma=cfg["gamma"], kernel="rbf",
                    probability=True, random_state=SEED)
            m.fit(Xts, ydat[tr_idx])
            oof[va_idx] += _aligned_proba(m, Xvs) / len(configs)
            test_p += _aligned_proba(m, Xes) / (len(the_folds) * len(configs))
    return oof, test_p

svm_oof_p, svm_test_p = svm_proba_bag(Xtr, y, Xte, folds, top)
svm_macro = macro_f1(y, svm_oof_p.argmax(1))
cat_wrong = (cat_oof.argmax(1) != y).astype(int)
svm_wrong = (svm_oof_p.argmax(1) != y).astype(int)
errcorr = float(np.corrcoef(cat_wrong, svm_wrong)[0, 1])
print(f"SVM-RBF bag (top-{TOPK}) OOF macro-F1 = {svm_macro:.5f}")
print("SVM bag per-class F1:", per_class_f1(y, svm_oof_p.argmax(1)))
print(f"error-correlation SVM vs CatBoost = {errcorr:.3f}  (lower = more diverse)")
np.save(ART / "svm_oof.npy", svm_oof_p); np.save(ART / "svm_test.npy", svm_test_p)
log_result("10_svm_ensemble", "svm_rbf_bag", "features_v1", svm_macro,
           per_class_f1(y, svm_oof_p.argmax(1)),
           f"top-{TOPK} SVM-RBF bag {[(d['C'], d['gamma']) for d in top]}; errcorr_vs_cat={errcorr:.3f}")'''

NB10_ENS = r'''# (3) Fixed equal-weight cross-family blend (a-priori; NO OOF weight search).
W_CAT, W_SVM = 0.5, 0.5
ens_oof = W_CAT * cat_oof + W_SVM * svm_oof_p
ens_test = W_CAT * cat_test + W_SVM * svm_test_p
ens_macro = macro_f1(y, ens_oof.argmax(1))
beats = ens_macro > anchor + gate["std"]
print(f"anchor single CatBoost      = {anchor:.5f}")
print(f"SVM-RBF bag                 = {svm_macro:.5f}")
print(f"ENSEMBLE 0.5*Cat + 0.5*SVM  = {ens_macro:.5f}   (delta vs anchor {ens_macro - anchor:+.5f})")
print(f"gate 1-sigma = {gate['std']:.5f} -> clears anchor+1sigma on the seed-42 split?", bool(beats))
print("ensemble per-class F1:", per_class_f1(y, ens_oof.argmax(1)))
print("(seed-42 CatBoost is ~+0.003 lucky, so this single-split delta understates the gain;")
print(" the paired repeated-CV below is the honest headline.)")
np.save(ART / "svm_ensemble_oof.npy", ens_oof); np.save(ART / "svm_ensemble_test.npy", ens_test)
json.dump({"weights": {"catboost": W_CAT, "svm": W_SVM}, "top_configs": top,
           "svm_macro": round(float(svm_macro), 5), "anchor": round(float(anchor), 5),
           "ensemble_macro": round(float(ens_macro), 5), "errcorr_vs_cat": round(errcorr, 3),
           "beats_gate_seed42": bool(beats)}, open(ART / "svm_ensemble.json", "w"), indent=2)
log_result("10_svm_ensemble", "catboost_plus_svm", "features_v1", ens_macro,
           per_class_f1(y, ens_oof.argmax(1)),
           f"fixed 0.5/0.5 Cat+SVM; vs anchor {anchor:.5f} ({ens_macro - anchor:+.5f}); errcorr={errcorr:.3f}")'''

NB10_PAIRED = r'''# (4) Honest paired comparison on fresh folds. SVM(probability=True) is costly,
# so we use RepeatedStratifiedKFold(5x3) with the top-2 SVM bag (vs 5x5 in nb09).
# Both arms leak-free: CatBoost fixed iterations; SVM scaler/impute inside the train
# fold. Weights fixed 0.5/0.5.
N_REP = 3
PAIR_TOP = top[:2]
CAT_ITERS = 900                              # ~ deployed best_iter (~921); fixed => leak-free
rskf = RepeatedStratifiedKFold(n_splits=N_FOLDS, n_repeats=N_REP, random_state=2026)
splits = list(rskf.split(np.zeros(len(y)), y))

def make_cat_fixed(seed=SEED):
    return CatBoostClassifier(loss_function="MultiClass", eval_metric="TotalF1:average=Macro",
        iterations=CAT_ITERS, learning_rate=bp["lr"], depth=bp["depth"], l2_leaf_reg=bp["l2"],
        random_seed=seed, allow_writing_files=False, thread_count=-1, verbose=False)

cat_sc, ens_sc = [], []
for r in range(N_REP):
    cat_o = np.zeros((len(y), len(CLASSES)))
    svm_o = np.zeros((len(y), len(CLASSES)))
    for tr_idx, va_idx in splits[r * N_FOLDS:(r + 1) * N_FOLDS]:
        cb = make_cat_fixed(); cb.fit(Xtr.iloc[tr_idx], y[tr_idx])
        cat_o[va_idx] = _aligned_proba(cb, Xtr.iloc[va_idx])
        Xt, Xv = Xtr.iloc[tr_idx].copy(), Xtr.iloc[va_idx].copy()
        med = Xt[EOG].median(); Xt[EOG] = Xt[EOG].fillna(med); Xv[EOG] = Xv[EOG].fillna(med)
        sc = StandardScaler().fit(Xt.values)
        Xts, Xvs = sc.transform(Xt.values), sc.transform(Xv.values)
        for cfg in PAIR_TOP:
            m = SVC(C=cfg["C"], gamma=cfg["gamma"], kernel="rbf",
                    probability=True, random_state=SEED)
            m.fit(Xts, y[tr_idx])
            svm_o[va_idx] += _aligned_proba(m, Xvs) / len(PAIR_TOP)
    cs = macro_f1(y, cat_o.argmax(1))
    es = macro_f1(y, (0.5 * cat_o + 0.5 * svm_o).argmax(1))
    cat_sc.append(cs); ens_sc.append(es)
    print(f"  repeat {r + 1}: CatBoost {cs:.5f} | Ensemble {es:.5f} | delta {es - cs:+.5f}")

cat_sc, ens_sc = np.array(cat_sc), np.array(ens_sc)
d = ens_sc - cat_sc
print(f"\nPAIRED repeated CV ({N_REP}x{N_FOLDS}, seed=2026):")
print(f"  single CatBoost : {cat_sc.mean():.5f} +/- {cat_sc.std():.5f}")
print(f"  Cat+SVM ensemble: {ens_sc.mean():.5f} +/- {ens_sc.std():.5f}")
print(f"  paired delta    : {d.mean():+.5f} +/- {d.std():.5f}  (positive in {int((d > 0).sum())}/{N_REP} repeats)")
print(f"  => honest gain beyond the {gate['std']:.5f} noise floor?", bool(d.mean() > gate["std"]))
json.dump({"n_repeats": N_REP, "n_folds": N_FOLDS, "cat_iters": CAT_ITERS,
           "pair_configs": PAIR_TOP, "cat_mean": round(float(cat_sc.mean()), 5),
           "cat_std": round(float(cat_sc.std()), 5), "ens_mean": round(float(ens_sc.mean()), 5),
           "ens_std": round(float(ens_sc.std()), 5), "delta_mean": round(float(d.mean()), 5),
           "delta_std": round(float(d.std()), 5), "pos_repeats": int((d > 0).sum())},
          open(ART / "svm_ensemble_paired_cv.json", "w"), indent=2)
log_result("10_svm_ensemble", "catboost_plus_svm_pairedCV", "features_v1", float(ens_sc.mean()),
           per_class_f1(y, ens_oof.argmax(1)),
           f"paired {N_REP}x{N_FOLDS}: ens {ens_sc.mean():.5f} vs cat {cat_sc.mean():.5f} (delta {d.mean():+.5f})")'''

NB10_SUB = r'''name6 = "sub06_CatBoostPlusSVM_Ensemble.csv"
pred6 = ens_test.argmax(1).astype(int)
sub6 = pd.DataFrame({"id": test_ids, "sleep_stage": pred6})
assert sub6.shape == (5000, 2)
assert sub6["id"].tolist() == list(range(9000, 14000))
assert sub6["sleep_stage"].isin(CLASSES).all()
sub6.to_csv(SUB / name6, index=False)
print("wrote", SUB / name6, "| ensemble OOF macro-F1:", round(ens_macro, 5))
print("class counts:", sub6["sleep_stage"].value_counts().sort_index().to_dict())
single_test = np.load(ART / "catboost_tuned_test.npy")
diff = int((ens_test.argmax(1) != single_test.argmax(1)).sum())
print(f"sub06 vs single-CatBoost (sub01) test preds differ on {diff} of 5000 rows")'''

nb10 = make_nb([
    md("# 10 — Cross-family ensemble: CatBoost + SVM-RBF (grid + bag)\n"
       "Same cross-family idea as `09`, but the non-tree partner is an **SVM-RBF** (the chosen "
       "learner) instead of an MLP. An RBF SVM draws smooth, curved decision boundaries that the "
       "axis-aligned trees only approximate — so it is diverse from CatBoost, just less so than the "
       "MLP was (error-corr ~0.79 vs ~0.65), which caps the blend gain.\n\n"
       "Steps: **(1) grid-CV** over `(C, gamma)` scored with cheap predict-only OOF; **(2) bag** the "
       "top-K configs with calibrated probabilities; **(3) fixed 0.5/0.5 blend** with the deployed "
       "CatBoost (weights a-priori — no OOF weight search, which the audit showed inflates the score); "
       "**(4) honest paired `RepeatedStratifiedKFold`** vs single CatBoost."),
    code(TOOLBOX),
    code(LOG_HELPER),
    code(NB10_LOAD),
    md("## (1) Grid-CV over (C, gamma)\nScored with predict-only OOF on the shared folds (no Platt "
       "calibration → fast). We keep the top-K configs for the bag. Sweet spot is C≈1–3, "
       "`gamma='scale'`; large C overfits."),
    code(NB10_GRID),
    md("## (2) Bag the top-K SVM-RBF configs\nRefit the kept configs **with** `probability=True` "
       "(Platt) and average their probabilities. Per fold: `StandardScaler` + EOG median fit on the "
       "**train fold only**. We report the error-correlation with CatBoost — the diversity that makes "
       "the blend work."),
    code(NB10_BAG),
    md("## (3) Fixed-weight cross-family ensemble (no OOF weight search)\nEqual-weight average of the "
       "CatBoost and SVM-RBF probabilities, reported on the seed-42 split. The paired test below is the "
       "honest gain."),
    code(NB10_ENS),
    md("## (4) Honest headline — paired `RepeatedStratifiedKFold(5×3)` vs single CatBoost\n"
       "Both arms on identical fresh folds (seed 2026), leak-free (CatBoost fixed iterations, SVM "
       "scaler/impute inside the train fold). Fewer repeats than nb09 (5×3 not 5×5) because "
       "`SVC(probability=True)` is costly; still enough to confirm the sign and rough size of the gain."),
    code(NB10_PAIRED),
    md("## Submission (`sub06`) — CatBoost + SVM-RBF cross-family ensemble\n"
       "The SVM-based cross-family pick (per the chosen direction). Compare its OOF/paired gain against "
       "the single CatBoost reference before finalizing the private-LB pair."),
    code(NB10_SUB),
    md("### Takeaways\n"
       "- SVM-RBF is a genuine non-tree partner: it edges CatBoost on its own and a grid+bag stabilizes "
       "the operating point.\n"
       "- The cross-family blend with CatBoost is the honest win; because SVM is more correlated with "
       "CatBoost than the MLP was (~0.79 vs ~0.65), expect a **smaller** gain than nb09 — read the "
       "paired number, not the lucky seed-42 split.\n"
       "- Weights fixed a-priori (0.5/0.5) → no in-sample weight overfitting. `sub06` is the SVM "
       "cross-family candidate; the single CatBoost (`sub01`) stays the deterministic reference."),
])

# ===========================================================================
# 11_multifamily_ensemble.ipynb  (add decorrelated NON-tree families: QDA/GMM/Nystroem)
# ===========================================================================
# The Cat+SVM blend (sub06, public 0.84157) wins mostly on variance: SVM-RBF is only
# moderately decorrelated from CatBoost (errcorr ~0.76). The biggest untapped lever is a
# NEW, genuinely decorrelated non-tree family. Generative/probabilistic learners (QDA,
# GaussianNB, per-class GMM) and a random-feature linear model (Nystroem+LogReg) have a
# different inductive bias from BOTH trees and the RBF SVM. Measurement-first: measure each
# family (strength + error-correlation vs CatBoost AND SVM), keep only the strong & decorrelated
# ones, then build a multi-family ensemble with (a) fixed a-priori weights, (b) honest nested-CV
# weights, (c) nested-CV LogReg stacking. Ship sub07 only if it beats sub06 on the paired CV.
NB11_LOAD = r'''from sklearn.discriminant_analysis import (QuadraticDiscriminantAnalysis,
                                              LinearDiscriminantAnalysis)
from sklearn.naive_bayes import GaussianNB
from sklearn.mixture import GaussianMixture
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_predict
from catboost import CatBoostClassifier

cols = json.load(open(ART / "feature_cols.json"))["columns"]
Xtr = pd.DataFrame(np.load(ART / "features_v1_train.npy"), columns=cols)
Xte = pd.DataFrame(np.load(ART / "features_v1_test.npy"), columns=cols)
y = np.load(ART / "y_train.npy")
test_ids = np.load(ART / "test_ids.npy")
folds = load_folds()
gate = json.load(open(ART / "repeated_cv_gate.json"))
bp = json.load(open(ART / "catboost_best_params.json"))

# Reuse the two deployed members WITHOUT retraining (saved, row-aligned to the shared folds).
cat_oof = np.load(ART / "catboost_tuned_oof.npy"); cat_test = np.load(ART / "catboost_tuned_test.npy")
svm_oof = np.load(ART / "svm_oof.npy"); svm_test = np.load(ART / "svm_test.npy")
anchor = macro_f1(y, cat_oof.argmax(1)); svm_macro = macro_f1(y, svm_oof.argmax(1))
cat_wrong = (cat_oof.argmax(1) != y).astype(int)
svm_wrong = (svm_oof.argmax(1) != y).astype(int)
print("anchor CatBoost:", round(anchor, 5), "| SVM bag:", round(svm_macro, 5),
      "| gate std:", gate["std"])

def scaled_fold(Xdf, tr_idx, va_idx):
    """Train-fold-only EOG median impute + StandardScaler; return scaled arrays."""
    Xt, Xv = Xdf.iloc[tr_idx].copy(), Xdf.iloc[va_idx].copy()
    med = Xt[EOG].median(); Xt[EOG] = Xt[EOG].fillna(med); Xv[EOG] = Xv[EOG].fillna(med)
    sc = StandardScaler().fit(Xt.values)
    return sc.transform(Xt.values), sc.transform(Xv.values), sc, med'''

NB11_FAMILIES = r'''def scaled_proba_oof(make_est, Xdf, ydat, Xtest_df, the_folds):
    """Generic non-tree OOF: per fold scale+impute on the train fold, fit, predict_proba."""
    ncl = len(CLASSES)
    oof = np.zeros((len(ydat), ncl)); test_p = np.zeros((len(Xtest_df), ncl))
    for tr_idx, va_idx in the_folds:
        Xt, Xv, Xe = Xdf.iloc[tr_idx].copy(), Xdf.iloc[va_idx].copy(), Xtest_df.copy()
        med = Xt[EOG].median()
        for d in (Xt, Xv, Xe):
            d[EOG] = d[EOG].fillna(med)
        sc = StandardScaler().fit(Xt.values)
        Xts, Xvs, Xes = sc.transform(Xt.values), sc.transform(Xv.values), sc.transform(Xe.values)
        m = make_est(); m.fit(Xts, ydat[tr_idx])
        oof[va_idx] = _aligned_proba(m, Xvs)
        test_p += _aligned_proba(m, Xes) / len(the_folds)
    return oof, test_p


class GMMClassifier:
    """Generative Bayes: one full-covariance GaussianMixture per class; posterior
    proportional to prior * density. reg_covar keeps covariances non-singular."""
    def __init__(self, n_components=2, reg_covar=1e-3, seed=SEED):
        self.k, self.reg, self.seed = n_components, reg_covar, seed
    def fit(self, X, yv):
        self.classes_ = np.unique(yv); self.gmms_, self.logpri_ = [], []
        for c in self.classes_:
            g = GaussianMixture(self.k, covariance_type="full", reg_covar=self.reg,
                                random_state=self.seed, max_iter=200).fit(X[yv == c])
            self.gmms_.append(g); self.logpri_.append(np.log((yv == c).mean()))
        return self
    def predict_proba(self, X):
        logp = np.column_stack([g.score_samples(X) + lp
                                for g, lp in zip(self.gmms_, self.logpri_)])
        logp -= logp.max(1, keepdims=True); p = np.exp(logp)
        return p / p.sum(1, keepdims=True)


FAMILIES = {
    "qda":      lambda: QuadraticDiscriminantAnalysis(reg_param=0.1),
    "lda":      lambda: LinearDiscriminantAnalysis(),
    "gnb":      lambda: GaussianNB(),
    "gmm2":     lambda: GMMClassifier(2),
    "gmm3":     lambda: GMMClassifier(3),
    "nystroem": lambda: make_pipeline(
                    Nystroem(gamma=0.02, n_components=300, random_state=SEED),
                    LogisticRegression(max_iter=2000, C=1.0)),
}'''

NB11_MEASURE = r'''# Measure each family: OOF macro-F1, Deep F1, and error-correlation vs BOTH CatBoost and SVM.
# Each fit is guarded so one family failing (e.g. singular covariance) skips, not kills the run.
results = {}
for name, mk in FAMILIES.items():
    try:
        oof, tst = scaled_proba_oof(mk, Xtr, y, Xte, folds)
    except Exception as e:
        print(f"  {name:<9} FAILED: {type(e).__name__}: {e}")
        continue
    m = macro_f1(y, oof.argmax(1))
    wrong = (oof.argmax(1) != y).astype(int)
    ec_cat = float(np.corrcoef(wrong, cat_wrong)[0, 1])
    ec_svm = float(np.corrcoef(wrong, svm_wrong)[0, 1])
    pcf = per_class_f1(y, oof.argmax(1))
    np.save(ART / f"{name}_oof.npy", oof); np.save(ART / f"{name}_test.npy", tst)
    results[name] = {"oof": oof, "test": tst, "macro": m,
                     "ec_cat": ec_cat, "ec_svm": ec_svm, "deep": pcf["Deep"]}
    print(f"  {name:<9} OOF={m:.5f}  Deep={pcf['Deep']:.4f}  errcorr cat={ec_cat:.3f} svm={ec_svm:.3f}")
    log_result("11_multifamily", name, "features_v1", m, pcf,
               f"non-tree family; errcorr cat={ec_cat:.3f} svm={ec_svm:.3f}")

# Eligible iff strong enough (not a weak dragger) AND decorrelated from BOTH cat and svm.
STRENGTH_FLOOR = anchor - 2 * gate["std"]
eligible = [n for n, r in results.items()
            if r["macro"] >= STRENGTH_FLOOR and r["ec_cat"] < 0.75 and r["ec_svm"] < 0.85]
print(f"\nstrength floor (anchor - 2*gate.std) = {STRENGTH_FLOOR:.5f}")
print("ELIGIBLE non-tree families (strong + decorrelated from cat & svm):", eligible)'''

NB11_ENSEMBLES = r'''# Members = CatBoost + SVM (reused) + eligible new families. Three leak-free combinations.
member_oof = {"cat": cat_oof, "svm": svm_oof}
member_test = {"cat": cat_test, "svm": svm_test}
for n in eligible:
    member_oof[n] = results[n]["oof"]; member_test[n] = results[n]["test"]
names = list(member_oof.keys())
nontree = [n for n in names if n != "cat"]
print("ensemble members:", names)

# (a) Fixed a-priori weights: half mass to CatBoost (tree), half split across non-tree members.
w_fixed = {"cat": 0.5}
for n in nontree:
    w_fixed[n] = 0.5 / len(nontree)
ens_fixed_oof = sum(w_fixed[n] * member_oof[n] for n in names)
ens_fixed_test = sum(w_fixed[n] * member_test[n] for n in names)
mf_fixed = macro_f1(y, ens_fixed_oof.argmax(1))
w_fixed_d = {k: round(float(v), 3) for k, v in w_fixed.items()}
print(f"(a) fixed weights {w_fixed_d} -> OOF {mf_fixed:.5f}")

# (b) Honest nested-CV convex weights: for each held-out fold, grid-search convex weights on the
#     OTHER folds' OOF and apply to the held-out fold. Aggregated => leak-free OOF (NOT a full-OOF
#     search, which previously inflated the score by ~0.006).
def comps(total, parts):
    if parts == 1:
        yield (total,); return
    for i in range(total + 1):
        for rest in comps(total - i, parts - 1):
            yield (i,) + rest

def best_convex_weights(idx, steps=10):
    mats = [member_oof[n][idx] for n in names]; yi = y[idx]; best_w, best_s = None, -1.0
    for comp in comps(steps, len(names)):
        w = np.array(comp) / steps
        s = macro_f1(yi, sum(wi * mm for wi, mm in zip(w, mats)).argmax(1))
        if s > best_s:
            best_s, best_w = s, w
    return best_w

ens_nested_oof = np.zeros_like(cat_oof)
for tr_idx, va_idx in folds:
    w = best_convex_weights(tr_idx)
    ens_nested_oof[va_idx] = sum(w[i] * member_oof[n][va_idx] for i, n in enumerate(names))
mf_nested = macro_f1(y, ens_nested_oof.argmax(1))
w_full = best_convex_weights(np.arange(len(y)))                 # deployed test weights
w_full_d = {n: round(float(w_full[i]), 3) for i, n in enumerate(names)}
ens_nested_test = sum(w_full[i] * member_test[n] for i, n in enumerate(names))
print(f"(b) nested-CV weights -> honest OOF {mf_nested:.5f} | deployed weights {w_full_d}")

# (c) Nested-CV stacking: multinomial LogisticRegression meta over the members' OOF probability
#     columns, honest via cross_val_predict on the shared folds; refit on full members for test.
stack_X_oof = np.hstack([member_oof[n] for n in names])
stack_X_test = np.hstack([member_test[n] for n in names])
meta = LogisticRegression(max_iter=3000, C=1.0)
stack_oof = cross_val_predict(meta, stack_X_oof, y, cv=folds, method="predict_proba")
mf_stack = macro_f1(y, stack_oof.argmax(1))
meta.fit(stack_X_oof, y)
stack_test = meta.predict_proba(stack_X_test)
print(f"(c) nested-CV stack (LogReg meta) -> honest OOF {mf_stack:.5f}")

# Choose the best HONEST variant.
variants = {"fixed": (mf_fixed, ens_fixed_oof, ens_fixed_test),
            "nested": (mf_nested, ens_nested_oof, ens_nested_test),
            "stack": (mf_stack, stack_oof, stack_test)}
best_name = max(variants, key=lambda k: variants[k][0])
mf_best, mf_oof, mf_test = variants[best_name]
variant_oof = {k: round(float(v[0]), 5) for k, v in variants.items()}
print("\nvariant OOF:", variant_oof, "| chosen:", best_name)
print(f"chosen multi-family OOF {mf_best:.5f} vs sub06 Cat+SVM 0.83269 vs anchor {anchor:.5f}")
print("multi-family per-class F1:", per_class_f1(y, mf_oof.argmax(1)))
np.save(ART / "mf_ensemble_oof.npy", mf_oof); np.save(ART / "mf_ensemble_test.npy", mf_test)
json.dump({"members": names, "eligible_new": eligible, "fixed_weights": w_fixed_d,
           "nested_weights": w_full_d, "variant_oof": variant_oof, "chosen": best_name,
           "best_oof": round(float(mf_best), 5)}, open(ART / "mf_ensemble.json", "w"), indent=2)
log_result("11_multifamily", f"mf_{best_name}", "features_v1", mf_best,
           per_class_f1(y, mf_oof.argmax(1)), f"members={names}; variants={variant_oof}")'''

NB11_PAIRED = r'''# Honest paired RepeatedStratifiedKFold(5x3, seed=2026). Three arms on IDENTICAL folds:
# single CatBoost, Cat+SVM (the sub06 bar), and the FIXED-weight multi-family blend (the
# conservative, fully leak-free family-contribution check). Both non-cat arms refit per fold:
# CatBoost at fixed iterations (no early-stop peek), SVM top-2 + eligible families with scaler/
# impute fit inside the train fold.
N_REP = 3
PAIR_TOP = json.load(open(ART / "svm_grid.json"))["topk"][:2]
PAIR_FAMILIES = list(eligible)
CAT_ITERS = 900
rskf = RepeatedStratifiedKFold(n_splits=N_FOLDS, n_repeats=N_REP, random_state=2026)
splits = list(rskf.split(np.zeros(len(y)), y))

def make_cat_fixed(seed=SEED):
    return CatBoostClassifier(loss_function="MultiClass", eval_metric="TotalF1:average=Macro",
        iterations=CAT_ITERS, learning_rate=bp["lr"], depth=bp["depth"], l2_leaf_reg=bp["l2"],
        random_seed=seed, allow_writing_files=False, thread_count=-1, verbose=False)

cat_s, cs_s, mf_s = [], [], []
for r in range(N_REP):
    cat_o = np.zeros((len(y), len(CLASSES)))
    svm_o = np.zeros((len(y), len(CLASSES)))
    fam_o = {n: np.zeros((len(y), len(CLASSES))) for n in PAIR_FAMILIES}
    for tr_idx, va_idx in splits[r * N_FOLDS:(r + 1) * N_FOLDS]:
        cb = make_cat_fixed(); cb.fit(Xtr.iloc[tr_idx], y[tr_idx])
        cat_o[va_idx] = _aligned_proba(cb, Xtr.iloc[va_idx])
        Xts, Xvs, _, _ = scaled_fold(Xtr, tr_idx, va_idx)
        for cfg in PAIR_TOP:
            m = SVC(C=cfg["C"], gamma=cfg["gamma"], kernel="rbf", probability=True, random_state=SEED)
            m.fit(Xts, y[tr_idx]); svm_o[va_idx] += _aligned_proba(m, Xvs) / len(PAIR_TOP)
        for n in PAIR_FAMILIES:
            est = FAMILIES[n](); est.fit(Xts, y[tr_idx])
            fam_o[n][va_idx] = _aligned_proba(est, Xvs)
    nt = 1 + len(PAIR_FAMILIES)
    mf = 0.5 * cat_o + (0.5 / nt) * svm_o + sum((0.5 / nt) * fam_o[n] for n in PAIR_FAMILIES)
    cat_s.append(macro_f1(y, cat_o.argmax(1)))
    cs_s.append(macro_f1(y, (0.5 * cat_o + 0.5 * svm_o).argmax(1)))
    mf_s.append(macro_f1(y, mf.argmax(1)))
    print(f"  repeat {r + 1}: Cat {cat_s[-1]:.5f} | Cat+SVM {cs_s[-1]:.5f} | MultiFam {mf_s[-1]:.5f}")

cat_s, cs_s, mf_s = np.array(cat_s), np.array(cs_s), np.array(mf_s)
d = mf_s - cs_s
print(f"\nPAIRED {N_REP}x{N_FOLDS} (seed 2026):")
print(f"  single Cat     : {cat_s.mean():.5f} +/- {cat_s.std():.5f}")
print(f"  Cat+SVM (sub06): {cs_s.mean():.5f} +/- {cs_s.std():.5f}")
print(f"  MultiFamily    : {mf_s.mean():.5f} +/- {mf_s.std():.5f}")
print(f"  delta MultiFam vs Cat+SVM: {d.mean():+.5f} +/- {d.std():.5f}  (positive in {int((d > 0).sum())}/{N_REP})")
print(f"  beats gate ({gate['std']:.5f})?", bool(d.mean() > gate["std"]))
json.dump({"n_repeats": N_REP, "n_folds": N_FOLDS, "members": names, "eligible_new": eligible,
           "cat_mean": round(float(cat_s.mean()), 5), "catsvm_mean": round(float(cs_s.mean()), 5),
           "catsvm_std": round(float(cs_s.std()), 5), "mf_mean": round(float(mf_s.mean()), 5),
           "mf_std": round(float(mf_s.std()), 5), "delta_vs_catsvm": round(float(d.mean()), 5),
           "delta_std": round(float(d.std()), 5), "pos_repeats": int((d > 0).sum()),
           "note": "multi-family arm uses FIXED a-priori weights (conservative, leak-free)"},
          open(ART / "mf_ensemble_paired_cv.json", "w"), indent=2)
log_result("11_multifamily", "mf_pairedCV", "features_v1", float(mf_s.mean()),
           per_class_f1(y, mf_oof.argmax(1)),
           f"paired {N_REP}x{N_FOLDS}: MF {mf_s.mean():.5f} vs Cat+SVM {cs_s.mean():.5f} (delta {d.mean():+.5f})")'''

NB11_SUB = r'''# Ship sub07 only if the new families ROBUSTLY help (fixed-weight multi-family beats Cat+SVM in
# every paired repeat) AND the chosen deployed variant beats sub06's OOF. Else keep sub06.
SUB06_OOF = 0.83269
families_help = bool((d.mean() > 0) and (int((d > 0).sum()) >= N_REP))
beats_oof = bool(mf_best > SUB06_OOF)
ship = families_help and beats_oof
print("families_help (paired 3/3 & delta>0):", families_help,
      "| chosen OOF > sub06 (%.5f)?" % SUB06_OOF, beats_oof, "| SHIP sub07?", ship)
if ship:
    tag = "-".join(["CatBoost", "SVM"] + [n.upper() for n in eligible])
    name7 = f"sub07_MultiFamily_{tag}_Ensemble.csv"
    pred7 = mf_test.argmax(1).astype(int)
    sub7 = pd.DataFrame({"id": test_ids, "sleep_stage": pred7})
    assert sub7.shape == (5000, 2)
    assert sub7["id"].tolist() == list(range(9000, 14000))
    assert sub7["sleep_stage"].isin(CLASSES).all()
    sub7.to_csv(SUB / name7, index=False)
    print("WROTE", SUB / name7, "| variant", best_name, "| OOF", round(float(mf_best), 5))
    print("class counts:", sub7["sleep_stage"].value_counts().sort_index().to_dict())
    base = np.load(ART / "svm_ensemble_test.npy")
    print("sub07 vs sub06 differ on", int((mf_test.argmax(1) != base.argmax(1)).sum()), "of 5000 rows")
else:
    print("Multi-family did NOT clear the bar -> NOT shipping sub07; sub06 stays primary (honest).")'''

nb11 = make_nb([
    md("# 11 — Multi-family ensemble: add decorrelated non-tree families\n"
       "`sub06` (CatBoost + SVM-RBF, public **0.84157**) wins mostly on **variance** — SVM is only "
       "moderately decorrelated from CatBoost (error-corr ~0.76). The next lever is a **new, more "
       "decorrelated non-tree family**. Generative/probabilistic learners (**QDA, GaussianNB, "
       "per-class GMM**) and a random-feature linear model (**Nystroem+LogReg**) use a bias different "
       "from *both* trees and the RBF SVM.\n\n"
       "**Measurement-first:** measure each family's OOF strength and error-correlation vs **both** "
       "CatBoost and SVM; keep only the strong & decorrelated ones. Then combine via (a) fixed "
       "a-priori weights, (b) **honest nested-CV** weights, (c) nested-CV LogReg **stacking** — all "
       "leak-free. Ship `sub07` only if it beats `sub06` on the paired CV. (MLP excluded by choice.)"),
    code(TOOLBOX),
    code(LOG_HELPER),
    code(NB11_LOAD),
    md("## Generic scaled-OOF runner + the family zoo\nNon-tree models need per-fold scaling + EOG "
       "imputation (`scaled_proba_oof`). `GMMClassifier` is a generative Bayes rule: one "
       "full-covariance Gaussian mixture per class, posterior ∝ prior · density (`reg_covar` keeps it "
       "non-singular — the part a prior attempt crashed on). Each fit is guarded."),
    code(NB11_FAMILIES),
    md("## Measure strength + decorrelation\nA new member only helps an ensemble if it is **strong** "
       "(not a weak dragger like RF/ET/GB) **and decorrelated**. Eligibility: OOF ≥ anchor − 2σ AND "
       "error-corr < 0.75 vs CatBoost AND < 0.85 vs SVM."),
    code(NB11_MEASURE),
    md("## Multi-family ensembles — fixed / nested-CV weights / stacking\nThree leak-free "
       "combinations of CatBoost + SVM + the eligible families. (b) and (c) tune **out-of-fold only** "
       "(per-fold weight fit; `cross_val_predict` meta) — never on the full OOF, which the audit "
       "showed inflates the score by ~0.006 of non-transferring noise. We deploy the best honest variant."),
    code(NB11_ENSEMBLES),
    md("## Honest headline — paired `RepeatedStratifiedKFold(5×3)` vs Cat+SVM (sub06)\nThe bar is "
       "**sub06**, not single CatBoost. The multi-family arm uses **fixed** weights here — a "
       "conservative, fully leak-free check that the new families genuinely add signal across many folds."),
    code(NB11_PAIRED),
    md("## Submission (`sub07`) — conditional\nShipped only if the new families robustly help "
       "(positive paired delta in every repeat) **and** the deployed variant beats sub06's OOF. "
       "Otherwise this is an honest negative result and sub06 stays primary."),
    code(NB11_SUB),
    md("### Takeaways\n"
       "- A member helps only when **strong AND decorrelated** — the eligibility gate enforces both, "
       "so weak/correlated families are dropped (the RF/ET/GB lesson, now automated).\n"
       "- Honest weighting/stacking is done strictly out-of-fold; the deployed variant is the best "
       "*honest*-OOF combination, not an in-sample-tuned one.\n"
       "- `sub07` ships only on a paired-CV win over `sub06`; the private-LB pick stays chosen on "
       "OOF/paired-CV, never on the public leaderboard.\n"
       "- Remaining ceiling is still the Deep↔Light/REM overlap — watch the Deep F1 column to see if a "
       "generative family (QDA/GMM) reshapes that boundary where trees and SVM could not."),
])

# ===========================================================================
# Write all notebooks
# ===========================================================================
NOTEBOOKS = {
    "01_eda.ipynb": nb01,
    "02_baseline.ipynb": nb02,
    "03_features.ipynb": nb03,
    "04_models.ipynb": nb04,
    "05_tune_blend.ipynb": nb05,
    "06_submit.ipynb": nb06,
    "07_robustness.ipynb": nb07,
    "08_catboost_ensemble.ipynb": nb08,
    "09_mlp_ensemble.ipynb": nb09,
    "10_svm_ensemble.ipynb": nb10,
    "11_multifamily_ensemble.ipynb": nb11,
}

if __name__ == "__main__":
    for name, nb in NOTEBOOKS.items():
        with open(name, "w", encoding="utf-8") as f:
            nbf.write(nb, f)
        print("wrote", name)
