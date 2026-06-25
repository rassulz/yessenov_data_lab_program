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
# Write all notebooks
# ===========================================================================
NOTEBOOKS = {
    "01_eda.ipynb": nb01,
    "02_baseline.ipynb": nb02,
    "03_features.ipynb": nb03,
    "04_models.ipynb": nb04,
    "05_tune_blend.ipynb": nb05,
    "06_submit.ipynb": nb06,
}

if __name__ == "__main__":
    for name, nb in NOTEBOOKS.items():
        with open(name, "w", encoding="utf-8") as f:
            nbf.write(nb, f)
        print("wrote", name)
