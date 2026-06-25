# Day9 — Kaggle: Sleep Stage Classification

Write all project files in English. This file is my working plan and source of truth.

## Task
Multi-class classification: predict `sleep_stage` per 30-second epoch from
physiological sensor channels (EEG, EMG, EOG, heart rate, respiration, SpO2).

- Classes (4): `0=Wake`, `1=Light (N1+N2)`, `2=Deep (N3)`, `3=REM`
- Data is synthetic, for learning. Not a clinical tool.

## Metric — macro-F1 (CRITICAL)
- Macro-averaged F1 over the 4 classes. Every class weighs equally regardless of
  frequency. The worst class drags the score, so optimize the weakest class, not
  overall accuracy.
- Public LB = 30% of test, Private LB = 70%. Final rank = private only.
  Do NOT tune to the public LB.
- Submission limit: 20/day.
- There is also a "best approach" prize judged by defense → keep the solution
  clear, justified, and fully reproducible (fixed seeds, one-command rerun).
- Rules: only provided data. No external datasets, no manual labeling of test.

## Data facts (verified)
- train.csv: 9000 rows = id + 21 features + `sleep_stage`
- test.csv: 5000 rows = id + 21 features
- Classes are nearly balanced in train: 1=2442, 3=2320, 2=2237, 0=2001
- Features are already normalized (look z-scored).
- `eog_burst_index`: separate EOG channel, available on ~half the recordings.
  - Missing: train 4501/9000 (50%), test 2477/5000 (~49.5%).
  - No other missing values anywhere.
  - Missingness is likely informative (channel on/off) → add `eog_burst_missing`
    flag; let GBDT handle the NaN natively (do NOT blindly impute the mean).

### Feature list
eeg_delta_power, eeg_theta_power, eeg_alpha_power, eeg_sigma_power,
eeg_beta_power, eeg_gamma_power, eeg_slow_osc_power, eeg_spectral_entropy,
eeg_spindle_density, eeg_kcomplex_rate, emg_chin_tone, emg_tone_variance,
eog_movement_density, eog_amplitude, heart_rate_mean, heart_rate_variability,
respiration_rate, respiration_variability, spo2_mean, body_movement_index,
eog_burst_index (50% missing).

## Validation strategy
- StratifiedKFold, 5 folds, fixed `random_state=42`. Classes balanced so
  stratified split is enough.
- Always report OOF macro-F1 as the primary local metric. Trust CV over public LB.
- Keep an OOF prediction matrix (probabilities) for ensembling + threshold search.
- No grouping/time leakage info available (epochs given independently), so plain
  StratifiedKFold is appropriate. Re-check if an id/recording grouping appears.

## Toolset — built on techniques from today's course (day9)
Lead with what the user just learned (better for the "best approach" defense):
- **sklearn ensembles**: `RandomForestClassifier`, `ExtraTreesClassifier`,
  `GradientBoostingClassifier`, `AdaBoostClassifier`, `BaggingClassifier`
  (`oob_score=True` as a quick sanity check), `VotingClassifier(voting='soft')`.
  Inspect `feature_importances_`.
- **CatBoost** (course notebook): `CatBoostClassifier`, `Pool`, `eval_set` +
  `early_stopping_rounds` + `use_best_model=True`, `cv(fold_count=5, shuffle=True)`,
  `get_feature_importance(prettified=True)`, `predict_proba`, `save_model`.
  - Strong default here: handles the 50% NaN in `eog_burst_index` natively, and
    supports macro-F1 directly via `eval_metric='TotalF1'` + `loss_function='MultiClass'`,
    plus `auto_class_weights='Balanced'`.
- LightGBM/XGBoost are optional extras only if time allows — not the focus, since
  the defense should rest on methods the user understands.

## Plan (step by step)
1. **EDA** (`01_eda.ipynb`): distributions per class, feature-vs-stage boxplots,
   correlations, confirm missingness pattern, sanity-check no leakage.
2. **Baseline** (`02_baseline.py`): CatBoost (`loss_function='MultiClass'`,
   `eval_metric='TotalF1'`) + 5-fold StratifiedKFold, eval macro-F1 on OOF.
   Cross-check with a quick RandomForest (`oob_score=True`). Establish the number
   to beat and make the submission pipeline work end-to-end.
3. **Feature engineering**: EEG band ratios (delta/theta, theta/alpha,
   slow_osc/delta, beta/delta, (delta+theta)/(alpha+beta)), total EEG power,
   relative band powers, autonomic combos (hr/respiration), `eog_burst_missing`
   flag. Keep features only if they improve OOF macro-F1.
4. **Models**: CatBoost (primary) vs RandomForest/ExtraTrees/GradientBoosting.
   Compare OOF macro-F1. For sklearn models, impute `eog_burst_index` + add the
   missing flag (sklearn trees need no NaN); CatBoost takes NaN as-is.
5. **Tune for macro-F1**: tune the best model (start with CatBoost `depth`,
   `learning_rate`, `l2_leaf_reg`, `iterations` via early stopping). Optimize OOF
   macro-F1 directly. Use `auto_class_weights='Balanced'` / `class_weight` even
   though classes look balanced — macro-F1 cares about the weak class.
6. **Decision-threshold / prior adjustment**: search per-class probability
   multipliers on OOF preds to maximize macro-F1 (coordinate ascent), then apply
   the same transform to test. Cheap, often +0.5–1.5 macro-F1.
7. **Ensemble**: soft `VotingClassifier` over CatBoost + RandomForest +
   ExtraTrees (blend OOF probabilities), re-run the threshold search on the blend.
8. **Final submission**: pick by OOF macro-F1, not public LB. Keep 2 diverse subs
   for private LB safety.

## Submission
- Format: `id,sleep_stage` (integer class), one row per test id. See sample_submission.csv.
- Save each as `subNN_<desc>.csv` and log: features, model, CV macro-F1, public LB.

## Reference — course material the user studied (reuse these methods)
- Repo: https://github.com/timurbakibayev/ydl-2026/tree/main/day9
  - `18-Decision-Trees-and-Random-Forests/07_ensemble_learning_and_random_forests.ipynb`
    (Hands-On ML ch.7: voting/bagging/RF/extra-trees/AdaBoost/GBDT, OOB, importances)
  - `19. CatBoost/catboost1.ipynb` (CatBoostClassifier, Pool, eval_set + early
    stopping, cv, get_feature_importance, predict_proba, save/load)

## Environment
- Use anaconda3 python (PATH python lacks sklearn). See user memory.
- CatBoost may need install: `pip install catboost` (check first).
- Fix all seeds. Pipeline must rerun with one command for the "best approach" prize.

## Log (CV macro-F1 / public LB)
Pipeline = 7 notebooks (`01_eda` → `07_robustness`), one-command rerun via
`C:/ProgramData/anaconda3/python.exe run_all.py`. Folds: StratifiedKFold(5, shuffle,
seed=42), shared via `artifacts/folds.npz`. Primary metric = global OOF macro-F1.
Full table in `results_log.csv`.

| step | model | features | OOF macro-F1 | per-class (Wake/Light/Deep/REM) |
|------|-------|----------|--------------|----------------------------------|
| 02 baseline | CatBoost | raw21 + missing | 0.82494 | .852/.847/.771/.829 |
| 02 baseline | RandomForest | raw21 + missing | 0.79899 | .820/.830/.738/.807 |
| 04 models | CatBoost | features_v1 (28) | **0.82898** | .852/.851/.777/.837 |
| 04 models | GradientBoosting | features_v1 | 0.81284 | .840/.838/.760/.814 |
| 04 models | ExtraTrees | features_v1 | 0.79462 | .828/.825/.724/.801 |
| 04 models | RandomForest | features_v1 | 0.79203 | .820/.823/.727/.799 |
| 06 submit | **CatBoost (sub01)** | features_v1 | **0.82898** | .852/.851/.777/.837 |
| 06 submit | SoftVotingEnsemble (sub02) | all 4 models | 0.81952 | .845/.846/.765/.823 |
| 07 robust | CatBoost RepeatedCV **(gate)** | features_v1 | **0.82598 ± 0.00169** | .851/.849/.773/.830 |
| 07 robust | CatBoost seed-bag (K=9, sub03) | features_v1 | 0.82415 | .849/.847/.773/.828 |

Public LB (uploaded; private 70% decides — do NOT tune to these):
| sub | OOF macro-F1 | public LB |
|-----|--------------|-----------|
| sub01 CatBoost (single, features_v1) | 0.82898 | **0.83026** |
| sub00 CatBoost baseline (raw21) | 0.82494 | 0.82809 |
| sub02 SoftVotingEnsemble | 0.81952 | 0.82478 |
| sub03 CatBoost seed-bag K=9 | 0.82415 | (not yet uploaded) |
Public LB ordering == OOF ordering → local CV is trustworthy; the soft-voting ensemble
genuinely hurts (confirmed on the public split too).

Findings:
- **Best = CatBoost (GBDT)** on `features_v1` = raw 21 + `eog_burst_missing` + band
  contrasts/sums (`delta-theta`, `theta-alpha`, `slowosc-delta`, `beta-delta`,
  `(delta+theta)-(alpha+beta)`, `eeg_total`). Greedy FE (full-power CatBoost) kept
  the **contrast** group (+0.004) and the missing flag; **rejected** softplus ratios,
  relative powers, autonomic combos, and EOG interactions (all hurt OOF). `eog_burst_index`
  is the #1 feature by importance.
- Hyperparameter tuning: defaults won (depth 6, lr 0.03, l2 3); class-balancing not needed.
- Blend search → CatBoost alone (weight 1.0); per-class threshold search → identity.
  CatBoost is already strongest & well-calibrated here; the ensemble/threshold add nothing
  on OOF, so they are not used in the primary submission (honest negative result).
- Weak class = **Deep (N3)**, F1 0.777 (confused with Light/REM).
- Reproducibility verified: CatBoost is deterministic (same run twice → bit-identical
  probabilities, best_iter 921=921); all seeds fixed; folds persisted.

Robustness (step 07 — added after public LB confirmed OOF; honest near-ceiling result):
- **Repeated-CV gate** `RepeatedStratifiedKFold(5×5)` = **0.82598 ± 0.00169** (5 repeats:
  .82898/.82389/.82509/.82627/.82567). So the canonical-fold + seed-42 number **0.82898 is
  ~+0.003 of combined fold+seed luck** above the true ~0.826. The std (≈0.0017) is now the
  accept/reject **noise floor**: no FE/ensemble tweak gaining < ~0.003–0.005 OOF is real.
- **Seed-bag (K=9)** per-seed OOF spans .822–.829 (std .00203); **seed 42 is the *best* of the
  9** by luck. The bag = **0.82415** ≈ the de-noised central estimate. So bagging does *not*
  raise OOF (it can't beat the lucky seed) — its payoff is **variance reduction / private-LB
  stability**, an honest textbook-bagging result, not a score chase.
- Error structure (OOF confusion): Deep errors are **diffuse** (≈evenly to Wake/Light/REM,
  both directions) and **72% of all errors are confident** (top1−top2 margin ≥0.15) → threshold
  / prior shifting has a tiny ceiling (consistent with the identity result). The ~0.83 wall is
  within-class overlap in the synthetic data, partly irreducible.
- Stacking the existing 4 OOF matrices (LR meta-learner) = 0.8243; any weight on RF/ET/GB drags
  CatBoost down → ensemble headroom needs a *new strong diverse* learner, not the weak trees.
- **Final private-LB pair recommendation:** `sub01` (single CatBoost, deterministic reference,
  higher-variance, best public 0.83026) + `sub03` (seed-bag, variance-reduced, honest ~0.826).
  `sub02` (soft-voting) is demoted — strictly worse on both OOF and public LB.

Submissions (`submissions/`, format `id,sleep_stage`, 5000 rows, ids 9000–13999):
- `sub01_CatBoost_GBDT.csv` — **primary pick**, OOF 0.82898, **public 0.83026**.
- `sub03_CatBoostSeedBagK9_GBDT.csv` — **variance-reduced safety** (K=9 seed-bag), OOF 0.82415.
  Recommended private-LB hedge in place of sub02.
- `sub02_SoftVotingEnsemble_CatBoost-RandomForest-ExtraTrees-GradientBoosting.csv`
  — diverse but strictly worse, OOF 0.81952, public 0.82478 (demoted).
- `sub00_catboost_baseline.csv` — first end-to-end sanity submission, public 0.82809.
