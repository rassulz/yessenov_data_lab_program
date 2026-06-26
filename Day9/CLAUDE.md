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
Pipeline = 12 notebooks (`01_eda` → `12_textbook_ensembles`), one-command rerun via
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
| 08 ensemble | CatBoost diversity ens. (sub04) | features_v1 | 0.82324 | .850/.847/.769/.827 |
| 09 mlp | **MLP seed-bag (K=3)** | features_v1 | **0.83677** | .863/.854/**.789**/.842 |
| 09 mlp | CatBoost+MLP ensemble (sub05) | features_v1 | 0.83711 | .862/.856/.790/.841 |
| 10 svm | **SVM-RBF bag (grid top-3)** | features_v1 | **0.83331** | .863/.850/.785/.835 |
| 10 svm | **CatBoost+SVM ensemble (sub06)** | features_v1 | **0.83269** | .859/.850/.785/.838 |
| 11 multifam | QDA (generative, most decorrelated) | features_v1 | 0.80836 | .846/.827/.758/.802 |
| 11 multifam | GMM(k=2) per-class Bayes | features_v1 | 0.78867 | .828/.804/.740/.782 |
| 11 multifam | Nystroem+LogReg | features_v1 | 0.82304 | .854/.845/.765/.829 |
| 11 multifam | Cat+SVM+QDA+GMM stack (honest) | features_v1 | 0.83374 | ≈ sub06 (within noise) |
| 12 textbook | StackingClassifier (RF+ET+HistGB, tree-only) | features_v1 | 0.81707 | the tree-only wall |
| 12 textbook | **Géron-ch7 stack (Cat+SVM+QDA+GMM, sub08)** | features_v1 | **0.83343** | .864/.849/**.783**/.837 |
| 12 textbook | stack **paired-CV (5×3)** | features_v1 | **0.83479 ± 0.00080** | +0.0125 vs Cat (3/3); +0.0014 vs sub06 (3/3) |

Public LB (uploaded; private 70% decides — do NOT tune to these):
| sub | OOF macro-F1 | public LB |
|-----|--------------|-----------|
| **sub06 CatBoost+SVM ensemble** | 0.83269 | **0.84157** |
| sub01 CatBoost (single, features_v1) | 0.82898 | 0.83026 |
| sub00 CatBoost baseline (raw21) | 0.82494 | 0.82809 |
| sub03 CatBoost seed-bag K=9 | 0.82415 | 0.82820 |
| sub02 SoftVotingEnsemble | 0.81952 | 0.82478 |
| sub04 CatBoost diversity ensemble | 0.82324 | (not yet uploaded) |
**sub06 (Cat+SVM) is the new best public 0.84157** — cross-family blending confirmed on the LB
(public even > its OOF 0.83269; encouraging but private 70% decides, keep selecting on OOF/paired-CV).
Public LB ordering == OOF ordering → local CV is trustworthy. sub03 (seed-bag) public 0.82820 <
sub01 0.83026: seed 42 is lucky on the public split too; the bag's value is private-LB variance
reduction, not a higher public number. The soft-voting ensemble genuinely hurts (confirmed on public).

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
- **CatBoost-only diversity ensemble (step 08, sub04)** — 6 strong members (depth 5/6/7/8 +
  `MultiClassOneVsAll` head + `rsm=0.7`), equal-weight averaged. Members .8215–.8290; ensemble
  **0.82324**, i.e. −0.00574 vs the single deployed model (beyond the ±0.0017 gate → a real,
  not-noise *decrease*). Mean pairwise disagreement only **3.7%**: same algorithm on the same
  features makes correlated errors, so averaging just pulls toward the (weaker-than-the-lucky-
  anchor) member mean. Confirmed negative — `sub04` is a documented diversity hedge, **not** a
  final pick. Same lesson as the seed-bag: averaging removes the anchor's luck, lands at honest center.
Breakthrough (step 09 — the ~0.83 wall was a SINGLE-FAMILY limit, not a data limit):
- Every model through step 08 is a **tree** (CatBoost/RF/ET/GB) → shared axis-aligned bias →
  every tree-ensemble (sub02, sub04) made the *same* errors and could not move. Multi-agent
  investigation tested non-tree families on the same folds: **MLP (neural net) breaks the wall.**
- **MLP(256,128), α=1e-3, seed-bag K=3** (per-fold StandardScaler + train-median EOG impute):
  OOF **0.83677**, and it finally lifts the weak **Deep** class **.777 → .789**. Error-correlation
  with CatBoost **0.68** (vs ~0.9 tree-vs-tree) = genuinely diverse. SVM-RBF (0.8315) and
  Nystroem+LogReg (0.8309) also edged CatBoost; KNN/LogReg/QDA were dead ends.
- **Cross-family ensemble `0.5·CatBoost + 0.5·MLP` (sub05)** = OOF **0.83711**. Weights FIXED
  a-priori — an OOF weight search inflated the score by ~0.006 of non-transferring noise (caught
  by an adversarial re-test), so no weight tuning is used.
- **Honest paired `RepeatedStratifiedKFold(5×5, seed=2026)`, leak-free both arms:** ensemble
  **0.83669 ± 0.00099** vs single CatBoost(900 fixed iters) 0.82255 ± 0.00105, paired delta
  **+0.01414**, positive in **5/5** repeats. (CatBoost at fixed iters is mildly handicapped vs its
  early-stopped 0.826 gate; vs the fair gate the honest gain is **≈ +0.010** — still far past the
  ±0.0017 noise floor.) Note the MLP **alone** (0.83677) ≈ the ensemble (0.83711): the MLP is the
  driver; the blend is a model-risk hedge that also keeps Deep at .790.
- Confirmed dead ends (do not retry): more CatBoost FE, threshold/calibration, EOG regime split,
  Deep-class specialist, and any tree-only ensemble. Remaining ceiling = Deep↔Light/REM overlap in
  feature space (needs new signal, not feature recombination).
- **SVM ceiling probes (post step 11, also dead ends):** (1) SVM-specific FE — every extra group on
  top of features_v1 *lowers* SVM OOF (relpower 0.8297, ratio 0.8288, +autonomic 0.8278, all-groups
  0.8233 vs v1 0.8299); the smooth kernel dislikes the noise dims just like CatBoost. (2) Richer/diverse
  SVM bag — no untested (C, gamma) beats the current best single C=3/g=0.02 (0.83305): C=8/g=0.01=0.8322,
  g=0.03=0.8301, g=0.05=0.8289; random 18/28-feature subspaces are weak (0.807–0.829). The top-3 bag
  (0.83331) is already optimal → the SVM member is at its ceiling.
Chosen direction (step 10 — SVM-RBF, MLP set aside by preference):
- Per the user's choice we do **not** ship the MLP; the non-tree partner is an **SVM-RBF**
  (`09_mlp_ensemble` stays on disk as an explored-but-unselected result). Same cross-family recipe.
- **Grid-CV over (C, gamma)** on the shared folds (predict-only scoring): best single SVM-RBF is
  `C=3, gamma=0.02` = **0.83305**; top-3 = (3,0.02)/(5,0.02)/(2,scale). **Bagged top-3** (calibrated)
  OOF **0.83331**, Deep .785. Error-correlation with CatBoost **0.759** (vs MLP's 0.68 → less diverse,
  so a smaller blend payoff — as predicted).
- **Cross-family ensemble `0.5·CatBoost + 0.5·SVM` (sub06)** = OOF **0.83269** (+0.0037 vs CatBoost),
  weights fixed a-priori. Honest paired `RepeatedStratifiedKFold(5×3, seed=2026)`: ensemble
  **0.83384 ± 0.00062** vs single CatBoost(900 fixed iters) 0.82229 ± 0.00128, paired delta
  **+0.01155**, positive in **3/3** (vs the fair 0.826 gate, honest gain ≈ +0.008).
- Honest caveat: the SVM bag **alone** (0.83331) ≈ the Cat+SVM ensemble (0.83269) on seed-42 — SVM is
  stronger than CatBoost and they correlate (0.759), so the blend wins on **variance** (paired std
  0.0006) and two-family risk-hedging, not on a higher mean. (For reference, the unshipped MLP blend
  was higher: 0.83711.)
- **Final private-LB pair recommendation:** `sub06` (CatBoost+SVM cross-family, the chosen pick) as
  **primary** (now confirmed best **public 0.84157**), + `sub01` (single deterministic CatBoost,
  validated reference, public 0.83026) as the hedge. `sub02`/`sub04` demoted (tree-only).

Multi-family / near-ceiling (step 11 — honest negative + a real diversity finding):
- Added non-tree, non-MLP families on the shared folds: **QDA, LDA, GaussianNB, per-class GMM(k=2,3),
  Nystroem+LogReg** (generic scaled-OOF runner + a guarded per-class-GMM Bayes rule that earlier crashed).
- **Eligibility = strong AND decorrelated** (OOF ≥ anchor−2σ ≈ 0.8256, errcorr<0.75 vs Cat, <0.85 vs SVM):
  **none qualified.** The generative families are *beautifully* decorrelated — QDA errcorr **0.61/0.65**,
  GMM(2) **0.52/0.56** (better than SVM's 0.76 and even MLP's 0.68) — but individually too weak
  (0.79–0.81). Nystroem is strong-ish (0.823) but too correlated with the SVM (both RBF). So the fixed
  multi-family blend degenerated to Cat+SVM (paired delta 0.0) → **no sub07; sub06 stays primary.**
- **Stacking probe** (honest nested-CV `cross_val_predict`, LogReg meta) — the one thing the eligibility
  gate excluded: a meta-learner *can* use a weak-but-orthogonal member. Best honest stack
  `cat+svm+qda+gmm2` (C=0.3) = **0.83374** vs sub06 0.83269 = **+0.001, WITHIN the 0.0017 noise floor**.
  Adding Nystroem *hurts* (correlated with SVM). Hand-weighted small blends look higher (~0.835) but that
  is in-sample weight tuning → not honest/shippable.
- **Verdict: ~0.834 is the honest ceiling for the SVM/non-MLP path.** The generative-diversity finding is
  real but too small to clear noise. For reference, the excluded MLP blend (0.83711) is still ~+0.003 above
  any honest multi-family stack — the largest single lever left, if MLP is ever reconsidered.

Textbook ensembles — Géron ch.7 variant (step 12, `12_textbook_ensembles.ipynb`, an alternate solution
built straight from the course slide deck `glava7`, "the same but better"):
- One self-contained notebook that walks **every** ch.7 method on the real task, honestly on the shared
  folds (macro-F1): voting (hard/soft) → bagging/pasting + OOB → random patches/subspaces → RF/Extra-Trees
  + importances → AdaBoost/GBRT → **stacking**. Reuses the saved member OOF (cat/svm/qda/gmm2/rf/et/gb/mlp)
  as leak-free stacking inputs; trains the textbook learners live. Each conclusion is **computed from the
  numbers** (no hardcoded narrative).
- Honest demonstrations that reproduce the project's whole arc: **soft voting (.753) fell *below* hard
  (.788)** because GaussianNB's miscalibrated near-0/1 probabilities poison the average (slide-6 caveat,
  live); the 5-learner vote sits below its best member (rf .793) — count loses to diversity-of-strength,
  exactly why `sub02` hurt. OOB macro-F1 (.785) ≈ 5-fold bagging (.787) — the free estimate is trustworthy.
  **Random subspaces (.800) > plain bagging (.787)** — feature subsampling is what turns bagging into a RF
  (subspaces ≈ RF .792). AdaBoost .767 / GBRT-earlystop .816 / HistGB .818 < CatBoost .829; **tree-only
  `StackingClassifier` = .817** = the wall (a blender can't invent signal no member has).
- **Climax = cross-family stacking** (the "better" lever): LogReg blender, out-of-fold via
  `cross_val_predict`, members **fixed a-priori** from the step-11 eligibility analysis (strong CatBoost+SVM
  + most-decorrelated generatives QDA/GMM) — **not** picked by OOF-ranking the layers (that printout is
  diagnostic only). Deployed `cat+svm+qda+gmm2` = OOF **0.83343** (Deep .783), C-stable across [0.1,3.0].
- **Honest paired `RepeatedStratifiedKFold(5×3, seed=2026)`**, all arms refit leak-free on fresh folds:
  STACK **0.83479 ± 0.00080** vs single CatBoost 0.82229 = **+0.01250, positive 5/5→3/3 repeats** (clears
  the ±0.0017 floor); vs `sub06` Cat+SVM 0.83337 = **+0.00142, 3/3** but *within* the noise floor → a
  **consistent-but-within-noise tie**, reported as such (no overclaim). MLP what-if (`+mlp` → 0.83698,
  +0.0035) shown but **not shipped** (MLP set aside by preference) — still the one real remaining lever.
- `sub08` shipped as a **principled diversity hedge** (textbook-faithful construction, differs from sub06
  on 159/5000 rows), not a claimed score win. Strong "best approach"-defense artifact: clear, justified,
  reproducible, and honest about the ceiling.

Submissions (`submissions/`, format `id,sleep_stage`, 5000 rows, ids 9000–13999):
- `sub08_Geron_Ch7_StackingEnsemble.csv` — **Géron ch.7 textbook variant** (cross-family stack:
  CatBoost+SVM+QDA+GMM via out-of-fold LogReg blender), OOF **0.83343**, paired-CV +0.0125 vs CatBoost
  (3/3) / +0.0014 vs sub06 (within noise), Deep .783. Diversity hedge; honest tie-at-ceiling. (public LB TBD)
- `sub06_CatBoostPlusSVM_Ensemble.csv` — **chosen primary pick** (cross-family: CatBoost + SVM-RBF
  grid-bag, fixed 0.5/0.5), OOF **0.83269**, paired-CV +0.0116 vs CatBoost, Deep .785. (public LB TBD)
- `sub01_CatBoost_GBDT.csv` — deterministic CatBoost **reference hedge**, OOF 0.82898, **public 0.83026**.
- `sub05_CatBoostPlusMLP_Ensemble.csv` — explored, **not selected** (MLP set aside by preference);
  higher OOF 0.83711 / Deep .790, kept on disk for reference.
- `sub03_CatBoostSeedBagK9_GBDT.csv` — **variance-reduced safety** (K=9 seed-bag), OOF 0.82415.
  Recommended private-LB hedge in place of sub02.
- `sub02_SoftVotingEnsemble_CatBoost-RandomForest-ExtraTrees-GradientBoosting.csv`
  — diverse but strictly worse, OOF 0.81952, public 0.82478 (demoted).
- `sub00_catboost_baseline.csv` — first end-to-end sanity submission, public 0.82809.
