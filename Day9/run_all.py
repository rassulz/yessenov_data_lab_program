"""
run_all.py — one-command reproducible rerun of the whole pipeline.

It (1) regenerates the 8 notebooks from build_notebooks.py (single source of
truth) and (2) executes them in order, writing results back into the notebooks
so the outputs/plots are preserved for the defense.

Usage (must use the Anaconda interpreter that has sklearn/catboost):
    C:/ProgramData/anaconda3/python.exe run_all.py

Determinism: fixed seeds live in the notebooks; PYTHONHASHSEED is pinned here so
the kernel inherits it.
"""
import os
import sys
import subprocess

os.environ["PYTHONHASHSEED"] = "0"

ORDER = [
    "01_eda.ipynb",
    "02_baseline.ipynb",
    "03_features.ipynb",
    "04_models.ipynb",
    "05_tune_blend.ipynb",
    "06_submit.ipynb",
    "07_robustness.ipynb",
    "08_catboost_ensemble.ipynb",
    "09_mlp_ensemble.ipynb",
    "10_svm_ensemble.ipynb",
    "11_multifamily_ensemble.ipynb",
]


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    os.chdir(here)

    # 1) regenerate notebooks from source
    print(">>> regenerating notebooks from build_notebooks.py")
    subprocess.run([sys.executable, "build_notebooks.py"], check=True)

    # 2) execute each notebook in order with the anaconda3 kernel
    for nb in ORDER:
        print(f"\n>>> executing {nb}")
        subprocess.run(
            [sys.executable, "-m", "jupyter", "nbconvert",
             "--to", "notebook", "--execute", "--inplace",
             "--ExecutePreprocessor.timeout=3600",
             "--ExecutePreprocessor.kernel_name=anaconda3",
             nb],
            check=True,
        )
    print("\n>>> pipeline complete. See submissions/ and results_log.csv")


if __name__ == "__main__":
    main()
