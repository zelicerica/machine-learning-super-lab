# Actual synthetic-data teaching runs

These reports were produced on November 14, 2026 using the included `synthetic_data.py`, seed 42, and the existing Linux Python 3.12 / scikit-learn 1.7.2 environment. All seed, generated, validation, and test rows are fictional simulator outputs. Metrics below use the same 400 simulated validation rows in every comparison. They do not establish performance on real services.

| Generated-data configuration | Seed-only balanced accuracy | Generated-only balanced accuracy | Mixed balanced accuracy |
|---|---:|---:|---:|
| 400 rows, full coverage, correct labels | 84.10% | 96.52% | 96.52% |
| 400 rows, narrow coverage | 84.10% | 96.54% | 95.57% |
| 400 rows, every generated label reversed | 84.10% | 3.48% | 6.59% |
| 400 rows, 40% label-flip probability | 84.10% | 89.50% | 96.22% |
| 2,000 rows, full coverage, correct labels | 84.10% | 96.74% | 96.74% |

What these runs teach:

- Data quality matters: deliberately reversing labels made generated-only and mixed training much worse than the starting baseline.
- Narrow coverage still scored well here. A simple model can extrapolate under the simulator's simple rule. The coverage plot and the missing high-error region remain important evidence; the score does not establish real-world robustness.
- Increasing volume had only a modest effect in this run. It also changed the mix and total training rows, so it is not an equal-compute comparison.
- The 40% setting is a flip probability, not a claim that exactly 40% of this finite sample was flipped. Inspect the measured fraction in the data card.

Each directory contains the actual `report.json`, `data-card.json`, and plots. Trained models and generated row files are omitted from these reference results; run the Practice commands to create your own complete artifacts in `lab/outputs`. Each reference run passed model reload verification. The data-card hashes identify its original generated files; they are reproducibility metadata, not a claim those files are included here.

Do not tune to this table. Choose settings on your own validation exercise, then assess the selected configuration with `--split test`. A separate real-world dataset is still required before making real-world utility claims.
