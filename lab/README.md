> **CURRENT WINDOWS CPU UPDATE — Python 3.12.** Read [SECURITY-TESTING.md](../SECURITY-TESTING.md) first. Use a new folder and its own environment on localhost port 8766. Earlier platform and dependency recipes below are historical; use the hashed Windows lock for this candidate. Pretrained language/vision and GPU compatibility are not yet certified.

# SML Lab Zero to Hero

Start with **SML-Lab-Zero-to-Hero.md**. It contains the complete course, setup
commands, expected observations, checkpoint questions and troubleshooting.

## Contents

- `lab.py`: diagnostics, model revision locking, tokenization, inference,
  strict evaluation, LoRA training and a transparent retrieval baseline.
- `scratch_lm.py`: train a tiny byte language model from random weights;
  choose Transformer, toy SSM, toy MoE or hybrid architecture.
- `mechanisms.py`: inspect sparse routing, recurrence, quantization and distillation.
- `vision_lab.py`: optional small vision-language inference.
- `compare_reports.py`: compare evaluation results on identical examples.
- `make_data.py` and `data/`: original synthetic classroom tickets and notes.
- `tests/`: meaningful checks for causal masking, padding, data boundaries,
  routing, state carryover, schemas and retrieval.
- `model-lock.json`: exact model repository revisions used by the loader.
- `requirements.txt`: direct package versions for the teaching environment.
- `environment-verified.txt`: complete CPU environment used during verification;
  this is a record, not a cross-platform installation command.
- `EXPERIMENT-LOG.md`: template for your own measurements and support requests.
- `VERIFICATION.md` and `reference-results/`: observed checks and raw results.

The bundle contains no pretrained weights, personal data, access tokens or hosted
generation API dependency. First use downloads public model files into the normal
Hugging Face cache. CPU inference/training then runs locally.

## Quick command sequence after setup

```bash
python lab.py doctor
python lab.py lock
python lab.py infer --prompt "Explain overfitting in one sentence."
python -m unittest discover -s tests -v
python lab.py eval --output outputs/baseline-validation.json
python lab.py train --steps 60 --output outputs/adapter-60
python lab.py eval --adapter outputs/adapter-60 --output outputs/adapter-validation.json
python compare_reports.py outputs/baseline-validation.json outputs/adapter-validation.json
```

Use the final test split only after choosing settings on validation. This is a
small educational dataset; its scores do not establish production reliability.
See the workbook for the CPU/GPU distinction and complete setup instructions.
