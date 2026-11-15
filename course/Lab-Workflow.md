# Run, compare, save, and explain an experiment

This is the shared operating guide for the local workbench. Each lesson supplies its own model, data contract, comparison, and assessment. Use the procedure that fits that lesson's scope.

## Start with the lesson's scope

| Scope | What you actually produce | What you must not claim |
|---|---|---|
| Concept activity | A design, calculation, or critique | That a model was trained |
| Numerical mechanism demonstration | Intermediate values, errors, gradients, or memory calculations | A deployable trained checkpoint |
| Pretrained inference | Outputs from existing weights and measured evaluation | Training a foundation model |
| Adaptation | Updated parameters or an adapter, with its required base model | A new model trained from random weights |
| Training from random weights | A tiny fitted model and its evaluation artifacts | Production quality from a teaching dataset |
| Retrieval experiment | An index, retrieved evidence, and optionally generated answers | Generator training during ordinary retrieval |

## Run a command in the correct place

The Practice buttons and guided terminal run the listed programs from the bundled `lab` folder, using the configured Python environment. Install packages through Package setup when required. Keep an existing working environment rather than recreating it for each lesson.

For an external Windows terminal, start in the extracted folder containing `server.py`, then enter `lab`. These examples assume the environment is `.venv` in the project root; substitute the interpreter you configured if different.

```powershell
Push-Location .\lab
& ..\.venv\Scripts\python.exe .\foundations.py linear --output outputs\linear-NEW
Pop-Location
```

```bat
pushd lab
..\.venv\Scripts\python.exe foundations.py linear --output outputs\linear-NEW
popd
```

Use the command belonging to your lesson. Replace `NEW` with a unique name. Practice buttons replace `{run}` automatically; a normal terminal does not. Experiments refuse to overwrite an existing nonempty output folder so earlier evidence survives.

## Decide what the comparison means

Write down the question before running. Keep the data, split, and seed fixed when changing one setting. A configured reference run is a comparison baseline; it is not necessarily a majority-class, mean-prediction, or random-policy baseline. Claim a built-in baseline only when the lesson and report actually include one. If a stronger baseline is a design activity, implement and identify it separately.

For ordinary train/validation/test experiments, fit preprocessing and parameters on training data, select settings using validation, and reserve the final test. Repeatedly choosing settings after viewing the test turns it into selection data.

Some lessons have different protocols. A fixed-graph experiment evaluates unseen labels on the same visible graph. A small adversarial generator is assessed on fresh simulator draws. A grid policy is assessed in the same environment. These results must not be renamed independent test sets or evidence of generalization to new populations. Follow the protocol stated in that lesson and report.

## Inspect actual evidence

For experiments that save `report.json`, read the dataset description, selected protocol, settings, metrics, limitations, and artifact paths. A training loss is an optimization signal; the lesson's evaluation explains useful performance. Model quality, latency, memory, and cost are separate measurements. Preserve failures and raw outputs.

Plots written by the program are results from that run. Diagrams inside lessons are teaching illustrations unless explicitly labeled as recorded results. A conceptual diagram is not an activation trace or a performance benchmark.

## Save, reload, and predict

Where supported, a successful run creates a model manifest, example input, learned state, and report. In Practice, choose **Load saved example**, edit values while respecting the lesson's shape and units, then choose **Predict new input**. The page remembers the successful folder by stable lesson ID.

The command-line alternative below applies to models supported by `predict_saved.py`. Replace both placeholders with your actual paths; run it from `lab` using the same interpreter as training.

```text
python predict_saved.py --model-dir outputs/MODEL-FOLDER --input outputs/MODEL-FOLDER/example-input.json --output outputs/prediction-NEW.json
```

A mechanism-only exercise may have no saved model. Language-model adapters require their matching base revision and tokenizer. The original scratch-model checkpoint uses its own loader. Only load model artifacts you created or trust, and preserve the architecture and preprocessing needed to interpret their state.

## Keep local prototypes and production claims distinct

The local workbench runs with your account's permissions. A shared service requires its own input validation, authentication, authorization, request limits, monitoring, versioning, rollback, and rejection behavior. Each lesson names the task-specific risks to monitor. A synthetic-data score does not establish real-world performance.

Start on the local central processing unit for these tiny exercises. Estimate memory and runtime before using optional graphics-processor instructions. Keep costs for data, training, inference, retained storage, and verification separate.

## Submit assessable work

Use the lesson's specific evidence checklist. Include your prediction before the experiment, actual measurements, a relevant comparison, one failure or limitation, and an explanation of whether the intended requirement was met. Include save/reload and new-input evidence only for exercises that implement them. Notes and progress remain associated with the browser profile and exact localhost address.

## Shared references

- [scikit-learn: common pitfalls](https://scikit-learn.org/1.7/common_pitfalls.html) — preprocessing, leakage, and consistent transformations.
- [PyTorch: saving and loading models](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html) — state dictionaries and reconstructing model instances.
- [Python: virtual environments](https://docs.python.org/3.12/library/venv.html) — project environments and interpreter paths.
