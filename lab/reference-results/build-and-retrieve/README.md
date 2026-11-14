# Measured teaching results: build and retrieve

These are actual development runs on Linux using Python 3.12 and PyTorch 2.6.0 on a central processing unit. Reports and plots are preserved without trained weight files. They are examples to inspect, not prefilled learner submissions or production benchmarks. Your own runs appear under `lab/outputs` and may differ by hardware, library version, or seed.

## Selected baseline measurements

| Experiment | Measured result | Interpretation |
|---|---|---|
| Deep nonlinear classifier | 98.33% validation accuracy; 1,218 parameters | Learned a synthetic curved decision boundary |
| Linear comparison | 91.67% validation accuracy; 6 parameters | A smaller baseline can still be useful |
| Recurrent word model | Validation perplexity 2.64 versus uniform baseline 24 | Improvement on a tiny templated vocabulary, not general language ability |
| Masked encoder | 58.33% masked-word accuracy | Predicts hidden words with bidirectional context |
| Encoder–decoder | 85% free-running exact sequence match | Evaluates complete generated outputs, not only teacher-forced loss |
| Fresh classifier branch | 100% validation accuracy | This simple downstream task did not require pretraining to win |
| Pretrained, then fine-tuned branch | 98.33% validation accuracy | Pretraining is not guaranteed to help |
| Pretrained, frozen encoder branch | 46.67% validation accuracy; 132 trainable parameters | Smaller adaptation capacity; encoder bytes remained unchanged |
| Default retrieval | Source recall and answer-evidence recall both 1.0 | Three answerable validation fixtures; not a broad benchmark |
| Eight-word chunks, top one result | Source recall 1.0, answer-evidence recall 0.3333 | Correct document identity did not guarantee answer-bearing text |

The transfer branches use the same 16 labeled examples and 80 adaptation updates. The pretrained branches also incur 300 earlier masked-token training updates; their total compute is not equal to the fresh branch. The four-label ablation is in `transfer-few`. Other shorter/depth/stale-source variants are kept alongside their baselines.

## Actual retrieval and generation comparison

The `rag-generation` report records model `HuggingFaceTB/SmolLM2-135M-Instruct` at revision `12fd25f77366fa6b3b4b768ec3050bf629380bac`.

- Without the fictional operations notes, it invented an unrelated answer about a window and wingspan.
- With the current source, it produced: “The current Orion restore window is 48 hours.”
- That fact matches the supplied source, but the requested bracketed citation was absent. Citation compliance failed.
- The generator's full parameter-byte hash remained unchanged across both calls. Context changed the input; it did not train the generator.

Preserve this partial success when grading. Correct answer text, citation presence, citation support, and appropriate abstention are separate questions. The report's manual review fields deliberately remain blank for the learner to assess. First-use timing includes downloading/loading the checkpoint and is not an isolated inference benchmark.

## Inspect and reproduce

Open each `report.json` for exact settings, splits, measured metrics, limitations, and reload evidence. Open its plot for loss or retrieval changes. Use the corresponding website Practice commands to reproduce runs in fresh output directories. After selecting settings with validation data, use the separate test partition once; do not tune to these reference scores or repeatedly inspect the final test while selecting settings.
