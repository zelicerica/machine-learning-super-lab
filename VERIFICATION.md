# Modern language-engineering expansion — 2026-11-15

The course now includes **58 lessons, 16 reference chapters, and 87 Mermaid diagram instances**. Lessons 43–58 form a progressive path through attention, architecture, adaptation, preferences and rewards, evaluation, and serving. Existing lesson identifiers and the learner-output conventions remain unchanged. The research timeline connects original papers to specific lessons and labels simplified mechanisms and unimplemented reading extensions.

## What passed in this revision

- **52 automated Python tests:** 44 course/server tests plus eight original language-model tests. New checks cover causality, incremental cache equivalence, head-sharing cache sizes, packed four-bit storage, adapter gradients, save/reload, split boundaries, preference gradients, clipping and sequence ratios, pass@k, confidence intervals, judge abstention, real streaming HTTP, and bounded requests.
- All **14 new experiment types** executed their baseline commands. Thirteen additional runs changed a setting, and the optional real local judge ran against its pinned checkpoint: **28 completed experiment runs** in total.
- Independently reloaded and predicted with **26 saved model outputs** through `predict_saved.py`. Reports and plots are retained in `lab/reference-results/modern`, without weight files. The expert-routing reports were rerun after fixing their measured token-count scope.
- The actual streaming service handled twelve client-timed requests per baseline and changed-setting run. Cached and uncached token outputs also match in the HTTP unit test. Invalid tokens, empty/oversized prompts and invalid generation counts are rejected.
- The frontend DOM check rendered the content and controls for **all 74 lesson/reference routes**, parsed all **87 diagram instances with the bundled Mermaid parser**, checked internal links and image paths, exercised the modern-track filter and practice controls, and verified retention of earlier progress and notes. This is DOM simulation, not a visual browser layout test.
- JavaScript syntax, course generation and Python compilation passed. The optional GPU script correctly refused CPU-only execution with an explicit explanation.

## What the results do and do not establish

Execution success does not guarantee model quality. The reference results preserve failed adaptation, reward-model generalization, policy-training regressions, and all twelve invalid responses from the tiny-model judge. The fixed-format toy policy and simplified latent/depth attention experiments are not reproductions of full frontier systems. Public benchmarks are covered as benchmark literacy; original arithmetic mini-suite scores are not public benchmark scores.

Validation used Linux, Python 3.12, PyTorch 2.6.0+cpu and the existing pinned CPU dependencies. The new track did not require a paid rental. Native Windows/macOS execution, GPU QLoRA, cloud installation and new browser visual/responsive checks were **not completed**. The connected browser displayed the user's older 34-lesson installation; the separate cloud browser could not reach the new local server, and the shell's browser launch was restricted. Those observations are not evidence that the new browser layout passed. Earlier browser evidence below applies only to the unchanged earlier content and interface.

## Repeat the checks

From the repository root, use the configured course Python:

```powershell
# Windows PowerShell
& .\.venv\Scripts\python.exe -m unittest discover -s tests -v
& .\.venv\Scripts\python.exe -m unittest discover -s lab/tests -v
& .\.venv\Scripts\python.exe build_course.py
```

The optional developer DOM check is `node tests/check_course_dom.cjs`; it requires jsdom 26.1.0 in a separate developer environment. Set `SML_JSDOM_PATH` to its absolute module directory if it is not on Node's module path. Node and jsdom are not required to use the downloaded course. A real browser check should additionally cover layout, keyboard interaction, charts, a Practice run, and saved-model prediction on the learner's platform.

---

# Expanded course verification — 2026-11-14

## Public repository preparation — 2026-11-14

The README now includes separate Windows PowerShell/Command Prompt, Apple silicon macOS, and Ubuntu 24.04 setup instructions, plus public companion learning links. The shell launcher selects Python 3.12 and forwards server arguments. Its syntax and `--help` path passed on Linux, and all 10 server tests passed after the launcher change. Full native Windows/macOS installs remain untested; Intel macOS limitations are documented. README file links resolve. Internal workspace prefixes in bundled reference reports have been replaced with `<verification-workspace>`; recorded measurements are unchanged.

## Synthetic-data addition — current release

The course now has **42 lessons, 12 references, and 58 Mermaid diagram instances**. Lesson 42 adds synthetic-data education and a runnable generation/audit/training comparison. Earlier lesson identifiers and the existing learner-output directories are preserved.

- Executed the four guided experiments: full coverage, narrow coverage, deliberately reversed labels, and increased volume. Also executed the optional 40% label-flip-probability variant. All five saved reports, plots, data cards, row files, model artifacts, and verified reloaded predictions.
- **39 automated tests passed:** 31 current server/foundation/learning/synthetic-data tests and eight original language-model tests. Six new checks cover independent random streams, label corruption without feature mutation, audit rejection, exclusion of audit metadata from model inputs, training-only scaling, reload/new-input contracts, output preservation, and bounded browser arguments.
- Verified the new lesson and reference links, commands, generated course data, and JavaScript syntax. The two new Mermaid diagrams rendered in the browser.
- Browser Practice completed an actual synthetic-data training comparison, loaded the saved numerical example, and ran predictions with changed inputs using all three saved models.
- Actual reports and plots are in `lab/reference-results/synthetic-data`; they include the counterexample that narrow-coverage data still scored well under this simple simulator and the poor result from reversed labels.

All data partitions are simulated. This is a workflow exercise, not a real-world utility benchmark, privacy mechanism, or reproduction of recursive model collapse. Windows instructions were retained and reviewed, but execution here used Linux. No additional package family, model download, or Vast.ai rental was needed.

## Training and retrieval expansion — preceding release

This release contains **41 lessons, 11 reference chapters, and 54 Mermaid diagram instances**. It preserves the earlier 34 lesson identifiers and adds seven lessons in the Build and retrieve track: one Concept lesson and six Runnable implementation labs. Earlier verification records below describe prior releases.

- Added decision education for pretrained reuse, fine-tuning, a frozen encoder, continued pretraining, random initialization, and retrieval. A checkpoint's training history is distinct from its architecture and deployment format.
- Executed all five new neural commands and their changed-setting experiments: deep classifier, recurrent language model, masked-token encoder, encoder–decoder, and the three-branch transfer comparison. Every run saved its artifacts, reloaded weights, and made predictions. Synthetic operations examples are split by underlying action/service/region group so paraphrases do not cross partitions; vocabulary fits training data only.
- Executed retrieval baseline, small-chunk, and stale-source experiments. Serialized indexes reloaded with the same probe rankings. Retrieval-only output is explicitly evidence and a prompt, not a generated answer.
- Executed real generation with the pinned SmolLM2-135M-Instruct checkpoint, both without and with retrieved context. The full generator parameter-byte hash stayed unchanged. The model invented an unrelated answer without notes, then produced the correct 48-hour fact with notes but omitted the required citation. This is a partial quality result, not universal correctness.
- **33 automated checks passed**: 25 current foundation/learning/server checks plus eight original language-model checks. These cover gradient agreement, causal prediction, hidden masked targets, padding invariance, split isolation, frozen encoder updates, retrieval evidence versus source hits, save/reload, and local runner boundaries. Browser testing exposed a scalar example serialization bug; the fix and regression check preserve both text and numerical examples as valid JavaScript Object Notation.
- Parsed every guided command through the server's argument validation; checked source syntax, generated lesson/reference links, unique identifiers, and embedded image paths.
- Browser checks verified the new track, rendered new architecture/decision diagrams, trained the deep classifier through Practice, loaded its saved example, and predicted on changed input. The retrieval Practice run also completed, loaded its saved question, and retrieved the Cedar backup note for a changed query without rebuilding the index. New diagrams rendered successfully; unchanged diagram coverage is recorded in the earlier checks below.
- Actual reports and plots for 14 teaching experiments are included under `lab/reference-results/build-and-retrieve`. They are separated from learner output and exclude downloaded and trained model weights.

**Limits:** tests ran on Linux with the existing Python 3.12 / PyTorch 2.6.0 central-processing-unit environment. Windows instructions were reviewed, not executed on Windows. No Vast.ai rental was needed or made; graphics-processor execution was not tested. These tiny synthetic datasets teach complete engineering workflows and do not establish production capability. Larger research reproductions and diffusion plans remain Concept material; the new compact recurrent, masked encoder, and encoder–decoder implementations are runnable.

## Atlas revision checks — 2026-11-14

This later revision changes reference content, an illustration, and sidebar navigation. It does not change the model-training programs.

- Rebuilt all 34 lessons and eight reference chapters; existing lesson identifiers remain unique and unchanged.
- Checked that all lesson/reference links in generated course content resolve and all embedded image paths exist.
- Verified the downloadable workbook and its copy in the lab folder contain the same revised content.
- Verified JavaScript syntax and generated 43 total Mermaid diagram instances. The new machine-learning relationship diagram rendered successfully with the bundled browser renderer; the earlier 42-diagram verification below remains the evidence for unchanged diagrams.
- Visually inspected the new vector/raster infographic and the rendered relationship diagram. Both atlas pages and the sidebar navigation displayed correctly in the local preview.
- Followed the language atlas's count-based model link to the existing statistical language-model lesson. No training job was run as part of this content revision.
- Included the substantive findings and primary sources in course/Atlas-Review.md. New research entries are Concept plans, not additional claimed implementations.

## Earlier implementation verification

Validated with Python 3.12.14, PyTorch 2.6.0+cpu, NumPy 2.2.6, scikit-learn 1.7.2, SciPy 1.16.3, and Matplotlib 3.10.7 on Linux. The browser checks used desktop Chrome.

## Completed checks

- All 18 new foundation implementations completed training/fitting or counting, evaluation, artifact serialization, reload, and example inference. Reports and generated plot files exist.
- All 18 changed-setting commands supplied in the lessons also completed successfully, including graph edge removal, the alternate adversarial training run, and the shorter reinforcement-learning run.
- All guided commands parsed through the server's whitelist. They remain Python argument arrays, not shell execution.
- Fourteen current server/foundation tests and eight original model tests passed. Checks cover actual execution, cancellation, error propagation, local request boundaries, output-path restrictions, leakage-free split and scaler contracts, terminal reward handling, known-coefficient prediction after reload, and normalized counting-model inference. Original tests cover causal masking, padding, expert routing, recurrence, and evaluation schema.
- JavaScript syntax check passed. Every lesson has a concept or implementation tag; all embedded lesson illustration paths exist.
- All 42 Mermaid diagrams rendered successfully using the bundled renderer with strict security mode. This includes algorithms, model-selection flows, build lifecycles, and the cloud architecture.
- Browser checks exercised the Foundations filter; linear-regression training; saved-model example loading; changed-input inference without retraining; actual generated plot display; comprehension feedback; and evidence-note persistence after navigation and reload.
- Browser inference on feature rows [10, 2, 1] and [20, 2, 1] returned approximately 38.9544 and 58.9022 from the noisy fitted regression model. These were actual program outputs.
- Six new infographics were generated and visually inspected. Source vector files and raster versions are included with the course.

## Reproducible reference results

See lab/reference-results/foundations for the measured reports and plots. They are intentionally separate from lab/outputs, where the learner's own experiments appear. Each report includes settings, dataset description, split, metrics, and limitations. The following selected results describe the baseline teaching runs, not production capability.

| Build | Selected measurement |
|---|---|
| Linear regression | Mean absolute error 6.60 versus training-mean baseline 49.08 |
| Convolutional digit classifier | Validation accuracy 98.33 percent on bundled small digits |
| Dense digit classifier | Validation accuracy 94.99 percent |
| Principal component projection | Reconstruction mean squared error 0.01137 with 16 dimensions |
| Autoencoder | Reconstruction mean squared error 0.02845 versus mean-image baseline 0.07349 |
| Statistical word model | Validation perplexity 4.26 versus uniform baseline 15 |
| Grid action-value learner | Greedy success rate 1.0 on the same deterministic training environment |

## Verification limits

Windows launcher and PowerShell/Command Prompt instructions were reviewed, but were not executed on Windows here. No Vast.ai instance was rented. Graphics-processor acceleration, fresh cloud installation, and cloud tunnels were documented from official guidance but not executed. macOS and narrow-screen browser interaction were not tested. The existing one-click installer and browser vision upload were not repeated in this expansion; package installation used the matching local environment and original vision evidence remains in VERIFICATION-ORIGINAL.md and lab/VERIFICATION.md.

At the time of the earlier foundation release, the optional recurrent, encoder, encoder–decoder, and diffusion project plans were Concept material. The current expansion adds compact runnable versions of the first three; larger reproductions and diffusion remain Concept plans. Some existing language lessons demonstrate numerical mechanisms or pretrained inference; their scope labels identify that boundary.

The release excludes virtual environments, downloaded model weights, executable developer model artifacts, learner input files, run logs, test-browser progress, and internal preview pages. It includes source code, course books, instructional assets, a local Mermaid renderer with its license, and explicitly labeled reference measurements.
