"""Real-world project specifications paired with each executable teaching model."""
PROJECTS={
'linear':('Batch-job duration estimator','One completed job per row: record count, transformation count, compression setting, and observed duration. Record units and exclude information available only after the prediction time.','A nonnegative duration estimate; route out-of-range workloads to a conservative fallback.','Absolute error by job size; feature-range shifts; late-job rate.'),
'logistic':('Operations review-queue ranker','One event per row with features observable at event time and a later reviewed outcome. Split by time and keep related events together in a real dataset.','A review score with a validation-selected threshold; preserve human review and record false-alert costs.','Precision, recall, calibration, review volume, and label delay.'),
'tree':('Inspectable equipment-fault triage','One machine observation per row with sensor values and a confirmed fault label. Split by machine and time to avoid near-duplicate leakage.','A fault category plus the sequence of thresholds; uncertain cases go to a technician.','Missed faults, false alarms per machine, and newly appearing operating regimes.'),
'ensemble':('Robust tabular event classifier','The same feature and label contract for all candidate models. Freeze train, validation, and test membership before comparing tree ensembles.','A class prediction from the chosen model; retain the simpler baseline and rollback artifact.','Held-out class recall, latency, feature availability, and performance under changed event mix.'),
'neighbors':('Similarity-based sensor-state classification','Two normalized sensor measurements and an expert state label in the prototype. Production requires representative coverage and a distance policy for unfamiliar inputs.','A neighbor vote; return an unknown state when the nearest observations are too far away.','Distance to training examples, rare-state recall, and lookup latency as the reference set grows.'),
'kernel':('Nonlinear boundary for two sensor signals','Two measurements with a binary state label. Fit scaling on training data and compare a linear boundary with the radial-basis boundary.','A predicted state from the selected support-vector model; a class output is not automatically a calibrated probability.','Errors near the boundary, input drift, support-vector count, and inference time.'),
'bayes':('Fast event-category baseline','Eight numeric measurements and a reviewed category for the Gaussian naive Bayes prototype. A text spam model would instead need token counts and a suitable multinomial model.','A category and an explicitly documented conditional-independence assumption.','Calibration, correlated-feature failures, changing class frequencies, and minority-class recall.'),
'gaussian':('Sensor calibration with uncertainty','Measurement position and noisy observed reading. Keep units, calibration provenance, and operating range. A real forecasting task needs a chronological split.','A predicted mean plus a model-based uncertainty interval; the saved batch helper returns the mean, while the training script demonstrates intervals.','Empirical interval coverage, absolute error, out-of-range inputs, and increasing fit cost as observations grow.'),
'cluster':('Exploratory grouping of operating patterns','Unlabeled numeric observations. K-means assigns groups; domain experts must inspect the groups before giving them business names.','An arbitrary cluster identifier from the saved scaler and K-means model. The density-based exploratory comparison has no out-of-sample predictor here.','Cluster size, distance to centers, instability after retraining, and unexplained new groups.'),
'projection':('Compact representation of scanned digits','Flattened eight-by-eight grayscale images normalized to zero through one. Fit the principal-component basis only on training images.','A lower-dimensional vector; inverse transformation reconstructs an approximation. This is compression, not digit classification.','Reconstruction error, retained variance, downstream task quality, and representation version compatibility.'),
'dense':('Small digit-recognition baseline','An image becomes 64 normalized pixel values; the target is a digit from zero through nine. The bundled handwritten-digit data provides a reproducible start.','Ten learned scores reduced to a class index. An unfamiliar document crop requires separate evaluation and possibly rejection.','Per-digit recall, confusion patterns, crop quality, and input normalization errors.'),
'convolution':('Visual inspection through a digit-recognition prototype','Keep the two-dimensional pixel layout while supplying the same 64-value input contract. Moving to defective-part photographs requires licensed images, new labels, new shapes, and retraining.','A digit class for the teaching build; demonstrate spatial feature extraction without claiming a factory-ready detector.','Per-class errors, lighting and camera shifts, latency, and false rejection cost.'),
'graph':('Relationship-aware community classification','Node features, edge list, and reviewed node labels. The lab generates a graph from hidden communities and evaluates unseen labels on the same visible graph.','Class indices for the original 180-node ordering. The saved graph is fixed; adding entities requires a new graph workflow and inductive evaluation.','Label leakage, edge-source quality, community assumptions, and performance on new graphs.'),
'autoencoder':('Denoising grayscale inputs','Clean training images paired with artificially corrupted versions. Production needs a justified noise model and examples from the actual capture process.','A reconstructed 64-value image from a noisy input, using the saved deterministic encoder and decoder.','Reconstruction error by corruption type, lost details, and downstream recognition quality.'),
'variational':('Learn a smooth space of digit variations','Normalized training images without digit labels as reconstruction targets. The encoder learns distribution parameters; the decoder learns reconstruction.','The saved helper reconstructs at the latent mean. The training implementation also draws prior samples; inspect both diversity and quality.','Reconstruction loss, divergence contribution, latent utilization, and sample coverage.'),
'adversarial':('Synthetic scalar generator for simulator experiments','Samples from a known two-mode distribution provide a controlled target. Replacing it with business data requires permission, coverage checks, and privacy evaluation.','Two noise values produce one synthetic scalar. The model is deliberately one-dimensional and is not an image generator.','Mode coverage, sample distribution, seed sensitivity, and collapse despite plausible training losses.'),
'ngram':('Small operations-text completion model','Training sentences tokenized into words; vocabulary and context counts are learned from the training split. Real logs require removal of secrets and stable tokenization.','A distribution over the next word for a supplied context. The saved counting model needs no neural framework.','Held-out perplexity, unknown-word rate, memory growth with context order, and inappropriate memorized text.'),
'reinforcement':('Sequential route decisions in a small simulator','States, permitted actions, transition rules, rewards, termination, and episode length define the data-generating environment.','An action index for an integer state: zero up, one right, two down, three left. Physical deployment is outside this lab.','Success rate, episode return, unsafe actions, reward exploitation, and transfer failure on changed maps.')}

def build_section(t):
    name,data,contract,monitor=PROJECTS[t['id']]
    return f'''## Build it from start to finish: {name}

**Runnable implementation lab — local prototype.** This build connects the real-world scenario to an executable teaching model. Simulated data is explicitly identified; a good score on it does not establish business performance.

1. **Write the decision and success criterion.** State who consumes the prediction and which mistake is expensive. Define a measurable baseline before choosing a model. For this project: {name.lower()}.
2. **Define and inspect the data.** {data} Use the bundled dataset first. Before replacing it, document permission, provenance, missing values, units, class coverage, duplicates, and when each feature becomes available.
3. **Create the evaluation boundary.** The program fixes a seed and separates fitting from held-out evaluation where that protocol applies. For real workloads, choose time-based or entity-based separation when random splitting would leak related observations. The graph, adversarial, and reinforcement modules explain their distinct protocols.
4. **Construct the model.** Read the algorithm and the matching function in foundations.py. Trace the input shape through every preprocessing and model stage. The numerical example above lets you check a small part by hand.
5. **Train and compare.** Run the first Practice experiment. Record its baseline, metrics, runtime, and failure cases. Run the changed setting and explain the result. Choose settings using validation; reserve final test evidence for the chosen configuration.
6. **Save a usable artifact.** The program writes a model-manifest.json, an example-input.json, a fitted model or learned counting/action table, and report.json. It reloads the artifact in a separate inference function before declaring success. Learned scaling stays with applicable classical models. A neural state dictionary alone would be insufficient without the matching architecture; this build also exports an executable inference module.
7. **Use it on a new input.** In Practice, select **Load saved example**, edit one value while preserving the documented shape, and select **Predict new input**. This reloads the saved model and runs inference without training again. {contract}
8. **Deploy the prototype locally.** The localhost website is the demonstration serving layer: input validation → saved artifact → inference → output. Keep the model's environment and feature contract with the release. Only load artifacts you created or trust; executable model formats are not passive documents.
9. **Plan a production handoff.** Add representative-data evaluation, request limits, authentication, authorization, observability, model versioning, rollback, and a suitable reject/fallback behavior before exposing a service to other users. These are engineering extensions, not completed production features of this local lab.
10. **Monitor and decide when to retrain.** {monitor} Define a threshold and response owner. Track whether the new data resembles the training population, and compare a candidate model with the deployed baseline before replacement.

### The build architecture

```mermaid
flowchart TD
 A[Problem and permitted data] --> B[Split and validate]
 B --> C[Fit preprocessing and model]
 C --> D[Evaluate held-out evidence]
 D --> E{{Meets agreed criterion}}
 E -->|Revise using validation| C
 E -->|Accept| F[Save versioned artifact]
 F --> G[Reload for new input]
 G --> H[Local prediction and monitoring plan]
```

### Command-line inference alternative

The website remembers the last successful model folder for this lesson. In its guided terminal, replace MODEL-FOLDER below with that actual folder name and NEW-RUN with a unique output name. Edit a copy of example-input.json to supply your own values.

```text
python predict_saved.py --model-dir outputs/MODEL-FOLDER --input outputs/MODEL-FOLDER/example-input.json --output outputs/prediction-NEW-RUN.json
```

### Build acceptance evidence

Submit a one-paragraph problem definition, feature/target contract, baseline and selected-model reports, one failure analysis, saved artifact, a changed-input prediction, and a deployment/monitoring decision. Passing a command demonstrates executable code; this evidence explains whether the model is appropriate for its intended use.

'''

def language_project_section(ident):
    cases={
      'scratch':('Train a tiny operations-text language model','Define a permitted operations-text corpus; split whole documents before extracting sequences; build byte embeddings, causal attention, feed-forward layers, and an output head; train next-byte prediction; compare held-out loss; inspect generated text; save weights and vocabulary; package the tokenizer with inference.'),
      'architectures':('Compare three sequence engines for streamed event text','Keep corpus, split, training steps, and evaluation fixed. Train the supplied toy mixture-of-experts, state-space, and hybrid alternatives. Inspect routing or state behavior, measured loss, saved checkpoints, and memory/runtime. A toy state-space model does not reproduce the full Mamba algorithm.'),
      'adapt':('Adapt a compact assistant to structured support triage','Define allowed support categories and an output schema; curate independently labeled examples; freeze the base checkpoint; train low-rank adapter matrices; measure category accuracy and valid structure on validation; select settings; evaluate once on final test; save the adapter with the base revision and prompt contract; use the adapter in inference.'),
      'code':('Build a local coding-assistant prototype','Choose a small pretrained coding checkpoint and permitted programming tasks; pin its revision; run generation; assess outputs using independent tests in a suitable isolated execution environment; track correctness and latency; package prompts, model revision, and test cases. This lesson runs a pretrained model; it does not train a coding foundation model from scratch.'),
      'vision':('Build a screenshot-description prototype','Choose a small vision-language checkpoint; pin its revision; prepare permitted images and reference answers; feed pixels through the model processor; generate a description; compare factual details and unsupported claims; record latency and memory; define a rejection behavior for unclear images. This lesson implements pretrained inference, not vision-language pretraining.'),
      'compression':('Prepare a compact model for an edge memory budget','Measure the full runtime memory first; choose precision and supported runtime; quantify rounding error on the numerical lab; for an actual compressed checkpoint, compare task quality, memory, and latency against the uncompressed baseline; preserve tokenizer and serving settings. The supplied code is a numerical compression demonstration, not a production checkpoint converter.'),
      'distillation':('Design a small student for a narrow support task','Define the student task and independent test set; obtain permitted teacher outputs only for training examples; choose soft-target or response-based learning; train student parameters; compare with the student trained without teacher signals; save a versioned model; assess cost and failure cases. The included code demonstrates teacher/student and pruning mechanisms; production text distillation is an extension.')}
    if ident not in cases:return ''
    title,steps=cases[ident]
    return f'''## Build from start to finish: {title}

{steps}

**Deployment flow:** permitted input → validated preprocessing/tokenizer → pinned model and optional adapter → bounded generation → output checks → recorded evaluation. Use the local runner as the prototype environment. Before a shared service, add identity controls, request limits, monitoring, and rollback. Keep training examples and operational logs free of secrets.

**Evidence to submit:** problem statement, data permission and split, baseline, trained or selected artifact, held-out predictions, one failure case, an example inference, compute cost, and a monitoring decision. Where the paragraph identifies a mechanism demonstration or pretrained inference, label the remaining training work as a planned extension.

'''
