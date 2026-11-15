# Content audit — release 2026.11-midnight.4

Reviewed 2026-11-15. Scope: all 58 lessons, their quizzes and runnable-command mappings, source explanations, teaching images, Mermaid diagrams, and corresponding implementation paths. The research timeline’s 47 landmark entries were checked for source identity and contribution; each lesson now ends with two to four annotated primary sources (148 placements). References include original research, official model cards and implementation documentation.

## What changed

- Replaced repeated survey images in all 18 runnable Foundations lessons with distinct, accurate navy/green illustrations. Broad comparison pictures now appear in the atlas where comparison is the purpose.
- Removed 18 duplicated deployment diagrams and repeated operational paragraphs. Shared setup, artifact handling and submission instructions live in the linked experiment workflow; model-specific evidence and production gaps remain in each lesson.
- Corrected method descriptions, numerical exercises, several advanced diagram arrows, evaluation partition claims, and one on-policy update error. The detailed coverage below records the relevant mechanism for every module.
- Separated math, quantization, distillation and architecture mechanism outputs. Existing evaluation reports are protected from overwrite.
- Updated historical reference metadata without inventing new measurements. Old REINFORCE measurements are explicitly identified as results of the previous implementation.
- Preserved stable lesson IDs, Foundations-first numbering, learner progress, midnight styling, green text and the locally bundled motion background.

## Evidence and limits

- 64 Python tests passed: 56 course/server/content tests plus 8 original language-model tests.
- All 75 lesson/reference routes passed the DOM interaction check, including all 69 bundled Mermaid syntax instances, internal links/assets, learning sequence and saved-progress retention.
- All 18 Foundations implementations completed their two lesson configurations: 36 fresh runs, artifact checks, saved-model reloads and new-input predictions.
- All 14 modern implementation entry points completed two-step smoke runs with consistent report/data-card metadata and existing reload checks. Separate validation/test judge fixtures, reward partition selection, PPO and timing-probe paths were exercised. Short smoke runs establish execution, not learning quality.
- The 18 new SVGs were rendered and visually reviewed. Desktop Chrome confirmed the clustering figure in its worked-example section, lesson-specific source footers, Bayes image mapping, the final-assessment diagram and diagram enlargement. Automated checks cover the other routes; this is not a claim of manually viewing every page at every viewport.
- Content and source verification do not reproduce published research. Previously recorded longer experiments remain historical evidence. Pretrained model downloads/inference, GPU/QLoRA, paid rentals, native Windows/macOS installation and mobile-device rendering were not newly rerun for this audit.

## Coverage by learning module

Every row includes reference matching. “Checked” records review, not proof that a model generalizes beyond the included teaching data.

| Lesson | Module | Review and correction | References |
|---|---|---|---|
| 01 | Understand how models learn (`learning-signals`) | Checked learning signals, architecture/task distinctions and examples; retained the relevant overview. | 4 |
| 02 | Choose a model and design the evidence (`model-selection`) | Checked train-only preprocessing and final-test boundaries; retained the selection diagram here. | 2 |
| 03 | Linear regression: learn a weighted prediction (`linear`) | Matched 39-unit prediction, residual 4 and squared error 16 to the coefficients and new figure. | 2 |
| 04 | Logistic regression: turn scores into probabilities (`logistic`) | Checked sigmoid values, threshold 0.8, inverse regularization and training-only scaling. | 3 |
| 05 | Decision trees: inspect the questions (`tree`) | Replaced survey image with the exact 6/4 to 5/0 and 1/4 split; completed overview-tree branches. | 2 |
| 06 | Ensembles: compare forests and boosting (`ensemble`) | Corrected forest probability averaging; labeled hard voting as illustrative and runtime as whole-experiment time. | 3 |
| 07 | Nearest neighbors: classify by similarity (`neighbors`) | Matched five-neighbor 3/2 vote; distinguished vote fraction from calibrated probability. | 2 |
| 08 | Kernel methods: learn a curved boundary (`kernel`) | Separated illustrative radius coordinates from the actual RBF kernel and half-circle dataset. | 2 |
| 09 | Naive Bayes: combine evidence with probabilities (`bayes`) | Added explicit illustrative event likelihoods and normalized posterior; distinguished Gaussian feature densities. | 2 |
| 10 | Gaussian processes: predict with uncertainty (`gaussian`) | Matched the 10 ± 1.96 × 2 interval and explained coverage assumptions; removed implied automatic baseline. | 2 |
| 11 | Clustering: discover groups without supplied labels (`cluster`) | Replaced leakage picture with center update to 2; separated K-means held-out assignments from training-only DBSCAN. | 3 |
| 12 | Principal component analysis: compress measurements (`projection`) | Matched principal-direction illustration and compression/reconstruction objective to PCA. | 2 |
| 13 | Dense neural networks: train a hidden representation (`dense`) | Matched 64 inputs, 32 hidden values and 10 logits; diagram shows representative units, not all neurons. | 3 |
| 14 | Convolutional networks: share a visual detector (`convolution`) | Checked feature-map sizes through convolution, pooling, flattening and classifier. | 2 |
| 15 | Graph neural networks: learn from connections (`graph`) | Separated unweighted numerical illustration from degree-normalized implementation; retained transductive boundary. | 2 |
| 16 | Autoencoders: reconstruct through a bottleneck (`autoencoder`) | Matched noisy-input to clean-target reconstruction; no digit-label supervision claim. | 2 |
| 17 | Variational autoencoders: learn a distribution (`variational`) | Checked reparameterization 2 + 0.5 × (−1) = 1.5 and reconstruction/KL roles. | 2 |
| 18 | Adversarial generation: train two competing models (`adversarial`) | Added mode-collapse illustration; corrected evaluation label and independent evaluation random generator. | 2 |
| 19 | Statistical language models: count before you train a network (`ngram`) | Matched 3/4 and 1/4 transition counts; checked add-one normalization and unknown-token handling. | 2 |
| 20 | Reinforcement learning: learn actions from rewards (`reinforcement`) | Matched Q update to 2.8; clarified terminal target, step cap and same-map evaluation. | 3 |
| 21 | Build your workspace (`setup`) | Corrected extracted-root versus lab subfolder instructions and linked the common workflow. | 2 |
| 22 | Run your first model (`inference`) | Checked frozen weights, evaluation mode, chat template, token slicing and sampling controls. | 2 |
| 23 | See the numbers underneath (`math`) | Removed appended adaptation survey; added a focused scalar-gradient and probability command. | 3 |
| 24 | Measure a baseline (`baseline`) | Merged duplicate evaluation explanations; added valid-but-wrong examples and protected existing reports. | 2 |
| 25 | Teach with a small adapter (`adapt`) | Matched LoRA targets/rank/scaling and assistant-only loss; fixed prerequisite navigation and specific evidence. | 2 |
| 26 | Give the model useful notes (`retrieval`) | Checked lexical retrieval, no-match behavior and context versus parameter updates. | 2 |
| 27 | Train from random weights (`scratch`) | Qualified synthetic-data scope and expected outcomes; real-document splitting is an explicit extension. | 2 |
| 28 | Make the numbers smaller (`compression`) | Fixed both scaling divisor and clamp for qmax 7; disclosed int8 storage without four-bit packing. | 2 |
| 29 | Learn from a teacher (`distillation`) | Removed implied pruning/trained-student implementation; demo returns soft targets, KL and a logit gradient. | 2 |
| 30 | Evaluate a coding model (`code`) | Checked checkpoint and independent-test activity; distinguished pretrained generation from training. | 2 |
| 31 | Connect images and language (`vision`) | Led with image upload and clarified external-terminal paths; distinguished pixels from filenames. | 2 |
| 32 | Experiment with other families (`architectures`) | Separated architecture-only outputs; retained parameter/implementation differences and SSM/Mamba boundary. | 3 |
| 33 | Move an experiment to a graphics processor (`cloud`) | Checked workflow and cost examples against official provider documentation; no paid rental executed. | 2 |
| 34 | Prove your engineering decision (`capstone`) | Preserved final-test boundary and report filenames; assessment evidence remains the shared endpoint. | 3 |
| 35 | Pretrained or from scratch: choose the route (`training-route`) | Checked reuse, frozen features, fine-tuning and scratch decision boundaries; added matched primary reading. | 3 |
| 36 | Deep learning: train and inspect every layer (`deep-network`) | Checked the 1,218-parameter count, logits/loss contract and independent split; labeled ReLU in the diagram. | 3 |
| 37 | Build a recurrent language model from scratch (`scratch-recurrent`) | Checked GRU state and next-byte objective; corrected the diagram’s next-time-step recurrence. | 2 |
| 38 | Build a masked-token encoder from scratch (`scratch-encoder`) | Checked two-layer masked-token encoder and supervised positions; distinguished the toy objective from BERT. | 2 |
| 39 | Build an encoder–decoder from scratch (`scratch-sequence`) | Checked GRU encoder–decoder, teacher forcing and free-running evaluation; no cross-attention implementation claimed. | 2 |
| 40 | Compare pretrained, frozen, and fresh models (`transfer-compare`) | Checked grouped 56/20/20 task split, frozen/full/fresh branches and separate pretraining cost. | 3 |
| 41 | Build and audit retrieval-augmented generation (`rag-pipeline`) | Checked normalized TF–IDF similarity, chunk provenance, abstention, stale context and frozen-generation comparisons. | 3 |
| 42 | Synthetic data: generate, audit, and compare (`synthetic-data`) | Checked independent simulator streams, label corruption, coverage and real-data/privacy limits. | 3 |
| 43 | A map of modern language-model engineering (`modern-map`) | Checked shared Transformer ideas across domains without implying identical inputs, objectives or architecture. | 4 |
| 44 | Build attention and measure its cache (`attention-cache`) | Corrected the values-to-weighted-output arrow; checked mask, cache equivalence and head storage ratios. | 4 |
| 45 | Give a Transformer position, scale, and gates (`position-normalization`) | Checked positions, normalization and gates against runnable block options and references. | 4 |
| 46 | Route tokens and account for expert costs (`expert-routing`) | Checked top-one routing, stored versus active parameters and illustrative animation boundaries. | 2 |
| 47 | Explore latent caches and attention across depth (`latent-depth`) | Separated latent key/value compression from attention across depth in the diagram. | 3 |
| 48 | Compare full tuning and small parameter updates (`adaptation-comparison`) | Disclosed that changing steps affects both warm-start pretraining and adaptation budget. | 3 |
| 49 | Plan and run the optional NormalFloat4 extension (`quantized-cloud`) | Checked toy packed integers versus optional NF4/QLoRA; GPU/cloud activity remains unexecuted. | 3 |
| 50 | Learn from preferred and rejected responses (`preference-learning`) | Checked DPO chosen/rejected ratios, frozen reference and training-pair role. | 2 |
| 51 | Build and audit a learned reward model (`reward-learning`) | Created distinct validation/test pairs and accurate data cards; retained style-only scorer limits. | 3 |
| 52 | Move from grid-world actions to generated tokens (`policy-gradient`) | Corrected REINFORCE to one policy update per fresh rollout; annotated older reference measurements. | 3 |
| 53 | Compare clipped and group-relative policy updates (`policy-optimization`) | Separated PPO critic-based advantages from group-relative alternatives in the diagram. | 3 |
| 54 | Choose a benchmark that answers your question (`benchmark-literacy`) | Checked named benchmark tasks, protocol caveats and the homemade arithmetic-suite distinction. | 4 |
| 55 | Build a reproducible evaluation harness (`evaluation-engineering`) | Checked pass@k estimator, problem-level Wilson intervals and byte perplexity units. | 3 |
| 56 | Calibrate judges and compare preferences (`judge-audit`) | Created disjoint question banks; preserved answer-order swapping and independent-case counts. | 2 |
| 57 | Serve a model and measure when tokens arrive (`streaming-service`) | Corrected timing-probe labels/data cards and first-token timestamp connection; serial timing is not a load test. | 3 |
| 58 | Defend a complete language-model engineering decision (`modern-capstone`) | Corrected validation feedback versus a single frozen final assessment; rejection requires fresh final evidence. | 3 |

The source annotations are maintained in [lesson_references.json](lesson_references.json), rendered at the end of each lesson, and collected in [Annotated lesson references](course/Annotated-Lesson-References.md). The broader chronological reading page remains [Research landmarks](course/Research-Landmarks.md).
