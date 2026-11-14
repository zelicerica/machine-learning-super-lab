# Verification record

Executed on a Linux CPU host with Python 3.12.14, PyTorch 2.6.0+cpu,
Transformers 4.57.1 and PEFT 0.17.1. Full installed versions are in
`environment-verified.txt`. These results describe this run and do not predict
another machine's runtime or a real-world dataset's quality.

## Checks performed

- Eight automated tests passed: split separation and balance, strict output
  schema, padding versus real EOS supervision, Transformer causal masking,
  experimental architecture causality and gradients, MoE token routing and
  router gradients, SSM state carryover, and retrieval's no-match behavior.
- All Python files passed syntax compilation.
- Actual SmolLM2 checkpoint download, tokenization, CPU generation, LoRA
  training, adapter save/reload, validation evaluation and final test evaluation
  completed successfully.
- Scratch Transformer completed 100 training updates. Toy SSM, MoE and hybrid
  language models each completed 20 training updates with finite losses.
- Quantization and distillation numerical demonstrations ran successfully.
- Retrieval selected `mem-1` for the CUDA memory question. Generation ran, but
  repeated advice, omitted the requested citation and reached the output limit.
  This is an observed quality limitation, not a successful grounded-QA assessment.
- SmolVLM-256M-Instruct downloaded and ran on CPU. On a synthetic red circle
  over a white background, it answered: `The circle is red.` This verifies the
  image-input path only, not general vision quality.
- The evaluation-report comparison script ran on the final test reports.

## Adapter results

Training used 36 original synthetic tickets, rank 8 query/value LoRA adapters,
60 updates, batch size 1, learning rate 0.001 and seed 42. The trainable adapter
contained 460,800 parameters. Training-loop time in this run was about 38 seconds,
excluding model loading, validation and other setup; it is not a runtime promise.

| Measurement | Original model | Trained adapter |
|---|---:|---:|
| Strict validation accuracy | 0/12 | 12/12 |
| Validation schema validity | 0/12 | 12/12 |
| Strict final test accuracy | 0/12 | 11/12 |
| Final test schema validity | 0/12 | 12/12 |
| Validation answer-token loss | 2.1122 | 0.02985 |

The original model failed the strict JSON contract. Its zero strict score does
not mean it had no semantic understanding. Raw responses are included in
`reference-results/`; inspect them before interpreting the metrics.

The failed final-test example was:

> The same customer document appears in both learning and scoring examples.

Expected label: `data`. Adapter prediction: `environment`. The training settings
were not changed after inspecting this final test. This error makes a useful
data-leakage and generalization discussion exercise.

The dataset is small, synthetic and stylistically narrow. A 12-example score is
not a statistically strong estimate of production performance. Generation lengths
differed between base and adapter, so shorter request latency does not establish
higher raw inference throughput. Timing was descriptive, not a controlled benchmark.

## Scratch model results

| Prototype | Parameters | Updates | Validation loss before | Validation loss after |
|---|---:|---:|---:|---:|
| Dense Transformer | 136,960 | 100 | 5.7052 | 1.4564 |
| Toy diagonal SSM | 120,448 | 20 | 5.6886 | 2.5481 |
| Attention with toy MoE | 104,840 | 20 | 5.7493 | 3.0064 |
| Toy SSM and attention hybrid | 128,704 | 20 | 5.6272 | 2.5701 |

These runs verify that the learning loops operate. The unequal parameter and
update budgets do not support architecture rankings. Generated samples remain
poor after these short runs. The source corpus is a tiny repetitive grammar.

## Not executed here

- Native Windows/macOS installation or NVIDIA GPU execution.
- A Vast.ai rental or billing transaction.
- The optional Qwen coding checkpoint download or generated-code benchmark.
- Production quantization/export, teacher-generated dataset distillation,
  pruning speed measurements or vision fine-tuning.
- A full Mamba implementation, a production MoE system or large-scale pretraining.

Those sections are documented learning activities and extensions. They are not
presented as completed validations. No rented compute or paid inference API was used.
