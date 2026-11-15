# Measured teaching results: modern language engineering

These are actual CPU development runs using Python 3.12 and PyTorch 2.6.0+cpu. The 14 experiment types each completed a baseline run; 13 additional runs changed one setting, and the judge was also run with the real pinned 135-million-parameter checkpoint. Each folder contains its recorded settings, limitations, report and applicable plots. Model weights are omitted; 26 saved model outputs were independently reloaded through the common prediction interface.

These are small teaching mechanisms with original synthetic data. Program completion verifies execution. It does not establish that a model is useful. No results here are scores on AIME, HLE, MMLU, SWE-bench, or GPQA.

| Experiment | Observed baseline result | What to learn |
|---|---|---|
| Multi-head / grouped-query / multi-query caches | 20,480 / 10,240 / 5,120 bytes for the measured 40-position cache | Head sharing reduces this cache component; total model memory does not fall by the same factor |
| Eight-position window | Bounded 4,096-byte cache | Local attention limits direct context; cache bytes exclude temporary attention tensors |
| Simplified latent cache | 2,560 persistent cache bytes | Compression trades representation and reconstruction work; this is not DeepSeek's exact MLA design |
| Dense versus four experts | 44,352 versus 93,768 stored parameters; held-out byte perplexity 1.74 versus 2.00 | More stored capacity did not improve this run; top-one routing still stores every expert |
| Full / low-rank / packed-four-bit tuning, 100 updates | 44,352 / 2,176 / 2,176 trainable parameters; exact label accuracy 1/3 / 0/3 / 0/3 | Small updates are not a promise of useful adaptation; three validation examples give weak evidence |
| Packed projection storage | 144,672 tensor bytes versus 177,440 for the full model | Only selected projections are quantized; retained weights, scales, and adapters remain. This is not total process memory or NF4 QLoRA |
| Preference optimization | Held-out preferred-response win rate remained 1.0 | A saturated pair-ranking metric can hide poor generation; inspect response loss and generated text |
| Learned reward scorer | 0/2 held-out pair accuracy | Four training pairs and mean pooling did not generalize; a reward score is not verified truth |
| Exact-reward REINFORCE | Held-out exact match fell from 20% to 15% | Optimizing sampled training rewards need not improve generalization |
| Deliberately flawed zero-prefix reward | Held-out exact match fell from 20% to 10% | An easier reward can reward the wrong behavior |
| PPO / GRPO / GSPO | Held-out exact match after updates: 20% / 15% / 15% | These tiny runs do not rank the algorithms. GRPO and GSPO use two-token answers, so they poorly expose variable-length differences |
| Evaluation harness | Greedy accuracy 20%; estimated pass@1 10.625%; pass@4 26.714% | Greedy decoding and stochastic sampling are different protocols; the problem-level 95% interval for greedy accuracy was about 8.1%–41.6% |
| Deliberately biased judge fixture | 100% valid choices, 0% agreement | Format compliance and swap consistency do not prove correctness |
| Real local judge | 0/12 strictly valid A/B-only outputs | Preserve raw output and abstain. A tiny checkpoint with a four-token budget was unsuitable for this protocol; this does not prove every possible prompt would fail |
| Streaming service | Twelve measured loopback requests per run, across prompt lengths and cache modes | Inspect actual first-token and later-token timing; small CPU workloads and scheduling can reverse the expected speed ranking |

The changed-setting runs use 50 updates, except tuning uses the lesson's 150-update command, preference optimization changes beta to 0.5, and REINFORCE changes only its reward rule. `matrix.json` and `changed-matrix.json` record exit status and elapsed command time. The expert-routing reports were rerun after correcting measurement to cover the whole probe prompt rather than only the final token of a cache check; the earlier matrix time is retained as first-pass timing.

To reproduce, use the corresponding lesson's Practice command and a fresh output folder. Compare settings in each report. Report at least three seeds for an engineering conclusion; these development references are not a multi-seed study. Select settings using validation data, then use the separate final-test partition once. Do not tune against a final test or treat this reference output as your own assessment submission.

Optional genuine NormalFloat4 QLoRA is documented separately and was not executed on a GPU in this release. No paid cloud resource was rented.
