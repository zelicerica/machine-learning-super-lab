# Modern language-model terminology

Labels below answer different questions. A single model can be a decoder-only Transformer with grouped-query attention, pretrained on next-token prediction, adapted with low-rank matrices, and deployed inside a retrieval system.

| Term | Expanded name | Meaning and learning destination |
|---|---|---|
| MHA | Multi-head attention | Several query/key/value attention heads; [cache build](#lesson/attention-cache) |
| MQA | Multi-query attention | Several query heads share one key/value head; [cache build](#lesson/attention-cache) |
| GQA | Grouped-query attention | Groups of query heads share key/value heads; [cache build](#lesson/attention-cache) |
| MLA | Multi-head latent attention | Compressed latent key/value representation; [research mechanisms](#lesson/latent-depth) |
| SWA | Sliding-window attention | A position directly attends within a limited local window; [cache build](#lesson/attention-cache) |
| AttnRes | Attention Residuals | Learned attention across earlier layer representations; [research mechanisms](#lesson/latent-depth) |
| MoE | Mixture of experts | A router selects expert computations; [expert costs](#lesson/expert-routing) |
| RoPE | Rotary position embedding | Position-dependent rotations of query/key coordinates; [block components](#lesson/position-normalization) |
| ALiBi | Attention with linear biases | Distance-proportional attention-score biases; [block components](#lesson/position-normalization) |
| LN | Layer normalization | Per-token feature normalization with learned scale and shift; [block components](#lesson/position-normalization) |
| RMSNorm | Root mean square normalization | Scale by root mean square without mean subtraction; [block components](#lesson/position-normalization) |
| QKNorm | Query-key normalization | A family of attention query/key normalization choices; paper variants differ; [block components](#lesson/position-normalization) |
| SwiGLU | Swish-gated linear unit | A feed-forward transformation with a Swish-controlled multiplicative gate; [block components](#lesson/position-normalization) |
| SiLU | Sigmoid linear unit | Activation x × sigmoid(x), also called Swish |
| GELU | Gaussian error linear unit | Smooth activation used by the baseline feed-forward network |
| SFT | Supervised fine-tuning | Learn desired responses from examples; [tuning comparison](#lesson/adaptation-comparison) |
| PEFT | Parameter-efficient fine-tuning | Adapt a relatively small parameter subset; also a software-library name |
| LoRA | Low-rank adaptation | Train small factorized updates to frozen matrices; [tuning comparison](#lesson/adaptation-comparison) |
| QLoRA | Quantized low-rank adaptation | Adapt a quantized frozen base; the paper uses NormalFloat4, double quantization and paged optimizers; [cloud extension](#lesson/quantized-cloud) |
| NF4 | Four-bit NormalFloat | A nonuniform four-bit representation motivated by normally distributed weights |
| DPO | Direct preference optimization | Train using preferred/rejected responses relative to a reference policy; [preference build](#lesson/preference-learning) |
| RL | Reinforcement learning | Learn a policy using rewards from actions and outcomes |
| RLHF | Reinforcement learning from human feedback | Human feedback guides reward-based policy optimization |
| RLVR | Reinforcement learning with verifiable rewards | A programmatically checkable outcome supplies reward; [token policy](#lesson/policy-gradient) |
| RM | Reward model | A learned scoring model; [reward build](#lesson/reward-learning) |
| PPO | Proximal policy optimization | A policy optimization family using constrained/clipped update objectives; [policy comparison](#lesson/policy-optimization) |
| GRPO | Group relative policy optimization | Uses response-group reward comparisons and policy ratios; [policy comparison](#lesson/policy-optimization) |
| GSPO | Group sequence policy optimization | Uses sequence-level importance ratios and clipping; [policy comparison](#lesson/policy-optimization) |
| AIME | American Invitational Mathematics Examination | Mathematics competition whose questions can be used under a specified evaluation protocol |
| HLE | Humanity's Last Exam | A named difficult academic question benchmark; versions matter |
| MMLU | Massive Multitask Language Understanding | A named broad subject evaluation |
| SWE-bench | Software-engineering benchmark | Repository issue-resolution evaluation; variants and agent harness matter |
| GPQA | Graduate-Level Google-Proof Question Answering | Expert-written science-question evaluation |
| PPL | Perplexity | Exponential of average negative log probability under the stated tokenization |
| LaaJ | Large language model as a judge | Use a language model to assess outputs against a rubric; [judge audit](#lesson/judge-audit) |
| KL | Kullback–Leibler divergence | Directional distribution difference; not symmetric and not an accuracy score |
| Pass@k | Pass at k | At least one correct candidate among k attempts under the sampling protocol |
| Elo | Named after Arpad Elo; not an acronym | Comparative ratings updated from outcomes |
| TTFT | Time to first token | Delay from request start to first observed token |
| TPOT | Time per output token | Here: average subsequent-token interval, excluding initial wait |
| ITL | Inter-token latency | Observed gap between successive tokens |
| KV cache | Key-value cache | Temporary attention state retained during generation |
| CPU | Central processing unit | General-purpose processor used for the core teaching experiments |
| GPU | Graphics processing unit | Parallel processor useful for larger tensor workloads |
| VRAM | Video random-access memory | Common name for graphics-device memory; total needs exceed stored weight bytes |
| CUDA | Compute Unified Device Architecture | NVIDIA's graphics-processor computing platform |
| HTTP | Hypertext Transfer Protocol | Request/response protocol used by the local streaming service |
| JSON | JavaScript Object Notation | Structured report and request format |
| SSH | Secure Shell | Encrypted remote connection and port-forwarding protocol |
| SCP | Secure copy | File-copy command commonly used over Secure Shell |

**Read a metric before comparing it:** record the denominator, tokenizer, dataset version, model revision, prompt, tools, sampling settings, time boundary, and hardware. A percentage without a protocol is not enough evidence for an engineering decision.

**Classify this example:** supervised fine-tuning using low-rank adaptation on a grouped-query Transformer, with retrieval at inference and a judged evaluation. The five labels respectively describe a training objective, parameter updates, architecture, evidence access, and evaluation. None replaces the others.
