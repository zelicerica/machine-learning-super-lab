# Language model atlas correctness review

Reviewed November 14, 2026. Scope: the existing language-model atlas, its relationship to the broader machine-learning atlas, and the associated workbook. This is a curriculum review, not a new benchmark of the named model checkpoints.

## Assessment

The existing atlas was a good beginner overview of common modern neural designs. Its explanations already distinguished active expert computation from total storage, qualified attention memory costs, described recurrent inference state correctly, and noted overlapping categories. It was not a complete overview of language modeling: classical count-based models were implemented elsewhere but absent from the language atlas, and the headings could still suggest an exclusive family tree.

The revised atlas is a representative map of major approaches and design dimensions. No finite introductory atlas can enumerate every research variant. Model names are examples, not a claim to list the newest or best checkpoint in each category.

## Corrections and additions

| Item | Review finding | Revision |
|---|---|---|
| Position within machine learning | The connection was not explicit enough | Added a relationship diagram and a visual explaining that language models include both count-based and neural approaches |
| Deep learning | A hierarchy placing every language model inside deep learning would be wrong | Explained that many modern neural language models use deep learning, while n-gram models do not require neural networks |
| Text processing versus language modeling | A text input alone does not identify a language model | Added the example of a logistic regression spam classifier and clarified encoder pretraining versus downstream outputs |
| Independent dimensions | Numbered “families” combined roles, mechanisms, routing, and generation | Renamed them design choices and added an eight-question model-card reading algorithm |
| Classical language modeling | Present as a runnable foundation lab but missing from the language atlas | Added a count-based algorithm, exact smoothing example, complete lifecycle, and link to the existing lab |
| Additional neural approaches | Fixed-window feed-forward, causal convolution, and linear attention were not covered clearly | Added sourced Concept entries, real use cases, a shared start-to-finish project specification, and design exercises |
| Base versus pretrained | “Base or pretrained” incorrectly implied synonymy | Split the definitions; instruction-tuned checkpoints can also be pretrained |
| Diffusion | The title grouped diffusion and iterative masked generation too broadly | Explained that not every masked predictor or iterative infiller is a diffusion model, and a diffusion denoiser can be a Transformer |
| Small versus large | Existing caveat was correct | Retained the absence of a universal size cutoff and the difference between active and total parameters |
| Model versus application | Existing agent explanation was correct | Reinforced that retrieval-augmented generation and agent loops are system designs, not standalone model architectures |
| Navigation and runnable scope | The sidebar exposed only the broader atlas | Added distinct machine-learning and language-model links; preserved all lesson identifiers and implementation tags |

## Primary sources checked

- [Bengio and coauthors: A Neural Probabilistic Language Model](https://www.jmlr.org/papers/v3/bengio03a.html) — count-based predecessor and learned fixed-window neural language model.
- [Dauphin and coauthors: Language Modeling with Gated Convolutional Networks](https://arxiv.org/abs/1612.08083) — finite-context convolutional alternative.
- [Devlin and coauthors: Bidirectional Encoder Representations from Transformers](https://arxiv.org/abs/1810.04805) — bidirectional pretraining and downstream adaptation.
- [Gu and Dao: Mamba](https://arxiv.org/abs/2312.00752) — selective state-space sequence modeling.
- [De and coauthors: Griffin](https://arxiv.org/abs/2402.19427) — gated linear recurrence combined with local attention.
- [Katharopoulos and coauthors: linear-attention Transformers](https://arxiv.org/abs/2006.16236) — a specific kernel-feature formulation and recurrent implementation.
- [Nie and coauthors: Large Language Diffusion Models](https://arxiv.org/abs/2502.09992) — forward masking and reverse generation parameterized by a Transformer.
- [Qwen team's architecture and parameter table](https://qwenlm.github.io/blog/qwen-moe/) — Qwen1.5-MoE-A2.7B has 14.3 billion total and 2.7 billion activated parameters.
- [DeepSeek-V2-Lite model card](https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite) — approximately 16 billion total and 2.4 billion activated parameters.

## Implementation boundary at the time of the atlas review

This revision changes course content, illustrations, and reference navigation. It adds no new training implementation or cloud resource. The statistical word-model lab was already included. The new feed-forward, convolutional, and linear-attention language-model entries are explicitly Concept project plans. Original implementation verification remains recorded in VERIFICATION.md; the atlas-specific checks are recorded separately there.

## Subsequent training and retrieval expansion

The next release adds actual recurrent, masked-encoder, and recurrent encoder–decoder builds, a deep-network workshop, a controlled transfer comparison, and a retrieval/generation workshop. Earlier statements about missing implementations describe the atlas-review version. See `Building-Models-and-Retrieval.md` and the latest verification section for current executable coverage.
