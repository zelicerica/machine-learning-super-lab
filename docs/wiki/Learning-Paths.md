# Learning paths and companion guides

[Wiki home](Home.md) · [Repository README](../../README.md)

## What you will build

Every lesson is marked **Concept** or **Runnable implementation lab**. A runnable label means executable code is included; the scope label distinguishes a mechanism demonstration, pretrained inference, adaptation, or training from scratch. Tiny models and fictional datasets teach the engineering process; their scores are not production benchmarks.

| Path | Examples | Evidence you produce |
|---|---|---|
| Foundations | Regression, trees, ensembles, nearest neighbors, kernel methods, clustering, dimensionality reduction | Baselines, evaluation metrics, saved artifacts, new-input predictions |
| Neural and generative models | Dense, convolutional and graph neural networks; autoencoders; variational and adversarial generation | Loss curves, tensor shapes, reconstructions, controlled comparisons |
| Language models | Statistical models, tiny Transformer, expert routing, state-space and hybrid mechanisms | Token predictions, causal checks, measured training and inference |
| Pretrained versus scratch | Reuse, fine-tuning, frozen encoders, and a three-branch transfer experiment | Outcomes with matched downstream examples and explicit earlier training cost |
| Build and retrieve | Deep classifier, recurrent language model, masked encoder, encoder–decoder, retrieval-augmented generation | Train → evaluate → save → reload → predict; retrieval and citation audits |
| Synthetic data | Generate, audit, corrupt labels, narrow coverage, and vary volume | Dataset cards, plots, seed-only/generated-only/mixed comparisons |
| Modern language engineering | Attention and caches, block components, sparse experts, tuning, preferences, policy learning, evaluation, and streaming | Controlled comparisons, reload checks, failure analysis, and measured latency |
| Decision systems | A small reinforcement-learning environment | Action values, reward measurements, and comparison with a random policy |

The **language model atlas** explains that language models are part of machine learning. The atlas is a representative map, not an exhaustive list of every possible architecture. Larger project plans are clearly labeled Concept.

## Learning order and companion guides

1. [Tiny AI — AI for Curious Minds](https://from-number-boxes-to-tiny-ai.ericazelic.chatgpt.site): approachable prerequisite concepts.
2. **Foundations:** how models learn, how to choose a baseline, and classical model builds.
3. **Language models:** inference, evaluation, adapter training, random-weight training, and architecture mechanisms.
4. **Build and retrieve:** deeper neural builds, pretrained-versus-scratch comparisons, retrieval-augmented generation, and synthetic data.
5. **Modern language engineering:** attention → architecture → adaptation → preferences and rewards → evaluation → serving. Start with Lesson 43; dependencies and optional research extensions are labeled.
6. **Capstone:** justify your design using measurements, limitations, and a saved reproducible artifact.

| Guide | Contents |
|---|---|
| [Language model workbook](../../course/SML-Lab-Zero-to-Hero.md) | Concepts, atlas, algorithms, and original language-model labs |
| [Foundation builds](../../course/Expanded-Foundations.md) | 18 end-to-end builds and two concept lessons |
| [Training and retrieval workbook](../../course/Building-Models-and-Retrieval.md) | Pretraining decisions, deep learning, neural builds, retrieval, and synthetic data |
| [Synthetic data](../../course/Synthetic-Data.md) | Generation methods, quality checks, risks, and runnable comparisons |
| [Modern language engineering](../../course/Modern-Language-Engineering.md) | 16 progressive lessons connecting architecture, post-training, evaluation, and serving |
| [Landmark research](../../course/Research-Landmarks.md) | A research timeline with primary papers, explanations, and lesson links |
| [Modern terminology](../../course/Modern-Glossary.md) | Every acronym on the Stanford overview, organized by engineering decision |
| [Optional QLoRA extension](../../course/QLoRA-and-Cloud.md) | Real NormalFloat4 adaptation, hardware prerequisites, and a cloud cost worksheet |
| [Additional build plans](../../course/Concept-Build-Plans.md) | Clearly labeled Concept projects |
| [Vast.ai guide](../../course/Vast-ai-Lab.md) | Optional rental, hardware verification, private connection, cost, backup, and cleanup |
| [Verification](../../VERIFICATION.md) | What was actually run and what remains untested |

The website is the best place to follow workbook links such as `#lesson/...` and `#ref/...`; those are application routes rather than GitHub page anchors. Markdown files remain useful for reading and exporting.

## More learning and projects

- **[Tiny AI — AI for Curious Minds](https://from-number-boxes-to-tiny-ai.ericazelic.chatgpt.site):** the recommended prerequisite introduction.
- **[ML Engineer Success](https://ml-engineer-success.ericazelic.chatgpt.site):** a companion learning workspace for useful, reliable artificial-intelligence products and engineering workflows.
- **[Erica Zelic — Security Engineering & Research](https://erica-zelic-security.ericazelic.chatgpt.site):** professional work and security research.
- **[Little World](https://github.com/zelicerica/little-world):** another public project to explore after the model-building foundations.
