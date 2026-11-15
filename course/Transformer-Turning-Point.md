# The Transformer turning point

**Concept · interactive architecture and historical evidence.** The 3D explorer above is a schematic of the original encoder–decoder Transformer. It does not run a trained model or display real activations. The numerical attention activity below uses invented vectors so you can inspect every calculation.

## Start with the problem

Imagine translating “The bank by the river is closed.” A useful representation of “bank” depends on surrounding words. Attention lets a position combine information from other positions with input-dependent weights. This is a computation over vectors, not a dictionary lookup or evidence that a model understands the sentence as a person does.

Before Transformers, attention already helped recurrent translation systems align input and output. The 2017 paper **Attention Is All You Need**, by Ashish Vaswani and colleagues, made attention central to a sequence architecture that removed recurrence and convolution. Its experiments focused on machine translation. It did not introduce the first attention mechanism, a conversational assistant, or a general solution to truthfulness.

## Read the 3D architecture

Drag to orbit, Shift-drag or right-drag to pan, and scroll to zoom. On a touchscreen, use one finger to rotate and two to pan and zoom. You can also focus the canvas and use arrow keys, Shift-arrow keys, +, −, and Home. Select any component in the scene or the component menu to read its job. Reset restores the starting view.

The gold column is the **encoder**: source tokens become context-aware representations. The mint column is the **decoder**: a shifted output prefix and the encoder's output produce a distribution for the next output token. The whole-model view displays **all six encoder layers and all six decoder layers**, with every attention, feed-forward, and Add & Norm block represented separately. Each layer has independent learned parameters; six is a design choice, not the definition of a Transformer. Orange connections carry the **final encoder layer’s output to cross-attention in every decoder layer**. Intermediate encoder layers do not supply those connections.

Select an Encoder or Decoder number to open that layer. The detail view shows solid main paths, dashed residual paths, and the external encoder-memory input where needed. The eight divisions on an attention block represent eight parallel heads, not eight sequential layers. Use **Whole model** to return to the complete architecture, **Explode layers** to increase spacing, and **Previous / Next** to walk through the components in calculation order. This is an architectural tour, not a live training run or a time-accurate animation.

| Component | What it does | What to notice |
|---|---|---|
| Token embeddings and positions | Turn token IDs into vectors and add position information | The original model used sinusoidal positions; later models can use different schemes |
| Encoder self-attention | Each source position gathers information from source positions | No causal restriction; padding can still be masked |
| Masked decoder self-attention | Each output position gathers information from its available prefix | Future target positions are hidden during training |
| Encoder–decoder cross-attention | Decoder queries attend to keys and values from the final encoder output | Queries and memory come from different streams |
| Position-wise feed-forward network | Apply learned transformations independently at each position | It mixes features, while attention mixes information across positions |
| Add & Norm | Add the sublayer input to its output and apply layer normalization | Original paper: post-normalization, with dropout during training; many later systems differ |
| Linear projection and softmax | Produce a probability distribution over output vocabulary tokens | A high probability is not proof that a generated statement is true |

The base configuration used eight attention heads and a model width of 512. Multiple heads use separate learned projections, combine their outputs, and apply an output projection. They are not eight copies of the same attention matrix. The picture is not to scale: block sizes do not represent parameter counts, memory, or latency.

## Attention you can calculate

Queries describe what a position is looking for. Keys are compared with queries. Values carry the information combined by the resulting weights. In a real model, learned matrices produce these vectors.

```text
Attention(Q, K, V) = softmax(Q Kᵀ / √d_k + mask) V

One query q = [1, 0]; key dimension d_k = 2
Keys:   A = [1, 0], B = [0, 1], C = [1, 1]
Values: A = [2, 0], B = [0, 2], C = [2, 2]
Scaled scores: [0.7071, 0, 0.7071]
Unmasked weights, rounded: [0.4011, 0.1978, 0.4011]
Weighted output, rounded: [1.6044, 1.1978]
```

In the activity above, the query is at position B. Turn on the causal mask: C is a future position and gets zero weight. The remaining weights renormalize to approximately [0.6698, 0.3302, 0], giving output [1.3395, 0.6605]. We set a prohibited score to negative infinity before softmax; simply setting its score to zero would still give it nonzero probability.

**Change one variable.** Keep the mask off and change the query's first coordinate from 1 to 0. All three scores become zero. Predict the equal weights and output before reading the result. The keys and values stay fixed, so the experiment isolates the effect of the query.

These are synthetic vectors for a single query and one attention head, without learned projections or gradients. Attention weights show this computation's mixing weights; they are not a complete explanation of a model's answer.

## Why this architecture mattered

Recurrent models process sequence states step by step. The Transformer made it possible to compute representations for many positions in parallel during training, including with a causal mask when the target sequence is already available. Direct attention connections also shortened the computational path between distant positions. These properties fit large matrix operations well.

There are tradeoffs. Standard dense self-attention forms an n-by-n set of position-pair scores: doubling sequence length quadruples the number of those scores per head. Optimized implementations can reduce memory traffic or avoid materializing the full matrix, but do not make the underlying dense pairwise work free. Autoregressive generation still needs earlier generated tokens to produce later ones. Parallel training does not mean an ordinary decoder generates an entire answer simultaneously.

## How the idea spread

The milestones below describe distinct research contributions. The connection between them explains the architecture's influence; it is not a claim that one paper alone caused every subsequent AI advance.

| When | Primary work | What changed | Everyday connection |
|---|---|---|---|
| 2014 preprint; 2015 conference | Neural Machine Translation by Jointly Learning to Align and Translate | Learned alignment supplied context to a recurrent translation model | Attention existed before Transformers |
| 2017 | Attention Is All You Need | Attention-based encoder–decoder translation without recurrence or convolution | A reusable architecture for sequence modeling |
| 2018 preprint; 2019 conference | BERT | Bidirectional Transformer encoder pretraining with masked language modeling | Adaptable representations for tasks such as classification and question answering |
| 2020 | Language Models are Few-Shot Learners (GPT-3) | Large autoregressive Transformer language models studied learning from examples in prompts | Prompt-based task interfaces, with important reliability limitations |
| 2020 preprint; 2021 conference | An Image is Worth 16x16 Words (Vision Transformer) | A Transformer consumed a sequence of image-patch embeddings | The architecture also supported visual recognition |
| 2022 | Training language models to follow instructions with human feedback | Demonstrations, preference comparisons, and reinforcement learning improved instruction following | Helpful conversational behavior requires more than the architecture alone |

**Three related families.** An encoder-only model such as BERT can use context on both sides to build representations. A decoder-only language model such as GPT-3 predicts subsequent tokens causally and need not contain the original encoder–decoder cross-attention. An encoder–decoder model transforms one sequence into another using both stacks. The explorer shows the third family. Do not assume every modern model contains every block in this picture.

The wider engineering consequence is that teams can adapt pretrained representations or models rather than training every application from scratch. Search, document processing, code assistance, translation, and vision can build on related components. This is an engineering interpretation of the cited research, not a quantified measure of economic impact. Data quality, compute, optimization, evaluation, product design, and later training methods also matter.

## Build from start to finish

**Design activity, not an implemented training pipeline in this lesson.** Design a small internal support-ticket classifier using an encoder. Continue to the lab's [encoder build](#lesson/scratch-encoder) for implementation practice; the [attention and cache lab](#lesson/attention-cache) directly implements attention mechanisms.

1. **Problem and success.** Route tickets to billing, access, or technical support. Write a minimum per-class recall target and a review policy for uncertain cases before experimenting.
2. **Data and contract.** Use permitted, de-identified tickets. Input is text; target is one of three documented labels. Exclude secrets and keep a record of allowed data use.
3. **Preprocess and split.** Deduplicate related tickets and split by time or customer to limit leakage. Fit preprocessing on training data; keep validation for tuning and a locked test set for final assessment.
4. **Architecture.** Compare a small pretrained encoder plus a classification head with a simple word-feature classifier. Encoder-only fits classification; a generative decoder is not automatically necessary.
5. **Training.** Use cross-entropy on training examples; start by training the head and measure whether further adaptation helps. Record model revision, tokenizer, seed, budget, and hyperparameters.
6. **Independent evaluation.** Compare both models on the same held-out tickets using per-class precision and recall, confusion matrices, latency, and memory. Inspect errors and test abstention behavior; training accuracy is not sufficient evidence.
7. **Save and reload.** Save weights, tokenizer, label mapping, input limits, and preprocessing configuration in a new run folder. Reload in a fresh process and compare predictions on fixed inputs.
8. **Predict and deploy locally.** Tokenize a new de-identified ticket, check input limits, run inference, and return a label with an uncertainty policy. Keep the service bound to localhost for the lab.
9. **Production gaps.** Document authentication, authorization, logging redaction, dependency review, data retention, multilingual evaluation, and abuse handling that a real service would need.
10. **Monitor and roll back.** Track reviewed misroutes, drift, per-class errors, and latency. Preserve the prior model and its input contract so rollback is testable.
11. **Cost and evidence.** Measure CPU runtime and memory before considering a graphics processor. Submit the diagram, baseline comparison plan, split rationale, proposed reload test, failure cases, and cost estimate. Do not invent measured results for this design.

## Explanation check

In the Checkpoint evidence journal, explain which information each of the three attention blocks can access. Then explain why the original paper enabled parallel training but did not eliminate sequential next-token generation. Name one contribution that predates the paper and one that came later.

For the numerical activity, record the output with the mask off, the output with it on, and your prediction when the query coordinate becomes zero. Explain why future-position weight becomes exactly zero. A successful response distinguishes architectural structure, a numerical demonstration, a training method, and a deployed product.

## Keep exploring

Try [attention and key/value caches](#lesson/attention-cache), [position and normalization](#lesson/position-normalization), [training a tiny model from random weights](#lesson/scratch), and [preference learning](#lesson/preference-learning). These lessons separate mechanisms that are often compressed into the single word “Transformer.”
