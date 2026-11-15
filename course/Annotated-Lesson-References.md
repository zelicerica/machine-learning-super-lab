# Annotated lesson references

Release 2026.11-midnight.4. Sources are grouped by the current website lesson order. Each annotation states its connection to the exercise; a paper is not a claim that its full experiment was reproduced.

## 01. Understand how models learn

Lesson ID: `learning-signals`. Track: Foundations.

### Reading

Use these sources to check the mechanism and extend the exercise. Research results are not results measured by this lab.

- [Supervised learning — scikit-learn](https://scikit-learn.org/1.7/supervised_learning.html) — Examples of learning from supplied targets, including regression and classification.
- [Unsupervised learning — scikit-learn](https://scikit-learn.org/1.7/unsupervised_learning.html) — Places clustering and representation learning within unlabeled-data methods.
- [A Simple Framework for Contrastive Learning of Visual Representations — Chen et al.](https://arxiv.org/abs/2002.05709) — An original self-supervised representation-learning example; optional further reading, not an implemented lab.
- [Q-learning — Watkins and Dayan](https://link.springer.com/article/10.1007/BF00992698) — Shows that learning from actions and rewards does not require a neural-network architecture.

## 02. Choose a model and design the evidence

Lesson ID: `model-selection`. Track: Foundations.

### Reading

Use these sources to check the mechanism and extend the exercise. Research results are not results measured by this lab.

- [Common pitfalls and recommended practices — scikit-learn](https://scikit-learn.org/1.7/common_pitfalls.html) — Explains data leakage, train-only preprocessing, and consistent transformations at prediction time.
- [Cross-validation: evaluating estimator performance — scikit-learn](https://scikit-learn.org/1.7/modules/cross_validation.html) — Explains evaluation boundaries, validation choices, and group- or time-aware splitting.

## 03. Linear regression: learn a weighted prediction

Lesson ID: `linear`. Track: Foundations.

### Reading

Use these sources to check the mechanism and extend the exercise. Research results are not results measured by this lab.

- [Linear models — scikit-learn](https://scikit-learn.org/1.7/modules/linear_model.html) — Explains weighted predictions and the ordinary least-squares objective.
- [LinearRegression API — scikit-learn](https://scikit-learn.org/1.7/modules/generated/sklearn.linear_model.LinearRegression.html) — Documents the fitted coefficients, intercept, solver, and prediction API used in this experiment.

## 04. Logistic regression: turn scores into probabilities

Lesson ID: `logistic`. Track: Foundations.

### Reading

Use these sources to check the mechanism and extend the exercise. Research results are not results measured by this lab.

- [Logistic regression — scikit-learn](https://scikit-learn.org/1.7/modules/linear_model.html#logistic-regression) — Connects linear scores to class probabilities and regularized training.
- [LogisticRegression API — scikit-learn](https://scikit-learn.org/1.7/modules/generated/sklearn.linear_model.LogisticRegression.html) — Defines C as inverse regularization strength and documents predict_proba.
- [Common pitfalls and recommended practices — scikit-learn](https://scikit-learn.org/1.7/common_pitfalls.html) — Explains why the lab fits its feature scaler only on training examples.

## 05. Decision trees: inspect the questions

Lesson ID: `tree`. Track: Foundations.

### Reading

Use these sources to check the mechanism and extend the exercise. Research results are not results measured by this lab.

- [Decision trees — scikit-learn](https://scikit-learn.org/1.7/modules/tree.html) — Explains recursive feature splits, leaf predictions, and overfitting.
- [DecisionTreeClassifier API — scikit-learn](https://scikit-learn.org/1.7/modules/generated/sklearn.tree.DecisionTreeClassifier.html) — Documents the Gini criterion, depth limit, and tree controls used by the lab.

## 06. Ensembles: compare forests and boosting

Lesson ID: `ensemble`. Track: Foundations.

### Reading

Use these sources to check the mechanism and extend the exercise. Research results are not results measured by this lab.

- [Ensemble methods — scikit-learn](https://scikit-learn.org/1.7/modules/ensemble.html) — Distinguishes forests from sequential boosting and explains forest probability averaging.
- [RandomForestClassifier API — scikit-learn](https://scikit-learn.org/1.7/modules/generated/sklearn.ensemble.RandomForestClassifier.html) — Defines prediction as the class with the highest average probability across trees.
- [HistGradientBoostingClassifier API — scikit-learn](https://scikit-learn.org/1.7/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html) — Documents the histogram boosting implementation and its iteration and leaf limits.

## 07. Nearest neighbors: classify by similarity

Lesson ID: `neighbors`. Track: Foundations.

### Reading

Use these sources to check the mechanism and extend the exercise. Research results are not results measured by this lab.

- [Nearest neighbors — scikit-learn](https://scikit-learn.org/1.7/modules/neighbors.html) — Explains distance-based prediction, neighbor selection, and the effects of dimensionality.
- [KNeighborsClassifier API — scikit-learn](https://scikit-learn.org/1.7/modules/generated/sklearn.neighbors.KNeighborsClassifier.html) — Documents k, vote weighting, distance metrics, and the classifier used here.

## 08. Kernel methods: learn a curved boundary

Lesson ID: `kernel`. Track: Foundations.

### Reading

Use these sources to check the mechanism and extend the exercise. Research results are not results measured by this lab.

- [Support vector machines — scikit-learn](https://scikit-learn.org/1.7/modules/svm.html) — Explains margin classifiers, the radial-basis kernel, and the effects of C and gamma.
- [SVC API — scikit-learn](https://scikit-learn.org/1.7/modules/generated/sklearn.svm.SVC.html) — Documents the linear/RBF kernels and scale-based gamma rule used in the comparison.

## 09. Naive Bayes: combine evidence with probabilities

Lesson ID: `bayes`. Track: Foundations.

### Reading

Use these sources to check the mechanism and extend the exercise. Research results are not results measured by this lab.

- [Naive Bayes — scikit-learn](https://scikit-learn.org/1.7/modules/naive_bayes.html) — Explains conditional independence and warns that classifier probabilities may need calibration.
- [GaussianNB API — scikit-learn](https://scikit-learn.org/1.7/modules/generated/sklearn.naive_bayes.GaussianNB.html) — Documents Gaussian feature likelihoods, priors, and the exact classifier fitted in this lab.

## 10. Gaussian processes: predict with uncertainty

Lesson ID: `gaussian`. Track: Foundations.

### Reading

Use these sources to check the mechanism and extend the exercise. Research results are not results measured by this lab.

- [Gaussian processes — scikit-learn](https://scikit-learn.org/1.7/modules/gaussian_process.html) — Explains conditioning a function prior on observations and predicting uncertainty.
- [GaussianProcessRegressor API — scikit-learn](https://scikit-learn.org/1.7/modules/generated/sklearn.gaussian_process.GaussianProcessRegressor.html) — Documents kernel fitting, target normalization, and returned predictive standard deviations.

## 11. Clustering: discover groups without supplied labels

Lesson ID: `cluster`. Track: Foundations.

### Reading

Use these sources to check the mechanism and extend the exercise. Research results are not results measured by this lab.

- [Clustering — scikit-learn](https://scikit-learn.org/1.7/modules/clustering.html) — Explains K-means, density-based clustering, silhouette, and adjusted Rand index.
- [KMeans API — scikit-learn](https://scikit-learn.org/1.7/modules/generated/sklearn.cluster.KMeans.html) — Documents fitted centers, inertia, and predictions for new observations.
- [DBSCAN API — scikit-learn](https://scikit-learn.org/1.7/modules/generated/sklearn.cluster.DBSCAN.html) — Explains neighborhood density, noise labels, and the separate training-data exploration.

## 12. Principal component analysis: compress measurements

Lesson ID: `projection`. Track: Foundations.

### Reading

Use these sources to check the mechanism and extend the exercise. Research results are not results measured by this lab.

- [Principal component analysis — scikit-learn](https://scikit-learn.org/1.7/modules/decomposition.html#pca) — Explains orthogonal principal directions, centering, and dimensionality reduction.
- [PCA API — scikit-learn](https://scikit-learn.org/1.7/modules/generated/sklearn.decomposition.PCA.html) — Documents retained variance, transform, and inverse_transform used to reconstruct digits.

## 13. Dense neural networks: train a hidden representation

Lesson ID: `dense`. Track: Foundations.

### Reading

Use these sources to check the mechanism and extend the exercise. Research results are not results measured by this lab.

- [Build the Neural Network — PyTorch](https://docs.pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html) — Introduces dense layers, nonlinear activations, and class-score outputs.
- [Linear layer — PyTorch](https://docs.pytorch.org/docs/stable/generated/torch.nn.Linear.html) — Specifies input/output dimensions, learned weights, and bias for affine layers.
- [CrossEntropyLoss — PyTorch](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html) — Explains why the classifier supplies unnormalized logits directly to the loss.

## 14. Convolutional networks: share a visual detector

Lesson ID: `convolution`. Track: Foundations.

### Reading

Use these sources to check the mechanism and extend the exercise. Research results are not results measured by this lab.

- [Conv2d — PyTorch 2.6](https://docs.pytorch.org/docs/2.6/generated/torch.nn.Conv2d.html) — Defines channel dimensions, padding, stride, and spatial filtering for the lab's convolutions.
- [MaxPool2d — PyTorch](https://docs.pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html) — Explains how the 2-by-2 pooling layer reduces each 8-by-8 map to 4-by-4.

## 15. Graph neural networks: learn from connections

Lesson ID: `graph`. Track: Foundations.

### Reading

Use these sources to check the mechanism and extend the exercise. Research results are not results measured by this lab.

- [Semi-Supervised Classification with Graph Convolutional Networks — Kipf and Welling](https://arxiv.org/abs/1609.02907) — Introduces graph convolution that combines node features and local connectivity.
- [GCNConv operator — PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GCNConv.html) — Shows symmetric degree normalization and self-connections; the lab implements the mechanism directly without this package.

## 16. Autoencoders: reconstruct through a bottleneck

Lesson ID: `autoencoder`. Track: Foundations.

### Reading

Use these sources to check the mechanism and extend the exercise. Research results are not results measured by this lab.

- [Autoencoders — Deep Learning, chapter 14](https://www.deeplearningbook.org/contents/autoencoders.html) — Explains encoder/decoder reconstruction, bottlenecks, and limitations of unconstrained copying.
- [Stacked Denoising Autoencoders — Vincent et al.](https://www.jmlr.org/papers/v11/vincent10a.html) — An original research treatment of learning representations by reconstructing clean inputs from corrupted versions.

## 17. Variational autoencoders: learn a distribution

Lesson ID: `variational`. Track: Foundations.

### Reading

Use these sources to check the mechanism and extend the exercise. Research results are not results measured by this lab.

- [Auto-Encoding Variational Bayes — Kingma and Welling](https://arxiv.org/abs/1312.6114) — Introduces the variational objective and differentiable reparameterization used in the lesson.
- [Probability distributions — PyTorch 2.6](https://docs.pytorch.org/docs/2.6/distributions.html) — Explains reparameterized samples, Normal distributions, and divergence operations behind the numerical example.

## 18. Adversarial generation: train two competing models

Lesson ID: `adversarial`. Track: Foundations.

### Reading

Use these sources to check the mechanism and extend the exercise. Research results are not results measured by this lab.

- [Generative Adversarial Networks — Goodfellow et al.](https://arxiv.org/abs/1406.2661) — Introduces joint generator/discriminator training; the lab uses a small one-dimensional teaching model.
- [BCEWithLogitsLoss — PyTorch](https://docs.pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html) — Documents the numerically stable binary objective used for discriminator and generator updates.

## 19. Statistical language models: count before you train a network

Lesson ID: `ngram`. Track: Foundations.

### Reading

Use these sources to check the mechanism and extend the exercise. Research results are not results measured by this lab.

- [N-gram Language Models — Jurafsky and Martin](https://web.stanford.edu/~jurafsky/slp3/3.pdf) — Explains context counts, language-model evaluation, perplexity, and smoothing.
- [Language modeling package — NLTK](https://www.nltk.org/api/nltk.lm.html) — Documents vocabulary handling, n-gram counts, and Laplace add-one smoothing.

## 20. Reinforcement learning: learn actions from rewards

Lesson ID: `reinforcement`. Track: Foundations.

### Reading

Use these sources to check the mechanism and extend the exercise. Research results are not results measured by this lab.

- [Q-learning — Watkins and Dayan](https://link.springer.com/article/10.1007/BF00992698) — The original tabular action-value learning paper; the publisher provides an abstract and publication record.
- [Handling Time Limits — Gymnasium](https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/) — Distinguishes true termination from time-limit truncation and explains when to bootstrap.
- [Reinforcement Learning (DQN) Tutorial — PyTorch](https://docs.pytorch.org/tutorials/intermediate/reinforcement_q_learning.html) — Further reading on Q targets and terminal states; the lab uses a table rather than this tutorial's neural network.

## 21. Build your workspace

Lesson ID: `setup`. Track: Language models.

### Reading

Use these sources to check the mechanism and extend the exercise. Research results are not results measured by this lab.

- [Python 3.12: virtual environments](https://docs.python.org/3.12/library/venv.html) — Explains package isolation, environment creation and activation used by this workspace.
- [PyTorch: install and verify locally](https://pytorch.org/get-started/locally/) — Select the CPU or supported CUDA installation and verify that tensors run on the intended device.

## 22. Run your first model

Lesson ID: `inference`. Track: Language models.

### Reading

Use these sources to check the mechanism and extend the exercise. Research results are not results measured by this lab.

- [SmolLM2-135M-Instruct model card](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct) — Documents this lesson's exact pretrained checkpoint, intended use and limitations.
- [Transformers 4.57.1: chat templates](https://huggingface.co/docs/transformers/v4.57.1/en/chat_templating) — Explains conversation-role formatting and the generation prompt used by the inference code.

## 23. See the numbers underneath

Lesson ID: `math`. Track: Language models.

### Reading

Use these sources to check the mechanism and extend the exercise. Research results are not results measured by this lab.

- [PyTorch 2.9: automatic differentiation mechanics](https://docs.pytorch.org/docs/2.9/notes/autograd.html) — Connects the lesson's loss, backward pass and gradients to the computation graph.
- [Attention Is All You Need — Vaswani et al. (2017)](https://arxiv.org/abs/1706.03762) — Primary source for query, key and value attention, masking and the Transformer block.
- [SmolLM2 configuration at the course's pinned revision](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct/blob/12fd25f77366fa6b3b4b768ec3050bf629380bac/config.json) — Verify the hidden width of 576 used in this lesson's tensor-shape example.

## 24. Measure a baseline

Lesson ID: `baseline`. Track: Language models.

### Reading

Use these sources to check the mechanism and extend the exercise. Research results are not results measured by this lab.

- [scikit-learn: evaluating estimator performance](https://scikit-learn.org/stable/modules/cross_validation.html) — Explains why validation guides choices and the final test must remain independent.
- [Python 3.12: JSON encoding and decoding](https://docs.python.org/3.12/library/json.html) — Supports the structured-output parsing exercise; the lab adds its own stricter label and key contract.

## 25. Teach with a small adapter

Lesson ID: `adapt`. Track: Language models.

### Reading

Use these sources to check the mechanism and extend the exercise. Research results are not results measured by this lab.

- [LoRA: Low-Rank Adaptation of Large Language Models — Hu et al. (2021)](https://arxiv.org/abs/2106.09685) — Primary source for learning low-rank updates while freezing pretrained weights.
- [PEFT 0.17: LoRA configuration](https://huggingface.co/docs/peft/v0.17.0/en/package_reference/lora) — Documents the rank, alpha, target modules and adapter options used by the training implementation.

## 26. Give the model useful notes

Lesson ID: `retrieval`. Track: Language models.

### Reading

Use these sources to check the mechanism and extend the exercise. Research results are not results measured by this lab.

- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks — Lewis et al. (2020)](https://arxiv.org/abs/2005.11401) — Historical foundation for combining retrieved evidence with a generator; this lab uses a simpler lexical retriever.
- [Dense Passage Retrieval for Open-Domain Question Answering — Karpukhin et al. (2020)](https://arxiv.org/abs/2004.04906) — Compare learned dense retrieval with this lesson's token-overlap retrieval; dense retrieval is an extension, not the implemented algorithm.

## 27. Train from random weights

Lesson ID: `scratch`. Track: Language models.

### Reading

Use these sources to check the mechanism and extend the exercise. Research results are not results measured by this lab.

- [Attention Is All You Need — Vaswani et al. (2017)](https://arxiv.org/abs/1706.03762) — Explains the attention and feed-forward components adapted into the tiny causal language model.
- [PyTorch 2.9: TransformerEncoderLayer](https://docs.pytorch.org/docs/2.9/generated/torch.nn.TransformerEncoderLayer.html) — Documents the self-attention layer used with an explicit causal mask in the scratch implementation.

## 28. Make the numbers smaller

Lesson ID: `compression`. Track: Language models.

### Reading

Use these sources to check the mechanism and extend the exercise. Research results are not results measured by this lab.

- [Transformers 4.57.1: bitsandbytes quantization](https://huggingface.co/docs/transformers/v4.57.1/en/quantization/bitsandbytes) — Explains actual checkpoint quantization, hardware support and memory measurement beyond the scalar numerical demo.
- [LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale — Dettmers et al. (2022)](https://arxiv.org/abs/2208.07339) — Shows why real model quantization needs more than changing an integer range and how outliers affect precision.

## 29. Learn from a teacher

Lesson ID: `distillation`. Track: Language models.

### Reading

Use these sources to check the mechanism and extend the exercise. Research results are not results measured by this lab.

- [Distilling the Knowledge in a Neural Network — Hinton et al. (2015)](https://arxiv.org/abs/1503.02531) — Primary source for softened teacher targets and temperature in student learning.
- [PyTorch: pruning tutorial](https://docs.pytorch.org/tutorials/intermediate/pruning_tutorial.html) — Supports the separate pruning extension, including masks and the distinction between zeroing weights and making execution smaller.

## 30. Evaluate a coding model

Lesson ID: `code`. Track: Language models.

### Reading

Use these sources to check the mechanism and extend the exercise. Research results are not results measured by this lab.

- [Qwen2.5-Coder-0.5B-Instruct model card](https://huggingface.co/Qwen/Qwen2.5-Coder-0.5B-Instruct) — Documents the exact coding checkpoint, model size and supported inference interface.
- [Evaluating Large Language Models Trained on Code — Chen et al. (2021)](https://arxiv.org/abs/2107.03374) — Introduces functional correctness evaluation with independently written programming tests and explains sampling-based evaluation.

## 31. Connect images and language

Lesson ID: `vision`. Track: Language models.

### Reading

Use these sources to check the mechanism and extend the exercise. Research results are not results measured by this lab.

- [SmolVLM-256M-Instruct model card](https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct) — Documents the exact vision-language checkpoint and its image-plus-text processing path.
- [Transformers 4.57.1: multimodal chat templates](https://huggingface.co/docs/transformers/v4.57.1/en/chat_templating_multimodal) — Explains why pixels must pass through a multimodal processor rather than being supplied as a filename to a text model.

## 32. Experiment with other families

Lesson ID: `architectures`. Track: Language models.

### Reading

Use these sources to check the mechanism and extend the exercise. Research results are not results measured by this lab.

- [Switch Transformers — Fedus et al. (2021)](https://arxiv.org/abs/2101.03961) — Primary example of sparse expert routing; compare its full training design with the lab's deliberately small top-one router.
- [Mamba: Linear-Time Sequence Modeling with Selective State Spaces — Gu and Dao (2023)](https://arxiv.org/abs/2312.00752) — Distinguishes a selective state-space architecture from the lab's basic diagonal recurrence.
- [Griffin: Mixing Gated Linear Recurrences with Local Attention — De et al. (2024)](https://arxiv.org/abs/2402.19427) — Shows a concrete recurrence-and-attention hybrid and why different hybrid architectures must be described individually.

## 33. Move an experiment to a graphics processor

Lesson ID: `cloud`. Track: Language models.

### Reading

Use these sources to check the mechanism and extend the exercise. Research results are not results measured by this lab.

- [Vast.ai: SSH connection guide](https://docs.vast.ai/guides/instances/connect/ssh) — Supports SSH-key setup, connection commands and forwarding the localhost lab through an SSH tunnel.
- [Vast.ai: billing](https://docs.vast.ai/guides/reference/billing) — Explains prepaid credits, stopped-instance storage charges and why preserving files and ending billable resources are separate steps.

## 34. Prove your engineering decision

Lesson ID: `capstone`. Track: Language models.

### Reading

Use these sources to check the mechanism and extend the exercise. Research results are not results measured by this lab.

- [Model Cards for Model Reporting — Mitchell et al. (2018)](https://arxiv.org/abs/1810.03993) — Provides a framework for reporting intended use, evaluation conditions, limitations and model provenance.
- [The ML Test Score — Breck et al. (2017)](https://research.google/pubs/the-ml-test-score-a-rubric-for-ml-production-readiness-and-technical-debt-reduction/) — Connects the capstone's evidence and monitoring plan to concrete tests for a production handoff.
- [scikit-learn: evaluating estimator performance](https://scikit-learn.org/stable/modules/cross_validation.html) — Explains the untouched final-test boundary and why tuning after inspecting test results invalidates that boundary.

## 35. Pretrained or from scratch: choose the route

Lesson ID: `training-route`. Track: Build and retrieve.

### Reading

Use these sources to check the mechanism and extend the exercise. Research results are not results measured by this lab.

- [PyTorch — Transfer learning tutorial](https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html) — Compare training an output head with fine-tuning the whole network; the tutorial uses images, while this lab uses synthetic operations text.
- [PyTorch — Automatic differentiation](https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html) — Connect the forward computation, gradient calculation and optimizer update to the runnable training loop.
- [LoRA: Low-Rank Adaptation of Large Language Models — Hu et al. (2021)](https://arxiv.org/abs/2106.09685) — Primary source for learning low-rank updates while freezing pretrained weights.

## 36. Deep learning: train and inspect every layer

Lesson ID: `deep-network`. Track: Build and retrieve.

### Reading

Use these sources to check the mechanism and extend the exercise. Research results are not results measured by this lab.

- [PyTorch — Automatic differentiation](https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html) — Connect the forward computation, gradient calculation and optimizer update to the runnable training loop.
- [Rumelhart, Hinton and Williams — Learning representations by back-propagating errors](https://www.cs.toronto.edu/~hinton/absps/naturebp.pdf) — The original backpropagation paper supplies the learning mechanism behind the multilayer network.
- [scikit-learn — Common pitfalls and recommended practices](https://scikit-learn.org/1.7/common_pitfalls.html) — Check how independent partitions and training-only preprocessing prevent data leakage.

## 37. Build a recurrent language model from scratch

Lesson ID: `scratch-recurrent`. Track: Build and retrieve.

### Reading

Use these sources to check the mechanism and extend the exercise. Research results are not results measured by this lab.

- [Cho et al. — Learning Phrase Representations using RNN Encoder–Decoder](https://arxiv.org/abs/1406.1078) — Read the gated recurrent encoder–decoder mechanism underlying this lab’s GRU; the small synthetic task does not reproduce the translation experiment.
- [PyTorch — GRU](https://docs.pytorch.org/docs/stable/generated/torch.nn.GRU.html) — Match reset/update gates, hidden-state shapes and outputs to the recurrent classifier. Use the installed package version for exact API behavior.

## 38. Build a masked-token encoder from scratch

Lesson ID: `scratch-encoder`. Track: Build and retrieve.

### Reading

Use these sources to check the mechanism and extend the exercise. Research results are not results measured by this lab.

- [Devlin et al. — BERT](https://aclanthology.org/N19-1423/) — Understand bidirectional masked-token pretraining. The lab trains a tiny encoder on synthetic text and does not reproduce BERT’s data, objectives or results.
- [PyTorch 2.6 — TransformerEncoderLayer](https://docs.pytorch.org/docs/2.6/generated/torch.nn.TransformerEncoderLayer.html) — Map the attention and feed-forward layers in the exercise to the official module interface.

## 39. Build an encoder–decoder from scratch

Lesson ID: `scratch-sequence`. Track: Build and retrieve.

### Reading

Use these sources to check the mechanism and extend the exercise. Research results are not results measured by this lab.

- [Cho et al. — Learning Phrase Representations using RNN Encoder–Decoder](https://arxiv.org/abs/1406.1078) — Read the gated recurrent encoder–decoder mechanism underlying this lab’s GRU; the small synthetic task does not reproduce the translation experiment.
- [Vaswani et al. — Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Compare the original Transformer encoder–decoder with this lab’s GRU sequence-to-sequence model. Cross-attention is further reading, not a component of the included GRU.

## 40. Compare pretrained, frozen, and fresh models

Lesson ID: `transfer-compare`. Track: Build and retrieve.

### Reading

Use these sources to check the mechanism and extend the exercise. Research results are not results measured by this lab.

- [PyTorch — Transfer learning tutorial](https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html) — Compare training an output head with fine-tuning the whole network; the tutorial uses images, while this lab uses synthetic operations text.
- [Devlin et al. — BERT](https://aclanthology.org/N19-1423/) — Understand bidirectional masked-token pretraining. The lab trains a tiny encoder on synthetic text and does not reproduce BERT’s data, objectives or results.
- [scikit-learn — Common pitfalls and recommended practices](https://scikit-learn.org/1.7/common_pitfalls.html) — Check how independent partitions and training-only preprocessing prevent data leakage.

## 41. Build and audit retrieval-augmented generation

Lesson ID: `rag-pipeline`. Track: Build and retrieve.

### Reading

Use these sources to check the mechanism and extend the exercise. Research results are not results measured by this lab.

- [Lewis et al. — Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) — The landmark retrieval-plus-generation design provides context; this lab uses sparse TF–IDF retrieval and a frozen generator rather than reproducing the paper’s learned retriever.
- [scikit-learn — TfidfVectorizer](https://scikit-learn.org/1.7/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html) — Follow how document tokens become weighted, normalized sparse vectors in this implementation.
- [scikit-learn — cosine_similarity](https://scikit-learn.org/1.7/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html) — Explain why dot products of L2-normalized TF–IDF vectors give the retrieval similarity used here.

## 42. Synthetic data: generate, audit, and compare

Lesson ID: `synthetic-data`. Track: Build and retrieve.

### Reading

Use these sources to check the mechanism and extend the exercise. Research results are not results measured by this lab.

- [Wang et al. — Self-Instruct](https://aclanthology.org/2023.acl-long.754/) — A research example of generating and filtering instruction data. The lab’s deterministic tabular simulator is a different mechanism.
- [Stadler, Oprisanu and Troncoso — Synthetic Data: Anonymisation Groundhog Day](https://www.usenix.org/conference/usenixsecurity22/presentation/stadler) — Understand why calling data synthetic does not by itself establish privacy.
- [Shumailov et al. — The Curse of Recursion: Training on Generated Data Makes Models Forget](https://arxiv.org/abs/2305.17493) — Research on recursive synthetic-data training motivates independent evaluation; the lab tests controlled quality and coverage changes, not a reproduction of model collapse.

## 43. A map of modern language-model engineering

Lesson ID: `modern-map`. Track: Modern language engineering.

### Reading

Use these sources to check the mechanism and extend the exercise. Research results are not results measured by this lab.

- [Vaswani et al. (2017), Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Read the attention equation and original encoder–decoder diagram; distinguish this architecture from the lab’s small causal decoder.
- [Dosovitskiy et al. (2020/2021), An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929) — See how image patches become input sequences; shared attention ideas do not imply identical preprocessing or model architecture.
- [Radford et al. (2022), Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356) — Connect audio representations, text targets and supervision to a concrete cross-domain Transformer example.
- [Chen et al. (2021), Decision Transformer: Reinforcement Learning via Sequence Modeling](https://arxiv.org/abs/2106.01345) — Distinguish return-conditioned trajectory modeling from the online policy-gradient training taught later.

## 44. Build attention and measure its cache

Lesson ID: `attention-cache`. Track: Modern language engineering.

### Reading

Use these sources to check the mechanism and extend the exercise. Research results are not results measured by this lab.

- [Vaswani et al. (2017), Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Read the attention equation and original encoder–decoder diagram; distinguish this architecture from the lab’s small causal decoder.
- [Shazeer (2019), Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150) — Explain why sharing keys and values across query heads changes decoder cache storage and memory bandwidth.
- [Ainslie et al. (2023), GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245) — Relate grouped key/value heads to MHA and MQA; the lab trains from scratch rather than reproducing the paper’s uptraining procedure.
- [Jiang et al. (2023), Mistral 7B](https://arxiv.org/abs/2310.06825) — Read the sliding-window and grouped-query design rationale; compare it with the lab’s eight-position teaching window.

## 45. Give a Transformer position, scale, and gates

Lesson ID: `position-normalization`. Track: Modern language engineering.

### Reading

Use these sources to check the mechanism and extend the exercise. Research results are not results measured by this lab.

- [Su et al. (2021), RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864) — Connect rotations of query/key coordinates to position-dependent attention.
- [Press et al. (2021/2022), Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation](https://arxiv.org/abs/2108.12409) — Compare distance biases with learned position embeddings and distinguish measured extrapolation from a general guarantee.
- [Zhang and Sennrich (2019), Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467) — Compare root-mean-square scaling with mean-centered layer normalization.
- [Shazeer (2020), GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202) — Connect the SwiGLU equation and feed-forward parameter budget to the gated variant.

## 46. Route tokens and account for expert costs

Lesson ID: `expert-routing`. Track: Modern language engineering.

### Reading

Use these sources to check the mechanism and extend the exercise. Research results are not results measured by this lab.

- [Shazeer et al. (2017), Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer](https://arxiv.org/abs/1701.06538) — Separate stored expert capacity from per-token computation and inspect routing and load-balancing costs.
- [Fedus et al. (2021/2022), Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://arxiv.org/abs/2101.03961) — Study top-one routing and balancing; identify the capacity, communication and scale details omitted by the local router.

## 47. Explore latent caches and attention across depth

Lesson ID: `latent-depth`. Track: Modern language engineering.

### Reading

Use these sources to check the mechanism and extend the exercise. Research results are not results measured by this lab.

- [DeepSeek-AI (2024), DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model](https://arxiv.org/abs/2405.04434) — Read multi-head latent attention and list the projection absorption and rotary details omitted by the simple reconstructed cache.
- [Kimi Team et al. (2026), Attention Residuals](https://arxiv.org/abs/2603.15031) — Compare attention over earlier depth representations with the lab’s block-output mixture; distinguish Full and Block Attention Residuals.
- [He et al. (2015/2016), Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) — Establish ordinary residual addition before comparing learned mixtures across depth.

## 48. Compare full tuning and small parameter updates

Lesson ID: `adaptation-comparison`. Track: Modern language engineering.

### Reading

Use these sources to check the mechanism and extend the exercise. Research results are not results measured by this lab.

- [Hu et al. (2021), LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) — Connect frozen base weights, low-rank updates, initialization and adapter parameter counts to the tuning experiment.
- [Dettmers et al. (2023), QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314) — Compare NormalFloat4, double quantization and paged optimizers with the lab’s narrower uniform packed-weight demonstration.
- [Ouyang et al. (2022), Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) — Separate demonstrations, reward-model training and policy optimization; compare human feedback with this lab’s authored synthetic examples.

## 49. Plan and run the optional NormalFloat4 extension

Lesson ID: `quantized-cloud`. Track: Modern language engineering.

### Reading

Use these sources to check the mechanism and extend the exercise. Research results are not results measured by this lab.

- [Dettmers et al. (2023), QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314) — Compare NormalFloat4, double quantization and paged optimizers with the lab’s narrower uniform packed-weight demonstration.
- [Hugging Face PEFT, Quantization guide](https://huggingface.co/docs/peft/en/developer_guides/quantization) — Follow supported low-bit model preparation and adapter attachment, and verify the actual backend configuration.
- [Hugging Face bitsandbytes, Integrations](https://huggingface.co/docs/bitsandbytes/main/integrations) — Check the supported quantization integration and distinguish storage precision from computation dtype and total device memory.

## 50. Learn from preferred and rejected responses

Lesson ID: `preference-learning`. Track: Modern language engineering.

### Reading

Use these sources to check the mechanism and extend the exercise. Research results are not results measured by this lab.

- [Rafailov et al. (2023), Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290) — Trace the preferred/rejected response log-likelihood ratios against a frozen reference and the resulting preference objective.
- [Ouyang et al. (2022), Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) — Separate demonstrations, reward-model training and policy optimization; compare human feedback with this lab’s authored synthetic examples.

## 51. Build and audit a learned reward model

Lesson ID: `reward-learning`. Track: Modern language engineering.

### Reading

Use these sources to check the mechanism and extend the exercise. Research results are not results measured by this lab.

- [Christiano et al. (2017), Deep reinforcement learning from human preferences](https://arxiv.org/abs/1706.03741) — Understand how pairwise preferences train a reward predictor and why optimizing that predictor is distinct from verifying task outcomes.
- [Ouyang et al. (2022), Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) — Separate demonstrations, reward-model training and policy optimization; compare human feedback with this lab’s authored synthetic examples.
- [Rafailov et al. (2023), Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290) — Trace the preferred/rejected response log-likelihood ratios against a frozen reference and the resulting preference objective.

## 52. Move from grid-world actions to generated tokens

Lesson ID: `policy-gradient`. Track: Modern language engineering.

### Reading

Use these sources to check the mechanism and extend the exercise. Research results are not results measured by this lab.

- [OpenAI Spinning Up, Vanilla Policy Gradient](https://spinningup.openai.com/en/latest/algorithms/vpg.html) — Trace the on-policy gradient estimator, advantage baseline and fresh rollout requirement for each policy update.
- [DeepSeek-AI (2025), DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948) — Contrast verifier-driven reasoning research with this lab’s supervised warm start and constrained two-digit task.
- [Ouyang et al. (2022), Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) — Separate demonstrations, reward-model training and policy optimization; compare human feedback with this lab’s authored synthetic examples.

## 53. Compare clipped and group-relative policy updates

Lesson ID: `policy-optimization`. Track: Modern language engineering.

### Reading

Use these sources to check the mechanism and extend the exercise. Research results are not results measured by this lab.

- [Schulman et al. (2017), Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) — Work through the clipped importance-ratio surrogate and distinguish old behavior-policy probabilities from the regularization reference.
- [Shao et al. (2024), DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300) — Read GRPO and compare within-prompt group reward normalization with a learned value baseline.
- [Zheng et al. (2025), Group Sequence Policy Optimization](https://arxiv.org/abs/2507.18071) — Derive the length-normalized sequence importance ratio and sequence-level clipping used by the GSPO teaching objective.

## 54. Choose a benchmark that answers your question

Lesson ID: `benchmark-literacy`. Track: Modern language engineering.

### Reading

Use these sources to check the mechanism and extend the exercise. Research results are not results measured by this lab.

- [Hendrycks et al. (2020/2021), Measuring Massive Multitask Language Understanding](https://arxiv.org/abs/2009.03300) — Inspect subject coverage and evaluation format before selecting a broad knowledge benchmark.
- [Jimenez et al. (2023), SWE-bench: Can Language Models Resolve Real-World GitHub Issues?](https://arxiv.org/abs/2310.06770) — Treat repository state, issue context and executable tests as part of the benchmark protocol.
- [Rein et al. (2023), GPQA: A Graduate-Level Google-Proof Q&A Benchmark](https://arxiv.org/abs/2311.12022) — Compare expert-authored science questions with your application’s actual information needs.
- [Phan et al. (2025), Humanity’s Last Exam](https://arxiv.org/abs/2501.14249) — Inspect benchmark scope and version before interpreting difficult academic questions as general product quality.

## 55. Build a reproducible evaluation harness

Lesson ID: `evaluation-engineering`. Track: Modern language engineering.

### Reading

Use these sources to check the mechanism and extend the exercise. Research results are not results measured by this lab.

- [Chen et al. (2021), Evaluating Large Language Models Trained on Code](https://arxiv.org/abs/2107.03374) — Section 2.1 and Appendix A derive the unbiased pass@k estimator used in the arithmetic harness; the course does not run HumanEval.
- [Liang et al. (2022), Holistic Evaluation of Language Models](https://arxiv.org/abs/2211.09110) — Choose evaluation scenarios and multiple metrics deliberately, and document protocols and limitations instead of treating one score as complete quality.
- [Mitchell et al. (2018/2019), Model Cards for Model Reporting](https://arxiv.org/abs/1810.03993) — Document intended use, evaluation conditions, observed limitations and the evidence behind a deployment decision.

## 56. Calibrate judges and compare preferences

Lesson ID: `judge-audit`. Track: Modern language engineering.

### Reading

Use these sources to check the mechanism and extend the exercise. Research results are not results measured by this lab.

- [Zheng et al. (2023), Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena](https://arxiv.org/abs/2306.05685) — Study position, verbosity and self-enhancement biases and compare them with strict parsing and order-swapped fixtures.
- [Chiang et al. (2024), Chatbot Arena: An Open Platform for Evaluating LLMs by Human Preference](https://arxiv.org/abs/2403.04132) — Compare real crowdsourced pairwise preference evaluation with a tiny author-keyed fixture and illustrative Elo updates.

## 57. Serve a model and measure when tokens arrive

Lesson ID: `streaming-service`. Track: Modern language engineering.

### Reading

Use these sources to check the mechanism and extend the exercise. Research results are not results measured by this lab.

- [Dao et al. (2022), FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135) — Distinguish mathematically exact attention from efficient kernel scheduling; the local Python model does not implement FlashAttention.
- [Kwon et al. (2023), Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180) — Separate cache allocation and serving throughput from cache tensor size; the course does not implement a paged allocator.
- [vLLM, Production Metrics](https://docs.vllm.ai/en/latest/usage/metrics/) — Compare request, prefill, first-token and inter-token metrics and state your client-side measurement boundaries.

## 58. Defend a complete language-model engineering decision

Lesson ID: `modern-capstone`. Track: Modern language engineering.

### Reading

Use these sources to check the mechanism and extend the exercise. Research results are not results measured by this lab.

- [Mitchell et al. (2018/2019), Model Cards for Model Reporting](https://arxiv.org/abs/1810.03993) — Document intended use, evaluation conditions, observed limitations and the evidence behind a deployment decision.
- [Gebru et al. (2018/2021), Datasheets for Datasets](https://arxiv.org/abs/1803.09010) — Document data origin, composition, collection, intended use and limitations alongside the split identity.
- [Liang et al. (2022), Holistic Evaluation of Language Models](https://arxiv.org/abs/2211.09110) — Choose evaluation scenarios and multiple metrics deliberately, and document protocols and limitations instead of treating one score as complete quality.

