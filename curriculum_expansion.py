"""Expanded curriculum source: learning explanations, experiments, and assessments."""
from pathlib import Path
import html
ROOT=Path(__file__).resolve().parent

def topic(id,title,group,summary,case,idea,example,algorithm,technical,commands,inspect,challenge,pitfall,question,options,correct,answer,source,image):
    return dict(id=id,title=title,group=group,summary=summary,case=case,idea=idea,example=example,algorithm=algorithm,technical=technical,commands=commands,inspect=inspect,challenge=challenge,pitfall=pitfall,question=question,options=options,correct=correct,answer=answer,source=source,image=image)

TOPICS=[
topic('linear','Linear regression: learn a weighted prediction','Classical models',
'Predict a number, understand each coefficient, and beat a simple baseline.',
'A service team wants to estimate how long a batch job will take from its record count, number of transformations, and compression setting. An interpretable baseline helps reveal whether a complex model adds value. Our measurements are simulated, not logs from a real company.',
'The prediction is an intercept plus a weighted sum of features. A coefficient answers how the prediction changes when one input rises by one unit while the other inputs stay fixed. Fitting adjusts the coefficients to reduce squared residuals, the differences between predictions and observed values.',
'With intercept 5, record-count coefficient 2, transformation coefficient 3, and compression coefficient 8, input [10, 2, 1] gives 5 + 20 + 6 + 8 = 39. If the measured duration is 43, the residual is 4 and its squared error is 16. Real training fits many examples simultaneously.',
['Generate simulated job measurements with known weights.','Split examples before fitting: 60 percent training, 20 percent validation, 20 percent final test.','Fit a linear model only on training examples.','Predict validation durations and compare with always predicting the training mean.','Inspect recovered coefficients and the actual-versus-predicted plot.'],
'Ordinary least squares minimizes the sum of squared residuals; a numerical solver can fit it without a neural-network training loop. Correlated inputs can make individual coefficients unstable even when predictions are useful. Mean absolute error measures typical absolute mistakes in the target units. Root mean squared error emphasizes larger mistakes. The coefficient of determination compares squared error with a constant prediction and can be negative on held-out data. A fitted coefficient is an association, not proof of causation.',
['python foundations.py linear --noise 8 --output outputs/linear-{run}','python foundations.py linear --noise 25 --output outputs/linear-noisy-{run}'],
'Open predictions.png. Points on the diagonal are perfect predictions. In report.json compare mean_absolute_error with baseline_mean_absolute_error and inspect learned_weights near the data generator’s [2, 3, 8]. Exact recovery is not expected with noise.',
'Predict which metric will increase when measurement noise increases from 8 to 25. Run both commands with seed 42, record the results, and explain why more noise does not necessarily require a deeper network. Then read linear() in foundations.py and change one generating coefficient in a copied experiment.',
'Never estimate preprocessing statistics on the complete dataset before splitting. Do not interpret generated feature names as measured business facts.',
'What does a regression coefficient describe?',['An association with one input, holding others fixed','A guarantee of causation','The number of training examples'],0,'A coefficient describes the fitted relationship under the model assumptions; intervention effects need additional causal evidence.',
'https://scikit-learn.org/1.7/modules/linear_model.html','model-mechanisms.svg'),

topic('logistic','Logistic regression: turn scores into probabilities','Classical models',
'Learn classification, thresholds, class imbalance, and confusion matrices.',
'An operations team may prioritize events for review using measured event features. A predicted probability helps rank the queue. It does not establish that a person or event is malicious. The lab uses generated event vectors so there is no private telemetry.',
'First compute a weighted score. Then use the sigmoid function to map that score into a number between zero and one. Training makes observed labels more probable. A decision threshold converts the probability into a label, and changing that threshold changes the kinds of mistakes.',
'A score of 0 maps to probability 0.5. A score of approximately 2.20 maps to approximately 0.90. A review threshold of 0.80 would flag the second example. The probability is not automatically calibrated to real-world frequencies.',
['Create a binary dataset with unequal class frequencies.','Split while preserving approximately the same class proportions.','Fit the feature scaler on training data inside a pipeline.','Fit logistic regression with regularization.','Predict validation labels and inspect false positives and false negatives.'],
'The loss is negative log likelihood, often called binary cross-entropy. Regularization discourages overly large coefficients. In the library used here, the parameter C is inverse regularization strength: smaller C means stronger regularization. Accuracy alone can hide failure on the less common class. Balanced accuracy averages recall across classes; macro F1 averages each class’s harmonic mean of precision and recall. These metrics answer different questions.',
['python foundations.py logistic --strength 1 --output outputs/logistic-{run}','python foundations.py logistic --strength 0.01 --output outputs/logistic-regularized-{run}'],
'Compare the confusion matrix with the majority-class baseline. Rows are actual classes and columns are predicted classes. A model can beat the baseline but still produce too many false alerts for an operational review queue.',
'Compare training accuracy and validation accuracy under the two regularization settings. In a copied workbench experiment, use predict_proba and compare thresholds 0.2, 0.5, and 0.8. Submit the false-positive and false-negative counts, not only a single accuracy score.',
'Choosing a threshold by inspecting final test labels invalidates that test as an independent final check. Select the threshold using validation data.',
'Does a probability of 0.9 guarantee correctness?',['Yes','No; it is a model estimate that needs calibration and evaluation','Only if accuracy is high'],1,'Probabilities depend on training data and modeling assumptions. Calibration checks whether predicted frequencies match observed frequencies.',
'https://scikit-learn.org/1.7/modules/linear_model.html#logistic-regression','evaluation-guide.svg'),

topic('tree','Decision trees: inspect the questions','Classical models',
'Build a branching classifier and see the cost of increasing depth.',
'A manufacturing team can investigate combinations of temperature, vibration, and operating state associated with defective output. A shallow tree offers a sequence of inspectable questions. Our generated features represent a teaching analogue, not a validated industrial diagnostic.',
'A tree repeatedly partitions examples using questions such as feature three less than a threshold. The training algorithm searches for splits that make the resulting groups more consistent in their labels. A leaf stores the prediction made after following a path.',
'Imagine a node with six normal and four fault examples. A useful split might create one branch with five normal and no faults and another with one normal and four faults. The branches are more informative than the original mixed group.',
['Start with all training examples at the root.','Try feature thresholds and score the resulting label impurity.','Choose a useful split and repeat on each child.','Stop at the configured depth or another stopping rule.','Predict by walking each held-out example from root to leaf.'],
'This implementation uses Gini impurity by default. A pure node has impurity zero. Deeper trees can fit small peculiarities in training data, increasing variance. Limiting depth or pruning constrains that flexibility. Trees do not require the same feature scaling as distance-based methods because each split tests one feature against a threshold. Correlated or proxy features can still make a plausible-looking explanation misleading.',
['python foundations.py tree --depth 3 --output outputs/tree-shallow-{run}','python foundations.py tree --depth 15 --output outputs/tree-deep-{run}'],
'Open tree.png: only the first three displayed levels are plotted to keep the picture readable. Inspect training_accuracy and validation accuracy in the report. A deep tree can achieve excellent training performance while worsening held-out results.',
'Walk one example through the displayed thresholds by hand. Then compare depth 3 and depth 15 using the same split. Submit a short explanation of which model you would retain and what evidence could change your decision.',
'A decision path explains this fitted model’s computation. It does not prove that the chosen feature causes the observed outcome.',
'What is a common risk of a very deep tree?',['It cannot fit training data','It can memorize training-specific patterns','It always ignores numeric features'],1,'Greater depth increases flexibility and may fit noise, so validation performance matters.',
'https://scikit-learn.org/1.7/modules/tree.html','model-mechanisms.svg'),

topic('ensemble','Ensembles: compare forests and boosting','Classical models',
'Combine learners, compare the actual results, and distinguish two different strategies.',
'Tabular prediction appears in inventory planning, service reliability, and transaction review. Tree ensembles are useful candidates when feature interactions matter. The exercise compares three approaches on identical generated event data, rather than claiming one algorithm always wins.',
'A random forest trains many varied trees and aggregates their votes. Gradient boosting builds a sequence of models that improve the current predictive objective. The first strategy reduces dependence on one tree; the second makes successive corrections.',
'If five trees vote [normal, fault, normal, normal, fault], the majority prediction is normal. That simple vote explains a forest. A boosting update instead adds a scaled new function to the current prediction; it is not simply another independent vote.',
['Create one fixed set of training and validation indices.','Fit a single tree as a baseline.','Fit a random forest with randomized trees.','Fit a gradient-boosting classifier on the same training examples.','Compare validation metrics, training accuracy, and the saved bar chart.'],
'Forests use bootstrap sampling and feature randomness to reduce correlation among trees. Boosting follows the gradient of a loss; the popular description of fitting residuals is exact for some regression objectives and an intuition for other losses. More estimators increase compute and can change fit. Feature-importance rankings require careful interpretation when features are correlated. The ensemble experiment’s --depth option controls the baseline tree and forest; the boosting model separately uses a fixed 15-leaf limit.',
['python foundations.py ensemble --depth 3 --output outputs/ensemble-shallow-{run}','python foundations.py ensemble --depth 10 --output outputs/ensemble-deep-{run}'],
'Inspect comparison.png and the nested metrics for single_tree, random_forest, and gradient_boosting. Explain any differences in the training–validation gap. The plotted scores are generated by your run.',
'Choose the simplest candidate meeting your declared accuracy and false-alert requirements. Record fit time and discuss whether a small gain warrants more complex deployment. For another experiment, change tree counts in a copied function while holding the split fixed.',
'Do not compare one model on validation data with another on test data. Also avoid assuming every boosting update trains only on previously misclassified examples.',
'How does gradient boosting differ from an ordinary forest?',['It builds successive corrections to an objective','It must use neural networks','It has no training phase'],0,'Boosting adds models sequentially to improve the current objective; forest trees are trained as varied members to aggregate.',
'https://scikit-learn.org/1.7/modules/ensemble.html','model-mechanisms.svg'),

topic('neighbors','Nearest neighbors: classify by similarity','Classical models',
'Observe how a distance rule and neighbor count reshape a decision boundary.',
'A maintenance application may compare a new sensor reading with previously labeled cases. A nearest-neighbor baseline is useful when similarity has a meaningful definition. The lab’s two-dimensional half-circles make the mechanism visible before applying it to higher-dimensional measurements.',
'Store the training examples. For a new example, measure its distance to those examples, select the nearest k, and aggregate their labels. Most of the work occurs during prediction rather than an iterative parameter-fitting stage.',
'For five nearby cases labeled [fault, normal, fault, fault, normal], the majority label is fault. If one feature is measured in thousands and another in tenths, raw Euclidean distance may be dominated by the large-number feature.',
['Split the generated examples.','Learn feature means and scales from training data only.','Store standardized training examples and their labels.','Find the requested number of nearest training neighbors for each validation example.','Vote and draw the resulting decision regions.'],
'One neighbor can follow noisy local detail; many neighbors smooth the boundary and can erase smaller groups. Scaling is part of the fitted pipeline. In high-dimensional spaces, distances can become less discriminating and search can become expensive. This is a useful bridge to retrieval, but a geometric neighbor vote is different from a language model generating an answer from retrieved documents.',
['python foundations.py neighbors --neighbors 1 --output outputs/neighbors-one-{run}','python foundations.py neighbors --neighbors 25 --output outputs/neighbors-many-{run}'],
'Open boundaries.png. Dots are held-out examples; background color is the fitted prediction across a grid. Compare jagged regions with smoothed regions and relate them to the confusion matrix.',
'Predict which neighbor count will be more sensitive to a mislabeled training point. Run both settings. Then try 5 neighbors while keeping noise and seed fixed, and justify the choice using validation evidence.',
'Similarity is a design decision, not an objective universal property. A representation that groups images well may be poor for semantic document retrieval.',
'Why scale features before distance-based prediction?',['To make all distances zero','To prevent units alone from dominating distance','To reveal validation labels'],1,'Scaling makes feature units more comparable; the scaler must be fitted only on the training subset.',
'https://scikit-learn.org/1.7/modules/neighbors.html','model-mechanisms.svg'),

topic('kernel','Kernel methods: learn a curved boundary','Classical models',
'Compare a linear support vector machine with a nonlinear similarity kernel.',
'Engineers can classify operating regimes whose measurements form nonlinear patterns. A kernel method is a compact candidate for a moderate-size dataset. In this exercise, interleaving half-circles demonstrate why a straight separating line can be insufficient.',
'A support vector classifier seeks a boundary with a useful margin between classes while allowing penalized mistakes. A kernel computes similarities corresponding to an implicit feature representation, allowing a nonlinear boundary in the original coordinates.',
'Two classes arranged as nested circles cannot be separated by one straight line in the plane. A radius-related representation can make separation easier. A radial basis function kernel provides a general smooth similarity mechanism without explicitly constructing every transformed coordinate.',
['Split and standardize the two input features.','Fit a linear-kernel classifier on training data.','Fit a radial-basis-kernel classifier on the same training data.','Evaluate both on the same validation examples.','Inspect the two boundary panels and the error counts.'],
'The C parameter controls the tradeoff between margin and penalized training mistakes. A larger value usually permits a more aggressive fit to training data; it does not guarantee better validation results. The radial basis function kernel’s width controls how local similarity is. Our implementation uses the library’s scale-based width rule. Kernel matrices can become expensive as training-set size grows.',
['python foundations.py kernel --strength 1 --output outputs/kernel-{run}','python foundations.py kernel --strength 100 --output outputs/kernel-flexible-{run}'],
'Compare linear_kernel and radial_basis_kernel metrics. Open boundaries.png and explain why the curved solution is plausible for this particular data geometry.',
'Increase observation noise while preserving the random seed. Compare whether increasing C now helps or merely fits noisy detail. Document a case where the nonlinear model’s additional complexity is justified.',
'Kernel methods are not a special kind of graphics processor. A kernel here is a mathematical similarity function; a hardware computation kernel means something different.',
'What does a nonlinear kernel provide?',['A richer similarity-based decision boundary','A guarantee of perfect unseen predictions','A new physical processor'],0,'A kernel changes the similarities available to the model, allowing a boundary that is nonlinear in the original feature space.',
'https://scikit-learn.org/1.7/modules/svm.html','model-mechanisms.svg'),

topic('bayes','Naive Bayes: combine evidence with probabilities','Classical models',
'Build a probability-based classifier and inspect its independence assumption.',
'A lightweight classifier can triage documents or sensor events when fast fitting and a simple baseline are valuable. Different naive Bayes variants fit different feature types. This lab uses continuous generated measurements and therefore Gaussian naive Bayes; text counts would call for another variant.',
'Estimate how likely each feature value is under each class, combine that evidence with the class prior, then compare class scores. The “naive” assumption treats features as conditionally independent given the class. It is a modeling simplification, not a claim that real features are independent.',
'If the prior chance of a fault is 0.1, then a highly unusual measurement under the normal class can increase the posterior fault probability. Multiplying probabilities requires normalization to obtain a probability distribution across classes. Correlated features can cause evidence to be counted more than once.',
['Estimate training class frequencies.','For each class and feature, estimate a mean and variance.','Compute class scores for held-out measurements using Gaussian feature likelihoods.','Normalize class scores and choose a label.','Compare with logistic regression on the identical split.'],
'Bayes’ rule connects posterior probability, likelihood, and prior. Implementations usually add log probabilities to avoid numerical underflow from multiplying many small values. Gaussian naive Bayes assumes a Gaussian distribution for each continuous feature within each class. Multinomial naive Bayes is often used for nonnegative counts. The laboratory’s generated features include dependencies, making the simplifying assumption visible rather than automatically favorable.',
['python foundations.py bayes --noise 0.05 --output outputs/bayes-{run}','python foundations.py bayes --noise 0.25 --output outputs/bayes-noisy-{run}'],
'Inspect naive_bayes and logistic_comparison in report.json. Which errors differ? Accuracy may remain useful even when the probability estimates are poorly calibrated.',
'Write down what would change if the inputs were word counts rather than continuous measurements. In a copied experiment, duplicate one input column and assess whether redundant evidence changes predictions or confidence.',
'A confident score from a naive Bayes model is not automatically reliable uncertainty. Evaluate calibration before using probabilities for consequential decisions.',
'What is the central naive Bayes assumption?',['Features are conditionally independent given the class','All classes are equally common','Every feature is normally distributed in every variant'],0,'Conditional independence is the common simplifying assumption; distributional choices depend on the variant.',
'https://scikit-learn.org/1.7/modules/naive_bayes.html','evaluation-guide.svg'),

topic('gaussian','Gaussian processes: predict with uncertainty','Classical models',
'Fit a smooth function and inspect an uncertainty band beyond observed inputs.',
'A calibration engineer may have a limited number of expensive measurements and need a smooth estimate between them. A Gaussian process can model a distribution over plausible functions. This exercise uses a synthetic noisy sensor curve, not a safety-critical instrument.',
'The model begins with assumptions about how values at nearby inputs relate. Observed examples update those assumptions into a predictive distribution. Predictions include a mean and uncertainty, allowing us to inspect where the model is less constrained by observations.',
'If the predictive mean is 10 and predictive standard deviation is 2, a Gaussian interval using 1.96 standard deviations is approximately [6.08, 13.92]. This interval is conditional on the model assumptions; empirical coverage must still be checked.',
['Generate noisy sine measurements.','Hold out validation and final test indices.','Choose a smooth covariance function plus a noise component.','Fit using training observations.','Predict means and standard deviations, then plot the interval and measure coverage.'],
'Unlike a fixed-size linear coefficient vector, a standard Gaussian process implementation uses relationships among training examples. Its covariance matrix can require quadratic storage and cubic factorization cost. Kernel choice expresses assumptions such as smoothness. The report’s coverage is the fraction of held-out observations inside the model-based interval, not a universal calibration guarantee. Extrapolation far beyond observed input ranges depends strongly on the prior.',
['python foundations.py gaussian --noise 0.12 --output outputs/gaussian-{run}','python foundations.py gaussian --noise 0.5 --output outputs/gaussian-noisy-{run}'],
'Open uncertainty.png. Compare intervals within the observed range and beyond it. Read interval_coverage and mean_predictive_standard_deviation along with prediction error.',
'Explain why a wider uncertainty interval can be appropriate after increasing measurement noise. Change the smoothness assumption in a copied experiment and compare both error and coverage.',
'Do not equate a narrow interval with truth. Misspecified noise or kernel assumptions can make a model confidently wrong.',
'What does a Gaussian process uncertainty band depend on?',['Only the number of graphics processors','The data and modeling assumptions','A guarantee built into every prediction'],1,'The covariance function, noise assumptions, and observations shape the predictive distribution.',
'https://scikit-learn.org/1.7/modules/gaussian_process.html','evaluation-guide.svg'),

topic('cluster','Clustering: discover groups without supplied labels','Unsupervised learning',
'Compare centroid-based groups with density-based exploration.',
'An engineering team may group incidents by measured behavior to discover recurring patterns before defining categories. Clusters are candidate patterns for investigation; they are not automatically known causes or reliable business segments.',
'K-means alternates between assigning points to nearby centers and moving the centers to their assigned points’ means. Density-based spatial clustering of applications with noise instead grows groups through dense neighborhoods and may leave isolated points unassigned.',
'For points [1, 2, 3] assigned to one one-dimensional cluster, its new center is 2. If two nearby clusters overlap heavily, a geometrically tidy grouping may no longer correspond to the hidden groups that generated the observations.',
['Generate observations with three hidden groups, withholding those labels from fitting.','Fit standardization and k-means on training observations.','Assign validation observations to the fitted centers.','Compute silhouette using geometry and adjusted Rand index using hidden labels only for evaluation.','Fit a separate density-based exploratory view on training observations.'],
'Silhouette compares within-cluster cohesion with separation from other clusters. It can be undefined when only one cluster is represented. Adjusted Rand index compares partitions while adjusting for chance; it requires a reference grouping. The density-based algorithm used here does not expose an ordinary predict method, so its panel is explicitly a training-data exploration. Distances and scaling influence both methods.',
['python foundations.py cluster --clusters 3 --noise 1.2 --output outputs/cluster-three-{run}','python foundations.py cluster --clusters 5 --noise 1.2 --output outputs/cluster-five-{run}'],
'Open clusters.png. Compare heldout_silhouette and heldout_adjusted_rand_index. The label identifiers may be permuted between runs without changing the partition’s meaning.',
'Predict what requesting five clusters does to a dataset generated from three groups. Run both settings. Explain why the best geometric score may not define the most useful operational categories.',
'Unsupervised does not mean assumption-free. The distance measure, cluster count, and density thresholds express strong assumptions about useful structure.',
'Are cluster numbers meaningful business labels?',['Yes, cluster zero always means normal','No; they are arbitrary identifiers requiring interpretation','Only with a larger model'],1,'Clusters need domain interpretation and validation. Their integer identifiers do not supply semantic meaning.',
'https://scikit-learn.org/1.7/modules/clustering.html','split-leakage.svg'),

topic('projection','Principal component analysis: compress measurements','Unsupervised learning',
'Reduce 64 pixel measurements to fewer coordinates and reconstruct the images.',
'A telemetry team may compress correlated measurements for storage or visualization. Principal component analysis can also provide a simple image-compression baseline. This lab uses small handwritten digits so information loss is visible.',
'Find perpendicular directions that capture large amounts of variation in training data. Express each example using its coordinates along the chosen directions. Reconstructing from only some directions discards the remaining variation.',
'If two measurements usually rise together, much of their variation may lie along a diagonal line. One coordinate along that line can preserve more information than simply discarding either measurement. With images, the directions represent patterns across pixels.',
['Load the bundled 8-by-8 digit images and normalize their known pixel range.','Split before learning principal directions.','Fit the requested number of components on training images.','Project validation images and reconstruct them.','Measure reconstruction error and inspect original/reconstructed pairs.'],
'The implementation centers the training data and uses a numerical decomposition to identify principal directions. Explained-variance ratio measures retained training variation, not recognition accuracy. High-variance directions are not necessarily the directions most useful for predicting a label. Scaling choices alter the result, especially when input features have different units.',
['python foundations.py projection --components 4 --output outputs/projection-four-{run}','python foundations.py projection --components 32 --output outputs/projection-thirtytwo-{run}'],
'Compare retained_training_variance with reconstruction_mean_squared_error. In reconstructions.png, original digits are above reconstructions. Use visible details and metrics together.',
'Choose a compression level and justify it using storage dimensions and reconstruction quality. Explain why you would still need a separate downstream classification test before claiming the compressed representation preserves recognition performance.',
'Fitting the principal directions on the whole dataset before the split leaks validation and test information into the representation.',
'Does retaining 95 percent of variance imply 95 percent classification accuracy?',['Yes','No; these measure different properties','Only for images'],1,'Variance preservation measures representation geometry; classification accuracy measures correct labels on a task.',
'https://scikit-learn.org/1.7/modules/decomposition.html#pca','neural-shapes.svg'),

topic('dense','Dense neural networks: train a hidden representation','Neural networks',
'Build a small multilayer perceptron and inspect gradients, shape, and loss.',
'A small neural network can recognize simple visual patterns or combine numerical features when nonlinear relationships matter. Here it recognizes handwritten digits; it is a controlled starting point for understanding a learned representation before language models.',
'A layer computes weighted sums, adds biases, then applies a nonlinear function. Stacking layers allows the model to represent relationships a single linear mapping cannot. Training uses gradients to adjust those parameters according to labeled examples.',
'An 8-by-8 image becomes 64 input numbers. The first layer maps 64 inputs to 32 hidden values; the final layer maps 32 values to 10 class scores. One hidden neuron has 64 weights and one bias. Tensor dimensions describe the structure, not the number of training examples.',
['Load normalized digit pixels and create disjoint splits.','Construct a 64-to-32-to-10 network with a rectified linear unit between the layers.','Compute class scores for a mini-batch.','Compute cross-entropy loss, backpropagate gradients, and take an optimizer step.','Repeat over training epochs and evaluate held-out predictions.'],
'The final outputs are logits, unnormalized class scores. Cross-entropy combines the necessary normalization with a label-based objective; adding a softmax before this loss would be unnecessary here. The optimizer is adaptive moment estimation. Each epoch visits the training examples, while a step updates parameters for one mini-batch. Input shape is batch size by 64. The saved weights require the same network architecture when reloaded.',
['python foundations.py dense --epochs 5 --output outputs/dense-five-{run}','python foundations.py dense --epochs 20 --output outputs/dense-twenty-{run}'],
'Inspect the loss chart and confusion.png. Find digit pairs that are mistaken for one another. Count the trainable parameters and explain the first layer’s contribution.',
'Predict whether additional epochs improve validation accuracy and test that prediction. Read neural_classifier() and change hidden width in a copied model. Compare parameter count and quality, then preserve one final test run for the selected model.',
'A falling training loss does not guarantee that the model generalizes. More neurons are not automatically more useful than better data or a simpler baseline.',
'What does the nonlinear hidden layer make possible?',['Nonlinear input–output relationships','Automatic access to new training data','Guaranteed zero validation error'],0,'A nonlinear activation lets stacked layers express relationships that collapse to one linear mapping without it.',
'https://docs.pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html','neural-shapes.svg'),

topic('convolution','Convolutional networks: share a visual detector','Neural networks',
'Train spatial filters, follow tensor shapes, and compare with a dense network.',
'Visual inspection systems look for local shapes such as edges, scratches, and symbols. A convolutional network reuses filters across positions. Our handwritten-digit task demonstrates this mechanism without pretending to solve factory inspection or full-page document reading.',
'A small filter slides across an image, combining a local neighborhood at each position. The same filter weights are reused everywhere. Learned channels can detect different local patterns. Later layers combine those patterns into a class prediction.',
'For a single-channel 8-by-8 image, eight padded filters produce eight 8-by-8 maps. Pooling reduces them to eight 4-by-4 maps. A second convolution creates sixteen 4-by-4 maps; flattening gives 256 numbers before the ten class scores.',
['Preserve image dimensions instead of treating pixels as an unordered list.','Apply shared convolution filters and a rectified linear activation.','Reduce spatial dimensions with max pooling.','Apply another convolution, flatten, and produce class scores.','Train with cross-entropy and evaluate held-out digits.'],
'Weight sharing means a filter is reused at different locations, reducing parameters compared with a separate connection pattern at every location. Padding controls border behavior. Pooling reduces spatial detail and changes sensitivity to small shifts; it does not create perfect translation invariance. The representation still depends on image scale, position, training distribution, and architecture choices.',
['python foundations.py convolution --epochs 5 --output outputs/convolution-five-{run}','python foundations.py convolution --epochs 20 --output outputs/convolution-twenty-{run}'],
'Compare the confusion matrix, parameter count, and timing with the dense-network lab using the same data split and epoch count. Different architectures can do different amounts of computation per epoch.',
'Draw every tensor shape before reading the code. In a copied experiment shift each validation image by one pixel, define the border fill, and measure how performance changes. Explain why a clean validation score alone does not establish robustness.',
'Changing image resolution requires revisiting the flattened layer’s input size. A shape error is often an architectural mismatch rather than a package-installation failure.',
'What is shared in a convolutional layer?',['The same learned filter weights across positions','The labels of all images','One output value for the whole dataset'],0,'Filters reuse their weights over local neighborhoods at different positions.',
'https://docs.pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html','neural-shapes.svg'),

topic('graph','Graph neural networks: learn from connections','Neural networks',
'Combine node features with neighborhood information and test an edge-removal ablation.',
'Dependency graphs, communication networks, and molecular structures contain relationships that a flat row of features can miss. A graph model can propagate information along those relationships. A graph prediction remains an inference, not proof that an unobserved real-world relationship exists.',
'Each node begins with measured features. A layer combines information from neighboring nodes and applies a learned transformation. Repeating this process gives a node access to information from a larger neighborhood. The graph determines which information can travel.',
'If a service has feature vector [1, 0] and its two neighbors have [0, 1] and [1, 1], a simple unweighted mean is [2/3, 2/3]. Our implementation uses degree-normalized aggregation with self-connections, so its numerical weights follow the graph’s degrees.',
['Generate a synthetic community graph and noisy node features.','Split node labels into training, validation, and test masks.','Add self-connections and normalize the adjacency matrix by node degrees.','Train two aggregation layers using only training-node labels.','Evaluate validation-node predictions and repeat after removing inter-node edges.'],
'This is transductive evaluation: all node features and graph edges are visible during training, but held-out labels are not supplied to the loss. That is a valid declared setting, distinct from predicting on a completely unseen graph. Two layers aggregate information across two steps of connectivity. Dense adjacency is convenient at 180 nodes but scales poorly; large graphs need sparse representations and sampling.',
['python foundations.py graph --epochs 100 --output outputs/graph-{run}','python foundations.py graph --epochs 100 --ablate-edges --output outputs/graph-noedges-{run}'],
'Open adjacency.png and inspect blocks of connections. Compare held-out accuracy with and without edges. The generated communities influence both edges and labels; this is an educational benchmark, not independent field evidence.',
'Explain why the ablation keeps self-connections but removes other links. Submit a diagram identifying which features, edges, and labels are visible to the training procedure. Describe a separate evaluation for unseen organizations or networks.',
'Randomly splitting nodes is not always enough to prevent operational leakage. Time, organizations, shared entities, and how edges were collected can all matter.',
'What makes this evaluation transductive?',['All node features and edges are visible, while some labels are held out','All test labels enter the loss','The model uses no graph'],0,'The held-out nodes are part of the visible graph, but their labels are excluded from training.',
'https://pytorch-geometric.readthedocs.io/en/latest/get_started/introduction.html','neural-shapes.svg'),

topic('autoencoder','Autoencoders: reconstruct through a bottleneck','Generative foundations',
'Learn a compact representation and reconstruct clean inputs from noisy observations.',
'Denoising and representation learning can help process imperfect measurements. An autoencoder can also supply reconstruction-error features for anomaly investigation, but a high or low reconstruction error alone is not a proven anomaly detector.',
'An encoder maps an input into a smaller representation. A decoder maps that representation back to the input shape. Training rewards useful reconstructions. A bottleneck restricts how much information can be carried directly between the two halves.',
'A 64-pixel digit is mapped through 32 hidden values into an 8-number representation, then reconstructed into 64 pixels. The target is the original image, not its digit label. In the denoising version the encoder sees a corrupted input but the decoder is scored against the clean original.',
['Load digits and split examples.','Create an encoder, bottleneck, and decoder.','Add controlled noise to training inputs while retaining clean targets.','Minimize reconstruction error using gradients.','Measure held-out reconstruction and compare with always outputting the training mean image.'],
'This is self-supervised target construction because the clean input supplies the target without requiring human class labels. Reconstruction mean squared error measures pixel differences, not whether a generated image is semantically correct. A large bottleneck can learn an unhelpful near-identity mapping. The training objective is summed pixel error per image; the report’s mean squared error averages across both images and pixels, so their scales differ.',
['python foundations.py autoencoder --latent 4 --noise 0.2 --epochs 25 --output outputs/autoencoder-small-{run}','python foundations.py autoencoder --latent 16 --noise 0.2 --epochs 25 --output outputs/autoencoder-wide-{run}'],
'Open representations.png: original, reconstruction, and noisy input appear in separate rows. Compare reconstruction_mean_squared_error with mean_image_baseline_error.',
'Explain how the bottleneck size changes information capacity. Compare four and sixteen latent dimensions. Propose a genuinely held-out anomaly dataset before attempting to use reconstruction error as an alert score.',
'An ordinary autoencoder does not guarantee that arbitrary random points in its representation space decode into plausible examples.',
'What is the reconstruction target in this denoising lab?',['The clean original input','The digit label','An unrelated image'],0,'The clean input provides the target while the encoder receives a corrupted version.',
'https://www.deeplearningbook.org/contents/autoencoders.html','generative-comparison.svg'),

topic('variational','Variational autoencoders: learn a distribution','Generative foundations',
'Add probabilistic latent variables and balance reconstruction with regularization.',
'A generative representation can support controlled exploration of data variation or serve as a component of a larger generative system. This lab generates tiny handwritten-digit-like images. It is not a privacy-preserving synthetic-data guarantee.',
'The encoder predicts parameters of a distribution over latent values rather than only a single code. Training samples from that distribution, reconstructs the input, and adds a penalty that encourages the latent distribution to stay near a chosen prior.',
'For one latent coordinate, suppose the mean is 2 and standard deviation is 0.5. A sampled standard-normal noise value of -1 gives a latent value of 2 + 0.5 × -1 = 1.5. This separates random noise from differentiable distribution parameters.',
['Encode each training image into a latent mean and log variance.','Draw noise and form a differentiable latent sample.','Decode the sample to reconstruct the input.','Combine reconstruction error with Kullback–Leibler divergence to the standard-normal prior.','Evaluate reconstruction at the latent mean and decode new samples from the prior.'],
'The reparameterization trick lets gradients reach the encoder through the sample construction. Kullback–Leibler divergence measures a directional difference between distributions. The beta parameter here controls the divergence penalty. Our squared reconstruction term corresponds to a simplified Gaussian-likelihood-style objective up to scaling; it is not a full likelihood benchmark. Excess regularization can produce posterior collapse, where latent variables carry little useful input information.',
['python foundations.py variational --latent 8 --beta 0.1 --epochs 25 --output outputs/variational-{run}','python foundations.py variational --latent 8 --beta 2 --epochs 25 --output outputs/variational-strong-{run}'],
'Open representations.png: the bottom row contains samples decoded from the prior. Inspect reconstruction quality separately from sample plausibility. Your samples may be blurry or incomplete.',
'Compare weak and strong divergence penalties. Explain why a better reconstruction score does not necessarily mean better samples from the prior. Record both the numeric error and a written review of sample diversity.',
'Do not call a latent distribution “understanding” without defining a task-based evaluation. Generated examples can preserve undesirable patterns from training data.',
'Why use a reparameterized latent sample?',['To let gradients reach learned distribution parameters','To eliminate randomness','To remove the decoder'],0,'Writing the sample as mean plus scaled noise separates randomness from the parameters being optimized.',
'https://arxiv.org/abs/1312.6114','generative-comparison.svg'),

topic('adversarial','Adversarial generation: train two competing models','Generative foundations',
'Watch a generator learn a two-mode distribution and diagnose mode collapse.',
'Adversarial training has been used for image synthesis and image-to-image translation. The simplest way to understand its dynamics is a one-dimensional target distribution, where missing modes are obvious. This lab learns numbers around -2 and +2 rather than generating full images.',
'A generator maps random noise into samples. A discriminator learns to distinguish training-like samples from generated samples. The generator is updated to make its outputs harder for the discriminator to reject. The two objectives interact, so neither loss is a complete quality score.',
'The target has roughly half its samples near -2 and half near +2. A generator that produces only values near +2 can look locally plausible while completely missing half the target. This is mode collapse, and a histogram makes it easier to see.',
['Draw independent samples from the synthetic two-mode target.','Update the discriminator using real samples and detached generated samples.','Freeze discriminator parameters while updating the generator through its scores.','Alternate these updates for a bounded number of steps.','Draw fresh target and generated samples and compare their distributions.'],
'The discriminator uses binary cross-entropy with logits. Detaching generated samples during its update prevents that loss from changing the generator. Freezing discriminator parameters during the generator update still permits gradients through the discriminator’s computation into the generator. Training can oscillate, collapse, or fail to converge. The saved generator is a tiny demonstration, not an image model.',
['python foundations.py adversarial --steps 200 --output outputs/adversarial-short-{run}','python foundations.py adversarial --steps 1200 --output outputs/adversarial-long-{run}'],
'Open distribution.png. Compare both modes, spread, and fraction_generated_below_zero. A value near one-half does not by itself prove a correct distribution; inspect shape and standard deviation too.',
'Run the longer experiment with a second seed after documenting the first. Explain whether more training improved coverage or caused collapse. Submit a failure analysis even if the generated distribution remains poor.',
'Never mark the lab successful merely because generator loss decreases. A discriminator can become weak, saturated, or uninformative while sample quality remains poor.',
'What is mode collapse?',['Producing only a narrow subset of the target variation','Using too many independent datasets','A required step in every generative model'],0,'A collapsed generator can produce plausible examples from one region while failing to cover other important modes.',
'https://arxiv.org/abs/1406.2661','generative-comparison.svg'),

topic('ngram','Statistical language models: count before you train a network','Language foundations',
'Build an inspectable word predictor with counts, smoothing, and perplexity.',
'Short-context statistical language models provide useful baselines and make next-token prediction concrete. They help engineers distinguish the definition of a language model from the particular neural architecture used by modern assistants.',
'Count how often a word follows a short context in training text. Convert those counts into a probability distribution. At prediction time, use the most recent context words to score or sample the next word. No neural-network parameters are required.',
'If “server” is followed by “starts” three times and “stops” once, the unsmoothed probabilities are 0.75 and 0.25. Add-one smoothing adds one count to every vocabulary option before normalization, reserving probability for events not observed in training.',
['Create distinct short operations sentences and split them.','Build vocabulary and context counts using only training sentences.','Apply add-one smoothing to each context distribution.','Score held-out tokens using their available context.','Compute perplexity and sample a continuation from the learned distribution.'],
'An order-two model uses one previous word; an order-three model uses two. Start and end markers define sentence boundaries. Unknown evaluation words map to an explicit unknown symbol. Perplexity is the exponential of average negative log probability, using natural logarithms here. Lower perplexity indicates better probability assignment only under compatible tokenization, vocabulary handling, and evaluation data.',
['python foundations.py ngram --order 1 --output outputs/ngram-one-{run}','python foundations.py ngram --order 3 --output outputs/ngram-three-{run}'],
'Inspect token-probabilities.png and compare perplexity with uniform_baseline_perplexity. The generated text is sampled from actual counts; it may be awkward because the corpus is small and smoothing spreads probability widely.',
'Hand-calculate one context’s smoothed probability from counts. Compare order one with order three and explain why longer context can create data sparsity instead of automatically improving results.',
'Do not compare this word-level perplexity directly with the byte-level perplexity of the scratch neural language model. The prediction units differ.',
'Why add smoothing?',['To avoid assigning zero probability to every unseen event','To introduce human labels','To make every word equally likely'],0,'Smoothing allocates probability to unseen events; it does not generally make the distribution uniform.',
'https://www.nltk.org/api/nltk.lm.html','model-mechanisms.svg'),

topic('reinforcement','Reinforcement learning: learn actions from rewards','Decision systems',
'Train a complete small action-value learner and compare it with random behavior.',
'Routing, scheduling, and simulated robot control involve actions whose consequences unfold over time. Reinforcement learning studies how a policy can improve from interaction. Our four-by-four grid is an explicit educational environment, not an autonomous real-world control system.',
'The agent observes a state, chooses an action, receives a reward, and sees a new state. A table estimates the value of each action in each state. Updates combine immediate reward with an estimate of future reward. Exploration allows the agent to discover actions it has not yet learned to prefer.',
'If an action currently has value 2, its target is 6, and the learning rate is 0.2, the updated value is 2 + 0.2 × (6 - 2) = 2.8. At the terminal goal, the target uses the immediate reward without adding a future-state value.',
['Initialize a table for sixteen states and four possible actions.','Choose random actions some of the time and otherwise choose the current best action.','Move in the grid, receive a step penalty or goal reward, and update the chosen entry.','Reduce exploration over training episodes.','Evaluate the greedy policy without exploration and compare with random actions.'],
'The update is tabular Q-learning; Q denotes an action-value function, not an acronym. The discount factor controls how future rewards contribute. An episode ends at the goal or after the step limit. The environment is deterministic and all evaluation episodes begin at the same state, so repeated success does not establish generalization to new maps. A neural value function is an extension, not a requirement for reinforcement learning.',
['python foundations.py reinforcement --episodes 50 --output outputs/reinforcement-short-{run}','python foundations.py reinforcement --episodes 500 --output outputs/reinforcement-long-{run}'],
'Open rewards.png. Training rewards include exploration, while greedy_success_rate is measured without it. Interpret greedy_policy using the report’s action_order. Compare greedy_mean_return with random_mean_return.',
'Trace a greedy path from the upper-left corner to the lower-right goal. Explain the terminal-state update. In a copied environment introduce an obstacle and evaluate the old policy before retraining, documenting what fails.',
'Reward design can encourage unintended behavior. A large reward score is only useful when the environment and reward represent the desired task faithfully.',
'Should a terminal transition bootstrap from a future value?',['Yes, always','No; the episode has ended','Only when a graphics processor is available'],1,'A terminal transition has no subsequent decision in the episode, so its target contains only the terminal reward.',
'https://docs.pytorch.org/tutorials/intermediate/reinforcement_q_learning.html','evaluation-guide.svg')
]

FLOW={
'linear':'flowchart TD\n A[Training measurements] --> B[Fit weights and bias]\n B --> C[Predict held-out durations]\n D[Training mean baseline] --> E[Compare errors]\n C --> E\n E --> F[Inspect coefficients and residuals]',
'logistic':'flowchart TD\n A[Input features] --> B[Training-fitted scaler]\n B --> C[Weighted score]\n C --> D[Class probability]\n D --> E{Above threshold?}\n E -->|Yes| F[Review]\n E -->|No| G[Do not flag]',
'tree':'flowchart TD\n A[Feature vector] --> B{Feature three below threshold?}\n B -->|Yes| C{Feature one below threshold?}\n B -->|No| D[Leaf prediction]\n C -->|Yes| E[Leaf prediction]\n C -->|No| F[Leaf prediction]',
'ensemble':'flowchart TD\n A[One training split] --> B[Single tree]\n A --> C[Randomized trees]\n A --> D[Sequential corrections]\n C --> E[Aggregate votes]\n B --> F[Shared validation comparison]\n E --> F\n D --> F',
'neighbors':'flowchart TD\n A[New features] --> B[Training-fitted scaling]\n C[Stored training examples] --> D[Compute distances]\n B --> D\n D --> E[Select nearest examples]\n E --> F[Vote and inspect neighbors]',
'kernel':'flowchart TD\n A[Scaled training features] --> B[Linear similarities]\n A --> C[Radial basis similarities]\n B --> D[Fit margin classifier]\n C --> E[Fit margin classifier]\n D --> F[Compare held-out boundaries]\n E --> F',
'bayes':'flowchart TD\n A[Training labels] --> B[Class priors]\n C[Training features] --> D[Class-conditional likelihoods]\n B --> E[Combine log probabilities]\n D --> E\n F[New measurements] --> E\n E --> G[Normalize and predict]',
'gaussian':'flowchart TD\n A[Covariance assumptions] --> C[Condition on observations]\n B[Training measurements] --> C\n C --> D[Predictive mean]\n C --> E[Predictive uncertainty]\n D --> F[Check held-out error and coverage]\n E --> F',
'cluster':'flowchart TD\n A[Unlabeled training features] --> B[Fit scaler]\n B --> C[Assign points to centers]\n C --> D[Move centers]\n D -->|Repeat| C\n D --> E[Assign held-out points]\n E --> F[Evaluate geometry and meaning]',
'projection':'flowchart TD\n A[Training images] --> B[Fit principal directions]\n B --> C[Keep selected components]\n D[Held-out images] --> E[Project into components]\n C --> E\n E --> F[Reconstruct pixels]\n F --> G[Measure lost information]',
'dense':'flowchart TD\n A[64 pixel inputs] --> B[32 hidden values]\n B --> C[Nonlinear activation]\n C --> D[10 class scores]\n E[Training digit label] --> F[Cross-entropy loss]\n D --> F\n F -->|Gradients update weights| B',
'convolution':'flowchart TD\n A[Image tensor] --> B[Shared local filters]\n B --> C[Pooling]\n C --> D[Second filter layer]\n D --> E[Class scores]\n F[Training labels] --> G[Loss and gradients]\n E --> G\n G -->|Update filters| B',
'graph':'flowchart TD\n A[Node features] --> C[Normalized neighbor aggregation]\n B[Edges and self-connections] --> C\n C --> D[Learned transformation]\n D --> E[Second aggregation]\n E --> F[Node class scores]\n G[Training-label mask] --> H[Training loss]\n F --> H',
'autoencoder':'flowchart TD\n A[Clean training image] --> B[Add noise]\n B --> C[Encoder]\n C --> D[Small representation]\n D --> E[Decoder]\n E --> F[Reconstruction loss]\n A --> F\n F -->|Update both halves| C',
'variational':'flowchart TD\n A[Input image] --> B[Encoder mean and variance]\n B --> C[Sample latent representation]\n D[Random noise] --> C\n C --> E[Decode reconstruction]\n E --> F[Reconstruction term]\n B --> G[Prior-divergence term]\n F --> H[Combined training objective]\n G --> H',
'adversarial':'flowchart TD\n A[Random noise] --> B[Generator]\n B --> C[Generated samples]\n D[Real samples] --> E[Discriminator]\n C --> E\n E --> F[Discriminator update]\n E --> G[Generator update through scores]\n G --> B',
'ngram':'flowchart TD\n A[Training sentences] --> B[Vocabulary and counts]\n B --> C[Smoothed distributions]\n D[Recent context words] --> E[Select context distribution]\n C --> E\n E --> F[Score next word]\n E --> G[Sample next word]',
'reinforcement':'flowchart TD\n A[Observe state] --> B{Explore?}\n B -->|Yes| C[Random action]\n B -->|No| D[Best table action]\n C --> E[Environment transition]\n D --> E\n E --> F[Reward and next state]\n F --> G[Update action value]\n G --> A'
}

METHODS='''# Learning signals: four ways to create a training objective

**Tag: Concept.** This is a reasoning and design activity. The linked implementation labs provide executable examples.

A model architecture describes the calculation. A learning method describes where the training signal comes from. A task describes what you want to accomplish. Keeping these separate prevents confusing a Transformer with reinforcement learning or a deployment format with a model family.

![Four separate axes for classifying a machine learning system](images/expanded/learning-map.svg)

## Supervised learning

A training example includes an input and a supplied target. A classifier might receive sensor measurements and a technician-confirmed condition. Regression predicts a quantity; classification predicts a category. The target’s quality, collection process, and timing determine what can actually be learned. A label obtained after an incident must not accidentally appear as an input available before the incident.

**Activity:** write an input-and-target contract for event triage. Identify when each feature exists and who supplies the target. Then run logistic regression and the decision-tree lab. Explain how the same labels can train different architectures.

## Self-supervised learning

Construct the target from the data itself. In next-token prediction, the input prefix supplies context and the following token supplies the target. In denoising, corrupt an example and reconstruct the original. Human category labels are not required for those targets, but preprocessing, data selection, and objective choices still require engineering.

**Activity:** take the sentence “the server starts safely.” Write every prefix and its next-token target. Then identify which part of the autoencoder lab’s code creates corrupted input and which part retains the clean target. Explain why using next-token prediction during training does not mean a model learns from every conversation after deployment.

## Unsupervised learning

Look for structure without supplied task labels, as in clustering and principal component analysis. There is still an objective: compact clusters or useful directions of variation. Self-supervised learning is often discussed under a broad unsupervised umbrella, but separating explicit target construction is pedagogically useful. The categories are conventions rather than mutually exclusive laws.

**Activity:** propose three possible ways to group support incidents: wording, root cause, and affected product. Explain why distance in one representation may group the first well while failing on the other two. Run clustering and inspect whether a numeric score answers your business question.

## Reinforcement learning

Actions affect future observations and rewards. The learner improves a policy using interaction, rather than receiving a correct label for every possible action. Exploration and delayed consequences matter. The grid lab is a complete tiny implementation; a production control system introduces much more demanding evaluation and operational constraints.

**Activity:** specify states, actions, transitions, reward, and episode termination for a simulated job scheduler. Identify one reward that could encourage an undesirable shortcut. Then trace a terminal Q-learning update by hand.

```mermaid
flowchart TD
 A[Define the problem] --> B{Where is the signal?}
 B --> C[Supplied target]
 B --> D[Target constructed from input]
 B --> E[Structure objective]
 B --> F[Reward after action]
 C --> G[Supervised training]
 D --> H[Self-supervised training]
 E --> I[Unsupervised training]
 F --> J[Reinforcement learning]
```

## Assessment evidence

Submit four small problem specifications, one per learning signal. For each, name the model candidate, input, objective, evaluation split, and deployment constraint. Reject the statement “reinforcement learning is a type of Transformer” with a concrete counterexample from the grid lab.
'''

SELECTION='''# Choosing a model and building an honest experiment

**Tag: Concept.** Complete the design worksheet before investing in a larger training run.

A useful engineering question is “What is the simplest model that meets a measured requirement?” It is not “Which model name is most impressive?” A model family is a candidate mechanism; representative data and task-specific evaluation decide whether it works.

## A practical selection algorithm

1. State the output: number, category, grouping, representation, generated content, or action.
2. Identify the data structure: rows of measurements, images, sequences, a graph, or interactive observations.
3. State the available learning signal and how it was collected.
4. Create a trivial baseline: training-mean prediction, majority class, random action, or a uniform token distribution.
5. Select one simple candidate and one justified alternative. Use the smallest representative experiment that distinguishes them.
6. Freeze a split strategy that respects time, people, documents, equipment, or organizations when those create dependence.
7. Fit preprocessing and the model using training data. Use validation for settings. Reserve final test data for the decision you will report.
8. Measure quality, latency, memory, and operating cost. Review individual failures.
9. Deploy only after checking the actual operating setting. Monitor drift and define a rollback criterion.

![Separate fitting, selection, and final evaluation](images/expanded/split-leakage.svg)

## Match a mechanism to the evidence

| Situation | Useful baseline | Candidate extension | Question that determines the choice |
|---|---|---|---|
| Predict a numeric quantity from a few measurements | Training mean and linear regression | Boosted trees or a neural network | Does nonlinear structure improve held-out error enough to justify complexity? |
| Classify tabular records | Majority class and logistic regression | Tree ensembles or a kernel classifier | What false-positive and false-negative rates are acceptable? |
| Group unlabeled records | Manual sample review | Clustering with a chosen representation | Are the resulting groups operationally useful? |
| Compress measurements | Principal component analysis | Autoencoder | Does the compact representation preserve the downstream task? |
| Classify images | A simple pixel-based classifier | Convolutional network | Does spatial structure help under realistic image variation? |
| Predict using relationships | Features-only classifier | Graph neural network | Are the graph edges available without leakage at prediction time? |
| Generate or transform language | Template or statistical baseline | Small pretrained language model | Is free-form generation needed, and how will correctness be measured? |
| Choose sequential actions | A rule-based or random policy | Reinforcement learning | Can you simulate and evaluate consequences faithfully? |

These are candidate comparisons, not automatic winners. A stronger baseline often reveals that the data contract or evaluation was the real problem.

## An architectural flow for a deployed predictor

```mermaid
flowchart TD
 A[Versioned raw records] --> B[Split by the deployment unit]
 B --> C[Training-only preprocessing]
 C --> D[Fit candidate models]
 D --> E[Validate and select]
 E --> F[One final test]
 F --> G{Meets acceptance criteria?}
 G -->|Yes| H[Package model and preprocessing]
 G -->|No| I[Revisit data or approach]
 H --> J[Monitor operating behavior]
 J -->|Drift or failure| I
 I --> D
```

## Real-world example: a service-time predictor

A team wants a duration estimate before a job begins. A record contains input size, requested transformation count, and eventual duration. Including final output size might be leakage if that value is not available when the prediction must be made. Randomly splitting repeated runs of the same customer job might also make validation unrealistically easy. A better deployment-aligned design could hold out later time periods or entire job templates. The correct choice depends on where the model will be used.

## Assessment contract

Before running a model, write: “For this dataset and deployment setting, I will accept the candidate if [quality metric], [latency], and [resource limit] meet [thresholds], with no known leakage.” Choose thresholds for the task, not after seeing the final test score. Include a baseline comparison and three failure cases in the report. A tiny demonstration proves that you can implement and explain a mechanism; it does not certify a production system.
'''

def module_markdown(t):
    from project_builds import build_section
    algorithm='\n'.join(f'{i+1}. {v}' for i,v in enumerate(t['algorithm']))
    commands='\n\n'.join('```text\n'+c+'\n```' for c in t['commands'])
    scaling='For this exact bundled exercise, use your local central processing unit. Renting a graphics processing unit is unnecessary.'
    if t['id'] in ('dense','convolution','graph','autoencoder','variational','adversarial'):
        scaling+=' For substantially larger images, graphs, or models, first record the local runtime and memory. Use the Vast.ai guide only when the measured limit justifies a rental, and pass --device cuda to use the remote graphics processor. More expensive hardware does not fix a flawed objective or split.'
    else:scaling+=' The implementation uses ordinary processor-based scikit-learn or small tables. Renting a graphics processor will not automatically accelerate this code.'
    return f'''# {t['title']}

**Tag: Runnable implementation lab.** Scope: an inspectable teaching implementation with real fitting, evaluation, and generated artifacts. This is not a production deployment certification.

**Learning outcome:** {t['summary']}

## Where this appears in the real world

{t['case']}

## Build the mental model

{t['idea']}

### Work through a concrete example

{t['example']}

![Visual explanation for {t['title']}](images/expanded/{t['image']})

## Follow the algorithm

{algorithm}

```mermaid
{FLOW[t['id']]}
```

## Go one level deeper

{t['technical']}

## Run the implementation

Open the Practice tab. Install the foundation packages once if the site requests them. The runner executes the actual program in lab/foundations.py. The commands below are for the website’s guided terminal; it selects your Python environment automatically. For external PowerShell, replace python with & .\\.venv\\Scripts\\python.exe and use .\\lab\\foundations.py from the website folder. Replace {{run}} with a unique name when typing commands manually; the website buttons generate it automatically.

{commands}

The random seed controls the generated data, splits, and applicable model initialization. Keep it fixed when investigating one changed parameter. Neural results can still vary across hardware and library implementations.

## Inspect the evidence

{t['inspect']}

Every completed run writes report.json with the experiment, split, seed, settings, dataset description, measured metrics, timing, and limitations. Where applicable it also saves a trained parameter dictionary. The Experiment results page shows the measured report and generated plots. Record an explanation of the result instead of copying only the score.

## Change one thing and prove what happened

{t['challenge']}

## Debugging checkpoint

{t['pitfall']}

If the program reports a missing module, install foundation packages in the runtime shown by the website. If it refuses an existing output directory, choose a new one; this protects earlier evidence. If a numerical result surprises you, inspect data shape, split sizes, and the baseline before changing model complexity.

## Compute plan

{scaling}

{build_section(t)}

## Submit an engineering assessment

Save the two reports, their visual artifacts, a prediction made before the second run, and a paragraph explaining the observed difference. State the dataset’s limits and what additional evidence a real deployment would need. Use validation to choose settings; for a final assessment, rerun the selected settings with --split test and a new output folder. The adversarial and reinforcement labs describe their special evaluation protocols explicitly rather than claiming an ordinary random test split.

## Primary reference

[Read the primary documentation or research]({t['source']}). The lesson’s scenario and simplified implementation are teaching examples.
'''

def additions(render,start):
    entries=[]
    for id,title,body,q,options,answer,explanation in [
        ('learning-signals','Understand how models learn',METHODS,'Is reinforcement learning a fixed neural architecture?',['Yes','No, it describes learning through actions and rewards','Only for large models'],1,'A table, tree, or neural network can be part of a reinforcement-learning system.'),
        ('model-selection','Choose a model and design the evidence',SELECTION,'Which data should fit the preprocessing parameters?',['All data before splitting','The training subset','Only the final test set'],1,'Training-only fitting keeps validation and test information outside the learned preprocessing.')]:
        entries.append({'id':id,'number':start+len(entries),'title':title,'description':'Design activity with real-world examples, diagrams, and an evidence worksheet.','topic':'Engineering foundations','track':'Foundations','kind':'concept','scope':'Design and explanation activity','html':render(body),'commands':[], 'quiz':{'question':q,'options':options,'correct':answer,'explanation':explanation}})
    for t in TOPICS:
        body=module_markdown(t)
        entries.append({'id':t['id'],'number':start+len(entries),'title':t['title'],'description':t['summary'],'topic':t['group'],'track':'Foundations','kind':'implementation','scope':'Local teaching implementation','needs':'foundations','html':render(body),'commands':t['commands'],'source':'foundations.py','quiz':{'question':t['question'],'options':t['options'],'correct':t['correct'],'explanation':t['answer']}})
    return entries

def atlas_markdown():
    rows='\n'.join(f"| {t['title']} | Runnable implementation lab | {t['summary']} |" for t in TOPICS)
    return f'''# The broader machine learning atlas

This atlas maps the areas implemented in this course. It is not a claim to enumerate every research architecture. Model mechanisms, learning signals, tasks, and deployment techniques overlap. A graph network can learn from supervised targets; a Transformer can learn from self-supervised targets and later from rewards.

**Language models are machine-learning models.** The Foundations and Language models tracks are two learning paths within the same field. Count-based word predictors learn from data without neural networks; modern neural language models usually use deep learning. Language modeling describes what is modeled, while a Transformer describes how computation works.

![Language models belong within machine learning, with count-based and neural approaches](images/expanded/language-within-ml.svg)

Open the [Language model atlas](#ref/families) for the full explanation, algorithms, overlapping design choices, and a classification exercise. A classifier that labels messages as spam is also machine learning, but a spam label alone does not make it a language model. Natural language processing is the broader application area for working with human language.

![Classify a system along four separate axes](images/expanded/learning-map.svg)

## Coverage you can verify

**Concept** means a guided explanation, diagram, decision exercise, and comprehension check. **Runnable implementation lab** means the package includes executable code and a practice action. The scope text explains whether it is a small mechanism demonstration, evaluation of a pretrained model, or actual training. No label claims production readiness.

| Topic | Status | What you will do |
|---|---|---|
{rows}

The earlier language-model track retains its dense Transformer, mixture-of-experts, state-space, and hybrid implementations. The Build and retrieve track adds runnable recurrent language modeling, masked-token encoder pretraining, and recurrent encoder–decoder training from random weights. It also adds a pretrained-versus-scratch comparison and retrieval evaluation with actual frozen generation. Diffusion-language modeling remains **Concept** material without a standalone training implementation. Coding and vision lessons run pretrained models; they do not train production code or vision-language models from scratch.

## A compact map of modeling choices

```mermaid
flowchart TD
 A[Predict from examples] --> B[Weighted or probabilistic relationships]
 A --> C[Partitions or similarity]
 A --> D[Learned neural representations]
 B --> E[Linear models and Gaussian processes]
 C --> F[Trees and nearest neighbors]
 D --> G[Dense and convolutional networks]
 D --> H[Graphs and sequence models]
 I[Choose a training signal] --> A
 J[Define the deployment task] --> A
```

The drawing is a teaching map of representative choices, not an exclusive mathematical taxonomy. Unsupervised methods and generative objectives introduce additional choices described in the modules below.
'''

def write_book():
    book='# Machine Learning Super Lab: Expanded Foundations\n\nA companion to the local website. All commands run locally unless explicitly marked remote Linux.\n\n'+METHODS+'\n\n'+SELECTION+'\n\n'+'\n\n'.join(module_markdown(t) for t in TOPICS)
    for appendix in ('Concept-Build-Plans.md','Foundation-Glossary.md'):
        book+='\n\n'+(ROOT/'course'/appendix).read_text(encoding='utf-8')
    (ROOT/'course/Expanded-Foundations.md').write_text(book,encoding='utf-8')
