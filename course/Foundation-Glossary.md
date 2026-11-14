# Additional terms used in the foundation builds

| Short form or term | Expanded name | Meaning in this lab |
|---|---|---|
| ML | Machine learning | Fitting a prediction or decision procedure from examples or experience. |
| CPU | Central processing unit | The general-purpose processor; sufficient for all 18 foundation builds. |
| GPU | Graphics processing unit | A processor suited to parallel tensor operations; optional for larger neural workloads. |
| MLP | Multilayer perceptron | A feed-forward neural network with dense layers and nonlinear activation functions. |
| CNN | Convolutional neural network | A network that shares learned filters over spatial positions. |
| GNN | Graph neural network | A network that combines node features with information along graph connections. |
| RNN | Recurrent neural network | A network that updates a hidden state across sequence positions. |
| LSTM | Long short-term memory | A recurrent architecture with gated updates and a cell state. |
| GRU | Gated recurrent unit | A recurrent architecture that uses gates to control state updates. |
| AE | Autoencoder | An encoder and decoder trained to reconstruct inputs. |
| VAE | Variational autoencoder | A latent-variable model trained using a reconstruction term and a divergence term. |
| GAN | Generative adversarial network | A generator and discriminator trained through competing objectives. |
| PCA | Principal component analysis | A linear projection onto directions of high training-data variance. |
| SVM | Support vector machine | A margin-based model that can use a kernel to represent nonlinear similarity. |
| SVC | Support vector classification | The classification form of a support vector machine. |
| RBF | Radial basis function | A distance-based similarity function used by the demonstrated kernel methods. |
| DBSCAN | Density-based spatial clustering of applications with noise | Groups dense regions and can label isolated observations as noise. |
| MSE | Mean squared error | The average squared prediction error. |
| MAE | Mean absolute error | The average absolute prediction error, in the target's units. |
| RMSE | Root mean squared error | The square root of mean squared error, emphasizing larger mistakes. |
| KL divergence | Kullback–Leibler divergence | A directional measure of mismatch between probability distributions. |
| F1 score | F-one score; a metric name, not an acronym | The harmonic mean of precision and recall; macro averaging weights each class equally. |
| SSH | Secure Shell | The encrypted remote connection used for the cloud lab. |
| SCP | Secure copy | File transfer over a Secure Shell connection. |
| JSON | JavaScript Object Notation | The structured text format for reports and model inputs. |
| API | Application programming interface | A defined interface for requesting a computation or data. |
| ReLU | Rectified linear unit | An activation that returns zero for a negative input and the input for a positive input. |
| Q value | Action-value estimate; Q is a conventional symbol | Expected discounted return associated with a state and action under the relevant policy assumptions. |
| Epoch | One pass through training examples | Different from one optimizer step; an epoch normally contains several mini-batch updates. |
| Inference | Use of an already fitted model | Generates a prediction without updating learned parameters. |
| Ablation | Controlled removal of a component | Tests what changes when, for example, graph edges are removed. |
| Transductive evaluation | Evaluation on held-out labels within a known dataset or graph | The graph lab can see all node features and edges but not held-out labels during training. |
| Inductive evaluation | Evaluation on previously unseen examples or structures | A new graph requires a separate evaluation beyond the lab's fixed graph. |
| Calibration | Agreement between estimated probabilities and observed frequencies | A score of 0.9 is useful as a probability only if such events occur at a compatible frequency. |
| Data drift | Change in the input distribution | It can undermine a previously useful model and requires investigation. |
| Leakage | Evaluation information entering fitting or model selection improperly | Includes fitting preprocessing on test data or splitting near duplicates across sets. |

scikit-learn, SciPy, Matplotlib, PyTorch, joblib, and Mermaid are software project names. Code keeps their official identifiers. An identifier in a code block is not an instruction to rename the package to its explanation.
