# generative_modelling_with_multivariate_gaussians

This repo intends to walkthrough a use case for generative modelling with multivariate gaussians, specifically by using a variational autoencoder. Data used for explanation - [The MNIST database of handwritten digits](https://www.tensorflow.org/datasets/catalog/mnist), which was chosen for visual purposes, to help explain the concept of this project as effectively as possible.
Using a variational autoencoder has significant benefits, like quantifying local changes in the feature space and stress testing. Due to the reliability of results and ease of configuration this should be a preliminary step in most investigations to gain a deeper understanding of the feature space being worked with. 


#### Interactive versions of the static plots can be found here - [Entry point to GitHub Page](https://tobycassidy.github.io/generative_modelling_with_multivariate_gaussians/)

---
## Introductory Theory - KL Divergence with Autoencoders
---

TBC 

---
## Benefits
---
### 1. Quantifying Local Change 
Why it is useful?

Being able to quantify local changes in a feature space is a very important aspect in ML, particularly for explainability. The aspect of a 'local change' in a feature space is a relatively straight forward task for more traditional feature spaces and thus the use of more traditional models (linear regressions, logistic regressions, tree-based methods).  Typically one could self-perturb a feature and see the impact on the results (whether this is supervised or unsupervised) or use a similar concept under the hood, yet in a more assisted way, such as obtaining the shap values with library support. 

However, these methods begin to break down with other techniques. For example, take the case where you are streaming unstructured data and have hypothsised that processing this data as graph objects might be the best way to go. What is the concept of a 'local change' to a graph and how can we quantify it (quantifying being important, as we need some measure for the degree of how local the change is)? One could suggest that graphs A and B are similar if they are one node different, A has one more/less node than B, similar meta data atrributed to each node etc. All sensisble suggestions, yet still hypothesis driven. This is sub-optimal - if graph A has one node different to B, the position of the node that is different could vary the degree of how local the change is (which we aren't capturing) and also in our hypothesis we may miss an idea entirely. The variational autoencoder handles all of this for us by trying to minimize a reconstruction loss whilst being penalized with a KL-divergence loss on the latent space.

If we refer to the figure below and imagine each digit as a graph to coincide with our example above and take a local patch of the latent space as shown. It is visually straight forward to see, that a graph in the centre of the patch (digit 7) has neighbouring points as slight perturbations away from this state, maybe small changes in the meta data attributed to nodes until we reach the bottom left of the patch or top right (digit 9 or digit 1 respectively). In application, one would be able to quantify these local pertubations and compare the degree of how similar / dissimilar they are. Naturally overlaying the goal of the experiment (e.g. probability of churn or cluster being assigned) onto this latent space, baked into the aspect of local changes would draw out the explainability desired.

This use then generalizes, not solely to our example of graph objects but to more fundamental principles, such as explaining any neural networks predictions or a standard approach to feature understanding. 

An important point to note when examining local changes is not to make conclusions on a global basis unless the majority of the local changes do in fact sum to a global change. For example, examining patch i and observing x and then concluding x is true on a global level is wrong. Whereas summing over i patches and observing x, gives more confidence in concluding x is global.


![local_changes](concepts/local_changes.png)

### 2. Stress testing 
Extreme vaues in feature space
What happens to y when x reaches a value we haven't seen yet?
![stress_testing](concepts/stress_testing.png)

### Add on comments (Other techniques, e.g. GANS)



---
## Usage
---
### 1. Data 
[The MNIST database of handwritten digits](https://www.tensorflow.org/datasets/catalog/mnist)

### 2. Environment
Please prepare an environment with python==3.8.0, and then use the command "pip install -r requirements.txt" for the dependencies. Note these dependencies in the requirements.txt are not as lean as they could be as this environment is set up to work for multiple tobycassidy repos. 

### 3. Run
Once the environment has been configured, simply go through the "walkthrough.ipynb" notebook at your own pace. The cell outputs have been committed to this repo so the walkthrough notebook does not need to be ran and environments do not need to be configured for convenience.  
