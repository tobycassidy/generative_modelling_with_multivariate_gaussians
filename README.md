# generative_modelling_with_multivariate_gaussians

This repo intends to walkthrough a use case for generative modelling with multivariate gaussians, specifically by using a variational autoencoder. Data used for explanation - [The MNIST database of handwritten digits](https://www.tensorflow.org/datasets/catalog/mnist), which was chosen for visual purposes, to help explain the concept of this project as effectively as possible.
Using a variational autoencoder has significant benefits, as we will come to, and should be a preliminary step in most investigations to gain a deeper understanding of the feature space being worked with. 


#### Interactive versions of the static plots can be found here - [Entry point to GitHub Page](https://tobycassidy.github.io/generative_modelling_with_multivariate_gaussians/)

---
## Introductory Theory - KL Divergence with Autoencoders
---

---
## Benefits
---
### 1. Quantifying Local Change 
Why it is useful? (explaining NN, or in general explaining features)
local changes summed up -> global conclusions 
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
