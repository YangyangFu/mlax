# MLAX
Machine learning in Jax. This repository is an implementation of classic machine learning algorithms leveraging the most recent development in JAX, which supports XLA, and accelorators such as GPUs, TPUs.

The classic algorithms include supervised learning algorithms such as linear regression (Bayesian regression, general linear regression etc), Kernel methods (i.e., support vector machine, relevance vector machine, Gaussian process, etc), Kernel density estimation (e.g., k-Nearest Neighbors), Mixture models (i.e., K-means clustering, Gaussian mixture, expectation maximiation algorithm etc), Continuous latent methods (e.g., Principle component analysis, etc). 
The main reference is [Pattern Recognition and Machine Learning](https://www.microsoft.com/en-us/research/uploads/prod/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf) by Bishop. 

## Thoughts
- can we categorize all the models based on the learning approaches. i.e., discriminative vs generative?
  - discriminative: linear regression, logistic regression, SVM, neural network, etc
  - generative: Gaussian process, Gaussian mixture, etc 
- maybe add a table for all models: discriminative vs generative, parametric vs nonparametric, etc

## Work Scope

- Linear Model for Regression
- Linear Model for Classification
- Kernel Methods
- Spare Kernel Methods
- Mixture Models and Expectation Maximization
- Approximate Inference
- Sampling Methods
- Continuous Latent Variables
- Linear Dynamical System
- Combining Models
- Neural Networks
  - Autodiff Implementation


