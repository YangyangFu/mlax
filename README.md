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
- Inference and Decision Produre
- Generalization and Regularization
- Supervised Learning
  - Linear regression
  - Logistic regression
  - Bayesian linear regression
  - Bayesian logistic regression
  - Generalized linear model
  - Generative models for classification
    - Gaussian discriminant analysis
    - Naive Bayes
  - Kernel methods
    - Gaussian process
  - Sparse kernel methods
    - Support vector machine
    - Relevance vector machine
  - Graphical models 
    - Bayesian networks
    - Markov models
    - Linear dynamical systems
- Unsupervised Learning 
  - Mixture models
    - k-means
    - Gaussian mixture
    - Expectation maximization algorithm 
  - Variational inference
  - Principle components analysis
  - Self-supervised learning
- Combing Models
  - Bayesian averaging 
  - Bagging
  - Boosting 

