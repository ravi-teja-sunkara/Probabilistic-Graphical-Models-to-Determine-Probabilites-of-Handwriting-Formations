# Probabilistic-Graphical-Models-to-Determine-Probabilites-of-Handwriting-Formations
Probabilistic graphical models (PGMs) have been developed to determine the probabilities of observations for handwriting patterns which are described by several variables obtained by document examiners.

### Objective 
<p> The goal of the project is to design Probabilistic Graphical Models (PGMs) to determine probabilities of observations which are described by several features. Worked with handwriting patterns which are described by document examiners. They can be used to determine whether a particular handwriting sample is common (high probability) or rare (low probability) and which in turn can be useful to determine whether a sample was written by a certain individual. </p>

Considered letter pair __th__ image dataset as it is the most commonly encountered pair of letters (called a bigram) in English. Also, evaluated __and__ image dataset. Multiple Bayesian models and Markov models were created using __pgmpy__ library in Python. 

### Results
* Inferences obtained using VariableElimination method for both Bayesian and Markov networks are very much accurate and same.
* Inference time using Markov model is less compared to bayesian model. 
  * Computation time for inference using Bayesian Network: 0.045
  * Computation time for inference using Markov Network: 0.034
  
### Libraries Used in Python
pgmpy, numpy, pandas, networkx, matplotlib 
  
### Bayesian Model Graphs
* Bayesian Graph of best model for 'th' image dataset.

![th dataset bayesian graph](https://github.com/ravi-teja-sunkara/Probabilistic-Graphical-Models-to-Determine-Probabilites-of-Handwriting-Formations/blob/master/Graphs/'th'%20bayesian%20model.png)

* Bayesian Graph of best model for 'and' image dataset.

![and dataset bayesian graph](https://github.com/ravi-teja-sunkara/Probabilistic-Graphical-Models-to-Determine-Probabilites-of-Handwriting-Formations/blob/master/Graphs/'and'%20bayesian%20model.png)
