# transfer
Python module implementing tools and methods for transfer learning.
## Table of contents
* [Motivation](#motivation)
* [Installation](#installation)
## Motivation
**Transfer learning** (TL) is an area of machine learning that studies how to leverage knowledge gained from a series of one or more source domains to train a model on a target domain. TL is an interesting topic because there are many situations, in reality, where we have access to a large dataset belonging to a specific domain, and we would like to develop a model to apply to a different situation.

To make some examples:
* Suppose we wanted to build a model for predicting the outcome of a treatment for a certain disease. Clearly, different categories of patients usually respond differently to medical treatments. Let's imagine that we have only been able to collect data concerning a set of elderly patients, but we know the disease equally affects everyone and thus we are also interested in the reaction of young people to the treatment. This is a scenario where TL comes in handy. The topic provides us tools to transfer the knowledge obtained from the first dataset in the second, more general and differently distributed dataset.
* One of the most common problems of NLP is sentiment analysis. There are many aspects of language that change in meaning and use based on the discussed topic. It would probably not be accurate to apply a sentiment analysis model built upon movie reviews to news articles. TL can help us tuning our first model to make it more suited for the second task.

These are just two (pretty frequent) examples of problems in which TL can help us. While studying and practicing ML the user could probably notice a lot more cases. TL is thus a collection of useful tools that can ultimately make the difference between an excellent and a good model.
## Installation
### Requirements
transfer relies on the following modules:
* NumPy
* sciki-learn
* CVXPY
### Setup
Currently, transfer is not distributed with PyPi. The only way to install the module is thus to directly refer to this GitHub repository:
```
pip install git+https://github.com/Steefano/transfer
```
