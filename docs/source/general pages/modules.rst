Modules
#####################

Here the submodules of transfer are briefly presented.
Each one refers to a general transfer learning strategy
and implements specific tools belonging to that strategy.

Dstribution difference metrics
===================================

This submodule doesn't implement any transfer learning tool. Instead,
different metrics for estimating distribution difference
between datasets are provided.

Roughly speaking, the main idea behind transfer learning is
to reduce the distribution difference between two (or more) domains. That's why
having tools to estimate this measure can help us assess the quality
of our strategy.


Instance weighting
===================================

Instance weighting methods work by assigning to data points coming
from the training domain weights in such a way that the observations
closer to the test domain distribution are given more importance.

These weights can then be used in training models, usually by
including them in the loss function. That should help us
developing models that are suited for the target domain, despite training
them in the source domain.

Feature transformation
===================================

Feature transformation methods work by changing the space in which
both the source domain and target domain data points live. Ideally,
we would like to find a space and two transformations such that the distribution difference
between the datasets in the new space is minimal.