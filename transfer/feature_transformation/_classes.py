"""
Model for performing domain adaptation by instance weighting.
"""
import numpy as np

#######
# FDA #
#######

class FeatureReplication:
  """
  Feature augmentation of the space through features' replication (FDA).
  """

  def __init__(self):
    pass

  def fit(self, Xs, Xt):
    """
    Fits the model with Xs and Xt.

    Parameters
    ----------

    Xs : list of arrays
      list of data points from the source domains
    
    Xt : array of shape (n_samples_t, n_features)
      target domain data points

    Returns
    -------

    self
      fitted model
    """

    self._k = len(Xs)
    self._n_features = Xt.shape[1]

    return self

  def transform(self, Xs, Xt):
    """
    Apply the model to source domains' point Xs and target domain points Xt.

    Parameters
    ----------

    Xs : list of arrays
      list of data points from the source domains

    Xt : array of shape (n_samples_t, n_features)
      target domain data points

    Returns
    -------

    array of shape (product(Xs.shape), self._n_features)
      data points in the new space of replicated features
    """

    to_return = np.array([np.zeros(self._n_features * (self._k + 2))])
    for index, source_domain in enumerate(Xs):
      transformed = np.hstack([source_domain, np.zeros((source_domain.shape[0], self._n_features * (self._k + 1)))])
      transformed[:, self._n_features * (index + 1) : self._n_features * (index + 2)] = source_domain
      to_return = np.vstack([to_return, transformed])
    
    transformed = np.hstack([Xt, np.zeros((Xt.shape[0], self._n_features * (self._k + 1)))])
    transformed[:, self._n_features * (self._k + 1) : self._n_features * (self._k + 2)] = Xt
    to_return = np.vstack([to_return, transformed])

    return to_return[1:]

  def fit_transform(self, Xs, Xt):
    self.fit(Xs, Xt)
    return self.transform(Xs, Xt)
