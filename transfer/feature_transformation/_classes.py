"""
Models for performing domain adaptation via feature transformation.
"""
import numpy as np

#######
# FDA #
#######

class FeatureReplication:
  """
  Feature augmentation of the space through features' replication (FDA).
  
  FDA works by triplicating the feature space so that every new point, in the new space, contains
  features referring to the source domain, features referring to the target domain and
  features referring to both.
  
  More specifically, the two transformations :math:`\phi_{1}` and :math:`\phi_{2}`
  are defined as :math:`\phi_{1}(x) = [x, x, 0]` and :math:`\phi_{2}(x) = [x, 0, x]`.
    
  The generalization to multiple source domains is straightforward.
  """

  def __init__(self):
    pass

  def fit(self, Xs, Xt):
    """
    Fits the model with :math:`X_{S}` and :math:`X_{T}`.

    Parameters
    ----------

    Xs : list of arrays
      List of arrays, each one containing points from a specific
      source domain.
    
    Xt : array of shape (n_samples_t, n_features)
      Array containing points from the target domain.

    Returns
    -------

    FeatureReplication
      Fitted model
    """

    self._k = len(Xs)
    self._n_features = Xt.shape[1]

    return self

  def transform(self, Xs, Xt):
    """
    Applies the model to source domains' point :math:`X_{S}` and target domain points :math:`X_{T}`.

    Parameters
    ----------

    Xs : list of arrays
      List of arrays, each one containing points from a specific
      source domain.

    Xt : array of shape (n_samples_t, n_features)
      Array containing points from the target domain.

    Returns
    -------

    array
      Array of transformed data points.
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
    """
    Fits the model with :math:`X_{S}` and :math:`X_{T}` and applies it.
    It returns the transformed points.

    Parameters
    ----------

    Xs : list of arrays
      List of arrays, each one containing points from a specific
      source domain.

    Xt : array of shape (n_samples_t, n_features)
      Array containing points from the target domain.

    Returns
    -------

    array
      Array of transformed data points.
    """
    self.fit(Xs, Xt)
    return self.transform(Xs, Xt)
