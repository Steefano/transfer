import numpy as np
import cvxpy as cv

import sklearn.metrics.pairwise as pw
from sklearn.tree import DecisionTreeClassifier


#######
# KMM #
#######


class KernelMeanMatching:
  '''
  Kernel Mean Matching (KMM).

  Parameters
  ----------

  B : float, default = 1000
    limit on the discrepancy between the probability
    distributions

  epsilon : float, default = 0.1
    measure to assure that the resulting weights define
    something close to a probability distribution

  kernel : {"gaussian", "linear", "polynomial", "sigmoid", "laplacian"} or callable, default = "gaussian"
    kernel used in the method.

    "gaussian" : rbf kernel

    "linear" : linear kernel

    "polynomial" : polynomial kernel

    "sigmoid" : sigmoid kernel

    "laplacian" : laplacian kernel

  gamma : float, default = None
    the gamma parameter in the kernel definition (e.g. the coefficient
    of the inner product). If none it defaults to 1.0 / n_features.
    if selected kernel is linear or custom, the parameter is ignored

  coef0 : float, default = None
    the coefficient paramenter in the kernel definition
    when selecting polynomial or sigmoid kernel.
    The parameter is ignored with other kernels.

  degree : int, default = 3
    the degree of the polynomial kernel.
    The parameter is ignored if the selected kernel
    is not polynomial
  '''

  def __init__(self, B = 1000, epsilon = 0.1, kernel = "gaussian", gamma = None, coef0 = 1.0, degree = 3):
    self._B = B
    self._epsilon = epsilon
    
    if kernel == "gaussian":
      self._kernel = lambda x, y : pw.rbf_kernel(x, y, gamma)
    elif kernel == "linear":
      self._kernel = pw.linear_kernel
    elif kernel == "polynomial":
      self._kernel = lambda x, y : pw.polynomial_kernel(x, y, degree, gamma, coef0)
    elif kernel == "sigmoid":
      self._kernel = lambda x, y : pw.sigmoid_kernel(x, y, gamma, coef0)
    elif kernel == "laplacian":
      self._kernel = lambda x, y : pw.laplacian_kernel(x, y, gamma)
    else:
      self._kernel = kernel

  def fit(self, Xs, Xt):
    """
    Computes the KMM weights based on given training and test samples.

    Parameters
    ----------

    Xs : array_like of shape (n_samples_source, n_features)
      Source domain data points

    Xt : array_like of shape (n_samples_target, n_features)
      Target domain data points

    Returns
    -------

    self
      fitted weighter
    """

    ms = Xs.shape[0]
    mt = Xt.shape[0]

    # Check if the source and target data points only have one feature
    if len(Xs.shape) == 1:
      Xs = np.reshape(Xs, (ms, 1))
      Xt = np.reshape(Xt, (mt, 1))

    K = self._kernel(Xs, Xs)
    kappa = (float(ms) / float(mt)) * np.sum(self._kernel(Xs, Xt), axis = 1)
    
    G = np.concatenate((
          np.identity(ms),
          -np.identity(ms),
          np.ones((1, ms)),
          -np.ones((1, ms))
    ))

    h = np.concatenate((
        np.repeat(self._B, ms),
        np.zeros(ms),
        np.array([ms * (1 + self._epsilon), ms * (self._epsilon - 1)])
    ))
    x = cv.Variable(ms)

    objective = cv.Minimize((1 / 2) * cv.quad_form(x, K) - kappa.T @ x)
    constraints = [G @ x <= h]
    cv.Problem(objective, constraints).solve(solver = "ECOS")

    self.weights_ = x.value

    return self

  def fit_predict(self, Xs, Xt):
    """
    Computes the KMM weights based on given training and test samples
    and returns them.

    Parameters
    ----------

    Xs : array_like of shape (n_samples_sourse, n_features)
      Source domain data points
    
    Xt : array_like of shape (n_samples_target, n_features)
      Target domain data points
      
    Returns
    -------
    
    self.weights_
        weights estimated
    """
    self.fit(Xs, Xt)
    return self.weights_


#########
# KLIEP #
#########


class KullbackLeiblerImportance:
  """
  Kullback-Leibler importance estimation procedure (KLIEP).

  Parameters
  ----------

  b : "auto" or int, default = "auto"
    numbers of target domain points to use as centers
    of gaussian basis functions.

    "auto" : b is set to be equale to the number
      of target domain points.
  
  gamma : float, default = None
    the gamma parameter in the gaussian kernel definition.
    If None defaults to 1 / n_features.

  lambda : float, default = 0.
    regularization parameter.

  B : float, default = None
    upper bound on the weights. It acts
    as a form of regularization. If None, no upper bound
    is considered.

  epsilon : float, default = 0
    allowed deviation of the weights' mean from one.
  """

  def __init__(self, b = "auto", gamma = None, reg = 0., B = None, epsilon = 0.):
    self._b = b
    self._gamma = gamma
    self._reg = reg
    self._B = B
    self._epsilon = epsilon

  def fit(self, Xs, Xt):
    """
    Computes the KLIEP weights based on given training and test inputs.

    Parameters
    ----------

    Xs : array_like of shape (n_samples_source, n_features)
      Source domain data points

    Xt : array_like of shape (n_samples_target, n_features)
      Target domain data points

    Returns
    -------

    self
      fitted weighter
    """

    ms = Xs.shape[0]
    mt = Xt.shape[0]
    nf = Xs.shape[1]
    
    if self._b == "auto":
      A = pw.rbf_kernel(Xt, Xt, self._gamma)
      phi_source = pw.rbf_kernel(Xs, Xt, self._gamma)
      alpha = cv.Variable(mt)
    else:
      base_points = Xt[np.random.choice(mt, self._b, replace = False)]
      A = pw.rbf_kernel(Xt, base_points, self._gamma)
      phi_source = pw.rbf_kernel(Xs, base_points, self._gamma)
      alpha = cv.Variable(self._b)
    
    objective = cv.sum(cv.log(A @ alpha))
    if self._reg > 0:
      objective = objective - self._reg * cv.sum_squares(alpha)
    constraints = [alpha >= 0]
    weights_sum = cv.sum(phi_source @ alpha)
    if self._epsilon != 0:
      constraints.append(weights_sum <= mt * (1 + self._epsilon))
      constraints.append(weights_sum >= mt * (1 - self._epsilon))
    else:
      constraints.append(weights_sum == mt)
    if not self._B is None:
      constraints.append(alpha <= self._B)
    prob = cv.Problem(cv.Maximize(objective), constraints)
    prob.solve()
    
    self.weights_ = np.dot(phi_source, alpha.value)
    return self
  
  def fit_predict(self, Xs, Xt):
    """
    Computes the KLIEP weights based on given source
    and target inputs and returns them.
    
    Parameters
    ----------
    
    Xs : array_like of shape (n_samples_source, n_features)
      Source domain data points

    Xt : array_like of shape (n_samples_target, n_features)
      Target domain data points

    Returns
    -------

    self.weights_
      weights_ estimated
    """
    self.fit(Xs, Xt)
    return self.weights_


###########
# 2SW-MDA #
###########

class TwoStageWeighting:
  """
  2-Stage weighting framework for multi-source domain adaptation (2SW-MDA).

  Parameters
  ----------
  
  base_estimator : estimator object, default = DecisionTreeClassifier(max_depth = 1)
    estimator to use to learn hypothesis during the second
    stage of the weighting process
    
  base_weigher : weighter object, default = KernelMeanMatching()
    weighter to use during the first stage of the
    weighting process
    
  similarity : {"euclidean", "cosine", "l1"} or callable, default = "euclidean"
    similarity function to use for computing similarity
    matrix of target inputs
    
    "euclidean" : euclidean similarity defined as
        1 / (1 + ||x - y||^2)
    
    "cosine" : cosine similarity
    
    "l1" : manhattan similarity defined as
        1 / (1 + ||x - y||)
  """

  def __init__(self, base_estimator = DecisionTreeClassifier(max_depth = 1), base_weighter = KernelMeanMatching(), similarity = "euclidean"):
    self._base_estimator = base_estimator
    self._base_weighter = base_weighter
    
    if similarity == "euclidean":
      self._similarity = lambda x, y : 1 / (1 + pw.euclidean_distances(x, y))
    elif similarity == "cosine":
      self._similarity = pw.cosine_similarity
    elif similarity == "l1":
      self._similarity = lambda x, y : 1 / (1 + pw.manhattan_distances(x, y))
    else:
      self._similarity = similarity
    
  def fit(self, Xs, Ys, Xt):
    """
    Computes the 2SW-MDA weights based on given training
    inputs and labels and the given test inputs.

    Parameters
    ----------

    Xs : list of arrays
      list of arrays each one containing
      data points from a single source.

    Ys : list of arrays
      list of arrays each one containing labels
      from a single source.
    
    Xt : array_like of shape (n_samples_t, n_features)
      array containing target domain
      data points.
      
    Returns
    -------
    
    self
        fitted weighter
    """
    k = len(Xs)

    self.alpha_weights_ = []
    H = np.array(np.zeros(Xt.shape[0]))
    for source_domain_x, source_domain_y in zip(Xs, Ys):
      self.alpha_weights_.append(self._base_weighter.fit_predict(source_domain_x, Xt))
      self._base_estimator.fit(source_domain_x, source_domain_y, sample_weight = self.alpha_weights_[-1])
      H = np.vstack([H, self._base_estimator.predict(Xt)])
    H = H[1:]

    beta = cv.Variable(k)
    W = self._similarity(Xt, Xt)
    D = np.diag(np.sum(W, axis = 1))
    Lu = D - W
    P = H @ Lu @ H.T
    objective = cv.quad_form(beta, P)
    constraints = [beta >= 0, cv.sum(beta) == k]
    cv.Problem(cv.Minimize(objective), constraints).solve(solver = "ECOS")
    self.beta_weights_ = beta.value
    
    return self
