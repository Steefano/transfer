"""
This module contains functions to compute different
criteria of distribution difference.
"""
import numpy as np
import sklearn.metrics.pairwise as pw

def MaximumMeanDiscrepancy(Xs, Xt, kernel = "gaussian", method = "quadratic"):
  """
  This function computes the estimated Maximum Mean Discrepancy (MMD).
  between two datasets using the kernel trick.
  
  Parameters
  ----------
  Xs : array
    Array of shape (n_samples_source, n_features) containing
    the datasets of points from the first domain.
    
  Xt : array
    Array of shape (n_samples_target, n_features) containing
    the datasets of points from the second domain.
    
  kernel : {"gaussian", "linear"}
    Kernel used in the computation of the MMD.
    
  method : {"quadratic", "linear"}
    Method to use to compute the MMD. Quadratic is slower,
    but it makes use of all the observatons. Linear is faster and
    consumes less memory, but only makes use of subsets
    of the domains.
    
  Returns
  -------
  float
    Estimated MMD between the two datasets.
  """
  
  # Save sizes of datasets
  m = Xs.shape[0]
  n = Xt.shape[0]
  n_features = Xs.shape[1]
  
  # Initialize kernel according to the parameter
  if kernel == "gaussian":
    kernel = pw.rbf_kernel
  elif kernel == "linear":
    kernel = pw.linear_kernel
  
  # Quadratic
  if method == "quadratic":
    K = kernel(Xs, Xs)
    first_term = (np.sum(K) - np.trace(K)) / (m * (m - 1))
    K = kernel(Xt, Xt)
    second_term = (np.sum(K) - np.trace(K)) / (n * (n - 1))
    K = kernel(Xs, Xt)
    third_term = (2 * np.sum(K)) / (m * n)
    return first_term + second_term - third_term
  # Linear
  elif method == "linear":
    N = min(m, n)
    if N % 2 == 1:
      N = N - 1
    Xs_couples = np.reshape(Xs[:N], newshape = (N // 2, 2, n_features))
    Xt_couples = np.reshape(Xt[:N], newshape = (N // 2, 2, n_features))
    all = np.stack((Xs_couples, Xt_couples), axis = 1)
    h_sum = 0
    for element in all:
      moved = np.moveaxis(element, 0, 1)
      kernel_matrix = kernel(moved[0], moved[1])
      h_sum = h_sum + 2 * np.trace(kernel_matrix) - np.sum(kernel_matrix)
    return (2 * h_sum) / N
