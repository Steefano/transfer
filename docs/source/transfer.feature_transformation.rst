Feature Transformation
###############################################

Feature transformation methods work by applying transformations
to the source and target domains. Ideally, the source and target
distributions should be a lot closer in the new space.

More formally, suppose we are given source and target domains :math:`D_{S}, D_{T}`.
Remember that a domain is a couple :math:`(X, Y, P)` where :math:`X` is the matrix
containing the features of the observations, :math:`Y` contains the labels and
:math:`P` is the domain distribution.

A feature transformation method tries to find a new space :math:`\bar{X}` and two
transformations

.. math::
	
	\begin{eqnarray}
	\phi_{1} : X_{S} \longrightarrow \bar{X} \\
	\phi_{2} : X_{T} \longrightarrow \bar{X}
	\end{eqnarray}
	
such that the distribution difference :math:`d\left(\phi_{1}\left(P\right), \phi_{2}\left(P\right)\right)`
is as small as possible.

In the end, a model is developed upon :math:`\bar{X}` by using the source domain points
as well as target domain labeled points.

Methods
---------------

.. automodule:: transfer.feature_transformation
   :members:
   :undoc-members:
   :show-inheritance: