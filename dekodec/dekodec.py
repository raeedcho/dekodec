'''
DekODec (Dekleva Orthonormal Decomposition) is based on the manifold 
view of neural activity, in which we can represent population activity 
as a point (state) within a low dimensional manifold. For different 
tasks, the neural state may inhabit different subspaces within the 
manifold. If the neural state projects in part onto an axis or subspace 
*only* during condition A, then we consider that axis/subspace to be
"A-unique". Likewise, if the neural state contains projections onto a 
different axis/subspace during multiple conditions (e.g. A, B, and C),
we consider it to be "ABC-shared".

The intuition behind the present method is as follows: 
Consider three conditions, A, B and C. If there is some axis/subspace
that is unique to A, then it must exist in the null space of the
space defeined by B and C. Likewise, a B-unique subspace exists in the
null space of A and C, etc. We use PCA and a variance cutoff to 
identify the relevant null spaces for identifying unique activity: 

    A_unique exists in {B,C}_null, B_unique in {A,C}_null, and C_unique
    in {A,B} null

Doing so gives us the **form** of the unique activity, but since each
condition is calculated separately, they will not be mathematically
orthogonal (each basis estimation will incorporate some degree of 
noise, largely due to low-variance dimensions in the original data). 

However, once we have identified the profiles of the unique activity,
we can find an orthonormal transformation of the original space that
reconstructs those known profiles. We do this by minimizing sum 
squared error using the Manopt toolbox. The output of the optimization
reflects the combination of unique spaces, and we can simply take the
null space of this to define the shared space. 

***The end result is an
orthonormal transformation of the original basis such that each axis
reflects either condition-unique or condition-shared activity***


----
created by Brian Dekleva 5/2/2023
adapted for python by Raeed Chowdhury 7/19/2023
'''
import numpy as np
import pandas as pd
from scipy.linalg import null_space
import torch
import pymanopt
import pymanopt.manifolds
import pymanopt.optimizers
from typing import Optional

from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.utils.validation import check_is_fitted

class DekODec(BaseEstimator, TransformerMixin):
    def __init__(
            self,
            var_cutoff=0.99,
            condition=None,
            split_transform=False,
        ):
        assert condition is not None, "Must provide condition column name"

        self.var_cutoff = var_cutoff
        self.condition = condition
        self.split_transform = split_transform

    def fit(self, X, y=None):
        X_conds_dict = {
            cond: tab.values
            for cond,tab in X.groupby(self.condition)
        }
        self.subspaces_ = fit_dekodec(X_conds_dict,var_cutoff=self.var_cutoff)

        return self

    def transform(self,X):
        '''
        Projects data into unique and shared subspaces.
        '''
        check_is_fitted(self, 'subspaces_')

        if self.split_transform:
            return self.transform_split(X)
        else:
            return self.transform_full(X)

    def transform_full(self, X):
        return X @ np.column_stack(tuple(self.subspaces_.values()))

    def transform_split(self, X):
        return pd.concat(
            {
                space: X @ proj_mat
                for space,proj_mat in self.subspaces_.items()
            },
            axis=1,
            names=['space','component'],
        )

def fit_dekodec(X_conds, var_cutoff=0.99, combinations=None):
    """
    Splits data into orthogonal subspaces containing condition-unique and 
    condition-shared activity.

    Parameters
    ----------
    X_conds : dict of numpy arrays
        dict of matrices containing neural firing rates for different conditions,labelled by condition.
        In each array, rows correspond to samples and columns correspond to neural dimensions.
        All arrays must have the same dimensionality.
    var_cutoff : float, optional
        Fraction variance cutoff used to delineate potent and null spaces.
        If no cutoff is given, the default value is 0.99, i.e. the potent space
        will explain a minimum of 99% of the total variance.
    do_plot : bool, optional
        Flag to return plot (True) [default] or not (False)
    combinations : list of tuples, optional
        Custom list of combinations to check. Currently not converted from original MATLAB code.

    Returns
    -------
    subspaces : dict
        Dictionary containing 'unique' and 'shared' fields, which contain the axes for each identified subspace.
        Together, they form a full orthonormal basis.
    
    Created by Brian Dekleva, 2023-05-02
    Adapted for Python by Raeed Chowdhury, 2023-07-19
    """
    
    num_conds = len(X_conds)

    assert num_conds > 1, 'Must have at least two conditions to compare'
    assert combinations is None, 'Custom combinations not yet implemented'

    cond_unique_projmats = {
        cond: get_cond_unique_basis(X_conds,cond,var_cutoff=var_cutoff)
        for cond in X_conds
    }
    subspaces = orthogonalize_spaces(X_conds,cond_unique_projmats)

    return subspaces

def get_cond_unique_basis(X_conds,which_cond,var_cutoff=0.99):
    """
    Calculates the conditional unique basis for a matrix based on the percent variance cutoff.

    Parameters:
    -----------
    X_conds: dict of numpy arrays
        List of matrices containing neural firing rates for different conditions.
        In each array, rows correspond to samples and columns correspond to neural dimensions.
        All arrays must have the same dimensionality.
    which_cond : dict key
        condition to calculate the conditional unique basis for.
    var_cutoff : float, optional (default=0.99)
        The fraction variance cutoff for the eigenvalues.

    Returns:
    --------
    cond_unique_projmat : array-like, shape (n_features, n_unique_dims)
        The condition unique projection matrix.
    """

    assert len({X.shape[1] for X in X_conds.values()}) == 1, 'All conditions must have the same number of dimensions'

    X_cond = X_conds[which_cond]
    X_notcond = np.vstack([
        X for cond,X in X_conds.items()
        if cond != which_cond
    ])

    _,notcond_null = get_potent_null(X_notcond, var_cutoff=var_cutoff)
    cond_unique_projmat = max_var_rotate(notcond_null,X_cond)
    num_unique_dims = get_num_projected_dims_to_keep(
        X_cond,
        cond_unique_projmat,
        var_cutoff=var_cutoff
    )

    return cond_unique_projmat[:,:num_unique_dims]

def orthogonalize_spaces(X_conds,cond_unique_projmats,backend='pymanopt'):
    """
    Return orthogonalized projection matrices for unique and shared subspaces
    given the input data matrices and the conditional unique projection matrices.

    Parameters
    ----------
    X_conds : dict
        A dictionary of input data matrices, where each key is a condition label
        and each value is a matrix with shape (n_samples, n_features).
    cond_unique_projmats : dict
        A dictionary of projection matrices, where each key is a condition label
        and each value is a matrix with shape (n_features, n_unique_dims).

    Returns
    -------
    dict
        A dictionary of orthogonalized projection matrices, where each key is a condition label and each value is a matrix with shape (n_features, n_unique_dims).

    """
    num_unique_dims = [projmat.shape[1] for projmat in cond_unique_projmats.values()]
    total_unique_dims = np.sum(num_unique_dims)
    if total_unique_dims == 0:
        subspaces = {
            f'{cond.lower()}_unique': np.zeros((arr.shape[1],0)) for cond,arr in  X_conds.items()
        }
        subspaces['shared'] = np.eye(next(iter(X_conds.values())).shape[1])
        return subspaces

    Z = torch.from_numpy(np.vstack(tuple(X_conds.values())).astype('float64'))
    Z_uniques = torch.column_stack([
        Z @ torch.from_numpy(projmat.astype('float64'))
        for projmat in cond_unique_projmats.values()
    ])

    if backend == 'pymanopt':
        manifold = pymanopt.manifolds.Stiefel(Z.shape[1],total_unique_dims)
        @pymanopt.function.pytorch(manifold)
        def cost(Q):
            return torch.sum(torch.square(Z @ Q - Z_uniques))
        problem = pymanopt.Problem(manifold,cost)
        optimizer = pymanopt.optimizers.TrustRegions()
        result = optimizer.run(problem)
        Q_all_uniques = flip_positive(result.point)
    elif backend == 'geotorch':
        # model = torch.nn.Linear(Z.shape[1],total_unique_dims, bias=False)
        # geotorch.grassmannian(model,'weight')
        # torch.optim.Adam(model.parameters(),lr=0.001)
        # def cost(Q):
        #     return torch.sum(torch.square(Z @ Q - Z_uniques))

        raise NotImplementedError('Geotorch backend not yet implemented')
    else:
        raise ValueError(f'Backend {backend} not recognized')

    subspaces = {
        f'{cond.lower()}_unique': arr for cond,arr in zip(
            X_conds.keys(),
            np.split(Q_all_uniques,np.cumsum(num_unique_dims),axis=1)[:-1]
        )
    }
    subspaces['shared'] = max_var_rotate(
        null_space(np.column_stack(tuple(subspaces.values())).T),
        np.vstack(tuple(X_conds.values())),
    )

    return subspaces

def get_potent_null(X: np.ndarray, var_cutoff: float = 0.99, num_dims: Optional[int] = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculates the potent and null spaces for a matrix based on the percent variance cutoff.
    
    Parameters
    ----------
    X : numpy array
        Matrix containing neural firing rates.
        Rows correspond to samples and columns correspond to neural dimensions.
    var_cutoff : float, optional
        Percent variance cutoff used to delineate potent and null spaces.
        If no cutoff is given, the default value is 0.99, i.e. the potent space
        will explain a minimum of 99% of the total variance.9.

    Returns
    -------
    potent_projmat : numpy array
        Projection matrix to transform data into the potent space basis
    null_projmat : numpy array
        Projection matrix to transform data into the null space basis
    """

    X_centered = X - np.mean(X, axis=0)

    if num_dims is not None:
        assert num_dims > 0, 'Number of dimensions must be greater than 0'
        assert num_dims <= X.shape[1], 'Number of dimensions cannot exceed number of features'
    else:
        assert 0 < var_cutoff <= 1, 'Variance cutoff must be between 0 and 1'
        num_dims, _ = get_dimensionality(X_centered, var_cutoff=var_cutoff)
    
    _, _, Vh = np.linalg.svd(X_centered, full_matrices=False)

    potent_projmat = Vh[:num_dims,:].T
    null_projmat = Vh[num_dims:,:].T

    return potent_projmat, null_projmat

def get_dimensionality(X, var_cutoff=0.99):
    """
    Calculates the dimensionality of a matrix based on the percent variance cutoff.

    Parameters
    ----------
    X : numpy array
        Matrix containing neural firing rates.
        Rows correspond to samples and columns correspond to neural dimensions.
    var_cutoff : float, optional
        Percent variance cutoff used to delineate potent and null spaces.
        If no cutoff is given, the default value is 0.99, i.e. the potent space
        will explain a minimum of 99% of the total variance.

    Returns
    -------
    num_dims : int
        Number of dimensions in the matrix.
    eigenvalues : numpy array
        Eigenvalues of the matrix.
    """

    assert 0 < var_cutoff <= 1, 'Variance cutoff must be between 0 and 1'

    X_centered = X - np.mean(X, axis=0)
    _, S, _ = np.linalg.svd(X_centered, full_matrices=False)

    eigenvalues = S**2
    cumulative_variance_explained = np.cumsum(eigenvalues) / np.sum(eigenvalues)
    num_dims = np.sum(cumulative_variance_explained <= var_cutoff) + 1

    if num_dims > X.shape[1]:
        num_dims = X.shape[1]

    return num_dims, eigenvalues

def max_var_rotate(proj_mat,X):
    """
    Rotate the columns of the projection matrix proj_mat to maximize the variance
    of the data (X) projected through it.

    Parameters
    ----------
    proj_mat : numpy.ndarray
        The initial projection matrix with shape (n_features, n_components).
    X : numpy.ndarray
        The input data matrix with shape (n_samples, n_features).

    Returns
    -------
    numpy.ndarray
        The updated projection matrix with shape (n_features, n_components).
    """
    X_centered = X - np.mean(X,axis=0)

    projected_activity = X_centered @ proj_mat
    _,_,Vh = np.linalg.svd(projected_activity,full_matrices=False)
    new_proj_mat = proj_mat @ Vh.T

    return new_proj_mat

def get_num_projected_dims_to_keep(X,proj_mat,var_cutoff=0.99):
    """
    Get number of dimensions to keep after projecting data (X) through proj_mat,
    based on the variance explained by the original data. Only keep dimensions that
    explain at least as much variance as the last original PC.

    Parameters
    ----------
    X : numpy.ndarray
        The input data matrix with shape (n_samples, n_features).
    proj_mat : numpy.ndarray
        The projection matrix with shape (n_features, n_components).
    var_cutoff : float, optional
        The proportion of variance to determine original data dimensionality (default is 0.99).

    Returns
    -------
    int
        The number of dimensions to keep after projection.

    """
    assert 0 < var_cutoff < 1, 'Variance cutoff must be between 0 and 1'

    full_dimensionality,full_eigvals = get_dimensionality(X, var_cutoff=var_cutoff)
    _,proj_eigvals = get_dimensionality(X @ proj_mat, var_cutoff=var_cutoff)
    proj_var_explained = proj_eigvals / np.sum(full_eigvals)
    proj_dim_var_cutoff = full_eigvals[full_dimensionality-1] / np.sum(full_eigvals)
    num_proj_dims = np.sum(proj_var_explained >= proj_dim_var_cutoff)
    
    return num_proj_dims

def flip_positive(Q):
    return Q