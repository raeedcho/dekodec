from dekodec import *

import numpy as np
import pymanopt

def test_fit_dekodec():
    # test that subspaces returned by fit_dekodec have the expected shapes
    # test that subspaces are orthonormal and full rank together
    num_samples = 100
    num_features = 8
    num_shared_dims = 3
    num_unique_dims = 2

    Z_unique0 = np.vstack([
        np.random.randn(num_samples,num_unique_dims),
        np.zeros((num_samples,num_unique_dims)),
    ])
    Z_unique1 = np.vstack([
        np.zeros((num_samples,num_unique_dims)),
        np.random.randn(num_samples,num_unique_dims),
    ])
    Z_shared = np.random.randn(2*num_samples,num_shared_dims)
    Z = np.column_stack([Z_unique0,Z_unique1,Z_shared])
    
    manifold = pymanopt.manifolds.Stiefel(num_features,num_shared_dims+2*num_unique_dims)
    rand_projmat = manifold.random_point()
    X_conds = {
        f'{condnum}': X
        for condnum,X in enumerate(np.split(Z @ rand_projmat.T,[num_samples]))
    }

    subspaces = fit_dekodec(X_conds)

    assert len(subspaces) == 3
    full_space = np.column_stack(tuple(subspaces.values()))
    assert full_space.shape == (num_features,num_features)
    assert np.allclose(full_space.T @ full_space, np.eye(num_features))

    total_vars = {cond: np.var(X,axis=0) for cond,X in X_conds.items()}
    assert not np.allclose(X_conds['0'] @ subspaces['0_unique'], 0)
    assert not np.allclose(X_conds['1'] @ subspaces['1_unique'], 0)
    assert not np.allclose(X_conds['0'] @ subspaces['shared'], 0)
    assert not np.allclose(X_conds['1'] @ subspaces['shared'], 0)
    assert np.var(X_conds['0'] @ subspaces['1_unique']).sum() <= 0.01 * total_vars['0'].sum()
    assert np.var(X_conds['1'] @ subspaces['0_unique']).sum() <= 0.01 * total_vars['1'].sum()

def test_get_cond_unique_basis():
    num_samples = 100
    num_features = 8
    num_shared_dims = 3
    num_unique_dims = 2

    Z_unique0 = np.vstack([
        np.random.randn(num_samples,num_unique_dims),
        np.zeros((num_samples,num_unique_dims)),
    ])
    Z_unique1 = np.vstack([
        np.zeros((num_samples,num_unique_dims)),
        np.random.randn(num_samples,num_unique_dims),
    ])
    Z_shared = np.random.randn(2*num_samples,num_shared_dims)
    Z = np.column_stack([Z_unique0,Z_unique1,Z_shared])
    
    manifold = pymanopt.manifolds.Stiefel(num_features,num_shared_dims+2*num_unique_dims)
    rand_projmat = manifold.random_point()
    X_conds = {
        f'{condnum}': X
        for condnum,X in enumerate(np.split(Z @ rand_projmat.T,[num_samples]))
    }

    inferred_projmat = get_cond_unique_basis(X_conds,'0')

    assert inferred_projmat.shape == (num_features,num_unique_dims)
    assert np.allclose(inferred_projmat.T @ inferred_projmat, np.eye(num_unique_dims))
    assert np.allclose(X_conds['1'] @ inferred_projmat, 0)
    assert not np.allclose(X_conds['0'] @ inferred_projmat, 0)

    _,inferred_sing_vals,_ = np.linalg.svd(X_conds['0'] @ inferred_projmat,full_matrices=False)
    _,true_sing_vals,_ = np.linalg.svd(Z_unique0,full_matrices=False)
    assert np.allclose(inferred_sing_vals,true_sing_vals)

def test_get_potent_null():
    num_samples = 100
    num_features = 5
    num_true_dims = 3
    
    manifold = pymanopt.manifolds.FixedRankEmbedded(num_samples,num_features,num_true_dims)
    rand_point = manifold.random_point()
    X = rand_point.u @ np.diag(rand_point.s) @ rand_point.vt

    potent_projmat,null_projmat = get_potent_null(X)

    assert potent_projmat.shape == (num_features,num_true_dims)
    assert null_projmat.shape == (num_features,num_features-num_true_dims)

    assert np.allclose(potent_projmat.T @ potent_projmat, np.eye(num_true_dims))
    assert np.allclose(null_projmat.T @ null_projmat, np.eye(num_features-num_true_dims))
    assert np.allclose(potent_projmat.T @ null_projmat, np.zeros((num_true_dims,num_features-num_true_dims)))

    potent_var = np.sum(np.var(X @ potent_projmat, axis=0))
    null_var = np.sum(np.var(X @ null_projmat, axis=0))
    total_var = np.sum(np.var(X, axis=0))
    assert np.allclose(potent_var + null_var, total_var)
    assert potent_var > null_var
    assert null_var <= 0.01*total_var
    assert potent_var >= 0.99*total_var

def test_get_dimensionality():
    # test that get_dimensionality returns the expected number of dimensions in noiseless data
    num_samples = 100
    num_features = 5
    num_true_dims = 3
    Z = np.random.randn(num_samples,num_true_dims)
    X = Z @ np.random.randn(num_true_dims, num_features)
    X_centered = X - X.mean(axis=0)
    num_dims,eigs = get_dimensionality(X)
    assert num_dims == num_true_dims
    assert np.allclose(np.sort(eigs), np.sort(np.linalg.eigvals(X_centered.T @ X_centered)))

    # test that get_dimensionality returns the expected number of dimensions in noisy data
    X_noisy = X + 1e-2*np.var(X)*np.random.randn(*X.shape)
    num_dims,_ = get_dimensionality(X_noisy)
    assert num_dims == num_true_dims

    # test that var_cutoff works as expected
    num_dims,_ = get_dimensionality(X,var_cutoff=0.5)
    assert num_dims < num_true_dims

    # ensure that mean shifts don't alter the dimensionality
    X_shifted = X + 100*np.var(X)*np.random.randn(1,num_features)
    num_dims,eigs = get_dimensionality(X_shifted)
    assert num_dims == num_true_dims
    assert np.allclose(np.sort(eigs), np.sort(np.linalg.eigvals(X_centered.T @ X_centered)))

def test_max_var_rotate():
    num_samples = 15
    num_features = 5
    num_proj_dims = 3

    proj_mat_manifold = pymanopt.manifolds.Stiefel(num_features,num_proj_dims)
    proj_mat = proj_mat_manifold.random_point() 
    X = np.random.randn(num_samples, num_features)
    new_proj_mat = max_var_rotate(proj_mat, X)

    # Test that max_var_rotate returns a matrix with the expected shape
    assert new_proj_mat.shape == (5, 3)

    # test for orthonormality
    assert np.allclose(new_proj_mat.T @ new_proj_mat, np.eye(num_proj_dims))

    # Test that max_var_rotate returns a matrix with the expected variance
    old_proj_X = X @ proj_mat
    _,sing_vals,_ = np.linalg.svd(old_proj_X-old_proj_X.mean(axis=0),full_matrices=False)
    assert np.allclose(sing_vals**2, num_samples * np.var(X @ new_proj_mat, axis=0))