# dekodec

A codebase to split neural population activity into condition-specific unique subspaces and shared subspaces.

---

## Method Overview

**DekODec (Dekleva Orthonormal Decomposition)** is based on the manifold view of neural activity, in which population activity is represented as a point (state) within a low-dimensional manifold. For different tasks, the neural state may inhabit different subspaces within the manifold. If the neural state projects in part onto an axis or subspace *only* during condition A, then we consider that axis/subspace to be **A-unique**. Likewise, if the neural state contains projections onto a different axis/subspace during multiple conditions (e.g., A, B, and C), we consider it to be **ABC-shared**.

### Intuition

Consider three conditions, A, B, and C:
- If there is some axis/subspace that is unique to A, then it must exist in the null space of the space defined by B and C.
- Likewise, a B-unique subspace exists in the null space of A and C, etc.

We use PCA and a variance cutoff to identify the relevant null spaces for identifying unique activity:

- **A_unique** exists in {B, C}_null
- **B_unique** in {A, C}_null
- **C_unique** in {A, B}_null

This gives us the **form** of the unique activity, but since each condition is calculated separately, they will not be mathematically orthogonal (each basis estimation will incorporate some degree of noise, largely due to low-variance dimensions in the original data).

Once we have identified the profiles of the unique activity, we can find an orthonormal transformation of the original space that reconstructs those known profiles. We do this by minimizing sum squared error using the Manopt toolbox. The output of the optimization reflects the combination of unique spaces, and we can simply take the null space of this to define the shared space.

> **The end result is an orthonormal transformation of the original basis such that each axis reflects either condition-unique or condition-shared activity.**

---

### Attribution

- Created by Brian Dekleva (5/2/2023)
- Adapted for Python by Raeed Chowdhury (7/19/2023)