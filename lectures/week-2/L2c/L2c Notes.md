L2c Notes

Eigendeconomposition
- vj is eigenvector
  - vj are all linearly independent
- λj is eigenvalue
- A is square matrix
  - columns of A represent feature vectors x1,...,xm with m nodes

Normalized eigenvectors of symmetric real valued matrix form orthonormal basis for space spanned by matrix A such that 
  - {vi,vj} = deltaij for all i,j in {1,...,m}
  - Kronecker Delta = 1 when i = j
  - Kronecker Delta = 0 when i ≠ j

Example - Stoichiometric Matrix
- σij > 0 if chemical species (metabolite) i is produced by reaction j. Species i is product of reaction j
- σij = 0 if Chemical species i is not connected with reaction j 
- σij < 0 if Chemical species i is consumed by reaction j. Species i is a reactant of reaction j

- If matrix isnt square, calculate the covariance matrix --> Σ = cov(S)

- Julia has built in eigen() function to compute eigenvectors of Σ matrix

Dimensionality Reduction Problem
- Suppose we have Dataset D = {x1, x2, ..., xn} with n data points where xi in R^m is an m-dimensional feature vector 
  that we want to compress into k dimensions: xi in R^m --> yi in R^k with k < m

  The φ vectors are the top 

QR Iteration
- technique used for computing the eigenvalues and eigenvectors of square matrices A 
- its for symmetric real matricies such that A = A^T
- A can be decomposed into A = QR where Q is orthogonal matrix and R is upper triangular matrix
* A = QR *
- Q^T * Q = In 
- Parameter selection Rule of Thumb
  - Convergence Tolerance: ε = 10^-6 to 10^-8, more precision 10^-10 to 10^-12
  - Maximum iterations: use maxiter function
  - Eigenvalue separation affects convergence rate: when eigen values are well separated --> |λ1| > |λ2| > ... > |λj|, algorithm converges rapidly in O(m) iterations 

- Steps
  - Compute QR decomposition of current matrix Ak = QkRk
  - Form next iteration matrix by reversing factors: Ak+1 <-- RkQk
  - Check for convergence:
    1. If ||Ak+1 - Ak||1 ≤ ε, then set converged <-- true and return eigenvalues from diagonal of Ak+1
    2. if k ≥ maxiter, then set converged <-- true to terminate algorithm, return Ak+1 as eigenvalue estimates 
    3. Increment k <-- k+1
