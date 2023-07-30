import numpy as np


def power(A, tol=1e-6, max_iter=1000):
    n = A.shape[0]
    v = np.random.rand(n)  # Start with a random vector
    v = v / np.linalg.norm(v)  # Normalize the vector

    for _ in range(max_iter):
        Av = np.dot(A, v)
        v_new = Av / np.linalg.norm(Av)

        if np.abs(np.dot(v_new, v)) > 1 - tol:
            return np.dot(Av, v_new), v_new

        v = v_new

    return None, None


def deflate(A, eigval, eigvec):
    return A - eigval * np.outer(eigvec, eigvec)


def power_w_deflate(A, tol=1e-6, max_iter=1000):
    n = A.shape[0]
    eigenvalues = []
    eigenvectors = []

    for _ in range(n):
        eigval, eigvec = power(A, tol, max_iter)
        if eigval is None:
            break

        eigenvalues.append(eigval)
        eigenvectors.append(eigvec)
        A = deflate(A, eigval, eigvec)

    return eigenvalues, np.array(eigenvectors).T


# Test the functions
A = np.array([[2, -1], [-1, 2]])
eigval, eigvec = power(A)
print("Dominant eigenpair from power method (2x2):", eigval, eigvec)

eigenvalues, eigenvectors = power_w_deflate(A)
print("All eigenpairs from power method with deflation (2x2):")
for i in range(len(eigenvalues)):
    print(eigenvalues[i], eigenvectors[:, i])

# Define a 3x3 symmetric positive definite matrix
B = np.array([[4, 1, 1], [1, 3, 1], [1, 1, 2]])

eigval, eigvec = power(B)
print("\nDominant eigenpair from power method for 3x3 matrix:", eigval, eigvec)

eigenvalues, eigenvectors = power_w_deflate(B)
print("All eigenpairs from power method with deflation for 3x3 matrix:")
for i in range(len(eigenvalues)):
    print(eigenvalues[i], eigenvectors[:, i])
