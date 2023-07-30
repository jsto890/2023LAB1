import numpy as np
from Lab12 import power_w_deflate


def construct_A(N, K):
    # Initialize A as a zero matrix
    A = np.zeros((N, N))

    # Fill the diagonal elements
    for i in range(N):
        if i == 0:
            A[i, i] = K[i] + K[i+1]
        elif i == N - 1:
            A[i, i] = K[i]
        else:
            A[i, i] = K[i] + K[i+1]

    # Fill the off-diagonal elements
    for i in range(N-1):
        A[i, i+1] = -K[i+1]
        A[i+1, i] = -K[i+1]

    return A


def display_modes(A):
    # Use the power method with deflation to find all eigenpairs of A
    eigenvalues, eigenvectors = power_w_deflate(A)

    # Display the natural frequency and mode shape for each mode
    for i in range(len(eigenvalues)):
        natural_frequency = np.sqrt(eigenvalues[i]) / (2 * np.pi)
        mode = eigenvectors[:, i] / np.linalg.norm(eigenvectors[:, i])
        print(f"Mode {i+1}:")
        print(f"Natural frequency: {natural_frequency} Hz")
        print(f"Mode shape: {mode}")


# Test the functions
N = 3
K = [1, 2, 3]
A = construct_A(N, K)
print("A matrix:", A)
display_modes(A)
