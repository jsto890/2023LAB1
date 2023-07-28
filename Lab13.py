import numpy as np

# Import the power_w_deflate function from the Lab12 module
from Lab12 import power_w_deflate


# Function to construct the A matrix for a mass-spring system
def construct_A(N, K):
    # Initialize A as a zero matrix of size N x N
    A = np.zeros((N, N))

    # Iterate over each row (mass) in the system
    for i in range(N):
        # If it's not the first mass, set the element to the left of the diagonal to -K[i-1]
        if i > 0:
            A[i, i-1] = -K[i-1]
        # If it's not the last mass, set the element to the right of the diagonal to -K[i]
        if i < N-1:
            A[i, i+1] = -K[i]
        # Set the diagonal element to the sum of the spring constants on either side of the mass
        A[i, i] = (K[i-1] if i > 0 else 0) + (K[i] if i < N-1 else 0)

    # Return the constructed A matrix
    return A


# Test the construct_A function
# Define the number of masses and the spring constants
N = 3
K = [1, 2, 3]

# Construct the A matrix
A = construct_A(N, K)

# Print the A matrix
print("A matrix:", A)


# Function to display the natural frequency and mode shape for each mode
def display_modes(A):
    # Use the power method with deflation to find all eigenpairs of A
    eigenvalues, eigenvectors = power_w_deflate(A)

    # Iterate over each eigenpair
    for i in range(len(eigenvalues)):
        # Calculate the natural frequency in Hz
        f = np.sqrt(eigenvalues[i]) / (2 * np.pi)
        # Normalize the mode shape (eigenvector)
        mode = eigenvectors[:, i] / np.linalg.norm(eigenvectors[:, i])

        # Print the mode number, natural frequency, and mode shape
        print(f"Mode {i+1}:")
        print(f"Natural frequency in Hz: {f}")
        print(f"Normalized eigenvector (modeshape): {mode}")


# Test the display_modes function
display_modes(A)