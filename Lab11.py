import numpy as np

# Define the matrix
A = np.array([[2, -1], [-1, 2]])

# Initialize the list of eigenvalues and eigenvectors
eigenvalues = []
eigenvectors = []

# Power method with deflation
for _ in range(A.shape[0]):
    # Initialize the eigenvector with a random vector
    v = np.random.rand(A.shape[1])

    # Set the number of iterations
    num_iterations = 1000

    # Power iteration
    for _ in range(num_iterations):
        # Apply the transformation
        v = np.dot(A, v)
        # Normalize the vector
        v = v / np.linalg.norm(v)

    dominant_eigenvector = v
    dominant_eigenvalue = np.dot(dominant_eigenvector.T, np.dot(A, dominant_eigenvector))

    # Add the dominant eigenpair to the list
    eigenvalues.append(dominant_eigenvalue)
    eigenvectors.append(dominant_eigenvector)

    # Deflate the matrix
    A = A - dominant_eigenvalue * np.outer(dominant_eigenvector, dominant_eigenvector)

# Convert the list of eigenvectors to a 2D array
eigenvectors = np.array(eigenvectors).T

# Print the eigenvalues and eigenvectors
print("Eigenvalues:", eigenvalues)
print("Eigenvectors:", eigenvectors)

# Find the index of the dominant (largest) eigenvalue
dominant_index = np.argmax(eigenvalues)

# Print the dominant eigenpair
print("Dominant Eigenvalue:", eigenvalues[dominant_index])
print("Dominant Eigenvector:", eigenvectors[:, dominant_index])

