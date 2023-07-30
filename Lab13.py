import numpy as np
import matplotlib.pyplot as plt
from Lab12 import power_w_deflate

def construct_A(N, K):
    A = np.zeros((N, N))
    for i in range(N):
        if i > 0:
            A[i, i-1] = -K[i-1]
        if i < N-1:
            A[i, i+1] = -K[i]
        A[i, i] = (K[i-1] if i > 0 else 0) + (K[i] if i < N-1 else 0)
    return A


def display_modes(A):
    eigenvalues, eigenvectors = power_w_deflate(A)
    for i in range(len(eigenvalues)):
        f = np.sqrt(eigenvalues[i]) / (2 * np.pi)
        mode = eigenvectors[:, i] / np.linalg.norm(eigenvectors[:, i])
        print(f"Mode {i+1}:")
        print(f"Natural frequency in Hz: {f}")
        print(f"Normalized eigenvector (modeshape): {mode}")


# Define the number of masses and the spring constants
N = 10
K = [1 for _ in range(N+1)]

# Construct the A matrix
A = construct_A(N, K)

# Print the A matrix
print("A matrix:", A)

# Display the modes
display_modes(A)

# Compare with numpy.linalg.eig
eigenvalues_np, _ = np.linalg.eig(A)
frequencies_np = [np.sqrt(eig) / (2 * np.pi) if eig >= 0 else np.nan for eig in eigenvalues_np]
frequencies_np.sort()

tolerances = [1e-2, 1e-4, 1e-6, 1e-8]
frequencies_power = []

for tol in tolerances:
    eigenvalues_power, _ = power_w_deflate(A, tol=tol)
    frequencies = [np.sqrt(eig) / (2 * np.pi) if eig >= 0 else np.nan for eig in eigenvalues_power]
    frequencies.sort()
    frequencies_power.append(frequencies)

# Plot the frequencies
plt.figure(figsize=(10, 6))
plt.plot(frequencies_np, label='numpy.linalg.eig')
for i, tol in enumerate(tolerances):
    plt.plot(frequencies_power[i], label=f'power method, tol={tol}')
plt.legend()
plt.xlabel('Mode number')
plt.ylabel('Natural frequency (Hz)')
plt.title('Comparison of natural frequencies')
plt.show()
