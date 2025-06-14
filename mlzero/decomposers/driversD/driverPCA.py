import numpy as np
from mlzero.decomposers.pca import PCA
import matplotlib.pyplot as plt

def main():
    # Generate synthetic data (2D for visualization)
    np.random.seed(42)
    X = np.random.randn(100, 2) @ np.array([[0.6, -0.8], [0.8, 0.6]]) + np.array([2, 5])

    # Fit PCA to reduce to 1D
    pca = PCA(n_components=1)
    pca.fit(X)
    X_pca = pca.transform(X)

    print("Original shape:", X.shape)
    print("Transformed shape:", X_pca.shape)
    print("PCA components:", pca.components)

    # Visualize original data and principal component
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], alpha=0.5, label='Original Data')
    origin = np.mean(X, axis=0)
    vector = pca.components[0] * 3  # scale for visualization
    plt.quiver(*origin, *vector, color='red', scale=1, scale_units='xy', angles='xy', label='Principal Component')
    plt.title('PCA: Original Data and First Principal Component')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
