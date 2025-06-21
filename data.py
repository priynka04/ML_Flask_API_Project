from sklearn.datasets import load_iris

iris = load_iris()

print("Feature Names:", iris.feature_names)
print("Target Names:", iris.target_names)
print("Data shape:", iris.data.shape)  # (150,4)  150 samples, 4 features
print("First 10 samples:\n", iris.data[:10])
print("First 10 targets:", iris.target[:10])
