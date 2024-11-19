import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from src.ann.neuralnetwork import NeuralNetwork
from src.pso.pso import particle_swarm_optimisation
from helper_functions.helpers import mse, mae

# Load and preprocess the dataset
df = pd.read_csv('data/concrete_data.csv')
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values.reshape(-1, 1)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scale the features
feature_scaler = MinMaxScaler()
X_train = feature_scaler.fit_transform(X_train)
X_test = feature_scaler.transform(X_test)

# Define ANN and PSO parameters
ann_architecture = [8, 6, 6, 1]
ann_activations = ['relu', 'relu', 'relu', 'linear']
num_parameters = sum(w.size for w in NeuralNetwork(ann_architecture, ann_activations).weights) + \
                 sum(b.size for b in NeuralNetwork(ann_architecture, ann_activations).biases)

# Run PSO optimisation
best_params = particle_swarm_optimisation(num_parameters, ann_architecture,
          ann_activations, X_train, y_train, 50, 1.5, 1.5)

# Initialize ANN with the optimised parameters
ann = NeuralNetwork(ann_architecture, ann_activations)
ann.update_parameters(best_params)

# Generate predictions on the test set
predictions = []
for x in X_test:
    pred = ann.forward_pass(x.reshape(-1, 1))
    predictions.append(pred)

predictions = np.array(predictions)

# Calculate metrics for training data
result_mae = mae(y_test, predictions)
result_mse = mse(y_test, predictions)
print(f"Test MAE: {result_mae}")
print(f"Test MSE: {result_mse}")

# Output data to csv
results_reshaped = predictions.reshape(-1)
results_df = pd.DataFrame(results_reshaped, columns=['predictions'])
results_df.to_csv('output.csv')

# Plot the results
'''
plt.figure(figsize=(10, 6))
plt.plot(y_test, label="Actual Values", linestyle='-', marker='o', alpha=0.6)
plt.plot(predictions, label="Predicted Values", linestyle='-', marker='x', alpha=0.6, color='orange')
plt.xlabel("Sample")
plt.ylabel("Compressive Strength")
plt.legend()
plt.title("Actual vs. Predicted Compressive Strength on Test Set")
plt.show()
'''