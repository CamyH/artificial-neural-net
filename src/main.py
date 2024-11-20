# Packages
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Custom Hooks
from ann.neural_net import setup, predict
from pso.pso import particle_swarm_optimisation
from helper_functions.helpers import mse, mae, params_count

# Load and preprocess the dataset
df = pd.read_csv('data/concrete_data.csv')
x = df.iloc[:, :-1].values
y = df.iloc[:, -1].values.reshape(-1, 1)

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# Scale the features
feature_scaler = MinMaxScaler()
x_train = feature_scaler.fit_transform(x_train)
x_test = feature_scaler.transform(x_test)

# Setup ANN and PSO params
layer_dimensions = [8, 6, 6, 1]
# Initialising weights and biases for the neural network
weights, biases = setup(layer_dimensions)
num_parameters = params_count(weights, biases)

# Run PSO optimisation
optimised_params = particle_swarm_optimisation(layer_dimensions, num_parameters,
           x_train, y_train, 50, 1.5, 1.5)

# Generate predictions on the test set
predictions = []
for x in x_test:
    pred = predict(x.reshape(-1, 1), weights, biases, optimised_params)
    predictions.append(pred)

predictions = np.array(predictions)

# Calculate metrics for test data
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