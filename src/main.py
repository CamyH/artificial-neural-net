# Packages
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import time

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

input_dimensions = x_train.shape[1]
output_dimensions = y_train.shape[1]

# Scale the features
feature_scaler = MinMaxScaler()
x_train = feature_scaler.fit_transform(x_train)
x_test = feature_scaler.transform(x_test)

# Setup ANN and PSO params
# Hard coded list used for experimentation purposes
#layer_dimensions = [8, 4, 6, 1]
start_time = time.time()
layer_dimensions = [input_dimensions]

# Ask the user to enter the number of layers they would like
# the neural network to have
hidden_layers = int(input("Please enter the number of hidden layers for the neural net: "))

for i in range(hidden_layers):
    layer_dimensions.append(int(input(f"Please enter the number of neurons for layer {i + 1}: ")))

# Add the output layer dimensions last
layer_dimensions.append(output_dimensions)

print(f'Selected Layer Dimensions {layer_dimensions}')

# Optionally ask the user to enter a different activation function
activation_function = str(input("Optionally enter the activation function (relu, tan, log) Press enter to skip: "))

# Initialising weights and biases for the neural network
weights, biases = setup(layer_dimensions)
num_parameters = params_count(weights, biases)

# Run PSO optimisation
optimised_params = particle_swarm_optimisation(layer_dimensions, num_parameters,
           x_train, y_train, 1.5, 1.5, activation_function, 50)

# Generate predictions on the test set
predictions = []
for x in x_test:
    pred = predict(x.reshape(-1, 1), weights, biases, optimised_params, activation_function)
    predictions.append(pred)

predictions = np.array(predictions)

# Calculate metrics for test data
result_mae = mae(y_test, predictions)
result_mse = mse(y_test, predictions)
test_df = pd.DataFrame(y_test)
#test_df.to_csv('y_test.csv', index=False)
end_time = time.time()
print(f"Test MAE: {result_mae}")
print(f"Test MSE: {result_mse}")
print(f'Elapsed Time: {end_time - start_time}')

# Output data to csv
results_reshaped = predictions.reshape(-1)
y_test_reshaped = y_test.reshape(-1)
results_df = pd.DataFrame({
    'Predictions': results_reshaped,
    'Target Values': y_test_reshaped
})
results_df.to_csv('baseline_run_10.csv')

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