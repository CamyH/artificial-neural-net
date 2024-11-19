import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from ann import ANN
from pso import PSO

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

# Scale the target variable
target_scaler = MinMaxScaler()
y_train = target_scaler.fit_transform(y_train)
y_test = target_scaler.transform(y_test)

# Define ANN and PSO parameters
ann_architecture = [8, 32, 16, 8, 1]
ann_activations = ['relu', 'relu', 'relu', 'linear']
num_parameters = sum(w.size for w in ANN(ann_architecture, ann_activations).weights) + \
                 sum(b.size for b in ANN(ann_architecture, ann_activations).biases)

# Set up PSO for optimizing the ANN parameters
pso = PSO(num_particles=50, num_parameters=num_parameters, ann_architecture=ann_architecture,
          ann_activations=ann_activations, data=X_train, labels=y_train, swarm_size=50, c1=1.5, c2=1.5, max_iter=200)

# Run PSO optimization
best_params = pso.optimize()

# Initialize ANN with the optimized parameters
ann = ANN(ann_architecture, ann_activations)
ann.set_parameters(best_params)

# Generate predictions on the test set
predictions = []
for x in X_test:
    pred = ann.forward(x.reshape(-1, 1))
    predictions.append(pred)

# Convert predictions to a 1D array and inverse transform to original scale
predictions = np.array(predictions).flatten()
predictions = target_scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()

# Inverse transform y_test to original scale for comparison
y_test = target_scaler.inverse_transform(y_test).flatten()

# Calculate and display error metrics
mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
print(f"Test MAE: {mae}")
print(f"Test MSE: {mse}")

# Create a DataFrame with actual and predicted values
results_df = pd.DataFrame({
    'Actual_Values': y_test,
    'Predicted_Values': predictions
})

# Save to CSV
results_df.to_csv('prediction_results.csv', index=False)

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(y_test, label="Actual Values", linestyle='-', marker='o', alpha=0.6)
plt.plot(predictions, label="Predicted Values", linestyle='-', marker='x', alpha=0.6, color='orange')
plt.xlabel("Sample")
plt.ylabel("Compressive Strength")
plt.legend()
plt.title("Actual vs. Predicted Compressive Strength on Test Set")
plt.show()
