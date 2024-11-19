from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
from ann import ANN
from pso import PSO
import requests
from bs4 import BeautifulSoup

# Function to fetch OEIS sequence data
def fetch_oeis_sequence(oeis_id, num_terms=100):
    url = f"https://oeis.org/{oeis_id}/b{oeis_id[1:]}.txt"
    response = requests.get(url)

    if response.status_code == 200:
        terms = []
        for line in response.text.strip().splitlines():
            index, value = line.split()
            terms.append(int(value))
            if len(terms) >= num_terms:
                break
        return terms
    else:
        raise ValueError(f"Failed to fetch sequence {oeis_id}. HTTP Status Code: {response.status_code}")

# Fetch sequence data from OEIS
oeis_id = "A000045"  # Example: Fibonacci sequence
sequence = fetch_oeis_sequence(oeis_id, num_terms=100)
print("Fetched sequence:", sequence)

# Prepare the sequence data for training
window_size = 3
terms_to_predict = 5

def prepare_sequence_data(sequence, window_size=3):
    X, y = [], []
    for i in range(len(sequence) - window_size):
        X.append(sequence[i:i + window_size])
        y.append(sequence[i + window_size])
    return np.array(X), np.array(y).reshape(-1, 1)

X, y = prepare_sequence_data(sequence, window_size=window_size)

# Scale the data
input_scaler = StandardScaler()
X = input_scaler.fit_transform(X)

target_scaler = StandardScaler()
y = target_scaler.fit_transform(y)

# Define and configure the ANN and PSO as before
ann_architecture = [window_size, 64, 32, 16, 1]
ann_activations = ['relu', 'relu', 'relu', 'linear']
num_parameters = sum(w.size for w in ANN(ann_architecture, ann_activations).weights) + \
                 sum(b.size for b in ANN(ann_architecture, ann_activations).biases)

# Initialize PSO
pso = PSO(num_particles=50, num_parameters=num_parameters, ann_architecture=ann_architecture,
          ann_activations=ann_activations, data=X, labels=y, swarm_size=50, c1=1.5, c2=1.5, max_iter=300)

# Run PSO optimization
best_params = pso.optimize()

# Initialize ANN with optimized parameters
ann = ANN(ann_architecture, ann_activations)
ann.set_parameters(best_params)

# Function to predict future terms
def predict_next_terms(ann, input_scaler, target_scaler, initial_sequence, terms_to_predict=5):
    current_sequence = initial_sequence[-window_size:]
    predicted_terms = []
    for _ in range(terms_to_predict):
        scaled_input = input_scaler.transform([current_sequence])
        next_term = ann.forward(scaled_input[0].reshape(-1, 1))
        next_term = np.array(next_term).reshape(1, 1)
        next_term = float(target_scaler.inverse_transform(next_term).flatten()[0])
        predicted_terms.append(next_term)
        current_sequence = current_sequence[1:] + [next_term]
    return predicted_terms

# Generate predictions for future terms
future_terms = predict_next_terms(ann, input_scaler, target_scaler, sequence, terms_to_predict=terms_to_predict)

def calculate_fibonacci_sequence(start_sequence, num_terms):
    sequence = start_sequence[:]
    while len(sequence) < len(start_sequence) + num_terms:
        sequence.append(sequence[-1] + sequence[-2])
    return sequence[len(start_sequence):]  # Return only the new terms

# Use this function to set `expected_future_values`
expected_future_values = calculate_fibonacci_sequence(sequence, terms_to_predict)

# Example of expected future terms (for Fibonacci, adjust based on actual sequence)
#expected_future_values = sequence[len(sequence):len(sequence) + terms_to_predict]

# Output results (add debugging prints here)
print("Future Terms:", future_terms)
print("Expected Future Values:", expected_future_values)

# Output results
print("Index | Expected Term | Predicted Term")
print("---------------------------------------")
for i, (expected, predicted) in enumerate(zip(expected_future_values, future_terms), start=len(sequence)):
    print(f"{i:<5} | {expected:<13} | {predicted:.2f}")

# Error metrics and plot code (same as previous implementation)
mae = mean_absolute_error(expected_future_values, future_terms)
mse = mean_squared_error(expected_future_values, future_terms)
rmse = np.sqrt(mse)
mape = np.mean(np.abs((np.array(expected_future_values) - np.array(future_terms)) / np.array(expected_future_values))) * 100

print("\nError Metrics:")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"Mean Absolute Percentage Error (MAPE): {mape}%")

# Plot expected vs. predicted future terms
indices = list(range(len(sequence), len(sequence) + len(expected_future_values)))

plt.figure(figsize=(10, 6))
plt.plot(indices, expected_future_values, label="Expected Terms", marker='o', linestyle='-', color='blue')
plt.plot(indices, future_terms, label="Predicted Terms", marker='x', linestyle='--', color='orange')
plt.xlabel("Index")
plt.ylabel("Term Value")
plt.title("OEIS Sequence Prediction: Expected vs. Predicted Future Terms")
plt.legend()
plt.grid(True)
plt.show()