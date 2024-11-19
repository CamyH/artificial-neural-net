from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from ann import ANN
from pso import PSO

# Define initial sequence (Fibonacci, for example)
sequence = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]
window_size = 3
terms_to_predict = 5

# Define manually the expected values for clarity (only for checking against predictions)
expected_future_values = [233, 377, 610, 987, 1597]  # Known Fibonacci sequence terms

# Prepare data
def prepare_sequence_data(sequence, window_size=3):
    X, y = [], []
    for i in range(len(sequence) - window_size):
        X.append(sequence[i:i + window_size])
        y.append(sequence[i + window_size])
    return np.array(X), np.array(y).reshape(-1, 1)

X, y = prepare_sequence_data(sequence, window_size=window_size)

# Scale data (log transform applied only if needed, and only to y)
input_scaler = StandardScaler()
X = input_scaler.fit_transform(X)

target_scaler = StandardScaler()
y = target_scaler.fit_transform(y)  # Log-transform removed to simplify

# Configure ANN and PSO
ann_architecture = [window_size, 64, 32, 16, 1]
ann_activations = ['relu', 'relu', 'relu', 'linear']
num_parameters = sum(w.size for w in ANN(ann_architecture, ann_activations).weights) + \
                 sum(b.size for b in ANN(ann_architecture, ann_activations).biases)

# Initialize PSO
pso = PSO(num_particles=50, num_parameters=num_parameters, ann_architecture=ann_architecture,
          ann_activations=ann_activations, data=X, labels=y, swarm_size=50, c1=1.5, c2=1.5, max_iter=300)

# Optimize using PSO
best_params = pso.optimize()
ann = ANN(ann_architecture, ann_activations)
ann.set_parameters(best_params)

# Predict next terms
def predict_next_terms(ann, input_scaler, target_scaler, initial_sequence, terms_to_predict=5):
    current_sequence = initial_sequence[-window_size:]
    predicted_terms = []
    for _ in range(terms_to_predict):
        scaled_input = input_scaler.transform([current_sequence])
        next_term = ann.forward(scaled_input[0].reshape(-1, 1))
        next_term = np.array(next_term).reshape(1, 1)
        next_term = float(target_scaler.inverse_transform(next_term).flatten()[0])  # Revert scaling
        predicted_terms.append(next_term)
        current_sequence = current_sequence[1:] + [next_term]
    return predicted_terms

# Generate predictions for future terms
future_terms = predict_next_terms(ann, input_scaler, target_scaler, sequence, terms_to_predict=terms_to_predict)

# Print predictions vs. expected values
print("Index | Expected Term | Predicted Term")
print("---------------------------------------")
for i, (expected, predicted) in enumerate(zip(expected_future_values, future_terms), start=len(sequence)):
    print(f"{i:<5} | {expected:<13} | {predicted:.2f}")

# Plot results
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
