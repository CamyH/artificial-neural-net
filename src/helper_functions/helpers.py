# Mean Squared Error (MSE) chosen for regression
# y = observed values
# pred = predicted values/yhat
def mse(y, pred):
    return (pred - y) ** 2

# Split the data into train and test data
def train_test_split(arr, percent=0.2):
    split_idx = int(len(arr) * percent)
    arr_train = arr[:split_idx]
    arr_test = arr[split_idx:]

    return arr_train, arr_test
