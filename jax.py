import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
import optax
import scipy.io
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd

# Load the .mat file and extract data
mat_data = scipy.io.loadmat('error1.mat')
time_series_data = mat_data['error1'].squeeze()

# Function to create input-output sequences for multi-step prediction
def create_sequences(data, seq_length, prediction_length):
    X, y = [], []
    for i in range(len(data) - seq_length - prediction_length):
        seq_X = data[i:i+seq_length]
        seq_y = data[i+seq_length:i+seq_length+prediction_length]
        X.append(seq_X)
        y.append(seq_y)
    return np.array(X), np.array(y)

# Define sequence length and prediction length
seq_length = 15
prediction_length = 5
X, y = create_sequences(time_series_data, seq_length, prediction_length)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Convert to JAX arrays
X_train = jnp.array(X_train, dtype=jnp.float32)
y_train = jnp.array(y_train, dtype=jnp.float32)
X_test = jnp.array(X_test, dtype=jnp.float32)
y_test = jnp.array(y_test, dtype=jnp.float32)

# Define LSTM model for multi-step prediction
class LSTMModel(nn.Module):
    hidden_dim: int
    output_dim: int = 1

    @nn.compact
    def __call__(self, x):
        x = nn.LSTMCell()(x)
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(features=self.output_dim)(x)
        return x

# Initialize model
model = LSTMModel(hidden_dim=64)

# Initialize parameters
rng = jax.random.PRNGKey(0)
params = model.init(rng, jnp.ones((seq_length, 1)))

# Define loss function (MSE)
def mse_loss(params, inputs, targets):
    predictions = model.apply(params, inputs)
    loss = jnp.mean((predictions - targets) ** 2)
    return loss

# Define update step
def train_step(state, batch):
    inputs, targets = batch
    loss, grads = jax.value_and_grad(mse_loss)(state.params, inputs, targets)
    state = state.apply_gradients(grads=grads)
    return state, loss

# Training state
learning_rate = 0.001
optimizer = optax.adam(learning_rate)
state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=optimizer)

# Training loop
epochs = 20
batch = (X_train, y_train)

for epoch in range(epochs):
    state, loss = train_step(state, batch)
    if epoch % 50 == 0:
        print(f"Epoch {epoch}, Loss: {loss}")

# Prediction function
def predict(params, inputs):
    return model.apply(params, inputs)

# Predict and plot
y_pred = predict(state.params, X_test)

# Plot real vs predicted values
plt.figure(figsize=(15, 6))
plt.plot(y_test.flatten(), label="Real")
plt.plot(y_pred.flatten(), label="Predicted")
plt.legend()
plt.title("Real vs Predicted Values")
plt.show()
