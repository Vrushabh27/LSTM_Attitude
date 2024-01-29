import scipy.io
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load the .mat file and extract data
mat_data = scipy.io.loadmat('error1.mat')
time_series_data = mat_data['error1'].squeeze()

# Function to create input-output sequences for multi-step prediction
def create_sequences(data, input_seq_length, output_seq_length):
    X, y = [], []
    for i in range(len(data) - input_seq_length - output_seq_length + 1):
        X.append(data[i:(i + input_seq_length)])
        y.append(data[(i + input_seq_length):(i + input_seq_length + output_seq_length)])
    return np.array(X), np.array(y)

# Define sequence lengths for input and output
input_seq_length = 10
output_seq_length = 5  # For example, predict the next 3 steps
X, y = create_sequences(time_series_data, input_seq_length, output_seq_length)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Initialize MinMaxScaler and scale the input features
scaler = MinMaxScaler()

# Reshape and scale data
X_train_reshaped = X_train.reshape(-1, 1)
X_test_reshaped = X_test.reshape(-1, 1)
y_train_reshaped = y_train.reshape(-1, 1)
y_test_reshaped = y_test.reshape(-1, 1)

# Scale data
X_train_scaled = scaler.fit_transform(X_train_reshaped).reshape(X_train.shape)
X_test_scaled = scaler.transform(X_test_reshaped).reshape(X_test.shape)
y_train_scaled = scaler.fit_transform(y_train_reshaped).reshape(y_train.shape)
y_test_scaled = scaler.transform(y_test_reshaped).reshape(y_test.shape)

# Convert scaled data to PyTorch tensors
X_train_scaled = torch.tensor(X_train_scaled, dtype=torch.float32).unsqueeze(-1)
y_train_scaled = torch.tensor(y_train_scaled, dtype=torch.float32)
X_test_scaled = torch.tensor(X_test_scaled, dtype=torch.float32).unsqueeze(-1)
y_test_scaled = torch.tensor(y_test_scaled, dtype=torch.float32)

# Define LSTM model for multi-step prediction
class MultiStepLSTMPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_seq_length, num_layers=1):
        super(MultiStepLSTMPredictor, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, output_seq_length)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        y_pred = self.linear(lstm_out[:, -1])
        return y_pred

# Model parameters
input_dim = 1
hidden_dim = 32
learning_rate = 0.001
epochs = 2000

# Instantiate model, define loss and optimizer
model = MultiStepLSTMPredictor(input_dim, hidden_dim, output_seq_length)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    y_pred = model(X_train_scaled)
    loss = criterion(y_pred, y_train_scaled.view(-1, output_seq_length))
    loss.backward()
    optimizer.step()
    if epoch % 50 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")

# Evaluate model
model.eval()
y_test_pred = model(X_test_scaled).detach().numpy()
y_test_actual = y_test_scaled.numpy()

# Plotting function for multi-step prediction
def plot_multi_step(t, actual, predicted, title):
    plt.figure(figsize=(10, 6))
    num_sequences = len(actual)
    for i in range(num_sequences):
        plt.plot(t[i], actual[i], label=f"Actual {i+1}")
        plt.plot(t[i], predicted[i], label=f"Predicted {i+1}", linestyle='--')
    plt.legend()
    plt.title(title)
    plt.show()

# Prepare data for plotting
t = np.linspace(0, output_seq_length-1, output_seq_length)
# plot_multi_step(np.tile(t, (len(y_test_actual), 1)), y_test_actual, y_test_pred, "Multi-step Prediction")
y_diff = y_test - y_test_pred
t = np.linspace(0, len(y_test), len(y_test))

# Saving and plotting functions
def save_and_plot_data(filename, y_data, label):
    pred_df = pd.DataFrame({'time': np.arange(len(y_data.flatten())), 'pred': y_data.flatten()})
    pred_df.to_csv(filename, index=False)

    plt.plot(y_data.flatten(), label=label)

# Call the functions to save and plot
save_and_plot_data("data/lstm_data_actual.csv", y_test_actual, "Real")
save_and_plot_data("data/lstm_data_pred.csv", y_test_pred, "Predicted")
save_and_plot_data("data/lstm_data_diff.csv", y_diff, "Difference")

plt.legend()
plt.title("Real vs Predicted Values")
plt.show()