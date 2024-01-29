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

# Function to create input-output sequences
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        seq = data[i:i+seq_length]
        X.append(seq)
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

# Define sequence length and create sequences
seq_length = 5
X, y = create_sequences(time_series_data, seq_length)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Initialize MinMaxScaler and scale the input features
scaler = MinMaxScaler()

# Fit the scaler using only the training data and transform training data
X_train_scaled = scaler.fit_transform(X_train.reshape(-1, 1)).reshape(X_train.shape)

# Transform the test data with the same scaler
X_test_scaled = scaler.transform(X_test.reshape(-1, 1)).reshape(X_test.shape)

# Convert scaled data to PyTorch tensors and reshape
X_train_scaled = torch.tensor(X_train_scaled, dtype=torch.float32).unsqueeze(-1)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test_scaled = torch.tensor(X_test_scaled, dtype=torch.float32).unsqueeze(-1)
y_test = torch.tensor(y_test, dtype=torch.float32)

# Define LSTM model
class TimeSeriesPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=1, num_layers=1):
        super(TimeSeriesPredictor, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        y_pred = self.linear(lstm_out[:, -1])
        return y_pred.squeeze()

# Model parameters
input_dim = 1
hidden_dim = 32
learning_rate = 0.001
epochs = 1000

# Instantiate model, define loss and optimizer
model = TimeSeriesPredictor(input_dim, hidden_dim)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    y_pred = model(X_train_scaled)
    loss = criterion(y_pred, y_train)
    loss.backward()
    optimizer.step()
    if epoch % 50 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")

# Evaluate model
model.eval()
y_test_pred = model(X_test_scaled).detach().numpy()
y_diff = y_test - y_test_pred
t = np.linspace(0, len(y_test), len(y_test))

# Saving the data and plotting
def save_and_plot_data(filename, y_data, label):
    pred_df = pd.DataFrame({'time': t, 'pred': y_data})
    pred_df.to_csv(filename, index=False)

    plt.plot(y_data, label=label)

save_and_plot_data("data/lstm_data_actual.csv", y_test.numpy(), "Real")
save_and_plot_data("data/lstm_data_pred.csv", y_test_pred, "Predicted")
save_and_plot_data("data/lstm_data_diff.csv", y_diff.numpy(), "Difference")

plt.legend()
plt.title("Real vs Predicted Values")
plt.show()
