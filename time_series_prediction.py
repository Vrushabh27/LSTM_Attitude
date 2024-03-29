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
seq_length = 15
X, y = create_sequences(time_series_data, seq_length)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Convert to PyTorch tensors and reshape
X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32).unsqueeze(-1)
y_test = torch.tensor(y_test, dtype=torch.float32)

# # Define LSTM model
# class TimeSeriesPredictor(nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim=1):
#         super(TimeSeriesPredictor, self).__init__()
#         self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
#         self.linear = nn.Linear(hidden_dim, output_dim)
        
#     def forward(self, x):
#         lstm_out, _ = self.lstm(x)
#         y_pred = self.linear(lstm_out[:, -1])
#         return y_pred.squeeze()

# Define LSTM model with increased hidden layers
class TimeSeriesPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=1, num_layers=2):
        super(TimeSeriesPredictor, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        y_pred = self.linear(lstm_out[:, -1])
        return y_pred.squeeze()

# Model parameters
input_dim = 1
hidden_dim = 64
learning_rate = 0.001
epochs = 2000

# Instantiate model, define loss and optimizer
model = TimeSeriesPredictor(input_dim, hidden_dim)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    y_pred = model(X_train)
    loss = criterion(y_pred, y_train)
    loss.backward()
    optimizer.step()
    if epoch % 50 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")

# Evaluate model
model.eval()
y_test_pred = model(X_test).detach().numpy()
y_diff=y_test-y_test_pred
t=np.linspace(0,len(y_test),len(y_test))
pred_df = pd.DataFrame({
'time': t,
'pred': y_test
})
filename=f"data/lstm_data_actual.csv"
pred_df.to_csv(filename, index=False)


pred_df = pd.DataFrame({
'time': t,
'pred': y_test_pred
})
filename=f"data/lstm_data_pred.csv"
pred_df.to_csv(filename, index=False)

pred_df = pd.DataFrame({
'time': t,
'pred': y_diff
})
filename=f"data/lstm_data_diff.csv"
pred_df.to_csv(filename, index=False)
# Plot real vs predicted values
plt.figure(figsize=(15, 6))
plt.plot(y_test, label="Real")
plt.plot(y_test_pred, label="Predicted")
plt.legend()
plt.title("Real vs Predicted Values")
plt.show()