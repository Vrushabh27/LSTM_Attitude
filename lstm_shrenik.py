# %%
import numpy as np
from scipy.io import loadmat

data = loadmat('error1.mat')
data = data['error1'].T
print(data.shape)

# %%
from sklearn.preprocessing import MinMaxScaler, StandardScaler

scaler = StandardScaler()
normalized_data = scaler.fit_transform(data)
print(normalized_data.shape)

# %%
train_size = int(len(normalized_data) * 0.8)  # Play with train test ratio
test_size = len(normalized_data) - train_size

train_data = normalized_data[:train_size]
test_data = normalized_data[train_size:]
print(train_data.shape)
print(test_data.shape)

# %%
def create_sequences(data, sequence_length):
    xs, ys = [], []
    for i in range(len(data) - sequence_length):
        x = data[i:(i + sequence_length)]
        y = data[i + sequence_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

sequence_length = 5 # Try playing with these
X_train, y_train = create_sequences(train_data, sequence_length)
X_test, y_test = create_sequences(test_data, sequence_length)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

# %%
import torch
import torch.nn as nn

class LSTMNetwork(nn.Module):
    def __init__(self, input_size, hidden_layer_sizes, output_size):
        super(LSTMNetwork, self).__init__()
        self.hidden_layers = nn.ModuleList()
        self.hidden_sizes = hidden_layer_sizes

        for i in range(len(hidden_layer_sizes)):
            if i == 0:
                self.hidden_layers.append(nn.LSTM(input_size, hidden_layer_sizes[i], batch_first=True))
            else:
                self.hidden_layers.append(nn.LSTM(hidden_layer_sizes[i-1], hidden_layer_sizes[i], batch_first=True))

        self.relu = nn.ReLU()
        self.linear = nn.Linear(hidden_layer_sizes[-1], output_size)

    def forward(self, x):
        for lstm in self.hidden_layers:
            x, _ = lstm(x)
            x = self.relu(x)
        x = self.linear(x[:, -1, :]) # Taking the last time step output
        return x

input_size = 1 # as our data is 1-dimensional
hidden_layer_sizes = [256, 128, 64]
output_size = 1 # predicting one value

model = LSTMNetwork(input_size, hidden_layer_sizes, output_size)


# %%
import torch.optim as optim

# Loss function
criterion = nn.MSELoss()

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Try playing with the learning rate

# Early stopping
patience = 30   # Good patience level if the learning rate is small. If its large decrease the patience level
min_val_loss = np.Inf
patience_counter = 0

# %%
import time

def train_model(model, train_data, train_labels, criterion, optimizer, epochs, patience, min_val_loss):
    epoch_times = []
    train_losses = []

    for epoch in range(epochs):
        start_time = time.time()

        # Set model to training mode
        model.train()

        # Convert data to torch tensors
        train_data_torch = torch.from_numpy(train_data).float()
        train_labels_torch = torch.from_numpy(train_labels).float()

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        output = model(train_data_torch)
        loss = criterion(output, train_labels_torch)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())
        epoch_time = time.time() - start_time
        epoch_times.append(epoch_time)

        print(f'Epoch {epoch+1}/{epochs} - Loss: {loss.item():.4f} - Time: {epoch_time:.2f} sec')

        # Early stopping
        if loss.item() < min_val_loss:
            min_val_loss = loss.item()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping triggered")
            break

    return train_losses, epoch_times

epochs = 20
train_losses, epoch_times = train_model(model, X_train, y_train, criterion, optimizer, epochs, patience, min_val_loss)


# %%
# Save the Model
model_path = 'lstm_model.pth'
torch.save(model.state_dict(), model_path)
print("Model saved!")

# %%
import matplotlib.pyplot as plt
import matplotlib_inline
matplotlib_inline.backend_inline.set_matplotlib_formats('svg')
import seaborn as sns
sns.set_context("paper")
sns.set_style("ticks");

# %%
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss vs. Epochs')
plt.legend()
plt.show()

# %%
def multi_step_prediction(model, test_data, sequence_length):
    model.eval() # Set the model to evaluation mode
    test_predictions = []
    test_input = test_data[0,:sequence_length,:] # Initial sequence

    for _ in range(len(test_data) - sequence_length):
        with torch.no_grad():
            test_input_tensor = torch.from_numpy(test_input).float()
            prediction = model(test_input_tensor.unsqueeze(0))
            test_predictions.append(prediction.item())
            test_input = np.append(test_input[1:], prediction.item()).reshape(-1, 1)

    return test_predictions

test_predictions = np.array(multi_step_prediction(model, X_test, sequence_length))

# %% [markdown]
# Now lets see the plot for multi step prediction on test data

# %%
def denormalize(data, scaler):
    return scaler.inverse_transform(data.reshape(-1, 1)).reshape(-1)

denormalized_predictions = denormalize(np.array(test_predictions), scaler)
denormalized_ground_truth = denormalize(y_test, scaler)

plt.figure(figsize=(10, 6), dpi = 200)
plt.plot(test_predictions, label='Predictions (Normalized)')
plt.plot(y_test, label='Ground Truth (Normalized)')
plt.xlabel('Time Step')
plt.ylabel('Normalized Value')
plt.title('Predictions vs. Ground Truth (Normalized)')
plt.legend()
plt.show()

plt.figure(figsize=(10, 6), dpi = 200)
plt.plot(denormalized_predictions, label='Predictions (Unnormalized)')
plt.plot(denormalized_ground_truth, label='Ground Truth (Unnormalized)')
plt.xlabel('Time Step')
plt.ylabel('Value')
plt.title('Predictions vs. Ground Truth (Unnormalized)')
plt.legend()
plt.show()

# %%
error = denormalized_ground_truth[5:] - denormalized_predictions
plt.figure(figsize=(10, 6), dpi = 200)
plt.plot(error, label='Prediction Error')
plt.xlabel('Time Step')
plt.ylabel('Error Value')
plt.title('Error between Prediction and Ground Truth')
plt.legend()
plt.show()

# %% [markdown]
# **Bad right** !!! Its because the model is trained in a single step manner. But when making predictions on test data we are making it in a multi step manner (i.e input for last k time steps are used to predict the output at next step and that output is used along wit last k - 1 time steps to predict the output at next time step and so on.... Hope you get it. So the predictions error will start to accumulate at every time step.
# 
# 
# Now let's plot the **single step** prediction. Its pretty straightforward.

# %%
X_test = torch.from_numpy(X_test).float()
test_predictions = model(X_test)
test_predictions = test_predictions.detach().numpy()

# %%
def denormalize(data, scaler):
    return scaler.inverse_transform(data.reshape(-1, 1)).reshape(-1)

denormalized_predictions = denormalize(test_predictions, scaler)
denormalized_ground_truth = denormalize(y_test, scaler)

plt.figure(figsize=(10, 6), dpi = 200)
plt.plot(test_predictions, label='Predictions (Normalized)')
plt.plot(y_test, label='Ground Truth (Normalized)')
plt.xlabel('Time Step')
plt.ylabel('Normalized Value')
plt.title('Predictions vs. Ground Truth (Normalized)')
plt.legend()
plt.show()

plt.figure(figsize=(10, 6), dpi = 200)
plt.plot(denormalized_predictions, label='Predictions (Unnormalized)')
plt.plot(denormalized_ground_truth, label='Ground Truth (Unnormalized)')
plt.xlabel('Time Step')
plt.ylabel('Value')
plt.title('Predictions vs. Ground Truth (Unnormalized)')
plt.legend()
plt.show()

# %%
error = denormalized_ground_truth - denormalized_predictions
plt.figure(figsize=(10, 6), dpi = 200)
plt.plot(error, label='Prediction Error')
plt.xlabel('Time Step')
plt.ylabel('Error Value')
plt.title('Error between Prediction and Ground Truth')
plt.legend()
plt.show()

# %% [markdown]
# You can see that the predictions are relatively improved.


