import scipy.io
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy.fft as fft

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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)

# Convert to PyTorch tensors and reshape
X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32).unsqueeze(-1)
y_test = torch.tensor(y_test, dtype=torch.float32)

# Define LSTM model
class TimeSeriesPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=1):
        super(TimeSeriesPredictor, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        y_pred = self.linear(lstm_out[:, -1])
        return y_pred.squeeze()

# Model parameters
input_dim = 1
hidden_dim = 20
learning_rate = 0.01
epochs = 10

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
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")

# Evaluate model
model.eval()
y_test_pred = model(X_test).detach().numpy()

# Plot real vs predicted values
plt.figure(figsize=(15, 6))
plt.plot(y_test, label="Real")
plt.plot(y_test_pred, label="Predicted")
plt.legend()
plt.title("Real vs Predicted Values")
plt.show()
from scipy.io import savemat

# Save the y_test_pred array to a .mat file
savemat('d_predictions.mat', {'y_test_pred': y_test_pred})


import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt

class PIDController:
    def __init__(self, Kp, Ki, Kd, set_point=0):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.set_point = set_point
        self.prev_error = 0
        self.integral = 0
        
    def compute(self, current_value):
        error = self.set_point - current_value
        self.integral += error
        derivative = error - self.prev_error
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        self.prev_error = error
        return output

def attitude_dynamics(attitude_rates, moments_of_inertia, control_input, disturbance,disturbance_estimate):
    p, q, r = attitude_rates
    Ix, Iy, Iz = moments_of_inertia
    tau_p, tau_q, tau_r = control_input

    # Convert the single-element disturbance into a 3-element array
    disturbance_3d = np.array([disturbance, disturbance, disturbance])
    disturbance_estimate_3d = np.array([disturbance_estimate, disturbance_estimate, disturbance_estimate])


    # Incorporate disturbances into external torques
    tau_p += disturbance_3d[0]-0*disturbance_estimate_3d[0]
    tau_q += disturbance_3d[1]-0*disturbance_estimate_3d[1]
    tau_r += disturbance_3d[2]-0*disturbance_estimate_3d[2]

    # Euler's rotational equations
    p_dot = (tau_p + (Iz - Iy) * q * r) / Ix
    q_dot = (tau_q + (Ix - Iz) * p * r) / Iy
    r_dot = (tau_r + (Iy - Ix) * p * q) / Iz

    return np.array([p_dot, q_dot, r_dot])
def euler_rates(attitude_rates, euler_angles):
    p, q, r = attitude_rates
    phi, theta, _ = euler_angles

    phi_dot = p + q * np.sin(phi) * np.tan(theta) + r * np.cos(phi) * np.tan(theta)
    theta_dot = q * np.cos(phi) - r * np.sin(phi)
    psi_dot = q * np.sin(phi) / np.cos(theta) + r * np.cos(phi) / np.cos(theta)

    return np.array([phi_dot, theta_dot, psi_dot])


# Initialize Euler angles and their log
euler_angles = np.array([0.00001, 0.00002, 0.00002])  # Initial angles: phi, theta, psi
euler_log = [euler_angles]

# Load disturbances from error1.mat
mat_data = loadmat('error1.mat')
disturbances = mat_data['error1'].squeeze()
mat_data = loadmat('d_predictions.mat')
disturbances_estimate = mat_data['y_test_pred'].squeeze()
# Simulation parameters
time_steps = int(0.3*len(disturbances))
dt = 0.1
moments_of_inertia = [1.0, 2.0, 3.0]  # Dummy values; adjust as necessary

# Initialize PID controllers for p, q, r rates
pid_p = PIDController(Kp=1.0, Ki=0.1, Kd=0.01, set_point=0)
pid_q = PIDController(Kp=1.0, Ki=0.1, Kd=0.01, set_point=0)
pid_r = PIDController(Kp=1.0, Ki=0.1, Kd=0.01, set_point=0)

# Initial attitude rates
attitude_rates = np.array([0.00002, 0.00003, 0.00003])

# Log for simulation
attitudes_log = [attitude_rates]

# Simulate
for t in range(time_steps):
    for _ in range(10):
        control_input_p = pid_p.compute(attitude_rates[0])
        control_input_q = pid_q.compute(attitude_rates[1])
        control_input_r = pid_r.compute(attitude_rates[2])

        control_input = np.array([control_input_p, control_input_q, control_input_r])

        rate_dots = attitude_dynamics(attitude_rates, moments_of_inertia, control_input, disturbances[t+60480],disturbances_estimate[t-1])
        attitude_rates = attitude_rates + rate_dots * dt
        # Update Euler angles
        euler_dot = euler_rates(attitude_rates, euler_angles)
        euler_angles = euler_angles + euler_dot * dt

        attitudes_log.append(attitude_rates.copy())  # Use copy() to append a separate array
        euler_log.append(euler_angles.copy())

# Convert the log to a numpy array for easy indexing
attitudes_log = np.array(attitudes_log)
euler_log=np.array(euler_log)
# print(attitudes_log)
# Plot the results for p, q, r rates
plt.figure(figsize=(15, 9))

plt.subplot(3, 1, 1)
plt.plot(attitudes_log[1000:, 0]/1e-6, label="p (1e-6rad/s)")
# plt.axhline(y=0, color='r', linestyle='--')
plt.title("p rate over Time")
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(attitudes_log[1000:, 1]/1e-6, label="q (1e-6rad/s)")
# plt.axhline(y=0, color='r', linestyle='--')
plt.title("q rate over Time")
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(attitudes_log[1000:, 2]/1e-6, label="r (1e-6rad/s)")
# plt.axhline(y=0, color='r', linestyle='--')
plt.title("r rate over Time")
plt.legend()

plt.tight_layout()
plt.show()

# Additional plots for Euler angles
plt.figure(figsize=(15, 9))

plt.subplot(3, 1, 1)
plt.plot(euler_log[1000:, 0]/4.84814e-6, label="Roll (arcsec)")
plt.title("Roll over Time")
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(euler_log[1000:, 1]/4.84814e-6, label="Pitch (arcsec)")
plt.title("Pitch over Time")
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(euler_log[1000:, 2]/4.84814e-6, label="Yaw (arcsec)")
plt.title("Yaw over Time")
plt.legend()

plt.tight_layout()
plt.show()

def compute_psd(data, dt):
    # Compute the FFT and the corresponding frequencies
    data_fft = fft.fft(data)
    frequencies = fft.fftfreq(len(data), dt)
    
    # Compute the power spectral density (PSD)
    psd = np.abs(data_fft)**2
    
    return frequencies, psd


# Compute PSD for euler_log and attitudes_log
euler_frequencies, euler_psd = compute_psd(euler_log, dt)
attitudes_frequencies, attitudes_psd = compute_psd(attitudes_log, dt)

# Filter function to get values within the desired frequency range
def filter_frequency(frequencies, psd):
    mask = (frequencies >= 1e-3) & (frequencies <= 1e-2)
    return frequencies[mask], psd[mask]

# Filter the frequencies and PSD values
euler_frequencies_filtered, euler_psd_filtered = filter_frequency(euler_frequencies, euler_psd)
attitudes_frequencies_filtered, attitudes_psd_filtered = filter_frequency(attitudes_frequencies, attitudes_psd)

# Plotting
plt.figure(figsize=(15, 12))

# Plot for euler_log
plt.subplot(3, 1, 1)
plt.plot(euler_frequencies_filtered, euler_psd_filtered[:, 0], label="Roll (phi) PSD")
plt.title("Power Spectral Density of Roll")
plt.xlabel("Frequency (Hz)")
plt.ylabel("PSD")
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(euler_frequencies_filtered, euler_psd_filtered[:, 1], label="Pitch (theta) PSD")
plt.title("Power Spectral Density of Pitch")
plt.xlabel("Frequency (Hz)")
plt.ylabel("PSD")
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(euler_frequencies_filtered, euler_psd_filtered[:, 2], label="Yaw (psi) PSD")
plt.title("Power Spectral Density of Yaw")
plt.xlabel("Frequency (Hz)")
plt.ylabel("PSD")
plt.legend()

# Separate figure for attitudes_log
plt.figure(figsize=(15, 9))

plt.subplot(3, 1, 1)
plt.plot(attitudes_frequencies_filtered, attitudes_psd_filtered[:, 0], label="p rate PSD")
plt.title("Power Spectral Density of p rate")
plt.xlabel("Frequency (Hz)")
plt.ylabel("PSD")
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(attitudes_frequencies_filtered, attitudes_psd_filtered[:, 1], label="q rate PSD")
plt.title("Power Spectral Density of q rate")
plt.xlabel("Frequency (Hz)")
plt.ylabel("PSD")
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(attitudes_frequencies_filtered, attitudes_psd_filtered[:, 2], label="r rate PSD")
plt.title("Power Spectral Density of r rate")
plt.xlabel("Frequency (Hz)")
plt.ylabel("PSD")
plt.legend()

plt.tight_layout()
plt.show()
