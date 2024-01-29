import scipy.io
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from scipy.io import savemat
import pandas as pd
from numpy import linalg as LA
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

parts=6
norm_error=[]
tt=[]
for ii in range(parts-3):
    filename = f"data_d/d_actual{ii}.mat"
    mat_data = scipy.io.loadmat(filename)
    if ii==0:
        time_series_data = mat_data['error1'].squeeze()
    else:
        time_series_data = mat_data['y_test_pred'].squeeze()
    print(time_series_data.shape)
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
    if ii==0:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=((parts-ii-1)/(parts-ii)), shuffle=False)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=((parts-ii-1)/(parts-ii)), shuffle=False)


    # # Convert to PyTorch tensors and reshape
    # X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1)
    # y_train = torch.tensor(y_train, dtype=torch.float32)

    # # From (ii+1)*delta t to T
    # X_test = torch.tensor(X_test, dtype=torch.float32).unsqueeze(-1)
    # y_test = torch.tensor(y_test, dtype=torch.float32)

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
    epochs = 2000

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
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")

    # Evaluate model
    model.eval()
    # y_test_pred = model(X_test).detach().numpy()
    y_test_pred = model(X_test_scaled).detach().numpy()

    # # Plot real vs predicted values
    # plt.figure(figsize=(15, 6))
    # plt.plot(y_test, label="Real")
    # plt.plot(y_test_pred, label="Predicted")
    # plt.legend()
    # plt.title("Real vs Predicted Values")
    # plt.show()
    filename = f"data_d/d_predictions{ii}.mat"
    X=torch.tensor(X, dtype=torch.float32).unsqueeze(-1)
    if ii==0:
    # y_test_pred = model(X).detach().numpy()
        y_test_pred = model(X_test_scaled).detach().numpy()
    else:
        y_test_pred = model(X_test_scaled).detach().numpy()

    ###
    savemat(filename, {'y_test_pred': y_test_pred})
    # savemat('d_predictions.mat', {'y_test_pred': y_test_pred})


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
        tau_p += disturbance_3d[0]-disturbance_estimate_3d[0]
        tau_q += disturbance_3d[1]-disturbance_estimate_3d[1]
        tau_r += disturbance_3d[2]-disturbance_estimate_3d[2]

        # Euler's rotational equations
        p_dot = (tau_p + (Iz - Iy) * q * r) / Ix
        q_dot = (tau_q + (Ix - Iz) * p * r) / Iy
        r_dot = (tau_r + (Iy - Ix) * p * q) / Iz

        return np.array([p_dot, q_dot, r_dot]), np.array([tau_p, tau_q, tau_r])
    def euler_rates(attitude_rates, euler_angles):
        p, q, r = attitude_rates
        phi, theta, _ = euler_angles

        phi_dot = p + q * np.sin(phi) * np.tan(theta) + r * np.cos(phi) * np.tan(theta)
        theta_dot = q * np.cos(phi) - r * np.sin(phi)
        psi_dot = q * np.sin(phi) / np.cos(theta) + r * np.cos(phi) / np.cos(theta)

        return np.array([phi_dot, theta_dot, psi_dot])


    # Initialize Euler angles and their log
    euler_angles = np.array([0.000001, 0.000002, 0.000002])  # Initial angles: phi, theta, psi
    euler_log = [euler_angles]
    controls_log=[]
    # Load disturbances from error1.mat
    filename=f"data_d/d_actual{ii}.mat"
    mat_data = loadmat(filename)
    if ii==0:
        disturbances = mat_data['error1'].squeeze()
    else:
        disturbances = mat_data['y_test_pred'].squeeze()

    filename = f"data_d/d_predictions{ii}.mat"
    mat_data = loadmat(filename)
    
    disturbances_estimate = mat_data['y_test_pred'].squeeze()

    #### Test
    # mat_data = loadmat('d_predictions.mat')
    # disturbances = mat_data['y_test_pred'].squeeze()
    # mat_data = loadmat('d_predictions_1.mat')
    # disturbances_estimate = mat_data['y_test_pred'].squeeze()
    # Simulation parameters
    
    frac=1/parts

    time_steps=int((1-(ii+1)/parts)*86400)
    dt = 0.1
    moments_of_inertia = [38.33, 345.0, 345.0]  # Dummy values; adjust as necessary

    # Initialize PID controllers for p, q, r rates
    pid_p = PIDController(Kp=1.0, Ki=0.1, Kd=0.01, set_point=0)
    pid_q = PIDController(Kp=1.0, Ki=0.1, Kd=0.01, set_point=0)
    pid_r = PIDController(Kp=1.0, Ki=0.1, Kd=0.01, set_point=0)

    # Initial attitude rates
    attitude_rates = np.array([0.000002, 0.000003, 0.000003])

    # Log for simulation
    attitudes_log = [attitude_rates]
    print(disturbances.shape)
    print(disturbances_estimate.shape)
    print(time_steps+int(frac*86400))
    print(time_steps)
    # Simulate
    for t in range(min(time_steps,len(disturbances_estimate))):
        control_input_p = pid_p.compute(attitude_rates[0])
        control_input_q = pid_q.compute(attitude_rates[1])
        control_input_r = pid_r.compute(attitude_rates[2])
        control_input = np.array([control_input_p, control_input_q, control_input_r])

        rate_dots, controls = attitude_dynamics(attitude_rates, moments_of_inertia, control_input, disturbances[t+int(frac*86400)-1],disturbances_estimate[t])
        # rate_dots = attitude_dynamics(attitude_rates, moments_of_inertia, control_input, disturbances[t+int(frac*86400)-1],disturbances[t+int(frac*86400)-1])
        for k in range(10):
            attitude_rates = attitude_rates + rate_dots * dt
            
        # Update Euler angles
        euler_dot = euler_rates(attitude_rates, euler_angles)
        for l in range(10):
            euler_angles = euler_angles + euler_dot * dt

        attitudes_log.append(attitude_rates.copy())  # Use copy() to append a separate array
        euler_log.append(euler_angles.copy())
        controls_log.append(controls.copy())


    euler_log=np.array(euler_log)
    controls_log=np.array(controls_log)
    # print(ii)
    # Convert the log to a numpy array for easy indexing
    attitudes_log = np.array(attitudes_log)
    norm_error.append(LA.norm(euler_log))
    tt.append(ii)
    print(norm_error)
    if ii==parts-1:
        norms_df = pd.DataFrame({
        'time': tt,
        'norm': norm_error
        })
        filename=f"data/norm.csv"
        norms_df.to_csv(filename, index=False)
    print(norm_error)
    print(tt)
    euler_log=np.array(euler_log)
    y_test_pred=attitudes_log[:,0]
    y_test_pred=np.transpose(y_test_pred)
    filename=f"data_d/d_actual{ii+1}.mat"
    savemat(filename, {'y_test_pred': y_test_pred})

    print(np.linalg.norm(attitudes_log[1000:,:]))
    # print(attitudes_log)
    # Plot the results for p, q, r rates
    plt.figure(figsize=(15, 9))
    offset=2000
    print(controls_log[:,0])
    t=np.linspace(ii*frac*86400+offset,ii*frac*86400+min(frac*86400,len(attitudes_log)), min(int(frac*86400),len(attitudes_log))-offset)
    print(len(t))
    print(len(controls_log[0+offset:min(int(frac*86400),len(attitudes_log))-1, 0]))
    controls_df = pd.DataFrame({
    'time': t,  # Assuming time or index values
    'control': controls_log[0+offset:min(int(frac*86400),len(attitudes_log)), 0]
    })
    filename=f"data/controls_data_1_iteration_{ii}.csv"
    controls_df.to_csv(filename, index=False)
    controls_df = pd.DataFrame({
    'time': t,  # Assuming time or index values
    'control': controls_log[0+offset:min(int(frac*86400),len(attitudes_log)), 1]
    })
    filename=f"data/controls_data_2_iteration_{ii}.csv"
    controls_df.to_csv(filename, index=False)
    controls_df = pd.DataFrame({
    'time': t,  # Assuming time or index values
    'control': controls_log[0+offset:min(int(frac*86400),len(attitudes_log)), 2]
    })
    filename=f"data/controls_data_3_iteration_{ii}.csv"
    controls_df.to_csv(filename, index=False)
    ### Data saving
    attitudes_df = pd.DataFrame({
    'time': t,  # Assuming time or index values
    'p_rate': attitudes_log[0+offset:min(int(frac*86400),len(attitudes_log)), 0]/1e-6
    })
    filename=f"data/attitudes_data_p_iteration_{ii}.csv"
    attitudes_df.to_csv(filename, index=False)
    attitudes_df = pd.DataFrame({
    'time': t,  # Assuming time or index values
    'p_rate': attitudes_log[0+offset:min(int(frac*86400),len(attitudes_log)), 1]/1e-6
    })
    filename=f"data/attitudes_data_q_iteration_{ii}.csv"
    attitudes_df.to_csv(filename, index=False)
    attitudes_df = pd.DataFrame({
    'time': t,  # Assuming time or index values
    'p_rate': attitudes_log[0+offset:min(int(frac*86400),len(attitudes_log)), 2]/1e-6
    })
    filename=f"data/attitudes_data_r_iteration_{ii}.csv"
    attitudes_df.to_csv(filename, index=False)

    euler_df = pd.DataFrame({
    'time': t,  # Assuming time or index values
    'p_rate': euler_log[0+offset:int(frac*86400), 0]/4.84814e-6
    })
    filename=f"data/attitudes_data_phi_iteration_{ii}.csv"
    euler_df.to_csv(filename, index=False)
    euler_df = pd.DataFrame({
    'time': t,  # Assuming time or index values
    'p_rate': euler_log[0+offset:int(frac*86400), 1]/4.84814e-6
    })
    filename=f"data/attitudes_data_theta_iteration_{ii}.csv"
    euler_df.to_csv(filename, index=False)
    euler_df = pd.DataFrame({
    'time': t,  # Assuming time or index values
    'p_rate': euler_log[0+offset:int(frac*86400), 2]/4.84814e-6
    })
    filename=f"data/attitudes_data_psi_iteration_{ii}.csv"
    euler_df.to_csv(filename, index=False)





    ####### Ploting
    t=np.linspace(ii*frac*86400+offset,ii*frac*86400+min(frac*86400,len(attitudes_log)), min(int(frac*86400),len(attitudes_log))-offset)
    
    plt.subplot(3, 1, 1)
    print(len(attitudes_log))
    plt.plot(t,attitudes_log[0+offset:min(int(frac*86400),len(attitudes_log)), 0]/1e-6, label="p (1e-6rad/s)")
    # plt.axhline(y=0, color='r', linestyle='--')
    plt.title("p rate over Time")
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(t,attitudes_log[0+offset:min(int(frac*86400),len(attitudes_log)), 1]/1e-6, label="q (1e-6rad/s)")
    # plt.axhline(y=0, color='r', linestyle='--')
    plt.title("q rate over Time")
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(t,attitudes_log[0+offset:min(int(frac*86400),len(attitudes_log)), 2]/1e-6, label="r (1e-6rad/s)")
    # plt.axhline(y=0, color='r', linestyle='--')
    plt.title("r rate over Time")
    plt.legend()

    plt.tight_layout()
    #plt.show()
    filename=f"figures/attitudes_plot{ii}.png"
    plt.savefig(filename)
    plt.close()



    # Additional plots for Euler angles
    plt.figure(figsize=(15, 9))

    plt.subplot(3, 1, 1)
    plt.plot(t,euler_log[0+offset:int(frac*86400), 0]/4.84814e-6, label="Roll (arcsec)")
    plt.title("Roll over Time")
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(t,euler_log[0+offset:int(frac*86400), 1]/4.84814e-6, label="Pitch (arcsec)")
    plt.title("Pitch over Time")
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(t,euler_log[0+offset:int(frac*86400), 2]/4.84814e-6, label="Yaw (arcsec)")
    plt.title("Yaw over Time")
    plt.legend()

    plt.tight_layout()
    #plt.show()
    filename=f"figures/rates_plot{ii}.png"
    plt.savefig(filename)
    plt.close()


    def plot_power_spectral_density(time_series, sample_spacing=1.0):
        # Compute the Power Spectral Density (PSD)
        fft_result = np.fft.fft(time_series)
        psd = np.abs(fft_result) ** 2

        # Compute the frequency bins
        sample_count = len(time_series)
        frequencies = np.fft.fftfreq(sample_count, sample_spacing)

        # Only plot the positive frequencies
        positive_frequencies = frequencies[:sample_count // 2]
        positive_psd = psd[:sample_count // 2]
        return positive_frequencies, positive_psd

    def plot_frequency_spectrum(time_series, sample_spacing=1.0):

        # Compute the FFT
        fft_result = np.fft.fft(time_series)

        # Get the magnitudes
        magnitudes = np.abs(fft_result)

        # Compute the frequency bins
        sample_count = len(time_series)
        frequencies = np.fft.fftfreq(sample_count, sample_spacing)

        # Only plot the positive frequencies
        positive_frequencies = frequencies[:sample_count // 2]
        positive_magnitudes = magnitudes[:sample_count // 2]

        # Filtering the frequencies to be within the range 10^-3 to 10^-1
        filter_mask = (positive_frequencies >= 1e-3) & (positive_frequencies <= 1e-1)
        filtered_frequencies = positive_frequencies[filter_mask]
        filtered_magnitudes = positive_magnitudes[filter_mask]
        return filtered_frequencies, filtered_magnitudes

    # Plotting the Frequencies and Magnitudes
    filtered_frequencies, filtered_magnitudes=plot_frequency_spectrum(attitudes_log[:,0], sample_spacing=1.0)
    frequency_df = pd.DataFrame({
    'time': filtered_frequencies,  # Assuming time or index values
    'p_rate': filtered_magnitudes
    })
    filename=f"data/frequency_attitudes_p_iteration_{ii}.csv"
    frequency_df.to_csv(filename, index=False)
    plt.figure(figsize=(10, 4))
    plt.plot(filtered_frequencies, filtered_magnitudes)
    plt.title('Frequency Spectrum (10^-4 to 10^-2 Hz)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.grid()
    plt.xscale('log')  # Log scale for x-axis
    filename=f"figures/frequency_spectrum_attitude_p_{ii}.png"
    plt.savefig(filename)
    plt.close()

    # Plotting the Frequencies and Magnitudes
    filtered_frequencies, filtered_magnitudes=plot_frequency_spectrum(attitudes_log[:,1], sample_spacing=1.0)
    frequency_df = pd.DataFrame({
    'time': filtered_frequencies,  # Assuming time or index values
    'p_rate': filtered_magnitudes
    })
    filename=f"data/frequency_attitudes_q_iteration_{ii}.csv"
    frequency_df.to_csv(filename, index=False)
    plt.figure(figsize=(10, 4))
    plt.plot(filtered_frequencies, filtered_magnitudes)
    plt.title('Frequency Spectrum (10^-4 to 10^-2 Hz)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.grid()
    plt.xscale('log')  # Log scale for x-axis
    filename=f"figures/frequency_spectrum_attitude_q_{ii}.png"
    plt.savefig(filename)
    plt.close()
    # Plotting the Frequencies and Magnitudes
    filtered_frequencies, filtered_magnitudes=plot_frequency_spectrum(attitudes_log[:,2], sample_spacing=1.0)
    frequency_df = pd.DataFrame({
    'time': filtered_frequencies,  # Assuming time or index values
    'p_rate': filtered_magnitudes
    })
    filename=f"data/frequency_attitudes_r_iteration_{ii}.csv"
    frequency_df.to_csv(filename, index=False)
    plt.figure(figsize=(10, 4))
    plt.plot(filtered_frequencies, filtered_magnitudes)
    plt.title('Frequency Spectrum (10^-4 to 10^-2 Hz)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.grid()
    plt.xscale('log')  # Log scale for x-axis
    filename=f"figures/frequency_spectrum_attitude_r_{ii}.png"
    plt.savefig(filename)
    plt.close()
    # Plotting the Frequencies and Magnitudes
    filtered_frequencies, filtered_magnitudes=plot_frequency_spectrum(euler_log[:,0], sample_spacing=1.0)
    frequency_df = pd.DataFrame({
    'time': filtered_frequencies,  # Assuming time or index values
    'p_rate': filtered_magnitudes
    })
    filename=f"data/frequency_attitudes_phit_iteration_{ii}.csv"
    frequency_df.to_csv(filename, index=False)
    plt.figure(figsize=(10, 4))
    plt.plot(filtered_frequencies, filtered_magnitudes)
    plt.title('Frequency Spectrum (10^-4 to 10^-2 Hz)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.grid()
    plt.xscale('log')  # Log scale for x-axis
    filename=f"figures/frequency_spectrum_euler_phi_{ii}.png"
    plt.savefig(filename)
    plt.close()
    # Plotting the Frequencies and Magnitudes
    filtered_frequencies, filtered_magnitudes=plot_frequency_spectrum(euler_log[:,1], sample_spacing=1.0)
    frequency_df = pd.DataFrame({
    'time': filtered_frequencies,  # Assuming time or index values
    'p_rate': filtered_magnitudes
    })
    filename=f"data/frequency_attitudes_theta_iteration_{ii}.csv"
    frequency_df.to_csv(filename, index=False)
    plt.figure(figsize=(10, 4))
    plt.plot(filtered_frequencies, filtered_magnitudes)
    plt.title('Frequency Spectrum (10^-4 to 10^-2 Hz)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.grid()
    plt.xscale('log')  # Log scale for x-axis
    filename=f"figures/frequency_spectrum_euler_psi_{ii}.png"
    plt.savefig(filename)
    plt.close()
    # Plotting the Frequencies and Magnitudes
    filtered_frequencies, filtered_magnitudes=plot_frequency_spectrum(euler_log[:,2], sample_spacing=1.0)
    frequency_df = pd.DataFrame({
    'time': filtered_frequencies,  # Assuming time or index values
    'p_rate': filtered_magnitudes
    })
    filename=f"data/frequency_attitudes_psi_iteration_{ii}.csv"
    frequency_df.to_csv(filename, index=False)
    plt.figure(figsize=(10, 4))
    plt.plot(filtered_frequencies, filtered_magnitudes)
    plt.title('Frequency Spectrum (10^-4 to 10^-2 Hz)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.grid()
    plt.xscale('log')  # Log scale for x-axis
    filename=f"figures/frequency_spectrum_euler_theta_{ii}.png"
    plt.savefig(filename)
    plt.close()

    def plot_frequency_domain(time_series):
        """
        Converts a time series to the frequency domain using FFT and plots the result.

        Parameters:
        time_series (array-like): The time series data.
        """
        # Compute the FFT
        fft_result = np.fft.fft(time_series)

        # Get the magnitudes
        magnitudes = np.abs(fft_result)

        # Compute the frequency bins
        sample_count = len(time_series)
        sample_spacing = 1  # Assuming unit sample spacing
        frequencies = np.fft.fftfreq(sample_count, sample_spacing)

    # Plotting the Frequencies and Magnitudes
        plt.figure(figsize=(10, 4))
        plt.plot(frequencies, magnitudes)
        plt.title('Frequency Domain Representation')
        plt.xlabel('Frequency')
        plt.ylabel('Magnitude')
        plt.grid()
        filename=f"figures/frequency_attitude{ii}.png"
        plt.savefig(filename)
        plt.close()
    # plot_frequency_domain(attitudes_log)
    # plot_frequency_spectrum(attitudes_log, sample_spacing=1.0) 