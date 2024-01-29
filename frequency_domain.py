import numpy as np
import matplotlib.pyplot as plt

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
    plt.show()

# Generating a sample time series data: a sine wave
time = np.linspace(0, 1, 500)
amplitude = np.sin(2 * np.pi * 5 * time)  # Sine wave with frequency of 5 Hz
print(amplitude.shape)
plot_frequency_domain(amplitude)
