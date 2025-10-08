import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from few import get_file_manager

# Load LISA ASD data (columns: frequency [Hz], ASD)
data = np.loadtxt(get_file_manager().get_file("LPA.txt"), skiprows=1)

# Convert ASD to PSD
frequencies = data[:, 0]
psd_values = data[:, 1] ** 2

# Create a smooth interpolation function
get_sensitivity = CubicSpline(frequencies, psd_values)

# Frequency array for plotting
f_plot = np.logspace(-4, -1, 500)  # 10^-4 to 10^-1 Hz

# Evaluate PSD
S_n = get_sensitivity(f_plot)

# Plot
plt.figure(figsize=(8,5))
plt.loglog(f_plot, S_n)
plt.xlabel("Frequency [Hz]")
plt.ylabel(r"$S_n(f)$ [Hz$^{-1}$]")
plt.title("LISA Sensitivity Curve")
plt.grid(True, which="both", ls="--")
plt.show()
