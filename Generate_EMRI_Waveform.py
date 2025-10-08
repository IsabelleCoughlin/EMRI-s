import time
import numpy as np
import matplotlib.pyplot as plt

from few.waveform import GenerateEMRIWaveform
from few.utils.constants import MTSUN_SI
from few.utils.utility import get_p_at_t
from few.utils.geodesic import get_fundamental_frequencies
from few.utils.fdutils import GetFDWaveformFromFD, GetFDWaveformFromTD
from few.trajectory.inspiral import EMRIInspiral
from few.trajectory.ode.flux import KerrEccEqFlux

from scipy.interpolate import CubicSpline
from few import get_file_manager

# produce sensitivity function
traj_module = EMRIInspiral(func=KerrEccEqFlux)

# import ASD
data = np.loadtxt(get_file_manager().get_file("LPA.txt"), skiprows=1)
data[:, 1] = data[:, 1] ** 2
# define PSD function
get_sensitivity = CubicSpline(*data.T)


# define inner product [eq 3 of https://www.nature.com/articles/s41550-022-01849-y]
def inner_product(x, y, psd):
    return 4 * np.real(np.sum(np.conj(x) * y / psd))

# Initialize waveform generators
# frequency domain generator
few_gen = GenerateEMRIWaveform(
    "FastKerrEccentricEquatorialFlux",
    sum_kwargs=dict(pad_output=True, output_type="fd", odd_len=True),
    return_list=True,
)

# time domain generator
td_gen = GenerateEMRIWaveform(
    "FastKerrEccentricEquatorialFlux",
    sum_kwargs=dict(pad_output=True, odd_len=True),
    return_list=True,
)

# Trick to share waveform generator between both few_gen and td_gen (and reduce
# memory consumption)
td_gen.waveform_generator.amplitude_generator = few_gen.waveform_generator.amplitude_generator

import gc
gc.collect()

# define the injection parameters
m1 = 0.5e6  # central object mass (solar masses)
a = 0.9    # dimensionless spin parameter for the primary - will be ignored in Schwarzschild waveform
m2 = 10.0  # secondary object mass (solar masses)
p0 = 12.0  # initial dimensionless semi-latus rectum
e0 = 0.1   # eccentricity

x0 = 1.0         # initial cos(inclination) - will be ignored in Schwarzschild waveform
qK = np.pi / 3   # polar spin angle
phiK = np.pi / 3 # azimuthal viewing angle
qS = np.pi / 3   # polar sky angle
phiS = np.pi / 3 # azimuthal viewing angle
dist = 1.0       # luminosity distance (Gpc)

# initial phases
Phi_phi0 = np.pi / 3
Phi_theta0 = 0.0
Phi_r0 = np.pi / 3

Tobs = 0.5  # observation time (years), if the inspiral is shorter, the it will be zero padded
dt = 5.0    # time interval (seconds)
mode_selection_threshold = 1e-4  # relative threshold for mode inclusion: only modes making a relative contribution to
            # the total power above this threshold will be included in the waveform.

waveform_kwargs = {
    "T": Tobs,
    "dt": dt,
    "mode_selection_threshold": mode_selection_threshold,
}

# get the initial p0 required for an inspiral of length Tobs, given the fixed values of the other parameters
p0 = get_p_at_t(
    traj_module,
    Tobs * 0.999,
    [m1, m2, a, e0, 1.0],       # list of trajectory arguments, with p removed
    index_of_p=3,               # index where to insert the new p values in the above traj_args list when traj_module is called
    index_of_a=2,               # index of a in the list of trajectory arguments when calling traj_module
    index_of_e=4,               # etc...
    index_of_x=5,
    traj_kwargs={},
    xtol=2e-12,                 # absolute tolerance for the brentq root finder
    rtol=8.881784197001252e-16, # relative tolerance for the brentq root finder
    bounds=None,
)
print("New p0: ", p0)

emri_injection_params = [
    m1,
    m2,
    a,
    p0,
    e0,
    x0,
    dist,
    qS,
    phiS,
    qK,
    phiK,
    Phi_phi0,
    Phi_theta0,
    Phi_r0,
]

# generate the TD signal, and time how long it takes
start = time.time()
data_channels_td = td_gen(*emri_injection_params, **waveform_kwargs) # Returns 2 arrays containing the plus and cross polarizations
end = time.time()
print("Time taken to generate the TD signal: ", end - start, "seconds")

# take the FFT of the plus polarization and shift it
fft_TD = np.fft.fftshift(np.fft.fft(data_channels_td[0])) * dt
freq = np.fft.fftshift(np.fft.fftfreq(len(data_channels_td[0]), dt))

# define the positive frequencies
positive_frequency_mask = freq >= 0.0
'''
plt.figure()
plt.loglog(freq[positive_frequency_mask], np.abs(fft_TD[positive_frequency_mask]) ** 2)
plt.loglog(
    freq[positive_frequency_mask], get_sensitivity(freq[positive_frequency_mask])
)
plt.ylabel(r"$| {\rm DFT} [h_{+}]|^2$")
plt.xlabel(r"$f$ [Hz]")
plt.xlim(1e-4, 1e-1)
plt.show()
'''
plt.figure()
#plt.loglog(freq[positive_frequency_mask], np.abs(fft_TD[positive_frequency_mask]) ** 2)
plt.loglog(
    freq[positive_frequency_mask], get_sensitivity(freq[positive_frequency_mask])
)
plt.ylabel(r"$| {\rm DFT} [h_{+}]|^2$")
plt.xlabel(r"$f$ [Hz]")
plt.xlim(1e-4, 1e-1)
plt.show()
'''
# you can specify the frequencies or obtain them directly from the waveform
fd_kwargs = waveform_kwargs.copy()
fd_kwargs["f_arr"] = freq           # frequencies at which to output the waveform (optional)
fd_kwargs["mask_positive"] = True   # only output FD waveform at positive frequencies

# generate the FD signal directly, and time how long it takes
start = time.time()
hf = few_gen(*emri_injection_params, **fd_kwargs)
end = time.time()
print("Time taken to generate the FD signal: ", end - start, "seconds")

# to get the frequencies:
freq_fd = few_gen.waveform_generator.create_waveform.frequency

# calculate the mismatch between the FFT'd TD waveform and the direct FD waveform:
psd = get_sensitivity(freq[positive_frequency_mask]) / np.diff(freq)[0]
td_td = inner_product(
    fft_TD[positive_frequency_mask], fft_TD[positive_frequency_mask], psd
)
fd_fd = inner_product(hf[0], hf[0], psd)
Mism = np.abs(
    1
    - inner_product(fft_TD[positive_frequency_mask], hf[0], psd)
    / np.sqrt(td_td * fd_fd)
)
print("mismatch", Mism)

# SNR
print("TD SNR", np.sqrt(td_td))
print("FD SNR", np.sqrt(fd_fd))

# FD plot
plt.figure()
plt.loglog(
    freq[positive_frequency_mask],
    np.abs(fft_TD[positive_frequency_mask]) ** 2,
    label="DFT of TD waveform",
)
plt.loglog(freq[positive_frequency_mask], np.abs(hf[0]) ** 2, "--", label="FD waveform")
plt.loglog(
    freq[positive_frequency_mask],
    get_sensitivity(freq[positive_frequency_mask]),
    "k:",
    label="LISA sensitivity",
)
plt.ylabel(r"$| \tilde{h}_{+} (f)|^2$", fontsize=16)
plt.grid()
plt.xlabel(r"$f$ [Hz]", fontsize=16)
plt.legend()
plt.ylim([0.5e-41, 1e-32])
plt.xlim([1e-4, 1e-1])
plt.show()
# plt.savefig('figures/FD_TD_frequency.pdf', bbox_inches='tight')

# regenerate the FD waveform, outputting the negative frequencies too this time
fd_kwargs_temp = waveform_kwargs.copy()
fd_kwargs_temp["f_arr"] = freq
fd_kwargs_temp["mask_positive"] = False # do not mask the positive frequencies

hf_temp = few_gen(*emri_injection_params, **fd_kwargs_temp)

# check the consistency with the previous masked waveform
assert np.sum(hf_temp[0][positive_frequency_mask] - hf[0]) == 0.0

# transform FD waveform to TD
hf_to_ifft = np.append(
    hf_temp[0][positive_frequency_mask], hf_temp[0][~positive_frequency_mask]
)

# Plot the waveforms in the time domain
plt.figure()
time_array = np.arange(0, len(data_channels_td[0])) * dt
plt.plot(time_array, data_channels_td[0].real, label="TD waveform")
ifft_fd = np.fft.ifft(hf_to_ifft / dt)
plt.plot(time_array, ifft_fd.real, "--", label="Inverse DFT (FD waveform)")
plt.ylabel(r"$h_{+}(t)$")
plt.xlabel(r"$t$ [s]")

t0 = time_array[-1] * 0.7
space_t = 10e3
plt.xlim([t0, t0 + space_t / 2])
plt.ylim([-4e-22, 6e-22])
plt.legend(loc="upper center")
plt.show()

# construct downsampled frequency array
df = np.diff(freq)[0]
fmin, fmax = df, freq.max()
fmin, fmax = 1e-3, 5e-2
num_pts = 500
p_freq = np.append(0.0, np.linspace(fmin, fmax, num_pts))
freq_temp = np.hstack((-p_freq[::-1][:-1], p_freq))

# populate kwarg dictionary
fd_kwargs_ds = waveform_kwargs.copy()
fd_kwargs_ds["f_arr"] = freq_temp
fd_kwargs_ds["mask_positive"] = False

# get FD waveform with downsampled frequencies
hf_ds = few_gen(*emri_injection_params, **fd_kwargs_ds)
hf = few_gen(*emri_injection_params, **fd_kwargs)

# time the generation of the FD signal
start = time.time()
hf_ds = few_gen(*emri_injection_params, **fd_kwargs_ds)
end = time.time()
print("Time taken to generate the FD signal: ", end - start, "seconds")

# to get the frequencies:
freq_fd = few_gen.waveform_generator.create_waveform.frequency

# freq_temp = freq_temp[freq_temp>=0.0]
print("freq_fd", freq_fd.shape, "h shape", hf[0].shape)

# FD plot
plt.figure()
plt.loglog(freq[positive_frequency_mask], np.abs(hf[0])**2, "-", label="FD waveform")
plt.loglog(freq_fd, np.abs(hf_ds[0])**2, "--", label="FD waveform Downsample")
plt.plot(freq_temp, get_sensitivity(freq_temp), "k:", label="LISA sensitivity")
plt.ylabel(r"$| \tilde{h}_{+} (f)|^2$", fontsize=16)
plt.grid()
plt.xlabel(r"$f$ [Hz]", fontsize=16)
plt.legend(loc="upper left")
plt.ylim([0.5e-41, 1e-32])
plt.xlim([1e-4, 5e-2])
plt.show()
# plt.savefig('figures/FD_TD_frequency.pdf', bbox_inches='tight')

from scipy.signal.windows import tukey

fd_kwargs_nomask = fd_kwargs.copy()
del fd_kwargs_nomask["mask_positive"]

# no windowing
window = np.ones(len(data_channels_td[0]))
fft_td_gen = GetFDWaveformFromTD(td_gen, positive_frequency_mask, dt, window=window) # generate an FD waveform by FFT'ing a TD waveform
fd_gen = GetFDWaveformFromFD(few_gen, positive_frequency_mask, dt, window=window) # generate an FD waveform directly using an FD generator

np.all(fd_gen(*emri_injection_params, **fd_kwargs_nomask)[0] == hf[0])

hf = fd_gen(*emri_injection_params, **fd_kwargs_nomask)
fft_TD = fft_td_gen(*emri_injection_params, **fd_kwargs_nomask)

# calculate SNRs and mismatch
psd = get_sensitivity(freq[positive_frequency_mask]) / np.diff(freq)[0]
td_td = inner_product(fft_TD[0], fft_TD[0], psd)
fd_fd = inner_product(hf[0], hf[0], psd)
Mism = np.abs(1 - inner_product(fft_TD[0], hf[0], psd) / np.sqrt(td_td * fd_fd))

print(" ***** No window ***** ")
print("mismatch", Mism)
print("TD SNR", np.sqrt(td_td))
print("FD SNR", np.sqrt(fd_fd))


# add windowing
window = np.asarray(tukey(len(data_channels_td[0]), 0.01))
fft_td_gen = GetFDWaveformFromTD(td_gen, positive_frequency_mask, dt, window=window)
fd_gen = GetFDWaveformFromFD(few_gen, positive_frequency_mask, dt, window=window)

hf_win = fd_gen(*emri_injection_params, **fd_kwargs_nomask)
fft_TD_win = fft_td_gen(*emri_injection_params, **fd_kwargs_nomask)

# calculate SNRs and mismatch
psd = get_sensitivity(freq[positive_frequency_mask]) / np.diff(freq)[0]
td_td = inner_product(fft_TD_win[0], fft_TD_win[0], psd)
fd_fd = inner_product(hf_win[0], hf_win[0], psd)
Mism = np.abs(1 - inner_product(fft_TD_win[0], hf_win[0], psd) / np.sqrt(td_td * fd_fd))

print("\n\n ***** With window ***** ")
print("mismatch", Mism)
print("TD SNR", np.sqrt(td_td))
print("FD SNR", np.sqrt(fd_fd))

# FD plot
plt.figure(figsize=(6, 9))
plt.subplot(2,1,1)
plt.title("Unwindowed", fontsize=16)
plt.loglog(
    freq[positive_frequency_mask], np.abs(fft_TD[0]) ** 2, label="DFT of TD waveform"
)
plt.loglog(freq[positive_frequency_mask], np.abs(hf[0]) ** 2, "--", label="FD waveform")
plt.loglog(
    freq[positive_frequency_mask],
    get_sensitivity(freq[positive_frequency_mask]),
    "k:",
    label="LISA sensitivity",
)
plt.ylabel(r"$| \tilde{h}_{+}(f)|^2$", fontsize=16)
plt.xlabel(r"$f$ [Hz]", fontsize=16)
plt.legend(loc='upper left')
plt.grid()
plt.ylim([0.5e-41, 1e-32])
plt.xlim([1e-4, 1e-1])
#plt.show()
# plt.savefig('figures/FD_TD_frequency_windowed.pdf', bbox_inches='tight')

plt.subplot(2,1,2)
plt.title("Windowed", fontsize=16)
plt.loglog(
    freq[positive_frequency_mask], np.abs(fft_TD_win[0]) ** 2, label="DFT of TD waveform"
)
plt.loglog(freq[positive_frequency_mask], np.abs(hf_win[0]) ** 2, "--", label="FD waveform")
plt.loglog(
    freq[positive_frequency_mask],
    get_sensitivity(freq[positive_frequency_mask]),
    "k:",
    label="LISA sensitivity",
)
plt.ylabel(r"$| \tilde{h}_{+}(f)|^2$", fontsize=16)
plt.xlabel(r"$f$ [Hz]", fontsize=16)
plt.legend(loc='upper left')
plt.grid()
plt.ylim([0.5e-41, 1e-32])
plt.xlim([1e-4, 1e-1])
plt.gcf().tight_layout()


def calculate_snr_mismatch(
    mode,
    emri_injection_params,
    waveform_kwargs,
    fd_kwargs,
    freq,
    positive_frequency_mask,
    dt,
):
    # Update fd_kwargs and td_kwargs with the current mode
    fd_kwargs = fd_kwargs.copy()
    fd_kwargs.pop("mode_selection_threshold")
    fd_kwargs["mode_selection"] = [mode]
    hf_mode = few_gen(*emri_injection_params, **fd_kwargs)

    td_kwargs2 = waveform_kwargs.copy()
    td_kwargs2.pop("mode_selection_threshold")
    td_kwargs2["mode_selection"] = [mode]
    data_channels_td_mode = td_gen(*emri_injection_params, **td_kwargs2)

    # Take the FFT of the plus polarization and shift it
    fft_TD_mode = np.fft.fftshift(np.fft.fft(data_channels_td_mode[0])) * dt

    # Calculate PSD
    psd = get_sensitivity(freq[positive_frequency_mask]) / np.diff(freq)[0]

    # Calculate inner products
    td_td = inner_product(
        fft_TD_mode[positive_frequency_mask], fft_TD_mode[positive_frequency_mask], psd
    )
    fd_fd = inner_product(hf_mode[0], hf_mode[0], psd)
    Mism = np.abs(
        1
        - inner_product(fft_TD_mode[positive_frequency_mask], hf_mode[0], psd)
        / np.sqrt(td_td * fd_fd)
    )

    # calculated frequency
    OmegaPhi, OmegaTheta, OmegaR = get_fundamental_frequencies(
        emri_injection_params[2],
        emri_injection_params[3],
        emri_injection_params[4],
        emri_injection_params[5],
    )
    harmonic_frequency = (OmegaPhi * mode[1] + OmegaTheta * mode[2] + OmegaR * mode[3]) / (
        emri_injection_params[0] * MTSUN_SI * 2 * np.pi
    )
    return np.sqrt(td_td), Mism, harmonic_frequency


# Initialize data storage
data_out = []

# mode vector
eccentricity_vector = [0.1, 0.3, 0.7]
max_n_vector = [10, 18, 26]
spin_vector = [0.0, 0.9]

for a in spin_vector:
    for l_set, m_set in zip([2], [2]):
        temp = emri_injection_params.copy()
        for e_temp, max_n in zip(eccentricity_vector, max_n_vector):
            modes = [(l_set, m_set, 0, ii) for ii in range(-3, max_n)]
            p_temp = get_p_at_t(
                traj_module,
                Tobs * 0.99,
                [m1, m2, a, e_temp, 1.0],
                index_of_p=3,
                index_of_a=2,
                index_of_e=4,
                index_of_x=5,
                traj_kwargs={},
                xtol=2e-12,
                rtol=8.881784197001252e-16,
                bounds=None,
            )
            temp[3] = p_temp
            temp[4] = e_temp
            temp[2] = a
            out = np.asarray(
                [
                    calculate_snr_mismatch(
                        mode,
                        temp,
                        waveform_kwargs,
                        fd_kwargs,
                        freq,
                        positive_frequency_mask,
                        dt,
                    )
                    for mode in modes
                ]
            )
            snr, Mism, harmonic_frequency = out.T
            data_out.append((harmonic_frequency, snr, l_set, m_set, e_temp, a))
# Plot the data
colors = {0.1: "royalblue", 0.3: "seagreen", 0.5: "crimson", 0.7: "darkviolet"}

plt.figure(figsize=(8, 5))
for harmonic_frequency, snr, l_set, m_set, e_temp, a in data_out:
    color = colors[e_temp]
    if a == 0.9:
        plt.plot(
            harmonic_frequency,
            20.0 * snr / np.sum(snr**2) ** 0.5,
            ":P",
            label=f"e = {e_temp}, a = {a}",
            color=color,
        )
        # plt.text(harmonic_frequency[-1], 20.0 * snr[-1]/np.sum(snr**2)**0.5, f"({l_set},{m_set})", fontsize=8)
    if a == 0.0:
        plt.plot(
            harmonic_frequency,
            20.0 * snr / np.sum(snr**2) ** 0.5,
            "--o",
            label=f"e = {e_temp}, a = {a}",
            color=color,
        )
        # for ii in range(len(harmonic_frequency)):
        #     plt.text(harmonic_frequency[ii], 20.0 * snr[ii]/np.sum(snr**2)**0.5, f"n={ii-3}", fontsize=8)

plt.xlabel("Initial Harmonic Frequency [Hz]")
plt.ylabel("20 x SNR / Total SNR")
plt.title(
    f"SNR per each harmonic (m,n) = ({2},n) for m1={m1:.2e}, m2={m2:.2e} and plunge in {Tobs} years"
)
plt.xlim(0, 0.012)
plt.grid()
plt.legend()
plt.show()


def calculate_snr_mismatch(
    mode,
    emri_injection_params,
    waveform_kwargs,
    fd_kwargs,
    freq,
    positive_frequency_mask,
    dt,
):
    # Update fd_kwargs and td_kwargs with the current mode
    fd_kwargs = fd_kwargs.copy()
    fd_kwargs.pop("mode_selection_threshold")
    fd_kwargs["mode_selection"] = [mode]
    hf_mode = few_gen(*emri_injection_params, **fd_kwargs)

    td_kwargs2 = waveform_kwargs.copy()
    td_kwargs2.pop("mode_selection_threshold")
    td_kwargs2["mode_selection"] = [mode]
    data_channels_td_mode = td_gen(*emri_injection_params, **td_kwargs2)

    # Take the FFT of the plus polarization and shift it
    fft_TD_mode = np.fft.fftshift(np.fft.fft(data_channels_td_mode[0])) * dt

    # Calculate PSD
    psd = get_sensitivity(freq[positive_frequency_mask]) / np.diff(freq)[0]

    # Calculate inner products
    td_td = inner_product(
        fft_TD_mode[positive_frequency_mask], fft_TD_mode[positive_frequency_mask], psd
    )
    fd_fd = inner_product(hf_mode[0], hf_mode[0], psd)
    Mism = np.abs(
        1
        - inner_product(fft_TD_mode[positive_frequency_mask], hf_mode[0], psd)
        / np.sqrt(td_td * fd_fd)
    )

    # calculated frequency
    OmegaPhi, OmegaTheta, OmegaR = get_fundamental_frequencies(
        emri_injection_params[2],
        emri_injection_params[3],
        emri_injection_params[4],
        emri_injection_params[5],
    )
    harmonic_frequency = (OmegaPhi * mode[1] + OmegaTheta * mode[2] + OmegaR * mode[3]) / (
        emri_injection_params[0] * MTSUN_SI * 2 * np.pi
    )
    return np.sqrt(td_td), Mism, harmonic_frequency


# Initialize data storage
data_out = []

# mode vector
eccentricity_vector = [0.1, 0.3, 0.7]
max_n_vector = [10, 18, 26]
spin_vector = [0.0, 0.9]

for a in spin_vector:
    for l_set, m_set in zip([2], [2]):
        temp = emri_injection_params.copy()
        for e_temp, max_n in zip(eccentricity_vector, max_n_vector):
            modes = [(l_set, m_set, 0, ii) for ii in range(-3, max_n)]
            p_temp = get_p_at_t(
                traj_module,
                Tobs * 0.99,
                [m1, m2, a, e_temp, 1.0],
                index_of_p=3,
                index_of_a=2,
                index_of_e=4,
                index_of_x=5,
                traj_kwargs={},
                xtol=2e-12,
                rtol=8.881784197001252e-16,
                bounds=None,
            )
            temp[3] = p_temp
            temp[4] = e_temp
            temp[2] = a
            out = np.asarray(
                [
                    calculate_snr_mismatch(
                        mode,
                        temp,
                        waveform_kwargs,
                        fd_kwargs,
                        freq,
                        positive_frequency_mask,
                        dt,
                    )
                    for mode in modes
                ]
            )
            snr, Mism, harmonic_frequency = out.T
            data_out.append((harmonic_frequency, snr, l_set, m_set, e_temp, a))
# Plot the data
colors = {0.1: "royalblue", 0.3: "seagreen", 0.5: "crimson", 0.7: "darkviolet"}

plt.figure(figsize=(8, 5))
for harmonic_frequency, snr, l_set, m_set, e_temp, a in data_out:
    color = colors[e_temp]
    if a == 0.9:
        plt.plot(
            harmonic_frequency,
            20.0 * snr / np.sum(snr**2) ** 0.5,
            ":P",
            label=f"e = {e_temp}, a = {a}",
            color=color,
        )
        # plt.text(harmonic_frequency[-1], 20.0 * snr[-1]/np.sum(snr**2)**0.5, f"({l_set},{m_set})", fontsize=8)
    if a == 0.0:
        plt.plot(
            harmonic_frequency,
            20.0 * snr / np.sum(snr**2) ** 0.5,
            "--o",
            label=f"e = {e_temp}, a = {a}",
            color=color,
        )
        # for ii in range(len(harmonic_frequency)):
        #     plt.text(harmonic_frequency[ii], 20.0 * snr[ii]/np.sum(snr**2)**0.5, f"n={ii-3}", fontsize=8)

plt.xlabel("Initial Harmonic Frequency [Hz]")
plt.ylabel("20 x SNR / Total SNR")
plt.title(
    f"SNR per each harmonic (m,n) = ({2},n) for m1={m1:.2e}, m2={m2:.2e} and plunge in {Tobs} years"
)
plt.xlim(0, 0.012)
plt.grid()
plt.legend()
plt.show()
'''