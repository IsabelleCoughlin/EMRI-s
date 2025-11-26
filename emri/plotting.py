from .constants import *
import matplotlib.pyplot as plt
from .peters import *


def plot_SNR(a_arr, e_arr, t_arr, N_harm_show=10, N_harm_sum=1000):

    ttp, SNR_tot, SNR_harm = snr_1yr_vs_time(a_arr, e_arr, t_arr, N_harm_show, N_harm_sum)

    plt.figure(figsize=(8,6))

    # Plot harmonic curves
    colors = plt.cm.tab10(np.linspace(0,1,N_harm_show))
    for n in range(N_harm_show):
        plt.loglog(ttp, SNR_harm[n], '--', lw=1, color=colors[n])

    # Plot total
    plt.loglog(ttp, SNR_tot, 'k', lw=2, label=f'Total SNR (sum to n={N_harm_sum})')

    plt.xlabel('Time to plunge (yr)')
    plt.ylabel('SNR for 1-yr observation')
    plt.title('SNR for 1-yr observation vs Time to Plunge')
    plt.grid(True, which='both', ls=':')
    plt.legend(loc='upper left', fontsize=8)
    plt.xlim(1e7, 1) 
    plt.ylim(1, 1e4)
    plt.show()
    
def plot_characteristic_strain_PM(t_to_plunge_yr_interpolated, e_interpolated, a_interpolated):  
    
    time_arrays = np.array([1e6, 1e5, 1e4, 1000, 1]) # Values to plot the characteristic strain at (years)
    idxs = [np.argmin(np.abs(t_to_plunge_yr_interpolated - T)) for T in time_arrays] # Grab the closest point at that time

    colors = ['blue', 'green', 'orange', 'red', 'purple']

    for j, (T, idx, color) in enumerate(zip(time_arrays, idxs, colors)):
        f = np.linspace(10**(-6),1, 500)
        n_har = 1000
        e = e_interpolated[idx]
        a = a_interpolated[idx]
        
        n_vals = np.arange(1, n_har+1)
        f_n = n_vals * f_orb_PM(a)
        h_n_vals = np.array([h_n_PM(n, e, a) for n in range(1, n_har+1)])
        dfdt2_vals = np.array([abs(dfdt_n_PM(n, a, e)) for n in range(1, n_har+1)])
        h_c_vals = h_n_vals * np.sqrt(2 * f_n**2 / dfdt2_vals)
    
        plt.loglog(f_n, h_c_vals, color=color)
        plt.plot(f_n[:n_har], h_c_vals[:n_har], 'o', color=color, markersize=3)
        
        # Annotate with Rp/Rs and e
        R_s = 2 * G * M / c**2
        Rp_Rs = a * (1 - e) / R_s
        plt.text(f_n[0], h_c_vals[0]*1.2, f"{int(T):.0e} yr: e={e:.3f}, Rp/Rs={Rp_Rs:.1f}", color=color, fontsize=7)
        print(T)
        print(e)
        print(Rp_Rs)

    plt.loglog(f, h_det(f), lw=2, color = 'green')
    plt.xlabel('Frequencies')
    plt.ylabel('Characteristic Strain')
    plt.ylim(10**(-22), 10**-10)
    plt.xlim(10**-6, 1)
    plt.grid(True)
    plt.show()
