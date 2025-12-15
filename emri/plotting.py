from .constants import *
import matplotlib.pyplot as plt
from .peters import *
from .harmonics import *
from emri.LISA_sensitivity import *
from .SNR import *
from .PN import *
from .integrator import *


def plot_SNR(a_arr, e_arr, t_arr,p,  N_harm_show=10, N_harm_sum=1000, step = 1000):

    length_array = len(t_arr)
    ttp, SNR_tot, SNR_harm = snr_1yr_vs_time(a_arr[:length_array:step], e_arr[:length_array:step], t_arr[:length_array:step],p, N_harm_show, N_harm_sum)

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
    
def plot_characteristic_strain_PM(t_to_plunge_yr_interpolated, e_interpolated, a_interpolated, p):  
    M = p.M_si
    mu = p.mu_si
    
    time_arrays = np.array([1e6, 1e5, 1e4, 1000, 1]) # Values to plot the characteristic strain at (years)
    idxs = [np.argmin(np.abs(t_to_plunge_yr_interpolated - T)) for T in time_arrays] # Grab the closest point at that time

    colors = ['blue', 'green', 'orange', 'red', 'purple']

    for j, (T, idx, color) in enumerate(zip(time_arrays, idxs, colors)):
        f = np.linspace(10**(-6),1, 500)
        n_har = 1000
        e = e_interpolated[idx]
        a = a_interpolated[idx]
        
        n_vals = np.arange(1, n_har+1)
        f_n = n_vals * f_orb_PM(a, M, mu)
        h_n_vals = np.array([h_n_PM(n, a, e, M, mu) for n in range(1, n_har+1)])
        dfdt2_vals = np.array([abs(dfdt_n_PM(n, a, e, M, mu)) for n in range(1, n_har+1)])
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

def plot_e(t_to_plunge_yr_interpolated, e_interpolated, title="Peters and Matthew"):

    # Plot results
    plt.figure(figsize=(8,5))
    plt.plot(t_to_plunge_yr_interpolated, e_interpolated, label='Eccentricity e(t)')
    plt.xlabel('Time evolved (yrs)')
    plt.ylabel('Eccentricity (unitless)')
    plt.title(f"Interpolated 1-yr dt Eccentricity using {title}")
    plt.grid(True)
    plt.legend()
    plt.gca().invert_xaxis()
    plt.show()

def plot_a(t_to_plunge_yr_interpolated, a_interpolated, title="Peters and Matthew"):

    # Plot results
    plt.figure(figsize=(8,5))
    plt.plot(t_to_plunge_yr_interpolated, a_interpolated, label='Semi-Major Axis a(t)')
    plt.xlabel('Time evolved (yrs)')
    plt.ylabel('Semi-Major Axis (meters)')
    plt.title(f"Interpolated 1-yr dt Semi-Major Axis using {title}")
    plt.grid(True)
    plt.legend()
    plt.gca().invert_xaxis()
    plt.show()
    
def plot_characteristic_strain_PN(t_to_plunge_yr_interpolated, e_interpolated, nu_interpolated, phi_interpolated, gamma_interpolated, p):
    time_arrays = np.array([1e6, 1e5, 1e4, 1000, 1]) # Values to plot the characteristic strain at (years)
    idxs = [np.argmin(np.abs(t_to_plunge_yr_interpolated - T)) for T in time_arrays] # Grab the closest point at that time

    colors = ['blue', 'green', 'orange', 'red', 'purple']

    for j, (T, idx, color) in enumerate(zip(time_arrays, idxs, colors)):
        f = np.linspace(10**(-6),1, 500)
        n_har = 1000
        e = e_interpolated[idx]
        nu = nu_interpolated[idx]
        phi = phi_interpolated[idx]
        gamma = gamma_interpolated[idx]
        
        #f_orbital = f_orb(n,"" phi, nu, gamma, e)  # Hz
        n_vals = np.arange(1, n_har+1) # Array of integers 1 to 10
        f_n = np.array([f_orb_PN(n, phi, nu, gamma, e, p)for n in range(1, n_har+1)])
        h_n_vals = np.array([h_n_PN(n, e, nu, phi, gamma, p) for n in range(1, n_har+1)])
        dfdt2_vals = np.array([abs(f_dot_PN(n, e, nu, phi, gamma, p)) for n in range(1, n_har+1)])
        h_c_vals = h_n_vals * np.sqrt(2 * f_n**2 / dfdt2_vals)
    
        plt.loglog(f_n, h_c_vals, color=color)
        plt.plot(f_n[:n_har], h_c_vals[:n_har], 'o', color=color, markersize=3)
        
        # Annotate with Rp/Rs and e
        a = p.M_seconds**(1/3) / ( (2*np.pi*nu)**(2/3) )

        
        R_s = 2 * p.M_seconds                   # Schwarzschild radius in seconds
        Rp   = a * (1 - e)                # pericenter distance in seconds
        Rp_Rs = Rp / R_s                  # dimensionless
        plt.text(f_n[0], h_c_vals[0]*1.2, f"{int(T):.0e} yr: e={e:.3f}, Rp/Rs={Rp_Rs:.1f}", color=color, fontsize=7)

    plt.loglog(f, h_det(f), lw=2, color = 'green')
    plt.xlabel('Frequencies')
    plt.ylabel('Characteristic Strain')
    plt.ylim(10**(-23), 10**-10)
    plt.xlim(10**-6, 1)
    plt.grid(True)
    plt.show()

def plot_characteristic_strain_PN_by_n(
    t_to_plunge_yr_interpolated,
    e_interpolated,
    nu_interpolated,
    phi_interpolated,
    gamma_interpolated,
    p,
    n_list=(1, 3, 9),                  # <-- choose harmonics here
    time_arrays=(10, 5, 2, 1),          # <-- years-to-plunge markers
    t_window_yr=None,                   # e.g. (0, 10) to show last 10 yrs only
    show_markers=True,
):
    # ---- choose which portion of the trajectory to use ----
    ttp = np.asarray(t_to_plunge_yr_interpolated)
    mask = np.isfinite(ttp)

    if t_window_yr is not None:
        tmin, tmax = t_window_yr
        mask &= (ttp >= tmin) & (ttp <= tmax)

    e     = np.asarray(e_interpolated)[mask]
    nu    = np.asarray(nu_interpolated)[mask]
    phi   = np.asarray(phi_interpolated)[mask]
    gamma = np.asarray(gamma_interpolated)[mask]
    ttp   = ttp[mask]

    # indices for marker times (computed in the masked arrays)
    idxs = [np.argmin(np.abs(ttp - T)) for T in time_arrays]

    # ---- plot ----
    plt.figure(figsize=(8, 6))

    # plot each requested harmonic as a parametric curve in time
    for n in n_list:
        # compute arrays along trajectory
        f_arr = np.array([f_orb_PN(n, phi[i], nu[i], gamma[i], e[i], p) for i in range(len(nu))])
        f_dot_arr = np.array([abs(f_dot_PN(n, e[i], nu[i], phi[i], gamma[i], p)) for i in range(len(nu))])
        h_arr = np.array([h_n_PN(n, e[i], nu[i], phi[i], gamma[i], p) for i in range(len(nu))])

        # characteristic strain along trajectory
        hc_arr = np.where(
            (f_dot_arr > 0) & np.isfinite(f_dot_arr) & np.isfinite(f_arr) & (f_arr > 0),
            #h_arr * np.sqrt(2 * f_arr**2 / f_dot_arr),
            h_c_n_PN(n, e, nu, phi, gamma, p),
            np.nan
        )

        valid = np.isfinite(hc_arr) & (f_arr > 0)
        plt.loglog(f_arr[valid], hc_arr[valid], lw=1.6, label=f"n={n}")

        # markers at specific years-to-plunge
        if show_markers:
            for T, idx in zip(time_arrays, idxs):
                if valid[idx]:
                    plt.plot(f_arr[idx], hc_arr[idx], 'o', color = 'black', markersize = 3, zorder = 6)

    # sensitivity curve overlay
    f = np.logspace(-4, -1, 500)
    plt.loglog(f, h_det(f), lw=2, color='green', label="Detector")
    #plt.loglog(f, analytical_sensitivity(f), lw=2, color='green', label="Detector")
    #plt.xlim(1e-4, 1e-1) 
    #plt.ylim(1e-21, 1e-19)
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Characteristic Strain")
    plt.grid(True, which="both", alpha=0.25)
    plt.title(r"$\frac{10}{10^6}$ Mass ratio at D = 1 Gpc with $e_{LSO} = 0.3$")
    plt.legend()
    plt.show()
    
def plot_analytical():
    f = np.logspace(-5, -1, 1000)   # Hz
    S = analytical_sensitivity(f)

    plt.figure(figsize=(7,5))
    plt.loglog(f, S, lw=2)
    plt.xlabel("Frequency [Hz]")
    plt.ylabel(r"$S_n(f)$")
    plt.grid(True, which="both", alpha=0.3)
    plt.show()

    
def plot_SNR_PN(phi_arr, nu_arr, gamma_arr, e_arr, t_arr, p, N_harm_show=10, N_harm_sum=1000, step = 1000):
    
    length_array = len(t_arr)

    ttp, SNR_tot, SNR_harm = snr_1yr_vs_time_PN(phi_arr[:length_array:step], nu_arr[:length_array:step], gamma_arr[:length_array:step], e_arr[:length_array:step], t_arr[:length_array:step],p, N_harm_show, N_harm_sum)
    #ttp, SNR_tot= snr_total_1yr_vs_time(phi_arr, nu_arr, gamma_arr, e_arr, t_arr, N_harm_sum)

    plt.figure(figsize=(8,6))

    # Plot harmonic curves
    colors = plt.cm.tab10(np.linspace(0,1,N_harm_show))
    for n in range(N_harm_show):
        plt.loglog(ttp, SNR_harm[n], '--', lw=1, color=colors[n])

    # Plot total
    plt.loglog(ttp, SNR_tot, 'k', lw=2,label=f'Total SNR (sum to n={N_harm_sum})')

    plt.xlabel('Time to plunge (yr)')
    plt.ylabel('SNR for 1-yr observation')
    plt.title('SNR for 1-yr observation vs Time to Plunge - PN')
    plt.grid(True, which='both', ls=':')
    plt.legend(loc='upper left', fontsize=8)
    
    plt.xlim(1e7, 1) 
    plt.ylim(1, 1e4)
    plt.show()
    



def plot_nu_LSO(p):
    plt.figure(figsize=(7, 5))

    # LSO curve
    e_vals = np.linspace(0.0, 0.9, 400)
    plt.plot(e_vals, nu_LSO(e_vals, p), 'b--', label='LSO')

    ecc_list = [0.01, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]

    for e_LSO in ecc_list:
        nu0 = nu_LSO(e_LSO, p)
        y0 = [e_LSO, nu0]

        # integrate backwards in time from t=0 (LSO) to earlier times (negative)
        t_span = (0.0, -10.0 * yr)  # ~10 years backward
        sol = integrate_trajectory(deriv_PN_reduced, y0, t_span, p=p)#solve_ivp(deriv_PN, t_span, y0, rtol=1e-9, atol=1e-12, max_step=1e5)

        
        t_array = sol.t
        e_arr, nu_arr = sol.y

        # plot the ν(e) trajectory
        plt.plot(e_arr, nu_arr, color='magenta', lw=1.5)

        # mark 10, 5, 2, 1 years BEFORE plunge (negative times)
        for T in [10, 5, 2, 1]:
            t_target = -T * yr
            if not (np.min(t_array) <= t_target <= np.max(t_array)):
                continue
            idx = np.argmin(np.abs(t_array - t_target))
            plt.plot(e_arr[idx], nu_arr[idx], 'o', color='blue', ms=3)

    plt.xlabel("eccentricity")
    plt.ylabel("orbital frequency (Hz)")
    plt.ylim(5e-4, 2.5e-3)
    plt.xlim(0.0, 0.7)
    plt.grid(alpha=0.3)
    plt.title("ν–e evolution for 1 M⊙ into 10⁶ M⊙ MBH (non-spinning)")
    plt.legend()
    plt.show()