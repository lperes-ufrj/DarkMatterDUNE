import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from astropy import coordinates as coords
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, AltAz, Galactic
import astropy.units as u
import dunestyle.matplotlib as dunestyle


# Constants
r_s = 20.0  # kpc #20.0
rho_0 = 0.184  # GeV/cm^3
R_sun = 8.122  # kpc


# Change the profile or parameters here if desired
def rho_NFW(r, r_s=20.0, rho0=0.184, R_sun=8.122):
    r  = np.maximum(r, 1e-6)
    x  = r / r_s
    xs = R_sun / r_s
    return rho0 * (xs/x) * ((1+xs)**2 / (1+x)**2)


def r_galcen(l_deg, b_deg, s):
    l_rad = np.radians(l_deg)
    b_rad = np.radians(b_deg)
    return np.sqrt(R_sun**2 + s**2 - 2 * R_sun * s * np.cos(b_rad) * np.cos(l_rad))


def los_integral_vec(l_deg, b_deg, s_max=300.0, steps=1000):
    s = np.linspace(0.01, s_max, steps)                     # (S,)
    l = np.radians(l_deg)[..., None]                        # (...,1)
    b = np.radians(b_deg)[..., None]                        # (...,1)
    r = np.sqrt(R_sun**2 + s**2 - 2*R_sun*s*np.cos(b)*np.cos(l))  # (...,S)
    rho = rho_NFW(r)
    return np.trapz(rho, s, axis=-1)                        # (...)

# Grid
n_pix = 100
l_vals = np.linspace(-180, 180, n_pix)
b_vals = np.linspace(-90, 90, n_pix // 2)
L, B = np.meshgrid(l_vals, b_vals)

# Compute LOS integral
print("Computing LOS integrals...")
rho_proj = np.zeros_like(L)
Alt = np.zeros_like(L)
Azimuth = np.zeros_like(L)
for i in range(B.shape[0]):
    for j in range(L.shape[1]):
        rho_proj[i, j] = los_integral_vec(L[i, j], B[i, j])

# Convert to radians
L_rad = np.radians(L)
B_rad = np.radians(B)
L_rad = np.where(L_rad > np.pi, L_rad - 2*np.pi, L_rad)

# --- DUNE location
dune_location = EarthLocation(lat=44.352*u.deg, lon=-103.751*u.deg, height=100*u.m)

# --- Time sampling
N_skies = 3*360 # Number of sampled skies in one year - 3 per day

base = Time('2024-06-01T00:00:00', scale='utc')
times = base + np.linspace(0, 365, N_skies) * u.day


# --- Alt-Az binning
n_alt_bins = 180
n_az_bins = 360
alt_edges = np.linspace(-90, 90, n_alt_bins + 1)
az_edges = np.linspace(0, 360, n_az_bins + 1)

# Centers
alt_centers = 0.5 * (alt_edges[:-1] + alt_edges[1:])
az_centers = 0.5 * (az_edges[:-1] + az_edges[1:])
ALT, AZ = np.meshgrid(alt_centers, az_centers)

ALT_flat = ALT.flatten()
AZ_flat = AZ.flatten()

# --- Initialize arrays
intensity_time_all = np.zeros((len(times), len(ALT_flat)))  # (n_times, n_bins)

gc_pos = []

# --- Calculate
for idx_time, t in enumerate(times):
    # Galactic center coordinates
    gc_gal = SkyCoord(l=0*u.deg, b=0*u.deg, frame='galactic')
    gc_altaz = gc_gal.transform_to(AltAz(obstime=t, location=dune_location))
    
    gc_pos.append((gc_altaz.az.rad, gc_altaz.alt.rad))
    
    # Compute Galactic coordinates for all AltAz at this time
    altaz = AltAz(alt=ALT_flat*u.deg, az=AZ_flat*u.deg, location=dune_location, obstime=t)
    gal = altaz.transform_to(Galactic())
    # Compute LOS integral for all these coordinates    
    intensity_time_all[idx_time] = los_integral_vec(gal.l.deg, gal.b.deg).ravel()


gc_pos = np.array(gc_pos)

# --- Now: for each bin, summarize over time
#intensity_median = np.median(intensity_time_all, axis=0)
intensity_sum = np.sum(intensity_time_all, axis=0)

intensity_sum_median = intensity_sum / N_skies

# --- Normalize
intensity_sum_median_norm = intensity_sum_median / np.max(intensity_sum_median)
intensity_sum_norm = intensity_sum / np.max(intensity_sum)

# Save arrays to .npy files
np.save("npy_arrays/az.npy", AZ_flat)
np.save("npy_arrays/alt.npy", ALT_flat)
np.save("npy_arrays/intensity_sum_norm.npy", intensity_sum_norm)  
np.save("npy_arrays/gc_pos.npy", gc_pos)



# Plot sum map
fig, ax = plt.subplots(dpi=100, figsize=(10, 6))
ax.set_xticks([0.0,1.570,3.14159,4.7123,6.283])
ax.set_xticklabels([0,r'$\pi / 2$', r'$\pi$',r'$3\pi/2$', r'$2\pi$'], fontsize = 15)
ax.set_yticks([-1.570,-0.78539,0.0,0.78539,1.570])
ax.set_yticklabels([ r'$-\pi / 2$', r'$-\pi / 4$',0,r'$\pi / 4$',r'$\pi / 2$'], fontsize = 15)
plt.hist2d(np.radians(AZ_flat), np.radians(ALT_flat), weights=intensity_sum_norm,
           bins=[np.radians(az_edges), np.radians(alt_edges)])
gc_scatter = plt.scatter(gc_pos[:,0],gc_pos[:,1], marker='.', color='red', label = r'$\bf{GC\;Position}$', s =0.1)
cbar = plt.colorbar()
cbar.set_label(r'$ \left<\int \rho_\mathrm{NFW}(r)\,ds\right> /\text{Max}\left(\left<\int \rho_\mathrm{NFW}(r)\,ds\right>\right)$ ', fontsize=16)
plt.xlabel('Azimuth (rad)')
plt.ylabel('Altitude (rad)')

lgnd = plt.legend(handles=[gc_scatter],fontsize=14, frameon=True, framealpha=0.05)

# Fix marker size using the handles returned by legend
for handle in lgnd.legend_handles:
    handle.set_sizes([30])  # marker size in points²
plt.savefig('plots/DUNE_AltAz_Density_LOS_Map.png', dpi=300)
plt.tight_layout()
plt.show()