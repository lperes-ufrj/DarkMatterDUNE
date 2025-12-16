import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from astropy import coordinates as coords
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, AltAz, Galactic
import astropy.units as u
import dunestyle.matplotlib as dunestyle


# --- Alt-Az binning
n_alt_bins = 90
n_az_bins = 180
alt_edges = np.linspace(-90, 90, n_alt_bins + 1)
az_edges = np.linspace(0, 360, n_az_bins + 1)

# Load data from .npy files
AZ_flat = np.load('npy_arrays/az.npy')
ALT_flat = np.load('npy_arrays/alt.npy')
intensity_sum_norm = np.load('npy_arrays/intensity_sum_norm.npy')
gc_pos = np.load('npy_arrays/gc_pos.npy')

# Convert degrees to radians
alt_rad = np.radians(ALT_flat)
az_rad = np.radians(AZ_flat)

# Local frame: x (south), y (east), z (zenith)
x = -np.cos(alt_rad) * np.cos(az_rad)
y = np.cos(alt_rad) * np.sin(az_rad)
z = np.sin(alt_rad)
# Convert GC position to Cartesian coordinates


# Local frame: x (south), y (east), z (zenith)
x_gc = -np.cos(gc_pos[:,1]) * np.cos(gc_pos[:,0])
y_gc = np.cos(gc_pos[:,1]) * np.sin(gc_pos[:,0])
z_gc = np.sin(gc_pos[:,1])


theta_gc = np.arccos(z_gc)
phi_gc = np.arctan2(y_gc,x_gc)


# Convert to spherical coordinates
theta = np.arccos(z)              # angle from zenith [0, pi]
phi = np.arctan2(y, x)            # azimuth from south [-pi, pi]
phi = np.mod(phi, 2 * np.pi)      # shift to [0, 2pi]



# Define bin edges
n_theta_bins = 90
n_phi_bins = 180
theta_edges = np.linspace(0, np.pi, n_theta_bins + 1)
phi_edges = np.linspace(0, 2*np.pi, n_phi_bins + 1)


# Adjust theta and phi ranges to match desired display:
# theta in [0, pi], phi in [-pi, pi] for symmetry
phi_signed = phi.copy()
phi_signed[phi_signed > np.pi] -= 2 * np.pi  # convert to range [-π, π]

# Adjust theta and phi ranges to match desired display:
# theta in [0, pi], phi in [-pi, pi] for symmetry
#phi_signed_det = phi_det.copy()
#phi_signed_det[phi_signed_det > np.pi] -= 2 * np.pi  # convert to range [-π, π]


# Save arrays to .npy files
np.save("npy_arrays/theta_honda.npy", theta)
np.save("npy_arrays/phi_honda.npy", phi_signed)
np.save("npy_arrays/gc_theta_honda.npy", theta_gc)
np.save("npy_arrays/gc_phi_honda.npy", phi_gc)

fig, ax = plt.subplots(dpi=100, figsize=(10, 6))

# Use the previously computed theta and phi_signed, and normalized intensity
h = ax.hist2d(theta, phi_signed, weights=intensity_sum_norm,
              bins=[theta_edges, np.linspace(-np.pi, np.pi, n_phi_bins + 1)],
               cmap='viridis')

gc_scatter = plt.scatter(theta_gc,phi_gc, marker='.', color='red', label = r'$\bf{GC\;Position}$', s =0.1)

# Axis annotation
ax.text(0.1, 1.356, "x-axis points towards geographic south\n"
                    "y-axis points towards geographic east\n"
                    "z-axis points towards zenith", fontsize=15)

# Axis labels
ax.set_xlabel(r'$\theta$', fontsize=17)
ax.set_ylabel(r'$\phi$', fontsize=17)

# X-axis ticks and labels
ax.set_xticks([0.0, 0.78539, 1.570, 2.356, 3.14159])
ax.set_xticklabels([r'$0$', r'$\pi / 4$', r'$\pi / 2$', r'$3 \pi / 4$', r'$\pi$'], fontsize=15)

# Y-axis ticks and labels
ax.set_yticks([-3.14159, -2.356, -1.570, -0.78539, 0.0, 0.78539, 1.570, 2.356, 3.14159])
ax.set_yticklabels([r'$-\pi$', r'$-3\pi / 4$', r'$-\pi / 2$', r'$-\pi / 4$', r'$0$',
                    r'$\pi / 4$', r'$\pi / 2$', r'$3\pi / 4$', r'$\pi$'], fontsize=15)

# Add colorbar
fig.colorbar(h[3], ax=ax, label=r'Normalized $\left< \int \rho_\mathrm{NFW}(r)\, ds \right>$')
lgnd = plt.legend(handles=[gc_scatter],fontsize=14, frameon=True, framealpha=0.05)

# Fix marker size using the handles returned by legend
for handle in lgnd.legend_handles:
    handle.set_sizes([30])  # marker size in points²


plt.tight_layout()
plt.savefig('plots/DUNE_Honda_Density_LOS_Map.png', dpi=300)
plt.show()

