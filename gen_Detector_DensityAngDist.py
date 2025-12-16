import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from astropy import coordinates as coords
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, AltAz, Galactic
import astropy.units as u
import dunestyle.matplotlib as dunestyle

FluxRotValues = np.array([
    [+0.9877, -0.1564, +0.0000],  # new x axis in old coordinates
    [+0.0000, +0.0000, +1.0000],  # new y axis in old coordinates
    [-0.1564, -0.9877, +0.0000]   # new z axis in old coordinates
])

# Define bin edges
n_theta_bins = 90
n_phi_bins = 180
theta_edges = np.linspace(0, np.pi, n_theta_bins + 1)
phi_edges = np.linspace(0, 2*np.pi, n_phi_bins + 1)

# Verify rotation matrix properties
print("det(R) ≈", np.linalg.det(FluxRotValues))      # should be +1
print("R R^T ≈ I:", np.allclose(FluxRotValues @ FluxRotValues.T, np.eye(3), 1e-4))

# Load data from .npy files
theta = np.load('npy_arrays/theta_honda.npy')
phi_signed = np.load('npy_arrays/phi_honda.npy')
intensity_sum_norm = np.load('npy_arrays/intensity_sum_norm.npy')
gc_pos = np.load('npy_arrays/gc_pos.npy')


# -------------------------------------------------------------
# 0) Your arrays ready: theta, phi_signed, intensity_sum_norm
#    and the edges (we will use phi in [-pi, pi] for hist2d)
# -------------------------------------------------------------
phi_edges_signed = np.linspace(-np.pi, np.pi, n_phi_bins + 1)

# -------------------------------------------------------------
# 1) Build histogram and PMF with Jacobian dOmega = sin(theta) dtheta dphi
# -------------------------------------------------------------
H, _, _ = np.histogram2d(
    theta, phi_signed,
    bins=[theta_edges, phi_edges_signed],
    weights=intensity_sum_norm
)  # shape (n_th, n_ph)

theta_centers = 0.5 * (theta_edges[:-1] + theta_edges[1:])
dtheta = theta_edges[1] - theta_edges[0]
dphi   = phi_edges_signed[1] - phi_edges_signed[0]

dOmega = (np.sin(theta_centers)[:, None]) * dtheta * dphi   # (n_th, n_ph)
W = np.clip(H, 0, np.inf) * dOmega
P = W / W.sum()                                             # PMF (sum=1)

# Flattened CDF for sampling
cdf = np.cumsum(P.ravel()); cdf[-1] = 1.0
n_th, n_ph = P.shape

def sample_theta_phi(N, rng=np.random.default_rng()):
    """Sample N (theta, phi) pairs from PMF; uniform jitter inside bin."""
    idx = np.searchsorted(cdf, rng.random(N), side="right")
    i_th = idx // n_ph
    i_ph = idx %  n_ph
    th = theta_edges[i_th] + (theta_edges[i_th+1] - theta_edges[i_th]) * rng.random(N)
    ph = phi_edges_signed[i_ph] + (phi_edges_signed[i_ph+1] - phi_edges_signed[i_ph]) * rng.random(N)
    # keep phi in [-pi, pi]
    ph = ((ph + np.pi) % (2*np.pi)) - np.pi
    return th, ph

# -------------------------------------------------------------
# 2) Convert (theta, phi) -> local vector (south, east, up) and rotate
# -------------------------------------------------------------
def theta_phi_to_local_vec(theta_rad, phi_rad):
    x_south = np.sin(theta_rad) * np.cos(phi_rad)
    y_east  = np.sin(theta_rad) * np.sin(phi_rad)
    z_up    = np.cos(theta_rad)
    return np.vstack([x_south, y_east, z_up])  # shape (3, N)

def rotate_to_detector(v_local, R):
    """Rotate: v_det = R @ v_local; R rows are (X,Y,Z) in (south,east,up)."""
    return R @ v_local  # returns (3, N)

# -------------------------------------------------------------
# 3) Detector angles: choose 'zenith' (theta from +Y) or 'beam' (theta from +Z)
# -------------------------------------------------------------
def detector_angles(v_det, mode="zenith"):
    X, Y, Z = v_det
    if mode == "zenith":
        theta_det = np.arccos(np.clip(Y, -1.0, 1.0))  # [0, pi], polar around +Y
        phi_det   = np.arctan2(Z, X)                  # (-pi, pi], azimuth in X–Z from +X to +Z
    elif mode == "beam":  # SingleGen
        theta_det = np.arccos(np.clip(Z, -1.0, 1.0))  # [0, pi], polar around +Z (beam)
        phi_det   = np.arctan2(Y, X)                  # (-pi, pi], azimuth in X–Y from +X to +Y
    else:
        raise ValueError("mode must be 'zenith' or 'beam'")
    # wrap phi to [-pi, pi] for plotting
    phi_det = ((phi_det + np.pi) % (2*np.pi)) - np.pi
    return theta_det, phi_det

# -------------------------------------------------------------
# 4) Sample, rotate
# -------------------------------------------------------------
N = 3_000_000  # number of directions to sample (consider smaller for speed)
theta_samp, phi_samp = sample_theta_phi(N)

# Local vectors (south,east,up)
v_local = theta_phi_to_local_vec(theta_samp, phi_samp)

# Rotate to detector frame
R = FluxRotValues
v_det = rotate_to_detector(v_local, R)

# Choose detector-angle mode:
mode = "beam"   # or "zenith"
theta_det, phi_det = detector_angles(v_det, mode=mode)

# -------------------------------------------------------------
# 4b) Galactic Center track → detector frame (overlay)
#     expects gc_pos: array of shape (N_times, 2) with [az_rad, alt_rad]
# -------------------------------------------------------------
def altaz_to_seu_rad(alt_rad, az_rad):
    """SEU unit vector from AltAz in radians (az: 0=N, 90=E). Returns (3, N)."""
    south = -np.cos(alt_rad) * np.cos(az_rad)
    east  =  np.cos(alt_rad) * np.sin(az_rad)
    up    =  np.sin(alt_rad)
    return np.vstack([south, east, up])  # (3, N)


az_gc = gc_pos[:, 0]
alt_gc = gc_pos[:, 1]

v_gc_local = altaz_to_seu_rad(alt_gc, az_gc)         # (3, N_times)
v_gc_det   = rotate_to_detector(v_gc_local, R)       # (3, N_times)
theta_gc_det, phi_gc_det = detector_angles(v_gc_det, mode=mode)

# -------------------------------------------------------------
# 5) Plot (samples + GC track)
# -------------------------------------------------------------
fig, ax = plt.subplots(dpi=100, figsize=(10, 6))
h = ax.hist2d(theta_det, phi_det, bins=[50, 50], cmap='viridis')

# GC overlay
ax.plot(theta_gc_det, phi_gc_det, '.', color='red', ms=2, alpha=0.7, label='GC track')

ax.set_xlabel(r'$\theta_{det}$ (rad)', fontsize=22)
ax.set_ylabel(r'$\phi_{det}$ (rad)', fontsize=22)

# X-axis ticks and labels
ax.set_xticks([0.0, 0.78539, 1.570, 2.356, 3.14159])
ax.set_xticklabels([r'$0$', r'$\pi / 4$', r'$\pi / 2$', r'$3 \pi / 4$', r'$\pi$'], fontsize=15)

# Y-axis ticks and labels
ax.set_yticks([-3.14159, -2.356, -1.570, -0.78539, 0.0, 0.78539, 1.570, 2.356, 3.14159])
ax.set_yticklabels([r'$-\pi$', r'$-3\pi / 4$', r'$-\pi / 2$', r'$-\pi / 4$', r'$0$',
                    r'$\pi / 4$', r'$\pi / 2$', r'$3\pi / 4$', r'$\pi$'], fontsize=15)

cbar = fig.colorbar(h[3], ax=ax)
cbar.set_label('Samples (arb. units)')

plt.legend(fontsize=18, frameon=True, framealpha=0.1, loc='upper right', markerscale=5)


plt.tight_layout()
plt.savefig('plots/DUNE_Detector_Frame_Directions_Sampled.png', dpi=300)
plt.show()

# -------------------------------------------------------------
# 6) export directions in detector frame
# -------------------------------------------------------------
dirs_det = v_det.T  # (N, 3) columns [X_drift, Y_zenith, Z_beam]
phi_beam_02pi = np.mod(phi_det, 2*np.pi)   # convert from [-pi,pi] to [0,2pi) for export

np.savetxt("dm_direction_samples_detector.csv",
    np.column_stack([theta_det, phi_beam_02pi, dirs_det]),
    delimiter=",",
    header="theta_rad,phi_rad,X_drift,Y_zenith,Z_beam", comments="")

print("Samples saved to dm_direction_samples_detector.csv")


# -------------------------------------------------------------
# 7) Plot Histogram (X,Y,Z) coordinates in detector frame
# -------------------------------------------------------------
fig, ax = plt.subplots(dpi=100, figsize=(10, 6))
ax.hist(v_det[0], bins=50, histtype='step',label='X_drift')
ax.hist(v_det[1], bins=50, histtype='step',label='Y_zenith')
ax.hist(v_det[2], bins=50, histtype='step',label='Z_beam')

ax.set_xlabel(r'Vector Direction Coordinate', fontsize=22)
ax.set_ylabel(r'Number of Directions', fontsize=22)


plt.legend(fontsize=18, frameon=True, framealpha=0.5, ncols=3,markerscale=5)


plt.tight_layout()
plt.savefig('plots/DUNE_Detector_Frame_Histograms_Coord_Sampled.png', dpi=300)
plt.show()

