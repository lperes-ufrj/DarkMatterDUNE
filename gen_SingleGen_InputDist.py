import csv, math
import numpy as np
import uproot

in_csv = "/home/leoperes/DarkMatterDUNE/dm_direction_samples_detector.csv"
out_root = "2dhist_los_density_sampled.root"
hist_name = "angle_hist"  # must match your fhicl ThetaXzYzHist entry

txz, tyz = [], []

with open(in_csv) as f:
    r = csv.DictReader(f)
    for row in r:
        x = float(row["X_drift"])
        y = float(row["Y_zenith"])
        z = float(row["Z_beam"])

        # normalize to unit direction (important!)
        norm = math.sqrt(x*x + y*y + z*z)
        if norm == 0: 
            continue
        x, y, z = x/norm, y/norm, z/norm

        thetaxz = math.atan2(x, z)      # [-pi, pi]
        thetayz = math.asin(max(-1.0, min(1.0, y)))  # [-pi/2, pi/2]

        txz.append(thetaxz)
        tyz.append(thetayz)

H, xedges, yedges = np.histogram2d(
    txz, tyz,
    bins=(100, 100),
    range=[[-np.pi, np.pi], [-0.5*np.pi, 0.5*np.pi]]
)

with uproot.recreate(out_root) as f:
    f[hist_name] = (H, xedges, yedges)
