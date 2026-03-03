import astropy.units as u

# stellar parameters
r_val = 1.0
n_points = 65000
u1 = 0.196092
u2 = 0.228748  # limb darkening coefficients
rotation_period = 1.90 * u.day
# Point of view
elev = 11.63
azim = 0
# List of spots
spots = []
# base lines time parameter
observing_baseline_days = 5 * u.day
cadence_time = 60 * u.minute
total_frames = int((observing_baseline_days / cadence_time).decompose().value)
