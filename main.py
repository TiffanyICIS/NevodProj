from utils import *
import datetime
import numpy as np

muon_mass = 105.7 # MeV/c^2
muon_charge = -1 # e
muon_energy = 1000 # MeV

lat, lon, alt_km = 45.0, -75.0, 100.0
x, y, z = geodetic_to_ecef(lat, lon, alt_km)

B_north, B_east, B_down = get_igrf_field(lat, lon, alt_km, datetime.datetime(2023, 1, 1))
B = np.array([B_north, B_east, B_down])

initial_speed = np.sqrt(2 * muon_energy / muon_mass)
initial_state = np.array([x, y, z, initial_speed, 0, 0]) # x, y, z, vx, vy, vz

dt = 1e-3
steps = 100000

trajectory = compute_trajectory(initial_state, 0, dt, steps, particle_motion, muon_mass, muon_charge, B)

visualize_trajectory(trajectory)




