from utils import *
import datetime
import numpy as np

p_mass = 938.2 # MeV/c^2
p_charge = 1 # e
p_energy = [x for x in range(3000, 15000, 1000)] # MeV

lat, lon, alt_km, date = 55.0, 37.0, 20.0, datetime.datetime(2023, 1, 1)
dt = 1e-3
steps = 10
for energy in p_energy:
    initial_speed = np.sqrt(2 * energy / p_mass)
    initial_state = np.array([*geog_to_cartesian(lat, lon, alt_km), initial_speed, initial_speed, initial_speed])  # x, y, z, vx, vy, vz
    trajectory = compute_trajectory(initial_state, 0, dt, steps, particle_motion, p_mass, p_charge, B_north, B_east, B_down)
    visualize_trajectory(trajectory)




