import numpy as np
from pyIGRF import igrf_value
from pyproj import Transformer
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev
import math

#GS84 ellipsoid model
def cartesian_to_geodetic(x, y, z):
    a = 6378137.0
    f = 1 / 298.257223563
    b = a * (1 - f)
    e = math.sqrt(f * (2 - f))

    lon = math.atan2(y, x)

    p = math.sqrt(x**2 + y**2)
    E = math.sqrt(a**2 - b**2)
    F = 54 * b**2 * z**2
    G = p**2 + (1 - e**2) * z**2 - e**2 * E**2
    c = (e**4 * F * p**2) / (G**3)
    s = (1 + c + math.sqrt(c**2 + 2 * c))**(1/3)
    P = F / (3 * (s + 1/s + 1)**2 * G**2)
    Q = math.sqrt(1 + 2 * e**4 * P)
    r_0 = -(P * e**2 * p) / (1 + Q) + math.sqrt(0.5 * a**2 * (1 + 1/Q) - P * (1 - e**2) * z**2 / (Q * (1 + Q)) - 0.5 * P * p**2)
    U = math.sqrt((p - e**2 * r_0)**2 + z**2)
    V = math.sqrt((p - e**2 * r_0)**2 + (1 - e**2) * z**2)
    z_0 = (b**2 * z) / (a * V)
    height = U * (1 - b**2 / (a * V))
    if p < 1e-10:
        lat = math.pi / 2 if z > 0 else -math.pi / 2
    else:
        lat = math.atan((z + e ** 2 * z_0) / p)

    lat = math.degrees(lat)
    lon = math.degrees(lon)
    height /= 1000

    return lat, lon, height


def B_north(lat, lon, alt_km):
    _, _, _, B_north, _, _, _ = igrf_value(lat, lon, alt_km)
    return B_north / 1e9

def B_east(lat, lon, alt_km):
    _, _, _, _, B_east, _, _ = igrf_value(lat, lon, alt_km)
    return B_east / 1e9

def B_down(lat, lon, alt_km):
    _, _, _, _, _, B_down, _ = igrf_value(lat, lon, alt_km)
    return B_down / 1e9

# Constant IGRF field
def get_igrf_field(lat, lon, alt, date):
    alt_km = alt / 1000
    decimal_year = date.year + (date.month - 1) / 12
    _, _, _, B_north, B_east, B_down, B_total = igrf_value(lat, lon, alt_km, decimal_year)
    return B_north / 1e9, B_east / 1e9, B_down / 1e9

def geodetic_to_ecef(lat, lon, alt):
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:4978")
    x, y, z = transformer.transform(lat, lon, alt)
    return x, y, z

def runge_kutta_step(func, y, t, dt, *args, **kwargs):
    k1 = dt * func(t, y, *args, **kwargs)
    k2 = dt * func(t + dt / 2, y + k1 / 2, *args, **kwargs)
    k3 = dt * func(t + dt / 2, y + k2 / 2, *args, **kwargs)
    k4 = dt * func(t + dt, y + k3, *args, **kwargs)
    return y + (k1 + 2 * k2 + 2 * k3 + k4) / 6

def compute_trajectory(y0, t0, dt, steps, func, *args, **kwargs):
    y = np.zeros((steps+1, y0.size))
    y[0] = y0
    t = t0
    for i in range(steps):
        y[i+1] = runge_kutta_step(func, y[i], t, dt, *args, **kwargs)
        t += dt
    return y

def particle_motion(t, state, mass, charge, B):
    x, y, z, vx, vy, vz = state
    # lat, lon, alt_km = cartesian_to_geodetic(x, y, z)
    # B = np.array([Bx(lat, lon, alt_km), By(lat, lon, alt_km), Bz(lat, lon, alt_km)])
    v = np.array([vx, vy, vz])
    force = charge * np.cross(v, B) / mass
    ax, ay, az = force

    return np.array([vx, vy, vz, ax, ay, az])


def visualize_trajectory(trajectory):
    x = trajectory[:, 0]
    y = trajectory[:, 1]
    z = trajectory[:, 2]
    print(x, x.shape)
    print(y, y.shape)
    print(z, z.shape)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    tck, u = splprep([x, y, z], s=0)
    u_fine = np.linspace(0, 1, len(x) * 10)
    x_fine, y_fine, z_fine = splev(u_fine, tck)

    ax.plot(x_fine, y_fine, z_fine, 'r-', linewidth=2)

    ax.set_title('Muon Trajectory')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')

    max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2.0
    mid_x = (x.max()+x.min()) * 0.5
    mid_y = (y.max()+y.min()) * 0.5
    mid_z = (z.max()+z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    plt.show()


