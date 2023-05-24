import numpy as np
from pyIGRF import igrf_value
from mpl_toolkits import mplot3d
from pyproj import Transformer
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev
import math

#GS84 ellipsoid model

def cartesian_to_geodetic_eberly(x, y, z):
    # Constants for WGS84 ellipsoid model of the Earth
    a = 6378137.0  # semi-major axis
    f = 1 / 298.257223563  # inverse flattening
    e2 = f * (2 - f)  # square of eccentricity

    lon = math.atan2(y, x)  # longitude is easy

    # Prepare for iterative solution for latitude and height
    r = math.sqrt(x*x + y*y + z*z)  # radial distance from Earth's center
    esinz = a * e2 * z / r  # e * sin(z), used in iteration

    # Iterative calculation of latitude and height
    p = x*x + y*y  # squared horizontal projection of the radial vector
    q = z*z * (1 - e2)  # adjusted squared vertical projection
    r = (p - q) / (6 * esinz)
    s = (p + q) / (6 * esinz)
    t = math.sqrt(r*r + s*s)
    u = math.pow(((s + t + r) / 3), (1 / 3))
    v = math.pow(((s + t - r) / 3), (1 / 3))
    w = math.sqrt(u*u + v*v + q / (4 * esinz))
    k = math.sqrt(u*u + v*v + w*w)

    # Final calculations for latitude and height
    e = (k - w) / (2 * k)
    f = (k + w) / (2 * k * k + w + e / 2)
    g = 1 - 2 * e * w / k
    h = 2 * (k - e) * f
    sinz = h * (1 - f)
    cosz = g * f
    lat = math.atan(sinz / cosz)
    height = (r * sinz + t * cosz) / math.sqrt(e2)

    # Convert to degrees and kilometers
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

def geog_to_cartesian(latitude, longitude, radius=1.0):
    lat_rad = math.radians(latitude)
    lon_rad = math.radians(longitude)

    x = radius * math.cos(lat_rad) * math.cos(lon_rad)
    y = radius * math.cos(lat_rad) * math.sin(lon_rad)
    z = radius * math.sin(lat_rad)

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

def particle_motion(t, state, mass, charge, Bx, By, Bz):
    x, y, z, vx, vy, vz = state
    lat, lon, alt_km = cartesian_to_geodetic_eberly(x, y, z)
    B = np.array([Bx(lat, lon, alt_km), By(lat, lon, alt_km), Bz(lat, lon, alt_km)])
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
    fig = plt.figure(figsize = (10, 7))
    ax = plt.axes(projection ="3d")

    # Creating plot
    ax.scatter3D(x, y, z, color='#ff0000', s=40)
    plt.show()
    # ax = fig.add_subplot(111, projection='3d')
    #
    # tck, u = splprep([x, y, z], s=0)
    # u_fine = np.linspace(0, 1, len(x) * 10)
    # x_fine, y_fine, z_fine = splev(u_fine, tck)
    #
    # ax.plot(x_fine, y_fine, z_fine, 'r-', linewidth=2)
    #
    # ax.set_title('Particle Trajectory')
    # ax.set_xlabel('X (m)')
    # ax.set_ylabel('Y (m)')
    # ax.set_zlabel('Z (m)')
    #
    # max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2.0
    # mid_x = (x.max()+x.min()) * 0.5
    # mid_y = (y.max()+y.min()) * 0.5
    # mid_z = (z.max()+z.min()) * 0.5
    # ax.set_xlim(mid_x - max_range, mid_x + max_range)
    # ax.set_ylim(mid_y - max_range, mid_y + max_range)
    # ax.set_zlim(mid_z - max_range, mid_z + max_range)


    plt.show()


