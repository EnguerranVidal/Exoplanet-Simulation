from opensimplex import OpenSimplex
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs


generation_seed = 1
gen = OpenSimplex(seed=generation_seed)


def elevation_map_ocean(width, height, factor, n_tiles, sea_level=0):
    # ELEVATION MAP
    elevation = np.zeros(shape=(height, width))
    long = np.flip(np.linspace(-np.pi, np.pi, num=width))
    lat = np.flip(np.linspace(-np.pi / 2, np.pi / 2, num=height))
    for j in range(height):
        for i in range(width):
            x = np.cos(lat[j]) * np.cos(long[i])
            y = np.cos(lat[j]) * np.sin(long[i])
            z = np.sin(lat[j])
            e = noise_3D(factor * x, factor * y, factor * z) \
                + 0.5 * noise_3D(factor * 2 * x, factor * 2 * y, factor * 2 * z) \
                + 0.25 * noise_3D(factor * 4 * x, factor * 4 * y, factor * 4 * z)
            e = e / (1 + 0.5 + 0.25)
            elevation[j][i] = int(e * n_tiles) / n_tiles + 0.5
    below = elevation <= sea_level
    elevation[below] = sea_level
    return elevation


def albedo_map_caps(width, height, factor, n_tiles, ocean_level=0, polar_caps=0):
    ocean_albedo = 0.1
    ice_albedo = 0.9
    pass


def noise(nx, ny):
    return gen.noise2d(nx, ny)


def noise_3D(nx, ny, nz):
    return gen.noise3d(nx, ny, nz)


def map_sphere(width, height, factor, power):
    value = np.zeros(shape=(height, width))
    long = np.flip(np.linspace(-np.pi, np.pi, num=width))
    lat = np.flip(np.linspace(-np.pi / 2, np.pi / 2, num=height))
    for j in range(height):
        for i in range(width):
            x = np.cos(lat[j]) * np.cos(long[i])
            y = np.cos(lat[j]) * np.sin(long[i])
            z = np.sin(lat[j])
            e = noise_3D(factor * x, factor * y, factor * z) \
                + 0.5 * noise_3D(factor * 2 * x, factor * 2 * y, factor * 2 * z) \
                + 0.25 * noise_3D(factor * 4 * x, factor * 4 * y, factor * 4 * z)
            e = e / (1 + 0.5 + 0.25)
            value[j][i] = int(e * power) / power
    return value


def sphere_plot(image):
    fig = plt.figure(figsize=(10, 5))
    # AXIS 1
    ax1 = fig.add_subplot(121, projection=ccrs.Orthographic(0, 90))
    ax1.set_title("North Pole")
    ax1.gridlines(color='black', linestyle='dotted')
    ax1.imshow(image, origin="upper", extent=(-180, 180, -90, 90), transform=ccrs.PlateCarree())
    # AXIS 2
    ax2 = fig.add_subplot(122, projection=ccrs.Orthographic(0, -90))
    ax2.set_title("South Pole")
    ax2.gridlines(color='black', linestyle='dotted')
    ax2.imshow(image, origin="upper", extent=(-180, 180, -90, 90), transform=ccrs.PlateCarree())
    plt.show()