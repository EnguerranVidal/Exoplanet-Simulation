import numpy as np
import opensimplex

import matplotlib.pyplot as plt
import matplotlib.colors as colors

import plotly.graph_objects as go
import cartopy.crs as ccrs

from scipy.spatial import SphericalVoronoi, geometric_slerp
from scipy.interpolate import griddata


def hours_to_seconds(t):
    return t * 3600


def fibonacciSphereDistribution(samples=1000, plot=False):
    phiAngle = np.pi * (3 - np.sqrt(5))  # Golden  np.Angle in radians
    y = np.linspace(1, -1, num=samples)
    radius = np.sqrt(1 - y ** 2)
    theta = phiAngle * np.arange(samples)
    x = np.cos(theta) * radius
    z = np.sin(theta) * radius

    if plot:
        # Create a scatter plot of the original points
        scatter = go.Scatter3d(x=x, y=y, z=z, mode='markers', marker=dict(size=3, color='blue', ))
        fig = go.Figure(data=[scatter])
        fig.update_layout(scene=dict(aspectmode='data'))
        fig.update_layout(coloraxis=dict(colorscale='Jet'))
        # Show the plot
        fig.show()

    return np.column_stack((x, y, z))


def geodesicDistance(point1, point2, radius=1):
    # point1, point2 = point1 - center, point2 - center
    point1, point2 = point1 / radius, point2 / radius
    dotProduct = np.dot(point1, point2)
    dotProduct = np.clip(dotProduct, -1, 1)
    deltaAngle = np.arccos(dotProduct)
    return deltaAngle * radius


def sunflowerSphereDistribution(samples=1000, plot=False):
    indices = np.arange(0, samples, dtype=float) + 0.5
    phi = np.arccos(1 - 2 * indices / samples)
    theta = np.pi * (1 + 5 ** 0.5) * indices
    x, y, z = np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)

    if plot:
        # Create a scatter plot of the original points
        scatter = go.Scatter3d(x=x, y=y, z=z, mode='markers', marker=dict(size=3, color='blue', ))
        fig = go.Figure(data=[scatter])
        fig.update_layout(scene=dict(aspectmode='data'))
        fig.update_layout(coloraxis=dict(colorscale='Jet'))
        # Show the plot
        fig.show()
    return np.column_stack((x, y, z))


def interpolateSphere3D(points, values, num_latitudes, num_longitudes, method='nearest'):
    # Convert the points from spherical coordinates to Cartesian coordinates
    X, Y, Z = points[:, 0], points[:, 1], points[:, 2]
    # Create a grid of longitudes and latitudes
    lon = np.linspace(0, 2*np.pi, num_longitudes)
    lat = np.linspace(-np.pi/2, np.pi/2, num_latitudes)
    lon, lat = np.meshgrid(lon, lat)
    # Convert the grid points to Cartesian coordinates
    gx, gy, gz = geographicToCartesian(np.ones_like(lon), lon, lat)
    # Interpolate the values on the grid using nearest-neighbor interpolation
    image = griddata((X, Y, Z), values, (gx, gy, gz), method)
    return image



def cartesianToGeographic(X, Y, Z):
    radii = np.sqrt(X ** 2 + Y ** 2 + Z ** 2)
    longitudes = np.arctan2(Y / radii, X / radii)
    latitudes = np.arcsin(Z / radii)
    return radii, longitudes, latitudes


def geographicToCartesian(radii, longitudes, latitudes):
    X = radii * np.cos(latitudes) * np.cos(longitudes)
    Y = radii * np.cos(latitudes) * np.sin(longitudes)
    Z = radii * np.sin(latitudes)
    return X, Y, Z


def sphericalVoronoiEqualCells(points, radius=1, center=np.array([0, 0, 0]), plot=False, stats=False):
    sv = SphericalVoronoi(points, radius, center)
    sv.sort_vertices_of_regions()
    if plot:
        fig = go.Figure(data=[go.Scatter3d(x=points[:, 0], y=points[:, 1], z=points[:, 2],
                                           mode='markers', marker=dict(size=3, color='blue'))])
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones(np.size(u)), np.cos(v))
        fig.add_trace(go.Surface(x=x, y=y, z=z, colorscale='YlOrRd', opacity=0.1))
        t_vals = np.linspace(0, 1, 2000)
        for region in sv.regions:
            n = len(region)
            for i in range(n):
                start = sv.vertices[region][i]
                end = sv.vertices[region][(i + 1) % n]
                result = geometric_slerp(start, end, t_vals)
                fig.add_trace(go.Scatter3d(
                    x=result[..., 0], y=result[..., 1], z=result[..., 2],
                    mode='lines', line=dict(color='black', width=1)
                ))
        fig.update_layout(
            scene=dict(
                xaxis=dict(showticklabels=False),
                yaxis=dict(showticklabels=False),
                zaxis=dict(showticklabels=False),
                aspectmode='cube',
                camera=dict(eye=dict(x=1.8, y=-1.8, z=0.8), up=dict(x=0, y=0, z=1))
            )
        )
        fig.show()
    areas = sv.calculate_areas()
    if stats:
        print('Mean Area : ', np.mean(areas))
        print('Variance : ', np.var(areas))

    # Get the indices of neighboring cells for each cell
    cells = []
    for region_index, region in enumerate(sv.regions):
        cells.append({})
        neighbours = {}
        for other_index, other_region in enumerate(sv.regions):
            vertices = list(set(region).intersection(other_region))
            if region_index != other_index and len(set(region).intersection(other_region)) >= 2:
                vertex1, vertex2 = sv.vertices[vertices[0]], sv.vertices[vertices[1]]
                neighbours[other_index] = geodesicDistance(vertex1, vertex2, radius=1)
        cells[-1]['NEIGHBOURS'] = neighbours
        cells[-1]['AREA'] = areas[region_index]
        cells[-1]['VERTICES'] = sv.vertices[region]
    return cells


def generateNoise3D(points, seed=0, octaves=1, scale=1.0):
    simplex = opensimplex.OpenSimplex(seed)
    size = points.shape[0]
    noiseMap = np.zeros(size, dtype=float)
    for i in range(size):
        value = 0.0
        amplitude = 1.0
        frequency = 1.0
        for _ in range(octaves):
            value += simplex.noise3(points[i, 0] * frequency, points[i, 1] * frequency,
                                    points[i, 2] * frequency) * amplitude
            amplitude *= 0.5
            frequency *= 2.0
        noiseMap[i] = value * scale
    return noiseMap


def generatePlanetMap(points, parameters: dict, oceans=True, caps=True, seed=0, plot=False):
    threshold = parameters['OCEAN_THRESHOLD']
    octaves = parameters['NB_OCTAVES']
    scale = parameters['SCALE']
    strength = parameters['STRENGTH']
    terrainAlbedo = parameters['TERRAIN_ALBEDO']
    oceanAlbedo = parameters['OCEAN_ALBEDO']
    capsAlbedo = parameters['CAPS_ALBEDO']
    capsExtent = parameters['CAPS_EXTENT']
    radii, longitudes, latitudes = cartesianToGeographic(points[:, 0], points[:, 1], points[:, 2])
    # ELEVATION MAP GENERATION
    elevation = generateNoise3D(points, seed, octaves, scale)
    if oceans:
        submerged = elevation < threshold
        elevation[~submerged] = elevation[~submerged] / (sum([0.5 ** i for i in range(octaves)]))
        elevation[~submerged] = elevation[~submerged] ** strength
        elevation[submerged] = 0
    else:
        elevation = elevation / (sum([0.5 ** i for i in range(octaves)]))
        elevation = elevation ** strength

    # ALBEDO MAP GENERATION
    albedo = np.full_like(elevation, terrainAlbedo)
    if oceans:
        submerged = elevation <= threshold
        print(np.any(elevation <= threshold))
        albedo[submerged] = oceanAlbedo
    if caps:
        capNorthLatitude = np.pi / 2 - np.radians(capsExtent)
        capSouthLatitude = - np.pi / 2 + np.radians(capsExtent)
        frozenNorth = latitudes > capNorthLatitude
        frozenSouth = latitudes < capSouthLatitude
        albedo[frozenNorth] = capsAlbedo
        albedo[frozenSouth] = capsAlbedo

    if plot:
        elevationImage = interpolateSphere3D(points, elevation, num_longitudes=500, num_latitudes=250)
        albedoImage = interpolateSphere3D(points, albedo, num_longitudes=500, num_latitudes=250)
        polesViewGraph([elevationImage, albedoImage], flip=[False, True])

    return albedo, elevation


def polesViewGraph(images, flip=None):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    fig = plt.figure(figsize=(15, 8))
    cmap = plt.get_cmap('inferno')
    reversed_cmap = colors.ListedColormap(cmap.colors[::-1])
    if isinstance(images, np.ndarray):
        images = [images]
    if flip is None:
        flip = False

    n_images = len(images)
    for i, img in enumerate(images):
        ax1 = fig.add_subplot(n_images, 3, (i * 3) + 1, projection=ccrs.Orthographic(0, 90))
        if isinstance(flip, list) and flip[i]:
            ax1.imshow(img, transform=ccrs.PlateCarree(), cmap=reversed_cmap)
        else:
            ax1.imshow(img, transform=ccrs.PlateCarree(), cmap=cmap)
        ax1.gridlines(color='black', linestyle='dotted')

        ax2 = fig.add_subplot(n_images, 3, (i * 3) + 2, projection=ccrs.Orthographic(0, -90))
        if isinstance(flip, list) and flip[i]:
            ax2.imshow(img, transform=ccrs.PlateCarree(), cmap=reversed_cmap)
        else:
            ax2.imshow(img, transform=ccrs.PlateCarree(), cmap=cmap)
        ax2.gridlines(color='black', linestyle='dotted')

        ax3 = fig.add_subplot(n_images, 3, (i * 3) + 3, projection=ccrs.PlateCarree())
        if isinstance(flip, list) and flip[i]:
            ax3.imshow(img, transform=ccrs.PlateCarree(), cmap=reversed_cmap)
        else:
            ax3.imshow(img, transform=ccrs.PlateCarree(), cmap=cmap)
        ax3.gridlines(color='black', linestyle='dotted')

        divider = make_axes_locatable(ax3)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(img, cax=cax)

    plt.show()
