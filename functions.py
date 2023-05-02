import numpy as np
import noise
import opensimplex
from matplotlib import pyplot as plt
from scipy.interpolate import griddata
import plotly.graph_objects as go
from scipy.spatial import SphericalVoronoi, geometric_slerp


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


def interpolateMercator(longitudes, latitudes, values, nbLongitudes=1000, nbLatitudes=500):
    newLongitudes = np.linspace(-np.pi, np.pi, num=nbLongitudes)
    newLatitudes = np.linspace(-np.pi / 2, np.pi / 2, num=nbLatitudes)
    meshLongitudes, meshLatitudes = np.meshgrid(newLongitudes, newLatitudes)

    gridValues = griddata((longitudes, latitudes), values, (meshLongitudes, meshLatitudes), method='linear')
    return gridValues


def cartesianToGeographic(points):
    X, Y, Z = points[:, 0], points[:, 1], points[:, 2]
    radii = np.sqrt(X ** 2 + Y ** 2 + Z ** 2)
    longitudes = np.arctan2(Y / radii, X / radii)
    latitudes = np.arcsin(Z / radii)
    return radii, longitudes, latitudes


def geographicToCartesian(radii, longitudes, latitudes):
    X = radii * np.cos(latitudes) * np.cos(longitudes)
    Y = radii * np.cos(latitudes) * np.sin(longitudes)
    Z = radii * np.sin(latitudes)
    return np.column_stack((X, Y, Z))


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
            if region_index != other_index and len(set(region).intersection(other_region)) >= 2:
                vertices = list(set(region).intersection(other_region))
                vertex1, vertex2 = sv.vertices[vertices[0]], sv.vertices[vertices[1]]
                neighbours[other_index] = geodesicDistance(vertex1, vertex2, radius=1, center=center)
        cells[-1]['NEIGHBOURS'] = neighbours
        cells[-1]['AREA'] = areas[region_index]

    return cells


def generateNoise3D(points, seed=0):
    simplex = opensimplex.OpenSimplex(seed)
    size = points.shape[0]
    noiseMap = np.zeros(size, dtype=float)
    for i in range(size):
        noiseMap[i] = simplex.noise3(points[i, 0], points[i, 1], points[i, 2])
    return noiseMap


def generateContinents(points, seed=0, threshold=0., plot=False):
    noiseMap = generateNoise3D(points, seed=seed)
    submerged = noiseMap < threshold
    noiseMap[submerged] = 0
    noiseMap[~submerged] = 1
    if plot:
        R, longitudes, latitudes = cartesianToGeographic(points)
        image = interpolateMercator(longitudes, latitudes, noiseMap, nbLongitudes=1000, nbLatitudes=500)
        plt.imshow(image, cmap='gray')
        plt.show()
    return noiseMap
