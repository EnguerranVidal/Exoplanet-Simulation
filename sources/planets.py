import matplotlib.pyplot as plt
import numpy as np

from classes import Orbit
from functions import *
from sources.stars import Star


class Planet:
    def __init__(self, mass=1, radius=1, orbit=Orbit()):
        # Basic Values
        self.orbit = orbit
        self.mass = mass * 5.9722 * 10 ** 24
        self.radius = radius * 6.371 * 10 ** 6
        self.rotationRate = None
        # Values Maps
        self.mapShape = None
        self.longitudes, self.latitudes = None, None
        self.albedoMap, self.elevationMap = None, None
        # Axial Tilt
        self.periLongitude = None
        self.axialTilt = None

    def setRotation(self, rotationRate=1):
        self.rotationRate = rotationRate * 7.292115 * 10 ** (-5)  # rad/sec

    def setOrbit(self, semiMajorAxis=1, eccentricity=0.01671, foyer=Star()):
        self.orbit = Orbit(semiMajorAxis, eccentricity, foyer)

    def setAxialTilt(self, axialTilt=23.456, perihelionLongitude=102.04):
        self.axialTilt = np.radians(axialTilt)
        self.periLongitude = np.radians(perihelionLongitude)

    def setArrays(self, nbPoints: int, parameters: dict, seed=0, oceans=True, caps=True, distribution='sunflower'):
        if distribution == 'sunflower':
            points = sunflowerSphereDistribution(nbPoints)
            radii, longitudes, latitudes = cartesianToGeographic(points[:, 0], points[:, 1], points[:, 2])

        elif distribution == 'fibonacci':
            points = fibonacciSphereDistribution(nbPoints)
            radii, longitudes, latitudes = cartesianToGeographic(points[:, 0], points[:, 1], points[:, 2])

        else:
            nbLong, nbLat = findClosestFactors(nbPoints)  # Grid dimensions from nbPoints
            longitudeSpace = np.linspace(-np.pi, np.pi, nbLong)
            latitudeSpace = np.linspace(-np.pi/2, np.pi/2, nbLat)
            longitudeMesh, latitudeMesh = np.meshgrid(longitudeSpace, latitudeSpace)
            longitudes, latitudes = longitudeMesh.ravel(), latitudeMesh.ravel()
            radii = np.full_like(longitudes, 1)
            X, Y, Z = geographicToCartesian(radii, longitudes, latitudes)
            points = np.column_stack([X, Y, Z])

        albedo, elevation = generatePlanetMap(points, parameters, oceans=oceans, caps=caps, seed=seed)
        # Defining Value Maps
        self.longitudes = longitudes
        self.latitudes = latitudes
        self.albedoMap = albedo
        self.elevationMap = elevation
        self.mapShape = self.longitudes.shape

    def starDeclination(self, trueAnomaly):
        return np.arcsin(np.sin(self.axialTilt) * np.sin(trueAnomaly + self.periLongitude))

    def starIrradiance(self, t):
        trueAnomaly, starDistance = self.orbit.orbitalPosition(t)
        # HOUR ANGLES
        hours = np.full(self.mapShape, self.rotationRate) * t + self.longitudes - np.full(self.mapShape, trueAnomaly)
        # DECLINATION ANGLES
        declination = self.starDeclination(trueAnomaly)
        declinations = np.full(self.mapShape, declination)
        # INCIDENCE ANGLES
        incidences = np.sin(self.latitudes) * np.sin(declinations) + np.cos(self.latitudes) * np.cos(declinations) * np.cos(hours)
        incidences = np.maximum(np.zeros(self.mapShape), incidences)
        return incidences * self.orbit.foyer.luminosity / (4 * np.pi * starDistance ** 2)


class TidallyLockedPlanet(Planet):
    def __init__(self, mass, radius, orbit=Orbit()):
        super().__init__(mass, radius, orbit)

    def starIrradiance(self, t):
        trueAnomaly, starDistance = self.orbit.orbitalPosition(t)
        # HOUR ANGLES
        hours = self.longitudes
        # INCIDENCE ANGLES
        incidences = np.cos(self.latitudes) * np.cos(hours)
        incidences = np.maximum(np.zeros(self.mapShape), incidences)
        return incidences * self.orbit.foyer.luminosity / (4 * np.pi * starDistance ** 2)

    def starDeclination(self, trueAnomaly):
        print('')


