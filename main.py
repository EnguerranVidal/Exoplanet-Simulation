from classes import *
from functions import *


def testGeneration():
    seed = 10
    octaves = 10
    scale = 1
    threshold = 0.0
    strength = 1.0
    points = sunflowerSphereDistribution(10000)
    noiseMap = generateContinents(points, seed, threshold, octaves, scale, strength, plot=True)


def test_orbit():
    Sun = Star()
    orbit = Orbit(foyer=Sun)
    orbit.orbitalPlot(1000)


def testPlanet():
    planet = PlanetNoAtm()
    planet.create_maps(n_longs=210, n_lats=110)
    planet.rotation()
    planet.define_tilt()
    sim = ExoSimulation(planet=planet)
    sim.run(T=10)


if __name__ == '__main__':
    testGeneration()
