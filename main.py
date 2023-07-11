from classes import *
from functions import *


def testGeneration():
    seed = 1
    parameters = {'OCEAN_THRESHOLD': 0,
                  'NB_OCTAVES': 10,
                  'SCALE': 1,
                  'STRENGTH': 3,
                  'TERRAIN_ALBEDO': 0.4,
                  'OCEAN_ALBEDO': 0.06,
                  'CAPS_ALBEDO': 0.9,
                  'CAPS_EXTENT': 0}
    points = sunflowerSphereDistribution(30000)
    generatePlanetMap(points, parameters, oceans=True, caps=True, seed=seed, plot=True)


def testOrbit():
    Sun = Star()
    orbit = Orbit(foyer=Sun)
    orbit.orbitalPlot(1000)


def testPlanet():
    planet = PlanetNoAtm()
    planet.createMaps(n_longs=210, n_lats=110)
    planet.rotation()
    planet.defineTilt()
    sim = ExoSimulation(planet=planet)
    sim.run(T=10)


if __name__ == '__main__':
    testGeneration()
