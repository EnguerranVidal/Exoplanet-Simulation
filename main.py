from classes import *
from functions import *


def testGeneration():
    seed = 10
    parameters = {'OCEAN_THRESHOLD': 0,
                  'NB_OCTAVES': 10,
                  'SCALE': 1,
                  'STRENGTH': 1,
                  'TERRAIN_ALBEDO': 0.3,
                  'OCEAN_ALBEDO': 0.7,
                  'CAPS_ALBEDO': 0.05,
                  'CAPS_EXTENT': 30}
    points = sunflowerSphereDistribution(10000)
    generatePlanetMap(points, parameters, oceans=True, caps=True,
                      seed=seed, plot=True)


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
