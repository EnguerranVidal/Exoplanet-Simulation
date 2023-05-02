from generator import *
from classes import *


def testGeneration():
    image = elevation_map_ocean(600, 300, 1, 80, 0.5)
    sphere_plot(image)
    # land_value = np.chararray(shape=image.shape)
    # print(land_value)


def test_orbit():
    Sun = Star()
    orbit = Orbit(foyer=Sun)
    orbit.plot_orbit(1000)


def testPlanet():
    planet = PlanetNoAtm()
    planet.create_maps(n_longs=210, n_lats=110)
    planet.rotation()
    planet.define_tilt()
    sim = ExoSimulation(planet=planet)
    sim.run(T=10)


if __name__ == '__main__':
    testPlanet()
