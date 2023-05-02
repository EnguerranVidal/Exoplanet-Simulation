import matplotlib.pyplot as plt
import numpy as np

from functions import *


class Star:
    def __init__(self, mass=1, temperature=5772, radius=1):
        self.mass = mass * 1.98847 * 10 ** 30  # In Solar Masses
        self.temperature = temperature  # In Kelvins
        self.radius = radius * 696342000  # In Solar Radii
        # Luminosity
        self.flux = 5.670374419 * 10 ** (-8) * self.temperature ** 4
        self.luminosity = self.flux * 4 * np.pi * self.radius ** 2

    def peakWaveLength(self, temperature=None):
        if temperature is None:
            temperature = self.temperature
        return 2.897771955 * 10 ** (-3) / temperature

    def peakFrequency(self, temperature=None):
        if temperature is None:
            temperature = self.temperature
        return 5.879 * 10 ** 10 * temperature

    def emissionSpectrum(self, start, end, N, temperature=None, plot=False):
        wavelengths = np.linspace(start, end, num=N)
        if temperature is None:
            temperature = self.temperature
        h = 6.62607015 * 10 ** (-34)  # Planck Constant
        c = 299792458  # Speed of light
        k = 1.380649 * 10 ** (-23)  # Boltzmann constant
        e0 = 8.85418781762039 * 10 ** (-12)  # Vacuum permittivity
        intensities = (2 * h * c ** 2 * e0) / (wavelengths ** 5 * (np.exp(h * c / (wavelengths * k * temperature)) - 1))

        if plot:
            fig = go.Figure(data=go.Scatter(x=wavelengths, y=intensities))
            fig.update_layout(
                title="Emission Spectrum",
                xaxis_title="Wavelength",
                yaxis_title="Intensity",
            )
            fig.add_shape(type="rect", xref="x", yref="paper",
                x0=4 * 10 ** (-7), y0=0,
                x1=8 * 10 ** (-7), y1=1,
                fillcolor="red", opacity=0.2, layer="below",
            )
            fig.show()
        return intensities


class Orbit:
    def __init__(self, semiMajorAxis=1, eccentricity=0.01671, foyer=Star()):
        self.semiMajorAxis = semiMajorAxis * 1.49 * 10 ** 11
        self.eccentricity = eccentricity
        self.foyer = foyer
        self.G = 6.6740831 * 10 ** (-11)
        self.period = self.orbitalPeriod()

    def orbitalPeriod(self):
        return np.sqrt((4 * (np.pi ** 2) * self.semiMajorAxis ** 3) / (self.foyer.mass * self.G))

    def orbitalPosition(self, epoch):
        n = (2 * np.pi) / self.period
        M = n * epoch
        e = self.eccentricity
        E = M  # We initialize the eccentric anomaly with the mean anomaly value
        while E - e * np.sin(E) - M > 10 ** (-8):
            E = E - (E - e * np.sin(E) - M) / (1 - e * np.cos(E))
        nu = 2 * np.arctan(np.sqrt((1 + e) / (1 - e)) * np.tan(E / 2))
        radius = self.radiusEquation(nu)
        return nu, radius

    def keplerEquation(self, E):
        return E - self.eccentricity * np.sin(E)

    def radiusEquation(self, anomaly):
        return (self.semiMajorAxis * (1 - self.eccentricity ** 2)) / (1 + self.eccentricity * np.cos(anomaly))

    def orbitalTrace(self, N=360):
        anomalies = np.linspace(-np.pi, np.pi, N)
        radii = self.radiusEquation(anomalies)
        return radii, anomalies

    def orbitalPlot(self, N):
        anomalies = np.linspace(-np.pi, np.pi, N)
        radii = self.radiusEquation(anomalies)
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        ax.plot(anomalies, radii)
        ax.set_rticks([0.5, 1, 1.5, 2])  # Less radial ticks
        plt.show()


class PlanetNoAtm:
    def __init__(self, mass=1, radius=1, orbit=Orbit()):
        self.mass = mass * 5.9722 * 10 ** 24
        self.radius = radius * 6.371 * 10 ** 6
        self.n_long = 0
        self.n_lat = 0
        self.axial_tilt = 0
        self.NSS_phase = 0
        self.rotation_rate = None
        self.rotation_period = None
        self.surface_heat_capacity = None
        self.orbit = orbit
        # Array features
        self.longitudes = None
        self.latitudes = None
        self.albedos = None
        self.temperatures = None
        self.emissivity = None
        self.surface_heat_capacity = None
        # Specifics of the simulation run
        self.rotation_type = None

    def create_maps(self, n_longs=100, n_lats=50, mean_albedo=0.3, mean_temperature=0,
                    heat_capacity=None, emissivity=None):
        long = np.linspace(-np.pi, np.pi, num=n_longs)
        lat = np.linspace(-np.pi / 2, np.pi / 2, num=n_lats)
        self.longitudes, self.latitudes = np.meshgrid(long, np.flip(lat))
        self.albedos = np.full_like(self.longitudes, mean_albedo)
        self.temperatures = np.full_like(self.longitudes, mean_temperature)
        if heat_capacity is None:
            self.surface_heat_capacity = np.full_like(self.longitudes, 9.184448 * 10 ** 7)
        else:
            self.surface_heat_capacity = np.full_like(self.longitudes, heat_capacity)
        if emissivity is None:
            self.emissivity = np.full_like(self.longitudes, 0.8)
        else:
            self.emissivity = np.full_like(self.longitudes, emissivity)

    def rotation(self, rotation_type="default", rotation_rate=1):
        if rotation_type == "default":
            self.rotation_type = rotation_type
            self.rotation_rate = rotation_rate * 7.292115 * 10 ** (-5)
            if rotation_rate != 0.0:
                self.rotation_period = 2 * np.pi / self.rotation_rate
            else:
                self.rotation_period = np.inf

        elif rotation_type == "tidally_locked":
            self.rotation_type = rotation_type
        else:

            modes = rotation_type.split(":")
            modes = [float(i) for i in modes]
            period_o = self.orbit.period
            self.rotation_period = modes[0] * period_o / modes[1]
            self.rotation_rate = 2 * np.pi / self.rotation_period
            self.rotation_type = rotation_type

    def define_tilt(self, axial_tilt=23.436, NSS_phase=167.05):
        self.axial_tilt = np.radians(axial_tilt)
        self.NSS_phase = np.radians(NSS_phase)

    def declination(self, anomaly):
        if self.rotation_type != 'tidally_locked':
            return self.axial_tilt * np.cos(anomaly - self.NSS_phase)
        else:
            return self.axial_tilt

    def insolation(self, t):
        anomaly, radius = self.orbit.orbitalPosition(t)
        # HOUR ANGLE
        if self.rotation_type != 'tidally_locked':
            assert self.rotation_rate is not None
            hours = np.full(self.longitudes.shape, self.rotation_rate) * t + self.longitudes - np.full(
                self.longitudes.shape, anomaly)
        else:
            hours = self.longitudes
        # DECLINATION ANGLE
        declination_value = self.declination(anomaly)
        declination = np.full_like(self.longitudes, declination_value)
        incidence = np.sin(self.latitudes) * np.sin(declination) + np.cos(self.latitudes) * np.cos(declination) * np.cos(hours)
        incidence = np.maximum(np.zeros_like(self.longitudes), incidence)
        return incidence * self.orbit.foyer.luminosity / (4 * np.pi * radius ** 2)

    def model_derivative(self, t):
        W_in = self.insolation(t) * (np.ones_like(self.longitudes) - self.albedos)
        W_out = 5.67 * 10 ** (-8) * self.temperatures ** 4
        dTdt = (W_in - W_out) / self.surface_heat_capacity
        return dTdt


class ExoSimulation:
    def __init__(self, planet=None):
        if planet is None:
            self.planet = PlanetNoAtm()
            self.planet.create_maps()
        else:
            self.planet = planet

    def run(self, T=None, dt=hours_to_seconds(2), save_gif=True):
        if T is None:
            T = self.planet.orbit.period
        T = self.planet.orbit.period * T
        # INITIALIZATION
        t = 0
        N = int(T // dt)
        print(N)
        fig = plt.figure()
        # Orbit Plot
        ax1 = fig.add_subplot(121, projection='polar')
        radii, angles = self.planet.orbit.orbitalTrace()
        trace = ax1.plot(angles, radii, c='b')
        angle, radius = self.planet.orbit.orbitalPosition(t)
        position = ax1.scatter(angle, radius, c='darkblue', alpha=0.5)
        # Heatmap plot
        ax2 = fig.add_subplot(122)
        I = self.planet.insolation(t)
        # heatmap = ax2.imshow(self.planets.temperatures, vmin=0, vmax=400, cmap='inferno')
        heatmap = ax2.imshow(I, vmin=0, vmax=1500, cmap='inferno')
        plt.colorbar(heatmap)
        # MAIN LOOP
        if save_gif:
            images = []
        for i in range(N):
            plt.pause(0.001)
            # Heatmap Computing
            self.planet.temperatures = self.planet.temperatures + dt * self.planet.model_derivative(t)
            I = self.planet.insolation(t)
            # Orbit Computing
            t = t + dt
            radius, angle = self.planet.orbit.orbitalPosition(t)
            Offset = np.array([radius, angle])
            Offset = Offset
            # Updating plot
            # heatmap.set_data(self.planets.temperatures)
            heatmap.set_data(I)
            position.set_offsets(Offset)
            plt.draw()
