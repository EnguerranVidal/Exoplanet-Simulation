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
