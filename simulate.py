#All numbers in SI units, without prefixes.
from numpy import *
from scipy.special import erf #Error function
from scipy.stats import truncexpon #Electron absorption profile
from scipy.constants import e, k #Electron charge and Boltzmann constant

#Other constants
permittivity = 1.07e-10 #Permittivity of silicon
resistivity = 30. #Resistivity of silicon, in ohm-m
saturation_velocity = 1.18e5 #Maximal electron velocity in silicon
attenuation_length = 2.78e-5 #Attenuation length for an x-ray
number_dopants = 1. / (e * resistivity * 0.048)

def el_model(size, voltage=35., temp=200., pixel_width = 1.35e-5, thickness=1e-4):
    low_field_mobility = 61204.9 * temp**(-2.26)
    depletion_voltage = e * number_dopants * square(thickness) / (2 * permittivity)
    depleted_thickness = sqrt(2 * permittivity * voltage / (e * number_dopants))
    undepleted_thickness = maximum(thickness - depleted_thickness, 0)
    voltage_ratio = voltage / depletion_voltage
    field_start = (1 - voltage_ratio) * thickness / 2
    def drift_time(conversion_depth):
        """Integral from conversion_depth to thickness of 1/velocity."""
        effective_depth = maximum(field_start, conversion_depth)
        #Begin black box - trust me, you don't want to know how this works!
        return ((thickness - effective_depth) / saturation_velocity + square(thickness) *
            log((1 + voltage_ratio) / (2 * effective_depth / thickness + voltage_ratio -
            1)) / (2 * low_field_mobility * depletion_voltage) + maximum((field_start -
            conversion_depth) * saturation_velocity, 0))
        #End black box
    def diffusion_width(conversion_depth): #Return value in PIXELS!!!
        """Combination of drift from depleted and undepleted regions."""
        return sqrt((drift_time(maximum(conversion_depth, undepleted_thickness)) *
                     2 * k * temp * low_field_mobility / e) + #depleted
                    where(conversion_depth < undepleted_thickness,
                          square(undepleted_thickness), 0)) / pixel_width #undepleted
    def absorption_profile(size): #Exponential random absorption profile
        return truncexpon.rvs(thickness/attenuation_length,
                              scale=attenuation_length, size=size)
    return diffusion_width(absorption_profile(size))


def qrat_model(size, sigma_max=1, h_bf=1e-5, thickness=1e-4): # fully-depleted only
    def diffusion_width(conversion_depth):
        return where(thickness-conversion_depth < h_bf, 0,
                     sigma_max * sqrt((thickness-h_bf-conversion_depth) /
                                      (thickness-h_bf)))
    def absorption_profile(size): #Exponential random absorption profile
        return truncexpon.rvs(thickness/attenuation_length,
                              scale=attenuation_length, size=size)
    return diffusion_width(absorption_profile(size))
