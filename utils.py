from scipy.constants import electron_mass as m_e, elementary_charge as e, speed_of_light as c
from math import pi


def vl(B):
    return (e*B)/(2*pi*m_e*c)


def vsp(d, z, g, vl):
    return (d/(1+z)) * ((g**2)*vl)


def vsscp(d, z, g, vl):
    return (d/(1+z)) * ((g**4)*vl)
