import numpy as nu
import pandas as pd
from matplotlib import pyplot as pl
from matplotlib import cm
from mpl_toolkits import mplot3d
import multiprocessing as mp
from numpy import abs, sqrt, cos, sin, pi

from scipy.integrate import quadrature, dblquad, tplquad
from scipy.special import ellipk, ellipe, ellipkm1


# Constants

mu0 = 4*nu.pi*1e-7
I = 100


# Model

# def Aphi(I, coilRadius, coilZ, lo, z):
#     squaredK = 4*coilRadius*lo/( (coilRadius+lo)**2 + (z-coilZ)**2 )
#     k = sqrt(squaredK)
#     Aphi =  mu0*I/pi * ( (sqrt((coilRadius+lo)**2+(z-coilZ)**2)/(2*lo) - coilRadius/sqrt((coilRadius+lo)**2+(z-coilZ)**2))*ellipk(squaredK) - ellipe(squaredK) )
#     return Aphi


def Acoil(lo, z, lo_, coilZs, I):
    Aphis = nu.zeros(len(coilZs))
    for i, z_ in enumerate(coilZs):
        squaredK = 4*lo*lo_ / ((lo_+lo)**2 + (z-z_)**2)
        k = sqrt(squaredK)
        Aphis[i] = 1/k * sqrt(lo_/lo) * ( (1-0.5*k**2)*ellipk(squaredK) - ellipe(squaredK) )
    return mu0*I/pi * Aphis.sum()


def _Aphif(z_, lo, z, lom):
    squaredK = 4*lo*lom(z_) / ((lom(z_)+lo)**2 + (z-z_)**2)
    k = sqrt(squaredK)
    return 1/k * sqrt(lom(z_)/lo) * ( (1-0.5*k**2)*ellipk(squaredK) - ellipe(squaredK) )


def Amag(lo, z, lom, Z_L, Z_U, k_phi):
    return mu0*k_phi/pi * quadrature(_Aphif, Z_L, Z_U, args=(lo, z, lom), maxiter=10000)[0]


def lomO(z_, Z0, R, FMThickness):
    return sqrt(R**2 - (z_-(Z0-R/2+FMThickness))**2)


def lomI(z_, Z0, R, FMThickness):
    return sqrt(R**2 - (z_-(Z0-R/2-FMThickness))**2)


if __name__ == '__main__':
    coilRadius = 1.5e-2
    Z0 = coilRadius * 2
    N = 31
    coilZs = nu.linspace(-Z0, Z0, N)
    points = 50
    I = 1.0
    k_phi = 500.0
    FMThickness = 1e-3
    Z_LO = Z0-coilRadius/2+FMThickness
    Z_UO = Z0+coilRadius/2+FMThickness
    Z_LI = Z0-coilRadius/2-FMThickness
    Z_UI = Z0+coilRadius/2-FMThickness

    los = nu.linspace(0.1*coilRadius, 0.9*coilRadius, points)
    zs = nu.linspace(0, 1.5*Z0, points)
    As = nu.zeros((points, points))

    for i, lo in enumerate(los):
        for j, z in enumerate(zs):
            As[i, j] = Acoil(lo, z, coilRadius, coilZs, I) +\
            Amag(lo, z, lambda z_: lomO(z_, Z0, coilRadius, FMThickness), Z_LO, Z_UO, k_phi) +\
            Amag(lo, z, lambda z_: lomI(z_, Z0, coilRadius, FMThickness), Z_LI, Z_UI, -k_phi)
            # As[i, j] = Acoil(lo, z, coilRadius, coilZs, I)
            print(As[i, j])

    _los, _zs = nu.meshgrid(los, zs, indexing='ij')
    pl.contourf(_los, _zs, As, levels=50)
    pl.colorbar()
    samples = nu.linspace(Z_LO, Z_UO, 100)
    pl.scatter(lomO(samples, Z0, coilRadius, FMThickness), samples)
    samples = nu.linspace(Z_LI, Z_UI, 100)
    pl.scatter(lomI(samples, Z0, coilRadius, FMThickness), samples)
    pl.xlim([los.min(), los.max()])
    pl.ylim([zs.min(), zs.max()])
    pl.show()
