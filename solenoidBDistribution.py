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

def Acoil(lo, z, lo_, coilZs, I):
    Aphis = nu.zeros(len(coilZs))
    for i, z_ in enumerate(coilZs):
        squaredK = 4*lo*lo_ / ((lo_+lo)**2 + (z-z_)**2)
        k = sqrt(squaredK)
        Aphis[i] = 1/k * sqrt(lo_/lo) * ( (1-0.5*k**2)*ellipk(squaredK) - ellipe(squaredK) )
    return mu0*I/pi * Aphis.sum()


def ellipk_dk(squaredK):
    k = sqrt(squaredK)
    return ellipe(squaredK)/(k*(1-k**2)) - ellipk(squaredK)/k


def ellipe_dk(squaredK):
    k = sqrt(squaredK)
    return (ellipe(squaredK) - ellipk(squaredK)) / k


def _Bmag_lo(z_, lo, z, lom):
    lo_ = lom(z_)
    beta = (lo_+lo)**2 + (z-z_)**2
    beta_r = sqrt(beta)
    squaredK = 4*lo*lo_ / beta
    k = sqrt(squaredK)
    dk_dz = -sqrt(4*lo*lo_) * beta**(-1.5) * (z-z_)
    return ( (z-z_)/beta_r + 2*lo_*lo*(z-z_)/beta**(-1.5) )*ellipk(squaredK) +\
    ( beta_r - 2*lo*lo_/beta_r )*ellipk_dk(squaredK)*dk_dz -\
    ( (z-z_)/beta_r )*ellipe(squaredK) -\
    ( beta_r )*ellipe_dk(squaredK)*dk_dz


def _Bmag_z(z_, lo, z, lom):
    lo_ = lom(z_)
    beta = (lo_+lo)**2 + (z-z_)**2
    beta_r = sqrt(beta)
    squaredK = 4*lo*lo_ / beta
    k = sqrt(squaredK)
    dk_dlo = -sqrt(lo_/lo/beta) - 2*sqrt(lo_*lo)*(lo+lo_)*beta**(-1.5)
    return ( (lo_+lo)/beta_r - 2*lo_/beta_r + 2*lo*lo_*(lo+lo_)/beta**(-1.5) )*ellipk(squaredK) +\
    ( beta_r - 2*lo*lo_/beta_r )*ellipk_dk(squaredK)*dk_dlo -\
    ( (lo+lo_)/beta_r )*ellipe(squaredK) -\
    ( beta_r )*ellipe_dk(squaredK)*dk_dlo


def Bmag(lo, z, lom, Z_L, Z_U, k_phi):
    B_lo = -mu0*k_phi/(2*pi*lo) * quadrature(_Bmag_lo, Z_L, Z_U, args=(lo, z, lom), maxiter=10000)[0]
    B_z = mu0*k_phi/(2*pi*lo) * quadrature(_Bmag_z, Z_L, Z_U, args=(lo, z, lom), maxiter=10000)[0]
    return nu.array([B_lo, B_z])


def Bcoil(lo, z, lo_, coilZs, I):
    Bs_lo = 0
    Bs_z = 0
    lom = lambda x: lo_
    for i, z_ in enumerate(coilZs):
        Bs_lo += -1.0 * _Bmag_lo(z_, lo, z, lom)
        Bs_z += 1.0 * _Bmag_z(z_, lo, z, lom)
    return mu0*k_phi/(2*pi*lo) * nu.array([Bs_lo, Bs_z])


def lomO(z_, Z0, R, FMThickness):
    return sqrt(R**2 - (z_-(Z0-R/2+FMThickness))**2)


def lomI(z_, Z0, R, FMThickness):
    return sqrt(R**2 - (z_-(Z0-R/2-FMThickness))**2)

def Ball(lo, z, coilRadius, coilZs, FMThickness, Z_LO, Z_UO, Z_LI, Z_UI, I, k_phi):
    return Bcoil(lo, z, coilRadius, coilZs, I) +\
    Bmag(lo, z, lambda z_: lomO(z_, Z0, coilRadius, FMThickness), Z_LO, Z_UO, k_phi) +\
    Bmag(lo, z, lambda z_: lomI(z_, Z0, coilRadius, FMThickness), Z_LI, Z_UI, -k_phi)


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
    bs_lo = nu.zeros((points, points))
    bs_z = nu.zeros((points, points))

    # for i, lo in enumerate(los):
    #     for j, z in enumerate(zs):
    #         bp = Bcoil(lo, z, coilRadius, coilZs, I) +\
    #         Bmag(lo, z, lambda z_: lomO(z_, Z0, coilRadius, FMThickness), Z_LO, Z_UO, k_phi) +\
    #         Bmag(lo, z, lambda z_: lomI(z_, Z0, coilRadius, FMThickness), Z_LI, Z_UI, -k_phi)
    #         print(Bp)
    #         bs_lo[i, j] = bp[0]
    #         bs_z[i, j] = bp[1]

    args = []
    for i, lo in enumerate(los):
        for j, z in enumerate(zs):
            args.append((lo, z, coilRadius, coilZs, FMThickness, Z_LO, Z_UO, Z_LI, Z_UI, I, k_phi))

    # with mp.Pool(processes=min(mp.cpu_count()-1, 50)) as pool:
    with mp.Pool(processes=mp.cpu_count()*3//4) as pool:
        bs = pool.starmap(Ball, args)

    for i, lo in enumerate(los):
        for j, z in enumerate(zs):
            bp = bs[i*len(zs) + j]
            bs_lo[i, j] = bp[0]
            bs_z[i, j] = bp[1]

    _los, _zs = nu.meshgrid(los, zs, indexing='ij')
    pl.contourf(_los, _zs, bs_lo, levels=50)
    pl.colorbar()
    # samples = nu.linspace(Z_LO, Z_UO, 100)
    # pl.scatter(lomO(samples, Z0, coilRadius, FMThickness), samples)
    # samples = nu.linspace(Z_LI, Z_UI, 100)
    # pl.scatter(lomI(samples, Z0, coilRadius, FMThickness), samples)
    # pl.xlim([los.min(), los.max()])
    # pl.ylim([zs.min(), zs.max()])
    pl.show()

    _los, _zs = nu.meshgrid(los, zs, indexing='ij')
    pl.contourf(_los, _zs, bs_z, levels=50)
    pl.colorbar()
    pl.show()

    pl.quiver(_los/coilRadius, _zs/Z0, bs_lo, bs_z)
    pl.show()
