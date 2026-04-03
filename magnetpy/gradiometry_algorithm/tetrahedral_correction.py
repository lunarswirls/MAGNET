#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Source fitting gradiometry algorithm.
"""
# Imports:
import time
import math
import numpy as np
import scipy as sp
import magnetpy.utilities.Math_Utils as MathUtils
from magnetpy.utilities.Constant_Enums import PhysicsConstants

# Magnetometers 1, 2, 3 and 4.
# Measurements are B1, B2, B3 and B4 (Bx1, By1, Bz1, etc.).
# Sensors are separated horizontally by distance "d" along X and Y
# and vertically by distance h=d/sqrt(2) along Z, i.e.,
# 1 and 2 at Z = 0
# 3 and 4 at Z = d/sqrt(2)
# 1 and 2 at -d/2 and +d/2 along X
# 3 and 4 at -d/2 and +d/2 along Y

mag1_loc, mag2_loc, mag3_loc, mag4_loc = MathUtils.fetch_magloc()

mu0 = float(PhysicsConstants.mu_0)


def array_correction(B1, B2, B3, B4, mkplots=False):
    """
    Calculates best fit source parameters given observed magnetic fields from tetrahedral mag array

    :param np.array B1: observed field at magnetometer 1 (Tesla)
    :param np.array B2: observed field at magnetometer 2 (Tesla)
    :param np.array B3: observed field at magnetometer 3 (Tesla)
    :param np.array B4: observed field at magnetometer 4 (Tesla)
    :param mkplots: Boolean to make cost function value plots, default = False
    :return:
    """

    start_time = time.time()

    # Define Bd as the field at the center of the tetrahedron. Eq. 17.
    Bd = 0.25 * (B1 + B2 + B3 + B4)

    # Calculate magnitudes
    Bd_mag = np.linalg.norm(Bd, ord=2, axis=0)
    B1_mag = np.linalg.norm(B1, ord=2, axis=0)
    B2_mag = np.linalg.norm(B2, ord=2, axis=0)
    B3_mag = np.linalg.norm(B3, ord=2, axis=0)
    B4_mag = np.linalg.norm(B4, ord=2, axis=0)
    # print(f"Bd mag:{Bd_mag}")

    dx = mag2_loc[0] - mag1_loc[0]
    dy = mag4_loc[1] - mag3_loc[1]
    dh = (1 / math.sqrt(2))*((mag3_loc[2]+mag4_loc[2]) - (mag1_loc[2]+mag2_loc[2]))
    # print(f"dx:{dx}, dy:{dy}, dh:{dh}")

    # Now take gradient of the squared field. Eqs. 18a, 18b and 18c.
    dBsq_x = (B2_mag**2 - B1_mag**2) / dx
    dBsq_y = (B4_mag**2 - B3_mag**2) / dy
    dBsq_z = (1 / 2)*((B4_mag**2 + B3_mag**2) - (B2_mag**2 + B1_mag**2)) / dh
    dBsq = np.array([dBsq_x, dBsq_y, dBsq_z])
    # print(f"dBsq_x:{dBsq_x}, dBsq_y:{dBsq_y}, dBsq_z:{dBsq_z}")

    dBsq_mag = np.linalg.norm(dBsq, ord=2, axis=0)

    # Initial estimate for source (re0) direction. Eq. 11.
    re0_hat = dBsq/dBsq_mag

    # Find basis to rotate into source direction coordinates
    transmat = MathUtils.new_vec_basis(re0_hat)

    # Project Bd on new coordinate system.
    B_re = np.dot(Bd, transmat[0])
    B_1 = np.dot(Bd, transmat[1])
    B_2 = np.dot(Bd, transmat[2])

    # First estimate of direction of mu in new coordinate system. Eq. 12.
    mue0re = 0.5 * B_re / math.sqrt(0.25 * B_re**2 + B_1**2 + B_2**2)
    mue01 = -B_1 / math.sqrt(0.25 * B_re**2 + B_1**2 + B_2**2)
    mue02 = -B_2 / math.sqrt(0.25 * B_re**2 + B_1**2 + B_2**2)
    mue0_hat = np.array([mue0re, mue01, mue02])

    # Go back to original coordinate system
    # linalg.solve(a,b) computes the “exact” solution, x, of the well-determined,
    # i.e., full rank, linear matrix equation ax = b
    mue0_hat = np.linalg.solve(transmat, mue0_hat)

    # Eq.13.
    V_Be = (3 * (np.dot(mue0_hat, re0_hat)) * re0_hat - mue0_hat)
    # Eq.14.
    V_De = (np.dot(mue0_hat, re0_hat) * mue0_hat - re0_hat - 4 * ((np.dot(mue0_hat, re0_hat))**2) * re0_hat)

    V_Be_mag = np.linalg.norm(V_Be, ord=2, axis=0)
    V_De_mag = np.linalg.norm(V_De, ord=2, axis=0)

    # Initial estimate of source distance magnitude.Eq.15.
    re0_mag = 6 * (V_De_mag / (V_Be_mag**2)) * ((Bd_mag**2) / dBsq_mag)
    # print(re0_mag)

    # Initial estimate of source moment magnitude. Eq.16.
    mue0_mag = (4 * math.pi / mu0) * ((Bd_mag * re0_mag**3) / V_Be_mag)
    # print(mue0_mag)

    # Initial estimate of source position.
    re0 = re0_mag * re0_hat
    # Initial estimate of source moment.
    mue0 = mue0_mag * mue0_hat

    # print(f"Initial source position est:{re0}")
    # print(f"Initial source moment est:{mue0}")

    # convert initial estimate into spherical coordinates
    [azimuthre0, elevationre0, _] = MathUtils.cart2sph(re0_hat[0], re0_hat[1], re0_hat[2])
    [azimuthmue0, elevationmue0, _] = MathUtils.cart2sph(mue0_hat[0], mue0_hat[1], mue0_hat[2])

    # set scale factors for magnitude parameters
    f_re0_mag = 1.0
    f_mue0_mag = 1.0

    # assemble cost function input array
    x0 = np.array([azimuthre0, elevationre0, azimuthmue0, elevationmue0, f_mue0_mag, f_re0_mag])

    # get initial estimate cost function
    cf_e0_val = cost_function(x0, mue0_mag, re0_mag, B1, B2, B3, B4)

    # adaptive amoeba method algorithm with no bounds
    allestimates = sp.optimize.minimize(cost_function, x0=x0, args=(mue0_mag, re0_mag, B1, B2, B3, B4),
                                        method='Nelder-Mead', options={'return_all': True, 'adaptive': True,
                                                                       'xatol': 1e-50, 'fatol': 1e-50,
                                                                       'maxfev': 25000, 'maxiter': 25000})

    all_results = allestimates.allvecs
    final_results = allestimates.x
    nsteps = allestimates.nit
    final_cf_val = allestimates.fun

    # split result array into components
    new_re0_az = final_results[0]
    new_re0_el = final_results[1]
    new_mue0_az = final_results[2]
    new_mue0_el = final_results[3]
    new_mue1_mag = final_results[4]*mue0_mag
    new_re1_mag = final_results[5]*re0_mag

    # convert results back to cartesian coordinates
    [muhat1, muhat2, muhat3] = MathUtils.sph2cart(new_mue0_az, new_mue0_el, 1)
    [rhat1, rhat2, rhat3] = MathUtils.sph2cart(new_re0_az, new_re0_el, 1)
    new_mue1_hat = np.array([muhat1, muhat2, muhat3])
    new_re1_hat = np.array([rhat1, rhat2, rhat3])

    if mkplots:
        import magnetpy.utilities.Plot_Utils as PlotUtils

        time_taken = time.time() - start_time
        # print("Time taken: " + str(time_taken))
        cf_vals = cost_func_evaluation(all_results, mue0_mag, re0_mag, B1, B2, B3, B4)
        PlotUtils.cost_func_plot(cf_vals, cf_e0_val, nsteps, final_cf_val, time_taken)

    return mue0_hat, re0_hat, mue0_mag, re0_mag, new_mue1_hat, new_re1_hat, new_mue1_mag, new_re1_mag, final_cf_val, cf_e0_val, nsteps


def mag_array(mu_mag, mu_hat, r_mag, r_hat):
    """
    Calculates magnetic field at magnetometer array

    :param float mu_mag: source moment magnitude (A*m^2)
    :param mu_hat: unit vector source moment direction
    :param float r_mag: source distance magnitude (meters)
    :param r_hat: unit vector source location wrt center of bottom of array
    :return: B1, B2, B3, B4 observed at mag1, mag2, mag3, mag4 in units of Tesla
    :rtype: float
    """
    # Location of source wrt each magnetometer
    r1 = mag1_loc - (r_mag * r_hat)
    r2 = mag2_loc - (r_mag * r_hat)
    r3 = mag3_loc - (r_mag * r_hat)
    r4 = mag4_loc - (r_mag * r_hat)

    # Calculate magnitudes (2-norm) of r
    r1_mag = np.linalg.norm(r1, ord=2, axis=0)
    r2_mag = np.linalg.norm(r2, ord=2, axis=0)
    r3_mag = np.linalg.norm(r3, ord=2, axis=0)
    r4_mag = np.linalg.norm(r4, ord=2, axis=0)

    # Now unit vectors
    r1_hat = r1/r1_mag
    r2_hat = r2/r2_mag
    r3_hat = r3/r3_mag
    r4_hat = r4/r4_mag

    # Calculate mu from input mu_mag and mu_hat
    mu = mu_mag*mu_hat

    # Calculate dipole field at each magnetometer
    B1 = (mu0 / (4 * math.pi)) * ((3 * np.dot(mu, r1_hat) * r1_hat - mu)/(r1_mag**3))
    B2 = (mu0 / (4 * math.pi)) * ((3 * np.dot(mu, r2_hat) * r2_hat - mu)/(r2_mag**3))
    B3 = (mu0 / (4 * math.pi)) * ((3 * np.dot(mu, r3_hat) * r3_hat - mu)/(r3_mag**3))
    B4 = (mu0 / (4 * math.pi)) * ((3 * np.dot(mu, r4_hat) * r4_hat - mu)/(r4_mag**3))

    return B1, B2, B3, B4


def cost_function(test, mu_mag, r_mag, B1, B2, B3, B4):
    """
    Calculates cost function of magnetic field estimate vs observation.

    :param test: array of input parameters for field estimate
    :param float mu_mag: source moment magnitude estimate
    :param float r_mag: source distance magnitude estimate
    :param float B1: observed field at magnetometer 1
    :param float B2: observed field at magnetometer 2
    :param float B3: observed field at magnetometer 3
    :param float B4: observed field at magnetometer 4
    :return: del_sq cost function value
    :rtype: float
    """

    # split input array into components
    r_az = test[0]
    r_el = test[1]
    mu_az = test[2]
    mu_el = test[3]
    f_mue_mag = test[4]
    f_re_mag = test[5]

    re_mag = f_re_mag*r_mag
    mue_mag = f_mue_mag*mu_mag

    # convert spherical coordinates to cartesian coordinates
    [rhat1, rhat2, rhat3] = MathUtils.sph2cart(r_az, r_el, 1)
    re_hat = np.array([rhat1, rhat2, rhat3])
    [muhat1, muhat2, muhat3] = MathUtils.sph2cart(mu_az, mu_el, 1)
    mue_hat = np.array([muhat1, muhat2, muhat3])

    # fetch magnetic field estimates based on input values
    [B1_e, B2_e, B3_e, B4_e] = mag_array(mue_mag, mue_hat, re_mag, re_hat)

    # Difference between B{1, 2, 3, 4} observations and B{1, 2, 3, 4}_e estimates
    B1_diff = B1 - B1_e
    B2_diff = B2 - B2_e
    B3_diff = B3 - B3_e
    B4_diff = B4 - B4_e

    # take 2-norm of row
    B1_diff_mag = np.linalg.norm(B1_diff, ord=2, axis=0)
    B2_diff_mag = np.linalg.norm(B2_diff, ord=2, axis=0)
    B3_diff_mag = np.linalg.norm(B3_diff, ord=2, axis=0)
    B4_diff_mag = np.linalg.norm(B4_diff, ord=2, axis=0)

    # take sum of squares of difference magnitudes for all 4 magnetometers
    del_sq = B1_diff_mag**2 + B2_diff_mag**2 + B3_diff_mag**2 + B4_diff_mag**2

    return del_sq


def cost_func_evaluation(all_results, mu_mag, r_mag, B1, B2, B3, B4):
    """
    Evaluates cost function for all steps of simplex algorithm

    :param all_results: array of input parameters for all field estimates
    :param float mu_mag: source moment magnitude estimate
    :param float r_mag: source distance magnitude estimate
    :param float B1: observed field at magnetometer 1
    :param float B2: observed field at magnetometer 2
    :param float B3: observed field at magnetometer 3
    :param float B4: observed field at magnetometer 4
    :return: list of cost function values for all steps of simplex algorithm
    """
    cost_func_vals = []

    for result in all_results:
        cost_func_eval = cost_function(result, mu_mag, r_mag, B1, B2, B3, B4)
        cost_func_vals.append(cost_func_eval)

    return cost_func_vals
