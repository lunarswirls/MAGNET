#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Container class of math utility static methods.
"""
# Imports:
import numpy as np
import math
from scipy.spatial.transform import Rotation as R
from magnetpy.utilities.Constant_Enums import SourceVector
from magnetpy.utilities.Constant_Enums import InstrumentParams


def dipole_source(d, theta=None, vector_dir=None):
    """
    Calculate source parameters given distance to place source wrt magnetometer array

    :param float d: distance d to place magnetic source wrt magnetometer array
    :param float theta: (Optional, default zero) rotation in degrees of source vector around z axis
    :param string vector_dir: (Optional, input none) specified unit vector of location or direction
    :return: array of new basis vectors
    """

    vect_dir = SourceVector.getEnum(vector_dir)

    if vect_dir is None and theta is None:
        # Define dipole source at arbitrary vector position with respect to center of bottom of the array.
        r_s_hat = np.array([0.0, 1.0, 0.0])
        r_s_mag = 0.53

        # Define source moment intensity and unit vector direction.
        # 1 nT = 0.0008 A/m
        # LVR: at 20 cm distance, 100 nT arises from M = 0.004 A*m^(2) (4 mA*m^(2)).
        # loop 10 cm on a side carrying 400 mA has this moment.
        # 1.25 x 0.6 x 0.6 cm (0.5 x 0.25 x 0.25 inch) AlNiCo magnet has a moment of 0.42 A*m^(2).
        # Would generate 10,000 nT at 20 cm!

        # mu_s = np.array([0.02, 0.1, 0.003])  # A*m^(2)
        # mu_s_mag = np.linalg.norm(mu_s, ord=2, axis=0)
        # mu_s_hat = mu_s / mu_s_mag

        mu_s_mag = 1.2  # A*m^(2)
        mu_s_hat = np.array([0.0, 1.0, 0.0])

    elif vect_dir is None and theta is not None:
        r_s = np.array([0, -0.47625, -0.06704375])  # meters
        r_s = z_rotation(r_s, -math.radians(theta))
        r_s_mag = np.linalg.norm(r_s, ord=2, axis=0)
        r_s_hat = r_s / r_s_mag

        mu_s_mag = 1.2  # A*m^(2)
        mu_s_hat = np.array([0.0, 1.0, 0.0])

    else:
        r_s_hat = np.fromstring(str(vect_dir.value), dtype=float, count=3, sep=' ')
        r_s_hat = z_rotation(r_s_hat, math.radians(theta))
        r_s_mag = d

        mu_s_hat = np.array([0.0, 0.0, 1.0])
        mu_s_mag = 1.5  # A*m^(-2)

    return mu_s_mag, mu_s_hat, r_s_mag, r_s_hat


def multipole_source(d, theta=None, vector_dir=None):
    """
    Calculate source parameters given distance to place source wrt magnetometer array

    :param float d: distance d to place magnetic source wrt magnetometer array
    :param float theta: (Optional, default zero) rotation in degrees of source vector around z axis
    :param string vector_dir: (Optional, input none) specified unit vector of location or direction
    :return: array of new basis vectors
    """

    vect_dir = SourceVector.getEnum(vector_dir)

    if vect_dir is None and theta is None:
        # Define dipole source at arbitrary vector position with respect to center of bottom of the array.
        r_s_hat_1 = np.array([0.0, 1.0, 0.0])
        r_s_mag_1 = d

        # Define source moment intensity and unit vector direction.
        # 1 nT = 0.0008 A/m
        # LVR: at 20 cm distance, 100 nT arises from M = 0.004 A*m^(2) (4 mA*m^(2)).
        # loop 10 cm on a side carrying 400 mA has this moment.
        # 1.25 x 0.6 x 0.6 cm (0.5 x 0.25 x 0.25 inch) AlNiCo magnet has a moment of 0.42 A*m^(2).
        # Would generate 10,000 nT at 20 cm!

        # mu_s_1 = np.array([0.02, 0.1, 0.003])  # A*m^(2)
        # mu_s_mag_1 = np.linalg.norm(mu_s_1, ord=2, axis=0)
        # mu_s_hat_1 = mu_s_1 / mu_s_mag_1

        mu_s_mag_1 = 1.25  # A*m^(2)
        mu_s_hat_1 = np.array([0.0, 1.0, 0.0])

        r_s_hat_2 = np.array([0.0, 1.0, 0.0])
        r_s_mag_2 = d

        mu_s_mag_2 = 2.5  # A*m^(2)
        mu_s_hat_2 = np.array([0.0, 1.0, 0.0])

    elif vect_dir is None and theta is not None:
        r_s_1 = np.array([0, -0.47625, -0.06704375])  # meters
        r_s_1 = z_rotation(r_s_1, -math.radians(theta))
        r_s_mag_1 = np.linalg.norm(r_s_1, ord=2, axis=0)
        r_s_hat_1 = r_s_1 / r_s_mag_1

        mu_s_mag_1 = 1.25  # A*m^(2)
        mu_s_hat_1 = np.array([0.0, 1.0, 0.0])

        r_s_2 = np.array([0, -0.47625, -0.06704375])  # meters
        r_s_2 = z_rotation(r_s_2, -math.radians(theta))
        r_s_mag_2 = np.linalg.norm(r_s_2, ord=2, axis=0)
        r_s_hat_2 = r_s_2 / r_s_mag_2

        mu_s_mag_2 = 2.5  # A*m^(2)
        mu_s_hat_2 = np.array([0.0, 1.0, 0.0])

    else:
        r_s_hat_1 = np.fromstring(str(vect_dir.value), dtype=float, count=3, sep=' ')
        r_s_hat_1 = z_rotation(r_s_hat_1, math.radians(theta))
        r_s_mag_1 = d

        mu_s_hat_1 = np.array([0.0, 0.0, 1.0])
        mu_s_mag_1 = 1.25  # A*m^(-2)

        r_s_hat_2 = np.fromstring(str(vect_dir.value), dtype=float, count=3, sep=' ')
        r_s_hat_2 = z_rotation(r_s_hat_1, math.radians(theta))
        r_s_mag_2 = d

        mu_s_hat_2 = np.array([0.0, 0.0, 1.0])
        mu_s_mag_2 = 2.5  # A*m^(-2)

    return mu_s_mag_1, mu_s_hat_1, r_s_mag_1, r_s_hat_1, mu_s_mag_2, mu_s_hat_2, r_s_mag_2, r_s_hat_2


def new_vec_basis(vector):
    """
    Find new basis using input vector

    :param vector: initial vector to construct basis of space
    :return: array of new basis vectors
    """
    axis_to_cross = []
    min_re0_val = min(abs(vector))

    if abs(vector[0]) == min_re0_val:
        axis_to_cross = np.array([1, 0, 0])
    elif abs(vector[1]) == min_re0_val:
        axis_to_cross = np.array([0, 1, 0])
    elif abs(vector[2]) == min_re0_val:
        axis_to_cross = np.array([0, 0, 1])

    e1 = np.cross(vector, axis_to_cross)
    e1 = unit_vector(e1)
    e2 = np.cross(vector, e1)
    e2 = unit_vector(e2)

    transmat = np.array([vector, e1, e2])

    # print("re0_hat input: {}".format(vector))
    # print("e1 input: {}".format(e1))
    # print("e2 input: {}".format(e2))

    return transmat


def sampleSphericalCap(coneAngleDegree, coneDir, N):
    """
    Generate points on spherical cap around the north pole
    See https://math.stackexchange.com/a/205589/81266

    :param float coneAngleDegree: cone angle in degrees
    :param coneDir: final direction of vector normal to spherical cap center
    :param int N: number of randomly sampled points on spherical cap
    """

    rnd_sample = np.random.random(size=int(N))
    north_pole = np.array([0, 0, 1])
    coneAngle = np.deg2rad(coneAngleDegree)

    z = rnd_sample * (1 - np.cos(coneAngle)) + np.cos(coneAngle)
    phi = rnd_sample * 2 * np.pi
    x = np.sqrt(1-z**2)*np.cos(phi)
    y = np.sqrt(1-z**2)*np.sin(phi)

    # If input vector is centered at north pole, done
    if (coneDir == north_pole).all():
        final_points = np.array([x, y, z])
    else:
        # fetch rotation matrix from north pole to new pole
        rotate = fetch_rotation(coneDir, north_pole)
        # print(rotate.as_matrix())

        # Rotate points from center around north pole to center around coneDir
        final_points = rotate.apply(np.transpose([x, y, z]))

    prev_hat = coneDir

    return final_points, coneDir


def fetch_magloc(future_vec=None):
    """
    :param future_vec: Optional offset vector, default None
    """
    sense_offset = np.fromstring(str(InstrumentParams.sense_rod_offset.value), dtype=float, count=3, sep=' ')

    # rotate sense rod offset into each sensor frame of reference
    sense_offset1 = z_rotation(sense_offset, math.radians(90))
    sense_offset2 = z_rotation(sense_offset, math.radians(-90))
    sense_offset3 = z_rotation(sense_offset, math.radians(0))
    sense_offset4 = z_rotation(sense_offset, math.radians(180))

    # rotate sense rod offset from sensor frame of reference to mast frame of reference
    sense_offset1_rot = B1_transform(sense_offset1)
    sense_offset2_rot = B2_transform(sense_offset2)
    sense_offset3_rot = B3_transform(sense_offset3)
    sense_offset4_rot = B4_transform(sense_offset4)

    mag1_loc = np.fromstring(str(InstrumentParams.mag1.value), dtype=float, count=3, sep=' ')
    mag2_loc = np.fromstring(str(InstrumentParams.mag2.value), dtype=float, count=3, sep=' ')
    mag3_loc = np.fromstring(str(InstrumentParams.mag3.value), dtype=float, count=3, sep=' ')
    mag4_loc = np.fromstring(str(InstrumentParams.mag4.value), dtype=float, count=3, sep=' ')
    # print(mag4_loc)

    # add sense rod offset to each magnetometer location in mast frame of reference
    mag1_loc = mag1_loc + sense_offset1_rot
    mag2_loc = mag2_loc + sense_offset2_rot
    mag3_loc = mag3_loc + sense_offset3_rot
    mag4_loc = mag4_loc + sense_offset4_rot
    # print(mag4_loc)

    return mag1_loc, mag2_loc, mag3_loc, mag4_loc


def fetch_rotation(new_pole, old_pole):
    """
    :param new_pole: final direction of vector normal to spherical cap center
    :param old_pole: vector normal to be rotated to final direction
    """
    normconeDir = unit_vector(new_pole)

    # Find the rotation axis u and rotation angle rot
    u = np.cross(old_pole, normconeDir)
    u = unit_vector(u)
    print(u)
    rot = math.acos(np.dot(old_pole, normconeDir))
    print(rot)
    rot_vec = u * rot

    rotate = R.from_rotvec(rot_vec)

    return rotate


def cart2sph(x, y, z):
    """
    Converts Cartesian x, y, z to spherical azimuth, elevation, r

    :param float x: cartesian x
    :param float y: cartesian y
    :param float z: cartesian z
    :return: azimuth, elevation, r spherical coordinates
    :rtype: float
    """
    az = np.arctan2(y, x)  # phi
    el = np.arctan2(z, math.sqrt(x**2 + y**2))  # theta
    r = math.sqrt(x**2 + y**2 + z**2)
    return az, el, r


def cart2cyl(x, y, z):
    """
    Converts Cartesian x, y, z to cylindrical rho, phi, z

    :param float x: cartesian x
    :param float y: cartesian y
    :param float z: cartesian z
    :return: rho, phi, z cylindrical coordinates
    :rtype: float
    """
    rho = math.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    z = z
    return rho, phi, z


def sph2cart(az, el, r):
    """
    Converts spherical azimuth, elevation, r to Cartesian x, y, z

    :param float az: spherical azimuth
    :param float el: spherical elevation
    :param float r: spherical r
    :return: x, y, z cartesian coordinates
    :rtype: float
    """
    x = r * np.cos(el) * np.cos(az)
    y = r * np.cos(el) * np.sin(az)
    z = r * np.sin(el)
    return x, y, z


def unit_vector(vector):
    """
    Returns the unit vector of input vector.

    :param vector: Input vector that is converted to unit vector
    :return: Unit vector
    """
    return vector / np.linalg.norm(vector, ord=2, axis=0)


def angle_between(v1, v2, uv=False):
    """
    Finds angle between two vectors.

    :param v1: Vector 1
    :param v2: Vector 2
    :param uv: (Optional, default False) Take unit vector of v1 and v2
    :return: Angle between v1 and v2 in radians
    :rtype: float
    """
    if (uv):
        v1_u = v1
        v2_u = v2
    else:
        v1_u = unit_vector(v1)
        v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def x_rotation(vector, theta):
    """
    Rotates 3-D vector around x-axis.

    :param vector: vector to be rotated
    :param theta: rotation around x-axis in radians
    :return: rotated vector
    """
    # rot = np.array([[1, 0, 0], [0, np.cos(theta), -np.sin(theta)], [0, np.sin(theta), np.cos(theta)]])
    # vector_rot = np.dot(rot, vector)

    r = R.from_euler('x', theta, degrees=False)
    vector_rot = r.apply(vector)

    return vector_rot


def y_rotation(vector, theta):
    """
    Rotates 3-D vector around y-axis.

    :param vector: vector to be rotated
    :param theta: rotation around y-axis in radians
    :return: rotated vector
    """
    # rot = np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]])
    # vector_rot = np.dot(rot, vector)

    r = R.from_euler('y', theta, degrees=False)
    vector_rot = r.apply(vector)

    return vector_rot


def z_rotation(vector, theta):
    """
    Rotates 3-D vector around z-axis.

    :param vector: vector to be rotated
    :param theta: rotation around z-axis in radians
    :return: rotated vector
    """
    # rot = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    # vector_rot = np.dot(rot, vector)

    r = R.from_euler('z', theta, degrees=False)
    vector_rot = r.apply(vector)

    return vector_rot


def B1_transform(vector):
    """
    Rotates 3-component vectors using the B1 transform:
        [x, y, z] → [-y, x, z]

    Works for:
        - a single vector shape (3,)
        - an array of vectors shape (N,3)
    """
    vec = np.asarray(vector)

    # Case 1: single vector (shape (3,))
    if vec.ndim == 1:
        if vec.size != 3:
            raise ValueError("Single vector must have exactly 3 components.")
        return np.array([-vec[1], vec[0], vec[2]])

    # Case 2: array of vectors (shape (N,3))
    if vec.ndim == 2 and vec.shape[1] == 3:
        B_rot = np.empty_like(vec)
        B_rot[:, 0] = -vec[:, 1]
        B_rot[:, 1] =  vec[:, 0]
        B_rot[:, 2] =  vec[:, 2]
        return B_rot

    raise ValueError("Input must be shape (3,) or (N,3).")


def B2_transform(vector):
    """
    Rotates B2 into mast reference frame.

    :param vector: vector to be rotated
    :return: rotated vector
    """
    vec = np.asarray(vector)

    # Case 1: single vector (shape (3,))
    if vec.ndim == 1:
        if vec.size != 3:
            raise ValueError("Single vector must have exactly 3 components.")
        return np.array([vec[1], -vec[0], vec[2]])

    # Case 2: array of vectors (shape (N,3))
    if vec.ndim == 2 and vec.shape[1] == 3:
        B_rot = np.empty_like(vec)
        B_rot[:, 0] = vec[:, 1]
        B_rot[:, 1] = -vec[:, 0]
        B_rot[:, 2] = vec[:, 2]
        return B_rot

    raise ValueError("Input must be shape (3,) or (N,3).")


def B3_transform(vector):
    """
    Rotates B3 into mast reference frame.

    :param vector: vector to be rotated
    :return: rotated vector
    """

    vec = np.asarray(vector)

    # Case 1: single vector (shape (3,))
    if vec.ndim == 1:
        if vec.size != 3:
            raise ValueError("Single vector must have exactly 3 components.")
        return np.array([-vec[0], -vec[1], vec[2]])

    # Case 2: array of vectors (shape (N,3))
    if vec.ndim == 2 and vec.shape[1] == 3:
        B_rot = np.empty_like(vec)
        B_rot[:, 0] = -vec[:, 0]
        B_rot[:, 1] = -vec[:, 1]
        B_rot[:, 2] = vec[:, 2]
        return B_rot

    raise ValueError("Input must be shape (3,) or (N,3).")


def B4_transform(vector):
    """
    Rotates B4 into mast reference frame.

    :param vector: vector to be rotated
    :return: rotated vector
    """
    vec = np.asarray(vector)

    # Case 1: single vector (shape (3,))
    if vec.ndim == 1:
        if vec.size != 3:
            raise ValueError("Single vector must have exactly 3 components.")
        return np.array([-vec[0], vec[1], vec[2]])

    # Case 2: array of vectors (shape (N,3))
    if vec.ndim == 2 and vec.shape[1] == 3:
        B_rot = np.empty_like(vec)
        B_rot[:, 0] = vec[:, 0]
        B_rot[:, 1] = vec[:, 1]
        B_rot[:, 2] = vec[:, 2]
        return B_rot

    raise ValueError("Input must be shape (3,) or (N,3).")


def vec_rmse(v1, v2):
    """
    Calculates the Root Mean Squared Error (RMSE) between two arrays of 3D vectors.

    :param v1: A NumPy array of shape (n, 3).
    :param v2: A NumPy array of shape (n, 3).

    :returns: The RMSE value (float).
    """
    vectors1 = np.array(v1)
    vectors2 = np.array(v2)

    if vectors1.shape != vectors2.shape:
        raise ValueError("Arrays must have the same shape.")

    squared_diffs = (vectors1 - vectors2) ** 2
    mse = np.mean(squared_diffs, axis=1)
    rmse = np.sqrt(mse)
    return rmse


def vec_rel_err(v1, v2):
    """
    Calculates the relative error between two arrays of 3D vectors.
    relative error = abs(true_value - approx_value) / abs(true_value)

    :param v1: A NumPy array of shape (n, 3).
    :param v2: A NumPy array of shape (n, 3).

    :returns: The RMSE value (float).
    """
    vectors1 = np.array(v1)
    vectors2 = np.array(v2)

    if vectors1.shape != vectors2.shape:
        raise ValueError("Arrays must have the same shape.")

    squared_diffs = (vectors1 - vectors2) ** 2
    mse = np.mean(squared_diffs, axis=1)
    rmse = np.sqrt(mse)
    return rmse


def get_first_non_zero_array(list_of_arrays):
    for array in list_of_arrays:
        if any(array):  # Check if any element in the array is non-zero
            return array
    return None
