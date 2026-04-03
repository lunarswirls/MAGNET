#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Contains static methods for generating thruster profiles.

@Author: Dany Waller
"""
# Imports
import math
import numpy as np
import pandas as pd
import magnetpy.utilities.Math_Utils as MathUtils
import magnetpy.gradiometry_algorithm.tetrahedral_correction as tc

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


def base_profile():
    """
    Generate a base profile of simulated overlapping thrusters
    :return: B1, B2, B3, B4 observed at mag1, mag2, mag3, mag4 in units of Tesla
    """
    # average thruster moments from lab testing
    mu_s1 = np.array([-0.2, 0.3, -0.5]) * 4.1  # gold
    mu_s_mag1 = np.linalg.norm(mu_s1, ord=2, axis=0)  # units [A*m^2]
    mu_s_hat1 = mu_s1 / mu_s_mag1

    # mu_s2 = np.array([-0.2, -0.3, -0.5]) * 4.7  # orange
    mu_s2 = np.array([-0.2, -0.3, -1.5]) * 4.7  # orange
    mu_s_mag2 = np.linalg.norm(mu_s2, ord=2, axis=0)  # units [A*m^2]
    mu_s_hat2 = mu_s2 / mu_s_mag2

    # mu_s3 = np.array([-0.2, -0.3, -0.5]) * 4.3  # red
    mu_s3 = np.array([-0.2, -0.3, -1.5]) * 4.7  # orange
    mu_s_mag3 = np.linalg.norm(mu_s3, ord=2, axis=0)  # units [A*m^2]
    mu_s_hat3 = mu_s3 / mu_s_mag3

    mu_s4 = np.array([-0.2, 0.3, -0.5]) * 4.2  # maroon
    mu_s_mag4 = np.linalg.norm(mu_s4, ord=2, axis=0)
    mu_s_hat4 = mu_s4 / mu_s_mag4

    # thruster 1 location
    r_s = np.array([0.75, 0.5, 1.0])
    r_s_mag = np.linalg.norm(r_s, ord=2, axis=0)  # units [m]
    r_s_hat = r_s / r_s_mag
    r_s1_hat = r_s_hat
    # create 3 more thruster locations
    r_s2_hat = MathUtils.z_rotation(r_s_hat, -math.radians(90))
    r_s3_hat = MathUtils.z_rotation(r_s_hat, -math.radians(180))
    r_s4_hat = MathUtils.z_rotation(r_s_hat, -math.radians(270))

    # get magnetic field from thruster 1
    [B1_s1, B2_s1, B3_s1, B4_s1] = tc.mag_array(mu_s_mag1, mu_s_hat1, r_s_mag, r_s1_hat)  # units Tesla
    # get magnetic field from thruster 2
    [B1_s2, B2_s2, B3_s2, B4_s2] = tc.mag_array(mu_s_mag2, mu_s_hat2, r_s_mag, r_s2_hat)  # units Tesla
    # get magnetic field from thruster 3
    [B1_s3, B2_s3, B3_s3, B4_s3] = tc.mag_array(mu_s_mag3, mu_s_hat3, r_s_mag, r_s3_hat)  # units Tesla
    # get magnetic field from thruster 4
    [B1_s4, B2_s4, B3_s4, B4_s4] = tc.mag_array(mu_s_mag4, mu_s_hat4, r_s_mag, r_s4_hat)  # units Tesla

    return B1_s1, B2_s1, B3_s1, B4_s1, B1_s2, B2_s2, B3_s2, B4_s2, B1_s3, B2_s3, B3_s3, B4_s3, B1_s4, B2_s4, B3_s4, B4_s4


def generate_thruster_bfield_data(
    time_arr, times, bg_field_nT, bg_mag,
    n_events=25, min_dur=30, max_dur=1000,
    overlap_prob=0.25,
    seed=75, do_plots=False):
    """
    Generate random overlapping thruster noise using base_profile().

    Parameters
    ----------
    time_arr : np.ndarray
        Array containing times in raw seconds.
    times : np.ndarray
        Array containing times in UTC.
    bg_field_nT : np.ndarray
        Background field in nanoTesla (nT), shape (N, 3).
    bg_mag : np.ndarray
        Background field magnitude, shape (N,).
    n_events : int
        Number of thruster activations to simulate.
    min_dur : int
        Minimum duration of thruster activations.
    max_dur : int
        Maximum duration of thruster activations.
    overlap_prob : float
        Probability that multiple thrusters overlap.
    seed : int
        Random seed.
    do_plots : bool
        If True, generate plots.

    Returns
    -------
    bfield_df : pd.DataFrame
        DataFrame containing B1–B4 fields, background, and time columns.
    thruster_df : pd.DataFrame
        DataFrame tracking thruster state (1 = on, 0 = off).
    """

    rng = np.random.default_rng(seed)
    print("Seed:\t", seed)

    # --- Load thruster field profiles from base_profile() ---
    ( B1_s1, B2_s1, B3_s1, B4_s1,
        B1_s2, B2_s2, B3_s2, B4_s2,
        B1_s3, B2_s3, B3_s3, B4_s3,
        B1_s4, B2_s4, B3_s4, B4_s4
    ) = base_profile()

    # Pack into dict for convenience: (sensor_id, thruster_id)
    B_signals = {
        (1, 1): B1_s1, (2, 1): B2_s1, (3, 1): B3_s1, (4, 1): B4_s1,
        (1, 2): B1_s2, (2, 2): B2_s2, (3, 2): B3_s2, (4, 2): B4_s2,
        (1, 3): B1_s3, (2, 3): B2_s3, (3, 3): B3_s3, (4, 3): B4_s3,
        (1, 4): B1_s4, (2, 4): B2_s4, (3, 4): B3_s4, (4, 4): B4_s4
    }

    n_thrusters = 4
    n_points = len(time_arr)

    bg_field = bg_field_nT.copy() * 1e-9  # units T

    # --- Initialize arrays ---
    thruster_state = np.zeros((n_points, n_thrusters), dtype=int)
    b_fields = {f"b{i + 1}_field": bg_field.copy() for i in range(4)}  # units T

    # --- Generate thruster activations ---
    for _ in range(n_events):
        n_active = rng.integers(1, n_thrusters + 1)
        active_thrusters = rng.choice(range(1, n_thrusters + 1), size=n_active, replace=False)

        for thruster_id in active_thrusters:
            dur = rng.integers(min_dur, max_dur)
            start = rng.integers(0, max(1, n_points - dur))
            end = min(start + dur, n_points)

            for sensor_id in range(1, 5):
                # Each signal is a 3-vector → broadcast across the slice
                vec = B_signals[(sensor_id, thruster_id)]
                b_fields[f"b{sensor_id}_field"][start:end] += vec

            thruster_state[start:end, thruster_id - 1] = 1

        # Optional overlapping firings
        if rng.random() < overlap_prob:
            n_extra = rng.integers(1, n_thrusters + 1)
            overlap_thrusters = rng.choice(range(1, n_thrusters + 1), size=n_extra, replace=False)

            for thruster_id in overlap_thrusters:
                dur = rng.integers(min_dur, max_dur)
                overlap_start = rng.integers(0, max(1, n_points - dur))
                overlap_end = min(overlap_start + dur, n_points)

                for sensor_id in range(1, 5):
                    # print("Sensor id:", str(sensor_id))
                    # print("Thruster id:", str(thruster_id))
                    vec = B_signals[(sensor_id, thruster_id)]
                    # print("Signal:", str(vec))
                    b_fields[f"b{sensor_id}_field"][overlap_start:overlap_end] += vec

                thruster_state[overlap_start:overlap_end, thruster_id - 1] = 1

    # Build B-field DataFrame
    b_df = pd.DataFrame(b_fields["b1_field"], columns=['B1X', 'B1Y', 'B1Z'])  # units T

    for i in range(2, 5):
        b_df[f'B{i}X'] = b_fields[f"b{i}_field"][:, 0]  # units T
        b_df[f'B{i}Y'] = b_fields[f"b{i}_field"][:, 1]  # units T
        b_df[f'B{i}Z'] = b_fields[f"b{i}_field"][:, 2]  # units T

    # print(b_df.head(10))
    # print(np.where(b_fields["b1_field"]-b_fields["b2_field"] > 0))

    b_df['BGX'] = bg_field_nT[:, 0]  # units nT
    b_df['BGY'] = bg_field_nT[:, 1]  # units nT
    b_df['BGZ'] = bg_field_nT[:, 2]  # units nT
    b_df['BG_mag'] = bg_mag  # units nT
    b_df['time_raw'] = time_arr
    b_df['time_fmt'] = times

    # --- Thruster state DataFrame ---
    t_df = pd.DataFrame(thruster_state, columns=[f"Thruster {i + 1}" for i in range(n_thrusters)])
    t_df['Time'] = time_arr

    if do_plots:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 1, figsize=(8, 4))
        t_df.plot(x="Time", y=["Thruster 1", "Thruster 2", "Thruster 3"], ax=ax, linewidth=2, style=['-','--','-.',':'])
        plt.xlabel("Time (s)", fontsize=16)
        plt.ylabel("State", fontsize=16)
        plt.legend(ncol=1, loc='center left', bbox_to_anchor=(1, 0.5), fancybox=True)
        plt.show()

        fig, ax = plt.subplots(1, 1, figsize=(8, 4))
        b_df.plot(x="time_raw", y=["BGX", "BGY", "BGZ"], ax=ax, linewidth=2, style=['-','--','-.',':'])
        plt.xlabel("Time (s)", fontsize=16)
        plt.ylabel("nT", fontsize=16)
        plt.legend(ncol=1, loc='center left', bbox_to_anchor=(1, 0.5), fancybox=True)
        plt.show()

    return b_df, t_df
