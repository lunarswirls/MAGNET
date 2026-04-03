#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Noise detection for gradiometry algorithm.
"""
import numpy as np
import matplotlib.pyplot as plt


def noise_detector(time_arr, b1_field, b2_field, b3_field, b4_field, db_dt_tol=1.5e-9, plot_noise_der=False):
    """
    Detects noise given input time series vector field from 4 magnetometer sensors

    :param time_arr: Array of time stamps associated wth observations from each sensor
    :param b1_field: Time series vector field observations from magnetometer #1
    :param b2_field: Time series vector field observations from magnetometer #2
    :param b3_field: Time series vector field observations from magnetometer #3
    :param b4_field: Time series vector field observations from magnetometer #4
    :param db_dt_tol: Tolerance for noise detection, default 1.5 nT
    :param plot_noise_der: Boolean to plot noise derivation
    :return: Indices of potential noise source(s) in time series vector fields from 4 sensors
    """

    b1_mag = np.linalg.norm(b1_field, ord=2, axis=1)
    b1_std = np.std(b1_mag)
    if b1_std > db_dt_tol:
        b1_threshold = b1_std
    else:
        b1_threshold = db_dt_tol

    b2_mag = np.linalg.norm(b2_field, ord=2, axis=1)
    b2_std = np.std(b2_mag)
    if b2_std > db_dt_tol:
        b2_threshold = b2_std
    else:
        b2_threshold = db_dt_tol

    b3_mag = np.linalg.norm(b3_field, ord=2, axis=1)
    b3_std = np.std(b3_mag)
    if b3_std > db_dt_tol:
        b3_threshold = b3_std
    else:
        b3_threshold = db_dt_tol

    b4_mag = np.linalg.norm(b4_field, ord=2, axis=1)
    b4_std = np.std(b4_mag)
    if b4_std > db_dt_tol:
        b4_threshold = b4_std
    else:
        b4_threshold = db_dt_tol

    # is the spike positive or negative
    step_ind_b1 = []
    step_dir_b1 = []
    step_ind_b2 = []
    step_dir_b2 = []
    step_ind_b3 = []
    step_dir_b3 = []
    step_ind_b4 = []
    step_dir_b4 = []

    for i in range(1, len(b1_mag)):
        if b1_mag[i] > b1_mag[i - 1] and abs(b1_mag[i] - b1_mag[i - 1]) > db_dt_tol:
            # step up
            step_ind_b1.append(i)
            step_dir_b1.append(1)
        elif b1_mag[i] < b1_mag[i - 1] and abs(b1_mag[i] - b1_mag[i - 1]) > db_dt_tol:
            # step down
            step_ind_b1.append(i)
            step_dir_b1.append(-1)

    for i in range(1, len(b2_mag)):
        if b2_mag[i] > b2_mag[i - 1] and abs(b2_mag[i] - b2_mag[i - 1]) > db_dt_tol:
            # step up
            step_ind_b2.append(i)
            step_dir_b2.append(1)
        elif b2_mag[i] < b2_mag[i - 1] and abs(b2_mag[i] - b2_mag[i - 1]) > db_dt_tol:
            # step down
            step_ind_b2.append(i)
            step_dir_b2.append(-1)

    for i in range(1, len(b3_mag)):
        if b3_mag[i] > b3_mag[i - 1] and abs(b3_mag[i] - b3_mag[i - 1]) > db_dt_tol:
            # step up
            step_ind_b3.append(i)
            step_dir_b3.append(1)
        elif b3_mag[i] < b3_mag[i - 1] and abs(b3_mag[i] - b3_mag[i - 1]) > db_dt_tol:
            # step down
            step_ind_b3.append(i)
            step_dir_b3.append(-1)

    for i in range(1, len(b4_mag)):
        if b4_mag[i] > b4_mag[i - 1] and abs(b4_mag[i] - b4_mag[i - 1]) > db_dt_tol:
            # step up
            step_ind_b4.append(i)
            step_dir_b4.append(1)
        elif b4_mag[i] < b4_mag[i - 1] and abs(b4_mag[i] - b4_mag[i - 1]) > db_dt_tol:
            # step down
            step_ind_b4.append(i)
            step_dir_b4.append(-1)

    signed_ind_b1 = np.multiply(step_ind_b1, step_dir_b1)
    signed_ind_b2 = np.multiply(step_ind_b2, step_dir_b2)
    signed_ind_b3 = np.multiply(step_ind_b3, step_dir_b3)
    signed_ind_b4 = np.multiply(step_ind_b4, step_dir_b4)

    def remove_consecutive_signs(lst):
        result = []
        prev_sign = None

        for m in np.arange(len(lst)):
            num = lst[m]
            sign = 1 if num >= 0 else -1

            if sign != prev_sign:
                result.append(num)
                prev_sign = sign

        return result

    signed_ind_b1 = remove_consecutive_signs(signed_ind_b1)
    signed_ind_b2 = remove_consecutive_signs(signed_ind_b2)
    signed_ind_b3 = remove_consecutive_signs(signed_ind_b3)
    signed_ind_b4 = remove_consecutive_signs(signed_ind_b4)

    all_edges = [len(signed_ind_b1), len(signed_ind_b2), len(signed_ind_b3), len(signed_ind_b4)]
    unique_len, counts = np.unique(all_edges, return_counts=True)
    max_edge = np.argmax(all_edges)
    min_edge = np.argmin(all_edges)

    # all 4 mags saw same change
    if max(counts) == 4:
        step_ind_total = len(signed_ind_b1)
    # at least two mags saw same change
    elif max(counts) == 3 or max(counts) == 2:
        if max_edge == 0:
            step_ind_total = len(signed_ind_b1)
            signed_ind_b1 = signed_ind_b1
            signed_ind_b2 = signed_ind_b1
            signed_ind_b3 = signed_ind_b1
            signed_ind_b4 = signed_ind_b1
        elif max_edge == 1:
            step_ind_total = len(signed_ind_b2)
            signed_ind_b1 = signed_ind_b2
            signed_ind_b2 = signed_ind_b2
            signed_ind_b3 = signed_ind_b2
            signed_ind_b4 = signed_ind_b2
        elif max_edge == 2:
            step_ind_total = len(signed_ind_b3)
            signed_ind_b1 = signed_ind_b3
            signed_ind_b2 = signed_ind_b3
            signed_ind_b3 = signed_ind_b3
            signed_ind_b4 = signed_ind_b3
        elif max_edge == 3:
            step_ind_total = len(signed_ind_b4)
            signed_ind_b1 = signed_ind_b4
            signed_ind_b2 = signed_ind_b4
            signed_ind_b3 = signed_ind_b4
            signed_ind_b4 = signed_ind_b4
    # only 1 mag saw change
    else:
        print("WARNING! Change was not detected in at least 2 mags! Taking minimum number of possible sources.")
        if max_edge == 0:
            if min_edge == 1:
                step_ind_total = len(signed_ind_b2)
                signed_ind_b1 = signed_ind_b2
                signed_ind_b2 = signed_ind_b2
                signed_ind_b3 = signed_ind_b2
                signed_ind_b4 = signed_ind_b2
            elif min_edge == 2:
                step_ind_total = len(signed_ind_b3)
                signed_ind_b1 = signed_ind_b3
                signed_ind_b2 = signed_ind_b3
                signed_ind_b3 = signed_ind_b3
                signed_ind_b4 = signed_ind_b3
            elif min_edge == 3:
                step_ind_total = len(signed_ind_b4)
                signed_ind_b1 = signed_ind_b4
                signed_ind_b2 = signed_ind_b4
                signed_ind_b3 = signed_ind_b4
                signed_ind_b4 = signed_ind_b4
        elif max_edge == 1:
            if min_edge == 0:
                step_ind_total = len(signed_ind_b1)
                signed_ind_b1 = signed_ind_b1
                signed_ind_b2 = signed_ind_b1
                signed_ind_b3 = signed_ind_b1
                signed_ind_b4 = signed_ind_b1
            elif min_edge == 2:
                step_ind_total = len(signed_ind_b3)
                signed_ind_b1 = signed_ind_b3
                signed_ind_b2 = signed_ind_b3
                signed_ind_b3 = signed_ind_b3
                signed_ind_b4 = signed_ind_b3
            elif min_edge == 3:
                step_ind_total = len(signed_ind_b4)
                signed_ind_b1 = signed_ind_b4
                signed_ind_b2 = signed_ind_b4
                signed_ind_b3 = signed_ind_b4
                signed_ind_b4 = signed_ind_b4
        elif max_edge == 2:
            if min_edge == 0:
                step_ind_total = len(signed_ind_b1)
                signed_ind_b1 = signed_ind_b1
                signed_ind_b2 = signed_ind_b1
                signed_ind_b3 = signed_ind_b1
                signed_ind_b4 = signed_ind_b1
            elif min_edge == 1:
                step_ind_total = len(signed_ind_b2)
                signed_ind_b1 = signed_ind_b2
                signed_ind_b2 = signed_ind_b2
                signed_ind_b3 = signed_ind_b2
                signed_ind_b4 = signed_ind_b2
            elif min_edge == 3:
                step_ind_total = len(signed_ind_b4)
                signed_ind_b1 = signed_ind_b4
                signed_ind_b2 = signed_ind_b4
                signed_ind_b3 = signed_ind_b4
                signed_ind_b4 = signed_ind_b4
        elif max_edge == 3:
            if min_edge == 0:
                step_ind_total = len(signed_ind_b1)
                signed_ind_b1 = signed_ind_b1
                signed_ind_b2 = signed_ind_b1
                signed_ind_b3 = signed_ind_b1
                signed_ind_b4 = signed_ind_b1
            elif min_edge == 1:
                step_ind_total = len(signed_ind_b2)
                signed_ind_b1 = signed_ind_b2
                signed_ind_b2 = signed_ind_b2
                signed_ind_b3 = signed_ind_b2
                signed_ind_b4 = signed_ind_b2
            elif min_edge == 2:
                step_ind_total = len(signed_ind_b3)
                signed_ind_b1 = signed_ind_b3
                signed_ind_b2 = signed_ind_b3
                signed_ind_b3 = signed_ind_b3
                signed_ind_b4 = signed_ind_b3

    plot_b1 = [time_arr[i] for i in np.abs(signed_ind_b1)]
    plot_b2 = [time_arr[i] for i in np.abs(signed_ind_b2)]
    plot_b3 = [time_arr[i] for i in np.abs(signed_ind_b3)]
    plot_b4 = [time_arr[i] for i in np.abs(signed_ind_b4)]

    if plot_noise_der:
        fig0, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
        ax1.plot(time_arr, b1_mag, color='tab:purple', label='B1')
        ax1.vlines(x=plot_b1, ymin=min(b1_mag)-min(b1_mag)*0.1, ymax=max(b1_mag)+max(b1_mag)*0.1,
                   colors='black', linestyles='dotted',
                   label='B1 indices')
        ax2.plot(time_arr, b2_mag, color='tab:blue', label='B2')
        ax2.vlines(x=plot_b2, ymin=min(b2_mag)-min(b2_mag)*0.1, ymax=max(b2_mag)+max(b2_mag)*0.1,
                   colors='black', linestyles='dotted',
                   label='B2 indices')
        ax3.plot(time_arr, b3_mag, color='tab:green', label='B3')
        ax3.vlines(x=plot_b3, ymin=min(b3_mag)-min(b3_mag)*0.1, ymax=max(b3_mag)+max(b3_mag)*0.1,
                   colors='black', linestyles='dotted',
                   label='B3 indices')
        ax4.plot(time_arr, b4_mag, color='tab:orange', label='B4')
        ax4.vlines(x=plot_b4, ymin=min(b4_mag)-min(b4_mag)*0.1, ymax=max(b4_mag)+max(b4_mag)*0.1,
                   colors='black', linestyles='dotted',
                   label='B4 indices')
        ax1.set_ylabel("B1 Field Mag", fontsize=14)
        ax2.set_ylabel("B2 Field Mag", fontsize=14)
        ax3.set_ylabel("B3 Field Mag", fontsize=14)
        ax4.set_ylabel("B4 Field Mag", fontsize=14)
        # plt.legend(loc='upper right')
        ax4.set_xlabel("Time (seconds)", fontsize=14)
        ax1.set_xlim(0, 500)
        ax2.set_xlim(0, 500)
        ax3.set_xlim(0, 500)
        ax4.set_xlim(0, 500)

        fig0.tight_layout()
        plt.show()

    return step_ind_total, signed_ind_b1, signed_ind_b2, signed_ind_b3, signed_ind_b4
