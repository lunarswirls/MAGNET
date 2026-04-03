#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Imports
import pandas as pd
import numpy as np
import magnetpy.gradiometry_algorithm.noise_removal as nr
import magnetpy.gradiometry_algorithm.noise_detection as nd


def correction_pipeline(time_arr, b1_field, b2_field, b3_field, b4_field, db_dt_tol=1.5e-9, step_width=1, noise_plots=False):
    """
    :param time_arr: time array
    :param b1_field: observed field at Mag1 (Tesla)
    :param b2_field: observed field at Mag2 (Tesla)
    :param b3_field: observed field at Mag3 (Tesla)
    :param b4_field: observed field at Mag4 (Tesla)
    :param db_dt_tol: tolerance for noise detection
    :param step_width: step width for noise removal indexing (default=3)
    :param noise_plots: If true, plot noise detection results (default=False)
    """
    # get uncorrected total field in nT
    Bd_uncorr = 0.25 * (np.array(b1_field) + np.array(b2_field) + np.array(b3_field) + np.array(b4_field)) * 1e9
    b1_field = np.stack(b1_field)
    b2_field = np.stack(b2_field)
    b3_field = np.stack(b3_field)
    b4_field = np.stack(b4_field)

    # get indices of changes in field magnitude greater than db_dt_tol to identify noise sources
    [step_ind_total, step_ind_b1, step_ind_b2, step_ind_b3, step_ind_b4] = nd.noise_detector(time_arr, b1_field, b2_field, b3_field, b4_field,
                                                                                             db_dt_tol=db_dt_tol, plot_noise_der=noise_plots)

    # get sign of change from indices
    b1_signs = np.sign(step_ind_b1)
    step_ind_b1 = np.abs(step_ind_b1)
    b2_signs = np.sign(step_ind_b2)
    step_ind_b2 = np.abs(step_ind_b2)
    b3_signs = np.sign(step_ind_b3)
    step_ind_b3 = np.abs(step_ind_b3)
    b4_signs = np.sign(step_ind_b4)
    step_ind_b4 = np.abs(step_ind_b4)

    try:
        # get field corrections for each identified source within time series
        B1_corr1, B2_corr1, B3_corr1, B4_corr1 = nr.noise_removal(step_ind_total, step_ind_b1, step_ind_b2, step_ind_b3,
                                                               step_ind_b4, b1_field, b2_field,
                                                               b3_field, b4_field, step_width)

        correction_b1 = [np.array([0, 0, 0])] * len(time_arr)
        correction_b2 = [np.array([0, 0, 0])] * len(time_arr)
        correction_b3 = [np.array([0, 0, 0])] * len(time_arr)
        correction_b4 = [np.array([0, 0, 0])] * len(time_arr)

        # get field magnitude positive change indices
        step_ind_b1_pos = step_ind_b1[b1_signs > 0]
        step_ind_b2_pos = step_ind_b2[b2_signs > 0]
        step_ind_b3_pos = step_ind_b3[b3_signs > 0]
        step_ind_b4_pos = step_ind_b4[b4_signs > 0]

        # get field magnitude negative change indices
        step_ind_b1_neg = step_ind_b1[b1_signs < 0]
        step_ind_b2_neg = step_ind_b2[b2_signs < 0]
        step_ind_b3_neg = step_ind_b3[b3_signs < 0]
        step_ind_b4_neg = step_ind_b4[b4_signs < 0]

        if len(step_ind_b1_pos) < len(step_ind_b1_neg):
            step_ind_b1_neg = step_ind_b1_neg[0:len(step_ind_b1_pos)]
        elif len(step_ind_b1_pos) > len(step_ind_b1_neg):
            step_ind_b1_pos = step_ind_b1_pos[0:len(step_ind_b1_neg)]

        if len(step_ind_b2_pos) < len(step_ind_b2_neg):
            step_ind_b2_neg = step_ind_b2_neg[0:len(step_ind_b2_pos)]
        elif len(step_ind_b2_pos) > len(step_ind_b2_neg):
            step_ind_b2_pos = step_ind_b2_pos[0:len(step_ind_b2_neg)]

        if len(step_ind_b3_pos) < len(step_ind_b3_neg):
            step_ind_b3_neg = step_ind_b3_neg[0:len(step_ind_b3_pos)]
        elif len(step_ind_b3_pos) > len(step_ind_b3_neg):
            step_ind_b3_pos = step_ind_b3_pos[0:len(step_ind_b3_neg)]

        if len(step_ind_b4_pos) < len(step_ind_b4_neg):
            step_ind_b4_neg = step_ind_b4_neg[0:len(step_ind_b4_pos)]
        elif len(step_ind_b4_pos) > len(step_ind_b4_neg):
            step_ind_b4_pos = step_ind_b4_pos[0:len(step_ind_b4_neg)]

        # get correction average in nT
        B1_corr1_avg = B1_corr1[::2] * 1e9
        B2_corr1_avg = B2_corr1[::2] * 1e9
        B3_corr1_avg = B3_corr1[::2] * 1e9
        B4_corr1_avg = B4_corr1[::2] * 1e9

        for j in np.arange(len(step_ind_b1_pos)):
            if step_ind_b1_pos[j] < step_ind_b1_neg[j]:
                correction_b1[step_ind_b1_pos[j]:step_ind_b1_neg[j]] += B1_corr1_avg[j]
            else:
                correction_b1[step_ind_b1_neg[j]:step_ind_b1_pos[j]] += B1_corr1_avg[j]

        for k in np.arange(len(step_ind_b2_pos)):
            if step_ind_b2_pos[k] < step_ind_b2_neg[k]:
                correction_b2[step_ind_b2_pos[k]:step_ind_b2_neg[k]] += B2_corr1_avg[k]
            else:
                correction_b2[step_ind_b2_neg[k]:step_ind_b2_pos[k]] += B2_corr1_avg[k]

        for m in np.arange(len(step_ind_b3_pos)):
            if step_ind_b3_pos[m] < step_ind_b3_neg[m]:
                correction_b3[step_ind_b3_pos[m]:step_ind_b3_neg[m]] += B3_corr1_avg[m]
            else:
                correction_b3[step_ind_b3_neg[m]:step_ind_b3_pos[m]] += B3_corr1_avg[m]

        for n in np.arange(len(step_ind_b4_pos)):
            if step_ind_b4_pos[n] < step_ind_b4_neg[n]:
                correction_b4[step_ind_b4_pos[n]:step_ind_b4_neg[n]] += B4_corr1_avg[n]
            else:
                correction_b4[step_ind_b4_neg[n]:step_ind_b4_pos[n]] += B4_corr1_avg[n]

        correction_b1 = np.stack(correction_b1)
        correction_b2 = np.stack(correction_b2)
        correction_b3 = np.stack(correction_b3)
        correction_b4 = np.stack(correction_b4)

        b1_field_nT = b1_field * 1e9
        b2_field_nT = b2_field * 1e9
        b3_field_nT = b3_field * 1e9
        b4_field_nT = b4_field * 1e9

        # corrected field in nT
        corrected_b1 = np.array(b1_field_nT) - np.array(correction_b1)
        corrected_b2 = np.array(b2_field_nT) - np.array(correction_b2)
        corrected_b3 = np.array(b3_field_nT) - np.array(correction_b3)
        corrected_b4 = np.array(b4_field_nT) - np.array(correction_b4)

        bfield_df = pd.DataFrame(np.array(b1_field_nT))
        bfield_df.columns = ['B1X', 'B1Y', 'B1Z']

        bfield_df['B2X'] = np.array(b2_field_nT)[:, 0]
        bfield_df['B2Y'] = np.array(b2_field_nT)[:, 1]
        bfield_df['B2Z'] = np.array(b2_field_nT)[:, 2]
        bfield_df['B3X'] = np.array(b3_field_nT)[:, 0]
        bfield_df['B3Y'] = np.array(b3_field_nT)[:, 1]
        bfield_df['B3Z'] = np.array(b3_field_nT)[:, 2]
        bfield_df['B4X'] = np.array(b4_field_nT)[:, 0]
        bfield_df['B4Y'] = np.array(b4_field_nT)[:, 1]
        bfield_df['B4Z'] = np.array(b4_field_nT)[:, 2]

        bfield_df['B1X_CORR'] = np.array(corrected_b1)[:, 0]
        bfield_df['B1Y_CORR'] = np.array(corrected_b1)[:, 1]
        bfield_df['B1Z_CORR'] = np.array(corrected_b1)[:, 2]

        bfield_df['B2X_CORR'] = np.array(corrected_b2)[:, 0]
        bfield_df['B2Y_CORR'] = np.array(corrected_b2)[:, 1]
        bfield_df['B2Z_CORR'] = np.array(corrected_b2)[:, 2]

        bfield_df['B3X_CORR'] = np.array(corrected_b3)[:, 0]
        bfield_df['B3Y_CORR'] = np.array(corrected_b3)[:, 1]
        bfield_df['B3Z_CORR'] = np.array(corrected_b3)[:, 2]

        bfield_df['B4X_CORR'] = np.array(corrected_b4)[:, 0]
        bfield_df['B4Y_CORR'] = np.array(corrected_b4)[:, 1]
        bfield_df['B4Z_CORR'] = np.array(corrected_b4)[:, 2]

        bfield_df['BDX_UNCORR'] = np.array(Bd_uncorr)[:, 0]
        bfield_df['BDY_UNCORR'] = np.array(Bd_uncorr)[:, 1]
        bfield_df['BDZ_UNCORR'] = np.array(Bd_uncorr)[:, 2]

        bfield_df['BDX_CORR'] = 0.25 * bfield_df['B1X_CORR'] + bfield_df['B2X_CORR'] + bfield_df['B3X_CORR'] + bfield_df['B4X_CORR']
        bfield_df['BDY_CORR'] = 0.25 * bfield_df['B1Y_CORR'] + bfield_df['B2Y_CORR'] + bfield_df['B3Y_CORR'] + bfield_df['B4Y_CORR']
        bfield_df['BDZ_CORR'] = 0.25 * bfield_df['B1Z_CORR'] + bfield_df['B2Z_CORR'] + bfield_df['B3Z_CORR'] + bfield_df['B4Z_CORR']

        return bfield_df
    except:
        print("No corrections produced")
        # no correction produced so return None
        return None
