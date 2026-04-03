#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Imports
import numpy as np
import magnetpy.gradiometry_algorithm.tetrahedral_correction as tc


def noise_removal(step_ind_tot, ind_b1, ind_b2, ind_b3, ind_b4, b1_in, b2_in, b3_in, b4_in, step_width):
    # initialize lists of results
    new_re1_mag_list = []
    new_mue1_mag_list = []

    old_re0_mag_list = []
    old_mue0_mag_list = []
    final_cf_val_list = []
    cf_e0_val_list = []

    new_re1_hat_list = []
    new_mue1_hat_list = []
    old_re0_hat_list = []
    old_mue0_hat_list = []

    B1_corr = []
    B2_corr = []
    B3_corr = []
    B4_corr = []

    for i in np.arange(step_ind_tot):
        # for first pass solution - assume no overlapping sources
        background_b1 = b1_in[ind_b1[i] - step_width]  # step width is variable
        background_b2 = b2_in[ind_b2[i] - step_width]
        background_b3 = b3_in[ind_b3[i] - step_width]
        background_b4 = b4_in[ind_b4[i] - step_width]

        noise_b1 = b1_in[ind_b1[i] + step_width]  # step width is variable
        noise_b2 = b2_in[ind_b2[i] + step_width]
        noise_b3 = b3_in[ind_b3[i] + step_width]
        noise_b4 = b4_in[ind_b4[i] + step_width]

        diff_b1 = noise_b1 - background_b1
        diff_b2 = noise_b2 - background_b2
        diff_b3 = noise_b3 - background_b3
        diff_b4 = noise_b4 - background_b4

        # calculate best fit variables using amoeba algorithm to minimize cost function
        [mue0_hat, re0_hat, mue0_mag, re0_mag,
         new_mue1_hat, new_re1_hat, new_mue1_mag, new_re1_mag,
         final_cf_val, cf_e0_val, nsteps] = tc.array_correction(diff_b1, diff_b2, diff_b3, diff_b4)

        new_re1_hat_list.append(new_re1_hat)

        new_mue1_hat_list.append(new_mue1_hat)

        new_re1_mag_list.append(new_re1_mag)
        new_mue1_mag_list.append(new_mue1_mag)

        old_re0_hat_list.append(re0_hat)

        old_mue0_hat_list.append(mue0_hat)

        old_re0_mag_list.append(re0_mag)
        old_mue0_mag_list.append(mue0_mag)

        final_cf_val_list.append(final_cf_val)
        cf_e0_val_list.append(cf_e0_val)

        [B1_e1, B2_e1, B3_e1, B4_e1] = tc.mag_array(new_mue1_mag, new_mue1_hat, new_re1_mag, new_re1_hat)

        B1_corr.append(B1_e1)
        B2_corr.append(B2_e1)
        B3_corr.append(B3_e1)
        B4_corr.append(B4_e1)

    try:
        # check if corrections produced before proceeding
        B1_corr = np.stack(B1_corr)
        B2_corr = np.stack(B2_corr)
        B3_corr = np.stack(B3_corr)
        B4_corr = np.stack(B4_corr)

        return B1_corr, B2_corr, B3_corr, B4_corr
    except:
        err_msg = "No noise removed"
        print(err_msg)
        # no noise produced so return None
        return err_msg, err_msg, err_msg, err_msg