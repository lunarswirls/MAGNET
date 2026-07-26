#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Contains common static methods for plotting vectors

@Author: Dany Waller
"""
import os
import numpy as np
import pandas as pd
import math
import magnetpy.utilities.Math_Utils as MathUtil
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.ticker import MaxNLocator
import matplotlib.dates as mdates


def _add_residual_target(ax):
    target_line = ax.axhline(1.5, color='black', linestyle=':', linewidth=1.5, label='1.5 nT target')
    ax.legend(handles=[target_line], fontsize='small', loc='upper right')


def rmse_quicklook(bfield_df, input_dir, test, n):
    # Create a directory or avoid error if it exists
    test_dir = os.path.join(input_dir, test)
    os.makedirs(test_dir, exist_ok=True)

    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    bfield_df.plot(x="Time", y=["RMSE_B1", "RMSE_B2", "RMSE_B3", "RMSE_B4"], ax=ax, style=['r', 'y--', 'g-.', 'b:'], )
    plt.xlabel("Time (seconds)", fontsize=16)
    plt.ylabel("RMSE (nT)", fontsize=16)
    plt.title("RMSE Quick Look", fontsize=20)
    plt.legend(ncol=1, loc='center left', bbox_to_anchor=(1, 0.5), fancybox=True)
    plt.tight_layout()
    plt.savefig(os.path.join(test_dir, 'quick_look_rmse_' + str(n) + '.png'), format='png', dpi=300)
    plt.close()
    # plt.show()


def correction_quicklook(raw_bfield_df, bfield_df, input_dir, test, n):
    # Create a directory or avoid error if it exists
    test_dir = os.path.join(input_dir, test)
    os.makedirs(test_dir, exist_ok=True)

    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    colors1 = ['tomato', 'gold', 'deepskyblue']
    colors2 = ['firebrick', 'darkgoldenrod', 'royalblue']
    raw_bfield_df.plot(x="time_fmt", y=["BGX", "BGY", "BGZ"], ax=ax, linewidth=2, color=colors1)
    bfield_df.plot(x="Time", y=["B1X_CORR", "B1Y_CORR", "B1Z_CORR"], ax=ax, style=[':', ':', ':'], color=colors2)
    plt.xlabel("Time (seconds)", fontsize=16)
    plt.ylabel("Field (nT)", fontsize=16)
    plt.title("Correction Quick Look", fontsize=20)
    plt.legend(ncol=1, loc='center left', bbox_to_anchor=(1, 0.5), fancybox=True)
    plt.tight_layout()
    plt.savefig(os.path.join(test_dir, 'quick_look_corr_' + str(n) + '.png'), format='png', dpi=300)
    plt.close()
    # plt.show()


def thruster_plot(thruster_df, input_dir, test):
    fig1, ax1 = plt.subplots(1, 1, figsize=(8, 4))
    thruster_df.plot(x="Time", y=["Thruster 1", "Thruster 2", "Thruster 3", "Thruster 4"], ax=ax1,
                     style=['r', 'y--', 'g-.', 'b:'], )
    plt.xlabel("Time (seconds)", fontsize=16)
    plt.ylabel("Thruster Firing State", fontsize=16)
    x = [0, 1]
    labels = ['Off', 'On']
    plt.yticks(x, labels, rotation='horizontal')
    plt.title("Thruster Firing State", fontsize=20)
    plt.legend(['Thruster 1', 'Thruster 2', 'Thruster 3', 'Thruster 4'], loc='center left', bbox_to_anchor=(1, 0.5),
               fontsize='medium')
    plt.tight_layout()
    thruster_output = os.path.join(input_dir, "thruster_state_" + test + ".png")
    plt.savefig(thruster_output, format='png', dpi=300)
    plt.close()
    # plt.show()


def vector_final_plot(bfield_df, input_dir, test):
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9), (ax10, ax11, ax12)) = plt.subplots(4, 3, figsize=(14, 14),
                                                                                                 sharex='all')
    try:
        bfield_df['Time_s'] = pd.to_timedelta(bfield_df['Time'].str.split(' ').str[1]).dt.total_seconds()
    except:
        bfield_df['Time'] = bfield_df['Time'].astype(str)
        bfield_df['Time_s'] = pd.to_timedelta(bfield_df['Time'].str.split(' ').str[1]).dt.total_seconds()

    colors1 = ['tomato', 'gold', 'deepskyblue']
    colors2 = ['firebrick', 'darkgoldenrod', 'royalblue']

    # ax1
    bfield_df.plot(x="Time_s", y=["B1X_UNCORR", "B1Y_UNCORR", "B1Z_UNCORR"], ax=ax1, color=['red', 'goldenrod', 'blue'])

    # ax2
    bfield_df.plot(x="Time_s", y="BGX", ax=ax2, style=':', linewidth=2, color='tomato')
    bfield_df.plot(x="Time_s", y="B1X_CORR", ax=ax2, color='firebrick', alpha=0.5)
    bfield_df.plot(x="Time_s", y="BGY", ax=ax2, style=':', linewidth=2, color='gold')
    bfield_df.plot(x="Time_s", y="B1Y_CORR", ax=ax2, color='darkgoldenrod', alpha=0.5)
    bfield_df.plot(x="Time_s", y="BGZ", ax=ax2, style=':', linewidth=2, color='deepskyblue')
    bfield_df.plot(x="Time_s", y="B1Z_CORR", ax=ax2, color='royalblue', alpha=0.5)
    # bfield_df.plot(x="Time_s", y=["BGX", "BGY", "BGZ"], ax=ax2, style=[':', ':', ':'], linewidth=2, color=colors1)
    # bfield_df.plot(x="Time_s", y=["B1X_CORR", "B1Y_CORR", "B1Z_CORR"], ax=ax2, color=colors2, alpha=0.5)

    # ax3
    bfield_df.plot(x="Time_s", y=["RMSE_B1"], ax=ax3, style=['-'], linewidth=2, color='magenta')

    # ax4
    bfield_df.plot(x="Time_s", y=["B2X_UNCORR", "B2Y_UNCORR", "B2Z_UNCORR"], ax=ax4, color=['red', 'goldenrod', 'blue'])

    # ax5
    bfield_df.plot(x="Time_s", y="BGX", ax=ax5, style=':', linewidth=2, color='tomato')
    bfield_df.plot(x="Time_s", y="B2X_CORR", ax=ax5, color='firebrick', alpha=0.5)
    bfield_df.plot(x="Time_s", y="BGY", ax=ax5, style=':', linewidth=2, color='gold')
    bfield_df.plot(x="Time_s", y="B2Y_CORR", ax=ax5, color='darkgoldenrod', alpha=0.5)
    bfield_df.plot(x="Time_s", y="BGZ", ax=ax5, style=':', linewidth=2, color='deepskyblue')
    bfield_df.plot(x="Time_s", y="B2Z_CORR", ax=ax5, color='royalblue', alpha=0.5)
    # bfield_df.plot(x="Time_s", y=["BGX", "BGY", "BGZ"], ax=ax5, style=[':', ':', ':'], linewidth=2, color=colors1)
    # bfield_df.plot(x="Time_s", y=["B2X_CORR", "B2Y_CORR", "B2Z_CORR"], ax=ax5, color=colors2, alpha=0.5)

    # ax6
    bfield_df.plot(x="Time_s", y=["RMSE_B2"], ax=ax6, style=['-'], linewidth=2, color='magenta')

    # ax7
    bfield_df.plot(x="Time_s", y=["B3X_UNCORR", "B3Y_UNCORR", "B3Z_UNCORR"], ax=ax7, color=['red', 'goldenrod', 'blue'])

    # ax8
    bfield_df.plot(x="Time_s", y="BGX", ax=ax8, style=':', linewidth=2, color='tomato')
    bfield_df.plot(x="Time_s", y="B3X_CORR", ax=ax8, color='firebrick', alpha=0.5)
    bfield_df.plot(x="Time_s", y="BGY", ax=ax8, style=':', linewidth=2, color='gold')
    bfield_df.plot(x="Time_s", y="B3Y_CORR", ax=ax8, color='darkgoldenrod', alpha=0.5)
    bfield_df.plot(x="Time_s", y="BGZ", ax=ax8, style=':', linewidth=2, color='deepskyblue')
    bfield_df.plot(x="Time_s", y="B3Z_CORR", ax=ax8, color='royalblue', alpha=0.5)
    # bfield_df.plot(x="Time_s", y=["BGX", "BGY", "BGZ"], ax=ax8, style=[':', ':', ':'], linewidth=2, color=colors1)
    # bfield_df.plot(x="Time_s", y=["B3X_CORR", "B3Y_CORR", "B3Z_CORR"], ax=ax8, color=colors2, alpha=0.5)

    # ax9
    bfield_df.plot(x="Time_s", y=["RMSE_B3"], ax=ax9, style=['-'], linewidth=2, color='magenta')

    # ax10
    bfield_df.plot(x="Time_s", y=["B4X_UNCORR", "B4Y_UNCORR", "B4Z_UNCORR"], ax=ax10, color=['red', 'goldenrod', 'blue'])

    # ax11
    bfield_df.plot(x="Time_s", y="BGX", ax=ax11, style=':', linewidth=2, color='tomato')
    bfield_df.plot(x="Time_s", y="B4X_CORR", ax=ax11, color='firebrick', alpha=0.5)
    bfield_df.plot(x="Time_s", y="BGY", ax=ax11, style=':', linewidth=2, color='gold')
    bfield_df.plot(x="Time_s", y="B4Y_CORR", ax=ax11, color='darkgoldenrod', alpha=0.5)
    bfield_df.plot(x="Time_s", y="BGZ", ax=ax11, style=':', linewidth=2, color='deepskyblue')
    bfield_df.plot(x="Time_s", y="B4Z_CORR", ax=ax11, color='royalblue', alpha=0.5)
    # bfield_df.plot(x="Time_s", y=["BGX", "BGY", "BGZ"], ax=ax11, style=[':', ':', ':'], linewidth=2, color=colors1)
    # bfield_df.plot(x="Time_s", y=["B4X_CORR", "B4Y_CORR", "B4Z_CORR"], ax=ax11, color=colors2, alpha=0.5)

    # ax12
    bfield_df.plot(x="Time_s", y=["RMSE_B4"], ax=ax12, style=['-'], linewidth=2, color='magenta')

    ax1.set_ylabel("nT", fontsize=16)
    # ax2.set_ylabel("B Field (nT)", fontsize=16)
    # ax3.set_ylabel("RMSE (nT)", fontsize=16)
    ax4.set_ylabel("nT", fontsize=16)
    # ax5.set_ylabel("B Field (nT)", fontsize=16)
    # ax6.set_ylabel("RMSE (nT)", fontsize=16)
    ax7.set_ylabel("nT", fontsize=16)
    # ax8.set_ylabel("B Field (nT)", fontsize=16)
    # ax9.set_ylabel("RMSE (nT)", fontsize=16)
    ax10.set_ylabel("nT", fontsize=16)
    ax10.set_xlabel("Time (seconds)", fontsize=16)
    # ax11.set_ylabel("nT", fontsize=16)
    ax11.set_xlabel("Time (seconds)", fontsize=16)
    # ax12.set_ylabel("RMSE (nT)", fontsize=16)
    ax12.set_xlabel("Time (seconds)", fontsize=16)

    # date_format = mdates.DateFormatter('%M:%S')

    # Apply the format to the x-axis
    # ax10.xaxis.set_major_formatter(date_format)
    # ax11.xaxis.set_major_formatter(date_format)
    # ax12.xaxis.set_major_formatter(date_format)
    # for ax in (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10, ax11, ax12):
        # ax.xaxis.set_tick_params(which='minor', bottom=False)

    ax1.set_title("Uncorrected Mag #0", fontsize=20)
    ax2.set_title("Corrected Mag #0", fontsize=20)
    ax3.set_title("RMS Error Mag #0", fontsize=20)

    ax4.set_title("Uncorrected Mag #1", fontsize=20)
    ax5.set_title("Corrected Mag #1", fontsize=20)
    ax6.set_title("RMS Error Mag #1", fontsize=20)

    ax7.set_title("Uncorrected Mag #2", fontsize=20)
    ax8.set_title("Corrected Mag #2", fontsize=20)
    ax9.set_title("RMS Error Mag #2", fontsize=20)

    ax10.set_title("Uncorrected Mag #3", fontsize=20)
    ax11.set_title("Corrected Mag #3", fontsize=20)
    ax12.set_title("RMS Error Mag #3", fontsize=20)

    ax1.legend(['X', 'Y', 'Z'], fontsize='medium', ncol=3, loc='lower left')
    ax2.legend(['True X', 'Final X', 'True Y', 'Final Y', 'True Z', 'Final Z'],
               fontsize='small', ncol=3, loc='lower left')
    _add_residual_target(ax3)
    ax4.legend(['X', 'Y', 'Z'], fontsize='medium', ncol=3, loc='lower left')
    ax5.legend(['True X', 'Final X', 'True Y', 'Final Y', 'True Z', 'Final Z'],
               fontsize='small', ncol=3, loc='lower left')
    _add_residual_target(ax6)
    ax7.legend(['X', 'Y', 'Z'], fontsize='medium', ncol=3, loc='lower left')
    ax8.legend(['True X', 'Final X', 'True Y', 'Final Y', 'True Z', 'Final Z'],
               fontsize='small', ncol=3, loc='lower left')
    _add_residual_target(ax9)
    ax10.legend(['X', 'Y', 'Z'], fontsize='medium', ncol=3, loc='lower left')
    ax11.legend(['True X', 'Final X', 'True Y', 'Final Y', 'True Z', 'Final Z'],
                fontsize='small', ncol=3, loc='lower left')
    _add_residual_target(ax12)

    ymin1, ymax1 = ax1.get_ylim()
    ymin4, ymax4 = ax4.get_ylim()
    ymin7, ymax7 = ax7.get_ylim()
    ymin10, ymax10 = ax10.get_ylim()

    if 'earth' not in test:
        ymax_raw = max(ymax1, ymax4, ymax7, ymax10) + (max(ymax1, ymax4, ymax7, ymax10) * 0.25)
        ymin_raw = ymax_raw
    elif '500km' in test:
        ymin_raw = 52000
        ymax_raw = 22000
    elif '10000km' in test:
        ymin_raw = 4500
        ymax_raw = 2500
    else:
        ymin_raw = 52000
        ymax_raw = 22000

    ymin2, ymax2 = ax2.get_ylim()
    ymin5, ymax5 = ax5.get_ylim()
    ymin8, ymax8 = ax8.get_ylim()
    ymin11, ymax11 = ax11.get_ylim()

    if 'earth' not in test:
        ymin_corr1 = 0 - math.floor((max(map(abs, [ymin2, ymin5, ymin8, ymin11])) + (max(map(abs, [ymin2, ymin5, ymin8, ymin11])) * 0.25)))
        ymax_corr1 = max(ymax2, ymax5, ymax8, ymax11) + (max(ymax2, ymax5, ymax8, ymax11) * 0.25)
        ymax_corr = max(np.abs(ymin_corr1), ymax_corr1)
        ymin_corr = ymax_corr
    elif '500km' in test:
        ymin_corr = 52000
        ymax_corr = 22000
    elif '10000km' in test:
        ymin_corr = 4500
        ymax_corr = 2500
    else:
        ymin_corr = 52000
        ymax_corr = 22000

    ymin3, ymax3 = ax3.get_ylim()
    ymin6, ymax6 = ax6.get_ylim()
    ymin9, ymax9 = ax9.get_ylim()
    ymin12, ymax12 = ax12.get_ylim()

    if 'earth' not in test:
        ymin_rmse = 0 - (math.ceil(min(ymax3, ymax6, ymax9, ymax12) * 0.25))
        ymax_rmse = math.ceil(max(ymax3, ymax6, ymax9, ymax12) + (max(ymax3, ymax6, ymax9, ymax12) * 0.25))
    else:
        ymin_rmse = -0.5
        ymax_rmse = 10

    ax1.set_ylim(-ymin_raw, ymax_raw)
    ax2.set_ylim(-ymin_corr, ymax_corr)
    ax3.set_ylim(ymin_rmse, ymax_rmse)
    ax4.set_ylim(-ymin_raw, ymax_raw)
    ax5.set_ylim(-ymin_corr, ymax_corr)
    ax6.set_ylim(ymin_rmse, ymax_rmse)
    ax7.set_ylim(-ymin_raw, ymax_raw)
    ax8.set_ylim(-ymin_corr, ymax_corr)
    ax9.set_ylim(ymin_rmse, ymax_rmse)
    ax10.set_ylim(-ymin_raw, ymax_raw)
    ax11.set_ylim(-ymin_corr, ymax_corr)
    ax12.set_ylim(ymin_rmse, ymax_rmse)

    for ax in [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10, ax11, ax12]:
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))

    fig.tight_layout()
    correction_vector_output = str(os.path.join(input_dir, test + "_vector_correction_error.png"))
    fig.savefig(correction_vector_output, format='png', dpi=300)
    plt.close()
    # plt.show()


def mag_final_plot(bfield_df, input_dir, test):
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9), (ax10, ax11, ax12)) = plt.subplots(4, 3, figsize=(14,14),
                                                                                                          sharex='all')

    try:
        bfield_df['Time_s'] = pd.to_timedelta(bfield_df['Time'].str.split(' ').str[1]).dt.total_seconds()
    except:
        bfield_df['Time'] = bfield_df['Time'].astype(str)
        bfield_df['Time_s'] = pd.to_timedelta(bfield_df['Time'].str.split(' ').str[1]).dt.total_seconds()

    bfield_df.plot(x="Time_s", y="B1_UNCORR_MAG", ax=ax1, color='firebrick')
    bfield_df.plot(x="Time_s", y="BG_MAG", ax=ax1, style='-', linewidth=2, color='royalblue', alpha=0.7)
    bfield_df.plot(x="Time_s", y="B1_CORR_MAG", ax=ax2, color='goldenrod')
    bfield_df.plot(x="Time_s", y="BG_MAG", ax=ax2, style='-', linewidth=2, color='royalblue', alpha=0.5)
    bfield_df.plot(x="Time_s", y="RMSE_B1", ax=ax3, style='-', linewidth=2, color='magenta')

    bfield_df.plot(x="Time_s", y="B2_UNCORR_MAG", ax=ax4, color='firebrick')
    bfield_df.plot(x="Time_s", y="BG_MAG", ax=ax4, style='-', linewidth=2, color='royalblue', alpha=0.7)
    bfield_df.plot(x="Time_s", y="B2_CORR_MAG", ax=ax5, color='goldenrod')
    bfield_df.plot(x="Time_s", y="BG_MAG", ax=ax5, style='-', linewidth=2, color='royalblue', alpha=0.5)
    bfield_df.plot(x="Time_s", y="RMSE_B2", ax=ax6, style='-', linewidth=2, color='magenta')

    bfield_df.plot(x="Time_s", y="B3_UNCORR_MAG", ax=ax7, color='firebrick')
    bfield_df.plot(x="Time_s", y="BG_MAG", ax=ax7, style='-', linewidth=2, color='royalblue', alpha=0.7)
    bfield_df.plot(x="Time_s", y="B3_CORR_MAG", ax=ax8, color='goldenrod')
    bfield_df.plot(x="Time_s", y="BG_MAG", ax=ax8, style='-', linewidth=2, color='royalblue', alpha=0.5)
    bfield_df.plot(x="Time_s", y="RMSE_B3", ax=ax9, style='-', linewidth=2, color='magenta')

    bfield_df.plot(x="Time_s", y="B4_UNCORR_MAG", ax=ax10, color='firebrick')
    bfield_df.plot(x="Time_s", y="BG_MAG", ax=ax10, style='-', linewidth=2, color='royalblue', alpha=0.7)
    bfield_df.plot(x="Time_s", y="B4_CORR_MAG", ax=ax11, color='goldenrod')
    bfield_df.plot(x="Time_s", y="BG_MAG", ax=ax11, style='-', linewidth=2, color='royalblue', alpha=0.5)
    bfield_df.plot(x="Time_s", y="RMSE_B4", ax=ax12, style='-', linewidth=2, color='magenta')

    ax1.set_ylabel("nT", fontsize=16)
    # ax2.set_ylabel("B Field (nT)", fontsize=16)
    # ax3.set_ylabel("RMSE (nT)", fontsize=16)
    ax4.set_ylabel("nT", fontsize=16)
    # ax5.set_ylabel("B Field (nT)", fontsize=16)
    # ax6.set_ylabel("RMSE (nT)", fontsize=16)
    ax7.set_ylabel("nT", fontsize=16)
    # ax8.set_ylabel("B Field (nT)", fontsize=16)
    # ax9.set_ylabel("RMSE (nT)", fontsize=16)
    ax10.set_ylabel("nT", fontsize=16)
    ax10.set_xlabel("Time (seconds)", fontsize=16)
    # ax11.set_ylabel("nT", fontsize=16)
    ax11.set_xlabel("Time (seconds)", fontsize=16)
    # ax12.set_ylabel("RMSE (nT)", fontsize=16)
    ax12.set_xlabel("Time (seconds)", fontsize=16)

    # date_format = mdates.DateFormatter('%M:%S')

    # Apply the format to the x-axis
    # ax10.xaxis.set_major_formatter(date_format)
    # ax11.xaxis.set_major_formatter(date_format)
    # ax12.xaxis.set_major_formatter(date_format)
    # for ax in (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10, ax11, ax12):
        # ax.xaxis.set_tick_params(which='minor', bottom=False)

    ax1.set_title("Uncorrected Mag #0", fontsize=20)
    ax2.set_title("Corrected Mag #0", fontsize=20)
    ax3.set_title("RMS Error Mag #0", fontsize=20)

    ax4.set_title("Uncorrected Mag #1", fontsize=20)
    ax5.set_title("Corrected Mag #1", fontsize=20)
    ax6.set_title("RMS Error Mag #1", fontsize=20)

    ax7.set_title("Uncorrected Mag #2", fontsize=20)
    ax8.set_title("Corrected Mag #2", fontsize=20)
    ax9.set_title("RMS Error Mag #2", fontsize=20)

    ax10.set_title("Uncorrected Mag #3", fontsize=20)
    ax11.set_title("Corrected Mag #3", fontsize=20)
    ax12.set_title("RMS Error Mag #3", fontsize=20)

    ax1.legend(['Uncorrected $B_{mag}$', 'True $B_{mag}$'], fontsize='medium', ncol=2, loc='lower left')
    ax2.legend(['Corrected $B_{mag}$', 'True $B_{mag}$'], fontsize='medium', ncol=2, loc='lower left')
    _add_residual_target(ax3)
    ax4.legend(['Uncorrected $B_{mag}$', 'True $B_{mag}$'], fontsize='medium', ncol=2, loc='lower left')
    ax5.legend(['Corrected $B_{mag}$', 'True $B_{mag}$'], fontsize='medium', ncol=2, loc='lower left')
    _add_residual_target(ax6)
    ax7.legend(['Uncorrected $B_{mag}$', 'True $B_{mag}$'], fontsize='medium', ncol=2, loc='lower left')
    ax8.legend(['Corrected $B_{mag}$', 'True $B_{mag}$'], fontsize='medium', ncol=2, loc='lower left')
    _add_residual_target(ax9)
    ax10.legend(['Uncorrected $B_{mag}$', 'True $B_{mag}$'], fontsize='medium', ncol=2, loc='lower left')
    ax11.legend(['Corrected $B_{mag}$', 'True $B_{mag}$'], fontsize='medium', ncol=2, loc='lower left')
    _add_residual_target(ax12)

    ymin1, ymax1 = ax1.get_ylim()
    ymin4, ymax4 = ax4.get_ylim()
    ymin7, ymax7 = ax7.get_ylim()
    ymin10, ymax10 = ax10.get_ylim()

    if 'earth' not in test:
        ymin_raw = 0 - (math.floor(min(ymax1, ymax4, ymax7, ymax10) * 0.25))
        ymax_raw = max(ymax1, ymax4, ymax7, ymax10) + (max(ymax1, ymax4, ymax7, ymax10) * 0.25)
    elif '500km' in test:
        ymin_raw = 39200
        ymax_raw = 39700
    elif '10000km' in test:
        ymin_raw = 2400
        ymax_raw = 2900
    else:
        ymin_raw = 39000
        ymax_raw = 40000

    ymin2, ymax2 = ax2.get_ylim()
    ymin5, ymax5 = ax5.get_ylim()
    ymin8, ymax8 = ax8.get_ylim()
    ymin11, ymax11 = ax11.get_ylim()

    if 'earth' not in test:
        ymin_corr = 0 - (math.ceil(min(ymax2, ymax5, ymax8, ymax11) * 0.25))
        ymax_corr = math.ceil(max(ymax2, ymax5, ymax8, ymax11) + (max(ymax2, ymax5, ymax8, ymax11) * 0.25))
    elif '500km' in test:
        ymin_corr = 39560
        ymax_corr = 39580
    elif '10000km' in test:
        ymin_corr = 2780
        ymax_corr = 2795
    else:
        ymin_corr = 39000
        ymax_corr = 40000

    ymin3, ymax3 = ax3.get_ylim()
    ymin6, ymax6 = ax6.get_ylim()
    ymin9, ymax9 = ax9.get_ylim()
    ymin12, ymax12 = ax12.get_ylim()

    if 'earth' not in test:
        ymin_rmse = 0 - (math.ceil(min(ymax3, ymax6, ymax9, ymax12) * 0.25))
        ymax_rmse = math.ceil(max(ymax3, ymax6, ymax9, ymax12) + (max(ymax3, ymax6, ymax9, ymax12) * 0.25))
    else:
        ymin_rmse = -0.5
        ymax_rmse = 10

    ax1.set_ylim(ymin_raw, ymax_raw)
    ax2.set_ylim(ymin_corr, ymax_corr)
    ax3.set_ylim(ymin_rmse, ymax_rmse)
    ax4.set_ylim(ymin_raw, ymax_raw)
    ax5.set_ylim(ymin_corr, ymax_corr)
    ax6.set_ylim(ymin_rmse, ymax_rmse)
    ax7.set_ylim(ymin_raw, ymax_raw)
    ax8.set_ylim(ymin_corr, ymax_corr)
    ax9.set_ylim(ymin_rmse, ymax_rmse)
    ax10.set_ylim(ymin_raw, ymax_raw)
    ax11.set_ylim(ymin_corr, ymax_corr)
    ax12.set_ylim(ymin_rmse, ymax_rmse)

    for ax in [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10, ax11, ax12]:
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))

    fig.tight_layout()
    correction_mag_output = str(os.path.join(input_dir, test + "_mag_correction_error.png"))
    fig.savefig(correction_mag_output, format='png', dpi=300)
    plt.close()
    # plt.show()


def center_mag_final_plot(bfield_df_in, input_dir, test):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12,4),  sharex='all')

    if 'log' in test:
        try:
            bfield_df_in['Time_s'] = pd.to_timedelta(bfield_df_in['Time'].str.split(' ').str[1]).dt.total_seconds() * 1.e9
        except:
            bfield_df_in['Time'] = bfield_df_in['Time'].astype(str)
            bfield_df_in['Time_s'] = pd.to_timedelta(bfield_df_in['Time'].str.split(' ').str[1]).dt.total_seconds() * 1.e9

        bfield_df = bfield_df_in[bfield_df_in['Time_s'] < 320]
    else:
        bfield_df = bfield_df_in.copy()
        try:
            bfield_df['Time_s'] = pd.to_timedelta(bfield_df['Time'].str.split(' ').str[1]).dt.total_seconds()
        except:
            bfield_df['Time'] = bfield_df['Time'].astype(str)
            bfield_df['Time_s'] = pd.to_timedelta(bfield_df['Time'].str.split(' ').str[1]).dt.total_seconds()


    bfield_df["BC_UNCORR_MAG"] = (bfield_df["B1_UNCORR_MAG"] + bfield_df["B2_UNCORR_MAG"] + bfield_df["B3_UNCORR_MAG"] + bfield_df["B4_UNCORR_MAG"])/4
    bfield_df["BC_CORR_MAG"] = (bfield_df["B1_CORR_MAG"] + bfield_df["B2_CORR_MAG"] + bfield_df["B3_CORR_MAG"] + bfield_df["B4_CORR_MAG"]) / 4
    # bfield_df["RMSE_BC"] = np.abs(((bfield_df["RMSE_B1"] + bfield_df["RMSE_B2"] + bfield_df["RMSE_B3"] + bfield_df["RMSE_B4"]) / 4) - bfield_df["BG_MAG"])
    bfield_df["RMSE_BC"] = np.abs(bfield_df["BC_CORR_MAG"] - bfield_df["BG_MAG"])

    bfield_df.plot(x="Time_s", y="BC_UNCORR_MAG", ax=ax1, color='firebrick')
    bfield_df.plot(x="Time_s", y="BG_MAG", ax=ax1, style='-', linewidth=2, color='royalblue', alpha=0.7)
    bfield_df.plot(x="Time_s", y="BC_CORR_MAG", ax=ax2, color='goldenrod')
    bfield_df.plot(x="Time_s", y="BG_MAG", ax=ax2, style='-', linewidth=2, color='royalblue', alpha=0.5)
    bfield_df.plot(x="Time_s", y="RMSE_BC", ax=ax3, style='-', linewidth=2, color='magenta')

    ax1.set_ylabel("nT", fontsize=16)
    ax1.set_xlabel("Time (seconds)", fontsize=16)
    ax2.set_xlabel("Time (seconds)", fontsize=16)
    ax3.set_xlabel("Time (seconds)", fontsize=16)

    ax1.set_title("(a.) Uncorrected |B|", fontsize=20)
    ax2.set_title("(b.) Corrected |B|", fontsize=20)
    ax3.set_title("(c.) RMS Error", fontsize=20)

    ax1.legend(['Uncorrected |B|', 'True |B|'], fontsize='medium', ncol=2, loc='lower left')
    ax2.legend(['Corrected |B|', 'True |B|'], fontsize='medium', ncol=2, loc='lower left')
    _add_residual_target(ax3)

    if '500km' in test:
        plt.suptitle("WMM-2025 Simulation", fontsize=20)
    if '10000km' in test:
        plt.suptitle("Earth 10,000 km altitude", fontsize=20)
    if 'log00004' in test:
        plt.suptitle("Laboratory test #2", fontsize=20)
    if 'log00003' in test:
        plt.suptitle("Laboratory test #1", fontsize=20)
    if 'am285_imf45_rotated_B' in test:
        plt.suptitle("Lunar Vertex Landing Simulation", fontsize=20)
    if 'sine' in test:
        plt.suptitle(r"$\text{Sine Wave: }A=0.5\text{ nT, }f=0.05\text{ Hz, }\varphi=30^{\circ}$", fontsize=20)
    if 'swarm' in test:
        plt.suptitle("SWARM Simulation", fontsize=20)

    ymin1, ymax1 = ax1.get_ylim()

    if 'earth' not in test and 'swarm' not in test:
        ymin_raw = 0 - math.floor(np.abs(ymax1) * 0.25)
        ymax_raw = math.ceil(ymax1 + ymax1 * 0.25)
    elif '500km' in test:
        ymin_raw = 39500
        ymax_raw = 39900
    elif '10000km' in test:
        ymin_raw = 2450
        ymax_raw = 2900
    elif 'swarm' in test:
        ymin_raw = 19500
        ymax_raw = 23000
    else:
        ymin_raw = 39000
        ymax_raw = 40000

    ymin2, ymax2 = ax2.get_ylim()

    if 'earth' not in test and 'log' not in test and 'am285_imf45_rotated_B' not in test and 'sine' not in test and 'swarm' not in test:
        ymin_corr = 0. - math.floor(np.abs(ymax1) * 0.25)
        ymax_corr = math.ceil(ymax2 + ymax2 * 0.25)
    elif 'earth' in test and '500km' in test:
        ymin_corr = 39566
        ymax_corr = 39572
    elif 'earth' in test and '10000km' in test:
        ymin_corr = 2782
        ymax_corr = 2792
    elif 'earth' not in test and 'log00003' in test:
        ymin_corr = 250
        ymax_corr = 350
    elif 'earth' not in test and 'log00004' in test:
        ymin_corr = 175
        ymax_corr = 350
    elif 'am285_imf45_rotated_B' in test:
        ymin_corr = -20
        ymax_corr = 100
    elif 'sine' in test:
        ymin_corr = -5
        ymax_corr = 5
    elif 'swarm' in test:
        ymin_corr = 19500
        ymax_corr = 23000
    else:
        ymin_corr = 39000
        ymax_corr = 40000

    ymin3, ymax3 = ax3.get_ylim()

    if 'earth' not in test and 'log' not in test and 'swarm' not in test:
        ymin_rmse = -0.05 - math.floor(np.abs(ymax3) * 0.25)
        ymax_rmse = 0.05 + math.ceil(ymax3 + ymax3 * 0.25)
    elif 'earth' not in test and 'log00003' in test:
        ymin_rmse = -0.5
        ymax_rmse = 30
    elif 'earth' not in test and 'log00004' in test:
        ymin_rmse = -0.5
        ymax_rmse = 30
    elif 'swarm' in test:
        ymin_rmse = -1
        ymax_rmse = 60
    else:
        ymin_rmse = -0.5
        ymax_rmse = 5

    ax1.set_ylim(ymin_raw, ymax_raw)
    ax2.set_ylim(ymin_corr, ymax_corr)
    ax3.set_ylim(ymin_rmse, ymax_rmse)

    for ax in [ax1, ax2, ax3]:
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))

    fig.tight_layout()
    correction_mag_output = str(os.path.join(input_dir, test + "_center_mag_correction_error.png"))
    fig.savefig(correction_mag_output, format='png', dpi=300)
    plt.close()
    # plt.show()


def max_rmse_plot(n, max_rmse, input_dir, test):
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    iters = np.arange(0, n, 1, dtype=int)

    ax.plot(iters, max_rmse, marker='o', color='purple', linestyle='dashed', linewidth=2)

    ax.set_title("Maximum RMSE Per Iteration", fontsize=20)
    ax.set_ylabel("nT")
    ax.set_xlabel("Iteration")

    # ax.set_ylim(ymin_raw, ymax_raw)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))

    fig.tight_layout()
    correction_mag_output = str(os.path.join(input_dir, test + "_max_rmse_iter.png"))
    fig.savefig(correction_mag_output, format='png', dpi=300)
    plt.close()
    # plt.show()


def dist_plot(dist_list, angular_diff_list1, angular_diff_list2):
    """
    Plots angular difference as a function of distance
    """

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.plot(dist_list, angular_diff_list1, color='tab:cyan', label='XYZ')
    # ax.plot(dist_list, angular_diff_list2, color='red', label='re1_hat')
    ax.yaxis.set_tick_params(labelsize='x-large')
    ax.xaxis.set_tick_params(labelsize='x-large')
    plt.legend(loc='upper right')
    plt.xlabel("Distance (meters)", fontsize=20)
    plt.ylabel("Angular difference", fontsize=20)
    plt.suptitle("Angular difference between truth vector and initial estimate vectors", fontsize=20)
    plt.title("Varying r_s distance, mu_s_hat = [0 1 0]", fontsize=16, y=1.025)
    plt.show()


def cost_func_plot(cf_vals, cf_e0_val, nsteps, final_cf_val, time_taken):
    """
    Plots angular difference as a function of distance
    """
    numplot = len(cf_vals)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(0, cf_e0_val, c='m', s=25, label='Initial')
    ax.scatter(range(1, numplot + 1), cf_vals, c='g', s=15, alpha=0.25, label='Optimizing')
    ax.scatter(nsteps, final_cf_val, c='b', s=25, label='Final')
    ax.set_yscale("log")
    ax.yaxis.set_tick_params(labelsize='medium')
    ax.xaxis.set_tick_params(labelsize='medium')
    plt.legend(loc='upper right')
    plt.xlabel("Iterations", fontsize=16)
    plt.ylabel("log(Cost function value)", fontsize=16)
    plt.title("Total time taken: " + str(round(time_taken, 3) * 1000) + " ms", fontsize=20)
    plt.show()


def vector_plot(vector, rotation):
    """
    Plots original vs rotated vector in mast reference frame
    """
    if rotation == 'B1':
        new_vect = MathUtil.B1_transform(vector)
    elif rotation == 'B2':
        new_vect = MathUtil.B2_transform(vector)
    elif rotation == 'B3':
        new_vect = MathUtil.B3_transform(vector)
    elif rotation == 'B4':
        new_vect = MathUtil.B4_transform(vector)
    else:
        new_vect = vector

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.quiver([0], [0], [0], [1], [0], [0], linestyle='dashed', colors='k')
    ax.quiver([0], [0], [0], [0], [1], [0], linestyle='dashed', colors='k')
    ax.quiver([0], [0], [0], [0], [0], [1], linestyle='dashed', colors='k')
    ax.plot(np.linspace(0, vector[0]), np.linspace(0, vector[1]), np.linspace(0, vector[2]), color='green', label='old')
    ax.plot(np.linspace(0, new_vect[0]), np.linspace(0, new_vect[1]), np.linspace(0, new_vect[2]), color='red', label='new')
    plt.legend(ncol=1, loc='center left', bbox_to_anchor=(1, 1), fancybox=True)
    plt.show()


def animate_vector(vector1, vector2, fname, outputdir, textstr1=None, textstr2=None):
    """
    Animates change in theta of vector at thruster test positions
    """

    if "_" in fname:
        fstring = fname.replace("_", " ")
    else:
        fstring = fname

    rot_angles = [0, 30, 60, 90]
    fig, ax = plt.subplots(subplot_kw=dict(projection="3d"), figsize=(7, 5))
    plt.figtext(0.05, 0.5, textstr1, fontsize=14, color='darkorange')
    plt.figtext(0.05, 0.25, textstr2, fontsize=14, color='cornflowerblue')
    # textstr3 = 'Truth\n\u03B8\u2081=%.0f\u1D52\n\u03B8\u2082=%.0f\u1D52\n\u03B8\u2083=%.0f\u1D52\n' % (30, 60, 90)
    # plt.figtext(0.05, 0.75, textstr3, fontsize=14, color='black')
    ax.set_xlim(-1, 1)
    ax.set_xticks(ax.get_xticks()[::2])
    ax.set_ylim(1, -1)
    ax.set_yticks(ax.get_yticks()[::2])
    ax.set_zlim(-1, 1)
    ax.set_zticks(ax.get_zticks()[::2])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(25, -45)

    def get_arrow1(num):
        x = 0
        y = 0
        z = 0
        u, v, w = np.hsplit(vector1[num], 3)
        return x, y, z, u, v, w

    def get_arrow2(num):
        x = 0
        y = 0
        z = 0
        u, v, w = np.hsplit(vector2[num], 3)
        return x, y, z, u, v, w

    def get_xvec(num):
        x = 0
        y = 0
        z = 0
        u, v, w = MathUtil.z_rotation(np.array([1.0, 0.0, 0.0]), math.radians(rot_angles[num]))
        return x, y, z, u, v, w

    def get_yvec(num):
        x = 0
        y = 0
        z = 0
        u, v, w = MathUtil.z_rotation(np.array([0.0, 1.0, 0.0]), math.radians(rot_angles[num]))
        return x, y, z, u, v, w

    ax.quiver([0], [0], [0], [1], [0], [0], linestyle='dashed', colors='k')
    ax.quiver([0], [0], [0], [0], [1], [0], linestyle='dashed', colors='k')
    ax.quiver([0], [0], [0], [0], [0], [1], linestyle='dashed', colors='k')
    q1 = ax.quiver(*get_arrow1(0), colors='darkorange')
    q2 = ax.quiver(*get_arrow2(0), colors='cornflowerblue')

    def update(theta):
        ax.clear()
        plt.title((r"Estimate $\hat{r}$ from " + fstring), fontsize=14)
        ax.set_xlim(-1, 1)
        ax.set_xticks(ax.get_xticks()[::2])
        ax.set_ylim(1, -1)
        ax.set_yticks(ax.get_yticks()[::2])
        ax.set_zlim(-1, 1)
        ax.set_zticks(ax.get_zticks()[::2])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        # a1 = ax.quiver(*get_xvec(theta), linestyle='dashed', colors='k')
        # a2 = ax.quiver(*get_yvec(theta), linestyle='dashed', colors='k')
        ax.quiver([0], [0], [0], [1], [0], [0], linestyle='dashed', colors='k')
        ax.quiver([0], [0], [0], [0], [1], [0], linestyle='dashed', colors='k')
        ax.quiver([0], [0], [0], [0], [0], [1], linestyle='dashed', colors='k')
        q1 = ax.quiver(*get_arrow1(theta), colors='darkorange')
        q2 = ax.quiver(*get_arrow2(theta), colors='cornflowerblue')

    ani = FuncAnimation(fig, update, frames=range(len(vector1)), interval=100)
    ani.save((outputdir + fname + "_rhat.gif"), writer='pillow', fps=2)
    plt.show()
