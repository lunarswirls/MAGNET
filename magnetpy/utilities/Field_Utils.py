#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Contains static methods for generating test fields that can be used as inputs for MAGNET algorithm.
"""
# Imports
import datetime
import math
import sys
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import magnetpy.utilities.Math_Utils as MathUtils
import magnetpy.gradiometry_algorithm.tetrahedral_correction as tc
import magnetpy.utilities.Thruster_Utils as Thruster


def generate_earth_field(alt):
    """
    Generates an Earth-like field at a fixed altitude [500 km or 10,000 km], then overlaying thruster noise profile.
    Returns dataframe of new magnetic field and a dataframe of thruster profile.

    This uses WMM-2025 data for Earth's magnetic field in Laurel, MD.
    See https://www.ngdc.noaa.gov/geomag/calculators/magcalc.shtml#igrfwmm
    Latitude: 	39.0993° N
    Longitude: 	76.8483° W
    Date:       2025-11-04
    B_north (+ N | - S)
    B_east (+ E | - W)
    B_vert (+ D | - U)
    """
    # generate time array
    time_step = 5000
    time_arr = np.linspace(0.0, 500.0, time_step)  # units seconds
    start_time = datetime.datetime(2025, 1, 1, 0, 0, 0)
    end_time = start_time + datetime.timedelta(seconds=500)
    times = pd.date_range(start_time, end_time, periods=time_step).to_list()

    if alt == '500' or alt == 500:
        # 500 km altitude
        # in sensor frame of reference, down is up
        bg_field = np.array([16576.5e-9, -2853.7e-9, -35815.8e-9])  # units Tesla
    elif alt == '10000' or alt == 10000:
        # 10,000 km altitude
        # in sensor frame of reference, down is up
        bg_field = np.array([1193.6e-9, -73.4e-9, -2518.0e-9])  # units Tesla
    else:
        print(f'Invalid altitude: {alt}')
        sys.exit("Please choose between '500' or '10000' altitude (km)")

    # create background field based on time sequence
    b1_field = [bg_field] * time_step  # units Tesla
    b2_field = [bg_field] * time_step  # units Tesla
    b3_field = [bg_field] * time_step  # units Tesla
    b4_field = [bg_field] * time_step  # units Tesla
    bg_field = [bg_field] * time_step  # units Tesla

    # create thruster state based on time sequence
    thruster_state = [np.array([0, 0, 0, 0])] * time_step

    B1_s1, B2_s1, B3_s1, B4_s1, B1_s2, B2_s2, B3_s2, B4_s2, B1_s3, B2_s3, B3_s3, B4_s3, B1_s4, B2_s4, B3_s4, B4_s4 = Thruster.base_profile()

    # add thruster 1 noise to background field
    b1_field[115:280] += B1_s1
    b2_field[115:280] += B2_s1
    b3_field[115:280] += B3_s1
    b4_field[115:280] += B4_s1

    # add thruster 2 noise to background field
    b1_field[475:650] += B1_s2
    b2_field[475:650] += B2_s2
    b3_field[475:650] += B3_s2
    b4_field[475:650] += B4_s2

    # add thruster 3 noise to background field
    b1_field[880:1100] += B1_s3
    b2_field[880:1100] += B2_s3
    b3_field[880:1100] += B3_s3
    b4_field[880:1100] += B4_s3

    # add thruster 4 noise to background field
    b1_field[1350:1575] += B1_s4
    b2_field[1350:1575] += B2_s4
    b3_field[1350:1575] += B3_s4
    b4_field[1350:1575] += B4_s4

    # add thruster 1 noise to background field
    b1_field[1865:1990] += B1_s1
    b2_field[1865:1990] += B2_s1
    b3_field[1865:1990] += B3_s1
    b4_field[1865:1990] += B4_s1

    # add thruster 1 + 3 noise to background field
    b1_field[1990:2140] += (B1_s1 + B1_s3)
    b2_field[1990:2140] += (B2_s1 + B2_s3)
    b3_field[1990:2140] += (B3_s1 + B3_s3)
    b4_field[1990:2140] += (B4_s1 + B4_s3)

    # add thruster 1 noise to background field
    b1_field[2355:2595] += B1_s1
    b2_field[2355:2595] += B2_s1
    b3_field[2355:2595] += B3_s1
    b4_field[2355:2595] += B4_s1

    # add thruster 2 + 3 + 4 noise to background field
    b1_field[2355:2630] += (B1_s2 + B1_s3 + B1_s4)
    b2_field[2355:2630] += (B2_s2 + B2_s3 + B2_s4)
    b3_field[2355:2630] += (B3_s2 + B3_s3 + B3_s4)
    b4_field[2355:2630] += (B4_s2 + B4_s3 + B4_s4)

    # add thruster 1 noise to background field
    b1_field[2860:3230] += B1_s1
    b2_field[2860:3230] += B2_s1
    b3_field[2860:3230] += B3_s1
    b4_field[2860:3230] += B4_s1

    # add thruster 2 noise to background field
    b1_field[2920:3350] += B1_s2
    b2_field[2920:3350] += B2_s2
    b3_field[2920:3350] += B3_s2
    b4_field[2920:3350] += B4_s2

    # add thruster 4 noise to background field
    b1_field[2995:3410] += B1_s4
    b2_field[2995:3410] += B2_s4
    b3_field[2995:3410] += B3_s4
    b4_field[2995:3410] += B4_s4

    # add thruster 3 noise to background field
    b1_field[3105:3505] += B1_s3
    b2_field[3105:3505] += B2_s3
    b3_field[3105:3505] += B3_s3
    b4_field[3105:3505] += B4_s3

    # add thruster 1+2+3+4 noise to background field
    b1_field[3915:4245] += (B1_s1 + B1_s2 + B1_s3 + B1_s4)
    b2_field[3915:4245] += (B2_s1 + B2_s2 + B2_s3 + B2_s4)
    b3_field[3915:4245] += (B3_s1 + B3_s2 + B3_s3 + B3_s4)
    b4_field[3915:4245] += (B4_s1 + B4_s2 + B4_s3 + B4_s4)

    # add thruster 1+2+3+4 noise to background field
    b1_field[4615:4705] += (B1_s1 + B1_s2 + B1_s3 + B1_s4)
    b2_field[4615:4705] += (B2_s1 + B2_s2 + B2_s3 + B2_s4)
    b3_field[4615:4705] += (B3_s1 + B3_s2 + B3_s3 + B3_s4)
    b4_field[4615:4705] += (B4_s1 + B4_s2 + B4_s3 + B4_s4)

    # add thruster 1+2+3+4 to state file
    thruster_state[115:280] += np.array([1, 0, 0, 0])
    thruster_state[475:650] += np.array([0, 1, 0, 0])
    thruster_state[880:1100] += np.array([0, 0, 1, 0])
    thruster_state[1350:1575] += np.array([0, 0, 0, 1])
    thruster_state[1865:1990] += np.array([1, 0, 0, 0])
    thruster_state[1990:2140] += np.array([1, 0, 1, 0])
    thruster_state[2355:2595] += np.array([1, 0, 0, 0])
    thruster_state[2355:2630] += np.array([0, 1, 1, 1])
    thruster_state[2860:3230] += np.array([1, 0, 0, 0])
    thruster_state[2920:3350] += np.array([0, 1, 0, 0])
    thruster_state[2995:3410] += np.array([0, 0, 0, 1])
    thruster_state[3105:3505] += np.array([0, 0, 1, 0])
    thruster_state[3815:4245] += np.array([1, 1, 1, 1])
    thruster_state[4615:4705] += np.array([1, 1, 1, 1])

    bg_field_nT = np.array(bg_field) * 1e9  # units nT
    bg_mag = np.sqrt(np.array(bg_field_nT)[:, 0] ** 2 + np.array(bg_field_nT)[:, 1] ** 2 + np.array(bg_field_nT)[:, 2] ** 2)  # units nT

    bfield_df = pd.DataFrame(np.array(b1_field))
    bfield_df.columns = ['B1X', 'B1Y', 'B1Z']

    bfield_df['B2X'] = np.array(b2_field)[:, 0]
    bfield_df['B2Y'] = np.array(b2_field)[:, 1]
    bfield_df['B2Z'] = np.array(b2_field)[:, 2]
    bfield_df['B3X'] = np.array(b3_field)[:, 0]
    bfield_df['B3Y'] = np.array(b3_field)[:, 1]
    bfield_df['B3Z'] = np.array(b3_field)[:, 2]
    bfield_df['B4X'] = np.array(b4_field)[:, 0]
    bfield_df['B4Y'] = np.array(b4_field)[:, 1]
    bfield_df['B4Z'] = np.array(b4_field)[:, 2]

    bfield_df['BGX'] = np.array(bg_field_nT)[:, 0]  # units nT
    bfield_df['BGY'] = np.array(bg_field_nT)[:, 1]  # units nT
    bfield_df['BGZ'] = np.array(bg_field_nT)[:, 2]  # units nT

    bfield_df['BG_mag'] = bg_mag
    bfield_df['time_raw'] = time_arr
    bfield_df['time_fmt'] = times

    thruster_df = pd.DataFrame(thruster_state)
    thruster_df.columns = ['Thruster 1', 'Thruster 2', 'Thruster 3', 'Thruster 4']
    thruster_df['Time'] = time_arr

    return bfield_df, thruster_df


def generate_amitis_field(df):
    """
    Given Amitis output data, generates a MAGNET test field by overlaying thruster noise profile.
    Returns dataframe of new magnetic field and a dataframe of thruster profile.

    :param df: Pandas DataFrame containing elapsed_time (seconds) and magnetic field in spacecraft frame of reference
        B_sc_mag, B_sc_x, B_sc_y, B_sc_z (all in nT)
    """
    # start time
    start_time = datetime.datetime(2025, 1, 1, 0, 0, 0)
    # Parse reference datetime
    ref_dt = pd.to_datetime(start_time, format='%m/%d/%Y %H:%M:%S.%f')

    # Convert elapsed_time to time deltas
    time_deltas = pd.to_timedelta(df['elapsed_time'], unit='s')

    df = df.copy()

    # Combine date part from reference with each time
    df['Time'] = ref_dt.normalize() + time_deltas
    # Get total seconds
    df['Seconds'] = (df['Time'] - df['Time'].iloc[0]).dt.total_seconds()

    df = df.set_index('Time')
    # resample to 10 Hz (VML)
    df = df.resample(rule='0.1s').mean().interpolate(method='linear')

    df = df.reset_index()

    time_arr = df['Seconds'].to_numpy()  # units seconds
    times = df['Time'].to_numpy()  # units seconds

    # simulation coordinates rotated into sc frame of reference
    bg_field_nT = df[['B_sc_x', 'B_sc_y', 'B_sc_z']].to_numpy()  # units nT
    bg_mag = df['B_sc_mag'].to_numpy()  # units nT

    bfield_df, thruster_df = Thruster.generate_thruster_bfield_data(time_arr, times, bg_field_nT, bg_mag)

    return bfield_df, thruster_df


def generate_swarm_field(df_in):
    """
    Given SWARM input data, generates a MAGNET test field by overlaying thruster noise profile.
    Returns dataframe of new magnetic field and a dataframe of thruster profile.

    :param df_in: Pandas DataFrame containing swa_time (datetime) and Earth field swa_bgeo_x, swa_bgeo_y, swa_bgeo_z, swa_btot (all in nT)
    """

    # Convert time column
    df_in['Time'] = pd.to_datetime(df_in["swa_time"], format="%Y-%m-%d/%H:%M:%S.%f")

    # Get total seconds
    df_in['Seconds'] = (df_in['Time'] - df_in['Time'].iloc[0]).dt.total_seconds()

    df_in.drop('swa_time', axis=1, inplace=True)

    df = df_in[df_in['Seconds'] < 500]

    df = df.set_index('Time')
    # resample to 10 Hz (max VML cadence)
    df = df.resample(rule='0.1s').mean().interpolate(method='linear')

    df = df.reset_index()

    time_arr = df['Seconds'].to_numpy()  # units seconds
    times = df['Time'].to_numpy()  # units seconds

    # SWARM data as background field
    bg_field_nT = df[['swa_bgeo_x', 'swa_bgeo_y', 'swa_bgeo_z']].to_numpy()  # units nT
    bg_mag = df['swa_btot'].to_numpy()  # units nT

    # create background field at each sensor
    b1_field = bg_field_nT * 1.e-9  # units Tesla
    b2_field = bg_field_nT * 1.e-9  # units Tesla
    b3_field = bg_field_nT * 1.e-9  # units Tesla
    b4_field = bg_field_nT * 1.e-9  # units Tesla

    # create thruster state based on time sequence
    thruster_state = [np.array([0, 0, 0, 0])] * len(bg_mag)

    B1_s1, B2_s1, B3_s1, B4_s1, B1_s2, B2_s2, B3_s2, B4_s2, B1_s3, B2_s3, B3_s3, B4_s3, B1_s4, B2_s4, B3_s4, B4_s4 = Thruster.base_profile()

    # add thruster 1 noise to background field
    b1_field[115:280] += B1_s1
    b2_field[115:280] += B2_s1
    b3_field[115:280] += B3_s1
    b4_field[115:280] += B4_s1

    # add thruster 2 noise to background field
    b1_field[475:650] += B1_s2
    b2_field[475:650] += B2_s2
    b3_field[475:650] += B3_s2
    b4_field[475:650] += B4_s2

    # add thruster 3 noise to background field
    b1_field[880:1100] += B1_s3
    b2_field[880:1100] += B2_s3
    b3_field[880:1100] += B3_s3
    b4_field[880:1100] += B4_s3

    # add thruster 4 noise to background field
    b1_field[1350:1575] += B1_s4
    b2_field[1350:1575] += B2_s4
    b3_field[1350:1575] += B3_s4
    b4_field[1350:1575] += B4_s4

    # add thruster 1 noise to background field
    b1_field[1865:1990] += B1_s1
    b2_field[1865:1990] += B2_s1
    b3_field[1865:1990] += B3_s1
    b4_field[1865:1990] += B4_s1

    # add thruster 1 + 3 noise to background field
    b1_field[1990:2140] += (B1_s1 + B1_s3)
    b2_field[1990:2140] += (B2_s1 + B2_s3)
    b3_field[1990:2140] += (B3_s1 + B3_s3)
    b4_field[1990:2140] += (B4_s1 + B4_s3)

    # add thruster 1 noise to background field
    b1_field[2355:2595] += B1_s1
    b2_field[2355:2595] += B2_s1
    b3_field[2355:2595] += B3_s1
    b4_field[2355:2595] += B4_s1

    # add thruster 2 + 3 + 4 noise to background field
    b1_field[2355:2630] += (B1_s2 + B1_s3 + B1_s4)
    b2_field[2355:2630] += (B2_s2 + B2_s3 + B2_s4)
    b3_field[2355:2630] += (B3_s2 + B3_s3 + B3_s4)
    b4_field[2355:2630] += (B4_s2 + B4_s3 + B4_s4)

    # add thruster 1 noise to background field
    b1_field[2860:3230] += B1_s1
    b2_field[2860:3230] += B2_s1
    b3_field[2860:3230] += B3_s1
    b4_field[2860:3230] += B4_s1

    # add thruster 2 noise to background field
    b1_field[2920:3350] += B1_s2
    b2_field[2920:3350] += B2_s2
    b3_field[2920:3350] += B3_s2
    b4_field[2920:3350] += B4_s2

    # add thruster 4 noise to background field
    b1_field[2995:3410] += B1_s4
    b2_field[2995:3410] += B2_s4
    b3_field[2995:3410] += B3_s4
    b4_field[2995:3410] += B4_s4

    # add thruster 3 noise to background field
    b1_field[3105:3505] += B1_s3
    b2_field[3105:3505] += B2_s3
    b3_field[3105:3505] += B3_s3
    b4_field[3105:3505] += B4_s3

    # add thruster 1+2+3+4 noise to background field
    b1_field[3915:4245] += (B1_s1 + B1_s2 + B1_s3 + B1_s4)
    b2_field[3915:4245] += (B2_s1 + B2_s2 + B2_s3 + B2_s4)
    b3_field[3915:4245] += (B3_s1 + B3_s2 + B3_s3 + B3_s4)
    b4_field[3915:4245] += (B4_s1 + B4_s2 + B4_s3 + B4_s4)

    # add thruster 1+2+3+4 noise to background field
    b1_field[4615:4705] += (B1_s1 + B1_s2 + B1_s3 + B1_s4)
    b2_field[4615:4705] += (B2_s1 + B2_s2 + B2_s3 + B2_s4)
    b3_field[4615:4705] += (B3_s1 + B3_s2 + B3_s3 + B3_s4)
    b4_field[4615:4705] += (B4_s1 + B4_s2 + B4_s3 + B4_s4)

    # add thruster 1+2+3+4 to state file
    thruster_state[115:280] += np.array([1, 0, 0, 0])
    thruster_state[475:650] += np.array([0, 1, 0, 0])
    thruster_state[880:1100] += np.array([0, 0, 1, 0])
    thruster_state[1350:1575] += np.array([0, 0, 0, 1])
    thruster_state[1865:1990] += np.array([1, 0, 0, 0])
    thruster_state[1990:2140] += np.array([1, 0, 1, 0])
    thruster_state[2355:2595] += np.array([1, 0, 0, 0])
    thruster_state[2355:2630] += np.array([0, 1, 1, 1])
    thruster_state[2860:3230] += np.array([1, 0, 0, 0])
    thruster_state[2920:3350] += np.array([0, 1, 0, 0])
    thruster_state[2995:3410] += np.array([0, 0, 0, 1])
    thruster_state[3105:3505] += np.array([0, 0, 1, 0])
    thruster_state[3815:4245] += np.array([1, 1, 1, 1])
    thruster_state[4615:4705] += np.array([1, 1, 1, 1])

    bfield_df = pd.DataFrame(np.array(b1_field))
    bfield_df.columns = ['B1X', 'B1Y', 'B1Z']

    bfield_df['B2X'] = np.array(b2_field)[:, 0]
    bfield_df['B2Y'] = np.array(b2_field)[:, 1]
    bfield_df['B2Z'] = np.array(b2_field)[:, 2]
    bfield_df['B3X'] = np.array(b3_field)[:, 0]
    bfield_df['B3Y'] = np.array(b3_field)[:, 1]
    bfield_df['B3Z'] = np.array(b3_field)[:, 2]
    bfield_df['B4X'] = np.array(b4_field)[:, 0]
    bfield_df['B4Y'] = np.array(b4_field)[:, 1]
    bfield_df['B4Z'] = np.array(b4_field)[:, 2]

    bfield_df['BGX'] = np.array(bg_field_nT)[:, 0]  # units nT
    bfield_df['BGY'] = np.array(bg_field_nT)[:, 1]  # units nT
    bfield_df['BGZ'] = np.array(bg_field_nT)[:, 2]  # units nT

    bfield_df['BG_mag'] = bg_mag
    bfield_df['time_raw'] = time_arr
    bfield_df['time_fmt'] = times

    thruster_df = pd.DataFrame(thruster_state)
    thruster_df.columns = ['Thruster 1', 'Thruster 2', 'Thruster 3', 'Thruster 4']
    thruster_df['Time'] = time_arr

    return bfield_df, thruster_df


def benchmark_test_field():
    """
    Generates a test field that can be used as a benchmark for MAGNET algorithm.
    """
    # generate time array
    time_step = 10  # Hz (max sampling rate of VML)
    time_len = 200  # total seconds
    time_arr = np.linspace(0.0, time_len, time_step*time_len)  # units seconds
    start_time = datetime.datetime(2025, 1, 1, 0, 0, 0)
    end_time = start_time + datetime.timedelta(seconds=time_len)
    times = pd.date_range(start_time, end_time, periods=len(time_arr)).to_list()

    bg_field = np.array([0.0, 0.0, 0.0])  # units Tesla

    # create background field based on time sequence
    b1_field = [bg_field] * len(time_arr)  # units Tesla
    b2_field = [bg_field] * len(time_arr)  # units Tesla
    b3_field = [bg_field] * len(time_arr)  # units Tesla
    b4_field = [bg_field] * len(time_arr)  # units Tesla
    bg_field = [bg_field] * len(time_arr)  # units Tesla

    # create thruster state based on time sequence
    thruster_state = [np.array([0, 0, 0, 0])] * time_step

    B1_s1, B2_s1, B3_s1, B4_s1, B1_s2, B2_s2, B3_s2, B4_s2, B1_s3, B2_s3, B3_s3, B4_s3, B1_s4, B2_s4, B3_s4, B4_s4 = Thruster.base_profile()

    # add thruster 1 noise to background field
    b1_field[115:280] += B1_s1
    b2_field[115:280] += B2_s1
    b3_field[115:280] += B3_s1
    b4_field[115:280] += B4_s1

    # add thruster 2 noise to background field
    b1_field[475:650] += B1_s2
    b2_field[475:650] += B2_s2
    b3_field[475:650] += B3_s2
    b4_field[475:650] += B4_s2

    # add thruster 3 noise to background field
    b1_field[880:1100] += B1_s3
    b2_field[880:1100] += B2_s3
    b3_field[880:1100] += B3_s3
    b4_field[880:1100] += B4_s3

    # add thruster 4 noise to background field
    b1_field[1350:1575] += B1_s4
    b2_field[1350:1575] += B2_s4
    b3_field[1350:1575] += B3_s4
    b4_field[1350:1575] += B4_s4

    # add thruster 1+2+3+4 to state file
    thruster_state[115:280] += np.array([1, 0, 0, 0])
    thruster_state[475:650] += np.array([0, 1, 0, 0])
    thruster_state[880:1100] += np.array([0, 0, 1, 0])
    thruster_state[1350:1575] += np.array([0, 0, 0, 1])

    bg_field_nT = np.array(bg_field) * 1e9  # units nT
    bg_mag = np.sqrt(np.array(bg_field_nT)[:, 0] ** 2 + np.array(bg_field_nT)[:, 1] ** 2 + np.array(bg_field_nT)[:, 2] ** 2)  # units nT

    bfield_df = pd.DataFrame(np.array(b1_field))
    bfield_df.columns = ['B1X', 'B1Y', 'B1Z']

    bfield_df['B2X'] = np.array(b2_field)[:, 0]
    bfield_df['B2Y'] = np.array(b2_field)[:, 1]
    bfield_df['B2Z'] = np.array(b2_field)[:, 2]
    bfield_df['B3X'] = np.array(b3_field)[:, 0]
    bfield_df['B3Y'] = np.array(b3_field)[:, 1]
    bfield_df['B3Z'] = np.array(b3_field)[:, 2]
    bfield_df['B4X'] = np.array(b4_field)[:, 0]
    bfield_df['B4Y'] = np.array(b4_field)[:, 1]
    bfield_df['B4Z'] = np.array(b4_field)[:, 2]

    bfield_df['BGX'] = np.array(bg_field_nT)[:, 0]  # units nT
    bfield_df['BGY'] = np.array(bg_field_nT)[:, 1]  # units nT
    bfield_df['BGZ'] = np.array(bg_field_nT)[:, 2]  # units nT

    bfield_df['BG_mag'] = bg_mag
    bfield_df['time_raw'] = time_arr
    bfield_df['time_fmt'] = times

    thruster_df = pd.DataFrame(thruster_state)
    thruster_df.columns = ['Thruster 1', 'Thruster 2', 'Thruster 3', 'Thruster 4']
    thruster_df['Time'] = time_arr

    return bfield_df, thruster_df


def generate_noisy_field(x_axis, y_axis, z_axis):
    """
    Generates a test field that contains thruster fires with random jitter noise applied.
    :param x_axis: X-axis noise level in %
    :param y_axis: Y-axis noise level in %
    :param z_axis: Z-axis noise level in %
    """
    # generate time array
    time_step = 10  # Hz (max sampling rate of VML)
    time_len = 200  # total seconds
    time_arr = np.linspace(0.0, time_len, time_step*time_len)  # units seconds
    start_time = datetime.datetime(2025, 1, 1, 0, 0, 0)
    end_time = start_time + datetime.timedelta(seconds=time_len)
    times = pd.date_range(start_time, end_time, periods=len(time_arr)).to_list()

    bg_field = np.array([0.0, 0.0, 0.0])  # units Tesla

    # create background field based on time sequence
    b1_field = [bg_field] * len(time_arr)  # units Tesla
    b2_field = [bg_field] * len(time_arr)  # units Tesla
    b3_field = [bg_field] * len(time_arr)  # units Tesla
    b4_field = [bg_field] * len(time_arr)  # units Tesla
    bg_field = [bg_field] * len(time_arr)  # units Tesla

    # create thruster state based on time sequence
    thruster_state = [np.array([0, 0, 0, 0])] * len(time_arr)

    # average thruster moments from lab testing
    mu_s1 = np.array([-0.2, 0.3, -0.75]) * 3.5
    mu_s_mag1 = np.linalg.norm(mu_s1, ord=2, axis=0)  # units [A*m^2]
    mu_s_hat1 = mu_s1 / mu_s_mag1

    mu_s2 = np.array([-0.2, 0.3, -0.75]) * 3.2
    mu_s_mag2 = np.linalg.norm(mu_s2, ord=2, axis=0)  # units [A*m^2]
    mu_s_hat2 = mu_s2 / mu_s_mag2

    mu_s3 = np.array([-0.2, 0.3, -0.75]) * 3.7
    mu_s_mag3 = np.linalg.norm(mu_s3, ord=2, axis=0)  # units [A*m^2]
    mu_s_hat3 = mu_s3 / mu_s_mag3

    mu_s4 = np.array([0.2, 0.3, -0.75]) * 3.5
    mu_s_mag4 = np.linalg.norm(mu_s4, ord=2, axis=0)
    mu_s_hat4 = mu_s4 / mu_s_mag4

    # thruster 1 location
    r_s = np.array([0.75, 1.5, 0])
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
    [B1_s2, B2_s2, B3_s2, B4_s2] = tc.mag_array(mu_s_mag3, mu_s_hat3, r_s_mag, r_s2_hat)  # units Tesla
    # get magnetic field from thruster 3
    [B1_s3, B2_s3, B3_s3, B4_s3] = tc.mag_array(mu_s_mag2, mu_s_hat2, r_s_mag, r_s3_hat)  # units Tesla
    # get magnetic field from thruster 4
    [B1_s4, B2_s4, B3_s4, B4_s4] = tc.mag_array(mu_s_mag4, mu_s_hat4, r_s_mag, r_s4_hat)  # units Tesla

    # add thruster 1 noise to background field
    t = time_arr[115:280]
    n = len(t)
    b1_field[115:280] += add_time_noise(B1_s1, x_axis, y_axis, z_axis, n)
    b2_field[115:280] += add_time_noise(B2_s1, x_axis, y_axis, z_axis, n)
    b3_field[115:280] += add_time_noise(B3_s1, x_axis, y_axis, z_axis, n)
    b4_field[115:280] += add_time_noise(B4_s1, x_axis, y_axis, z_axis, n)

    # add thruster 2 noise to background field
    t = time_arr[475:650]
    n = len(t)
    b1_field[475:650] += add_time_noise(B1_s2, x_axis, y_axis, z_axis, n)
    b2_field[475:650] += add_time_noise(B2_s2, x_axis, y_axis, z_axis, n)
    b3_field[475:650] += add_time_noise(B3_s2, x_axis, y_axis, z_axis, n)
    b4_field[475:650] += add_time_noise(B4_s2, x_axis, y_axis, z_axis, n)

    # add thruster 3 noise to background field
    t = time_arr[880:1100]
    n = len(t)
    b1_field[880:1100] += add_time_noise(B1_s3, x_axis, y_axis, z_axis, n)
    b2_field[880:1100] += add_time_noise(B2_s3, x_axis, y_axis, z_axis, n)
    b3_field[880:1100] += add_time_noise(B3_s3, x_axis, y_axis, z_axis, n)
    b4_field[880:1100] += add_time_noise(B4_s3, x_axis, y_axis, z_axis, n)

    # add thruster 4 noise to background field
    t = time_arr[1350:1575]
    n = len(t)
    b1_field[1350:1575] += add_time_noise(B1_s4, x_axis, y_axis, z_axis, n)
    b2_field[1350:1575] += add_time_noise(B2_s4, x_axis, y_axis, z_axis, n)
    b3_field[1350:1575] += add_time_noise(B3_s4, x_axis, y_axis, z_axis, n)
    b4_field[1350:1575] += add_time_noise(B4_s4, x_axis, y_axis, z_axis, n)

    # add thruster 1+2+3+4 to state file
    thruster_state[115:280] += np.array([1, 0, 0, 0])
    thruster_state[475:650] += np.array([0, 1, 0, 0])
    thruster_state[880:1100] += np.array([0, 0, 1, 0])
    thruster_state[1350:1575] += np.array([0, 0, 0, 1])

    bg_field_nT = np.array(bg_field) * 1e9  # units nT
    bg_mag = np.sqrt(np.array(bg_field_nT)[:, 0] ** 2 + np.array(bg_field_nT)[:, 1] ** 2 + np.array(bg_field_nT)[:, 2] ** 2)  # units nT

    bfield_df = pd.DataFrame(np.array(b1_field))
    bfield_df.columns = ['B1X', 'B1Y', 'B1Z']

    bfield_df['B2X'] = np.array(b2_field)[:, 0]
    bfield_df['B2Y'] = np.array(b2_field)[:, 1]
    bfield_df['B2Z'] = np.array(b2_field)[:, 2]
    bfield_df['B3X'] = np.array(b3_field)[:, 0]
    bfield_df['B3Y'] = np.array(b3_field)[:, 1]
    bfield_df['B3Z'] = np.array(b3_field)[:, 2]
    bfield_df['B4X'] = np.array(b4_field)[:, 0]
    bfield_df['B4Y'] = np.array(b4_field)[:, 1]
    bfield_df['B4Z'] = np.array(b4_field)[:, 2]

    bfield_df['BGX'] = np.array(bg_field_nT)[:, 0]  # units nT
    bfield_df['BGY'] = np.array(bg_field_nT)[:, 1]  # units nT
    bfield_df['BGZ'] = np.array(bg_field_nT)[:, 2]  # units nT

    bfield_df['BG_mag'] = bg_mag
    bfield_df['time_raw'] = time_arr
    bfield_df['time_fmt'] = times

    thruster_df = pd.DataFrame(thruster_state)
    thruster_df.columns = ['Thruster 1', 'Thruster 2', 'Thruster 3', 'Thruster 4']
    thruster_df['Time'] = time_arr

    return bfield_df, thruster_df


def generate_sine_test_field(wave_amplitude: np.ndarray, wave_frequency: np.ndarray, phase_shift:np.ndarray=np.array([0,0,0])):
    """
    This field simulates thrusters firing on top of a slowly varying sine wave.
    Thruster conditions: overlapping 1+2+3+4 with no temporal offsets

    :param wave_amplitude: amplitude of wave in nT
    :param wave_frequency: frequency of wave in Hz
    :param phase_shift: phase shift of wave in degrees, default=[0,0,0]
    """
    # generate time array
    time_step = 10  # Hz (max sampling rate of VML)
    time_len = 500  # total seconds
    time_arr = np.linspace(0.0, time_len, time_step*time_len, endpoint=False)  # units seconds
    start_time = datetime.datetime(2025, 1, 1, 0, 0, 0)
    end_time = start_time + datetime.timedelta(seconds=500)
    times = pd.date_range(start_time, end_time, periods=time_step*time_len).to_list()

    # Amplitudes for each axis (in Tesla)
    amplitude_x = wave_amplitude[0]*1e-9
    amplitude_y = wave_amplitude[1]*1e-9
    amplitude_z = wave_amplitude[2]*1e-9

    # Frequencies for each axis (in Hz)
    frequency_x = wave_frequency[0]
    frequency_y = wave_frequency[1]
    frequency_z = wave_frequency[2]

    if time_step < 2*frequency_x or time_step < 2*frequency_y or time_step < 2*frequency_z:
        raise ValueError("Sampling rate is too small for wave frequency per Nyquist shannon theorem")

    # convert phase shift to radians
    phase_shift = np.deg2rad(phase_shift)
    phase_x = phase_shift[0]
    phase_y = phase_shift[1]
    phase_z = phase_shift[2]

    bg_field = [np.array([
            amplitude_x * np.sin(2 * np.pi * frequency_x * t + phase_x),
            amplitude_y * np.sin(2 * np.pi * frequency_y * t + phase_y),
            amplitude_z * np.sin(2 * np.pi * frequency_z * t + phase_z),
        ]) for t in time_arr ]

    # create background field (units Tesla) based on time sequence
    b1_field = np.stack(bg_field)
    b2_field = np.stack(bg_field)
    b3_field = np.stack(bg_field)
    b4_field = np.stack(bg_field)
    bg_field = np.stack(bg_field)

    # create thruster state based on time sequence
    thruster_state = [np.array([0, 0, 0, 0])] * len(time_arr)

    B1_s1, B2_s1, B3_s1, B4_s1, B1_s2, B2_s2, B3_s2, B4_s2, B1_s3, B2_s3, B3_s3, B4_s3, B1_s4, B2_s4, B3_s4, B4_s4 = Thruster.base_profile()

    # add thruster 1 noise to background field
    b1_field[115:280] += B1_s1
    b2_field[115:280] += B2_s1
    b3_field[115:280] += B3_s1
    b4_field[115:280] += B4_s1

    # add thruster 2 noise to background field
    b1_field[475:650] += B1_s2
    b2_field[475:650] += B2_s2
    b3_field[475:650] += B3_s2
    b4_field[475:650] += B4_s2

    # add thruster 3 noise to background field
    b1_field[880:1100] += B1_s3
    b2_field[880:1100] += B2_s3
    b3_field[880:1100] += B3_s3
    b4_field[880:1100] += B4_s3

    # add thruster 4 noise to background field
    b1_field[1350:1575] += B1_s4
    b2_field[1350:1575] += B2_s4
    b3_field[1350:1575] += B3_s4
    b4_field[1350:1575] += B4_s4

    # add thruster 1 noise to background field
    b1_field[1865:1990] += B1_s1
    b2_field[1865:1990] += B2_s1
    b3_field[1865:1990] += B3_s1
    b4_field[1865:1990] += B4_s1

    # add thruster 1 + 3 noise to background field
    b1_field[1990:2140] += (B1_s1 + B1_s3)
    b2_field[1990:2140] += (B2_s1 + B2_s3)
    b3_field[1990:2140] += (B3_s1 + B3_s3)
    b4_field[1990:2140] += (B4_s1 + B4_s3)

    # add thruster 1 noise to background field
    b1_field[2355:2595] += B1_s1
    b2_field[2355:2595] += B2_s1
    b3_field[2355:2595] += B3_s1
    b4_field[2355:2595] += B4_s1

    # add thruster 2 + 3 + 4 noise to background field
    b1_field[2355:2630] += (B1_s2 + B1_s3 + B1_s4)
    b2_field[2355:2630] += (B2_s2 + B2_s3 + B2_s4)
    b3_field[2355:2630] += (B3_s2 + B3_s3 + B3_s4)
    b4_field[2355:2630] += (B4_s2 + B4_s3 + B4_s4)

    # add thruster 1 noise to background field
    b1_field[2860:3230] += B1_s1
    b2_field[2860:3230] += B2_s1
    b3_field[2860:3230] += B3_s1
    b4_field[2860:3230] += B4_s1

    # add thruster 2 noise to background field
    b1_field[2920:3350] += B1_s2
    b2_field[2920:3350] += B2_s2
    b3_field[2920:3350] += B3_s2
    b4_field[2920:3350] += B4_s2

    # add thruster 4 noise to background field
    b1_field[2995:3410] += B1_s4
    b2_field[2995:3410] += B2_s4
    b3_field[2995:3410] += B3_s4
    b4_field[2995:3410] += B4_s4

    # add thruster 3 noise to background field
    b1_field[3105:3505] += B1_s3
    b2_field[3105:3505] += B2_s3
    b3_field[3105:3505] += B3_s3
    b4_field[3105:3505] += B4_s3

    # add thruster 1+2+3+4 noise to background field
    b1_field[3915:4245] += (B1_s1 + B1_s2 + B1_s3 + B1_s4)
    b2_field[3915:4245] += (B2_s1 + B2_s2 + B2_s3 + B2_s4)
    b3_field[3915:4245] += (B3_s1 + B3_s2 + B3_s3 + B3_s4)
    b4_field[3915:4245] += (B4_s1 + B4_s2 + B4_s3 + B4_s4)

    # add thruster 1+2+3+4 noise to background field
    b1_field[4615:4705] += (B1_s1 + B1_s2 + B1_s3 + B1_s4)
    b2_field[4615:4705] += (B2_s1 + B2_s2 + B2_s3 + B2_s4)
    b3_field[4615:4705] += (B3_s1 + B3_s2 + B3_s3 + B3_s4)
    b4_field[4615:4705] += (B4_s1 + B4_s2 + B4_s3 + B4_s4)

    # add thruster 1+2+3+4 to state file
    thruster_state[115:280] += np.array([1, 0, 0, 0])
    thruster_state[475:650] += np.array([0, 1, 0, 0])
    thruster_state[880:1100] += np.array([0, 0, 1, 0])
    thruster_state[1350:1575] += np.array([0, 0, 0, 1])
    thruster_state[1865:1990] += np.array([1, 0, 0, 0])
    thruster_state[1990:2140] += np.array([1, 0, 1, 0])
    thruster_state[2355:2595] += np.array([1, 0, 0, 0])
    thruster_state[2355:2630] += np.array([0, 1, 1, 1])
    thruster_state[2860:3230] += np.array([1, 0, 0, 0])
    thruster_state[2920:3350] += np.array([0, 1, 0, 0])
    thruster_state[2995:3410] += np.array([0, 0, 0, 1])
    thruster_state[3105:3505] += np.array([0, 0, 1, 0])
    thruster_state[3915:4245] += np.array([1, 1, 1, 1])
    thruster_state[4615:4705] += np.array([1, 1, 1, 1])

    bg_field_nT = np.array(bg_field) * 1e9  # units nT
    bg_mag = np.sqrt(np.array(bg_field_nT)[:, 0] ** 2 + np.array(bg_field_nT)[:, 1] ** 2 + np.array(bg_field_nT)[:, 2] ** 2)  # units nT

    bfield_df = pd.DataFrame(np.array(b1_field))
    bfield_df.columns = ['B1X', 'B1Y', 'B1Z']

    bfield_df['B2X'] = np.array(b2_field)[:, 0]
    bfield_df['B2Y'] = np.array(b2_field)[:, 1]
    bfield_df['B2Z'] = np.array(b2_field)[:, 2]
    bfield_df['B3X'] = np.array(b3_field)[:, 0]
    bfield_df['B3Y'] = np.array(b3_field)[:, 1]
    bfield_df['B3Z'] = np.array(b3_field)[:, 2]
    bfield_df['B4X'] = np.array(b4_field)[:, 0]
    bfield_df['B4Y'] = np.array(b4_field)[:, 1]
    bfield_df['B4Z'] = np.array(b4_field)[:, 2]

    bfield_df['BGX'] = np.array(bg_field_nT)[:, 0]  # units nT
    bfield_df['BGY'] = np.array(bg_field_nT)[:, 1]  # units nT
    bfield_df['BGZ'] = np.array(bg_field_nT)[:, 2]  # units nT

    bfield_df['BG_mag'] = bg_mag
    bfield_df['time_raw'] = time_arr
    bfield_df['time_fmt'] = times

    thruster_df = pd.DataFrame(thruster_state)
    thruster_df.columns = ['Thruster 1', 'Thruster 2', 'Thruster 3', 'Thruster 4']
    thruster_df['Time'] = time_arr

    return bfield_df, thruster_df


def add_time_noise(B, x_pct, y_pct, z_pct, n):
    """
    Generate smoothed time-varying noise proportional to B
    using exponential smoothing + moving-average smoothing.
    """
    # replicate base vector over time
    B_stack = np.tile(B, (n, 1))

    # generate Gaussian noise
    noise = np.zeros_like(B_stack)
    noise[:, 0] = B[0] * np.random.normal(0, x_pct / 100.0, n)
    noise[:, 1] = B[1] * np.random.normal(0, y_pct / 100.0, n)
    noise[:, 2] = B[2] * np.random.normal(0, z_pct / 100.0, n)

    # EXPONENTIAL SMOOTHING
    def smooth_exp(x, alpha=0.10):
        y = np.zeros_like(x)
        y[0] = x[0]
        for i in range(1, len(x)):
            y[i] = alpha * x[i] + (1 - alpha) * y[i - 1]
        return y

    for i in range(3):
        # alpha=0.05   # smoother drift
        # alpha=0.15   # more jitter
        noise[:, i] = smooth_exp(noise[:, i], alpha=0.1)

    # MOVING AVERAGE
    # kernel_size = 3   # light blur
    # kernel_size = 10  # smooth like butter :)
    kernel_size = 5
    kernel = np.ones(kernel_size) / kernel_size

    for i in range(3):
        noise[:, i] = np.convolve(noise[:, i], kernel, mode='same')

    # return smoothed noise added to replicated base field
    return B_stack + noise


def add_sine_jitter(B, x_pct, y_pct, z_pct, n, time_arr):
    """
    Generate jitter noise proportional to B.
    n: number of timesteps
    time_arr: portion of the global time array
    """

    # replicate base vector across time
    B_stack = np.tile(B, (n, 1))

    jitter = np.zeros_like(B_stack)

    # small jitter sinusoid
    freq = 0.1  # Hz (adjustable)
    phase = np.random.uniform(0, 2 * np.pi)  # random phase

    jitter[:, 0] = (x_pct / 100) * np.sin(2 * np.pi * freq * time_arr + phase)
    jitter[:, 1] = (y_pct / 100) * np.sin(2 * np.pi * freq * time_arr + phase)
    jitter[:, 2] = (z_pct / 100) * np.sin(2 * np.pi * freq * time_arr + phase)

    # Combine drift + jitter scaled by base B
    return B_stack + jitter * B


def reaction_wheel_micro_oscillation(time_arr, base_B, pct_scale):
    """
    Deterministic oscillation components caused by reaction wheel torque ripple.

    :param time_arr: time segment (e.g., 115:280)
    :param base_B: nominal magnetic field vector (3,)
    :param pct_scale: % of base field magnitude
    :return: time-varying oscillatory noise array (n, 3)
    """
    n = len(time_arr)
    osc = np.zeros((n, 3))

    # Reaction wheel rotational frequency (Hz)
    f = 12.0  # ~720 RPM
    # Harmonics (1×, 2×, 3×)
    harmonics = np.array([f, 2*f, 3*f])

    # Axis coupling strength (tunable)
    axis_amp = np.array([0.004, 0.0025, 0.003])  # X, Y, Z multipliers

    # Base amplitude scaled by percent
    amp = (pct_scale / 100.0)

    for i, t in enumerate(time_arr):
        # deterministic sum-of-sines
        osc_x = (np.sin(2*np.pi*harmonics[0]*t + 0.1)
                +0.3*np.sin(2*np.pi*harmonics[1]*t + 0.7)
                +0.2*np.sin(2*np.pi*harmonics[2]*t + 1.3))

        osc_y = (0.6*np.sin(2*np.pi*harmonics[0]*t + 0.2)
                +0.4*np.sin(2*np.pi*harmonics[1]*t + 1.4))

        osc_z = (0.5*np.sin(2*np.pi*harmonics[0]*t + 0.9)
                +0.2*np.sin(2*np.pi*harmonics[2]*t + 2.2))

        osc[i, 0] = osc_x * axis_amp[0]
        osc[i, 1] = osc_y * axis_amp[1]
        osc[i, 2] = osc_z * axis_amp[2]

    # scale relative to field magnitude
    return osc * base_B * amp
