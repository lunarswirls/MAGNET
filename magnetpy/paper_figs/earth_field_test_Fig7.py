#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Run test with Earth-like background field at 500 km altitude
"""
# Imports
import datetime
import numpy as np
import pandas as pd
import magnetpy.utilities.Field_Utils as FieldUtils
from magnetpy.gradiometry_algorithm.run_correction_pipeline import correction_pipeline
import magnetpy.utilities.Math_Utils as MathUtils
import magnetpy.utilities.Plot_Utils as PlotUtils
from magnetpy.paper_figs._paths import example_data_path

verbose = False
rmse_testing = False
corr_testing = False
input_dir = example_data_path("earth_field_test")
input_dir.mkdir(parents=True, exist_ok=True)

alt=500

test = "earth_test_" + str(alt) + "km"

raw_bfield_df, thruster_df = FieldUtils.generate_earth_field(alt)

time_arr = raw_bfield_df['time_raw']
times = raw_bfield_df['time_fmt']
bg_field_nT = raw_bfield_df[['BGX', 'BGY', 'BGZ']].to_numpy()  # units nT
bg_mag = raw_bfield_df['BG_mag']
b1_field = raw_bfield_df[['B1X', 'B1Y', 'B1Z']].to_numpy()  # units T
b2_field = raw_bfield_df[['B2X', 'B2Y', 'B2Z']].to_numpy()  # units T
b3_field = raw_bfield_df[['B3X', 'B3Y', 'B3Z']].to_numpy()  # units T
b4_field = raw_bfield_df[['B4X', 'B4Y', 'B4Z']].to_numpy()  # units T

b1_field_nT = np.array(b1_field) * 1e9  # units nT
b2_field_nT = np.array(b2_field) * 1e9  # units nT
b3_field_nT = np.array(b3_field) * 1e9  # units nT
b4_field_nT = np.array(b4_field) * 1e9  # units nT

bfield_df = correction_pipeline(time_arr, b1_field, b2_field, b3_field, b4_field)

if corr_testing:
    PlotUtils.correction_quicklook(raw_bfield_df, bfield_df, input_dir, test, 0)

corrected_b1 = bfield_df[['B1X_CORR', 'B1Y_CORR', 'B1Z_CORR']].to_numpy()  # units nT
corrected_b2 = bfield_df[['B2X_CORR', 'B2Y_CORR', 'B2Z_CORR']].to_numpy()  # units nT
corrected_b3 = bfield_df[['B3X_CORR', 'B3Y_CORR', 'B3Z_CORR']].to_numpy()  # units nT
corrected_b4 = bfield_df[['B4X_CORR', 'B4Y_CORR', 'B4Z_CORR']].to_numpy()  # units nT

corr_b1_mag = np.linalg.norm(bg_field_nT - corrected_b1, axis=1)  # units nT
corr_b2_mag = np.linalg.norm(bg_field_nT - corrected_b2, axis=1)  # units nT
corr_b3_mag = np.linalg.norm(bg_field_nT - corrected_b3, axis=1)  # units nT
corr_b4_mag = np.linalg.norm(bg_field_nT - corrected_b4, axis=1)  # units nT

# calculate root mean square error
bfield_df['RMSE_B1'] = MathUtils.vec_rmse(bg_field_nT, corrected_b1)  # units nT
bfield_df['RMSE_B2'] = MathUtils.vec_rmse(bg_field_nT, corrected_b2)  # units nT
bfield_df['RMSE_B3'] = MathUtils.vec_rmse(bg_field_nT, corrected_b3)  # units nT
bfield_df['RMSE_B4'] = MathUtils.vec_rmse(bg_field_nT, corrected_b4)  # units nT

# store max RMSE per iteration for plotting later
max_rmse = []
init_max_rmse = bfield_df[['RMSE_B1', 'RMSE_B2', 'RMSE_B3', 'RMSE_B4']].max().max()
max_rmse.append(init_max_rmse)

# assume solution exists and start search
solution = True
n = 1
start_time = datetime.datetime.now()

while bfield_df[['RMSE_B1', 'RMSE_B2', 'RMSE_B3', 'RMSE_B4']].max().max() > 5.0 and n <= 249:
    # print('Iteration: ' + str(n))

    # store previous correction magnitudes to check for change between iterations
    prev1 = corr_b1_mag  # units nT
    prev2 = corr_b2_mag  # units nT
    prev3 = corr_b3_mag  # units nT
    prev4 = corr_b4_mag  # units nT

    b1_field1 = corrected_b1 * 1e-9  # units Tesla
    b2_field1 = corrected_b2 * 1e-9  # units Tesla
    b3_field1 = corrected_b3 * 1e-9  # units Tesla
    b4_field1 = corrected_b4 * 1e-9  # units Tesla

    # calculate new corrected field
    bfield_df = correction_pipeline(time_arr, b1_field1, b2_field1, b3_field1, b4_field1)

    if bfield_df is not None:
        bfield_df['Time'] = times
        corrected_b1 = bfield_df[['B1X_CORR', 'B1Y_CORR', 'B1Z_CORR']].to_numpy()  # units nT
        corrected_b2 = bfield_df[['B2X_CORR', 'B2Y_CORR', 'B2Z_CORR']].to_numpy()  # units nT
        corrected_b3 = bfield_df[['B3X_CORR', 'B3Y_CORR', 'B3Z_CORR']].to_numpy()  # units nT
        corrected_b4 = bfield_df[['B4X_CORR', 'B4Y_CORR', 'B4Z_CORR']].to_numpy()  # units nT

        # calculate relative error = abs(true_value - approx_value) / abs(true_value)
        corr_b1_mag = np.linalg.norm(bg_field_nT - corrected_b1, axis=1)  # units nT
        corr_b2_mag = np.linalg.norm(bg_field_nT - corrected_b2, axis=1)  # units nT
        corr_b3_mag = np.linalg.norm(bg_field_nT - corrected_b3, axis=1)  # units nT
        corr_b4_mag = np.linalg.norm(bg_field_nT - corrected_b4, axis=1)  # units nT

        bfield_df['RMSE_B1'] = MathUtils.vec_rmse(bg_field_nT, corrected_b1)  # units nT
        bfield_df['RMSE_B2'] = MathUtils.vec_rmse(bg_field_nT, corrected_b2)  # units nT
        bfield_df['RMSE_B3'] = MathUtils.vec_rmse(bg_field_nT, corrected_b3)  # units nT
        bfield_df['RMSE_B4'] = MathUtils.vec_rmse(bg_field_nT, corrected_b4)  # units nT

        new_max_rmse = bfield_df[['RMSE_B1', 'RMSE_B2', 'RMSE_B3', 'RMSE_B4']].max().max()
        max_rmse.append(new_max_rmse)

        if rmse_testing:
            PlotUtils.rmse_quicklook(bfield_df, input_dir, test, n)

        if corr_testing:
            PlotUtils.correction_quicklook(raw_bfield_df, bfield_df, input_dir, test, n)

        if np.max(prev1-corr_b1_mag)<0.05 and np.max(prev2-corr_b2_mag)<0.05 and np.max(prev3-corr_b3_mag)<0.05 and np.max(prev4-corr_b4_mag)<0.05:
            print("No change between iterations! Breaking denoising loop")
            n += 1
            break
        else:
            n += 1
    else:
        print("No solution found! Breaking denoising loop")
        test = test + '_FAIL'
        solution = False
        break
end_time = datetime.datetime.now()

# output algo performance metrics and test parameters for plotting later
perform_dir = input_dir / "parameters"
perform_dir.mkdir(parents=True, exist_ok=True)
perform_dict = {
    'Iterations': n,
    'Time': (end_time - start_time).total_seconds(),
    'Max RMSE': bfield_df[['RMSE_B1', 'RMSE_B2', 'RMSE_B3', 'RMSE_B4']].max().max() if solution else np.nan,
    'Status': 'PASS' if solution else 'FAIL'
}
perform_df = pd.DataFrame([perform_dict])
performance_output = perform_dir / f"{test}_performance.csv"
perform_df.to_csv(performance_output, index=False)

if solution:
    # put all results in dataframe and dump to csv
    bfield_df['B1_CORR_MAG'] = np.sqrt(bfield_df['B1X_CORR'] ** 2 + bfield_df['B1Y_CORR'] ** 2 + bfield_df['B1Z_CORR'] ** 2)  # units nT
    bfield_df['B2_CORR_MAG'] = np.sqrt(bfield_df['B2X_CORR'] ** 2 + bfield_df['B2Y_CORR'] ** 2 + bfield_df['B2Z_CORR'] ** 2)  # units nT
    bfield_df['B3_CORR_MAG'] = np.sqrt(bfield_df['B3X_CORR'] ** 2 + bfield_df['B3Y_CORR'] ** 2 + bfield_df['B3Z_CORR'] ** 2)  # units nT
    bfield_df['B4_CORR_MAG'] = np.sqrt(bfield_df['B4X_CORR'] ** 2 + bfield_df['B4Y_CORR'] ** 2 + bfield_df['B4Z_CORR'] ** 2)  # units nT

    bfield_df['B1X_UNCORR'] = np.array(b1_field_nT)[:, 0]  # units nT
    bfield_df['B1Y_UNCORR'] = np.array(b1_field_nT)[:, 1]  # units nT
    bfield_df['B1Z_UNCORR'] = np.array(b1_field_nT)[:, 2]  # units nT

    bfield_df['B2X_UNCORR'] = np.array(b2_field_nT)[:, 0]  # units nT
    bfield_df['B2Y_UNCORR'] = np.array(b2_field_nT)[:, 1]  # units nT
    bfield_df['B2Z_UNCORR'] = np.array(b2_field_nT)[:, 2]  # units nT

    bfield_df['B3X_UNCORR'] = np.array(b3_field_nT)[:, 0]  # units nT
    bfield_df['B3Y_UNCORR'] = np.array(b3_field_nT)[:, 1]  # units nT
    bfield_df['B3Z_UNCORR'] = np.array(b3_field_nT)[:, 2]  # units nT

    bfield_df['B4X_UNCORR'] = np.array(b4_field_nT)[:, 0]  # units nT
    bfield_df['B4Y_UNCORR'] = np.array(b4_field_nT)[:, 1]  # units nT
    bfield_df['B4Z_UNCORR'] = np.array(b4_field_nT)[:, 2]  # units nT

    bfield_df['B1_UNCORR_MAG'] = np.sqrt(bfield_df['B1X_UNCORR'] ** 2 + bfield_df['B1Y_UNCORR'] ** 2 + bfield_df['B1Z_UNCORR'] ** 2)  # units nT
    bfield_df['B2_UNCORR_MAG'] = np.sqrt(bfield_df['B2X_UNCORR'] ** 2 + bfield_df['B2Y_UNCORR'] ** 2 + bfield_df['B2Z_UNCORR'] ** 2)  # units nT
    bfield_df['B3_UNCORR_MAG'] = np.sqrt(bfield_df['B3X_UNCORR'] ** 2 + bfield_df['B3Y_UNCORR'] ** 2 + bfield_df['B3Z_UNCORR'] ** 2)  # units nT
    bfield_df['B4_UNCORR_MAG'] = np.sqrt(bfield_df['B4X_UNCORR'] ** 2 + bfield_df['B4Y_UNCORR'] ** 2 + bfield_df['B4Z_UNCORR'] ** 2)  # units nT

    bfield_df['BGX'] = np.array(bg_field_nT)[:, 0]  # units nT
    bfield_df['BGY'] = np.array(bg_field_nT)[:, 1]  # units nT
    bfield_df['BGZ'] = np.array(bg_field_nT)[:, 2]  # units nT

    bfield_df['BG_MAG'] = bg_mag  # units nT

    csv_output = input_dir / f"{test}_output.csv"
    bfield_df.to_csv(csv_output, index=False)

    # create all plots
    PlotUtils.thruster_plot(thruster_df, input_dir, test)
    PlotUtils.vector_final_plot(bfield_df, input_dir, test)
    PlotUtils.mag_final_plot(bfield_df, input_dir, test)
    PlotUtils.max_rmse_plot(n, max_rmse, input_dir, test)
