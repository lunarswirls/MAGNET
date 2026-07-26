#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Run a host-platform test with nearby non-dipolar interference
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import magnetpy.utilities.Field_Utils as fieldutils
import magnetpy.utilities.Math_Utils as mathutils
import magnetpy.gradiometry_algorithm.tetrahedral_correction as tc
from magnetpy.gradiometry_algorithm.run_correction_pipeline import correction_pipeline
from magnetpy.paper_figs._paths import example_data_path


input_dir = example_data_path('near_source_nondipolar_test')
input_dir.mkdir(parents=True, exist_ok=True)
test = 'near_source_nondipolar'

raw_bfield_df, source_df = fieldutils.generate_near_source_nondipolar_field()
time_arr = raw_bfield_df['time_raw'].to_numpy()
bg_field_nt = raw_bfield_df[['BGX', 'BGY', 'BGZ']].to_numpy()
sensor_fields = [
    raw_bfield_df[[f'B{sensor_id}X', f'B{sensor_id}Y', f'B{sensor_id}Z']].to_numpy()
    for sensor_id in range(1, 5)
]

# verify that the sources are close enough to produce a non-dipolar field
source_distances = [np.linalg.norm(source['position']) for source in source_df.attrs['source_parameters']]
sensor_locations = mathutils.fetch_magloc()
array_baseline = max(
    np.linalg.norm(first - second)
    for first_index, first in enumerate(sensor_locations)
    for second in sensor_locations[first_index + 1:]
)
assert min(source_distances) < 1.5 * array_baseline

# quantify how poorly one dipole represents the injected source region
event_index = source_df['Source 1'].to_numpy().astype(bool)
observed_interference = [
    sensor_field[event_index][0] - bg_field_nt[event_index][0] * 1e-9
    for sensor_field in sensor_fields
]
fit = tc.array_correction(*observed_interference)
fitted_interference = tc.mag_array(fit[6], fit[4], fit[7], fit[5])
fit_error = np.sqrt(
    sum(np.linalg.norm(observed - fitted) ** 2 for observed, fitted in zip(observed_interference, fitted_interference))
    / sum(np.linalg.norm(observed) ** 2 for observed in observed_interference)
)
assert fit_error > 0.5

uncorrected_fields_nt = [sensor_field * 1e9 for sensor_field in sensor_fields]
corrected_fields_nt = uncorrected_fields_nt
initial_rms = np.sqrt(
    np.mean(np.concatenate([sensor_field - bg_field_nt for sensor_field in uncorrected_fields_nt]) ** 2)
)
max_event_residual = [
    max(
        np.linalg.norm(sensor_field[event_index] - bg_field_nt[event_index], axis=1).max()
        for sensor_field in uncorrected_fields_nt
    )
]

# repeated fits allow the dipole model to remove successive components
iterations = 0
two_iteration_rms = None
for iteration in range(1, 11):
    pipeline_input = [sensor_field * 1e-9 for sensor_field in corrected_fields_nt]
    bfield_df = correction_pipeline(time_arr, *pipeline_input)
    assert bfield_df is not None

    corrected_fields_nt = [
        bfield_df[[f'B{sensor_id}X_CORR', f'B{sensor_id}Y_CORR', f'B{sensor_id}Z_CORR']].to_numpy()
        for sensor_id in range(1, 5)
    ]
    iterations = iteration
    max_event_residual.append(
        max(
            np.linalg.norm(sensor_field[event_index] - bg_field_nt[event_index], axis=1).max()
            for sensor_field in corrected_fields_nt
        )
    )
    if iteration == 2:
        two_iteration_rms = np.sqrt(
            np.mean(np.concatenate([sensor_field - bg_field_nt for sensor_field in corrected_fields_nt]) ** 2)
        )
    if max_event_residual[-1] < 1.5:
        break

assert two_iteration_rms is not None
assert two_iteration_rms < 0.15 * initial_rms

final_rms = np.sqrt(
    np.mean(np.concatenate([sensor_field - bg_field_nt for sensor_field in corrected_fields_nt]) ** 2)
)
reduction = 1.0 - final_rms / initial_rms

performance_df = pd.DataFrame([{
    'Sources': 3,
    'Closest source distance (m)': min(source_distances),
    'Array maximum baseline (m)': array_baseline,
    'Single-dipole relative RMS misfit': fit_error,
    'Iterations': iterations,
    'Initial RMS residual (nT)': initial_rms,
    'Final RMS residual (nT)': final_rms,
    'Residual reduction': reduction,
    'Status': 'PASS' if max_event_residual[-1] < 1.5 else 'FAIL',
}])
performance_df.to_csv(input_dir / f'{test}_performance.csv', index=False)

source_parameters = []
for source_id, source in enumerate(source_df.attrs['source_parameters'], start=1):
    direction = source['direction'] / np.linalg.norm(source['direction'])
    source_parameters.append({
        'Source': source_id,
        'Moment (A m^2)': source['moment'],
        'X position (m)': source['position'][0],
        'Y position (m)': source['position'][1],
        'Z position (m)': source['position'][2],
        'X moment direction': direction[0],
        'Y moment direction': direction[1],
        'Z moment direction': direction[2],
    })
pd.DataFrame(source_parameters).to_csv(input_dir / f'{test}_sources.csv', index=False)

output_df = raw_bfield_df.copy()
for column in ['Source 1', 'Source 2', 'Source 3']:
    output_df[column] = source_df[column]
for sensor_id, sensor_field in enumerate(corrected_fields_nt, start=1):
    for axis_index, axis in enumerate('XYZ'):
        output_df[f'B{sensor_id}{axis}_CORR'] = sensor_field[:, axis_index]
output_df.to_csv(input_dir / f'{test}_output.csv', index=False)

uncorrected_residual = np.max(
    [np.linalg.norm(sensor_field - bg_field_nt, axis=1) for sensor_field in uncorrected_fields_nt],
    axis=0,
)
corrected_residual = np.max(
    [np.linalg.norm(sensor_field - bg_field_nt, axis=1) for sensor_field in corrected_fields_nt],
    axis=0,
)

fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
axes[0].plot(time_arr, uncorrected_residual, color='firebrick')
axes[1].plot(time_arr, corrected_residual, color='goldenrod')
axes[2].plot(
    range(len(max_event_residual)),
    max_event_residual,
    marker='o',
    color='purple',
    linestyle='dashed',
    linewidth=2,
)
axes[1].axhline(1.5, color='black', linestyle=':', linewidth=1.5, label='1.5 nT target')
axes[2].axhline(1.5, color='black', linestyle=':', linewidth=1.5, label='1.5 nT target')

axes[0].set_ylabel('nT', fontsize=16)
axes[0].set_xlabel('Time (seconds)', fontsize=16)
axes[1].set_xlabel('Time (seconds)', fontsize=16)
axes[2].set_xlabel('Iteration', fontsize=16)

axes[0].set_title('(a.) Uncorrected Residual', fontsize=20)
axes[1].set_title('(b.) Corrected Residual', fontsize=20)
axes[2].set_title('(c.) Maximum Residual', fontsize=20)

axes[1].legend(fontsize='medium', loc='upper right')
axes[2].legend(fontsize='medium', loc='upper right')

fig.suptitle(f'Near-Source Non-Dipolar Simulation (single-dipole misfit = {fit_error:.2f})', fontsize=20)
fig.tight_layout(rect=[0, 0, 1, 0.92])
fig.savefig(input_dir / f'{test}_residual_correction_error.png', dpi=300)
plt.close(fig)
