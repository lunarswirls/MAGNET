#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Imports
from magnetpy.noise_parameter_space.sine_wave_test import sine_wave_thruster_test

wave_amps = [0.5, 0.75, 1.0, 1.25, 1.5]  # nT
wave_freqs = [0.01, 0.05, 0.1, 0.5, 1]  # Hz
phases = [0, 30, 45, 60, 90]  # degrees

test = 1
for wave_amp in wave_amps:
    for wave_freq in wave_freqs:
        for phase in phases:
            print(f'\nTest:\t{test}')
            print(f'Amplitude:\t{wave_amp}')
            print(f'Frequency:\t{wave_freq}')
            print(f'Phase:\t{phase}\n')
            sine_wave_thruster_test(test, wave_amp, wave_freq, phase)
            test += 1
