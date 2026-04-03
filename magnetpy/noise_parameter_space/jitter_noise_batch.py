#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Imports
import numpy as np
from magnetpy.noise_parameter_space.jitter_noise_test import noise_test

x_axis = np.linspace(0, 25.0, 6)
y_axis = np.linspace(0, 25.0, 6)
z_axis = np.linspace(0, 25.0, 6)

test = 1
for xax in x_axis:
    for yax in y_axis:
        for zax in z_axis:
            print(f'\nTest:\t{test}')
            print(f'X-Axis Noise:\t{xax}%')
            print(f'Y-Axis Noise:\t{yax}%')
            print(f'Z-Axis Noise:\t{zax}%')
            noise_test(test, xax, yax, zax)
            print('Done!')
            test += 1
