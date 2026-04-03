#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import pandas as pd
import glob

# TODO: update this
indir = "/Users/danywaller/Projects/moon/gradiometry_algorithm/noise_test/parameters"

all_files = glob.glob(os.path.join(indir, "*.csv"))

dfs = list()
for f in all_files:
    data = pd.read_csv(f)
    dfs.append(data)

df = pd.concat(dfs, ignore_index=True)
df.sort_values("File", inplace=True)

total_filepath = os.path.join(indir, "noisy_test_parameters.xlsx")
df.to_excel(total_filepath, index=False)