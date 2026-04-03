#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
import glob

# TODO: update this
indir = "/Users/danywaller/Projects/moon/gradiometry_algorithm/sine_XYZ/parameters"

outdir = "/Users/danywaller/Projects/moon/gradiometry_algorithm/sine_results"
os.makedirs(outdir, exist_ok=True)

all_files = glob.glob(os.path.join(indir, "*.csv"))

dfs = list()
for f in all_files:
    data = pd.read_csv(f)
    # .stem is method for pathlib objects to get the filename w/o the extension
    if 'FAIL' in f:
        data['status'] = 'FAIL'
        data['file'] = int(f.split("/")[-1].split(".")[0][len('sine_XXY_test'):-len('_FAIL_performance')])
    else:
        data['status'] = 'PASS'
        data['file'] = int(f.split("/")[-1].split(".")[0][len('sine_XXY_test'):-len('_performance')])

    results_file = os.path.join(os.path.dirname(indir), f.split("/")[-1].replace("performance", "output"))
    if os.path.exists(results_file):
        data_results = pd.read_csv(results_file)
        data['Max RMSE'] = data_results[['RMSE_B1', 'RMSE_B2', 'RMSE_B3', 'RMSE_B4']].max().max()
    else:
        data['Max RMSE'] = np.nan
    dfs.append(data)

df = pd.concat(dfs, ignore_index=True)
df.sort_values("file", inplace=True)

total_filepath = os.path.join(outdir, "sine_XYZ_test_parameters.xlsx")
df.to_excel(total_filepath, index=False)