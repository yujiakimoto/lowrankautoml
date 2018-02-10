#usage: in terminal, type `python generate_error_matrix_on_single_dataset.py $(PATH_TO_PREPROCESSED_DATASET)`

import numpy as np
import pandas as pd
import time
from os.path import basename

import sys
sys.path.append('../package')

import ML_algorithms as ml
import util
from model import Model


headings = util.generate_headings('all', 'default')

fname = sys.argv[1]

print(basename(fname))

dataset = pd.read_csv(fname, header=None).values
features = dataset[:,0:-1]
labels = dataset[:,-1].astype('int')

n = len(headings[0])
index = [util.generate_settings_single_model(headings, j) for j in range(n)]

row = []
for i in range(n):
    clf = Model(settings=util.generate_settings(headings, i), verbose=False)
    start = time.time()
    clf = clf.fit(features, labels)
    time_elapsed = time.time() - start
    row.append([clf.error, time_elapsed])
    
errmtx=pd.DataFrame(row, index=index, columns=['error', 'time'])
errmtx.to_csv('error_matrix/' +basename(fname))
