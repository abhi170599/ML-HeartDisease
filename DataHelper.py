'''
Created on Sep 23, 2017

@author: Azda Firmansyah
'''
import numpy as np
import csv
#1st way use numpy

def datasetHeartDisease():
    file_path = 'HeartDisease.csv'
    with open(file_path,encoding='utf-8') as csv_file:
        data_file = csv.reader(csv_file)
        first_line = next(data_file)
        n_samples = int(first_line[0])
        n_features = int(first_line[1])
        data = np.empty((n_samples, n_features))
        target = np.empty((n_samples,), dtype=np.int)

        for count, value in enumerate(data_file):
            data[count] = np.asarray(value[:-1], dtype=np.float64)
            target[count] = np.asarray(value[-1], dtype=np.int)        
    return data, target 
