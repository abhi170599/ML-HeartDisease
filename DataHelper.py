'''
Created on Sep 23, 2017

@author: Azda Firmansyah
'''
import numpy as np
import csv
#1st way use numpy

def datasetHeartDisease():
    file_path = '../Data/HeartDisease.csv'
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
#datasetHeartDisease()    
    #file_path_labels = '../Data/HeartDiseaseLabels.csv'

"""    
def getFeaturesHeartDisease(): 
    file_path = '../Data/HeartDiseaseFeatures.csv'   
    raw_data = open(file_path,encoding='utf-8')
    data = np.loadtxt(raw_data, delimiter=',')
    return data

def getLabelsHeartDisease():
    file_path = '../Data/HeartDiseaseLabels.csv'
    raw_data = open(file_path,encoding='utf-8')
    data = np.loadtxt(raw_data)
    return data

#2nd way open file
def useStandard():
    with open(file_path, encoding='utf-8') as csv_file:
        data_file = csv.reader(csv_file)
        list_data = list(data_file)
        print("Number of Rows :",len(list_data))
        print("Number of Cols :",len(list_data[0]))
        #for i,items in enumerate(data_file):            
        #    print(items)
useStandard()        
npData = useNp()
print(npData.shape)
"""