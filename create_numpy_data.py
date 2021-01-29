#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  8 15:42:06 2020
​
@author: degerli
"""
import os
import argparse
import numpy as np
import imageio

def convertNumpy(path):
    data_list = sorted(os.listdir(path))
    data = []           
    for i in data_list:
        #save data
        img = imageio.imread(path + i)
        img_resized_float = img.astype('float32')
        data.append(np.expand_dims(np.array(img_resized_float), axis = 2))
    
    data = np.array(data)
    return data

ap = argparse.ArgumentParser()
ap.add_argument('--dir_early', required=True, help="path to early COVID-19 images")
ap.add_argument('--dir_mild', required=True, help="path to mild COVID-19 images")
ap.add_argument('--dir_control', required=True, help="path to control group images")
args = vars(ap.parse_args())

np.save(('data/covid_early.npy'), convertNumpy(args['dir_early']))
np.save(('data/covid_mild.npy'), convertNumpy(args['dir_mild']))
np.save(('data/normal.npy'), convertNumpy(args['dir_control']))