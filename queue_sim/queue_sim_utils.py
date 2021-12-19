# -*- coding: utf-8 -*-
"""
Created on Tue Dec 14 20:51:04 2021

@author: duzhe
"""

import numpy as np
import os
import pandas as pd


input_file = "trace_0.csv"

file_folder = os.getcwd().strip("queue_sim") + "VR_traces" + "\\"  
file_to_plot = file_folder + input_file 


data = pd.read_csv(file_to_plot, encoding='utf-16-LE')

fps = 30
total_frames = data['frame'].iloc[-1] + 1
total_time = total_frames / fps # in seconds
print(total_time)
sizes = data['size'].to_numpy()

bps = sum(sizes) * 8 / total_time # bits / seconds

print(bps/1e6)