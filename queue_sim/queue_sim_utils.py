# -*- coding: utf-8 -*-
"""
Created on Tue Dec 14 20:51:04 2021

@author: duzhe
"""

import numpy as np
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import time


tic = time.perf_counter()

pd.options.mode.chained_assignment = None  # default='warn'

file_folder = os.getcwd().strip("queue_sim")

file_name = "trace_APP50_0.6.csv"
trace_folder = file_folder + "VR_traces\\"
input_trace = trace_folder + file_name
sim_trace = pd.read_csv(input_trace, encoding='utf-16-LE') 
sim_trace.rename(columns = {'Unnamed: 0': 'index'}, inplace = True)

output_file_folder = file_folder + "Output\\local\\" 
parameters = "SEED1 - 5Q - 20.0% Load - BG - 6.0us" # TODO 
csv_folder = output_file_folder + parameters + "\\" + file_name.strip(".csv")


all_files = glob.glob(os.path.join(csv_folder, "*.csv"))

timestamps = pd.concat((pd.read_csv(f, index_col=False, header=None)
                            for f in all_files)).to_numpy().flatten()
timestamps.sort()

new_timestamps = pd.DataFrame(data=timestamps, columns=["timestamps"])
# new_timestamps.sort_values(by='timestamps', ignore_index=True)
new_timestamps.index = [x for x in range(0, len(new_timestamps.values))]
new_timestamps.reset_index(drop=True)

nr_packets = len(new_timestamps)
sim_trace = sim_trace[(sim_trace['index'] < nr_packets)]
sim_trace['time'] = new_timestamps


output_save_path = f'{file_folder}VR_traces\\{file_name.strip(".csv")}\\{parameters}'      
os.makedirs(output_save_path, exist_ok=True)

file_to_save = f'{output_save_path}\\{file_name.strip(".csv")}_full.csv'
sim_trace.to_csv(file_to_save, encoding='utf-16-LE')

# Load into dataframe
# sim_time = 59.999 



"""
From Burstiness function


# Adjust timestamps to avoid sorting problems!  
# Packet timestamp count starts at zero 
# Set all packets belonging to same frame to frame generation time 
fps = 30 # int(np.ceil(sim_data["frame"].iloc[-1] / sim_data["time"].iloc[-1]))
frame_time = 1 / fps       

# # Cut dataframe until total simulation time by frame number
# sim_data = sim_data[((sim_data['frame']/fps) <= sim_time)] 
sim_data['time'] = sim_data['frame'] * frame_time 

# Create list for start and final packet index of each frame
packets_in_frame = list(range(sim_data["frame"].iloc[-1] + 1))
interpacket_time = 0
# Add inter-packet time of 1 microsecond per packet for each frame  
# Add specific tau based on interframe time and burstiness 
for frame in range(sim_data['frame'].iloc[-1] + 1):
    # Save indices of current frame in list
    packets_in_frame[frame] = [] # start with empty list
    packets_in_frame[frame].append(sim_data.index[sim_data['frame'] == 
                                               frame][0].tolist())
    packets_in_frame[frame].append(sim_data.index[sim_data['frame'] == 
                                               frame][-1].tolist())


nr_packets_in_frame = list(range(len(packets_in_frame)))
interpacket_time_frame = list(range(len(packets_in_frame)))

burstiness = 1.0

dispersion_per_frame = frame_time * (1 - burstiness)

for frame in range(len(packets_in_frame)):
    # Add one microsecond inter-packet time 
    nr_packets_in_frame[frame] = packets_in_frame[frame][1] - packets_in_frame[
                                 frame][0] + 1 
    interpacket_time_frame[frame] =  dispersion_per_frame / \
                                        nr_packets_in_frame[frame]

max_inter_packet_time = frame_time / max(nr_packets_in_frame)

interpacket_time = sum(interpacket_time_frame) / len(interpacket_time_frame)

if interpacket_time >= max_inter_packet_time:
    interpacket_time = max_inter_packet_time
    print("max")

for frame in range(len(packets_in_frame)):
    # Add one microsecond inter-packet time 
    packet_counter = 0            
    for packet in range(packets_in_frame[frame][0], 
                        packets_in_frame[frame][1] + 1):
        sim_data['time'][packet] += packet_counter * interpacket_time
        packet_counter += 1


file_to_save = f'{file_folder}VR_traces\\trace_APP50_{burstiness}.csv'
sim_data_save = sim_data
# save to csv with dispersion as suffix
sim_data_save.to_csv(file_to_save, encoding='utf-16-LE')


# print(bps/1e6)
toc = time.perf_counter()

print(f"{toc-tic:0.4f}")
"""

# %% For plots?
"""
plot_data = sim_data['time'].to_numpy()
tti_duration = 0.00025
fps = 30
key_int = 10
GoP_duration = key_int * (1/30)
total_duration = np.ceil(plot_data[-1])
total_ttis = 4000 # int((total_duration / tti_duration) / 240)
GoP_ttis = int(GoP_duration / tti_duration)

packets_per_tti1 = np.zeros((total_ttis))

# Create list with how many packets arrive per TTI
for tti in range(total_ttis): 
    # if tti == total_ttis - 1: print(tti)
    start_time = tti * tti_duration
    end_time = start_time + tti_duration
    
    packet_list1 = np.where((plot_data >= start_time) & (plot_data < end_time))        
    packets_per_tti1[tti] = np.size(packet_list1)
    
x_axis = np.arange(0,int(total_ttis))

plt.figure(figsize=(30,20))
plt.title("Arrival packets per TTI after all queues" + "\n" +
          "VR background traffic", 
          # f"{input_folder1.split('- VR')[0]} & \n" + 
          # f"{input_folder2.split('- VR')[0]} & \n" + 
          # f"{input_folder3.split('- VR')[0]}",
            # f"\n Serving Bitrate: {serving_bitrate/1000000}Mbps" +  
            # "\n Load: 20%",
            # f"\n Number of queues: {n_queues[0]}",  
             # f"\n System Load: {sys_load[0]}%",                     
             fontsize=20, fontweight='bold') 

# plt.subplot(121)
# plt.title('Mean - time spend in total by packets of each frame', fontsize=20)
plt.xlabel('TTIs', fontsize=20)
plt.xticks(fontsize=16)
plt.ylabel('Number of packets', fontsize=20)
plt.yticks(fontsize=16)
plt.ylim(0, np.nanmax(packets_per_tti1) + 3)  

# plt.scatter(x_axis, packets_per_tti1[0:int(total_ttis)], marker='s', 
#             s=100, label='20% Load') # 0:int(total_ttis/2)
plt.bar(x_axis, packets_per_tti1[0:int(total_ttis)], # '-s',  
          label='20% Load')
"""


