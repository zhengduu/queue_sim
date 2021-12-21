# -*- coding: utf-8 -*-
"""
Created on Sun Dec 12 17:46:06 2021

@author: duzhe
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys


def plot_dispersion(input_folder1, input_folder2, input_folder3, input_file, 
                    plot, save):
    
    serving_bitrate = 1e9
    n_queues = [input_folder1.strip(" - ")[1], input_folder2.strip(" - ")[1],
                input_folder3.strip(" - ")[1],]
    sys_load = [20, 50, 90]
    
    # file_folder = r'C:\Zheng Data\TU Delft\Thesis\Thesis Work\GitHub\queue_sim\Output\test'
    file_folder = os.getcwd().strip("queue_sim") + "Output"    
    # "trace_APP50_end.csv"
    file_to_plot1 = file_folder + '\\' + input_folder1 + '\\' + input_file 
    file_to_plot2 = file_folder + '\\' + input_folder2 + '\\' + input_file
    file_to_plot3 = file_folder + '\\' + input_folder3 + '\\' + input_file
    
    output_folder = os.getcwd().strip("queue_sim") + '\\Plots\\'  
    
    data1 = pd.read_csv(file_to_plot1, encoding='utf-8')
    data2 = pd.read_csv(file_to_plot2, encoding='utf-8')
    data3 = pd.read_csv(file_to_plot3, encoding='utf-8')
    
    np_data = data1.to_numpy()  
    plot_data1 = np_data.flatten()
    
    np_data = data2.to_numpy()  
    plot_data2 = np_data.flatten()

    np_data = data3.to_numpy()  
    plot_data3 = np_data.flatten()

    
    tti_duration = 0.00025
    fps = 30
    key_int = 10
    GoP_duration = key_int * (1/30)
    total_duration = np.ceil(plot_data1[-1])
    total_ttis = int((total_duration / tti_duration) / 10)
    GoP_ttis = int(GoP_duration / tti_duration)
    
    packets_per_tti1 = np.zeros((total_ttis))
    packets_per_tti2 = np.zeros((total_ttis))
    packets_per_tti3 = np.zeros((total_ttis))
    
    # Create list with how many packets arrive per TTI
    for tti in range(total_ttis): 
        # if tti == total_ttis - 1: print(tti)
        start_time = tti * tti_duration
        end_time = start_time + tti_duration
        
        packet_list1 = np.where((plot_data1 >= start_time) & (plot_data1 < end_time))        
        packets_per_tti1[tti] = np.size(packet_list1)
        
        packet_list2 = np.where((plot_data2 >= start_time) & (plot_data2 < end_time))        
        packets_per_tti2[tti] = np.size(packet_list2)
        
        packet_list3 = np.where((plot_data3 >= start_time) & (plot_data3 < end_time))        
        packets_per_tti3[tti] = np.size(packet_list3)
        # if np.size(packet_list) == 0:
        #     packets_per_tti[tti] = float('nan')           
        # else:
        #    packets_per_tti[tti] = np.size(packet_list)          
        
    # print("TTIs: ", np.size(np.where(packets_per_tti > 0)))

    """
    # Queues: 5 - 10 - 15 - diff load
    mean_queues_20 = [0.138, 0.194, 0.419]
    mean_queues_50 = [0.138, 0.194, 0.419]
    mean_queues_90 = [0.138, 0.194, 0.419]
        
    mean_queues_20_I = [0.138, 0.194, 0.419]
    mean_queues_50_I = [0.138, 0.194, 0.419]
    mean_queues_90_I = [0.138, 0.194, 0.419]
    
    mean_queues_20_P = [0.138, 0.194, 0.419]
    mean_queues_50_P = [0.138, 0.194, 0.419]
    mean_queues_90_P = [0.138, 0.194, 0.419]
        
    sd_queues_20 = [0.138, 0.194, 0.419]
    sd_queues_50 = [0.138, 0.194, 0.419]
    sd_queues_90 = [0.138, 0.194, 0.419]
        
    sd_queues_20_I = [0.138, 0.194, 0.419]
    sd_queues_50_I = [0.138, 0.194, 0.419]
    sd_queues_90_I = [0.138, 0.194, 0.419]
    
    sd_queues_20_P = [0.138, 0.194, 0.419]
    sd_queues_50_P = [0.138, 0.194, 0.419]
    sd_queues_90_P = [0.138, 0.194, 0.419]
    
    # System Load: 20% - 50% - 90% 
    mean_load_5 = [2.165, 0.395, 0.077]
    mean_load_10 = [2.165, 0.395, 0.077]
    mean_load_15 = [2.165, 0.395, 0.077]

    mean_load_5_I = [2.165, 0.395, 0.077]
    mean_load_10_I = [2.165, 0.395, 0.077]
    mean_load_15_I = [2.165, 0.395, 0.077]

    mean_load_5_P = [2.165, 0.395, 0.077]
    mean_load_10_P = [2.165, 0.395, 0.077]
    mean_load_15_P = [2.165, 0.395, 0.077]
        
    sd_load_5 = [2.165, 0.395, 0.077]
    sd_load_10 = [2.165, 0.395, 0.077]
    sd_load_15 = [2.165, 0.395, 0.077]

    sd_load_5_I = [2.165, 0.395, 0.077]
    sd_load_10_I = [2.165, 0.395, 0.077]
    sd_load_15_I = [2.165, 0.395, 0.077]

    sd_load_5_P = [2.165, 0.395, 0.077]
    sd_load_10_P = [2.165, 0.395, 0.077]
    sd_load_15_P = [2.165, 0.395, 0.077]
    """
    
    # raise SystemExit
    x_axis = np.arange(0,int(total_ttis))
    x_axis_GoP = np.arange(0,GoP_ttis)
    
    x_axis_queues = ['5', '10', '15']
    x_axis_load = ['20%', '50%', '90%']
    
    # plot = False
    if plot:
#############################
#############################        
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
        plt.ylim(0, np.nanmax(np.concatenate([packets_per_tti1, 
                                              packets_per_tti2,
                                              packets_per_tti3])) + 5)  
        
        # plt.scatter(x_axis, packets_per_tti1[0:int(total_ttis)], marker='s', 
        #             s=100, label='20% Load') # 0:int(total_ttis/2)
        plt.plot(x_axis, packets_per_tti1[0:int(total_ttis)], # '-s',  
                  label='20% Load')
        
        # plt.scatter(x_axis, packets_per_tti2[0:int(total_ttis)], marker='v', 
        #             s=100, label='50% Load')
        plt.plot(x_axis, packets_per_tti2[0:int(total_ttis)], # '-*', 
                 label='50% Load')
        
        # plt.scatter(x_axis, packets_per_tti3[0:int(total_ttis)], marker='o', 
        #             s=100, label='90% Load')
        plt.plot(x_axis, packets_per_tti3[0:int(total_ttis)], # '-o', 
                 label='90% Load')
        plt.legend(prop={'size': 25})
        # plt.show()
        print(input_folder3.split('VR - ')[1])
        
        if(save):
            save_path = output_folder + "New" + "\\"
            save_name = f'Packets per TTI - Queues_{input_folder3.split("VR - ")[1]}_cut2.png'
            plt.savefig(save_path + save_name, dpi=300, bbox_inches='tight')
            print(f'Figure saved: {save_name}')
                
        return # raise SystemExit
        
        """
        # TODO: Make plots with multiple inputs - show multiple queues together
#############################
#############################        
        plt.figure(figsize=(40,20))
        plt.title("Mean Interpacket-time for different loads " + "\n" +
                     f"\n Serving Bitrate: {serving_bitrate/1000000}Mbps" +  
                     "\n Load: 50%",                     
                     fontsize=20, fontweight='bold') 
        
        # plt.subplot(121)
        # plt.title('Mean - time spend in total by packets of each frame', fontsize=20)
        plt.xlabel('Loads in %', fontsize=14)
        plt.ylabel('time in ms', fontsize=14)
        plt.ylim(0, np.nanmax(packets_per_tti1, packets_per_tti2) + 10)     
        # plt.scatter(x_axis, packets_per_tti)
        plt.plot(x_axis, packets_per_tti, '-o')
        plt.show()
        if(save):
            save_path = r'C:\Zheng Data\TU Delft\Thesis\Thesis Work\GitHub' + \
                '\SXRSIMv3\Zheng - Plots\Queue Sim' + "\\"
            save_name = f'Queue Sim - .png'
            plt.savefig(save_path + save_name, dpi=300, bbox_inches='tight')
            print(f'Figure saved: {save_name}')
        # raise SystemExit()
#############################  
#############################      
        plt.figure(figsize=(40,20))
        plt.title("Mean Interpacket-time for different loads " + "\n" +
                     f"\n Serving Bitrate: {serving_bitrate/1000000}Mbps" +  
                     f"\n Number of queues: {n_queues}",                     
                     fontsize=20, fontweight='bold') 
        
        # plt.subplot(121)
        # plt.title('Mean - time spend in total by packets of each frame', fontsize=20)
        plt.xlabel('Loads in %', fontsize=14)
        plt.ylabel('time in ms', fontsize=14)
        plt.ylim(0, np.nanmax(packets_per_tti) + 10)     
        # plt.scatter(x_axis, packets_per_tti)
        plt.plot(x_axis, packets_per_tti, '-o')
        plt.show()
        if(save):
            save_path = r'C:\Zheng Data\TU Delft\Thesis\Thesis Work\GitHub' + \
                '\SXRSIMv3\Zheng - Plots\Queue Sim' + "\\"
            save_name = f'Queue Sim - .png'
            plt.savefig(save_path + save_name, dpi=300, bbox_inches='tight')
            print(f'Figure saved: {save_name}')
                
        # save = False
        """
           
    


input_folder = "\\5Q - 20.0% Load - 1Gbps - 1.0s\\"
input_folder2 = "\\5Q - 50.0% Load - 1Gbps - 1.0s\\"
input_folder3 = "\\5Q - 90.0% Load - 1Gbps - 1.0s\\"

input_folder1 = "5Q - 20.0% Load - 1Gbps - 1.0s - VR - 10.0us"
input_folder2 = "5Q - 50.0% Load - 1Gbps - 1.0s - VR - 10.0us"
input_folder3 = "5Q - 90.0% Load - 1Gbps - 1.0s - VR - 10.0us"

# input_name = "\\10Q - 3.0s - 1Gbps - 20.0% Load\\"
input_file = "trace_APP50_end.csv"

plot_dispersion(input_folder1, input_folder2, input_folder3, input_file, 
                plot = True, save = not True)    


