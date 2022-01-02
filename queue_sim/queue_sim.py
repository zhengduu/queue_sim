# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 21:40:30 2021

@author: duzhe
"""

import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import time 
import os
import sys


"""
TODOs:
    - Correct choice/tuning of parameters, especially for random variables 
    
    - Performance optimization (10s, 5 queues => 120 seconds)
    - Implementation: 
          - "BUG" - IF PACKETS HAVE THE SAME ARRIVAL TIME, THE ORDER IS THE REVERSE 
              OF THE INDICES 
              - Default interpacket time? -> Very small (1 us delta)
              - (Second sort command to sort by packet index!)
    - "Last event time" not needed - Last departure time for each queue enough!
    - Nr. of queues: Do research on realistic number of hops -> traceroute
    - Print: Packets per TTI (visualize dispersion)
    - Hyper-exponential distribution for packet sizes
    
"""
pd.options.mode.chained_assignment = None  # default='warn'

class Sim_Par:
    def __init__(self, use_pcap, fps=0, GoP=0, bitrate=0, IP_ratio=0, 
                 packet_size=0):
        
        # Use simulation parameters, when pcap trace is not used
        self.use_pcap = use_pcap # True vs False
        self.fps = int(fps) # Integer
        self.GoP = int(GoP)
        self.bitrate = bitrate # In bps (or Mbps?) 
        self.IP_ratio = IP_ratio # Size of P-frame / Size of I-frame
        self.packet_size = packet_size # in Byte


class Event: 
    def __init__(self, event_time, action, queue, packet):
        
        # Time at which an event occurs
        self.time = event_time
        # Type of action: Packet arrival vs departure  
        self.action = action
        # Location of event (queue i)  
        self.queue = queue
        # Packet involved in the event
        self.packet = packet
        self.packet_type = packet.packet_type


class Packet:
    def __init__(self, packet_id, packet_size, queue, # arr_time, dep_time, 
                 packet_type):
        
        # ID of packet - null/-1 if background packet, otherwise VR idx 
        self.id = packet_id
        # Size of packet
        self.size = packet_size
        # Current location / queue of packet
        self.queue = queue
        # Time of entry into whole queueing system
        # self.arrival = arr_time
        # Time of departure from last queue (and into BS buffer)
        # self.departure = dep_time
        # Background traffic vs. VR packet - 'BG' & 'VR'
        self.packet_type = packet_type 
        
        
def initialise_event_calendar(vr_timestamps, vr_sizes, queues, sys_load, 
                              start_time, end_time, bg_traffic_type, debug): 
    """
    

    Parameters
    ----------
    vr_timestamps : TYPE
        DESCRIPTION.
    vr_sizes : TYPE
        DESCRIPTION.
    queues : TYPE
        DESCRIPTION.
    sys_load : TYPE
        DESCRIPTION.
    start_time : TYPE
        DESCRIPTION.
    end_time : TYPE
        DESCRIPTION.
    bg_traffic_type : TYPE
        DESCRIPTION.
    debug : TYPE
        DESCRIPTION.

    Raises
    ------
    SystemExit
        DESCRIPTION.

    Returns
    -------
    event_calendar : TYPE
        DESCRIPTION.
    event_times_lst : TYPE
        DESCRIPTION.
    vr_packet_counter : TYPE
        DESCRIPTION.
    all_bg_sizes : TYPE
        DESCRIPTION.

    """
    print("\nStarting Initialization...")

    # Initialize event calendar to track all packets etc.
    event_calendar = []
    # Numpy array to track event times for faster simulation
    event_times_lst = [] # np.empty(0)
    
    all_bg_sizes = []
    
    # Trimodal packet size distribution
    packet_sizes = np.array([44, 576, 1500]) # In Byte
    packet_prob = np.array([0.44, 0.19, 0.37]) 
        
    # Calculate exponential inter-packet arrival time through packet 
    # distribution and desired system load 
    bg_throughput = sys_load * 1e9 # In Gbps
    avg_packet_size = round(np.sum(packet_sizes * packet_prob)) * 8 # in bit
    nr_packets_per_s = int(bg_throughput / avg_packet_size)
    exp_time = round((1 / nr_packets_per_s), 9)
    
    # print(exp_time, nr_packets_per_s, bg_throughput)
      
    # Generate all background packet arrivals in each queue   
    bg_sizes = [] 
    total_bg_sizes = 0
    if bg_traffic_type == "BG":         
        for q in range(queues):
            
            random_seed = q
            
            curr_time = start_time
            bg_count = 0
            # For debugging
            bg_times = []
            
            # Give every queue a different, but known seed
            # if q == 1:
            #     np.random.seed(0) # Seed 1 gives 100000+ background packets!!!
                
            # else: 
            np.random.seed(random_seed+5)
                
            while curr_time < end_time: 
                
                if debug[0]:
                                    
                    if bg_count > debug[1]:
                        break
                
                """                     
                # Hyper-exponentially distributed packet size
                # Flip biased coin
                # if ut.success_coin_flip(exp_size[0]):
                    # First exponential distribution       
                    # if debug[0]:    
                    #     print(f"Queue: {q} - Distribution {exp_size[1]} ")
                #     new_size = int(np.random.exponential(exp_size[1]))                    
                    
                # else: 
                    # Second exponential distribution
                    # if debug[0]:    
                    #     print(f"Queue: {q} - Distribution {exp_size[2]} ")
                #     new_size = int(np.random.exponential(exp_size[2]))                    
                """
                            
                # Sample from trimodal packet size distribution
                new_size = np.random.choice(packet_sizes, p=packet_prob)
                
                # Get exponentially distributed inter-packet arrival times
                inter_arr_time = np.random.exponential(exp_time)   
                # Avoid values smaller than 1 microsecond (or nanosecond?)
                if inter_arr_time <= 0:
                    inter_arr_time = 1e-6
                    
                curr_time += inter_arr_time 
                
                # Create new BG packet and add to event calendar 
                if curr_time < end_time:
                    bg_packet = Packet(packet_id=-1, packet_size=new_size, 
                                       queue=q, packet_type='BG')
                    if debug[0]:
                        event_calendar.append(Event(curr_time, 'packet_arrival', 
                                                    q, bg_packet))
                        bg_sizes.append(new_size)
                        bg_times.append(round(curr_time, 9))
    
                    else:
                        event_calendar.append(Event(curr_time, 'packet_arrival', 
                                                    q, bg_packet))
                        event_times_lst.append(curr_time) 
                        # = np.append(event_times, curr_time)
                        
                bg_count += 1
                total_bg_sizes += new_size
            all_bg_sizes.append(total_bg_sizes)
            
            # if debug[0]:
            # print(f"\nQueue: {q} - BG packets: {bg_count}" + 
            #       f"\n Total: {total_bg_sizes}")
                      # f"\nTimes: {bg_times} - \nSizes: {bg_sizes} ")
    elif bg_traffic_type == "VR": 
        # Take total simulation time 
        # For each queue, split up into sequences based on load
        # Get random intervals, summing up to interframe time
        # Add interval as delta to all packets of one VR timestamp set
        # Append all in event calendar
        
        # total_time = sim_time
        nr_vr_streams = int(sys_load/0.05) # 50Mbps BG streams
        frametime = 1 / 30
        np.random.seed(0)
        stream_delays = np.random.uniform(0, frametime, nr_vr_streams*queues)
        stream_counter = 0
        
        for q in range(queues):
            for stream in range(nr_vr_streams):
                curr_time = 0.0
                bg_counter = 0
                
                new_timestamps = vr_timestamps + stream_delays[stream_counter]
                stream_counter += 1
                
                total_packets = len(new_timestamps)
                while curr_time < end_time and bg_counter < total_packets:
                    
                    curr_time = new_timestamps[bg_counter]
                    new_size = vr_sizes[bg_counter]
                    
                    if curr_time < end_time:
                        bg_packet = Packet(packet_id=-1, packet_size=new_size, 
                                           queue=q, packet_type='BG')                                                   
                        event_calendar.append(Event(curr_time, 'packet_arrival', 
                                                    q, bg_packet))
                        # Copy new timestamp to list as well
                        event_times_lst.append(curr_time)
                        
                    bg_counter += 1    
                    total_bg_sizes += new_size
                all_bg_sizes.append(total_bg_sizes)
        
    else: 
        print("Please choose one of the following as background traffic:" +
              "\n'BG' - 'VR'")
        raise SystemExit
            
    # print("all BG sizes", all_bg_sizes)
        
    # Generate all packet arrivals for VR packets at the first queue
    # dont need to check time, just go by timestamps one by one
    curr_time = 0 # only to be double sure 
    vr_packet_counter = 0
    total_packets = len(vr_timestamps)
    
    while curr_time < end_time and vr_packet_counter < total_packets:
        
        curr_time = vr_timestamps[vr_packet_counter]
        new_size = vr_sizes[vr_packet_counter]
        
        if curr_time < end_time:
            
            vr_packet = Packet(packet_id=vr_packet_counter, packet_size=new_size,
                               queue=0, packet_type = 'VR')                                
            event_calendar.append(Event(curr_time, 'packet_arrival', 0, 
                                        vr_packet))
            # Copy new timestamp to list as well
            event_times_lst.append(curr_time) #np.append(event_times, curr_time)
            
        vr_packet_counter += 1    
        
    print("VR packets:", vr_packet_counter)
    
    # print("VR timestamps: ", vr_timestamps)
    
    # raise SystemExit
        
    return event_calendar, event_times_lst, vr_packet_counter, all_bg_sizes



def calc_dispersion(timestamps_arr, timestamps_dep, timestamps_end, frames, 
                    frametypes, sizes, packets_in_each_frame, queue_par, 
                    bg_packets, txt_log_file):
  
    
    # Queue variables for figure labelling 
    n_queues  = queue_par[0]     
    serving_bitrate = queue_par[1]
    sys_load  = queue_par[2]
    start_time  = queue_par[3]   
    end_time  = queue_par[4]
    bg_traffic_type = queue_par[5]
    
    # Time spend in each queue by each packet dim (nr_packets x nr_queues)
    timestamps_diff = timestamps_dep - timestamps_arr
        
    timestamps_end_diff = timestamps_end - timestamps_arr[0]  
        
    packets_in_frame = np.array(packets_in_each_frame)
        
    nr_frames = frames[-1] + 1
    nr_I_frames = int(np.floor(nr_frames / 10))
    nr_P_frames = int(nr_frames - nr_I_frames)
    
    # Mean                                
    mean_per_frame = np.zeros((n_queues,len(packets_in_frame)))
    mean_per_queue = np.zeros(n_queues)
    mean_all_queues = np.zeros(len(packets_in_frame))
    mean_total = np.mean(timestamps_end)
    
    # Variance 
    var_per_frame = np.zeros((n_queues,len(packets_in_frame)))
    var_per_queue = np.zeros(n_queues)
    var_all_queues = np.zeros(len(packets_in_frame))
    var_total = np.var(timestamps_end)
    
    # Standard deviation
    sd_per_frame = 0
    sd_per_queue = 0
    sd_all_queues = 0
    sd_total = np.sqrt(var_total)
    
    sizes_I = []
    sizes_P = []
        
    # Calculate Dispersion -> Interpacket time metrics
    timestamps_per_I_frame = []
    timestamps_per_P_frame = []

    for frame in range(nr_frames):
        if frame % 10 == 0:
            timestamps_per_I_frame.append(timestamps_end[
                packets_in_frame[frame][0]:packets_in_frame[frame][1] + 1])            
            sizes_I.extend(sizes[
                packets_in_frame[frame][0]:packets_in_frame[frame][1] + 1])
            
        else: 
            timestamps_per_P_frame.append(timestamps_end[
                packets_in_frame[frame][0]:packets_in_frame[frame][1] + 1])
            sizes_P.extend(sizes[
                packets_in_frame[frame][0]:packets_in_frame[frame][1] + 1])
    
    interpacket_time_I = []
    interpacket_time_P = []
    interpacket_time_all = []
    
    # # Calculate for all frames: 
    # for packet in range(len(timestamps_end) - 1):
    #     interpacket_time_all.append(timestamps_end[packet + 1] - \
    #                                     timestamps_end[packet])
        
    # Calculate for I-frames
    for frame in range(len(timestamps_per_I_frame)):
        if type(timestamps_per_I_frame[frame]) == np.ndarray:
            for packet in range(len(timestamps_per_I_frame[frame]) - 1):
                interpacket_time_I.append(
                    timestamps_per_I_frame[frame][packet + 1] - \
                        timestamps_per_I_frame[frame][packet])
                                        
    # Calculate for P-frames
    for frame in range(len(timestamps_per_P_frame)):   
        if type(timestamps_per_P_frame[frame]) == np.ndarray:
            for packet in range(len((timestamps_per_P_frame[frame])) - 1):
                interpacket_time_P.append(
                    timestamps_per_P_frame[frame][packet + 1] - \
                        timestamps_per_P_frame[frame][packet])
                    
    interpacket_time_all.extend(interpacket_time_I + interpacket_time_P)
    
    mean_time_I = np.mean(interpacket_time_I)
    mean_time_P = np.mean(interpacket_time_P)
    mean_time_all = np.mean(interpacket_time_all)
    
    
    var_time_I = np.var(interpacket_time_I)
    var_time_P = np.var(interpacket_time_P)
    var_time_all = np.var(interpacket_time_all)
    
    sd_time_I = np.sqrt(var_time_I)
    sd_time_P = np.sqrt(var_time_P)
    sd_time_all = np.sqrt(var_time_all)
    
    mean_sizes_I = round(np.mean(sizes_I), 2)
    mean_sizes_P = round(np.mean(sizes_P), 2)           
    
    print("\nSimulation Parameters:",
         f"\n Simulation start: {start_time} s" +
         f"\n Simulation end: {end_time} s" +
         f"\n Number of queues: {n_queues}" + 
         f"\n Background traffic type: {bg_traffic_type}" + 
         f"\n Serving Bitrate: {serving_bitrate/1000000}Mbps" +  
          f"\n System Load: {sys_load*100}%")
    
    with open(txt_log_file, "a") as log_file:
        print("\nSimulation Parameters:",
              f"\n Simulation start: {start_time} s" +
              f"\n Simulation end: {end_time} s" +
              f"\n Number of queues: {n_queues}" + 
              f"\n Background traffic type: {bg_traffic_type}" + 
              f"\n Serving Bitrate: {serving_bitrate/1000000}Mbps" +  
              f"\n System Load: {sys_load*100}%", file=log_file)
        
        print("\nMean Packet Sizes:" + 
              f"\n I-frames: {mean_sizes_I} Byte"
              f"\n P-frames: {mean_sizes_P} Byte"
              "\n\nMean inter-packet time:" +
              f"\n All frames: {round(mean_time_all * 1000, 6)} ms" + 
              f"\n I-frames:   {round(mean_time_I * 1000, 6)} ms" + 
              f"\n P-frames:   {round(mean_time_P * 1000, 6)} ms" +           
              "\n\nStandard Deviation: " +
              f"\n All frames: {round(sd_time_all * 1000, 6)} ms" + 
              f"\n I-frames:   {round(sd_time_I * 1000, 6)} ms" + 
              f"\n P-frames:   {round(sd_time_P * 1000, 6)} ms",
              file=log_file)  
              # "\n\nVariance: " +
              # f"\n All frames: {round(var_time_all * 1000, 6)} ms" + 
              # f"\n I-frames:   {round(var_time_I * 1000, 6)} ms" + 
              # f"\n P-frames:   {round(var_time_P * 1000, 6)} ms")
        log_file.close()  
    
    
    return


def main(input_args, serving_bitrate, sim_par, debug):

    """Simulate dispersion of VR packets going through network hops before 
    arriving at radio BS
    

    Parameters
    ----------
    input_args : str
        Simulation parameters taken from console input.
    serving_bitrate : int
        Serving bitrate of simulated network routers.
    sim_par : Object
        Simulation parameters if real VR sequence is not used.
    debug : List
        Enter debug mode with debug parameters.

    Raises
    ------
    SystemExit
        DESCRIPTION.

    Returns
    -------
    None.
     
    """
    tic = time.perf_counter()      
    
    vr_timestamps = []
    vr_sizes = []
    
    vr_file_name = input_args.trace # "trace_0.csv"
    n_queues = input_args.queues
    sys_load = input_args.load
    start_time = input_args.start_time
    sim_time = input_args.sim_time
    end_time = start_time + sim_time
    bg_traffic_type = input_args.bg
    q_latency = 10e-6 # 10 us, based on paper 
    # (Essentially in a sense similar to just adding more load)
    # Alternative: 500 ns (based on newer stat sheet)
    
    try:
        debug = input_args.debug
    except: 
        print("Starting simulation - no debug mode")
    
    # Folder with packet traces 
    # file_folder = r"C:\Zheng Data\TU Delft\Thesis\Thesis Work\GitHub\SXRSIMv3\PCAP\Trace"
    file_folder = os.getcwd() # + '\\queue_sim'
    file_to_simulate = file_folder + '\\VR_traces\\' + vr_file_name    
    
    # Create output save folder
    save_folder_name = f'{n_queues}Q - {sys_load*100}% Load - ' + \
                       f'{round(start_time, 2)}-{round(end_time, 2)}s - ' + \
                       f'{bg_traffic_type} - ' + \
                       f'{round(q_latency * 1e6, 2)}us'                           
                       # f'{int(serving_bitrate/(1e9))}Gbps - ' + \
    output_save_path = file_folder + "\\Output\\" + save_folder_name    
    os.makedirs(output_save_path, exist_ok=True)
    
    # Txt file with print logs
    log_name = save_folder_name + f' - {vr_file_name.strip(".csv")} - log.txt' 
    txt_log_file = output_save_path + "\\" + log_name
        
    print("\nSimulation Parameters:",
          f"\n Simulation start: {start_time} s" +
          f"\n Simulation end: {end_time} s" +
          f"\n Number of queues: {n_queues}" + 
          f"\n Background traffic type: {bg_traffic_type}" + 
          f"\n Serving Bitrate: {serving_bitrate/1000000}Mbps" +  
          f"\n System Load: {sys_load*100}%")
    
    with open(txt_log_file, "w") as log_file:        
        print("\nSimulation Parameters:",
              f"\n Simulation start: {start_time} s" +
              f"\n Simulation end: {end_time} s" +
              f"\n Number of queues: {n_queues}" + 
              f"\n Background traffic type: {bg_traffic_type}" + 
              f"\n Serving Bitrate: {serving_bitrate/1000000}Mbps" +  
              f"\n System Load: {sys_load*100}%", file=log_file)
        log_file.close()
        
    already_adjusted = True    
    # Use real video data   
    if sim_par.use_pcap == True: 
                
        # Load into dataframe                   
        # Use traces with already adjusted burstiness etc. 
        sim_data = pd.read_csv(file_to_simulate, encoding='utf-16-LE')    
        
        fps = 30 
        # int(np.ceil(sim_data["frame"].iloc[-1] / sim_data["time"].iloc[-1]))
        frame_time = 1 / fps
        
        if not already_adjusted: 
            # Adjust timestamps to avoid sorting problems!  
            # Packet timestamp count starts at zero 
            # Set all packets belonging to same frame to frame generation time 
                        
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
            
            # Calculate for every frame the number of packets
            nr_packets_in_frame = list(range(len(packets_in_frame)))
            # Calculate desired dispersion over nr_packets oer frame
            interpacket_time_frame = list(range(len(packets_in_frame)))
            # Average interpacket time over all frames
            
            # Desired burstiness
            burstiness = 0.6 # Set by asking Rick
            dispersion_per_frame = frame_time * (1 - burstiness)
    
            for frame in range(len(packets_in_frame)):
                nr_packets_in_frame[frame] = packets_in_frame[frame][1] - \
                                             packets_in_frame[frame][0] + 1 
                interpacket_time_frame[frame] =  dispersion_per_frame / \
                                                 nr_packets_in_frame[frame]
    
            max_inter_packet_time = frame_time / max(nr_packets_in_frame)
    
            interpacket_time = sum(interpacket_time_frame) / \
                               len(interpacket_time_frame)
    
            if interpacket_time >= max_inter_packet_time:
                interpacket_time = max_inter_packet_time
    
            for frame in range(len(packets_in_frame)):
                # Add calculated interpacket time to packets 
                packet_counter = 0            
                for packet in range(packets_in_frame[frame][0], 
                                    packets_in_frame[frame][1] + 1):
                    sim_data['time'][packet] += packet_counter * interpacket_time
                    packet_counter += 1
                    
        # Cut from start until desired total simulation time from console input
        sim_data = sim_data[((sim_data['frame']/fps) >= start_time)]            
        sim_data = sim_data[((sim_data['frame']/fps) < end_time)]     
        
        # Copy timestamps 
        vr_timestamps = sim_data['time'].values.copy()
        vr_sizes = sim_data['size'].values.copy() 
        # For calculations later
        vr_frames = sim_data['frame'].values.copy()
        vr_frametypes = sim_data['frametype'].values.copy()         
        
        if debug[0]:
            test_number = debug[1]
            
            vr_timestamps = sim_data['time'][0:test_number].values.copy() 
            print("vr_timestamps", vr_timestamps)
            
            vr_sizes = sim_data['size'][0:test_number].values.copy()  
            print("vr_sizes", vr_sizes)
    
    # Generate own packet stream from simulation parameters 
    else: 
        pass
        # TODO
        # FPS = sim_par.fps
        # GoP = sim_par.GoP
        # n_periods = round(sim_time * FPS / GoP)
        # bitrate = sim_par.bitrate # In bps (or Mbps?) 
        # IP_ratio = sim_par.IP_ratio # Size of P-frame / Size of I-frame
        # packet_size = sim_par.packet_size # in Byte
        
        # Consider using generated frame sequence directly as input???
        # Only Timestamps and Sizes needed for one GoP, then cycle for sim_time
    
    
    # Add propagation delay depending on distance between Src and Dst
    if n_queues == 5: 
        # In NL - 0.3 ms
        prop_delay = (0.3 / n_queues) * 0.001 # convert ms to s 
    elif n_queues == 10:
        # In West EU - 3ms 
        prop_delay = (3 / n_queues) * 0.001
    elif n_queues == 15:
        # Intercontinental - 30ms 
        prop_delay = (30 / n_queues) * 0.001
    else: 
        print("Please choose a number of queues on this list: [5 - 10 - 15]")
        raise SystemExit
    

    toc = time.perf_counter()
    print(f"\nInitializing Video File: {toc-tic:0.4f} seconds")
    with open(txt_log_file, "a") as log_file:            
        print(f"\nInitializing Video File: {toc-tic:0.4f} seconds", 
              file=log_file)
        log_file.close()
        
    tic = time.perf_counter()    
    event_calendar, event_times_lst, total_vr_packets, bg_packets = \
        initialise_event_calendar(vr_timestamps, vr_sizes, n_queues, sys_load, 
                                  start_time, end_time, bg_traffic_type, debug)
        
    event_times = np.array(event_times_lst)
    
    toc = time.perf_counter()        
    
    print(f"Initializing Event Calendar: {toc-tic:0.4f} seconds")
    print(f"Total start events: {len(event_calendar)}")        
    print("Starting Event Simulation...")
    
    with open(txt_log_file, "a") as log_file:            
        print(f"Initializing Event Calendar: {toc-tic:0.4f} seconds", file=log_file)
        print(f"Total start events: {len(event_calendar)}", file=log_file)        
        log_file.close()

########## Start of event simulation ##########

    tic = time.perf_counter()    
    curr_time = 0.000
    
    # For performance and debugging
    counter = 0
    vr_packet_counter = 0
    
    # To save arrival and departure timestamps for each packet and queue 
    vr_timestamps_dep = np.zeros((n_queues, total_vr_packets))
    vr_timestamps_arr = np.zeros((n_queues, total_vr_packets))
    vr_timestamps_end = np.zeros(total_vr_packets)    
    
    # last_event_time = np.zeros(n_queues)
    last_departure_time = np.zeros(n_queues)

    start = time.time()       
    
    # test = "list"
    # print("--", test)
    
    # event_calendar.sort(key = operator.attrgetter('time'), reverse = True)
    while event_calendar != []: 
        tic_tic = time.perf_counter()
        
        # Check np.array for the index of the smallest timestamp, this is the 
        # next event to be processed                 
        # Pop appropriate event and delete np.array entry
        # if test == "numpy":
        next_event_idx = np.argmin(event_times)
        event_times = np.delete(event_times, next_event_idx)        
                    
        # # Try the same with python list and see which is faster
        # if test == "list":
        #     next_event_idx = event_times_lst.index(min(event_times_lst))
        #     del event_times_lst[next_event_idx]
        
        next_event = event_calendar.pop(next_event_idx)
        
        # else:     
        #     next_event = event_calendar.pop()
        
        # event_calendar.sort(key = operator.attrgetter('time'), reverse = True)
        # next_event = event_calendar.pop(event_calendar.index(min(
        # #                  event_calendar, key = operator.attrgetter('time'))))
        
        # toc_toc = time.perf_counter()               
        if counter % 100000 == 0:
            now = time.time()
            print(f"\nSimulation Duration: {now-start:0.5f} s" + # 
                  f"\nIterations: {counter} - Time: {round(curr_time, 3)}" + 
                  f"\nEvent Calendar: {len(event_calendar)} Entries left")
                  
        with open(txt_log_file, "a") as log_file:     
            if counter % 100000 == 0:
                now = time.time()
                print(f"\nSimulation Duration: {now-start:0.5f} s" + # 
                      f"\nIterations: {counter} - Time: {round(curr_time, 3)}" + 
                      f"\nEvent Calendar: {len(event_calendar)} Entries left",
                      file=log_file)                
            log_file.close()

        # tic_tic = time.perf_counter()

        # Keep track of location of events for proper timestamping
        curr_queue = next_event.queue        
        # Keep track of current simulation time
        curr_time = next_event.time
        
        if debug[2]:
            print(f"Current time: {curr_time} - Queue: {next_event.queue}")
            print(f"Event: {next_event.packet.packet_type} - {next_event.action}" + \
                  f" - {next_event.packet.size} ")
        
        # Simulate arrival of packet in queue
        if next_event.action == 'packet_arrival':
            
            # Save arrival time at queue if VR packet
            # if next_event.packet_type == 'VR': 
            #     vr_timestamps_arr[next_event.queue][next_event.packet.id] = \
            #         round(next_event.time, 9)              
                            
            # Calculate departure time for packet
            serving_time = next_event.packet.size / serving_bitrate      
            
            if curr_time >= last_departure_time[curr_queue]:     
                new_departure_time = q_latency + curr_time + serving_time 
            else: 
                new_departure_time = q_latency + serving_time + last_departure_time[
                                                        curr_queue] 
                
            # Update last departure time for respective queue
            last_departure_time[curr_queue] = new_departure_time
            
            if debug[2]:
                print("New departure time:", new_departure_time, 
                      next_event.packet_type)      
                           
            # Create new event for packet departure into new queue
            event_calendar.append(Event(new_departure_time, 'packet_departure', 
                                        next_event.queue, next_event.packet))            
            # Append new time for event also to np.array            
            # if test == "numpy":
            event_times = np.append(event_times, new_departure_time)
            # if test == "list":
                # event_times_lst.append(new_departure_time)
                        
        elif next_event.action == 'packet_departure': 
                        
            if debug[2]:
                print(next_event.packet.packet_type)
                
            # VR packets are send to next queue or if at last queue, to the BS
            if next_event.packet.packet_type == 'VR':            
                
                # last_event_time[curr_queue] += curr_time
                # print("Departure - new last event time:", last_event_time)
                next_queue = next_event.queue + 1
                
                # Departure from last queue - send to BS
                if next_queue >= n_queues:
                    # Save time for correct packet ID
                    new_time = curr_time + prop_delay
                    vr_timestamps_end[next_event.packet.id] = round(new_time,9)   
                    vr_timestamps_dep[next_queue-1][next_event.packet.id] = \
                        round(curr_time, 9)
                    vr_packet_counter += 1
                    
                # Otherwise send to next queue in simulation 
                else: 
                    # Create new arrival event with new parameters
                    # Departure time is new arrival time 
                    # Add some propagation delay/ping based on traceroutes
                    new_arr_time = curr_time + prop_delay
                    packet = next_event.packet
                    vr_timestamps_dep[next_queue - 1][packet.id] = round(
                                                                  curr_time, 9)                    
                    event_calendar.append(Event(new_arr_time, 'packet_arrival', 
                                                next_queue, packet))          
                    # Append new time for event also to np.array        
                    # if test == "numpy":
                    event_times = np.append(event_times, new_arr_time)
                    
                    # if test == "list":
                    #     event_times_lst.append(new_arr_time)
        counter += 1
        
        toc_toc = time.perf_counter()
        
        if counter % 100000 == 0:
            print(f"Event Handling Duration: {toc_toc-tic_tic:0.5f} s")
        with open(txt_log_file, "a") as log_file:        
            if counter % 100000 == 0:
                print(f"Event Handling Duration: {toc_toc-tic_tic:0.5f} s", 
                      file=log_file)
            log_file.close()
            
                    
    toc = time.perf_counter()    
    print(f"\nFinished Simulation: {counter} Events - {toc-tic:0.4f} seconds")

    with open(txt_log_file, "a") as log_file:        
        print(f"\nFinished Simulation: {counter} Events - {toc-tic:0.4f} seconds",
              file=log_file)
        log_file.close()

    # print("Final VR packets:", vr_packet_counter)
       
    output_file_name = f'{vr_file_name.strip(".csv")}_end.csv'
    # output_file_name_arr = f'{vr_file_name.strip(".csv")}_arr.csv'
    # output_file_name_dep = f'{vr_file_name.strip(".csv")}_dep.csv'

    full_file_name = os.path.join(output_save_path, output_file_name)
    # full_file_name_arr = os.path.join(output_save_path, output_file_name_arr)
    # full_file_name_dep = os.path.join(output_save_path, output_file_name_dep)
    
    np.savetxt(full_file_name, vr_timestamps_end, delimiter=",")# , fmt='%s')
    # np.savetxt(full_file_name_arr, vr_timestamps_arr, delimiter=",")#     
    # np.savetxt(full_file_name_dep, vr_timestamps_dep, delimiter=",")#     
    
    print("Saving output files...")

    dispersion = True
    if dispersion:
        
        queue_par = [n_queues, serving_bitrate, sys_load, start_time, 
                     end_time, bg_traffic_type]
        
        print("Calculating Dispersion Metrics...")
        
        calc_dispersion(vr_timestamps_arr, vr_timestamps_dep, vr_timestamps_end,
                        vr_frames, vr_frametypes, vr_sizes, packets_in_frame, 
                        queue_par, bg_packets, txt_log_file)
                
    if debug[0]:
        np.set_printoptions(threshold=sys.maxsize)
        print("Final:", vr_timestamps_end[0:debug[1]])
        
    return 



if __name__ == "__main__":
    
    input_args = 0    
    input_file = "trace_0.csv"    
    sim_par = Sim_Par(use_pcap = True, fps = 30, GoP = 6, bitrate = 10000000, 
                      IP_ratio = 0.2, packet_size = 1500)  
    
    main(input_args, serving_bitrate=1e9, sim_par=sim_par, debug=[False, 5, False])
   
    # Debugging mode - #Packets - Print statements



