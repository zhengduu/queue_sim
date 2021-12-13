# -*- coding: utf-8 -*-
"""
Created on Sat Dec 11 15:56:22 2021

@author: duzhe
"""

"""Program entrypoint."""

from argparse import ArgumentParser, Namespace

from queue_sim.queue_sim import main, Sim_Par


def _parse_args() -> Namespace:
    parser = ArgumentParser(prog="python3 -m queue_sim",
                            description="Reads and simulates VR packets going through network hops.")

    parser.add_argument('-trace', action='store', type=str, required=True, help="Input PCAP traces")
    parser.add_argument('-queues', action='store', type=int, required=True, help="Number of queues")
    parser.add_argument('-load', action='store', type=float, required=True, help="System load")
    parser.add_argument('-sim_time', action='append', type=float, required=True, help="Total simulation time")
    # parser.add_argument('--debug', action='store', type=bool, required=False, help="Enter debug mode")

    # parser.add_argument('--output', action='store', type=str, required=True, help="Output PCAP file")

    args = parser.parse_args()
    return args


args = _parse_args()

sim_par = Sim_Par(use_pcap = True, fps = 30, GoP = 6, bitrate = 10000000, 
                  IP_ratio = 0.2, packet_size = 1500)    

main(input_args=args, serving_bitrate=1e9, sim_par=sim_par, debug=[False, 5, False])
