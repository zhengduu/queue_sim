# queue_sim
 Simulator for VR traffic going through network queues

# Dependencies
- Python Libraries:
    * numpy
    * matplotlib
    * pandas

# Use following command to run simulation:
In repo folder: 

```bash 
python -m queue_sim -trace trace_to_simulate.csv -queues number_of_queues -load network_load -sim_time start_time -sim_time end_time
```
# Notes:
- All VR traces need to be in "VR_traces" folder
- All output traces will be saved in "Output" folder (in gitignore) with corresponding parameters as labels

# TODOs:
...
