# queue_sim
 Simulator for VR traffic going through network queues

# Dependencies
- Python Libraries:
    * numpy
    * matplotlib
    * pandas

# Use following (example) command to run simulation:
In repo folder: 

```bash 
python -m queue_sim -seed 1 -trace trace_APP30_0.6.csv - bg BG -queues 10 -load 0.7 -sim_time 1.0 -start_time 0.0
```
# Notes:
- All VR traces need to be in "VR_traces" folder
- All output traces will be saved in "Output" folder (in gitignore) with corresponding parameters as labels

# TODOs:
...
