# Overview
This is the GitHub repository for my independent project course "JCP410 - Modelling of Biochemical Systems".

The final paper is located in `Final Paper.pdf`



# How Modules Fit Together
`Paper_extension.py` holds our gillespie simulation functions, and is the actual “experiment”. We use main.py to run our experiment and perform our data analysis. We use single_pass() as a harness to tell handle the experiment and gillespie simulation. The function full_gillespie() in main.py is what iterates through our RVF. Autocorrelate performs our data analysis (fast fourier transform). We then return the attributes we have received for that single trial.

`multi.py` is what runs our multithreaded application. Running a single time trace iteratively for all experiments would have taken over 250 hours. Instead, we ran experiments in parallel using 112 CPU cores to bring the time down to only 10 hours. multi.py handled this multithreading and saved our results with h5py. `Data_analysis.py` takes the data imported from the cluster and displays the data in a readable graph. 
