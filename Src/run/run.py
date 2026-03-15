#for running the model
# run with:    python -m CAN.run.run
# Authors: Connor Marks
import sys
import os
from BERT.dataset import preprocess_data
from BERT.train import train
from BERT import config



def create_run_directory():
    base_dir = os.path.dirname(__file__) 
    run_id = 1
    while os.path.exists(os.path.join(base_dir, f"run_{run_id}")): # checks for a run_n in the files
        run_id += 1 #gets to the last one

    run_dir = os.path.join(base_dir, f"run_{run_id}") 
    os.makedirs(run_dir) #makes a new run_n file so we can track our runs.
    return run_dir


def create_run_summary(run_dir):
    summary_path = os.path.join(run_dir, "run_summary.txt") #make a summary file for the run
    summary_file = open(summary_path, "w")

    summary_file.write("CONFIGURATION\n")
    for k, v in vars(config).items(): # copy everything from config into it
        if not k.startswith("__"):
            summary_file.write(f"{k}: {v}\n")
    summary_file.write("\n")
    summary_file.flush()
    return summary_file




if __name__ == "__main__":
    run_dir = create_run_directory() # code above, just saving the run.

    summary_file = create_run_summary(run_dir)

    train_loader, val_loader, pos_weights = preprocess_data() # preprocessing

    train(train_loader, val_loader, pos_weights, run_dir, summary_file) # training the model
