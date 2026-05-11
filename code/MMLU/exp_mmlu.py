#!/usr/bin/env python3
# filepath: /data/lisj/Code/Baselines/DyLAN/code/MMLU/exp_mmlu.py

import os
import glob
import subprocess
import concurrent.futures
import multiprocessing

def process_file(file_path):
    """Process a single CSV file."""
    # Extract filename without extension
    filename = os.path.basename(file_path)
    filename_without_ext = os.path.splitext(filename)[0]
    
    # Define paths for results and logs
    output_dir = "mmlu_downsampled_Economist_Doctor_Lawyer_Mathematician_Psychologist_Programmer_Historian"
    result_file = f"{output_dir}/{filename_without_ext}_73.txt"
    log_file = f"{output_dir}/{filename_without_ext}_73.log"
    
    # Check if result file exists and has exactly 4 lines
    if os.path.exists(result_file):
        with open(result_file, 'r') as f:
            if len(f.readlines()) == 4:
                return  # Skip this file
    
    # Run the Python script with appropriate arguments
    cmd = [
        "python", 
        "llmlp_listwise_mmlu.py", 
        file_path, 
        filename_without_ext, 
        MODEL, 
        EXP_NAME, 
        ROLES
    ]

    with open(log_file, 'w') as log:
        subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)

def main():
    global MODEL, EXP_NAME, ROLES
    
    # Define variables
    MODEL = "gpt-3.5-turbo"
    # MODEL = "gpt-4"  # Uncomment to use GPT-4 instead
    
    # Specify your directory - replace with actual path
    directory = "<path-to-your-directory>"
    EXP_NAME = "mmlu_downsampled"
    
    # Define roles
    ROLES = "['Economist', 'Doctor', 'Lawyer', 'Mathematician', 'Psychologist', 'Programmer', 'Historian']"
    
    # Get all CSV files in the directory
    csv_files = glob.glob(os.path.join(directory, "*.csv"))
    
    # Process files in parallel
    max_workers = min(multiprocessing.cpu_count(), len(csv_files))
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        executor.map(process_file, csv_files)
    
    print("All done")
    
    # Run the analysis script
    # subprocess.run(["bash", "anal_imp.sh"])

if __name__ == "__main__":
    main()
