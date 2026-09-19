import mne
import os

print("--- MNE Data Directory Check ---")
try:
    data_path = mne.get_data_dir() 
    print(f"MNE's configured default data directory is: {data_path}")
except Exception as e:
    print(f"Error accessing MNE path: {e}")